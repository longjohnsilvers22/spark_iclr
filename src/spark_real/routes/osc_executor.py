"""
Subprocess-managed OSC executor backend (robots_realtime integration).

Provides an A/B-testable alternative to SPARK's default franky CartesianMotion
path. When activated, SPARK releases its FCI control session and the
robots_realtime rr-session takes over with the panda-py operational-space
controller (KP_pos=150, KD_pos=30, tuned values, not exposed as knobs).

Wire protocol:
  SPARK -> HTTP POST http://127.0.0.1:9009/move_to_pose
          {"position": [x,y,z], "wxyz": [w,x,y,z],
           "duration_s": float, "gripper_width": float (optional)}
  The bridge agent (FrankaOscCartesianTargetAgent) lives inside the
  rr-session subprocess and exposes the HTTP server. It solves IK with
  pyroki and publishes joint_pos onto the ZMQ bus, which the RobotNode
  forwards to the panda-py OSC controller at 300 Hz.

Mutually exclusive with:
  - SPARK's franky driver (./api/velocity, /api/execute, calibration moves)
  - viser_teleop_proc (same FCI exclusivity)
"""

from __future__ import annotations

import logging
import os
import signal
import subprocess
import tempfile
import time
import urllib.error
import urllib.request
import json
from pathlib import Path
from typing import Optional

import yaml
from fastapi import APIRouter
from fastapi.responses import JSONResponse

from spark_real.robots.franka.osc_robot_proxy import OSCRobotProxy
from spark_real.routes import state

logger = logging.getLogger(__name__)
router = APIRouter()

RR_ROOT = Path(os.path.expanduser("~/robots_realtime"))
RR_VENV_PY = RR_ROOT / ".venv" / "bin" / "python"
RR_CONFIG = RR_ROOT / "configs" / "franka" / "franka_spark_osc_executor.yaml"
OSC_HTTP_HOST = "127.0.0.1"
OSC_HTTP_PORT = 9009
OSC_HTTP_URL = f"http://{OSC_HTTP_HOST}:{OSC_HTTP_PORT}"

# Generous upper bound; the rr-session has a 3 s SIGTERM force-kill timer.
_STOP_TIMEOUT_S = 6.0

# Maximum seconds to wait for the HTTP bridge to come up after start.
# pyroki JIT warm-up takes ~5-8 s in the agent's __init__.
_HTTP_READY_TIMEOUT_S = 20.0


def _proc_alive() -> bool:
    p = state.osc_executor_proc
    return p is not None and p.poll() is None


def _http_get(path: str, timeout: float = 2.0) -> Optional[dict]:
    try:
        with urllib.request.urlopen(f"{OSC_HTTP_URL}{path}", timeout=timeout) as r:
            return json.loads(r.read().decode())
    except Exception:
        return None


@router.post("/api/control/osc/start")
async def osc_executor_start():
    """
    Release FCI from SPARK and spawn the rr-session OSC backend.
    """
    if _proc_alive():
        return JSONResponse(
            status_code=409,
            content={
                "error": "osc executor already running",
                "pid": state.osc_executor_proc.pid,
                "url": OSC_HTTP_URL,
            },
        )
    # Mutual exclusion with viser teleop, both can't own the FCI.
    if state.viser_teleop_proc is not None and state.viser_teleop_proc.poll() is None:
        return JSONResponse(
            status_code=409,
            content={
                "error": (
                    "viser teleop subprocess owns the FCI. Stop it via "
                    "POST /api/teleop/viser/stop before starting OSC"
                ),
            },
        )

    # franka-only backend (panda-py OSC controller). Refuse other families.
    pipeline = state.pipeline
    family = (
        getattr(getattr(pipeline, "config", None), "robot_family", "ur10e") or "ur10e"
    ).lower()
    if family != "franka":
        return JSONResponse(
            status_code=400,
            content={"error": f"osc executor is franka-only (family={family})"},
        )

    if not RR_VENV_PY.exists():
        return JSONResponse(
            status_code=400,
            content={
                "error": (
                    f"robots_realtime venv not found at {RR_VENV_PY}. "
                    "Install with: cd ~/robots_realtime && "
                    "uv venv --python 3.11 && uv pip install -e ."
                ),
            },
        )
    if not RR_CONFIG.exists():
        return JSONResponse(
            status_code=400,
            content={"error": f"osc executor config missing at {RR_CONFIG}"},
        )

    # Orphan reaper: a prior SPARK server may have died before its OSC
    # subprocess was killed (the subprocess uses start_new_session=True
    # so it survives the parent). Any leftover rr-session still holds
    # ZMQ port 5555 and blocks the new spawn with "Address already in
    # use". Sweep before spawning.
    try:
        _scan = subprocess.run(
            ["pgrep", "-f", "robots_realtime.rr_session_cli"],
            capture_output=True,
            text=True,
            timeout=2.0,
        )
        orphan_pids = [int(p) for p in _scan.stdout.split() if p.isdigit()]
        # Don't kill the proc we already own (shouldn't happen since the
        # liveness guard above returned, but defensive).
        own_pid = (
            state.osc_executor_proc.pid if state.osc_executor_proc is not None else None
        )
        for pid in orphan_pids:
            if pid == own_pid:
                continue
            try:
                _pgid = os.getpgid(pid)
                logger.info(
                    "osc_executor_start: reaping orphan rr-session pgid=%d "
                    "(pid=%d) before spawn",
                    _pgid,
                    pid,
                )
                os.killpg(_pgid, signal.SIGTERM)
            except Exception as _kex:
                logger.debug("orphan reap pid=%d: %s", pid, _kex)
        if orphan_pids:
            time.sleep(2.0)  # allow ZMQ port to release
            # Hard-kill survivors
            for pid in orphan_pids:
                try:
                    os.killpg(os.getpgid(pid), signal.SIGKILL)
                except Exception:
                    pass
            time.sleep(0.5)
    except Exception as exc:
        logger.warning(
            "osc_executor_start: orphan sweep raised: %s " "(proceeding anyway)", exc
        )

    # Release the FCI before spawning. Same logic as viser_teleop_start.
    released_robot = False
    if pipeline is not None and getattr(pipeline, "_robot", None) is not None:
        try:
            pipeline._robot.disconnect()
            pipeline._robot = None
            released_robot = True
            logger.info(
                "osc_executor_start: released SPARK FCI session for handoff",
            )
        except Exception as exc:
            logger.warning(
                "osc_executor_start: SPARK disconnect raised: %s "
                "(spawning subprocess anyway; it will error if FCI is "
                "still held)",
                exc,
            )

    # Build a per-spawn robot_config YAML pointing at FrankaFranky.
    # FrankaFranky (franky 1.1.3 / libfranka 0.18.0) drives the FR3
    # over server protocol v10; panda-py 0.8.1 / libfranka 0.13.3
    # cannot do this firmware. The shim implements the i2rt.Robot
    # interface RobotNode expects, with franky's async
    # JointWaypointMotion as the inner setpoint-streaming loop.
    desk_user = os.environ.get("FRANKA_DESK_USER")
    desk_pass = os.environ.get("FRANKA_DESK_PASSWORD")
    # FR3 FCI host: prefer the active pipeline's resolved robot IP (yaml /
    # --ip override), then FRANKA_IP, then the franka-family fallback.
    franka_ip = getattr(getattr(pipeline, "config", None), "robot_ip", "") or \
        os.environ.get("FRANKA_IP") or "172.16.0.2"
    session_cfg_path = str(RR_CONFIG)
    if desk_user and desk_pass:
        try:
            with open(RR_CONFIG) as _fh:
                _session_cfg = yaml.safe_load(_fh) or {}
            # The rr-session loader passes robot_config as a STRING path
            # to _instantiate_from_target_yaml, it does not accept an
            # inline dict. Write a sibling robot_config YAML pointing at
            # FrankaFranky, with the FR3 Desk credentials baked in so
            # the subprocess can auto-unlock + enable FCI on its own.
            _robot_cfg_tmp = tempfile.NamedTemporaryFile(
                mode="w",
                suffix=".yaml",
                delete=False,
                dir=str(RR_CONFIG.parent),
                prefix="osc_robot_cfg_with_creds_",
            )
            yaml.safe_dump(
                {
                    "_target_": "robots_realtime.robots.franka_franky.FrankaFranky",
                    "host_name": franka_ip,
                    "username": desk_user,
                    "password": desk_pass,
                    "enable_gripper": True,
                    "relative_dynamics_factor": 0.18,
                },
                _robot_cfg_tmp,
                sort_keys=False,
            )
            _robot_cfg_tmp.close()
            for _node in _session_cfg.get("nodes", []):
                if _node.get("type") == "RobotNode":
                    _node["robot_config"] = _robot_cfg_tmp.name
                    break
            _tmp = tempfile.NamedTemporaryFile(
                mode="w",
                suffix=".yaml",
                delete=False,
                dir=str(RR_CONFIG.parent),
                prefix="osc_session_with_creds_",
            )
            yaml.safe_dump(_session_cfg, _tmp, sort_keys=False)
            _tmp.close()
            session_cfg_path = _tmp.name
            logger.info(
                "osc_executor_start: wrote session config with creds to %s",
                session_cfg_path,
            )
        except Exception as _ex:
            logger.warning(
                "osc_executor_start: cred-injection failed (%s); "
                "spawning with credential-less config; panda-py will "
                "probably hang on FCI handshake.",
                _ex,
            )

    cmd = [
        # `python -m robots_realtime` would require __main__.py at the
        # top of the package; there isn't one. The actual CLI lives
        # at robots_realtime.rr_session_cli (entry-point: rr-session).
        str(RR_VENV_PY),
        "-m",
        "robots_realtime.rr_session_cli",
        session_cfg_path,
        "--no-tui",
    ]
    env = os.environ.copy()
    # Strip conda's LD_LIBRARY_PATH so the uv-venv's pinned libfranka
    # (0.13.3 bundled with panda-py 0.8.1) wins over spark_conda's
    # franky build.
    env.pop("LD_LIBRARY_PATH", None)
    uv_bin = os.path.expanduser("~/.local/bin")
    env["PATH"] = f"{uv_bin}:{env.get('PATH', '')}"
    # Force unbuffered stdout/stderr so we see RobotNode / panda-py
    # init errors in real time (default block buffering can hide a
    # crash that happens immediately on FCI connect, leaving the bridge
    # to silently IK-solve moves without actually executing them).
    env["PYTHONUNBUFFERED"] = "1"

    log_dir = RR_ROOT / ".spark_logs"
    log_dir.mkdir(parents=True, exist_ok=True)
    log_path = log_dir / "osc_executor.log"
    log_fh = open(log_path, "ab", buffering=0)

    try:
        proc = subprocess.Popen(
            cmd,
            cwd=str(RR_ROOT),
            env=env,
            stdout=log_fh,
            stderr=subprocess.STDOUT,
            stdin=subprocess.DEVNULL,
            start_new_session=True,
        )
    except Exception as exc:
        log_fh.close()
        if released_robot:
            try:
                pipeline._init_robot()
            except Exception:
                pass
        return JSONResponse(
            status_code=500,
            content={"error": f"failed to spawn rr-session: {exc}"},
        )

    state.osc_executor_proc = proc
    state.osc_executor_log_fh = log_fh
    logger.info(
        "osc_executor_start: pid=%d log=%s",
        proc.pid,
        log_path,
    )

    # Wait for the HTTP bridge to come up. The agent's __init__ does
    # a pyroki JIT warm-up that can take several seconds.
    deadline = time.monotonic() + _HTTP_READY_TIMEOUT_S
    ready = False
    last_err = "no response"
    while time.monotonic() < deadline:
        if proc.poll() is not None:
            break
        st = _http_get("/state", timeout=1.0)
        if st is not None and st.get("ok"):
            ready = True
            break
        time.sleep(0.5)

    if proc.poll() is not None:
        try:
            tail = log_path.read_text()[-4000:]
        except Exception:
            tail = "(log unavailable)"
        state.osc_executor_proc = None
        log_fh.close()
        state.osc_executor_log_fh = None
        if released_robot:
            try:
                pipeline._init_robot()
            except Exception as exc:
                logger.warning(
                    "osc_executor_start: failed to reclaim FCI after "
                    "subprocess died: %s",
                    exc,
                )
        return JSONResponse(
            status_code=500,
            content={
                "error": "rr-session subprocess exited immediately",
                "exit_code": proc.returncode,
                "log_tail": tail,
            },
        )

    if not ready:
        return JSONResponse(
            status_code=503,
            content={
                "error": (
                    f"rr-session is alive but HTTP bridge at {OSC_HTTP_URL} "
                    f"did not respond within {_HTTP_READY_TIMEOUT_S} s. "
                    f"Last error: {last_err}. "
                    "Check /api/control/osc/log for details."
                ),
                "pid": proc.pid,
            },
        )

    # Install the OSC robot proxy so SPARK's executor sees a usable
    # robot at pipeline._robot instead of None. The proxy forwards
    # every primitive (open/close gripper, get_joint_positions,
    # move_linear, go_home, etc.) to the bridge's HTTP endpoints.
    # Without this, every executor call after the FCI handoff would
    # hit None and the trial would no-op.
    try:
        proxy = OSCRobotProxy(bridge_url=OSC_HTTP_URL)
        pipeline._robot = proxy
        if getattr(pipeline, "_executor", None) is not None:
            pipeline._executor.robot = proxy
        logger.info(
            "osc_executor_start: installed OSCRobotProxy at "
            "pipeline._robot (and pipeline._executor.robot)"
        )
    except Exception as exc:
        logger.warning(
            "osc_executor_start: OSCRobotProxy install failed: %s "
            "(subprocess still running; executor will see no robot)",
            exc,
        )

    return {
        "success": True,
        "pid": proc.pid,
        "url": OSC_HTTP_URL,
        "log_path": str(log_path),
        "note": (
            "OSC backend ready. pipeline._robot is now an OSCRobotProxy "
            "forwarding to the bridge. Set SPARK_EXECUTOR_BACKEND=osc "
            "in the server env (or pass backend=osc) before /api/execute."
        ),
    }


@router.post("/api/control/osc/stop")
async def osc_executor_stop():
    """
    Terminate the rr-session subprocess and reclaim FCI for SPARK.
    """
    if not _proc_alive():
        pipeline = state.pipeline
        reclaimed = False
        if pipeline is not None and getattr(pipeline, "_robot", None) is None:
            try:
                pipeline._init_robot()
                reclaimed = pipeline._robot is not None
            except Exception as exc:
                return JSONResponse(
                    status_code=500,
                    content={"error": f"FCI reclaim raised: {exc}"},
                )
        return {"success": True, "was_running": False, "reclaimed": reclaimed}

    proc = state.osc_executor_proc
    pid = proc.pid
    logger.info("osc_executor_stop: SIGTERM pid=%d", pid)

    try:
        os.killpg(os.getpgid(pid), signal.SIGTERM)
    except ProcessLookupError:
        pass
    except Exception as exc:
        logger.warning("osc_executor_stop: killpg SIGTERM raised: %s", exc)

    deadline = time.monotonic() + _STOP_TIMEOUT_S
    while time.monotonic() < deadline and proc.poll() is None:
        time.sleep(0.1)

    if proc.poll() is None:
        logger.warning("osc_executor_stop: SIGTERM timed out, SIGKILL")
        try:
            os.killpg(os.getpgid(pid), signal.SIGKILL)
        except Exception:
            pass
        try:
            proc.wait(timeout=2.0)
        except subprocess.TimeoutExpired:
            pass

    exit_code = proc.returncode
    state.osc_executor_proc = None
    if state.osc_executor_log_fh is not None:
        try:
            state.osc_executor_log_fh.close()
        except Exception:
            pass
        state.osc_executor_log_fh = None

    pipeline = state.pipeline
    reclaimed = False
    if pipeline is not None:
        try:
            pipeline._init_robot()
            reclaimed = pipeline._robot is not None
        except Exception as exc:
            return JSONResponse(
                status_code=500,
                content={
                    "error": f"FCI reclaim failed: {exc}",
                    "subprocess_exit_code": exit_code,
                },
            )

    logger.info(
        "osc_executor_stop: pid=%d exit=%s reclaimed=%s",
        pid,
        exit_code,
        reclaimed,
    )
    return {
        "success": True,
        "was_running": True,
        "subprocess_exit_code": exit_code,
        "reclaimed": reclaimed,
    }


@router.get("/api/control/osc/status")
async def osc_executor_status():
    """
    Report subprocess liveness + bridge state.
    """
    alive = _proc_alive()
    payload = {
        "running": alive,
        "venv_installed": RR_VENV_PY.exists(),
        "config_path": str(RR_CONFIG),
        "url": OSC_HTTP_URL if alive else None,
        "backend_env": os.environ.get("SPARK_EXECUTOR_BACKEND", "franky"),
    }
    if alive:
        payload["pid"] = state.osc_executor_proc.pid
        st = _http_get("/state", timeout=1.0)
        if st is not None:
            payload["bridge"] = st
    return payload


@router.post("/api/control/osc/move_to_pose")
async def osc_executor_move_to_pose(body: dict):
    """
    Forward a Cartesian-pose move to the rr-session HTTP bridge.

    Body:
        position:     [x, y, z]            (meters, base frame)
        wxyz:         [w, x, y, z]         (unit quaternion)
        duration_s:   optional float       (default 2.0)
        gripper_width: optional float      (meters; default = last setting)

    Blocks until the bridge has solved IK and the requested duration
    has elapsed (so the call has move_linear semantics).
    """
    if not _proc_alive():
        return JSONResponse(
            status_code=400,
            content={
                "error": (
                    "OSC executor subprocess is not running. "
                    "Start with POST /api/control/osc/start first."
                ),
            },
        )

    try:
        position = [float(x) for x in body["position"]]
        wxyz = [float(x) for x in body["wxyz"]]
    except (KeyError, TypeError, ValueError) as exc:
        return JSONResponse(
            status_code=400,
            content={"error": f"bad position/wxyz: {exc}"},
        )

    duration_s = float(body.get("duration_s", 2.0))
    payload = {
        "position": position,
        "wxyz": wxyz,
        "duration_s": duration_s,
    }
    if "gripper_width" in body:
        payload["gripper_width"] = float(body["gripper_width"])

    raw = json.dumps(payload).encode()
    req = urllib.request.Request(
        f"{OSC_HTTP_URL}/move_to_pose",
        data=raw,
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    # HTTP timeout slightly larger than the duration to allow IK + return.
    http_timeout = max(duration_s + 5.0, 15.0)
    try:
        with urllib.request.urlopen(req, timeout=http_timeout) as r:
            resp = json.loads(r.read().decode())
        return resp
    except urllib.error.HTTPError as exc:
        try:
            body_err = exc.read().decode()
        except Exception:
            body_err = ""
        return JSONResponse(
            status_code=exc.code,
            content={
                "error": f"bridge HTTP {exc.code}",
                "body": body_err,
            },
        )
    except Exception as exc:
        return JSONResponse(
            status_code=502,
            content={"error": f"bridge request failed: {exc!r}"},
        )


@router.post("/api/control/osc/gripper")
async def osc_executor_gripper(body: dict):
    """
    Open or close the gripper via the bridge.

    Body: {"action": "open" | "close"}
    """
    if not _proc_alive():
        return JSONResponse(
            status_code=400,
            content={"error": "OSC executor subprocess is not running"},
        )
    action = body.get("action", "").strip().lower()
    if action not in ("open", "close"):
        return JSONResponse(
            status_code=400,
            content={"error": "action must be 'open' or 'close'"},
        )
    req = urllib.request.Request(
        f"{OSC_HTTP_URL}/{action}",
        data=b"{}",
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    try:
        with urllib.request.urlopen(req, timeout=5.0) as r:
            return json.loads(r.read().decode())
    except Exception as exc:
        return JSONResponse(
            status_code=502,
            content={"error": f"bridge request failed: {exc!r}"},
        )


@router.get("/api/control/osc/log")
async def osc_executor_log(tail_bytes: int = 8192):
    """
    Return the tail of the rr-session log.
    """
    log_path = RR_ROOT / ".spark_logs" / "osc_executor.log"
    if not log_path.exists():
        return {"log": "", "exists": False}
    try:
        with open(log_path, "rb") as fh:
            fh.seek(0, 2)
            size = fh.tell()
            fh.seek(max(0, size - tail_bytes))
            data = fh.read().decode("utf-8", errors="replace")
        return {"log": data, "size": size}
    except Exception as exc:
        return JSONResponse(
            status_code=500,
            content={"error": str(exc)},
        )

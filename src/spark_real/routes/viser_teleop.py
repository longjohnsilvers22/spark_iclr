"""
Subprocess-managed viser IK teleop (robots_realtime integration).

Bridges SPARK's FastAPI server to the external `robots_realtime` package
(https://github.com/uynitsuj/robots_realtime), which provides a browser-
based SE(3) IK gizmo teleop powered by pyroki at 100 Hz.

Why a subprocess and not in-process?
  - robots_realtime requires Python 3.11 (SPARK runs in spark_conda).
  - It uses panda-py / libfranka (different version vs SPARK's franky).
  - The FCI session is process-exclusive, only one libfranka client can
    drive the controller at a time.

So the contract is: SPARK owns the FCI most of the time; when the operator
hits "Start Viser Teleop":
  1. SPARK disconnects its franky driver (releases the FCI control session)
  2. We spawn `uv run rr-session <franka_spark_viser_teleop.yaml> --no-tui`
     in the external uv venv at ~/robots_realtime/.venv
  3. Operator drags the viser gizmo (http://<host>:8765) to teleop the arm
  4. On "Stop": SIGTERM the subprocess, wait for clean exit, then call
     pipeline._init_robot() to take FCI back.

This is mutually exclusive with the existing gamepad/keyboard teleop and
with any executor run. The SPARK status endpoint will report no
robot_connected while the subprocess is alive.
"""

from __future__ import annotations

import logging
import os
import signal
import subprocess
import time
from pathlib import Path
from typing import Optional

from fastapi import APIRouter
from fastapi.responses import JSONResponse

from spark_real.routes import state

logger = logging.getLogger(__name__)
router = APIRouter()

# External package location and venv. The wrapper is installed at
# ~/robots_realtime by the user; the uv venv lives at .venv inside it.
RR_ROOT = Path(os.path.expanduser("~/robots_realtime"))
RR_VENV_PY = RR_ROOT / ".venv" / "bin" / "python"
RR_CONFIG = RR_ROOT / "configs" / "franka" / "franka_spark_viser_teleop.yaml"
VISER_PORT = 8765

# Maximum seconds to wait for the subprocess to exit on SIGTERM before
# escalating. The rr_session_cli sets a 3s SIGTERM handler that hard-kills
# the process group, so 6s is a generous upper bound.
_STOP_TIMEOUT_S = 6.0

# Maximum seconds to wait for viser to be reachable after start.
_START_TIMEOUT_S = 30.0


def _proc_alive() -> bool:
    p = state.viser_teleop_proc
    return p is not None and p.poll() is None


def _viser_url(request_host: Optional[str] = None) -> str:
    """
    Build a URL the browser can reach. If request_host is provided
    (Host: header from the incoming HTTP request), we reuse its hostname
    so a browser that hit SPARK on http://lab-pc:8888 doesn't get sent
    to localhost (which would be wrong on a remote browser).
    """
    host = "localhost"
    if request_host:
        host = request_host.split(":", 1)[0]
    return f"http://{host}:{VISER_PORT}"


@router.post("/api/teleop/viser/start")
async def viser_teleop_start():
    """
    Release FCI from SPARK and spawn the rr-session subprocess.

    Returns the viser URL on success. Returns 409 if a subprocess is
    already running (the caller should /stop first), 400 if the external
    venv or config is missing.
    """
    if _proc_alive():
        return JSONResponse(
            status_code=409,
            content={
                "error": "viser teleop already running",
                "viser_url": _viser_url(),
                "pid": state.viser_teleop_proc.pid,
            },
        )

    # franka-only backend (panda-py / libfranka). Refuse other families.
    pipeline = state.pipeline
    family = (
        getattr(getattr(pipeline, "config", None), "robot_family", "ur10e") or "ur10e"
    ).lower()
    if family != "franka":
        return JSONResponse(
            status_code=400,
            content={"error": f"viser teleop is franka-only (family={family})"},
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
            content={"error": f"viser teleop config missing at {RR_CONFIG}"},
        )

    # Release the FCI before spawning. The new process will fail to
    # acquire the controller if SPARK is still holding it.
    released_robot = False
    if pipeline is not None and getattr(pipeline, "_robot", None) is not None:
        try:
            pipeline._robot.disconnect()
            pipeline._robot = None
            released_robot = True
            logger.info(
                "viser_teleop_start: released SPARK FCI session for handoff",
            )
        except Exception as exc:
            logger.warning(
                "viser_teleop_start: SPARK disconnect raised: %s "
                "(spawning subprocess anyway; it will error if FCI is "
                "still held)",
                exc,
            )

    # Spawn in a new process group so we can clean-kill descendants
    # (the rr-session runtime forks per-node subprocesses).
    # `python -m robots_realtime` would require __main__.py at the top
    # of the package; there isn't one. Use the CLI module path that
    # backs the `rr-session` installed entry point.
    cmd = [
        str(RR_VENV_PY),
        "-m",
        "robots_realtime.rr_session_cli",
        str(RR_CONFIG),
        "--no-tui",
    ]
    env = os.environ.copy()
    # Strip conda's LD_LIBRARY_PATH so the uv-venv's pinned libfranka
    # (0.10.x bundled with panda-py) wins over spark_conda's franky build.
    env.pop("LD_LIBRARY_PATH", None)
    # uv's PATH so the rr-session entrypoint can find its own bins.
    uv_bin = os.path.expanduser("~/.local/bin")
    env["PATH"] = f"{uv_bin}:{env.get('PATH', '')}"

    log_dir = RR_ROOT / ".spark_logs"
    log_dir.mkdir(parents=True, exist_ok=True)
    log_path = log_dir / "viser_teleop.log"
    log_fh = open(log_path, "ab", buffering=0)

    try:
        proc = subprocess.Popen(
            cmd,
            cwd=str(RR_ROOT),
            env=env,
            stdout=log_fh,
            stderr=subprocess.STDOUT,
            stdin=subprocess.DEVNULL,
            start_new_session=True,  # own process group
        )
    except Exception as exc:
        log_fh.close()
        # Restore SPARK driver if we couldn't even spawn.
        if released_robot:
            try:
                pipeline._init_robot()
            except Exception:
                pass
        return JSONResponse(
            status_code=500,
            content={"error": f"failed to spawn rr-session: {exc}"},
        )

    state.viser_teleop_proc = proc
    state.viser_teleop_log_fh = log_fh
    logger.info(
        "viser_teleop_start: pid=%d log=%s",
        proc.pid,
        log_path,
    )

    # Wait briefly so we can report whether the process at least started
    # cleanly. Full viser readiness can take several seconds (jax JIT,
    # urdf load); we don't block on it here.
    time.sleep(1.5)
    if proc.poll() is not None:
        # Already exited, read tail of log for the caller.
        try:
            tail = log_path.read_text()[-2000:]
        except Exception:
            tail = "(log unavailable)"
        state.viser_teleop_proc = None
        log_fh.close()
        state.viser_teleop_log_fh = None
        # Reclaim FCI for SPARK so the rig isn't stranded.
        if released_robot:
            try:
                pipeline._init_robot()
            except Exception as exc:
                logger.warning(
                    "viser_teleop_start: failed to reclaim FCI after "
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

    return {
        "success": True,
        "pid": proc.pid,
        "viser_url": _viser_url(),
        "log_path": str(log_path),
        "note": (
            "Viser is loading (jax JIT + URDF). Allow ~10s before the "
            "gizmo becomes responsive."
        ),
    }


@router.post("/api/teleop/viser/stop")
async def viser_teleop_stop():
    """
    Terminate the rr-session subprocess and reclaim the FCI for SPARK.
    """
    if not _proc_alive():
        # Even if no proc, attempt the reclaim in case state got out of
        # sync with reality (e.g. user killed the proc manually).
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

    proc = state.viser_teleop_proc
    pid = proc.pid
    logger.info("viser_teleop_stop: SIGTERM pid=%d", pid)

    # SIGTERM the whole process group (rr-session spawns per-node procs).
    try:
        os.killpg(os.getpgid(pid), signal.SIGTERM)
    except ProcessLookupError:
        pass
    except Exception as exc:
        logger.warning("viser_teleop_stop: killpg SIGTERM raised: %s", exc)

    deadline = time.monotonic() + _STOP_TIMEOUT_S
    while time.monotonic() < deadline and proc.poll() is None:
        time.sleep(0.1)

    if proc.poll() is None:
        # rr-session's own SIGTERM handler will hard-kill at +3s; this is
        # the belt + suspenders if that didn't fire.
        logger.warning("viser_teleop_stop: SIGTERM timed out, sending SIGKILL")
        try:
            os.killpg(os.getpgid(pid), signal.SIGKILL)
        except Exception:
            pass
        try:
            proc.wait(timeout=2.0)
        except subprocess.TimeoutExpired:
            pass

    exit_code = proc.returncode
    state.viser_teleop_proc = None
    if state.viser_teleop_log_fh is not None:
        try:
            state.viser_teleop_log_fh.close()
        except Exception:
            pass
        state.viser_teleop_log_fh = None

    # Reclaim FCI for SPARK.
    pipeline = state.pipeline
    reclaimed = False
    if pipeline is not None:
        try:
            # _init_robot disconnects any stale driver first, so this is
            # safe to call unconditionally.
            pipeline._init_robot()
            reclaimed = pipeline._robot is not None
        except Exception as exc:
            return JSONResponse(
                status_code=500,
                content={
                    "error": f"FCI reclaim failed after subprocess exit: {exc}",
                    "subprocess_exit_code": exit_code,
                    "subprocess_terminated": True,
                },
            )

    logger.info(
        "viser_teleop_stop: pid=%d exit=%s reclaimed=%s",
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


@router.get("/api/teleop/viser/status")
async def viser_teleop_status():
    """
    Liveness probe, useful for the UI button state.
    """
    alive = _proc_alive()
    payload = {
        "running": alive,
        "venv_installed": RR_VENV_PY.exists(),
        "config_path": str(RR_CONFIG),
        "viser_url": _viser_url() if alive else None,
    }
    if alive:
        payload["pid"] = state.viser_teleop_proc.pid
    return payload


@router.get("/api/teleop/viser/log")
async def viser_teleop_log(tail_bytes: int = 8192):
    """
    Return the tail of the rr-session log for debugging.
    """
    log_path = RR_ROOT / ".spark_logs" / "viser_teleop.log"
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

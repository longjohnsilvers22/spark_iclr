# Robot control routes: gripper, movement, velocity, home, stop, recover.

import logging
import os
import subprocess
import time
from pathlib import Path

import numpy as np
from fastapi import APIRouter
from fastapi.responses import JSONResponse

from spark_real.robots.franka.franka_base import HOME_CONFIG as _FRANKA_HOME_CONFIG
from spark_real.perception import camera as camera_mod
from spark_real.perception.camera import USB_LIFECYCLE_LOCK, usb_reset_disabled
from spark_real.routes import state
from spark_real.routes.models import (
    ConnectRobotRequest,
    GripperRequest,
    GripperPositionRequest,
    GraspTestRequest,
)
from spark_real.routes.control_teleop import router as _teleop_router

logger = logging.getLogger("spark_server")
router = APIRouter()


@router.post("/api/connect_robot")
def connect_robot(req: ConnectRobotRequest):
    pipeline = state.pipeline
    if pipeline is None:
        return JSONResponse(
            status_code=400, content={"error": "Pipeline not initialized"}
        )
    target_ip = (req.robot_ip or "").strip() or pipeline.config.robot_ip
    if not target_ip:
        return {"connected": False, "error": "No robot IP configured."}
    try:
        pipeline.config.robot_ip = target_ip
        pipeline._init_robot()
        connected = pipeline._robot is not None
        return {
            "connected": connected,
            "robot_ip": target_ip,
            "robot_family": getattr(pipeline.config, "robot_family", "ur10e"),
        }
    except Exception as e:
        return {"connected": False, "robot_ip": target_ip, "error": str(e)}


@router.post("/api/robot/release")
def release_robot():
    """Disconnect the robot so an external process (teleop) can own the
    exclusive RTDE channel. Reacquire with POST /api/connect_robot."""
    pipeline = state.pipeline
    if pipeline is None:
        return JSONResponse(
            status_code=400, content={"error": "Pipeline not initialized"}
        )
    robot = getattr(pipeline, "_robot", None)
    if robot is None:
        return {"released": True, "note": "no robot connected"}
    inner = getattr(robot, "_robot", None) or robot  # unwrap SafeRobot
    try:
        inner.disconnect()
    except Exception as e:  # noqa: BLE001
        logger.warning("robot release: disconnect raised: %s", e)
    pipeline._robot = None
    pipeline._executor = None
    logger.info("Robot released for external control (reacquire: /api/connect_robot)")
    return {"released": True}


# Reset-endpoint lock budgets (module-level so tests can shrink them).
# _RESET_READ_LOCK_TIMEOUT_S: max wait for a per-device read lock before
# aborting the reset (a reader is mid-read; never close underneath it).
# _LIFECYCLE_ACQUIRE_TIMEOUT_S: max wait for the process-wide USB lifecycle
# lock before refusing (another open/close/reset is in flight).
_RESET_READ_LOCK_TIMEOUT_S = 2.0
_LIFECYCLE_ACQUIRE_TIMEOUT_S = 5.0


@router.post("/api/kinect_reset")
def kinect_reset():
    """
    Software power-cycle Azure Kinects via uhubctl.

    This is the most dangerous operation in the codebase: it cuts Vbus on a
    hub of a controller that is simultaneously streaming other cameras. It
    is disabled by default (SPARK_ALLOW_USB_RESET is unset) and every guard
    below has to pass before anything touches the bus.

    - the reset kill-switch refuses outright (zero reset traffic mode);
    - the process-wide USB_LIFECYCLE_LOCK is held for the whole endpoint so
      no other camera lifecycle transition (including RealSense
      hardware_reset recovery) can overlap it;
    - the wrist RealSense is quiesced first: it is on the SAME controller;
    - EVERY Kinect must report a clean close (thread joined AND libk4a
      stop() returned) before the power-cycle runs. A device whose capture
      thread is still alive is never power-cycled.

    The per-device read locks do not exclude the capture thread: read() is a
    cached frame accessor that never enters pyk4a, so those locks serialize
    readers of a numpy cache and nothing more. The "mid-read, reset aborted"
    check below is kept because racing
    a reader still returns garbage frames, but the real exclusion is the
    close() result check. Do not re-derive safety from the read locks.
    """
    if usb_reset_disabled():
        return JSONResponse(
            status_code=409,
            content={
                "error": "USB reset paths are disabled (default). "
                "kinect_reset power-cycles a hub on the controller that is "
                "streaming every other camera; set SPARK_ALLOW_USB_RESET=1 "
                "to arm it, with an operator present."
            },
        )
    if not USB_LIFECYCLE_LOCK.acquire(timeout=_LIFECYCLE_ACQUIRE_TIMEOUT_S):
        return JSONResponse(
            status_code=503,
            content={
                "error": "another camera lifecycle transition is in flight; "
                "retry shortly"
            },
        )
    try:
        return _kinect_reset_locked()
    finally:
        USB_LIFECYCLE_LOCK.release()


def _kinect_reset_locked():
    """
    Body of /api/kinect_reset; caller holds USB_LIFECYCLE_LOCK.
    """
    pipeline = state.pipeline
    closed = []
    if pipeline is not None:
        # Quiesce the wrist RealSense FIRST. It sits on the SAME xHCI
        # controller as both Kinects (2-1 vs 2-2/2-7 on 0000:00:14.0); its
        # capture thread must not submit bulk URBs through the Vbus cut and
        # the re-enumeration that follows.
        rs_dev = getattr(pipeline, "_realsense", None)
        if rs_dev is not None and hasattr(rs_dev, "request_stop"):
            try:
                rs_dev.request_stop()
            except Exception as exc:  # noqa: BLE001
                logger.warning("realsense request_stop raised: %s", exc)
        # Phase 1: take ALL per-device read locks up front. Abort the whole
        # reset (nothing closed) if any reader holds one past the budget.
        to_close = []
        held = []
        try:
            for attr, label in (("_kinect", "sideview"), ("_kinect2", "birdview")):
                dev = getattr(pipeline, attr, None)
                if dev is None:
                    continue
                lock_attr = (
                    "_kinect_read_lock" if attr == "_kinect" else "_kinect2_read_lock"
                )
                lock = getattr(pipeline, lock_attr, None)
                if lock is not None:
                    if not lock.acquire(timeout=_RESET_READ_LOCK_TIMEOUT_S):
                        return JSONResponse(
                            status_code=503,
                            content={
                                "error": f"{label} is mid-read; reset aborted "
                                "(nothing was closed)"
                            },
                        )
                    held.append(lock)
                to_close.append((attr, label, dev))
            # Phase 2: all locks held; now it is safe to close.
            #
            # The close RESULT is load-bearing. AzureKinectCamera.close()
            # returns False when its capture thread would not exit or when
            # libk4a's stop() had to be abandoned -- i.e. when a live thread
            # may still be doing libusb I/O on that device. Cutting Vbus or
            # deauthorizing a device in that state is precisely the
            # concurrent-teardown pattern behind the xHCI ring corruption
            # this endpoint is supposed to recover from.
            failed = []
            for attr, label, dev in to_close:
                try:
                    released = dev.close()
                except Exception as exc:  # noqa: BLE001
                    logger.warning("%s close raised: %s", label, exc)
                    released = False
                if released is False:
                    failed.append(label)
                    continue
                setattr(pipeline, attr, None)
                closed.append(label)
        finally:
            for lock in held:
                lock.release()
        if failed:
            return JSONResponse(
                status_code=409,
                content={
                    "error": "refusing to power-cycle: "
                    + ", ".join(failed)
                    + " did not release (capture thread still live or "
                    "libk4a stop() abandoned). Cutting Vbus under a live "
                    "libusb transfer is the exact xHCI-corruption pattern "
                    "this endpoint exists to recover from. Restart the "
                    "server (scripts/spark_server.sh) instead.",
                    "released_handles": closed,
                    "unreleased": failed,
                },
            )
        # An abandoned libk4a stop() keeps running with USB_LIFECYCLE_LOCK
        # free; do not touch the bus until it actually returns.
        if not camera_mod.await_teardown_clear(timeout=15.0):
            return JSONResponse(
                status_code=409,
                content={
                    "error": "a device teardown is still in flight after 15s; "
                    "refusing to power-cycle. Restart the server instead.",
                    "released_handles": closed,
                },
            )
        time.sleep(0.5)

    script_path = str(
        Path(__file__).resolve().parents[3] / "scripts" / "kinect_software_reset.sh"
    )
    try:
        proc = subprocess.run(
            ["bash", script_path],
            capture_output=True,
            text=True,
            timeout=30,
        )
    except Exception as exc:
        # Do NOT reopen here. The devices were closed cleanly (we verified
        # that above) and the bus may be half-cycled; reopening through
        # _init_kinects at this point re-enters the post-torch libk4a
        # load-ordering hazard on top of an unknown bus state. Report and let
        # the operator restart.
        return JSONResponse(
            status_code=500,
            content={
                "error": f"reset script failed: {exc}",
                "released_handles": closed,
                "detail": "Kinects are closed and NOT reopened; restart the "
                "server (scripts/spark_server.sh).",
            },
        )

    # sysfs authorize toggle. On this host the Kinect hubs are at 2-2/2-7 (SS)
    # and 1-1/1-8 (HS); a hardcoded path list risks deauthorizing whatever
    # unrelated device enumerates there. scripts/kinect_authorize_reset.sh
    # discovers the hubs by VID/PID (045e:097a/097b), verifies depth cameras
    # came back, and escalates to a root-hub toggle only if needed.
    sysfs_rc = None
    sysfs_script = str(
        Path(__file__).resolve().parents[3] / "scripts" / "kinect_authorize_reset.sh"
    )
    try:
        sysfs_proc = subprocess.run(
            ["sudo", "-n", "bash", sysfs_script],
            capture_output=True,
            text=True,
            timeout=60,
        )
        sysfs_rc = sysfs_proc.returncode
    except Exception:
        sysfs_rc = -1

    # Reopen is OPT-IN and off by default.
    #
    # pipeline._init_kinects() calls PyK4A().start() at a point where torch,
    # scipy BLAS, SAM3, RF-DETR, EquiGraspFlow and possibly franky are all
    # resident. That is exactly the load ordering server.py's early-Kinect
    # block exists to avoid (libk4a's device threads must be created before
    # those libraries set PTHREAD_PRIO_INHERIT on mutexes; see server.py's
    # module docstring). Re-entering it deliberately, unattended, on a
    # controller that has just been power-cycled is not a recovery strategy.
    # Default: report that the bus was reset and let the operator restart the
    # server, which reopens through the safe early path.
    reopened = False
    detail = ""
    reopen_requested = os.environ.get("SPARK_KINECT_RESET_REOPEN", "") not in (
        "",
        "0",
        "false",
        "False",
    )
    if reopen_requested and pipeline is not None and hasattr(pipeline, "_init_kinects"):
        try:
            pipeline._init_kinects()
            reopened = (
                getattr(pipeline, "_kinect", None) is not None
                or getattr(pipeline, "_kinect2", None) is not None
            )
        except Exception as exc:
            detail = str(exc)
    elif not reopen_requested:
        detail = (
            "bus reset done; Kinects NOT reopened in-process (that would run "
            "libk4a device-thread creation after torch/scipy are resident). "
            "Restart the server: scripts/spark_server.sh. Set "
            "SPARK_KINECT_RESET_REOPEN=1 to reopen in place anyway."
        )

    return {
        "released_handles": closed,
        "uhubctl_rc": proc.returncode,
        "sysfs_authorized_rc": sysfs_rc,
        "kinect_reopened": reopened,
        "detail": detail,
    }


@router.post("/api/stream_mode")
async def set_stream_mode(req: dict):
    state.stream_mode = req.get("mode", "rgb")
    return {"mode": state.stream_mode}


@router.post("/api/stream_camera")
async def set_stream_camera(req: dict):
    state.stream_camera = req.get("camera", "all")
    return {"camera": state.stream_camera}


@router.post("/api/gripper")
def control_gripper(req: GripperRequest):
    pipeline = state.pipeline
    if pipeline is None or pipeline._robot is None:
        return JSONResponse(status_code=400, content={"error": "Robot not connected"})
    if req.action == "open":
        pipeline._robot.open_gripper()
    elif req.action == "close":
        pipeline._robot.close_gripper()
    return {"action": req.action, "success": True}


@router.post("/api/gripper_position")
def control_gripper_position(req: GripperPositionRequest):
    """
    Variable-width gripper command. position 0.0 (open) to 1.0 (closed).
    """
    pipeline = state.pipeline
    if pipeline is None or pipeline._robot is None:
        return JSONResponse(status_code=400, content={"error": "Robot not connected"})
    pos = float(max(0.0, min(1.0, req.position)))
    try:
        pipeline._robot.set_gripper_position(pos, speed=req.speed, force=req.force)
    except TypeError:
        pipeline._robot.set_gripper_position(pos)
    except AttributeError:
        return JSONResponse(
            status_code=400, content={"error": "Driver has no set_gripper_position."}
        )
    return {"position": pos, "success": True}


@router.post("/api/grasp_test")
def grasp_test(req: GraspTestRequest):
    """
    Run one calibrated grasp_to_width and return metrics.
    """
    pipeline = state.pipeline
    if pipeline is None or pipeline._robot is None:
        return JSONResponse(status_code=400, content={"error": "Robot not connected"})
    if state.executor_running:
        return JSONResponse(status_code=409, content={"error": "Executor running"})
    robot = pipeline._robot
    if not hasattr(robot, "grasp_to_width"):
        return JSONResponse(
            status_code=400, content={"error": "No grasp_to_width support"}
        )

    def _read_wrench():
        try:
            if hasattr(robot, "get_tcp_force"):
                ft = np.asarray(robot.get_tcp_force(), dtype=float)
                if ft.size >= 3:
                    return [float(ft[0]), float(ft[1]), float(ft[2])]
        except Exception:
            pass
        return [0.0, 0.0, 0.0]

    try:
        robot.open_gripper()
    except Exception as exc:
        return JSONResponse(
            status_code=500, content={"error": f"open_gripper failed: {exc}"}
        )
    time.sleep(1.0)
    pre = _read_wrench()

    try:
        success_flag = bool(
            robot.grasp_to_width(
                width=float(req.target_width),
                force=float(req.force),
                speed=float(req.speed),
            )
        )
    except Exception as exc:
        return JSONResponse(
            status_code=500, content={"error": f"grasp_to_width raised: {exc}"}
        )

    time.sleep(float(req.settle_s))
    try:
        achieved_w = float(robot.get_gripper_width())
    except Exception:
        achieved_w = 0.0

    post = _read_wrench()
    delta_mag = float(np.linalg.norm([post[i] - pre[i] for i in range(3)]))
    inferred = bool(
        0.005 < achieved_w < 0.085 and abs(achieved_w - req.target_width) < 0.020
    )
    if req.release_after:
        try:
            robot.open_gripper()
        except Exception:
            pass

    return {
        "label": req.label,
        "target_width": req.target_width,
        "achieved_width": round(achieved_w, 4),
        "libfranka_grasp_success": success_flag,
        "delta_force_mag": round(delta_mag, 3),
        "inferred_holding": inferred,
    }


# no home for the family. /api/home reads the profile first; these literals
# mirror the family YAMLs (configs/ur10e_default.yaml, franka_base.HOME_CONFIG,
# configs/bimanual_franka_default.yaml home_config_left/right).
_HOME_JOINTS_BY_FAMILY = {
    "ur10e": [3.0247, -1.9862, -1.2566, 4.7979, 1.5987, -3.1957],
    "franka": list(_FRANKA_HOME_CONFIG),
    "bimanual_franka": [
        -0.010472,
        -0.537561,
        0.0,
        -2.197370,
        0.019199,
        1.673771,
        -0.797615,
        0.010472,
        -0.537561,
        0.0,
        -2.197370,
        -0.019199,
        1.673771,
        0.797615,
    ],
}


def _home_target_for(pipeline, family):
    """
    Resolve the home joint target as a flat list from the RobotProfile so
    every family reads one source (its family yaml). Single-arm yaml gives a
    joint list; bimanual gives {"left", "right"}, flattened to left+right.
    Falls back to the legacy dict only if the profile has no home.
    """
    profile = getattr(pipeline, "profile", None)
    if profile is not None:
        home = profile.home_config()
        if isinstance(home, list) and home:
            return list(home)
        if isinstance(home, dict):
            left = list(home.get("left") or [])
            right = list(home.get("right") or [])
            if left or right:
                return left + right
    # Fallback: legacy dict (mirrors the reconciled per-family yaml poses).
    return _HOME_JOINTS_BY_FAMILY.get(family)


@router.post("/api/home")
def go_home():
    pipeline = state.pipeline
    if pipeline is None or pipeline._robot is None:
        return JSONResponse(status_code=400, content={"error": "Robot not connected"})
    family = (getattr(pipeline.config, "robot_family", "ur10e") or "ur10e").lower()
    target = _home_target_for(pipeline, family)

    for stop_fn in ("stop", "stop_motion", "servo_stop"):
        if hasattr(pipeline._robot, stop_fn):
            try:
                getattr(pipeline._robot, stop_fn)()
                break
            except Exception:
                pass
    time.sleep(0.3)

    try:
        if hasattr(pipeline._robot, "_send_script") and family == "ur10e":
            joints_str = ",".join(f"{j:.4f}" for j in target)
            if not pipeline._robot._send_script(
                f"movej([{joints_str}], a=1.0, v=0.8)"
            ):
                # A dropped home command must not return success:true after
                # the poll loop breaks on a read exception.
                return JSONResponse(
                    status_code=500,
                    content={"error": "home movej dropped: URScript channel down"},
                )
        elif hasattr(pipeline._robot, "go_home"):
            pipeline._robot.go_home()
        else:
            pipeline._robot.move_to_joint_config(
                target,
                velocity=0.8,
                acceleration=1.0,
            )
    except Exception as e:
        return JSONResponse(status_code=500, content={"error": f"go_home failed: {e}"})

    for _ in range(100):
        time.sleep(0.1)
        try:
            joints = np.array(pipeline._robot.get_joint_positions())
            if len(joints) == len(target) and np.allclose(joints, target, atol=0.05):
                break
        except Exception:
            break
    return {"success": True}


@router.post("/api/stop")
async def stop_robot():
    """Brake the arm. The operator's stop button ends up here.

    This must reach the arm, not just set a flag: URScript motion is
    fire-and-forget, so an unbraked stop lets the arm run on to its target.
    Both entry points below brake -- the executor's abort() for the running
    score, and the driver's own escalation (URScript stopj -> reopened socket
    -> bounded Dashboard stop) for the arm itself, which is also the path when
    no score is running at all. The reply reports whether a stop actually left
    this process; `braked: false` means THE ARM MAY STILL BE MOVING.
    """
    pipeline = state.pipeline
    if pipeline is None or pipeline._robot is None:
        return JSONResponse(status_code=400, content={"error": "Robot not connected"})
    errors = []
    steps = []
    braked = False

    if pipeline._executor is not None:
        try:
            braked = bool(pipeline._executor.abort())
            steps.append("executor.abort" if braked else "executor.abort:no-brake")
        except Exception as e:
            errors.append(f"executor.abort: {e}")

    # Brake the driver directly whenever the executor did not: no score
    # running, abort() raised, or its escalation failed. Skipped once the arm is
    # already decelerating, because a second stop program REPLACES the first.
    # The escalation is the driver's (it owns the socket and the Dashboard
    # client); the old stop/stop_motion/servo_stop first-hit-wins loop is gone
    # -- it resolved to rtde_c.stopJ, which cannot stop this rig, and its
    # success masked the failure of the URScript stop above it.
    if not braked:
        emergency = getattr(pipeline._robot, "emergency_stop", None)
        if callable(emergency):
            try:
                result = emergency()
                steps.extend(result.get("steps", []))
                braked = bool(result.get("braked"))
            except Exception as e:
                errors.append(f"emergency_stop: {e}")
        else:
            for stop_fn in ("stop_motion", "stop", "servo_stop"):
                fn = getattr(pipeline._robot, stop_fn, None)
                if callable(fn):
                    try:
                        braked = fn() is not False or braked
                        steps.append(stop_fn)
                    except Exception as e:
                        errors.append(f"{stop_fn}: {e}")

    try:
        if pipeline._executor is not None:
            pipeline._executor._running = False
            # Belt-and-braces after executor.abort() above, but it must go
            # through note_abort_requested: a bare `_abort = True` does not
            # advance the abort epoch, so execute_score would wipe it when the
            # run finally reaches it. That matters most when abort() above
            # raised or when no score is running YET -- the executor thread is
            # in capture/detect/plan and execute_score is still seconds away.
            note = getattr(pipeline._executor, "note_abort_requested", None)
            if callable(note):
                note()
            else:
                pipeline._executor._abort = True
        state.executor_running = False
        with pipeline._activity_lock:
            pipeline._activity_stack.clear()
    except Exception:
        pass

    if not braked:
        logger.critical("/api/stop: THE ARM WAS NOT BRAKED (%s)", errors or steps)
    return {"success": True, "braked": braked, "steps": steps, "errors": errors}


@router.post("/api/recover")
def recover_robot():
    """
    Clear protective stop / reflex state so teleop can continue.
    """
    pipeline = state.pipeline
    if pipeline is None or pipeline._robot is None:
        return JSONResponse(status_code=400, content={"error": "Robot not connected"})

    robot = pipeline._robot
    actions, errors = [], []

    for fn in ("stop_velocity", "stop_motion", "servo_stop"):
        if hasattr(robot, fn):
            try:
                getattr(robot, fn)()
                actions.append(fn)
                break
            except Exception as e:
                errors.append(f"{fn}: {e}")

    for fn in (
        "recover_from_errors",
        "clear_protective_stop",
        "_send_dashboard_command",
    ):
        if hasattr(robot, fn):
            try:
                if fn == "_send_dashboard_command":
                    getattr(robot, fn)("close safety popup")
                else:
                    getattr(robot, fn)()
                actions.append(fn)
                break
            except Exception as e:
                errors.append(f"{fn}: {e}")

    if hasattr(robot, "cbf_deviated"):
        robot.cbf_deviated = False
        actions.append("cleared_cbf_deviated")

    nudge_info = _nudge_off_limit_joints(robot)
    if nudge_info:
        actions.append(nudge_info)

    return {"success": True, "actions": actions, "errors": errors}


def _nudge_off_limit_joints(robot):
    """
    Nudge joints near limits partway toward home.
    """
    try:
        target_robot = robot
        for attr in ("_robot", "_driver"):
            inner = getattr(target_robot, attr, None)
            if inner is not None:
                target_robot = inner
        limits = getattr(target_robot, "JOINT_LIMITS", None)
        home = getattr(target_robot, "HOME_CONFIG", None)
        if not limits or not home or not hasattr(robot, "get_joint_positions"):
            return None
        q_now = list(robot.get_joint_positions())
        if len(q_now) != len(limits):
            return None
        q_target = list(q_now)
        nudged = []
        for i, (q, (lo, hi), q_home) in enumerate(zip(q_now, limits, home)):
            rng = hi - lo
            if rng <= 0:
                continue
            margin = rng * 0.12
            if (q - lo) < margin or (hi - q) < margin:
                q_target[i] = q + 0.35 * (q_home - q)
                nudged.append(i)
        if not nudged:
            return None
        if hasattr(robot, "move_to_joint_config"):
            try:
                robot.move_to_joint_config(q_target, velocity=0.25)
            except TypeError:
                robot.move_to_joint_config(q_target)
            return f"nudged_joints={nudged}"
    except Exception:
        pass
    return None


@router.get("/api/robot_state")
async def get_robot_state():
    pipeline = state.pipeline
    if pipeline is None or pipeline._robot is None:
        return {"connected": False}
    try:
        if hasattr(pipeline._robot, "get_observation"):
            obs = pipeline._robot.get_observation()
        else:
            obs = {}
            for attr, key in (
                ("get_joint_positions", "joint_positions"),
                ("get_tcp_pose", "tcp_pose"),
                ("get_gripper_position", "gripper_position"),
            ):
                fn = getattr(pipeline._robot, attr, None)
                if callable(fn):
                    try:
                        obs[key] = fn()
                    except Exception:
                        pass
            obs.setdefault("gripper_position", 0.0)
    except Exception as e:
        return {"connected": False, "error": str(e)}

    def _list_or_none(val):
        if val is None:
            return None
        return val.tolist() if hasattr(val, "tolist") else list(val)

    raw_grip = obs.get("gripper_position")
    grip_value = None
    if raw_grip is not None:
        try:
            grip_value = float(raw_grip)
        except (TypeError, ValueError):
            pass

    # gObj object-detect flag + TCP speed (diagnostic: lets a caller confirm the
    # arm is stationary before trusting the gripper register read).
    obj_detected = None
    try:
        if hasattr(pipeline._robot, "is_object_detected"):
            obj_detected = bool(pipeline._robot.is_object_detected())
    except Exception:
        pass
    tcp_speed = None
    try:
        rr = getattr(pipeline._robot, "_rtde_r", None) or getattr(
            getattr(pipeline._robot, "_robot", None), "_rtde_r", None
        )
        if rr is not None:
            spd = np.asarray(rr.getActualTCPSpeed(), dtype=float)
            if spd.size >= 3:
                tcp_speed = float(np.linalg.norm(spd[:3]))
    except Exception:
        pass

    return {
        "connected": True,
        "joint_positions": _list_or_none(obs.get("joint_positions")),
        "tcp_pose": _list_or_none(obs.get("tcp_pose")),
        "gripper_position": grip_value,
        "object_detected": obj_detected,
        "tcp_speed": tcp_speed,
    }


# Teleop endpoints (move_relative / velocity / test_yaw) live in control_teleop.py.
router.include_router(_teleop_router)

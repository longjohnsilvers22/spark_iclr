"""
Core manipulation primitives for the real robot pipeline.

Primitives: move_to_keypoint, grasp, release, move_relative, wait,
tilt_wrist.

Extended primitives live one-per-module (pour.py, sweep.py, scrub.py,
place_in_slot.py, cloth_fold.py) and are auto-registered via @spark_skill.
"""

import time
import logging

import numpy as np
from scipy.spatial.transform import Rotation as _R

from spark_real.skills.registry import spark_skill
from spark_real.control import success_verifier
from spark_real.control.executor_types import ExecutionResult
from spark_real.control.waypoints import WaypointBuffer

logger = logging.getLogger(__name__)


def _result(action_type: str, success: bool, message: str = "", duration: float = 0.0):
    """
    Build an ExecutionResult.
    """
    return ExecutionResult(
        action_type=action_type,
        success=success,
        message=message,
        duration=duration,
    )


@spark_skill(
    name="move_to_keypoint",
    description="Move end-effector to a labeled keypoint (detected object)",
    params={
        "keypoint_label": str,
        "offset_x": float,
        "offset_y": float,
        "offset_z": float,
    },
)
def move_to_keypoint(executor, params: dict):
    """
    Move the end-effector to a detected object's position.
    """
    t0 = time.time()
    label = params.get("keypoint_label", "")
    offset_x = params.get("offset_x", 0.0)
    offset_y = params.get("offset_y", 0.0)
    # Default goes TO the keypoint (0.0) so a following grasp closes on it; a
    # 0.10 hover default made the gripper close 10cm above the object.
    offset_z = params.get("offset_z", 0.0)

    # Universal pre-approach re-bind (perception.rebind): a target moved by
    # a human between approval and this approach becomes a silent parameter
    # update, not a failed attempt + recovery. See execution_recovery.
    from spark_real.control.execution_recovery import rebind_before_approach
    rebind_before_approach(executor, label)
    det = executor.detection_map.get(label)
    if det is None:
        return _result(
            "move_to_keypoint",
            False,
            f"Object '{label}' not found in detections",
            time.time() - t0,
        )

    pos = np.array(det["position_3d"])
    target = pos + np.array([offset_x, offset_y, offset_z])

    if executor._holding:
        # target_label binds the arrival record to this container, which the
        # release gate checks.
        delivered = executor._transport_to(
            target, target_detection=det, target_label=label, params=params
        )
        if not executor._holding:
            return _result(
                "move_to_keypoint",
                False,
                f"Object lost during transport to '{label}'",
                time.time() - t0,
            )
        if not delivered:
            record = getattr(executor, "_place_arrival_record", None) or {}
            return _result(
                "move_to_keypoint",
                False,
                (
                    f"Did not arrive over '{label}': "
                    f"{record.get('detail', 'transport stalled short')}"
                ),
                time.time() - t0,
            )
    else:
        # Wrist camera is video-only; move directly to the birdview/sideview target.
        executor._approach_target(target, detection=det, params=params)
        # Keep the orientation the approach resolved. Re-commanding
        # GRASP_ORIENTATION here would un-rotate the wrist immediately after
        # an oriented approach, throwing away the yaw the approach just set.
        executor._move_to(
            target,
            getattr(executor, "_active_grasp_orient", None) or executor.GRASP_ORIENTATION,
            velocity=executor.velocity * 0.5,
        )

    return _result(
        "move_to_keypoint",
        True,
        f"Moved to '{label}' at {target.tolist()}",
        time.time() - t0,
    )


@spark_skill(
    name="grasp",
    description="Close gripper to grasp an object",
    params={"force": float},
)
def grasp(executor, params: dict):
    """
    Close the gripper with configurable force (0-100).
    """
    t0 = time.time()
    force = params.get("force", 100)
    if hasattr(executor.robot, "_send_gripper_command"):
        executor.robot._send_gripper_command(1.0, speed=100, force=min(force, 100))
    else:
        executor.robot.close_gripper()
    time.sleep(0.5)
    executor._holding = True
    return _result("grasp", True, f"Grasped with force={force}", time.time() - t0)


@spark_skill(
    name="close_gripper",
    description=(
        "Close the jaws WITHOUT asserting a grasp. Turns the two fingers into "
        "a flat tool for pressing, tamping, or holding. Never fails on empty."
    ),
    params={
        "force": float,
        "target_width": float,
    },
)
def close_gripper(executor, params: dict):
    """Close the jaws as a TOOL, not as a grasp.

    `grasp` verifies that something is held and fails the branch when nothing
    is, which is correct for grasping and wrong for every composition that
    uses the closed jaws as a pusher (a seat-the-screwdriver sequence of
    release, lift, close, press must not fail at the close).

    So this primitive closes and reports success on the CLOSING, not on the
    contents. It never sets _holding: nothing is held, and a later
    _grip_intact check must not be told otherwise.
    """
    t0 = time.time()
    force = float(params.get("force", 40.0))
    width = params.get("target_width", None)
    try:
        if width is not None and hasattr(executor.robot, "set_gripper_width"):
            executor.robot.set_gripper_width(float(width), force=force)
        else:
            executor.robot.close_gripper(force=int(force))
    except Exception as exc:  # noqa: BLE001
        logger.warning("[close_gripper] %s", exc)
        return _result("close_gripper", False, f"gripper command failed: {exc}",
                       time.time() - t0)
    time.sleep(0.5)
    logger.info(
        "[close_gripper] jaws closed as a tool (force=%.0f); NOT claiming a grasp",
        force,
    )
    return _result("close_gripper", True, f"Jaws closed (force={force})",
                   time.time() - t0)


@spark_skill(
    name="release",
    description="Open gripper to release held object and retract upward",
    params={
        "tilt_angle": float,
        "pitch_sign": float,
        "retract_z": float,
    },
)
def release(executor, params: dict):
    """
    Open the gripper, wait for settling, then retract upward.
    """
    t0 = time.time()

    # Flush pending waypoints before opening jaws
    _buf = getattr(executor, "waypoint_buffer", None)
    _franky = None
    if _buf is not None and hasattr(_buf, "pending") and _buf.pending > 0:
        try:
            _franky = WaypointBuffer.resolve_franky_robot(executor.robot)
        except Exception:
            _franky = None
    if (
        _buf is not None
        and _franky is not None
        and hasattr(_buf, "pending")
        and _buf.pending > 0
    ):
        try:
            logger.info("[release] flushing %d pending waypoint(s)", _buf.pending)
            _buf.barrier(_franky)
        except Exception as exc:
            logger.warning("[release] pre-barrier flush failed: %s", exc)
            try:
                _buf.clear()
            except Exception:
                pass

    # Arrival gate, same as the builtin _release path (which normally shadows
    # this skill). Kept in sync so a transport that never got over the
    # container cannot open the jaws just because dispatch routed release here.
    _delivered, _why = executor._transport_delivered()
    if not _delivered:
        success_verifier.set_gate(executor, "transport", False, _why)
        logger.warning("Release REFUSED, transport did not arrive: %s", _why)
        return _result(
            "release", False, f"Refused to open the jaws: {_why}", time.time() - t0
        )

    # G3 release witness, same as the builtin _release path (which normally
    # shadows this skill). Kept in sync so the gate cannot silently go missing
    # if dispatch ever routes release here.
    _witness_state = success_verifier.begin_release_witness(executor)
    executor.robot.open_gripper()
    time.sleep(0.5)
    executor._holding = False
    executor._place_arrival_record = None
    success_verifier.finish_release_witness(
        executor, _witness_state, getattr(executor, "_last_place_label", "")
    )

    current = executor._get_current_position()
    retract = current.copy()
    retract[2] += 0.05
    executor._move_to(retract, executor.GRASP_ORIENTATION)
    return _result("release", True, "Released", time.time() - t0)


@spark_skill(
    name="move_relative",
    description="Move end-effector relative to current position",
    params={"dx": float, "dy": float, "dz": float},
)
def move_relative(executor, params: dict):
    """
    Translate the TCP by (dx, dy, dz), preserving wrist orientation.
    """
    t0 = time.time()
    dx = params.get("dx", 0.0)
    dy = params.get("dy", 0.0)
    dz = params.get("dz", 0.0)
    current = executor._get_current_position()
    target = current + np.array([dx, dy, dz])

    orient = executor.GRASP_ORIENTATION
    try:
        obs = executor.robot.get_observation()
        tcp = obs.get("tcp_pose") or obs.get("tcp_pos")
        if tcp is not None and len(tcp) >= 6:
            orient = list(np.array(tcp[3:6], dtype=float))
    except Exception:
        pass

    executor._move_to(target, orient)
    return _result(
        "move_relative", True, f"Moved by ({dx}, {dy}, {dz})", time.time() - t0
    )


@spark_skill(
    name="wait",
    description="Pause execution for a specified duration",
    params={"duration": float},
)
def wait(executor, params: dict):
    """
    Sleep for the given duration in seconds.
    """
    t0 = time.time()
    duration = params.get("duration", 0.5)  # default when the plan omits it
    time.sleep(duration)
    return _result("wait", True, f"Waited {duration}s", time.time() - t0)


@spark_skill(
    name="tilt_wrist",
    description=(
        "Rotate the wrist in place by a specified angle about a local axis "
        "(x/y/z). Does NOT translate."
    ),
    params={"axis": str, "angle_rad": float},
)
def tilt_wrist(executor, params: dict):
    """
    Rotate the wrist in place about a local axis.
    """
    t0 = time.time()
    axis = str(params.get("axis", "y")).lower()
    angle = float(params.get("angle_rad", 0.0))
    if axis not in ("x", "y", "z"):
        return _result("tilt_wrist", False, f"Invalid axis '{axis}' (use x/y/z)")

    try:
        obs = executor.robot.get_observation()
        tcp = obs.get("tcp_pose") or obs.get("tcp_pos")
        if tcp is None:
            return _result(
                "tilt_wrist", False, "No tcp_pose in observation", time.time() - t0
            )
        current_pos = np.array(tcp[:3], dtype=float)
        current_orient_rv = np.array(tcp[3:6], dtype=float)
    except Exception as exc:
        return _result("tilt_wrist", False, f"Cannot read TCP: {exc}", time.time() - t0)

    current_R = _R.from_rotvec(current_orient_rv)
    tilt_R = _R.from_euler(axis, angle)
    new_orient = (tilt_R * current_R).as_rotvec().tolist()

    logger.info(
        "[tilt_wrist] axis=%s angle=%.1f deg at pos=(%.3f,%.3f,%.3f)",
        axis,
        np.rad2deg(angle),
        *current_pos,
    )

    try:
        executor._move_to(
            current_pos.tolist(), new_orient, velocity=executor.velocity * 0.3
        )
    except Exception as exc:
        return _result("tilt_wrist", False, f"Motion failed: {exc}", time.time() - t0)

    return _result(
        "tilt_wrist",
        True,
        f"Tilted {axis} by {np.rad2deg(angle):.1f} deg",
        time.time() - t0,
    )

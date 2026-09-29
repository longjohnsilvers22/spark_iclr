"""
Tool-use primitives: sponge washing, scrubbing, rinsing, pen-writing,
pouring, sweeping, throwing, and compliant insertion.
"""

from __future__ import annotations

import json
import logging
import math
import time
from pathlib import Path
from typing import Any, Dict, Optional

import numpy as np

from spark_real.skills.registry import spark_skill
from spark_real.skills.primitives import _result
from spark_real.control.executor_types import ExecutionResult

logger = logging.getLogger(__name__)


def tool_library_path() -> Path:
    p = Path(__file__).resolve().parents[1] / "output" / "tool_library.json"
    p.parent.mkdir(parents=True, exist_ok=True)
    return p


def _record_tool_use(name: str, success: bool) -> None:
    path = tool_library_path()
    try:
        if path.exists():
            data = json.loads(path.read_text())
        else:
            data = {}
        entry = data.setdefault(
            name, {"first_used_ts": time.time(), "uses": 0, "successes": 0}
        )
        entry["uses"] += 1
        if success:
            entry["successes"] += 1
        entry["last_used_ts"] = time.time()
        path.write_text(json.dumps(data, indent=2))
    except Exception:
        logger.exception("failed to record tool use for %s", name)


def _get_robot(executor):
    """
    The underlying driver used by both single-arm and per-arm sub-executors.
    """
    return executor.robot


def _tcp_xyz(executor) -> np.ndarray:
    return np.asarray(_get_robot(executor).get_tcp_pose(), dtype=float)[:3]


def _normalize(v: np.ndarray) -> np.ndarray:
    n = np.linalg.norm(v)
    return v / n if n > 1e-9 else v


@spark_skill(
    name="pour_to_level",
    description=(
        "Vision-grounded continuous pour: tilt the source container "
        "incrementally and re-detect the fluid mask in the target each "
        "tick. Stop tilting once the fluid mask exceeds "
        "target_fill_fraction of the target's interior mask, OR when "
        "max_tilt is reached. Use for 'pour X into Y until full / "
        "half-full'. Replaces open-loop pour for level-controlled tasks."
    ),
    params={
        "source_label": str,
        "target_label": str,
        "target_fill_fraction": float,
        "max_tilt_rad": float,
        "fluid_label": str,
    },
)
def pour_to_level(executor, params: dict):
    t0 = time.time()
    if not getattr(executor, "_holding", False):
        return _result(
            action_type="pour_to_level",
            success=False,
            message="not holding source container",
            duration=time.time() - t0,
        )
    source = params.get("source_label", "source")
    target = params.get("target_label")
    fluid = params.get("fluid_label", "water")
    if not target or target not in executor.detection_map:
        return _result(
            action_type="pour_to_level",
            success=False,
            message=f"target {target!r} not detected",
            duration=time.time() - t0,
        )
    fill_frac = float(params.get("target_fill_fraction", 0.5))
    max_tilt = float(params.get("max_tilt_rad", 1.7))
    tilt_increments = 8
    dwell_per_tick = 0.4

    target_det = executor.detection_map[target]
    target_pos = np.asarray(target_det.get("position_3d"), dtype=float)
    above = target_pos + np.array([0.0, -0.05, 0.18])
    base_ori = np.asarray(executor.GRASP_ORIENTATION, dtype=float)
    if not executor._servo.servo_to(np.concatenate([above, base_ori])):
        return _result(
            action_type="pour_to_level",
            success=False,
            message="approach failed",
            duration=time.time() - t0,
        )

    pipeline = getattr(executor, "_pipeline", None)
    robot = _get_robot(executor)
    if not hasattr(robot, "send_velocity"):
        return _result(
            action_type="pour_to_level",
            success=False,
            message="driver lacks send_velocity",
            duration=time.time() - t0,
        )

    last_fill: float = 0.0
    for step in range(1, tilt_increments + 1):
        if getattr(executor, "_abort", False):
            break
        tilt = max_tilt * step / tilt_increments
        robot.send_velocity(
            [0.0, 0.0, 0.0],
            [tilt / dwell_per_tick, 0.0, 0.0],
            acceleration=1.5,
            duration=dwell_per_tick,
        )
        time.sleep(0.05)
        fill = _measure_fill_fraction(pipeline, target_det, fluid_label=fluid)
        last_fill = fill if fill is not None else last_fill
        logger.info(
            "pour_to_level: tilt=%.2f rad, fill=%.2f / %.2f", tilt, last_fill, fill_frac
        )
        if last_fill is not None and last_fill >= fill_frac:
            break

    robot.send_velocity(
        [0.0, 0.0, 0.0], [-max_tilt / 0.6, 0.0, 0.0], acceleration=2.0, duration=0.6
    )
    try:
        robot.stop_velocity()
    except Exception:
        pass

    success = last_fill is not None and last_fill >= fill_frac * 0.85
    _record_tool_use("pour_to_level", success)
    return _result(
        action_type="pour_to_level",
        success=bool(success),
        message=f"final fill={last_fill:.2f} target={fill_frac:.2f}",
        duration=time.time() - t0,
    )


def _measure_fill_fraction(
    pipeline, target_det, fluid_label: str = "water"
) -> Optional[float]:
    """
    Estimate fluid coverage inside target container as [0,1] fraction.
    """
    if pipeline is None or not hasattr(pipeline, "capture"):
        return None
    try:
        caps = pipeline.capture()
    except Exception:
        return None
    primary_keys = ["external", "birdview", "sideview"]
    cap = next(
        (caps[k] for k in primary_keys if k in caps and caps[k] is not None), None
    )
    if cap is None or "rgb" not in cap:
        return None
    perception = getattr(pipeline, "_perception", None)
    if perception is None:
        return None
    try:
        dets = perception.detect(
            rgb=cap["rgb"],
            depth=cap.get("depth"),
            cam_pos=None,
            cam_mat=None,
            cam_fovy=None,
            prompts=[fluid_label],
        )
    except Exception:
        try:
            dets = perception._sam3.detect(cap["rgb"], [fluid_label])
        except Exception:
            return None
    if not dets:
        return 0.0
    fluid_mask_area = max(getattr(d, "mask_area", 0) or 0 for d in dets)
    container_area = max(
        1,
        int(
            target_det.get("mask_area")
            or target_det.get("bbox", [0, 0, 1, 1])[2]
            * target_det.get("bbox", [0, 0, 1, 1])[3]
        ),
    )
    return float(min(1.0, fluid_mask_area / container_area))


@spark_skill(
    name="sweep_to_container",
    description=(
        "With a brush/broom held, sweep clutter items toward a target "
        "container. Iteratively detects remaining items and pushes them; "
        "stops when no clutter remains in workspace OR max_passes reached. "
        "Use AFTER grasp_se3 of the brush and tool_grip_pose alignment."
    ),
    params={
        "container_label": str,
        "clutter_label": str,
        "max_passes": int,
        "sweep_height": float,
    },
)
def sweep_to_container(executor, params: dict):
    t0 = time.time()
    if not getattr(executor, "_holding", False):
        return _result(
            action_type="sweep_to_container",
            success=False,
            message="executor is not holding a brush",
            duration=time.time() - t0,
        )
    container_label = params.get("container_label")
    clutter_label = params.get("clutter_label", "screw")
    max_passes = int(params.get("max_passes", 6))
    sweep_h = float(params.get("sweep_height", 0.005))

    container = executor.detection_map.get(container_label)
    if container is None:
        return _result(
            action_type="sweep_to_container",
            success=False,
            message=f"container {container_label!r} not detected",
            duration=time.time() - t0,
        )
    target = np.asarray(container.get("position_3d"), dtype=float)
    base_ori = np.asarray(executor.GRASP_ORIENTATION, dtype=float)
    n_swept = 0

    pipeline = getattr(executor, "_pipeline", None)
    for pass_idx in range(max_passes):
        if getattr(executor, "_abort", False):
            break
        clutter_dets = _redetect_clutter(
            pipeline, executor.detection_map, label=clutter_label
        )
        if not clutter_dets:
            logger.info(
                "sweep_to_container: no clutter remaining after %d passes", pass_idx
            )
            break
        clutter_dets.sort(
            key=lambda d: -float(
                np.linalg.norm(
                    np.asarray(d.get("position_3d", [0, 0, 0]))[:2] - target[:2]
                )
            )
        )
        item = clutter_dets[0]
        item_xy = np.asarray(item.get("position_3d", target.tolist()))[:2]
        toward = target[:2] - item_xy
        toward_n = _normalize(np.append(toward, 0.0))[:2]
        start_xyz = np.array(
            [
                item_xy[0] - 0.05 * toward_n[0],
                item_xy[1] - 0.05 * toward_n[1],
                target[2] + sweep_h,
            ]
        )
        end_xyz = np.array(
            [
                target[0] - 0.03 * toward_n[0],
                target[1] - 0.03 * toward_n[1],
                target[2] + sweep_h,
            ]
        )
        if not executor._servo.servo_to(np.concatenate([start_xyz, base_ori])):
            logger.warning("sweep pass %d: approach failed", pass_idx)
            continue
        if executor._servo.servo_to(np.concatenate([end_xyz, base_ori])):
            n_swept += 1
        lift_xyz = end_xyz.copy()
        lift_xyz[2] += 0.08
        executor._servo.servo_to(np.concatenate([lift_xyz, base_ori]))

    success = n_swept > 0
    _record_tool_use("sweep_to_container", success)
    return _result(
        action_type="sweep_to_container",
        success=success,
        message=f"{n_swept} pass(es) completed",
        duration=time.time() - t0,
    )


def _redetect_clutter(pipeline, detection_map, label: str) -> list:
    """
    Return detections matching label; falls back to cached map.
    """
    matching = [d for k, d in detection_map.items() if k and label.lower() in k.lower()]
    if matching:
        return matching
    if pipeline is None or not hasattr(pipeline, "capture"):
        return []
    try:
        caps = pipeline.capture()
        rgb = next(
            (
                caps[k]["rgb"]
                for k in ("external", "birdview", "sideview")
                if k in caps and caps[k] is not None
            ),
            None,
        )
        if rgb is None:
            return []
        dets = pipeline._perception.detect(rgb=rgb, prompts=[label])
        return [
            {
                "position_3d": getattr(d, "position_3d", None),
                "mask_area": getattr(d, "mask_area", 0),
                "label": getattr(d, "label", label),
            }
            for d in (dets or [])
        ]
    except Exception:
        return []


@spark_skill(
    name="tool_grip_pose",
    description=(
        "Reorient the wrist so the held tool's working face is flat "
        "against the workpiece's mask normal. Use AFTER a grasp_se3 of "
        "the tool and BEFORE constrained_scrub / wipe."
    ),
    params={"tool_label": str, "workpiece_label": str, "approach_height": float},
)
def tool_grip_pose(executor, params: dict) -> ExecutionResult:
    t0 = time.time()
    workpiece_label = params.get("workpiece_label")
    workpiece = executor.detection_map.get(workpiece_label)
    if workpiece is None:
        return _result(
            action_type="tool_grip_pose",
            success=False,
            message=f"workpiece {workpiece_label!r} not in detection map",
            duration=time.time() - t0,
        )
    approach_h = float(params.get("approach_height", 0.05))
    wp_pos = np.asarray(workpiece.get("position_3d"), dtype=float)
    if wp_pos.size != 3:
        return _result(
            action_type="tool_grip_pose",
            success=False,
            message="workpiece detection missing 3D position",
            duration=time.time() - t0,
        )

    target = np.zeros(6)
    target[:3] = wp_pos + np.array([0.0, 0.0, approach_h])
    angle = float(workpiece.get("orientation_angle", 0.0))
    base = np.asarray(executor.GRASP_ORIENTATION, dtype=float)
    target[3:] = base
    target[5] += angle

    ok = executor._servo.servo_to(target)
    _record_tool_use("tool_grip_pose", ok)
    return _result(
        action_type="tool_grip_pose",
        success=bool(ok),
        message=f"aligned to {workpiece_label} ({math.degrees(angle):.1f} deg)",
        duration=time.time() - t0,
    )


@spark_skill(
    name="constrained_scrub_oscillating",
    description=(
        "Hold a target normal force into a workpiece while oscillating the "
        "EE in the tangent plane (velocity-streaming variant). Requires the "
        "executor to be holding a tool (e.g. sponge, brush) and the driver "
        "to support send_velocity. For the keypoint-based positional scrub "
        "with parametric patterns, prefer 'constrained_scrub'."
    ),
    params={
        "workpiece_label": str,
        "duration": float,
        "amplitude": float,
        "frequency": float,
        "normal_force": float,
        "axis": str,
    },
)
def constrained_scrub_oscillating(executor, params: dict) -> ExecutionResult:
    t0 = time.time()
    if not getattr(executor, "_holding", False):
        return _result(
            action_type="constrained_scrub",
            success=False,
            message="executor is not holding a tool",
            duration=time.time() - t0,
        )
    duration = float(params.get("duration", 5.0))
    amplitude = float(params.get("amplitude", 0.020))
    frequency = float(params.get("frequency", 1.5))
    normal_force = float(params.get("normal_force", 6.0))
    axis = params.get("axis", "xy").lower()

    robot = _get_robot(executor)
    if not hasattr(robot, "send_velocity"):
        return _result(
            action_type="constrained_scrub",
            success=False,
            message="driver lacks send_velocity; cannot run scrub loop",
            duration=time.time() - t0,
        )

    rate_hz = 30.0
    dt = 1.0 / rate_hz
    t_end = time.time() + duration
    step = 0
    omega = 2 * math.pi * frequency
    initial_force_mag = float(np.linalg.norm(robot.get_tcp_force()))
    try:
        while time.time() < t_end and not getattr(executor, "_abort", False):
            elapsed = time.time() - t0
            if axis == "xy":
                vx = amplitude * omega * math.cos(omega * elapsed)
                vy = amplitude * omega * math.sin(omega * elapsed * 1.3)
                lin = np.array([vx, vy, 0.0])
            elif axis == "yz":
                vy = amplitude * omega * math.cos(omega * elapsed)
                vz = amplitude * omega * math.sin(omega * elapsed * 1.3)
                lin = np.array([0.0, vy, vz])
            else:
                lin = np.array(
                    [amplitude * omega * math.cos(omega * elapsed), 0.0, 0.0]
                )
            f = robot.get_tcp_force()
            f_mag = float(np.linalg.norm(f))
            delta = f_mag - initial_force_mag
            kp_force = 0.005
            v_normal = (normal_force - delta) * kp_force
            v_normal = max(-0.05, min(0.05, v_normal))
            lin[2] = -v_normal
            robot.send_velocity(
                lin.tolist(), [0.0, 0.0, 0.0], acceleration=0.5, duration=dt
            )
            step += 1
            time.sleep(dt)
    finally:
        try:
            robot.stop_velocity()
        except Exception:
            pass
    _record_tool_use("constrained_scrub", True)
    return _result(
        action_type="constrained_scrub",
        success=True,
        message=f"scrubbed for {duration:.1f}s ({step} ticks)",
        duration=time.time() - t0,
    )


@spark_skill(
    name="rinse",
    description=(
        "Lower the held tool into a container, agitate briefly, lift back "
        "to safe height. Use for rinsing a sponge / brush between scrub "
        "cycles."
    ),
    params={"container_label": str, "dunk_depth": float, "agitate_duration": float},
)
def rinse(executor, params: dict) -> ExecutionResult:
    t0 = time.time()
    container_label = params.get("container_label")
    container = executor.detection_map.get(container_label)
    if container is None:
        return _result(
            action_type="rinse",
            success=False,
            message=f"container {container_label!r} not detected",
            duration=time.time() - t0,
        )
    dunk_depth = float(params.get("dunk_depth", 0.05))
    agitate_duration = float(params.get("agitate_duration", 1.0))

    pos = np.asarray(container.get("position_3d"), dtype=float)
    if pos.size != 3:
        return _result(
            action_type="rinse",
            success=False,
            message="container missing 3D position",
            duration=time.time() - t0,
        )

    above = pos.copy()
    above[2] += 0.10
    base_ori = np.asarray(executor.GRASP_ORIENTATION, dtype=float)
    target = np.concatenate([above, base_ori])
    if not executor._servo.servo_to(target):
        return _result(
            action_type="rinse",
            success=False,
            message="approach failed",
            duration=time.time() - t0,
        )

    target[2] = pos[2] - dunk_depth
    if not executor._servo.servo_to(target):
        return _result(
            action_type="rinse",
            success=False,
            message="dunk failed",
            duration=time.time() - t0,
        )

    agitate = constrained_scrub_oscillating(
        executor,
        {
            "workpiece_label": container_label,
            "duration": agitate_duration,
            "amplitude": 0.010,
            "frequency": 2.0,
            "normal_force": 2.0,
            "axis": "xy",
        },
    )

    target[2] = pos[2] + 0.10
    executor._servo.servo_to(target)
    ok = bool(agitate.success)
    _record_tool_use("rinse", ok)
    return _result(
        action_type="rinse",
        success=ok,
        message=f"rinsed in {container_label}",
        duration=time.time() - t0,
    )


@spark_skill(
    name="sponge_wash",
    description=(
        "Single-arm dish-wash: pick the sponge by its width, align over "
        "the dish, scrub for `duration` seconds, rinse in the sink. "
        "Use for plates / bowls. For glasses use the bimanual handover."
    ),
    params={
        "sponge_label": str,
        "dish_label": str,
        "sink_label": str,
        "duration": float,
        "normal_force": float,
    },
)
def sponge_wash(executor, params: dict) -> ExecutionResult:
    t0 = time.time()
    duration = float(params.get("duration", 5.0))
    normal_force = float(params.get("normal_force", 6.0))
    sponge_label = params.get("sponge_label", "sponge")
    dish_label = params.get("dish_label", "dish")
    sink_label = params.get("sink_label")

    pick_params = {
        "keypoint_label": sponge_label,
        "force": 12.0,
        "target_width": 0.025,
        "prefer_angled": True,
    }
    pick_result = executor._dispatch_action("grasp_se3", pick_params)
    if not pick_result.success:
        return _result(
            action_type="sponge_wash",
            success=False,
            message=f"sponge pick failed: {pick_result.message}",
            duration=time.time() - t0,
        )

    align = tool_grip_pose(
        executor,
        {
            "tool_label": sponge_label,
            "workpiece_label": dish_label,
            "approach_height": 0.05,
        },
    )
    if not align.success:
        return _result(
            action_type="sponge_wash",
            success=False,
            message=f"alignment failed: {align.message}",
            duration=time.time() - t0,
        )

    scrub = constrained_scrub_oscillating(
        executor,
        {
            "workpiece_label": dish_label,
            "duration": duration,
            "amplitude": 0.020,
            "frequency": 1.5,
            "normal_force": normal_force,
            "axis": "xy",
        },
    )

    rinse_msg = ""
    if sink_label:
        r = rinse(
            executor,
            {
                "container_label": sink_label,
                "dunk_depth": 0.04,
                "agitate_duration": 1.0,
            },
        )
        rinse_msg = r.message

    ok = bool(scrub.success)
    _record_tool_use("sponge_wash", ok)
    return _result(
        action_type="sponge_wash",
        success=ok,
        message=f"scrub:{scrub.message}  rinse:{rinse_msg}",
        duration=time.time() - t0,
    )


@spark_skill(
    name="throw_to",
    description=(
        "Open-loop ballistic toss: open the gripper at a specific release "
        "velocity so the held object lands inside `target_label`. Use for "
        "soft balls / bean bags into bins; not for fragile objects."
    ),
    params={"target_label": str, "release_velocity": float, "release_angle_deg": float},
)
def throw_to(executor, params: dict) -> ExecutionResult:
    t0 = time.time()
    if not getattr(executor, "_holding", False):
        return _result(
            action_type="throw_to",
            success=False,
            message="not holding an object",
            duration=time.time() - t0,
        )
    target_label = params.get("target_label")
    target = executor.detection_map.get(target_label)
    if target is None:
        return _result(
            action_type="throw_to",
            success=False,
            message=f"target {target_label!r} not detected",
            duration=time.time() - t0,
        )
    v = float(params.get("release_velocity", 1.2))
    ang = math.radians(float(params.get("release_angle_deg", 30)))

    target_xyz = np.asarray(target.get("position_3d"), dtype=float)
    if target_xyz.size != 3:
        return _result(
            action_type="throw_to",
            success=False,
            message="target missing 3D position",
            duration=time.time() - t0,
        )
    tcp_xyz = _tcp_xyz(executor)
    delta_xy = target_xyz[:2] - tcp_xyz[:2]
    dir_xy = _normalize(np.append(delta_xy, 0.0))[:2]
    robot = _get_robot(executor)
    if not hasattr(robot, "send_velocity"):
        return _result(
            action_type="throw_to",
            success=False,
            message="driver lacks send_velocity",
            duration=time.time() - t0,
        )
    robot.send_velocity(
        [-dir_xy[0] * 0.3, -dir_xy[1] * 0.3, -0.1],
        [0, 0, 0],
        acceleration=1.0,
        duration=0.20,
    )
    vx = v * math.cos(ang) * dir_xy[0]
    vy = v * math.cos(ang) * dir_xy[1]
    vz = v * math.sin(ang)
    robot.send_velocity([vx, vy, vz], [0, 0, 0], acceleration=2.0, duration=0.30)
    try:
        if hasattr(robot, "open_gripper"):
            robot.open_gripper()
    except Exception:
        pass
    try:
        robot.stop_velocity()
    except Exception:
        pass
    executor._holding = False
    _record_tool_use("throw_to", True)
    return _result(
        action_type="throw_to",
        success=True,
        message=f"thrown toward {target_label}",
        duration=time.time() - t0,
    )


@spark_skill(
    name="compliant_insert",
    description=(
        "F/T-driven insertion of a held object into a target slot/socket. "
        "Maintains a low normal force while descending until either the "
        "force budget is reached (seated) OR the max descent depth is "
        "hit. Use for plug-into-socket, peg-in-hole, USB-into-port; "
        "anywhere a fixed-depth move would jam."
    ),
    params={
        "target_label": str,
        "max_descent_m": float,
        "seat_force_n": float,
        "wiggle_amplitude_m": float,
        "approach_offset_z": float,
    },
)
def compliant_insert(executor, params: dict):
    t0 = time.time()
    if not getattr(executor, "_holding", False):
        return _result(
            action_type="compliant_insert",
            success=False,
            message="not holding an object to insert",
            duration=time.time() - t0,
        )
    target_label = params.get("target_label")
    target = executor.detection_map.get(target_label)
    if target is None:
        return _result(
            action_type="compliant_insert",
            success=False,
            message=f"target {target_label!r} not detected",
            duration=time.time() - t0,
        )
    target_xyz = np.asarray(target.get("position_3d"), dtype=float)
    if target_xyz.size != 3:
        return _result(
            action_type="compliant_insert",
            success=False,
            message="target missing 3D position",
            duration=time.time() - t0,
        )
    max_descent = float(params.get("max_descent_m", 0.04))
    seat_force = float(params.get("seat_force_n", 8.0))
    wiggle_amp = float(params.get("wiggle_amplitude_m", 0.003))
    approach_offset = float(params.get("approach_offset_z", 0.08))

    base_ori = np.asarray(executor.GRASP_ORIENTATION, dtype=float)
    pre_xyz = target_xyz + np.array([0.0, 0.0, approach_offset])
    if not executor._servo.servo_to(np.concatenate([pre_xyz, base_ori])):
        return _result(
            action_type="compliant_insert",
            success=False,
            message="pre-insert approach failed",
            duration=time.time() - t0,
        )

    robot = _get_robot(executor)
    if not hasattr(robot, "send_velocity"):
        return _result(
            action_type="compliant_insert",
            success=False,
            message="driver lacks send_velocity",
            duration=time.time() - t0,
        )

    f0_mag = float(np.linalg.norm(robot.get_tcp_force()))
    dt = 1.0 / 30.0
    descent = 0.0
    seated = False
    elapsed_step = 0
    while descent < max_descent and not getattr(executor, "_abort", False):
        f_now = float(np.linalg.norm(robot.get_tcp_force()))
        delta = f_now - f0_mag
        if delta >= seat_force:
            seated = True
            break
        phase = 2 * math.pi * 2.5 * (elapsed_step * dt)
        vx = wiggle_amp * 2.5 * math.cos(phase)
        vy = wiggle_amp * 2.5 * math.sin(phase * 1.3)
        vz = -0.01 if delta < seat_force * 0.5 else -0.004
        robot.send_velocity(
            [vx, vy, vz], [0.0, 0.0, 0.0], acceleration=0.5, duration=dt
        )
        descent += abs(vz) * dt
        elapsed_step += 1
        time.sleep(dt)
    try:
        robot.stop_velocity()
    except Exception:
        pass

    _record_tool_use("compliant_insert", seated)
    return _result(
        action_type="compliant_insert",
        success=seated,
        message=(
            "seated" if seated else f"max descent reached ({descent*1000:.1f} mm)"
        ),
        duration=time.time() - t0,
    )


@spark_skill(
    name="pen_write",
    description=(
        "With a pen / sharpie held, trace a path of (dx, dy) deltas on "
        "the page. Maintains a constant downward force so the line is "
        "continuous. Use after grasp_se3 of the pen."
    ),
    params={"path": list, "normal_force": float, "speed": float},
)
def pen_write(executor, params: dict) -> ExecutionResult:
    t0 = time.time()
    if not getattr(executor, "_holding", False):
        return _result(
            action_type="pen_write",
            success=False,
            message="executor is not holding a pen",
            duration=time.time() - t0,
        )
    path = params.get("path") or []
    if not path:
        return _result(
            action_type="pen_write",
            success=False,
            message="empty path",
            duration=time.time() - t0,
        )
    speed = float(params.get("speed", 0.05))
    normal_force = float(params.get("normal_force", 4.0))

    robot = _get_robot(executor)
    if not hasattr(robot, "send_velocity"):
        return _result(
            action_type="pen_write",
            success=False,
            message="driver lacks send_velocity; cannot stream pen path",
            duration=time.time() - t0,
        )

    f0_mag = float(np.linalg.norm(robot.get_tcp_force()))
    dt = 1.0 / 30.0
    for _ in range(120):
        f_now = float(np.linalg.norm(robot.get_tcp_force()))
        if f_now - f0_mag >= normal_force:
            break
        robot.send_velocity(
            [0.0, 0.0, -0.02], [0.0, 0.0, 0.0], acceleration=0.5, duration=dt
        )
        time.sleep(dt)
    kp = 0.005
    try:
        for dx, dy in path:
            distance = math.hypot(dx, dy)
            if distance < 1e-6:
                continue
            steps = max(1, int(distance / (speed * dt)))
            vx = dx / (steps * dt)
            vy = dy / (steps * dt)
            for _ in range(steps):
                if getattr(executor, "_abort", False):
                    break
                f_now = float(np.linalg.norm(robot.get_tcp_force()))
                vz = -kp * (normal_force - (f_now - f0_mag))
                vz = max(-0.03, min(0.03, vz))
                robot.send_velocity(
                    [vx, vy, vz], [0.0, 0.0, 0.0], acceleration=0.5, duration=dt
                )
                time.sleep(dt)
    finally:
        try:
            robot.stop_velocity()
        except Exception:
            pass

    for _ in range(15):
        robot.send_velocity(
            [0.0, 0.0, 0.05], [0.0, 0.0, 0.0], acceleration=0.5, duration=dt
        )
        time.sleep(dt)
    robot.stop_velocity()

    _record_tool_use("pen_write", True)
    return _result(
        action_type="pen_write",
        success=True,
        message=f"wrote {len(path)} segments",
        duration=time.time() - t0,
    )

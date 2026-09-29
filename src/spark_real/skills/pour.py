# Vision-terminated liquid pour from a held source container.

import logging
import time
import numpy as np
from scipy.spatial.transform import Rotation as _R
from spark_real.skills.registry import spark_skill
from spark_real.skills.primitives import _result
from spark_real.perception.water_level import WaterLevelTracker

logger = logging.getLogger(__name__)


@spark_skill(
    name="pour",
    description=(
        "Pour liquid from an already-held source container into a target "
        "container, terminating on a vision-based fill predicate or after "
        "max_pour_s."
    ),
    params={
        "source_label": str,
        "target_label": str,
        "pour_angle_rad": float,
        "target_fill_fraction": float,
        "max_pour_s": float,
    },
)
def pour(executor, params: dict):
    """
    Tilt-pour primitive with vision-based fill termination.
    """

    t0 = time.time()
    source_label = params.get("source_label", "")
    target_label = params.get("target_label", "")
    # Paper appendix default (90 deg); runtime clamps to the 0.6 rad pour limit below.
    pour_angle_rad = float(params.get("pour_angle_rad", 1.5708))
    target_fill_fraction = float(params.get("target_fill_fraction", 0.7))
    max_pour_s = float(params.get("max_pour_s", 15.0))

    if not target_label:
        return _result(
            "pour", False, "pour: target_label is required", time.time() - t0
        )

    target_det = executor.detection_map.get(target_label)
    if target_det is None:
        return _result(
            "pour",
            False,
            f"pour: target '{target_label}' not in detection_map",
            time.time() - t0,
        )
    target_pos = target_det.get("position_3d")
    if target_pos is None:
        return _result(
            "pour",
            False,
            f"pour: target '{target_label}' has no position_3d",
            time.time() - t0,
        )
    target_pos = np.asarray(target_pos, dtype=float)[:3]

    if not hasattr(executor.robot, "get_tcp_pose"):
        return _result(
            "pour", False, "pour: robot driver lacks get_tcp_pose()", time.time() - t0
        )
    pose_now = np.asarray(executor.robot.get_tcp_pose(), dtype=float)
    if pose_now.size < 6:
        return _result(
            "pour",
            False,
            f"pour: get_tcp_pose returned size {pose_now.size}",
            time.time() - t0,
        )
    pos_now = pose_now[:3].copy()
    rotvec_now = pose_now[3:6].copy()

    # Handle-aware pour geometry: the pour is a pitch that tips the rim opposite
    # the handle down, so the hover parks the gripper offset from the bowl along
    # the handle direction (far rim hangs over the bowl). Falls back to a
    # bowl-direction tilt when no handle detection exists.
    handle_dir = None
    _hdet = executor.detection_map.get(f"{source_label} handle")
    _sdet = executor.detection_map.get(source_label)
    if _hdet is not None and _sdet is not None:
        try:
            _hp = np.asarray(_hdet.get("position_3d"), dtype=float)[:2]
            _sp = np.asarray(_sdet.get("position_3d"), dtype=float)[:2]
            _v = _hp - _sp
            _n = float(np.linalg.norm(_v))
            if _n > 0.02:
                handle_dir = _v / _n
        except Exception:
            handle_dir = None

    # Pre-pour hover above target rim
    if handle_dir is not None:
        hover_pos = np.array(
            [
                target_pos[0] + 0.10 * handle_dir[0],
                target_pos[1] + 0.10 * handle_dir[1],
                target_pos[2] + 0.07,
            ],
            dtype=float,
        )
    else:
        hover_pos = np.array(
            [target_pos[0] + 0.03, target_pos[1], target_pos[2] + 0.07], dtype=float
        )

    logger.info(
        "[pour] source='%s' target='%s' hover=(%.3f,%.3f,%.3f)",
        source_label,
        target_label,
        *hover_pos,
    )

    try:
        executor._move_to(hover_pos, rotvec_now.tolist())
    except Exception as exc:
        return _result("pour", False, f"pour: hover failed: {exc}", time.time() - t0)

    # Start water level tracker
    tracker = None
    tracker_ok = False
    try:
        tracker = WaterLevelTracker(
            pipeline=getattr(executor, "_pipeline", None), target_label=target_label
        )
        start_info = tracker.start(target_fill_fraction=target_fill_fraction)
        tracker_ok = bool(start_info.get("ok"))
        if not tracker_ok:
            logger.warning(
                "[pour] tracker failed (%s); open-loop pour",
                start_info.get("reason", "unknown"),
            )
    except Exception as exc:
        logger.warning("[pour] tracker init raised (%s); open-loop pour", exc)

    # Build tilt pose: apply tilt in the WORLD frame on top of the current
    # gripper orientation so the pour is always a pitch (rim tips down toward
    # the target) regardless of the gripper's yaw. Composing with R_home (0-deg
    # yaw) instead would force a yaw change and produce a roll at 90-deg yaw.
    try:
        R_current = _R.from_rotvec(rotvec_now)
        if handle_dir is not None:
            # Tip the rim opposite the handle down; tilting about the handle
            # bar's own axis would look like a roll and could flip the wrist.
            pour_dir = np.array([-handle_dir[0], -handle_dir[1], 0.0])
        else:
            # Fallback: from source toward target (in XY)
            pour_dir = np.array(
                [target_pos[0] - pos_now[0], target_pos[1] - pos_now[1], 0.0]
            )
        pnorm = float(np.linalg.norm(pour_dir))
        pour_dir = pour_dir / pnorm if pnorm >= 1e-6 else np.array([1.0, 0.0, 0.0])
        # Tilt axis cross(Z_up, pour_dir) so a positive angle tips the rim
        # (which faces along pour_dir) downward.
        tilt_axis = np.cross([0.0, 0.0, 1.0], pour_dir)
        tilt_axis = tilt_axis / (float(np.linalg.norm(tilt_axis)) + 1e-12)
        # Clamp to 0.6 rad; larger orientation jumps can switch IK branches
        # and flip the wrist.
        if pour_angle_rad > 0.6:
            logger.info("[pour] clamping pour_angle %.2f -> 0.60 rad", pour_angle_rad)
            pour_angle_rad = 0.6
        R_tilted = _R.from_rotvec(tilt_axis * pour_angle_rad) * R_current
        rotvec_tilted = R_tilted.as_rotvec().tolist()
        # Incremental tilt waypoints keep the IK in one joint branch; a single
        # large orientation jump flips J5/J7.
        n_steps = max(1, int(np.ceil(pour_angle_rad / 0.3)))
        tilt_steps = [
            (
                _R.from_rotvec(tilt_axis * (pour_angle_rad * (si + 1) / n_steps))
                * R_current
            )
            .as_rotvec()
            .tolist()
            for si in range(n_steps)
        ]
    except Exception as exc:
        if tracker:
            try:
                tracker.stop()
            except Exception:
                pass
        return _result(
            "pour", False, f"pour: tilt computation failed: {exc}", time.time() - t0
        )

    logger.info(
        "[pour] tilting by %.2f rad in %d step(s)", pour_angle_rad, len(tilt_steps)
    )
    try:
        for _rv in tilt_steps:
            executor._move_to(hover_pos, _rv)
    except Exception as exc:
        if tracker:
            try:
                tracker.stop()
            except Exception:
                pass
        return _result("pour", False, f"pour: tilt failed: {exc}", time.time() - t0)

    # Poll fill predicate
    poll_dt = 0.2
    pour_start = time.time()
    final_fill = float("nan")
    stop_reason = "timeout"
    while True:
        if time.time() - pour_start >= max_pour_s:
            break
        if tracker and tracker_ok:
            try:
                state = tracker.update()
            except Exception:
                state = None
            if state and state.get("ok"):
                final_fill = float(state.get("fill_fraction", final_fill))
                if state.get("should_stop"):
                    stop_reason = f"fill_target_reached (fill={final_fill:.2f})"
                    break
        time.sleep(poll_dt)

    logger.info(
        "[pour] ended after %.1fs: %s (fill=%.2f)",
        time.time() - pour_start,
        stop_reason,
        final_fill,
    )

    # Return to upright
    try:
        executor._move_to(hover_pos, rotvec_now.tolist())
    except Exception as exc:
        logger.warning("[pour] upright recovery failed: %s", exc)

    if tracker:
        try:
            tracker.stop()
        except Exception:
            pass

    # Step away
    try:
        away = hover_pos.copy()
        away[0] -= 0.05
        executor._move_to(away.tolist(), rotvec_now.tolist())
    except Exception:
        pass

    return _result(
        "pour",
        True,
        f"pour: source='{source_label}' target='{target_label}' "
        f"stop={stop_reason} fill={final_fill:.2f}",
        time.time() - t0,
    )

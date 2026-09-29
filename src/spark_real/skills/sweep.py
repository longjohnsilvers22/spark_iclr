# Sweep debris with a held tool along an IK waypoint chain.

import logging
import time
import numpy as np
from scipy.spatial.transform import Rotation as _R
from spark_real.skills.registry import spark_skill
from spark_real.skills.primitives import _result

logger = logging.getLogger(__name__)

# pyroki/JAX IK; fall back to executor._move_to when unavailable.
try:
    from spark_real.control.fr3_ik_pyroki import solve_ik as _sweep_solve_ik
except ImportError:
    _sweep_solve_ik = None


def _sweep_get_field(det, key, default=None):
    """
    Read a field from a detection dict or ObjectDetection dataclass.
    """
    if hasattr(det, "get"):
        v = det.get(key, default)
    else:
        v = getattr(det, key, default)
    return default if v is None else v


def _sweep_ik_chain(executor, waypoints, orient, velocity=0.08):
    """Pre-compute IK for all waypoints with seed chaining, dry-run verify,
    then execute via move_to_joint_config (Bamboo torque-control path).

    Falls back to executor._move_to per-waypoint if IK precompute is
    unavailable.

    Returns (n_executed, error_msg | None).
    """
    # orient: a single rotvec applied to all waypoints, OR a list of rotvecs
    # (one per waypoint) for stroke-synchronized wrist motion like the
    # bristle roll-flick.
    orients = (
        orient
        if isinstance(orient, list)
        and orient
        and isinstance(orient[0], (list, np.ndarray))
        else [orient] * len(waypoints)
    )
    if _sweep_solve_ik is None or not hasattr(executor.robot, "move_to_joint_config"):
        for i, wp in enumerate(waypoints):  # fallback: execute each waypoint directly
            executor._move_to(wp.tolist(), orients[i], velocity=velocity)
        return len(waypoints), None

    # Read current joint config as first seed
    try:
        q_seed = np.array(executor.robot.get_joint_positions(), dtype=float)
    except Exception:
        # IK seed only (not the ready pose): rounded J5=0 variant, distinct
        # from franka_base.HOME_CONFIG / franka_default.yaml home_config.
        q_seed = np.array([0, -0.785, 0, -2.356, 0, 1.571, 0.785])

    # Pre-compute all IK solutions with seed chaining
    q_solutions = []
    for i, wp in enumerate(waypoints):
        q_sol = _sweep_solve_ik(np.asarray(wp, dtype=float), orients[i], q_seed=q_seed)
        if q_sol is None:
            logger.warning(
                "sweep: IK failed at waypoint %d/%d " "pos=(%.3f,%.3f,%.3f)",
                i,
                len(waypoints),
                wp[0],
                wp[1],
                wp[2],
            )
            return i, f"IK failed at waypoint {i}"
        q_solutions.append(q_sol)
        q_seed = q_sol  # chain for continuity

    # Dry-run: check for large joint jumps (arm flips). Execute the safe
    # prefix up to the first flip rather than refusing the whole pass, since
    # a late-stroke flip would otherwise abort the entire pass.
    n_safe = len(q_solutions)
    for i in range(1, len(q_solutions)):
        dq = np.abs(q_solutions[i] - q_solutions[i - 1])
        if dq.max() > np.deg2rad(45):
            logger.warning(
                "sweep: arm flip between wp %d->%d "
                "(dq_max=%.1f deg at j%d); executing safe "
                "prefix only",
                i - 1,
                i,
                np.rad2deg(dq.max()),
                dq.argmax(),
            )
            n_safe = i
            break
    if n_safe < 2:
        return 0, "arm flip at waypoint 1"
    q_solutions = q_solutions[:n_safe]

    # Execute all waypoints
    for i, q_target in enumerate(q_solutions):
        executor._check_abort()
        executor.robot.move_to_joint_config(q_target.tolist(), velocity=velocity)
    return len(q_solutions), None


@spark_skill(
    name="sweep",
    description=(
        "With a brush already grasped, sweep scattered items toward "
        "container 'target_label'. Uses Bamboo torque-control with "
        "pre-computed IK + seed chaining for smooth, reflex-free sweeps."
    ),
    params={
        "target_label": str,
        "object_labels": str,
        "n_passes": int,
        "sweep_height_m": float,
        "waypoint_spacing_m": float,
        "sweep_velocity": float,
        "re_detect": bool,
        "dir_x": float,
        "dir_y": float,
        "roll_amp_deg": float,
        "roll_period_wp": int,
    },
)
def sweep(executor, params: dict):
    """
    Geometric multi-pass sweep into a container.

    Design constraints:
    - Constant Z throughout the sweep (no dipping/rising).
    - Fine waypoints every 25 mm to prevent IK arm flips.
    - Path goes THROUGH each object position toward the dustpan.
    - Bamboo move_to_joint_config with pyroki IK and seed chaining.
    - Post-sweep re-detection to check for remaining objects.
    """
    t0 = time.time()
    target_label = params.get("target_label", "")
    # object_labels: comma-separated list of individual objects to sweep, OR a
    # single cluster label (e.g. "crumbs"). When individual objects are listed
    # the sweep path visits each one.
    object_labels_raw = params.get("object_labels", "") or ""
    # Accept area_label as fallback for old plans
    if not object_labels_raw:
        object_labels_raw = params.get("area_label", "")
    n_passes = max(1, int(params.get("n_passes", 2) or 2))
    sweep_height_m = float(params.get("sweep_height_m", 0.04) or 0.04)
    wp_spacing = float(params.get("waypoint_spacing_m", 0.025) or 0.025)
    sweep_vel = float(params.get("sweep_velocity", 0.08) or 0.08)
    do_re_detect = bool(params.get("re_detect", True))

    # Validate grasp
    if not getattr(executor, "_holding", False):
        return _result(
            "sweep", False, "sweep: not holding a brush/tool", time.time() - t0
        )

    # Check gripper width: if fully closed (< 1 mm), the grasp missed
    gripper_width = None
    try:
        obs = executor.robot.get_observation()
        gripper_width = obs.get("gripper_width")
        if gripper_width is None and hasattr(executor.robot, "get_gripper_width"):
            gripper_width = executor.robot.get_gripper_width()
    except Exception:
        pass
    if gripper_width is not None and float(gripper_width) < 0.001:
        executor._holding = False
        return _result(
            "sweep",
            False,
            f"sweep: gripper closed fully "
            f"(width={gripper_width:.4f}m): brush grasp missed",
            time.time() - t0,
        )

    # Resolve detections
    target = executor.detection_map.get(target_label)
    if target is None:
        return _result(
            "sweep",
            False,
            f"sweep: target '{target_label}' not found",
            time.time() - t0,
        )
    target_pos = np.asarray(_sweep_get_field(target, "position_3d"), dtype=float)

    # Parse individual object labels
    obj_labels = [s.strip() for s in object_labels_raw.split(",") if s.strip()]
    obj_positions = []
    for lbl in obj_labels:
        det = executor.detection_map.get(lbl)
        if det is not None:
            pos = _sweep_get_field(det, "position_3d")
            if pos is not None:
                obj_positions.append((lbl, np.asarray(pos, dtype=float)))
            else:
                logger.warning("sweep: '%s' has no position_3d, skipping", lbl)
        else:
            logger.warning("sweep: object '%s' not in detection_map", lbl)

    if not obj_positions:
        # Fallback: treat the first label as a cluster area detection
        fallback_lbl = obj_labels[0] if obj_labels else ""
        area = executor.detection_map.get(fallback_lbl)
        if area is None:
            return _result(
                "sweep",
                False,
                f"sweep: no objects found for " f"'{object_labels_raw}'",
                time.time() - t0,
            )
        area_pos = np.asarray(_sweep_get_field(area, "position_3d"), dtype=float)
        obj_positions = [(fallback_lbl, area_pos)]

    # Compute sweep geometry. Direction: mean-of-objects -> target (dustpan)
    mean_obj_xy = np.mean([p[:2] for _, p in obj_positions], axis=0)
    target_xy = target_pos[:2]

    delta_xy = target_xy - mean_obj_xy
    delta_norm = float(np.linalg.norm(delta_xy))
    if delta_norm < 1e-3:
        return _result(
            "sweep", False, "sweep: objects and target co-located", time.time() - t0
        )
    sweep_dir = delta_xy / delta_norm
    # Optional explicit direction override (e.g. dir_y=1.0 for a pure +Y push).
    # The cluster-mean->target direction turns diagonal when one object sits
    # far off-axis; an explicit dir_x/dir_y override forces the push axis.
    _dx, _dy = params.get("dir_x", None), params.get("dir_y", None)
    dir_overridden = _dx is not None or _dy is not None
    if dir_overridden:
        sweep_dir = np.array([float(_dx or 0.0), float(_dy or 0.0)])
        _n = float(np.linalg.norm(sweep_dir))
        if _n < 1e-6:
            return _result("sweep", False, "sweep: zero dir override", time.time() - t0)
        sweep_dir = sweep_dir / _n
    perp_dir = np.array([-sweep_dir[1], sweep_dir[0]])

    # Table Z and sweep Z (constant throughout)
    table_z = float(getattr(executor, "TABLE_Z_FLOOR", -0.04))
    sweep_z = table_z + sweep_height_m
    safe_z = sweep_z + 0.15

    # Orientation: align brush bristles with sweep direction.
    # Yaw = angle of sweep_dir from +X.
    sweep_yaw = float(np.arctan2(sweep_dir[1], sweep_dir[0]))
    # Bristles broadside to travel. Stack convention: wrist yaw equals the
    # held object's major-axis angle, so the wrist sits at sweep_yaw - 90 deg
    # for the brush's long axis to stay perpendicular to the push (wrist yaw=0
    # sweeps a +Y push correctly). Rz(sweep_yaw) would sweep edge-on.
    wrist_yaw = sweep_yaw - np.pi / 2
    if wrist_yaw > np.pi / 2:
        wrist_yaw -= np.pi
    elif wrist_yaw < -np.pi / 2:
        wrist_yaw += np.pi
    if hasattr(executor, "_nearest_symmetric_yaw"):
        wrist_yaw = executor._nearest_symmetric_yaw(wrist_yaw)
    base_R = _R.from_rotvec(executor.GRASP_ORIENTATION)
    sweep_R = _R.from_euler("z", wrist_yaw) * base_R
    orient = sweep_R.as_rotvec().tolist()

    # Pre-sweep: reorient at safe height
    logger.info(
        "[sweep] reorienting at safe z=%.3f, yaw=%.1f deg",
        safe_z,
        np.rad2deg(sweep_yaw),
    )
    current_pos = executor._get_current_position()
    executor._move_to([current_pos[0], current_pos[1], safe_z], orient)

    # Start position: behind objects (opposite to sweep dir)
    # Project each object along -sweep_dir from the target to find the
    # one that is farthest back.
    obj_projs = [np.dot(p[:2] - target_xy, -sweep_dir) for _, p in obj_positions]
    farthest_back_dist = max(obj_projs) if obj_projs else 0.10
    start_behind_m = 0.08  # behind the farthest object

    # Lateral spread determines pass width
    obj_perp = [np.dot(p[:2] - mean_obj_xy, perp_dir) for _, p in obj_positions]
    spread_half = max(
        abs(min(obj_perp, default=0)), abs(max(obj_perp, default=0)), 0.02
    )
    spread_half += 0.03  # margin so edge objects get fully swept

    # End position: short of the dustpan center. Reaching the center drives
    # the chain into a configuration boundary (J4 flip) at the far end of the
    # stroke; objects still cross the lip without entering the flip zone.
    end_xy = target_xy - 0.04 * sweep_dir

    # Execute sweep passes
    passes_completed = 0
    total_wp_executed = 0
    try:
        for pass_idx in range(n_passes):
            executor._check_abort()
            # Lateral offset for this pass
            if n_passes == 1:
                lat_offset = 0.0
            else:
                lat_offset = -spread_half + 2.0 * spread_half * pass_idx / (
                    n_passes - 1
                )
            perp_offset = lat_offset * perp_dir

            # Start: behind objects
            start_xy = (
                mean_obj_xy
                - (farthest_back_dist + start_behind_m) * sweep_dir
                + perp_offset
            )

            # All passes end on the target line (no lateral offset) so each
            # stroke herds its side of the cluster into the pan; a
            # parallel-ends variant drifts strokes off toward the table edge.
            pass_end_xy = end_xy

            # Build fine waypoints at constant sweep_z
            path_vec = pass_end_xy - start_xy
            path_len = float(np.linalg.norm(path_vec))
            n_wp = max(2, int(np.ceil(path_len / wp_spacing)) + 1)

            waypoints = []
            for wi in range(n_wp):
                t_frac = wi / max(n_wp - 1, 1)
                xy = start_xy + t_frac * path_vec
                waypoints.append(np.array([xy[0], xy[1], sweep_z]))

            # Optional bristle roll-flick: alternate a roll about the
            # bristle line (perpendicular to travel) every half period,
            # like a hand-broom scrub. roll_amp_deg=0 disables.
            roll_amp = np.deg2rad(float(params.get("roll_amp_deg", 0.0) or 0.0))
            stroke_orients = orient
            if roll_amp > 1e-3:
                period = max(2, int(params.get("roll_period_wp", 4) or 4))
                half = max(1, period // 2)
                axis3 = np.append(perp_dir, 0.0)
                stroke_orients = []
                for wi in range(n_wp):
                    sgn = 1.0 if (wi // half) % 2 == 0 else -1.0
                    R_wi = _R.from_rotvec(axis3 * (sgn * roll_amp)) * sweep_R
                    stroke_orients.append(R_wi.as_rotvec().tolist())

            start_high = np.array([start_xy[0], start_xy[1], safe_z])

            logger.info(
                "[sweep] pass %d/%d: %d wp, lat=%.3f m",
                pass_idx + 1,
                n_passes,
                len(waypoints),
                lat_offset,
            )

            # Move to start at safe height then descend
            executor._move_to(start_high.tolist(), orient)
            executor._move_to(waypoints[0].tolist(), orient, velocity=sweep_vel)

            # Sweep stroke with IK chain
            n_exec, err = _sweep_ik_chain(
                executor, waypoints, stroke_orients, velocity=sweep_vel
            )
            total_wp_executed += n_exec
            if err is not None:
                logger.warning(
                    "[sweep] pass %d aborted: %s (%d/%d wp)",
                    pass_idx + 1,
                    err,
                    n_exec,
                    len(waypoints),
                )
                # Lift and try next pass
                try:
                    cur = executor._get_current_position()
                    executor._move_to([cur[0], cur[1], safe_z], orient)
                except Exception:
                    pass
                continue

            passes_completed += 1

            # Lift after pass
            try:
                cur = executor._get_current_position()
                executor._move_to([cur[0], cur[1], safe_z], orient)
            except Exception as exc:
                logger.warning("[sweep] post-pass lift failed: %s", exc)

    except Exception as exc:
        return _result(
            "sweep",
            False,
            f"sweep failed at pass {passes_completed + 1}/{n_passes}: " f"{exc}",
            time.time() - t0,
        )

    # Post-sweep: re-detect and report remaining
    remaining_labels = []
    if (
        do_re_detect
        and hasattr(executor, "_pipeline")
        and executor._pipeline is not None
    ):
        try:
            pipeline = executor._pipeline
            re_prompts = [lbl for lbl, _ in obj_positions]
            re_prompts.append(target_label)
            logger.info("[sweep] re-detecting: %s", re_prompts)
            captures = pipeline.capture()
            re_dets = pipeline.detect(captures, re_prompts, multi_instance=True)
            re_dets = pipeline.merge_detections(re_dets)

            for rd in re_dets:
                lbl = _sweep_get_field(rd, "label", "")
                if lbl == target_label:
                    continue
                pos = _sweep_get_field(rd, "position_3d")
                if pos is None:
                    continue
                pos = np.asarray(pos, dtype=float)
                dist_to_target = float(np.linalg.norm(pos[:2] - target_xy))
                if dist_to_target > 0.06:
                    remaining_labels.append(lbl)
                    logger.info(
                        "[sweep] remaining: '%s' " "(%.3f m from target)",
                        lbl,
                        dist_to_target,
                    )
        except Exception as exc:
            logger.warning("[sweep] re-detection failed: %s", exc)

    # Retract
    try:
        cur = executor._get_current_position()
        executor._move_to([cur[0], cur[1], safe_z], orient)
    except Exception as exc:
        logger.warning("[sweep] final retract failed: %s", exc)

    msg = (
        f"Swept toward '{target_label}' "
        f"({passes_completed}/{n_passes} passes, "
        f"{total_wp_executed} waypoints)"
    )
    if remaining_labels:
        msg += f"; remaining: {remaining_labels}"

    return _result("sweep", passes_completed > 0, msg, time.time() - t0)

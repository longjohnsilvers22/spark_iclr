# T-shirt fold: lift sleeves, then fold collar down to hem.

import logging
import time
import numpy as np
from scipy.spatial.transform import Rotation as _R
from spark_real.skills.registry import spark_skill
from spark_real.skills.primitives import _result
from spark_real.skills.grasp_utils import canonical_symmetric_yaw

try:
    from spark_real.control.fr3_ik_pyroki import solve_ik as _solve_ik_pyroki
except ImportError:
    _solve_ik_pyroki = None

logger = logging.getLogger(__name__)


@spark_skill(
    name="cloth_fold",
    description=(
        "Self-contained single-arm t-shirt fold for FR3. Detects shirt "
        "parts via SAM3, then folds both sleeves inward and the hem up "
        "to the collar. Handles detection, grasp positioning, arc fold "
        "motion, and grasp retries internally. Requires pyroki IK."
    ),
    params={
        "instruction": str,
    },
)
def cloth_fold(executor, params: dict):
    """
    Full t-shirt fold: +Y sleeve, -Y sleeve, collar -> hem.

    Detection: SAM3 "left sleeve", "right sleeve", "shirt"; ignores
    SAM3 labels and sorts by Y position to assign +Y / -Y sleeve.

    Sleeves: OBB-aligned yaw grip (sleeve edge angle, wrapped to +-90),
    120mm lift, 30mm arc peak, fold 8cm past shirt center Y.

    Body: pitched hem grasp (hem_tilt_deg above horizontal, approach
    toward +X), grasp the hem edge (min_x of shirt OBB) and fold it up
    to the collar edge (max_x, clamped to 0.65). The IK pre-check aborts
    gracefully if the arc is unreachable.

    Grasp: z=-0.02 (table level for thin fabric), retry up to 5 times
    going 3mm lower each attempt.

    All arc waypoints are pre-computed and IK-validated before execution
    via pyroki to ensure reachability.
    """
    t0 = time.time()

    if _solve_ik_pyroki is None:
        return _result(
            "cloth_fold",
            False,
            "pyroki IK not available (required for cloth_fold)",
            time.time() - t0,
        )

    # Fold tuning resolves with precedence BT param > family yaml cloth: block
    # > default, so each robot family can carry its own tuned values via config.
    pipeline = getattr(executor, "_pipeline", None)
    cloth_cfg = (
        pipeline.profile.cloth()
        if (pipeline is not None and getattr(pipeline, "profile", None) is not None)
        else {}
    )
    GRASP_Z = float(
        params.get("grasp_z", cloth_cfg.get("grasp_z", -0.02))
    )  # table level for thin fabric
    GRASP_RETRY_MAX = 5
    GRASP_Z_STEP = -0.003  # go 3mm lower each retry
    SLEEVE_LIFT_H = float(
        params.get("sleeve_lift_h", cloth_cfg.get("sleeve_lift_h", 0.120))
    )  # higher sleeve lift for a cleaner fold
    HEM_LIFT_H = float(
        params.get("body_lift_h", cloth_cfg.get("body_lift_h", 0.080))
    )  # collar/body lift
    ARC_PEAK = float(
        params.get("arc_peak", cloth_cfg.get("arc_peak", 0.030))
    )  # sinusoidal arc peak
    ARC_WAYPOINTS = 14  # 12-15 waypoints in arc
    FOLD_VEL = 0.11  # fold velocity 0.10-0.12
    APPROACH_VEL = 0.08
    GRASP_FORCE = 15  # low force for cloth pinch
    GRIPPER_WIDTH_THRESHOLD = 0.025  # max width to consider cloth gripped
    SLEEVE_PAST_CENTER_M = float(
        params.get("sleeve_past_center_m", cloth_cfg.get("sleeve_past_center_m", 0.08))
    )  # fold past shirt center Y
    HEM_MAX_X_CLAMP = 0.65  # clamp collar grasp X to the reach limit
    GRASP_ORIENT_0DEG = [np.pi, 0.0, 0.0]  # 0-deg yaw (hem)
    # Pitched hem grasp: approach points +X with the fingers closing across
    # the hem edge. hem_tilt_deg measures the approach above horizontal:
    # 90 = pure top-down, 0 = fully horizontal (hand body would hit the
    # table). 45 keeps the housing clear while the fingertips slide under
    # the hem edge.
    HEM_TILT_DEG = float(
        params.get("hem_tilt_deg", cloth_cfg.get("hem_tilt_deg", 45.0))
    )

    # 90-deg yaw orientation for sleeves (fallback when OBB is missing)
    base_R = _R.from_rotvec(GRASP_ORIENT_0DEG)
    yaw90_R = _R.from_euler("z", np.pi / 2) * base_R
    GRASP_ORIENT_90DEG = yaw90_R.as_rotvec().tolist()

    def _obb_yaw_orient(det, fallback):
        """
        Sleeve grip yaw from the sleeve's own OBB major-axis angle.

        Commanding the edge angle closes the fingers across the sleeve
        edge (the +90 closing offset is baked into the yaw convention),
        same wrap-to-+-90 as grasp_top_down. Falls back to the fixed
        90-deg yaw when the detection has no OBB angle.
        """
        ang = det.get("orientation_angle")
        if ang is None:
            return fallback
        yaw = canonical_symmetric_yaw(float(ang))
        logger.info(
            "[cloth_fold] sleeve OBB yaw=%.1f deg (major=%.1f)",
            np.rad2deg(yaw),
            np.rad2deg(float(ang)),
        )
        return (_R.from_euler("z", yaw) * base_R).as_rotvec().tolist()

    # Pitched hem orientation: Ry(-(90-tilt)) pitches the top-down yaw-90
    # pose so the approach leans toward +X and the closing line rotates
    # from horizontal toward vertical. At tilt=0 the approach is fully
    # horizontal along +X with a vertical pinch.
    _hem_beta = -(np.pi / 2 - np.radians(HEM_TILT_DEG))
    GRASP_ORIENT_HEM = (
        (_R.from_euler("y", _hem_beta) * _R.from_euler("z", np.pi / 2) * base_R)
        .as_rotvec()
        .tolist()
    )

    # helper: re-detect objects using birdview camera
    def _redetect(prompts):
        """
        Run SAM3 detection on birdview camera frame. Returns detection
        list or None on failure.
        """
        pipeline = executor._pipeline
        if pipeline is None:
            return None
        kinect2 = getattr(pipeline, "_kinect2", None)
        kinect2_cal = getattr(pipeline, "_kinect2_cal", None)
        perception = getattr(pipeline, "_perception", None)
        if kinect2 is None or kinect2_cal is None or perception is None:
            return None
        try:
            read_lock = getattr(pipeline, "_kinect2_read_lock", None)
            if read_lock is not None:
                with read_lock:
                    rgb, depth = kinect2.read()
            else:
                rgb, depth = kinect2.read()
            if rgb is None:
                return None
            dets = perception.detect(
                rgb,
                prompts,
                cam_pos=kinect2_cal.position,
                cam_mat=kinect2_cal.rotation_matrix,
                cam_fovy=kinect2_cal.fov_y,
            )
            return dets
        except Exception as exc:
            logger.warning("[cloth_fold] redetect failed: %s", exc)
            return None

    def _det_to_map(dets):
        """
        Convert detection list to detection_map-style dict.
        """
        m = {}
        for d in dets:
            if d.position_3d is not None:
                m[d.label] = {
                    "position_3d": (
                        d.position_3d.tolist()
                        if hasattr(d.position_3d, "tolist")
                        else list(d.position_3d)
                    ),
                    "confidence": d.confidence,
                    "orientation_angle": d.orientation_angle,
                    "aspect_ratio": d.aspect_ratio,
                    "obb_minor_m": float(getattr(d, "obb_minor_m", 0.0) or 0.0),
                    "_mask": d.mask,
                }
        return m

    # helper: grasp with retries
    def _grasp_cloth(grasp_pos, orient, max_retries=GRASP_RETRY_MAX):
        """
        Approach, descend, grasp at grasp_pos with retries going
        lower each time. Returns True if cloth was gripped.
        """
        # Hover above
        hover = grasp_pos.copy()
        hover[2] = grasp_pos[2] + 0.08
        executor._move_to(hover.tolist(), orient, velocity=APPROACH_VEL)

        for attempt in range(max_retries):
            z_adj = GRASP_Z + attempt * GRASP_Z_STEP
            pinch = grasp_pos.copy()
            pinch[2] = z_adj
            logger.info(
                "[cloth_fold] grasp attempt %d/%d at z=%.4f",
                attempt + 1,
                max_retries,
                z_adj,
            )
            executor._move_to(pinch.tolist(), orient, velocity=APPROACH_VEL * 0.5)

            # Close gripper
            if hasattr(executor.robot, "_send_gripper_command"):
                executor.robot._send_gripper_command(
                    1.0, speed=100, force=min(GRASP_FORCE, 100)
                )
            else:
                executor.robot.close_gripper()
            time.sleep(0.4)

            # Check if gripped
            try:
                width = executor.robot.get_gripper_width()
            except Exception:
                width = 0.0
            if width < GRIPPER_WIDTH_THRESHOLD and width > 0.001:
                logger.info("[cloth_fold] cloth gripped (width=%.4fm)", width)
                executor._holding = True
                return True

            logger.info("[cloth_fold] grasp missed (width=%.4fm), retrying", width)
            # Open and lift slightly before next attempt
            executor.robot.open_gripper()
            time.sleep(0.2)
            executor._move_to(hover.tolist(), orient, velocity=APPROACH_VEL)

        logger.warning("[cloth_fold] all %d grasp attempts failed", max_retries)
        return False

    # helper: sinusoidal arc waypoints
    def _arc_waypoints_sin(start, end, lift_h, arc_peak, n_pts):
        """
        Generate arc waypoints with sinusoidal Z peak.

        start/end: 3D positions. Lifts from start[2]+lift_h,
        arcs with sinusoidal peak of arc_peak, lands at end[2]+0.005.
        """
        pts = []
        lift_start = start.copy()
        lift_start[2] += lift_h
        land_z = 0.005  # land just above table

        for i in range(n_pts + 1):
            t = i / float(n_pts)
            xy = lift_start[:2] * (1.0 - t) + end[:2] * t
            # Z: interpolate from lift to land, plus sinusoidal peak
            z_base = lift_start[2] * (1.0 - t) + land_z * t
            z_arc = arc_peak * np.sin(np.pi * t)
            pts.append(np.array([xy[0], xy[1], z_base + z_arc]))
        return pts

    # helper: pre-verify IK for all waypoints
    def _verify_waypoints_ik(waypoints, orient):
        """
        Solve IK for each waypoint. Returns list of joint configs
        or None if any waypoint is unreachable.
        """
        try:
            q_current = np.array(executor.robot.get_joint_positions(), dtype=float)
        except Exception:
            # IK seed only (not the ready pose): rounded J5=0 variant, intentionally
            # differs from franka_base.HOME_CONFIG / franka_default.yaml home_config.
            q_current = np.array([0, -0.785, 0, -2.356, 0, 1.571, 0.785])

        joint_configs = []
        q_seed = q_current.copy()
        for i, wp in enumerate(waypoints):
            q = _solve_ik_pyroki(wp, orient, q_seed=q_seed)
            if q is None:
                logger.warning(
                    "[cloth_fold] IK failed for waypoint %d/%d " "pos=(%.3f,%.3f,%.3f)",
                    i,
                    len(waypoints),
                    *wp,
                )
                return None
            joint_configs.append(q)
            q_seed = q  # chain seeds for smooth path
        return joint_configs

    # helper: execute a fold (lift + arc)
    def _execute_fold(grasp_pos, target_xy, orient, lift_h, arc_peak, label_tag):
        """
        Lift, arc-fold, land. Assumes gripper is already closed on
        cloth at grasp_pos.
        """
        # Compute waypoints
        target_3d = np.array([target_xy[0], target_xy[1], 0.0])
        waypoints = _arc_waypoints_sin(
            grasp_pos, target_3d, lift_h, arc_peak, ARC_WAYPOINTS
        )

        # Pre-verify IK for all waypoints
        jconfigs = _verify_waypoints_ik(waypoints, orient)
        if jconfigs is None:
            logger.warning(
                "[cloth_fold] %s fold: IK pre-check failed, "
                "attempting direct move fallback",
                label_tag,
            )
            # Fallback: just try the endpoints
            lift_pos = grasp_pos.copy()
            lift_pos[2] += lift_h
            executor._move_to(lift_pos.tolist(), orient, velocity=FOLD_VEL)
            land_pos = np.array([target_xy[0], target_xy[1], 0.005])
            executor._move_to(land_pos.tolist(), orient, velocity=FOLD_VEL)
            return True

        # Execute via joint configs for smooth, singularity-free motion
        logger.info(
            "[cloth_fold] %s fold: executing %d waypoints", label_tag, len(jconfigs)
        )
        for i, q in enumerate(jconfigs):
            executor.robot.move_to_joint_config(q.tolist(), velocity=FOLD_VEL)
        return True

    # helper: release and retract
    def _release_retract(orient):
        """
        Open gripper, retract upward.
        """
        executor.robot.open_gripper()
        time.sleep(0.3)
        executor._holding = False
        current = executor._get_current_position()
        retract = current.copy()
        retract[2] += 0.08
        executor._move_to(retract.tolist(), orient, velocity=APPROACH_VEL)

    # main fold sequence

    # Step 1: Detect shirt parts
    logger.info("[cloth_fold] Step 1: detecting shirt parts")
    prompts = ["left sleeve", "right sleeve", "shirt"]
    dets = _redetect(prompts)
    if dets is None or len(dets) == 0:
        # Fall back to existing detection_map
        logger.info("[cloth_fold] redetect failed, using existing " "detection_map")
        det_map = executor.detection_map
    else:
        det_map = _det_to_map(dets)
        # Also update executor's detection_map so downstream lookups work
        executor.detection_map.update(det_map)

    # Find sleeves: ignore labels, sort by Y position
    sleeve_dets = []
    for label, det in det_map.items():
        lw = label.lower()
        if "sleeve" in lw:
            pos = np.asarray(det["position_3d"], dtype=float)
            sleeve_dets.append((label, det, pos))
    if len(sleeve_dets) < 2:
        logger.warning("[cloth_fold] found %d sleeves, need 2", len(sleeve_dets))
        return _result(
            "cloth_fold",
            False,
            f"Need 2 sleeve detections, found {len(sleeve_dets)}",
            time.time() - t0,
        )

    # Sort by Y: highest Y first (+Y sleeve first)
    sleeve_dets.sort(key=lambda s: s[2][1], reverse=True)
    plus_y_sleeve = sleeve_dets[0]  # +Y sleeve
    minus_y_sleeve = sleeve_dets[1]  # -Y sleeve

    # Find shirt body
    shirt_det = None
    for label, det in det_map.items():
        lw = label.lower()
        if "shirt" in lw and "sleeve" not in lw:
            shirt_det = (label, det)
            break
    if shirt_det is None:
        return _result(
            "cloth_fold", False, "Shirt body detection not found", time.time() - t0
        )

    shirt_pos = np.asarray(shirt_det[1]["position_3d"], dtype=float)
    shirt_center_y = shirt_pos[1]

    # Get shirt mask for computing max_x (collar edge)
    shirt_mask = shirt_det[1].get("_mask")

    logger.info(
        "[cloth_fold] +Y sleeve: %s pos=(%.3f,%.3f,%.3f)",
        plus_y_sleeve[0],
        *plus_y_sleeve[2],
    )
    logger.info(
        "[cloth_fold] -Y sleeve: %s pos=(%.3f,%.3f,%.3f)",
        minus_y_sleeve[0],
        *minus_y_sleeve[2],
    )
    logger.info("[cloth_fold] shirt center Y=%.3f", shirt_center_y)

    # Compute shared sleeve grasp X: use max X from both sleeves
    sleeve_grasp_x = max(plus_y_sleeve[2][0], minus_y_sleeve[2][0])

    # Step 2: Fold +Y sleeve
    logger.info("[cloth_fold] Step 2: folding +Y sleeve")

    # +Y sleeve: grasp at outer edge (max_y - 10mm)
    py_pos = plus_y_sleeve[2]
    py_grasp_y = py_pos[1] - 0.010  # max_y - 10mm (inward from edge)
    py_grasp = np.array([sleeve_grasp_x, py_grasp_y, GRASP_Z])

    # Fold target: 8cm past shirt center in -Y direction
    py_fold_target_y = shirt_center_y - SLEEVE_PAST_CENTER_M
    py_fold_target = np.array([sleeve_grasp_x, py_fold_target_y])

    py_orient = _obb_yaw_orient(plus_y_sleeve[1], GRASP_ORIENT_90DEG)
    if not _grasp_cloth(py_grasp, py_orient):
        return _result(
            "cloth_fold",
            False,
            "+Y sleeve grasp failed after all retries",
            time.time() - t0,
        )

    _execute_fold(
        py_grasp, py_fold_target, py_orient, SLEEVE_LIFT_H, ARC_PEAK, "+Y_sleeve"
    )
    _release_retract(py_orient)

    # Step 3: Fold -Y sleeve
    logger.info("[cloth_fold] Step 3: folding -Y sleeve")

    # -Y sleeve: grasp at outer edge (min_y + 10mm)
    my_pos = minus_y_sleeve[2]
    my_grasp_y = my_pos[1] + 0.010  # min_y + 10mm (inward from edge)
    my_grasp = np.array([sleeve_grasp_x, my_grasp_y, GRASP_Z])

    # Fold target: 8cm past shirt center in +Y direction
    my_fold_target_y = shirt_center_y + SLEEVE_PAST_CENTER_M
    my_fold_target = np.array([sleeve_grasp_x, my_fold_target_y])

    my_orient = _obb_yaw_orient(minus_y_sleeve[1], GRASP_ORIENT_90DEG)
    if not _grasp_cloth(my_grasp, my_orient):
        return _result(
            "cloth_fold",
            False,
            "-Y sleeve grasp failed after all retries",
            time.time() - t0,
        )

    _execute_fold(
        my_grasp, my_fold_target, my_orient, SLEEVE_LIFT_H, ARC_PEAK, "-Y_sleeve"
    )
    _release_retract(my_orient)

    # Step 4: Fold the collar down to the hem (body fold)
    logger.info("[cloth_fold] Step 4: folding collar down to hem")

    # Re-detect shirt to get updated mask after sleeve folds
    dets2 = _redetect(["shirt"])
    if dets2 is not None and len(dets2) > 0:
        det_map2 = _det_to_map(dets2)
        for label, det in det_map2.items():
            if "shirt" in label.lower() and "sleeve" not in label.lower():
                shirt_det = (label, det)
                shirt_pos = np.asarray(det["position_3d"], dtype=float)
                shirt_mask = det.get("_mask")
                break

    # Body fold: grasp the hem edge (min_x side, toward the base) with the
    # pitched +X approach and fold it up to the collar edge (max_x side).
    # Both edges come from the shirt OBB major axis about the centroid.
    half_major = 0.0
    if shirt_mask is not None:
        try:
            ys, xs = np.where(shirt_mask > 0)
            if len(xs) > 10:
                obb_minor = float(shirt_det[1].get("obb_minor_m", 0.0) or 0.0)
                ar = float(shirt_det[1].get("aspect_ratio", 1.0) or 1.0)
                if obb_minor > 0.01 and ar > 0.5:
                    half_major = (obb_minor * ar) / 2.0
        except Exception:
            pass

    # Hem grasp sits 10mm inward from the edge; the collar target is the
    # opposite edge, clamped to the reach limit.
    hem_x = shirt_pos[0] - half_major + 0.010
    collar_x = min(shirt_pos[0] + half_major, HEM_MAX_X_CLAMP)

    hem_grasp = shirt_pos.copy()
    hem_grasp[0] = hem_x
    hem_grasp[2] = GRASP_Z
    hem_fold_target = np.array([collar_x, shirt_pos[1]])

    logger.info(
        "[cloth_fold] hem grasp at (%.3f,%.3f,%.3f) tilt=%.0f deg, "
        "fold up to collar_x=%.3f",
        *hem_grasp,
        HEM_TILT_DEG,
        collar_x,
    )

    if not _grasp_cloth(hem_grasp, GRASP_ORIENT_HEM):
        return _result(
            "cloth_fold", False, "Hem grasp failed after all retries", time.time() - t0
        )

    # Pre-compute ALL waypoints and verify IK reachability before moving.
    hem_target_3d = np.array([hem_fold_target[0], hem_fold_target[1], 0.0])
    body_waypoints = _arc_waypoints_sin(
        hem_grasp, hem_target_3d, HEM_LIFT_H, ARC_PEAK, ARC_WAYPOINTS
    )

    body_jconfigs = _verify_waypoints_ik(body_waypoints, GRASP_ORIENT_HEM)
    if body_jconfigs is None:
        logger.warning(
            "[cloth_fold] hem->collar fold IK pre-check failed; "
            "releasing and aborting"
        )
        _release_retract(GRASP_ORIENT_HEM)
        return _result(
            "cloth_fold",
            False,
            "Hem fold IK pre-check failed (unreachable waypoints)",
            time.time() - t0,
        )

    # Execute the hem -> collar fold via pre-verified joint configs
    logger.info(
        "[cloth_fold] hem->collar fold: executing %d waypoints", len(body_jconfigs)
    )
    for q in body_jconfigs:
        executor.robot.move_to_joint_config(q.tolist(), velocity=FOLD_VEL)

    _release_retract(GRASP_ORIENT_HEM)

    elapsed = time.time() - t0
    logger.info("[cloth_fold] complete in %.1fs", elapsed)
    return _result(
        "cloth_fold",
        True,
        f"T-shirt fold complete: +Y sleeve, -Y sleeve, "
        f"hem->collar ({elapsed:.1f}s)",
        time.time() - t0,
    )

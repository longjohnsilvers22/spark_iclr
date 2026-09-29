# Horizontal / angled side grasp for upright cylindrical objects.

import logging
import time

import numpy as np
from typing import Optional

from spark_real.skills.primitives import _result
from spark_real.skills.grasp_utils import (
    horizontal_grasp_orientation,
    is_benign_franky_reflex,
    joint_motion_to,
)

logger = logging.getLogger(__name__)


def grasp_horizontal(
    executor,
    label: str,
    det: dict,
    force: float,
    target_width: Optional[float],
    object_height_m: float,
    pre_grasp_distance: float,
    approach_dir_xy: Optional[np.ndarray],
    t0: float,
    approach_pitch_rad: float = 0.0,
    grasp_z_m: Optional[float] = None,
    grasp_height_fraction: float = 0.5,
):
    """
    Side/horizontal grasp: geometric approach, no EquiGraspFlow.

    Splits approach into 3 sub-motions (top-down hover, wrist rotation,
    side translate) to avoid combined translation+rotation reflexes.
    """
    pos = det.get("position_3d")
    if pos is None:
        return _result(
            "grasp_se3",
            False,
            f"horizontal: '{label}' has no position_3d",
            time.time() - t0,
        )
    obj_xyz = np.array(pos, dtype=float).copy()

    # Compute grasp Z
    if grasp_z_m is not None:
        grasp_z = float(grasp_z_m)
    else:
        frac = float(np.clip(grasp_height_fraction, 0.0, 1.0))
        grasp_z = float(obj_xyz[2]) - float(object_height_m) * (1.0 - frac)
        logger.info(
            "[horizontal] grasp_height_fraction=%.2f -> grasp_z=%.3f "
            "(det_z=%.3f, h=%.3f)",
            frac,
            grasp_z,
            obj_xyz[2],
            object_height_m,
        )

    floor_z = getattr(executor, "TABLE_Z_FLOOR", -0.04) + 0.030
    if grasp_z < floor_z:
        logger.info(
            "[horizontal] clamp grasp_z %.3f -> %.3f (table+30mm margin)",
            grasp_z,
            floor_z,
        )
        grasp_z = floor_z

    # Approach direction in world XY
    if approach_dir_xy is not None:
        approach = np.asarray(approach_dir_xy, dtype=float).reshape(-1)[:2]
        if float(np.linalg.norm(approach)) < 1e-6:
            approach_dir_xy = None
    if approach_dir_xy is None:
        radial = obj_xyz[:2].copy()
        if float(np.linalg.norm(radial)) < 1e-6:
            radial = np.array([1.0, 0.0])
        approach = radial / float(np.linalg.norm(radial))
    else:
        approach = approach / float(np.linalg.norm(approach))

    orient = horizontal_grasp_orientation(approach, pitch_rad=approach_pitch_rad)

    grasp_pos = np.array([obj_xyz[0], obj_xyz[1], grasp_z], dtype=float)
    pre_d = float(pre_grasp_distance)
    cos_p = float(np.cos(approach_pitch_rad))
    sin_p = float(np.sin(approach_pitch_rad))
    hover_xy = obj_xyz[:2] - approach * (pre_d * cos_p)
    hover_z = grasp_z + pre_d * sin_p
    hover_pos = np.array([hover_xy[0], hover_xy[1], hover_z], dtype=float)

    logger.info(
        "[horizontal] '%s' obj=(%.3f,%.3f,%.3f) grasp_z=%.3f "
        "approach=(%+.2f,%+.2f) hover=(%.3f,%.3f,%.3f)",
        label,
        *obj_xyz,
        grasp_z,
        approach[0],
        approach[1],
        *hover_pos,
    )

    # Workspace check
    ws_min = np.asarray(executor.WORKSPACE_MIN, dtype=float)
    ws_max = np.asarray(executor.WORKSPACE_MAX, dtype=float)
    if not (np.all(hover_pos >= ws_min) and np.all(hover_pos <= ws_max)):
        return _result(
            "grasp_se3",
            False,
            f"horizontal: pre-grasp hover outside workspace",
            time.time() - t0,
        )

    try:
        executor.robot.open_gripper()
        time.sleep(0.3)
    except Exception as exc:
        logger.warning("[horizontal] open_gripper failed: %s", exc)

    executor._last_grasp_target_width = (
        float(target_width) if target_width is not None else None
    )

    # 3-phase safe approach: top-down hover -> rotate wrist -> side translate
    TOP_DOWN_ORIENT = np.array([np.pi, 0.0, 0.0], dtype=float)
    cup_top_est = float(obj_xyz[2]) + float(object_height_m) / 2.0
    hover_high_z = max(grasp_z + 0.15, cup_top_est + 0.10)
    if hover_high_z > ws_max[2] - 0.02:
        hover_high_z = ws_max[2] - 0.02
    hover_high_pos = np.array([obj_xyz[0], obj_xyz[1], hover_high_z], dtype=float)

    # Phase 1a: top-down hover above object
    p1a_ok = joint_motion_to(
        executor,
        hover_high_pos,
        TOP_DOWN_ORIENT,
        velocity_factor=0.5,
        label="hover_topdown",
    )
    if not p1a_ok:
        try:
            executor._move_to(hover_high_pos.tolist(), TOP_DOWN_ORIENT)
        except Exception as exc:
            return _result(
                "grasp_se3",
                False,
                f"horizontal '{label}': phase 1a failed: {exc}",
                time.time() - t0,
            )

    # Phase 1b: rotate wrist to horizontal at same position
    p1b_ok = joint_motion_to(
        executor,
        hover_high_pos,
        orient,
        velocity_factor=0.3,
        label="rotate_to_horizontal",
    )
    if not p1b_ok:
        try:
            executor._move_to(hover_high_pos.tolist(), orient)
        except Exception as exc:
            return _result(
                "grasp_se3",
                False,
                f"horizontal '{label}': phase 1b rotate failed: {exc}",
                time.time() - t0,
            )

    # Phase 1c: translate to side-hover
    hover_ok = joint_motion_to(
        executor,
        hover_pos,
        orient,
        velocity_factor=0.5,
        label="hover_horizontal_side",
    )
    if not hover_ok:
        executor._move_to(hover_pos, orient)

    # Phase 2: radial in-feed
    descent_ok = joint_motion_to(
        executor,
        grasp_pos,
        orient,
        velocity_factor=0.4,
        label="approach_horizontal",
    )
    if not descent_ok:
        try:
            executor._servo_to(grasp_pos, orient, velocity=executor.velocity * 0.5)
        except Exception as exc:
            if not is_benign_franky_reflex(exc, ("singular",)):
                raise

    # Close gripper
    if target_width is not None and hasattr(executor.robot, "grasp_to_width"):
        executor.robot.grasp_to_width(
            width=float(target_width),
            force=min(float(force), 100.0),
            speed=0.05,
        )
    else:
        executor._gripper_squeeze(force=float(force), speed=80, settle=0.3)

    grasped = executor._verify_grasp()
    if not grasped:
        executor._holding = False
        return _result(
            "grasp_se3",
            False,
            f"horizontal '{label}': gripper closed empty",
            time.time() - t0,
        )
    executor._holding = True
    executor._last_keypoint_label = label

    # Post-grasp lift keeping horizontal orient
    lift_pos = np.array([grasp_pos[0], grasp_pos[1], grasp_pos[2] + 0.10], dtype=float)
    if lift_pos[2] > ws_max[2]:
        lift_pos[2] = ws_max[2]
    try:
        executor._move_to(lift_pos.tolist(), orient, velocity=executor.velocity * 0.4)
    except Exception as exc:
        logger.warning("[horizontal] post-grasp lift failed (%s)", exc)

    return _result(
        "grasp_se3",
        True,
        f"horizontal '{label}' grasped at z={grasp_z:.3f} "
        f"approach=({approach[0]:+.2f},{approach[1]:+.2f}), closure verified",
        time.time() - t0,
    )

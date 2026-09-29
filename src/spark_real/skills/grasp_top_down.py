# Top-down grasp using SAM3 OBB minor axis as closing axis.

import logging
import time

import numpy as np
from typing import Optional

from spark_real.skills.primitives import _result
from scipy.spatial.transform import Rotation as _R
from spark_real.skills.grasp_utils import (
    canonical_symmetric_yaw,
    joint_motion_to,
    orient_diag,
)
from spark_real.control.waypoints import Waypoint as _WP, WaypointBuffer as _WB

try:
    from spark_real.control.fr3_ik_pyroki import fk as _pyroki_fk
except ImportError:
    _pyroki_fk = None

logger = logging.getLogger(__name__)


def grasp_top_down(
    executor,
    label: str,
    det: dict,
    force: float,
    target_width: Optional[float],
    t0: float,
    grip_mode: str = "force",
):
    """
    Top-down grasp: OBB-aligned yaw, descent via joint-space, close + verify.
    """

    aspect_ratio = float(det.get("aspect_ratio", 1.0) or 1.0)
    orientation_angle = float(det.get("orientation_angle", 0.0) or 0.0)
    obb_minor_m = float(det.get("obb_minor_m", 0.0) or 0.0)
    pos = det.get("position_3d")
    if pos is None:
        return _result(
            "grasp_se3",
            False,
            f"top_down: '{label}' has no position_3d",
            time.time() - t0,
        )
    grasp_pos = np.array(pos, dtype=float).copy()

    # Small Z descent bias for thin handles
    Z_DESCENT_BIAS_M = 0.005 if aspect_ratio >= 1.5 else 0.0
    if Z_DESCENT_BIAS_M:
        grasp_pos[2] -= Z_DESCENT_BIAS_M

    floor_z = getattr(executor, "TABLE_Z_FLOOR", -0.04) + 0.003
    if grasp_pos[2] < floor_z:
        logger.info(
            "[top_down] clamp grasp_z %.3f -> %.3f (table floor)",
            grasp_pos[2],
            floor_z,
        )
        grasp_pos[2] = floor_z

    # Yaw alignment from OBB major-axis angle. The +-90 wrap is re-picked
    # against the current wrist yaw (two-jaw symmetry) so the wrist never
    # travels more than ~90 deg to align.
    yaw = canonical_symmetric_yaw(orientation_angle)
    if hasattr(executor, "_nearest_symmetric_yaw"):
        yaw = executor._nearest_symmetric_yaw(yaw)

    _base_R = _R.from_rotvec(executor.GRASP_ORIENTATION)
    base_orient = list(executor.GRASP_ORIENTATION)
    yawed_orient = (_R.from_euler("z", yaw) * _base_R).as_rotvec().tolist()
    orient = yawed_orient

    logger.info(
        "[top_down] '%s' grasp=(%.3f,%.3f,%.3f) yaw=%.1f deg "
        "(major=%.1f, ar=%.2f, obb_minor=%.0fmm)",
        label,
        *grasp_pos,
        np.rad2deg(yaw),
        np.rad2deg(orientation_angle),
        aspect_ratio,
        obb_minor_m * 1000.0,
    )

    try:
        executor.robot.open_gripper()
        time.sleep(0.3)
    except Exception as exc:
        logger.warning("[top_down] open_gripper failed: %s", exc)

    executor._last_grasp_target_width = (
        float(target_width) if target_width is not None else None
    )

    current = executor._get_current_position()
    safe_z = executor._safe_clearance_z(current, grasp_pos)
    hover_pos = np.array([grasp_pos[0], grasp_pos[1], safe_z], dtype=float)

    orient_diag("base_orient_target", base_orient)
    orient_diag("yawed_orient_target", yawed_orient)

    # Phase 1: hover above object
    hover_ok = joint_motion_to(
        executor, hover_pos, base_orient, velocity_factor=0.9, label="hover"
    )
    if not hover_ok:
        logger.warning("[top_down] joint hover failed; cartesian fallback")
        executor._move_to(hover_pos, base_orient)

    if _pyroki_fk is not None:
        try:
            q_actual = np.asarray(executor.robot.get_joint_positions(), dtype=float)[:7]
            _, R_ach = _pyroki_fk(q_actual)
            orient_diag("hover_AFTER_motion", base_orient, R_ach)
        except Exception:
            pass

    # Phase 2: descend straight to the birdview/sideview XY target. The wrist
    # camera is video-only and never used for control.

    # Phase 3: descend + yaw via joint-space
    descent_ok = joint_motion_to(
        executor, grasp_pos, yawed_orient, velocity_factor=0.3, label="descend"
    )
    if not descent_ok:
        logger.warning("[top_down] joint descent failed; cartesian fallback")
        executor._move_to(grasp_pos, yawed_orient)

    if _pyroki_fk is not None:
        try:
            q_actual = np.asarray(executor.robot.get_joint_positions(), dtype=float)[:7]
            _, R_ach = _pyroki_fk(q_actual)
            orient_diag("descend_AFTER_motion", yawed_orient, R_ach)
        except Exception:
            pass

    # Waypoint buffer handles for post-grasp lift chaining
    _buf = getattr(executor, "waypoint_buffer", None)
    _franky = None
    if _buf is not None:
        try:
            _franky = _WB.resolve_franky_robot(executor.robot)
        except Exception:
            _franky = None

    # Pre-close gate: closing before the descent reaches the grasp pose can
    # dislodge the object. Measure the actual TCP; if it is not at the grasp
    # pose, re-approach once, and if still off fail without closing so the
    # object is left undisturbed for the executor retry.
    PRECLOSE_XY_TOL_M = 0.012
    PRECLOSE_Z_TOL_M = 0.020

    def _preclose_err():
        _pre_tcp = executor._get_current_position()
        _exy = float(np.linalg.norm(np.array(_pre_tcp[:2]) - np.array(grasp_pos[:2])))
        _ez = float(_pre_tcp[2] - grasp_pos[2])
        logger.info(
            "[top_down] pre-close TCP=(%.3f,%.3f,%.3f) "
            "target=(%.3f,%.3f,%.3f) xy_err=%.3fm z_err=%+.3fm",
            _pre_tcp[0],
            _pre_tcp[1],
            _pre_tcp[2],
            grasp_pos[0],
            grasp_pos[1],
            grasp_pos[2],
            _exy,
            _ez,
        )
        return _exy, _ez

    try:
        _err_xy, _err_z = _preclose_err()
        if _err_xy > PRECLOSE_XY_TOL_M or abs(_err_z) > PRECLOSE_Z_TOL_M:
            logger.warning(
                "[top_down] '%s' not at grasp pose (xy_err=%.3f z_err=%+.3f); "
                "re-approaching once",
                label,
                _err_xy,
                _err_z,
            )
            executor._move_to(grasp_pos, yawed_orient)
            _err_xy, _err_z = _preclose_err()
            if _err_xy > PRECLOSE_XY_TOL_M or abs(_err_z) > PRECLOSE_Z_TOL_M:
                executor._holding = False
                return _result(
                    "grasp_se3",
                    False,
                    f"top_down '{label}': descent never reached grasp pose "
                    f"(xy_err={_err_xy:.3f}m z_err={_err_z:+.3f}m); "
                    "not closing",
                    time.time() - t0,
                )
    except Exception as _exc:
        logger.warning("[top_down] pre-close TCP read failed: %s", _exc)

    # Close gripper
    if grip_mode == "position" and hasattr(executor.robot, "set_gripper_position"):
        grip_w = float(target_width) if target_width is not None else 0.040
        pos_normalized = max(0.0, min(1.0, 1.0 - (grip_w / 0.080)))
        logger.info(
            "[top_down] grip_mode=position: width=%.1fmm (pos=%.2f)",
            grip_w * 1000,
            pos_normalized,
        )
        executor.robot.set_gripper_position(pos_normalized)
        time.sleep(0.5)
    else:
        executor._gripper_squeeze(force=float(force), speed=80, settle=0.2)

    # Verify hold with re-detect retry
    grasped = executor._verify_grasp()
    if not grasped:
        logger.info("[top_down] '%s': first attempt empty, re-detecting", label)
        try:
            executor.robot.open_gripper()
            cur = np.array(executor.robot.get_tcp_pose()[:3])
            cur[2] = min(cur[2] + 0.08, 0.50)
            executor._move_to(cur.tolist(), executor.GRASP_ORIENTATION)
        except Exception:
            pass

        redetected = False
        if hasattr(executor, "_redetect_single"):
            try:
                executor._redetect_single(label)
                redetected = True
            except Exception:
                pass

        if redetected:
            # In-container check is log-only: instance renumbering across
            # re-detects can make this guard fire on the wrong physical object,
            # so log what it would have done then retry the grasp regardless.
            try:
                # Imported here to break a circular import: execution_recovery
                # imports spark_real.skills.registry at module load.
                from spark_real.control.execution_recovery import (
                    grasp_target_in_container,
                )

                _in = grasp_target_in_container(executor, label)
            except Exception:
                _in = None
            if _in is not None:
                logger.info(
                    "[top_down] '%s' re-detected inside container '%s' "
                    "(guard is log-only; proceeding with retry)",
                    label,
                    _in,
                )
            new_det = executor.detection_map.get(label)
            if new_det is not None:
                new_pos = new_det.get("position_3d")
                if new_pos is not None:
                    new_xyz = np.array(new_pos, dtype=float)
                    new_grasp = np.array(
                        [
                            new_xyz[0],
                            new_xyz[1],
                            max(float(new_xyz[2]), executor.TABLE_Z_FLOOR + 0.002),
                        ]
                    )
                    logger.info(
                        "[top_down] retry at new pos (%.3f,%.3f,%.3f)",
                        *new_grasp,
                    )
                    try:
                        executor._move_to(new_grasp.tolist(), orient)
                        executor._gripper_squeeze(
                            force=float(force), speed=80, settle=0.2
                        )
                        grasped = executor._verify_grasp()
                    except Exception:
                        pass

        if not grasped:
            executor._holding = False
            return _result(
                "grasp_se3",
                False,
                f"top_down '{label}': gripper closed empty (after re-detect retry)",
                time.time() - t0,
            )

    executor._holding = True
    executor._last_keypoint_label = label

    # Post-grasp lift (queued as waypoint if buffer available)
    lift_pos = grasp_pos.copy()
    lift_pos[2] = safe_z
    if _buf is not None and _franky is not None and hasattr(_buf, "add"):
        try:
            _buf.add(_WP(position=lift_pos, orientation=orient, label="lift"))
            logger.info(
                "[top_down] queued post-grasp lift (buffer pending=%d)",
                _buf.pending,
            )
        except Exception:
            try:
                executor._move_to(lift_pos, orient)
            except Exception:
                pass
    else:
        try:
            executor._move_to(lift_pos, orient)
        except Exception:
            pass

    return _result(
        "grasp_se3",
        True,
        f"top_down '{label}' grasped at yaw={np.rad2deg(yaw):.1f} deg, "
        f"closure verified",
        time.time() - t0,
    )

# SE(3) 6-DOF grasp using EquiGraspFlow for candidate generation.

import logging
import time

import numpy as np
from typing import Optional

from spark_real.skills.primitives import _result
from spark_real.skills.grasp_utils import (
    _resolve_franky_robot,
    _franka_joint_chain,
    canonical_symmetric_yaw,
    is_benign_franky_reflex,
)
from scipy.spatial.transform import Rotation
from spark_real.perception.equigrasp import (
    EquiGraspGenerator,
    generate_grasps_from_detection,
    _workspace_bounds as _ws_bounds,
)

try:
    import torch
except ImportError:
    torch = None

try:
    from PIL import Image as _PILImage
except ImportError:
    _PILImage = None

try:
    import franky
except ImportError:
    franky = None

try:
    from spark_real.control.fr3_ik_pyroki import solve_ik as _solve_ik
except ImportError:
    _solve_ik = None

logger = logging.getLogger(__name__)


class _DetProxy:
    """Lightweight detection stand-in for grasp candidate generation.

    Wraps a fresh mask and the OBB-derived geometry fields that
    generate_grasps_from_detection reads. Optional fields default to None
    so the re-detect path can build one from label and mask alone.
    """

    def __init__(self, lbl, msk, orient_angle=None, ar=None, obb_minor_m=None):
        self.label = lbl
        self.mask = msk
        self.orientation_angle = orient_angle
        self.aspect_ratio = ar
        self.obb_minor_m = obb_minor_m


def grasp_se3_flow(
    executor,
    label: str,
    det: dict,
    force: float,
    target_width: Optional[float],
    n_candidates: int,
    prefer_side: bool,
    prefer_angled: bool,
    grip_end: str,
    pre_grasp_dist: float,
    max_angle: float,
    t0: float,
):
    """
    Full SE(3) grasp: EquiGraspFlow candidate generation + arched approach.
    """
    if torch is None or _PILImage is None:
        return _result(
            "grasp_se3",
            False,
            "torch/PIL not available (required for SE(3) grasp)",
            time.time() - t0,
        )

    # Boost force for thin rigid objects
    try:
        _ar = float(det.get("aspect_ratio", 0.0) or 0.0)
        _ob = float(det.get("obb_minor_m", 0.0) or 0.0)
        if _ar >= 3.0 and 0 < _ob < 0.020 and force < 80.0:
            force = 80.0
    except (TypeError, ValueError):
        pass

    pipeline = executor._pipeline
    if pipeline is None:
        return _result(
            "grasp_se3",
            False,
            "No pipeline reference for SE(3) grasp",
            time.time() - t0,
        )

    captures = pipeline.capture()

    # Pick best camera with RGB + depth + calibration
    camera_data = None
    camera_name = None
    for cam_key in ("birdview", "sideview", "wrist"):
        if cam_key in captures:
            cd = captures[cam_key]
            if (
                cd.get("rgb") is not None
                and cd.get("depth") is not None
                and cd.get("calibration") is not None
            ):
                camera_data = cd
                camera_name = cam_key
                break

    if camera_data is None:
        return _result(
            "grasp_se3",
            False,
            "No camera with RGB + depth + calibration available",
            time.time() - t0,
        )

    # Get or generate mask
    try:
        stored_mask = det.get("_mask")
        stored_cam = det.get("_camera")
        fresh_mask = None

        if (
            stored_mask is not None
            and stored_cam == camera_name
            and hasattr(stored_mask, "shape")
            and stored_mask.shape[:2] == camera_data["rgb"].shape[:2]
        ):
            fresh_mask = stored_mask > 0 if stored_mask.dtype != bool else stored_mask
        else:
            sam3 = pipeline._perception._sam3
            pil_img = _PILImage.fromarray(camera_data["rgb"])
            state = sam3.set_image(pil_img)
            sam3_prompt = (
                label.rsplit(" ", 1)[0] if label and label[-1].isdigit() else label
            )
            state = sam3.set_text_prompt(prompt=sam3_prompt, state=state)
            masks = state.get("masks", torch.tensor([]))
            scores = state.get("scores", torch.tensor([]))

            if masks.numel() == 0:
                return _result(
                    "grasp_se3",
                    False,
                    f"'{label}' not detected in {camera_name}",
                    time.time() - t0,
                )

            # Pick mask closest to expected 3D position
            det_pos = np.array(det["position_3d"][:3])
            cal = camera_data.get("calibration")
            best_idx = int(scores.argmax())

            if cal is not None and masks.shape[0] > 1:
                R = cal.extrinsic[:3, :3]
                t = cal.extrinsic[:3, 3]
                p_cam = R.T @ (det_pos - t)
                if p_cam[2] > 0.01:
                    expected_cx = cal.fx * p_cam[0] / p_cam[2] + cal.cx
                    expected_cy = cal.fy * p_cam[1] / p_cam[2] + cal.cy
                    best_dist = float("inf")
                    for mi in range(masks.shape[0]):
                        m = masks[mi].cpu().numpy().squeeze()
                        ys, xs = np.where(m > 0)
                        if len(xs) == 0:
                            continue
                        d = np.sqrt(
                            (float(xs.mean()) - expected_cx) ** 2
                            + (float(ys.mean()) - expected_cy) ** 2
                        )
                        if d < best_dist:
                            best_dist = d
                            best_idx = mi

            fresh_mask = masks[best_idx].cpu().numpy().squeeze()

        det_obj = _DetProxy(
            label,
            fresh_mask,
            orient_angle=det.get("orientation_angle"),
            ar=det.get("aspect_ratio"),
            obb_minor_m=det.get("obb_minor_m"),
        )

        candidates = generate_grasps_from_detection(
            detection=det_obj,
            camera_data=camera_data,
            n_candidates=n_candidates,
            max_approach_angle=max_angle,
            grip_end=grip_end,
        )
    except Exception as e:
        logger.error("EquiGraspFlow generation failed: %s", e, exc_info=True)
        return _result(
            "grasp_se3",
            False,
            f"EquiGraspFlow generation failed: {e}",
            time.time() - t0,
        )

    if not candidates:
        return _result(
            "grasp_se3",
            False,
            f"No valid SE(3) grasps generated for '{label}'",
            time.time() - t0,
        )

    # Select best grasp
    current_tcp = None
    try:
        obs = executor.robot.get_observation()
        current_tcp = np.array(obs["tcp_pose"][:3])
    except Exception:
        pass

    gen = EquiGraspGenerator.__new__(EquiGraspGenerator)
    best = gen.select_best_grasp(
        candidates,
        current_tcp=current_tcp,
        prefer_side=prefer_side,
    )
    if best is None:
        return _result(
            "grasp_se3",
            False,
            f"No suitable grasp found for '{label}'",
            time.time() - t0,
        )

    grasp_pos = best.position.copy()
    approach = best.approach.copy()
    approach = approach / (np.linalg.norm(approach) + 1e-8)

    # Workspace check
    _ws = _ws_bounds()
    WORKSPACE_MIN = np.array([_ws["x_min"], _ws["y_min"], _ws["z_min"]])
    WORKSPACE_MAX = np.array([_ws["x_max"], _ws["y_max"], _ws["z_max"]])
    if not (np.all(grasp_pos >= WORKSPACE_MIN) and np.all(grasp_pos <= WORKSPACE_MAX)):
        return _result(
            "grasp_se3",
            False,
            "Grasp position outside workspace",
            time.time() - t0,
        )

    # Compute yaw from closing direction
    closing_dir = best.rotation @ np.array([1, 0, 0])
    closing_xy = closing_dir[:2]
    closing_angle = float(np.arctan2(closing_xy[1], closing_xy[0]))
    yaw_offset_egf = canonical_symmetric_yaw(closing_angle - np.pi / 2)

    # For elongated objects, prefer SAM3 OBB yaw over EGF
    aspect_ratio = float(det.get("aspect_ratio", 1.0) or 1.0)
    orientation_angle = float(det.get("orientation_angle", 0.0) or 0.0)
    if aspect_ratio >= 1.3:
        yaw_offset = canonical_symmetric_yaw(orientation_angle)
    else:
        yaw_offset = yaw_offset_egf

    # Build orientation
    if prefer_angled:
        approach_dir = best.approach / (np.linalg.norm(best.approach) + 1e-8)
        pitch = min(np.arccos(np.clip(-approach_dir[2], 0, 1)), np.deg2rad(60))
        base_rot = Rotation.from_rotvec(executor.GRASP_ORIENTATION)
        yaw_rot = Rotation.from_euler("z", yaw_offset)
        pitch_rot = Rotation.from_euler("y", pitch)
        orient = (base_rot * yaw_rot * pitch_rot).as_rotvec().tolist()
    else:
        orient = executor._oriented_grasp(yaw_offset)

    # Use detection position for XY and Z (more accurate than EGF)
    det_pos = np.array(det["position_3d"])
    if prefer_angled:
        grasp_pos = best.position.copy()
    else:
        grasp_pos = det_pos.copy()
        if aspect_ratio >= 1.5:
            grasp_pos[2] -= 0.018
            Z_MIN = -0.037
            if grasp_pos[2] < Z_MIN:
                grasp_pos[2] = Z_MIN

    # Open gripper
    executor.robot.open_gripper()
    time.sleep(0.3)

    # Arched approach via joint-space
    current = executor._get_current_position()
    safe_z = executor._safe_clearance_z(current, grasp_pos)
    hover_pos = np.array([grasp_pos[0], grasp_pos[1], safe_z])

    arched_ok = _arched_approach(
        executor, hover_pos, orient, grasp_pos, yaw_offset, safe_z
    )

    if not arched_ok:
        executor._servo_to(
            hover_pos,
            executor.GRASP_ORIENTATION,
            velocity=executor.velocity * 1.0,
        )
        executor._servo_to(
            hover_pos,
            orient,
            velocity=executor.velocity * 0.8,
        )

    # Final descent
    try:
        executor._servo_to(
            grasp_pos,
            orient,
            velocity=executor.velocity * 0.6,
        )
    except Exception as exc:
        if not is_benign_franky_reflex(exc, ("singular",)):
            raise

    # Grasp with retry + re-detect
    executor._last_grasp_target_width = (
        float(target_width) if target_width is not None else None
    )
    grasped = _grasp_retry_loop(
        executor,
        pipeline,
        label,
        det,
        force,
        grasp_pos,
        orient,
        n_candidates,
        max_angle,
        grip_end,
        prefer_side,
        target_width,
        camera_data,
        fresh_mask,
        t0,
    )

    if not grasped:
        executor._holding = False
        return _result(
            "grasp_se3",
            False,
            f"Grasp failed at ({grasp_pos[0]:.3f},{grasp_pos[1]:.3f},{grasp_pos[2]:.3f})",
            time.time() - t0,
        )

    # Lift to confirm hold
    lift_pos = executor._get_current_position()
    lift_pos[2] += 0.05
    executor._move_to(
        lift_pos,
        executor.GRASP_ORIENTATION,
        velocity=executor.velocity * 0.3,
    )

    # Post-grasp J5 escape (emergency only)
    if hasattr(executor.robot, "get_joint_positions"):
        try:
            q_post = np.asarray(executor.robot.get_joint_positions(), dtype=float)
            if abs(q_post[4]) < 0.02:
                q_esc = q_post[:7].copy()
                q_esc[4] = -0.8 if q_post[4] <= 0 else 0.8
                executor.robot.move_to_joint_config(
                    q_esc.tolist(),
                    velocity=executor.velocity * 0.7,
                )
        except Exception:
            pass

    # Post-lift width verification
    try:
        grip_pos = executor._get_gripper_position()
        if grip_pos is not None and grip_pos >= executor.GRIPPER_FULLY_CLOSED:
            grasped = False
            executor._holding = False
            try:
                executor.robot.open_gripper()
            except Exception:
                pass
    except Exception:
        pass

    if not grasped:
        executor._holding = False
        return _result(
            "grasp_se3",
            False,
            f"Grasp failed (visual verification)",
            time.time() - t0,
        )

    return _result(
        "grasp_se3",
        True,
        f"SE(3) grasped '{label}' yaw={np.rad2deg(yaw_offset):.1f} deg "
        f"score={best.score:.3f}",
        time.time() - t0,
    )


def _arched_approach(executor, hover_pos, orient, grasp_pos, yaw_offset, safe_z):
    """
    Flowing 3-waypoint arched approach via joint-space motion.
    """
    if not hasattr(executor.robot, "get_joint_positions"):
        return False
    # 7-DOF Franka only: solve_ik raises on a <7 seed (UR10e has 6). Signal the
    # caller (grasp_se3_flow) to use its _servo_to Cartesian fallback.
    if not _franka_joint_chain(executor):
        logger.info("[SE3 Grasp] non-FR3 joint chain; cartesian fallback")
        return False
    if _solve_ik is None:
        logger.info("[SE3 Grasp] pyroki IK unavailable; cartesian fallback")
        return False
    try:
        solve_ik = _solve_ik

        q_now = np.asarray(executor.robot.get_joint_positions(), dtype=float)
        q_seed = q_now[:7].copy()

        # J5 singularity escape
        if abs(q_seed[4]) < 0.05:
            q_escape = q_seed.copy()
            q_escape[4] = -0.8
            try:
                executor.robot.move_to_joint_config(
                    q_escape.tolist(),
                    velocity=executor.velocity * 0.9,
                )
                q_seed = q_escape
            except Exception:
                pass

        # Solve IK for 3 waypoints
        current_tcp = executor._get_current_position()
        cur_safe_z = max(current_tcp[2], safe_z)
        wp1_pos = np.array([current_tcp[0], current_tcp[1], cur_safe_z])
        wp2_pos = np.array([hover_pos[0], hover_pos[1], cur_safe_z])
        wp3_pos = hover_pos

        q_wp1 = solve_ik(wp1_pos, np.asarray(orient, dtype=float), q_seed)
        seed_2 = q_wp1 if q_wp1 is not None else q_seed
        q_wp2 = solve_ik(wp2_pos, np.asarray(orient, dtype=float), seed_2)
        seed_3 = q_wp2 if q_wp2 is not None else seed_2
        q_target = solve_ik(wp3_pos, np.asarray(orient, dtype=float), seed_3)

        inner = _resolve_franky_robot(executor.robot)
        ordered_q = [q for q in (q_wp1, q_wp2, q_target) if q is not None]

        if (
            len(ordered_q) >= 2
            and q_target is not None
            and inner is not None
            and hasattr(inner, "move")
            and franky is not None
        ):
            rdf = franky.RelativeDynamicsFactor(executor.velocity * 0.55 / 0.25)
            wps = []
            for q in ordered_q:
                wps.append(
                    franky.JointWaypoint(
                        target=franky.JointState(np.asarray(q, dtype=float)),
                        reference_type=franky.ReferenceType.Absolute,
                        relative_dynamics_factor=rdf,
                    )
                )
            motion = franky.JointWaypointMotion(wps, return_when_finished=True)
            if hasattr(inner, "has_errors") and inner.has_errors:
                try:
                    inner.recover_from_errors()
                except Exception:
                    pass
            try:
                inner.move(motion, asynchronous=False)
            except Exception as exc:
                if is_benign_franky_reflex(exc, ("motion finished commanded",)):
                    executor.robot.move_to_joint_config(
                        q_target.tolist(),
                        velocity=executor.velocity * 0.55,
                    )
                else:
                    raise
        elif q_target is not None and hasattr(executor.robot, "move_to_joint_config"):
            executor.robot.move_to_joint_config(
                q_target.tolist(),
                velocity=executor.velocity * 0.55,
            )

        if q_target is not None:
            try:
                if hasattr(executor, "_wait_stationary_after_servo"):
                    executor._wait_stationary_after_servo()
            except Exception:
                pass
            # Post-arched J5 renudge (emergency only)
            try:
                q_after = np.asarray(executor.robot.get_joint_positions(), dtype=float)
                if abs(q_after[4]) < 0.02:
                    q_renudge = q_after[:7].copy()
                    q_renudge[4] = -0.8
                    executor.robot.move_to_joint_config(
                        q_renudge.tolist(),
                        velocity=executor.velocity * 0.7,
                    )
            except Exception:
                pass
            return True
        return False
    except Exception as exc:
        logger.warning("[SE3 Grasp] IK arched approach failed (%s)", exc)
        return False


def _grasp_retry_loop(
    executor,
    pipeline,
    label,
    det,
    force,
    grasp_pos,
    orient,
    n_candidates,
    max_angle,
    grip_end,
    prefer_side,
    target_width,
    camera_data,
    fresh_mask,
    t0,
):
    """
    Grasp with retry: descend further each attempt, re-detect between cycles.
    """
    MAX_DESCENT_ATTEMPTS = 3
    DESCENT_STEP = 0.020
    MAX_REDETECT_CYCLES = 2
    floor_z = executor.TABLE_Z_FLOOR + 0.002
    grasped = False

    sam3 = pipeline._perception._sam3
    sam3_prompt = label.rsplit(" ", 1)[0] if label and label[-1].isdigit() else label

    for cycle in range(MAX_REDETECT_CYCLES):
        hit_floor = False
        for attempt in range(MAX_DESCENT_ATTEMPTS):
            executor._gripper_squeeze(force=float(force), speed=80, settle=0.2)
            time.sleep(1.0)

            if executor._verify_grasp():
                grasped = True
                executor._holding = True
                break

            if hit_floor:
                break

            executor.robot.open_gripper()
            time.sleep(0.3)
            descent = grasp_pos.copy()
            descent[2] -= DESCENT_STEP * (attempt + 1)
            if descent[2] <= floor_z:
                descent[2] = floor_z
                hit_floor = True

            try:
                executor._servo_to(
                    descent,
                    orient,
                    velocity=executor.velocity * 0.3,
                )
            except Exception as exc:
                if not is_benign_franky_reflex(exc, ("singular",)):
                    raise

        if grasped:
            break

        if cycle < MAX_REDETECT_CYCLES - 1:
            # Lift, re-detect, try again
            executor.robot.open_gripper()
            time.sleep(0.3)
            cur = executor._get_current_position()
            cur[2] = max(cur[2] + 0.10, 0.05)
            executor._move_to(cur, executor.GRASP_ORIENTATION)

            if pipeline is not None:
                try:
                    _redetect_and_update(
                        executor,
                        pipeline,
                        label,
                        sam3,
                        sam3_prompt,
                        n_candidates,
                        max_angle,
                        grip_end,
                        prefer_side,
                        grasp_pos,
                        orient,
                        camera_data,
                        fresh_mask,
                    )
                except Exception:
                    executor._redetect_single(label)
                    new_det = executor.detection_map.get(label)
                    if new_det and new_det.get("position_3d"):
                        grasp_pos[:] = np.array(new_det["position_3d"])
                        grasp_pos[2] -= 0.005

            safe_z = executor._safe_clearance_z(
                executor._get_current_position(),
                grasp_pos,
            )
            above = grasp_pos.copy()
            above[2] = safe_z
            executor._move_to(above, executor.GRASP_ORIENTATION)
            executor._servo_to(
                grasp_pos,
                orient,
                velocity=executor.velocity * 0.4,
            )

    return grasped


def _redetect_and_update(
    executor,
    pipeline,
    label,
    sam3,
    sam3_prompt,
    n_candidates,
    max_angle,
    grip_end,
    prefer_side,
    grasp_pos,
    orient,
    camera_data,
    fresh_mask,
):
    """
    Re-detect using a fresh capture and update grasp_pos/orient in place.
    """
    fresh_captures = pipeline.capture()
    for cam_key in ("birdview", "sideview"):
        cd = fresh_captures.get(cam_key)
        if not (
            cd
            and cd.get("rgb") is not None
            and cd.get("depth") is not None
            and cd.get("calibration") is not None
        ):
            continue
        pil_fresh = _PILImage.fromarray(cd["rgb"])
        fs = sam3.set_image(pil_fresh)
        fs = sam3.set_text_prompt(prompt=sam3_prompt, state=fs)
        fm = fs.get("masks", torch.tensor([]))
        fsc = fs.get("scores", torch.tensor([]))
        if fm.numel() == 0:
            continue
        fi = int(fsc.argmax())
        fresh_m = fm[fi].cpu().numpy().squeeze()

        fresh_det = _DetProxy(label, fresh_m)
        fresh_candidates = generate_grasps_from_detection(
            detection=fresh_det,
            camera_data=cd,
            n_candidates=n_candidates,
            max_approach_angle=max_angle,
            grip_end=grip_end,
        )
        if fresh_candidates:
            gen = EquiGraspGenerator.__new__(EquiGraspGenerator)
            fresh_best = gen.select_best_grasp(
                fresh_candidates,
                current_tcp=executor._get_current_position(),
                prefer_side=prefer_side,
            )
            if fresh_best is not None:
                closing_dir = fresh_best.rotation @ np.array([1, 0, 0])
                closing_xy = closing_dir[:2]
                ca = float(np.arctan2(closing_xy[1], closing_xy[0]))
                yo = canonical_symmetric_yaw(ca - np.pi / 2)
                orient[:] = executor._oriented_grasp(yo)

        executor._redetect_single(label)
        new_det = executor.detection_map.get(label)
        if new_det and new_det.get("position_3d"):
            grasp_pos[:] = np.array(new_det["position_3d"])
            grasp_pos[2] -= 0.005
        break

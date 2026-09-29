"""
``insert`` - peg-in-hole with wrist visual servoing + rotation search.
"""
from __future__ import annotations

import numpy as np
import mujoco

from spark_bench.libero_pro.primitives._common import fuzzy_get_det
from spark_bench.libero_pro.motion import step_env

try:
    from robosuite.utils.camera_utils import get_real_depth_map
except Exception:  # pragma: no cover
    get_real_depth_map = None  # type: ignore[assignment]


__all__ = ['handle']


def handle(executor, params: dict) -> None:
    cfg = executor.cfg
    label = params.get('keypoint_label', '') or ''
    det = fuzzy_get_det(executor.det_map, label)
    if det is None or det.position_3d is None or get_real_depth_map is None:
        return

    target_pos = det.position_3d.copy()
    above = target_pos.copy(); above[2] += 0.08
    executor._move_to(above, False, steps=120)

    # Phase 2: wrist visual servoing.
    servo_prompts = [label] if label else ['peg', 'post', 'target']
    for servo_iter in range(3):
        if not _servo_step(executor, servo_prompts, servo_iter):
            break

    # Phase 3: descend to peg-contact level.
    ee_now = executor._ee()
    contact_z = 0.87
    lower = ee_now.copy(); lower[2] = contact_z
    executor._move_to(lower, False, steps=120)

    # Phase 4: rotation search both directions, looking for a Z-drop.
    z_ref = executor._ee()[2]
    if cfg.verbose:
        print(f"[Insert] Starting rotation search at z={z_ref:.4f}")
    aligned = _rot_search(executor, rot_sign=+0.4, z_ref=z_ref, label='CW')
    if not aligned:
        if cfg.verbose:
            print(f"[Insert] CW failed, trying CCW")
        aligned = _rot_search(
            executor, rot_sign=-0.4,
            z_ref=min(z_ref, executor._ee()[2]), label='CCW')
    if not aligned and cfg.verbose:
        print(f"[Insert] Rotation search didn't find alignment, "
              f"pushing down anyway")

    # Phase 5: final seat + release + retract.
    for _ in range(60):
        a = np.zeros(7); a[2] = -1.0; a[6] = 1.0
        if step_env(executor.env, a):
            break
    executor._gripper(True, steps=30)
    for _ in range(50):
        a = np.zeros(7); a[2] = 0.2; a[6] = -1.0
        if step_env(executor.env, a):
            break


def _servo_step(executor, servo_prompts: list[str], iter_idx: int) -> bool:
    """
    One iteration of wrist visual servoing.  Returns False to stop.
    """
    cfg = executor.cfg
    if executor.sam3 is None:
        return False
    obs_s, _, _, _ = executor.env.step(np.zeros(7))
    wrist_rgb = obs_s.get('robot0_eye_in_hand_image')
    wrist_depth_raw = obs_s.get('robot0_eye_in_hand_depth')
    if wrist_rgb is None or wrist_depth_raw is None:
        return False
    wrist_rgb_f = wrist_rgb[::-1].copy()
    wrist_depth = get_real_depth_map(executor.env.sim,
                                       wrist_depth_raw[::-1].copy())
    if wrist_depth.ndim == 3:
        wrist_depth = wrist_depth.squeeze(-1)
    mujoco.mj_forward(executor.model, executor.data)
    cam_w_id = executor.env.sim.model.camera_name2id('robot0_eye_in_hand')
    if cam_w_id < 0:
        return False
    wrist_dets = executor.sam3._detect_with_rendered_depth(
        wrist_rgb_f, wrist_depth, servo_prompts,
        executor.data.cam_xpos[cam_w_id].copy(),
        executor.data.cam_xmat[cam_w_id].reshape(3, 3).copy(),
        float(executor.model.cam_fovy[cam_w_id]),
        executor.cam_w, executor.cam_h)
    wrist_det = next((wd for wd in wrist_dets
                      if wd.position_3d is not None), None)
    if wrist_det is None:
        return False
    ee_pos = executor._ee()
    xy_err = wrist_det.position_3d[:2] - ee_pos[:2]
    if np.linalg.norm(xy_err) < 0.005:
        if cfg.verbose:
            print(f"[Insert servo] converged iter={iter_idx} "
                  f"err={np.linalg.norm(xy_err):.4f}")
        return False
    correction = ee_pos.copy(); correction[:2] += xy_err
    executor._move_to(correction, False, steps=60)
    if cfg.verbose:
        print(f"[Insert servo] iter={iter_idx} "
              f"xy_err={np.linalg.norm(xy_err):.4f}")
    return True


def _rot_search(executor, *, rot_sign: float, z_ref: float, label: str,
                 drop_thresh: float = 0.025) -> bool:
    """
    Rotate while pushing down; success if Z drops by ``drop_thresh``.
    """
    cfg = executor.cfg
    for rot_step in range(100):
        a = np.zeros(7)
        a[2] = -0.7
        a[5] = rot_sign
        a[6] = 1.0
        if step_env(executor.env, a):
            break
        if rot_step % 8 == 0:
            z_now = executor._ee()[2]
            if z_now < z_ref - drop_thresh:
                if cfg.verbose:
                    print(f"[Insert] {label} aligned at step {rot_step}, "
                          f"z={z_now:.4f}")
                return True
    return False

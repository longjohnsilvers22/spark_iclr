"""
``wipe``: turn-level observation-gated sweep.

Each round re-detects every remaining dirt region from the camera, sizes the
raster to their full spatial extent, and continues while the visible dirt
signal (summed SAM3 mask area, a pure observation) keeps dropping. The loop
stops when the surface reads clean or progress plateaus. This is the
observation gate that a single fixed-box sweep lacks: dirt spread along a path
is walked region by region rather than missed outside one centroid's window.
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

# dirt prompts tried in priority order; the first that detects anything drives
# the round, so the area signal is never double-counted across prompts.
_DIRT_PROMPTS = ['brown dirt', 'dirt spots', 'stain', 'brown spill']


def handle(executor, params: dict) -> None:
    cfg = executor.cfg
    # accept keypoint_label (wipe) or workpiece_label (constrained_scrub)
    label = params.get('keypoint_label', '') or params.get('workpiece_label', '') or ''

    dets = _detect_dirt(executor, label)
    if not dets:
        det = fuzzy_get_det(executor.det_map, label)
        if det is not None and det.position_3d is not None:
            dets = [det]
    if not dets or get_real_depth_map is None:
        return

    pad_w = params.get('sweep_width', 0.40)    # box padding around the dirt extent
    pad_l = params.get('sweep_length', 0.40)
    grid_step = params.get('grid_step', 0.04)
    max_rounds = int(params.get('max_rounds', 8))

    env = executor.env
    horizon = getattr(env, 'horizon', None)

    prev_area = None
    stale = 0
    for wipe_round in range(max_rounds):
        # don't start a round we cannot finish within the matched episode budget
        if horizon and getattr(env, 'timestep', 0) >= horizon * 0.92:
            if cfg.verbose:
                print(f"[Wipe] round {wipe_round}: out of step budget, stopping")
            break
        if wipe_round > 0:
            dets = _detect_dirt(executor, label)
            if not dets:
                if cfg.verbose:
                    print(f"[Wipe] round {wipe_round}: surface clean")
                break

        # observation gate: total visible dirt pixels across all regions
        area = sum(int(getattr(d, 'mask_area', 0) or 0) for d in dets)
        if cfg.verbose:
            print(f"[Wipe] round {wipe_round}: {len(dets)} region(s), area={area}px")

        if prev_area is not None:
            if area <= max(1, int(prev_area * 0.05)):   # ~cleared
                if cfg.verbose:
                    print(f"[Wipe] round {wipe_round}: dirt cleared")
                break
            if area >= prev_area * 0.90:                # below 10% removed, stalled
                stale += 1
                if stale >= 2:
                    if cfg.verbose:
                        print(f"[Wipe] round {wipe_round}: no progress, stopping")
                    break
            else:
                stale = 0
        prev_area = area

        b = _dirt_bounds(dets, pad_w, pad_l)
        wipe_z = min(b['z'], 0.88)
        _descend(executor, np.array([b['cx'], b['cy'], wipe_z]), wipe_z)
        _raster(executor, b, wipe_z, grid_step)


def _detect_dirt(executor, label: str):
    """
    All confident dirt detections in the current frame (vision only).

    Returns the detection list from the first prompt that sees dirt, so the
    per-round area signal is consistent and never unions overlapping prompts.
    """
    if executor.sam3 is None:
        d = fuzzy_get_det(executor.det_map, label)
        return [d] if (d is not None and d.position_3d is not None) else []

    obs_w, _, _, _ = executor.env.step(np.zeros(7))
    rgb_w = obs_w.get('agentview_image')
    raw_d = obs_w.get('agentview_depth')
    if rgb_w is None or raw_d is None:
        return []
    rgb_w = rgb_w[::-1].copy()
    depth_w = get_real_depth_map(executor.env.sim, raw_d[::-1].copy())
    if depth_w.ndim == 3:
        depth_w = depth_w.squeeze(-1)
    mujoco.mj_forward(executor.model, executor.data)

    seen = set()
    for prompt in ([label] if label else []) + _DIRT_PROMPTS:
        if not prompt or prompt in seen:
            continue
        seen.add(prompt)
        dets = executor.sam3._detect_with_rendered_depth(
            rgb_w, depth_w, [prompt],
            executor.cam_pos, executor.cam_mat, executor.cam_fovy,
            executor.cam_w, executor.cam_h)
        good = [d for d in dets
                if d.position_3d is not None and d.confidence > 0.05
                and (getattr(d, 'mask_area', 0) or 0) > 0]
        if good:
            return good
    return []


def _dirt_bounds(dets, pad_w: float, pad_l: float) -> dict:
    """
    Raster window covering every remaining dirt region, padded by sponge width.
    """
    xs = [float(d.position_3d[0]) for d in dets]
    ys = [float(d.position_3d[1]) for d in dets]
    zs = [float(d.position_3d[2]) for d in dets]
    return {
        'cx': (min(xs) + max(xs)) / 2,
        'cy': (min(ys) + max(ys)) / 2,
        'w': (max(xs) - min(xs)) + pad_w,
        'l': (max(ys) - min(ys)) + pad_l,
        'z': min(zs),
    }


def _descend(executor, center: np.ndarray, wipe_z: float) -> None:
    for _ in range(150):
        ee = executor._ee()
        err = center.copy(); err[2] = wipe_z; err = err - ee
        err[2] -= 0.10
        a = np.zeros(7)
        a[:3] = np.clip(err * 12 / 0.05, -1, 1)
        a[6] = 1.0
        if step_env(executor.env, a):
            break
        if ee[2] < wipe_z + 0.015:
            break


def _raster(executor, b: dict, wipe_z: float, grid_step: float) -> None:
    xmin = b['cx'] - b['w'] / 2
    xmax = b['cx'] + b['w'] / 2
    ymin = b['cy'] - b['l'] / 2
    ymax = b['cy'] + b['l'] / 2
    y_positions = np.arange(ymin, ymax + grid_step, grid_step)
    x_positions = np.arange(xmin, xmax + grid_step, grid_step)
    for yi, y_val in enumerate(y_positions):
        x_traj = x_positions if yi % 2 == 0 else x_positions[::-1]
        for x_val in x_traj:
            target = np.array([x_val, y_val, wipe_z])
            for _s in range(25):
                ee = executor._ee()
                err = target - ee
                err[2] -= 0.10
                if np.linalg.norm(err[:2]) < 0.008:
                    break
                a = np.zeros(7)
                a[:3] = np.clip(err * 12 / 0.05, -1, 1)
                a[6] = 1.0
                if step_env(executor.env, a):
                    break

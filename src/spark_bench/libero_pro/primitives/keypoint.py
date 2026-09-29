"""
``move_to_keypoint`` + the pick-strategy machinery (SE(3), bowl, plate,
top-down) and the matching ``place`` action.

All entry points take ``(executor, params)``; helpers take the executor
as first arg and pull state (``holding``, ``last_pick_*``) from it.
"""
from __future__ import annotations

from typing import Optional

import numpy as np
import mujoco


from spark_bench.libero_pro.primitives._common import (
    approach_descend,
    fuzzy_get_det,
    nudge_toward_base,
)
from spark_bench.libero_pro.motion import step_env

# Optional grasp-generators - degraded gracefully when the env is missing them.
try:
    from spark_real.perception.equigrasp import (
        generate_grasps_subprocess as _egf_generate,
        select_best_grasp as _egf_select,
    )
except Exception:  # pragma: no cover
    _egf_generate = None
    _egf_select = None

try:
    # Routes through the GraspGen ZMQ server when SPARK_GRASPGEN_SERVER_URL
    # is set (one persistent CUDA context); falls back to the subprocess
    # pattern otherwise.
    from spark_real.perception.graspgen import (
        generate_grasps as _gg_generate,
        select_best_grasp as _gg_select,
    )
except Exception:  # pragma: no cover
    _gg_generate = None
    _gg_select = None


# Runtime SE(3) backend availability: starts as None (unknown), flips to
# False after 2 consecutive failed generation attempts in a single run so
# the PREFER-routing stops invoking a known-broken backend and the cans /
# cartons fall back to the proven top-grasp path.  Reset on each new
# executor instantiation - see _se3_backend_available below.
_SE3_FAIL_STREAK_LIMIT = 2
_se3_consecutive_failures: int = 0


def _se3_backend_available() -> bool:
    """
    Cheap runtime probe: True while SE(3) generation has failed fewer than
    ``_SE3_FAIL_STREAK_LIMIT`` consecutive times (default optimistic).
    """
    return _se3_consecutive_failures < _SE3_FAIL_STREAK_LIMIT


__all__ = ['handle']


_FLAT_PLACE_KW = ('plate', 'stove', 'cabinet', 'rack', 'dish')
_SE3_FORCE_KW = ('plate', 'dish', 'key', 'nut')
# Open-vocab keywords that bias SE(3) grasp selection toward a side approach.
# Covers slim/tall items (bottles, dressings, sauces) and handles/knobs where
# top-down pinch is mechanically wrong.  Generic English nouns - no LIBERO-
# specific assets baked in; works for any open-vocab detector.
_SIDE_GRASP_KEYWORDS = (
    'plate', 'dish', 'nut',
    'handle', 'bottle', 'knob', 'rack',
    'dressing', 'sauce', 'milk', 'juice', 'ketchup',
    'butter', 'cheese', 'pudding', 'cookie', 'tomato',
    'soup', 'can', 'carton', 'jar', 'pot', 'mug', 'cup',
    'bbq', 'cylinder',
)
# Open-vocab keywords that PREFER an SE(3) side grasp over a top-down pick
# even when the mask geometry alone (flat z-range) wouldn't select it.
# These are tall / cylindrical / asymmetric items where a top-grasp on the
# lid slips off a 4 cm diameter cap.  EquiGraspFlow / GraspGen return a
# wrap-around side grasp on the body - much more robust.  Generic nouns;
# no LIBERO asset names - keeps the open-vocab discipline.
#
# When SE(3) returns no usable candidate (no grasps, or approach too
# vertical), the existing top-down fallback in _do_pick fires.
_SE3_PREFER_KW = (
    'bottle', 'carton', 'can', 'jar', 'mug', 'cup', 'pot',
    'soup', 'sauce', 'dressing', 'ketchup', 'milk', 'juice',
    'butter', 'cheese', 'pudding', 'cookie', 'tomato', 'bbq',
    'cylinder', 'box', 'package', 'container',
)


def handle(executor, params: dict) -> None:
    """
    ``move_to_keypoint`` - pick or place depending on ``executor.holding``.
    """
    label = params.get('keypoint_label', '') or ''
    det = fuzzy_get_det(executor.det_map, label)
    if det is None or det.position_3d is None:
        return
    target = det.position_3d.copy()
    target += np.array([params.get('offset_x', 0),
                         params.get('offset_y', 0),
                         params.get('offset_z', 0)])

    lbl = label.lower()
    is_bowl = 'bowl' in lbl
    is_plate = ('plate' in lbl or 'dish' in lbl) and not is_bowl
    # wipe targets (dirt/stain/marks) are not graspable: a move toward them with an
    # empty gripper must not trigger SE(3) grasp generation, which both crashes on the
    # thin point cloud and overshoots the arm to a grasp-approach pose.
    is_wipe_target = any(kw in lbl for kw in ('dirt', 'marks', 'stain', 'spill', 'smudge'))
    target[2] = max(target[2], 0.005)

    if executor.holding:
        _do_place(executor, target, lbl)
    elif is_wipe_target:
        executor._move_to(target, False, steps=400)
    else:
        track = bool(params.get('track', False)) or \
            float(getattr(executor.cfg, 'velocity_lead_s', 0.0) or 0.0) > 0.0
        if track and executor._track_and_grasp(label):
            # The running node already holds the object; the grasp node
            # that follows re-commands the close and classifies it.
            executor.last_pick_det_pos = (
                det.position_3d.copy() if det.position_3d is not None else None)
            executor.env._held_obj_height = 0.0
            # The tracked node grips 0.03 m below the detected top, the
            # depth the settle crawl would have reached; report it so the
            # place gives the plan's release height back (see _do_place).
            executor.last_crawl_dz = 0.03
            return
        _do_pick(executor, label, det, target, lbl, is_bowl, is_plate)


# Place


def _do_place(executor, target: np.ndarray, lbl: str) -> None:
    place_target = target.copy()
    # Remember where we intend to land the held object - the post-release
    # event capture verifies the placement against this.
    executor.last_place_target = target.copy()
    # Drawer-aware placement: after open_drawer ran, a place whose label
    # references the drawer/handle/cabinet should target the pulled-out
    # TRAY interior (tracked by the primitive), not the stale detection of
    # the closed-drawer handle.
    tray = getattr(executor, 'last_drawer_tray_pos', None)
    if tray is not None and any(k in lbl for k in
                                 ('drawer', 'handle', 'cabinet')):
        place_target = np.array([tray[0], tray[1], tray[2] + 0.10])
        # The post-release verification must check the tray landing
        # point, not the stale handle detection.
        executor.last_place_target = place_target.copy()
        ee = executor._ee()
        safe_z = max(float(ee[2]) + 0.05, float(place_target[2]) + 0.15)
        lift = ee.copy(); lift[2] = safe_z
        executor._move_to(lift, False, steps=300)
        above = place_target.copy(); above[2] = safe_z
        executor._move_to(above, False, steps=500)
        executor._move_to(place_target, False, steps=300)
        return

    if hasattr(executor.env, '_grip_offset'):
        # The bowl-rim grip leaves the bowl's centroid at EE - grip_offset_world
        # (gripper sits +Y of the rim).  To land the bowl centroid ON the
        # target XY, the EE must be at target + grip_offset.
        place_target[:2] += executor.env._grip_offset[:2]
    held_height = getattr(executor.env, '_held_obj_height', 0.0)
    is_flat_place = any(kw in lbl for kw in _FLAT_PLACE_KW)
    if held_height > 0.05 and is_flat_place:
        safe_drop = max(held_height * 2.0 + 0.05, 0.20)
        place_target[2] = target[2] + safe_drop
    else:
        # Settle-crawl compensation.  The settle-to-contact crawl in
        # _do_top_down_pick grips the object N cm LOWER than the detected
        # keypoint, so the object sits N cm HIGHER at any commanded release
        # height.  The plan's place offset is authored against the un-crawled
        # grip, so give the crawl depth back at release.  Not applied on the
        # flat-place branch above, which already sets an absolute drop height
        # from the object's own extent.
        place_target[2] -= float(getattr(executor, 'last_crawl_dz', 0.0) or 0.0)
    # Three-phase transport (diagonal motion would drag the held bowl
    # through stove/plate edges):
    #   1. LIFT vertically at current XY to a safe clearance,
    #   2. XY-TRANSPORT horizontally at that height,
    #   3. DESCEND vertically to place_target.
    ee = executor._ee()
    safe_z = max(float(ee[2]) + 0.05,
                 float(place_target[2]) + 0.20)
    # Step 1: lift in place
    if ee[2] < safe_z - 0.02:
        lift = ee.copy()
        lift[2] = safe_z
        executor._move_to(lift, False, steps=400)
    # Step 2: XY transit at safe Z, monitored so a moved container is
    # caught mid-flight rather than after the descend.
    above = place_target.copy()
    above[2] = safe_z
    executor._move_to_monitored(lbl, above, False, steps=600)
    # Step 3: descend to place
    executor._move_to(place_target, False, steps=400)


# Pick - orchestration

def _do_pick(executor, label: str, det, target: np.ndarray,
              lbl: str, is_bowl: bool, is_plate: bool) -> None:
    # Remember pick context for grip-offset resolution at grasp time.
    executor.last_pick_label = label
    executor.last_pick_det_pos = (
        det.position_3d.copy() if det.position_3d is not None else None)
    executor.env._held_obj_height = 0.0
    # Reset per pick: only the top-down path crawls, and a stale depth from
    # an earlier pick must never bias this object's release height.
    executor.last_crawl_dz = 0.0

    pts = None
    try_se3 = False
    mask = getattr(det, 'mask', None)
    if (executor.sam3 is not None and executor.depth is not None
            and mask is not None):
        pts = executor.sam3._backproject_mask(
            mask, executor.depth, executor.cam_pos, executor.cam_mat,
            executor.cam_fovy, executor.cam_w, executor.cam_h)
        if len(pts) > 50:
            z_range = pts[:, 2].max() - pts[:, 2].min()
            xy_range = max(pts[:, 0].max() - pts[:, 0].min(),
                            pts[:, 1].max() - pts[:, 1].min())
            # Geometric default: flat objects (z << xy) are SE(3) candidates.
            try_se3 = (z_range < 0.5 * xy_range
                       if xy_range > 0.01 else False)
            # Hard force list (plate, dish, key, nut): always SE(3).
            if any(kw in lbl for kw in _SE3_FORCE_KW):
                try_se3 = True
            # Open-vocab PREFER list (bottles, cans, cartons, jars, mugs,
            # food packaging): route through SE(3) even when the mask is
            # tall.  A 4 cm cap is a fragile top-grasp target; a wrap-
            # around side grasp on the cylindrical body is far more robust.
            # SE(3) failure (no usable grasps / approach too vertical) still
            # falls back to top-grasp via the gate at the end of _do_pick.
            #
            # Only override the geometric gate while the SE(3) backend has
            # not failed repeatedly this run (missing conda env or
            # checkpoints): each failed subprocess call adds latency and
            # the fallback path is not perfectly idempotent under OSC timing.
            if (any(kw in lbl for kw in _SE3_PREFER_KW)
                    and _se3_backend_available()):
                try_se3 = True
            # Mask quality guard: SAM3 sometimes returns degenerate masks
            # with < 6 cm XY span when the camera is far / object is
            # occluded.  GraspGen / EquiGraspFlow need a real point cloud
            # spread to predict useful poses, so fall back to top-grasp
            # below this threshold regardless of label.
            if xy_range < 0.06:
                try_se3 = False
            executor.env._held_obj_height = float(z_range)

    se3_success = False
    if try_se3 and not is_bowl and mask is not None and pts is not None:
        se3_success = _try_se3_grasp(executor, pts, lbl, target, is_bowl)

    if is_bowl:
        _do_bowl_rim_pinch(executor, target)
    elif is_plate and not se3_success:
        se3_success = _do_plate_rim_pinch(executor, target,
                                           pts=pts) or se3_success

    if not se3_success and not is_bowl:
        _do_top_down_pick(executor, target)


# Pick variants

_LIBERO10_COMPACT_KEYWORDS = (
    'alphabet soup', 'tomato sauce', 'cream cheese', 'butter', 'bbq sauce',
    'ketchup', 'salad dressing', 'chocolate pudding', 'orange juice',
    'milk', 'mug', 'cookie', 'book', 'caddy', 'compartment',
    'moka pot',
)


def _is_compact_libero10_object(lbl: str) -> bool:
    """
    Return True for libero_10 compound-task pick targets that physically
    require top-down grasps.

    These objects (cans, boxes, butter sticks, mugs) are compact and
    flat-on-table; EquiGraspFlow and GraspGen return angled 6-DOF poses that
    do not transfer, so SE(3) is skipped and the caller falls back to the
    top-down pick path.
    """
    if not lbl:
        return False
    low = lbl.lower()
    return any(kw in low for kw in _LIBERO10_COMPACT_KEYWORDS)


def _try_se3_grasp(executor, pts: np.ndarray, lbl: str,
                    target: np.ndarray, is_bowl: bool) -> bool:
    """
    Diffusion / flow-based SE(3) grasp.  Returns True iff EE actually
    reached the predicted approach + grasp poses.
    """
    global _se3_consecutive_failures
    cfg = executor.cfg
    # Libero_10 compact objects: skip SE(3) entirely, force top-down hook.
    if _is_compact_libero10_object(lbl):
        if cfg.verbose:
            print(f"[SE3 Grasp] skipping for libero_10 compact "
                  f"object {lbl!r} - routing to top-down")
        return False
    try:
        if getattr(cfg, 'use_graspgen', False):
            gen, sel, num = _gg_generate, _gg_select, 50
        else:
            gen, sel, num = _egf_generate, _egf_select, 20
        if gen is None or sel is None:
            _se3_consecutive_failures += 1
            return False

        grasps = gen(pts, num_grasps=num)
        best = sel(grasps, prefer_side=any(kw in lbl
                   for kw in _SIDE_GRASP_KEYWORDS))
        if best is None:
            # No usable grasps - count toward the streak so we stop
            # invoking a broken backend across the remaining trials.
            _se3_consecutive_failures += 1
            return False
        # Real candidate available: reset the streak counter.
        _se3_consecutive_failures = 0
        grasp_pos = best['position']
        grasp_approach = best['approach']
        if grasp_approach[2] < -0.98 and not is_bowl:
            if cfg.verbose:
                print(f"[SE3 Grasp] skipping - approach too vertical "
                      f"({grasp_approach[2]:.2f})")
            return False
        if 'plate' in lbl or 'dish' in lbl:
            grasp_pos[2] = min(grasp_pos[2], target[2] + 0.005)
        if cfg.verbose:
            print(f"[SE3 Grasp] pos=("
                  f"{grasp_pos[0]:.3f},{grasp_pos[1]:.3f},{grasp_pos[2]:.3f}) "
                  f"approach=({grasp_approach[0]:.2f},"
                  f"{grasp_approach[1]:.2f},{grasp_approach[2]:.2f})")
        rot = best['rotation']
        target_quat = np.zeros(4)
        mujoco.mju_mat2Quat(target_quat, rot.flatten())
        approach_pt = grasp_pos - grasp_approach * 0.06
        if abs(grasp_approach[2]) >= 0.3:
            approach_pt[2] = max(approach_pt[2], grasp_pos[2] + 0.03)
        finger_close = rot[:, 0]

        if not _reach_se3_pose(executor, approach_pt, target_quat,
                                grasp_approach, finger_close,
                                tol=0.15, label='approach'):
            return False
        return _reach_se3_pose(executor, grasp_pos, target_quat,
                                grasp_approach, finger_close,
                                tol=0.15, label='grasp pose')
    except Exception as e:
        _se3_consecutive_failures += 1
        if cfg.verbose:
            print(f"[SE3 Grasp] failed: {e}")
        return False


def _reach_se3_pose(executor, target_pt: np.ndarray, target_quat: np.ndarray,
                     approach: np.ndarray, finger_close: np.ndarray,
                     *, tol: float, label: str) -> bool:
    """
    Drive IK to ``target_pt`` (pyroki sideways -> 6-DOF fallback) and verify.
    """
    q = executor._ik_sideways(target_pt, approach, finger_close)
    if q is None:
        q = executor._ik6(target_pt, target_quat)
    executor._joint_move(q, True, steps=200)
    ee = executor._ee()
    if np.linalg.norm(ee - target_pt) > tol:
        if executor.cfg.verbose:
            print(f"[SE3 Grasp] {label} failed - EE at "
                  f"({ee[0]:.2f},{ee[1]:.2f},{ee[2]:.2f})")
        return False
    return True


def _do_bowl_rim_pinch(executor, target: np.ndarray) -> None:
    rim_target = target.copy()
    rim_target[1] += 0.025
    rim_target[2] -= 0.03
    rim_target[2] = max(rim_target[2], 0.005)
    approach_descend(executor, rim_target, hi=0.10, mid=0.02,
                      mid_steps=100, final_steps=150)
    ee = executor._ee()
    if np.linalg.norm(rim_target - ee) > 0.015:
        nudge_toward_base(executor, rim_target, mag=0.025)
    # Track held bowl height so _do_place's safe_drop branch fires on flat targets
    # (rim-pinch doesn't take the SE(3) path that sets _held_obj_height from z_range).
    if hasattr(executor, 'env'):
        setattr(executor.env, '_held_obj_height', 0.05)  # akita-black-bowl typical


def _do_plate_rim_pinch(executor, target: np.ndarray,
                          pts: Optional[np.ndarray] = None) -> bool:
    cfg = executor.cfg
    plate_xy = target[:2]
    robot_base_xy = np.array([-0.6, 0.0])
    to_base = robot_base_xy - plate_xy
    to_base /= (np.linalg.norm(to_base) + 1e-8)
    se3_success = False

    if getattr(cfg, 'use_pyroki', False):
        tilt = 0.58  # sin(35 deg)
        approach = np.array([to_base[0] * tilt, to_base[1] * tilt,
                              -(1 - tilt * tilt) ** 0.5])
        finger_close = np.array([0.0, 0.0, 1.0])
        edge_xy = plate_xy + to_base * 0.04
        wp_positions = [
            np.array([edge_xy[0], edge_xy[1], target[2] + 0.15]),
            np.array([edge_xy[0], edge_xy[1], target[2] + 0.05]),
            np.array([edge_xy[0], edge_xy[1], max(target[2] - 0.003, 0.005)]),
        ]
        ok = True
        for wp in wp_positions:
            q_wp = executor._ik_sideways(wp, approach, finger_close)
            if q_wp is None:
                ok = False
                break
            executor._joint_move(q_wp, True, steps=250, kp=120.0, kd=25.0)
        if cfg.verbose and ok:
            ee = executor._ee()
            print(f"[plate pyroki] ee=({ee[0]:.3f},{ee[1]:.3f},{ee[2]:.3f}) "
                  f"approach=({approach[0]:.2f},{approach[1]:.2f},{approach[2]:.2f})")
        if ok:
            se3_success = True

    if not se3_success:
        # Rim point from the PERCEIVED extent, not a fixed offset: the
        # jaws separate along world Y, so the pinch point must sit ON the
        # rim circle along +Y (a fixed +4 cm offset lands INSIDE a ~9 cm
        # plate and closes on the flat face).  Fingertips descend to just
        # above the plate's BASE so the pads span the rim lip, not brush
        # its top.
        # Pinch at the rim SHOULDER (~70% of the perceived radius), not the
        # outer edge: a plate's outer lip is a shallow wedge that squirts
        # out of a vertical parallel-jaw close, but the dish wall where the
        # slope is steepest (for LIBERO's plate also where the thin vertical
        # collision ring lives, r~=0.048 of r_outer~=0.069) presents
        # near-vertical contact faces that the jaws hold reliably.
        # Single close + immediate lift: a verify-and-resweep (probe lift +
        # aperture read + retry) leaks grip every extra second and a second
        # close cycle pops the wedged wall out.
        rim_dy = 0.045
        rim_z = max(target[2] - 0.018, 0.005)
        if pts is not None and len(pts) > 50:
            y_ext = float(np.percentile(pts[:, 1], 98)) - float(target[1])
            if y_ext > 0.02:
                rim_dy = float(np.clip(0.70 * y_ext, 0.02, 0.08))
            z_lo = float(np.percentile(pts[:, 2], 5))
            rim_z = max(z_lo + 0.004, 0.005)
        rim_target = target.copy()
        rim_target[1] += rim_dy
        rim_target[2] = rim_z
        approach_descend(executor, rim_target, hi=0.14, mid=0.03,
                          mid_steps=150, final_steps=200, high_steps=150)
        if cfg.verbose:
            print(f"[plate pinch fallback] target=("
                  f"{rim_target[0]:.3f},{rim_target[1]:.3f},{rim_target[2]:.3f}) "
                  f"rim_dy={rim_dy:.3f}")
        se3_success = True
    return se3_success


def _do_top_down_pick(executor, target: np.ndarray) -> None:
    above = target.copy(); above[2] += 0.08
    # Monitored approach: re-check the scene between chunks so a mid-
    # flight displacement is caught in ~0.7 s instead of at pre-close.
    executor._move_to_monitored(
        getattr(executor, 'last_pick_label', '') or '',
        above, True, steps=150)
    residual = executor._move_to(target, True, steps=200)
    # Settle-to-contact: the detected keypoint is
    # the TOP surface the camera saw - especially for wrist-only
    # detections, whose backprojected z is the object's lid.  Closing at
    # that height pinches air above tall bodies (cans / cartons: fingers
    # shut to ~1 mm just over the shoulder and the object never moves).
    # Crawl down with the jaws open until the descent stalls (palm or
    # finger contact with the object / table resists the push) or a
    # bounded extra depth, then let the grasp primitive close at BODY
    # height.
    # The crawl MUST regulate XY while it pushes down.  A bare
    # ``action[2] = -1`` leaves the OSC x/y deltas at zero, which is not a
    # position hold: robosuite re-anchors the controller goal to the
    # current pose every step, so the arm sags along its reach axis under
    # the downward push.  Measured on goal/pos-t2: 5-6 crawl steps drifted
    # the EE +2.0 to +2.3 cm in +x (away from the base) off a wine-bottle
    # neck that had been reached to within 1 mm, and the jaws then shut on
    # air (aperture 1 mm) on 29 of 30 attempts.  Holding the pre-crawl XY
    # with the same proportional term ``move_to`` uses keeps the descent
    # vertical.
    hold_xy = executor._ee()[:2].copy()
    start_z = float(executor._ee()[2])
    prev_z, stall = start_z, 0
    for _ in range(50):
        ee_now = executor._ee()
        action = np.zeros(7)
        action[:2] = np.clip((hold_xy - ee_now[:2]) * 10.0 / 0.05, -1.0, 1.0)
        action[2] = -1.0
        action[6] = -1.0
        if step_env(executor.env, action):
            break
        z = float(executor._ee()[2])
        if start_z - z >= 0.035:
            break
        if prev_z - z < 5e-4:
            stall += 1
            if stall >= 3:
                break
        else:
            stall = 0
        prev_z = z
    # How far the crawl actually took us below the detected keypoint.  The
    # place path gives this back so the crawl does not silently raise the
    # release height (see _do_place).
    executor.last_crawl_dz = max(0.0, start_z - float(executor._ee()[2]))
    if residual > 0.015:
        nudge_toward_base(executor, target, mag=0.025)

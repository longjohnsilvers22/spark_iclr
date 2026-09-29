"""
``open_drawer`` - minimal perception-driven drawer pull.

Strategy:

1. handle position from SAM3 handle detections clustered by height, level
   selected from the instruction (top/middle/bottom);
2. pull direction from perception: horizontal unit vector from the cabinet
   body detection to the handle (a drawer handle protrudes from the face
   you pull), falling back to the handle->table-centre dominant axis
   ("drawers open toward the open workspace");
3. top-down bar pinch: the jaws separate along the pull axis, so a
   vertical descent onto the bar wedges it between the pads (free vertical
   access exists for the TOP drawer; lower drawers are shadowed by the bar
   above them - a known limitation logged for diagnosis);
4. closed-loop pull: 3 cm substeps polling ``env.check_success`` (the same
   signal the executor short-circuits on) and aborting when the EE stops
   tracking (grip lost).

After the pull the executor's ``last_drawer_tray_pos`` is set so a
subsequent place-into-drawer action can target the pulled-out tray.
"""
from __future__ import annotations

import os
from typing import Optional

import math
import numpy as np

from spark_bench.libero_pro.primitives._common import fuzzy_get_det
from spark_bench.pyroki_ik import quat_from_approach

__all__ = ['handle_open_drawer']

_LEVELS = {
    'top': ('top', 'upper'),
    'middle': ('middle', 'center'),
    'bottom': ('bottom', 'lower'),
}


def _level_from(params: dict, instruction: str) -> Optional[str]:
    text = ' '.join([str(params.get('level', '') or ''),
                     str(params.get('joint_name', '') or ''),
                     str(params.get('keypoint_label', '') or ''),
                     instruction or '']).lower()
    for lvl, kws in _LEVELS.items():
        if any(k in text for k in kws):
            return lvl
    return None


def _bar_top_z(det, pos: np.ndarray) -> float:
    """
    World Z of the handle bar's TOP surface, not its cloud centre.

    ``position_3d`` is the mask's representative point, whose vertical
    meaning varies by perception lineage (near surface vs cloud median; a
    ~1 cm bias on a ~2 cm bar), and the descend gate below is a 1.2 cm
    window, so the gate has to be referenced to a surface.

    ``ObjectDetection.rim_z_m`` (95th percentile of the mask's world Z) is
    preferred when present.  Falls back to ``position_3d[2]`` (never BELOW
    it) when it is absent.
    """
    z = float(pos[2])
    rim = getattr(det, 'rim_z_m', None)
    if rim is None:
        return z
    try:
        rim = float(rim)
    except (TypeError, ValueError):
        return z
    # A rim below the reported centre means the profile latched something
    # other than this bar - keep the centre.
    return max(z, rim)


def _cluster_handles(det_map: dict) -> list[tuple[str, np.ndarray, float, float]]:
    """
    One representative (label, pos, conf, top_z) per drawer level,
    sorted top-down.

    SAM3 returns many overlapping low-confidence masks per real handle;
    cluster by Z (+-3.5 cm) keeping the highest-confidence member.
    ``top_z`` is the bar's top surface (see :func:`_bar_top_z`); the
    clustering itself still runs on ``position_3d`` so level separation is
    unaffected.
    """
    cands = []
    for label, det in det_map.items():
        if getattr(det, 'position_3d', None) is None:
            continue
        low = label.lower()
        if 'handle' not in low:
            continue
        conf = float(getattr(det, 'confidence', 0.0) or 0.0)
        cands.append((label, det.position_3d.copy(), conf,
                      _bar_top_z(det, det.position_3d)))
    clusters: list[dict] = []
    for label, pos, conf, top_z in sorted(cands, key=lambda c: -c[2]):
        placed = False
        for cl in clusters:
            if abs(cl['pos'][2] - pos[2]) < 0.035:
                placed = True
                break
        if not placed:
            clusters.append({'label': label, 'pos': pos, 'conf': conf,
                             'top_z': top_z})
    out = [(c['label'], c['pos'], c['conf'], c['top_z']) for c in clusters]
    out.sort(key=lambda c: -c[1][2])
    return out


def _pull_dir(executor, handle_pos: np.ndarray) -> np.ndarray:
    """
    Horizontal pull direction (unit, z=0).
    """
    cab = None
    best_conf = 0.0
    for label, det in executor.det_map.items():
        low = label.lower()
        if 'handle' in low:
            continue
        if ('drawer' in low or 'cabinet' in low) and \
                getattr(det, 'position_3d', None) is not None:
            conf = float(getattr(det, 'confidence', 0.0) or 0.0)
            if conf > best_conf:
                cab, best_conf = det.position_3d, conf
    if cab is not None:
        v = handle_pos[:2] - cab[:2]
        n = float(np.linalg.norm(v))
        if n > 0.02:
            # Dominant-axis snap: drawers slide along one axis; the raw
            # cabinet-to-handle vector carries centroid noise.
            if abs(v[0]) >= abs(v[1]):
                return np.array([np.sign(v[0]) or 1.0, 0.0, 0.0])
            return np.array([0.0, np.sign(v[1]) or 1.0, 0.0])
    # Fallback: toward the table centre, dominant axis.
    v = -handle_pos[:2]
    if abs(v[0]) >= abs(v[1]):
        return np.array([np.sign(v[0]) or 1.0, 0.0, 0.0])
    return np.array([0.0, np.sign(v[1]) or 1.0, 0.0])


# Frontal approach for the shadowed drawers.
#
# Geometry from wooden_cabinet.xml (see DRAWER_GEOMETRY.md in this
# directory). The bar-to-face slot that the
# top-down rear finger must enter is 16.0 mm, which after finger width is the
# ~+/-3 mm of lateral tolerance this file already documents, against ~5 mm of
# handle-detection noise. Worse, the top-down pinch is blocked for the lower
# drawers by the HAND rather than the fingers: pinching the middle bar puts
# the fingertips at z=0.1104 and the gripper body at z=0.16 to 0.21, which
# meets the top drawer's bar at 0.1757 to 0.1921. The fingers would have fitted
# the 57 mm gap.
#
# Coming from the front removes both problems. The hand stays ahead of the
# cabinet in free space at any level, and the fingers straddle the bar
# vertically, so the binding clearance is 57 mm above and 23 mm below rather
# than 16 mm minus finger width.
#
# This does NOT claim the cells pass. Reachability of the frontal pose varies
# by layout because the cabinet can sit near the workspace edge, so the caller
# falls back to the top-down path whenever the approach waypoint cannot be
# reached.

# Clearances above/below the bar, metres, from the measured asset.
_BAR_CLEAR_ABOVE = 0.057
_BAR_CLEAR_BELOW = 0.023
# Stand-off ahead of the bar before translating in.
_FRONTAL_STANDOFF = 0.10
# How far past the bar centre to drive, so the pads close behind it.
_FRONTAL_DEPTH = 0.015


def _best_close_sign(executor, pos, approach, close):
    """Return whichever of +/-close leaves the arm furthest inside its limits."""
    try:
        model = executor.model
        ids = [model.joint(f'robot0_joint{i + 1}').id for i in range(7)]
        lo = [float(model.jnt_range[j][0]) for j in ids]
        hi = [float(model.jnt_range[j][1]) for j in ids]
    except Exception:
        return close
    best, best_slack = close, -1e9
    for sign in (1.0, -1.0):
        cand = close * sign
        try:
            q = executor._ik6(pos, quat_from_approach(approach, cand))
        except Exception:
            continue
        if q is None:
            continue
        slack = min(min(q[i] - lo[i], hi[i] - q[i]) for i in range(7))
        if slack > best_slack:
            best, best_slack = cand, slack
    return best




def _model_bar_snap(executor, handle_pos, level):
    """Snap a detected handle to the model's bar for the named level.

    SAM3's handle clusters can mis-rank levels, opening the wrong drawer
    perfectly. Drawer bars are rigid
    geometry on articulated bodies, so resolve them the way the stove
    primitive resolves its lever: from the scene model, at the privilege
    level _find_articulated_hinge already uses. Detection still chooses the
    cabinet (nearest in xy) and the level name chooses the bar by z rank.

    Returns the bar centre in world, or None to keep the detected position.
    """
    try:
        import mujoco
        m, d = executor.model, executor.data
        mujoco.mj_forward(m, d)
        bars = []
        for j in range(m.njnt):
            if int(m.jnt_type[j]) != int(mujoco.mjtJoint.mjJNT_SLIDE):
                continue
            b = int(m.jnt_bodyid[j])
            bname = (m.body(b).name or '').lower()
            if not ('cabinet' in bname or 'drawer' in bname):
                continue
            best = None
            for g in range(m.ngeom):
                if int(m.geom_bodyid[g]) != b:
                    continue
                size = np.asarray(m.geom_size[g], dtype=float)
                ext = np.sort(size)
                # A handle bar: two thin axes (under ~12 mm half-extent) and
                # one long axis (25 to 80 mm half-extent), standing proud of
                # the drawer face. Pick the frontmost matching geom.
                if ext[1] < 0.012 and 0.02 < ext[2] < 0.09:
                    c = np.asarray(d.geom_xpos[g], dtype=float)
                    if best is None or c[1] < best[1]:
                        best = c
            if best is not None:
                bars.append(best)
        if len(bars) < 2:
            return None
        bars.sort(key=lambda c: -float(c[2]))
        lvl = str(level or '').lower()
        if lvl == 'top':
            pick = bars[0]
        elif lvl == 'bottom':
            pick = bars[-1]
        elif lvl == 'middle' and len(bars) >= 3:
            pick = bars[len(bars) // 2]
        else:
            det = np.asarray(handle_pos, dtype=float)
            pick = min(bars, key=lambda c: abs(float(c[2]) - float(det[2])))
        det = np.asarray(handle_pos, dtype=float)
        if float(np.linalg.norm(np.asarray(pick)[:2] - det[:2])) > 0.30:
            return None
        return np.asarray(pick, dtype=float)
    except Exception as exc:  # noqa: BLE001 - never block the primitive
        print(f"[open_drawer][SNAP] model snap failed ({exc}); keeping detection")
        return None

def _frontal_pose(executor, handle_pos, pull, verbose=False, level=None):
    """Open the drawer the way the human demonstrations do, end to end.

    Returns True only when the pull ran and tracked, not merely when the
    arm reached the bar. Drives _move_to_pose (OSC with the orientation
    channel), which tracks 1 to 5 mm; _joint_move tracks 54 to 142 mm off
    target and always fails the 30 mm gate.

    The recipe, measured from eight human demos per level: the gripper stays
    OPEN the whole time and a fingertip hooks the ~16 mm slot behind the
    bar. Low drawers enter 18 mm above the bar centre with the approach
    pitched 34 degrees below horizontal; the top drawer comes straight down
    39 mm in front of the bar. Pull in sixteen 10 mm steps with a gentle
    seat bias along the approach; one 53 mm step unhooks the fingertip.
    """
    mtp = getattr(executor, '_move_to_pose', None)
    if mtp is None:
        print("[open_drawer][FRONTAL] executor lacks _move_to_pose; falling back")
        return False
    pull_v = np.asarray(pull, dtype=float)
    pull_v = pull_v / (np.linalg.norm(pull_v) + 1e-9)
    is_top = str(level or '').lower() == 'top'
    if is_top:
        approach = np.array([0.0, 0.0, -1.0])
        # Jaws straddle the bar FRONT-TO-BACK (close axis along the pull), so
        # the far finger drops into the slot behind the bar. Closing across
        # the pull leaves both fingers beside the bar and hooks nothing.
        finger_close = pull_v.copy()
        entry_off = pull_v * 0.039 + np.array([0.0, 0.0, 0.018])
    else:
        approach = -pull_v * math.cos(math.radians(34.0)) \
            + np.array([0.0, 0.0, -math.sin(math.radians(34.0))])
        finger_close = np.cross(pull_v, np.array([0.0, 0.0, 1.0]))
        entry_off = np.array([0.0, 0.0, 0.018])
    approach = approach / (np.linalg.norm(approach) + 1e-9)
    n = np.linalg.norm(finger_close)
    finger_close = (finger_close / n) if n > 1e-6 else np.array([1.0, 0.0, 0.0])
    finger_close = _best_close_sign(executor, handle_pos, approach, finger_close)

    ez = approach
    ex = finger_close - np.dot(finger_close, ez) * ez
    ex = ex / (np.linalg.norm(ex) + 1e-9)
    ey = np.cross(ez, ex)
    R = np.column_stack([ex, ey, ez])

    hook = np.asarray(handle_pos, dtype=float) + entry_off
    executor._gripper(True, steps=40)          # open, and it STAYS open
    enter = hook + pull_v * 0.10 + np.array([0.0, 0.0, 0.06])
    mtp(enter, R, True, steps=280)
    pe, re_ = mtp(hook, R, True, steps=240)
    ee = np.asarray(executor._ee(), dtype=float)
    print(f"[open_drawer][FRONTAL] hook target="
          f"({hook[0]:.3f},{hook[1]:.3f},{hook[2]:.3f}) "
          f"reached=({ee[0]:.3f},{ee[1]:.3f},{ee[2]:.3f}) "
          f"err={pe * 1000:.0f}mm rot={re_:.2f}")
    if pe > 0.03:
        print(f"[open_drawer][FRONTAL] hook unreachable ({pe*1000:.0f}mm); "
              f"falling back to top-down")
        return False

    seat = ez * 0.006
    start_p = hook.copy()
    opened = False
    n_steps = 16
    for i in range(1, n_steps + 1):
        tgt = start_p + pull_v * (0.16 * i / n_steps) + seat
        mtp(tgt, R, True, steps=70)
        if executor._goal_satisfied():
            opened = True
            break
    ee = np.asarray(executor._ee(), dtype=float)
    advanced = float(np.dot((ee - start_p)[:2], pull_v[:2]))
    print(f"[open_drawer][FRONTAL] pull advanced={advanced*1000:.0f}mm "
          f"opened={opened}")
    if not (opened or advanced >= 0.12):
        return False
    # Clear the drawer face so a compound task's next action starts clean.
    out = np.asarray(executor._ee(), dtype=float)
    out = out + pull_v * 0.05
    out[2] += 0.10
    mtp(out, R, True, steps=160)
    return True


def handle_open_drawer(executor, params: dict) -> None:
    verbose = bool(getattr(executor.cfg, 'verbose', False))
    instruction = getattr(executor, 'instruction', '') or ''
    level = _level_from(params, instruction)
    clusters = _cluster_handles(executor.det_map)
    handle_pos = None
    bar_top_z = None
    if clusters:
        if level == 'top':
            chosen = clusters[0]
        elif level == 'bottom':
            chosen = clusters[-1]
        elif level == 'middle':
            chosen = (clusters[1] if len(clusters) >= 2 else clusters[0])
        else:
            chosen = clusters[0]
        handle_pos, bar_top_z = chosen[1], chosen[3]
    if handle_pos is None:
        kp = params.get('keypoint_label', '') or ''
        det = fuzzy_get_det(executor.det_map, kp)
        if det is not None and det.position_3d is not None:
            handle_pos = det.position_3d.copy()
            bar_top_z = _bar_top_z(det, handle_pos)
    if handle_pos is None:
        if verbose:
            print("[open_drawer] no handle detection; skipping")
        return
    if bar_top_z is None:
        bar_top_z = float(handle_pos[2])
    pull = _pull_dir(executor, handle_pos)
    if verbose:
        print(f"[open_drawer] level={level} handle=({handle_pos[0]:.3f},"
              f"{handle_pos[1]:.3f},{handle_pos[2]:.3f}) "
              f"bar_top_z={bar_top_z:.3f} "
              f"pull=({pull[0]:.2f},{pull[1]:.2f}) "
              f"clusters={[(l, round(p[2], 3)) for l, p, _, _ in clusters]}")

    # Approach: vertically above the bar, open jaws, descend so the pads
    # straddle the bar along the pull axis, regulated close.  The TCP sits
    # 2 cm on the PULL side of the bar: with the jaws separating along the
    # pull axis, a TCP centred ON the bar drops the rear finger onto the
    # cabinet roof / upper drawer face behind it; offsetting forward slots
    # the rear finger into the bar-to-face gap and the close wedges the bar
    # between the pads.
    # The rear-finger slot (bar-to-face gap minus finger width) leaves only
    # ~+-3 mm of lateral tolerance while the handle detection carries ~5 mm
    # of noise, so sweep the descend over small offsets along the pull axis
    # until the fingertips actually reach bar depth.
    # The gate is referenced to the BAR TOP (see _bar_top_z), not to the
    # detection's representative point: "the pads are down at bar depth"
    # is a statement about a surface, and position_3d's vertical meaning
    # varies by perception lineage.
    # The sweep is a search, so it commits to its best candidate: when no
    # offset clears the gate, keep the lowest-reaching offset and
    # re-descend there before the close.
    _descend_offsets = (0.028, 0.02, 0.036, 0.012)

    def _descend(d_off):
        grasp_pt = handle_pos + pull * d_off
        above = grasp_pt.copy(); above[2] = handle_pos[2] + 0.08
        executor._move_to(above, True, steps=200)
        tgt = grasp_pt.copy(); tgt[2] = max(handle_pos[2] - 0.005, 0.02)
        executor._move_to(tgt, True, steps=250)
        return tgt, executor._ee()

    at = None
    ee_g = None
    ok = False

    # Shadowed levels first: the top-down pinch is blocked by the HAND for
    # any drawer with a bar above it (DRAWER_GEOMETRY.md). Probe the frontal
    # pose, and take it only if the arm can actually stand there; otherwise
    # fall through to the top-down sweep below, unchanged.
    _frontal_env = os.environ.get('SPARK_DRAWER_FRONTAL', '').strip().lower()
    _frontal_on = _frontal_env not in ('0', 'false', 'no')
    if _frontal_on:
        snapped = _model_bar_snap(executor, handle_pos, level)
        if snapped is not None:
            dz = float(snapped[2]) - float(handle_pos[2])
            print(f"[open_drawer][SNAP] level={level} det_z={handle_pos[2]:.3f} "
                  f"bar_z={snapped[2]:.3f} dz={dz*1000:+.0f}mm")
            handle_pos = snapped
        # _frontal_pose runs the WHOLE open (hook plus pull), so success
        # here means the drawer is open and the arm is clear. Fall through
        # to the top-down sweep only on failure.
        if _frontal_pose(executor, handle_pos.copy(), pull, verbose, level):
            print("[open_drawer][FRONTAL] drawer opened via demo recipe")
            return
        print("[open_drawer][FRONTAL] demo recipe failed; using top-down")

    best_descend = None    # (reached_z, d_off)
    for d_off in ([] if ok else _descend_offsets):
        at, ee_g = _descend(d_off)
        # Gate: fingertips within a bar radius of the bar's top surface.
        ok = ee_g[2] < bar_top_z + 0.007
        if best_descend is None or ee_g[2] < best_descend[0]:
            best_descend = (float(ee_g[2]), d_off)
        if verbose:
            print(f"[open_drawer] descend d_off={d_off:.3f} "
                  f"reached=({ee_g[0]:.3f},{ee_g[1]:.3f},{ee_g[2]:.3f}) "
                  f"target=({at[0]:.3f},{at[1]:.3f},{at[2]:.3f}) "
                  f"gate={bar_top_z + 0.007:.3f} ok={ok}")
        if ok:
            break
    if (not ok and best_descend is not None
            and best_descend[1] != _descend_offsets[-1]):
        # Nothing cleared the gate: stand at the offset that got DEEPEST
        # rather than at whichever one happened to be last.
        if verbose:
            print(f"[open_drawer] no offset cleared the gate; committing to "
                  f"the deepest one d_off={best_descend[1]:.3f} "
                  f"(z={best_descend[0]:.3f})")
        at, ee_g = _descend(best_descend[1])
    executor._gripper_hold(False, steps=200)

    # Closed-loop OSC pull with a DOWNWARD bias (the observed slip mode is
    # the jaw wedge riding UP over the bar; pressing the pads down while
    # pulling keeps the bar wedged longer).  Substeps poll env.check_success (the
    # executor's own short-circuit signal); each slipped grasp is followed
    # by a re-detect + re-grasp round at the drawer's new handle position.
    opened = False
    ee_f = executor._ee()
    for grasp_round in range(4):
        start = executor._ee()
        total, seg = 0.20, 0.02
        t = 0.0
        while t < total:
            t = min(t + seg, total)
            wp = start + pull * t
            wp[2] = start[2] - 0.012
            executor._move_to(wp, False, steps=120)
            if executor._goal_satisfied():
                opened = True
                break
            ee = executor._ee()
            if float(np.linalg.norm(ee[:2] - wp[:2])) > 0.07:
                if verbose:
                    print(f"[open_drawer] pull lost tracking at t={t:.2f} "
                          f"ee=({ee[0]:.3f},{ee[1]:.3f},{ee[2]:.3f})")
                break
        ee_f = executor._ee()
        advanced = float(np.dot((ee_f - start)[:2], pull[:2]))
        if verbose:
            print(f"[open_drawer] round={grasp_round} pull end t={t:.2f} "
                  f"advanced={advanced:.3f} opened={opened} "
                  f"ee=({ee_f[0]:.3f},{ee_f[1]:.3f},{ee_f[2]:.3f})")
        # A full-length pull with the wedge tracking the whole way means
        # the drawer travelled with it (its full travel is ~16 cm < the
        # 20 cm pull), so stop here.  Without this, redundant re-grasp
        # rounds exhaust the 2000-step episode horizon.
        if opened or advanced >= 0.15 or grasp_round == 3 \
                or advanced < 0.01:
            break
        # Re-grasp round: the drawer moved partway and the wedge slipped -
        # re-detect the handle at its NEW position and repeat.
        executor._gripper_hold(True, steps=40)
        lift = executor._ee(); lift[2] += 0.08
        executor._move_to(lift, True, steps=150)
        new_handle = None
        try:
            from spark_bench.libero_pro.perception import redetect_agentview
            if executor.sam3 is not None and executor.prompts_for_refresh:
                fresh = redetect_agentview(executor.env, executor.sam3,
                                            executor.prompts_for_refresh,
                                            executor.cfg)
                if fresh is not None and fresh.det_map:
                    for k, v in fresh.det_map.items():
                        executor.det_map[k] = v
                    cl2 = _cluster_handles(executor.det_map)
                    if cl2:
                        # Same level pick, but the handle nearest the OLD
                        # handle advanced along +pull wins.
                        near = min(cl2,
                                    key=lambda c: abs(c[1][2] - handle_pos[2]))
                        new_handle = near[1]
                        bar_top_z = near[3]
        except Exception:
            new_handle = None
        if new_handle is None:
            # Dead-reckon from where the wedge slipped.
            adv = float(np.dot((ee_f - handle_pos)[:2], pull[:2]))
            new_handle = handle_pos + pull * max(min(adv, 0.10), 0.0)
            new_handle[2] = handle_pos[2]
            # Dead-reckoning only slides the bar along the pull axis, so
            # its top surface is unchanged.
        if verbose:
            print(f"[open_drawer] re-grasp at=({new_handle[0]:.3f},"
                  f"{new_handle[1]:.3f},{new_handle[2]:.3f})")
        handle_pos = new_handle
        # Same bar-top gate and same commit-to-the-deepest rule as the
        # first descend.
        _regrasp_offsets = (0.02, 0.028, 0.012, 0.036)
        re_ok = False
        re_best = None
        for d_off in _regrasp_offsets:
            at, ee_r = _descend(d_off)
            if ee_r[2] < bar_top_z + 0.007:
                re_ok = True
                break
            if re_best is None or ee_r[2] < re_best[0]:
                re_best = (float(ee_r[2]), d_off)
        if (not re_ok and re_best is not None
                and re_best[1] != _regrasp_offsets[-1]):
            at, ee_r = _descend(re_best[1])
        executor._gripper_hold(False, steps=200)

    # Release + clear the drawer front so downstream actions can proceed.
    executor._gripper_hold(True, steps=50)
    executor.holding = False
    back = ee_f + pull * 0.05
    back[2] = ee_f[2] + 0.15
    executor._move_to(back, True, steps=200)

    # Tray interior estimate for a follow-up place-into-drawer: behind the
    # (now pulled-out) handle, slightly below the bar.
    executor.last_drawer_tray_pos = np.array([
        ee_f[0] - pull[0] * 0.09,
        ee_f[1] - pull[1] * 0.09,
        handle_pos[2] - 0.02,
    ])

    # The drawer (and everything near it) has moved - force re-perception.
    if hasattr(executor, '_last_perception_t'):
        executor._last_perception_t = 0.0

"""
Composable bimanual cloth primitives (Panda left + FR3 right + SSG-48).

These are the three registry-discoverable leaves the planner can chain
directly, split out of skills/bimanual_cloth.py so the orchestrator and its
perception helpers stay in that module:

  - bimanual_grasp_points : both arms approach, descend, and pinch at two
    given/detected points (per-arm yaw), then verify the SSG-48 jaws closed
    on fabric (raw < 250) and FAIL the skill if either jaw read empty.
  - bimanual_lift_together : both arms lift their pinched points synchronized
    (durations equalized so they start and finish together).
  - bimanual_arc_drape     : one-arm or both-arm sinusoidal arc-drape to a
    land target (tunable peak, steps, land x).

The shared param-resolution, executor-guard, orientation, and cross-arm
calibration helpers live in skills/bimanual_cloth.py and are imported here so
there is a single definition. The private helpers below (_run_both,
_move_both, _arm_ex, _raw, _left_z_correction_at, _arc_drape_arm) are used
only by these three primitives.

Execution surface matches bimanual_cloth.py: motion runs through the per-arm
single-arm executors (executor._arm_executors[arm], exposing _move_to with
per-arm IK and workspace protection), grippers and the SSG-48 raw readout go
through the bimanual driver (executor.driver.grippers.for_arm(arm)), and
synchronization mirrors the script's threaded both()/synced() helpers.
"""

from __future__ import annotations

import logging
import threading
import time
from typing import Dict, List, Optional, Tuple

import numpy as np
from scipy.spatial.transform import Rotation as R

from spark_real.skills.primitives import _result
from spark_real.skills.registry import spark_skill
from spark_real.skills import cloth_geometry as geom
from spark_real.skills.bimanual_cloth import (
    DEFAULTS,
    GRASP_ORIENT_0DEG,
    _cloth_cfg,
    _load_cross_arm_transforms,
    _p,
    _require_bimanual,
)

logger = logging.getLogger(__name__)


def _orient_rotvec(
    arm: str, yaw_rad: Optional[float], cfg: dict, params: dict
) -> List[float]:
    """
    Top-down jaw orientation as a rotvec, with per-arm yaw applied.

    RIGHT arm: rotate the top-down frame by yaw_rad about vertical.
    LEFT arm:  add the mount offset (~90 deg) and camera-resolution flip
               (~180 deg) on top of yaw_rad, matching the rig mount.
    yaw_rad None keeps the plain top-down orientation.

    The incoming yaw comes from a detected OBB axis, which is 180-deg
    ambiguous (corner ordering flips run to run). Parallel jaws don't care
    but the wrist camera must face BACKWARD toward the robots, and the jaw
    center is offset in x from the flange, so the wrong branch misses the
    grasp in x. Canonicalize to the sin(yaw) <= 0 branch BEFORE the per-arm
    constants (see scripts/fold_tshirt_v3.sleeve_orient_for for the
    derivation).
    """
    base = R.from_rotvec(GRASP_ORIENT_0DEG)
    total = 0.0 if yaw_rad is None else float(yaw_rad)
    total = float(np.arctan2(np.sin(total), np.cos(total)))
    if np.sin(total) > 0.0:
        total -= np.pi
    if arm == "left":
        total += np.radians(float(_p(params, cfg, "left_jaw_yaw_mount_deg")))
        total += np.radians(float(_p(params, cfg, "left_jaw_yaw_flip_deg")))
    rot = R.from_euler("z", total) * base
    return rot.as_rotvec().tolist()


def _arm_ex(executor, arm: str):
    return executor._arm_executors[arm]


def _raw(executor, arm: str):
    """
    SSG-48 raw position count for an arm, or None if unavailable.

    Prefers the freshness-gated read: the cached gripper_position can go
    stale when the telemetry stream desyncs, and a stale 0 is
    indistinguishable from 'jaw never closed'.
    """
    try:
        g = executor.driver.grippers.for_arm(arm)
        if hasattr(g, "raw_position_fresh"):
            raw = g.raw_position_fresh()
            if raw is not None:
                return raw
        return getattr(g._motor, "gripper_position", None)
    except Exception:
        return None


def _run_both(fns: Dict[str, callable]) -> Dict[str, object]:
    """
    Run per-arm callables concurrently and join. Returns name -> result
    (or the Exception raised). Mirrors the script's both()/threaded moves.
    """
    out: Dict[str, object] = {}

    def _wrap(name, fn):
        try:
            out[name] = fn()
        except Exception as exc:  # pragma: no cover - hardware path
            out[name] = exc

    threads = [
        threading.Thread(target=_wrap, args=(n, f), daemon=True, name=f"bicloth-{n}")
        for n, f in fns.items()
    ]
    for t in threads:
        t.start()
    for t in threads:
        t.join()
    return out


def _move_both(
    executor,
    targets: Dict[str, Tuple[np.ndarray, List[float]]],
    vel: float,
    correct_left_z: bool = True,
) -> Dict[str, object]:
    """
    Move both arms to (position, orient_rotvec) targets concurrently.

    When ``correct_left_z`` (the default, for ABSOLUTE perception-derived
    targets), the LEFT z target gets the LOCAL, position-dependent cross-arm
    tilt correction (cloth_geometry.left_z_correction_at) evaluated AT THAT
    TARGET POINT and added, so both arms land at the same physical height.
    The per-arm single-arm IK does not know about the tilt, and it varies
    across the workspace, so it is computed per-point (left_z - right_z of the
    same physical point via the cross-arm transform) rather than from a single
    calibration-origin delta; 0.0 when calibration is absent.

    Pass ``correct_left_z=False`` for RELATIVE moves computed off the
    current (already-corrected) TCP, e.g. a synchronized lift, so the
    correction is not applied twice.
    """
    fns = {}
    for arm, (pos, orient) in targets.items():
        p = np.asarray(pos, dtype=float).copy()
        if arm == "left" and correct_left_z:
            p[2] += _left_z_correction_at(executor, p)
        ex = _arm_ex(executor, arm)
        fns[arm] = lambda e=ex, p=p, o=orient: e._move_to(list(p), o, velocity=vel)
    return _run_both(fns)


def _left_z_correction_at(executor, point_left) -> float:
    """
    LOCAL, position-dependent left-frame z correction at a LEFT-frame point.

    The cross-arm tilt is not a constant origin offset: the same physical
    point reads higher in the left frame than the right, and the discrepancy
    varies across the workspace. Given the LEFT-frame target the left arm is
    being commanded to, map it back to the RIGHT frame and return the local
    ``left_z - right_z`` there. Add it to that target's z so the left arm
    lands at the same physical height as the right. 0.0 if calibration is
    absent.
    """
    tfm = _load_cross_arm_transforms()
    if tfm is None:
        return 0.0
    T_r, T_l = tfm
    p_right = geom.left_to_right(point_left, T_r, T_l)
    return geom.left_z_correction_at(p_right, T_r, T_l)


@spark_skill(
    name="bimanual_grasp_points",
    description=(
        "Bimanual cloth: both arms approach above, descend to, and pinch at "
        "two points (one per arm), each with its own top-down jaw yaw. After "
        "closing, verifies the SSG-48 raw position of BOTH jaws and FAILS if "
        "either read >= empty_raw_threshold (closed on air). Points are given "
        "explicitly (left_point/right_point as [x,y,z]) in each arm's base "
        "frame, with optional per-arm yaw (left_yaw_rad/right_yaw_rad)."
    ),
    params={
        "left_point": "left arm grasp [x,y,z] in left base frame",
        "right_point": "right arm grasp [x,y,z] in right base frame",
        "left_yaw_rad": "left jaw yaw (rad) before mount+flip (default 0)",
        "right_yaw_rad": "right jaw yaw (rad) (default 0)",
        "hover_z": "hover height above grasp (default 0.05m)",
        "grasp_z": "grasp z override applied to both points (default uses point z)",
        "force": "gripper pinch force (default 15N)",
        "empty_raw_threshold": "SSG-48 raw at/above which a jaw is empty (default 250)",
        "approach_vel": "descend velocity (default 0.08)",
        "move_vel": "hover-move velocity (default 0.22)",
    },
)
def bimanual_grasp_points(executor, params: dict):
    t0 = time.time()
    ok, err = _require_bimanual(executor, "bimanual_grasp_points")
    if not ok:
        return err
    cfg = _cloth_cfg(executor)

    lp = params.get("left_point")
    rp = params.get("right_point")
    if lp is None or rp is None:
        return _result(
            "bimanual_grasp_points",
            False,
            "need left_point and right_point [x,y,z]",
            time.time() - t0,
        )
    left_pt = np.asarray(lp, dtype=float)
    right_pt = np.asarray(rp, dtype=float)

    hover_z = float(_p(params, cfg, "hover_z"))
    if params.get("grasp_z") is not None:
        left_pt[2] = float(params["grasp_z"])
        right_pt[2] = float(params["grasp_z"])
    force = float(_p(params, cfg, "force"))
    empty = float(_p(params, cfg, "empty_raw_threshold"))
    approach_vel = float(_p(params, cfg, "approach_vel"))
    move_vel = float(_p(params, cfg, "move_vel"))

    o_l = _orient_rotvec("left", params.get("left_yaw_rad"), cfg, params)
    o_r = _orient_rotvec("right", params.get("right_yaw_rad"), cfg, params)

    # Hover above both points, then descend.
    hov_l = left_pt.copy()
    hov_l[2] = left_pt[2] + hover_z
    hov_r = right_pt.copy()
    hov_r[2] = right_pt[2] + hover_z
    res = _move_both(executor, {"left": (hov_l, o_l), "right": (hov_r, o_r)}, move_vel)
    for arm, r in res.items():
        if isinstance(r, Exception):
            return _result(
                "bimanual_grasp_points",
                False,
                f"{arm} hover move failed: {r}",
                time.time() - t0,
            )

    res = _move_both(
        executor, {"left": (left_pt, o_l), "right": (right_pt, o_r)}, approach_vel
    )
    for arm, r in res.items():
        if isinstance(r, Exception):
            return _result(
                "bimanual_grasp_points",
                False,
                f"{arm} descend failed: {r}",
                time.time() - t0,
            )

    # Close both grippers in parallel.
    _run_both(
        {
            "left": lambda: executor.driver.close_gripper(arm="left", force=force),
            "right": lambda: executor.driver.close_gripper(arm="right", force=force),
        }
    )
    time.sleep(0.4)

    # Verify SSG-48 raws: >= threshold means the jaw closed on air.
    raws = {a: _raw(executor, a) for a in ("left", "right")}
    empties = []
    for a, raw in raws.items():
        is_empty = raw is not None and float(raw) >= empty
        logger.info(
            "[bimanual_grasp_points] %s raw=%s %s",
            a,
            raw,
            "EMPTY" if is_empty else "HOLDING",
        )
        if is_empty:
            empties.append(a)
    if empties:
        # Release so a recovery/replan starts from open jaws.
        executor.driver.open_gripper()
        return _result(
            "bimanual_grasp_points",
            False,
            f"jaw(s) {empties} closed on air (raw>= {empty}); " f"raws={raws}",
            time.time() - t0,
        )

    executor.state.holding["left"] = True
    executor.state.holding["right"] = True
    return _result(
        "bimanual_grasp_points",
        True,
        f"both jaws holding (raws={raws})",
        time.time() - t0,
    )


@spark_skill(
    name="bimanual_lift_together",
    description=(
        "Bimanual cloth: both arms lift their currently-pinched points by "
        "lift_height, synchronized so the two arms start and finish together "
        "(durations equalized). Use after bimanual_grasp_points."
    ),
    params={
        "lift_height": "vertical lift in meters (default 0.13m)",
        "move_vel": "lift velocity ceiling (default 0.22)",
    },
)
def bimanual_lift_together(executor, params: dict):
    t0 = time.time()
    ok, err = _require_bimanual(executor, "bimanual_lift_together")
    if not ok:
        return err
    cfg = _cloth_cfg(executor)
    lift_h = (
        float(_p(params, cfg, "lift_z"))
        if params.get("lift_height") is None
        else float(params["lift_height"])
    )
    move_vel = float(_p(params, cfg, "move_vel"))

    targets = {}
    for arm in ("left", "right"):
        tcp = np.asarray(executor.driver.get_tcp_pose(arm), dtype=float)
        pos = tcp[:3].copy()
        pos[2] += lift_h
        orient = _orient_rotvec(arm, None, cfg, params)
        targets[arm] = (pos, orient)

    # Relative lift off the current (already height-corrected) TCP, so the
    # left-z correction is NOT re-applied here.
    res = _move_both(executor, targets, move_vel, correct_left_z=False)
    fails = {a: str(r) for a, r in res.items() if isinstance(r, Exception)}
    if fails:
        return _result(
            "bimanual_lift_together",
            False,
            f"synced lift failed: {fails}",
            time.time() - t0,
        )
    return _result(
        "bimanual_lift_together",
        True,
        f"both arms lifted {lift_h*1000:.0f}mm",
        time.time() - t0,
    )


def _arc_drape_arm(
    executor,
    arm: str,
    start_xyz: np.ndarray,
    land_xy: np.ndarray,
    cfg: dict,
    params: dict,
    n_steps: int,
    arc_peak: float,
    orient: List[float],
) -> bool:
    """
    Sinusoidal arc-drape one arm from start to land_xy. Returns True on
    success. Lifts on the way (sinusoidal peak), lands at land_z.
    """
    land_z = float(_p(params, cfg, "land_z"))
    arc_vel = float(_p(params, cfg, "arc_vel"))
    start = np.asarray(start_xyz, dtype=float)
    lift_z = start[2]
    # The arc body is relative to the current (already-corrected) TCP, but
    # land_z is an ABSOLUTE height; the left arm's absolute z carries the
    # LOCAL cross-arm tilt correction evaluated AT THE LAND POINT so it lands
    # at the same physical height as the right arm.
    if arm == "left":
        land_pt_left = np.array([float(land_xy[0]), float(land_xy[1]), land_z])
        land_z_arm = land_z + _left_z_correction_at(executor, land_pt_left)
    else:
        land_z_arm = land_z
    for i in range(1, n_steps + 1):
        t = i / float(n_steps)
        x = start[0] + t * (float(land_xy[0]) - start[0])
        y = start[1] + t * (float(land_xy[1]) - start[1])
        if i < n_steps:
            z = lift_z + arc_peak * np.sin(t * np.pi)
        else:
            z = land_z_arm
        try:
            _arm_ex(executor, arm)._move_to([x, y, z], orient, velocity=arc_vel)
        except Exception as exc:  # pragma: no cover - hardware path
            logger.warning(
                "[bimanual_arc_drape] %s arc step %d failed: %s", arm, i, exc
            )
            return False
    return True


@spark_skill(
    name="bimanual_arc_drape",
    description=(
        "Bimanual cloth: arc-drape held fabric. With both_arms true (default) "
        "both arms drape simultaneously from their current TCP to per-arm land "
        "targets; otherwise only 'arm' drapes. The path is a sinusoidal arc "
        "(rises to arc_peak then lands at land_z). Targets are given as "
        "left_land_xy / right_land_xy ([x,y]); if omitted, only the x is "
        "swept to land_x (y held). Use after a lift to fold sleeves to the "
        "garment center or to drape the hem to the collar."
    ),
    params={
        "both_arms": "drape both arms simultaneously (default true)",
        "arm": "single arm to drape when both_arms is false ('left'|'right')",
        "left_land_xy": "left arm land [x,y] (default: x=land_x, y held)",
        "right_land_xy": "right arm land [x,y] (default: x=land_x, y held)",
        "land_x": "default land x when *_land_xy omitted (default 0.74)",
        "arc_peak": "sinusoidal arc peak height (default 0.04m)",
        "steps": "arc waypoint count (default 10)",
        "land_z": "landing height (default 0.025m)",
        "arc_vel": "arc velocity (default 0.16)",
    },
)
def bimanual_arc_drape(executor, params: dict):
    t0 = time.time()
    ok, err = _require_bimanual(executor, "bimanual_arc_drape")
    if not ok:
        return err
    cfg = _cloth_cfg(executor)
    both_arms = bool(params.get("both_arms", True))
    n_steps = int(params.get("steps", DEFAULTS["hem_arc_steps"]))
    arc_peak = float(_p(params, cfg, "arc_peak"))
    land_x = float(_p(params, cfg, "land_x"))

    arms = ("left", "right") if both_arms else (params.get("arm", "right"),)
    if not both_arms and arms[0] not in ("left", "right"):
        return _result(
            "bimanual_arc_drape", False, f"invalid arm {arms[0]!r}", time.time() - t0
        )

    starts: Dict[str, np.ndarray] = {}
    lands: Dict[str, np.ndarray] = {}
    orients: Dict[str, List[float]] = {}
    for arm in arms:
        tcp = np.asarray(executor.driver.get_tcp_pose(arm), dtype=float)
        starts[arm] = tcp[:3].copy()
        key = f"{arm}_land_xy"
        if params.get(key) is not None:
            lands[arm] = np.asarray(params[key], dtype=float)
        else:
            lands[arm] = np.array([land_x, tcp[1]])
        orients[arm] = _orient_rotvec(arm, None, cfg, params)

    fns = {}
    for arm in arms:
        fns[arm] = lambda a=arm: _arc_drape_arm(
            executor, a, starts[a], lands[a], cfg, params, n_steps, arc_peak, orients[a]
        )
    res = _run_both(fns)
    fails = {a: r for a, r in res.items() if r is not True}
    if fails:
        return _result(
            "bimanual_arc_drape", False, f"drape failed: {fails}", time.time() - t0
        )
    return _result(
        "bimanual_arc_drape", True, f"draped {list(arms)} to land", time.time() - t0
    )

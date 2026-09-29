# Constrained scrubbing motion over a target surface.

import logging
import time
import numpy as np
from scipy.spatial.transform import Rotation as _Rot
from spark_real.skills.registry import spark_skill
from spark_real.skills.primitives import _result

logger = logging.getLogger(__name__)


@spark_skill(
    name="constrained_scrub",
    description=(
        "Scrub a flat target surface with a held sponge while tracking "
        "a target normal force."
    ),
    params={
        "target_label": str,
        "force_n": float,
        "n_cycles": int,
        "scrub_radius_m": float,
        "pattern": str,
    },
)
def constrained_scrub(executor, params: dict):
    """
    Force-tracked scrub of a flat surface with a held sponge.
    """
    t0 = time.time()
    label = params.get("target_label", "")
    force_n = float(params.get("force_n", 8.0))
    n_cycles = max(1, int(params.get("n_cycles", 4)))
    scrub_radius = float(params.get("scrub_radius_m", 0.04))
    pattern = str(params.get("pattern", "circle")).lower()

    HOVER_Z, FINAL_LIFT_Z = 0.02, 0.10
    DESCENT_STEP, MAX_DESCENT = 0.001, 0.04
    CORRECTION_STEP, HARD_F_LIMIT = 0.001, 25.0
    WAYPOINTS_PER_CYCLE, FORCE_POLL_EVERY = 6, 2
    OPEN_LOOP_PUSH = 0.012

    if not executor._holding:
        return _result(
            "constrained_scrub", False, "Not holding sponge", time.time() - t0
        )

    det = executor.detection_map.get(label)
    if det is None:
        return _result(
            "constrained_scrub", False, f"Target '{label}' not found", time.time() - t0
        )

    plate = np.array(det["position_3d"], dtype=float)
    cx, cy, plate_z = plate[0], plate[1], plate[2]
    orient = executor.GRASP_ORIENTATION

    hover = np.array([cx, cy, plate_z + HOVER_Z])
    executor._move_to(hover, orient)

    # Force baseline probe
    has_force = hasattr(executor.robot, "get_tcp_force")
    baseline_fz = 0.0
    if has_force:
        try:
            time.sleep(0.15)
            samples = []
            for _ in range(5):
                w = executor.robot.get_tcp_force()
                if w is not None and len(w) >= 3:
                    samples.append(float(w[2]))
                time.sleep(0.02)
            if samples:
                baseline_fz = float(np.mean(samples))
                if all(abs(s) < 1e-9 for s in samples):
                    has_force = False
        except Exception:
            has_force = False

    used_force = has_force
    contact_z = None

    if has_force:
        descent = hover.copy()
        for i in range(int(MAX_DESCENT / DESCENT_STEP)):
            if getattr(executor, "_abort", False):
                return _result("constrained_scrub", False, "Aborted", time.time() - t0)
            descent[2] -= DESCENT_STEP
            executor._move_to(descent, orient, velocity=executor.velocity * 0.10)
            time.sleep(0.03)
            try:
                w = executor.robot.get_tcp_force()
                fz = float(w[2]) if (w is not None and len(w) >= 3) else 0.0
            except Exception:
                fz = 0.0
            delta = abs(fz - baseline_fz)
            if delta > HARD_F_LIMIT:
                descent[2] += DESCENT_STEP * 3
                executor._move_to(descent, orient, velocity=executor.velocity * 0.10)
                return _result(
                    "constrained_scrub",
                    False,
                    f"Hard force limit at z={descent[2]:.3f}",
                    time.time() - t0,
                )
            if delta >= force_n:
                contact_z = descent[2]
                break
        if contact_z is None:
            used_force = False
            contact_z = hover[2] - OPEN_LOOP_PUSH
            executor._move_to(
                np.array([cx, cy, contact_z]), orient, velocity=executor.velocity * 0.25
            )
    else:
        contact_z = hover[2] - OPEN_LOOP_PUSH
        executor._move_to(
            np.array([cx, cy, contact_z]), orient, velocity=executor.velocity * 0.25
        )

    # Scrub pattern
    f_lo, f_hi = 0.5 * force_n, 1.5 * force_n

    def _pattern_xy(t_norm):
        a = scrub_radius
        if pattern == "figure8":
            theta = 2 * np.pi * t_norm
            denom = 1.0 + np.sin(theta) ** 2
            return a * np.cos(theta) / denom, a * np.sin(theta) * np.cos(theta) / denom
        if pattern == "lines":
            seg = 0.25
            stroke_idx = int(t_norm / seg)
            local = (t_norm - stroke_idx * seg) / seg
            x_frac = 1.0 - abs(2.0 * local - 1.0)
            y_frac = stroke_idx / 3.0
            return a * (2 * x_frac - 1), a * (2 * y_frac - 1)
        theta = 2 * np.pi * t_norm
        return a * np.cos(theta), a * np.sin(theta)

    cur_z = contact_z
    n_wp = WAYPOINTS_PER_CYCLE * n_cycles
    # Optional lean into the stroke tangent; lean_deg=0 keeps a flat drag.
    LEAN_RAD = np.deg2rad(float(params.get("lean_deg", 0.0) or 0.0))
    _base_R = _Rot.from_rotvec(np.asarray(orient, dtype=float))
    for k in range(n_wp):
        if getattr(executor, "_abort", False):
            break
        t_norm = (k % WAYPOINTS_PER_CYCLE) / float(WAYPOINTS_PER_CYCLE)
        dx, dy = _pattern_xy(t_norm)
        if LEAN_RAD > 1e-6:
            theta = 2 * np.pi * t_norm
            tx, ty = -np.sin(theta), np.cos(theta)  # circle tangent
            _lean_axis = np.array([-ty, tx, 0.0]) * LEAN_RAD
            orient_k = (_Rot.from_rotvec(_lean_axis) * _base_R).as_rotvec().tolist()
        else:
            orient_k = orient
        executor._move_to(
            np.array([cx + dx, cy + dy, cur_z]),
            orient_k,
            velocity=executor.velocity * 0.7,
        )
        if used_force and (k % FORCE_POLL_EVERY == 0):
            try:
                w = executor.robot.get_tcp_force()
                delta = (
                    abs(float(w[2]) - baseline_fz) if (w and len(w) >= 3) else force_n
                )
            except Exception:
                delta = force_n
            if delta > HARD_F_LIMIT:
                cur_z += CORRECTION_STEP * 5
                break
            if delta < f_lo:
                cur_z -= CORRECTION_STEP
            elif delta > f_hi:
                cur_z += CORRECTION_STEP

    executor._move_to(np.array([cx, cy, plate_z + FINAL_LIFT_Z]), orient)
    mode = "force-tracked" if used_force else "open-loop"
    return _result(
        "constrained_scrub",
        True,
        f"Scrubbed '{label}' ({mode}, {pattern}, {n_cycles} cycles)",
        time.time() - t0,
    )


@spark_skill(
    name="wipe",
    description=(
        "Alias for constrained_scrub: wipe/scrub a flat target surface with a "
        "held sponge while tracking a target normal force (paper grammar name)."
    ),
    params={
        "target_label": str,
        "force_n": float,
        "n_cycles": int,
        "scrub_radius_m": float,
        "pattern": str,
    },
)
def wipe(executor, params: dict):
    """
    Paper-grammar alias; delegates to the canonical constrained_scrub.
    """
    return constrained_scrub(executor, params)

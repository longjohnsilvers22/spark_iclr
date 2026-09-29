"""Where to open the jaws over a container, from PERCEIVED geometry.

A leaf module: stdlib + numpy only, imports nothing from ``spark_real``.

Inputs (metres, robot base frame), measured by
``perception.mask_geometry.mask_height_profile``:

    rim_z_m       the container's rim: the TOP of its mask's depth cloud
    interior_z_m  the container's interior floor: the BOTTOM of the same cloud
    (the held object carries the same two fields for its own mask)

``held_drop_m`` is how far the held object hangs below the TCP: the grasp
descended to ``held.position_3d[2] + GRIPPER_OPEN_Z_OFFSET`` and the object's
bottom is ``held.interior_z_m``. Unknown -> ``HELD_DROP_FALLBACK_M`` (a larger
drop raises the TCP, the safe direction).

Aim the held object's BOTTOM ``FLOOR_TARGET_GAP_M`` above the interior floor::

    nominal = interior_z + FLOOR_TARGET_GAP_M + held_drop_m

    lo = max(rim_z + MIN_TCP_ABOVE_RIM_M,                 # jaws stay out
             interior_z + MIN_FLOOR_CLEARANCE_M + drop)   # object stays up
    hi = rim_z + MAX_TCP_ABOVE_RIM_M + held_drop_m        # no free-fall drops

The TCP never crosses the rim plane, so the jaws cannot strike a rigid rim or
interior however wrong the height reading is. Lowering the tool INTO a
container is out of scope (the planar extent is not trustworthy on this rig);
this module adds NO descent of its own, it only chooses the Z the place
transport flies to.

PRECEDENCE
----------
1. ``strict_offset_z: true`` on the plan node -> the plan's Z verbatim, logged.
   Source ``plan-strict``.
2. No usable container height profile (missing, non-finite, inverted,
   implausible, or too few depth samples) -> the plan's Z verbatim. Source
   ``plan-no-geometry``.
3. Otherwise ``clip(min(plan_z, nominal), lo, hi)``: geometry may only LOWER
   the plan (``offset_z: 0.02`` stays 0.02); the BOUNDS may only raise it
   (``offset_z: 0.0`` aims below the rim and is clamped up).

The chosen height, its source, and both bounds are reported on the result.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Any, Optional, Tuple
from spark_real.utils.det_fields import det_field

# --- geometry constants (metres) -------------------------------------------

# Where the held object's bottom is aimed, above the container's interior
# floor. This IS the remaining fall.
FLOOR_TARGET_GAP_M = 0.02
# Hard lower bound on the same gap. Everything above is aiming; this is the
# line the clamp enforces.
MIN_FLOOR_CLEARANCE_M = 0.01
# The TCP never comes closer than this to the rim plane, and never crosses it.
MIN_TCP_ABOVE_RIM_M = 0.02
# ... and never sits further above the rim than this. A larger gap is a drop,
# not a place. Caps a rim reading that came back too high.
MAX_TCP_ABOVE_RIM_M = 0.05

# How far the held object hangs below the TCP when it cannot be measured.
# Errs LONG on purpose: a longer assumed overhang raises the release.
# Below this, a measured top-to-bottom extent is not a measurement: a depth
# camera cannot see an object's underside, so anything thinner than this is
# noise or an inversion. Triggers the OBB-minor fallback in held_drop.
MIN_MEASURABLE_OVERHANG_M = 0.005

HELD_DROP_FALLBACK_M = 0.06
# Nothing this gripper carries hangs further below the TCP than this; a larger
# reading is a mask that swallowed the table, not a very long object.
MAX_HELD_DROP_M = 0.15

# Validity band for a measured container. A rim more than this above the table
# is not a container placed into from above, and an interior below the table is
# depth punching through the container to the surface it stands on.
MAX_CONTAINER_RIM_ABOVE_TABLE_M = 0.40
# Minimum depth pixels inside the mask before its height profile counts as
# measured rather than as three stray points.
MIN_PROFILE_SAMPLES = 50


@dataclass
class ReleaseHeight:
    """The resolved place-target Z, plus everything that decided it."""

    z: float
    source: str  # "geometry" | "plan-strict" | "plan-no-geometry"
    detail: str
    held_drop_m: float
    rim_z: Optional[float] = None
    interior_z: Optional[float] = None
    geometric_z: Optional[float] = None
    lo: Optional[float] = None
    hi: Optional[float] = None
    clamped: bool = False

    def describe(self) -> str:
        if self.geometric_z is None:
            return f"z={self.z:.3f} ({self.source}: {self.detail})"
        return (
            f"z={self.z:.3f} ({self.source}: rim={self.rim_z:.3f} "
            f"interior={self.interior_z:.3f} drop={self.held_drop_m*100:.1f}cm "
            f"band=[{self.lo:.3f},{self.hi:.3f}]"
            f"{' CLAMPED' if self.clamped else ''}; {self.detail})"
        )


def _finite(value) -> Optional[float]:
    try:
        f = float(value)
    except (TypeError, ValueError):
        return None
    return f if math.isfinite(f) else None


def height_profile(det: Any) -> Tuple[Optional[Tuple[float, float]], str]:
    """``((rim_z, interior_z), reason)`` for a detection, or ``(None, reason)``.

    Every rejection is a distinct reason string so a run's log says WHICH way
    the measurement was unusable, not just that it was.
    """
    if det is None:
        return None, "no detection"
    rim = _finite(det_field(det, "rim_z_m"))
    interior = _finite(det_field(det, "interior_z_m"))
    if rim is None or interior is None:
        return None, "no measured height profile"
    n = _finite(det_field(det, "height_samples", 0.0)) or 0.0
    if n < MIN_PROFILE_SAMPLES:
        return None, f"only {n:.0f} depth samples in the mask"
    if interior > rim + 1e-6:
        return None, f"inverted profile (rim {rim:.3f} below interior {interior:.3f})"
    return (rim, interior), "measured"


def _container_profile(
    det: Any, table_z: Optional[float]
) -> Tuple[Optional[Tuple[float, float]], str]:
    """``height_profile`` plus the checks that only apply to a container."""
    prof, reason = height_profile(det)
    if prof is None:
        return None, reason
    rim, interior = prof
    if table_z is not None and math.isfinite(table_z):
        if rim > table_z + MAX_CONTAINER_RIM_ABOVE_TABLE_M:
            return None, (
                f"rim {rim:.3f} is more than "
                f"{MAX_CONTAINER_RIM_ABOVE_TABLE_M*100:.0f}cm above the table"
            )
        # Depth that punched through a translucent container reads the surface
        # it stands ON. Nothing inside it can be below that surface.
        if interior < table_z:
            interior = float(table_z)
            reason = "measured (interior raised to the table plane)"
    return (rim, interior), reason


def held_drop(held: Any, grasp_z_offset: float = 0.0) -> Tuple[float, str]:
    """How far the held object hangs below the TCP, in metres.

    ``grasp_z_offset`` is the executor's ``GRIPPER_OPEN_Z_OFFSET``: the grasp
    descended to the object's perceived Z plus that offset, so the TCP-to-
    bottom distance is ``(perceived_z + offset) - measured_bottom``.
    """
    prof, reason = height_profile(held)
    top = _finite(det_field(held, "position_3d", [None, None, None])[2]) if held else None
    if prof is None or top is None:
        return HELD_DROP_FALLBACK_M, f"held extent unknown ({reason}); fallback"
    _rim, bottom = prof
    raw = (top + float(grasp_z_offset)) - bottom
    # A depth camera sees an object's TOP surface, never its underside, so for
    # a tool lying flat this difference collapses to noise and can invert
    # (raw <= 0 would model the tool as hanging nothing below the gripper).
    # The OBB minor axis (measured SHORT dimension, e.g. 13.4 mm for a
    # screwdriver) is a better estimate of the protrusion for anything a
    # parallel gripper can close on. Use it when the extent is degenerate.
    if raw < MIN_MEASURABLE_OVERHANG_M:
        minor = _finite(det_field(held, "obb_minor_m", None)) if held else None
        if minor is not None and minor > MIN_MEASURABLE_OVERHANG_M:
            drop = min(float(minor), MAX_HELD_DROP_M)
            return drop, (
                f"held overhang from OBB minor {minor*1000:.0f}mm "
                f"(depth extent {raw*1000:.0f}mm is degenerate)"
            )
    drop = min(max(raw, 0.0), MAX_HELD_DROP_M)
    note = "measured"
    if raw > MAX_HELD_DROP_M:
        note = f"measured {raw*100:.0f}cm, capped"
    return drop, f"held overhang {note}"


def resolve_release_z(
    plan_z: float,
    container: Any,
    held: Any,
    table_z: Optional[float] = None,
    strict: bool = False,
    grasp_z_offset: float = 0.0,
) -> ReleaseHeight:
    """Resolve the Z a place transport should fly to. See the module docstring."""
    drop, drop_note = held_drop(held, grasp_z_offset=grasp_z_offset)
    plan_z = float(plan_z)

    if strict:
        return ReleaseHeight(
            z=plan_z,
            source="plan-strict",
            detail="strict_offset_z set; the plan's offset_z is authoritative",
            held_drop_m=drop,
        )

    prof, reason = _container_profile(container, table_z)
    if prof is None:
        return ReleaseHeight(
            z=plan_z, source="plan-no-geometry", detail=reason, held_drop_m=drop
        )

    rim, interior = prof
    depth = max(float(rim) - float(interior), 0.0)

    # Aiming FLOOR_TARGET_GAP above the floor is wrong for a recess shallower
    # than that gap: the aim point lands above the rim. For such a recess aim
    # at HALF its depth (a 1.7 cm cutout aims at 0.85 cm); a container at
    # least as deep as the gap (a 3.9 cm bowl) keeps the full 2 cm.
    target_gap = (
        FLOOR_TARGET_GAP_M
        if depth >= FLOOR_TARGET_GAP_M
        else max(depth * 0.5, 0.0)
    )
    nominal = interior + target_gap + drop

    # Rim guard: the jaws never cross the rim plane. For a recess shallower
    # than the fixed 2 cm margin (e.g. rim -0.253, interior -0.270: the floor
    # target needs TCP -0.2466, 6.4 mm above the rim but 13.6 mm below
    # rim+2cm), shrink the margin to what still lets the object reach its
    # floor target, never below zero. A deeper container keeps the full 2 cm.
    rim_guard = MIN_TCP_ABOVE_RIM_M
    if depth < MIN_TCP_ABOVE_RIM_M:
        needed = (interior + MIN_FLOOR_CLEARANCE_M + drop) - rim
        rim_guard = min(MIN_TCP_ABOVE_RIM_M, max(needed, 0.0))

    lo = max(rim + rim_guard, interior + MIN_FLOOR_CLEARANCE_M + drop)
    hi = rim + MAX_TCP_ABOVE_RIM_M + drop
    # rim >= interior is guaranteed by height_profile, so lo <= hi always; the
    # max() keeps clip() well defined even if a future edit breaks that.
    hi = max(hi, lo)
    geometric = min(max(nominal, lo), hi)

    # Geometry may only lower the plan; the bounds may only raise it.
    chosen = min(plan_z, geometric)
    clamped = chosen < lo - 1e-12
    chosen = min(max(chosen, lo), hi)

    return ReleaseHeight(
        z=float(chosen),
        source="geometry",
        detail=f"{reason}; {drop_note}",
        held_drop_m=drop,
        rim_z=rim,
        interior_z=interior,
        geometric_z=float(geometric),
        lo=float(lo),
        hi=float(hi),
        clamped=bool(clamped),
    )

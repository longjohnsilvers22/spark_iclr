"""Container planar-extent geometry, a leaf module (numpy only).

Imports nothing from spark_real, so skills can import it without a circular
import through execution_recovery.
"""

from typing import Optional

import numpy as np

# Fallback container half-extent (m) when a detection lacks slot geometry: the
# region radius is max(this, OBB-minor * 0.6), widened to cover detected slots.
TRAY_FALLBACK_HALF_EXTENT_M = 0.10


def container_region(det) -> Optional[tuple]:
    """Return (centroid_xy[2], radius_m) for a container detection, or None.

    None when the detection has no usable 3D centroid.
    """
    if det is None:
        return None
    get = det.get if hasattr(det, "get") else (lambda k, d=None: getattr(det, k, d))
    pos = get("position_3d", None)
    if pos is None:
        return None
    try:
        centroid_xy = np.asarray(pos[:2], dtype=float)
    except Exception:
        return None
    minor = float(get("obb_minor_m", 0.0) or 0.0)
    radius = max(TRAY_FALLBACK_HALF_EXTENT_M, minor * 0.6)
    # Widen to cover detected slots if present (real tray extent).
    slots = get("slots", None) or []
    for s in slots:
        wxyz = s.get("world_xyz") if hasattr(s, "get") else None
        if wxyz is None:
            continue
        try:
            d = float(np.linalg.norm(np.asarray(wxyz[:2], dtype=float) - centroid_xy))
            radius = max(radius, d + TRAY_FALLBACK_HALF_EXTENT_M)
        except Exception:
            continue
    return centroid_xy, radius


def xy_inside_container(xy, det) -> bool:
    """True if planar point ``xy`` lies within container ``det``'s extent."""
    region = container_region(det)
    if region is None:
        return False
    centroid_xy, radius = region
    try:
        p = np.asarray(xy[:2], dtype=float)
    except Exception:
        return False
    return bool(np.linalg.norm(p - centroid_xy) <= radius)

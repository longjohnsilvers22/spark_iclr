"""Cross-label spatial dedup: when two labels are ONE object, and when they are two.

SAM3 fires several prompts onto the same mask ("knife handle" landing on a
spoon), so a rule that keeps the higher-confidence label and drops the other is
right. But the colocation test that finds those duplicates also fires on the
exact geometry a placement task exists to CREATE. Put the plushie in the bowl
and the two centroids land centimetres apart, so the dedup deletes the plushie
and the verifier that asks "is the plushie in the bowl" has nothing left to
bind.

The discriminator is that a DUPLICATE is the same mask wearing two names, while
a CONTAINED object is a smaller thing at a different height inside a bigger one:

  * in 3D (the cross-camera merge) two detections are one object only when they
    agree in Z as well as in XY, and never when one of them is a container
    whose measured interior floor lies below the other;
  * in image space (inside one camera) two detections are one object only when
    their masks substantially overlap. A bowl mask that merely CONTAINS a small
    plushie mask scores a low IoU against it; two prompts that resolved to the
    same mask score ~1.0.

Both rules only ever REFUSE a dedup the colocation rule would have made.

``SPARK_CONTAINMENT_DEDUP=0`` restores the colocation-only rule.
"""

from __future__ import annotations

import logging
from typing import Any, Optional

import numpy as np
from spark_real.utils.env_flags import env_flag

logger = logging.getLogger(__name__)

ENV_FLAG = "SPARK_CONTAINMENT_DEDUP"

# Two detections of one physical object share a height, not just a footprint.
# 4cm is under the height of every tabletop object in the demo set and well
# over the cross-camera Z disagreement the merge already tolerates. The logged
# plushie/bowl pair differed by 6.9cm.
CROSS_LABEL_DEDUP_DZ_M = 0.04

# Mask overlap (IoU) above which two masks in ONE camera are the same mask.
# A duplicate prompt scores near 1.0; a small object sitting inside a large
# container scores at most area_small/area_large.
CROSS_LABEL_DEDUP_MIN_IOU = 0.5


def containment_aware(default: bool = True) -> bool:
    """Is the containment-aware rule on? Env overrides the default."""
    return env_flag(ENV_FLAG, default)


def _get(det: Any, name: str, default: Any = None) -> Any:
    if det is None:
        return default
    if hasattr(det, name):
        value = getattr(det, name)
    elif hasattr(det, "get"):
        value = det.get(name, default)
    else:
        return default
    return default if value is None else value


def _z(det: Any) -> Optional[float]:
    pos = _get(det, "position_3d")
    try:
        return float(pos[2])
    except (TypeError, ValueError, IndexError):
        return None


def is_container(det: Any) -> bool:
    """Did mask geometry measure this detection as a container?

    ``slots`` is only ever populated for containers, and ``rim_z_m`` /
    ``interior_z_m`` only when the mask's depth cloud actually resolved a rim
    above a floor (``height_samples`` counts the pixels behind them). Neither
    is a label heuristic, so a container this rig has never seen still counts.
    """
    if _get(det, "slots"):
        return True
    if int(_get(det, "height_samples", 0) or 0) <= 0:
        return False
    rim, interior = _get(det, "rim_z_m"), _get(det, "interior_z_m")
    if rim is None or interior is None:
        return False
    try:
        return float(rim) > float(interior)
    except (TypeError, ValueError):
        return False


def _one_contains_the_other(a: Any, b: Any) -> bool:
    """Is one of these a container the other could be sitting in?

    True when either detection is a measured container and the other's centroid
    is at or above that container's interior floor -- which is precisely the
    ``inside`` predicate's own geometry, so the dedup must not pre-empt it.
    """
    for container, other in ((a, b), (b, a)):
        if not is_container(container):
            continue
        floor = _get(container, "interior_z_m")
        z_other = _z(other)
        if floor is None or z_other is None:
            # A container with no measured floor still shields the pair: the
            # dedup cannot tell containment from duplication without it, and
            # deleting the contained object is the unrecoverable error.
            return True
        try:
            if z_other >= float(floor) - CROSS_LABEL_DEDUP_DZ_M:
                return True
        except (TypeError, ValueError):
            return True
    return False


def same_object_3d(a: Any, b: Any, xy_tol_m: float, aware: bool = True) -> bool:
    """Are these two world-frame detections one physical object?

    ``aware`` False reproduces the XY-only colocation test.
    """
    pa, pb = _get(a, "position_3d"), _get(b, "position_3d")
    if pa is None or pb is None:
        return False
    if float(np.linalg.norm(np.asarray(pa[:2], dtype=float) - np.asarray(pb[:2], dtype=float))) >= float(xy_tol_m):
        return False
    if not aware:
        return True
    if _one_contains_the_other(a, b):
        return False
    za, zb = _z(a), _z(b)
    if za is None or zb is None:
        return True
    return abs(za - zb) < CROSS_LABEL_DEDUP_DZ_M


def mask_iou(a: Any, b: Any) -> Optional[float]:
    """IoU of two boolean masks, or None when either is missing/degenerate."""
    ma, mb = _get(a, "mask"), _get(b, "mask")
    if ma is None or mb is None:
        return None
    ma, mb = np.asarray(ma).astype(bool), np.asarray(mb).astype(bool)
    if ma.shape != mb.shape:
        return None
    union = int(np.count_nonzero(ma | mb))
    if union == 0:
        return None
    return float(np.count_nonzero(ma & mb)) / float(union)


def same_object_2d(a: Any, b: Any, px_tol: float, aware: bool = True) -> bool:
    """Are these two detections in ONE camera's image the same mask?

    ``aware`` False reproduces the centroid-distance-only test.
    """
    ca, cb = _get(a, "centroid_2d"), _get(b, "centroid_2d")
    if ca is None or cb is None:
        return False
    try:
        dist = float(np.hypot(float(ca[0]) - float(cb[0]), float(ca[1]) - float(cb[1])))
    except (TypeError, ValueError, IndexError):
        return False
    if dist >= float(px_tol):
        return False
    if not aware:
        return True
    iou = mask_iou(a, b)
    if iou is None:
        # No comparable masks: fall back to the colocation-only behaviour
        # rather than silently keeping every colocated pair.
        return True
    return iou >= CROSS_LABEL_DEDUP_MIN_IOU


__all__ = [
    "CROSS_LABEL_DEDUP_DZ_M",
    "CROSS_LABEL_DEDUP_MIN_IOU",
    "ENV_FLAG",
    "containment_aware",
    "is_container",
    "mask_iou",
    "same_object_2d",
    "same_object_3d",
]

"""
Mask plausibility, measured from the mask alone. No second model, no network.

The question answered here is not "how sure is SAM3 that this is a plushie"
but "is this mask's major axis a real axis": a blob assigned aspect_ratio
3.14 clears the 1.8 yaw gate and drives a -68.9 deg wrist rotation off an
OBB angle that means nothing.

That question is answerable with cv2 and nothing else, and the sharpest test
is STABILITY: erode and dilate the mask a little and recompute the major axis.
A knife's axis does not move. A blob's axis swings, because for a blob the
axis was decided by a few boundary pixels. Anything that swings more than a
few degrees under a perturbation that small cannot be used to aim a wrist.

Second, independent signal: the aspect_ratio the detection carries comes from
world-XY PCA over backprojected depth, while the mask has its own aspect
ratio in pixels. When the mask says 1.2 and the record says 3.14, the
elongation lives in the depth noise, not in the object.

Everything here is a pure function of a mask. Safe to call anywhere.
"""

from __future__ import annotations

import logging
import math
from dataclasses import dataclass, field
from typing import Optional

import cv2
import numpy as np

logger = logging.getLogger(__name__)

_MIN_PIXELS = 24  # below this nothing is measurable


@dataclass
class QualityThresholds:
    """All tunable. Loaded from config; these defaults are the shipped policy."""

    min_pixels: int = _MIN_PIXELS
    min_fill: float = 0.45  # mask_area / OBB area
    min_solidity: float = 0.80  # mask_area / convex hull area
    min_component_frac: float = 0.85  # largest connected component share
    max_angle_swing_deg: float = 12.0  # axis movement under erode/dilate
    max_ar_inflation: float = 1.6  # reported_ar / mask_ar ceiling, see below
    min_mask_ar: float = 1.5  # a yaw needs elongation in the mask itself
    perturb_frac: float = 0.06  # erosion radius as a fraction of sqrt(area)


@dataclass
class MaskQuality:
    """Measured geometry trust for one mask."""

    pixels: int = 0
    mask_aspect_ratio: float = 1.0
    major_axis_deg: float = 0.0
    fill: float = 0.0
    solidity: float = 0.0
    component_frac: float = 0.0
    angle_swing_deg: float = 180.0
    ar_inflation: float = 1.0  # reported_ar / mask_ar; >1 means depth stretched it
    measured: bool = False
    angle_trustworthy: bool = False
    # WHY the axis is untrustworthy, split by whether a different mask could
    # repair it. Both of these are statements about the object's SHAPE that
    # the pixels AGREE with, so no re-prompt can change them; the mask-
    # integrity failures (fill / components / swing) are the ones a better
    # mask fixes. Consumers wanting "can I aim a wrist" still read
    # angle_trustworthy -- these only explain it.
    shape_round: bool = False   # mask has no major axis to align a wrist to
    ar_inflated: bool = False   # world OBB stretched by depth noise, not text
    geometry_trust: float = 0.0  # [0,1], 0 = do not use this shape
    # Mask-measured answer to the SAME question perception's obb_confidence
    # asks: how well-defined is this OBB's major axis. 0.0 when there is no
    # usable axis at all, so it can gate grasp.yaw_min_obb_conf directly.
    axis_trust: float = 0.0
    reasons: list = field(default_factory=list)

    def describe(self) -> str:
        if not self.measured:
            return "unmeasurable mask"
        return (
            f"trust={self.geometry_trust:.2f} axis_trust={self.axis_trust:.2f} "
            f"fill={self.fill:.2f} "
            f"solid={self.solidity:.2f} cc={self.component_frac:.2f} "
            f"swing={self.angle_swing_deg:.1f}deg mask_ar={self.mask_aspect_ratio:.2f}"
            + (f" [{', '.join(self.reasons)}]" if self.reasons else "")
        )


def _as_u8(mask) -> Optional[np.ndarray]:
    if mask is None:
        return None
    m = np.asarray(mask)
    if m.ndim != 2 or m.size == 0:
        return None
    return (m > 0).astype(np.uint8)


def _major_axis(mask_u8):
    """(angle_deg in [0,180), major_px, minor_px, obb_area) from minAreaRect."""
    pts = cv2.findNonZero(mask_u8)
    if pts is None or len(pts) < 5:
        return None
    (_, _), (w, h), ang = cv2.minAreaRect(pts)
    if w <= 0 or h <= 0:
        return None
    # minAreaRect's angle names the `w` edge; rotate to name the LONG edge.
    major, minor = (w, h) if w >= h else (h, w)
    deg = ang if w >= h else ang + 90.0
    return deg % 180.0, float(major), float(minor), float(w * h)


def _angdiff(a: float, b: float) -> float:
    """Difference of two undirected axes, in [0,90]."""
    d = abs(a - b) % 180.0
    return min(d, 180.0 - d)


def measure_mask(
    mask, reported_aspect_ratio: Optional[float] = None, th: Optional[QualityThresholds] = None
) -> MaskQuality:
    """Score one mask. Never raises."""
    th = th or QualityThresholds()
    q = MaskQuality()
    try:
        m = _as_u8(mask)
        if m is None:
            q.reasons.append("no mask")
            return q
        q.pixels = int(m.sum())
        if q.pixels < th.min_pixels:
            q.reasons.append(f"only {q.pixels}px")
            return q

        axis = _major_axis(m)
        if axis is None:
            q.reasons.append("degenerate OBB")
            return q
        deg, major, minor, obb_area = axis
        q.major_axis_deg = deg
        q.mask_aspect_ratio = major / max(minor, 1e-6)
        q.fill = q.pixels / max(obb_area, 1e-6)

        hull = cv2.convexHull(cv2.findNonZero(m))
        q.solidity = q.pixels / max(cv2.contourArea(hull), 1e-6)

        n, _, stats, _ = cv2.connectedComponentsWithStats(m, connectivity=8)
        if n > 1:
            areas = stats[1:, cv2.CC_STAT_AREA]
            q.component_frac = float(areas.max()) / max(q.pixels, 1)

        q.angle_swing_deg = _axis_swing(m, deg, th.perturb_frac)

        if reported_aspect_ratio is not None:
            q.ar_inflation = float(reported_aspect_ratio) / max(q.mask_aspect_ratio, 1e-6)

        q.measured = True
        _verdict(q, th, reported_aspect_ratio)
        return q
    except Exception as exc:  # noqa: BLE001 - scoring must never stop perception
        logger.warning("[mask-quality] measure failed (%s)", exc)
        q.reasons.append("measure error")
        return q


def _axis_swing(m: np.ndarray, base_deg: float, perturb_frac: float) -> float:
    """Max axis movement under a small erode and dilate.

    This is the direct test of whether the axis is real or is being set by a
    handful of boundary pixels.
    """
    k = max(1, int(round(perturb_frac * math.sqrt(float(m.sum())))))
    kern = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2 * k + 1, 2 * k + 1))
    swing = 0.0
    for op in (cv2.MORPH_ERODE, cv2.MORPH_DILATE):
        pert = cv2.morphologyEx(m, op, kern)
        if int(pert.sum()) < _MIN_PIXELS:
            continue  # eroded away: contributes no evidence either way
        axis = _major_axis(pert)
        if axis is not None:
            swing = max(swing, _angdiff(base_deg, axis[0]))
    return swing


def _verdict(q: MaskQuality, th: QualityThresholds, reported_ar) -> None:
    """Fold the metrics into geometry_trust and angle_trustworthy."""
    checks = []
    if q.fill < th.min_fill:
        q.reasons.append(f"fill {q.fill:.2f}<{th.min_fill:.2f}")
        checks.append(0.0)
    else:
        checks.append(min(1.0, (q.fill - th.min_fill) / max(1e-6, 1.0 - th.min_fill)))

    if q.solidity < th.min_solidity:
        q.reasons.append(f"solidity {q.solidity:.2f}<{th.min_solidity:.2f}")
        checks.append(0.0)
    else:
        checks.append(1.0)

    if q.component_frac < th.min_component_frac:
        q.reasons.append(f"fragmented cc={q.component_frac:.2f}")
        checks.append(0.0)
    else:
        checks.append(1.0)

    if q.angle_swing_deg > th.max_angle_swing_deg:
        q.reasons.append(f"axis swings {q.angle_swing_deg:.1f}deg")
        checks.append(0.0)
    else:
        checks.append(1.0 - q.angle_swing_deg / max(th.max_angle_swing_deg, 1e-6))

    # Directional on purpose. Backprojection noise STRETCHES the world OBB, it
    # does not shrink it, so reported >> mask is a fabricated axis. The reverse
    # (mask longer than world) is ordinary perspective foreshortening, benign.
    inflated = reported_ar is not None and q.ar_inflation > th.max_ar_inflation
    q.ar_inflated = bool(inflated)
    if inflated:
        q.reasons.append(
            f"reported ar {float(reported_ar):.2f} is {q.ar_inflation:.1f}x the "
            f"mask ar {q.mask_aspect_ratio:.2f}"
        )
        checks.append(0.0)

    # The record claims elongation the mask does not have, so there is no
    # major axis to align a wrist to.
    round_mask = q.mask_aspect_ratio < th.min_mask_ar
    q.shape_round = bool(round_mask)
    if round_mask:
        q.reasons.append(f"mask ar {q.mask_aspect_ratio:.2f} is round")

    q.geometry_trust = float(min(checks)) if checks else 0.0
    # A yaw needs BOTH a stable axis and a shape that actually has one.
    q.angle_trustworthy = (
        q.measured
        and q.angle_swing_deg <= th.max_angle_swing_deg
        and q.fill >= th.min_fill
        and q.component_frac >= th.min_component_frac
        and not inflated
        and not round_mask
    )
    # No usable axis -> 0, so a min() against perception's obb_confidence
    # vetoes. Deliberately NOT scaled by elongation: mask_ar shorter than the
    # world OBB is foreshortening, and min_mask_ar above is already the one
    # elongation policy. This is a stability/quality number only.
    q.axis_trust = float(q.geometry_trust) if q.angle_trustworthy else 0.0

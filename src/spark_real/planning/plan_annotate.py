"""
Planner-facing scene annotation.

The executor's overlay (`_annotate_image`) draws a tint and a contour, which
is enough for a human to read the labels but not enough for the planner to
judge SHAPE. This module adds what the grasp-strategy decision needs: the
mask's oriented bounding box, an arrow along its major axis, and the label
with its detection confidence -- so a 24%-confidence blob is visibly a blob
and not an elongated tool.

Pure drawing. No detection, no depth, no I/O. Total on degenerate masks.
"""

from __future__ import annotations

import cv2
import numpy as np
from PIL import Image, ImageDraw
from spark_real.utils.fonts import load_font

# Same palette as the executor overlay so the two views stay comparable.
PALETTE = (
    (255, 60, 60),
    (60, 220, 120),
    (60, 120, 255),
    (255, 200, 40),
    (220, 60, 220),
    (60, 220, 220),
    (255, 140, 40),
    (140, 255, 40),
    (40, 140, 255),
)

# Below this many mask pixels an OBB is meaningless; label only.
_MIN_OBB_PIXELS = 8


def _mask_points(mask, shape) -> np.ndarray | None:
    """
    Nx1x2 int32 point array of the mask's set pixels, or None.
    """
    if mask is None:
        return None
    mask = np.asarray(mask)
    if mask.ndim != 2 or mask.shape != shape:
        return None
    ys, xs = np.nonzero(mask > 0)
    if xs.size < _MIN_OBB_PIXELS:
        return None
    return np.stack([xs, ys], axis=1).astype(np.int32).reshape(-1, 1, 2)


def _obb(points):
    """
    (box_corners, centre, major_axis_unit_vec, (w, h), angle_deg) or None.
    """
    if points is None:
        return None
    rect = cv2.minAreaRect(points)
    (cx, cy), (w, h), angle = rect
    if not np.isfinite([cx, cy, w, h, angle]).all():
        return None
    box = cv2.boxPoints(rect).astype(np.int32)
    # cv2 angle is that of the `w` edge; the major axis is the longer edge.
    major_deg = angle if w >= h else angle + 90.0
    rad = np.deg2rad(major_deg)
    return box, (cx, cy), (float(np.cos(rad)), float(np.sin(rad))), (w, h), major_deg


def _text_with_outline(draw, xy, text, fill, font):
    x, y = xy
    for dx in (-1, 0, 1):
        for dy in (-1, 0, 1):
            if dx or dy:
                draw.text((x + dx, y + dy), text, fill=(0, 0, 0), font=font)
    draw.text((x, y), text, fill=fill, font=font)


def annotate_for_planner(rgb, detections, font_size: int = 16):
    """
    Draw tint + contour + OBB + major-axis arrow + "label conf%" per detection.

    Args:
        rgb: HxWx3 uint8 image. Not modified.
        detections: objects with ``mask``, ``label``, ``confidence`` and
            optionally ``centroid_2d`` / ``bbox``. Anything missing is skipped.

    Returns:
        A new HxWx3 uint8 image.
    """
    vis = np.ascontiguousarray(np.asarray(rgb).copy())
    if vis.ndim == 2:
        vis = cv2.cvtColor(vis, cv2.COLOR_GRAY2RGB)
    shape = vis.shape[:2]

    geometry = []
    for i, det in enumerate(detections or []):
        colour = PALETTE[i % len(PALETTE)]
        colour_arr = np.array(colour)
        mask = getattr(det, "mask", None)
        points = _mask_points(mask, shape)

        if mask is not None and np.asarray(mask).shape == shape:
            sel = np.asarray(mask) > 0
            if sel.any():
                vis[sel] = (vis[sel] * 0.55 + colour_arr * 0.45).astype(np.uint8)
                contours, _ = cv2.findContours(
                    sel.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
                )
                cv2.drawContours(vis, contours, -1, colour, 2)

        obb = _obb(points)
        anchor = getattr(det, "centroid_2d", None)
        if obb is not None:
            box, centre, axis, (w, h), _deg = obb
            cv2.polylines(vis, [box], True, colour, 2, cv2.LINE_AA)
            half = max(w, h) * 0.5
            cx, cy = centre
            tip = (int(cx + axis[0] * half), int(cy + axis[1] * half))
            tail = (int(cx - axis[0] * half), int(cy - axis[1] * half))
            cv2.arrowedLine(vis, tail, tip, colour, 2, cv2.LINE_AA, tipLength=0.18)
            anchor = anchor or centre
        geometry.append((colour, anchor))

    pil = Image.fromarray(vis)
    draw = ImageDraw.Draw(pil)
    font = load_font(font_size)
    for det, (colour, anchor) in zip(detections or [], geometry):
        if anchor is None:
            continue
        conf = getattr(det, "confidence", None)
        text = str(getattr(det, "label", "?"))
        if conf is not None:
            text = f"{text} {float(conf):.0%}"
        x = int(np.clip(anchor[0] + 14, 0, shape[1] - 1))
        y = int(np.clip(anchor[1] - 10, 0, shape[0] - 1))
        _text_with_outline(draw, (x, y), text, colour, font)

    return np.array(pil)

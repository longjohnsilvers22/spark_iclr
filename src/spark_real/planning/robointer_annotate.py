"""
Overlays that make the RoboInter coordinate space answerable, and readable.

:func:`draw_coordinate_grid` composes ON TOP of
``plan_annotate.annotate_for_planner``'s output; it does not replace it. A
faint 0..1000 grid with edge ticks gives the planner a coordinate frame and
costs the model no tokens.

:func:`draw_annotations` draws a returned annotation back onto the frame:
contact point, boxes, trace. That is the check on the model -- a proposal that
looks wrong is visible immediately, before it is ever consumed.

Pure drawing, no I/O, no detection. Both functions return a NEW array.
"""

from __future__ import annotations

from typing import Iterable, Tuple

import cv2
import numpy as np

from spark_real.planning.robointer import COORD_SCALE, NodeAnnotation

# Grid line every 100/1000 of the frame, labelled every 200 to stay legible
# on a 640x480 capture.
_GRID_STEP = 100
_LABEL_EVERY = 200

_GRID_COLOR = (150, 150, 150)
_GRID_ALPHA = 0.30
_TICK_COLOR = (255, 255, 255)

_CONTACT_COLOR = (0, 255, 255)
_OBJECT_COLOR = (0, 200, 255)
_PLACE_COLOR = (120, 255, 120)
_AFFORD_COLOR = (255, 160, 60)
_TRACE_COLOR = (255, 80, 255)


def draw_coordinate_grid(
    rgb: np.ndarray,
    step: int = _GRID_STEP,
    label_every: int = _LABEL_EVERY,
    alpha: float = _GRID_ALPHA,
) -> np.ndarray:
    """Faint 0..1000 grid with edge ticks. Returns a new HxWx3 uint8 array."""
    img = np.ascontiguousarray(np.asarray(rgb)).copy()
    if img.ndim != 3 or img.shape[2] != 3:
        return img
    h, w = img.shape[:2]
    step = max(10, int(step))

    overlay = img.copy()
    for permille in range(step, int(COORD_SCALE), step):
        x = int(round(permille / COORD_SCALE * (w - 1)))
        y = int(round(permille / COORD_SCALE * (h - 1)))
        cv2.line(overlay, (x, 0), (x, h - 1), _GRID_COLOR, 1, cv2.LINE_AA)
        cv2.line(overlay, (0, y), (w - 1, y), _GRID_COLOR, 1, cv2.LINE_AA)
    cv2.addWeighted(overlay, float(alpha), img, 1.0 - float(alpha), 0.0, dst=img)

    # Ticks are drawn opaque, outside the grid blend, so they stay readable
    # over a bright tabletop.
    for permille in range(0, int(COORD_SCALE) + 1, max(step, int(label_every))):
        x = int(round(permille / COORD_SCALE * (w - 1)))
        y = int(round(permille / COORD_SCALE * (h - 1)))
        _label(img, str(permille), (min(x + 2, w - 26), 12))
        if permille:
            _label(img, str(permille), (2, min(y + 12, h - 3)))
    return img


def _label(img, text: str, org: Tuple[int, int], color=_TICK_COLOR) -> None:
    cv2.putText(img, text, org, cv2.FONT_HERSHEY_SIMPLEX, 0.32, (0, 0, 0), 3, cv2.LINE_AA)
    cv2.putText(img, text, org, cv2.FONT_HERSHEY_SIMPLEX, 0.32, color, 1, cv2.LINE_AA)


def _px(pt, size: Tuple[int, int]) -> Tuple[int, int]:
    w, h = size
    return (int(round(pt.x * (w - 1))), int(round(pt.y * (h - 1))))


def draw_annotations(
    rgb: np.ndarray,
    annotations: Iterable[NodeAnnotation],
    show_subtask: bool = True,
) -> np.ndarray:
    """Draw what the planner proposed back onto the frame it read."""
    img = np.ascontiguousarray(np.asarray(rgb)).copy()
    if img.ndim != 3 or img.shape[2] != 3:
        return img
    h, w = img.shape[:2]
    size = (w, h)

    for i, ann in enumerate(annotations or []):
        for box, color, tag in (
            (ann.object_box, _OBJECT_COLOR, "obj"),
            (ann.affordance_box, _AFFORD_COLOR, "afford"),
            (ann.placement_proposal, _PLACE_COLOR, "place"),
        ):
            if box is None:
                continue
            b = box.ordered()
            p1, p2 = _px(b.corners()[0], size), _px(b.corners()[2], size)
            cv2.rectangle(img, p1, p2, color, 2, cv2.LINE_AA)
            _label(img, tag, (p1[0] + 3, max(11, p1[1] - 4)), color)

        if ann.trace:
            pts = np.array([_px(p, size) for p in ann.trace], dtype=np.int32)
            cv2.polylines(img, [pts], False, _TRACE_COLOR, 2, cv2.LINE_AA)
            for p in pts:
                cv2.circle(img, tuple(int(v) for v in p), 3, _TRACE_COLOR, -1, cv2.LINE_AA)

        if ann.contact_point is not None:
            c = _px(ann.contact_point, size)
            cv2.drawMarker(img, c, _CONTACT_COLOR, cv2.MARKER_CROSS, 16, 2, cv2.LINE_AA)
            cv2.circle(img, c, 7, _CONTACT_COLOR, 1, cv2.LINE_AA)

        if show_subtask and ann.subtask:
            _label(img, f"{i}: {ann.subtask[:60]}", (6, h - 8 - 13 * i))
    return img

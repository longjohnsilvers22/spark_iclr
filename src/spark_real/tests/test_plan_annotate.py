"""
Offline tests for the planner-facing scene overlay.

Synthetic images and masks only; no camera, no SAM3. The point of these is
totality: annotate_for_planner runs on the planning path, so a degenerate mask
must not take the run down.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import cv2
import numpy as np
import pytest

from spark_real.planning.plan_annotate import annotate_for_planner

H, W = 240, 320


@dataclass
class FakeDet:
    label: str = "obj"
    confidence: float = 0.9
    mask: np.ndarray | None = None
    centroid_2d: tuple | None = None
    bbox: tuple | None = None
    camera: str = "birdview"
    extras: dict = field(default_factory=dict)


def blank():
    rgb = np.zeros((H, W, 3), np.uint8)
    rgb[:] = (30, 30, 30)
    return rgb


def elongated_mask():
    m = np.zeros((H, W), np.uint8)
    cv2.rectangle(m, (60, 110), (250, 130), 1, -1)
    return m


def test_returns_new_image_of_the_same_shape_and_does_not_mutate_input():
    rgb = blank()
    original = rgb.copy()
    det = FakeDet(mask=elongated_mask(), centroid_2d=(155, 120))
    out = annotate_for_planner(rgb, [det])
    assert out.shape == rgb.shape
    assert out.dtype == np.uint8
    assert np.array_equal(rgb, original)
    assert not np.array_equal(out, original)


def test_draws_inside_the_obb():
    rgb = blank()
    mask = elongated_mask()
    out = annotate_for_planner(rgb, [FakeDet(mask=mask, centroid_2d=(155, 120))])
    inside = out[110:130, 60:250]
    # Tint + contour + arrow: the mask interior must be substantially repainted.
    changed = np.any(inside != np.array([30, 30, 30], np.uint8), axis=-1)
    assert changed.mean() > 0.5

    # The OBB rectangle is drawn on the mask border, outside the tinted core.
    border = out[104:110, 60:250]
    assert np.any(border != np.array([30, 30, 30], np.uint8))


@pytest.mark.parametrize(
    "mask_name",
    ["empty", "single_pixel", "full_frame", "none", "wrong_shape", "float"],
)
def test_degenerate_masks_are_total(mask_name):
    masks = {
        "empty": np.zeros((H, W), np.uint8),
        "single_pixel": None,
        "full_frame": np.ones((H, W), np.uint8),
        "none": None,
        "wrong_shape": np.ones((10, 10), np.uint8),
        "float": (np.random.default_rng(0).random((H, W)) > 0.5).astype(np.float32),
    }
    if mask_name == "single_pixel":
        m = np.zeros((H, W), np.uint8)
        m[120, 160] = 1
        masks["single_pixel"] = m
    det = FakeDet(mask=masks[mask_name], centroid_2d=(160, 120))
    out = annotate_for_planner(blank(), [det])
    assert out.shape == (H, W, 3)
    assert out.dtype == np.uint8


def test_no_detections_and_no_centroid_are_fine():
    assert annotate_for_planner(blank(), []).shape == (H, W, 3)
    assert annotate_for_planner(blank(), None).shape == (H, W, 3)
    det = FakeDet(mask=elongated_mask(), centroid_2d=None)
    assert annotate_for_planner(blank(), [det]).shape == (H, W, 3)


def test_label_includes_confidence_percent():
    # Rendered text is hard to assert on pixel-wise; assert the two labels
    # differ, which they only do if confidence reaches the drawing.
    mask = elongated_mask()
    low = annotate_for_planner(blank(), [FakeDet("plushie", 0.24, mask, (155, 120))])
    high = annotate_for_planner(blank(), [FakeDet("plushie", 0.97, mask, (155, 120))])
    assert not np.array_equal(low, high)


def test_many_detections_cycle_the_palette():
    dets = []
    rng = np.random.default_rng(1)
    for i in range(12):
        m = np.zeros((H, W), np.uint8)
        x = 10 + i * 24
        cv2.rectangle(m, (x, 40), (x + 18, 90), 1, -1)
        dets.append(FakeDet(f"obj {i}", float(rng.random()), m, (x + 9, 65)))
    out = annotate_for_planner(blank(), dets)
    assert out.shape == (H, W, 3)
    assert len(np.unique(out.reshape(-1, 3), axis=0)) > 8


def test_grayscale_input_is_promoted():
    gray = np.full((H, W), 40, np.uint8)
    out = annotate_for_planner(gray, [FakeDet(mask=elongated_mask(), centroid_2d=(155, 120))])
    assert out.shape == (H, W, 3)

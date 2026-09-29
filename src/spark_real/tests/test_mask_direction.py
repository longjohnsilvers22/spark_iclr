# Directed mask geometry: heavy_end_sign + best_fit_rotation.
#
# Background (measured 2026-08-20): "place the screwdriver in its slot" seated
# the tool 180 deg off whenever the operator had not pre-aligned it -- axes
# parallel, tip at the wrong end. The place resolves a LINE (PCA axis, mod pi);
# these two functions supply the missing directed half, and both were imported
# in five places without ever having been implemented. These tests pin the
# conventions the executors rely on:
#   - heavy_end_sign: +1 means the fat end lies toward +(cos a, sin a), image
#     coords; 0.0 is an abstention and must be common for symmetric shapes.
#   - best_fit_rotation: rotating the held mask by +angle (same atan2-in-image
#     convention as _pca_obb) overlays it on the target; margin collapses for
#     shapes that cannot decide a direction.

import numpy as np
import pytest

cv2 = pytest.importorskip("cv2")

from spark_real.perception.mask_geometry import (  # noqa: E402
    _pca_obb,
    best_fit_rotation,
    heavy_end_sign,
)

H = W = 320
JIGSAW_MIN_MARGIN = 0.05  # executor_motion.PlaceMixin gate
JIGSAW_MIN_IOU = 0.20


def _wrap(a):
    return (a + np.pi) % (2 * np.pi) - np.pi


def _screwdriver(angle_rad, length_px=140, shaft_w=10, handle_w=28,
                 handle_frac=0.42):
    """Shaft plus fat handle at the +axis end. Returns (mask, handle_dir)."""
    m = np.zeros((H, W), np.uint8)
    u = np.array([np.cos(angle_rad), np.sin(angle_rad)])
    v = np.array([-np.sin(angle_rad), np.cos(angle_rad)])
    c = np.array([W / 2, H / 2])
    L, hf = length_px, handle_frac

    def rect(t0, t1, half_w):
        p = np.array(
            [c + t0 * u + half_w * v, c + t1 * u + half_w * v,
             c + t1 * u - half_w * v, c + t0 * u - half_w * v], np.int32)
        cv2.fillPoly(m, [p], 1)

    rect(-L / 2, L / 2 - hf * L, shaft_w / 2)
    rect(L / 2 - hf * L, L / 2, handle_w / 2)
    return m, angle_rad


def _slot(angle_rad, dilate_px=6, **kw):
    m, d = _screwdriver(angle_rad, **kw)
    k = cv2.getStructuringElement(
        cv2.MORPH_ELLIPSE, (2 * dilate_px + 1, 2 * dilate_px + 1))
    return cv2.dilate(m, k), d


def _heavy_dir(mask):
    ang, _, _, _ = _pca_obb(mask)
    sgn = heavy_end_sign(mask, ang)
    if sgn == 0.0:
        return ang, None
    return ang, _wrap(ang + (0.0 if sgn > 0 else np.pi))


# ------------------------------------------------------------------ signs


@pytest.mark.parametrize("deg", [7.0, 90.0, -14.0, 33.0, 141.0, -100.0])
def test_heavy_end_sign_points_at_the_handle(deg):
    # 7 and 90 deg are the axes of the 2026-08-20 10:56 failing run.
    m, true_dir = _screwdriver(np.deg2rad(deg))
    _, d = _heavy_dir(m)
    assert d is not None, "screwdriver silhouette must be decidable"
    assert abs(_wrap(d - true_dir)) < np.pi / 2


def test_heavy_end_sign_noisy_trials():
    rng = np.random.default_rng(0)
    nz = corr = 0
    for _ in range(50):
        a = rng.uniform(-np.pi, np.pi)
        m, true_dir = _screwdriver(a)
        # boundary bites/bumps
        ys, xs = np.where(m > 0)
        for _b in range(30):
            i = rng.integers(0, len(xs))
            cv2.circle(m, (int(xs[i]), int(ys[i])), int(rng.uniform(1, 4)),
                       int(rng.random() < 0.5), -1)
        _, d = _heavy_dir(m)
        if d is not None:
            nz += 1
            corr += abs(_wrap(d - true_dir)) < np.pi / 2
    assert nz >= 45, f"abstained too often: {nz}/50 decided"
    assert corr == nz, f"{nz - corr} confidently WRONG signs"


def test_bare_handle_abstains():
    # A 'screwdriver handle' sub-part mask is symmetric end-for-end; a sign
    # here would be noise. This is why _record_held_axis upgrades to the
    # parent 'screwdriver' mask before measuring.
    m = np.zeros((H, W), np.uint8)
    cv2.rectangle(m, (120, 140), (200, 180), 1, -1)
    ang, _, _, _ = _pca_obb(m)
    assert heavy_end_sign(m, ang) == 0.0


def test_detached_fragment_never_flips_the_sign():
    # Largest-component + axis-agreement hardening: a satellite blob may force
    # an abstention, never a confidently wrong direction (2026-08-20: 10/200
    # wrong before the axis-agreement gate, 0/200 after).
    rng = np.random.default_rng(123)
    wrong = 0
    for _ in range(60):
        a = rng.uniform(-np.pi, np.pi)
        m, true_dir = _screwdriver(a)
        r = int(np.sqrt(0.20 * m.sum() / np.pi))
        cv2.circle(m, (int(rng.integers(30, W - 30)),
                       int(rng.integers(30, H - 30))), r, 1, -1)
        _, d = _heavy_dir(m)
        if d is not None and abs(_wrap(d - true_dir)) >= np.pi / 2:
            wrong += 1
    assert wrong == 0


# ------------------------------------------------------------------ fit


@pytest.mark.parametrize("delta_deg", [0, 30, 90, 150, 180, -120, -45])
def test_best_fit_rotation_recovers_known_rotation(delta_deg):
    held, _ = _screwdriver(np.deg2rad(10))
    tgt, _ = _screwdriver(np.deg2rad(10 + delta_deg))
    ang, iou, margin = best_fit_rotation(held, tgt)
    assert iou >= 0.9
    assert margin >= JIGSAW_MIN_MARGIN
    assert abs(_wrap(ang - np.deg2rad(delta_deg))) < np.deg2rad(6)


def test_best_fit_rotation_flags_the_180_off_slot():
    # The failing configuration: tool at ~90 deg, slot at ~7 deg with the
    # handle recess at the FAR end. The fit must land ~180 away from the
    # mod-pi axis alignment (-83), i.e. ~+97, with a clear margin.
    held, _ = _screwdriver(np.deg2rad(90))
    slot, _ = _slot(np.deg2rad(7 + 180))
    ang, iou, margin = best_fit_rotation(held, slot)
    assert iou >= JIGSAW_MIN_IOU
    assert margin >= JIGSAW_MIN_MARGIN
    axis_rot = _wrap(np.deg2rad(7 - 90))  # reduced axis alignment
    assert abs(_wrap(ang - axis_rot)) > np.pi / 2, (
        "fit failed to notice the tool would seat end-for-end")


def test_best_fit_rotation_abstains_on_round_target():
    bowl = np.zeros((H, W), np.uint8)
    cv2.circle(bowl, (W // 2, H // 2), 80, 1, -1)
    held, _ = _screwdriver(np.deg2rad(33))
    _, _, margin = best_fit_rotation(held, bowl)
    assert margin < JIGSAW_MIN_MARGIN


def test_best_fit_rotation_abstains_on_symmetric_rectangle():
    t = np.zeros((H, W), np.uint8)
    cv2.rectangle(t, (90, 130), (230, 190), 1, -1)
    h = np.zeros((H, W), np.uint8)
    cv2.rectangle(h, (100, 140), (220, 180), 1, -1)
    _, _, margin = best_fit_rotation(h, t)
    assert margin < JIGSAW_MIN_MARGIN


def test_fit_agrees_with_pca_axis_difference():
    # The jigsaw rung compares the fit against wrap(mt - ang_h) REDUCED into
    # [-pi/2, pi/2] (a PCA eigenvector's sign is arbitrary, so the raw branch
    # is meaningless -- measured 36/50 correct raw, 50/50 reduced).
    rng = np.random.default_rng(7)
    for _ in range(10):
        a_h = rng.uniform(-np.pi, np.pi)
        flip = bool(rng.integers(0, 2))
        a_t = rng.uniform(-np.pi, np.pi)
        slot_dir = _wrap(a_t + np.pi) if flip else a_t
        held, held_dir = _screwdriver(a_h)
        slot, _ = _slot(slot_dir)
        ang, iou, margin = best_fit_rotation(held, slot)
        assert margin >= JIGSAW_MIN_MARGIN
        ang_h, _, _, _ = _pca_obb(held)
        mt, _, _, _ = _pca_obb(slot)
        axis_rot = _wrap(mt - ang_h)
        if axis_rot > np.pi / 2:
            axis_rot -= np.pi
        elif axis_rot < -np.pi / 2:
            axis_rot += np.pi
        wants_flip = abs(_wrap(ang - axis_rot)) > np.pi / 2
        backwards = abs(_wrap(slot_dir - (held_dir + axis_rot))) > np.pi / 2
        assert wants_flip == backwards

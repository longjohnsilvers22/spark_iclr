"""
Slot detector for compartmentalized trays / silverware drawers.

Given a birdview RGB-D capture and a SAM3 tray mask, find the individual
compartments ("slots") inside the tray so a heterogeneous batch of
objects (knife, fork, spoon) can each be placed in its own slot.

The key signal is surface normals computed from the depth image:
flat slot FLOORS have normals pointing world +Z, while slot DIVIDERS
have normals pointing horizontally. Pixels that are (a) inside the
tray mask, (b) horizontal-up by normal, and (c) at the LOW end of the
world-Z distribution form per-slot connected components.

Pipeline (mirrors the style in ``spark_perception.py``):

  1. Sobel on depth(meters) plus K (intrinsics) gives per-pixel cam-frame
     normals N_cam.
  2. Rotate to base frame: N_base = R_cam_to_base @ N_cam.
  3. Keep up-facing pixels (N_base.z > 0.9), then keep those within
     slot_floor_band_m of the 25th-percentile world-Z.
  4. Connected components on the surviving pixels become slots.
  5. Order slots by major-axis projection (world XY PCA of the tray mask).

Returns a list of dicts; each describes one slot in world frame.
Falls back to N evenly-spaced poses along the tray's major axis when
the normal-based path produces too few clusters (flat tray, no
visible dividers, bad depth).

No SPARK server / no robot motion is touched by this module; it's a
pure perception utility intended to be called by the executor BEFORE
deciding placement targets.
"""

from __future__ import annotations

import logging
from typing import List, Optional

import cv2
import numpy as np

logger = logging.getLogger(__name__)


# Internals.


def _compute_normals_from_depth(
    depth_m: np.ndarray,
    K: np.ndarray,
    smooth_ksize: int = 5,
) -> np.ndarray:
    """
    Per-pixel surface normals in camera frame from a metric depth map.

    For each pixel (u, v) with depth d:
        P_cam = ((u-cx)*d/fx, (v-cy)*d/fy, d)            # OpenCV
    Tangent vectors come from finite differences via Sobel on the depth
    image; the normal is the normalised cross product of the row/col
    tangents. Output normals always point toward the camera (z < 0 in
    OpenCV camera frame); the sign is handled on conversion to world.

    A small bilateral pre-smoothing tames depth quantisation noise so
    horizontal slot floors don't look bumpy.

    Returns (H, W, 3) float32. Invalid pixels (d<=0) get zeros.
    """
    h, w = depth_m.shape
    fx = float(K[0, 0])
    fy = float(K[1, 1])
    cx = float(K[0, 2])
    cy = float(K[1, 2])

    # Pre-smooth depth to suppress single-pixel noise without blurring
    # across slot dividers (bilateral preserves depth edges).
    d = depth_m.astype(np.float32)
    valid = d > 0.01
    d_smooth = cv2.bilateralFilter(d, d=5, sigmaColor=0.01, sigmaSpace=5)
    # Bilateral can zero out invalid neighbourhoods; keep originals there.
    d_smooth = np.where(valid, d_smooth, d)

    # Build pixel-grid coordinates once.
    us = np.arange(w, dtype=np.float32)
    vs = np.arange(h, dtype=np.float32)
    uu, vv = np.meshgrid(us, vs)

    # Camera-frame XYZ for every pixel.
    X = (uu - cx) * d_smooth / fx
    Y = (vv - cy) * d_smooth / fy
    Z = d_smooth

    # Sobel along u (columns) and v (rows). Larger ksize averages out
    # more depth noise on the slot floor.
    ksize = smooth_ksize if smooth_ksize in (3, 5, 7) else 5
    dX_du = cv2.Sobel(X, cv2.CV_32F, 1, 0, ksize=ksize)
    dY_du = cv2.Sobel(Y, cv2.CV_32F, 1, 0, ksize=ksize)
    dZ_du = cv2.Sobel(Z, cv2.CV_32F, 1, 0, ksize=ksize)
    dX_dv = cv2.Sobel(X, cv2.CV_32F, 0, 1, ksize=ksize)
    dY_dv = cv2.Sobel(Y, cv2.CV_32F, 0, 1, ksize=ksize)
    dZ_dv = cv2.Sobel(Z, cv2.CV_32F, 0, 1, ksize=ksize)

    # Normal = (dP/dv) x (dP/du). Sign chosen so a flat surface in
    # front of the camera yields N_cam ~ (0, 0, -1) (pointing back at
    # the camera in OpenCV convention).
    nx = dY_dv * dZ_du - dZ_dv * dY_du
    ny = dZ_dv * dX_du - dX_dv * dZ_du
    nz = dX_dv * dY_du - dY_dv * dX_du

    norm = np.sqrt(nx * nx + ny * ny + nz * nz) + 1e-9
    nx /= norm
    ny /= norm
    nz /= norm

    normals = np.stack([nx, ny, nz], axis=-1)
    normals[~valid] = 0.0
    return normals


def _tray_axes_world(
    tray_mask: np.ndarray,
    depth_m: np.ndarray,
    K: np.ndarray,
    T_cam_to_base: np.ndarray,
) -> Optional[dict]:
    """
    World-XY PCA on the tray mask. Returns mean, major, minor (XY).

    Mirrors ``_world_xy_pca_obb`` in spark_perception.py, same idea,
    same OpenCV-convention backprojection. Returns None if the tray
    has too few valid depth pixels for a stable axis.
    """
    ys, xs = np.where(tray_mask > 0)
    if len(xs) < 50:
        return None
    depths = depth_m[ys, xs].astype(np.float64)
    valid = (depths > 0.01) & (depths < 10.0)
    if int(valid.sum()) < 50:
        return None
    xs_v = xs[valid].astype(np.float64)
    ys_v = ys[valid].astype(np.float64)
    d_v = depths[valid]
    fx = float(K[0, 0])
    fy = float(K[1, 1])
    cx_k = float(K[0, 2])
    cy_k = float(K[1, 2])
    x_cam = (xs_v - cx_k) * d_v / fx
    y_cam = (ys_v - cy_k) * d_v / fy
    z_cam = d_v
    pts_cam = np.column_stack([x_cam, y_cam, z_cam])
    R = T_cam_to_base[:3, :3]
    t = T_cam_to_base[:3, 3]
    pts_world = (R @ pts_cam.T).T + t
    xy = pts_world[:, :2]
    mean_xy = xy.mean(axis=0)
    cov = np.cov((xy - mean_xy).T)
    if cov.ndim != 2 or cov.shape != (2, 2):
        return None
    eigvals, eigvecs = np.linalg.eigh(cov)
    major = eigvecs[:, 1]  # largest variance
    minor = eigvecs[:, 0]
    proj_mj = (xy - mean_xy) @ major
    proj_mn = (xy - mean_xy) @ minor
    return {
        "mean_xy": mean_xy,
        "major": major,
        "minor": minor,
        "major_len_m": float(proj_mj.max() - proj_mj.min()),
        "minor_len_m": float(proj_mn.max() - proj_mn.min()),
        "z_med": float(np.median(pts_world[:, 2])),
    }


def _backproject_pixels_to_world(
    ys: np.ndarray,
    xs: np.ndarray,
    depth_m: np.ndarray,
    K: np.ndarray,
    T_cam_to_base: np.ndarray,
) -> np.ndarray:
    """
    Backproject a batch of (v, u) image pixels to world XYZ.

    Returns (N, 3) float64 with NaN rows for pixels with invalid depth.
    """
    d = depth_m[ys, xs].astype(np.float64)
    fx = float(K[0, 0])
    fy = float(K[1, 1])
    cx_k = float(K[0, 2])
    cy_k = float(K[1, 2])
    x_cam = (xs.astype(np.float64) - cx_k) * d / fx
    y_cam = (ys.astype(np.float64) - cy_k) * d / fy
    z_cam = d
    pts_cam = np.column_stack([x_cam, y_cam, z_cam])
    R = T_cam_to_base[:3, :3]
    t = T_cam_to_base[:3, 3]
    pts_world = (R @ pts_cam.T).T + t
    pts_world[d <= 0.01] = np.nan
    return pts_world


def _fallback_axis_slots(
    tray_axes: dict,
    n_slots: int,
    lift_offset_m: float,
) -> List[dict]:
    """
    N evenly-spaced poses along the tray's major axis.

    Used when the normal-based detector finds <2 reliable components.
    Slots are spaced so that the OUTERMOST slot centers sit at
    approximately +/-0.5*(major_len - slot_len/N) so the executor's
    release pose is comfortably inside the tray footprint, not on the
    rim.
    """
    if n_slots < 1:
        n_slots = 1
    mean_xy = tray_axes["mean_xy"]
    major = tray_axes["major"]
    L = tray_axes["major_len_m"]
    # Inset by one half-slot from each end so centers don't sit on rims.
    slot_len = L / max(n_slots, 1)
    starts = -L / 2 + slot_len / 2 + slot_len * np.arange(n_slots)
    z = tray_axes["z_med"] + lift_offset_m
    slots = []
    for i, s in enumerate(starts):
        xy = mean_xy + s * major
        slots.append(
            {
                "slot_idx": int(i),
                "world_xyz": np.array([xy[0], xy[1], z], dtype=np.float64),
                "n_pixels": 0,
                "confidence": 0.3,
                "mode": "fallback",
            }
        )
    return slots


# Public API.


def detect_slots(
    rgb: np.ndarray,
    depth_m: np.ndarray,
    tray_mask: np.ndarray,
    K: np.ndarray,
    T_cam_to_base: np.ndarray,
    depth_scale: float = 1.0,
    n_fallback_slots: int = 2,
    min_slot_area_px: int = 200,
    slot_floor_band_m: float = 0.015,
    lift_offset_m: float = 0.04,
    normal_up_threshold: float = 0.9,
) -> List[dict]:
    """
    Detect compartments inside a tray using depth surface normals.

    Args:
        rgb: (H, W, 3) uint8. Only used for shape checks / future RGB
            cues; not consumed by the current detector.
        depth_m: (H, W) float metric depth in METERS, aligned to the
            color frame. Multiplied internally by ``depth_scale``.
        tray_mask: (H, W) bool or uint8. SAM3 mask of the tray.
        K: (3, 3) color-camera intrinsics (OpenCV convention).
        T_cam_to_base: (4, 4) homogeneous transform from camera to
            robot base frame.
        depth_scale: per-camera depth bias correction (see SPARK
            ``depth_scale_correction`` in the handeye JSON).
        n_fallback_slots: number of evenly-spaced slots to emit along
            the tray's major axis when the normal-based detector
            can't find >=2 components. Caller should pass the actual
            object count when known so the fallback matches.
        min_slot_area_px: minimum connected-component area for a
            slot. Smaller components are dropped as noise.
        slot_floor_band_m: world-Z thickness around the 25th-percentile
            up-pixel Z to count as "slot floor".
        lift_offset_m: world-Z offset added to each slot's floor for
            the released-above-floor pose.
        normal_up_threshold: minimum N_base.z to count as "up-facing"
            (default 0.9 ~ within 26 deg of world +Z).

    Returns:
        List of dicts sorted by major-axis projection. Each dict:
            slot_idx: int          (0 is the "low" end along major axis)
            world_xyz: np.ndarray  shape (3,) float64
            n_pixels: int          number of slot-floor pixels in component
            confidence: float      0..1
            mode: str              "normals" or "fallback"

    On total perception failure (no tray axes computable) returns an
    empty list; the caller should treat this the same as "no slots".
    """
    # step 0: input validation + scale
    if depth_m is None or rgb is None or tray_mask is None:
        return []
    if depth_m.ndim != 2:
        return []
    if rgb.shape[:2] != depth_m.shape:
        logger.warning(
            "rgb shape %s and depth shape %s disagree", rgb.shape[:2], depth_m.shape
        )
    if tray_mask.dtype != bool:
        tray_mask = tray_mask > 0
    if K is None or T_cam_to_base is None:
        return []
    K = np.asarray(K, dtype=np.float64)
    T_cam_to_base = np.asarray(T_cam_to_base, dtype=np.float64)

    depth_m = depth_m.astype(np.float32)
    if depth_scale and float(depth_scale) != 1.0:
        depth_m = depth_m * float(depth_scale)

    # step 1: tray axes for ordering / fallback
    tray_axes = _tray_axes_world(tray_mask, depth_m, K, T_cam_to_base)
    if tray_axes is None:
        logger.warning("slot_detector: tray has too few valid-depth pixels")
        return []

    # step 2: per-pixel normals in camera frame, transform to world
    normals_cam = _compute_normals_from_depth(depth_m, K)
    R = T_cam_to_base[:3, :3]
    H, W = depth_m.shape
    # Flatten matrix multiply: (3,3) @ (3, N) -> (3, N).
    n_flat = normals_cam.reshape(-1, 3).T  # (3, H*W)
    n_world_flat = (R @ n_flat).T  # (H*W, 3)
    n_world = n_world_flat.reshape(H, W, 3)

    # Use |Nz| so the test is sign-agnostic: a horizontal surface gives a
    # large |Nz| regardless of which side the camera-ward normal points.
    up_mask = (np.abs(n_world[:, :, 2]) > float(normal_up_threshold)) & tray_mask
    up_mask &= depth_m > 0.01

    n_up = int(up_mask.sum())
    if n_up < min_slot_area_px:
        logger.info(
            "slot_detector: only %d up-pixels in tray (min %d), falling back",
            n_up,
            min_slot_area_px,
        )
        return _fallback_axis_slots(tray_axes, n_fallback_slots, lift_offset_m)

    # step 3: filter up-pixels down to slot floors (lowest world Z)
    ys, xs = np.where(up_mask)
    pts_world = _backproject_pixels_to_world(ys, xs, depth_m, K, T_cam_to_base)
    finite = np.isfinite(pts_world[:, 2])
    if int(finite.sum()) < min_slot_area_px:
        return _fallback_axis_slots(tray_axes, n_fallback_slots, lift_offset_m)
    pts_world = pts_world[finite]
    ys = ys[finite]
    xs = xs[finite]
    z_world = pts_world[:, 2]

    # Slot floor altitude: the deepest cluster of horizontal pixels (the
    # slot interiors). Dividers often have more birdview area than the
    # interiors, so a single percentile is fragile; instead histogram the
    # up-pixel Z values and pick the lowest bin with significant count.
    z_lo = float(np.percentile(z_world, 2))  # robust min (drops noise)
    z_hi = float(np.percentile(z_world, 98))  # robust max
    z_span = max(z_hi - z_lo, 0.001)
    n_bins = 24
    bin_edges = np.linspace(z_lo, z_hi, n_bins + 1)
    counts, _ = np.histogram(z_world, bins=bin_edges)
    # Significance threshold: a bin counts as a "real surface" if it
    # has at least min_slot_area_px / n_bins pixels (uniform-distribution
    # baseline scaled by half).
    sig_thresh = max(int(min_slot_area_px) // (n_bins // 2), 50)
    sig_bins = np.where(counts >= sig_thresh)[0]
    if len(sig_bins) == 0:
        logger.info(
            "slot_detector: no significant horizontal band in Z; "
            "falling back to axis stripes"
        )
        return _fallback_axis_slots(tray_axes, n_fallback_slots, lift_offset_m)
    # LOWEST significant bin = slot-floor altitude (deepest horizontal
    # surface inside the tray mask).
    lo_bin = int(sig_bins[0])
    z_floor = float(bin_edges[lo_bin])
    logger.info(
        "slot_detector: Z range [%.3f, %.3f], slot_floor band starts at "
        "z=%.3f (bin %d/%d), n_sig_bins=%d",
        z_lo,
        z_hi,
        z_floor,
        lo_bin,
        n_bins,
        len(sig_bins),
    )
    floor_keep = z_world < (z_floor + slot_floor_band_m)
    if int(floor_keep.sum()) < min_slot_area_px:
        return _fallback_axis_slots(tray_axes, n_fallback_slots, lift_offset_m)

    floor_ys = ys[floor_keep]
    floor_xs = xs[floor_keep]
    floor_pts = pts_world[floor_keep]

    # Rebuild a 2D mask of just the slot-floor pixels for connected-
    # components labelling.
    slot_floor_mask = np.zeros((H, W), dtype=np.uint8)
    slot_floor_mask[floor_ys, floor_xs] = 1

    # Light morphological cleanup: close 1-pixel gaps that confuse
    # connectedComponents (slot floors aren't perfectly contiguous
    # at the rim), then open to drop salt-and-pepper noise.
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
    slot_floor_mask = cv2.morphologyEx(slot_floor_mask, cv2.MORPH_CLOSE, kernel)
    slot_floor_mask = cv2.morphologyEx(slot_floor_mask, cv2.MORPH_OPEN, kernel)

    # step 4: connected components
    n_labels, labels, stats, _ = cv2.connectedComponentsWithStats(
        slot_floor_mask,
        connectivity=8,
    )

    # Labels: 0 is background; 1..n_labels-1 are components.
    slot_components = []
    for lbl in range(1, n_labels):
        area = int(stats[lbl, cv2.CC_STAT_AREA])
        if area < min_slot_area_px:
            continue
        comp_mask = labels == lbl
        # Restrict to the floor-pixel set (comp_mask is in image coords,
        # but only floor pixels have a depth backprojection; the world
        # points are re-picked from floor_pts at the matching rows).
        comp_pix_in_floor = comp_mask[floor_ys, floor_xs]
        if int(comp_pix_in_floor.sum()) < min_slot_area_px:
            continue
        comp_world = floor_pts[comp_pix_in_floor]
        # Per-axis median for robustness.
        cx_w = float(np.median(comp_world[:, 0]))
        cy_w = float(np.median(comp_world[:, 1]))
        cz_w = float(np.median(comp_world[:, 2]))
        slot_components.append(
            {
                "world_xy": np.array([cx_w, cy_w]),
                "world_z_floor": cz_w,
                "n_pixels": int(comp_pix_in_floor.sum()),
            }
        )

    if len(slot_components) < 2:
        logger.info(
            "slot_detector: found %d slot component(s), falling back to "
            "%d-stripe along major axis",
            len(slot_components),
            n_fallback_slots,
        )
        return _fallback_axis_slots(tray_axes, n_fallback_slots, lift_offset_m)

    # step 5: order slots along the tray's major axis
    mean_xy = tray_axes["mean_xy"]
    major = tray_axes["major"]
    for sc in slot_components:
        sc["proj"] = float((sc["world_xy"] - mean_xy) @ major)
    slot_components.sort(key=lambda s: s["proj"])

    # step 6: build the output list
    # Confidence proxy: pixel count vs the largest component. Trays
    # with very-unequal components likely have a partial occlusion or
    # one slot already full of an object; flag the small ones.
    max_n = max(sc["n_pixels"] for sc in slot_components)
    out: List[dict] = []
    for i, sc in enumerate(slot_components):
        xy = sc["world_xy"]
        z = sc["world_z_floor"] + float(lift_offset_m)
        conf = 0.5 + 0.5 * (sc["n_pixels"] / float(max_n))
        conf = float(min(max(conf, 0.0), 1.0))
        out.append(
            {
                "slot_idx": i,
                "world_xyz": np.array([xy[0], xy[1], z], dtype=np.float64),
                "n_pixels": int(sc["n_pixels"]),
                "confidence": conf,
                "mode": "normals",
            }
        )
    return out

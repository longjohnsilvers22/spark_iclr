"""
Hardware-free geometry helpers for the bimanual t-shirt fold.

Everything in this module is pure NumPy/OpenCV math: no robot driver, no
camera, no franky/franka, no JAX. It is importable on any host so the
bimanual cloth skills (and their tests) can exercise the geometry without
touching hardware.

The math here is a faithful port of the standalone fold
(scripts/fold_tshirt_v3.py):

  - cross-arm frame transforms (right base to/from left base via ZED frame),
  - garment hem-corner extraction from a DEPTH-VALID point cloud
    (lowest-x band, plus/minus-y extremes),
  - sleeve outer-edge (tip) extraction,
  - OBB major/minor axis to jaw yaw.

cv2 is an optional top-level import (set to None when OpenCV is absent) so
the module imports even where OpenCV is unavailable; the transforms and
corner picks are pure NumPy and work regardless. The OBB helpers that need
cv2 raise if it is missing.
"""

from __future__ import annotations

from typing import Tuple

import numpy as np

try:
    import cv2  # optional OpenCV dep
except ImportError:
    cv2 = None

# cross-arm frame transforms


def right_to_left(pos_r, T_r: np.ndarray, T_l: np.ndarray) -> np.ndarray:
    """
    Map a point from the RIGHT arm base frame to the LEFT arm base frame.

    Routes through the shared ZED camera frame (base_R to zed to base_L).
    T_r and T_l are the 4x4 T_zed_to_base matrices for each arm (the same
    matrices detect.load_calibration returns).
    """
    pos_r = np.asarray(pos_r, dtype=float)
    pt_zed = (np.linalg.inv(T_r) @ np.append(pos_r, 1.0))[:3]
    return (T_l @ np.append(pt_zed, 1.0))[:3]


def left_to_right(pos_l, T_r: np.ndarray, T_l: np.ndarray) -> np.ndarray:
    """
    Map a point from the LEFT arm base frame to the RIGHT arm base frame.
    """
    pos_l = np.asarray(pos_l, dtype=float)
    pt_zed = (np.linalg.inv(T_l) @ np.append(pos_l, 1.0))[:3]
    return (T_r @ np.append(pt_zed, 1.0))[:3]


def left_z_correction_at(point_right, T_r: np.ndarray, T_l: np.ndarray) -> float:
    """
    LOCAL, position-dependent left-frame z correction at a grasp point.

    The cross-arm calibration tilt is not a constant origin offset: the same
    physical point reads higher in the left frame than the right, and the
    discrepancy varies across the workspace. Given a point in the RIGHT arm
    base frame, map it into the LEFT frame via the shared ZED frame and return
    ``left_z - right_z``:

        dz(p) = (T_l @ inv(T_r) @ p)[2] - p[2]

    Callers add this dz to that point's LEFT z target so both arms land at the
    same physical height. Use this per grasp/land target rather than a single
    calibration-origin delta. T_r and T_l are the 4x4 T_zed_to_base matrices.
    """
    p_r = np.asarray(point_right, dtype=float)
    p_l = right_to_left(p_r, T_r, T_l)
    return float(p_l[2] - p_r[2])


# point-cloud back-projection


def backproject_cloud(
    mask: np.ndarray,
    depth: np.ndarray,
    K: np.ndarray,
    T_to_base: np.ndarray,
    depth_min: float = 0.1,
    depth_max: float = 3.0,
) -> np.ndarray:
    """
    Back-project the DEPTH-VALID pixels of a mask into a base-frame cloud.

    Returns an (N, 3) array of base-frame points. Pixels with depth outside
    (depth_min, depth_max) are dropped (the depth-valid filter the rollout
    uses before picking hem corners). K is the 3x3 intrinsics, T_to_base the
    4x4 camera-to-base extrinsic.
    """
    mask = np.asarray(mask) > 0
    ys, xs = np.where(mask)
    if xs.size == 0:
        return np.empty((0, 3), dtype=float)
    dpx = depth[ys, xs]
    ok = (dpx > depth_min) & (dpx < depth_max)
    if not np.any(ok):
        return np.empty((0, 3), dtype=float)
    u = xs[ok].astype(float)
    v = ys[ok].astype(float)
    dd = dpx[ok].astype(float)
    xc = (u - K[0, 2]) / K[0, 0] * dd
    yc = (v - K[1, 2]) / K[1, 1] * dd
    homog = np.stack([xc, yc, dd, np.ones_like(dd)], axis=1)
    return (T_to_base @ homog.T).T[:, :3]


def backproject_pixel(
    u: float, v: float, depth: np.ndarray, K: np.ndarray, T_to_base: np.ndarray
) -> np.ndarray:
    """
    Back-project a single (u, v) pixel to a base-frame point.

    Uses a small median window around the pixel for depth robustness, then
    lifts via the pinhole model and the camera-to-base extrinsic.
    """
    u_i = int(round(u))
    v_i = int(round(v))
    h, w = depth.shape[:2]
    u_i = max(0, min(w - 1, u_i))
    v_i = max(0, min(h - 1, v_i))
    win = depth[max(0, v_i - 2) : v_i + 3, max(0, u_i - 2) : u_i + 3]
    valid = win[(win > 0.1) & (win < 3.0)]
    dd = float(np.median(valid)) if valid.size else float(depth[v_i, u_i])
    xc = (u_i - K[0, 2]) / K[0, 0] * dd
    yc = (v_i - K[1, 2]) / K[1, 1] * dd
    return (T_to_base @ np.array([xc, yc, dd, 1.0]))[:3]


# hem corner extraction (lowest-x band, plus/minus-y extremes)


def hem_corners_from_cloud(
    cloud: np.ndarray, x_band_m: float = 0.05, y_band_m: float = 0.02
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Pick the two hem corners of a garment cloud (RIGHT arm base frame).

    The hem is the lowest-x edge of the garment. Within an x_band_m band of
    the minimum x, the corners are the mean of the points at the plus/minus-y
    extremes (within y_band_m of each extreme). Returns (right_corner,
    left_corner) where right/left are CAMERA-relative (camera-right is lower
    y, camera-left is higher y), matching the rollout's right_c / left_c.

    Raises ValueError on an empty cloud.
    """
    cloud = np.asarray(cloud, dtype=float)
    if cloud.shape[0] == 0:
        raise ValueError("hem_corners_from_cloud: empty cloud")
    band = cloud[cloud[:, 0] < cloud[:, 0].min() + x_band_m]
    if band.shape[0] == 0:
        raise ValueError("hem_corners_from_cloud: empty x-band")
    right_c = band[band[:, 1] < band[:, 1].min() + y_band_m].mean(axis=0)
    left_c = band[band[:, 1] > band[:, 1].max() - y_band_m].mean(axis=0)
    return right_c, left_c


def apply_hem_x_offsets(
    right_corner: np.ndarray,
    left_corner: np.ndarray,
    T_r: np.ndarray,
    T_l: np.ndarray,
    hem_x_offset_right_m: float,
    hem_x_offset_left_m: float,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Turn the two camera-relative hem corners into per-arm grasp targets.

    Mirrors the rollout's assignment exactly:

      - RIGHT arm grasps the CAMERA-LEFT corner, plus-x offset (right) applied
        in the RIGHT base frame.
      - LEFT arm grasps the CAMERA-RIGHT corner, plus-x offset (left) applied
        in the RIGHT base frame, then transformed into the LEFT base frame.

    Returns (hem_R, hem_L) in the RIGHT and LEFT base frames respectively.
    """
    hem_R = np.asarray(left_corner, dtype=float).copy()
    hem_R[0] += float(hem_x_offset_right_m)
    hem_L_rf = np.asarray(right_corner, dtype=float).copy()
    hem_L_rf[0] += float(hem_x_offset_left_m)
    hem_L = right_to_left(hem_L_rf, T_r, T_l)
    return hem_R, hem_L


# OBB axes and jaw yaw


def obb_major_axis_2d(mask: np.ndarray) -> np.ndarray:
    """
    Return the 2D (pixel-space) unit vector along the mask OBB major axis.

    Uses cv2.minAreaRect; the major axis is the longer side. Requires
    OpenCV; raises if it is unavailable.
    """
    if cv2 is None:
        raise RuntimeError("OpenCV not installed; obb_major_axis_2d unavailable")

    pts = cv2.findNonZero((np.asarray(mask) > 0).astype(np.uint8))
    if pts is None:
        raise ValueError("obb_major_axis_2d: empty mask")
    (_cx, _cy), (w, h), ang = cv2.minAreaRect(pts)
    a = np.radians(ang)
    if w >= h:
        return np.array([np.cos(a), np.sin(a)])
    return np.array([-np.sin(a), np.cos(a)])


def obb_center_2d(mask: np.ndarray) -> np.ndarray:
    """
    Return the (cx, cy) pixel-space center of the mask OBB.
    """
    if cv2 is None:
        raise RuntimeError("OpenCV not installed; obb_center_2d unavailable")

    pts = cv2.findNonZero((np.asarray(mask) > 0).astype(np.uint8))
    if pts is None:
        raise ValueError("obb_center_2d: empty mask")
    (cx, cy), (_w, _h), _ang = cv2.minAreaRect(pts)
    return np.array([cx, cy])


def yaw_from_major_axis(
    mask: np.ndarray,
    depth: np.ndarray,
    K: np.ndarray,
    T_to_base: np.ndarray,
    span_px: float = 60.0,
) -> float:
    """
    Base-frame yaw (rad) of the garment OBB major axis.

    Back-projects two points either side of the OBB center along the major
    axis and takes atan2 of their base-frame XY delta. This is the hem yaw
    the rollout feeds to the right arm (the left arm adds the 180 deg
    camera-resolution flip on top).
    """
    major = obb_major_axis_2d(mask)
    center = obb_center_2d(mask)
    e0 = backproject_pixel(*(center - major * span_px), depth, K, T_to_base)
    e1 = backproject_pixel(*(center + major * span_px), depth, K, T_to_base)
    return float(np.arctan2(e1[1] - e0[1], e1[0] - e0[0]))


def _backproject_pixel_detect(
    u: float, v: float, depth: np.ndarray, K: np.ndarray, T_to_base: np.ndarray
):
    """
    Single-pixel back-projection matching scripts/detect._backproject_pixel.

    Exact port of the detect.py pinhole lift (single-pixel depth, a +/-5px
    median fallback when the pixel reads <0.01, no median smoothing on a valid
    pixel). Kept separate from the +/-2px-windowed :func:`backproject_pixel` so
    the ported sleeve geometry reproduces detect.py's output. Returns a
    base-frame (3,) point or None when no valid depth is found.
    """
    fx = K[0, 0]
    fy = K[1, 1]
    cx_k = K[0, 2]
    cy_k = K[1, 2]
    h, w = depth.shape[:2]
    ui, vi = int(round(u)), int(round(v))
    ui = max(0, min(ui, w - 1))
    vi = max(0, min(vi, h - 1))
    d_val = float(depth[vi, ui])
    if d_val < 0.01:
        patch = depth[max(0, vi - 5) : vi + 5, max(0, ui - 5) : ui + 5]
        valid = patch[(patch > 0.01) & (patch < 10)]
        if len(valid) > 0:
            d_val = float(np.median(valid))
        else:
            return None
    x_cam = (u - cx_k) * d_val / fx
    y_cam = (v - cy_k) * d_val / fy
    z_cam = d_val
    R = T_to_base[:3, :3]
    t = T_to_base[:3, 3]
    return R @ np.array([x_cam, y_cam, z_cam]) + t


def compute_sleeve_outer_edge(
    mask: np.ndarray,
    ref_px,
    depth: np.ndarray,
    K: np.ndarray,
    T_to_base: np.ndarray,
    inset_m: float = 0.030,
):
    """
    Outer-edge (tip) grab point of a sleeve mask, in the primary base frame.

    Faithful port of scripts/detect.compute_sleeve_outer_edge: the fold
    pinches the sleeve TIP, not the centroid. Take the band of mask pixels
    farthest (97th-percentile distance band) from ``ref_px`` (the shirt-body
    centroid in pixels), then pull the resulting tip back toward the body by
    ``inset_m`` so the pinch lands on cloth rather than the very edge.

    Returns ``(world_point, (u, v))`` in the primary frame, or
    ``(None, None)`` when the mask is too small / has no valid depth.
    """
    ys, xs = np.where(np.asarray(mask) > 0)
    if len(xs) < 10:
        return None, None
    px = np.stack([xs.astype(np.float64), ys.astype(np.float64)], axis=1)
    ref = np.asarray(ref_px, dtype=np.float64)
    dist = np.linalg.norm(px - ref, axis=1)
    d_hi = float(np.percentile(dist, 97.0))
    band = px[dist >= d_hi]
    tip = band.mean(axis=0)
    fx = K[0, 0]
    d_val = float(depth[int(tip[1]), int(tip[0])]) if depth is not None else 0.5
    if d_val < 0.01:
        valid = depth[np.asarray(mask) > 0]
        valid = valid[(valid > 0.01) & (valid < 10)]
        d_val = float(np.median(valid)) if len(valid) > 0 else 0.5
    direction = ref - tip
    n = float(np.linalg.norm(direction))
    if n > 1e-6:
        tip = tip + direction / n * (inset_m * fx / d_val)
    pt = _backproject_pixel_detect(tip[0], tip[1], depth, K, T_to_base)
    return pt, (float(tip[0]), float(tip[1]))


def compute_jaw_yaw_minor(
    mask: np.ndarray,
    depth: np.ndarray,
    K: np.ndarray,
    T_r: np.ndarray,
    T_l: np.ndarray,
    T_primary: np.ndarray,
):
    """
    OBB MINOR-axis jaw yaw in BOTH robot frames (radians).

    Faithful port of scripts/detect.compute_jaw_yaw_minor: fit an OBB with
    cv2.minAreaRect, take the MINOR (short) axis in image space, backproject
    two points +/-15 px along it through depth to get a world direction in the
    primary frame, then transform that direction into the LEFT frame via the
    cross-arm calibration so each arm gets a yaw in its OWN base frame.

    Returns ``(yaw_r, yaw_l)`` or ``(None, None)`` on failure. ``T_primary``
    is the frame the detection's positions are expressed in (right on this
    rig); ``T_r`` / ``T_l`` are the per-arm T_zed_to_base used to map the
    direction into the left frame.
    """
    if cv2 is None:
        raise RuntimeError("OpenCV not installed; jaw-yaw helper unavailable")

    m_pts = cv2.findNonZero((np.asarray(mask) > 0).astype(np.uint8))
    if m_pts is None:
        return None, None
    (mcx, mcy), (mw, mh), mang = cv2.minAreaRect(m_pts)
    ma = np.radians(mang)
    dir_w = np.array([np.cos(ma), np.sin(ma)])
    dir_h = np.array([-np.sin(ma), np.cos(ma)])
    # minAreaRect width spans the dir_w edge; the MINOR axis is the direction
    # of the SHORTER side.
    minor = dir_h if mw >= mh else dir_w
    p0 = _backproject_pixel_detect(
        mcx - minor[0] * 15, mcy - minor[1] * 15, depth, K, T_primary
    )
    p1 = _backproject_pixel_detect(
        mcx + minor[0] * 15, mcy + minor[1] * 15, depth, K, T_primary
    )
    if p0 is None or p1 is None:
        return None, None
    p0 = np.asarray(p0, dtype=float)
    p1 = np.asarray(p1, dtype=float)
    yaw_r = float(np.arctan2(p1[1] - p0[1], p1[0] - p0[0]))
    q0 = right_to_left(p0, T_r, T_l)
    q1 = right_to_left(p1, T_r, T_l)
    yaw_l = float(np.arctan2(q1[1] - q0[1], q1[0] - q0[0]))
    return yaw_r, yaw_l


def sleeve_tip_from_cloud(
    cloud: np.ndarray, body_center_xy: np.ndarray, tip_band_m: float = 0.03
) -> np.ndarray:
    """
    Outer-edge (tip) point of a sleeve cloud, farthest from the body.

    Selects the band of cloud points farthest (in XY) from the garment body
    center and returns their mean. This is the autonomous equivalent of
    detect.py's outer_edge field: the grasp point at the sleeve tip rather
    than the mid-sleeve centroid.
    """
    cloud = np.asarray(cloud, dtype=float)
    if cloud.shape[0] == 0:
        raise ValueError("sleeve_tip_from_cloud: empty cloud")
    body_xy = np.asarray(body_center_xy, dtype=float)[:2]
    d = np.linalg.norm(cloud[:, :2] - body_xy, axis=1)
    far = cloud[d > d.max() - tip_band_m]
    return far.mean(axis=0)


def drape_to_center_y(
    grab_y: float, body_center_y: float, fraction: float = 0.80
) -> float:
    """
    Sleeve drape target y, equal to ``fraction`` of the way from grab to
    body center.

    The exact midpoint sits at the workspace edge for a wide shirt; 80% still
    folds the sleeve over the body. Mirrors fold_tshirt_v3 center_y calc.
    """
    return float(grab_y + fraction * (body_center_y - grab_y))

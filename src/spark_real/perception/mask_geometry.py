"""
Free geometry helpers for SPARK mask-to-3D perception.

Pure numpy/cv2 functions extracted from spark_perception. They turn SAM3
masks plus depth and camera extrinsics into world-frame positions and
oriented bounding boxes. No model state, no torch.

Contents:
    _sigma_n                 cardinality penalty (disabled, returns 1.0)
    _cloud_median_world      backproject a mask to its per-axis world median
    _top_layer_mask          keep only the top-surface depth layer of a mask
    _world_xy_pca_obb        world-XY footprint OBB via backprojection + PCA
    _pca_obb                 image-frame OBB via PCA
    heavy_end_sign           which end of a mask's axis is the fat end
    best_fit_rotation        image rotation registering one mask onto another
    _ray_intersect_table     intersect a pixel ray with a world z-plane
    _compute_orientation     world-frame orientation via ray-plane projection
    mask_color_stats         median HSV + hue concentration over a mask
    mask_height_profile      world-Z top/bottom of a mask's depth cloud
    resolve_slot_direction   world angle a placed utensil lies along
"""

import logging
import os

import cv2
import numpy as np
from spark_real.utils.det_fields import det_field

logger = logging.getLogger(__name__)

# A near-square world-XY OBB (aspect_ratio below this) has an untrustworthy
# major axis: PCA can swap the major/minor eigenvectors when the two spreads
# are close, so the reported world OBB angle may actually be the SHORT axis.
# For the 3-D cutlery tray this fired (world PCA reported ~103 deg / AR~1.3-1.8,
# the tray's short axis mislabelled as major). Below this threshold do NOT
# trust the world OBB major; derive slot direction from a more reliable source.
TRAY_OBB_AR_TRUST = 1.5

# Fraction of a uniform extent covered by the 5th-95th percentile band.
_TRIM_COVERAGE = 0.90

# Aspect ratio at which the PCA major axis counts as fully resolved. Below it
# the two eigenvalues are close and the reported angle is partly arbitrary.
_AXIS_CONF_AR_REF = 2.0

# mask_height_profile: world-Z percentiles standing in for a mask's top and
# bottom surface. Asymmetric on purpose -- see that function's docstring.
_RIM_PCT = 95.0
_INTERIOR_PCT = 20.0
# Below this many valid depth pixels a height profile is noise, not a surface.
_HEIGHT_PROFILE_MIN_PIXELS = 50


def _sigma_n(n_candidates: int) -> float:
    """
    Cardinality penalty disabled; returns 1.0 unconditionally.
    """
    return 1.0


def _cloud_median_world(
    mask,
    depth,
    cam_mat,
    cam_pos,
    fx=None,
    fy=None,
    cx_k=None,
    cy_k=None,
    fovy_deg=None,
    use_opencv=False,
    min_depth=0.01,
    max_depth=10.0,
):
    """
    Backproject every mask pixel to 3D, return per-axis median.
    """
    h, w = mask.shape
    ys, xs = np.where(mask > 0)
    if len(xs) == 0:
        return None, 0.0
    depths = depth[ys, xs].astype(np.float64)
    valid = (depths > min_depth) & (depths < max_depth)
    if valid.sum() < 3:
        return None, 0.0
    xs = xs[valid].astype(np.float64)
    ys = ys[valid].astype(np.float64)
    depths = depths[valid]

    if use_opencv:
        if fx is None or fy is None or cx_k is None or cy_k is None:
            return None, 0.0
        x_cam = (xs - cx_k) * depths / fx
        y_cam = (ys - cy_k) * depths / fy
        z_cam = depths
    else:
        if fovy_deg is None:
            return None, 0.0
        f = h / (2 * np.tan(np.deg2rad(fovy_deg) / 2))
        x_cam = (xs - w / 2) * depths / f
        y_cam = -(ys - h / 2) * depths / f
        z_cam = -depths

    pts_cam = np.column_stack([x_cam, y_cam, z_cam])
    pts_world = (cam_mat @ pts_cam.T).T + cam_pos

    # Z: use 95th percentile when bimodal (table + object surface)
    z_all = pts_world[:, 2]
    z_range = float(np.percentile(z_all, 95) - np.percentile(z_all, 5))
    if z_range > 0.04:
        top_z = float(np.percentile(z_all, 95))
        top_mask = z_all > (top_z - 0.03)
        if top_mask.sum() >= 3:
            centroid_xy = np.median(pts_world[top_mask, :2], axis=0)
            centroid = np.array([centroid_xy[0], centroid_xy[1], top_z])
        else:
            centroid = np.median(pts_world, axis=0)
            centroid[2] = top_z
    else:
        centroid = np.median(pts_world, axis=0)
    d_val = float(np.percentile(depths, 25))
    return centroid, d_val


def _top_layer_mask(
    mask: np.ndarray, depth: np.ndarray, layer_m: float = 0.02, min_pixels: int = 30
) -> np.ndarray:
    """
    Filter mask to its top-surface layer for OBB computation.
    """
    if mask is None or depth is None:
        return mask
    ys, xs = np.where(mask > 0)
    if len(xs) < min_pixels:
        return mask
    depths = depth[ys, xs]
    valid = depths > 0.01
    if int(valid.sum()) < min_pixels:
        return mask
    valid_depths = depths[valid]
    top_d = float(np.percentile(valid_depths, 10))
    layer_keep = valid & (depths <= top_d + layer_m)
    if int(layer_keep.sum()) < min_pixels:
        return mask
    filtered = np.zeros_like(mask)
    filtered[ys[layer_keep], xs[layer_keep]] = mask[ys[layer_keep], xs[layer_keep]]
    return filtered


def _trimmed_span(proj: np.ndarray) -> float:
    """5th-95th percentile span, rescaled to a full-extent equivalent.

    max()-min() over backprojected depth is a RANGE statistic: two stray
    pixels set it (a plushie read aspect ratio 3.14 that way). The 5-95 band
    of a uniform footprint covers 90% of its extent, so divide by 0.9 to keep
    obb_minor_m comparable with the untrimmed values the grasp width prior
    and container extents are tuned against.
    """
    lo, hi = np.percentile(proj, [5.0, 95.0])
    return float((hi - lo) / _TRIM_COVERAGE)


def _world_xy_pca_obb(
    mask,
    depth,
    cam_pos,
    cam_mat,
    fx=None,
    fy=None,
    cx_k=None,
    cy_k=None,
    fovy_deg=None,
    use_opencv=False,
    min_pixels=30,
    return_confidence=False,
):
    """
    World-XY footprint OBB via backprojection + PCA.

    Returns (orient_world_rad, ar, mj_len_m, mn_len_m), or with
    ``return_confidence`` a 5th element obb_confidence in [0,1]. All None on
    failure.
    """
    _fail = (None, None, None, None, None) if return_confidence else (None,) * 4
    if mask is None or depth is None or cam_mat is None or cam_pos is None:
        return _fail
    ys, xs = np.where(mask > 0)
    if len(xs) < min_pixels:
        return _fail
    depths = depth[ys, xs].astype(np.float64)
    valid = depths > 0.01
    if int(valid.sum()) < min_pixels:
        return _fail
    xs_v, ys_v, d_v = (
        xs[valid].astype(np.float64),
        ys[valid].astype(np.float64),
        depths[valid],
    )
    h, w = mask.shape
    if use_opencv:
        if fx is None or fy is None or cx_k is None or cy_k is None:
            return _fail
        x_cam = (xs_v - cx_k) * d_v / fx
        y_cam = (ys_v - cy_k) * d_v / fy
        z_cam = d_v
    else:
        if fovy_deg is None:
            return _fail
        f = h / (2 * np.tan(np.deg2rad(fovy_deg) / 2))
        x_cam = (xs_v - w / 2) * d_v / f
        y_cam = -(ys_v - h / 2) * d_v / f
        z_cam = -d_v
    pts_cam = np.column_stack([x_cam, y_cam, z_cam])
    pts_world = (cam_mat @ pts_cam.T).T + cam_pos
    # Footprint from the object's TOP surface layer, not the full mask. For a tall
    # object the camera sees the top AND a side face; the side backprojects into
    # the XY footprint and stretches one axis, inflating the AR (a cube read ~1.4
    # full-mask vs ~1.0 from its top layer -> spurious yaw on symmetric cubes).
    # Taking the top band gives the true top-down footprint: cubes ~1.0, 2:1
    # blocks ~2.0. Tunable via SPARK_OBB_TOP_BAND_M; falls back to the full mask
    # if the top band is too sparse.
    z_all = pts_world[:, 2]
    top_band = float(os.environ.get("SPARK_OBB_TOP_BAND_M", "0.008"))
    full_xy = pts_world[:, :2]
    _sel = z_all >= (float(z_all.max()) - top_band)
    band_ok = int(_sel.sum()) >= min_pixels
    if band_ok:
        band_xy = pts_world[_sel, :2]
        # A band whose XY spread EXCEEDS the full mask's is not a top surface:
        # it is depth noise smeared across the footprint. Reject it.
        band_spread = float(np.hypot(*band_xy.std(axis=0)))
        full_spread = float(np.hypot(*full_xy.std(axis=0)))
        if full_spread > 1e-9 and band_spread > 1.5 * full_spread:
            logger.debug(
                "[obb] top band spread %.3f > 1.5x full %.3f; using full mask",
                band_spread, full_spread,
            )
            band_ok = False
    xy = pts_world[_sel, :2] if band_ok else full_xy
    mean_xy = xy.mean(axis=0)
    cov = np.cov((xy - mean_xy).T)
    if cov.ndim != 2 or cov.shape != (2, 2):
        return _fail
    eigvals, eigvecs = np.linalg.eigh(cov)
    mj, mn = eigvecs[:, 1], eigvecs[:, 0]
    proj_mj = (xy - mean_xy) @ mj
    proj_mn = (xy - mean_xy) @ mn
    mj_len_m = _trimmed_span(proj_mj)
    mn_len_m = _trimmed_span(proj_mn)
    if mj_len_m <= 0 or mn_len_m <= 0:
        return _fail
    # Trimming can reorder the two axes on a near-square footprint; keep
    # "major" meaning the longer one (same guard as resolve_slot_direction).
    if mj_len_m < mn_len_m:
        mj, mn = mn, mj
        mj_len_m, mn_len_m = mn_len_m, mj_len_m
    orient = float(np.arctan2(mj[1], mj[0]))
    if not return_confidence:
        return orient, mj_len_m / mn_len_m, mj_len_m, mn_len_m

    # Confidence has two independent factors, both needed:
    #   band quality   - a flat, well-populated top band means the footprint
    #                    came from a real top surface, not smeared depth;
    #   axis definition - PCA on a near-circular footprint returns an ARBITRARY
    #                    major axis (eigenvalues tie), so a round dome must
    #                    score low however clean its depth is.
    band_z = pts_world[_sel, 2] if band_ok else z_all
    z_std = float(np.std(band_z))
    n_band = int(_sel.sum()) if band_ok else int(len(z_all))
    band_q = float(
        np.clip(1.0 - z_std / max(top_band, 1e-6), 0.0, 1.0)
        * min(1.0, n_band / 200.0)
    )
    ar = mj_len_m / mn_len_m
    axis_q = float(np.clip((ar - 1.0) / (_AXIS_CONF_AR_REF - 1.0), 0.0, 1.0))
    return orient, ar, mj_len_m, mn_len_m, band_q * axis_q


def _pca_obb(mask: np.ndarray):
    """
    Return (img_angle_rad, aspect_ratio, short_px, long_px) via PCA.
    """
    ys, xs = np.where(mask > 0)
    if len(xs) < 5:
        return 0.0, 1.0, 0.0, 0.0
    pts = np.column_stack([xs.astype(np.float64), ys.astype(np.float64)])
    mean = pts.mean(axis=0)
    cov = np.cov(pts - mean, rowvar=False)
    if cov.ndim == 0 or cov.shape != (2, 2):
        return 0.0, 1.0, 0.0, 0.0
    eigvals, eigvecs = np.linalg.eigh(cov)
    major_dir, minor_dir = eigvecs[:, 1], eigvecs[:, 0]
    centered = pts - mean
    long_px = float((centered @ major_dir).max() - (centered @ major_dir).min())
    short_px = float((centered @ minor_dir).max() - (centered @ minor_dir).min())
    if short_px <= 0:
        return float(np.arctan2(major_dir[1], major_dir[0])), 1.0, 0.0, long_px
    return (
        float(np.arctan2(major_dir[1], major_dir[0])),
        long_px / short_px,
        short_px,
        long_px,
    )


# heavy_end_sign: below this mass imbalance between the two axis halves the
# shape is called symmetric and the sign abstains (0.0). Sized against noise:
# a uniform rectangle of N pixels has imbalance sd ~ 1/sqrt(N) (~0.014 at
# N=5000), while a synthetic screwdriver silhouette (handle 2.5x the shaft
# width over ~40% of the length) measures ~0.3. See
# tests/test_mask_direction.py.
_HEAVY_END_IMBALANCE_GATE = 0.08
_HEAVY_END_MIN_PIXELS = 50


def _largest_component(mask8: np.ndarray) -> np.ndarray:
    """The largest 8-connected component of a binary mask.

    SAM3 masks fragment, and satellite blobs drag the extent midpoint of a
    DIRECTED measurement toward a confidently wrong sign. The body of the
    object is the largest component; measure that.
    """
    n, lab = cv2.connectedComponents(mask8, connectivity=8)
    if n <= 2:  # background + at most one component
        return mask8
    counts = np.bincount(lab.ravel())
    counts[0] = 0  # background
    return (lab == int(np.argmax(counts))).astype(np.uint8)


def heavy_end_sign(mask: np.ndarray, axis_rad: float) -> float:
    """Which END of `axis_rad` the mask's fat half lies toward: +1.0 / -1.0.

    A PCA axis is a LINE -- identical under a 180 deg flip -- so it cannot say
    which end of a screwdriver is the tip. This breaks that tie with the only
    directed signal a silhouette has: one end being fatter than the other.
    Pixels are projected onto the axis direction and split at the midpoint of
    the (percentile-trimmed) extent; whichever half holds more mask pixels is
    the heavy end. +1.0 means it lies toward +(cos axis, sin axis) in image
    coords (x right, y down), -1.0 the opposite end.

    Returns 0.0 -- "too symmetric to call" -- below _HEAVY_END_IMBALANCE_GATE,
    and every consumer must treat 0.0 as an abstention, never a direction. A
    bare handle mask is symmetric end-for-end, so the direction must be
    measured on the WHOLE tool (see _record_held_axis's parent-label upgrade).
    """
    m8 = _largest_component((np.asarray(mask) > 0).astype(np.uint8))
    ys, xs = np.where(m8 > 0)
    if len(xs) < _HEAVY_END_MIN_PIXELS:
        return 0.0
    # The caller's axis usually comes from _pca_obb over the FULL mask. When
    # fragments dragged that axis away from the body's own, a sign measured
    # along it describes the fragments, not the object. If the two axes
    # disagree by more than 45 deg the premise is broken: abstain.
    ang_cc, _, _, _ = _pca_obb(m8)
    d_ax = abs(ang_cc - float(axis_rad)) % np.pi
    if min(d_ax, np.pi - d_ax) > np.pi / 4:
        return 0.0
    proj = xs * float(np.cos(axis_rad)) + ys * float(np.sin(axis_rad))
    # Trim the extent, not the mass: a few speck pixels must not drag the
    # midpoint, but every real pixel still votes on which half is heavier.
    lo, hi = np.percentile(proj, [2.0, 98.0])
    if hi - lo < 1e-6:
        return 0.0
    mid = 0.5 * (lo + hi)
    n_pos = int(np.count_nonzero(proj > mid))
    n_neg = int(np.count_nonzero(proj < mid))
    total = n_pos + n_neg
    if total == 0:
        return 0.0
    imbalance = (n_pos - n_neg) / float(total)
    if abs(imbalance) < _HEAVY_END_IMBALANCE_GATE:
        return 0.0
    return 1.0 if imbalance > 0 else -1.0


# best_fit_rotation resamples both masks onto this square canvas. 192 px keeps
# a full 360-sweep of warps under ~10 ms while leaving a ~2 px quantization,
# far under the 90-deg-class margins the consumers gate on.
_FIT_CANVAS_PX = 192
_FIT_STEP_DEG = 3.0
_FIT_MIN_PIXELS = 50


def best_fit_rotation(held_mask: np.ndarray, target_mask: np.ndarray):
    """Image rotation that best registers `held_mask` onto `target_mask`.

    Returns (angle_rad, best_iou, margin):
      angle_rad  signed image-frame rotation in (-pi, pi]. Rotating the held
                 silhouette by +angle (the same convention as _pca_obb angles:
                 atan2 in image coords, x right, y down) best overlays it on
                 the target, i.e. angle ~= target_axis - held_axis when the
                 shapes are elongated.
      best_iou   IoU at that rotation, both masks centroid-centered and drawn
                 at a SHARED scale -- relative size is real signal (a cutout
                 slightly larger than its tool still overlaps well; a bowl
                 twice its size does not).
      margin     best_iou minus the best IoU among rotations >= 90 deg away.
                 This is the DIRECTED part of the answer: for a directional
                 shape in a directional recess the 180-off candidate scores
                 visibly worse and the margin is large; for a round or
                 twofold-symmetric pair every rotation ties and the margin
                 collapses toward 0 -- which is the honest "shape does not
                 decide" answer the JIGSAW_MIN_MARGIN gate turns into an
                 abstention (executor_motion, gate 0.05).

    (0.0, 0.0, 0.0) when either mask is degenerate (< _FIT_MIN_PIXELS).
    """
    # Largest component only, for the same reason as heavy_end_sign: a
    # satellite blob shifts the centroid and radius, and the registration then
    # scores rotations of the wrong center.
    h8 = _largest_component((np.asarray(held_mask) > 0).astype(np.uint8))
    t8 = _largest_component((np.asarray(target_mask) > 0).astype(np.uint8))
    if int(h8.sum()) < _FIT_MIN_PIXELS or int(t8.sum()) < _FIT_MIN_PIXELS:
        return 0.0, 0.0, 0.0

    c = _FIT_CANVAS_PX
    ctr = (c - 1) / 2.0

    def _centroid_and_radius(m):
        ys, xs = np.where(m > 0)
        cx, cy = float(xs.mean()), float(ys.mean())
        r = float(np.sqrt(((xs - cx) ** 2 + (ys - cy) ** 2).max()))
        return cx, cy, max(r, 1.0)

    hx, hy, hr = _centroid_and_radius(h8)
    tx, ty, tr = _centroid_and_radius(t8)
    # ONE scale for both, sized so the larger just fits under every rotation.
    s = (c / 2.0 - 2.0) / max(hr, tr)

    def _to_canvas(m, cx, cy):
        M = np.array([[s, 0.0, ctr - s * cx], [0.0, s, ctr - s * cy]])
        return cv2.warpAffine(m, M, (c, c), flags=cv2.INTER_NEAREST)

    held_c = _to_canvas(h8, hx, hy)
    tgt_c = _to_canvas(t8, tx, ty).astype(bool)
    tgt_area = int(tgt_c.sum())

    angles = np.deg2rad(np.arange(0.0, 360.0, _FIT_STEP_DEG))
    ious = np.zeros(len(angles))
    for i, a in enumerate(angles):
        ca, sa = float(np.cos(a)), float(np.sin(a))
        # Forward map [[c,-s],[s,c]] about the canvas center: a pixel at image
        # angle b moves to b+a, matching _pca_obb's atan2 convention, so the
        # returned rotation composes directly with axis angles.
        M = np.array(
            [[ca, -sa, ctr - ca * ctr + sa * ctr],
             [sa, ca, ctr - sa * ctr - ca * ctr]]
        )
        rot = cv2.warpAffine(held_c, M, (c, c), flags=cv2.INTER_NEAREST).astype(bool)
        inter = int(np.count_nonzero(rot & tgt_c))
        union = int(rot.sum()) + tgt_area - inter
        ious[i] = inter / union if union > 0 else 0.0

    best = int(np.argmax(ious))
    best_iou = float(ious[best])
    # Circular distance to the winner; rivals live >= 90 deg away.
    d = np.abs(np.rad2deg(angles) - np.rad2deg(angles[best]))
    d = np.minimum(d, 360.0 - d)
    far = d >= 90.0
    rival = float(ious[far].max()) if far.any() else 0.0
    ang = float((angles[best] + np.pi) % (2 * np.pi) - np.pi)
    return ang, best_iou, best_iou - rival


def _ray_intersect_table(u, v, f, cx_img, cy_img, cam_mat, cam_pos, z_plane, use_opencv=False):
    """
    Intersect camera ray through pixel (u,v) with world plane z=z_plane.
    """
    if use_opencv:
        dir_cam = np.array([(u - cx_img) / f, (v - cy_img) / f, 1.0])
    else:
        dir_cam = np.array([(u - cx_img) / f, -(v - cy_img) / f, -1.0])
    dir_world = cam_mat @ dir_cam
    denom = dir_world[2]
    if abs(denom) < 1e-9:
        return None
    t = (z_plane - cam_pos[2]) / denom
    if t <= 0:
        return None
    return cam_pos + t * dir_world


def _compute_orientation(
    mask,
    depth,
    cam_mat,
    cam_pos,
    cam_fovy,
    cx,
    cy,
    h,
    w,
    img_angle_rad,
    long_px,
    metric_depth,
    use_opencv=False,
    fx_b=None,
    fy_b=None,
    cx_b=None,
    cy_b=None,
):
    """
    Compute world-frame orientation via ray-plane intersection.
    """
    if long_px <= 0 or metric_depth <= 0.01 or cam_mat is None or cam_pos is None:
        return 0.0
    du = np.cos(img_angle_rad)
    dv = np.sin(img_angle_rad)
    if fx_b is None:
        fx_b = h / (2 * np.tan(np.deg2rad(cam_fovy) / 2)) if cam_fovy else float(h)
    if fy_b is None:
        fy_b = fx_b
    if cx_b is None:
        cx_b = w / 2.0
    if cy_b is None:
        cy_b = h / 2.0
    half_step = float(long_px) * 0.4

    # Get table_z from centroid backprojection
    if use_opencv:
        _cx_c = (cx - cx_b) * metric_depth / fx_b
        _cy_c = (cy - cy_b) * metric_depth / fy_b
        _cz_c = metric_depth
    else:
        _cx_c = (cx - cx_b) * metric_depth / fx_b
        _cy_c = -(cy - cy_b) * metric_depth / fy_b
        _cz_c = -metric_depth
    centroid_world = cam_mat @ np.array([_cx_c, _cy_c, _cz_c]) + cam_pos
    table_z = float(centroid_world[2])

    p1 = _ray_intersect_table(
        cx + half_step * du,
        cy + half_step * dv,
        fx_b,
        cx_b,
        cy_b,
        cam_mat,
        cam_pos,
        table_z,
        use_opencv,
    )
    p2 = _ray_intersect_table(
        cx - half_step * du,
        cy - half_step * dv,
        fx_b,
        cx_b,
        cy_b,
        cam_mat,
        cam_pos,
        table_z,
        use_opencv,
    )
    if p1 is not None and p2 is not None:
        dir_xy = p1[:2] - p2[:2]
        return float(np.arctan2(dir_xy[1], dir_xy[0]))
    return 0.0


def _erode_mask(mask, iterations):
    """Binary erosion by `iterations` steps of a 3x3 cross.

    Shrinking the mask before sampling colour matters: a SAM3 mask tracks
    the object silhouette, so its outermost ring of pixels is antialiased
    against the background and drags the median hue toward the table.
    """
    out = np.asarray(mask, dtype=bool)
    for _ in range(max(0, int(iterations))):
        shrunk = out.copy()
        shrunk[1:, :] &= out[:-1, :]
        shrunk[:-1, :] &= out[1:, :]
        shrunk[:, 1:] &= out[:, :-1]
        shrunk[:, :-1] &= out[:, 1:]
        shrunk[0, :] = False
        shrunk[-1, :] = False
        shrunk[:, 0] = False
        shrunk[:, -1] = False
        if not shrunk.any():
            return out  # eroding further would erase a thin object entirely
        out = shrunk
    return out


def mask_color_stats(rgb, mask, erode_px=3, max_samples=20000):
    """Median HSV of the pixels under `mask`, plus a hue-concentration score.

    Returns ``(hue_deg, saturation, value, hue_conf)`` or ``None`` when the
    mask is empty. Hue is degrees in [0, 360); saturation and value are on
    the 0-255 scale OpenCV uses, so thresholds are comparable with the rest
    of the codebase. ``hue_conf`` is the resultant length of the
    saturation-weighted circular mean, in [0, 1]: near 1 for a uniformly
    coloured object, near 0 for a grey or multicoloured one.

    Hue is averaged circularly rather than with a plain median because hue
    wraps -- red pixels straddling 0/360 would otherwise average to cyan.
    Weighting by saturation keeps near-grey pixels, whose hue is numerically
    unstable, from steering the result.

    Used only by the optional ``disambiguate.hsv_cluster`` path in the task
    prompt registry; the shipped task set resolves on counts and geometry
    alone, so this stays off by default.
    """
    m = np.asarray(mask, dtype=bool)
    if m.ndim > 2:
        m = m.squeeze()
    if not m.any():
        return None
    arr = np.asarray(rgb)
    if arr.ndim != 3 or arr.shape[2] < 3 or arr.shape[:2] != m.shape:
        return None

    eroded = _erode_mask(m, erode_px)
    px = arr[eroded][:, :3].astype(np.float64)
    if px.size == 0:
        return None
    if px.shape[0] > max_samples:
        # Deterministic stride, not a random sample: the same mask must give
        # the same colour on every run or instance ordering stops being stable.
        px = px[:: max(1, px.shape[0] // max_samples)]

    px /= 255.0
    mx = px.max(axis=1)
    mn = px.min(axis=1)
    delta = mx - mn

    sat = np.where(mx > 0, delta / np.maximum(mx, 1e-12), 0.0)

    r, g, b = px[:, 0], px[:, 1], px[:, 2]
    hue = np.zeros_like(mx)
    safe = delta > 1e-12
    idx = safe & (mx == r)
    hue[idx] = ((g[idx] - b[idx]) / delta[idx]) % 6.0
    idx = safe & (mx == g) & (mx != r)
    hue[idx] = (b[idx] - r[idx]) / delta[idx] + 2.0
    idx = safe & (mx == b) & (mx != r) & (mx != g)
    hue[idx] = (r[idx] - g[idx]) / delta[idx] + 4.0
    hue *= 60.0

    rad = np.deg2rad(hue)
    wsum = float(sat.sum())
    if wsum > 1e-9:
        cx = float((sat * np.cos(rad)).sum()) / wsum
        cy = float((sat * np.sin(rad)).sum()) / wsum
        hue_deg = float(np.rad2deg(np.arctan2(cy, cx)) % 360.0)
        hue_conf = float(min(1.0, np.hypot(cx, cy)))
    else:
        hue_deg = 0.0
        hue_conf = 0.0

    return (
        hue_deg,
        float(np.median(sat) * 255.0),
        float(np.median(mx) * 255.0),
        hue_conf,
    )


def mask_height_profile(
    mask,
    depth,
    cam_pos,
    cam_mat,
    fx=None,
    fy=None,
    cx_k=None,
    cy_k=None,
    fovy_deg=None,
    use_opencv=False,
    min_pixels=_HEIGHT_PROFILE_MIN_PIXELS,
    min_depth=0.01,
    max_depth=10.0,
):
    """World-Z extremes of a mask's depth cloud: the TOP and BOTTOM surfaces.

    Returns ``{"rim_z", "interior_z", "n_valid"}`` or ``None`` when the mask
    carries too little valid depth to say anything. Nothing is invented: no
    depth means no answer, and the caller falls back.

    For a CONTAINER these two numbers are its rim and its interior floor. A
    bowl's mask is bimodal in world Z -- the rim ring projects high, the
    interior floor low -- which is the same bimodality ``_cloud_median_world``
    already exploits when it reports the 95th percentile as the object top.
    For any other object they are simply its top and its bottom, which is what
    the release-height computation needs for the object being CARRIED (see
    ``control.release_height``).

    Percentiles, not min/max: a handful of dropout or edge-bleed pixels sets
    the extremes and would report a rim that does not exist. The interior uses
    a deliberately HIGH percentile (_INTERIOR_PCT, 20th) rather than a
    symmetric one, because an interior read too LOW lowers the release, and
    the safe error here is to read it high.
    """
    if mask is None or depth is None or cam_mat is None or cam_pos is None:
        return None
    ys, xs = np.where(mask > 0)
    if len(xs) < min_pixels:
        return None
    depths = depth[ys, xs].astype(np.float64)
    valid = (depths > min_depth) & (depths < max_depth)
    n_valid = int(valid.sum())
    if n_valid < min_pixels:
        return None
    xs_v = xs[valid].astype(np.float64)
    ys_v = ys[valid].astype(np.float64)
    d_v = depths[valid]

    h, w = mask.shape
    if use_opencv:
        if fx is None or fy is None or cx_k is None or cy_k is None:
            return None
        x_cam = (xs_v - cx_k) * d_v / fx
        y_cam = (ys_v - cy_k) * d_v / fy
        z_cam = d_v
    else:
        if fovy_deg is None:
            return None
        f = h / (2 * np.tan(np.deg2rad(fovy_deg) / 2))
        x_cam = (xs_v - w / 2) * d_v / f
        y_cam = -(ys_v - h / 2) * d_v / f
        z_cam = -d_v
    pts_world = (cam_mat @ np.column_stack([x_cam, y_cam, z_cam]).T).T + cam_pos
    wz = pts_world[:, 2]
    rim_z, interior_z = np.percentile(wz, [_RIM_PCT, _INTERIOR_PCT])
    return {
        "rim_z": float(rim_z),
        "interior_z": float(min(interior_z, rim_z)),
        "n_valid": n_valid,
    }


def _pca_major_angle(points_xy):
    """World-frame angle (rad) of the dominant-spread axis of 2D points.

    Used on detected slot CENTROIDS: parallel-compartment trays array their
    slots along the tray's LONG axis, so the centroids' principal axis IS the
    tray long axis. Returns None if fewer than 2 distinct points.
    """
    pts = np.asarray(points_xy, dtype=float)
    if pts.ndim != 2 or pts.shape[0] < 2:
        return None
    mean = pts.mean(axis=0)
    centered = pts - mean
    if float(np.max(np.linalg.norm(centered, axis=1))) < 1e-4:
        return None  # all points coincident
    cov = np.cov(centered, rowvar=False)
    if np.asarray(cov).shape != (2, 2):
        return None
    eigvals, eigvecs = np.linalg.eigh(cov)
    major = eigvecs[:, 1]  # largest eigenvalue = dominant spread
    return float(np.arctan2(major[1], major[0]))


def resolve_slot_direction(det, container_label=""):
    """World angle (rad) a placed utensil must lie ALONG for a slotted container.

    Returns (slot_direction_rad, source_str). Shared by place_in_slot and the
    generic-place path (executor_motion._transport_to) so BOTH align to the
    slots, not the axis-swapped container OBB whose major axis is perpendicular
    to the slots. `det` may be a dict (detection_map) or an ObjectDetection.

    A parallel-compartment tray arrays its slots along the tray's LONG (major)
    axis, and each slot (hence each utensil dropped into it) is elongated
    PERPENDICULAR to that array, along the tray's SHORT (minor) axis. So:
    slot_direction = tray_long_axis + 90 deg (hardware-validated on the real
    tray).

    The tray_long_axis must come from a source that CANNOT axis-swap:
      1. Detected slot CENTROIDS (world frame, slot_detector "normals" mode):
         their principal axis is the true array = tray long axis. Swap-immune
         (real world positions, not the near-square tray OBB). PREFERRED.
      2. The mask's minimum-area rectangle: a rectangle's angle is carried by
         its outline, so no elongation gate is needed.
      3. World OBB major axis, ONLY when aspect_ratio >= TRAY_OBB_AR_TRUST;
         a near-square tray's world OBB may report its SHORT axis as "major".
      4. Dense image-frame mask minAreaRect / PCA fallback (top-down birdview
         image angle maps to world).
      5. No reliable source: keep the tray-OBB perpendicular but WARN.
    """
    # 1. Slot centroids (world frame), real (normals) detections only.
    _slots = det_field(det, "slots", None)
    if _slots:
        centroids = []
        has_real = False
        for s in _slots:
            w = s.get("world_xyz") if hasattr(s, "get") else getattr(s, "world_xyz", None)
            mode = s.get("mode", "") if hasattr(s, "get") else getattr(s, "mode", "")
            if w is None:
                continue
            centroids.append(np.asarray(w, dtype=float)[:2])
            if str(mode) == "normals":
                has_real = True
        if has_real and len(centroids) >= 2:
            array_axis = _pca_major_angle(centroids)
            if array_axis is not None:
                return (
                    array_axis + np.pi / 2.0,
                    f"slot_centroids(n={len(centroids)})",
                )

    # 2. The mask's MINIMUM-AREA RECTANGLE. A tray is a rectangle, so its
    # direction lives in its OUTLINE and needs no elongation gate; the
    # mass-distribution estimators below degrade as a shape approaches square.
    # PCA is pulled by every mask irregularity (a lip on one edge, a shadow,
    # the utensils already in the tray): on the real tray it reported image AR
    # 1.36 then 1.46 against a measured world extent of 0.348 x 0.217 (=1.60)
    # and swung 25 deg between runs (-78.3 deg placed the spoon correctly,
    # -53.1 deg laid it across its slots). minAreaRect keys off the four sides.
    _m = det_field(det, "_mask", None)
    if _m is None:
        _m = det_field(det, "mask", None)
    if _m is not None:
        try:
            _m8 = (np.asarray(_m) > 0).astype(np.uint8)
            _cnts, _ = cv2.findContours(
                _m8, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
            )
            if _cnts:
                (_cx, _cy), (_w, _h), _deg = cv2.minAreaRect(
                    max(_cnts, key=cv2.contourArea)
                )
                # minAreaRect's angle names the WIDTH side; take the long one.
                _long = _deg if _w >= _h else _deg + 90.0
                _ar = max(_w, _h) / max(min(_w, _h), 1e-6)
                return (
                    float(np.deg2rad(_long)) + np.pi / 2.0,
                    f"image_minrect_perp(AR={_ar:.2f})",
                )
        except Exception as exc:  # noqa: BLE001 - fall through to the old rungs
            logger.warning("[slot_direction] minAreaRect failed: %s", exc)

    # 3. World OBB major axis, only when AR is trustworthy.
    tray_ar = float(det_field(det, "aspect_ratio", 1.0) or 1.0)
    if 1e-6 < tray_ar < 1.0:
        tray_ar = 1.0 / tray_ar
    world_major = det_field(det, "world_major_axis_rad", None)
    if world_major is None:
        world_major = det_field(det, "orientation_angle", None)
    if world_major is not None and tray_ar >= TRAY_OBB_AR_TRUST:
        return (
            float(world_major) + np.pi / 2.0,
            f"world_obb_perp(AR={tray_ar:.2f})",
        )

    # 4. Dense image-frame mask PCA (kept as a fallback).
    mask = det_field(det, "_mask", None)
    if mask is None:
        mask = det_field(det, "mask", None)
    if mask is not None:
        try:
            # minAreaRect, not PCA: a rectangle's angle is carried by its
            # OUTLINE, not its mass distribution (see rung 2).
            m8 = (np.asarray(mask) > 0).astype(np.uint8)
            cnts, _ = cv2.findContours(m8, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            if cnts:
                (_cx, _cy), (_w, _h), _deg = cv2.minAreaRect(
                    max(cnts, key=cv2.contourArea)
                )
                # minAreaRect's angle names the width side; take the LONG side.
                long_deg = _deg if _w >= _h else _deg + 90.0
                rect_ar = (max(_w, _h) / max(min(_w, _h), 1e-6))
                return (
                    float(np.deg2rad(long_deg)) + np.pi / 2.0,
                    f"image_minrect_perp(AR={rect_ar:.2f})",
                )
            img_major, img_ar, _, _ = _pca_obb(np.asarray(mask))
            return (
                float(img_major) + np.pi / 2.0,
                f"image_pca_perp(AR={img_ar:.2f})",
            )
        except Exception as exc:  # noqa: BLE001
            logger.warning("[slot_direction] image-PCA slot axis failed: %s", exc)

    # 5. No reliable source: tray-OBB perpendicular per the geometry above, WARN.
    logger.warning(
        "[slot_direction] no reliable slot-direction source for '%s' "
        "(no real slots, world OBB AR<%.2f, no mask); using tray-OBB "
        "perpendicular; placement yaw may be unreliable",
        container_label,
        TRAY_OBB_AR_TRUST,
    )
    fallback = float(world_major) if world_major is not None else 0.0
    return fallback + np.pi / 2.0, "tray_obb_perp(fallback)"

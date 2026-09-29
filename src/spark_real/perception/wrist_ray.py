"""RGB-only pixel -> 3D via ray/plane intersection.

A single RGB pixel fixes a RAY through the camera centre, not a point: the
range along it is unconstrained. This module recovers the missing scalar from
a horizontal-plane prior (z = z_plane in robot base frame) instead of from a
depth sensor, so wrist refinement works with the D435i opened COLOR-ONLY.

Pure geometry: no robot, camera, or SAM3 imports. See
control.executor_motion._refine_with_wrist for the caller.
"""

import numpy as np

# A direction whose |dz| is under this is treated as parallel to the plane:
# the intersection is either nonexistent or so far out that the pixel carries
# no usable lateral information.
PARALLEL_EPS = 1e-6


def unpack_intrinsics(intr):
    """Pull (fx, fy, cx, cy) out of a calibration object, mapping, or sequence.

    Accepts CameraCalibration (attributes), a dict, or a 4-sequence in
    fx, fy, cx, cy order.
    """
    if hasattr(intr, "fx"):
        return float(intr.fx), float(intr.fy), float(intr.cx), float(intr.cy)
    if isinstance(intr, dict):
        return (
            float(intr["fx"]),
            float(intr["fy"]),
            float(intr["cx"]),
            float(intr["cy"]),
        )
    fx, fy, cx, cy = intr
    return float(fx), float(fy), float(cx), float(cy)


def pixel_ray_base(u: float, v: float, intr, T_cam_to_base: np.ndarray):
    """Ray through pixel (u, v), expressed in the robot base frame.

    Returns (origin, direction) with direction unit-length. origin is the
    camera centre, i.e. the translation column of T_cam_to_base.
    """
    fx, fy, cx, cy = unpack_intrinsics(intr)
    T = np.asarray(T_cam_to_base, dtype=float)
    # Pinhole: the ray through (u, v) hits z_cam=1 at these camera-frame coords.
    d_cam = np.array([(float(u) - cx) / fx, (float(v) - cy) / fy, 1.0])
    direction = T[:3, :3] @ d_cam
    norm = float(np.linalg.norm(direction))
    if norm < PARALLEL_EPS:
        raise ValueError("degenerate camera->base rotation")
    return T[:3, 3].astype(float).copy(), direction / norm


def intersect_ray_plane(origin, direction, z_plane: float):
    """Intersect a ray with the horizontal plane z = z_plane.

    Returns the xyz hit point, or None when the ray is parallel to the plane
    or points AWAY from it (t <= 0 would put the "hit" behind the camera).
    """
    o = np.asarray(origin, dtype=float)
    d = np.asarray(direction, dtype=float)
    if abs(d[2]) < PARALLEL_EPS:
        return None
    t = (float(z_plane) - o[2]) / d[2]
    if t <= 0.0:
        return None
    return o + t * d


def pixel_to_plane_point(
    u: float, v: float, intr, T_cam_to_base: np.ndarray, z_plane: float
):
    """pixel_ray_base + intersect_ray_plane. None on either degenerate case."""
    origin, direction = pixel_ray_base(u, v, intr, T_cam_to_base)
    return intersect_ray_plane(origin, direction, z_plane)


def project_point_to_pixel(point, intr, T_cam_to_base: np.ndarray):
    """Inverse of pixel_to_plane_point: base-frame xyz -> (u, v).

    Returns None when the point is at or behind the camera plane (z_cam <= 0).
    Used to round-trip the geometry in tests and to sanity-check a refinement.
    """
    fx, fy, cx, cy = unpack_intrinsics(intr)
    T = np.asarray(T_cam_to_base, dtype=float)
    p_cam = T[:3, :3].T @ (np.asarray(point, dtype=float) - T[:3, 3])
    if p_cam[2] <= PARALLEL_EPS:
        return None
    return (
        fx * p_cam[0] / p_cam[2] + cx,
        fy * p_cam[1] / p_cam[2] + cy,
    )

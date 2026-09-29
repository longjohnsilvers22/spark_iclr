"""
Image-space <-> base-frame conversions for RoboInter representations.

RoboInter (arXiv 2602.09973) states most of its intermediate representations in
2D image space: boxes, contact points, traces. The executor works in the robot
base frame. This module is the bridge, and it is deliberately explicit about
what each direction costs:

  base -> pixel            NO depth needed. A 3D point projects to exactly one
                           pixel. Used to draw a proposal back onto the image
                           and to score the planner against perception.

  pixel -> base            NEEDS a depth VALUE in metres for that pixel. A
                           pixel is a ray, not a point. Supply it from the
                           hardware depth map (`sample_depth`) or from a
                           detection's already-solved ``position_3d[2]``.

  pixel -> base on a plane NO depth map needed, but needs a known plane height
                           ``z_base``. This is the one that matters for a
                           tabletop: a placement proposal, a contact point on
                           a lying object and a transport trace all live on a
                           horizontal plane whose height is already known (the
                           table, or the top of the grasped object). It is also
                           the ONLY option for the RGB-only human corpus.

Camera convention is the hardware/OpenCV one used by
``perception.mask_geometry._cloud_median_world`` on the calibrated branch::

    x_cam = (u - cx) * d / fx
    y_cam = (v - cy) * d / fy
    z_cam = d
    p_base = R @ p_cam + t          # R,t from the 4x4 camera->base extrinsic

The MuJoCo/fovy branch of that function uses a different sign convention
(y and z negated). It is NOT supported here: a sim camera model would silently
mirror every proposal. Pass a calibrated CameraCalibration or nothing.

Pure numpy. No torch, no cv2, no camera, no I/O.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional, Sequence, Tuple

import numpy as np

# A ray whose direction is this close to parallel with the plane cannot be
# intersected with it to any useful accuracy.
_MIN_PLANE_COS = 1e-6


@dataclass(frozen=True)
class CameraModel:
    """Pinhole intrinsics + a 4x4 camera->base extrinsic."""

    fx: float
    fy: float
    cx: float
    cy: float
    width: int
    height: int
    extrinsic: np.ndarray  # 4x4, camera frame -> robot base frame
    name: str = ""

    def __post_init__(self):
        ext = np.asarray(self.extrinsic, dtype=float)
        if ext.shape != (4, 4):
            raise ValueError(f"extrinsic must be 4x4, got {ext.shape}")
        if not np.isfinite(ext).all():
            raise ValueError("extrinsic contains non-finite values")
        for name in ("fx", "fy"):
            if not float(getattr(self, name)) > 0.0:
                raise ValueError(f"{name} must be positive, got {getattr(self, name)}")
        if self.width <= 0 or self.height <= 0:
            raise ValueError("width and height must be positive")
        object.__setattr__(self, "extrinsic", ext)

    @classmethod
    def from_calibration(cls, cal, name: str = "") -> "CameraModel":
        """Build from a ``calibration.model.CameraCalibration`` (duck-typed).

        Raises if the extrinsic is identity: an uncalibrated camera cannot
        place anything in the base frame, and silently returning camera-frame
        coordinates would be worse than failing.
        """
        ext = np.asarray(getattr(cal, "extrinsic", np.eye(4)), dtype=float)
        if np.allclose(ext, np.eye(4)):
            raise ValueError(
                f"camera {name or getattr(cal, 'name', '?')!r} is uncalibrated "
                "(identity extrinsic); refusing to produce base-frame points"
            )
        return cls(
            fx=float(cal.fx),
            fy=float(cal.fy),
            cx=float(cal.cx),
            cy=float(cal.cy),
            width=int(cal.width),
            height=int(cal.height),
            extrinsic=ext,
            name=name or str(getattr(cal, "name", "")),
        )

    @property
    def rotation(self) -> np.ndarray:
        return self.extrinsic[:3, :3]

    @property
    def position(self) -> np.ndarray:
        return self.extrinsic[:3, 3]

    @property
    def image_size(self) -> Tuple[int, int]:
        return (int(self.width), int(self.height))

    # pixel -> base

    def ray_base(self, u: float, v: float) -> Tuple[np.ndarray, np.ndarray]:
        """(origin, unit direction) in the base frame for pixel (u, v)."""
        d_cam = np.array([(float(u) - self.cx) / self.fx, (float(v) - self.cy) / self.fy, 1.0])
        d_base = self.rotation @ d_cam
        n = float(np.linalg.norm(d_base))
        if n < 1e-12:
            raise ValueError("degenerate camera rotation")
        return self.position.copy(), d_base / n

    def pixel_to_base(self, u: float, v: float, depth_m: float) -> np.ndarray:
        """Backproject a pixel with a known depth (metres along camera Z)."""
        d = float(depth_m)
        if not np.isfinite(d) or d <= 0.0:
            raise ValueError(f"depth must be a positive finite metre value, got {d!r}")
        p_cam = np.array(
            [(float(u) - self.cx) * d / self.fx, (float(v) - self.cy) * d / self.fy, d]
        )
        return self.rotation @ p_cam + self.position

    def pixel_to_base_on_plane(self, u: float, v: float, z_base: float) -> Optional[np.ndarray]:
        """Intersect the pixel ray with the horizontal plane ``z = z_base``.

        Returns None when the ray is parallel to the plane or hits it behind
        the camera -- both mean the pixel does not correspond to a point on
        that plane, and inventing one would put the arm somewhere arbitrary.
        """
        origin, direction = self.ray_base(u, v)
        if abs(direction[2]) < _MIN_PLANE_COS:
            return None
        t = (float(z_base) - origin[2]) / direction[2]
        if t <= 0.0:
            return None
        return origin + t * direction

    # base -> pixel

    def base_to_pixel(self, point_base: Sequence[float]) -> Optional[Tuple[float, float]]:
        """Project a base-frame point to (u, v). None if behind the camera."""
        p = np.asarray(point_base, dtype=float).reshape(3)
        p_cam = self.rotation.T @ (p - self.position)
        if p_cam[2] <= 1e-6:
            return None
        return (
            float(self.cx + p_cam[0] * self.fx / p_cam[2]),
            float(self.cy + p_cam[1] * self.fy / p_cam[2]),
        )

def sample_depth(
    depth: np.ndarray,
    u: float,
    v: float,
    radius: int = 2,
    min_depth: float = 0.01,
    max_depth: float = 10.0,
) -> Optional[float]:
    """Median valid depth in a small window around (u, v), or None.

    A single depth pixel at an LLM-proposed coordinate is a coin flip on an
    object edge; the window is what makes it usable.
    """
    if depth is None:
        return None
    arr = np.asarray(depth)
    if arr.ndim != 2:
        return None
    h, w = arr.shape
    ui, vi = int(round(float(u))), int(round(float(v)))
    if not (0 <= ui < w and 0 <= vi < h):
        return None
    r = max(0, int(radius))
    patch = arr[max(0, vi - r) : vi + r + 1, max(0, ui - r) : ui + r + 1].astype(float)
    valid = patch[(patch > min_depth) & (patch < max_depth)]
    if valid.size == 0:
        return None
    return float(np.median(valid))


def box_to_sam3_prompt(
    x1: float, y1: float, x2: float, y2: float
) -> Tuple[float, float, float, float]:
    """Normalized corner box -> SAM3's ``[cx, cy, w, h]`` geometric prompt.

    SAM3's ``add_geometric_prompt`` takes centre/extent in [0,1]; RoboInter
    states boxes as corners. Nothing on the detect path consumes this yet.
    """
    lo_x, hi_x = sorted((float(x1), float(x2)))
    lo_y, hi_y = sorted((float(y1), float(y2)))
    return ((lo_x + hi_x) / 2.0, (lo_y + hi_y) / 2.0, hi_x - lo_x, hi_y - lo_y)

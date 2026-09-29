"""
Quick camera-to-robot calibration for SPARK real deployment.

Two calibration methods:
1. N-point calibration: Touch robot TCP to N objects, click on them in camera.
   Solves for rigid transform via SVD.
2. Manual: Set camera position and orientation by hand.

Usage (from pipeline):
    from spark_real.calibration import compute_transform, CalibrationPoint
    points = [CalibrationPoint(robot=[x,y,z], camera=[cx,cy,cz]), ...]
    T = compute_transform(points)
"""

import logging
from dataclasses import dataclass
from typing import List, Optional

import numpy as np

logger = logging.getLogger(__name__)


@dataclass
class CalibrationPoint:
    # A single calibration correspondence.
    robot: np.ndarray  # TCP position [x, y, z] in robot base frame
    camera: np.ndarray  # Detected position [x, y, z] in camera frame


def compute_transform(points: List[CalibrationPoint]) -> np.ndarray:
    """
    Compute the 4x4 rigid transform from camera frame to robot base frame.

    Uses SVD-based point cloud registration (Arun et al., 1987).
    Minimum 3 non-collinear points required.

    Args:
        points: List of CalibrationPoint with robot and camera positions.

    Returns:
        4x4 homogeneous transform T such that: p_robot = T @ p_camera_hom
    """
    n = len(points)
    if n < 3:
        raise ValueError(f"Need at least 3 points, got {n}")

    src = np.array([p.camera for p in points])  # (N, 3) camera positions
    dst = np.array([p.robot for p in points])  # (N, 3) robot positions

    # Centroids
    src_mean = src.mean(axis=0)
    dst_mean = dst.mean(axis=0)

    # Center the points
    src_c = src - src_mean
    dst_c = dst - dst_mean

    # Cross-covariance matrix
    H = src_c.T @ dst_c  # (3, 3)

    # SVD
    U, S, Vt = np.linalg.svd(H)

    # Rotation
    d = np.linalg.det(Vt.T @ U.T)
    sign_matrix = np.diag([1, 1, d])  # Correct for reflection
    R = Vt.T @ sign_matrix @ U.T

    # Translation
    t = dst_mean - R @ src_mean

    # Build 4x4 transform
    T = np.eye(4)
    T[:3, :3] = R
    T[:3, 3] = t

    # Report error
    errors = []
    for p in points:
        p_cam_hom = np.append(p.camera, 1.0)
        p_robot_est = (T @ p_cam_hom)[:3]
        err = np.linalg.norm(p_robot_est - p.robot)
        errors.append(err)
    mean_err = np.mean(errors)
    max_err = np.max(errors)
    logger.info(
        "Calibration: %d points, mean error = %.1f mm, max error = %.1f mm",
        n,
        mean_err * 1000,
        max_err * 1000,
    )

    return T


def compute_transform_from_pairs(
    robot_positions: List[List[float]],
    camera_positions: List[List[float]],
) -> np.ndarray:
    # Convenience wrapper taking raw lists.
    points = [
        CalibrationPoint(
            robot=np.array(r),
            camera=np.array(c),
        )
        for r, c in zip(robot_positions, camera_positions)
    ]
    return compute_transform(points)


def estimate_transform_1point(
    robot_pos: np.ndarray,
    camera_pos: np.ndarray,
    camera_looking_direction: str = "down",
) -> np.ndarray:
    """
    Rough single-point calibration.

    Assumes camera is mounted above the workspace looking down (or at an angle).
    Uses the single correspondence to estimate translation, assumes identity rotation
    (camera axes ~aligned with robot axes, possibly with axis swaps).

    Good enough for initial testing - refine with N-point later.
    """
    # For a camera looking down at the workspace:
    # Camera -Z axis ~= robot +Z (or close)
    # This is a rough estimate - just offset translation
    T = np.eye(4)
    T[:3, 3] = robot_pos - camera_pos
    return T

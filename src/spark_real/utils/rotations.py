"""
Consolidated rotation conversion helpers for SPARK.

All functions use scipy.spatial.transform.Rotation internally.
Quaternion convention is [x, y, z, w] (SciPy / ROS / SPARK default)
unless a function name says otherwise (e.g. euler_to_quat_wxyz).
"""

from __future__ import annotations

import math
from typing import Sequence

import numpy as np
from scipy.spatial.transform import Rotation

__all__ = [
    "axis_angle_to_quat",
    "quat_to_axis_angle",
    "rotmat_to_rotvec",
    "rotvec_to_rotmat",
    "orientation_error",
    "euler_to_quat_wxyz_str",
    "quat_xyzw_to_rotvec",
    "rotvec_to_quat_xyzw",
]


def axis_angle_to_quat(rx: float, ry: float, rz: float) -> np.ndarray:
    # Convert axis-angle 3-vector (Rodrigues) to [x, y, z, w] quaternion.
    rotvec = np.array([rx, ry, rz], dtype=float)
    return Rotation.from_rotvec(rotvec).as_quat()  # xyzw


def quat_to_axis_angle(q_xyzw: Sequence[float]) -> np.ndarray:
    # Convert [x, y, z, w] quaternion to Rodrigues 3-vector.
    q = np.asarray(q_xyzw, dtype=float)
    return Rotation.from_quat(q).as_rotvec()


def rotmat_to_rotvec(R: np.ndarray) -> np.ndarray:
    # Convert a 3x3 rotation matrix to a Rodrigues rotation vector.
    return Rotation.from_matrix(R).as_rotvec()


def rotvec_to_rotmat(rv: np.ndarray) -> np.ndarray:
    # Convert a Rodrigues rotation vector to a 3x3 rotation matrix.
    return Rotation.from_rotvec(np.asarray(rv, dtype=float)).as_matrix()


def orientation_error(R_desired: np.ndarray, R_current: np.ndarray) -> np.ndarray:
    """
    Orientation error as a rotvec in the base frame.

    Computes R_err = R_desired @ R_current^T, then converts to rotvec.
    """
    R_err = R_desired @ R_current.T
    return Rotation.from_matrix(R_err).as_rotvec()


def euler_to_quat_wxyz_str(yaw: float) -> str:
    # Convert a Z-axis yaw (radians) to MuJoCo quaternion string "w x y z".
    w = math.cos(yaw / 2.0)
    z = math.sin(yaw / 2.0)
    return f"{w:.6f} 0 0 {z:.6f}"


def quat_xyzw_to_rotvec(q_xyzw: np.ndarray) -> np.ndarray:
    # Convert [x, y, z, w] quaternion to Rodrigues rotation vector.
    return Rotation.from_quat(q_xyzw).as_rotvec()


def rotvec_to_quat_xyzw(rotvec) -> np.ndarray:
    # Convert Rodrigues rotation vector to [x, y, z, w] quaternion.
    rotvec = np.asarray(rotvec, dtype=float).reshape(3)
    return Rotation.from_rotvec(rotvec).as_quat()

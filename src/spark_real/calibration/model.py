"""
Camera-to-robot calibration utilities for SPARK real deployment.

Handles:
- Intrinsics extraction from Azure Kinect and RealSense cameras
- Extrinsics management (camera-to-robot-base transforms)
- Hand-eye calibration helpers for wrist-mounted cameras
- Calibration persistence (save/load YAML)
"""

import numpy as np
import yaml
from pathlib import Path
from dataclasses import dataclass, field
from typing import Optional

from scipy.spatial.transform import Rotation as R


@dataclass
class CameraCalibration:
    # Complete camera calibration: intrinsics + extrinsics + depth scale.

    name: str
    width: int
    height: int
    fx: float
    fy: float
    cx: float
    cy: float
    # 4x4 transform: camera frame -> robot base frame
    extrinsic: np.ndarray = field(default_factory=lambda: np.eye(4))
    # Per-camera depth-bias scalar. Multiplied into raw sensor depth at
    # capture time before any 3D backprojection. Real Azure Kinects /
    # RealSense at typical workspace ranges ship a ~15% systematic
    # depth bias that the RGB-D Procrustes hand-eye captures during cal
    # (see scripts/spark_calibrate.py). 1.0 = no correction.
    depth_scale: float = 1.0
    # Per-camera depth-bias offset in meters, applied with the scale as
    # z_corrected = z_raw * depth_scale + depth_offset. A scalar scale alone
    # only makes depth correct at the calibration board's height; the offset
    # term is what keeps Z accurate across the workspace (fit from board
    # captures at multiple heights). 0.0 = no offset.
    depth_offset: float = 0.0

    def __post_init__(self):
        # Callers that forward an optional extrinsic (getattr(..., None)) would
        # otherwise replace the identity default with None and crash every
        # backprojection downstream. Coerce here so a missing extrinsic degrades
        # to "camera frame" rather than to an exception, and so no caller has to
        # remember to build kwargs conditionally.
        if self.extrinsic is None:
            self.extrinsic = np.eye(4)
        else:
            self.extrinsic = np.asarray(self.extrinsic, dtype=float).reshape(4, 4)

    def correct_depth(self, depth: Optional[np.ndarray]) -> Optional[np.ndarray]:
        """Apply the per-camera bias correction z = z*depth_scale + depth_offset.

        The RGB-D Procrustes hand-eye solves its extrinsic WITH this
        correction in the loop (rmse ~1 mm on the board), so any depth that
        gets deprojected through `extrinsic` must have it applied first.
        This is the single implementation: pipeline.capture() and the
        streaming capture path (routes/streaming.capture_single_camera) both
        call it (on birdview depth_scale=0.9879, raw depth is ~19 mm long at
        1.55 m).

        Invalid pixels (<= 0, i.e. no depth return) stay invalid so the
        offset cannot turn them into spurious near readings. No-op when the
        correction is identity or depth is None.
        """
        if depth is None:
            return depth
        s = float(self.depth_scale or 1.0)
        b = float(self.depth_offset or 0.0)
        if s == 1.0 and b == 0.0:
            return depth
        out = depth.astype(np.float32) * s + b
        out[depth <= 0] = 0.0
        return out.astype(depth.dtype)

    @property
    def intrinsic_matrix(self) -> np.ndarray:
        # 3x3 camera intrinsic matrix K.
        return np.array(
            [
                [self.fx, 0, self.cx],
                [0, self.fy, self.cy],
                [0, 0, 1],
            ]
        )

    @property
    def fovy_degrees(self) -> float:
        # Vertical field of view in degrees.
        return float(2 * np.degrees(np.arctan(self.height / (2 * self.fy))))

    @property
    def rotation_matrix(self) -> np.ndarray:
        # 3x3 rotation from camera to world.
        return self.extrinsic[:3, :3]

    @property
    def position(self) -> np.ndarray:
        # Camera position in world frame.
        return self.extrinsic[:3, 3]

    def save(self, path: str):
        # Save calibration to YAML.
        data = {
            "name": self.name,
            "width": self.width,
            "height": self.height,
            "fx": float(self.fx),
            "fy": float(self.fy),
            "cx": float(self.cx),
            "cy": float(self.cy),
            "extrinsic": self.extrinsic.flatten().tolist(),
        }
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        with open(path, "w") as f:
            yaml.dump(data, f, default_flow_style=False)

    @classmethod
    def load(cls, path: str) -> "CameraCalibration":
        # Load calibration from YAML.
        with open(path) as f:
            data = yaml.safe_load(f)
        cal = cls(
            name=data["name"],
            width=data["width"],
            height=data["height"],
            fx=data["fx"],
            fy=data["fy"],
            cx=data["cx"],
            cy=data["cy"],
        )
        if "extrinsic" in data:
            cal.extrinsic = np.array(data["extrinsic"]).reshape(4, 4)
        return cal


def compute_wrist_camera_extrinsic(
    tcp_pose: np.ndarray,
    tool_offset: np.ndarray = None,
) -> np.ndarray:
    """
    Compute wrist camera extrinsic from current TCP pose.

    For a wrist-mounted camera, the extrinsic changes with robot pose.
    T_cam_to_base = T_tcp_to_base @ T_cam_to_tcp

    Args:
        tcp_pose: Current TCP pose [x, y, z, rx, ry, rz] (axis-angle).
        tool_offset: 4x4 transform from camera to TCP (fixed, from calibration).
                    If None, assumes camera is at TCP with identity rotation.

    Returns:
        4x4 camera-to-base transform.
    """
    position = tcp_pose[:3]
    axis_angle = tcp_pose[3:6]
    angle = np.linalg.norm(axis_angle)

    if angle < 1e-6:
        rot_matrix = np.eye(3)
    else:
        rot_matrix = R.from_rotvec(axis_angle).as_matrix()

    t_tcp_to_base = np.eye(4)
    t_tcp_to_base[:3, :3] = rot_matrix
    t_tcp_to_base[:3, 3] = position

    if tool_offset is None:
        tool_offset = np.eye(4)

    return t_tcp_to_base @ tool_offset


def create_default_agentview_calibration(
    position: np.ndarray = None,
    look_at: np.ndarray = None,
    up: np.ndarray = None,
) -> CameraCalibration:
    """
    Create a default agentview calibration for the Azure Kinect.

    Uses a look-at formulation to compute the extrinsic matrix.

    Args:
        position: Camera position in robot base frame [x, y, z].
        look_at: Point the camera is looking at [x, y, z].
        up: World up direction (default: [0, 0, 1]).
    """
    if position is None:
        position = np.array([0.0, -0.8, 1.2])
    if look_at is None:
        look_at = np.array([0.0, 0.0, 0.0])
    if up is None:
        up = np.array([0.0, 0.0, 1.0])

    forward = look_at - position
    forward = forward / np.linalg.norm(forward)

    right = np.cross(forward, up)
    right = right / np.linalg.norm(right)

    cam_up = np.cross(right, forward)

    extrinsic = np.eye(4)
    extrinsic[:3, 0] = right
    extrinsic[:3, 1] = -cam_up
    extrinsic[:3, 2] = forward
    extrinsic[:3, 3] = position

    return CameraCalibration(
        name="agentview",
        width=1920,
        height=1080,
        fx=915.0,
        fy=915.0,
        cx=960.0,
        cy=540.0,
        extrinsic=extrinsic,
    )

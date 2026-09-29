"""
DA3 Nested Giant-Large pose + depth estimation for SPARK real deployment.

Provides automatic camera pose estimation from multi-view images.
Replaces manual hand-eye calibration with learned pose estimation.

Usage:
    estimator = DA3PoseEstimator()
    result = estimator.estimate(images=[pil_img_cam0, pil_img_cam1])
    # result.extrinsics[i] -> 3x4 w2c matrix for camera i
    # result.depth[i]      -> metric depth map for camera i
    # result.intrinsics[i] -> 3x3 intrinsic matrix for camera i
"""

import numpy as np
import torch
import logging
from typing import List, Optional
from dataclasses import dataclass
from PIL import Image

try:
    from depth_anything_3.api import DepthAnything3
except ImportError:
    DepthAnything3 = None

logger = logging.getLogger(__name__)


@dataclass
class PoseEstimationResult:
    """
    Result from DA3 multi-view pose + depth estimation.
    """

    depth: np.ndarray  # (N, H, W) metric depth maps
    extrinsics: np.ndarray  # (N, 3, 4) world-to-camera transforms
    intrinsics: np.ndarray  # (N, 3, 3) camera intrinsic matrices
    confidence: np.ndarray  # (N, H, W) depth confidence maps
    camera_positions: np.ndarray  # (N, 3) camera positions in world frame

    def get_camera_position(self, idx: int) -> np.ndarray:
        """
        Get camera position in world frame: -R^T @ t.
        """
        return self.camera_positions[idx]

    def get_extrinsic_4x4(self, idx: int) -> np.ndarray:
        """
        Get 4x4 homogeneous w2c transform.
        """
        T = np.eye(4)
        T[:3, :] = self.extrinsics[idx]
        return T

    def get_c2w(self, idx: int) -> np.ndarray:
        """
        Get 4x4 camera-to-world transform (inverse of w2c).
        """
        w2c = self.get_extrinsic_4x4(idx)
        return np.linalg.inv(w2c)


class DA3PoseEstimator:
    """
    DA3 Nested Giant-Large for automatic camera pose + metric depth.

    Feeds multi-camera images through DA3 to get:
    - Metric depth maps per view
    - Camera extrinsics (world-to-camera, OpenCV convention)
    - Camera intrinsics
    - Depth confidence maps

    The first image is treated as the reference (placed at origin).
    """

    MODEL_NAME = "depth-anything/DA3NESTED-GIANT-LARGE"

    def __init__(self, device: str = "cuda", use_ray_pose: bool = True):
        """
        Args:
            device: Torch device.
            use_ray_pose: Derive pose from ray head (slower but more accurate).
        """
        self.device = device
        self.use_ray_pose = use_ray_pose
        self._model = None

    def load(self):
        """
        Load the DA3 Nested model.
        """
        if self._model is not None:
            return
        logger.info("Loading DA3 Nested Giant-Large...")
        if DepthAnything3 is None:
            raise RuntimeError("depth_anything_3 not installed; DA3 pose unavailable")

        self._model = DepthAnything3.from_pretrained(self.MODEL_NAME)
        self._model = self._model.to(device=self.device)
        logger.info("DA3 Nested loaded")

    def estimate(
        self,
        images: List[Image.Image],
        known_intrinsics: Optional[np.ndarray] = None,
    ) -> PoseEstimationResult:
        """
        Estimate depth + camera poses from multiple views.

        Args:
            images: List of PIL images (one per camera view).
            known_intrinsics: (N, 3, 3) known camera intrinsic matrices.
                If provided, DA3 uses these instead of estimating its own.

        Returns:
            PoseEstimationResult with depth, extrinsics, intrinsics.
        """
        self.load()

        pred = self._model.inference(
            image=images,
            intrinsics=known_intrinsics,
            use_ray_pose=self.use_ray_pose,
        )

        # Extract camera positions from w2c matrices
        n = len(images)
        cam_positions = np.zeros((n, 3))
        for i in range(n):
            R = pred.extrinsics[i, :3, :3]
            t = pred.extrinsics[i, :3, 3]
            cam_positions[i] = -R.T @ t

        return PoseEstimationResult(
            depth=pred.depth,
            extrinsics=pred.extrinsics,
            intrinsics=pred.intrinsics,
            confidence=pred.conf,
            camera_positions=cam_positions,
        )

    def _compute_registration(
        self,
        result: PoseEstimationResult,
        cam_idx: int,
        pixel: tuple,
        robot_pos: np.ndarray,
    ) -> np.ndarray:
        """
        Compute DA3 world frame -> robot base frame transform from one anchor.

        Uses the known correspondence: pixel in camera -> robot position.
        """
        u, v = int(pixel[0]), int(pixel[1])
        h, w = result.depth[cam_idx].shape
        u = min(max(u, 0), w - 1)
        v = min(max(v, 0), h - 1)

        depth_val = float(result.depth[cam_idx, v, u])
        if depth_val <= 0:
            logger.warning("No depth at anchor pixel (%d, %d)", u, v)
            return np.eye(4)

        K = result.intrinsics[cam_idx]
        fx, fy = K[0, 0], K[1, 1]
        cx, cy = K[0, 2], K[1, 2]

        # Backproject to camera frame
        x_cam = (u - cx) * depth_val / fx
        y_cam = (v - cy) * depth_val / fy
        z_cam = depth_val
        p_cam = np.array([x_cam, y_cam, z_cam, 1.0])

        # Camera to DA3 world
        c2w = result.get_c2w(cam_idx)
        p_da3_world = (c2w @ p_cam)[:3]

        # DA3 world -> robot: a translation offset (assumes DA3 axes ~ robot
        # axes). For more robustness, solve rotation from multiple anchors too.
        offset = robot_pos - p_da3_world
        T = np.eye(4)
        T[:3, 3] = offset
        return T

    def _backproject_mask(
        self,
        result: PoseEstimationResult,
        cam_idx: int,
        mask: np.ndarray,
        da3_to_robot: np.ndarray,
    ) -> Optional[np.ndarray]:
        """
        Backproject a binary mask to 3D using DA3 depth + pose.
        """
        depth = result.depth[cam_idx]
        K = result.intrinsics[cam_idx]
        c2w = result.get_c2w(cam_idx)

        # Handle resolution mismatch (DA3 may resize)
        if mask.shape != depth.shape:
            mask_resized = (
                np.array(
                    Image.fromarray(mask.astype(np.uint8)).resize(
                        (depth.shape[1], depth.shape[0]),
                        Image.NEAREST,
                    )
                )
                > 0
            )
        else:
            mask_resized = mask > 0

        ys, xs = np.where(mask_resized)
        if len(xs) == 0:
            return None

        mask_depths = depth[mask_resized]
        valid = (mask_depths > 0.01) & (mask_depths < 20)
        if valid.sum() < 3:
            return None

        vx = xs[valid].astype(np.float64)
        vy = ys[valid].astype(np.float64)
        vd = mask_depths[valid]

        fx, fy = K[0, 0], K[1, 1]
        cx, cy = K[0, 2], K[1, 2]

        # Camera frame
        x_cam = (vx - cx) * vd / fx
        y_cam = (vy - cy) * vd / fy
        z_cam = vd
        pts_cam = np.stack([x_cam, y_cam, z_cam, np.ones_like(z_cam)], axis=1)

        # DA3 world frame
        pts_world = (c2w @ pts_cam.T).T[:, :3]

        # Robot frame
        pts_robot = (da3_to_robot[:3, :3] @ pts_world.T).T + da3_to_robot[:3, 3]

        return np.median(pts_robot, axis=0)

"""
Standalone SAM3 perception - no ROS2 dependency.

Wraps SAM3 for:
- Text-prompted object detection
- Video tracking across frames
- 3D backprojection using depth maps
"""

import logging
from dataclasses import dataclass
from typing import Optional

import numpy as np

from spark_real.perception.camera import CameraConfig

logger = logging.getLogger(__name__)

try:
    import torch

    HAS_TORCH = True
except ImportError:
    HAS_TORCH = False

try:
    from sam3.model_builder import build_sam3_image_model
    from sam3.predictor import SAM3ImagePredictor

    HAS_SAM3 = True
except ImportError:
    HAS_SAM3 = False
    logger.warning(
        "sam3 not installed. Detection/tracking will not work. "
        "Install sam3 from its repository: pip install -e ."
    )

try:
    import mujoco  # optional, sim-only path
except ImportError:
    mujoco = None


@dataclass
class Detection:
    """
    Single object detection result.
    """

    label: str
    mask: np.ndarray  # (H, W) binary mask
    bbox: np.ndarray  # [x1, y1, x2, y2]
    confidence: float
    centroid_2d: np.ndarray  # [u, v] pixel coordinates
    position_3d: Optional[np.ndarray] = None  # [x, y, z] in world frame


class SAM3Detector:
    """
    Standalone SAM3 object detector and tracker.

    Usage:
        detector = SAM3Detector()
        detector.load_model()

        # Single-frame detection
        detections = detector.detect(rgb_image, ["red block", "blue cylinder"])

        # With depth for 3D positions
        detections = detector.detect(rgb_image, prompts, depth_map, camera_config)

        # Video tracking
        detector.start_tracking(initial_detections)
        tracked = detector.track_frame(new_rgb_frame)
    """

    def __init__(self, model_size: str = "large", device: str = "cuda"):
        self.model_size = model_size
        self.device = device if HAS_TORCH and torch.cuda.is_available() else "cpu"
        self._model = None
        self._predictor = None
        self._video_predictor = None
        self._tracking_state = None

    def load_model(self):
        """
        Load SAM3 model.
        """
        if not HAS_SAM3:
            raise RuntimeError("SAM3 not installed")

        logger.info("Loading %s model on %s...", self.model_size, self.device)
        self._model = build_sam3_image_model(
            model_size=self.model_size, device=self.device
        )
        self._predictor = SAM3ImagePredictor(self._model)
        logger.info("SAM3 model loaded")

    def detect(
        self,
        image: np.ndarray,
        prompts: list,
        depth: np.ndarray = None,
        camera_config=None,
    ) -> list:
        """
        Detect objects using text prompts.

        Args:
            image: RGB image (H, W, 3)
            prompts: List of text descriptions ["red block", "blue cylinder"]
            depth: Optional depth map (H, W) in meters
            camera_config: Optional CameraConfig for 3D backprojection

        Returns:
            List of Detection objects
        """
        if self._predictor is None:
            raise RuntimeError("Model not loaded. Call load_model() first.")

        self._predictor.set_image(image)

        detections = []
        for prompt in prompts:
            # SAM3 text-prompted prediction
            masks, scores, _ = self._predictor.predict(
                text=prompt,
                multimask_output=True,
            )

            if masks is None or len(masks) == 0:
                continue

            best_idx = np.argmax(scores)
            mask = masks[best_idx]
            score = float(scores[best_idx])

            if score < 0.3:  # Confidence threshold
                continue

            # Compute bbox and centroid from mask
            ys, xs = np.where(mask > 0)
            if len(xs) == 0:
                continue

            bbox = np.array([xs.min(), ys.min(), xs.max(), ys.max()])
            centroid = np.array([xs.mean(), ys.mean()])

            det = Detection(
                label=prompt,
                mask=mask.astype(np.uint8),
                bbox=bbox,
                confidence=score,
                centroid_2d=centroid,
            )

            # 3D backprojection if depth available
            if depth is not None and camera_config is not None:
                u, v = int(centroid[0]), int(centroid[1])
                if 0 <= v < depth.shape[0] and 0 <= u < depth.shape[1]:
                    d = float(depth[v, u])
                    if d > 0.01:  # Valid depth
                        point_cam = camera_config.backproject(u, v, d)
                        det.position_3d = camera_config.to_world(point_cam)

            detections.append(det)

        return detections

    def detect_in_mujoco(
        self,
        model,
        data,
        cam_idx: int = 0,
        prompts: list = None,
        width: int = 640,
        height: int = 480,
    ) -> list:
        """
        Detect objects directly from MuJoCo rendered images.

        Args:
            model: MuJoCo model
            data: MuJoCo data
            cam_idx: Camera index in model
            prompts: Text prompts for detection
            width: Render width
            height: Render height

        Returns:
            List of Detection objects with 3D positions
        """
        if mujoco is None:
            raise RuntimeError("mujoco not installed; detect_in_mujoco unavailable")

        # Render RGB
        renderer = mujoco.Renderer(model, height, width)
        renderer.update_scene(data, camera=cam_idx)
        rgb = renderer.render().copy()

        # Render depth
        renderer.enable_depth_rendering()
        depth = renderer.render().copy()
        renderer.disable_depth_rendering()
        renderer.close()

        if len(depth.shape) == 3:
            depth = depth[:, :, 0]

        # Get camera intrinsics from MuJoCo
        fovy = model.cam_fovy[cam_idx]
        f = height / (2.0 * np.tan(np.deg2rad(fovy) / 2.0))
        cam_config = CameraConfig(
            width=width,
            height=height,
            fx=f,
            fy=f,
            cx=width / 2.0,
            cy=height / 2.0,
        )

        # Get camera extrinsic (cam-to-world transform)
        cam_pos = data.cam_xpos[cam_idx].copy()
        cam_mat = data.cam_xmat[cam_idx].reshape(3, 3).copy()
        extrinsic = np.eye(4)
        extrinsic[:3, :3] = cam_mat
        extrinsic[:3, 3] = cam_pos
        cam_config.extrinsic = extrinsic

        if prompts is None:
            prompts = ["object"]

        return self.detect(rgb, prompts, depth, cam_config)

    def start_tracking(self, detections: list, frame_buffer: list = None):
        """
        Initialize video tracking from initial detections.

        Args:
            detections: List of Detection objects (from detect())
            frame_buffer: Optional list of recent RGB frames for context
        """
        self._tracking_state = {
            "detections": detections,
            "frame_count": 0,
        }
        # SAM3 video tracking initialization would go here
        # For now, track using per-frame re-detection as fallback

    def track_frame(
        self, image: np.ndarray, depth: np.ndarray = None, camera_config=None
    ) -> list:
        """
        Track objects in new frame.

        Args:
            image: New RGB frame
            depth: Optional depth
            camera_config: Optional camera config for 3D

        Returns:
            Updated Detection list
        """
        if self._tracking_state is None:
            raise RuntimeError("Call start_tracking() first")

        # Re-detect using stored labels
        labels = [d.label for d in self._tracking_state["detections"]]
        new_detections = self.detect(image, labels, depth, camera_config)
        self._tracking_state["detections"] = new_detections
        self._tracking_state["frame_count"] += 1
        return new_detections

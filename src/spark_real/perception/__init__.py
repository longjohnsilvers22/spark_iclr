"""
SPARK perception package.

Single-arm pipeline uses three named slots (``_kinect``, ``_kinect2``,
``_realsense``) on :class:`SPARKRealPipeline`. Bimanual pipeline uses
:class:`CameraRegistry` from :mod:`bimanual_camera_registry`:
role-keyed (``external``, ``wrist_left``, ``wrist_right``) so the
streaming, visualisation, and calibration code can iterate cameras
without embedding their names.
"""

from spark_real.perception.spark_perception import (  # noqa: F401
    ObjectDetection,
    SPARKPerception,
)
from spark_real.perception.camera import (  # noqa: F401
    AzureKinectCamera,
    CameraConfig,
    RealSenseCamera,
    USBCamera,
)

__all__ = [
    "ObjectDetection",
    "SPARKPerception",
    "AzureKinectCamera",
    "CameraConfig",
    "RealSenseCamera",
    "USBCamera",
]

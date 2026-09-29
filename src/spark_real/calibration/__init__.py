"""
Camera-to-robot calibration package for SPARK real deployment.

Modules:
- ``model``: CameraCalibration dataclass, YAML IO, wrist-camera extrinsics.
- ``solver``: SVD point-pair registration.

Re-exports the full public API of both, so
``from spark_real.calibration import X`` works for every symbol.
"""

from spark_real.calibration.model import (
    CameraCalibration,
    compute_wrist_camera_extrinsic,
    create_default_agentview_calibration,
)
from spark_real.calibration.solver import (
    CalibrationPoint,
    compute_transform,
    compute_transform_from_pairs,
    estimate_transform_1point,
)

__all__ = [
    "CameraCalibration",
    "compute_wrist_camera_extrinsic",
    "create_default_agentview_calibration",
    "CalibrationPoint",
    "compute_transform",
    "compute_transform_from_pairs",
    "estimate_transform_1point",
]

"""
Same-source stamping in merge_detections (port step 1).

The merged detection keeps the PRIMARY camera's own backprojection in
position_agentview even after the sideview elevation win rewrites
position_3d[2]; displacement gates read the same-source field so a 1-2 cm
systematic cross-camera disagreement never reads as a phantom move.
Defensive, not load-bearing (dyn journal it-6).
"""

import numpy as np

from spark_real.perception.spark_perception import ObjectDetection
from spark_real.pipeline_perception import PerceptionMixin


def _det(label, camera, pos, conf=0.9):
    d = ObjectDetection(
        label=label,
        confidence=conf,
        centroid_2d=(10.0, 10.0),
        mask_area=100,
        depth_meters=0.5,
        position_3d=np.array(pos, dtype=float),
    )
    d.camera = camera
    return d


class _MergeOnly(PerceptionMixin):
    """Bare host for merge_detections (no pipeline state needed)."""


def _merge(dets):
    return _MergeOnly().merge_detections.__func__(_MergeOnly(), dets)


def test_stamp_survives_sideview_elevation_win():
    bird = _det("bottle", "birdview", [0.10, 0.20, -0.25])
    # Sideview sees the same label >=5cm higher -> Z-override fires.
    side = _det("bottle", "sideview", [0.11, 0.21, -0.10])
    merged = _merge([bird, side])
    assert len(merged) == 1
    m = merged[0]
    # Fused position took the sideview Z...
    assert abs(m.position_3d[2] - (-0.10)) < 1e-9
    # ...but the same-source stamp keeps the birdview backprojection.
    assert m.position_agentview is not None
    assert np.allclose(m.position_agentview, [0.10, 0.20, -0.25])


def test_z_override_no_longer_mutates_the_source_detection():
    bird = _det("bottle", "birdview", [0.10, 0.20, -0.25])
    side = _det("bottle", "sideview", [0.11, 0.21, -0.10])
    _merge([bird, side])
    # copy.copy is shallow; the merge must re-materialize position_3d so
    # the per-camera original is untouched by the Z-override.
    assert abs(bird.position_3d[2] - (-0.25)) < 1e-9


def test_secondary_only_detection_stamps_its_own_source():
    side = _det("mug", "sideview", [0.4, 0.1, -0.2])
    merged = _merge([side])
    assert len(merged) == 1
    assert np.allclose(merged[0].position_agentview, [0.4, 0.1, -0.2])


def test_delta_application_preserves_cross_camera_correction():
    # The _shift_binding pattern: an accepted same-source move is applied
    # as a DELTA to the fused position, preserving the sideview Z win.
    bird = _det("bottle", "birdview", [0.10, 0.20, -0.25])
    side = _det("bottle", "sideview", [0.11, 0.21, -0.10])
    m = _merge([bird, side])[0]
    fresh_primary = np.array([0.15, 0.20, -0.25])  # moved +5cm in x
    delta = fresh_primary - m.position_agentview
    shifted = np.asarray(m.position_3d, dtype=float) + delta
    assert np.allclose(shifted, [0.15, 0.20, -0.10])

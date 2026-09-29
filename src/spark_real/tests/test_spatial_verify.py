"""The remembered-region (spatial) verification rung.

Pins the 2026-08-20 design: identity is carried by the APPROVED scene's
geometry, never by re-matching labels -- the property that makes the rung
work for a tray of identical sockets and for slots that vanish when filled.
"""

import types

import numpy as np

from spark_real.control import spatial_verify
from spark_real.control.spatial_verify import remember_targets, spatial_rung


def _det(label, cam, centroid, conf=0.9):
    d = types.SimpleNamespace()
    d.label, d.camera, d.centroid_2d, d.confidence = label, cam, centroid, conf
    return d


class _Pipeline:
    def __init__(self, dets):
        self._dets = dets
        self.detect_prompts = None

    def capture(self):
        return {"birdview": {"rgb": np.zeros((4, 4, 3), np.uint8)}}

    def detect(self, captures, prompts, **kw):
        self.detect_prompts = list(prompts)
        return self._dets

    def merge_detections(self, dets, **kw):
        return dets


class _Exec:
    def __init__(self, dets, place_label="screwdriver slot", obj="screwdriver"):
        self.detection_map = {
            place_label: {"bbox": [100, 100, 300, 160], "_camera": "birdview",
                          "confidence": 0.6},
        }
        self._last_place_label = place_label
        self._active_grasp_label = obj
        self._pipeline = _Pipeline(dets)


def test_the_memory_is_a_snapshot_not_a_live_view():
    ex = _Exec([])
    remember_targets(ex)
    ex.detection_map["screwdriver slot"]["bbox"] = [0, 0, 1, 1]  # mid-run mutation
    assert ex._spatial_memory["screwdriver slot"]["bbox"] == [100, 100, 300, 160]


def test_object_landing_in_the_remembered_region_passes():
    ex = _Exec([_det("screwdriver", "birdview", (210.0, 128.0))])
    remember_targets(ex)
    v = spatial_rung(ex)
    assert v["status"] == "pass"
    # The target label must never be re-detected: its disappearance is a
    # success signal, and with N identical slots a re-match is meaningless.
    assert ex._pipeline.detect_prompts == ["screwdriver"]


def test_object_confidently_elsewhere_fails():
    ex = _Exec([_det("screwdriver", "birdview", (700.0, 500.0))])
    remember_targets(ex)
    assert spatial_rung(ex)["status"] == "fail"


def test_object_not_seen_abstains_rather_than_guessing():
    ex = _Exec([])  # seated so deep the slot swallowed it, or out of frame
    remember_targets(ex)
    assert spatial_rung(ex)["status"] == "abstain"


def test_low_confidence_redetection_is_not_evidence():
    ex = _Exec([_det("screwdriver", "birdview", (700.0, 500.0),
                     conf=spatial_verify.MIN_CONF - 0.05)])
    remember_targets(ex)
    assert spatial_rung(ex)["status"] == "abstain"


def test_wrong_camera_detections_are_ignored():
    ex = _Exec([_det("screwdriver", "sideview", (210.0, 128.0))])
    remember_targets(ex)
    assert spatial_rung(ex)["status"] == "abstain"

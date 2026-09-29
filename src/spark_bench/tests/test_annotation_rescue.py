"""
Annotation-rescue rung tests (mocked provider + mocked SAM3 click head).

Requires robosuite/mujoco at import time only because the module under
test (libero_pro.perception) does; skipped where those are absent.
Run: PYTHONPATH=src conda run -n openvla_env python -m pytest \
        src/spark_bench/tests/test_annotation_rescue.py -q
"""

import numpy as np
import pytest

pytest.importorskip("robosuite")

import spark_real.perception.annotations as annotations_mod
from spark_real.perception.annotations import Annotation

import spark_bench.libero_pro.perception as percep
from spark_bench.libero_pro.perception import (
    CameraParams,
    DetectionResult,
    annotation_rescue,
)

W, H = 640, 480


class _Cfg:
    annotation_rescue = True
    annotation_provider = "mock"
    use_sam3_service = True  # fake service below: keeps _get_sam3 untouched
    verbose = False
    cam_width = W
    cam_height = H


class _MockProvider:
    """Points at the frame centre for every query; counts calls."""

    name = "mock"

    def __init__(self, respond=True):
        self.respond = respond
        self.queries = []

    def available(self):
        return True

    def annotate(self, image, query, kind="point"):
        self.queries.append((query, kind))
        if not self.respond:
            return []
        return [Annotation(kind="point", points=[(0.5, 0.5)],
                           provider=self.name, label=query)]


def _det_result():
    rgb = np.zeros((H, W, 3), np.uint8)
    depth = np.full((H, W), 0.5, np.float32)
    cam = CameraParams(pos=np.array([0.0, 0.0, 1.0]), mat=np.eye(3), fovy=90.0)
    return DetectionResult(rgb=rgb, depth=depth, wrist_rgb=None, cam=cam,
                           det_map={}, dets=[])


def _center_mask_click(*args, **kwargs):
    """Stand-in for _refine_from_click: a 20x20 blob at the click."""
    mask = np.zeros((H, W), bool)
    mask[H // 2 - 10:H // 2 + 10, W // 2 - 10:W // 2 + 10] = True
    return {"phrase": "<click>", "mask": mask, "score": 0.9,
            "centroid": (W / 2, H / 2), "bbox": (0, 0, 1, 1)}


@pytest.fixture
def wired(monkeypatch):
    provider = _MockProvider()
    monkeypatch.setattr(annotations_mod, "get_provider",
                        lambda name, **kw: provider)
    monkeypatch.setattr(percep, "_refine_from_click", _center_mask_click)
    monkeypatch.setattr(percep, "_service_client_if_enabled",
                        lambda cfg: object())
    return provider


class TestAnnotationRescue:
    def test_missing_label_gets_rescued(self, wired):
        det = _det_result()
        meta = {}
        annotation_rescue(det, ["plushie"], _Cfg(), trial_meta=meta)
        assert wired.queries == [("plushie", "point")]
        assert "plushie" in det.det_map
        d = det.det_map["plushie"]
        # Centre of a top-down camera at 1 m, object plane at 0.5 m depth.
        np.testing.assert_allclose(d.position_3d, [0.0, 0.0, 0.5], atol=0.02)
        assert d.confidence == pytest.approx(0.9)
        assert det.dets and det.dets[0] is d
        (rec,) = meta["annotation_rescue"]
        assert rec["ok"] and rec["label"] == "plushie"
        assert rec["point_px"] == [W / 2, H / 2]

    def test_present_labels_never_queried(self, wired):
        det = _det_result()
        det.det_map["bowl"] = object()
        annotation_rescue(det, ["bowl"], _Cfg(), trial_meta={})
        assert wired.queries == []

    def test_gate_off_is_a_noop(self, wired):
        det = _det_result()
        cfg = _Cfg()
        cfg.annotation_rescue = False
        annotation_rescue(det, ["plushie"], cfg, trial_meta={})
        assert wired.queries == []
        assert det.det_map == {}

    def test_provider_miss_fails_open(self, monkeypatch):
        provider = _MockProvider(respond=False)
        monkeypatch.setattr(annotations_mod, "get_provider",
                            lambda name, **kw: provider)
        monkeypatch.setattr(percep, "_service_client_if_enabled",
                            lambda cfg: object())
        det = _det_result()
        meta = {}
        annotation_rescue(det, ["plushie"], _Cfg(), trial_meta=meta)
        assert det.det_map == {}
        (rec,) = meta["annotation_rescue"]
        assert not rec["ok"] and rec["error"] == "no_point"

    def test_no_depth_fails_open(self, wired):
        det = _det_result()
        det.depth = None
        annotation_rescue(det, ["plushie"], _Cfg(), trial_meta={})
        assert wired.queries == []
        assert det.det_map == {}

    def test_multiphase_rename_respected(self, wired):
        # A variant phrase whose concept IS bound must not be re-queried.
        det = _det_result()
        det.det_map["plate"] = object()
        cfg = _Cfg()
        cfg._mp_phrase_to_concept = {"ceramic dish": "plate"}
        annotation_rescue(det, ["ceramic dish"], cfg, trial_meta={})
        assert wired.queries == []

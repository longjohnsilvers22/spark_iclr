"""
The annotation-rescue rung of the real detect escalation ladder. Offline.

Rung ordering context (pipeline_perception.detect_for_task): relax the
secondary score -> alt_prompts -> learned prompts -> LLM proposal -> THIS
rung -> on_mismatch. Every earlier rung re-words the text query; this one
has a pointing provider put a pixel on the missing group and seeds SAM3's
click head with it. No network and no SAM3 here: the provider is a mock
and the click head is a scripted fake, so what is tested is control flow
-- the rung fires only when enabled and only after the prompt rungs, its
output carries the canonical group label, and a pointing miss degrades to
``on_mismatch`` instead of becoming a failure of its own.
"""

import numpy as np
import pytest
import torch

import spark_real.perception.annotations as annotations_mod
from spark_real.perception.annotations import Annotation
from spark_real.perception.prompt_registry import PromptCountMismatch
from spark_real.tests.test_llm_prompt_proposal import (
    TASK,
    ScriptedPipeline,
    _od,
    write_spec,
)

H, W = 16, 16


class _Cal:
    """Minimal calibration double: pinhole + camera 1 m above the table."""

    fx = fy = 100.0
    cx = W / 2.0
    cy = H / 2.0
    fovy_degrees = 90.0
    extrinsic = np.diag([1.0, 1.0, 1.0, 1.0]) + 0.0
    extrinsic[2, 3] = 1.0  # non-identity so _is_camera_calibrated passes
    rotation_matrix = np.eye(3)
    position = np.array([0.0, 0.0, 1.0])


class _FakeSAM3ClickHead:
    """Yields one 4x4 mask around whatever pixel was prompted."""

    def set_image(self, pil_img):
        return {}


class _FakePerception:
    def __init__(self):
        self._sam3 = _FakeSAM3ClickHead()

    def load_models(self, load_da3=True):
        pass

    def set_point_prompt(self, x, y, state, label=1):
        mask = np.zeros((H, W), dtype=np.uint8)
        u, v = int(x), int(y)
        mask[max(0, v - 2):v + 2, max(0, u - 2):u + 2] = 1
        state["masks"] = torch.tensor(mask)[None]
        state["scores"] = torch.tensor([0.88])
        return state


class _MockProvider:
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


class RescuePipeline(ScriptedPipeline):
    """ScriptedPipeline + the camera/click plumbing the rescue rung touches."""

    def __init__(self, script, prompts_dir, planner=None, rescue=True):
        super().__init__(script, prompts_dir, planner)
        self.config.annotation_rescue = rescue
        self.config.annotation_provider = "mock"
        self._kinect_cal = _Cal()
        self._kinect2_cal = _Cal()
        self._realsense_cal = None
        self._perception = _FakePerception()


def captures():
    rgb = np.zeros((H, W, 3), np.uint8)
    depth = np.full((H, W), 0.5, np.float32)
    return {"birdview": {"rgb": rgb, "depth": depth, "calibration": None}}


@pytest.fixture
def provider(monkeypatch):
    p = _MockProvider()
    monkeypatch.setattr(annotations_mod, "get_provider", lambda name, **kw: p)
    monkeypatch.delenv("SPARK_ANNOTATION_RESCUE", raising=False)
    return p


class TestRealAnnotationRescue:
    def test_rescue_passes_the_count_gate(self, tmp_path, provider):
        # 3 scripted text passes all miss (default, relaxed, alt prompts);
        # the point rescue must then satisfy the gate with NO 4th pass.
        p = RescuePipeline([[_od("plushie", 0.09)]] * 3, write_spec(tmp_path))
        merged, all_dets, res, _, _ = p.detect_for_task(captures(), TASK)
        assert provider.queries == [("plushie", "point")]
        assert res.ok, res.describe()
        assert len(p.calls) == 3, "the rescue must use the click head, not detect()"
        rescued = [d for d in all_dets if getattr(d, "rescued_by", None)]
        assert len(rescued) == 1
        assert rescued[0].label == "plushie", "canonical group label required"
        assert rescued[0].camera == "birdview"
        assert rescued[0].confidence == pytest.approx(0.88)
        assert "plushie" in {d.label for d in merged}

    def test_rung_off_by_default(self, tmp_path, provider):
        p = RescuePipeline([[_od("plushie", 0.09)]] * 3, write_spec(tmp_path),
                           rescue=False)
        with pytest.raises(PromptCountMismatch):
            p.detect_for_task(captures(), TASK)
        assert provider.queries == []

    def test_env_kill_switch_wins_over_config(self, tmp_path, provider,
                                              monkeypatch):
        monkeypatch.setenv("SPARK_ANNOTATION_RESCUE", "0")
        p = RescuePipeline([[_od("plushie", 0.09)]] * 3, write_spec(tmp_path))
        with pytest.raises(PromptCountMismatch):
            p.detect_for_task(captures(), TASK)
        assert provider.queries == []

    def test_env_enables_over_config_off(self, tmp_path, provider, monkeypatch):
        monkeypatch.setenv("SPARK_ANNOTATION_RESCUE", "mock")
        p = RescuePipeline([[_od("plushie", 0.09)]] * 3, write_spec(tmp_path),
                           rescue=False)
        _, _, res, _, _ = p.detect_for_task(captures(), TASK)
        assert res.ok
        assert provider.queries == [("plushie", "point")]

    def test_pointing_miss_degrades_to_on_mismatch(self, tmp_path, monkeypatch):
        p = _MockProvider(respond=False)
        monkeypatch.setattr(annotations_mod, "get_provider",
                            lambda name, **kw: p)
        monkeypatch.delenv("SPARK_ANNOTATION_RESCUE", raising=False)
        pipe = RescuePipeline([[_od("plushie", 0.09)]] * 3, write_spec(tmp_path))
        with pytest.raises(PromptCountMismatch):
            pipe.detect_for_task(captures(), TASK)
        # It DID try (once per calibrated camera with a frame), then aborted.
        assert p.queries, "provider should have been consulted before aborting"

    def test_rescue_not_consulted_when_text_path_works(self, tmp_path, provider):
        p = RescuePipeline([[_od("plushie", 0.9)]], write_spec(tmp_path))
        _, _, res, _, _ = p.detect_for_task(captures(), TASK)
        assert res.ok
        assert provider.queries == []

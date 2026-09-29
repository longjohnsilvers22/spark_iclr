"""The two condition leaves that make a recovery branch reachable.

``verify_grasp`` and ``verify_placed`` are the only nodes in the grammar that
report a fact rather than move the arm; a fallback branch is dead code without
them. These tests drive them through the REAL SkillRegistry dispatch path with
a fake pipeline, so a rename or a registration slip fails here rather than on
the robot.

The abstention rule is asserted explicitly: no evidence must read as success.
On a real arm, re-running a place because no camera could see the result drives
the gripper back into a scene we have no model of.
"""

from __future__ import annotations

import numpy as np

from spark_real.skills import registry as skill_registry


class _Det:
    """Duck-typed stand-in for ObjectDetection."""

    def __init__(self, label, xyz, confidence=0.9, obb_minor_m=0.0, aspect_ratio=1.0):
        self.label = label
        self.position_3d = np.asarray(xyz, dtype=float)
        self.confidence = confidence
        self.obb_minor_m = obb_minor_m
        self.aspect_ratio = aspect_ratio
        self.world_major_axis_rad = 0.0
        self.mask = None
        self.slots = []


class _FakePipeline:
    def __init__(self, detections):
        self._detections = detections
        self.prompts_seen = []

    def capture(self):
        return {"birdview": {"rgb": np.zeros((8, 8, 3), dtype=np.uint8)}}

    def detect(self, captures, prompts=None):
        self.prompts_seen.append(list(prompts or []))
        return list(self._detections)

    def merge_detections(self, dets):
        return dets


class _FakeExecutor:
    def __init__(self, pipeline=None, holding=False, verify=None):
        self._pipeline = pipeline
        self._holding = holding
        self._verify = verify

    def _verify_grasp(self):
        if self._verify is None:
            raise RuntimeError("no gripper")
        return self._verify


BOWL = _Det("blue bowl", [0.45, -0.15, 0.05], obb_minor_m=0.16)


def _dispatch(name, executor, params):
    return skill_registry.dispatch(name, executor, params)


# verify_grasp


def test_verify_grasp_is_registered():
    assert skill_registry.get("verify_grasp") is not None


def test_verify_grasp_reports_the_gripper_state():
    assert _dispatch("verify_grasp", _FakeExecutor(verify=True), {}).success is True
    assert _dispatch("verify_grasp", _FakeExecutor(verify=False), {}).success is False


def test_verify_grasp_abstains_when_the_gripper_cannot_be_read():
    result = _dispatch("verify_grasp", _FakeExecutor(verify=None), {})
    assert result.success is True
    assert "unavailable" in result.message


# verify_placed


def test_verify_placed_is_registered():
    assert skill_registry.get("verify_placed") is not None


def test_verify_placed_passes_when_the_object_is_in_the_container():
    obj = _Det("stuffed animal", [0.45, -0.15, 0.06])
    exe = _FakeExecutor(pipeline=_FakePipeline([obj, BOWL]))
    result = _dispatch(
        "verify_placed",
        exe,
        {"obj": "stuffed animal", "container": "blue bowl"},
    )
    assert result.success is True
    assert "confirmed" in result.message


def test_verify_placed_fails_when_the_object_landed_outside():
    """The exact hardware failure: the plushie dropped beside the bowl."""
    obj = _Det("stuffed animal", [0.45, 0.10, 0.05])
    exe = _FakeExecutor(pipeline=_FakePipeline([obj, BOWL]))
    result = _dispatch(
        "verify_placed",
        exe,
        {"obj": "stuffed animal", "container": "blue bowl"},
    )
    assert result.success is False
    assert "FALSE" in result.message


def test_verify_placed_abstains_without_a_pipeline():
    result = _dispatch(
        "verify_placed",
        _FakeExecutor(pipeline=None),
        {"obj": "a", "container": "b"},
    )
    assert result.success is True
    assert "abstain" in result.message


def test_verify_placed_abstains_when_the_object_is_not_detected():
    exe = _FakeExecutor(pipeline=_FakePipeline([BOWL]))
    result = _dispatch(
        "verify_placed",
        exe,
        {"obj": "stuffed animal", "container": "blue bowl"},
    )
    assert result.success is True


def test_verify_placed_abstains_when_the_container_extent_is_unknown():
    """No measured width means no OBB; that is ignorance, not a miss."""
    bowl = _Det("blue bowl", [0.45, -0.15, 0.05])  # no obb_minor_m
    obj = _Det("stuffed animal", [0.45, 0.30, 0.05])
    exe = _FakeExecutor(pipeline=_FakePipeline([obj, bowl]))
    result = _dispatch(
        "verify_placed",
        exe,
        {"obj": "stuffed animal", "container": "blue bowl"},
    )
    assert result.success is True


def test_verify_placed_survives_a_raising_pipeline():
    class _Boom:
        def capture(self):
            raise RuntimeError("camera unplugged")

    result = _dispatch(
        "verify_placed",
        _FakeExecutor(pipeline=_Boom()),
        {"obj": "a", "container": "b"},
    )
    assert result.success is True
    assert "abstain" in result.message


def test_verify_placed_strips_the_instance_suffix_for_the_sam3_prompt():
    obj = _Det("fork 2", [0.45, -0.15, 0.06])
    pipeline = _FakePipeline([obj, BOWL])
    exe = _FakeExecutor(pipeline=pipeline)
    result = _dispatch(
        "verify_placed", exe, {"obj": "fork 2", "container": "blue bowl"}
    )
    assert pipeline.prompts_seen == [["fork", "blue bowl"]]
    assert result.success is True

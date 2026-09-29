"""``verify_placed`` is a condition LEAF, so its cost is part of its contract.

The rig, 2026-08-18: the leaf took a fresh capture off every camera, ran a SAM3
pass per label per camera with the fusion gate's reprompts on top, merged
across cameras, and 54 seconds later reported ``abstain [none] 'plushie'
unbound`` -- because the merge had deduped the plushie against the bowl it was
sitting in. The primitive budget is 8 seconds, so the watchdog aborted the run
at 10:49:15, fourteen seconds before the operator could.

These tests count what the leaf ASKS FOR and assert what it still catches.
"""

from __future__ import annotations

import time
import types

import numpy as np

from spark_real.bt_label_resolver import LabelResolvingDetectionMap
from spark_real.skills import registry as skill_registry

BOWL_XYZ = (-0.922, -0.023, -0.227)
BOWL_MINOR = 0.16
IN_BOWL = (BOWL_XYZ[0] + 0.02, BOWL_XYZ[1] + 0.01, BOWL_XYZ[2] + 0.02)
BESIDE_BOWL = (BOWL_XYZ[0] + 0.35, BOWL_XYZ[1], BOWL_XYZ[2] + 0.02)


def det(label, camera, xyz, conf=0.9, **kw):
    return types.SimpleNamespace(
        label=label,
        camera=camera,
        confidence=conf,
        position_3d=np.asarray(xyz, dtype=float),
        mask=kw.pop("mask", None),
        obb_minor_m=kw.pop("obb_minor_m", 0.0),
        aspect_ratio=kw.pop("aspect_ratio", 1.0),
        world_major_axis_rad=0.0,
        slots=None,
        **kw,
    )


class CountingPipeline:
    def __init__(self, dets, cameras=("birdview", "sideview"), scope=None):
        self._dets = list(dets)
        self._cameras = cameras
        self.profile = types.SimpleNamespace(raw={"verification": {"scope": scope or {}}})
        self.captures = 0
        self.prompt_sets = []
        self.detect_cameras = []
        self.merges = 0
        self._last_captures = None
        self._last_captures_t = None

    def capture(self):
        self.captures += 1
        out = {
            cam: {
                "rgb": np.zeros((4, 4, 3), np.uint8),
                "depth": np.ones((4, 4), np.float32),
                "calibration": None,
            }
            for cam in self._cameras
        }
        self._last_captures = out
        self._last_captures_t = time.time()
        return out

    def detect(self, captures, prompts=None, **kw):
        self.prompt_sets.append(list(prompts or []))
        self.detect_cameras.append(sorted(captures))
        return [d for d in self._dets if d.camera in captures]

    def merge_detections(self, dets, **kw):
        self.merges += 1
        return list(dets)

    @property
    def inferences(self):
        return sum(len(p) * max(len(c), 1) for p, c in zip(self.prompt_sets, self.detect_cameras))


class Exec:
    def __init__(self, pipeline, plan=None, holding=False, grip=None):
        self._pipeline = pipeline
        self.detection_map = plan if plan is not None else _plan()
        self._holding = holding
        self._grip = grip

    def _verify_grasp(self):
        if self._grip is None:
            raise RuntimeError("no gripper on this fake")
        return self._grip


def _plan():
    return LabelResolvingDetectionMap(
        {
            "bowl": {
                "position_3d": list(BOWL_XYZ),
                "obb_minor_m": BOWL_MINOR,
                "aspect_ratio": 1.0,
                "world_major_axis_rad": 0.0,
                "confidence": 0.93,
            },
            "plushie": {"position_3d": [-0.90, 0.30, -0.278], "confidence": 0.88},
        }
    )


def scene(obj_xyz=IN_BOWL, cameras=("birdview", "sideview"), with_bowl=True):
    out = []
    for cam in cameras:
        out.append(det("plushie", cam, obj_xyz))
        if with_bowl:
            out.append(det("bowl", cam, BOWL_XYZ, obb_minor_m=BOWL_MINOR))
    return out


def run(exe):
    return skill_registry.dispatch(
        "verify_placed", exe, {"obj": "plushie", "container": "bowl"}
    )


# --- rung 0: telemetry -----------------------------------------------------


def test_jaws_that_never_let_go_fail_the_place_with_no_perception_at_all():
    pipe = CountingPipeline(scene())
    result = run(Exec(pipe, grip=True))
    assert result.success is False
    assert "still reads HOLDING" in result.message
    assert pipe.captures == 0 and pipe.inferences == 0


def test_the_telemetry_rung_is_reversible():
    pipe = CountingPipeline(scene(), scope={"telemetry_short_circuit": False})
    result = run(Exec(pipe, grip=True))
    assert result.success is True  # the object IS in the bowl; vision says so
    assert pipe.inferences > 0


def test_open_jaws_are_never_taken_as_proof_the_object_stayed():
    """The converse of the telemetry rung is not claimed. Vision still runs."""
    pipe = CountingPipeline(scene())
    run(Exec(pipe, grip=False))
    assert pipe.inferences > 0


# --- the regression: a place into a container is verifiable -----------------


def test_a_placed_object_is_confirmed_without_a_cross_camera_merge():
    pipe = CountingPipeline(scene())
    result = run(Exec(pipe))
    assert result.success is True
    assert "confirmed" in result.message
    assert pipe.merges == 0, "the merge is where the placed object got deduped"


def test_an_object_beside_the_container_still_fails():
    pipe = CountingPipeline(scene(obj_xyz=BESIDE_BOWL))
    result = run(Exec(pipe))
    assert result.success is False
    assert "is FALSE" in result.message


def test_nothing_detected_still_abstains_rather_than_failing():
    pipe = CountingPipeline([])
    result = run(Exec(pipe))
    assert result.success is True
    assert "abstain" in result.message


# --- label scoping ---------------------------------------------------------


def test_the_container_prompt_is_not_issued_when_the_plan_already_has_its_pose():
    pipe = CountingPipeline(scene(with_bowl=False))
    result = run(Exec(pipe))
    assert pipe.prompt_sets == [["plushie"]]
    assert pipe.inferences == 2, "one prompt over two cameras, was 2 prompts + reprompts"
    assert result.success is True and "confirmed" in result.message


def test_an_unknown_container_is_still_re_detected():
    """No plan-time pose to anchor on: the prompt has to be issued."""
    plan = LabelResolvingDetectionMap({"plushie": {"position_3d": [-0.9, 0.3, -0.278]}})
    pipe = CountingPipeline(scene())
    run(Exec(pipe, plan=plan))
    assert pipe.prompt_sets == [["plushie", "bowl"]]


def test_the_leaf_can_be_told_to_keep_re_detecting_the_container():
    pipe = CountingPipeline(scene(), scope={"condition_anchor_references": False})
    run(Exec(pipe))
    assert pipe.prompt_sets == [["plushie", "bowl"]]


# --- capture reuse ---------------------------------------------------------


def test_fresh_frames_are_taken_by_default():
    """The leaf runs right after a release; reuse would risk a pre-release frame."""
    pipe = CountingPipeline(scene(with_bowl=False))
    pipe.capture()
    run(Exec(pipe))
    assert pipe.captures == 2


def test_opting_in_reuses_a_recent_capture():
    pipe = CountingPipeline(scene(with_bowl=False), scope={"reuse_capture_s": 2.0})
    pipe.capture()
    run(Exec(pipe))
    assert pipe.captures == 1


# --- the escape hatch ------------------------------------------------------


def test_scope_off_restores_the_merged_path():
    pipe = CountingPipeline(scene(), scope={"enabled": False})
    result = run(Exec(pipe))
    assert pipe.merges == 1
    assert pipe.prompt_sets == [["plushie", "bowl"]]
    assert result.success is True


def test_a_raising_pipeline_abstains_rather_than_failing_the_place():
    class Boom(CountingPipeline):
        def detect(self, *a, **kw):
            raise RuntimeError("SAM3 fell over")

    result = run(Exec(Boom(scene())))
    assert result.success is True
    assert "abstain" in result.message

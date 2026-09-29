"""verify_scope wired into the task-level SuccessVerifier.

Counts what the verifier ASKS FOR, not just what it concludes: one prompt per
camera is one SAM3 inference, and on the rig each one costs ~8 seconds.
"""

import types

import numpy as np

from spark_real.bt_label_resolver import LabelResolvingDetectionMap
from spark_real.control import success_verifier as sv
from spark_real.control.success_predicates import PASS
from spark_real.tests.test_container_reference_frame import (
    BOWL_MINOR,
    BOWL_XYZ,
    PLUSHIE_IN_BOWL,
    PLACE_ACTIONS,
    det,
)


class Cal:
    """Enough CameraCalibration for the frustum test, looking straight down."""

    def __init__(self, position=(-0.92, -0.02, 1.5)):
        self.width, self.height = 640, 480
        self.intrinsic_matrix = np.array(
            [[500.0, 0.0, 320.0], [0.0, 500.0, 240.0], [0.0, 0.0, 1.0]]
        )
        self.extrinsic = np.array(
            [
                [1.0, 0.0, 0.0, position[0]],
                [0.0, -1.0, 0.0, position[1]],
                [0.0, 0.0, -1.0, position[2]],
                [0.0, 0.0, 0.0, 1.0],
            ]
        )


class CountingPipeline:
    """Records every capture and every prompt-set the verifier asks for."""

    def __init__(self, dets, cameras=("birdview", "sideview"), scope=None, cals=None):
        self._dets = list(dets)
        self._cameras = cameras
        self._cals = cals or {}
        self.config = types.SimpleNamespace(use_hardware_depth=True, output_dir=None)
        self.profile = types.SimpleNamespace(raw={"verification": {"scope": scope or {}}})
        self.captures = 0
        self.prompt_sets = []
        self.detect_kwargs = []
        self.detect_cameras = []
        self._last_captures = None
        self._last_captures_t = None

    def capture(self):
        self.captures += 1
        out = {
            cam: {
                "rgb": np.zeros((4, 4, 3), np.uint8),
                "depth": np.ones((4, 4), np.float32),
                "calibration": self._cals.get(cam),
            }
            for cam in self._cameras
        }
        self._last_captures = out
        self._last_captures_t = __import__("time").time()
        return out

    def detect(self, captures, prompts, **kw):
        self.prompt_sets.append(list(prompts))
        self.detect_kwargs.append(kw)
        self.detect_cameras.append(sorted(captures))
        return [d for d in self._dets if d.camera in captures]

    def merge_detections(self, *a, **kw):
        raise AssertionError("merge_detections must not run during verification")

    @property
    def inferences(self):
        """Prompts x cameras: what this pipeline was asked to run SAM3 for."""
        return sum(len(p) * len(c) for p, c in zip(self.prompt_sets, self.detect_cameras))


class Exec:
    def __init__(self, pipeline, plan, holding=False):
        self._pipeline = pipeline
        self.detection_map = plan
        self._results = []
        self._holding = holding
        self._last_place_label = "bowl"

    def _gripper_type(self):
        return "none"


def plan():
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


def scene(cameras=("birdview", "sideview")):
    """The plushie really is in the bowl, and both cameras see both."""
    out = []
    for cam in cameras:
        out.append(det("plushie", cam, PLUSHIE_IN_BOWL))
        out.append(
            det("bowl", cam, BOWL_XYZ, obb_minor_m=BOWL_MINOR, aspect_ratio=1.0)
        )
    return out


def run(pipe):
    return sv.SuccessVerifier(executor=Exec(pipe, plan())).verify({}, PLACE_ACTIONS)


# --- the default shape -----------------------------------------------------


def test_the_task_verdict_still_re_detects_its_container_by_default():
    """Plan-time geometry is a reference frame, not a licence to skip seeing it."""
    pipe = CountingPipeline(scene())
    assert run(pipe).status == PASS
    assert pipe.prompt_sets == [["plushie", "bowl"]]
    assert pipe.inferences == 4


def test_the_fusion_gate_is_off_for_the_task_verdict():
    """3 of the logged run's 7 inferences were rejected ASPIRE reprompts."""
    pipe = CountingPipeline(scene())
    run(pipe)
    assert pipe.detect_kwargs[0]["use_fusion"] is False


def test_fusion_is_restorable():
    pipe = CountingPipeline(scene(), scope={"fusion": True})
    run(pipe)
    assert "use_fusion" not in pipe.detect_kwargs[0]


def test_scope_off_restores_the_old_call_shape():
    pipe = CountingPipeline(scene(), scope={"enabled": False})
    assert run(pipe).status == PASS
    assert pipe.prompt_sets == [["plushie", "bowl"]]
    assert "use_fusion" not in pipe.detect_kwargs[0]


# --- 3. label scoping ------------------------------------------------------


def test_anchoring_the_container_removes_its_prompt_from_every_camera():
    pipe = CountingPipeline(scene(), scope={"anchor_references": True})
    outcome = run(pipe)
    assert pipe.prompt_sets == [["plushie"]]
    assert pipe.inferences == 2, "one prompt over two cameras"
    assert outcome.status == PASS, outcome.reason


def test_the_anchored_container_is_still_the_reference_frame():
    """The bowl is never re-detected, yet inside() is still measured against it."""
    dets = [det("plushie", cam, PLUSHIE_IN_BOWL) for cam in ("birdview", "sideview")]
    pipe = CountingPipeline(dets, scope={"anchor_references": True})
    outcome = run(pipe)
    assert pipe.prompt_sets == [["plushie"]]
    assert outcome.status == PASS, outcome.reason
    # A 3D pass vote is what fuse_votes requires, and the bowl's extent is the
    # only way to get one -- so the plan-time pose really was the frame used.
    assert any(v.mode == "3d" and v.vote == PASS for v in outcome.votes)


def test_the_binder_serves_an_unprompted_reference_from_plan_time():
    binder = sv.IdentityBinder(
        plan(),
        [det("plushie", "birdview", PLUSHIE_IN_BOWL)],
        manipulated={"plushie"},
        plan_anchored={"bowl"},
        pose_only={"bowl"},
    )
    bowl = binder.lookup("bowl")
    assert bowl is not None
    assert bowl["plan_anchored"] is True
    assert bowl["mask"] is None, "there is no fresh mask; the 2D fallback must abstain"
    assert np.allclose(bowl["position_3d"], BOWL_XYZ)
    assert "prompt not issued" in binder.explain("bowl")


def test_without_pose_only_an_unprompted_reference_still_abstains():
    binder = sv.IdentityBinder(
        plan(),
        [det("plushie", "birdview", PLUSHIE_IN_BOWL)],
        manipulated={"plushie"},
        plan_anchored={"bowl"},
    )
    assert binder.lookup("bowl") is None
    assert "not detected in this camera" in binder.explain("bowl")


def test_an_object_beside_the_bowl_still_fails_with_the_container_anchored():
    """The cut must not cost the check the verifier exists for."""
    beside = (BOWL_XYZ[0] + 0.35, BOWL_XYZ[1], BOWL_XYZ[2] + 0.02)
    dets = [det("plushie", cam, beside) for cam in ("birdview", "sideview")]
    pipe = CountingPipeline(dets, scope={"anchor_references": True})
    assert run(pipe).status == sv.FAIL


def test_anchoring_is_refused_while_the_2d_fallback_needs_a_fresh_mask():
    pipe = CountingPipeline(
        scene(),
        scope={"anchor_references": True},
    )
    pipe.profile.raw["verification"]["fallback_2d_containment"] = 0.9
    run(pipe)
    assert pipe.prompt_sets == [["plushie", "bowl"]]


# --- 4. camera scoping -----------------------------------------------------


def test_a_camera_the_scene_does_not_project_into_is_not_detected_on():
    cals = {"birdview": Cal(), "elsewhere": Cal(position=(9.0, 9.0, 1.5))}
    pipe = CountingPipeline(
        scene(cameras=("birdview", "elsewhere")),
        cameras=("birdview", "elsewhere"),
        cals=cals,
    )
    outcome = run(pipe)
    assert pipe.detect_cameras == [["birdview"]]
    assert pipe.inferences == 2, "two prompts over the one camera that can see"
    assert outcome.status == PASS, outcome.reason


def test_a_camera_with_no_calibration_is_kept():
    pipe = CountingPipeline(scene())  # cals default to None -> untestable
    run(pipe)
    assert pipe.detect_cameras == [["birdview", "sideview"]]


# --- 2. capture reuse ------------------------------------------------------


def test_fresh_frames_are_taken_by_default_because_the_arm_just_moved():
    """Reuse is opt-in: a frame from before the jaws opened is a wrong answer."""
    pipe = CountingPipeline(scene())
    pipe.capture()
    assert run(pipe).status == PASS
    assert pipe.captures == 2


def test_opting_in_makes_a_second_verification_free():
    pipe = CountingPipeline(scene(), scope={"reuse_capture_s": 2.0})
    pipe.capture()
    ex = Exec(pipe, plan())
    assert sv.SuccessVerifier(executor=ex).verify({}, PLACE_ACTIONS).status == PASS
    first = pipe.inferences
    assert sv.SuccessVerifier(executor=ex).verify({}, PLACE_ACTIONS).status == PASS
    assert pipe.inferences == first, "the second verify re-ran SAM3 on the same frames"
    assert pipe.captures == 1

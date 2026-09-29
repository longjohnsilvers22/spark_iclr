"""The container of a containment predicate is a PLAN-TIME reference frame.

Three hardware runs on 2026-08-04 placed an object into a bowl, physically
succeeded, and were all scored a failure by the same mechanism:

    15:08 sideview: 'blue bowl' unbound -- nearest is 13.7cm from its plan-time position
    15:59 birdview: 'bowl'      unbound -- nearest is  8.5cm
    15:59 sideview: 'bowl'      unbound -- nearest is 17.3cm
    => VERIFY UNVERIFIED: no 3d pass vote (birdview=abstain, sideview=abstain)

``IdentityBinder._bind_static`` re-detects the container and demands it sit
near its plan-time position. The object we JUST PLACED occludes the container's
interior, so SAM3's mask fragments to a partial rim and the backprojected
centroid walks off -- further than the bowl's own radius, because the surviving
pixels also sample depth off the placed object, which slides the 3D point along
the camera ray. A successful place is exactly what breaks the test.

Two separate defects are covered here.

FIX 1 -- the container is not re-detected as a pose source. For the reference
argument of a containment predicate (``inside``/``on``/``removed_from``
container/surface, ``stacked`` base, ``near`` target) the binder returns the
PLAN-TIME detection: position, OBB, slots. The object still gets re-detected;
it is the only thing that moved.

FIX 2 -- a container whose margin-adjusted extent collapses must ABSTAIN. The
15:59 bowl came back half_a=6.9cm half_b=1.9cm; with inside_xy_margin_m=-0.02
the minor-axis half-extent is max(0.019 - 0.02, 0) = 0.0, so the test is
unsatisfiable for every possible object position. A fail vote on an
unsatisfiable test is a false negative by construction.

Neither fix may weaken the check this gate exists for: an object dropped
BESIDE the container must still FAIL. That case is asserted here too.
"""

import types

import numpy as np
import pytest

from spark_real.bt_label_resolver import LabelResolvingDetectionMap
from spark_real.control import success_verifier as sv
from spark_real.control.success_predicates import (
    ABSTAIN,
    FAIL,
    PASS,
    UNVERIFIED,
    EvalConfig,
    Predicate,
    SceneView,
    VerifySpec,
    displaced_labels,
    evaluate_predicate,
)

# ---------------------------------------------------------------- the scene

BOWL_XYZ = (-0.922, -0.023, -0.227)
# The bowl's plan-time footprint. 16cm across -> _static_tol widens to its own
# 8cm radius, which is what every one of the three measured drifts exceeded.
BOWL_MINOR = 0.16
PLUSHIE_IN_BOWL = (BOWL_XYZ[0] + 0.02, BOWL_XYZ[1] + 0.01, BOWL_XYZ[2] + 0.02)
PLUSHIE_BESIDE_BOWL = (BOWL_XYZ[0] + 0.35, BOWL_XYZ[1], BOWL_XYZ[2] + 0.02)

# The three drifts measured on the rig, in metres.
MEASURED_DRIFTS_M = (0.085, 0.137, 0.173)

PLACE_ACTIONS = [
    {"type": "move_to_keypoint", "params": {"keypoint_label": "plushie"}},
    {"type": "grasp", "params": {}},
    {"type": "move_to_keypoint", "params": {"keypoint_label": "bowl"}},
    {"type": "release", "params": {}},
]


def det(label, camera, xyz, conf=0.9, mask=None, **kw):
    return types.SimpleNamespace(
        label=label,
        camera=camera,
        confidence=conf,
        position_3d=None if xyz is None else np.asarray(xyz, dtype=float),
        mask=mask,
        obb_minor_m=kw.pop("obb_minor_m", 0.0),
        aspect_ratio=kw.pop("aspect_ratio", 1.0),
        world_major_axis_rad=kw.pop("world_major_axis_rad", 0.0),
        slots=kw.pop("slots", None),
        **kw,
    )


def plan_map(bowl_minor=BOWL_MINOR, bowl_ar=1.0):
    return LabelResolvingDetectionMap(
        {
            "bowl": {
                "position_3d": list(BOWL_XYZ),
                "obb_minor_m": bowl_minor,
                "aspect_ratio": bowl_ar,
                "world_major_axis_rad": 0.0,
                "confidence": 0.93,
            },
            "plushie": {"position_3d": [-0.90, 0.30, -0.278], "confidence": 0.88},
        }
    )


def shift(xyz, dx):
    return (xyz[0] + dx, xyz[1], xyz[2])


class FakePipeline:
    def __init__(self, dets, cameras=("birdview", "sideview")):
        self._dets = dets
        self._cameras = cameras
        self.config = types.SimpleNamespace(use_hardware_depth=True, output_dir=None)
        self.profile = None

    def capture(self):
        return {
            cam: {
                "rgb": np.zeros((4, 4, 3), np.uint8),
                "depth": np.ones((4, 4), np.float32),
                "calibration": None,
            }
            for cam in self._cameras
        }

    def detect(self, captures, prompts, **kw):
        return list(self._dets)

    def merge_detections(self, *a, **kw):
        raise AssertionError("merge_detections must not run during verification")


class FakeExecutor:
    def __init__(self, pipeline, detection_map, holding=False):
        self._pipeline = pipeline
        self.detection_map = detection_map
        self._results = []
        self._holding = holding
        self._last_place_label = "bowl"

    def _gripper_type(self):
        return "none"


def verify(dets, cameras=("birdview", "sideview"), plan=None, actions=None, score=None):
    pipe = FakePipeline(dets, cameras=cameras)
    ex = FakeExecutor(pipe, plan if plan is not None else plan_map())
    return sv.SuccessVerifier(executor=ex).verify(score or {}, actions or PLACE_ACTIONS)


def place_scene(drift_m, cameras=("birdview", "sideview"), obj_xyz=PLUSHIE_IN_BOWL):
    """The plushie really is in the bowl; the bowl's mask drifted by ``drift_m``."""
    out = []
    for cam in cameras:
        out.append(det("plushie", cam, obj_xyz))
        # The re-detected bowl: fragmented rim, centroid walked off, and the
        # surviving OBB is the 3.8cm sliver the rig actually reported.
        out.append(
            det(
                "bowl",
                cam,
                shift(BOWL_XYZ, drift_m),
                obb_minor_m=0.038,
                aspect_ratio=0.069 / 0.019,
            )
        )
    return out


# ------------------------------------------------- FIX 1: the three drifts


@pytest.mark.parametrize("drift", MEASURED_DRIFTS_M)
def test_measured_container_drift_still_resolves(drift, capsys):
    """Every drift measured on the rig must resolve, not abstain."""
    out = verify(place_scene(drift))
    with capsys.disabled():
        print(f"\n  bowl mask drifted {drift * 100:.1f}cm -> {out.status}: {out.reason}")
    assert out.status == PASS, out.reason
    unbound = [v for v in out.votes if "unbound" in v.detail]
    assert not unbound, [v.detail for v in unbound]


def test_all_three_drifts_together_reproduce_the_run():
    """birdview 8.5cm + sideview 17.3cm in one pass, as logged at 15:59."""
    dets = []
    for cam, drift in (("birdview", 0.085), ("sideview", 0.173)):
        dets += place_scene(drift, cameras=(cam,))
    out = verify(dets)
    assert out.status == PASS, out.reason
    assert "no 3d pass vote" not in out.reason


def test_the_container_reference_is_plan_time_not_re_detected():
    """The bound container carries the plan-time pose, and says so."""
    dets = place_scene(0.173, cameras=("birdview",))
    binder = sv.IdentityBinder(
        plan_map(), dets, manipulated={"plushie"}, plan_anchored={"bowl"}
    )
    bound = binder.lookup("bowl")
    assert bound is not None, binder.explain("bowl")
    assert np.allclose(np.asarray(bound["position_3d"], float), BOWL_XYZ)
    assert "plan-time" in binder.explain("bowl")


def test_a_disagreeing_redetection_is_reported_but_does_not_rebind():
    """The drift is logged so the trace shows it; it is not the reference."""
    dets = place_scene(0.173, cameras=("birdview",))
    binder = sv.IdentityBinder(
        plan_map(), dets, manipulated={"plushie"}, plan_anchored={"bowl"}
    )
    assert binder.lookup("bowl") is not None
    assert "17.3cm" in binder.explain("bowl")


def test_a_container_absent_from_this_camera_still_abstains():
    """Plan-time geometry is a reference frame, not a licence to skip seeing it."""
    dets = [det("plushie", "birdview", PLUSHIE_IN_BOWL)]
    out = verify(dets, cameras=("birdview",))
    assert out.status == UNVERIFIED, out.reason
    assert any("unbound" in v.detail for v in out.votes)


# ------------------------------------------------------- the protective case


def test_object_dropped_beside_the_container_still_fails():
    """THE check this gate exists for. 35cm outside must stay a FAIL."""
    dets = place_scene(0.173, obj_xyz=PLUSHIE_BESIDE_BOWL)
    out = verify(dets)
    assert out.status == FAIL, out.reason


def test_object_just_outside_the_rim_still_fails():
    """Not merely the gross case: 12cm out of an 8cm-radius bowl fails too."""
    dets = place_scene(0.085, obj_xyz=shift(BOWL_XYZ, 0.12))
    out = verify(dets)
    assert out.status == FAIL, out.reason


# ------------------------------------------- FIX 2: the degenerate sliver OBB


SLIVER = det(
    "bowl",
    "sideview",
    BOWL_XYZ,
    obb_minor_m=0.038,  # half_b = 1.9cm
    aspect_ratio=0.069 / 0.019,  # half_a = 6.9cm
)


def _lookup(mapping):
    return lambda label: mapping.get(label)


def test_sliver_obb_abstains_instead_of_voting_fail():
    """half_b=1.9cm with margin -2cm -> the minor-axis test is unsatisfiable."""
    obj = det("plushie", "sideview", PLUSHIE_IN_BOWL)
    view = SceneView(camera="sideview", lookup=_lookup({"plushie": obj, "bowl": SLIVER}))
    pred = Predicate(
        "inside",
        {"obj": "plushie", "container": "bowl", "xy_margin_m": -0.02, "z_tol_m": 0.06},
    )
    vote = evaluate_predicate(pred, view, EvalConfig())
    assert vote.vote == ABSTAIN, f"{vote.vote}: {vote.detail}"
    assert vote.mode == "none"


def test_sliver_obb_cannot_hand_removed_from_a_free_pass():
    """removed_from inverts the test: a collapsed extent would PASS on nothing."""
    obj = det("plushie", "sideview", PLUSHIE_IN_BOWL)
    view = SceneView(camera="sideview", lookup=_lookup({"plushie": obj, "bowl": SLIVER}))
    pred = Predicate("removed_from", {"obj": "plushie", "container": "bowl"})
    vote = evaluate_predicate(pred, view, EvalConfig())
    assert vote.vote == ABSTAIN, f"{vote.vote}: {vote.detail}"


def test_a_real_container_still_votes():
    """The abstain must not swallow containers that carry real information."""
    cont = det("tray", "birdview", BOWL_XYZ, obb_minor_m=0.20, aspect_ratio=1.5)
    obj = det("fork", "birdview", shift(BOWL_XYZ, 0.30))
    view = SceneView(camera="birdview", lookup=_lookup({"fork": obj, "tray": cont}))
    pred = Predicate(
        "inside", {"obj": "fork", "container": "tray", "xy_margin_m": -0.02, "z_tol_m": 0.06}
    )
    assert evaluate_predicate(pred, view, EvalConfig()).vote == FAIL


# ------------------------------------------ a container the tree DID manipulate


DRAWER_PLAN = LabelResolvingDetectionMap(
    {
        "drawer": {
            "position_3d": [-0.90, 0.40, -0.10],
            "obb_minor_m": 0.24,
            "aspect_ratio": 1.2,
            "world_major_axis_rad": 0.0,
            "confidence": 0.9,
        },
        "block": {"position_3d": [-0.60, 0.00, -0.278], "confidence": 0.9},
    }
)
DRAWER_OPENED = (-0.90, 0.22, -0.10)  # pulled 18cm toward the robot

DRAWER_ACTIONS = [
    {"type": "move_to_keypoint", "params": {"keypoint_label": "drawer"}},
    {"type": "open_drawer", "params": {"keypoint_label": "drawer"}},
    {"type": "move_to_keypoint", "params": {"keypoint_label": "block"}},
    {"type": "grasp", "params": {}},
    {"type": "move_to_keypoint", "params": {"keypoint_label": "drawer"}},
    {"type": "release", "params": {}},
]


def _drawer_dets(cameras=("birdview", "sideview")):
    out = []
    for cam in cameras:
        out.append(
            det("drawer", cam, DRAWER_OPENED, obb_minor_m=0.24, aspect_ratio=1.2)
        )
        out.append(det("block", cam, (DRAWER_OPENED[0], DRAWER_OPENED[1], -0.08)))
    return out


def test_a_container_the_tree_moved_is_re_detected_not_plan_anchored():
    """open_drawer displaced it, so its plan-time pose is stale by construction."""
    out = verify(_drawer_dets(), plan=DRAWER_PLAN, actions=DRAWER_ACTIONS)
    assert out.status == PASS, out.reason
    assert not any("@plan" in v.detail for v in out.votes), [v.detail for v in out.votes]


def test_displaced_labels_covers_shoving_not_just_carrying():
    """The set that decides "may this pose be reused" must not miss a shove."""
    assert displaced_labels(DRAWER_ACTIONS) == ["drawer", "block"]
    assert displaced_labels([{"type": "push_object", "params": {"keypoint_label": "mug"}}]) == [
        "mug"
    ]
    # sweep moves its object_labels; its target_label is the container.
    swept = displaced_labels(
        [{"type": "sweep", "params": {"object_labels": "crumb, chip", "target_label": "bin"}}]
    )
    assert swept == ["crumb", "chip"], swept
    assert "bin" not in swept


def test_a_label_that_is_also_an_object_under_test_is_never_anchored():
    """If any predicate tests it as ``obj``, it moved, so it must be re-detected."""
    spec = VerifySpec(
        predicates=[
            Predicate("inside", {"obj": "bowl", "container": "tray"}),
            Predicate("on", {"obj": "plushie", "surface": "bowl"}),
        ]
    )
    plan = LabelResolvingDetectionMap(
        {
            "bowl": {"position_3d": list(BOWL_XYZ), "obb_minor_m": 0.16},
            "tray": {"position_3d": [-0.5, 0.0, -0.24], "obb_minor_m": 0.30},
            "plushie": {"position_3d": [-0.9, 0.3, -0.28]},
        }
    )
    keys = sv.SuccessVerifier._plan_anchored_keys(spec, plan, {"plushie"})
    assert keys == {"tray"}, keys


def test_a_moved_container_judged_at_its_new_pose_can_still_fail():
    """Control: the block left at the drawer's OLD pose is not in the drawer."""
    dets = []
    for cam in ("birdview", "sideview"):
        dets.append(det("drawer", cam, DRAWER_OPENED, obb_minor_m=0.24, aspect_ratio=1.2))
        dets.append(det("block", cam, (-0.90, 0.40, -0.08)))
    out = verify(dets, plan=DRAWER_PLAN, actions=DRAWER_ACTIONS)
    assert out.status == FAIL, out.reason

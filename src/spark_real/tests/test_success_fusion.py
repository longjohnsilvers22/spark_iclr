"""Fusion, gates and fail-closed behaviour of the SuccessVerifier.

No robot, no camera, no SAM3: the pipeline and executor are fakes.
Everything here is about the decision, not the perception.
"""

import itertools
import types

import numpy as np
import pytest

from spark_real.bt_label_resolver import LabelResolvingDetectionMap
from spark_real.control.success_predicates import (
    ABSTAIN,
    FAIL,
    PASS,
    UNVERIFIED,
    CameraVote,
    EvalConfig,
    Predicate,
    SceneView,
    evaluate_predicate,
)
from spark_real.control import success_verifier as sv

TRAY_XY = (-0.90, 0.00)
TRAY_Z = -0.245
TABLE_Z = -0.278

ACTIONS = [
    {"type": "move_to_keypoint", "params": {"keypoint_label": "fork"}},
    {"type": "grasp", "params": {}},
    {"type": "move_to_keypoint", "params": {"keypoint_label": "tray"}},
    {"type": "release", "params": {}},
]


def make_det(label, camera, xyz, conf=0.8, mask=None, **kw):
    d = types.SimpleNamespace(
        label=label,
        camera=camera,
        confidence=conf,
        position_3d=None if xyz is None else np.asarray(xyz, dtype=float),
        mask=mask,
        obb_minor_m=kw.pop("obb_minor_m", 0.0),
        aspect_ratio=kw.pop("aspect_ratio", 1.0),
        world_major_axis_rad=kw.pop("world_major_axis_rad", None),
        slots=kw.pop("slots", None),
    )
    for k, v in kw.items():
        setattr(d, k, v)
    return d


def tray_det(camera, xyz=(TRAY_XY[0], TRAY_XY[1], TRAY_Z), mask=None):
    return make_det(
        "tray",
        camera,
        xyz,
        conf=0.9,
        mask=mask,
        obb_minor_m=0.20,
        aspect_ratio=1.5,
        world_major_axis_rad=0.0,
    )


class FakePipeline:
    def __init__(self, dets, cameras=("birdview", "sideview"), depth=True):
        self._dets = dets
        self._cameras = cameras
        self._depth = depth
        self.config = types.SimpleNamespace(use_hardware_depth=True, output_dir=None)
        self.profile = None
        self.capture_calls = 0
        self.detect_calls = 0

    def capture(self):
        self.capture_calls += 1
        return {
            cam: {
                "rgb": np.zeros((4, 4, 3), np.uint8),
                "depth": np.ones((4, 4), np.float32) if self._depth else None,
                "calibration": None,
            }
            for cam in self._cameras
        }

    def detect(self, captures, prompts, **kw):
        self.detect_calls += 1
        return list(self._dets)

    def merge_detections(self, *a, **kw):  # must never be called on the verify path
        raise AssertionError("merge_detections must not run during verification")


# Planning-time map the derivation consults to tell a container from a surface.
DET_MAP = {
    "tray": {
        "position_3d": [TRAY_XY[0], TRAY_XY[1], TRAY_Z],
        "obb_minor_m": 0.20,
        "aspect_ratio": 1.5,
        "world_major_axis_rad": 0.0,
    },
    "fork": {"position_3d": [-0.90, 0.30, TABLE_Z]},
}


class FakeExecutor:
    def __init__(self, pipeline, detection_map=None, holding=False):
        self._pipeline = pipeline
        self.detection_map = dict(DET_MAP) if detection_map is None else detection_map
        self._results = []
        self._holding = holding
        self._last_place_label = "tray"

    def _gripper_type(self):
        return "none"


def verifier(pipeline, config=None, **kw):
    ex = FakeExecutor(pipeline, **kw)
    return sv.SuccessVerifier(executor=ex, config=config), ex


# The 2D fallback ships disabled (see VerifyConfig); tests that exercise the
# mechanism have to ask for it explicitly.
CFG_2D_ON = sv.VerifyConfig(fallback_2d_containment=0.90)


# fusion table


def vote(camera, v, mode):
    return CameraVote(
        camera=camera, predicate="inside(fork, tray)", vote=v, mode=mode, confidence=0.8, detail=""
    )


STATES = [(PASS, "3d"), (PASS, "2d"), (FAIL, "3d"), (FAIL, "2d"), (ABSTAIN, "none")]


def oracle(states, require_two=False):
    if any(v == FAIL for v, _ in states):
        return FAIL
    passes = [m for v, m in states if v == PASS]
    if not any(m == "3d" for m in passes):
        return UNVERIFIED
    if require_two and len(passes) < 2:
        return UNVERIFIED
    return PASS


@pytest.mark.parametrize("n", [1, 2, 3])
@pytest.mark.parametrize("require_two", [False, True])
def test_fusion_table(n, require_two):
    cams = ["birdview", "sideview", "wrist"][:n]
    bad = []
    for combo in itertools.product(STATES, repeat=n):
        votes = [vote(c, v, m) for c, (v, m) in zip(cams, combo)]
        status, reason = sv.fuse_votes(votes, require_two_views=require_two)
        want = oracle(list(combo), require_two)
        if status != want:
            bad.append((combo, want, status, reason))
    assert not bad, bad


@pytest.mark.parametrize(
    "votes,expected",
    [
        ([(PASS, "3d")], PASS),
        ([(PASS, "2d")], UNVERIFIED),
        ([(PASS, "2d"), (PASS, "2d")], UNVERIFIED),
        ([(PASS, "3d"), (FAIL, "3d")], FAIL),
        ([(PASS, "3d"), (FAIL, "2d")], FAIL),
        ([(ABSTAIN, "none"), (ABSTAIN, "none")], UNVERIFIED),
        ([(PASS, "3d"), (ABSTAIN, "none")], PASS),
        ([(PASS, "2d"), (PASS, "3d")], PASS),
    ],
)
def test_fusion_critical_cases(votes, expected):
    cams = ["birdview", "sideview", "wrist"]
    got, _ = sv.fuse_votes([vote(cams[i], v, m) for i, (v, m) in enumerate(votes)])
    assert got == expected


def test_no_votes_is_unverified():
    assert sv.fuse_votes([])[0] == UNVERIFIED


def test_camera_ands_its_predicates():
    votes = [vote("birdview", PASS, "3d"), vote("birdview", ABSTAIN, "none")]
    assert sv.fuse_votes(votes)[0] == UNVERIFIED
    votes = [vote("birdview", PASS, "3d"), vote("birdview", FAIL, "3d")]
    assert sv.fuse_votes(votes)[0] == FAIL


# end-to-end through SuccessVerifier


def test_both_cameras_see_it_inside():
    dets = [
        make_det("fork", "birdview", (-0.90, 0.02, -0.240)),
        tray_det("birdview"),
        make_det("fork", "sideview", (-0.90, 0.01, -0.238)),
        tray_det("sideview"),
    ]
    v, _ = verifier(FakePipeline(dets))
    out = v.verify({}, ACTIONS)
    assert out.status == PASS
    assert out.predicates == ["inside(fork, tray)", "held(fork, False)"]
    assert out.depth_source == "hardware"


def test_camera_disagreement_vetoes():
    """birdview says inside, sideview says beside -> a single fail vetoes."""
    dets = [
        make_det("fork", "birdview", (-0.90, 0.02, -0.240)),
        tray_det("birdview"),
        make_det("fork", "sideview", (-0.90, 0.13, TABLE_Z)),
        tray_det("sideview"),
    ]
    v, _ = verifier(FakePipeline(dets))
    out = v.verify({}, ACTIONS)
    assert out.status == FAIL
    assert "sideview" in out.reason


def test_missing_camera_is_not_an_abstention():
    dets = [make_det("fork", "birdview", (-0.90, 0.02, -0.240)), tray_det("birdview")]
    v, _ = verifier(FakePipeline(dets, cameras=("birdview",)))
    assert v.verify({}, ACTIONS).status == PASS


def test_all_abstain_is_unverified_not_success():
    dets = [
        make_det("fork", "birdview", (-0.90, 0.02, -0.240), conf=0.10),
        tray_det("birdview"),
    ]
    v, _ = verifier(FakePipeline(dets))
    out = v.verify({}, ACTIONS)
    assert out.status == UNVERIFIED
    assert (out.status == PASS) is False


def test_2d_only_pass_never_reaches_pass():
    obj_mask = np.zeros((40, 40), bool)
    obj_mask[10:20, 10:20] = True
    cont_mask = np.zeros((40, 40), bool)
    cont_mask[5:30, 5:30] = True
    dets = [
        make_det("fork", "birdview", None, mask=obj_mask),
        tray_det("birdview", mask=cont_mask),
    ]
    v, ex = verifier(FakePipeline(dets, cameras=("birdview",)), config=CFG_2D_ON)
    out = v.verify({}, ACTIONS)
    assert [x.mode for x in out.votes if x.mode != "none"] == ["2d"]
    assert out.status == UNVERIFIED

    # ... and with the shipped config the same scene abstains, so a depth-less
    # camera can neither carry nor veto a verdict.
    v, _ = verifier(FakePipeline(dets, cameras=("birdview",)))
    out = v.verify({}, ACTIONS)
    assert [x.mode for x in out.votes] == ["none", "none"]
    assert out.status == UNVERIFIED


# gates run first and short-circuit


def test_transport_drop_fails_without_capturing():
    pipe = FakePipeline([])
    v, ex = verifier(pipe)
    sv.note_transport_drop(ex, "lift")
    out = v.verify({}, ACTIONS)
    assert out.status == FAIL
    assert pipe.capture_calls == 0 and pipe.detect_calls == 0
    assert out.gates["transport"] is False


def test_regrasp_clears_an_earlier_drop():
    pipe = FakePipeline(
        [
            make_det("fork", "birdview", (-0.90, 0.02, -0.240)),
            tray_det("birdview"),
        ],
        cameras=("birdview",),
    )
    v, ex = verifier(pipe)
    sv.note_transport_drop(ex, "lift")
    sv.record_grasp_verdict(ex, True, "gObj")
    assert v.verify({}, ACTIONS).status == PASS


def test_failed_grasp_fails_without_capturing():
    pipe = FakePipeline([])
    v, ex = verifier(pipe)
    sv.record_grasp_verdict(ex, False, "empty jaws")
    out = v.verify({}, ACTIONS)
    assert out.status == FAIL and pipe.capture_calls == 0


def test_release_witness_gate():
    """Object gone BEFORE the gripper opened -> dropped in transit -> fail."""
    pipe = FakePipeline([])
    v, ex = verifier(pipe)
    # `released` carries the verdict now: held_after alone cannot express it,
    # because an OPEN gripper reads "holding" on the closed-jaw predicate.
    ex._release_witness = sv.ReleaseWitness(
        held_before=False, held_after=False, released=True, tcp_xyz=(0, 0, 0)
    )
    out = v.verify({}, ACTIONS)
    assert out.status == FAIL and out.gates["release"] is False
    assert pipe.capture_calls == 0


def test_still_holding_after_release_fails_the_predicate():
    dets = [
        make_det("fork", "birdview", (-0.90, 0.02, -0.240)),
        tray_det("birdview"),
    ]
    pipe = FakePipeline(dets, cameras=("birdview",))
    v, ex = verifier(pipe)
    ex._release_witness = sv.ReleaseWitness(
        held_before=True, held_after=True, released=False, tcp_xyz=(0, 0, 0)
    )
    out = v.verify({}, ACTIONS)
    assert out.gates["release"] is False
    assert out.status == FAIL


def test_release_witness_records_container_region():
    det_map = {
        "tray": {
            "position_3d": [TRAY_XY[0], TRAY_XY[1], TRAY_Z],
            "obb_minor_m": 0.20,
            "aspect_ratio": 1.5,
            "world_major_axis_rad": 0.0,
        }
    }
    pipe = FakePipeline([])
    ex = FakeExecutor(pipe, detection_map=det_map, holding=True)
    ex._get_current_position = lambda: np.array([-0.90, 0.01, -0.14])
    state = sv.begin_release_witness(ex)
    assert state["held_before"] is True
    ex._holding = False
    w = sv.finish_release_witness(ex, state, container_label="tray")
    assert w.inside_region is True
    assert w.tcp_dz_to_container == pytest.approx(0.105)
    assert sv.collect_gates(ex)[0]["release"] is True


# fail-closed


def test_raising_detector_is_unverified_not_success():
    pipe = FakePipeline([])

    def boom(*a, **kw):
        raise RuntimeError("SAM3 exploded")

    pipe.detect = boom
    v, _ = verifier(pipe)
    out = v.verify({}, ACTIONS)
    assert out.status == UNVERIFIED
    assert "exception" in out.reason


def test_raising_capture_is_unverified():
    pipe = FakePipeline([])
    pipe.capture = lambda: (_ for _ in ()).throw(RuntimeError("camera gone"))
    v, _ = verifier(pipe)
    assert v.verify({}, ACTIONS).status == UNVERIFIED


def test_no_derivable_predicate_is_unverified():
    pipe = FakePipeline([])
    v, _ = verifier(pipe)
    out = v.verify({}, [{"type": "pour", "params": {"target_label": "cup"}}])
    assert out.status == UNVERIFIED
    assert pipe.capture_calls == 0


def test_no_camera_is_unverified():
    pipe = FakePipeline([], cameras=())
    v, _ = verifier(pipe)
    assert v.verify({}, ACTIONS).status == UNVERIFIED


# spec resolution


def test_planner_block_wins():
    dets = [
        make_det("fork", "birdview", (-0.90, 0.02, -0.240)),
        tray_det("birdview"),
    ]
    v, _ = verifier(FakePipeline(dets, cameras=("birdview",)))
    score = {
        "verify": {"all": [{"pred": "near", "obj": "fork", "target": "tray", "max_dist_m": 0.05}]}
    }
    out = v.verify(score, ACTIONS)
    assert out.predicates == ["near(fork, tray, 0.050m)"]
    assert out.status == PASS


def test_malformed_planner_block_falls_back_to_derivation():
    dets = [
        make_det("fork", "birdview", (-0.90, 0.02, -0.240)),
        tray_det("birdview"),
    ]
    v, _ = verifier(FakePipeline(dets, cameras=("birdview",)))
    score = {"verify": {"all": [{"pred": "teleported", "obj": "fork"}]}}
    out = v.verify(score, ACTIONS)
    assert out.predicates == ["inside(fork, tray)", "held(fork, False)"]


def test_unknown_label_in_planner_block_is_rejected():
    dets = [
        make_det("fork", "birdview", (-0.90, 0.02, -0.240)),
        tray_det("birdview"),
    ]
    det_map = LabelResolvingDetectionMap(DET_MAP)
    v, ex = verifier(FakePipeline(dets, cameras=("birdview",)), detection_map=det_map)
    score = {"verify": {"all": [{"pred": "inside", "obj": "ghost", "container": "tray"}]}}
    out = v.verify(score, ACTIONS)
    assert out.predicates == ["inside(fork, tray)", "held(fork, False)"]


def test_label_instance_binds_to_bare_detection():
    """'fork 1' in the predicate binds to a re-detection labelled 'fork'."""
    dets = [
        make_det("fork", "birdview", (-0.90, 0.02, -0.240)),
        tray_det("birdview"),
    ]
    v, _ = verifier(FakePipeline(dets, cameras=("birdview",)))
    score = {"verify": {"all": [{"pred": "inside", "obj": "fork 1", "container": "tray"}]}}
    assert v.verify(score, ACTIONS).status == PASS


def test_require_two_views_downgrades_single_camera_pass():
    dets = [make_det("fork", "birdview", (-0.90, 0.02, -0.240)), tray_det("birdview")]
    v, _ = verifier(FakePipeline(dets, cameras=("birdview",)))
    score = {
        "verify": {
            "all": [{"pred": "inside", "obj": "fork", "container": "tray"}],
            "require_two_views": True,
        }
    }
    assert v.verify(score, ACTIONS).status == UNVERIFIED


def test_occlusion_ok_cannot_manufacture_a_pass_from_no_vision():
    """The bounce-out failure: jaws opened over the tray, object landed elsewhere.

    Vision abstains everywhere -- which is what an opaque bin looks like AND
    what an object that bounced out to somewhere the cameras did not resolve
    looks like. Proprioception cannot tell those apart, so it must not pass.
    """
    pipe = FakePipeline([tray_det("birdview")], cameras=("birdview",))
    det_map = {
        "tray": {
            "position_3d": [TRAY_XY[0], TRAY_XY[1], TRAY_Z],
            "obb_minor_m": 0.20,
            "aspect_ratio": 1.5,
            "world_major_axis_rad": 0.0,
        }
    }
    v, ex = verifier(pipe, detection_map=det_map)
    ex._release_witness = sv.ReleaseWitness(
        held_before=True,
        held_after=False,
        tcp_xyz=(-0.90, 0.01, -0.20),
        container_label="tray",
        inside_region=True,
        tcp_dz_to_container=0.045,
    )
    score = {
        "verify": {
            "all": [{"pred": "inside", "obj": "fork", "container": "tray", "occlusion_ok": True}]
        }
    }
    out = v.verify(score, ACTIONS)
    assert out.status == UNVERIFIED
    assert out.status != PASS
    # the witness is still reported -- it is evidence, just not proof
    assert "occlusion_ok" in out.reason and "cannot prove" in out.reason
    assert [x.vote for x in out.votes] == [ABSTAIN]

    # ... and every gate green does not change that.
    assert out.gates == {"grasp": True, "transport": True, "release": True}

    # same scene without the flag -> also unverified, and with no annotation
    score["verify"]["all"][0]["occlusion_ok"] = False
    out = v.verify(score, ACTIONS)
    assert out.status == UNVERIFIED and "occlusion_ok" not in out.reason


def test_occlusion_ok_cannot_override_a_fail():
    dets = [
        make_det("fork", "birdview", (-0.90, 0.13, TABLE_Z)),
        tray_det("birdview"),
    ]
    pipe = FakePipeline(dets, cameras=("birdview",))
    v, ex = verifier(pipe)
    ex._release_witness = sv.ReleaseWitness(
        held_before=True,
        held_after=False,
        tcp_xyz=(-0.90, 0.01, -0.20),
        container_label="tray",
        inside_region=True,
        tcp_dz_to_container=0.045,
    )
    score = {
        "verify": {
            "all": [{"pred": "inside", "obj": "fork", "container": "tray", "occlusion_ok": True}]
        }
    }
    assert v.verify(score, ACTIONS).status == FAIL


def test_da3_depth_is_reported():
    dets = [make_det("fork", "birdview", (-0.90, 0.02, -0.240)), tray_det("birdview")]
    v, _ = verifier(FakePipeline(dets, cameras=("birdview",), depth=False))
    assert v.verify({}, ACTIONS).depth_source == "da3"


# -------------------------------------------------------------------------
# multi-instance identity: pick_up_the_silverware
#
# Instance numbers are assigned per detection pass and re-sort the moment an
# object moves, so a predicate written about "fork 1" at plan time can bind to
# a DIFFERENT physical fork at verify time. Scene below: three forks, one
# already sitting in the tray, the manipulated one dropped 35cm outside the
# rim. Binding by number scores that as success; binding by geometry fails it.
# -------------------------------------------------------------------------

# tray is 0.30 x 0.20 centred at (-0.90, 0.00): x in [-1.05,-0.75], y in [-0.10, 0.10]
IN_TRAY_LEFT = (-1.00, 0.00, -0.240)  # a fork that was already in the tray
IN_TRAY_RIGHT = (-0.80, 0.01, -0.240)  # where a successful place lands
ON_TABLE_FAR = (-1.20, 0.30, TABLE_Z)  # an untouched fork on the table
BESIDE_TRAY = (-0.90, 0.45, TABLE_Z)  # 35cm beyond the tray rim -- the failure
PICK_FROM = (-0.90, 0.30, TABLE_Z)  # where the manipulated fork started

SILVER_PLAN = {
    "fork 1": {"position_3d": list(PICK_FROM)},  # the one the tree picks up
    "fork 2": {"position_3d": list(IN_TRAY_LEFT)},  # distractor, already in tray
    "fork 3": {"position_3d": list(ON_TABLE_FAR)},  # distractor, on the table
    "tray": {
        "position_3d": [TRAY_XY[0], TRAY_XY[1], TRAY_Z],
        "obb_minor_m": 0.20,
        "aspect_ratio": 1.5,
        "world_major_axis_rad": 0.0,
    },
}

SILVER_ACTIONS = [
    {"type": "move_to_keypoint", "params": {"keypoint_label": "fork 1"}},
    {"type": "grasp", "params": {}},
    {"type": "move_to_keypoint", "params": {"keypoint_label": "tray"}},
    {"type": "release", "params": {}},
]

INSIDE_FORK1 = Predicate(
    "inside", {"obj": "fork 1", "container": "tray", "xy_margin_m": -0.02, "z_tol_m": 0.06}
)


def silverware_dets(positions, names=("fork 1", "fork 2", "fork 3"), cameras=("birdview", "sideview")):
    """One detection per (name, position) pair, in every camera."""
    out = []
    for cam in cameras:
        for name, xyz in zip(names, positions):
            out.append(make_det(name, cam, xyz))
        out.append(tray_det(cam))
    return out


def legacy_fuse(dets, predicates, held=False):
    """The PRE-FIX binding, reproduced verbatim: one map per camera keyed by
    detection label, most-confident-wins, and a predicate label looked up by
    name. This is what produced the false pass."""
    per_cam = {}
    for d in dets:
        m = per_cam.setdefault(d.camera, LabelResolvingDetectionMap())
        prev = dict.get(m, d.label)
        if prev is None or d.confidence > prev.confidence:
            m[d.label] = d
    votes = []
    for cam, m in per_cam.items():
        view = SceneView(camera=cam, lookup=m.get, held=held)
        votes.extend(evaluate_predicate(p, view, EvalConfig()) for p in predicates)
    return sv.fuse_votes(votes)[0]


def silverware_verdict(positions, names=("fork 1", "fork 2", "fork 3"), **kw):
    dets = silverware_dets(positions, names=names, **kw)
    v, _ = verifier(
        FakePipeline(dets, cameras=kw.get("cameras", ("birdview", "sideview"))),
        detection_map=LabelResolvingDetectionMap(SILVER_PLAN),
    )
    return v.verify({}, SILVER_ACTIONS)


def test_silverware_false_pass_is_the_blocker(capsys):
    """THE bug: predicate about the fork 35cm outside; a fork sits in the tray.

    SAM3 renumbers by position, so the verify-time "fork 1" is the one in the
    tray -- a different physical object than the tree manipulated.
    """
    positions = [IN_TRAY_LEFT, BESIDE_TRAY, ON_TABLE_FAR]
    dets = silverware_dets(positions)

    legacy = legacy_fuse(dets, [INSIDE_FORK1])
    out = silverware_verdict(positions)

    with capsys.disabled():
        print("\n  pick_up_the_silverware, predicate inside(fork 1, tray)")
        print(f"    plan-time  fork 1 @ {PICK_FROM} (picked)")
        print(f"    verify     'fork 1' @ {IN_TRAY_LEFT} (a DIFFERENT fork, in the tray)")
        print(f"               manipulated fork @ {BESIDE_TRAY} (35cm beyond the rim)")
        print(f"    pre-fix  (bind by instance number) -> {legacy}")
        print(f"    post-fix (bind by geometry)        -> {out.status}: {out.reason}")

    assert legacy == PASS, "pre-fix contrast: the old binding really did pass this"
    assert out.status != PASS
    assert out.status == FAIL


@pytest.mark.parametrize("perm", list(itertools.permutations(range(3))))
def test_instance_number_shuffle_never_flips_to_pass(perm):
    """Permute the instance numbers over the same three physical objects."""
    positions = [IN_TRAY_LEFT, BESIDE_TRAY, ON_TABLE_FAR]
    names = tuple(f"fork {i + 1}" for i in perm)
    out = silverware_verdict(positions, names=names)
    assert out.status != PASS, (names, out.reason)
    assert out.status == FAIL


def test_silverware_true_success_still_passes():
    """Control: the manipulated fork really is in the tray -> pass.

    Without this the fix would just be "never pass", which rejects nothing.
    """
    out = silverware_verdict([IN_TRAY_LEFT, IN_TRAY_RIGHT, ON_TABLE_FAR])
    assert out.status == PASS


def test_ambiguous_binding_abstains_rather_than_guessing():
    """A distractor went missing, so elimination cannot single anyone out."""
    dets = [
        make_det("fork 1", "birdview", IN_TRAY_LEFT),
        make_det("fork 2", "birdview", BESIDE_TRAY),
        tray_det("birdview"),
    ]
    v, _ = verifier(
        FakePipeline(dets, cameras=("birdview",)),
        detection_map=LabelResolvingDetectionMap(SILVER_PLAN),
    )
    out = v.verify({}, SILVER_ACTIONS)
    assert out.status == UNVERIFIED
    assert "ambiguous" in " ".join(x.detail for x in out.votes)


def test_two_candidates_on_top_of_each_other_abstain():
    """Two forks 2cm apart: no geometric evidence which is which."""
    dets = [
        make_det("fork 1", "birdview", IN_TRAY_LEFT),
        make_det("fork 2", "birdview", (IN_TRAY_LEFT[0] + 0.02, IN_TRAY_LEFT[1], IN_TRAY_LEFT[2])),
        make_det("fork 3", "birdview", ON_TABLE_FAR),
        tray_det("birdview"),
    ]
    v, _ = verifier(
        FakePipeline(dets, cameras=("birdview",)),
        detection_map=LabelResolvingDetectionMap(SILVER_PLAN),
    )
    assert v.verify({}, SILVER_ACTIONS).status == UNVERIFIED


def test_static_container_binds_by_position_not_by_number():
    """Two trays: the predicate's tray is the one that did not move."""
    plan = dict(SILVER_PLAN)
    plan["tray 1"] = plan.pop("tray")
    plan["tray 2"] = {
        "position_3d": [-0.40, 0.00, TRAY_Z],
        "obb_minor_m": 0.20,
        "aspect_ratio": 1.5,
        "world_major_axis_rad": 0.0,
    }
    # verify-time numbering swapped: "tray 1" is now the far tray
    dets = [
        tray_det("birdview", xyz=(-0.40, 0.00, TRAY_Z)),
        tray_det("birdview"),
        make_det("fork 1", "birdview", IN_TRAY_RIGHT),
        make_det("fork 2", "birdview", IN_TRAY_LEFT),
        make_det("fork 3", "birdview", ON_TABLE_FAR),
    ]
    dets[0].label, dets[1].label = "tray 1", "tray 2"
    actions = [
        {"type": "move_to_keypoint", "params": {"keypoint_label": "fork 1"}},
        {"type": "grasp", "params": {}},
        {"type": "place_in_slot", "params": {"container_label": "tray 1"}},
    ]
    v, _ = verifier(
        FakePipeline(dets, cameras=("birdview",)),
        detection_map=LabelResolvingDetectionMap(plan),
    )
    out = v.verify({}, actions)
    # bound to the tray at (-0.90, 0), which does contain the placed fork
    assert out.status == PASS
    # and the binder says so by position, not by the (swapped) number
    binder = sv.IdentityBinder(LabelResolvingDetectionMap(plan), dets, {"fork 1"})
    assert binder.lookup("tray 1") is dets[1]  # the one now labelled "tray 2"
    assert "0.0cm from its plan-time position" in binder.explain("tray 1")


def test_identity_confusion_matrix(capsys):
    """Confusion matrix over the multi-instance cases, pre-fix vs post-fix."""
    cases = [
        # name, verify-time positions, expected post-fix status
        ("manipulated_35cm_outside", [IN_TRAY_LEFT, BESIDE_TRAY, ON_TABLE_FAR], FAIL),
        ("manipulated_placed_in_tray", [IN_TRAY_LEFT, IN_TRAY_RIGHT, ON_TABLE_FAR], PASS),
        ("manipulated_beside_rim", [IN_TRAY_LEFT, (-0.90, 0.13, TABLE_Z), ON_TABLE_FAR], FAIL),
        ("manipulated_never_left_pick_spot", [IN_TRAY_LEFT, PICK_FROM, ON_TABLE_FAR], FAIL),
    ]
    labels = [PASS, FAIL, UNVERIFIED]
    matrix = {e: {a: 0 for a in labels} for e in labels}
    rows, wrong = [], []
    for name, positions, expected in cases:
        for perm in itertools.permutations(range(3)):
            names = tuple(f"fork {i + 1}" for i in perm)
            got = silverware_verdict(positions, names=names).status
            legacy = legacy_fuse(silverware_dets(positions, names=names), [INSIDE_FORK1])
            matrix[expected][got] += 1
            rows.append((name, names, expected, got, legacy))
            if got != expected:
                wrong.append((name, names, expected, got))

    with capsys.disabled():
        print("\n  identity-binding confusion matrix (rows = expected, cols = actual)")
        print("           " + "".join(f"{a:>11}" for a in labels))
        for e in labels:
            print(f"  {e:>8} " + "".join(f"{matrix[e][a]:>11}" for a in labels))
        n = sum(sum(r.values()) for r in matrix.values())
        print(f"  cases={n} (4 scenes x 6 instance-number permutations) mismatches={len(wrong)}")
        print("\n  per-scene, over all 6 permutations of the instance numbers:")
        seen = set()
        for name, names, expected, got, legacy in rows:
            if name in seen:
                continue
            seen.add(name)
            post = {r[3] for r in rows if r[0] == name}
            pre = {r[4] for r in rows if r[0] == name}
            print(f"    {name:34} pre-fix={sorted(pre)}  post-fix={sorted(post)}  want={expected}")
        for w in wrong:
            print(f"    MISMATCH {w}")
    assert not wrong
    # the point of the whole exercise: no permutation ever yields a false pass
    assert not [r for r in rows if r[2] == FAIL and r[3] == PASS]


def test_single_instance_beside_tray_still_fails(capsys):
    """Regression guard for the case that was already fixed: ONE fork, lying
    beside a 0.30x0.20 tray, whose mask is 100% contained in the tray's."""
    tray_mask = np.zeros((480, 640), bool)
    tray_mask[200:360, 200:440] = True
    fork_mask = np.zeros((480, 640), bool)
    fork_mask[300:340, 400:440] = True  # entirely inside the tray mask
    dets = [
        make_det("fork", "birdview", (-0.90, 0.13, TABLE_Z), mask=fork_mask),
        tray_det("birdview", mask=tray_mask),
    ]
    v, _ = verifier(FakePipeline(dets, cameras=("birdview",)), config=CFG_2D_ON)
    out = v.verify({}, ACTIONS)
    with capsys.disabled():
        print(f"\n  single-instance fork beside tray (100% mask containment) -> {out.status}")
    assert out.status == FAIL
    assert [x.mode for x in out.votes if x.mode != "none"] == ["3d"]


def test_absent_miss_cannot_launder_a_2d_pass():
    """`absent` used to return a 3d PASS on a non-detection, which satisfied
    fusion's "one 3d pass" rule and promoted a 2D-only result to pass."""
    obj_mask = np.zeros((40, 40), bool)
    obj_mask[10:20, 10:20] = True
    cont_mask = np.zeros((40, 40), bool)
    cont_mask[5:30, 5:30] = True
    dets = [
        make_det("fork", "birdview", None, mask=obj_mask),
        tray_det("birdview", mask=cont_mask),
    ]
    v, _ = verifier(FakePipeline(dets, cameras=("birdview",)), config=CFG_2D_ON)
    score = {
        "verify": {
            "all": [
                {"pred": "inside", "obj": "fork", "container": "tray"},
                {"pred": "absent", "obj": "spoon"},  # never detected -- a miss, not a removal
            ]
        }
    }
    out = v.verify(score, ACTIONS)
    modes = {(x.vote, x.mode) for x in out.votes}
    assert (PASS, "3d") not in modes
    assert out.status == UNVERIFIED


def test_outcome_serializes():
    dets = [make_det("fork", "birdview", (-0.90, 0.02, -0.240)), tray_det("birdview")]
    v, _ = verifier(FakePipeline(dets, cameras=("birdview",)))
    data = v.verify({}, ACTIONS).to_dict()
    assert data["status"] == PASS
    assert data["gates"] == {"grasp": True, "transport": True, "release": True}
    assert data["votes"][0]["camera"] == "birdview"


# identity: which physical object a predicate is about


def test_plan_key_refuses_an_ambiguous_instance_label():
    """A predicate label must not silently collapse onto the lowest instance.

    LabelResolvingDetectionMap.resolve_label maps "fork"/"fork 9" -> "fork 1"
    so a cached tree keeps DRIVING through label drift. Reusing that here bound
    the predicate to a fork the tree never touched: obj:"fork" scored a 3D pass
    off the untouched fork in the tray while the manipulated one lay outside.
    """
    plan = LabelResolvingDetectionMap()
    plan["fork 1"] = {"position_3d": [-0.90, 0.30, TABLE_Z]}
    plan["fork 2"] = {"position_3d": [-0.90, 0.45, TABLE_Z]}
    plan["tray"] = {"position_3d": [TRAY_XY[0], TRAY_XY[1], TRAY_Z]}

    assert sv._plan_key(plan, "fork 1") == "fork 1"  # exact still wins
    assert sv._plan_key(plan, "tray") == "tray"
    for ambiguous in ("fork", "fork 3", "fork 9"):
        assert sv._plan_key(plan, ambiguous) is None, ambiguous


def test_plan_key_keeps_the_resolutions_that_are_unambiguous():
    sole = LabelResolvingDetectionMap()
    sole["fork 1"] = {"position_3d": [-0.90, 0.30, TABLE_Z]}
    assert sv._plan_key(sole, "fork") == "fork 1"

    subpart = LabelResolvingDetectionMap()
    subpart["knife handle"] = {"position_3d": [-0.90, 0.30, TABLE_Z]}
    assert sv._plan_key(subpart, "knife") == "knife handle"

    aliased = LabelResolvingDetectionMap()
    aliased["blue block 1"] = {"position_3d": [-0.90, 0.30, TABLE_Z]}
    aliased["blue block 2"] = {"position_3d": [-0.90, 0.45, TABLE_Z]}
    aliased.register_alias("same color block 1", "blue block 2")
    assert sv._plan_key(aliased, "same color block 1") == "blue block 2"


@pytest.mark.parametrize("obj", ["fork", "fork 3", "fork 9"])
def test_ambiguous_predicate_label_cannot_pass(obj):
    """End-to-end reproduction of the wrong PASS B1 was opened for.

    The tree picked `fork 2` and left it 45cm outside the tray. `fork 1` was
    never touched and is sitting IN the tray. A predicate naming the base label
    (or any absent instance number) used to collapse onto `fork 1`, take the
    static branch, and fuse to a 3D pass on the wrong physical fork.
    """
    plan = LabelResolvingDetectionMap()
    plan["fork 1"] = {"position_3d": [-0.90, 0.02, TABLE_Z]}  # untouched, in the tray
    plan["fork 2"] = {"position_3d": [-0.90, 0.30, TABLE_Z]}  # the one the tree picked
    plan["tray"] = dict(DET_MAP["tray"])
    dets = [
        make_det("fork 1", "birdview", (-0.90, 0.02, -0.240)),
        make_det("fork 2", "birdview", (-0.90, 0.45, TABLE_Z)),  # 45cm out of the tray
        tray_det("birdview"),
    ]
    actions = [
        {"type": "move_to_keypoint", "params": {"keypoint_label": "fork 2"}},
        {"type": "grasp", "params": {}},
        {"type": "move_to_keypoint", "params": {"keypoint_label": "tray"}},
        {"type": "release", "params": {}},
    ]
    v, _ = verifier(FakePipeline(dets, cameras=("birdview",)), detection_map=plan)
    score = {"verify": {"all": [{"pred": "inside", "obj": obj, "container": "tray"}]}}
    out = v.verify(score, actions)
    assert out.status != PASS
    assert out.status == UNVERIFIED

    # Control: naming the fork the tree actually moved still decides, and FAILS.
    v2, _ = verifier(FakePipeline(dets, cameras=("birdview",)), detection_map=plan)
    exact = {"verify": {"all": [{"pred": "inside", "obj": "fork 2", "container": "tray"}]}}
    assert v2.verify(exact, actions).status == FAIL


def test_sole_detection_must_still_be_where_it_was_planned():
    """Being the only candidate in view is not evidence of identity.

    A label the tree never grasped that is now far from its plan-time position
    is either a different object or one that moved unobserved. The old code
    short-circuited on `len(cands) == 1` before any geometry ran.
    """
    plan = {"fork": {"position_3d": [-0.90, 0.30, TABLE_Z]}}
    moved = sv.IdentityBinder(
        plan, [make_det("fork", "birdview", (-0.90, 0.60, TABLE_Z))], manipulated=set()
    )
    assert moved.lookup("fork") is None
    assert "from its plan-time position" in moved.explain("fork")

    still = sv.IdentityBinder(
        plan, [make_det("fork", "birdview", (-0.90, 0.32, TABLE_Z))], manipulated=set()
    )
    assert still.lookup("fork") is not None


def test_a_visible_duplicate_makes_the_grasped_object_ambiguous():
    """Two forks in view, one plan-time anchor: elimination cannot pick one."""
    plan = {"fork": {"position_3d": [-0.90, 0.30, TABLE_Z]}}
    dets = [
        make_det("fork", "birdview", (-0.90, 0.30, TABLE_Z)),
        make_det("fork", "birdview", (-0.90, 0.02, -0.240)),
    ]
    binder = sv.IdentityBinder(plan, dets, manipulated={"fork"})
    assert binder.lookup("fork") is None


def test_verify_asks_for_every_instance_of_the_object_under_test():
    """Top-1 per prompt would hide the duplicate the binder needs to see."""
    seen = {}

    class RecordingPipeline(FakePipeline):
        def detect(self, captures, prompts, **kw):
            seen.update(kw)
            seen["prompts"] = list(prompts)
            return list(self._dets)

    dets = [make_det("fork", "birdview", (-0.90, 0.02, -0.240)), tray_det("birdview")]
    v, _ = verifier(RecordingPipeline(dets, cameras=("birdview",)))
    v.verify({}, ACTIONS)
    # The object under test is promoted; the container stays top-1.
    assert "fork" in seen["multi_instance_prompts"]
    assert "tray" not in seen["multi_instance_prompts"]


def test_a_camera_that_saw_nothing_is_named_as_silent():
    """Non-participation must not read the same as agreement.

    With require_two_views off a lone camera carries the whole verdict, so an
    unplugged or occluded birdview has to be visible in the reason.
    """
    v = vote("sideview", PASS, "3d")
    status, reason = sv.fuse_votes([v], silent_cameras=["birdview"])
    assert status == PASS
    assert "birdview=silent" in reason
    assert "sideview=pass" in reason

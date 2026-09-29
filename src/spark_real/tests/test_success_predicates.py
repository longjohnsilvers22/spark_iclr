"""Synthetic-geometry tests for the success predicates.

Pure numpy, no images beyond hand-built masks, no robot, no SAM3.

The load-bearing case is ``fork_beside_overlapping_masks``: a fork dropped
NEXT TO the tray whose 2D mask still overlaps the tray by 65%. The old
``intersection > 0`` verdict scored that as success; ``inside`` must fail it.
"""

import numpy as np
import pytest

from spark_real.control.success_predicates import (
    ABSTAIN,
    FAIL,
    PASS,
    EvalConfig,
    Predicate,
    SceneView,
    VerifyBlockError,
    container_obb_xy,
    derive_default_verify,
    evaluate_predicate,
    grasped_labels,
    mask_containment,
    parse_verify_block,
)

# Rig-realistic geometry (robot base frame, metres): tray 0.30 x 0.20 centred
# at (-0.90, 0.00), rim z = -0.245, table surface z = -0.278.
TRAY_XY = (-0.90, 0.00)
TRAY_Z = -0.245
TABLE_Z = -0.278


def det(label, xyz=None, conf=0.8, mask=None, **kw):
    d = {"label": label, "confidence": conf, "position_3d": None if xyz is None else list(xyz)}
    d["mask"] = mask
    d.update(kw)
    return d


def tray(**kw):
    base = dict(
        obb_minor_m=0.20,  # full short-axis length
        aspect_ratio=1.5,  # 0.30 x 0.20
        world_major_axis_rad=0.0,
    )
    base.update(kw)
    return det("tray", (TRAY_XY[0], TRAY_XY[1], TRAY_Z), conf=0.9, **base)


def view(objects, held=None, camera="birdview"):
    table = {o["label"]: o for o in objects}
    return SceneView(camera=camera, lookup=table.get, held=held)


def rect_mask(x0, y0, x1, y1, shape=(480, 640)):
    m = np.zeros(shape, dtype=bool)
    m[y0:y1, x0:x1] = True
    return m


CFG = EvalConfig()
INSIDE = Predicate(
    "inside", {"obj": "fork", "container": "tray", "xy_margin_m": -0.02, "z_tol_m": 0.06}
)


def vote_of(pred, objects, held=None):
    return evaluate_predicate(pred, view(objects, held=held), CFG)


# the scenario table -- also the confusion matrix's rows


def scenarios():
    tray_mask = rect_mask(200, 200, 440, 360)
    # a fork lying beside the tray whose mask still overlaps it by ~65%
    fork_overlap_mask = rect_mask(400, 300, 460, 340)
    fork_clear_mask = rect_mask(500, 380, 580, 420)
    fork_in_mask = rect_mask(260, 240, 340, 280)

    return [
        ("fork_inside", INSIDE, [det("fork", (-0.90, 0.02, -0.240)), tray()], None, PASS),
        ("fork_beside_on_table", INSIDE, [det("fork", (-0.90, 0.13, TABLE_Z)), tray()], None, FAIL),
        (
            "fork_beside_overlapping_masks",
            INSIDE,
            [det("fork", (-0.90, 0.13, TABLE_Z), mask=fork_overlap_mask), tray(mask=tray_mask)],
            None,
            FAIL,
        ),
        ("fork_above_not_in", INSIDE, [det("fork", (-0.90, 0.00, -0.100)), tray()], None, FAIL),
        (
            "fork_below_rim_through_floor",
            INSIDE,
            [det("fork", (-0.90, 0.00, -0.300)), tray()],
            None,
            FAIL,
        ),
        (
            "fork_on_rim_outside_margin",
            INSIDE,
            [det("fork", (-0.90, 0.095, -0.242)), tray()],
            None,
            FAIL,
        ),
        (
            "fork_just_inside_margin",
            INSIDE,
            [det("fork", (-0.90, 0.075, -0.242)), tray()],
            None,
            PASS,
        ),
        (
            "fork_low_confidence",
            INSIDE,
            [det("fork", (-0.90, 0.02, -0.240), conf=0.24), tray()],
            None,
            ABSTAIN,
        ),
        ("fork_not_detected", INSIDE, [tray()], None, ABSTAIN),
        ("tray_not_detected", INSIDE, [det("fork", (-0.90, 0.02, -0.240))], None, ABSTAIN),
        (
            "2d_only_beside_overlapping",
            INSIDE,
            [det("fork", None, mask=fork_overlap_mask), tray(mask=tray_mask)],
            None,
            FAIL,
        ),
        (
            "2d_only_clear_of_tray",
            INSIDE,
            [det("fork", None, mask=fork_clear_mask), tray(mask=tray_mask)],
            None,
            FAIL,
        ),
        (
            "2d_only_inside",
            INSIDE,
            [det("fork", None, mask=fork_in_mask), tray(mask=tray_mask)],
            None,
            PASS,
        ),
        ("2d_no_masks_either", INSIDE, [det("fork", None), tray()], None, ABSTAIN),
        (
            "removed_from_beside",
            Predicate("removed_from", {"obj": "fork", "container": "tray"}),
            [det("fork", (-0.90, 0.13, TABLE_Z)), tray()],
            None,
            PASS,
        ),
        (
            "removed_from_still_inside",
            Predicate("removed_from", {"obj": "fork", "container": "tray"}),
            [det("fork", (-0.90, 0.02, -0.240)), tray()],
            None,
            FAIL,
        ),
        (
            "removed_from_missing",
            Predicate("removed_from", {"obj": "fork", "container": "tray"}),
            [tray()],
            None,
            ABSTAIN,
        ),
        (
            "stacked_barely_touching",
            Predicate("stacked", {"obj": "blue block", "base": "gray block"}),
            [det("blue block", (-0.7, 0.1, -0.2405)), det("gray block", (-0.7, 0.1, -0.2455))],
            None,
            FAIL,
        ),
        (
            "stacked_one_block_high",
            Predicate("stacked", {"obj": "blue block", "base": "gray block"}),
            [det("blue block", (-0.7, 0.1, -0.2055)), det("gray block", (-0.7, 0.1, -0.2455))],
            None,
            PASS,
        ),
        (
            "stacked_way_above",
            Predicate("stacked", {"obj": "blue block", "base": "gray block"}),
            [det("blue block", (-0.7, 0.1, -0.0455)), det("gray block", (-0.7, 0.1, -0.2455))],
            None,
            FAIL,
        ),
        (
            "stacked_beside",
            Predicate("stacked", {"obj": "blue block", "base": "gray block"}),
            [det("blue block", (-0.7, 0.2, -0.2055)), det("gray block", (-0.7, 0.1, -0.2455))],
            None,
            FAIL,
        ),
        (
            "near_within",
            Predicate("near", {"obj": "pen", "target": "bin", "max_dist_m": 0.10}),
            [det("pen", (-0.9, 0.0, -0.2)), det("bin", (-0.9, 0.05, -0.2))],
            None,
            PASS,
        ),
        (
            "near_outside",
            Predicate("near", {"obj": "pen", "target": "bin", "max_dist_m": 0.10}),
            [det("pen", (-0.9, 0.0, -0.2)), det("bin", (-0.9, 0.30, -0.2))],
            None,
            FAIL,
        ),
        (
            "held_released",
            Predicate("held", {"obj": "fork", "value": False}),
            [tray()],
            False,
            PASS,
        ),
        (
            "held_still_gripped",
            Predicate("held", {"obj": "fork", "value": False}),
            [tray()],
            True,
            FAIL,
        ),
        (
            "held_no_grip_state",
            Predicate("held", {"obj": "fork", "value": False}),
            [tray()],
            None,
            FAIL,
        ),
        # `absent` is asymmetric on purpose: a non-detection is a SAM3 miss or
        # an occlusion just as easily as a removal, so it abstains and can
        # never carry a verdict. Only a positive re-detection is decisive.
        ("absent_not_detected_abstains", Predicate("absent", {"obj": "fork"}), [tray()], None, ABSTAIN),
        (
            "absent_low_conf_abstains",
            Predicate("absent", {"obj": "fork"}),
            [det("fork", (-0.9, 0.3, TABLE_Z), conf=0.10), tray()],
            None,
            ABSTAIN,
        ),
        (
            "absent_still_there",
            Predicate("absent", {"obj": "fork"}),
            [det("fork", (-0.9, 0.3, TABLE_Z)), tray()],
            None,
            FAIL,
        ),
    ]


@pytest.mark.parametrize("name,pred,objects,held,expected", scenarios())
def test_scenario(name, pred, objects, held, expected):
    got = vote_of(pred, objects, held=held)
    assert got.vote == expected, f"{name}: expected {expected}, got {got.vote} ({got.detail})"


def test_confusion_matrix(capsys):
    """Confusion matrix over every synthetic case; off-diagonal must be empty."""
    labels = [PASS, FAIL, ABSTAIN]
    matrix = {e: {a: 0 for a in labels} for e in labels}
    wrong = []
    for name, pred, objects, held, expected in scenarios():
        got = vote_of(pred, objects, held=held).vote
        matrix[expected][got] += 1
        if got != expected:
            wrong.append((name, expected, got))

    with capsys.disabled():
        print("\n  predicate confusion matrix (rows = expected, cols = actual)")
        print("           " + "".join(f"{a:>9}" for a in labels))
        for e in labels:
            print(f"  {e:>8} " + "".join(f"{matrix[e][a]:>9}" for a in labels))
        print(f"  cases={sum(sum(r.values()) for r in matrix.values())} mismatches={len(wrong)}")
        for name, exp, got in wrong:
            print(f"    MISMATCH {name}: expected {exp}, got {got}")
    assert not wrong


def test_beside_but_overlapping_is_the_deliverable():
    """The exact silverware failure: 2D masks overlap, world geometry says no."""
    tray_mask = rect_mask(200, 200, 440, 360)
    fork_mask = rect_mask(400, 300, 460, 340)
    frac = mask_containment(det("f", None, mask=fork_mask), det("t", None, mask=tray_mask))
    assert frac > 0.5, "test setup: masks must genuinely overlap"

    got = vote_of(
        INSIDE,
        [det("fork", (-0.90, 0.13, TABLE_Z), mask=fork_mask), tray(mask=tray_mask)],
    )
    assert got.vote == FAIL
    assert got.mode == "3d", "3D must win whenever it is available"


def test_container_obb_from_slots():
    slots = [{"world_xyz": [-0.98, -0.06, TRAY_Z]}, {"world_xyz": [-0.82, 0.06, TRAY_Z]}]
    centre, half_a, half_b, theta = container_obb_xy(tray(slots=slots))
    assert np.allclose(centre, TRAY_XY)
    assert half_a == pytest.approx(0.08 + 0.02)
    assert half_b == pytest.approx(0.06 + 0.02)
    assert theta == pytest.approx(0.0)


def test_container_obb_from_minor_axis():
    centre, half_a, half_b, theta = container_obb_xy(tray())
    assert half_b == pytest.approx(0.10)  # 0.20 full width -> 0.10 half
    assert half_a == pytest.approx(0.15)


def test_container_with_no_measured_extent_abstains():
    """An unmeasured container has no extent, and must not be given one.

    This used to fabricate an isotropic 10cm radius, so anything within 8cm of
    the centroid passed with mode "3d" -- decisive enough to carry a fused pass
    by itself, whatever the container's real size.
    """
    bare = det("bowl", (-0.9, 0.0, TRAY_Z))
    assert container_obb_xy(bare) is None
    got = evaluate_predicate(
        Predicate("inside", {"obj": "plushie", "container": "bowl"}),
        view([det("plushie", (-0.9, 0.0, -0.24)), bare]),
        CFG,
    )
    assert got.vote == "abstain"
    assert got.mode == "none"
    assert "extent unknown" in got.detail

    # A camera that DID measure the container still decides.
    measured = det("bowl", (-0.9, 0.0, TRAY_Z), obb_minor_m=0.20)
    ok = evaluate_predicate(
        Predicate("inside", {"obj": "plushie", "container": "bowl"}),
        view([det("plushie", (-0.9, 0.0, -0.24)), measured]),
        CFG,
    )
    assert ok.vote == "pass" and ok.mode == "3d"


def test_pca_inverted_container_swaps_axes():
    """aspect_ratio < 1 means the stored major axis is really the short one."""
    upright = tray()  # 0.30 along x, 0.20 along y
    inverted = tray(aspect_ratio=1 / 1.5, world_major_axis_rad=-np.pi / 2)
    probe_x = det("fork", (-0.90 + 0.12, 0.00, -0.242))
    probe_y = det("fork", (-0.90, 0.12, -0.242))
    for cont in (upright, inverted):
        assert vote_of(INSIDE, [probe_x, cont]).vote == PASS
        assert vote_of(INSIDE, [probe_y, cont]).vote == FAIL


def test_2d_pass_needs_high_containment():
    tray_mask = rect_mask(200, 200, 440, 360)
    partly = rect_mask(400, 300, 460, 340)  # 67% of the fork inside
    got = vote_of(INSIDE, [det("fork", None, mask=partly), tray(mask=tray_mask)])
    assert got.vote == FAIL and got.mode == "2d"


def test_da3_depth_widens_the_z_window():
    fork = det("fork", (-0.90, 0.02, -0.175))  # 7cm above the rim
    v = view([fork, tray()])
    assert evaluate_predicate(INSIDE, v, EvalConfig()).vote == FAIL
    assert evaluate_predicate(INSIDE, v, EvalConfig(z_tol_scale=2.0)).vote == PASS


# parsing


def test_parse_valid_block():
    spec = parse_verify_block(
        {
            "all": [
                {"pred": "inside", "obj": "knife 1", "container": "tray", "z_tol_m": 0.08},
                {"pred": "held", "obj": "knife 1", "value": False},
            ],
            "min_conf": 0.4,
        }
    )
    assert spec.describe() == ["inside(knife 1, tray)", "held(knife 1, False)"]
    assert spec.min_conf == 0.4
    assert spec.predicates[0].params["z_tol_m"] == 0.08


@pytest.mark.parametrize(
    "block",
    [
        {"all": [{"pred": "teleported", "obj": "a", "container": "b"}]},
        {"all": [{"pred": "near", "obj": "a", "target": "b"}]},  # no max_dist_m
        {"all": []},
        {"any": [{"pred": "inside", "obj": "a", "container": "b"}]},
        [{"pred": "inside", "obj": "a", "container": "b"}],
        {"all": [{"pred": "inside", "obj": "a"}]},
        {"all": [{"pred": "inside", "obj": "a", "container": "b"}], "min_conf": "high"},
    ],
)
def test_parse_rejects_malformed(block):
    with pytest.raises(VerifyBlockError):
        parse_verify_block(block)


def test_parse_rejects_unknown_label():
    known = {"tray"}
    with pytest.raises(VerifyBlockError):
        parse_verify_block(
            {"all": [{"pred": "inside", "obj": "ghost", "container": "tray"}]},
            label_resolver=lambda lab: lab if lab in known else None,
        )


# derivation


def test_derive_pick_and_place():
    actions = [
        {"type": "move_to_keypoint", "params": {"keypoint_label": "knife 1"}},
        {"type": "grasp", "params": {}},
        {"type": "move_to_keypoint", "params": {"keypoint_label": "tray"}},
        {"type": "release", "params": {}},
    ]
    spec = derive_default_verify(actions, {"tray": tray()})
    assert spec.describe() == ["inside(knife 1, tray)", "held(knife 1, False)"]
    assert spec.source == "derived"


def test_derive_place_on_small_target():
    actions = [
        {"type": "move_to_keypoint", "params": {"keypoint_label": "blue block"}},
        {"type": "grasp", "params": {}},
        {"type": "move_to_keypoint", "params": {"keypoint_label": "gray block"}},
        {"type": "release", "params": {}},
    ]
    spec = derive_default_verify(
        actions, {"gray block": det("gray block", (0, 0, 0), obb_minor_m=0.04)}
    )
    assert spec.describe()[0] == "on(blue block, gray block)"


def test_derive_place_in_slot_and_stack_and_sweep():
    spec = derive_default_verify(
        [
            {"type": "move_to_keypoint", "params": {"keypoint_label": "fork 1"}},
            {"type": "grasp", "params": {}},
            {"type": "place_in_slot", "params": {"container_label": "tray"}},
        ]
    )
    assert spec.describe()[0] == "inside(fork 1, tray)"

    spec = derive_default_verify(
        [
            {"type": "move_to_keypoint", "params": {"keypoint_label": "blue block"}},
            {"type": "grasp", "params": {}},
            {"type": "stack", "params": {"target_label": "gray block"}},
        ]
    )
    assert spec.describe()[0] == "stacked(blue block, gray block)"

    spec = derive_default_verify(
        [
            {
                "type": "sweep",
                "params": {"target_label": "dustpan", "object_labels": "crumb 1, crumb 2"},
            }
        ]
    )
    assert spec.describe() == ["inside(crumb 1, dustpan)", "inside(crumb 2, dustpan)"]


def test_derive_returns_none_when_nothing_derivable():
    assert derive_default_verify([{"type": "pour", "params": {"target_label": "cup"}}]) is None
    assert derive_default_verify([]) is None


def test_grasped_labels_tracks_the_manipulated_objects():
    """The binder needs to know which labels were allowed to move."""
    assert grasped_labels(
        [
            {"type": "move_to_keypoint", "params": {"keypoint_label": "fork 2"}},
            {"type": "grasp", "params": {}},
            {"type": "move_to_keypoint", "params": {"keypoint_label": "tray"}},
            {"type": "release", "params": {}},
        ]
    ) == ["fork 2"]
    # explicit label on the grasp wins over the pending keypoint
    assert grasped_labels(
        [
            {"type": "move_to_keypoint", "params": {"keypoint_label": "tray"}},
            {"type": "grasp_se3", "params": {"label": "knife 1"}},
        ]
    ) == ["knife 1"]
    assert grasped_labels([{"type": "release", "params": {}}]) == []


# `absent`: undetected is not achieved


def test_absent_never_passes_from_a_non_detection():
    """A missed detection is absence of evidence -- it must not score success.

    It must also not be mode ``3d``: fusion demands one 3d PASS vote before a
    fused pass, so a passing `absent` used to launder a 2D-only result.
    """
    got = vote_of(Predicate("absent", {"obj": "fork"}), [tray()])
    assert got.vote == ABSTAIN
    assert got.mode == "none"
    assert got.vote != PASS


def test_absent_still_fails_on_a_positive_detection():
    got = vote_of(Predicate("absent", {"obj": "fork"}), [det("fork", (-0.9, 0.3, TABLE_Z)), tray()])
    assert got.vote == FAIL and got.mode == "3d"


def test_absent_cannot_supply_the_3d_pass_vote():
    """The laundering path, end to end at the vote level."""
    votes = [
        vote_of(INSIDE, [det("fork", None, mask=rect_mask(260, 240, 340, 280)),
                         tray(mask=rect_mask(200, 200, 440, 360))]),
        vote_of(Predicate("absent", {"obj": "spoon"}), [tray()]),
    ]
    assert votes[0].vote == PASS and votes[0].mode == "2d"
    assert not [v for v in votes if v.vote == PASS and v.mode == "3d"]


def test_explain_hook_reaches_the_vote_detail():
    v = SceneView(
        camera="birdview",
        lookup=lambda lab: None,
        explain=lambda lab: "3 candidates; identity ambiguous",
    )
    got = evaluate_predicate(INSIDE, v, CFG)
    assert got.vote == ABSTAIN
    assert "identity ambiguous" in got.detail

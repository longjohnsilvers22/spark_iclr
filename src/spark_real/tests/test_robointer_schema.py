"""
Offline tests for the RoboInter plan extension's schema.

No Gemini, no network, no robot, no camera. Every "LLM response" here is a
fabricated string in tests/fixtures/robointer/, parsed through exactly the path
generate_score() uses (parse_plan_yaml -> sanitize_score) and then through the
RoboInter layer.

Covers:
  - a well-formed annotation parses, in the 0..1000 space the prompt asks for;
  - the same block in normalized [0,1] parses identically;
  - malformed fields are reported and DROPPED, never guessed at, and one bad
    field does not void the rest of the node;
  - a contact point outside its own object box is caught;
  - the planner's own validator is indifferent to the annotations.
"""

from __future__ import annotations

from pathlib import Path

import pytest
import yaml

from spark_real.planning.robointer import (
    ANNOTATION_KEY,
    COORD_SCALE,
    FCOT_KEY,
    MAX_TRACE_POINTS,
    Box2D,
    NodeAnnotation,
    Point2D,
    Pose6D,
    RoboInterSchemaError,
    extract_annotations,
    parse_annotation,
    sanitize_plan_annotations,
    set_annotation,
    strip_annotations,
    validate_plan_annotations,
)
from spark_real.planning.spark_planner import SPARKPlanner

FIXTURES = Path(__file__).parent / "fixtures" / "robointer"
LABELS = ["knife handle 1", "tray", "plushie", "bowl", "pen", "bin"]


def load(name: str) -> str:
    return (FIXTURES / name).read_text()


def parse(name: str, labels=LABELS) -> dict:
    """Through the real planner post-network path, as production does."""
    return SPARKPlanner.parse_response(load(name), labels)


# --------------------------------------------------------------------------
# 1. A well-formed annotation.
# --------------------------------------------------------------------------


def test_annotated_plan_parses_through_the_planner_path():
    score = parse("annotated_plan.yaml")
    assert score["task"] == "put the knife in the tray"
    # The planner is indifferent to the new keys: they survive sanitize_score.
    assert score[FCOT_KEY][0].startswith("the knife lies")
    assert score["verify"]["all"][0]["pred"] == "inside"

    anns = extract_annotations(score, LABELS)
    assert [p for p, _ in anns] == ["tree/0", "tree/3"]

    pick = anns[0][1]
    assert pick.primitive_skill == "pick"
    assert pick.label == "knife handle 1"
    assert pick.camera == "sideview"
    # 425/1000 of the width, 508/1000 of the height.
    assert pick.contact_point.to_permille() == [425, 508]
    assert pick.object_box.to_list() == [[0.38, 0.47], [0.7, 0.545]]
    assert pick.affordance_box is not None

    place = anns[1][1]
    assert place.primitive_skill == "place"
    assert place.placement_proposal.center.to_permille() == [210, 365]
    assert len(place.trace) == 3


def test_validation_of_a_good_plan_is_silent():
    assert validate_plan_annotations(parse("annotated_plan.yaml"), LABELS) == []


def test_permille_and_normalized_coordinates_agree():
    permille = {"object_box": [[380, 470], [700, 545]], "contact_point": [425, 508]}
    normalized = {"object_box": [[0.38, 0.47], [0.70, 0.545]], "contact_point": [0.425, 0.508]}
    a, ia = parse_annotation(permille)
    b, ib = parse_annotation(normalized)
    assert ia == [] and ib == []
    assert a.to_dict() == b.to_dict()


def test_the_scale_is_decided_once_per_block_not_per_value():
    # A block mixing a small permille value with large ones must not read the
    # small one as already-normalized -- that would put it at the frame edge.
    ann, issues = parse_annotation({"object_box": [[1, 470], [700, 545]]})
    assert issues == []
    assert ann.object_box.to_permille_pair() == [[1, 470], [700, 545]]


def test_a_flat_four_number_box_is_accepted():
    ann, issues = parse_annotation({"object_box": [380, 470, 700, 545]})
    assert issues == []
    assert ann.object_box.to_list() == [[0.38, 0.47], [0.7, 0.545]]


def test_box_corners_are_ordered_not_rejected():
    ann, issues = parse_annotation({"object_box": [[700, 545], [380, 470]]})
    assert issues == []
    assert ann.object_box.x1 < ann.object_box.x2 and ann.object_box.y1 < ann.object_box.y2


# --------------------------------------------------------------------------
# 2. Malformed annotations: reported, dropped, never guessed.
# --------------------------------------------------------------------------


def test_malformed_annotations_are_all_reported():
    score = yaml.safe_load(load("malformed_annotations.yaml"))
    issues = validate_plan_annotations(score, LABELS)
    joined = "\n".join(issues)
    assert "unknown primitive_skill 'teleport'" in joined
    assert "label 'banana' is not a detected keypoint" in joined
    assert "outside its own object_box" in joined
    assert "a trace needs at least 2" in joined
    assert "unknown field 'grip_force'" in joined
    assert "expected [[x1,y1],[x2,y2]]" in joined
    assert "must be a mapping" in joined
    assert "probably degrees" in joined


def test_sanitize_drops_the_bad_fields_and_keeps_the_good_ones():
    score = yaml.safe_load(load("malformed_annotations.yaml"))
    clean, issues = sanitize_plan_annotations(score, LABELS)
    assert issues, "expected the issues to be reported, not swallowed"

    node0 = clean["tree"]["children"][0][ANNOTATION_KEY]
    # Kept: the fields that were well formed.
    assert node0["subtask"] == "grab it"
    assert node0["object_box"] == [[0.38, 0.47], [0.7, 0.545]]
    # Dropped: unknown skill, unknown label, 1-point trace, degrees-not-radians
    # pose, and the field nobody defined.
    for gone in ("primitive_skill", "label", "grip_force"):
        assert gone not in node0
    assert "trace" not in node0 or len(node0["trace"]) >= 2

    # A whole non-mapping block disappears rather than being coerced.
    assert ANNOTATION_KEY not in clean["tree"]["children"][2]
    # The plan itself is untouched.
    assert [c["type"] for c in clean["tree"]["children"]] == [
        "move_to_keypoint",
        "grasp",
        "release",
    ]


def test_a_contact_point_outside_its_object_box_is_refused_not_clamped():
    ann, issues = parse_annotation(
        {"object_box": [[380, 470], [700, 545]], "contact_point": [910, 120]}
    )
    assert any("outside its own object_box" in i for i in issues)
    # Still parsed -- the caller decides. sanitize is what drops it.
    assert ann.contact_point.to_permille() == [910, 120]


def test_strict_mode_raises():
    score = yaml.safe_load(load("malformed_annotations.yaml"))
    with pytest.raises(RoboInterSchemaError):
        sanitize_plan_annotations(score, LABELS, strict=True)


def test_an_oversized_trace_is_reported():
    pts = [[i * 10, i * 10] for i in range(MAX_TRACE_POINTS + 5)]
    _ann, issues = parse_annotation({"trace": pts})
    assert any("exceeds the" in i for i in issues)


def test_non_numeric_junk_never_becomes_a_coordinate():
    ann, issues = parse_annotation({"contact_point": ["left", "of the handle"], "object_box": None})
    assert ann is None
    assert any("contact_point" in i for i in issues)


def test_booleans_are_not_coordinates():
    _ann, issues = parse_annotation({"contact_point": [True, 500]})
    assert any("contact_point" in i for i in issues)


# --------------------------------------------------------------------------
# 3. Small-object behaviour.
# --------------------------------------------------------------------------


def test_empty_annotation_is_the_same_as_none():
    ann, issues = parse_annotation({})
    assert ann is None and issues == []
    node = {"type": "grasp", "params": {}}
    set_annotation(node, NodeAnnotation())
    assert ANNOTATION_KEY not in node


def test_to_dict_is_yaml_safe():
    # EpisodeRecorder writes bt.yaml with yaml.safe_dump; a dataclass or a
    # numpy scalar leaking in there kills the whole episode bundle.
    score = parse("annotated_plan.yaml")
    dumped = yaml.safe_dump(score, sort_keys=False)
    assert yaml.safe_load(dumped) == score


def test_strip_annotations_removes_every_key_at_every_level():
    score = parse("annotated_plan.yaml")
    bare = strip_annotations(score)
    assert FCOT_KEY not in bare
    assert all(ANNOTATION_KEY not in c for c in bare["tree"]["children"])
    # ...and did not touch the original.
    assert FCOT_KEY in score


def test_box_helpers():
    b = Box2D(0.2, 0.4, 0.6, 0.8)
    assert b.center.to_list() == [0.4, 0.6]
    assert b.to_pixels((1000, 500)) == (200.0, 200.0, 600.0, 400.0)
    assert Box2D.from_pixels(200, 200, 600, 400, (1000, 500)).to_list() == b.to_list()
    assert b.iou(b) == pytest.approx(1.0)
    assert b.iou(Box2D(0.8, 0.9, 0.9, 0.95)) == 0.0


def test_point_pixel_round_trip():
    p = Point2D.from_pixels(320, 240, (640, 480))
    assert p.to_list() == [0.5, 0.5]
    assert p.to_pixels((640, 480)) == (320.0, 240.0)
    assert p.to_permille() == [int(COORD_SCALE // 2)] * 2


def test_pose6d_flags_a_pose_that_is_not_in_metres():
    assert Pose6D(0.4, 0.1, 0.2, 2.2, -2.2, 0.0).issues("p") == []
    assert any("not metres" in i for i in Pose6D(400, 100, 200, 0, 0, 0).issues("p"))

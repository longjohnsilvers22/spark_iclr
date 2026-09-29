"""One interface for SPARK's two spatial-annotation routes.

RoboInter (the planner's INLINE ``__robointer`` blocks) and the annotation
providers (OUT-OF-BAND ER2 / Molmo / human pointing) are two routes to the
same capability -- a model supplying spatial corrections to plan steps -- and
these tests pin down the unification:

* the conversions in ``perception/annotations.py`` round-trip a
  ``__robointer`` block (0..1000 grid, **x first**) through ``Annotation``
  objects losslessly, on exactly the conventions the existing robointer
  geometry/schema tests fix (which stay green untouched);
* the ER2 converters and the RoboInter converters agree on the internal
  representation for the same physical pixel, so the [y, x] transpose can
  never sneak in through the new seam;
* ``planning.robointer.source`` selects who supplies the correction pixel
  (``planner`` inline -- the default, byte-identical to before -- or an
  out-of-band provider), the provider path runs through the SAME bounded
  guard, and every provider failure mode falls open to the planner value.

The live-path harness is borrowed from test_robointer_live_path: the REAL
``ExecutionMixin.execute()`` over fake cameras, with only the annotation
provider mocked.
"""

from __future__ import annotations


import numpy as np
import pytest

from spark_real.perception.annotations import (
    Annotation,
    annotations_from_robointer,
    annotations_from_robointer_block,
    point_from_er2,
    robointer_block_from_annotations,
    robointer_from_annotations,
    role_label,
    split_role_label,
)
from spark_real.planning import robointer_gate
from spark_real.planning.robointer import (
    Point2D,
    parse_annotation,
)
from spark_real.planning.spark_planner import parse_plan_yaml
from spark_real.tests.test_robointer_geometry import TABLE_Z, birdview
from spark_real.tests.test_robointer_live_path import (
    KNIFE_CENTROID,
    KNIFE_HANDLE,
    OTHER_OBJECT,
    SHAPE,
    TRAY,
    LivePipeline,
    annotated_reply,
    det,
)

# ---------------------------------------------------------------------------
# 1. One representation: __robointer <-> Annotation round trips
# ---------------------------------------------------------------------------


BLOCK = {
    "subtask": "reach the knife's handle, not its blade",
    "primitive_skill": "pick",
    "label": "knife 1",
    "camera": "sideview",
    "object_box": [[412, 300], [655, 372]],
    "contact_point": [455, 337],
    "placement_proposal": [[120, 300], [300, 430]],
    "trace": [[455, 337], [520, 300], [610, 250]],
}


class TestRoundTrips:
    def test_block_round_trips_exactly_on_the_permille_grid(self):
        """0..1000 integers survive block -> Annotations -> block unchanged.

        This is the grid-and-axis-order contract: the emitted block must be
        byte-equal to the planner-native input, so the conversions cannot
        have rescaled or transposed anything.
        """
        anns = annotations_from_robointer_block(BLOCK)
        assert {a.kind for a in anns} == {"point", "box", "trace"}
        back = robointer_block_from_annotations(anns)
        assert back == BLOCK

    def test_node_annotation_round_trips_through_annotations(self):
        ann, issues = parse_annotation(BLOCK)
        assert ann is not None and not issues
        back = robointer_from_annotations(annotations_from_robointer(ann))
        assert back.contact_point == ann.contact_point
        assert back.object_box == ann.object_box
        assert back.placement_proposal == ann.placement_proposal
        assert back.trace == ann.trace
        assert back.label == ann.label
        assert back.camera == ann.camera
        assert back.subtask == ann.subtask
        assert back.primitive_skill == ann.primitive_skill

    def test_axis_order_agrees_with_er2_for_the_same_physical_pixel(self):
        """RoboInter [x=900, y=100] and ER2 [y=100, x=900] are ONE point.

        The asymmetric coordinates make a transpose loud: getting this wrong
        lands at (0.1, 0.9) instead of (0.9, 0.1).
        """
        anns = annotations_from_robointer_block({"contact_point": [900, 100]})
        assert anns[0].point == pytest.approx((0.9, 0.1))
        assert point_from_er2([100, 900]) == pytest.approx(anns[0].point)

    def test_the_role_and_target_ride_in_the_label_hint(self):
        anns = annotations_from_robointer_block(BLOCK)
        by_role = {split_role_label(a.label)[0]: a for a in anns}
        assert set(by_role) == {
            "contact_point",
            "object_box",
            "placement_proposal",
            "trace",
        }
        for a in anns:
            role, target = split_role_label(a.label)
            assert a.label == role_label(role, "knife 1")
            assert target == "knife 1"
            assert a.provider == "planner"

    def test_a_role_free_label_is_not_mistaken_for_a_role(self):
        assert split_role_label("plushie") == (None, "plushie")
        assert split_role_label("contact_point:") == ("contact_point", "")

    def test_a_provider_placement_point_becomes_a_box_centred_on_it(self):
        """The out-of-band shape (a point) fits the inline schema (a box)."""
        pt = Annotation(
            kind="point",
            points=[(0.4, 0.6)],
            provider="er2",
            label=role_label("placement_proposal", "tray"),
        )
        node_ann = robointer_from_annotations([pt])
        box = node_ann.placement_proposal
        assert box is not None
        assert (box.center.x, box.center.y) == pytest.approx((0.4, 0.6))
        assert not box.issues("p"), "the synthesized box must be schema-valid"

    def test_kind_role_mismatch_drops_the_field_not_the_rest(self):
        good = Annotation(kind="point", points=[(0.2, 0.3)], label="contact_point:knife")
        bad = Annotation(kind="trace", points=[(0.1, 0.1), (0.2, 0.2)], label="object_box:knife")
        node_ann = robointer_from_annotations([good, bad])
        assert node_ann.contact_point == Point2D(0.2, 0.3)
        assert node_ann.object_box is None

    def test_duplicate_roles_first_wins(self):
        a1 = Annotation(kind="point", points=[(0.2, 0.3)], label="contact_point:knife")
        a2 = Annotation(kind="point", points=[(0.8, 0.8)], label="contact_point:knife")
        assert robointer_from_annotations([a1, a2]).contact_point == Point2D(0.2, 0.3)

    def test_an_unusable_block_converts_to_nothing(self):
        assert annotations_from_robointer_block({"contact_point": "nope"}) == []
        assert annotations_from_robointer_block({}) == []
        assert annotations_from_robointer(None) == []

    def test_a_normalized_block_round_trips_within_grid_quantization(self):
        """Blocks stored normalized (the sanitized form) survive to 1/1000."""
        block = {"contact_point": [0.4553, 0.3371]}
        anns = annotations_from_robointer_block(block)
        back = robointer_block_from_annotations(anns)
        assert back["contact_point"] == [455, 337]
        # ...and parsing the emitted block lands on the same internal point.
        ann2, _ = parse_annotation(back)
        assert ann2.contact_point.x == pytest.approx(0.4553, abs=1e-3)
        assert ann2.contact_point.y == pytest.approx(0.3371, abs=1e-3)


# ---------------------------------------------------------------------------
# 2. The source-selection seam (mocked provider, real execute())
# ---------------------------------------------------------------------------


def norm_point_at(xyz) -> tuple:
    """A base-frame point as the provider's normalized (x, y) answer."""
    cam = birdview()
    u, v = cam.base_to_pixel(xyz)
    w, h = cam.image_size
    return (u / w, v / h)


class FakeProvider:
    """An AnnotationProvider that answers every query with one fixed point."""

    name = "er2"

    def __init__(self, answer_xy=None, exc=None):
        self.answer_xy = answer_xy
        self.exc = exc
        self.queries = []

    def available(self):
        return True

    def annotate(self, image, query, kind="point"):
        self.queries.append(query)
        if self.exc is not None:
            raise self.exc
        if self.answer_xy is None:
            return []
        return [
            Annotation(
                kind="point",
                points=[self.answer_xy],
                provider=self.name,
                label=query,
                confidence=0.9,
            )
        ]


# Where the provider points: NOT where the planner pointed, and within the
# 8 cm contact guard of the knife's measured centroid.
PROVIDER_TARGET = KNIFE_CENTROID + np.array([0.0, 0.03, 0.0])
PROVIDER_PLACE = TRAY + np.array([0.05, 0.0, 0.0])


@pytest.fixture
def source_er2(monkeypatch):
    monkeypatch.setenv(robointer_gate.ROBOINTER_ENV, "1")
    monkeypatch.setenv(robointer_gate.ROBOINTER_SOURCE_ENV, "er2")


def pipe_with_frames(provider, monkeypatch, frames=True):
    monkeypatch.setattr(robointer_gate, "_get_annotation_provider", lambda name: provider)
    pipe = LivePipeline()
    if frames:
        rgb = np.zeros((*SHAPE, 3), np.uint8)
        pipe._last_captures = {"birdview": {"rgb": rgb, "depth": None}}
    return pipe


def test_default_source_is_planner():
    cfg = robointer_gate.RoboInterConfig.from_dict({"enabled": True})
    assert cfg.source == "planner"


def test_planner_source_is_todays_behaviour(monkeypatch):
    """source: planner must not touch the provider registry at all."""
    monkeypatch.setenv(robointer_gate.ROBOINTER_ENV, "1")
    monkeypatch.setenv(robointer_gate.ROBOINTER_SOURCE_ENV, "planner")

    def boom(name):  # pragma: no cover - the assertion is that it never runs
        raise AssertionError("planner source must not build a provider")

    monkeypatch.setattr(robointer_gate, "_get_annotation_provider", boom)
    pipe = LivePipeline()
    dets = [det("knife handle 1", KNIFE_CENTROID)]
    pipe.execute(parse_plan_yaml(annotated_reply(contact=KNIFE_HANDLE)), dets)
    entry = pipe._executor.detection_map["knife handle 1"]
    got = np.asarray(entry[robointer_gate.CONTACT_KEY], dtype=float)
    assert np.allclose(got[:2], KNIFE_HANDLE[:2], atol=5e-3)


def test_er2_source_replaces_the_planner_point(source_er2, monkeypatch):
    provider = FakeProvider(answer_xy=norm_point_at(PROVIDER_TARGET))
    pipe = pipe_with_frames(provider, monkeypatch)
    dets = [det("knife handle 1", KNIFE_CENTROID)]
    pipe.execute(parse_plan_yaml(annotated_reply(contact=KNIFE_HANDLE)), dets)

    entry = pipe._executor.detection_map["knife handle 1"]
    got = np.asarray(entry[robointer_gate.CONTACT_KEY], dtype=float)
    assert np.allclose(got[:2], PROVIDER_TARGET[:2], atol=5e-3), got
    assert not np.allclose(
        got[:2], KNIFE_HANDLE[:2], atol=5e-3
    ), "the planner's inline point was used despite source: er2"
    assert got[2] == pytest.approx(TABLE_Z), "Z still comes from perception"
    assert provider.queries and "knife handle 1" in provider.queries[0]
    assert pipe._last_robointer[0]["source"] == "er2"


def test_er2_source_answers_placement_queries_too(source_er2, monkeypatch):
    provider = FakeProvider(answer_xy=norm_point_at(PROVIDER_PLACE))
    pipe = pipe_with_frames(provider, monkeypatch)
    dets = [det("knife handle 1", KNIFE_CENTROID), det("tray", TRAY)]
    score = parse_plan_yaml(annotated_reply(contact=KNIFE_HANDLE, placement=TRAY))
    pipe.execute(score, dets)

    entry = pipe._executor.detection_map["tray"]
    got = np.asarray(entry[robointer_gate.PLACEMENT_KEY], dtype=float)
    assert np.allclose(got[:2], PROVIDER_PLACE[:2], atol=6e-3), got


def test_provider_errors_fail_open_to_the_planner_value(source_er2, monkeypatch):
    provider = FakeProvider(exc=RuntimeError("quota"))
    pipe = pipe_with_frames(provider, monkeypatch)
    dets = [det("knife handle 1", KNIFE_CENTROID)]
    pipe.execute(parse_plan_yaml(annotated_reply(contact=KNIFE_HANDLE)), dets)

    entry = pipe._executor.detection_map["knife handle 1"]
    got = np.asarray(entry[robointer_gate.CONTACT_KEY], dtype=float)
    assert np.allclose(got[:2], KNIFE_HANDLE[:2], atol=5e-3), got


def test_a_provider_miss_fails_open_to_the_planner_value(source_er2, monkeypatch):
    provider = FakeProvider(answer_xy=None)  # answered, but with nothing
    pipe = pipe_with_frames(provider, monkeypatch)
    dets = [det("knife handle 1", KNIFE_CENTROID)]
    pipe.execute(parse_plan_yaml(annotated_reply(contact=KNIFE_HANDLE)), dets)

    got = np.asarray(
        pipe._executor.detection_map["knife handle 1"][robointer_gate.CONTACT_KEY],
        dtype=float,
    )
    assert np.allclose(got[:2], KNIFE_HANDLE[:2], atol=5e-3)


def test_no_captured_frame_fails_open_to_the_planner_value(source_er2, monkeypatch):
    provider = FakeProvider(answer_xy=norm_point_at(PROVIDER_TARGET))
    pipe = pipe_with_frames(provider, monkeypatch, frames=False)
    dets = [det("knife handle 1", KNIFE_CENTROID)]
    pipe.execute(parse_plan_yaml(annotated_reply(contact=KNIFE_HANDLE)), dets)

    got = np.asarray(
        pipe._executor.detection_map["knife handle 1"][robointer_gate.CONTACT_KEY],
        dtype=float,
    )
    assert np.allclose(got[:2], KNIFE_HANDLE[:2], atol=5e-3)
    assert not provider.queries, "no frame -> the provider must not be asked"


def test_an_unavailable_provider_falls_back_to_planner(source_er2, monkeypatch):
    monkeypatch.setattr(robointer_gate, "_get_annotation_provider", lambda name: None)
    pipe = LivePipeline()
    dets = [det("knife handle 1", KNIFE_CENTROID)]
    pipe.execute(parse_plan_yaml(annotated_reply(contact=KNIFE_HANDLE)), dets)

    entry = pipe._executor.detection_map["knife handle 1"]
    got = np.asarray(entry[robointer_gate.CONTACT_KEY], dtype=float)
    assert np.allclose(got[:2], KNIFE_HANDLE[:2], atol=5e-3)
    assert pipe._last_robointer[0]["source"] == "planner"


def test_the_guard_bounds_the_provider_exactly_like_the_planner(source_er2, monkeypatch):
    """A specialist pointer gets no more trust than the generalist planner."""
    provider = FakeProvider(answer_xy=norm_point_at(OTHER_OBJECT))
    pipe = pipe_with_frames(provider, monkeypatch)
    dets = [det("knife handle 1", KNIFE_CENTROID)]
    pipe.execute(parse_plan_yaml(annotated_reply(contact=KNIFE_HANDLE)), dets)

    entry = pipe._executor.detection_map["knife handle 1"]
    assert (
        robointer_gate.CONTACT_KEY not in entry
    ), "a 40 cm provider 'correction' was accepted; the guard is not source-agnostic"


def test_source_off_publishes_no_correction_from_anybody(monkeypatch):
    monkeypatch.setenv(robointer_gate.ROBOINTER_ENV, "1")
    monkeypatch.setenv(robointer_gate.ROBOINTER_SOURCE_ENV, "off")
    pipe = LivePipeline()
    dets = [det("knife handle 1", KNIFE_CENTROID)]
    pipe.execute(parse_plan_yaml(annotated_reply(contact=KNIFE_HANDLE)), dets)

    for entry in pipe._executor.detection_map.values():
        assert not [k for k in entry if k.startswith("robointer")], entry


def test_an_unknown_source_on_an_enabled_block_raises():
    with pytest.raises(RuntimeError, match="source"):
        robointer_gate.RoboInterConfig.from_dict({"enabled": True, "source": "gpt7"})


def test_an_unknown_source_on_a_disabled_block_degrades_to_planner():
    cfg = robointer_gate.RoboInterConfig.from_dict({"enabled": False, "source": "gpt7"})
    assert cfg.source == "planner"


def test_only_the_declared_fields_are_re_asked(source_er2, monkeypatch):
    """A contact-only block makes exactly one provider query, not two."""
    provider = FakeProvider(answer_xy=norm_point_at(PROVIDER_TARGET))
    pipe = pipe_with_frames(provider, monkeypatch)
    dets = [det("knife handle 1", KNIFE_CENTROID), det("tray", TRAY)]
    pipe.execute(parse_plan_yaml(annotated_reply(contact=KNIFE_HANDLE)), dets)
    assert len(provider.queries) == 1
    assert "grasp" in provider.queries[0]

"""
The consumer path and the prompt payload.

The consumer functions are pure: they turn an annotation into the same
base-frame target the executor already moves to, or refuse. Nothing here
touches ``control/``, and nothing here is wired into a run -- see the module
docstring of ``planning/robointer_consume.py``.

The property that matters is the guard: an LLM pixel coordinate is accepted
only as a BOUNDED correction to a perception result, so a confidently wrong
coordinate costs a few centimetres, never a lunge at the next object.
"""

from __future__ import annotations

import numpy as np
import pytest

from spark_real.planning.robointer import Box2D, NodeAnnotation, Point2D
from spark_real.planning.robointer_consume import (
    DEFAULT_MAX_SHIFT_M,
    annotations_for_record,
    contact_target_xyz,
    placement_target_xyz,
    resolved_for_record,
)
from spark_real.planning.robointer_prompt import (
    ROBOINTER_PROMPT_SECTION,
    build_robointer_context,
    robointer_fewshot,
)
from spark_real.tests.test_robointer_geometry import TABLE_Z, birdview


class FakeDet:
    """Duck-typed stand-in, same shape the planner helpers already accept."""

    def __init__(self, label, position_3d, bbox=None, centroid_2d=None, camera=None):
        self.label = label
        self.position_3d = np.asarray(position_3d, dtype=float)
        self.bbox = bbox
        self.centroid_2d = centroid_2d
        self.camera = camera
        self.confidence = 0.9
        self.mask = None


def _point_at(cam, xyz) -> Point2D:
    u, v = cam.base_to_pixel(xyz)
    return Point2D.from_pixels(u, v, cam.image_size)


# --------------------------------------------------------------------------
# contact_point -> grasp XY
# --------------------------------------------------------------------------


def test_a_contact_point_corrects_the_grasp_xy_and_keeps_perceptions_z():
    cam = birdview()
    # The mask centroid sits mid-knife; the handle is 5 cm along -X.
    detected = np.array([-0.80, 0.10, TABLE_Z])
    handle = np.array([-0.85, 0.10, TABLE_Z])
    ann = NodeAnnotation(contact_point=_point_at(cam, handle))

    target, reason = contact_target_xyz(ann, FakeDet("knife 1", detected), cam)
    assert target is not None, reason
    assert np.allclose(target[:2], handle[:2], atol=2e-3)
    assert target[2] == pytest.approx(TABLE_Z), "Z must come from perception"
    assert "accepted" in reason and "5.0 cm" in reason


def test_a_contact_point_on_the_wrong_object_is_refused():
    cam = birdview()
    detected = np.array([-0.80, 0.10, TABLE_Z])
    other_object = np.array([-0.50, 0.35, TABLE_Z])  # 40 cm away
    ann = NodeAnnotation(contact_point=_point_at(cam, other_object))

    target, reason = contact_target_xyz(ann, FakeDet("knife 1", detected), cam)
    assert target is None
    assert "keeping the detection" in reason
    assert f"cap {DEFAULT_MAX_SHIFT_M * 100:.0f} cm" in reason


def test_no_annotation_means_no_override():
    cam = birdview()
    det = FakeDet("knife 1", [-0.80, 0.10, TABLE_Z])
    target, reason = contact_target_xyz(NodeAnnotation(), det, cam)
    assert target is None and "not emitted" in reason


def test_a_detection_without_a_3d_position_cannot_be_corrected():
    cam = birdview()
    ann = NodeAnnotation(contact_point=Point2D(0.5, 0.5))
    det = FakeDet("knife 1", [0, 0, 0])
    det.position_3d = None
    target, reason = contact_target_xyz(ann, det, cam)
    assert target is None and "no position_3d" in reason


def test_the_shift_cap_is_a_parameter_not_a_law():
    cam = birdview()
    detected = np.array([-0.80, 0.10, TABLE_Z])
    far = np.array([-0.92, 0.10, TABLE_Z])  # 12 cm
    ann = NodeAnnotation(contact_point=_point_at(cam, far))
    assert contact_target_xyz(ann, FakeDet("k", detected), cam)[0] is None
    loose, _ = contact_target_xyz(ann, FakeDet("k", detected), cam, max_shift_m=0.15)
    assert loose is not None and np.allclose(loose[:2], far[:2], atol=2e-3)


# --------------------------------------------------------------------------
# placement_proposal -> release XY
# --------------------------------------------------------------------------


def test_a_placement_proposal_moves_the_release_inside_the_container():
    cam = birdview()
    tray_centre = np.array([-0.60, 0.20, TABLE_Z])
    empty_half = np.array([-0.68, 0.20, TABLE_Z])
    c = _point_at(cam, empty_half)
    ann = NodeAnnotation(placement_proposal=Box2D(c.x - 0.03, c.y - 0.03, c.x + 0.03, c.y + 0.03))
    target, reason = placement_target_xyz(ann, FakeDet("tray", tray_centre), cam)
    assert target is not None, reason
    assert np.allclose(target[:2], empty_half[:2], atol=3e-3)
    assert target[2] == pytest.approx(TABLE_Z)


def test_a_placement_proposal_off_the_table_is_refused():
    cam = birdview()
    tray_centre = np.array([-0.60, 0.20, TABLE_Z])
    ann = NodeAnnotation(placement_proposal=Box2D(0.01, 0.01, 0.05, 0.05))
    target, reason = placement_target_xyz(ann, FakeDet("tray", tray_centre), cam)
    assert target is None and "keeping the detection" in reason


# --------------------------------------------------------------------------
# Recording
# --------------------------------------------------------------------------


def test_annotations_for_record_is_json_able_and_carries_the_node_path():
    import json

    _instruction, score = robointer_fewshot()
    records = annotations_for_record(score)
    assert [r["node_path"] for r in records] == ["tree/0", "tree/3"]
    assert records[0]["contact_point"] == [0.425, 0.508]
    json.dumps(records)  # would raise on a dataclass or a numpy scalar


def test_resolved_records_are_json_able():
    import json

    from spark_real.planning.robointer import resolve_plan_annotations

    cam = birdview()
    _instruction, score = robointer_fewshot()
    resolved = resolve_plan_annotations(
        score, {"birdview": cam}, TABLE_Z, default_camera="birdview"
    )
    payload = resolved_for_record(resolved)
    assert len(payload) == 2
    assert payload[0]["contact_xyz"] is not None
    json.dumps(payload)


# --------------------------------------------------------------------------
# Prompt payload
# --------------------------------------------------------------------------


def test_the_prompt_section_states_the_coordinate_space_and_the_key():
    assert "__robointer" in ROBOINTER_PROMPT_SECTION
    assert "0 to 1000" in ROBOINTER_PROMPT_SECTION
    for field in (
        "object_box",
        "contact_point",
        "placement_proposal",
        "affordance_box",
        "trace",
        "state_affordance",
        "primitive_skill",
        "subtask",
    ):
        assert field in ROBOINTER_PROMPT_SECTION, field
    # The section must be additive: it never tells the model a field is
    # required, because a plan omitting all of them has to stay legal.
    assert "optional" in ROBOINTER_PROMPT_SECTION


def test_the_context_block_gives_the_model_worked_coordinates():
    dets = [
        FakeDet(
            "knife 1",
            [-0.8, 0.1, TABLE_Z],
            bbox=(64, 96, 320, 240),
            centroid_2d=(192, 168),
            camera="sideview",
        ),
        FakeDet("tray", [-0.6, 0.2, TABLE_Z], bbox=(320, 240, 640, 480), camera="sideview"),
        FakeDet("bowl", [-0.5, 0.0, TABLE_Z], bbox=(0, 0, 10, 10), camera="birdview"),
    ]
    text = build_robointer_context(dets, image_size=(640, 480), camera="sideview")
    assert "640x480" in text and "0..1000" in text
    assert "knife 1: observed_box=[[100, 200], [500, 500]]" in text
    assert "mask_centroid=[300, 350]" in text
    # A detection from another camera must not leak in: its pixels refer to a
    # different image.
    assert "bowl" not in text


def test_the_context_block_is_empty_when_there_is_nothing_to_say():
    assert build_robointer_context(None, None) == ""
    assert build_robointer_context([], image_size=None) == ""


def test_the_context_block_infers_the_image_size_from_a_mask():
    det = FakeDet("knife 1", [-0.8, 0.1, TABLE_Z], bbox=(0, 0, 32, 24))
    det.mask = np.zeros((480, 640), dtype=np.uint8)
    text = build_robointer_context([det])
    assert "640x480" in text


# --------------------------------------------------------------------------
# The overlay the model reads coordinates off
# --------------------------------------------------------------------------


def test_the_grid_composes_on_top_of_the_existing_planner_overlay():
    from spark_real.planning.plan_annotate import annotate_for_planner
    from spark_real.planning.robointer_annotate import draw_annotations, draw_coordinate_grid

    rgb = np.full((480, 640, 3), 90, dtype=np.uint8)
    det = FakeDet(
        "knife 1", [-0.8, 0.1, TABLE_Z], bbox=(100, 200, 300, 260), centroid_2d=(200, 230)
    )
    det.mask = np.zeros((480, 640), dtype=np.uint8)
    det.mask[200:260, 100:300] = 1

    base = annotate_for_planner(rgb, [det])
    gridded = draw_coordinate_grid(base)
    assert gridded.shape == rgb.shape and gridded.dtype == np.uint8
    assert gridded is not base and not np.array_equal(gridded, base)

    ann = NodeAnnotation(
        subtask="grip the handle",
        object_box=Box2D(0.15, 0.4, 0.47, 0.55),
        contact_point=Point2D(0.18, 0.47),
        placement_proposal=Box2D(0.6, 0.6, 0.8, 0.8),
        trace=[Point2D(0.18, 0.47), Point2D(0.4, 0.55), Point2D(0.7, 0.7)],
    )
    drawn = draw_annotations(gridded, [ann])
    assert drawn.shape == rgb.shape
    assert not np.array_equal(drawn, gridded)


def test_the_overlays_are_total_on_degenerate_input():
    from spark_real.planning.robointer_annotate import draw_annotations, draw_coordinate_grid

    tiny = np.zeros((2, 2, 3), dtype=np.uint8)
    assert draw_coordinate_grid(tiny).shape == tiny.shape
    assert draw_annotations(tiny, []).shape == tiny.shape
    gray = np.zeros((8, 8), dtype=np.uint8)
    assert draw_coordinate_grid(gray).shape == gray.shape  # returns it unchanged
    assert draw_annotations(np.zeros((8, 8, 3), np.uint8), [NodeAnnotation()]).shape == (8, 8, 3)

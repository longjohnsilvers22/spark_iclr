"""Offline tests for fingertip-target resolution. Synthetic images only."""

import json

import cv2
import numpy as np
import pytest

from spark_real.perception.fingertip_tags import (
    CONFIDENCE_DEFAULT,
    CONFIDENCE_TABLE,
    CONFIDENCE_TAGS,
    DEFAULT_DICTIONARY,
    SOURCE_DEFAULT,
    SOURCE_TABLE,
    SOURCE_TAGS,
    ApertureTable,
    FingertipTargetResolver,
    resolve_aruco_dictionary,
)

WIDTH, HEIGHT = 640, 480
MARKER_PX = 80


def render_markers(placements, dictionary=DEFAULT_DICTIONARY, side=MARKER_PX):
    """
    White canvas with real AprilTags pasted at known top-left corners.

    ``placements`` is [(tag_id, x0, y0), ...]. Returns (rgb, {id: center}),
    where a marker occupying columns x0..x0+side-1 has its continuous-image
    centre at x0 + side/2 - 0.5.
    """
    dictionary_obj = resolve_aruco_dictionary(dictionary)
    canvas = np.full((HEIGHT, WIDTH), 255, dtype=np.uint8)
    truth = {}
    for tag_id, x0, y0 in placements:
        marker = cv2.aruco.generateImageMarker(dictionary_obj, tag_id, side)
        canvas[y0 : y0 + side, x0 : x0 + side] = marker
        truth[tag_id] = (x0 + side / 2.0 - 0.5, y0 + side / 2.0 - 0.5)
    rgb = cv2.cvtColor(canvas, cv2.COLOR_GRAY2RGB)
    return rgb, truth


def make_resolver(**kwargs):
    return FingertipTargetResolver(WIDTH, HEIGHT, **kwargs)


def test_detects_rendered_tag_centers():
    rgb, truth = render_markers([(0, 60, 300), (1, 420, 300)])
    centers = make_resolver().detect_tag_centers(rgb)
    assert set(centers) == {0, 1}
    for tag_id, (u, v) in truth.items():
        assert centers[tag_id][0] == pytest.approx(u, abs=2.0)
        assert centers[tag_id][1] == pytest.approx(v, abs=2.0)


def test_midpoint_matches_ground_truth():
    rgb, truth = render_markers([(0, 60, 300), (1, 420, 260)])
    expected_u = (truth[0][0] + truth[1][0]) / 2.0
    expected_v = (truth[0][1] + truth[1][1]) / 2.0

    target = make_resolver(tag_ids=(0, 1)).resolve(image=rgb)
    assert target.source == SOURCE_TAGS
    assert target.confidence == CONFIDENCE_TAGS
    assert target.u == pytest.approx(expected_u, abs=2.0)
    assert target.v == pytest.approx(expected_v, abs=2.0)
    assert sorted(target.detail["tag_ids"]) == [0, 1]


def test_midpoint_tracks_the_tags_when_they_move():
    # Closed fingers: tags converge toward the frame centre.
    rgb_open, truth_open = render_markers([(0, 40, 340), (1, 500, 340)])
    rgb_closed, truth_closed = render_markers([(0, 180, 340), (1, 320, 340)])
    resolver = make_resolver(tag_ids=(0, 1))

    open_target = resolver.resolve(image=rgb_open)
    closed_target = resolver.resolve(image=rgb_closed)

    assert open_target.u == pytest.approx((truth_open[0][0] + truth_open[1][0]) / 2.0, abs=2.0)
    assert closed_target.u == pytest.approx(
        (truth_closed[0][0] + truth_closed[1][0]) / 2.0, abs=2.0
    )


def test_any_two_tags_used_when_ids_unconfigured():
    rgb, truth = render_markers([(7, 80, 200), (9, 440, 200)])
    target = make_resolver().resolve(image=rgb)
    assert target.source == SOURCE_TAGS
    assert target.detail["tag_ids"] == [7, 9]
    assert target.u == pytest.approx((truth[7][0] + truth[9][0]) / 2.0, abs=2.0)


def test_configured_ids_ignore_a_stray_tag():
    rgb, truth = render_markers([(0, 60, 300), (1, 420, 300), (5, 260, 40)])
    target = make_resolver(tag_ids=(0, 1)).resolve(image=rgb)
    assert sorted(target.detail["tag_ids"]) == [0, 1]
    assert target.u == pytest.approx((truth[0][0] + truth[1][0]) / 2.0, abs=2.0)


def test_no_tags_falls_back_to_default_pixel_and_flags_it():
    blank = np.full((HEIGHT, WIDTH, 3), 255, dtype=np.uint8)
    target = make_resolver().resolve(image=blank)
    assert target.source == SOURCE_DEFAULT
    assert target.confidence == CONFIDENCE_DEFAULT
    assert target.pixel == (WIDTH / 2.0, HEIGHT / 2.0)
    assert "fallback_reason" in target.detail


def test_only_one_tag_falls_back():
    rgb, _ = render_markers([(0, 60, 300)])
    target = make_resolver(tag_ids=(0, 1)).resolve(image=rgb)
    assert target.source == SOURCE_DEFAULT
    assert "0" in target.detail["fallback_reason"]


def test_no_tags_falls_back_to_table_when_aperture_known():
    blank = np.full((HEIGHT, WIDTH, 3), 255, dtype=np.uint8)
    table = ApertureTable.from_entries(
        [{"aperture": 0.0, "u": 300.0, "v": 400.0}, {"aperture": 1.0, "u": 320.0, "v": 300.0}]
    )
    target = make_resolver(table=table).resolve(image=blank, aperture=0.5)
    assert target.source == SOURCE_TABLE
    assert target.confidence == CONFIDENCE_TABLE
    assert target.u == pytest.approx(310.0)
    assert target.v == pytest.approx(350.0)
    assert target.detail["aperture"] == pytest.approx(0.5)


def test_table_beats_default_but_tags_beat_table():
    rgb, truth = render_markers([(0, 60, 300), (1, 420, 300)])
    table = ApertureTable.from_entries([(0.0, 111.0, 222.0)])
    target = make_resolver(table=table, tag_ids=(0, 1)).resolve(image=rgb, aperture=0.0)
    assert target.source == SOURCE_TAGS
    assert target.u == pytest.approx((truth[0][0] + truth[1][0]) / 2.0, abs=2.0)


def test_no_image_at_all_uses_table():
    table = ApertureTable.from_entries([(0.0, 111.0, 222.0), (1.0, 131.0, 262.0)])
    target = make_resolver(table=table).resolve(aperture=1.0)
    assert target.source == SOURCE_TABLE
    assert target.pixel == pytest.approx((131.0, 262.0))


def test_table_interpolates_and_clamps():
    table = ApertureTable.from_entries(
        [(0.0, 100.0, 200.0), (0.5, 120.0, 260.0), (1.0, 130.0, 300.0)]
    )
    assert len(table) == 3
    assert table.lookup(0.25) == pytest.approx((110.0, 230.0))
    assert table.lookup(-5.0) == pytest.approx((100.0, 200.0))
    assert table.lookup(9.0) == pytest.approx((130.0, 300.0))
    # Unsorted input must still interpolate correctly.
    shuffled = ApertureTable.from_entries(
        [(1.0, 130.0, 300.0), (0.0, 100.0, 200.0), (0.5, 120.0, 260.0)]
    )
    assert shuffled.lookup(0.25) == pytest.approx((110.0, 230.0))


def test_table_load_json_and_yaml(tmp_path):
    entries = [
        {"aperture": 0.0, "u": 300.0, "v": 400.0},
        {"aperture": 1.0, "u": 320.0, "v": 300.0},
    ]
    json_path = tmp_path / "fingertips.json"
    json_path.write_text(json.dumps({"fingertip_table": entries}))
    yaml_path = tmp_path / "fingertips.yaml"
    yaml_path.write_text(
        "fingertip_table:\n"
        "  - {aperture: 0.0, u: 300.0, v: 400.0}\n"
        "  - {aperture: 1.0, u: 320.0, v: 300.0}\n"
    )
    for path in (json_path, yaml_path):
        table = ApertureTable.load(path)
        assert table.lookup(0.5) == pytest.approx((310.0, 350.0))


def test_from_config_returns_none_without_entries():
    assert ApertureTable.from_config(None) is None
    assert ApertureTable.from_config({}) is None
    assert ApertureTable.from_config({"fingertip_table": []}) is None
    table = ApertureTable.from_config({"fingertip_table": [{"aperture": 0.0, "u": 1.0, "v": 2.0}]})
    assert table.lookup(0.0) == pytest.approx((1.0, 2.0))


def test_from_calibration_defaults_to_principal_point():
    from spark_real.calibration.model import CameraCalibration

    cal = CameraCalibration(
        name="wrist", width=640, height=480, fx=608.26, fy=607.97, cx=327.96, cy=245.63
    )
    resolver = FingertipTargetResolver.from_calibration(cal)
    target = resolver.resolve()
    assert target.source == SOURCE_DEFAULT
    assert target.pixel == pytest.approx((327.96, 245.63))
    assert resolver.width == 640 and resolver.height == 480


def test_explicit_default_pixel_wins():
    target = make_resolver(default_pixel=(11.0, 22.0)).resolve()
    assert target.pixel == (11.0, 22.0)
    assert target.as_array().tolist() == [11.0, 22.0]


def test_bad_configuration_rejected():
    with pytest.raises(ValueError):
        make_resolver(dictionary="DICT_NOT_A_THING")
    with pytest.raises(ValueError):
        make_resolver(tag_ids=(0, 1, 2))
    with pytest.raises(ValueError):
        ApertureTable.from_entries([])


def test_alternate_dictionary_round_trips():
    rgb, truth = render_markers([(0, 60, 300), (1, 420, 300)], dictionary="DICT_APRILTAG_25h9")
    resolver = make_resolver(dictionary="DICT_APRILTAG_25h9", tag_ids=(0, 1))
    target = resolver.resolve(image=rgb)
    assert target.source == SOURCE_TAGS
    assert target.u == pytest.approx((truth[0][0] + truth[1][0]) / 2.0, abs=2.0)


def test_grayscale_image_accepted():
    rgb, truth = render_markers([(0, 60, 300), (1, 420, 300)])
    gray = cv2.cvtColor(rgb, cv2.COLOR_RGB2GRAY)
    centers = make_resolver().detect_tag_centers(gray)
    assert set(centers) == {0, 1}

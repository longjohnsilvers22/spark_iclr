"""
Tests for the provider-agnostic annotation interface.

The round-trip conversion tests are the point of this file: the ER2
integration notes rank a silent [y, x] transpose as the number-one burn
risk, so every converter is exercised against an ASYMMETRIC frame
(640x480) where a swap cannot cancel out.
"""

import numpy as np
import pytest

from spark_real.perception.annotations import (
    ER2_SCALE,
    Annotation,
    HumanUIProvider,
    box_from_er2,
    box_to_er2,
    get_provider,
    point_from_er2,
    point_from_molmo,
    point_to_er2,
    trace_from_er2,
    trace_to_waypoints_3d,
    waypoints_to_move_nodes,
)
from spark_real.perception.providers.molmo import parse_molmo_points

# 640 wide, 480 tall: u=600 is legal, v=600 is not - a transpose is loud.
W, H = 640, 480


class TestAnnotationBasics:
    def test_kind_arity_enforced(self):
        with pytest.raises(ValueError):
            Annotation(kind="point", points=[(0.1, 0.2), (0.3, 0.4)])
        with pytest.raises(ValueError):
            Annotation(kind="box", points=[(0.1, 0.2)])
        with pytest.raises(ValueError):
            Annotation(kind="trace", points=[(0.1, 0.2)])
        with pytest.raises(ValueError):
            Annotation(kind="squiggle", points=[(0.1, 0.2)])

    def test_pixel_round_trip_asymmetric(self):
        ann = Annotation.from_pixels("point", [(600.0, 120.0)], (W, H))
        (x, y), = ann.points
        assert x == pytest.approx(600.0 / W)
        assert y == pytest.approx(120.0 / H)
        (u, v), = ann.to_pixels((W, H))
        assert (u, v) == (pytest.approx(600.0), pytest.approx(120.0))

    def test_box_xyxy_sorts_corners(self):
        ann = Annotation(kind="box", points=[(0.8, 0.9), (0.2, 0.1)])
        assert ann.box_xyxy() == (0.2, 0.1, 0.8, 0.9)

    def test_clamp_and_bounds(self):
        ann = Annotation(kind="point", points=[(1.1, -0.2)])
        assert not ann.in_bounds()
        clamped = ann.clamped()
        assert clamped.points == [(1.0, 0.0)]
        assert clamped.in_bounds()


class TestER2Conversions:
    """[y, x] on 0-1000 <-> internal (x, y) normalized."""

    def test_point_round_trip(self):
        # A point near the RIGHT edge, upper part of the frame.
        internal = (0.9, 0.25)
        er2 = point_to_er2(internal)
        assert er2 == [250, 900]  # y first
        back = point_from_er2(er2)
        assert back == (pytest.approx(0.9), pytest.approx(0.25))

    def test_transpose_is_caught_in_pixels(self):
        # ER2 says [y=250, x=900]: on a 640x480 frame that is pixel
        # (576, 120). A transposed read would claim (160, 432) - assert
        # the exact untransposed pixel so a swap cannot sneak through.
        ann = Annotation(kind="point", points=[point_from_er2([250, 900])])
        (u, v), = ann.to_pixels((W, H))
        assert u == pytest.approx(0.9 * W)   # 576
        assert v == pytest.approx(0.25 * H)  # 120
        assert not (u == pytest.approx(0.25 * W) and v == pytest.approx(0.9 * H))

    def test_box_round_trip(self):
        # ymin, xmin, ymax, xmax
        pts = box_from_er2([100, 200, 300, 400])
        ann = Annotation(kind="box", points=pts)
        assert ann.box_xyxy() == (
            pytest.approx(0.2), pytest.approx(0.1),
            pytest.approx(0.4), pytest.approx(0.3))
        assert box_to_er2(ann) == [100, 200, 300, 400]

    def test_trace_preserves_order_and_axes(self):
        yx = [[500, 100], [500, 500], [900, 500]]
        pts = trace_from_er2(yx)
        ann = Annotation(kind="trace", points=pts)
        px = ann.to_pixels((W, H))
        # Leg 1 is horizontal (x moves, y fixed), leg 2 vertical.
        assert px[0][1] == pytest.approx(px[1][1])
        assert px[1][0] == pytest.approx(px[2][0])
        assert px[1][0] > px[0][0]
        assert px[2][1] > px[1][1]

    def test_permille_quantization_bounded(self):
        # int rounding on the 0-1000 grid loses at most half a permille.
        internal = (0.12345, 0.67891)
        back = point_from_er2(point_to_er2(internal))
        assert abs(back[0] - internal[0]) <= 0.5 / ER2_SCALE
        assert abs(back[1] - internal[1]) <= 0.5 / ER2_SCALE


class TestMolmoConversions:
    def test_molmo_is_x_first(self):
        # Molmo (x=90, y=25) on the 0-100 grid = same physical point as
        # the ER2 [250, 900] case above; both must land identically.
        assert point_from_molmo((90.0, 25.0), 100.0) == \
            (pytest.approx(0.9), pytest.approx(0.25))
        assert point_from_molmo((900.0, 250.0), 1000.0) == \
            (pytest.approx(0.9), pytest.approx(0.25))

    def test_parse_molmo2_coords_format(self):
        text = '<points coords="1 0 900 250 1 100 500">two mugs</points>'
        pts, scale = parse_molmo_points(text)
        assert scale == 1000.0
        assert pts == [(900.0, 250.0), (100.0, 500.0)]

    def test_parse_molmo1_point_tag(self):
        pts, scale = parse_molmo_points('Sure: <point x="90.0" y="25.0">mug')
        assert scale == 100.0
        assert pts == [(90.0, 25.0)]

    def test_parse_legacy_points_tag(self):
        pts, scale = parse_molmo_points(
            '<points x1="10" y1="20" x2="30" y2="40">pair</points>')
        assert scale == 100.0
        assert pts == [(10.0, 20.0), (30.0, 40.0)]

    def test_parse_prose_returns_empty(self):
        pts, _ = parse_molmo_points("I cannot see that object.")
        assert pts == []


class TestER2ReplyParsing:
    """ER2Provider._parse without any network."""

    def _provider(self):
        from spark_real.perception.providers.er2 import ER2Provider
        return ER2Provider()

    def test_point_reply(self):
        anns = self._provider()._parse(
            '```json\n[{"point": [250, 900], "label": "mug"}]\n```',
            "point", "mug")
        assert len(anns) == 1
        assert anns[0].kind == "point"
        assert anns[0].label == "mug"
        assert anns[0].point == (pytest.approx(0.9), pytest.approx(0.25))

    def test_box_reply(self):
        anns = self._provider()._parse(
            '[{"box_2d": [100, 200, 300, 400], "label": "tray"}]',
            "box", "tray")
        assert len(anns) == 1
        assert anns[0].box_xyxy() == (
            pytest.approx(0.2), pytest.approx(0.1),
            pytest.approx(0.4), pytest.approx(0.3))

    def test_trace_reply(self):
        anns = self._provider()._parse(
            '[{"point": [500, 100], "label": "step 1"},'
            ' {"point": [500, 500], "label": "step 2"},'
            ' {"point": [900, 500], "label": "step 3"}]',
            "trace", "move the mug to the sink")
        assert len(anns) == 1
        assert anns[0].kind == "trace"
        assert len(anns[0].points) == 3

    def test_junk_reply_fails_open(self):
        assert self._provider()._parse("no objects here", "point", "mug") == []
        assert self._provider()._parse('[{"pt": [1, 2]}]', "point", "mug") == []


class TestTraceToWaypoints:
    def _cam(self):
        # Camera 1 m above the origin looking straight down. MuJoCo cameras
        # view along their own -Z, so "looking down" with axes aligned to
        # the world is simply the identity world-from-camera rotation.
        cam_pos = np.array([0.0, 0.0, 1.0])
        cam_mat = np.eye(3)
        return cam_pos, cam_mat

    def test_backprojection_at_object_plane(self):
        cam_pos, cam_mat = self._cam()
        h, w, fovy = 480, 640, 90.0
        depth = np.full((h, w), 0.5, dtype=np.float32)  # object plane 0.5 m
        ann = Annotation(kind="trace",
                         points=[(0.5, 0.5), (0.75, 0.5)])  # centre -> right
        wps = trace_to_waypoints_3d(ann, depth, cam_pos, cam_mat, fovy)
        assert len(wps) == 2
        # Centre pixel back-projects to straight below the camera at z=0.5.
        np.testing.assert_allclose(wps[0], [0.0, 0.0, 0.5], atol=1e-6)
        # Right of centre moves +x in camera frame -> +x world; y unchanged.
        f = h / (2 * np.tan(np.deg2rad(fovy) / 2))
        exp_x = (0.75 * w - w / 2) * 0.5 / f
        np.testing.assert_allclose(wps[1], [exp_x, 0.0, 0.5], atol=1e-6)

    def test_plane_depth_ignores_background(self):
        cam_pos, cam_mat = self._cam()
        h, w = 480, 640
        depth = np.full((h, w), 2.0, dtype=np.float32)  # far background
        depth[230:250, 310:330] = 0.5                    # object at centre
        ann = Annotation(kind="trace", points=[(0.5, 0.5), (0.9, 0.5)])
        wps = trace_to_waypoints_3d(ann, depth, cam_pos, cam_mat, 90.0)
        # BOTH points sit on the 0.5 m object plane even though the second
        # pixel's own depth reads the 2.0 m background.
        assert wps[0][2] == pytest.approx(0.5)
        assert wps[1][2] == pytest.approx(0.5)

    def test_no_depth_fails_open(self):
        cam_pos, cam_mat = self._cam()
        depth = np.zeros((480, 640), dtype=np.float32)
        ann = Annotation(kind="trace", points=[(0.5, 0.5), (0.6, 0.5)])
        assert trace_to_waypoints_3d(ann, depth, cam_pos, cam_mat, 90.0) == []

    def test_waypoints_to_move_nodes(self):
        wps = [np.array([0.0, 0.0, 0.5]),
               np.array([0.10, 0.0, 0.5]),
               np.array([0.10, 0.002, 0.5]),   # sub-cm jitter: merged
               np.array([0.10, 0.20, 0.5])]
        nodes = waypoints_to_move_nodes(wps)
        assert [n["type"] for n in nodes] == ["move_relative", "move_relative"]
        assert nodes[0]["params"] == {"dx": 0.1, "dy": 0.0, "dz": 0.0}
        # The merged jitter point is absorbed into the following delta.
        assert nodes[1]["params"]["dy"] == pytest.approx(0.2, abs=1e-9)

    def test_move_nodes_from_start_pose(self):
        nodes = waypoints_to_move_nodes(
            [np.array([0.3, 0.0, 0.5])], start_xyz=np.array([0.0, 0.0, 0.5]))
        assert len(nodes) == 1
        assert nodes[0]["params"]["dx"] == pytest.approx(0.3)


class TestProviderRegistry:
    def test_human_provider(self):
        p = get_provider("human")
        assert isinstance(p, HumanUIProvider)
        assert p.available()
        assert p.annotate(np.zeros((4, 4, 3), np.uint8), "anything") == []
        ann = HumanUIProvider.from_click(600.0, 120.0, (W, H), label="mug")
        assert ann.provider == "human"
        assert ann.point == (pytest.approx(600.0 / W), pytest.approx(120.0 / H))

    def test_unknown_provider_fails_open(self):
        assert get_provider("nonesuch") is None

    def test_er2_and_molmo_constructible(self):
        # Construction must never require a key / GPU; availability may
        # legitimately be False on CI.
        for name in ("er2", "molmo"):
            p = get_provider(name)
            assert p is not None
            assert isinstance(p.available(), bool)

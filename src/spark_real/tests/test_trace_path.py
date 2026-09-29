"""Operator-drawn approach paths: 2D trace -> 3D waypoints -> transport lead.

The tap perception/annotations.py has described since it was written ("the
executor's waypoint path (a trace back-projected at the object's depth
plane)") and nothing ever wired. Built 2026-08-24 because SPARK has no lateral
motion planning: on 2026-08-21 a straight-line approach swept the open jaws
through a socket standing beside the target.
"""

import numpy as np
import pytest

from spark_real.calibration.model import CameraCalibration
from spark_real.perception.annotations import Annotation
from spark_real.perception.trace_path import trace_to_waypoints, waypoints_to_lead


def _overhead_cal(height_m=1.5):
    """A camera looking straight DOWN from height_m, centred over the origin."""
    T = np.eye(4)
    # camera +z (forward) -> world -z; +y (down in image) -> world +y stays,
    # so the rotation flips z and x: a standard overhead OpenCV pose.
    T[:3, :3] = np.array([[1.0, 0.0, 0.0], [0.0, -1.0, 0.0], [0.0, 0.0, -1.0]])
    T[:3, 3] = [0.0, 0.0, height_m]
    return CameraCalibration(
        name="birdview", width=1280, height=720,
        fx=900.0, fy=900.0, cx=640.0, cy=360.0, extrinsic=T,
    )


def _trace(points):
    return Annotation(kind="trace", points=points, provider="operator", label="t")


def test_the_image_centre_lands_under_the_camera():
    cal = _overhead_cal()
    wp = trace_to_waypoints(_trace([(0.5, 0.5), (0.5, 0.9)]), cal, z_plane=-0.25)
    assert len(wp) >= 2
    assert wp[0][0] == pytest.approx(0.0, abs=1e-6)
    assert wp[0][1] == pytest.approx(0.0, abs=1e-6)


def test_every_waypoint_sits_on_the_requested_plane():
    cal = _overhead_cal()
    wp = trace_to_waypoints(
        _trace([(0.2, 0.2), (0.5, 0.4), (0.8, 0.8)]), cal, z_plane=-0.13
    )
    assert wp, "nothing projected"
    for p in wp:
        assert p[2] == pytest.approx(-0.13, abs=1e-9)


def test_a_dense_scribble_is_decimated_but_keeps_its_ends():
    cal = _overhead_cal()
    dense = [(0.2 + 0.006 * i, 0.5) for i in range(100)]
    wp = trace_to_waypoints(_trace(dense), cal, z_plane=-0.25, max_points=6)
    assert 2 <= len(wp) <= 6
    ends = trace_to_waypoints(
        _trace([dense[0], dense[-1]]), cal, z_plane=-0.25, max_points=6
    )
    # The decimated path still starts and finishes where the operator drew.
    assert wp[0][0] == pytest.approx(ends[0][0], abs=0.02)
    assert wp[-1][0] == pytest.approx(ends[-1][0], abs=0.02)


def test_points_whose_ray_misses_the_plane_are_dropped_not_guessed():
    cal = _overhead_cal()
    # A plane ABOVE the camera: every downward ray points away from it.
    wp = trace_to_waypoints(_trace([(0.4, 0.4), (0.6, 0.6)]), cal, z_plane=2.5)
    assert wp == []


def test_the_lead_drops_the_final_point_because_the_transport_owns_it():
    pts = [[0.1, 0.2, 0.3], [0.4, 0.5, 0.3], [0.7, 0.8, 0.3]]
    rows = waypoints_to_lead(pts, [2.2, 2.2, 0.0])
    assert len(rows) == 2  # last point is the transport's own target
    assert list(rows[0][0]) == pytest.approx([0.1, 0.2, 0.3])
    assert rows[0][1] == [2.2, 2.2, 0.0]


def test_a_two_point_path_yields_exactly_one_lead_row():
    rows = waypoints_to_lead([[0.0, 0.0, 0.1], [1.0, 1.0, 0.1]], [0, 0, 0])
    assert len(rows) == 1


def test_a_trace_needs_at_least_two_points():
    with pytest.raises(ValueError):
        _trace([(0.5, 0.5)])

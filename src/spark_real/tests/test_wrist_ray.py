"""Offline geometry tests for RGB-only wrist refinement. No robot/camera/SAM3."""

import numpy as np
import pytest
from scipy.spatial.transform import Rotation

from spark_real.calibration import CameraCalibration
from spark_real.control import executor_motion
from spark_real.control.executor_motion import MotionMixin
from spark_real.perception.wrist_ray import (
    intersect_ray_plane,
    pixel_ray_base,
    pixel_to_plane_point,
    project_point_to_pixel,
)

# D435i-ish 640x480 color intrinsics.
INTR = {"fx": 600.0, "fy": 600.0, "cx": 320.0, "cy": 240.0}


def looking_down(cam_pos):
    """Camera at cam_pos with optical +Z along world -Z (straight down).

    Camera +X -> world +X, camera +Y -> world -Y, so the frame stays
    right-handed.
    """
    T = np.eye(4)
    T[:3, :3] = np.array([[1.0, 0.0, 0.0], [0.0, -1.0, 0.0], [0.0, 0.0, -1.0]])
    T[:3, 3] = cam_pos
    return T


def test_principal_point_maps_straight_down():
    T = looking_down([0.4, -0.1, 0.5])
    hit = pixel_to_plane_point(INTR["cx"], INTR["cy"], INTR, T, -0.276)
    assert hit is not None
    np.testing.assert_allclose(hit, [0.4, -0.1, -0.276], atol=1e-12)


def test_offcentre_pixel_lateral_offset():
    # Height above plane h=0.776; a pixel du right of centre subtends
    # du/fx radians of tangent, so the lateral offset is h*du/fx exactly.
    T = looking_down([0.4, -0.1, 0.5])
    z_plane = -0.276
    h = 0.5 - z_plane
    du, dv = 80.0, -45.0
    hit = pixel_to_plane_point(
        INTR["cx"] + du, INTR["cy"] + dv, INTR, T, z_plane
    )
    assert hit is not None
    # Camera +Y is world -Y under this rotation, hence the sign on dy.
    expected = [0.4 + h * du / INTR["fx"], -0.1 - h * dv / INTR["fy"], z_plane]
    np.testing.assert_allclose(hit, expected, atol=1e-12)
    assert abs(np.linalg.norm(hit[:2] - np.array(expected[:2]))) < 1e-6


def test_tilted_camera_matches_independent_computation():
    # Down-looking frame pitched 20 deg about world X.
    tilt = np.deg2rad(20.0)
    T = looking_down([0.3, 0.0, 0.6])
    T[:3, :3] = Rotation.from_euler("x", tilt).as_matrix() @ T[:3, :3]
    z_plane = 0.0
    u, v = INTR["cx"], INTR["cy"] + 60.0

    hit = pixel_to_plane_point(u, v, INTR, T, z_plane)
    assert hit is not None

    # Independent: build the camera-frame ray, rotate, walk to z=0 by hand.
    d_cam = np.array([0.0, 60.0 / INTR["fy"], 1.0])
    d_world = T[:3, :3] @ d_cam
    t = (z_plane - 0.6) / d_world[2]
    expected = np.array([0.3, 0.0, 0.6]) + t * d_world
    np.testing.assert_allclose(hit, expected, atol=1e-12)


def test_ray_parallel_to_plane_returns_none():
    origin = np.array([0.3, 0.0, 0.4])
    assert intersect_ray_plane(origin, [1.0, 0.0, 0.0], 0.0) is None
    # Also a horizontal camera: optical axis along world +X.
    T = np.eye(4)
    T[:3, :3] = np.array([[0.0, 0.0, 1.0], [-1.0, 0.0, 0.0], [0.0, -1.0, 0.0]])
    T[:3, 3] = origin
    assert pixel_to_plane_point(INTR["cx"], INTR["cy"], INTR, T, 0.0) is None


def test_ray_pointing_away_returns_none():
    # Camera below the plane but still looking down: the plane is behind it.
    origin = np.array([0.3, 0.0, -0.5])
    assert intersect_ray_plane(origin, [0.0, 0.0, -1.0], 0.0) is None
    T = looking_down(origin)
    assert pixel_to_plane_point(INTR["cx"], INTR["cy"], INTR, T, 0.0) is None


def test_direction_is_unit_length():
    T = looking_down([0.4, -0.1, 0.5])
    _, d = pixel_ray_base(500.0, 100.0, INTR, T)
    assert abs(np.linalg.norm(d) - 1.0) < 1e-12


@pytest.mark.parametrize(
    "point",
    [
        [0.40, -0.10, -0.276],
        [0.55, 0.12, -0.276],
        [0.22, -0.31, -0.276],
    ],
)
def test_round_trip_project_then_backproject(point):
    # Tilted+yawed camera so the round trip is not accidentally axis-aligned.
    T = looking_down([0.42, -0.05, 0.38])
    T[:3, :3] = (
        Rotation.from_euler("xyz", [0.12, -0.08, 0.35]).as_matrix() @ T[:3, :3]
    )
    px = project_point_to_pixel(point, INTR, T)
    assert px is not None
    hit = pixel_to_plane_point(px[0], px[1], INTR, T, point[2])
    assert hit is not None
    np.testing.assert_allclose(hit, point, atol=1e-9)


def test_accepts_calibration_object_intrinsics():
    cal = wrist_cal(looking_down([0.4, -0.1, 0.5]))
    hit = pixel_to_plane_point(cal.cx, cal.cy, cal, cal.extrinsic, -0.276)
    np.testing.assert_allclose(hit, [0.4, -0.1, -0.276], atol=1e-12)


# --- executor wiring (still offline: no robot, camera or SAM3) ---


def wrist_cal(extrinsic):
    cal = CameraCalibration(
        name="wrist", width=640, height=480, fx=600.0, fy=600.0, cx=320.0, cy=240.0
    )
    cal.extrinsic = extrinsic
    return cal


class FakeExecutor(MotionMixin):
    """Just enough state for the wrist-refine helpers."""

    TABLE_Z_FLOOR = -0.276

    def __init__(self, mode=None):
        self._pipeline = None
        self._wrist_refine_mode_cfg = mode

    def _robot_family(self):
        return "ur10e"


def test_mode_defaults_to_ray_plane(monkeypatch):
    monkeypatch.delenv("SPARK_WRIST_REFINE_MODE", raising=False)
    assert FakeExecutor()._wrist_refine_mode() == "ray_plane"


def test_mode_from_config_and_env(monkeypatch):
    monkeypatch.delenv("SPARK_WRIST_REFINE_MODE", raising=False)
    assert FakeExecutor("depth")._wrist_refine_mode() == "depth"
    assert FakeExecutor("off")._wrist_refine_mode() == "off"
    # Unknown values fall back rather than disabling refinement silently.
    assert FakeExecutor("banana")._wrist_refine_mode() == "ray_plane"
    monkeypatch.setenv("SPARK_WRIST_REFINE_MODE", "depth")
    assert FakeExecutor("ray_plane")._wrist_refine_mode() == "depth"


def test_plane_z_priority(monkeypatch):
    monkeypatch.setattr(executor_motion, "load_table_plane", lambda fam: None)
    exe = FakeExecutor()
    coarse = np.array([0.4, -0.1, -0.20])
    # Explicit object prior wins.
    assert exe._wrist_plane_z(coarse, 0.5, z_plane=-0.25) == (-0.25, "object_prior")
    # No prior -> the coarse target's own Z.
    assert exe._wrist_plane_z(coarse, 0.5) == (-0.20, "coarse_target")
    # Prior level with the camera is rejected: it would graze the plane.
    assert exe._wrist_plane_z(np.array([0.4, -0.1, 0.5]), 0.5) == (
        -0.276,
        "table_z_floor",
    )
    # search_keypoint hands us its own TCP waypoint as the coarse target. The
    # wrist cam sits ABOVE the TCP, so the camera test alone would accept it;
    # the TCP test is what sends this to the table plane.
    assert exe._wrist_plane_z(np.array([0.4, -0.1, 0.35]), 0.47, tcp_z=0.35) == (
        -0.276,
        "table_z_floor",
    )
    # A genuine hover above an object still uses the object's Z.
    assert exe._wrist_plane_z(coarse, 0.57, tcp_z=0.45) == (-0.20, "coarse_target")


def test_plane_z_uses_table_calibration(monkeypatch):
    monkeypatch.setattr(
        executor_motion, "load_table_plane", lambda fam: {"surface_z": -0.2801}
    )
    exe = FakeExecutor()
    z, src = exe._wrist_plane_z(np.array([0.4, -0.1, 0.5]), 0.5)
    assert src == "table_cal"
    assert abs(z - (-0.2801)) < 1e-12


def test_ray_path_works_with_no_depth(monkeypatch):
    """The whole point: RGB-only refinement while the D435i is color-only."""
    monkeypatch.setattr(executor_motion, "load_table_plane", lambda fam: None)
    exe = FakeExecutor()
    cal = wrist_cal(looking_down([0.42, -0.05, 0.30]))
    coarse = np.array([0.40, -0.05, -0.276])

    pos, detail = exe._wrist_pos_from_ray(cal, 380.0, 240.0, coarse, None, "spoon")
    assert pos is not None
    h = 0.30 - (-0.276)
    np.testing.assert_allclose(
        pos, [0.42 + h * 60.0 / 600.0, -0.05, -0.276], atol=1e-12
    )
    assert "plane_z=-0.276" in detail

    # The depth path with no depth stream degrades to None, it does not raise.
    assert exe._wrist_pos_from_depth(cal, None, None, 380.0, 240.0, "spoon") == (
        None,
        None,
    )

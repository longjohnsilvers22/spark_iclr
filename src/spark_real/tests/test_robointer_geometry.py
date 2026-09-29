"""
2D <-> base-frame conversions for RoboInter representations, on synthetic
geometry with an exactly known answer.

No camera, no calibration file, no robot. Two synthetic rigs stand in for the
real ones: an overhead camera looking straight down at the table, and a
sideview camera looking across it -- which is the pair the rig actually has,
and the pair whose coordinates must never be mixed up.
"""

from __future__ import annotations

import numpy as np
import pytest

from spark_real.planning.robointer import (
    Box2D,
    NodeAnnotation,
    Point2D,
    resolve_annotation,
    resolve_plan_annotations,
)
from spark_real.planning.robointer_geometry import (
    CameraModel,
    box_to_sam3_prompt,
    sample_depth,
)

TABLE_Z = -0.28  # metres, matches the corpus grasp height for flat objects


def _look_at(eye, target, up=(0.0, 0.0, 1.0)) -> np.ndarray:
    """OpenCV-convention camera->base extrinsic looking from eye at target."""
    eye = np.asarray(eye, dtype=float)
    fwd = np.asarray(target, dtype=float) - eye
    fwd /= np.linalg.norm(fwd)
    right = np.cross(fwd, np.asarray(up, dtype=float))
    right /= np.linalg.norm(right)
    down = np.cross(fwd, right)
    ext = np.eye(4)
    ext[:3, 0] = right  # +x_cam
    ext[:3, 1] = down  # +y_cam (image rows increase downward)
    ext[:3, 2] = fwd  # +z_cam
    ext[:3, 3] = eye
    return ext


def birdview() -> CameraModel:
    # 1.3 m above the table, straight down, +x_cam along +X_base.
    ext = np.eye(4)
    ext[:3, :3] = np.diag([1.0, -1.0, -1.0])
    ext[:3, 3] = [-0.80, 0.10, 1.02]
    return CameraModel(600.0, 600.0, 320.0, 240.0, 640, 480, ext, name="birdview")


def sideview() -> CameraModel:
    return CameraModel(
        615.0,
        615.0,
        320.0,
        240.0,
        640,
        480,
        _look_at((-0.10, 0.10, 0.35), (-0.85, 0.10, TABLE_Z)),
        name="sideview",
    )


@pytest.mark.parametrize("cam_fn", [birdview, sideview], ids=["birdview", "sideview"])
def test_project_then_backproject_with_true_depth_is_exact(cam_fn):
    cam = cam_fn()
    for point in ([-0.80, 0.10, TABLE_Z], [-0.70, -0.05, -0.15], [-0.95, 0.25, TABLE_Z]):
        uv = cam.base_to_pixel(point)
        assert uv is not None
        depth = float((cam.rotation.T @ (np.asarray(point) - cam.position))[2])
        back = cam.pixel_to_base(uv[0], uv[1], depth)
        assert np.allclose(back, point, atol=1e-9)


@pytest.mark.parametrize("cam_fn", [birdview, sideview], ids=["birdview", "sideview"])
def test_a_point_on_the_table_needs_no_depth_at_all(cam_fn):
    # The conversion that makes RoboInter usable on RGB-only frames.
    cam = cam_fn()
    for point in ([-0.80, 0.10, TABLE_Z], [-0.70, -0.05, TABLE_Z], [-0.95, 0.25, TABLE_Z]):
        u, v = cam.base_to_pixel(point)
        back = cam.pixel_to_base_on_plane(u, v, TABLE_Z)
        assert back is not None
        assert np.allclose(back, point, atol=1e-9)


def test_the_plane_assumption_is_what_costs_you_when_it_is_wrong():
    # An object 6 cm tall, projected as if it were flat on the table: the
    # error is a lateral slide, and it is bounded by the viewing angle. This
    # is the reason contact_target_xyz keeps perception's Z.
    cam = sideview()
    true_point = np.array([-0.80, 0.10, TABLE_Z + 0.06])
    u, v = cam.base_to_pixel(true_point)
    on_table = cam.pixel_to_base_on_plane(u, v, TABLE_Z)
    assert on_table[2] == pytest.approx(TABLE_Z)
    err = float(np.linalg.norm(on_table[:2] - true_point[:2]))
    assert 0.01 < err < 0.20, err
    # Overhead, the same 6 cm costs almost nothing laterally.
    cam2 = birdview()
    u2, v2 = cam2.base_to_pixel(true_point)
    err2 = float(np.linalg.norm(cam2.pixel_to_base_on_plane(u2, v2, TABLE_Z)[:2] - true_point[:2]))
    assert err2 < 0.01, err2


def test_a_ray_parallel_to_the_plane_returns_none_rather_than_a_guess():
    # A camera looking exactly horizontally: its principal ray never meets a
    # horizontal plane. Returning any point here would be an invented target.
    ext = _look_at((-0.10, 0.10, 0.0), (-0.90, 0.10, 0.0))
    cam = CameraModel(615.0, 615.0, 320.0, 240.0, 640, 480, ext, name="level")
    assert cam.pixel_to_base_on_plane(320.0, 240.0, 0.0) is None


def test_a_plane_behind_the_camera_returns_none():
    cam = birdview()
    # z above the camera: the downward ray never reaches it going forward.
    assert cam.pixel_to_base_on_plane(320.0, 240.0, 2.0) is None


def test_a_point_behind_the_camera_does_not_project():
    cam = birdview()
    assert cam.base_to_pixel([-0.80, 0.10, 2.0]) is None


def test_an_uncalibrated_camera_is_refused():
    class Cal:
        name, width, height = "wrist", 640, 480
        fx = fy = 600.0
        cx, cy = 320.0, 240.0
        extrinsic = np.eye(4)

    with pytest.raises(ValueError, match="uncalibrated"):
        CameraModel.from_calibration(Cal(), name="wrist")


def test_from_calibration_matches_a_hand_built_model():
    from spark_real.calibration.model import CameraCalibration

    ext = birdview().extrinsic
    cal = CameraCalibration(
        name="birdview",
        width=640,
        height=480,
        fx=600.0,
        fy=600.0,
        cx=320.0,
        cy=240.0,
        extrinsic=ext,
    )
    cam = CameraModel.from_calibration(cal)
    assert np.allclose(cam.extrinsic, ext)
    assert cam.image_size == (640, 480)
    assert np.allclose(
        cam.pixel_to_base_on_plane(300, 200, TABLE_Z),
        birdview().pixel_to_base_on_plane(300, 200, TABLE_Z),
    )


def test_bad_intrinsics_are_rejected_at_construction():
    with pytest.raises(ValueError):
        CameraModel(0.0, 600.0, 320.0, 240.0, 640, 480, np.eye(4))
    with pytest.raises(ValueError):
        CameraModel(600.0, 600.0, 320.0, 240.0, 640, 480, np.eye(3))


def test_sample_depth_is_a_window_not_a_pixel():
    depth = np.zeros((20, 20), dtype=np.float32)
    depth[9:12, 9:12] = 1.5
    depth[10, 10] = 0.0  # a dropout right on the requested pixel
    assert sample_depth(depth, 10, 10, radius=2) == pytest.approx(1.5)
    assert sample_depth(depth, 0, 0, radius=1) is None  # all invalid
    assert sample_depth(depth, 100, 100) is None  # out of frame
    assert sample_depth(None, 5, 5) is None


def test_box_to_sam3_prompt_is_centre_extent():
    assert box_to_sam3_prompt(0.2, 0.4, 0.6, 0.8) == pytest.approx((0.4, 0.6, 0.4, 0.4))
    # Corner order does not matter.
    assert box_to_sam3_prompt(0.6, 0.8, 0.2, 0.4) == pytest.approx((0.4, 0.6, 0.4, 0.4))


# --------------------------------------------------------------------------
# Whole-annotation resolution.
# --------------------------------------------------------------------------


def _annotation_for(cam: CameraModel, contact_xyz, place_xyz, trace_xyz):
    """Build an annotation by projecting known base-frame points."""
    size = cam.image_size

    def pt(p):
        u, v = cam.base_to_pixel(p)
        return Point2D.from_pixels(u, v, size)

    c = pt(contact_xyz)
    return NodeAnnotation(
        subtask="grasp the handle",
        primitive_skill="pick",
        camera=cam.name,
        object_box=Box2D(
            max(0.0, c.x - 0.05), max(0.0, c.y - 0.05), min(1.0, c.x + 0.05), min(1.0, c.y + 0.05)
        ),
        contact_point=c,
        placement_proposal=Box2D(
            pt(place_xyz).x - 0.02,
            pt(place_xyz).y - 0.02,
            pt(place_xyz).x + 0.02,
            pt(place_xyz).y + 0.02,
        ),
        trace=[pt(p) for p in trace_xyz],
    )


def test_resolve_annotation_recovers_the_points_it_was_built_from():
    cam = birdview()
    contact = [-0.82, 0.05, TABLE_Z]
    place = [-0.70, 0.22, TABLE_Z]
    trace = [contact, [-0.76, 0.13, TABLE_Z], place]
    ann = _annotation_for(cam, contact, place, trace)

    res = resolve_annotation(ann, cam, z_plane_m=TABLE_Z, node_path="tree/0")
    assert res.node_path == "tree/0"
    assert res.primitive_skill == "pick"
    assert np.allclose(res.contact_xyz, contact, atol=1e-6)
    assert np.allclose(res.placement_xyz, place, atol=1e-3)
    assert res.trace_xyz.shape == (3, 3)
    assert np.allclose(res.trace_xyz[0], contact, atol=1e-6)
    assert np.allclose(res.trace_xyz[-1], place, atol=1e-6)
    assert res.object_footprint.shape == (4, 3)
    assert any("z-plane assumption" in n for n in res.notes)


def test_a_measured_depth_beats_the_plane_assumption():
    cam = sideview()
    true_point = np.array([-0.80, 0.10, TABLE_Z + 0.06])
    u, v = cam.base_to_pixel(true_point)
    depth = float((cam.rotation.T @ (true_point - cam.position))[2])
    depth_map = np.full((480, 640), depth, dtype=np.float32)

    ann = NodeAnnotation(contact_point=Point2D.from_pixels(u, v, cam.image_size))
    res = resolve_annotation(ann, cam, z_plane_m=TABLE_Z, depth=depth_map)
    assert np.allclose(res.contact_xyz, true_point, atol=1e-3)
    assert any("measured depth" in n for n in res.notes)


def test_resolution_never_silently_uses_the_wrong_camera():
    cam = birdview()
    ann = NodeAnnotation(camera="wrist", contact_point=Point2D(0.5, 0.5))
    score = {
        "tree": {
            "type": "sequence",
            "children": [{"type": "grasp", "params": {}, "__robointer": ann.to_dict()}],
        }
    }
    out = resolve_plan_annotations(score, {"birdview": cam}, TABLE_Z, default_camera="birdview")
    assert len(out) == 1
    assert out[0].contact_xyz is None
    assert any("no camera model for 'wrist'" in n for n in out[0].notes)


def test_default_camera_is_used_when_the_planner_omits_it():
    cam = birdview()
    ann = NodeAnnotation(contact_point=Point2D(0.5, 0.5))
    score = {
        "tree": {
            "type": "sequence",
            "children": [{"type": "grasp", "params": {}, "__robointer": ann.to_dict()}],
        }
    }
    out = resolve_plan_annotations(score, {"birdview": cam}, TABLE_Z, default_camera="birdview")
    assert out[0].contact_xyz is not None
    assert out[0].to_dict()["contact_xyz"] is not None

"""
The published camera frame must be one consistent (rgb, depth, camera) triple.

``state.last_rgb`` / ``last_depth`` / ``last_cam_name`` were three
independent globals with three independent producers:

  * ``capture_frame_sync`` on ``_STREAM_POOL`` (3 workers), reached from
    ``GET /api/capture/stream``;
  * the SAME function reached from the ``/ws/camera`` websocket loop, which
    does NOT take ``_STREAM_INFLIGHT`` -- so two producers really can run at
    once, whatever the semaphore suggests;
  * ``GET /api/capture``.

and ``POST /api/detect_click`` consumed them at three separate moments:
``last_rgb`` up front, ``last_depth`` when backprojecting the mask, and
``last_cam_name`` later still when choosing which camera's calibration to
deproject through. Any interleaving of a tiled publish (which sets the
active camera's frame) with a single-camera publish therefore handed the
click handler one camera's RGB, a second camera's depth, and a third
camera's 4x4.

Nothing raises when that happens: both Kinects run the same resolution, so
the mask backprojects cleanly through the wrong depth map and the wrong
extrinsics, /api/detect_approve turns the result into a keypoint, and the
arm is commanded to a coordinate that corresponds to nothing in the scene.

No hardware, no server; the routes are exercised against fakes.
"""

import contextlib
import threading

import numpy as np
import pytest

from spark_real.routes import state


@pytest.fixture(autouse=True)
def _clean_frame_state():
    state.set_last_frame(None, None, None)
    yield
    state.set_last_frame(None, None, None)


def _frame(tag, h=8, w=8):
    """RGB and depth arrays stamped with the same tag, so a mismatched pair
    is detectable by value."""
    return (
        np.full((h, w, 3), tag, dtype=np.uint8),
        np.full((h, w), float(tag), dtype=np.float32),
    )


# 0. Control: the shape this replaced really does tear
#
# The fixed code cannot exhibit the bug, so this reproduces the OLD write
# pattern -- three independent module globals, assigned one at a time -- in a
# scratch namespace, to show the defect was real and not theoretical. If this
# ever stops tearing, the interpreter has changed, not the argument.


def test_control_three_independent_globals_do_tear():
    """
    A reader interleaved between two of the three stores sees birdview's RGB
    and camera name paired with sideview's depth. Deterministic: the reader is
    pinned exactly where a GIL switch would drop it, and there is no lock to
    stop it, which is precisely the old code's situation.
    """
    ns = {}
    rgb_a, depth_a = _frame(1)
    rgb_b, depth_b = _frame(2)
    ns["rgb"], ns["depth"], ns["cam"] = rgb_a, depth_a, "sideview"

    reader_in = threading.Event()
    may_read = threading.Event()
    seen = {}

    def _read():
        reader_in.set()
        may_read.wait(2.0)
        # Three separate loads, exactly as detect_click did them.
        seen["triple"] = (ns["rgb"], ns["depth"], ns["cam"])

    t = threading.Thread(target=_read, daemon=True)
    t.start()
    assert reader_in.wait(2.0)

    # The old streaming.capture_frame_sync publish: three stores, no lock.
    ns["rgb"] = rgb_b
    ns["cam"] = "birdview"
    may_read.set()  # the reader lands HERE
    t.join(timeout=3.0)
    ns["depth"] = depth_b

    rgb, depth, cam = seen["triple"]
    assert cam == "birdview"
    assert int(rgb[0, 0, 0]) == 2
    assert float(depth[0, 0]) == 1.0, (
        "expected the torn pairing: birdview rgb + birdview name + SIDEVIEW "
        "depth. Nothing raises on it -- the mask just backprojects through "
        "the wrong depth map."
    )


# 1. The triple never tears under concurrent producers


def test_concurrent_publishers_never_produce_a_mixed_triple():
    """
    Two producers, mirroring the /api/capture/stream pool and the /ws/camera
    loop, publishing different cameras as fast as they can. Every consumer
    read must see one camera's rgb, that same camera's depth and that same
    camera's name.
    """
    cameras = {"sideview": (10, _frame(10)), "birdview": (20, _frame(20))}
    stop = threading.Event()
    torn = []

    def _publish(cam_name):
        tag, (rgb, depth) = cameras[cam_name]
        while not stop.is_set():
            state.set_last_frame(rgb, depth, cam_name)

    def _consume():
        while not stop.is_set():
            rgb, depth, cam = state.get_last_frame()
            if rgb is None:
                continue
            expected_tag = cameras[cam][0]
            if int(rgb[0, 0, 0]) != expected_tag or float(depth[0, 0]) != expected_tag:
                torn.append(
                    (cam, int(rgb[0, 0, 0]), float(depth[0, 0]))
                )
                return

    threads = [
        threading.Thread(target=_publish, args=("sideview",), daemon=True),
        threading.Thread(target=_publish, args=("birdview",), daemon=True),
    ] + [threading.Thread(target=_consume, daemon=True) for _ in range(3)]
    for t in threads:
        t.start()
    stop.wait(1.0)
    stop.set()
    for t in threads:
        t.join(timeout=3.0)

    assert torn == [], f"consumer saw a mixed (rgb, depth, camera) triple: {torn[:3]}"


def test_publish_is_indivisible_against_an_interleaved_reader():
    """
    Deterministic version of the same thing: a reader pinned mid-publish must
    still see a complete, self-consistent triple rather than half of each.
    """
    rgb_a, depth_a = _frame(1)
    rgb_b, depth_b = _frame(2)
    state.set_last_frame(rgb_a, depth_a, "sideview")

    reader_in = threading.Event()
    seen = {}

    def _read():
        reader_in.set()
        seen["triple"] = state.get_last_frame()

    # Hold the publish lock, start the reader, and confirm it cannot observe
    # anything while a publish would be halfway through.
    with state._last_frame_lock:
        t = threading.Thread(target=_read, daemon=True)
        t.start()
        assert reader_in.wait(2.0)
        # Simulate the middle of a publish: two of the three names updated.
        state.last_rgb = rgb_b
        state.last_cam_name = "birdview"
        assert "triple" not in seen, "reader observed a half-written publish"
        state.last_depth = depth_b
    t.join(timeout=3.0)
    rgb, depth, cam = seen["triple"]
    assert cam == "birdview"
    assert int(rgb[0, 0, 0]) == 2 and float(depth[0, 0]) == 2.0


# 2. The click handler consumes the triple exactly once


class _Cal:
    def __init__(self, name):
        self.name = name
        self.fx = self.fy = 100.0
        self.cx = self.cy = 4.0
        self.fovy_degrees = 60.0
        self.rotation_matrix = np.eye(3)
        self.position = np.zeros(3)


def test_click_detect_uses_one_camera_for_rgb_depth_and_calibration(monkeypatch):
    """
    The consequence, pinned end to end: a stream publish that lands between
    the handler's reads must not be able to change which calibration the
    mask is deprojected through.
    """
    from spark_real.routes import detection as det_mod

    rgb_side, depth_side = _frame(10, h=16, w=16)
    rgb_bird, depth_bird = _frame(20, h=16, w=16)
    state.set_last_frame(rgb_side, depth_side, "sideview")

    class _Pipe:
        _kinect_cal = _Cal("sideview")
        _kinect2_cal = _Cal("birdview")
        _realsense_cal = None

        def capture(self):
            raise AssertionError("must not need a fresh capture")

        def activity(self, _name):
            return contextlib.nullcontext()

    used = {}

    def _fake_point_prompt(pipeline, rgb, depth, px, py, label="", cam_name=None):
        # A competing producer publishes the OTHER camera right here, exactly
        # where the handler used to re-read state.last_cam_name.
        state.set_last_frame(rgb_bird, depth_bird, "birdview")
        used["rgb_tag"] = int(rgb[0, 0, 0])
        used["depth_tag"] = float(depth[0, 0])
        used["cam_name"] = cam_name
        return []

    def _fake_box_prompt(
        pipeline, rgb, depth, x1, y1, x2, y2, label="", cam_name=None
    ):
        used.setdefault("box_cam_name", cam_name)
        used.setdefault("box_rgb_tag", int(rgb[0, 0, 0]))
        return []

    monkeypatch.setattr(state, "pipeline", _Pipe())
    monkeypatch.setattr(det_mod, "_run_point_prompt", _fake_point_prompt)
    monkeypatch.setattr(det_mod, "_run_box_prompt", _fake_box_prompt)
    monkeypatch.setattr(det_mod, "create_detection_overlay", lambda *a, **k: "")
    monkeypatch.setattr(det_mod, "serialize_detections", lambda d: [])

    req = det_mod.ClickDetectRequest(x=4, y=4, img_width=16, img_height=16, label="x")
    det_mod.detect_click(req)

    assert used["rgb_tag"] == 10
    assert used["depth_tag"] == 10.0
    assert used["cam_name"] == "sideview"
    # The box fallback in the same request must use the same camera too.
    assert used["box_cam_name"] == "sideview"
    assert used["box_rgb_tag"] == 10


def test_calibration_is_selected_from_the_passed_camera_not_global_state(monkeypatch):
    """
    _run_point_prompt / _run_box_prompt must deproject through the camera
    they were HANDED, whatever state.last_cam_name says by the time they run.
    """
    from spark_real.routes import detection as det_mod
    import inspect

    for fn in (det_mod._run_point_prompt, det_mod._run_box_prompt):
        src = inspect.getsource(fn)
        assert "state.last_cam_name" not in src.split('"""')[-1], (
            f"{fn.__name__} still re-reads state.last_cam_name"
        )
        assert "cam_name" in inspect.signature(fn).parameters


# 3. The cold-start fallback publishes a camera name


def test_cold_start_snapshot_publishes_the_camera_it_captured(monkeypatch):
    """
    The old fallback set last_rgb/last_depth from state.active_camera but left
    last_cam_name alone, so the first click after a restart could deproject
    through a stale camera's extrinsics with no concurrency at all.
    """
    from spark_real.routes import detection as det_mod

    rgb, depth = _frame(7)

    class _Pipe:
        def capture(self):
            return {"birdview": {"rgb": rgb, "depth": depth}}

    monkeypatch.setattr(state, "active_camera", "sideview")
    out_rgb, out_depth, out_cam = det_mod._snapshot_frame(_Pipe())
    assert out_cam == "birdview"
    assert state.get_last_frame()[2] == "birdview"
    assert int(out_rgb[0, 0, 0]) == 7 and float(out_depth[0, 0]) == 7.0


def test_cold_start_snapshot_prefers_the_active_camera(monkeypatch):
    from spark_real.routes import detection as det_mod

    rgb_s, depth_s = _frame(1)
    rgb_b, depth_b = _frame(2)

    class _Pipe:
        def capture(self):
            return {
                "sideview": {"rgb": rgb_s, "depth": depth_s},
                "birdview": {"rgb": rgb_b, "depth": depth_b},
            }

    monkeypatch.setattr(state, "active_camera", "birdview")
    _, _, cam = det_mod._snapshot_frame(_Pipe())
    assert cam == "birdview"


def test_no_route_touches_the_three_names_directly():
    """
    Regression guard. Every producer must publish through set_last_frame and
    every consumer must take the triple from one get_last_frame call; a single
    stray ``state.last_depth`` read reintroduces the tear.

    Parsed, not grepped, so the explanatory prose in docstrings and comments
    does not count as a use.
    """
    import ast
    import pathlib

    banned = {"last_rgb", "last_depth", "last_cam_name"}
    routes_dir = pathlib.Path(state.__file__).parent
    offenders = []
    for path in sorted(routes_dir.glob("*.py")):
        if path.name == "state.py":
            continue
        tree = ast.parse(path.read_text(), filename=str(path))
        for node in ast.walk(tree):
            if (
                isinstance(node, ast.Attribute)
                and node.attr in banned
                and isinstance(node.value, ast.Name)
                and node.value.id == "state"
            ):
                offenders.append(f"{path.name}:{node.lineno}: state.{node.attr}")
    assert offenders == [], offenders


def test_snapshot_returns_none_when_there_is_nothing_to_capture():
    from spark_real.routes import detection as det_mod

    class _Pipe:
        def capture(self):
            return {}

    assert det_mod._snapshot_frame(_Pipe()) == (None, None, None)


if __name__ == "__main__":  # pragma: no cover
    pytest.main([__file__, "-v"])

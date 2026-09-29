"""
Teardown ordering: stop and join the readers before closing what they read.

The rule the whole shutdown path exists to keep: no device or robot handle is
closed while a background thread still holds it. The order that satisfies it
is

    brake the arm
      -> stop + JOIN every reader thread (recorders)
        -> quiesce the RealSense recovery machinery
          -> close the Kinects
            -> close the RealSense
              -> disconnect the robot

Nothing used to stop the per-camera VideoRecorder threads. pipeline.shutdown()
went straight to closing cameras, so a SIGTERM during a recorded task left
three threads polling streaming.capture_single_camera at 10 Hz across every
close below them -- and the mp4s were never written either, because stop() is
the only thing that encodes the file.

VideoRecorder.stop() had its own hole: it ignored the join result, so a
capture thread still blocked in a camera read (up to a 2 s staleness deadline)
kept appending to the same deque that _encode was iterating, and
``is_recording`` stayed True so every later start() silently no-op'd.

Everything is mocked; no cameras, no robot, no server.
"""

import threading
import time
from collections import deque

import numpy as np
import pytest

from spark_real.pipeline import SPARKRealPipeline
from spark_real.video_recorder import VideoRecorder


# 1. pipeline.shutdown() ordering


class _Recorder:
    def __init__(self, order, live):
        self._order = order
        self._live = live

    def stop(self):
        self._order.append("video_recorder.stop")
        self._live.discard(self)
        return None


def _ordered_pipeline(order, live_recorders):
    p = object.__new__(SPARKRealPipeline)

    class _K:
        def __init__(self, label):
            self.label = label

        def close(self):
            # A camera must never be closed while a reader is still live.
            assert not live_recorders, (
                f"{self.label}.close() ran with {len(live_recorders)} recorder "
                "thread(s) still reading it"
            )
            order.append(f"{self.label}.close")
            return True

    class _R(_K):
        def request_stop(self):
            order.append("realsense.request_stop")

    class _Robot:
        def disconnect(self):
            assert not live_recorders, (
                "robot.disconnect() ran with a recorder still polling it"
            )
            order.append("robot.disconnect")

    p._kinect = _K("kinect")
    p._kinect2 = _K("kinect2")
    p._realsense = _R("realsense")
    p._robot = _Robot()
    p._initialized = True
    p._video_recorders = {}
    return p


def test_shutdown_joins_recorders_before_closing_any_device():
    order = []
    live = set()
    p = _ordered_pipeline(order, live)
    recs = {cam: _Recorder(order, live) for cam in ("sideview", "birdview")}
    live.update(recs.values())
    p._video_recorders = dict(recs)

    p.shutdown()

    assert order[:2] == ["video_recorder.stop", "video_recorder.stop"], order
    assert order[2:] == [
        "realsense.request_stop",
        "kinect.close",
        "kinect2.close",
        "realsense.close",
        "robot.disconnect",
    ], order
    assert p._video_recorders == {}


def test_shutdown_still_releases_devices_when_a_recorder_will_not_stop():
    """A recorder that raises must not keep the depth MCUs streaming."""
    order = []
    live = set()
    p = _ordered_pipeline(order, live)

    class _BadRecorder:
        def stop(self):
            raise RuntimeError("encoder wedged")

    p._video_recorders = {"sideview": _BadRecorder()}
    p.shutdown()
    assert "kinect.close" in order
    assert "robot.disconnect" in order


def test_shutdown_ordering_is_unchanged_with_no_recorders():
    order = []
    p = _ordered_pipeline(order, set())
    p.shutdown()
    assert order == [
        "realsense.request_stop",
        "kinect.close",
        "kinect2.close",
        "realsense.close",
        "robot.disconnect",
    ]


# 2. VideoRecorder.stop() vs a capture thread that will not exit


def _stuck_recorder(tmp_path, blocker):
    """A recorder whose frame provider blocks, so stop()'s join times out."""
    frames_served = []

    def _provider():
        frames_served.append(1)
        if len(frames_served) > 3:
            blocker.wait(timeout=10.0)
        return np.zeros((4, 4, 3), dtype=np.uint8)

    rec = VideoRecorder(_provider, tmp_path, fps=50)
    rec.JOIN_TIMEOUT_S = 0.2
    return rec, frames_served


def test_stop_does_not_iterate_a_buffer_a_live_thread_is_appending_to(tmp_path):
    """
    _encode used to do list(self._frames) on a deque the loop was still
    appending to. The encode is handed a detached snapshot instead, and the
    zombie loop gets a fresh deque to append into.
    """
    blocker = threading.Event()
    rec, _ = _stuck_recorder(tmp_path, blocker)
    encoded = {}

    def _slow_encode(frames, out_path, fps):
        # Iterate the snapshot slowly while the zombie thread is still alive.
        for _ in range(200):
            encoded["n"] = len(list(frames))
        encoded["path"] = out_path

    rec._encode = lambda frames=None: (
        _slow_encode(frames, tmp_path / "x.mp4", rec.fps) or None
    )
    rec.start("stuck")
    # Let a few frames land before the provider wedges.
    time.sleep(0.2)
    zombie = rec._thread
    rec.stop()
    assert zombie is not None and zombie.is_alive(), (
        "test needs the capture thread to still be running at stop()"
    )
    # The buffer the encode read is not the one the zombie now appends to.
    assert rec._frames is not None
    blocker.set()
    zombie.join(timeout=5.0)


def test_stop_leaves_the_recorder_restartable_after_a_timed_out_join(tmp_path):
    """
    is_recording stayed True forever after a timed-out join, so every later
    start() logged "already running, ignoring" and recorded nothing.
    """
    blocker = threading.Event()
    rec, _ = _stuck_recorder(tmp_path, blocker)
    rec._encode = lambda frames=None: None
    rec.start("first")
    time.sleep(0.2)
    zombie = rec._thread
    rec.stop()
    assert rec.is_recording is False, (
        "recorder still reports itself as recording after stop(); the next "
        "start() would silently do nothing"
    )
    rec.start("second")
    assert rec.is_recording is True
    assert rec._thread is not zombie
    blocker.set()
    rec._stop_event.set()
    zombie.join(timeout=5.0)


def test_zombie_appends_do_not_contaminate_the_next_recording(tmp_path):
    """
    A capture thread that outlives stop() must append into a buffer nobody
    reads, not into the next episode's frames.
    """
    rec = VideoRecorder(lambda: None, tmp_path, fps=10)
    old = rec._frames
    old.append(np.full((2, 2, 3), 9, dtype=np.uint8))
    with rec._frames_lock:
        detached = list(rec._frames)
        rec._frames = deque(maxlen=rec.max_frames)
    # The zombie still holds `old` and keeps appending to it.
    old.append(np.full((2, 2, 3), 9, dtype=np.uint8))
    assert len(detached) == 1
    assert len(rec._frames) == 0


def test_normal_stop_still_encodes(tmp_path):
    """The clean path is unchanged: a healthy thread joins and the mp4 is
    encoded from what it captured."""
    rec = VideoRecorder(
        lambda: np.zeros((4, 4, 3), dtype=np.uint8), tmp_path, fps=50
    )
    seen = {}
    rec._encode = lambda frames=None: seen.setdefault("n", len(frames))
    rec.start("clean")
    time.sleep(0.15)
    rec.stop()
    assert rec.is_recording is False
    assert seen["n"] >= 2


# 3. server._release_devices: one-shot, and readers stopped before devices


@pytest.fixture
def _server_mod():
    import spark_real.server as server_mod

    latch = server_mod._shutdown_latch
    # Give each test a fresh, un-acquired latch.
    server_mod._shutdown_latch = threading.Lock()
    try:
        yield server_mod
    finally:
        server_mod._shutdown_latch = latch


def test_release_devices_runs_exactly_once_under_reentry(_server_mod, monkeypatch):
    """
    `if _shutdown_done: return` then `_shutdown_done = True` is two bytecodes
    apart, and a signal handler interrupts the main thread BETWEEN them -- so
    a Ctrl-C landing there re-entered the whole teardown, braking, discarding
    and closing everything a second time. The latch is atomic.
    """
    from spark_real.routes import state as state_mod

    calls = []

    class _Pipe:
        _robot = None

        def shutdown(self):
            calls.append("shutdown")

    monkeypatch.setattr(state_mod, "pipeline", _Pipe())
    monkeypatch.setattr(state_mod, "vla_recorder", None, raising=False)

    barrier = threading.Barrier(4, timeout=3.0)

    def _fire():
        try:
            barrier.wait()
        except threading.BrokenBarrierError:
            pass
        _server_mod._release_devices("test")

    threads = [threading.Thread(target=_fire, daemon=True) for _ in range(4)]
    for t in threads:
        t.start()
    for t in threads:
        t.join(timeout=5.0)
    assert calls == ["shutdown"], calls


def test_release_devices_stops_the_demo_recorder_before_the_pipeline(
    _server_mod, monkeypatch
):
    """
    The DemoRecorder loop calls pipeline.capture() and reads proprio over RTDE
    at 15 Hz for the whole episode. Left running, its get_observation() lands
    in rtde_r.getActualQ() while robot.disconnect() destroys that interface.
    """
    from spark_real.routes import state as state_mod

    order = []

    class _Recorder:
        def discard(self):
            order.append("recorder.discard")

    class _Pipe:
        _robot = None

        def shutdown(self):
            assert "recorder.discard" in order, (
                "pipeline.shutdown() ran with the demo recorder still polling "
                "the cameras and the robot"
            )
            order.append("pipeline.shutdown")

    monkeypatch.setattr(state_mod, "pipeline", _Pipe())
    monkeypatch.setattr(state_mod, "vla_recorder", _Recorder(), raising=False)
    monkeypatch.setattr(state_mod, "vla_recording", True, raising=False)

    _server_mod._release_devices("test")

    assert order == ["recorder.discard", "pipeline.shutdown"]
    assert state_mod.vla_recorder is None
    assert state_mod.vla_recording is False


def test_release_devices_survives_a_recorder_that_raises(_server_mod, monkeypatch):
    from spark_real.routes import state as state_mod

    order = []

    class _BadRecorder:
        def discard(self):
            raise RuntimeError("writer wedged")

    class _Pipe:
        _robot = None

        def shutdown(self):
            order.append("pipeline.shutdown")

    monkeypatch.setattr(state_mod, "pipeline", _Pipe())
    monkeypatch.setattr(state_mod, "vla_recorder", _BadRecorder(), raising=False)
    _server_mod._release_devices("test")
    assert order == ["pipeline.shutdown"]


if __name__ == "__main__":  # pragma: no cover
    pytest.main([__file__, "-v"])

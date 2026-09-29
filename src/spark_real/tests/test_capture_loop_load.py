"""
Capture-loop work-reduction tests.

Context: the Kinect capture loop decoded transformed_depth AND an IR
preview for every frame of every camera regardless of what any consumer
asked for, and pinned the k4a capture handle across the next blocking
get_capture. On a host where two Kinects share one xHCI controller with a
RealSense and one GPU with three resident models, that is sustained,
bursty, interrupt-heavy work nobody consumes. These tests pin the reduced
behaviour. A fake pyk4a capture stands in for the SDK -- zero hardware.
"""

import threading

import numpy as np
import pytest

import spark_real.perception.camera as cam_mod
import spark_real.routes.streaming as streaming_mod


class _FakeCapture:
    """Counts which lazily-decoded properties the loop actually touches."""

    def __init__(self, counters, released):
        self._counters = counters
        self._released = released

    @property
    def color(self):
        self._counters["color"] += 1
        return np.zeros((8, 8, 4), dtype=np.uint8)

    @property
    def transformed_depth(self):
        self._counters["transformed_depth"] += 1
        return np.full((8, 8), 1000, dtype=np.uint16)

    @property
    def ir(self):
        self._counters["ir"] += 1
        return np.full((8, 8), 500, dtype=np.uint16)

    def __del__(self):
        # Stands in for the PyCapsule's k4a_capture_release destructor.
        self._released.append(1)


class _FakeDevice:
    """Serves N captures then blocks the loop's exit check."""

    def __init__(self, counters, released, n_frames):
        self._counters = counters
        self._released = released
        self._n = n_frames
        self.served = 0
        # len(released) observed at the top of each BLOCKING get_capture --
        # i.e. how many handles were already free when the loop went back to
        # waiting on the device.
        self.released_at_call = []
        self.exhausted = threading.Event()

    def get_capture(self, timeout=None):
        if timeout == 0:
            # Non-blocking drain poll: nothing buffered.
            raise cam_mod.K4ATimeoutException()
        if self.served >= self._n:
            self.exhausted.set()
            raise cam_mod.K4ATimeoutException()
        self.released_at_call.append(len(self._released))
        self.served += 1
        return _FakeCapture(self._counters, self._released)


def _run_loop(monkeypatch, want_depth=True, want_ir=False, n_frames=3):
    class _TE(Exception):
        pass

    monkeypatch.setattr(cam_mod, "K4ATimeoutException", _TE, raising=False)
    counters = {"color": 0, "transformed_depth": 0, "ir": 0}
    released = []
    cam = cam_mod.AzureKinectCamera.__new__(cam_mod.AzureKinectCamera)
    cam.device_id = 0
    cam._lock = threading.Lock()
    cam._stop_flag = False
    cam._want_depth = want_depth
    cam._want_ir = want_ir
    cam._latest_rgb = None
    cam._latest_depth = None
    cam._latest_ir = None
    cam._latest_frame_ts = 0.0
    dev = _FakeDevice(counters, released, n_frames)
    cam._device = dev

    t = threading.Thread(target=cam._capture_loop, daemon=True)
    t.start()
    assert dev.exhausted.wait(5.0)
    cam._stop_flag = True
    t.join(timeout=5.0)
    assert not t.is_alive()
    return cam, counters, released, dev


def test_ir_is_not_computed_unless_asked_for(monkeypatch):
    cam, counters, _, dev = _run_loop(monkeypatch, want_ir=False)
    assert counters["color"] == dev.served
    assert counters["ir"] == 0
    assert cam._latest_ir is None


def test_ir_is_computed_when_enabled(monkeypatch):
    cam, counters, _, dev = _run_loop(monkeypatch, want_ir=True)
    assert counters["ir"] == dev.served
    assert cam._latest_ir is not None
    assert cam._latest_ir.shape == (8, 8, 3)


def test_transformed_depth_is_skipped_when_depth_not_wanted(monkeypatch):
    cam, counters, _, _ = _run_loop(monkeypatch, want_depth=False)
    assert counters["transformed_depth"] == 0
    assert cam._latest_depth is None


def test_transformed_depth_is_decoded_by_default(monkeypatch):
    cam, counters, _, dev = _run_loop(monkeypatch, want_depth=True)
    assert counters["transformed_depth"] == dev.served
    assert cam._latest_depth is not None
    assert cam._latest_depth[0, 0] == pytest.approx(1.0)


def test_capture_handle_is_released_every_iteration(monkeypatch):
    """
    The handle must not stay pinned across the next 250 ms get_capture: it
    holds a buffer out of libk4a's bounded internal pool for an extra loop.
    """
    _, _, released, dev = _run_loop(monkeypatch, n_frames=4)
    # Every previous handle is already released each time the loop blocks on
    # the device again. Pinning one across the wait would make this
    # [0, 0, 1, 2].
    assert dev.released_at_call == [0, 1, 2, 3]
    assert len(released) == dev.served


def test_defaults_on_the_real_constructor():
    cam = cam_mod.AzureKinectCamera(device_id=0)
    assert cam._want_depth is True
    assert cam._want_ir is False
    cam.set_ir_enabled(True)
    assert cam._want_ir is True
    cam.set_ir_enabled(False)
    assert cam._want_ir is False


# streaming: the ir consumer must turn IR on, and must tolerate devices
# without the knob.


class _IRSpy:
    def __init__(self):
        self.calls = []

    def set_ir_enabled(self, enabled):
        self.calls.append(enabled)

    def read(self, depth=True, ir=False):
        if ir:
            return "rgb", "depth", "ir"
        return "rgb", "depth"


def _pipeline_with(kinect):
    class _P:
        pass

    p = _P()
    p._kinect = kinect
    p._kinect2 = None
    p._realsense = None
    p._kinect_read_lock = threading.Lock()
    p._kinect2_read_lock = threading.Lock()
    p._realsense_read_lock = threading.Lock()
    # capture_single_camera applies the per-camera hand-eye depth bias; None
    # means "no calibration loaded", which _corrected_depth passes through.
    p._kinect_cal = None
    p._kinect2_cal = None
    p._realsense_cal = None
    return p


def test_streaming_enables_ir_only_in_ir_mode(monkeypatch):
    spy = _IRSpy()
    monkeypatch.setattr(streaming_mod.state, "pipeline", _pipeline_with(spy))
    streaming_mod.capture_single_camera("sideview", ir_as_rgb=False)
    assert spy.calls == [False]
    rgb, _ = streaming_mod.capture_single_camera("sideview", ir_as_rgb=True)
    assert spy.calls == [False, True]
    assert rgb == "ir"


def test_set_ir_is_a_noop_on_devices_without_the_knob():
    class _Old:
        def read(self, depth=True, ir=False):
            return "rgb", "depth"

    streaming_mod._set_ir(_Old(), True)  # must not raise


# streaming: drop-if-busy backpressure on /api/capture/stream.


def test_stream_endpoint_drops_when_a_capture_is_already_in_flight(monkeypatch):
    import asyncio

    assert streaming_mod._STREAM_INFLIGHT.acquire(blocking=False)
    try:
        resp = asyncio.get_event_loop_policy().new_event_loop().run_until_complete(
            streaming_mod.capture_stream()
        )
    finally:
        streaming_mod._STREAM_INFLIGHT.release()
    assert resp.status_code == 204


def test_stream_endpoint_releases_the_slot_after_a_failed_capture(monkeypatch):
    import asyncio

    def _boom():
        raise RuntimeError("camera exploded")

    monkeypatch.setattr(streaming_mod, "capture_frame_sync", _boom)
    monkeypatch.setattr(streaming_mod.state, "stream_mode", "rgb")
    loop = asyncio.get_event_loop_policy().new_event_loop()
    with pytest.raises(RuntimeError):
        loop.run_until_complete(streaming_mod.capture_stream())
    # Slot must be free again, or the stream wedges permanently after one error.
    assert streaming_mod._STREAM_INFLIGHT.acquire(blocking=False)
    streaming_mod._STREAM_INFLIGHT.release()

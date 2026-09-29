"""
USB lifecycle-safety tests: serialization, reset bounding/backoff,
the reset kill-switch (SPARK_ALLOW_USB_RESET, default OFF), and
shutdown join-before-close.

Context: the host hard-crashes only while the SPARK server is live, with a
kernel fingerprint of unclaimed-usbfs control traffic on the two Kinects at
every lifecycle transition plus xHCI TRB-ring corruption. These tests pin
the process-level invariants that keep USB lifecycle churn bounded and
serialized. All SDK objects are mocked -- zero hardware contact.
"""

import asyncio
import logging
import threading
import time

import pytest

import spark_real.perception.camera as cam_mod
import spark_real.pipeline as pipeline_mod
import spark_real.routes.control as control_mod
from spark_real.pipeline import SPARKRealPipeline
from spark_real.routes import state


# Fake pyrealsense2 surface


class _FakeRSDevice:
    def __init__(self, serial="123", usb_type="3.2"):
        self._serial = serial
        self._usb = usb_type
        self.reset_calls = 0

    def get_info(self, key):
        if key == "serial_number":
            return self._serial
        return self._usb

    def hardware_reset(self):
        self.reset_calls += 1


class _FakeRS:
    """Counts context creations; serves one fake device."""

    class camera_info:
        serial_number = "serial_number"
        usb_type_descriptor = "usb_type_descriptor"

    def __init__(self):
        self.context_calls = 0
        self.device = _FakeRSDevice()

    def context(self):
        self.context_calls += 1
        outer = self

        class _Ctx:
            def query_devices(self):
                return [outer.device]

        return _Ctx()


class _FakePipe:
    def __init__(self):
        self.stop_calls = 0

    def stop(self):
        self.stop_calls += 1


def _make_rs_cam(monkeypatch, fake_rs):
    monkeypatch.setattr(cam_mod, "HAS_REALSENSE", True)
    monkeypatch.setattr(cam_mod, "rs", fake_rs)
    monkeypatch.setattr(cam_mod, "_RS_CONTEXT", None, raising=False)
    monkeypatch.delenv("SPARK_WRIST_COLOR_ONLY", raising=False)
    monkeypatch.delenv("SPARK_NO_USB_RESET", raising=False)
    # Reset paths are OFF by default now; the recovery tests below are about
    # what happens once an operator arms them.
    monkeypatch.setenv("SPARK_ALLOW_USB_RESET", "1")
    c = cam_mod.RealSenseCamera(serial="123")
    c._usb2_mode = False
    c._pipeline = _FakePipe()
    # Model an opened, streaming camera (open() sets this); recovery is
    # only ever attempted on a camera that was successfully opened.
    c._running = True
    return c


@pytest.fixture(autouse=True)
def _reset_process_state():
    """Recovery attempts are bounded per process; reset between tests."""
    cam_mod.RealSenseCamera._recover_attempts_total = 0
    cam_mod.RealSenseCamera._last_recover_ts = None
    yield
    cam_mod.RealSenseCamera._recover_attempts_total = 0
    cam_mod.RealSenseCamera._last_recover_ts = None


# 1. Kill-switch: zero reset traffic, log-and-degrade


def test_kill_switch_disables_hardware_reset(monkeypatch, caplog):
    fake_rs = _FakeRS()
    c = _make_rs_cam(monkeypatch, fake_rs)
    monkeypatch.setenv("SPARK_NO_USB_RESET", "1")
    with caplog.at_level(logging.WARNING):
        assert c._recovery_allowed() is False
        assert c._try_recover() is False
    # No USB traffic of any kind: no enumeration, no reset, no pipeline churn.
    assert fake_rs.context_calls == 0
    assert fake_rs.device.reset_calls == 0
    assert c._pipeline.stop_calls == 0
    assert any("reset paths disabled" in r.getMessage() for r in caplog.records)


# 2. Reset bounding + exponential backoff (max attempts per process lifetime)


def test_recovery_bounded_with_exponential_backoff(monkeypatch):
    fake_rs = _FakeRS()
    c = _make_rs_cam(monkeypatch, fake_rs)
    c._start_pipeline = lambda: None
    # Do not actually sleep the 6s re-enumeration wait.
    monkeypatch.setattr(c._closing, "wait", lambda *a, **k: False)
    now = [1000.0]
    monkeypatch.setattr(cam_mod._time, "monotonic", lambda: now[0])

    assert cam_mod.RealSenseCamera.MAX_RECOVER_ATTEMPTS == 3

    # Attempt 1 allowed immediately.
    assert c._recovery_allowed() is True
    assert c._try_recover() is True
    assert fake_rs.device.reset_calls == 1

    # Backoff gap 1: 30 s.
    now[0] += 5.0
    assert c._recovery_allowed() is False
    now[0] += 26.0  # t + 31
    assert c._recovery_allowed() is True
    assert c._try_recover() is True
    assert fake_rs.device.reset_calls == 2

    # Backoff gap 2: 60 s.
    now[0] += 31.0
    assert c._recovery_allowed() is False
    now[0] += 30.0  # t + 61
    assert c._recovery_allowed() is True
    assert c._try_recover() is True
    assert fake_rs.device.reset_calls == 3

    # Attempts exhausted for the rest of the process lifetime.
    now[0] += 1e6
    assert c._recovery_allowed() is False
    assert c._try_recover() is False
    assert fake_rs.device.reset_calls == 3


# 3. Lifecycle serialization: recovery cannot reset while another camera is
#    mid-open/close (both funnel through USB_LIFECYCLE_LOCK)


def test_recovery_serialized_by_lifecycle_lock(monkeypatch):
    fake_rs = _FakeRS()
    c = _make_rs_cam(monkeypatch, fake_rs)
    c._start_pipeline = lambda: None
    monkeypatch.setattr(c._closing, "wait", lambda *a, **k: False)

    started = threading.Event()
    done = threading.Event()

    def _worker():
        started.set()
        c._try_recover()
        done.set()

    assert cam_mod.USB_LIFECYCLE_LOCK.acquire(timeout=2.0)
    try:
        t = threading.Thread(target=_worker, daemon=True)
        t.start()
        assert started.wait(2.0)
        time.sleep(0.3)
        # Blocked on the lifecycle lock: no reset traffic yet.
        assert fake_rs.device.reset_calls == 0
        assert not done.is_set()
    finally:
        cam_mod.USB_LIFECYCLE_LOCK.release()
    assert done.wait(2.0)
    assert fake_rs.device.reset_calls == 1


# 4. RealSense shutdown: reader thread joined BEFORE pipeline.stop()


def test_realsense_close_joins_thread_before_stop(monkeypatch):
    fake_rs = _FakeRS()
    c = _make_rs_cam(monkeypatch, fake_rs)
    order = []

    t = threading.Thread(target=lambda: c._closing.wait(5.0), daemon=True)

    class _OrderedPipe:
        def stop(self):
            order.append(("stop", t.is_alive()))

    c._pipeline = _OrderedPipe()
    t.start()
    c._thread = t
    c.close()
    assert order == [("stop", False)]


def test_realsense_close_skips_stop_when_thread_stuck(monkeypatch, caplog):
    fake_rs = _FakeRS()
    c = _make_rs_cam(monkeypatch, fake_rs)
    monkeypatch.setattr(cam_mod.RealSenseCamera, "CLOSE_JOIN_TIMEOUT_S", 0.2)
    stuck = threading.Event()
    t = threading.Thread(target=stuck.wait, args=(10.0,), daemon=True)
    t.start()
    c._thread = t
    pipe = _FakePipe()
    c._pipeline = pipe
    with caplog.at_level(logging.WARNING):
        c.close()
    # Never stop the pipeline out from under a live reader thread.
    assert pipe.stop_calls == 0
    stuck.set()


# 5. Kinect shutdown: capture thread joined BEFORE device.stop()


def test_kinect_close_joins_capture_thread_before_device_stop():
    k = cam_mod.AzureKinectCamera(device_id=0)
    order = []

    t_holder = {}

    class _Dev:
        def stop(self):
            order.append(("stop", t_holder["t"].is_alive()))

    k._device = _Dev()

    def _loop():
        while not k._stop_flag:
            time.sleep(0.01)

    t = threading.Thread(target=_loop, daemon=True)
    t_holder["t"] = t
    t.start()
    k._thread = t
    k.close()
    assert order == [("stop", False)]


def test_kinect_close_skips_device_stop_when_thread_stuck(monkeypatch, caplog):
    monkeypatch.setattr(
        cam_mod.AzureKinectCamera, "JOIN_TIMEOUT_S", 0.1, raising=False
    )
    monkeypatch.setattr(cam_mod.AzureKinectCamera, "JOIN_GRACE_S", 0.1, raising=False)
    k = cam_mod.AzureKinectCamera(device_id=0)
    calls = []

    class _Dev:
        def stop(self):
            calls.append("stop")

    k._device = _Dev()
    stuck = threading.Event()
    t = threading.Thread(target=stuck.wait, args=(10.0,), daemon=True)
    t.start()
    k._thread = t
    with caplog.at_level(logging.WARNING):
        k.close()
    # A capture thread that will not die means libk4a is wedged; stopping the
    # device from another thread races its in-flight get_capture.
    assert calls == []
    stuck.set()


# 6. Enumerate/probe once per process


def test_realsense_probe_enumerates_once(monkeypatch):
    fake = _FakeRS()
    monkeypatch.setattr(pipeline_mod, "rs", fake)
    monkeypatch.setattr(pipeline_mod, "_REALSENSE_PROBE_RESULT", None, raising=False)
    assert pipeline_mod._realsense_device_available() is True
    assert pipeline_mod._realsense_device_available() is True
    assert pipeline_mod._realsense_device_available() is True
    assert fake.context_calls == 1


# 7. Pipeline shutdown ordering: quiesce RealSense recovery machinery FIRST,
#    so a hardware_reset can never fire while a Kinect is mid-close.


def test_shutdown_quiesces_realsense_recovery_before_closing_kinects():
    p = object.__new__(SPARKRealPipeline)
    order = []

    class _K:
        def close(self):
            order.append("kinect_close")

    class _R:
        def request_stop(self):
            order.append("rs_request_stop")

        def close(self):
            order.append("rs_close")

    p._kinect = _K()
    p._kinect2 = None
    p._realsense = _R()
    p._robot = None
    p._initialized = True
    p.shutdown()
    assert order == ["rs_request_stop", "kinect_close", "rs_close"]


# 8. /api/kinect_reset: kill-switch gate, no close-under-a-held-read-lock,
#    and refusal while another lifecycle transition is in flight.


class _FakeKinectDev:
    def __init__(self, log):
        self._log = log

    def close(self):
        self._log.append("close")


def _fake_reset_pipeline(closed_log):
    class _P:
        pass

    p = _P()
    p._kinect = _FakeKinectDev(closed_log)
    p._kinect2 = None
    p._kinect_read_lock = threading.Lock()
    p._kinect2_read_lock = threading.Lock()
    return p


def _forbid_subprocess(*args, **kwargs):
    raise AssertionError("kinect_reset must not reach the uhubctl subprocess here")


def test_kinect_reset_refused_by_kill_switch(monkeypatch):
    closed = []
    pipe = _fake_reset_pipeline(closed)
    monkeypatch.setattr(state, "pipeline", pipe)
    monkeypatch.setenv("SPARK_NO_USB_RESET", "1")
    monkeypatch.setattr(control_mod.subprocess, "run", _forbid_subprocess)
    monkeypatch.setattr(control_mod.time, "sleep", lambda s: None)
    resp = control_mod.kinect_reset()
    assert getattr(resp, "status_code", 200) == 409
    assert closed == []


def test_kinect_reset_aborts_when_camera_mid_read(monkeypatch):
    closed = []
    pipe = _fake_reset_pipeline(closed)
    monkeypatch.setattr(state, "pipeline", pipe)
    monkeypatch.delenv("SPARK_NO_USB_RESET", raising=False)
    monkeypatch.setenv("SPARK_ALLOW_USB_RESET", "1")
    monkeypatch.setattr(control_mod.subprocess, "run", _forbid_subprocess)
    monkeypatch.setattr(control_mod.time, "sleep", lambda s: None)
    monkeypatch.setattr(
        control_mod, "_RESET_READ_LOCK_TIMEOUT_S", 0.1, raising=False
    )
    # A reader (streaming / detect / calibration) holds the per-device lock.
    assert pipe._kinect_read_lock.acquire(timeout=1.0)
    try:
        resp = control_mod.kinect_reset()
    finally:
        pipe._kinect_read_lock.release()
    # Must abort, and must NOT have closed the device under the reader.
    assert getattr(resp, "status_code", 200) == 503
    assert closed == []


def test_kinect_reset_refuses_while_lifecycle_transition_in_flight(monkeypatch):
    closed = []
    pipe = _fake_reset_pipeline(closed)
    monkeypatch.setattr(state, "pipeline", pipe)
    monkeypatch.delenv("SPARK_NO_USB_RESET", raising=False)
    monkeypatch.setenv("SPARK_ALLOW_USB_RESET", "1")
    monkeypatch.setattr(control_mod.subprocess, "run", _forbid_subprocess)
    monkeypatch.setattr(control_mod.time, "sleep", lambda s: None)
    monkeypatch.setattr(
        control_mod, "_LIFECYCLE_ACQUIRE_TIMEOUT_S", 0.1, raising=False
    )

    # Another thread is mid open/close/reset (holds the module lifecycle lock).
    held = threading.Event()
    release = threading.Event()

    def _holder():
        with cam_mod.USB_LIFECYCLE_LOCK:
            held.set()
            release.wait(5.0)

    t = threading.Thread(target=_holder, daemon=True)
    t.start()
    assert held.wait(2.0)
    try:
        resp = control_mod.kinect_reset()
    finally:
        release.set()
        t.join(timeout=2.0)
    assert getattr(resp, "status_code", 200) == 503
    assert closed == []


# 9. close() reports failure, stays retryable, and gates the reset path.
#
# The old close() set self._closed = True BEFORE joining the capture thread,
# then returned early if the thread would not die. Result: a failed close was
# indistinguishable from a successful one, could never be retried (the
# reentrancy guard was already tripped), and silently satisfied
# /api/kinect_reset's "handles released" precondition -- so uhubctl cut Vbus
# on a hub whose device still had a live capture thread doing libusb I/O.


def _stuck_kinect(monkeypatch, stuck):
    monkeypatch.setattr(
        cam_mod.AzureKinectCamera, "JOIN_TIMEOUT_S", 0.05, raising=False
    )
    monkeypatch.setattr(cam_mod.AzureKinectCamera, "JOIN_GRACE_S", 0.05, raising=False)
    k = cam_mod.AzureKinectCamera(device_id=0)

    class _Dev:
        def stop(self):
            pass

    k._device = _Dev()
    t = threading.Thread(target=stuck.wait, args=(10.0,), daemon=True)
    t.start()
    k._thread = t
    return k


def test_close_returns_false_when_capture_thread_will_not_exit(monkeypatch):
    stuck = threading.Event()
    k = _stuck_kinect(monkeypatch, stuck)
    try:
        assert k.close() is False
        # And it must stay retryable: _closed must NOT be latched by a
        # failed attempt.
        assert getattr(k, "_closed", False) is False
        assert k.close() is False
    finally:
        stuck.set()


def test_close_succeeds_on_retry_once_the_thread_exits(monkeypatch):
    stuck = threading.Event()
    k = _stuck_kinect(monkeypatch, stuck)
    assert k.close() is False
    stuck.set()
    k._thread.join(timeout=2.0)
    assert k.close() is True
    assert k._closed is True
    # A completed close short-circuits and still reports success.
    assert k.close() is True


def test_close_returns_true_on_a_clean_release():
    k = cam_mod.AzureKinectCamera(device_id=0)
    calls = []

    class _Dev:
        def stop(self):
            calls.append("stop")

    k._device = _Dev()
    k._thread = None
    assert k.close() is True
    assert calls == ["stop"]
    assert cam_mod.teardown_in_flight() is False


def test_kinect_reset_refuses_to_power_cycle_after_a_failed_close(monkeypatch):
    """The whole point: never cut Vbus on a device with a live capture thread."""

    class _FailingDev:
        def __init__(self):
            self.close_calls = 0

        def close(self):
            self.close_calls += 1
            return False

    class _P:
        pass

    p = _P()
    dev = _FailingDev()
    p._kinect = dev
    p._kinect2 = None
    p._realsense = None
    p._kinect_read_lock = threading.Lock()
    p._kinect2_read_lock = threading.Lock()
    monkeypatch.setattr(state, "pipeline", p)
    monkeypatch.delenv("SPARK_NO_USB_RESET", raising=False)
    monkeypatch.setenv("SPARK_ALLOW_USB_RESET", "1")
    monkeypatch.setattr(control_mod.subprocess, "run", _forbid_subprocess)
    monkeypatch.setattr(control_mod.time, "sleep", lambda s: None)
    resp = control_mod.kinect_reset()
    assert getattr(resp, "status_code", 200) == 409
    assert dev.close_calls == 1
    # Slot must NOT be nulled: the handle is still live.
    assert p._kinect is dev


def test_kinect_reset_quiesces_the_realsense_before_touching_the_bus(monkeypatch):
    """
    The wrist camera is on the same xHCI controller. It used to keep
    submitting bulk URBs straight through the Vbus cut.
    """
    order = []

    class _RS:
        def request_stop(self):
            order.append("rs_request_stop")

    class _K:
        def close(self):
            order.append("kinect_close")
            return False  # abort before the subprocess; we only test order

    class _P:
        pass

    p = _P()
    p._kinect = _K()
    p._kinect2 = None
    p._realsense = _RS()
    p._kinect_read_lock = threading.Lock()
    p._kinect2_read_lock = threading.Lock()
    monkeypatch.setattr(state, "pipeline", p)
    monkeypatch.delenv("SPARK_NO_USB_RESET", raising=False)
    monkeypatch.setenv("SPARK_ALLOW_USB_RESET", "1")
    monkeypatch.setattr(control_mod.subprocess, "run", _forbid_subprocess)
    monkeypatch.setattr(control_mod.time, "sleep", lambda s: None)
    control_mod.kinect_reset()
    assert order == ["rs_request_stop", "kinect_close"]


def test_reset_no_longer_deauthorizes_hardcoded_bus_paths():
    """
    '1-5 1-8 2-9' does not describe this host (Kinect hubs are 2-2/2-7 SS and
    1-1/1-8 HS), so the inline loop hit one hub of four and reported success.
    Hub discovery lives in scripts/kinect_authorize_reset.sh; the route must
    call it rather than carry its own worse copy.
    """
    import inspect

    src = inspect.getsource(control_mod._kinect_reset_locked)
    # No inline sysfs writes at all any more (the historical paths survive
    # only in the comment explaining why they were wrong).
    assert "/sys/bus/usb/devices/$d/authorized" not in src
    assert "echo 0 >" not in src
    assert "kinect_authorize_reset.sh" in src


# 10. Teardown-in-flight: USB_LIFECYCLE_LOCK does not cover an abandoned
#     libk4a stop(), so a separate flag has to.


def test_teardown_flag_outlives_the_abandoned_release_wait(monkeypatch):
    monkeypatch.setattr(
        cam_mod.AzureKinectCamera, "RELEASE_JOIN_TIMEOUT_S", 0.1, raising=False
    )
    k = cam_mod.AzureKinectCamera(device_id=0)
    hang = threading.Event()

    class _HangingDev:
        def stop(self):
            hang.wait(10.0)

    k._device = _HangingDev()
    k._thread = None
    try:
        assert k.close() is False  # abandoned wait is not a clean release
        # The lifecycle lock is free again...
        assert cam_mod.USB_LIFECYCLE_LOCK.acquire(blocking=False)
        cam_mod.USB_LIFECYCLE_LOCK.release()
        # ...but libk4a is still on the bus, and the flag says so.
        assert cam_mod.teardown_in_flight() is True
        assert cam_mod.await_teardown_clear(timeout=0.2) is False
    finally:
        hang.set()
    assert cam_mod.await_teardown_clear(timeout=5.0) is True
    assert cam_mod.teardown_in_flight() is False


def test_realsense_recovery_refuses_while_a_teardown_is_in_flight(monkeypatch):
    fake_rs = _FakeRS()
    c = _make_rs_cam(monkeypatch, fake_rs)
    cam_mod._teardown_begin()
    try:
        assert c._recovery_allowed() is False
        assert c._try_recover() is False
        assert fake_rs.device.reset_calls == 0
    finally:
        cam_mod._teardown_end()
    assert c._recovery_allowed() is True


def test_kinect_open_refuses_while_a_teardown_is_in_flight(monkeypatch):
    monkeypatch.setattr(cam_mod, "HAS_K4A", True)
    k = cam_mod.AzureKinectCamera(device_id=0)
    cam_mod._teardown_begin()
    try:
        monkeypatch.setattr(cam_mod, "await_teardown_clear", lambda *a, **kw: False)
        with pytest.raises(RuntimeError, match="teardown is still in flight"):
            k.open()
    finally:
        cam_mod._teardown_end()


# 11. The kill-switch is inverted: reset paths are OFF unless armed.


def test_usb_reset_is_disabled_by_default(monkeypatch):
    monkeypatch.delenv("SPARK_NO_USB_RESET", raising=False)
    monkeypatch.delenv("SPARK_ALLOW_USB_RESET", raising=False)
    assert cam_mod.usb_reset_disabled() is True


def test_usb_reset_can_be_armed_explicitly(monkeypatch):
    monkeypatch.delenv("SPARK_NO_USB_RESET", raising=False)
    monkeypatch.setenv("SPARK_ALLOW_USB_RESET", "1")
    assert cam_mod.usb_reset_disabled() is False


def test_legacy_no_reset_still_wins_over_the_arm_flag(monkeypatch):
    monkeypatch.setenv("SPARK_ALLOW_USB_RESET", "1")
    monkeypatch.setenv("SPARK_NO_USB_RESET", "1")
    assert cam_mod.usb_reset_disabled() is True


def test_kinect_reset_endpoint_refused_by_default(monkeypatch):
    closed = []
    pipe = _fake_reset_pipeline(closed)
    monkeypatch.setattr(state, "pipeline", pipe)
    monkeypatch.delenv("SPARK_NO_USB_RESET", raising=False)
    monkeypatch.delenv("SPARK_ALLOW_USB_RESET", raising=False)
    monkeypatch.setattr(control_mod.subprocess, "run", _forbid_subprocess)
    resp = control_mod.kinect_reset()
    assert getattr(resp, "status_code", 200) == 409
    assert closed == []


def test_capture_loop_device_restart_is_off_by_default(monkeypatch):
    """
    The bounded stop/start in _capture_loop re-runs SET_INTERFACE bandwidth
    negotiation while the OTHER Kinect's isochronous reservation is live. It
    used to fire unattended after 40 consecutive failures.
    """
    monkeypatch.delenv("SPARK_NO_USB_RESET", raising=False)
    monkeypatch.delenv("SPARK_ALLOW_USB_RESET", raising=False)

    class _TE(Exception):
        pass

    monkeypatch.setattr(cam_mod, "K4ATimeoutException", _TE, raising=False)
    k = cam_mod.AzureKinectCamera(device_id=0)
    k._lock = threading.Lock()
    k._stop_flag = False
    calls = []
    seen = threading.Event()

    class _BrokenDev:
        def get_capture(self, timeout=None):
            calls.append("get")
            if len(calls) > 200:
                seen.set()
            raise RuntimeError("bus is angry")

        def stop(self):
            calls.append("stop")

        def start(self):
            calls.append("start")

    k._device = _BrokenDev()
    monkeypatch.setattr(cam_mod._time, "sleep", lambda s: None)
    t = threading.Thread(target=k._capture_loop, daemon=True)
    t.start()
    assert seen.wait(5.0)
    k._stop_flag = True
    t.join(timeout=5.0)
    # Well past _FAIL_RESTART_THRESHOLD=40 and no lifecycle traffic at all.
    assert "stop" not in calls and "start" not in calls

"""
Concurrent-close safety for the two SDK-backed cameras.

The interleaving these pin: TWO first-time closers reaching the same camera
object at once. On this rig there are four independent callers and none of
them coordinate -- server._signal_handler (SIGTERM/SIGINT/SIGHUP),
app.on_event("shutdown"), atexit._release_devices, and
/api/kinect_reset._kinect_reset_locked -- plus pipeline.shutdown() itself.

AzureKinectCamera.close() built its reentrancy lock LAZILY::

    if not hasattr(self, "_close_lock"):
        self._close_lock = _threading.Lock()
    with self._close_lock:
        ...

which is the very race it exists to prevent. Both threads read hasattr ->
False, each builds a SEPARATE Lock, each acquires its own, and both run
_close_locked -- whose docstring promises "exactly one thread is in here at
a time". Downstream both read self._device before either nulls it, and both
hand the same libk4a handle to a _release worker: device.stop() then runs
twice, concurrently, on one C handle.

Everything is mocked; no hardware is touched.
"""

import threading

import pytest

import spark_real.perception.camera as cam_mod


def _concurrency_probe(cls, method_name):
    """
    Wrap ``cls.method_name`` so the test can see the PEAK number of threads
    inside it at once, and force both callers to arrive together.

    ``gate`` is released by the test once it has observed the peak, so a
    thread that did get in cannot finish and let the second one in
    afterwards -- overlap has to be genuine, not sequential.

    ``state["entered"]`` fires on the FIRST entry. The test waits on that
    rather than sampling the peak after a fixed delay: the forcing harness
    deliberately stalls the leader on a bounded wait before it reaches the
    body, so any fixed sample point races that stall instead of measuring
    overlap.
    """
    state = {"peak": 0, "live": 0, "entered": threading.Event()}
    lock = threading.Lock()
    gate = threading.Event()
    original = getattr(cls, method_name)

    def wrapper(self, *args, **kwargs):
        with lock:
            state["live"] += 1
            state["peak"] = max(state["peak"], state["live"])
        state["entered"].set()
        try:
            gate.wait(timeout=2.0)
            return original(self, *args, **kwargs)
        finally:
            with lock:
                state["live"] -= 1

    return wrapper, state, gate


def _await_peak(probe, want, timeout=2.0, poll=0.02):
    """Poll until ``probe["peak"]`` reaches ``want``, then return it.

    Returns early on success, so the passing case is fast; on the failing
    case it spends the full budget establishing that the second thread really
    did not get in.
    """
    deadline = threading.Event()
    elapsed = 0.0
    while elapsed < timeout:
        if probe["peak"] >= want:
            break
        deadline.wait(poll)
        elapsed += poll
    return probe["peak"]


def _force_lazy_lock_race(obj):
    """
    Drive the two closers through the ONE interleaving in which a lazily
    built ``_close_lock`` fails to guard anything.

    The lazy pattern is nearly-but-not-quite safe, which is why it survived:
    both threads re-READ ``self._close_lock`` at the ``with`` statement, so
    last-writer-wins usually reconverges them onto a single Lock object.
    It only loses when the leader reads back its OWN lock before the
    follower's assignment lands::

        L: hasattr(_close_lock) -> absent          F: hasattr -> absent
        L: self._close_lock = L1
                                                   F: self._close_lock = L2
        L: with self._close_lock (reads L1) -- acquires L1
        L: _closed?            -> False
        L: _close_in_progress? -> False
                                                   F: with self._close_lock
                                                      (reads L2) -- acquires
                                                      L2, uncontended
                                                   F: _closed? -> False
                                                   F: _close_in_progress?
                                                      -> False  << stale
        L: _close_in_progress = True               F: _close_in_progress = True
        L: -> _close_locked()                      F: -> _close_locked()

    Both then read ``self._device`` before either nulls it and both hand the
    SAME libk4a handle to a ``_release`` worker: ``device.stop()`` runs twice,
    concurrently, on one C handle.

    Forcing it: a barrier makes both threads observe the lock as absent; the
    follower's assignment is held until the leader has acquired; and the
    leader's ``_close_in_progress = True`` write is held until the follower
    has read the stale False. Every wait is bounded, so the fixed code (where
    the lock pre-exists and the follower simply blocks on it) just proceeds.
    """
    absent = threading.Barrier(2, timeout=3.0)
    leader_acquired = threading.Event()
    follower_checked = threading.Event()
    roles = {}
    roles_lock = threading.Lock()
    cls = type(obj)

    def _role():
        ident = threading.get_ident()
        with roles_lock:
            if ident not in roles:
                roles[ident] = "leader" if not roles else "follower"
            return roles[ident]

    def _wait(barrier_or_event, timeout=3.0):
        try:
            if isinstance(barrier_or_event, threading.Barrier):
                barrier_or_event.wait()
            else:
                barrier_or_event.wait(timeout)
        except threading.BrokenBarrierError:
            pass

    class _Forced(cls):
        def __getattribute__(self, name):
            if name == "_close_lock":
                role = _role()
                try:
                    value = object.__getattribute__(self, name)
                except AttributeError:
                    _wait(absent)  # both learn it is missing
                    raise
                if role == "leader" and not leader_acquired.is_set():
                    # About to enter `with`; let the follower's assignment go.
                    leader_acquired.set()
                return value
            return object.__getattribute__(self, name)

        def __setattr__(self, name, value):
            if name == "_close_lock" and _role() == "follower":
                _wait(leader_acquired)
            if (
                name == "_close_in_progress"
                and value is True
                and _role() == "leader"
            ):
                _wait(follower_checked, timeout=0.3)
            object.__setattr__(self, name, value)
            if name == "_close_in_progress" and value is True:
                if _role() == "follower":
                    follower_checked.set()

    obj.__class__ = _Forced
    return absent


def _run_two_closers(cam):
    results = []
    res_lock = threading.Lock()

    def _close():
        out = cam.close()
        with res_lock:
            results.append(out)

    threads = [threading.Thread(target=_close, daemon=True) for _ in range(2)]
    for t in threads:
        t.start()
    return threads, results


# AzureKinectCamera


def test_kinect_two_first_time_closers_never_overlap_in_close_locked(monkeypatch):
    """At most ONE thread may be inside _close_locked; the other reports False."""
    wrapper, probe, gate = _concurrency_probe(
        cam_mod.AzureKinectCamera, "_close_locked"
    )
    monkeypatch.setattr(cam_mod.AzureKinectCamera, "_close_locked", wrapper)

    k = cam_mod.AzureKinectCamera(device_id=0)
    stops = []

    class _Dev:
        def stop(self):
            stops.append("stop")

    k._device = _Dev()
    k._thread = None

    _force_lazy_lock_race(k)
    threads, results = _run_two_closers(k)

    # Wait for the FIRST closer to be inside the body, then give the second a
    # real budget to join it. Sampling at a fixed instant instead raced the
    # forcing harness's own bounded stall and read peak=0.
    assert probe["entered"].wait(3.0), "no closer reached _close_locked"
    peak = _await_peak(probe, want=2)
    gate.set()
    for t in threads:
        t.join(timeout=5.0)
        assert not t.is_alive()

    assert peak == 1, (
        "two threads entered _close_locked concurrently: the lazily-built "
        "_close_lock let each closer create its own lock"
    )
    # One clean release, one refusal -- and libk4a's stop() ran exactly once.
    assert sorted(results) == [False, True]
    assert stops == ["stop"]


def test_kinect_close_lock_exists_before_any_close():
    """The guard has to pre-exist; building it inside close() is the race."""
    k = cam_mod.AzureKinectCamera(device_id=0)
    assert isinstance(k._close_lock, type(threading.Lock()))
    assert k._closed is False
    assert k._close_in_progress is False


def test_kinect_device_handed_to_exactly_one_release_worker(monkeypatch):
    """
    The consequence of the overlap, stated directly: the SAME libk4a handle
    must never be stopped from two threads at once.
    """
    k = cam_mod.AzureKinectCamera(device_id=0)
    calls = []
    concurrent = {"peak": 0, "live": 0}
    lock = threading.Lock()
    hold = threading.Event()

    class _Dev:
        def stop(self):
            with lock:
                calls.append("stop")
                concurrent["live"] += 1
                concurrent["peak"] = max(concurrent["peak"], concurrent["live"])
            # Stay inside the SDK call long enough that a second entry
            # overlaps rather than merely following.
            hold.wait(timeout=1.0)
            with lock:
                concurrent["live"] -= 1

    k._device = _Dev()
    k._thread = None
    _force_lazy_lock_race(k)
    threads, _ = _run_two_closers(k)
    # Let both _release workers be in flight before releasing them.
    threading.Event().wait(0.5)
    peak = concurrent["peak"]
    n_calls = len(calls)
    hold.set()
    for t in threads:
        t.join(timeout=5.0)
    assert cam_mod.await_teardown_clear(timeout=5.0) is True
    assert n_calls == 1, "the same libk4a handle was stopped twice"
    assert peak <= 1, "two threads were inside device.stop() at once"


# RealSenseCamera


class _CountingPipe:
    def __init__(self):
        self.stop_calls = 0

    def stop(self):
        self.stop_calls += 1


def _rs_cam(monkeypatch):
    monkeypatch.setattr(cam_mod, "HAS_REALSENSE", True)
    c = cam_mod.RealSenseCamera.__new__(cam_mod.RealSenseCamera)
    c._pipeline = _CountingPipe()
    c._thread = None
    c._running = False
    c._closing = threading.Event()
    c._close_lock = threading.Lock()
    c._closed = False
    c._close_in_progress = False
    return c


def test_realsense_pipeline_stopped_exactly_once(monkeypatch):
    """
    Two closers on one librealsense pipeline handle. USB_LIFECYCLE_LOCK
    serialises them but does not stop the second stop() from reaching the
    same handle; the reentrancy guard does.
    """
    c = _rs_cam(monkeypatch)
    pipe = c._pipeline
    threads = [threading.Thread(target=c.close, daemon=True) for _ in range(2)]
    for t in threads:
        t.start()
    for t in threads:
        t.join(timeout=5.0)
        assert not t.is_alive()
    assert pipe.stop_calls == 1
    assert c._pipeline is None
    assert c._closed is True


def test_realsense_close_lock_is_not_held_across_the_teardown_body(monkeypatch):
    """
    LOCK ORDER. _close_locked joins the capture thread and then takes
    USB_LIFECYCLE_LOCK. Holding _close_lock across it nests

        _close_lock -> USB_LIFECYCLE_LOCK

    while /api/kinect_reset._kinect_reset_locked already holds
    USB_LIFECYCLE_LOCK when it reaches a camera close -- an inversion, and a
    deadlock as soon as anything closes a camera from under that endpoint.
    The guard must cover the flag transitions only, as the Kinect's always has.
    """
    c = _rs_cam(monkeypatch)
    observed = {}
    real_body = c._close_locked

    def _watch():
        observed["held"] = c._close_lock.locked()
        return real_body()

    c._close_locked = _watch
    c.close()
    assert observed["held"] is False, (
        "_close_lock was held across _close_locked, which goes on to take "
        "USB_LIFECYCLE_LOCK -- the inverted order that deadlocks against "
        "/api/kinect_reset"
    )


def test_kinect_close_lock_is_not_held_across_the_teardown_body():
    """Same invariant on the Kinect, so it cannot regress either."""
    k = cam_mod.AzureKinectCamera(device_id=0)

    class _Dev:
        def stop(self):
            pass

    k._device = _Dev()
    k._thread = None
    observed = {}
    real_body = k._close_locked

    def _watch():
        observed["held"] = k._close_lock.locked()
        return real_body()

    k._close_locked = _watch
    k.close()
    assert observed["held"] is False


def test_realsense_close_is_idempotent(monkeypatch):
    c = _rs_cam(monkeypatch)
    pipe = c._pipeline
    c.close()
    c.close()
    c.close()
    assert pipe.stop_calls == 1


def test_realsense_failed_close_stays_retryable(monkeypatch):
    """A capture thread that will not exit means the close did NOT happen;
    _closed must stay unset so a later attempt can still release."""
    monkeypatch.setattr(cam_mod.RealSenseCamera, "CLOSE_JOIN_TIMEOUT_S", 0.05)
    c = _rs_cam(monkeypatch)
    pipe = c._pipeline
    stuck = threading.Event()
    t = threading.Thread(target=stuck.wait, args=(10.0,), daemon=True)
    t.start()
    c._thread = t
    c.close()
    assert pipe.stop_calls == 0
    assert c._closed is False
    stuck.set()
    t.join(timeout=2.0)
    c.close()
    assert pipe.stop_calls == 1
    assert c._closed is True


if __name__ == "__main__":  # pragma: no cover
    pytest.main([__file__, "-v"])

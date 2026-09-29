"""The Kinect capture loop must not hold the GIL while it blocks.

pyk4a's ``thread_safe`` flag is inverted relative to how it reads. Every
entry point in k4a_module.c is::

    PyThreadState *s = NULL;
    if (!thread_safe) { s = PyEval_SaveThread(); }   /* release the GIL */
    ... call into libk4a ...

so ``thread_safe=True`` -- pyk4a's default, and what this camera used --
means "hold the GIL for the whole call". Confirmed by disassembling
device_get_capture in the installed k4a_module .so: PyEval_SaveThread sits
on the ``thread_safe == 0`` branch only.

_capture_loop's steady state is ``device.get_capture(timeout=250)``, which
blocks in libusb until the next frame lands -- about 66 ms at 15 fps.
Holding the GIL across that wait freezes every other Python thread in the
process. With two Kinects the two capture threads between them held the GIL
~99% of the time while using almost no CPU, and a SAM3 detect -- which is
CPU-bound Python kernel-launch code, 0.20 s of CPU for six prompts -- took
48 s of wall time on an idle GPU. Measured in-process on this rig:

    no cameras                      0.204 s
    1 capture thread @ 30 fps       2.425 s
    1 capture thread @ 15 fps       4.572 s
    2 capture threads @ 30 fps     21.997 s
    2 capture threads @ 15 fps     48.223 s
    2 capture threads, GIL released 0.199 s

So: clear the flag on every device this camera binds, whether it opened it
or was handed one pre-started.

Everything is mocked; no hardware is touched.
"""

import pytest

import spark_real.perception.camera as cam_mod


class _FakeCapture:
    color = None


class _FakeCal:
    def get_camera_matrix(self, _kind):
        import numpy as np

        return np.eye(3, dtype=float)


class _FakeDevice:
    """Stands in for pyk4a.PyK4A, tracking the flag the fix must clear."""

    def __init__(self, thread_safe=True):
        # pyk4a's own default, which is the bug.
        self.thread_safe = thread_safe
        self.started = False

    def start(self):
        self.started = True

    def get_capture(self, timeout=None):
        return _FakeCapture()

    @property
    def calibration(self):
        return _FakeCal()


def _open_camera(monkeypatch, *, pre_started=None, **kwargs):
    """Open an AzureKinectCamera against a fake device, no thread started."""
    made = []

    def _fake_pyk4a(*_a, **kw):
        d = _FakeDevice()
        made.append(d)
        return d

    monkeypatch.setattr(cam_mod, "PyK4A", _fake_pyk4a, raising=False)
    # open() does `global PyK4A; PyK4A = _k4a.PyK4A` on its lazy-import path,
    # which OVERWRITES the patch above before the constructor is reached. So
    # patch the source module too -- that is the binding open() actually reads.
    try:
        import pyk4a as _real_pyk4a

        monkeypatch.setattr(_real_pyk4a, "PyK4A", _fake_pyk4a, raising=False)
    except ImportError:
        pass
    # Stop open() before it spawns the capture thread or builds rect maps:
    # the flag is set well before either, and neither is under test here.
    monkeypatch.setattr(
        cam_mod.AzureKinectCamera, "_build_rect_maps", lambda *a, **k: None
    )

    camera = cam_mod.AzureKinectCamera(
        device_id=0, pre_started_device=pre_started, **kwargs
    )
    try:
        camera.open()
    except Exception:
        # open() does more than we stub (threads, calibration extras). The
        # flag is set immediately after the device is bound, so whatever
        # fails later, self._device is already the object under test.
        pass
    return camera, made


pytestmark = pytest.mark.skipif(
    getattr(cam_mod, "PyK4A", None) is None and not hasattr(cam_mod, "pyk4a"),
    reason="pyk4a not importable in this environment",
)


def test_opened_device_drops_the_gil(monkeypatch):
    monkeypatch.delenv("SPARK_KINECT_HOLD_GIL", raising=False)
    camera, made = _open_camera(monkeypatch)
    assert made, "test did not exercise the PyK4A construction path"
    assert camera._device is not None
    assert camera._device.thread_safe is False, (
        "pyk4a thread_safe left set: every get_capture(timeout=250) will "
        "hold the GIL for the whole blocking wait and stall SAM3 detects"
    )


def test_pre_started_device_drops_the_gil(monkeypatch):
    """pipeline_init and server open devices themselves and hand them in.

    Those PyK4A objects never pass through the constructor call site, so a
    fix applied only at PyK4A(...) would miss the rig's real path.
    """
    monkeypatch.delenv("SPARK_KINECT_HOLD_GIL", raising=False)
    pre = _FakeDevice()
    camera, _ = _open_camera(monkeypatch, pre_started=pre)
    assert camera._device is pre
    assert pre.thread_safe is False


def test_env_escape_hatch_restores_stock_pyk4a(monkeypatch):
    monkeypatch.setenv("SPARK_KINECT_HOLD_GIL", "1")
    camera, _ = _open_camera(monkeypatch)
    assert camera._device.thread_safe is True


def test_explicit_kwarg_beats_the_env_default(monkeypatch):
    monkeypatch.setenv("SPARK_KINECT_HOLD_GIL", "1")
    camera, _ = _open_camera(monkeypatch, release_gil=True)
    assert camera._device.thread_safe is False


def test_apply_release_gil_survives_a_device_that_rejects_the_flag():
    """Never let the optimisation take a camera down."""

    class _Frozen:
        __slots__ = ()

    cam_mod._apply_release_gil(_Frozen(), True)  # must not raise

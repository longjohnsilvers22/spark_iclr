"""
Camera interface for real-world SPARK perception.

Supports:
- USB cameras (via OpenCV)
- Intel RealSense D435/D455 (RGB-D)
- Calibration loading (intrinsics + extrinsics)
"""

import os
import logging
import threading
import threading as _threading
import time as _time
from typing import Optional, Tuple

import numpy as np

logger = logging.getLogger(__name__)

# Per-subsystem liveness: every camera interface here drives the xHCI
# controller from userspace (Driver=[none]), so which capture thread stops
# first is diagnostic after a silent host death.
from spark_real import health_beat as _beat  # noqa: E402

_beat.start()

# One process-wide lock serializing EVERY USB lifecycle transition (device
# open, close, hardware_reset, bus re-enumeration) across all camera classes.
# Steady-state streaming is fine; concurrent lifecycle churn corrupts the
# xHCI controller's TRB rings (kernel logs "Event dma ... not part of TD")
# and can hard-kill the host. RLock so a composite transition (e.g.
# /api/kinect_reset closing then reopening) can nest the per-device close
# under its own hold.
USB_LIFECYCLE_LOCK = threading.RLock()

# Count of device teardowns whose SDK call has been ABANDONED but may still
# be executing. close() runs device.stop() on a worker and gives up waiting
# after 4 s, releasing USB_LIFECYCLE_LOCK while libk4a may still be issuing
# SET_INTERFACE, endpoint halts and hub traffic on the shared controller.
# Anything that would touch the bus (uhubctl power-cycle, sysfs deauthorize,
# hardware_reset, a fresh open) must refuse while this is non-zero.
# Incremented before the release worker starts, decremented by the worker
# itself, so it stays set for as long as the SDK call really runs.
_TEARDOWN_LOCK = threading.Lock()
_TEARDOWN_IN_FLIGHT = 0


def teardown_in_flight() -> bool:
    """True while an abandoned-but-still-running device teardown may be
    driving USB traffic. See _TEARDOWN_IN_FLIGHT."""
    with _TEARDOWN_LOCK:
        return _TEARDOWN_IN_FLIGHT > 0


def _teardown_begin() -> None:
    global _TEARDOWN_IN_FLIGHT
    with _TEARDOWN_LOCK:
        _TEARDOWN_IN_FLIGHT += 1


def _teardown_end() -> None:
    global _TEARDOWN_IN_FLIGHT
    with _TEARDOWN_LOCK:
        _TEARDOWN_IN_FLIGHT = max(0, _TEARDOWN_IN_FLIGHT - 1)


def await_teardown_clear(timeout: float = 15.0, poll: float = 0.1) -> bool:
    """
    Block until no abandoned teardown is in flight. Returns False on timeout.

    Callers about to touch the bus (open, reset, power-cycle) use this
    instead of assuming USB_LIFECYCLE_LOCK covered the teardown -- it does
    not once close() has stopped waiting for libk4a.
    """
    deadline = _time.monotonic() + timeout
    while teardown_in_flight():
        if _time.monotonic() >= deadline:
            return False
        _time.sleep(poll)
    return True


def usb_reset_disabled() -> bool:
    """
    True (reset paths OFF) unless SPARK_ALLOW_USB_RESET is set.

    Every software-initiated USB reset path (the Kinect capture loop's
    bounded device stop/start, the RealSense hardware_reset recovery, and
    /api/kinect_reset) fires unattended, precisely when the controller is
    ALREADY misbehaving (40 consecutive capture failures, 25 consecutive
    frame timeouts), and reset traffic on a sick controller precedes the
    ring desync in the kernel log. A degraded wrist camera is strictly
    better than a dead host.

    SPARK_NO_USB_RESET still wins if set. Read at call time, not import time.
    """
    def _set(name: str) -> bool:
        # kill switch: any value except empty/0/false counts, so a typo cannot re-enable resets
        return os.environ.get(name, "") not in ("", "0", "false", "False")
    if _set("SPARK_NO_USB_RESET"):
        return True
    return not _set("SPARK_ALLOW_USB_RESET")


# Shared librealsense context. On the RSUSB backend (mandatory on this host;
# see /etc/modprobe.d/blacklist-uvcvideo.conf) every rs.context() re-inits
# libusb and re-walks the bus, so fresh contexts per call multiply
# enumeration traffic. One context per process, created on first use.
_RS_CONTEXT = None


def _get_rs_context():
    global _RS_CONTEXT
    if _RS_CONTEXT is None:
        _RS_CONTEXT = rs.context()
    return _RS_CONTEXT


try:
    import cv2

    HAS_CV2 = True
except ImportError:
    HAS_CV2 = False

try:
    import pyrealsense2 as rs

    HAS_REALSENSE = True
except ImportError:
    HAS_REALSENSE = False

try:
    import yaml
except ImportError:
    yaml = None

# pyk4a is imported inside AzureKinectCamera.open() (not at module top)
# to avoid a glibc tpp assertion on PREEMPT_RT kernels when other native
# libs (franky, scipy BLAS) are loaded first. This is a load-ordering
# hazard on the realtime-kernel rig, not a circular import.
HAS_K4A = False
pyk4a = None
PyK4A = None
K4AConfig = None


class K4ATimeoutException(Exception):
    pass


class CameraConfig:
    """
    Camera intrinsics and extrinsics.
    """

    def __init__(
        self,
        width: int = 640,
        height: int = 480,
        fx: float = 615.0,
        fy: float = 615.0,
        cx: float = 320.0,
        cy: float = 240.0,
        extrinsic: np.ndarray = None,
    ):
        self.width = width
        self.height = height
        self.fx = fx
        self.fy = fy
        self.cx = cx
        self.cy = cy
        self.extrinsic = extrinsic if extrinsic is not None else np.eye(4)

    @property
    def intrinsic_matrix(self) -> np.ndarray:
        """
        3x3 camera intrinsic matrix.
        """
        return np.array([[self.fx, 0, self.cx], [0, self.fy, self.cy], [0, 0, 1]])

    def backproject(self, u: float, v: float, depth: float) -> np.ndarray:
        """
        Backproject pixel + depth to 3D camera frame.
        """
        x = (u - self.cx) * depth / self.fx
        y = (v - self.cy) * depth / self.fy
        return np.array([x, y, depth])

    def to_world(self, point_camera: np.ndarray) -> np.ndarray:
        """
        Transform point from camera frame to world frame.
        """
        p_hom = np.append(point_camera, 1.0)
        return (self.extrinsic @ p_hom)[:3]

    @classmethod
    def from_yaml(cls, path: str) -> "CameraConfig":
        """
        Load from YAML calibration file.
        """
        if yaml is None:
            raise RuntimeError("PyYAML not installed; CameraConfig.from_yaml unavailable")

        with open(path) as f:
            data = yaml.safe_load(f)
        return cls(
            width=data.get("width", 640),
            height=data.get("height", 480),
            fx=data["fx"],
            fy=data["fy"],
            cx=data["cx"],
            cy=data["cy"],
            extrinsic=(
                np.array(data["extrinsic"]).reshape(4, 4)
                if "extrinsic" in data
                else None
            ),
        )


class USBCamera:
    """
    Simple USB camera via OpenCV.
    """

    def __init__(self, device_id: int = 0, config: CameraConfig = None):
        self.device_id = device_id
        self.config = config or CameraConfig()
        self._cap = None

    def open(self):
        self._cap = cv2.VideoCapture(self.device_id)
        self._cap.set(cv2.CAP_PROP_FRAME_WIDTH, self.config.width)
        self._cap.set(cv2.CAP_PROP_FRAME_HEIGHT, self.config.height)
        if not self._cap.isOpened():
            raise RuntimeError(f"Cannot open camera {self.device_id}")

    def read(self, **kwargs) -> Tuple[np.ndarray, None]:
        """
        Returns (rgb_image, None). No depth from a USB camera; accepts and
        ignores depth/ir kwargs so registry-wide read loops (streaming) can
        call every camera uniformly.
        """
        if self._cap is None:
            return None, None
        ret, frame = self._cap.read()
        if not ret:
            return None, None
        return cv2.cvtColor(frame, cv2.COLOR_BGR2RGB), None

    def close(self):
        if self._cap:
            self._cap.release()

    def __enter__(self):
        self.open()
        return self

    def __exit__(self, *args):
        self.close()


class RealSenseCamera:
    """
    Intel RealSense D435/D455 RGB-D camera with background capture thread.
    """

    # hardware_reset recovery is bounded PER PROCESS (class-level counter):
    # a reset storm on a wedged link is a prime xHCI-corruption aggravator on
    # this host, where the D435i must run direct to the motherboard and a
    # reset is anything but cheap. Gaps between attempts back off 30/60 s.
    MAX_RECOVER_ATTEMPTS = 3
    RECOVER_BACKOFF_BASE_S = 30.0
    CLOSE_JOIN_TIMEOUT_S = 10.0
    _recover_attempts_total = 0
    _last_recover_ts = None

    def __init__(
        self, serial: str = None, config: CameraConfig = None, color_only: bool = False
    ):
        if not HAS_REALSENSE:
            raise ImportError("pyrealsense2 not installed. pip install pyrealsense2")
        self.serial = serial
        self.config = config or CameraConfig()
        # Stream color only (skip depth) to roughly halve the SuperSpeed UVC
        # bandwidth: the D435i color+depth+IMU triplet on a marginal link
        # desyncs the xHCI event ring ("Event dma ... not part of TD" on the
        # video endpoint) and the stream wedges. The wrist is a 2D camera, so
        # depth is dead weight.
        self._color_only = bool(color_only)
        # Env override (reversible test hook): SPARK_WRIST_COLOR_ONLY=0 re-enables
        # depth without touching config, e.g. to trial wrist-depth refinement.
        _env = os.environ.get("SPARK_WRIST_COLOR_ONLY")
        if _env is not None:
            self._color_only = _env not in ("0", "false", "False", "")
        self._pipeline = None
        self._align = None
        self._latest_rgb = None
        self._latest_depth = None
        self._lock = __import__("threading").Lock()
        self._thread = None
        self._running = False
        # Set once at the start of shutdown (request_stop). The capture loop
        # and recovery path poll it so close() can interrupt a recovery
        # mid-flight instead of racing it.
        self._closing = threading.Event()
        self._reset_disabled_logged = False
        self._recover_exhausted_logged = False
        # Teardown reentrancy state, same rationale as AzureKinectCamera:
        # pipeline.shutdown(), /api/kinect_reset, the SIGTERM handler and
        # atexit can all reach close() and none of them coordinate. Without
        # this, two closers each call rs.pipeline.stop() on the SAME
        # librealsense handle.
        self._close_lock = threading.Lock()
        self._closed = False
        self._close_in_progress = False

    def open(self):
        # Probe the USB link before configuring streams: D435i firmware needs
        # SuperSpeed for the color+depth+IMU triplet, so a USB 2.x link falls
        # back to color-only at low fps for calibration and 2D detection.
        # The whole open (probe + stream start) is a lifecycle transition and
        # is serialized against every other open/close/reset in the process.
        with USB_LIFECYCLE_LOCK:
            usb_type = "3.2"
            try:
                ctx = _get_rs_context()
                for d in ctx.query_devices():
                    if (
                        not self.serial
                        or d.get_info(rs.camera_info.serial_number) == self.serial
                    ):
                        usb_type = d.get_info(rs.camera_info.usb_type_descriptor)
                        break
            except Exception:
                pass
            usb2_mode = usb_type.startswith("2.")
            if usb2_mode:
                logger.warning(
                    "RealSense link is USB %s (not SuperSpeed): depth disabled, "
                    "color forced to 6 fps. Replace the cable with a SuperSpeed-"
                    "rated USB-A/C to USB-C to restore depth + 30 fps.",
                    usb_type,
                )

            self._usb2_mode = usb2_mode
            self._start_pipeline()

        # Start background capture thread
        self._running = True
        self._thread = __import__("threading").Thread(
            target=self._capture_loop, daemon=True
        )
        self._thread.start()

    def _start_pipeline(self):
        """
        Build and start the stream pipeline.

        Shared by open() and the capture-loop recovery path, so it must be
        safe to call repeatedly.
        """
        self._pipeline = rs.pipeline()
        rs_config = rs.config()
        if self.serial:
            rs_config.enable_device(self.serial)
        color_fps = 6 if self._usb2_mode else 15
        rs_config.enable_stream(
            rs.stream.color,
            self.config.width,
            self.config.height,
            rs.format.rgb8,
            color_fps,
        )
        self._depth_enabled = False
        if not self._usb2_mode and not self._color_only:
            try:
                rs_config.enable_stream(
                    rs.stream.depth,
                    self.config.width,
                    self.config.height,
                    rs.format.z16,
                    15,
                )
                self._depth_enabled = True
            except Exception:
                pass
        profile = self._pipeline.start(rs_config)
        if self._depth_enabled:
            self._align = rs.align(rs.stream.color)
        else:
            self._align = None

        intrinsics = (
            profile.get_stream(rs.stream.color)
            .as_video_stream_profile()
            .get_intrinsics()
        )
        self.config.fx = intrinsics.fx
        self.config.fy = intrinsics.fy
        self.config.cx = intrinsics.ppx
        self.config.cy = intrinsics.ppy

        # Drop warmup frames so the first read() is usable: early frames are
        # underexposed while auto-exposure settles and depth comes back nearly
        # empty. Tolerate early timeouts so a flaky link cannot wedge open.
        for _ in range(30):
            if self._closing.is_set():
                break
            try:
                self._pipeline.wait_for_frames(timeout_ms=1000)
            except RuntimeError:
                break

    def _recovery_allowed(self) -> bool:
        """
        Gate on hardware_reset recovery: kill-switch, per-process attempt
        bound, and exponential backoff between attempts.

        Returns False (with one-time logging) instead of raising so the
        capture loop can keep serving whatever frames still arrive; a
        disallowed recovery means the wrist camera degrades to no-frames
        rather than generating reset traffic on the shared controller.
        """
        if self._closing.is_set() or not self._running:
            return False
        if teardown_in_flight():
            # A Kinect stop() is still running on the shared controller even
            # though USB_LIFECYCLE_LOCK is free. A hardware_reset now is
            # exactly the overlapping-lifecycle pattern the lock exists to prevent.
            logger.warning(
                "RealSense: teardown in flight on another camera; skipping "
                "hardware_reset recovery"
            )
            return False
        if usb_reset_disabled():
            if not self._reset_disabled_logged:
                logger.warning(
                    "USB reset paths disabled (SPARK_ALLOW_USB_RESET unset): "
                    "RealSense hardware_reset recovery off; wrist camera "
                    "degrades to no-frames until replug/restart"
                )
                self._reset_disabled_logged = True
            return False
        cls = RealSenseCamera
        if cls._recover_attempts_total >= self.MAX_RECOVER_ATTEMPTS:
            if not self._recover_exhausted_logged:
                logger.warning(
                    "RealSense: %d/%d hardware_reset attempts used this "
                    "process; giving up on reset recovery (wrist camera "
                    "degraded). Replug or restart to retry.",
                    cls._recover_attempts_total,
                    self.MAX_RECOVER_ATTEMPTS,
                )
                self._recover_exhausted_logged = True
            return False
        if cls._last_recover_ts is not None:
            # Gap after attempt N is base * 2**(N-1): 30 s, 60 s, ...
            gap = self.RECOVER_BACKOFF_BASE_S * (
                2.0 ** (cls._recover_attempts_total - 1)
            )
            if _time.monotonic() - cls._last_recover_ts < gap:
                return False
        return True

    def _try_recover(self):
        """
        Recover a wedged D435i via hardware_reset.

        The camera can wedge into frame-timeout storms (USB link up, zero
        frames); hardware_reset clears it short of replugging. Stop the dead
        pipeline, reset, wait for re-enumeration, restart streams.

        Bounded (MAX_RECOVER_ATTEMPTS per process, exponential backoff) and
        serialized under USB_LIFECYCLE_LOCK: a reset must never overlap a
        Kinect open/close on the same xHCI controller. Honors the
        reset kill-switch via _recovery_allowed(), and aborts
        early if close() is racing it (self._closing).
        """
        if not self._recovery_allowed():
            return False
        with USB_LIFECYCLE_LOCK:
            if self._closing.is_set():
                return False
            cls = RealSenseCamera
            cls._recover_attempts_total += 1
            cls._last_recover_ts = _time.monotonic()
            logger.warning(
                "RealSense: no frames, attempting hardware_reset recovery "
                "(attempt %d/%d this process)",
                cls._recover_attempts_total,
                self.MAX_RECOVER_ATTEMPTS,
            )
            try:
                self._pipeline.stop()
            except Exception:
                pass
            try:
                ctx = _get_rs_context()
                for d in ctx.query_devices():
                    if (
                        not self.serial
                        or d.get_info(rs.camera_info.serial_number) == self.serial
                    ):
                        d.hardware_reset()
                        break
                # Re-enumeration takes ~4-5 s. Event-wait so close() can
                # interrupt instead of racing a sleeping recovery.
                if self._closing.wait(6.0):
                    return False
            except Exception as exc:
                logger.warning("RealSense hardware_reset failed: %s", exc)
            # Re-probe the link speed before re-opening: the camera may have
            # been replugged onto a different port since open() (USB2 at boot,
            # SuperSpeed now, or vice versa), and _usb2_mode decides whether
            # depth streams are enabled.
            try:
                ctx = _get_rs_context()
                for d in ctx.query_devices():
                    if (
                        not self.serial
                        or d.get_info(rs.camera_info.serial_number) == self.serial
                    ):
                        usb_type = d.get_info(rs.camera_info.usb_type_descriptor)
                        new_usb2 = usb_type.startswith("2.")
                        if new_usb2 != self._usb2_mode:
                            logger.info(
                                "RealSense link changed: usb2_mode %s -> %s",
                                self._usb2_mode,
                                new_usb2,
                            )
                            self._usb2_mode = new_usb2
                        break
            except Exception:
                pass
            if self._closing.is_set():
                return False
            try:
                self._start_pipeline()
                logger.info("RealSense: recovered after hardware_reset")
                return True
            except Exception as exc:
                logger.warning("RealSense re-open after reset failed: %s", exc)
                return False

    def _capture_loop(self):
        """
        Background thread: continuously grab frames, keep latest.
        """
        frame_count = 0
        fail_count = 0
        consec_fails = 0
        while self._running:
            try:
                frames = self._pipeline.wait_for_frames(timeout_ms=500)
                if self._align:
                    aligned = self._align.process(frames)
                    color_frame = aligned.get_color_frame()
                    depth_frame = aligned.get_depth_frame()
                else:
                    color_frame = frames.get_color_frame()
                    depth_frame = None
                if color_frame:
                    rgb = np.asanyarray(color_frame.get_data()).copy()
                    depth = None
                    if depth_frame:
                        depth = (
                            np.asanyarray(depth_frame.get_data()).astype(np.float32)
                            / 1000.0
                        )
                    with self._lock:
                        self._latest_rgb = rgb
                        self._latest_depth = depth
                    frame_count += 1
                    consec_fails = 0
                    _beat.tick("realsense")
                    if frame_count % 100 == 0:
                        logger.debug("RealSense: %d frames captured", frame_count)
                _time.sleep(0.01)
            except RuntimeError as e:
                fail_count += 1
                consec_fails += 1
                if fail_count <= 3 or fail_count % 50 == 0:
                    logger.warning("RealSense frame timeout #%d: %s", fail_count, e)
                # Self-heal: a long run of consecutive misses means the link
                # is wedged, not flaky, so try to recover. _recovery_allowed
                # enforces the kill-switch, the per-process attempt bound,
                # and exponential backoff between attempts.
                if consec_fails >= 25 and self._recovery_allowed():
                    if self._try_recover():
                        consec_fails = 0
                # Event-wait instead of sleep so close() interrupts promptly.
                self._closing.wait(0.1)

    def read(self, **kwargs) -> Tuple[Optional[np.ndarray], Optional[np.ndarray]]:
        """
        Returns latest cached (rgb, depth). Non-blocking.
        """
        with self._lock:
            return self._latest_rgb, self._latest_depth

    def request_stop(self):
        """
        Non-blocking quiesce: stop the capture loop and abort any in-flight
        recovery at its next checkpoint. Called by pipeline.shutdown() BEFORE
        the Kinects close, so a hardware_reset can never fire while another
        camera on the same controller is mid-close. close() completes the
        teardown afterwards; request_stop() alone touches no USB state.
        """
        self._running = False
        self._closing.set()

    def close(self):
        self.request_stop()
        # LOCK ORDER. _close_lock is held ONLY around the flag transitions,
        # never across _close_locked, which joins the capture thread and
        # then takes USB_LIFECYCLE_LOCK. Nesting _close_lock ->
        # USB_LIFECYCLE_LOCK would invert against
        # /api/kinect_reset._kinect_reset_locked, which already holds
        # USB_LIFECYCLE_LOCK when it reaches a camera close.
        with self._close_lock:
            if self._closed:
                return
            if self._close_in_progress:
                logger.warning(
                    "RealSense: close() already in progress on another "
                    "thread; not racing it"
                )
                return
            self._close_in_progress = True
        try:
            self._close_locked()
        finally:
            with self._close_lock:
                self._close_in_progress = False

    def _close_locked(self):
        """Body of close(); exactly one thread is in here at a time."""
        t = self._thread
        if t is not None:
            # JOIN BEFORE CLOSE: the capture thread owns the pipeline handle
            # (wait_for_frames, recovery). _closing interrupts its sleeps and
            # any recovery at the next checkpoint, so this join is normally
            # near-instant.
            t.join(timeout=self.CLOSE_JOIN_TIMEOUT_S)
            if t.is_alive():
                logger.warning(
                    "RealSense capture thread did not exit within %.0fs; "
                    "SKIPPING pipeline.stop() rather than stopping the "
                    "stream out from under a live reader (that race "
                    "corrupts xHCI state). Device releases when the "
                    "process exits.",
                    self.CLOSE_JOIN_TIMEOUT_S,
                )
                return
            self._thread = None
        pipe = self._pipeline
        if pipe is not None:
            # Drop the reference BEFORE stopping so nothing can hand the same
            # handle to a second stop() even if this one raises.
            self._pipeline = None
            with USB_LIFECYCLE_LOCK:
                try:
                    pipe.stop()
                except Exception:
                    pass
        self._closed = True

    def __enter__(self):
        self.open()
        return self

    def __exit__(self, *args):
        self.close()


def _kinect_release_gil() -> bool:
    """Whether Kinect device calls should drop the GIL while they block.

    Default True. Set SPARK_KINECT_HOLD_GIL=1 to restore pyk4a's stock
    behaviour without editing config -- an escape hatch, not a tuning knob.
    """
    return os.environ.get("SPARK_KINECT_HOLD_GIL", "").strip().lower() not in (
        "1",
        "true",
        "yes",
        "on",
    )


def _apply_release_gil(device, release_gil: bool) -> None:
    """Clear pyk4a's `thread_safe` flag so blocking k4a calls drop the GIL.

    pyk4a's flag is INVERTED relative to how it reads: in k4a_module.c every
    entry point releases the GIL (PyEval_SaveThread) only when
    `thread_safe == 0`, so pyk4a's default `thread_safe=True` means "hold
    the GIL for the whole call".

    _capture_loop blocks in `device.get_capture(timeout=250)` for ~66 ms at
    15 fps. Holding the GIL across that wait freezes the WHOLE interpreter:
    with two Kinects running the capture threads hold the GIL ~99% of the
    time, and a CPU-bound SAM3 detect (0.20 s of CPU for a 6-prompt detect)
    takes 20-50 s of wall time (measured slowdown 22x with one capture
    thread, 241x with two).

    Clearing the flag is safe because after open() returns only the capture
    thread touches the device, and close() joins that thread before it stops
    the device (and refuses to stop it at all if the join fails). pyk4a's
    thread_safe does not cover the capture decoders or the shared
    transformation_handle anyway.

    Capture/calibration objects inherit the flag from the device at creation,
    so setting it here also covers capture.color and capture.transformed_depth.
    """
    if not release_gil:
        return
    try:
        device.thread_safe = False
    except Exception as exc:  # noqa: BLE001
        logger.warning(
            "could not clear pyk4a thread_safe (%s); Kinect calls will hold "
            "the GIL and SAM3 detects will be 10-100x slower",
            exc,
        )


class AzureKinectCamera:
    """
    Azure Kinect DK RGB-D camera.
    """

    # close(): how long to wait for the capture thread before the grace
    # retry, and the grace itself. If the thread outlives both, the device
    # is NOT stopped from another thread (see close()).
    JOIN_TIMEOUT_S = 2.0
    JOIN_GRACE_S = 6.0
    # How long close() waits for libk4a's device.stop() before it gives up
    # waiting (the call itself keeps running; see teardown_in_flight).
    RELEASE_JOIN_TIMEOUT_S = 4.0

    def __init__(
        self,
        device_id: int = 0,
        config: CameraConfig = None,
        color_resolution: str = "1080P",
        depth_mode: str = "NFOV_UNBINNED",
        sync_mode: str = "STANDALONE",
        subordinate_delay_usec: int = 0,
        camera_fps: int = 30,
        pre_started_device=None,
        want_depth: bool = True,
        want_ir: bool = False,
        release_gil: bool = None,
    ):
        """
        Args:
            device_id: Device index (0 for first Kinect).
            config: Camera intrinsics/extrinsics config.
            color_resolution: "720P", "1080P", "1440P", "2160P".
            depth_mode: "NFOV_UNBINNED", "NFOV_2X2BINNED",
                "WFOV_UNBINNED", "WFOV_2X2BINNED".
            sync_mode: "STANDALONE" | "MASTER" | "SUBORDINATE".  When two
                Kinects are wired together via the 3.5mm sync cable, set
                one to MASTER and the other to SUBORDINATE.  Subordinates
                wait for the master trigger and only emit one frame per
                trigger, which dramatically reduces aggregate USB
                bandwidth versus two free-running STANDALONE devices.
                Open subordinates FIRST then master because the SDK
                requires the slave already be waiting for the trigger
                before the master starts firing.
            subordinate_delay_usec: For SUBORDINATE only.  Phase offset
                relative to master pulse; 0 means stagger as tightly as
                the depth-laser duty cycle allows (~160 us).
            camera_fps: 5, 15, or 30. WFOV_UNBINNED caps at 15 in
                hardware; higher requests are clamped.
            want_depth: Decode transformed_depth in the capture loop.
                Leave True for any camera whose depth is consumed (both
                Kinects). Setting it False stops the k4a
                transformation engine running for that camera and stops
                the per-frame colour-sized depth allocation.
            release_gil: Drop the GIL while blocking k4a calls run (see
                _apply_release_gil). None takes the process default, which
                is True unless SPARK_KINECT_HOLD_GIL is set. Leave it True:
                False is pyk4a's stock behaviour and it freezes the whole
                interpreter for the duration of every get_capture wait.
            want_ir: Build the IR preview image in the capture loop.
                Default False: it is consumed only by the ir=True
                preview/detect mode. Call set_ir_enabled(True) to turn it
                on at runtime.

        NOTE: neither flag changes what is on the USB wire. Depth mode is
        fixed at device-open time, so the depth stream is transmitted at
        the configured fps regardless. These flags cut host-side CPU, GPU
        transformation dispatches and allocation churn only. The wire is
        controlled by resolution / fps / depth_mode.
        """
        self.device_id = device_id
        self.config = config or CameraConfig(width=1920, height=1080)
        self._color_resolution = color_resolution
        self._depth_mode = depth_mode
        self._sync_mode = (sync_mode or "STANDALONE").upper()
        self._subordinate_delay_usec = int(subordinate_delay_usec)
        # WFOV_UNBINNED is the only mode that physically caps at 15 fps.
        if depth_mode == "WFOV_UNBINNED" and camera_fps > 15:
            camera_fps = 15
        self._camera_fps = int(camera_fps)
        self._device = None
        self._pre_started = pre_started_device
        self._warmup_frames = 30
        self._want_depth = bool(want_depth)
        self._want_ir = bool(want_ir)
        self._release_gil = _kinect_release_gil() if release_gil is None else bool(release_gil)

        # Dedicated capture-thread state. Multiple FastAPI executor threads can
        # call read() concurrently, but pyk4a's thread_safe flag does not cover
        # the capture property decoders or the shared transformation_handle, so
        # concurrent device access can abort the process from libk4a's C++
        # threads. Serialise into one producer thread that owns the device and
        # caches the latest RGB/depth/IR under a lock for read(). Mirrors
        # RealSenseCamera above.
        self._lock = threading.Lock()
        self._thread = None
        self._stop_flag = False
        self._latest_rgb = None
        self._latest_depth = None
        self._latest_ir = None
        self._latest_frame_ts = 0.0
        # Staleness gate state: once the cached frame ages past STALE_S,
        # read() refuses to serve it (logged once per staleness episode).
        self._stale_logged = False
        # Bounded in-process capture-loop restarts (see _capture_loop).
        self._loop_restarts = 0
        # Teardown reentrancy state. Built HERE, not lazily in close(): a
        # lazy `hasattr` check is itself a race, and two first-time closers
        # (SIGTERM handler and on_event("shutdown"), or /api/kinect_reset
        # and atexit) would each build a SEPARATE Lock and both run
        # device.stop() on the same libk4a handle.
        self._close_lock = threading.Lock()
        self._closed = False
        self._close_in_progress = False

    # Max age (seconds) of the cached frame that read() will serve. A wedged
    # device otherwise serves the SAME frozen frame forever: perception, the
    # executor's redetect and the wrist servo all silently operate on a stale
    # scene and the arm moves on stale coordinates. Env-tunable.
    STALE_S = float(os.environ.get("SPARK_KINECT_STALE_S", "2.0"))
    # Consecutive get_capture failures before the loop attempts one bounded
    # in-process device restart (mirrors the RealSense _try_recover pattern).
    _FAIL_RESTART_THRESHOLD = 40
    _MAX_LOOP_RESTARTS = 2

    def set_ir_enabled(self, enabled: bool) -> None:
        """
        Turn the IR preview image on/off at runtime (see want_ir).

        Takes effect on the next captured frame; a consumer that has just
        enabled it may get ir=None for one frame period. Cheap and
        idempotent, so callers can set it on every request.
        """
        self._want_ir = bool(enabled)

    def last_frame_age_s(self):
        """Age of the newest cached frame in seconds, or None before the
        first frame. Surfaced into pipeline.get_status so the UI can show a
        dead camera instead of a frozen picture."""
        with self._lock:
            ts = self._latest_frame_ts
        if not ts:
            return None
        return max(0.0, _time.time() - ts)

    def open(self):
        """
        Start the Azure Kinect device and update intrinsics.
        """
        # USB_LIFECYCLE_LOCK is necessary but NOT sufficient: an abandoned
        # device.stop() can still be running with the lock released, and
        # opening a device while libk4a is mid-teardown on the same
        # controller is the overlap this whole subsystem exists to prevent.
        # Checked before anything else so no SDK state is built first.
        if not await_teardown_clear():
            raise RuntimeError(
                f"Kinect {self.device_id}: refusing to open while a device "
                "teardown is still in flight (libk4a stop() has not "
                "returned). Restart the server rather than layering an open "
                "on top of an unfinished close."
            )
        global pyk4a, PyK4A, K4AConfig, HAS_K4A, K4ATimeoutException
        if not HAS_K4A:
            import pyk4a as _k4a

            pyk4a = _k4a
            PyK4A = _k4a.PyK4A
            K4AConfig = _k4a.Config
            from pyk4a.errors import K4ATimeoutException as _kte

            K4ATimeoutException = _kte
            HAS_K4A = True
        res_map = {
            "720P": pyk4a.ColorResolution.RES_720P,
            "1080P": pyk4a.ColorResolution.RES_1080P,
            "1440P": pyk4a.ColorResolution.RES_1440P,
            "2160P": pyk4a.ColorResolution.RES_2160P,
        }
        depth_map = {
            "NFOV_UNBINNED": pyk4a.DepthMode.NFOV_UNBINNED,
            "NFOV_2X2BINNED": pyk4a.DepthMode.NFOV_2X2BINNED,
            "WFOV_UNBINNED": pyk4a.DepthMode.WFOV_UNBINNED,
            "WFOV_2X2BINNED": pyk4a.DepthMode.WFOV_2X2BINNED,
        }
        fps_map = {
            5: pyk4a.FPS.FPS_5,
            15: pyk4a.FPS.FPS_15,
            30: pyk4a.FPS.FPS_30,
        }
        sync_map = {
            "STANDALONE": pyk4a.WiredSyncMode.STANDALONE,
            "MASTER": pyk4a.WiredSyncMode.MASTER,
            "SUBORDINATE": pyk4a.WiredSyncMode.SUBORDINATE,
        }
        k4a_config = K4AConfig(
            color_resolution=res_map.get(
                self._color_resolution, pyk4a.ColorResolution.RES_720P
            ),
            depth_mode=depth_map.get(self._depth_mode, pyk4a.DepthMode.NFOV_UNBINNED),
            camera_fps=fps_map.get(self._camera_fps, pyk4a.FPS.FPS_15),
            synchronized_images_only=True,
            wired_sync_mode=sync_map.get(
                self._sync_mode, pyk4a.WiredSyncMode.STANDALONE
            ),
            subordinate_delay_off_master_usec=(
                self._subordinate_delay_usec if self._sync_mode == "SUBORDINATE" else 0
            ),
        )
        # Device bring-up (start + warmup + calibration read) is a lifecycle
        # transition: serialize it against every other camera open/close/
        # reset in the process (notably RealSense hardware_reset recovery).
        with USB_LIFECYCLE_LOCK:
            if self._pre_started is not None:
                self._device = self._pre_started
                self._pre_started = None
            else:
                self._device = PyK4A(k4a_config, device_id=self.device_id)
                self._device.start()

            # Do this before the first get_capture below, and do it here
            # rather than at the PyK4A() call so it also covers a device
            # handed in pre-started (pipeline_init/server open those).
            _apply_release_gil(self._device, self._release_gil)

            # Warm up auto-exposure. Subordinates are armed but produce no
            # frames until the master starts firing triggers; calling
            # get_capture() on them here would block forever. Skip warmup
            # in that case and let the caller do a deferred warmup after
            # the master is up.
            if self._sync_mode != "SUBORDINATE":
                for _ in range(self._warmup_frames):
                    self._device.get_capture()

            # Update intrinsics from calibration. Calibration is read from
            # the device factory data and does not require a captured frame.
            cal = self._device.calibration
            K = cal.get_camera_matrix(pyk4a.CalibrationType.COLOR)
            self.config.fx = float(K[0, 0])
            self.config.fy = float(K[1, 1])
            self.config.cx = float(K[0, 2])
            self.config.cy = float(K[1, 2])

            # Update resolution from first capture, but skip on subordinate
            # for the same reason as the warmup loop.
            if self._sync_mode != "SUBORDINATE":
                test_cap = self._device.get_capture()
                if test_cap.color is not None:
                    self.config.height, self.config.width = test_cap.color.shape[:2]

        if self._sync_mode == "SUBORDINATE":
            # Subordinate resolution matches master's; the calibration
            # block above already set fx/fy/cx/cy from the device's
            # factory intrinsics, which is what downstream code needs.
            color_res_map = {
                "720P": (1280, 720),
                "1080P": (1920, 1080),
                "1440P": (2560, 1440),
                "2160P": (3840, 2160),
            }
            w, h = color_res_map.get(self._color_resolution, (1920, 1080))
            self.config.width, self.config.height = w, h

        # The SDK serves images in DISTORTED pixel space (per k4a docs "2D
        # coordinates always refer to the distorted image"), but every
        # downstream consumer here deprojects with a pure pinhole model.
        # Rectify RGB + depth once in the capture thread so the pinhole
        # math is exact everywhere. The LUT is built the way the SDK's own
        # undistort example does it (k4a_calibration_3d_to_2d per ray);
        # the maps reproject onto the SAME K, so fx/fy/cx/cy stay valid.
        self._rect_maps = None
        try:
            dist = cal.get_distortion_coefficients(pyk4a.CalibrationType.COLOR)
            if np.any(np.abs(np.asarray(dist)) > 1e-9):
                self._rect_maps = self._build_rect_maps(
                    cal,
                    K,
                    np.asarray(dist, dtype=np.float64),
                    (self.config.width, self.config.height),
                )
        except Exception as exc:
            logger.warning(
                "Kinect %d: distortion rectification unavailable (%s); "
                "captures stay in distorted pixel space",
                self.device_id,
                exc,
            )

        # All single-threaded device setup is done; from this point on,
        # ONLY the capture thread touches self._device.
        self._stop_flag = False
        self._latest_frame_ts = _time.time()
        self._thread = threading.Thread(
            target=self._capture_loop,
            daemon=True,
            name=f"kinect{self.device_id}-capture",
        )
        self._thread.start()

    def _build_rect_maps(self, cal, K, dist, size):
        """
        Undistortion LUT built the way the SDK's undistort example does:
        for every pixel of the target pinhole image, cast the ray through K
        and ask the SDK (k4a_calibration_3d_to_2d via pyk4a) where that ray
        lands in the real, distorted image. The field is smooth, so it is
        sampled every 8 px and bilinearly upsampled; cv2 only executes the
        remap. The OpenCV Brown-Conrady map is computed as a cross-check
        (disagreement logged) and fills the few fringe pixels the SDK marks
        invalid. Returns CV_16SC2 maps for fast remap.
        """
        w, h = size
        fx, fy = float(K[0, 0]), float(K[1, 1])
        cx, cy = float(K[0, 2]), float(K[1, 2])
        step = 8
        us = np.linspace(0.0, w - 1.0, int(np.ceil(w / step)) + 1)
        vs = np.linspace(0.0, h - 1.0, int(np.ceil(h / step)) + 1)
        mx = np.full((len(vs), len(us)), np.nan, np.float32)
        my = np.full((len(vs), len(us)), np.nan, np.float32)
        for j, v in enumerate(vs):
            ry = (v - cy) / fy
            for i, u in enumerate(us):
                rx = (u - cx) / fx
                try:
                    du, dv = cal.convert_3d_to_2d(
                        (rx * 1000.0, ry * 1000.0, 1000.0), pyk4a.CalibrationType.COLOR
                    )
                    mx[j, i] = du
                    my[j, i] = dv
                except Exception:
                    pass  # ray outside the calibrated FOV; cv2 fills below
        map_x = cv2.resize(mx, size, interpolation=cv2.INTER_LINEAR)
        map_y = cv2.resize(my, size, interpolation=cv2.INTER_LINEAR)
        c1, c2 = cv2.initUndistortRectifyMap(K, dist, None, K, size, cv2.CV_32FC1)
        ok = np.isfinite(map_x) & np.isfinite(map_y)
        if ok.any():
            dd = np.hypot(map_x[ok] - c1[ok], map_y[ok] - c2[ok])
            logger.info(
                "Kinect %d rectification: SDK-built LUT (%.0f%% px), OpenCV "
                "model agrees to %.2f px max / %.2f px median",
                self.device_id,
                100.0 * ok.mean(),
                float(dd.max()),
                float(np.median(dd)),
            )
        map_x = np.where(ok, map_x, c1).astype(np.float32)
        map_y = np.where(ok, map_y, c2).astype(np.float32)
        gx, gy = np.meshgrid(
            np.arange(w, dtype=np.float32), np.arange(h, dtype=np.float32)
        )
        disp = np.hypot(map_x - gx, map_y - gy)
        logger.info(
            "Kinect %d color distortion: max correction %.1f px, median "
            "%.1f px; rectifying all captures",
            self.device_id,
            float(disp.max()),
            float(np.median(disp)),
        )
        m1, m2 = cv2.convertMaps(map_x, map_y, cv2.CV_16SC2)
        return m1, m2

    def _capture_loop(self):
        """
        Single-producer capture loop.

        Polls get_capture with a short timeout so the loop can react
        to close(); pulls every consumer-relevant image off the
        capture handle inside this thread (so the C++ lazy decoders
        and the shared transformation_handle are never touched from
        anywhere else); stashes copies into latest_* under the lock.
        Consumers read those copies via read(); they never see the
        underlying capture handle.

        QUEUE DRAIN: after each blocking get_capture, non-blocking-poll
        the device for additional buffered frames and skip ahead to the
        newest. The K4A SDK maintains an internal frame queue; if a
        consumer stalls this thread the SDK keeps queueing frames, and
        processing them one-at-a-time falls further behind until the SDK
        drops frames and wedges the device. Draining at the C level keeps
        the loop on the latest frame. ``timeout_ms=0`` polls without blocking.
        """
        device = self._device
        consecutive_failures = 0
        while not self._stop_flag:
            try:
                # 250ms is short enough for close() to be responsive
                # and long enough that no frame is missed at 30fps.
                capture = device.get_capture(timeout=250)
                consecutive_failures = 0
                _beat.tick("kinect%d" % self.device_id)
            except K4ATimeoutException:
                continue
            except Exception as exc:
                consecutive_failures += 1
                logger.warning("Kinect %d get_capture failed: %s", self.device_id, exc)
                # Bounded in-process recovery, mirroring RealSense
                # _try_recover: after a run of hard failures, stop+start the
                # device once or twice under USB_LIFECYCLE_LOCK. Honors the
                # reset kill-switch (OFF by default): a device restart
                # re-runs SET_INTERFACE bandwidth negotiation while the OTHER
                # Kinect's isochronous reservation is live on the same
                # controller.
                if (
                    consecutive_failures >= self._FAIL_RESTART_THRESHOLD
                    and self._loop_restarts < self._MAX_LOOP_RESTARTS
                    and not usb_reset_disabled()
                ):
                    self._loop_restarts += 1
                    logger.error(
                        "Kinect %d: %d consecutive capture failures; bounded "
                        "device restart %d/%d",
                        self.device_id,
                        consecutive_failures,
                        self._loop_restarts,
                        self._MAX_LOOP_RESTARTS,
                    )
                    try:
                        with USB_LIFECYCLE_LOCK:
                            if self._stop_flag:
                                break
                            try:
                                device.stop()
                            except Exception:  # noqa: BLE001
                                pass
                            _time.sleep(1.0)
                            device.start()
                        consecutive_failures = 0
                        continue
                    except Exception as rexc:  # noqa: BLE001
                        logger.error(
                            "Kinect %d device restart failed: %s",
                            self.device_id,
                            rexc,
                        )
                _time.sleep(0.05)
                continue
            # Non-blocking drain: skip ahead to the newest buffered frame
            # if the consumer stalled this loop. Intermediate capture
            # handles are discarded; only the latest survives to decode.
            drained = 0
            while True:
                try:
                    newer = device.get_capture(timeout=0)
                except K4ATimeoutException:
                    break
                except Exception:
                    break
                if newer is None:
                    break
                capture = newer
                drained += 1
                if drained >= 30:  # cap so a runaway stall doesn't spin forever
                    break
            if drained >= 3:
                logger.debug(
                    "Kinect %d drained %d backlogged frames (consumer " "fell behind)",
                    self.device_id,
                    drained,
                )
            try:
                color = capture.color
                if color is None:
                    continue
                rgb = color[:, :, :3][:, :, ::-1].copy()
                # transformed_depth runs the k4a transformation engine (a GPU
                # dispatch) and allocates a colour-sized uint16 buffer per
                # frame. Gated so a camera whose depth nobody consumes stops
                # paying for it; default stays on because sideview depth is
                # what grasp height comes from.
                depth_m = None
                if self._want_depth:
                    tdepth = capture.transformed_depth
                    if tdepth is not None:
                        depth_m = tdepth.astype(np.float32) / 1000.0
                # Rectification is deferred to read(): remapping every captured
                # frame here starves the velocity-servo loop of CPU, and most
                # frames are never consumed.
                #
                # IR is OFF by default; it is consumed only in the ir=True
                # preview/detect mode. Consumers call set_ir_enabled(True) and
                # the next frames carry it.
                ir_rgb = None
                if self._want_ir:
                    ir_raw = getattr(capture, "ir", None)
                    if ir_raw is not None and ir_raw.size > 0:
                        ir_f = ir_raw.astype(np.float32)
                        hi = max(float(np.percentile(ir_f, 99.5)), 200.0)
                        ir_stretched = np.clip(ir_f / hi, 0.0, 1.0) ** 0.6
                        ir_norm = (ir_stretched * 255.0).astype(np.uint8)
                        ir_rgb = np.stack([ir_norm, ir_norm, ir_norm], axis=-1)
            except Exception as exc:
                logger.warning("Kinect %d decode failed: %s", self.device_id, exc)
                continue
            finally:
                # Drop the handle NOW, not at the next rebind: pyk4a hands
                # back a PyCapsule with a k4a_capture_release destructor, and
                # holding it across the next blocking 250 ms get_capture keeps
                # a buffer out of libk4a's bounded internal pool.
                capture = None
            with self._lock:
                self._latest_rgb = rgb
                self._latest_depth = depth_m
                self._latest_ir = ir_rgb
                self._latest_rectified = False
                self._latest_frame_ts = _time.time()

    def read(self, depth: bool = True, ir: bool = False):
        """
        Returns (rgb, depth) by default; (rgb, depth, ir) when ir=True.

        Served from the single-producer capture thread's cached frame
        under self._lock. Multiple consumers (streamer, perception,
        CBF) can call this concurrently without entering pyk4a from
        more than one thread.

        Args:
            depth: If False, callers receive None in the depth slot
                regardless of what the capture thread cached.
            ir: If True, return (rgb, depth, ir) instead of (rgb, depth).
        """
        # Wait briefly for the first frame to arrive (open() may have
        # just spawned the thread; subordinate cameras only start
        # producing once master is firing).
        deadline = _time.time() + 2.0
        while _time.time() < deadline:
            with self._lock:
                # STALENESS GATE: never serve a cached frame older than
                # STALE_S. A wedged device (USB drop, depth-engine stall)
                # otherwise freezes the scene image silently and the arm
                # moves on stale coordinates. Within the deadline, keep
                # polling in case the capture thread recovers.
                _fresh = (
                    self._latest_frame_ts > 0
                    and (_time.time() - self._latest_frame_ts) <= self.STALE_S
                )
                if self._latest_rgb is not None and _fresh:
                    self._stale_logged = False
                    # Deferred rectification: the capture thread stores raw
                    # frames; the first consumer of a frame pays one remap and
                    # the result is cached for everyone else.
                    if (
                        self._rect_maps is not None
                        and not getattr(self, "_latest_rectified", True)
                        and self._latest_rgb.shape[1] == self._rect_maps[0].shape[1]
                    ):
                        m1, m2 = self._rect_maps
                        self._latest_rgb = cv2.remap(
                            self._latest_rgb, m1, m2, cv2.INTER_LINEAR
                        )
                        if self._latest_depth is not None:
                            # Nearest neighbour: never blend depths across
                            # an object edge.
                            self._latest_depth = cv2.remap(
                                self._latest_depth, m1, m2, cv2.INTER_NEAREST
                            )
                        self._latest_rectified = True
                    rgb = self._latest_rgb
                    depth_m = self._latest_depth if depth else None
                    ir_img = self._latest_ir
                    if ir:
                        return rgb, depth_m, ir_img
                    return rgb, depth_m
            _time.sleep(0.005)
        # First-frame timeout OR stale cache; treat as no signal. Logged once
        # per staleness episode so a 15 Hz stream doesn't flood the log.
        with self._lock:
            if self._latest_rgb is not None and not self._stale_logged:
                self._stale_logged = True
                age = _time.time() - self._latest_frame_ts
                logger.error(
                    "Kinect %d: cached frame is %.1fs old (> %.1fs); serving "
                    "NO frame instead of a frozen scene. Device is likely "
                    "wedged; see /api/kinect_reset.",
                    self.device_id,
                    age,
                    self.STALE_S,
                )
        return (None, None, None) if ir else (None, None)

    def close(self) -> bool:
        """
        Release the device. Returns True ONLY on a full, clean release.

        Returns False when the capture thread would not exit, or when the
        SDK's stop() had to be abandoned -- i.e. when a live thread may
        still be doing libusb I/O on this device. Callers that are about to
        power-cycle, deauthorize or reopen the device MUST check this;
        doing any of those to a device with a live capture thread is the
        concurrent-teardown pattern behind the xHCI ring corruption.

        A failed close is RETRYABLE: _closed is set only after a clean
        release, so a later close() can try again.
        """
        # Reentrancy guard: SIGTERM handler + atexit + on_event("shutdown")
        # may all race to close the same device. One closer at a time; a
        # completed close short-circuits, an in-flight one reports failure
        # rather than tearing down the same handle twice.
        with self._close_lock:
            if self._closed:
                return True
            if self._close_in_progress:
                logger.warning(
                    "Kinect %d: close() already in progress on another "
                    "thread; reporting failure rather than racing it",
                    self.device_id,
                )
                return False
            self._close_in_progress = True
        ok = False  # an exception out of _close_locked is a failed close
        try:
            ok = self._close_locked()
        finally:
            with self._close_lock:
                self._close_in_progress = False
                if ok:
                    self._closed = True
        return ok

    def _close_locked(self) -> bool:
        """Body of close(); exactly one thread is in here at a time."""
        ok = True
        self._stop_flag = True
        t = self._thread
        if t is not None:
            # JOIN BEFORE CLOSE: the capture thread owns the device handle
            # (get_capture + decoders). get_capture polls with a 250 ms
            # timeout, so a healthy thread exits almost immediately; give a
            # grace retry, and if it STILL will not die, libk4a is wedged
            # and stopping the device from this thread would race the
            # in-flight get_capture -- the exact concurrent-teardown pattern
            # behind the xHCI TRB corruption. Leave the handle to process
            # exit instead.
            t.join(timeout=self.JOIN_TIMEOUT_S)
            if t.is_alive():
                logger.warning(
                    "Kinect %d capture thread still running after %.0fs; "
                    "waiting up to %.0fs more before releasing the device",
                    self.device_id,
                    self.JOIN_TIMEOUT_S,
                    self.JOIN_GRACE_S,
                )
                t.join(timeout=self.JOIN_GRACE_S)
            if t.is_alive():
                logger.warning(
                    "Kinect %d capture thread did not exit; NOT stopping "
                    "the device from another thread (would race libk4a "
                    "mid-capture). Device releases at process exit; the "
                    "depth MCU may need a power-cycle on next launch.",
                    self.device_id,
                )
                # FAILED close: the handle is still open and a live thread
                # may still be inside libusb on it. Report it so callers do
                # not power-cycle or reopen underneath that thread, and
                # leave _closed unset so a later attempt can retry.
                return False
            self._thread = None

        device = self._device
        if device is None:
            return ok
        self._device = None

        # Run device.stop() on a worker so a hung call inside libk4a
        # can't wedge the whole shutdown path. pyk4a's stop() already
        # calls _device_close() internally; calling close() again
        # raises "Device is not opened", so it is not called here. The stop itself is a
        # lifecycle transition: hold USB_LIFECYCLE_LOCK for its bounded
        # window so it cannot overlap a RealSense reset or another open.
        # _teardown_begin/_teardown_end bracket the SDK call itself, not the
        # wait for it: when the 4 s budget expires USB_LIFECYCLE_LOCK is dropped
        # while device.stop() may still be issuing SET_INTERFACE / endpoint
        # halts / hub traffic. teardown_in_flight() stays True until the
        # worker really returns, and every bus-touching path checks it.
        def _release():
            try:
                device.stop()
            except Exception as exc:
                logger.warning("Kinect %d stop failed: %s", self.device_id, exc)
            finally:
                _teardown_end()

        _teardown_begin()
        with USB_LIFECYCLE_LOCK:
            t = _threading.Thread(
                target=_release, daemon=True, name=f"kinect{self.device_id}-release"
            )
            t.start()
            t.join(timeout=self.RELEASE_JOIN_TIMEOUT_S)
            if t.is_alive():
                logger.warning(
                    "Kinect %d stop/close exceeded %.0fs, abandoning the wait. "
                    "teardown_in_flight() stays True until libk4a returns, so "
                    "no reset/power-cycle/open can overlap it. Depth MCU may "
                    "need a reset on next launch.",
                    self.device_id,
                    self.RELEASE_JOIN_TIMEOUT_S,
                )
                ok = False
        return ok

    def __enter__(self):
        self.open()
        return self

    def __exit__(self, *args):
        self.close()

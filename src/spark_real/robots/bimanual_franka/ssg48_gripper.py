"""
Source Robotics SSG-48 parallel-jaw gripper driver (Spectral BLDC / slcan).

Port of the external ROS2 gripper driver into a
standalone class that satisfies the same surface as
:class:`spark_real.robots.bimanual_franka.dynamixel_gripper.DynamixelGripper`,
so :class:`BimanualFrankaDriver` can be wired up with either backend by
selecting ``grippers.type`` in the bimanual YAML.

Key hardware differences vs. the Dynamixel rig:

* Each gripper sits on its OWN slcan CAN bus (e.g. left=``/dev/ttyACM1``,
  right=``/dev/ttyACM0``), driven via ``Spectral_BLDC``. The bus is
  point-to-point, so the dual wrapper opens two independent
  ``CanCommunication`` instances rather than sharing one port.
* Position is reported as integer encoder counts 0..255 (0 = fully open,
  255 = fully closed), *not* in meters. The width band in meters is a
  per-rig calibration: configure ``width_open_m`` / ``width_closed_m``.
* Activation + calibration are a stateful handshake (see
  ``_ACTIVATING`` / ``_CALIBRATING`` states). The firmware sweeps both
  limits autonomously when ``Send_gripper_calibrate()`` is invoked, and
  the host polls a 0-byte ``Send_gripper_data_pack()`` to keep status
  responses flowing without commanding motion.

This driver is hardware-only. ``Spectral_BLDC`` MUST be installed and
the CAN bus reachable: failures raise loudly rather than degrade to a
silent simulator, so a missing gripper never masquerades as a working
one in production.
"""

from __future__ import annotations

import logging
import threading
import time
from dataclasses import dataclass
from typing import Optional

logger = logging.getLogger(__name__)

# Optional dep: Spectral_BLDC is the CAN/BLDC motor lib for the SSG48 gripper,
# installed only on the bimanual rig. Guard it like the other backends (franky,
# unitree) so importing this module does NOT crash single-arm hosts (ur10e,
# franka). It's only touched at gripper connect time; _require() raises a
# clear error then if it's genuinely missing.
try:
    import Spectral_BLDC as _Spectral
except ImportError:
    _Spectral = None
    logger.warning(
        "Spectral_BLDC not installed; SSG48 gripper unavailable "
        "(fine unless this is the bimanual rig). pip install Spectral_BLDC"
    )


def _require_spectral():
    if _Spectral is None:
        raise RuntimeError(
            "SSG48 gripper requires the Spectral_BLDC package "
            "(pip install Spectral_BLDC); not installed on this host."
        )
    return _Spectral


# Hardware defaults (override via SSG48Config).
_DEFAULT_BUSTYPE = "slcan"
_DEFAULT_BITRATE = 1_000_000
_DEFAULT_SPEED = 60  # 0..255; the ROS configs use 60..100
_DEFAULT_FORCE_MA = 500  # 0..1300 mA
_HARDWARE_MAX_FORCE_MA = 1300

# Position frame counts (matches msg_gripper_node.py docstring).
_COUNTS_OPEN = 0
_COUNTS_CLOSED = 255

# Width band assumed when the YAML omits it. Tune per rig.
WIDTH_MIN_M = 0.000
WIDTH_MAX_M = 0.090

# State machine, mirrors msg_gripper_node.py exactly so a side-by-side
# read against the ROS source is unambiguous.
_ACTIVATING = "activating"
_CALIBRATING = "calibrating"
_READY = "ready"
_FAILED = "failed"

# Cadence (50 ms) matches the ROS timer. The poll thread runs even in _READY
# so present_position stays fresh for get_present_width() callers.
_POLL_PERIOD_S = 0.05
_ACTIVATE_TIMEOUT_S = 10.0
_CALIBRATE_TIMEOUT_S = 120.0


@dataclass
class SSG48Config:
    # Per-gripper wiring + calibration.

    channel: str  # e.g. "/dev/ttyACM0"
    node_id: int = 0
    bustype: str = _DEFAULT_BUSTYPE
    bitrate: int = _DEFAULT_BITRATE
    default_speed: int = _DEFAULT_SPEED
    default_force_ma: int = _DEFAULT_FORCE_MA
    max_force_ma: int = _HARDWARE_MAX_FORCE_MA
    position_deadband: int = 5  # drop commands within N counts of last
    auto_calibrate: bool = False  # run sweep on connect if uncalibrated
    invert_position: bool = False  # flip if the linkage is mounted reversed
    width_open_m: float = WIDTH_MAX_M
    width_closed_m: float = WIDTH_MIN_M


class SSG48Gripper:
    """
    Single Source Robotics SSG-48 parallel jaw on its own slcan bus.

    Public surface mirrors :class:`DynamixelGripper` so the bimanual driver
    is gripper-backend-agnostic. All blocking helpers (``grasp_to_width``)
    poll the same background telemetry thread that the activation /
    calibration handshake uses.
    """

    GRIPPER_TYPE = "ssg48"

    def __init__(self, config: SSG48Config):
        self.config = config
        self._lock = threading.Lock()
        # Serializes ALL CAN TX (motion commands + telemetry polls). The poll
        # thread and command callers share one pyserial handle; slcan frames are
        # ASCII lines, and interleaved writes from two threads corrupt both.
        self._tx_lock = threading.Lock()
        # monotonic time of the last successfully unpacked status frame;
        # lets readers distinguish live telemetry from a stale cache.
        self._last_rx: float = 0.0
        self._last_commanded_counts: int = _COUNTS_OPEN
        self._holding: bool = False
        self._connected: bool = False

        # Spectral_BLDC handles, populated in connect().
        self._comm = None
        self._motor = None

        # Background poll thread + state.
        self._poll_thread: Optional[threading.Thread] = None
        self._poll_stop = threading.Event()
        self._state: str = _ACTIVATING
        self._state_entered: float = 0.0
        self._state_lock = threading.Lock()

    # connection lifecycle
    def connect(self) -> bool:
        """
        Open the CAN bus and run the activation/calibration handshake.

        Raises ``RuntimeError`` if the bus cannot be opened or the
        handshake fails to reach the ``_READY`` state inside the
        activate / calibrate timeouts. No silent simulator path: a
        missing gripper must surface loudly.
        """
        try:
            _spectral = _require_spectral()
            self._comm = _spectral.CanCommunication(
                bustype=self.config.bustype,
                channel=self.config.channel,
                bitrate=self.config.bitrate,
            )
            self._motor = _spectral.SpectralCAN(
                node_id=self.config.node_id,
                communication=self._comm,
            )
        except Exception as e:
            raise RuntimeError(
                f"SSG48 CAN init failed on {self.config.channel}: {e}"
            ) from e

        self._set_state(_ACTIVATING)
        self._poll_stop.clear()
        self._poll_thread = threading.Thread(
            target=self._poll_loop,
            name=f"ssg48-{self.config.channel}",
            daemon=True,
        )
        self._poll_thread.start()

        # Block until handshake completes. auto_calibrate=True can take up
        # to _CALIBRATE_TIMEOUT_S; without it the handshake should finish
        # well inside the activate window.
        deadline = time.monotonic() + (
            _ACTIVATE_TIMEOUT_S
            + (_CALIBRATE_TIMEOUT_S if self.config.auto_calibrate else 0.0)
        )
        while time.monotonic() < deadline:
            with self._state_lock:
                st = self._state
            if st == _READY:
                self._connected = True
                return True
            if st == _FAILED:
                self._stop_poll()
                raise RuntimeError(
                    f"SSG48 on {self.config.channel} failed startup "
                    f"handshake (state=_FAILED). Check power, CAN wiring, "
                    f"and that the jaws move freely."
                )
            time.sleep(0.05)

        self._stop_poll()
        raise RuntimeError(
            f"SSG48 on {self.config.channel} did not reach _READY within "
            f"{int(_ACTIVATE_TIMEOUT_S + (_CALIBRATE_TIMEOUT_S if self.config.auto_calibrate else 0.0))}s"
        )

    def disconnect(self) -> None:
        self._stop_poll()
        if self._comm is not None:
            try:
                self._comm.bus.shutdown()
            except Exception:
                logger.exception(
                    "SSG48 CAN bus shutdown failed (%s)", self.config.channel
                )
        self._connected = False

    def _stop_poll(self) -> None:
        self._poll_stop.set()
        if self._poll_thread is not None:
            self._poll_thread.join(timeout=1.0)
            self._poll_thread = None

    # commands
    def homing(self) -> None:
        # Drive to fully-open as a baseline.
        self.move_to_width(self.config.width_open_m)

    def move_to_width(self, width_m: float, speed: Optional[float] = None) -> None:
        counts = self._width_to_counts(width_m)
        speed_int = self._coerce_speed(speed)
        force = self.config.default_force_ma
        with self._lock:
            self._last_commanded_counts = counts
            self._holding = False
        self._send_motion(counts, speed_int, force)

    def grasp_to_width(
        self,
        width_m: float,
        force_n: Optional[float] = None,
        speed: Optional[float] = None,
    ) -> bool:
        """
        Close to ``width_m`` with a current-limited grip.

        ``force_n`` is mapped linearly to mA via an 80 mA/N proxy (same
        scaling the Dynamixel driver uses) and clamped to the configured
        ``max_force_ma``. Returns ``True`` once the reported position
        stops moving for 100 ms, ``False`` after 2 s.
        """
        counts = self._width_to_counts(width_m)
        speed_int = self._coerce_speed(speed)
        force_ma = self.config.default_force_ma
        if force_n is not None:
            force_ma = int(max(50, min(self.config.max_force_ma, force_n * 80.0)))
        with self._lock:
            self._last_commanded_counts = counts
            self._holding = True
        self._send_motion(counts, speed_int, force_ma)

        last = self.get_present_width()
        deadline = time.time() + 2.0
        stable_since: Optional[float] = None
        while time.time() < deadline:
            time.sleep(0.030)
            now = self.get_present_width()
            if abs(now - last) < 0.0005:
                stable_since = stable_since or time.time()
                if time.time() - stable_since > 0.10:
                    return True
            else:
                stable_since = None
            last = now
        return False

    def open(self, speed: Optional[float] = None) -> None:
        self.move_to_width(self.config.width_open_m, speed=speed)

    def close(
        self, force_n: Optional[float] = None, speed: Optional[float] = None
    ) -> bool:
        return self.grasp_to_width(
            self.config.width_closed_m, force_n=force_n, speed=speed
        )

    # queries
    def get_present_width(self) -> float:
        if self._motor is None:
            raise RuntimeError(
                f"SSG48({self.config.channel}) not connected, call connect() first"
            )
        raw = getattr(self._motor, "gripper_position", None)
        if raw is None:
            return self._counts_to_width(self._last_commanded_counts)
        return self._counts_to_width(self._from_motor(int(raw)))

    def get_position_norm(self) -> float:
        # Return 0.0..1.0 (0 = open, 1 = closed).
        w = self.get_present_width()
        span = max(1e-6, self.config.width_open_m - self.config.width_closed_m)
        frac = 1.0 - (w - self.config.width_closed_m) / span
        return max(0.0, min(1.0, frac))

    def is_object_detected(self) -> bool:
        with self._lock:
            if not self._holding:
                return False
        present_counts = self._width_to_counts(self.get_present_width())
        return present_counts + 2 < self._last_commanded_counts

    @property
    def state(self) -> str:
        with self._state_lock:
            return self._state

    # internal: count <-> width + invert handling
    def _width_to_counts(self, width_m: float) -> int:
        span = (self.config.width_open_m - self.config.width_closed_m) or 1e-6
        clamped = max(
            min(self.config.width_open_m, self.config.width_closed_m),
            min(
                max(self.config.width_open_m, self.config.width_closed_m),
                float(width_m),
            ),
        )
        frac = 1.0 - (clamped - self.config.width_closed_m) / span
        return int(max(0, min(255, round(frac * 255.0))))

    def _counts_to_width(self, counts: int) -> float:
        span = self.config.width_open_m - self.config.width_closed_m
        frac = max(0.0, min(1.0, counts / 255.0))
        return self.config.width_closed_m + (1.0 - frac) * span

    def _to_motor(self, counts: int) -> int:
        return 255 - counts if self.config.invert_position else counts

    def _from_motor(self, counts: int) -> int:
        return 255 - counts if self.config.invert_position else counts

    def _coerce_speed(self, speed: Optional[float]) -> int:
        if speed is None:
            return int(self.config.default_speed)
        return int(max(0, min(255, round(float(speed)))))

    # internal: CAN tx/rx + state machine
    def _send_motion(self, counts: int, speed: int, force_ma: int) -> None:
        """
        Issue a motion command if the gripper is _READY.

        Honours the same deadband as msg_gripper_node so spam-tight loops
        don't flood the bus.
        """
        if self._motor is None:
            raise RuntimeError(
                f"SSG48({self.config.channel}) not connected, call connect() first"
            )
        with self._state_lock:
            if self._state != _READY:
                logger.debug(
                    "SSG48(%s): drop command in state=%s",
                    self.config.channel,
                    self._state,
                )
                return
        # deadband against last commanded motor counts (post-invert)
        motor_counts = self._to_motor(counts)
        force_ma = max(0, min(self.config.max_force_ma, int(force_ma)))
        try:
            with self._tx_lock:
                self._motor.Send_gripper_data_pack(
                    motor_counts, speed, force_ma, 1, 1, 0, 0
                )
        except Exception:
            logger.exception("SSG48(%s) send failed", self.config.channel)

    def send_pack_locked(self, *args) -> None:
        """
        TX a raw Send_gripper_data_pack under the bus lock.

        External callers (e.g. the fold scripts) MUST use this instead of
        touching ``_motor`` directly, or their frame races the poll
        thread's telemetry solicitations on the shared serial handle.
        """
        if self._motor is None:
            raise RuntimeError(
                f"SSG48({self.config.channel}) not connected, call connect() first"
            )
        with self._tx_lock:
            self._motor.Send_gripper_data_pack(*args)

    def raw_position_fresh(
        self, max_age_s: float = 0.25, timeout_s: float = 1.5
    ) -> Optional[int]:
        """
        Cached firmware position counts, guaranteed freshly received.

        Waits until the poll loop has unpacked a status frame newer than
        ``max_age_s`` (so a desynced/stale cache can't masquerade as a
        live reading). Returns None if no fresh frame lands in time.
        """
        deadline = time.monotonic() + timeout_s
        while time.monotonic() < deadline:
            if time.monotonic() - self._last_rx <= max_age_s:
                return getattr(self._motor, "gripper_position", None)
            time.sleep(0.02)
        logger.warning(
            "SSG48(%s) no fresh status frame within %.1fs",
            self.config.channel,
            timeout_s,
        )
        return None

    def _set_state(self, st: str) -> None:
        with self._state_lock:
            self._state = st
            self._state_entered = time.monotonic()

    def _reinit_bus(self) -> bool:
        """
        Reopen the CAN bus in place (recovers from bus-off).

        Arm motion can storm the gripper's CAN line with errors (cable flex /
        motor EMI); the slcan controller then goes bus-off and falls silent
        until reinitialized. A fresh bus open revives it.
        """
        logger.warning(
            "SSG48(%s) telemetry silent, reinitializing CAN bus", self.config.channel
        )
        with self._tx_lock:
            try:
                if self._comm is not None:
                    self._comm.bus.shutdown()
            except Exception:
                logger.debug(
                    "SSG48(%s) stale bus shutdown failed",
                    self.config.channel,
                    exc_info=True,
                )
            try:
                _spectral = _require_spectral()
                self._comm = _spectral.CanCommunication(
                    bustype=self.config.bustype,
                    channel=self.config.channel,
                    bitrate=self.config.bitrate,
                )
                self._motor = _spectral.SpectralCAN(
                    node_id=self.config.node_id,
                    communication=self._comm,
                )
                logger.info("SSG48(%s) CAN bus reinitialized", self.config.channel)
                return True
            except Exception:
                logger.exception("SSG48(%s) CAN bus reinit FAILED", self.config.channel)
                return False

    def _poll_loop(self) -> None:
        """
        Background telemetry + state-machine loop (50 ms cadence).

        Identical structure to msg_gripper_node.update_state() so behavior
        across the ROS and standalone deployments stays in lock-step.
        """
        assert self._motor is not None and self._comm is not None
        last_reinit = 0.0
        while not self._poll_stop.is_set():
            try:
                with self._state_lock:
                    st = self._state

                # Bus-off watchdog: in READY we poll every tick, so >1s of
                # silence means the CAN controller has stopped responding.
                now = time.monotonic()
                if (
                    st == _READY
                    and self._last_rx > 0.0
                    and now - self._last_rx > 1.0
                    and now - last_reinit > 3.0
                ):
                    last_reinit = now
                    self._reinit_bus()

                if st == _ACTIVATING:
                    with self._tx_lock:
                        self._motor.Send_gripper_data_pack(
                            0,
                            self.config.default_speed,
                            self.config.default_force_ma,
                            1,
                            0,
                            0,
                            0,
                        )
                elif st == _CALIBRATING:
                    # 0-byte poll: solicits a status frame without
                    # commanding motion; sending a motion frame during
                    # calibration aborts the firmware sweep.
                    with self._tx_lock:
                        self._motor.Send_gripper_data_pack(
                            None, None, None, None, None, None, None
                        )
                elif st == _READY:
                    # Same 0-byte poll: the SSG-48 firmware does NOT
                    # auto-broadcast position in READY, so without an
                    # explicit poll the cached gripper_position goes
                    # stale immediately after a motion command. Source
                    # Robotics' own GUI sends this every tick. We do the
                    # same so get_present_width() reflects reality.
                    with self._tx_lock:
                        self._motor.Send_gripper_data_pack(
                            None, None, None, None, None, None, None
                        )

                # Drain up to a few CAN frames per tick (same as ROS node).
                for _ in range(3):
                    msg, uid = self._comm.receive_can_messages(timeout=0.001)
                    if msg is None or uid is None:
                        break
                    try:
                        self._motor.UnpackData(msg, uid)
                        self._last_rx = time.monotonic()
                    except Exception:
                        logger.debug(
                            "SSG48(%s) UnpackData failed",
                            self.config.channel,
                            exc_info=True,
                        )

                self._advance_state()
            except Exception:
                logger.exception(
                    "SSG48(%s) poll loop iteration failed", self.config.channel
                )
            time.sleep(_POLL_PERIOD_S)

    def _advance_state(self) -> None:
        with self._state_lock:
            st = self._state
            entered = self._state_entered

        activated = int(getattr(self._motor, "gripper_activated", 0) or 0) == 1
        calibrated = int(getattr(self._motor, "gripper_calibrated", 0) or 0) == 1

        if st == _ACTIVATING:
            if activated:
                if calibrated:
                    self._set_state(_READY)
                    logger.info(
                        "SSG48(%s) ready (already calibrated)", self.config.channel
                    )
                elif self.config.auto_calibrate:
                    try:
                        with self._tx_lock:
                            self._motor.Send_gripper_calibrate()
                    except Exception:
                        logger.exception(
                            "SSG48(%s) calibrate-trigger failed", self.config.channel
                        )
                    self._set_state(_CALIBRATING)
                    logger.info("SSG48(%s) calibrating...", self.config.channel)
                else:
                    logger.warning(
                        "SSG48(%s) not calibrated and auto_calibrate=False, "
                        "run scripts/gripper_reset.py to recalibrate",
                        self.config.channel,
                    )
                    self._set_state(_READY)
            elif time.monotonic() - entered > _ACTIVATE_TIMEOUT_S:
                logger.error(
                    "SSG48(%s) failed to activate in %.0fs",
                    self.config.channel,
                    _ACTIVATE_TIMEOUT_S,
                )
                self._set_state(_FAILED)
        elif st == _CALIBRATING:
            if calibrated:
                self._set_state(_READY)
                logger.info(
                    "SSG48(%s) ready (calibration complete)", self.config.channel
                )
            elif time.monotonic() - entered > _CALIBRATE_TIMEOUT_S:
                logger.error(
                    "SSG48(%s) calibration timed out at %.0fs, "
                    "check that the jaws move freely",
                    self.config.channel,
                    _CALIBRATE_TIMEOUT_S,
                )
                self._set_state(_FAILED)


class DualSSG48Gripper:
    # Pair of SSG-48 grippers, one per arm, on independent slcan buses.

    def __init__(self, left: SSG48Gripper, right: SSG48Gripper):
        self.left = left
        self.right = right

    def connect(self) -> None:
        # Open in sequence so ACM port enumeration order is deterministic
        # if both come up at once. Each gripper owns its own bus.
        self.left.connect()
        self.right.connect()

    def disconnect(self) -> None:
        for g in (self.left, self.right):
            try:
                g.disconnect()
            except Exception:
                logger.exception("error disconnecting SSG48 on %s", g.config.channel)

    def homing(self) -> None:
        for g in (self.left, self.right):
            g.homing()

    def for_arm(self, arm: str) -> SSG48Gripper:
        if arm == "left":
            return self.left
        if arm == "right":
            return self.right
        raise ValueError(f"unknown arm {arm!r}; expected 'left' or 'right'")

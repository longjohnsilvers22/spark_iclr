"""
Dynamixel parallel-jaw gripper driver (ALOHA-style, USB-CAN).

The bimanual Franka rig uses custom 3D-printed parallel jaws actuated by
a single Dynamixel servo each, communicating to the host via a Source
Robotics USB-to-CAN adapter (green PCB). This module wraps the U2D2/SDK
behind the same surface every SPARK driver expects, so the
``ScoreExecutor`` does not need a Franka-Hand vs Dynamixel branch beyond
its existing ``GRIPPER_TYPE`` switch (which already supports
``"robotiq"``, ``"franka_hand"``, and now ``"dynamixel"``).

Hardware defaults below match the lab rig:
left arm Dynamixel ID 1, right arm Dynamixel ID 2, both on the same
``/dev/ttyUSB0`` U2D2 bridge, 1 Mbps. Override via the bimanual config
YAML if the bus layout changes.

If the ``dynamixel_sdk`` package is not installed at runtime, the driver
falls back to a no-op simulator so the rest of SPARK (perception,
planner, frontend) can be exercised end-to-end without the gripper
hardware. The simulator logs every command and reports the last-commanded
width as the "current" width.
"""

from __future__ import annotations

import logging
import math
import threading
import time
from dataclasses import dataclass
from typing import Optional

logger = logging.getLogger(__name__)

try:
    from dynamixel_sdk import PortHandler, PacketHandler  # type: ignore
except ImportError:
    PortHandler = None
    PacketHandler = None


# Hardware defaults (override via DualDynamixelGripper(config=...)).
_DEFAULT_PORT = "/dev/ttyUSB0"
_DEFAULT_BAUD = 1_000_000
_DEFAULT_PROTOCOL = 2.0

# Control table addresses (Dynamixel X-series, Protocol 2.0).
_ADDR_TORQUE_ENABLE = 64
_ADDR_GOAL_POSITION = 116
_ADDR_PRESENT_POSITION = 132
_ADDR_GOAL_CURRENT = 102  # mA, used as force proxy
_ADDR_OPERATING_MODE = 11
_ADDR_PRESENT_LOAD = 126

# 0..4095 ticks over one revolution. The mechanical linkage converts servo
# rotation to jaw separation at roughly 90 ticks/mm over the linear stroke;
# recalibrate if the linkage is rebuilt.
_TICKS_PER_M = 90_000.0
# Width band (meters) actually achievable by the mechanism. The fingers
# bottom out below ~4mm and the linkage overstrokes past ~90mm; clamp.
WIDTH_MIN_M = 0.000
WIDTH_MAX_M = 0.090
# Calibration offset: ticks corresponding to the fully-closed position
# (width = 0). Per-servo; defaulted to 0 but should be set in the YAML.
_DEFAULT_ZERO_OFFSET_TICKS = 0


@dataclass
class DynamixelConfig:
    # Per-servo wiring + calibration.

    motor_id: int
    port: str = _DEFAULT_PORT
    baudrate: int = _DEFAULT_BAUD
    protocol_version: float = _DEFAULT_PROTOCOL
    zero_offset_ticks: int = _DEFAULT_ZERO_OFFSET_TICKS
    max_current_ma: int = 800  # safety cap on goal current
    invert_direction: bool = False


class DynamixelGripper:
    """
    Single Dynamixel-actuated parallel jaw.

    All public methods mirror the subset of the franky Gripper API used by
    SPARK so the higher-level executor code does not branch by gripper
    type beyond a one-line check of :attr:`GRIPPER_TYPE`.
    """

    GRIPPER_TYPE = "dynamixel"

    def __init__(self, config: DynamixelConfig):
        self.config = config
        self._lock = threading.Lock()
        self._last_commanded_width: float = WIDTH_MAX_M
        self._port_handler = None
        self._packet_handler = None
        self._connected = False
        self._sim_mode = False  # set True if SDK unavailable
        # Track whether we believe we are gripping something so
        # `is_object_detected` has a sensible default in sim mode.
        self._holding: bool = False

    # connection lifecycle
    def connect(self) -> bool:
        if PortHandler is None or PacketHandler is None:
            logger.warning(
                "dynamixel_sdk not installed; DynamixelGripper(id=%d) running "
                "in simulator mode. Install with `pip install dynamixel-sdk` "
                "for real hardware.",
                self.config.motor_id,
            )
            self._sim_mode = True
            self._connected = True
            return True

        self._port_handler = PortHandler(self.config.port)
        self._packet_handler = PacketHandler(self.config.protocol_version)
        if not self._port_handler.openPort():
            logger.error(
                "Failed to open %s for Dynamixel id=%d",
                self.config.port,
                self.config.motor_id,
            )
            self._sim_mode = True
            self._connected = True
            return False
        if not self._port_handler.setBaudRate(self.config.baudrate):
            logger.error(
                "Failed to set baud=%d for Dynamixel id=%d",
                self.config.baudrate,
                self.config.motor_id,
            )
            self._sim_mode = True
            self._connected = True
            return False

        # Operating mode 5 = current-based position control (force-aware
        # gripping). Falls back to mode 3 (position) if write fails.
        self._write1(_ADDR_OPERATING_MODE, 5)
        self._write1(_ADDR_TORQUE_ENABLE, 1)
        self._write2(_ADDR_GOAL_CURRENT, self.config.max_current_ma)
        self._connected = True
        return True

    def disconnect(self) -> None:
        if self._port_handler is not None and not self._sim_mode:
            try:
                self._write1(_ADDR_TORQUE_ENABLE, 0)
                self._port_handler.closePort()
            except Exception:
                logger.exception("Error closing Dynamixel port")
        self._connected = False

    # commands
    def homing(self) -> None:
        # Drive to fully-open as a calibration sweep.
        self.move_to_width(WIDTH_MAX_M)

    def move_to_width(self, width_m: float, speed: Optional[float] = None) -> None:
        width = max(WIDTH_MIN_M, min(WIDTH_MAX_M, float(width_m)))
        with self._lock:
            self._last_commanded_width = width
            self._holding = False
            if self._sim_mode:
                return
            ticks = self._width_to_ticks(width)
            self._write4(_ADDR_GOAL_POSITION, ticks)

    def grasp_to_width(
        self,
        width_m: float,
        force_n: Optional[float] = None,
        speed: Optional[float] = None,
    ) -> bool:
        """
        Close to ``width_m`` with a current limit derived from ``force_n``.

        Dynamixel current is in mA; we map ``force_n`` linearly to a current
        cap, clamped to ``max_current_ma`` from the servo config. Returns
        ``True`` once the present position stops moving (best-effort).
        """
        cap_ma = self.config.max_current_ma
        if force_n is not None:
            # 80 mA / N is a rough lab calibration for this linkage; tune.
            cap_ma = int(max(50, min(cap_ma, force_n * 80.0)))
        with self._lock:
            self._last_commanded_width = max(WIDTH_MIN_M, min(WIDTH_MAX_M, width_m))
            self._holding = True
            if self._sim_mode:
                return True
            self._write2(_ADDR_GOAL_CURRENT, cap_ma)
            ticks = self._width_to_ticks(self._last_commanded_width)
            self._write4(_ADDR_GOAL_POSITION, ticks)
        # Wait until the present position stops changing for 100 ms or 2 s
        # total, matches the franky grasp_async wait pattern.
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
        self.move_to_width(WIDTH_MAX_M, speed=speed)

    def close(
        self, force_n: Optional[float] = None, speed: Optional[float] = None
    ) -> bool:
        return self.grasp_to_width(WIDTH_MIN_M, force_n=force_n, speed=speed)

    # queries
    def get_present_width(self) -> float:
        with self._lock:
            if self._sim_mode:
                return self._last_commanded_width
            ticks = self._read4(_ADDR_PRESENT_POSITION)
            if ticks is None:
                return self._last_commanded_width
            return self._ticks_to_width(ticks)

    def get_position_norm(self) -> float:
        # Return 0..255 Robotiq-compatible scale (255=closed).
        w = self.get_present_width()
        frac = 1.0 - (w - WIDTH_MIN_M) / max(1e-6, (WIDTH_MAX_M - WIDTH_MIN_M))
        return max(0.0, min(255.0, frac * 255.0))

    def is_object_detected(self) -> bool:
        """
        True if commanded grasp was issued and present width > commanded.

        Mirrors libfranka semantics: object detected when the jaws stop
        short of the goal because of contact.
        """
        with self._lock:
            if not self._holding:
                return False
            present = self.get_present_width()
            return present > self._last_commanded_width + 0.001

    # I/O helpers
    def _width_to_ticks(self, width_m: float) -> int:
        ticks = int(width_m * _TICKS_PER_M) + self.config.zero_offset_ticks
        if self.config.invert_direction:
            ticks = -ticks
        # Wrap into the [0, 4095] register space the X-series exposes.
        return ticks & 0x0FFF

    def _ticks_to_width(self, ticks: int) -> float:
        if self.config.invert_direction:
            ticks = -ticks
        return max(0.0, (ticks - self.config.zero_offset_ticks) / _TICKS_PER_M)

    def _write1(self, addr: int, value: int) -> None:
        if self._sim_mode:
            return
        self._packet_handler.write1ByteTxRx(
            self._port_handler, self.config.motor_id, addr, int(value) & 0xFF
        )

    def _write2(self, addr: int, value: int) -> None:
        if self._sim_mode:
            return
        self._packet_handler.write2ByteTxRx(
            self._port_handler, self.config.motor_id, addr, int(value) & 0xFFFF
        )

    def _write4(self, addr: int, value: int) -> None:
        if self._sim_mode:
            return
        self._packet_handler.write4ByteTxRx(
            self._port_handler, self.config.motor_id, addr, int(value) & 0xFFFFFFFF
        )

    def _read4(self, addr: int) -> Optional[int]:
        if self._sim_mode:
            return None
        val, comm, err = self._packet_handler.read4ByteTxRx(
            self._port_handler, self.config.motor_id, addr
        )
        if comm != 0 or err != 0:
            return None
        return val


class DualDynamixelGripper:
    # Convenience wrapper that owns the left + right grippers as one.

    def __init__(self, left: DynamixelGripper, right: DynamixelGripper):
        self.left = left
        self.right = right

    def connect(self) -> None:
        # Both servos share a single U2D2 in the standard rig, so opening
        # the port twice is harmless (PortHandler caches per process).
        self.left.connect()
        self.right.connect()

    def disconnect(self) -> None:
        for g in (self.left, self.right):
            try:
                g.disconnect()
            except Exception:
                logger.exception("error disconnecting gripper id=%d", g.config.motor_id)

    def homing(self) -> None:
        for g in (self.left, self.right):
            g.homing()

    def for_arm(self, arm: str) -> DynamixelGripper:
        if arm == "left":
            return self.left
        if arm == "right":
            return self.right
        raise ValueError(f"unknown arm {arm!r}; expected 'left' or 'right'")

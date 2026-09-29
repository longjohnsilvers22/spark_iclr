"""
Franka Hand gripper implementations backed by the franky library.

Concrete gripper methods for the franky motion-generator driver, split out
of franka_driver so the driver class body stays focused on arm motion. The
mixin satisfies the gripper abstractmethods declared on FrankaDriverBase and
overrides set_gripper_position with a debounced async variant for streaming
analog triggers. It expects the host class to provide self._gripper (a
franky Gripper client), self._check_connected, and the shared self._cfg_gripper
helper from FrankaDriverBase.
"""

from __future__ import annotations

import logging
import time
from typing import Optional

import numpy as np

from spark_real.robots.franka.franka_base import (
    GRIPPER_DEFAULT_SPEED,
    GRIPPER_DEFAULT_FORCE,
    GRIPPER_OPEN_WIDTH,
)

logger = logging.getLogger(__name__)


class FrankyGripperMixin:
    """
    Franka Hand gripper logic via franky's Gripper client.

    Provides the gripper abstractmethods FrankaDriverBase declares plus a
    debounced set_gripper_position override. Must precede FrankaDriverBase in
    the MRO so its concrete methods satisfy the base abstractmethods and the
    override stays effective.
    """

    # Gripper (FrankaDriverBase abstract implementations)

    def _gripper_open(self, speed: float):
        self._check_connected()
        self._gripper.open(float(speed))

    def _gripper_grasp(
        self,
        width: float,
        speed: float,
        force: float,
        epsilon_inner: float,
        epsilon_outer: float,
    ) -> bool:
        self._check_connected()
        try:
            self._gripper.stop()
        except Exception:
            pass
        for attempt in range(2):
            try:
                return bool(
                    self._gripper.grasp(
                        float(width),
                        float(speed),
                        float(force),
                        float(epsilon_inner),
                        float(epsilon_outer),
                    )
                )
            except Exception as exc:
                if attempt == 0:
                    logger.warning("grasp() raised (%s); stop+retry", exc)
                    try:
                        self._gripper.stop()
                    except Exception:
                        pass
                    time.sleep(0.25)
                else:
                    raise
        return False

    def _gripper_width(self) -> float:
        self._check_connected()
        return float(self._gripper.width)

    def _gripper_is_grasped(self) -> bool:
        self._check_connected()
        return bool(self._gripper.is_grasped)

    def _gripper_max_width(self) -> float:
        try:
            return float(self._gripper.max_width)
        except Exception:
            return self._cfg_gripper("open_width", GRIPPER_OPEN_WIDTH)

    def _gripper_homing(self):
        self._check_connected()
        ok = self._gripper.homing()
        time.sleep(0.5)
        if not ok:
            logger.warning("Franka Hand homing did not report success")

    # Franky-specific gripper overrides
    # These override FrankaDriverBase's shared implementations to use
    # franky-specific async methods for debounced set_gripper_position.

    def set_gripper_position(
        self,
        position: float,
        speed: Optional[float] = None,
        force: Optional[float] = None,
    ):
        # Set gripper width with debounce for streaming analog triggers.
        pos = float(np.clip(position, 0.0, 1.0))
        max_w = self._gripper_max_width()
        width = max_w * (1.0 - pos)
        spd = (
            speed
            if speed is not None
            else self._cfg_gripper("speed", GRIPPER_DEFAULT_SPEED)
        )
        self._check_connected()

        now = time.monotonic()
        last_w = getattr(self, "_last_grip_target_m", None)
        last_t = getattr(self, "_last_grip_target_t", 0.0)
        if last_w is not None and abs(width - last_w) < 0.003 and now - last_t < 0.150:
            return
        self._last_grip_target_m = width
        self._last_grip_target_t = now

        if pos < 0.95:
            self._gripper.move_async(float(width), float(spd))
        else:
            frc = (
                force
                if force is not None
                else self._cfg_gripper("force", GRIPPER_DEFAULT_FORCE)
            )
            self._gripper.grasp_async(
                float(width),
                float(spd),
                float(frc),
                float(max_w),
                float(max_w),
            )

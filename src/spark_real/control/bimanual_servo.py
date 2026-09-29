"""
Per-arm Cartesian servoing with the inter-arm CBF applied as a multiplier.

Wraps two :class:`spark_real.control.cartesian_servo.CartesianServo`
instances and intercepts ``servo_to()`` so the commanded velocity is
scaled down by the inter-arm barrier returned by
:class:`BimanualSafeRobot`.

The per-arm servo loop runs at the same 30 Hz cadence as the single-arm
Franka path (franky Ruckig limitation). For parallel bimanual motion the
two ``servo_to()`` calls run from independent threads; they share no
franky state because each FrankaDriver session is per-arm.
"""

from __future__ import annotations

import logging
import threading
from typing import Optional

import numpy as np

from spark_real.control.cartesian_servo import CartesianServo
from spark_real.control.bimanual_safe_robot import BimanualSafeRobot

logger = logging.getLogger(__name__)


class BimanualCartesianServo:
    """
    Two ``CartesianServo`` instances + inter-arm velocity gating.
    """

    def __init__(self, safe: BimanualSafeRobot, rate_hz: float = 30.0):
        self.safe = safe
        self.rate_hz = float(rate_hz)
        # CartesianServo also caps non-UR drivers at 30 Hz internally.
        self._left = CartesianServo(safe.left, rate_hz=self.rate_hz)
        self._right = CartesianServo(safe.right, rate_hz=self.rate_hz)

    @property
    def left(self) -> CartesianServo:
        return self._left

    @property
    def right(self) -> CartesianServo:
        return self._right

    def for_arm(self, arm: str) -> CartesianServo:
        if arm == "left":
            return self._left
        if arm == "right":
            return self._right
        raise ValueError(f"unknown arm {arm!r}")

    def servo_to(
        self, arm: str, target_pose: np.ndarray, timeout_s: float = 5.0
    ) -> bool:
        """
        Servo ``arm`` to ``target_pose`` (xyz + rotvec) with inter-arm gating.

        Wraps :meth:`CartesianServo.move_to_pose` and clamps the per-arm
        servo's ``max_vel_linear`` according to the inter-arm CBF scale
        on entry. If the scale collapses to zero (the other arm has
        moved into the keep-out zone) the call returns False without
        commanding any motion.
        """
        target_pose = np.asarray(target_pose, dtype=float).ravel()
        if not self.safe.is_motion_safe(arm, target_pose[:3]):
            return False

        # Apply inter-arm velocity scale as a one-shot reduction of the
        # per-arm servo's max linear velocity. Restore after the call.
        servo = self.for_arm(arm)
        original_max = servo.max_vel_linear
        scale = self.safe.velocity_scale_for(arm, target_pose[:3])
        if scale <= 0.0:
            return False
        try:
            servo.max_vel_linear = original_max * scale
            return servo.move_to_pose(
                target_pose.tolist(), velocity=original_max * scale, timeout=timeout_s
            )
        finally:
            servo.max_vel_linear = original_max

    def servo_to_parallel(
        self, target_left: np.ndarray, target_right: np.ndarray, timeout_s: float = 5.0
    ) -> dict:
        """
        Drive both arms to their targets in parallel; returns per-arm success.
        """
        results: dict = {}

        def _go(arm: str, tgt) -> None:
            results[arm] = self.servo_to(arm, tgt, timeout_s=timeout_s)

        threads = [
            threading.Thread(
                target=_go, args=("left", target_left), name="left-servo", daemon=True
            ),
            threading.Thread(
                target=_go,
                args=("right", target_right),
                name="right-servo",
                daemon=True,
            ),
        ]
        for t in threads:
            t.start()
        for t in threads:
            t.join()
        return results

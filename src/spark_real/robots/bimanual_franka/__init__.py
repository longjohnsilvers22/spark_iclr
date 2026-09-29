"""
Bimanual Franka package (Panda left + FR3 right).

Composes two :class:`spark_real.robots.franka.FrankaDriver` instances,
plus two Dynamixel parallel grippers driven over USB-CAN, into a single
driver that satisfies the UR10eDriver shim while exposing per-arm
sub-drivers via :attr:`BimanualFrankaDriver.left` and ``.right``.

This is a *separate family* from the single-arm Franka driver: shared
behaviour (e.g. franky control-mode invariants, libfranka reflex
auto-recovery, joint limits) is reused by composition rather than by
inheritance, so changes to single-arm semantics do not silently propagate
to bimanual without an explicit decision.
"""

from .bimanual_franka_driver import (
    ARMS,
    BimanualFrankaDriver,
    BimanualObservation,
    DEFAULT_LEFT_IP,
    DEFAULT_RIGHT_IP,
    make_dual_gripper,
)
from .dynamixel_gripper import DynamixelGripper, DualDynamixelGripper
from .ssg48_gripper import SSG48Gripper, DualSSG48Gripper, SSG48Config

__all__ = [
    "ARMS",
    "BimanualFrankaDriver",
    "BimanualObservation",
    "DEFAULT_LEFT_IP",
    "DEFAULT_RIGHT_IP",
    "DynamixelGripper",
    "DualDynamixelGripper",
    "SSG48Gripper",
    "DualSSG48Gripper",
    "SSG48Config",
    "make_dual_gripper",
]

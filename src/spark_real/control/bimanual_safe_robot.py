"""
Safety wrapper for the bimanual Franka rig.

Composes two per-arm :class:`SafeRobot` instances, each with its own
workspace cuboid and joint-limit guard, and adds a cross-arm distance
constraint enforced before every motion command.

Per-arm safety (workspace box, reach sphere, joint limits, force barrier)
is delegated to the single-arm ``SafeRobot``; the bimanual-specific concern
(TCP-to-TCP separation) is handled here.

Wraps a :class:`BimanualFrankaDriver` and exposes:

* :attr:`left` / :attr:`right`: the per-arm SafeRobot views.
* :meth:`is_motion_safe(arm, target_pose)`: checks the inter-arm
  distance constraint *before* dispatching the motion.
* :meth:`get_inter_arm_distance()`: current TCP-to-TCP distance.

Higher-level callers (the bimanual score executor, the BT primitives)
talk to ``BimanualSafeRobot.left`` / ``.right`` for motion and to the
parent object for inter-arm gates.
"""

from __future__ import annotations

import logging
import math
from dataclasses import dataclass, field
from typing import Optional, Tuple

import numpy as np

from spark_real.control.safe_robot import SafeRobot, SafetyConfig

logger = logging.getLogger(__name__)


def _rpy_to_R(rpy: Tuple[float, float, float]) -> np.ndarray:
    """
    Roll-pitch-yaw (intrinsic XYZ, radians) to 3x3 rotation.
    """
    r, p, y = (float(rpy[0]), float(rpy[1]), float(rpy[2]))
    cr, sr = math.cos(r), math.sin(r)
    cp, sp = math.cos(p), math.sin(p)
    cy, sy = math.cos(y), math.sin(y)
    Rx = np.array([[1, 0, 0], [0, cr, -sr], [0, sr, cr]], dtype=float)
    Ry = np.array([[cp, 0, sp], [0, 1, 0], [-sp, 0, cp]], dtype=float)
    Rz = np.array([[cy, -sy, 0], [sy, cy, 0], [0, 0, 1]], dtype=float)
    return Rz @ Ry @ Rx


def _se3_from_xyz_rpy(
    xyz: Tuple[float, float, float], rpy: Tuple[float, float, float]
) -> np.ndarray:
    T = np.eye(4)
    T[:3, :3] = _rpy_to_R(rpy)
    T[:3, 3] = np.asarray(xyz, dtype=float)
    return T


@dataclass
class BaseFrames:
    """
    Per-arm base SE3 in the bimanual world frame.

    The world frame is the rig table center (matches the YAML convention
    ``base_left.xyz`` / ``base_right.xyz``). Both transforms are 4x4
    homogeneous; ``T_left_world`` and ``T_right_world`` map a point given
    in that arm's own base frame to a point in world.

    Default identity matrices are intentionally wrong-but-loud: any
    consumer that forgets to wire in real base offsets will see arms
    overlapping at the origin in world coordinates rather than getting
    silently-wrong distances back. Construct via :func:`from_yaml_block`.
    """

    T_left_world: np.ndarray = field(default_factory=lambda: np.eye(4, dtype=float))
    T_right_world: np.ndarray = field(default_factory=lambda: np.eye(4, dtype=float))

    def T_world_left(self) -> np.ndarray:
        return np.linalg.inv(self.T_left_world)

    def T_world_right(self) -> np.ndarray:
        return np.linalg.inv(self.T_right_world)

    def T_arm_world(self, arm: str) -> np.ndarray:
        if arm == "left":
            return self.T_left_world
        if arm == "right":
            return self.T_right_world
        raise ValueError(f"unknown arm {arm!r}")

    def arm_to_world(self, arm: str, xyz_arm: np.ndarray) -> np.ndarray:
        T = self.T_arm_world(arm)
        p = np.asarray(xyz_arm, dtype=float).reshape(3)
        return (T @ np.array([p[0], p[1], p[2], 1.0]))[:3]

    def world_to_arm(self, arm: str, xyz_world: np.ndarray) -> np.ndarray:
        T = self.T_world_left() if arm == "left" else self.T_world_right()
        p = np.asarray(xyz_world, dtype=float).reshape(3)
        return (T @ np.array([p[0], p[1], p[2], 1.0]))[:3]

    @classmethod
    def from_yaml_block(cls, robot_cfg: dict) -> "BaseFrames":
        """
        Build from the ``robot:`` block of a bimanual YAML.

        Expects ``base_left`` / ``base_right`` sub-blocks of the form
        ``{xyz: [x, y, z], rpy: [r, p, y]}``. Missing fields default to
        zero translation / zero rotation; missing blocks raise so a
        copy-paste YAML without base offsets fails loudly rather than
        silently colliding at the origin.
        """

        def _block(name: str) -> dict:
            blk = robot_cfg.get(name)
            if not blk:
                raise ValueError(
                    f"bimanual config missing required {name!r} block "
                    "(xyz + rpy); see configs/bimanual_franka_default.yaml"
                )
            return blk

        bl = _block("base_left")
        br = _block("base_right")
        T_left = _se3_from_xyz_rpy(bl.get("xyz", [0, 0, 0]), bl.get("rpy", [0, 0, 0]))
        T_right = _se3_from_xyz_rpy(br.get("xyz", [0, 0, 0]), br.get("rpy", [0, 0, 0]))
        return cls(T_left_world=T_left, T_right_world=T_right)


@dataclass
class InterArmConfig:
    """
    Cross-arm distance constraint.

    The bimanual CBF is a soft-then-hard barrier on TCP-to-TCP distance:

    * Above ``dist_soft_m`` motions proceed at full commanded speed.
    * Between ``dist_soft_m`` and ``dist_hard_m`` the maximum commanded
      velocity toward the other arm is scaled by
      ``eta_kinematic * (d - dist_hard) / (dist_soft - dist_hard)``.
    * Below ``dist_hard_m`` motions toward the other arm are rejected
      outright and the controller switches to hold-position.
    """

    dist_hard_m: float = 0.12
    dist_soft_m: float = 0.20
    eta_kinematic: float = 8.0


class BimanualSafeRobot:
    """
    Two per-arm SafeRobot views + an inter-arm distance barrier.

    ``bases`` carries the per-arm base offsets that turn each per-arm
    libfranka pose (returned in that arm's OWN base frame) into a world
    point so cross-arm distances are computed in a single shared frame.
    Without this the inter-arm gate compares poses in two different
    frames and is wrong by the inter-arm base offset (~0.64 m on the
    default rig, ~0.88 m on the ANON-LAB rig).
    """

    def __init__(
        self,
        driver,
        left_safety: SafetyConfig,
        right_safety: SafetyConfig,
        inter_arm: Optional[InterArmConfig] = None,
        bases: Optional[BaseFrames] = None,
    ):
        self._driver = driver
        self.inter_arm = inter_arm or InterArmConfig()
        if bases is None:
            logger.warning(
                "BimanualSafeRobot constructed without base frames; "
                "inter-arm distances will treat both arm bases as the "
                "origin (i.e. arm-base frames = world). Pass a BaseFrames "
                "built from the YAML to get correct cross-arm distances."
            )
            bases = BaseFrames()
        self.bases = bases

        # The per-arm SafeRobot wraps the per-arm sub-driver: the single-arm
        # SafeRobot does not know about the bimanual driver, so it is handed
        # the underlying FrankaDriver directly.
        self.left = SafeRobot(driver.for_arm("left"), config=left_safety)
        self.right = SafeRobot(driver.for_arm("right"), config=right_safety)

    # arm view accessors
    @property
    def driver(self):
        return self._driver

    def for_arm(self, arm: str) -> SafeRobot:
        if arm == "left":
            return self.left
        if arm == "right":
            return self.right
        raise ValueError(f"unknown arm {arm!r}; expected 'left' or 'right'")

    def other(self, arm: str) -> SafeRobot:
        return self.for_arm("right" if arm == "left" else "left")

    # inter-arm CBF
    def get_tcp_positions(self) -> Tuple[np.ndarray, np.ndarray]:
        """
        (left_tcp_xyz, right_tcp_xyz) in WORLD frame.

        ``driver.get_tcp_pose(arm)`` returns each arm's flange in that
        arm's OWN base frame. Composing with the per-arm base SE3 lands
        both points in a single shared frame, so the cross-arm distance
        below is geometrically meaningful.
        """
        lp_base = np.asarray(self._driver.get_tcp_pose("left"))[:3]
        rp_base = np.asarray(self._driver.get_tcp_pose("right"))[:3]
        lp = self.bases.arm_to_world("left", lp_base)
        rp = self.bases.arm_to_world("right", rp_base)
        return lp, rp

    def get_inter_arm_distance(self) -> float:
        lp, rp = self.get_tcp_positions()
        return float(np.linalg.norm(lp - rp))

    def is_motion_safe(self, arm: str, target_xyz: np.ndarray) -> bool:
        """
        Reject motions that would punch into the other arm's keep-out radius.

        ``target_xyz`` is interpreted in ``arm``'s OWN base frame because
        it is passed straight through from per-arm callers (CartesianServo
        targets are arm-base). It is promoted to world here so the
        cross-arm comparison sees a single frame.

        Cheap check: compare proposed target to the *current* other-arm
        TCP. This pre-gate plus the velocity ramp on servo entry
        (bimanual_servo.py) is the whole inter-arm protection; there is
        no inter-arm QP filter.
        """
        target_xyz = np.asarray(target_xyz, dtype=float).ravel()[:3]
        target_world = self.bases.arm_to_world(arm, target_xyz)
        other_world = self.get_tcp_positions()[0 if arm == "right" else 1]
        d = float(np.linalg.norm(target_world - other_world))
        if d < self.inter_arm.dist_hard_m:
            logger.warning(
                "BimanualSafeRobot: rejecting %s arm motion to %s "
                "(arm-base) / %s (world), TCP-to-TCP distance would be "
                "%.3f m (hard floor %.3f m)",
                arm,
                target_xyz.tolist(),
                target_world.tolist(),
                d,
                self.inter_arm.dist_hard_m,
            )
            return False
        return True

    def velocity_scale_for(
        self, arm: str, target_xyz: Optional[np.ndarray] = None
    ) -> float:
        """
        Return a multiplier in [0, 1] for commanded velocity toward the other arm.

        ``target_xyz`` (when given) is interpreted in ``arm``'s OWN base
        frame, matching ``is_motion_safe``. Distance is computed in
        world frame via the per-arm base offsets.
        """
        if target_xyz is None:
            d = self.get_inter_arm_distance()
        else:
            t = np.asarray(target_xyz, dtype=float).ravel()[:3]
            t_world = self.bases.arm_to_world(arm, t)
            other = self.get_tcp_positions()[0 if arm == "right" else 1]
            d = float(np.linalg.norm(t_world - other))
        if d >= self.inter_arm.dist_soft_m:
            return 1.0
        if d <= self.inter_arm.dist_hard_m:
            return 0.0
        # Linear ramp.
        span = max(1e-6, self.inter_arm.dist_soft_m - self.inter_arm.dist_hard_m)
        return float((d - self.inter_arm.dist_hard_m) / span)

    # convenience pass-throughs
    def go_home(self, arm: Optional[str] = None) -> None:
        """
        Send the arms to their per-arm HOME_CONFIG.

        Sequenced left -> right by default. Even for the mirror-symmetric
        home configs in ``bimanual_franka_default.yaml`` (elbows fold
        outward), sequential is the conservative choice: per-rig home
        configs may diverge from the default (e.g. ANON-LAB FR3 with a
        different mounting yaw), and any planning bug that lands one
        arm in the other's swept volume costs only homing time, not a
        collision.
        """
        if arm is not None:
            self._driver.go_home(arm=arm)
            return
        self._driver.go_home(arm="left")
        self._driver.go_home(arm="right")

    def stop(self, arm: Optional[str] = None) -> None:
        self._driver.stop(arm=arm)

    def send_velocity(
        self,
        linear,
        angular=None,
        arm: Optional[str] = None,
        duration: float = 0.1,
        acceleration: float = 0.5,
    ):
        """
        Per-arm Cartesian velocity teleop command, CBF-filtered.

        Routes through the per-arm :class:`SafeRobot.send_velocity` so the
        QP-based barrier on joint limits, workspace bounds, and singularity
        proximity applies on every tick. ``linear``/``angular`` are 3-vecs;
        they're concatenated into the 6-vec the underlying API expects.
        The bimanual API includes an ``arm`` kwarg (required) so a single
        endpoint can drive either side without consumers having to walk
        through ``.left`` / ``.right`` explicitly.
        """
        if arm is None:
            raise ValueError(
                "BimanualSafeRobot.send_velocity requires arm='left' or 'right'"
            )
        lin = np.asarray(linear, dtype=float).ravel()
        if lin.size < 3:
            lin = np.pad(lin, (0, 3 - lin.size))
        lin = lin[:3]
        if angular is None:
            ang = np.zeros(3, dtype=float)
        else:
            ang = np.asarray(angular, dtype=float).ravel()
            if ang.size < 3:
                ang = np.pad(ang, (0, 3 - ang.size))
            ang = ang[:3]
        six = np.concatenate([lin, ang])
        self.for_arm(arm).send_velocity(
            six, acceleration=acceleration, time_duration=duration
        )

    def open_gripper(self, arm: Optional[str] = None, speed=None, force=None) -> None:
        """
        SSG-48 open. Delegates to the bimanual driver's gripper layer
        (not the per-arm SafeRobot, its gripper passthrough hits the
        non-existent Franka Hand on the ANON-LAB rig).
        """
        self._driver.open_gripper(arm=arm, speed=speed, force=force)

    def close_gripper(self, arm: Optional[str] = None, speed=None, force=None) -> bool:
        return self._driver.close_gripper(arm=arm, speed=speed, force=force)

    def set_gripper_position(
        self, position: float, arm: Optional[str] = None, speed=None, force=None
    ) -> None:
        self._driver.set_gripper_position(position, arm=arm, speed=speed, force=force)

    # Read-only state pass-throughs. SafeRobot's safety envelope is about
    # MOTION COMMANDS; observation calls are just data. Forward straight
    # to the bimanual driver so HTTP /api/bimanual/state and friends work
    # without a per-arm round-trip. Explicit, so the wrapper API stays visible
    # and no motion command is ever forwarded.
    def get_observation(self, arm=None):
        return self._driver.get_observation(arm=arm)

    def get_tcp_pose(self, arm=None):
        return self._driver.get_tcp_pose(arm=arm)

    def get_joint_positions(self, arm=None):
        return self._driver.get_joint_positions(arm=arm)

    def get_robot_mode(self, arm=None):
        return self._driver.get_robot_mode(arm=arm)

    def is_steady(self, arm=None) -> bool:
        return self._driver.is_steady(arm=arm)

    @property
    def has_errors(self) -> bool:
        return self._driver.has_errors

    def recover_from_errors(self, arm=None) -> bool:
        return self._driver.recover_from_errors(arm=arm)

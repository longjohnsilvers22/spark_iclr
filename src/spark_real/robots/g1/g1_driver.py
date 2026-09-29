"""
Unitree G1 humanoid bimanual driver for SPARK.

Thin wrapper around `unitree_sdk2_python` that exposes a left/right separated
primitive surface compatible with SPARK's BT executor.  Every method that
touches the robot maps onto a verified upstream call: see the
`# Upstream:` comments.  We do NOT invent any SDK behaviour.

Authoritative upstream:
  git@github.com:unitreerobotics/unitree_sdk2_python.git
  Commit pinned: d72a9515 (2026-02-26 / merge #152)
  Cloned to:    ~/spark/external/unitree_upstream/

References within that repo (paths relative to repo root):
  example/g1/high_level/g1_arm7_sdk_dds_example.py     : dual-arm joint path
  example/g1/high_level/g1_arm5_sdk_dds_example.py     : 23-DOF variant
  example/g1/low_level/g1_low_level_example.py         : full-body low-level
  unitree_sdk2py/g1/loco/g1_loco_client.py             : locomotion + balance
  unitree_sdk2py/g1/arm/g1_arm_action_client.py        : canned arm motions
  unitree_sdk2py/idl/unitree_hg/msg/dds_/_LowCmd_.py   : LowCmd_ IDL
  unitree_sdk2py/idl/unitree_hg/msg/dds_/_HandCmd_.py  : HandCmd_ IDL (Dex3)

Hand control reference (separate Unitree project, same wire format):
  github.com/unitreerobotics/xr_teleoperate
    teleop/robot_control/robot_hand_unitree.py
  Topics: rt/dex3/left/cmd, rt/dex3/right/cmd
          rt/dex3/left/state, rt/dex3/right/state    (also rt/lf/dex3/...)
"""

from __future__ import annotations

import logging
import time
from dataclasses import dataclass
from typing import Optional, Sequence, Tuple

import numpy as np

logger = logging.getLogger(__name__)

# Optional upstream import.
# The Unitree SDK ships its own cyclonedds wheel.  Make the import optional so
# this file is at least loadable on a workstation that does not have the SDK
# (e.g. for unit tests or for editing on a laptop).
try:
    # Upstream: unitree_sdk2py/core/channel.py
    from unitree_sdk2py.core.channel import (
        ChannelFactoryInitialize,
        ChannelPublisher,
        ChannelSubscriber,
    )

    # Upstream: unitree_sdk2py/idl/unitree_hg/msg/dds_/__init__.py
    from unitree_sdk2py.idl.unitree_hg.msg.dds_ import (
        LowCmd_,
        LowState_,
        HandCmd_,
        HandState_,
    )

    # Upstream: unitree_sdk2py/idl/default.py
    from unitree_sdk2py.idl.default import (
        unitree_hg_msg_dds__LowCmd_,
        unitree_hg_msg_dds__HandCmd_,
    )

    # Upstream: unitree_sdk2py/utils/crc.py
    from unitree_sdk2py.utils.crc import CRC

    # Upstream: unitree_sdk2py/utils/thread.py
    from unitree_sdk2py.utils.thread import RecurrentThread

    # Upstream: unitree_sdk2py/g1/loco/g1_loco_client.py
    from unitree_sdk2py.g1.loco.g1_loco_client import LocoClient

    # Upstream: unitree_sdk2py/g1/arm/g1_arm_action_client.py
    from unitree_sdk2py.g1.arm.g1_arm_action_client import G1ArmActionClient

    HAS_UNITREE = True
except ImportError as _e:
    HAS_UNITREE = False
    logger.warning(
        "unitree_sdk2_python not importable (%s); driver will run "
        "in stub-only mode.",
        _e,
    )


# Joint indices, pulled verbatim from upstream, see file/line below.
# Upstream: example/g1/high_level/g1_arm7_sdk_dds_example.py:21-60
# Mirrored at:  example/g1/low_level/g1_low_level_example.py:35-67
# These are the only authoritative indices.  Do NOT renumber.
class G1JointIndex:
    # Left leg
    LeftHipPitch = 0
    LeftHipRoll = 1
    LeftHipYaw = 2
    LeftKnee = 3
    LeftAnklePitch = 4  # = LeftAnkleB
    LeftAnkleRoll = 5  # = LeftAnkleA
    # Right leg
    RightHipPitch = 6
    RightHipRoll = 7
    RightHipYaw = 8
    RightKnee = 9
    RightAnklePitch = 10  # = RightAnkleB
    RightAnkleRoll = 11  # = RightAnkleA
    # Waist (only WaistYaw is valid for the locked-waist 23/29-DOF G1)
    WaistYaw = 12
    WaistRoll = 13  # INVALID on 23/29-dof waist-locked builds
    WaistPitch = 14  # INVALID on 23/29-dof waist-locked builds
    # Left arm (7-DOF on G1 29-DOF; WristPitch/WristYaw absent on 23-DOF)
    LeftShoulderPitch = 15
    LeftShoulderRoll = 16
    LeftShoulderYaw = 17
    LeftElbow = 18
    LeftWristRoll = 19
    LeftWristPitch = 20  # INVALID on 23-DOF
    LeftWristYaw = 21  # INVALID on 23-DOF
    # Right arm
    RightShoulderPitch = 22
    RightShoulderRoll = 23
    RightShoulderYaw = 24
    RightElbow = 25
    RightWristRoll = 26
    RightWristPitch = 27  # INVALID on 23-DOF
    RightWristYaw = 28  # INVALID on 23-DOF
    # Upstream comment "NOTE: Weight": this is the arm_sdk enable channel.
    # Set motor_cmd[kNotUsedJoint].q = 1 to hand the arms over to the SDK,
    # set it back to 0 (or ramp it down) to return them to the on-robot
    # balance controller.
    # Upstream: example/g1/high_level/g1_arm7_sdk_dds_example.py:62, 135
    kNotUsedJoint = 29


# Convenience tuples (7-DOF builds)
LEFT_ARM_JOINTS_7DOF: Tuple[int, ...] = (
    G1JointIndex.LeftShoulderPitch,
    G1JointIndex.LeftShoulderRoll,
    G1JointIndex.LeftShoulderYaw,
    G1JointIndex.LeftElbow,
    G1JointIndex.LeftWristRoll,
    G1JointIndex.LeftWristPitch,
    G1JointIndex.LeftWristYaw,
)
RIGHT_ARM_JOINTS_7DOF: Tuple[int, ...] = (
    G1JointIndex.RightShoulderPitch,
    G1JointIndex.RightShoulderRoll,
    G1JointIndex.RightShoulderYaw,
    G1JointIndex.RightElbow,
    G1JointIndex.RightWristRoll,
    G1JointIndex.RightWristPitch,
    G1JointIndex.RightWristYaw,
)
LEFT_ARM_JOINTS_5DOF: Tuple[int, ...] = LEFT_ARM_JOINTS_7DOF[:5]
RIGHT_ARM_JOINTS_5DOF: Tuple[int, ...] = RIGHT_ARM_JOINTS_7DOF[:5]


# DDS topic names, copied from upstream / Unitree docs.  Do not rename.
TOPIC_ARM_SDK = "rt/arm_sdk"  # Upstream: g1_arm7_sdk_dds_example.py:107
TOPIC_LOW_CMD = "rt/lowcmd"  # Upstream: g1_low_level_example.py:101
TOPIC_LOW_STATE = "rt/lowstate"  # Upstream: g1_arm7_sdk_dds_example.py:111
TOPIC_DEX3_LEFT_CMD = "rt/dex3/left/cmd"  # see file header references
TOPIC_DEX3_RIGHT_CMD = "rt/dex3/right/cmd"
TOPIC_DEX3_LEFT_STATE = "rt/dex3/left/state"
TOPIC_DEX3_RIGHT_STATE = "rt/dex3/right/state"


@dataclass
class DualArmState:
    """
    Snapshot of both arms.

    `*_q` are joint angles in radians, indexed by the upstream joint order
    (LeftShoulderPitch...LeftWristYaw, then the same for right).  EE pose
    fields are populated by an external FK; the SDK does not expose them.
    """

    left_q: np.ndarray
    right_q: np.ndarray
    left_dq: np.ndarray
    right_dq: np.ndarray
    waist_q: float
    timestamp: float
    left_ee_pose: Optional[np.ndarray] = None  # filled by FK (e.g. pinocchio)
    right_ee_pose: Optional[np.ndarray] = None


class G1Driver:
    """
    Bimanual driver shim for the Unitree G1.

    Public API mirrors UR10eDriver where possible but adds explicit left/right
    separation that the BT executor needs (see `score_executor.py`).

    All low-level writes use the `rt/arm_sdk` channel via a `RecurrentThread`,
    exactly as in the upstream `g1_arm7_sdk_dds_example.py`.  We do NOT touch
    the leg/foot motors (those stay under the on-robot balance controller,
    `LocoClient` `BalanceStand`).  See README.md for the full rationale.
    """

    # Defaults taken from upstream arm7 example: kp=60, kd=1.5
    # Upstream: example/g1/high_level/g1_arm7_sdk_dds_example.py:71-72
    DEFAULT_KP = 60.0
    DEFAULT_KD = 1.5
    CONTROL_DT = 0.02  # 50 Hz, upstream sets control_dt_ = 0.02

    def __init__(
        self,
        net_iface: str = "eth0",
        dof: int = 29,
        enable_hands: bool = False,
        domain_id: int = 0,
    ):
        """
        Args:
            net_iface: Network interface connected to the G1 (e.g. "eth0").
                See upstream `ChannelFactoryInitialize(domain_id, iface)`.
            dof: 23 or 29.  Selects whether wrist pitch/yaw are commanded.
            enable_hands: If True, also init Dex3 left/right hand DDS channels.
            domain_id: DDS domain id (default 0).
        """
        if not HAS_UNITREE:
            raise RuntimeError(
                "unitree_sdk2_python not installed.  See "
                "~/spark/src/spark_real/docs/g1.md "
                "for install instructions."
            )
        if dof not in (23, 29):
            raise ValueError("G1 dof must be 23 or 29")

        self.net_iface = net_iface
        self.dof = dof
        self.enable_hands = enable_hands
        self.domain_id = domain_id

        # Arm channel state
        self._low_cmd: LowCmd_ = unitree_hg_msg_dds__LowCmd_()
        self._low_state: Optional[LowState_] = None
        self._first_state_seen = False
        self._crc = CRC()
        self._arm_pub: Optional[ChannelPublisher] = None
        self._state_sub: Optional[ChannelSubscriber] = None
        self._cmd_thread: Optional[RecurrentThread] = None

        # Hand channel state (only used if enable_hands=True)
        self._left_hand_pub: Optional[ChannelPublisher] = None
        self._right_hand_pub: Optional[ChannelPublisher] = None
        self._left_hand_sub: Optional[ChannelSubscriber] = None
        self._right_hand_sub: Optional[ChannelSubscriber] = None
        self._left_hand_msg: Optional[HandCmd_] = None
        self._right_hand_msg: Optional[HandCmd_] = None
        self._left_hand_state: Optional[HandState_] = None
        self._right_hand_state: Optional[HandState_] = None

        # Service clients (high-level RPC)
        self._loco: Optional[LocoClient] = None
        self._arm_action: Optional[G1ArmActionClient] = None

        # Targets that the recurrent-thread writer applies (rad).  These are
        # the *desired* joint angles; the writer interpolates 1:1 each tick.
        self._target_q = np.zeros(30, dtype=np.float64)
        # Enable flag for arm_sdk (0..1, ramped externally if desired)
        self._arm_sdk_enable = 0.0

        # Arm sets for the chosen dof
        self.left_arm_joints = (
            LEFT_ARM_JOINTS_7DOF if dof == 29 else LEFT_ARM_JOINTS_5DOF
        )
        self.right_arm_joints = (
            RIGHT_ARM_JOINTS_7DOF if dof == 29 else RIGHT_ARM_JOINTS_5DOF
        )

    # life
    def connect(self) -> None:
        """
        Open the DDS channels and start the 50 Hz writer thread.

        Mirrors upstream `Custom.Init() + Custom.Start()` from
        g1_arm7_sdk_dds_example.py:106-118.
        """
        # Upstream: example/g1/high_level/g1_arm7_sdk_dds_example.py:178-181
        ChannelFactoryInitialize(self.domain_id, self.net_iface)

        # Upstream: example/g1/high_level/g1_arm7_sdk_dds_example.py:107-109
        self._arm_pub = ChannelPublisher(TOPIC_ARM_SDK, LowCmd_)
        self._arm_pub.Init()

        # Upstream: example/g1/high_level/g1_arm7_sdk_dds_example.py:111-113
        self._state_sub = ChannelSubscriber(TOPIC_LOW_STATE, LowState_)
        self._state_sub.Init(self._on_low_state, 10)

        # High-level service clients
        # Upstream: unitree_sdk2py/g1/loco/g1_loco_client.py
        self._loco = LocoClient()
        self._loco.SetTimeout(5.0)
        self._loco.Init()
        # Upstream: unitree_sdk2py/g1/arm/g1_arm_action_client.py
        self._arm_action = G1ArmActionClient()
        self._arm_action.SetTimeout(5.0)
        self._arm_action.Init()

        if self.enable_hands:
            self._init_hands()

        # Wait for first state, just like upstream's busy loop.
        # Upstream: g1_arm7_sdk_dds_example.py:117-119
        t0 = time.time()
        while not self._first_state_seen:
            if time.time() - t0 > 10:
                raise RuntimeError(
                    "No LowState_ received in 10s; check NIC / "
                    "domain id / robot power."
                )
            time.sleep(0.05)

        # Seed targets to the current state so the first tick doesn't jump.
        for j in (*self.left_arm_joints, *self.right_arm_joints, G1JointIndex.WaistYaw):
            self._target_q[j] = self._low_state.motor_state[j].q

        # Start the 50 Hz writer.
        # Upstream: g1_arm7_sdk_dds_example.py:114-118
        self._cmd_thread = RecurrentThread(
            interval=self.CONTROL_DT, target=self._tick, name="g1_arm_write"
        )
        self._cmd_thread.Start()
        logger.info(
            "G1 connected: dof=%d hands=%s iface=%s",
            self.dof,
            self.enable_hands,
            self.net_iface,
        )

    def disconnect(self) -> None:
        """
        Hand the arms back to the balance controller and tear down.

        Replicates the upstream "Stage 4" ramp where `kNotUsedJoint.q` is
        decreased from 1 to 0.  We just drop to 0 here; for a smooth release
        callers should ramp via `set_arm_sdk_enable()`.
        Upstream: g1_arm7_sdk_dds_example.py:155-159
        """
        try:
            self._arm_sdk_enable = 0.0
            time.sleep(0.2)  # allow one or two ticks to publish enable=0
        finally:
            self._cmd_thread = None
            self._arm_pub = None
            self._state_sub = None
            logger.info("G1 disconnected.")

    def __enter__(self):
        self.connect()
        return self

    def __exit__(self, *exc):
        self.disconnect()
        return False

    # arm_sdk callback
    def _on_low_state(self, msg: LowState_) -> None:
        # Upstream: g1_arm7_sdk_dds_example.py:121-126
        self._low_state = msg
        if not self._first_state_seen:
            self._first_state_seen = True

    def _tick(self) -> None:
        """
        Writer callback: emits one LowCmd_ at the 50 Hz cadence.

        We do not do trajectory ramping here; we simply publish the current
        `_target_q` with the default PD gains.  This is the same pattern as
        upstream's Stage-2 block in g1_arm7_sdk_dds_example.py:140-148, except
        the target is supplied by the caller via `move_*_arm_to_joint` rather
        than hard-coded.
        """
        if self._low_state is None or self._arm_pub is None:
            return
        cmd = self._low_cmd
        # Hand the arms to the SDK (or release them).
        # Upstream: g1_arm7_sdk_dds_example.py:135
        cmd.motor_cmd[G1JointIndex.kNotUsedJoint].q = self._arm_sdk_enable

        for j in (*self.left_arm_joints, *self.right_arm_joints, G1JointIndex.WaistYaw):
            cmd.motor_cmd[j].tau = 0.0
            cmd.motor_cmd[j].q = float(self._target_q[j])
            cmd.motor_cmd[j].dq = 0.0
            cmd.motor_cmd[j].kp = self.DEFAULT_KP
            cmd.motor_cmd[j].kd = self.DEFAULT_KD

        cmd.crc = self._crc.Crc(cmd)
        self._arm_pub.Write(cmd)

    # public arm primitives
    def set_arm_sdk_enable(self, value: float) -> None:
        """
        Hand the arms to (or release them from) the SDK.

        1.0 = arms under SDK control, 0.0 = back to balance controller.
        Upstream: g1_arm7_sdk_dds_example.py:135 ("Enable arm_sdk").
        """
        self._arm_sdk_enable = float(np.clip(value, 0.0, 1.0))

    def move_left_arm_to_joint(
        self,
        q: Sequence[float],
        blocking: bool = True,
        tol_rad: float = 0.02,
        timeout_s: float = 8.0,
    ) -> bool:
        """
        Command the left arm joints to ``q`` (length = len(left_arm_joints)).

        Implementation: writes ``_target_q`` at the indices in
        ``self.left_arm_joints`` and lets the 50 Hz writer publish them under
        the standard kp/kd.  This is the dual-arm pattern from
        `g1_arm7_sdk_dds_example.py` decomposed into a per-arm call.
        """
        return self._move_arm(self.left_arm_joints, q, blocking, tol_rad, timeout_s)

    def move_right_arm_to_joint(
        self,
        q: Sequence[float],
        blocking: bool = True,
        tol_rad: float = 0.02,
        timeout_s: float = 8.0,
    ) -> bool:
        return self._move_arm(self.right_arm_joints, q, blocking, tol_rad, timeout_s)

    def _move_arm(self, indices, q, blocking, tol_rad, timeout_s) -> bool:
        if len(q) != len(indices):
            raise ValueError(f"expected {len(indices)} joints, got {len(q)}")
        # Need arm_sdk enabled to actually move.  Upstream Stage 1->2 ramp.
        if self._arm_sdk_enable < 0.5:
            self._arm_sdk_enable = 1.0
        for j, qi in zip(indices, q):
            self._target_q[j] = float(qi)
        if not blocking:
            return True
        t0 = time.time()
        while time.time() - t0 < timeout_s:
            cur = np.array([self._low_state.motor_state[j].q for j in indices])
            if np.max(np.abs(cur - np.array(q))) < tol_rad:
                return True
            time.sleep(0.02)
        logger.warning(
            "move_arm timeout; residual=%.3f rad",
            float(np.max(np.abs(cur - np.array(q)))),
        )
        return False

    def move_dual_arm_to_pose(
        self,
        left_pose: np.ndarray,
        right_pose: np.ndarray,
        ik_solver,
    ) -> bool:
        """
        Synchronised dual-arm Cartesian move.

        The Unitree SDK does NOT ship a Cartesian endpoint for arbitrary
        targets: only canned `G1ArmActionClient` motions are end-effector
        aware.  To do Cartesian targets we need an external IK solver.

        Args:
            left_pose, right_pose: 4x4 SE(3) homogeneous transforms in the
                G1 body frame.
            ik_solver: callable ``(left_pose, right_pose, q_init) -> (q_l, q_r)``.
                Implementation suggestions in README.md (pinocchio dual-chain
                or pyroki).

        # TODO: not in upstream, needs IK (pinocchio / pyroki).  See
        # README.md "Bimanual primitive contract" and `Robotics-Ark/
        # ark_unitree_g1/tests/test_demo/robot_arm_ik.py` for a pinocchio
        # reference (single-arm; needs dual-chain extension).
        """
        if ik_solver is None:
            raise NotImplementedError(
                "move_dual_arm_to_pose requires an IK solver, see "
                "docs/g1.md.  Pass `ik_solver=YourIK()`."
            )
        q_init_left = np.array(
            [self._low_state.motor_state[j].q for j in self.left_arm_joints]
        )
        q_init_right = np.array(
            [self._low_state.motor_state[j].q for j in self.right_arm_joints]
        )
        q_l, q_r = ik_solver(
            left_pose, right_pose, np.concatenate([q_init_left, q_init_right])
        )
        ok_l = self._move_arm(
            self.left_arm_joints, q_l, blocking=False, tol_rad=0.02, timeout_s=0
        )
        ok_r = self._move_arm(
            self.right_arm_joints, q_r, blocking=False, tol_rad=0.02, timeout_s=0
        )
        # block on both
        t0 = time.time()
        while time.time() - t0 < 8.0:
            cur_l = np.array(
                [self._low_state.motor_state[j].q for j in self.left_arm_joints]
            )
            cur_r = np.array(
                [self._low_state.motor_state[j].q for j in self.right_arm_joints]
            )
            if (
                np.max(np.abs(cur_l - q_l)) < 0.02
                and np.max(np.abs(cur_r - q_r)) < 0.02
            ):
                return True
            time.sleep(0.02)
        return False

    def bimanual_grasp(
        self,
        object_left_pose: np.ndarray,
        object_right_pose: np.ndarray,
        ik_solver,
        approach_offset_z: float = 0.10,
    ) -> bool:
        """
        Two-arm simultaneous grasp coordination primitive.

        This is the entry point the BT executor calls when an action node is
        flagged ``arms: [left, right]``.  The
        controller handles both Cartesian targets at once; no centroid
        merging happens here, since each arm is given its own pose.
        """
        # 1. approach: lift each target by approach_offset_z
        appr_l = object_left_pose.copy()
        appr_l[2, 3] += approach_offset_z
        appr_r = object_right_pose.copy()
        appr_r[2, 3] += approach_offset_z
        if not self.move_dual_arm_to_pose(appr_l, appr_r, ik_solver):
            return False
        # 2. open both hands
        self.set_left_gripper(0.0)
        self.set_right_gripper(0.0)
        # 3. descend to grasp pose
        if not self.move_dual_arm_to_pose(
            object_left_pose, object_right_pose, ik_solver
        ):
            return False
        # 4. close both hands
        self.set_left_gripper(1.0)
        self.set_right_gripper(1.0)
        time.sleep(0.6)
        return True

    # hand control
    def _init_hands(self) -> None:
        """
        Open the four Dex3 DDS channels.

        Upstream reference (HandCmd message structure):
          unitree_sdk2py/idl/unitree_hg/msg/dds_/_HandCmd_.py
          unitree_sdk2py/idl/default.py:231-232  (7-motor HandCmd_ default)
        Topic names confirmed in:
          github.com/unitreerobotics/xr_teleoperate
            teleop/robot_control/robot_hand_unitree.py
        """
        self._left_hand_pub = ChannelPublisher(TOPIC_DEX3_LEFT_CMD, HandCmd_)
        self._left_hand_pub.Init()
        self._right_hand_pub = ChannelPublisher(TOPIC_DEX3_RIGHT_CMD, HandCmd_)
        self._right_hand_pub.Init()
        self._left_hand_sub = ChannelSubscriber(TOPIC_DEX3_LEFT_STATE, HandState_)
        self._left_hand_sub.Init(self._on_left_hand_state, 10)
        self._right_hand_sub = ChannelSubscriber(TOPIC_DEX3_RIGHT_STATE, HandState_)
        self._right_hand_sub.Init(self._on_right_hand_state, 10)
        self._left_hand_msg = unitree_hg_msg_dds__HandCmd_()
        self._right_hand_msg = unitree_hg_msg_dds__HandCmd_()

    def _on_left_hand_state(self, msg: HandState_) -> None:
        self._left_hand_state = msg

    def _on_right_hand_state(self, msg: HandState_) -> None:
        self._right_hand_state = msg

    def set_left_gripper(self, closure: float) -> None:
        """
        Command the left Dex3 hand closure (0.0=open, 1.0=closed).

        # TODO: per-finger joint targets and the RIS-mode byte must come from
        # the Dex3 manual.  Right now we send a uniform q on all 7 motors,
        # which is fine for power-grasp closure but does not pinch.
        # Upstream reference for the exact byte layout:
        #   github.com/unitreerobotics/xr_teleoperate
        #     teleop/robot_control/robot_hand_unitree.py (`_RIS_Mode`)
        """
        if not self.enable_hands or self._left_hand_pub is None:
            logger.warning("hands not enabled, set_left_gripper ignored")
            return
        self._set_hand(self._left_hand_msg, self._left_hand_pub, closure)

    def set_right_gripper(self, closure: float) -> None:
        if not self.enable_hands or self._right_hand_pub is None:
            logger.warning("hands not enabled, set_right_gripper ignored")
            return
        self._set_hand(self._right_hand_msg, self._right_hand_pub, closure)

    def _set_hand(self, msg: HandCmd_, pub: ChannelPublisher, closure: float) -> None:
        c = float(np.clip(closure, 0.0, 1.0))
        # Dex3-1 has 7 motors; mapping 0..1 closure -> uniform joint angle
        # is an approximation good enough for envelope grasps.
        # TODO: verify on upstream, closed-angle per finger differs.
        q_closed = 1.2  # rad, conservative power-grasp angle
        for i in range(7):
            msg.motor_cmd[i].q = c * q_closed
            msg.motor_cmd[i].dq = 0.0
            msg.motor_cmd[i].tau = 0.0
            msg.motor_cmd[i].kp = 1.5
            msg.motor_cmd[i].kd = 0.2
        pub.Write(msg)

    # state read
    def read_dual_state(self) -> DualArmState:
        # Snapshot both arms.  EE poses left empty: compute with FK.
        if self._low_state is None:
            raise RuntimeError("not connected / no state yet")
        ms = self._low_state.motor_state
        left_q = np.array([ms[j].q for j in self.left_arm_joints])
        right_q = np.array([ms[j].q for j in self.right_arm_joints])
        left_dq = np.array([ms[j].dq for j in self.left_arm_joints])
        right_dq = np.array([ms[j].dq for j in self.right_arm_joints])
        return DualArmState(
            left_q=left_q,
            right_q=right_q,
            left_dq=left_dq,
            right_dq=right_dq,
            waist_q=float(ms[G1JointIndex.WaistYaw].q),
            timestamp=time.time(),
        )

    # safety / e-stop
    def damp(self) -> None:
        """
        Switch to damping mode (no position hold, gravity comp only).

        Upstream: g1_loco_client.py: `LocoClient.Damp()` -> `SetFsmId(1)`.
        """
        if self._loco is not None:
            self._loco.Damp()
        self._arm_sdk_enable = 0.0

    def balance_stand(self) -> None:
        """
        Put the robot into balanced-stand (legs).

        REQUIRED before running arm_sdk on a free-standing robot.  See
        upstream issue #108: low-level full-body and balance-stand are
        mutually exclusive; the supported combo is BalanceStand + arm_sdk.
        Upstream: g1_loco_client.py:113-114; LocoClient.BalanceStand().
        """
        if self._loco is None:
            return
        self._loco.BalanceStand(balance_mode=0)

"""
Franka FR3 driver via franky (motion generators).

Thin translation layer delegating every Franka-specific call to the
franky library (TimSchneider42/franky, v1.1.3). Inherits shared
constants, gripper logic, and observation synthesis from FrankaDriverBase.
"""

from __future__ import annotations

import logging
import math
import os
import threading
import time
from typing import Optional, Sequence

import numpy as np

from spark_real.robots.franka.franka_base import (
    FrankaDriverBase,
    FLANGE_T_TCP,
    GRIPPER_DEFAULT_SPEED,
    GRIPPER_DEFAULT_FORCE,
    GRIPPER_OPEN_WIDTH,
)
from spark_real.robots.franka.franka_gripper_mixin import FrankyGripperMixin
from spark_real.utils.rotations import (
    axis_angle_to_quat as _axis_angle_to_quat,
    quat_to_axis_angle as _quat_to_axis_angle,
)

logger = logging.getLogger(__name__)

try:
    from franky import (
        Affine,
        CartesianMotion,
        CartesianStopMotion,
        CartesianVelocityMotion,
        CartesianVelocityStopMotion,
        Duration,
        Frame,
        Gripper,
        JointMotion,
        JointStopMotion,
        JointVelocityMotion,
        JointVelocityStopMotion,
        JointWaypoint,
        JointWaypointMotion,
        RealtimeConfig,
        ReferenceType,
        RelativeDynamicsFactor,
        Robot,
        RobotVelocity,
        Twist,
    )

    HAS_FRANKY = True
except ImportError:
    HAS_FRANKY = False
    logger.warning("franky not installed. Install with: pip install franky-control")


class FrankaDriver(FrankyGripperMixin, FrankaDriverBase):
    """
    Direct control interface for Franka FR3 / Panda via franky.

    Public API mirrors UR10eDriver. Joint vectors are length-7.
    Cartesian poses use [x, y, z, rx, ry, rz] (axis-angle) on the boundary;
    internally franky uses Affine + quaternion.
    """

    SUPPORTS_VELOCITY_STREAMING = True
    DEFAULT_ACCELERATION = 0.5
    DEFAULT_JOINT_ACC = 1.4

    _MAX_LINEAR_VEL_FR3 = 1.7
    _MAX_JOINT_VEL_FR3 = 2.62

    DEFAULT_COLLISION_TORQUE_THRESHOLD = 20.0
    DEFAULT_COLLISION_FORCE_THRESHOLD = 30.0

    # Stationarity gate: max |dq| (rad/s) the arm may carry into a new joint
    # motion. Below this, a JointMotion starts without tripping the
    # velocity/acceleration discontinuity reflex.
    STATIONARY_DQ_TOL = 0.01
    STATIONARY_GATE_TIMEOUT = 0.5

    def __init__(
        self,
        robot_ip: str,
        frequency: float = 1000.0,
        gripper: Optional[dict] = None,
        collision: Optional[dict] = None,
    ):
        if not HAS_FRANKY:
            raise RuntimeError(
                "franky not installed. pip install franky-control "
                "(libfranka must also be available)"
            )
        super().__init__(robot_ip, frequency)
        self._robot: Optional["Robot"] = None
        self._gripper: Optional["Gripper"] = None
        self._max_linear_vel = self._MAX_LINEAR_VEL_FR3
        self._max_joint_vel = self._MAX_JOINT_VEL_FR3

        # Config-driven gripper + collision settings from the RobotProfile.
        # When absent, gripper methods fall back to the module-level
        # GRIPPER_* constants and connect() falls back to the scalar
        # collision threshold, so behavior is unchanged without a profile.
        self._gripper_cfg = dict(gripper) if gripper else {}
        self._collision_cfg = dict(collision) if collision else {}

        # Realtime joint-servo session state
        self._servo_lock = threading.Lock()
        self._servo_active = False
        self._servo_last_target: Optional[np.ndarray] = None

        # Realtime Cartesian-velocity session state
        self._velocity_lock = threading.Lock()
        self._velocity_active = False
        self._velocity_last_target: Optional[np.ndarray] = None

    # Connection lifecycle

    def connect(self):
        # Open FCI connection to the Franka controller.
        logger.info("Connecting to Franka at %s...", self.robot_ip)
        try:
            self._robot = Robot(self.robot_ip)
        except Exception as e:
            if "realtime" in str(e).lower():
                logger.warning("Non-RT kernel (%s). Using RealtimeConfig.Ignore.", e)
                self._robot = Robot(
                    self.robot_ip, realtime_config=RealtimeConfig.Ignore
                )
            else:
                raise
        # Franka Hand client only when this arm wears one. A gripper type of
        # none/ssg48/dynamixel means the jaws are external (e.g. SSG-48 on CAN);
        # franky.Gripper(ip) against a no-Hand arm fails with a misleading
        # NetworkException ("Connection to FCI refused ... enable FCI in Desk").
        gtype = str(self._gripper_cfg.get("type", "franka_hand")).lower()
        if gtype in ("franka_hand", "franka"):
            self._gripper = Gripper(self.robot_ip)
        else:
            self._gripper = None
            logger.info(
                "No Franka Hand on %s (gripper type %r); skipping "
                "franky Gripper client.",
                self.robot_ip,
                gtype,
            )
        try:
            self._max_linear_vel = float(self._robot.translation_velocity_limit.max)
            self._max_joint_vel = float(
                np.max(np.asarray(self._robot.joint_velocity_limit.max))
            )
        except Exception:
            logger.debug("Could not read franky dynamics limits; using defaults.")
        self._connected = True
        self._apply_collision_behavior()
        logger.info("Connected. has_errors=%s", self._robot.has_errors)

    def _apply_collision_behavior(self):
        """
        Apply RobotProfile collision thresholds when present, else scalar 100/100.
        """
        cfg = self._collision_cfg
        torque_lower = cfg.get("joint_torque_nom_lower")
        torque_upper = cfg.get("joint_torque_nom_upper")
        force_lower = cfg.get("cartesian_force_nom_lower")
        force_upper = cfg.get("cartesian_force_nom_upper")
        torque_acc_lower = cfg.get("joint_torque_acc_lower")
        torque_acc_upper = cfg.get("joint_torque_acc_upper")
        force_acc_lower = cfg.get("cartesian_force_acc_lower")
        force_acc_upper = cfg.get("cartesian_force_acc_upper")
        nominal = (torque_lower, torque_upper, force_lower, force_upper)
        accel = (torque_acc_lower, torque_acc_upper, force_acc_lower, force_acc_upper)
        try:
            if all(v is not None for v in nominal):
                self.set_collision_behavior(
                    lower_torque_threshold=torque_lower,
                    upper_torque_threshold=torque_upper,
                    lower_force_threshold=force_lower,
                    upper_force_threshold=force_upper,
                    lower_torque_threshold_acceleration=torque_acc_lower,
                    upper_torque_threshold_acceleration=torque_acc_upper,
                    lower_force_threshold_acceleration=force_acc_lower,
                    upper_force_threshold_acceleration=force_acc_upper,
                )
                logger.info(
                    "Applied profile collision thresholds (acc=%s)",
                    all(v is not None for v in accel),
                )
            else:
                self.set_collision_behavior(
                    torque_threshold=100.0, force_threshold=100.0
                )
        except Exception as exc:
            logger.warning("Could not apply collision behavior: %s", exc)

    def disconnect(self):
        # Tear down the FCI connection.
        with self._servo_lock:
            self._servo_active = False
            self._servo_last_target = None
        with self._velocity_lock:
            self._velocity_active = False
            self._velocity_last_target = None
        if self._robot is not None:
            try:
                self._robot.stop()
            except Exception:
                pass
        self._robot = None
        self._gripper = None
        self._connected = False
        logger.info("Disconnected")

    def _check_connected(self):
        if not self._connected:
            raise RuntimeError("Not connected to robot. Call connect() first.")

    # State queries

    def get_joint_positions(self) -> np.ndarray:
        # Return current joint positions (radians, length 7).
        self._check_connected()
        return np.asarray(self._robot.current_joint_state.position, dtype=float)

    def get_joint_velocities(self) -> np.ndarray:
        # Return current joint velocities (rad/s, length 7).
        self._check_connected()
        return np.asarray(self._robot.current_joint_state.velocity, dtype=float)

    def get_tcp_pose(self) -> np.ndarray:
        # Return current TCP pose [x, y, z, rx, ry, rz] (axis-angle).
        self._check_connected()
        pose = self._robot.current_pose.end_effector_pose
        t = np.asarray(pose.translation, dtype=float)
        q = np.asarray(pose.quaternion, dtype=float)
        rxyz = _quat_to_axis_angle(q)
        return np.concatenate([t, rxyz])

    def get_tcp_force(self) -> np.ndarray:
        # Return external wrench [fx, fy, fz, tx, ty, tz] at the TCP.
        self._check_connected()
        try:
            return np.asarray(self._robot.state.O_F_ext_hat_K, dtype=float)
        except AttributeError:
            return np.zeros(6)

    def get_robot_mode(self) -> int:
        # Return robot mode integer.
        self._check_connected()
        try:
            return int(self._robot.state.robot_mode)
        except Exception:
            return -1

    def is_steady(self) -> bool:
        # Return True when the robot has finished its current motion.
        self._check_connected()
        return bool(self._robot.poll_motion())

    @property
    def has_errors(self) -> bool:
        self._check_connected()
        return bool(self._robot.has_errors)

    def recover_from_errors(self) -> bool:
        self._check_connected()
        return bool(self._robot.recover_from_errors())

    # Motion commands

    def move_to_joint_config(
        self,
        q: Sequence[float],
        velocity: Optional[float] = None,
        acceleration: Optional[float] = None,
        asynchronous: bool = False,
    ):
        # Move to a 7-DOF joint configuration.
        self._check_connected()
        q_arr = list(q)
        if len(q_arr) != 7:
            raise ValueError(f"Franka expects 7 joints, got {len(q_arr)}.")

        self._preempt_velocity_if_active()
        self._preempt_servo_if_active()

        # Auto-recover if in error state
        try:
            if self._robot.has_errors:
                self._robot.recover_from_errors()
        except Exception as exc:
            logger.warning("auto recovery raised: %s", exc)

        # Stationarity gate: only command the JointMotion once residual velocity
        # from the previous motion has bled off, to avoid the velocity/
        # acceleration discontinuity reflex.
        self.gate_joint_stationary()

        try:
            if self._robot.has_errors:
                self._robot.recover_from_errors()
        except Exception:
            pass

        vel_rdf = self._velocity_to_rdf(
            velocity, self._max_joint_vel, self.DEFAULT_JOINT_VEL
        )
        rdf = RelativeDynamicsFactor(vel_rdf, vel_rdf * 0.5, vel_rdf * 0.25)
        motion = JointMotion(q_arr, relative_dynamics_factor=rdf)

        try:
            self._dispatch_motion(motion, asynchronous=asynchronous)
        except Exception as exc:
            msg = str(exc).lower()
            if any(
                k in msg
                for k in (
                    "motion finished commanded",
                    "joint_motion_generator_velocity_discontinuity",
                    "joint_motion_generator_acceleration_discontinuity",
                )
            ):
                logger.info("joint move residual velocity (%s); recovering", exc)
                try:
                    if self._robot.has_errors:
                        self._robot.recover_from_errors()
                except Exception:
                    pass
                self._wait_joints_stationary(max_wait=2.0, vel_tol=2e-3)
                return
            raise

    def gate_joint_stationary(
        self,
        vel_tol: Optional[float] = None,
        timeout: Optional[float] = None,
    ) -> float:
        """
        Fast pre-motion gate: ensure the arm is still before a joint move.

        Reads current joint velocities once. If max |dq| is already below
        vel_tol the call returns immediately (no sleep), the common case.
        Otherwise it polls at ~100 Hz until still or timeout elapses.

        Returns the wait duration in seconds (0.0 on the fast path).
        """
        tol = self.STATIONARY_DQ_TOL if vel_tol is None else vel_tol
        t_max = self.STATIONARY_GATE_TIMEOUT if timeout is None else timeout

        def _dq_max() -> Optional[float]:
            try:
                v = np.asarray(self._robot.current_joint_velocities, dtype=float)
            except Exception:
                try:
                    v = np.asarray(
                        self._robot.current_joint_state.velocity, dtype=float
                    )
                except Exception:
                    return None
            return float(np.max(np.abs(v)))

        dqm = _dq_max()
        if dqm is None or dqm < tol:
            return 0.0  # already still (or unreadable): single read, no sleep

        t0 = time.monotonic()
        deadline = t0 + t_max
        while time.monotonic() < deadline:
            time.sleep(0.01)
            dqm = _dq_max()
            if dqm is None or dqm < tol:
                break
        waited = time.monotonic() - t0
        logger.info(
            "joint stationarity gate waited %.0f ms (dq_max=%.4f rad/s, "
            "tol=%.3f) before joint move",
            waited * 1e3,
            dqm if dqm is not None else float("nan"),
            tol,
        )
        return waited

    def _wait_joints_stationary(self, max_wait: float = 1.0, vel_tol: float = 5e-3):
        # Block until joint velocities settle below vel_tol (slow fallback path).
        t_end = time.time() + max_wait
        while time.time() < t_end:
            try:
                v = np.asarray(self._robot.current_joint_state.velocity, dtype=float)
            except Exception:
                return
            if float(np.max(np.abs(v))) < vel_tol:
                return
            time.sleep(0.01)

    def _check_workspace(self, pose: Sequence[float]):
        # Verify a Cartesian target is inside workspace bounds.
        x, y, z = pose[0], pose[1], pose[2]
        if not (-0.9 <= x <= 0.9 and -0.9 <= y <= 0.9 and -0.05 <= z <= 1.2):
            raise ValueError(
                f"Target ({x:.3f}, {y:.3f}, {z:.3f}) outside workspace bounds."
            )

    def move_linear(
        self,
        pose: Sequence[float],
        velocity: Optional[float] = None,
        acceleration: Optional[float] = None,
        asynchronous: bool = False,
    ):
        # Move linearly to a Cartesian pose [x, y, z, rx, ry, rz].
        self._check_connected()
        self._check_workspace(pose)
        self._preempt_velocity_if_active()
        self._preempt_servo_if_active()

        x, y, z, rx, ry, rz = (float(v) for v in pose[:6])
        quat = _axis_angle_to_quat(rx, ry, rz)
        target = Affine([x, y, z], quat.tolist())

        vel_rdf = self._velocity_to_rdf(
            velocity, self._max_linear_vel, self.DEFAULT_VELOCITY
        )
        vel_rdf = min(vel_rdf, 0.35)
        rdf = RelativeDynamicsFactor(vel_rdf, vel_rdf * 0.5, vel_rdf * 0.25)

        try:
            self._robot.move(CartesianStopMotion(), asynchronous=False)
        except Exception:
            pass

        motion = CartesianMotion(target, relative_dynamics_factor=rdf)
        self._dispatch_motion(motion, asynchronous=asynchronous)

        # Wait for joints to physically stop after synchronous move
        if not asynchronous:
            t0 = time.monotonic()
            while time.monotonic() - t0 < 1.0:
                try:
                    dq = np.asarray(
                        self._robot.current_joint_state.velocity, dtype=float
                    )
                except Exception:
                    break
                if float(np.max(np.abs(dq))) < 0.005:
                    break
                time.sleep(0.02)

    def move_linear_relative(
        self,
        delta: Sequence[float],
        velocity: Optional[float] = None,
        acceleration: Optional[float] = None,
    ):
        # Move by [dx, dy, dz, drx, dry, drz] relative to the current TCP.
        self._check_connected()
        self._preempt_velocity_if_active()
        self._preempt_servo_if_active()
        dx, dy, dz, drx, dry, drz = (float(v) for v in delta[:6])
        quat = _axis_angle_to_quat(drx, dry, drz)
        delta_affine = Affine([dx, dy, dz], quat.tolist())
        vel_rdf = self._velocity_to_rdf(
            velocity, self._max_linear_vel, self.DEFAULT_VELOCITY
        )
        rdf = RelativeDynamicsFactor(vel_rdf, vel_rdf * 0.5, vel_rdf * 0.25)
        motion = CartesianMotion(
            delta_affine,
            reference_type=ReferenceType.Relative,
            relative_dynamics_factor=rdf,
        )
        self._dispatch_motion(motion)

    # Joint servo

    def servo_joint(
        self,
        q: Sequence[float],
        velocity: float = 0.5,
        acceleration: float = 0.5,
        dt: float = 0.002,
        lookahead_time: float = 0.1,
        gain: int = 300,
    ):
        """
        Real-time joint servo: stream a new target to the running motion.

        Each call issues an asynchronous JointWaypointMotion; franky's Ruckig
        planner preempts and re-plans on the fly, producing smooth motion at
        the 1 kHz control loop while accepting Python-side updates at 200-500 Hz.
        """
        self._check_connected()
        q_arr = list(q)
        if len(q_arr) != 7:
            raise ValueError(f"Franka expects 7 joints, got {len(q_arr)}.")
        self._preempt_velocity_if_active()

        vel_rdf = self._velocity_to_rdf(
            velocity, self._max_joint_vel, self.DEFAULT_JOINT_VEL
        )
        rdf = RelativeDynamicsFactor(vel_rdf, vel_rdf * 0.5, vel_rdf * 0.25)
        target = np.asarray(q_arr, dtype=float)

        with self._servo_lock:
            waypoint = JointWaypoint(q_arr)
            motion = JointWaypointMotion(
                [waypoint],
                relative_dynamics_factor=rdf,
                return_when_finished=False,
            )
            self._dispatch_motion(motion, asynchronous=True)
            self._servo_active = True
            self._servo_last_target = target

    def servo_stop(self):
        # Tear down an active servo streaming session.
        self._check_connected()
        with self._servo_lock:
            try:
                self._robot.move(JointStopMotion(), asynchronous=False)
            except Exception as exc:
                logger.warning("servo_stop: %s", exc)
            finally:
                self._servo_active = False
                self._servo_last_target = None

    def _preempt_servo_if_active(self):
        if self._servo_active:
            logger.debug("Preempting active servo session")
            self.servo_stop()

    # Cartesian velocity streaming

    def send_velocity(
        self,
        linear: Sequence[float],
        angular: Optional[Sequence[float]] = None,
        acceleration: float = 0.5,
        duration: float = 0.0,
    ):
        # Stream a Cartesian velocity to the robot.
        self._check_connected()

        linear_arr = np.asarray(linear, dtype=float).ravel()
        if angular is None:
            if linear_arr.size != 6:
                raise ValueError(
                    f"send_velocity: expected 6-vec, got {linear_arr.size}."
                )
            twist6 = linear_arr
        else:
            angular_arr = np.asarray(angular, dtype=float).ravel()
            if linear_arr.size != 3 or angular_arr.size != 3:
                raise ValueError("send_velocity: expected linear=3, angular=3.")
            twist6 = np.concatenate([linear_arr, angular_arr])

        self._preempt_servo_if_active()

        # Auto-recover from reflex and suppress this tick
        if self._robot.has_errors:
            try:
                self._robot.recover_from_errors()
                logger.debug("send_velocity: auto-recovered from reflex")
            except Exception as _err:
                logger.warning("send_velocity: recover failed: %s", _err)
            self._velocity_active = False
            self._velocity_last_target = None
            return

        # Joint-limit-aware Cartesian velocity shaping
        shaped_twist = twist6
        try:
            qdot = self._safe_joint_velocity(twist6)
            J = np.asarray(
                self._robot.model.zero_jacobian(Frame.EndEffector, self._robot.state),
                dtype=float,
            )
            shaped_twist = J @ qdot
        except Exception as _exc:
            logger.warning("send_velocity: twist shaping unavailable (%s)", _exc)

        twist = Twist(shaped_twist[:3].copy(), shaped_twist[3:].copy())
        robot_vel = RobotVelocity(twist)
        rdf = 0.15

        motion = CartesianVelocityMotion(
            robot_vel,
            relative_dynamics_factor=rdf,
        )

        with self._velocity_lock:
            try:
                self._dispatch_motion(motion, asynchronous=True)
            except Exception as exc:
                self._velocity_active = False
                self._velocity_last_target = None
                msg = str(exc).lower()
                if any(
                    k in msg
                    for k in (
                        "motion finished commanded",
                        "velocity_discontinuity",
                        "acceleration_discontinuity",
                        "reflex",
                        "current mode",
                    )
                ):
                    try:
                        if self._robot.has_errors:
                            self._robot.recover_from_errors()
                    except Exception:
                        pass
                    logger.debug("send_velocity swallowed: %s", exc)
                    return
                raise
            self._velocity_active = True
            self._velocity_last_target = twist6.copy()

    def _safe_joint_velocity(self, twist6: np.ndarray) -> np.ndarray:
        """
        Convert a base-frame twist into a 7-DOF joint velocity
        with joint-limit repulsion via gradient projection (Liegeois 1977).
        """
        if self._robot.model is None:
            raise RuntimeError("franky model not loaded")

        state = self._robot.state
        q = np.asarray(state.q, dtype=float)
        J = np.asarray(
            self._robot.model.zero_jacobian(Frame.EndEffector, state), dtype=float
        )

        # Damped pseudo-inverse
        lam2 = 0.05**2
        Jpinv = J.T @ np.linalg.inv(J @ J.T + lam2 * np.eye(6))
        qdot_task = Jpinv @ np.asarray(twist6, dtype=float)

        # Joint-limit repulsion in the null space
        margin_frac = 0.10
        repel_gain = 1.5
        qdot_repel = np.zeros(7)
        for i, (lo, hi) in enumerate(self.JOINT_LIMITS):
            rng = hi - lo
            if rng <= 0:
                continue
            m = margin_frac * rng
            if q[i] > hi - m:
                qdot_repel[i] = -((q[i] - (hi - m)) / m) * repel_gain
            elif q[i] < lo + m:
                qdot_repel[i] = (((lo + m) - q[i]) / m) * repel_gain

        N = np.eye(7) - Jpinv @ J
        qdot_null = N @ qdot_repel
        qdot = qdot_task + qdot_null

        # Hard clamp near limits
        for i, (lo, hi) in enumerate(self.JOINT_LIMITS):
            rng = hi - lo
            if rng <= 0:
                continue
            m = margin_frac * rng
            if q[i] > hi - m and qdot[i] > 0:
                qdot[i] *= max(0.0, (hi - q[i]) / m)
            elif q[i] < lo + m and qdot[i] < 0:
                qdot[i] *= max(0.0, (q[i] - lo) / m)

        return qdot

    def stop_velocity(self):
        # Tear down an active velocity streaming session.
        self._check_connected()
        with self._velocity_lock:
            stopped = False
            for stop_cls in (CartesianVelocityStopMotion, JointVelocityStopMotion):
                try:
                    self._robot.move(stop_cls(), asynchronous=False)
                    stopped = True
                    break
                except Exception:
                    continue
            if not stopped:
                logger.warning("stop_velocity: neither stop type accepted")
            self._velocity_active = False
            self._velocity_last_target = None

    def _preempt_velocity_if_active(self):
        if self._velocity_active:
            logger.debug("Preempting active velocity session")
            self.stop_velocity()

    # Collision behavior

    def set_collision_behavior(
        self,
        torque_threshold: Optional[float] = None,
        force_threshold: Optional[float] = None,
        lower_torque_threshold: Optional[Sequence[float]] = None,
        upper_torque_threshold: Optional[Sequence[float]] = None,
        lower_force_threshold: Optional[Sequence[float]] = None,
        upper_force_threshold: Optional[Sequence[float]] = None,
        lower_torque_threshold_acceleration: Optional[Sequence[float]] = None,
        upper_torque_threshold_acceleration: Optional[Sequence[float]] = None,
        lower_force_threshold_acceleration: Optional[Sequence[float]] = None,
        upper_force_threshold_acceleration: Optional[Sequence[float]] = None,
    ) -> bool:
        """
        Configure libfranka collision detection thresholds.

        Three forms, matching franky's set_collision_behavior overloads:
          1. all eight acc + nominal lists -> per-mode per-joint/axis
          2. the four nominal lists        -> single threshold per joint/axis
          3. nothing -> scalar torque/force broadcast (default)
        """
        self._check_connected()
        acc_lists = (
            lower_torque_threshold_acceleration,
            upper_torque_threshold_acceleration,
            lower_force_threshold_acceleration,
            upper_force_threshold_acceleration,
        )
        nom_lists = (
            lower_torque_threshold,
            upper_torque_threshold,
            lower_force_threshold,
            upper_force_threshold,
        )
        try:
            if all(v is not None for v in acc_lists + nom_lists):
                self._robot.set_collision_behavior(
                    list(lower_torque_threshold_acceleration),
                    list(upper_torque_threshold_acceleration),
                    list(lower_torque_threshold),
                    list(upper_torque_threshold),
                    list(lower_force_threshold_acceleration),
                    list(upper_force_threshold_acceleration),
                    list(lower_force_threshold),
                    list(upper_force_threshold),
                )
            elif all(v is not None for v in nom_lists):
                self._robot.set_collision_behavior(
                    list(lower_torque_threshold),
                    list(upper_torque_threshold),
                    list(lower_force_threshold),
                    list(upper_force_threshold),
                )
            else:
                tt = torque_threshold or self.DEFAULT_COLLISION_TORQUE_THRESHOLD
                ft = force_threshold or self.DEFAULT_COLLISION_FORCE_THRESHOLD
                self._robot.set_collision_behavior(float(tt), float(ft))
            return True
        except Exception as exc:
            logger.warning("set_collision_behavior rejected: %s", exc)
            return False

    def stop(self):
        # Emergency stop.
        self._check_connected()
        with self._servo_lock:
            self._servo_active = False
            self._servo_last_target = None
        with self._velocity_lock:
            self._velocity_active = False
            self._velocity_last_target = None
        self._robot.stop()

    # Helpers

    def _dispatch_motion(self, motion, asynchronous=False):
        # Send a motion to franky, auto-clearing latched Reflex first.
        if self._robot.has_errors:
            try:
                self._robot.recover_from_errors()
                logger.info("auto-cleared reflex before motion")
            except Exception as exc:
                logger.warning("recover_from_errors failed: %s", exc)
        return self._robot.move(motion, asynchronous=asynchronous)

    def _velocity_to_rdf(
        self, commanded_vel: Optional[float], hard_limit: float, default_vel: float
    ) -> float:
        # Translate an absolute velocity into franky's relative dynamics factor.
        v = float(commanded_vel) if commanded_vel is not None else float(default_vel)
        if hard_limit <= 1e-6:
            return 1.0
        return float(np.clip(v / hard_limit, 1e-3, 1.0))

    # Context manager

    def __enter__(self):
        self.connect()
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.disconnect()
        return False

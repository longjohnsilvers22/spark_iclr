"""
Franka FR3 torque-mode impedance driver via pylibfranka.

Uses Robot.start_torque_control() to run an external (Python-side)
Cartesian impedance controller at 1 kHz in a daemon thread. The
impedance math lives in control/impedance_controller.py; this module
owns the pylibfranka lifecycle and threading.

Inherits shared constants, gripper logic, and observation synthesis
from FrankaDriverBase.
"""

from __future__ import annotations

import logging
import math
import threading
import time
from typing import Optional, Sequence

import numpy as np
from scipy.spatial.transform import Rotation

from spark_real.robots.franka.franka_base import FrankaDriverBase, FLANGE_T_TCP
from spark_real.control.impedance_controller import ImpedanceController
from spark_real.utils.rotations import (
    rotmat_to_rotvec as _rotmat_to_rotvec,
    rotvec_to_rotmat as _rotvec_to_rotmat,
    orientation_error as _orientation_error,
)

logger = logging.getLogger(__name__)

try:
    from pylibfranka import (
        ActiveControlBase,
        ControllerMode,
        Gripper,
        GripperState,
        Model,
        RealtimeConfig,
        Robot,
        RobotState,
        Torques,
    )

    HAS_PYLIBFRANKA = True
except ImportError:
    HAS_PYLIBFRANKA = False
    logger.warning("pylibfranka not installed. pip install pylibfranka")


def _colmajor16_to_mat4(flat: Sequence[float]) -> np.ndarray:
    # Convert a 16-element column-major flat array to a 4x4 matrix.
    return np.array(flat, dtype=float).reshape(4, 4, order="F")


# Precomputed flange-to-TCP transform
_FLANGE_T_TCP = FLANGE_T_TCP


class FrankaTorqueDriver(FrankaDriverBase):
    """
    Direct torque-mode impedance control for Franka FR3 via pylibfranka.

    Public API mirrors FrankaDriver so SafeRobot / ScoreExecutor can use
    this as a drop-in replacement.
    """

    SUPPORTS_VELOCITY_STREAMING = True
    DEFAULT_ACCELERATION = 0.5
    DEFAULT_JOINT_ACC = 1.4
    DEFAULT_COLLISION_TORQUE_THRESHOLD = 100.0
    DEFAULT_COLLISION_FORCE_THRESHOLD = 100.0

    def __init__(self, robot_ip: str, frequency: float = 1000.0):
        if not HAS_PYLIBFRANKA:
            raise RuntimeError("pylibfranka not installed. pip install pylibfranka")
        super().__init__(robot_ip, frequency)
        self._robot: Optional[Robot] = None
        self._model: Optional[Model] = None
        self._gripper: Optional[Gripper] = None

        # Impedance controller
        self._impedance = ImpedanceController()

        # Control thread state
        self._control_lock = threading.Lock()
        self._control_thread: Optional[threading.Thread] = None
        self._active_control: Optional[ActiveControlBase] = None
        self._control_running = False

        # Target pose (4x4 homogeneous, base frame)
        self._target_lock = threading.Lock()
        self._target_pose: Optional[np.ndarray] = None
        self._reference_pose: Optional[np.ndarray] = None
        self._target_joints: Optional[np.ndarray] = None
        self._reference_joints: Optional[np.ndarray] = None
        self._control_mode = "cartesian"

        # Nullspace preferred joint configuration
        self._q_nullspace = np.array(self.HOME_CONFIG, dtype=float)

        # Velocity streaming state
        self._target_velocity: Optional[np.ndarray] = None
        self._velocity_mode = False

        # Motion completion event
        self._motion_done = threading.Event()
        self._motion_done.set()
        self._motion_threshold_pos = 0.002
        self._motion_threshold_ori = 0.02
        self._motion_threshold_joint = 0.01

        # Latest robot state (updated at 1kHz)
        self._latest_state: Optional[RobotState] = None
        self._state_lock = threading.Lock()

        # Gripper debounce
        self._last_grip_target_m: Optional[float] = None
        self._last_grip_target_t: float = 0.0

        # Frame correction
        self._EE_T_TCP: np.ndarray = np.eye(4)
        self._TCP_T_EE: np.ndarray = np.eye(4)

    # Connection lifecycle

    def connect(self):
        # Open FCI connection and start the torque control loop.
        logger.info("Connecting to Franka at %s (torque mode)...", self.robot_ip)
        try:
            self._robot = Robot(self.robot_ip)
        except Exception as e:
            if "realtime" in str(e).lower():
                logger.warning("Non-RT kernel (%s). Using kIgnore.", e)
                self._robot = Robot(
                    self.robot_ip, realtime_config=RealtimeConfig.kIgnore
                )
            else:
                raise

        self._model = self._robot.load_model()
        self._gripper = Gripper(self.robot_ip)
        self._connected = True

        try:
            self._robot.set_collision_behavior(
                [100.0] * 7, [100.0] * 7, [100.0] * 6, [100.0] * 6
            )
        except Exception as exc:
            logger.warning("Could not apply collision behavior: %s", exc)

        state = self._robot.read_once()
        self._latest_state = state

        # Compute frame correction (Desk F_T_EE vs URDF fr3_hand_tcp)
        if hasattr(state, "F_T_EE") and state.F_T_EE is not None:
            F_T_EE_desk = _colmajor16_to_mat4(state.F_T_EE)
        else:
            F_T_EE_desk = _FLANGE_T_TCP.copy()
        EE_T_TCP = np.linalg.inv(F_T_EE_desk) @ _FLANGE_T_TCP
        if np.linalg.norm(EE_T_TCP - np.eye(4)) > 1e-6:
            logger.warning(
                "Frame correction active: offset %.4f m, %.2f deg",
                np.linalg.norm(EE_T_TCP[:3, 3]),
                np.degrees(
                    np.linalg.norm(Rotation.from_matrix(EE_T_TCP[:3, :3]).as_rotvec())
                ),
            )
        self._EE_T_TCP = EE_T_TCP
        self._TCP_T_EE = np.linalg.inv(EE_T_TCP)

        T_init = _colmajor16_to_mat4(state.O_T_EE)
        with self._target_lock:
            self._target_pose = T_init.copy()
            self._reference_pose = T_init.copy()
            self._target_joints = np.array(state.q, dtype=float)
            self._reference_joints = np.array(state.q, dtype=float)
            self._q_nullspace = np.array(state.q, dtype=float)

        self._start_control_loop()
        logger.info("Connected (torque mode). Control loop running.")

    def disconnect(self):
        # Stop the control loop and release FCI connection.
        self._stop_control_loop()
        self._robot = None
        self._model = None
        self._gripper = None
        self._connected = False
        logger.info("Disconnected (torque mode)")

    def _check_connected(self):
        if not self._connected:
            raise RuntimeError("Not connected to robot. Call connect() first.")

    # Control loop

    def _start_control_loop(self):
        if self._control_running:
            return
        self._control_running = True
        self._control_thread = threading.Thread(
            target=self._control_loop_body, name="franka-torque-1kHz", daemon=True
        )
        self._control_thread.start()

    def _stop_control_loop(self):
        self._control_running = False
        if self._control_thread is not None:
            self._control_thread.join(timeout=3.0)
            self._control_thread = None
        self._active_control = None

    def _control_loop_body(self):
        # Main 1 kHz torque control loop (runs in daemon thread).
        while self._control_running:
            try:
                self._run_torque_session()
            except Exception as exc:
                if not self._control_running:
                    break
                logger.warning("Torque session ended: %s. Recovering...", exc)
                self._active_control = None
                time.sleep(0.5)
                for attempt in range(3):
                    try:
                        if self._robot is not None:
                            self._robot.automatic_error_recovery()
                        break
                    except Exception as rec_exc:
                        logger.warning("Recovery %d/3 failed: %s", attempt + 1, rec_exc)
                        time.sleep(0.5)
                try:
                    state = self._robot.read_once()
                    T_now = _colmajor16_to_mat4(state.O_T_EE)
                    with self._target_lock:
                        self._reference_pose = T_now.copy()
                        self._target_pose = T_now.copy()
                        q_now = np.array(state.q, dtype=float)
                        self._reference_joints = q_now.copy()
                        self._target_joints = q_now.copy()
                except Exception:
                    pass
                time.sleep(0.5)
        logger.debug("Control loop exited")

    def _run_torque_session(self):
        # Run one torque control session until stopped or error.
        active = self._robot.start_torque_control()
        self._active_control = active
        self._tick_count = 0
        self._gain_scale = 0.0

        state, _ = active.readOnce()
        with self._target_lock:
            T_now = _colmajor16_to_mat4(state.O_T_EE)
            self._reference_pose = T_now.copy()
            if self._target_pose is None:
                self._target_pose = T_now.copy()
            q_now = np.array(state.q, dtype=float)
            self._reference_joints = q_now.copy()
            if self._target_joints is None:
                self._target_joints = q_now.copy()

        while self._control_running:
            state, duration = active.readOnce()
            with self._state_lock:
                self._latest_state = state

            q = np.array(state.q, dtype=float)
            dq = np.array(state.dq, dtype=float)
            tau_J_d = np.array(state.tau_J_d, dtype=float)
            coriolis = np.array(self._model.coriolis(state), dtype=float)

            if self._tick_count < 200:
                # Startup: send zero delta to avoid communication_constraints_violation
                tau_cmd = tau_J_d.copy()
            else:
                self._gain_scale = min(
                    1.0, self._gain_scale + self._impedance.GAIN_RAMP_ALPHA
                )
                tau_cmd = self._compute_torques(state)
                tau_task = tau_cmd - coriolis
                tau_cmd = self._gain_scale * tau_task + coriolis
                tau_cmd += self._impedance.joint_limit_avoidance(q, dq)

            tau_cmd = np.clip(
                tau_cmd, -self._impedance.TAU_LIMITS, self._impedance.TAU_LIMITS
            )
            tau_cmd = self._impedance.saturate_torque_rate(tau_cmd, tau_J_d)

            torques = Torques(tau_cmd.tolist())
            active.writeOnce(torques)
            self._tick_count += 1

        try:
            final = Torques([0.0] * 7)
            final.motion_finished = True
            active.writeOnce(final)
        except Exception:
            pass
        self._active_control = None

    def _compute_torques(self, state: RobotState) -> np.ndarray:
        # Compute commanded joint torques based on current mode.
        q = np.array(state.q, dtype=float)
        dq = np.array(state.dq, dtype=float)
        coriolis = np.array(self._model.coriolis(state), dtype=float)
        J_flat = np.array(self._model.zero_jacobian(state), dtype=float)
        J = J_flat.reshape(6, 7)
        T_ee = _colmajor16_to_mat4(state.O_T_EE)
        p_ee = T_ee[:3, 3]
        R_ee = T_ee[:3, :3]
        v_ee = J @ dq

        with self._target_lock:
            velocity_mode = self._velocity_mode
            control_mode = self._control_mode

            if velocity_mode:
                target_vel = (
                    self._target_velocity.copy()
                    if self._target_velocity is not None
                    else np.zeros(6)
                )
                dt = 0.001
                if self._reference_pose is not None:
                    ref = self._reference_pose.copy()
                    ref[:3, 3] += target_vel[:3] * dt
                    omega = target_vel[3:] * dt
                    if np.linalg.norm(omega) > 1e-10:
                        dR = Rotation.from_rotvec(omega).as_matrix()
                        ref[:3, :3] = dR @ ref[:3, :3]
                    self._reference_pose = ref
                    self._target_pose = ref.copy()
                target_pose = self._reference_pose.copy()
                return self._impedance.cartesian_torques(
                    target_pose,
                    p_ee,
                    R_ee,
                    v_ee,
                    J,
                    q,
                    dq,
                    coriolis,
                    self._q_nullspace,
                    velocity_ff=target_vel,
                )

            if control_mode == "joint":
                target_joints = (
                    self._target_joints.copy()
                    if self._target_joints is not None
                    else q.copy()
                )
                ref_joints = (
                    self._reference_joints.copy()
                    if self._reference_joints is not None
                    else q.copy()
                )
                ref_joints = self._impedance.apply_joint_reference_limiting(
                    ref_joints, target_joints
                )
                self._reference_joints = ref_joints
                return self._impedance.joint_torques(ref_joints, q, dq, coriolis)

            # Cartesian mode
            target_pose = (
                self._target_pose.copy()
                if self._target_pose is not None
                else T_ee.copy()
            )
            ref_pose = (
                self._reference_pose.copy()
                if self._reference_pose is not None
                else T_ee.copy()
            )

        ref_pose = self._impedance.apply_reference_limiting(ref_pose, target_pose)
        with self._target_lock:
            self._reference_pose = ref_pose
        self._check_motion_done(ref_pose, target_pose, p_ee, R_ee)
        return self._impedance.cartesian_torques(
            ref_pose, p_ee, R_ee, v_ee, J, q, dq, coriolis, self._q_nullspace
        )

    def _check_motion_done(self, ref_pose, target_pose, p_ee, R_ee):
        if self._motion_done.is_set():
            return
        ref_pos_err = np.linalg.norm(target_pose[:3, 3] - ref_pose[:3, 3])
        ref_ori_err = np.linalg.norm(
            _orientation_error(target_pose[:3, :3], ref_pose[:3, :3])
        )
        ee_pos_err = np.linalg.norm(target_pose[:3, 3] - p_ee)
        ee_ori_err = np.linalg.norm(_orientation_error(target_pose[:3, :3], R_ee))
        if (
            ref_pos_err < 0.0005
            and ref_ori_err < 0.005
            and ee_pos_err < self._motion_threshold_pos
            and ee_ori_err < self._motion_threshold_ori
        ):
            self._motion_done.set()

    # State queries

    def _get_state(self) -> RobotState:
        with self._state_lock:
            if self._latest_state is not None:
                return self._latest_state
        return self._robot.read_once()

    def get_joint_positions(self) -> np.ndarray:
        self._check_connected()
        return np.array(self._get_state().q, dtype=float)

    def get_joint_velocities(self) -> np.ndarray:
        self._check_connected()
        return np.array(self._get_state().dq, dtype=float)

    def get_tcp_pose(self) -> np.ndarray:
        # Return current TCP pose [x, y, z, rx, ry, rz] in fr3_hand_tcp frame.
        self._check_connected()
        T_ee = _colmajor16_to_mat4(self._get_state().O_T_EE)
        T_tcp = T_ee @ self._EE_T_TCP
        pos = T_tcp[:3, 3]
        rotvec = _rotmat_to_rotvec(T_tcp[:3, :3])
        return np.concatenate([pos, rotvec])

    def get_tcp_force(self) -> np.ndarray:
        self._check_connected()
        try:
            return np.array(self._get_state().O_F_ext_hat_K, dtype=float)
        except AttributeError:
            return np.zeros(6)

    def get_robot_mode(self) -> int:
        self._check_connected()
        try:
            return int(self._get_state().robot_mode)
        except Exception:
            return -1

    def is_steady(self) -> bool:
        return self._motion_done.is_set()

    @property
    def has_errors(self) -> bool:
        self._check_connected()
        try:
            mode = int(self._get_state().robot_mode)
            return mode in (4, 5, 6)
        except Exception:
            return False

    def recover_from_errors(self) -> bool:
        self._check_connected()
        try:
            self._stop_control_loop()
            self._robot.automatic_error_recovery()
            time.sleep(0.1)
            state = self._robot.read_once()
            self._latest_state = state
            T_now = _colmajor16_to_mat4(state.O_T_EE)
            with self._target_lock:
                self._target_pose = T_now.copy()
                self._reference_pose = T_now.copy()
                self._target_joints = np.array(state.q, dtype=float)
                self._reference_joints = np.array(state.q, dtype=float)
                self._velocity_mode = False
                self._target_velocity = None
            self._start_control_loop()
            logger.info("Recovered from errors, control loop restarted")
            return True
        except Exception as exc:
            logger.warning("recover_from_errors failed: %s", exc)
            return False

    # Motion commands

    def _pose_6_to_mat4(self, pose: Sequence[float]) -> np.ndarray:
        # Convert [x, y, z, rx, ry, rz] to 4x4 matrix.
        R = _rotvec_to_rotmat(
            np.array([float(pose[3]), float(pose[4]), float(pose[5])])
        )
        T = np.eye(4)
        T[:3, :3] = R
        T[:3, 3] = [float(pose[0]), float(pose[1]), float(pose[2])]
        return T

    def _check_workspace(self, pose: Sequence[float]):
        x, y, z = pose[0], pose[1], pose[2]
        if not (-0.9 <= x <= 0.9 and -0.9 <= y <= 0.9 and -0.05 <= z <= 1.2):
            raise ValueError(f"Target ({x:.3f}, {y:.3f}, {z:.3f}) outside workspace.")

    def move_to_joint_config(
        self,
        q: Sequence[float],
        velocity: Optional[float] = None,
        acceleration: Optional[float] = None,
        asynchronous: bool = False,
    ):
        # Move to a 7-DOF joint configuration via impedance control.
        self._check_connected()
        q_arr = np.array(q, dtype=float)
        if len(q_arr) != 7:
            raise ValueError(f"Franka expects 7 joints, got {len(q_arr)}.")

        if velocity is not None:
            ratio = min(float(velocity) / self.DEFAULT_JOINT_VEL, 2.0)
            self._impedance.MAX_JOINT_STEP = max(0.0005, 0.001 * ratio)
        else:
            self._impedance.MAX_JOINT_STEP = 0.001

        with self._target_lock:
            self._control_mode = "joint"
            self._velocity_mode = False
            self._target_velocity = None
            self._target_joints = q_arr.copy()

        self._motion_done.clear()

        if not asynchronous:
            q_now = self.get_joint_positions()
            max_delta = float(np.max(np.abs(q_arr - q_now)))
            timeout = max(max_delta / self._impedance.MAX_JOINT_STEP * 0.001 + 2.0, 5.0)
            self._wait_joint_convergence(q_arr, timeout=timeout)
            state = self._get_state()
            T_now = _colmajor16_to_mat4(state.O_T_EE)
            with self._target_lock:
                self._control_mode = "cartesian"
                self._target_pose = T_now.copy()
                self._reference_pose = T_now.copy()

    def _wait_joint_convergence(
        self, q_target: np.ndarray, timeout: float = 30.0, threshold: float = 0.01
    ):
        t0 = time.monotonic()
        while time.monotonic() - t0 < timeout:
            q_now = self.get_joint_positions()
            if float(np.max(np.abs(q_target - q_now))) < threshold:
                return
            time.sleep(0.01)
        logger.warning("move_to_joint_config timed out after %.1fs", timeout)

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
        T_tcp_target = self._pose_6_to_mat4(pose)
        T_ee_target = T_tcp_target @ self._TCP_T_EE

        if velocity is not None:
            ratio = min(float(velocity) / self.DEFAULT_VELOCITY, 3.0)
            self._impedance.MAX_POS_STEP = max(0.0003, 0.001 * ratio)
        else:
            self._impedance.MAX_POS_STEP = 0.001

        with self._target_lock:
            self._control_mode = "cartesian"
            self._velocity_mode = False
            self._target_velocity = None
            self._target_pose = T_ee_target.copy()

        self._motion_done.clear()

        if not asynchronous:
            p_now = self.get_tcp_pose()[:3]
            dist = float(np.linalg.norm(T_tcp_target[:3, 3] - p_now))
            timeout = max(dist / self._impedance.MAX_POS_STEP * 0.001 + 2.0, 5.0)
            self._wait_cartesian_convergence(T_tcp_target, timeout=timeout)

    def _wait_cartesian_convergence(self, T_target: np.ndarray, timeout: float = 30.0):
        t0 = time.monotonic()
        while time.monotonic() - t0 < timeout:
            tcp = self.get_tcp_pose()
            p_err = np.linalg.norm(T_target[:3, 3] - tcp[:3])
            R_now = _rotvec_to_rotmat(tcp[3:])
            o_err = np.linalg.norm(_orientation_error(T_target[:3, :3], R_now))
            if (
                p_err < self._motion_threshold_pos
                and o_err < self._motion_threshold_ori
            ):
                self._motion_done.set()
                return
            time.sleep(0.01)
        logger.warning("move_linear timed out after %.1fs", timeout)

    def move_linear_relative(
        self,
        delta: Sequence[float],
        velocity: Optional[float] = None,
        acceleration: Optional[float] = None,
    ):
        # Move by [dx, dy, dz, drx, dry, drz] relative to current TCP.
        self._check_connected()
        tcp = self.get_tcp_pose()
        new_pos = tcp[:3] + np.array(delta[:3], dtype=float)
        new_ori = tcp[3:] + np.array(delta[3:6], dtype=float)
        target = np.concatenate([new_pos, new_ori])
        self.move_linear(target, velocity=velocity, acceleration=acceleration)

    # Velocity streaming

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
                    f"send_velocity: expected 6-vec, got {linear_arr.size}"
                )
            twist6 = linear_arr
        else:
            angular_arr = np.asarray(angular, dtype=float).ravel()
            if linear_arr.size != 3 or angular_arr.size != 3:
                raise ValueError("send_velocity: expected linear=3, angular=3")
            twist6 = np.concatenate([linear_arr, angular_arr])

        with self._target_lock:
            self._velocity_mode = True
            self._control_mode = "cartesian"
            self._target_velocity = twist6.copy()

    def stop_velocity(self):
        self._check_connected()
        T_now = _colmajor16_to_mat4(self._get_state().O_T_EE)
        with self._target_lock:
            self._velocity_mode = False
            self._target_velocity = None
            self._target_pose = T_now.copy()
            self._reference_pose = T_now.copy()
            self._control_mode = "cartesian"

    def stop(self):
        self._check_connected()
        self.stop_velocity()

    # Servo

    def servo_joint(
        self,
        q: Sequence[float],
        velocity: float = 0.5,
        acceleration: float = 0.5,
        dt: float = 0.002,
        lookahead_time: float = 0.1,
        gain: int = 300,
    ):
        # Update joint target for impedance control (no motion generators).
        self._check_connected()
        q_arr = np.array(q, dtype=float)
        if len(q_arr) != 7:
            raise ValueError(f"Franka expects 7 joints, got {len(q_arr)}.")
        with self._target_lock:
            self._control_mode = "joint"
            self._velocity_mode = False
            self._target_velocity = None
            self._target_joints = q_arr.copy()

    def servo_stop(self):
        self._check_connected()
        state = self._get_state()
        T_now = _colmajor16_to_mat4(state.O_T_EE)
        with self._target_lock:
            self._control_mode = "cartesian"
            self._velocity_mode = False
            self._target_joints = np.array(state.q, dtype=float)
            self._reference_joints = np.array(state.q, dtype=float)
            self._target_pose = T_now.copy()
            self._reference_pose = T_now.copy()

    # Collision behavior

    def set_collision_behavior(
        self,
        torque_threshold: Optional[float] = None,
        force_threshold: Optional[float] = None,
        lower_torque_threshold: Optional[Sequence[float]] = None,
        upper_torque_threshold: Optional[Sequence[float]] = None,
        lower_force_threshold: Optional[Sequence[float]] = None,
        upper_force_threshold: Optional[Sequence[float]] = None,
    ) -> bool:
        self._check_connected()
        try:
            if all(
                v is not None
                for v in (
                    lower_torque_threshold,
                    upper_torque_threshold,
                    lower_force_threshold,
                    upper_force_threshold,
                )
            ):
                self._robot.set_collision_behavior(
                    list(lower_torque_threshold),
                    list(upper_torque_threshold),
                    list(lower_force_threshold),
                    list(upper_force_threshold),
                )
            else:
                tt = torque_threshold or self.DEFAULT_COLLISION_TORQUE_THRESHOLD
                ft = force_threshold or self.DEFAULT_COLLISION_FORCE_THRESHOLD
                self._robot.set_collision_behavior(
                    [float(tt)] * 7, [float(tt)] * 7, [float(ft)] * 6, [float(ft)] * 6
                )
            return True
        except Exception as exc:
            logger.warning("set_collision_behavior rejected: %s", exc)
            return False

    # Impedance gain tuning

    # Gripper (FrankaDriverBase abstract implementations)

    def _gripper_open(self, speed: float):
        self._check_connected()
        self._gripper.move(self.GRIPPER_OPEN_WIDTH, float(speed))

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
        try:
            return float(self._gripper.read_once().width)
        except Exception:
            return 0.0

    def _gripper_is_grasped(self) -> bool:
        self._check_connected()
        try:
            return bool(self._gripper.read_once().is_grasped)
        except Exception:
            return False

    def _gripper_max_width(self) -> float:
        try:
            return float(self._gripper.read_once().max_width)
        except Exception:
            return self.GRIPPER_OPEN_WIDTH

    def _gripper_homing(self):
        self._check_connected()
        ok = self._gripper.homing()
        time.sleep(0.5)
        if not ok:
            logger.warning("Franka Hand homing did not report success")

    # Franky-specific gripper override (debounced set_gripper_position)

    def set_gripper_position(
        self,
        position: float,
        speed: Optional[float] = None,
        force: Optional[float] = None,
    ):
        pos = float(np.clip(position, 0.0, 1.0))
        max_w = self._gripper_max_width()
        width = max_w * (1.0 - pos)
        spd = speed if speed is not None else self.GRIPPER_DEFAULT_SPEED
        self._check_connected()

        now = time.monotonic()
        if (
            self._last_grip_target_m is not None
            and abs(width - self._last_grip_target_m) < 0.003
            and now - self._last_grip_target_t < 0.150
        ):
            return
        self._last_grip_target_m = width
        self._last_grip_target_t = now

        if pos < 0.95:
            self._gripper.move(float(width), float(spd))
        else:
            frc = force if force is not None else self.GRIPPER_DEFAULT_FORCE
            self._gripper.grasp(
                float(width), float(spd), float(frc), float(max_w), float(max_w)
            )

    @property
    def gripper_width(self) -> float:
        return self.get_gripper_width()

    # Context manager

    def __enter__(self):
        self.connect()
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.disconnect()
        return False

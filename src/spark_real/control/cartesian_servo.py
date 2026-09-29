"""
Cartesian velocity servo controller for UR10e.

OSC-style controller that drives the TCP to a target pose using
speedl velocity commands at high rate. Gives precise control over
both position and orientation, bypassing the UR IK solver's wrist
configuration selection.

Usage:
    servo = CartesianServo(robot, rate_hz=125)
    servo.move_to_pose(target_pose, velocity=0.15)
"""

import logging
import time

import numpy as np
from scipy.spatial.transform import Rotation

from spark_real.control.command_latch import SOURCE_SERVO, command_source

logger = logging.getLogger(__name__)


class CartesianServo:
    """
    Velocity-based Cartesian servo for UR10e via speedl.
    """

    # Slowest commanded linear approach while outside pos_threshold (m/s).
    MIN_APPROACH_VEL_LINEAR = 0.03

    def __init__(self, robot, rate_hz: float = 125.0):
        self.robot = robot
        # On Franka, send_velocity goes through franky's per-tick
        # CartesianVelocityMotion; above ~30 Hz Ruckig can't replan fast
        # enough and the velocity-discontinuity reflex fires. Cap non-UR
        # drivers at 30 Hz; UR's URScript speedl tolerates the higher rate.
        if not getattr(robot, "SUPPORTS_URSCRIPT", False):
            rate_hz = min(rate_hz, 30.0)
        self.dt = 1.0 / rate_hz

        # PD gains kept conservative: near FR3's wrist singularity (J5 ~ 0,
        # the default top-down config) small Cartesian velocities map to
        # large joint velocities and trip the joint_velocity reflex.
        self.kp_pos = 2.0
        self.kp_ori = 1.5
        self.kd_pos = 0.5
        self.kd_ori = 0.4

        # Conservative linear/angular caps tuned for stable PD servoing near
        # the FR3 wrist singularity.
        self.max_vel_linear = 0.15  # m/s
        self.max_vel_angular = 0.6  # rad/s
        # Slowest the arm may approach while still outside pos_threshold. See
        # _compute_velocity: without this the proportional law crawls the last
        # centimetres. Small enough to stop inside one tick of the arrival
        # tolerance (0.03 m/s * 8 ms = 0.24 mm, vs a 3 mm threshold).
        self.min_vel_linear = self.MIN_APPROACH_VEL_LINEAR
        self.accel = 1.0  # m/s^2 for speedl

        # Convergence thresholds
        self.pos_threshold = 0.003  # 3mm
        # ~3 deg. The speedl angular PD has a steady-state yaw error above
        # 1 deg on oriented grasps, so a tighter tolerance never converges
        # and the servo runs the full timeout.
        self.ori_threshold = 0.05
        self.settle_time = 0.15  # dwell at target before exit

        # Stall detection. A target a few mm past a mechanical stop (object
        # bottomed out on a rim, target Z estimate slightly low) never
        # converges: the PD holds a small steady command into the stop for the
        # whole timeout. Measured offline: 4 mm of unreachability turned a
        # 3.8 s descent into 15.01 s / 1877 consecutive speedl commands. If
        # neither error has improved by a meaningful margin for this long,
        # more time will not help.
        self.stall_window = 1.2  # s without progress before giving up
        self.stall_pos_eps = 0.0005  # m of improvement that counts as progress
        self.stall_ori_eps = 0.005  # rad, ditto

        self._abort = False
        self._prev_vel = np.zeros(6)
        # Diagnostics for the caller / logs, set by every move_to_pose call.
        self.last_pos_err = float("nan")
        self.last_ori_err = float("nan")
        self.last_exit = "none"

    def abort(self):
        self._abort = True

    def _get_tcp_pose(self) -> np.ndarray:
        """
        Read current TCP pose [x, y, z, rx, ry, rz].
        """
        if hasattr(self.robot, "get_tcp_pose"):
            p = self.robot.get_tcp_pose()
            if isinstance(p, np.ndarray) and p.shape == (4, 4):
                pos = p[:3, 3]
                rotvec = Rotation.from_matrix(p[:3, :3]).as_rotvec()
                return np.concatenate([pos, rotvec])
            return np.array(p[:6])
        # No zeros fallback: that would make the servo chase from the origin.
        raise AttributeError(
            "CartesianServo needs a robot with get_tcp_pose; got "
            f"{type(self.robot).__name__}."
        )

    def _send_speedl(self, velocity: np.ndarray):
        """
        Send a base-frame Cartesian velocity command.

        Family-aware: UR robots get URScript ``speedl`` via _send_script;
        other drivers (Franka, G1) take the same 6-vector through their
        send_velocity shim. Dispatch is gated on the SUPPORTS_URSCRIPT flag,
        not ``hasattr(_send_script)``: the SafeRobot wrapper defines
        ``_send_script`` as a method that always raises for non-UR drivers.
        """
        v = velocity.tolist()
        with command_source(SOURCE_SERVO):
            if getattr(self.robot, "SUPPORTS_URSCRIPT", False) and hasattr(
                self.robot, "_send_script"
            ):
                # The URScript branch is kept separate from send_velocity on
                # purpose: on UR the driver is wrapped in SafeRobot, whose
                # send_velocity runs the CBF-QP while _send_script passes
                # speedl through untouched. The commanded twist is latched
                # explicitly so there is still exactly one latch object.
                latch = getattr(self.robot, "latch_command_velocity", None)
                if callable(latch):
                    latch(v)
                script = (
                    f"speedl([{v[0]:.5f}, {v[1]:.5f}, {v[2]:.5f}, "
                    f"{v[3]:.5f}, {v[4]:.5f}, {v[5]:.5f}], "
                    f"{self.accel}, {self.dt + 0.02})"
                )
                self.robot._send_script(script)
            elif hasattr(self.robot, "send_velocity"):
                try:
                    self.robot.send_velocity(
                        v, acceleration=self.accel, duration=self.dt + 0.02
                    )
                except TypeError:
                    # Legacy positional / time_duration variant
                    self.robot.send_velocity(v, self.accel, self.dt + 0.02)

    def _stop(self):
        """
        Decelerate to zero. Family-aware; see _send_speedl.
        """
        # Zero is a command too: without this the recorder would keep
        # reporting the last non-zero twist through the whole settle dwell.
        latch = getattr(self.robot, "latch_command_velocity", None)
        if callable(latch):
            latch([0.0] * 6, SOURCE_SERVO)
        if getattr(self.robot, "SUPPORTS_URSCRIPT", False) and hasattr(
            self.robot, "_send_script"
        ):
            self.robot._send_script("speedl([0,0,0,0,0,0], 1.0, 0.1)\nstopl(1.0)")
        elif hasattr(self.robot, "stop_velocity"):
            try:
                self.robot.stop_velocity()
            except Exception:
                pass
        elif hasattr(self.robot, "send_velocity"):
            try:
                self.robot.send_velocity(
                    [0, 0, 0, 0, 0, 0], acceleration=1.0, duration=0.1
                )
            except TypeError:
                self.robot.send_velocity([0, 0, 0, 0, 0, 0], 1.0, 0.1)

    def _compute_velocity(self, current: np.ndarray, target: np.ndarray) -> np.ndarray:
        """
        Compute 6D Cartesian velocity command from pose error.
        """
        pos_err = target[:3] - current[:3]

        # Orientation error as compact rotvec
        R_cur = Rotation.from_rotvec(current[3:6])
        R_tgt = Rotation.from_rotvec(target[3:6])
        R_err = R_tgt * R_cur.inv()
        ori_err = R_err.as_rotvec()

        # PD control (damping on velocity change)
        v_pos = self.kp_pos * pos_err - self.kd_pos * self._prev_vel[:3]
        v_ori = self.kp_ori * ori_err - self.kd_ori * self._prev_vel[3:]

        vel = np.concatenate([v_pos, v_ori])

        # Clamp linear and angular separately
        lin_norm = np.linalg.norm(vel[:3])
        if lin_norm > self.max_vel_linear:
            vel[:3] *= self.max_vel_linear / lin_norm
        ang_norm = np.linalg.norm(vel[3:])
        if ang_norm > self.max_vel_angular:
            vel[3:] *= self.max_vel_angular / ang_norm

        # Floor the approach speed while still OUTSIDE the arrival tolerance.
        # v = kp * err decays exponentially, so the tail dominates the move.
        # Measured descending into the bowl: 12.7 cm in 0.4 s, then 2.2 cm in
        # 2.2 s, ending at 0.001 m/s. The floor applies only outside
        # pos_threshold; flooring inside it would command motion at the target
        # and buzz around it forever.
        pos_dist = float(np.linalg.norm(pos_err))
        if pos_dist > self.pos_threshold:
            lin_norm = float(np.linalg.norm(vel[:3]))
            floor = min(self.min_vel_linear, self.max_vel_linear)
            if 0.0 < lin_norm < floor:
                vel[:3] *= floor / lin_norm
            elif lin_norm <= 0.0:
                # PD damping cancelled the command outright while still short
                # of the target: drive straight at it rather than stall.
                vel[:3] = pos_err / max(pos_dist, 1e-9) * floor

        return vel

    def move_to_pose(
        self, target_pose: list, velocity: float = 0.15, timeout: float = 15.0
    ) -> bool:
        """
        Drive TCP to target pose using Cartesian velocity servo.

        Args:
            target_pose: [x, y, z, rx, ry, rz] target in axis-angle
            velocity: Max linear velocity scale (adjusts kp)
            timeout: Abort after this many seconds

        Returns:
            True if converged, False if timed out or aborted
        """
        self._abort = False
        self._prev_vel = np.zeros(6)
        target = np.array(target_pose, dtype=np.float64)

        # `velocity` reaches the loop only through the caller's max_vel_linear
        # clamp (see executor_motion._servo_to), not through the gains.

        t0 = time.time()
        settle_start = None
        best_pos = np.inf
        best_ori = np.inf
        last_progress = t0
        pos_err = np.inf
        ori_err = np.inf

        while time.time() - t0 < timeout:
            if self._abort:
                self._stop()
                self.last_exit = "abort"
                return False

            current = self._get_tcp_pose()

            pos_err = np.linalg.norm(target[:3] - current[:3])
            R_cur = Rotation.from_rotvec(current[3:6])
            R_tgt = Rotation.from_rotvec(target[3:6])
            ori_err = np.linalg.norm((R_tgt * R_cur.inv()).as_rotvec())

            if pos_err < self.pos_threshold and ori_err < self.ori_threshold:
                if settle_start is None:
                    settle_start = time.time()
                elif time.time() - settle_start > self.settle_time:
                    self._stop()
                    self._finish("converged", pos_err, ori_err)
                    logger.info(
                        "Servo converged: pos_err=%.4f ori_err=%.4f (%.2fs)",
                        pos_err,
                        ori_err,
                        time.time() - t0,
                    )
                    return True
            else:
                settle_start = None
                # Progress on EITHER channel resets the clock: a descent that
                # has reached its Z but is still rotating is making progress.
                now = time.time()
                if pos_err < best_pos - self.stall_pos_eps:
                    best_pos = pos_err
                    last_progress = now
                if ori_err < best_ori - self.stall_ori_eps:
                    best_ori = ori_err
                    last_progress = now
                if now - last_progress > self.stall_window:
                    self._stop()
                    self._finish("stalled", pos_err, ori_err)
                    logger.warning(
                        "Servo STALLED: no progress for %.1fs at pos_err=%.4f "
                        "ori_err=%.4f (%.2fs elapsed). Target is unreachable "
                        "-- blocked, or past a mechanical stop.",
                        self.stall_window,
                        pos_err,
                        ori_err,
                        now - t0,
                    )
                    return False

            vel = self._compute_velocity(current, target)
            self._send_speedl(vel)
            self._prev_vel = vel

            time.sleep(self.dt)

        self._stop()
        current = self._get_tcp_pose()
        pos_err = np.linalg.norm(target[:3] - current[:3])
        self._finish("timeout", pos_err, ori_err)
        logger.warning(
            "Servo timed out: pos_err=%.4f (timeout=%.1fs)", pos_err, timeout
        )
        return False

    def _finish(self, exit_reason: str, pos_err: float, ori_err: float) -> None:
        """Record why the last move_to_pose ended, for the caller and the logs."""
        self.last_exit = exit_reason
        self.last_pos_err = float(pos_err)
        self.last_ori_err = float(ori_err)

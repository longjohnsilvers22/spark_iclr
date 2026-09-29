"""
FrankaBambooDriver: joint impedance control via Bamboo C++ node.

Drop-in replacement for FrankaDriver. Bamboo runs a 1kHz C++ joint
impedance controller that eliminates ALL franky motion generator reflex
errors (joint_motion_generator_velocity_discontinuity,
cartesian_motion_generator_acceleration_discontinuity, etc.).

Requires:
  - bamboo_control_node binary (built in external_controllers/bamboo)
  - pip install bamboo-franka-client
  - LD_LIBRARY_PATH includes bamboo/install/lib and /opt/openrobots/lib

Bamboo reports ee_pose as O_T_EE from libfranka (flange frame when no
Franka Hand is configured in Desk, which is the case for SSG-48 rigs).

The driver manages the bamboo C++ subprocess lifecycle: _start_bamboo()
launches it, _stop_bamboo() tears it down, and _restart_bamboo() is
called automatically when a trajectory fails (with one retry).

Source: chsahit/bamboo (GitHub)
"""

from __future__ import annotations

import logging
import os
import signal
import subprocess
import threading
import time
from pathlib import Path
from typing import Optional, Sequence

import numpy as np
from scipy.spatial.transform import Rotation

from spark_real.control.pyroki_planner import BimanualPyrokiPlanner

logger = logging.getLogger(__name__)

try:
    import panda_py
except ImportError:
    panda_py = None

try:
    from bamboo.client import BambooFrankaClient
except ImportError:
    BambooFrankaClient = None

_BAMBOO_BINARY = Path(
    os.environ.get(
        "SPARK_BAMBOO_BINARY",
        str(
            Path.home()
            / "spark/src/external_controllers/bamboo/controller/build"
            / "bamboo_control_node"
        ),
    )
)
_BAMBOO_LD_PATH = ":".join(
    [
        "/opt/openrobots/lib",
        str(Path.home() / "spark/src/external_controllers/bamboo/install/lib"),
    ]
)

# Bamboo SSG-48 ready pose: rounded J5=0 variant, distinct from
# franka_base.HOME_CONFIG / franka_default.yaml home_config (do not unify).
HOME_CONFIG = [0.0, -0.785, 0.0, -2.356, 0.0, 1.571, 0.785]


class FrankaBambooDriver:
    """
    FrankaDriver-shaped wrapper around BambooFrankaClient.
    """

    SUPPORTS_URSCRIPT = False
    SUPPORTS_VELOCITY_STREAMING = True
    GRIPPER_TYPE = "ssg48"
    HOME_CONFIG = np.array(HOME_CONFIG, dtype=float)

    JOINT_CONVERGE_RAD = 0.015
    JOINT_CONVERGE_TIMEOUT_S = 8.0

    def __init__(self, ip: str, port: int = 5555):
        self._ip = ip
        self._port = port
        self._client = None
        self._connected = False
        self._lock = threading.Lock()
        self._bamboo_proc: Optional[subprocess.Popen] = None
        self._bamboo_log = Path(f"/tmp/bamboo_{port}.log")
        self.has_errors = False

    # Bamboo subprocess management

    def _start_bamboo(self):
        # Launch the bamboo C++ control node as a managed subprocess.
        if not _BAMBOO_BINARY.exists():
            raise FileNotFoundError(f"Bamboo binary not found: {_BAMBOO_BINARY}")
        env = os.environ.copy()
        existing = env.get("LD_LIBRARY_PATH", "")
        env["LD_LIBRARY_PATH"] = (
            f"{_BAMBOO_LD_PATH}:{existing}" if existing else _BAMBOO_LD_PATH
        )
        cmd = [
            str(_BAMBOO_BINARY),
            "-r",
            self._ip,
            "-p",
            str(self._port),
            "-l",
            "*",
            "-g",
            "none",
            "-m",
        ]
        logger.info("Starting Bamboo: %s", " ".join(cmd))
        # Append, never truncate: a mid-run restart must not destroy the
        # failed trajectory's reflex string (the only diagnosis we get).
        log_f = open(self._bamboo_log, "a")
        log_f.write(f"\nbamboo start {time.strftime('%H:%M:%S')}\n")
        log_f.flush()
        start_off = log_f.tell()
        self._bamboo_proc = subprocess.Popen(
            cmd,
            env=env,
            stdout=log_f,
            stderr=subprocess.STDOUT,
            preexec_fn=os.setsid,
        )
        deadline = time.monotonic() + 15.0
        while time.monotonic() < deadline:
            if self._bamboo_proc.poll() is not None:
                raise RuntimeError(
                    f"Bamboo exited {self._bamboo_proc.returncode}. "
                    f"See {self._bamboo_log}"
                )
            with open(self._bamboo_log, "rb") as lf:
                lf.seek(start_off)
                tail = lf.read().decode(errors="ignore")
            if "Server listening" in tail:
                logger.info("Bamboo ready on port %d", self._port)
                return
            time.sleep(0.3)
        raise TimeoutError(f"Bamboo didn't start in 15s. See {self._bamboo_log}")

    def _stop_bamboo(self):
        # Terminate the managed bamboo subprocess if running.
        if self._bamboo_proc is None or self._bamboo_proc.poll() is not None:
            self._bamboo_proc = None
            return
        try:
            os.killpg(os.getpgid(self._bamboo_proc.pid), signal.SIGTERM)
            self._bamboo_proc.wait(timeout=5)
        except subprocess.TimeoutExpired:
            os.killpg(os.getpgid(self._bamboo_proc.pid), signal.SIGKILL)
            self._bamboo_proc.wait(timeout=3)
        except Exception:
            pass
        self._bamboo_proc = None

    def _restart_bamboo(self):
        # Stop bamboo, restart it, and reconnect the ZMQ client.
        logger.warning("Restarting bamboo on port %d ...", self._port)
        self._stop_bamboo()
        time.sleep(2)
        self._start_bamboo()
        time.sleep(1)
        if BambooFrankaClient is None:
            raise ImportError("bamboo client not installed. pip install bamboo-franka-client")
        self._client = BambooFrankaClient(
            control_port=self._port,
            server_ip="localhost",
            enable_gripper=False,
        )
        logger.info("Bamboo restarted successfully on port %d", self._port)

    # Connection

    def connect(self):
        if BambooFrankaClient is None:
            raise ImportError("bamboo client not installed. pip install bamboo-franka-client")
        self._start_bamboo()
        time.sleep(0.5)
        self._client = BambooFrankaClient(
            control_port=self._port,
            server_ip="localhost",
            enable_gripper=False,
        )
        self._connected = True
        q = np.round(self._client.get_joint_positions(), 3).tolist()
        logger.info("BambooDriver connected %s:%d q=%s", self._ip, self._port, q)

    def reconnect(self):
        # Full restart: stop bamboo, relaunch, reconnect ZMQ client.
        try:
            if self._client is not None:
                self._client.close()
        except Exception:
            pass
        self._connected = False
        self._restart_bamboo()
        self._connected = True
        q = np.round(self._client.get_joint_positions(), 3).tolist()
        logger.info("BambooDriver reconnected %s:%d q=%s", self._ip, self._port, q)

    def disconnect(self):
        if self._client is not None:
            try:
                self._client.close()
            except Exception:
                pass
        self._client = None
        self._connected = False
        self._stop_bamboo()

    def _check(self):
        if not self._connected:
            raise RuntimeError("BambooDriver not connected")

    # state

    def get_joint_positions(self) -> list:
        self._check()
        return list(self._client.get_joint_positions())

    def get_joint_velocities(self) -> np.ndarray:
        self._check()
        st = self._client.get_joint_states()
        return np.array(st.get("dq", [0.0] * 7), dtype=float)

    def get_tcp_pose(self) -> list:
        # Return flange pose (not hand_tcp) to match franky and calibration.
        self._check()
        q = self._client.get_joint_positions()
        if panda_py is None:
            raise ImportError("panda_py not installed. pip install panda-python")
        T = panda_py.fk(q[:7])
        pos = T[:3, 3]
        rotvec = Rotation.from_matrix(T[:3, :3]).as_rotvec()
        return [
            float(pos[0]),
            float(pos[1]),
            float(pos[2]),
            float(rotvec[0]),
            float(rotvec[1]),
            float(rotvec[2]),
        ]

    def get_tcp_force(self) -> list:
        self._check()
        st = self._client.get_joint_states()
        tau = st.get("tau_J", [0.0] * 7)
        return [0.0, 0.0, 0.0, 0.0, 0.0, 0.0]

    def get_observation(self) -> dict:
        return {
            "joint_positions": self.get_joint_positions(),
            "joint_velocities": self.get_joint_velocities().tolist(),
            "tcp_pose": self.get_tcp_pose(),
            "tcp_force": self.get_tcp_force(),
            "ts": time.time(),
        }

    # motion

    def move_to_joint_config(
        self,
        q,
        velocity=None,
        acceleration=None,
        asynchronous=False,
        dynamics_factor=None,
    ):
        self._check()
        q_arr = np.array(q, dtype=float).reshape(1, 7)
        vel = np.zeros((1, 7))
        q_current = np.array(self._client.get_joint_positions())
        max_dq = float(np.max(np.abs(q_arr[0] - q_current)))
        speed = max(float(velocity if velocity is not None else 0.3), 0.05)
        # Acceleration-limited duration. Min-jerk peak accel = 5.77*D/T^2, so a
        # pure velocity->duration map (T = D/v) lets short high-speed moves spike
        # the joint-impedance torque past libfranka's wrench model and trip
        # cartesian_reflex. Bounding T >= sqrt(5.77*D/A_MAX) caps the spike so big
        # moves run fast and small moves get enough time.
        A_MAX = 1.5
        t_vel = max_dq / speed
        t_acc = float(np.sqrt(5.77 * max_dq / A_MAX)) if max_dq > 0 else 0.0
        dur = float(np.clip(max(t_vel, t_acc), 0.3, 10.0))
        with self._lock:
            result = self._client.execute_joint_impedance_path(
                q_arr, joint_vels=vel, durations=[dur]
            )
            if isinstance(result, dict) and not result.get("success", True):
                logger.warning(
                    "Bamboo trajectory failed (dur=%.1f delta=%.2f), "
                    "restarting controller...",
                    dur,
                    max_dq,
                )
                self._restart_bamboo()
                result = self._client.execute_joint_impedance_path(
                    q_arr, joint_vels=vel, durations=[dur]
                )
                if isinstance(result, dict) and not result.get("success", True):
                    raise RuntimeError(
                        f"Bamboo trajectory failed after restart: "
                        f"{result.get('error', 'unknown')}"
                    )

    def move_linear(self, pose, velocity=None, acceleration=None, asynchronous=False):
        if panda_py is None:
            raise ImportError("panda_py not installed. pip install panda-python")
        x, y, z = float(pose[0]), float(pose[1]), float(pose[2])
        rx, ry, rz = float(pose[3]), float(pose[4]), float(pose[5])
        quat_xyzw = Rotation.from_rotvec([rx, ry, rz]).as_quat()
        q_init = np.array(self._client.get_joint_positions()[:7])
        # Try panda_py analytical IK first (fast)
        q_target = panda_py.ik(np.array([x, y, z]), quat_xyzw, q_init=q_init)
        if q_target is None or np.any(np.isnan(q_target)):
            planner = BimanualPyrokiPlanner()
            wxyz = [
                float(quat_xyzw[3]),
                float(quat_xyzw[0]),
                float(quat_xyzw[1]),
                float(quat_xyzw[2]),
            ]
            arm = self._arm_hint if hasattr(self, "_arm_hint") else "right"
            q_target = planner.solve(
                arm=arm,
                target_position_base=np.array([x, y, z]),
                target_wxyz_base=np.array(wxyz),
                prev_cfg=q_init,
            )
        self.move_to_joint_config(list(q_target[:7]))

    def go_home(self, velocity=None):
        self.move_to_joint_config(self.HOME_CONFIG.tolist())

    def stop(self):
        self._check()
        with self._lock:
            q = self._client.get_joint_positions()
            self._client.execute_joint_impedance_path(
                np.array(q).reshape(1, 7), joint_vels=np.zeros((1, 7)), durations=[0.1]
            )

    def get_robot_mode(self) -> int:
        # Return 2 (idle/running) if bamboo is alive, 4 (error) if dead.
        if self._bamboo_proc and self._bamboo_proc.poll() is not None:
            return 4
        return 2

    def is_steady(self) -> bool:
        """
        Bamboo executes blocking trajectories, so control always returns
        after the impedance path has settled. Treat the arm as steady
        whenever it is connected and the controller subprocess is alive.

        (The franky driver derives this from libfranka's motion-generator
        state; bamboo has no equivalent async motion flag because
        ``execute_joint_impedance_path`` is synchronous.)
        """
        if not self._connected:
            return False
        if self._bamboo_proc is not None and self._bamboo_proc.poll() is not None:
            return False
        return True

    def move_linear_relative(
        self, delta, velocity=None, acceleration=None, asynchronous=False
    ):
        """
        Cartesian relative move: target = FK(current) composed with delta.

        ``delta`` is a 6-vec [dx, dy, dz, drx, dry, drz] interpreted in the
        BASE frame (matching the single-arm FrankaDriver convention: the
        translation is added in base coordinates and the rotvec is applied
        as a base-frame rotation pre-multiplying the current orientation).
        Routes through the existing IK + move_to_joint_config path that
        :meth:`move_linear` already uses, so no extra realtime capability
        is assumed.
        """
        self._check()
        if panda_py is None:
            raise ImportError("panda_py not installed. pip install panda-python")
        d = np.asarray(delta, dtype=float).ravel()
        if d.size < 6:
            d = np.pad(d, (0, 6 - d.size))

        q_cur = np.array(self._client.get_joint_positions(), dtype=float)
        T = panda_py.fk(q_cur[:7])
        pos = T[:3, 3].copy()
        rot = Rotation.from_matrix(T[:3, :3])

        pos = pos + d[:3]
        drot = Rotation.from_rotvec(d[3:6])
        rot = drot * rot

        target_pose = [float(pos[0]), float(pos[1]), float(pos[2])]
        target_pose.extend(rot.as_rotvec().tolist())
        self.move_linear(
            target_pose,
            velocity=velocity,
            acceleration=acceleration,
            asynchronous=asynchronous,
        )

    def servo_joint(self, q, *args, **kwargs):
        """
        Realtime joint servo is NOT supported on the bamboo backend.

        Bamboo's contract is blocking ``execute_joint_impedance_path``
        trajectories with accel-limited durations: there is no 1 kHz
        external-torque/position servo stream to push setpoints into. The
        single-arm FrankaDriver.servo_joint feeds franky's JointWaypointMotion
        in async mode; bamboo has no analogue. Raise explicitly so callers
        see the unsupported path rather than silently getting a blocking
        point-to-point move with the wrong dynamics.

        If a non-realtime equivalent is acceptable at the call site, use
        ``move_to_joint_config`` directly.
        """
        raise NotImplementedError(
            "FrankaBambooDriver does not support servo_joint (no realtime "
            "joint-position servo stream). Use move_to_joint_config for "
            "blocking point-to-point moves instead."
        )

    def servo_stop(self):
        # No realtime servo loop to stop; fall back to a braking hold.
        if self._connected:
            self.stop()

    # error / safety

    def recover_from_errors(self):
        self.has_errors = False

    def set_collision_behavior(self, *args, **kwargs):
        pass

    def stop_velocity(self):
        self.stop()

    def send_velocity(
        self, linear, angular=None, acceleration: float = 0.5, duration: float = 0.1
    ) -> None:
        """
        Integrate a Cartesian twist into a joint target and send it.

        Signature matches the single-arm/franky FrankaDriver.send_velocity
        so the bimanual driver can forward calls unchanged:
          * ``linear`` may be a full 6-vec twist (when ``angular is None``),
            or the 3-vec linear part with ``angular`` supplying the 3-vec
            angular part.
          * ``duration`` is the integration window (s); a non-positive value
            falls back to a 0.1 s tick.

        Mirrors the FrankaOSCDriver approach: compute a small pose delta
        from the twist, IK to joint space, then command via
        execute_joint_impedance_path with a short duration so the
        impedance controller interpolates smoothly.
        """
        self._check()
        if panda_py is None:
            raise ImportError("panda_py not installed. pip install panda-python")
        linear_arr = np.asarray(linear, dtype=float).ravel()
        if angular is None:
            v = linear_arr[:6]
            if v.size < 6:
                v = np.pad(v, (0, 6 - v.size))
        else:
            angular_arr = np.asarray(angular, dtype=float).ravel()
            v = np.concatenate([linear_arr[:3], angular_arr[:3]])
            if v.size < 6:
                v = np.pad(v, (0, 6 - v.size))

        dt = float(duration) if float(duration) > 0 else 0.1

        # Current pose from FK (matches get_tcp_pose)
        q_cur = np.array(self._client.get_joint_positions(), dtype=float)
        T = panda_py.fk(q_cur[:7])
        pos = T[:3, 3].copy()
        rot = Rotation.from_matrix(T[:3, :3])

        # Integrate linear velocity
        pos += v[:3] * dt

        # Integrate angular velocity (axis-angle delta)
        omega = v[3:6]
        angle = np.linalg.norm(omega)
        if angle > 1e-8:
            delta_rot = Rotation.from_rotvec(omega * dt)
            rot = delta_rot * rot

        quat_xyzw = rot.as_quat()

        # IK for joint target
        try:
            q_target = panda_py.ik(pos, quat_xyzw, q_init=q_cur[:7])
        except Exception:
            quat_wxyz = np.roll(quat_xyzw, 1)
            q_target = panda_py.ik(pos, quat_wxyz, q_init=q_cur[:7])

        q_target = np.array(q_target, dtype=float)
        if q_target.size != 7 or not np.all(np.isfinite(q_target)):
            logger.warning("send_velocity: IK failed, skipping tick")
            return

        # Send as a single-waypoint impedance path with short duration.
        # Use zero terminal velocity so the controller decelerates at the
        # waypoint (the next tick will update the target).
        with self._lock:
            self._client.execute_joint_impedance_path(
                q_target.reshape(1, 7),
                joint_vels=np.zeros((1, 7)),
                durations=[max(dt, 0.05)],
            )

    # gripper stubs (SSG-48 handled externally)

    def activate_gripper(self):
        pass

    def open_gripper(self, **kwargs):
        pass

    def close_gripper(self, **kwargs):
        pass

    def get_gripper_width(self) -> float:
        return 0.09

    def is_object_detected(self) -> bool:
        return False

    @property
    def current_joint_state(self):
        class _JS:
            def __init__(self, q):
                self.position = q

        return _JS(self.get_joint_positions())

"""
UR10e real-time control via ur_rtde.

Provides joint/Cartesian motion, Robotiq 2F-85 gripper control, and
force/torque reading over the RTDE protocol. Motion/state go through
ur_rtde; raw URScript (speedl/speedj for CartesianServo, VLA velocity
streaming) goes through a primary-interface TCP socket on port 30002,
mirroring the external teleop UR10eRobot the legacy pipeline imported.
"""

import logging
import os
import re
import socket
import threading
import numpy as np
import time
from pathlib import Path
from typing import Optional, Sequence

from spark_real.control.command_latch import CommandLatch, CommandSnapshot

logger = logging.getLogger(__name__)

# Commanded-speed parsers, used by _note_motion_script to size the brake.
# A blended program carries one v= per row; the leading row wins.
_MOVE_V_RE = re.compile(r"move[jl]\([^)]*?\bv=([-\d.eE+]+)")
_SPEED_VEC_RE = re.compile(r"speed([jl])\(\[([^\]]*)\]")

try:
    import rtde_control
    import rtde_receive
    import rtde_io

    HAS_RTDE = True
except ImportError:
    HAS_RTDE = False
    logger.warning("ur-rtde not installed. Install with: pip install ur-rtde")


class UR10eDriver:
    """
    Direct control interface for UR10e robot via RTDE protocol.
    """

    # Family + capability flags consumed by the family-agnostic stack
    # (CartesianServo, executor_ik, executor_grasp). SUPPORTS_URSCRIPT
    # routes CartesianServo through speedl via _send_script; GRIPPER_TYPE
    # selects the Robotiq fast path in executor_grasp.
    robot_family = "ur10e"
    SUPPORTS_URSCRIPT = True
    SUPPORTS_VELOCITY_STREAMING = True
    GRIPPER_TYPE = "robotiq_2f85"

    # UR10e joint limits (radians)
    JOINT_LIMITS = [(-2 * np.pi, 2 * np.pi)] * 6

    # Default motion parameters
    DEFAULT_VELOCITY = 0.25  # m/s for linear moves
    DEFAULT_ACCELERATION = 0.5  # m/s^2
    DEFAULT_JOINT_VEL = 1.05  # rad/s
    DEFAULT_JOINT_ACC = 1.4  # rad/s^2

    # Robotiq 2F-85 gripper parameters
    GRIPPER_OPEN = 0.0
    GRIPPER_CLOSED = 255.0
    GRIPPER_SPEED = 255
    GRIPPER_FORCE = 50

    # Robotiq 2F-85 max jaw opening (meters). Used to map a target grasp
    # width in meters to the 0-100 percent-closed URScript command.
    GRIPPER_MAX_WIDTH_M = 0.085

    # Primary-interface port for raw URScript (speedl/speedj/movej).
    URSCRIPT_PORT = 30002
    # Brake. The stop that works on this rig is a URScript stopj/stopl sent to
    # URSCRIPT_PORT: a program sent there REPLACES the running program, so it
    # brakes a fire-and-forget movej. rtde_c.stopJ cannot (see stop()).
    # The decel is derived from the LAST COMMANDED speed, never hardcoded:
    # stopping distance is v**2/(2a), so holding the overshoot at a budget
    # means a = v**2/(2*budget). A raised velocity cap therefore raises the
    # brake with it instead of silently lengthening the stopping distance.
    BRAKE_OVERSHOOT_BUDGET_RAD = 0.15  # leading joint, rad
    BRAKE_DECEL_MIN = 2.0              # rad/s^2, the historical stopj(2.0)
    BRAKE_DECEL_MAX = 8.0              # above this UR trips C153/C173
    BRAKE_OVERSHOOT_BUDGET_M = 0.05    # TCP, m
    BRAKE_DECEL_LIN_MIN = 1.2          # m/s^2
    BRAKE_DECEL_LIN_MAX = 5.0
    # Bounded wait for the Dashboard escalation. DashboardClient.connect() can
    # block on an unreachable/locked controller, and a stop must not hang.
    DASHBOARD_TIMEOUT_S = 2.0

    # Common configurations
    # Home: TCP xyz=(-0.8144, 0.1089, 0.1304) rot=(-2.2836, -2.1399, 0.0231).
    # Matches the human-teleop collector (vla_interp_pi05/ur10e_pi05_deploy
    # home_joints) so demos start from the same pose as the teleop corpus; a
    # home 170 deg off at wrist 3 rolls the wrist camera a half turn and
    # flips every recorded wrist image against that corpus.
    HOME_CONFIG = [3.2551, -1.7610, -1.9082, 5.2296, 1.5712, -2.8697]

    def __init__(self, robot_ip: str, frequency: float = 500.0):
        """
        Args:
            robot_ip: IP address of the UR10e controller
            frequency: RTDE communication frequency (Hz)
        """
        if not HAS_RTDE:
            raise RuntimeError("ur-rtde not installed. pip install ur-rtde")

        self.robot_ip = robot_ip
        self.frequency = frequency
        self._rtde_c: Optional[rtde_control.RTDEControlInterface] = None
        self._rtde_r: Optional[rtde_receive.RTDEReceiveInterface] = None
        self._rtde_io: Optional[rtde_io.RTDEIOInterface] = None
        self._urscript_socket: Optional[socket.socket] = None
        self._connected = False
        # Serializes writes on the single URScript socket. _send_script is
        # called concurrently from the executor thread, the primitive-timeout
        # watchdog's abort -> brake, and teleop /api/velocity; interleaved
        # partial writes on one socket corrupt programs.
        self._script_lock = threading.Lock()
        # True after a send failed even through the reconnect attempt.
        # Surfaced so callers/status can tell "command dropped" from "sent".
        self._urscript_down = False
        self._gripper_script_header = None
        self._gripper_initialized = False
        # Last commanded twist / gripper target, for the demo recorder's
        # action label. Stamped by send_velocity + the gripper entry points;
        # read (never written) by recording/action_source.py.
        self._command_latch = CommandLatch()
        # Monotonic deadline until which a motion URScript may still be
        # starting or stopping. See _note_motion_script / _motion_in_flight.
        self._motion_lease_until = -1.0
        # Last commanded speed and its space, for sizing the brake. See brake().
        self._last_motion_kind: Optional[str] = None  # "joint" | "linear"
        self._last_joint_vel = 0.0   # rad/s, leading joint
        self._last_linear_vel = 0.0  # m/s, TCP

    @property
    def connected(self) -> bool:
        return self._connected

    # Motion lease (see _publish_gripper_state for why this exists)

    # A speedl is only valid for its own duration; the servo re-sends every
    # ~8 ms, so a short lease self-expires within one tick of the loop ending.
    _SPEEDL_LEASE_S = 0.15
    # movej/movel are fire-and-forget with a ~1 s upload/start lag during which
    # the arm is stationary, so the measured-speed guard cannot see them.
    _MOVEJ_LEASE_S = 1.5

    def _note_motion_script(self, script: str) -> None:
        """Extend the motion lease when ``script`` commands arm motion.

        Also records the commanded speed and its space, which is what brake()
        sizes its deceleration from.
        """
        if "speedl(" in script or "speedj(" in script or "servoj(" in script:
            lease = self._SPEEDL_LEASE_S
        elif "movej(" in script or "movel(" in script:
            lease = self._MOVEJ_LEASE_S
        else:
            return
        self._note_commanded_speed(script)
        self._motion_lease_until = max(
            self._motion_lease_until, time.monotonic() + lease
        )

    def _note_commanded_speed(self, script: str) -> None:
        """Remember how fast, and in which space, the arm was last told to go."""
        for m in _SPEED_VEC_RE.finditer(script):
            try:
                vec = [float(x) for x in m.group(2).split(",")]
            except ValueError:
                continue
            if m.group(1) == "l":
                self._last_motion_kind = "linear"
                self._last_linear_vel = float(np.linalg.norm(vec[:3]))
            else:
                self._last_motion_kind = "joint"
                self._last_joint_vel = max(abs(v) for v in vec) if vec else 0.0
        vels = [float(v) for v in _MOVE_V_RE.findall(script)]
        if vels:
            # movej v is rad/s on the leading joint; movel v is m/s at the TCP.
            if "movel(" in script:
                self._last_motion_kind = "linear"
                self._last_linear_vel = max(vels)
            else:
                self._last_motion_kind = "joint"
                self._last_joint_vel = max(vels)

    def clear_motion_lease(self) -> None:
        """
        Positive confirmation that the last commanded motion has landed.

        The lease is a pessimistic backstop sized for the worst-case upload
        lag, so a move that arrives sooner would otherwise keep suppressing
        gripper publishes and hand the next stationary read a stale register.
        Callers that can prove arrival (executor_motion._wait_for_motion, on a
        joint-target match or Cartesian proximity) retire it early.
        """
        self._motion_lease_until = -1.0

    def _motion_in_flight(self) -> bool:
        """
        True while an arm motion may still be starting, running or stopping.

        Two independent signals, OR-ed, because neither alone is sufficient:
        the measured TCP speed misses the URScript start lag and the final
        millimetres of a servo convergence, while the lease misses a movej
        that outlives its lease.
        """
        return time.monotonic() < self._motion_lease_until or self._tcp_is_moving()

    def _tcp_is_moving(self) -> bool:
        """Measured TCP speed above the standstill threshold."""
        try:
            if self._rtde_r is not None:
                spd = np.asarray(self._rtde_r.getActualTCPSpeed(), dtype=float)
                if spd.size >= 3 and float(np.linalg.norm(spd[:3])) > 0.01:
                    return True
        except Exception:  # noqa: BLE001 - a read failure must not block motion
            pass
        return False

    # Command latch (see control/command_latch.py)

    def latch_command_velocity(
        self, velocity: Sequence[float], source: Optional[str] = None
    ) -> None:
        """
        Stamp a commanded base-frame twist without sending anything.

        Exists for the one caller that formats its own ``speedl`` URScript
        (``CartesianServo`` on the URScript branch) so the latch still has a
        single implementation even though there are two emit paths.
        """
        self._command_latch.latch_velocity(velocity, source)

    def get_last_command(self) -> CommandSnapshot:
        """Most recent commanded velocity / gripper, with monotonic stamps."""
        return self._command_latch.snapshot()

    def connect(self):
        """
        Establish RTDE connection to robot.

        Runs a best-effort Dashboard preflight (port 29999) first so a fresh
        connect recovers WITHOUT a trip to the teach pendant: it closes
        blocking popups, unlocks a protective stop, and powers on / releases
        brakes if the arm is idle. The one thing it cannot override is Local
        mode (a physical safety choice); it warns clearly in that case.
        The RTDEControl construction is then retried, because the usual cause
        of "Failed to start control script" is a half-open control script
        left by a previous client's unclean exit, which a dashboard stop +
        retry clears.
        """
        logger.info("Connecting to %s...", self.robot_ip)
        self._dashboard_preflight()
        self._rtde_c = self._connect_control_with_retry()
        # getTCPOffset() fails once a _send_script has replaced the control
        # script (first move), so cache the static pendant TCP offset now. The
        # pyroki IK path (ur10e_ik_pyroki.solve_ik_rtde) needs it to map a
        # gripper-TIP target into the tool0 frame; without it the UR10e falls
        # back to the legacy Cartesian path.
        try:
            self._tcp_offset_cache = np.array(
                self._rtde_c.getTCPOffset(), dtype=float
            )
            logger.info("Cached pendant TCP offset: %s", self._tcp_offset_cache.round(4))
        except Exception as e:
            logger.warning(
                "getTCPOffset() at connect failed (%s); pyroki IK will use the "
                "legacy fallback until this is available.", e,
            )
        self._rtde_r = rtde_receive.RTDEReceiveInterface(self.robot_ip)
        # RTDEIOInterface is best-effort: it reserves RTDE input registers and
        # raises "input registers already in use" when a fieldbus (EtherNet/IP,
        # PROFINET, MODBUS) is enabled. It is currently UNUSED (the Robotiq
        # gripper is driven via URScript, not RTDE IO), so a failure here must
        # not abort the connection. Construct opportunistically, None on failure.
        try:
            self._rtde_io = rtde_io.RTDEIOInterface(self.robot_ip)
        except RuntimeError as e:
            self._rtde_io = None
            logger.warning(
                "RTDEIOInterface unavailable (%s); continuing without RTDE IO "
                "(unused; gripper goes through URScript).", e,
            )
        self._open_urscript_socket()
        self._connected = True
        self._load_gripper_functions()
        # The Robotiq 2F-85 deactivates on every controller power-cycle and then
        # silently ignores move commands until re-activated, so activate on
        # connect (the URCap auto-activation this URScript path bypasses).
        # rq_activate_and_wait self-guards, so it is a no-op when already active.
        try:
            self.activate_gripper()
        except Exception as e:  # noqa: BLE001
            logger.warning(
                "Gripper activation failed on connect (%s); gripper moves will "
                "be ignored until it is activated.", e,
            )
        logger.info("Connected. Mode: %s", self.get_robot_mode())

    def _dashboard(self):
        """Open a connected DashboardClient, or None if unavailable."""
        try:
            import dashboard_client
        except ImportError:
            return None
        try:
            d = dashboard_client.DashboardClient(self.robot_ip)
            d.connect()
            return d
        except Exception as e:  # noqa: BLE001
            logger.warning("Dashboard (29999) connect failed: %s", e)
            return None

    def _dashboard_preflight(self):
        """Clear states that block RTDE control-script upload. No-op when the
        dashboard is unreachable. Requires Remote Control mode (one-time
        pendant toggle); everything else is recovered programmatically."""
        d = self._dashboard()
        if d is None:
            return
        try:
            if not d.isInRemoteControl():
                logger.warning(
                    "UR is in LOCAL mode; RTDE cannot start the control "
                    "script. Flip the pendant top-right menu to Remote "
                    "Control (one-time); after that, power/brake/protective "
                    "stop are all recovered from here without the pendant."
                )
                return
            for fn in ("closeSafetyPopup", "closePopup"):
                try:
                    getattr(d, fn)()
                except Exception:  # noqa: BLE001
                    pass
            safety = d.safetystatus()
            mode = d.robotmode()
            if "PROTECTIVE_STOP" in safety:
                logger.info("Protective stop -> unlocking via dashboard")
                try:
                    d.unlockProtectiveStop()
                    # UR enforces a ~5s settle after unlock before motion (or a
                    # control-script upload) is accepted. Waiting less is the
                    # usual cause of RTDEControl's "failed to start control
                    # script" right after a protective stop.
                    time.sleep(5.0)
                except Exception as e:  # noqa: BLE001
                    logger.warning("unlockProtectiveStop failed: %s", e)
            if "FAULT" in safety or "VIOLATION" in safety:
                logger.info("Safety fault -> restartSafety via dashboard")
                try:
                    d.restartSafety()
                    time.sleep(2.0)
                except Exception:  # noqa: BLE001
                    pass
            if any(s in mode for s in ("POWER_OFF", "IDLE", "BOOTING")):
                logger.info("Arm not running -> powerOn + brakeRelease")
                try:
                    d.powerOn()
                    time.sleep(0.5)
                    d.brakeRelease()
                    time.sleep(3.0)
                except Exception as e:  # noqa: BLE001
                    logger.warning("powerOn/brakeRelease failed: %s", e)
        finally:
            try:
                d.disconnect()
            except Exception:  # noqa: BLE001
                pass

    def _connect_control_with_retry(self, attempts: int = 3):
        """RTDEControlInterface with retry. A stale control script from a
        prior unclean exit is the common cause of the 5 s start timeout;
        a dashboard stop between tries clears it."""
        last = None
        for i in range(attempts):
            try:
                return rtde_control.RTDEControlInterface(
                    self.robot_ip, self.frequency
                )
            except RuntimeError as e:
                last = e
                logger.warning(
                    "RTDEControl start failed (attempt %d/%d): %s",
                    i + 1, attempts, e,
                )
                d = self._dashboard()
                if d is not None:
                    for fn in ("stop", "closeSafetyPopup", "closePopup"):
                        try:
                            getattr(d, fn)()
                        except Exception:  # noqa: BLE001
                            pass
                    try:
                        d.disconnect()
                    except Exception:  # noqa: BLE001
                        pass
                time.sleep(1.5)
        raise last

    def unlock_protective_stop(self) -> bool:
        """Dashboard-unlock a protective stop mid-session (no pendant).
        Returns True if the unlock command was issued. Note the UR enforces
        a ~5 s settle after unlock before motion resumes."""
        d = self._dashboard()
        if d is None:
            return False
        try:
            d.unlockProtectiveStop()
            for fn in ("closeSafetyPopup", "closePopup"):
                try:
                    getattr(d, fn)()
                except Exception:  # noqa: BLE001
                    pass
            return True
        except Exception as e:  # noqa: BLE001
            logger.warning("unlock_protective_stop failed: %s", e)
            return False
        finally:
            try:
                d.disconnect()
            except Exception:  # noqa: BLE001
                pass

    def _open_urscript_socket(self):
        """
        Open the primary-interface TCP socket for raw URScript.

        speedl/speedj can't go through ur_rtde's blocking moveL/servoJ, so
        CartesianServo + VLA velocity streaming push raw URScript to port
        30002, exactly as the external teleop UR10eRobot did. A failure
        here degrades to RTDE-only motion (move_linear/servo_joint) rather
        than crashing the bring-up; SUPPORTS_URSCRIPT stays True so the
        servo path is still attempted and reconnects lazily in _send_script.
        """
        try:
            sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
            sock.settimeout(5)
            sock.connect((self.robot_ip, self.URSCRIPT_PORT))
            self._urscript_socket = sock
            logger.info("URScript socket up (port %d)", self.URSCRIPT_PORT)
        except OSError as e:
            self._urscript_socket = None
            logger.warning(
                "URScript socket (port %d) failed: %s; speedl/speedj will "
                "retry-connect on first use",
                self.URSCRIPT_PORT,
                e,
            )

    def _drop_urscript_socket(self):
        """
        Close and forget the URScript socket, holding ``_script_lock``.

        NEVER close this socket without the lock. _send_script resolves
        self._urscript_socket and then calls sendall() on it, both inside
        _script_lock; a closer that skips the lock can free the descriptor
        between those two steps:

            executor thread                 /api/abort (event loop)
            ---------------                 ------------------------
            with _script_lock:
              sock = self._urscript_socket
                                            sock.close()          # fd N freed
                                            self._urscript_socket = None
                                            (another thread open()s -> gets N)
              sock.sendall(script)          # writes URScript into fd N,
                                            # which is now someone else's
                                            # socket / log file / device node

        The fd is freed the moment close() returns and the kernel reissues
        the lowest free number, so this is not merely "the send fails": it
        is a cross-descriptor write of a multi-KB URScript program into an
        unrelated open file. /api/stop and /api/abort are async handlers on
        the event loop and run concurrently with the executor thread, so
        both closers below were reachable during any movel/movej.

        Must not be called from inside _send_script, which already holds
        the lock (it is a plain Lock, not an RLock).
        """
        with self._script_lock:
            sock = self._urscript_socket
            self._urscript_socket = None
            if sock is not None:
                try:
                    sock.close()
                except OSError:
                    pass

    def _load_gripper_functions(self):
        """
        Load Robotiq gripper URScript function defs from grippy.script.
        """
        search_paths = [Path(__file__).parent.parent / "robots/ur10e/grippy.script"]
        override = os.environ.get("SPARK_GRIPPER_SCRIPT")
        if override:
            search_paths.append(Path(override))
        for script_path in search_paths:
            if not script_path.exists():
                continue
            try:
                full_script = script_path.read_text(encoding="utf-8")
                lines = full_script.split("\n")
                gripper_start = None
                gripper_end = None
                for i, line in enumerate(lines):
                    if "#   Type: Gripper" in line:
                        for j in range(i, max(0, i - 10), -1):
                            if (
                                "# begin: URCap Installation Node" in lines[j]
                                and "#   Source: Robotiq_Grippers" in lines[j + 1]
                            ):
                                gripper_start = j
                                break
                    if (
                        gripper_start
                        and "# end: URCap Installation Node" in line
                        and i > gripper_start + 10
                    ):
                        gripper_end = i + 1
                        break
                if gripper_start and gripper_end:
                    self._gripper_script_header = "\n".join(
                        lines[gripper_start:gripper_end]
                    )
                    logger.info("Loaded Robotiq gripper functions")
                    return
            except Exception as e:
                logger.info("Gripper functions not loaded from %s: %s", script_path, e)
        logger.warning("grippy.script not found; gripper may not work correctly")

    def disconnect(self):
        """
        Safely disconnect from robot.

        Brakes FIRST. Closing the URScript socket does not stop a program that
        is already running on the controller, so a teardown mid-move (server
        Ctrl-C -> pipeline.shutdown -> disconnect) used to leave the arm
        running to its target with nobody left to stop it -- the operator's
        "Ctrl-C made the arm just go high".
        """
        if self._connected:
            self.brake()
        # Clear _connected FIRST. A demo-recorder or streaming thread polling
        # get_joint_positions()/get_observation() during teardown must get the
        # "Not connected to robot" RuntimeError, not reach rtde_r.getActualQ()
        # on an interface rtde_r.disconnect() is tearing down: ur_rtde reads
        # its state buffer through a pointer disconnect() drops, and that read
        # can segfault the process rather than raise.
        self._connected = False
        if self._rtde_c:
            self._rtde_c.stopScript()
            self._rtde_c.disconnect()
        if self._rtde_r:
            self._rtde_r.disconnect()
        # Under _script_lock: a concurrent _send_script may have already
        # resolved this socket and be about to sendall() on it.
        self._drop_urscript_socket()
        logger.info("Disconnected")

    def _check_connected(self):
        if not self._connected:
            raise RuntimeError("Not connected to robot. Call connect() first.")

    def health_check(self) -> bool:
        """
        Probe the ACTUAL RTDE link, not the _connected flag.

        Checks rtde_r.isConnected() plus a getActualQ read; on failure clears
        _connected so _check_connected reflects reality. reconnect() (or
        /api/connect_robot) is the recovery.
        """
        if not self._connected:
            return False
        try:
            rr = self._rtde_r
            if rr is None:
                raise RuntimeError("no RTDE receive interface")
            is_conn = getattr(rr, "isConnected", None)
            if callable(is_conn) and not is_conn():
                raise RuntimeError("RTDE receive link reports disconnected")
            q = rr.getActualQ()
            if q is None or len(q) < 6:
                raise RuntimeError("joint state unavailable")
            return True
        except Exception as exc:  # noqa: BLE001
            logger.error("Driver health check FAILED: %s (marking disconnected)", exc)
            self._connected = False
            return False

    def reconnect(self) -> bool:
        """
        Tear down and re-run the full connect sequence (dashboard preflight,
        RTDE control retry, URScript socket, gripper). Recovery path after
        health_check() fails, so /api/connect_robot is not the only way back.
        """
        logger.info("Driver reconnect requested")
        try:
            self.disconnect()
        except Exception as exc:  # noqa: BLE001
            logger.warning("reconnect: disconnect raised: %s", exc)
        self._rtde_c = None
        self._rtde_r = None
        self._urscript_socket = None
        self._connected = False
        self.connect()
        return self._connected

    #State queries

    def get_joint_positions(self) -> np.ndarray:
        """
        Get current joint positions (radians).
        """
        self._check_connected()
        return np.array(self._rtde_r.getActualQ())

    def get_joint_velocities(self) -> np.ndarray:
        """
        Get current joint velocities (rad/s).
        """
        self._check_connected()
        return np.array(self._rtde_r.getActualQd())

    def get_tcp_pose(self) -> np.ndarray:
        """
        Get current TCP pose [x, y, z, rx, ry, rz] (axis-angle).
        """
        self._check_connected()
        return np.array(self._rtde_r.getActualTCPPose())

    def get_tcp_offset(self) -> np.ndarray:
        """
        Pendant-configured TCP offset [x, y, z, rx, ry, rz] (gripper tip
        relative to tool0, tool0 frame). Static, so cached after first read.
        Used by the pyroki IK path (ur10e_ik_pyroki.solve_ik_rtde) to convert
        a gripper-TIP target into the tool0 frame pyroki solves for.
        """
        off = getattr(self, "_tcp_offset_cache", None)
        if off is None:
            self._check_connected()
            off = np.array(self._rtde_c.getTCPOffset(), dtype=float)
            self._tcp_offset_cache = off
        return off

    def get_tcp_force(self) -> np.ndarray:
        """
        Get TCP force/torque [fx, fy, fz, tx, ty, tz].
        """
        self._check_connected()
        return np.array(self._rtde_r.getActualTCPForce())

    def get_robot_mode(self) -> int:
        """
        Get robot mode (7=running, 5=idle, etc.).
        """
        self._check_connected()
        return self._rtde_r.getRobotMode()

    def is_steady(self) -> bool:
        """
        Check if robot has stopped moving.
        """
        self._check_connected()
        return self._rtde_r.isSteady()

    #Motion commands

    def move_to_joint_config(
        self,
        q: list,
        velocity: float = None,
        acceleration: float = None,
        asynchronous: bool = False,
    ):
        """
        Move to joint configuration.

        Args:
            q: Target joint positions [j0..j5] in radians
            velocity: Joint velocity (rad/s)
            acceleration: Joint acceleration (rad/s^2)
            asynchronous: If True, return immediately
        """
        self._check_connected()
        vel = velocity or self.DEFAULT_JOINT_VEL
        acc = acceleration or self.DEFAULT_JOINT_ACC
        self._ensure_control_script("moveJ")
        self._rtde_c.moveJ(list(q), vel, acc, asynchronous)

    def move_to_joint_config_urscript(
        self, q: list, velocity: float = None, acceleration: float = None
    ):
        """Joint move via raw URScript on the primary interface (port 30002),
        the SAME channel as speedl/speedj.

        Use this after velocity streaming: a speedl program stops the control
        script that RTDEControlInterface uploaded, so a subsequent rtde_c.moveJ
        silently no-ops. This routes movej through the URScript socket instead,
        which is conflict-free with speedl. Fire-and-forget: returns
        immediately, it does NOT block until the move completes.
        """
        vel = velocity or self.DEFAULT_JOINT_VEL
        acc = acceleration or self.DEFAULT_JOINT_ACC
        q_str = ", ".join(f"{float(x):.6f}" for x in q)
        if not self._send_script(f"movej([{q_str}], a={acc}, v={vel})"):
            # A dropped motion command must FAIL the primitive, not silently
            # pass while the caller sleeps and proceeds as if the arm moved.
            raise RuntimeError(
                "movej dropped: URScript channel down (see health_check/reconnect)"
            )

    def go_home_urscript(self, velocity: float = None):
        """Home via the URScript channel (see move_to_joint_config_urscript).
        Safe to call from velocity-streaming teleop where go_home() no-ops."""
        self.move_to_joint_config_urscript(
            self.HOME_CONFIG, velocity=velocity or 0.6, acceleration=1.0
        )

    def _check_workspace(self, pose: list):
        """
        Driver-level sanity envelope for the UR10e cell: a coarse backstop
        against wild targets only, since the ScoreExecutor already clips every
        move to the precise control.workspace_min/max ([-1.1,-0.5] x, [-0.5,0.7]
        y, [-0.27,0.50] z). The UR10e grasps at NEGATIVE z (table ~ -0.25 in
        base frame) and reaches X to -1.1, so the envelope spans the real
        reachable range (with margin).
        """
        x, y, z = pose[0], pose[1], pose[2]
        if not (-1.2 <= x <= 0.9 and -0.9 <= y <= 0.9 and -0.35 <= z <= 1.2):
            raise ValueError(
                f"Target ({x:.3f}, {y:.3f}, {z:.3f}) outside workspace bounds. "
                f"x:[-1.2,0.9] y:[-0.9,0.9] z:[-0.35,1.2]"
            )

    def move_linear(
        self,
        pose: list,
        velocity: float = None,
        acceleration: float = None,
        asynchronous: bool = False,
    ):
        """
        Linear motion to Cartesian pose.

        Args:
            pose: Target [x, y, z, rx, ry, rz] in meters and axis-angle
            velocity: Linear velocity (m/s)
            acceleration: Linear acceleration (m/s^2)
            asynchronous: If True, return immediately
        """
        self._check_connected()
        self._check_workspace(pose)
        vel = velocity or self.DEFAULT_VELOCITY
        acc = acceleration or self.DEFAULT_ACCELERATION
        self._ensure_control_script("moveL")
        self._rtde_c.moveL(list(pose), vel, acc, asynchronous)

    SAFETY_MODE_NAMES = {
        1: "NORMAL", 2: "REDUCED", 3: "PROTECTIVE_STOP", 4: "RECOVERY",
        5: "SAFEGUARD_STOP", 6: "SYSTEM_EMERGENCY_STOP",
        7: "ROBOT_EMERGENCY_STOP", 8: "VIOLATION", 9: "FAULT",
    }

    def _check_safety_and_recover(self, what: str):
        """Make protective stops VISIBLE and self-clearing.

        Measured 2026-08-21 12:39: a wrist yaw at low height collided, the
        controller went into PROTECTIVE_STOP, and the server's log showed
        nothing but moved=False timeouts -- zero occurrences of the word
        'protective' across the whole session, while the operator cleared the
        pendant three times. (--auto-unlock never applied here: it is
        Franka-only.)

        On PROTECTIVE_STOP: wait the 5 s the UR firmware requires, unlock via
        the dashboard, re-upload the control script -- and then RAISE anyway.
        A cleared protective stop means the pending motion's predecessor
        COLLIDED; silently continuing with the same target would re-run the
        collision (which is exactly what the operator watched happen when
        they cleared it by hand). Raising drops this motion into the
        executor's failure path, whose recovery ladder begins with a lift --
        the correct next move after any collision. Emergency stops are never
        auto-cleared: a physical stop is an operator decision.
        """
        try:
            mode = int(self._rtde_r.getSafetyMode())
        except Exception:
            return  # can't read; let the motion surface its own error
        if mode in (1, 2):
            return
        name = self.SAFETY_MODE_NAMES.get(mode, str(mode))
        if mode in (6, 7):
            raise RuntimeError(
                f"robot is in {name}; refusing {what} -- emergency stops "
                "must be cleared by the operator"
            )
        logger.warning("robot is in %s before %s; attempting recovery", name, what)
        if mode == 3:
            time.sleep(5.0)  # UR firmware refuses unlock before 5 s
            d = self._dashboard()
            if d is not None:
                try:
                    d.unlockProtectiveStop()
                    logger.warning("protective stop UNLOCKED via dashboard")
                except Exception as exc:  # noqa: BLE001
                    logger.warning("unlockProtectiveStop failed: %s", exc)
                finally:
                    try:
                        d.disconnect()
                    except Exception:  # noqa: BLE001
                        pass
            try:
                self._rtde_c.reuploadScript()
                time.sleep(0.2)
            except Exception as exc:  # noqa: BLE001
                logger.warning("script re-upload after unlock failed: %s", exc)
            # CONTINUE, do not raise. The colliding motion is not the one
            # waiting behind this check -- it already died with moved=False
            # and routed to the failure path before the arm ever got here;
            # the pending command is recovery's own next move (typically a
            # lift). This mirrors the teleop /api/recover behavior the
            # operator relies on: unlock and keep driving. Only a stop that
            # WON'T clear escalates.
            try:
                if int(self._rtde_r.getSafetyMode()) in (1, 2):
                    # Stamp the recovery so the executor can reconcile its
                    # world model at the next action boundary: a protective
                    # stop means a COLLISION happened, and the collision may
                    # have changed things the plan believes (the object may
                    # have been knocked from the jaws, the target nudged).
                    self.last_protective_recovery_ts = time.time()
                    logger.warning(
                        "protective stop cleared; continuing with %s", what
                    )
                    return
            except Exception:  # noqa: BLE001
                pass
        raise RuntimeError(
            f"robot is in {name} and did not recover before {what}; "
            "operator attention needed"
        )

    def _ensure_control_script(self, what: str):
        """Re-upload the RTDE control script if it has died.

        After a servoj session ends (servo_stop / speed_stop), ur_rtde's
        control script on the controller can stop and stay stopped; every
        subsequent moveL is then a silent no-op that returns immediately.
        """
        self._check_safety_and_recover(what)
        try:
            if self._rtde_c.isProgramRunning():
                return
        except Exception:
            return  # can't ask; let the motion call surface its own error
        logger.warning(
            "RTDE control script not running before %s; re-uploading", what
        )
        try:
            ok = self._rtde_c.reuploadScript()
            time.sleep(0.1)  # give the controller a beat to start it
            if not ok or not self._rtde_c.isProgramRunning():
                raise RuntimeError("reuploadScript did not take")
        except Exception as exc:
            # Reconnect is the heavier fallback, the same path a fresh connect
            # uses, so a controller that can run at all will accept it.
            logger.warning(
                "reuploadScript failed (%s); reconnecting RTDE control", exc
            )
            self._rtde_c.reconnect()
            if not self._rtde_c.isProgramRunning():
                raise RuntimeError(
                    f"RTDE control script would not restart before {what}; "
                    "refusing to silently drop the motion"
                )

    def move_linear_relative(
        self, delta: list, velocity: float = None, acceleration: float = None
    ):
        """
        Move relative to current TCP pose.

        Args:
            delta: [dx, dy, dz, drx, dry, drz] relative offset
        """
        current = self.get_tcp_pose()
        target = current + np.array(delta)
        self.move_linear(target.tolist(), velocity, acceleration)

    #Raw URScript + velocity streaming
    # speedl/speedj go straight to the primary interface (port 30002), not
    # through ur_rtde's blocking moveL/servoJ. CartesianServo gates on
    # SUPPORTS_URSCRIPT + _send_script and emits speedl(...) at high rate,
    # which URScript speedl tolerates (the FR3 path is capped at 30 Hz).

    def _send_script(self, script: str) -> bool:
        """
        Send a raw URScript line to the primary interface (port 30002).

        Auto-reconnects once on a broken pipe. Returns True on success,
        False if the socket can't be (re)established.
        """
        if not self._connected:
            return False
        if not script.endswith("\n"):
            script += "\n"
        # rtde_c keeps a control-script keepalive running that re-stomps a
        # one-shot movej/movel sent over _send_script before the arm moves (arm
        # silently never moves; /api/home no-ops cold). A gripper _send_script
        # stops the keepalive, which is why a move works right after a grasp but
        # not from a fresh state. All motion here goes through _send_script (see
        # executor_ik._movej_to_pose: rtde_c.moveL no-ops), so release the
        # keepalive right before a one-shot move. speedl/speedj are EXCLUDED: the
        # 30 Hz servo re-sends faster than the stomp and must not eat a
        # stopScript() per tick. SPARK_UR_RELEASE_KEEPALIVE=0 restores legacy.
        if (
            self._rtde_c is not None
            and ("movej(" in script or "movel(" in script)
            and os.environ.get("SPARK_UR_RELEASE_KEEPALIVE", "1") == "1"
        ):
            try:
                self._rtde_c.stopScript()
            except Exception as e:  # noqa: BLE001
                logger.debug("rtde_c.stopScript() before move failed: %s", e)
        # One writer at a time on the socket: concurrent sends (executor +
        # watchdog brake + teleop) interleaving on one TCP stream corrupt
        # programs mid-line.
        with self._script_lock:
            for attempt in range(2):
                try:
                    if self._urscript_socket is None:
                        self._open_urscript_socket()
                        if self._urscript_socket is None:
                            self._urscript_down = True
                            logger.error(
                                "URScript DOWN: could not open socket; "
                                "command dropped: %s", script.split("\n", 1)[0]
                            )
                            return False
                    # sendall, not send: gripper scripts embed the multi-KB
                    # URCap header, and an unchecked partial send truncates a
                    # program mid-line (the next send then concatenates
                    # garbage onto it).
                    self._urscript_socket.sendall(script.encode())
                    # Stamp AFTER a successful send: a script that never left
                    # the socket must not suppress the next gripper publish.
                    self._note_motion_script(script)
                    self._urscript_down = False
                    return True
                except (BrokenPipeError, ConnectionResetError, OSError) as e:
                    if attempt == 0:
                        logger.warning("URScript socket lost, reconnecting (%s)", e)
                        try:
                            if self._urscript_socket is not None:
                                self._urscript_socket.close()
                        except OSError:
                            pass
                        self._urscript_socket = None
                    else:
                        self._urscript_down = True
                        logger.error(
                            "URScript DOWN: send failed after reconnect (%s); "
                            "command dropped: %s", e, script.split("\n", 1)[0]
                        )
                        return False
            return False

    def send_velocity(
        self,
        velocity: Sequence[float],
        acceleration: float = 0.5,
        duration: float = 0.1,
    ):
        """
        Stream a base-frame Cartesian velocity via URScript speedl.

        Args:
            velocity: [vx, vy, vz, wrx, wry, wrz] in m/s and rad/s
            acceleration: linear acceleration limit (m/s^2)
            duration: how long the command stays active (seconds)

        CartesianServo calls this with keyword ``acceleration``/``duration``
        on the non-URScript fallback branch; here the URScript branch is
        used directly, but we keep the same signature so the servo's
        TypeError fallback never has to fire on this driver.
        """
        v = [float(x) for x in velocity]
        # Latch BEFORE emitting: the recorder's staleness gate is keyed on the
        # commit time of this value.
        self._command_latch.latch_velocity(v)
        vel_str = ", ".join(f"{x:.5f}" for x in v)
        self._send_script(f"speedl([{vel_str}], a={acceleration}, t={duration})")

    def stop_velocity(self):
        """
        Decelerate any active speedl/speedj to zero. True == on the wire.

        Named for the velocity-streaming path, but several abort handlers reach
        for it generically (/api/abort, /api/recover), so it brakes in whatever
        space the arm was last commanded in: stopl after a speedl, stopj after
        a movej, where a Cartesian decel constraint would be the wrong one.
        """
        return self.brake(kind=self._last_motion_kind or "linear")

    # The brake

    def brake_decel(self, kind: Optional[str] = None) -> float:
        """Deceleration for a stop, sized from the last commanded speed.

        overshoot = v**2 / (2a), so a = v**2 / (2 * budget). Returned in
        rad/s^2 for a joint stop and m/s^2 for a linear one. Clamped: below
        BRAKE_DECEL_MIN a stop is not a stop, above BRAKE_DECEL_MAX the UR
        answers with a C153/C173 protective stop instead of braking.
        """
        kind = kind or self._last_motion_kind or "joint"
        if kind == "linear":
            v, budget = self._last_linear_vel, self.BRAKE_OVERSHOOT_BUDGET_M
            lo, hi = self.BRAKE_DECEL_LIN_MIN, self.BRAKE_DECEL_LIN_MAX
        else:
            v, budget = self._last_joint_vel, self.BRAKE_OVERSHOOT_BUDGET_RAD
            lo, hi = self.BRAKE_DECEL_MIN, self.BRAKE_DECEL_MAX
        return float(min(max(v * v / (2.0 * budget), lo), hi))

    def brake(self, decel: Optional[float] = None, kind: Optional[str] = None) -> bool:
        """Decelerate the arm NOW over the URScript socket. True == on the wire.

        This is the only mechanism that brakes this rig, and the abort path for
        blended paths already proved it on the wire: a program sent to
        URSCRIPT_PORT REPLACES the running program, so a stopj both brakes the
        arm and kills whatever movej/blended path was running. Fire-and-forget
        like every other motion here -- it returns as soon as the bytes are
        sent, the deceleration happens on the controller.

        Safe to call when disconnected or mid-abort: _send_script returns False
        and the caller escalates (see emergency_stop).
        """
        kind = kind or self._last_motion_kind or "joint"
        a = self.brake_decel(kind) if decel is None else float(decel)
        cmd = "stopl" if kind == "linear" else "stopj"
        ok = False
        try:
            ok = bool(self._send_script(f"{cmd}({a:.3f})"))
        except Exception as exc:  # noqa: BLE001 - a stop never raises
            logger.warning("brake: %s send failed: %s", cmd, exc)
        if ok:
            # The commanded motion is over: retire the lease so the next
            # gripper publish is not suppressed by a move that no longer runs.
            # The commanded-speed memory is deliberately NOT cleared: /api/stop
            # brakes twice (executor abort, then the driver), and a second
            # stopj REPLACES the first one's program; computed from a zeroed
            # velocity it would be the BRAKE_DECEL_MIN floor, i.e. a redundant
            # stop that decelerates more gently than the one already running.
            self.clear_motion_lease()
        else:
            logger.error("brake: %s(%.2f) DID NOT REACH THE ARM", cmd, a)
        return ok

    def emergency_stop(self, decel: Optional[float] = None) -> dict:
        """Brake the arm, escalating until something reaches it.

        1. URScript stopj/stopl on the primary socket.
        2. Same, on a freshly reopened socket (a dead socket is the one
           failure mode step 1 has).
        3. Dashboard ``stop`` on 29999, bounded by DASHBOARD_TIMEOUT_S. This
           ends the running program too, just slower to issue and vulnerable
           to another dashboard client holding the single slot.

        Returns {"braked": bool, "steps": [...]}: ``braked`` is True only if a
        stop actually left this process. Deliberately NOT raising: every caller
        is already on an abort path.
        """
        steps = []
        if self.brake(decel):
            return {"braked": True, "steps": ["urscript"]}
        steps.append("urscript:failed")
        # A stale socket is invisible until a send fails; drop it and retry.
        # Under _script_lock (see _drop_urscript_socket): /api/stop and
        # /api/abort reach here on the event loop while the executor thread
        # may be mid-_send_script on this very descriptor.
        self._drop_urscript_socket()
        if self.brake(decel):
            steps.append("urscript-reconnect")
            return {"braked": True, "steps": steps}
        steps.append("urscript-reconnect:failed")
        if self._send_dashboard_command("stop"):
            steps.append("dashboard")
            return {"braked": True, "steps": steps}
        steps.append("dashboard:failed")
        logger.critical(
            "EMERGENCY STOP DID NOT REACH THE ARM (%s). The arm may still be "
            "moving; use the physical E-stop.",
            ", ".join(steps),
        )
        return {"braked": False, "steps": steps}

    def _send_dashboard_command(self, command: str, timeout_s: float = None) -> bool:
        """Run one Dashboard (29999) command under a bounded wait.

        DashboardClient.connect() blocks on an unreachable or Local-mode
        controller, and an escalation that hangs is not an escalation, so the
        call runs on a daemon worker that is joined only for timeout_s.
        """
        timeout_s = self.DASHBOARD_TIMEOUT_S if timeout_s is None else timeout_s
        result = {"ok": False}

        def _run():
            d = self._dashboard()
            if d is None:
                return
            try:
                fn = getattr(d, command, None)
                if not callable(fn):
                    logger.warning("Dashboard has no command %r", command)
                    return
                fn()
                result["ok"] = True
            except Exception as exc:  # noqa: BLE001
                logger.warning("Dashboard %s failed: %s", command, exc)
            finally:
                try:
                    d.disconnect()
                except Exception:  # noqa: BLE001
                    pass

        worker = threading.Thread(
            target=_run, name="ur-dashboard-%s" % command, daemon=True
        )
        worker.start()
        worker.join(timeout_s)
        if worker.is_alive():
            logger.warning(
                "Dashboard %s did not answer in %.1fs", command, timeout_s
            )
            return False
        return bool(result["ok"])

    def get_observation(self) -> dict:
        """
        Proprioceptive observation dict used by the VLA controller path.

        Mirrors the external teleop UR10eRobot keys so a swapped-in VLA
        policy sees the same surface: joint positions/velocities, TCP pose
        (axis-angle 6-vec), and gripper position (0-255, 0=open).
        """
        self._check_connected()
        obs = {
            "joint_positions": self.get_joint_positions(),
            "joint_velocities": self.get_joint_velocities(),
            "tcp_pose": np.array(self._rtde_r.getActualTCPPose()),
            "tcp_velocity": np.array(self._rtde_r.getActualTCPSpeed()),
        }
        # PASSIVE read. get_observation is polled by the demo recorder thread
        # at record_hz WHILE the arm is moving, and a publishing read uploads a
        # URScript program that cancels the in-flight move. The recorder's own
        # docstring already says this channel is diagnostic and that the
        # training label comes from the command latch, so a register that is at
        # worst one gripper actuation stale costs nothing and a cancelled
        # trajectory costs the episode.
        try:
            obs["gripper_position"] = self.get_gripper_position(publish=False)
        except Exception:
            obs["gripper_position"] = 0.0
        return obs

    def servo_joint(
        self,
        q: list,
        velocity: float = 0.5,
        acceleration: float = 0.5,
        dt: float = 0.002,
        lookahead_time: float = 0.1,
        gain: int = 300,
    ):
        """
        Real-time joint servo (for VLA policy control loop).

        Args:
            q: Target joint positions
            velocity: Not used directly but defines profile
            acceleration: Not used directly but defines profile
            dt: Time step (1/frequency)
            lookahead_time: Smoothing parameter (0.03-0.2)
            gain: Proportional gain (100-2000)
        """
        self._check_connected()
        self._rtde_c.servoJ(list(q), velocity, acceleration, dt, lookahead_time, gain)

    def servo_stop(self):
        """
        Stop servo mode.
        """
        if self._rtde_c is not None:
            try:
                self._rtde_c.servoStop()
            except Exception as exc:  # noqa: BLE001
                logger.debug("rtde_c.servoStop failed: %s", exc)
        # servoStop is an RTDE register command read by the control script, so
        # it is a no-op once _send_script has stopped that script. Brake for
        # real on the way out, and report whether that reached the arm; a
        # caller polling stop methods must not read None as success.
        return self.brake()

    def stop(self):
        """
        Emergency stop: decelerate to zero velocity.

        This used to be ``rtde_c.stopJ(2.0)``, which CANNOT stop this rig.
        stopJ writes an RTDE input register that ur_rtde's control script
        polls, and every motion here goes through _send_script, which
        deliberately stops that control script (see _send_script). No control
        script, nobody reads the register: a silent no-op that satisfied every
        best-effort stop loop in the codebase while the arm ran on to its
        target. The brake is URScript on the primary socket; the arm actually
        decelerates. Kept as ``stop`` because SafeRobot.stop_motion and several
        route handlers look this name up by getattr.
        """
        return self.brake()

    def go_home(self, velocity: float = None):
        """
        Move to home configuration.
        """
        self.move_to_joint_config(self.HOME_CONFIG, velocity=velocity)

    #Gripper control (Robotiq 2F-85)
    # Requires Robotiq URCap installed on the UR controller.
    # The rq_* functions are URScript functions provided by the URCap.

    def activate_gripper(self):
        """
        Activate gripper (required once after power-on).
        """
        self._check_connected()
        # Sent over the primary-interface URScript socket (_send_script),
        # the same transport speedl uses, rather than RTDEControl's
        # sendCustomScriptFunction: this ur_rtde build only accepts the
        # 2-arg (name, script) form AND auto-wraps+indents the body into a
        # def, which double-wraps our already-self-invoking scripts. The
        # header is embedded so rq_* are defined even when the running
        # program has not loaded the Robotiq URCap; the script self-invokes.
        if self._gripper_script_header:
            # reset=True forces deactivate -> re-activate, which re-runs the
            # Robotiq auto-calibration (physically cycles the gripper to relearn
            # its full open/close stroke). Plain rq_activate_and_wait only
            # activates if needed and keeps the stale calibration that was
            # capping the close short of the mechanical stop.
            script = (
                "def rq_activate():\n"
                f"{self._gripper_script_header}\n"
                "  rq_activate_all_grippers(True)\n"
                "end\n"
                "rq_activate()\n"
            )
        else:
            script = "def rq_activate():\n  rq_activate_and_wait()\nend\nrq_activate()\n"
        if not self._send_script(script):
            raise RuntimeError("gripper activation dropped: URScript channel down")
        time.sleep(5.0)  # reset+re-activate cycles the gripper (~3-5s)
        logger.info("Gripper reset + activated (re-calibrated)")

    def open_gripper(self, speed: int = None, force: int = None):
        """
        Open the Robotiq 2F-85 gripper.
        """
        self._send_gripper_command(0.0, speed, force)

    def close_gripper(self, speed: int = None, force: int = None):
        """
        Fully close the Robotiq 2F-85: drive to the MECHANICAL stop (rPR=255).

        rq_move_and_wait_norm(100) stops short of the mechanical stop (rPR~226,
        a ~29/255 gap), so the FULL close uses rq_close_and_wait() to drive the
        raw position register to the hard stop. Intermediate widths keep the
        norm-scaled path (set_gripper_position), where the mapping is correct.
        """
        # Latched here as well as in _send_gripper_command: the script-header
        # branch below bypasses that method entirely.
        self._command_latch.latch_gripper(1.0)
        self._check_connected()
        spd = int(np.clip(speed if speed is not None else 100, 0, 100))
        frc = int(np.clip(force if force is not None else 100, 0, 100))
        if self._gripper_script_header:
            script = (
                "def gripper_close():\n"
                f"{self._gripper_script_header}\n"
                f'    rq_set_force_norm({frc}, "1")\n'
                f'    rq_set_speed_norm({spd}, "1")\n'
                '    rq_close_and_wait("1")\n'
                "end\n"
                "gripper_close()\n"
            )
            if not self._send_script(script):
                raise RuntimeError(
                    "gripper close dropped: URScript channel down"
                )
            time.sleep(0.5)
        else:
            self._send_gripper_command(1.0, speed, force)

    def set_gripper_position(
        self, position: float, speed: int = None, force: int = None
    ):
        """
        Set gripper to specific position.

        Args:
            position: 0.0 (fully open) to 1.0 (fully closed)
            speed: 0-100 (default: 100)
            force: 0-100 (default: 50)
        """
        self._send_gripper_command(position, speed, force)

    def _send_gripper_command(
        self, position: float, speed: int = None, force: int = None
    ):
        """
        Send command to Robotiq gripper via URScript.

        Args:
            position: 0.0 (open) to 1.0 (closed)
            speed: 0-100 percent (default: 100)
            force: 0-100 percent (default: 50)
        """
        # Single funnel for open_gripper / set_gripper_position / the
        # no-header close: the commanded target is the recorder's label.
        self._command_latch.latch_gripper(position)
        self._check_connected()
        speed = speed if speed is not None else 100
        force = force if force is not None else 50
        # rq_*_norm take percent (0-100). configs/ur10e_default.yaml carries
        # gripper.speed/force = 255 (the raw Robotiq register scale); those
        # CLIP to 100 here, which is full speed/force, so the legacy YAML
        # values still mean "max" rather than overflowing the norm command.
        pos_pct = int(np.clip(position * 100, 0, 100))
        spd = int(np.clip(speed, 0, 100))
        frc = int(np.clip(force, 0, 100))

        if self._gripper_script_header:
            script = f"""def gripper_move():
{self._gripper_script_header}

    rq_set_force_norm({frc}, "1")
    rq_set_speed_norm({spd}, "1")
    rq_move_and_wait_norm({pos_pct}, "1")
end
gripper_move()
"""
        else:
            # Fallback without script header (may not work on all setups)
            pos_raw = int(np.clip(position * 255, 0, 255))
            spd_raw = int(np.clip(speed * 2.55, 0, 255))
            frc_raw = int(np.clip(force * 2.55, 0, 255))
            script = (
                f"def gripper_cmd():\n"
                f"  rq_set_speed({spd_raw})\n"
                f"  rq_set_force({frc_raw})\n"
                f"  rq_move_and_wait({pos_raw})\n"
                f"end\n"
            )
        # Self-contained, self-invoking script -> primary-interface URScript
        # socket (see activate_gripper for why not sendCustomScriptFunction).
        if not self._send_script(script):
            raise RuntimeError("gripper command dropped: URScript channel down")
        time.sleep(0.5)

    def grasp_to_width(
        self,
        width: float,
        force: float = None,
        speed: float = None,
        epsilon_inner: float = None,
        epsilon_outer: float = None,
    ) -> bool:
        """
        Close the Robotiq to a target jaw width in METERS.

        Matches the FrankaDriverBase.grasp_to_width signature so the
        family-agnostic ScoreExecutor (_gripper_squeeze) can target a
        mask-derived grasp width on either embodiment. The executor passes
        ``speed`` as a 0-1 fraction; both that and a raw 0-100 percent are
        accepted. The 2F-85 has no width API, so width is mapped to a
        percent-closed command via GRIPPER_MAX_WIDTH_M. epsilon_* are
        accepted for signature parity and ignored (no Franka-style grasp
        tolerance window on the Robotiq).
        """
        w = float(np.clip(width, 0.0, self.GRIPPER_MAX_WIDTH_M))
        # 0 m -> fully closed (1.0); max width -> fully open (0.0).
        position = 1.0 - (w / self.GRIPPER_MAX_WIDTH_M)
        spd = speed
        if spd is not None and spd <= 1.0:
            spd = spd * 100.0  # fractional 0-1 -> percent
        frc = force
        if frc is not None:
            frc = min(float(frc), 100.0)
        try:
            self._send_gripper_command(position, speed=spd, force=frc)
            return True
        except Exception as e:
            logger.warning("grasp_to_width(%.3f) failed: %s", width, e)
            return False

    def get_gripper_width(self) -> float:
        """
        Current jaw opening in meters (0=closed, GRIPPER_MAX_WIDTH_M=open).
        """
        try:
            pos255 = self.get_gripper_position()  # 0=open, 255=closed
        except Exception:
            return 0.0
        frac_closed = float(np.clip(pos255 / 255.0, 0.0, 1.0))
        return (1.0 - frac_closed) * self.GRIPPER_MAX_WIDTH_M

    # RTDE output integer registers used to surface gripper state. This
    # ur_rtde build only exposes getOutputIntRegister for indices 12-19; the
    # general-purpose pair below is read back spin-free (see _publish_gripper_state).
    _POS_REGISTER = 12
    _OBJ_REGISTER = 13
    # Written LAST by _pub_grip as a completion token: when it reads back, the
    # controller has already written 12 and 13, so the publish is done
    # (typically ~10-30 ms).
    _SEQ_REGISTER = 14
    _PUBLISH_POLL_S = 0.005
    # A build that does not surface register 14 waits the full timeout.
    _PUBLISH_TIMEOUT_S = 0.2

    def _publish_gripper_state(self, force: bool = False):
        """Push gripper position + object-detect into RTDE output registers.

        The Robotiq state (rq_current_pos_norm / rq_is_object_detected) is only
        reachable from URScript, but a FIRE-AND-FORGET _send_script that WRITES
        an output register does NOT need the rtde_c control script and never
        spins (unlike a sendCustomScriptFunction round-trip, which hangs once
        _send_script has stopped the control script). The values are then read
        back with rtde_r.getOutputIntRegister, a pure receive read, also
        spin-free. The embedded URCap header defines the rq_* functions.
        """
        # Do not publish gripper state while the ARM is moving: _send_script
        # replaces the running URScript program, so the _pub_grip write mid-move
        # cancels the in-flight movel/movej/speedl. Return immediately and let
        # the read fall back to the last-published register values (the gripper
        # isn't actuating during a move, so they're current). The grasp squeeze
        # moves the gripper with the arm STATIONARY, so its verify reads still
        # refresh.
        #
        # A measured TCP speed > 0.01 m/s guard alone has two holes: a
        # Cartesian servo converging on its target drops BELOW 0.01 m/s while
        # still emitting speedl every 8 ms, and a fire-and-forget movej is
        # stationary for its whole ~1 s upload lag. A publish landing in either
        # hole cancels the motion, which then re-accelerates on the next tick.
        # _motion_in_flight closes both with a command-side lease.
        #
        # ``force`` is for a caller that KNOWS the arm is parked and needs a
        # fresh sample to decide something (a drop check at a transport
        # waypoint). It skips the lease (only a pessimistic upper bound on an
        # unconfirmed move) but still honours measured motion, so it can never
        # cancel a move that is genuinely running.
        if force:
            if self._tcp_is_moving():
                return
        elif self._motion_in_flight():
            return

        header = self._gripper_script_header or ""
        # write_output_integer_register requires an INTEGER: rq_is_object_detected
        # returns a Bool and rq_current_pos_norm a float (0-100), so floor the
        # position and map the bool to 0/1 via if/else. A raw bool write throws
        # "Must be an integer, not 'Bool'" and leaves register 13 unwritten.
        # Sequence token cycles 1..255 (never 0, so a cold register reads as
        # "not yet acknowledged" rather than matching by accident).
        self._publish_seq = getattr(self, "_publish_seq", 0) % 255 + 1
        seq = self._publish_seq
        script = (
            "def _pub_grip():\n"
            f"{header}\n"
            f'  write_output_integer_register({self._POS_REGISTER}, floor(rq_current_pos_norm("1")))\n'
            "  obj_i = 0\n"
            '  if (rq_is_object_detected("1")):\n'
            "    obj_i = 1\n"
            "  end\n"
            f"  write_output_integer_register({self._OBJ_REGISTER}, obj_i)\n"
            f"  write_output_integer_register({self._SEQ_REGISTER}, {seq})\n"
            "end\n"
            "_pub_grip()\n"
        )
        if not self._send_script(script):
            return
        self._await_publish(seq)

    def _await_publish(self, seq: int) -> bool:
        """
        Block until the controller acknowledges publish ``seq``.

        The token is written last, so seeing it means registers 12 and 13
        already hold this publish's values. Returns False on timeout, in which
        case the caller reads whatever the registers currently hold.
        """
        deadline = time.monotonic() + self._PUBLISH_TIMEOUT_S
        while time.monotonic() < deadline:
            try:
                if self._rtde_r.getOutputIntRegister(self._SEQ_REGISTER) == seq:
                    return True
            except Exception:  # noqa: BLE001 - unsupported register: just wait
                time.sleep(max(0.0, deadline - time.monotonic()))
                return False
            time.sleep(self._PUBLISH_POLL_S)
        return False

    def get_gripper_position(self, publish: bool = True) -> float:
        """Current gripper position, 0 (open) to 255 (closed). Spin-free.

        rq_current_pos_norm is 0-100; scaled to 0-255 to match the Robotiq
        convention the verify thresholds use.

        ``publish=False`` reads the last-published register without uploading
        a URScript program. Use it from any loop that runs CONCURRENTLY with
        motion (the demo recorder), where the refresh is worth less than the
        motion it would cancel.
        """
        self._check_connected()
        if publish:
            self._publish_gripper_state()
        return self._rtde_r.getOutputIntRegister(self._POS_REGISTER) * 2.55

    def is_object_detected(self, publish: bool = True) -> bool:
        """Robotiq gOBJ flag: True when the jaws stopped on an object (gOBJ
        1/2) rather than reaching the commanded position empty (gOBJ 3). The
        reliable grasp signal; detects a lateral grip TCP force can't. Read
        spin-free via the output register (see _publish_gripper_state)."""
        self._check_connected()
        if publish:
            self._publish_gripper_state()
        return self._rtde_r.getOutputIntRegister(self._OBJ_REGISTER) == 1

    #Context manager

    def __enter__(self):
        self.connect()
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.disconnect()
        return False

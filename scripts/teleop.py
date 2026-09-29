#!/usr/bin/env python3
"""Unified serverless teleop: one script for UR10e, FR3, and bimanual FR3.

Merges the per-rig scripts (gamepad_teleop_ur10e.py, gamepad_teleop.py,
teleop_fr3.py) behind one CLI with robot autodetection:

    cd ~/spark/src
    python ../scripts/teleop.py                      # autodetect robot + input
    python ../scripts/teleop.py --robot ur10e        # force a rig
    python ../scripts/teleop.py --robot fr3 --input keyboard
    python ../scripts/teleop.py --robot bimanual --arm left

Autodetection probes the rigs' known control ports (UR dashboard :29999,
Franka Desk :443) at the IPs from the family YAML configs (env overrides:
UR10E_IP, FRANKA_IP, FRANKA_LEFT_IP, FRANKA_RIGHT_IP). Input defaults to
gamepad when /dev/input/js0 exists, else keyboard.

GAMEPAD (HORI HORIPAD S numbering, same pad as teleop_collect.py; "author
controls" default, --invert for the author wrist-POV mapping):
  Left stick      X/Y translation      D-pad up/down   Z translation
  Right stick     yaw (X) / pitch (Y)  L / R bumpers   roll - / +
  ZL / ZR         gripper open / close
  Plus  (9) print TCP   Home (12) home   L3/R3 (10/11) slower/faster
  X     (3) bimanual: toggle active arm
  Minus (8) quit

KEYBOARD (same bindings as teleop_collect.py; hold-to-move via pynput):
  I/K  +/-X   J/L  +/-Y   U/O  +/-Z
  Q/E  yaw    W/S  pitch  A/D  roll
  F/G  gripper open/close   H home   P print pose
  [ ]  slower/faster        TAB bimanual arm toggle   ESC quit

UR10e-only extras (mirrors gamepad_teleop_ur10e.py): active
protective-stop dashboard auto-unlock + safety-popup dismiss, soft
--z-floor table guard, live TCP publish to /tmp/teleop_tcp.json for the
calib app. NOTE: ur-rtde is exclusive; stop the :8888 server's robot
connection first. On connect the Robotiq activation recalibrates by
physically cycling the gripper once (close+open) -- suppress with
--no-activate when the gripper is already calibrated.
"""

from __future__ import annotations

import dataclasses
import json
import os
import socket
import sys
import threading
import time
from pathlib import Path
from typing import Literal, Optional

import numpy as np

# Rig/input-specific third-party deps. Each is optional: a missing module
# only disables its own backend (failure surfaces at connect time).
try:
    import dashboard_client
except ImportError:
    dashboard_client = None
try:
    import tyro
except ImportError:
    tyro = None
try:
    import yaml
except ImportError:
    yaml = None
try:
    from franky import (
        CartesianVelocityMotion, CartesianVelocityStopMotion, Gripper,
        JointMotion, JointStopMotion, RelativeDynamicsFactor, Robot,
        RobotVelocity, Twist,
    )
except ImportError:
    CartesianVelocityMotion = CartesianVelocityStopMotion = Gripper = None
    JointMotion = JointStopMotion = RelativeDynamicsFactor = None
    Robot = RobotVelocity = Twist = None
try:
    from pynput import keyboard  # hold-to-move needs release events
except ImportError:
    keyboard = None

_SRC = Path(__file__).resolve().parents[1] / "src"
if (_SRC / "spark_real").exists():
    sys.path.insert(0, str(_SRC))

from spark_real.config import load_family_yaml  # noqa: E402
from spark_real.control.ur10e_driver import UR10eDriver  # noqa: E402

TELEOP_TCP_PATH = Path("/tmp/teleop_tcp.json")
_CONFIG_DIR = _SRC / "spark_real" / "configs"


# --------------------------------------------------------------------------
# Input backends
# --------------------------------------------------------------------------

# GamepadReader is imported from the UR script so there is exactly one
# copy of the ioctl/event logic.
sys.path.insert(0, str(Path(__file__).resolve().parent))
from gamepad_teleop_ur10e import (  # noqa: E402
    GamepadReader, build_velocity, find_gamepad, server_robot,
)


class KeyboardReader:
    """Hold-to-move keyboard input via pynput (same keys as teleop_collect).

    Exposes the same surface the control loop needs: velocity(), plus
    one-shot events ('quit', 'home', 'print_pose', 'open', 'close',
    'speed_up', 'speed_down', 'toggle_arm').
    """

    VEL_KEYS = {
        "i": (1, 0, 0, 0, 0, 0), "k": (-1, 0, 0, 0, 0, 0),
        "j": (0, 1, 0, 0, 0, 0), "l": (0, -1, 0, 0, 0, 0),
        "u": (0, 0, 1, 0, 0, 0), "o": (0, 0, -1, 0, 0, 0),
        "q": (0, 0, 0, 0, 0, 1), "e": (0, 0, 0, 0, 0, -1),
        "w": (0, 0, 0, 1, 0, 0), "s": (0, 0, 0, -1, 0, 0),
        "a": (0, 0, 0, 0, 1, 0), "d": (0, 0, 0, 0, -1, 0),
    }
    EVENT_KEYS = {
        "f": "open", "g": "close", "h": "home", "p": "print_pose",
        "[": "speed_down", "]": "speed_up",
    }

    def __init__(self):
        if keyboard is None:
            raise ImportError("pynput is not installed")
        self._kb = keyboard
        self._lock = threading.Lock()
        self._pressed: set = set()
        self._events: list = []
        self._listener = keyboard.Listener(
            on_press=self._on_press, on_release=self._on_release
        )

    def connect(self) -> bool:
        self._listener.start()
        print("Keyboard: pynput listener active (focus any window)")
        return True

    def disconnect(self):
        self._listener.stop()

    def _on_press(self, key):
        try:
            ch = key.char.lower() if getattr(key, "char", None) else None
        except AttributeError:
            ch = None
        with self._lock:
            if ch in self.VEL_KEYS:
                self._pressed.add(ch)
            elif ch in self.EVENT_KEYS:
                self._events.append(self.EVENT_KEYS[ch])
            elif key == self._kb.Key.esc:
                self._events.append("quit")
            elif key == self._kb.Key.tab:
                self._events.append("toggle_arm")

    def _on_release(self, key):
        try:
            ch = key.char.lower() if getattr(key, "char", None) else None
        except AttributeError:
            return
        with self._lock:
            self._pressed.discard(ch)

    def velocity(self, lin: float, ang: float) -> np.ndarray:
        v = np.zeros(6)
        with self._lock:
            for ch in self._pressed:
                m = self.VEL_KEYS[ch]
                v[:3] += np.array(m[:3], dtype=float) * lin
                v[3:] += np.array(m[3:], dtype=float) * ang
        return v

    def pop_events(self) -> list:
        with self._lock:
            ev, self._events = self._events, []
        return ev


# --------------------------------------------------------------------------
# Robot backends: connect / get_tcp_pose / send_velocity / stop / home /
# open_gripper / close_gripper / safety_tick / disconnect
# --------------------------------------------------------------------------


class UR10eBackend:
    """UR10e over spark_real's UR10eDriver (mirrors gamepad_teleop_ur10e.py)."""

    name = "ur10e"
    # Gripper scripts and speedl share the primary URScript interpreter:
    # speedl sent while rq_close/open runs kills the gripper program
    # mid-travel and can corrupt the Robotiq auto-calibration. The loop
    # holds motion this long after a gripper command.
    GRIPPER_HOLD_S = 1.5

    def __init__(self, ip: str, rate: float, z_floor: Optional[float],
                 publish_tcp: bool, activate: bool):
        self.ip = ip
        self.z_floor = z_floor
        self.publish_tcp = publish_tcp
        self.activate = activate
        self._speedl_dur = max(0.1, 2.0 / rate)
        self._safety_every = max(1, int(rate * 0.7))
        self._tick = 0
        self._was_moving = False
        self._dash = None
        self.robot = None

    def connect(self):
        print(f"Connecting to UR10e at {self.ip} ...")
        self.robot = UR10eDriver(self.ip)
        self.robot.connect()
        try:
            self._dash = dashboard_client.DashboardClient(self.ip)
            self._dash.connect()
        except Exception as e:  # noqa: BLE001
            print(f"(dashboard poll unavailable: {e}; auto-unlock degraded)")
            self._dash = None
        # LOCAL mode silently discards primary-interface scripts: activation
        # no-ops, stale calibration survives, full-close stops short. Say so.
        if self._dash is not None:
            try:
                if not self._dash.isInRemoteControl():
                    print("\n  *** PENDANT IS IN LOCAL MODE ***\n"
                          "  Gripper activation/close will be silently "
                          "ignored.\n  Flip the pendant to Remote Control and "
                          "re-run.\n")
            except Exception:  # noqa: BLE001
                pass
        if self.activate:
            try:
                # NOTE: physically cycles the gripper once (auto-calibration).
                self.robot.activate_gripper()
                # Cold controllers can outlast the driver's 5s settle; extra
                # margin so the first speedl can't kill the calibration cycle.
                time.sleep(2.0)
            except Exception as e:  # noqa: BLE001
                print(f"(gripper activate skipped: {e})")

    def safety_tick(self) -> bool:
        """Poll for protective stop; unlock + dismiss popups. True = handled
        a stop this tick (caller should zero smoothing and skip the frame)."""
        self._tick += 1
        if self._dash is None or self._tick % self._safety_every:
            return False
        try:
            ss = (self._dash.safetystatus() or "").upper()
        except Exception:  # noqa: BLE001
            return False
        if "PROTECTIVE_STOP" not in ss:
            return False
        print("  protective stop -> unlocking, hold ~5s ...")
        self._was_moving = False
        try:
            self._dash.unlockProtectiveStop()
            for fn in ("closeSafetyPopup", "closePopup"):
                try:
                    getattr(self._dash, fn)()
                except Exception:  # noqa: BLE001
                    pass
        except Exception as ue:  # noqa: BLE001
            print(f"  unlock failed: {ue}")
        time.sleep(5.0)
        print("  resumed.")
        return True

    def get_tcp_pose(self):
        return self.robot.get_tcp_pose()

    def send_velocity(self, v: np.ndarray):
        pose = None
        try:
            pose = self.robot.get_tcp_pose()
        except Exception:  # noqa: BLE001
            pass
        if self.z_floor is not None and pose is not None:
            if pose[2] <= self.z_floor and v[2] < 0:
                v = v.copy()
                v[2] = 0.0
        if np.any(v != 0.0):
            self.robot.send_velocity(list(v), acceleration=0.5,
                                     duration=self._speedl_dur)
            self._was_moving = True
        elif self._was_moving:
            self.robot.stop_velocity()
            self._was_moving = False
        if self.publish_tcp and pose is not None:
            try:
                TELEOP_TCP_PATH.write_text(json.dumps(
                    {"tcp_xyz": [float(pose[0]), float(pose[1]),
                                 float(pose[2])], "ts": time.time()}))
            except Exception:  # noqa: BLE001
                pass

    def recover(self) -> bool:
        """After a RuntimeError from speedl (protective stop killed the
        control script): dashboard unlock + reconnect."""
        try:
            self.robot.unlock_protective_stop()
            time.sleep(5.0)
            self.robot.disconnect()
            self.robot.connect()
            return True
        except Exception as e:  # noqa: BLE001
            print(f"  recovery failed: {e}; fix on pendant, then re-run.")
            return False

    def stop(self):
        self.robot.stop_velocity()
        self._was_moving = False

    def home(self):
        # speedl streaming stopped RTDEControl's script; movej must ride the
        # same URScript socket (see gamepad_teleop_ur10e.py).
        print("  homing (release sticks to let it finish) ...")
        self.robot.stop_velocity()
        self.robot.go_home_urscript(velocity=0.6)
        self._was_moving = False

    def open_gripper(self):
        self.robot.open_gripper()

    def close_gripper(self):
        self.robot.close_gripper()

    def print_pose(self):
        p = self.robot.get_tcp_pose()
        print(f"  TCP xyz=({p[0]:.4f}, {p[1]:.4f}, {p[2]:.4f}) "
              f"rot=({p[3]:.3f}, {p[4]:.3f}, {p[5]:.3f})")

    def disconnect(self):
        try:
            self.robot.stop_velocity()
        except Exception:  # noqa: BLE001
            pass
        if self._dash is not None:
            try:
                self._dash.disconnect()
            except Exception:  # noqa: BLE001
                pass
        self.robot.disconnect()


class FR3Backend:
    """Single Franka FR3 via franky async CartesianVelocityMotion (mirrors
    gamepad_teleop.py, plus angular twist so all 6 DOF work)."""

    name = "fr3"
    GRIPPER_HOLD_S = 0.0  # franky gripper is a separate connection
    HOME_Q = [0, -0.785398, 0, -2.356194, -0.15, 1.570796, 0.785398]

    def __init__(self, ip: str, label: str = "fr3"):
        self.ip = ip
        self.label = label
        self.robot = None
        self.gripper = None
        self._vel_active = False

    def connect(self):
        if Robot is None:
            raise ImportError("franky is not installed; FR3 teleop unavailable")
        print(f"Connecting to FR3 ({self.label}) at {self.ip} ...")
        self.robot = Robot(self.ip)
        self.robot.recover_from_errors()
        self.gripper = Gripper(self.ip)
        try:
            self.robot.set_collision_behavior(100.0, 100.0)
        except Exception:  # noqa: BLE001
            pass

    def safety_tick(self) -> bool:
        if self.robot.has_errors:
            try:
                self.robot.recover_from_errors()
            except Exception:  # noqa: BLE001
                pass
            self._vel_active = False
            return True
        return False

    def get_tcp_pose(self):
        ee = self.robot.current_cartesian_state.pose.end_effector_pose
        t = np.asarray(ee.translation, dtype=float)
        return [t[0], t[1], t[2], 0.0, 0.0, 0.0]

    def _stop_velocity(self):
        try:
            self.robot.move(CartesianVelocityStopMotion())
        except Exception:  # noqa: BLE001
            pass
        self._vel_active = False

    def send_velocity(self, v: np.ndarray):
        if not np.any(v != 0.0):
            if self._vel_active:
                self._stop_velocity()
            return
        twist = Twist(list(v[:3]), list(v[3:]))
        try:
            self.robot.move(
                CartesianVelocityMotion(RobotVelocity(twist),
                                        relative_dynamics_factor=0.15),
                asynchronous=True,
            )
            self._vel_active = True
        except Exception:  # noqa: BLE001
            pass

    def recover(self) -> bool:
        try:
            self.robot.recover_from_errors()
            return True
        except Exception:  # noqa: BLE001
            return False

    def stop(self):
        self._stop_velocity()

    def home(self):
        print(f"  homing {self.label} ...")
        self._stop_velocity()
        try:
            self.robot.recover_from_errors()
            try:
                self.robot.move(JointStopMotion())
            except Exception:  # noqa: BLE001
                pass
            rdf = RelativeDynamicsFactor(0.2, 0.1, 0.05)
            self.robot.move(JointMotion(self.HOME_Q,
                                        relative_dynamics_factor=rdf))
        except Exception as e:  # noqa: BLE001
            print(f"  home failed: {e}")

    def open_gripper(self):
        self._stop_velocity()
        try:
            self.gripper.open(0.1)
        except Exception:  # noqa: BLE001
            pass

    def close_gripper(self):
        self._stop_velocity()
        try:
            mw = self.gripper.max_width
            self.gripper.grasp(0.0, 0.1, 70.0, mw, mw)
        except Exception:  # noqa: BLE001
            pass

    def print_pose(self):
        t = self.get_tcp_pose()
        print(f"  {self.label} TCP: x={t[0]:.4f} y={t[1]:.4f} "
              f"z={t[2]*1000:.1f}mm")

    def disconnect(self):
        self._stop_velocity()
        try:
            self.robot.stop()
        except Exception:  # noqa: BLE001
            pass


class BimanualBackend:
    """Two FR3 arms; teleop drives ONE arm at a time (X button / TAB toggles).
    The idle arm holds position. --arm left|right pins the active arm."""

    name = "bimanual"
    GRIPPER_HOLD_S = 0.0

    def __init__(self, left_ip: str, right_ip: str, start_arm: str = "left"):
        self.arms = {
            "left": FR3Backend(left_ip, "left"),
            "right": FR3Backend(right_ip, "right"),
        }
        self.active = start_arm

    def connect(self):
        for arm in self.arms.values():
            arm.connect()
        print(f"Active arm: {self.active} (X button / TAB to switch)")

    def toggle_arm(self):
        self.arms[self.active].stop()
        self.active = "right" if self.active == "left" else "left"
        print(f"  active arm -> {self.active}")

    def _a(self) -> FR3Backend:
        return self.arms[self.active]

    def safety_tick(self):
        return any(arm.safety_tick() for arm in self.arms.values())

    def get_tcp_pose(self):
        return self._a().get_tcp_pose()

    def send_velocity(self, v):
        self._a().send_velocity(v)

    def recover(self):
        return self._a().recover()

    def stop(self):
        for arm in self.arms.values():
            arm.stop()

    def home(self):
        self._a().home()

    def open_gripper(self):
        self._a().open_gripper()

    def close_gripper(self):
        self._a().close_gripper()

    def print_pose(self):
        for arm in self.arms.values():
            arm.print_pose()

    def disconnect(self):
        for arm in self.arms.values():
            arm.disconnect()


# --------------------------------------------------------------------------
# Robot autodetection
# --------------------------------------------------------------------------


def _yaml_get(path: Path, *keys, default=None):
    try:
        raw = yaml.safe_load(path.read_text()) or {}
        for k in keys:
            raw = raw[k]
        return raw
    except Exception:  # noqa: BLE001
        return default


def _port_open(ip: str, port: int, timeout: float = 0.6) -> bool:
    try:
        with socket.create_connection((ip, port), timeout=timeout):
            return True
    except OSError:
        return False


def resolve_ips(machine: str = "ANON-LAB") -> dict:
    bimanual = load_family_yaml("bimanual_franka", machine).get("robot") or {}
    return {
        "ur10e": os.environ.get(
            "UR10E_IP",
            _yaml_get(_CONFIG_DIR / "ur10e_default.yaml", "robot", "ip",
                      default="192.168.56.101"),
        ),
        "fr3": os.environ.get(
            "FRANKA_IP",
            _yaml_get(_CONFIG_DIR / "franka_default.yaml", "robot", "ip",
                      default="172.16.0.2"),
        ),
        "left": os.environ.get(
            "FRANKA_LEFT_IP", bimanual.get("left_ip", "172.16.0.101")),
        "right": os.environ.get(
            "FRANKA_RIGHT_IP", bimanual.get("right_ip", "172.16.0.102")),
    }


def autodetect_robot(ips: dict) -> str:
    """Probe known control ports: UR dashboard :29999, Franka Desk :443."""
    print("Autodetecting robot ...")
    if _port_open(ips["ur10e"], 29999):
        print(f"  UR10e dashboard reachable at {ips['ur10e']}:29999")
        return "ur10e"
    left = _port_open(ips["left"], 443)
    right = _port_open(ips["right"], 443)
    if left and right:
        print(f"  both Franka Desks reachable ({ips['left']}, {ips['right']})")
        return "bimanual"
    if _port_open(ips["fr3"], 443):
        print(f"  Franka Desk reachable at {ips['fr3']}:443")
        return "fr3"
    if left or right:
        which = "left" if left else "right"
        print(f"  single bimanual arm ({which}) reachable; driving it as fr3")
        ips["fr3"] = ips[which]
        return "fr3"
    sys.exit("No robot reachable (UR :29999, Franka :443). "
             "Pass --robot and/or check IPs / env overrides.")


# --------------------------------------------------------------------------
# CLI + main loop
# --------------------------------------------------------------------------


@dataclasses.dataclass
class Config:
    """Unified teleop across the lab's rigs. See module docstring for keys."""

    robot: Literal["auto", "ur10e", "fr3", "bimanual"] = "auto"
    """Which rig to drive; auto probes UR :29999 then Franka Desk :443."""
    input: Literal["auto", "gamepad", "keyboard", "both"] = "auto"
    """Input source; auto = gamepad when /dev/input/js0 exists, else keyboard."""
    ip: Optional[str] = None
    """Robot IP override (single-arm rigs)."""
    arm: Literal["left", "right"] = "left"
    """Bimanual: arm that starts active (toggle at runtime with X / TAB)."""
    machine: str = "ANON-LAB"
    """Bimanual: configs/machines/<machine>.yaml overlay that holds the arm IPs."""
    linear_vel: float = 0.1
    """m/s at full stick / held key."""
    angular_vel: float = 0.3
    """rad/s at full stick / held key."""
    rate: float = 30.0
    """Control loop Hz."""
    deadzone: float = 0.15
    """Gamepad stick deadzone (0-1)."""
    device: str = "auto"
    """Joystick device path, or 'auto' to find the pad by name (port-independent)."""
    invert: bool = False
    """author wrist-POV stick mapping instead of the author default."""
    z_floor: Optional[float] = None
    """UR10e only: soft table floor (base Z, m); -Z is canceled below it."""
    no_publish: bool = False
    """UR10e only: do not write /tmp/teleop_tcp.json for the calib app."""
    no_activate: bool = False
    """UR10e only: skip Robotiq activation (avoids the close/open
    recalibration cycle at startup when the gripper is already calibrated)."""


def main(cfg: Config):
    ips = resolve_ips(cfg.machine)
    if cfg.ip:
        ips["ur10e"] = ips["fr3"] = cfg.ip

    robot_kind = cfg.robot if cfg.robot != "auto" else autodetect_robot(ips)

    if robot_kind == "ur10e":
        backend = UR10eBackend(ips["ur10e"], cfg.rate, cfg.z_floor,
                               not cfg.no_publish, not cfg.no_activate)
    elif robot_kind == "fr3":
        backend = FR3Backend(ips["fr3"])
    else:
        backend = BimanualBackend(ips["left"], ips["right"], cfg.arm)

    input_kind = cfg.input
    if input_kind == "auto":
        input_kind = "gamepad" if os.path.exists(cfg.device) else "keyboard"

    gp = kb = None
    if input_kind in ("gamepad", "both"):
        gp_device = cfg.device
        if gp_device == "auto":
            gp_device, gp_name = find_gamepad()
            if gp_device is None:
                if input_kind == "gamepad":
                    sys.exit("No joystick under /dev/input/js*; plug in the pad.")
                gp_device = cfg.device
            else:
                print(f"Auto-selected gamepad: {gp_name}  ({gp_device})")
        gp = GamepadReader(gp_device, cfg.deadzone) if gp_device != "auto" else None
        if gp is None or not gp.connect():
            if input_kind == "gamepad":
                sys.exit(1)
            gp = None
    if input_kind in ("keyboard", "both"):
        try:
            kb = KeyboardReader()
            kb.connect()
        except Exception as e:  # noqa: BLE001
            if gp is None:
                sys.exit(f"keyboard input unavailable ({e}); "
                         "pip install pynput or use --input gamepad")
            print(f"(keyboard unavailable: {e}; gamepad only)")
            kb = None

    # Robot control channels (RTDE / libfranka) are exclusive: if the spark
    # server holds this robot, borrow it and hand it back on exit. Works for
    # every family (the server endpoint drops whatever pipeline._robot is).
    rel = server_robot("robot/release")
    if rel and rel.get("released") and not rel.get("note"):
        print("(server released the robot for teleop; returned on exit)")

    backend.connect()

    dt = 1.0 / cfg.rate
    alpha = 0.4  # EMA on commanded velocity (damps accel spikes)
    smoothed = np.zeros(6)
    speed_scale = 1.0
    gripper_hold_until = 0.0

    print(f"\n  teleop [{robot_kind}] "
          f"({'author' if cfg.invert else 'author'} mapping, "
          f"input={'+'.join(k for k, v in [('gamepad', gp), ('keyboard', kb)] if v)})"
          "\n  Minus/ESC quit, Plus/P pose, Home/H home, ZL-ZR/F-G gripper\n")

    try:
        while True:
            t0 = time.time()

            if backend.safety_tick():
                smoothed[:] = 0.0
                continue

            events = kb.pop_events() if kb else []
            if gp:
                if gp.button_pressed(8):
                    events.append("quit")
                if gp.button_pressed(9):
                    events.append("print_pose")
                if gp.button_pressed(12):
                    events.append("home")
                if gp.button_pressed(11):
                    events.append("speed_up")
                if gp.button_pressed(10):
                    events.append("speed_down")
                if gp.button_pressed(6):
                    events.append("open")
                if gp.button_pressed(7):
                    events.append("close")
                if gp.button_pressed(3):
                    events.append("toggle_arm")

            quit_now = False
            for ev in events:
                if ev == "quit":
                    quit_now = True
                elif ev == "print_pose":
                    backend.print_pose()
                elif ev == "home":
                    backend.home()
                    smoothed[:] = 0.0
                elif ev == "speed_up":
                    speed_scale = min(2.0, speed_scale + 0.25)
                    print(f"  speed x{speed_scale:.2f}")
                elif ev == "speed_down":
                    speed_scale = max(0.25, speed_scale - 0.25)
                    print(f"  speed x{speed_scale:.2f}")
                elif ev == "open":
                    try:
                        backend.open_gripper()
                        gripper_hold_until = time.time() + backend.GRIPPER_HOLD_S
                    except Exception as e:  # noqa: BLE001
                        print(f"  gripper open failed: {e}")
                elif ev == "close":
                    try:
                        backend.close_gripper()
                        gripper_hold_until = time.time() + backend.GRIPPER_HOLD_S
                    except Exception as e:  # noqa: BLE001
                        print(f"  gripper close failed: {e}")
                elif ev == "toggle_arm" and hasattr(backend, "toggle_arm"):
                    backend.toggle_arm()
                    smoothed[:] = 0.0
            if quit_now:
                print("quit")
                break

            lin = cfg.linear_vel * speed_scale
            ang = cfg.angular_vel * speed_scale
            target = np.zeros(6)
            if gp:
                target += build_velocity(gp, lin, ang, cfg.invert)
            if kb:
                target += kb.velocity(lin, ang)

            if time.time() < gripper_hold_until:
                target[:] = 0.0  # let the gripper program finish (see backend)

            smoothed = alpha * target + (1 - alpha) * smoothed
            smoothed[np.abs(smoothed) < 1e-4] = 0.0

            try:
                backend.send_velocity(smoothed)
            except RuntimeError as e:
                print(f"  motion fault: {e}\n  attempting recovery ...")
                smoothed[:] = 0.0
                if not backend.recover():
                    break
                print("  recovered.")

            sleep = dt - (time.time() - t0)
            if sleep > 0:
                time.sleep(sleep)
    except KeyboardInterrupt:
        print("\ninterrupted")
    finally:
        try:
            backend.stop()
        except Exception:  # noqa: BLE001
            pass
        if gp:
            gp.disconnect()
        if kb:
            kb.disconnect()
        backend.disconnect()
        if rel and rel.get("released") and not rel.get("note"):
            back = server_robot("connect_robot")
            print("(robot handed back to server)" if back and back.get("connected")
                  else "(server did not reacquire; POST /api/connect_robot or restart it)")
        print("disconnected.")


if __name__ == "__main__":
    if tyro is None:
        # tyro not installed in this env: fall back to defaults-only run
        # (the flags all have safe defaults; --robot etc. need tyro).
        print("(tyro not installed; running with all defaults)")
        main(Config())
    else:
        main(tyro.cli(Config))

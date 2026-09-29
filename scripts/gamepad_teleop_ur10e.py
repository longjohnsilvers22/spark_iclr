#!/usr/bin/env python3
"""Serverless gamepad teleop for the UR10e (no spark_real server needed).

Drives the arm directly through spark_real's own UR10eDriver (ur-rtde
speedl velocity streaming), so it does NOT need the :8888 FastAPI server
up. It also publishes the live TCP to /tmp/teleop_tcp.json at the control
rate, which is exactly the file calib_app._current_tcp() falls back to
when the server can't read the robot -- so you can hand-drive the gripper
to ChArUco corners with the pad and hit "Record touch" in the calib app
with no server running.

Button/stick mapping is mirrored from the UR10e VLA data-collection
teleop (robotsteering/.../ur10e_pi05_deploy/scripts/teleop_collect.py,
the "author controls" default), read raw from /dev/input/jsX via the Linux
joystick API -- no pygame dependency.

  Left stick      X/Y translation     (push up = +X forward, left = +Y)
  D-pad up/down   Z translation       (up = +Z)
  Right stick     yaw (X) / pitch (Y)
  L / R bumpers   roll - / +
  ZL / ZR         gripper open / close
  Plus  (9)       print TCP pose
  Home  (12)      move to home config
  L3 / R3 (10/11) slow down / speed up
  Minus (8)       quit

HORI HORIPAD S axis/button numbers (same pad as teleop_collect):
  Axes:  0=LX 1=LY 2=RX 3=RY 4=DpadX 5=DpadY
  Btns:  0=B 1=A 2=Y 3=X 4=L 5=R 6=ZL 7=ZR 8=Minus 9=Plus 10=L3 11=R3 12=Home

ur-rtde's RTDEControl is exclusive: stop the :8888 server's robot
connection before running this (they cannot both command the arm).

Usage:
    cd ~/spark/src && \
      ~/miniconda3/envs/spark_conda/bin/python ../scripts/gamepad_teleop_ur10e.py
    # options: --ip, --linear-vel, --angular-vel, --rate, --invert, --no-publish
"""

import argparse
import array
import fcntl
import glob
import json
import os
import struct
import sys
import threading
import time
import urllib.request
from pathlib import Path

import numpy as np

# The UR dashboard client ships with ur-rtde; the script degrades gracefully
# (no protective-stop auto-unlock) when it is unavailable.
try:
    import dashboard_client
except ImportError:
    dashboard_client = None

# spark_real lives under ~/spark/src; add it so this script runs from anywhere.
_SRC = Path(__file__).resolve().parents[1] / "src"
if (_SRC / "spark_real").exists():
    sys.path.insert(0, str(_SRC))

from spark_real.control.ur10e_driver import UR10eDriver  # noqa: E402

TELEOP_TCP_PATH = Path("/tmp/teleop_tcp.json")
SERVER_URL = "http://localhost:8888"

# Home for the Home button: the human-teleop collector's home_joints, so SPARK
# demos start from the same pose as the human corpus in /data/teleop_episodes.
# NOT UR10eDriver.HOME_CONFIG, which differs by 170 deg at wrist 3; that rolls
# the wrist camera a half turn and would flip every recorded wrist image
# relative to the human episodes.
# TCP there: [-0.8144, 0.1089, 0.1304] m, rot [-2.2836, -2.1399, 0.0231] rad.
HOME_JOINTS = [3.2070, -1.8788, -1.7903, 5.2496, 1.5762, 0.0916]


def server_robot(action: str):
    """Best-effort POST to the spark server (release/reacquire the robot).
    Returns the parsed response, or None if the server is down."""
    try:
        req = urllib.request.Request(
            f"{SERVER_URL}/api/{action}", data=b"{}", method="POST",
            headers={"Content-Type": "application/json"})
        with urllib.request.urlopen(req, timeout=5) as r:
            return json.loads(r.read().decode())
    except Exception:
        return None


_JSIOCGNAME = 0x80406A13


def find_gamepad(prefer=("hori", "horipad", "gamepad", "controller", "joystick")):
    """Return the /dev/input/jsN path of the preferred pad, wherever it landed.

    The joystick device NUMBER depends on USB port / plug order, so a fixed
    /dev/input/js0 reads whatever enumerated first (a different pad, or nothing)
    after a replug. Scans every jsN, reads its name via the joystick ioctl, and
    picks the first name matching `prefer`; else the lowest-numbered joystick;
    else None. The raw button/axis NUMBERING for a given pad is stable across
    ports, so the mapping is correct regardless of where it is plugged.
    """
    devs = sorted(glob.glob("/dev/input/js*"),
                  key=lambda p: int("".join(filter(str.isdigit, p)) or 0))
    named = []
    for path in devs:
        try:
            with open(path, "rb") as fd:
                buf = array.array("B", [0] * 128)
                fcntl.ioctl(fd, _JSIOCGNAME, buf)
                name = buf.tobytes().split(b"\x00", 1)[0].decode("utf-8", "ignore")
            named.append((path, name))
        except OSError:
            continue
    for want in prefer:
        for path, name in named:
            if want in name.lower():
                return path, name
    if named:
        return named[0]
    return None, ""


class GamepadReader:
    """Raw Linux joystick reader (/dev/input/jsX), no external deps.

    Button/axis numbers line up with teleop_collect.py exactly.
    """

    JSIOCGAXES = 0x80016A11
    JSIOCGBUTTONS = 0x80016A12
    JSIOCGNAME = 0x80406A13
    EVENT_FORMAT = "IhBB"
    EVENT_SIZE = struct.calcsize(EVENT_FORMAT)
    JS_EVENT_BUTTON = 0x01
    JS_EVENT_AXIS = 0x02
    JS_EVENT_INIT = 0x80

    def __init__(self, device="/dev/input/js0", deadzone=0.15):
        self.device = device
        self.deadzone = deadzone
        self._lock = threading.Lock()
        self._axes = {}
        self._buttons = {}
        self._button_edges = {}
        self._running = False
        self._thread = None
        self._js_fd = None
        self.name = ""
        self.num_axes = 0
        self.num_buttons = 0

    def connect(self) -> bool:
        try:
            self._js_fd = open(self.device, "rb")
        except FileNotFoundError:
            print(f"Gamepad not found at {self.device}")
            return False
        except PermissionError:
            print(f"Permission denied for {self.device} (try: sudo chmod a+r {self.device})")
            return False

        buf = array.array("B", [0])
        fcntl.ioctl(self._js_fd, self.JSIOCGAXES, buf)
        self.num_axes = buf[0]
        buf = array.array("B", [0])
        fcntl.ioctl(self._js_fd, self.JSIOCGBUTTONS, buf)
        self.num_buttons = buf[0]
        buf = array.array("B", [0] * 64)
        fcntl.ioctl(self._js_fd, self.JSIOCGNAME, buf)
        self.name = buf.tobytes().decode("utf-8", "ignore").rstrip("\x00")

        flags = fcntl.fcntl(self._js_fd, fcntl.F_GETFL)
        fcntl.fcntl(self._js_fd, fcntl.F_SETFL, flags | os.O_NONBLOCK)

        for i in range(self.num_axes):
            self._axes[i] = 0.0
        for i in range(self.num_buttons):
            self._buttons[i] = False

        self._running = True
        self._thread = threading.Thread(target=self._read_loop, daemon=True)
        self._thread.start()
        print(f"Gamepad: {self.name} ({self.num_axes} axes, {self.num_buttons} buttons)")
        return True

    def disconnect(self):
        self._running = False
        if self._thread:
            self._thread.join(timeout=1.0)
        if self._js_fd:
            self._js_fd.close()
            self._js_fd = None

    def _read_loop(self):
        while self._running:
            try:
                data = self._js_fd.read(self.EVENT_SIZE)
                if data is None or len(data) < self.EVENT_SIZE:
                    time.sleep(0.001)
                    continue
                _ts, value, evt_type, number = struct.unpack(self.EVENT_FORMAT, data)
                base = evt_type & ~self.JS_EVENT_INIT
                with self._lock:
                    if base == self.JS_EVENT_AXIS:
                        raw = value / 32767.0
                        self._axes[number] = 0.0 if abs(raw) < self.deadzone else raw
                    elif base == self.JS_EVENT_BUTTON:
                        pressed = bool(value)
                        if pressed and not self._buttons.get(number, False):
                            self._button_edges[number] = True
                        self._buttons[number] = pressed
            except BlockingIOError:
                time.sleep(0.001)
            except Exception:
                if self._running:
                    time.sleep(0.01)

    def axis(self, num):
        with self._lock:
            return self._axes.get(num, 0.0)

    def button(self, num):
        with self._lock:
            return self._buttons.get(num, False)

    def button_pressed(self, num):
        with self._lock:
            if self._button_edges.get(num, False):
                self._button_edges[num] = False
                return True
            return False


def build_velocity(gp, lin, ang, invert):
    """Base-frame [vx,vy,vz,wrx,wry,wrz] from pad state. Mirrors
    teleop_collect.py's contribution block exactly."""
    inv = -1.0 if invert else 1.0
    v = np.zeros(6)
    # Left stick -> XY (stick Y is inverted: push up = negative raw)
    v[0] += -gp.axis(1) * inv * lin  # forward/back (+X)
    v[1] += -gp.axis(0) * inv * lin  # left/right (+Y)
    # D-pad Y -> Z
    v[2] += -gp.axis(5) * lin
    if not invert:
        # "author controls" (base-frame intuitive)
        v[5] += -gp.axis(2) * ang  # yaw
        v[3] += -gp.axis(3) * ang  # pitch
        if gp.button(4):
            v[4] += ang  # roll left
        if gp.button(5):
            v[4] -= ang  # roll right
    else:
        # "author controls" (wrist-camera-POV intuitive)
        v[3] -= -gp.axis(2) * ang  # yaw
        v[4] += -gp.axis(3) * ang  # pitch
        if gp.button(4):
            v[5] += ang
        if gp.button(5):
            v[5] -= ang
    return v


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--ip", default="192.168.56.101", help="UR10e controller IP")
    ap.add_argument("--device", default="auto",
                    help="joystick device path, or 'auto' (default) to find the "
                         "HORIPAD by name across /dev/input/js* — port-independent")
    ap.add_argument("--linear-vel", type=float, default=0.1, help="m/s at full stick")
    ap.add_argument("--angular-vel", type=float, default=0.3, help="rad/s at full stick")
    ap.add_argument("--deadzone", type=float, default=0.15)
    ap.add_argument("--rate", type=float, default=30.0, help="control loop Hz")
    ap.add_argument("--invert", action="store_true",
                    help="author wrist-POV stick mapping instead of author default")
    ap.add_argument("--no-publish", action="store_true",
                    help="do not write /tmp/teleop_tcp.json (calib app TCP bridge)")
    ap.add_argument("--z-floor", type=float, default=None,
                    help="soft table floor (base-frame Z, m): zero out -Z below "
                         "this so you cannot drive into the table. Measured "
                         "table on host ~ -0.2854; pass e.g. -0.275 for a "
                         "1 cm margin. Default: no floor (fully manual).")
    ap.add_argument("--identify", action="store_true",
                    help="print each button's index as you press it (no robot, "
                         "no motion) to correct the layout for a new pad")
    ap.add_argument("--no-activate", action="store_true",
                    help="skip Robotiq activation at startup. Activation "
                         "force-recalibrates by physically cycling the gripper "
                         "(the close+open you see on launch); skip it when the "
                         "gripper is already activated and calibrated.")
    args = ap.parse_args()

    device = args.device
    if device == "auto":
        device, found_name = find_gamepad()
        if device is None:
            sys.exit("No joystick found under /dev/input/js*. Plug in the pad "
                     "(or pass --device /dev/input/jsN).")
        print(f"Auto-selected gamepad: {found_name}  ({device})")

    gp = GamepadReader(device, args.deadzone)
    if not gp.connect():
        sys.exit(1)

    # --identify: print the index of every button you press (and deflected
    # axis), no robot connection, no motion. Use it to read what THIS pad
    # calls each physical button so the constants can be corrected once.
    if args.identify:
        print("\nIDENTIFY MODE — press each button; Ctrl-C to quit. "
              "No robot connected, nothing moves.\n")
        prev = {}
        try:
            while True:
                for b in range(gp.num_buttons):
                    now = gp.button(b)
                    if now and not prev.get(b):
                        print(f"  button {b} pressed")
                    prev[b] = now
                for a in range(gp.num_axes):
                    v = gp.axis(a)
                    if abs(v) > 0.7:
                        print(f"  axis {a} = {v:+.2f}")
                time.sleep(0.05)
        except KeyboardInterrupt:
            print("\nidentify done.")
        gp.disconnect()
        return

    # RTDEControl is exclusive: if the spark server holds the robot, borrow
    # it (released here, handed back on exit). No-op when the server is down
    # or already robot-less.
    rel = server_robot("robot/release")
    if rel and rel.get("released") and not rel.get("note"):
        print("(server released the robot for teleop; returned on exit)")

    print(f"Connecting to UR10e at {args.ip} ...")
    robot = UR10eDriver(args.ip)
    robot.connect()

    # Gripper scripts (activation included) ride the primary URScript socket,
    # which the controller silently DISCARDS in LOCAL mode: activation no-ops,
    # the stale Robotiq calibration survives, and full-close stops short until
    # someone activates by hand on the pendant. Detect and say so up front.
    try:
        _d0 = dashboard_client.DashboardClient(args.ip)
        _d0.connect()
        if not _d0.isInRemoteControl():
            print("\n  *** PENDANT IS IN LOCAL MODE ***\n"
                  "  Gripper activation/close scripts will be silently\n"
                  "  ignored (full-close will stop short). Flip the pendant\n"
                  "  top-right menu to Remote Control, then re-run.\n")
        _d0.disconnect()
    except Exception:
        pass

    if not args.no_activate:
        try:
            robot.activate_gripper()
            # The Robotiq auto-calibration cycle can outlast the driver's
            # fixed 5s settle on a cold controller; a speedl sent while it is
            # still cycling kills the script mid-calibration (full-close then
            # stops short). Extra margin costs 2s once per launch.
            time.sleep(2.0)
        except Exception as e:
            print(f"(gripper activate skipped: {e})")

    # Persistent dashboard client for ACTIVE protective-stop recovery. Pressing
    # the gripper into the table trips a protective stop that silently halts
    # speedl WITHOUT raising, so the try/except below never sees it. Polling
    # safetystatus here auto-unlocks with no trip to the pendant. Best-effort:
    # if the connection can't be made, the RuntimeError path still covers
    # RTDEControl-side faults.
    _dash = None
    try:
        _dash = dashboard_client.DashboardClient(args.ip)
        _dash.connect()
    except Exception as _e:
        print(f"(dashboard poll unavailable: {_e}; protective-stop auto-unlock degraded)")
        _dash = None
    _safety_every = max(1, int(args.rate * 0.7))  # check ~1.4x/sec
    _safety_tick = 0

    dt = 1.0 / args.rate
    speedl_dur = max(0.1, 2.0 * dt)  # overlap commands so motion stays smooth
    alpha = 0.4  # EMA: damps accel spikes (UR C306A3 sanity faults)
    smoothed = np.zeros(6)
    was_moving = False
    speed_scale = 1.0
    _gripper_hold_until = 0.0  # motion hold while a gripper script runs

    print(f"""
  UR10e gamepad teleop ({'author' if args.invert else 'author'} mapping) -- {gp.name}
    L-stick: XY   D-pad U/D: Z   R-stick: yaw/pitch   L/R: roll
    ZL/ZR: gripper open/close   Plus: print pose   Home: home
    L3/R3: slower/faster   Minus: quit
    publishing TCP -> {'(disabled)' if args.no_publish else TELEOP_TCP_PATH}
    z-floor  -> {'OFF (watch the table!)' if args.z_floor is None else f'{args.z_floor:.4f} m (-Z canceled below)'}
    protective stop -> AUTO dashboard-unlock while driving (no pendant; ~5s hold each)
""")

    try:
        while True:
            t0 = time.time()

            # Active protective-stop auto-recovery (~1.4 Hz). speedl never
            # raises on a protective stop, so detect it via the dashboard and
            # unlock in software -- you can keep driving without the pendant.
            _safety_tick += 1
            if _dash is not None and _safety_tick % _safety_every == 0:
                try:
                    ss = (_dash.safetystatus() or "").upper()
                except Exception:
                    ss = ""
                if "PROTECTIVE_STOP" in ss:
                    print("  protective stop (table contact?) -> unlocking, hold ~5s ...")
                    smoothed[:] = 0.0
                    was_moving = False
                    try:
                        _dash.unlockProtectiveStop()
                        for _fn in ("closeSafetyPopup", "closePopup"):
                            try:
                                getattr(_dash, _fn)()
                            except Exception:
                                pass
                    except Exception as ue:
                        print(f"  unlock failed: {ue}")
                    time.sleep(5.0)  # UR-enforced settle after unlock
                    print("  resumed -- easier on the table next time.")
                    continue

            if gp.button_pressed(8):  # Minus -> quit
                print("quit")
                break
            if gp.button_pressed(9):  # Plus -> print pose
                p = robot.get_tcp_pose()
                print(f"  TCP xyz=({p[0]:.4f}, {p[1]:.4f}, {p[2]:.4f}) "
                      f"rot=({p[3]:.3f}, {p[4]:.3f}, {p[5]:.3f})")
            if gp.button_pressed(12):  # Home
                # Home over the URScript channel: speedl streaming has stopped
                # the RTDEControl script, so rtde_c.moveJ would silently no-op.
                # This sends movej on the same socket as speedl.
                print("  homing (release sticks; push any stick to override) ...")
                robot.stop_velocity()
                robot.move_to_joint_config_urscript(
                    HOME_JOINTS, velocity=0.6, acceleration=1.0
                )
                smoothed[:] = 0.0
                was_moving = False  # don't let the zero-vel branch stopl the move
                continue
            if gp.button_pressed(11):  # R3 -> faster
                speed_scale = min(2.0, speed_scale + 0.25)
                print(f"  speed x{speed_scale:.2f}")
            if gp.button_pressed(10):  # L3 -> slower
                speed_scale = max(0.25, speed_scale - 0.25)
                print(f"  speed x{speed_scale:.2f}")
            if gp.button_pressed(6):  # ZL -> open
                try:
                    robot.open_gripper()
                    _gripper_hold_until = time.time() + 1.5
                except Exception as e:
                    print(f"  gripper open failed: {e}")
            if gp.button_pressed(7):  # ZR -> close
                try:
                    robot.close_gripper()
                    _gripper_hold_until = time.time() + 1.5
                except Exception as e:
                    print(f"  gripper close failed: {e}")

            target = build_velocity(gp, args.linear_vel * speed_scale,
                                    args.angular_vel * speed_scale, args.invert)

            # Gripper scripts and speedl share the primary URScript
            # interpreter: sending speedl while rq_close/open is still running
            # KILLS the gripper program mid-travel, which can corrupt the
            # Robotiq auto-calibration (close stops short until you
            # recalibrate). Hold motion until the gripper program finishes.
            if time.time() < _gripper_hold_until:
                target[:] = 0.0

            # Soft table floor: at/under the floor and still being
            # pushed down, cancel the -Z component so the gripper can't drive
            # through the table. XY/up motion stays free.
            pose = None
            try:
                pose = robot.get_tcp_pose()
            except Exception:
                pass
            if args.z_floor is not None and pose is not None:
                if pose[2] <= args.z_floor and target[2] < 0:
                    target[2] = 0.0

            smoothed = alpha * target + (1 - alpha) * smoothed
            smoothed[np.abs(smoothed) < 1e-4] = 0.0

            try:
                if np.any(smoothed != 0.0):
                    robot.send_velocity(list(smoothed), acceleration=0.5,
                                        duration=speedl_dur)
                    was_moving = True
                elif was_moving:
                    robot.stop_velocity()
                    was_moving = False
            except RuntimeError as e:
                # Usually a protective stop killed the control script. Try to
                # recover without the pendant, then re-establish the link.
                print(f"  motion fault: {e}\n  attempting dashboard unlock + reconnect ...")
                smoothed[:] = 0.0
                try:
                    robot.unlock_protective_stop()
                    time.sleep(5.0)  # UR-enforced settle after unlock
                    robot.disconnect()
                    robot.connect()
                    print("  recovered.")
                except Exception as re:
                    print(f"  recovery failed: {re}\n  fix on pendant, then re-run.")
                    break

            if not args.no_publish and pose is not None:
                try:
                    TELEOP_TCP_PATH.write_text(json.dumps(
                        {"tcp_xyz": [float(pose[0]), float(pose[1]), float(pose[2])],
                         "ts": time.time()}))
                except Exception:
                    pass

            sleep = dt - (time.time() - t0)
            if sleep > 0:
                time.sleep(sleep)
    except KeyboardInterrupt:
        print("\ninterrupted")
    finally:
        try:
            robot.stop_velocity()
        except Exception:
            pass
        gp.disconnect()
        if _dash is not None:
            try:
                _dash.disconnect()
            except Exception:
                pass
        robot.disconnect()
        if rel and rel.get("released") and not rel.get("note"):
            back = server_robot("connect_robot")
            print("(robot handed back to server)" if back and back.get("connected")
                  else "(server did not reacquire; POST /api/connect_robot or restart it)")
        print("disconnected.")


if __name__ == "__main__":
    main()

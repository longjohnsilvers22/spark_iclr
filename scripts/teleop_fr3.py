#!/usr/bin/env python3
"""
Keyboard/gamepad teleop for Franka FR3 via franky + Jacobian IK.

Uses JointWaypointMotion (NOT CartesianVelocityMotion) to avoid FR3
reflex issues. Cartesian deltas are mapped to joint deltas via the
6x7 geometric Jacobian from pyroki FK.

Keyboard mode (default):
  W/S         : +/- X (forward/back)
  A/D         : +/- Y (left/right)
  Q/E         : +/- Z (up/down)
  Arrow keys  : rotate (Up/Down = pitch, Left/Right = yaw)
  ,/.         : roll CW/CCW
  O           : open gripper
  C           : close gripper (grasp)
  P           : print TCP pose (position + quaternion)
  H           : go to home configuration
  +/-         : increase/decrease step size
  ESC         : quit

Gamepad mode (--gamepad):
  Left stick  : XY motion (continuous)
  RStick L/R  : yaw (continuous)
  RStick U/D  : roll (continuous)
  D-pad L/R   : pitch (step)
  D-pad U/D   : Z up / down (step)
  L2 / R2     : Z down / up (continuous, held)
  L1          : close gripper
  R1          : open gripper
  Select      : home
  Start       : quit

Usage:
  conda activate spark_conda
  cd ~/spark/src
  python ../scripts/teleop_fr3.py              # keyboard
  python ../scripts/teleop_fr3.py --gamepad    # gamepad (/dev/input/js0)
  python ../scripts/teleop_fr3.py --ip 10.0.0.1  # different robot IP
"""

from __future__ import annotations

import argparse
import json
import os
import select
import signal
import struct
import sys
import termios
import time
import tty

# Live TCP publisher target, read by scripts/calib_app.py when the spark
# server cannot read the robot because this process owns the connection.
TCP_PUB_PATH = "/tmp/teleop_tcp.json"

import numpy as np
from scipy.spatial.transform import Rotation

# franky is only installed on the FR3 rig machines. Keep the script importable
# (and --help / JIT warm-up working) elsewhere; failure surfaces at
# connect_robot().
try:
    from franky import (
        Gripper, JointMotion, JointStopMotion, JointWaypoint,
        JointWaypointMotion, RealtimeConfig, RelativeDynamicsFactor, Robot,
    )
except ImportError:
    Gripper = JointMotion = JointStopMotion = JointWaypoint = None
    JointWaypointMotion = RealtimeConfig = RelativeDynamicsFactor = Robot = None

# Ensure spark_real is importable
_src = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "src")
if _src not in sys.path:
    sys.path.insert(0, os.path.abspath(_src))

# Pyroki JAX caps (before any JAX import)
os.environ.setdefault("JAX_PLATFORMS", "cpu")
os.environ.setdefault("XLA_PYTHON_CLIENT_PREALLOCATE", "false")
os.environ.setdefault("XLA_PYTHON_CLIENT_MEM_FRACTION", "0.15")
os.environ["JAX_ENABLE_X64"] = "true"

# JAX imports must come AFTER the env caps above.
import jax  # noqa: E402
import jax.numpy as jnp  # noqa: E402
import jaxlie  # noqa: E402

# FK / Jacobian
# pyroki FK, but a local Jacobian: fr3_ik_pyroki.jacobian() has float32
# precision issues in the rotational part (trace(dR) slightly above 3.0 ->
# arccos clips angle to 0).
from spark_real.control import fr3_ik_pyroki  # noqa: E402
from spark_real.control.fr3_ik_pyroki import _load, fk  # noqa: E402

# Float64 Jacobian with proper small-angle extraction
_pyroki_loaded = False
_pk_robot = None
_pk_tli = None
_pk_nact = None


def _ensure_pyroki():
    global _pyroki_loaded, _pk_robot, _pk_tli, _pk_nact
    if _pyroki_loaded:
        return
    jax.config.update("jax_enable_x64", True)
    _load()
    _pk_robot = fr3_ik_pyroki._robot
    _pk_tli = fr3_ik_pyroki._tli
    _pk_nact = fr3_ik_pyroki._nact
    _pyroki_loaded = True


def jacobian(q: np.ndarray, eps: float = 1e-6) -> np.ndarray:
    """
    6x7 geometric Jacobian using float64 pyroki FK.

    Uses the skew-symmetric part of dR for the angular velocity columns,
    which is accurate for small perturbations and avoids the arccos
    singularity at angle=0.
    """
    _ensure_pyroki()
    q = np.asarray(q, dtype=np.float64).ravel()
    qf = np.zeros(_pk_nact, dtype=np.float64)
    qf[:min(7, _pk_nact)] = q[:min(7, _pk_nact)]
    if _pk_nact > 7:
        qf[7:] = 0.02

    jq = jnp.asarray(qf, dtype=jnp.float64)
    T0 = jaxlie.SE3(_pk_robot.forward_kinematics(jq)[_pk_tli])
    p0 = np.asarray(T0.translation(), dtype=np.float64)
    R0 = np.asarray(T0.rotation().as_matrix(), dtype=np.float64)

    J = np.zeros((6, 7), dtype=np.float64)
    for i in range(7):
        qp = qf.copy()
        qp[i] += eps
        Tp = jaxlie.SE3(_pk_robot.forward_kinematics(jnp.asarray(qp, dtype=jnp.float64))[_pk_tli])
        pp = np.asarray(Tp.translation(), dtype=np.float64)
        Rp = np.asarray(Tp.rotation().as_matrix(), dtype=np.float64)
        # Linear velocity
        J[:3, i] = (pp - p0) / eps
        # Angular velocity: skew-symmetric extraction (exact for small angles)
        dR = Rp @ R0.T
        J[3:, i] = np.array([
            dR[2, 1] - dR[1, 2],
            dR[0, 2] - dR[2, 0],
            dR[1, 0] - dR[0, 1],
        ]) / (2 * eps)
    return J

# Constants
DEFAULT_IP = "172.16.0.2"
HOME_Q = np.array([0.0, -0.785398, 0.0, -2.356194, -0.15, 1.570796, 0.785398])

JOINT_LIMITS_LO = np.array([-2.7437, -1.7837, -2.9007, -3.0421, -2.8065, 0.5445, -3.0159])
JOINT_LIMITS_HI = np.array([2.7437, 1.7837, 2.9007, -0.1518, 2.8065, 4.5169, 3.0159])

# Step sizes
STEP_TRANS = 0.002       # 2 mm per keystroke
STEP_ROT = 0.015         # ~0.86 deg per keystroke
MIN_STEP = 0.0005
MAX_STEP = 0.010

# Gamepad
JS_DEVICE = "/dev/input/js0"
DEADZONE = 4000
GAMEPAD_SPEED = 0.06     # m/s
GAMEPAD_SLOW = 0.03
GAMEPAD_FAST = 0.12
GAMEPAD_ROT_SPEED = 0.4  # rad/s
GAMEPAD_DT = 0.05        # 20 Hz gamepad loop

# Rate limit
LOOP_DT = 0.02  # 50 Hz keyboard loop

# Damping for pseudo-inverse
LAMBDA_DAMP = 0.05

# Keyboard input

# Switch stdin to raw mode for non-blocking key reads.
def _setup_keyboard():
    fd = sys.stdin.fileno()
    old = termios.tcgetattr(fd)
    return fd, old


# Context manager that keeps stdin in raw mode for the entire session.
class RawTerminal:

    def __init__(self):
        self.fd = sys.stdin.fileno()
        self.old = termios.tcgetattr(self.fd)

    def __enter__(self):
        tty.setraw(self.fd)
        return self

    def __exit__(self, *_):
        termios.tcsetattr(self.fd, termios.TCSADRAIN, self.old)


# Read a single keypress (non-blocking). Assumes stdin is already raw.
def _get_key(timeout=0.02):
    rlist, _, _ = select.select([sys.stdin], [], [], timeout)
    if not rlist:
        return None
    ch = sys.stdin.read(1)
    if ch == "\x1b":
        rlist2, _, _ = select.select([sys.stdin], [], [], 0.05)
        if rlist2:
            ch2 = sys.stdin.read(1)
            if ch2 == "[":
                rlist3, _, _ = select.select([sys.stdin], [], [], 0.05)
                if rlist3:
                    ch3 = sys.stdin.read(1)
                    return "\x1b[" + ch3
                return "\x1b["
            return "\x1b" + ch2
        return "ESC"
    return ch


# Jacobian helpers

# Damped pseudo-inverse: J^T (J J^T + lambda^2 I)^{-1}.
def damped_pinv(J: np.ndarray, lam: float = LAMBDA_DAMP) -> np.ndarray:
    m = J.shape[0]
    return J.T @ np.linalg.inv(J @ J.T + lam**2 * np.eye(m))


# Hard-clamp to joint limits with a small margin.
def clamp_joints(q: np.ndarray) -> np.ndarray:
    margin = 0.02
    return np.clip(q, JOINT_LIMITS_LO + margin, JOINT_LIMITS_HI - margin)


# Return True if all joints are within limits (with margin).
def check_joint_limits(q: np.ndarray) -> bool:
    margin = 0.01
    return bool(np.all(q >= JOINT_LIMITS_LO + margin) and np.all(q <= JOINT_LIMITS_HI - margin))


def apply_cartesian_delta(q: np.ndarray, dx: np.ndarray) -> np.ndarray | None:
    """
    Compute new joint config for a 6-DOF Cartesian delta [vx,vy,vz,wx,wy,wz].

    Returns new q or None if the motion would violate joint limits.
    """
    J = jacobian(q)
    Jpinv = damped_pinv(J)
    dq = Jpinv @ dx

    # Scale down if any joint would move too much in one step
    max_dq = np.max(np.abs(dq))
    if max_dq > 0.05:
        dq *= 0.05 / max_dq

    q_new = q + dq
    q_new = clamp_joints(q_new)

    if not check_joint_limits(q_new):
        return None
    return q_new


# TCP printing

# Format TCP position + quaternion for display.
def format_tcp(q: np.ndarray) -> str:
    pos, R = fk(q)
    quat = Rotation.from_matrix(R).as_quat()  # [x, y, z, w]
    return (
        f"pos: [{pos[0]:+.4f}, {pos[1]:+.4f}, {pos[2]:+.4f}]  "
        f"quat(xyzw): [{quat[0]:+.4f}, {quat[1]:+.4f}, {quat[2]:+.4f}, {quat[3]:+.4f}]"
    )


# Format joint positions for display.
def format_joints(q: np.ndarray) -> str:
    return "joints: [" + ", ".join(f"{v:+.4f}" for v in q) + "]"


# Robot connection

# Connect to the FR3, recover errors, return (robot, gripper).
def connect_robot(ip: str):
    if Robot is None:
        raise ImportError("franky is not installed; FR3 teleop unavailable")
    print(f"Connecting to FR3 at {ip}...")
    try:
        robot = Robot(ip)
    except Exception as e:
        if "realtime" in str(e).lower():
            print(f"  Non-RT kernel, using RealtimeConfig.Ignore")
            robot = Robot(ip, realtime_config=RealtimeConfig.Ignore)
        else:
            raise

    if robot.has_errors:
        print("  Recovering from errors...")
        robot.recover_from_errors()

    try:
        robot.set_collision_behavior(100.0, 100.0)
    except Exception:
        pass

    gripper = Gripper(ip)
    print("  Connected.")
    return robot, gripper


# Read current joint positions from the robot.
def get_current_q(robot) -> np.ndarray:
    return np.asarray(robot.current_joint_state.position, dtype=float)


# Send a joint target via async JointWaypointMotion.
def send_joint_target(robot, q: np.ndarray, velocity: float = 0.3):
    if robot.has_errors:
        try:
            robot.recover_from_errors()
        except Exception:
            pass

    rdf = RelativeDynamicsFactor(velocity, velocity * 0.5, velocity * 0.25)
    wp = JointWaypoint(q.tolist())
    motion = JointWaypointMotion(
        [wp],
        relative_dynamics_factor=rdf,
        return_when_finished=False,
    )
    try:
        robot.move(motion, asynchronous=True)
    except Exception as exc:
        msg = str(exc).lower()
        if "motion finished commanded" in msg or "discontinuity" in msg:
            try:
                robot.recover_from_errors()
            except Exception:
                pass
        else:
            raise


# Stop all robot motion cleanly.
def stop_robot(robot):
    try:
        robot.move(JointStopMotion(), asynchronous=False)
    except Exception:
        pass


# Move to home configuration (blocking).
def go_home(robot, gripper):
    stop_robot(robot)
    if robot.has_errors:
        try:
            robot.recover_from_errors()
        except Exception:
            pass
    time.sleep(0.1)

    rdf = RelativeDynamicsFactor(0.2, 0.1, 0.05)
    motion = JointMotion(HOME_Q.tolist(), relative_dynamics_factor=rdf)
    try:
        robot.move(motion, asynchronous=False)
    except Exception as exc:
        print(f"  Home failed: {exc}")
        try:
            robot.recover_from_errors()
        except Exception:
            pass


# Open the gripper.
def gripper_open(gripper):
    try:
        gripper.open(0.1)
    except Exception as exc:
        print(f"  Gripper open failed: {exc}")


# Close the gripper (grasp).
def gripper_close(gripper):
    try:
        mw = gripper.max_width
        gripper.grasp(0.0, 0.1, 70.0, mw, mw)
    except Exception as exc:
        print(f"  Gripper close failed: {exc}")


# Keyboard teleop

# Convert angular velocity from tool frame to base frame.
def _tool_frame_angular(q: np.ndarray, omega_tool: np.ndarray) -> np.ndarray:
    _, R = fk(q)
    return R @ omega_tool


# Main keyboard teleop loop.
def run_keyboard(robot, gripper):
    running = True
    step_trans = STEP_TRANS
    step_rot = STEP_ROT

    def quit_handler(sig, frame):
        nonlocal running
        running = False

    signal.signal(signal.SIGINT, quit_handler)
    signal.signal(signal.SIGTERM, quit_handler)

    q = get_current_q(robot)
    print()
    print(f"  {format_tcp(q)}")
    print(f"  {format_joints(q)}")
    print(f"  Step: {step_trans*1000:.1f} mm / {np.degrees(step_rot):.1f} deg")
    print()
    print("Controls:")
    print("  W/S = X  |  A/D = Y  |  Q/E = Z")
    print("  Arrows = pitch/yaw (base frame)  |  ,/. = roll (tool Z axis)")
    print("  O = open  |  C = close  |  P = print TCP  |  H = home  |  +/- = step size")
    print("  ESC = quit")
    print()

    last_move_time = 0.0

    with RawTerminal():
        try:
            while running:
                key = _get_key(timeout=LOOP_DT)
                if key is None:
                    continue

                dx = np.zeros(6)
                action = None

                if key == "ESC":
                    running = False
                    continue

                # Translation
                if key in ("w", "W"):
                    dx[0] = step_trans
                elif key in ("s", "S"):
                    dx[0] = -step_trans
                elif key in ("a", "A"):
                    dx[1] = step_trans
                elif key in ("d", "D"):
                    dx[1] = -step_trans
                elif key == "q":
                    dx[2] = step_trans
                elif key == "e":
                    dx[2] = -step_trans

                # Rotation via arrow keys (base frame pitch/yaw)
                elif key == "\x1b[A":
                    dx[4] = step_rot
                elif key == "\x1b[B":
                    dx[4] = -step_rot
                elif key == "\x1b[C":
                    dx[5] = -step_rot
                elif key == "\x1b[D":
                    dx[5] = step_rot

                # Roll around tool Z axis
                elif key in (",", "<"):
                    q = get_current_q(robot)
                    omega_base = _tool_frame_angular(q, np.array([0.0, 0.0, step_rot]))
                    dx[3:6] = omega_base
                elif key in (".", ">"):
                    q = get_current_q(robot)
                    omega_base = _tool_frame_angular(q, np.array([0.0, 0.0, -step_rot]))
                    dx[3:6] = omega_base

                # Gripper
                elif key in ("o", "O"):
                    action = "open"
                elif key in ("c", "C"):
                    action = "close"

                # Print TCP
                elif key in ("p", "P"):
                    action = "print"

                # Home
                elif key in ("h", "H"):
                    action = "home"

                # Step size
                elif key in ("+", "="):
                    step_trans = min(step_trans * 1.5, MAX_STEP)
                    step_rot = min(step_rot * 1.5, 0.10)
                    print(f"  Step: {step_trans*1000:.1f} mm / {np.degrees(step_rot):.1f} deg")
                    continue
                elif key in ("-", "_"):
                    step_trans = max(step_trans / 1.5, MIN_STEP)
                    step_rot = max(step_rot / 1.5, 0.003)
                    print(f"  Step: {step_trans*1000:.1f} mm / {np.degrees(step_rot):.1f} deg")
                    continue

                # Handle actions
                if action == "open":
                    stop_robot(robot)
                    print("  Opening gripper...")
                    gripper_open(gripper)
                    q = get_current_q(robot)
                    print(f"  {format_tcp(q)}")
                    continue
                elif action == "close":
                    stop_robot(robot)
                    print("  Closing gripper...")
                    gripper_close(gripper)
                    q = get_current_q(robot)
                    print(f"  {format_tcp(q)}")
                    continue
                elif action == "print":
                    q = get_current_q(robot)
                    print(f"  {format_tcp(q)}")
                    print(f"  {format_joints(q)}")
                    gw = "?"
                    try:
                        gw = f"{gripper.width*1000:.1f} mm"
                    except Exception:
                        pass
                    print(f"  gripper width: {gw}")
                    continue
                elif action == "home":
                    print("  Going home...")
                    go_home(robot, gripper)
                    q = get_current_q(robot)
                    print(f"  {format_tcp(q)}")
                    continue

                # Apply Cartesian delta if nonzero
                if np.any(dx != 0):
                    now = time.time()
                    if now - last_move_time < 0.01:
                        continue
                    last_move_time = now

                    q = get_current_q(robot)
                    q_new = apply_cartesian_delta(q, dx)
                    if q_new is not None:
                        send_joint_target(robot, q_new, velocity=0.3)
                    else:
                        print("  (joint limit reached)")

        except Exception as exc:
            print(f"\nError: {exc}")
    print()


# Gamepad teleop

# Read one joystick event (non-blocking). Returns (type, number, value) or None.
def _read_js_event(fd):
    buf = fd.read(8)
    if buf is None or len(buf) < 8:
        return None
    _t, val, typ, num = struct.unpack("IhBB", buf)
    return typ & 0x7F, num, val


# Main gamepad teleop loop.
def run_gamepad(robot, gripper, device=JS_DEVICE):
    if not os.path.exists(device):
        print(f"Gamepad device not found: {device}")
        print("Plug in a gamepad and try again, or use keyboard mode (no --gamepad flag).")
        sys.exit(1)

    running = True

    def quit_handler(sig, frame):
        nonlocal running
        running = False

    signal.signal(signal.SIGINT, quit_handler)
    signal.signal(signal.SIGTERM, quit_handler)

    js = open(device, "rb")
    os.set_blocking(js.fileno(), False)
    time.sleep(0.3)
    while _read_js_event(js) is not None:
        pass

    q = get_current_q(robot)
    print()
    print(f"  {format_tcp(q)}")
    print(f"  {format_joints(q)}")
    print()
    print("Gamepad controls:")
    print("  LStick      = XY motion      RStick L/R  = yaw")
    print("  RStick U/D  = roll            D-pad L/R   = pitch")
    print("  D-pad U/D   = Z up / down    L2 / R2     = Z down / up")
    print("  L1          = gripper close   R1          = gripper open")
    print("  A/B/X/Y     = print TCP        Select      = home")
    print("  Start       = quit")
    print()

    # Exact mapping from js_test.py
    # Axes:  LX=0  LY=1  RX=2  RY=3  DpadX=4  DpadY=5
    # Buttons: L1=4  R1=5  L2=6  R2=7  Select=8  Start=9
    AX_LX, AX_LY = 0, 1
    AX_RX, AX_RY = 2, 3
    AX_DPAD_X, AX_DPAD_Y = 4, 5

    BTN_L1, BTN_R1 = 4, 5
    BTN_L2, BTN_R2 = 6, 7
    BTN_SELECT, BTN_START = 8, 9
    # Face buttons (0-3), use any for TCP print
    BTN_TCP_PRINT = {0, 1, 2, 3}

    axes = {}
    buttons = {}
    last_print = time.time()
    speed = GAMEPAD_SPEED
    _tcp_pub = [0.0]

    while running:
        while True:
            evt = _read_js_event(js)
            if evt is None:
                break
            typ, num, val = evt
            if typ == 1:
                buttons[num] = val
                if num == BTN_L1 and val == 1:
                    stop_robot(robot)
                    print("  Closing gripper...")
                    gripper_close(gripper)
                    q = get_current_q(robot)
                    print(f"  {format_tcp(q)}")
                elif num == BTN_R1 and val == 1:
                    stop_robot(robot)
                    print("  Opening gripper...")
                    gripper_open(gripper)
                    q = get_current_q(robot)
                    print(f"  {format_tcp(q)}")
                elif num == BTN_SELECT and val == 1:
                    print("  Going home...")
                    go_home(robot, gripper)
                    q = get_current_q(robot)
                    print(f"  {format_tcp(q)}")
                elif num in BTN_TCP_PRINT and val == 1:
                    q = get_current_q(robot)
                    print(f"  {format_tcp(q)}")
                    print(f"  {format_joints(q)}")
                elif num == BTN_START and val == 1:
                    running = False
            elif typ == 2:
                axes[num] = val

        # Sticks with deadzone
        lx = axes.get(AX_LX, 0)
        ly = axes.get(AX_LY, 0)
        rx = axes.get(AX_RX, 0)
        ry = axes.get(AX_RY, 0)
        lx = 0 if abs(lx) < DEADZONE else lx
        ly = 0 if abs(ly) < DEADZONE else ly
        rx = 0 if abs(rx) < DEADZONE else rx
        ry = 0 if abs(ry) < DEADZONE else ry

        # D-pad (axes 4,5, latch at -32767/+32767, return to 0 on release)
        dpad_x = axes.get(AX_DPAD_X, 0)
        dpad_y = axes.get(AX_DPAD_Y, 0)

        # L2/R2 are digital buttons (held = 1, released = 0)
        l2_held = buttons.get(BTN_L2, 0)
        r2_held = buttons.get(BTN_R2, 0)

        dx = np.zeros(6)

        # Left stick -> XY continuous
        dx[0] = (-ly / 32767.0) * speed * GAMEPAD_DT
        dx[1] = (-lx / 32767.0) * speed * GAMEPAD_DT

        # Right stick L/R -> yaw, U/D -> roll (as observed on controller)
        dx[5] = (-rx / 32767.0) * GAMEPAD_ROT_SPEED * GAMEPAD_DT
        dx[3] = (-ry / 32767.0) * GAMEPAD_ROT_SPEED * GAMEPAD_DT

        # D-pad left/right -> pitch
        if abs(dpad_x) > DEADZONE:
            dx[4] += (-1.0 if dpad_x > 0 else 1.0) * STEP_ROT * 2

        # D-pad up/down -> Z up/down
        if abs(dpad_y) > DEADZONE:
            dx[2] += (-1.0 if dpad_y > 0 else 1.0) * STEP_TRANS * 2

        # L2/R2 -> Z (while held)
        if l2_held:
            dx[2] -= speed * GAMEPAD_DT
        if r2_held:
            dx[2] += speed * GAMEPAD_DT

        if np.max(np.abs(dx)) > 1e-5:
            q = get_current_q(robot)
            q_new = apply_cartesian_delta(q, dx)
            if q_new is not None:
                send_joint_target(robot, q_new, velocity=0.3)

        # Publish the live TCP for other tools (the calibration app reads
        # this when the server's robot reads return null because THIS
        # process owns the libfranka connection). Same fk module as the
        # control stack, so the frame matches. Atomic write, ~5 Hz.
        now = time.time()
        if now - _tcp_pub[0] > 0.2:
            _tcp_pub[0] = now
            try:
                pos, _ = fk(get_current_q(robot))
                tmp = TCP_PUB_PATH + ".tmp"
                with open(tmp, "w") as fh:
                    json.dump({"tcp_xyz": [float(v) for v in pos],
                               "ts": now}, fh)
                os.replace(tmp, TCP_PUB_PATH)
            except Exception:
                pass

        time.sleep(GAMEPAD_DT)

    js.close()


# Main

def main():
    parser = argparse.ArgumentParser(
        description="Teleop FR3 via franky + Jacobian IK",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    parser.add_argument("--ip", default=DEFAULT_IP,
                        help=f"Robot IP (default: {DEFAULT_IP})")
    parser.add_argument("--gamepad", action="store_true",
                        help="Use gamepad instead of keyboard")
    parser.add_argument("--js-device", default=JS_DEVICE,
                        help=f"Joystick device (default: {JS_DEVICE})")
    args = parser.parse_args()

    # Warm up pyroki FK/Jacobian (JIT compile)
    print("Warming up pyroki FK/Jacobian...")
    t0 = time.time()
    _warmup_q = HOME_Q.copy()
    _ = fk(_warmup_q)
    _ = jacobian(_warmup_q)
    print(f"  JIT warm-up done in {time.time()-t0:.1f}s")

    # Connect
    robot, gripper = connect_robot(args.ip)

    # Print startup state
    q = get_current_q(robot)
    print(f"  {format_tcp(q)}")
    print(f"  {format_joints(q)}")

    try:
        if args.gamepad:
            run_gamepad(robot, gripper, device=args.js_device)
        else:
            run_keyboard(robot, gripper)
    finally:
        print("\nStopping robot...")
        stop_robot(robot)
        try:
            robot.stop()
        except Exception:
            pass
        print("Done.")


if __name__ == "__main__":
    main()

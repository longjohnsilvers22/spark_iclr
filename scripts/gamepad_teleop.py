#!/usr/bin/env python3
"""Gamepad teleop for Franka FR3 via franky (no server needed).

Controls (HORIPAD / Xbox-style):
  Left stick X/Y  : move in XY plane
  Right stick Y    : move in Z
  A (btn 304)      : close gripper
  B (btn 305)      : open gripper
  X (btn 307)      : home
  Y (btn 308)      : print TCP
  LB (btn 310)     : slow mode (hold)
  RB (btn 311)     : fast mode (hold)
  Start (btn 315)  : quit

Usage:
  ~/spark/scripts/controller.sh franky
  cd ~/spark/src && python ../scripts/gamepad_teleop.py
"""

import os
import struct
import signal
import sys
import time

import numpy as np
from franky import (
    CartesianVelocityMotion, CartesianVelocityStopMotion, Gripper,
    JointMotion, JointStopMotion, RelativeDynamicsFactor, Robot,
    RobotVelocity, Twist,
)

ROBOT_IP = os.environ.get("FRANKA_IP", "172.16.0.2")
DEADZONE = 4000
BASE_SPEED = 0.06
SLOW_SPEED = 0.03
FAST_SPEED = 0.12
DT = 0.05
JS_DEVICE = "/dev/input/js0"
HOME_Q = [0, -0.785398, 0, -2.356194, -0.15, 1.570796, 0.785398]


def read_js_event(fd):
    buf = fd.read(8)
    if buf is None or len(buf) < 8:
        return None
    t, val, typ, num = struct.unpack("IhBB", buf)
    return typ & 0x7f, num, val


def main():
    print(f"Connecting to {ROBOT_IP}...")
    robot = Robot(ROBOT_IP)
    robot.recover_from_errors()
    gripper = Gripper(ROBOT_IP)

    try:
        robot.set_collision_behavior(100.0, 100.0)
    except Exception:
        pass

    def get_tcp():
        ee = robot.current_cartesian_state.pose.end_effector_pose
        return np.array(ee.translation)

    def print_tcp():
        tcp = get_tcp()
        print(f"  TCP: x={tcp[0]:.4f}  y={tcp[1]:.4f}  z={tcp[2]*1000:.1f}mm")

    print("Connected.")
    print_tcp()

    axes = {}
    buttons = {}
    running = True
    vel_active = False

    def quit_handler(sig, frame):
        nonlocal running
        running = False
    signal.signal(signal.SIGINT, quit_handler)
    signal.signal(signal.SIGTERM, quit_handler)

    js = open(JS_DEVICE, "rb")
    os.set_blocking(js.fileno(), False)
    time.sleep(0.1)
    while read_js_event(js) is not None:
        pass

    print("Teleop active. LStick=XY, RStick-Y=Z, A=close, B=open, X=home, Y=TCP, Start=quit")

    last_print = time.time()

    while running:
        while True:
            evt = read_js_event(js)
            if evt is None:
                break
            typ, num, val = evt
            if typ == 1:
                buttons[num] = val
                if num == 304 and val == 1:
                    print("Closing gripper...")
                    try:
                        robot.move(CartesianVelocityStopMotion())
                    except Exception:
                        pass
                    vel_active = False
                    try:
                        mw = gripper.max_width
                        gripper.grasp(0.0, 0.1, 70.0, mw, mw)
                    except Exception:
                        pass
                    print_tcp()
                elif num == 305 and val == 1:
                    print("Opening gripper...")
                    try:
                        robot.move(CartesianVelocityStopMotion())
                    except Exception:
                        pass
                    vel_active = False
                    try:
                        gripper.open(0.1)
                    except Exception:
                        pass
                    print_tcp()
                elif num == 307 and val == 1:
                    print("Homing...")
                    try:
                        try:
                            robot.move(CartesianVelocityStopMotion())
                        except Exception:
                            pass
                        vel_active = False
                        robot.recover_from_errors()
                        try:
                            robot.move(JointStopMotion())
                        except Exception:
                            pass
                        rdf = RelativeDynamicsFactor(0.2, 0.1, 0.05)
                        robot.move(JointMotion(HOME_Q, relative_dynamics_factor=rdf))
                    except Exception as e:
                        print(f"  Home failed: {e}")
                    print_tcp()
                elif num == 308 and val == 1:
                    print_tcp()
                elif num == 315 and val == 1:
                    running = False
            elif typ == 2:
                axes[num] = val

        lx = axes.get(0, 0)
        ly = axes.get(1, 0)
        rz = axes.get(4, 0)

        if abs(lx) < DEADZONE: lx = 0
        if abs(ly) < DEADZONE: ly = 0
        if abs(rz) < DEADZONE: rz = 0

        if lx == 0 and ly == 0 and rz == 0:
            if vel_active:
                try:
                    robot.move(CartesianVelocityStopMotion())
                except Exception:
                    pass
                vel_active = False
                print_tcp()
            now = time.time()
            if now - last_print > 3.0:
                print_tcp()
                last_print = now
            time.sleep(0.01)
            continue

        nx = lx / 32767.0
        ny = -ly / 32767.0
        nz = -rz / 32767.0

        speed = BASE_SPEED
        if buttons.get(310, 0):
            speed = SLOW_SPEED
        if buttons.get(311, 0):
            speed = FAST_SPEED

        twist = Twist([nx * speed, ny * speed, nz * speed], [0, 0, 0])
        robot_vel = RobotVelocity(twist)

        if robot.has_errors:
            try:
                robot.recover_from_errors()
            except Exception:
                pass
            vel_active = False
            time.sleep(0.05)
            continue

        try:
            motion = CartesianVelocityMotion(
                robot_vel, relative_dynamics_factor=0.15)
            robot.move(motion, asynchronous=True)
            vel_active = True
        except Exception:
            pass

        now = time.time()
        if now - last_print > 0.5:
            print_tcp()
            last_print = now

        time.sleep(DT)

    if vel_active:
        try:
            robot.move(CartesianVelocityStopMotion())
        except Exception:
            pass
    js.close()
    try:
        robot.stop()
    except Exception:
        pass
    print("Teleop ended.")


if __name__ == "__main__":
    main()

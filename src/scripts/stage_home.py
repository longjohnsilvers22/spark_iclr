"""Gently home both bimanual arms with a staged joint-space interpolation,
for when a single go_home is too big a move (recovery after a bad pose).
No gripper changes. Run on the bimanual host."""
import os
import sys
import time
import threading

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
os.environ.setdefault("JAX_PLATFORMS", "cpu")

from spark_real.config import load_family_yaml
from spark_real.robots.franka.franka_bamboo_driver import FrankaBambooDriver

# Machine overlay name (configs/machines/<name>.yaml); optional first argv.
MACHINE = sys.argv[1] if len(sys.argv) > 1 else "ANON-LAB"

HOME = np.array([0.0, -0.785, 0.0, -2.356, 0.0, 1.571, 0.785])
N_STEPS = 8


def stage_home(robot, name):
    q0 = np.array(robot.get_joint_positions(), dtype=float)[:7]
    dmax = float(np.max(np.abs(HOME - q0)))
    print(f"{name}: max delta to home = {dmax:.2f} rad, staging in {N_STEPS}")
    for i in range(1, N_STEPS + 1):
        qi = q0 + (i / N_STEPS) * (HOME - q0)
        try:
            robot.move_to_joint_config(qi.tolist(), velocity=0.25)
        except Exception as exc:
            print(f"  {name} step {i} failed: {exc}; retrying slower")
            robot.move_to_joint_config(qi.tolist(), velocity=0.12)
    qf = np.array(robot.get_joint_positions(), dtype=float)[:7]
    print(f"{name} final: {[round(float(x), 3) for x in qf]} "
          f"(|err|={float(np.max(np.abs(qf - HOME))):.3f})")


def main():
    robot_cfg = load_family_yaml("bimanual_franka", MACHINE).get("robot", {})
    left_ip = robot_cfg.get("left_ip", "172.16.0.101")
    right_ip = robot_cfg.get("right_ip", "172.16.0.102")
    l = FrankaBambooDriver(ip=left_ip, port=5555)
    r = FrankaBambooDriver(ip=right_ip, port=5556)
    l.connect()
    r.connect()
    try:
        tl = threading.Thread(target=lambda: stage_home(l, "LEFT"))
        tr = threading.Thread(target=lambda: stage_home(r, "RIGHT"))
        tl.start(); tr.start(); tl.join(); tr.join()
    finally:
        l.disconnect()
        r.disconnect()
    print("Staged home complete.")


if __name__ == "__main__":
    main()

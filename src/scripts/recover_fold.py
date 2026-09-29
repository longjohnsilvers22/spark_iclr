"""Recover after an interrupted fold: open both grippers, home both arms.
Motion only: gripper open + go_home. Run on the bimanual host."""
import os
import sys
import threading
import time

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
os.environ.setdefault("JAX_PLATFORMS", "cpu")

from spark_real.config import load_family_yaml
from spark_real.robots.franka.franka_bamboo_driver import FrankaBambooDriver
from spark_real.robots.bimanual_franka.bimanual_franka_driver import make_dual_gripper

# Machine overlay name (configs/machines/<name>.yaml); optional first argv.
MACHINE = sys.argv[1] if len(sys.argv) > 1 else "ANON-LAB"


def main():
    print("Opening grippers...")
    cfg = load_family_yaml("bimanual_franka", MACHINE)
    grip = make_dual_gripper(cfg.get("grippers", {}))
    grip.connect()
    time.sleep(0.5)
    for a in ("left", "right"):
        grip.for_arm(a).send_pack_locked(0, 255, 500, 1, 1, 0, 0)
    time.sleep(1.5)
    grip.disconnect()
    print("Grippers open.")

    print("Homing arms...")
    robot_cfg = cfg.get("robot", {})
    left_ip = robot_cfg.get("left_ip", "172.16.0.101")
    right_ip = robot_cfg.get("right_ip", "172.16.0.102")
    l = FrankaBambooDriver(ip=left_ip, port=5555)
    r = FrankaBambooDriver(ip=right_ip, port=5556)
    l.connect()
    r.connect()
    try:
        tl = threading.Thread(target=lambda: l.go_home(velocity=0.2))
        tr = threading.Thread(target=lambda: r.go_home(velocity=0.2))
        tl.start()
        tr.start()
        tl.join()
        tr.join()
        print("L homed:", [round(float(x), 3) for x in l.get_joint_positions()])
        print("R homed:", [round(float(x), 3) for x in r.get_joint_positions()])
    finally:
        l.disconnect()
        r.disconnect()
    print("Recovered.")


if __name__ == "__main__":
    main()

#!/usr/bin/env python3
# Register the SSG-48 payload with both Frankas via franky setLoad, so the
# idle gravity compensation carries the gripper (fixes the slow sag of
# unheld extended arms; libfranka persists the load until changed/reboot).
# Run with NO other FCI client connected (stop servers/holders first).
#
#   $PY scripts/set_payload.py [machine]     # configs/machines/<machine>.yaml
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from spark_real.config import load_family_yaml

cfg = load_family_yaml("bimanual_franka", sys.argv[1] if len(sys.argv) > 1 else "ANON-LAB")
pl = (cfg.get("grippers") or {}).get("payload") or {}
mass = float(pl.get("mass_kg", 0.7))
com = [float(v) for v in pl.get("com_m", [0.0, 0.0, 0.08])]
inertia = [0.001, 0, 0, 0, 0.001, 0, 0, 0, 0.001]

robot_cfg = cfg.get("robot") or {}
left_ip = robot_cfg.get("left_ip", "172.16.0.101")
right_ip = robot_cfg.get("right_ip", "172.16.0.102")

from franky import Robot
for name, ip in (("PANDA-left", left_ip), ("FR3-right", right_ip)):
    try:
        r = Robot(ip)
        r.set_load(mass, com, inertia)
        print(f"{name}: set_load mass={mass}kg com={com}")
    except Exception as e:
        print(f"{name}: FAILED {type(e).__name__}: {e}")
        sys.exit(1)
print("payload registered on both arms")

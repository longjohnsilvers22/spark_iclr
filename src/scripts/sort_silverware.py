#!/usr/bin/env python3
"""Bimanual silverware sort (FrankaBambooDriver + PyRoki IK).

Both arms pick silverware from the table and place it into a shared tray.
Each arm picks from its own reachable side.

CRITICAL: Camera-frame sides are FLIPPED from robot sides!
  - Camera LEFT  = FR3   (RIGHT robot, robot.right_ip, bamboo port 5556)
  - Camera RIGHT = Panda (LEFT robot,  robot.left_ip,  bamboo port 5555)

So when SAM3 detects items on the "left side" of the image, those are
on the RIGHT robot's side. The RIGHT robot picks those. And vice versa.

Arm IPs and gripper buses come from the machine overlay
configs/machines/<machine>.yaml deep-merged over bimanual_franka_default.yaml
(spark_real.config.load_family_yaml). The tracked template is
configs/machines/ANON-LAB.example.yaml; the real overlay is gitignored.

Pipeline:
  Phase 0: Detect silverware + tray via SAM3 (src/scripts/detect.py)
  Phase 1: Assign items to arms (LEFT arm = camera-right items,
           RIGHT arm = camera-left items)
  Phase 2: Pick-and-place each item into tray (one arm at a time, or
           simultaneous if items are well-separated)
  Phase 3: Home both arms

Usage:
    cd ~/spark/src
    export JAX_PLATFORMS=cpu
    PY=$(conda run -n spark_conda which python3)

    $PY scripts/sort_silverware.py                    # full pipeline
    $PY scripts/sort_silverware.py --skip-detect      # reuse last detection
    $PY scripts/sort_silverware.py --dry-run          # IK validation only
    $PY scripts/sort_silverware.py --sequential       # one arm at a time
    $PY scripts/sort_silverware.py --machine myhost   # other overlay
"""
import argparse
import sys, os, json, time, threading, subprocess
import numpy as np
from scipy.spatial.transform import Rotation

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
os.environ.setdefault("JAX_PLATFORMS", "cpu")

from spark_real.config import load_family_yaml
from spark_real.robots.franka.franka_bamboo_driver import FrankaBambooDriver
from spark_real.robots.bimanual_franka.bimanual_franka_driver import (
    BimanualFrankaDriver as _BFD,
    make_dual_gripper,
)

# --- Constants ---
HOME_Q = np.array([0.0, -0.785, 0.0, -2.356, 0.0, 1.571, 0.785])
PYROKI_Z_OFFSET = 0.103

# Top-down orientation (yaw=0)
R_TD = np.array([[1, 0, 0], [0, -1, 0], [0, 0, -1]], dtype=float)

SAFE_Z = 0.10
APPROACH_Z = 0.05
GRASP_Z = -0.015
LIFT_Z = 0.06
PLACE_Z = 0.03

MOVE_VEL = 0.13
APPROACH_VEL = 0.06

GRASP_FORCE = 60

DET_PATH = os.path.expanduser("~/.spark_real/detections/detections.json")
DETECT_PY = os.path.join(os.path.dirname(os.path.abspath(__file__)), "detect.py")

# --- IK helper ---
_planner = None
def _get_planner():
    global _planner
    if _planner is None:
        from spark_real.control.pyroki_planner import BimanualPyrokiPlanner
        _planner = BimanualPyrokiPlanner()
    return _planner


def make_orientation(yaw):
    """Top-down orientation with given yaw (rotation around base Z)."""
    c, s = np.cos(yaw), np.sin(yaw)
    Rz = np.array([[c, -s, 0], [s, c, 0], [0, 0, 1]])
    R = Rz @ R_TD
    q_xyzw = Rotation.from_matrix(R).as_quat()
    return np.array([float(q_xyzw[3]), float(q_xyzw[0]),
                     float(q_xyzw[1]), float(q_xyzw[2])])


def ik(arm, pos, q_seed, yaw=0.0):
    """PyRoki IK via BimanualPyrokiPlanner. Returns joint config or None."""
    t = np.array(pos, dtype=float).copy()
    t[2] += PYROKI_Z_OFFSET
    orient = make_orientation(yaw)
    try:
        q = _get_planner().solve(arm=arm, target_position_base=t,
                                  target_wxyz_base=orient, prev_cfg=q_seed)[:7]
        return q
    except Exception:
        return None


# --- Gripper helpers ---
def grippers(cfg):
    dual = make_dual_gripper(cfg.get("grippers", {}))
    dual.connect()
    time.sleep(0.5)
    return dual


def grip_open(grip, arms=("left", "right")):
    for a in arms:
        grip.for_arm(a).send_pack_locked(0, 255, 500, 1, 1, 0, 0)
    time.sleep(1.0)


def grip_close(grip, arms=("left", "right")):
    for a in arms:
        grip.for_arm(a).send_pack_locked(255, 255, 1300, 1, 1, 0, 0)
    time.sleep(2.0)
    for a in arms:
        raw = getattr(grip.for_arm(a)._motor, "gripper_position", None)
        print(f"    {a}: raw={raw}")


def grip_check(grip):
    for a in ["left", "right"]:
        raw = getattr(grip.for_arm(a)._motor, "gripper_position", None)
        state = "CLOSED" if raw and raw > 100 else "OPEN"
        print(f"    {a}: raw={raw} [{state}]")


# --- Detection ---
def run_detection():
    """Run SAM3 detection for silverware + tray."""
    # TODO: Tune prompts for actual scene. Try multiple item labels.
    # SAM3 works best with specific object names. If the scene has
    # specific pieces (e.g., "butter knife", "dessert fork"), add those.
    prompts = ["fork", "knife", "spoon", "tray"]
    print(f"Running SAM3 detection: {prompts}")
    subprocess.run([sys.executable, DETECT_PY] + prompts +
                   ["--grab", "--annotate"],
                   capture_output=True, text=True)


def load_detections():
    """Load detection results."""
    det = json.loads(open(DET_PATH).read())
    d = {e["label"]: e for e in det}
    if "tray" not in d:
        print(f"  MISSING: tray. Found: {list(d.keys())}")
        return None
    silverware = {k: v for k, v in d.items() if k != "tray"}
    if not silverware:
        print(f"  MISSING: no silverware detected. Found: {list(d.keys())}")
        return None
    return d


def assign_items_to_arms(d):
    """Assign silverware items to LEFT or RIGHT arm based on position.

    CRITICAL FRAME FLIP:
      Items on the camera-LEFT side are on the RIGHT robot's side.
      Items on the camera-RIGHT side are on the LEFT robot's side.

    The Y coordinate in the RIGHT arm's frame determines the side:
      - RIGHT frame Y > 0 = camera-right side = LEFT robot picks
      - RIGHT frame Y < 0 = camera-left side  = RIGHT robot picks

    Each arm gets items as coordinates in ITS OWN frame for IK.
    """
    tray_r = np.array(d["tray"]["right_frame"])
    tray_l = np.array(d["tray"]["left_frame"])

    # Workspace midline in RIGHT frame Y. Items with Y > midline are
    # on the Panda (LEFT robot) side; Y < midline on the FR3 (RIGHT robot) side.
    # Using Y=0 as the midline (robot base center).
    midline_y = 0.0

    left_arm_items = []   # items the LEFT robot (Panda) will pick
    right_arm_items = []  # items the RIGHT robot (FR3) will pick

    for label, entry in d.items():
        if label == "tray":
            continue
        pos_r = np.array(entry["right_frame"])
        pos_l = np.array(entry["left_frame"])

        # Determine which side in camera frame.
        # RIGHT frame Y > 0 = camera right = LEFT robot's reachable side.
        # RIGHT frame Y < 0 = camera left  = RIGHT robot's reachable side.
        if pos_r[1] > midline_y:
            left_arm_items.append({
                "label": label,
                "pos": pos_l,
                "pos_other": pos_r,
                "grab": np.array(entry.get("grab_left", pos_l)),
                "yaw": float(entry.get("obb_angle_left", 0.0)),
            })
            print(f"  {label:>12} -> LEFT arm  (R.y={pos_r[1]:+.3f}, yaw={entry.get('obb_angle_left', 0.0):.2f}rad)")
        else:
            right_arm_items.append({
                "label": label,
                "pos": pos_r,
                "pos_other": pos_l,
                "grab": np.array(entry.get("grab_right", pos_r)),
                "yaw": float(entry.get("obb_angle_right", 0.0)),
            })
            print(f"  {label:>12} -> RIGHT arm (R.y={pos_r[1]:+.3f}, yaw={entry.get('obb_angle_right', 0.0):.2f}rad)")

    return left_arm_items, right_arm_items, tray_l, tray_r


# --- Dry-run validation ---
def dry_run_pick_place(arm, label, grab_pos, tray_pos, q_seed, yaw=0.0):
    """Validate IK for a full pick-and-place sequence."""
    q = q_seed.copy()
    wps = [
        ("approach",    [grab_pos[0], grab_pos[1], APPROACH_Z],  yaw),
        ("grasp",       [grab_pos[0], grab_pos[1], GRASP_Z],    yaw),
        ("lift",        [grab_pos[0], grab_pos[1], GRASP_Z + LIFT_Z], yaw),
        ("transit",     [tray_pos[0], tray_pos[1], SAFE_Z],     0.0),
        ("above_tray",  [tray_pos[0], tray_pos[1], PLACE_Z + 0.03], 0.0),
        ("place",       [tray_pos[0], tray_pos[1], PLACE_Z],    0.0),
        ("retreat",     [tray_pos[0], tray_pos[1], SAFE_Z],     0.0),
    ]
    for name, p, y in wps:
        qs = ik(arm, p, q, yaw=y)
        if qs is None:
            print(f"  {arm} {label} {name}: IK FAIL at ({p[0]:.3f},{p[1]:.3f},{p[2]:.3f}) yaw={y:.2f}")
            return False
        q = qs
    print(f"  {arm} {label}: all IK OK (yaw={yaw:.2f}rad)")
    return True


# --- Single-arm pick-and-place ---
def pick_and_place(robot, grip, arm, label, grab_pos, tray_pos, yaw=0.0):
    """Pick a silverware item and place it in the tray."""
    print(f"\n--- {arm.upper()}: pick {label} (yaw={yaw:.2f}rad) ---")

    grip_open(grip, arms=(arm,))
    q = np.array(robot.get_joint_positions())

    qs = ik(arm, [grab_pos[0], grab_pos[1], APPROACH_Z], q, yaw=yaw)
    if qs is not None:
        robot.move_to_joint_config(qs.tolist(), velocity=MOVE_VEL)
        q = qs
    time.sleep(0.2)

    qs = ik(arm, [grab_pos[0], grab_pos[1], GRASP_Z], q, yaw=yaw)
    if qs is not None:
        robot.move_to_joint_config(qs.tolist(), velocity=APPROACH_VEL)
        q = qs
    time.sleep(0.3)

    grip_close(grip, arms=(arm,))
    tcp = robot.get_tcp_pose()
    print(f"  Grasped at ({tcp[0]:.3f},{tcp[1]:.3f},{tcp[2]:.3f})")

    qs = ik(arm, [grab_pos[0], grab_pos[1], GRASP_Z + LIFT_Z], q, yaw=yaw)
    if qs is not None:
        robot.move_to_joint_config(qs.tolist(), velocity=MOVE_VEL)
        q = qs
    time.sleep(0.2)

    qs = ik(arm, [tray_pos[0], tray_pos[1], SAFE_Z], q)
    if qs is not None:
        robot.move_to_joint_config(qs.tolist(), velocity=MOVE_VEL)
        q = qs
    time.sleep(0.2)

    qs = ik(arm, [tray_pos[0], tray_pos[1], PLACE_Z + 0.03], q)
    if qs is not None:
        robot.move_to_joint_config(qs.tolist(), velocity=APPROACH_VEL)
        q = qs
    time.sleep(0.2)

    qs = ik(arm, [tray_pos[0], tray_pos[1], PLACE_Z], q)
    if qs is not None:
        robot.move_to_joint_config(qs.tolist(), velocity=APPROACH_VEL)
        q = qs
    time.sleep(0.2)

    grip_open(grip, arms=(arm,))
    tcp = robot.get_tcp_pose()
    print(f"  Released at ({tcp[0]:.3f},{tcp[1]:.3f},{tcp[2]:.3f})")
    time.sleep(0.3)

    qs = ik(arm, [tray_pos[0], tray_pos[1], SAFE_Z], q)
    if qs is not None:
        robot.move_to_joint_config(qs.tolist(), velocity=MOVE_VEL)
        q = qs
    time.sleep(0.2)

    return True


def _both(fl, fr):
    """Run two arm moves concurrently and wait for both."""
    tl = threading.Thread(target=fl); tr = threading.Thread(target=fr)
    tl.start(); tr.start(); tl.join(); tr.join()


def pick_and_place_simultaneous(l_robot, r_robot, grip,
                                 l_item, r_item, tray_l, tray_r):
    """Pick one item per arm simultaneously, then place both into tray."""
    l_yaw = l_item.get("yaw", 0.0)
    r_yaw = r_item.get("yaw", 0.0)
    print(f"\n--- SIMULTANEOUS: L picks {l_item['label']} (yaw={l_yaw:.2f}), "
          f"R picks {r_item['label']} (yaw={r_yaw:.2f}) ---")

    grip_open(grip)

    def step(l_target, r_target, vel, l_yaw_=0.0, r_yaw_=0.0):
        lq = np.array(l_robot.get_joint_positions())
        rq = np.array(r_robot.get_joint_positions())
        lsol = ik("left", l_target, lq, yaw=l_yaw_)
        rsol = ik("right", r_target, rq, yaw=r_yaw_)
        if lsol is not None and rsol is not None:
            _both(lambda: l_robot.move_to_joint_config(lsol.tolist(), velocity=vel),
                  lambda: r_robot.move_to_joint_config(rsol.tolist(), velocity=vel))

    lg, rg = l_item["grab"], r_item["grab"]
    step([lg[0], lg[1], APPROACH_Z], [rg[0], rg[1], APPROACH_Z], MOVE_VEL, l_yaw, r_yaw)
    time.sleep(0.2)
    step([lg[0], lg[1], GRASP_Z], [rg[0], rg[1], GRASP_Z], APPROACH_VEL, l_yaw, r_yaw)
    time.sleep(0.3)

    # Both close grippers
    grip_close(grip)

    step([lg[0], lg[1], GRASP_Z + LIFT_Z], [rg[0], rg[1], GRASP_Z + LIFT_Z], MOVE_VEL, l_yaw, r_yaw)
    time.sleep(0.2)

    TRAY_Y_OFFSET = 0.03
    l_tray_place = tray_l.copy()
    l_tray_place[1] += TRAY_Y_OFFSET
    r_tray_place = tray_r.copy()
    r_tray_place[1] -= TRAY_Y_OFFSET

    step([l_tray_place[0], l_tray_place[1], SAFE_Z], [r_tray_place[0], r_tray_place[1], SAFE_Z], MOVE_VEL)
    time.sleep(0.2)
    # Both descend into tray
    step([l_tray_place[0], l_tray_place[1], PLACE_Z], [r_tray_place[0], r_tray_place[1], PLACE_Z], APPROACH_VEL)
    time.sleep(0.2)

    # Both release
    grip_open(grip)
    time.sleep(0.3)

    # Both retreat
    step([l_tray_place[0], l_tray_place[1], SAFE_Z], [r_tray_place[0], r_tray_place[1], SAFE_Z], MOVE_VEL)
    time.sleep(0.2)

    print(f"  Both placed. L: {l_item['label']}, R: {r_item['label']}")
    return True


# --- Main ---
def main():
    p = argparse.ArgumentParser(description="Bimanual silverware sort")
    p.add_argument("--skip-detect", action="store_true",
                   help="Reuse last detection instead of running SAM3")
    p.add_argument("--dry-run", action="store_true",
                   help="Validate IK only, no motion")
    p.add_argument("--sequential", action="store_true",
                   help="Pick one arm at a time (safer, slower)")
    p.add_argument("--machine", default="ANON-LAB",
                   help="configs/machines/<machine>.yaml overlay (arm IPs, grippers)")
    args = p.parse_args()

    cfg = load_family_yaml("bimanual_franka", args.machine)
    robot_cfg = cfg.get("robot", {})

    # === GRIPPERS ===
    grip = grippers(cfg)
    print("=== GRIPPERS ===")
    grip_open(grip)
    grip_check(grip)

    # === DETECT ===
    if not args.skip_detect:
        print("\n=== DETECT ===")
        run_detection()

    d = load_detections()
    if d is None:
        grip.disconnect()
        return

    # === ASSIGN ITEMS ===
    print("\n=== ITEM ASSIGNMENT ===")
    print("  (Camera-left = RIGHT robot side, Camera-right = LEFT robot side)")
    left_items, right_items, tray_l, tray_r = assign_items_to_arms(d)
    print(f"\n  LEFT arm:  {len(left_items)} items: {[i['label'] for i in left_items]}")
    print(f"  RIGHT arm: {len(right_items)} items: {[i['label'] for i in right_items]}")
    print(f"  Tray L: ({tray_l[0]:.3f}, {tray_l[1]:.3f}, {tray_l[2]:.3f})")
    print(f"  Tray R: ({tray_r[0]:.3f}, {tray_r[1]:.3f}, {tray_r[2]:.3f})")

    # === DRY RUN ===
    print("\n=== DRY RUN ===")
    ok = True
    for item in left_items:
        if not dry_run_pick_place("left", item["label"], item["grab"], tray_l, HOME_Q, yaw=item.get("yaw", 0.0)):
            ok = False
    for item in right_items:
        if not dry_run_pick_place("right", item["label"], item["grab"], tray_r, HOME_Q, yaw=item.get("yaw", 0.0)):
            ok = False
    if not ok:
        print("ABORT: dry run failed")
        grip.disconnect()
        return
    print("  All dry runs passed")

    if args.dry_run:
        print("Dry run only -- not executing.")
        grip.disconnect()
        return

    # === CONNECT ROBOTS ===
    # Each FrankaBambooDriver spawns its own bamboo_control_node on connect.
    print("\n=== CONNECT ===")
    # One bamboo control node per arm (BimanualFrankaDriver.BAMBOO_*_PORT).
    l_robot = FrankaBambooDriver(ip=robot_cfg["left_ip"], port=_BFD.BAMBOO_LEFT_PORT)
    r_robot = FrankaBambooDriver(ip=robot_cfg["right_ip"], port=_BFD.BAMBOO_RIGHT_PORT)
    l_robot.connect()
    r_robot.connect()

    def home_both():
        _both(lambda: l_robot.go_home(velocity=0.2),
              lambda: r_robot.go_home(velocity=0.2))

    print("Homing...")
    home_both()
    print("  Both homed")

    try:
        if args.sequential:
            # === SEQUENTIAL MODE: one arm at a time ===
            print("\n=== SEQUENTIAL PICK-AND-PLACE ===")

            SLOT_SPACING = 0.025
            for idx, item in enumerate(right_items):
                t = tray_r.copy()
                t[1] += (idx - len(right_items) / 2.0) * SLOT_SPACING
                pick_and_place(r_robot, grip, "right", item["label"],
                               item["grab"], t, yaw=item.get("yaw", 0.0))
                r_robot.go_home(velocity=0.2)
                time.sleep(0.3)

            for idx, item in enumerate(left_items):
                t = tray_l.copy()
                t[1] += (idx - len(left_items) / 2.0) * SLOT_SPACING
                pick_and_place(l_robot, grip, "left", item["label"],
                               item["grab"], t, yaw=item.get("yaw", 0.0))
                l_robot.go_home(velocity=0.2)
                time.sleep(0.3)

        else:
            # === INTERLEAVED MODE: pair up items, pick simultaneously ===
            print("\n=== INTERLEAVED PICK-AND-PLACE ===")

            # Pair items: pick one from each side simultaneously
            n_pairs = min(len(left_items), len(right_items))
            for i in range(n_pairs):
                pick_and_place_simultaneous(
                    l_robot, r_robot, grip,
                    left_items[i], right_items[i],
                    tray_l, tray_r,
                )
                # Home between pairs for safety
                home_both()
                time.sleep(0.3)

            # Handle remaining unpaired items sequentially
            for item in left_items[n_pairs:]:
                pick_and_place(l_robot, grip, "left", item["label"],
                               item["grab"], tray_l)
                l_robot.go_home(velocity=0.2)
                time.sleep(0.3)
            for item in right_items[n_pairs:]:
                pick_and_place(r_robot, grip, "right", item["label"],
                               item["grab"], tray_r)
                r_robot.go_home(velocity=0.2)
                time.sleep(0.3)

        # === FINAL HOME ===
        print("\n=== FINAL HOME ===")
        home_both()
        print("  Both homed. DONE.")
        print(f"\n  Sorted {len(left_items) + len(right_items)} items into tray")
        print(f"    LEFT arm handled:  {[i['label'] for i in left_items]}")
        print(f"    RIGHT arm handled: {[i['label'] for i in right_items]}")

    except Exception as e:
        print(f"\nFAILED: {e}")
        import traceback; traceback.print_exc()
        try: grip_open(grip)
        except Exception: pass
        try:
            l_robot.go_home(velocity=0.2)
            r_robot.go_home(velocity=0.2)
        except Exception: pass

    l_robot.disconnect()
    r_robot.disconnect()
    grip.disconnect()


if __name__ == "__main__":
    main()

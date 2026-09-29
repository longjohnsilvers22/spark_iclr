#!/usr/bin/env python3
"""Pick-pour-place sequence for a mug using franky + pyroki IK.

Detects "mug handle" and "plate" via offline SAM3, then executes:
  open gripper -> approach handle -> grasp -> lift -> move above plate ->
  pour (pitch forward) -> hold -> pitch back -> move to place -> descend ->
  release -> retract -> home

No SPARK server, no Gemini. Pure franky + pyroki.

Usage:
    conda activate spark_conda
    cd ~/spark/src
    python ../scripts/pour_mug.py
    python ../scripts/pour_mug.py --dry-run
    python ../scripts/pour_mug.py --ip 10.0.0.1
"""
from __future__ import annotations

import argparse
import os
import sys
import time

import numpy as np
from scipy.spatial.transform import Rotation

# franky is only installed on the FR3 rig. --dry-run (planning/IK only) must
# keep working without it, so a missing module is tolerated here; failure
# surfaces at connect_robot().
try:
    from franky import (
        Gripper, JointMotion, JointStopMotion, RealtimeConfig,
        RelativeDynamicsFactor, Robot,
    )
except ImportError:
    Gripper = JointMotion = JointStopMotion = None
    RealtimeConfig = RelativeDynamicsFactor = Robot = None

# Path setup (must come before local imports)
_src = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "src")
if _src not in sys.path:
    sys.path.insert(0, os.path.abspath(_src))
_scripts = os.path.dirname(os.path.abspath(__file__))
if _scripts not in sys.path:
    sys.path.insert(0, _scripts)

# JAX caps (before any JAX import)
# NOTE: Do NOT set JAX_ENABLE_X64=true here. The pyroki IK solver uses jaxls
# which has int32/int64 dtype conflicts when x64 mode is enabled.
os.environ["JAX_PLATFORMS"] = "cpu"
os.environ.setdefault("XLA_PYTHON_CLIENT_PREALLOCATE", "false")
os.environ.setdefault("XLA_PYTHON_CLIENT_MEM_FRACTION", "0.15")

from spark_real.control.fr3_ik_pyroki import solve_ik, fk  # noqa: E402
from detect_offline import detect, open_kinects, load_calibrations  # noqa: E402

# Constants
DEFAULT_IP = "172.16.0.2"
HOME_Q = np.array([0.0, -0.785398, 0.0, -2.356194, -0.15, 1.570796, 0.785398])
DYN_FACTOR = 0.18

APPROACH_HEIGHT = 0.10       # 100 mm above handle for approach
LIFT_HEIGHT = 0.10           # 100 mm lift after grasp
POUR_ANGLE = 1.4             # ~80 deg pitch forward
POUR_HOLD_SEC = 3.0          # hold pour duration
RETRACT_HEIGHT = 0.05        # 50 mm retract after place
GRASP_APPROACH_PITCH = 0.0   # top-down approach
HANDLE_STANDOFF = 0.0        # no lateral standoff for top-down
ABOVE_PLATE_HEIGHT = 0.20    # height above plate for pour position

# Gripper orientations (both point straight down, differ in yaw)
# "horizontal" = default home: gripper fingers close along Y axis
GRIP_HORIZONTAL_R = Rotation.from_rotvec([np.pi, 0, 0]).as_matrix()
# "vertical" = 90 deg yaw: gripper fingers close along X axis
GRIP_VERTICAL_R = (Rotation.from_rotvec([np.pi, 0, 0])
                   * Rotation.from_euler('z', np.pi / 2)).as_matrix()

# Top-down orientation: gripper pointing straight down
R_TOP_DOWN = np.diag([1.0, -1.0, -1.0])


# Helpers

def phase(name: str):
    print(f"\n[ PHASE: {name} ]")


# Ask for confirmation. In dry-run or auto-confirm mode, always proceed.
def confirm(prompt: str, dry_run: bool):
    if dry_run or _AUTO_CONFIRM:
        print(f"  {prompt} -> auto-confirmed")
        return True
    resp = input(f"  {prompt} [y/N] ").strip().lower()
    return resp in ("y", "yes")


_AUTO_CONFIRM = False


# Print TCP position and orientation after a move.
def print_tcp(q: np.ndarray, label: str = ""):
    pos, R = fk(q)
    euler = Rotation.from_matrix(R).as_euler("xyz", degrees=True)
    prefix = f"  [{label}] " if label else "  "
    print(f"{prefix}TCP pos: [{pos[0]:+.4f}, {pos[1]:+.4f}, {pos[2]:+.4f}]  "
          f"euler(xyz): [{euler[0]:+.1f}, {euler[1]:+.1f}, {euler[2]:+.1f}] deg")


# Send a blocking joint motion and print TCP afterwards.
def move_blocking(robot, q: np.ndarray, label: str = "", dry_run: bool = False,
                   slow: bool = False):
    if dry_run:
        print_tcp(q, label or "planned")
        return
    if robot.has_errors:
        robot.recover_from_errors()
    factor = DYN_FACTOR * 0.5 if slow else DYN_FACTOR
    rdf = RelativeDynamicsFactor(factor, factor * 0.5, factor * 0.25)
    motion = JointMotion(q.tolist(), relative_dynamics_factor=rdf)
    robot.move(motion, asynchronous=False)
    time.sleep(0.1)
    q_actual = np.asarray(robot.current_joint_state.position, dtype=float)
    print_tcp(q_actual, label or "actual")


# Solve IK and raise on failure.
def compute_ik(target_pos: np.ndarray, target_rotmat: np.ndarray,
               q_seed: np.ndarray, label: str = "") -> np.ndarray:
    q = solve_ik(target_pos, target_rotmat, q_seed)
    if q is None:
        raise RuntimeError(f"IK failed for {label}: pos={target_pos}")
    return q


# Open the gripper.
def gripper_open(gripper, dry_run: bool = False):
    print("  Opening gripper...")
    if dry_run:
        print("  [dry-run] gripper open")
        return
    gripper.open(0.1)
    time.sleep(0.5)
    print("  Gripper open.")


# Close the gripper to grasp.
def gripper_grasp(gripper, dry_run: bool = False):
    print("  Closing gripper (grasp)...")
    if dry_run:
        print("  [dry-run] gripper grasp")
        return
    mw = gripper.max_width
    gripper.grasp(0.0, 0.1, 60.0, mw, mw)
    time.sleep(0.5)
    print("  Gripper grasped.")


# Orientation helpers

def side_approach_rotmat(approach_dir_xy: np.ndarray, pitch_from_vertical: float) -> np.ndarray:
    """
    Compute a gripper rotation matrix for a side approach.

    The gripper Z axis (pointing into the object) is tilted from vertical by
    pitch_from_vertical radians in the direction of approach_dir_xy.

    Parameters:
    approach_dir_xy : unit vector in XY plane pointing FROM the robot TOWARD the handle
    pitch_from_vertical : angle in radians from straight-down (0 = top-down, pi/2 = horizontal)
    """
    # Start with top-down: gripper Z = -world Z, gripper X = world X.
    # The approach direction in XY defines the pitch plane.

    ax = np.array(approach_dir_xy[:2])
    ax = ax / (np.linalg.norm(ax) + 1e-9)
    # Yaw angle of the approach direction (angle in XY plane from +X)
    yaw = np.arctan2(ax[1], ax[0])

    # Build rotation: start top-down, yaw to face approach dir, then pitch
    R_yaw = Rotation.from_euler("z", yaw).as_matrix()
    R_pitch = Rotation.from_euler("y", pitch_from_vertical).as_matrix()
    # Top-down base: Z points down, X points forward
    R_base = R_TOP_DOWN.copy()
    # Apply yaw then pitch (in tool frame)
    R = R_yaw @ R_pitch @ R_base
    return R


def pour_rotmat(current_rotmat: np.ndarray, pour_angle: float,
                pour_dir_xy: np.ndarray) -> np.ndarray:
    """
    Pitch the gripper toward the plate by pour_angle.

    Computes the tilt axis in the base frame as the horizontal axis
    perpendicular to the pour direction (handle -> plate). This way
    the pour is always a pitch regardless of gripper yaw.
    """
    # Pour direction in XY (from handle toward plate)
    d = np.array([pour_dir_xy[0], pour_dir_xy[1], 0.0])
    d = d / (np.linalg.norm(d) + 1e-9)
    # Tilt axis: perpendicular to pour direction, horizontal.
    # cross(Z_up, pour_dir) gives axis such that positive rotation tips the
    # rim (which faces along pour_dir) DOWNWARD, i.e. a true pitch that
    # pours toward the plate.
    tilt_axis = np.cross(np.array([0.0, 0.0, 1.0]), d)
    tilt_axis = tilt_axis / (np.linalg.norm(tilt_axis) + 1e-9)
    # Apply rotation in base frame: R_new = R_tilt @ R_current
    R_tilt = Rotation.from_rotvec(tilt_axis * pour_angle).as_matrix()
    return R_tilt @ current_rotmat


# Detection

def run_detection(dry_run: bool = False):
    """
    Detect mug handle and plate, determine handle orientation.

    Returns (handle_pos, plate_pos, handle_along_y).
    handle_along_y: True if the handle extends along Y (need 90 deg yaw grip).
    """
    phase("DETECTION")

    if dry_run:
        handle_pos = np.array([0.45, 0.10, 0.03])
        plate_pos = np.array([0.50, -0.15, 0.02])
        print(f"  [dry-run] Using mock positions:")
        print(f"    mug handle: [{handle_pos[0]:.4f}, {handle_pos[1]:.4f}, {handle_pos[2]:.4f}]")
        print(f"    plate:      [{plate_pos[0]:.4f}, {plate_pos[1]:.4f}, {plate_pos[2]:.4f}]")
        return handle_pos, plate_pos, True

    print("  Opening Kinects and running SAM3 on both cameras...")
    kinects = open_kinects()
    cals = load_calibrations()

    results = detect(["mug handle", "plate"], cameras="both",
                     kinects=kinects, cals=cals)

    for k in kinects.values():
        try:
            k.stop(); k.close()
        except Exception:
            pass

    # Collect per-camera handle mask areas to determine orientation
    handle_area = {}
    handle_pos_by_cam = {}
    handle_pos = None
    plate_pos = None

    for r in results:
        p = r.get("position_3d")
        if p is None:
            continue
        pos = np.array(p)
        cam = r.get("camera", "")
        if r["label"] == "mug handle":
            handle_area[cam] = r.get("mask_area", 0)
            handle_pos_by_cam[cam] = pos
            print(f"    mug handle ({cam}): [{pos[0]:.4f}, {pos[1]:.4f}, {pos[2]:.4f}]  "
                  f"conf={r['confidence']:.3f}  area={handle_area[cam]}")
        elif r["label"] == "plate":
            if plate_pos is None or r["confidence"] > 0.5:
                plate_pos = pos
                print(f"    plate ({cam}):      [{pos[0]:.4f}, {pos[1]:.4f}, {pos[2]:.4f}]  "
                      f"conf={r['confidence']:.3f}")

    # Use sideview Z (more reliable for height), birdview XY
    if "sideview" in handle_pos_by_cam and "birdview" in handle_pos_by_cam:
        handle_pos = handle_pos_by_cam["birdview"].copy()
        handle_pos[2] = handle_pos_by_cam["sideview"][2]
    elif "sideview" in handle_pos_by_cam:
        handle_pos = handle_pos_by_cam["sideview"]
    elif "birdview" in handle_pos_by_cam:
        handle_pos = handle_pos_by_cam["birdview"]

    # Determine handle orientation from mask area ratio
    bird_area = handle_area.get("birdview", 0)
    side_area = handle_area.get("sideview", 0)
    # If birdview sees a bigger handle mask, the handle extends along Y
    # (visible from above = extends horizontally in the Y direction)
    handle_along_y = bird_area >= side_area
    print(f"    Handle orientation: birdview_area={bird_area} sideview_area={side_area} "
          f"-> {'along Y (use 90 deg yaw)' if handle_along_y else 'along X (default grip)'}")

    if handle_pos is None:
        raise RuntimeError("Failed to detect 'mug handle'.")
    if plate_pos is None:
        raise RuntimeError("Failed to detect 'plate'.")

    return handle_pos, plate_pos, handle_along_y


# Phase functions

def phase_go_home(robot, gripper, dry_run: bool = False):
    # Move to the home configuration.
    phase("HOME")
    move_blocking(robot, HOME_Q, label="home", dry_run=dry_run)


def phase_open_gripper(gripper, dry_run: bool = False):
    # Open the gripper in preparation for grasp.
    phase("OPEN GRIPPER")
    gripper_open(gripper, dry_run=dry_run)


def phase_grasp_handle(robot, gripper, handle_pos: np.ndarray,
                       q_seed: np.ndarray, dry_run: bool = False,
                       **kwargs) -> np.ndarray:
    """
    Approach and grasp the mug handle from the side.

    Returns the joint config after grasping.
    """
    phase("GRASP MUG HANDLE")

    if not confirm("Proceed with handle grasp?", dry_run):
        raise RuntimeError("User aborted grasp phase.")

    # Pick orientation based on handle direction
    handle_along_y = kwargs.get("handle_along_y", True)
    if handle_along_y:
        R_grasp = GRIP_VERTICAL_R.copy()
        print("    Using 90 deg yaw grip (handle along Y)")
    else:
        R_grasp = GRIP_HORIZONTAL_R.copy()
        print("    Using default grip (handle along X)")

    # Above-handle position
    above_pos = handle_pos.copy()
    above_pos[2] += APPROACH_HEIGHT

    # Grasp position: at handle height + small margin
    grasp_pos = handle_pos.copy()
    grasp_pos[2] += 0.005

    print("\n  Step 1: Move above handle...")
    q_above = compute_ik(above_pos, R_grasp, q_seed, label="above handle")
    move_blocking(robot, q_above, label="above handle", dry_run=dry_run)

    # Step 2: Descend to grasp position (slow to avoid velocity violation)
    print("\n  Step 2: Descend to handle...")
    q_grasp = compute_ik(grasp_pos, R_grasp, q_above, label="grasp pos")
    move_blocking(robot, q_grasp, label="grasp pos", dry_run=dry_run, slow=True)

    print("\n  Step 4: Grasp...")
    gripper_grasp(gripper, dry_run=dry_run)

    return q_grasp


def phase_lift(robot, handle_pos: np.ndarray, q_grasp: np.ndarray,
               dry_run: bool = False) -> np.ndarray:
    """
    Lift the mug 100mm straight up from grasp position.

    Returns the joint config after lifting.
    """
    phase("LIFT")

    # Get current TCP pose
    grasp_tcp_pos, grasp_tcp_R = fk(q_grasp)

    lift_pos = grasp_tcp_pos.copy()
    lift_pos[2] += LIFT_HEIGHT

    print(f"  Lifting {LIFT_HEIGHT*1000:.0f} mm...")
    q_lift = compute_ik(lift_pos, grasp_tcp_R, q_grasp, label="lift")
    move_blocking(robot, q_lift, label="lifted", dry_run=dry_run)

    return q_lift


def phase_pour(robot, plate_pos: np.ndarray, handle_pos: np.ndarray,
               q_lift: np.ndarray, dry_run: bool = False) -> np.ndarray:
    """
    Move above plate and pour by pitching forward.

    Returns the joint config after pouring (upright, above plate).
    """
    phase("POUR")

    if not confirm("Proceed with pour over plate?", dry_run):
        raise RuntimeError("User aborted pour phase.")

    # Get current orientation from lift pose
    lift_pos, lift_R = fk(q_lift)

    # Pour direction: from handle toward plate (in XY)
    pour_dir = plate_pos[:2] - handle_pos[:2]
    pour_dir = pour_dir / (np.linalg.norm(pour_dir) + 1e-9)
    print(f"  Pour direction: [{pour_dir[0]:.3f}, {pour_dir[1]:.3f}]")

    # Pour position: move partway from handle toward plate, stay high
    # Don't go all the way to the plate, just extend the mug rim over it
    pour_pos = handle_pos.copy()
    pour_pos[:2] += pour_dir * 0.08  # 80mm toward plate
    pour_pos[2] = max(lift_pos[2], plate_pos[2] + ABOVE_PLATE_HEIGHT)

    # Keep the same orientation as the lift pose.
    print("\n  Step 1: Move to pour position...")
    q_above_plate = compute_ik(pour_pos, lift_R, q_lift, label="pour position")
    move_blocking(robot, q_above_plate, label="above plate", dry_run=dry_run)

    # Step 2: Pitch forward to pour
    above_plate_pos, above_plate_R = fk(q_above_plate)
    R_poured = pour_rotmat(above_plate_R, POUR_ANGLE, pour_dir)
    print(f"\n  Step 2: Pitching forward {np.degrees(POUR_ANGLE):.0f} deg to pour...")
    q_pour = compute_ik(pour_pos, R_poured, q_above_plate, label="pour")
    move_blocking(robot, q_pour, label="pouring", dry_run=dry_run)

    print(f"\n  Step 3: Holding pour for {POUR_HOLD_SEC:.0f} seconds...")
    if not dry_run:
        time.sleep(POUR_HOLD_SEC)
    else:
        print(f"  [dry-run] Would hold for {POUR_HOLD_SEC:.0f}s")

    print("\n  Step 4: Returning to upright...")
    q_upright = compute_ik(pour_pos, above_plate_R, q_pour, label="upright")
    move_blocking(robot, q_upright, label="upright", dry_run=dry_run)

    return q_upright


def phase_place(robot, gripper, handle_pos: np.ndarray,
                q_current: np.ndarray, dry_run: bool = False) -> np.ndarray:
    """
    Place the mug back above its original position.

    Returns the joint config after placing.
    """
    phase("PLACE")

    if not confirm("Proceed with placing the mug?", dry_run):
        raise RuntimeError("User aborted place phase.")

    current_pos, current_R = fk(q_current)

    # Step 1: Move above original mug position
    place_above_pos = handle_pos.copy()
    place_above_pos[2] = current_pos[2]  # keep current height
    print("\n  Step 1: Move above original mug position...")
    q_above_place = compute_ik(place_above_pos, current_R, q_current, label="above place")
    move_blocking(robot, q_above_place, label="above place", dry_run=dry_run)

    # Step 2: Descend to place
    place_pos = handle_pos.copy()
    place_pos[2] += 0.01  # 10 mm above table surface
    print("\n  Step 2: Descending to place position...")
    q_place = compute_ik(place_pos, current_R, q_above_place, label="place")
    move_blocking(robot, q_place, label="placed", dry_run=dry_run)

    print("\n  Step 3: Releasing...")
    gripper_open(gripper, dry_run=dry_run)

    # Step 4: Retract upward
    retract_pos = place_pos.copy()
    retract_pos[2] += RETRACT_HEIGHT
    print(f"\n  Step 4: Retracting {RETRACT_HEIGHT*1000:.0f} mm up...")
    q_retract = compute_ik(retract_pos, current_R, q_place, label="retract")
    move_blocking(robot, q_retract, label="retracted", dry_run=dry_run)

    return q_retract


# Robot connection

# Connect to the FR3 and return (robot, gripper).
def connect_robot(ip: str):
    if Robot is None:
        raise ImportError("franky is not installed; FR3 execution unavailable")
    print(f"Connecting to FR3 at {ip}...")
    try:
        robot = Robot(ip)
    except Exception as e:
        if "realtime" in str(e).lower():
            print("  Non-RT kernel, using RealtimeConfig.Ignore")
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


# Main

def main():
    parser = argparse.ArgumentParser(
        description="Pick-pour-place a mug using franky + pyroki (no server, no Gemini)",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument("--ip", default=DEFAULT_IP,
                        help=f"Robot IP (default: {DEFAULT_IP})")
    parser.add_argument("--dry-run", action="store_true",
                        help="Print planned sequence without moving the robot")
    parser.add_argument("-y", "--yes", action="store_true",
                        help="Auto-confirm all prompts")
    parser.add_argument("--phase", type=str, default=None,
                        choices=["detect", "grasp", "lift", "pour", "place", "home", "all"],
                        help="Run a single phase, or 'all' for full sequence")
    parser.add_argument("--handle-xyz", type=str, default=None,
                        help="Skip detection, use these handle coords: 'x,y,z'")
    parser.add_argument("--plate-xyz", type=str, default=None,
                        help="Skip detection, use these plate coords: 'x,y,z'")
    args = parser.parse_args()
    global _AUTO_CONFIRM
    dry_run = args.dry_run
    _AUTO_CONFIRM = args.yes
    run_phase = args.phase or "all"

    if dry_run:
        print("\n*** DRY RUN MODE -- no robot motion ***\n")

    print("Warming up pyroki IK...")
    t0 = time.time()
    warmup_pos = np.array([0.5, 0.0, 0.3])
    warmup_R = Rotation.from_euler("xyz", [np.pi, 0, 0]).as_matrix()
    _ = solve_ik(warmup_pos, warmup_R, HOME_Q)
    _ = fk(HOME_Q)
    print(f"  IK warm-up done in {time.time()-t0:.1f}s")

    # 2. Get target positions
    if args.handle_xyz:
        handle_pos = np.array([float(x) for x in args.handle_xyz.split(",")])
        print(f"  Handle (manual): [{handle_pos[0]:.4f}, {handle_pos[1]:.4f}, {handle_pos[2]:.4f}]")
    else:
        handle_pos = None

    if args.plate_xyz:
        plate_pos = np.array([float(x) for x in args.plate_xyz.split(",")])
        print(f"  Plate (manual):  [{plate_pos[0]:.4f}, {plate_pos[1]:.4f}, {plate_pos[2]:.4f}]")
    else:
        plate_pos = None

    handle_along_y = True  # default: assume handle along Y (90 deg yaw)

    if run_phase in ("detect", "all") and (handle_pos is None or plate_pos is None):
        det_handle, det_plate, handle_along_y = run_detection(dry_run=dry_run)
        if handle_pos is None:
            handle_pos = det_handle
        if plate_pos is None:
            plate_pos = det_plate

    if run_phase == "detect":
        print("\n  Phase 'detect' complete. Use --handle-xyz and --plate-xyz to pass coords to next phase.")
        return

    # 3. Connect to robot
    robot = None
    gripper = None
    if not dry_run:
        phase("CONNECT")
        robot, gripper = connect_robot(args.ip)
        q_now = np.asarray(robot.current_joint_state.position, dtype=float)
        print_tcp(q_now, "current")
    else:
        print("\n  [dry-run] Skipping robot connection")

    phases_to_run = {
        "all":   ["home", "open", "grasp", "lift", "pour", "place", "home_end"],
        "home":  ["home"],
        "grasp": ["open", "grasp"],
        "lift":  ["lift"],
        "pour":  ["pour"],
        "place": ["place", "home_end"],
    }
    active = phases_to_run.get(run_phase, ["all"])

    try:
        q_current = np.asarray(robot.current_joint_state.position, dtype=float) if robot else HOME_Q

        if "home" in active:
            phase_go_home(robot, gripper, dry_run=dry_run)
            q_current = HOME_Q.copy()

        if "open" in active:
            phase_open_gripper(gripper, dry_run=dry_run)

        if "grasp" in active:
            q_current = phase_grasp_handle(robot, gripper, handle_pos, q_current,
                                          dry_run=dry_run, handle_along_y=handle_along_y)

        if "lift" in active:
            q_current = phase_lift(robot, handle_pos, q_current, dry_run=dry_run)

        if "pour" in active:
            q_current = phase_pour(robot, plate_pos, handle_pos, q_current, dry_run=dry_run)

        if "place" in active:
            q_current = phase_place(robot, gripper, handle_pos, q_current, dry_run=dry_run)

        if "home_end" in active:
            phase_go_home(robot, gripper, dry_run=dry_run)

        phase("DONE")
        if run_phase == "all":
            print("  Mug pick-pour-place sequence complete.")
        else:
            print(f"  Phase '{run_phase}' complete.")

    except KeyboardInterrupt:
        print("\n\nAborted by user.")
    except RuntimeError as e:
        print(f"\n\nError: {e}")
    finally:
        if robot is not None and not dry_run:
            print("\nStopping robot...")
            try:
                robot.move(JointStopMotion(), asynchronous=False)
            except Exception:
                pass


if __name__ == "__main__":
    main()

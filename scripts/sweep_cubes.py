#!/usr/bin/env python3
"""Sweep cubes into dustpan on FR3 via Bamboo + pyroki.

Sequence: grasp brush -> reorient bristles -> sweep cubes toward dustpan -> lift -> home.
Uses pyroki IK (avoids J5 singularity), Bamboo torque control,
RolloutRecorder for paper figures, and optionally Gemini BT generation.

Usage:
    cd ~/spark/src
    python ../scripts/sweep_cubes.py
    python ../scripts/sweep_cubes.py --phase grasp       # grasp brush only
    python ../scripts/sweep_cubes.py --phase sweep        # sweep only (brush already held)
    python ../scripts/sweep_cubes.py --no-record          # skip video recording
    python ../scripts/sweep_cubes.py --coords /tmp/sweep_coords.json
    python ../scripts/sweep_cubes.py --objects "red cube,blue cube,yellow cube"
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time
import traceback

import cv2
import numpy as np
import yaml
from PIL import Image
from scipy.spatial.transform import Rotation as R

# Path setup (must come before local imports)
_src = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "src")
if _src not in sys.path:
    sys.path.insert(0, os.path.abspath(_src))
_scripts = os.path.dirname(os.path.abspath(__file__))
if _scripts not in sys.path:
    sys.path.insert(0, _scripts)

# JAX caps (before any JAX import)
os.environ["JAX_PLATFORMS"] = "cpu"
os.environ.setdefault("XLA_PYTHON_CLIENT_PREALLOCATE", "false")
os.environ.setdefault("XLA_PYTHON_CLIENT_MEM_FRACTION", "0.15")

from spark_real.control.fr3_ik_pyroki import solve_ik  # noqa: E402
from spark_real.robots.franka.franka_bamboo_driver import FrankaBambooDriver  # noqa: E402
from detect_offline import open_kinects, capture, load_calibrations, backproject  # noqa: E402
from rollout_recorder import RolloutRecorder  # noqa: E402

# Constants
DEFAULT_IP = "172.16.0.2"
HOME_Q = np.array([0.0, -0.785398, 0.0, -2.356194, -0.15, 1.570796, 0.785398])

# Gripper orientation: straight-down, pi rotation about X
ORIENT_DOWN = R.from_rotvec([np.pi, 0, 0]).as_rotvec()

# Brush grasp
BRUSH_GRASP_Z = 0.02          # handle sits close to table
BRUSH_HOVER_Z = 0.08          # hover above brush before descend
BRUSH_LIFT_Z = 0.10           # lift after grasping brush
BRUSH_GRASP_FORCE = 60        # firm grip on handle
GRASP_RETRIES = 4
GRASP_DZ = 0.003              # descend per retry

# Sweep
SWEEP_Z = 0.037               # constant Z during sweep (bristle contact)
SWEEP_WAYPOINT_SPACING = 0.025  # 25 mm between fine waypoints
SWEEP_VEL = 0.08              # velocity during sweep
SWEEP_APPROACH_MARGIN = 0.060  # start 60 mm behind objects (in -Y)
SWEEP_OVERSHOOT = 0.02        # push 20 mm past dustpan center

# Transit
TRANSIT_Z = 0.12              # safe transit height


def detect_objects(caps, cals, object_labels):
    """
    Detect brush, objects, and dustpan via SAM3.

    Uses sideview centroid for brush XY (midpoint lands in empty space).
    Returns dict with all detected positions, OBB yaw, etc.
    """
    # DELIBERATELY DEFERRED IMPORT: SPARKPerception pulls in torch, and the
    # Kinect capture (open_kinects/capture in main) must run BEFORE torch is
    # loaded in this process (torch loaded first segfaults pyk4a capture; see
    # rollout_recorder.py's subprocess workaround). Do NOT move to module top.
    from spark_real.perception.spark_perception import SPARKPerception

    perc = SPARKPerception()
    perc.load_models(load_da3=False)

    all_prompts = ["brush"] + object_labels + ["black dustpan"]
    results = {}

    for cam in ["sideview", "birdview"]:
        c = caps[cam]
        pil = Image.fromarray(c["rgb"])
        intr = {k: c[k] for k in ("fx", "fy", "cx", "cy")}

        for prompt in all_prompts:
            state = perc._sam3.set_image(pil)
            state = perc._sam3.set_text_prompt(prompt=prompt, state=state)
            masks = state.get("masks")
            scores = state.get("scores")
            if masks is None or masks.numel() == 0:
                continue
            best = int(scores.argmax())
            score = float(scores[best])
            mask = masks[best].cpu().numpy().squeeze()
            ys, xs = np.where(mask > 0)
            if len(xs) == 0:
                continue

            mcx, mcy = float(xs.mean()), float(ys.mean())
            pos = backproject(mcx, mcy, c["depth"], intr, cals[cam])

            # OBB yaw via PCA of mask pixels
            obb_yaw = None
            if len(xs) >= 5:
                pts_px = np.column_stack([xs.astype(np.float64),
                                          ys.astype(np.float64)])
                cov = np.cov(pts_px, rowvar=False)
                eigvals, eigvecs = np.linalg.eigh(cov)
                major = eigvecs[:, np.argmax(eigvals)]
                obb_yaw = float(np.arctan2(major[1], major[0]))

            key = f"{cam}_{prompt}"
            results[key] = {
                "pos": pos.tolist() if pos is not None else None,
                "conf": score,
                "centroid_px": (mcx, mcy),
                "camera": cam,
                "label": prompt,
                "obb_yaw": obb_yaw,
                "mask_area": len(xs),
            }

    # Build coordinates dict from best detections
    coords = {}

    # Brush: prefer sideview centroid (midpoint lands in empty space)
    brush = _best_det(results, "brush", prefer="sideview")
    if brush and brush["pos"]:
        coords["brush_x"] = brush["pos"][0]
        coords["brush_y"] = brush["pos"][1]
        coords["brush_obb_yaw"] = brush.get("obb_yaw", 0.0)
    else:
        print("  WARNING: brush not detected")
        return None

    # Objects (cubes etc.)
    obj_positions = []
    for label in object_labels:
        det = _best_det(results, label, prefer="birdview")
        if det and det["pos"]:
            coords[f"{label}_x"] = det["pos"][0]
            coords[f"{label}_y"] = det["pos"][1]
            coords[f"{label}_z"] = det["pos"][2]
            obj_positions.append(det["pos"][:2])
        else:
            print(f"  WARNING: '{label}' not detected")

    if obj_positions:
        obj_arr = np.array(obj_positions)
        coords["objects_min_y"] = float(obj_arr[:, 1].min())
        coords["objects_max_y"] = float(obj_arr[:, 1].max())
        coords["objects_center_x"] = float(obj_arr[:, 0].mean())
        coords["objects_center_y"] = float(obj_arr[:, 1].mean())
    else:
        print("  WARNING: no objects detected")
        return None

    # Dustpan: prefer birdview
    dustpan = _best_det(results, "black dustpan", prefer="birdview")
    if dustpan and dustpan["pos"]:
        coords["dustpan_x"] = dustpan["pos"][0]
        coords["dustpan_y"] = dustpan["pos"][1]
    else:
        print("  WARNING: dustpan not detected")
        return None

    return coords


# Pick best detection for a label, preferring the given camera.
def _best_det(results, label, prefer="sideview"):
    pref = results.get(f"{prefer}_{label}")
    other = "birdview" if prefer == "sideview" else "sideview"
    alt = results.get(f"{other}_{label}")
    if pref and pref.get("pos"):
        return pref
    return alt


# Generate BT via Gemini planner (optional, for recording).
def generate_bt(coords, object_labels, instruction):
    try:
        # DELIBERATELY DEFERRED IMPORT: SPARKPlanner pulls in torch, which
        # must not load before the Kinect capture (see detect_objects note).
        # Also guarded: BT generation is optional and non-fatal.
        from spark_real.planning.spark_planner import SPARKPlanner
        planner = SPARKPlanner()
        keypoint_labels = ["brush"] + object_labels + ["black dustpan"]
        detection_details = []
        for label in keypoint_labels:
            pos = None
            for prefix in [f"{label}_x", f"{label}_y"]:
                pass  # simplified, use coords
            # Build minimal detection details
            if label == "brush":
                pos = [coords.get("brush_x", 0), coords.get("brush_y", 0), BRUSH_GRASP_Z]
            elif label == "black dustpan":
                pos = [coords.get("dustpan_x", 0), coords.get("dustpan_y", 0), 0.0]
            else:
                pos = [coords.get(f"{label}_x", 0), coords.get(f"{label}_y", 0),
                       coords.get(f"{label}_z", 0)]
            detection_details.append({
                "label": label,
                "confidence": 0.9,
                "position_3d": pos,
            })

        score = planner.generate_score(
            instruction=instruction,
            keypoint_labels=keypoint_labels,
            detection_details=detection_details,
        )
        return score
    except Exception as e:
        print(f"  BT generation failed: {e}")
        return None


# Pre-verify IK reachability for key waypoints.
def dry_run_ik(coords):
    q = HOME_Q.copy()
    orient = ORIENT_DOWN.tolist()

    waypoints = [
        ("hover_brush", [coords["brush_x"], coords["brush_y"], BRUSH_HOVER_Z]),
        ("grasp_brush", [coords["brush_x"], coords["brush_y"], BRUSH_GRASP_Z]),
        ("lift_brush",  [coords["brush_x"], coords["brush_y"], BRUSH_LIFT_Z]),
    ]

    # Sweep start and end
    sweep_start_y = coords["objects_min_y"] - SWEEP_APPROACH_MARGIN
    sweep_end_y = coords["dustpan_y"] + SWEEP_OVERSHOOT
    sweep_x = coords["objects_center_x"]

    waypoints += [
        ("transit_to_sweep", [sweep_x, sweep_start_y, TRANSIT_Z]),
        ("sweep_start", [sweep_x, sweep_start_y, SWEEP_Z]),
        ("sweep_mid",   [sweep_x, coords["objects_center_y"], SWEEP_Z]),
        ("sweep_end",   [sweep_x, sweep_end_y, SWEEP_Z]),
        ("sweep_lift",  [sweep_x, sweep_end_y, TRANSIT_Z]),
    ]

    ok = True
    for name, pos in waypoints:
        qs = solve_ik(np.array(pos), orient, q_seed=q)
        if qs is None:
            print(f"  {name}: FAIL ({pos[0]:.3f}, {pos[1]:.3f}, {pos[2]:.3f})")
            ok = False
        else:
            print(f"  {name}: OK   ({pos[0]:.3f}, {pos[1]:.3f}, {pos[2]:.3f})")
            q = qs
    return ok


def compute_sweep_waypoints(coords):
    """
    Generate fine sweep waypoints from behind objects to dustpan.

    Path runs at constant Z, from (center_x, min_y - margin) through
    objects toward (center_x, dustpan_y + overshoot), with waypoints
    every 25 mm.
    """
    sweep_x = coords["objects_center_x"]
    y_start = coords["objects_min_y"] - SWEEP_APPROACH_MARGIN
    y_end = coords["dustpan_y"] + SWEEP_OVERSHOOT

    dist = abs(y_end - y_start)
    n_pts = max(int(dist / SWEEP_WAYPOINT_SPACING), 2)
    ys = np.linspace(y_start, y_end, n_pts)

    waypoints = []
    for y in ys:
        waypoints.append([sweep_x, float(y), SWEEP_Z])
    return waypoints


# Hover, descend, grasp brush with retries.
def execute_grasp_brush(robot, rec, coords):
    bx, by = coords["brush_x"], coords["brush_y"]
    obb_yaw = coords.get("brush_obb_yaw", 0.0)

    # Orientation: straight down with OBB yaw alignment
    orient_yaw = (R.from_euler("z", obb_yaw) * R.from_rotvec([np.pi, 0, 0])).as_rotvec()

    print(f"  Brush at ({bx:.4f}, {by:.4f}), obb_yaw={np.degrees(obb_yaw):.1f} deg")

    # Hover above brush
    q = np.array(robot.get_joint_positions())
    qs = solve_ik(np.array([bx, by, BRUSH_HOVER_Z]), orient_yaw.tolist(), q_seed=q)
    if qs is not None:
        robot.move_to_joint_config(qs.tolist(), velocity=0.15)
    else:
        robot.move_linear([bx, by, BRUSH_HOVER_Z] + orient_yaw.tolist(), velocity=0.15)
    time.sleep(0.3)

    # Descend to grasp height
    gz = BRUSH_GRASP_Z
    q = np.array(robot.get_joint_positions())
    qs = solve_ik(np.array([bx, by, gz]), orient_yaw.tolist(), q_seed=q)
    if qs is not None:
        robot.move_to_joint_config(qs.tolist(), velocity=0.06)
    else:
        robot.move_linear([bx, by, gz] + orient_yaw.tolist(), velocity=0.06)
    time.sleep(0.3)

    # Grasp with retries
    grasped = False
    for attempt in range(GRASP_RETRIES):
        robot.close_gripper(force=BRUSH_GRASP_FORCE)
        time.sleep(0.8)
        w = robot._gripper_width()
        print(f"    attempt {attempt + 1}: width={w:.4f} z={gz:.4f}")
        if w > 0.001:
            grasped = True
            break
        robot.open_gripper()
        time.sleep(0.3)
        gz -= GRASP_DZ
        q = np.array(robot.get_joint_positions())
        qs = solve_ik(np.array([bx, by, gz]), orient_yaw.tolist(), q_seed=q)
        if qs is not None:
            robot.move_to_joint_config(qs.tolist(), velocity=0.03)
        time.sleep(0.2)

    if not grasped:
        print("  FAILED to grasp brush")
        robot.open_gripper()
        time.sleep(0.3)
        return False

    if rec:
        rec.mark_keyframe("grasp_brush")

    q = np.array(robot.get_joint_positions())
    qs = solve_ik(np.array([bx, by, BRUSH_LIFT_Z]), orient_yaw.tolist(), q_seed=q)
    if qs is not None:
        robot.move_to_joint_config(qs.tolist(), velocity=0.10)
    time.sleep(0.3)

    print("  Brush grasped and lifted")
    return True


def execute_reorient(robot, rec, coords):
    """
    Reorient brush so bristles face sweep direction (+Y).

    Rotate wrist to yaw=0 degrees while holding the brush at lift height.
    """
    tcp = robot.get_tcp_pose()
    orient_zero = ORIENT_DOWN.tolist()  # yaw=0: bristles face +Y

    q = np.array(robot.get_joint_positions())
    qs = solve_ik(np.array([tcp[0], tcp[1], BRUSH_LIFT_Z]), orient_zero, q_seed=q)
    if qs is not None:
        robot.move_to_joint_config(qs.tolist(), velocity=0.10)
    time.sleep(0.3)

    if rec:
        rec.mark_keyframe("reorient_brush")
    print("  Brush reoriented to yaw=0 (bristles face +Y)")


# Transit to sweep start, lower, sweep through objects, lift.
def execute_sweep(robot, rec, coords):
    sweep_x = coords["objects_center_x"]
    y_start = coords["objects_min_y"] - SWEEP_APPROACH_MARGIN
    orient = ORIENT_DOWN.tolist()

    # Transit to sweep start at safe height
    print(f"  Transit to sweep start ({sweep_x:.4f}, {y_start:.4f}, {TRANSIT_Z:.3f})")
    q = np.array(robot.get_joint_positions())
    qs = solve_ik(np.array([sweep_x, y_start, TRANSIT_Z]), orient, q_seed=q)
    if qs is not None:
        robot.move_to_joint_config(qs.tolist(), velocity=0.15)
    time.sleep(0.3)

    # Lower to sweep Z
    q = np.array(robot.get_joint_positions())
    qs = solve_ik(np.array([sweep_x, y_start, SWEEP_Z]), orient, q_seed=q)
    if qs is not None:
        robot.move_to_joint_config(qs.tolist(), velocity=0.06)
    time.sleep(0.2)

    if rec:
        rec.mark_keyframe("sweep_start")

    # Fine sweep waypoints
    waypoints = compute_sweep_waypoints(coords)
    print(f"  Sweeping {len(waypoints)} waypoints from y={waypoints[0][1]:.4f} to y={waypoints[-1][1]:.4f}")

    # Pre-compute all joint configs for continuity
    q = np.array(robot.get_joint_positions())
    sweep_joints = []
    for wp in waypoints:
        qs = solve_ik(np.array(wp), orient, q_seed=q)
        if qs is not None:
            # J7 continuity check
            if abs(qs[6] - q[6]) > 1.0:
                qs_c = qs.copy()
                qs_c[6] = q[6]
                qs2 = solve_ik(np.array(wp), orient, q_seed=qs_c)
                if qs2 is not None and abs(qs2[6] - q[6]) < 1.0:
                    qs = qs2
                else:
                    qs[6] = q[6] + np.clip(qs[6] - q[6], -0.5, 0.5)
            sweep_joints.append(qs)
            q = qs
        else:
            sweep_joints.append(None)
            print(f"    IK failed at waypoint ({wp[0]:.3f}, {wp[1]:.3f}, {wp[2]:.3f})")

    # Execute sweep
    for i, qs in enumerate(sweep_joints):
        if qs is not None:
            robot.move_to_joint_config(qs.tolist(), velocity=SWEEP_VEL)
        time.sleep(0.02)

    if rec:
        rec.mark_keyframe("sweep_end")

    # Lift after sweep
    tcp = robot.get_tcp_pose()
    q = np.array(robot.get_joint_positions())
    qs = solve_ik(np.array([tcp[0], tcp[1], TRANSIT_Z]), orient, q_seed=q)
    if qs is not None:
        robot.move_to_joint_config(qs.tolist(), velocity=0.15)
    time.sleep(0.3)

    print("  Sweep complete")


def main():
    parser = argparse.ArgumentParser(description="Sweep cubes into dustpan on FR3")
    parser.add_argument("--phase", choices=["grasp", "sweep", "all"], default="all",
                        help="Which phase to execute (default: all)")
    parser.add_argument("--no-record", action="store_true",
                        help="Skip video recording")
    parser.add_argument("--coords", type=str, default=None,
                        help="Path to saved coords JSON (skip detection)")
    parser.add_argument("--objects", type=str, default="orange cube,green cube",
                        help="Comma-separated object labels (default: 'orange cube,green cube')")
    parser.add_argument("--ip", type=str, default=DEFAULT_IP,
                        help=f"Robot IP (default: {DEFAULT_IP})")
    args = parser.parse_args()

    object_labels = [s.strip() for s in args.objects.split(",") if s.strip()]
    instruction = f"pick up the brush and sweep {', '.join(object_labels)} into the dustpan"
    do_grasp = args.phase in ("grasp", "all")
    do_sweep = args.phase in ("sweep", "all")

    # Capture and detect (or load coords)

    if args.coords:
        coords = json.load(open(args.coords))
        print(f"LOADED COORDS from {args.coords}")
        caps = None
    else:
        print("CAPTURE")
        kinects = open_kinects()
        caps = capture(kinects)
        for k in kinects.values():
            try:
                k.stop()
            except Exception:
                pass
            try:
                k.close()
            except Exception:
                pass
        cals = load_calibrations()

        print("\nDETECT")
        coords = detect_objects(caps, cals, object_labels)
        if coords is None:
            print("Detection failed")
            return

        json.dump(coords, open("/tmp/sweep_coords.json", "w"), indent=2)
        print(f"  Coords saved to /tmp/sweep_coords.json")

    print(f"\n  Brush:   ({coords['brush_x']:.4f}, {coords['brush_y']:.4f})")
    for label in object_labels:
        xk, yk = f"{label}_x", f"{label}_y"
        if xk in coords:
            print(f"  {label}: ({coords[xk]:.4f}, {coords[yk]:.4f})")
    print(f"  Dustpan: ({coords['dustpan_x']:.4f}, {coords['dustpan_y']:.4f})")
    print(f"  Objects center: ({coords['objects_center_x']:.4f}, {coords['objects_center_y']:.4f})")
    print(f"  Objects Y range: [{coords['objects_min_y']:.4f}, {coords['objects_max_y']:.4f}]")

    # Generate BT via Gemini (optional, for recording)

    print("\nBT GENERATION")
    bt_score = generate_bt(coords, object_labels, instruction)
    bt_yaml_str = None
    if bt_score:
        bt_yaml_str = yaml.safe_dump(bt_score, sort_keys=False)
        print(bt_yaml_str)
    else:
        print("  Skipped (planner unavailable or failed)")

    # Dry run IK

    print("\nDRY RUN")
    if not dry_run_ik(coords):
        print("ABORT: IK dry run failed")
        return

    # Recording setup

    rec = None
    if not args.no_record:
        rec = RolloutRecorder("sweep_cubes")

        # Save scene images
        if caps is not None:
            for cam in caps:
                cv2.imwrite(os.path.join(rec.out_dir, f"scene_{cam}.png"),
                            cv2.cvtColor(caps[cam]["rgb"], cv2.COLOR_RGB2BGR))

        # Save BT
        if bt_yaml_str:
            rec.save_bt(bt_yaml_str)

        # Save annotated images via detect_scene (if we have captures)
        if caps is not None:
            # Store caps on recorder so detect_scene can use them
            rec._caps = caps
            rec._cals = cals
            try:
                all_prompts = ["brush"] + object_labels + ["black dustpan"]
                rec.detect_scene(all_prompts)
            except Exception as e:
                print(f"  Annotated image save failed: {e}")

        rec.start_recording()

    # Execute

    print("\nEXECUTE")
    robot = FrankaBambooDriver(robot_ip=args.ip)
    robot.connect()

    success = False
    try:
        # Home + open gripper
        robot.go_home(velocity=0.3)
        robot.open_gripper()
        time.sleep(0.3)
        if rec:
            rec.mark_keyframe("start")

        if do_grasp:
            print("\nGRASP BRUSH")
            if not execute_grasp_brush(robot, rec, coords):
                raise RuntimeError("Brush grasp failed")

            print("\nREORIENT")
            execute_reorient(robot, rec, coords)

        if do_sweep:
            print("\nSWEEP")
            execute_sweep(robot, rec, coords)

        robot.go_home(velocity=0.3)
        robot.open_gripper()
        time.sleep(0.3)

        if rec:
            rec.mark_keyframe("end")
        success = True
        print("\nSUCCESS")

    except Exception as e:
        print(f"\nFAILED: {e}")
        traceback.print_exc()
        try:
            robot.open_gripper()
        except Exception:
            pass
        try:
            robot.go_home(velocity=0.3)
        except Exception:
            pass
        if rec:
            rec.mark_keyframe("fail")

    robot.disconnect()

    # Save recording

    if rec:
        print("\nSAVE")
        rec.stop_recording()
        rec.set_result(success=success,
                       notes="clean sweep" if success else "execution error")


if __name__ == "__main__":
    main()

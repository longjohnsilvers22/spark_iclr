#!/usr/bin/env python3
"""Sponge-wash-plate on FR3 via Bamboo + pyroki IK.

Sequence: pick up sponge -> move to plate -> circular scrub (4 cycles) ->
          lift -> place sponge back -> home.

Uses pyroki IK, Bamboo torque control, RolloutRecorder for paper figures,
and optionally SPARKPlanner for BT generation via Gemini.

Usage:
    cd ~/spark/src
    python ../scripts/sponge_wash.py
    python ../scripts/sponge_wash.py --phase detect     # detection only
    python ../scripts/sponge_wash.py --phase grasp      # grasp sponge only
    python ../scripts/sponge_wash.py --phase scrub      # scrub only (needs --coords)
    python ../scripts/sponge_wash.py --no-record         # skip video recording
    python ../scripts/sponge_wash.py --coords /tmp/wash_coords.json  # skip detection
"""
from __future__ import annotations

import argparse
import json
import math
import os
import sys
import time
import traceback

import cv2
import numpy as np
import yaml
from PIL import Image
from scipy.spatial.transform import Rotation as R

# Path setup
_src = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "src")
if _src not in sys.path:
    sys.path.insert(0, os.path.abspath(_src))

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
os.environ.setdefault("SPARK_IK", "pinocchio")

# JAX caps (before any JAX import via pyroki)
os.environ.setdefault("JAX_PLATFORMS", "cpu")
os.environ.setdefault("XLA_PYTHON_CLIENT_PREALLOCATE", "false")
os.environ.setdefault("XLA_PYTHON_CLIENT_MEM_FRACTION", "0.15")

from spark_real.control.fr3_ik_pyroki import solve_ik  # noqa: E402
from spark_real.robots.franka.franka_bamboo_driver import FrankaBambooDriver  # noqa: E402
from detect_offline import open_kinects, capture, load_calibrations, backproject  # noqa: E402
from rollout_recorder import RolloutRecorder  # noqa: E402

# Constants
ORIENT_0 = R.from_rotvec([np.pi, 0, 0]).as_rotvec()          # 0deg yaw, top-down
HOME_Q = np.array([0.0, -0.785398, 0.0, -2.356194, -0.15, 1.570796, 0.785398])

APPROACH_HEIGHT = 0.08      # hover above object (m)
GRASP_Z_OFFSET = 0.003      # small offset above centroid Z for grasp
LIFT_HEIGHT = 0.10           # lift after grasp (m)
SCRUB_RADIUS = 0.030         # circular scrub radius (m)
SCRUB_CYCLES = 4             # number of circular scrub passes
SCRUB_STEPS_PER_CYCLE = 16   # waypoints per circle
SCRUB_PRESS_Z = 0.005        # how far below plate surface to press sponge (m)
SCRUB_VEL = 0.08             # joint velocity for scrub waypoints
PLACE_HEIGHT = 0.04          # release height above sponge origin (m)

GRASP_RETRIES = 5
GRASP_DZ = 0.003
GRASP_FORCE = 20             # sponge is compressible: low force
GRASP_WIDTH_THRESH = 0.005   # gripper width > this = grasped


# Detection

# Detect sponge and plate via SAM3, return 3D positions.
def detect_objects(caps, cals):
    # DELIBERATELY DEFERRED IMPORT: SPARKPerception pulls in torch, and the
    # Kinect capture (open_kinects/capture in main) must run BEFORE torch is
    # loaded in this process (torch loaded first segfaults pyk4a capture; see
    # rollout_recorder.py's subprocess workaround). Do NOT move to module top.
    from spark_real.perception.spark_perception import SPARKPerception

    perc = SPARKPerception()
    perc.load_models(load_da3=False)

    results = {}
    prompts = ["sponge", "plate"]

    for cam in ["sideview", "birdview"]:
        c = caps[cam]
        pil = Image.fromarray(c["rgb"])
        intr = {k: c[k] for k in ("fx", "fy", "cx", "cy")}

        for prompt in prompts:
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

            # Centroid backprojection
            mcx, mcy = float(xs.mean()), float(ys.mean())
            pos = backproject(mcx, mcy, c["depth"], intr, cals[cam])

            # Also sample mask for Z statistics
            n = min(200, len(ys))
            idx = np.linspace(0, len(ys) - 1, n, dtype=int)
            zs = []
            for i in idx:
                p = backproject(float(xs[i]), float(ys[i]),
                                c["depth"], intr, cals[cam])
                if p is not None:
                    zs.append(p[2])
            max_z = max(zs) if zs else None
            min_z = min(zs) if zs else None

            key = f"{cam}_{prompt}"
            results[key] = {
                "pos": pos.tolist() if pos is not None else None,
                "conf": score,
                "max_z": max_z,
                "min_z": min_z,
                "centroid_px": (mcx, mcy),
                "camera": cam,
                "label": prompt,
                "mask_area": len(xs),
            }
            if pos is not None:
                print(f"  {cam:10s} {prompt:10s} conf={score:.3f} "
                      f"x={pos[0]:.4f} y={pos[1]:.4f} z={pos[2]:.4f}")
            else:
                print(f"  {cam:10s} {prompt:10s} conf={score:.3f} NO DEPTH")

    # Pick best detection per object (prefer sideview for Z accuracy)
    def _pick(label):
        for cp in ["sideview", "birdview"]:
            d = results.get(f"{cp}_{label}")
            if d and d.get("pos") is not None:
                return d
        return None

    sponge = _pick("sponge")
    plate = _pick("plate")
    return sponge, plate, results


# Build a coords dict from detection results.
def build_coords(sponge_det, plate_det):
    if sponge_det is None or sponge_det.get("pos") is None:
        print("  ERROR: sponge not detected")
        return None
    if plate_det is None or plate_det.get("pos") is None:
        print("  ERROR: plate not detected")
        return None

    sp = sponge_det["pos"]
    pp = plate_det["pos"]
    return {
        "sponge_x": sp[0], "sponge_y": sp[1], "sponge_z": sp[2],
        "plate_x": pp[0], "plate_y": pp[1], "plate_z": pp[2],
        "sponge_conf": sponge_det.get("conf", 0),
        "plate_conf": plate_det.get("conf", 0),
    }


# IK Dry Run

# Pre-verify all critical IK solutions before moving the robot.
def dry_run_ik(d):
    orient = ORIENT_0.tolist()
    q = HOME_Q.copy()

    waypoints = [
        ("hover_sponge", [d["sponge_x"], d["sponge_y"],
                          d["sponge_z"] + APPROACH_HEIGHT]),
        ("grasp_sponge", [d["sponge_x"], d["sponge_y"],
                          d["sponge_z"] + GRASP_Z_OFFSET]),
        ("lift_sponge",  [d["sponge_x"], d["sponge_y"],
                          d["sponge_z"] + LIFT_HEIGHT]),
        ("above_plate",  [d["plate_x"], d["plate_y"],
                          d["plate_z"] + APPROACH_HEIGHT]),
        ("scrub_center", [d["plate_x"], d["plate_y"],
                          d["plate_z"] + SCRUB_PRESS_Z]),
    ]

    # Add one circle of scrub waypoints
    cx, cy = d["plate_x"], d["plate_y"]
    sz = d["plate_z"] + SCRUB_PRESS_Z
    for i in range(SCRUB_STEPS_PER_CYCLE):
        theta = 2.0 * math.pi * i / SCRUB_STEPS_PER_CYCLE
        wx = cx + SCRUB_RADIUS * math.cos(theta)
        wy = cy + SCRUB_RADIUS * math.sin(theta)
        waypoints.append((f"scrub_{i}", [wx, wy, sz]))

    # Place sponge back
    waypoints.append(("place_sponge", [d["sponge_x"], d["sponge_y"],
                                       d["sponge_z"] + PLACE_HEIGHT]))

    ok = True
    for name, pos in waypoints:
        qs = solve_ik(np.array(pos), orient, q_seed=q)
        if qs is None:
            print(f"  {name}: FAIL ({pos[0]:.3f}, {pos[1]:.3f}, {pos[2]:.3f})")
            ok = False
        else:
            q = qs
            print(f"  {name}: OK")
    return ok


# Execution helpers

# Solve IK and move. Returns new joint config or raises.
def move_ik(robot, pos, orient, q_seed, label="", velocity=0.15):
    qs = solve_ik(np.array(pos), orient, q_seed=q_seed)
    if qs is None:
        raise RuntimeError(f"IK failed for {label}: pos={pos}")
    robot.move_to_joint_config(qs.tolist(), velocity=velocity)
    time.sleep(0.15)
    return qs


# Descend and grasp with retries, lowering Z each attempt.
def grasp_with_retries(robot, gx, gy, gz, orient, q_seed):
    z = gz
    for attempt in range(GRASP_RETRIES):
        q = move_ik(robot, [gx, gy, z], orient, q_seed,
                    label=f"grasp_attempt_{attempt}", velocity=0.06)
        time.sleep(0.2)
        robot.close_gripper(force=GRASP_FORCE)
        time.sleep(0.8)
        w = robot._gripper_width()
        print(f"  Attempt {attempt + 1}: width={w:.4f} z={z:.4f}")
        if w > GRASP_WIDTH_THRESH:
            return q, True
        robot.open_gripper()
        time.sleep(0.3)
        z -= GRASP_DZ
        q_seed = q
    return q_seed, False


# Execute circular scrub motion on the plate surface.
def circular_scrub(robot, cx, cy, cz, orient, q_seed, cycles=SCRUB_CYCLES):
    q = q_seed
    for cycle in range(cycles):
        print(f"  Scrub cycle {cycle + 1}/{cycles}")
        for i in range(SCRUB_STEPS_PER_CYCLE):
            theta = 2.0 * math.pi * i / SCRUB_STEPS_PER_CYCLE
            wx = cx + SCRUB_RADIUS * math.cos(theta)
            wy = cy + SCRUB_RADIUS * math.sin(theta)
            qs = solve_ik(np.array([wx, wy, cz]), orient, q_seed=q)
            if qs is not None:
                robot.move_to_joint_config(qs.tolist(), velocity=SCRUB_VEL)
                q = qs
            time.sleep(0.02)
    return q


# BT Generation (optional)

# Generate a BT plan via SPARKPlanner and save it.
def generate_bt(d, rec):
    try:
        # DELIBERATELY DEFERRED IMPORT: SPARKPlanner pulls in torch, which
        # must not load before the Kinect capture (see detect_objects note).
        # Also guarded: BT generation is optional and non-fatal.
        from spark_real.planning.spark_planner import SPARKPlanner

        instruction = "Pick up the sponge, move it to the plate, scrub the plate in a circular pattern, then place the sponge back."
        keypoint_labels = ["sponge", "plate"]
        detection_details = [
            {"label": "sponge", "confidence": d.get("sponge_conf", 0.9),
             "position_3d": [d["sponge_x"], d["sponge_y"], d["sponge_z"]]},
            {"label": "plate", "confidence": d.get("plate_conf", 0.9),
             "position_3d": [d["plate_x"], d["plate_y"], d["plate_z"]]},
        ]

        planner = SPARKPlanner()
        result = planner.generate_score(
            instruction, keypoint_labels=keypoint_labels,
            detection_details=detection_details)

        if result:
            yaml_str = yaml.safe_dump(result, sort_keys=False)
            print(f"  BT generated ({len(yaml_str)} chars)")
            if rec:
                rec.save_bt(yaml_str)
            return result
    except Exception as e:
        print(f"  BT generation failed (non-fatal): {e}")
    return None


# Main

def main():
    parser = argparse.ArgumentParser(description="Sponge wash plate on FR3")
    parser.add_argument("--phase", choices=["detect", "grasp", "scrub", "all"],
                        default="all",
                        help="Run a single phase or all (default: all)")
    parser.add_argument("--no-record", action="store_true",
                        help="Skip video recording")
    parser.add_argument("--coords", type=str, default=None,
                        help="Path to saved coords JSON (skip detection)")
    args = parser.parse_args()

    # Detection or load coords
    caps = None
    all_detections = {}

    if args.coords:
        d = json.load(open(args.coords))
        print(f"LOADED COORDS from {args.coords}")
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
        sponge_det, plate_det, all_detections = detect_objects(caps, cals)
        d = build_coords(sponge_det, plate_det)
        if d is None:
            print("Detection failed")
            return

        # Save coords for reuse
        json.dump(d, open("/tmp/wash_coords.json", "w"), indent=2)
        print(f"  Coords saved to /tmp/wash_coords.json")

    print(f"\n  Sponge: ({d['sponge_x']:.4f}, {d['sponge_y']:.4f}, {d['sponge_z']:.4f})")
    print(f"  Plate:  ({d['plate_x']:.4f}, {d['plate_y']:.4f}, {d['plate_z']:.4f})")

    if args.phase == "detect":
        print("\n  Phase 'detect' complete. Use --coords /tmp/wash_coords.json for next phase.")
        return

    # IK Dry Run
    print("\nDRY RUN")
    if not dry_run_ik(d):
        print("ABORT: IK dry run failed")
        return

    # Recording setup
    rec = None
    if not args.no_record:
        rec = RolloutRecorder("sponge_wash")
        # Save scene images
        if caps is not None:
            for cam in caps:
                cv2.imwrite(
                    os.path.join(rec.out_dir, f"scene_{cam}.png"),
                    cv2.cvtColor(caps[cam]["rgb"], cv2.COLOR_RGB2BGR))
                np.save(os.path.join(rec.out_dir, f"depth_{cam}.npy"),
                        caps[cam]["depth"])
        # Save detection results
        def _json_safe(obj):
            if isinstance(obj, np.ndarray):
                return obj.tolist()
            if isinstance(obj, (np.float32, np.float64)):
                return float(obj)
            return str(obj)
        if all_detections:
            with open(os.path.join(rec.out_dir, "detections.json"), "w") as f:
                json.dump(all_detections, f, indent=2, default=_json_safe)

        # Generate BT plan (optional, non-blocking)
        generate_bt(d, rec)

        rec.start_recording()

    # Execute
    print("\nEXECUTE")
    orient = ORIENT_0.tolist()

    robot = FrankaBambooDriver(robot_ip="172.16.0.2")
    robot.connect()

    success = False

    try:
        # Home + open gripper
        robot.go_home(velocity=0.3)
        robot.open_gripper()
        time.sleep(0.3)
        if rec:
            rec.mark_keyframe("start")

        q = HOME_Q.copy()
        do_grasp = args.phase in ("grasp", "all")
        do_scrub = args.phase in ("scrub", "all")

        if do_grasp:
            # Phase: Hover above sponge
            print("\nHOVER SPONGE")
            q = move_ik(robot,
                        [d["sponge_x"], d["sponge_y"],
                         d["sponge_z"] + APPROACH_HEIGHT],
                        orient, q, label="hover_sponge", velocity=0.20)

            # Phase: Descend + grasp sponge
            print("\nGRASP SPONGE")
            q, grasped = grasp_with_retries(
                robot, d["sponge_x"], d["sponge_y"],
                d["sponge_z"] + GRASP_Z_OFFSET, orient, q)
            if not grasped:
                print("  FAILED to grasp sponge")
                robot.open_gripper()
                time.sleep(0.3)
                robot.go_home(velocity=0.3)
                if rec:
                    rec.mark_keyframe("fail_grasp")
                    rec.stop_recording()
                    rec.set_result(success=False, notes="failed to grasp sponge")
                robot.disconnect()
                return
            if rec:
                rec.mark_keyframe("grasp_sponge")

            # Phase: Lift sponge
            print("\nLIFT")
            q = move_ik(robot,
                        [d["sponge_x"], d["sponge_y"],
                         d["sponge_z"] + LIFT_HEIGHT],
                        orient, q, label="lift", velocity=0.15)

        if args.phase == "grasp":
            print("\n  Phase 'grasp' complete.")
            robot.open_gripper()
            time.sleep(0.3)
            robot.go_home(velocity=0.3)
            if rec:
                rec.mark_keyframe("end_grasp_phase")
                rec.stop_recording()
                rec.set_result(success=True, notes="grasp phase only")
            robot.disconnect()
            return

        if do_scrub:
            # Phase: Move above plate
            print("\nMOVE ABOVE PLATE")
            q = move_ik(robot,
                        [d["plate_x"], d["plate_y"],
                         d["plate_z"] + APPROACH_HEIGHT],
                        orient, q, label="above_plate", velocity=0.18)
            if rec:
                rec.mark_keyframe("above_plate")

            # Phase: Descend to plate surface
            print("\nDESCEND TO PLATE")
            scrub_z = d["plate_z"] + SCRUB_PRESS_Z
            q = move_ik(robot,
                        [d["plate_x"], d["plate_y"], scrub_z],
                        orient, q, label="scrub_start", velocity=0.08)
            if rec:
                rec.mark_keyframe("scrub_start")

            # Phase: Circular scrub
            print("\nCIRCULAR SCRUB")
            q = circular_scrub(robot, d["plate_x"], d["plate_y"],
                               scrub_z, orient, q)
            if rec:
                rec.mark_keyframe("scrub_done")

            # Phase: Lift from plate
            print("\nLIFT FROM PLATE")
            q = move_ik(robot,
                        [d["plate_x"], d["plate_y"],
                         d["plate_z"] + LIFT_HEIGHT],
                        orient, q, label="lift_from_plate", velocity=0.15)

        if args.phase == "scrub":
            print("\n  Phase 'scrub' complete.")
            robot.open_gripper()
            time.sleep(0.3)
            robot.go_home(velocity=0.3)
            if rec:
                rec.mark_keyframe("end_scrub_phase")
                rec.stop_recording()
                rec.set_result(success=True, notes="scrub phase only")
            robot.disconnect()
            return

        # Phase: Place sponge back
        print("\nPLACE SPONGE BACK")
        q = move_ik(robot,
                    [d["sponge_x"], d["sponge_y"],
                     d["sponge_z"] + APPROACH_HEIGHT],
                    orient, q, label="above_sponge_origin", velocity=0.18)
        q = move_ik(robot,
                    [d["sponge_x"], d["sponge_y"],
                     d["sponge_z"] + PLACE_HEIGHT],
                    orient, q, label="place_sponge", velocity=0.08)
        robot.open_gripper()
        time.sleep(0.4)
        if rec:
            rec.mark_keyframe("place_sponge")

        # Retract + Home
        print("\nRETRACT + HOME")
        q = move_ik(robot,
                    [d["sponge_x"], d["sponge_y"],
                     d["sponge_z"] + APPROACH_HEIGHT],
                    orient, q, label="retract", velocity=0.20)
        robot.go_home(velocity=0.3)
        if rec:
            rec.mark_keyframe("end")
        print("\nSUCCESS")
        success = True

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

    if rec:
        print("\nSAVE")
        rec.stop_recording()
        rec.set_result(success=success,
                       notes="clean run" if success else "exception during execution")


if __name__ == "__main__":
    main()

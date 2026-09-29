#!/usr/bin/env python3
"""Silverware sorting on FR3 via Bamboo + pyroki.

Detects forks, knives, and spoons scattered on the table, picks each one up
with OBB-aligned top-down grasp, and places it in the correct tray slot.

Tray slot assignment (ordered along the tray's major axis):
  slot 0 -> fork
  slot 1 -> knife
  slot 2 -> spoon

Uses:
  - FrankaBambooDriver with pyroki IK
  - RolloutRecorder for video/keyframe capture
  - SAM3 text prompts via detect_offline for detection
  - OBB yaw from mask PCA for elongated-object alignment
  - Slot detector (perception.slot_detector) when dividers are visible,
    or fallback to evenly-spaced poses along the tray's major axis

Usage:
    cd ~/spark/src
    python ../scripts/silverware_sort.py
    python ../scripts/silverware_sort.py --no-record
    python ../scripts/silverware_sort.py --coords /tmp/sort_coords.json
    python ../scripts/silverware_sort.py --utensils "fork,spoon"
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
import pyk4a
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
from spark_real.robots.franka.franka_driver import FrankaDriver  # noqa: E402
from detect_offline import open_kinects, capture, load_calibrations, backproject  # noqa: E402
from rollout_recorder import RolloutRecorder  # noqa: E402

# Constants
DEFAULT_IP = "172.16.0.2"
HOME_Q = np.array([0.0, -0.785398, 0.0, -2.356194, -0.15, 1.570796, 0.785398])

# Orientations
ORIENT_DOWN = R.from_rotvec([np.pi, 0, 0]).as_rotvec()

# Grasp parameters
HOVER_Z = 0.08              # hover above object before descend
GRASP_Z_BIAS = -0.005       # small descent bias for thin utensils
GRASP_FORCE = 40            # moderate force for silverware
GRASP_RETRIES = 4
GRASP_DZ = 0.003            # descend per retry
LIFT_Z = 0.12               # lift height after grasping

# Place parameters
PLACE_HOVER_Z = 0.12        # hover above slot before descend
PLACE_RELEASE_Z_OFFSET = 0.04  # release height above slot floor

# Transit
TRANSIT_Z = 0.15            # safe transit height

# Tray slot assignment: utensil type -> slot index
UTENSIL_SLOT = {"fork": 0, "knife": 1, "spoon": 2}


# Detection

# Detect utensils and tray via SAM3. Returns coords dict or None.
def detect_scene(caps, cals, utensil_labels):
    # DELIBERATELY DEFERRED IMPORT: SPARKPerception pulls in torch, and the
    # Kinect capture (open_kinects/capture in main) must run BEFORE torch is
    # loaded in this process (torch loaded first segfaults pyk4a capture; see
    # rollout_recorder.py's subprocess workaround). Do NOT move to module top.
    from spark_real.perception.spark_perception import SPARKPerception

    perc = SPARKPerception()
    perc.load_models(load_da3=False)

    all_prompts = utensil_labels + ["tray"]
    results = {}

    for cam in ["birdview", "sideview"]:
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

            # OBB yaw via PCA on mask pixels (world-frame)
            obb_yaw = 0.0
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
                "mask": mask,
            }

    # Build coordinates dict
    coords = {"utensils": []}

    # Utensils: prefer birdview (top-down gives better OBB for elongated objects)
    for label in utensil_labels:
        det = _best_det(results, label, prefer="birdview")
        if det and det["pos"]:
            coords["utensils"].append({
                "label": label,
                "x": det["pos"][0],
                "y": det["pos"][1],
                "z": det["pos"][2],
                "obb_yaw": det.get("obb_yaw", 0.0),
            })
        else:
            print(f"  WARNING: '{label}' not detected")

    if not coords["utensils"]:
        print("  ERROR: no utensils detected")
        return None

    # Tray: prefer birdview
    tray_det = _best_det(results, "tray", prefer="birdview")
    if tray_det and tray_det["pos"]:
        coords["tray_x"] = tray_det["pos"][0]
        coords["tray_y"] = tray_det["pos"][1]
        coords["tray_z"] = tray_det["pos"][2]
        coords["tray_obb_yaw"] = tray_det.get("obb_yaw", 0.0)

        # Try slot detection from birdview depth + mask
        tray_mask = tray_det.get("mask")
        if tray_mask is not None:
            slots = _detect_tray_slots(
                caps, cals, tray_mask, tray_det["camera"],
                n_slots=len(utensil_labels),
            )
            if slots:
                coords["slots"] = [
                    {"slot_idx": s["slot_idx"],
                     "x": float(s["world_xyz"][0]),
                     "y": float(s["world_xyz"][1]),
                     "z": float(s["world_xyz"][2])}
                    for s in slots
                ]

        # Fallback: divide tray Y range into N slots
        if "slots" not in coords:
            coords["slots"] = _fallback_slots(coords, len(utensil_labels))
    else:
        print("  WARNING: tray not detected, will use fixed slot positions")
        coords["slots"] = _fallback_slots(coords, len(utensil_labels))

    return coords


# Pick best detection for a label, preferring the given camera.
def _best_det(results, label, prefer="birdview"):
    pref = results.get(f"{prefer}_{label}")
    other = "sideview" if prefer == "birdview" else "birdview"
    alt = results.get(f"{other}_{label}")
    if pref and pref.get("pos"):
        return pref
    return alt


# Run the depth-normal slot detector on the tray mask.
def _detect_tray_slots(caps, cals, tray_mask, camera, n_slots):
    try:
        # DELIBERATELY DEFERRED IMPORT: slot_detector pulls in torch, which
        # must not load before the Kinect capture (see detect_scene note).
        from spark_real.perception.slot_detector import detect_slots

        c = caps[camera]
        depth_mm = c["depth"]
        depth_m = depth_mm.astype(np.float32) / 1000.0
        intr = np.array([
            [c["fx"], 0, c["cx"]],
            [0, c["fy"], c["cy"]],
            [0, 0, 1],
        ], dtype=np.float64)
        T = cals[camera]["T"]
        ds = cals[camera].get("depth_scale", 1.0)

        slots = detect_slots(
            rgb=c["rgb"],
            depth_m=depth_m,
            tray_mask=tray_mask,
            K=intr,
            T_cam_to_base=T,
            depth_scale=ds,
            n_fallback_slots=n_slots,
            lift_offset_m=PLACE_RELEASE_Z_OFFSET,
        )
        if len(slots) >= 2:
            print(f"  Slot detector found {len(slots)} slots (mode={slots[0]['mode']})")
            return slots
    except Exception as e:
        print(f"  Slot detector failed: {e}")
    return None


# Divide tray Y range into N evenly-spaced slots.
def _fallback_slots(coords, n_slots):
    tray_x = coords.get("tray_x", 0.45)
    tray_y = coords.get("tray_y", 0.0)
    tray_z = coords.get("tray_z", 0.0) + PLACE_RELEASE_Z_OFFSET

    # Estimate tray extent: ~0.25 m along Y (typical silverware tray)
    tray_half_len = 0.12
    y_start = tray_y - tray_half_len
    y_end = tray_y + tray_half_len
    slot_len = (y_end - y_start) / max(n_slots, 1)
    slots = []
    for i in range(n_slots):
        sy = y_start + slot_len * (i + 0.5)
        slots.append({"slot_idx": i, "x": tray_x, "y": sy, "z": tray_z})
    print(f"  Using {n_slots} fallback slots along Y=[{y_start:.3f}, {y_end:.3f}]")
    return slots


# OBB yaw computation for top-down grasps

def compute_grasp_yaw(obb_yaw_raw):
    """
    Convert raw OBB yaw to grasp yaw for an elongated object.

    PCA on mask pixels gives the major axis angle in image space.
    Normalise to [0, pi), then fold into [-pi/2, pi/2) so the
    gripper fingers always close across the short axis.
    """
    orient = float(obb_yaw_raw) % np.pi
    if orient > np.pi / 2:
        orient -= np.pi
    return orient


def grasp_orientation(yaw):
    """
    Build the full SE(3) orientation (rotvec) for a top-down grasp
    with the given yaw angle.

    R_grasp = Rz(yaw) * Rx(pi)   [straight down, rotated about Z]
    """
    return (R.from_euler("z", yaw) * R.from_rotvec([np.pi, 0, 0])).as_rotvec()


# IK dry run

# Pre-verify IK reachability for key waypoints.
def dry_run_ik(coords):
    q = HOME_Q.copy()
    orient = ORIENT_DOWN.tolist()

    waypoints = []
    for u in coords["utensils"]:
        label = u["label"]
        yaw = compute_grasp_yaw(u["obb_yaw"])
        yaw_orient = grasp_orientation(yaw).tolist()
        waypoints += [
            (f"hover_{label}", [u["x"], u["y"], HOVER_Z], yaw_orient),
            (f"grasp_{label}", [u["x"], u["y"], u["z"] + GRASP_Z_BIAS], yaw_orient),
            (f"lift_{label}", [u["x"], u["y"], LIFT_Z], orient),
        ]

    for s in coords["slots"]:
        idx = s["slot_idx"]
        waypoints += [
            (f"slot_{idx}_hover", [s["x"], s["y"], PLACE_HOVER_Z], orient),
            (f"slot_{idx}_place", [s["x"], s["y"], s["z"]], orient),
        ]

    ok = True
    for name, pos, ori in waypoints:
        qs = solve_ik(np.array(pos), ori, q_seed=q)
        if qs is None:
            print(f"  {name}: FAIL ({pos[0]:.3f}, {pos[1]:.3f}, {pos[2]:.3f})")
            ok = False
        else:
            print(f"  {name}: OK   ({pos[0]:.3f}, {pos[1]:.3f}, {pos[2]:.3f})")
            q = qs
    return ok


# Execution helpers

# Move via joint-space IK. Falls back to move_linear.
def move_to(robot, pos, orient, velocity=0.15, q_seed=None):
    q = q_seed if q_seed is not None else np.array(robot.get_joint_positions())
    qs = solve_ik(np.array(pos), orient if isinstance(orient, list) else orient.tolist(),
                  q_seed=q)
    if qs is not None:
        robot.move_to_joint_config(qs.tolist(), velocity=velocity)
        return qs
    else:
        robot.move_linear(list(pos) + list(orient), velocity=velocity)
        return np.array(robot.get_joint_positions())


# Hover, descend, grasp a single utensil with OBB yaw alignment.
def pick_utensil(robot, rec, utensil):
    label = utensil["label"]
    ux, uy, uz = utensil["x"], utensil["y"], utensil["z"]
    yaw = compute_grasp_yaw(utensil["obb_yaw"])
    yaw_orient = grasp_orientation(yaw)

    print(f"\nPICK {label}")
    print(f"    pos=({ux:.4f}, {uy:.4f}, {uz:.4f})  yaw={np.degrees(yaw):.1f} deg")

    # Transit to hover above utensil
    q = move_to(robot, [ux, uy, HOVER_Z], yaw_orient.tolist(), velocity=0.20)
    time.sleep(0.3)

    # Descend to grasp height
    gz = uz + GRASP_Z_BIAS
    q = move_to(robot, [ux, uy, gz], yaw_orient.tolist(), velocity=0.06, q_seed=q)
    time.sleep(0.3)

    # Grasp with retries
    grasped = False
    for attempt in range(GRASP_RETRIES):
        robot.close_gripper(force=GRASP_FORCE)
        time.sleep(0.8)
        w = robot._gripper_width()
        print(f"    attempt {attempt + 1}: width={w:.4f}  z={gz:.4f}")
        if w > 0.001:
            grasped = True
            break
        robot.open_gripper()
        time.sleep(0.3)
        gz -= GRASP_DZ
        q = move_to(robot, [ux, uy, gz], yaw_orient.tolist(), velocity=0.03, q_seed=q)
        time.sleep(0.2)

    if not grasped:
        print(f"    FAILED to grasp {label}")
        robot.open_gripper()
        time.sleep(0.3)
        return False

    if rec:
        rec.mark_keyframe(f"grasp_{label}")

    # Lift
    q = move_to(robot, [ux, uy, LIFT_Z], ORIENT_DOWN.tolist(), velocity=0.12, q_seed=q)
    time.sleep(0.3)

    print(f"    {label} grasped and lifted")
    return True


# Move above tray slot, descend, release.
def place_in_slot(robot, rec, slot, label):
    sx, sy, sz = slot["x"], slot["y"], slot["z"]
    orient = ORIENT_DOWN.tolist()

    print(f"\nPLACE {label} -> slot {slot['slot_idx']}")
    print(f"    slot=({sx:.4f}, {sy:.4f}, {sz:.4f})")

    # Transit above slot
    q = move_to(robot, [sx, sy, PLACE_HOVER_Z], orient, velocity=0.15)
    time.sleep(0.2)

    # Descend to release height
    q = move_to(robot, [sx, sy, sz], orient, velocity=0.06, q_seed=q)
    time.sleep(0.3)

    # Release
    robot.open_gripper()
    time.sleep(0.5)

    if rec:
        rec.mark_keyframe(f"place_{label}_slot{slot['slot_idx']}")

    # Retract upward
    q = move_to(robot, [sx, sy, PLACE_HOVER_Z], orient, velocity=0.15, q_seed=q)
    time.sleep(0.2)

    print(f"    {label} placed in slot {slot['slot_idx']}")
    return True


def assign_slot(label, slots):
    """
    Map a utensil label to its target slot.

    Uses the UTENSIL_SLOT lookup; if the label contains a known utensil
    type (fork/knife/spoon), map to that index. Clamp to available
    slot count.
    """
    for utype, idx in UTENSIL_SLOT.items():
        if utype in label.lower():
            return slots[min(idx, len(slots) - 1)]
    # Unknown utensil type, put in the last slot
    return slots[-1]


# Main

def main():
    parser = argparse.ArgumentParser(description="Silverware sorting on FR3")
    parser.add_argument("--no-record", action="store_true",
                        help="Skip video recording")
    parser.add_argument("--coords", type=str, default=None,
                        help="Path to saved coords JSON (skip detection)")
    parser.add_argument("--utensils", type=str, default="fork,knife,spoon",
                        help="Comma-separated utensil labels "
                             "(default: 'fork,knife,spoon')")
    parser.add_argument("--ip", type=str, default=DEFAULT_IP,
                        help=f"Robot IP (default: {DEFAULT_IP})")
    args = parser.parse_args()

    utensil_labels = [s.strip() for s in args.utensils.split(",") if s.strip()]

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
        coords = detect_scene(caps, cals, utensil_labels)
        if coords is None:
            print("Detection failed")
            return

        # Save coords for reuse
        json.dump(coords, open("/tmp/sort_coords.json", "w"), indent=2)
        print(f"  Coords saved to /tmp/sort_coords.json")

    # Print detections
    print(f"\n  Utensils ({len(coords['utensils'])}):")
    for u in coords["utensils"]:
        yaw = compute_grasp_yaw(u["obb_yaw"])
        print(f"    {u['label']:8s}: ({u['x']:.4f}, {u['y']:.4f}, {u['z']:.4f}) "
              f"yaw={np.degrees(yaw):.1f} deg")

    print(f"  Slots ({len(coords['slots'])}):")
    for s in coords["slots"]:
        print(f"    slot {s['slot_idx']}: ({s['x']:.4f}, {s['y']:.4f}, {s['z']:.4f})")

    # Dry run IK

    print("\nDRY RUN")
    if not dry_run_ik(coords):
        print("ABORT: IK dry run failed")
        return

    # Recording setup

    rec = None
    if not args.no_record:
        rec = RolloutRecorder("silverware_sort")

        # Save scene images
        if caps is not None:
            for cam in caps:
                cv2.imwrite(os.path.join(rec.out_dir, f"scene_{cam}.png"),
                            cv2.cvtColor(caps[cam]["rgb"], cv2.COLOR_RGB2BGR))

        # Save annotated images
        if caps is not None:
            try:
                all_prompts = utensil_labels + ["tray"]
                rec._caps = caps
                rec._cals = cals
                rec.detect_scene(all_prompts)
            except Exception as e:
                print(f"  Annotated image save failed: {e}")

        rec.start_recording()

    # Execute

    print("\nEXECUTE")
    robot = FrankaDriver(robot_ip=args.ip)
    robot.connect()

    success = False
    n_placed = 0
    try:
        # Home + open gripper
        robot.go_home(velocity=0.3)
        robot.open_gripper()
        time.sleep(0.3)
        if rec:
            rec.mark_keyframe("start")

        slots = coords["slots"]

        for u in coords["utensils"]:
            label = u["label"]

            # Pick
            if not pick_utensil(robot, rec, u):
                print(f"  Skipping {label} (grasp failed)")
                robot.go_home(velocity=0.3)
                time.sleep(0.3)
                continue

            # Place in assigned slot
            slot = assign_slot(label, slots)
            place_in_slot(robot, rec, slot, label)

            # Return home between utensils
            robot.go_home(velocity=0.3)
            time.sleep(0.3)
            n_placed += 1

        if rec:
            rec.mark_keyframe("end")
        success = n_placed == len(coords["utensils"])
        status = "SUCCESS" if success else "PARTIAL"
        print(f"\n{status}: placed {n_placed}/{len(coords['utensils'])} utensils")

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
        rec.set_result(
            success=success,
            notes=f"placed {n_placed}/{len(coords['utensils'])} utensils",
        )


if __name__ == "__main__":
    main()

#!/usr/bin/env python3
"""Single-arm t-shirt fold on FR3 via Bamboo + pyroki.

Sequence: +Y sleeve, -Y sleeve, hem to collar.
Uses pyroki IK (avoids J5 singularity), Bamboo torque control,
and RolloutRecorder for paper figures.

Usage:
    cd ~/spark/src
    python ../scripts/fold_shirt.py
    python ../scripts/fold_shirt.py --phase sleeve   # sleeves only
    python ../scripts/fold_shirt.py --phase hem      # hem only (after sleeves)
    python ../scripts/fold_shirt.py --no-record      # skip video recording
"""
from __future__ import annotations

import argparse
import json
import sys
import os
import time
import traceback

import cv2
import numpy as np
from PIL import Image
from scipy.spatial.transform import Rotation as R

_src = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "src")
if _src not in sys.path:
    sys.path.insert(0, os.path.abspath(_src))

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
os.environ.setdefault("SPARK_IK", "pinocchio")

from spark_real.control.fr3_ik_pyroki import solve_ik
from spark_real.robots.franka.franka_bamboo_driver import FrankaBambooDriver
from detect_offline import open_kinects, capture, load_calibrations, backproject
from rollout_recorder import RolloutRecorder

ORIENT_90 = (R.from_euler("z", np.pi / 2) * R.from_rotvec([np.pi, 0, 0])).as_rotvec()
ORIENT_0 = R.from_rotvec([np.pi, 0, 0]).as_rotvec()
HOME_Q = np.array([0.0, -0.785398, 0.0, -2.356194, -0.15, 1.570796, 0.785398])

ARC_PEAK = 0.030
LAND_Z = 0.005
LIFT_Z = 0.08
SLEEVE_OFFSET = 0.010
COLLAR_X_MAX = 0.65
FOLD_PAST_CENTER = 0.08
GRASP_Z_START = -0.02
GRASP_RETRIES = 5
GRASP_DZ = 0.003


def detect_shirt(caps, cals):
    # DELIBERATELY DEFERRED IMPORT: SPARKPerception pulls in torch, and the
    # Kinect capture in main ("kinects before torch") must run BEFORE torch
    # is loaded in this process (torch loaded first segfaults pyk4a capture;
    # see rollout_recorder.py's subprocess workaround). Do NOT move to
    # module top.
    from spark_real.perception.spark_perception import SPARKPerception

    perc = SPARKPerception()
    perc.load_models(load_da3=False)

    def _get_sleeve(prompt, cam):
        c = caps[cam]
        pil = Image.fromarray(c["rgb"])
        intr = {k: c[k] for k in ("fx", "fy", "cx", "cy")}
        state = perc._sam3.set_image(pil)
        state = perc._sam3.set_text_prompt(prompt=prompt, state=state)
        masks = state.get("masks")
        scores = state.get("scores")
        if masks is None:
            return None
        best_mask = None
        best_n = 0
        for mi in range(masks.shape[0]):
            m = masks[mi].cpu().numpy().squeeze()
            area = int(m.sum())
            nv = int((c["depth"][m > 0] > 0).sum())
            if area > 50000 or nv < 100:
                continue
            if nv > best_n:
                best_n = nv
                best_mask = m
        if best_mask is None:
            return None
        ys, xs = np.where(best_mask > 0)
        n = min(200, len(ys))
        idx = np.linspace(0, len(ys) - 1, n, dtype=int)
        pts = [backproject(float(xs[i]), float(ys[i]), c["depth"], intr, cals[cam])
               for i in idx]
        pts = np.array([p for p in pts if p is not None])
        return pts if len(pts) >= 10 else None

    def _get_shirt(cam):
        c = caps[cam]
        pil = Image.fromarray(c["rgb"])
        intr = {k: c[k] for k in ("fx", "fy", "cx", "cy")}
        state = perc._sam3.set_image(pil)
        state = perc._sam3.set_text_prompt(prompt="shirt", state=state)
        masks = state.get("masks")
        scores = state.get("scores")
        if masks is None:
            return None
        best_mask = None
        best_n = 0
        for mi in range(masks.shape[0]):
            m = masks[mi].cpu().numpy().squeeze()
            nv = int((c["depth"][m > 0] > 0).sum())
            if nv > best_n:
                best_n = nv
                best_mask = m
        if best_n < 1000:
            return None
        ys, xs = np.where(best_mask > 0)
        idx = np.linspace(0, len(ys) - 1, min(500, len(ys)), dtype=int)
        pts = [backproject(float(xs[i]), float(ys[i]), c["depth"], intr, cals[cam])
               for i in idx]
        pts = np.array([p for p in pts if p is not None])
        med = np.median(pts, axis=0)
        return pts[np.linalg.norm(pts - med, axis=1) < 0.4]

    # Detect sleeves, collect all valid masks
    all_masks = []
    for prompt in ["left sleeve", "right sleeve"]:
        for cam in ["birdview", "sideview"]:
            pts = _get_sleeve(prompt, cam)
            if pts is not None:
                cx = float(pts[:, 0].mean())
                if cx < 0.45:
                    continue  # centroid_x < 0.45 is shirt body, not a sleeve
                all_masks.append({
                    "pts": pts,
                    "cy": float(pts[:, 1].mean()),
                    "min_y": float(pts[:, 1].min()),
                    "max_y": float(pts[:, 1].max()),
                })

    # Split into +Y and -Y groups by centroid
    pos_masks = [m for m in all_masks if m["cy"] > 0]
    neg_masks = [m for m in all_masks if m["cy"] <= 0]

    if not pos_masks or not neg_masks:
        print(f"  Sleeve detection incomplete (+Y={len(pos_masks)} -Y={len(neg_masks)}), using shirt mask fallback")
        for cam in ["birdview", "sideview"]:
            shirt_fb = _get_shirt(cam)
            if shirt_fb is None:
                continue
            cy_fb = float(shirt_fb[:, 1].mean())
            high_x = shirt_fb[shirt_fb[:, 0] > np.percentile(shirt_fb[:, 0], 60)]
            if not pos_masks:
                pr = high_x[high_x[:, 1] > cy_fb]
                if len(pr) > 10:
                    pos_masks = [{"pts": pr, "cy": float(pr[:, 1].mean()),
                                  "min_y": float(pr[:, 1].min()), "max_y": float(pr[:, 1].max())}]
            if not neg_masks:
                nr = high_x[high_x[:, 1] < cy_fb]
                if len(nr) > 10:
                    neg_masks = [{"pts": nr, "cy": float(nr[:, 1].mean()),
                                  "min_y": float(nr[:, 1].min()), "max_y": float(nr[:, 1].max())}]
            if pos_masks and neg_masks:
                break
        if not pos_masks or not neg_masks:
            print(f"  Fallback also failed: +Y={len(pos_masks)} -Y={len(neg_masks)}")
            return None

    # +Y sleeve: any mask works (they're all similar)
    pos_s = max(pos_masks, key=lambda m: m["max_y"])
    # -Y sleeve: pick the mask with the most negative min_y
    neg_s = min(neg_masks, key=lambda m: m["min_y"])

    # Grasp points: centroid X, literal outer Y edge with offset
    p_gx = float(pos_s["pts"][:, 0].mean())
    p_gy = float(pos_s["pts"][:, 1].max()) - SLEEVE_OFFSET
    n_gx = float(neg_s["pts"][:, 0].mean())
    n_gy = float(neg_s["pts"][:, 1].min()) + 0.020  # 20mm offset for -Y sleeve

    # Shirt for center/hem/collar
    shirt_pts = None
    for cam in ["sideview", "birdview"]:
        shirt_pts = _get_shirt(cam)
        if shirt_pts is not None:
            break
    if shirt_pts is None:
        print("  No shirt depth")
        return None

    center_y = float(shirt_pts[:, 1].mean())
    hem_region = shirt_pts[shirt_pts[:, 0] < np.percentile(shirt_pts[:, 0], 10)]
    hem_gx = float(hem_region[:, 0].mean())
    hem_gy = float(hem_region[:, 1].mean())
    collar_fx = min(float(shirt_pts[:, 0].max()), COLLAR_X_MAX)

    return {
        "p_gx": p_gx, "n_gx": n_gx,
        "p_gy": p_gy, "n_gy": n_gy,
        "center_y": center_y,
        "p_fold_y": center_y - FOLD_PAST_CENTER,
        "n_fold_y": center_y + FOLD_PAST_CENTER,
        "hem_gx": hem_gx, "hem_gy": hem_gy,
        "collar_fx": collar_fx, "collar_fy": center_y,
    }


NY_SEED = HOME_Q.copy()
NY_SEED[6] -= np.pi / 4  # J7-45: low joint-step seed for the -Y sleeve

HEM_SEED = HOME_Q.copy()  # hem uses 0deg yaw from HOME, a stable config


def dry_run(label, gx, gy, fx, fy, orient, keep_x=True, seed=None):
    q = seed.copy() if seed is not None else HOME_Q.copy()
    wps = [("h", [gx, gy, 0.05]), ("g", [gx, gy, GRASP_Z_START]),
           ("l", [gx, gy, GRASP_Z_START + LIFT_Z])]
    lift = GRASP_Z_START + LIFT_Z
    for i in range(1, 13):
        t = i / 12
        ax = gx if keep_x else gx + t * (fx - gx)
        ay = gy + t * (fy - gy)
        az = lift + ARC_PEAK * np.sin(t * np.pi) if i < 12 else LAND_Z
        wps.append((f"a{i}", [ax, ay, az]))
    for name, pos in wps:
        qs = solve_ik(np.array(pos), orient, q_seed=q)
        if qs is None:
            print(f"  {label} {name}: FAIL ({pos[0]:.3f},{pos[1]:.3f},{pos[2]:.3f})")
            return False
        # J7 continuity check
        if abs(qs[6] - q[6]) > 1.0:
            qs_c = qs.copy(); qs_c[6] = q[6]
            qs2 = solve_ik(np.array(pos), orient, q_seed=qs_c)
            if qs2 is not None and abs(qs2[6] - q[6]) < 1.0:
                qs = qs2
            else:
                qs[6] = q[6] + np.clip(qs[6] - q[6], -0.5, 0.5)
        q = qs
    print(f"  {label}: OK")
    return True


def grasp_fold(robot, rec, label, gx, gy, fx, fy, orient, keep_x=True):
    gz = GRASP_Z_START
    print(f"\n{label}")

    # Approach
    if "hem" in label.lower():
        q = np.array(robot.get_joint_positions())
        for wp in [[0.35, gy, 0.10], [gx, gy, 0.05]]:
            qs = solve_ik(np.array(wp), orient.tolist(), q_seed=q)
            if qs is not None:
                robot.move_to_joint_config(qs.tolist(), velocity=0.15)
                q = qs
                time.sleep(0.2)
    else:
        robot.move_linear([gx, gy, 0.05] + orient.tolist(), velocity=0.20)
        time.sleep(0.2)

    # Descend
    q = np.array(robot.get_joint_positions())
    qs = solve_ik(np.array([gx, gy, gz]), orient.tolist(), q_seed=q)
    if qs is not None:
        robot.move_to_joint_config(qs.tolist(), velocity=0.08)
    else:
        robot.move_linear([gx, gy, gz] + orient.tolist(), velocity=0.08)
    time.sleep(0.3)

    # Grasp with retries
    grasped = False
    for a in range(GRASP_RETRIES):
        robot.close_gripper(force=30)
        time.sleep(0.8)
        w = robot._gripper_width()
        print(f"  {a + 1}: w={w:.4f} z={gz:.4f}")
        if w > 0.001:
            grasped = True
            break
        robot.open_gripper()
        time.sleep(0.3)
        gz -= GRASP_DZ
        q = np.array(robot.get_joint_positions())
        qs = solve_ik(np.array([gx, gy, gz]), orient.tolist(), q_seed=q)
        if qs is not None:
            robot.move_to_joint_config(qs.tolist(), velocity=0.03)
        time.sleep(0.2)

    if not grasped:
        print(f"  FAILED")
        robot.open_gripper()
        time.sleep(0.3)
        return False

    if rec:
        rec.mark_keyframe(f"grasp_{label.replace(' ', '_')}")

    # Lift
    lift = gz + LIFT_Z
    q = np.array(robot.get_joint_positions())
    qs = solve_ik(np.array([gx, gy, lift]), orient.tolist(), q_seed=q)
    if qs is not None:
        robot.move_to_joint_config(qs.tolist(), velocity=0.10)
    time.sleep(0.2)

    # Arc fold: pre-compute all waypoints with J7 continuity check
    q = np.array(robot.get_joint_positions())
    arc_joints = []
    for i in range(1, 13):
        t = i / 12
        ax = gx if keep_x else gx + t * (fx - gx)
        ay = gy + t * (fy - gy)
        az = lift + ARC_PEAK * np.sin(t * np.pi) if i < 12 else LAND_Z
        qs = solve_ik(np.array([ax, ay, az]), orient.tolist(), q_seed=q)
        if qs is not None:
            # J7 continuity: reject if wrist flips > 1.0 rad from previous
            if abs(qs[6] - q[6]) > 1.0:
                qs_clamped = qs.copy()
                qs_clamped[6] = q[6]  # keep previous J7
                qs2 = solve_ik(np.array([ax, ay, az]), orient.tolist(), q_seed=qs_clamped)
                if qs2 is not None and abs(qs2[6] - q[6]) < 1.0:
                    qs = qs2
                else:
                    qs[6] = q[6] + np.clip(qs[6] - q[6], -0.5, 0.5)
            arc_joints.append(qs)
            q = qs
        else:
            arc_joints.append(None)

    for qs in arc_joints:
        if qs is not None:
            robot.move_to_joint_config(qs.tolist(), velocity=0.11)
        time.sleep(0.02)

    if rec:
        rec.mark_keyframe(f"fold_{label.replace(' ', '_')}")
    tcp = robot.get_tcp_pose()
    print(f"  End: ({tcp[0]:.4f},{tcp[1]:.4f},{tcp[2]:.4f})")
    robot.open_gripper()
    time.sleep(0.4)

    # Retract
    q = np.array(robot.get_joint_positions())
    qs = solve_ik(np.array([tcp[0], tcp[1], 0.08]), orient.tolist(), q_seed=q)
    if qs is not None:
        robot.move_to_joint_config(qs.tolist(), velocity=0.20)
    time.sleep(0.2)
    return True


def main():
    parser = argparse.ArgumentParser(description="Single-arm t-shirt fold")
    parser.add_argument("--phase", choices=["sleeve", "hem", "all"], default="all")
    parser.add_argument("--no-record", action="store_true")
    parser.add_argument("--coords", type=str, default=None,
                        help="Path to saved coords JSON (skip detection)")
    args = parser.parse_args()

    if args.coords:
        d = json.load(open(args.coords))
        print(f"LOADED COORDS from {args.coords}")
        caps = None
    else:
        # Capture (kinects before torch)
        print("CAPTURE")
        kinects = open_kinects()
        caps = capture(kinects)
        for k in kinects.values():
            try: k.stop()
            except: pass
            try: k.close()
            except: pass
        cals = load_calibrations()

        # Detect
        print("\nDETECT")
        d = detect_shirt(caps, cals)
        if d is None:
            print("Detection failed")
            return

        # Save coords for reuse
        json.dump(d, open("/tmp/fold_coords.json", "w"), indent=2)

    print(f"  +Y: ({d['p_gx']:.4f}, {d['p_gy']:.4f}) -> fold y={d['p_fold_y']:.4f}")
    print(f"  -Y: ({d['n_gx']:.4f}, {d['n_gy']:.4f}) -> fold y={d['n_fold_y']:.4f}")
    print(f"  Hem: ({d['hem_gx']:.4f},{d['hem_gy']:.4f}) -> collar ({d['collar_fx']:.4f},{d['collar_fy']:.4f})")

    # Dry run
    print("\nDRY RUN")
    do_sleeves = args.phase in ("sleeve", "all")
    do_hem = args.phase in ("hem", "all")

    if do_sleeves:
        if not dry_run("+Y", d["p_gx"], d["p_gy"], d["p_gx"], d["p_fold_y"],
                       ORIENT_90.tolist(), seed=HOME_Q):
            print("ABORT: +Y sleeve IK failed")
            return
        if not dry_run("-Y", d["n_gx"], d["n_gy"], d["n_gx"], d["n_fold_y"],
                       ORIENT_90.tolist(), seed=NY_SEED):
            print("ABORT: -Y sleeve IK failed")
            return
    if do_hem:
        if not dry_run("hem", d["hem_gx"], d["hem_gy"], d["collar_fx"], d["collar_fy"],
                       ORIENT_0.tolist(), keep_x=False, seed=HEM_SEED):
            print("ABORT: hem IK failed")
            return

    # Recording
    rec = None
    if not args.no_record:
        rec = RolloutRecorder("shirt_fold")
        # Save scene images (if we have them from detection)
        if caps is not None:
            for cam in caps:
                cv2.imwrite(os.path.join(rec.out_dir, f"scene_{cam}.png"),
                            cv2.cvtColor(caps[cam]["rgb"], cv2.COLOR_RGB2BGR))
        rec.start_recording()

    # Execute
    print("\nEXECUTE")
    robot = FrankaBambooDriver(robot_ip="172.16.0.2")
    robot.connect()
    robot.go_home(velocity=0.3)
    robot.open_gripper()
    time.sleep(0.3)
    if rec:
        rec.mark_keyframe("start")

    try:
        if do_sleeves:
            grasp_fold(robot, rec, "+Y sleeve",
                       d["p_gx"], d["p_gy"], d["p_gx"], d["p_fold_y"], ORIENT_90)
            robot.go_home(velocity=0.3)
            time.sleep(0.3)

            # Move to NY_SEED config for stable -Y sleeve IK
            robot.move_to_joint_config(NY_SEED.tolist(), velocity=0.3)
            time.sleep(0.3)

            ok = grasp_fold(robot, rec, "-Y sleeve",
                            d["n_gx"], d["n_gy"], d["n_gx"], d["n_fold_y"], ORIENT_90)
            if not ok:
                robot.go_home(velocity=0.3)
                time.sleep(0.3)
                robot.move_to_joint_config(NY_SEED.tolist(), velocity=0.3)
                time.sleep(0.3)
                grasp_fold(robot, rec, "-Y retry",
                           d["n_gx"], d["n_gy"] + 0.015, d["n_gx"], d["n_fold_y"], ORIENT_90)
            robot.go_home(velocity=0.3)
            time.sleep(0.3)

        if do_hem:
            grasp_fold(robot, rec, "hem",
                       d["hem_gx"], d["hem_gy"], d["collar_fx"], d["collar_fy"],
                       ORIENT_0, keep_x=False)

        robot.go_home(velocity=0.3)
        if rec:
            rec.mark_keyframe("end")
        print("\nSUCCESS")

    except Exception as e:
        print(f"\nFAILED: {e}")
        traceback.print_exc()
        try:
            robot.open_gripper()
        except:
            pass
        try:
            robot.go_home(velocity=0.3)
        except:
            pass
        if rec:
            rec.mark_keyframe("fail")

    robot.disconnect()

    if rec:
        print("\nSAVE")
        rec.stop_recording()


if __name__ == "__main__":
    main()

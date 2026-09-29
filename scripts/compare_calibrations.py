#!/usr/bin/env python3
"""Re-project the saved object orientations through each candidate
calibration and compare to the operator's demonstrated TCP-X yaws.

This isolates the calibration-rotation question:
- For each detection, the IMAGE-frame major-axis direction (from PCA on
  the SAM3 mask) is the same regardless of calibration.
- Only the conversion image-direction -> world-direction depends on
  cam_to_base rotation.
- So feeding the same image-frame direction through different cam_mats
  gives the world-frame orientation each calibration WOULD have reported.

Each calibration's predicted world orientation is compared to the
operator's demoed TCP-X yaw (the GROUND TRUTH of the object's long-axis
direction in world).
"""
import json
import numpy as np
from pathlib import Path
from scipy.spatial.transform import Rotation

# Operator's demonstrated poses (TCP-X yaw = handle major-axis direction
# in world, mod 180 since gripper jaws are symmetric).
DEMOS = [
    {"obj": 1, "tcp_x_yaw_deg": -44.5,  "tcp_pos": [0.647, 0.307, 0.031]},
    {"obj": 2, "tcp_x_yaw_deg": -86.3,  "tcp_pos": [0.454, 0.346, 0.032]},
    {"obj": 3, "tcp_x_yaw_deg":  11.8,  "tcp_pos": [0.244, 0.298, 0.044]},
    {"obj": 4, "tcp_x_yaw_deg": -82.2,  "tcp_pos": [0.306, 0.479, 0.038]},
]

# Detection data saved at the labeled-capture step. Provides:
# - centroid_2d in birdview image pixels
# - orientation_angle in WORLD frame, under the CURRENT calibration
_OUT = Path.home() / "spark" / "src" / "spark_real" / "output"
LABELED = _OUT / "diagnostics" / "labeled_20260521_043418.json"

# All birdview calibration files to test.
CALS = [
    ("CURRENT (rgbd_procrustes, 161 pairs, 1.55mm)",
     str(_OUT / "calibrations" / "handeye_birdview.json")),
    ("OLDER eye_to_hand (charuco, 34 pairs)",
     str(_OUT / "calibrations_backup_20260519_031308" / "handeye_birdview.json")),
    ("BAD anchor SVD (13 anchors, 12.34mm)",
     str(_OUT / "calibrations" / "handeye_birdview.bad_13pair.bak.json")),
]

# Kinect 4K factory intrinsics (from server log: fx=909.3, 1920x1080)
H, W = 1080, 1920
F = 909.3  # focal length used by perception to project img-dir -> cam-dir


# Return 3x3 rotation portion of T_cam_to_base.
def load_cam_mat(path):
    d = json.loads(Path(path).read_text())
    # Some calibrations store T as nested list, others as flat
    if "T_cam_to_base_4x4" in d:
        T = np.array(d["T_cam_to_base_4x4"])
    elif "transform_4x4" in d:
        T = np.array(d["transform_4x4"])
    elif "T_cam_to_base" in d:
        T = np.array(d["T_cam_to_base"])
    elif "extrinsic" in d:
        T = np.array(d["extrinsic"])
    elif "transform" in d:
        T = np.array(d["transform"])
    else:
        raise KeyError(f"no transform field in {path}: keys={list(d.keys())[:10]}")
    if T.shape != (4, 4):
        T = np.array(T).reshape(4, 4)
    return T[:3, :3].astype(float)


def world_orient_from_image_angle(img_angle_rad, cam_mat):
    """
    Apply spark_perception.py's conversion from image-pixel major-axis
    angle to world-frame major-axis angle.
    """
    du = np.cos(img_angle_rad)
    dv = np.sin(img_angle_rad)
    dx_cam = du / F
    dy_cam = -dv / F      # image-Y is down, camera-Y is up
    dir_cam = np.array([dx_cam, dy_cam, 0.0])
    dir_world = cam_mat @ dir_cam
    return float(np.arctan2(dir_world[1], dir_world[0]))


def recover_image_angle(world_angle_rad, cam_mat_used):
    """
    Invert spark_perception.py's image->world transform: given the
    world angle that WAS reported under cam_mat_used, recover what the
    image-pixel major-axis angle must have been.
    """
    # The forward direction:
    #   dir_world = cam_mat_used @ (cos(img)/F, -sin(img)/F, 0)
    # Since cam_mat is a rotation,
    #   cam_mat^T @ dir_world = (cos(img)/F, -sin(img)/F, 0)
    dir_world = np.array([np.cos(world_angle_rad),
                          np.sin(world_angle_rad), 0.0])
    dir_cam = cam_mat_used.T @ dir_world
    du = dir_cam[0] * F
    dv = -dir_cam[1] * F
    return float(np.arctan2(dv, du))


# Wrap angle in degrees to [-90, +90] (gripper-symmetric).
def mod180_deg(a):
    a = a % 180.0
    if a > 90:
        a -= 180
    return a


# Smallest signed difference between two gripper-symmetric angles.
def angle_diff_mod180_deg(a, b):
    d = (a - b) % 180.0
    if d > 90:
        d -= 180
    return d


def main():
    if not LABELED.exists():
        raise SystemExit(f"missing {LABELED}")
    data = json.loads(LABELED.read_text())
    objects = data["objects"][:4]  # skip tray

    # Cam_mat under which the saved orientation_angle was computed:
    cam_mat_current = load_cam_mat(CALS[0][1])

    # For each object, recover the IMAGE-frame major-axis angle so we
    # can re-project through other calibrations.
    rows = []
    for obj, demo in zip(objects, DEMOS):
        world_orient_rad = np.deg2rad(obj["orientation_angle_deg"])
        img_angle_rad = recover_image_angle(world_orient_rad, cam_mat_current)
        rows.append({
            "n": obj["n"],
            "label": obj["label"],
            "pos": obj["position_3d"],
            "img_angle_deg": float(np.rad2deg(img_angle_rad)),
            "demo_tcp_x_yaw_deg": demo["tcp_x_yaw_deg"],
            "img_angle_rad": img_angle_rad,
            "saved_world_orient_deg": obj["orientation_angle_deg"],
        })

    print(f"{'Cal':<55} {'Mean |err|':>12} {'Max |err|':>12} {'Edge (#1,#3)':>14}")
    results = []
    for cal_name, cal_path in CALS:
        try:
            cam_mat = load_cam_mat(cal_path)
        except Exception as exc:
            print(f"  SKIP {cal_name}: {exc}")
            continue
        per_obj_err = []
        for row in rows:
            world_pred_rad = world_orient_from_image_angle(
                row["img_angle_rad"], cam_mat)
            world_pred_deg = float(np.rad2deg(world_pred_rad))
            err = angle_diff_mod180_deg(
                world_pred_deg, row["demo_tcp_x_yaw_deg"])
            per_obj_err.append((row["n"], world_pred_deg, err))
        errs = np.array([abs(e[2]) for e in per_obj_err])
        edge_err = np.mean([abs(per_obj_err[0][2]),
                             abs(per_obj_err[2][2])])  # #1, #3
        print(f"{cal_name:<55} {errs.mean():>11.1f}deg {errs.max():>11.1f}deg {edge_err:>13.1f}deg")
        results.append((cal_name, per_obj_err, errs))

    print()
    print("PER-OBJECT detail (world_orient_pred_deg | err_vs_demo):")
    print(f"{'obj':<5} {'demo':>9} ", end="")
    for n, _, _ in results:
        print(f"{n[:30]:<32}", end="")
    print()
    for i in range(len(rows)):
        print(f"#{rows[i]['n']:<4} {rows[i]['demo_tcp_x_yaw_deg']:>+8.1f}deg ", end="")
        for _, per_obj_err, _ in results:
            pred, err = per_obj_err[i][1], per_obj_err[i][2]
            print(f"  {pred:+7.1f}deg (err={err:+6.1f}deg)        ", end="")
        print()


if __name__ == "__main__":
    main()

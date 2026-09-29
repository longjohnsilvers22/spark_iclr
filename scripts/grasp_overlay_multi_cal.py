#!/usr/bin/env python3
"""Take the most recent labeled-capture data and draw the planned
grasp footprint per object UNDER EACH CANDIDATE CALIBRATION.

Outputs 3 PNGs in output/diagnostics/multical_<ts>/, one per cal:
  - grasp_CURRENT.png        (161-pair RGBD procrustes, 1.55 mm)
  - grasp_OLDER.png          (34-pair eye_to_hand, charuco)
  - grasp_BAD13.png          (13-pair anchor SVD, 12.34 mm)

Each image shows:
  - Green dot at detected centroid (image-pixel coords, same across cals)
  - Yellow line = jaw CLOSING direction (perpendicular to handle major axis
    under THAT calibration's cam_to_base rotation)
  - Green outline = jaw FOOTPRINT rectangle
  - Numbered label and the yaw angle in degrees

Uses the saved birdview image (no fresh capture needed). The yellow
closing direction is what changes between calibrations.
"""
import base64
import io
import json
import sys
import time
import traceback
from pathlib import Path

import numpy as np
import requests
from PIL import Image, ImageDraw, ImageFont

_OUT = Path.home() / "spark" / "src" / "spark_real" / "output"
OUT_ROOT = _OUT / "diagnostics"

LABELED_JSON = OUT_ROOT / "labeled_20260521_043418.json"
LABELED_PNG = OUT_ROOT / "labeled_20260521_043418.png"

CALS = [
    ("CURRENT", "rgbd_procrustes 161p 1.55mm",
     str(_OUT / "calibrations" / "handeye_birdview.json")),
    ("OLDER", "eye_to_hand 34p charuco",
     str(_OUT / "calibrations_backup_20260519_031308" / "handeye_birdview.json")),
    ("BAD13", "anchor SVD 13p 12.34mm",
     str(_OUT / "calibrations" / "handeye_birdview.bad_13pair.bak.json")),
]

H, W = 1080, 1920
F = 909.3

JAW_LENGTH_M = 0.04
DEFAULT_TARGET_WIDTH_M = 0.012

DEMOS_TCP_X_YAW_DEG = {  # ground-truth handle major angles
    "knife handle 1": -44.5,
    "knife handle 2": -86.3,
    "knife handle 3":  11.8,
    "knife handle 4": -82.2,
    "spoon handle":   -89.7,  # if relabeled
}


def load_cal(path):
    d = json.loads(Path(path).read_text())
    for k in ("T_cam_to_base_4x4", "transform_4x4", "T_cam_to_base",
              "extrinsic", "transform"):
        if k in d:
            T = np.array(d[k])
            if T.shape != (4, 4):
                T = T.reshape(4, 4)
            return T.astype(float)
    raise KeyError(f"no transform in {path}")


def world_to_pixel(p_world, T_cam_to_base, fx=F, fy=F, cx=W/2, cy=H/2):
    T_b2c = np.linalg.inv(T_cam_to_base)
    p_w = np.array([*p_world[:3], 1.0])
    p_c = T_b2c @ p_w
    if p_c[2] <= 0.01:
        return None
    return (fx * p_c[0] / p_c[2] + cx, fy * p_c[1] / p_c[2] + cy)


def world_orient_from_image_angle(img_angle_rad, cam_mat):
    du = np.cos(img_angle_rad)
    dv = np.sin(img_angle_rad)
    dir_cam = np.array([du / F, -dv / F, 0.0])
    dir_world = cam_mat @ dir_cam
    return float(np.arctan2(dir_world[1], dir_world[0]))


def recover_image_angle(world_angle_rad, cam_mat):
    dir_world = np.array([np.cos(world_angle_rad),
                          np.sin(world_angle_rad), 0.0])
    dir_cam = cam_mat.T @ dir_world
    du = dir_cam[0] * F
    dv = -dir_cam[1] * F
    return float(np.arctan2(dv, du))


def compute_yaw(orient_rad):
    yaw = orient_rad % np.pi
    if yaw > np.pi / 2:
        yaw -= np.pi
    return yaw


def load_font(size):
    for p in ("/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf",
              "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf"):
        if Path(p).exists():
            try:
                return ImageFont.truetype(p, size)
            except Exception:
                pass
    return ImageFont.load_default()


def draw_grasp(draw, det, yaw_rad, target_width_m, color, label_idx, demo_yaw_deg=None):
    grasp_xyz = [det["position_3d"][0], det["position_3d"][1],
                  det["position_3d"][2]]
    aspect_ratio = float(det.get("aspect_ratio", 1.0) or 1.0)
    if aspect_ratio >= 1.5:
        grasp_xyz[2] -= 0.005  # match grasping.py top-down

    cam_mat = current_cam_mat
    px = world_to_pixel(grasp_xyz, _current_T)
    if px is None:
        return None

    half_w = target_width_m / 2.0 + 0.005
    mx_x, mx_y = np.cos(yaw_rad), np.sin(yaw_rad)
    cy_x, cy_y = np.sin(yaw_rad), -np.cos(yaw_rad)

    corners_world = []
    for sc in (-1, 1):
        for sm in (-1, 1):
            corners_world.append([
                grasp_xyz[0] + sc * half_w * cy_x + sm * (JAW_LENGTH_M / 2) * mx_x,
                grasp_xyz[1] + sc * half_w * cy_y + sm * (JAW_LENGTH_M / 2) * mx_y,
                grasp_xyz[2],
            ])
    ordered = [corners_world[0], corners_world[1],
                corners_world[3], corners_world[2], corners_world[0]]
    proj = [world_to_pixel(p, _current_T) for p in ordered]
    if any(p is None for p in proj):
        return None
    draw.line(proj, fill=color, width=4)

    # closing direction line (yellow)
    cl = [
        world_to_pixel([grasp_xyz[0] - half_w * cy_x,
                        grasp_xyz[1] - half_w * cy_y,
                        grasp_xyz[2]], _current_T),
        world_to_pixel([grasp_xyz[0] + half_w * cy_x,
                        grasp_xyz[1] + half_w * cy_y,
                        grasp_xyz[2]], _current_T),
    ]
    if all(p is not None for p in cl):
        draw.line(cl, fill=(255, 220, 0), width=6)

    # centroid dot
    draw.ellipse([px[0]-9, px[1]-9, px[0]+9, px[1]+9],
                  fill=color, outline=(0, 0, 0))

    # text
    yaw_deg = float(np.rad2deg(yaw_rad))
    txt_lines = [f"#{label_idx} {det['label']}",
                 f"yaw={yaw_deg:+.0f}deg"]
    if demo_yaw_deg is not None:
        err = ((yaw_deg - demo_yaw_deg + 90) % 180) - 90
        txt_lines.append(f"demo={demo_yaw_deg:+.0f}deg err={err:+.0f}deg")
    font = load_font(22)
    for li, line in enumerate(txt_lines):
        draw.text((px[0]+15, px[1]+15+li*26), line, font=font,
                   fill=color, stroke_width=2, stroke_fill=(0,0,0))
    return px


def main():
    if not LABELED_JSON.exists():
        raise SystemExit(f"missing {LABELED_JSON}")
    if not LABELED_PNG.exists():
        raise SystemExit(f"missing {LABELED_PNG}")

    data = json.loads(LABELED_JSON.read_text())
    # labeled_capture saved objects with key 'objects' not 'detections';
    # also it remapped keys (orientation_angle_deg, n), normalize here.
    raw = data.get("objects") or data.get("detections", [])
    bv_dets = []
    for d in raw:
        lbl = d.get("label", "")
        if lbl == "tray":
            continue
        # Restore radians if only degrees stored.
        orient_rad = d.get("orientation_angle")
        if orient_rad is None and "orientation_angle_deg" in d:
            orient_rad = float(np.deg2rad(d["orientation_angle_deg"]))
        norm = {
            "label": lbl,
            "n": d.get("n"),
            "position_3d": d["position_3d"],
            "orientation_angle": float(orient_rad or 0.0),
            "aspect_ratio": d.get("aspect_ratio", 1.0),
            "obb_minor_m": d.get("obb_minor_m", 0.0),
            "centroid_2d": d.get("centroid_2d"),
        }
        bv_dets.append(norm)

    # The saved orientation_angle was computed under CURRENT cal.
    global _current_T, current_cam_mat
    _current_T = load_cal(CALS[0][2])
    current_cam_mat = _current_T[:3, :3]

    # Recover image-frame angle (cal-invariant) for each detection.
    for det in bv_dets:
        orient_rad = float(det.get("orientation_angle", 0.0))
        det["_img_angle_rad"] = recover_image_angle(orient_rad, current_cam_mat)

    ts = time.strftime("%Y%m%d_%H%M%S")
    out_dir = OUT_ROOT / f"multical_{ts}"
    out_dir.mkdir(parents=True, exist_ok=True)

    for short_name, desc, cal_path in CALS:
        try:
            T = load_cal(cal_path)
        except Exception as exc:
            print(f"SKIP {short_name}: {exc}")
            continue
        cam_mat = T[:3, :3]

        # Open image fresh per cal
        img = Image.open(LABELED_PNG).convert("RGB")
        draw = ImageDraw.Draw(img)

        # header
        font_h = load_font(28)
        font_med = load_font(20)
        draw.text((20, 20), f"CAL: {short_name}", font=font_h,
                   fill=(255,255,255), stroke_width=3, stroke_fill=(0,0,0))
        draw.text((20, 55), desc, font=font_med,
                   fill=(255,255,255), stroke_width=2, stroke_fill=(0,0,0))
        draw.text((20, 85), "yellow = jaw closing dir under THIS cal",
                   font=font_med, fill=(255,220,0),
                   stroke_width=2, stroke_fill=(0,0,0))

        colors = [(255,80,80), (80,200,255), (255,200,0), (140,80,255)]
        per_obj = []
        for i, det in enumerate(bv_dets):
            world_orient_this_cal = world_orient_from_image_angle(
                det["_img_angle_rad"], cam_mat)
            yaw = compute_yaw(world_orient_this_cal)
            tw = (det.get("obb_minor_m", 0.0) or 0.0) * 0.6
            if not tw or tw <= 0:
                tw = DEFAULT_TARGET_WIDTH_M
            demo_yaw = DEMOS_TCP_X_YAW_DEG.get(det["label"])
            draw_grasp(draw, det, yaw, tw, colors[i % len(colors)],
                        det.get("n", i+1), demo_yaw_deg=demo_yaw)
            per_obj.append({
                "n": det.get("n", i+1),
                "label": det["label"],
                "yaw_under_cal_deg": float(np.rad2deg(yaw)),
                "demo_yaw_deg": demo_yaw,
            })

        out_png = out_dir / f"grasp_{short_name}.png"
        img.save(out_png)
        print(f"  wrote {out_png}")

    print(f"\nAll wrote to {out_dir}")


if __name__ == "__main__":
    try:
        main()
    except Exception as e:
        print(f"ERROR: {e}", file=sys.stderr)
        traceback.print_exc()
        sys.exit(1)

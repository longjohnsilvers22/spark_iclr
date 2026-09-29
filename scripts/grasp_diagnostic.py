#!/usr/bin/env python3
"""Visual grasp-plan diagnostic.

Captures the birdview image, runs SAM3 detection for silverware + tray,
then for each detected object draws the *planned* top-down grasp pose
(closing axis direction, jaw rectangle, descent target) projected onto
the image. Lets the operator visually verify what the robot WOULD do
before actually running the task.

Output: output/diagnostics/grasp_plan_<timestamp>.png + .json
"""
import base64
import json
import sys
import time
import traceback
from pathlib import Path

import numpy as np
import requests
from PIL import Image, ImageDraw, ImageFont

API = "http://localhost:8888"
PROMPTS = ["knife handle", "spoon handle", "fork handle", "tray"]
_SPARK_REAL = Path.home() / "spark" / "src" / "spark_real"
OUT_DIR = _SPARK_REAL / "output" / "diagnostics"
CAL_PATH = _SPARK_REAL / "output" / "calibrations" / "handeye_birdview.json"

# Franka Hand jaw geometry (fingertip-to-fingertip max width = 80 mm; the
# closing rectangle is drawn at the planned target_width).
DEFAULT_TARGET_WIDTH_M = 0.012  # silverware default
JAW_LENGTH_M = 0.04             # how far down each fingertip protrudes


# Pull live intrinsics from a fresh capture (the route includes them).
def load_birdview_intrinsics():
    r = requests.get(f"{API}/api/capture", timeout=15)
    r.raise_for_status()
    data = r.json()
    if "birdview" not in data:
        raise RuntimeError("no birdview in capture")
    bv = data["birdview"]
    # /api/capture serializes width+height but not fx/cx; use the known
    # Kinect 1920x1080 intrinsics (fx=909.3).
    return {
        "width": bv["width"],
        "height": bv["height"],
        "fx": 909.3,
        "fy": 909.3,
        "cx": bv["width"] / 2.0,
        "cy": bv["height"] / 2.0,
        "rgb_b64": bv["rgb"],
    }


# Project a 3D world point to image pixels via birdview calibration.
def world_to_pixel(p_world, T_cam_to_base, fx, fy, cx, cy):
    T_base_to_cam = np.linalg.inv(T_cam_to_base)
    p_w = np.array([p_world[0], p_world[1], p_world[2], 1.0])
    p_c = T_base_to_cam @ p_w
    if p_c[2] <= 0.01:
        return None
    u = fx * (p_c[0] / p_c[2]) + cx
    v = fy * (p_c[1] / p_c[2]) + cy
    return (float(u), float(v))


# Same logic as _grasp_top_down: wrap OBB-major-angle into [-pi/2, pi/2].
def compute_grasp_yaw(orientation_angle_rad: float) -> float:
    yaw = orientation_angle_rad % np.pi
    if yaw > np.pi / 2:
        yaw -= np.pi
    return yaw


def main():
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    ts = time.strftime("%Y%m%d_%H%M%S")

    print(f"[diag] hitting {API}/api/detect ...")
    r = requests.post(
        f"{API}/api/detect",
        json={"prompts": PROMPTS, "multi_instance": True},
        timeout=60,
    )
    r.raise_for_status()
    detect_out = r.json()
    detections = detect_out["detections"]
    print(f"[diag] got {len(detections)} detection(s)")

    # Pull birdview image (the annotated_image from /api/detect is the
    # multi-camera tile; the raw birdview is needed to draw on).
    cap = load_birdview_intrinsics()
    rgb_bytes = base64.b64decode(cap["rgb_b64"])
    img = Image.open(__import__("io").BytesIO(rgb_bytes)).convert("RGB")
    draw = ImageDraw.Draw(img)

    cal = json.loads(CAL_PATH.read_text())
    T_cam_to_base = np.array(cal["T_cam_to_base_4x4"])

    fx, fy, cx, cy = cap["fx"], cap["fy"], cap["cx"], cap["cy"]

    # Filter to birdview detections only
    bv_dets = [d for d in detections if d.get("camera") == "birdview"]
    if not bv_dets:
        # /api/detect serializes only merged dets; assume all are visible in birdview
        bv_dets = detections

    print(f"[diag] {len(bv_dets)} detection(s) used for overlay")

    planned = []
    for det in bv_dets:
        label = det.get("label", "?")
        pos = det.get("position_3d")
        if pos is None or len(pos) < 3:
            continue
        ar = float(det.get("aspect_ratio", 1.0) or 1.0)
        orient = float(det.get("orientation_angle", 0.0) or 0.0)
        obb_minor_m = float(det.get("obb_minor_m", 0.0) or 0.0)

        is_silverware = label != "tray"
        # Match _grasp_top_down: 5mm Z bias for elongated objects.
        z_bias = 0.005 if (is_silverware and ar >= 1.5) else 0.0
        grasp_xyz = [pos[0], pos[1], pos[2] - z_bias]
        yaw = compute_grasp_yaw(orient) if is_silverware else 0.0
        target_width = obb_minor_m * 0.6 if obb_minor_m > 0 else DEFAULT_TARGET_WIDTH_M

        # Project the grasp center
        px = world_to_pixel(grasp_xyz, T_cam_to_base, fx, fy, cx, cy)
        if px is None:
            continue

        planned.append({
            "label": label,
            "pos": grasp_xyz,
            "yaw_deg": float(np.rad2deg(yaw)),
            "aspect_ratio": ar,
            "obb_minor_mm": obb_minor_m * 1000,
            "target_width_mm": target_width * 1000,
            "image_px": [px[0], px[1]],
        })

        # Compute the jaw endpoints in world (closing axis is perpendicular
        # to the OBB major axis ~ yaw + pi/2 in world XY)
        half_w = target_width / 2.0 + 0.005  # +5mm to show the jaws beyond fabric
        # TCP-Y (closing axis) in world frame after Rz(yaw) on base (TCP-Y -> world -Y):
        # closing_world = Rz(yaw) @ (-Y) = (sin(yaw), -cos(yaw), 0)
        cy_x = np.sin(yaw)
        cy_y = -np.cos(yaw)
        # Major axis (along jaws' length direction = TCP-X after yaw): (cos, sin, 0)
        mx_x = np.cos(yaw)
        mx_y = np.sin(yaw)
        # Jaw end points = grasp_center +/- half_w * closing + JAW_LENGTH/2 * major
        jaw_corners_world = []
        for sign_close in (-1, 1):
            for sign_major in (-1, 1):
                jaw_corners_world.append([
                    grasp_xyz[0] + sign_close * half_w * cy_x
                                  + sign_major * (JAW_LENGTH_M / 2) * mx_x,
                    grasp_xyz[1] + sign_close * half_w * cy_y
                                  + sign_major * (JAW_LENGTH_M / 2) * mx_y,
                    grasp_xyz[2],
                ])
        # Order corners as a rectangle path
        ordered = [jaw_corners_world[0], jaw_corners_world[1],
                   jaw_corners_world[3], jaw_corners_world[2],
                   jaw_corners_world[0]]
        proj = [world_to_pixel(p, T_cam_to_base, fx, fy, cx, cy)
                for p in ordered]
        if any(p is None for p in proj):
            continue

        # Color: green for silverware grasp, red for tray (place target)
        color = (0, 255, 80) if is_silverware else (255, 80, 80)

        # Draw the jaw rectangle outline
        draw.line(proj + [proj[0]], fill=color, width=4)

        # Draw a thick line along the closing axis (perpendicular to major).
        # Make it 50mm long (much longer than the 9mm jaw-width) so the
        # angle vs the major-axis line is visually unambiguous in the image.
        CLOSE_LINE_HALF_M = 0.025
        close_pts_world = [
            [grasp_xyz[0] - CLOSE_LINE_HALF_M * cy_x,
             grasp_xyz[1] - CLOSE_LINE_HALF_M * cy_y,
             grasp_xyz[2]],
            [grasp_xyz[0] + CLOSE_LINE_HALF_M * cy_x,
             grasp_xyz[1] + CLOSE_LINE_HALF_M * cy_y,
             grasp_xyz[2]],
        ]
        close_px = [world_to_pixel(p, T_cam_to_base, fx, fy, cx, cy)
                    for p in close_pts_world]
        if all(p is not None for p in close_px):
            draw.line(close_px, fill=(255, 220, 0), width=8)  # yellow CLOSING

        # Also draw the MAJOR axis (along the handle) for visual reference.
        # Cyan so it doesn't compete with the green rectangle/yellow line.
        MAJ_LINE_HALF_M = 0.025
        maj_pts_world = [
            [grasp_xyz[0] - MAJ_LINE_HALF_M * mx_x,
             grasp_xyz[1] - MAJ_LINE_HALF_M * mx_y,
             grasp_xyz[2]],
            [grasp_xyz[0] + MAJ_LINE_HALF_M * mx_x,
             grasp_xyz[1] + MAJ_LINE_HALF_M * mx_y,
             grasp_xyz[2]],
        ]
        maj_px = [world_to_pixel(p, T_cam_to_base, fx, fy, cx, cy)
                  for p in maj_pts_world]
        if all(p is not None for p in maj_px):
            draw.line(maj_px, fill=(0, 255, 255), width=4)  # cyan MAJOR-axis

        # Center dot
        draw.ellipse([px[0]-8, px[1]-8, px[0]+8, px[1]+8],
                      fill=color, outline=(0, 0, 0))

        # Label
        text = (f"{label}\n"
                f"  yaw={np.rad2deg(yaw):.0f}deg\n"
                f"  jaw={target_width*1000:.0f}mm\n"
                f"  ar={ar:.1f}")
        try:
            draw.multiline_text((px[0]+15, px[1]+15), text,
                                fill=color, stroke_width=2,
                                stroke_fill=(0, 0, 0))
        except Exception:
            draw.text((px[0]+15, px[1]+15), text, fill=color)

    # Legend
    legend = ("LEGEND\n"
              "  yellow line = jaw closing direction\n"
              "  green outline = silverware grasp footprint\n"
              "  red outline = tray (place target)")
    try:
        draw.multiline_text((20, 20), legend, fill=(255, 255, 255),
                            stroke_width=2, stroke_fill=(0, 0, 0))
    except Exception:
        pass

    out_png = OUT_DIR / f"grasp_plan_{ts}.png"
    out_json = OUT_DIR / f"grasp_plan_{ts}.json"
    img.save(out_png)
    out_json.write_text(json.dumps({
        "timestamp": ts,
        "detections": detections,
        "planned_grasps": planned,
    }, indent=2, default=str))
    print(f"[diag] wrote {out_png}")
    print(f"[diag] wrote {out_json}")
    print()
    print("PLANNED GRASPS:")
    for p in planned:
        print(f"  {p['label']:25s}  pos=({p['pos'][0]:+.3f},{p['pos'][1]:+.3f},{p['pos'][2]:+.3f})  "
              f"yaw={p['yaw_deg']:+.0f}deg  jaw={p['target_width_mm']:.0f}mm  ar={p['aspect_ratio']:.1f}")
    return str(out_png)


if __name__ == "__main__":
    try:
        main()
    except Exception as e:
        print(f"[diag] ERROR: {e}", file=sys.stderr)
        traceback.print_exc()
        sys.exit(1)

#!/usr/bin/env python3
"""Take a fresh birdview capture, run detection, and produce a clean
labeled image so the operator can manually move the gripper to each
object and report what 'perpendicular' looks like for ground-truth
comparison vs perception's reported orientation.

Outputs:
  output/diagnostics/labeled_<ts>.png  -- the image with labels
  output/diagnostics/labeled_<ts>.json -- raw detection data per label
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

API = "http://localhost:8888"
PROMPTS = ["knife handle", "spoon handle", "fork handle", "tray"]
OUT_DIR = Path.home() / "spark" / "src" / "spark_real" / "output" / "diagnostics"

# Colors per detection index for clarity
COLORS = [
    (255, 80, 80),    # red
    (80, 200, 255),   # cyan
    (255, 200, 0),    # yellow
    (140, 80, 255),   # purple
    (80, 255, 120),   # green (last = tray)
]


def load_font(size: int):
    candidates = [
        "/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf",
        "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf",
    ]
    for c in candidates:
        if Path(c).exists():
            try:
                return ImageFont.truetype(c, size)
            except Exception:
                pass
    return ImageFont.load_default()


def main():
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    ts = time.strftime("%Y%m%d_%H%M%S")

    print(f"[label] /api/detect ...")
    r = requests.post(
        f"{API}/api/detect",
        json={"prompts": PROMPTS, "multi_instance": True},
        timeout=60,
    )
    r.raise_for_status()
    detect_out = r.json()
    detections = detect_out["detections"]

    # Pull the raw birdview separately so we get the un-annotated frame
    cap = requests.get(f"{API}/api/capture", timeout=15).json()
    bv = cap["birdview"]
    rgb_bytes = base64.b64decode(bv["rgb"])
    img = Image.open(io.BytesIO(rgb_bytes)).convert("RGB")
    draw = ImageDraw.Draw(img)
    font_lg = load_font(36)
    font_md = load_font(24)
    font_sm = load_font(18)

    # Filter to birdview detections (or all if "camera" not set)
    bv_dets = [d for d in detections if d.get("camera") in (None, "birdview")]

    # Robot pose at capture time (for the operator to use as reference)
    try:
        joints_resp = requests.get(f"{API}/api/joints", timeout=3)
        if joints_resp.status_code == 200:
            joints_now = joints_resp.json().get("joints")
        else:
            joints_now = None
    except Exception:
        joints_now = None

    labels_table = []
    for i, det in enumerate(bv_dets):
        label = det.get("label", "?")
        cx, cy = det.get("centroid_2d", [None, None])
        if cx is None or cy is None:
            continue
        cx, cy = float(cx), float(cy)
        pos = det.get("position_3d", [0, 0, 0])
        orient = float(det.get("orientation_angle", 0.0) or 0.0)
        ar = float(det.get("aspect_ratio", 1.0) or 1.0)
        obb_minor_m = float(det.get("obb_minor_m", 0.0) or 0.0)

        color = COLORS[i % len(COLORS)]
        # Big numbered circle at centroid
        ring_r = 28
        draw.ellipse([cx-ring_r, cy-ring_r, cx+ring_r, cy+ring_r],
                      outline=color, width=5)
        draw.ellipse([cx-8, cy-8, cx+8, cy+8], fill=color, outline=(0, 0, 0))
        # Number label inside the circle
        try:
            tw = font_lg.getbbox(str(i+1))
            text_w = tw[2] - tw[0]
            text_h = tw[3] - tw[1]
        except Exception:
            text_w, text_h = 18, 24
        # background for visibility
        draw.text((cx - text_w/2 + 1, cy - text_h - 38),
                   f"#{i+1}", font=font_lg,
                   fill=(255, 255, 255),
                   stroke_width=3, stroke_fill=(0, 0, 0))

        # Annotation block to the right of the object
        bbox = det.get("bbox", [cx, cy, cx, cy])
        text_x = bbox[2] + 15
        text_y = bbox[1]
        # Stay on image
        if text_x > img.width - 320:
            text_x = bbox[0] - 320
        lines = [
            f"#{i+1}  {label}",
            f"world: ({pos[0]:+.3f}, {pos[1]:+.3f}, {pos[2]:+.3f})",
            f"perc.orient: {np.rad2deg(orient):+.1f} deg",
            f"aspect: {ar:.1f}",
            f"obb_minor: {obb_minor_m*1000:.0f}mm",
        ]
        for li, line in enumerate(lines):
            draw.text((text_x, text_y + li*22), line, font=font_sm,
                       fill=(255, 255, 255),
                       stroke_width=2, stroke_fill=(0, 0, 0))
        labels_table.append({
            "n": i+1, "label": label,
            "centroid_2d": [cx, cy],
            "position_3d": pos,
            "orientation_angle_deg": float(np.rad2deg(orient)),
            "aspect_ratio": ar,
            "obb_minor_m": obb_minor_m,
        })

    # Header banner
    header_lines = [
        "GROUND-TRUTH DEMO CAPTURE",
        f"timestamp: {ts}",
        "for each numbered object: move the gripper into the closing pose",
        "you'd expect (perpendicular to long axis), then GET /api/joints",
        "and tell me the 7-vec, I'll compare to perception.orient.",
    ]
    if joints_now is not None:
        header_lines.append(f"robot now: {[round(j, 3) for j in joints_now]}")
    for li, line in enumerate(header_lines):
        draw.text((20, 20 + li*28), line, font=font_md,
                   fill=(255, 255, 255),
                   stroke_width=2, stroke_fill=(0, 0, 0))

    out_png = OUT_DIR / f"labeled_{ts}.png"
    out_json = OUT_DIR / f"labeled_{ts}.json"
    img.save(out_png)
    out_json.write_text(json.dumps({
        "timestamp": ts,
        "robot_joints_at_capture": joints_now,
        "objects": labels_table,
    }, indent=2, default=str))
    print(f"[label] wrote {out_png}")
    print(f"[label] wrote {out_json}")
    print()
    print("PERCEPTION REPORT (per object):")
    for row in labels_table:
        print(f"  #{row['n']}  {row['label']:18s}  "
              f"pos=({row['position_3d'][0]:+.3f},{row['position_3d'][1]:+.3f},{row['position_3d'][2]:+.3f})  "
              f"orient={row['orientation_angle_deg']:+.1f} deg  ar={row['aspect_ratio']:.1f}")


if __name__ == "__main__":
    try:
        main()
    except Exception as e:
        print(f"[label] ERROR: {e}", file=sys.stderr)
        traceback.print_exc()
        sys.exit(1)

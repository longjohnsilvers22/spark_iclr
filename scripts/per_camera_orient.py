#!/usr/bin/env python3
"""Fresh detect, extract per-camera detections (birdview & sideview),
report orientation_angle for each per object, to check whether sideview
gives a different (and possibly more accurate) reading at the edges
than birdview."""
import base64
import io
import json
import time
from pathlib import Path

import numpy as np
import requests
from PIL import Image, ImageDraw, ImageFont

API = "http://localhost:8888"
PROMPTS = ["knife handle", "spoon handle", "fork handle", "tray"]
OUT_DIR = Path.home() / "spark" / "src" / "spark_real" / "output" / "diagnostics"


def load_font(s):
    for p in ("/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf",
              "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf"):
        if Path(p).exists():
            try: return ImageFont.truetype(p, s)
            except Exception: pass
    return ImageFont.load_default()


def main():
    ts = time.strftime("%Y%m%d_%H%M%S")
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    print("hitting /api/detect (timeout 90s)...")
    r = requests.post(f"{API}/api/detect",
                      json={"prompts": PROMPTS, "multi_instance": True},
                      timeout=90)
    r.raise_for_status()
    data = r.json()
    merged = data["detections"]
    all_dets = data.get("all_detections", [])

    print(f"merged={len(merged)}  all={len(all_dets)}")

    # Group all_detections by label
    by_label = {}
    for d in all_dets:
        lbl = d.get("label", "?")
        by_label.setdefault(lbl, []).append(d)

    # Report per-camera orient for each merged object
    print()
    print(f"{'#':<3} {'label':<22} {'cam':<10} {'world (m)':<24} "
          f"{'orientdeg':>9} {'ar':>6} {'minor(mm)':>10}")
    rows = []
    for i, m in enumerate(merged):
        lbl = m["label"]
        # All per-camera variants
        variants = by_label.get(lbl, [])
        # Filter to per-instance: pick variants whose position_3d is near
        # the merged one (within 10cm) so they refer to the same physical
        # object (not a different instance).
        m_pos = np.array(m.get("position_3d", [0, 0, 0]))
        for v in variants:
            vpos = np.array(v.get("position_3d", [0, 0, 0]))
            if np.linalg.norm(vpos - m_pos) > 0.10:
                continue
            cam = v.get("camera", "?")
            orient = float(v.get("orientation_angle", 0.0) or 0.0)
            ar = float(v.get("aspect_ratio", 1.0) or 1.0)
            minor = float(v.get("obb_minor_m", 0.0) or 0.0)
            print(f"#{i+1:<2} {lbl:<22} {cam:<10} "
                  f"({vpos[0]:+.3f},{vpos[1]:+.3f},{vpos[2]:+.3f})  "
                  f"{np.rad2deg(orient):+8.1f}deg {ar:>5.1f} {minor*1000:>9.1f}")
            rows.append({
                "merged_n": i+1, "label": lbl, "camera": cam,
                "position_3d": vpos.tolist(),
                "orientation_angle_deg": float(np.rad2deg(orient)),
                "aspect_ratio": ar, "obb_minor_mm": minor * 1000,
            })

    out = OUT_DIR / f"per_camera_orient_{ts}.json"
    out.write_text(json.dumps(rows, indent=2, default=str))
    print(f"\nsaved -> {out}")


if __name__ == "__main__":
    main()

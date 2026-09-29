"""Standalone slot-detector probe.

Captures one frame from the BIRDVIEW Azure Kinect (SN 000000000000),
runs SAM3 with prompt "grey tray", picks the best mask, then calls
``perception.slot_detector.detect_slots`` and dumps the result.

Mirrors ``scripts/sam3_standalone_probe.py``: no SPARK server, no
behavior tree, no robot motion. Use to validate slot detection on
the actual tray without the executor.

Usage:
    conda activate spark_conda
    python ~/spark/scripts/slot_detector_probe.py
    python ~/spark/scripts/slot_detector_probe.py --tray-prompt "silverware tray"
    python ~/spark/scripts/slot_detector_probe.py --n-fallback 3 --lift 0.05

Outputs (in /tmp/slot_probe/):
    birdview_rgb.png         raw RGB
    birdview_tray_mask.png   SAM3 mask overlay
    birdview_up_mask.png     up-pointing normals inside tray
    birdview_floor_mask.png  slot-floor pixels after band filter
    birdview_slots.png       RGB with slot centroids + indices + axis
    slot_detector_result.json   JSON dump of slot list (world coords + meta)
"""
from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

import cv2
import numpy as np
import pyk4a
from PIL import Image
from pyk4a import ColorResolution, Config, DepthMode, FPS

# SPARK + SAM3 path setup (mirror sam3_standalone_probe.py)
SPARK_SRC = Path.home() / "spark" / "src"
sys.path.insert(0, str(SPARK_SRC))

SAM3_PATHS = [
    Path.home() / "mv_sam3" / "sam3",
]
if os.environ.get("SPARK_SAM3_DIR"):
    SAM3_PATHS.insert(0, Path(os.environ["SPARK_SAM3_DIR"]))
for p in SAM3_PATHS:
    if p.exists():
        sys.path.insert(0, str(p))
        break

OUT_DIR = Path("/tmp/slot_probe")
OUT_DIR.mkdir(parents=True, exist_ok=True)

BIRDVIEW_SN = "000000000000"


# Capture (birdview only).

# Open the birdview Kinect by serial. Returns the pyk4a device.
def open_birdview():
    n_devices = pyk4a.connected_device_count()
    print(f"[probe] connected device count: {n_devices}")
    for i in range(n_devices):
        k = pyk4a.PyK4A(
            Config(
                color_resolution=ColorResolution.RES_720P,
                depth_mode=DepthMode.NFOV_UNBINNED,
                camera_fps=FPS.FPS_15,
                synchronized_images_only=True,
            ),
            device_id=i,
        )
        try:
            k.open()
        except Exception as e:
            print(f"[probe] device {i} open failed: {e}")
            continue
        sn = k.serial
        if sn == BIRDVIEW_SN:
            try:
                k.start()
            except Exception as e:
                print(f"[probe] birdview start failed: {e}")
                return None
            print(f"[probe] opened birdview kinect (sn={sn}, idx={i})")
            return k
        try:
            k.close()
        except Exception:
            pass
    print(f"[probe] FATAL: birdview Kinect (SN {BIRDVIEW_SN}) not found")
    return None


# Discard frames so AE settles, then return one capture.
def capture_warm(k, n_skip: int = 8):
    for _ in range(n_skip):
        try:
            k.get_capture(timeout=1500)
        except Exception:
            pass
    return k.get_capture(timeout=2000)


# Calibration loader (mirror SPARK's loader).

def load_birdview_calibration():
    cal_path = (
        Path.home() / "spark/src/spark_real/output/calibrations/handeye_birdview.json"
    )
    if not cal_path.exists():
        print(f"[probe] FATAL: no calibration at {cal_path}")
        return None
    data = json.loads(cal_path.read_text())
    T = np.array(data["T_cam_to_base_4x4"], dtype=np.float64)
    scale = float(data.get("depth_scale_correction", 1.0) or 1.0)
    rmse = data.get("rmse_mm")
    print(f"[probe] calibration: RMSE={rmse} mm  depth_scale={scale:.4f}  t={T[:3,3]}")
    return {"T_cam_to_base": T, "depth_scale": scale, "rmse_mm": rmse}


# SAM3 tray segmentation (mirror sam3_standalone_probe.py).

def sam3_best_mask(rgb: np.ndarray, prompt: str, threshold: float):
    # DELIBERATELY DEFERRED IMPORTS: torch (and the sam3 package, which pulls
    # torch) must load AFTER the birdview Kinect has been opened and captured
    # (torch loaded first segfaults pyk4a capture; see rollout_recorder.py's
    # subprocess workaround). Do NOT move these to module top.
    import torch
    from sam3.model_builder import build_sam3_image_model
    from sam3.model.sam3_image_processor import Sam3Processor

    print("[probe] loading SAM3 (slow first time)...")
    sam3 = Sam3Processor(build_sam3_image_model())
    sam3.set_confidence_threshold(threshold)
    print("[probe] SAM3 loaded")

    pil = Image.fromarray(rgb)
    state = sam3.set_image(pil)
    state = sam3.set_text_prompt(prompt=prompt, state=state)
    masks = state.get("masks", torch.tensor([]))
    scores = state.get("scores", torch.tensor([]))
    n = int(masks.shape[0]) if masks.ndim > 0 and masks.numel() > 0 else 0
    print(f"[probe] SAM3 prompt {prompt!r}: {n} mask(s)")
    if n == 0:
        return None, 0.0

    # Match the server's selection: pick the HIGHEST-CONFIDENCE mask above
    # threshold (NOT largest), as spark_perception's top-1 scores.argmax().
    # Largest-above-threshold picks low-confidence whole-table masks.
    best_i = -1
    best_score = 0.0
    for i in range(n):
        s = float(scores[i])
        if s < threshold:
            continue
        if s > best_score:
            best_score = s
            best_i = i
    if best_i < 0:
        return None, 0.0
    mask = masks[best_i].cpu().numpy().squeeze().astype(bool)
    print(f"[probe] picked tray mask: idx={best_i} score={best_score:.3f} "
          f"area={int(mask.sum())} px")
    return mask, best_score


# Visualisation helpers.

# Translucent mask overlay on rgb -> returns a copy.
def _draw_mask(rgb: np.ndarray, mask: np.ndarray, color, alpha: float = 0.35):
    img = rgb.copy()
    layer = img.copy()
    layer[mask] = color
    return cv2.addWeighted(layer, alpha, img, 1 - alpha, 0)


# Project a world-frame XYZ point back to image pixel for overlay.
def _world_xy_to_pixel(
    xy_w: np.ndarray, z_w: float, K: np.ndarray, T_cam_to_base: np.ndarray,
):
    R = T_cam_to_base[:3, :3]
    t = T_cam_to_base[:3, 3]
    p_world = np.array([xy_w[0], xy_w[1], z_w], dtype=np.float64)
    p_cam = R.T @ (p_world - t)
    if p_cam[2] <= 0.01:
        return None
    u = float(K[0, 0]) * p_cam[0] / p_cam[2] + float(K[0, 2])
    v = float(K[1, 1]) * p_cam[1] / p_cam[2] + float(K[1, 2])
    return int(round(u)), int(round(v))


def draw_slots(
    rgb: np.ndarray,
    slots: list,
    K: np.ndarray,
    T_cam_to_base: np.ndarray,
    tray_mask: np.ndarray,
) -> np.ndarray:
    img = _draw_mask(rgb, tray_mask, (0, 64, 64), alpha=0.25)
    for s in slots:
        xy = s["world_xyz"][:2]
        z = float(s["world_xyz"][2])
        pix = _world_xy_to_pixel(xy, z, K, T_cam_to_base)
        if pix is None:
            continue
        u, v = pix
        if not (0 <= u < img.shape[1] and 0 <= v < img.shape[0]):
            continue
        color = (0, 255, 0) if s["mode"] == "normals" else (0, 165, 255)
        cv2.circle(img, (u, v), 12, color, 2)
        label = f"{s['slot_idx']} ({s['n_pixels']}px, {s['confidence']:.2f})"
        cv2.putText(img, label, (u + 15, v + 5),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.55, color, 2)
    return img


# Main.

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--tray-prompt", default="tray",
                        help="SAM3 text prompt for the tray "
                             "(default 'tray' matches server)")
    parser.add_argument("--threshold", type=float, default=0.05,
                        help="SAM3 confidence threshold")
    parser.add_argument("--n-fallback", type=int, default=2,
                        help="Number of evenly-spaced slots in fallback mode")
    parser.add_argument("--min-slot-area-px", type=int, default=200)
    parser.add_argument("--slot-floor-band-m", type=float, default=0.015)
    parser.add_argument("--lift", type=float, default=0.04,
                        help="World-Z lift offset added to each slot floor")
    parser.add_argument("--normal-up-threshold", type=float, default=0.9)
    args = parser.parse_args()

    # 1. Calibration first (cheap; fail early if missing).
    cal = load_birdview_calibration()
    if cal is None:
        return 1

    # 2. Open birdview & capture.
    k = open_birdview()
    if k is None:
        return 1
    cap = capture_warm(k, n_skip=8)
    if cap is None or cap.color is None:
        print("[probe] no color frame")
        try: k.stop(); k.close()
        except Exception: pass
        return 1

    bgra = cap.color
    rgb = cv2.cvtColor(bgra, cv2.COLOR_BGRA2RGB)
    depth_mm = cap.transformed_depth
    if depth_mm is None:
        print("[probe] no transformed depth")
        try: k.stop(); k.close()
        except Exception: pass
        return 1
    depth_m = depth_mm.astype(np.float32) / 1000.0
    K_color = k.calibration.get_camera_matrix(1)  # 1 = COLOR

    print(f"[probe] rgb={rgb.shape}  depth={depth_m.shape}  "
          f"valid_depth_pct={float((depth_m > 0.01).mean()*100):.1f}%")
    print(f"[probe] K=fx,fy={K_color[0,0]:.1f},{K_color[1,1]:.1f}  "
          f"cx,cy={K_color[0,2]:.1f},{K_color[1,2]:.1f}")

    cv2.imwrite(str(OUT_DIR / "birdview_rgb.png"),
                cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR))

    # Release the Kinect before SAM3, which takes a while.
    try:
        k.stop(); k.close()
    except Exception:
        pass

    # 3. SAM3 tray mask.
    tray_mask, tray_score = sam3_best_mask(rgb, args.tray_prompt, args.threshold)
    if tray_mask is None:
        print(f"[probe] FAIL: SAM3 found no tray for prompt {args.tray_prompt!r}")
        return 1
    cv2.imwrite(str(OUT_DIR / "birdview_tray_mask.png"),
                cv2.cvtColor(_draw_mask(rgb, tray_mask, (0, 255, 255), 0.45),
                             cv2.COLOR_RGB2BGR))

    # 4. detect_slots.
    # DELIBERATELY DEFERRED IMPORT: slot_detector pulls in torch, which must
    # not be loaded before the Kinect capture above (see sam3_best_mask).
    from spark_real.perception.slot_detector import detect_slots
    slots = detect_slots(
        rgb=rgb,
        depth_m=depth_m,
        tray_mask=tray_mask,
        K=K_color,
        T_cam_to_base=cal["T_cam_to_base"],
        depth_scale=cal["depth_scale"],
        n_fallback_slots=args.n_fallback,
        min_slot_area_px=args.min_slot_area_px,
        slot_floor_band_m=args.slot_floor_band_m,
        lift_offset_m=args.lift,
        normal_up_threshold=args.normal_up_threshold,
    )

    print(f"\n[probe] detect_slots returned {len(slots)} slot(s):")
    for s in slots:
        x, y, z = s["world_xyz"]
        print(f"  slot {s['slot_idx']}: world=({x:.3f}, {y:.3f}, {z:.3f}) "
              f"n_px={s['n_pixels']} conf={s['confidence']:.2f} mode={s['mode']}")

    # 5. Intermediate masks (re-run the inner steps for visualisation).
    # Cheap to do here, since SAM3 has already paid the heavy cost.
    # DELIBERATELY DEFERRED IMPORT: same torch-after-Kinect constraint as
    # detect_slots above.
    from spark_real.perception.slot_detector import (
        _compute_normals_from_depth,
    )
    scale = float(cal["depth_scale"])
    d_scaled = depth_m.astype(np.float32) * (scale if scale != 1.0 else 1.0)
    normals_cam = _compute_normals_from_depth(d_scaled, K_color)
    R = cal["T_cam_to_base"][:3, :3]
    H, W = d_scaled.shape
    n_world = (R @ normals_cam.reshape(-1, 3).T).T.reshape(H, W, 3)
    up_mask = (np.abs(n_world[:, :, 2]) > args.normal_up_threshold) & tray_mask
    up_mask &= d_scaled > 0.01
    cv2.imwrite(str(OUT_DIR / "birdview_up_mask.png"),
                cv2.cvtColor(_draw_mask(rgb, up_mask, (0, 255, 0), 0.45),
                             cv2.COLOR_RGB2BGR))

    # 6. Slot overlay.
    overlay = draw_slots(rgb, slots, K_color, cal["T_cam_to_base"], tray_mask)
    out_path = OUT_DIR / "birdview_slots.png"
    cv2.imwrite(str(out_path), cv2.cvtColor(overlay, cv2.COLOR_RGB2BGR))
    print(f"[probe] slot overlay -> {out_path}")

    # 7. Dump JSON for offline inspection.
    json_path = OUT_DIR / "slot_detector_result.json"
    payload = {
        "tray_prompt": args.tray_prompt,
        "tray_score": float(tray_score),
        "tray_mask_area_px": int(tray_mask.sum()),
        "n_slots": len(slots),
        "calibration_rmse_mm": cal["rmse_mm"],
        "depth_scale": cal["depth_scale"],
        "slots": [
            {
                "slot_idx": int(s["slot_idx"]),
                "world_xyz": [float(v) for v in s["world_xyz"]],
                "n_pixels": int(s["n_pixels"]),
                "confidence": float(s["confidence"]),
                "mode": s["mode"],
            }
            for s in slots
        ],
        "params": {
            "n_fallback_slots": args.n_fallback,
            "min_slot_area_px": args.min_slot_area_px,
            "slot_floor_band_m": args.slot_floor_band_m,
            "lift_offset_m": args.lift,
            "normal_up_threshold": args.normal_up_threshold,
        },
    }
    json_path.write_text(json.dumps(payload, indent=2))
    print(f"[probe] result JSON -> {json_path}")

    print(f"\n[probe] DONE. Outputs in: {OUT_DIR}")
    return 0


if __name__ == "__main__":
    sys.exit(main())

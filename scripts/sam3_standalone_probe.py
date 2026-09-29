"""Standalone SAM3 probe: captures one frame from each Kinect and runs
SAM3 detection on user-provided text prompts. No SPARK server, no
behavior tree, no Gemini. Used to isolate "is SAM3 actually working
on the current scene?" from the full perception/planning stack.

Usage:
    conda activate spark_conda
    python ~/spark/scripts/sam3_standalone_probe.py \
        --prompts "knife handle" "grey tray"

Outputs:
    /tmp/sam3_probe/<cam>_rgb.png         raw RGB
    /tmp/sam3_probe/<cam>_overlay.png     RGB + SAM3 masks + bboxes
    Console: per-camera, per-detection bbox + score + mask area.
"""
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

# Make SPARK's perception module importable to use the SAME SAM3 loader
# the server uses.
SPARK_SRC = Path.home() / "spark" / "src"
sys.path.insert(0, str(SPARK_SRC))

# SAM3 lives outside the spark tree; spark_perception adds its parent
# to sys.path when imported. Mirror that here.
SAM3_PATHS = [
    Path.home() / "mv_sam3" / "sam3",
]
if os.environ.get("SPARK_SAM3_DIR"):
    SAM3_PATHS.insert(0, Path(os.environ["SPARK_SAM3_DIR"]))
for p in SAM3_PATHS:
    if p.exists():
        # SPARK adds the OUTER mv_sam3/sam3 dir (which contains the inner
        # `sam3/` Python package) to sys.path, not the parent. Without
        # this, importlib.resources.files("sam3") resolves to the wrong
        # directory and BPE vocab lookup fails.
        sys.path.insert(0, str(p))
        break

OUT_DIR = Path("/tmp/sam3_probe")
OUT_DIR.mkdir(parents=True, exist_ok=True)


def open_kinects():
    """
    Open both Kinects with the same config SPARK uses. Returns a
    list of (name, pyk4a_device) tuples.
    """
    cams = []
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
        # Match SPARK's name mapping
        if sn == "000000000000":
            name = "birdview"
        elif sn == "000000000000":
            name = "sideview"
        else:
            name = f"kinect_{i}"
        cams.append((name, k))
        try:
            k.start()
        except Exception as e:
            print(f"[probe] device {i} start failed: {e}")
    return cams


# Discard the first few frames so AE/exposure stabilizes.
def capture_warm(k, n_skip: int = 5):
    for _ in range(n_skip):
        try:
            k.get_capture(timeout=1500)
        except Exception:
            pass
    cap = k.get_capture(timeout=2000)
    return cap


# Draw mask outlines + bbox + label on rgb.
def overlay_masks(rgb: np.ndarray, dets) -> np.ndarray:
    img = rgb.copy()
    colors = [(0, 255, 255), (255, 0, 255), (0, 255, 0), (255, 128, 0)]
    for i, det in enumerate(dets):
        color = colors[i % len(colors)]
        mask = det.mask
        if mask is not None:
            cnts, _ = cv2.findContours(
                mask.astype(np.uint8), cv2.RETR_EXTERNAL,
                cv2.CHAIN_APPROX_SIMPLE,
            )
            cv2.drawContours(img, cnts, -1, color, 2)
        x1, y1, x2, y2 = (int(v) for v in det.bbox)
        cv2.rectangle(img, (x1, y1), (x2, y2), color, 2)
        cv2.putText(
            img, f"{det.label} {det.confidence:.2f}",
            (x1, max(15, y1 - 6)),
            cv2.FONT_HERSHEY_SIMPLEX, 0.55, color, 2,
        )
    return img


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--prompts", nargs="+",
        default=["knife handle", "grey tray"],
        help="Text prompts for SAM3 detection",
    )
    parser.add_argument(
        "--threshold", type=float, default=0.05,
        help="SAM3 confidence threshold",
    )
    args = parser.parse_args()

    print(f"[probe] prompts: {args.prompts}")
    print(f"[probe] threshold: {args.threshold}")

    # 1. Open Kinects
    print("[probe] opening Kinects...")
    cams = open_kinects()
    if not cams:
        print("[probe] FATAL: no Kinects available")
        return 1
    print(f"[probe] opened {len(cams)}: {[n for n, _ in cams]}")

    # 2. Capture one warm frame per camera
    captures = {}
    for name, k in cams:
        print(f"[probe] capturing {name}...")
        cap = capture_warm(k, n_skip=8)
        if cap is None or cap.color is None:
            print(f"[probe] {name}: no color frame")
            continue
        # K4A returns BGRA; convert to RGB uint8
        bgra = cap.color
        rgb = cv2.cvtColor(bgra, cv2.COLOR_BGRA2RGB)
        # transformed_depth_image is depth aligned to the COLOR camera frame
        # (uint16, mm). This is what SPARK's pipeline uses.
        depth_mm = cap.transformed_depth
        depth_m = (
            depth_mm.astype(np.float32) / 1000.0
            if depth_mm is not None else None
        )
        # Kinect color intrinsics (after color resolution, so 1280x720 here)
        K_color = k.calibration.get_camera_matrix(1)  # 1 = COLOR
        captures[name] = {
            "rgb": rgb,
            "depth": depth_m,
            "K": K_color,
            "shape": rgb.shape,
        }
        cv2.imwrite(
            str(OUT_DIR / f"{name}_rgb.png"),
            cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR),
        )
        print(f"[probe] {name}: saved rgb {rgb.shape}, depth={'yes' if depth_m is not None else 'no'}")

    # Release Kinects so they don't stay locked
    for name, k in cams:
        try:
            k.stop()
            k.close()
        except Exception:
            pass

    # 3. Load SAM3 directly (skip DA3; masks only)
    print("[probe] loading SAM3 (this is the slow part)...")
    # DELIBERATELY DEFERRED IMPORTS: torch (and the sam3 package, which pulls
    # torch) must load AFTER the Kinects have been opened and captured above
    # (torch loaded first segfaults pyk4a capture; see rollout_recorder.py's
    # subprocess workaround). Do NOT move these to module top.
    import torch
    from sam3.model_builder import build_sam3_image_model
    from sam3.model.sam3_image_processor import Sam3Processor
    sam3 = Sam3Processor(build_sam3_image_model())
    sam3.set_confidence_threshold(args.threshold)
    print("[probe] SAM3 loaded.")

    # Lightweight detection container, just the fields the overlay needs.
    class _Det:
        __slots__ = ("label", "confidence", "bbox", "mask")
        def __init__(self, label, confidence, bbox, mask):
            self.label = label
            self.confidence = confidence
            self.bbox = bbox
            self.mask = mask

    # 3b. Load SPARK calibrations
    SPARK_CAL_DIR = Path.home() / "spark/src/spark_real/output/calibrations"
    calibrations = {}
    for cam_name in captures:
        cal_path = SPARK_CAL_DIR / f"handeye_{cam_name}.json"
        if cal_path.exists():
            cal = json.loads(cal_path.read_text())
            T = np.array(cal["T_cam_to_base_4x4"])
            calibrations[cam_name] = {
                "T_cam_to_base": T,
                "rmse_mm": cal.get("rmse_mm"),
            }
            print(f"[probe] loaded calibration for {cam_name}: RMSE={cal.get('rmse_mm', '?')} mm, t={T[:3,3]}")
        else:
            print(f"[probe] WARN: no calibration found at {cal_path}")

    # Mirror spark_perception._cloud_median_world (OpenCV convention).
    def world_from_mask(mask, depth, K, T_cam_to_base):
        if depth is None:
            return None, "no_depth"
        ys, xs = np.where(mask > 0)
        if len(xs) == 0:
            return None, "empty_mask"
        depths = depth[ys, xs].astype(np.float64)
        valid = (depths > 0.01) & (depths < 10.0)
        if valid.sum() < 3:
            return None, f"too_few_valid_depth_px({int(valid.sum())}/{len(depths)})"
        xs_v = xs[valid].astype(np.float64)
        ys_v = ys[valid].astype(np.float64)
        depths_v = depths[valid]
        fx, fy = K[0, 0], K[1, 1]
        cx_k, cy_k = K[0, 2], K[1, 2]
        # OpenCV convention: z forward, x right, y down
        x_cam = (xs_v - cx_k) * depths_v / fx
        y_cam = (ys_v - cy_k) * depths_v / fy
        z_cam = depths_v
        pts_cam = np.column_stack([x_cam, y_cam, z_cam])
        R = T_cam_to_base[:3, :3]
        t = T_cam_to_base[:3, 3]
        pts_world = (R @ pts_cam.T).T + t
        centroid = np.median(pts_world, axis=0)
        return centroid, f"ok ({int(valid.sum())}/{len(depths)} px valid, depth_median={float(np.median(depths_v)):.3f}m)"

    # 4. Run SAM3 on each captured frame
    for cam_name, cap_data in captures.items():
        rgb = cap_data["rgb"]
        depth = cap_data["depth"]
        K = cap_data["K"]
        print(f"\n[probe]{cam_name}")
        if depth is not None:
            valid_pct = float((depth > 0.01).mean() * 100)
            print(f"[probe] depth: shape={depth.shape}  valid={valid_pct:.1f}%  "
                  f"range=[{depth[depth > 0.01].min():.2f}, {depth.max():.2f}]m")
            print(f"[probe] K=fx,fy={K[0,0]:.1f},{K[1,1]:.1f}  cx,cy={K[0,2]:.1f},{K[1,2]:.1f}")
        cal = calibrations.get(cam_name)
        pil_img = Image.fromarray(rgb)
        dets = []
        for prompt in args.prompts:
            state = sam3.set_image(pil_img)
            state = sam3.set_text_prompt(prompt=prompt, state=state)
            masks = state.get("masks", torch.tensor([]))
            scores = state.get("scores", torch.tensor([]))
            n = int(masks.shape[0]) if masks.ndim > 0 and masks.numel() > 0 else 0
            print(f"  prompt={prompt!r}: SAM3 returned {n} mask(s)")
            if n == 0:
                continue
            # Show ALL instances above threshold, not just the best.
            for i in range(n):
                score_i = float(scores[i])
                if score_i < args.threshold:
                    continue
                mask_i = masks[i].cpu().numpy().squeeze().astype(bool)
                ys, xs = np.where(mask_i)
                if len(xs) == 0:
                    continue
                bbox = (
                    float(xs.min()), float(ys.min()),
                    float(xs.max()), float(ys.max()),
                )
                dets.append(_Det(prompt, score_i, bbox, mask_i))
                # Compute world position via SPARK's same pipeline
                world_str = "(no calibration)"
                if cal is not None and depth is not None:
                    pos, info = world_from_mask(
                        mask_i, depth, K, cal["T_cam_to_base"],
                    )
                    if pos is not None:
                        world_str = (
                            f"world=({pos[0]:.3f}, {pos[1]:.3f}, {pos[2]:.3f}) {info}"
                        )
                    else:
                        world_str = f"world=NONE ({info})"
                print(f"    [{i}] score={score_i:.3f}  bbox={[int(v) for v in bbox]}  "
                      f"mask_px={int(mask_i.sum())}  {world_str}")
        print(f"[probe] {cam_name}: {len(dets)} det(s) above threshold")

        overlay = overlay_masks(rgb, dets)
        out_path = OUT_DIR / f"{cam_name}_overlay.png"
        cv2.imwrite(str(out_path), cv2.cvtColor(overlay, cv2.COLOR_RGB2BGR))
        print(f"[probe] saved overlay -> {out_path}")

    print(f"\n[probe] DONE. Outputs in: {OUT_DIR}")
    return 0


if __name__ == "__main__":
    sys.exit(main())

#!/usr/bin/env python3
"""Offline SAM3 detection with Kinect depth backprojection.

Bypasses the SPARK server entirely. Opens Kinects, runs SAM3, backprojects
mask centroids through calibrated hand-eye transforms to robot base frame.

Usage:
    conda activate spark_conda
    cd ~/spark/src
    python ../scripts/detect_offline.py "mug" "mug rim" "mug handle"
    python ../scripts/detect_offline.py --cam birdview "fork"
"""
from __future__ import annotations

import argparse
import json
import sys
import os

import numpy as np
import pyk4a
from PIL import Image

_src = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "src")
if _src not in sys.path:
    sys.path.insert(0, os.path.abspath(_src))

CAL_DIR = os.path.join(_src, "spark_real", "output", "calibrations")
CAL_FILES = {
    "birdview": os.path.join(CAL_DIR, "handeye_birdview.json"),
    "sideview": os.path.join(CAL_DIR, "handeye_sideview.json"),
}
SERIALS = {"000000000000": "birdview", "000000000000": "sideview"}


def load_calibrations():
    cals = {}
    for cam, path in CAL_FILES.items():
        with open(path) as f:
            d = json.load(f)
        cals[cam] = {
            "T": np.array(d["T_cam_to_base_4x4"]).reshape(4, 4),
            "depth_scale": d.get("depth_scale_correction", 1.0),
        }
    return cals


def open_kinects():
    devs = {}
    for i in range(pyk4a.connected_device_count()):
        k = pyk4a.PyK4A(device_id=i)
        k.open()
        k.start()
        sn = k.serial
        cam = SERIALS.get(sn, f"unknown_{sn}")
        devs[cam] = k
    return devs


def capture(kinects):
    result = {}
    for cam, k in kinects.items():
        cap = k.get_capture()
        rgb = cap.color[:, :, :3][:, :, ::-1].copy()
        depth = cap.transformed_depth.copy()
        intr = k.calibration.get_camera_matrix(pyk4a.CalibrationType.COLOR)
        result[cam] = {
            "rgb": rgb, "depth": depth,
            "fx": intr[0, 0], "fy": intr[1, 1],
            "cx": intr[0, 2], "cy": intr[1, 2],
        }
    return result


def backproject(mask_cx, mask_cy, depth_img, intrinsics, cal):
    iy = max(0, min(depth_img.shape[0] - 1, int(round(mask_cy))))
    ix = max(0, min(depth_img.shape[1] - 1, int(round(mask_cx))))
    r = 5
    h, w = depth_img.shape
    patch = depth_img[max(0, iy-r):min(h, iy+r+1), max(0, ix-r):min(w, ix+r+1)]
    valid = patch[patch > 0]
    if len(valid) == 0:
        return None
    z_m = float(np.median(valid)) / 1000.0 * cal["depth_scale"]
    fx, fy = intrinsics["fx"], intrinsics["fy"]
    cx, cy = intrinsics["cx"], intrinsics["cy"]
    x_cam = (mask_cx - cx) * z_m / fx
    y_cam = (mask_cy - cy) * z_m / fy
    p_base = cal["T"] @ np.array([x_cam, y_cam, z_m, 1.0])
    return p_base[:3]


def detect_point(px_x, px_y, camera="birdview", kinects=None, cals=None):
    """
    Segment via a single point click and backproject to 3D.

    Args:
        px_x: Pixel x-coordinate in the camera image.
        px_y: Pixel y-coordinate in the camera image.
        camera: Which camera to use ('birdview' or 'sideview').
        kinects: Pre-opened Kinect dict (or None to open fresh).
        cals: Pre-loaded calibration dict (or None to load from disk).

    Returns:
        dict with label, camera, confidence, position_3d, centroid_px,
        mask_area -- or None if no mask was produced.
    """
    own_kinects = kinects is None
    if own_kinects:
        kinects = open_kinects()
    if cals is None:
        cals = load_calibrations()

    caps = capture(kinects)
    if camera not in caps:
        if own_kinects:
            for k in kinects.values():
                try:
                    k.stop(); k.close()
                except Exception:
                    pass
        return None

    c = caps[camera]
    pil_img = Image.fromarray(c["rgb"])
    intrinsics = {k: c[k] for k in ("fx", "fy", "cx", "cy")}

    # DELIBERATELY DEFERRED IMPORT: SPARKPerception pulls in torch, and the
    # Kinects must be opened BEFORE torch is loaded in this process (loading
    # torch first segfaults pyk4a capture; see rollout_recorder.py's
    # subprocess workaround). Do NOT move this to module top.
    from spark_real.perception.spark_perception import SPARKPerception
    perc = SPARKPerception()
    perc.load_models(load_da3=False)

    sam3_state = perc._sam3.set_image(pil_img)
    sam3_state = perc.set_point_prompt(px_x, px_y, sam3_state, label=1)

    import torch  # DELIBERATELY DEFERRED: torch after Kinect open (see above)
    masks = sam3_state.get("masks", None)
    scores = sam3_state.get("scores", None)
    if masks is None or (hasattr(masks, "numel") and masks.numel() == 0):
        if own_kinects:
            for k in kinects.values():
                try:
                    k.stop(); k.close()
                except Exception:
                    pass
        return None

    best = int(scores.argmax())
    score = float(scores[best])
    mask = masks[best].cpu().numpy().squeeze()
    ys, xs = np.where(mask > 0)
    if len(xs) == 0:
        if own_kinects:
            for k in kinects.values():
                try:
                    k.stop(); k.close()
                except Exception:
                    pass
        return None

    mcx, mcy = float(xs.mean()), float(ys.mean())
    pos = backproject(mcx, mcy, c["depth"], intrinsics, cals[camera])

    if own_kinects:
        for k in kinects.values():
            try:
                k.stop(); k.close()
            except Exception:
                pass

    return {
        "label": "point_click",
        "camera": camera,
        "confidence": score,
        "position_3d": pos.tolist() if pos is not None else None,
        "centroid_px": (mcx, mcy),
        "mask_area": len(xs),
    }


def detect(prompts, cameras="both", kinects=None, cals=None):
    own_kinects = kinects is None
    if own_kinects:
        kinects = open_kinects()
    if cals is None:
        cals = load_calibrations()

    caps = capture(kinects)

    # DELIBERATELY DEFERRED IMPORT: torch must load after the Kinects are
    # opened (see note in detect_point). Do NOT move to module top.
    from spark_real.perception.spark_perception import SPARKPerception
    perc = SPARKPerception()
    perc.load_models(load_da3=False)

    cam_list = list(caps.keys()) if cameras == "both" else [cameras]
    results = []

    for cam in cam_list:
        if cam not in caps:
            continue
        c = caps[cam]
        pil_img = Image.fromarray(c["rgb"])
        intrinsics = {k: c[k] for k in ("fx", "fy", "cx", "cy")}
        for prompt in prompts:
            state = perc._sam3.set_image(pil_img)
            state = perc._sam3.set_text_prompt(prompt=prompt, state=state)
            masks = state.get("masks", None)
            scores = state.get("scores", None)
            if masks is None or masks.numel() == 0:
                continue
            best = int(scores.argmax())
            score = float(scores[best])
            mask = masks[best].cpu().numpy().squeeze()
            ys, xs = np.where(mask > 0)
            if len(xs) == 0:
                continue
            mcx, mcy = float(xs.mean()), float(ys.mean())
            pos = backproject(mcx, mcy, c["depth"], intrinsics, cals[cam])
            results.append({
                "label": prompt, "camera": cam, "confidence": score,
                "position_3d": pos.tolist() if pos is not None else None,
                "centroid_px": (mcx, mcy), "mask_area": len(xs),
            })

    if own_kinects:
        for k in kinects.values():
            try:
                k.stop(); k.close()
            except Exception:
                pass

    return results


def main():
    parser = argparse.ArgumentParser(description="Offline SAM3 detection")
    parser.add_argument("prompts", nargs="*", help="Text prompts for SAM3")
    parser.add_argument("--cam", default="both", choices=["both", "birdview", "sideview"])
    parser.add_argument("--point", nargs=2, type=float, metavar=("X", "Y"),
                        help="Detect via point click at pixel (X, Y) instead of text prompts")
    args = parser.parse_args()

    if args.point:
        cam = args.cam if args.cam != "both" else "birdview"
        r = detect_point(args.point[0], args.point[1], camera=cam)
        if r is None:
            print("  No detection from point prompt")
        else:
            p = r["position_3d"]
            if p:
                print(f"  {r['camera']:10s}  {r['label']:15s}  conf={r['confidence']:.3f}  "
                      f"x={p[0]:.4f}  y={p[1]:.4f}  z={p[2]:.4f}")
            else:
                print(f"  {r['camera']:10s}  {r['label']:15s}  conf={r['confidence']:.3f}  NO DEPTH")
        return

    if not args.prompts:
        parser.error("text prompts are required unless --point is used")

    results = detect(args.prompts, cameras=args.cam)

    for r in results:
        p = r["position_3d"]
        if p:
            print(f"  {r['camera']:10s}  {r['label']:15s}  conf={r['confidence']:.3f}  "
                  f"x={p[0]:.4f}  y={p[1]:.4f}  z={p[2]:.4f}")
        else:
            print(f"  {r['camera']:10s}  {r['label']:15s}  conf={r['confidence']:.3f}  NO DEPTH")


if __name__ == "__main__":
    main()

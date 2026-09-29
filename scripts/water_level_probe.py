"""Standalone WaterLevelTracker probe.

Opens the sideview Azure Kinect directly (no SPARK server), runs
``WaterLevelTracker.start()`` on a single frame, prints the rim
plane + initial fill estimate, then optionally runs ``update()``
for a few frames so you can sanity-check the per-frame surface_z.

Usage:
    conda activate spark_conda
    python ~/spark/scripts/water_level_probe.py \
        --label "cup" --target-fill 0.7 --update-frames 5

Outputs:
    /tmp/water_level_probe/sideview_rgb.png
    /tmp/water_level_probe/sideview_overlay.png   (RGB + container mask)
    Console: rim_z, floor_z, container height, initial fill_fraction.

Mirrors scripts/sam3_standalone_probe.py for Kinect opening + SAM3
load + calibration loading; reuses SPARKPerception in inprocess mode,
exercising the same code path the server uses.
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time
from pathlib import Path

import cv2
import numpy as np
import pyk4a

# Import SPARK's perception module so the SAME SAM3 loader as the server
# is used.
SPARK_SRC = Path.home() / "spark" / "src"
sys.path.insert(0, str(SPARK_SRC))

# SAM3 lives outside the spark tree; mirror what spark_perception does.
SAM3_PATHS = [
    Path.home() / "mv_sam3" / "sam3",
]
if os.environ.get("SPARK_SAM3_DIR"):
    SAM3_PATHS.insert(0, Path(os.environ["SPARK_SAM3_DIR"]))
for _p in SAM3_PATHS:
    if _p.exists():
        sys.path.insert(0, str(_p))
        break

from spark_real.calibration import CameraCalibration  # noqa: E402
from spark_real.perception.camera import AzureKinectCamera  # noqa: E402
from spark_real.perception.spark_perception import SPARKPerception  # noqa: E402
from spark_real.perception.water_level import WaterLevelTracker  # noqa: E402

OUT_DIR = Path("/tmp/water_level_probe")
OUT_DIR.mkdir(parents=True, exist_ok=True)

SIDEVIEW_SERIAL = "000000000000"
SPARK_CAL_DIR = Path.home() / "spark/src/spark_real/output/calibrations"


def open_sideview_kinect():
    """
    Open the sideview Kinect (by serial). Returns the SPARK
    ``AzureKinectCamera`` wrapper so its capture thread + lock behavior
    match what the server runs.

    Returns (camera, device_id) or (None, None) on failure.
    """
    n = pyk4a.connected_device_count()
    print(f"[probe] connected Kinects: {n}")
    target_id = None
    for i in range(n):
        try:
            dev = pyk4a.PyK4A(device_id=i)
            dev.open()
            sn = dev.serial
            dev.close()
            print(f"[probe]   device {i}: serial={sn}")
            if sn == SIDEVIEW_SERIAL:
                target_id = i
        except Exception as e:
            print(f"[probe]   device {i}: open failed ({e})")
    if target_id is None:
        print(f"[probe] FATAL: sideview Kinect (SN={SIDEVIEW_SERIAL}) not found")
        return None, None

    cam = AzureKinectCamera(
        device_id=target_id,
        color_resolution="1080P",
        depth_mode="WFOV_2X2BINNED",
        sync_mode="STANDALONE",
        camera_fps=15,
    )
    cam.open()
    print(f"[probe] sideview opened: {cam.config.width}x{cam.config.height} "
          f"fx={cam.config.fx:.1f} fy={cam.config.fy:.1f}")
    return cam, target_id


def build_calibration(cam):
    """
    Build a CameraCalibration that mirrors what pipeline._load_handeye
    does at server boot. Loads handeye_sideview.json and stamps in the
    extrinsic + depth_scale.
    """
    cal = CameraCalibration(
        name="sideview",
        width=cam.config.width, height=cam.config.height,
        fx=cam.config.fx, fy=cam.config.fy,
        cx=cam.config.cx, cy=cam.config.cy,
    )
    cal_path = SPARK_CAL_DIR / "handeye_sideview.json"
    if cal_path.exists():
        data = json.loads(cal_path.read_text())
        T_key = ("T_cam_to_base_4x4" if "T_cam_to_base_4x4" in data
                 else "transform_4x4")
        cal.extrinsic = np.array(data[T_key], dtype=np.float64).reshape(4, 4)
        cal.depth_scale = float(data.get("depth_scale_correction", 1.0))
        rmse = data.get("rmse_mm", data.get("residual_mm", 0.0))
        print(f"[probe] loaded sideview calibration: RMSE={rmse:.2f}mm "
              f"depth_scale={cal.depth_scale:.4f}")
    else:
        print(f"[probe] WARN: no calibration at {cal_path}, using identity "
              "extrinsic. Rim/floor Z values will be in CAMERA frame, not robot.")
    return cal


# Draw the container mask + bbox on rgb.
def overlay_mask(rgb: np.ndarray, mask: np.ndarray, bbox) -> np.ndarray:
    img = rgb.copy()
    if mask is not None and mask.any():
        cnts, _ = cv2.findContours(
            mask.astype(np.uint8), cv2.RETR_EXTERNAL,
            cv2.CHAIN_APPROX_SIMPLE,
        )
        cv2.drawContours(img, cnts, -1, (0, 255, 255), 2)
    if bbox is not None:
        x1, y1, x2, y2 = (int(v) for v in bbox)
        cv2.rectangle(img, (x1, y1), (x2, y2), (255, 0, 255), 2)
    return img


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--label", default="cup",
                        help="SAM3 text prompt for the container")
    parser.add_argument("--target-fill", type=float, default=0.70,
                        help="Target fill fraction (0..1)")
    parser.add_argument("--threshold", type=float, default=0.05,
                        help="SAM3 confidence threshold")
    parser.add_argument("--update-frames", type=int, default=0,
                        help="Optionally poll update() N times after start()")
    parser.add_argument("--update-period", type=float, default=0.2,
                        help="Seconds between update() polls")
    args = parser.parse_args()

    # 1. Open sideview Kinect
    print("[probe] opening sideview Kinect...")
    cam, _ = open_sideview_kinect()
    if cam is None:
        return 1

    try:
        cal = build_calibration(cam)

        # 2. Load SPARKPerception (same SAM3 backend the server uses).
        print("[probe] loading SAM3 (slow on first run)...")
        perc = SPARKPerception(sam3_threshold=args.threshold)
        # No DA3 needed: hardware depth from the Kinect.
        perc.load_models(load_da3=False)
        print("[probe] SAM3 ready")

        # 3. Build the tracker in standalone mode.
        tracker = WaterLevelTracker(
            pipeline=None,
            target_label=args.label,
            perception=perc,
            camera=cam,
            calibration=cal,
        )

        # 4. Snap a frame to save the RGB + overlay regardless of whether
        #    start() succeeds.
        rgb, depth = cam.read()
        if rgb is None:
            print("[probe] FATAL: no RGB frame from sideview")
            return 2
        # Match how pipeline.capture() applies depth_scale.
        if depth is not None and cal.depth_scale != 1.0:
            depth = (depth.astype(np.float32) * cal.depth_scale).astype(depth.dtype)
        print(f"[probe] captured frame: rgb={rgb.shape} "
              f"depth={'yes' if depth is not None else 'no'}")
        cv2.imwrite(str(OUT_DIR / "sideview_rgb.png"),
                    cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR))

        # 5. Call start(), this internally re-captures, runs SAM3, fits
        #    the rim/floor plane.
        print(f"[probe] tracker.start(label={args.label!r}, "
              f"target_fill={args.target_fill})...")
        t0 = time.time()
        baseline = tracker.start(target_fill_fraction=args.target_fill)
        print(f"[probe] start() took {time.time() - t0:.2f}s")
        print(f"[probe] baseline = {json.dumps({k: (list(v) if isinstance(v, tuple) else v) for k, v in baseline.items()}, indent=2, default=float)}")

        # Save overlay if start() managed to grab a mask.
        if baseline.get("ok") and tracker._state is not None:
            ov = overlay_mask(rgb, tracker._state.mask_init, baseline.get("bbox"))
            cv2.imwrite(str(OUT_DIR / "sideview_overlay.png"),
                        cv2.cvtColor(ov, cv2.COLOR_RGB2BGR))
            print(f"[probe] saved mask overlay -> {OUT_DIR / 'sideview_overlay.png'}")
        else:
            print(f"[probe] start() failed: {baseline.get('reason')}")
            return 3

        # 6. Optional: poll update() to print per-frame fill estimates.
        if args.update_frames > 0:
            print(f"[probe] polling update() {args.update_frames}x at "
                  f"~{1.0 / args.update_period:.1f}Hz")
            for i in range(args.update_frames):
                time.sleep(args.update_period)
                st = tracker.update()
                print(f"[probe]   {i:02d} {st}")

        tracker.stop()
        print(f"\n[probe] DONE. Outputs in: {OUT_DIR}")
        return 0
    finally:
        try:
            cam.close()
        except Exception as e:
            print(f"[probe] cam.close() raised: {e}")


if __name__ == "__main__":
    sys.exit(main())

# Check both Azure Kinects can detect the ChArUco board, no robot needed.
# Reuses the calibration's detector (Calib3R-unified/spark_calibrate.py). Opens
# the Kinects in the server's safe early order, captures, detects corners on
# each, reports corner count + depth coverage, and saves annotated PNGs.
import os
from pathlib import Path as _P
os.environ.setdefault("DISPLAY", ":1")
os.environ.setdefault("XAUTHORITY", "/run/user/1000/gdm/Xauthority")

EARLY = {}
import pyk4a as _k4a
for _i in range(_k4a.connected_device_count()):
    _d = _k4a.PyK4A(_k4a.Config(
        color_resolution=_k4a.ColorResolution.RES_1080P,
        depth_mode=_k4a.DepthMode.WFOV_2X2BINNED,
        camera_fps=_k4a.FPS.FPS_30, synchronized_images_only=True), device_id=_i)
    _d.start()
    EARLY[_i] = _d
print("[early-kinect] opened:", [d.serial for d in EARLY.values()])

import sys
sys.path.insert(0, str(_P(__file__).resolve().parent.parent / "src"))
sys.path.insert(0, str(_P(__file__).resolve().parent))
import numpy as np
import cv2
from spark_real.pipeline import SPARKRealPipeline, PipelineConfig
from spark_calibrate import load_board, detect_charuco_corners

board, dictionary, meta = load_board(str(_P(__file__).resolve().parent / "charuco_5x7_30mm_4x4_100.json"))
n_total = board.getChessboardCorners().shape[0]
# Family from argv/env, defaulting to PipelineConfig's own default -- never
# hardcoded (role<->serial maps differ per family; wrong family swaps roles).
_FAM = (
    (sys.argv[1] if len(sys.argv) > 1 else None)
    or os.environ.get("SPARK_ROBOT")
    or PipelineConfig.__dataclass_fields__["robot_family"].default
)
print(f"[kinect_board_check] family={_FAM}")
cfg = PipelineConfig(robot_ip="", use_kinect=True, use_realsense=False, robot_family=_FAM)
p = SPARKRealPipeline(cfg)
p._pre_opened_kinects = EARLY
try:
    p._init_kinects_from_early(EARLY)
    p._load_handeye_calibrations()
    caps = p.capture()
    for name in ["birdview", "sideview"]:
        cap = caps.get(name)
        if cap is None:
            print(f"{name:9s}: NOT captured")
            continue
        rgb = cap["rgb"]
        depth = cap.get("depth")
        ids, pix = detect_charuco_corners(rgb, board, dictionary)
        out = cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR)
        if ids is None:
            print(f"{name:9s}: 0/{n_total} corners, board NOT detected")
        else:
            zs = []
            for (u, v) in pix:
                ui, vi = int(round(u)), int(round(v))
                if depth is not None and 0 <= vi < depth.shape[0] and 0 <= ui < depth.shape[1]:
                    z = float(depth[vi, ui])
                    if 0.05 < z < 5.0:
                        zs.append(z)
                    cv2.circle(out, (ui, vi), 6, (0, 255, 0), -1)
            msg = f"{name:9s}: {len(ids)}/{n_total} corners, {len(zs)} with valid depth"
            if zs:
                msg += f", depth median {np.median(zs):.3f} m"
            print(msg)
        cv2.imwrite(os.path.expanduser(f"~/kinect_{name}_board.png"), out)
        print(f"           -> ~/kinect_{name}_board.png")
finally:
    p.shutdown()

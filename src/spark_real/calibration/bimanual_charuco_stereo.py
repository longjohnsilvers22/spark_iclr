"""Stereo ChArUco analysis via ZED Mini.

Grabs a LEFT + RIGHT rectified pair at HD2K, detects ChArUco in each eye
independently and triangulates each shared corner using the ZED SDK's
baseline + intrinsics.

Output: a single JSON with per-corner 3D positions in ZED LEFT optical
frame (the canonical ZED "camera" frame, same one ZED depth uses) plus
per-frame reprojection error.

Usage:
  pkill -f spark_real.server     # ZED is exclusive
  python -m spark_real.calibration.bimanual_charuco_stereo

Defaults match ANON-LAB ChArUco: DICT_4X4_100, 5x7, sq=30mm, marker=22mm.
"""

from __future__ import annotations
import argparse
import json
import sys
import time
from pathlib import Path
from typing import Dict, Tuple

import cv2
import cv2.aruco as aruco
import numpy as np


OUT_DIR = Path.home() / ".spark_real" / "ba_frames"


def triangulate_corners(gray_l: np.ndarray, gray_r: np.ndarray,
                        Kl: np.ndarray, Kr: np.ndarray, baseline_m: float,
                        detector) -> Tuple[Dict[int, np.ndarray], dict]:
    """Detect the ChArUco board in both eyes and triangulate shared corners.

    Returns ({corner_id: xyz in ZED-left optical frame}, info) where info
    carries the per-eye corner counts, the raw detections and the
    reprojection error per eye. The dict is empty when either eye sees no
    corners.
    """
    cc_l, ci_l, mc_l, mi_l = detector.detectBoard(gray_l)
    cc_r, ci_r, mc_r, mi_r = detector.detectBoard(gray_r)
    n_l = 0 if cc_l is None else len(cc_l)
    n_r = 0 if cc_r is None else len(cc_r)
    info = {"n_left": n_l, "n_right": n_r,
            "left": (cc_l, ci_l, mc_l, mi_l), "right": (cc_r, ci_r, mc_r, mi_r)}
    if n_l == 0 or n_r == 0:
        return {}, info

    ids_l = ci_l.ravel().tolist()
    ids_r = ci_r.ravel().tolist()
    common = sorted(set(ids_l) & set(ids_r))
    pts_l = np.asarray([cc_l[ids_l.index(c)].ravel() for c in common], dtype=np.float64)
    pts_r = np.asarray([cc_r[ids_r.index(c)].ravel() for c in common], dtype=np.float64)

    # Rectified-stereo geometry: left cam is the reference, right cam sits
    # +baseline along x in the LEFT optical frame.
    P_l = Kl @ np.hstack([np.eye(3), np.zeros((3, 1))])
    P_r = Kr @ np.hstack([np.eye(3), np.array([[-baseline_m], [0], [0]])])
    X_h = cv2.triangulatePoints(P_l, P_r, pts_l.T, pts_r.T)
    X = (X_h[:3] / X_h[3]).T  # N x 3, meters

    def proj(P, X3):
        x_h = P @ np.hstack([X3, np.ones((len(X3), 1))]).T
        return (x_h[:2] / x_h[2]).T
    info["reproj_err_left_px"] = np.linalg.norm(proj(P_l, X) - pts_l, axis=1)
    info["reproj_err_right_px"] = np.linalg.norm(proj(P_r, X) - pts_r, axis=1)
    return {int(c): X[i] for i, c in enumerate(common)}, info


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--out", type=Path,
                     default=OUT_DIR / "zed_charuco_stereo.json")
    ap.add_argument("--charuco-dict", default="DICT_4X4_100")
    ap.add_argument("--squares-x", type=int, default=5)
    ap.add_argument("--squares-y", type=int, default=7)
    ap.add_argument("--square-m", type=float, default=0.030)
    ap.add_argument("--marker-m", type=float, default=0.022)
    ap.add_argument("--n-frames", type=int, default=5,
                     help="Average over this many frames for noise reduction.")
    args = ap.parse_args(argv)

    # ZED SDK is only needed at run time; keep the module importable without it.
    import pyzed.sl as sl

    OUT_DIR.mkdir(parents=True, exist_ok=True)

    print(f"\nOpening ZED at HD2K...")
    zed = sl.Camera()
    init = sl.InitParameters()
    init.camera_resolution = sl.RESOLUTION.HD2K
    init.camera_fps = 15
    init.depth_mode = sl.DEPTH_MODE.NONE   # only rectified images are needed
    init.coordinate_units = sl.UNIT.METER
    status = zed.open(init)
    if status != sl.ERROR_CODE.SUCCESS:
        print(f"ZED open failed: {status}", file=sys.stderr)
        return 2

    info = zed.get_camera_information()
    cam_cfg = info.camera_configuration
    print(f"  opened. resolution={cam_cfg.resolution.width}x{cam_cfg.resolution.height}")

    # Rectified intrinsics and baseline from the SDK (same source as
    # perception.zed.ZEDCamera.get_camera_info).
    cam_params = cam_cfg.calibration_parameters
    Kl_sdk = np.array([
        [cam_params.left_cam.fx, 0,                        cam_params.left_cam.cx],
        [0,                       cam_params.left_cam.fy,  cam_params.left_cam.cy],
        [0, 0, 1],
    ])
    Kr_sdk = np.array([
        [cam_params.right_cam.fx, 0,                         cam_params.right_cam.cx],
        [0,                        cam_params.right_cam.fy,  cam_params.right_cam.cy],
        [0, 0, 1],
    ])
    baseline_m = abs(cam_params.stereo_transform.get_translation().get()[0])
    print(f"  SDK K_left  fx={Kl_sdk[0,0]:.2f} fy={Kl_sdk[1,1]:.2f} cx={Kl_sdk[0,2]:.2f} cy={Kl_sdk[1,2]:.2f}")
    print(f"  SDK K_right fx={Kr_sdk[0,0]:.2f} fy={Kr_sdk[1,1]:.2f} cx={Kr_sdk[0,2]:.2f} cy={Kr_sdk[1,2]:.2f}")
    print(f"  baseline = {baseline_m*1000:.3f} mm")

    # Build ChArUco
    d = aruco.getPredefinedDictionary(getattr(aruco, args.charuco_dict))
    board = aruco.CharucoBoard((args.squares_x, args.squares_y),
                                args.square_m, args.marker_m, d)
    detector = aruco.CharucoDetector(board)

    # Average corner positions over n frames
    rt = sl.RuntimeParameters()
    mat_l = sl.Mat(); mat_r = sl.Mat()
    per_frame = []

    for f in range(args.n_frames):
        # Skip a couple frames between captures to let autoexposure settle
        for _ in range(3):
            zed.grab(rt)

        if zed.grab(rt) != sl.ERROR_CODE.SUCCESS:
            print(f"frame {f}: grab failed")
            continue
        zed.retrieve_image(mat_l, sl.VIEW.LEFT)
        zed.retrieve_image(mat_r, sl.VIEW.RIGHT)

        img_l = cv2.cvtColor(mat_l.get_data(), cv2.COLOR_BGRA2BGR)
        img_r = cv2.cvtColor(mat_r.get_data(), cv2.COLOR_BGRA2BGR)

        gray_l = cv2.cvtColor(img_l, cv2.COLOR_BGR2GRAY)
        gray_r = cv2.cvtColor(img_r, cv2.COLOR_BGR2GRAY)

        corners, det = triangulate_corners(gray_l, gray_r, Kl_sdk, Kr_sdk,
                                           baseline_m, detector)
        print(f"\nframe {f+1}/{args.n_frames}: LEFT corners={det['n_left']}  RIGHT corners={det['n_right']}")
        if not corners:
            print("  (skipping; need both eyes to triangulate)")
            continue

        ids = sorted(corners)
        err_l = det["reproj_err_left_px"]
        err_r = det["reproj_err_right_px"]
        print(f"  shared corner ids: {len(ids)} = {ids}")
        print(f"  triangulation reproj err (px): left mean={err_l.mean():.3f} max={err_l.max():.3f}  right mean={err_r.mean():.3f} max={err_r.max():.3f}")

        per_frame.append({
            "frame": f,
            "n_left": det["n_left"], "n_right": det["n_right"], "n_common": len(ids),
            "common_ids": ids,
            "stereo_X_in_zed_left_optical": [corners[c].tolist() for c in ids],
            "reproj_err_left_mean_px": float(err_l.mean()),
            "reproj_err_right_mean_px": float(err_r.mean()),
        })

        # Save the first annotated debug pair
        if f == 0:
            for img, (cc, ci, mc, mi), name in ((img_l, det["left"], "left"),
                                                 (img_r, det["right"], "right")):
                ann = img.copy()
                if mi is not None and len(mi) > 0:
                    aruco.drawDetectedMarkers(ann, mc, mi, borderColor=(0,255,0))
                if cc is not None and len(cc) > 0:
                    aruco.drawDetectedCornersCharuco(ann, cc, ci, cornerColor=(0,0,255))
                cv2.imwrite(str(OUT_DIR / f"zed_{name}_annotated_HD2K.png"), ann)

    zed.close()
    print()

    # Average corner positions across accepted frames
    all_corner_ids = sorted({cid for f in per_frame for cid in f["common_ids"]})
    avg_X = {}
    for cid in all_corner_ids:
        Xs = []
        for f in per_frame:
            if cid in f["common_ids"]:
                Xs.append(np.array(f["stereo_X_in_zed_left_optical"][f["common_ids"].index(cid)]))
        if Xs:
            avg_X[cid] = np.mean(Xs, axis=0).tolist()
            std = np.std(Xs, axis=0)
            print(f"  corner id {cid}: avg=({avg_X[cid][0]:.4f},{avg_X[cid][1]:.4f},{avg_X[cid][2]:.4f}) m  std={(std*1000).round(2)} mm  (n={len(Xs)})")

    out = {
        "ts": time.time(),
        "calib_source": "zed_sdk",
        "K_left_sdk":  Kl_sdk.tolist(),
        "K_right_sdk": Kr_sdk.tolist(),
        "baseline_m": baseline_m,
        "board": {
            "dict": args.charuco_dict,
            "squares_x": args.squares_x, "squares_y": args.squares_y,
            "square_m": args.square_m, "marker_m": args.marker_m,
        },
        "n_frames_used": len(per_frame),
        "per_frame": per_frame,
        "averaged_corner_positions_in_zed_left_optical": avg_X,
    }
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(out, indent=2))
    print(f"\nWrote {args.out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())

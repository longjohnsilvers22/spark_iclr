#!/usr/bin/env python3
"""Hand-eye calibration for SPARK Real.

Drives the FR3 through ~25 poses while detecting an AprilTag in the
chosen camera, then solves cam-to-base (or cam-to-TCP for the wrist)
via cv2.calibrateHandEye and saves the result to
output/calibrations/handeye_<camera>.json.

Three camera types:
  - birdview  (static Kinect)   : eye-to-hand. AprilTag on TCP (lollipop).
  - sideview  (static Kinect)   : eye-to-hand. AprilTag on TCP (lollipop).
  - wrist     (RealSense)       : eye-in-hand. AprilTag fixed on table.

Usage:
    # AUTO mode: arm steps through a pre-planned pose set (default 25).
    python scripts/handeye_calibrate.py --camera birdview
    python scripts/handeye_calibrate.py --camera sideview
    python scripts/handeye_calibrate.py --camera wrist

    # MANUAL mode: you move the arm by hand / teleop and press Enter
    # to capture each pose. Good for the first run; safer.
    python scripts/handeye_calibrate.py --camera birdview --mode manual

    # Tag parameters (default: tag36h11 family, 6 cm side):
    python scripts/handeye_calibrate.py --camera birdview \
        --tag-size 0.06 --tag-dict DICT_APRILTAG_36h11 --tag-id 0

The server must be running and have the AprilTag visible to the chosen
camera. Talks to it over HTTP at http://localhost:8888.

What "good" looks like (typical numbers after a clean session):
  - mean reprojection error: < 2 px on Kinects, < 1 px on wrist (close)
  - residual translation:    < 3 mm Kinect, < 1 mm wrist
"""

from __future__ import annotations

import argparse
import base64
import io
import json
import logging
import sys
import time
from pathlib import Path
from typing import List, Optional, Tuple

import cv2
import numpy as np
import requests
from PIL import Image
from scipy.spatial.transform import Rotation

logger = logging.getLogger("handeye")

DEFAULT_SERVER = "http://localhost:8888"
TAG_FAMILIES = {
    name: getattr(cv2.aruco, name) for name in [
        "DICT_APRILTAG_16h5", "DICT_APRILTAG_25h9",
        "DICT_APRILTAG_36h10", "DICT_APRILTAG_36h11",
        "DICT_4X4_50", "DICT_4X4_100", "DICT_4X4_250", "DICT_4X4_1000",
        "DICT_5X5_50", "DICT_5X5_100", "DICT_5X5_250", "DICT_5X5_1000",
        "DICT_6X6_50", "DICT_6X6_100", "DICT_6X6_250", "DICT_6X6_1000",
        "DICT_7X7_50", "DICT_7X7_100", "DICT_7X7_250", "DICT_7X7_1000",
        "DICT_ARUCO_ORIGINAL",
    ] if hasattr(cv2.aruco, name)
}
POSE_PRESET_DIR = Path(__file__).parent / "calibration_poses"


# server I/O

def req_post(url: str, body: dict, timeout: float = 30.0) -> dict:
    r = requests.post(url, json=body, timeout=timeout)
    r.raise_for_status()
    return r.json()


def req_get(url: str, timeout: float = 15.0) -> dict:
    r = requests.get(url, timeout=timeout)
    r.raise_for_status()
    return r.json()


# Return 4x4 TCP pose in robot base frame, or None on failure.
def fetch_tcp(server: str) -> Optional[np.ndarray]:
    try:
        r = req_get(f"{server}/api/robot_state")
        tcp = r.get("tcp_pose")
        if not tcp or len(tcp) < 6:
            return None
    except Exception as exc:
        logger.warning("fetch_tcp failed: %s", exc)
        return None
    # tcp_pose is [x,y,z, rx,ry,rz] axis-angle rotvec
    T = np.eye(4)
    T[:3, :3] = Rotation.from_rotvec(tcp[3:6]).as_matrix()
    T[:3, 3] = tcp[:3]
    return T


# Return (RGB array, intrinsics dict) from the server.
def fetch_camera_frame(server: str, camera: str
                        ) -> Tuple[np.ndarray, dict]:
    r = req_get(f"{server}/api/calibrate/capture_one?camera={camera}",
                 timeout=10.0)
    if r.get("rgb_png_base64") is None:
        raise RuntimeError(f"capture_one failed: {r}")
    png = base64.b64decode(r["rgb_png_base64"])
    img = np.array(Image.open(io.BytesIO(png)).convert("RGB"))
    return img, r["intrinsics"]


def move_to_pose(server: str, pose6: List[float], velocity: float = 0.10
                  ) -> None:
    """
    Drive TCP to the absolute pose [x,y,z,rx,ry,rz]. Blocks.

    On reflex/discontinuity error (returned as HTTP 500 with a JSON
    body), post /api/recover and retry once at half-velocity.  Franka
    FR3 trips a cartesian-velocity reflex when the commanded jump
    from the current pose is too aggressive; after recover, a slower
    motion almost always succeeds.

    Uses ``requests`` directly (not :func:`req_post`) so the 500
    response body is parsed instead of being swallowed by
    ``raise_for_status``.
    """
    payload = {"pose": list(pose6), "velocity": velocity, "wait": True}
    url = f"{server}/api/calibrate/move_to_pose"

    def _post(p: dict) -> dict:
        resp = requests.post(url, json=p, timeout=60.0)
        try:
            body = resp.json()
        except Exception:
            body = {"error": f"non-json response: {resp.text[:200]}"}
        body["__status"] = resp.status_code
        return body

    r = _post(payload)
    if r.get("success"):
        return
    err = str(r.get("error", ""))
    is_reflex = ("discontinuity" in err.lower()
                 or "reflex" in err.lower()
                 or "motion finished" in err.lower()
                 or "control_command" in err.lower())
    if is_reflex:
        try:
            requests.post(f"{server}/api/recover", json={}, timeout=10.0)
        except Exception:
            pass
        payload["velocity"] = max(0.03, velocity * 0.5)
        r = _post(payload)
        if r.get("success"):
            return
    raise RuntimeError(f"move_to_pose failed (status {r.get('__status')}): "
                       f"{r.get('error', r)}")


# tag detection

def detect_tag(rgb: np.ndarray, intrinsics: dict, tag_dict: str,
               tag_id: int, tag_size: float
               ) -> Optional[Tuple[np.ndarray, np.ndarray, float]]:
    """
    Detect one AprilTag and return (R_tag_in_cam, t_tag_in_cam, mean_corner_px).

    Returns None if the tag isn't visible or solvePnP fails.
    mean_corner_px is the mean reprojection error in pixels after PnP
    (a sanity check, values > 3 px suggest a wrong intrinsic or a
    distorted tag).
    """
    aruco = cv2.aruco
    dictionary = aruco.getPredefinedDictionary(TAG_FAMILIES[tag_dict])
    # OpenCV 4.7+ has ArucoDetector; older has detectMarkers
    if hasattr(aruco, "ArucoDetector"):
        params = aruco.DetectorParameters()
        params.cornerRefinementMethod = aruco.CORNER_REFINE_SUBPIX
        detector = aruco.ArucoDetector(dictionary, params)
        corners, ids, _ = detector.detectMarkers(rgb)
    else:
        params = aruco.DetectorParameters_create()
        params.cornerRefinementMethod = aruco.CORNER_REFINE_SUBPIX
        corners, ids, _ = aruco.detectMarkers(rgb, dictionary,
                                              parameters=params)

    if ids is None or len(ids) == 0:
        return None
    ids = ids.flatten().tolist()
    if tag_id not in ids:
        return None
    idx = ids.index(tag_id)
    tag_corners = corners[idx].reshape(4, 2).astype(np.float64)

    # Object-space corner positions for the tag (Z=0 plane, centered)
    s = tag_size / 2.0
    obj_pts = np.array([
        [-s,  s, 0.0],
        [ s,  s, 0.0],
        [ s, -s, 0.0],
        [-s, -s, 0.0],
    ], dtype=np.float64)

    K = np.array([
        [intrinsics["fx"], 0, intrinsics["cx"]],
        [0, intrinsics["fy"], intrinsics["cy"]],
        [0, 0, 1],
    ], dtype=np.float64)
    dist = np.zeros(5)   # SPARK cameras are pre-rectified by their SDKs

    ok, rvec, tvec = cv2.solvePnP(obj_pts, tag_corners, K, dist,
                                   flags=cv2.SOLVEPNP_ITERATIVE)
    if not ok:
        return None

    # Per-corner reprojection error for QA
    proj, _ = cv2.projectPoints(obj_pts, rvec, tvec, K, dist)
    err = float(np.linalg.norm(
        proj.reshape(-1, 2) - tag_corners, axis=1).mean())

    R = cv2.Rodrigues(rvec)[0]
    t = tvec.flatten()
    return R, t, err


# pose collection

def load_presets(camera: str) -> List[List[float]]:
    """
    Pose presets per camera, in robot base frame [x,y,z,rx,ry,rz].
    Returns an empty list if no preset file is present.
    """
    p = POSE_PRESET_DIR / f"poses_{camera}.json"
    if not p.exists():
        return []
    return json.loads(p.read_text()).get("poses", [])


def collect_auto(server: str, camera: str, poses: List[List[float]],
                 tag_dict: str, tag_id: int, tag_size: float,
                 velocity: float = 0.10
                 ) -> List[Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, float]]:
    """
    Move arm through poses, detect tag at each, return list of
    (R_gripper2base, t_gripper2base, R_tag2cam, t_tag2cam, reproj_err_px).
    Skips poses where no tag detection or motion failure.
    """
    samples = []
    for i, p in enumerate(poses):
        logger.info("[%d/%d] move_to_pose(%s)", i + 1, len(poses),
                    [round(v, 3) for v in p])
        try:
            move_to_pose(server, p, velocity=velocity)
        except Exception as exc:
            logger.warning("  move failed: %s, skipping", exc)
            continue
        time.sleep(0.8)   # settle
        # Two reads: one to flush any stale frame, one to use
        try:
            fetch_camera_frame(server, camera)
            rgb, intr = fetch_camera_frame(server, camera)
            tcp = fetch_tcp(server)
        except Exception as exc:
            logger.warning("  capture failed: %s, skipping", exc)
            continue
        if tcp is None or intr is None:
            logger.warning("  missing TCP or intrinsics, skipping")
            continue
        det = detect_tag(rgb, intr, tag_dict, tag_id, tag_size)
        if det is None:
            logger.warning("  tag not detected, skipping")
            continue
        R_tag, t_tag, err_px = det
        logger.info("  tag detected, reproj err %.2f px", err_px)
        samples.append((tcp[:3, :3], tcp[:3, 3], R_tag, t_tag, err_px))
    return samples


def collect_manual(server: str, camera: str, max_poses: int,
                   tag_dict: str, tag_id: int, tag_size: float
                   ) -> List[Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, float]]:
    """
    Operator drives arm by teleop/freedrive. Press Enter at each
    good pose to capture; type 'done' to finish early.
    """
    samples = []
    print(f"\nManual mode: move arm to {max_poses} poses with the AprilTag "
          f"visible to '{camera}'. Press ENTER to capture, 'done' to finish.\n")
    for i in range(max_poses):
        cmd = input(f"[{i+1}/{max_poses}] press ENTER (or 'done'): ").strip()
        if cmd.lower() == "done":
            break
        try:
            rgb, intr = fetch_camera_frame(server, camera)
            tcp = fetch_tcp(server)
        except Exception as exc:
            print(f"  capture failed: {exc}"); continue
        if tcp is None or intr is None:
            print("  missing TCP or intrinsics, try again"); continue
        det = detect_tag(rgb, intr, tag_dict, tag_id, tag_size)
        if det is None:
            print("  tag not detected, reposition + retry"); continue
        R_tag, t_tag, err_px = det
        samples.append((tcp[:3, :3], tcp[:3, 3], R_tag, t_tag, err_px))
        print(f"  captured ({len(samples)} total, reproj {err_px:.2f} px)")
    return samples


# solve + persist

def solve_hand_eye(samples, mode: str) -> Tuple[np.ndarray, float]:
    """
    Run cv2.calibrateHandEye. Returns (T_4x4, residual_mm).

    Pipeline:
      1. Reject outlier samples whose tag-detection reproj error is
         > median + 2*MAD of the batch.  Bad corner localization on a
         few poses (oblique angle, motion blur) dominates an otherwise-
         clean solve.
      2. Run all five OpenCV hand-eye methods (TSAI, PARK, HORAUD,
         ANDREFF, DANIILIDIS) and pick the one with the lowest AX=XB
         chain residual.  Different methods are stable in different
         noise regimes; Park is a sane default but isn't always best.

    For ``eye_to_hand`` (Kinects, static cam, tag on TCP):
      Returns cam_to_base (4x4). Inputs: gripper-to-base poses + tag-to-cam.

    For ``eye_in_hand`` (wrist cam, tag fixed in world):
      Returns cam_to_tcp (4x4). Inputs: base-to-gripper poses + tag-to-cam.
    """
    # Outlier filter on reproj error.  Each sample is
    # (R_g, t_g, R_t, t_t, reproj_err_px).
    errs = np.array([s[4] for s in samples], dtype=float)
    med = float(np.median(errs))
    mad = float(np.median(np.abs(errs - med))) or 1e-6
    cutoff = med + 2.0 * 1.4826 * mad   # 1.4826 ~= MAD-to-sigma for Gaussian
    kept = [s for s in samples if s[4] <= cutoff]
    dropped = len(samples) - len(kept)
    if dropped > 0:
        logger.info("Dropped %d outlier samples (reproj > %.2f px); "
                    "%d samples remain", dropped, cutoff, len(kept))
    samples = kept

    R_g, t_g, R_t, t_t = [], [], [], []
    for Rg, tg, Rt, tt, _ in samples:
        R_g.append(Rg); t_g.append(tg.reshape(3, 1))
        R_t.append(Rt); t_t.append(tt.reshape(3, 1))

    if mode == "eye_to_hand":
        # Pre-invert gripper poses so cv2's input convention yields
        # cam_to_base directly.
        R_in, t_in = [], []
        for Rg, tg in zip(R_g, t_g):
            R_in.append(Rg.T)
            t_in.append((-Rg.T @ tg).reshape(3, 1))
    else:
        R_in, t_in = R_g, t_g

    methods = {
        "TSAI":       cv2.CALIB_HAND_EYE_TSAI,
        "PARK":       cv2.CALIB_HAND_EYE_PARK,
        "HORAUD":     cv2.CALIB_HAND_EYE_HORAUD,
        "ANDREFF":    cv2.CALIB_HAND_EYE_ANDREFF,
        "DANIILIDIS": cv2.CALIB_HAND_EYE_DANIILIDIS,
    }

    best_T = None
    best_residual = float("inf")
    best_name = None
    for name, mid in methods.items():
        try:
            R_out, t_out = cv2.calibrateHandEye(
                R_in, t_in, R_t, t_t, method=mid)
        except Exception as exc:
            logger.info("  method %-10s failed: %s", name, exc)
            continue
        Tk = np.eye(4)
        Tk[:3, :3] = R_out
        Tk[:3, 3] = t_out.flatten()
        rk = _ax_xb_residual_mm(samples, Tk, mode)
        logger.info("  method %-10s residual=%.2f mm", name, rk)
        if rk < best_residual:
            best_residual = rk
            best_T = Tk
            best_name = name
    if best_T is None:
        raise RuntimeError("all hand-eye methods failed")
    logger.info("Best method: %s (residual=%.2f mm)", best_name, best_residual)
    return best_T, best_residual


# Median translation residual of AX=XB chain in mm.
def _ax_xb_residual_mm(samples, T: np.ndarray, mode: str) -> float:
    residuals = []
    for k in range(len(samples) - 1):
        Rg1, tg1, Rt1, tt1, _ = samples[k]
        Rg2, tg2, Rt2, tt2, _ = samples[k + 1]
        B_R = Rt2 @ Rt1.T
        B_t = tt2 - B_R @ tt1
        if mode == "eye_in_hand":
            A_R = Rg2.T @ Rg1
            A_t = Rg2.T @ (tg1 - tg2)
        else:
            A_R = Rg2 @ Rg1.T
            A_t = tg2 - A_R @ tg1
        A = np.eye(4); A[:3,:3] = A_R; A[:3,3] = A_t
        B = np.eye(4); B[:3,:3] = B_R; B[:3,3] = B_t
        AX = A @ T
        XB = T @ B
        residuals.append(
            float(np.linalg.norm(AX[:3, 3] - XB[:3, 3])) * 1000.0)
    return float(np.median(residuals)) if residuals else 0.0


def save_result(server: str, camera: str, mode: str, T: np.ndarray,
                num_poses: int, reproj_err_px: float, residual_mm: float
                ) -> dict:
    r = req_post(f"{server}/api/calibrate/save_handeye", {
        "camera": camera, "mode": mode,
        "transform_4x4": T.tolist(),
        "reprojection_error_px": round(reproj_err_px, 3),
        "residual_mm": round(residual_mm, 3),
        "num_poses": int(num_poses),
        "method": "cv2.calibrateHandEye(PARK)",
    })
    return r


# main

def main():
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--camera",
                        choices=["birdview", "sideview", "wrist"],
                        required=True)
    parser.add_argument("--mode-eye", choices=["auto-detect", "eye_to_hand",
                                               "eye_in_hand"],
                        default="auto-detect",
                        help="auto-detect picks eye-to-hand for static cams, eye-in-hand for wrist")
    parser.add_argument("--mode", choices=["auto", "manual"], default="auto",
                        help="how poses are collected")
    parser.add_argument("--num-poses", type=int, default=25)
    parser.add_argument("--tag-dict", default="DICT_APRILTAG_36h11",
                        choices=list(TAG_FAMILIES.keys()))
    parser.add_argument("--tag-id", type=int, default=0)
    parser.add_argument("--tag-size", type=float, default=0.06,
                        help="AprilTag side length in metres (default 6 cm)")
    parser.add_argument("--server", default=DEFAULT_SERVER)
    parser.add_argument("--poses-file", default=None,
                        help="JSON file with pose list (overrides preset)")
    parser.add_argument("--velocity", type=float, default=0.10,
                        help="TCP move velocity m/s (default 0.10; "
                             "FR3 reflex-trips above ~0.15 on long jumps)")
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO,
                        format="[%(asctime)s] %(message)s",
                        datefmt="%H:%M:%S")

    # Resolve eye mode
    if args.mode_eye == "auto-detect":
        eye = "eye_in_hand" if args.camera == "wrist" else "eye_to_hand"
    else:
        eye = args.mode_eye
    logger.info("Calibration: camera=%s eye_mode=%s collect=%s "
                "tag=%s id=%d size=%.3fm",
                args.camera, eye, args.mode, args.tag_dict, args.tag_id,
                args.tag_size)

    # Sanity check server is up
    try:
        st = req_get(f"{args.server}/api/status", timeout=3.0)
        rok = st.get("robot_connected")
        cam_key = {"birdview": "kinect2_connected",
                   "sideview": "kinect_connected",
                   "wrist": "realsense_connected"}[args.camera]
        cok = st.get(cam_key)
        if not rok:
            logger.error("Robot not connected; aborting"); sys.exit(2)
        if not cok:
            logger.error("Camera %s not connected; aborting", args.camera)
            sys.exit(2)
    except Exception as exc:
        logger.error("Server unreachable at %s: %s", args.server, exc)
        sys.exit(2)

    # Collect samples
    if args.mode == "auto":
        if args.poses_file:
            poses = json.loads(Path(args.poses_file).read_text())["poses"]
        else:
            poses = load_presets(args.camera)
        if not poses:
            logger.error(
                "No pose preset for %s and no --poses-file given. "
                "Either create %s, or use --mode manual.",
                args.camera, POSE_PRESET_DIR / f"poses_{args.camera}.json")
            sys.exit(2)
        if args.num_poses < len(poses):
            poses = poses[:args.num_poses]
        samples = collect_auto(args.server, args.camera, poses,
                                args.tag_dict, args.tag_id, args.tag_size,
                                velocity=args.velocity)
    else:
        samples = collect_manual(args.server, args.camera, args.num_poses,
                                  args.tag_dict, args.tag_id, args.tag_size)

    if len(samples) < 5:
        logger.error("Need >= 5 valid samples; got %d. Aborting.",
                     len(samples))
        sys.exit(2)
    mean_err = float(np.mean([s[4] for s in samples]))
    logger.info("Collected %d valid samples (mean reproj err %.2f px)",
                len(samples), mean_err)

    # Solve
    logger.info("Running cv2.calibrateHandEye (Park)...")
    T, residual_mm = solve_hand_eye(samples, eye)
    logger.info("Solved: residual_mm=%.2f", residual_mm)
    logger.info("Transform 4x4:\n%s", np.array2string(T, precision=4,
                                                      suppress_small=True))

    # Save + apply
    r = save_result(args.server, args.camera, eye, T,
                     num_poses=len(samples),
                     reproj_err_px=mean_err,
                     residual_mm=residual_mm)
    logger.info("Saved -> %s, applied to live pipeline: %s",
                r.get("saved"), r.get("applied"))


if __name__ == "__main__":
    main()

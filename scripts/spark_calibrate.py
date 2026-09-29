"""RGB-D Procrustes hand-eye calibration adapter for SPARK Real.

Drives the arm via /api/calibrate/move_to_pose and captures RGB+depth via
/api/calibrate/capture_one. Uses a ChArUco board (anchored at a known
position in the robot base frame for eye-to-hand, or fixed to the table for
eye-in-hand). Algorithm follows cvg25/Calib3R-unified/calibrate.py:
  1. For each robot pose: detect feature in image -> pixel + depth -> 3D
     in cam frame. Pair with the known 3D in robot base frame.
  2. SVD-Procrustes for the rigid transform between the two point sets.
  3. Nelder-Mead optimisation of a scalar depth-scale to absorb the
     sensor's depth bias.

Usage:
  # 1. Print the ChArUco board, tape it flat on the table.
  # 2. Have the robot touch 3-4 corners to establish board pose in
  #    robot base frame (one-time, saves a JSON sidecar).
  # 3. Then run this script per camera.

  python spark_calibrate.py --mode anchor_board \
      --board scripts/charuco_5x7_30mm_4x4_100.json \
      --out /tmp/board_pose.json
  python spark_calibrate.py --mode calibrate \
      --camera birdview \
      --board-pose /tmp/board_pose.json \
      --num-poses 20

Requires the spark_real server running at http://localhost:8888.
"""

import argparse
import base64
import io
import json
import os
import sys
import time
from datetime import datetime
from pathlib import Path

import cv2
import numpy as np
import requests
import yaml
from cv2 import aruco
from PIL import Image
from scipy import optimize
from scipy.spatial.transform import Rotation as R

SERVER = "http://localhost:8888"
SCRIPT_DIR = Path(__file__).resolve().parent
REPO_ROOT = SCRIPT_DIR.parent
INSTALLED_CAL_DIR = REPO_ROOT / "src/spark_real/output/calibrations"
CONFIG_DIR = REPO_ROOT / "src/spark_real/configs"
DEFAULT_BOARD = str(SCRIPT_DIR / "charuco_5x7_30mm_4x4_100.json")
OUT_DIR = INSTALLED_CAL_DIR
# Frames per height for the static (eye-to-hand) capture.
DEFAULT_STATIC_FRAMES = 10

# Per-family wrist pose spread. Eye-in-hand needs rotational diversity to
# converge; the FR3 wrist hits joint limits so its yaws/tilts stay small, while
# the UR10e's continuous wrist takes a wider (better-conditioned) spread.
# max_yaw_deg is a HARDWARE limit, not a tuning knob: the wrist camera's USB
# cable will not survive a half turn, so every pose is checked against it
# before anything is sent to the arm.
WRIST_POSE_PROFILES = {
    "franka": {
        "heights": [0.22, 0.26, 0.30, 0.34, 0.40],
        "lateral_z": 0.30,
        "yaw_deg": [-60, -30, 30, 60],
        "yaw_z": 0.35,
        "tilt_rad": 0.15,
        "tilt_z": 0.38,
        "max_yaw_deg": 90.0,
    },
    # UR10e heights are ABOVE THE BOARD, not absolute base Z: this rig's table
    # sits at base z ~= -0.28, so an absolute 0.22 would put the camera half a
    # metre up, far outside the D435i's useful range for 34 mm ChArUco squares.
    "ur10e": {
        "heights_above_board": [0.15, 0.20, 0.25, 0.30, 0.35],
        "lateral_dz": 0.22,
        "yaw_deg": [-75, -50, -25, 25, 50, 75],
        "yaw_dz": 0.25,
        # Tilts also ride on the lateral poses so pitch/roll is not confined to
        # the last four moves; the solve needs off-axis rotation to pin the
        # camera's tilt relative to the TCP.
        "lateral_tilt_rad": 0.12,
        "tilt_rad": 0.25,
        "tilt_dz": 0.28,
        "max_yaw_deg": 90.0,
    },
}
OUT_DIR.mkdir(parents=True, exist_ok=True)

# Per-camera workspace planning. (TCP poses where the camera will look
# at the board from a useful angle.) These get sampled into a small
# grid the user can override.
WORKSPACES = {
    "wrist": {  # camera moves; pose is TCP itself
        "x": (0.30, 0.55),
        "y": (-0.20, 0.20),
        "z": (0.22, 0.45),
        "rotvec": [3.1416, 0.0, 0.0],   # down-looking
    },
}


# spark_real I/O

def fetch_frame(camera: str):
    """
    Return (rgb, intrinsic_dict, depth) for one capture from `camera`.

    RGB+intrinsics come from /api/calibrate/capture_one, depth from
    /api/calibrate/capture_depth (None when that endpoint is unavailable).
    """
    r = requests.get(f"{SERVER}/api/calibrate/capture_one",
                     params={"camera": camera}, timeout=10)
    r.raise_for_status()
    d = r.json()
    rgb = np.array(Image.open(io.BytesIO(base64.b64decode(d["rgb_png_base64"]))))
    intr = d["intrinsics"]
    rd = requests.get(f"{SERVER}/api/calibrate/capture_depth",
                      params={"camera": camera}, timeout=10)
    if rd.status_code == 200:
        dd = rd.json()
        if "depth_png_base64" in dd:
            depth = np.array(Image.open(
                io.BytesIO(base64.b64decode(dd["depth_png_base64"]))),
                dtype=np.float32)
            scale = float(dd.get("depth_scale_m_per_unit", 0.001))
            depth = depth * scale
        else:
            depth = None
    else:
        depth = None
    return rgb, intr, depth


# Drive the TCP to a base-frame pose.
def move_to(pose, velocity=0.10):
    r = requests.post(f"{SERVER}/api/calibrate/move_to_pose",
                      json={"pose": list(pose), "velocity": velocity,
                            "wait": True}, timeout=30)
    return r.status_code, r.text


def fetch_tcp():
    r = requests.get(f"{SERVER}/api/robot_state", timeout=5)
    r.raise_for_status()
    return np.array(r.json()["tcp_pose"], dtype=np.float64)


# ChArUco detection

# Construct an OpenCV CharucoBoard from a sidecar JSON.
def load_board(meta_path: Path):
    meta = json.loads(Path(meta_path).read_text())
    dict_name = meta["dictionary"]
    dict_id = getattr(aruco, dict_name)
    sx, sy = meta["squares_x"], meta["squares_y"]
    sq_m = meta["square_length_m"]
    mk_m = meta["marker_length_m"]
    dictionary = aruco.getPredefinedDictionary(dict_id)
    board = aruco.CharucoBoard((sx, sy), sq_m, mk_m, dictionary)
    return board, dictionary, meta


def detect_charuco_corners(rgb, board, dictionary):
    """
    Return (corner_ids, corner_pixels) as ints+float32 arrays.

    Uses CharucoDetector (OpenCV 4.7+) when available; falls back to
    the legacy two-pass detectMarkers + interpolateCornersCharuco.
    """
    gray = cv2.cvtColor(rgb, cv2.COLOR_RGB2GRAY)
    if hasattr(aruco, "CharucoDetector"):
        det = aruco.CharucoDetector(board)
        ch_corners, ch_ids, _, _ = det.detectBoard(gray)
        if ch_ids is None:
            return None, None
        return ch_ids.flatten().astype(int), ch_corners.reshape(-1, 2).astype(np.float64)
    else:
        params = aruco.DetectorParameters()
        m_corners, m_ids, _ = aruco.detectMarkers(gray, dictionary, parameters=params)
        if m_ids is None or len(m_ids) == 0:
            return None, None
        n_ok, ch_corners, ch_ids = aruco.interpolateCornersCharuco(
            m_corners, m_ids, gray, board)
        if n_ok < 4:
            return None, None
        return ch_ids.flatten().astype(int), ch_corners.reshape(-1, 2).astype(np.float64)


def charuco_corner_3d_in_board(corner_id: int, board) -> np.ndarray:
    """
    Return the 3D position of a ChArUco inner corner in BOARD frame.

    OpenCV's CharucoBoard exposes chessboardCorners (Nx3, board frame,
    Z=0). corner_id is 0..(N-1) in row-major order.
    """
    pts = board.getChessboardCorners()
    return pts[corner_id]


# Core algorithm

# Solve T s.t. T @ A = B in the least-squares sense (Kabsch/Procrustes).
def get_rigid_transform(A: np.ndarray, B: np.ndarray):
    assert len(A) == len(B)
    centroid_A = A.mean(axis=0)
    centroid_B = B.mean(axis=0)
    AA = A - centroid_A
    BB = B - centroid_B
    H = AA.T @ BB
    U, S, Vt = np.linalg.svd(H)
    R = Vt.T @ U.T
    if np.linalg.det(R) < 0:
        Vt[2, :] *= -1
        R = Vt.T @ U.T
    t = -R @ centroid_A + centroid_B
    return R, t


def get_rigid_transform_plane_side(A, B, nA, nB):
    """
    Rigid A->B fit CONSTRAINED so the unit plane normal nA maps exactly to
    nB (both oriented by the caller to the physically correct side). Only
    the in-plane rotation angle is free (closed form). Always proper.

    Needed for single-height (planar) eye-to-hand solves: at grazing view
    angles the depth-backprojected constellation is so foreshortened that
    unconstrained Kabsch can return the mirror twin (camera under the
    table) with the SAME residual as the true pose.
    """
    cA, cB = A.mean(axis=0), B.mean(axis=0)

    def _basis(n):
        a = np.array([1.0, 0.0, 0.0]) if abs(n[0]) < 0.9 else np.array([0.0, 1.0, 0.0])
        e1 = np.cross(n, a); e1 /= np.linalg.norm(e1)
        e2 = np.cross(n, e1)  # (e1, e2, n) right-handed
        return e1, e2

    e1, e2 = _basis(nA)
    E1, E2 = _basis(nB)
    X = np.stack([(A - cA) @ e1, (A - cA) @ e2], axis=1)
    Y = np.stack([(B - cB) @ E1, (B - cB) @ E2], axis=1)
    dot = float((X * Y).sum())
    cross = float((X[:, 0] * Y[:, 1] - X[:, 1] * Y[:, 0]).sum())
    th = np.arctan2(cross, dot)
    Rz = np.array([[np.cos(th), -np.sin(th), 0.0],
                   [np.sin(th),  np.cos(th), 0.0],
                   [0.0, 0.0, 1.0]])
    R = np.stack([E1, E2, nB], axis=1) @ Rz @ np.stack([e1, e2, nA], axis=1).T
    t = cB - R @ cA
    return R, t


def fit_with_depth_scale(measured_pts: np.ndarray,
                          observed_pts: np.ndarray,
                          observed_pix: np.ndarray,
                          intr: dict,
                          plane_side: bool = False):
    """
    Joint estimate of (camera_pose, depth_scale, depth_offset).

    The depth correction is affine: z_corrected = z_raw * scale + offset
    (offset in meters), applied to the depth column before the Procrustes
    solve. A scalar scale alone only makes depth correct at one distance
    (the board's height); the offset is what keeps Z accurate across the
    workspace. The offset is only observable when the corner depths span a
    range, board at multiple heights (--heights) for a static camera, or a
    moving wrist camera. With single-height data the offset is
    underdetermined, so it is pinned to 0 (scale-only) and a warning prints.
    Returns (world2cam, scale, offset, rmse).
    """
    fx, fy = intr["fx"], intr["fy"]
    cx, cy = intr["cx"], intr["cy"]
    z_raw = observed_pts[:, 2]
    z_spread = float(z_raw.max() - z_raw.min())
    fit_offset = z_spread >= 0.05  # need >= 5 cm of depth variation
    state = {"world2cam": np.eye(4)}

    n_meas = None
    if plane_side:
        _c = measured_pts.mean(axis=0)
        n_meas = np.linalg.svd(measured_pts - _c)[2][-1]
        if n_meas[2] < 0:
            n_meas = -n_meas  # base-frame board normal, oriented UP

    def residual(scale, offset):
        obs_z = (z_raw * scale + offset).reshape(-1, 1)
        obs_x = (observed_pix[:, [0]] - cx) * obs_z / fx
        obs_y = (observed_pix[:, [1]] - cy) * obs_z / fy
        new_obs = np.concatenate([obs_x, obs_y, obs_z], axis=1)
        if plane_side:
            # cam-frame board normal, oriented TOWARD the camera (origin);
            # face-up board + camera above => base-up maps to toward-camera.
            _co = new_obs.mean(axis=0)
            n_obs = np.linalg.svd(new_obs - _co)[2][-1]
            if float(n_obs @ _co) > 0:
                n_obs = -n_obs
            R, t = get_rigid_transform_plane_side(
                measured_pts, new_obs, n_meas, n_obs)
        else:
            R, t = get_rigid_transform(measured_pts, new_obs)
        state["world2cam"] = np.block([[R, t.reshape(3, 1)],
                                        [np.zeros((1, 3)), np.array([[1.0]])]])
        registered = (R @ measured_pts.T).T + t
        return float(np.sqrt(np.mean(np.sum((registered - new_obs) ** 2, axis=1))))

    if fit_offset:
        def err(p):
            return residual(p[0], p[1])
        res = optimize.minimize(err, np.array([1.0, 0.0]), method="Nelder-Mead",
                                 options={"xatol": 1e-7, "fatol": 1e-9})
        scale, offset = float(res.x[0]), float(res.x[1])
    else:
        print("  WARN: corner depths span only %.0f mm, depth OFFSET is "
              "underdetermined; fitting scale only. Capture the board at "
              "multiple heights (--heights) to fit the offset."
              % (z_spread * 1000))

        def err(p):
            return residual(p[0], 0.0)
        res = optimize.minimize(err, np.array([1.0]), method="Nelder-Mead",
                                 options={"xatol": 1e-7, "fatol": 1e-9})
        scale, offset = float(res.x[0]), 0.0
    rmse = residual(scale, offset)
    return state["world2cam"], scale, offset, rmse


# Mode: anchor_board (touch-test based)

def anchor_board_via_touch(args, board, meta):
    """
    Establish the board's pose in the robot base frame by having the
    operator drive the gripper to >=3 board corners and pressing Enter
    at each contact.

    Workflow:
      1. TOUCH LOOP: user drives gripper to each corner, presses Enter.
         Script records the TCP only (no snapping during touches,
         gripper occluding the board doesn't matter).
      2. CLEAN SNAP: after the user types 'done', they park the arm
         clear of the board. Script snaps the birdview once, detects
         all ChArUco corners, and uses the rough Kinect cal to match
         each recorded TCP to the nearest corner.
      3. PROCRUSTES: solve T_board->base from the matched pairs.

    XY-only matching ignores any Z bias in the rough cal. Board corners are
    30 mm apart in XY, so disambiguation is unambiguous as long as the rough
    cal is correct to better than a half-square (~15 mm) in XY.
    """
    # Corner auto-ID camera: must be a camera whose installed cal is trusted.
    cam_for_id = "birdview"
    print(f"Touch >=3 ChArUco inner corners with the gripper tip.")
    print(f"Board has {len(board.getChessboardCorners())} inner corners.")
    print(f"Drive the gripper above a corner via the web UI / teleop,")
    print(f"lower until the tip touches the corner, press ENTER here.")
    print(f"Corner identity is auto-detected from {cam_for_id} after each press.")
    print(f"Type 'done' to finish, 'quit' to abort.\n")

    # Camera-to-base extrinsic for auto-ID comes from the saved hand-eye file.
    he_path = INSTALLED_CAL_DIR / f"handeye_{cam_for_id}.json"
    if he_path.exists():
        bv_cal = json.loads(he_path.read_text())
        T_bv_cam2base = np.array(
            bv_cal.get("T_cam_to_base_4x4") or bv_cal["transform_4x4"]
        )
        print(f"  using {cam_for_id} cal (residual={bv_cal.get('residual_mm', '?')} mm)\n")
    else:
        # Fresh rig bootstrap: no installed hand-eye yet. The web calib
        # app's rough anchor (saved as da3_anchor_<family>.json with a
        # "transform" 4x4) is accurate to well under the half-square
        # ~15 mm the XY corner auto-ID needs, so accept it as the rough
        # cal. Run the web app anchor step first if neither file exists.
        spark_real_root = REPO_ROOT / "src/spark_real"
        fam = os.environ.get("SPARK_FAMILY", "").lower()
        candidates = []
        if fam:
            candidates.append(spark_real_root / f"da3_anchor_{fam}.json")
        candidates += sorted(spark_real_root.glob("da3_anchor_*.json"))
        candidates.append(spark_real_root / "da3_anchor.json")
        anchor_path = next((p for p in candidates if p.exists()), None)
        if anchor_path is None:
            sys.exit(
                f"Need {he_path} or a web-app rough anchor "
                f"(src/spark_real/da3_anchor_<family>.json) to auto-identify "
                f"corners. Run the calib app anchor step (:8892) first."
            )
        T_bv_cam2base = np.array(
            json.loads(anchor_path.read_text())["transform"]
        ).reshape(4, 4)
        print(f"  no installed hand-eye; using rough anchor {anchor_path.name}\n")

    dictionary = board_meta_dictionary(meta)

    # 1. TOUCH LOOP: record TCP at each user-driven contact. Every TCP is
    # appended to a JSONL progress file immediately; --resume picks it up.
    progress_path = Path(args.out).with_suffix(".touches.jsonl")
    tcp_touches = []  # list of np.ndarray (TCP xyz only)
    if args.resume and progress_path.exists():
        for line in progress_path.read_text().splitlines():
            if not line.strip():
                continue
            rec = json.loads(line)
            tcp_touches.append(np.array(rec["tcp_xyz"], dtype=np.float64))
        print(f"RESUMED with {len(tcp_touches)} prior touches: "
              f"{[p.tolist() for p in tcp_touches]}")
    else:
        progress_path.unlink(missing_ok=True)

    print("\nSTEP 1: Touch >=3 ChArUco inner corners with the closed "
          "fingertip.")
    print("        Drive via the web UI; press ENTER here after each "
          "contact.")
    print("        (Corner identity is auto-resolved at the end.)")
    print()
    while True:
        s = input(f"  [touch #{len(tcp_touches)+1}] ENTER to record, "
                  f"'done', 'quit': ").strip().lower()
        if s == "quit":
            sys.exit(1)
        if s == "done":
            break
        tcp = fetch_tcp()
        tcp_xyz = tcp[:3].copy()
        tcp_touches.append(tcp_xyz)
        with progress_path.open("a") as fh:
            fh.write(json.dumps({"tcp_xyz": tcp_xyz.tolist()}) + "\n")
        print(f"  recorded: TCP=({tcp_xyz[0]:.3f}, {tcp_xyz[1]:.3f}, "
              f"{tcp_xyz[2]:.3f})   (progress -> {progress_path.name})")

    if len(tcp_touches) < 3:
        sys.exit("Need >=3 touches.")

    # 2. CLEAN SNAP: park the arm, snap once, build corner XY map.
    print(f"\nSTEP 2: Park the arm CLEAR of the board so the {cam_for_id}")
    print("        Kinect sees every ChArUco corner. (Freedrive the arm")
    print("        up and back, e.g. TCP z >= 0.40, x <= 0.25.)")
    while True:
        s = input("  ENTER once arm is clear (or 'quit'): ").strip().lower()
        if s == "quit":
            sys.exit(1)
        rgb, intr, depth = fetch_frame(cam_for_id)
        if depth is None:
            print("  no depth, retry"); continue
        ids, pixels = detect_charuco_corners(rgb, board, dictionary)
        if ids is None or len(ids) == 0:
            print("  no ChArUco corners detected, try a clearer view")
            continue
        corner_xy_map = {}
        for cid, (u, v) in zip(ids, pixels):
            ui, vi = int(round(u)), int(round(v))
            h, w = depth.shape
            if not (0 <= ui < w and 0 <= vi < h):
                continue
            z = float(depth[vi, ui])
            if z < 0.05 or z > 5.0:
                continue
            p_cam = np.array([
                (u - intr["cx"]) * z / intr["fx"],
                (v - intr["cy"]) * z / intr["fy"],
                z, 1.0,
            ])
            p_base = (T_bv_cam2base @ p_cam)[:3]
            corner_xy_map[int(cid)] = (float(p_base[0]), float(p_base[1]))
        print(f"  cached {len(corner_xy_map)} corner XY positions: "
              f"{sorted(corner_xy_map.keys())}")
        if len(corner_xy_map) < max(len(tcp_touches), 3):
            print(f"  WARN: need at least {len(tcp_touches)} visible "
                  f"corners; got {len(corner_xy_map)}. Reposition arm.")
            continue
        break

    # 3. CORNER MATCHING: nearest-XY for each recorded TCP.
    touches = []  # list of (corner_id, tcp_xyz)
    used_corners = set()
    for k, tcp_xyz in enumerate(tcp_touches):
        # Greedy nearest, with no-reuse constraint
        best_id = None; best_d = np.inf
        for cid, (cx, cy) in corner_xy_map.items():
            if cid in used_corners:
                continue
            d = float(np.hypot(cx - tcp_xyz[0], cy - tcp_xyz[1]))
            if d < best_d:
                best_d = d; best_id = cid
        if best_id is None:
            print(f"  touch #{k+1}: no free corner, skipping")
            continue
        used_corners.add(best_id)
        touches.append((best_id, tcp_xyz))
        flag = "" if best_d < 0.030 else "  <-- LARGE MISMATCH"
        print(f"  touch #{k+1}: corner #{best_id}  "
              f"TCP=({tcp_xyz[0]:.3f}, {tcp_xyz[1]:.3f}, {tcp_xyz[2]:.3f})  "
              f"xy-mismatch={best_d*1000:.1f}mm{flag}")
    if len(touches) < 3:
        sys.exit("Could not resolve >=3 touches.")
    tcp_pts = np.array([t for _, t in touches])
    board_pts = np.array([board.getChessboardCorners()[cid] for cid, _ in touches])
    R, t = get_rigid_transform(board_pts, tcp_pts)
    T_board2base = np.block([[R, t.reshape(3, 1)],
                              [np.zeros((1, 3)), np.array([[1.0]])]])
    err = np.linalg.norm((R @ board_pts.T).T + t - tcp_pts, axis=1)
    print(f"\nBoard anchored. Per-touch residuals (mm): "
          f"{[f'{e*1000:.1f}' for e in err]}")
    print(f"RMS: {np.sqrt(np.mean(err ** 2)) * 1000:.2f} mm")
    out = {
        "T_board2base": T_board2base.tolist(),
        "touches": [{"corner_id": cid, "tcp_xyz": list(map(float, p))}
                    for cid, p in touches],
        "rms_mm": float(np.sqrt(np.mean(err ** 2)) * 1000),
        "board_meta": meta,
        "timestamp": datetime.now().isoformat(timespec="seconds"),
    }
    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(out, indent=2))
    print(f"Saved -> {out_path}")


# Mode: calibrate (eye-to-hand for static Kinects via known board pose)

def _accumulate_static_corners(camera, board, dictionary, T_board2base, h,
                               num_poses, measured, observed_3d, observed_pix):
    """
    Snap num_poses frames from a static camera, detect ChArUco corners, and
    append (base-frame, cam-frame, pixel) tuples for the board raised by riser
    height h. Mutates the three lists in place; returns the intrinsics dict.
    """
    intr_used = None
    n_before = len(measured)
    for i in range(num_poses):
        rgb, intr, depth = fetch_frame(camera)
        if depth is None:
            sys.exit("camera returned no depth, /api/calibrate/capture_depth missing or broken")
        intr_used = intr
        ids, pixels = detect_charuco_corners(rgb, board, dictionary)
        if ids is None:
            continue
        dh, dw = depth.shape[:2]
        for cid, (u, v) in zip(ids, pixels):
            ui, vi = int(round(u)), int(round(v))
            if not (0 <= ui < dw and 0 <= vi < dh):
                continue
            # Median of a 3x3 depth window, robust to single bad pixels.
            patch = depth[max(0, vi-1):vi+2, max(0, ui-1):ui+2]
            valid = patch[(patch > 0.05) & (patch < 5.0)]
            if valid.size < 3:
                continue
            z = float(np.median(valid))
            # Corner 3D in board frame to base frame via the known anchor,
            # raised by the riser height (base Z is up on the table).
            p_board = board.getChessboardCorners()[cid]
            p_base = (T_board2base @ np.append(p_board, 1.0))[:3] \
                + np.array([0.0, 0.0, h])
            measured.append(p_base)
            observed_3d.append([
                (u - intr["cx"]) * z / intr["fx"],
                (v - intr["cy"]) * z / intr["fy"],
                z,
            ])
            observed_pix.append([u, v])
        time.sleep(0.3)
    print(f"  {camera} @ {h*1000:.0f} mm: +{len(measured) - n_before} pairs")
    return intr_used


# Affine RGB-D Procrustes solve + save for one static camera.
def _solve_and_save_static(camera, measured, observed_3d, observed_pix, intr,
                           heights, board_anchor_rms):
    if len(measured) < 6:
        sys.exit(f"{camera}: only {len(measured)} corner pairs, need >=6")
    measured = np.array(measured)
    observed_3d = np.array(observed_3d)
    observed_pix = np.array(observed_pix)
    T_base2cam, depth_scale, depth_offset, rmse = fit_with_depth_scale(
        measured, observed_3d, observed_pix, intr)
    T_cam2base = np.linalg.inv(T_base2cam)

    # Sanity: camera must sit ABOVE the board plane. If this fires, the
    # capture geometry is broken -- do not save.
    _c = measured.mean(axis=0)
    _n = np.linalg.svd(measured - _c)[2][-1]
    if _n[2] < 0:
        _n = -_n
    if float(_n @ (T_cam2base[:3, 3] - _c)) < 0:
        sys.exit(f"{camera}: solved camera is BELOW the board plane even "
                 "with the constrained fit -- aborting, nothing saved.")

    print(f"\n{camera.upper()} eye-to-hand")
    print(f"  pairs {len(measured)} across {len(heights)} height(s)")
    print(f"  RMSE (residual in cam frame): {rmse*1000:.2f} mm")
    print(f"  Depth scale:  {depth_scale:.5f}   (1.0 = no scale bias)")
    print(f"  Depth offset: {depth_offset*1000:+.1f} mm  (0 = no offset bias)")
    print(f"  T_cam2base translation: {T_cam2base[:3, 3]}")
    out_path = OUT_DIR / f"calib_eyeToHand_{camera}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
    out_path.write_text(json.dumps({
        "camera": camera,
        "mode": "eye_to_hand_rgbd_procrustes",
        "T_cam_to_base_4x4": T_cam2base.tolist(),
        "depth_scale_correction": float(depth_scale),
        "depth_offset_correction": float(depth_offset),
        "heights_m": [float(h) for h in heights],
        "rmse_mm": float(rmse * 1000),
        "num_pairs": int(len(measured)),
        "board_anchor_rms_mm": board_anchor_rms,
        "timestamp": datetime.now().isoformat(timespec="seconds"),
    }, indent=2))
    print(f"  Saved -> {out_path}")


def calibrate_eye_to_hand(args, board, board_meta):
    """
    One static camera, multi-height affine RGB-D Procrustes.

    A flat board at one height cannot separate depth scale from offset
    (underdetermined), so Z is only correct at that height. Raising the board
    to a few known riser heights (--heights) gives the depth variation needed
    to fit the affine offset. --heights 0 (default) keeps the legacy scale-only
    behavior.
    """
    board_pose = json.loads(Path(args.board_pose).read_text())
    T_board2base = np.array(board_pose["T_board2base"])
    print(f"Using board anchor with RMS {board_pose['rms_mm']:.1f} mm")
    heights = args.heights if args.heights else [0.0]
    dictionary = board_meta_dictionary(board_meta)
    measured, observed_3d, observed_pix = [], [], []
    intr_used = None
    for h in heights:
        if len(heights) > 1 or h != 0.0:
            print(f"\n>>> Place the board on a riser {h*1000:.0f} mm thick "
                  f"(flat = 0). Same x,y, raised straight up in Z.")
        print(f"Park the arm CLEAR of the board so {args.camera} can see every "
              f"ChArUco corner.")
        input(f"  ENTER once arm parked and board at {h*1000:.0f} mm: ")
        intr_used = _accumulate_static_corners(
            args.camera, board, dictionary, T_board2base, h,
            args.num_poses or DEFAULT_STATIC_FRAMES,
            measured, observed_3d, observed_pix) or intr_used
    _solve_and_save_static(args.camera, measured, observed_3d, observed_pix,
                           intr_used, heights, board_pose["rms_mm"])


def _solve_joint_from_samples(cameras, sample_specs):
    """
    Joint solve per camera from corner-pair sets saved with --samples-out.

    Each spec is the prefix of a previous calibrate_static run at ONE board
    placement (anchored at its own x,y). Concatenating placements spreads the
    correspondence set across the workspace, so the solved rotation is
    constrained by the full span instead of one sheet: extrapolation error
    scales with (residual / point-set extent), and this grows the extent.
    """
    for cam in cameras:
        m, o3, pix = [], [], []
        intr, heights, rms = None, set(), []
        for spec in sample_specs:
            p = Path(spec)
            if not (p.suffix == ".npz" and p.exists()):
                p = Path(f"{spec}_{cam}.npz")
            if not p.exists():
                sys.exit(f"{cam}: no sample file {p} (run each placement "
                         f"with --samples-out first)")
            d = np.load(p, allow_pickle=False)
            m.append(d["measured"])
            o3.append(d["observed_3d"])
            pix.append(d["observed_pix"])
            intr = json.loads(str(d["intr"]))
            heights.update(float(h) for h in d["heights"])
            rms.append(float(d["anchor_rms"]))
        print(f"{cam}: joint solve over {len(sample_specs)} placements, "
              f"{sum(len(x) for x in m)} corner pairs")
        _solve_and_save_static(cam, np.concatenate(m), np.concatenate(o3),
                               np.concatenate(pix), intr, sorted(heights),
                               max(rms))


def calibrate_static_multi(args, board, board_meta):
    """
    Calibrate several static cameras against ONE board placement, capturing
    all of them at each height in a single pass. One board re-placement per
    height instead of one per camera. Each camera is solved independently from
    its own accumulated multi-height pairs.

    Multi-PLACEMENT workflow (spreads accuracy across the table): run this
    once per board placement with --samples-out /tmp/cal_A (each placement
    freshly anchored via anchor_board), then joint-solve all placements with
    --samples-in /tmp/cal_A /tmp/cal_B /tmp/cal_C (no capture; board state
    irrelevant by then).
    """
    cameras = args.cameras
    if args.samples_in:
        _solve_joint_from_samples(cameras, args.samples_in)
        return
    board_pose = json.loads(Path(args.board_pose).read_text())
    T_board2base = np.array(board_pose["T_board2base"])
    print(f"Using board anchor with RMS {board_pose['rms_mm']:.1f} mm")
    heights = args.heights if args.heights else [0.0]
    dictionary = board_meta_dictionary(board_meta)
    acc = {c: {"m": [], "o3": [], "pix": [], "intr": None} for c in cameras}
    for h in heights:
        if len(heights) > 1 or h != 0.0:
            print(f"\n>>> Place the board on a riser {h*1000:.0f} mm thick "
                  f"(flat = 0). Same x,y, raised straight up in Z.")
        print(f"Park the arm CLEAR so all of [{', '.join(cameras)}] see every "
              f"ChArUco corner.")
        input(f"  ENTER once arm parked and board at {h*1000:.0f} mm: ")
        for cam in cameras:
            d = acc[cam]
            d["intr"] = _accumulate_static_corners(
                cam, board, dictionary, T_board2base, h,
                args.num_poses or DEFAULT_STATIC_FRAMES,
                d["m"], d["o3"], d["pix"]) or d["intr"]
    for cam in cameras:
        d = acc[cam]
        if args.samples_out:
            p = Path(f"{args.samples_out}_{cam}.npz")
            np.savez(p, measured=np.array(d["m"]),
                     observed_3d=np.array(d["o3"]),
                     observed_pix=np.array(d["pix"]),
                     intr=json.dumps(d["intr"]),
                     heights=np.array(heights),
                     anchor_rms=board_pose["rms_mm"])
            print(f"  {cam}: saved {len(d['m'])} pairs -> {p}")
        _solve_and_save_static(cam, d["m"], d["o3"], d["pix"], d["intr"],
                               heights, board_pose["rms_mm"])


def board_meta_dictionary(meta):
    return aruco.getPredefinedDictionary(getattr(aruco, meta["dictionary"]))


# Mode: calibrate_wrist (eye-in-hand with ChArUco on table at known pose)

def detect_robot_family():
    """
    Ask the running server which robot it is driving; None if unavailable.

    Callers must not guess a fallback: each family has a different down-facing
    rotation, and this script auto-drives the arm.
    """
    try:
        r = requests.get(f"{SERVER}/api/robot", timeout=4)
        if r.status_code != 200:
            return None
        return (r.json().get("family") or "").lower() or None
    except Exception:
        return None


# Down-facing TCP rotation for families whose config omits it. Mirrors
# ScoreExecutorCore; keep in step with that file.
FALLBACK_GRASP_ORIENTATION = {
    "ur10e": [2.103, -2.329, 0.059],
    "franka": [np.pi, 0.0, 0.0],
}


def load_family_control(family: str) -> dict:
    """
    Read ``control:`` from ``configs/<family>_default.yaml``.

    Shares the executor's orientation and bounds so this script can't drift
    from it. Missing grasp_orientation falls back to the code default (franka);
    missing bounds disable bounds filtering.
    """
    path = CONFIG_DIR / f"{family}_default.yaml"
    if not path.exists():
        sys.exit(f"No config for family {family!r}: {path} not found")
    cfg = yaml.safe_load(path.read_text()) or {}
    control = dict(cfg.get("control") or {})
    if not control.get("grasp_orientation"):
        fallback = FALLBACK_GRASP_ORIENTATION.get(family)
        if fallback is None:
            sys.exit(
                f"{path} has no control.grasp_orientation and there is no code "
                f"fallback for family {family!r}; cannot aim the wrist"
            )
        control["grasp_orientation"] = list(fallback)
        print(f"  {family}: no control.grasp_orientation in config, using the "
              f"executor default {[round(v, 3) for v in fallback]}")
    return control


def _down_rotvec(control: dict) -> np.ndarray:
    """
    Base orientation for the wrist sweep: the HOME TCP rotation.

    Falls back to grasp_orientation for families with no home rotvec in config
    (franka), preserving their historical pose set.
    """
    return np.asarray(
        control.get("home_tcp_rotvec") or control["grasp_orientation"], dtype=float
    )


def _tool_yaw_deg(base_rotvec, pose_rotvec) -> float:
    """
    Tool-axis (wrist 3) rotation of a pose relative to ``base``, in degrees.

    Measures what the cable actually feels, rather than trusting the requested
    yaw list: tilts perturb orientation too, so the spread is verified after
    composition, not assumed.
    """
    rel = R.from_rotvec(base_rotvec).inv() * R.from_rotvec(pose_rotvec)
    return float(np.rad2deg(rel.as_rotvec()[2]))


def _in_bounds(pose, control) -> bool:
    """True when a pose's XYZ sits inside the family's workspace box."""
    lo = control.get("workspace_min")
    hi = control.get("workspace_max")
    if not lo or not hi:
        return True
    return all(lo[i] <= pose[i] <= hi[i] for i in range(3))


def _generate_wrist_poses(T_board2base: np.ndarray, board_meta: dict,
                          family: str, control: dict):
    """
    Pose set over the board centre, with the rotational diversity the
    eye-in-hand solve needs to converge.

    Down rotation is the HOME TCP rotation (control.home_tcp_rotvec), not
    grasp_orientation -- on the UR10e those differ by ~172 deg at wrist 3, and
    anchoring here keeps the sweep near home so the wrist camera's USB cable is
    never wound up. Spread comes from WRIST_POSE_PROFILES; any pose exceeding
    max_yaw_deg of tool-axis rotation from home is dropped.

    Out-of-bounds poses are dropped, not clamped: a clamped pose is no longer
    the pose the solver was handed.
    """
    profile = WRIST_POSE_PROFILES.get(family)
    if profile is None:
        sys.exit(
            f"No wrist pose profile for family {family!r}. "
            f"Known: {sorted(WRIST_POSE_PROFILES)}"
        )
    cx_m = float(T_board2base[0, 3])
    cy_m = float(T_board2base[1, 3])
    board_z = float(T_board2base[2, 3])
    poses = []
    R_down = _down_rotvec(control)
    R_down_obj = R.from_rotvec(R_down)

    # Heights above the BOARD where the profile gives them that way; families
    # still on absolute base Z keep their historical set.
    rel = "heights_above_board" in profile

    def _z(key_rel, key_abs):
        return board_z + profile[key_rel] if rel else profile[key_abs]

    # Group 1: down-looking at varied height (covers translation in TCP-z).
    heights = ([board_z + h for h in profile["heights_above_board"]] if rel
               else list(profile["heights"]))
    for z in heights:
        poses.append([cx_m, cy_m, z] + list(R_down))

    # Group 2: lateral offsets. Each carries a small tilt so pitch/roll is
    # spread through the set rather than confined to group 4.
    lat_z = _z("lateral_dz", "lateral_z")
    lat_tilt = float(profile.get("lateral_tilt_rad", 0.0))
    offsets = [(0.04, 0.0), (-0.04, 0.0), (0.0, 0.04), (0.0, -0.04),
               (0.03, 0.03), (-0.03, -0.03)]
    for i, (dx, dy) in enumerate(offsets):
        if lat_tilt:
            # Tilt roughly toward the board centre, cycling roll/pitch sign.
            pitch = lat_tilt * (1 if dx > 0 else -1 if dx < 0 else 0)
            roll = lat_tilt * (1 if dy > 0 else -1 if dy < 0 else 0)
            if pitch == 0 and roll == 0:
                pitch = lat_tilt * (1 if i % 2 == 0 else -1)
            rv = (R_down_obj * R.from_rotvec([roll, pitch, 0])).as_rotvec()
        else:
            rv = R_down
        poses.append([cx_m + dx, cy_m + dy, lat_z] + list(rv))

    # Group 3: yaw rotations about world Z. Critical for fixing the
    # T_cam_to_TCP rotation about the cam's optical axis.
    yaw_z = _z("yaw_dz", "yaw_z")
    for yaw_deg in profile["yaw_deg"]:
        rv = (R.from_rotvec([0, 0, np.deg2rad(yaw_deg)]) * R_down_obj).as_rotvec()
        poses.append([cx_m, cy_m, yaw_z] + list(rv))

    # Group 4: larger pitch/roll tilts, applied in the TCP local frame.
    tilt = profile["tilt_rad"]
    tilt_z = _z("tilt_dz", "tilt_z")
    for pitch, roll in [(tilt, 0), (-tilt, 0), (0, tilt), (0, -tilt)]:
        rv = (R_down_obj * R.from_rotvec([roll, pitch, 0])).as_rotvec()
        poses.append([cx_m, cy_m, tilt_z] + list(rv))

    # Cable guard: measure each pose's actual tool-axis rotation from home and
    # drop anything beyond the limit, whatever produced it.
    max_yaw = float(profile.get("max_yaw_deg", 90.0))
    safe, over = [], []
    for p in poses:
        yaw = _tool_yaw_deg(R_down, p[3:])
        (over if abs(yaw) > max_yaw + 1e-6 else safe).append((p, yaw))
    if over:
        worst = max(abs(y) for _, y in over)
        print(f"  WARNING: dropped {len(over)} pose(s) exceeding the "
              f"+/-{max_yaw:.0f} deg wrist-3 limit (worst {worst:.1f} deg).")
    poses = [p for p, _ in safe]
    yaws = [y for _, y in safe]
    if yaws:
        print(f"  wrist-3 yaw from home: {min(yaws):+.1f} .. {max(yaws):+.1f} deg "
              f"(limit +/-{max_yaw:.0f})")

    kept = [p for p in poses if _in_bounds(p, control)]
    dropped = len(poses) - len(kept)
    if dropped:
        print(f"  WARNING: {dropped}/{len(poses)} generated poses fall outside "
              f"the {family} workspace box and were dropped. If many were "
              f"dropped the board is probably outside the arm's reach -- move "
              f"it and re-run anchor_board.")
    return kept


def calibrate_eye_in_hand_wrist(args, board, board_meta):
    """
    Wrist (eye-in-hand) calibration using RGB-D Procrustes.

    Drives the wrist through auto-generated poses above the board.
    At each pose:
      - Reads TCP pose from FK (T_TCP_to_base).
      - Snaps wrist RGB+depth.
      - Detects ChArUco corners.
      - For each detected corner:
          known_board   = corner_xyz_in_board_frame (from print geometry)
          observed_cam  = backproject(pixel, depth)  in WRIST cam frame
          known_TCP     = inv(T_TCP_to_base) @ T_board_to_base @ known_board
        i.e. where each corner LIVES in the TCP frame at this pose.

    Then a single global Procrustes (+ depth-scale fit) solves
    T_cam_to_TCP such that T_cam_to_TCP @ observed_cam_ij == known_TCP_ij
    for every (pose, corner). All poses contribute their corners to one
    big paired-point cloud.
    """
    board_pose = json.loads(Path(args.board_pose).read_text())
    T_board2base = np.array(board_pose["T_board2base"])
    print(f"Board anchor at translation {T_board2base[:3, 3]} "
          f"(rms={board_pose.get('rms_mm', '?')} mm)")

    family = args.robot_family
    if family in (None, "auto"):
        family = detect_robot_family()
        if family is None:
            sys.exit(
                "Could not read the robot family from the server "
                f"({SERVER}/api/robot). Start the server, or pass "
                "--robot-family ur10e|franka explicitly. Refusing to guess: "
                "this mode drives the arm and each family has a different "
                "down-facing TCP rotation."
            )
        print(f"Robot family (from server): {family}")
    else:
        print(f"Robot family (from --robot-family): {family}")

    control = load_family_control(family)
    poses = _generate_wrist_poses(T_board2base, board_meta, family, control)
    # Only truncate when explicitly asked. The generated set is ordered
    # down-looking first, rotations last, so a default cap would silently
    # drop every rotated pose and leave the solve unconditioned.
    if args.num_poses is not None and args.num_poses < len(poses):
        poses = poses[:args.num_poses]
        rotated = sum(
            1 for p in poses
            if not np.allclose(p[3:], control["grasp_orientation"])
        )
        print(f"  WARNING: --num-poses {args.num_poses} keeps only {rotated} "
              f"rotated pose(s). Eye-in-hand needs rotational diversity; "
              f"omit --num-poses to use the whole set.")
    print(f"Generated {len(poses)} wrist poses centred on the board "
          f"(anchored on the HOME rotvec "
          f"{np.round(_down_rotvec(control), 3).tolist()}).")

    if args.dry_run_poses:
        print("\n--dry-run-poses: listing only, the arm will NOT move.\n")
        for i, p in enumerate(poses):
            print(f"  [{i+1:2d}] xyz={[round(v, 3) for v in p[:3]]} "
                  f"rotvec={[round(v, 3) for v in p[3:]]}")
        print(f"\n{len(poses)} poses. Re-run without --dry-run-poses to execute.")
        return

    dictionary = board_meta_dictionary(board_meta)
    all_observed_cam = []   # 3D in wrist cam frame
    all_observed_pix = []   # 2D pixel
    all_known_TCP = []      # 3D in TCP frame
    intr_used = None

    for i, pose in enumerate(poses):
        print(f"\n[{i+1}/{len(poses)}] move_to {[round(v,3) for v in pose]}")
        # Drive (CartesianServo via /api/calibrate/move_to_pose)
        try:
            sc, _ = move_to(pose, velocity=0.10)
            if sc != 200:
                print(f"  move failed (status {sc}), skipping pose")
                continue
        except Exception as exc:
            print(f"  move exception: {exc}, skipping pose")
            continue
        time.sleep(0.5)

        # Read TCP pose -> T_TCP_to_base
        tcp = fetch_tcp()
        T_tcp = np.eye(4)
        T_tcp[:3, :3] = R.from_rotvec(tcp[3:6]).as_matrix()
        T_tcp[:3, 3] = tcp[:3]
        T_tcp_inv = np.linalg.inv(T_tcp)

        # Snap wrist
        rgb, intr, depth = fetch_frame("wrist")
        if depth is None:
            print("  no depth from wrist"); continue
        intr_used = intr
        ids, pixels = detect_charuco_corners(rgb, board, dictionary)
        if ids is None or len(ids) == 0:
            print("  no ChArUco corners detected"); continue
        h, w = depth.shape
        n_added = 0
        for cid, (u, v) in zip(ids, pixels):
            ui, vi = int(round(u)), int(round(v))
            if not (0 <= ui < w and 0 <= vi < h):
                continue
            # Median over 3x3 depth window, robust to single bad pixels
            patch = depth[max(0, vi-1):vi+2, max(0, ui-1):ui+2]
            valid = patch[(patch > 0.05) & (patch < 5.0)]
            if valid.size < 3:
                continue
            z = float(np.median(valid))
            # corner in board frame -> base frame -> TCP frame
            p_board = np.append(board.getChessboardCorners()[int(cid)], 1.0)
            p_base = T_board2base @ p_board
            p_tcp = (T_tcp_inv @ p_base)[:3]
            all_observed_cam.append([
                (u - intr["cx"]) * z / intr["fx"],
                (v - intr["cy"]) * z / intr["fy"],
                z,
            ])
            all_observed_pix.append([u, v])
            all_known_TCP.append(p_tcp)
            n_added += 1
        print(f"  added {n_added} corner pairs ({len(ids)} corners detected)")

    if len(all_observed_cam) < 8:
        sys.exit(f"Only {len(all_observed_cam)} pairs collected -- need >=8")

    observed_cam = np.array(all_observed_cam)
    observed_pix = np.array(all_observed_pix)
    known_TCP = np.array(all_known_TCP)
    print(f"\nTotal pairs across all poses: {len(observed_cam)}")

    # Procrustes + depth_scale optimisation. `fit_with_depth_scale`
    # returns T_measured->observed, which here is T_TCP->cam. Invert
    # for the convention everywhere else uses (T_cam->TCP).
    # The wrist drives ~19 poses at varying heights, so its corner depths
    # already span a range and the affine offset is observable without any
    # extra capture (unlike the static cameras, which need --heights).
    T_TCP_to_cam, depth_scale, depth_offset, rmse = fit_with_depth_scale(
        known_TCP, observed_cam, observed_pix, intr_used)
    T_cam_to_TCP = np.linalg.inv(T_TCP_to_cam)
    print(f"\nWRIST eye-in-hand (RGB-D Procrustes)")
    print(f"  RMSE (residual): {rmse*1000:.2f} mm")
    print(f"  depth_scale:  {depth_scale:.5f}")
    print(f"  depth_offset: {depth_offset*1000:+.1f} mm")
    print(f"  T_cam_to_TCP translation: {T_cam_to_TCP[:3, 3]}")
    print(f"  T_cam_to_TCP:\n{T_cam_to_TCP}")

    out_path = OUT_DIR / f"calib_wrist_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
    out_path.write_text(json.dumps({
        "camera": "wrist",
        "mode": "eye_in_hand_rgbd_procrustes",
        # spark_real loads this key for cam->TCP transform via
        # _resolve_wrist_tool_offset, which expects T_cam_to_TCP.
        "T_cam_to_base_4x4": T_cam_to_TCP.tolist(),
        "depth_scale_correction": float(depth_scale),
        "depth_offset_correction": float(depth_offset),
        "rmse_mm": float(rmse * 1000),
        "num_pairs": int(len(observed_cam)),
        "board_anchor_rms_mm": board_pose.get("rms_mm"),
        "timestamp": datetime.now().isoformat(timespec="seconds"),
    }, indent=2))
    print(f"  Saved -> {out_path}")


# Entrypoint

# Median valid depth (m) in a 3x3 window at pixel (u, v), or None.
def _depth_median_at(depth, u, v):
    vi, ui = int(round(v)), int(round(u))
    dh, dw = depth.shape[:2]
    if not (0 <= ui < dw and 0 <= vi < dh):
        return None
    patch = depth[max(0, vi-1):vi+2, max(0, ui-1):ui+2]
    valid = patch[(patch > 0.05) & (patch < 5.0)]
    if valid.size < 3:
        return None
    return float(np.median(valid))


def validate_z(args, board, board_meta):
    """
    Independent absolute accuracy check for the static cameras.

    Hand-guide the gripper TIP onto ChArUco corners at several heights. The TCP
    (from forward kinematics) is the ground-truth 3D of each touched corner,
    independent of the cameras. For each camera, localize every visible corner
    through the installed calibration (corrected depth, deproject, T_cam_to_base)
    and compare the corner nearest the TCP. Reports X/Y/Z error per touch and a
    summary, so you can see whether Z stays accurate across heights. Reads the
    installed handeye_<camera>.json (extrinsic + depth_scale/offset).
    """
    cameras = args.cameras
    dictionary = board_meta_dictionary(board_meta)
    cal_dir = Path(args.cal_dir)
    cals = {}
    for cam in cameras:
        p = cal_dir / f"handeye_{cam}.json"
        if not p.exists():
            sys.exit(f"missing calibration {p}")
        data = json.loads(p.read_text())
        T_key = "T_cam_to_base_4x4" if "T_cam_to_base_4x4" in data else "transform_4x4"
        cals[cam] = {
            "T": np.array(data[T_key], dtype=np.float64),
            "scale": float(data.get("depth_scale_correction", 1.0)),
            "offset": float(data.get("depth_offset_correction", 0.0)),
        }
        print(f"loaded {cam}: scale={cals[cam]['scale']:.4f} "
              f"offset={cals[cam]['offset']*1000:+.1f} mm")

    print("\nHand-guide the gripper TIP onto a ChArUco corner, then ENTER to "
          "record. Spread touches across heights and the workspace. Type 'q' "
          "to finish.")
    rows = []  # list of (tcp, {camera: error_vector_m})
    while True:
        s = input(f"  [touch #{len(rows)+1}] ENTER to record, 'q' to finish: ")
        if s.strip().lower() == "q":
            break
        tcp = np.array(fetch_tcp(), dtype=np.float64)
        per_cam = {}
        for cam in cameras:
            rgb, intr, depth = fetch_frame(cam)
            if depth is None:
                print(f"    {cam}: no depth")
                continue
            ids, pixels = detect_charuco_corners(rgb, board, dictionary)
            if ids is None:
                print(f"    {cam}: no corners detected")
                continue
            best = None
            for (u, v) in pixels:
                z = _depth_median_at(depth, u, v)
                if z is None:
                    continue
                zc = z * cals[cam]["scale"] + cals[cam]["offset"]
                p_cam = np.array([(u - intr["cx"]) * zc / intr["fx"],
                                  (v - intr["cy"]) * zc / intr["fy"], zc])
                p_base = (cals[cam]["T"] @ np.append(p_cam, 1.0))[:3]
                e = float(np.linalg.norm(p_base - tcp))
                if best is None or e < best[0]:
                    best = (e, p_base)
            if best is None:
                print(f"    {cam}: no corner with valid depth")
                continue
            ev = best[1] - tcp
            per_cam[cam] = ev
            print(f"    {cam}: |err|={best[0]*1000:5.1f} mm  "
                  f"dx={ev[0]*1000:+6.1f} dy={ev[1]*1000:+6.1f} dz={ev[2]*1000:+6.1f}")
        rows.append((tcp, per_cam))

    if not rows:
        return
    print("\nsummary (mean over touches)")
    for cam in cameras:
        evs = np.array([r[1][cam] for r in rows if cam in r[1]])
        if len(evs) == 0:
            continue
        mae = np.abs(evs).mean(axis=0) * 1000
        rms = float(np.sqrt((evs ** 2).sum(axis=1)).mean()) * 1000
        print(f"  {cam}: n={len(evs)}  RMS={rms:.1f} mm  "
              f"mean|dz|={mae[2]:.1f} mm  (mean|dx| {mae[0]:.1f}, mean|dy| {mae[1]:.1f})")


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                  formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--mode", required=True,
                     choices=["anchor_board", "calibrate", "calibrate_static",
                              "calibrate_wrist", "validate_z"])
    ap.add_argument("--board", default=DEFAULT_BOARD,
                     help="Path to the ChArUco board sidecar JSON.")
    ap.add_argument("--out", default="/tmp/board_pose.json",
                     help="(anchor_board mode) Where to save the board pose.")
    ap.add_argument("--board-pose", default="/tmp/board_pose.json",
                     help="(calibrate mode) Board-anchor JSON produced by "
                          "anchor_board.")
    ap.add_argument("--camera", choices=["birdview", "sideview", "wrist"],
                     help="(calibrate mode) Which single camera to calibrate.")
    ap.add_argument("--cameras", nargs="*", default=["birdview", "sideview"],
                     help="(calibrate_static / validate_z) Static cameras to "
                          "capture in one pass, e.g. --cameras birdview sideview.")
    ap.add_argument("--cal-dir", default=str(INSTALLED_CAL_DIR),
                     help="(validate_z) Directory holding the installed "
                          "handeye_<camera>.json files to check.")
    ap.add_argument("--num-poses", type=int, default=None,
                     help="Frames to capture (eye-to-hand, default 10). For "
                          "eye-in-hand the default is the whole generated set; "
                          "capping it drops the rotated poses the solve needs.")
    ap.add_argument("--robot-family", default="auto",
                     choices=["auto", "ur10e", "franka"],
                     help="(calibrate_wrist) Which arm is being driven. "
                          "'auto' reads it from the running server. Sets the "
                          "down-facing TCP rotation and the workspace box, so "
                          "an explicit value must match the real robot.")
    ap.add_argument("--dry-run-poses", action="store_true",
                     help="(calibrate_wrist) Print the generated pose set and "
                          "exit WITHOUT moving the arm. Run this first.")
    ap.add_argument("--heights", type=float, nargs="*", default=[0.0],
                     help="(calibrate eye-to-hand) board riser heights in "
                          "METERS, e.g. --heights 0 0.05 0.10. Multiple heights "
                          "let the affine fit recover the depth OFFSET (not just "
                          "scale), keeping Z accurate across the workspace. "
                          "Default [0] keeps the legacy scale-only behavior.")
    ap.add_argument("--resume", action="store_true",
                     help="(anchor_board mode) Resume from a previous "
                          "incomplete touch session. Reads "
                          "<out>.touches.jsonl and skips already-recorded "
                          "touches.")
    ap.add_argument("--samples-out", default=None,
                     help="(calibrate_static) Also save this placement's "
                          "corner pairs to <prefix>_<camera>.npz for a later "
                          "multi-placement joint solve.")
    ap.add_argument("--samples-in", nargs="*", default=None,
                     help="(calibrate_static) Skip capture; joint-solve each "
                          "camera from these saved sample prefixes/files "
                          "(one per board placement).")
    args = ap.parse_args()

    board, dictionary, meta = load_board(args.board)
    print(f"Loaded board: {meta['squares_x']}x{meta['squares_y']} squares, "
          f"{meta['square_length_m']*1000:.1f}mm, dict={meta['dictionary']}")

    if args.mode == "anchor_board":
        anchor_board_via_touch(args, board, meta)
    elif args.mode == "calibrate":
        if not args.camera:
            sys.exit("--camera required in calibrate mode")
        calibrate_eye_to_hand(args, board, meta)
    elif args.mode == "calibrate_static":
        calibrate_static_multi(args, board, meta)
    elif args.mode == "calibrate_wrist":
        calibrate_eye_in_hand_wrist(args, board, meta)
    elif args.mode == "validate_z":
        validate_z(args, board, meta)


if __name__ == "__main__":
    main()

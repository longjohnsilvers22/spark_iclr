"""Bimanual anchor-tap calibration via ChArUco corners measured by ZED stereo.

PRECONDITIONS (run these first, once):
1. Place ChArUco board flat on the table within both arms' reach + ZED FOV.
2. Run ``python -m spark_real.calibration.bimanual_charuco_stereo`` to measure all 24 inner
   corners in ZED-left optical frame. Writes
   ``~/.spark_real/ba_frames/zed_charuco_stereo.json``.
3. Start ``spark_real.server`` (this script reads TCPs over HTTP -- no FCI
   conflict with the server).

PROCEDURE:
For each arm, the operator drives the gripper (pilot button) so the closed
SSG-48 jaw tip touches each of the 5 designated ChArUco inner corners. Press
ENTER after each tap. The script:

    anchor_in_arm_base = TCP_translation + R_flange @ [0, 0, GRIP_OFFSET_M]

The 3D position of the same corner in ZED-left optical frame comes from the
stereo measurement file. Per-arm SVD (Umeyama / Arun) over N anchor pairs
gives ``T_arm_base_to_zed``. ``T_right_to_left`` falls out:

    T_right_to_left = T_zed_to_baseL @ inv(T_zed_to_baseR)

Output: ``~/.spark_real/calibration_bimanual.json`` with all transforms,
plus per-arm ``base_T_zed_<arm>.json`` for downstream consumers.

USAGE:
    python -m spark_real.calibration.bimanual_anchor_tap [--corners 0,3,11,20,23]
                                              [--grip-offset 0.1275]
                                              [--arms left,right]
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path
from typing import Dict, List, Sequence, Tuple

import cv2
import numpy as np
import requests

from spark_real.calibration.bimanual_charuco_stereo import triangulate_corners
from spark_real.calibration.solver import compute_transform_from_pairs


OUT_DIR = Path.home() / ".spark_real"
STEREO_JSON = OUT_DIR / "ba_frames" / "zed_charuco_stereo.json"

# Default 5 anchor corner IDs spread across the ChArUco inner grid
# (4 cols x 6 rows, ids 0-23; layout per spark_real.calibration.bimanual_charuco_stereo):
#     +3  +7  +11  +15  +19  +23   <- far row
#     +2  +6  +10  +14  +18  +22
#     +1  +5  +9   +13  +17  +21
#     +0  +4  +8   +12  +16  +20   <- near row
DEFAULT_CORNERS = [0, 3, 20, 23, 11]
DEFAULT_GRIP_OFFSET_M = 0.1275  # flange to closed jaw tip; keep in sync with grippers.jaw_offset_m
# Stereo capture through the server (multi-position mode). Board layout
# matches the bimanual_charuco_stereo defaults: DICT_4X4_100, 5x7 squares,
# 30 mm squares, 22 mm markers.
N_FRAMES = 5
BOARD = (5, 7, 0.030, 0.022)


def _capture_stereo_via_server(server: str, out_json: Path):
    """Grab stereo frames from the running server and run ChArUco detection."""
    import base64
    import cv2.aruco as aruco

    url = f"{server.rstrip('/')}/api/capture/stereo"
    per_frame = []

    info_resp = requests.get(url, timeout=10)
    info_resp.raise_for_status()
    data = info_resp.json()
    cam_info = data["info"]

    Kl = np.array([
        [cam_info["fx_l"], 0, cam_info["cx_l"]],
        [0, cam_info["fy_l"], cam_info["cy_l"]],
        [0, 0, 1],
    ])
    Kr = np.array([
        [cam_info["fx_r"], 0, cam_info["cx_r"]],
        [0, cam_info["fy_r"], cam_info["cy_r"]],
        [0, 0, 1],
    ])
    baseline_m = cam_info["baseline_m"]

    sq_x, sq_y, sq_m, mk_m = BOARD
    d = aruco.getPredefinedDictionary(aruco.DICT_4X4_100)
    board = aruco.CharucoBoard((sq_x, sq_y), sq_m, mk_m, d)
    detector = aruco.CharucoDetector(board)

    for f in range(N_FRAMES):
        time.sleep(0.3)
        resp = requests.get(url, timeout=10)
        resp.raise_for_status()
        frame = resp.json()
        left = cv2.imdecode(np.frombuffer(base64.b64decode(frame["left_png_b64"]),
                            np.uint8), cv2.IMREAD_COLOR)
        right = cv2.imdecode(np.frombuffer(base64.b64decode(frame["right_png_b64"]),
                             np.uint8), cv2.IMREAD_COLOR)
        gray_l = cv2.cvtColor(left, cv2.COLOR_BGR2GRAY)
        gray_r = cv2.cvtColor(right, cv2.COLOR_BGR2GRAY)
        corners, det = triangulate_corners(gray_l, gray_r, Kl, Kr, baseline_m, detector)
        print(f"  frame {f+1}/{N_FRAMES}: LEFT={det['n_left']} RIGHT={det['n_right']} corners")
        if corners:
            per_frame.append({cid: xyz.tolist() for cid, xyz in corners.items()})

    if not per_frame:
        raise RuntimeError("no frames with detected corners in both eyes")

    all_ids = sorted(set().union(*(d.keys() for d in per_frame)))
    averaged = {}
    for cid in all_ids:
        pts = [d[cid] for d in per_frame if cid in d]
        averaged[str(cid)] = np.mean(pts, axis=0).tolist()

    out_json.parent.mkdir(parents=True, exist_ok=True)
    result = {
        "averaged_corner_positions_in_zed_left_optical": averaged,
        "n_frames": len(per_frame),
        "baseline_m": baseline_m,
    }
    out_json.write_text(json.dumps(result, indent=2))
    print(f"  wrote {out_json} ({len(averaged)} corners)")


def fetch_tcp(server: str, arm: str) -> Tuple[np.ndarray, np.ndarray]:
    """GET /api/bimanual/state and return (R_flange_in_base, t_flange_in_base)."""
    url = f"{server.rstrip('/')}/api/bimanual/state"
    r = requests.get(url, timeout=5)
    r.raise_for_status()
    data = r.json()
    p = data.get(arm, {}).get("tcp_pose")
    if p is None or len(p) < 6:
        raise RuntimeError(f"/api/bimanual/state did not return tcp_pose for arm={arm}")
    t = np.asarray(p[:3], dtype=np.float64)
    rvec = np.asarray(p[3:6], dtype=np.float64).reshape(3, 1)
    R, _ = cv2.Rodrigues(rvec)
    return R, t


def fit_residuals(P_src: np.ndarray, P_dst: np.ndarray, T: np.ndarray
                   ) -> np.ndarray:
    P_src_h = np.hstack([P_src, np.ones((len(P_src), 1))])
    pred = (T @ P_src_h.T).T[:, :3]
    return np.linalg.norm(pred - P_dst, axis=1)


def load_stereo_corners(path: Path) -> Dict[int, np.ndarray]:
    if not path.exists():
        raise RuntimeError(
            f"Stereo corner file not found at {path}. Run "
            f"`python -m spark_real.calibration.bimanual_charuco_stereo` first.")
    data = json.loads(path.read_text())
    avg = data.get("averaged_corner_positions_in_zed_left_optical")
    if not avg:
        raise RuntimeError(f"{path} has no averaged corner positions")
    return {int(k): np.asarray(v, dtype=np.float64) for k, v in avg.items()}


def collect_taps(server: str, arm: str, corner_ids: Sequence[int],
                 grip_offset_m: float
                 ) -> List[Tuple[int, np.ndarray, np.ndarray, np.ndarray]]:
    """Interactive tap collection for one arm.

    Returns list of (corner_id, tip_in_base, R_flange_in_base, t_flange_in_base).
    """
    out = []
    print()
    print(f"=" * 60)
    print(f"ARM = {arm.upper()}    grip offset = {grip_offset_m*1000:.1f} mm")
    print(f"You will tap {len(corner_ids)} corners in order: {list(corner_ids)}")
    print(f"For each tap:")
    print(f"  1. Pilot-button drive the gripper so the CLOSED jaw tip")
    print(f"     touches the inner corner labeled with the printed ID.")
    print(f"  2. Confirm the arm is stable.")
    print(f"  3. RELEASE the pilot button.")
    print(f"  4. Press ENTER here.")
    print(f"Type 's' + ENTER to skip a corner, 'q' + ENTER to abort.")
    print(f"=" * 60)

    # The prompt names the corner ID for each tap; the operator follows the
    # order shown in the annotated board image.
    for i, cid in enumerate(corner_ids):
        while True:
            try:
                ans = input(
                    f"\n[arm={arm}] tap {i+1}/{len(corner_ids)} -> corner "
                    f"{cid}, ENTER when ready: ").strip().lower()
            except (KeyboardInterrupt, EOFError):
                print("\naborted")
                raise SystemExit(1)
            if ans == "q":
                raise SystemExit(1)
            if ans == "s":
                print(f"   skipped tap {i+1}")
                break
            try:
                R, t = fetch_tcp(server, arm)
            except Exception as e:
                print(f"   ! fetch failed: {e}; retry")
                continue
            tip = t + R @ np.array([0.0, 0.0, grip_offset_m])
            print(f"   flange_pos (mm): {(t*1000).round(1).tolist()}")
            print(f"   tip_pos    (mm): {(tip*1000).round(1).tolist()}")
            out.append((cid, tip, R, t))
            break
    return out


def solve_arm(taps, corner_pos_zed: Dict[int, np.ndarray]):
    """Run SVD on the (tip_in_base, corner_in_zed) pairs.

    Returns dict with T_zed_to_arm (the convention used downstream),
    T_arm_to_zed, residuals, and per-tap data.
    """
    if len(taps) < 3:
        raise RuntimeError(f"need >= 3 taps for SVD, got {len(taps)}")
    src = []  # corner positions in ZED left optical frame
    dst = []  # tip positions in arm base frame
    used = []
    for (cid, tip, R, t) in taps:
        if cid not in corner_pos_zed:
            print(f"   ! corner {cid} not in stereo file; skip")
            continue
        src.append(corner_pos_zed[cid])
        dst.append(tip)
        used.append(cid)
    src = np.asarray(src)
    dst = np.asarray(dst)

    # T_zed_to_arm satisfies T @ corner_in_zed_h ~ tip_in_arm_h (the
    # solver's camera -> robot convention).
    T_zed_to_arm = compute_transform_from_pairs(dst.tolist(), src.tolist())
    residuals = fit_residuals(src, dst, T_zed_to_arm)

    return {
        "T_zed_to_arm": T_zed_to_arm,
        "T_arm_to_zed": np.linalg.inv(T_zed_to_arm),
        "residuals_m": residuals,
        "corners_used": used,
        "src_zed": src.tolist(),
        "dst_arm_base": dst.tolist(),
    }


def _self_check() -> int:
    """Recover a random rigid transform from 5 noiseless point pairs."""
    rng = np.random.default_rng(0)
    R, _ = cv2.Rodrigues(rng.normal(size=(3, 1)))
    T = np.eye(4)
    T[:3, :3] = R
    T[:3, 3] = rng.normal(size=3)
    src = rng.normal(size=(5, 3))
    dst = (T[:3, :3] @ src.T).T + T[:3, 3]
    T_hat = compute_transform_from_pairs(dst.tolist(), src.tolist())
    assert np.allclose(T_hat, T, atol=1e-9), T_hat - T
    assert fit_residuals(src, dst, T_hat).max() < 1e-9
    print("solver self-check ok")
    return 0


def main(argv=None) -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--server", default="http://localhost:8888")
    p.add_argument("--stereo-json", type=Path, default=STEREO_JSON)
    p.add_argument("--corners", default=",".join(str(c) for c in DEFAULT_CORNERS),
                    help="Comma-sep corner IDs to tap, in tap order.")
    p.add_argument("--grip-offset", type=float, default=DEFAULT_GRIP_OFFSET_M,
                    help="Flange-to-jaw-tip offset along flange +Z (m).")
    p.add_argument("--arms", default="left,right",
                    help="Comma-sep arms to calibrate (default both).")
    p.add_argument("--positions", type=int, default=1,
                    help="Multi-position mode: number of NEW board positions "
                         "to tap. Combined with --reuse-existing, the center "
                         "taps from the previous calibration are kept.")
    p.add_argument("--reuse-existing", action="store_true",
                    help="Load taps from the existing calibration_bimanual.json "
                         "as seed data. Only new positions need to be tapped.")
    p.add_argument("--out", type=Path,
                    default=OUT_DIR / "calibration_bimanual.json")
    p.add_argument("--self-check", action="store_true",
                    help="Verify the SVD solver on synthetic data and exit.")
    args = p.parse_args(argv)
    if args.self_check:
        return _self_check()

    corner_ids = [int(c.strip()) for c in args.corners.split(",") if c.strip()]
    arms = [a.strip().lower() for a in args.arms.split(",") if a.strip()]
    for a in arms:
        if a not in ("left", "right"):
            print(f"unknown arm {a!r}", file=sys.stderr); return 2

    # Sanity: server reachable
    try:
        r = requests.get(f"{args.server.rstrip('/')}/api/status", timeout=3)
        r.raise_for_status()
    except Exception as e:
        print(f"ERROR: cannot reach spark_real server at {args.server}: {e}",
              file=sys.stderr)
        print("Start it with:")
        print("  python -m spark_real.server --robot bimanual_franka "
              "--bimanual-variant ANON-LAB --auto-unlock --port 8888")
        return 2

    n_positions = max(1, args.positions)

    all_taps: Dict[str, list] = {arm: [] for arm in arms}
    all_corners_in_zed: Dict[int, np.ndarray] = {}
    seed_count = 0

    if args.reuse_existing:
        existing_cal = OUT_DIR / "calibration_bimanual.json"
        if existing_cal.exists():
            prev = json.loads(existing_cal.read_text())
            prev_stereo_path = Path(prev.get("stereo_source", str(args.stereo_json)))
            if prev_stereo_path.exists():
                prev_corners = load_stereo_corners(prev_stereo_path)
            else:
                prev_corners = {}
            for arm in arms:
                arm_data = prev.get("arms", {}).get(arm, {})
                src_list = arm_data.get("src_zed", [])
                dst_list = arm_data.get("dst_arm_base", [])
                used_ids = arm_data.get("corners_used", [])
                for i, cid in enumerate(used_ids):
                    if i < len(src_list) and i < len(dst_list):
                        seed_id = cid * 1000 + 999
                        src_pt = np.asarray(src_list[i], dtype=np.float64)
                        dst_pt = np.asarray(dst_list[i], dtype=np.float64)
                        all_corners_in_zed[seed_id] = src_pt
                        all_taps[arm].append(
                            (seed_id, dst_pt, np.eye(3), dst_pt))
                        seed_count += 1
            print(f"Loaded {seed_count} existing taps from previous calibration.")
            print(f"  (per arm: {', '.join(f'{a}={len(all_taps[a])}' for a in arms)})")
        else:
            print(f"No existing calibration at {existing_cal}; starting fresh.")

    for pos_idx in range(n_positions):
        if n_positions > 1 or args.reuse_existing:
            pos_label = pos_idx + 1
            if pos_idx == 0 and seed_count == 0:
                print(f"\nMulti-position calibration: {n_positions} board positions.")
                print(f"Place the ChArUco board at position 1 (e.g. table center).")
            elif pos_idx == 0 and seed_count > 0:
                print(f"\nAdding {n_positions} new position(s) to existing calibration.")
                print(f"Place the board at a NEW location (e.g. left side of table).")
            else:
                print(f"\nMove the ChArUco board to position {pos_label}.")
                print(f"Spread positions across the workspace for best accuracy.")
            input(f"Press ENTER when board is placed at position {pos_idx+1}...")

            print(f"\nCapturing stereo corners at position {pos_idx+1}...")
            try:
                _capture_stereo_via_server(args.server, args.stereo_json)
            except Exception as exc:
                print(f"  stereo capture failed: {exc}")
                return 3
            print(f"  stereo capture done.")

        print(f"\nLoading stereo corner measurements from {args.stereo_json}...")
        corners_in_zed = load_stereo_corners(args.stereo_json)
        print(f"  {len(corners_in_zed)} corners available; using {corner_ids}")
        for cid in corner_ids:
            if cid not in corners_in_zed:
                print(f"  ! corner {cid} missing from stereo file "
                      f"(have {sorted(corners_in_zed.keys())})",
                      file=sys.stderr)
                return 3

        if n_positions > 1:
            suffix = f"_pos{pos_idx}"
            pos_corners = {(cid * 1000 + pos_idx): corners_in_zed[cid]
                           for cid in corner_ids}
        else:
            suffix = ""
            pos_corners = {cid: corners_in_zed[cid] for cid in corner_ids}
        all_corners_in_zed.update(pos_corners)

        for arm in arms:
            if n_positions > 1:
                print(f"\nPosition {pos_idx+1}/{n_positions}, arm={arm}")
            taps = collect_taps(args.server, arm, corner_ids, args.grip_offset)
            if n_positions > 1:
                remapped = []
                for (cid, tip, R, t) in taps:
                    remapped.append((cid * 1000 + pos_idx, tip, R, t))
                all_taps[arm].extend(remapped)
            else:
                all_taps[arm].extend(taps)

    results: Dict[str, dict] = {}
    for arm in arms:
        taps = all_taps[arm]
        if len(taps) < 3:
            print(f"\nNot enough taps for arm={arm} ({len(taps)} < 3); aborting.",
                  file=sys.stderr)
            return 4
        r = solve_arm(taps, all_corners_in_zed)
        results[arm] = r
        T = r["T_zed_to_arm"]
        res = r["residuals_m"]
        print()
        print(f"== ARM={arm.upper()} solve ==")
        print(f"   T_zed_to_{arm}_base (mm translation): "
              f"{(T[:3,3]*1000).round(2).tolist()}")
        rvec, _ = cv2.Rodrigues(T[:3, :3])
        ang_deg = float(np.linalg.norm(rvec) * 180 / np.pi)
        print(f"   rotation angle from identity: {ang_deg:.2f} deg")
        print(f"   per-tap residuals (mm): {(res*1000).round(2).tolist()}")
        print(f"   mean / max residual (mm): {res.mean()*1000:.2f} / {res.max()*1000:.2f}")
        if res.mean()*1000 > 10:
            print(f"   WARNING: mean residual > 10 mm; check tap accuracy or grip offset")

    # Derive cross-arm transform
    cross = None
    if "left" in results and "right" in results:
        T_zed_to_L = results["left"]["T_zed_to_arm"]
        T_zed_to_R = results["right"]["T_zed_to_arm"]
        # T_right_to_left: maps a point in RIGHT base frame to LEFT base frame
        # T_right_to_left = T_zed_to_L @ inv(T_zed_to_R)
        T_right_to_left = T_zed_to_L @ np.linalg.inv(T_zed_to_R)
        cross = T_right_to_left
        print()
        print("== CROSS-ARM derived ==")
        print(f"   T_right_to_left translation (mm): "
              f"{(T_right_to_left[:3,3]*1000).round(2).tolist()}")
        rvec, _ = cv2.Rodrigues(T_right_to_left[:3, :3])
        ang_deg = float(np.linalg.norm(rvec) * 180 / np.pi)
        print(f"   rotation angle: {ang_deg:.2f} deg")

    # Persist
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    payload = {
        "ts": time.time(),
        "method": "anchor_tap_stereo_charuco_umeyama",
        "grip_offset_m": args.grip_offset,
        "stereo_source": str(args.stereo_json),
        "n_positions": n_positions,
        "corners_per_position": corner_ids,
        "arms": {},
    }
    for arm, r in results.items():
        payload["arms"][arm] = {
            "T_zed_to_base":    r["T_zed_to_arm"].tolist(),
            "T_base_to_zed":    r["T_arm_to_zed"].tolist(),
            "residuals_m":      r["residuals_m"].tolist(),
            "mean_residual_mm": float(r["residuals_m"].mean() * 1000),
            "max_residual_mm":  float(r["residuals_m"].max() * 1000),
            "corners_used":     r["corners_used"],
            "src_zed":          r["src_zed"],
            "dst_arm_base":     r["dst_arm_base"],
        }
        # Also write per-arm json for downstream consumers expecting
        # base_T_cam_<arm>.json
        per_arm_path = OUT_DIR / f"base_T_zed_{arm}.json"
        per_arm_path.write_text(json.dumps({
            "arm": arm,
            "ts": time.time(),
            "T_base_to_zed":  r["T_arm_to_zed"].tolist(),
            "T_zed_to_base":  r["T_zed_to_arm"].tolist(),
            "mean_residual_mm": float(r["residuals_m"].mean() * 1000),
            "source": "anchor_tap_stereo_charuco",
        }, indent=2))
        print(f"  wrote {per_arm_path}")

    if cross is not None:
        payload["T_right_to_left"] = cross.tolist()
        cross_path = OUT_DIR / "T_right_to_left.json"
        cross_path.write_text(json.dumps({
            "ts": time.time(),
            "T_right_to_left": cross.tolist(),
            "derived_from": "anchor_tap_stereo_charuco",
        }, indent=2))
        print(f"  wrote {cross_path}")

    args.out.write_text(json.dumps(payload, indent=2))
    print(f"\nWrote summary: {args.out}")
    print()
    print("Done. The ChArUco board can be removed.")
    return 0


if __name__ == "__main__":
    sys.exit(main())

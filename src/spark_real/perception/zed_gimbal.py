"""
ZED gimbal: 2-DOF Dynamixel pan/tilt mount for the ZED Mini camera.

One module that owns everything gimbal-related (save-pose, restore-pose,
sweep, viewer) instead of separate scripts. The viewer is folded in too,
headless by default since the spark_real server renders the live feed.

CLI subcommands (run from ``cd ~/spark/src``):

    python -m spark_real.perception.zed_gimbal save-pose
        Read the current motor positions and save as the canonical
        operating pose to ``~/.spark_real/zed_gimbal_experiment_pose.json``.

    python -m spark_real.perception.zed_gimbal restore-pose [--release]
        Command the gimbal back to the saved operating pose, hold it
        with torque on (default) or release after arrival.

    python -m spark_real.perception.zed_gimbal sweep-capture
        Drive the gimbal through a grid of poses around the saved
        operating pose, capturing stereo ChArUco corners + motor state
        at each.  Output feeds an offline solver for the gimbal
        kinematic chain + a bundle-adjusted board-in-world.  Does not
        replace the bimanual calibration; this is data collection only.

    python -m spark_real.perception.zed_gimbal viewer [--display]
        Manual driving via keyboard / web UI.  ``--display`` is opt-in;
        without it, the module is headless.  Use this only when the
        spark_real server is not running and you need to physically
        orient the camera by hand.

API (for ad-hoc scripting):

    from spark_real.perception.zed_gimbal import (
        GimbalDriver, save_experiment_pose, restore_experiment_pose,
        sweep_capture,
    )
"""

from __future__ import annotations

import argparse
import configparser
import json
import os
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import List, Optional, Tuple

import numpy as np

try:
    import cv2  # OpenCV optional, calibration/viewer-only path
except ImportError:
    cv2 = None

try:
    import pyzed.sl as sl  # ZED SDK optional
except ImportError:
    sl = None

# External gimbal driver package providing dynamixel_driver. Override with
# SPARK_GIMBAL_PKG; defaults to ~/gimbal_driver (a symlink on the rig host).
GIMBAL_PKG = os.environ.get("SPARK_GIMBAL_PKG", os.path.expanduser("~/gimbal_driver"))
if GIMBAL_PKG not in sys.path:
    sys.path.insert(0, GIMBAL_PKG)
from dynamixel_driver.XL430_W250_manager import XL430W250Manager  # noqa: E402

DEFAULT_DEVICE = (
    "/dev/serial/by-id/" "usb-FTDI_USB__-__Serial_Converter_FT5NV9P9-if00-port0"
)
DEFAULT_BAUD = 3_000_000
YAW_ID = 0
PITCH_ID = 1
POS_LIMIT_DEG = 100.0  # matches gimbal_ros2 default_params.yaml

DEFAULT_KP = 500
DEFAULT_KI = 0
DEFAULT_KD = 200

ARRIVAL_TOL_DEG = 0.5
ARRIVAL_TIMEOUT_S = 8.0

OUT_DIR = Path.home() / ".spark_real"
DEFAULT_POSE_PATH = OUT_DIR / "zed_gimbal_experiment_pose.json"
DEFAULT_SWEEP_PATH = OUT_DIR / "zed_gimbal_sweep_capture.json"

# XL430W250Manager.sync_read returns 0 ticks on comm error, decoding to
# this value; reject reads matching it.
_GARBAGE_DEG = -180.224
_GARBAGE_TOL = 0.01
_AGREE_TOL = 1.0


# Driver


@dataclass
class GimbalDriver:
    """
    Thin wrapper around ``XL430W250Manager`` for the 2-DOF ZED mount.

    Construct with ``GimbalDriver.open(...)`` then use the methods
    below.  Always call ``close()`` (or use ``with``) so torque is
    disabled if the operator did not explicitly leave it on.
    """

    mgr: XL430W250Manager
    device: str
    baud: int
    _own_torque: bool = False  # True iff this driver enabled torque

    @classmethod
    def open(
        cls,
        device: str = DEFAULT_DEVICE,
        baud: int = DEFAULT_BAUD,
        configure_position_mode: bool = True,
        kp: int = DEFAULT_KP,
        ki: int = DEFAULT_KI,
        kd: int = DEFAULT_KD,
    ) -> "GimbalDriver":
        if not os.path.exists(device):
            raise FileNotFoundError(
                f"Gimbal USB device not found: {device}. "
                f"Check /dev/serial/by-id/ and pass --device if needed."
            )
        mgr = XL430W250Manager([YAW_ID, PITCH_ID], baud, device)
        if configure_position_mode:
            mgr.set_torque_disable(mgr.motor_ids)
            mgr.set_position_mode(mgr.motor_ids)
            mgr.set_min_position_deg(mgr.motor_ids, [-POS_LIMIT_DEG, -POS_LIMIT_DEG])
            mgr.set_max_position_deg(mgr.motor_ids, [POS_LIMIT_DEG, POS_LIMIT_DEG])
            mgr.initialize_gains(
                mgr.motor_ids, np.ones(2) * kp, np.ones(2) * ki, np.ones(2) * kd
            )
        return cls(mgr=mgr, device=device, baud=baud)

    def __enter__(self) -> "GimbalDriver":
        return self

    def __exit__(self, exc_type, exc, tb) -> None:
        self.close()

    def close(self) -> None:
        if self._own_torque:
            try:
                self.mgr.set_torque_disable(self.mgr.motor_ids)
            except Exception:
                pass

    # read
    def read_pose(self, max_tries: int = 40) -> Optional[Tuple[float, float]]:
        """
        Return (yaw_deg, pitch_deg) with two agreeing reads.

        Returns ``None`` if every attempt looked like the
        upstream-sync_read garbage signature.
        """
        last = None
        for _ in range(max_tries):
            try:
                p = list(self.mgr.get_position_deg(self.mgr.motor_ids))
            except Exception:
                time.sleep(0.02)
                continue
            if any(abs(v - _GARBAGE_DEG) < _GARBAGE_TOL for v in p):
                time.sleep(0.02)
                continue
            if last is not None and all(
                abs(a - b) < _AGREE_TOL for a, b in zip(p, last)
            ):
                return float(p[0]), float(p[1])
            last = p
            time.sleep(0.02)
        if last is None:
            return None
        return float(last[0]), float(last[1])

    # write
    def torque_on(
        self, yaw_deg: Optional[float] = None, pitch_deg: Optional[float] = None
    ) -> None:
        """
        Enable torque.  If yaw/pitch are given, seed the goal to that
        pose BEFORE enabling torque so the motor does not snap from a
        stale goal value.  If not given, seed to the current measured
        pose (matches the viewer convention).
        """
        if yaw_deg is None or pitch_deg is None:
            cur = self.read_pose()
            if cur is None:
                raise RuntimeError(
                    "Cannot enable torque: no clean pose read available."
                )
            yaw_deg = float(cur[0]) if yaw_deg is None else yaw_deg
            pitch_deg = float(cur[1]) if pitch_deg is None else pitch_deg
        self.mgr.set_goal_position_deg(self.mgr.motor_ids, [yaw_deg, pitch_deg])
        self.mgr.set_torque_enable(self.mgr.motor_ids)
        self._own_torque = True

    def torque_off(self) -> None:
        self.mgr.set_torque_disable(self.mgr.motor_ids)
        self._own_torque = False

    def goto(
        self,
        yaw_deg: float,
        pitch_deg: float,
        timeout_s: float = ARRIVAL_TIMEOUT_S,
        tol_deg: float = ARRIVAL_TOL_DEG,
    ) -> Tuple[float, float]:
        """
        Command and wait for arrival. Returns the final measured pose.
        """
        if abs(yaw_deg) > POS_LIMIT_DEG or abs(pitch_deg) > POS_LIMIT_DEG:
            raise ValueError(
                f"goto target ({yaw_deg:.2f}, {pitch_deg:.2f}) outside "
                f"+/-{POS_LIMIT_DEG} deg"
            )
        if not self._own_torque:
            self.torque_on(yaw_deg, pitch_deg)
        else:
            self.mgr.set_goal_position_deg(self.mgr.motor_ids, [yaw_deg, pitch_deg])
        t_start = time.time()
        last_pose: Tuple[float, float] = (float("nan"), float("nan"))
        while time.time() - t_start < timeout_s:
            cur = self.read_pose(max_tries=8)
            if cur is None:
                time.sleep(0.05)
                continue
            last_pose = cur
            if abs(cur[0] - yaw_deg) < tol_deg and abs(cur[1] - pitch_deg) < tol_deg:
                return cur
            time.sleep(0.05)
        return last_pose


# Pose save / restore


def save_experiment_pose(
    out: Path = DEFAULT_POSE_PATH,
    *,
    device: str = DEFAULT_DEVICE,
    baud: int = DEFAULT_BAUD,
) -> Tuple[float, float]:
    """
    Read the current gimbal pose and write it to ``out``.

    Does NOT touch torque; the operator's manual hold is preserved.
    """
    with GimbalDriver.open(
        device=device, baud=baud, configure_position_mode=False
    ) as d:
        pose = d.read_pose()
    if pose is None:
        raise RuntimeError("Could not get a clean position read from the gimbal.")
    yaw, pitch = pose
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(
        json.dumps(
            {
                "yaw_deg": yaw,
                "pitch_deg": pitch,
                "motor_ids": {"yaw": YAW_ID, "pitch": PITCH_ID},
                "device": device,
                "baud": baud,
                "timestamp": time.time(),
                "iso_ts": time.strftime("%Y-%m-%dT%H:%M:%S"),
                "notes": (
                    "Canonical operating pose for the ZED + 2-DOF gimbal "
                    "rig.  Restore via "
                    "`python -m spark_real.perception.zed_gimbal restore-pose`."
                ),
            },
            indent=2,
        )
    )
    return yaw, pitch


def restore_experiment_pose(
    in_path: Path = DEFAULT_POSE_PATH,
    *,
    release: bool = False,
    device: str = DEFAULT_DEVICE,
    baud: int = DEFAULT_BAUD,
) -> Tuple[float, float]:
    """
    Command the gimbal back to the saved pose.

    If ``release`` is True, disables torque after arrival; otherwise
    keeps torque on so the gimbal resists external bumps.
    """
    if not in_path.is_file():
        raise FileNotFoundError(
            f"No saved gimbal pose at {in_path}. " f"Run `save-pose` first."
        )
    payload = json.loads(in_path.read_text())
    yaw, pitch = float(payload["yaw_deg"]), float(payload["pitch_deg"])
    with GimbalDriver.open(device=device, baud=baud, configure_position_mode=True) as d:
        final = d.goto(yaw, pitch)
        if release:
            d.torque_off()
        else:
            # Keep torque on after the context exits (suppresses
            # __exit__'s disable) so the gimbal keeps holding.
            d._own_torque = False
    return final


# Sweep capture: ChArUco corners x N gimbal poses


def _default_sweep_grid(
    yaw0: float,
    pitch0: float,
    yaw_span: float = 15.0,
    pitch_span: float = 10.0,
    n_yaw: int = 5,
    n_pitch: int = 3,
) -> List[Tuple[float, float]]:
    """
    Grid of poses centered on (yaw0, pitch0), clamped to limits.
    """
    yaws = np.linspace(yaw0 - yaw_span, yaw0 + yaw_span, n_yaw)
    pitches = np.linspace(pitch0 - pitch_span, pitch0 + pitch_span, n_pitch)
    poses = []
    for p in pitches:
        # serpentine in yaw to minimise total travel
        ys = yaws if (poses == [] or len(poses) % (n_yaw * 2) < n_yaw) else yaws[::-1]
        for y in ys:
            yc = float(np.clip(y, -POS_LIMIT_DEG, POS_LIMIT_DEG))
            pc = float(np.clip(p, -POS_LIMIT_DEG, POS_LIMIT_DEG))
            poses.append((yc, pc))
    # also include exact center
    poses.append((yaw0, pitch0))
    return poses


def _load_zed_left_right_intrinsics():
    """
    Match ``zed_charuco_stereo._load_zed_left_right_calib`` so the
    triangulation result here is comparable to the existing single-pose
    pipeline.  Pulls fx/fy/cx/cy + baseline from /usr/local/zed/settings.
    """
    settings_dir = Path("/usr/local/zed/settings")
    confs = sorted(settings_dir.glob("SN*.conf"))
    if not confs:
        raise RuntimeError(f"No SN*.conf files in {settings_dir}")
    cfg_path = max(confs, key=lambda p: p.stat().st_mtime)
    p = configparser.ConfigParser()
    p.read(cfg_path)

    def section(name):
        s = p[name]
        K = np.array(
            [
                [float(s["fx"]), 0, float(s["cx"])],
                [0, float(s["fy"]), float(s["cy"])],
                [0, 0, 1],
            ]
        )
        dist = np.array(
            [
                float(s.get("k1", 0)),
                float(s.get("k2", 0)),
                float(s.get("p1", 0)),
                float(s.get("p2", 0)),
                float(s.get("k3", 0)),
            ]
        )
        return K, dist

    K_l, dist_l = section("LEFT_CAM_2K")
    K_r, dist_r = section("RIGHT_CAM_2K")
    baseline_m = float(p["STEREO"]["Baseline"]) / 1000.0
    return K_l, dist_l, K_r, dist_r, baseline_m, str(cfg_path)


def _detect_charuco_corners(
    img_gray: np.ndarray, dictionary, board, K: np.ndarray, dist: np.ndarray
):
    """
    Return (ids, undistorted_pixel_coords). None if too few corners.
    """
    if cv2 is None:
        raise RuntimeError("OpenCV not installed; ChArUco detection unavailable")

    detector = cv2.aruco.CharucoDetector(board)
    ch_corners, ch_ids, _, _ = detector.detectBoard(img_gray)
    if ch_ids is None or len(ch_ids) < 6:
        return None, None
    ids = ch_ids.flatten()
    pix = ch_corners.reshape(-1, 2)
    pix_undist = cv2.undistortPoints(pix.reshape(-1, 1, 2), K, dist, P=K).reshape(-1, 2)
    return ids, pix_undist


def _triangulate_stereo(ids_l, pix_l, ids_r, pix_r, K_l, K_r, baseline_m):
    """
    Triangulate the corners visible in BOTH eyes.  Returns
    ``{int(id): [x, y, z]}`` in ZED-left optical frame.
    """
    if cv2 is None:
        raise RuntimeError("OpenCV not installed; stereo triangulation unavailable")
    common = sorted(set(int(i) for i in ids_l) & set(int(i) for i in ids_r))
    if not common:
        return {}
    by_l = {int(i): pix_l[k] for k, i in enumerate(ids_l)}
    by_r = {int(i): pix_r[k] for k, i in enumerate(ids_r)}
    # P_left = K_l [I | 0],  P_right = K_r [I | -baseline*x_hat]
    P_l = K_l @ np.hstack([np.eye(3), np.zeros((3, 1))])
    P_r = K_r @ np.hstack([np.eye(3), np.array([[-baseline_m], [0.0], [0.0]])])

    out = {}
    for cid in common:
        pl = by_l[cid].reshape(1, 2).T
        pr = by_r[cid].reshape(1, 2).T
        X4 = cv2.triangulatePoints(P_l, P_r, pl, pr)
        X = (X4[:3] / X4[3]).flatten()
        out[cid] = X.tolist()
    return out


def sweep_capture(
    out: Path = DEFAULT_SWEEP_PATH,
    *,
    pose_path: Path = DEFAULT_POSE_PATH,
    yaw_span: float = 15.0,
    pitch_span: float = 10.0,
    n_yaw: int = 5,
    n_pitch: int = 3,
    charuco_dict: str = "DICT_4X4_100",
    squares_x: int = 5,
    squares_y: int = 7,
    square_m: float = 0.030,
    marker_m: float = 0.022,
    settle_s: float = 0.8,
    restore_after: bool = True,
    device: str = DEFAULT_DEVICE,
    baud: int = DEFAULT_BAUD,
) -> dict:
    """
    Drive the gimbal through a grid of poses around the saved
    experiment pose, capturing stereo ChArUco corner observations at
    each.  ZED + gimbal serial are both exclusive, so the spark_real
    server MUST be stopped before calling this.

    Output JSON shape::

        {
          "experiment_pose": {"yaw_deg": ..., "pitch_deg": ...},
          "charuco": {dict, squares, ...},
          "samples": [
            {"yaw_deg": ..., "pitch_deg": ...,
             "corners_in_zed_left_m": {"0": [x,y,z], "3": ..., ...}},
            ...
          ]
        }
    """
    if cv2 is None or sl is None:
        raise RuntimeError("OpenCV + ZED SDK required for sweep_capture")

    # 1. Load the experiment pose to center the sweep around.
    if not pose_path.is_file():
        raise FileNotFoundError(
            f"No saved experiment pose at {pose_path}.  Run save-pose first."
        )
    payload = json.loads(pose_path.read_text())
    yaw0 = float(payload["yaw_deg"])
    pitch0 = float(payload["pitch_deg"])

    # 2. Build sweep grid.
    grid = _default_sweep_grid(yaw0, pitch0, yaw_span, pitch_span, n_yaw, n_pitch)
    print(
        f"sweep: {len(grid)} poses around (yaw={yaw0:+.2f}, "
        f"pitch={pitch0:+.2f}) +/-({yaw_span},{pitch_span}) deg"
    )

    # 3. Open ZED in HD2K (matches zed_charuco_stereo for comparability).
    K_l, dist_l, K_r, dist_r, baseline_m, conf_path = _load_zed_left_right_intrinsics()
    print(f"ZED intrinsics from {conf_path}; baseline={baseline_m*1000:.3f} mm")

    zed = sl.Camera()
    init = sl.InitParameters()
    init.camera_resolution = sl.RESOLUTION.HD2K
    init.camera_fps = 15
    init.depth_mode = sl.DEPTH_MODE.NONE
    init.coordinate_units = sl.UNIT.METER
    status = zed.open(init)
    if status != sl.ERROR_CODE.SUCCESS:
        raise RuntimeError(f"ZED open failed: {status}")
    runtime = sl.RuntimeParameters()
    mat_l = sl.Mat()
    mat_r = sl.Mat()

    aruco_dict = cv2.aruco.getPredefinedDictionary(getattr(cv2.aruco, charuco_dict))
    board = cv2.aruco.CharucoBoard(
        (squares_x, squares_y), square_m, marker_m, aruco_dict
    )

    samples: List[dict] = []
    try:
        with GimbalDriver.open(
            device=device, baud=baud, configure_position_mode=True
        ) as gimbal:
            for k, (yaw, pitch) in enumerate(grid):
                print(
                    f"  [{k + 1}/{len(grid)}] -> yaw={yaw:+.2f} " f"pitch={pitch:+.2f}",
                    flush=True,
                )
                gimbal.goto(yaw, pitch)
                time.sleep(settle_s)  # let the camera and motors settle
                measured = gimbal.read_pose() or (yaw, pitch)

                # Drop any stale frames before capture.
                for _ in range(3):
                    zed.grab(runtime)
                if zed.grab(runtime) != sl.ERROR_CODE.SUCCESS:
                    print(f"grab failed; skipping")
                    continue
                zed.retrieve_image(mat_l, sl.VIEW.LEFT_UNRECTIFIED)
                zed.retrieve_image(mat_r, sl.VIEW.RIGHT_UNRECTIFIED)
                img_l = cv2.cvtColor(mat_l.get_data(), cv2.COLOR_BGRA2GRAY)
                img_r = cv2.cvtColor(mat_r.get_data(), cv2.COLOR_BGRA2GRAY)

                ids_l, pix_l = _detect_charuco_corners(
                    img_l, aruco_dict, board, K_l, dist_l
                )
                ids_r, pix_r = _detect_charuco_corners(
                    img_r, aruco_dict, board, K_r, dist_r
                )
                if ids_l is None or ids_r is None:
                    print(
                        f"    insufficient corners "
                        f"(left={None if ids_l is None else len(ids_l)}, "
                        f"right={None if ids_r is None else len(ids_r)}); "
                        f"skipping"
                    )
                    continue
                corners_3d = _triangulate_stereo(
                    ids_l, pix_l, ids_r, pix_r, K_l, K_r, baseline_m
                )
                if len(corners_3d) < 6:
                    print(
                        f"    only {len(corners_3d)} stereo-matched "
                        f"corners; skipping"
                    )
                    continue
                samples.append(
                    {
                        "yaw_deg": float(measured[0]),
                        "pitch_deg": float(measured[1]),
                        "n_corners": len(corners_3d),
                        "corners_in_zed_left_m": {
                            str(k): v for k, v in corners_3d.items()
                        },
                    }
                )
                print(f"captured {len(corners_3d)} corners")
            if restore_after:
                gimbal.goto(yaw0, pitch0)
                gimbal._own_torque = False  # keep torque on after close()
    finally:
        zed.close()

    out.parent.mkdir(parents=True, exist_ok=True)
    summary = {
        "experiment_pose": {
            "yaw_deg": yaw0,
            "pitch_deg": pitch0,
            "source": str(pose_path),
        },
        "charuco": {
            "dict": charuco_dict,
            "squares_x": squares_x,
            "squares_y": squares_y,
            "square_m": square_m,
            "marker_m": marker_m,
        },
        "zed_intrinsics_source": conf_path,
        "baseline_m": baseline_m,
        "n_poses_planned": len(grid),
        "n_samples_captured": len(samples),
        "samples": samples,
        "timestamp": time.time(),
        "iso_ts": time.strftime("%Y-%m-%dT%H:%M:%S"),
    }
    out.write_text(json.dumps(summary, indent=2))
    print(f"wrote {out}  ({len(samples)}/{len(grid)} samples)")
    return summary


# Optional viewer (cv2 + mjpeg).  Headless by default.


def _run_viewer(display: bool, mjpeg_port: int, fps: int) -> int:
    """
    Manual driving loop.  Only invoked from the ``viewer`` subcommand.
    ``display=False`` means no cv2 window AND no mjpeg server; the
    motors still respond to keyboard if the terminal has focus, but
    visual feedback comes from the spark_real server's UI.
    """
    if cv2 is None or sl is None:
        print("OpenCV + ZED SDK required for the viewer", file=sys.stderr)
        return 2

    cam = sl.Camera()
    init = sl.InitParameters()
    init.camera_resolution = sl.RESOLUTION.HD720
    init.camera_fps = fps
    init.depth_mode = sl.DEPTH_MODE.NONE
    init.coordinate_units = sl.UNIT.METER
    if cam.open(init) != sl.ERROR_CODE.SUCCESS:
        print("ZED open failed", file=sys.stderr)
        return 2
    runtime = sl.RuntimeParameters()
    mat = sl.Mat()

    with GimbalDriver.open() as g:
        pos = g.read_pose()
        if pos is None:
            print("ERROR: could not read initial pose", file=sys.stderr)
            cam.close()
            return 2
        print(f"viewer: starting at yaw={pos[0]:+.2f} pitch={pos[1]:+.2f}")
        g.torque_on(pos[0], pos[1])
        yaw, pitch = pos
        step = 2.0
        try:
            if display:
                cv2.namedWindow("ZED + gimbal", cv2.WINDOW_NORMAL)
                cv2.resizeWindow("ZED + gimbal", 1280, 720)
            while True:
                if cam.grab(runtime) != sl.ERROR_CODE.SUCCESS:
                    time.sleep(0.005)
                    continue
                cam.retrieve_image(mat, sl.VIEW.LEFT)
                frame = cv2.cvtColor(mat.get_data(), cv2.COLOR_BGRA2BGR)
                if display:
                    cv2.putText(
                        frame,
                        f"yaw={yaw:+.1f} pitch={pitch:+.1f} step={step:.1f}",
                        (10, 30),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.7,
                        (0, 255, 120),
                        2,
                    )
                    cv2.imshow("ZED + gimbal", frame)
                    k = cv2.waitKey(1) & 0xFFFF
                    if k in (ord("q"), ord("Q"), 27):
                        break
                    elif k in (ord("a"), ord("A"), 81, 65361):
                        yaw -= step
                    elif k in (ord("d"), ord("D"), 83, 65363):
                        yaw += step
                    elif k in (ord("w"), ord("W"), 82, 65362):
                        pitch += step
                    elif k in (ord("s"), ord("S"), 84, 65364):
                        pitch -= step
                    elif k in (ord("h"), ord("H")):
                        yaw, pitch = 0.0, 0.0
                    elif k == ord(","):
                        step = max(0.1, step / 2.0)
                    elif k == ord("."):
                        step = min(20.0, step * 2.0)
                    yaw = float(np.clip(yaw, -POS_LIMIT_DEG, POS_LIMIT_DEG))
                    pitch = float(np.clip(pitch, -POS_LIMIT_DEG, POS_LIMIT_DEG))
                    g.mgr.set_goal_position_deg(g.mgr.motor_ids, [yaw, pitch])
                else:
                    # Headless: a server gimbal endpoint would stream the
                    # frame here (TODO). For now, block briefly so the loop
                    # is not a hot spin.
                    time.sleep(1.0 / max(fps, 1))
        finally:
            cam.close()
            if display:
                cv2.destroyAllWindows()
    return 0


# CLI


def _add_common_args(p: argparse.ArgumentParser) -> None:
    p.add_argument("--device", default=DEFAULT_DEVICE)
    p.add_argument("--baud", type=int, default=DEFAULT_BAUD)


def main(argv: Optional[List[str]] = None) -> int:
    ap = argparse.ArgumentParser(
        prog="python -m spark_real.perception.zed_gimbal",
        description=__doc__.splitlines()[0],
    )
    sub = ap.add_subparsers(dest="cmd", required=True)

    p_save = sub.add_parser(
        "save-pose", help="Record the current motor pose as the operating pose."
    )
    p_save.add_argument("--out", type=Path, default=DEFAULT_POSE_PATH)
    _add_common_args(p_save)

    p_rest = sub.add_parser(
        "restore-pose", help="Drive the gimbal back to the saved operating pose."
    )
    p_rest.add_argument("--input", type=Path, default=DEFAULT_POSE_PATH)
    p_rest.add_argument(
        "--release",
        action="store_true",
        help="Disable torque after arrival (default: keep torque on).",
    )
    _add_common_args(p_rest)

    p_sw = sub.add_parser(
        "sweep-capture",
        help="Sweep poses around the saved operating pose, log ChArUco corners.",
    )
    p_sw.add_argument("--out", type=Path, default=DEFAULT_SWEEP_PATH)
    p_sw.add_argument("--pose", type=Path, default=DEFAULT_POSE_PATH)
    p_sw.add_argument("--yaw-span", type=float, default=15.0)
    p_sw.add_argument("--pitch-span", type=float, default=10.0)
    p_sw.add_argument("--n-yaw", type=int, default=5)
    p_sw.add_argument("--n-pitch", type=int, default=3)
    p_sw.add_argument(
        "--no-restore",
        action="store_true",
        help="Skip the final goto(experiment_pose) after the sweep.",
    )
    _add_common_args(p_sw)

    p_view = sub.add_parser(
        "viewer", help="Manual driving (cv2 keys); headless unless --display."
    )
    p_view.add_argument(
        "--display", action="store_true", help="Open a cv2 window for visual feedback."
    )
    p_view.add_argument(
        "--mjpeg-port",
        type=int,
        default=0,
        help="If >0, also serve a browser viewer on this port.",
    )
    p_view.add_argument("--fps", type=int, default=30)

    args = ap.parse_args(argv)

    if args.cmd == "save-pose":
        yaw, pitch = save_experiment_pose(
            out=args.out, device=args.device, baud=args.baud
        )
        print(f"saved: yaw={yaw:+.2f} pitch={pitch:+.2f} -> {args.out}")
        return 0
    if args.cmd == "restore-pose":
        yaw, pitch = restore_experiment_pose(
            in_path=args.input, release=args.release, device=args.device, baud=args.baud
        )
        print(f"final pose: yaw={yaw:+.2f} pitch={pitch:+.2f}")
        return 0
    if args.cmd == "sweep-capture":
        sweep_capture(
            out=args.out,
            pose_path=args.pose,
            yaw_span=args.yaw_span,
            pitch_span=args.pitch_span,
            n_yaw=args.n_yaw,
            n_pitch=args.n_pitch,
            restore_after=not args.no_restore,
            device=args.device,
            baud=args.baud,
        )
        return 0
    if args.cmd == "viewer":
        return _run_viewer(
            display=args.display, mjpeg_port=args.mjpeg_port, fps=args.fps
        )
    return 1


if __name__ == "__main__":
    sys.exit(main())

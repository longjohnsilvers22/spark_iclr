#!/usr/bin/env python3
"""Live ChArUco corner viewer -- see, in real time, which calibration corners
each camera can detect. Use it to aim/reposition the cameras before calibrating.

Opens the two Azure Kinects (and optionally the wrist RealSense) DIRECTLY via
pyk4a / pyrealsense2 -- no spark server needed. Each camera shows green dots on
every detected ChArUco corner plus an "N/24" count. Aim for >=10 per camera you
intend to calibrate.

  Live window (default):
    cd ~/spark/src && python ../scripts/view_charuco.py
    # add the wrist RealSense (works out of the box: librealsense is the
    # RSUSB build in /usr/local, no uvcvideo/kernel driver needed):
    python ../scripts/view_charuco.py --wrist
    # press 'q' (or Esc) in the window to quit

  One-shot snapshot to a file instead of a window (headless / over SSH):
    python ../scripts/view_charuco.py --save /tmp/charuco_view.jpg

NOTE: cameras are exclusive -- stop the spark server first (it owns the Kinects).
"""
import argparse
import sys
from pathlib import Path

import cv2
import numpy as np
import pyk4a

# The wrist RealSense is optional (librealsense RSUSB build); the Kinect-only
# path must keep working when pyrealsense2 is not installed.
try:
    import pyrealsense2 as rs
except ImportError:
    rs = None

_SCRIPTS = Path(__file__).resolve().parent
sys.path.insert(0, str(_SCRIPTS))
import spark_calibrate as sc  # noqa: E402

# Must match the roster in configs/ur10e_default.yaml: 627 is the ELEVATED unit
# over the robot mount (birdview), 626 is the across-the-table unit (sideview).
# An inverted roster would flip camera_0/camera_1 in every recorded episode.
KINECT_ROLES = {"000000000000": "birdview", "000000000000": "sideview"}


def annotate(rgb, board, dictionary, role):
    """Return a BGR image with corners drawn + a count banner."""
    ids, px = sc.detect_charuco_corners(rgb, board, dictionary)
    n = 0 if ids is None else len(ids)
    total = len(board.getChessboardCorners())
    bgr = cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR)
    if ids is not None:
        for cid, (u, v) in zip(ids, px):
            cv2.circle(bgr, (int(u), int(v)), 6, (0, 255, 0), -1)
            cv2.putText(bgr, str(int(cid)), (int(u) + 6, int(v) - 6),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 200, 255), 1)
    color = (0, 255, 0) if n >= 10 else (0, 165, 255) if n >= 6 else (0, 0, 255)
    cv2.rectangle(bgr, (0, 0), (bgr.shape[1], 46), (0, 0, 0), -1)
    cv2.putText(bgr, f"{role}: {n}/{total} corners", (12, 33),
                cv2.FONT_HERSHEY_SIMPLEX, 1.0, color, 2)
    return bgr


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--wrist", action="store_true",
                    help="also show the wrist RealSense")
    ap.add_argument("--save", metavar="PATH", default=None,
                    help="one-shot: write a single annotated tile here and exit "
                         "(no live window)")
    ap.add_argument("--width", type=int, default=640, help="display width per cam")
    args = ap.parse_args()

    board, dictionary, meta = sc.load_board(sc.DEFAULT_BOARD)
    print(f"board: {meta['squares_x']}x{meta['squares_y']} "
          f"{meta['dictionary']} ({len(board.getChessboardCorners())} corners)")

    kinects = []
    for i in range(pyk4a.connected_device_count()):
        k = pyk4a.PyK4A(pyk4a.Config(
            color_resolution=pyk4a.ColorResolution.RES_1080P,
            depth_mode=pyk4a.DepthMode.OFF, camera_fps=pyk4a.FPS.FPS_15,
            synchronized_images_only=False), device_id=i)
        k.start()
        role = KINECT_ROLES.get(k.serial, f"kinect{i}")
        kinects.append((role, k))
        print(f"  opened {role} ({k.serial})")

    rs_pipe = None
    if args.wrist:
        try:
            rs_pipe = rs.pipeline()
            cfg = rs.config()
            cfg.enable_stream(rs.stream.color, 640, 480, rs.format.rgb8, 30)
            rs_pipe.start(cfg)
            print("  opened wrist (RealSense)")
        except Exception as e:
            print(f"  wrist unavailable: {e} (is another process using it?)")
            rs_pipe = None

    def grab_tiles():
        tiles = []
        for role, k in kinects:
            cap = k.get_capture()
            if cap.color is None:
                continue
            rgb = cv2.cvtColor(cap.color[:, :, :3], cv2.COLOR_BGR2RGB)
            bgr = annotate(rgb, board, dictionary, role)
            h = int(bgr.shape[0] * args.width / bgr.shape[1])
            tiles.append(cv2.resize(bgr, (args.width, h)))
        if rs_pipe is not None:
            try:
                frames = rs_pipe.wait_for_frames(1000)
                cf = frames.get_color_frame()
                if cf:
                    rgb = np.asanyarray(cf.get_data())
                    bgr = annotate(rgb, board, dictionary, "wrist")
                    h = int(bgr.shape[0] * args.width / bgr.shape[1])
                    tiles.append(cv2.resize(bgr, (args.width, h)))
            except Exception:
                pass
        if not tiles:
            return None
        h = max(t.shape[0] for t in tiles)
        tiles = [cv2.copyMakeBorder(t, 0, h - t.shape[0], 0, 0,
                                    cv2.BORDER_CONSTANT, value=(0, 0, 0)) for t in tiles]
        return cv2.hconcat(tiles)

    try:
        if args.save:
            # warm up auto-exposure, then one annotated tile to disk
            tile = None
            for _ in range(15):
                tile = grab_tiles()
            if tile is not None:
                cv2.imwrite(args.save, tile)
                print(f"saved -> {args.save}")
            return
        print("live view: press 'q' or Esc to quit")
        while True:
            tile = grab_tiles()
            if tile is not None:
                cv2.imshow("ChArUco corner check (q to quit)", tile)
            if cv2.waitKey(1) & 0xFF in (ord("q"), 27):
                break
    finally:
        for _, k in kinects:
            k.stop()
        if rs_pipe is not None:
            rs_pipe.stop()
        cv2.destroyAllWindows()


if __name__ == "__main__":
    main()

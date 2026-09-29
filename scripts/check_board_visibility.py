"""ChArUco board visibility check for the calibration cameras.

One-shot (default): grab a frame per camera, print marker/corner counts,
save annotated PNGs to output/calibration_board/.

Live (--live): continuous annotated view in a window while you move the
board -- counts overlay updates each cycle (~1-2 Hz, limited by the
server capture endpoint). Keys: q/ESC quit, s save current annotated
frames as the PNGs.

Needs the spark server running (frames come from /api/calibrate/capture_one).

Usage:
    python check_board_visibility.py            # one-shot
    python check_board_visibility.py --live     # live positioning view
"""
import os
import sys
from pathlib import Path

import cv2
import numpy as np
from cv2 import aruco

# Anchor everything to this file's location so the script runs from any CWD.
SCRIPT_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(SCRIPT_DIR))
import spark_calibrate as sc  # noqa: E402

os.environ.setdefault("DISPLAY", ":1")
os.environ.setdefault("XAUTHORITY", "/run/user/1000/gdm/Xauthority")

CAMS = ("birdview", "sideview")
board, dictionary, meta = sc.load_board(str(SCRIPT_DIR / "charuco_5x7_letter.json"))
N_TOTAL = board.getChessboardCorners().shape[0]
OUT = str(SCRIPT_DIR.parent / "output" / "calibration_board")
os.makedirs(OUT, exist_ok=True)
det = aruco.ArucoDetector(dictionary, aruco.DetectorParameters())


def annotated_frame(cam):
    """Fetch one frame and annotate detections. Returns (img_bgr, n_markers,
    n_corners) or (placeholder, -1, -1) on fetch failure."""
    try:
        rgb, intr, depth = sc.fetch_frame(cam)
    except Exception as e:  # noqa: BLE001
        ph = np.zeros((540, 960, 3), dtype=np.uint8)
        cv2.putText(ph, f"{cam}: fetch FAILED", (20, 44),
                    cv2.FONT_HERSHEY_SIMPLEX, 1.1, (0, 0, 255), 3)
        cv2.putText(ph, str(e)[:80], (20, 90),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 1)
        return ph, -1, -1
    img = np.ascontiguousarray(rgb[..., ::-1].copy())
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    mcorners, mids, _ = det.detectMarkers(gray)
    n_markers = 0 if mids is None else len(mids)
    if n_markers:
        aruco.drawDetectedMarkers(img, mcorners, mids)
    ids, pix = sc.detect_charuco_corners(rgb, board, dictionary)
    n_corners = 0 if ids is None else len(ids)
    if n_corners:
        for (x, y) in pix:
            cv2.circle(img, (int(x), int(y)), 12, (0, 255, 0), 3)
        xs, ys = pix[:, 0], pix[:, 1]
        cv2.rectangle(img, (int(xs.min()) - 15, int(ys.min()) - 15),
                      (int(xs.max()) + 15, int(ys.max()) + 15),
                      (0, 200, 255), 3)
    h, w = img.shape[:2]
    ok = n_corners >= 12
    color = (0, 255, 0) if ok else (0, 255, 255)
    cv2.putText(img,
                f"{cam}: {n_markers} markers, {n_corners}/{N_TOTAL} corners"
                f"{' OK' if ok else ''} ({w}x{h})",
                (20, 44), cv2.FONT_HERSHEY_SIMPLEX, 1.1, color, 3)
    return img, n_markers, n_corners


def save_pngs(frames):
    for cam, img in frames.items():
        p = f"{OUT}/view_{cam}.png"
        cv2.imwrite(p, img)
        print(f"  saved {p}")


def one_shot():
    frames = {}
    for cam in CAMS:
        img, n_markers, n_corners = annotated_frame(cam)
        frames[cam] = img
        if n_markers < 0:
            print(f"{cam}: fetch FAILED (server up? camera online?)")
        else:
            print(f"{cam}: {n_markers} markers, {n_corners}/{N_TOTAL} corners"
                  f" -> {OUT}/view_{cam}.png")
    save_pngs(frames)


def live():
    win = "board visibility (q quit, s save PNGs)"
    cv2.namedWindow(win, cv2.WINDOW_NORMAL)
    cv2.resizeWindow(win, 1760, 500)
    print("live view: move the board until BOTH cameras read >=12 corners "
          "(green). q/ESC to quit, s to save PNGs.")
    while True:
        frames, tiles, status = {}, [], []
        for cam in CAMS:
            img, n_markers, n_corners = annotated_frame(cam)
            frames[cam] = img
            h, w = img.shape[:2]
            scale = 480.0 / h
            tiles.append(cv2.resize(img, (int(w * scale), 480)))
            status.append(f"{cam} {max(n_corners, 0)}/{N_TOTAL}")
        cv2.imshow(win, np.hstack(tiles))
        print("\r  " + "   ".join(status) + "        ", end="", flush=True)
        key = cv2.waitKey(50) & 0xFF
        if key in (ord("q"), 27):
            break
        if key == ord("s"):
            print()
            save_pngs(frames)
    print()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    if "--live" in sys.argv:
        live()
    else:
        one_shot()

#!/usr/bin/env python3
"""Regenerate the SPARK ChArUco calibration board, print-ready for US Letter.

Produces a PNG + PDF sized so that printing at 100% / "actual size" (NO "fit to
page" scaling) yields a board with the NOMINAL square size below. After
printing, MEASURE the real square size with a ruler and update
scripts/charuco_5x7_30mm_4x4_100.json's square_length_m (and marker_length_m
proportionally) -- the calibration math uses the physical millimeters, so this
measure-after-print step is what keeps scale correct.

Matches the existing board's dictionary (DICT_4X4_100), 5x7 layout, and
marker/square ratio so the detector in spark_calibrate.py / calib_app.py works
unchanged.

Usage:
    python scripts/generate_charuco_board.py            # 34.0mm squares, Letter
    python scripts/generate_charuco_board.py --square-mm 34.0 --dpi 300
"""
import argparse
from pathlib import Path

import cv2
import numpy as np
from cv2 import aruco
from PIL import Image, ImageDraw, ImageFont

# Match the existing board spec (scripts/charuco_5x7_30mm_4x4_100.json).
SQUARES_X, SQUARES_Y = 5, 7
DICT_NAME = "DICT_4X4_100"
MARKER_RATIO = 0.025116 / 0.03425  # marker_len / square_len from the existing board spec


def _font(px):
    for p in ("/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf",
              "/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf"):
        if Path(p).exists():
            return ImageFont.truetype(p, px)
    return ImageFont.load_default()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--square-mm", type=float, default=34.0,
                    help="nominal square size in mm (default 34.0)")
    ap.add_argument("--dpi", type=int, default=300)
    ap.add_argument("--out-dir", default=str(Path(__file__).resolve().parent.parent
                                             / "output" / "calibration_board"))
    ap.add_argument("--annotate", action="store_true",
                    help="add title + corner crop marks + 50mm scale bar (default: "
                         "clean board only, nothing but the markers)")
    args = ap.parse_args()

    mm_to_px = args.dpi / 25.4
    sq_px = int(round(args.square_mm * mm_to_px))
    square_m = args.square_mm / 1000.0
    marker_m = square_m * MARKER_RATIO

    dictionary = aruco.getPredefinedDictionary(getattr(aruco, DICT_NAME))
    board = aruco.CharucoBoard((SQUARES_X, SQUARES_Y), square_m, marker_m, dictionary)
    board_px = board.generateImage(
        (SQUARES_X * sq_px, SQUARES_Y * sq_px), marginSize=0, borderBits=1
    )  # grayscale uint8

    # Letter canvas (portrait) at the chosen DPI.
    W = int(round(8.5 * args.dpi))
    H = int(round(11.0 * args.dpi))
    canvas = Image.new("RGB", (W, H), "white")
    bw, bh = board_px.shape[1], board_px.shape[0]
    ox, oy = (W - bw) // 2, int(0.55 * args.dpi)  # centered X, top margin for title
    canvas.paste(Image.fromarray(board_px).convert("RGB"), (ox, oy))

    # Default: CLEAN board -- nothing on the page but the markers + white quiet
    # zone (the centering margin). --annotate adds print-helper marks/text.
    if args.annotate:
        d = ImageDraw.Draw(canvas)
        t = max(2, args.dpi // 100)
        L = int(0.25 * args.dpi)
        for (cx, cy) in [(ox, oy), (ox + bw, oy), (ox, oy + bh), (ox + bw, oy + bh)]:
            d.line([(cx - L, cy), (cx + L, cy)], fill="black", width=t)
            d.line([(cx, cy - L), (cx, cy + L)], fill="black", width=t)
        f_big, f_sm = _font(int(0.16 * args.dpi)), _font(int(0.11 * args.dpi))
        d.text((ox, int(0.30 * args.dpi)),
               f"ChArUco 5x7  {DICT_NAME}  nominal {args.square_mm:.1f} mm/square",
               fill="black", font=f_big)
        bar_px = int(round(50.0 * mm_to_px))
        by = oy + bh + int(0.22 * args.dpi)
        bx = (W - bar_px) // 2
        d.line([(bx, by), (bx + bar_px, by)], fill="black", width=max(3, args.dpi // 75))
        for ex in (bx, bx + bar_px):
            d.line([(ex, by - int(0.07 * args.dpi)), (ex, by + int(0.07 * args.dpi))],
                   fill="black", width=max(3, args.dpi // 75))
        d.text((ox, by + int(0.10 * args.dpi)),
               "^ this bar must measure 50 mm.  PRINT AT 100% / ACTUAL SIZE (no 'fit to page').",
               fill="black", font=f_sm)
        d.text((ox, by + int(0.27 * args.dpi)),
               "Then measure across 4 squares; set square_length_m = (measured_mm/4)/1000.",
               fill="black", font=f_sm)

    out = Path(args.out_dir)
    out.mkdir(parents=True, exist_ok=True)
    png = out / "charuco_5x7_letter.png"
    pdf = out / "charuco_5x7_letter.pdf"
    canvas.save(png, dpi=(args.dpi, args.dpi))
    canvas.save(pdf, "PDF", resolution=float(args.dpi))

    board_w_mm, board_h_mm = SQUARES_X * args.square_mm, SQUARES_Y * args.square_mm
    print(f"nominal square : {args.square_mm:.2f} mm  (marker {marker_m*1000:.2f} mm)")
    print(f"board bbox     : {board_w_mm:.0f} x {board_h_mm:.0f} mm "
          f"({board_w_mm/25.4:.1f} x {board_h_mm/25.4:.1f} in)")
    print(f"page           : US Letter 8.5 x 11 in @ {args.dpi} DPI")
    print(f"PNG -> {png}")
    print(f"PDF -> {pdf}  (print this at 100% / actual size)")


if __name__ == "__main__":
    main()

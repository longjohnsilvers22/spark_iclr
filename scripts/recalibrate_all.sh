#!/usr/bin/env bash
# Recalibrate all three SPARK cameras in one session, anchored to a single
# board placement, depth-supervised (RGB-D Procrustes) with the multi-height
# affine depth correction so Z stays accurate across the workspace (not just at
# the board height). Reuses ~/Calib3R-unified/spark_calibrate.py.
#
# Requires: robot powered, SPARK server running (this drives the arm and
# captures RGB+depth over the server API), the ChArUco board, and 1-2 known
# risers (e.g. 50 mm and 100 mm blocks) for the static-camera heights.
#
# The Kinects (bird/side) get the board at multiple heights to fit the depth
# OFFSET; the wrist already drives poses that span height, so it needs no risers.
# Keep the board in the SAME x,y for birdview and sideview so they share one
# anchor (that is what makes them agree); only raise it in Z for the heights.
set -e

PY=${PY:-"$HOME/miniconda3/envs/calib_handeye/bin/python"}
CAL="$(cd "$(dirname "$0")" && pwd)/spark_calibrate.py"
BOARD=${BOARD:-"$(cd "$(dirname "$0")" && pwd)/charuco_5x7_30mm_4x4_100.json"}
HEIGHTS=${HEIGHTS:-0 0.05 0.10}     # riser heights in meters for the Kinects
ANCHOR=/tmp/board_pose.json
NPOSES=${NPOSES:-10}

echo "board=$BOARD"
echo "static-camera heights (m)=$HEIGHTS"
echo

echo "[1/3] Anchor the board to the robot base frame (gripper touches corners)"
$PY "$CAL" --mode anchor_board --board "$BOARD" --out "$ANCHOR"

echo "[2/3] Birdview + sideview, one pass, multi-height affine"
echo "      (board captured by BOTH Kinects at each height; one re-placement"
echo "       per height instead of one per camera, so keep the board's x,y put)"
$PY "$CAL" --mode calibrate_static --cameras birdview sideview \
    --board "$BOARD" --board-pose "$ANCHOR" --num-poses "$NPOSES" --heights $HEIGHTS

echo "[3/3] Wrist, auto-driven poses already span height"
$PY "$CAL" --mode calibrate_wrist \
    --board "$BOARD" --board-pose "$ANCHOR" --num-poses 19

echo
echo "Done. For each camera check both numbers in the output:"
echo "  depth_scale_correction  (multiplicative bias)"
echo "  depth_offset_correction (additive bias; nonzero means single-height"
echo "                           scale-only cal was leaving Z off by this much)"
echo
echo "The new cals are in ~/Calib3R-unified/calibrations/ as calib_*.json."
echo "Install them by copying into spark_real/output/calibrations/ as"
echo "handeye_birdview.json / handeye_sideview.json / handeye_wrist.json,"
echo "then restart the server (it reads depth_offset_correction and applies"
echo "z*scale+offset)."
echo
echo "Then verify Z independently against gripper-touch ground truth:"
echo "  $PY $CAL --mode validate_z --cameras birdview sideview"
echo "  (hand-guide the tip onto corners at several heights; watch mean|dz|)"

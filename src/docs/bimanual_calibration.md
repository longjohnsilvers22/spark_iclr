# Bimanual SPARK Calibration

The bimanual rig calibrates each arm base to the static ZED Mini and
derives the cross-arm rigid transform `T_right_to_left` from the two
results. The method is anchor-tap: the operator drives each arm so the
closed jaw tip touches known ChArUco inner corners whose 3D positions
were measured by ZED stereo triangulation. A per-arm SVD (Umeyama) over
the tap pairs gives `T_zed_to_<arm>_base`, and

```
T_right_to_left = T_zed_to_left_base @ inv(T_zed_to_right_base)
```

Two modules under `spark_real.calibration` produce the files that the
rest of the pipeline reads (`skills/bimanual_cloth.py`, `src/scripts/detect.py`,
`control/pyroki_planner.py`).

## Prerequisites

* ChArUco board: `DICT_4X4_100`, 5x7 squares, 30 mm square, 22 mm marker
  (`scripts/charuco_5x7_30mm_4x4_100.json`). Place it flat on the table
  inside both arms' reach and the ZED field of view.
* `grippers.jaw_offset_m` in the machine overlay (flange to closed jaw
  tip). The tap script uses the same value as `--grip-offset`.

## 1. Measure the board corners with ZED stereo

The ZED SDK is exclusive, so stop the server for this step.

```bash
cd ~/spark/src
pkill -f spark_real.server
python -m spark_real.calibration.bimanual_charuco_stereo [--n-frames 5]
```

Writes `~/.spark_real/ba_frames/zed_charuco_stereo.json` with every
shared inner corner in ZED-left optical frame, averaged over `--n-frames`,
plus per-frame reprojection error and annotated left/right PNGs.
Corner ids follow the 4 x 6 inner grid:

```
+3  +7  +11  +15  +19  +23   <- far row
+2  +6  +10  +14  +18  +22
+1  +5  +9   +13  +17  +21
+0  +4  +8   +12  +16  +20   <- near row
```

## 2. Tap the corners with each arm

Start the server (the tap script reads TCPs over HTTP, so there is no
FCI conflict), then run the tap procedure:

```bash
python -m spark_real.server --robot bimanual_franka --machine ANON-LAB --auto-unlock --port 8888
python -m spark_real.calibration.bimanual_anchor_tap \
    [--corners 0,3,20,23,11] [--grip-offset 0.1275] [--arms left,right]
```

For each arm the prompt names a corner id. Pilot-drive the closed jaw
tip onto that corner, release the pilot button, press ENTER. `s` skips a
tap, `q` aborts. At least 3 taps per arm are needed; the default 5 are
spread across the board for conditioning.

Options:

* `--positions N --reuse-existing`: tap the board at N new locations and
  merge with the taps stored in the previous calibration. In this mode
  the script grabs stereo frames itself through `GET /api/capture/stereo`
  on the running server, so step 1 is not repeated by hand.

The script prints per-arm residuals. Mean residual above 10 mm means a
mistapped corner or a wrong grip offset; re-tap.

## Output files

All under `~/.spark_real/`:

| File | Content | Readers |
|---|---|---|
| `calibration_bimanual.json` | `arms.<arm>.T_zed_to_base`, `T_base_to_zed`, residuals, taps, `grip_offset_m`, `T_right_to_left` | `skills/bimanual_cloth.py`, `src/scripts/detect.py` |
| `T_right_to_left.json` | `T_right_to_left` 4x4 | `control/pyroki_planner.py` (`BimanualPyrokiPlanner.from_default`) |
| `base_T_zed_<arm>.json` | per-arm `T_base_to_zed` / `T_zed_to_base` | downstream per-arm consumers |

Missing calibration fails closed: the readers raise instead of falling
back to identity.

## Stereo capture route

`GET /api/capture/stereo` (routes/streaming.py) returns the ZED left and
right frames as base64 PNG plus SDK intrinsics (`fx_l`, `fy_l`, `cx_l`,
`cy_l`, `fx_r`, ..., `baseline_m`). It only answers on the bimanual
pipeline whose `external` camera exposes `read_stereo`; single-arm
pipelines get 400.

## When to recalibrate

* After moving either arm's base bolts or the ZED tripod.
* After changing the jaws (re-measure `jaw_offset_m` first).
* If a point detected in the ZED lands more than 5 mm apart between the
  two arm frames.

Single-arm calibration (`/api/anchor_point`, hand-eye) is unchanged by
the bimanual code path.

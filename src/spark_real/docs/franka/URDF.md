# FR3 URDF + Meshes (SPARK)

This document describes how the Franka Research 3 (FR3) URDF + mesh assets
are organized for SPARK.

## TL;DR

| What | Where | Notes |
|---|---|---|
| Default URDF (FR3 + Franka Hand) | `src/spark_real/robots/franka/urdf/fr3_franka_hand.urdf` | absolute mesh paths, loadable without ROS2 |
| Bare-arm URDF (no end-effector) | `src/spark_real/robots/franka/urdf/fr3_no_hand.urdf` | use when a non-Franka gripper is fitted |
| SRDF (semantic groups, default poses) | `src/spark_real/robots/franka/urdf/fr3_franka_hand.srdf` | needed by MoveIt2 |
| Mesh files (`.dae` visual, `.stl` collision) | `external/franka_description/meshes/robots/fr3/{visual,collision}/` | referenced by absolute path from the URDFs |
| Franka Hand meshes | `external/franka_description/meshes/robot_ee/franka_hand_black/` | included via `ee_id:=franka_hand` |
| Upstream xacro sources | `external/franka_description/robots/{fr3,common}/`, `end_effectors/` | regenerate URDF from these |
| Regenerate script | `src/spark_real/robots/franka/urdf/regenerate_fr3_urdf.sh` | re-run if upstream changes |

Pointed at by `configs/franka_default.yaml` under the `robot.urdf` block
(`path`, `srdf_path`, `bare_arm_path`, `base_link`, `tip_link`, `joint_names`).

## Provenance

- **Upstream:** [`frankaemika/franka_description`](https://github.com/frankaemika/franka_description), Apache-2.0.
- **Clone type:** VCS-imported (not vendored, not a submodule). Registered in `~/spark/spark_real.repos` so a fresh clone of SPARK + `vcs import` reproduces it.
- **Local path:** `~/spark/external/franka_description`
- **Pinned to:** `main` (commit `2fd6aaa` at time of import, package version `2.7.1`).
- **Clone depth:** `--depth 1` (only what's needed to regenerate URDFs).

The upstream package ships *xacro* macros only; no plain URDF file. The
`scripts/create_urdf.py` helper inside the upstream repo expands them into a
flat URDF, and the spark_real version of that flow is captured in
`urdf/regenerate_fr3_urdf.sh`. The expanded URDFs are checked in (the spark_real
package needs to be usable without re-running xacro every boot).

## Why two URDFs

| File | Use case |
|---|---|
| `fr3_franka_hand.urdf` | Stock setup: FR3 arm + Franka Hand parallel jaw gripper. Matches the YAML `gripper.type: "franka_hand"`. This is the default consumers should load. |
| `fr3_no_hand.urdf` | Bare arm. Use when a Robotiq 2F-85 or other gripper is fitted via a flange adapter; an SRDF / wrapper xacro then attaches the gripper URDF at `fr3_hand_joint`. |

The two share the same 7 revolute joints (`fr3_joint1`..`fr3_joint7`) and
same arm link tree (`fr3_link0`..`fr3_link7`, plus the fixed `fr3_joint8`
flange). Only the end-effector subtree differs.

## Why absolute mesh paths

`urdf_parser_py`, `pinocchio`, `trimesh`, and `kdl_parser` all resolve mesh
filenames literally - they don't know about ROS2's `package://` URI scheme
unless they're invoked from a sourced ament workspace. The URDFs in this
directory have been post-processed (via `scripts/create_urdf.py --abs-path`
or `sed`) to use absolute paths into `external/franka_description/meshes`,
so they load identically from any Python tool in the `spark_conda` env
without sourcing ROS2.

**Trade-off:** the URDFs are *not portable*. If `external/franka_description`
moves, the mesh refs break. For a MoveIt2 deployment that runs inside ROS2
(which can resolve `package://`), use the URDFs under
`external/franka_description/urdfs/` instead - those are generated from the
same xacro but keep `package://franka_description/...` filenames.

## What's vendored vs. what's not

| Item | Disposition | Why |
|---|---|---|
| URDFs (`.urdf`) | **vendored** (checked in) | small (~20 KB each), avoid re-running xacro on every checkout, lets `urdf_parser_py.URDF.from_xml_file(...)` work in CI without ament |
| SRDF (`.srdf`) | **vendored** | same reason; required by MoveIt2 |
| Meshes (`.dae`, `.stl`, `.obj`) | **NOT vendored**, lives in `external/franka_description` | ~30 MB; pulling them through VCS import is faster than `git lfs` |
| Upstream xacro source | **VCS-imported** | regenerate URDFs after upgrading franka_description; see `regenerate_fr3_urdf.sh` |

## Joint list (URDF order)

Output of `urdf_parser_py.URDF.from_xml_file(fr3_franka_hand.urdf)`:

| # | name | range (rad) | v_max (rad/s) | effort (Nm) |
|--:|---|---|--:|--:|
| 1 | `fr3_joint1` | [-2.9007, +2.9007] | 2.62 | 87 |
| 2 | `fr3_joint2` | [-1.8361, +1.8361] | 2.62 | 87 |
| 3 | `fr3_joint3` | [-2.9007, +2.9007] | 2.62 | 87 |
| 4 | `fr3_joint4` | [-3.0770, -0.1169] | 2.62 | 87 |
| 5 | `fr3_joint5` | [-2.8763, +2.8763] | 5.26 | 12 |
| 6 | `fr3_joint6` | [+0.4398, +4.6216] | 4.18 | 12 |
| 7 | `fr3_joint7` | [-3.0508, +3.0508] | 5.26 | 12 |

Matches the official FR3 datasheet limits. Note `fr3_joint4` is bilaterally
bounded *below zero* (elbow can't fully straighten), and `fr3_joint6` is
shifted into a positive range - these are FR3 specifics, not Panda.

With the Franka Hand attached there are two prismatic finger joints
(`fr3_finger_joint1`, `fr3_finger_joint2`) mirrored 0..40 mm.

## Regenerating after upstream changes

```bash
# Inside ~/spark/external, refresh the clone:
cd ~/spark/external/franka_description && git pull

# Re-expand the xacros (requires ROS2 Jazzy xacro on PATH):
conda activate spark_conda     # also sources Jazzy
bash ~/spark/src/spark_real/robots/franka/urdf/regenerate_fr3_urdf.sh
```

## MoveIt2 / ROS2 use

The spark_real URDFs are designed for Python-side (pinocchio, IK, perception
TF) use. For MoveIt2's collision-aware planner inside ROS2 Jazzy:

1. Install the `franka_description` package from the clone, e.g.
   `cd ~/spark && colcon build --packages-select franka_description` after
   symlinking the clone into `src/franka_description`, **or** install
   `ros-jazzy-franka-description` from apt if Franka has shipped a Jazzy
   release (last checked May 2026: no `ros-jazzy-franka-description` deb,
   so build from source).
2. Point MoveIt at the *package://* URDF inside
   `external/franka_description/urdfs/fr3_franka_hand.urdf`, *not* the
   absolute-path copy in `src/spark_real/robots/franka/urdf/`.

The collision pairs in `fr3_franka_hand.srdf` cover the stock self-collision
matrix; if the cell adds a table or fixture, append those to a wrapper SRDF.

## Caveats

- **No Pinocchio in `spark_conda` (yet).** As of 2026-05 the env has
  `urdf_parser_py` (loadable in Python 3.12) but not `pinocchio`. For IK
  the `franky` package wraps libfranka's *internal* kinematics, so this
  isn't blocking control. Add `pinocchio` only if a SPARK module needs
  jacobians or frame transforms outside of franky.
- **Urdf parser warnings.** Franka's URDFs include extension attributes
  (`gear_ratio`, `motor_inertia`, `<position_based_velocity_limits>`,
  named `<visual name="...">`) that ROS's strict `urdf_parser_py` doesn't
  know about. These show up as `Unknown attribute / Unknown tag` warnings
  on import but are **non-fatal** - the kinematic chain is parsed correctly.
- **Hand vs. flange convention.** The default URDF's tip is `fr3_hand_tcp`,
  matching libfranka's `EE_T_K` (10.34 cm past the flange). If you bypass
  the hand (custom EE), set `tip_link: "fr3_link8"` in `franka_default.yaml`
  and load `fr3_no_hand.urdf` instead.
- **FER vs. FR3.** The lab has FR3 (confirmed 2026-05-11). The upstream
  repo also ships `fer/fer.urdf.xacro` (the original Panda) and could be
  swapped if needed via the same regenerate script (`xacro robots/fer/...`).

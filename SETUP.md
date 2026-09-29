# SPARK Setup

SPARK does not vendor its dependencies. Cloning this repository gives you
the SPARK code only; external packages and simulation assets are installed
separately by the steps below. The only git submodule is EquiGraspFlow
(section 2).

## 1. Python environment

```bash
conda create -n spark python=3.12 && conda activate spark
# torch/torchvision first, from the CUDA wheel index (see src/setup_spark_conda.bash)
pip install -e ".[planning,perception,sim,ik]"   # add a robot extra, e.g. ur10e
```

The `ik` extra installs pyroki and jaxls (pinned git commits) for the
joint-space IK descent; without it the UR10e executor silently falls back to
the Cartesian servo and the CaP-Bench Wipe task drops from 60 to about 5.
The pinned `environment.yml` is the same set for conda.

## 2. Simulation assets (sim and benchmark paths)

Public mesh/asset repositories are fetched with VCStool into `src/`:

```bash
pip install vcstool
cd src && vcs import < ../spark.repos && cd ..
```

This pulls mujoco_menagerie, robosuite, object_sim, furniture_sim, PartNeXt,
SAM3, GraspGen, the LIBERO-PRO benchmark (`src/libero_pro`, imported as
`libero.libero` through `PYTHONPATH=src/libero_pro:src`), and the CaP-X
baseline (`src/cap_gym`). They are gitignored and never committed.

EquiGraspFlow is a git submodule pulled by a recursive clone:

```bash
git clone --recursive <repo-url>
# or, in an existing checkout:
git submodule update --init
```

It points at the public upstream https://github.com/bdlim99/EquiGraspFlow.

## 3. Perception and planning backends

- SAM3 (open-vocabulary detection): install from the upstream
  https://github.com/facebookresearch/sam3 (fetched by `spark.repos` into
  `src/sam3`), then `pip install -e .`. Point `SPARK_SAM3_PATH` at the
  checkout if it is not on the Python path.
- Gemini planner: `pip install google-genai` (in requirements) and export
  `GEMINI_API_KEY`.
- Depth Anything 3 (optional depth backend): install from
  https://github.com/ByteDance-Seed/Depth-Anything-3.

## 4. Optional components

Install only if you use the matching feature:

- EquiGraspFlow (SE(3) grasp synthesis, used by `grasp_se3`): obtained via
  the recursive clone above. Pretrained weights download separately per its
  README.
- panda-py (bimanual FCI / Panda arm): `pip install panda-python`
  (https://github.com/JeanElsner/panda-py).
- bamboo (FR3 torque-impedance controller): external C++ controller,
  installed separately on the robot host; the franky driver is the default
  and needs none of this.
- CaP-Bench baseline (`cap_gym`): needed only to reproduce the CaP-Bench
  comparison, not for SPARK itself.

## 5. Weights, data, and keys

Nothing below is in the repository; each is fetched on first use or by hand.

- SAM3 checkpoint: `facebook/sam3` from the Hugging Face hub, downloaded on
  first detection into `~/.cache/huggingface`.
- Depth Anything 3 (optional): `depth-anything/DA3METRIC-LARGE` from the hub.
- EquiGraspFlow weights (optional, `grasp_se3`): per the submodule README
  into `src/EquiGraspFlow/train_results/`.
- LIBERO-PRO init states and scenes: shipped inside the LIBERO-PRO checkout
  (`src/libero_pro/hf_data`); the runner reads them through `libero.libero`.
- Gemini planner key: put it in `src/.gemini_api_key` (one key per line; the
  second line is used as primary) or export `GEMINI_API_KEY`.
- Camera extrinsics and rig IPs: per-machine files under
  `src/spark_real/configs/machines/` and `~/.spark_real/`, never committed.
  `configs/machines/ANON-LAB.example.yaml` is the schema for a bimanual rig.
- VLA baselines (OpenVLA, pi0.5, MolmoAct2) run in their own environments
  with their released LIBERO checkpoints; see `src/spark_bench/README.md`
  "Environments per benchmark".

## 6. Run

```bash
# Hardware-free boot (no robot, no cameras, no API key needed)
python -m spark_real.server --no-robot --no-init --no-kinect --port 8888

# Real robot
python -m spark_real.server --robot franka --port 8888    # or ur10e, bimanual_franka

# LIBERO-PRO benchmark
MUJOCO_GL=egl python -m spark_bench.run_spark_libero_pro_fair --suite object --perturbation position
```

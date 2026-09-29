# SPARK Repo Map

What is in this repository, what is first-party, what is vendored, and what is
cruft. The goal is to make the tree understandable and to drive cleanup. Numbers
are from `reconcile-jul07` (`git ls-files`).

## The numbers

- About 700 tracked files, 445 tracked `.py`.
- First-party Python is about 300 files: `spark_real` 226 (excluding 98 test
  files), `spark_bench` 57, `spark_dsl` 15. `spark_sim` tracks 44 scene XML
  files and no Python.
- The rest is configs, MuJoCo XML, assets and docs. The vendored dependencies
  under `src/` are gitignored and pulled by VCStool or per-repo clone.

## First-party code (keep)

- `src/spark_real/`: the real-robot FastAPI deployment. This is the primary path
  and is currently frozen for refactoring.
- `src/spark_bench/`: benchmark harness (LIBERO, LIBERO-PRO, robosuite). The
  canonical runner is `run_spark_libero_pro_fair.py` with the `libero_pro/`
  subpackage and `pyroki_ik.py`.
- `src/spark_sim/`: the MuJoCo simulation engine. `spark_real` re-exports from it.
- `src/spark_dsl/`: typed behavior-tree DSL grammar (lark) and its tests.

## Deprecated (ROS)

The ROS 2 packages (`spark_bringup`, `spark_controller`, `spark_sequencer`,
`spark_moveit_config`, `spark_perception`, `spark_interfaces`, `spark_fallback`)
are not tracked on `reconcile-jul07`. They exist only on older branches.

## Vendored dependencies (not our code)

Imported via VCStool or per-repo clone, gitignored except the `EquiGraspFlow`
pin: `mujoco_menagerie`, `robosuite`, `object_sim`, `furniture_sim`, `PartNeXt`,
`cap_gym`, `libero_pro/libero`, `EquiGraspFlow`, `graspgen`,
`external_controllers`, plus the `sam3` and `depth_anything_v3` symlinks.

## The folders that caused confusion

- `scripts/` (31): standalone CLI utilities mixed with one-off probes. Real
  utilities: `handeye_calibrate.py`, `spark_server.sh`, `kinect_*_reset.sh`,
  `rollout_recorder.py`, task demos. One-off probes that belong in an archive:
  `grasp_profile.py`, `slot_detector_probe.py`, `water_level_probe.py`,
  `sam3_standalone_probe.py`, `teleop_smoothness_test.py`, `compare_calibrations.py`.
- `tests/` (31): misleading name. `tests/corl/` is 22 CoRL experiment scripts (all
  have `__main__`/argparse, none have `test_`/pytest), so it is an experiments
  folder, not a test suite. Plus 7 loose files and 2 archived.
- `patches/` (on `main` and bimanual only): a single file
  `sam3_rtx5090_bf16_fix.patch` that fixes SAM3 for the RTX5090 BF16 path. Small
  and legitimate. It is a patch to a vendored dep, so it belongs in a documented
  `patches/` with a README.
- `refinement_corpus/` (on `main` and author): 724K of dated `.jsonl` experiment
  data (`2026-05-19_host3/`), tracked in git. This is data, not code, and should
  live outside the repo or in a gitignored data directory.

## Why the repo looks different on each machine

The branches have diverged at the top level. `main` carries `patches/`,
`refinement_corpus/`, `bt_library/`, `BENCHMARK_RESULTS.md`, `screenshots/`,
`todo.md`. `franka-deploy` carries `scripts/` and `spark_real.repos` but not those.
bimanual is on its own branch again. This divergence is why `patches/` and
`refinement_corpus/` appear on some machines and not others, and it is the main
obstacle to a usable, understandable repo across machines.

## What we actually need vs cruft

- Need: the four first-party packages, the canonical benchmark runner plus
  `libero_pro/`, the real utilities in `scripts/`, the docs in `src/docs/`.
- Cruft already removed: orphan `spark_bench/controller.py`,
  `run_spark_libero_pro_fair.py.bak2`, the old disposable runners
  (`run_spark_libero_full.py`, `run_spark_libero_pro.py`, `run_libero_native.py`),
  the data in `refinement_corpus/`.
- Cruft to remove or relocate (needs a decision): `run_spark_robosuite.py`, the
  experiment scripts mislabeled as tests, the one-off probes in `scripts/`.

## Cleanup plan

- Phase A (safe, done or trivial): remove backups and the orphan module; clean
  untracked `__pycache__`.
- Phase B (reorg, needs sign-off): rename `tests/corl/` to `experiments/`; move
  one-off probes to `scripts/archive/`; move `refinement_corpus/` data out of git.
- Phase C (consolidation): archive the old disposable runners under
  `spark_bench/legacy/` once their numbers are captured, keeping only the fair
  runner active.
- Phase D (branches): pick `main` as canonical, port the `franka-deploy` and
  bimanual deltas onto it, and delete the divergent cruft so every machine checks
  out the same tree. This is the real fix for cross-machine usability.

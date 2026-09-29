# spark_bench

Benchmarking harness for SPARK across LIBERO, LIBERO-PRO, robosuite, and
BEHAVIOR. Each runner is a standalone tyro CLI; the typed BT-DSL wires in via
`--use-dsl`.

## Quickstart

```bash
cd ~/spark/src

# LIBERO-PRO with adaptive prompts + DSL (50t reference run)
PYTHONPATH=~/spark/src/libero_pro:$PYTHONPATH \
  python -m spark_bench.run_spark_libero_pro_fair \
    --suite goal --perturbation position --num-trials 50 \
    --start-task-id 1 --adaptive-prompts --use-dsl

# Robosuite (Panda) with DSL
python -m spark_bench.run_spark_robosuite --task Lift --num-trials 10 --use-dsl

```

Conda envs:

| Runner | env | Why |
|---|---|---|
| `run_spark_libero_pro_fair` | `openvla` | bundles robosuite 1.4 (LIBERO requires it) |
| `run_spark_robosuite` | `spark_conda` | robosuite 1.5 (newer composite controller API) |

## Active modules

| File | Purpose |
|---|---|
| `run_spark_libero_pro_fair.py` | LIBERO-PRO suite runner: adaptive prompts, DSL integration, BDDL success checking (behavior tree + legato execution). |
| `run_spark_robosuite.py` | Robosuite tasks (Lift, Stack, Wipe, NutAssembly, TwoArmLift/Handover) |
| `run_spark_behavior.py` | BEHAVIOR-1K runner (Isaac Sim 4.5; blocked on RTX 5090 PhysX) |
| `dual_arm_bt.py` | Bimanual coordination for TwoArmLift/Handover |
| `prompt_sweep.py` | Adaptive-prompt offline search (BDDL-derived) |
| `libero_success_checker.py` | BDDL goal-state checker (subprocess to openvla env) |
| `generate_libero_pro_perturbations.py` | Generate position/task perturbations from base BDDLs |
| `generate_charts.py` | Matplotlib chart generation |
| `pyroki_ik.py` | PyRoki 6-DOF IK fallback for side-grasp primitives |
| `auto_chain.sh` | Sequential A/B chain runner (baseline -> DSL) |

## Results layout

```
results/
  libero_pro_logs/    Verbose run-by-run logs (sv## naming)
  dsl_benchmarks/     A/B vs baseline harness output
    baseline_*.log    Without --use-dsl
    dsl_*.log         With --use-dsl
    RESULTS.md        Per-run summary tables
  robosuite_*.json    Robosuite per-task JSON
```

`results/dsl_benchmarks/RESULTS.md` is the canonical A/B record updated after
each comparison run.

## Adding a runner

1. Define a `@dataclass` config (tyro-friendly) at top of the file.
2. Per-task loop: `env.reset()` -> capture -> SAM3 detect -> Gemini plan ->
   execute -> `env.check_success()`.
3. To enable DSL: import `BTExecutor` + `DEFAULT_LIBRARY`, call
   `validate_bt` then `expand_macros` then your existing dispatch loop. See
   `run_spark_libero_pro_fair.py:1922` for the canonical pattern (validate ->
   expand -> recursive flatten -> dispatch).

## libero_pro/ module layout

The LIBERO-PRO runner is decomposed into a package so each concern is testable
on its own:

```
spark_bench/libero_pro/
  bddl.py            BDDL parsing and init-state lookup
  perception.py      SAM3 detection for the LIBERO scene
  planning.py        Gemini, scripted, and DSL planners
  motion.py, ik.py   motion helpers and 6-DOF IK
  executor.py        action dispatch
  atomic_decomp.py   atomic task decomposition
  primitives/        per-primitive implementations
```

## Environments per benchmark (read before re-running anything)

- LIBERO / LIBERO-PRO / LIBERO-Dyn: `openvla_env` (robosuite 1.4.1, pinned by the
  LIBERO harness). No Pyroki there: the joint-IK descent falls back to the
  Cartesian servo.
- CaP-Bench (`run_spark_robosuite`): `capbench_env` = clone of `spark_conda`
  (Pyroki, jax, genai) + `robosuite==1.5.2` + `numpy==2.4.2` (numba needs <2.5,
  jax needs 2.x; robosuite's install downgrades numpy to 1.26 and must be undone)
  + `PyOpenGL==3.1.10` (MuJoCo EGL). Needed for the composite controllers the
  bimanual tasks use and for the Pyroki descent the Wipe scrub depends on. The
  June 2026 numbers (Wipe 60, TwoArmLift 63, TwoArmHandover 24) came from such an
  environment; in `openvla_env` Wipe scores ~0 (joint-limit termination) and the
  bimanual tasks fail to import.
- SAM3 is imported from a local checkout via a path insert in
  `spark_real.perception` (set `SPARK_SAM3_PATH` to the checkout); it is not a site-packages install.
- Set `SPARK_GEMINI_MODEL` for `run_spark_robosuite` (default falls back to
  `gemini-3-flash-preview`); pass `--results-json` to keep a machine record.

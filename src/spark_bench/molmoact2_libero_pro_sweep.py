"""
Full 50-trial sweep orchestrator for MolmoAct2-LIBERO on LIBERO-PRO.

Loads the model ONCE on GPU, then loops (suite, perturbation, task_id) x
n_trials.  Per-task results saved as JSON in `output_dir`; per-cell aggregates
collected into a top-level `sweep_summary_<stamp>.json`.

Mirrors the SPARK paper protocol exactly:
    3 suites (object, goal, spatial) x 2 perturbations (position, task)
        x 10 tasks/suite x 50 trials/task = 3000 trials total.

Usage::

    conda activate molmoact2
    cd ~/spark/src
    MUJOCO_GL=egl PYTHONPATH=src/libero_pro:src:$PYTHONPATH \\
        python -m spark_bench.molmoact2_libero_pro_sweep \\
            --suites goal,object,spatial \\
            --perturbations position,task \\
            --task-ids 0-9 --n-trials 50 \\
            --output-dir ~/spark/output/molmoact2_libero_pro
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time
from pathlib import Path

os.environ.setdefault("MUJOCO_GL", "egl")
os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")

import numpy as np
import torch

from libero.libero import benchmark
from libero.libero.envs import OffScreenRenderEnv

from spark_bench.molmoact2_libero_pro_runner import (
    RunConfig,
    _perturbation_suite,
    _resolve_bddl,
    load_molmoact2,
    rollout_one_trial,
)


def _parse_task_ids(spec: str) -> list[int]:
    """
    Accepts '0-9', '0,3,6', 'all', or a single int.
    """
    spec = spec.strip()
    if spec == "all":
        return list(range(10))
    if "-" in spec:
        a, b = spec.split("-")
        return list(range(int(a), int(b) + 1))
    return [int(x) for x in spec.split(",")]


def run_one_cell(
    *,
    suite: str,
    perturbation: str,
    task_id: int,
    n_trials: int,
    processor,
    model,
    base_cfg: RunConfig,
) -> dict:
    suite_name = _perturbation_suite(suite, perturbation)
    benchmark_dict = benchmark.get_benchmark_dict()
    if suite_name not in benchmark_dict:
        return {
            "suite": suite_name, "task_id": task_id, "n_trials": 0,
            "error": f"suite '{suite_name}' not registered",
        }
    task_suite = benchmark_dict[suite_name]()
    if task_id >= task_suite.n_tasks:
        return {
            "suite": suite_name, "task_id": task_id, "n_trials": 0,
            "error": f"task_id {task_id} out of range [0, {task_suite.n_tasks})",
        }

    task = task_suite.get_task(task_id)
    bddl = _resolve_bddl(task)
    try:
        init_states = task_suite.get_task_init_states(task_id)
    except Exception:
        init_states = None

    env = OffScreenRenderEnv(
        bddl_file_name=bddl,
        camera_heights=base_cfg.cam_height,
        camera_widths=base_cfg.cam_width,
        camera_depths=False,
        horizon=2000,
    )

    cfg = RunConfig(**{**base_cfg.__dict__, "suite": suite,
                       "perturbation": perturbation, "task_id": task_id,
                       "n_trials": n_trials})
    results = []
    t_cell = time.time()
    for trial in range(n_trials):
        init_state = (init_states[trial % len(init_states)]
                      if init_states is not None and len(init_states) > 0 else None)
        rec = rollout_one_trial(env, processor, model, task.language, cfg,
                                init_state, trial)
        rec["trial"] = trial
        rec["suite"] = suite_name
        rec["task_id"] = task_id
        rec["task_name"] = task.name
        results.append(rec)
        # Per-trial single-line log.
        print(f"[sweep] {suite_name} T{task_id} trial {trial:2d}: "
              f"success={int(rec['success'])} steps={rec['steps']:3d} "
              f"wc={rec['wall_clock_s']:.1f}s", flush=True)
    cell_wc = time.time() - t_cell
    try:
        env.close()
    except Exception:
        pass

    n_success = sum(1 for r in results if r["success"])
    cell = {
        "suite": suite_name,
        "perturbation": perturbation,
        "task_id": task_id,
        "task_name": task.name,
        "instruction": task.language,
        "n_trials": n_trials,
        "n_success": n_success,
        "success_rate": n_success / n_trials if n_trials else 0.0,
        "wall_clock_s": cell_wc,
        "trials": results,
    }
    return cell


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--suites", default="goal,object,spatial",
                   help="Comma-separated list of suites")
    p.add_argument("--perturbations", default="position,task",
                   help="Comma-separated list of perturbations")
    p.add_argument("--task-ids", default="0-9",
                   help="Task-id spec: '0-9' / '0,3,6' / 'all'")
    p.add_argument("--n-trials", type=int, default=50)
    p.add_argument("--max-steps", type=int, default=600)
    p.add_argument("--dtype", default="bfloat16", choices=["bfloat16", "float32"])
    p.add_argument("--n-action-steps", type=int, default=10)
    p.add_argument("--num-flow-steps", type=int, default=10)
    p.add_argument("--enable-cuda-graph", action="store_true")
    p.add_argument("--output-dir",
                   default=str(Path(__file__).resolve().parents[2] / "output" / "molmoact2_libero_pro_sweep"))
    p.add_argument("--num-steps-wait", type=int, default=50)
    args = p.parse_args()

    suites = [s.strip() for s in args.suites.split(",") if s.strip()]
    perturbations = [s.strip() for s in args.perturbations.split(",") if s.strip()]
    task_ids = _parse_task_ids(args.task_ids)

    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    stamp = time.strftime("%Y%m%d_%H%M%S")

    print(f"[sweep] suites={suites} perturbations={perturbations} "
          f"task_ids={task_ids} n_trials={args.n_trials}", flush=True)
    print(f"[sweep] total cells={len(suites)*len(perturbations)*len(task_ids)} "
          f"total trials={len(suites)*len(perturbations)*len(task_ids)*args.n_trials}",
          flush=True)

    t_load = time.time()
    processor, model = load_molmoact2(args.dtype)
    print(f"[sweep] model load: {time.time() - t_load:.1f}s; "
          f"cuda alloc={torch.cuda.memory_allocated() / 1e9:.2f}GB", flush=True)

    base_cfg = RunConfig(
        suite="goal", perturbation="position", task_id=0,
        n_trials=args.n_trials, max_steps=args.max_steps,
        num_steps_wait=args.num_steps_wait, dtype=args.dtype,
        n_action_steps=args.n_action_steps,
        num_flow_steps=args.num_flow_steps,
        enable_cuda_graph=args.enable_cuda_graph,
        output_dir=args.output_dir,
    )

    cells = []
    t_sweep = time.time()
    for suite in suites:
        for pert in perturbations:
            for tid in task_ids:
                t_cell = time.time()
                print(f"\n[sweep] >>> {suite}/{pert}/T{tid} <<<", flush=True)
                cell = run_one_cell(suite=suite, perturbation=pert, task_id=tid,
                                    n_trials=args.n_trials,
                                    processor=processor, model=model,
                                    base_cfg=base_cfg)
                cells.append(cell)
                # Save per-cell JSON immediately so a crash doesn't lose results.
                cell_path = (out_dir /
                             f"{cell['suite']}_T{tid}_n{args.n_trials}_{stamp}.json")
                with open(cell_path, "w") as f:
                    json.dump(cell, f, indent=2, default=str)
                sr = cell.get("success_rate", 0.0)
                wc = cell.get("wall_clock_s", 0.0)
                print(f"[sweep]{cell['suite']} T{tid}: "
                      f"{cell.get('n_success', 0)}/{cell.get('n_trials', 0)} "
                      f"({sr*100:.0f}%) wall_clock={wc:.1f}s", flush=True)
    sweep_wc = time.time() - t_sweep

    # Per-cell summary table.
    print("\n[sweep]FINAL TABLE", flush=True)
    print(f"{'suite':<25s} {'task':>4s} {'n_succ':>6s}/{'n_trials':<8s} {'SR':>6s} {'wc(s)':>8s}",
          flush=True)
    for c in cells:
        if "error" in c:
            print(f"{c['suite']:<25s} T{c['task_id']:<3d}  ERROR: {c['error']}", flush=True)
            continue
        print(f"{c['suite']:<25s} T{c['task_id']:<3d} {c['n_success']:>6d}/"
              f"{c['n_trials']:<8d} {c['success_rate']*100:>5.1f}% "
              f"{c['wall_clock_s']:>8.1f}", flush=True)

    # Aggregate per (suite, pert)
    aggr = {}
    for c in cells:
        if "error" in c:
            continue
        key = (c["suite"], c["perturbation"])
        a = aggr.setdefault(key, {"n_success": 0, "n_trials": 0})
        a["n_success"] += c["n_success"]
        a["n_trials"] += c["n_trials"]
    print("\n[sweep]PER-CELL AGGREGATE", flush=True)
    for (suite, pert), a in aggr.items():
        sr = a["n_success"] / a["n_trials"] if a["n_trials"] else 0.0
        print(f"{suite} ({pert}): {a['n_success']}/{a['n_trials']} = {sr*100:.1f}%",
              flush=True)
    sweep_summary = {
        "stamp": stamp, "suites": suites, "perturbations": perturbations,
        "task_ids": task_ids, "n_trials": args.n_trials,
        "wall_clock_s": sweep_wc, "cells": cells, "aggregate": {
            f"{s}|{p}": {"n_success": a["n_success"], "n_trials": a["n_trials"],
                         "success_rate": (a["n_success"]/a["n_trials"]
                                          if a["n_trials"] else 0.0)}
            for (s, p), a in aggr.items()
        },
    }
    summary_path = out_dir / f"sweep_summary_{stamp}.json"
    with open(summary_path, "w") as f:
        json.dump(sweep_summary, f, indent=2, default=str)
    print(f"\n[sweep] wrote {summary_path}  total_wall={sweep_wc:.0f}s "
          f"({sweep_wc/3600:.2f}h)", flush=True)


if __name__ == "__main__":
    main()

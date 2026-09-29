#!/usr/bin/env python3
"""
Build the MolmoAct2 vs SPARK head-to-head table from sweep JSONs.

Aggregates per-cell results from both sweep shards (one host covers
position+task, another covers object+language) and emits a LaTeX table row +
a per-task breakdown for the appendix.

Usage::

    # Single sweep directory
    python -m spark_bench.build_headtohead_table \\
        --in ~/spark/output/molmoact2_libero_pro_sweep/

    # Both sweeps (merge the second shard's directory)
    python -m spark_bench.build_headtohead_table \\
        --in ~/spark/output/molmoact2_libero_pro_sweep/ \\
             /tmp/host_obj_lan_sweep/

    # SPARK numbers for comparison (from main.tex Table 1)
    --spark-pos-pos 39.8 --spark-pos-task 36.4 --spark-goal-pos 40.0 \\
        --spark-goal-task 22.4 --spark-spat-pos 53.6 --spark-spat-task 72.4
"""
from __future__ import annotations

import argparse
import json
import re
from collections import defaultdict
from pathlib import Path


SUITE_NAME = re.compile(r"libero_(goal|object|spatial|10)_(swap|task|object|lan|env)")
PERT_LABEL = {
    "swap": "Pos",
    "task": "Task",
    "object": "Obj",
    "lan": "Lang",
    "env": "Env",
}


def load_cells(roots: list[Path]) -> dict[tuple[str, str, int], dict]:
    cells: dict[tuple[str, str, int], dict] = {}
    for root in roots:
        for path in root.glob("*_n50_*.json"):
            try:
                d = json.loads(path.read_text())
            except Exception:
                continue
            suite = d.get("suite", "")
            m = SUITE_NAME.match(suite)
            if not m:
                continue
            base_suite, axis = m.group(1), m.group(2)
            tid = int(d.get("task_id", -1))
            key = (base_suite, axis, tid)
            # Latest file wins on overlap.
            existing = cells.get(key)
            if existing is None or path.stat().st_mtime > existing["_mtime"]:
                cells[key] = {**d, "_mtime": path.stat().st_mtime, "_path": str(path)}
    return cells


def aggregate(cells: dict[tuple[str, str, int], dict]
              ) -> dict[tuple[str, str], dict]:
    agg: dict[tuple[str, str], dict] = defaultdict(
        lambda: {"n_success": 0, "n_trials": 0, "tasks": {}}
    )
    for (base, axis, tid), c in sorted(cells.items()):
        a = agg[(base, axis)]
        a["n_success"] += c.get("n_success", 0)
        a["n_trials"] += c.get("n_trials", 0)
        a["tasks"][tid] = {
            "n_success": c.get("n_success", 0),
            "n_trials": c.get("n_trials", 0),
            "rate": c.get("success_rate", 0.0),
            "task_name": c.get("task_name", ""),
        }
    for v in agg.values():
        v["rate"] = (v["n_success"] / v["n_trials"]) if v["n_trials"] else 0.0
    return agg


def print_per_cell(agg: dict[tuple[str, str], dict]) -> None:
    print(f"\n{'suite':<10s} {'axis':<6s} {'n_succ':>6s}/{'n_trials':<8s} {'SR':>6s}")
    print("-" * 50)
    for (base, axis), v in sorted(agg.items()):
        print(f"{base:<10s} {PERT_LABEL[axis]:<6s} "
              f"{v['n_success']:>6d}/{v['n_trials']:<8d} {v['rate']*100:>5.1f}%")


def print_per_task_table(agg: dict[tuple[str, str], dict]) -> None:
    print("\nPer-task breakdown (for paper appendix)")
    for (base, axis), v in sorted(agg.items()):
        if not v["tasks"]:
            continue
        print(f"\n  {base} / {PERT_LABEL[axis]}:")
        for tid, t in sorted(v["tasks"].items()):
            marker = "ok" if t["rate"] >= 0.5 else (" " if t["rate"] > 0 else "x")
            tn = t["task_name"][:48]
            print(f"{marker} T{tid:<2d} {t['n_success']:>3d}/{t['n_trials']:<3d} "
                  f"({t['rate']*100:>5.1f}%)  {tn}")


def latex_row(agg: dict[tuple[str, str], dict], spark_args: dict) -> str:
    """
    Emit a LaTeX row for the Table 1 main result table.

    Schema: Object-Pos / Object-Task / Goal-Pos / Goal-Task / Spatial-Pos /
    Spatial-Task / Mean - same ordering as main.tex Table 1.
    """
    def cell(base, axis):
        v = agg.get((base, axis))
        if v is None or v["n_trials"] == 0:
            return "--"
        return f"{v['rate']*100:.1f}"

    cells = [
        cell("object", "swap"),
        cell("object", "task"),
        cell("goal",   "swap"),
        cell("goal",   "task"),
        cell("spatial", "swap"),
        cell("spatial", "task"),
    ]
    # Mean across non-missing cells.
    vals = [float(c) for c in cells if c != "--"]
    mean = (sum(vals) / len(vals)) if vals else 0.0

    row = (f"MolmoAct2-LIBERO~\\citep{{fang2026molmoact2}} & "
           + " & ".join(f"${c}$" for c in cells)
           + f" & ${mean:.1f}$ \\\\")
    return row


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--in", dest="inputs", nargs="+", required=True,
                   type=Path, help="Sweep output dirs (one or many)")
    p.add_argument("--show-tasks", action="store_true",
                   help="Also print per-task breakdown")
    p.add_argument("--latex", action="store_true",
                   help="Emit a LaTeX row for Table 1")
    args = p.parse_args()

    cells = load_cells(args.inputs)
    if not cells:
        print(f"No *_n50_*.json files found under: {args.inputs}")
        return

    print(f"Loaded {len(cells)} cells across {len(args.inputs)} dir(s).")
    agg = aggregate(cells)
    print_per_cell(agg)

    if args.show_tasks:
        print_per_task_table(agg)

    if args.latex:
        print("\nLaTeX row (paste into main.tex tab:libero_main)")
        print(latex_row(agg, {}))


if __name__ == "__main__":
    main()

"""
Wilson intervals and task-clustered bootstraps for the LIBERO-PRO tables.

Every LIBERO-PRO cell is ten tasks at ten initial states, so the hundred trials
in a cell are NOT a hundred independent Bernoulli draws: trials within a task
share a scene, a prompt and a primitive path, and the per-task rates are wildly
heterogeneous (goal-position runs 0.0, 0.9, 0.7, 0.0, 0.5, 0.0, 0.2, 0.4, 1.0,
0.0). A naive Wilson interval on n=100 therefore reports a width the design does
not earn.

Both are computed here. Wilson on the pooled n is reported because it is what a
reader will compute otherwise, and the task-clustered bootstrap is reported
as primary because it is the one the design supports. Where they disagree the
clustered interval is roughly twice as wide, and that gap is the honest cost of
ten tasks rather than a hundred.

Arm comparisons are paired on task index, which the runner guarantees: both arms
walk identical task ordering, so task i in arm A and task i in arm B are the same
scene under the same perturbation.
"""
from __future__ import annotations

import json
import math
import random
from pathlib import Path

Z = 1.959963984540054  # two-sided 95%
BOOT = 10000
SEED = 20260827


def wilson(k: int, n: int, z: float = Z) -> tuple[float, float]:
    """Wilson score interval. Returns (lo, hi) as proportions."""
    if n == 0:
        return (0.0, 0.0)
    p = k / n
    d = 1.0 + z * z / n
    c = p + z * z / (2 * n)
    h = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n))
    return (max(0.0, (c - h) / d), min(1.0, (c + h) / d))


def _cell_matrix(cell: dict) -> list[list[bool]]:
    """Rows are tasks, columns are trials at that task's initial states."""
    return [list(td['trials']) for td in cell['task_details']]


def cluster_boot(rows: list[list[bool]], reps: int = BOOT,
                 seed: int = SEED) -> tuple[float, float]:
    """Percentile CI resampling whole TASKS with replacement, not trials."""
    rng = random.Random(seed)
    n = len(rows)
    if n == 0:
        return (0.0, 0.0)
    means = []
    for _ in range(reps):
        pick = [rows[rng.randrange(n)] for _ in range(n)]
        flat = [t for r in pick for t in r]
        means.append(sum(flat) / len(flat))
    means.sort()
    return (means[int(0.025 * reps)], means[int(0.975 * reps)])


def paired_boot(a: list[list[bool]], b: list[list[bool]], reps: int = BOOT,
                seed: int = SEED) -> tuple[float, float, float]:
    """
    Paired task-clustered bootstrap of (mean_b - mean_a).

    Returns (point, lo, hi). An interval excluding zero is the claim the paper
    is entitled to make; one straddling zero is not, however large the point
    difference looks.
    """
    rng = random.Random(seed)
    n = min(len(a), len(b))
    point = (sum(t for r in b[:n] for t in r) / sum(len(r) for r in b[:n])
             - sum(t for r in a[:n] for t in r) / sum(len(r) for r in a[:n]))
    diffs = []
    for _ in range(reps):
        idx = [rng.randrange(n) for _ in range(n)]
        fa = [t for i in idx for t in a[i]]
        fb = [t for i in idx for t in b[i]]
        diffs.append(sum(fb) / len(fb) - sum(fa) / len(fa))
    diffs.sort()
    return (point, diffs[int(0.025 * reps)], diffs[int(0.975 * reps)])


def load_arm(path: str | Path) -> dict:
    """Map perturbation name -> task-by-trial matrix for one arm's JSON."""
    d = json.loads(Path(path).read_text())
    return {p: _cell_matrix(c) for p, c in d['perturbations'].items()}


def describe(rows: list[list[bool]]) -> dict:
    flat = [t for r in rows for t in r]
    k, n = sum(flat), len(flat)
    wlo, whi = wilson(k, n)
    blo, bhi = cluster_boot(rows)
    return {
        'k': k, 'n': n, 'rate': k / n if n else 0.0,
        'wilson': (wlo, whi), 'cluster': (blo, bhi),
        'tasks': len(rows), 'zero_tasks': sum(1 for r in rows if not any(r)),
    }


def macro_boot(arm_cells: list[list[list[bool]]], reps: int = BOOT,
               seed: int = SEED) -> tuple[float, float, float]:
    """
    Clustered CI for a macro average over cells, resampling tasks within each
    cell independently. Returns (point, lo, hi).
    """
    rng = random.Random(seed)
    point = sum(sum(t for r in c for t in r) / sum(len(r) for r in c)
                for c in arm_cells) / len(arm_cells)
    out = []
    for _ in range(reps):
        acc = []
        for c in arm_cells:
            idx = [rng.randrange(len(c)) for _ in range(len(c))]
            flat = [t for i in idx for t in c[i]]
            acc.append(sum(flat) / len(flat))
        out.append(sum(acc) / len(acc))
    out.sort()
    return (point, out[int(0.025 * reps)], out[int(0.975 * reps)])

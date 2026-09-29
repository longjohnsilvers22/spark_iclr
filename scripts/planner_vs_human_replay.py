"""Offline dry-run: what would the PLANNER have done on the human's own scene?

READ-ONLY over the recorded teleop corpus. No robot, no camera, no live scene.
The only side effects are Gemini API calls (one per sampled episode, all logged)
and files under ``--out``.

    export SPARK_HUMAN_EPISODES=/path/to/teleop_episodes
    PYTHONPATH=src conda run -n spark_conda python scripts/planner_vs_human_replay.py \
        --episodes-per-task 3 --out output/planner_vs_human

WHAT IT MEASURES
    For each sampled episode: feed the planner the FIRST frame of camera_0 (the
    initial scene, which is what a real run would plan from) plus the recorded
    task string, and the WHOLE corpus vocabulary as the candidate keypoint set
    -- so the planner has to pick the right object and the right container out
    of ten labels, not out of a two-item giveaway. Then compare the plan's
    grasp/place targets against what the human actually did in that episode,
    derived from ``trajectory.npz``:
        human grasp xyz   = TCP at the frame the commanded trigger first closes
        human release xyz = TCP at the last frame before it opens again

    Only episodes whose own metadata.json says ``success: true`` are scored.

HOW A SYMBOLIC PLAN BECOMES METRES (and why the number has a noise floor)
    The corpus is RGB-only: no depth, no camera calibration. A plan names
    LABELS, not coordinates, so a metric comparison needs a label -> base-frame
    position map, and the corpus does not ship one. This script fits one:
    objects sit on a table plane, so a single affine map takes a camera_0 image
    point to base-frame XY. It is fitted per task, LEAVE-ONE-EPISODE-OUT, from
    correspondences the corpus already provides -- (object mask centroid before
    the grasp, human grasp XY) and (container mask centroid after the release,
    human release XY) -- using the offline SAM3 mask cache written by
    ``verify_replay_harvest``.

    Consequences:
      * XY only. Z is not recoverable from a planar fit; the reported distance
        is a table-plane distance.
      * The fit's same-role leave-one-out residual is the NOISE FLOOR. A plan
        that picks the correct label cannot score better than that, so every
        table prints the floor beside the measurement.
      * Containers barely move between episodes, so a fit anchored mostly on
        containers is a near-degenerate blob that extrapolates by a metre when
        asked where a knife is. ``TaskFit`` refuses those instead of printing
        them: too few pairs, too little spread, or too far outside the fitted
        region all report ``unresolved`` with the reason.
      * Only labels this episode has a mask for can be resolved. A plan naming
        some third object is ``unresolved``, never silently scored.

    ``--oracle`` substitutes the ground-truth labels and calls no LLM. That is
    the APPARATUS CEILING: whatever error it leaves is the measuring rig's, not
    the planner's. Read any planner number against it, never on its own.
"""

from __future__ import annotations

import argparse
import json
import sys
import textwrap
import time
from collections import defaultdict
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import matplotlib

matplotlib.use("Agg")  # headless: no display, no camera, no robot

import matplotlib.pyplot as plt
import numpy as np
from PIL import Image

from spark_real.planning.spark_planner import SPARKPlanner
from spark_real.tests.human_corpus import (
    TASK_PROMPTS,
    VOCABULARY,
    corpus_root,
    episode_events,
    episode_success,
    labels_match,
    load_mask_index,
)

FIT_CAMERA = "camera_0"
PLAN_CAMERA = "camera_0"
GRASP_TYPES = ("grasp", "grasp_se3", "grasp_top_down", "grasp_cgn", "grasp_se3_flow")
MIN_FIT_PAIRS = 5  # below this a per-task affine is not worth trusting
# Correspondences must span at least this much on their SHORT image axis, else
# the affine is a blob being asked to extrapolate. Containers alone fail this.
MIN_UV_SPREAD_PX = 6.0
# How far past the fitted region a query point may sit before it is refused.
MAX_EXTRAPOLATION = 1.25


# ---------------------------------------------------------------------------
# plan parsing
# ---------------------------------------------------------------------------


def flatten(score) -> List[dict]:
    """Leaf nodes of a SPARK score tree, in execution order."""
    out: List[dict] = []

    def walk(node):
        if not isinstance(node, dict):
            return
        kids = node.get("children")
        if isinstance(kids, list) and kids:
            for child in kids:
                walk(child)
        else:
            out.append(node)

    walk((score or {}).get("tree") if isinstance(score, dict) else None)
    return out


def plan_targets(score) -> Tuple[Optional[str], Optional[str], List[str]]:
    """(grasp label, place label, primitive-type sequence) from a plan."""
    grasp_label = place_label = None
    pending = None
    holding = False
    types: List[str] = []
    for node in flatten(score):
        atype = str(node.get("type", "") or "")
        params = node.get("params") or {}
        label = params.get("keypoint_label") or params.get("label")
        types.append(atype)
        if atype == "move_to_keypoint":
            pending = label
            if holding and label and place_label is None:
                place_label = label
        elif atype in GRASP_TYPES:
            if grasp_label is None:
                grasp_label = label or pending
            holding = True
        elif atype == "place_in_slot":
            place_label = place_label or params.get("container_label") or pending
            holding = False
        elif atype == "stack":
            place_label = place_label or params.get("target_label") or pending
            holding = False
        elif atype == "release":
            place_label = place_label or pending
            holding = False
    return grasp_label, place_label, types


# ---------------------------------------------------------------------------
# image -> table-plane affine
# ---------------------------------------------------------------------------


def fit_affine(uv: np.ndarray, xy: np.ndarray) -> np.ndarray:
    design = np.hstack([uv, np.ones((len(uv), 1))])
    sol, *_ = np.linalg.lstsq(design, xy, rcond=None)
    return sol


def apply_affine(sol: np.ndarray, uv) -> np.ndarray:
    return np.asarray([*np.asarray(uv, dtype=float), 1.0]) @ sol


def loo_residual(uv: np.ndarray, xy: np.ndarray) -> np.ndarray:
    """Leave-one-out held-out error of the affine fit, in metres."""
    errs = []
    for i in range(len(uv)):
        keep = np.ones(len(uv), bool)
        keep[i] = False
        if keep.sum() < 4:
            continue
        sol = fit_affine(uv[keep], xy[keep])
        errs.append(float(np.linalg.norm(apply_affine(sol, uv[i]) - xy[i])))
    return np.asarray(errs)


def build_correspondences(
    root: Path, records: List[dict], flags: Dict[Tuple[str, str], Optional[bool]]
) -> Dict[str, list]:
    """(image centroid, base XY) pairs per task, from the mask cache.

    Only success-flagged episodes anchor the fit: a botched demo's release XY is
    not where the container is.
    """
    pairs: Dict[str, list] = defaultdict(list)
    events: Dict[Tuple[str, str], object] = {}
    for rec in records:
        if rec["camera"] != FIT_CAMERA:
            continue
        key = (rec["task"], rec["episode"])
        if flags.get(key) is not True:
            continue
        if key not in events:
            events[key] = episode_events(root / rec["task"] / rec["episode"])
        ev = events[key]
        if ev is None:
            continue
        # pre-grasp frame localises the object; post-release frame the container
        role = "obj" if rec["phase"] == "negative" else "container"
        centroid = rec.get(f"{role}_centroid")
        if centroid is None:
            continue
        target = ev.grasp_xyz[:2] if role == "obj" else ev.release_xyz[:2]
        pairs[rec["task"]].append((rec["episode"], role, centroid, np.asarray(target)))
    return pairs


class TaskFit:
    """One task's affine, fitted on every correspondence EXCEPT one episode's.

    Leave-one-episode-out rather than a fixed holdout split: containers barely
    move between episodes, so a fit trained on containers alone is a
    near-degenerate blob that extrapolates by a metre when asked where a knife
    is. See ``degenerate``.
    """

    def __init__(self, rows: List[tuple]):
        self.n = len(rows)
        self.sol = None
        self.uv = np.zeros((0, 2))
        self.spread_px = 0.0
        self.floor_by_role: Dict[str, float] = {}
        if self.n < MIN_FIT_PAIRS:
            return
        self.uv = np.asarray([r[2] for r in rows], dtype=float)
        xy = np.asarray([r[3] for r in rows], dtype=float)
        # Second singular value of the centred image points: how far the
        # correspondences span the SHORT axis. Near zero = collinear/clustered.
        centred = self.uv - self.uv.mean(0)
        svals = np.linalg.svd(centred, compute_uv=False)
        self.spread_px = float(svals[-1] / np.sqrt(self.n))
        self.sol = fit_affine(self.uv, xy)
        res = loo_residual(self.uv, xy)
        roles = [r[1] for r in rows]
        if len(res) == len(rows):
            for role in ("obj", "container"):
                sub = [e for e, rl in zip(res, roles) if rl == role]
                if sub:
                    self.floor_by_role[role] = float(np.median(sub))

    @property
    def degenerate(self) -> bool:
        return self.sol is None or self.spread_px < MIN_UV_SPREAD_PX

    def solve(self, centroid) -> Tuple[Optional[np.ndarray], str]:
        """Map an image centroid to base XY, or explain why the map is untrustworthy."""
        if self.sol is None:
            return None, f"only {self.n} correspondence(s), need {MIN_FIT_PAIRS}"
        if self.spread_px < MIN_UV_SPREAD_PX:
            return None, (
                f"fit degenerate: correspondences span {self.spread_px:.1f}px on their "
                f"short axis (need {MIN_UV_SPREAD_PX})"
            )
        query = np.asarray(centroid, dtype=float)
        centre = self.uv.mean(0)
        reach = float(np.max(np.linalg.norm(self.uv - centre, axis=1)))
        dist = float(np.linalg.norm(query - centre))
        if reach <= 0 or dist > MAX_EXTRAPOLATION * reach:
            return None, (
                f"extrapolated: query is {dist:.0f}px from the fit centre, "
                f"fit only reaches {reach:.0f}px"
            )
        return apply_affine(self.sol, query), "ok"

    def floor(self, role: str) -> float:
        return self.floor_by_role.get(role, float("nan"))


class TableFit:
    """Leave-one-episode-out affines, per task. Only success episodes anchor it."""

    def __init__(self, pairs: Dict[str, list]):
        self._pairs = pairs
        self._cache: Dict[Tuple[str, str], TaskFit] = {}

    def for_episode(self, task: str, episode: str) -> TaskFit:
        key = (task, episode)
        if key not in self._cache:
            rows = [p for p in self._pairs.get(task, []) if p[0] != episode]
            self._cache[key] = TaskFit(rows)
        return self._cache[key]


# ---------------------------------------------------------------------------
# the replay
# ---------------------------------------------------------------------------


def success_flags(root: Path, records: List[dict]) -> Dict[Tuple[str, str], Optional[bool]]:
    """metadata.json success per (task, episode) seen in the mask cache."""
    return {
        (r["task"], r["episode"]): episode_success(root / r["task"] / r["episode"])
        for r in records
    }


def sample_episodes(
    records: List[dict], per_task: int, flags: Dict[Tuple[str, str], Optional[bool]]
) -> Tuple[Dict[str, List[str]], Dict[str, List[str]]]:
    """Episodes with BOTH centroids in camera_0; non-success ones are held out."""
    have_obj, have_cont = defaultdict(set), defaultdict(set)
    for rec in records:
        if rec["camera"] != FIT_CAMERA:
            continue
        if rec["phase"] == "negative" and rec.get("obj_centroid"):
            have_obj[rec["task"]].add(rec["episode"])
        if rec["phase"] == "positive" and rec.get("container_centroid"):
            have_cont[rec["task"]].add(rec["episode"])
    out, dropped = {}, {}
    for task in TASK_PROMPTS:
        both = sorted(have_obj[task] & have_cont[task])
        good = [e for e in both if flags.get((task, e)) is True]
        dropped[task] = [e for e in both if flags.get((task, e)) is not True]
        out[task] = good[:per_task]
    return out, dropped


def centroid_table(records: List[dict], task: str, episode: str) -> Dict[str, tuple]:
    """label -> camera_0 image centroid, for the labels this episode can resolve."""
    obj_prompt, cont_prompt, _ = TASK_PROMPTS[task]
    table = {}
    for rec in records:
        if rec["camera"] != FIT_CAMERA or rec["task"] != task or rec["episode"] != episode:
            continue
        if rec["phase"] == "negative" and rec.get("obj_centroid"):
            table.setdefault(obj_prompt, rec["obj_centroid"])
        if rec["phase"] == "positive" and rec.get("container_centroid"):
            table[cont_prompt] = rec["container_centroid"]
    return table


def resolve(label: Optional[str], table: Dict[str, tuple], task: str) -> Optional[tuple]:
    if not label:
        return None
    for known, centroid in table.items():
        if labels_match(label, known, task):
            return centroid
    return None


def run_planner(planner, instruction: str, image, call_log: list) -> Tuple[Optional[dict], str]:
    """One Gemini call. Every call is logged with its wall time and outcome."""
    t0 = time.time()
    try:
        score = planner.generate_score(
            instruction=instruction,
            annotated_image=image,
            keypoint_labels=list(VOCABULARY),
        )
        err = ""
    except Exception as exc:  # a failed call is data, not a crash
        score, err = None, f"{type(exc).__name__}: {exc}"
    entry = {
        "instruction": instruction,
        "model": planner.model,
        "temperature": planner.temperature,
        "seconds": round(time.time() - t0, 2),
        "ok": score is not None,
        "error": err,
    }
    call_log.append(entry)
    print(
        f"    [gemini call {len(call_log):2d}] {planner.model} "
        f"{entry['seconds']:5.1f}s ok={entry['ok']} {err}",
        flush=True,
    )
    return score, err


# ---------------------------------------------------------------------------
# plotting
# ---------------------------------------------------------------------------


def plot_task(task: str, rows: List[dict], out_dir: Path, label: str) -> Optional[Path]:
    plotted = [r for r in rows if r.get("human") is not None]
    if not plotted:
        return None
    fig, axes = plt.subplots(1, len(plotted), figsize=(4.6 * len(plotted), 4.6), squeeze=False)
    for ax, row in zip(axes[0], plotted):
        path = np.asarray(row["human"]["tcp_xy"])
        ax.plot(path[:, 0], path[:, 1], "-", color="0.55", lw=1.2, label="human TCP path")
        gx, gy = row["human"]["grasp_xy"]
        rx, ry = row["human"]["release_xy"]
        ax.plot(gx, gy, "o", color="#1f77b4", ms=11, mfc="none", mew=2, label="human grasp")
        ax.plot(rx, ry, "s", color="#2ca02c", ms=11, mfc="none", mew=2, label="human release")
        for key, colour, marker, name in (
            ("grasp", "#1f77b4", "x", "planned grasp target"),
            ("place", "#2ca02c", "+", "planned place target"),
        ):
            xy = row["planned"].get(f"{key}_xy")
            if xy is None:
                continue
            ax.plot(xy[0], xy[1], marker, color=colour, ms=13, mew=2.5, label=name)
            anchor = (gx, gy) if key == "grasp" else (rx, ry)
            ax.plot([anchor[0], xy[0]], [anchor[1], xy[1]], ":", color=colour, lw=1.4)
        errs = []
        for key, fkey in (("grasp", "grasp_floor"), ("place", "place_floor")):
            err = row["errors"].get(key)
            floor = row["errors"].get(fkey)
            tail = f" (floor {floor * 100:.1f})" if floor == floor else ""
            errs.append(
                f"{key} {err * 100:.1f}cm{tail}" if err is not None else f"{key} unresolved"
            )
        ax.set_title(f"{row['episode']}\n" + "  |  ".join(errs), fontsize=9)
        ax.set_xlabel("robot base X (m)")
        ax.set_ylabel("robot base Y (m)")
        ax.set_aspect("equal", adjustable="datalim")
        ax.grid(alpha=0.3)
    axes[0][0].legend(fontsize=7, loc="best")
    fig.suptitle(
        f'"{task}"   {label}  vs recorded human   (table-plane XY, camera_0)',
        fontsize=11,
    )
    fig.tight_layout()
    path = out_dir / f"{task.replace(' ', '_')}.png"
    fig.savefig(path, dpi=130)
    plt.close(fig)
    return path


def plot_summary(rows: List[dict], out_dir: Path, label: str) -> Optional[Path]:
    pts = [(r["errors"].get("grasp"), r["errors"].get("place"), r) for r in rows]
    pts = [(g, p, r) for g, p, r in pts if g is not None or p is not None]
    if not pts:
        return None
    fig, ax = plt.subplots(figsize=(9, 4.6))
    tasks = sorted({r["task"] for _, _, r in pts})
    xpos = {t: i for i, t in enumerate(tasks)}
    for kind, colour, off in (("grasp", "#1f77b4", -0.13), ("place", "#2ca02c", 0.13)):
        xs, ys = [], []
        for g, p, r in pts:
            val = g if kind == "grasp" else p
            if val is None:
                continue
            xs.append(xpos[r["task"]] + off)
            ys.append(val * 100)
        ax.plot(xs, ys, "o", color=colour, ms=8, alpha=0.75, label=f"{kind} target error")
    floors = [
        r["errors"][k] * 100
        for _, _, r in pts
        for k in ("grasp_floor", "place_floor")
        if r["errors"].get(k) == r["errors"].get(k)
    ]
    if floors:
        ax.axhline(
            float(np.median(floors)),
            ls="--",
            color="0.4",
            label=f"fit noise floor (median {np.median(floors):.1f} cm)",
        )
    ax.set_xticks(range(len(tasks)))
    ax.set_xticklabels(
        [textwrap.fill(t, 16) for t in tasks], fontsize=7, rotation=30, ha="right"
    )
    ax.set_xlim(-0.6, len(tasks) - 0.4)
    ax.set_ylabel("distance planned target -> human actual (cm, table plane)")
    ax.set_title(f"{label} vs recorded human, per episode")
    ax.grid(alpha=0.3, axis="y")
    ax.legend(fontsize=8)
    fig.tight_layout()
    path = out_dir / "summary.png"
    fig.savefig(path, dpi=130)
    plt.close(fig)
    return path


# ---------------------------------------------------------------------------


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--root", default=None, help="corpus root; defaults to $SPARK_HUMAN_EPISODES")
    ap.add_argument("--cache", default="output/verify_replay_cache")
    ap.add_argument("--out", default="output/planner_vs_human")
    ap.add_argument("--episodes-per-task", type=int, default=3)
    ap.add_argument("--model", default=None)
    ap.add_argument("--temperature", type=float, default=0.0)
    ap.add_argument(
        "--reuse",
        action="store_true",
        help="reuse a plan already saved under --out instead of re-calling Gemini",
    )
    ap.add_argument(
        "--oracle",
        action="store_true",
        help="APPARATUS CEILING: substitute the ground-truth labels for the plan "
        "and make no Gemini call. Whatever error this leaves is the measuring "
        "rig's own, not the planner's -- always run it alongside a real pass.",
    )
    args = ap.parse_args()

    root = corpus_root(args.root)
    if root is None or not root.is_dir():
        sys.exit("no corpus: set SPARK_HUMAN_EPISODES=/path/to/teleop_episodes or pass --root")
    cache = Path(args.cache)
    if not (cache / "index.json").exists():
        sys.exit(
            f"no mask cache at {cache}. Produce it with:\n"
            "  PYTHONPATH=src conda run -n sam3 python -m "
            "spark_real.tests.verify_replay_harvest --episodes-per-task 9"
        )
    out_dir = Path(args.out)
    (out_dir / "plans").mkdir(parents=True, exist_ok=True)

    records = load_mask_index(cache)
    flags = success_flags(root, records)
    sample, dropped = sample_episodes(records, args.episodes_per_task, flags)
    fit = TableFit(build_correspondences(root, records, flags))

    n_flagged = sum(1 for v in flags.values() if v is not True)
    print(f"corpus            : {root}")
    print(f"mask cache        : {cache}  ({len(records)} records)")
    print(f"candidate labels  : {VOCABULARY}")
    print(
        f"metadata success  : {len(flags) - n_flagged}/{len(flags)} cached episodes "
        f"flagged success=true; {n_flagged} excluded"
    )
    for task, eps in dropped.items():
        if eps:
            print(f"    EXCLUDED (not success): {task} -> {eps}")
    print("image->table fit  : per task, leave-one-EPISODE-out, success episodes only")
    for task in sorted(sample):
        if not sample[task]:
            print(f"    {task[:38]:38} sample=0 (no episode resolves both centroids)")
            continue
        tf = fit.for_episode(task, sample[task][0])
        print(
            f"    {task[:38]:38} sample={len(sample[task])} fit_n={tf.n} "
            f"spread={tf.spread_px:.1f}px"
            f"{'  DEGENERATE' if tf.degenerate else ''}"
            f"  floor obj={tf.floor('obj') * 100:.1f}cm cont={tf.floor('container') * 100:.1f}cm"
        )
    n_calls = 0 if (args.reuse or args.oracle) else sum(len(v) for v in sample.values())
    mode = "ORACLE (ground-truth labels)" if args.oracle else "planner"
    print(f"mode              : {mode}")
    print(f"planned Gemini calls: {n_calls}\n")

    # Oracle mode never calls out, so it must not even need a key.
    planner = None if args.oracle else SPARKPlanner(model=args.model, temperature=args.temperature)
    call_log: List[dict] = []
    rows: List[dict] = []

    for task, episodes in sample.items():
        obj_truth, cont_truth, _ = TASK_PROMPTS[task]
        print(f"{task}")
        for episode in episodes:
            ep_dir = root / task / episode
            ev = episode_events(ep_dir)
            frame = ep_dir / "images" / PLAN_CAMERA / "frame_0000.jpg"
            if ev is None or not frame.exists():
                print(f"  {episode}: no clean grip or no first frame, skipped")
                continue
            plan_path = out_dir / "plans" / f"{task.replace(' ', '_')}__{episode}.json"

            if args.oracle:
                # The ceiling: perfect labels, no LLM. Any error left is the rig's.
                score, err = None, ""
                grasp_label, place_label, types = obj_truth, cont_truth, ["oracle"]
            else:
                if args.reuse and plan_path.exists():
                    score = json.loads(plan_path.read_text())["score"]
                    err = ""
                else:
                    with Image.open(frame) as img:
                        score, err = run_planner(planner, task, img.convert("RGB"), call_log)
                    plan_path.write_text(json.dumps({"task": task, "score": score}, indent=1))
                grasp_label, place_label, types = plan_targets(score)

            tf = fit.for_episode(task, episode)
            table = centroid_table(records, task, episode)
            errors: Dict[str, Optional[float]] = {}
            planned: Dict[str, object] = {
                "grasp_label": grasp_label,
                "place_label": place_label,
                "primitives": types,
            }
            for key, label, role, truth_xy in (
                ("grasp", grasp_label, "obj", ev.grasp_xyz[:2]),
                ("place", place_label, "container", ev.release_xyz[:2]),
            ):
                centroid = resolve(label, table, task)
                if centroid is None:
                    xy, why = None, f"no cached mask for {label!r} in this episode"
                else:
                    xy, why = tf.solve(centroid)
                planned[f"{key}_xy"] = None if xy is None else [float(v) for v in xy]
                planned[f"{key}_unresolved_reason"] = None if xy is not None else why
                errors[key] = None if xy is None else float(np.linalg.norm(xy - truth_xy))
                errors[f"{key}_floor"] = tf.floor(role)

            row = {
                "task": task,
                "episode": episode,
                "instruction": task,
                "metadata_success": flags.get((task, episode)),
                "planner_error": err,
                "planned": planned,
                "grasp_label_correct": labels_match(grasp_label, obj_truth, task),
                "place_label_correct": labels_match(place_label, cont_truth, task),
                "truth_labels": [obj_truth, cont_truth],
                "errors": errors,
                "fit": {
                    "n": tf.n,
                    "spread_px": tf.spread_px,
                    "degenerate": tf.degenerate,
                    "floor_obj_m": tf.floor("obj"),
                    "floor_container_m": tf.floor("container"),
                },
                "human": {
                    "tcp_xy": ev.tcp[:, :2].tolist(),
                    "grasp_xy": [float(v) for v in ev.grasp_xyz[:2]],
                    "release_xy": [float(v) for v in ev.release_xyz[:2]],
                    "grasp_z": float(ev.grasp_xyz[2]),
                    "release_z": float(ev.release_xyz[2]),
                    "n_frames": ev.n_frames,
                },
            }
            rows.append(row)
            fmt = lambda v: "unresolved" if v is None else f"{v * 100:5.1f}cm"  # noqa: E731
            print(
                f"  {episode}: grasp={grasp_label!r}({'OK' if row['grasp_label_correct'] else 'WRONG'})"
                f" {fmt(errors['grasp'])}   place={place_label!r}"
                f"({'OK' if row['place_label_correct'] else 'WRONG'}) {fmt(errors['place'])}"
            )

    # report
    fig_label = "oracle ceiling" if args.oracle else "planner dry-run"
    figures = []
    for task in sample:
        path = plot_task(task, [r for r in rows if r["task"] == task], out_dir, fig_label)
        if path:
            figures.append(path)
    summary_fig = plot_summary(rows, out_dir, fig_label)

    title = "ORACLE CEILING (ground-truth labels)" if args.oracle else "PLANNER DRY-RUN"
    print("\n" + "=" * 104)
    print(f"{title} vs RECORDED HUMAN -- table-plane XY. 'floor' = same-role LOO fit residual.")
    print("=" * 104)
    header = (
        f"{'task':38} {'episode':12} {'grasp':6} {'grasp err':11} {'floor':8} "
        f"{'place':6} {'place err':11} {'floor':8}"
    )
    print(header)
    print("-" * len(header))
    fmt = lambda v: "     unres." if v is None else f"{v * 100:8.1f}cm"  # noqa: E731
    ffmt = lambda v: "      --" if v != v else f"{v * 100:6.1f}cm"  # noqa: E731
    for row in rows:
        e = row["errors"]
        print(
            f"{row['task'][:38]:38} {row['episode']:12} "
            f"{'OK' if row['grasp_label_correct'] else 'WRONG':6} {fmt(e['grasp']):11} "
            f"{ffmt(e['grasp_floor']):8} "
            f"{'OK' if row['place_label_correct'] else 'WRONG':6} {fmt(e['place']):11} "
            f"{ffmt(e['place_floor']):8}"
        )

    print("-" * len(header))
    n = len(rows)
    print(f"episodes                     : {n} (all metadata success=true)")
    for key, role in (("grasp", "obj"), ("place", "container")):
        ok = sum(1 for r in rows if r[f"{key}_label_correct"])
        vals = [r["errors"][key] for r in rows if r["errors"][key] is not None]
        fl = [r["errors"][f"{key}_floor"] for r in rows if r["errors"][f"{key}_floor"] == r["errors"][f"{key}_floor"]]
        line = f"{key:6} label correct         : {ok}/{n}"
        if vals:
            line += (
                f"   |  resolved n={len(vals)} median={np.median(vals) * 100:.1f}cm "
                f"mean={np.mean(vals) * 100:.1f}cm max={np.max(vals) * 100:.1f}cm"
            )
        if fl:
            line += f"   |  floor median={np.median(fl) * 100:.1f}cm"
        print(line)
    for key in ("grasp", "place"):
        why = [
            r["planned"].get(f"{key}_unresolved_reason")
            for r in rows
            if r["errors"][key] is None
        ]
        if why:
            print(f"{key:6} unresolved            : {len(why)}")
            for reason in sorted(set(w for w in why if w)):
                print(f"        - {reason}")
    print(f"gemini calls made            : {len(call_log)}")

    payload = {
        "corpus": str(root),
        "mask_cache": str(cache),
        "vocabulary": VOCABULARY,
        "mode": "oracle" if args.oracle else "planner",
        # Without this a --reuse rerun writes gemini_calls=[] and reads as if
        # the plans had cost nothing to obtain.
        "plans_reused_from_disk": bool(args.reuse),
        "model": None if planner is None else planner.model,
        "temperature": None if planner is None else planner.temperature,
        "gemini_calls": call_log,
        "metadata_success": {
            "cached_episodes": len(flags),
            "flagged_success": sum(1 for v in flags.values() if v is True),
            "excluded": {t: eps for t, eps in dropped.items() if eps},
        },
        "rows": rows,
        "figures": [str(p) for p in figures] + ([str(summary_fig)] if summary_fig else []),
    }
    (out_dir / "results.json").write_text(json.dumps(payload, indent=1))
    print(f"\nwrote {out_dir / 'results.json'}")
    for path in figures + ([summary_fig] if summary_fig else []):
        print(f"wrote {path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

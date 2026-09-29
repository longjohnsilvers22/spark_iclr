"""Offline dry-run: the planner plans the human's own scene, scored against the
human -- and against the baseline that says whether the score means anything.

READ-ONLY over the recorded teleop corpus. No robot, no camera, no capture, no
live scene, no endpoint. The only side effects are Gemini calls (one per sampled
episode, every one logged and counted) and files under ``--out``.

    PYTHONPATH=src SPARK_HUMAN_EPISODES=/path/to/teleop_episodes \
    conda run -n spark_conda python scripts/planner_vs_human_calibrated.py \
        --episodes-per-task 3 --out output/planner_vs_human_calibrated/planner

WHAT IT DOES
    Per sampled successful episode: feed ``SPARKPlanner.generate_score`` the
    FIRST camera_0 frame (initial scene -- arm out of shot) plus the recorded
    task string, with the WHOLE corpus vocabulary as the candidate keypoint set,
    so the planner picks an object AND a container out of ten labels rather than
    out of a two-item giveaway. Then compare the plan's grasp/place targets with
    what the human did in that same episode, from ``trajectory.npz``.

THREE NUMBERS, ALWAYS SIDE BY SIDE
    planner   distance from the plan's target to the human's actual grasp /
              release point, in the table plane.
    human     what a planner that never looked at the image would score:
              predict this task's MEAN human grasp point, this episode held out.
              Object XY is re-randomised every episode, so this is large. A
              planner error only means something when read against it.
    floor     the measuring rig's own leave-one-out residual. A plan naming the
              CORRECT label still lands here. Nothing can score below it.

    A planner that beats ``human`` is demonstrably reading the scene. A planner
    at or above ``human`` is indistinguishable from a constant guess, whatever
    the absolute centimetres look like.

WHAT THE METRE NUMBER IS AND IS NOT
    The corpus is RGB-only: no depth, no calibration. A plan names LABELS, so a
    metric comparison needs a label -> base-frame map and the corpus ships none.
    ``spark_real.tests.human_table_fit`` fits one per task, leave-one-episode-
    out, from correspondences the corpus already contains. Consequences:
      * XY only. Z is not recoverable from a planar fit.
      * The distance is therefore a LABEL CHOICE scored in metres: right label
        -> the floor; wrong label -> the distance between the two objects.
      * Only labels this episode has a cached mask for can be resolved. A plan
        naming some third object is ``unresolved``, never silently scored.
    ``--oracle`` substitutes ground-truth labels and calls no LLM: that pass is
    the apparatus ceiling. Read any planner number against it, never alone.

A planner call that fails, or returns a plan with no usable target, is a RESULT
and is counted in the outcome census -- it is never allowed to abort the run.
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

import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
from PIL import Image  # noqa: E402

from spark_real.planning.spark_planner import SPARKPlanner  # noqa: E402
from spark_real.tests.human_corpus import (  # noqa: E402
    TASK_PROMPTS,
    VOCABULARY,
    corpus_root,
    episode_events,
    episode_success,
    labels_match,
    load_mask_index,
)
from spark_real.tests.human_table_fit import (  # noqa: E402
    FIT_CAMERA,
    TableFit,
    build_correspondences,
    centroid_table,
    human_spread,
    plan_targets,
    resolve_centroid,
    sign_test,
)

PLAN_CAMERA = "camera_0"

# Outcome census buckets. Everything that is not SCORED is a planner or
# apparatus result that is reported, not an exception that is raised.
OUTCOME_SCORED = "scored"
OUTCOME_CALL_FAILED = "planner call failed"
OUTCOME_NO_LABEL = "plan named no target for this role"
OUTCOME_UNRESOLVED = "apparatus could not place the named label"


# ---------------------------------------------------------------------------
# sampling
# ---------------------------------------------------------------------------


def success_flags(root: Path, records: List[dict]) -> Dict[Tuple[str, str], Optional[bool]]:
    return {
        (r["task"], r["episode"]): episode_success(root / r["task"] / r["episode"]) for r in records
    }


def sample_episodes(
    records: List[dict], per_task: int, flags: Dict[Tuple[str, str], Optional[bool]]
) -> Tuple[Dict[str, List[str]], Dict[str, List[str]]]:
    """Success episodes with at least ONE resolvable centroid in camera_0.

    Deliberately weaker than requiring both: SAM3 finds the bin in only one
    ``put the pen in the bin`` episode, and dropping the task entirely would
    hide a whole seventh of the corpus. Such an episode still scores its grasp;
    its place lands in the unresolved census with the reason attached.
    """
    have_obj: Dict[str, set] = defaultdict(set)
    have_cont: Dict[str, set] = defaultdict(set)
    dropped: Dict[str, List[str]] = {}
    for rec in records:
        if rec["camera"] != FIT_CAMERA:
            continue
        if rec["phase"] == "negative" and rec.get("obj_centroid"):
            have_obj[rec["task"]].add(rec["episode"])
        if rec["phase"] == "positive" and rec.get("container_centroid"):
            have_cont[rec["task"]].add(rec["episode"])
    out: Dict[str, List[str]] = {}
    for task in TASK_PROMPTS:
        eps = sorted(have_obj[task] | have_cont[task])
        good = [e for e in eps if flags.get((task, e)) is True]
        dropped[task] = [e for e in eps if flags.get((task, e)) is not True]
        # Episodes that resolve BOTH roles first: they spend the same Gemini
        # call on two scored cells instead of one.
        good.sort(key=lambda e: (e not in have_obj[task] or e not in have_cont[task], e))
        out[task] = good[:per_task]
    return out, dropped


# ---------------------------------------------------------------------------
# the one network call
# ---------------------------------------------------------------------------


def run_planner(planner, instruction: str, image, call_log: list) -> Tuple[Optional[dict], str]:
    """One Gemini call. Logged with wall time and outcome; never raises."""
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


def _episode_panel(ax, row: dict) -> None:
    """Human TCP path + events for one episode, with the planned targets on top."""
    path = np.asarray(row["human"]["tcp_xy"])
    ax.plot(path[:, 0], path[:, 1], "-", color="0.6", lw=1.2, label="human TCP path")
    gx, gy = row["human"]["grasp_xy"]
    rx, ry = row["human"]["release_xy"]
    ax.plot(gx, gy, "o", color="#1f77b4", ms=11, mfc="none", mew=2, label="human grasp")
    ax.plot(rx, ry, "s", color="#2ca02c", ms=11, mfc="none", mew=2, label="human release")
    for key, colour, marker, name in (
        ("grasp", "#1f77b4", "x", "planned grasp target"),
        ("place", "#2ca02c", "+", "planned place target"),
    ):
        xy = row["planned"].get(f"{key}_xy")
        anchor = (gx, gy) if key == "grasp" else (rx, ry)
        if xy is not None:
            ax.plot(xy[0], xy[1], marker, color=colour, ms=13, mew=2.5, label=name)
            ax.plot([anchor[0], xy[0]], [anchor[1], xy[1]], ":", color=colour, lw=1.4)
        base = row["baseline"].get(f"{key}_mean_xy")
        if base is not None:
            ax.plot(
                base[0],
                base[1],
                "v",
                color="#d62728" if key == "grasp" else "#9467bd",
                ms=9,
                mfc="none",
                mew=1.8,
                label=f"task-mean {key} (blind baseline)",
            )
            ax.plot([anchor[0], base[0]], [anchor[1], base[1]], "--", color="0.75", lw=1.0)
    bits = []
    for key in ("grasp", "place"):
        err = row["errors"].get(key)
        base = row["baseline"].get(key)
        floor = row["errors"].get(f"{key}_floor")
        piece = f"{key}: " + ("unres." if err is None else f"{err * 100:.1f}cm")
        if base is not None:
            piece += f" vs blind {base * 100:.0f}cm"
        if floor == floor:
            piece += f" (floor {floor * 100:.1f})"
        bits.append(piece)
    ax.set_title(f"{row['episode']}\n" + "\n".join(bits), fontsize=8)
    ax.set_xlabel("robot base X (m)")
    ax.set_ylabel("robot base Y (m)")
    ax.set_aspect("equal", adjustable="datalim")
    ax.grid(alpha=0.3)


def _calibration_panel(ax, task: str, rows: List[dict], spread) -> None:
    """The whole argument in one axes: planner error against the blind baseline."""
    groups = []
    # One colour per SERIES, not per role -- the x groups already separate the
    # roles, and re-colouring the planner bar makes the legend read as a lie.
    for key, colour in (("grasp", "#1f77b4"), ("place", "#1f77b4")):
        planner_v = [r["errors"][key] for r in rows if r["errors"].get(key) is not None]
        blind_v = [r["baseline"][key] for r in rows if r["baseline"].get(key) is not None]
        floor_v = [
            r["errors"][f"{key}_floor"]
            for r in rows
            if r["errors"].get(f"{key}_floor") == r["errors"].get(f"{key}_floor")
        ]
        groups.append((key, colour, planner_v, blind_v, floor_v))
    width = 0.26
    for i, (key, colour, planner_v, blind_v, floor_v) in enumerate(groups):
        for j, (vals, face, name) in enumerate(
            (
                (floor_v, "0.75", "apparatus floor"),
                (planner_v, colour, "planner"),
                (blind_v, "#d62728", "blind task-mean"),
            )
        ):
            if not vals:
                continue
            x = i + (j - 1) * width
            ax.bar(
                x,
                float(np.median(vals)) * 100,
                width=width * 0.9,
                color=face,
                label=name if i == 0 else None,
            )
            ax.plot(
                np.full(len(vals), x),
                np.asarray(vals) * 100,
                ".",
                color="0.15",
                ms=5,
                zorder=3,
            )
    # the corpus-wide spread, not just the sampled episodes
    if spread is not None and spread.n > 1:
        ax.axhline(
            spread.rms_radial("grasp") * 100,
            ls="--",
            color="#d62728",
            lw=1.2,
            label=f"human grasp spread, all {spread.n} eps (RMS)",
        )
    ax.set_xticks(range(len(groups)))
    ax.set_xticklabels([g[0] for g in groups])
    ax.set_ylabel("distance to human actual (cm)")
    ax.set_title("median error vs baselines\n(bars=median, dots=episodes)", fontsize=9)
    ax.grid(alpha=0.3, axis="y")
    ax.set_ylim(0, ax.get_ylim()[1] * 1.45)  # headroom so the legend clears the bars
    ax.legend(fontsize=6.5, loc="upper left", framealpha=0.92)


def plot_task(task: str, rows: List[dict], spread, out_dir: Path, label: str) -> Optional[Path]:
    plotted = [r for r in rows if r.get("human") is not None]
    if not plotted:
        return None
    ncol = len(plotted) + 1
    fig, axes = plt.subplots(1, ncol, figsize=(4.5 * ncol, 5.0), squeeze=False)
    for ax, row in zip(axes[0], plotted):
        _episode_panel(ax, row)
    _calibration_panel(axes[0][-1], task, plotted, spread)
    axes[0][0].legend(fontsize=6.5, loc="best")
    fig.suptitle(
        f'"{task}"   {label} vs recorded human   (table-plane XY, robot base frame, camera_0)',
        fontsize=11,
    )
    fig.tight_layout(rect=(0, 0, 1, 0.94))
    path = out_dir / f"{task.replace(' ', '_')}.png"
    fig.savefig(path, dpi=130)
    plt.close(fig)
    return path


def plot_summary(rows: List[dict], spreads: Dict[str, object], out_dir: Path, label: str):
    tasks = sorted({r["task"] for r in rows})
    if not tasks:
        return None
    fig, axes = plt.subplots(1, 2, figsize=(15, 5.4))
    for ax, key in zip(axes, ("grasp", "place")):
        xpos = {t: i for i, t in enumerate(tasks)}
        drew_any = False
        for name, colour, off, getter in (
            ("planner", "#1f77b4", -0.16, lambda r: r["errors"].get(key)),
            ("blind task-mean baseline", "#d62728", 0.16, lambda r: r["baseline"].get(key)),
        ):
            xs, ys = [], []
            for r in rows:
                val = getter(r)
                if val is None:
                    continue
                xs.append(xpos[r["task"]] + off)
                ys.append(val * 100)
            if xs:
                drew_any = True
                ax.plot(xs, ys, "o", color=colour, ms=8, alpha=0.75, label=name)
        floors = [
            r["errors"][f"{key}_floor"] * 100
            for r in rows
            if r["errors"].get(f"{key}_floor") == r["errors"].get(f"{key}_floor")
        ]
        for t, i in xpos.items():
            sp = spreads.get(t)
            if sp is None or sp.n < 2:
                continue
            role = "grasp" if key == "grasp" else "release"
            ax.plot(
                [i - 0.3, i + 0.3],
                [sp.rms_radial(role) * 100] * 2,
                "-",
                color="#d62728",
                lw=2.5,
                alpha=0.45,
                label="human spread, all episodes (RMS)" if i == 0 else None,
            )
        if floors:
            ax.axhline(
                float(np.median(floors)),
                ls="--",
                color="0.4",
                label=f"apparatus floor (median {np.median(floors):.1f} cm)",
            )
        ax.set_xticks(range(len(tasks)))
        ax.set_xticklabels(
            [textwrap.fill(t, 16) for t in tasks], fontsize=7, rotation=30, ha="right"
        )
        ax.set_xlim(-0.6, len(tasks) - 0.4)
        ax.set_ylabel("distance to human actual (cm, table plane)")
        ax.set_title(f"{key} target")
        ax.grid(alpha=0.3, axis="y")
        ax.set_ylim(0, ax.get_ylim()[1] * 1.28)  # headroom so the legend clears the points
        if drew_any:
            ax.legend(fontsize=8, loc="upper right", framealpha=0.92)
    fig.suptitle(
        f"{label} vs recorded human, per episode -- calibrated against human variance",
        fontsize=12,
    )
    fig.tight_layout(rect=(0, 0, 1, 0.93))
    path = out_dir / "summary.png"
    fig.savefig(path, dpi=130)
    plt.close(fig)
    return path


# ---------------------------------------------------------------------------


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--root", default=None, help="corpus root; defaults to $SPARK_HUMAN_EPISODES")
    ap.add_argument("--cache", default="output/pvh_mask_cache", help="offline SAM3 mask cache")
    ap.add_argument("--out", default="output/planner_vs_human_calibrated")
    ap.add_argument("--episodes-per-task", type=int, default=3)
    ap.add_argument("--model", default=None)
    ap.add_argument(
        "--temperature",
        type=float,
        default=0.0,
        help="0.0 for a reproducible dry-run. The LIVE rig plans at 0.3.",
    )
    ap.add_argument("--reuse", action="store_true", help="reuse plans saved under --out")
    ap.add_argument(
        "--oracle",
        action="store_true",
        help="APPARATUS CEILING: ground-truth labels, no Gemini call.",
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
            "spark_real.tests.verify_replay_harvest --episodes-per-task 20"
        )
    out_dir = Path(args.out)
    (out_dir / "plans").mkdir(parents=True, exist_ok=True)

    records = load_mask_index(cache)
    flags = success_flags(root, records)
    sample, dropped = sample_episodes(records, args.episodes_per_task, flags)
    fit = TableFit(build_correspondences(root, records, flags))
    spreads = {task: human_spread(root, task) for task in TASK_PROMPTS}

    print(f"corpus              : {root}")
    print(f"mask cache          : {cache}  ({len(records)} records)")
    print(f"candidate labels    : {VOCABULARY}")
    print("\nHUMAN VARIANCE BASELINE -- measured over EVERY successful episode, not the sample")
    head = (
        f"{'task':38} {'n':>4} {'grasp RMS':>10} {'blind med':>10} {'grasp Y ptp':>12} "
        f"{'grasp z std':>12} {'release RMS':>12} {'blind med':>10}"
    )
    print(head)
    print("-" * len(head))
    for task in TASK_PROMPTS:
        sp = spreads[task]
        if sp.n < 2:
            print(f"{task[:38]:38} {sp.n:>4}  (too few episodes)")
            continue
        print(
            f"{task[:38]:38} {sp.n:>4} {sp.rms_radial('grasp') * 100:>9.1f}cm "
            f"{np.median(sp.loo_errors('grasp')) * 100:>9.1f}cm "
            f"{sp.ptp('grasp')[1] * 100:>11.1f}cm {sp.z_std('grasp') * 1000:>11.1f}mm "
            f"{sp.rms_radial('release') * 100:>11.1f}cm "
            f"{np.median(sp.loo_errors('release')) * 100:>9.1f}cm"
        )

    print("\nimage->table fit    : per task, leave-one-EPISODE-out, success episodes only")
    for task in sorted(sample):
        if not sample[task]:
            print(f"    {task[:38]:38} sample=0 (no cached camera_0 centroid)")
            continue
        tf = fit.for_episode(task, sample[task][0])
        print(
            f"    {task[:38]:38} sample={len(sample[task])} fit_n={tf.n} "
            f"spread={tf.spread_px:.1f}px"
            f"{'  DEGENERATE' if tf.degenerate else ''}"
            f"  floor obj={tf.floor('obj') * 100:.1f}cm cont={tf.floor('container') * 100:.1f}cm"
        )
    for task, eps in dropped.items():
        if eps:
            print(f"    EXCLUDED (metadata success is not true): {task} -> {eps}")

    n_calls = 0 if (args.reuse or args.oracle) else sum(len(v) for v in sample.values())
    mode = "ORACLE (ground-truth labels)" if args.oracle else "planner"
    print(f"\nmode                : {mode}")
    print(f"planned gemini calls: {n_calls}\n")

    # Oracle mode never calls out, so it must not even need a key.
    planner = None if args.oracle else SPARKPlanner(model=args.model, temperature=args.temperature)
    call_log: List[dict] = []
    rows: List[dict] = []

    for task, episodes in sample.items():
        obj_truth, cont_truth, _ = TASK_PROMPTS[task]
        if episodes:
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
                score, err = None, ""
                grasp_label, place_label, types = obj_truth, cont_truth, ["oracle"]
            else:
                if args.reuse and plan_path.exists():
                    score, err = json.loads(plan_path.read_text())["score"], ""
                else:
                    with Image.open(frame) as img:
                        score, err = run_planner(planner, task, img.convert("RGB"), call_log)
                    plan_path.write_text(
                        json.dumps({"task": task, "error": err, "score": score}, indent=1)
                    )
                grasp_label, place_label, types = plan_targets(score)

            tf = fit.for_episode(task, episode)
            table = centroid_table(records, task, episode)
            sp = spreads[task]
            errors: Dict[str, Optional[float]] = {}
            baseline: Dict[str, object] = {}
            outcome: Dict[str, str] = {}
            planned: Dict[str, object] = {
                "grasp_label": grasp_label,
                "place_label": place_label,
                "primitives": types,
            }
            for key, label, role, sp_role, truth_xy in (
                ("grasp", grasp_label, "obj", "grasp", ev.grasp_xyz[:2]),
                ("place", place_label, "container", "release", ev.release_xyz[:2]),
            ):
                centroid = resolve_centroid(label, table, task)
                if centroid is None:
                    xy, why = None, f"no cached mask for {label!r} in this episode"
                else:
                    xy, why = tf.solve(centroid)
                planned[f"{key}_xy"] = None if xy is None else [float(v) for v in xy]
                planned[f"{key}_unresolved_reason"] = None if xy is not None else why
                errors[key] = None if xy is None else float(np.linalg.norm(xy - truth_xy))
                errors[f"{key}_floor"] = tf.floor(role)
                baseline[key] = sp.loo_error(sp_role, episode)
                pts = sp.grasp_xyz if sp_role == "grasp" else sp.release_xyz
                baseline[f"{key}_mean_xy"] = (
                    [float(v) for v in pts[:, :2].mean(0)] if len(pts) > 1 else None
                )
                if err:
                    outcome[key] = OUTCOME_CALL_FAILED
                elif not label:
                    outcome[key] = OUTCOME_NO_LABEL
                elif xy is None:
                    outcome[key] = OUTCOME_UNRESOLVED
                else:
                    outcome[key] = OUTCOME_SCORED

            rows.append(
                {
                    "task": task,
                    "episode": episode,
                    "metadata_success": flags.get((task, episode)),
                    "planner_error": err,
                    "planned": planned,
                    "outcome": outcome,
                    "grasp_label_correct": labels_match(grasp_label, obj_truth, task),
                    "place_label_correct": labels_match(place_label, cont_truth, task),
                    "truth_labels": [obj_truth, cont_truth],
                    "errors": errors,
                    "baseline": baseline,
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
            )
            fmt = lambda v: "unresolved" if v is None else f"{v * 100:5.1f}cm"  # noqa: E731
            print(
                f"  {episode}: grasp={grasp_label!r}"
                f"({'OK' if rows[-1]['grasp_label_correct'] else 'WRONG'}) {fmt(errors['grasp'])}"
                f" [blind {fmt(baseline['grasp'])}]   place={place_label!r}"
                f"({'OK' if rows[-1]['place_label_correct'] else 'WRONG'}) {fmt(errors['place'])}"
                f" [blind {fmt(baseline['place'])}]"
            )

    # ---------------- report ----------------
    fig_label = "oracle ceiling" if args.oracle else "planner dry-run"
    figures = []
    for task in sample:
        path = plot_task(
            task, [r for r in rows if r["task"] == task], spreads.get(task), out_dir, fig_label
        )
        if path:
            figures.append(path)
    summary_fig = plot_summary(rows, spreads, out_dir, fig_label)

    title = "ORACLE CEILING (ground-truth labels)" if args.oracle else "PLANNER DRY-RUN"
    print("\n" + "=" * 122)
    print(f"{title} vs RECORDED HUMAN -- table-plane XY, robot base frame.")
    print(
        "  planner = plan target -> human actual   blind = task-mean predictor (LOO)"
        "   floor = apparatus LOO residual"
    )
    print("=" * 122)
    header = (
        f"{'task':36} {'episode':12} "
        f"{'g?':4}{'g planner':>11}{'g blind':>10}{'g floor':>9}  "
        f"{'p?':4}{'p planner':>11}{'p blind':>10}{'p floor':>9}"
    )
    print(header)
    print("-" * len(header))
    fmt = lambda v: "    unres." if v is None else f"{v * 100:8.1f}cm"  # noqa: E731
    bfmt = lambda v: "      --" if v is None else f"{v * 100:7.1f}cm"  # noqa: E731
    ffmt = lambda v: "      --" if v != v else f"{v * 100:6.1f}cm"  # noqa: E731
    for row in rows:
        e, b = row["errors"], row["baseline"]
        print(
            f"{row['task'][:36]:36} {row['episode']:12} "
            f"{'OK' if row['grasp_label_correct'] else 'BAD':4}{fmt(e['grasp']):>11}"
            f"{bfmt(b['grasp']):>10}{ffmt(e['grasp_floor']):>9}  "
            f"{'OK' if row['place_label_correct'] else 'BAD':4}{fmt(e['place']):>11}"
            f"{bfmt(b['place']):>10}{ffmt(e['place_floor']):>9}"
        )

    print("-" * len(header))
    n = len(rows)
    print(f"episodes                 : {n} (all metadata success=true)")
    verdicts = {}
    sampled_tasks = sorted({r["task"] for r in rows})
    for key, role, sp_role in (("grasp", "obj", "grasp"), ("place", "container", "release")):
        ok = sum(1 for r in rows if r[f"{key}_label_correct"])
        paired = [
            (r["errors"][key], r["baseline"][key])
            for r in rows
            if r["errors"].get(key) is not None and r["baseline"].get(key) is not None
        ]
        vals = [r["errors"][key] for r in rows if r["errors"].get(key) is not None]
        fl = [
            r["errors"][f"{key}_floor"]
            for r in rows
            if r["errors"].get(f"{key}_floor") == r["errors"].get(f"{key}_floor")
        ]
        print(f"{key:6} label correct     : {ok}/{n}")
        if vals:
            print(
                f"{key:6} planner error     : n={len(vals)} median={np.median(vals) * 100:.1f}cm "
                f"mean={np.mean(vals) * 100:.1f}cm max={np.max(vals) * 100:.1f}cm"
            )
        if fl:
            print(f"{key:6} apparatus floor   : median={np.median(fl) * 100:.1f}cm")
        if paired:
            pv = [p[0] for p in paired]
            bv = [p[1] for p in paired]
            wins, ncmp, pval = sign_test(pv, bv)
            ratio = float(np.median(bv)) / max(float(np.median(pv)), 1e-9)
            # Same baseline over EVERY episode of the sampled tasks. If it is
            # bigger than the paired figure, the sampled episodes happened to
            # sit near their task mean and the comparison was harder than
            # typical, not easier.
            corpus_b = np.concatenate(
                [spreads[t].loo_errors(sp_role) for t in sampled_tasks if spreads[t].n > 1]
            )
            verdicts[key] = {
                "n_paired": ncmp,
                "planner_beats_blind": wins,
                "median_planner_cm": float(np.median(pv)) * 100,
                "median_blind_cm": float(np.median(bv)) * 100,
                "median_blind_all_episodes_cm": float(np.median(corpus_b)) * 100,
                "ratio": ratio,
                "sign_test_p": pval,
            }
            print(
                f"{key:6} BLIND baseline    : median={np.median(bv) * 100:.1f}cm on these "
                f"episodes ({np.median(corpus_b) * 100:.1f}cm over all "
                f"{len(corpus_b)} episodes of these tasks)"
            )
            print(
                f"{key:6} VERDICT           : planner is {ratio:.1f}x closer than blind; "
                f"beats blind on {wins}/{ncmp} episodes, sign-test p={pval:.2g}"
            )
    census: Dict[str, Dict[str, int]] = {"grasp": defaultdict(int), "place": defaultdict(int)}
    for r in rows:
        for key in ("grasp", "place"):
            census[key][r["outcome"][key]] += 1
    print("outcome census           :")
    for key in ("grasp", "place"):
        parts = ", ".join(f"{v} {k}" for k, v in sorted(census[key].items()))
        print(f"    {key:6} {parts}")
    for key in ("grasp", "place"):
        why = [
            r["planned"].get(f"{key}_unresolved_reason")
            for r in rows
            if r["errors"].get(key) is None
        ]
        for reason in sorted(set(w for w in why if w)):
            print(f"    {key:6} unresolved because: {reason}")
    # One task names both roles "block", so the centroid table has a single key
    # and grasp and place resolve to the SAME point. Its numbers are an upper
    # bound on the apparatus, not a planner measurement; say so rather than
    # letting the row read like the others.
    ambiguous = sorted({t for t in sampled_tasks if TASK_PROMPTS[t][0] == TASK_PROMPTS[t][1]})
    for task in ambiguous:
        print(
            f"CAVEAT                   : '{task}' labels both roles "
            f"{TASK_PROMPTS[task][0]!r}; the apparatus cannot tell the two apart, "
            f"so its rows measure the rig (floor ~{fit.for_episode(task, sample[task][0]).floor('obj') * 100:.0f}cm), not the plan"
        )
    n_failed = sum(1 for r in rows if r["planner_error"])
    print(f"planner calls that raised: {n_failed}/{n}")
    print(f"gemini calls made        : {len(call_log)}")

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
        "verdicts": verdicts,
        "ambiguous_tasks": ambiguous,
        "outcome_census": {k: dict(v) for k, v in census.items()},
        "human_variance": {
            task: {
                "n": sp.n,
                "grasp_rms_radial_m": sp.rms_radial("grasp"),
                "release_rms_radial_m": sp.rms_radial("release"),
                "grasp_ptp_m": [float(v) for v in sp.ptp("grasp")],
                "grasp_z_std_m": sp.z_std("grasp"),
                "grasp_mean_xyz": ([float(v) for v in sp.grasp_xyz.mean(0)] if sp.n else None),
            }
            for task, sp in spreads.items()
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

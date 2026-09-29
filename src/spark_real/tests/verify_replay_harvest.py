"""Offline SAM3 harvest for the corpus-replay verification test.

READ-ONLY over the recorded human corpus: no robot, no camera, no live scene.
SAM3 lives in the `sam3` env (not spark_conda), so this is a standalone script
rather than part of the test:

    export SPARK_HUMAN_EPISODES=/path/to/teleop_episodes
    PYTHONPATH=src conda run -n sam3 python -m spark_real.tests.verify_replay_harvest \
        --episodes-per-task 9 --out output/verify_replay_cache

The corpus root comes from ``$SPARK_HUMAN_EPISODES`` or ``--root``. There is no
default: this package ships publicly and the corpus path is rig-local.

It writes masks_<n>.npz + index.json; test_verify_replay.py replays them under
spark_conda with no GPU.

Two frames per episode per camera:
  positive  the last frame (arm stationary after release)
  negative  a few frames before the gripper first closes -- the object is
            provably still on the table, so any `inside` pass is a false one.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

import numpy as np
from PIL import Image

from sam3.model.sam3_image_processor import Sam3Processor
from sam3.model_builder import build_sam3_image_model

# Corpus vocabulary + event extraction are shared with the planner-vs-human
# comparison; keeping one copy stops the two from drifting apart.
from spark_real.tests.human_corpus import (
    CAMERAS,
    TASK_PROMPTS,
    episode_events,
    iter_episodes,
)

DOWNSAMPLE = 2


def frame_indices(ep_dir: Path):
    ev = episode_events(ep_dir)
    if ev is None:
        return None
    return {"positive": ev.n_frames - 1, "negative": ev.pregrasp_index}


def run_prompt(proc, img, prompt, rank=0):
    """Best (rank=0) or second-best (rank=1) mask for a text prompt."""
    state = proc.set_image(img)
    state = proc.set_text_prompt(prompt=prompt, state=state)
    masks = state.get("masks")
    scores = state.get("scores")
    if masks is None or masks.numel() == 0 or len(scores) <= rank:
        return 0.0, None
    i = int(scores.argsort(descending=True)[rank])
    mask = masks[i].detach().cpu().numpy().squeeze().astype(bool)
    return float(scores[i]), mask[::DOWNSAMPLE, ::DOWNSAMPLE]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument(
        "--root",
        default=os.environ.get("SPARK_HUMAN_EPISODES") or None,
        help="human corpus root; defaults to $SPARK_HUMAN_EPISODES, never to a path in code",
    )
    ap.add_argument("--episodes-per-task", type=int, default=9)
    ap.add_argument("--out", default="output/verify_replay_cache")
    ap.add_argument(
        "--cross-task",
        action="store_true",
        help="harvest ONLY cross-task negatives: task A's final frame judged "
        "with task B's prompts and predicate",
    )
    args = ap.parse_args()

    if not args.root:
        sys.exit(
            "no corpus root: set SPARK_HUMAN_EPISODES=/path/to/teleop_episodes or pass --root"
        )
    root = Path(args.root)
    if not root.is_dir():
        sys.exit(f"corpus root {root} does not exist")
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)

    model = build_sam3_image_model()
    proc = Sam3Processor(model)

    index, arrays = [], {}
    names = list(TASK_PROMPTS)
    for task, (obj_p, cont_p, pred) in TASK_PROMPTS.items():
        tdir = root / task
        if not tdir.is_dir():
            continue
        if args.cross_task:
            # judge this task's final frame with the NEXT task's goal
            other = names[(names.index(task) + 1) % len(names)]
            obj_p, cont_p, pred = TASK_PROMPTS[other]
            if TASK_PROMPTS[task][1] == cont_p:  # same container, not a negative
                other = names[(names.index(task) + 2) % len(names)]
                obj_p, cont_p, pred = TASK_PROMPTS[other]
            print(f"cross: {task} judged as '{other}'")
        for ep in iter_episodes(root, task, args.episodes_per_task):
            idxs = frame_indices(ep)
            if idxs is None:
                print("skip (no post-release frames):", task, ep.name)
                continue
            if args.cross_task:
                idxs = {"cross": idxs["positive"]}
            for phase, fi in idxs.items():
                for cam in CAMERAS:
                    jpg = ep / "images" / cam / f"frame_{fi:04d}.jpg"
                    if not jpg.exists():
                        continue
                    img = Image.open(jpg).convert("RGB")
                    rec = {
                        "task": task,
                        "episode": ep.name,
                        "camera": cam,
                        "phase": phase,
                        "frame": fi,
                        "predicate": pred,
                        "obj_prompt": obj_p,
                        "container_prompt": cont_p,
                    }
                    # same prompt for both roles (same-colour stack): the
                    # container is the second-best instance, not the same mask.
                    same = obj_p == cont_p
                    for role, prompt, rank in (
                        ("obj", obj_p, 0),
                        ("container", cont_p, 1 if same else 0),
                    ):
                        score, mask = run_prompt(proc, img, prompt, rank=rank)
                        rec[f"{role}_score"] = score
                        if mask is None:
                            rec[f"{role}_key"] = None
                            continue
                        key = f"{len(arrays)}"
                        arrays[key] = np.packbits(mask)
                        rec[f"{role}_key"] = key
                        rec["shape"] = list(mask.shape)
                    index.append(rec)
            print(f"{task} {ep.name}: {len(index)} records")

    np.savez_compressed(out / "masks.npz", **arrays)
    (out / "index.json").write_text(json.dumps(index, indent=1))
    print(f"wrote {len(index)} records, {len(arrays)} masks to {out}")


if __name__ == "__main__":
    main()

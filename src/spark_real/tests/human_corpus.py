"""Shared read-only accessors for the recorded human teleop corpus.

No robot, no camera, no GPU: numpy + stdlib only, so this imports cleanly in
the ``sam3`` env (for ``verify_replay_harvest``) and in ``spark_conda`` (for
the replay test and the planner-vs-human comparison).

The corpus root is ``$SPARK_HUMAN_EPISODES``. There is NO hardcoded default --
this package ships publicly and the corpus lives on one rig.

Corpus layout, per episode dir::

    images/{camera_0,camera_1,wrist}/frame_NNNN.jpg
    trajectory.npz   tcp_poses (N,6) base frame, gripper_positions (N,)
    metadata.json    {"task": ..., "success": ..., "num_frames": ...}

``gripper_positions`` is the *commanded* trigger (bit-identical to
``actions[:,6]``), so it indexes phases; it is never grasp evidence.
"""

from __future__ import annotations

import json
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np

CORPUS_ENV = "SPARK_HUMAN_EPISODES"

# task -> (object prompt, container prompt, predicate)
TASK_PROMPTS: Dict[str, Tuple[str, str, str]] = {
    "put the knife in the tray": ("knife", "tray", "inside"),
    "put the spoon in the tray": ("spoon", "tray", "inside"),
    "put the pen in the bin": ("pen", "bin", "inside"),
    "pick up the plushie and place in bowl": ("plushie", "bowl", "inside"),
    "stack the blue block on the gray block": ("blue block", "gray block", "on"),
    "stack the gray block on the blue block": ("gray block", "blue block", "on"),
    "stack the blocks of same color": ("block", "block", "on"),
}

CAMERAS = ("camera_0", "camera_1", "wrist")

# Every label the corpus uses, deduped. Handed to the planner as the candidate
# keypoint set so it has to pick the right object AND the right container out
# of the whole vocabulary, not out of a two-item giveaway.
VOCABULARY: List[str] = sorted({p for triple in TASK_PROMPTS.values() for p in triple[:2]})

# The stack-same-colour task labels both roles "block"; any block reads as
# correct there, because the corpus itself does not distinguish them.
BLOCK_ALIASES = {"block", "blue block", "gray block", "grey block"}

GRIP_CLOSED = 0.5  # commanded-trigger threshold


def corpus_root(explicit: Optional[str] = None) -> Optional[Path]:
    """Corpus root from ``--root`` then ``$SPARK_HUMAN_EPISODES``, else None."""
    raw = (explicit or os.environ.get(CORPUS_ENV, "")).strip()
    return Path(raw) if raw else None


@dataclass
class EpisodeEvents:
    """What the human actually did, straight out of ``trajectory.npz``."""

    task: str
    episode: str
    n_frames: int
    tcp: np.ndarray  # (N,6) base frame
    gripper: np.ndarray  # (N,) commanded trigger
    grasp_index: int
    release_index: int
    grasp_xyz: np.ndarray  # TCP at the frame the jaws first close
    release_xyz: np.ndarray  # TCP at the last frame before they open again
    # A few frames before the first close: the object is provably still on the
    # table, so this frame shows its pre-grasp position.
    pregrasp_index: int


def episode_events(ep_dir: Path, pregrasp_lead: int = 3) -> Optional[EpisodeEvents]:
    """Derive grasp/release events, or None when the episode has no clean grip."""
    traj = ep_dir / "trajectory.npz"
    if not traj.exists():
        return None
    data = np.load(traj)
    grip = np.asarray(data["gripper_positions"], dtype=float).reshape(-1)
    tcp = np.asarray(data["tcp_poses"], dtype=float)
    closed = np.where(grip > GRIP_CLOSED)[0]
    n = len(grip)
    # A grip that never opens again means the release was never recorded.
    if not len(closed) or closed[-1] >= n - 1:
        return None
    task = ep_dir.parent.name
    meta = ep_dir / "metadata.json"
    if meta.exists():
        task = json.loads(meta.read_text()).get("task", task) or task
    return EpisodeEvents(
        task=task,
        episode=ep_dir.name,
        n_frames=n,
        tcp=tcp,
        gripper=grip,
        grasp_index=int(closed[0]),
        release_index=int(closed[-1]),
        grasp_xyz=tcp[closed[0], :3].copy(),
        release_xyz=tcp[closed[-1], :3].copy(),
        pregrasp_index=max(0, int(closed[0]) - pregrasp_lead),
    )


def episode_success(ep_dir: Path) -> Optional[bool]:
    """The episode's own ``metadata.json`` success flag; None if it records none."""
    meta = ep_dir / "metadata.json"
    if not meta.exists():
        return None
    try:
        value = json.loads(meta.read_text()).get("success")
    except (OSError, ValueError):
        return None
    return value if isinstance(value, bool) else None


def iter_episodes(root: Path, task: str, limit: Optional[int] = None) -> List[Path]:
    tdir = root / task
    if not tdir.is_dir():
        return []
    eps = sorted(p for p in tdir.iterdir() if p.is_dir())
    return eps[:limit] if limit else eps


def load_mask_index(cache: Path) -> List[dict]:
    """Records from an offline SAM3 harvest, each with a ``centroid`` per role.

    Centroids are in the DOWNSAMPLED mask frame the harvest wrote; that is the
    frame the image->table fit is done in, so no rescale is needed as long as
    both sides come from this loader.
    """
    index_path = cache / "index.json"
    if not index_path.exists():
        return []
    index = json.loads(index_path.read_text())
    packed = np.load(cache / "masks.npz")
    for rec in index:
        for role in ("obj", "container"):
            key = rec.get(f"{role}_key")
            rec[f"{role}_centroid"] = None
            if key is None:
                continue
            shape = tuple(rec["shape"])
            bits = np.unpackbits(packed[key])[: shape[0] * shape[1]]
            mask = bits.reshape(shape).astype(bool)
            ys, xs = np.nonzero(mask)
            if len(xs):
                rec[f"{role}_centroid"] = (float(xs.mean()), float(ys.mean()))
    return index


def labels_match(planned: Optional[str], truth: str, task: str) -> bool:
    """Label comparison that tolerates the corpus's own naming slack."""
    if not planned:
        return False
    a, b = planned.strip().lower(), truth.strip().lower()
    if a == b:
        return True
    if task == "stack the blocks of same color":
        return a in BLOCK_ALIASES and b in BLOCK_ALIASES
    # "knife handle" / "the tray" etc. still name the same object.
    return b in a or a in b


__all__ = [
    "BLOCK_ALIASES",
    "CAMERAS",
    "CORPUS_ENV",
    "EpisodeEvents",
    "GRIP_CLOSED",
    "TASK_PROMPTS",
    "VOCABULARY",
    "corpus_root",
    "episode_events",
    "episode_success",
    "iter_episodes",
    "labels_match",
    "load_mask_index",
]

"""Metric apparatus for comparing a symbolic plan against a recorded human demo.

numpy + stdlib only, like ``human_corpus``: no torch, no matplotlib, no robot.

THE PROBLEM THIS SOLVES
    A SPARK plan names LABELS ("knife", "tray"). The human corpus records METRES
    (TCP poses in the robot base frame). To put the two on one axis you need a
    label -> base-frame position map, and the RGB-only corpus ships none: no
    depth, no camera calibration, no extrinsics.

    Objects sit on one table plane, so a single affine takes a camera_0 image
    point to base-frame XY. ``TableFit`` fits one per task from correspondences
    the corpus already contains -- (object mask centroid just before the grasp,
    human grasp XY) and (container mask centroid just after the release, human
    release XY) -- LEAVE-ONE-EPISODE-OUT, so no episode helps measure itself.

    XY only: a planar fit cannot recover Z. Every distance here is a table-plane
    distance.

THE TWO REFERENCE NUMBERS ANY PLANNER ERROR MUST BE READ AGAINST
    ``TaskFit.floor``      the apparatus' own leave-one-out residual. A plan that
                           names the CORRECT label still lands here; nothing can
                           score better. This is the ceiling of the measurement,
                           not of the planner.
    ``HumanSpread``        what a planner that ignores the image would score, by
                           predicting the task's mean human grasp point every
                           time (leave-one-out). Object XY is re-randomised each
                           episode, so this baseline is large -- and a planner
                           only demonstrates it is READING THE SCENE if it beats
                           it. "Within X cm" is meaningless until you print this
                           beside it.
"""

from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass, field
from math import comb
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np

from spark_real.tests.human_corpus import (
    TASK_PROMPTS,
    episode_events,
    episode_success,
    iter_episodes,
    labels_match,
)

FIT_CAMERA = "camera_0"
GRASP_TYPES = ("grasp", "grasp_se3", "grasp_top_down", "grasp_cgn", "grasp_se3_flow")

MIN_FIT_PAIRS = 5  # below this a per-task affine is not worth trusting
# Correspondences must span this much on their SHORT image axis, else the affine
# is a blob being asked to extrapolate. Containers alone fail this.
MIN_UV_SPREAD_PX = 6.0
# How far past the fitted region a query point may sit before it is refused.
MAX_EXTRAPOLATION = 1.25


# ---------------------------------------------------------------------------
# plan -> (grasp label, place label)
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
    """(grasp label, place label, primitive-type sequence) from a plan.

    ``move_to_keypoint`` is what actually carries the label; the grasp/release
    that follows inherits it. Tracking ``holding`` is how the SECOND
    move_to_keypoint becomes the place target rather than a second grasp.
    """
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
    """Least-squares 2x3 map from image (u,v) to base (x,y)."""
    design = np.hstack([np.asarray(uv, float), np.ones((len(uv), 1))])
    sol, *_ = np.linalg.lstsq(design, np.asarray(xy, float), rcond=None)
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


class TaskFit:
    """One task's affine, fitted on every correspondence EXCEPT one episode's."""

    def __init__(self, rows: Sequence[tuple]):
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
        """Image centroid -> base XY, or why the map is untrustworthy here."""
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


def build_correspondences(
    root: Path, records: List[dict], flags: Dict[Tuple[str, str], Optional[bool]]
) -> Dict[str, list]:
    """(episode, role, image centroid, base XY) per task, from the mask cache."""
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


def centroid_table(records: List[dict], task: str, episode: str) -> Dict[str, tuple]:
    """label -> camera_0 image centroid, for labels this episode can resolve."""
    obj_prompt, cont_prompt, _ = TASK_PROMPTS[task]
    table: Dict[str, tuple] = {}
    for rec in records:
        if rec["camera"] != FIT_CAMERA or rec["task"] != task or rec["episode"] != episode:
            continue
        if rec["phase"] == "negative" and rec.get("obj_centroid"):
            table.setdefault(obj_prompt, rec["obj_centroid"])
        if rec["phase"] == "positive" and rec.get("container_centroid"):
            table[cont_prompt] = rec["container_centroid"]
    return table


def resolve_centroid(label: Optional[str], table: Dict[str, tuple], task: str) -> Optional[tuple]:
    if not label:
        return None
    for known, centroid in table.items():
        if labels_match(label, known, task):
            return centroid
    return None


# ---------------------------------------------------------------------------
# the human-variance baseline
# ---------------------------------------------------------------------------


@dataclass
class HumanSpread:
    """Where the human put their hands across a whole task, and how much it moved.

    ``loo_error`` is the baseline that matters: predict the task's mean grasp
    (or release) point, fitted WITHOUT this episode, and see how far off it is.
    That is the score of a planner that never looks at the picture.
    """

    task: str
    n: int
    grasp_xyz: np.ndarray = field(default_factory=lambda: np.zeros((0, 3)))
    release_xyz: np.ndarray = field(default_factory=lambda: np.zeros((0, 3)))
    episodes: List[str] = field(default_factory=list)

    def _pts(self, role: str) -> np.ndarray:
        return self.grasp_xyz if role == "grasp" else self.release_xyz

    def rms_radial(self, role: str) -> float:
        """RMS distance of each episode from the task mean, table plane."""
        pts = self._pts(role)[:, :2]
        if len(pts) < 2:
            return float("nan")
        return float(np.sqrt(np.mean(np.sum((pts - pts.mean(0)) ** 2, axis=1))))

    def ptp(self, role: str) -> np.ndarray:
        pts = self._pts(role)
        return np.ptp(pts, axis=0) if len(pts) else np.full(3, np.nan)

    def z_std(self, role: str) -> float:
        pts = self._pts(role)
        return float(np.std(pts[:, 2])) if len(pts) > 1 else float("nan")

    def loo_error(self, role: str, episode: str) -> Optional[float]:
        """Task-mean baseline error for one episode, that episode held out."""
        pts = self._pts(role)
        if episode not in self.episodes or len(pts) < 2:
            return None
        i = self.episodes.index(episode)
        keep = np.ones(len(pts), bool)
        keep[i] = False
        return float(np.linalg.norm(pts[keep, :2].mean(0) - pts[i, :2]))

    def loo_errors(self, role: str) -> np.ndarray:
        """The baseline over every episode of the task -- the honest spread."""
        vals = [self.loo_error(role, e) for e in self.episodes]
        return np.asarray([v for v in vals if v is not None])


def human_spread(root: Path, task: str, success_only: bool = True) -> HumanSpread:
    """Grasp/release points over EVERY episode of one task. Reads npz only."""
    grasps, releases, episodes = [], [], []
    for ep_dir in iter_episodes(root, task):
        if success_only and episode_success(ep_dir) is not True:
            continue
        ev = episode_events(ep_dir)
        if ev is None:  # no clean close-then-open; not usable as a reference
            continue
        grasps.append(ev.grasp_xyz)
        releases.append(ev.release_xyz)
        episodes.append(ep_dir.name)
    return HumanSpread(
        task=task,
        n=len(episodes),
        grasp_xyz=np.asarray(grasps) if grasps else np.zeros((0, 3)),
        release_xyz=np.asarray(releases) if releases else np.zeros((0, 3)),
        episodes=episodes,
    )


def sign_test(paired_a: Sequence[float], paired_b: Sequence[float]) -> Tuple[int, int, float]:
    """(wins for a, comparisons, two-sided p) that a < b. Exact binomial, no scipy."""
    wins = sum(1 for a, b in zip(paired_a, paired_b) if a < b)
    n = sum(1 for a, b in zip(paired_a, paired_b) if a != b)
    if n == 0:
        return wins, 0, float("nan")
    # two-sided exact binomial at p=0.5, summing the tail at least as extreme
    k = max(wins, n - wins)
    tail = sum(comb(n, i) for i in range(k, n + 1))
    return wins, n, min(1.0, 2.0 * tail / (2.0**n))


__all__ = [
    "FIT_CAMERA",
    "GRASP_TYPES",
    "HumanSpread",
    "TableFit",
    "TaskFit",
    "apply_affine",
    "build_correspondences",
    "centroid_table",
    "fit_affine",
    "flatten",
    "human_spread",
    "loo_residual",
    "plan_targets",
    "resolve_centroid",
    "sign_test",
]

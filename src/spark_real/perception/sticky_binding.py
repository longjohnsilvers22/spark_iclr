"""
Sticky position-based binding, dedup, self-occlusion hold, and the
pre-primitive scene diff with self-caused-change subtraction.

Shared sim/real home of these mechanisms, so the real pipeline can use
them without importing spark_bench.

Every distance constant is a PARAMETER with the sim-tuned value as its
default; the real rig must re-measure them (calibration notes put
per-camera disagreement at 1-2 cm, so e.g. the sim's 1.5 cm
moved_threshold would flag noise as motion).

Before dispatching each primitive, the executor diffs the current
``det_map`` against the plan-time binding: per-label position delta,
presence lost, and a presence/position re-check for the primitive's own
target label.  Self-caused change is subtracted (CheckVLA's caveat,
arXiv 2607.26789): the currently-held object and anything within a small
radius of the gripper is not evidence of an external scene change.

Diff-before-act is the scene-graph proactive-replanning pattern
(arXiv 2508.11286); the perception cost is already paid by the freshness
gate / event captures - the diff itself is arithmetic.

Pure module (numpy only) so it unit-tests without an env.
"""
from __future__ import annotations

import time
from dataclasses import dataclass, field
from typing import Mapping, Optional

import numpy as np


__all__ = [
    'STATUS_OK', 'STATUS_MOVED', 'STATUS_MISSING', 'STATUS_SELF',
    'STATUS_UNBOUND',
    'LabelDiff', 'SceneDiff', 'fuzzy_key', 'compute_scene_diff',
    'StickyResult', 'sticky_associate', 'dedup_candidate_indices',
    'is_self_occluded', '_labels_match',
    'STICKY_ASSOCIATED', 'STICKY_AMBIGUOUS', 'STICKY_OUT_OF_GATE',
    'STICKY_NO_CANDIDATES', 'STICKY_OCCLUDED', 'STICKY_HELD',
]


STATUS_OK = 'ok'
STATUS_MOVED = 'moved'
STATUS_MISSING = 'missing'
STATUS_SELF = 'self'          # change attributed to the robot itself
STATUS_UNBOUND = 'unbound'    # target label never existed in the binding


# Position-sticky instance association

STICKY_ASSOCIATED = 'associated'
STICKY_AMBIGUOUS = 'ambiguous'
STICKY_OUT_OF_GATE = 'out_of_gate'
STICKY_NO_CANDIDATES = 'no_candidates'
STICKY_OCCLUDED = 'occluded'
# Label is currently held in the gripper: camera evidence cannot relocate
# it (the mask is the object-in-hand or a phantom), so association is
# suppressed entirely for the duration of the hold.
STICKY_HELD = 'held'


def is_self_occluded(prev_pos, ee_pos, *,
                       xy_radius_m: float = 0.08,
                       z_below_m: float = 0.12) -> bool:
    """
    Is a bound object plausibly hidden UNDER the robot's own arm?

    At the pre-close moment the gripper hovers directly over the target
    and blocks the agentview; the target's detection then either
    vanishes or re-latches a lookalike far away.  When the EE is within
    ``xy_radius_m`` of the
    object's last bound position and at or above its height, an
    out-of-gate re-association is self-occlusion, not a scene change -
    keep the binding and let the grasp telemetry (which sees through
    occlusion by construction) render the verdict milliseconds later.
    """
    if prev_pos is None or ee_pos is None:
        return False
    p = np.asarray(prev_pos, dtype=float).reshape(-1)[:3]
    e = np.asarray(ee_pos, dtype=float).reshape(-1)[:3]
    if float(np.linalg.norm(e[:2] - p[:2])) > xy_radius_m:
        return False
    # The XY hover is the load-bearing condition.  The Z guard only
    # excludes an EE far BELOW the object (camera not blocked): at grasp
    # depth the EE sits several cm below the DETECTED position (which is
    # the top surface seen by the camera) while the arm above it still
    # blocks the view, so the margin must cover rim-pinch/descend depths.
    return bool(e[2] >= p[2] - z_below_m)


@dataclass
class StickyResult:
    """
    Outcome of associating a bound label to one of several fresh
    instance detections by position (never by confidence rank).
    """
    status: str
    chosen_index: Optional[int] = None
    chosen_dist_m: Optional[float] = None


def sticky_associate(last_pos, candidates, *,
                       gate_m: float = 0.12,
                       ambiguity_sep_m: float = 0.03) -> StickyResult:
    """
    Associate a bound label to the fresh instance NEAREST its last bound
    position, instead of SAM3's top-1 confidence.

    Rationale: with two visually identical instances (e.g. the spatial
    suite's twin black bowls) SAM3's per-frame confidence order flips,
    which reads as a ~20 cm scene change.  Physical objects, unlike
    confidence ranks, move continuously - so identity follows position.

    Rules:
    * nearest candidate within ``gate_m`` of ``last_pos`` wins
      (``associated``);
    * if the two nearest in-gate candidates are within
      ``ambiguity_sep_m`` of EACH OTHER, the association is undecidable
      - keep the previous binding (``ambiguous``, no index);
    * no candidate in gate: return the nearest overall as
      ``out_of_gate`` (the caller's scene diff then sees the genuine
      large displacement and routes to recovery);
    * empty candidate list: ``no_candidates``.
    """
    if last_pos is None:
        return StickyResult(STICKY_NO_CANDIDATES)
    pts = [None if c is None else np.asarray(c, dtype=float).reshape(-1)[:3]
           for c in candidates]
    pts = [(i, p) for i, p in enumerate(pts) if p is not None]
    if not pts:
        return StickyResult(STICKY_NO_CANDIDATES)
    lp = np.asarray(last_pos, dtype=float).reshape(-1)[:3]
    dists = sorted(((float(np.linalg.norm(p - lp)), i, p) for i, p in pts),
                   key=lambda t: t[0])
    in_gate = [t for t in dists if t[0] <= gate_m]
    if not in_gate:
        d, i, _ = dists[0]
        return StickyResult(STICKY_OUT_OF_GATE, chosen_index=i,
                              chosen_dist_m=d)
    if len(in_gate) >= 2:
        _, _, p0 = in_gate[0]
        _, _, p1 = in_gate[1]
        if float(np.linalg.norm(p0 - p1)) <= ambiguity_sep_m:
            return StickyResult(STICKY_AMBIGUOUS)
    d, i, _ = in_gate[0]
    return StickyResult(STICKY_ASSOCIATED, chosen_index=i, chosen_dist_m=d)


def dedup_candidate_indices(positions, confidences, *,
                              merge_radius_m: float = 0.02) -> list:
    """
    Collapse near-duplicate instance detections before association.

    SAM3 multi-instance mode frequently emits several slightly-offset
    masks of the SAME physical object; feeding them to
    :func:`sticky_associate` biases nearest-to-previous selection toward
    the duplicate that moved least, systematically under-measuring real
    displacement (a 2 cm scripted shift can read as <1.5 cm).  Greedy
    confidence-ordered suppression: keep the highest-confidence
    detection, drop everything within ``merge_radius_m`` of a keeper.

    Returns kept indices into ``positions`` (original order preserved).
    """
    idx = [i for i, p in enumerate(positions) if p is not None]
    if len(idx) <= 1:
        return idx
    conf = [float(confidences[i]) if confidences is not None else 0.0
            for i in idx]
    order = [i for _, i in sorted(zip(conf, idx), key=lambda t: -t[0])]
    kept: list = []
    for i in order:
        p = np.asarray(positions[i], dtype=float).reshape(-1)[:3]
        dup = False
        for j in kept:
            q = np.asarray(positions[j], dtype=float).reshape(-1)[:3]
            if float(np.linalg.norm(p - q)) <= merge_radius_m:
                dup = True
                break
        if not dup:
            kept.append(i)
    return sorted(kept)


def fuzzy_key(d: Mapping, label: str) -> Optional[str]:
    """
    Resolve ``label`` to a key of ``d`` the way ``fuzzy_get_det`` does:
    exact -> substring containment -> best word overlap.  Returns the KEY
    (not the value) or None.
    """
    if not label:
        return None
    if label in d:
        return label
    label_lower = label.lower()
    label_words = set(label_lower.split())
    best_score, best_key = 0, None
    for k in d:
        k_lower = str(k).lower()
        if label_lower in k_lower or k_lower in label_lower:
            return k
        overlap = len(label_words & set(k_lower.split()))
        if overlap > best_score:
            best_score, best_key = overlap, k
    return best_key


def _labels_match(a: Optional[str], b: Optional[str]) -> bool:
    """
    Loose label identity (held-object matching): normalized containment
    or full word overlap in either direction.
    """
    if not a or not b:
        return False
    an = a.lower().replace('_', ' ').strip()
    bn = b.lower().replace('_', ' ').strip()
    if an == bn or an in bn or bn in an:
        return True
    aw, bw = set(an.split()), set(bn.split())
    return bool(aw) and (aw <= bw or bw <= aw)


@dataclass
class LabelDiff:
    label: str
    status: str
    delta_m: Optional[float] = None
    bound_pos: Optional[tuple] = None
    current_pos: Optional[tuple] = None

    def to_meta(self) -> dict:
        out = {'status': self.status}
        if self.delta_m is not None:
            out['delta_cm'] = round(self.delta_m * 100.0, 2)
        return out


@dataclass
class SceneDiff:
    """
    One pre-primitive diff verdict.

    ``target_status`` is the load-bearing field: ``moved`` / ``missing``
    for the primitive's own target drives retarget-vs-recovery;
    everything else is diagnostic.
    """
    target_label: Optional[str]
    target_key: Optional[str]
    target_status: str
    target_delta_m: float
    labels: dict = field(default_factory=dict)   # label -> LabelDiff
    t: float = 0.0

    @property
    def ok(self) -> bool:
        return self.target_status in (STATUS_OK, STATUS_SELF, STATUS_UNBOUND)

    def to_meta(self) -> dict:
        return {
            't': self.t,
            'target_label': self.target_label,
            'target_key': self.target_key,
            'target_status': self.target_status,
            'target_delta_cm': round(self.target_delta_m * 100.0, 2),
            'labels': {k: v.to_meta() for k, v in self.labels.items()},
        }


def compute_scene_diff(bound: Mapping[str, Optional[np.ndarray]],
                         current: Mapping[str, Optional[np.ndarray]],
                         *,
                         target_label: Optional[str] = None,
                         held_label: Optional[str] = None,
                         gripper_pos: Optional[np.ndarray] = None,
                         moved_threshold_m: float = 0.015,
                         self_radius_m: float = 0.12,
                         now: Optional[float] = None) -> SceneDiff:
    """
    Diff ``current`` label positions against the plan-time ``bound`` map.

    Parameters
    ----------
    bound / current:
        label -> xyz position (array-like) or None.  ``bound`` is the
        plan-time binding; ``current`` the freshest det_map positions.
    target_label:
        The dispatching primitive's own target (fuzzy-resolved against
        ``bound``).  The target is EXEMPT from the gripper-radius
        self-caused subtraction - the arm is near it by design and its
        motion is exactly what matters - but a held target still counts
        as self-caused.
    held_label:
        Label of the currently-held object (self-caused: it moves with
        the gripper).
    gripper_pos:
        Current EE position.  Non-target labels whose bound OR current
        position lies within ``self_radius_m`` of it are attributed to
        the robot (bumped / occluded by the arm), not the scene.
    moved_threshold_m:
        Position delta above which a label counts as ``moved``.

    Returns a :class:`SceneDiff`; per-label statuses are one of
    ok / moved / missing / self.
    """
    t = time.time() if now is None else float(now)
    grip = None
    if gripper_pos is not None:
        grip = np.asarray(gripper_pos, dtype=float).reshape(-1)[:3]

    target_key = fuzzy_key(bound, target_label) if target_label else None

    def _near_gripper(pos) -> bool:
        if grip is None or pos is None:
            return False
        p = np.asarray(pos, dtype=float).reshape(-1)[:3]
        return float(np.linalg.norm(p - grip)) < self_radius_m

    labels: dict = {}
    for lbl, pos_b in bound.items():
        if pos_b is None:
            continue
        pos_b = np.asarray(pos_b, dtype=float).reshape(-1)[:3]
        is_target = (lbl == target_key)

        # Self-caused rule 1: the held object moves with the gripper.
        if _labels_match(lbl, held_label):
            labels[lbl] = LabelDiff(lbl, STATUS_SELF)
            continue

        pos_c = current.get(lbl)
        if pos_c is None:
            ck = fuzzy_key(current, lbl)
            pos_c = current.get(ck) if ck is not None else None
        if pos_c is None:
            # Presence lost.  If the label was under the arm, blame the
            # robot (occlusion), not the scene - unless it is the target.
            if not is_target and _near_gripper(pos_b):
                labels[lbl] = LabelDiff(lbl, STATUS_SELF,
                                          bound_pos=tuple(pos_b))
            else:
                labels[lbl] = LabelDiff(lbl, STATUS_MISSING,
                                          bound_pos=tuple(pos_b))
            continue

        pos_c = np.asarray(pos_c, dtype=float).reshape(-1)[:3]
        delta = float(np.linalg.norm(pos_c - pos_b))

        # Self-caused rule 2: non-target labels near the gripper.
        if not is_target and (_near_gripper(pos_c) or _near_gripper(pos_b)):
            labels[lbl] = LabelDiff(lbl, STATUS_SELF, delta_m=delta,
                                      bound_pos=tuple(pos_b),
                                      current_pos=tuple(pos_c))
            continue

        status = STATUS_MOVED if delta > moved_threshold_m else STATUS_OK
        labels[lbl] = LabelDiff(lbl, status, delta_m=delta,
                                  bound_pos=tuple(pos_b),
                                  current_pos=tuple(pos_c))

    if target_label and target_key is None:
        t_status, t_delta = STATUS_UNBOUND, 0.0
    elif target_key is not None and target_key in labels:
        ld = labels[target_key]
        t_status = ld.status
        t_delta = float(ld.delta_m or 0.0)
    else:
        t_status, t_delta = STATUS_OK, 0.0

    return SceneDiff(target_label=target_label, target_key=target_key,
                       target_status=t_status, target_delta_m=t_delta,
                       labels=labels, t=t)

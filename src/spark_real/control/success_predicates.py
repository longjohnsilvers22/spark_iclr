"""Predicate vocabulary for task-success verification.

Leaf module: numpy + stdlib only, imports nothing from ``spark_real`` so it can
be used from anywhere (and unit-tested with no perception stack).

Evaluation is per-camera and returns a three-valued vote (pass/fail/abstain).
Fusion across cameras lives in ``success_verifier``.

Mode semantics on a vote:
  ``3d``   world-frame geometric test (the trustworthy one)
  ``2d``   image-space containment fallback, corroborating only
  ``none`` non-visual (proprioception / release witness) or no evidence

``absent`` can FAIL but never PASS: a non-detection is absence of evidence, not
evidence of absence. See ``_eval_absent``.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Any, Callable, Dict, List, Mapping, Optional, Sequence, Tuple

import numpy as np
from spark_real.utils.env_flags import as_bool
from spark_real.utils.det_fields import det_field

Vote = str  # "pass" | "fail" | "abstain"
Status = str  # "pass" | "fail" | "unverified"
Mode = str  # "3d" | "2d" | "none"

PASS = "pass"
FAIL = "fail"
ABSTAIN = "abstain"
UNVERIFIED = "unverified"

# Slot bounds are padded by this much before they define the container extent.
SLOT_PAD_M = 0.02
# An object may sit this far below the container's own centroid z and still
# count as inside (rim-referenced centroids read high on deep bins).
INSIDE_Z_BELOW_M = 0.02
# Smallest margin-adjusted half-extent that still carries information. A
# re-detected centroid moves ~1-2cm between passes (the binder's "stationary"
# tolerance is 6cm), so a pass region narrower than 1cm is finer than the
# sensor and the test must abstain, not fail. Example: an occluded bowl mask
# with half_b=1.9cm under inside's default -2cm margin collapses to 0.0.
DEGENERATE_EXTENT_M = 0.01


class VerifyBlockError(ValueError):
    """A ``verify:`` block is malformed; the whole block must be rejected."""


@dataclass
class CameraVote:
    camera: str
    predicate: str
    vote: Vote
    mode: Mode
    confidence: float
    detail: str


@dataclass
class VerifyOutcome:
    status: Status
    predicates: List[str] = field(default_factory=list)
    votes: List[CameraVote] = field(default_factory=list)
    gates: Dict[str, bool] = field(default_factory=dict)
    reason: str = ""
    depth_source: str = "none"

    def to_dict(self) -> dict:
        return {
            "status": self.status,
            "predicates": list(self.predicates),
            "votes": [
                {
                    "camera": v.camera,
                    "predicate": v.predicate,
                    "vote": v.vote,
                    "mode": v.mode,
                    "confidence": round(float(v.confidence), 3),
                    "detail": v.detail,
                }
                for v in self.votes
            ],
            "gates": dict(self.gates),
            "reason": self.reason,
            "depth_source": self.depth_source,
        }


@dataclass
class EvalConfig:
    """Thresholds for a single camera's evaluation."""

    min_conf: float = 0.35
    fallback_2d_containment: float = 0.90
    inside_xy_margin_m: float = -0.02
    inside_z_tol_m: float = 0.06
    z_tol_scale: float = 1.0  # 2.0 on monocular (DA3) depth


@dataclass
class SceneView:
    """One camera's view of the scene, plus non-visual state.

    ``lookup`` maps a predicate label to the ONE physical detection that label
    was written about, or None. ``explain`` (optional) says why a lookup came
    back empty; "not detected" and "three candidates, identity ambiguous" are
    both abstentions but very different diagnoses.
    """

    camera: str
    lookup: Callable[[str], Optional[Any]]
    held: Optional[bool] = None
    depth_source: str = "hardware"
    explain: Optional[Callable[[str], str]] = None


# predicate schema: required params -> optional params with defaults

_SCHEMA: Dict[str, Tuple[Tuple[str, ...], Dict[str, Any]]] = {
    "inside": (
        ("obj", "container"),
        {"xy_margin_m": -0.02, "z_tol_m": 0.06, "occlusion_ok": False},
    ),
    "on": (("obj", "surface"), {"xy_margin_m": 0.0, "z_max_m": 0.10}),
    "stacked": (
        ("obj", "base"),
        {"xy_tol_m": 0.035, "dz_min_m": 0.010, "dz_max_m": 0.120},
    ),
    "near": (("obj", "target", "max_dist_m"), {}),
    "removed_from": (("obj", "container"), {}),
    "held": (("obj", "value"), {}),
    "absent": (("obj",), {}),
}

# which params name a detection label
_LABEL_KEYS = ("obj", "container", "surface", "base", "target")
# ... and which of those name the REFERENCE FRAME rather than the object under
# test. The tree moves ``obj``; the container/surface/base/target it is judged
# against is scenery the tree never touched.
_REFERENCE_KEYS = ("container", "surface", "base", "target")


@dataclass(frozen=True)
class Predicate:
    pred: str
    params: Mapping[str, Any]

    @property
    def labels(self) -> List[str]:
        """Detection labels this predicate needs (``held`` needs none)."""
        return self._labels_for(_LABEL_KEYS)

    @property
    def reference_labels(self) -> List[str]:
        """Labels naming the reference frame this predicate measures against."""
        return self._labels_for(_REFERENCE_KEYS)

    @property
    def object_labels(self) -> List[str]:
        """Labels naming the object under test."""
        return self._labels_for(("obj",))

    def _labels_for(self, keys: Sequence[str]) -> List[str]:
        if self.pred == "held":
            return []
        out = []
        for key in keys:
            val = self.params.get(key)
            if isinstance(val, str) and val:
                out.append(val)
        return out

    def describe(self) -> str:
        if self.pred == "held":
            return f"held({self.params.get('obj', '?')}, {bool(self.params.get('value'))})"
        args = [str(self.params[k]) for k in _LABEL_KEYS if k in self.params]
        if self.pred == "near":
            args.append(f"{float(self.params['max_dist_m']):.3f}m")
        return f"{self.pred}({', '.join(args)})"


@dataclass
class VerifySpec:
    predicates: List[Predicate]
    min_conf: float = 0.35
    require_two_views: bool = False
    source: str = "planner"

    @property
    def labels(self) -> List[str]:
        return self._union("labels")

    @property
    def reference_labels(self) -> List[str]:
        return self._union("reference_labels")

    @property
    def object_labels(self) -> List[str]:
        return self._union("object_labels")

    def _union(self, attr: str) -> List[str]:
        seen, out = set(), []
        for p in self.predicates:
            for lab in getattr(p, attr):
                if lab not in seen:
                    seen.add(lab)
                    out.append(lab)
        return out

    def describe(self) -> List[str]:
        return [p.describe() for p in self.predicates]


def _xyz(det: Any) -> Optional[np.ndarray]:
    pos = det_field(det, "position_3d")
    if pos is None:
        return None
    try:
        arr = np.asarray(pos, dtype=float).reshape(-1)
    except Exception:
        return None
    if arr.size < 3 or not np.all(np.isfinite(arr[:3])):
        return None
    return arr[:3]


def _conf(det: Any) -> float:
    try:
        return float(det_field(det, "confidence", 0.0) or 0.0)
    except Exception:
        return 0.0


def _mask(det: Any) -> Optional[np.ndarray]:
    m = det_field(det, "mask")
    if m is None:
        return None
    try:
        arr = np.asarray(m)
    except Exception:
        return None
    return arr if arr.ndim == 2 and arr.size else None


# container geometry


def _container_obb(det: Any) -> Tuple[Optional[Tuple], str]:
    """Return ((centre_xy, half_a, half_b, theta), source-tag).

    The tag is echoed into the vote detail; a reference frame taken from the
    plan-time detection rather than the verify re-detection is marked ``@plan``.
    """
    tag = "@plan" if det_field(det, "plan_anchored") else ""
    xyz = _xyz(det)
    if xyz is None:
        return None, "no-position" + tag
    centre = xyz[:2].astype(float)

    theta = det_field(det, "world_major_axis_rad")
    theta = float(theta) if theta is not None else 0.0
    ar = float(det_field(det, "aspect_ratio", 1.0) or 1.0)
    # PCA inversion: a sub-1 aspect ratio means the stored "major" axis is
    # actually the short one (same guard as mask_geometry.resolve_slot_direction).
    if 0.0 < ar < 1.0:
        theta += math.pi / 2.0
        ar = 1.0 / ar

    slots = det_field(det, "slots") or []
    a_max = b_max = 0.0
    n_slots = 0
    ca, sa = math.cos(theta), math.sin(theta)
    for slot in slots:
        wxyz = det_field(slot, "world_xyz")
        if wxyz is None:
            continue
        try:
            p = np.asarray(wxyz, dtype=float).reshape(-1)[:2] - centre
        except Exception:
            continue
        a_max = max(a_max, abs(p[0] * ca + p[1] * sa))
        b_max = max(b_max, abs(-p[0] * sa + p[1] * ca))
        n_slots += 1
    if n_slots:
        return (centre, a_max + SLOT_PAD_M, b_max + SLOT_PAD_M, theta), "slots" + tag

    minor = float(det_field(det, "obb_minor_m", 0.0) or 0.0)
    if minor > 0.0:
        # obb_minor_m is the FULL short-axis length, so halve it.
        half_b = minor / 2.0
        half_a = minor * max(ar, 1.0) / 2.0
        return (centre, half_a, half_b, theta), "obb" + tag

    # No slots and no measured width: the container's extent is UNKNOWN.
    # Abstain rather than invent a radius; a camera that DID measure the
    # container still decides.
    return None, "no-extent" + tag


def container_obb_xy(det: Any) -> Optional[Tuple]:
    """``(centre_xy, half_a, half_b, theta)`` for a container detection."""
    obb, _ = _container_obb(det)
    return obb


def _obb_contains(point_xy: np.ndarray, obb: Tuple, margin: float) -> bool:
    centre, half_a, half_b, theta = obb
    d = np.asarray(point_xy, dtype=float)[:2] - np.asarray(centre, dtype=float)
    ca, sa = math.cos(theta), math.sin(theta)
    a = abs(d[0] * ca + d[1] * sa)
    b = abs(-d[0] * sa + d[1] * ca)
    # A margin may shrink the box but never invert it.
    return a <= max(half_a + margin, 0.0) and b <= max(half_b + margin, 0.0)


# public aliases (used by success_verifier's release witness and its
# plan-anchored reference builder)
obb_contains = _obb_contains
detection_xyz = _xyz
detection_field = det_field


def mask_containment(obj_mask: Any, container_mask: Any) -> Optional[float]:
    """Fraction of the object mask inside the container mask, same camera."""
    a, b = _mask(obj_mask), _mask(container_mask)
    if a is None or b is None or a.shape != b.shape:
        return None
    am = a.astype(bool)
    area = int(am.sum())
    if area <= 0:
        return None
    return float(np.logical_and(am, b.astype(bool)).sum()) / float(area)


# evaluation


def _vote(view, pred, vote, mode, conf, detail) -> CameraVote:
    return CameraVote(
        camera=view.camera,
        predicate=pred.describe(),
        vote=vote,
        mode=mode,
        confidence=float(conf),
        detail=detail,
    )


def _binding_note(view: SceneView, label: Any) -> str:
    """Why ``view.lookup`` came back empty, if the view can say."""
    if view.explain is None or not label:
        return ""
    try:
        note = view.explain(label)
    except Exception:
        return ""
    return f": {note}" if note else ""


def _resolve_pair(view: SceneView, pred: Predicate, cfg: EvalConfig, keys):
    """Bind labels -> detections; returns (dets, abstain_vote_or_None)."""
    dets = []
    for key in keys:
        label = pred.params.get(key)
        det = view.lookup(label) if label else None
        if det is None:
            return None, _vote(
                view, pred, ABSTAIN, "none", 0.0, f"'{label}' unbound{_binding_note(view, label)}"
            )
        dets.append(det)
    conf = min(_conf(d) for d in dets)
    if conf < cfg.min_conf:
        return None, _vote(
            view,
            pred,
            ABSTAIN,
            "none",
            0.0,
            f"confidence {conf:.2f} < min_conf {cfg.min_conf:.2f}",
        )
    return dets, None


def _eval_containment(
    view: SceneView, pred: Predicate, cfg: EvalConfig, obj_key: str, cont_key: str
) -> CameraVote:
    """Shared body of ``inside`` / ``on`` / ``removed_from``."""
    dets, abstain = _resolve_pair(view, pred, cfg, (obj_key, cont_key))
    if abstain is not None:
        return abstain
    obj, cont = dets
    conf = min(_conf(obj), _conf(cont))
    o_xyz, c_xyz = _xyz(obj), _xyz(cont)

    if o_xyz is not None and c_xyz is not None:
        obb, src = _container_obb(cont)
        if obb is None:
            return _vote(
                view, pred, ABSTAIN, "none", 0.0, f"container extent unknown ({src})"
            )
        if pred.pred == "on":
            margin = float(pred.params.get("xy_margin_m", 0.0))
            z_lo, z_hi = 0.0, float(pred.params.get("z_max_m", 0.10))
            dz = float(o_xyz[2] - c_xyz[2])
            z_ok = z_lo <= dz <= z_hi * cfg.z_tol_scale
        else:
            margin = float(pred.params.get("xy_margin_m", cfg.inside_xy_margin_m))
            z_tol = float(pred.params.get("z_tol_m", cfg.inside_z_tol_m))
            z_tol *= cfg.z_tol_scale
            dz = float(o_xyz[2] - c_xyz[2])
            z_ok = -INSIDE_Z_BELOW_M * cfg.z_tol_scale <= dz <= z_tol
        # A collapsed margin-adjusted extent carries NO information: every
        # object position fails the xy test, and for removed_from (which
        # inverts the test) it would be a free PASS. Abstain.
        eff_a = max(float(obb[1]) + margin, 0.0)
        eff_b = max(float(obb[2]) + margin, 0.0)
        if min(eff_a, eff_b) < DEGENERATE_EXTENT_M:
            return _vote(
                view,
                pred,
                ABSTAIN,
                "none",
                0.0,
                f"container extent collapses under margin {margin * 100:+.1f}cm: "
                f"half=({obb[1] * 100:.1f},{obb[2] * 100:.1f})cm -> "
                f"({eff_a * 100:.1f},{eff_b * 100:.1f})cm, "
                f"below the {DEGENERATE_EXTENT_M * 100:.0f}cm information floor "
                f"(obb={src})",
            )
        xy_ok = _obb_contains(o_xyz[:2], obb, margin)
        ok = bool(xy_ok and z_ok)
        detail = (
            f"3d xy_ok={xy_ok} dz={dz*100:+.1f}cm obb={src} "
            f"half=({obb[1]*100:.1f},{obb[2]*100:.1f})cm"
        )
        if pred.pred == "removed_from":
            return _vote(view, pred, FAIL if ok else PASS, "3d", conf, "removed_from: " + detail)
        return _vote(view, pred, PASS if ok else FAIL, "3d", conf, detail)

    # 2D fallback: only when this camera lacks 3D for object or container.
    # A threshold above 1.0 is the kill switch: abstain rather than fail
    # everything (a fail would veto every other camera).
    if cfg.fallback_2d_containment > 1.0:
        return _vote(view, pred, ABSTAIN, "none", 0.0, "no 3d; 2d fallback disabled")
    frac = mask_containment(obj, cont)
    if frac is None:
        return _vote(view, pred, ABSTAIN, "none", 0.0, "no 3d and no comparable masks")
    ok = frac >= cfg.fallback_2d_containment
    detail = f"2d containment={frac:.2f} (>= {cfg.fallback_2d_containment:.2f})"
    if pred.pred == "removed_from":
        return _vote(view, pred, FAIL if ok else PASS, "2d", conf, detail)
    return _vote(view, pred, PASS if ok else FAIL, "2d", conf, detail)


def _eval_stacked(view: SceneView, pred: Predicate, cfg: EvalConfig) -> CameraVote:
    dets, abstain = _resolve_pair(view, pred, cfg, ("obj", "base"))
    if abstain is not None:
        return abstain
    obj, base = dets
    conf = min(_conf(obj), _conf(base))
    o, b = _xyz(obj), _xyz(base)
    if o is None or b is None:
        return _vote(view, pred, ABSTAIN, "none", 0.0, "stacked needs 3d, none here")
    d_xy = float(np.linalg.norm(o[:2] - b[:2]))
    dz = float(o[2] - b[2])
    xy_tol = float(pred.params.get("xy_tol_m", 0.035))
    dz_min = float(pred.params.get("dz_min_m", 0.010))
    dz_max = float(pred.params.get("dz_max_m", 0.120)) * cfg.z_tol_scale
    ok = d_xy <= xy_tol and dz_min <= dz <= dz_max
    return _vote(
        view,
        pred,
        PASS if ok else FAIL,
        "3d",
        conf,
        f"3d d_xy={d_xy*100:.1f}cm dz={dz*100:+.1f}cm",
    )


def _eval_near(view: SceneView, pred: Predicate, cfg: EvalConfig) -> CameraVote:
    dets, abstain = _resolve_pair(view, pred, cfg, ("obj", "target"))
    if abstain is not None:
        return abstain
    obj, tgt = dets
    conf = min(_conf(obj), _conf(tgt))
    o, t = _xyz(obj), _xyz(tgt)
    if o is None or t is None:
        return _vote(view, pred, ABSTAIN, "none", 0.0, "near needs 3d, none here")
    dist = float(np.linalg.norm(o - t))
    ok = dist <= float(pred.params["max_dist_m"])
    return _vote(view, pred, PASS if ok else FAIL, "3d", conf, f"3d dist={dist*100:.1f}cm")


def _eval_held(view: SceneView, pred: Predicate, cfg: EvalConfig) -> CameraVote:
    want = bool(pred.params.get("value"))
    have = view.held
    if have is None:
        # Never abstains: no witness at all reads as "not holding" only when
        # the caller supplied nothing; treat as a failure to prove release.
        return _vote(view, pred, FAIL, "none", 1.0, "no grip state available")
    return _vote(
        view,
        pred,
        PASS if bool(have) == want else FAIL,
        "none",
        1.0,
        f"held={bool(have)} want={want}",
    )


def _eval_absent(view: SceneView, pred: Predicate, cfg: EvalConfig) -> CameraVote:
    """Asymmetric on purpose: ``absent`` can FAIL, and can never PASS.

    A SAM3 miss, an occlusion, and an object genuinely removed all produce the
    same non-detection, so a non-detection abstains (mode ``none``); only a
    positive re-detection is decisive, and it decides FAIL.
    """
    label = pred.params.get("obj")
    det = view.lookup(label)
    if det is None:
        return _vote(
            view,
            pred,
            ABSTAIN,
            "none",
            0.0,
            f"'{label}' not detected -- absence of evidence{_binding_note(view, label)}",
        )
    conf = _conf(det)
    if conf < cfg.min_conf:
        return _vote(
            view, pred, ABSTAIN, "none", 0.0, f"detected at {conf:.2f} < min_conf {cfg.min_conf:.2f}"
        )
    return _vote(view, pred, FAIL, "3d", conf, f"still detected at {conf:.2f}")


def evaluate_predicate(pred: Predicate, view: SceneView, cfg: EvalConfig) -> CameraVote:
    """Evaluate one predicate in one camera. Never raises for scene reasons."""
    if pred.pred in ("inside", "removed_from"):
        return _eval_containment(view, pred, cfg, "obj", "container")
    if pred.pred == "on":
        return _eval_containment(view, pred, cfg, "obj", "surface")
    if pred.pred == "stacked":
        return _eval_stacked(view, pred, cfg)
    if pred.pred == "near":
        return _eval_near(view, pred, cfg)
    if pred.pred == "held":
        return _eval_held(view, pred, cfg)
    if pred.pred == "absent":
        return _eval_absent(view, pred, cfg)
    return _vote(view, pred, ABSTAIN, "none", 0.0, f"unknown predicate {pred.pred}")


def camera_verdict(votes: Sequence[CameraVote]) -> Vote:
    """AND over predicates, abstain-absorbing."""
    if not votes:
        return ABSTAIN
    if any(v.vote == FAIL for v in votes):
        return FAIL
    if any(v.vote == ABSTAIN for v in votes):
        return ABSTAIN
    return PASS


# parsing


def _norm_bool(val: Any) -> bool:
    if isinstance(val, str):
        return as_bool(val, False)
    return bool(val)


def parse_verify_block(
    block: Any,
    *,
    label_resolver: Optional[Callable[[str], Optional[str]]] = None,
    default_min_conf: float = 0.35,
    source: str = "planner",
) -> VerifySpec:
    """Parse a ``verify:`` block. Raises :class:`VerifyBlockError` on anything
    malformed; the whole block is rejected, never partially honoured."""
    if not isinstance(block, Mapping):
        raise VerifyBlockError(f"verify block must be a mapping, got {type(block)}")
    items = block.get("all")
    if not isinstance(items, (list, tuple)) or not items:
        raise VerifyBlockError("verify block needs a non-empty 'all' list")

    preds: List[Predicate] = []
    for raw in items:
        if not isinstance(raw, Mapping):
            raise VerifyBlockError(f"predicate entry must be a mapping: {raw!r}")
        name = str(raw.get("pred", "")).strip()
        if name not in _SCHEMA:
            raise VerifyBlockError(f"unknown predicate '{name}'")
        required, optional = _SCHEMA[name]
        params: Dict[str, Any] = dict(optional)
        for key in required:
            if key not in raw or raw[key] is None:
                raise VerifyBlockError(f"'{name}' missing required param '{key}'")
            params[key] = raw[key]
        for key in optional:
            if key in raw and raw[key] is not None:
                params[key] = raw[key]
        # types
        if name == "held":
            params["value"] = _norm_bool(params["value"])
        if name == "inside":
            params["occlusion_ok"] = _norm_bool(params["occlusion_ok"])
        for key in (
            "xy_margin_m",
            "z_tol_m",
            "z_max_m",
            "xy_tol_m",
            "dz_min_m",
            "dz_max_m",
            "max_dist_m",
        ):
            if key in params:
                try:
                    params[key] = float(params[key])
                except (TypeError, ValueError):
                    raise VerifyBlockError(f"'{name}.{key}' is not a number")
        pred = Predicate(name, params)
        if label_resolver is not None:
            for lab in pred.labels:
                if not label_resolver(lab):
                    raise VerifyBlockError(f"'{name}' references unknown label '{lab}'")
        preds.append(pred)

    min_conf = block.get("min_conf", default_min_conf)
    try:
        min_conf = float(min_conf)
    except (TypeError, ValueError):
        raise VerifyBlockError("min_conf is not a number")
    return VerifySpec(
        predicates=preds,
        min_conf=min_conf,
        require_two_views=_norm_bool(block.get("require_two_views", False)),
        source=source,
    )


# derivation from tree structure (so the cached BT library verifies)

_GRASP_TYPES = frozenset(
    ("grasp", "grasp_se3", "grasp_top_down", "grasp_cgn", "grasp_se3_flow")
)
# Primitives that push, pull or drag the object they NAME (as opposed to the
# object already in the jaws). A label one of these touched has a stale
# plan-time pose, so it can never serve as a plan-time reference frame.
_DISPLACING_TYPES = frozenset(
    (
        "open_drawer",
        "close_drawer",
        "pull",
        "push",
        "push_object",
        "drag",
        "wiggle",
        "screw",
        "compliant_push",
        "compliant_insert",
        "throw_to",
        "grasp_perturb",
        "hold_in_place",
        "tilt_wrist",
    )
)
_CONTAINER_MIN_MINOR_M = 0.08


def _labels_touched_by(actions: Sequence[Mapping], types_) -> List[str]:
    """Labels the given action types name, in order, deduplicated.

    A primitive that omits its own label inherits the one from the preceding
    ``move_to_keypoint``.
    """
    out: List[str] = []
    pending: Optional[str] = None
    for act in actions or []:
        atype = str(act.get("type", act.get("name", "")))
        params = act.get("params", {}) or {}
        label = params.get("keypoint_label") or params.get("label")
        if atype == "move_to_keypoint":
            pending = label
            continue
        if atype in types_:
            lab = label or params.get("joint_name") or pending
            if lab and lab not in out:
                out.append(str(lab))
    return out


def _is_container(det: Any) -> bool:
    if det is None:
        return False
    if det_field(det, "slots"):
        return True
    return float(det_field(det, "obb_minor_m", 0.0) or 0.0) >= _CONTAINER_MIN_MINOR_M


def grasped_labels(actions: Sequence[Mapping]) -> List[str]:
    """Labels the tree actually closed the jaws on, in order.

    These are the only objects allowed to have moved during the run, which is
    what lets the verifier re-identify them geometrically after a re-detection
    (see ``success_verifier.IdentityBinder``).
    """
    return _labels_touched_by(actions, _GRASP_TYPES)


def displaced_labels(actions: Sequence[Mapping]) -> List[str]:
    """Every label the tree physically moved, carried OR shoved.

    Superset of :func:`grasped_labels`. Decides whether a label's plan-time
    detection may be reused as a reference frame: everything NOT in this set is
    scenery whose plan-time pose is still valid at verify time.
    """
    out = _labels_touched_by(actions, _GRASP_TYPES | _DISPLACING_TYPES)
    for act in actions or []:
        if str(act.get("type", act.get("name", ""))) != "sweep":
            continue
        # sweep's target_label is the container swept INTO (scenery). Only the
        # object_labels move.
        raw = (act.get("params", {}) or {}).get("object_labels") or ""
        objs = (
            list(raw)
            if isinstance(raw, (list, tuple))
            else [o.strip() for o in str(raw).split(",") if o.strip()]
        )
        for obj in objs:
            if obj and obj not in out:
                out.append(str(obj))
    return out


def derive_default_verify(
    actions: Sequence[Mapping],
    detections: Optional[Mapping[str, Any]] = None,
    *,
    min_conf: float = 0.35,
) -> Optional[VerifySpec]:
    """Derive a predicate block from a flattened action list.

    Returns None when nothing is derivable (pour/scrub/cloth/bimanual); the
    caller must then report ``unverified``, never ``pass``.
    """
    detections = detections or {}
    preds: List[Predicate] = []
    seen = set()

    def add(pred: Predicate):
        key = pred.describe()
        if key not in seen:
            seen.add(key)
            preds.append(pred)

    last_grasped: Optional[str] = None
    pending_keypoint: Optional[str] = None
    place_label: Optional[str] = None
    holding = False

    for act in actions:
        atype = str(act.get("type", act.get("name", "")))
        params = act.get("params", {}) or {}
        label = params.get("keypoint_label") or params.get("label")

        if atype == "move_to_keypoint":
            pending_keypoint = label
            if holding and label:
                place_label = label
            continue

        if atype in _GRASP_TYPES:
            last_grasped = label or pending_keypoint or last_grasped
            holding = True
            place_label = None
            continue

        if atype == "place_in_slot":
            cont = params.get("container_label")
            if last_grasped and cont:
                add(
                    Predicate(
                        "inside",
                        {
                            "obj": last_grasped,
                            "container": cont,
                            "xy_margin_m": -0.02,
                            "z_tol_m": 0.06,
                            "occlusion_ok": False,
                        },
                    )
                )
            holding = False
            place_label = None
            continue

        if atype == "stack":
            base = params.get("target_label")
            if last_grasped and base:
                add(
                    Predicate(
                        "stacked",
                        {
                            "obj": last_grasped,
                            "base": base,
                            "xy_tol_m": 0.035,
                            "dz_min_m": 0.010,
                            "dz_max_m": 0.120,
                        },
                    )
                )
            holding = False
            continue

        if atype == "sweep":
            target = params.get("target_label") or params.get("area_label")
            raw_objs = params.get("object_labels") or ""
            objs = (
                list(raw_objs)
                if isinstance(raw_objs, (list, tuple))
                else [o.strip() for o in str(raw_objs).split(",") if o.strip()]
            )
            for obj in objs:
                if target:
                    add(
                        Predicate(
                            "inside",
                            {
                                "obj": obj,
                                "container": target,
                                "xy_margin_m": -0.02,
                                "z_tol_m": 0.06,
                                "occlusion_ok": False,
                            },
                        )
                    )
            continue

        if atype == "release":
            if last_grasped and place_label:
                cont_det = detections.get(place_label)
                if _is_container(cont_det):
                    add(
                        Predicate(
                            "inside",
                            {
                                "obj": last_grasped,
                                "container": place_label,
                                "xy_margin_m": -0.02,
                                "z_tol_m": 0.06,
                                "occlusion_ok": False,
                            },
                        )
                    )
                else:
                    add(
                        Predicate(
                            "on",
                            {
                                "obj": last_grasped,
                                "surface": place_label,
                                "xy_margin_m": 0.0,
                                "z_max_m": 0.10,
                            },
                        )
                    )
            holding = False
            place_label = None
            continue

    if not preds:
        return None
    if last_grasped:
        add(Predicate("held", {"obj": last_grasped, "value": False}))
    return VerifySpec(predicates=preds, min_conf=min_conf, source="derived")


__all__ = [
    "ABSTAIN",
    "CameraVote",
    "DEGENERATE_EXTENT_M",
    "EvalConfig",
    "FAIL",
    "PASS",
    "Predicate",
    "SceneView",
    "UNVERIFIED",
    "VerifyBlockError",
    "VerifyOutcome",
    "VerifySpec",
    "camera_verdict",
    "container_obb_xy",
    "derive_default_verify",
    "detection_field",
    "detection_xyz",
    "displaced_labels",
    "grasped_labels",
    "obb_contains",
    "evaluate_predicate",
    "mask_containment",
    "parse_verify_block",
]

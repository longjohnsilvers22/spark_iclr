"""
RoboInter intermediate representations as an OPTIONAL plan extension.

RoboInter (arXiv 2602.09973, InternRobotics) defines a per-frame vocabulary
that bridges a high-level plan and low-level actions -- object boxes, contact
points, placement proposals, 2D traces, affordance boxes, 6D state
affordances, primitive skills and subtasks -- and bridges them with F-CoT, a
chain-of-thought that mixes text and geometry inside one plan-then-execute
loop.

This module lets the planner say *where*, in the frame it was actually
shown, instead of only *what*.

Where it lives in a score
-------------------------
Per node, under the single key ``__robointer``; per score, ``__fcot``::

    - type: move_to_keypoint
      params: {keypoint_label: "knife handle 1"}
      __robointer:
        subtask: "grasp the knife by the handle"
        primitive_skill: pick
        object_box: [[412, 300], [655, 372]]
        contact_point: [455, 337]
        trace: [[455, 337], [500, 300], [610, 250]]

Both keys start with ``__``, which is load-bearing:
``bt_library._strip_nonsemantic`` and ``spark_planner.normalize_plan_for_hash``
drop every ``__``-prefixed key at EVERY level of the score before hashing, so
an annotated plan hashes to the same BT cache key as the un-annotated tree it
came from and all 43 cached trees keep matching (see
``tests/test_robointer_compat.py``).

**Anything under ``__robointer`` is advisory.** Two plans that differ only in
their annotations are the same tree to the cache, so an annotation must never
be the sole determinant of a motion. A representation that graduates to
steering the arm has to move out of the ``__`` namespace and become a hashed
param.

Coordinate convention
---------------------
2D coordinates are INTEGERS IN [0, 1000], both axes, x then y, relative to the
image the planner was shown. That is Gemini's native normalized-box space, so
it is what the model is best at producing. They are stored internally as
floats in [0, 1] and only turned into pixels against an explicit image size.
Values already in [0, 1] are accepted as normalized. See ``_coord_scale``.

``state_affordance`` is the exception: it is a 6D pose in the ROBOT BASE FRAME,
metres and rotation-vector radians, never scaled.

Nothing here calls a network, a camera or a robot.
"""

from __future__ import annotations

import copy
import logging
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np

from spark_real.planning.robointer_geometry import CameraModel, sample_depth
from spark_real.utils.score_tree import is_num, label_known, walk_nodes

logger = logging.getLogger(__name__)


# Per-node annotation key, and the score-level F-CoT key. The `__` prefix is
# what makes both hash-neutral -- see the module docstring.
ANNOTATION_KEY = "__robointer"
FCOT_KEY = "__fcot"

# LLM-facing coordinate range. 1000 == the full width (or height) of the image
# the planner was shown.
COORD_SCALE = 1000.0

# RoboInter's primitive-skill vocabulary, restricted to what this rig can
# express. Unknown values are dropped rather than passed through: a skill name
# nobody implements reads as a promise the plan does not keep.
PRIMITIVE_SKILLS = (
    "pick",
    "place",
    "push",
    "pull",
    "lift",
    "twist",
    "open",
    "close",
    "pour",
    "wipe",
    "insert",
    "press",
    "reach",
    "retract",
)

# A trace is a coarse path hint, not a trajectory. Longer than this and the
# model is inventing precision it does not have.
MAX_TRACE_POINTS = 24
MIN_TRACE_POINTS = 2

# Slack allowed outside [0, 1] before a coordinate is called an error rather
# than a rounding artefact. ~1% of the frame.
_COORD_SLACK = 0.01

# Text fields are truncated, not rejected -- a long subtask string is sloppy,
# not dangerous.
_MAX_TEXT = 240


class RoboInterSchemaError(ValueError):
    """Raised only when a caller asks for strict validation."""


# --------------------------------------------------------------------------
# Coordinate handling
# --------------------------------------------------------------------------


def _coord_scale(values: Sequence[float]) -> float:
    """Divisor that maps a block's raw coordinates into [0, 1].

    Decided ONCE per annotation block, from the largest value in it, so a
    block never ends up with half its numbers in one space and half in the
    other. Anything above 1.0 is read as the 0..1000 space the prompt asks
    for; a block whose values are all <= 1.0 is already normalized.
    """
    finite = [abs(float(v)) for v in values if is_num(v) and np.isfinite(v)]
    if not finite:
        return 1.0
    return COORD_SCALE if max(finite) > 1.0 + _COORD_SLACK else 1.0


def _norm_coord(value, scale: float) -> Optional[float]:
    if not is_num(value) or not np.isfinite(float(value)):
        return None
    return float(value) / scale


@dataclass(frozen=True)
class Point2D:
    """A normalized image-space point."""

    x: float
    y: float

    def to_list(self) -> List[float]:
        return [round(self.x, 4), round(self.y, 4)]

    def to_permille(self) -> List[int]:
        return [int(round(self.x * COORD_SCALE)), int(round(self.y * COORD_SCALE))]

    def to_pixels(self, image_size: Tuple[int, int]) -> Tuple[float, float]:
        w, h = image_size
        return (self.x * float(w), self.y * float(h))

    @classmethod
    def from_pixels(cls, u: float, v: float, image_size: Tuple[int, int]) -> "Point2D":
        w, h = image_size
        return cls(float(u) / float(w), float(v) / float(h))

    def issues(self, path: str) -> List[str]:
        out = []
        for name, val in (("x", self.x), ("y", self.y)):
            if not np.isfinite(val) or not (-_COORD_SLACK <= val <= 1.0 + _COORD_SLACK):
                out.append(f"{path}: {name}={val:.3f} outside the image")
        return out

    def clamped(self) -> "Point2D":
        return Point2D(float(np.clip(self.x, 0.0, 1.0)), float(np.clip(self.y, 0.0, 1.0)))


@dataclass(frozen=True)
class Box2D:
    """A normalized axis-aligned image-space box, ``[[x1,y1],[x2,y2]]``."""

    x1: float
    y1: float
    x2: float
    y2: float

    @property
    def center(self) -> Point2D:
        return Point2D((self.x1 + self.x2) / 2.0, (self.y1 + self.y2) / 2.0)

    @property
    def width(self) -> float:
        return abs(self.x2 - self.x1)

    @property
    def height(self) -> float:
        return abs(self.y2 - self.y1)

    @property
    def area(self) -> float:
        return self.width * self.height

    def corners(self) -> List[Point2D]:
        return [
            Point2D(self.x1, self.y1),
            Point2D(self.x2, self.y1),
            Point2D(self.x2, self.y2),
            Point2D(self.x1, self.y2),
        ]

    def to_list(self) -> List[List[float]]:
        return [
            [round(self.x1, 4), round(self.y1, 4)],
            [round(self.x2, 4), round(self.y2, 4)],
        ]

    def to_permille_pair(self) -> List[List[int]]:
        return [
            Point2D(self.x1, self.y1).to_permille(),
            Point2D(self.x2, self.y2).to_permille(),
        ]

    def to_pixels(self, image_size: Tuple[int, int]) -> Tuple[float, float, float, float]:
        w, h = image_size
        return (self.x1 * w, self.y1 * h, self.x2 * w, self.y2 * h)

    @classmethod
    def from_pixels(cls, x1, y1, x2, y2, image_size: Tuple[int, int]) -> "Box2D":
        w, h = image_size
        return cls(float(x1) / w, float(y1) / h, float(x2) / w, float(y2) / h)

    def ordered(self) -> "Box2D":
        """Corners sorted so x1<=x2 and y1<=y2. Lossless for an AABB."""
        lo_x, hi_x = sorted((self.x1, self.x2))
        lo_y, hi_y = sorted((self.y1, self.y2))
        return Box2D(lo_x, lo_y, hi_x, hi_y)

    def iou(self, other: "Box2D") -> float:
        a, b = self.ordered(), other.ordered()
        ix = max(0.0, min(a.x2, b.x2) - max(a.x1, b.x1))
        iy = max(0.0, min(a.y2, b.y2) - max(a.y1, b.y1))
        inter = ix * iy
        union = a.area + b.area - inter
        return float(inter / union) if union > 0 else 0.0

    def issues(self, path: str) -> List[str]:
        out = []
        for pt, name in ((Point2D(self.x1, self.y1), "p1"), (Point2D(self.x2, self.y2), "p2")):
            out.extend(pt.issues(f"{path}.{name}"))
        if self.width <= 0.0 or self.height <= 0.0:
            out.append(f"{path}: degenerate box (w={self.width:.3f}, h={self.height:.3f})")
        return out


@dataclass(frozen=True)
class Pose6D:
    """A base-frame 6D pose: metres + rotation vector radians. NOT scaled."""

    x: float
    y: float
    z: float
    rx: float
    ry: float
    rz: float

    def to_list(self) -> List[float]:
        return [round(float(v), 6) for v in (self.x, self.y, self.z, self.rx, self.ry, self.rz)]

    @property
    def position(self) -> np.ndarray:
        return np.array([self.x, self.y, self.z], dtype=float)

    @property
    def rotvec(self) -> np.ndarray:
        return np.array([self.rx, self.ry, self.rz], dtype=float)

    def issues(self, path: str, workspace_radius_m: float = 2.0) -> List[str]:
        out = []
        vals = self.to_list()
        if not all(np.isfinite(vals)):
            return [f"{path}: non-finite pose {vals}"]
        if float(np.linalg.norm(self.position)) > workspace_radius_m:
            out.append(
                f"{path}: position {self.position.round(3).tolist()} is more than "
                f"{workspace_radius_m} m from the base -- probably not metres"
            )
        if float(np.linalg.norm(self.rotvec)) > np.pi + 1e-6:
            out.append(
                f"{path}: rotvec norm {np.linalg.norm(self.rotvec):.3f} > pi -- "
                "probably degrees, or not a rotation vector"
            )
        return out


# --------------------------------------------------------------------------
# The per-node annotation
# --------------------------------------------------------------------------


@dataclass
class NodeAnnotation:
    """RoboInter representations attached to one behaviour-tree node.

    Every field is optional. An empty annotation is equivalent to no
    annotation at all and is dropped on serialisation.
    """

    subtask: Optional[str] = None
    primitive_skill: Optional[str] = None
    label: Optional[str] = None  # which detection the 2D fields refer to
    camera: Optional[str] = None  # frame the 2D coords live in
    object_box: Optional[Box2D] = None
    affordance_box: Optional[Box2D] = None
    placement_proposal: Optional[Box2D] = None
    contact_point: Optional[Point2D] = None
    trace: Optional[List[Point2D]] = None
    state_affordance: Optional[Pose6D] = None

    def is_empty(self) -> bool:
        return not any(
            getattr(self, f) is not None
            for f in (
                "subtask",
                "primitive_skill",
                "label",
                "camera",
                "object_box",
                "affordance_box",
                "placement_proposal",
                "contact_point",
                "trace",
                "state_affordance",
            )
        )

    def to_dict(self) -> Dict[str, Any]:
        """Plain builtins only, so ``yaml.safe_dump`` and ``json.dumps`` work.

        ``EpisodeRecorder.end()`` writes the score to ``bt.yaml`` with
        ``yaml.safe_dump``, which is how annotations reach the episode bundle.
        """
        out: Dict[str, Any] = {}
        if self.subtask:
            out["subtask"] = self.subtask
        if self.primitive_skill:
            out["primitive_skill"] = self.primitive_skill
        if self.label:
            out["label"] = self.label
        if self.camera:
            out["camera"] = self.camera
        if self.object_box is not None:
            out["object_box"] = self.object_box.to_list()
        if self.affordance_box is not None:
            out["affordance_box"] = self.affordance_box.to_list()
        if self.placement_proposal is not None:
            out["placement_proposal"] = self.placement_proposal.to_list()
        if self.contact_point is not None:
            out["contact_point"] = self.contact_point.to_list()
        if self.trace:
            out["trace"] = [p.to_list() for p in self.trace]
        if self.state_affordance is not None:
            out["state_affordance"] = self.state_affordance.to_list()
        return out

    def issues(self, path: str = ANNOTATION_KEY, keypoint_labels=None) -> List[str]:
        out: List[str] = []
        if self.primitive_skill is not None and self.primitive_skill not in PRIMITIVE_SKILLS:
            out.append(
                f"{path}: unknown primitive_skill {self.primitive_skill!r}; "
                f"expected one of {list(PRIMITIVE_SKILLS)}"
            )
        if self.label is not None and keypoint_labels:
            if not label_known(self.label, set(keypoint_labels)):
                out.append(f"{path}: label {self.label!r} is not a detected keypoint")
        for name in ("object_box", "affordance_box", "placement_proposal"):
            box = getattr(self, name)
            if box is not None:
                out.extend(box.issues(f"{path}.{name}"))
        if self.contact_point is not None:
            out.extend(self.contact_point.issues(f"{path}.contact_point"))
        if self.trace is not None:
            n = len(self.trace)
            if n < MIN_TRACE_POINTS:
                out.append(f"{path}.trace: {n} point(s); a trace needs at least {MIN_TRACE_POINTS}")
            if n > MAX_TRACE_POINTS:
                out.append(f"{path}.trace: {n} points exceeds the {MAX_TRACE_POINTS}-point cap")
            for i, pt in enumerate(self.trace):
                out.extend(pt.issues(f"{path}.trace[{i}]"))
        if self.state_affordance is not None:
            out.extend(self.state_affordance.issues(f"{path}.state_affordance"))
        # A contact point outside the object it claims to touch is the single
        # most likely LLM mistake here, and the cheapest one to catch.
        if not self._contact_inside_box():
            out.append(
                f"{path}: contact_point {self.contact_point.to_permille()} lies "
                f"outside its own object_box {self.object_box.ordered().to_list()}"
            )
        return out

    def _contact_inside_box(self) -> bool:
        if self.contact_point is None or self.object_box is None:
            return True
        box = self.object_box.ordered()
        p = self.contact_point
        return (
            box.x1 - _COORD_SLACK <= p.x <= box.x2 + _COORD_SLACK
            and box.y1 - _COORD_SLACK <= p.y <= box.y2 + _COORD_SLACK
        )

    def pruned(self, keypoint_labels=None) -> "NodeAnnotation":
        """Copy with every field that :meth:`issues` complains about removed.

        Fields are independent representations, so a bad trace does not cost
        a good contact point.
        """
        out = NodeAnnotation(
            subtask=self.subtask,
            camera=self.camera,
            label=self.label,
            primitive_skill=self.primitive_skill,
            state_affordance=self.state_affordance,
        )
        if out.primitive_skill not in PRIMITIVE_SKILLS:
            out.primitive_skill = None
        if out.label is not None and keypoint_labels:
            if not label_known(out.label, set(keypoint_labels)):
                out.label = None
        if out.state_affordance is not None and out.state_affordance.issues("p"):
            out.state_affordance = None
        for name in ("object_box", "affordance_box", "placement_proposal"):
            box = getattr(self, name)
            if box is not None and not box.issues("p"):
                setattr(out, name, box)
        if self.contact_point is not None and not self.contact_point.issues("p"):
            out.contact_point = self.contact_point
        # Re-check containment against whichever box survived.
        if not out._contact_inside_box():
            out.contact_point = None
        if self.trace:
            pts = [p for p in self.trace if not p.issues("p")][:MAX_TRACE_POINTS]
            out.trace = pts if len(pts) >= MIN_TRACE_POINTS else None
        return out


def _parse_point(raw, scale: float) -> Optional[Point2D]:
    if not isinstance(raw, (list, tuple)) or len(raw) != 2:
        return None
    x, y = _norm_coord(raw[0], scale), _norm_coord(raw[1], scale)
    if x is None or y is None:
        return None
    return Point2D(x, y)


def _parse_box(raw, scale: float) -> Optional[Box2D]:
    """Accept ``[[x1,y1],[x2,y2]]`` (RoboInter) or a flat ``[x1,y1,x2,y2]``."""
    if isinstance(raw, (list, tuple)) and len(raw) == 2:
        p1, p2 = _parse_point(raw[0], scale), _parse_point(raw[1], scale)
        if p1 is None or p2 is None:
            return None
        return Box2D(p1.x, p1.y, p2.x, p2.y).ordered()
    if isinstance(raw, (list, tuple)) and len(raw) == 4:
        vals = [_norm_coord(v, scale) for v in raw]
        if any(v is None for v in vals):
            return None
        return Box2D(*vals).ordered()
    return None


def _collect_coords(raw: Dict[str, Any]) -> List[float]:
    """Every 2D coordinate in the block, for the one-shot scale decision."""
    out: List[float] = []

    def walk(value):
        if isinstance(value, (list, tuple)):
            for item in value:
                walk(item)
        elif is_num(value):
            out.append(float(value))

    for key in (
        "object_box",
        "affordance_box",
        "placement_proposal",
        "contact_point",
        "trace",
    ):
        walk(raw.get(key))
    return out


def _text(value) -> Optional[str]:
    if not isinstance(value, str):
        return None
    cleaned = value.strip()
    return cleaned[:_MAX_TEXT] if cleaned else None


def parse_annotation(raw, path: str = ANNOTATION_KEY) -> Tuple[Optional[NodeAnnotation], List[str]]:
    """Parse one raw ``__robointer`` mapping.

    Returns ``(annotation_or_None, issues)``. A field that cannot be parsed is
    reported and left unset -- never guessed at. Fields are independent
    representations, so one bad box does not void the rest of the block (this
    deliberately differs from ``verify:``, whose predicates AND together and
    are therefore rejected whole).
    """
    issues: List[str] = []
    if raw is None:
        return None, issues
    if not isinstance(raw, dict):
        return None, [f"{path}: must be a mapping, got {type(raw).__name__}"]

    known = {
        "subtask",
        "primitive_skill",
        "label",
        "camera",
        "object_box",
        "affordance_box",
        "placement_proposal",
        "contact_point",
        "trace",
        "state_affordance",
    }
    for key in raw:
        if key not in known:
            issues.append(f"{path}: unknown field {key!r} (dropped)")

    scale = _coord_scale(_collect_coords(raw))
    ann = NodeAnnotation(
        subtask=_text(raw.get("subtask")),
        camera=_text(raw.get("camera")),
        label=_text(raw.get("label")),
    )

    skill = _text(raw.get("primitive_skill"))
    if skill is not None:
        ann.primitive_skill = skill.lower()

    for name in ("object_box", "affordance_box", "placement_proposal"):
        if raw.get(name) is None:
            continue
        box = _parse_box(raw[name], scale)
        if box is None:
            issues.append(f"{path}.{name}: expected [[x1,y1],[x2,y2]], got {raw[name]!r}")
        else:
            setattr(ann, name, box)

    if raw.get("contact_point") is not None:
        pt = _parse_point(raw["contact_point"], scale)
        if pt is None:
            issues.append(f"{path}.contact_point: expected [x,y], got {raw['contact_point']!r}")
        else:
            ann.contact_point = pt

    if raw.get("trace") is not None:
        seq = raw["trace"]
        if not isinstance(seq, (list, tuple)):
            issues.append(f"{path}.trace: expected a list of [x,y], got {type(seq).__name__}")
        else:
            pts = []
            for i, item in enumerate(seq):
                pt = _parse_point(item, scale)
                if pt is None:
                    issues.append(f"{path}.trace[{i}]: expected [x,y], got {item!r}")
                else:
                    pts.append(pt)
            # Keep the points that parsed; a dropped waypoint coarsens the
            # hint, it does not invert it. The issues are already reported.
            if pts:
                ann.trace = pts

    if raw.get("state_affordance") is not None:
        sa = raw["state_affordance"]
        if (
            isinstance(sa, (list, tuple))
            and len(sa) == 6
            and all(is_num(v) and np.isfinite(float(v)) for v in sa)
        ):
            ann.state_affordance = Pose6D(*[float(v) for v in sa])
        else:
            issues.append(
                f"{path}.state_affordance: expected 6 numbers [x,y,z,rx,ry,rz], got {sa!r}"
            )

    issues.extend(ann.issues(path))
    if ann.is_empty():
        return None, issues
    return ann, issues


# --------------------------------------------------------------------------
# Score-level API
# --------------------------------------------------------------------------


def annotation_of(node: dict) -> Any:
    return node.get(ANNOTATION_KEY) if isinstance(node, dict) else None


def set_annotation(node: dict, ann: Optional[NodeAnnotation]) -> dict:
    """Attach (or clear) an annotation on a node, in place."""
    if ann is None or ann.is_empty():
        node.pop(ANNOTATION_KEY, None)
    else:
        node[ANNOTATION_KEY] = ann.to_dict()
    return node


def extract_annotations(score, keypoint_labels=None) -> List[Tuple[str, NodeAnnotation]]:
    """(node_path, annotation) for every annotated node, depth first.

    Malformed blocks are skipped with a WARNING, matching how the planner
    treats a bad extension field.
    """
    out: List[Tuple[str, NodeAnnotation]] = []
    if not isinstance(score, dict):
        return out
    for path, node in walk_nodes(score.get("tree")):
        raw = annotation_of(node)
        if raw is None:
            continue
        ann, issues = parse_annotation(raw, f"{path}.{ANNOTATION_KEY}")
        for issue in issues:
            logger.warning("[robointer] %s", issue)
        if ann is not None:
            if keypoint_labels and ann.label and not label_known(ann.label, set(keypoint_labels)):
                ann.label = None
            out.append((path, ann))
    return out


def validate_plan_annotations(score, keypoint_labels=None) -> List[str]:
    """Every issue in the score's RoboInter blocks. Empty for an old plan."""
    issues: List[str] = []
    if not isinstance(score, dict):
        return ["score: must be a mapping"]
    fcot = score.get(FCOT_KEY)
    if fcot is not None and not isinstance(fcot, (str, list)):
        issues.append(f"{FCOT_KEY}: must be a string or a list of strings")
    for path, node in walk_nodes(score.get("tree")):
        raw = annotation_of(node)
        if raw is None:
            continue
        ann, node_issues = parse_annotation(raw, f"{path}.{ANNOTATION_KEY}")
        issues.extend(node_issues)
        if ann is not None and keypoint_labels:
            issues.extend(
                i
                for i in ann.issues(f"{path}.{ANNOTATION_KEY}", keypoint_labels)
                if "is not a detected keypoint" in i
            )
    return issues


def sanitize_plan_annotations(score, keypoint_labels=None, strict: bool = False):
    """Return ``(clean_score, issues)`` with malformed annotations rewritten.

    Lenient (the production path): every issue is logged at WARNING and the
    offending FIELD is dropped. ``strict=True`` raises instead.

    A score with no annotations is returned unchanged and un-copied, so this
    is free for the 43 cached trees.
    """
    if not isinstance(score, dict):
        raise RoboInterSchemaError(f"score must be a mapping, got {type(score).__name__}")

    issues = validate_plan_annotations(score, keypoint_labels)
    has_any = any(annotation_of(n) is not None for _p, n in walk_nodes(score.get("tree")))
    if not issues and not has_any:
        return score, []
    if strict and issues:
        raise RoboInterSchemaError("invalid RoboInter annotations: " + "; ".join(issues))
    for issue in issues:
        logger.warning("[robointer] %s", issue)

    clean = copy.deepcopy(score)
    if clean.get(FCOT_KEY) is not None and not isinstance(clean[FCOT_KEY], (str, list)):
        clean.pop(FCOT_KEY, None)
    for path, node in walk_nodes(clean.get("tree")):
        raw = annotation_of(node)
        if raw is None:
            continue
        ann, _ = parse_annotation(raw, f"{path}.{ANNOTATION_KEY}")
        # Re-serialising from the PRUNED object is the drop: anything that
        # failed to parse, or that parsed into something the validator
        # complains about, simply is not in the round trip.
        set_annotation(node, None if ann is None else ann.pruned(keypoint_labels))
    return clean, issues


def strip_annotations(score):
    """Deep copy with every RoboInter key removed, at every level.

    For a consumer that must see exactly the pre-RoboInter score. Note that
    the BT hash does NOT need this -- ``_strip_nonsemantic`` already ignores
    ``__`` keys -- so this is for export and diffing, not for caching.
    """
    if isinstance(score, dict):
        return {
            k: strip_annotations(v) for k, v in score.items() if k not in (ANNOTATION_KEY, FCOT_KEY)
        }
    if isinstance(score, list):
        return [strip_annotations(v) for v in score]
    return score


# --------------------------------------------------------------------------
# 2D -> base frame
# --------------------------------------------------------------------------


@dataclass
class ResolvedAnnotation:
    """A NodeAnnotation lifted into the robot base frame.

    Fields are None when the lift was not possible; the reasons are in
    ``notes`` rather than silently dropped, because "the planner proposed a
    placement that could not be projected" and "the planner proposed nothing"
    must not look the same in a trace.
    """

    node_path: str = ""
    subtask: Optional[str] = None
    primitive_skill: Optional[str] = None
    label: Optional[str] = None
    camera: Optional[str] = None
    z_plane_m: Optional[float] = None
    contact_xyz: Optional[np.ndarray] = None
    placement_xyz: Optional[np.ndarray] = None
    object_footprint: Optional[np.ndarray] = None  # 4x3 base-frame corners
    trace_xyz: Optional[np.ndarray] = None  # Nx3
    state_affordance: Optional[np.ndarray] = None  # 6, base frame, as given
    notes: List[str] = field(default_factory=list)

    def to_dict(self) -> Dict[str, Any]:
        def arr(a):
            return None if a is None else np.asarray(a, dtype=float).round(4).tolist()

        return {
            "node_path": self.node_path,
            "subtask": self.subtask,
            "primitive_skill": self.primitive_skill,
            "label": self.label,
            "camera": self.camera,
            "z_plane_m": (None if self.z_plane_m is None else round(float(self.z_plane_m), 4)),
            "contact_xyz": arr(self.contact_xyz),
            "placement_xyz": arr(self.placement_xyz),
            "object_footprint": arr(self.object_footprint),
            "trace_xyz": arr(self.trace_xyz),
            "state_affordance": arr(self.state_affordance),
            "notes": list(self.notes),
        }


def resolve_annotation(
    ann: NodeAnnotation,
    camera: CameraModel,
    z_plane_m: float,
    depth: Optional[np.ndarray] = None,
    node_path: str = "",
) -> ResolvedAnnotation:
    """Lift one annotation's 2D fields into the base frame.

    ``z_plane_m`` is the horizontal plane the 2D points are assumed to lie on
    -- the table height for a placement, or the grasped object's top for a
    contact point. It is REQUIRED because a pixel is a ray: without either a
    plane or a depth value there is no 3D point, and guessing one puts the arm
    somewhere arbitrary.

    ``depth`` is optional. When a hardware depth map for the same camera is
    supplied, the contact point is backprojected with the measured depth at
    that pixel (which does not assume the object is flat) and the plane is
    used only as the fallback. Everything else stays on the plane: a placement
    proposal and a transport trace are goals, and the depth map does not know
    about them.
    """
    res = ResolvedAnnotation(
        node_path=node_path,
        subtask=ann.subtask,
        primitive_skill=ann.primitive_skill,
        label=ann.label,
        camera=ann.camera or camera.name or None,
        z_plane_m=float(z_plane_m),
    )
    size = camera.image_size

    def on_plane(pt: Point2D, what: str):
        u, v = pt.to_pixels(size)
        world = camera.pixel_to_base_on_plane(u, v, z_plane_m)
        if world is None:
            res.notes.append(f"{what}: ray does not meet the z={z_plane_m:.3f} plane")
        return world

    if ann.contact_point is not None:
        u, v = ann.contact_point.to_pixels(size)
        d = sample_depth(depth, u, v) if depth is not None else None
        if d is not None:
            res.contact_xyz = camera.pixel_to_base(u, v, d)
            res.notes.append("contact_point: from measured depth")
        else:
            res.contact_xyz = on_plane(ann.contact_point, "contact_point")
            if res.contact_xyz is not None:
                res.notes.append("contact_point: from the z-plane assumption")

    if ann.placement_proposal is not None:
        res.placement_xyz = on_plane(ann.placement_proposal.center, "placement_proposal")

    if ann.object_box is not None:
        pts = [on_plane(c, "object_box") for c in ann.object_box.ordered().corners()]
        if all(p is not None for p in pts):
            res.object_footprint = np.vstack(pts)

    if ann.trace:
        pts = [on_plane(p, "trace") for p in ann.trace]
        good = [p for p in pts if p is not None]
        if len(good) >= MIN_TRACE_POINTS:
            res.trace_xyz = np.vstack(good)
        elif pts:
            res.notes.append("trace: too few points survived projection")

    if ann.state_affordance is not None:
        res.state_affordance = np.concatenate(
            [ann.state_affordance.position, ann.state_affordance.rotvec]
        )

    return res


def resolve_plan_annotations(
    score,
    cameras: Dict[str, CameraModel],
    z_plane_m: float,
    depths: Optional[Dict[str, np.ndarray]] = None,
    default_camera: Optional[str] = None,
    keypoint_labels=None,
) -> List[ResolvedAnnotation]:
    """Resolve every annotation in a score against the named cameras.

    An annotation whose ``camera`` is unknown is skipped with a note rather
    than resolved against the wrong frame -- sideview and birdview coordinates
    are not interchangeable and mixing them would silently move the target
    across the table.
    """
    out: List[ResolvedAnnotation] = []
    depths = depths or {}
    for path, ann in extract_annotations(score, keypoint_labels):
        cam_name = ann.camera or default_camera
        cam = cameras.get(cam_name) if cam_name else None
        if cam is None:
            res = ResolvedAnnotation(node_path=path, subtask=ann.subtask, camera=cam_name)
            res.notes.append(f"no camera model for {cam_name!r}; not resolved")
            out.append(res)
            continue
        out.append(
            resolve_annotation(ann, cam, z_plane_m, depth=depths.get(cam_name), node_path=path)
        )
    return out

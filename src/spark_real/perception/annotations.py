"""
Provider-agnostic 2D annotation interface: points, boxes, drawn traces.

Three producers speak through this one type:

* the human web UI (click / drawn box, via /api/detect_click and
  /api/detect_box),
* Gemini Robotics-ER 2 (pointing + trajectory queries),
* MolmoAct / Molmo pointing (local molmoact2 conda env).

and two consumers read it:

* SAM3's click / box heads (seed a mask from a point or box), and
* the executor's waypoint path (a trace back-projected at the object's
  depth plane).

AXIS ORDER -- read this before touching a coordinate
----------------------------------------------------
Internal convention, everywhere in this module and in every
:class:`Annotation`: ``(x, y)`` normalized to ``[0, 1]``, where
``x = column / width`` and ``y = row / height``.  This matches
``robointer.Point2D``, SAM3's pixel prompts ``(u, v)``, and OpenCV.

The providers do NOT share it:

* Gemini Robotics-ER 2 answers ``[y, x]`` on a 0-1000 grid (row first!),
  and boxes as ``[ymin, xmin, ymax, xmax]``.
* Molmo answers ``(x, y)`` -- column first -- on a 0-100 grid (Molmo 1)
  or 0-1000 grid (Molmo 2 / MolmoAct2).

The converters below are the ONLY place the swap happens and they are
covered by round-trip unit tests against asymmetric frames (see
tests/test_annotations.py).  Never inline a ``[y, x]`` swap at a call site.

TWO ROUTES TO ONE CAPABILITY -- spatial corrections to plans
------------------------------------------------------------
A model supplying "grip HERE / release THERE" pixels to a plan step can
arrive by two routes, and both are expressed as :class:`Annotation`:

* **Inline (planner)**: during BT generation Gemini emits ``__robointer``
  blocks inside the planning reply itself (``planning/robointer*.py``:
  contact points, placement proposals, boxes, traces -- INTEGERS on a
  0..1000 grid, **x first**, matching this module's internal axis order).
  One LLM call; the geometry is only as fresh as the frame the plan was
  made from.

* **Out-of-band (provider)**: a separate pointing call through an
  :class:`AnnotationProvider` (ER2 / Molmo / the human UI) targeted at a
  plan step.  Specialist accuracy (ER2 measured ~7 px on recorded real-run
  frames) and per-step freshness (the point is read off the CURRENT frame)
  at the cost of one extra network/GPU call per step.

The converters below (:func:`annotations_from_robointer` and friends)
map ``__robointer`` blocks to and from ``Annotation`` lists, with the
correction ROLE ("contact_point", "placement_proposal", ...) carried in
the label hint.  ``planning/robointer_gate.py`` selects the route via
``planning.robointer.source`` (``planner`` | ``er2`` | ``molmo`` |
``off``, env ``$SPARK_ROBOINTER_SOURCE``): ``planner`` trusts the inline
blocks (the default); a provider name re-asks that
provider for each inline block's contact/placement point and falls back
to the inline value on any provider trouble; ``off`` publishes nothing.
Either way the SAME bounded-correction guard (``robointer_consume``)
sits between the point and the arm.  ``scripts/probe_robointer_vs_er2.py``
measures the two routes side by side on recorded frames.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import List, Optional, Protocol, Sequence, Tuple, runtime_checkable

import numpy as np

logger = logging.getLogger(__name__)

# The ER2 (and Molmo 2) grid: coordinates are integers in [0, 1000].
ER2_SCALE = 1000.0
# Molmo 1 / legacy <point x=".." y=".."> tags use a 0-100 grid.
MOLMO1_SCALE = 100.0

VALID_KINDS = ("point", "box", "trace")


@dataclass
class Annotation:
    """
    One 2D annotation in NORMALIZED image space, axis order ``(x, y)``.

    ``points`` length by kind:
      * ``point`` -- exactly 1,
      * ``box``   -- exactly 2 (two opposite corners, any order),
      * ``trace`` -- 2 or more, in path order.
    """

    kind: str
    points: List[Tuple[float, float]]
    provider: str = ""
    label: str = ""
    """Label HINT from the provider ("plushie", "step 3"), not a detection."""
    confidence: float = 0.0
    """Provider-reported confidence; 0.0 when the provider gives none."""
    raw: Optional[dict] = field(default=None, repr=False)
    """The provider's raw reply block, for debugging / trial_meta."""

    def __post_init__(self):
        if self.kind not in VALID_KINDS:
            raise ValueError(f"unknown annotation kind {self.kind!r}")
        n = len(self.points)
        if (
            (self.kind == "point" and n != 1)
            or (self.kind == "box" and n != 2)
            or (self.kind == "trace" and n < 2)
        ):
            raise ValueError(f"kind={self.kind!r} with {n} point(s)")
        self.points = [(float(x), float(y)) for x, y in self.points]

    # -- pixel space -------------------------------------------------------

    def to_pixels(self, image_size: Tuple[int, int]) -> List[Tuple[float, float]]:
        """``image_size`` is ``(width, height)`` -- PIL order, not numpy shape."""
        w, h = image_size
        return [(x * float(w), y * float(h)) for x, y in self.points]

    @classmethod
    def from_pixels(
        cls,
        kind: str,
        pixels: Sequence[Tuple[float, float]],
        image_size: Tuple[int, int],
        **kw,
    ) -> "Annotation":
        w, h = image_size
        pts = [(float(u) / float(w), float(v) / float(h)) for u, v in pixels]
        return cls(kind=kind, points=pts, **kw)

    def clamped(self) -> "Annotation":
        """Same annotation with every coordinate clipped into [0, 1]."""
        pts = [(float(np.clip(x, 0.0, 1.0)), float(np.clip(y, 0.0, 1.0))) for x, y in self.points]
        return Annotation(
            kind=self.kind,
            points=pts,
            provider=self.provider,
            label=self.label,
            confidence=self.confidence,
            raw=self.raw,
        )

    def in_bounds(self, slack: float = 0.02) -> bool:
        return all(-slack <= v <= 1.0 + slack for p in self.points for v in p)

    # -- convenience accessors --------------------------------------------

    @property
    def point(self) -> Tuple[float, float]:
        """The single ``(x, y)`` of a point annotation (or a trace's start)."""
        return self.points[0]

    def box_xyxy(self) -> Tuple[float, float, float, float]:
        """Normalized ``(x1, y1, x2, y2)`` with corners sorted (box kind)."""
        (xa, ya), (xb, yb) = self.points[0], self.points[1]
        return (min(xa, xb), min(ya, yb), max(xa, xb), max(ya, yb))


# ---------------------------------------------------------------------------
# ER2 conversions -- [y, x] on a 0-1000 grid.  The ONLY transpose site.
# ---------------------------------------------------------------------------


def point_from_er2(yx: Sequence[float]) -> Tuple[float, float]:
    """ER2 ``[y, x]`` (0-1000) -> internal normalized ``(x, y)``."""
    y, x = float(yx[0]), float(yx[1])
    return (x / ER2_SCALE, y / ER2_SCALE)


def point_to_er2(xy: Tuple[float, float]) -> List[int]:
    """Internal normalized ``(x, y)`` -> ER2 ``[y, x]`` (0-1000)."""
    x, y = xy
    return [int(round(y * ER2_SCALE)), int(round(x * ER2_SCALE))]


def box_from_er2(box: Sequence[float]) -> List[Tuple[float, float]]:
    """ER2 ``[ymin, xmin, ymax, xmax]`` (0-1000) -> two ``(x, y)`` corners."""
    ymin, xmin, ymax, xmax = (float(v) for v in box[:4])
    return [(xmin / ER2_SCALE, ymin / ER2_SCALE), (xmax / ER2_SCALE, ymax / ER2_SCALE)]


def box_to_er2(ann: Annotation) -> List[int]:
    """Box annotation -> ER2 ``[ymin, xmin, ymax, xmax]`` (0-1000)."""
    x1, y1, x2, y2 = ann.box_xyxy()
    return [
        int(round(y1 * ER2_SCALE)),
        int(round(x1 * ER2_SCALE)),
        int(round(y2 * ER2_SCALE)),
        int(round(x2 * ER2_SCALE)),
    ]


def trace_from_er2(yx_list: Sequence[Sequence[float]]) -> List[Tuple[float, float]]:
    return [point_from_er2(p) for p in yx_list]


# ---------------------------------------------------------------------------
# Molmo conversions -- (x, y), column first, 0-100 or 0-1000 grid.
# ---------------------------------------------------------------------------


def point_from_molmo(xy: Sequence[float], scale: float) -> Tuple[float, float]:
    """Molmo ``(x, y)`` at ``scale`` (100 or 1000) -> internal normalized."""
    x, y = float(xy[0]), float(xy[1])
    return (x / scale, y / scale)


# ---------------------------------------------------------------------------
# RoboInter (inline planner) conversions -- (x, y), x first, 0-1000 grid.
#
# The planner's __robointer blocks and this module's Annotations are the same
# geometry in two shapes.  These converters are the ONLY bridge; the grid and
# axis order are robointer.py's (COORD_SCALE == 1000, x then y), which is also
# this module's internal order, so no transpose happens here; the round-trip
# tests in test_annotation_unify.py hold that fixed.
# ---------------------------------------------------------------------------

# Provider name for the inline route: annotations converted from a planning
# reply's __robointer block.
PLANNER_PROVIDER = "planner"

# Correction roles, i.e. which NodeAnnotation field a converted Annotation
# came from / returns to.  Carried in the Annotation's label hint as
# "<role>" or "<role>:<target label>" -- see role_label / split_role_label.
ROBOINTER_ROLES = (
    "contact_point",
    "placement_proposal",
    "object_box",
    "affordance_box",
    "trace",
)

_ROLE_KIND = {
    "contact_point": "point",
    "placement_proposal": "box",
    "object_box": "box",
    "affordance_box": "box",
    "trace": "trace",
}

# A provider answers a PLACEMENT query with a point; NodeAnnotation stores a
# placement_proposal as a box.  The point is widened into a box of this
# half-width (normalized) so it survives the round trip -- its center, which
# is all the consumer reads, is exactly the point.
PLACEMENT_POINT_HALF = 0.02


def role_label(role: str, target: str = "") -> str:
    """Label hint for a converted annotation: ``"contact_point:knife 1"``."""
    if role not in ROBOINTER_ROLES:
        raise ValueError(f"unknown robointer role {role!r}")
    return f"{role}:{target}" if target else role


def split_role_label(label: str) -> Tuple[Optional[str], str]:
    """``"contact_point:knife 1"`` -> ``("contact_point", "knife 1")``.

    A label that does not start with a known role returns ``(None, label)``
    -- e.g. a rescue-rung annotation whose label is the object text itself.
    """
    head, _, rest = (label or "").partition(":")
    head = head.strip()
    if head in ROBOINTER_ROLES:
        return head, rest.strip()
    return None, (label or "").strip()


def _infer_role(ann: Annotation) -> str:
    """Role for an annotation with no role prefix, from its kind."""
    return {"point": "contact_point", "box": "object_box", "trace": "trace"}[ann.kind]


def annotations_from_robointer(node_ann, provider: str = PLANNER_PROVIDER) -> List[Annotation]:
    """One planner ``NodeAnnotation`` -> provider-agnostic ``Annotation`` list.

    Duck-typed on :class:`spark_real.planning.robointer.NodeAnnotation`
    (``contact_point.x/.y``, boxes with ``x1..y2``, ``trace`` of points), so
    perception never imports planning on this path.  Coordinates copy over
    unchanged: both sides store normalized ``(x, y)``, x first.

    ``subtask``/``camera`` ride along in ``raw`` so a later
    :func:`robointer_from_annotations` can restore them.
    """
    if node_ann is None:
        return []
    target = getattr(node_ann, "label", None) or ""
    meta = {
        "role": None,  # set per annotation below
        "target": target or None,
        "camera": getattr(node_ann, "camera", None),
        "subtask": getattr(node_ann, "subtask", None),
        "primitive_skill": getattr(node_ann, "primitive_skill", None),
    }

    def make(kind: str, role: str, points) -> Annotation:
        return Annotation(
            kind=kind,
            points=points,
            provider=provider,
            label=role_label(role, target),
            raw={**meta, "role": role},
        )

    out: List[Annotation] = []
    pt = getattr(node_ann, "contact_point", None)
    if pt is not None:
        out.append(make("point", "contact_point", [(pt.x, pt.y)]))
    for role in ("placement_proposal", "object_box", "affordance_box"):
        box = getattr(node_ann, role, None)
        if box is not None:
            out.append(make("box", role, [(box.x1, box.y1), (box.x2, box.y2)]))
    trace = getattr(node_ann, "trace", None)
    if trace:
        out.append(make("trace", "trace", [(p.x, p.y) for p in trace]))
    return out


def robointer_from_annotations(
    annotations: Sequence[Annotation],
    label: Optional[str] = None,
    camera: Optional[str] = None,
    subtask: Optional[str] = None,
):
    """``Annotation`` list -> one planner ``NodeAnnotation`` (the inverse).

    Role comes from each annotation's label prefix (``"contact_point:..."``),
    falling back to the kind (point -> contact_point, box -> object_box,
    trace -> trace).  A POINT-kind ``placement_proposal`` -- what an
    out-of-band provider answers a placement query with -- is widened into a
    ``PLACEMENT_POINT_HALF`` box centred on the point, because that is the
    shape ``robointer_consume.placement_target_xyz`` reads (it only ever uses
    the box's center, so nothing is lost).

    Duplicate roles: the first wins, later ones are logged and dropped.
    ``label``/``camera``/``subtask`` override anything found in the
    annotations' ``raw`` metadata.
    """
    from spark_real.planning.robointer import Box2D, NodeAnnotation, Point2D

    ann_out = NodeAnnotation()
    for a in annotations or []:
        role, target = split_role_label(a.label)
        if role is None:
            role = _infer_role(a)
        raw = a.raw or {}
        if ann_out.label is None and (target or raw.get("target")):
            ann_out.label = target or raw.get("target")
        if ann_out.camera is None and raw.get("camera"):
            ann_out.camera = raw.get("camera")
        if ann_out.subtask is None and raw.get("subtask"):
            ann_out.subtask = raw.get("subtask")
        if ann_out.primitive_skill is None and raw.get("primitive_skill"):
            ann_out.primitive_skill = raw.get("primitive_skill")

        if getattr(ann_out, role) is not None:
            logger.warning("robointer_from_annotations: duplicate %s dropped", role)
            continue
        if role == "contact_point":
            if a.kind != "point":
                logger.warning("contact_point annotation must be a point, got %s", a.kind)
                continue
            x, y = a.point
            ann_out.contact_point = Point2D(x, y)
        elif role in ("placement_proposal", "object_box", "affordance_box"):
            if a.kind == "box":
                (x1, y1), (x2, y2) = a.points
                setattr(ann_out, role, Box2D(x1, y1, x2, y2).ordered())
            elif a.kind == "point" and role == "placement_proposal":
                x, y = a.clamped().point
                h = PLACEMENT_POINT_HALF
                ann_out.placement_proposal = Box2D(
                    float(np.clip(x - h, 0.0, 1.0)),
                    float(np.clip(y - h, 0.0, 1.0)),
                    float(np.clip(x + h, 0.0, 1.0)),
                    float(np.clip(y + h, 0.0, 1.0)),
                ).ordered()
            else:
                logger.warning("%s annotation must be a box, got %s", role, a.kind)
        elif role == "trace":
            if a.kind != "trace":
                logger.warning("trace annotation must be a trace, got %s", a.kind)
                continue
            ann_out.trace = [Point2D(x, y) for x, y in a.points]
    if label is not None:
        ann_out.label = label
    if camera is not None:
        ann_out.camera = camera
    if subtask is not None:
        ann_out.subtask = subtask
    return ann_out


def annotations_from_robointer_block(
    raw: dict, provider: str = PLANNER_PROVIDER
) -> List[Annotation]:
    """One raw ``__robointer`` mapping (0-1000 grid, x first) -> Annotations.

    Parsing -- the scale decision, field validation, pruning -- is delegated
    to ``robointer.parse_annotation`` so the grid convention has exactly one
    owner.  Malformed fields are dropped there with warnings; an unusable
    block returns ``[]``.
    """
    from spark_real.planning.robointer import parse_annotation

    ann, issues = parse_annotation(raw)
    for issue in issues:
        logger.warning("[robointer->annotation] %s", issue)
    if ann is None:
        return []
    return annotations_from_robointer(ann, provider=provider)


def robointer_block_from_annotations(
    annotations: Sequence[Annotation],
    label: Optional[str] = None,
    camera: Optional[str] = None,
    subtask: Optional[str] = None,
) -> dict:
    """``Annotation`` list -> a raw ``__robointer`` mapping on the 0-1000 grid.

    Emits the planner-native integer grid (x first), i.e. what
    ``robointer_prompt`` teaches Gemini to write, so the output is a valid
    block for ``parse_annotation`` and round-trips exactly (integers survive
    /1000 -> *1000).
    """
    node_ann = robointer_from_annotations(annotations, label=label, camera=camera, subtask=subtask)
    out: dict = {}
    if node_ann.subtask:
        out["subtask"] = node_ann.subtask
    if node_ann.primitive_skill:
        out["primitive_skill"] = node_ann.primitive_skill
    if node_ann.label:
        out["label"] = node_ann.label
    if node_ann.camera:
        out["camera"] = node_ann.camera
    for role in ("object_box", "affordance_box", "placement_proposal"):
        box = getattr(node_ann, role)
        if box is not None:
            out[role] = box.to_permille_pair()
    if node_ann.contact_point is not None:
        out["contact_point"] = node_ann.contact_point.to_permille()
    if node_ann.trace:
        out["trace"] = [p.to_permille() for p in node_ann.trace]
    return out


# ---------------------------------------------------------------------------
# Provider protocol
# ---------------------------------------------------------------------------


@runtime_checkable
class AnnotationProvider(Protocol):
    """
    A source of image annotations.

    ``annotate`` must be fail-open: on any provider trouble (no key, no
    GPU, network, parse junk) it returns ``[]`` and logs -- callers use
    it inside perception rescue chains where a raised exception costs a
    whole trial.
    """

    name: str

    def available(self) -> bool:
        """Cheap capability check (key present, env exists); no network."""
        ...

    def annotate(self, image: np.ndarray, query: str, kind: str = "point") -> List[Annotation]:
        """
        Annotate ``image`` (RGB, HxWx3 uint8) for ``query``.

        ``kind`` is the REQUESTED kind ('point', 'box' or 'trace'); the
        provider returns annotations of that kind when it can, else [].
        """
        ...


class HumanUIProvider:
    """
    The human operator as an annotation provider.

    Clicks and drawn boxes arrive asynchronously through the web UI
    (/api/detect_click, /api/detect_box) -- there is nothing to call at
    annotate() time, so this provider only wraps already-collected pixel
    input into :class:`Annotation` via :meth:`from_click` / :meth:`from_box`.
    ``annotate`` itself reports nothing: a rescue rung configured with the
    human provider simply falls through to its on-miss policy (which for
    the real pipeline is ``operator_click`` -- the UI path).
    """

    name = "human"

    def available(self) -> bool:
        return True

    def annotate(self, image, query, kind: str = "point") -> List[Annotation]:
        return []

    @staticmethod
    def from_click(u: float, v: float, image_size: Tuple[int, int], label: str = "") -> Annotation:
        return Annotation.from_pixels(
            "point", [(u, v)], image_size, provider="human", label=label, confidence=1.0
        )

    @staticmethod
    def from_box(
        x1: float, y1: float, x2: float, y2: float, image_size: Tuple[int, int], label: str = ""
    ) -> Annotation:
        return Annotation.from_pixels(
            "box", [(x1, y1), (x2, y2)], image_size, provider="human", label=label, confidence=1.0
        )


# ---------------------------------------------------------------------------
# Provider registry
# ---------------------------------------------------------------------------

_PROVIDER_CACHE: dict = {}


def get_provider(name: str, **kwargs) -> Optional[AnnotationProvider]:
    """
    Build (and cache per-process) a provider by name: human | er2 | molmo.

    Fail-open: an unknown name or a constructor error returns ``None`` --
    the rescue rungs treat that exactly like an unavailable provider.
    Imports are deferred so the base perception path never pays for (or
    breaks on) provider-only dependencies.
    """
    key = (name, tuple(sorted(kwargs.items())))
    if key in _PROVIDER_CACHE:
        return _PROVIDER_CACHE[key]
    provider: Optional[AnnotationProvider] = None
    try:
        if name == "human":
            provider = HumanUIProvider()
        elif name == "er2":
            from spark_real.perception.providers.er2 import ER2Provider

            provider = ER2Provider(**kwargs)
        elif name == "molmo":
            from spark_real.perception.providers.molmo import MolmoProvider

            provider = MolmoProvider(**kwargs)
        else:
            logger.warning("unknown annotation provider %r", name)
    except Exception as exc:  # noqa: BLE001 - registry must fail open
        logger.warning("annotation provider %r failed to build: %s", name, exc)
        provider = None
    _PROVIDER_CACHE[key] = provider
    return provider


# ---------------------------------------------------------------------------
# Trace -> executor waypoints
# ---------------------------------------------------------------------------


def trace_to_waypoints_3d(
    ann: Annotation,
    depth: np.ndarray,
    cam_pos: np.ndarray,
    cam_mat: np.ndarray,
    fovy_deg: float,
    plane_depth_m: Optional[float] = None,
) -> List[np.ndarray]:
    """
    Back-project a trace annotation into world-frame waypoints.

    Uses the SAME pinhole + camera-frame convention as the sim detection
    path (``_detect_via_service`` / ``_detect_with_rendered_depth``):
    ``f = h / (2 tan(fovy/2))``, camera looks down -Z, +Y up in image.

    Every trace point is projected at ONE depth -- the object's depth
    plane -- not at the per-pixel depth: a drawn trace crosses free space
    where per-pixel depth reads the table or a far wall, which would fold
    the path onto the background.  ``plane_depth_m`` overrides the plane;
    otherwise the median valid depth in a 5x5 window at the trace START
    (which the providers put on the manipulated object) is used.

    Returns [] when no usable depth plane exists -- fail-open, like the
    providers themselves.
    """
    if ann.kind != "trace":
        raise ValueError(f"expected a trace annotation, got {ann.kind!r}")
    h, w = depth.shape[:2]
    px = ann.to_pixels((w, h))

    d_plane = plane_depth_m
    if d_plane is None:
        u0, v0 = int(round(px[0][0])), int(round(px[0][1]))
        u_lo, u_hi = max(0, u0 - 2), min(w, u0 + 3)
        v_lo, v_hi = max(0, v0 - 2), min(h, v0 + 3)
        window = np.asarray(depth[v_lo:v_hi, u_lo:u_hi], dtype=np.float64)
        valid = window[(window > 0.01) & (window < 10.0)]
        if valid.size == 0:
            logger.warning("trace_to_waypoints_3d: no valid depth at trace start")
            return []
        d_plane = float(np.median(valid))

    f = h / (2.0 * np.tan(np.deg2rad(fovy_deg) / 2.0))
    cam_mat = np.asarray(cam_mat, dtype=np.float64).reshape(3, 3)
    cam_pos = np.asarray(cam_pos, dtype=np.float64).reshape(3)
    out: List[np.ndarray] = []
    for u, v in px:
        x_c = (u - w / 2.0) * d_plane / f
        y_c = -(v - h / 2.0) * d_plane / f
        z_c = -d_plane
        out.append(cam_mat @ np.array([x_c, y_c, z_c]) + cam_pos)
    return out


def waypoints_to_move_nodes(
    waypoints_3d: Sequence[np.ndarray],
    start_xyz: Optional[np.ndarray] = None,
    min_step_m: float = 0.01,
) -> List[dict]:
    """
    Expand world-frame waypoints into chained ``move_relative`` BT nodes.

    ``move_relative`` is the one motion primitive both executors (sim
    libero_pro and the real ScoreExecutor) already dispatch, so a trace
    needs no new primitive: each waypoint becomes the delta from its
    predecessor (from ``start_xyz`` for the first when given, else the
    first waypoint is treated as the current pose and skipped).  Steps
    shorter than ``min_step_m`` are merged into the next one -- SAM3
    centroid noise puts sub-centimetre jitter on drawn traces.
    """
    wps = [np.asarray(p, dtype=np.float64).reshape(3) for p in waypoints_3d]
    if not wps:
        return []
    if start_xyz is not None:
        prev = np.asarray(start_xyz, dtype=np.float64).reshape(3)
    else:
        prev, wps = wps[0], wps[1:]
    nodes: List[dict] = []
    for wp in wps:
        delta = wp - prev
        if float(np.linalg.norm(delta)) < min_step_m:
            continue  # merged: prev is kept, so the next delta absorbs this
        nodes.append(
            {
                "type": "move_relative",
                "params": {
                    "dx": round(float(delta[0]), 4),
                    "dy": round(float(delta[1]), 4),
                    "dz": round(float(delta[2]), 4),
                },
            }
        )
        prev = wp
    return nodes

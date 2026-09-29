"""
Declarative per-task SAM3 prompt registry and same-instance disambiguation.

The same instruction must produce the same prompts and the same instance
labels on every run, or a cached behaviour tree that says ``blue block 1``
grasps a different physical block each episode. A registry hit skips the LLM
prompt generation (which runs at non-zero temperature), and ``order_by``
replaces SAM3 confidence-rank instance numbering with a geometric total order.

Data lives in ``configs/tasks/<slug>.yaml``, one file per task, glob
discovered. The directory is overridable by the ``SPARK_TASK_PROMPTS`` env var
or by passing ``path=`` (a relative path resolves against ``configs/``, which
is how a ``perception.task_prompts_dir`` key in the family YAML is threaded
in).

Schema (every key optional except ``task`` and ``groups``)::

    task: "stack the blocks of same color"
    aliases: ["stack the same colored blocks"]
    bt: "bt_seeds/stack_the_blocks_of_same_color.yaml"
    ordering_camera: birdview       # frame whose image axes define the index
    per_instance_z: true            # disable merge-time median-Z flattening
    groups:
      - text: "blue block"
        multi_instance: true
        expect: {min: 0, max: 2}
        min_conf: 0.15
        order_by: [centroid_x, centroid_y]
        alt_prompts: ["blue cube"]
    require: {any_group_count: 2, alias_as: "same color block"}
    disambiguate: {hsv_cluster: false, hue_tol_deg: 15.0}
    fallback:
      alt_prompts: {"blue block": ["blue cube"]}
      relax_secondary_score: 0.30
      llm_propose_prompts: true     # ask the planner LLM to name the missing
      llm_max_prompts: 4            # group, AFTER the two rungs above fail
      on_mismatch: abort            # abort | operator_click | best_effort

Entry points
------------
``load_registry(path=None)``          -> :class:`Registry`
``Registry.lookup(instruction)``      -> :class:`TaskSpec` | ``None``
``TaskSpec.order(detections)``        -> ordered + labelled detections
``TaskSpec.resolve(detections)``      -> :class:`LabelResolution` (adds the
                                         count gate and the mismatch policy)

Instruction matching
--------------------
Every key derived from an instruction comes from ``spark_real.bt_library``:
``normalize_instruction`` (exact tier), ``near_exact_key``
(spelling/plural/filler tier) and ``_tokens`` / ``_jaccard`` (similarity
tier). This module must never grow its own dialect: an instruction that
resolves in the BT cache but not here executes a cached tree against
un-grounded prompts.

The tiers mirror ``BTLibrary.lookup``'s 2-4 (the pin tier has no analogue).
This module is stricter at the last tier: a near-tie between two specs is
refused, because ``stack the blue block on the gray block`` and its mirror
tokenize identically. The near-exact tier preserves content-word order, so it
separates that pair.
"""

from __future__ import annotations

import logging
import os
import re
from dataclasses import dataclass, field, replace
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple

import yaml

from spark_real.bt_library import (
    _jaccard,
    _tokens,
    near_exact_key,
    normalize_instruction,
)
from spark_real.config import CONFIGS_DIR

logger = logging.getLogger(__name__)

# Directory of task YAMLs, relative to configs/. Overridable per deployment.
DEFAULT_TASK_PROMPTS_DIR = "tasks"
TASK_PROMPTS_ENV = "SPARK_TASK_PROMPTS"

# Jaccard floor for a fuzzy registry hit. Below this, no spec is returned and
# the caller falls through to its existing prompt path.
DEFAULT_MIN_SIMILARITY = 0.60

# Two specs whose similarity differs by less than this are treated as
# indistinguishable and the lookup is refused.
SIMILARITY_TIE_EPS = 1e-6

# SAM3 score a non-best mask must clear to be considered a second instance.
# A task may relax it via fallback.relax_secondary_score.
DEFAULT_SECONDARY_MASK_SCORE = 0.50

# Mismatch policies, in the order of decreasing strictness.
ON_MISMATCH_ABORT = "abort"
ON_MISMATCH_OPERATOR_CLICK = "operator_click"
ON_MISMATCH_BEST_EFFORT = "best_effort"
_ON_MISMATCH_VALUES = (
    ON_MISMATCH_ABORT,
    ON_MISMATCH_OPERATOR_CLICK,
    ON_MISMATCH_BEST_EFFORT,
)

_INSTANCE_SUFFIX_RE = re.compile(r"\s+\d+$")
_SLUG_RE = re.compile(r"[^a-z0-9]+")


class PromptCountMismatch(RuntimeError):
    """
    Raised when a registered task's detected object counts violate its
    declared ``expect`` ranges and the policy is ``abort``.

    Recording a mislabelled episode poisons a training set far worse than
    losing one episode, so this is the default for demo collection.
    """

    def __init__(self, resolution: "LabelResolution"):
        super().__init__(resolution.describe())
        self.resolution = resolution


# --------------------------------------------------------------------------
# small helpers
# --------------------------------------------------------------------------


def slugify(task: str) -> str:
    """
    Filename slug for a task string: "put the knife in the tray" ->
    "put_the_knife_in_the_tray".

    Used for configs/tasks/<slug>.yaml. Note this is NOT the same convention
    configs/bt_seeds uses (those drop articles), and nothing matches the two
    directories by filename -- task specs are discovered by glob and matched
    on the `task:` field inside, so the filename is documentation only.
    """
    return _SLUG_RE.sub("_", (task or "").strip().lower()).strip("_")


def base_label(label: str) -> str:
    """
    Strip a trailing instance index: "blue block 2" -> "blue block".

    Makes label resolution idempotent, so re-resolving an already-numbered
    detection set (a re-detect during recovery) does not compound suffixes.

    Normalises through the shared instruction dialect, so a prompt written
    "blue_block" in YAML and a detection labelled "Blue Block 2" reduce to the
    same key. NOTE: ``bt_label_resolver.base_label`` is a *different* function
    with the same name -- it strips the same suffix but preserves case and
    internal spacing, because its output is fed back to the operator and to
    detection-map lookups keyed on the label SAM3 actually produced. Do not
    assume the two are interchangeable.
    """
    return _INSTANCE_SUFFIX_RE.sub("", normalize_instruction(label)).strip()


def _centroid(det: Any) -> Tuple[float, float]:
    c = getattr(det, "centroid_2d", None)
    if c is None or len(c) < 2:
        return (0.0, 0.0)
    return (float(c[0]), float(c[1]))


def _world(det: Any) -> Tuple[float, float, float]:
    p = getattr(det, "position_3d", None)
    if p is None or len(p) < 3:
        return (0.0, 0.0, 0.0)
    return (float(p[0]), float(p[1]), float(p[2]))


# order_by vocabulary. Each entry maps a detection to a float.
_ORDER_KEYS = {
    "centroid_x": lambda d: _centroid(d)[0],
    "centroid_y": lambda d: _centroid(d)[1],
    "world_x": lambda d: _world(d)[0],
    "world_y": lambda d: _world(d)[1],
    "world_z": lambda d: _world(d)[2],
    "depth": lambda d: float(getattr(d, "depth_meters", 0.0) or 0.0),
    "mask_area": lambda d: float(getattr(d, "mask_area", 0) or 0),
    "confidence": lambda d: float(getattr(d, "confidence", 0.0) or 0.0),
}

# Keys whose natural direction is "largest first". Everything else ascends.
# confidence descending is the numbering when a task declares no order_by.
_DESCENDING_BY_DEFAULT = frozenset({"confidence", "mask_area"})

DEFAULT_ORDER_BY: Tuple[str, ...] = ("confidence",)

# Appended to every comparator so the order is total and therefore identical
# across runs even when the declared keys tie exactly.
_TIEBREAK_KEYS: Tuple[str, ...] = ("centroid_x", "centroid_y", "-mask_area")


def _parse_order_key(key: str) -> Tuple[str, bool]:
    """
    ("centroid_x", False) ascending, ("confidence", True) descending.
    A leading '-' forces descending, '+' forces ascending.
    """
    k = str(key).strip()
    if k.startswith("-"):
        return k[1:], True
    if k.startswith("+"):
        return k[1:], False
    return k, k in _DESCENDING_BY_DEFAULT


def _sort_key(det: Any, keys: Sequence[str]) -> tuple:
    out: List[Any] = []
    for raw in list(keys) + list(_TIEBREAK_KEYS):
        name, descending = _parse_order_key(raw)
        getter = _ORDER_KEYS.get(name)
        if getter is None:
            continue
        value = float(getter(det))
        out.append(-value if descending else value)
    # Final string tiebreak: camera role, then the base label itself. Both are
    # stable strings, so identical scenes sort identically across processes.
    out.append(str(getattr(det, "camera", "") or ""))
    out.append(base_label(getattr(det, "label", "") or ""))
    return tuple(out)


def order_detections(dets: Sequence[Any], order_by: Sequence[str]) -> List[Any]:
    """
    Deterministic total order over a set of same-label detections.
    """
    return sorted(dets, key=lambda d: _sort_key(d, order_by))


# --------------------------------------------------------------------------
# schema
# --------------------------------------------------------------------------


@dataclass(frozen=True)
class PromptGroup:
    """
    One SAM3 text prompt plus everything needed to label its instances.
    """

    text: str
    multi_instance: bool = False
    expect_min: Optional[int] = None
    expect_max: Optional[int] = None
    min_conf: float = 0.0
    order_by: Tuple[str, ...] = DEFAULT_ORDER_BY
    alt_prompts: Tuple[str, ...] = ()

    @property
    def key(self) -> str:
        return normalize_instruction(self.text)

    def accepts_label(self, label: str) -> bool:
        b = base_label(label)
        return b == self.key or b in {normalize_instruction(a) for a in self.alt_prompts}

    def count_ok(self, n: int) -> bool:
        if self.expect_min is not None and n < self.expect_min:
            return False
        if self.expect_max is not None and n > self.expect_max:
            return False
        return True

    @classmethod
    def from_yaml(cls, raw: Dict[str, Any], source: str) -> "PromptGroup":
        text = str(raw.get("text") or "").strip()
        if not text:
            raise ValueError(f"{source}: prompt group missing 'text'")
        expect = raw.get("expect") or {}
        if not isinstance(expect, dict):
            raise ValueError(f"{source}: 'expect' must be a mapping for {text!r}")
        order_by = raw.get("order_by") or list(DEFAULT_ORDER_BY)
        if isinstance(order_by, str):
            order_by = [order_by]
        for k in order_by:
            name, _ = _parse_order_key(k)
            if name not in _ORDER_KEYS:
                raise ValueError(
                    f"{source}: unknown order_by key {k!r} for {text!r}; "
                    f"known: {sorted(_ORDER_KEYS)}"
                )
        alts = raw.get("alt_prompts") or []
        if isinstance(alts, str):
            alts = [alts]
        return cls(
            text=text,
            multi_instance=bool(raw.get("multi_instance", False)),
            expect_min=None if expect.get("min") is None else int(expect["min"]),
            expect_max=None if expect.get("max") is None else int(expect["max"]),
            min_conf=float(raw.get("min_conf", 0.0) or 0.0),
            order_by=tuple(str(k) for k in order_by),
            alt_prompts=tuple(str(a) for a in alts),
        )


@dataclass(frozen=True)
class FallbackSpec:
    """
    What to do when the scene does not match the declared arity.
    """

    alt_prompts: Dict[str, Tuple[str, ...]] = field(default_factory=dict)
    relax_secondary_score: Optional[float] = None
    on_mismatch: str = ON_MISMATCH_ABORT
    # Last rung before on_mismatch: show the frame to the planner LLM and let
    # it name the missing group in SAM3's vocabulary. Default ON: it costs an
    # API call and ~1s of latency, but is only reached after the two free
    # rungs (relax the secondary-mask score, then alt_prompts) have failed,
    # when the alternative is on_mismatch (abort for the shipped task set).
    llm_propose_prompts: bool = True
    # Cap on proposals accepted per missing group. Each one is another SAM3
    # pass over every camera, so a chatty model must not be able to turn one
    # retry into twenty. Four is enough room for a couple of genuinely
    # different guesses at what the top-down blob is.
    llm_max_prompts: int = 4

    @classmethod
    def from_yaml(cls, raw: Optional[Dict[str, Any]], source: str) -> "FallbackSpec":
        raw = raw or {}
        alt = {}
        for k, v in (raw.get("alt_prompts") or {}).items():
            alt[normalize_instruction(k)] = tuple(
                str(x) for x in (v if isinstance(v, list) else [v])
            )
        policy = str(raw.get("on_mismatch", ON_MISMATCH_ABORT)).strip().lower()
        if policy not in _ON_MISMATCH_VALUES:
            raise ValueError(
                f"{source}: on_mismatch must be one of {_ON_MISMATCH_VALUES}, got {policy!r}"
            )
        relax = raw.get("relax_secondary_score")
        max_prompts = int(raw.get("llm_max_prompts", 4) or 0)
        if max_prompts < 0:
            raise ValueError(f"{source}: llm_max_prompts must be >= 0")
        return cls(
            alt_prompts=alt,
            relax_secondary_score=None if relax is None else float(relax),
            on_mismatch=policy,
            llm_propose_prompts=bool(raw.get("llm_propose_prompts", True)),
            llm_max_prompts=max_prompts,
        )


@dataclass(frozen=True)
class DisambiguateSpec:
    """
    Optional HSV cross-check. Off by default: for the shipped task set the
    count gate plus geometric ordering is sufficient, and colour clustering is
    only worth enabling if the count gate proves flaky on the rig.
    """

    hsv_cluster: bool = False
    hue_tol_deg: float = 15.0
    min_saturation: float = 40.0
    min_value: float = 40.0

    @classmethod
    def from_yaml(cls, raw: Optional[Dict[str, Any]]) -> "DisambiguateSpec":
        raw = raw or {}
        return cls(
            hsv_cluster=bool(raw.get("hsv_cluster", False)),
            hue_tol_deg=float(raw.get("hue_tol_deg", 15.0)),
            min_saturation=float(raw.get("min_saturation", 40.0)),
            min_value=float(raw.get("min_value", 40.0)),
        )


@dataclass(frozen=True)
class TaskSpec:
    """
    Frozen perception contract for one task string.
    """

    task: str
    groups: Tuple[PromptGroup, ...]
    aliases: Tuple[str, ...] = ()
    bt: Optional[str] = None
    ordering_camera: Optional[str] = None
    per_instance_z: bool = False
    require: Dict[str, Any] = field(default_factory=dict)
    fallback: FallbackSpec = field(default_factory=FallbackSpec)
    disambiguate: DisambiguateSpec = field(default_factory=DisambiguateSpec)
    source_path: Optional[str] = None
    # Optional `verify:` block, same schema as the planner's (see
    # control/success_predicates.parse_verify_block). This is the operator's
    # no-code success override and the ONLY way to set occlusion_ok on a
    # cached tree, which never re-plans. Passed through verbatim; it is
    # validated at verify time so a typo rejects the block rather than the task.
    verify: Optional[Dict[str, Any]] = None

    # -- prompt surface ---------------------------------------------------

    @property
    def slug(self) -> str:
        return slugify(self.task)

    @property
    def keys(self) -> Tuple[str, ...]:
        """
        Every normalised instruction string that resolves to this spec by
        exact match. Same construction as ``BTEntry.keys``.
        """
        out = [normalize_instruction(t) for t in (self.task, *self.aliases)]
        return tuple(k for k in out if k)

    @property
    def near_keys(self) -> Tuple[str, ...]:
        """
        Every near-exact key (spelling, plurals and filler folded, content-word
        order preserved) that resolves to this spec. Same construction as
        ``BTEntry.near_keys``, so an instruction the BT cache resolves at its
        ``normalized`` tier resolves here too.
        """
        out = [near_exact_key(t) for t in (self.task, *self.aliases)]
        return tuple(k for k in out if k)

    @property
    def prompts(self) -> List[str]:
        """
        SAM3 prompts in declaration order. Declaration order is the order the
        registry author intended and is stable across runs.
        """
        return [g.text for g in self.groups]

    @property
    def multi_instance_prompts(self) -> set:
        """
        Prompts allowed to return more than the top-1 mask. multi_instance
        defaults to False everywhere in spark_real, so any group that can have
        two instances must say so explicitly or the second is never seen.
        """
        return {g.text for g in self.groups if g.multi_instance}

    def group_for(self, label: str) -> Optional[PromptGroup]:
        for g in self.groups:
            if g.accepts_label(label):
                return g
        return None

    def alt_prompts_for(self, labels: Iterable[str]) -> List[str]:
        """
        Retry prompts for the named groups: the group's own alt_prompts plus
        anything the task-level fallback map declares for it.
        """
        out: List[str] = []
        for label in labels:
            g = self.group_for(label)
            key = g.key if g is not None else base_label(label)
            for alt in (g.alt_prompts if g is not None else ()):
                if alt not in out:
                    out.append(alt)
            for alt in self.fallback.alt_prompts.get(key, ()):
                if alt not in out:
                    out.append(alt)
        return out

    # -- resolution -------------------------------------------------------

    def order(self, detections: Sequence[Any]) -> List[Any]:
        """
        Ordered, canonically-labelled detections. Convenience wrapper around
        :meth:`resolve` for callers that do not care about the count gate.
        """
        return self.resolve(detections).detections

    def resolve(self, detections: Sequence[Any]) -> "LabelResolution":
        return resolve_labels(self, detections)

    def with_alt_prompts(self, extra: Dict[str, Sequence[str]]) -> "TaskSpec":
        """
        Copy of this spec whose named groups also answer to ``extra`` prompts.

        ``extra`` maps a group's declared ``text`` (or any label that group
        already accepts) to additional prompt strings. This is the ONLY
        mechanism by which a prompt discovered at run time (an LLM proposal)
        becomes indistinguishable from a hand-written ``alt_prompts`` entry:
        :meth:`PromptGroup.accepts_label` consults ``alt_prompts``, so a
        detection returned under the new prompt is claimed by the group and
        relabelled to the canonical name in :func:`resolve_labels`.

        The spec itself is frozen and shared (the registry caches it for the
        process), so this returns a copy and never mutates the registered one.
        """
        if not extra:
            return self
        wanted = {}
        for key, prompts in extra.items():
            wanted[base_label(key)] = [str(p) for p in prompts if str(p).strip()]
        groups = []
        for group in self.groups:
            adds = wanted.get(group.key, [])
            known = {group.key} | {normalize_instruction(a) for a in group.alt_prompts}
            adds = [a for a in adds if normalize_instruction(a) not in known]
            if adds:
                group = replace(group, alt_prompts=group.alt_prompts + tuple(adds))
            groups.append(group)
        return replace(self, groups=tuple(groups))

    # -- construction -----------------------------------------------------

    @classmethod
    def from_yaml(cls, raw: Dict[str, Any], source: str) -> "TaskSpec":
        task = str(raw.get("task") or "").strip()
        if not task:
            raise ValueError(f"{source}: missing 'task'")
        groups_raw = raw.get("groups") or []
        if not groups_raw:
            raise ValueError(f"{source}: missing 'groups'")
        groups = tuple(PromptGroup.from_yaml(g, source) for g in groups_raw)
        seen = set()
        for g in groups:
            if g.key in seen:
                raise ValueError(f"{source}: duplicate prompt {g.text!r}")
            seen.add(g.key)
        aliases = raw.get("aliases") or []
        if isinstance(aliases, str):
            aliases = [aliases]
        require = raw.get("require") or {}
        if not isinstance(require, dict):
            raise ValueError(f"{source}: 'require' must be a mapping")
        verify = raw.get("verify")
        if verify is not None and not isinstance(verify, dict):
            raise ValueError(f"{source}: 'verify' must be a mapping")
        return cls(
            task=task,
            groups=groups,
            aliases=tuple(str(a) for a in aliases),
            bt=(str(raw["bt"]) if raw.get("bt") else None),
            ordering_camera=(str(raw["ordering_camera"]) if raw.get("ordering_camera") else None),
            per_instance_z=bool(raw.get("per_instance_z", False)),
            require=dict(require),
            fallback=FallbackSpec.from_yaml(raw.get("fallback"), source),
            disambiguate=DisambiguateSpec.from_yaml(raw.get("disambiguate")),
            source_path=source,
            verify=dict(verify) if verify else None,
        )


# --------------------------------------------------------------------------
# resolution result
# --------------------------------------------------------------------------


@dataclass
class GroupResult:
    text: str
    found: int
    expect_min: Optional[int]
    expect_max: Optional[int]
    ok: bool
    labels: List[str] = field(default_factory=list)


@dataclass
class LabelResolution:
    """
    Outcome of applying a :class:`TaskSpec` to a merged detection set.

    ``detections`` are the accepted detections with canonical, ordered labels.
    ``ok`` is the count gate; ``on_mismatch`` is the policy the caller must
    apply when it is False.
    """

    task: str
    detections: List[Any]
    groups: List[GroupResult]
    extras: List[Any] = field(default_factory=list)
    ok: bool = True
    on_mismatch: str = ON_MISMATCH_ABORT
    acting_group: Optional[str] = None
    aliases_added: List[str] = field(default_factory=list)
    dropped_low_conf: int = 0
    dropped_hsv: int = 0
    retry_prompts: List[str] = field(default_factory=list)

    @property
    def counts(self) -> Dict[str, int]:
        return {g.text: g.found for g in self.groups}

    @property
    def labels(self) -> List[str]:
        return [str(getattr(d, "label", "")) for d in self.detections]

    @property
    def mismatched_groups(self) -> List[str]:
        return [g.text for g in self.groups if not g.ok]

    def describe(self) -> str:
        parts = [
            f"{g.text}: {g.found} (expect {'*' if g.expect_min is None else g.expect_min}"
            f"..{'*' if g.expect_max is None else g.expect_max})"
            + ("" if g.ok else "  <-- MISMATCH")
            for g in self.groups
        ]
        head = f"task {self.task!r} count gate {'OK' if self.ok else 'FAILED'}"
        return head + "\n  " + "\n  ".join(parts)

    def raise_if_abort(self) -> "LabelResolution":
        """
        Enforce the strict policy. Returns self so it can be chained.
        """
        if not self.ok and self.on_mismatch == ON_MISMATCH_ABORT:
            raise PromptCountMismatch(self)
        return self


# --------------------------------------------------------------------------
# resolver
# --------------------------------------------------------------------------


def _hue_distance_deg(a: float, b: float) -> float:
    d = abs(float(a) - float(b)) % 360.0
    return min(d, 360.0 - d)


def _hsv_cluster(dets: Sequence[Any], spec: DisambiguateSpec) -> List[Any]:
    """
    Keep the largest mutually-close-hue cluster. Detections without an
    ``hsv_median`` or below the saturation/value floors are kept unconditionally
    (a gray block has no meaningful hue, so hue must not be allowed to reject
    it). Clusters, never names: colour naming belongs in the YAML, not here.
    """
    hued: List[Tuple[int, float]] = []
    for i, d in enumerate(dets):
        hsv = getattr(d, "hsv_median", None)
        if hsv is None or len(hsv) < 3:
            continue
        h, s, v = float(hsv[0]), float(hsv[1]), float(hsv[2])
        if s < spec.min_saturation or v < spec.min_value:
            continue
        hued.append((i, h))
    if len(hued) < 2:
        return list(dets)

    best: List[int] = []
    for _, anchor in hued:
        members = [i for i, h in hued if _hue_distance_deg(h, anchor) <= spec.hue_tol_deg]
        if len(members) > len(best):
            best = members
    if not best:
        return list(dets)
    keep = set(best)
    # Anything that never entered the hue vote (no colour, or too dark/gray)
    # is not a cluster outlier and stays.
    voted = {i for i, _ in hued}
    return [d for i, d in enumerate(dets) if i in keep or i not in voted]


def resolve_labels(spec: TaskSpec, detections: Sequence[Any]) -> LabelResolution:
    """
    Assign stable, deterministic instance labels to a merged detection set.

    Labels are written onto the detection objects in place (the merged set is
    already a set of copies owned by the caller) and are also returned on the
    result. Resolution is idempotent: an already-numbered set re-resolves to
    the same labels.

    Numbering rule: ``base`` when a group yielded exactly one instance,
    ``base 1 .. base N`` when it yielded several. Arity aliasing (registering
    both forms) happens downstream in the detection map, not here.
    """
    result_groups: List[GroupResult] = []
    accepted: List[Any] = []
    extras: List[Any] = []
    dropped_low_conf = 0
    dropped_hsv = 0

    claimed: set = set()
    for group in spec.groups:
        members = []
        for det in detections:
            if id(det) in claimed:
                continue
            if group.accepts_label(getattr(det, "label", "") or ""):
                claimed.add(id(det))
                members.append(det)

        kept = []
        for det in members:
            if float(getattr(det, "confidence", 0.0) or 0.0) < group.min_conf:
                dropped_low_conf += 1
                continue
            kept.append(det)

        if spec.disambiguate.hsv_cluster and len(kept) > 1:
            before = len(kept)
            kept = _hsv_cluster(kept, spec.disambiguate)
            dropped_hsv += before - len(kept)

        # Trim to the declared maximum using the group's own ordering, so the
        # instances that survive are the geometrically-first ones rather than
        # whichever SAM3 happened to score highest.
        ordered = order_detections(kept, group.order_by)
        if group.expect_max is not None and len(ordered) > group.expect_max:
            extras.extend(ordered[group.expect_max :])
            ordered = ordered[: group.expect_max]

        labels = []
        for idx, det in enumerate(ordered):
            label = group.text if len(ordered) == 1 else f"{group.text} {idx + 1}"
            det.label = label
            labels.append(label)
        accepted.extend(ordered)

        result_groups.append(
            GroupResult(
                text=group.text,
                found=len(ordered),
                expect_min=group.expect_min,
                expect_max=group.expect_max,
                ok=group.count_ok(len(ordered)),
                labels=labels,
            )
        )

    # Detections for prompts the task never declared (e.g. an operator typed an
    # extra prompt). Left untouched and passed through.
    for det in detections:
        if id(det) not in claimed:
            extras.append(det)

    ok = all(g.ok for g in result_groups)

    # require: pick the acting group and optionally alias it to a role name so
    # a cached BT can address "the pair" without knowing which colour it is.
    acting: Optional[str] = None
    aliases_added: List[str] = []
    want = spec.require.get("any_group_count")
    if want is not None:
        want = int(want)
        matches = [g for g in result_groups if g.found == want]
        if len(matches) == 1:
            acting = matches[0].text
        elif len(matches) > 1:
            ok = False
            logger.warning(
                "task %r: require.any_group_count=%d is ambiguous, groups %s all match",
                spec.task,
                want,
                [g.text for g in matches],
            )
        else:
            ok = False
            logger.warning(
                "task %r: require.any_group_count=%d unsatisfied, counts=%s",
                spec.task,
                want,
                {g.text: g.found for g in result_groups},
            )

    alias_as = spec.require.get("alias_as")
    if acting is not None and alias_as:
        acting_group = next(g for g in result_groups if g.text == acting)
        by_label = {str(getattr(d, "label", "")): d for d in accepted}
        for idx, label in enumerate(acting_group.labels):
            det = by_label.get(label)
            if det is None:
                continue
            alias = str(alias_as) if len(acting_group.labels) == 1 else f"{alias_as} {idx + 1}"
            existing = list(getattr(det, "role_labels", None) or [])
            if alias not in existing:
                existing.append(alias)
            det.role_labels = existing
            aliases_added.append(alias)

    min_total = spec.require.get("min_total")
    if min_total is not None and len(accepted) < int(min_total):
        ok = False

    return LabelResolution(
        task=spec.task,
        detections=accepted,
        groups=result_groups,
        extras=extras,
        ok=ok,
        on_mismatch=spec.fallback.on_mismatch,
        acting_group=acting,
        aliases_added=aliases_added,
        dropped_low_conf=dropped_low_conf,
        dropped_hsv=dropped_hsv,
        retry_prompts=spec.alt_prompts_for([g.text for g in result_groups if not g.ok]),
    )


# --------------------------------------------------------------------------
# registry
# --------------------------------------------------------------------------


@dataclass
class Registry:
    """
    All task specs discovered under one directory.
    """

    specs: Tuple[TaskSpec, ...] = ()
    root: Optional[Path] = None
    min_similarity: float = DEFAULT_MIN_SIMILARITY

    def __len__(self) -> int:
        return len(self.specs)

    @property
    def tasks(self) -> List[str]:
        return [s.task for s in self.specs]

    def get(self, task: str) -> Optional[TaskSpec]:
        """
        Exact (normalised) task-string or alias lookup. No similarity.
        """
        want = normalize_instruction(task)
        if not want:
            return None
        for spec in self.specs:
            if want in spec.keys:
                return spec
        return None

    def get_near(self, task: str) -> Optional[TaskSpec]:
        """
        Near-exact lookup: spelling variants, plurals and filler words folded,
        content-word order preserved. ``None`` when nothing matches and when
        two specs match (never silently pick one).
        """
        want = near_exact_key(task)
        if not want:
            return None
        hits = [spec for spec in self.specs if want in spec.near_keys]
        if len(hits) == 1:
            return hits[0]
        if len(hits) > 1:
            logger.warning(
                "prompt registry: refusing ambiguous near-exact match for %r "
                "(key %r shared by %s); falling through",
                task,
                want,
                [s.task for s in hits],
            )
        return None

    def lookup(self, instruction: str) -> Optional[TaskSpec]:
        """
        Resolve an instruction to a spec, or None to fall through to the
        caller's existing prompt path.

        Three tiers, mirroring ``BTLibrary.lookup``'s 2-4 so that the prompt
        registry and the BT cache never disagree about whether two instruction
        spellings are the same task:

            1. exact    -- normalised string equality (case, punctuation,
                           underscores and whitespace folded).
            2. near     -- spelling variants, plurals and filler folded, with
                           content-word ORDER preserved.
            3. similar  -- Jaccard over the shared tokenizer, with a floor.

        A near-tie at tier 3 is refused -- for this corpus a bag-of-words score
        genuinely cannot separate "stack the blue block on the gray block" from
        its mirror, and answering anyway would silently execute the wrong plan.
        Tier 2 keeps order, so it resolves that pair rather than refusing it.
        """
        if not instruction:
            return None
        exact = self.get(instruction)
        if exact is not None:
            logger.info("prompt registry: exact hit %r", exact.task)
            return exact

        near = self.get_near(instruction)
        if near is not None:
            logger.info("prompt registry: near-exact hit %r for %r", near.task, instruction)
            return near

        query = _tokens(instruction)
        if not query:
            return None
        scored: List[Tuple[float, TaskSpec]] = []
        for spec in self.specs:
            candidates = [spec.task] + list(spec.aliases)
            best = max(_jaccard(query, _tokens(c)) for c in candidates)
            if best >= self.min_similarity:
                scored.append((best, spec))
        if not scored:
            return None
        scored.sort(key=lambda x: (-x[0], normalize_instruction(x[1].task)))
        if len(scored) > 1 and (scored[0][0] - scored[1][0]) < SIMILARITY_TIE_EPS:
            tied = [s.task for sim, s in scored if (scored[0][0] - sim) < SIMILARITY_TIE_EPS]
            logger.warning(
                "prompt registry: refusing ambiguous match for %r "
                "(%s all score %.3f); falling through",
                instruction,
                tied,
                scored[0][0],
            )
            return None
        logger.info(
            "prompt registry: fuzzy hit %r (sim=%.3f) for %r",
            scored[0][1].task,
            scored[0][0],
            instruction,
        )
        return scored[0][1]


def resolve_task_prompts_dir(path: Optional[Any] = None) -> Path:
    """
    Directory holding the task YAMLs.

    Precedence: explicit argument, then $SPARK_TASK_PROMPTS, then
    configs/tasks. A relative value resolves against configs/, which is how a
    ``perception.task_prompts_dir`` key from the family YAML is threaded in.
    """
    raw = path or os.environ.get(TASK_PROMPTS_ENV) or DEFAULT_TASK_PROMPTS_DIR
    candidate = Path(str(raw)).expanduser()
    if not candidate.is_absolute():
        candidate = Path(CONFIGS_DIR) / candidate
    return candidate


def load_registry(
    path: Optional[Any] = None,
    min_similarity: Optional[float] = None,
) -> Registry:
    """
    Glob-load every ``*.yaml`` under the task-prompts directory.

    A malformed file is logged and skipped rather than taking the server down;
    the registry is an optimisation over the LLM path, and a missing entry
    degrades to the existing behaviour instead of a boot failure.
    """
    root = resolve_task_prompts_dir(path)
    floor = DEFAULT_MIN_SIMILARITY if min_similarity is None else float(min_similarity)
    if not root.is_dir():
        logger.warning("prompt registry: %s is not a directory; registry empty", root)
        return Registry(specs=(), root=root, min_similarity=floor)

    specs: List[TaskSpec] = []
    for yaml_path in sorted(root.glob("*.yaml")):
        try:
            with open(yaml_path) as handle:
                raw = yaml.safe_load(handle) or {}
            specs.append(TaskSpec.from_yaml(raw, source=yaml_path.name))
        except Exception as exc:
            logger.error("prompt registry: skipping %s (%s)", yaml_path.name, exc)

    by_task: Dict[str, TaskSpec] = {}
    for spec in specs:
        key = normalize_instruction(spec.task)
        if key in by_task:
            logger.error(
                "prompt registry: duplicate task %r in %s and %s; keeping the first",
                spec.task,
                by_task[key].source_path,
                spec.source_path,
            )
            continue
        by_task[key] = spec

    # Two specs sharing a near-exact key are unresolvable at lookup tier 2 and
    # would fall through to the tie-refusing similarity tier, i.e. both tasks
    # would silently lose their registry entry. Say so at boot, not mid-demo.
    by_near: Dict[str, TaskSpec] = {}
    for spec in by_task.values():
        for near in spec.near_keys:
            other = by_near.get(near)
            if other is not None and other is not spec:
                logger.error(
                    "prompt registry: %s and %s share near-exact key %r; "
                    "instructions matching it will not resolve",
                    other.source_path,
                    spec.source_path,
                    near,
                )
            by_near.setdefault(near, spec)

    logger.info("prompt registry: loaded %d task(s) from %s", len(by_task), root)
    return Registry(specs=tuple(by_task.values()), root=root, min_similarity=floor)

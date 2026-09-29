"""The RoboInter extension's LIVE seam: config resolution, prompt, consume.

``planning/robointer*.py`` describes representations; this module is the one
place that puts them ON the running pipeline. Two call sites, both in
``pipeline_execution.py``:

  plan()     -> :func:`annotate_planner_image`, :func:`planner_context`,
                :func:`fewshot_examples`
                The planner image gains a 0..1000 grid, the prompt gains the
                schema section plus every detection's OBSERVED box in the
                same space the answer must use.

  execute()  -> :func:`apply_to_detection_map`
                The reply's contact points and placement proposals are lifted
                into the robot base frame and published onto the SAME
                ``detection_map`` entries the executor already reads.

Everything here is DEFAULT OFF and resolved the way ``_fusion_settings``
resolves the fusion gate: ``profile.raw`` when the server loaded an overlay,
else the packaged ``configs/<family>_default.yaml``, and ``$SPARK_ROBOINTER``
last. With the gate off, nothing is drawn, no prompt text changes, and no key
is added to the detection map.

The guard is load-bearing and lives in ``robointer_consume``: a proposal is
only ever accepted as a BOUNDED CORRECTION to a measured detection. An LLM
pixel can be wrong in a way a mask centroid cannot -- not noisy, but
confidently pointing at a different object -- so a proposal that disagrees
with perception by more than ``max_shift_m`` is refused and the detection
wins. What reaches the executor is at most an ``max_shift_m`` nudge.

WHO supplies the correction pixel is a config choice, not an architecture
(``planning.robointer.source`` / $SPARK_ROBOINTER_SOURCE): ``planner``
(default) trusts the reply's inline ``__robointer`` blocks; ``er2`` /
``molmo`` / ``human`` re-ask that out-of-band pointing provider
(``perception/annotations.py``) for the fields each inline block declared,
against the CURRENT frame, failing open per field to the inline value;
``off`` publishes no correction. Both routes converge on the same
detection-map keys through the same guard, so the executor cannot tell them
apart -- see the "two routes" section of ``perception/annotations.py`` and
``scripts/probe_robointer_vs_er2.py`` for the measured comparison.

That guard is also what makes a CACHE-SERVED tree safe here. Annotations are
hash-neutral, so a BT stored with a ``__robointer`` block is replayed verbatim
into a scene it was never read off: its pixels describe the old image. Nothing
special is done about that on purpose -- the bound is measured against THIS
run's detection, so a stale coordinate that has drifted is refused outright and
one that has not is within ``max_shift_m`` of the fresh measurement either way.
If a future change wants cached trees annotation-free, strip them on the
library WRITE path (``robointer.strip_annotations``), not here.

Failure convention, matching ``perception.box_proposals.build_proposer``: a
feature that was ASKED FOR and cannot be built RAISES. Switching this on and
getting silence would be worse than leaving it off.
"""

from __future__ import annotations

import logging
import os
from dataclasses import dataclass
from types import SimpleNamespace
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np
from PIL import Image as _PILImage

from spark_real.utils.env_flags import as_bool
from spark_real.config import family_block
from spark_real.planning.robointer import NodeAnnotation, extract_annotations
from spark_real.planning.robointer_annotate import draw_coordinate_grid
from spark_real.planning.robointer_consume import (
    DEFAULT_MAX_PLACE_SHIFT_M,
    DEFAULT_MAX_SHIFT_M,
    contact_target_xyz,
    placement_target_xyz,
)
from spark_real.planning.robointer_geometry import CameraModel
from spark_real.planning.robointer_prompt import build_robointer_context, robointer_fewshot

logger = logging.getLogger(__name__)

# Env override, read LAST so an operator can flip the extension on for one
# run without editing the family YAML.
ROBOINTER_ENV = "SPARK_ROBOINTER"

# Env override for the correction SOURCE (see RoboInterConfig.source), read
# last for the same reason.
ROBOINTER_SOURCE_ENV = "SPARK_ROBOINTER_SOURCE"

# Valid values for RoboInterConfig.source. "planner" trusts the reply's
# inline __robointer blocks (the default); a provider name re-asks
# that out-of-band pointing provider (perception/annotations.get_provider)
# for each block's contact/placement point, falling back to the inline value
# on any provider trouble; "off" publishes no correction at all.
SOURCES = ("planner", "er2", "molmo", "human", "off")

# Keys published onto a detection_map entry. Namespaced so the executor's
# `det.get(...)` is unambiguous and so an OFF run is trivially auditable:
# no key starting with `robointer` may exist.
CONTACT_KEY = "robointer_contact_xyz"
PLACEMENT_KEY = "robointer_placement_xyz"
SUBTASK_KEY = "robointer_subtask"

# Pipeline attribute -> camera name, matching PerceptionMixin.capture().
_CAL_SOURCES = (
    ("sideview", "_kinect_cal"),
    ("birdview", "_kinect2_cal"),
    ("wrist", "_realsense_cal"),
)

# Node params that name the detection an annotation is about, used when the
# annotation itself omits `label`. Ordered: a place node carries both.
_LABEL_PARAMS = ("keypoint_label", "container_label", "target_label")


@dataclass(frozen=True)
class RoboInterConfig:
    """The resolved ``planning.robointer`` block.

    ``enabled`` is the master switch; the rest only narrow what the extension
    does once it is on, so an operator can ask for the prompt half without the
    consume half (useful for harvesting annotations into episodes before
    letting them touch a motion).
    """

    enabled: bool = False
    prompt: bool = True  # schema section + observed 2D geometry
    grid_overlay: bool = True  # 0..1000 grid on the planner image
    fewshot: bool = True  # the hand-written worked example
    consume: bool = True  # publish resolved targets into the detection map
    strict: bool = False  # raise on a malformed block instead of pruning it
    source: str = "planner"  # who supplies the correction points; see SOURCES
    max_shift_m: float = DEFAULT_MAX_SHIFT_M
    max_place_shift_m: float = DEFAULT_MAX_PLACE_SHIFT_M

    @classmethod
    def from_dict(cls, cfg: Optional[dict]) -> "RoboInterConfig":
        cfg = dict(cfg or {})
        unknown = sorted(set(cfg) - set(cls.__dataclass_fields__))
        if unknown:
            msg = (
                f"unknown planning.robointer key(s) {unknown}; known: "
                f"{sorted(cls.__dataclass_fields__)}"
            )
            # Loud either way, fatal only when the operator actually asked for
            # the extension: a typo in a switched-off block must not take the
            # server down, but a typo in a switched-ON one is a setting that
            # silently does nothing.
            if bool(cfg.get("enabled", False)):
                raise RuntimeError(f"robointer is enabled but its config is unusable: {msg}")
            logger.warning("[robointer] %s (block is disabled; ignoring)", msg)
            for key in unknown:
                cfg.pop(key)
        source = str(cfg.get("source", "planner")).strip().lower() or "planner"
        if source not in SOURCES:
            msg = f"unknown planning.robointer source {source!r}; known: {list(SOURCES)}"
            if bool(cfg.get("enabled", False)):
                # Same loud-vs-fatal convention as unknown keys above: a bad
                # source on an enabled block would silently change which model
                # steers the correction.
                raise RuntimeError(f"robointer is enabled but its config is unusable: {msg}")
            logger.warning("[robointer] %s (block is disabled; using 'planner')", msg)
            source = "planner"
        return cls(
            enabled=bool(cfg.get("enabled", False)),
            prompt=bool(cfg.get("prompt", True)),
            grid_overlay=bool(cfg.get("grid_overlay", True)),
            fewshot=bool(cfg.get("fewshot", True)),
            consume=bool(cfg.get("consume", True)),
            strict=bool(cfg.get("strict", False)),
            source=source,
            max_shift_m=float(cfg.get("max_shift_m", DEFAULT_MAX_SHIFT_M)),
            max_place_shift_m=float(cfg.get("max_place_shift_m", DEFAULT_MAX_PLACE_SHIFT_M)),
        )


# --------------------------------------------------------------------------
# Config resolution (same order the fusion gate uses)
# --------------------------------------------------------------------------


def family_raw(profile=None, config=None) -> dict:
    """The family config mapping: ``profile.raw``, else the packaged YAML."""
    family = (getattr(config, "robot_family", "ur10e") or "ur10e").lower()
    return family_block(profile, family)


def resolve_settings(profile=None, config=None) -> RoboInterConfig:
    """Resolve ``planning.robointer``, env last. Absent block -> OFF."""
    raw = family_raw(profile, config)
    block = dict(((raw.get("planning") or {}).get("robointer") or {}))
    env = os.environ.get(ROBOINTER_ENV)
    if env is not None:
        block["enabled"] = as_bool(env, False)
    env_src = os.environ.get(ROBOINTER_SOURCE_ENV)
    if env_src is not None and env_src.strip():
        block["source"] = env_src.strip().lower()
    return RoboInterConfig.from_dict(block)


# --------------------------------------------------------------------------
# Prompt side
# --------------------------------------------------------------------------


def image_size(image) -> Optional[Tuple[int, int]]:
    """(width, height) of a PIL image or an HxW[xC] array, else None."""
    if image is None:
        return None
    size = getattr(image, "size", None)
    if isinstance(size, tuple) and len(size) == 2:  # PIL
        return (int(size[0]), int(size[1]))
    arr = np.asarray(image)
    if arr.ndim < 2:
        return None
    return (int(arr.shape[1]), int(arr.shape[0]))


def annotate_planner_image(cfg: RoboInterConfig, image):
    """Overlay the 0..1000 grid the answer's coordinates are measured on.

    Returns ``image`` UNCHANGED (same object) when the extension is off, so an
    off run hands the planner exactly the bytes it hands it today.
    """
    if image is None or not cfg.enabled or not cfg.grid_overlay:
        return image
    if isinstance(image, _PILImage.Image):
        return _PILImage.fromarray(draw_coordinate_grid(np.asarray(image.convert("RGB"))))
    return draw_coordinate_grid(np.asarray(image))


def planner_context(
    cfg: RoboInterConfig,
    detections: Optional[Sequence[Any]] = None,
    scene_image=None,
    camera: Optional[str] = None,
) -> Optional[str]:
    """User-side context block, or None when the extension is off.

    None vs "" is the ON/OFF signal the planner branches on: an enabled run
    with nothing useful to say still gets the schema section, because the
    model can read the image even when no box can be stated for it.
    """
    if not cfg.enabled or not cfg.prompt:
        return None
    cams = {getattr(d, "camera", None) for d in (detections or []) if getattr(d, "camera", None)}
    if camera is None and len(cams) == 1:
        camera = next(iter(cams))
    if camera is None and len(cams) > 1:
        # Pixel coordinates from two cameras are not interchangeable; stating
        # both against one image would put the model's boxes in the wrong
        # frame. Say the size and nothing else.
        logger.warning(
            "[robointer] detections span cameras %s and no scene camera was "
            "given; emitting the image size without per-object boxes",
            sorted(c for c in cams if c),
        )
        detections = []
    return build_robointer_context(detections, image_size=image_size(scene_image), camera=camera)


def fewshot_examples(cfg: RoboInterConfig) -> List[Tuple[str, dict]]:
    """The worked (instruction, score) pair, or [] when off.

    The cached BT library predates the schema, so without this the model has
    no in-context example of a filled-in block.
    """
    return [robointer_fewshot()] if (cfg.enabled and cfg.fewshot) else []


# --------------------------------------------------------------------------
# Consume side
# --------------------------------------------------------------------------


def camera_models(pipeline) -> Dict[str, CameraModel]:
    """Base-frame camera models for the pipeline's calibrated cameras.

    An uncalibrated camera (identity extrinsic) is SKIPPED, not faked:
    ``CameraModel.from_calibration`` refuses to turn camera-frame coordinates
    into base-frame ones. Whether that emptiness is fatal is decided by the
    caller, which knows whether anything actually asked for a frame.
    """
    out: Dict[str, CameraModel] = {}
    for name, attr in _CAL_SOURCES:
        cal = getattr(pipeline, attr, None)
        if cal is None:
            continue
        try:
            out[name] = CameraModel.from_calibration(cal, name=name)
        except Exception as exc:  # noqa: BLE001 - reported by the caller
            logger.debug("[robointer] no camera model for %s: %s", name, exc)
    return out


def _nodes_by_path(score) -> Dict[str, dict]:
    """node_path -> node, keyed exactly as ``extract_annotations`` paths."""
    out: Dict[str, dict] = {}

    def walk(node, path):
        if not isinstance(node, dict):
            return
        out[path] = node
        for i, child in enumerate(node.get("children") or []):
            walk(child, f"{path}/{i}")

    walk((score or {}).get("tree"), "tree")
    return out


def _node_label(node: Optional[dict]) -> Optional[str]:
    params = (node or {}).get("params") or {}
    for key in _LABEL_PARAMS:
        value = params.get(key)
        if isinstance(value, str) and value.strip():
            return value.strip()
    return None


def _has_2d(ann: NodeAnnotation) -> bool:
    return any(
        getattr(ann, name) is not None
        for name in ("contact_point", "placement_proposal", "object_box", "affordance_box", "trace")
    )


def _get_annotation_provider(name: str):
    """An available out-of-band AnnotationProvider, or None.

    Deferred import (perception.annotations pulls the perception package) and
    a single seam for the tests to monkeypatch.  Fail-open by contract: a
    missing/unavailable provider is None, never a raise; the caller then
    keeps the planner's inline geometry.
    """
    from spark_real.perception.annotations import get_provider

    provider = get_provider(name)
    if provider is None or not provider.available():
        return None
    return provider


def _capture_rgb(pipeline, cam_name: Optional[str]):
    """The most recent RGB frame for a camera, from the pipeline's stash."""
    captures = getattr(pipeline, "_last_captures", None) or {}
    data = captures.get(cam_name) if cam_name else None
    return (data or {}).get("rgb")


def _provider_query(role: str, label: Optional[str], subtask: Optional[str]) -> str:
    """Noun phrase for the provider's 'Point to the {query}' template."""
    what = label or "target object"
    if role == "contact_point":
        q = f"best point for a parallel gripper to grasp the {what}"
    else:
        q = f"free spot inside the {what} to place the held object"
    if subtask:
        q += f" ({subtask})"
    return q


def _apply_source_override(
    cfg: RoboInterConfig,
    provider,
    ann: NodeAnnotation,
    label: Optional[str],
    cam_name: Optional[str],
    pipeline,
    notes: List[str],
) -> NodeAnnotation:
    """Replace the planner's contact/placement points with the provider's.

    The ONE consumption seam for both routes: whichever fields come back are
    fed to the SAME bounded-correction guard the inline path uses, so the
    source changes who supplies the pixel, never what may be done with it.

    Only fields the planner's inline block DECLARED are re-asked -- the block
    states which corrections this plan step wants; the provider supplies
    fresher pixels for them.  Fail-open PER FIELD: any provider trouble (no
    frame, no answer, exception, out-of-frame point) keeps the planner's
    inline value for that field and says so in ``notes``.
    """
    import copy

    from spark_real.perception import annotations as ann_mod

    image = _capture_rgb(pipeline, cam_name)
    if image is None:
        notes.append(
            f"source {cfg.source!r}: no captured frame for camera {cam_name!r}; "
            "using the planner's inline geometry"
        )
        return ann

    out = copy.copy(ann)
    out.camera = cam_name  # the provider read its pixels off THIS frame
    for role, field_name in (
        ("contact_point", "contact_point"),
        ("placement_proposal", "placement_proposal"),
    ):
        if getattr(ann, field_name) is None:
            continue
        query = _provider_query(role, label, ann.subtask)
        try:
            answers = provider.annotate(np.asarray(image), query, kind="point")
        except Exception as exc:  # noqa: BLE001 - fail-open to the inline value
            notes.append(f"{role}: {provider.name} annotate failed ({exc}); using planner")
            continue
        answers = [a for a in answers if a.kind == "point" and a.in_bounds()]
        if not answers:
            notes.append(f"{role}: {provider.name} returned no usable point; using planner")
            continue
        point = answers[0].clamped()
        point.label = ann_mod.role_label(role, label or "")
        fragment = ann_mod.robointer_from_annotations([point], camera=cam_name)
        setattr(out, field_name, getattr(fragment, field_name))
        notes.append(f"{role}: from {provider.name} (out-of-band), planner value replaced")
    return out


def apply_to_detection_map(
    cfg: RoboInterConfig,
    score,
    detections: Sequence[Any],
    detection_map,
    pipeline,
) -> List[Dict[str, Any]]:
    """Publish the planner's spatial annotations onto ``detection_map``.

    Returns JSON-able records (one per annotated node) for the run trace, and
    an EMPTY list plus an untouched map whenever the extension is off.

    Only two things are written, both bounded corrections to a measured
    detection and both XY-only -- Z always stays the perceived value:
    ``robointer_contact_xyz`` on the object a node grips and
    ``robointer_placement_xyz`` on the container a node releases over.
    """
    if not cfg.enabled or not cfg.consume:
        return []
    if cfg.source == "off":
        # The operator asked for the prompt half (annotations recorded into
        # the episode) with NO correction from anybody -- planner or provider.
        return []
    provider = None
    if cfg.source != "planner":
        provider = _get_annotation_provider(cfg.source)
        if provider is None:
            # Fail-open to the planner's inline blocks.
            logger.warning(
                "[robointer] source %r is unavailable; falling back to the "
                "planner's inline annotations",
                cfg.source,
            )
    labels = [getattr(d, "label", None) for d in detections or []]
    pairs = extract_annotations(score, labels)
    if not pairs:
        return []

    by_label = {getattr(d, "label", None): d for d in detections or []}
    nodes = _nodes_by_path(score)
    cameras = camera_models(pipeline)
    if not cameras and any(_has_2d(ann) for _p, ann in pairs):
        # ASKED FOR and unbuildable: the planner answered with geometry and
        # there is no calibrated frame to read it in. Silence here would look
        # exactly like "the planner proposed nothing".
        raise RuntimeError(
            "robointer is enabled and the plan carries spatial annotations, "
            "but no calibrated camera model could be built (every camera has "
            "an identity extrinsic); refusing to guess a frame"
        )

    records: List[Dict[str, Any]] = []
    for path, ann in pairs:
        label = ann.label or _node_label(nodes.get(path))
        record: Dict[str, Any] = {
            "node_path": path,
            "label": label,
            "subtask": ann.subtask,
            "primitive_skill": ann.primitive_skill,
            "notes": [],
        }
        records.append(record)

        entry = detection_map.get(label) if label else None
        if entry is None or entry.get("position_3d") is None:
            record["notes"].append(f"no detection named {label!r}; annotation not applied")
            continue
        det = by_label.get(label)
        if det is None:
            # Label drift: LabelResolvingDetectionMap re-bound this label onto
            # a differently-named detection (a cached BT's "knife handle 2"
            # onto this run's "knife handle"). The measured position is right
            # there in the entry it resolved to, so bound the correction
            # against THAT rather than abstaining on a name mismatch.
            det = SimpleNamespace(
                position_3d=np.asarray(entry["position_3d"], dtype=float),
                camera=entry.get("_camera"),
            )
        cam_name = ann.camera or getattr(det, "camera", None)
        cam = cameras.get(cam_name)
        if cam is None and _has_2d(ann):
            record["notes"].append(f"no camera model for {cam_name!r}; not resolved")
            logger.warning("[robointer] %s: %s", path, record["notes"][-1])
            continue

        if provider is not None:
            # Out-of-band source: re-ask the pointing provider for the fields
            # this node's inline block declared, then run the SAME guard.
            record["source"] = provider.name
            ann = _apply_source_override(
                cfg, provider, ann, label, cam_name, pipeline, record["notes"]
            )
        else:
            record["source"] = "planner"

        if ann.subtask:
            entry[SUBTASK_KEY] = ann.subtask
        if ann.contact_point is not None:
            target, reason = contact_target_xyz(ann, det, cam, max_shift_m=cfg.max_shift_m)
            record["notes"].append(reason)
            if target is not None:
                entry[CONTACT_KEY] = [float(v) for v in target]
            logger.info("[robointer] %s %s -> %s", path, label, reason)
        if ann.placement_proposal is not None:
            target, reason = placement_target_xyz(ann, det, cam, max_shift_m=cfg.max_place_shift_m)
            record["notes"].append(reason)
            if target is not None:
                entry[PLACEMENT_KEY] = [float(v) for v in target]
            logger.info("[robointer] %s %s -> %s", path, label, reason)
    return records


__all__ = [
    "CONTACT_KEY",
    "PLACEMENT_KEY",
    "ROBOINTER_ENV",
    "ROBOINTER_SOURCE_ENV",
    "SOURCES",
    "RoboInterConfig",
    "SUBTASK_KEY",
    "annotate_planner_image",
    "apply_to_detection_map",
    "camera_models",
    "family_raw",
    "fewshot_examples",
    "image_size",
    "planner_context",
    "resolve_settings",
]

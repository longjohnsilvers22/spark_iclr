"""
Runtime hook that bridges the calibration DB into primitive execution.

``preprocess`` adds learned (object, surface) offsets to spatial slot kwargs
before a primitive runs; ``observe`` records outcomes back into the DB.
Heuristics in :func:`infer_surface_class` and :func:`normalize_object` are
intentionally minimal -- a VLM-based replacement is future work.
"""
from __future__ import annotations

import pathlib
from typing import Any, Dict, Mapping, Optional

from .calibration import CalibrationDB

# Primitives that take spatial offsets (where calibration matters).
# ``place`` is included tentatively in case a future primitive lands; if it
# doesn't exist in the registry it simply never gets called.
SPATIAL_PRIMITIVES = frozenset({"move_to_keypoint", "place", "insert", "wipe"})

# Mapping from canonical (x, y, z) axes to per-primitive slot names.
# Default uses ``offset_{axis}`` (move_to_keypoint, place, insert, wipe).
_OFFSET_SLOT_MAP: Dict[str, Dict[str, str]] = {
    "move_to_keypoint": {"x": "offset_x", "y": "offset_y", "z": "offset_z"},
    "place":            {"x": "offset_x", "y": "offset_y", "z": "offset_z"},
    "insert":           {"x": "offset_x", "y": "offset_y", "z": "offset_z"},
    "wipe":             {"x": "offset_x", "y": "offset_y", "z": "offset_z"},
}

# Minimal color/brand modifier list for normalization. Real vocabularies should
# come from the perception ontology -- keep this short and document the gap.
_MODIFIERS = {
    "red", "blue", "green", "yellow", "orange", "purple", "pink", "white",
    "black", "gray", "grey", "brown", "tan", "beige", "gold", "silver",
    "akita", "alphabet", "wooden", "plastic", "ceramic", "metal", "glass",
    "small", "large", "big", "tiny", "tall", "short",
}


def normalize_object(label: Optional[str]) -> str:
    """
    Strip color/brand modifiers from a perception label.

    Best-effort only: ``"akita black bowl"`` -> ``"bowl"``,
    ``"red mug"`` -> ``"mug"``. Multi-word object classes (e.g. "salad
    dressing") will be flattened to the last token; document this limitation
    so callers know to override via ``object_hint`` when wrong.
    """
    if not label:
        return ""
    tokens = [t.lower() for t in str(label).strip().split() if t.strip()]
    filtered = [t for t in tokens if t not in _MODIFIERS]
    if not filtered:
        return tokens[-1] if tokens else ""
    # Heuristic: take the last remaining token (typically the noun head).
    return filtered[-1]


def infer_surface_class(scene_context: Optional[Mapping[str, Any]]) -> str:
    """
    Infer a surface class from a dict-like scene context.

    Recognised hints (best-effort):
      * explicit ``"surface"`` key wins;
      * if any detected label contains ``"bowl"`` -> ``bowl_interior``;
      * if any label contains ``"shelf"`` -> ``shelf``;
      * if any label contains ``"drawer"`` -> ``drawer_interior``;
      * if any label contains ``"floor"`` -> ``floor``;
      * else ``flat_table`` (the LIBERO/robosuite default).

    Improving this with a VLM or scene-graph classifier is future work.
    """
    if not scene_context:
        return "flat_table"
    explicit = scene_context.get("surface") if isinstance(scene_context, Mapping) else None
    if explicit:
        return str(explicit)
    labels = []
    if isinstance(scene_context, Mapping):
        objs = scene_context.get("objects") or scene_context.get("detections") or []
        for o in objs:
            if isinstance(o, str):
                labels.append(o.lower())
            elif isinstance(o, Mapping):
                lab = o.get("label") or o.get("class") or o.get("name")
                if lab:
                    labels.append(str(lab).lower())
    blob = " ".join(labels)
    if "bowl" in blob:
        return "bowl_interior"
    if "shelf" in blob:
        return "shelf"
    if "drawer" in blob:
        return "drawer_interior"
    if "floor" in blob:
        return "floor"
    return "flat_table"


def _target_label(slots: Mapping[str, Any]) -> str:
    """
    Pick the slot that names the target object for calibration lookup.
    """
    for key in ("keypoint_label", "target_label", "object", "label"):
        v = slots.get(key)
        if v:
            return str(v)
    return ""


class CalibrationWrapper:
    """
    Thin wrapper that injects + observes calibration around primitives.
    """

    def __init__(self, db_path: str | pathlib.Path,
                 db: Optional[CalibrationDB] = None) -> None:
        self.db_path = pathlib.Path(db_path)
        self.db = db if db is not None else CalibrationDB.load(self.db_path)
        # Ensure the path is wired so save() works without arguments.
        if self.db.path is None:
            self.db.path = self.db_path

    # pre

    def preprocess(self, *, primitive_name: str,
                   slots: Mapping[str, Any],
                   surface_hint: Optional[str] = None,
                   object_hint: Optional[str] = None,
                   scene_context: Optional[Mapping[str, Any]] = None,
                   ) -> Dict[str, Any]:
        """
        Return a copy of ``slots`` with learned offsets folded in.

        Non-spatial primitives are returned unchanged (still as a fresh dict).
        """
        out: Dict[str, Any] = dict(slots)
        if primitive_name not in SPATIAL_PRIMITIVES:
            return out
        axis_slots = _OFFSET_SLOT_MAP.get(primitive_name)
        if not axis_slots:
            return out

        obj = normalize_object(object_hint or _target_label(slots))
        if not obj:
            return out
        surface = surface_hint or infer_surface_class(scene_context)
        learned = self.db.get(obj, surface)
        if not learned:
            return out

        for axis, slot_name in axis_slots.items():
            delta = learned.get(axis)
            if delta is None:
                continue
            cur = float(out.get(slot_name, 0.0) or 0.0)
            out[slot_name] = cur + float(delta)
        return out

    # post

    def observe(self, *, primitive_name: str,
                slots: Mapping[str, Any],
                success: bool,
                observed_correction: Mapping[str, float],
                surface_hint: Optional[str] = None,
                object_hint: Optional[str] = None,
                scene_context: Optional[Mapping[str, Any]] = None,
                ) -> None:
        """
        Record the (object, surface) outcome into the DB.

        ``observed_correction`` should be a dict of axis -> meters describing
        the offset that *would have worked*. If the primitive isn't spatial
        or it can't be attributed to an object, this is a no-op.
        """
        if primitive_name not in SPATIAL_PRIMITIVES:
            return
        if not observed_correction:
            return
        # Skip vacuous updates: if all axes are exactly zero AND the trial
        # succeeded, the runtime didn't actually extract a real post-EE
        # error; recording (0,0,0) would pull the running mean toward zero
        # and erode genuine signal from real observations.
        nonzero = any(abs(float(v)) > 1e-6 for v in observed_correction.values())
        if not nonzero and bool(success):
            return
        obj = normalize_object(object_hint or _target_label(slots))
        if not obj:
            return
        surface = surface_hint or infer_surface_class(scene_context)
        self.db.update(
            object=obj, surface=surface,
            observed_offset={k: float(v) for k, v in observed_correction.items()},
            success=bool(success),
        )

    # io

    def save(self, path: Optional[str | pathlib.Path] = None) -> None:
        self.db.save(path or self.db_path)


__all__ = [
    "CalibrationWrapper",
    "SPATIAL_PRIMITIVES",
    "infer_surface_class",
    "normalize_object",
]

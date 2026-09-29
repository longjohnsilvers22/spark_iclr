"""
What the RoboInter annotations are actually USED for.

  RECORDED. An annotated score reaches ``output/episodes_cache/<id>/bt.yaml``
  because ``EpisodeRecorder.end()`` writes the score verbatim through
  ``yaml.safe_dump`` and ``NodeAnnotation.to_dict`` emits builtins only. Every
  field (subtask, contact point, trace, placement) is therefore a per-episode
  training signal aligned with the trajectory in the same folder.

  CONSUMED only through :func:`contact_target_xyz` and
  :func:`placement_target_xyz`: pure functions that turn an annotation into
  the same ``np.ndarray(3)`` base-frame target the executor already moves
  to, with the guards that make that safe, behind a default-off config flag
  (``planning.robointer``).

The guard is the load-bearing part. A pixel coordinate from an LLM can be
wrong in a way a mask centroid cannot: not noisy, but confidently pointing at
a different object. So an override is only ever accepted as a BOUNDED
CORRECTION to a perception result -- never as a target in its own right. If
the proposal disagrees with the detection by more than
``max_shift_m`` it is refused and the detection wins.
"""

from __future__ import annotations

import logging
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np

from spark_real.planning.robointer import (
    NodeAnnotation,
    ResolvedAnnotation,
    extract_annotations,
    resolve_annotation,
)
from spark_real.planning.robointer_geometry import CameraModel

logger = logging.getLogger(__name__)

# How far a planner-proposed point may move the target away from what
# perception measured. 8 cm is about the length of a Robotiq 2F-85 fingertip
# pad: enough to slide from a knife's centroid onto its handle, not enough to
# reach the next object.
DEFAULT_MAX_SHIFT_M = 0.08

# Same idea for a placement, but a container is large and the whole point of
# a placement_proposal is to pick a spot inside it that is not its centre.
DEFAULT_MAX_PLACE_SHIFT_M = 0.20


def _guarded(
    proposed: Optional[np.ndarray],
    measured: Optional[Sequence[float]],
    max_shift_m: float,
    what: str,
) -> Tuple[Optional[np.ndarray], str]:
    if proposed is None:
        return None, f"{what}: not resolvable to a base-frame point"
    if measured is None:
        return None, f"{what}: no measured position to bound the correction against"
    m = np.asarray(measured, dtype=float).reshape(3)
    shift = float(np.linalg.norm(proposed[:2] - m[:2]))
    if shift > max_shift_m:
        return None, (
            f"{what}: proposal is {shift * 100:.1f} cm from the detection "
            f"(cap {max_shift_m * 100:.0f} cm); keeping the detection"
        )
    return proposed, f"{what}: accepted, {shift * 100:.1f} cm correction"


def contact_target_xyz(
    ann: NodeAnnotation,
    detection: Any,
    camera: CameraModel,
    depth: Optional[np.ndarray] = None,
    max_shift_m: float = DEFAULT_MAX_SHIFT_M,
) -> Tuple[Optional[np.ndarray], str]:
    """Base-frame XY the gripper should close on, or ``(None, reason)``.

    Z always comes from perception, never from the annotation: the plane
    assumption behind a 2D point is exactly as good as the plane, and the
    detection's Z was measured. Only XY is corrected -- which is the whole of
    the handle-vs-centroid problem.

    ``detection`` is duck-typed; it needs ``position_3d``.
    """
    measured = getattr(detection, "position_3d", None)
    if ann.contact_point is None:
        return None, "contact_point: not emitted"
    if measured is None:
        return None, "contact_point: detection has no position_3d"
    m = np.asarray(measured, dtype=float).reshape(3)
    res = resolve_annotation(ann, camera, z_plane_m=float(m[2]), depth=depth)
    target, reason = _guarded(res.contact_xyz, m, max_shift_m, "contact_point")
    if target is None:
        return None, reason
    out = np.array([target[0], target[1], m[2]], dtype=float)
    return out, reason


def placement_target_xyz(
    ann: NodeAnnotation,
    container_detection: Any,
    camera: CameraModel,
    max_shift_m: float = DEFAULT_MAX_PLACE_SHIFT_M,
) -> Tuple[Optional[np.ndarray], str]:
    """Base-frame XY to release over, or ``(None, reason)``.

    Same contract as :func:`contact_target_xyz`: XY only, bounded against the
    container's measured position, Z untouched.
    """
    measured = getattr(container_detection, "position_3d", None)
    if ann.placement_proposal is None:
        return None, "placement_proposal: not emitted"
    if measured is None:
        return None, "placement_proposal: container has no position_3d"
    m = np.asarray(measured, dtype=float).reshape(3)
    res = resolve_annotation(ann, camera, z_plane_m=float(m[2]))
    target, reason = _guarded(res.placement_xyz, m, max_shift_m, "placement_proposal")
    if target is None:
        return None, reason
    return np.array([target[0], target[1], m[2]], dtype=float), reason


def annotations_for_record(score, keypoint_labels=None) -> List[Dict[str, Any]]:
    """JSON-able annotation records for a trace or episode metadata.

    Shaped for ``control.primitive_trace.PrimitiveTrace.extra`` and for
    ``EpisodeRecorder.end(result=...)``. 2D only -- no camera model needed, so
    it works on any score, including one replayed from the cache or the
    RGB-only human corpus.
    """
    out: List[Dict[str, Any]] = []
    for path, ann in extract_annotations(score, keypoint_labels):
        record = {"node_path": path}
        record.update(ann.to_dict())
        out.append(record)
    return out


def resolved_for_record(resolved: Sequence[ResolvedAnnotation]) -> List[Dict[str, Any]]:
    """Base-frame resolutions, JSON-able, for the same two sinks."""
    return [r.to_dict() for r in resolved or []]

"""Spatially-anchored end-of-task verification (the "mask memory" rung).

Checks whether the placed object's mask sits in the remembered target region
(the approved pre-execution detection). Identity is carried by the commanded
location, never by re-matching labels. The rung only upgrades an ABSTAIN with
real evidence or contributes a FAIL with evidence; it never passes on silence.
"""

import logging

import numpy as np

logger = logging.getLogger("spark_server")

# The placed object's mask centroid must land within this fraction of the
# remembered target bbox's diagonal from the bbox center. Loose on purpose:
# a screwdriver seated in its slot can hang a few cm past either end.
CENTER_FRAC = 0.75
# Minimum confidence for the end-capture object detection to count as
# evidence in either direction. Below this the rung abstains.
MIN_CONF = 0.35


def remember_targets(executor):
    """Snapshot the pre-execution detection map's masks and boxes.

    Taken ONCE when the executor receives its detection map; mid-run
    re-detections overwrite detection_map in place. The memory must be the
    scene as APPROVED, not as mutated.
    """
    mem = {}
    for label, det in (executor.detection_map or {}).items():
        if not isinstance(det, dict):
            continue
        bbox = det.get("bbox")
        cam = det.get("_camera")
        if bbox is None or cam is None:
            continue
        mem[str(label)] = {
            "bbox": [float(v) for v in bbox],
            "camera": str(cam),
            "confidence": float(det.get("confidence") or 0.0),
        }
    executor._spatial_memory = mem
    if mem:
        logger.info("[spatial-verify] remembered %d region(s): %s",
                    len(mem), ", ".join(sorted(mem)))


def spatial_rung(executor, score=None):
    """The end-of-task check. Returns a dict verdict or None (abstain).

    {"status": "pass"|"fail"|"abstain", "detail": str, ...evidence}
    """
    mem = getattr(executor, "_spatial_memory", None) or {}
    target_label = getattr(executor, "_last_place_label", "") or ""
    obj_label = getattr(executor, "_active_grasp_label", "") or ""
    if not mem or not target_label or not obj_label:
        return None
    remembered = mem.get(target_label)
    if remembered is None:
        return None

    pipeline = getattr(executor, "_pipeline", None)
    if pipeline is None:
        return None

    # One wide capture with the arm retracted. Detect ONLY the object's
    # label: the target is not re-detected, because "the slot is gone" is a
    # success signal, not a matching failure.
    try:
        captures = pipeline.capture()
        dets = pipeline.detect(captures, [obj_label])
        merged = pipeline.merge_detections(dets)
    except Exception as exc:  # noqa: BLE001 - verification must not crash a run
        logger.warning("[spatial-verify] capture/detect failed: %s", exc)
        return None

    cam = remembered["camera"]
    x1, y1, x2, y2 = remembered["bbox"]
    cx_t, cy_t = (x1 + x2) / 2.0, (y1 + y2) / 2.0
    diag = float(np.hypot(x2 - x1, y2 - y1)) or 1.0

    best = None
    for d in merged:
        if str(getattr(d, "label", "")) != obj_label:
            continue
        if str(getattr(d, "camera", "")) != cam:
            continue
        if float(getattr(d, "confidence", 0) or 0) < MIN_CONF:
            continue
        c2d = getattr(d, "centroid_2d", None)
        if c2d is None:
            continue
        dist = float(np.hypot(c2d[0] - cx_t, c2d[1] - cy_t)) / diag
        if best is None or dist < best["dist_frac"]:
            best = {
                "dist_frac": round(dist, 3),
                "confidence": round(float(d.confidence), 2),
            }

    if best is None:
        # Not visible on the remembered camera: seated deep or dropped out
        # of frame. Abstain rather than guess.
        return {"status": "abstain",
                "detail": f"'{obj_label}' not re-detected on {cam} "
                          f"(conf floor {MIN_CONF})"}

    ok = best["dist_frac"] <= CENTER_FRAC
    verdict = {
        "status": "pass" if ok else "fail",
        "detail": (
            f"'{obj_label}' centroid {best['dist_frac']:.2f} of the "
            f"remembered '{target_label}' bbox diagonal from its center on "
            f"{cam} (gate {CENTER_FRAC}, conf {best['confidence']})"
        ),
        **best,
        "camera": cam,
    }
    logger.info("[spatial-verify] %s: %s", verdict["status"], verdict["detail"])
    return verdict

"""Operator-drawn approach paths.

SPARK has no lateral motion planning: every approach is a straight line, so
the open jaws can sweep through an object standing beside the target. This is
the human-in-the-loop answer (draw the path you want and the transport flies
it) and the correction channel for whatever a planner proposes later. The
2D->3D half lives in perception/trace_path.py; the annotation type it speaks
is in perception/annotations.py.
"""

import logging
from typing import List, Optional

from fastapi import APIRouter
from fastapi.responses import JSONResponse
from pydantic import BaseModel

from spark_real.perception.annotations import Annotation
from spark_real.perception.trace_path import trace_to_waypoints
from spark_real.routes import state

logger = logging.getLogger("spark_server")
router = APIRouter()


class TraceRequest(BaseModel):
    # Normalized (x, y) in [0, 1], in path order, as drawn on the live view.
    points: List[List[float]]
    camera: str = "birdview"
    # The move this path belongs to. The transport consumes the trace only
    # when it is heading to THIS label, so a stale path cannot hijack an
    # unrelated motion.
    label: str
    # Height to fly at. Omitted -> the bound label's detected z plus a
    # clearance, i.e. "over the scene at the object's height".
    z: Optional[float] = None


@router.get("/api/annotate_trace")
async def get_trace():
    t = state.pending_trace
    return {"trace": t, "armed": t is not None}


@router.delete("/api/annotate_trace")
async def clear_trace():
    state.pending_trace = None
    logger.info("[trace] cleared by the operator")
    return {"ok": True, "armed": False}


@router.post("/api/annotate_trace")
async def set_trace(req: TraceRequest):
    pipeline = state.pipeline
    if pipeline is None:
        return JSONResponse(status_code=400, content={"error": "pipeline not ready"})
    if len(req.points or []) < 2:
        return JSONResponse(
            status_code=400, content={"error": "a trace needs at least 2 points"}
        )
    cal = None
    getter = getattr(pipeline, "_annotation_cal_for", None)
    if callable(getter):
        cal = getter(req.camera)
    if cal is None:
        return JSONResponse(
            status_code=400,
            content={"error": f"camera '{req.camera}' has no calibration; "
                              "a drawn path cannot be back-projected from it"},
        )

    z = req.z
    if z is None:
        dmap = getattr(getattr(pipeline, "_executor", None), "detection_map", None) or {}
        pos = (dmap.get(req.label) or {}).get("position_3d")
        if pos is None:
            pos = next(
                (
                    d.position_3d
                    for d in (state.pending_detections or [])
                    if str(getattr(d, "label", "")) == req.label
                    and getattr(d, "position_3d", None) is not None
                ),
                None,
            )
        if pos is None:
            return JSONResponse(
                status_code=400,
                content={"error": f"no detection for '{req.label}' to take a "
                                  "height from; pass z explicitly"},
            )
        # Fly ABOVE the object's own height: the drawn path is transit, and
        # the descent that follows owns the vertical move.
        z = float(pos[2]) + 0.12

    try:
        ann = Annotation(
            kind="trace",
            points=[(float(p[0]), float(p[1])) for p in req.points],
            provider="operator",
            label=req.label,
        ).clamped()
        waypoints = trace_to_waypoints(ann, cal, z)
    except Exception as exc:  # noqa: BLE001
        logger.warning("[trace] rejected: %s", exc)
        return JSONResponse(status_code=400, content={"error": str(exc)})

    if len(waypoints) < 2:
        return JSONResponse(
            status_code=400,
            content={"error": "the drawn path did not project onto the plane "
                              "(is the camera calibrated and the path on the table?)"},
        )
    state.pending_trace = {
        "label": req.label,
        "waypoints": waypoints,
        "camera": req.camera,
        "z": float(z),
    }
    logger.info(
        "[trace] armed for '%s': %d waypoint(s) at z=%.3f from %s",
        req.label, len(waypoints), z, req.camera,
    )
    return {"ok": True, "armed": True, **state.pending_trace}

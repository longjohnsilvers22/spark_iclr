"""
Bimanual-only API endpoints.

Mounted under ``/api/bimanual`` so existing single-arm routes are
untouched. Frontend only invokes these when ``robot_family ==
"bimanual_franka"``; they 400-out cleanly on other families.

Endpoints (grouped):

* ``GET /api/bimanual/state``                      : both arms' obs + gripper
* ``POST /api/bimanual/{arm}/velocity``            : per-arm teleop twist
* ``POST /api/bimanual/{arm}/gripper``             : per-arm open/close
* ``POST /api/bimanual/{arm}/gripper_position``    : per-arm 0..1 position
* ``POST /api/bimanual/{arm}/home``                : home one arm
* ``POST /api/bimanual/home_both``                 : coordinated dual home
* ``POST /api/bimanual/handoff``                   : direct handoff trigger
* ``GET  /api/bimanual/inter_arm_distance``        : current TCP-to-TCP gap
* ``GET  /api/bimanual/primitives``                : primitive registry + use counts
"""

from __future__ import annotations

import json
import logging
from typing import Optional

import numpy as np
from fastapi import APIRouter
from fastapi.responses import JSONResponse
from pydantic import BaseModel

from spark_real.routes import state
from spark_real.skills import registry
from spark_real.skills.tool_use import tool_library_path

logger = logging.getLogger("spark_server")
router = APIRouter(prefix="/api/bimanual")


def _resolve_driver():
    pipeline = state.pipeline
    if pipeline is None or pipeline._robot is None:
        return None, JSONResponse(
            status_code=400, content={"error": "robot not connected"}
        )
    if not _is_bimanual(pipeline):
        return None, JSONResponse(
            status_code=400,
            content={"error": "active robot is not the bimanual family"},
        )
    return pipeline._robot, None


def _is_bimanual(pipeline) -> bool:
    fam = (getattr(pipeline.config, "robot_family", "") or "").lower()
    return fam == "bimanual_franka"


# request bodies
class ArmVelocityRequest(BaseModel):
    vx: float = 0.0
    vy: float = 0.0
    vz: float = 0.0
    wrx: float = 0.0
    wry: float = 0.0
    wrz: float = 0.0
    duration: float = 0.1


class ArmGripperRequest(BaseModel):
    action: str  # "open" | "close"
    force: Optional[float] = None
    speed: Optional[float] = None


class ArmGripperPositionRequest(BaseModel):
    position: float  # 0.0=open, 1.0=closed
    speed: Optional[float] = None
    force: Optional[float] = None


class HandoffRequest(BaseModel):
    from_arm: str  # "left" | "right"
    to_arm: str  # "right" | "left"
    keypoint_label: Optional[str] = None
    meeting_point: Optional[list] = None
    grasp_width: float = 0.030
    force: float = 15.0


# routes
@router.get("/state")
async def get_bimanual_state():
    drv, err = _resolve_driver()
    if err is not None:
        return err
    try:
        obs = drv.get_observation()
        return _jsonable(obs.to_dict() if hasattr(obs, "to_dict") else obs)
    except Exception as e:
        logger.exception("bimanual.get_state failed")
        return JSONResponse(status_code=500, content={"error": str(e)})


@router.post("/{arm}/velocity")
async def arm_velocity(arm: str, body: ArmVelocityRequest):
    drv, err = _resolve_driver()
    if err is not None:
        return err
    if arm not in ("left", "right"):
        return JSONResponse(
            status_code=400, content={"error": "arm must be left|right"}
        )
    if state.executor_running:
        return {"status": "ignored", "reason": "executor running"}
    try:
        drv.send_velocity(
            [body.vx, body.vy, body.vz],
            [body.wrx, body.wry, body.wrz],
            arm=arm,
            duration=body.duration,
        )
        return {"status": "ok", "arm": arm}
    except Exception as e:
        logger.exception("bimanual.velocity failed")
        return JSONResponse(status_code=500, content={"error": str(e)})


@router.post("/{arm}/gripper")
async def arm_gripper(arm: str, body: ArmGripperRequest):
    drv, err = _resolve_driver()
    if err is not None:
        return err
    if arm not in ("left", "right"):
        return JSONResponse(
            status_code=400, content={"error": "arm must be left|right"}
        )
    try:
        if body.action == "open":
            drv.open_gripper(arm=arm, speed=body.speed, force=body.force)
        elif body.action == "close":
            drv.close_gripper(arm=arm, speed=body.speed, force=body.force)
        else:
            return JSONResponse(
                status_code=400, content={"error": "action must be open|close"}
            )
        return {"status": "ok"}
    except Exception as e:
        logger.exception("bimanual.gripper failed")
        return JSONResponse(status_code=500, content={"error": str(e)})


@router.post("/{arm}/gripper_position")
async def arm_gripper_position(arm: str, body: ArmGripperPositionRequest):
    drv, err = _resolve_driver()
    if err is not None:
        return err
    if arm not in ("left", "right"):
        return JSONResponse(
            status_code=400, content={"error": "arm must be left|right"}
        )
    try:
        drv.set_gripper_position(
            body.position, arm=arm, speed=body.speed, force=body.force
        )
        return {"status": "ok"}
    except Exception as e:
        logger.exception("bimanual.gripper_position failed")
        return JSONResponse(status_code=500, content={"error": str(e)})


@router.post("/{arm}/home")
async def arm_home(arm: str):
    drv, err = _resolve_driver()
    if err is not None:
        return err
    if arm not in ("left", "right"):
        return JSONResponse(
            status_code=400, content={"error": "arm must be left|right"}
        )
    try:
        drv.go_home(arm=arm)
        return {"status": "ok"}
    except Exception as e:
        logger.exception("bimanual.home failed")
        return JSONResponse(status_code=500, content={"error": str(e)})


@router.post("/home_both")
async def home_both():
    drv, err = _resolve_driver()
    if err is not None:
        return err
    try:
        drv.go_home()  # sequenced left then right
        return {"status": "ok"}
    except Exception as e:
        logger.exception("bimanual.home_both failed")
        return JSONResponse(status_code=500, content={"error": str(e)})


@router.post("/handoff")
async def handoff(body: HandoffRequest):
    """
    Execute a handoff directly from the UI (bypassing the planner).

    Useful for the "handoff L-to-R" button and for diagnostics. Internally
    builds a single-node BT score with `handoff` as the only primitive
    and feeds it through the bimanual executor.
    """
    pipeline = state.pipeline
    if pipeline is None or pipeline._executor is None:
        return JSONResponse(
            status_code=400, content={"error": "pipeline not initialised"}
        )
    if not _is_bimanual(pipeline):
        return JSONResponse(
            status_code=400, content={"error": "robot family is not bimanual"}
        )
    if {body.from_arm, body.to_arm} != {"left", "right"}:
        return JSONResponse(
            status_code=400, content={"error": "from_arm/to_arm must be {left,right}"}
        )

    score = {
        "task": f"handoff {body.from_arm}->{body.to_arm}",
        "tree": {
            "type": "sequence",
            "children": [
                {
                    "type": "handoff",
                    "params": {
                        "from_arm": body.from_arm,
                        "to_arm": body.to_arm,
                        "keypoint_label": body.keypoint_label,
                        "meeting_point": body.meeting_point or [0.0, 0.0, 0.30],
                        "grasp_width": body.grasp_width,
                        "force": body.force,
                    },
                }
            ],
        },
    }
    state.executor_running = True
    try:
        results = pipeline._executor.execute_score(score)
        ok = all(r.success for r in results)
        return {
            "status": "ok" if ok else "failed",
            "results": [
                {"action": r.action_type, "success": r.success, "message": r.message}
                for r in results
            ],
        }
    finally:
        state.executor_running = False


@router.get("/inter_arm_distance")
async def inter_arm_distance():
    pipeline = state.pipeline
    if pipeline is None or pipeline._robot is None:
        return JSONResponse(status_code=400, content={"error": "not connected"})
    safe = getattr(pipeline, "_safe", None)
    if safe is None or not hasattr(safe, "get_inter_arm_distance"):
        return JSONResponse(
            status_code=400, content={"error": "bimanual safe robot not available"}
        )
    return {"distance_m": safe.get_inter_arm_distance()}


# primitive registry + use counts
@router.get("/primitives")
async def list_primitives():
    """
    Return the primitive registry + per-primitive use counts.

    Combines the SkillRegistry's static set with the
    ``output/tool_library.json`` use-count file written by tool-use
    skills.
    """
    lib_path = tool_library_path()
    lib_data = {}
    if lib_path.exists():
        try:
            lib_data = json.loads(lib_path.read_text())
        except Exception:
            logger.exception("could not read tool library")

    entries = []
    for name in registry.names():
        entry = registry.get(name)
        usage = lib_data.get(name, {})
        learned = bool(usage.get("uses", 0))
        entries.append(
            {
                "name": name,
                "description": entry.description if entry else "",
                "params": (
                    {p: t.__name__ for p, t in (entry.params or {}).items()}
                    if entry
                    else {}
                ),
                "uses": int(usage.get("uses", 0)),
                "successes": int(usage.get("successes", 0)),
                "first_used_ts": usage.get("first_used_ts"),
                "last_used_ts": usage.get("last_used_ts"),
                "learned": learned,
            }
        )
    return {
        "primitives": entries,
        "count": len(entries),
        "learned_count": sum(1 for e in entries if e["learned"]),
    }



# helpers
def _jsonable(obj):
    if isinstance(obj, dict):
        return {k: _jsonable(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [_jsonable(v) for v in obj]
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    if isinstance(obj, (np.floating,)):
        return float(obj)
    if isinstance(obj, (np.integer,)):
        return int(obj)
    return obj

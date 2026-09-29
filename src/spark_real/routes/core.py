"""
Core API routes: status, capture, shutdown, abort, progress.

Detection routes are in detection.py; execution routes in execution.py.
This module aggregates all sub-routers under a single ``router`` object
so server.py can import just ``from spark_real.routes.core import router``.
"""

import logging
import os
import time
import threading

from fastapi import APIRouter
from fastapi.responses import JSONResponse

from spark_real.pipeline import SPARKRealPipeline, PipelineConfig
from spark_real.routes import state
from spark_real.routes.streaming import get_available_cameras
from spark_real.routes.visualization import (
    encode_image,
    depth_to_colormap,
)

# Sub-routers
from spark_real.routes.detection import router as _detection_router
from spark_real.routes.execution import router as _execution_router

logger = logging.getLogger("spark_server")
router = APIRouter()

# Merge sub-routers so server.py sees a single router
router.include_router(_detection_router)
router.include_router(_execution_router)


@router.post("/api/abort")
async def abort_task():
    """
    Cancel any in-flight task without killing the server.
    """
    pipeline = state.pipeline
    if pipeline is None:
        return {"aborted": False, "reason": "pipeline not running"}
    executor = getattr(pipeline, "_executor", None) or getattr(
        pipeline, "executor", None
    )
    actions = []
    if executor is not None:
        # Through note_abort_requested, never `executor._abort = True`: a bare
        # assignment does not advance the abort epoch, so execute_score's
        # arming check cannot see it and wipes it when the run starts. That is
        # the whole window /api/abort exists to cover -- the executor thread is
        # still in capture/detect/plan and has not reached execute_score yet.
        note = getattr(executor, "note_abort_requested", None)
        if callable(note):
            note()
            actions.append("executor.note_abort_requested")
        elif hasattr(executor, "_abort"):
            executor._abort = True
            actions.append("set executor._abort")
    robot = getattr(pipeline, "_robot", None)
    if robot is not None:
        for method in ("stop_velocity", "stop_motion", "recover_from_errors"):
            fn = getattr(robot, method, None)
            if callable(fn):
                try:
                    fn()
                    actions.append(method)
                except Exception as exc:
                    actions.append(f"{method}:err={exc}")
    return {"aborted": bool(actions), "actions": actions}


@router.post("/api/shutdown")
def shutdown_server():
    """
    Release Kinects + robot and exit the process cleanly.
    """
    pipeline = state.pipeline
    if pipeline is not None:
        try:
            pipeline.shutdown()
        except Exception as exc:
            logger.warning("pipeline shutdown raised: %s", exc)

    def _exit_soon():
        time.sleep(0.5)
        os._exit(0)

    threading.Thread(target=_exit_soon, daemon=True).start()
    return {"shutdown": "ok"}


@router.get("/api/status")
async def get_status():
    pipeline = state.pipeline
    if pipeline is None:
        return {"initialized": False, "error": "Pipeline not started"}
    status = pipeline.get_status()
    status["active_camera"] = state.active_camera
    status["available_cameras"] = get_available_cameras()
    return status


_ROBOT_DISPLAY = {
    "ur10e": {"name": "UR10e", "icon": "\U0001f916"},
    "franka": {"name": "Franka FR3", "icon": "\U0001f9be"},
    "g1": {"name": "Unitree G1", "icon": "\U0001f9cd"},
    "bimanual_franka": {"name": "Bimanual Franka (Panda+FR3)", "icon": "\U0001f932"},
}


@router.get("/api/robot")
async def get_robot():
    """
    Return the active robot family + model for UI banner.
    """
    pipeline = state.pipeline
    family = "ur10e"
    model = "UR10e"
    if pipeline is not None and getattr(pipeline, "config", None) is not None:
        family = getattr(pipeline.config, "robot_family", family) or family
        model = getattr(pipeline.config, "robot_model", model) or model
    disp = _ROBOT_DISPLAY.get(family, _ROBOT_DISPLAY["ur10e"])
    return {
        "family": family,
        "model": model,
        "name": disp["name"],
        "icon": disp["icon"],
    }


@router.post("/api/initialize")
async def initialize_pipeline():
    async with state.pipeline_lock:
        if state.pipeline is not None and state.pipeline._initialized:
            return {"status": "already_initialized", **state.pipeline.get_status()}
        # Build through the SAME profile machinery main() uses, for the
        # family the server was launched with. A bare PipelineConfig()
        # ignores --robot and takes the dataclass default family
        # (pipeline_types.py robot_family, "ur10e"), so on any other rig
        # this endpoint would build the wrong grasp orientation, camera roles
        # and workspace.
        profile = None
        if state.launch_config is not None:
            from spark_real.config import load_profile

            profile = load_profile(state.launch_config)
            config = PipelineConfig(**profile.to_pipeline_kwargs())
        else:
            config = PipelineConfig(robot_ip="", use_kinect=True, use_realsense=None)

        # RELEASE THE OLD PIPELINE BEFORE BUILDING A NEW ONE. Reaching here
        # with state.pipeline set means a previous pipeline exists whose
        # initialize() did not finish -- and a half-built pipeline still owns
        # whatever it DID open: PyK4A handles with live capture threads, the
        # RealSense stream, the RTDE connection. Overwriting the reference
        # would leave two handles on one depth MCU, with the old capture
        # threads calling get_capture() forever and nothing releasing them
        # (atexit's _release_devices only walks state.pipeline). shutdown()
        # is idempotent and each close is isolated, so this is safe even on a
        # pipeline that barely got started.
        old = state.pipeline
        if old is not None:
            logger.info("initialize: releasing the previous, uninitialized pipeline")
            try:
                old.shutdown()
            except Exception as exc:  # noqa: BLE001
                logger.warning("initialize: previous pipeline shutdown raised: %s", exc)

        # Build and initialize into a LOCAL, and publish only on success.
        # Assigning state.pipeline first would expose a half-constructed
        # object to every concurrent reader (stream pool, /api/status,
        # recorder) for the whole multi-second initialize, and a raising
        # initialize() would leave that wreck installed as the live pipeline.
        # state.pipeline stays None/old until the new one is ready.
        pipeline = SPARKRealPipeline(config, profile=profile)
        state.pipeline = None
        try:
            pipeline.initialize()
        except BaseException:
            logger.critical(
                "initialize: pipeline.initialize() failed; releasing whatever "
                "it opened rather than leaving devices held by an unreachable "
                "object"
            )
            try:
                pipeline.shutdown()
            except Exception as exc:  # noqa: BLE001
                logger.warning("initialize: cleanup shutdown raised: %s", exc)
            raise
        state.pipeline = pipeline
        return {"status": "initialized", **pipeline.get_status()}


@router.get("/api/capture")
def capture_images():
    pipeline = state.pipeline
    if pipeline is None:
        return JSONResponse(
            status_code=400, content={"error": "Pipeline not initialized"}
        )
    captures = pipeline.capture()
    result = {}
    for cam_name, data in captures.items():
        rgb = data["rgb"]
        depth = data["depth"]
        rgb_b64 = encode_image(rgb)
        depth_b64 = (
            encode_image(depth_to_colormap(depth)) if depth is not None else None
        )
        result[cam_name] = {
            "rgb": rgb_b64,
            "depth": depth_b64,
            "width": rgb.shape[1],
            "height": rgb.shape[0],
        }
        if cam_name == state.active_camera or state.get_last_frame()[0] is None:
            state.set_last_frame(rgb, depth, cam_name)
    return result


@router.get("/api/progress")
async def get_progress(since: int = 0):
    """
    Get recent pipeline progress log entries.

    `since` is a monotonic sequence number, not an index: pass the `seq`
    of the last entry you rendered and you get strictly what followed it.
    Callers that omit it get the whole retained buffer, which is the
    pre-existing contract, so an old frontend against a new server keeps
    working.

    `next_since` is what the caller should send next time. It is returned
    even when `log` is empty -- otherwise a poll that happens to land on
    an empty window would leave the client's cursor stale.
    """
    with state.progress_lock:
        if since > 0:
            entries = [e for e in state.progress_log if e.get("seq", 0) > since]
        else:
            entries = list(state.progress_log)
        latest = state.progress_seq
    return {"log": entries, "next_since": latest}


@router.delete("/api/progress")
async def clear_progress():
    """
    Clear progress log.

    Deliberately does NOT reset progress_seq. The counter is monotonic for
    the life of the process so that a client holding a pre-clear cursor
    cannot be handed entries it already rendered. `next_since` in the
    response lets the caller resynchronise in the same round trip.
    """
    with state.progress_lock:
        state.progress_log.clear()
        latest = state.progress_seq
    return {"cleared": True, "next_since": latest}

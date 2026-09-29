# Calibration routes: DA3 estimate, manual point-pair calibration, and the
# top-level router that composes the anchor, hand-eye, and auto sub-routers.

import logging

import numpy as np
from PIL import Image
from fastapi import APIRouter
from fastapi.responses import JSONResponse

from spark_real.calibration import compute_transform_from_pairs
from spark_real.perception.da3_pose import DA3PoseEstimator
from spark_real.routes import state
from spark_real.routes.models import CalibPointRequest

# Anchor helpers + flow live in calibration_anchor.py. Re-exported here
# because server.py pulls load_saved_anchor / load_saved_anchor_points from
# this module.
from spark_real.routes.calibration_anchor import (  # noqa: F401
    load_saved_anchor,
    load_saved_anchor_points,
)

logger = logging.getLogger("spark_server")
router = APIRouter()


# DA3 estimation


@router.post("/api/da3_estimate")
async def da3_estimate():
    """
    Run DA3 Nested Giant-Large on all active cameras.
    """
    pipeline = state.pipeline
    if pipeline is None:
        return JSONResponse(
            status_code=400, content={"error": "Pipeline not initialized"}
        )
    if state.da3_estimator is None:
        state.da3_estimator = DA3PoseEstimator()
        state.da3_estimator.load()

    captures = pipeline.capture()
    if not captures:
        return JSONResponse(status_code=500, content={"error": "No captures"})

    images, cam_names = [], []
    for name, data in captures.items():
        images.append(Image.fromarray(data["rgb"]))
        cam_names.append(name)

    result = state.da3_estimator.estimate(images)
    cameras = []
    for i, name in enumerate(cam_names):
        R = result.extrinsics[i, :3, :3]
        t = result.extrinsics[i, :3, 3]
        d = result.depth[i]
        cameras.append(
            {
                "name": name,
                "position": (-R.T @ t).tolist(),
                "extrinsic": result.extrinsics[i].tolist(),
                "intrinsic": result.intrinsics[i].tolist(),
                "depth_median": float(np.median(d[d > 0])) if (d > 0).any() else 0,
            }
        )
    return {"num_views": len(images), "cameras": cameras}


# Anchor endpoints + the manual per-camera flow live in calibration_anchor.py.
from spark_real.routes.calibration_anchor import router as _anchor_router

router.include_router(_anchor_router)

# Auto-calibrate and DA3 anchor endpoints live in calibration_auto.py.
from spark_real.routes.calibration_auto import router as _auto_router

router.include_router(_auto_router)


# Manual calibration


@router.post("/api/calibration/add_point")
async def add_calibration_point(req: CalibPointRequest):
    state.calibration_points.append(
        {
            "robot": [req.robot_x, req.robot_y, req.robot_z],
            "camera": [req.camera_x, req.camera_y, req.camera_z],
        }
    )
    return {"count": len(state.calibration_points)}


@router.post("/api/calibration/collect")
async def collect_calibration_point():
    """
    One-click: read TCP, detect closest object, record pair.
    """
    pipeline = state.pipeline
    if pipeline is None or pipeline._robot is None:
        return JSONResponse(
            status_code=400, content={"error": "Robot/pipeline not ready"}
        )
    obs = pipeline._robot.get_observation()
    tcp = obs["tcp_pose"][:3].tolist()
    captures = pipeline.capture()
    if not captures:
        return JSONResponse(status_code=500, content={"error": "No captures"})
    dets = pipeline.detect(captures, ["gripper", "gripper tip", "object"])
    if not dets:
        return JSONResponse(status_code=400, content={"error": "No detections found"})
    best = max(dets, key=lambda d: d.confidence)
    if best.position_3d is None:
        return JSONResponse(status_code=400, content={"error": "No 3D position"})
    cam_pos = (
        best.position_3d.tolist()
        if isinstance(best.position_3d, np.ndarray)
        else best.position_3d
    )
    state.calibration_points.append({"robot": tcp, "camera": cam_pos})
    return {"count": len(state.calibration_points), "robot": tcp, "camera": cam_pos}


@router.post("/api/calibration/solve")
async def solve_calibration():
    """
    Compute camera->robot transform from collected points.
    """
    if len(state.calibration_points) < 3:
        return JSONResponse(
            status_code=400,
            content={"error": f"Need 3+ points, have {len(state.calibration_points)}"},
        )
    pipeline = state.pipeline
    if pipeline is None:
        return JSONResponse(
            status_code=400, content={"error": "Pipeline not initialized"}
        )
    robot_pts = [p["robot"] for p in state.calibration_points]
    camera_pts = [p["camera"] for p in state.calibration_points]
    T = compute_transform_from_pairs(robot_pts, camera_pts)
    for cal in [pipeline._kinect_cal, pipeline._kinect2_cal, pipeline._realsense_cal]:
        if cal is not None:
            cal.extrinsic = T
    errors = [
        float(np.linalg.norm(np.array(r) - (T @ np.append(c, 1.0))[:3]))
        for r, c in zip(robot_pts, camera_pts)
    ]
    return {
        "success": True,
        "num_points": len(state.calibration_points),
        "mean_error_mm": float(np.mean(errors) * 1000),
        "transform": T.tolist(),
    }


@router.post("/api/calibration/reset")
async def reset_calibration():
    state.calibration_points.clear()
    return {"count": 0}


@router.get("/api/calibration/points")
async def get_calibration_points():
    return {"count": len(state.calibration_points), "points": state.calibration_points}


# Hand-eye script-driver endpoints live in calibration_handeye.py.
from spark_real.routes.calibration_handeye import router as _handeye_router

router.include_router(_handeye_router)


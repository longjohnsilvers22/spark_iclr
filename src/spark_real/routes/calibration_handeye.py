"""
Hand-eye script-driver endpoints (split from calibration.py).

These endpoints back the external hand-eye calibration script: drive the
TCP to absolute poses via CartesianServo, return RGB/depth frames, probe
ArUco/AprilTag visibility, and persist the solved transform. The frame
and pose helpers _wait_stationary and _resolve_cam_lock are owned here
because only this endpoint group uses them.
"""

import base64
import io
import json
import time
from contextlib import nullcontext as _nullctx
from pathlib import Path

import cv2
import numpy as np
from PIL import Image
from fastapi import APIRouter
from fastapi.responses import JSONResponse

from spark_real.config import family_block
from spark_real.calibration_table import save_table_plane
from spark_real.control.cartesian_servo import CartesianServo
from spark_real.routes import state

router = APIRouter()


@router.post("/api/calibrate/move_to_pose")
async def calibrate_move_to_pose(req: dict):
    """
    Drive TCP to an absolute pose. Used by hand-eye script.
    """
    pipeline = state.pipeline
    if pipeline is None or pipeline._robot is None:
        return JSONResponse(status_code=400, content={"error": "Robot not connected"})
    pose = req.get("pose")
    if not pose or len(pose) != 6:
        return JSONResponse(
            status_code=400, content={"error": "pose must be [x,y,z,rx,ry,rz]"}
        )
    velocity = float(req.get("velocity", 0.15))
    # The servo needs the driver interface (get_tcp_pose / send_velocity).
    # Unwrap one SafeRobot layer for calibration moves, but only while the
    # inner object still speaks that interface; the bare franky handle does
    # not, and unwrapping onto it leaves the servo reading the pose as zeros.
    raw_robot = getattr(pipeline._robot, "_robot", pipeline._robot)
    if not hasattr(raw_robot, "get_tcp_pose"):
        raw_robot = pipeline._robot
    if hasattr(raw_robot, "stop_velocity"):
        try:
            raw_robot.stop_velocity()
        except Exception:
            pass
    state.cal_active = True
    try:
        servo = getattr(pipeline, "_cal_servo", None)
        if servo is None or servo.robot is not raw_robot:
            servo = CartesianServo(raw_robot, rate_hz=30.0)
            pipeline._cal_servo = servo
        servo.max_vel_linear = max(0.03, min(velocity, 0.25))
        converged = servo.move_to_pose(list(pose), velocity=velocity, timeout=15.0)
        if not converged:
            return JSONResponse(
                status_code=500, content={"error": "servo did not converge"}
            )
    except Exception as exc:
        return JSONResponse(status_code=500, content={"error": f"move failed: {exc}"})
    finally:
        state.cal_active = False

    _wait_stationary(pipeline._robot)  # wait until stationary
    try:
        tcp = pipeline._robot.get_tcp_pose()
        if hasattr(tcp, "tolist"):
            tcp = tcp.tolist()
    except Exception:
        tcp = None
    return {"success": True, "tcp_pose": tcp}


@router.get("/api/calibrate/capture_one")
async def calibrate_capture_one(camera: str = "birdview"):
    """
    Return latest RGB frame as PNG for hand-eye calibration.
    """
    pipeline = state.pipeline
    if pipeline is None:
        return JSONResponse(
            status_code=400, content={"error": "Pipeline not initialized"}
        )
    cam, lock = _resolve_cam_lock(pipeline, camera)
    if cam is None:
        return JSONResponse(
            status_code=400, content={"error": f"camera {camera!r} not online"}
        )
    try:
        with lock if lock is not None else _nullctx():
            rgb, _ = (
                cam.read(depth=False)
                if camera in ("birdview", "sideview")
                else cam.read()
            )
    except Exception as exc:
        return JSONResponse(status_code=500, content={"error": f"read failed: {exc}"})
    if rgb is None:
        return JSONResponse(status_code=500, content={"error": "no frame"})
    buf = io.BytesIO()
    Image.fromarray(rgb).save(buf, format="PNG")
    cal = getattr(
        pipeline,
        {
            "birdview": "_kinect2_cal",
            "sideview": "_kinect_cal",
            "wrist": "_realsense_cal",
        }[camera],
        None,
    )
    intr = None
    if cal is not None:
        intr = {
            "fx": float(getattr(cal, "fx", 0.0) or 0.0),
            "fy": float(getattr(cal, "fy", 0.0) or 0.0),
            "cx": float(getattr(cal, "cx", 0.0) or 0.0),
            "cy": float(getattr(cal, "cy", 0.0) or 0.0),
        }
    return {
        "camera": camera,
        "rgb_png_base64": base64.b64encode(buf.getvalue()).decode("ascii"),
        "intrinsics": intr,
        "shape": list(rgb.shape),
    }


@router.get("/api/calibrate/capture_depth")
async def calibrate_capture_depth(camera: str = "birdview"):
    """
    Return latest depth frame as 16-bit PNG.
    """
    pipeline = state.pipeline
    if pipeline is None:
        return JSONResponse(
            status_code=400, content={"error": "Pipeline not initialized"}
        )
    cam, lock = _resolve_cam_lock(pipeline, camera)
    if cam is None:
        return JSONResponse(
            status_code=400, content={"error": f"camera {camera!r} not online"}
        )
    try:
        with lock if lock is not None else _nullctx():
            _, depth = cam.read()
    except Exception as exc:
        return JSONResponse(status_code=500, content={"error": f"read failed: {exc}"})
    if depth is None:
        return JSONResponse(status_code=500, content={"error": "no depth"})
    depth_mm = np.clip(np.nan_to_num(depth, nan=0.0) * 1000.0, 0, 65535).astype(
        np.uint16
    )
    buf = io.BytesIO()
    Image.fromarray(depth_mm, mode="I;16").save(buf, format="PNG")
    return {
        "camera": camera,
        "depth_png_base64": base64.b64encode(buf.getvalue()).decode("ascii"),
        "depth_scale_m_per_unit": 0.001,
        "shape": list(depth.shape),
    }


@router.get("/api/calibrate/detect_tag")
async def calibrate_detect_tag(camera: str = "birdview", tag_dict: str = "auto"):
    """
    Quick tag visibility probe for hand-eye calibration.
    """
    pipeline = state.pipeline
    if pipeline is None:
        return JSONResponse(
            status_code=400, content={"error": "Pipeline not initialized"}
        )
    cam, lock = _resolve_cam_lock(pipeline, camera)
    if cam is None:
        return {"detected": False, "error": f"{camera} not connected"}
    try:
        with lock if lock is not None else _nullctx():
            rgb, _ = cam.read()
    except Exception as exc:
        return {"detected": False, "error": f"read failed: {exc}"}
    if rgb is None:
        return {"detected": False, "error": "no frame"}
    bgr = cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR)

    params = cv2.aruco.DetectorParameters()
    params.adaptiveThreshWinSizeMin = 3
    params.adaptiveThreshWinSizeMax = 53
    params.adaptiveThreshWinSizeStep = 4
    params.minMarkerPerimeterRate = 0.005
    params.cornerRefinementMethod = cv2.aruco.CORNER_REFINE_SUBPIX

    ALL_DICTS = {
        "DICT_APRILTAG_36h11": cv2.aruco.DICT_APRILTAG_36h11,
        "DICT_4X4_50": cv2.aruco.DICT_4X4_50,
        "DICT_5X5_50": cv2.aruco.DICT_5X5_50,
        "DICT_6X6_50": cv2.aruco.DICT_6X6_50,
        "DICT_ARUCO_ORIGINAL": cv2.aruco.DICT_ARUCO_ORIGINAL,
    }
    dicts_to_try = (
        ALL_DICTS
        if tag_dict == "auto"
        else {
            tag_dict: getattr(cv2.aruco, tag_dict, None),
        }
    )

    hits = []
    for name, did in dicts_to_try.items():
        if did is None:
            continue
        detector = cv2.aruco.ArucoDetector(
            cv2.aruco.getPredefinedDictionary(did),
            params,
        )
        corners, ids, _ = detector.detectMarkers(bgr)
        if ids is None:
            continue
        for k in range(len(ids)):
            c = corners[k].reshape(4, 2)
            s1 = float(np.linalg.norm(c[0] - c[1]))
            s2 = float(np.linalg.norm(c[1] - c[2]))
            min_side = min(s1, s2)
            if min_side < 25.0:
                continue
            hits.append(
                {
                    "dict_used": name,
                    "id": int(ids[k][0]),
                    "side_px": round(min_side, 1),
                    "center_xy": [
                        round(float(c.mean(axis=0)[0])),
                        round(float(c.mean(axis=0)[1])),
                    ],
                }
            )

    if not hits:
        return {"detected": False, "camera": camera}
    hits.sort(key=lambda h: -h["side_px"])
    return {"detected": True, "camera": camera, **hits[0]}


@router.post("/api/calibrate/save_handeye")
async def calibrate_save_handeye(req: dict):
    """
    Persist hand-eye result to disk and apply to live pipeline.
    """
    camera = req.get("camera")
    T = req.get("transform_4x4")
    if camera not in ("birdview", "sideview", "wrist") or T is None:
        return JSONResponse(
            status_code=400, content={"error": "need {camera, transform_4x4}"}
        )
    Tnp = np.array(T, dtype=float)
    if Tnp.shape != (4, 4):
        return JSONResponse(status_code=400, content={"error": "transform must be 4x4"})

    out_dir = Path(__file__).resolve().parents[1] / "output" / "calibrations"
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / f"handeye_{camera}.json"
    out_path.write_text(
        json.dumps(
            {
                "camera": camera,
                "mode": req.get("mode"),
                "transform_4x4": Tnp.tolist(),
                "residual_mm": req.get("residual_mm"),
                "num_poses": req.get("num_poses"),
                "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
            },
            indent=2,
        )
    )

    pipeline = state.pipeline
    applied = False
    if pipeline is not None:
        if camera == "wrist":
            pipeline._wrist_tool_offset = Tnp
            applied = True
        else:
            cal = getattr(
                pipeline,
                {"birdview": "_kinect2_cal", "sideview": "_kinect_cal"}[camera],
                None,
            )
            if cal is not None:
                cal.extrinsic = Tnp
                applied = True
    return {"saved": str(out_path), "applied": applied}


# Helpers


def _wait_stationary(robot, timeout=2.0):
    try:
        deadline = time.monotonic() + timeout
        prev = np.asarray(robot.get_joint_positions(), dtype=float)
        while time.monotonic() < deadline:
            time.sleep(0.08)
            now = np.asarray(robot.get_joint_positions(), dtype=float)
            if float(np.max(np.abs(now - prev))) < 1e-4:
                break
            prev = now
    except Exception:
        time.sleep(0.5)


@router.post("/api/calibrate_table")
async def calibrate_table(req: dict):
    """Record the table surface from the CURRENT TCP z.

    The operator jogs the CLOSED gripper down until the fingertips touch the
    table, then calls this. We read TCP z, persist it as surface_z, and the
    floor is surface_z + safety_margin_m (a few mm above the surface). The
    executor prefers this cal over the YAML control.table_z_floor fallback.
    """
    pipeline = state.pipeline
    if pipeline is None or pipeline._robot is None:
        return JSONResponse(status_code=400, content={"error": "Robot not connected"})

    # Unwrap SafeRobot to reach get_tcp_pose, same as the hand-eye moves.
    raw_robot = getattr(pipeline._robot, "_robot", pipeline._robot)
    if not hasattr(raw_robot, "get_tcp_pose"):
        raw_robot = pipeline._robot
    if not hasattr(raw_robot, "get_tcp_pose"):
        return JSONResponse(
            status_code=400, content={"error": "robot has no get_tcp_pose"}
        )
    current_tcp_z = float(raw_robot.get_tcp_pose()[2])

    family = (
        getattr(getattr(pipeline, "config", None), "robot_family", "ur10e") or "ur10e"
    ).lower()

    # Margin from the request, else the family control block, else 0.003.
    margin = req.get("safety_margin_m")
    if margin is None:
        margin = family_block(
            getattr(pipeline, "profile", None), family, "control"
        ).get("table_safety_margin_m")
    margin = float(margin) if margin is not None else 0.003

    path = save_table_plane(family, current_tcp_z, margin)
    return {
        "surface_z": current_tcp_z,
        "floor_z": current_tcp_z + margin,
        "path": str(path),
    }


def _resolve_cam_lock(pipeline, camera):
    cam_attr = {
        "birdview": "_kinect2",
        "sideview": "_kinect",
        "wrist": "_realsense",
    }.get(camera)
    lock_attr = {
        "birdview": "_kinect2_read_lock",
        "sideview": "_kinect_read_lock",
        "wrist": "_realsense_read_lock",
    }.get(camera)
    if cam_attr is None:
        return None, None
    return getattr(pipeline, cam_attr, None), getattr(pipeline, lock_attr, None)

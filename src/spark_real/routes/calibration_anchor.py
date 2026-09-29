"""
Manual per-camera anchor-point calibration flow (split from calibration.py).

Owns the click-to-collect anchor workflow: read the TCP, back-project a
clicked pixel to a 3D camera point, accumulate per-camera (robot, camera)
pairs, and solve each camera extrinsic by Procrustes/SVD. This module is
the single owner of the family-path helpers (_active_family, _anchor_path,
_saved_points_path, _cam_cal_map) and the legacy path constants; the shared
mutable anchor-point lists live in routes.state. calibration.py re-exports
the helpers so the existing calibration_auto import path keeps working.
"""

import json
import logging
from pathlib import Path

import numpy as np
from fastapi import APIRouter
from fastapi.responses import JSONResponse

from spark_real.calibration import compute_transform_from_pairs
from spark_real.routes import state
from spark_real.routes.models import AnchorRequest

logger = logging.getLogger("spark_server")
router = APIRouter()

_PARENT = Path(__file__).parent.parent
_LEGACY_ANCHOR_PATH = _PARENT / "da3_anchor.json"
_LEGACY_SAVED_POINTS_PATH = _PARENT / "saved_anchor_points.json"


def _active_family() -> str:
    pipeline = state.pipeline
    if pipeline is not None:
        fam = getattr(pipeline.config, "robot_family", None)
        if fam:
            return str(fam).lower()
    return "ur10e"


def _anchor_path(family: str = None) -> Path:
    return _PARENT / f"da3_anchor_{(family or _active_family()).lower()}.json"


def _saved_points_path(family: str = None) -> Path:
    return _PARENT / f"saved_anchor_points_{(family or _active_family()).lower()}.json"


def _cam_cal_map(pipeline):
    return {
        "sideview": pipeline._kinect_cal,
        "birdview": pipeline._kinect2_cal,
        "wrist": pipeline._realsense_cal,
    }


# Anchor points (manual per-camera calibration)


def _save_anchor_points():
    data = {}
    for cam_name, pts in state.anchor_points_per_cam.items():
        if pts:
            data[cam_name] = [{"tcp": r.tolist(), "camera": c.tolist()} for r, c in pts]
    _saved_points_path().write_text(json.dumps(data, indent=2))


def load_saved_anchor_points():
    """
    Load saved anchor points for the active family.

    The legacy untagged saved_anchor_points.json is an FR3-era artifact;
    only fall back to it for the franka family (matches load_saved_anchor).
    """
    fam = _active_family()
    suffixed = _saved_points_path(fam)
    if suffixed.exists():
        src = suffixed
    elif fam == "franka" and _LEGACY_SAVED_POINTS_PATH.exists():
        src = _LEGACY_SAVED_POINTS_PATH
        logger.warning(
            "anchor: family-specific %s absent; falling back to LEGACY "
            "FR3-era %s",
            suffixed.name,
            _LEGACY_SAVED_POINTS_PATH.name,
        )
    else:
        src = None
    if src is None:
        return
    try:
        data = json.loads(src.read_text())
        for cam_name, points in data.items():
            state.anchor_points_per_cam[cam_name] = [
                (np.array(p["tcp"]), np.array(p["camera"])) for p in points
            ]
    except Exception as e:
        logger.warning("Failed to load saved anchor points: %s", e)


@router.post("/api/anchor_point")
async def add_anchor_point(req: AnchorRequest):
    """
    Add one calibration point by clicking on gripper tip in a camera tile.
    """
    pipeline = state.pipeline
    if pipeline is None or pipeline._robot is None:
        return JSONResponse(
            status_code=400, content={"error": "Robot/pipeline not ready"}
        )
    obs = pipeline._robot.get_observation()
    tcp_pos = np.array(obs["tcp_pose"][:3])

    if not state.tile_layout:
        return JSONResponse(status_code=400, content={"error": "No tile layout"})

    tiled_w = sum(t["width"] for t in state.tile_layout)
    click_x = req.x * tiled_w / req.img_width
    click_y = req.y * 480 / req.img_height

    clicked_cam, click_in_tile_x = None, 0
    for t in state.tile_layout:
        if t["x_start"] <= click_x < t["x_start"] + t["width"]:
            clicked_cam = t["cam"]
            click_in_tile_x = click_x - t["x_start"]
            break
    if clicked_cam is None:
        return JSONResponse(
            status_code=400, content={"error": "Click inside a camera tile"}
        )

    tile_info = next(t for t in state.tile_layout if t["cam"] == clicked_cam)
    orig_u = click_in_tile_x * tile_info["orig_w"] / tile_info["width"]
    orig_v = click_y * tile_info["orig_h"] / 480

    captures = pipeline.capture()
    cam_data = captures.get(clicked_cam)
    if cam_data is None:
        return JSONResponse(
            status_code=400, content={"error": f"No capture for {clicked_cam}"}
        )

    depth = cam_data["depth"]
    cal = cam_data["calibration"]
    u, v = int(orig_u), int(orig_v)
    h, w = depth.shape
    patch_r = 5
    patch = depth[
        max(0, v - patch_r) : min(h, v + patch_r + 1),
        max(0, u - patch_r) : min(w, u + patch_r + 1),
    ]
    valid = patch[(patch > 0.01) & (patch < 10)]
    if len(valid) == 0:
        return JSONResponse(status_code=400, content={"error": "No depth at click"})

    d = float(np.median(valid))
    K = cal.intrinsic_matrix
    cam_point = np.array(
        [
            (orig_u - K[0, 2]) * d / K[0, 0],
            (orig_v - K[1, 2]) * d / K[1, 1],
            d,
        ]
    )

    state.anchor_points_per_cam.setdefault(clicked_cam, [])
    state.anchor_points_per_cam[clicked_cam].append((tcp_pos.copy(), cam_point))
    _save_anchor_points()
    counts = {k: len(v) for k, v in state.anchor_points_per_cam.items() if v}
    return {
        "point_index": sum(len(v) for v in state.anchor_points_per_cam.values()),
        "tcp_position": tcp_pos.tolist(),
        "camera_position": cam_point.tolist(),
        "depth": d,
        "camera": clicked_cam,
        "per_camera_counts": counts,
    }


@router.post("/api/anchor_compute")
async def compute_anchor():
    """
    Compute per-camera transforms from collected anchor points via SVD.
    """
    pipeline = state.pipeline
    if pipeline is None:
        return JSONResponse(
            status_code=400, content={"error": "Pipeline not initialized"}
        )

    calibratable = {k: v for k, v in state.anchor_points_per_cam.items() if len(v) >= 3}
    if not calibratable:
        counts = {k: len(v) for k, v in state.anchor_points_per_cam.items() if v}
        return JSONResponse(
            status_code=400,
            content={"error": f"Need 3+ points per camera. Current: {counts}"},
        )

    cal_map = _cam_cal_map(pipeline)
    results = {}
    cam_extrinsics_save = {}

    for cam_name, points in calibratable.items():
        robot_pts = [p[0].tolist() for p in points]
        camera_pts = [p[1].tolist() for p in points]
        T = compute_transform_from_pairs(robot_pts, camera_pts)

        cal = cal_map.get(cam_name)
        if cal:
            cal.extrinsic = T.copy()
            cam_extrinsics_save[cam_name] = T.tolist()

        errors = [
            float(np.linalg.norm(T[:3, :3] @ c + T[:3, 3] - r)) for r, c in points
        ]
        results[cam_name] = {
            "num_points": len(points),
            "mean_error_m": float(np.mean(errors)),
            "max_error_m": float(np.max(errors)),
        }

    # Copy best transform to uncalibrated cameras
    best_cam = min(results, key=lambda k: results[k]["mean_error_m"])
    best_T = cal_map[best_cam].extrinsic
    for cam_name in ["sideview", "birdview", "wrist"]:
        if cam_name not in calibratable and cal_map.get(cam_name):
            cal_map[cam_name].extrinsic = best_T.copy()
            cam_extrinsics_save[cam_name] = best_T.tolist()

    state.da3_anchor = best_T
    out_path = _anchor_path()
    with open(out_path, "w") as f:
        json.dump(
            {
                "method": "per_camera_svd",
                "camera_extrinsics": cam_extrinsics_save,
                "per_camera_results": results,
                "robot_family": _active_family(),
            },
            f,
            indent=2,
        )

    return {
        "success": True,
        "per_camera": results,
        "calibrated": list(calibratable.keys()),
    }


@router.post("/api/anchor_clear")
async def clear_anchor_points(delete_files: bool = True, camera: str = ""):
    """
    Clear anchor calibration for the active robot family.
    """
    selective = bool(camera)
    cam_key = (camera or "").lower()

    if selective:
        if cam_key not in ("sideview", "birdview", "wrist"):
            return JSONResponse(
                status_code=400, content={"error": f"Unknown camera '{camera}'"}
            )
        state.anchor_points_per_cam[cam_key] = []
    else:
        state.anchor_points_per_cam = {
            "sideview": [],
            "birdview": [],
            "wrist": [],
        }

    removed = []
    if delete_files and not selective:
        for path in (_saved_points_path(), _anchor_path()):
            try:
                if path.exists():
                    path.unlink()
                    removed.append(path.name)
            except Exception:
                pass
    elif selective:
        try:
            _save_anchor_points()
        except Exception:
            pass

    pipeline = state.pipeline
    if pipeline is not None:
        I4 = np.eye(4)
        cam_attrs = {
            "sideview": "_kinect_cal",
            "birdview": "_kinect2_cal",
            "wrist": "_realsense_cal",
        }
        targets = [cam_key] if selective else list(cam_attrs.keys())
        for name in targets:
            cal = getattr(pipeline, cam_attrs[name], None)
            if cal is not None and hasattr(cal, "extrinsic"):
                cal.extrinsic = I4.copy()
        if not selective and hasattr(pipeline, "_persp_affine"):
            pipeline._persp_affine = None
    if not selective:
        state.da3_anchor = None

    return {
        "cleared": True,
        "camera": cam_key or "all",
        "removed_files": removed,
        "remaining": {k: len(v) for k, v in state.anchor_points_per_cam.items() if v},
    }


@router.post("/api/anchor_remove")
async def remove_anchor_point(camera: str = "sideview", index: int = -1):
    """
    Remove a specific anchor point by camera and index.
    """
    if camera not in state.anchor_points_per_cam:
        return JSONResponse(
            status_code=400, content={"error": f"Unknown camera: {camera}"}
        )
    pts = state.anchor_points_per_cam[camera]
    if not pts:
        return JSONResponse(
            status_code=400, content={"error": f"No points for {camera}"}
        )
    if index < 0:
        index = len(pts) + index
    if index < 0 or index >= len(pts):
        return JSONResponse(
            status_code=400, content={"error": f"Index {index} out of range"}
        )
    pts.pop(index)
    return {
        "removed": camera,
        "index": index,
        "remaining": {k: len(v) for k, v in state.anchor_points_per_cam.items() if v},
    }


@router.get("/api/anchor_points")
async def list_anchor_points():
    result = {}
    for cam, pts in state.anchor_points_per_cam.items():
        if pts:
            result[cam] = [
                {"index": i, "tcp": p[0].tolist(), "cam_3d": p[1].tolist()}
                for i, p in enumerate(pts)
            ]
    return result


def load_saved_anchor(pipeline):
    """
    Auto-load saved anchor calibration for the active family.

    The legacy untagged da3_anchor.json is an FR3-era artifact, so it is only
    an acceptable fallback for the franka family. On other families it carries
    the wrong extrinsics and, loading after _load_handeye_calibrations, would
    clobber the hand-eye transforms that loader applied; gate it to franka and
    log loudly when the legacy file overwrites the cals.
    """
    fam = (getattr(pipeline.config, "robot_family", "ur10e") or "ur10e").lower()
    suffixed = _anchor_path(fam)
    if suffixed.exists():
        src = suffixed
    elif fam == "franka" and _LEGACY_ANCHOR_PATH.exists():
        src = _LEGACY_ANCHOR_PATH
        logger.warning(
            "anchor: family-specific %s absent; falling back to LEGACY "
            "FR3-era %s which will OVERWRITE the hand-eye extrinsics loaded "
            "by _load_handeye_calibrations (sideview/birdview)",
            suffixed.name,
            _LEGACY_ANCHOR_PATH.name,
        )
    else:
        src = None
    if src is None:
        return
    try:
        data = json.loads(src.read_text())
        cam_extrinsics = data.get("camera_extrinsics", {})
        if not cam_extrinsics:
            return
        for name, ext_list in cam_extrinsics.items():
            ext = np.array(ext_list).reshape(4, 4)
            if name == "sideview" and pipeline._kinect_cal:
                pipeline._kinect_cal.extrinsic = ext
            elif name == "birdview" and pipeline._kinect2_cal:
                pipeline._kinect2_cal.extrinsic = ext
        if "transform" in data:
            state.da3_anchor = np.array(data["transform"]).reshape(4, 4)
        pa = data.get("perspective_affine")
        if pa:
            pipeline._persp_affine = pa
        th = data.get("table_height")
        if th is not None:
            pipeline.config.table_height = float(th)
    except Exception as e:
        logger.warning("Failed to load anchor calibration: %s", e)

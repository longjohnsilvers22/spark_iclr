# Auto-calibrate and DA3 anchor calibration endpoints (split from calibration.py).

import json
import logging
import time

import cv2
import numpy as np
from PIL import Image
from scipy.spatial.transform import Rotation as _R
from fastapi import APIRouter
from fastapi.responses import JSONResponse

from spark_real.perception.da3_pose import DA3PoseEstimator
from spark_real.routes import state
from spark_real.routes.calibration_anchor import _active_family, _anchor_path, _cam_cal_map
from spark_real.routes.models import AnchorRequest

logger = logging.getLogger("spark_server")
router = APIRouter()


# TCP read + guarded move used only by the auto-calibrate flow.
def _read_tcp_orient(robot):
    if hasattr(robot, "get_observation"):
        obs = robot.get_observation()
        return np.array(obs["tcp_pose"][:3]), np.array(obs["tcp_pose"][3:6])
    p = robot.get_tcp_pose()
    if isinstance(p, np.ndarray) and p.shape == (4, 4):
        return p[:3, 3].copy(), _R.from_matrix(p[:3, :3]).as_rotvec()
    return np.array(p[:3]), np.array(p[3:6])


def _read_tcp_pos(robot):
    if hasattr(robot, "get_observation"):
        return np.array(robot.get_observation()["tcp_pose"][:3])
    p = robot.get_tcp_pose()
    if isinstance(p, np.ndarray) and p.shape == (4, 4):
        return p[:3, 3].copy()
    return np.array(p[:3])


def _calib_move(robot, servo_obj, pose, vel):
    for _ in range(2):
        try:
            if hasattr(robot, "recover_from_errors"):
                try:
                    robot.recover_from_errors()
                except Exception:
                    pass
            if servo_obj is not None:
                servo_obj.max_vel_linear = min(vel, 0.15)
                if servo_obj.move_to_pose(list(pose), velocity=vel):
                    return
            elif hasattr(robot, "move_to_pose"):
                robot.move_to_pose(pose, velocity=vel)
                return
            else:
                robot.move_linear(pose, velocity=vel)
                return
        except Exception:
            time.sleep(1.0)
    raise RuntimeError("calib move failed twice")


@router.post("/api/auto_calibrate")
async def auto_calibrate():
    """
    Automatic hand-eye calibration via image differencing + PnP.
    """
    pipeline = state.pipeline
    if pipeline is None or pipeline._robot is None:
        return JSONResponse(
            status_code=400, content={"error": "Robot/pipeline not ready"}
        )

    cam_name = "birdview"
    cal = pipeline._kinect2_cal
    if cal is None:
        cam_name = "sideview"
        cal = pipeline._kinect_cal
    if cal is None:
        return JSONResponse(status_code=400, content={"error": "No camera available"})

    K = cal.intrinsic_matrix
    fx, fy = K[0, 0], K[1, 1]
    cx, cy = K[0, 2], K[1, 2]

    if hasattr(pipeline._robot, "move_home"):
        pipeline._robot.move_home(velocity=0.3)
    else:
        pipeline._robot.go_home()
    time.sleep(8)

    home_tcp, home_orient = _read_tcp_orient(pipeline._robot)

    offsets = [
        [0.0, 0.0, 0.0],
        [0.18, 0.0, 0.0],
        [0.18, 0.0, -0.18],
        [0.18, 0.0, 0.0],
        [-0.18, 0.0, 0.0],
        [0.0, 0.18, 0.0],
        [0.0, 0.18, -0.18],
        [0.0, 0.18, 0.0],
        [0.0, -0.18, 0.0],
        [0.0, -0.18, -0.18],
        [0.0, 0.0, -0.18],
        [0.10, 0.10, -0.10],
        [-0.10, -0.10, -0.10],
    ]

    _servo_obj = getattr(pipeline, "_executor", None) and pipeline._executor._servo
    state.anchor_points_per_cam["birdview"] = []
    results = []

    for i, offset in enumerate(offsets):
        target = home_tcp + np.array(offset)
        pose = list(target) + list(home_orient)
        try:
            _calib_move(pipeline._robot, _servo_obj, pose, 0.08)
        except Exception:
            try:
                if hasattr(pipeline._robot, "recover_from_errors"):
                    pipeline._robot.recover_from_errors()
            except Exception:
                pass
            continue
        time.sleep(1.0)

        tcp_pos = _read_tcp_pos(pipeline._robot)
        cap_a = pipeline.capture()
        if cam_name not in cap_a:
            continue
        rgb_a = cap_a[cam_name]["rgb"]
        depth_a = cap_a[cam_name]["depth"]

        jog_pose = list(tcp_pos.copy()) + list(home_orient)
        jog_pose[2] += 0.05
        _calib_move(pipeline._robot, _servo_obj, jog_pose, 0.05)
        time.sleep(0.5)
        cap_b = pipeline.capture()
        if cam_name not in cap_b:
            continue
        _calib_move(pipeline._robot, _servo_obj, pose, 0.08)
        time.sleep(0.5)

        gray_a = cv2.cvtColor(rgb_a, cv2.COLOR_RGB2GRAY).astype(np.float32)
        gray_b = cv2.cvtColor(cap_b[cam_name]["rgb"], cv2.COLOR_RGB2GRAY).astype(
            np.float32
        )
        diff = cv2.GaussianBlur(np.abs(gray_a - gray_b), (15, 15), 0)
        mask = diff > diff.max() * 0.3
        if mask.sum() < 10:
            continue

        ys, xs = np.where(mask)
        gu, gv = float(xs.mean()), float(ys.mean())
        pr = 8
        patch = depth_a[
            max(0, int(gv) - pr) : min(depth_a.shape[0], int(gv) + pr + 1),
            max(0, int(gu) - pr) : min(depth_a.shape[1], int(gu) + pr + 1),
        ]
        valid = patch[(patch > 0.01) & (patch < 10)]
        if len(valid) == 0:
            continue

        d = float(np.median(valid))
        cam_point = np.array([(gu - cx) * d / fx, (gv - cy) * d / fy, d])
        state.anchor_points_per_cam["birdview"].append((tcp_pos.copy(), cam_point))
        results.append(
            {
                "tcp": tcp_pos.tolist(),
                "pixel": [gu, gv],
                "depth": d,
                "camera_point": cam_point.tolist(),
            }
        )

    try:
        _calib_move(
            pipeline._robot, _servo_obj, list(home_tcp) + list(home_orient), 0.20
        )
    except Exception:
        pass

    good_results = [r for r in results if r["depth"] < 2.0]
    if len(good_results) < 3:
        return JSONResponse(
            status_code=400,
            content={
                "error": f"Only {len(good_results)} good points. Need 3+.",
            },
        )

    obj_pts = np.array([r["tcp"] for r in results], dtype=np.float64)
    img_pts = np.array([r["pixel"] for r in results], dtype=np.float64)
    success, rvec, tvec = cv2.solvePnP(
        obj_pts,
        img_pts,
        K.astype(np.float64),
        np.zeros(4),
        flags=cv2.SOLVEPNP_SQPNP,
    )
    if success:
        rvec, tvec = cv2.solvePnPRefineLM(
            obj_pts,
            img_pts,
            K.astype(np.float64),
            np.zeros(4),
            rvec,
            tvec,
        )
    if not success:
        return JSONResponse(status_code=400, content={"error": "PnP solve failed"})

    R_c2r = cv2.Rodrigues(rvec)[0].T
    T = np.eye(4)
    T[:3, :3] = R_c2r
    T[:3, 3] = -R_c2r @ tvec.flatten()

    mean_err = float(
        np.linalg.norm(
            cv2.projectPoints(obj_pts, rvec, tvec, K.astype(np.float64), np.zeros(4))[
                0
            ].squeeze()
            - img_pts,
            axis=1,
        ).mean()
    )

    if cam_name == "birdview" and pipeline._kinect2_cal:
        pipeline._kinect2_cal.extrinsic = T.copy()
    elif cam_name == "sideview" and pipeline._kinect_cal:
        pipeline._kinect_cal.extrinsic = T.copy()

    state.da3_anchor = T
    with open(_anchor_path(), "w") as f:
        json.dump(
            {
                "method": "auto_calibrate_pnp",
                "num_points": len(good_results),
                "mean_reproj_error_px": mean_err,
                "transform": T.tolist(),
            },
            f,
            indent=2,
        )

    return {
        "success": True,
        "num_points": len(good_results),
        "mean_reproj_error_px": mean_err,
    }


@router.post("/api/anchor")
async def anchor_calibration(req):
    """
    Click-based DA3 anchor calibration: click on gripper tip in tiled view.
    """
    pipeline = state.pipeline
    if pipeline is None or pipeline._robot is None:
        return JSONResponse(
            status_code=400, content={"error": "Robot/pipeline not ready"}
        )

    obs = pipeline._robot.get_observation()
    tcp_pos = obs["tcp_pose"][:3]

    if state.da3_estimator is None:
        state.da3_estimator = DA3PoseEstimator()
        state.da3_estimator.load()

    captures = pipeline.capture()
    if not captures or not state.tile_layout:
        return JSONResponse(
            status_code=400, content={"error": "No captures or tile layout"}
        )

    tiled_w = sum(t["width"] for t in state.tile_layout)
    click_x = req.x * tiled_w / req.img_width
    click_y = req.y * 480 / req.img_height

    clicked_cam, click_in_tile_x = None, 0
    for t in state.tile_layout:
        if t["x_start"] <= click_x < t["x_start"] + t["width"]:
            clicked_cam = t["cam"]
            click_in_tile_x = click_x - t["x_start"]
            break
    if clicked_cam is None or clicked_cam == "wrist":
        return JSONResponse(
            status_code=400, content={"error": "Click on a fixed camera"}
        )

    tile_info = next(t for t in state.tile_layout if t["cam"] == clicked_cam)
    orig_u = click_in_tile_x * tile_info["orig_w"] / tile_info["width"]
    orig_v = click_y * tile_info["orig_h"] / 480

    images, cam_names, known_K = [], [], []
    for name, data in captures.items():
        images.append(Image.fromarray(data["rgb"]))
        cam_names.append(name)
        cal = _cam_cal_map(pipeline).get(name)
        known_K.append(cal.intrinsic_matrix if cal else None)

    intrinsics_array = (
        np.stack(known_K, axis=0) if all(k is not None for k in known_K) else None
    )
    da3_result = state.da3_estimator.estimate(
        images,
        known_intrinsics=intrinsics_array,
    )

    ref_idx = cam_names.index(clicked_cam)
    da3_depth = da3_result.depth[ref_idx]
    da3_K = da3_result.intrinsics[ref_idx]
    da3_c2w = da3_result.get_c2w(ref_idx)
    da3_h, da3_w = da3_depth.shape

    da3_u = min(max(int(orig_u * da3_w / tile_info["orig_w"]), 0), da3_w - 1)
    da3_v = min(max(int(orig_v * da3_h / tile_info["orig_h"]), 0), da3_h - 1)

    pr = 5
    patch = da3_depth[
        max(0, da3_v - pr) : min(da3_h, da3_v + pr + 1),
        max(0, da3_u - pr) : min(da3_w, da3_u + pr + 1),
    ]
    valid_patch = patch[(patch > 0.01) & (patch < 20)]
    if len(valid_patch) == 0:
        return JSONResponse(status_code=400, content={"error": "No valid depth"})

    depth_val = float(np.median(valid_patch))
    fx, fy = float(da3_K[0, 0]), float(da3_K[1, 1])
    cx, cy = float(da3_K[0, 2]), float(da3_K[1, 2])
    p_cam = np.array(
        [
            (da3_u - cx) * depth_val / fx,
            (da3_v - cy) * depth_val / fy,
            depth_val,
            1.0,
        ]
    )
    p_da3_world = (da3_c2w @ p_cam)[:3]

    offset = tcp_pos - p_da3_world
    T = np.eye(4)
    T[:3, 3] = offset
    state.da3_anchor = T

    applied = []
    cam_extrinsics_save = {}
    for i, name in enumerate(cam_names):
        da3_c2w_i = da3_result.get_c2w(i)
        cam_ext = np.eye(4)
        cam_ext[:3, :3] = da3_c2w_i[:3, :3]
        cam_ext[:3, 3] = (T @ np.append(da3_result.camera_positions[i], 1.0))[:3]
        cal = _cam_cal_map(pipeline).get(name)
        if cal is not None:
            cal.extrinsic = cam_ext
            applied.append(name)
            cam_extrinsics_save[name] = cam_ext.tolist()

    with open(_anchor_path(), "w") as f:
        json.dump(
            {
                "tcp_position": tcp_pos.tolist(),
                "offset": offset.tolist(),
                "transform": T.tolist(),
                "camera_extrinsics": cam_extrinsics_save,
                "robot_family": _active_family(),
            },
            f,
            indent=2,
        )

    return {"success": True, "applied_to": applied}

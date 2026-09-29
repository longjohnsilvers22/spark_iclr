"""
Teleop control routes split out of control.py.

Houses the operator-driven teleop endpoints: relative jog
(/api/move_relative), continuous EMA-smoothed velocity streaming
(/api/velocity), and the wrist-yaw diagnostic sweep (/api/test_yaw).

The EMA-smoothing state and gains (state.smoothed_vel, state.VEL_ALPHA,
state.MAX_DOWN_VEL) stay owned by routes.state where the rest of the
server expects them; this module reads them through the imported state
module. The router defined here is composed onto the top-level control
router via include_router, so server.py stays unchanged.
"""

import time

import numpy as np
from scipy.spatial.transform import Rotation
from fastapi import APIRouter
from fastapi.responses import JSONResponse

from spark_real.routes import state
from spark_real.routes.models import MoveRequest, VelocityRequest

router = APIRouter()


@router.post("/api/move_relative")
async def move_relative(req: MoveRequest):
    """
    Move robot by sending a short velocity command (teleop-style).
    """
    pipeline = state.pipeline
    if pipeline is None or pipeline._robot is None:
        return JSONResponse(status_code=400, content={"error": "Robot not connected"})
    if state.executor_running:
        return {"success": True, "skipped": "executor running"}
    if state.cal_active:
        return {"success": True, "skipped": "calibration active"}
    try:
        speed = 0.08
        vx = speed if req.dx > 0 else (-speed if req.dx < 0 else 0)
        vy = speed if req.dy > 0 else (-speed if req.dy < 0 else 0)
        vz = speed if req.dz > 0 else (-speed if req.dz < 0 else 0)
        try:
            pipeline._robot.send_velocity(
                [vx, vy, vz, 0, 0, 0],
                acceleration=0.3,
                duration=0.3,
            )
        except TypeError:
            pipeline._robot.send_velocity(
                [vx, vy, vz, 0, 0, 0],
                acceleration=0.3,
                time_duration=0.3,
            )
        return {"success": True, "velocity": [vx, vy, vz]}
    except Exception as e:
        return JSONResponse(status_code=500, content={"error": str(e)})


@router.post("/api/velocity")
async def send_velocity(req: VelocityRequest):
    """
    Send EMA-smoothed velocity command for continuous teleop.
    """
    if state.viser_teleop_proc is not None and state.viser_teleop_proc.poll() is None:
        return {"success": True, "skipped": "viser teleop active"}
    pipeline = state.pipeline
    if pipeline is None or pipeline._robot is None:
        return JSONResponse(status_code=400, content={"error": "Robot not connected"})
    if state.executor_running:
        return {"success": True, "skipped": "executor running"}
    if state.cal_active:
        return {"success": True, "skipped": "calibration active"}
    try:
        if state.smoothed_vel is None:
            state.smoothed_vel = np.zeros(6)
        raw = np.array([req.vx, req.vy, req.vz, req.wrx, req.wry, req.wrz])
        if state.MAX_DOWN_VEL is not None and raw[2] < -state.MAX_DOWN_VEL:
            raw[2] = -state.MAX_DOWN_VEL
        state.smoothed_vel = (
            state.VEL_ALPHA * raw + (1 - state.VEL_ALPHA) * state.smoothed_vel
        )
        state.smoothed_vel[np.abs(state.smoothed_vel) < 1e-4] = 0.0
        try:
            pipeline._robot.send_velocity(
                state.smoothed_vel.tolist(),
                acceleration=0.4,
                duration=req.duration,
            )
        except TypeError:
            pipeline._robot.send_velocity(
                state.smoothed_vel.tolist(),
                acceleration=0.4,
                time_duration=req.duration,
            )
        return {"success": True}
    except Exception as e:
        return JSONResponse(status_code=500, content={"error": str(e)})


@router.post("/api/test_yaw")
def test_yaw():
    """
    Diagnostic: cycle through 3 wrist yaw angles.
    """
    pipeline = state.pipeline
    if pipeline is None or pipeline._robot is None:
        return JSONResponse(status_code=400, content={"error": "Robot not connected"})
    try:
        GRASP_ORI = [2.22, -2.22, 0.0]
        base_rot = Rotation.from_rotvec(GRASP_ORI)
        safe_pose = [-0.80, 0.10, 0.10] + GRASP_ORI
        ps = ",".join(f"{v:.4f}" for v in safe_pose)
        pipeline._robot._send_script(f"movel(p[{ps}], a=0.3, v=0.15)")
        time.sleep(5)

        results = []
        for yaw_deg in [0, 30, 60]:
            yaw_rad = np.deg2rad(yaw_deg)
            final_rot = (
                (Rotation.from_euler("z", yaw_rad) * base_rot).as_rotvec().tolist()
            )
            pose = [-0.80, 0.10, 0.10] + final_rot
            ps = ",".join(f"{v:.4f}" for v in pose)
            pipeline._robot._send_script(f"movel(p[{ps}], a=0.3, v=0.10)")
            time.sleep(5)
            raw = np.array(pipeline._robot.get_tcp_pose())
            if raw.shape == (4, 4):
                R = raw[:3, :3]
            else:
                R = Rotation.from_rotvec(raw[3:6]).as_matrix()
            tx = R @ [1, 0, 0]
            results.append(
                {
                    "yaw_deg": yaw_deg,
                    "tool_X_deg": round(float(np.rad2deg(np.arctan2(tx[1], tx[0]))), 1),
                }
            )
        return {"results": results}
    except Exception as e:
        return JSONResponse(status_code=500, content={"error": str(e)})

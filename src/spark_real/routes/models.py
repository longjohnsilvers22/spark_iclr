"""
Pydantic request models for the SPARK server API.
"""

from typing import List, Optional
from pydantic import BaseModel


class DetectRequest(BaseModel):
    prompts: List[str]
    multi_instance: bool = False
    # Optional task string. When it resolves in the prompt registry the task's
    # declared instance ordering and per-instance-Z policy are applied to the
    # merged set, so labels match what a cached behaviour tree expects. An
    # empty prompts list then falls back to the task's registered prompts.
    instruction: Optional[str] = None


class ClickDetectRequest(BaseModel):
    # Non-empty -> persist this annotation for that task so /api/execute
    # can reuse it. Empty/None -> one-off rescue, nothing is stored.
    save_for_task: Optional[str] = None
    x: float
    y: float
    img_width: float
    img_height: float
    label: str = ""


class BoxDetectRequest(BaseModel):
    # Non-empty -> persist this annotation for that task so /api/execute
    # can reuse it. Empty/None -> one-off rescue, nothing is stored.
    save_for_task: Optional[str] = None
    x1: float
    y1: float
    x2: float
    y2: float
    img_width: float
    img_height: float
    label: str = ""


class PlanRequest(BaseModel):
    instruction: str
    prompts: Optional[List[str]] = None
    temperature: Optional[float] = None


class ExecuteRequest(BaseModel):
    instruction: str
    prompts: Optional[List[str]] = None
    dry_run: bool = False
    save_to_library: Optional[bool] = None
    record_video: bool = False
    # Record this run as a VLA training demonstration: synchronized
    # image + proprio + commanded-action frames written in the human teleop
    # schema (see spark_real/recording/). Independent of record_video, which
    # only produces an mp4 for review. Off by default so an ordinary execute
    # never silently appends to the training corpus.
    record_demo: bool = False
    temperature: Optional[float] = None
    # Per-run closed-loop override. None = use config.max_task_passes.
    # >1 re-perceives and re-plans over what remains after each pass until
    # the goal holds (handles dropped/missed objects and scene perturbation).
    closed_loop_passes: Optional[int] = None


class DetectApproveRequest(BaseModel):
    instruction: str
    prompts: Optional[List[str]] = None
    # Low-light test path: when True, swap each Kinect's RGB frame for
    # its active-IR image before feeding to SAM3. The wrist camera (if
    # present) stays on its normal RGB since RealSense doesn't have an
    # IR substitute via this code path.
    use_ir: bool = False


class GripperRequest(BaseModel):
    action: str


class GraspTestRequest(BaseModel):
    """
    Calibrate a width-targeted grasp on a known object.

    Drives a single grasp_to_width(target_width, force) cycle and
    returns the achieved width plus the TCP-force delta. Used by
    scripts/grasp_profile.py to build a per-category calibration
    table that informs the planner prompt defaults.
    """

    target_width: float
    force: float = 20.0
    speed: float = 0.05
    settle_s: float = 1.5
    label: Optional[str] = None
    release_after: bool = True


class GripperPositionRequest(BaseModel):
    # 0.0 = fully open, 1.0 = fully closed. Continuous; analog gamepad
    # triggers stream this at modest rate to drive Franka Hand variably.
    position: float
    speed: Optional[float] = None
    force: Optional[float] = None


class MoveRequest(BaseModel):
    dx: float = 0.0
    dy: float = 0.0
    dz: float = 0.0


class VelocityRequest(BaseModel):
    vx: float = 0.0
    vy: float = 0.0
    vz: float = 0.0
    wrx: float = 0.0
    wry: float = 0.0
    wrz: float = 0.0
    duration: float = 0.25


class ConnectRobotRequest(BaseModel):
    # Empty string = use the currently configured robot_ip on the pipeline
    # (set by server.py from --ip or configs/<family>_default.yaml). Avoids
    # a UR10e default leaking into Franka/G1 flows.
    robot_ip: str = ""


class AnchorRequest(BaseModel):
    x: float
    y: float
    img_width: float
    img_height: float


class SwitchCameraRequest(BaseModel):
    camera: str


class CalibPointRequest(BaseModel):
    robot_x: float
    robot_y: float
    robot_z: float
    camera_x: float
    camera_y: float
    camera_z: float

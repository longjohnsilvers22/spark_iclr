"""
The episode format contract, as constants and dataclasses.

This is the single source of truth for the on-disk demonstration schema.
Everything else in ``spark_real.recording`` imports from here, and the parity
test asserts a freshly written SPARK episode matches a human-teleoperated one
key-for-key and dtype-for-dtype.

On-disk layout (one directory per episode)::

    <data_root>/<task string, verbatim>/episode_NNNN/
        images/camera_0/frame_NNNN.jpg
        images/camera_1/frame_NNNN.jpg
        images/wrist/frame_NNNN.jpg
        trajectory.npz
        metadata.json
        episode_video.mp4        (optional QA artifact)
        trajectory_plot.png      (optional QA artifact)

``trajectory.npz`` carries six required float64 arrays of length N (row ``i``
is time-aligned with ``frame_{i:04d}.jpg`` of every camera):

    timestamps        (N,)   relative seconds, timestamps[0] == 0.0
    tcp_poses         (N,6)  [x,y,z,rx,ry,rz] metres + axis-angle, base frame
    joint_positions   (N,6)  rad
    joint_velocities  (N,6)  rad/s, measured (RTDE actual_qd)
    gripper_positions (N,)   COMMANDED normalized label, 0 = open, 1 = closed
    actions           (N,7)  [vx,vy,vz,wx,wy,wz,gripper]; cols 0-5 are the
                             commanded base-frame TCP twist, col 6 is
                             bit-identical to gripper_positions[i]

plus four SPARK-only extras that both downstream converters ignore (``np.load``
indexes by name, so unknown keys cost nothing):

    wall_time         (N,)   absolute epoch seconds
    tcp_velocity      (N,6)  measured base-frame twist
    wrench            (N,6)  TCP force/torque
    action_source     (N,)   per-step provenance, '<U16'

``action_source`` is mandatory: it is the only way to detect that a run
silently fell off the commanded-velocity path, and the SPARK-vs-human
comparison must be conditionable on it.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

# Required npz keys, in the order the human producer writes them, mapped to
# their row width (1 == a flat (N,) array).
NPZ_CORE_KEYS: Tuple[str, ...] = (
    "timestamps",
    "tcp_poses",
    "joint_positions",
    "joint_velocities",
    "gripper_positions",
    "actions",
)
NPZ_CORE_WIDTHS: Dict[str, int] = {
    "timestamps": 1,
    "tcp_poses": 6,
    "joint_positions": 6,
    "joint_velocities": 6,
    "gripper_positions": 1,
    "actions": 7,
}

# SPARK-only provenance keys. Extra npz keys are safe for both converters.
NPZ_EXTRA_KEYS: Tuple[str, ...] = (
    "wall_time",
    "tcp_velocity",
    "wrench",
    "action_source",
)
NPZ_EXTRA_WIDTHS: Dict[str, int] = {
    "wall_time": 1,
    "tcp_velocity": 6,
    "wrench": 6,
    "action_source": 1,
}

NPZ_DTYPE = np.float64
ACTION_DIM = 7
JOINT_DIM = 6
POSE_DIM = 6

# Per-step action provenance. Only ``servo_cmd`` (PD Cartesian servo) and
# ``velocity_cmd`` (explicit send_velocity / teleop) are genuine commanded
# velocities; the other two are honest post-hoc reconstructions.
ACTION_SOURCE_SERVO = "servo_cmd"
ACTION_SOURCE_VELOCITY = "velocity_cmd"
ACTION_SOURCE_MEASURED = "measured_twist"
ACTION_SOURCE_FINITE_DIFF = "finite_diff"
ACTION_SOURCES: Tuple[str, ...] = (
    ACTION_SOURCE_SERVO,
    ACTION_SOURCE_VELOCITY,
    ACTION_SOURCE_MEASURED,
    ACTION_SOURCE_FINITE_DIFF,
)
ACTION_SOURCE_DTYPE = "<U16"
COMMANDED_SOURCES: Tuple[str, ...] = (ACTION_SOURCE_SERVO, ACTION_SOURCE_VELOCITY)

# Camera slot names, in the order they appear in metadata["cameras"] and
# left-to-right in episode_video.mp4.
CAMERA_SLOTS: Tuple[str, ...] = ("camera_0", "camera_1", "wrist")

# Images: exactly 640x480 for every camera. The non-aspect-preserving stretch
# from 720p is deliberate: the LeRobot converter stretches again to 320x180,
# which restores native 16:9. Preserving aspect here breaks the deploy-time
# geometry match.
IMAGE_WIDTH = 640
IMAGE_HEIGHT = 480
IMAGE_SIZE: Tuple[int, int] = (IMAGE_WIDTH, IMAGE_HEIGHT)  # (w, h), cv2 order
JPEG_QUALITY = 95

FRAME_FMT = "frame_{:04d}.jpg"
EPISODE_FMT = "episode_{:04d}"
IMAGES_DIRNAME = "images"
TRAJECTORY_FILENAME = "trajectory.npz"
METADATA_FILENAME = "metadata.json"
VIDEO_FILENAME = "episode_video.mp4"
PLOT_FILENAME = "trajectory_plot.png"

# metadata.json: exactly these 11 keys, in this order, in every human episode.
METADATA_KEYS: Tuple[str, ...] = (
    "episode_id",
    "task",
    "success",
    "num_frames",
    "start_time",
    "end_time",
    "duration_seconds",
    "robot_ip",
    "cameras",
    "prompt",
    "actual_fps",
)

# SPARK appends these after ``actual_fps``. Both converters read known keys with
# .get() and ignore the rest, so the provenance rides along for free.
SPARK_SOURCE_KEY = "source"
SPARK_PROVENANCE_KEY = "spark"
SPARK_SOURCE_VALUE = "spark"
# Task-success verdict block. Absent on human episodes.
VERIFY_KEY = "verify"

# Recorder modes. "teleop" requires the executor to be idle (the operator
# drives); "autonomous" requires it to be running (SPARK drives).
MODE_TELEOP = "teleop"
MODE_AUTONOMOUS = "autonomous"
MODES: Tuple[str, ...] = (MODE_TELEOP, MODE_AUTONOMOUS)

# Action profiles. Human actions[:,3] (wx) and [:,5] (wz) are identically zero
# across all 60,417 recorded frames and [:,4] (wy) is one-sided negative;
# ``human_parity`` keeps SPARK on that same manifold so the comparison is not
# confounded by action dimensions the human corpus never visited.
PROFILE_HUMAN_PARITY = "human_parity"
PROFILE_NATIVE = "native"
ACTION_PROFILES: Tuple[str, ...] = (PROFILE_HUMAN_PARITY, PROFILE_NATIVE)

# Acceptance band for a matched-set episode.
MIN_ACCEPTABLE_FPS = 14.90
MAX_ACCEPTABLE_FPS = 15.00
MAX_ACCEPTABLE_DT = 0.10


@dataclass
class EpisodeMetadata:
    """
    The 11 human metadata keys, in order, plus SPARK's two extras.

    ``to_dict`` emits the human keys first and in the human order, so a
    byte-level diff against a teleop episode shows only value changes.
    """

    episode_id: int
    task: str
    success: bool
    num_frames: int
    start_time: str
    end_time: str
    duration_seconds: float
    robot_ip: str
    cameras: List[str]
    prompt: str
    actual_fps: float

    # SPARK-only provenance, written after actual_fps. Left None for a
    # recording that wants to be indistinguishable from a human episode.
    source: Optional[str] = None
    spark: Optional[Dict[str, Any]] = field(default=None)
    # Task-success verdict (control/success_verifier.VerifyOutcome.to_dict()).
    # ``success`` above is exactly ``verify["status"] == "pass"``. Dataset
    # consumers must filter positives on the STATUS and drop "unverified" from
    # both the positive and the negative set -- an unverified episode is not a
    # quiet failure, it is an episode nobody checked.
    verify: Optional[Dict[str, Any]] = field(default=None)

    def to_dict(self) -> Dict[str, Any]:
        out: Dict[str, Any] = {
            "episode_id": int(self.episode_id),
            "task": str(self.task),
            "success": bool(self.success),
            "num_frames": int(self.num_frames),
            "start_time": str(self.start_time),
            "end_time": str(self.end_time),
            "duration_seconds": float(self.duration_seconds),
            "robot_ip": str(self.robot_ip),
            "cameras": list(self.cameras),
            "prompt": str(self.prompt),
            "actual_fps": float(self.actual_fps),
        }
        if self.source is not None:
            out[SPARK_SOURCE_KEY] = self.source
        if self.spark is not None:
            out[SPARK_PROVENANCE_KEY] = self.spark
        if self.verify is not None:
            out[VERIFY_KEY] = self.verify
        return out

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "EpisodeMetadata":
        return cls(
            episode_id=int(data.get("episode_id", 0)),
            task=str(data.get("task", "")),
            success=bool(data.get("success", False)),
            num_frames=int(data.get("num_frames", 0)),
            start_time=str(data.get("start_time", "")),
            end_time=str(data.get("end_time", "")),
            duration_seconds=float(data.get("duration_seconds", 0.0)),
            robot_ip=str(data.get("robot_ip", "")),
            cameras=list(data.get("cameras", [])),
            prompt=str(data.get("prompt", "")),
            actual_fps=float(data.get("actual_fps", 0.0)),
            source=data.get(SPARK_SOURCE_KEY),
            spark=data.get(SPARK_PROVENANCE_KEY),
            verify=data.get(VERIFY_KEY),
        )


def actual_fps(timestamps) -> float:
    """
    Frames-per-second exactly as the human producer computes it.

    Derived from the relative timestamps, NOT from ``duration_seconds`` (which
    is wall clock including save overhead and runs ~3% long).
    """
    ts = np.asarray(timestamps, dtype=np.float64).ravel()
    if ts.size < 2 or not (ts[-1] > ts[0]):
        return 0.0
    return round(float((ts.size - 1) / (ts[-1] - ts[0])), 2)


__all__ = [
    "ACTION_DIM",
    "ACTION_PROFILES",
    "ACTION_SOURCES",
    "ACTION_SOURCE_DTYPE",
    "ACTION_SOURCE_FINITE_DIFF",
    "ACTION_SOURCE_MEASURED",
    "ACTION_SOURCE_SERVO",
    "ACTION_SOURCE_VELOCITY",
    "CAMERA_SLOTS",
    "COMMANDED_SOURCES",
    "EPISODE_FMT",
    "EpisodeMetadata",
    "FRAME_FMT",
    "IMAGES_DIRNAME",
    "IMAGE_HEIGHT",
    "IMAGE_SIZE",
    "IMAGE_WIDTH",
    "JOINT_DIM",
    "JPEG_QUALITY",
    "MAX_ACCEPTABLE_DT",
    "MAX_ACCEPTABLE_FPS",
    "METADATA_FILENAME",
    "METADATA_KEYS",
    "MIN_ACCEPTABLE_FPS",
    "MODES",
    "MODE_AUTONOMOUS",
    "MODE_TELEOP",
    "NPZ_CORE_KEYS",
    "NPZ_CORE_WIDTHS",
    "NPZ_DTYPE",
    "NPZ_EXTRA_KEYS",
    "NPZ_EXTRA_WIDTHS",
    "PLOT_FILENAME",
    "POSE_DIM",
    "PROFILE_HUMAN_PARITY",
    "PROFILE_NATIVE",
    "SPARK_PROVENANCE_KEY",
    "SPARK_SOURCE_KEY",
    "SPARK_SOURCE_VALUE",
    "TRAJECTORY_FILENAME",
    "VIDEO_FILENAME",
    "actual_fps",
]

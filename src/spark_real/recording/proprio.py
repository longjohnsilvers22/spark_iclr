"""
One proprioceptive sample from a robot driver.

The recorder thread only sequences work; the driver-surface knowledge
lives here. Everything is
best-effort: a driver call that raises is logged once and its channel comes
back missing, which the writer pads with NaN. A demonstration must never be
aborted by a transient RTDE hiccup.

The keys are the schema's names, not DROID's, so the row can be appended
straight into the npz columns:

    tcp_pose         (6,)  [x,y,z,rx,ry,rz], metres + axis-angle, base frame
    joint_positions  (6,)  rad          (RTDE actual_q)
    joint_velocities (6,)  rad/s        (RTDE actual_qd, measured)
    tcp_velocity     (6,)  measured base-frame twist (SPARK extra)
    wrench           (6,)  fx..fz tx..tz (SPARK extra)
    gripper_measured  ()   normalized 0..1, MEASURED

``gripper_measured`` is deliberately NOT the training label. The Robotiq read
is documented-stale mid-motion; the label comes from the command latch (see
``action_source``). It is kept only as a diagnostic channel.
"""

from __future__ import annotations

import logging
from typing import Any, Dict, Optional

import numpy as np

from spark_real.recording.schema import JOINT_DIM, POSE_DIM

logger = logging.getLogger(__name__)

# Robotiq raw register scale (0 open .. 255 closed).
GRIPPER_RAW_MAX = 255.0


def normalize_gripper(raw: Any) -> float:
    """Robotiq position -> 0..1 (0 open, 1 closed). NaN when unreadable."""
    try:
        v = float(raw)
    except (TypeError, ValueError):
        return float("nan")
    if not np.isfinite(v):
        return float("nan")
    if v > 1.0:
        v = v / GRIPPER_RAW_MAX
    return float(np.clip(v, 0.0, 1.0))


def _passive(fn: Any) -> Any:
    """
    Call a driver getter, asking it not to actuate the robot to answer.

    This reader runs on a background thread CONCURRENTLY with motion. On UR10e
    a publishing gripper read uploads a URScript program, which replaces the
    running movej/speedl and stops the arm. Drivers that understand
    ``publish=False`` get the non-invasive path; the rest are called plainly.
    """
    try:
        return fn(publish=False)
    except TypeError:
        return fn()


def _vec(value: Any, width: int) -> Optional[list]:
    try:
        arr = np.asarray(value, dtype=np.float64).ravel()
    except (TypeError, ValueError):
        return None
    if arr.size == 0:
        return None
    return list(arr[:width])


class ProprioReader:
    """
    Reads one aligned proprio row per call. Warnings are emitted once each.
    """

    def __init__(self, robot: Any):
        self.robot = robot
        self._warned: Dict[str, bool] = {}

    def _warn(self, key: str, exc: Exception) -> None:
        if not self._warned.get(key):
            logger.warning("ProprioReader: %s failed: %s", key, exc)
            self._warned[key] = True

    def read(self) -> Dict[str, Any]:
        """
        One sample. Missing channels are simply absent from the dict.

        ``get_observation()`` is preferred because it is a single RTDE
        round-trip for four channels; the per-getter fallbacks below cover
        drivers that do not implement it.
        """
        out: Dict[str, Any] = {}
        robot = self.robot
        if robot is None:
            return out

        obs = None
        getter = getattr(robot, "get_observation", None)
        if callable(getter):
            try:
                obs = getter()
            except Exception as exc:
                self._warn("get_observation", exc)
        if isinstance(obs, dict):
            for src, dst, width in (
                ("tcp_pose", "tcp_pose", POSE_DIM),
                ("joint_positions", "joint_positions", JOINT_DIM),
                ("joint_velocities", "joint_velocities", JOINT_DIM),
                ("tcp_velocity", "tcp_velocity", POSE_DIM),
            ):
                val = _vec(obs.get(src), width)
                if val is not None:
                    out[dst] = val
            if obs.get("gripper_position") is not None:
                out["gripper_measured"] = normalize_gripper(obs["gripper_position"])

        for key, attr, width in (
            ("tcp_pose", "get_tcp_pose", POSE_DIM),
            ("joint_positions", "get_joint_positions", JOINT_DIM),
            ("joint_velocities", "get_joint_velocities", JOINT_DIM),
            ("wrench", "get_tcp_force", POSE_DIM),
        ):
            if key in out:
                continue
            fn = getattr(robot, attr, None)
            if not callable(fn):
                continue
            try:
                val = _vec(fn(), width)
            except Exception as exc:
                self._warn(attr, exc)
                continue
            if val is not None:
                out[key] = val

        if "gripper_measured" not in out:
            fn = getattr(robot, "get_gripper_position", None)
            if callable(fn):
                try:
                    out["gripper_measured"] = normalize_gripper(_passive(fn))
                except Exception as exc:
                    self._warn("get_gripper_position", exc)
        return out


__all__ = ["GRIPPER_RAW_MAX", "ProprioReader", "normalize_gripper"]

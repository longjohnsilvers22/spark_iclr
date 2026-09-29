"""
Derivation of the per-step action label.

The human corpus records ``actions[:, 0:6]`` as the *commanded* base-frame TCP
twist handed to ``speedl`` on that tick, and ``actions[:, 6]`` as the
*commanded* normalized gripper target. Nothing measured goes in there. On the
autonomous SPARK path the same signal exists only where a controller actually
issues per-timestep velocities, so this module resolves, in order:

1. the driver's command latch (``robot.get_last_command()``), if the stamp is
   fresher than ``2 / record_hz``: ``servo_cmd`` or ``velocity_cmd``;
2. the measured TCP twist (``getActualTCPSpeed`` via the proprio sample):
   ``measured_twist``;
3. a finite difference of consecutive TCP poses: ``finite_diff``, the last
   resort when a driver reports no twist at all.

Whichever it resolves, the value is then put through the *same* post-processing
the human teleop stack applied to its own stick input, so the two action
distributions are comparable: per-axis clip, EMA smoothing, and a deadband that
snaps near-zero to exact zero. Under ``action_profile: human_parity`` the wx
and wz channels are additionally zeroed, because they are identically zero in
all 60,417 recorded human frames.

The resolved provenance is written to the ``action_source`` npz column. It is
mandatory: an episode that silently fell back to ``measured_twist`` has a
different action distribution and the analysis has to be able to condition on
that.
"""

from __future__ import annotations

import logging
import time
from typing import Any, Optional, Sequence, Tuple

import numpy as np
from scipy.spatial.transform import Rotation

from spark_real.recording.schema import (
    ACTION_DIM,
    ACTION_SOURCE_FINITE_DIFF,
    ACTION_SOURCE_MEASURED,
    ACTION_SOURCES,
    ACTION_SOURCE_VELOCITY,
    POSE_DIM,
    PROFILE_HUMAN_PARITY,
)
from spark_real.recording.settings import RecordingSettings

logger = logging.getLogger(__name__)

# Indices of the rotation channels the human corpus never exercised.
_HUMAN_ZERO_ROT_AXES: Tuple[int, ...] = (3, 5)


def _as6(vec: Optional[Sequence[float]]) -> Optional[np.ndarray]:
    """A 6-vector of float64, or None if the input is unusable."""
    if vec is None:
        return None
    try:
        v = np.asarray(vec, dtype=np.float64).ravel()
    except (TypeError, ValueError):
        return None
    if v.size < POSE_DIM:
        v = np.pad(v, (0, POSE_DIM - v.size))
    v = v[:POSE_DIM]
    if not np.all(np.isfinite(v)):
        return None
    return v


def finite_difference_twist(
    pose: Optional[Sequence[float]],
    prev_pose: Optional[Sequence[float]],
    dt: float,
) -> Optional[np.ndarray]:
    """
    Base-frame twist from two consecutive TCP poses.

    ``v_lin = dp/dt``; ``v_ang = rotvec(R_t @ R_{t-1}^T) / dt``; the rotation
    is composed in the base frame (left multiplication), matching the frame
    ``speedl`` interprets its angular channels in.
    """
    p1 = _as6(pose)
    p0 = _as6(prev_pose)
    if p1 is None or p0 is None or dt <= 0.0:
        return None
    lin = (p1[:3] - p0[:3]) / dt
    r1 = Rotation.from_rotvec(p1[3:6])
    r0 = Rotation.from_rotvec(p0[3:6])
    ang = (r1 * r0.inv()).as_rotvec() / dt
    return np.concatenate([lin, ang])


class ActionSampler:
    """
    Stateful per-episode action derivation. One instance, one episode.

    ``sample()`` is called once per recorded frame, after the proprio read for
    that frame, and returns ``(action7, source)`` ready to append.
    """

    def __init__(self, settings: RecordingSettings, robot: Any = None):
        self.settings = settings
        self.robot = robot
        self.dt = 1.0 / max(1e-3, float(settings.record_hz))
        # A latch older than two frame periods is not "the command for this
        # tick": a movej segment leaves the last servo command sitting there.
        self.max_latch_age = 2.0 * self.dt
        self._ema: Optional[np.ndarray] = None
        self._prev_pose: Optional[np.ndarray] = None
        self._gripper: float = 0.0
        self._warned = False
        self.counts = {src: 0 for src in ACTION_SOURCES}

    # latch access

    def _latched(self) -> Tuple[Optional[np.ndarray], Optional[str], Optional[float]]:
        """``(velocity, source, gripper)`` from the driver latch, all optional."""
        robot = self.robot
        if robot is None:
            return None, None, None
        getter = getattr(robot, "get_last_command", None)
        if not callable(getter):
            return None, None, None
        try:
            snap = getter()
        except Exception as exc:
            if not self._warned:
                logger.warning("ActionSampler: get_last_command failed: %s", exc)
                self._warned = True
            return None, None, None
        if snap is None:
            return None, None, None
        now = time.monotonic()
        vel = None
        source = None
        if snap.velocity is not None and (now - snap.velocity_time) < self.max_latch_age:
            vel = _as6(snap.velocity)
            source = snap.velocity_source or ACTION_SOURCE_VELOCITY
            if source not in ACTION_SOURCES:
                source = ACTION_SOURCE_VELOCITY
        return vel, source, snap.gripper

    # gripper

    def _gripper_label(self, latched: Optional[float]) -> float:
        """
        Commanded gripper label, ramped when ``demo_mode`` asks for it.

        A SPARK grasp commands 1.0 in one tick where a human produced a
        ~15-frame ramp; ``gripper_ramp_rate`` (units/s) bridges that so the
        label distributions are comparable. Rate <= 0 disables the ramp.
        """
        target = self._gripper if latched is None else float(np.clip(latched, 0.0, 1.0))
        rate = float(self.settings.gripper_ramp_rate)
        if not self.settings.demo_mode or rate <= 0.0:
            self._gripper = target
            return self._gripper
        step = rate * self.dt
        delta = target - self._gripper
        if abs(delta) <= step:
            self._gripper = target
        else:
            self._gripper += step * np.sign(delta)
        return float(np.clip(self._gripper, 0.0, 1.0))

    # shaping

    def _shape(self, vel: np.ndarray) -> np.ndarray:
        """Profile zeroing, clip, EMA, deadband, in the human producer's order."""
        v = np.asarray(vel, dtype=np.float64).copy()
        if self.settings.action_profile == PROFILE_HUMAN_PARITY:
            for axis in _HUMAN_ZERO_ROT_AXES:
                v[axis] = 0.0
        lin_clip = abs(float(self.settings.linear_velocity_clip))
        ang_clip = abs(float(self.settings.angular_velocity_clip))
        v[:3] = np.clip(v[:3], -lin_clip, lin_clip)
        v[3:] = np.clip(v[3:], -ang_clip, ang_clip)

        alpha = float(self.settings.action_ema_alpha)
        if 0.0 < alpha < 1.0:
            if self._ema is None:
                self._ema = v.copy()
            else:
                self._ema = alpha * v + (1.0 - alpha) * self._ema
            v = self._ema.copy()
        else:
            self._ema = v.copy()

        deadband = abs(float(self.settings.action_deadband))
        if deadband > 0.0:
            v[np.abs(v) < deadband] = 0.0
        return v

    # public

    def sample(self, proprio: dict) -> Tuple[np.ndarray, str]:
        """
        One ``(action7, source)`` for the frame described by ``proprio``.

        ``proprio`` is a row from :mod:`spark_real.recording.proprio`; the
        measured twist and TCP pose come from there rather than being re-read,
        so the action is aligned with the state row it is stored beside.
        """
        vel, source, latched_grip = self._latched()
        if vel is None:
            vel = _as6(proprio.get("tcp_velocity"))
            source = ACTION_SOURCE_MEASURED if vel is not None else None
        pose = _as6(proprio.get("tcp_pose"))
        if vel is None:
            vel = finite_difference_twist(pose, self._prev_pose, self.dt)
            source = ACTION_SOURCE_FINITE_DIFF if vel is not None else None
        if vel is None:
            vel = np.zeros(POSE_DIM, dtype=np.float64)
            source = ACTION_SOURCE_FINITE_DIFF
        self._prev_pose = pose

        action = np.zeros(ACTION_DIM, dtype=np.float64)
        action[:POSE_DIM] = self._shape(vel)
        action[6] = self._gripper_label(latched_grip)
        self.counts[source] = self.counts.get(source, 0) + 1
        return action, source

    def commanded_fraction(self) -> float:
        """Share of frames whose action came from a genuine command latch."""
        total = sum(self.counts.values())
        if not total:
            return 0.0
        commanded = self.counts.get("servo_cmd", 0) + self.counts.get("velocity_cmd", 0)
        return commanded / float(total)


__all__ = ["ActionSampler", "finite_difference_twist"]

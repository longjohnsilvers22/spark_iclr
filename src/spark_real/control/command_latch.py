"""
Last-commanded-motion latch, shared by every driver.

Each driver stamps its :class:`CommandLatch` inside its velocity/gripper entry
point; the recorder reads ``robot.get_last_command()`` once per frame and
applies its own staleness gate (``recording/action_source.py``).

The caller declares the provenance channel via the thread-local
:func:`command_source` context manager:

    with command_source(ACTION_SOURCE_SERVO):
        ...                       # CartesianServo's PD loop
    robot.send_velocity(v)        # anything else -> velocity_cmd

This module imports nothing from spark_real; ``_DEFAULT_SOURCE`` restates
``recording.schema.ACTION_SOURCE_VELOCITY`` and the parity test asserts the
two stay equal.
"""

from __future__ import annotations

import threading
import time
from contextlib import contextmanager
from dataclasses import dataclass
from typing import Iterator, Optional, Sequence

import numpy as np

# Mirrors of spark_real.recording.schema.ACTION_SOURCE_*.
# tests/test_episode_parity.py asserts these stay equal to the schema values.
SOURCE_VELOCITY = "velocity_cmd"
SOURCE_SERVO = "servo_cmd"

_DEFAULT_SOURCE = SOURCE_VELOCITY

_local = threading.local()


def current_source() -> str:
    """Provenance tag for a command issued from this thread right now."""
    return getattr(_local, "source", _DEFAULT_SOURCE)


@contextmanager
def command_source(name: str) -> Iterator[str]:
    """Tag every velocity command issued from this thread inside the block."""
    previous = getattr(_local, "source", _DEFAULT_SOURCE)
    _local.source = str(name)
    try:
        yield _local.source
    finally:
        _local.source = previous


@dataclass
class CommandSnapshot:
    """What the driver last commanded, with monotonic stamps."""

    velocity: Optional[np.ndarray] = None
    velocity_source: str = _DEFAULT_SOURCE
    velocity_time: float = -1.0
    gripper: Optional[float] = None
    gripper_time: float = -1.0


class CommandLatch:
    """Thread-safe holder for the most recent commanded velocity / gripper."""

    def __init__(self) -> None:
        self._lock = threading.Lock()
        self._snapshot = CommandSnapshot()

    def latch_velocity(self, velocity: Sequence[float], source: Optional[str] = None) -> None:
        """Record a commanded base-frame twist. Never raises."""
        try:
            v = np.asarray(velocity, dtype=np.float64).ravel()[:6]
            if v.size < 6:
                v = np.pad(v, (0, 6 - v.size))
        except (TypeError, ValueError):
            return
        with self._lock:
            self._snapshot.velocity = v
            self._snapshot.velocity_source = source or current_source()
            self._snapshot.velocity_time = time.monotonic()

    def latch_gripper(self, position: float) -> None:
        """
        Record the COMMANDED normalized gripper target (0 open .. 1 closed).

        The measured Robotiq aperture is stale mid-motion and must never be
        substituted here.
        """
        try:
            g = float(np.clip(float(position), 0.0, 1.0))
        except (TypeError, ValueError):
            return
        with self._lock:
            self._snapshot.gripper = g
            self._snapshot.gripper_time = time.monotonic()

    def snapshot(self) -> CommandSnapshot:
        """Immutable copy of the current latch state."""
        with self._lock:
            s = self._snapshot
            return CommandSnapshot(
                velocity=None if s.velocity is None else s.velocity.copy(),
                velocity_source=s.velocity_source,
                velocity_time=s.velocity_time,
                gripper=s.gripper,
                gripper_time=s.gripper_time,
            )


__all__ = [
    "CommandLatch",
    "CommandSnapshot",
    "SOURCE_SERVO",
    "SOURCE_VELOCITY",
    "command_source",
    "current_source",
]

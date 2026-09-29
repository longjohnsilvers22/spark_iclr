"""
Recording settings resolved off the pipeline config.

The knobs live in the ``recording:`` block of
``configs/<family>_default.yaml``; this module is the one place that reads
them, so no other recording module carries a literal.

Resolution order for every field, first hit wins:

1. ``config.recording``: a nested dict or object (``recording.record_hz``)
2. ``config.recording_<field>``: a flat prefixed attribute
3. the default below

That tolerance is deliberate: it lets the recorder work against whichever
shape the config loader ends up exposing, without a second edit here.
``config`` may also be the merged yaml dict itself (``load_family_yaml``),
which is what an offline training script has in hand.
"""

from __future__ import annotations

import os
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

from spark_real.recording.schema import (
    CAMERA_SLOTS,
    IMAGE_HEIGHT,
    IMAGE_WIDTH,
    JPEG_QUALITY,
    PROFILE_NATIVE,
)

# Where episodes land when the config says nothing. Deliberately NOT a rig
# path: a clone with no writable system-level data volume must still record.
# A deployment's real root is a value in `recording.data_dir` in
# configs/<family>_default.yaml, and $SPARK_EPISODES overrides the fallback
# for a one-off session.
DATA_DIR_ENV = "SPARK_EPISODES"
FALLBACK_DATA_DIR = "~/spark_episodes"


def default_data_dir() -> str:
    """``$SPARK_EPISODES`` if set, else ``~/spark_episodes`` (expanded later)."""
    return os.environ.get(DATA_DIR_ENV) or FALLBACK_DATA_DIR


DEFAULT_RECORD_HZ = 15.0
DEFAULT_ACTION_EMA_ALPHA = 0.4
DEFAULT_ACTION_DEADBAND = 1.0e-4
DEFAULT_LINEAR_VELOCITY_CLIP = 0.2
DEFAULT_ANGULAR_VELOCITY_CLIP = 0.3
DEFAULT_GRIPPER_RAMP_RATE = 1.0
DEFAULT_MAX_BLACK_FRAMES = 0

# What to do when a slot listed in ``required_cameras`` produced no frames at
# all (typically: the sideview Kinect was unplugged). ``fill`` writes a black
# JPEG directory of the right length (DROID's convention for an absent third
# camera) so the training converter, which hard-requires all three camera
# directories and SILENTLY SKIPS an episode missing one, still ingests it.
# ``fail`` leaves the episode as-is and marks it unsuccessful instead.
POLICY_FILL = "fill"
POLICY_FAIL = "fail"
MISSING_CAMERA_POLICIES: List[str] = [POLICY_FILL, POLICY_FAIL]
DEFAULT_MISSING_CAMERA_POLICY = POLICY_FILL

# SPARK camera key -> human episode slot. sideview (master Kinect) is the
# oblique end-on view the converter maps to the masked secondary DROID slot;
# birdview (subordinate) is the good overhead/frontal view and becomes the
# primary. Overridable via recording.camera_map.
DEFAULT_CAMERA_MAP: Dict[str, str] = {
    "sideview": "camera_0",
    "birdview": "camera_1",
    "wrist": "wrist",
}

# Slots a VLA actually trains on. Perception is free to use BOTH Kinects
# (sideview carries the better grasp-depth signal) but the manipulation
# observation streams are birdview + wrist only. Every slot is still written
# to disk; this only narrows what the loader hands to training.
DEFAULT_TRAIN_CAMERAS: List[str] = ["camera_1", "wrist"]


def _lookup(config: Any, name: str) -> Any:
    """Find one setting on ``config``; see the module docstring for order."""
    if config is None:
        return None
    if isinstance(config, dict):
        # A raw merged yaml (config.load_family_yaml) rather than a
        # PipelineConfig: the same three shapes, read off keys.
        block = config.get("recording")
        if isinstance(block, dict) and block.get(name) is not None:
            return block[name]
        if config.get(f"recording_{name}") is not None:
            return config[f"recording_{name}"]
        return None
    block = getattr(config, "recording", None)
    if isinstance(block, dict):
        if block.get(name) is not None:
            return block[name]
    elif block is not None:
        val = getattr(block, name, None)
        if val is not None:
            return val
    return getattr(config, f"recording_{name}", None)


@dataclass
class RecordingSettings:
    """Everything the recorder needs, with no config object in tow."""

    data_dir: str = field(default_factory=default_data_dir)
    record_hz: float = DEFAULT_RECORD_HZ
    image_size: List[int] = field(default_factory=lambda: [IMAGE_WIDTH, IMAGE_HEIGHT])
    jpeg_quality: int = JPEG_QUALITY
    camera_map: Dict[str, str] = field(default_factory=lambda: dict(DEFAULT_CAMERA_MAP))
    train_cameras: List[str] = field(default_factory=lambda: list(DEFAULT_TRAIN_CAMERAS))
    # Slots the episode must contain on disk to be convertible. None => every
    # slot in camera_map. A required slot that produced no frames is handled
    # by missing_camera_policy, never left silently absent.
    required_cameras: Optional[List[str]] = None
    missing_camera_policy: str = DEFAULT_MISSING_CAMERA_POLICY
    demo_mode: bool = False
    action_profile: str = PROFILE_NATIVE
    action_ema_alpha: float = DEFAULT_ACTION_EMA_ALPHA
    action_deadband: float = DEFAULT_ACTION_DEADBAND
    linear_velocity_clip: float = DEFAULT_LINEAR_VELOCITY_CLIP
    angular_velocity_clip: float = DEFAULT_ANGULAR_VELOCITY_CLIP
    gripper_ramp_rate: float = DEFAULT_GRIPPER_RAMP_RATE
    max_black_frames: int = DEFAULT_MAX_BLACK_FRAMES
    save_depth: bool = False
    emit_video: bool = True
    emit_plot: bool = True
    robot_ip: str = ""

    @property
    def slots(self) -> List[str]:
        """Human camera slots in metadata order, restricted to what is mapped."""
        mapped = set(self.camera_map.values())
        return [s for s in CAMERA_SLOTS if s in mapped]

    @property
    def required_slots(self) -> List[str]:
        """Slots every episode must carry, in metadata order."""
        if self.required_cameras is None:
            return self.slots
        wanted = {str(s) for s in self.required_cameras}
        ordered = [s for s in CAMERA_SLOTS if s in wanted]
        ordered += [s for s in self.required_cameras if s not in CAMERA_SLOTS]
        return ordered

    @property
    def fill_missing_cameras(self) -> bool:
        """True when an absent required slot is black-filled rather than failed."""
        return str(self.missing_camera_policy).lower() != POLICY_FAIL

    @classmethod
    def from_config(cls, config: Any) -> "RecordingSettings":
        s = cls()
        for name in (
            "data_dir",
            "action_profile",
            "missing_camera_policy",
        ):
            val = _lookup(config, name)
            if val is not None:
                setattr(s, name, str(val))
        for name in (
            "record_hz",
            "action_ema_alpha",
            "action_deadband",
            "linear_velocity_clip",
            "angular_velocity_clip",
            "gripper_ramp_rate",
        ):
            val = _lookup(config, name)
            if val is not None:
                setattr(s, name, float(val))
        for name in ("jpeg_quality", "max_black_frames"):
            val = _lookup(config, name)
            if val is not None:
                setattr(s, name, int(val))
        for name in ("demo_mode", "save_depth", "emit_video", "emit_plot"):
            val = _lookup(config, name)
            if val is not None:
                setattr(s, name, bool(val))
        size = _lookup(config, "image_size")
        if size is not None:
            dims = [int(v) for v in size]
            if len(dims) == 2:
                s.image_size = dims
        cam_map = _lookup(config, "camera_map")
        if isinstance(cam_map, dict) and cam_map:
            s.camera_map = {str(k): str(v) for k, v in cam_map.items()}
        train = _lookup(config, "train_cameras")
        if train:
            s.train_cameras = [str(v) for v in train]
        required = _lookup(config, "required_cameras")
        if required:
            s.required_cameras = [str(v) for v in required]
        # robot_ip is a top-level config field, never a literal here.
        if isinstance(config, dict):
            ip = (config.get("robot") or {}).get("ip") if config else None
        else:
            ip = getattr(config, "robot_ip", None) if config is not None else None
        if ip:
            s.robot_ip = str(ip)
        return s


# The settings this process last resolved from its config. The dataset loader
# reads ``train_cameras`` off it so a configured value is honoured by callers
# that never see the config object (see vla_dataset._default_train_cameras);
# without this the documented key would be a silent no-op for them.
_ACTIVE: Optional[RecordingSettings] = None


def active_settings() -> Optional[RecordingSettings]:
    """The last settings ``resolve_settings`` produced, or None."""
    return _ACTIVE


def set_active_settings(settings: Optional[RecordingSettings]) -> None:
    """Publish (or clear, with None) the process-wide resolved settings."""
    global _ACTIVE
    _ACTIVE = settings


def resolve_settings(config: Any, overrides: Optional[Dict[str, Any]] = None):
    """
    Settings from ``config``, with an optional per-request override dict
    (the API route passes e.g. ``{"demo_mode": True}``).
    """
    s = RecordingSettings.from_config(config)
    for key, val in (overrides or {}).items():
        if val is None or not hasattr(s, key):
            continue
        setattr(s, key, val)
    set_active_settings(s)
    return s


__all__ = [
    "DATA_DIR_ENV",
    "DEFAULT_CAMERA_MAP",
    "DEFAULT_MISSING_CAMERA_POLICY",
    "DEFAULT_TRAIN_CAMERAS",
    "FALLBACK_DATA_DIR",
    "MISSING_CAMERA_POLICIES",
    "POLICY_FAIL",
    "POLICY_FILL",
    "RecordingSettings",
    "active_settings",
    "default_data_dir",
    "resolve_settings",
    "set_active_settings",
]

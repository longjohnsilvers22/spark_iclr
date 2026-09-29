"""
Demonstration recording for spark_real.

Writes and reads episodes in the *human teleop schema* (the exact layout of
``/data/teleop_episodes``) so a VLA trained on autonomous SPARK runs and one
trained on human teleoperation differ only in who drove the arm.

    schema         the format contract (constants + EpisodeMetadata)
    naming         task / episode directory allocation
    settings       the ``recording:`` config block
    proprio        one robot state sample
    action_source  the commanded-action label and its provenance
    writer         frames, trajectory.npz, metadata.json, QA artifacts
    vla_recorder   DemoRecorder: the 15 Hz loop
    vla_dataset    Episode / load_episode: the reader, human + SPARK alike
"""

from spark_real.recording.action_source import ActionSampler
from spark_real.recording.naming import (
    episode_dir,
    frame_path,
    images_dir,
    iter_episode_dirs,
    next_episode_number,
    task_dir,
)
from spark_real.recording.proprio import ProprioReader
from spark_real.recording.schema import (
    CAMERA_SLOTS,
    METADATA_KEYS,
    NPZ_CORE_KEYS,
    EpisodeMetadata,
    actual_fps,
)
from spark_real.recording.settings import (
    DEFAULT_CAMERA_MAP,
    DEFAULT_TRAIN_CAMERAS,
    RecordingSettings,
    resolve_settings,
)
from spark_real.recording.vla_dataset import (
    ALL_CAMERAS,
    Episode,
    iter_episodes,
    load_dataset,
    load_episode,
    validate_episode,
)
from spark_real.recording.vla_recorder import DemoRecorder
from spark_real.recording.writer import TrajectoryBuffer

__all__ = [
    "ALL_CAMERAS",
    "ActionSampler",
    "CAMERA_SLOTS",
    "DEFAULT_CAMERA_MAP",
    "DEFAULT_TRAIN_CAMERAS",
    "DemoRecorder",
    "Episode",
    "EpisodeMetadata",
    "METADATA_KEYS",
    "NPZ_CORE_KEYS",
    "ProprioReader",
    "RecordingSettings",
    "TrajectoryBuffer",
    "actual_fps",
    "episode_dir",
    "frame_path",
    "images_dir",
    "iter_episode_dirs",
    "iter_episodes",
    "load_dataset",
    "load_episode",
    "next_episode_number",
    "resolve_settings",
    "task_dir",
    "validate_episode",
]

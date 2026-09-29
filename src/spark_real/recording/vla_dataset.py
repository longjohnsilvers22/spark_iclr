"""
Reader for demonstration episodes: human teleop and SPARK, one code path.

Both corpora are on-disk identical (SPARK's extra npz columns and its nested
``spark`` metadata block are additive), so nothing here branches on producer.
That is the point: a VLA trained on human data and one trained on SPARK data
must see byte-for-byte the same loader, or the comparison has a second
variable in it.

    <root>/<task string>/episode_NNNN/
        images/camera_0/frame_NNNN.jpg   sideview   (may be absent)
        images/camera_1/frame_NNNN.jpg   birdview
        images/wrist/frame_NNNN.jpg      wrist
        trajectory.npz
        metadata.json

Camera subset
-------------
Perception may use both Kinects, but the observation streams a manipulation
policy trains on are birdview + wrist. ``load_episode`` therefore defaults to
``RecordingSettings.train_cameras`` (``["camera_1", "wrist"]`` out of the box,
whatever the config says in practice) and only widens when asked::

    load_episode(path)                     # bird + wrist  (training)
    load_episode(path, cameras="all")      # every slot the episode has
    load_episode(path, settings=settings)  # this deployment's train_cameras

``metadata["cameras"]`` is the authority on what an episode contains, not a
directory glob, and a requested slot the episode does not have is skipped,
never an error. An episode without ``camera_0`` is valid here; note that the
recorder still black-fills it on disk, because the training converter is
stricter than this loader and drops an episode missing any camera directory.

Only numpy, OpenCV and the stdlib are needed; no tensorflow-datasets, no
lerobot.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Iterator, List, Optional, Sequence, Union

import cv2
import numpy as np

from spark_real.recording.naming import frame_path, images_dir, iter_episode_dirs
from spark_real.recording.schema import (
    CAMERA_SLOTS,
    IMAGES_DIRNAME,
    METADATA_FILENAME,
    NPZ_CORE_KEYS,
    SPARK_SOURCE_VALUE,
    TRAJECTORY_FILENAME,
    EpisodeMetadata,
)
from spark_real.recording.settings import (
    DEFAULT_TRAIN_CAMERAS,
    RecordingSettings,
    active_settings,
)

# Sentinel accepted by ``cameras=`` meaning "every slot the episode has".
ALL_CAMERAS = "all"

CameraSpec = Union[None, str, Sequence[str]]


def default_train_cameras(settings: Optional[RecordingSettings] = None) -> List[str]:
    """
    The configured training subset: ``recording.train_cameras``.

    ``settings`` wins; failing that, whatever this process last resolved from
    its config (``resolve_settings``); only then the built-in default. Reading
    the module constant directly would make the shipped config key a silent
    no-op, and bird+wrist selection IS that key.
    """
    resolved = settings if settings is not None else active_settings()
    if resolved is not None and resolved.train_cameras:
        return [str(s) for s in resolved.train_cameras]
    return list(DEFAULT_TRAIN_CAMERAS)


def _resolve_cameras(
    spec: CameraSpec,
    available: Sequence[str],
    settings: Optional[RecordingSettings] = None,
) -> List[str]:
    """Requested slots intersected with what the episode has, in schema order."""
    if spec is None:
        wanted: Sequence[str] = default_train_cameras(settings)
    elif isinstance(spec, str):
        wanted = list(available) if spec == ALL_CAMERAS else [spec]
    else:
        wanted = list(spec)
    have = set(available)
    ordered = [s for s in CAMERA_SLOTS if s in wanted and s in have]
    # Preserve any non-standard slot a future rig adds.
    ordered += [s for s in wanted if s not in CAMERA_SLOTS and s in have]
    return ordered


@dataclass
class Episode:
    """
    One episode: metadata + trajectory in memory, images left on disk.

    ``load_episode`` stays cheap (a JSON read and a small npz); frames are
    pulled only when :meth:`images`, :meth:`frame` or :meth:`observations`
    asks for them.
    """

    path: Path
    meta: Dict[str, Any]
    trajectory: Dict[str, np.ndarray]
    cameras: List[str]

    @property
    def task(self) -> str:
        return str(self.meta.get("task", self.path.parent.name))

    @property
    def prompt(self) -> str:
        return str(self.meta.get("prompt", self.task))

    # Alias for callers written against the previous reader.
    @property
    def language_instruction(self) -> str:
        return self.prompt

    @property
    def success(self) -> bool:
        return bool(self.meta.get("success", False))

    @property
    def num_frames(self) -> int:
        ts = self.trajectory.get("timestamps")
        if ts is not None:
            return int(len(ts))
        return int(self.meta.get("num_frames", 0))

    @property
    def available_cameras(self) -> List[str]:
        """Every slot the episode carries, per ``metadata["cameras"]``."""
        return list(self.meta.get("cameras", []))

    @property
    def is_spark(self) -> bool:
        """True for a SPARK-generated episode, False for human teleop."""
        return self.meta.get("source") == SPARK_SOURCE_VALUE

    @property
    def metadata(self) -> EpisodeMetadata:
        return EpisodeMetadata.from_dict(self.meta)

    def frame(self, camera: str, index: int) -> Optional[np.ndarray]:
        """One RGB frame, or None when it is missing / unreadable."""
        bgr = cv2.imread(str(frame_path(self.path, camera, index)))
        if bgr is None:
            return None
        return cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)

    def images(self, camera: str) -> np.ndarray:
        """Every frame of one slot as ``(N, H, W, 3)`` uint8 RGB."""
        frames = [self.frame(camera, i) for i in range(self.num_frames)]
        frames = [f for f in frames if f is not None]
        if not frames:
            return np.zeros((0, 0, 0, 3), dtype=np.uint8)
        return np.stack(frames, axis=0)

    def observations(self, index: int) -> Dict[str, np.ndarray]:
        """``{slot: rgb}`` at one timestep, for the selected cameras only."""
        out: Dict[str, np.ndarray] = {}
        for cam in self.cameras:
            img = self.frame(cam, index)
            if img is not None:
                out[cam] = img
        return out

    def __len__(self) -> int:
        return self.num_frames

    def __repr__(self) -> str:  # pragma: no cover - cosmetic
        return (
            f"Episode(task={self.task!r}, id={self.meta.get('episode_id')}, "
            f"frames={self.num_frames}, cameras={self.cameras}, "
            f"success={self.success}, spark={self.is_spark})"
        )


def load_episode(
    episode_dir,
    cameras: CameraSpec = None,
    settings: Optional[RecordingSettings] = None,
) -> Episode:
    """
    Load one episode directory.

    ``cameras`` defaults to the configured training subset
    (``recording.train_cameras``, bird + wrist out of the box); pass ``"all"``
    for every slot on disk, or an explicit list of slot names. ``settings``
    supplies that config to an offline caller that has no live pipeline::

        s = RecordingSettings.from_config(load_family_yaml("ur10e"))
        load_episode(path, settings=s)
    """
    path = Path(episode_dir)
    meta_path = path / METADATA_FILENAME
    if not meta_path.exists():
        raise FileNotFoundError(f"not an episode (no {METADATA_FILENAME}): {path}")
    meta = json.loads(meta_path.read_text())

    traj: Dict[str, np.ndarray] = {}
    npz_path = path / TRAJECTORY_FILENAME
    if npz_path.exists():
        with np.load(npz_path, allow_pickle=False) as data:
            traj = {k: data[k] for k in data.files}

    available = [str(c) for c in (meta.get("cameras") or [])]
    if not available:
        # Hand-assembled episode with no camera list: fall back to the dirs.
        available = [s for s in CAMERA_SLOTS if (path / IMAGES_DIRNAME / s).is_dir()]
    return Episode(
        path=path,
        meta=meta,
        trajectory=traj,
        cameras=_resolve_cameras(cameras, available, settings),
    )


def iter_episodes(
    data_root,
    cameras: CameraSpec = None,
    settings: Optional[RecordingSettings] = None,
) -> Iterator[Episode]:
    """Every episode under ``data_root``, sorted; unreadable ones are skipped."""
    for ep_dir in iter_episode_dirs(data_root):
        try:
            yield load_episode(ep_dir, cameras=cameras, settings=settings)
        except (OSError, ValueError):
            continue


def load_dataset(
    data_root,
    cameras: CameraSpec = None,
    success_only: bool = False,
    settings: Optional[RecordingSettings] = None,
) -> List[Episode]:
    """
    Every episode as a list.

    ``success_only`` exists because the human corpus is success-curated
    (failed episodes were deleted from disk) while SPARK records its honest
    failures too; a matched training set must filter both sides the same way.
    """
    out = list(iter_episodes(data_root, cameras=cameras, settings=settings))
    if success_only:
        out = [ep for ep in out if ep.success]
    return out


def validate_episode(episode: Episode) -> List[str]:
    """
    Structural problems with one episode, as human-readable strings.

    Empty list means it satisfies the format contract: all six core npz keys,
    equal lengths, ``timestamps[0] == 0``, ``actions[:,6]`` equal to
    ``gripper_positions``, and one frame per row for every declared camera.
    """
    problems: List[str] = []
    traj = episode.trajectory
    missing = [k for k in NPZ_CORE_KEYS if k not in traj]
    if missing:
        problems.append(f"missing npz keys: {missing}")
        return problems

    n = len(traj["timestamps"])
    for key in NPZ_CORE_KEYS:
        if len(traj[key]) != n:
            problems.append(f"{key} has {len(traj[key])} rows, expected {n}")
    if n and abs(float(traj["timestamps"][0])) > 1e-9:
        problems.append(f"timestamps[0] = {traj['timestamps'][0]!r}, expected 0.0")
    if n and not np.allclose(traj["actions"][:, 6], traj["gripper_positions"]):
        problems.append("actions[:,6] does not match gripper_positions")
    if int(episode.meta.get("num_frames", n)) != n:
        problems.append(f"metadata.num_frames={episode.meta.get('num_frames')} but npz has {n}")
    for cam in episode.available_cameras:
        cam_dir = images_dir(episode.path, cam)
        if not cam_dir.is_dir():
            problems.append(f"camera {cam} declared but has no directory")
            continue
        found = len(list(cam_dir.glob("frame_*.jpg")))
        if found != n:
            problems.append(f"camera {cam} has {found} frames, expected {n}")
    return problems


# Callers written against the previous reader used these names.
VLAEpisode = Episode

__all__ = [
    "ALL_CAMERAS",
    "Episode",
    "VLAEpisode",
    "default_train_cameras",
    "iter_episodes",
    "load_dataset",
    "load_episode",
    "validate_episode",
]

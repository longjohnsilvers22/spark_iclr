"""
Disk side of an episode: frames, trajectory.npz, metadata.json, QA artifacts.

Split out of the recorder so the recorder thread does nothing but sequence
work at a fixed rate, and so the on-disk format has one implementation that
the parity test can exercise without a robot, a camera or a thread.

Everything here writes the *human* schema (see ``schema.py``); the SPARK-only
npz columns and the nested ``spark`` metadata block ride along as extras that
a strict human-format reader never sees.

The two QA artifacts are optional (``emit_video`` / ``emit_plot``) and are
never read back by a converter:

* ``episode_video.mp4``: camera slots tiled left-to-right, encoded through
  ``video_recorder._encode_h264_posix_spawn``. That is the only encoder in the
  tree that does not deadlock the threaded server (it avoids imageio's
  ``preexec_fn``/fork path). Tiles are written at half resolution and the frame
  count is capped, because the encoder materializes the whole clip in RAM.
* ``trajectory_plot.png``: delegated to ``control.trajectory._save_plot``, the
  same 4-panel figure the executor already produces.
"""

from __future__ import annotations

import json
import logging
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence

import cv2
import numpy as np

from spark_real.control.trajectory import _save_plot
from spark_real.recording.naming import frame_path
from spark_real.recording.schema import (
    ACTION_DIM,
    ACTION_SOURCE_DTYPE,
    IMAGES_DIRNAME,
    IMAGE_HEIGHT,
    IMAGE_WIDTH,
    JOINT_DIM,
    METADATA_FILENAME,
    NPZ_CORE_KEYS,
    NPZ_CORE_WIDTHS,
    NPZ_DTYPE,
    NPZ_EXTRA_WIDTHS,
    PLOT_FILENAME,
    POSE_DIM,
    TRAJECTORY_FILENAME,
    VIDEO_FILENAME,
    EpisodeMetadata,
    actual_fps,
)
from spark_real.video_recorder import _encode_h264_posix_spawn

logger = logging.getLogger(__name__)

# QA-video budget. The encoder materializes every frame, so a long episode at
# full tile resolution would be gigabytes for an artifact nothing reads.
VIDEO_TILE_SCALE = 0.5
VIDEO_MAX_FRAMES = 1200


def write_frame(
    episode_dir: Path,
    slot: str,
    index: int,
    rgb: np.ndarray,
    *,
    image_size: Sequence[int] = (IMAGE_WIDTH, IMAGE_HEIGHT),
    jpeg_quality: int = 95,
) -> bool:
    """
    One ``images/<slot>/frame_NNNN.jpg``. Returns False on failure.

    RGB in, BGR on disk, matching the human producer's
    ``cv2.imwrite(cvtColor(img, RGB2BGR))``, so ``cv2.imread`` and PIL both
    read correct colour back without a swap.
    """
    try:
        arr = np.asarray(rgb)
        if arr.ndim != 3 or arr.shape[2] != 3:
            return False
        w, h = int(image_size[0]), int(image_size[1])
        if arr.shape[1] != w or arr.shape[0] != h:
            arr = cv2.resize(arr, (w, h), interpolation=cv2.INTER_AREA)
        if arr.dtype != np.uint8:
            arr = np.clip(arr, 0, 255).astype(np.uint8)
        path = frame_path(episode_dir, slot, index)
        path.parent.mkdir(parents=True, exist_ok=True)
        return bool(
            cv2.imwrite(
                str(path),
                cv2.cvtColor(arr, cv2.COLOR_RGB2BGR),
                [int(cv2.IMWRITE_JPEG_QUALITY), int(jpeg_quality)],
            )
        )
    except Exception as exc:
        logger.warning("write_frame(%s, %d) failed: %s", slot, index, exc)
        return False


def black_frame(image_size: Sequence[int] = (IMAGE_WIDTH, IMAGE_HEIGHT)) -> np.ndarray:
    """Placeholder for a camera that returned nothing on a tick."""
    return np.zeros((int(image_size[1]), int(image_size[0]), 3), dtype=np.uint8)


class TrajectoryBuffer:
    """
    Column store for the episode, appended one row per recorded frame.

    Rows are buffered (they are tiny) while images stream straight to disk.
    ``to_arrays`` produces exactly the npz contract: float64 everywhere except
    the ``action_source`` provenance column, and ``timestamps`` rebased so
    ``timestamps[0] == 0.0``.
    """

    def __init__(self) -> None:
        self.wall_time: List[float] = []
        self.tcp_poses: List[Sequence[float]] = []
        self.joint_positions: List[Sequence[float]] = []
        self.joint_velocities: List[Sequence[float]] = []
        self.tcp_velocity: List[Sequence[float]] = []
        self.wrench: List[Sequence[float]] = []
        self.gripper_positions: List[float] = []
        self.actions: List[Sequence[float]] = []
        self.action_source: List[str] = []

    def __len__(self) -> int:
        return len(self.wall_time)

    def append(
        self,
        *,
        wall_time: float,
        proprio: Dict[str, Any],
        action: Sequence[float],
        action_source: str,
    ) -> None:
        self.wall_time.append(float(wall_time))
        self.tcp_poses.append(proprio.get("tcp_pose") or [])
        self.joint_positions.append(proprio.get("joint_positions") or [])
        self.joint_velocities.append(proprio.get("joint_velocities") or [])
        self.tcp_velocity.append(proprio.get("tcp_velocity") or [])
        self.wrench.append(proprio.get("wrench") or [])
        act = np.asarray(action, dtype=np.float64).ravel()
        self.actions.append(act)
        # The human corpus has actions[:,6] bit-identical to
        # gripper_positions; take the single source of truth from the action.
        self.gripper_positions.append(float(act[6]) if act.size > 6 else float("nan"))
        self.action_source.append(str(action_source))

    def to_arrays(self) -> Dict[str, np.ndarray]:
        wall = np.asarray(self.wall_time, dtype=NPZ_DTYPE)
        ts = wall - wall[0] if wall.size else wall
        actions = pad_rows(self.actions, ACTION_DIM, name="actions")
        out: Dict[str, np.ndarray] = {
            "timestamps": ts,
            "tcp_poses": pad_rows(self.tcp_poses, POSE_DIM, name="tcp_poses"),
            "joint_positions": pad_rows(self.joint_positions, JOINT_DIM, name="joint_positions"),
            "joint_velocities": pad_rows(self.joint_velocities, JOINT_DIM, name="joint_velocities"),
            # Taken from the (already hole-filled) action column rather than
            # from the buffered scalars, so the two stay bit-identical after a
            # carry-fill exactly as they are in the human corpus.
            "gripper_positions": _gripper_column(actions, self.gripper_positions),
            "actions": actions,
            "wall_time": wall,
            "tcp_velocity": pad_rows(self.tcp_velocity, POSE_DIM, name="tcp_velocity"),
            "wrench": pad_rows(self.wrench, POSE_DIM, name="wrench"),
            "action_source": np.asarray(self.action_source, dtype=ACTION_SOURCE_DTYPE),
        }
        return out


def _gripper_column(actions: np.ndarray, fallback: Sequence[float]) -> np.ndarray:
    """``actions[:,6]`` when the action array is wide enough, else the buffer."""
    if actions.ndim == 2 and actions.shape[1] > 6:
        return np.array(actions[:, 6], dtype=NPZ_DTYPE)
    return np.asarray(fallback, dtype=NPZ_DTYPE)


def _forward_fill(arr: np.ndarray) -> np.ndarray:
    """Each NaN replaced by the nearest earlier valid value in its column."""
    holes = np.isnan(arr)
    if not holes.any():
        return arr
    index = np.where(~holes, np.arange(arr.shape[0])[:, None], -1)
    last = np.maximum.accumulate(index, axis=0)
    have = last >= 0
    gathered = np.take_along_axis(arr, np.where(have, last, 0), axis=0)
    return np.where(holes & have, gathered, arr)


def carry_fill(arr: np.ndarray) -> np.ndarray:
    """
    Hole-fill a column store from its neighbours: previous valid row first,
    then the next one for a hole at the very start.

    A single dropped RTDE tick is a hiccup, not a corrupt demonstration, and
    NaN is not a value a converter can survive: the LeRobot converter RAISES
    on one NaN and aborts the whole run. A channel with no valid sample
    anywhere is left NaN on purpose: that is a real missing channel, and the
    recorder fails the episode on it rather than inventing data.
    """
    # Order matters: the previous valid row wins, and the reverse pass only
    # reaches holes at the very start, where there is no previous row.
    filled = _forward_fill(arr)
    return _forward_fill(filled[::-1])[::-1]


def pad_rows(rows: Sequence[Sequence[float]], width: int, *, name: str = "") -> np.ndarray:
    """
    ``(N, width)`` float64. Short/missing rows are carry-filled from their
    neighbours (see :func:`carry_fill`); only a wholly absent channel stays NaN.
    """
    if not len(rows):
        return np.zeros((0, width), dtype=NPZ_DTYPE)
    out = np.full((len(rows), width), np.nan, dtype=NPZ_DTYPE)
    for i, row in enumerate(rows):
        arr = np.asarray(row, dtype=NPZ_DTYPE).ravel()
        n = min(arr.size, width)
        if n:
            out[i, :n] = arr[:n]
    holes = int(np.isnan(out).sum())
    if not holes:
        return out
    out = carry_fill(out)
    filled = holes - int(np.isnan(out).sum())
    if filled:
        logger.warning(
            "pad_rows(%s): carry-filled %d missing value(s) from a neighbouring row",
            name or f"width={width}",
            filled,
        )
    return out


def save_trajectory(episode_dir: Path, arrays: Dict[str, np.ndarray]) -> Path:
    """
    Write ``trajectory.npz``.

    Uncompressed ``savez``, like the human producer: the arrays are small and
    an uncompressed npz memory-maps for the training loader.
    """
    path = Path(episode_dir) / TRAJECTORY_FILENAME
    path.parent.mkdir(parents=True, exist_ok=True)
    np.savez(path, **arrays)
    return path


def save_metadata(episode_dir: Path, meta: EpisodeMetadata) -> Path:
    """Write ``metadata.json`` with the human keys first, in the human order."""
    path = Path(episode_dir) / METADATA_FILENAME
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(meta.to_dict(), indent=2))
    return path


def build_metadata(
    *,
    episode_id: int,
    task: str,
    success: bool,
    arrays: Dict[str, np.ndarray],
    start_time: datetime,
    end_time: datetime,
    robot_ip: str,
    cameras: Sequence[str],
    prompt: Optional[str] = None,
    source: Optional[str] = None,
    spark: Optional[Dict[str, Any]] = None,
    verify: Optional[Dict[str, Any]] = None,
) -> EpisodeMetadata:
    """
    The 11 human keys, computed the way the human producer computes them.

    ``duration_seconds`` is wall clock from record-start to save and runs
    longer than ``timestamps[-1]``; ``actual_fps`` therefore comes from the
    timestamps, never from the duration.
    """
    ts = arrays.get("timestamps", np.zeros(0))
    return EpisodeMetadata(
        episode_id=int(episode_id),
        task=str(task),
        success=bool(success),
        num_frames=int(len(ts)),
        start_time=start_time.isoformat(),
        end_time=end_time.isoformat(),
        duration_seconds=float((end_time - start_time).total_seconds()),
        robot_ip=str(robot_ip),
        cameras=list(cameras),
        prompt=str(prompt if prompt is not None else task),
        actual_fps=actual_fps(ts),
        source=source,
        spark=spark,
        verify=verify,
    )


def save_plot(
    episode_dir: Path,
    arrays: Dict[str, np.ndarray],
    transitions: Optional[Sequence] = None,
) -> Optional[Path]:
    """``trajectory_plot.png`` via the executor's existing 4-panel figure."""
    try:
        _save_plot(
            Path(episode_dir),
            {
                "tcp_poses": np.asarray(arrays["tcp_poses"]),
                "timestamps": np.asarray(arrays["timestamps"]),
                "gripper_positions": np.asarray(arrays["gripper_positions"]),
                "action_transitions": list(transitions or []),
            },
        )
    except Exception as exc:
        logger.warning("save_plot failed: %s", exc)
        return None
    path = Path(episode_dir) / PLOT_FILENAME
    return path if path.exists() else None


def save_video(
    episode_dir: Path,
    slots: Sequence[str],
    num_frames: int,
    fps: float,
) -> Optional[Path]:
    """
    ``episode_video.mp4``: the recorded slots tiled left-to-right.

    Reads the JPEGs back off disk rather than holding a second copy in RAM
    during the run. QA only; nothing downstream parses it.
    """
    if num_frames < 2 or not slots:
        return None
    stride = max(1, int(np.ceil(num_frames / VIDEO_MAX_FRAMES)))
    tiles: List[np.ndarray] = []
    for idx in range(0, num_frames, stride):
        row: List[np.ndarray] = []
        for slot in slots:
            fp = frame_path(episode_dir, slot, idx)
            bgr = cv2.imread(str(fp)) if fp.exists() else None
            if bgr is None:
                continue
            small = cv2.resize(
                bgr,
                (0, 0),
                fx=VIDEO_TILE_SCALE,
                fy=VIDEO_TILE_SCALE,
                interpolation=cv2.INTER_AREA,
            )
            row.append(cv2.cvtColor(small, cv2.COLOR_BGR2RGB))
        if not row:
            continue
        height = min(r.shape[0] for r in row)
        row = [r[:height] for r in row]
        tiles.append(np.hstack(row))
    if len(tiles) < 2:
        return None
    shape = tiles[0].shape
    tiles = [t for t in tiles if t.shape == shape]
    path = Path(episode_dir) / VIDEO_FILENAME
    try:
        _encode_h264_posix_spawn(
            np.stack(tiles, axis=0), path, fps=max(1, int(round(fps / stride)))
        )
    except Exception as exc:
        logger.warning("save_video failed: %s", exc)
        return None
    return path if path.exists() else None


def slots_on_disk(episode_dir: Path) -> List[str]:
    """Camera slots that actually have a frame directory with frames in it."""
    root = Path(episode_dir)
    out: List[str] = []
    images = root / IMAGES_DIRNAME
    if not images.is_dir():
        return out
    for child in sorted(images.iterdir()):
        if child.is_dir() and any(child.glob("frame_*.jpg")):
            out.append(child.name)
    return out


__all__ = [
    "NPZ_CORE_KEYS",
    "NPZ_CORE_WIDTHS",
    "NPZ_EXTRA_WIDTHS",
    "TrajectoryBuffer",
    "black_frame",
    "carry_fill",
    "pad_rows",
    "build_metadata",
    "save_metadata",
    "save_plot",
    "save_trajectory",
    "save_video",
    "slots_on_disk",
    "write_frame",
]

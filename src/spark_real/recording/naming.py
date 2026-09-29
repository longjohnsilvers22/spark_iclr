"""
Episode directory naming and allocation.

The task string IS the directory name, verbatim, spaces and all
(``put the knife in the tray``, not ``put_the_knife_in_the_tray``). Episode
numbers are ``max(existing) + 1``, never first-hole: gaps are permanent and
expected (the human corpus has them where failed episodes were deleted).
"""

from __future__ import annotations

from pathlib import Path
from typing import Iterator, List

from spark_real.recording.schema import (
    EPISODE_FMT,
    FRAME_FMT,
    IMAGES_DIRNAME,
    METADATA_FILENAME,
)


def slug(text: str) -> str:
    """
    Filesystem-safe slug. NOT used for episode directories (those keep the
    verbatim task string), only for auxiliary artifacts that need a token.
    """
    s = "".join(c if c.isalnum() or c in "-_." else "_" for c in (text or "")).strip("_")
    return s or "task"


def safe_task_name(task: str) -> str:
    """
    The task string as a single path component.

    Task strings arrive over the network (``/api/execute``,
    ``/api/vla_record/start``), so a separator, a ``..`` segment or a leading
    ``/`` would let a caller write outside the configured data root. Spaces are
    preserved untouched: the directory name has to match the human corpus
    verbatim. A malformed name raises rather than being rewritten to something
    safe-but-wrong, which would silently merge two tasks into one directory.
    """
    name = (task or "").strip()
    if not name:
        return "task"
    if name in (".", "..") or any(ch in name for ch in ("/", "\\", "\x00")):
        raise ValueError(f"task name is not a single path component: {task!r}")
    return name


def task_dir(data_root, task: str) -> Path:
    """``<data_root>/<task string, verbatim>``, confined to ``data_root``."""
    root = Path(data_root).expanduser()
    candidate = root / safe_task_name(task)
    resolved_root = root.resolve()
    if candidate.resolve() != resolved_root and resolved_root not in candidate.resolve().parents:
        raise ValueError(f"task directory escapes the data root: {task!r}")
    return candidate


def existing_episode_numbers(root) -> List[int]:
    """Every ``episode_NNNN`` number already under ``root``, sorted."""
    path = Path(root)
    if not path.is_dir():
        return []
    numbers = []
    for child in path.glob("episode_*"):
        if not child.is_dir():
            continue
        try:
            numbers.append(int(child.name.split("_")[1]))
        except (IndexError, ValueError):
            continue
    return sorted(numbers)


def next_episode_number(data_root, task: str) -> int:
    """
    Next free episode id for one task: ``max(existing) + 1``, 0 when empty.

    Deliberately not the first hole: reusing a deleted episode's number would
    silently collide with anything that already referenced it.
    """
    numbers = existing_episode_numbers(task_dir(data_root, task))
    return (numbers[-1] + 1) if numbers else 0


def episode_dir(data_root, task: str, episode_id: int) -> Path:
    """``<data_root>/<task>/episode_NNNN``."""
    return task_dir(data_root, task) / EPISODE_FMT.format(int(episode_id))


def images_dir(episode_path, slot: str) -> Path:
    """``<episode>/images/<slot>``."""
    return Path(episode_path) / IMAGES_DIRNAME / slot


def frame_path(episode_path, slot: str, index: int) -> Path:
    """``<episode>/images/<slot>/frame_NNNN.jpg``."""
    return images_dir(episode_path, slot) / FRAME_FMT.format(int(index))


def iter_episode_dirs(data_root) -> Iterator[Path]:
    """
    Yield every episode directory under ``data_root``, sorted.

    An episode is any directory containing ``metadata.json``.
    """
    root = Path(data_root).expanduser()
    if not root.exists():
        return
    for meta in sorted(root.rglob(METADATA_FILENAME)):
        yield meta.parent


__all__ = [
    "episode_dir",
    "existing_episode_numbers",
    "frame_path",
    "images_dir",
    "iter_episode_dirs",
    "next_episode_number",
    "slug",
    "task_dir",
]

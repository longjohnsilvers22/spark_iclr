"""Persistent per-task operator annotations (click / box) for SAM3.

An operator can rescue a detection by clicking or dragging a box in the UI
(``/api/detect_click``, ``/api/detect_box``); this store keeps that rescue
across runs, since ``/api/execute`` takes its own capture and re-detects from
text. It follows the same contract as ``perception/prompt_cache.LearnedPrompts``:

* An anchor can only ever ADD a candidate, never force one. It is folded in as
  an extra box-prompted SAM3 pass whose result still faces the normal quality
  gate and count gate. A stale anchor loses to a good text detection.
* Only annotations the operator explicitly SAVED are recorded. A one-off
  rescue click does not silently become permanent state.
* A corrupt or missing file is an empty store, never an error.
* Anchors are per (task, label, camera): a box is meaningless in another
  camera's pixels.

The box is stored NORMALISED (0..1) against the image it was drawn on, so it
survives a resolution change -- the raw pixel box would not.
"""

from __future__ import annotations

import json
import logging
import os
import tempfile
import threading
from pathlib import Path
from typing import Dict, List, Optional

logger = logging.getLogger(__name__)

SCHEMA = 1


def _slug(text: str) -> str:
    keep = [c.lower() if c.isalnum() else "_" for c in str(text).strip()]
    out = "".join(keep).strip("_")
    while "__" in out:
        out = out.replace("__", "_")
    return out or "task"


class AnnotationAnchors:
    """JSON-backed {task: {label: [anchor, ...]}} with atomic writes."""

    def __init__(self, path):
        self._path = Path(path)
        self._lock = threading.Lock()
        self._data: Dict[str, Dict[str, List[dict]]] = {}
        self._load()

    def _load(self) -> None:
        try:
            raw = json.loads(self._path.read_text())
            if isinstance(raw, dict) and raw.get("schema") == SCHEMA:
                self._data = raw.get("tasks", {}) or {}
            elif isinstance(raw, dict):
                logger.info(
                    "annotation anchors: schema %r != %d; starting empty",
                    raw.get("schema"), SCHEMA,
                )
        except FileNotFoundError:
            pass
        except Exception as exc:  # noqa: BLE001 - a bad file is an empty store
            logger.warning("annotation anchors: unreadable (%s); starting empty", exc)

    def _flush(self) -> None:
        payload = json.dumps(
            {"schema": SCHEMA, "tasks": self._data}, indent=2, sort_keys=True
        )
        try:
            self._path.parent.mkdir(parents=True, exist_ok=True)
            fd, tmp = tempfile.mkstemp(
                dir=str(self._path.parent), prefix=".annotation_anchors."
            )
            with os.fdopen(fd, "w") as fh:
                fh.write(payload)
            os.replace(tmp, self._path)
        except Exception as exc:  # noqa: BLE001 - never fail a run over this
            logger.warning("annotation anchors: could not persist (%s)", exc)

    # -- write ------------------------------------------------------------
    def save(self, task, label, camera, box_norm, note="") -> dict:
        """Record one normalised box for (task, label, camera). Replaces any
        previous anchor for that exact triple -- the newest annotation is the
        operator's current intent, and keeping both would make the fold-in
        order decide which wins."""
        if not task or not label or box_norm is None or len(box_norm) != 4:
            raise ValueError("task, label and a 4-element box are required")
        x1, y1, x2, y2 = (float(v) for v in box_norm)
        if not all(0.0 <= v <= 1.0 for v in (x1, y1, x2, y2)):
            raise ValueError(f"box must be normalised 0..1, got {box_norm}")
        anchor = {
            "label": str(label),
            "camera": str(camera or ""),
            "box_norm": [min(x1, x2), min(y1, y2), max(x1, x2), max(y1, y2)],
            "note": str(note or ""),
        }
        key = _slug(task)
        with self._lock:
            per_task = self._data.setdefault(key, {})
            kept = [
                a for a in per_task.get(str(label), [])
                if a.get("camera") != anchor["camera"]
            ]
            kept.append(anchor)
            per_task[str(label)] = kept
            self._flush()
        logger.info(
            "annotation anchors: saved %r/%r on %s -> %s",
            task, label, anchor["camera"], anchor["box_norm"],
        )
        return anchor

    # -- read -------------------------------------------------------------
    def for_task(self, task) -> Dict[str, List[dict]]:
        return dict(self._data.get(_slug(task), {})) if task else {}

    def labels(self, task) -> List[str]:
        return sorted(self.for_task(task).keys())

    def drop(self, task, label=None) -> int:
        key = _slug(task)
        with self._lock:
            if key not in self._data:
                return 0
            if label is None:
                n = sum(len(v) for v in self._data[key].values())
                del self._data[key]
            else:
                n = len(self._data[key].pop(str(label), []))
                if not self._data[key]:
                    del self._data[key]
            self._flush()
        return n

    @staticmethod
    def pixels(anchor, width, height):
        """Normalised anchor -> (x1, y1, x2, y2) pixels for this image size."""
        x1, y1, x2, y2 = anchor["box_norm"]
        return (
            int(round(x1 * float(width))), int(round(y1 * float(height))),
            int(round(x2 * float(width))), int(round(y2 * float(height))),
        )

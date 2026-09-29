"""Persistent learned SAM3 prompts per task+group.

A JSON file anchored off the output root (next to ``grasp_calibration.json``
and the BT library) so a prompt that rescued one run is not paid for again
(one LLM call + latency) on the next run of the same task. The contract that
keeps it safe:

* The cache can only ever SKIP an LLM call, never force a detection -- cached
  prompts are folded in as alt_prompts and re-validated by the count gate on
  every single run. A phrase that stops working simply fails the gate and the
  ladder proceeds to the LLM rung exactly as if the cache were empty.
* Only prompts that actually MADE THE GATE PASS are recorded; a rejected
  proposal never touches the file.
* A corrupt or missing file is an empty cache, never an error: losing this
  file costs one extra LLM call per task, which is not worth failing a
  detection over.

Writes are atomic (tmp + rename) and serialized by a lock so concurrent
detect calls cannot interleave partial JSON.
"""

from __future__ import annotations

import json
import logging
import os
import tempfile
import threading
from pathlib import Path
from typing import Dict, List, Sequence

logger = logging.getLogger(__name__)

# Per task+group cap. New winners are prepended; the tail falls off. Keeps
# the retry pass bounded (every cached phrase is one more SAM3 prompt) while
# retaining a couple of alternates for scenes where the newest phrase fails.
MAX_PROMPTS_PER_GROUP = 6


class LearnedPrompts:
    """Tiny durable map: (task, group text) -> prompts that have worked."""

    def __init__(self, path):
        self._path = Path(path)
        self._lock = threading.Lock()

    def _load(self) -> Dict[str, Dict[str, List[str]]]:
        try:
            with open(self._path) as fh:
                data = json.load(fh)
        except FileNotFoundError:
            return {}
        except (OSError, ValueError) as exc:
            logger.warning(
                "learned prompts at %s unreadable (%s); starting empty",
                self._path,
                exc,
            )
            return {}
        if not isinstance(data, dict):
            return {}
        return data

    def lookup(self, task: str, group: str) -> List[str]:
        """Prompts previously accepted for this task+group, newest first."""
        entry = self._load().get(str(task), {}).get(str(group))
        if not isinstance(entry, list):
            return []
        return [str(p) for p in entry if isinstance(p, str) and p.strip()]

    def record(self, task: str, group: str, prompts: Sequence[str]) -> None:
        """Persist gate-accepted prompts, newest first, deduplicated, capped."""
        fresh = [str(p) for p in prompts if isinstance(p, str) and p.strip()]
        if not fresh:
            return
        with self._lock:
            data = self._load()
            entry = data.setdefault(str(task), {})
            merged = list(dict.fromkeys(fresh + self.lookup(task, group)))
            entry[str(group)] = merged[:MAX_PROMPTS_PER_GROUP]
            try:
                self._path.parent.mkdir(parents=True, exist_ok=True)
                fd, tmp = tempfile.mkstemp(
                    dir=str(self._path.parent), prefix=".learned_prompts."
                )
                try:
                    with os.fdopen(fd, "w") as fh:
                        json.dump(data, fh, indent=2, sort_keys=True)
                    os.replace(tmp, self._path)
                finally:
                    if os.path.exists(tmp):
                        os.unlink(tmp)
            except OSError as exc:
                # Losing a cache write costs one future LLM call; a detection
                # must never fail over it.
                logger.warning(
                    "could not persist learned prompts to %s: %s",
                    self._path,
                    exc,
                )

"""
Online per-object grasp-depth calibration for spark_real.

Tracks how much the descent-retry loop in ``ScoreExecutor`` had to drop
below the perception-predicted Z to actually grasp each object class.
Future grasps for the same label apply the learned offset upfront, so
the first descent succeeds instead of needing 1-5 retries.

Why this matters: SAM3 + DA3 backprojection consistently puts thin /
shiny / edge-detected objects (silverware, plates, coins) ~1-3 cm
above the real surface.  Recording the actual descent-that-worked lets
us learn this pose-correction online, no offline calibration sweep.

Storage: a JSON file (grasp_calibration.json) under the configured output dir::

    {
      "knife handle": {
        "n": 4,
        "mean": -0.024,        # learned dz to add (m)
        "M2": 0.00012,         # Welford running variance accumulator
        "n_retries_avg": 1.5,  # avg retries when this offset was logged
      },
      "red plushie": {...},
    }

Welford updates keep the file size bounded, no growing samples list.

Only depth is cached, not width or force. Width is pose-dependent
(the minor-axis we grip on changes as the object rotates), so caching
it by label would be misleading; the mask-derived prior in
``ScoreExecutor`` handles pose-aware width on every grasp.
"""

from __future__ import annotations

import json
import logging
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Optional

logger = logging.getLogger(__name__)


def _normalise_label(label: str) -> str:
    # Lowercase + collapse instance suffixes ("knife handle 2" -> "knife handle").
    s = re.sub(r"\s+\d+$", "", label.lower().strip())
    return s


@dataclass
class _Stat:
    n: int = 0
    mean: float = 0.0
    M2: float = 0.0  # Welford running sum-of-squared-deltas
    n_retries_avg: float = 0.0

    def update(self, x: float, n_retries: int) -> None:
        self.n += 1
        delta = x - self.mean
        self.mean += delta / self.n
        delta2 = x - self.mean
        self.M2 += delta * delta2
        # rolling avg of retry count
        self.n_retries_avg += (n_retries - self.n_retries_avg) / self.n

    def variance(self) -> float:
        return self.M2 / max(1, self.n - 1)

    def to_dict(self) -> Dict:
        return {
            "n": self.n,
            "mean": round(self.mean, 5),
            "M2": round(self.M2, 7),
            "n_retries_avg": round(self.n_retries_avg, 2),
        }

    @classmethod
    def from_dict(cls, d: Dict) -> "_Stat":
        return cls(
            n=int(d.get("n", 0)),
            mean=float(d.get("mean", 0.0)),
            M2=float(d.get("M2", 0.0)),
            n_retries_avg=float(d.get("n_retries_avg", 0.0)),
        )


class GraspCalibration:
    """
    Disk-backed per-object grasp depth-correction memory.
    """

    def __init__(self, path: Path, *, min_samples: int = 2):
        self.path = Path(path).expanduser()
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self._stats: Dict[str, _Stat] = {}
        self._min_samples = min_samples
        self._load()

    # I/O

    def _load(self) -> None:
        if not self.path.exists():
            return
        try:
            blob = json.loads(self.path.read_text())
            self._stats = {k: _Stat.from_dict(v) for k, v in blob.items()}
        except (OSError, json.JSONDecodeError) as e:
            logger.warning("GraspCalibration: bad %s: %s; starting fresh", self.path, e)

    def _flush(self) -> None:
        self.path.write_text(
            json.dumps(
                {k: v.to_dict() for k, v in self._stats.items()},
                indent=2,
                sort_keys=True,
            )
        )

    # API

    def learned_offset(self, label: str) -> Optional[float]:
        """
        Return learned dz (m) to add to perception-predicted grasp Z.

        Returns None if fewer than ``min_samples`` successes have been
        recorded for this label, caller should fall back to default.
        """
        s = self._stats.get(_normalise_label(label))
        if s is None or s.n < self._min_samples:
            return None
        return s.mean

    def confidence(self, label: str) -> float:
        """
        Confidence in the learned offset, in [0, 1].  Grows with n,
        shrinks with variance.  Useful for blending learned vs default.
        """
        s = self._stats.get(_normalise_label(label))
        if s is None or s.n < 1:
            return 0.0
        # n_obs / (n_obs + kappa); kappa scales with variance
        var = s.variance()
        kappa = 1.0 + 100.0 * var  # 1cm std -> kappa ~= 2
        return s.n / (s.n + kappa)

    def record_success(self, label: str, applied_dz: float, n_retries: int = 0) -> None:
        """
        Log a successful grasp: ``applied_dz`` is how far below the
        perception-predicted Z the gripper actually ended up (negative
        = went lower, positive = went higher).  Pass ``n_retries`` so
        we can track which labels are converging.
        """
        key = _normalise_label(label)
        if key not in self._stats:
            self._stats[key] = _Stat()
        self._stats[key].update(float(applied_dz), int(n_retries))
        self._flush()

    def stats(self) -> Dict[str, Dict]:
        return {k: v.to_dict() for k, v in self._stats.items()}

    def __len__(self) -> int:
        return len(self._stats)


__all__ = ["GraspCalibration"]

"""Per-primitive execution traces (ASPIRE-style failure attribution).

One JSON per primitive under ``output/<run>/trace/<idx>_<primitive>.json`` so a
failure is attributable to a primitive rather than to the whole rollout.
Overlay PNGs are opt-in (``trace.save_images``) because they dominate the run
directory otherwise.

Never raises: tracing must not be able to fail a run.
"""

from __future__ import annotations

import json
import logging
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional

import numpy as np

logger = logging.getLogger(__name__)


def _jsonable(obj: Any) -> Any:
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    if isinstance(obj, (np.floating, np.integer)):
        return obj.item()
    if isinstance(obj, dict):
        return {str(k): _jsonable(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [_jsonable(v) for v in obj]
    if hasattr(obj, "to_dict"):
        try:
            return _jsonable(obj.to_dict())
        except Exception:
            return str(obj)
    if isinstance(obj, (str, int, float, bool)) or obj is None:
        return obj
    return str(obj)


@dataclass
class PrimitiveTrace:
    index: int
    primitive: str
    params: Dict[str, Any] = field(default_factory=dict)
    detections_used: List[Dict[str, Any]] = field(default_factory=list)
    grasp_strategy_resolved: Optional[str] = None
    gates: Dict[str, Any] = field(default_factory=dict)
    votes: List[Dict[str, Any]] = field(default_factory=list)
    elapsed_s: float = 0.0
    extra: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict:
        return _jsonable(
            {
                "index": self.index,
                "primitive": self.primitive,
                "params": self.params,
                "detections_used": self.detections_used,
                "grasp_strategy_resolved": self.grasp_strategy_resolved,
                "gates": self.gates,
                "votes": self.votes,
                "elapsed_s": round(float(self.elapsed_s), 3),
                **self.extra,
            }
        )


def detection_summary(det: Any) -> Dict[str, Any]:
    """Compact record of one detection: what the primitive actually acted on."""
    get = det.get if hasattr(det, "get") else (lambda k, d=None: getattr(det, k, d))
    keys = (
        "label",
        "confidence",
        "camera",
        "position_3d",
        "aspect_ratio",
        "obb_minor_m",
        "obb_confidence",
        "world_major_axis_rad",
        "low_quality",
        "reprompt_attempts",
    )
    return _jsonable({k: get(k, None) for k in keys})


class TraceWriter:
    """Writes primitive traces into ``<output_dir>/trace``."""

    def __init__(self, output_dir: Any, enabled: bool = True, save_images: bool = False):
        self.enabled = bool(enabled)
        self.save_images = bool(save_images)
        self.root: Optional[Path] = None
        if self.enabled and output_dir:
            try:
                self.root = Path(output_dir) / "trace"
                self.root.mkdir(parents=True, exist_ok=True)
            except Exception as exc:
                logger.warning("Trace disabled (cannot create dir): %s", exc)
                self.enabled = False
                self.root = None
        # Resume from what is already on disk: writers are constructed per
        # call (from_pipeline) and output_dir has no per-run component, so a
        # fresh counter would overwrite earlier attempts' traces.
        self._index = self._highest_on_disk()

    @classmethod
    def from_pipeline(cls, pipeline: Any) -> "TraceWriter":
        raw = getattr(getattr(pipeline, "profile", None), "raw", None) or {}
        block = raw.get("trace") if hasattr(raw, "get") else None
        block = block if isinstance(block, dict) else {}
        out = getattr(getattr(pipeline, "config", None), "output_dir", None)
        return cls(
            out,
            enabled=bool(block.get("enabled", True)),
            save_images=bool(block.get("save_images", False)),
        )

    def _highest_on_disk(self) -> int:
        """Largest ``NNN_`` prefix already written under ``root``, else 0."""
        if self.root is None:
            return 0
        best = 0
        try:
            for path in self.root.glob("*.json"):
                head = path.name.split("_", 1)[0]
                if head.isdigit():
                    best = max(best, int(head))
        except Exception:
            return best
        return best

    def next_index(self) -> int:
        self._index += 1
        return self._index

    def write(self, trace: PrimitiveTrace) -> Optional[Path]:
        if not self.enabled or self.root is None:
            return None
        try:
            name = "".join(c if c.isalnum() or c in "-_" else "_" for c in trace.primitive)
            path = self.root / f"{trace.index:03d}_{name}.json"
            path.write_text(json.dumps(trace.to_dict(), indent=2, default=str))
            return path
        except Exception as exc:
            logger.warning("Trace write failed: %s", exc)
            return None

    def write_verify(self, outcome: Any, elapsed_s: float = 0.0) -> Optional[Path]:
        """Trace for the terminal verification step."""
        data = outcome.to_dict() if hasattr(outcome, "to_dict") else {"outcome": str(outcome)}
        return self.write(
            PrimitiveTrace(
                index=self.next_index(),
                primitive="verify",
                gates=data.get("gates", {}),
                votes=data.get("votes", []),
                elapsed_s=elapsed_s,
                extra={
                    "status": data.get("status"),
                    "predicates": data.get("predicates", []),
                    "reason": data.get("reason", ""),
                    "depth_source": data.get("depth_source"),
                },
            )
        )


__all__ = ["PrimitiveTrace", "TraceWriter", "detection_summary"]

"""
On-disk Bayesian-updating calibration DB.

Stores per-(object_class, surface_class) parameter offsets that are applied
to perception output before primitives execute.  Updates use a Welford-style
running mean for numerical stability.

Schema (YAML)::

    calibrations:
      - object: "mug"
        surface: "flat_table"
        offset: {z: 0.012}
        n_obs: 47
        confidence: 0.91
        last_updated: "2026-05-04T18:23:00"

Confidence = ``n_obs / (n_obs + KAPPA)`` (KAPPA=5 by default).
"""
from __future__ import annotations

import dataclasses
import datetime as _dt
import math
import pathlib
from typing import Any, Dict, Iterable, Iterator, List, Optional

import yaml

DEFAULT_KAPPA = 5


@dataclasses.dataclass
class CalibrationEntry:
    object: str
    surface: str
    offset: Dict[str, float] = dataclasses.field(default_factory=dict)
    # Welford's sum of squared deviations per axis (for variance/stddev).
    _m2: Dict[str, float] = dataclasses.field(default_factory=dict)
    n_obs: int = 0
    n_success: int = 0
    last_updated: Optional[str] = None

    def observe(self, observed_offset: Dict[str, float], success: bool) -> None:
        """
        Welford running update for the mean of each axis.

        Failed observations still count toward ``n_obs`` (so confidence
        reflects total experience) and update the mean with the same sample;
        callers can gate on success themselves if they want success-only means.
        """
        self.n_obs += 1
        if success:
            self.n_success += 1
        for axis, x in observed_offset.items():
            x = float(x)
            mean = float(self.offset.get(axis, 0.0))
            m2 = float(self._m2.get(axis, 0.0))
            delta = x - mean
            mean = mean + delta / self.n_obs
            m2 = m2 + delta * (x - mean)
            self.offset[axis] = mean
            self._m2[axis] = m2
        self.last_updated = _dt.datetime.now().replace(microsecond=0).isoformat()

    def confidence(self, kappa: float = DEFAULT_KAPPA) -> float:
        denom = self.n_obs + kappa
        return self.n_obs / denom if denom else 0.0

    def stddev(self) -> Dict[str, float]:
        if self.n_obs < 2:
            return {axis: 0.0 for axis in self.offset}
        return {axis: math.sqrt(max(m2, 0.0) / (self.n_obs - 1))
                for axis, m2 in self._m2.items()}

    def success_rate(self) -> float:
        return self.n_success / self.n_obs if self.n_obs else 0.0

    def to_dict(self, kappa: float = DEFAULT_KAPPA) -> Dict[str, Any]:
        return {
            "object": self.object,
            "surface": self.surface,
            "offset": {k: round(float(v), 6) for k, v in self.offset.items()},
            "_m2": {k: round(float(v), 9) for k, v in self._m2.items()},
            "n_obs": int(self.n_obs),
            "n_success": int(self.n_success),
            "confidence": round(self.confidence(kappa), 4),
            "last_updated": self.last_updated,
        }

    @classmethod
    def from_dict(cls, d: Dict[str, Any]) -> "CalibrationEntry":
        return cls(
            object=str(d["object"]),
            surface=str(d["surface"]),
            offset={k: float(v) for k, v in (d.get("offset") or {}).items()},
            _m2={k: float(v) for k, v in (d.get("_m2") or {}).items()},
            n_obs=int(d.get("n_obs", 0)),
            n_success=int(d.get("n_success", 0)),
            last_updated=d.get("last_updated"),
        )


class CalibrationDB:
    """
    Mapping (object, surface) -> CalibrationEntry, with YAML persistence.
    """

    def __init__(self, path: Optional[pathlib.Path] = None,
                 kappa: float = DEFAULT_KAPPA) -> None:
        self.path = pathlib.Path(path) if path is not None else None
        self.kappa = kappa
        self._entries: Dict[tuple, CalibrationEntry] = {}

    @classmethod
    def load(cls, path: pathlib.Path | str,
             kappa: float = DEFAULT_KAPPA) -> "CalibrationDB":
        db = cls(pathlib.Path(path), kappa=kappa)
        if db.path and db.path.exists():
            data = yaml.safe_load(db.path.read_text()) or {}
            for row in data.get("calibrations", []) or []:
                if isinstance(row, dict):
                    e = CalibrationEntry.from_dict(row)
                    db._entries[(e.object, e.surface)] = e
        return db

    def save(self, path: Optional[pathlib.Path | str] = None) -> None:
        target = pathlib.Path(path) if path is not None else self.path
        if target is None:
            raise ValueError("CalibrationDB.save() needs a path (none configured).")
        target.parent.mkdir(parents=True, exist_ok=True)
        rows = [e.to_dict(self.kappa) for e in self._sorted_entries()]
        target.write_text(yaml.safe_dump({"calibrations": rows}, sort_keys=False))

    # Lookup / mutation

    def get(self, object: str, surface: str) -> Dict[str, float]:
        """
        Return the offset dict for (object, surface) or {} if unknown.
        """
        e = self._entries.get((object, surface))
        if e is None or e.n_obs == 0:
            return {}
        return {k: float(v) for k, v in e.offset.items()}

    def get_entry(self, object: str, surface: str) -> Optional[CalibrationEntry]:
        return self._entries.get((object, surface))

    def update(self, *, object: str, surface: str,
               observed_offset: Dict[str, float], success: bool = True
               ) -> CalibrationEntry:
        key = (object, surface)
        entry = self._entries.get(key) or CalibrationEntry(object=object, surface=surface)
        self._entries[key] = entry
        entry.observe(observed_offset, success=success)
        return entry

    # Enumeration

    def surfaces_for_object(self, object: str) -> List[str]:
        return sorted({s for (o, s) in self._entries if o == object})

    def objects_for_surface(self, surface: str) -> List[str]:
        return sorted({o for (o, s) in self._entries if s == surface})

    def __len__(self) -> int:
        return len(self._entries)

    def __iter__(self) -> Iterator[CalibrationEntry]:
        return iter(self._sorted_entries())

    def _sorted_entries(self) -> List[CalibrationEntry]:
        return [self._entries[k] for k in sorted(self._entries.keys())]

    # Pretty print

    def dump_pretty(self, axes: Iterable[str] = ("x", "y", "z")) -> str:
        axes = list(axes)
        header = (["object", "surface"] + [f"d{ax}(mm)" for ax in axes]
                  + ["n", "succ%", "conf"])
        rows: List[List[str]] = []
        for e in self._sorted_entries():
            row = [e.object, e.surface]
            for ax in axes:
                v = e.offset.get(ax)
                row.append(f"{1000.0 * v:+.1f}" if v is not None else "  -- ")
            row.append(str(e.n_obs))
            row.append(f"{100.0 * e.success_rate():.0f}")
            row.append(f"{e.confidence(self.kappa):.2f}")
            rows.append(row)
        widths = [len(h) for h in header]
        for r in rows:
            for i, c in enumerate(r):
                widths[i] = max(widths[i], len(c))
        fmt = lambda cells: "  ".join(c.ljust(widths[i]) for i, c in enumerate(cells))
        lines = [fmt(header), fmt(["-" * w for w in widths])]
        lines.extend(fmt(r) for r in rows)
        return "\n".join(lines)


__all__ = ["CalibrationDB", "CalibrationEntry", "DEFAULT_KAPPA"]

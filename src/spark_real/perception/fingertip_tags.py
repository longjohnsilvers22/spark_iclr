"""
Where the fingertips land in the wrist image.

The Robotiq 2F-85 and the wrist D435i share the flange, so for a given
aperture the fingertip midpoint sits at a FIXED pixel regardless of arm pose.
That pixel is the IBVS target (see control/wrist_servo.py). Two sources,
one interface:

1. AprilTags stuck on the fingers -- detected live with cv2.aruco, midpoint
   of the two tag centres. Most accurate, needs the tags in frame.
2. A measured aperture -> pixel table, linearly interpolated. The operator
   records it once from saved frames; no tags required.

Falls back table -> default pixel (image centre) so a caller always gets an
answer, with ``source`` and ``confidence`` saying which path produced it.

Usage::

    resolver = FingertipTargetResolver.from_calibration(wrist_cal, table=tbl)
    tgt = resolver.resolve(image=rgb, aperture=0.0)
    logger.info("fingertip target %s from %s", tgt.pixel, tgt.source)
"""

import json
import logging
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, Mapping, Optional, Sequence, Tuple

import cv2
import numpy as np
import yaml

logger = logging.getLogger(__name__)

DEFAULT_DICTIONARY = "DICT_APRILTAG_36h11"

SOURCE_TAGS = "apriltag"
SOURCE_TABLE = "aperture_table"
SOURCE_DEFAULT = "default_pixel"

# Confidence per source: tags are measured live, the table is a stale
# one-off measurement, the default pixel is a guess.
CONFIDENCE_TAGS = 1.0
CONFIDENCE_TABLE = 0.7
CONFIDENCE_DEFAULT = 0.2


@dataclass(frozen=True)
class FingertipTarget:
    """Resolved IBVS target pixel plus provenance."""

    u: float
    v: float
    source: str
    confidence: float
    detail: Dict[str, Any] = field(default_factory=dict)

    @property
    def pixel(self) -> Tuple[float, float]:
        return (self.u, self.v)

    def as_array(self) -> np.ndarray:
        return np.array([self.u, self.v], dtype=float)


@dataclass(frozen=True)
class ApertureTable:
    """
    Measured aperture -> fingertip-midpoint pixel, linearly interpolated.

    Aperture is the driver's normalized gripper position: 0.0 fully OPEN,
    1.0 fully CLOSED (matches UR10eDriver.set_gripper_position). Lookups
    outside the recorded range clamp to the nearest entry.
    """

    apertures: Tuple[float, ...]
    pixels: Tuple[Tuple[float, float], ...]

    def __len__(self) -> int:
        return len(self.apertures)

    @classmethod
    def from_entries(cls, entries: Sequence[Any]) -> "ApertureTable":
        """Entries are {aperture, u, v} mappings or (aperture, u, v) triples."""
        rows = []
        for entry in entries:
            if isinstance(entry, Mapping):
                rows.append((float(entry["aperture"]), float(entry["u"]), float(entry["v"])))
            else:
                a, u, v = entry
                rows.append((float(a), float(u), float(v)))
        if not rows:
            raise ValueError("aperture table needs at least one entry")
        rows.sort(key=lambda r: r[0])
        return cls(
            apertures=tuple(r[0] for r in rows),
            pixels=tuple((r[1], r[2]) for r in rows),
        )

    @classmethod
    def load(cls, path) -> "ApertureTable":
        """Load from .json or .yaml/.yml holding a list of entries."""
        p = Path(path)
        text = p.read_text()
        data = json.loads(text) if p.suffix == ".json" else yaml.safe_load(text)
        if isinstance(data, Mapping):
            data = data.get("fingertip_table", data.get("entries"))
        if data is None:
            raise ValueError(f"no fingertip table entries in {p}")
        return cls.from_entries(data)

    @classmethod
    def from_config(cls, cfg: Optional[Mapping]) -> Optional["ApertureTable"]:
        """Build from a config block, or None when the key is absent/empty."""
        if not cfg:
            return None
        entries = cfg.get("fingertip_table")
        if not entries:
            return None
        return cls.from_entries(entries)

    def lookup(self, aperture: float) -> Tuple[float, float]:
        """Clamped linear interpolation."""
        a = float(aperture)
        u = float(np.interp(a, self.apertures, [p[0] for p in self.pixels]))
        v = float(np.interp(a, self.apertures, [p[1] for p in self.pixels]))
        return (u, v)


def resolve_aruco_dictionary(name: str):
    """Name -> cv2.aruco predefined dictionary."""
    if not hasattr(cv2.aruco, name):
        raise ValueError(f"unknown cv2.aruco dictionary '{name}'")
    return cv2.aruco.getPredefinedDictionary(getattr(cv2.aruco, name))


class FingertipTargetResolver:
    """Resolves the fingertip-midpoint pixel from tags, table, or default."""

    def __init__(
        self,
        width: int,
        height: int,
        *,
        default_pixel: Optional[Sequence[float]] = None,
        table: Optional[ApertureTable] = None,
        dictionary: str = DEFAULT_DICTIONARY,
        tag_ids: Optional[Sequence[int]] = None,
    ):
        """
        Args:
            width, height: wrist image size; sets the default pixel when none
                is given (image centre).
            default_pixel: last-resort target, overriding the image centre.
            table: measured aperture -> pixel table, if the operator has one.
            dictionary: cv2.aruco predefined dictionary name.
            tag_ids: the two finger tag ids. When None, any frame containing
                exactly two markers is accepted.
        """
        self.width = int(width)
        self.height = int(height)
        if default_pixel is None:
            self.default_pixel = (self.width / 2.0, self.height / 2.0)
        else:
            self.default_pixel = (float(default_pixel[0]), float(default_pixel[1]))
        self.table = table
        self.dictionary_name = dictionary
        self.tag_ids = tuple(int(t) for t in tag_ids) if tag_ids else None
        if self.tag_ids is not None and len(self.tag_ids) != 2:
            raise ValueError(f"tag_ids must name exactly 2 tags, got {self.tag_ids}")

        self._dictionary = resolve_aruco_dictionary(dictionary)
        params = cv2.aruco.DetectorParameters()
        self._detector = cv2.aruco.ArucoDetector(self._dictionary, params)

    @classmethod
    def from_calibration(cls, calibration, **kwargs) -> "FingertipTargetResolver":
        """
        Build from a CameraCalibration.

        The default pixel becomes the PRINCIPAL POINT (cx, cy) -- the optical
        centre, which is what "straight ahead of the camera" means -- rather
        than the geometric centre of the frame.
        """
        kwargs.setdefault("default_pixel", (calibration.cx, calibration.cy))
        return cls(width=calibration.width, height=calibration.height, **kwargs)

    def detect_tag_centers(self, image: np.ndarray) -> Dict[int, Tuple[float, float]]:
        """Tag id -> centre pixel for every marker found in the image."""
        if image is None:
            return {}
        gray = image
        if image.ndim == 3:
            gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
        corners, ids, _ = self._detector.detectMarkers(gray)
        if ids is None or len(ids) == 0:
            return {}
        centers = {}
        for tag_id, quad in zip(ids.flatten(), corners):
            pts = np.asarray(quad).reshape(-1, 2)
            centers[int(tag_id)] = (float(pts[:, 0].mean()), float(pts[:, 1].mean()))
        return centers

    def resolve(
        self,
        image: Optional[np.ndarray] = None,
        aperture: Optional[float] = None,
    ) -> FingertipTarget:
        """
        Best available fingertip-midpoint pixel: tags, else table, else default.

        Args:
            image: wrist RGB. Omit to skip tag detection entirely.
            aperture: normalized gripper position (0 open .. 1 closed) for
                the table path. Omit to skip the table.
        """
        reason = "no image supplied"
        if image is not None:
            centers = self.detect_tag_centers(image)
            chosen = self._select_pair(centers)
            if chosen is not None:
                (id_a, pa), (id_b, pb) = chosen
                return FingertipTarget(
                    u=(pa[0] + pb[0]) / 2.0,
                    v=(pa[1] + pb[1]) / 2.0,
                    source=SOURCE_TAGS,
                    confidence=CONFIDENCE_TAGS,
                    detail={"tag_ids": [id_a, id_b], "centers": [pa, pb]},
                )
            reason = f"tags not resolvable (found ids {sorted(centers)})"

        if aperture is not None and self.table is not None:
            u, v = self.table.lookup(aperture)
            return FingertipTarget(
                u=u,
                v=v,
                source=SOURCE_TABLE,
                confidence=CONFIDENCE_TABLE,
                detail={"aperture": float(aperture), "fallback_reason": reason},
            )

        return FingertipTarget(
            u=self.default_pixel[0],
            v=self.default_pixel[1],
            source=SOURCE_DEFAULT,
            confidence=CONFIDENCE_DEFAULT,
            detail={"fallback_reason": reason, "has_table": self.table is not None},
        )

    def _select_pair(self, centers: Dict[int, Tuple[float, float]]):
        """The two finger tags, or None if they are not unambiguously present."""
        if self.tag_ids is not None:
            a, b = self.tag_ids
            if a in centers and b in centers:
                return (a, centers[a]), (b, centers[b])
            return None
        if len(centers) == 2:
            ids = sorted(centers)
            return (ids[0], centers[ids[0]]), (ids[1], centers[ids[1]])
        return None

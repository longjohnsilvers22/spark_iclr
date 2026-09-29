"""
Gemini Robotics-ER 2 annotation provider (pointing / boxes / trajectories).

Model ``gemini-robotics-er-2-preview`` (a Flash-tier embodied-reasoning
tune) answers spatial queries with coordinates in **[y, x] order on a
0-1000 grid**.  All axis handling is delegated to
``annotations.point_from_er2`` / ``box_from_er2``; nothing in this file
swaps coordinates by hand.

Uses ``client.models.generate_content``.  If a future SDK drops
generateContent for this model, add the interactions call behind the same
_call seam.

Config-gated, fail-open: no key / SDK / model access -> ``annotate``
returns [] and logs, never raises.  Preview-model caveat: the model id
is verified against ``models.list`` once per process, falling back
through ``_MODEL_FALLBACKS``.
"""

from __future__ import annotations

import json
import logging
import os
import re
from pathlib import Path
from typing import List, Optional

import numpy as np

from spark_real.perception.annotations import (
    Annotation,
    box_from_er2,
    point_from_er2,
    trace_from_er2,
)
from spark_real.utils.json_fence import strip_json_fence

logger = logging.getLogger(__name__)

try:
    from google import genai as _genai
    from google.genai import types as _gtypes
except ImportError:  # pragma: no cover - provider is optional
    _genai = None
    _gtypes = None

try:
    from PIL import Image
except ImportError:  # pragma: no cover
    Image = None

DEFAULT_MODEL = "gemini-robotics-er-2-preview"
# Older ER preview id, then plain Flash (which points worse but parses the
# same reply schema) - the probe reports which one actually answered.
_MODEL_FALLBACKS = (
    "gemini-robotics-er-2-preview",
    "gemini-robotics-er-1.5-preview",
)

_POINT_PROMPT = (
    'Point to the {query}. The answer should follow the json format: '
    '[{{"point": [y, x], "label": "<label>"}}]. '
    "The points are in [y, x] format normalized to 0-1000."
)

_BOX_PROMPT = (
    'Return a bounding box for the {query}. The answer should follow the '
    'json format: [{{"box_2d": [ymin, xmin, ymax, xmax], "label": "<label>"}}]. '
    "Coordinates are normalized to 0-1000."
)

_TRACE_PROMPT = (
    "Plan a 2D trajectory of about {n} waypoints for a robot gripper to "
    "accomplish: {query}. The first waypoint must start on the object being "
    "moved. The answer should follow the json format: "
    '[{{"point": [y, x], "label": "step 1"}}, ...] in path order. '
    "The points are in [y, x] format normalized to 0-1000."
)


def _load_api_key() -> str:
    """Same resolution order as SPARKPlanner: env var, then src/.gemini_api_key
    (second key in the file is the higher-quota one, so prefer it)."""
    env_key = os.environ.get("GEMINI_API_KEY")
    if env_key:
        return env_key.strip()
    key_file_env = os.environ.get("SPARK_GEMINI_KEY_FILE")
    key_file = (
        Path(key_file_env)
        if key_file_env
        else Path(__file__).resolve().parents[3] / ".gemini_api_key"
    )
    if key_file.exists():
        keys = [k.strip() for k in key_file.read_text().split("\n") if k.strip()]
        if len(keys) >= 2:
            return keys[1]
        if keys:
            return keys[0]
    return ""


class ER2Provider:
    """AnnotationProvider backed by Gemini Robotics-ER 2."""

    name = "er2"

    def __init__(self, model: str = DEFAULT_MODEL,
                 thinking_level: Optional[str] = "low",
                 trace_waypoints: int = 8):
        self.model = model
        # Docs: low thinking is the right latency/accuracy trade for spatial
        # queries (high is for counting). Passed only if the SDK supports it.
        self.thinking_level = thinking_level
        self.trace_waypoints = trace_waypoints
        self._client = None
        self._resolved_model: Optional[str] = None

    # -- protocol ----------------------------------------------------------

    def available(self) -> bool:
        return _genai is not None and Image is not None and bool(_load_api_key())

    def annotate(self, image: np.ndarray, query: str,
                 kind: str = "point") -> List[Annotation]:
        if not self.available():
            logger.warning("ER2Provider unavailable (sdk or key missing)")
            return []
        try:
            if kind == "point":
                prompt = _POINT_PROMPT.format(query=query)
            elif kind == "box":
                prompt = _BOX_PROMPT.format(query=query)
            elif kind == "trace":
                prompt = _TRACE_PROMPT.format(query=query,
                                              n=self.trace_waypoints)
            else:
                logger.warning("ER2Provider: unknown kind %r", kind)
                return []
            text = self._call(image, prompt)
            if not text:
                return []
            return self._parse(text, kind, query)
        except Exception as exc:  # noqa: BLE001 - provider must fail open
            logger.warning("ER2Provider.annotate(%r, %r) failed: %s",
                           query, kind, exc)
            return []

    # -- internals ---------------------------------------------------------

    def _get_client(self):
        if self._client is None:
            # Same request deadline as the planner: a stalled Gemini
            # connection must fail the annotate call, never hang the
            # perception thread.
            from spark_real.planning.spark_planner import GEMINI_TIMEOUT_MS

            self._client = _genai.Client(
                api_key=_load_api_key(),
                http_options={"timeout": GEMINI_TIMEOUT_MS},
            )
        return self._client

    def _resolve_model(self) -> str:
        """Verify the requested model against models.list once per instance."""
        if self._resolved_model is not None:
            return self._resolved_model
        requested = self.model
        candidates = [requested] + [m for m in _MODEL_FALLBACKS if m != requested]
        try:
            listed = {m.name.split("/", 1)[-1]
                      for m in self._get_client().models.list()}
        except Exception as exc:  # noqa: BLE001 - listing is best-effort
            logger.warning("ER2Provider: models.list failed (%s); "
                           "assuming %r works", exc, requested)
            self._resolved_model = requested
            return requested
        for name in candidates:
            if name in listed:
                if name != requested:
                    logger.warning("ER2Provider: %r not listed; using %r",
                                   requested, name)
                self._resolved_model = name
                return name
        logger.warning("ER2Provider: none of %r listed; trying %r anyway",
                       candidates, requested)
        self._resolved_model = requested
        return requested

    def _call(self, image: np.ndarray, prompt: str) -> str:
        model = self._resolve_model()
        cfg_kwargs = {"temperature": 0}
        if self.thinking_level is not None:
            # thinking_config exists on newer SDKs only; drop it silently
            # elsewhere -- the answer schema is the same either way.
            try:
                cfg_kwargs["thinking_config"] = _gtypes.ThinkingConfig(
                    thinking_level=self.thinking_level)
            except Exception:  # noqa: BLE001
                pass
        try:
            config = _gtypes.GenerateContentConfig(**cfg_kwargs)
        except Exception:  # noqa: BLE001 - e.g. thinking_level value rejected
            config = _gtypes.GenerateContentConfig(temperature=0)
        resp = self._get_client().models.generate_content(
            model=model,
            contents=[prompt, Image.fromarray(np.asarray(image))],
            config=config,
        )
        return (resp.text or "").strip()

    def _parse(self, text: str, kind: str, query: str) -> List[Annotation]:
        try:
            data = json.loads(strip_json_fence(text))
        except json.JSONDecodeError:
            # Some replies wrap the JSON in prose; grab the first array.
            m = re.search(r"\[.*\]", text, flags=re.DOTALL)
            if not m:
                logger.warning("ER2Provider: unparseable reply %r", text[:200])
                return []
            try:
                data = json.loads(m.group(0))
            except json.JSONDecodeError:
                logger.warning("ER2Provider: unparseable reply %r", text[:200])
                return []
        if isinstance(data, dict):
            data = [data]
        if not isinstance(data, list):
            return []

        out: List[Annotation] = []
        if kind == "trace":
            pts = [d["point"] for d in data
                   if isinstance(d, dict) and self._valid_pair(d.get("point"))]
            if len(pts) >= 2:
                ann = Annotation(kind="trace", points=trace_from_er2(pts),
                                 provider=self.name, label=query,
                                 raw={"reply": data})
                if ann.in_bounds():
                    out.append(ann.clamped())
            return out

        for d in data:
            if not isinstance(d, dict):
                continue
            label = str(d.get("label", query))
            if kind == "point" and self._valid_pair(d.get("point")):
                ann = Annotation(kind="point",
                                 points=[point_from_er2(d["point"])],
                                 provider=self.name, label=label,
                                 raw={"reply": d})
            elif kind == "box" and self._valid_quad(d.get("box_2d")):
                ann = Annotation(kind="box", points=box_from_er2(d["box_2d"]),
                                 provider=self.name, label=label,
                                 raw={"reply": d})
            else:
                continue
            if ann.in_bounds():
                out.append(ann.clamped())
        return out

    @staticmethod
    def _valid_pair(v) -> bool:
        return (isinstance(v, (list, tuple)) and len(v) == 2
                and all(isinstance(c, (int, float)) for c in v))

    @staticmethod
    def _valid_quad(v) -> bool:
        return (isinstance(v, (list, tuple)) and len(v) == 4
                and all(isinstance(c, (int, float)) for c in v))

"""
Thin HTTP client for the SAM3 FastAPI service.

Mirrors ``cap_gym/capx/integrations/vision/sam3.py`` but talks to the
SPARK service at ``127.0.0.1:8115`` (CaP-X owns 8114).  The wrapper does
not import torch / sam3, so worker processes that use
``cfg.use_sam3_service=True`` stay lightweight and never load the local
model.

On any HTTP failure the client logs a warning and returns empty results
so a benchmark sweep does not crash because the service is down.

Public API
* ``init_sam3_client(service_url, timeout)`` returns a ``Sam3ServiceClient``
  with two methods:
    - ``segment_text(image, text_prompt) -> list[dict]``
    - ``segment_point(image, point_coords) -> dict``
"""

from __future__ import annotations

import base64
import io
import logging
import time
from collections.abc import Sequence
from typing import Any, Optional

import numpy as np
import requests
from PIL import Image

logger = logging.getLogger(__name__)

DEFAULT_SERVICE_URL = "http://127.0.0.1:8115"


# Encoding helpers (no torch import)

def _encode_image(image: Any) -> str:
    """
    PNG-encode the image and return base64. Accepts numpy/PIL/tensor-like.
    """
    if isinstance(image, np.ndarray):
        arr = image
        if arr.dtype != np.uint8:
            arr = np.clip(arr, 0, 255).astype(np.uint8)
        pil = Image.fromarray(arr).convert("RGB")
    elif isinstance(image, Image.Image):
        pil = image.convert("RGB")
    else:
        # Fall back via numpy conversion (covers torch tensors w/ .numpy()).
        arr = np.asarray(image)
        if arr.dtype != np.uint8:
            arr = np.clip(arr, 0, 255).astype(np.uint8)
        pil = Image.fromarray(arr).convert("RGB")
    buf = io.BytesIO()
    pil.save(buf, format="PNG")
    return base64.b64encode(buf.getvalue()).decode("utf-8")


def _decode_mask(mask_b64: str, shape: Sequence[int]) -> np.ndarray:
    if not mask_b64:
        return np.zeros(tuple(shape) if shape else (0, 0), dtype=bool)
    raw = base64.b64decode(mask_b64)
    arr = np.frombuffer(raw, dtype=np.uint8).reshape(tuple(shape))
    return arr.astype(bool)


# Retry helper (inlined so the client has zero CaP-X imports)

def _post_with_retries(url: str, payload: dict, *, timeout: float,
                        max_retries: int = 3) -> Optional[dict]:
    """
    POST with bounded retries.  Returns parsed JSON or ``None`` on failure.
    """
    interval = 0.5
    last_err: Optional[Exception] = None
    for _ in range(max_retries):
        try:
            resp = requests.post(url, json=payload, timeout=timeout)
            resp.raise_for_status()
            return resp.json()
        except requests.RequestException as exc:
            last_err = exc
            time.sleep(interval)
            interval = min(interval * 2.0, 4.0)
    logger.warning("SAM3 service POST %s failed after %d retries: %s",
                   url, max_retries, last_err)
    return None


# Client object

class Sam3ServiceClient:
    """
    Stateless client for the SAM3 FastAPI service.

    Methods return the same shape the in-process ``Sam3Processor`` callers
    expect downstream:
      * ``segment_text`` -> ``list[{mask, box, score, label}]`` (mask is
        boolean numpy array).
      * ``segment_point`` -> ``{mask, box, score}`` or ``None`` on failure.
    """

    def __init__(self, service_url: str = DEFAULT_SERVICE_URL,
                 timeout: float = 60.0):
        self.url = service_url.rstrip("/")
        self.timeout = float(timeout)

    # text prompt

    def segment_text(self, image: Any, text_prompt: str) -> list[dict[str, Any]]:
        encoded = _encode_image(image)
        payload = {"image_base64": encoded, "text_prompt": text_prompt}
        data = _post_with_retries(f"{self.url}/segment", payload,
                                    timeout=self.timeout)
        if not data or not isinstance(data, dict):
            return []
        results: list[dict[str, Any]] = []
        for item in data.get("results", []):
            try:
                shape = tuple(item["shape"])
                mask = _decode_mask(item["mask_base64"], shape)
                results.append({
                    "mask": mask,
                    "box": list(item.get("box", [0.0, 0.0, 0.0, 0.0])),
                    "score": float(item.get("score", 0.0)),
                    "label": str(item.get("label", text_prompt)),
                })
            except Exception as exc:
                logger.warning("Failed to decode SAM3 result: %s", exc)
                continue
        return results

    # point prompt

    def segment_point(self, image: Any,
                       point_coords: Sequence[float]) -> Optional[dict[str, Any]]:
        if (not hasattr(point_coords, "__len__")) or len(point_coords) != 2:
            raise ValueError(
                f"point_coords must be (x, y); got {point_coords!r}")
        encoded = _encode_image(image)
        payload = {
            "image_base64": encoded,
            "point_coords": [float(point_coords[0]), float(point_coords[1])],
        }
        data = _post_with_retries(f"{self.url}/point_prompt", payload,
                                    timeout=self.timeout)
        if not data or not isinstance(data, dict):
            return None
        shape = tuple(data.get("shape", (0, 0)))
        mask = _decode_mask(data.get("mask_base64", ""), shape)
        if mask.size == 0 or not mask.any():
            return None
        return {
            "mask": mask,
            "box": list(data.get("box", [0.0, 0.0, 0.0, 0.0])),
            "score": float(data.get("score", 0.0)),
        }

    # health

    def is_alive(self) -> bool:
        try:
            resp = requests.get(f"{self.url}/health", timeout=2.0)
            resp.raise_for_status()
            data = resp.json()
            return bool(data.get("status") == "ok"
                         and data.get("model_loaded", False))
        except requests.RequestException:
            return False


def init_sam3_client(service_url: str = DEFAULT_SERVICE_URL,
                       timeout: float = 60.0) -> Sam3ServiceClient:
    """
    Instantiate a SAM3 service client. Does not block on a health check.
    """
    return Sam3ServiceClient(service_url=service_url, timeout=timeout)


__all__ = [
    "DEFAULT_SERVICE_URL",
    "Sam3ServiceClient",
    "init_sam3_client",
]

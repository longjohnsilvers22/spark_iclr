"""
MolmoAct annotation provider -- local pointing via the molmoact2 conda env.

Two transports, both fail-open:

* ``mode='subprocess'`` (default): keeps ``molmo_worker.py`` resident in
  the ``molmoact2`` env (``~/miniconda3/envs/molmoact2/bin/python``) and
  talks JSONL over stdin/stdout; a persistent worker because the
  checkpoint (allenai/MolmoAct2-LIBERO, bf16 ~16 GB) is too slow to
  reload per query.
* ``mode='http'``: an OpenAI-compatible chat endpoint serving a Molmo
  model (the cap_gym vision integration's local vLLM at
  http://127.0.0.1:8122/v1).  No GPU claim from this process at all.

Reply parsing handles the Molmo family's point formats (cap_gym's
integrations/vision/molmo.py is the reference):

* Molmo2 / MolmoAct2: ``<points coords="t i x y i x y ...">label</points>``
  with x, y on a 0-1000 grid, **x first**;
* Molmo1: ``<point x="X" y="Y">`` on a 0-100 grid;
* plain ``x, y`` pairs (0-100) as a last resort.

Axis order note: Molmo answers (x, y) column-first -- the OPPOSITE of
ER2's [y, x].  Both are normalized through the tested converters in
``annotations.py``; nothing here swaps by hand.
"""

from __future__ import annotations

import base64
import io
import json
import logging
import os
import re
import subprocess
import tempfile
from pathlib import Path
from typing import List, Optional, Tuple

import numpy as np

from spark_real.perception.annotations import (
    MOLMO1_SCALE,
    ER2_SCALE,
    Annotation,
    point_from_molmo,
)

logger = logging.getLogger(__name__)

_DEFAULT_ENV_PYTHON = str(
    Path.home() / "miniconda3" / "envs" / "molmoact2" / "bin" / "python"
)
_WORKER = str(Path(__file__).resolve().parent / "molmo_worker.py")
_DEFAULT_SERVICE_URL = "http://127.0.0.1:8122/v1"
# HF cache presence check -- available() must not trigger a 16 GB download.
_CKPT_CACHE_DIR = (
    Path.home() / ".cache" / "huggingface" / "hub" / "models--allenai--MolmoAct2-LIBERO"
)


def parse_molmo_points(text: str) -> Tuple[List[Tuple[float, float]], float]:
    """
    Extract ``[(x, y), ...]`` and their grid scale from a Molmo reply.

    Returns ``(points, scale)`` with scale 1000.0 for the Molmo2 coords
    format and 100.0 for the older tag formats.  Empty list when the
    reply carries no parseable points (prose refusals, action tokens).
    """
    # Molmo2: <points coords="type (idx x y)*">label</points>, 0-1000 grid.
    m = re.search(r'<points\s+coords\s*=\s*["\']([^"\']+)["\']', text,
                  flags=re.IGNORECASE)
    if m:
        try:
            nums = [float(n) for n in m.group(1).split()]
        except ValueError:
            nums = []
        pts = []
        i = 1  # skip the leading type indicator, then (idx, x, y) triplets
        while i + 2 < len(nums):
            pts.append((nums[i + 1], nums[i + 2]))
            i += 3
        pts = [(x, y) for x, y in pts if 0.0 <= x <= 1000.0 and 0.0 <= y <= 1000.0]
        if pts:
            return pts, ER2_SCALE

    # Molmo1: one or more <point x=".." y=".."> tags, 0-100 grid.
    pts = []
    for tag in re.findall(r"<point\b[^>]*>", text, flags=re.IGNORECASE):
        mx = re.search(r"\bx\s*=\s*['\"]([0-9]*\.?[0-9]+)['\"]", tag, re.IGNORECASE)
        my = re.search(r"\by\s*=\s*['\"]([0-9]*\.?[0-9]+)['\"]", tag, re.IGNORECASE)
        if mx and my:
            pts.append((float(mx.group(1)), float(my.group(1))))
    pts = [(x, y) for x, y in pts if 0.0 <= x <= 100.0 and 0.0 <= y <= 100.0]
    if pts:
        return pts, MOLMO1_SCALE

    # Legacy <points x1=".." y1=".." ...> tag, 0-100 grid.
    m = re.search(r"<points\b[^>]*>", text, flags=re.IGNORECASE)
    if m:
        src = m.group(0)
        xs = {int(i): float(v) for i, v in
              re.findall(r"x(\d+)\s*=\s*['\"]([0-9]*\.?[0-9]+)['\"]", src)}
        ys = {int(i): float(v) for i, v in
              re.findall(r"y(\d+)\s*=\s*['\"]([0-9]*\.?[0-9]+)['\"]", src)}
        pts = [(xs[i], ys[i]) for i in sorted(set(xs) & set(ys))
               if 0.0 <= xs[i] <= 100.0 and 0.0 <= ys[i] <= 100.0]
        if pts:
            return pts, MOLMO1_SCALE

    # Bare "x, y" pairs (0-100). Last resort - prose with numbers in it
    # can false-positive here, which is why this rung comes last.
    pts = [(float(x), float(y)) for x, y in
           re.findall(r"([0-9]*\.?[0-9]+)\s*,\s*([0-9]*\.?[0-9]+)", text)]
    pts = [(x, y) for x, y in pts if 0.0 <= x <= 100.0 and 0.0 <= y <= 100.0]
    return pts, MOLMO1_SCALE


class MolmoProvider:
    """AnnotationProvider backed by a locally-run MolmoAct / Molmo model."""

    name = "molmo"

    def __init__(self, mode: str = "subprocess",
                 env_python: str = _DEFAULT_ENV_PYTHON,
                 service_url: str = _DEFAULT_SERVICE_URL,
                 timeout_s: float = 300.0):
        self.mode = mode
        self.env_python = env_python
        self.service_url = service_url
        # First serve-mode query pays the full checkpoint load, hence the
        # generous default; later queries answer in seconds.
        self.timeout_s = timeout_s
        self._proc: Optional[subprocess.Popen] = None

    # -- protocol ----------------------------------------------------------

    def available(self) -> bool:
        if self.mode == "http":
            return True  # liveness is checked (and failed open) per call
        return os.path.exists(self.env_python) and _CKPT_CACHE_DIR.exists()

    def annotate(self, image: np.ndarray, query: str,
                 kind: str = "point") -> List[Annotation]:
        if kind not in ("point", "trace"):
            logger.warning("MolmoProvider: kind %r unsupported", kind)
            return []
        if not self.available():
            logger.warning("MolmoProvider unavailable (env or checkpoint missing)")
            return []
        try:
            if self.mode == "http":
                text = self._ask_http(image, query, kind)
            else:
                text = self._ask_subprocess(image, query, kind)
        except Exception as exc:  # noqa: BLE001 - provider must fail open
            logger.warning("MolmoProvider.annotate(%r) failed: %s", query, exc)
            return []
        if not text:
            return []
        pts, scale = parse_molmo_points(text)
        if not pts:
            logger.warning("MolmoProvider: no points in reply %r", text[:200])
            return []
        norm = [point_from_molmo(p, scale) for p in pts]
        if kind == "trace" and len(norm) >= 2:
            anns = [Annotation(kind="trace", points=norm, provider=self.name,
                               label=query, raw={"text": text})]
        else:
            anns = [Annotation(kind="point", points=[p], provider=self.name,
                               label=query, raw={"text": text})
                    for p in norm]
        return [a.clamped() for a in anns if a.in_bounds()]

    def close(self) -> None:
        if self._proc is not None:
            try:
                self._proc.stdin.close()
                self._proc.wait(timeout=10)
            except Exception:  # noqa: BLE001
                self._proc.kill()
            self._proc = None

    # -- subprocess transport ---------------------------------------------

    def _ensure_worker(self) -> subprocess.Popen:
        if self._proc is not None and self._proc.poll() is None:
            return self._proc
        logger.info("MolmoProvider: starting worker in %s", self.env_python)
        self._proc = subprocess.Popen(
            [self.env_python, _WORKER, "--serve"],
            stdin=subprocess.PIPE, stdout=subprocess.PIPE,
            stderr=subprocess.DEVNULL, text=True,
        )
        return self._proc

    def _ask_subprocess(self, image: np.ndarray, query: str, kind: str) -> str:
        from PIL import Image as PILImage

        with tempfile.NamedTemporaryFile(suffix=".png", delete=False) as f:
            img_path = f.name
        try:
            PILImage.fromarray(np.asarray(image)).save(img_path)
            proc = self._ensure_worker()
            req = json.dumps({"image": img_path, "query": query, "kind": kind})
            proc.stdin.write(req + "\n")
            proc.stdin.flush()
            line = proc.stdout.readline()
            if not line:
                raise RuntimeError("molmo worker died (empty reply)")
            res = json.loads(line)
            if not res.get("ok"):
                raise RuntimeError(f"molmo worker error: {res.get('error')}")
            return str(res.get("text", ""))
        finally:
            try:
                os.unlink(img_path)
            except OSError:
                pass

    # -- http transport (OpenAI-compatible, cap_gym vLLM pattern) ----------

    def _ask_http(self, image: np.ndarray, query: str, kind: str) -> str:
        import requests
        from PIL import Image as PILImage

        buf = io.BytesIO()
        PILImage.fromarray(np.asarray(image)).save(buf, format="PNG")
        b64 = base64.b64encode(buf.getvalue()).decode()
        prompt = (f"Point to the {query}." if kind == "point" else
                  f"Show the visual trace of the gripper to accomplish: {query}.")
        resp = requests.post(
            f"{self.service_url}/chat/completions",
            json={
                "model": "molmo",
                "messages": [{
                    "role": "user",
                    "content": [
                        {"type": "image_url",
                         "image_url": {"url": f"data:image/png;base64,{b64}"}},
                        {"type": "text", "text": prompt},
                    ],
                }],
                "max_tokens": 256,
                "temperature": 0,
            },
            timeout=self.timeout_s,
        )
        resp.raise_for_status()
        return resp.json()["choices"][0]["message"]["content"]

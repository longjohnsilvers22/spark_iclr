"""
Runtime control of the second-opinion box proposer (RF-DETR).

  read back    GET  /api/perception/fusion_status
  switch       POST /api/perception/proposer_mode   {"mode": "auto"|"always"|"off"}
               POST /api/perception/proposer_mode   {"mode": null}  -> back to config

Three modes, defined in pipeline_perception:

  auto    default, = the shipped YAML. The proposer is consulted only for the
          labels it has earned an opinion on (perception.fusion.proposer.labels).
  always  the allowlist is dropped; every in-vocabulary label is scored.
  off     nobody is consulted. The fusion gate itself STAYS UP: its
          mask-quality channel is what refuses a wrist rotation off a
          meaningless OBB, and it is not this switch's business.

Same shape as the BT cache toggle (routes/bt_library.py + routes/state.py):
route state holds the override, ``None`` means fall back to config, and the
pipeline reads it back through its own resolver. The one addition is cache
invalidation -- ``detection_gate()`` memoises the gate, so a toggle that does
not call ``reset_detection_gate()`` silently does nothing until the next
restart.
"""

from __future__ import annotations

import logging
from typing import Optional

from fastapi import APIRouter
from fastapi.responses import JSONResponse
from pydantic import BaseModel

from spark_real.pipeline_perception import PROPOSER_MODES, normalize_proposer_mode
from spark_real.routes import state

logger = logging.getLogger(__name__)
router = APIRouter()


class ProposerModeBody(BaseModel):
    # None is meaningful: it clears the override rather than selecting a mode.
    mode: Optional[str] = None


def _offline_status():
    """Status shape when no pipeline exists yet, so the UI can still render."""
    return {
        "available": False,
        "mode": None,
        "override": state.proposer_mode,
        "source": "runtime" if state.proposer_mode else "config",
        "modes": list(PROPOSER_MODES),
        "fusion_enabled": None,
        "gate_state": "no_pipeline",
        "backend": None,
        "loaded_backend": None,
        "labels": [],
        "error": None,
    }


@router.get("/api/perception/fusion_status")
async def fusion_status():
    """Which mode is live, where it came from, and what is behind the gate.

    Read-only and cheap: it resolves config, it never builds a detector.
    """
    pipe = state.pipeline
    if pipe is None:
        return _offline_status()
    out = dict(pipe.fusion_status())
    out["available"] = True
    return out


@router.post("/api/perception/proposer_mode")
async def set_proposer_mode(body: ProposerModeBody):
    """Switch the proposer mode for every subsequent detect. No restart.

    The cached gate is dropped here and only here: without that the new mode
    would not reach the live detect path at all.
    """
    try:
        mode = normalize_proposer_mode(body.mode)
    except ValueError as exc:
        return JSONResponse({"error": str(exc)}, status_code=400)

    state.proposer_mode = mode
    pipe = state.pipeline
    if pipe is not None:
        pipe.reset_detection_gate()
    logger.info(
        "proposer_mode: override = %s (gate cache %s)",
        mode or "cleared (config)",
        "dropped" if pipe is not None else "no pipeline",
    )
    return await fusion_status()

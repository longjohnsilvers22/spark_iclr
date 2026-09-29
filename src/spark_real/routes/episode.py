"""
Episode bundle + BT library routes.

Episode bundle endpoints:
* ``POST /api/episode/save``: moves the cache folder to
  ``episodes_kept/<task_name>_<id>/`` and copies in any per-camera
  videos. Body: ``{task_name: "..."}``.
* ``POST /api/episode/discard``: wipes the cache. Idempotent.

BT library endpoints:
* ``GET /api/library/list``: all BT entries with success counts and
  ``promoted`` flag, sorted promoted-first then by success descending.
* ``POST /api/library/promote``: toggle promotion for one entry.
  Body: ``{hash: "...", promoted: true|false}``.

Promotion semantics: a promoted BT is always injected into the
planner's few-shot context (no jaccard-similarity gate), so the LLM
treats it as a canonical recipe for related instructions. The episode
bundle and library promote are independent: you can promote a
pattern without saving its full bundle, and save a failure bundle for
a pattern you'd never promote.
"""

from __future__ import annotations

import logging
from typing import Optional

from fastapi import APIRouter
from fastapi.responses import JSONResponse
from pydantic import BaseModel

from spark_real.routes import state

logger = logging.getLogger("spark_server")
router = APIRouter()


class SaveEpisodeRequest(BaseModel):
    # PASS/FAIL is recorded, never inferred. Task-success predicates are not
    # reliable enough yet to label a corpus with, and a FAILED episode is not
    # rubbish -- it is the negative half a verifier study needs and the
    # expensive half to collect deliberately. So a failed run is SAVED with
    # outcome="fail", not discarded.
    outcome: Optional[str] = None  # "pass" | "fail" | None (unlabelled)
    task_name: Optional[str] = None


class PromoteRequest(BaseModel):
    hash: str
    promoted: bool = True


@router.post("/api/episode/save")
async def episode_save(req: SaveEpisodeRequest):
    rec = state.current_episode
    if rec is None:
        return JSONResponse(
            status_code=404, content={"error": "No cached episode to save"}
        )
    task_name = req.task_name or rec.instruction or "episode"
    outcome = (req.outcome or "").strip().lower() or None
    if outcome not in (None, "pass", "fail"):
        return JSONResponse(
            status_code=400,
            content={"error": f"outcome must be pass/fail, got {outcome!r}"},
        )
    try:
        kept = rec.save_as(task_name, outcome=outcome)
    except Exception as exc:
        logger.error("episode_save failed: %s", exc)
        return JSONResponse(status_code=500, content={"error": str(exc)})
    summary = rec.summary()
    summary["kept_path"] = str(kept)
    summary["outcome"] = outcome
    logger.info("episode_save: %r outcome=%s -> %s", task_name, outcome, kept)
    return {"saved": True, "outcome": outcome, "episode": summary}


@router.post("/api/episode/discard")
async def episode_discard():
    rec = state.current_episode
    if rec is None:
        return {"discarded": False, "reason": "no cached episode"}
    try:
        rec.discard()
    except Exception as exc:
        logger.warning("episode_discard failed: %s", exc)
    state.current_episode = None
    return {"discarded": True}


# Demonstration recording lives in routes/vla_record.py (/api/vla_record/*).
# The /api/episode/* bundle below stores bt.yaml + result.json beside a run,
# which the VLA episode format deliberately does not carry.


@router.get("/api/library/list")
async def library_list():
    pipeline = state.pipeline
    if pipeline is None or pipeline._bt_library is None:
        return {"entries": []}
    return {"entries": pipeline._bt_library.list_all()}


@router.post("/api/library/promote")
async def library_promote(req: PromoteRequest):
    pipeline = state.pipeline
    if pipeline is None or pipeline._bt_library is None:
        return JSONResponse(
            status_code=400, content={"error": "BT library not initialised"}
        )
    entry = pipeline._bt_library.set_promoted(req.hash, req.promoted)
    if entry is None:
        return JSONResponse(
            status_code=404, content={"error": f"No BT entry with hash " f"{req.hash}"}
        )
    return {
        "hash": entry.hash,
        "promoted": entry.promoted,
        "instruction": entry.instruction,
    }

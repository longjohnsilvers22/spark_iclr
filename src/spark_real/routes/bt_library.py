"""
BT library inspection, curation and runtime overrides.

This is the operator's control surface for the capture-once-reuse-forever
loop: run a task with the planner on, watch it succeed, and every later run
of that task is served from here with no LLM call.

  browse / inspect   GET  /api/bt/list, /api/bt/get
  will it hit?       GET  /api/bt/resolve?instruction=...   (dry run, no
                          planning, no robot -- answers "is this task
                          cached?" before you commit to a session)
  offline switch     POST /api/bt/toggle_cache        (PERSISTED)
  bind a task        POST /api/bt/pin  / /api/bt/unpin (PERSISTED, per-task,
                          not consumed -- 30 consecutive runs serve the
                          same tree)
  one-off override   POST /api/bt/select              (single-shot, legacy)
  curate             POST /api/bt/promote, /api/bt/alias,
                     DELETE /api/bt/by_instruction, DELETE /api/bt/{hash}
  freeze / restore   GET  /api/bt/export, POST /api/bt/import,
                     POST /api/bt/reload_seeds

The pin map and the cache toggle live in ``<library>/runtime_state.json``,
so they survive a server restart. The previous in-process globals silently
reverted to the ``SPARK_DISABLE_LLM`` env default on every bounce, which
mid-collection meant a run the operator believed was offline quietly
called Gemini.
"""

from __future__ import annotations

import datetime as _dt
import logging
from typing import Any, Dict, List, Optional

import yaml
from fastapi import APIRouter
from fastapi.responses import JSONResponse, PlainTextResponse
from pydantic import BaseModel

from spark_real.planning.bt_seeds import (
    SeedError,
    dump_seed,
    ensure_seeded,
    parse_seed,
    seed_library,
)
from spark_real.routes import state

logger = logging.getLogger(__name__)
router = APIRouter()


# helpers


def _lib():
    """
    The live BTLibrary, seeded on first use.

    Seeding here (rather than only at boot) means a fresh checkout serves
    the packaged demo-task trees from the very first request, even if the
    pipeline was constructed before the seed directory existed.
    """
    pipe = state.pipeline
    if pipe is None:
        return None
    lib = getattr(pipe, "_bt_library", None)
    if lib is None:
        return None
    # Honour bt.seed_on_start here too. This runs on ANY /api/bt/* request, so
    # without the guard a library that was deliberately cleared for a fresh
    # collection run silently re-seeds itself the moment the UI reads it.
    if getattr(pipe.config, "bt_seed_on_start", True):
        try:
            ensure_seeded(lib, seed_dir=getattr(pipe.config, "bt_seed_dir", None))
        except Exception:
            logger.exception("bt: seed install failed; serving the library as-is")
    return lib


def _no_lib():
    return JSONResponse(
        status_code=503,
        content={"error": "pipeline not initialised"},
    )


def _count_actions(score: Any) -> int:
    """
    Leaf action nodes in a BT score, so the list view can show
    'this is a 14-step plan' at a glance.
    """
    if not isinstance(score, dict):
        return 0
    tree = score.get("tree") if "tree" in score else score
    count = 0

    def walk(node):
        nonlocal count
        if not isinstance(node, dict):
            return
        kids = node.get("children")
        if isinstance(kids, list) and kids:
            for c in kids:
                walk(c)
        else:
            count += 1

    walk(tree)
    return count


def _created_at_for(lib, h: str) -> Optional[str]:
    """
    Entries carry no explicit created_at, so fall back to file mtime.
    """
    try:
        p = lib.root / f"{h}.json"
        if not p.exists():
            return None
        return _dt.datetime.utcfromtimestamp(p.stat().st_mtime).isoformat() + "Z"
    except OSError:
        return None


def _row(lib, entry, pinned_hashes) -> Dict[str, Any]:
    return {
        "hash": entry.hash,
        "instruction": entry.instruction,
        "aliases": list(entry.aliases or []),
        "n_actions": _count_actions(entry.score),
        "n_objects": len(entry.objects or []),
        "objects": list(entry.objects or []),
        "promoted": bool(entry.promoted),
        "seed": bool(entry.seed),
        "priority": int(entry.priority),
        "verified": bool(entry.verified),
        "notes": entry.notes,
        "success": int(entry.success),
        "fail": int(entry.fail),
        "pinned": entry.hash in pinned_hashes,
        "created_at": _created_at_for(lib, entry.hash),
    }


# inspection


@router.get("/api/bt/list")
async def bt_list():
    """
    Flat index for the frontend panel. Seeds first, then promoted, then by
    success count -- the order the resolver itself prefers.
    """
    lib = _lib()
    if lib is None:
        return {"entries": [], "error": "pipeline not initialised"}
    pinned_hashes = set(lib.pins().values())
    rows = [_row(lib, e, pinned_hashes) for e in lib.entries()]
    rows.sort(
        key=lambda r: (
            not r["seed"],
            not r["promoted"],
            -r["success"],
            r["fail"],
            r["hash"],
        )
    )
    return {
        "entries": rows,
        "count": len(rows),
        "stats": lib.stats(),
        "pins": lib.pins(),
        "pinned": state.pinned_bt_hash,
    }


@router.get("/api/bt/get")
async def bt_get(hash: str):
    """
    Full entry including the tree, for inspection before pinning.
    """
    lib = _lib()
    if lib is None:
        return _no_lib()
    entry = lib.get(hash)
    if entry is None:
        return JSONResponse(status_code=404, content={"error": f"no such BT: {hash}"})
    row = _row(lib, entry, set(lib.pins().values()))
    row["score"] = entry.score
    return row


@router.get("/api/bt/resolve")
async def bt_resolve(instruction: str, instructions: Optional[str] = None):
    """
    Dry-run the resolver. No planning, no detection, no robot.

    The single most useful pre-flight check before a collection session:
    it answers "will this task be served from cache, or will it call the
    planner?" for every task, in one request. Pass several tasks at once as
    a newline- or ``|``-separated ``instructions`` string.
    """
    lib = _lib()
    if lib is None:
        return _no_lib()
    wanted: List[str] = []
    for blob in (instruction, instructions):
        if not blob:
            continue
        for part in str(blob).replace("|", "\n").split("\n"):
            if part.strip():
                wanted.append(part.strip())

    pipe = state.pipeline
    mode = pipe.plan_mode() if pipe is not None else "auto"
    out = []
    for text in wanted:
        match = lib.lookup(text)
        if match is None:
            out.append(
                {
                    "instruction": text,
                    "cached": False,
                    "plan_source": "llm" if mode != "cache_only" else "ERROR",
                    "detail": (
                        "no cached BT; this task would call the planner"
                        if mode != "cache_only"
                        else "no cached BT and plan_mode=cache_only: this "
                        "task would FAIL rather than plan"
                    ),
                }
            )
            continue
        entry = match.entry
        out.append(
            {
                "instruction": text,
                "cached": True,
                "plan_source": ("seed" if entry.seed and match.source != "pin" else match.source),
                "similarity": round(match.similarity, 3),
                "hash": entry.hash,
                "matched_instruction": entry.instruction,
                "success": entry.success,
                "fail": entry.fail,
                "promoted": entry.promoted,
                "verified": entry.verified,
                "n_actions": _count_actions(entry.score),
                "labels": sorted(
                    {
                        str(v)
                        for node in _flatten(entry.score)
                        for k, v in (node.get("params") or {}).items()
                        if k.endswith("_label") and v
                    }
                ),
            }
        )
    return {"plan_mode": mode, "results": out, "n_cached": sum(r["cached"] for r in out)}


def _flatten(score: Any) -> List[Dict[str, Any]]:
    out: List[Dict[str, Any]] = []

    def walk(node):
        if not isinstance(node, dict):
            return
        kids = node.get("children")
        if isinstance(kids, list) and kids:
            for c in kids:
                walk(c)
        else:
            out.append(node)

    walk((score or {}).get("tree") if isinstance(score, dict) else None)
    return out


# runtime mode


class _ToggleBody(BaseModel):
    enabled: bool


@router.post("/api/bt/toggle_cache")
async def bt_toggle_cache(body: _ToggleBody):
    """
    Turn strict offline mode on/off. PERSISTED to the library directory, so
    a server restart mid-collection cannot silently revert to the env
    default.
    """
    lib = _lib()
    state.use_cached_bt = bool(body.enabled)
    if lib is not None:
        lib.set_use_cached_bt(bool(body.enabled))
    logger.info("bt_toggle_cache: cache_only = %s (persisted)", body.enabled)
    return await bt_cache_status()


@router.post("/api/bt/clear_toggle")
async def bt_clear_toggle():
    """
    Drop the persisted override and fall back to config / env.
    """
    lib = _lib()
    state.use_cached_bt = None
    if lib is not None:
        lib.set_use_cached_bt(None)
    return await bt_cache_status()


@router.get("/api/bt/cache_status")
async def bt_cache_status():
    """
    Which plan source is active and where the decision came from.
    """
    lib = _lib()
    pipe = state.pipeline
    mode = pipe.plan_mode() if pipe is not None else "auto"
    if lib is not None and lib.use_cached_bt is not None:
        source = "persisted"
    elif state.use_cached_bt is not None:
        source = "ui"
    elif pipe is not None and getattr(pipe.config, "bt_plan_mode", None):
        source = "config"
    else:
        source = "env"
    return {
        "plan_mode": mode,
        # Legacy field the existing frontend toggle reads.
        "enabled": mode == "cache_only",
        "source": source,
        "library_dir": str(lib.root) if lib is not None else None,
        "stats": lib.stats() if lib is not None else {},
        "min_similarity": lib.min_similarity if lib is not None else None,
        "auto_promote_after": lib.auto_promote_after if lib is not None else None,
    }


# pinning


class _SelectBody(BaseModel):
    hash: str


@router.post("/api/bt/select")
async def bt_select(body: _SelectBody):
    """
    Single-shot override: use this exact BT for the NEXT execute, then
    revert. For repeated runs of one task use /api/bt/pin instead.
    """
    lib = _lib()
    if lib is None:
        return _no_lib()
    entry = lib.get(body.hash)
    if entry is None:
        return JSONResponse(status_code=404, content={"error": f"no such BT: {body.hash}"})
    state.pinned_bt_hash = body.hash
    logger.info("bt_select: %s pinned for the next execute only", body.hash)
    return {"pinned": body.hash, "instruction": entry.instruction}


class _PinBody(BaseModel):
    instruction: str
    hash: str


@router.post("/api/bt/pin")
async def bt_pin(body: _PinBody):
    """
    Bind one task string to one BT, permanently and persistently.

    This is the operator's "I know this tree is the good one" control. It
    outranks every other retrieval tier and is NOT consumed, so a 50-episode
    collection session serves the same tree 50 times without re-pinning.
    """
    lib = _lib()
    if lib is None:
        return _no_lib()
    entry = lib.pin(body.instruction, body.hash)
    if entry is None:
        return JSONResponse(
            status_code=404,
            content={"error": f"no such BT: {body.hash} (or empty instruction)"},
        )
    logger.info("bt_pin: %r -> %s (persisted)", body.instruction, body.hash)
    return {"instruction": body.instruction, "hash": entry.hash, "pins": lib.pins()}


class _UnpinBody(BaseModel):
    instruction: Optional[str] = None


@router.post("/api/bt/unpin")
async def bt_unpin(body: _UnpinBody = _UnpinBody()):
    """
    With an instruction: drop that task's persistent pin.
    Without: clear the single-shot pin.
    """
    lib = _lib()
    if body.instruction:
        if lib is None:
            return _no_lib()
        prev = lib.unpin(body.instruction)
        return {"cleared": prev, "instruction": body.instruction, "pins": lib.pins()}
    prev = state.pinned_bt_hash
    state.pinned_bt_hash = None
    return {"cleared": prev}


# curation


class _PromoteBody(BaseModel):
    hash: str
    promoted: bool = True


@router.post("/api/bt/promote")
async def bt_promote(body: _PromoteBody):
    """
    Mark an entry canonical. Promoted entries win ties in the resolver and
    are always offered to the planner as few-shot examples. Entries also
    auto-promote at ``bt.auto_promote_after`` clean successes.
    """
    lib = _lib()
    if lib is None:
        return _no_lib()
    entry = lib.set_promoted(body.hash, body.promoted)
    if entry is None:
        return JSONResponse(status_code=404, content={"error": f"no such BT: {body.hash}"})
    return {"hash": entry.hash, "promoted": entry.promoted}


class _AliasBody(BaseModel):
    hash: str
    instruction: str


@router.post("/api/bt/alias")
async def bt_alias(body: _AliasBody):
    """
    Teach an existing entry another phrasing, so that phrasing becomes an
    exact hit instead of relying on fuzzy retrieval.
    """
    lib = _lib()
    if lib is None:
        return _no_lib()
    entry = lib.alias(body.hash, body.instruction)
    if entry is None:
        return JSONResponse(status_code=404, content={"error": f"no such BT: {body.hash}"})
    return {"hash": entry.hash, "aliases": entry.aliases}


@router.delete("/api/bt/by_instruction")
async def bt_delete_by_instruction(instruction: str):
    """
    Delete every entry keyed on a task string. Pruning tool for a task whose
    cached tree turned out to be wrong.
    """
    lib = _lib()
    if lib is None:
        return _no_lib()
    removed = lib.delete_by_instruction(instruction)
    logger.info("bt_delete_by_instruction: %r -> removed %s", instruction, removed)
    return {"instruction": instruction, "deleted": removed, "count": len(removed)}


@router.delete("/api/bt/{hash}")
async def bt_delete(hash: str):
    """
    Delete one entry and any pins that referenced it.
    """
    lib = _lib()
    if lib is None:
        return _no_lib()
    entry = lib.delete(hash)
    if entry is None:
        return JSONResponse(status_code=404, content={"error": f"no such BT: {hash}"})
    if state.pinned_bt_hash == hash:
        state.pinned_bt_hash = None
    logger.info("bt_delete: removed %s (%r)", hash, entry.instruction)
    return {"deleted": hash}


# freeze / restore


@router.get("/api/bt/export", response_class=PlainTextResponse)
async def bt_export(hash: str):
    """
    Render an entry as seed YAML, ready to drop into ``configs/bt_seeds/``.

    Closes the capture loop: a tree that proved itself on the rig becomes a
    version-controlled file, so a wiped output directory or a fresh clone
    still serves it.
    """
    lib = _lib()
    if lib is None:
        return PlainTextResponse("pipeline not initialised", status_code=503)
    entry = lib.get(hash)
    if entry is None:
        return PlainTextResponse(f"no such BT: {hash}", status_code=404)
    return dump_seed(entry)


class _ImportBody(BaseModel):
    yaml: str
    force: bool = False


@router.post("/api/bt/import")
async def bt_import(body: _ImportBody):
    """
    Install a seed document posted as YAML text.
    """
    lib = _lib()
    if lib is None:
        return _no_lib()
    try:
        seed = parse_seed(yaml.safe_load(body.yaml))
    except (SeedError, yaml.YAMLError, AttributeError) as exc:
        return JSONResponse(status_code=400, content={"error": str(exc)})
    report = seed_library(lib, seeds=[seed], force=body.force)
    return report.as_dict()


class _ReloadBody(BaseModel):
    force: bool = False


@router.post("/api/bt/reload_seeds")
async def bt_reload_seeds(body: _ReloadBody = _ReloadBody()):
    """
    Re-read ``configs/bt_seeds/`` without restarting the server.
    """
    lib = _lib()
    if lib is None:
        return _no_lib()
    pipe = state.pipeline
    report = seed_library(
        lib,
        seed_dir=getattr(pipe.config, "bt_seed_dir", None) if pipe else None,
        force=body.force,
    )
    return {**report.as_dict(), "stats": lib.stats()}

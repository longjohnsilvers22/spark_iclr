"""
Task-execution scores across the different BTs the planner generated.

Aggregates saved task runs (``output/real_runs/<ts>/result.json``) and the
disk-backed BT library, grouped by instruction. For each instruction it
surfaces every distinct behaviour-tree the Gemini planner produced (keyed by
a content hash of the plan) alongside how that tree performed when executed:
success rate over runs, per-action pass/fail from the latest run, duration,
and the final verify outcome where one exists.

This backs the ``/scores`` page, which shows that the system can execute the
same task with different generated BTs and reports each tree's score.

Reads are best-effort and tolerate a missing pipeline: the output paths are
resolved from ``pipeline.config`` when available, else from the historical
``output/`` layout relative to the server cwd, so the page renders headless.
"""

from __future__ import annotations

import hashlib
import json
import logging
from pathlib import Path
from typing import Any, Dict, List

from fastapi import APIRouter
from fastapi.responses import HTMLResponse, JSONResponse

from spark_real.bt_library import _canonical_score_str, _hash_entry, _tokens
from spark_real.routes import state

logger = logging.getLogger(__name__)
router = APIRouter()

_PAGE = Path(__file__).resolve().parent.parent / "frontend" / "bt_scores.html"

# Instructions naming any of these collapse into one "silverware" category so
# the dozens of phrasings (pick up the knife / knives / silverware / sort the
# fork...) score together. Each BT still carries its own prompt for display.
_SILVERWARE_WORDS = frozenset(
    {
        "silverware",
        "cutlery",
        "utensil",
        "utensils",
        "knife",
        "knives",
        "fork",
        "forks",
        "spoon",
        "spoons",
    }
)


def _category(instruction: str) -> str:
    if _tokens(instruction) & _SILVERWARE_WORDS:
        return "silverware"
    return instruction


def _user_aborted(run: Dict[str, Any]) -> bool:
    """
    True for runs the operator killed with the abort button, which are not
    task outcomes and are excluded from scoring. Robot reflex aborts are real
    failures and do not match this message.
    """
    return any(
        "Aborted by user" in (r.get("message") or "") for r in run.get("execution_results") or []
    )


def _output_dir() -> Path:
    """
    Resolve the real_runs output directory (pipeline config or default).
    """
    pipe = state.pipeline
    if pipe is not None:
        cfg = getattr(pipe, "config", None)
        out = getattr(cfg, "output_dir", None)
        if out:
            return Path(out).expanduser()
    return Path("output/real_runs")


def _plan_hash(plan: Any, instruction: str) -> str:
    """
    Content hash for a plan, matching the BT library's scheme.

    Falls back to a plan-only hash when the instruction is empty so two runs
    of the same tree still merge.
    """
    try:
        return _hash_entry(instruction or "", plan)
    except Exception:
        h = hashlib.blake2b(digest_size=6)
        h.update(_canonical_score_str(plan).encode("utf-8"))
        return h.hexdigest()


def _count_actions(plan: Any) -> int:
    """
    Leaf action nodes in a BT plan (anything without children).
    """
    if not isinstance(plan, dict):
        return 0
    tree = plan.get("tree") if "tree" in plan else plan
    count = 0

    def walk(node):
        nonlocal count
        if not isinstance(node, dict):
            return
        kids = node.get("children")
        if isinstance(kids, list) and kids:
            for c in kids:
                walk(c)
        elif node.get("type") not in (None, "sequence", "selector", "fallback"):
            count += 1

    walk(tree)
    return count


def _action_types(plan: Any) -> List[str]:
    """
    Ordered list of leaf action types for a compact tree summary.
    """
    out: List[str] = []
    if not isinstance(plan, dict):
        return out
    tree = plan.get("tree") if "tree" in plan else plan

    def walk(node):
        if not isinstance(node, dict):
            return
        kids = node.get("children")
        if isinstance(kids, list) and kids:
            for c in kids:
                walk(c)
        else:
            t = node.get("type")
            if t and t not in ("sequence", "selector", "fallback"):
                out.append(t)

    walk(tree)
    return out


def _load_runs(out_dir: Path) -> List[Dict[str, Any]]:
    """
    Load every result.json under the output dir, newest-first.
    """
    runs: List[Dict[str, Any]] = []
    if not out_dir.exists():
        return runs
    for sub in out_dir.iterdir():
        if not sub.is_dir():
            continue
        rj = sub / "result.json"
        if not rj.exists():
            continue
        try:
            d = json.loads(rj.read_text())
        except (OSError, json.JSONDecodeError):
            continue
        if _user_aborted(d):
            continue
        d["_dir"] = sub.name
        runs.append(d)
    runs.sort(key=lambda r: r.get("timestamp") or r.get("_dir") or "", reverse=True)
    return runs


def _verify_status(run: Dict[str, Any], exec_results: List[Dict[str, Any]]) -> str:
    """
    The run's tri-state verdict: ``pass`` | ``fail`` | ``unverified``.

    Prefers the explicit field written by the verifier. Runs saved before that
    field existed fall back to the verify ExecutionResult row, whose boolean
    cannot distinguish "checked and failed" from "could not be checked" -- so
    they report ``unverified`` rather than inventing a ``fail``.
    """
    status = run.get("verify_status")
    if status in ("pass", "fail", "unverified"):
        return status
    verifies = [r for r in (exec_results or []) if r.get("action") == "verify"]
    if verifies and bool(verifies[-1].get("success")):
        return "pass"
    return "unverified"


def _build() -> Dict[str, Any]:
    """
    Aggregate runs into category -> distinct BTs -> execution scores.

    Categories collapse related phrasings (every silverware/knife/fork/spoon
    instruction is one "silverware" category); other instructions are their
    own category. Within a category BTs are keyed by a plan-only hash so the
    same tree run under two phrasings merges, and each BT keeps the prompt(s)
    it actually ran under.
    """
    out_dir = _output_dir()
    runs = _load_runs(out_dir)

    # category -> plan_hash -> bt record
    groups: Dict[str, Dict[str, Dict[str, Any]]] = {}
    total_runs = 0

    for run in runs:
        instr = (run.get("instruction") or "").strip()
        if not instr:
            continue
        plan = run.get("plan") or {}
        if not plan:
            continue
        total_runs += 1
        cat = _category(instr)
        ph = _plan_hash(plan, "")
        bts = groups.setdefault(cat, {})
        bt = bts.get(ph)
        exec_results = run.get("execution_results") or []
        verify_status = _verify_status(run, exec_results)
        success = verify_status == "pass"
        run_row = {
            "timestamp": run.get("timestamp") or run.get("_dir"),
            "success": success,
            "duration": float(run.get("duration") or 0.0),
            "verify": verify_status,
            "verify_status": verify_status,
            "verify_detail": run.get("verify"),
            "instruction": instr,
            # Where this run's tree came from. ``_maybe_save_bt`` feeds the
            # same verify outcome back into the library, so a run showing
            # plan_source="llm" is one that cost an API call and a run
            # showing a cache source is one that did not. Runs saved before
            # this field existed report None.
            "plan_source": run.get("plan_source"),
            "bt_hash": run.get("bt_hash"),
            "label_resolutions": run.get("label_resolutions") or {},
            "actions": [
                {
                    "action": r.get("action"),
                    "success": bool(r.get("success")),
                    "message": r.get("message") or "",
                    "duration": float(r.get("duration") or 0.0),
                }
                for r in exec_results
            ],
        }
        if bt is None:
            bts[ph] = {
                "hash": ph,
                "n_actions": _count_actions(plan),
                "action_types": _action_types(plan),
                "plan": plan,
                "runs": [run_row],
                "prompts": [instr],
                # Library keys hash (instruction, plan); track every pairing
                # this tree ran under so the library badge still resolves.
                "lib_keys": {_plan_hash(plan, instr)},
            }
        else:
            bt["runs"].append(run_row)
            if instr not in bt["prompts"]:
                bt["prompts"].append(instr)
            bt["lib_keys"].add(_plan_hash(plan, instr))

    # In-library hashes (promoted / few-shot) so the page can flag them.
    lib_hashes = set()
    lib = getattr(state.pipeline, "_bt_library", None) if state.pipeline else None
    if lib is not None:
        try:
            lib_hashes = set(lib._entries.keys())
        except Exception:
            lib_hashes = set()

    categories: List[Dict[str, Any]] = []
    for cat, bts in groups.items():
        bt_list = []
        for bt in bts.values():
            run_rows = bt["runs"]
            n = len(run_rows)
            ok = sum(1 for r in run_rows if r["success"])
            n_unverified = sum(1 for r in run_rows if r["verify_status"] == "unverified")
            n_failed = sum(1 for r in run_rows if r["verify_status"] == "fail")
            # Latest run drives the per-action breakdown shown by default.
            latest = run_rows[0]
            bt_list.append(
                {
                    "hash": bt["hash"],
                    "n_actions": bt["n_actions"],
                    "action_types": bt["action_types"],
                    "instruction": bt["prompts"][0],
                    "prompts": bt["prompts"],
                    "n_runs": n,
                    "n_success": ok,
                    "success_rate": (ok / n) if n else 0.0,
                    # A tree whose runs are mostly unverified is not a bad
                    # tree -- it is one the verifier could not judge. Shown
                    # separately so a cache going cold is visible, not
                    # mistaken for a tree that stopped working.
                    "n_fail": n_failed,
                    "n_unverified": n_unverified,
                    "unverified_rate": (n_unverified / n) if n else 0.0,
                    "in_library": bool(bt["lib_keys"] & lib_hashes),
                    # How many of this tree's runs were served from cache.
                    # The number the operator watches during collection: it
                    # should be every run after the first.
                    "n_cache_served": sum(
                        1 for r in run_rows if r.get("plan_source") and r["plan_source"] != "llm"
                    ),
                    "latest": latest,
                    "runs": run_rows,
                }
            )
        # Best-performing tree first, then more-tried trees.
        bt_list.sort(key=lambda b: (b["success_rate"], b["n_runs"]), reverse=True)
        total = sum(b["n_runs"] for b in bt_list)
        total_ok = sum(b["n_success"] for b in bt_list)
        total_unverified = sum(b["n_unverified"] for b in bt_list)
        prompts = sorted({p for b in bt_list for p in b["prompts"]})
        categories.append(
            {
                "category": cat,
                "n_prompts": len(prompts),
                "n_bts": len(bt_list),
                "n_runs": total,
                "n_success": total_ok,
                "success_rate": (total_ok / total) if total else 0.0,
                "n_fail": sum(b["n_fail"] for b in bt_list),
                "n_unverified": total_unverified,
                "unverified_rate": (total_unverified / total) if total else 0.0,
                "bts": bt_list,
            }
        )

    # Categories with the most distinct BTs first (most interesting), then
    # by run count, so the side-by-side comparisons lead the page.
    categories.sort(key=lambda g: (g["n_bts"], g["n_runs"]), reverse=True)

    return {
        "categories": categories,
        "totals": {
            "categories": len(categories),
            "runs": total_runs,
            "distinct_bts": sum(g["n_bts"] for g in categories),
        },
        "output_dir": str(out_dir),
    }


@router.get("/api/bt/scores")
async def bt_scores():
    """
    Aggregated task-execution scores grouped by instruction and BT.
    """
    try:
        return _build()
    except Exception as exc:
        logger.exception("bt_scores aggregation failed")
        return JSONResponse(status_code=500, content={"error": str(exc)})


@router.get("/scores")
async def scores_page():
    """
    Serve the BT-execution-scores page.
    """
    if not _PAGE.exists():
        return HTMLResponse(
            "<h1>BT scores page not built</h1>" "<p>Expected frontend/bt_scores.html</p>",
            status_code=404,
        )
    return HTMLResponse(
        _PAGE.read_text(),
        headers={
            "Cache-Control": "no-cache, no-store, must-revalidate",
            "Pragma": "no-cache",
        },
    )

"""
Macro mining for SPARK behaviour trees.

Pattern-mines successful BT executions to surface recurring subtrees that
should become typed macros (new primitives composed from old ones).

Input log schema (one JSON per execution)::

    {"task_string": str, "bt"|"bt_yaml": ..., "success": bool,
     "slot_bindings": dict, "trial_id": str}

Entries whose tree lives under ``score.tree`` are also accepted.

CLI: ``python -m spark_dsl.macro_mining --logs-dir DIR --out macros.yaml --min-support 3``
"""
from __future__ import annotations

import argparse
import collections
import hashlib
import json
import os
import pathlib
import re
from typing import Any, Dict, Iterable, List, Optional, Tuple

import yaml

try:
    from google import genai
except ImportError:  # optional dependency for --llm-refine
    genai = None

ParamMap = Dict[str, Any]


# Tree extraction

def _coerce_tree(blob: Dict[str, Any]) -> Optional[Dict[str, Any]]:
    """
    Pull a BT dict out of a log entry, accepting several shapes.
    """
    if not isinstance(blob, dict):
        return None
    if "type" in blob and ("children" in blob or "params" in blob):
        return blob
    if isinstance(blob.get("bt"), dict):
        return _coerce_tree(blob["bt"])
    if isinstance(blob.get("bt_yaml"), str):
        try:
            parsed = yaml.safe_load(blob["bt_yaml"])
        except yaml.YAMLError:
            parsed = None
        if isinstance(parsed, dict):
            return _coerce_tree(parsed)
    if isinstance(blob.get("score"), dict) and isinstance(blob["score"].get("tree"), dict):
        return blob["score"]["tree"]
    if isinstance(blob.get("tree"), dict):
        return blob["tree"]
    return None


def _flatten_sequence(tree: Dict[str, Any]) -> List[Dict[str, Any]]:
    """
    Linear list of children for a sequence-rooted tree (else single node).
    """
    if tree.get("type") == "sequence":
        return [k for k in (tree.get("children") or []) if isinstance(k, dict)]
    return [tree]


# Canonical hashing and n-grams

def _node_signature(node: Dict[str, Any]) -> str:
    keys = ",".join(sorted((node.get("params") or {}).keys()))
    return f"{node.get('type', '?')}({keys})"


def _ngrams(seq: List[Dict[str, Any]], n_min: int, n_max: int
            ) -> Iterable[List[Dict[str, Any]]]:
    for n in range(n_min, n_max + 1):
        if n > len(seq):
            break
        for i in range(0, len(seq) - n + 1):
            yield seq[i : i + n]


def _canonical_hash(window: List[Dict[str, Any]]) -> str:
    sig = "|".join(_node_signature(n) for n in window)
    return hashlib.sha1(sig.encode("utf-8")).hexdigest()[:12]


# Slot inference

def _freeze(v: Any) -> Any:
    if isinstance(v, list):
        return tuple(_freeze(x) for x in v)
    if isinstance(v, dict):
        return tuple(sorted((k, _freeze(val)) for k, val in v.items()))
    return v


def _infer_slots(occurrences: List[List[Dict[str, Any]]]
                 ) -> Tuple[ParamMap, ParamMap]:
    """
    Split params into bound (constant) vs free (varying) across occurrences.
    """
    bound: ParamMap = {}
    free: ParamMap = {}
    if not occurrences:
        return bound, free
    for idx in range(len(occurrences[0])):
        per_key: Dict[str, List[Any]] = collections.defaultdict(list)
        for occ in occurrences:
            for k, v in (occ[idx].get("params") or {}).items():
                per_key[k].append(v)
        for k, vals in per_key.items():
            slot_key = f"node{idx}.{k}"
            if len(vals) == len(occurrences) and all(v == vals[0] for v in vals):
                bound[slot_key] = vals[0]
            else:
                free[slot_key] = {"example": vals[0],
                                  "n_distinct": len(set(map(_freeze, vals)))}
    return bound, free


# Naming

_NAME_HINTS = {
    ("move_to_keypoint", "grasp"): "approach_grasp",
    ("move_to_keypoint", "grasp", "move_relative"): "pick_top_down",
    ("move_to_keypoint", "release"): "place_at",
    ("move_relative", "move_to_keypoint", "release"): "lift_and_place",
    ("grasp", "move_relative"): "grasp_and_lift",
    ("move_to_keypoint", "grasp", "move_relative",
     "move_to_keypoint", "release"): "pick_and_place",
}


def _suggest_name(window: List[Dict[str, Any]]) -> str:
    types = tuple(n.get("type", "?") for n in window)
    if types in _NAME_HINTS:
        return _NAME_HINTS[types]
    short = [re.sub(r"[^a-z0-9]+", "_", t.lower()) for t in types]
    return "_then_".join(short)[:60]


# Public API

def _build_candidate(name: str, window: List[Dict[str, Any]],
                     occurrences: List[List[Dict[str, Any]]],
                     task_strings: List[str]) -> Dict[str, Any]:
    bound, free = _infer_slots(occurrences)
    body: List[Dict[str, Any]] = []
    for idx, node in enumerate(window):
        params = (node.get("params") or {}).copy()
        for slot_key in free:
            ni, key = slot_key.split(".", 1)
            if ni == f"node{idx}" and key in params:
                params[key] = f"<{slot_key}>"
        body.append({"type": node.get("type"), "params": params})
    return {
        "name": name,
        "support": len(occurrences),
        "distinct_tasks": len(set(task_strings)),
        "bound_params": bound,
        "free_params": free,
        "body": body,
        "example_tasks": sorted(set(task_strings))[:5],
    }


def load_logs(logs_dir: pathlib.Path) -> List[Dict[str, Any]]:
    out: List[Dict[str, Any]] = []
    for path in sorted(pathlib.Path(logs_dir).glob("*.json")):
        try:
            blob = json.loads(path.read_text())
        except (OSError, json.JSONDecodeError):
            continue
        if isinstance(blob, list):
            out.extend(b for b in blob if isinstance(b, dict))
        elif isinstance(blob, dict):
            out.append(blob)
    return out


def mine_macros(logs: List[Dict[str, Any]], *, min_support: int = 3,
                n_min: int = 2, n_max: int = 5,
                assume_success: bool = False) -> List[Dict[str, Any]]:
    """
    Return ranked macro candidates from successful logs.

    Candidates must occur across ``min_support`` distinct task_strings.
    Ranking prefers longer windows then higher distinct-task support.
    If ``assume_success`` is True, every loaded log is treated as a success
    (used when the source directory only stores successful executions).
    """
    buckets: Dict[str, List[Tuple[List[Dict[str, Any]], str]]] = collections.defaultdict(list)
    for log in logs:
        if not assume_success and not log.get("success", False):
            continue
        tree = _coerce_tree(log)
        if tree is None:
            continue
        seq = _flatten_sequence(tree)
        if not seq:
            continue
        task = log.get("task_string") or log.get("instruction") or ""
        for window in _ngrams(seq, n_min, n_max):
            buckets[_canonical_hash(window)].append((window, task))

    candidates: List[Dict[str, Any]] = []
    for entries in buckets.values():
        tasks = [t for _, t in entries]
        if len(set(tasks)) < min_support:
            continue
        windows = [w for w, _ in entries]
        candidates.append(_build_candidate(_suggest_name(windows[0]),
                                           windows[0], windows, tasks))
    candidates.sort(key=lambda c: (-len(c["body"]), -c["distinct_tasks"], -c["support"]))
    return candidates


# Optional Gemini refine

def llm_refine(candidate: Dict[str, Any]) -> Dict[str, Any]:  # pragma: no cover
    """
    Best-effort LLM rename. Silent no-op if google-genai/key absent.
    """
    if genai is None:
        return candidate
    try:
        key = os.environ.get("GEMINI_API_KEY")
        if not key:
            return candidate
        client = genai.Client(api_key=key)
        prompt = ("Suggest a concise snake_case macro name and one-line description "
                  "for this BT subtree (return JSON {name, description}):\n"
                  f"{json.dumps(candidate['body'], indent=2)}")
        resp = client.models.generate_content(
            model="gemini-2.5-flash", contents=prompt,
            config={"response_mime_type": "application/json", "temperature": 0})
        data = json.loads(resp.text)
        if isinstance(data, dict) and data.get("name"):
            candidate = dict(candidate)
            candidate["name"] = re.sub(r"[^a-z0-9_]+", "_", data["name"].lower())[:40]
            if data.get("description"):
                candidate["description"] = data["description"]
    except Exception:  # noqa: BLE001 -- best-effort
        pass
    return candidate


# CLI

def _cli(argv: Optional[List[str]] = None) -> int:
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument("--logs-dir", required=True, type=pathlib.Path)
    p.add_argument("--out", required=True, type=pathlib.Path)
    p.add_argument("--min-support", type=int, default=3)
    p.add_argument("--n-min", type=int, default=2)
    p.add_argument("--n-max", type=int, default=5)
    p.add_argument("--llm-refine", action="store_true",
                   help="(optional) rename via Gemini; needs GEMINI_API_KEY.")
    p.add_argument("--assume-success", action="store_true",
                   help="treat every loaded entry as success (success-only dir).")
    args = p.parse_args(argv)
    logs = load_logs(args.logs_dir)
    cands = mine_macros(logs, min_support=args.min_support,
                        n_min=args.n_min, n_max=args.n_max,
                        assume_success=args.assume_success)
    if args.llm_refine:
        cands = [llm_refine(c) for c in cands]
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(yaml.safe_dump({"macros": cands}, sort_keys=False))
    print(f"[macro_mining] wrote {len(cands)} candidate(s) to {args.out}")
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(_cli())

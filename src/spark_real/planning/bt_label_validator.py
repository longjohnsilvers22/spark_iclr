"""Hard post-parse label validator for SPARK BT scores.

The planner routinely ignores the prompt's list of allowed detection
labels, emitting strings like ``"second drawer"`` when SAM3 only detected
``"drawer handle"``.  After the BT is parsed but before it is executed,
every label-bearing param is either snapped to the closest canonical label
or triggers a single sharper-prompt retry.

Public entry point: :func:`validate_and_repair_bt`.  Lower-level helpers
(``validate_bt_labels``, ``fuzzy_snap_bt_labels``) are exported for
testing and for the DSL path that wants snap-only behaviour.
"""
from __future__ import annotations

import difflib
from typing import Any, Callable, Iterable, Optional

# Param keys treated as label-bearing for the validator walk.  Mirrors the
# executor's ``_action_uses_keypoint`` plus the legacy plural ``labels``
# field that some macros emit.
_LABEL_KEYS_STR = ('keypoint_label', 'target_label', 'label',
                   'pick_label', 'place_label')
_LABEL_KEYS_LIST = ('labels',)

_FUZZY_CUTOFF = 0.4


def _iter_action_params(node: dict) -> Iterable[dict]:
    """Yield every action node's ``params`` dict (mutable) under ``node``.

    Accepts both ``{"tree": {...}}`` and bare-tree dicts.  Composite
    nodes (``sequence``, ``selector``) are recursed; leaves yield their
    own ``params`` dict (creating an empty one if absent so callers can
    rewrite slots in place).
    """
    if not isinstance(node, dict):
        return
    if 'tree' in node and isinstance(node['tree'], dict):
        yield from _iter_action_params(node['tree'])
        return
    ntype = node.get('type')
    if ntype in ('sequence', 'selector', 'retry', 'fallback'):
        for child in node.get('children') or []:
            yield from _iter_action_params(child)
        return
    # Leaf action: surface its params.
    params = node.get('params')
    if params is None:
        params = {}
        node['params'] = params
    if isinstance(params, dict):
        yield params


def validate_bt_labels(tree: dict,
                        allowed_labels: set[str]) -> tuple[bool, list[str]]:
    """Walk ``tree`` and collect every label that is not in ``allowed_labels``.

    Returns ``(all_valid, invalid_strings)``.  Strings are reported in
    visitation order; duplicates are preserved so the retry prompt can
    show frequency.  Empty strings, ``None``, and the literal ``"none"``
    are tolerated (handled by the primitives as no-op fallbacks).
    """
    invalid: list[str] = []
    for params in _iter_action_params(tree):
        for key in _LABEL_KEYS_STR:
            val = params.get(key)
            if not isinstance(val, str) or not val:
                continue
            if val.lower() == 'none':
                continue
            if val not in allowed_labels:
                invalid.append(val)
        for key in _LABEL_KEYS_LIST:
            vals = params.get(key)
            if not isinstance(vals, list):
                continue
            for val in vals:
                if not isinstance(val, str) or not val:
                    continue
                if val.lower() == 'none':
                    continue
                if val not in allowed_labels:
                    invalid.append(val)
    return (len(invalid) == 0, invalid)


def _closest_label(s: str, allowed: list[str]) -> Optional[tuple[str, float]]:
    """Return ``(match, similarity)`` for the closest allowed label, else None.

    Uses ``difflib.get_close_matches`` with cutoff 0.4.  Substring hits
    are boosted: if any allowed label contains ``s`` (or vice versa) we
    return that pair first; this catches "drawer" -> "drawer handle"
    cases where Levenshtein alone scores too low because the strings
    have very different lengths.
    """
    s_low = s.lower()
    for a in allowed:
        a_low = a.lower()
        if s_low in a_low or a_low in s_low:
            ratio = difflib.SequenceMatcher(None, s_low, a_low).ratio()
            return (a, max(ratio, 0.5))
    matches = difflib.get_close_matches(s, allowed, n=1, cutoff=_FUZZY_CUTOFF)
    if matches:
        ratio = difflib.SequenceMatcher(None, s.lower(),
                                          matches[0].lower()).ratio()
        return (matches[0], ratio)
    # Try case-insensitive match.
    lower_map = {a.lower(): a for a in allowed}
    matches = difflib.get_close_matches(s_low, list(lower_map.keys()),
                                          n=1, cutoff=_FUZZY_CUTOFF)
    if matches:
        canon = lower_map[matches[0]]
        ratio = difflib.SequenceMatcher(None, s_low, matches[0]).ratio()
        return (canon, ratio)
    return None


def fuzzy_snap_bt_labels(tree: dict, allowed_labels: set[str],
                          log: Optional[Callable[[str], None]] = None
                          ) -> tuple[list[dict], list[str]]:
    """Snap invalid labels in ``tree`` to the closest allowed label in place.

    Returns ``(snap_actions, unresolved)`` where:
      * ``snap_actions`` is a list of dicts ``{from, to, similarity, key}``
        suitable for ``trial_meta['validator_actions']``.
      * ``unresolved`` is the list of invalid strings that had no match
        above the fuzzy cutoff (callers should trigger the retry path).
    """
    allowed_list = list(allowed_labels)
    snap_actions: list[dict] = []
    unresolved: list[str] = []
    log = log or (lambda _msg: None)

    for params in _iter_action_params(tree):
        for key in _LABEL_KEYS_STR:
            val = params.get(key)
            if not isinstance(val, str) or not val:
                continue
            if val.lower() == 'none' or val in allowed_labels:
                continue
            match = _closest_label(val, allowed_list)
            if match is None:
                unresolved.append(val)
                continue
            canon, sim = match
            params[key] = canon
            snap_actions.append({'from': val, 'to': canon,
                                  'similarity': round(sim, 3),
                                  'key': key})
            log(f"[validator] snapped '{val}' -> '{canon}' "
                f"(similarity {sim:.2f})")
        for key in _LABEL_KEYS_LIST:
            vals = params.get(key)
            if not isinstance(vals, list):
                continue
            for i, val in enumerate(vals):
                if not isinstance(val, str) or not val:
                    continue
                if val.lower() == 'none' or val in allowed_labels:
                    continue
                match = _closest_label(val, allowed_list)
                if match is None:
                    unresolved.append(val)
                    continue
                canon, sim = match
                vals[i] = canon
                snap_actions.append({'from': val, 'to': canon,
                                      'similarity': round(sim, 3),
                                      'key': f'{key}[{i}]'})
                log(f"[validator] snapped '{val}' -> '{canon}' "
                    f"(similarity {sim:.2f})")
    return snap_actions, unresolved


def build_strict_retry_prompt_suffix(allowed_labels: Iterable[str],
                                      invalid: list[str]) -> str:
    """STRICT-mode suffix appended to the planner call on retry."""
    bullet = '\n'.join(f'  - {lbl}' for lbl in allowed_labels)
    bad = ', '.join(f"'{x}'" for x in invalid) or '(unknown)'
    return (
        "\n\nSTRICT MODE - PREVIOUS PLAN REJECTED\n"
        "The previous plan referenced labels NOT in the detection set: "
        f"{bad}.\n"
        "Available detection labels (you may ONLY use these for "
        "target_label / keypoint_label / labels):\n"
        f"{bullet}\n"
        "Any value not in the list above will be REJECTED.  Pick the "
        "closest available label.  Do NOT invent strings.\n"
    )


def validate_and_repair_bt(tree: dict, allowed_labels: set[str],
                            *,
                            retry_fn: Optional[Callable[[str], Optional[dict]]] = None,
                            log: Optional[Callable[[str], None]] = None
                            ) -> tuple[dict, list[dict]]:
    """Validate, fuzzy-snap, and (if needed) retry once.

    ``retry_fn`` is called at most once with a strict-mode prompt suffix
    and must return a fresh BT score dict (with ``tree``) or ``None``.
    The returned ``actions`` log is suitable for serialising into
    ``trial_meta['validator_actions']``.
    """
    log = log or print
    actions: list[dict] = []
    ok, invalid = validate_bt_labels(tree, allowed_labels)
    if ok:
        return tree, actions

    actions.append({'event': 'invalid_labels_detected',
                    'invalid': list(invalid)})
    snap_actions, unresolved = fuzzy_snap_bt_labels(
        tree, allowed_labels, log=log)
    actions.extend({'event': 'snap', **a} for a in snap_actions)

    if not unresolved:
        return tree, actions

    if retry_fn is None:
        actions.append({'event': 'fail_closed', 'unresolved': unresolved})
        log(f"[validator] no retry_fn — leaving {len(unresolved)} "
            f"unresolved label(s): {unresolved}")
        return tree, actions

    log(f"[validator] retry triggered due to invalid labels: {unresolved}")
    actions.append({'event': 'retry_triggered', 'unresolved': unresolved})
    suffix = build_strict_retry_prompt_suffix(sorted(allowed_labels),
                                                unresolved)
    retry_tree = retry_fn(suffix)
    if retry_tree is None:
        actions.append({'event': 'retry_failed'})
        log("[validator] retry returned no BT; keeping original")
        return tree, actions

    ok2, invalid2 = validate_bt_labels(retry_tree, allowed_labels)
    if ok2:
        actions.append({'event': 'retry_clean'})
        return retry_tree, actions

    # Retry still has invalids: snap whatever can be snapped.
    snap2, unresolved2 = fuzzy_snap_bt_labels(retry_tree, allowed_labels,
                                                 log=log)
    actions.extend({'event': 'retry_snap', **a} for a in snap2)
    if unresolved2:
        actions.append({'event': 'retry_unresolved',
                        'unresolved': unresolved2})
        log(f"[validator] retry still has unresolved labels: "
            f"{unresolved2} (fail-closed)")
    return retry_tree, actions

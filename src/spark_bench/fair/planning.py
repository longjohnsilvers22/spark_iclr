"""
Planning, diverse-candidate generation, label validation, and the safety gate.
"""
from __future__ import annotations

import yaml

from .config import (
    FairConfig,
    Optional,
    WMSafetyGate,
    _gemini_plan,
    _gemini_plan_via_dsl,
    _scripted_plan,
    np,
    validate_and_repair_bt,
)

try:
    from spark_real.world_model.vjepa2_ac import load_vjepa2_ac
except ImportError:
    load_vjepa2_ac = None

try:
    from spark_bench.trace_selector.select import TraceModel, rank_plans
except ImportError:  # trace_selector is self-contained; be defensive anyway
    TraceModel = None  # type: ignore[assignment]
    rank_plans = None  # type: ignore[assignment]


def _gemini_planner_fn(cfg: FairConfig):
    return _gemini_plan_via_dsl if getattr(cfg, 'use_dsl', False) else _gemini_plan


def _plan(cfg: FairConfig, instruction: str, det_map: dict, rgb,
           wrist_rgb, pick_hint: str, place_hint: str,
           picks: Optional[list] = None,
           places: Optional[list] = None,
           pick_counts: Optional[list] = None,
           temperature: float = 0.0,
           prompts: Optional[list] = None):
    """
    Run the configured planner; ``no_gemini`` swaps in the scripted plan.

    ``picks`` / ``places`` / ``pick_counts`` carry libero_10-style
    compound-goal info (multi-pair "put both X and Y in Z").  When
    None or 1-entry, behaviour matches the legacy single pick/place flow.

    ``temperature`` > 0 routes through Gemini sampling (no BT-library
    few-shot, no in-memory cache) for shadow-sim best-of-N diversity.
    """
    labels = list(det_map.keys())
    if cfg.no_gemini:
        return _scripted_plan(labels, pick_hint, place_hint, instruction,
                                picks=picks, places=places, det_map=det_map,
                                prompts=prompts)
    return _gemini_planner_fn(cfg)(
        instruction, labels, pick_hint, place_hint,
        det_map=det_map, rgb_image=rgb, wrist_image=wrist_rgb,
        no_bddl_hints=getattr(cfg, 'no_bddl_hints', False),
        picks=picks, places=places, pick_counts=pick_counts,
        temperature=temperature)


def _plan_diverse_candidates(cfg: FairConfig, instruction: str, det_map: dict,
                              rgb, wrist_rgb, pick_hint: str, place_hint: str,
                              *, k: int = 3,
                              diversity_temperature: float = 0.7,
                              picks: Optional[list] = None,
                              places: Optional[list] = None,
                              pick_counts: Optional[list] = None) -> list[dict]:
    """
    Return up to ``k`` distinct BT scores: first temp=0 (cache-friendly),
    rest sampled at ``diversity_temperature``. De-duplicated by canonical
    YAML string. Used by shadow-sim best-of-N.
    """
    seen: set[str] = set()
    out: list[dict] = []

    def _is_useful(score) -> bool:
        if not isinstance(score, dict):
            return False
        tree = score.get('tree') if 'tree' in score else score
        children = (tree or {}).get('children') if isinstance(tree, dict) else None
        return bool(children)

    def _push(score):
        if not _is_useful(score):
            return
        # Canonicalize on raw YAML so different in-memory dicts dedup.
        try:
            canon = yaml.safe_dump({'tree': score.get('tree', score)},
                                     sort_keys=True)
        except Exception:
            canon = str(score)
        if canon in seen:
            return
        seen.add(canon)
        out.append(score)

    # First call: deterministic (temp=0) so the cache fast-path stays warm.
    s0 = _plan(cfg, instruction, det_map, rgb, wrist_rgb,
                pick_hint, place_hint, picks=picks, places=places,
                pick_counts=pick_counts, temperature=0.0)
    _push(s0)

    # Diversity samples: bump temperature so Gemini explores alternative
    # primitive choices / keypoint label choices for the same instruction.
    for _ in range(max(0, k - 1)):
        si = _plan(cfg, instruction, det_map, rgb, wrist_rgb,
                    pick_hint, place_hint, picks=picks, places=places,
                    pick_counts=pick_counts,
                    temperature=diversity_temperature)
        _push(si)

    return out


def _run_label_validator(cfg: FairConfig, score: dict, det_map: dict,
                          instruction: str, rgb, wrist_rgb,
                          pick_hint: str, place_hint: str,
                          trial_meta: Optional[dict] = None,
                          slot_key: str = 'validator_actions',
                          picks: Optional[list] = None,
                          places: Optional[list] = None,
                          pick_counts: Optional[list] = None) -> dict:
    """
    Hard post-parse label validator + fuzzy-snap + at-most-one retry.

    Walks the BT and rewrites any ``keypoint_label`` / ``target_label`` /
    ``label`` / ``labels`` / ``pick_label`` / ``place_label`` value that
    is not in ``det_map.keys()``.  Strings within Levenshtein cutoff 0.4
    are snapped in place; the rest trigger ONE retry with a sharper
    STRICT-mode prompt suffix appended to the instruction.

    Returns the (possibly rewritten) score.  Records snap / retry events
    into ``trial_meta[slot_key]`` when ``trial_meta`` is provided.
    """
    if not isinstance(score, dict) or not det_map:
        return score
    allowed = set(det_map.keys()) | {'none'}

    def _retry_fn(suffix: str) -> Optional[dict]:
        # One sharper-prompt call.  The STRICT preamble is prepended so the
        # model sees the allowed-labels list BEFORE the task instruction.
        retry_instruction = suffix + "\n\n" + instruction
        return _plan(cfg, retry_instruction, det_map, rgb, wrist_rgb,
                      pick_hint, place_hint,
                      picks=picks, places=places, pick_counts=pick_counts)

    log_fn = print if cfg.verbose else (lambda _msg: None)
    repaired, actions = validate_and_repair_bt(
        score, allowed, retry_fn=_retry_fn, log=log_fn)
    if trial_meta is not None and actions:
        trial_meta.setdefault(slot_key, []).extend(actions)
    return repaired


# WM-safety-gate plumbing

_VJEPA_INSTANCE: object = None  # type: ignore[assignment]
_TRACE_MODEL_CACHE: dict = {}


def _maybe_trace_ranker(cfg: FairConfig, det_map: dict, instruction: str,
                         pick_hint: str, place_hint: str):
    """
    Build the trace-model ranker callable for selection='trace_model'.

    Returns ``None`` (gate falls back to shield ranking) when the mode is
    off, the module is unavailable, or the artifact fails to load -
    fail-open by design, and default-off via ``cfg.selection``.
    """
    if getattr(cfg, 'selection', 'shield') != 'trace_model':
        return None
    if TraceModel is None or rank_plans is None:
        return None
    path = getattr(cfg, 'trace_model_path', '')
    model = _TRACE_MODEL_CACHE.get(path)
    if model is None:
        try:
            model = TraceModel.load(path)
        except Exception as e:
            if cfg.verbose:
                print(f"[trace_model] artifact load failed ({e}); "
                      f"falling back to shield ranking")
            return None
        _TRACE_MODEL_CACHE[path] = model

    def _ranker(bts: list) -> list:
        return rank_plans(bts, det_map, model=model,
                           instruction=instruction,
                           pick_hint=pick_hint, place_hint=place_hint)

    return _ranker


def _maybe_load_vjepa(cfg: FairConfig):
    """
    Load V-JEPA 2-AC if --wm-load-vjepa is set.  Returns None on failure.
    """
    global _VJEPA_INSTANCE
    if not (getattr(cfg, 'wm_safety_gate', False)
            and getattr(cfg, 'wm_load_vjepa', False)):
        return None
    if _VJEPA_INSTANCE is not None:
        return _VJEPA_INSTANCE
    if load_vjepa2_ac is None:
        return None
    try:
        _VJEPA_INSTANCE = load_vjepa2_ac()
        return _VJEPA_INSTANCE
    except Exception as e:
        if cfg.verbose:
            print(f"[wm_safety_gate] V-JEPA load failed ({e}); "
                  f"falling back to Stage 1 (kinematic) only")
        return None


def _gated_plan(cfg: FairConfig, instruction: str, det_map: dict, rgb,
                 wrist_rgb, pick_hint: str, place_hint: str,
                 ee_init_xyz: np.ndarray,
                 goal_obs: Optional[np.ndarray] = None,
                 picks: Optional[list] = None,
                 places: Optional[list] = None,
                 pick_counts: Optional[list] = None):
    """
    Sample K candidate BTs and pick the safest via the WM safety gate.

    Returns the chosen BT score dict, or ``None`` if every candidate failed
    the kinematic shield (caller treats this as "trigger recovery").
    """
    K = max(1, int(getattr(cfg, 'wm_safety_gate_k', 4)))
    candidates: list[dict] = []
    # Generation matches the shadow arm: candidate 0 is the temperature-0
    # primary, the rest are temperature-0.3 diversity samples. Generating
    # every candidate at temperature 0 gives near-duplicates and confounds
    # the selection-rule comparison against the shadow arm.
    for i in range(K):
        score = _plan(cfg, instruction, det_map, rgb, wrist_rgb,
                       pick_hint, place_hint, picks=picks, places=places,
                       pick_counts=pick_counts,
                       temperature=(0.0 if i == 0 else 0.3))
        if score is not None:
            candidates.append(score)
    if not candidates:
        return None

    # Build the keypoint_xyz map from detections for the FK trace.
    keypoint_xyz: dict = {}
    for lbl, det in det_map.items():
        if getattr(det, 'position_3d', None) is not None:
            keypoint_xyz[lbl] = np.asarray(det.position_3d, dtype=np.float32)

    ranker = _maybe_trace_ranker(cfg, det_map, instruction,
                                  pick_hint, place_hint)
    gate = WMSafetyGate(
        z_min=float(getattr(cfg, 'wm_unsafe_workspace_z_min', 0.78)),
        z_max=float(getattr(cfg, 'wm_unsafe_workspace_z_max', 1.30)),
        xy_radius=float(getattr(cfg, 'wm_unsafe_workspace_xy_radius', 1.5)),
        wm=_maybe_load_vjepa(cfg),
        ranker=ranker,
    )
    result = gate.evaluate_candidates(
        candidates,
        ee_init_xyz=ee_init_xyz,
        keypoint_xyz=keypoint_xyz,
        current_obs=rgb,
        goal_obs=goal_obs,
    )
    if cfg.verbose:
        print(f"[wm_safety_gate] {result.n_passed}/{len(candidates)} "
              f"candidates passed shield "
              f"(elapsed={result.total_time_s:.2f}s)")
    # Selection: the shield VETOES, it does not rank. The gate's
    # ascending-action-length ranking measured 49.5 overall against the
    # 62.5 plan-once baseline on spatial: shortest-plan preference
    # systematically picks plans with missing steps. The deployable rule
    # is planner order among passing candidates, so the temperature-0
    # primary wins unless the shield kills it.
    #
    # selection='trace_model' is the one sanctioned exception: when the
    # trace-model ranker was requested AND actually scored the passing
    # candidates (finite trace_prob on the gate's best), the gate's
    # ranked best_idx wins. If the ranker was off, failed to load, or
    # raised mid-rank (trace_prob stays NaN), selection falls back to
    # planner order - never to the retired shortest-plan ranking.
    passed = [i for i, cs in enumerate(result.candidates)
              if bool(getattr(cs, 'safe', False))]
    import math as _math
    ranked_mode = (
        ranker is not None and result.best_idx >= 0
        and not _math.isnan(
            float(getattr(result.candidates[result.best_idx],
                          'trace_prob', float('nan')))))
    chosen_idx = (result.best_idx if ranked_mode
                  else (passed[0] if passed else -1))
    # Full candidate persistence for the trace-selector corpus: one JSON
    # line per gated trial with every candidate's YAML and shield verdict.
    # The [bt-emit] log line truncates YAML at ~250 chars, which made
    # losing candidates unminable; this line is the durable record.
    try:
        import json as _json
        import yaml as _yaml
        record = {
            'chosen_idx': int(chosen_idx),
            'candidates': [
                {
                    'yaml': _yaml.safe_dump(
                        {k: v for k, v in cs.bt.items()
                         if not str(k).startswith('__')},
                        sort_keys=False),
                    'safe': bool(getattr(cs, 'safe', True)),
                    'unsafe_reason': str(getattr(cs, 'unsafe_reason', '')),
                    # Per-candidate ranker probability (trace-selector
                    # request): None unless selection='trace_model' ran.
                    'trace_prob': (
                        None if _math.isnan(float(getattr(
                            cs, 'trace_prob', float('nan'))))
                        else float(cs.trace_prob)),
                }
                for cs in result.candidates],
        }
        print(f"[gate-candidates] {_json.dumps(record)}", flush=True)
    except Exception:
        pass
    if chosen_idx < 0:
        return None
    return result.candidates[chosen_idx].bt



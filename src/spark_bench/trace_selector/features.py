"""
Pre-execution feature extraction for the predicate-trace selection model.

Every feature is computable BEFORE execution from:
  - the plan (BT score dict or raw YAML string),
  - the detection context (det_map at deploy time; the SAM3 prompt
    vocabulary as a mining-time proxy),
  - the task instruction and BDDL pick/place hints,
  - the label-validator action log (runs pre-execution in the harness).

Determinism contract: ``extract_features`` returns a dict with exactly the
keys in ``FEATURE_NAMES``, in that order, for any input.  Missing /
inapplicable numeric context yields ``float('nan')`` (imputed at train
time; the trained artifact records the imputation values).

Only stdlib + PyYAML - importable in spark_conda and on the live path.
"""
from __future__ import annotations

import difflib
import math
from typing import Optional

import yaml

# Leaf primitive vocabulary (mine.py verifies coverage and buckets anything
# new into n_other_leaf).
_LEAF_TYPES = (
    'move_to_keypoint', 'grasp', 'release', 'move_relative',
    'verify_grasp', 'verify_placed', 'search_keypoint', 'push_object',
    'open_drawer', 'turn_knob', 'grasp_se3', 'pull', 'wait',
)
_CONTROL_TYPES = ('sequence', 'fallback', 'retry', 'selector', 'parallel')

_VERIFY_PREDS = ('inside', 'held', 'on', 'near', 'stacked', 'absent',
                 'removed_from')

PLAN_FEATURES = [
    'n_actions',
    *[f'n_{t}' for t in _LEAF_TYPES],
    'n_other_leaf',
    'n_sequence', 'n_fallback', 'n_retry',
    'retry_max_attempts', 'tree_depth',
    'first_offset_z', 'max_offset_z', 'mean_offset_z', 'place_offset_z',
    'mr_total_abs_dz', 'mr_max_mag', 'grasp_force', 'grasp_target_width',
    'release_tilt_max',
    'has_verify_block', 'n_verify_clauses',
    *[f'verify_has_{p}' for p in _VERIFY_PREDS],
    'n_labels', 'n_unique_labels', 'has_label_corrections',
]

GROUNDING_FEATURES = [
    'frac_labels_in_vocab', 'n_labels_missing',
    'pick_hint_sim', 'place_hint_sim',
    'n_validator_snaps', 'n_validator_invalid', 'n_validator_retries',
    # det_map-dependent features: NaN when only the prompt vocabulary is
    # available (mining-time logs without det_map).  Populated on the live path.
    'det_mean_conf', 'det_min_conf', 'det_pick_conf', 'det_place_conf',
    'det_pick_place_dist', 'det_pick_count', 'det_place_count',
]

TASK_FEATURES = [
    'instr_n_words', 'instr_has_and', 'instr_has_both', 'instr_has_then',
    'instr_open_close', 'instr_turn', 'instr_push', 'instr_drawer',
    'instr_stove',
]

FEATURE_NAMES = PLAN_FEATURES + GROUNDING_FEATURES + TASK_FEATURES

FEATURE_GROUPS = {
    'plan': PLAN_FEATURES,
    'grounding': GROUNDING_FEATURES,
    'task': TASK_FEATURES,
}


def _norm_label(s: str) -> str:
    return str(s).lower().replace('_', ' ').strip()


def _sim(a: str, b: str) -> float:
    if not a or not b:
        return float('nan')
    return difflib.SequenceMatcher(None, _norm_label(a), _norm_label(b)).ratio()


def _best_vocab_sim(label: str, vocab: list[str]) -> float:
    if not vocab:
        return float('nan')
    return max(_sim(label, v) for v in vocab)


def parse_plan(plan) -> Optional[dict]:
    """YAML string or score dict -> score dict (None if unparseable)."""
    if isinstance(plan, dict):
        return plan
    if isinstance(plan, str):
        try:
            out = yaml.safe_load(plan)
        except yaml.YAMLError:
            return None
        return out if isinstance(out, dict) else None
    return None


def _walk(node, depth, acc):
    """Pre-order traversal collecting leaves, controls, and depth."""
    if not isinstance(node, dict):
        return
    t = node.get('type')
    children = node.get('children') or []
    acc['max_depth'] = max(acc['max_depth'], depth)
    if t in _CONTROL_TYPES:
        acc['controls'].append((t, node.get('params') or {}))
        for c in children:
            _walk(c, depth + 1, acc)
    elif t is not None:
        acc['leaves'].append((t, node.get('params') or {}))
        for c in children:  # defensive: leaves should be childless
            _walk(c, depth + 1, acc)
    else:
        for c in children:
            _walk(c, depth + 1, acc)


def extract_features(plan,
                     *,
                     prompts: Optional[list] = None,
                     det_map: Optional[dict] = None,
                     instruction: str = '',
                     pick_hint: str = '',
                     place_hint: str = '',
                     validator_actions: Optional[list] = None) -> dict:
    """
    Extract the full pre-execution feature vector for one candidate plan.

    Args:
        plan: BT score dict (with 'tree') or raw YAML string.
        prompts: SAM3 prompt vocabulary (mining-time grounding proxy).
        det_map: label -> detection object/dict with optional
            ``confidence`` and ``position_3d`` attributes or keys
            (deploy-time grounding).  Overrides ``prompts`` for the
            vocabulary when provided.
        instruction: task language instruction.
        pick_hint / place_hint: BDDL-derived object hints ('' if unknown).
        validator_actions: label-validator event list for THIS plan
            (None when unavailable, e.g. shadow diversity candidates).

    Returns:
        dict with exactly FEATURE_NAMES keys, in order.
    """
    f = {k: float('nan') for k in FEATURE_NAMES}
    score = parse_plan(plan)

    # --- task features (independent of plan parse) ---
    instr = (instruction or '').lower()
    words = instr.split()
    f['instr_n_words'] = float(len(words))
    f['instr_has_and'] = float(' and ' in f' {instr} ')
    f['instr_has_both'] = float('both' in words)
    f['instr_has_then'] = float('then' in words)
    f['instr_open_close'] = float(any(w in words for w in ('open', 'close')))
    f['instr_turn'] = float('turn' in words)
    f['instr_push'] = float('push' in words)
    f['instr_drawer'] = float('drawer' in instr)
    f['instr_stove'] = float('stove' in instr)

    # --- validator features (0 when a log was provided but empty) ---
    if validator_actions is not None:
        events = [a.get('event') for a in validator_actions
                  if isinstance(a, dict)]
        f['n_validator_snaps'] = float(events.count('snap'))
        f['n_validator_retries'] = float(
            sum(1 for e in events if e and 'retry' in e))
        n_invalid = 0
        for a in validator_actions:
            if isinstance(a, dict) and a.get('event') == 'invalid_labels_detected':
                n_invalid += len(a.get('invalid') or [])
        f['n_validator_invalid'] = float(n_invalid)
    else:
        f['n_validator_snaps'] = float('nan')
        f['n_validator_retries'] = float('nan')
        f['n_validator_invalid'] = float('nan')

    if score is None:
        return f

    acc = {'leaves': [], 'controls': [], 'max_depth': 0}
    _walk(score.get('tree') or {}, 1, acc)
    leaves = acc['leaves']

    # --- plan-structure features ---
    f['n_actions'] = float(len(leaves))
    known = set(_LEAF_TYPES)
    for t in _LEAF_TYPES:
        f[f'n_{t}'] = float(sum(1 for lt, _ in leaves if lt == t))
    f['n_other_leaf'] = float(sum(1 for lt, _ in leaves if lt not in known))
    for t in ('sequence', 'fallback', 'retry'):
        f[f'n_{t}'] = float(sum(1 for ct, _ in acc['controls'] if ct == t))
    retry_attempts = [p.get('max_attempts') for ct, p in acc['controls']
                      if ct == 'retry' and isinstance(p.get('max_attempts'),
                                                      (int, float))]
    f['retry_max_attempts'] = (float(max(retry_attempts))
                               if retry_attempts else 0.0)
    f['tree_depth'] = float(acc['max_depth'])

    def _num(v):
        return float(v) if isinstance(v, (int, float)) and math.isfinite(v) \
            else None

    offsets, labels = [], []
    pick_label, place_label = None, None
    last_move_label, last_move_offset = None, None
    seen_grasp = False
    place_offset = None
    mr_abs_dz, mr_mags = 0.0, []
    grasp_force, grasp_width = None, None
    tilt_max = None
    for lt, p in leaves:
        if lt == 'move_to_keypoint':
            lbl = p.get('keypoint_label')
            if lbl is not None:
                labels.append(str(lbl))
                last_move_label = str(lbl)
            oz = _num(p.get('offset_z'))
            last_move_offset = oz
            if oz is not None:
                offsets.append(oz)
            if not seen_grasp and pick_label is None and lbl is not None:
                pick_label = str(lbl)
        elif lt in ('grasp', 'grasp_se3'):
            seen_grasp = True
            if grasp_force is None:
                grasp_force = _num(p.get('force'))
            if grasp_width is None:
                grasp_width = _num(p.get('target_width'))
        elif lt == 'release':
            if place_label is None and last_move_label is not None:
                place_label = last_move_label
                place_offset = last_move_offset
            tilt = _num(p.get('tilt_angle'))
            if tilt is not None:
                tilt_max = max(abs(tilt), tilt_max or 0.0)
        elif lt == 'move_relative':
            dx = _num(p.get('dx')) or 0.0
            dy = _num(p.get('dy')) or 0.0
            dz = _num(p.get('dz')) or 0.0
            mr_abs_dz += abs(dz)
            mr_mags.append(math.sqrt(dx * dx + dy * dy + dz * dz))
        elif lt == 'search_keypoint':
            lbl = p.get('keypoint_label')
            if lbl is not None:
                labels.append(str(lbl))

    if offsets:
        f['first_offset_z'] = offsets[0]
        f['max_offset_z'] = max(offsets)
        f['mean_offset_z'] = sum(offsets) / len(offsets)
    f['place_offset_z'] = (place_offset if place_offset is not None
                           else float('nan'))
    f['mr_total_abs_dz'] = mr_abs_dz
    f['mr_max_mag'] = max(mr_mags) if mr_mags else 0.0
    f['grasp_force'] = grasp_force if grasp_force is not None else float('nan')
    f['grasp_target_width'] = (grasp_width if grasp_width is not None
                               else float('nan'))
    f['release_tilt_max'] = tilt_max if tilt_max is not None else 0.0

    # --- verify-block features (top-level 'verify' key) ---
    vb = score.get('verify')
    clauses = []
    if isinstance(vb, dict):
        for key in ('all', 'any'):
            for cl in vb.get(key) or []:
                if isinstance(cl, dict):
                    clauses.append(cl)
    f['has_verify_block'] = float(bool(clauses))
    f['n_verify_clauses'] = float(len(clauses))

    def _pred_name(v):
        # YAML 1.1 parses a bare ``on`` as boolean True.
        if v is True:
            return 'on'
        if v is False:
            return 'off'
        return str(v)

    preds = {_pred_name(cl.get('pred')) for cl in clauses}
    for p in _VERIFY_PREDS:
        f[f'verify_has_{p}'] = float(p in preds)

    f['n_labels'] = float(len(labels))
    f['n_unique_labels'] = float(len(set(labels)))
    f['has_label_corrections'] = float('label_corrections' in score)

    # --- grounding features ---
    vocab = None
    det_info = {}
    if det_map:
        vocab = [str(k) for k in det_map.keys()]
        for k, d in det_map.items():
            conf = getattr(d, 'confidence', None)
            if conf is None and isinstance(d, dict):
                # 'conf' is the trial_meta det_summary spelling
                conf = d.get('confidence', d.get('conf'))
            pos = getattr(d, 'position_3d', None)
            if pos is None and isinstance(d, dict):
                # 'pos' is the trial_meta det_summary spelling
                pos = d.get('position_3d', d.get('pos'))
            det_info[str(k)] = {'confidence': conf, 'position_3d': pos}
    elif prompts:
        vocab = [str(p) for p in prompts]

    uniq = sorted(set(labels))
    if vocab is not None and uniq:
        sims = {l: _best_vocab_sim(l, vocab) for l in uniq}
        in_vocab = [l for l in uniq
                    if _norm_label(l) in {_norm_label(v) for v in vocab}
                    or sims[l] >= 0.85]
        f['frac_labels_in_vocab'] = len(in_vocab) / len(uniq)
        f['n_labels_missing'] = float(len(uniq) - len(in_vocab))
    f['pick_hint_sim'] = _sim(pick_label or '', pick_hint or '')
    f['place_hint_sim'] = _sim(place_label or '', place_hint or '')

    if det_info:
        confs = [v['confidence'] for v in det_info.values()
                 if isinstance(v['confidence'], (int, float))]
        if confs:
            f['det_mean_conf'] = float(sum(confs) / len(confs))
            f['det_min_conf'] = float(min(confs))

        def _lookup(lbl):
            if lbl is None:
                return None
            if lbl in det_info:
                return det_info[lbl]
            best, best_s = None, 0.0
            for k, v in det_info.items():
                s = _sim(lbl, k)
                if s > best_s:
                    best, best_s = v, s
            return best if best_s >= 0.85 else None

        pk, pl = _lookup(pick_label), _lookup(place_label)
        if pk and isinstance(pk.get('confidence'), (int, float)):
            f['det_pick_conf'] = float(pk['confidence'])
        if pl and isinstance(pl.get('confidence'), (int, float)):
            f['det_place_conf'] = float(pl['confidence'])
        if (pk and pl and pk.get('position_3d') is not None
                and pl.get('position_3d') is not None):
            try:
                a, b = pk['position_3d'], pl['position_3d']
                f['det_pick_place_dist'] = math.sqrt(
                    sum((float(a[i]) - float(b[i])) ** 2 for i in range(3)))
            except (TypeError, ValueError, IndexError):
                pass
        # detection multiplicity: count vocab entries fuzzily matching label
        if vocab:
            if pick_label:
                f['det_pick_count'] = float(sum(
                    1 for v in vocab if _sim(pick_label, v) >= 0.85))
            if place_label:
                f['det_place_count'] = float(sum(
                    1 for v in vocab if _sim(place_label, v) >= 0.85))

    return f

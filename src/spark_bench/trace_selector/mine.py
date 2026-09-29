"""
Mine the P0-a trial JSONs into a flat (trial, plan) dataset.

One row per (trial, plan-with-a-recorded-outcome):
  - baseline arms: the primary plan (outcome = episode success WITHOUT
    recovery) and each recovery re-plan (outcome = last-recovery-and-
    episode-succeeded).
  - shadow arms: each candidate whose full YAML AND shadow rollout outcome
    are both recorded.  The harness early-exits on the first succeeding
    rollout and overwrites ``bt_yaml`` with the winner, so per trial
    exactly ONE candidate has a full YAML: candidate 0 when winner_idx==0,
    else candidate 1 (its outcome = rollouts[1].success).  Candidate 0's
    YAML is LOST whenever its rollout failed (the run logs truncate
    [bt-emit] YAML at ~250 chars) - those censored candidates are counted
    in the datasheet, and eval_selection.py reconstructs a proxy for them.

Outputs (to --out, default src/spark_bench/trace_selector/data/):
  dataset.csv    one row per (trial, plan): metadata + FEATURE_NAMES
  plans.json     sha1 -> raw YAML for every plan referenced
  pairs.csv      shadow K=2 pairs for off-policy eval (with proxy c0 refs)
  datasheet.json row counts per suite/arm/kind, class balance, censoring

Runs on stdlib + PyYAML only (spark_conda-safe):
  PYTHONPATH=src python -m spark_bench.trace_selector.mine \
      --data-root <repo>/videos/libero_pro_fair
"""
from __future__ import annotations

import argparse
import collections
import csv
import hashlib
import json
import os
from typing import Optional

from .features import FEATURE_NAMES, extract_features

META_COLS = ['suite', 'arm', 'perturbation', 'task_name', 'trial',
             'row_kind', 'cand_idx', 'outcome', 'plan_sha', 'instruction',
             'shield_safe', 'has_det_summary']

# (dirname, suite, arm) triples relative to --data-root.  goal_shadow_k2 is
# picked up automatically if it has appeared by mining time.
ARMS = [
    ('p0a_spatial_baseline', 'spatial', 'baseline'),
    ('p0a_object_baseline', 'object', 'baseline'),
    ('p0a_goal_baseline', 'goal', 'baseline'),
    ('p0a_spatial_shadow_k2', 'spatial', 'shadow_k2'),
    ('p0a_object_shadow_k2', 'object', 'shadow_k2'),
    ('p0a_goal_shadow_k2', 'goal', 'shadow_k2'),
]

# Arm C runs: gated path with full [gate-candidates] persistence (every
# candidate's YAML + shield verdict, one JSON line per gated trial) and
# per-trial det_summary snapshots in trial_meta.  (json dirname, suite,
# arm tag, log filename) - v3's JSON was written by the v3b log (the v3
# log aborted after 31 trials and the run was restarted).
ARMC_RUNS = [
    ('p0a_spatial_armC_gate', 'spatial', 'armC_gate',
     'p0a_spatial_armC_gate.log'),
    ('p0a_spatial_armC_v2', 'spatial', 'armC_v2', 'p0a_spatial_armC_v2.log'),
    ('p0a_spatial_armC_v3', 'spatial', 'armC_v3', 'p0a_spatial_armC_v3b.log'),
    ('p0a_spatial_armC_v4', 'spatial', 'armC_v4', 'p0a_spatial_armC_v4.log'),
    ('p0a_spatial_armC_v5', 'spatial', 'armC_v5', 'p0a_spatial_armC_v5.log'),
    ('p0a_spatial_armC_v6', 'spatial', 'armC_v6', 'p0a_spatial_armC_v6.log'),
]

_REPO = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..', '..'))
DEFAULT_LOG_ROOT = os.path.join(_REPO, 'src', 'spark_bench', 'results', 'libero_pro_logs')

GATE_TAG = '[gate-candidates] '


def parse_gate_candidates(log_path: str) -> list[dict]:
    """Parse the [gate-candidates] JSON lines from an arm C run log."""
    records = []
    with open(log_path, errors='replace') as fh:
        for line in fh:
            if line.startswith(GATE_TAG):
                try:
                    records.append(json.loads(line[len(GATE_TAG):]))
                except json.JSONDecodeError:
                    pass
    return records


def _trees_equal(yaml_a: str, yaml_b: str) -> bool:
    """Loose structural check: same 'tree' after parsing (label snaps by
    the post-gate validator can still make these differ)."""
    import yaml as _yaml
    try:
        a = _yaml.safe_load(yaml_a)
        b = _yaml.safe_load(yaml_b)
    except Exception:
        return False
    if not (isinstance(a, dict) and isinstance(b, dict)):
        return False
    return a.get('tree') == b.get('tree')


def _sha(yaml_str: str) -> str:
    return hashlib.sha1(yaml_str.encode('utf-8')).hexdigest()[:16]


def _load_arm(path: str) -> Optional[dict]:
    if not os.path.isfile(path):
        return None
    with open(path) as fh:
        return json.load(fh)


def _iter_tasks(arm_json: dict):
    for pert, pd in arm_json.get('perturbations', {}).items():
        for td in pd.get('task_details', []):
            yield pert, td


def mine(data_root: str, log_root: str = DEFAULT_LOG_ROOT):
    rows: list[dict] = []
    plans: dict[str, str] = {}
    pairs: list[dict] = []
    sheet = {
        'arms_found': [], 'arms_missing': [],
        'rows_by': collections.Counter(),
        'outcome_by': collections.Counter(),
        'censored_shadow_candidates': 0,
        'shadow_pairs_total': 0,
        'shadow_pairs_disagreeing': 0,
        'skipped_no_yaml': 0,
        'skipped_unparseable': 0,
        'armc_join': {},
    }

    # First pass on shadow arms: collect candidate-0 plans (winner_idx==0
    # trials) per (suite, pert, task) for proxy reconstruction; also
    # baseline primary plans per (suite, pert, task) as fallback proxies.
    c0_pool: dict = collections.defaultdict(collections.Counter)
    base_pool: dict = collections.defaultdict(collections.Counter)

    arm_data = []
    for dirname, suite, arm in ARMS:
        fname = os.path.join(data_root, dirname, f'{suite}.json')
        d = _load_arm(fname)
        if d is None:
            sheet['arms_missing'].append(dirname)
            continue
        sheet['arms_found'].append(dirname)
        arm_data.append((suite, arm, d))
        for pert, td in _iter_tasks(d):
            key = (suite, pert, td.get('task_name'))
            for tm in td.get('trial_meta', []):
                y = tm.get('bt_yaml')
                if not y:
                    continue
                if arm == 'baseline':
                    base_pool[key][y] += 1
                elif (tm.get('shadow_sim') or {}).get('winner_idx') == 0:
                    c0_pool[key][y] += 1

    def _add_row(suite, arm, pert, td, tm, *, row_kind, cand_idx, outcome,
                 yaml_str, validator_actions, det_map=None, shield_safe=None):
        """``outcome=None`` emits an UNLABELED context row (empty cell)."""
        score_ok = yaml_str is not None
        if not score_ok:
            sheet['skipped_no_yaml'] += 1
            return
        sha = _sha(yaml_str)
        plans.setdefault(sha, yaml_str)
        feats = extract_features(
            yaml_str,
            prompts=td.get('prompts'),
            det_map=det_map,
            instruction=td.get('instruction', ''),
            pick_hint=td.get('pick', ''),
            place_hint=td.get('place', ''),
            validator_actions=validator_actions,
        )
        if feats['n_actions'] != feats['n_actions']:  # NaN -> unparseable
            sheet['skipped_unparseable'] += 1
            return
        row = {
            'suite': suite, 'arm': arm, 'perturbation': pert,
            'task_name': td.get('task_name'), 'trial': tm.get('trial'),
            'row_kind': row_kind, 'cand_idx': cand_idx,
            'outcome': '' if outcome is None else int(bool(outcome)),
            'plan_sha': sha,
            'instruction': td.get('instruction', ''),
            'shield_safe': ('' if shield_safe is None
                            else int(bool(shield_safe))),
            'has_det_summary': int(det_map is not None),
        }
        row.update(feats)
        rows.append(row)
        sheet['rows_by'][f'{suite}/{arm}/{row_kind}'] += 1
        tag = ('unlabeled' if outcome is None
               else ('pos' if outcome else 'neg'))
        sheet['outcome_by'][f'{suite}/{arm}/{row_kind}/{tag}'] += 1

    for suite, arm, d in arm_data:
        for pert, td in _iter_tasks(d):
            key = (suite, pert, td.get('task_name'))
            for tm in td.get('trial_meta', []):
                success = bool(tm.get('success'))
                recovery = tm.get('bt_yaml_recovery') or []
                ss = tm.get('shadow_sim')

                if arm == 'baseline' or ss is None:
                    # Primary plan: succeeded iff the episode succeeded
                    # without needing a recovery re-plan.
                    if tm.get('bt_yaml'):
                        _add_row(suite, arm, pert, td, tm,
                                 row_kind='primary', cand_idx=0,
                                 outcome=success and not recovery,
                                 yaml_str=tm['bt_yaml'],
                                 validator_actions=tm.get('validator_actions'))
                else:
                    # Shadow arm: label by the recorded rollout outcome of
                    # the ONE candidate whose full YAML survived.
                    rollouts = ss.get('rollouts') or []
                    widx = ss.get('winner_idx')
                    if tm.get('bt_yaml') and rollouts:
                        if widx == 0:
                            cand_idx, outcome = 0, True
                        else:
                            # bt_yaml is the last candidate tried
                            last = rollouts[-1]
                            cand_idx = int(last.get('idx', len(rollouts) - 1))
                            outcome = bool(last.get('success'))
                        _add_row(suite, arm, pert, td, tm,
                                 row_kind='shadow_cand', cand_idx=cand_idx,
                                 outcome=outcome, yaml_str=tm['bt_yaml'],
                                 validator_actions=(
                                     tm.get('validator_actions')
                                     if cand_idx == 0 else None))
                        # censoring bookkeeping
                        sheet['censored_shadow_candidates'] += max(
                            0, len(rollouts) - 1)
                    # Pair record for off-policy eval (K=2, both outcomes).
                    if len(rollouts) == 2:
                        o0 = bool(rollouts[0].get('success'))
                        o1 = bool(rollouts[1].get('success'))
                        sheet['shadow_pairs_total'] += 1
                        disagree = o0 != o1
                        if disagree:
                            sheet['shadow_pairs_disagreeing'] += 1
                        c1_sha = (_sha(tm['bt_yaml'])
                                  if tm.get('bt_yaml') else '')
                        if tm.get('bt_yaml'):
                            plans.setdefault(c1_sha, tm['bt_yaml'])
                        # candidate-0 proxy: modal winner-0 plan for the
                        # same task cell, else modal baseline primary.
                        proxy_sha, proxy_src = '', 'none'
                        if c0_pool.get(key):
                            y0 = c0_pool[key].most_common(1)[0][0]
                            proxy_sha, proxy_src = _sha(y0), 'shadow_winner0'
                            plans.setdefault(proxy_sha, y0)
                        elif base_pool.get(key):
                            y0 = base_pool[key].most_common(1)[0][0]
                            proxy_sha, proxy_src = _sha(y0), 'baseline_primary'
                            plans.setdefault(proxy_sha, y0)
                        pairs.append({
                            'suite': suite, 'arm': arm, 'perturbation': pert,
                            'task_name': td.get('task_name'),
                            'trial': tm.get('trial'),
                            'instruction': td.get('instruction', ''),
                            'pick': td.get('pick', ''),
                            'place': td.get('place', ''),
                            'prompts': json.dumps(td.get('prompts') or []),
                            'outcome0': int(o0), 'outcome1': int(o1),
                            'disagree': int(disagree),
                            'c1_sha': c1_sha,
                            'c0_proxy_sha': proxy_sha,
                            'c0_proxy_source': proxy_src,
                        })

                # Recovery re-plans (both arms): plan i succeeds iff it is
                # the last attempt and the episode ended in success.
                for i, entry in enumerate(recovery):
                    y = (entry or {}).get('bt_yaml')
                    _add_row(suite, arm, pert, td, tm,
                             row_kind='recovery', cand_idx=-(i + 1),
                             outcome=success and i == len(recovery) - 1,
                             yaml_str=y,
                             validator_actions=tm.get(
                                 'validator_actions_recovery'))

    # ---- Arm C runs: [gate-candidates] log records joined to trials ----
    for dirname, suite, arm, logname in ARMC_RUNS:
        jpath = os.path.join(data_root, dirname, f'{suite}.json')
        lpath = os.path.join(log_root, logname)
        d = _load_arm(jpath)
        if d is None or not os.path.isfile(lpath):
            sheet['arms_missing'].append(dirname)
            continue
        records = parse_gate_candidates(lpath)
        # Flatten trials in file order (json preserves insertion order:
        # perturbation -> task -> trial), which is the run's execution
        # order and therefore the [gate-candidates] emission order.
        flat = [(pert, td, tm) for pert, td in _iter_tasks(d)
                for tm in td.get('trial_meta', [])]
        diag = {'n_records': len(records), 'n_trials': len(flat),
                'veto_mismatch': 0, 'tree_match': 0, 'tree_diff': 0}
        if len(records) != len(flat):
            diag['status'] = 'SKIPPED: record/trial count mismatch'
            sheet['armc_join'][arm] = diag
            continue
        # Join validation: chosen_idx == -1 (all candidates vetoed) must
        # correspond to trials with no primary bt_yaml, and vice versa.
        for rec, (_p, _td, tm) in zip(records, flat):
            if (rec.get('chosen_idx', -1) < 0) != ('bt_yaml' not in tm):
                diag['veto_mismatch'] += 1
        if diag['veto_mismatch'] > 0.05 * len(flat):
            diag['status'] = 'SKIPPED: order-join validation failed'
            sheet['armc_join'][arm] = diag
            continue
        diag['status'] = 'joined'
        sheet['arms_found'].append(dirname)

        for rec, (pert, td, tm) in zip(records, flat):
            success = bool(tm.get('success'))
            recovery = tm.get('bt_yaml_recovery') or []
            det_map = tm.get('det_summary') or None
            chosen = rec.get('chosen_idx', -1)
            for ci, cand in enumerate(rec.get('candidates') or []):
                y = cand.get('yaml')
                if ci == chosen:
                    # diagnostic: does the gate candidate match the
                    # (post-validator) bt_yaml structurally?
                    if tm.get('bt_yaml'):
                        if _trees_equal(y or '', tm['bt_yaml']):
                            diag['tree_match'] += 1
                        else:
                            diag['tree_diff'] += 1
                    # Chosen candidate carries the trial's label
                    # (success without a recovery re-plan).  Validator
                    # features are left NaN: at deploy time the gate
                    # ranks PRE-validation candidates.
                    _add_row(suite, arm, pert, td, tm,
                             row_kind='gate_chosen', cand_idx=ci,
                             outcome=success and not recovery,
                             yaml_str=y, validator_actions=None,
                             det_map=det_map,
                             shield_safe=cand.get('safe'))
                else:
                    # Unchosen candidates never ran: context only, no
                    # supervision (outcome cell left empty).
                    _add_row(suite, arm, pert, td, tm,
                             row_kind='gate_cand_unchosen', cand_idx=ci,
                             outcome=None,
                             yaml_str=y, validator_actions=None,
                             det_map=det_map,
                             shield_safe=cand.get('safe'))
            for i, entry in enumerate(recovery):
                y = (entry or {}).get('bt_yaml')
                _add_row(suite, arm, pert, td, tm,
                         row_kind='recovery', cand_idx=-(i + 1),
                         outcome=success and i == len(recovery) - 1,
                         yaml_str=y,
                         validator_actions=tm.get(
                             'validator_actions_recovery'),
                         det_map=det_map)
        sheet['armc_join'][arm] = diag

    return rows, plans, pairs, sheet


def write_outputs(out_dir: str, rows, plans, pairs, sheet) -> dict:
    os.makedirs(out_dir, exist_ok=True)
    cols = META_COLS + FEATURE_NAMES
    with open(os.path.join(out_dir, 'dataset.csv'), 'w', newline='') as fh:
        w = csv.DictWriter(fh, fieldnames=cols)
        w.writeheader()
        for r in rows:
            w.writerow(r)
    # parquet is optional sugar; CSV is the canonical artifact.
    try:
        import pandas as pd
        pd.DataFrame(rows, columns=cols).to_parquet(
            os.path.join(out_dir, 'dataset.parquet'))
    except Exception:
        pass
    with open(os.path.join(out_dir, 'plans.json'), 'w') as fh:
        json.dump(plans, fh, indent=0)
    if pairs:
        with open(os.path.join(out_dir, 'pairs.csv'), 'w', newline='') as fh:
            w = csv.DictWriter(fh, fieldnames=list(pairs[0].keys()))
            w.writeheader()
            for p in pairs:
                w.writerow(p)

    labeled = [r for r in rows if r['outcome'] != '']
    n = len(rows)
    n_lab = len(labeled)
    n_pos = sum(int(r['outcome']) for r in labeled)
    datasheet = {
        'n_rows': n,
        'n_labeled': n_lab,
        'n_unlabeled_context': n - n_lab,
        'n_positive': n_pos,
        'positive_rate': round(n_pos / n_lab, 4) if n_lab else None,
        'n_unique_plans': len(plans),
        'armc_join': sheet['armc_join'],
        'arms_found': sheet['arms_found'],
        'arms_missing': sheet['arms_missing'],
        'rows_by_suite_arm_kind': dict(sheet['rows_by']),
        'outcome_by_suite_arm_kind': dict(sheet['outcome_by']),
        'shadow_pairs_total': sheet['shadow_pairs_total'],
        'shadow_pairs_disagreeing': sheet['shadow_pairs_disagreeing'],
        'censored_shadow_candidates': sheet['censored_shadow_candidates'],
        'skipped_no_yaml': sheet['skipped_no_yaml'],
        'skipped_unparseable': sheet['skipped_unparseable'],
        'notes': [
            'Shadow selector early-exits on first success and only the '
            'winning (or last-tried) candidate YAML is persisted, so every '
            'disagreeing K=2 pair is (cand0 fail, cand1 success) by '
            'construction and cand0 full YAML is censored; run logs '
            'truncate [bt-emit] YAML so logs cannot fill the gap. '
            'pairs.csv carries a modal-plan proxy for cand0 instead.',
            'det_map is not persisted in the P0-a JSONs or logs, so '
            'det_* grounding features are NaN at mining time (prompt '
            'vocabulary is the grounding proxy); they activate on the '
            'live path where det_map exists.',
        ],
    }
    with open(os.path.join(out_dir, 'datasheet.json'), 'w') as fh:
        json.dump(datasheet, fh, indent=2)
    return datasheet


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--data-root',
                    default=os.path.join(_REPO, 'videos', 'libero_pro_fair'))
    ap.add_argument('--log-root', default=DEFAULT_LOG_ROOT)
    ap.add_argument('--out', default=os.path.join(
        os.path.dirname(__file__), 'data'))
    args = ap.parse_args()
    rows, plans, pairs, sheet = mine(args.data_root, args.log_root)
    ds = write_outputs(args.out, rows, plans, pairs, sheet)
    print(json.dumps(ds, indent=2))


if __name__ == '__main__':
    main()

"""
Off-policy selection accuracy on the shadow-arm logs.

For every shadow trial where K=2 candidates BOTH have recorded rollout
outcomes (i.e. candidate 0 failed, so candidate 1 was also rolled): when
the outcomes disagree, how often does the model rank the actual rollout
winner first?

STRUCTURAL CAVEATS (quantified in the output, do not hide them):

1. The shadow selector early-exits on the first success, so a "both
   outcomes recorded" pair exists ONLY when candidate 0 failed.  Every
   disagreeing pair is therefore (c0 fail, c1 success) by construction:
   the "always-candidate-0" baseline scores 0% on disagreeing pairs BY
   CONSTRUCTION, not by merit, and 50% (random) is the honest reference.

2. Candidate 0's full YAML is overwritten by the winner and the run logs
   truncate [bt-emit] YAML, so candidate 0 is reconstructed as a PROXY:
   the modal winner-0 plan of the same task cell, else the modal baseline
   primary plan (pairs.csv records which).  In pairs where the proxy is
   byte-identical to candidate 1's plan the model must tie (0.5 credit)
   - those pairs are feature-indistinguishable and are also reported
   separately.

3. The kinematic-shield baseline is NOT reconstructable offline: the
   shield compiles plans against per-trial keypoint 3D positions
   (det_map), which the P0-a logs do not persist.  Reported as n/a.

Implied arm-C' success rate: for every shadow trial, the trial's outcome
had the model-picked candidate run.  When the model picks a candidate
whose rollout was never recorded (it picks c1 on a winner-0 trial), the
outcome is censored - reported as optimistic/pessimistic bounds.

Runs on stdlib + PyYAML (spark_conda-safe):
  PYTHONPATH=src conda run -n spark_conda python -m spark_bench.trace_selector.eval_selection
"""
from __future__ import annotations

import argparse
import csv
import json
import os

from .features import extract_features
from .select import TraceModel

DATA_DIR = os.path.join(os.path.dirname(__file__), 'data')
ART_DIR = os.path.join(os.path.dirname(__file__), 'artifacts')


def _score(model: TraceModel, yaml_str: str, pair: dict) -> float:
    feats = extract_features(
        yaml_str,
        prompts=json.loads(pair['prompts'] or '[]'),
        instruction=pair['instruction'],
        pick_hint=pair['pick'], place_hint=pair['place'],
        validator_actions=None,
    )
    return model.score_features(feats)


def evaluate(pairs_path: str, plans_path: str, artifact_path: str) -> dict:
    with open(plans_path) as fh:
        plans = json.load(fh)
    pairs = list(csv.DictReader(open(pairs_path)))
    model = TraceModel.load(artifact_path)

    dis = [p for p in pairs if p['disagree'] == '1'
           and p['c1_sha'] and p['c0_proxy_sha']]
    dis_dropped = sum(1 for p in pairs if p['disagree'] == '1') - len(dis)

    n_correct = 0.0
    n_identical = 0
    per_suite: dict = {}
    for p in dis:
        y0, y1 = plans[p['c0_proxy_sha']], plans[p['c1_sha']]
        identical = p['c0_proxy_sha'] == p['c1_sha']
        if identical:
            n_identical += 1
            credit = 0.5  # feature-indistinguishable -> forced tie
        else:
            s0, s1 = _score(model, y0, p), _score(model, y1, p)
            # winner is candidate 1 in every recorded disagreement (see
            # module docstring); tie -> 0.5 credit.
            credit = 1.0 if s1 > s0 else (0.5 if s1 == s0 else 0.0)
        n_correct += credit
        ps = per_suite.setdefault(p['suite'], {'n': 0, 'correct': 0.0})
        ps['n'] += 1
        ps['correct'] += credit

    n = len(dis)
    n_distinct = n - n_identical
    # accuracy restricted to feature-distinguishable pairs
    distinct_correct = n_correct - 0.5 * n_identical

    # ---- implied arm-C' success rate over ALL shadow trials ----
    # Reconstruct per-trial: candidates' recorded outcomes.  pairs.csv has
    # one row per K=2 trial; winner-0 trials (single rollout) are absent
    # from pairs.csv, so re-derive them from the dataset rows.
    ds = list(csv.DictReader(open(os.path.join(DATA_DIR, 'dataset.csv'))))
    shadow_rows = [r for r in ds if r['row_kind'] == 'shadow_cand']
    pair_key = {(p['suite'], p['perturbation'], p['task_name'], p['trial']): p
                for p in pairs}
    n_trials = 0
    picked_success = 0.0
    censored = 0
    seq_success = 0  # actual shadow-arm rollout success (any candidate won)
    c0_success = 0.0      # baseline: always pick candidate 0
    random_success = 0.0  # baseline: uniform pick (0.5 credit per side)
    for r in shadow_rows:
        key = (r['suite'], r['perturbation'], r['task_name'], r['trial'])
        n_trials += 1
        p = pair_key.get(key)
        if p is None:
            # winner-0 trial: candidate 0 succeeded, candidate 1 unrecorded.
            seq_success += 1
            sha0 = r['plan_sha']
            # model chooses between c0 (known success) and... no recorded
            # alternative plan exists; the harness would have offered a
            # second sample, but it is unlogged.  Treat as a K=1 decision:
            # picking c0 -> success.  (No censoring: any ranker that can
            # only see one plan picks it.)
            picked_success += 1.0
            c0_success += 1.0
            random_success += 1.0
            continue
        o0, o1 = int(p['outcome0']), int(p['outcome1'])
        seq_success += 1 if (o0 or o1) else 0
        c0_success += o0
        random_success += 0.5 * (o0 + o1)
        if not p['c0_proxy_sha'] or not p['c1_sha']:
            censored += 1
            continue
        if p['c0_proxy_sha'] == p['c1_sha']:
            pick = 0  # tie -> planner order -> candidate 0
        else:
            y0, y1 = plans[p['c0_proxy_sha']], plans[p['c1_sha']]
            s0, s1 = _score(model, y0, p), _score(model, y1, p)
            pick = 1 if s1 > s0 else 0
        picked_success += o1 if pick == 1 else o0

    implied_sr = picked_success / n_trials if n_trials else None
    seq_sr = seq_success / n_trials if n_trials else None

    out = {
        'artifact': artifact_path,
        'disagreeing_pairs': {
            'n': n,
            'n_dropped_no_proxy': dis_dropped,
            'n_proxy_identical_to_winner': n_identical,
            'selection_accuracy_model': round(n_correct / n, 4) if n else None,
            'selection_accuracy_model_distinguishable_only': (
                round(distinct_correct / n_distinct, 4) if n_distinct else None),
            'n_distinguishable': n_distinct,
            'baseline_random': 0.5,
            'baseline_always_candidate0': 0.0,
            'baseline_always_candidate0_note': (
                '0% BY CONSTRUCTION: recorded disagreements only exist '
                'when candidate 0 failed (early-exit logging)'),
            'baseline_kinematic_shield': None,
            'baseline_kinematic_shield_note': (
                'not reconstructable offline: shield needs per-trial '
                'keypoint 3D positions (det_map), which P0-a logs do not '
                'persist'),
            'per_suite': {k: {'n': v['n'],
                              'acc': round(v['correct'] / v['n'], 4)}
                          for k, v in per_suite.items()},
        },
        'implied_arm_c_prime': {
            'n_shadow_trials': n_trials,
            'model_selection_sr': (round(implied_sr, 4)
                                   if implied_sr is not None else None),
            'always_candidate0_sr': (round(c0_success / n_trials, 4)
                                     if n_trials else None),
            'random_pick_sr': (round(random_success / n_trials, 4)
                               if n_trials else None),
            'actual_sequential_shadow_sr': (round(seq_sr, 4)
                                            if seq_sr is not None else None),
            'n_censored_trials': censored,
            'note': ('winner-0 trials contribute a K=1 decision (the only '
                     'recorded plan succeeded); sequential shadow SR is the '
                     'oracle-with-restore upper bound actually achieved by '
                     'the arm'),
        },
    }
    return out


def gbt_pair_check(pairs_path: str, plans_path: str, dataset_path: str) -> dict:
    """
    Reference check (needs sklearn): does a GBT trained on baseline-arm rows
    only beat the LR artifact on the disagreeing pairs?  Not deployable
    (no JSON artifact) - reported for calibration of the headline only.
    """
    import numpy as np
    from sklearn.ensemble import HistGradientBoostingClassifier

    from .features import FEATURE_NAMES

    ds = list(csv.DictReader(open(dataset_path)))
    tr = [r for r in ds if r['arm'] == 'baseline'
          and r['outcome'] in ('0', '1')]

    def matrix(rows):
        X = np.full((len(rows), len(FEATURE_NAMES)), np.nan)
        for i, r in enumerate(rows):
            for j, name in enumerate(FEATURE_NAMES):
                try:
                    X[i, j] = float(r.get(name, ''))
                except (TypeError, ValueError):
                    pass
        return X

    Xtr = matrix(tr)
    ytr = np.array([int(r['outcome']) for r in tr])
    m = HistGradientBoostingClassifier(max_depth=3, max_iter=150,
                                       learning_rate=0.1,
                                       min_samples_leaf=20, random_state=0)
    m.fit(Xtr, ytr)

    with open(plans_path) as fh:
        plans = json.load(fh)
    pairs = [p for p in csv.DictReader(open(pairs_path))
             if p['disagree'] == '1' and p['c1_sha'] and p['c0_proxy_sha']]
    from .features import extract_features as ef
    correct = 0.0
    n_distinct = 0
    for p in pairs:
        if p['c0_proxy_sha'] == p['c1_sha']:
            correct += 0.5
            continue
        n_distinct += 1
        rows = []
        for sha in (p['c0_proxy_sha'], p['c1_sha']):
            rows.append(ef(plans[sha],
                           prompts=json.loads(p['prompts'] or '[]'),
                           instruction=p['instruction'],
                           pick_hint=p['pick'], place_hint=p['place']))
        Xp = np.array([[r.get(n, float('nan')) for n in FEATURE_NAMES]
                       for r in rows])
        s0, s1 = m.predict_proba(Xp)[:, 1]
        correct += 1.0 if s1 > s0 else (0.5 if s1 == s0 else 0.0)
    n = len(pairs)
    return {
        'gbt_selection_accuracy': round(correct / n, 4) if n else None,
        'n_pairs': n, 'n_distinguishable': n_distinct,
        'note': 'baseline-arm-trained HistGBT reference, not deployable',
    }


def armc_cross_run_pairs(dataset_path: str) -> dict:
    """
    Pairwise selection accuracy on arm C cross-run outcome disagreements.

    The six arm C runs replay the SAME per-trial init states, so two runs'
    chosen plans for one (perturbation, task, trial) cell are two plans
    executed from the same start.  When their success-without-recovery
    outcomes disagree and the plans differ, we ask: does the model rank
    the succeeding plan above the failing one?

    Leakage-safe: for each held-out TASK, a logistic model is trained on
    every labeled row of all other tasks (same features/imputation as
    train.py), then scores that task's gate_chosen rows.  Needs sklearn.

    Honest caveat: a cross-run outcome flip can be caused by perception
    noise rather than plan quality; 0.5 is the informative null.
    """
    import itertools

    import numpy as np
    from sklearn.linear_model import LogisticRegression

    from .features import FEATURE_NAMES

    ds = [r for r in csv.DictReader(open(dataset_path))
          if r['outcome'] in ('0', '1')]

    def matrix(rows):
        X = np.full((len(rows), len(FEATURE_NAMES)), np.nan)
        for i, r in enumerate(rows):
            for j, name in enumerate(FEATURE_NAMES):
                try:
                    X[i, j] = float(r.get(name, ''))
                except (TypeError, ValueError):
                    pass
        return X

    X_all = matrix(ds)
    y_all = np.array([int(r['outcome']) for r in ds])
    tasks = np.array([r['task_name'] for r in ds])

    def fit_score(train_mask, test_idx):
        Xtr = X_all[train_mask]
        med = np.nanmedian(Xtr, axis=0)
        med = np.where(np.isfinite(med), med, 0.0)

        def fill(X):
            X = X.copy()
            idx = ~np.isfinite(X)
            X[idx] = np.take(med, np.where(idx)[1])
            return X
        Xtr = fill(Xtr)
        mu, sd = Xtr.mean(axis=0), Xtr.std(axis=0)
        sd = np.where(sd < 1e-12, 1.0, sd)
        lr = LogisticRegression(C=1.0, max_iter=2000,
                                class_weight='balanced')
        lr.fit((Xtr - mu) / sd, y_all[train_mask])
        Xte = (fill(X_all[test_idx]) - mu) / sd
        return lr.predict_proba(Xte)[:, 1]

    # score every gate_chosen row with its task held out
    probs = np.full(len(ds), np.nan)
    gc_idx = [i for i, r in enumerate(ds) if r['row_kind'] == 'gate_chosen']
    for task in sorted({ds[i]['task_name'] for i in gc_idx}):
        te = [i for i in gc_idx if ds[i]['task_name'] == task]
        tr_mask = tasks != task
        probs[te] = fit_score(tr_mask, te)

    cells: dict = {}
    for i in gc_idx:
        r = ds[i]
        cells.setdefault((r['perturbation'], r['task_name'], r['trial']),
                         []).append(i)
    n = 0
    correct = 0.0
    shortest_correct = 0.0
    per_pert: dict = {}
    for key, idxs in cells.items():
        for a, b in itertools.combinations(idxs, 2):
            ra, rb = ds[a], ds[b]
            if ra['outcome'] == rb['outcome']:
                continue
            if ra['plan_sha'] == rb['plan_sha']:
                continue
            win, lose = (a, b) if ra['outcome'] == '1' else (b, a)
            n += 1
            if probs[win] > probs[lose]:
                credit = 1.0
            elif probs[win] == probs[lose]:
                credit = 0.5
            else:
                credit = 0.0
            correct += credit
            # shortest-plan heuristic (the arm C v1 ranking)
            na = float(ds[win]['n_actions'])
            nb = float(ds[lose]['n_actions'])
            shortest_correct += 1.0 if na < nb else (0.5 if na == nb else 0.0)
            pp = per_pert.setdefault(key[0], {'n': 0, 'correct': 0.0})
            pp['n'] += 1
            pp['correct'] += credit
    return {
        'n_disagreeing_cross_run_pairs': n,
        'selection_accuracy_model_LOTO': (round(correct / n, 4)
                                          if n else None),
        'baseline_random': 0.5,
        'baseline_shortest_plan': (round(shortest_correct / n, 4)
                                   if n else None),
        'per_perturbation': {k: {'n': v['n'],
                                 'acc': round(v['correct'] / v['n'], 4)}
                             for k, v in per_pert.items()},
        'note': ('bidirectional pairs from same-init-state cross-run '
                 'replays; leave-one-task-out LR; outcome flips can also '
                 'reflect perception noise, so 0.5 is the null'),
    }


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--pairs', default=os.path.join(DATA_DIR, 'pairs.csv'))
    ap.add_argument('--plans', default=os.path.join(DATA_DIR, 'plans.json'))
    ap.add_argument('--artifact', default=os.path.join(ART_DIR, 'trace_lr.json'))
    ap.add_argument('--gbt-check', action='store_true',
                    help='also train+eval a baseline-only GBT on the pairs '
                         '(needs sklearn, e.g. the openvla_env conda env)')
    ap.add_argument('--armc-pairs', action='store_true',
                    help='also evaluate leave-one-task-out pairwise '
                         'accuracy on arm C cross-run outcome '
                         'disagreements (needs sklearn)')
    args = ap.parse_args()
    out = evaluate(args.pairs, args.plans, args.artifact)
    if args.gbt_check:
        out['gbt_reference'] = gbt_pair_check(
            args.pairs, args.plans, os.path.join(DATA_DIR, 'dataset.csv'))
    if args.armc_pairs:
        out['armc_cross_run_pairs'] = armc_cross_run_pairs(
            os.path.join(DATA_DIR, 'dataset.csv'))
    print(json.dumps(out, indent=2))


if __name__ == '__main__':
    main()

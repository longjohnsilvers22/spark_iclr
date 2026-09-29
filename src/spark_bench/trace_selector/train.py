"""
Train the predicate-trace plan-success model on the mined P0-a dataset.

Logistic regression (the deployable artifact) and a gradient-boosted
tree (metric reference), with TASK-LEVEL cross-validation
- trials of one task never split across train/test - reported as
AUROC/AUPRC per held-out suite and per feature-ablation (plan-only /
grounding-only / both).

Needs scikit-learn; spark_conda does not have it, openvla_env does:
  PYTHONPATH=src conda run -n openvla_env python -m spark_bench.trace_selector.train

The exported artifact (artifacts/trace_lr.json) is plain JSON
(feature names, imputation medians, standardization, coefficients) so
``select.py`` can score plans with numpy only - no sklearn on the live
path, fully deterministic.
"""
from __future__ import annotations

import argparse
import csv
import json
import math
import os
from collections import defaultdict

import numpy as np
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import average_precision_score, roc_auc_score
from sklearn.model_selection import GroupKFold

from .features import FEATURE_GROUPS, FEATURE_NAMES

DATA_DIR = os.path.join(os.path.dirname(__file__), 'data')
ART_DIR = os.path.join(os.path.dirname(__file__), 'artifacts')


def load_dataset(path: str):
    rows = list(csv.DictReader(open(path)))
    # Unchosen gate candidates are context-only (empty outcome cell):
    # they never executed, so they carry no supervision.
    rows = [r for r in rows if r['outcome'] in ('0', '1')]
    y = np.array([int(r['outcome']) for r in rows])
    groups = np.array([f"{r['suite']}::{r['task_name']}" for r in rows])
    suites = np.array([r['suite'] for r in rows])
    X = np.full((len(rows), len(FEATURE_NAMES)), np.nan)
    for i, r in enumerate(rows):
        for j, name in enumerate(FEATURE_NAMES):
            v = r.get(name, '')
            try:
                X[i, j] = float(v)
            except (TypeError, ValueError):
                X[i, j] = np.nan
    return X, y, groups, suites, rows


def usable_columns(X: np.ndarray) -> list[int]:
    """Drop all-NaN and zero-variance columns (mined det_* are all-NaN)."""
    keep = []
    for j in range(X.shape[1]):
        col = X[:, j]
        finite = col[np.isfinite(col)]
        if finite.size == 0:
            continue
        if np.nanstd(finite) < 1e-12:
            continue
        keep.append(j)
    return keep


def impute_standardize(Xtr, Xte):
    med = np.nanmedian(Xtr, axis=0)
    med = np.where(np.isfinite(med), med, 0.0)
    def fill(X):
        X = X.copy()
        idx = ~np.isfinite(X)
        X[idx] = np.take(med, np.where(idx)[1])
        return X
    Xtr, Xte = fill(Xtr), fill(Xte)
    mu, sd = Xtr.mean(axis=0), Xtr.std(axis=0)
    sd = np.where(sd < 1e-12, 1.0, sd)
    return (Xtr - mu) / sd, (Xte - mu) / sd, med, mu, sd


def make_models(seed=0):
    return {
        'logreg': LogisticRegression(C=1.0, max_iter=2000,
                                     class_weight='balanced'),
        'gbt': HistGradientBoostingClassifier(
            max_depth=3, max_iter=150, learning_rate=0.1,
            min_samples_leaf=20, random_state=seed),
    }


def eval_split(Xtr, ytr, Xte, yte, seed=0):
    out = {}
    Xtr_s, Xte_s, *_ = impute_standardize(Xtr, Xte)
    for name, m in make_models(seed).items():
        if name == 'gbt':
            # HGBT handles NaN natively; feed unstandardized w/ NaN
            m.fit(np.where(np.isfinite(Xtr), Xtr, np.nan), ytr)
            p = m.predict_proba(np.where(np.isfinite(Xte), Xte, np.nan))[:, 1]
        else:
            m.fit(Xtr_s, ytr)
            p = m.predict_proba(Xte_s)[:, 1]
        if len(set(yte)) < 2:
            out[name] = {'auroc': None, 'auprc': None, 'n': len(yte)}
        else:
            out[name] = {
                'auroc': round(float(roc_auc_score(yte, p)), 4),
                'auprc': round(float(average_precision_score(yte, p)), 4),
                'n': int(len(yte)), 'base_rate': round(float(yte.mean()), 4),
            }
    return out


def feature_indices(subset: str, keep: list[int]) -> list[int]:
    if subset == 'both':
        wanted = set(FEATURE_NAMES)
    elif subset == 'plan':
        wanted = set(FEATURE_GROUPS['plan']) | set(FEATURE_GROUPS['task'])
    elif subset == 'grounding':
        wanted = set(FEATURE_GROUPS['grounding']) | set(FEATURE_GROUPS['task'])
    else:
        raise ValueError(subset)
    return [j for j in keep if FEATURE_NAMES[j] in wanted]


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--data', default=os.path.join(DATA_DIR, 'dataset.csv'))
    ap.add_argument('--out', default=ART_DIR)
    ap.add_argument('--folds', type=int, default=5)
    ap.add_argument('--seed', type=int, default=0)
    ap.add_argument('--exclude-recovery', action='store_true',
                    help='Train on primary + shadow-candidate rows only '
                         '(recovery re-plans execute from a disturbed '
                         'state - a different outcome distribution).')
    ap.add_argument('--exclude-shadow', action='store_true',
                    help='Train on baseline-arm rows only.  Use this '
                         'artifact for the off-policy shadow-pair eval: '
                         'shadow candidate rows ARE the eval pairs, so '
                         'training on them leaks labels.')
    ap.add_argument('--artifact-name', default='trace_lr.json')
    args = ap.parse_args()

    X_all, y, groups, suites, _rows = load_dataset(args.data)
    if args.exclude_recovery:
        m = np.array([r['row_kind'] != 'recovery' for r in _rows])
        X_all, y, groups, suites = X_all[m], y[m], groups[m], suites[m]
        _rows = [r for r, k in zip(_rows, m) if k]
    if args.exclude_shadow:
        m = np.array([r['arm'] == 'baseline' for r in _rows])
        X_all, y, groups, suites = X_all[m], y[m], groups[m], suites[m]
        _rows = [r for r, k in zip(_rows, m) if k]
    keep = usable_columns(X_all)
    report = {'n_rows': int(len(y)), 'base_rate': round(float(y.mean()), 4),
              'n_features_usable': len(keep),
              'dropped_features': [FEATURE_NAMES[j] for j in
                                   range(len(FEATURE_NAMES)) if j not in keep],
              'ablations': {}, 'held_out_suite': {}}

    rng = np.random.RandomState(args.seed)

    for subset in ('both', 'plan', 'grounding'):
        cols = feature_indices(subset, keep)
        X = X_all[:, cols]

        # ---- task-level grouped CV (pooled predictions) ----
        gkf = GroupKFold(n_splits=args.folds)
        pooled = {name: np.zeros(len(y)) for name in ('logreg', 'gbt')}
        for tr, te in gkf.split(X, y, groups):
            Xtr_s, Xte_s, *_ = impute_standardize(X[tr], X[te])
            for name, m in make_models(args.seed).items():
                if name == 'gbt':
                    m.fit(X[tr], y[tr])
                    pooled[name][te] = m.predict_proba(X[te])[:, 1]
                else:
                    m.fit(Xtr_s, y[tr])
                    pooled[name][te] = m.predict_proba(Xte_s)[:, 1]
        report['ablations'][subset] = {
            name: {
                'auroc': round(float(roc_auc_score(y, p)), 4),
                'auprc': round(float(average_precision_score(y, p)), 4),
            } for name, p in pooled.items()
        }

        # ---- leave-one-suite-out ----
        for held in sorted(set(suites)):
            te = suites == held
            tr = ~te
            key = f'{subset}/{held}'
            report['held_out_suite'][key] = eval_split(
                X[tr], y[tr], X[te], y[te], seed=args.seed)

    # ---- final deployable artifact: logistic regression on all rows ----
    cols = feature_indices('both', keep)
    X = X_all[:, cols]
    med = np.nanmedian(X, axis=0)
    med = np.where(np.isfinite(med), med, 0.0)
    Xf = X.copy()
    nan_idx = ~np.isfinite(Xf)
    Xf[nan_idx] = np.take(med, np.where(nan_idx)[1])
    mu, sd = Xf.mean(axis=0), Xf.std(axis=0)
    sd = np.where(sd < 1e-12, 1.0, sd)
    Xs = (Xf - mu) / sd
    lr = LogisticRegression(C=1.0, max_iter=2000, class_weight='balanced')
    lr.fit(Xs, y)

    os.makedirs(args.out, exist_ok=True)
    artifact = {
        'kind': 'trace_selector_logreg',
        'version': 1,
        'feature_names': [FEATURE_NAMES[j] for j in cols],
        'impute_median': [round(float(v), 6) for v in med],
        'standardize_mean': [round(float(v), 6) for v in mu],
        'standardize_scale': [round(float(v), 6) for v in sd],
        'coef': [round(float(v), 6) for v in lr.coef_[0]],
        'intercept': round(float(lr.intercept_[0]), 6),
        'train_rows': int(len(y)),
        'train_base_rate': round(float(y.mean()), 4),
    }
    art_path = os.path.join(args.out, args.artifact_name)
    with open(art_path, 'w') as fh:
        json.dump(artifact, fh, indent=1)

    # top coefficients for the report
    order = np.argsort(-np.abs(lr.coef_[0]))
    report['top_coefficients'] = [
        {'feature': artifact['feature_names'][i],
         'coef': artifact['coef'][i]} for i in order[:15]]
    report['artifact'] = art_path

    rep_path = os.path.join(args.out, args.artifact_name.replace('.json', '') + '_report.json')
    with open(rep_path, 'w') as fh:
        json.dump(report, fh, indent=2)
    print(json.dumps(report, indent=2))


if __name__ == '__main__':
    main()

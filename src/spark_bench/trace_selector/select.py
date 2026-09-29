"""
Deployable selection rule: rank K candidate BT plans by predicted success.

Loads the JSON logistic-regression artifact exported by ``train.py`` and
scores candidates with stdlib math only - no sklearn or numpy needed,
deterministic, CPU, well under 10 ms for K<=8.

Usage on the live path:

    from spark_bench.trace_selector.select import TraceModel, rank_plans
    ranked = rank_plans(candidates, det_map, instruction=instr)
    best = candidates[ranked[0][0]]

``rank_plans`` returns ``[(candidate_index, success_prob), ...]`` sorted
by descending probability with the ORIGINAL candidate index as a stable
tie-break (so equal-probability candidates keep planner order, and
candidate 0 - the temperature-0 primary - wins exact ties).
"""
from __future__ import annotations

import json
import math
import os
from typing import Optional, Sequence

from .features import extract_features

DEFAULT_ARTIFACT = os.path.join(os.path.dirname(__file__), 'artifacts',
                                'trace_lr.json')


class TraceModel:
    """Logistic plan-success model from a plain-JSON artifact."""

    def __init__(self, artifact: dict):
        if artifact.get('kind') != 'trace_selector_logreg':
            raise ValueError(
                f"unsupported artifact kind: {artifact.get('kind')!r}")
        self.feature_names: list[str] = list(artifact['feature_names'])
        self.impute = [float(v) for v in artifact['impute_median']]
        self.mean = [float(v) for v in artifact['standardize_mean']]
        self.scale = [float(v) for v in artifact['standardize_scale']]
        self.coef = [float(v) for v in artifact['coef']]
        self.intercept = float(artifact['intercept'])
        n = len(self.feature_names)
        if not (len(self.impute) == len(self.mean) == len(self.scale)
                == len(self.coef) == n):
            raise ValueError('artifact arrays disagree in length')

    @classmethod
    def load(cls, path: str = DEFAULT_ARTIFACT) -> 'TraceModel':
        with open(path) as fh:
            return cls(json.load(fh))

    def score_features(self, feats: dict) -> float:
        """Feature dict (from ``extract_features``) -> success probability."""
        z = self.intercept
        for j, name in enumerate(self.feature_names):
            v = feats.get(name, float('nan'))
            if not isinstance(v, (int, float)) or not math.isfinite(v):
                v = self.impute[j]
            s = self.scale[j] if self.scale[j] != 0.0 else 1.0
            z += self.coef[j] * ((float(v) - self.mean[j]) / s)
        # numerically-safe sigmoid
        if z >= 0:
            return 1.0 / (1.0 + math.exp(-z))
        e = math.exp(z)
        return e / (1.0 + e)

    def score_plan(self, plan, *, det_map: Optional[dict] = None,
                   prompts: Optional[list] = None, instruction: str = '',
                   pick_hint: str = '', place_hint: str = '',
                   validator_actions: Optional[list] = None) -> float:
        feats = extract_features(
            plan, det_map=det_map, prompts=prompts, instruction=instruction,
            pick_hint=pick_hint, place_hint=place_hint,
            validator_actions=validator_actions)
        return self.score_features(feats)


_DEFAULT_MODEL: Optional[TraceModel] = None


def _default_model() -> TraceModel:
    global _DEFAULT_MODEL
    if _DEFAULT_MODEL is None:
        _DEFAULT_MODEL = TraceModel.load(DEFAULT_ARTIFACT)
    return _DEFAULT_MODEL


def rank_plans(candidate_scores: Sequence, det_map: Optional[dict] = None,
               *, model: Optional[TraceModel] = None,
               instruction: str = '', prompts: Optional[list] = None,
               pick_hint: str = '', place_hint: str = '') -> list[tuple[int, float]]:
    """
    Rank candidate BT plans by predicted execution success.

    Args:
        candidate_scores: BT score dicts (or raw YAML strings).
        det_map: label -> detection map from the live perception pass
            (confidence / position_3d used when present).
        model: a loaded TraceModel; defaults to the committed artifact.
        instruction / prompts / pick_hint / place_hint: task context.

    Returns:
        [(index, probability)] sorted by descending probability;
        ties broken by ascending original index (deterministic).
    """
    m = model if model is not None else _default_model()
    scored = []
    for i, cand in enumerate(candidate_scores):
        try:
            p = m.score_plan(cand, det_map=det_map, prompts=prompts,
                             instruction=instruction, pick_hint=pick_hint,
                             place_hint=place_hint)
        except Exception:
            p = 0.0  # unparseable plan ranks last
        scored.append((i, p))
    scored.sort(key=lambda t: (-t[1], t[0]))
    return scored

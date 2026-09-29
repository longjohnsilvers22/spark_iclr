"""Unit tests for the deployable trace-model ranker (select.py)."""
import json
import math
import time

import pytest

from spark_bench.trace_selector.features import FEATURE_NAMES
from spark_bench.trace_selector.select import (
    DEFAULT_ARTIFACT, TraceModel, rank_plans,
)


def _stub_artifact():
    """Artifact whose score is driven ONLY by n_grasp (positive coef)."""
    names = list(FEATURE_NAMES)
    n = len(names)
    return {
        'kind': 'trace_selector_logreg',
        'version': 1,
        'feature_names': names,
        'impute_median': [0.0] * n,
        'standardize_mean': [0.0] * n,
        'standardize_scale': [1.0] * n,
        'coef': [1.0 if f == 'n_grasp' else 0.0 for f in names],
        'intercept': 0.0,
    }


PLAN_ONE_GRASP = {
    'tree': {'type': 'sequence', 'children': [
        {'type': 'move_to_keypoint',
         'params': {'keypoint_label': 'bowl', 'offset_z': 0}},
        {'type': 'grasp', 'params': {'force': 60}},
    ]}}

PLAN_TWO_GRASPS = {
    'tree': {'type': 'sequence', 'children': [
        {'type': 'move_to_keypoint',
         'params': {'keypoint_label': 'bowl', 'offset_z': 0}},
        {'type': 'grasp', 'params': {'force': 60}},
        {'type': 'release', 'params': {}},
        {'type': 'grasp', 'params': {'force': 60}},
    ]}}


def test_stub_model_ranks_by_grasp_count():
    m = TraceModel(_stub_artifact())
    ranked = rank_plans([PLAN_ONE_GRASP, PLAN_TWO_GRASPS], None, model=m)
    assert ranked[0][0] == 1  # two grasps -> higher score
    assert ranked[0][1] > ranked[1][1]
    assert all(0.0 <= p <= 1.0 for _, p in ranked)


def test_tie_break_keeps_planner_order():
    m = TraceModel(_stub_artifact())
    ranked = rank_plans([PLAN_ONE_GRASP, dict(PLAN_ONE_GRASP)], None, model=m)
    assert [i for i, _ in ranked] == [0, 1]
    assert ranked[0][1] == ranked[1][1]


def test_determinism():
    m = TraceModel(_stub_artifact())
    r1 = rank_plans([PLAN_TWO_GRASPS, PLAN_ONE_GRASP], None, model=m,
                    instruction='pick up the bowl')
    r2 = rank_plans([PLAN_TWO_GRASPS, PLAN_ONE_GRASP], None, model=m,
                    instruction='pick up the bowl')
    assert r1 == r2


def test_unparseable_candidate_ranks_last():
    m = TraceModel(_stub_artifact())
    ranked = rank_plans([':::{{{not yaml', PLAN_ONE_GRASP], None, model=m)
    assert ranked[0][0] == 1


def test_latency_under_10ms():
    m = TraceModel(_stub_artifact())
    cands = [PLAN_ONE_GRASP, PLAN_TWO_GRASPS] * 2  # K=4
    rank_plans(cands, None, model=m)  # warm-up
    t0 = time.perf_counter()
    rank_plans(cands, None, model=m, instruction='pick up the bowl')
    dt = time.perf_counter() - t0
    assert dt < 0.010, f'rank_plans took {dt * 1e3:.2f} ms'


def test_committed_artifact_loads_and_scores():
    m = TraceModel.load(DEFAULT_ARTIFACT)
    p = m.score_plan(PLAN_ONE_GRASP, prompts=['bowl'],
                     instruction='pick up the bowl')
    assert 0.0 <= p <= 1.0 and math.isfinite(p)


def test_artifact_kind_checked():
    bad = _stub_artifact()
    bad['kind'] = 'something_else'
    with pytest.raises(ValueError):
        TraceModel(bad)


def test_det_map_objects_and_dicts_both_work():
    class Det:
        def __init__(self, c, p):
            self.confidence = c
            self.position_3d = p
    m = TraceModel(_stub_artifact())
    det_obj = {'bowl': Det(0.9, [0, 0, 1])}
    det_dic = {'bowl': {'confidence': 0.9, 'position_3d': [0, 0, 1]}}
    p1 = m.score_plan(PLAN_ONE_GRASP, det_map=det_obj)
    p2 = m.score_plan(PLAN_ONE_GRASP, det_map=det_dic)
    assert p1 == p2

"""Unit tests for the selection='trace_model' harness hook.

Exercises the WMSafetyGate ranker plumbing directly with a stub model so
no LIBERO / Gemini / GPU machinery is needed (the fair.config module pulls
the whole benchmark stack, so the cfg-level wiring is exercised in the
integration environment, not here).
"""
import numpy as np
import pytest

from spark_bench.libero_pro.wm_safety_gate import WMSafetyGate

pytest.importorskip('spark_real.world_model.bt_to_actions',
                    reason='bt_to_actions needed to compile candidate BTs')


def _plan(label='bowl', offset_z=0.0, extra_dz=0.0):
    children = [
        {'type': 'move_to_keypoint',
         'params': {'keypoint_label': label, 'offset_z': offset_z}},
        {'type': 'grasp', 'params': {'force': 60}},
        {'type': 'move_relative', 'params': {'dx': 0, 'dy': 0,
                                             'dz': 0.2 + extra_dz}},
    ]
    return {'tree': {'type': 'sequence', 'children': children}}


KEYPOINTS = {'bowl': np.array([0.1, 0.1, 0.9], dtype=np.float32),
             'plate': np.array([0.3, 0.0, 0.9], dtype=np.float32)}
EE0 = np.array([0.0, 0.0, 1.0], dtype=np.float32)


def _gate(ranker=None):
    return WMSafetyGate(z_min=0.78, z_max=1.30, xy_radius=1.5, ranker=ranker)


def test_default_gate_unchanged_without_ranker():
    gate = _gate(ranker=None)
    res = gate.evaluate_candidates([_plan(), _plan(extra_dz=0.05)],
                                   ee_init_xyz=EE0, keypoint_xyz=KEYPOINTS)
    assert res.n_passed == 2
    # legacy behaviour: shortest action path wins
    assert res.best_idx == 0
    assert all(np.isnan(c.trace_prob) for c in res.candidates)


def test_ranker_overrides_default_ordering():
    # Stub model prefers candidate 1.
    def ranker(bts):
        assert len(bts) == 2
        return [(1, 0.9), (0, 0.2)]
    gate = _gate(ranker=ranker)
    res = gate.evaluate_candidates([_plan(), _plan(extra_dz=0.05)],
                                   ee_init_xyz=EE0, keypoint_xyz=KEYPOINTS)
    assert res.best_idx == 1
    assert res.candidates[1].trace_prob == pytest.approx(0.9)
    assert res.candidates[0].trace_prob == pytest.approx(0.2)


def test_ranker_only_sees_shield_passing_candidates():
    seen = []

    def ranker(bts):
        seen.append(len(bts))
        return [(0, 0.7)]

    # Candidate 1 dives below the floor -> shield rejects it.
    unsafe = _plan()
    unsafe['tree']['children'].append(
        {'type': 'move_relative', 'params': {'dx': 0, 'dy': 0, 'dz': -1.0}})
    gate = _gate(ranker=ranker)
    res = gate.evaluate_candidates([_plan(), unsafe],
                                   ee_init_xyz=EE0, keypoint_xyz=KEYPOINTS)
    assert res.n_passed == 1
    assert seen == [1]  # ranker saw only the safe candidate
    assert res.best_idx == 0
    assert not res.candidates[1].safe


def test_ranker_exception_falls_back_to_shield_ranking():
    def ranker(bts):
        raise RuntimeError('boom')
    gate = _gate(ranker=ranker)
    res = gate.evaluate_candidates([_plan(), _plan(extra_dz=0.05)],
                                   ee_init_xyz=EE0, keypoint_xyz=KEYPOINTS)
    # fail-open: default shortest-path ranking
    assert res.best_idx == 0


def test_tie_prob_keeps_planner_order():
    def ranker(bts):
        return [(0, 0.5), (1, 0.5)]
    gate = _gate(ranker=ranker)
    res = gate.evaluate_candidates([_plan(extra_dz=0.05), _plan()],
                                   ee_init_xyz=EE0, keypoint_xyz=KEYPOINTS)
    # equal probs -> candidate 0 (the temp-0 primary) wins, even though it
    # has the LONGER action path (would lose under legacy ranking).
    assert res.best_idx == 0


def test_real_trace_model_ranker_end_to_end():
    """The committed artifact drives the gate via select.rank_plans."""
    from spark_bench.trace_selector.select import TraceModel, rank_plans
    model = TraceModel.load()

    def ranker(bts):
        return rank_plans(bts, None, model=model,
                          instruction='pick up the bowl',
                          prompts=['bowl', 'plate'])

    gate = _gate(ranker=ranker)
    res = gate.evaluate_candidates([_plan(), _plan(label='plate')],
                                   ee_init_xyz=EE0, keypoint_xyz=KEYPOINTS)
    assert res.best_idx in (0, 1)
    probs = [c.trace_prob for c in res.candidates]
    assert all(0.0 <= p <= 1.0 for p in probs)

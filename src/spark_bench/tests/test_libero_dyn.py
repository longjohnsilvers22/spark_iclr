"""
Unit tests for the LIBERO-Dyn perturbation protocol
(spark_bench.libero_dyn): schedule determinism, injector firing logic
(mocked env), and latency-metric extraction from synthetic timelines.
"""
from types import SimpleNamespace

import numpy as np
import pytest

from spark_bench.libero_dyn import (
    DIRECTION_TABLE,
    PerturbationInjector,
    Phase,
    compute_latency_metrics,
    displacement_direction,
    displacement_vector,
    schedule_seed,
)


# Schedule determinism

def test_schedule_seed_deterministic_and_distinct():
    assert schedule_seed('taskA', 0) == schedule_seed('taskA', 0)
    assert schedule_seed('taskA', 0) != schedule_seed('taskA', 1)
    assert schedule_seed('taskA', 0) != schedule_seed('taskB', 0)


def test_direction_depends_only_on_task_and_trial():
    d1 = displacement_direction('libero_spatial_task_0', 3)
    d2 = displacement_direction('libero_spatial_task_0', 3)
    np.testing.assert_allclose(d1, d2)
    assert tuple(np.round(d1, 4)) in {tuple(np.round(v, 4))
                                        for v in np.array(DIRECTION_TABLE)}
    # Unit norm.
    assert np.linalg.norm(d1) == pytest.approx(1.0)


def test_magnitude_sweep_shares_direction():
    v2 = displacement_vector('t', 5, 2.0)
    v10 = displacement_vector('t', 5, 10.0)
    assert np.linalg.norm(v2) == pytest.approx(0.02)
    assert np.linalg.norm(v10) == pytest.approx(0.10)
    # Same direction, scaled.
    np.testing.assert_allclose(v10, v2 * 5.0)
    assert v2[2] == 0.0  # in-plane only


def test_zero_magnitude_is_zero_vector():
    assert np.linalg.norm(displacement_vector('t', 0, 0.0)) == 0.0


def test_directions_cover_table_across_trials():
    seen = {tuple(np.round(displacement_direction('taskX', t), 4))
            for t in range(64)}
    assert len(seen) > 1  # not degenerate


# Injector firing logic (mocked env)

class _MockEnv:
    def __init__(self):
        self.stepped = []

    def step(self, action):
        self.stepped.append(np.asarray(action).copy())
        return ({}, 0.0, False, {})


def test_post_grasp_plan_fires_on_first_close_command():
    env = _MockEnv()
    fired = []
    inj = PerturbationInjector(env, Phase.POST_GRASP_PLAN,
                                 displace_fn=lambda: fired.append(1) or 'body',
                                 meta={})
    inj.install()
    open_a = np.zeros(7); open_a[6] = -1.0
    close_a = np.zeros(7); close_a[6] = 1.0
    env.step(open_a)
    assert not inj.fired
    env.step(close_a)
    assert inj.fired and fired == [1]
    assert 't_perturb' in inj.meta
    assert inj.meta['perturb_phase'] == 'post_grasp_plan'
    assert inj.meta['perturbed_body'] == 'body'
    # One-shot: further closes do not re-fire.
    env.step(close_a)
    assert fired == [1]
    inj.uninstall()
    assert env.step == inj._orig_step or inj._orig_step is None


def test_mid_approach_fires_at_half_progress():
    env = _MockEnv()
    ee = {'pos': np.array([0.4, 0.0, 1.0])}
    target = np.array([0.0, 0.0, 0.9])
    inj = PerturbationInjector(env, Phase.MID_APPROACH,
                                 displace_fn=lambda: 'b',
                                 pick_target_pos=target,
                                 get_ee_pos=lambda: ee['pos'],
                                 meta={})
    inj.install()
    a = np.zeros(7)
    env.step(a)              # first step latches start distance
    assert not inj.fired
    ee['pos'] = np.array([0.3, 0.0, 0.98])   # ~25% progress
    env.step(a)
    assert not inj.fired
    ee['pos'] = np.array([0.15, 0.0, 0.93])  # >50% progress
    env.step(a)
    assert inj.fired
    inj.uninstall()


def test_mid_approach_without_target_never_fires():
    env = _MockEnv()
    inj = PerturbationInjector(env, Phase.MID_APPROACH,
                                 displace_fn=lambda: 'b', meta={})
    inj.install()
    for _ in range(5):
        env.step(np.zeros(7))
    assert not inj.fired
    inj.uninstall()


def test_displace_fn_exception_never_breaks_stepping():
    env = _MockEnv()

    def _boom():
        raise RuntimeError('injection failed')

    inj = PerturbationInjector(env, Phase.POST_GRASP_PLAN,
                                 displace_fn=_boom, meta={})
    inj.install()
    close_a = np.zeros(7); close_a[6] = 1.0
    env.step(close_a)  # must not raise
    assert inj.fired
    assert 'perturb_error' in inj.meta
    assert 't_perturb' in inj.meta  # timestamp still recorded


# Latency metrics from synthetic timelines

def test_metrics_no_perturbation():
    m = compute_latency_metrics({'scene_diffs': [
        {'target_status': 'moved', 't': 10.0}]})
    assert m['detection_latency_s'] is None
    assert m['recovery_used'] is False


def test_metrics_scene_diff_then_retarget():
    meta = {
        't_perturb': 100.0,
        'scene_diffs': [
            {'target_status': 'ok', 't': 99.0},       # pre-perturb: ignored
            {'target_status': 'moved', 't': 100.4},
        ],
        'retarget_events': [
            {'t_flag': 100.4, 't_resume': 100.9},
        ],
    }
    m = compute_latency_metrics(meta)
    assert m['detected_by'] == 'scene_diff'
    assert m['detection_latency_s'] == pytest.approx(0.4)
    assert m['adapted_by'] == 'retarget'
    assert m['adaptation_latency_s'] == pytest.approx(0.5)


def test_metrics_telemetry_fallback_and_recovery():
    meta = {
        't_perturb': 50.0,
        'grasp_outcomes': [
            {'outcome': 'secured', 't': 40.0},
            {'outcome': 'empty_close', 't': 51.2},
        ],
        'recovery_attributions': [
            {'layer': 'perception', 't': 52.0, 't_resume': 53.0},
        ],
    }
    m = compute_latency_metrics(meta)
    assert m['detected_by'] == 'telemetry'
    assert m['detection_latency_s'] == pytest.approx(1.2)
    assert m['adapted_by'] == 'recovery_perception'
    assert m['adaptation_latency_s'] == pytest.approx(53.0 - 51.2)
    assert m['recovery_used'] is True


def test_metrics_earliest_flag_wins():
    # Telemetry fired BEFORE the scene diff (object moved as the jaws
    # closed): the earlier flag must win.
    meta = {
        't_perturb': 100.0,
        'grasp_outcomes': [{'outcome': 'empty_close', 't': 101.0}],
        'scene_diffs': [{'target_status': 'moved', 't': 103.5}],
    }
    m = compute_latency_metrics(meta)
    assert m['detected_by'] == 'telemetry'
    assert m['detection_latency_s'] == pytest.approx(1.0)


def test_resolve_pick_target_from_score_binding():
    from spark_bench.libero_dyn import _resolve_pick_target_pos

    class _Det:
        def __init__(self, p):
            self.position_3d = np.asarray(p, float)

    det_map = {'dark bowl': _Det([0.1, 0.2, 0.9]),
               'plate': _Det([0.4, 0.0, 0.9])}
    score = {'tree': {'type': 'sequence', 'children': [
        {'type': 'move_to_keypoint',
         'params': {'keypoint_label': 'dark bowl'}},
        {'type': 'grasp', 'params': {}},
    ]}}
    # BDDL hint shares NO literal substring with the det_map keys - the
    # score binding must resolve it anyway.
    p = _resolve_pick_target_pos(det_map, 'akita_black_bowl_1', score)
    np.testing.assert_allclose(p, [0.1, 0.2, 0.9])
    # Without a score, normalized token overlap still finds partial hits.
    p2 = _resolve_pick_target_pos({'black bowl': _Det([0.3, 0.3, 0.9])},
                                    'akita_black_bowl_1', None)
    np.testing.assert_allclose(p2, [0.3, 0.3, 0.9])
    p3 = _resolve_pick_target_pos({'bowl': _Det([0.3, 0.3, 0.9])},
                                    'the bowl on the stove', None)
    np.testing.assert_allclose(p3, [0.3, 0.3, 0.9])


def test_metrics_detected_but_never_adapted():
    meta = {'t_perturb': 10.0,
            'scene_diffs': [{'target_status': 'missing', 't': 10.3}]}
    m = compute_latency_metrics(meta)
    assert m['detection_latency_s'] == pytest.approx(0.3)
    assert m['adaptation_latency_s'] is None


def test_astar_grid_routes_around_obstacle():
    from spark_bench.libero_dyn import astar_grid
    occ = np.zeros((11, 11), dtype=bool)
    occ[2:9, 5] = True            # a wall with gaps at the top and bottom
    path = astar_grid(occ, (5, 0), (5, 10))
    assert path is not None
    assert path[0] == (5, 0) and path[-1] == (5, 10)
    assert all(not occ[r, c] for r, c in path)
    assert all(max(abs(a[0] - b[0]), abs(a[1] - b[1])) == 1
               for a, b in zip(path, path[1:]))
    occ[:, 5] = True              # seal the wall: no path
    assert astar_grid(occ, (5, 0), (5, 10)) is None

"""Unit tests for the planner-vs-human measuring apparatus.

Synthetic and offline: no corpus, no Gemini, no robot. The one corpus-backed
test skips itself when ``$SPARK_HUMAN_EPISODES`` is unset, so CI stays green on
a machine that has never seen the teleop rig.

What is worth testing here is not the arithmetic but the REFUSALS: a fit that
silently extrapolates would print a confident centimetre figure that means
nothing, which is the exact failure mode this apparatus exists to avoid.
"""

from __future__ import annotations

import os
from pathlib import Path

import numpy as np
import pytest

from spark_real.tests.human_corpus import TASK_PROMPTS, corpus_root
from spark_real.tests.human_table_fit import (
    HumanSpread,
    MIN_FIT_PAIRS,
    TableFit,
    TaskFit,
    apply_affine,
    fit_affine,
    human_spread,
    loo_residual,
    plan_targets,
    resolve_centroid,
    sign_test,
)

# A plausible image->table map: ~1 mm per pixel, y flipped, offset into the
# robot's reach. Any invertible affine would do.
TRUE_SOL = np.array([[0.0012, 0.0001], [-0.0002, -0.0014], [-1.05, 0.45]])


def _uv_grid(n: int = 12, seed: int = 0) -> np.ndarray:
    rng = np.random.default_rng(seed)
    return rng.uniform([80, 60], [560, 420], size=(n, 2))


def _rows(uv: np.ndarray, roles=None, noise: float = 0.0, seed: int = 1):
    rng = np.random.default_rng(seed)
    xy = np.array([apply_affine(TRUE_SOL, p) for p in uv])
    if noise:
        xy = xy + rng.normal(0.0, noise, xy.shape)
    roles = roles or ["obj" if i % 2 == 0 else "container" for i in range(len(uv))]
    return [(f"episode_{i:04d}", roles[i], tuple(uv[i]), xy[i]) for i in range(len(uv))]


# ---------------------------------------------------------------------------
# the affine itself
# ---------------------------------------------------------------------------


def test_affine_recovers_a_known_map_exactly():
    uv = _uv_grid()
    xy = np.array([apply_affine(TRUE_SOL, p) for p in uv])
    sol = fit_affine(uv, xy)
    assert np.allclose(sol, TRUE_SOL, atol=1e-9)
    assert np.allclose(apply_affine(sol, uv[3]), xy[3], atol=1e-12)


def test_loo_residual_is_zero_on_noiseless_data_and_tracks_noise():
    uv = _uv_grid()
    xy = np.array([apply_affine(TRUE_SOL, p) for p in uv])
    assert float(np.max(loo_residual(uv, xy))) < 1e-9
    noisy = xy + np.random.default_rng(7).normal(0.0, 0.01, xy.shape)
    assert 0.005 < float(np.median(loo_residual(uv, noisy))) < 0.05


# ---------------------------------------------------------------------------
# the refusals -- the part that keeps a meaningless number off the table
# ---------------------------------------------------------------------------


def test_too_few_correspondences_refuses_rather_than_fitting():
    fit = TaskFit(_rows(_uv_grid(MIN_FIT_PAIRS - 1)))
    assert fit.degenerate
    xy, why = fit.solve((300.0, 240.0))
    assert xy is None and "need" in why


def test_clustered_correspondences_are_refused_as_degenerate():
    """Containers barely move between episodes; a fit anchored on them alone is
    a blob that extrapolates by a metre when asked where a knife is."""
    rng = np.random.default_rng(3)
    uv = np.array([320.0, 240.0]) + rng.normal(0.0, 1.0, (10, 2))
    fit = TaskFit(_rows(uv))
    assert fit.degenerate
    xy, why = fit.solve((320.0, 240.0))
    assert xy is None and "degenerate" in why


def test_query_far_outside_the_fitted_region_is_refused():
    fit = TaskFit(_rows(_uv_grid()))
    assert not fit.degenerate
    inside, why = fit.solve(fit.uv.mean(0))
    assert inside is not None and why == "ok"
    outside, why = fit.solve((6000.0, 6000.0))
    assert outside is None and "extrapolated" in why


def test_floor_is_reported_per_role_and_is_nan_when_the_role_is_absent():
    fit = TaskFit(_rows(_uv_grid(), roles=["obj"] * 12, noise=0.01))
    assert fit.floor("obj") > 0.0
    assert fit.floor("container") != fit.floor("container")  # NaN


def test_table_fit_holds_out_the_queried_episode():
    rows = _rows(_uv_grid())
    table = TableFit({"t": rows})
    fit = table.for_episode("t", "episode_0000")
    assert fit.n == len(rows) - 1
    assert table.for_episode("t", "episode_0000") is fit  # cached


# ---------------------------------------------------------------------------
# plan parsing
# ---------------------------------------------------------------------------


def test_plan_targets_reads_a_place_in_slot_plan():
    score = {
        "tree": {
            "type": "sequence",
            "children": [
                {"type": "move_to_keypoint", "params": {"keypoint_label": "knife"}},
                {"type": "grasp", "params": {"grasp_strategy": "obb"}},
                {"type": "move_relative", "params": {"dz": 0.2}},
                {"type": "place_in_slot", "params": {"container_label": "tray"}},
            ],
        }
    }
    grasp, place, types = plan_targets(score)
    assert (grasp, place) == ("knife", "tray")
    assert types == ["move_to_keypoint", "grasp", "move_relative", "place_in_slot"]


def test_plan_targets_reads_a_second_keypoint_as_the_place_target():
    score = {
        "tree": {
            "type": "sequence",
            "children": [
                {"type": "move_to_keypoint", "params": {"keypoint_label": "pen"}},
                {"type": "grasp", "params": {}},
                {"type": "move_to_keypoint", "params": {"keypoint_label": "bin"}},
                {"type": "release", "params": {}},
            ],
        }
    }
    assert plan_targets(score)[:2] == ("pen", "bin")


def test_plan_targets_reads_stack_and_survives_junk():
    stack = {
        "tree": {
            "type": "sequence",
            "children": [
                {"type": "move_to_keypoint", "params": {"keypoint_label": "blue block"}},
                {"type": "grasp", "params": {}},
                {"type": "stack", "params": {"target_label": "gray block"}},
            ],
        }
    }
    assert plan_targets(stack)[:2] == ("blue block", "gray block")
    for junk in (None, {}, {"tree": None}, {"tree": {"type": "sequence"}}, "not a dict"):
        assert plan_targets(junk)[:2] == (None, None)


def test_resolve_centroid_tolerates_the_corpus_naming_slack():
    table = {"knife": (100.0, 200.0), "tray": (300.0, 220.0)}
    task = "put the knife in the tray"
    assert resolve_centroid("knife handle", table, task) == (100.0, 200.0)
    assert resolve_centroid("the tray", table, task) == (300.0, 220.0)
    assert resolve_centroid("plushie", table, task) is None
    assert resolve_centroid(None, table, task) is None


# ---------------------------------------------------------------------------
# the human-variance baseline
# ---------------------------------------------------------------------------


def _spread(points: np.ndarray) -> HumanSpread:
    return HumanSpread(
        task="t",
        n=len(points),
        grasp_xyz=points,
        release_xyz=points,
        episodes=[f"episode_{i:04d}" for i in range(len(points))],
    )


def test_rms_radial_matches_a_hand_computed_case():
    pts = np.array([[1.0, 0.0, 0.0], [-1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, -1.0, 0.0]])
    assert _spread(pts).rms_radial("grasp") == pytest.approx(1.0)


def test_loo_baseline_holds_the_episode_out():
    pts = np.array([[0.0, 0.0, 0.0], [0.0, 0.0, 0.0], [0.0, 0.0, 0.0], [0.3, 0.0, 0.0]])
    sp = _spread(pts)
    # the outlier is measured against the mean of the other three, which is 0
    assert sp.loo_error("grasp", "episode_0003") == pytest.approx(0.3)
    # an inlier is measured against a mean pulled 0.1 m towards the outlier
    assert sp.loo_error("grasp", "episode_0000") == pytest.approx(0.1)
    assert sp.loo_error("grasp", "no_such_episode") is None
    assert len(sp.loo_errors("grasp")) == 4


def test_z_std_is_tight_when_z_is_and_ptp_spans_the_range():
    pts = np.array([[0.0, 0.0, -0.28], [0.5, 0.0, -0.281], [0.0, 0.9, -0.279]])
    sp = _spread(pts)
    assert sp.z_std("grasp") < 0.001
    assert sp.ptp("grasp")[1] == pytest.approx(0.9)


def test_sign_test_calls_a_clean_win_and_a_tie():
    a = [1.0] * 10
    b = [2.0] * 10
    wins, n, p = sign_test(a, b)
    assert (wins, n) == (10, 10)
    assert p == pytest.approx(2.0 / 1024)
    wins, n, p = sign_test([1.0, 2.0], [2.0, 1.0])
    assert (wins, n) == (1, 2) and p == pytest.approx(1.0)
    _, n, p = sign_test([1.0, 1.0], [1.0, 1.0])
    assert n == 0 and p != p  # NaN: nothing to compare


# ---------------------------------------------------------------------------
# corpus-backed, skipped when the rig is not present
# ---------------------------------------------------------------------------


@pytest.mark.skipif(
    corpus_root() is None or not Path(str(corpus_root())).is_dir(),
    reason=f"no human corpus (set $SPARK_HUMAN_EPISODES); saw {os.environ.get('SPARK_HUMAN_EPISODES')!r}",
)
def test_real_corpus_grasp_spread_dwarfs_any_plausible_planner_error():
    """The calibration this whole comparison rests on.

    Object XY is re-randomised every episode, so a constant predictor is tens of
    centimetres out. If this ever collapses, the tasks stopped being randomised
    and every "the planner is N cm from the human" claim needs re-reading.
    """
    root = Path(str(corpus_root()))
    task = "put the knife in the tray"
    sp = human_spread(root, task)
    assert task in TASK_PROMPTS
    assert sp.n > 20
    assert sp.rms_radial("grasp") > 0.15, "grasp XY should be randomised across the table"
    assert float(np.median(sp.loo_errors("grasp"))) > 0.10
    # Z is the one transferable prior: the table does not move.
    assert sp.z_std("grasp") < 0.02

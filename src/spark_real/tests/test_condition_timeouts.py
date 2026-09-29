"""A primitive budget has to be satisfiable, or it is a scheduled abort.

Rig, 2026-08-18: ``verify_placed`` is not in PAPER_TIMEOUTS_S, so it inherited
DEFAULT_TIMEOUT_S = 8.0s while a full re-detect cost 30-62s. The watchdog fired
at 10:49:15 -- 14 seconds before the operator's stop -- on every place, no
matter how fast detection got. These tests pin the reconciliation from both
ends: the scoped cost must fit the budget, and the budget must still catch a
genuine hang.
"""

import pathlib

import yaml

from spark_real.control import primitive_timeouts as pt

CONFIG = pathlib.Path(__file__).resolve().parents[1] / "configs" / "ur10e_default.yaml"


def _ur10e_timeouts():
    raw = yaml.safe_load(CONFIG.read_text())
    return pt.load_timeouts(profile=type("Profile", (), {"raw": raw})())


# Measured on the rig: 30s for 4 SAM3 passes on sideview, 10:49:07-10:49:37.
MEASURED_PASS_S = 7.5
# What the scoped leaf asks for: 1 prompt (container plan-anchored) x 2
# contributing cameras, plus a capture when the reuse window has expired.
SCOPED_PASSES = 2
SCOPED_CAPTURE_S = 1.0


def test_the_condition_leaves_are_no_longer_on_the_generic_default():
    timeouts = _ur10e_timeouts()
    assert pt.timeout_for(timeouts, "verify_placed") != pt.DEFAULT_TIMEOUT_S
    assert pt.timeout_for(timeouts, "verify_grasp") != pt.DEFAULT_TIMEOUT_S


def test_every_family_gets_a_condition_budget_not_just_the_ur10e():
    """A family with no timeouts: block must not inherit the 8s scheduled abort."""
    defaults = pt.default_timeouts()
    assert defaults["verify_placed"] == pt.NON_PAPER_TIMEOUTS_S["verify_placed"]
    assert pt.timeout_for(defaults, "verify_placed") > pt.DEFAULT_TIMEOUT_S


def test_the_scoped_cost_fits_the_budget_with_margin():
    budget = pt.timeout_for(_ur10e_timeouts(), "verify_placed")
    worst_case = SCOPED_PASSES * MEASURED_PASS_S + SCOPED_CAPTURE_S
    assert worst_case < budget, "the budget is unsatisfiable by construction again"
    assert budget >= worst_case * 1.5, "no margin for a slow frame"


def test_the_budget_still_catches_the_unscoped_run_that_hung():
    """54s of verification must still be aborted; the fix is scope, not slack."""
    budget = pt.timeout_for(_ur10e_timeouts(), "verify_placed")
    assert budget < 54.0


def test_the_paper_table_is_left_alone():
    """NON_PAPER_TIMEOUTS_S exists so PAPER_TIMEOUTS_S stays the paper's table."""
    assert "verify_placed" not in pt.PAPER_TIMEOUTS_S
    assert "verify_grasp" not in pt.PAPER_TIMEOUTS_S
    assert set(pt.PAPER_TIMEOUTS_S) & set(pt.NON_PAPER_TIMEOUTS_S) == set()


def test_a_proprioceptive_check_gets_a_proprioceptive_budget():
    budget = pt.timeout_for(_ur10e_timeouts(), "verify_grasp")
    assert 1.0 <= budget <= 10.0, "a gripper register read, not a perception pass"


def test_the_yaml_can_still_override_the_module_default():
    timeouts = pt.load_timeouts(
        profile=type("Profile", (), {"raw": {"timeouts": {"verify_placed": 3.5}}})()
    )
    assert pt.timeout_for(timeouts, "verify_placed") == 3.5

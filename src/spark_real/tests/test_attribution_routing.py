"""
Port step 4: attribution consult at the recovery seams.

verification.attribution default-off preserves the legacy ladder;
enabled, PERCEPTION skips the tier-1 in-place perturb (re-ground
directly), EXECUTION keeps the perturb, PLAN stops recovery and surfaces
to the operator (never an autonomous replan).
"""

import numpy as np

from spark_real.control import execution_recovery as er
from spark_real.control.executor_types import ExecutionResult


class _Verdict:
    def __init__(self, outcome):
        self.outcome = outcome


class _FakeExecutor:
    MAX_GRASP_RETRIES = 1

    def __init__(self, det_map=None, outcome=None, attribution=True):
        self.detection_map = det_map or {}
        self._last_grasp_verdict = (
            _Verdict(outcome) if outcome is not None else None
        )
        self._pipeline = type(
            "P",
            (),
            {
                "profile": type(
                    "Pr",
                    (),
                    {"raw": {"verification": {"attribution": attribution}}},
                )()
            },
        )()
        self._holding = False
        self._abort = False

    def _verify_settings(self):
        return True, True, 2


def _fail(msg="grasp failed"):
    return ExecutionResult(action_type="grasp", success=False, message=msg)


def _actions():
    return [
        {"type": "move_to_keypoint", "params": {"keypoint_label": "block"}},
        {"type": "grasp", "params": {}},
    ]


def test_perception_layer_skips_tier1(monkeypatch):
    calls = {}

    def fake_recover(executor, params, actions, idx, skip_tier1=False):
        calls["skip_tier1"] = skip_tier1
        return None

    monkeypatch.setattr(er, "recover_grasp", fake_recover)
    # detection missing -> PERCEPTION
    ex = _FakeExecutor(det_map={}, outcome="empty_close")
    er.attempt_recovery(ex, "grasp", {}, _fail(), _actions(), 1)
    assert calls["skip_tier1"] is True


def test_execution_layer_keeps_tier1(monkeypatch):
    calls = {}

    def fake_recover(executor, params, actions, idx, skip_tier1=False):
        calls["skip_tier1"] = skip_tier1
        return None

    monkeypatch.setattr(er, "recover_grasp", fake_recover)
    # detection present + confident -> EXECUTION
    ex = _FakeExecutor(
        det_map={
            "block": {"position_3d": [0.1, 0.2, -0.2], "confidence": 0.9}
        },
        outcome="empty_close",
    )
    er.attempt_recovery(ex, "grasp", {}, _fail(), _actions(), 1)
    assert calls["skip_tier1"] is False


def test_plan_layer_stops_recovery(monkeypatch):
    def fake_recover(*a, **kw):
        raise AssertionError("PLAN must not reach a recovery tier")

    monkeypatch.setattr(er, "recover_grasp", fake_recover)
    ex = _FakeExecutor(
        det_map={
            "block": {"position_3d": [0.1, 0.2, -0.2], "confidence": 0.9}
        }
    )
    # retries exhausted -> PLAN: the per-label grasp retry counter has
    # already spent the local budget.
    ex._grasp_retry_count = {"block": 2}
    got = er.attempt_recovery(ex, "grasp", {}, _fail(), _actions(), 1)
    assert got is None


def test_flag_off_keeps_legacy_dispatch(monkeypatch):
    calls = {}

    def fake_recover(executor, params, actions, idx, skip_tier1=False):
        calls["skip_tier1"] = skip_tier1
        return None

    monkeypatch.setattr(er, "recover_grasp", fake_recover)
    ex = _FakeExecutor(det_map={}, attribution=False)
    er.attempt_recovery(ex, "grasp", {}, _fail(), _actions(), 1)
    assert calls["skip_tier1"] is False  # legacy: tier-1 always attempted


def test_build_verify_result_reads_typed_outcome():
    ex = _FakeExecutor(
        det_map={"block": {"position_3d": [0, 0, 0], "confidence": 0.4}},
        outcome="slip",
    )
    out = er.build_verify_result(ex, "block")
    assert out["grasp_outcome"] == "slip"
    assert out["detection_missing"] is False
    assert abs(out["detection_confidence"] - 0.4) < 1e-9

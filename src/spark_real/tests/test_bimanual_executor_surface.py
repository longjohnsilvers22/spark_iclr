"""
BimanualScoreExecutor exposes the surface ExecutionMixin and the stop
routes call on ScoreExecutor: update_detections, abort,
note_abort_requested, and a verify_outcome that task_success can read.

Usage:
    PYTHONPATH=src python -m pytest src/spark_real/tests/test_bimanual_executor_surface.py -q
    PYTHONPATH=src python src/spark_real/tests/test_bimanual_executor_surface.py
"""

from __future__ import annotations

import sys
from pathlib import Path
from types import SimpleNamespace

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
sys.path.insert(0, str(Path(__file__).resolve().parent))

from test_selector_semantics import _FakePipeline, _MockRobot  # noqa: E402

from spark_real.control.bimanual_executor import BimanualScoreExecutor  # noqa: E402
from spark_real.control.success_verifier import (  # noqa: E402
    reset_run_state,
    task_success,
)


class _MockSafe:
    """Two mock arms behind the BimanualSafeRobot attribute surface."""

    def __init__(self):
        self.left = _MockRobot()
        self.right = _MockRobot()
        self.driver = None

    def is_motion_safe(self, arm, xyz):
        return True

    def velocity_scale_for(self, arm, xyz):
        return 1.0


def _make():
    pipeline = _FakePipeline()
    # Kill switch: verification runs its config path and records a verdict
    # without a capture.
    pipeline.profile = SimpleNamespace(raw={"verification": {"enabled": False}})
    return BimanualScoreExecutor(_MockSafe(), pipeline=pipeline)


def test_update_detections_reaches_both_arms():
    ex = _make()
    dets = {"cup": {"position_3d": [0.4, 0.0, 0.05]}}
    ex.update_detections(dets)
    assert ex.detection_map is dets
    assert ex._arm_executors["left"].detection_map is dets
    assert ex._arm_executors["right"].detection_map is dets


def test_verify_outcome_set_after_trivial_score():
    ex = _make()
    assert ex.verify_outcome is None
    score = {
        "task": "wait",
        "tree": {
            "type": "sequence",
            "children": [
                {"type": "wait", "params": {"arm": "left", "duration": 0.0}},
            ],
        },
    }
    results = ex.execute_score(score)
    assert results[0].action_type == "wait[left]" and results[0].success
    assert ex.verify_outcome is not None
    assert ex.verify_outcome.status == "unverified"
    assert task_success(ex.verify_outcome) is False
    reset_run_state(ex)
    assert ex.verify_outcome is None


def test_note_abort_requested_and_abort():
    ex = _make()
    ex.note_abort_requested()
    assert ex._abort is True
    for arm in ex._arm_executors.values():
        assert arm._abort is True and arm._abort_epoch == 1

    ex = _make()
    braked = ex.abort()
    assert isinstance(braked, bool)
    assert ex._abort is True
    for arm in ex._arm_executors.values():
        assert arm._abort is True
        assert "stop" in arm.robot._log
    # An aborted wrapper runs nothing.
    score = {"tree": {"type": "wait", "params": {"arm": "left", "duration": 0.0}}}
    ex._abort = True
    ex._running = True
    assert ex._run_node(score["tree"]) is False


def main():
    for name, fn in list(globals().items()):
        if name.startswith("test_") and callable(fn):
            fn()
            print("PASS", name)
    return 0


if __name__ == "__main__":
    sys.exit(main())

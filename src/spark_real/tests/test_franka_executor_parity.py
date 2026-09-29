"""
Franka / bimanual parity for executor code paths the UR10e tests never reach.

Two doubles, built like test_selector_semantics._MockRobot: a franka_hand
FR3 (robot_family "franka") and a dynamixel bimanual arm (robot_family
"bimanual_franka"). Every test runs against the real ScoreExecutor methods;
only motion and sleeps are stubbed. Behaviours the audit found missing are
strict xfails carrying the audit ID, so closing them flips the test.

Usage:
    PYTHONPATH=src python -m pytest src/spark_real/tests/test_franka_executor_parity.py -q
"""

from __future__ import annotations

import math
import sys
import time
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
sys.path.insert(0, str(Path(__file__).resolve().parent))

from test_bimanual_executor_surface import _MockSafe  # noqa: E402
from test_selector_semantics import _FakePipeline, _MockRobot  # noqa: E402

import spark_real.control.executor_ik as executor_ik  # noqa: E402
from spark_real.control.bimanual_executor import BimanualScoreExecutor  # noqa: E402
from spark_real.control.executor_ik import resolve_joint_limits  # noqa: E402
from spark_real.control.grasp_outcome import GraspOutcome  # noqa: E402
from spark_real.control.score_executor import ScoreExecutor  # noqa: E402
from spark_real.control.success_verifier import (  # noqa: E402
    begin_release_witness,
    finish_release_witness,
)


class _FrankaHand(_MockRobot):
    """FR3 with Franka Hand: 0..255 jaw position, is_grasped flag."""

    # Closed on an object: inside the executor's holding band
    # [GRIPPER_EMPTY_THRESHOLD, GRIPPER_FULLY_CLOSED) = [235, 250).
    closed_pos = 240

    def __init__(self):
        super().__init__()
        self.pos_reads = 0
        self.joint_moves = []

    def get_gripper_position(self):
        self.pos_reads += 1
        return 0 if self._open else self.closed_pos

    def move_to_joint_config(self, q, velocity=None, **kwargs):
        self.joint_moves.append((list(q), velocity))


class _Dynamixel(_FrankaHand):
    """Bimanual per-arm view: SSG-48 on a Dynamixel, same 0..255 scale."""

    robot_family = "bimanual_franka"
    GRIPPER_TYPE = "dynamixel"


DOUBLES = [_FrankaHand, _Dynamixel]


def _executor(robot, velocity=0.25, pipeline=None):
    ex = ScoreExecutor(robot, detection_map={}, velocity=velocity, pipeline=pipeline)
    ex._servo = type(
        "S",
        (),
        {
            "abort": lambda self: None,
            "max_vel_linear": 0.1,
            "move_to_pose": lambda self, *a, **kw: True,
            "_get_tcp_pose": lambda self: np.concatenate([robot._tcp, [np.pi, 0, 0]]),
        },
    )()
    return ex


# Velocity clamp (executor_core: 0.10 m/s for Franka PD stability).


def test_franka_velocity_is_clamped():
    assert _executor(_FrankaHand(), velocity=0.25).velocity == pytest.approx(0.10)
    assert _executor(_FrankaHand(), velocity=0.05).velocity == pytest.approx(0.05)


@pytest.mark.xfail(
    strict=True,
    reason="D06: executor_core clamp keys on robot_family == 'franka'; the "
    "bimanual per-arm drivers are not clamped",
)
def test_bimanual_arm_velocity_is_clamped():
    assert _executor(_Dynamixel(), velocity=0.25).velocity == pytest.approx(0.10)


def test_movej_via_ik_runs_at_the_clamped_velocity(monkeypatch):
    robot = _FrankaHand()
    ex = _executor(robot, velocity=0.25)
    q0 = np.asarray(robot.get_joint_positions(), dtype=float)
    q1 = q0.copy()
    q1[0] += 0.5

    monkeypatch.setattr(executor_ik, "solve_ik_pyroki", lambda pos, orient, q_seed: q1)
    monkeypatch.delenv("SPARK_LEGACY_JOINT_LIMITS", raising=False)
    ex._movej_via_ik([0.4, 0.0, 0.3], [math.pi, 0.0, 0.0], ex.velocity)

    assert len(robot.joint_moves) == 1
    q_cmd, vel_j = robot.joint_moves[0]
    assert np.allclose(q_cmd, q1)
    dq = np.abs(q1 - q0)
    assert vel_j == pytest.approx(resolve_joint_limits(0.10, dq=dq)[0])
    assert vel_j != pytest.approx(resolve_joint_limits(0.25, dq=dq)[0])


# _ensure_jaws_open settle.


@pytest.mark.xfail(
    strict=True,
    reason="D05/PP-04: _ensure_jaws_open still waits the Robotiq full-stroke "
    "settle on non-Robotiq grippers",
)
def test_ensure_jaws_open_does_not_use_the_robotiq_settle_on_franka():
    robot = _FrankaHand()
    robot._open = False
    ex = _executor(robot)
    sleeps = []
    ex._abort_sleep = lambda d, tick=0.05: sleeps.append(d)
    assert ex._ensure_jaws_open("test") is True
    assert "open_gripper" in robot._log
    robotiq_full_stroke = ex._robotiq_travel_settle_s(ex.ROBOTIQ_OPEN_SPEED_NORM_ASSUMED)
    assert robotiq_full_stroke == pytest.approx(0.517, abs=1e-3)
    assert all(abs(s - robotiq_full_stroke) > 1e-6 for s in sleeps), sleeps


# Release witness (G3) reads the non-Robotiq jaw position.


@pytest.mark.parametrize("double", DOUBLES)
def test_release_witness_reads_gripper_position(double):
    robot = double()
    robot._open = False  # holding: jaws on the object, is_object_detected True
    ex = _executor(robot)
    ex.RELEASE_CONFIRM_TIMEOUT_S = 0.1
    ex._holding = True

    state = begin_release_witness(ex)
    assert state["jaw_pos_before"] == pytest.approx(robot.closed_pos)
    reads_before = robot.pos_reads
    assert reads_before >= 1

    robot.open_gripper()
    ex._holding = False
    w = finish_release_witness(ex, state, "bowl")
    assert robot.pos_reads > reads_before
    assert w.released is True
    assert w.jaw_pos_after == pytest.approx(0.0)
    assert w.held_after is False


@pytest.mark.parametrize("double", DOUBLES)
def test_release_witness_sees_stuck_jaws(double):
    robot = double()
    robot._open = False
    robot.open_gripper = lambda: None  # jaws never move
    ex = _executor(robot)
    ex.RELEASE_CONFIRM_TIMEOUT_S = 0.1
    ex._holding = True
    state = begin_release_witness(ex)
    w = finish_release_witness(ex, state, "bowl")
    assert w.released is False


# Legacy _grasp path (non-Robotiq) records a GraspVerdict (G1).


@pytest.mark.parametrize("double", DOUBLES)
def test_legacy_grasp_records_verdict(double):
    robot = double()
    ex = _executor(robot)
    ex._move_to = lambda *a, **k: None
    ex._abort_sleep = lambda *a, **k: None
    ex._last_keypoint_label = ""

    assert getattr(ex, "_last_grasp_verdict", None) is None
    result = ex._grasp({"force": 40}, time.time())
    assert result.success is True, result.message
    assert "set_gripper_position(1.0)" in robot._log
    verdict = ex._last_grasp_verdict
    assert verdict.held is True
    assert verdict.source == "legacy_verify"
    assert verdict.gripper_pos == pytest.approx(robot.closed_pos)
    assert verdict.outcome == GraspOutcome.SECURED.value


# Per-arm bimanual bounds (group B) and the wrapper surface (group A).


class _BimanualPipeline(_FakePipeline):
    def __init__(self):
        super().__init__()
        self.config.robot_family = "bimanual_franka"
        self.profile = None


def test_per_arm_bounds_differ_from_ur_defaults():
    ex = BimanualScoreExecutor(_MockSafe(), pipeline=_BimanualPipeline())
    # Mirrors pipeline_init_bimanual: name the arm, then re-apply.
    for arm, arm_ex in ex._arm_executors.items():
        assert np.allclose(arm_ex.WORKSPACE_MIN, ScoreExecutor.WORKSPACE_MIN)
        arm_ex.arm = arm
        arm_ex._profile = None
        arm_ex._apply_family_workspace_bounds()
    left, right = ex._arm_executors["left"], ex._arm_executors["right"]
    for arm_ex in (left, right):
        assert not np.allclose(arm_ex.WORKSPACE_MIN, ScoreExecutor.WORKSPACE_MIN)
        assert not np.allclose(arm_ex.WORKSPACE_MAX, ScoreExecutor.WORKSPACE_MAX)
        assert arm_ex.GRASP_ORIENTATION == pytest.approx([math.pi, 0.0, 0.0], abs=1e-4)
        assert arm_ex.TABLE_Z_FLOOR == pytest.approx(0.0)
        assert arm_ex.TABLE_Z_FLOOR != ScoreExecutor.TABLE_Z_FLOOR
    # The two arms read their own workspace_<arm> boxes (mirrored in y).
    assert left.WORKSPACE_MIN[1] != right.WORKSPACE_MIN[1]
    assert left.WORKSPACE_MAX[1] != right.WORKSPACE_MAX[1]


def test_bimanual_executor_surface_and_verify_outcome():
    from types import SimpleNamespace

    for name in ("update_detections", "abort", "note_abort_requested"):
        assert callable(getattr(BimanualScoreExecutor, name)), name
    assert "verify_outcome" in vars(BimanualScoreExecutor)

    pipeline = _FakePipeline()
    pipeline.profile = SimpleNamespace(raw={"verification": {"enabled": False}})
    ex = BimanualScoreExecutor(_MockSafe(), pipeline=pipeline)
    assert ex.verify_outcome is None
    ex.execute_score(
        {"tree": {"type": "wait", "params": {"arm": "right", "duration": 0.0}}}
    )
    assert ex.verify_outcome is not None


def main():
    sys.exit(pytest.main([__file__, "-q"]))


if __name__ == "__main__":
    main()

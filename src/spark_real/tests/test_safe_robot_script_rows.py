"""
Regression tests for SafeRobot._send_script row scanning.

Two distinct failures are pinned here, both of which shipped live:

1. The scan used regex .search(), so only the FIRST movej row of a blended
   program was validated. With blending on by default, rows 1..N-1 of every
   motion bypassed the workspace/singularity filter.

2. The multi-row rewrite then routed joint-space rows to a predicate that
   looked up a non-existent `_fk_position` attribute and returned True when it
   wasn't callable -- which was always. That made `movej([q1..q6])` scripts
   completely unvalidated, dropping the elbow-singularity check the previous
   handler enforced. Strictly worse than the bug being fixed.

The joints cases below are the ones that matter: they must refuse a
near-singular elbow, and they must do so on a LATER row, not just row 0.
"""

import numpy as np
import pytest

from spark_real.control.safe_robot import BarrierSet, SafeRobot, SafetyConfig


class _FakeDriver:
    """Records what actually reaches the wire."""

    SUPPORTS_URSCRIPT = True
    robot_family = "ur10e"

    def __init__(self):
        self.sent = []

    def _send_script(self, script: str) -> bool:
        self.sent.append(script)
        return True


@pytest.fixture
def rig():
    cfg = SafetyConfig()
    sr = SafeRobot.__new__(SafeRobot)
    sr._cfg = cfg
    sr._barriers = BarrierSet(cfg)
    sr._robot = _FakeDriver()
    return sr


# A TCP inside the default workspace box (x in [-1.1,-0.5], y in [-0.5,0.7],
# z in [-0.25,0.50]) and one far outside it.
_POSE_OK = "-0.80,0.10,0.10,2.30,2.08,0.0"
_POSE_BAD = "5.00,0.10,0.10,2.30,2.08,0.0"

# Elbow (q2) values either side of the singularity barrier
# h = q2**2 * (q2 - pi)**2 - eps**2, eps = 0.20.
_Q_OK = "3.2070,-1.8788,-1.7903,5.2496,1.5762,0.0916"   # h ~ 77.9  -> safe
_Q_SINGULAR = "3.2070,-1.8788,0.0500,5.2496,1.5762,0.0916"  # h ~ -0.016 -> unsafe


def _movej_joints(q, r=0.05):
    return f"movej([{q}], a=1.4, v=1.05, r={r})\n"


def _movej_pose(p, r=0.05):
    return f"movej(p[{p}], a=1.4, v=1.05, r={r})\n"


def test_singular_elbow_is_refused(rig):
    """The whole point: a near-singular joint target must not reach the wire."""
    assert rig._send_script(_movej_joints(_Q_SINGULAR)) is False
    assert rig._robot.sent == []


def test_safe_joint_row_is_forwarded(rig):
    assert rig._send_script(_movej_joints(_Q_OK)) is True
    assert len(rig._robot.sent) == 1


def test_singular_elbow_refused_on_a_later_blended_row(rig):
    """
    Regression for the .search() bug: row 0 is safe, row 1 is singular. A
    single-row scan forwards this; the whole script must be refused.
    """
    script = _movej_joints(_Q_OK) + _movej_joints(_Q_SINGULAR)
    assert rig._send_script(script) is False
    assert rig._robot.sent == []


def test_out_of_workspace_refused_on_a_later_blended_row(rig):
    script = _movej_pose(_POSE_OK) + _movej_pose(_POSE_BAD)
    assert rig._send_script(script) is False
    assert rig._robot.sent == []


def test_all_safe_blended_rows_are_forwarded(rig):
    script = (
        _movej_pose(_POSE_OK)
        + _movej_joints(_Q_OK)
        + _movej_pose(_POSE_OK, r=0.0)
    )
    assert rig._send_script(script) is True
    assert len(rig._robot.sent) == 1


def test_joint_predicate_actually_evaluates_the_barrier(rig):
    """
    Guards against the predicate degenerating to `return True` again (e.g. by
    keying off an attribute that does not exist). Asserts the boundary is
    where the barrier math says it is, not merely that something was refused.
    """
    eps = rig._cfg.eps_singularity
    for q2 in (0.02, 0.05, np.pi - 0.05):
        h = (q2**2) * ((q2 - np.pi) ** 2) - eps**2
        assert h < 0, f"test fixture wrong: q2={q2} is not inside the barrier"
        q = f"3.2,-1.87,{q2},5.24,1.57,0.09"
        assert rig._send_script(_movej_joints(q)) is False, f"q2={q2} let through"

    for q2 in (-1.7903, 1.5, -2.5):
        h = (q2**2) * ((q2 - np.pi) ** 2) - eps**2
        assert h > 0, f"test fixture wrong: q2={q2} is inside the barrier"
        q = f"3.2,-1.87,{q2},5.24,1.57,0.09"
        assert rig._send_script(_movej_joints(q)) is True, f"q2={q2} wrongly refused"


def test_disabled_barrier_forwards_everything(rig):
    """eps_singularity <= 0 disables the check, matching _singularity_barrier."""
    rig._cfg.eps_singularity = 0.0
    assert rig._send_script(_movej_joints(_Q_SINGULAR)) is True

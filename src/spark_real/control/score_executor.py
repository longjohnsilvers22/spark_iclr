"""
SPARK Score Executor (robot-agnostic).

Maps SPARK behavior tree primitives to concrete robot motions via a
mixin-composed class. Each mixin lives in its own module under
control/executor_*.py:

  - executor_core.py:    ScoreExecutorCore (init, orchestration, dispatch)
  - executor_motion.py:  MotionMixin (move_to, servo, approach, transport)
  - executor_ik.py:      IkMixin (movej_via_ik, movej_to_pose, legato)
  - executor_grasp.py:   GraspMixin (gripper, grasp, verify grasp)
  - executor_release.py: ReleaseMixin (release with tilt)
  - executor_verify.py:  VerifyMixin (task verification, recovery, snapshots)
  - executor_types.py:   AbortRequested, ExecutionResult, OSC helpers
"""

from spark_real.control.executor_core import ScoreExecutorCore
from spark_real.control.executor_grasp import GraspMixin
from spark_real.control.executor_ik import IkMixin
from spark_real.control.executor_motion import MotionMixin
from spark_real.control.executor_release import ReleaseMixin
from spark_real.control.executor_types import AbortRequested, ExecutionResult
from spark_real.control.executor_verify import VerifyMixin


class ScoreExecutor(
    ScoreExecutorCore,
    MotionMixin,
    IkMixin,
    GraspMixin,
    ReleaseMixin,
    VerifyMixin,
):
    """
    Executes SPARK behavior tree scores against any shim-compliant robot.

    Designed for composition, not inheritance: pass any driver (or
    SafeRobot wrapping one) that implements the shared shim interface
    (UR10e, Franka FR3, ...). UR-specific shortcuts are guarded by
    hasattr checks so the executor degrades cleanly on other arms.
    """

    pass


__all__ = ["ScoreExecutor", "ExecutionResult", "AbortRequested"]

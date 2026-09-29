"""A run must report ITS OWN verdict, never the previous run's.

`reset_run_state` used to live only in pipeline_execution.execute(). A dry run,
a no-robot run and a capture failure all skip execute(), so the outcome read at
the end of run_task was whatever the LAST run left on the executor -- a dry run
that moved nothing reported the earlier run's PASS as its own success, wrote
`success: true` for /scores, and returned that run's verify evidence.

No robot, no camera: the pipeline surface run_task touches is faked.
"""

from __future__ import annotations

import contextlib
import tempfile

import numpy as np

from spark_real.control.success_predicates import PASS, UNVERIFIED, VerifyOutcome
from spark_real.pipeline_io import IOMixin
from spark_real.pipeline_run import RunMixin
from spark_real.pipeline_types import PipelineConfig

STALE = VerifyOutcome(status=PASS, reason="the PREVIOUS run really did pass")


class FakeExecutor:
    def __init__(self):
        self.verify_outcome = STALE
        self._placed_labels = set()
        self._abort = False


class FakePipeline(RunMixin, IOMixin):
    """Only the surface run_task calls; nothing here touches hardware."""

    def __init__(self):
        self.config = PipelineConfig(save_captures=False, output_dir=tempfile.mkdtemp())
        self.config.max_task_passes = 1
        self._robot = None  # dry run / no robot -> execute() never runs
        self._executor = FakeExecutor()
        self._bt_library = None
        self._planner = None
        self._task_history = []
        self._last_plan_source = "cache"
        self._last_bt_hash = "cafebabe"
        self._last_label_resolutions = {}

    @contextlib.contextmanager
    def activity(self, _name):
        yield

    def capture(self):
        return {"sideview": {"rgb": np.zeros((16, 16, 3), dtype=np.uint8), "depth": None}}

    def detect(self, *a, **k):
        return []

    def plan(self, *a, **k):
        return {"task": "t", "tree": {"type": "sequence", "children": []}}

    def resolve_cached_bt(self, instruction):
        return None

    def plan_mode(self):
        return "auto"

    def _extract_prompts(self, instruction):
        return ["thing"]

    def task_spec(self, instruction):
        return None

    def merge_detections(self, dets, **k):
        return []


def test_a_run_that_executed_nothing_does_not_inherit_the_last_verdict():
    pipe = FakePipeline()
    assert pipe._executor.verify_outcome.status == PASS  # left over from before

    result = pipe.run_task("pick up the fork", execute=False)

    assert result.execution_results == []
    assert result.success is False
    assert result.verify_status == UNVERIFIED
    assert result.verify is None


def test_the_stale_outcome_is_cleared_from_the_executor_itself():
    """Not just masked at the read site: the executor state is reset."""
    pipe = FakePipeline()
    pipe.run_task("pick up the fork", execute=False)
    assert pipe._executor.verify_outcome is None

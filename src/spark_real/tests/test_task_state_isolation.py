"""One executor runs every task; task-scoped state must not cross the boundary.

The rig failure this pins: "take the plushie out of the bowl" finished, the
release marked 'plushie' in ``_placed_labels``, and nothing cleared that set.
The NEXT task opened with

    move_to_keypoint 'plushie': already in container 'bowl', skipping pick
    approach

so the arm never descended and the gripper closed at hover height while the
plushie sat on the table.

No robot: a fake driver plus a ScoreExecutor whose motion leaves only record.
The dispatch loop, the container-skip branch and the reset boundary are the
real ones.
"""

from __future__ import annotations

import numpy as np

from spark_real.control import success_verifier
from spark_real.control.executor_types import ExecutionResult
from spark_real.control.score_executor import ScoreExecutor

BOWL_POS = [-0.75, 0.20, -0.20]
# Where the first task's release left the plushie, and where the second task
# re-detects it: 8 cm from the bowl centroid, i.e. still inside the container
# region (radius max(0.10, obb_minor*0.6)). That is the rig geometry -- the
# position gate alone cannot tell "in the bowl" from "beside the bowl", which
# is exactly why the placed-set has to be per-task.
PLUSHIE_AFTER = [-0.75, 0.28, -0.22]
PLUSHIE_BEFORE = [-0.90, -0.10, -0.22]

SCORE = {
    "task": "put the plushie in the bowl",
    "tree": {
        "type": "sequence",
        "children": [
            {"type": "move_to_keypoint", "params": {"keypoint_label": "plushie"}},
            {"type": "grasp", "params": {"force": 100}},
            {
                "type": "move_to_keypoint",
                "params": {"keypoint_label": "bowl", "offset_z": 0.05},
            },
            {"type": "release", "params": {}},
        ],
    },
}


def _detections(plushie_pos):
    return {
        "plushie": {"label": "plushie", "position_3d": list(plushie_pos), "confidence": 0.8},
        "bowl": {"label": "bowl", "position_3d": list(BOWL_POS), "confidence": 0.9},
    }


class FakeRobot:
    GRIPPER_TYPE = "robotiq_2f85"
    robot_family = "ur10e"

    def __init__(self):
        self.tcp = np.array([-0.80, 0.0, 0.30, 2.3038, 2.0802, -0.0048])
        self.jaw = 0.0  # 0=open .. 255=closed
        self.obj = False

    def get_tcp_pose(self):
        return self.tcp.copy()

    def get_observation(self):
        return {"tcp_pose": self.tcp.copy()}

    def get_gripper_position(self, publish=True):
        return self.jaw

    def is_object_detected(self, publish=True):
        return self.obj

    def _publish_gripper_state(self, force=False):
        return None

    def open_gripper(self):
        self.jaw = 0.0
        self.obj = False

    def close_gripper(self, **kwargs):
        self.jaw = 180.0
        self.obj = True


class RecordingExecutor(ScoreExecutor):
    """Real dispatch and real skip logic; the motion leaves only record."""

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.approached = []
        self.transported = []

    def _approach_target(self, target, detection=None, params=None):
        self.approached.append(np.asarray(target, dtype=float).copy())

    def _transport_to(
        self, target, target_detection=None, target_label="", params=None, **kwargs
    ):
        self._last_place_label = target_label
        self.transported.append(target_label)
        return True

    def _grasp(self, params, t0):
        self.robot.close_gripper()
        self._holding = True
        return ExecutionResult(action_type="grasp", success=True, message="fake grasp")

    def _release(self, params, t0):
        self.robot.open_gripper()
        self._holding = False
        return ExecutionResult(action_type="release", success=True, message="fake release")


def _new_executor():
    return RecordingExecutor(FakeRobot(), detection_map={}, velocity=0.2)


def _run_task(ex, plushie_pos, reset=True):
    """One task the way the pipeline runs one: reset, then detections, then score."""
    if reset:
        success_verifier.reset_run_state(ex)
    ex.update_detections(_detections(plushie_pos))
    # pipeline_execution derives this from the plan's place targets, AFTER the
    # reset -- so a fresh run really does re-arm the container gate.
    ex._destination_labels = {"bowl"}
    ex.approached.clear()
    ex.transported.clear()
    return ex.execute_score(SCORE)


def test_first_task_places_the_plushie_and_records_it():
    ex = _new_executor()
    results = _run_task(ex, PLUSHIE_BEFORE)
    assert [r.success for r in results] == [True, True, True, True]
    assert len(ex.approached) == 1
    assert ex.transported == ["bowl"]
    assert ex._placed_labels == {"plushie"}


def test_the_second_task_still_approaches_the_same_label():
    """THE rig failure. Same label, new location, and it must be picked."""
    ex = _new_executor()
    _run_task(ex, PLUSHIE_BEFORE)
    assert ex._placed_labels == {"plushie"}

    results = _run_task(ex, PLUSHIE_AFTER)

    assert len(ex.approached) == 1, "second task skipped its pick approach"
    assert np.allclose(ex.approached[0], PLUSHIE_AFTER)
    assert "skipping" not in (results[0].message or "")
    assert all(r.success for r in results)


def test_without_the_reset_the_skip_fires():
    """The leak is real: drop the boundary and the pick is skipped again.

    Keeps the test above honest -- it passes because of the reset, not because
    the container gate happened to miss.
    """
    ex = _new_executor()
    _run_task(ex, PLUSHIE_BEFORE)

    results = _run_task(ex, PLUSHIE_AFTER, reset=False)

    assert ex.approached == []
    assert "already in container 'bowl'" in results[0].message


def test_holding_does_not_survive_an_aborted_task():
    """A run that died holding must not make the next pick a place."""
    ex = _new_executor()
    ex.update_detections(_detections(PLUSHIE_BEFORE))
    ex._holding = True

    success_verifier.reset_run_state(ex)
    assert ex._holding is False

    _run_task(ex, PLUSHIE_BEFORE)
    assert len(ex.approached) == 1, "pick went down the transport branch"


def test_grasp_retry_budget_is_per_task():
    ex = _new_executor()
    ex._grasp_retry_count = {"plushie": 1}  # burned this task's one retry
    success_verifier.reset_run_state(ex)
    assert ex._grasp_retry_count == {}


def test_every_task_scoped_field_is_actually_cleared():
    """Guards the audit list itself: add a field, add it to _TASK_SCOPED_STATE."""
    ex = _new_executor()
    dirty = {
        "_holding": True,
        "_placed_labels": {"fork"},
        "_last_pick_label": "fork",
        "_last_place_label": "tray",
        "_last_keypoint_label": "fork",
        "_last_action_failed": True,
        "_grasp_retry_count": {"fork": 1},
        "_destination_labels": {"tray"},
        "_grasp_perception_target_z": -0.19,
        "_last_grasp_target_width": 0.04,
        "_active_grasp_label": "fork",
        "_active_grasp_orient": [0.0, 0.0, 0.0],
        "_active_grasp_strategy": "obb",
    }
    for attr, value in dirty.items():
        setattr(ex, attr, value)

    success_verifier.reset_run_state(ex)

    for attr in dirty:
        assert not getattr(ex, attr), f"{attr} survived the task boundary"


def test_closed_loop_passes_keep_the_placed_set():
    """A PASS boundary is not a task boundary: pass 3 must still know pass 1
    handled the plushie, or the loop re-picks it out of the bowl."""
    ex = _new_executor()
    _run_task(ex, PLUSHIE_BEFORE)

    success_verifier.reset_run_state(ex, task_scope=False)

    assert ex._placed_labels == {"plushie"}
    assert ex.verify_outcome is None  # verdict state still cleared per pass

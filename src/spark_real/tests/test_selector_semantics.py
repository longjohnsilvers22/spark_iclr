"""
Tests for BT selector (fallback) node semantics in ScoreExecutor.

Verifies:
  1. Flat sequence BTs still work identically (backward compat).
  2. Selector node: first branch succeeds -> second never runs.
  3. Selector node: first branch fails -> recovery + re-detect -> second branch runs.
  4. Selector node: all branches fail -> selector fails.
  5. Nested selector inside a sequence works.
  6. _flatten_tree preserves selector markers but flattens sequences.

Usage:
    python -m spark_real.tests.test_selector_semantics
    # or
    python src/spark_real/tests/test_selector_semantics.py
Exit 0 = all passed; non-zero = failed.
"""

from __future__ import annotations
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

import numpy as np
from dataclasses import dataclass
from typing import List

from spark_real.control.score_executor import ScoreExecutor, ExecutionResult

# BT selector/fallback execution IS implemented: _flatten_tree emits an opaque
# _bt_selector marker and _run_selector walks the branches (executor_core.py).
# These tests were xfail while that was a real gap; they are live now.

# Mock infrastructure


class _MockRobot:
    """
    Bare-minimum robot shim for the executor.
    """

    robot_family = "franka"
    GRIPPER_TYPE = "franka_hand"
    SUPPORTS_URSCRIPT = False
    HOME_CONFIG = [0.0, -0.78, 0.0, -2.36, 0.0, 1.57, 0.78]

    def __init__(self):
        self._tcp = np.array([0.4, 0.0, 0.35])
        self._open = True
        self._log: List[str] = []

    def get_tcp_pose(self):
        return np.concatenate([self._tcp, [np.pi, 0, 0]])

    def get_joint_positions(self):
        return np.array([0.0, -0.78, 0.0, -2.36, 0.0, 1.57, 0.78])

    def open_gripper(self):
        self._open = True
        self._log.append("open_gripper")

    def close_gripper(self):
        self._open = False
        self._log.append("close_gripper")

    def set_gripper_position(self, pos, speed=50, force=50):
        self._open = pos < 0.5
        self._log.append(f"set_gripper_position({pos})")

    def get_gripper_position(self):
        return 0 if self._open else 200

    def get_gripper_width(self):
        return 0.08 if self._open else 0.02

    def is_object_detected(self):
        return not self._open

    def get_tcp_force(self):
        return np.zeros(6)

    def move_linear(self, pose, velocity=0.1, asynchronous=False):
        self._tcp = np.array(pose[:3])
        self._log.append(f"move_linear({pose[:3]})")

    def stop(self):
        self._log.append("stop")

    def go_home(self):
        self._tcp = np.array([0.4, 0.0, 0.35])
        self._log.append("go_home")

    def grasp_to_width(self, width, force=20, speed=0.1):
        self._open = False
        self._log.append(f"grasp_to_width({width})")


class _FakeRecorder:
    """
    Stands in for TrajectoryRecorder.
    """

    def set_action_label(self, label):
        pass

    def mark_transition(self, label):
        pass

    def record(self):
        pass

    def save(self):
        pass


class _FakePipeline:
    """
    Minimal pipeline mock for re-detect calls.
    """

    def __init__(self):
        self.config = type(
            "C",
            (),
            {
                "output_dir": "/tmp",
                "robot_family": "franka",
            },
        )()
        self._detect_results = []

    def capture(self):
        return {"birdview": {"rgb": np.zeros((480, 640, 3), dtype=np.uint8)}}

    def detect(self, captures, prompts=None):
        return self._detect_results

    def merge_detections(self, dets):
        return dets


def _make_executor(
    detections=None,
    pipeline=None,
    fail_actions=None,
):
    """
    Build a ScoreExecutor with a mock robot.

    ``fail_actions`` is a set of (action_type, call_count) tuples that
    should fail. E.g. {("grasp_se3", 1)} means the first call to
    grasp_se3 fails; subsequent calls succeed.
    """
    robot = _MockRobot()
    exe = ScoreExecutor(
        robot,
        detection_map=detections or {},
        velocity=0.10,
        pipeline=pipeline,
    )
    exe._recorder = _FakeRecorder()
    exe._servo = type(
        "S",
        (),
        {
            "abort": lambda self: None,
            "max_vel_linear": 0.1,
            "move_to_pose": lambda self, *a, **kw: True,
            "_get_tcp_pose": lambda self: np.concatenate([robot._tcp, [np.pi, 0, 0]]),
        },
    )()

    # Patch _dispatch_action to track calls and optionally fail.
    _original_dispatch = exe._dispatch_action
    exe._dispatch_call_counts = {}
    exe._fail_actions = fail_actions or set()

    def _patched_dispatch(action_type, params):
        exe._dispatch_call_counts[action_type] = (
            exe._dispatch_call_counts.get(action_type, 0) + 1
        )
        count = exe._dispatch_call_counts[action_type]
        for fail_type, fail_at in exe._fail_actions:
            if action_type == fail_type and count == fail_at:
                return ExecutionResult(
                    action_type=action_type,
                    success=False,
                    message=f"Simulated failure #{count}",
                )
        return ExecutionResult(
            action_type=action_type, success=True, message=f"OK #{count}"
        )

    exe._dispatch_action = _patched_dispatch
    return exe


# Tests


def test_flat_sequence_backward_compat():
    """
    Existing flat-sequence BTs execute identically.
    """
    exe = _make_executor(
        detections={
            "cup": {"position_3d": [0.4, 0.1, 0.05]},
            "tray": {"position_3d": [0.4, -0.1, 0.05]},
        }
    )
    score = {
        "tree": {
            "type": "sequence",
            "children": [
                {"type": "move_to_keypoint", "params": {"keypoint_label": "cup"}},
                {"type": "grasp", "params": {"force": 50}},
                {"type": "move_to_keypoint", "params": {"keypoint_label": "tray"}},
                {"type": "release", "params": {}},
            ],
        }
    }
    results = exe.execute_score(score)
    action_types = [r.action_type for r in results]
    assert "move_to_keypoint" in action_types
    assert "grasp" in action_types
    assert "release" in action_types
    assert all(
        r.success for r in results
    ), f"Some actions failed: {[(r.action_type, r.message) for r in results if not r.success]}"
    print("PASS: flat_sequence_backward_compat")


def test_flatten_tree_preserves_selector():
    """
    _flatten_tree returns a _bt_selector marker for selector nodes.
    """
    exe = _make_executor()
    tree = {
        "type": "selector",
        "children": [
            {
                "type": "sequence",
                "children": [
                    {"type": "grasp", "params": {"force": 50}},
                    {"type": "release", "params": {}},
                ],
            },
            {
                "type": "sequence",
                "children": [
                    {"type": "grasp", "params": {"force": 80}},
                    {"type": "release", "params": {}},
                ],
            },
        ],
    }
    actions = exe._flatten_tree(tree)
    assert len(actions) == 1, f"Expected 1 selector marker, got {len(actions)}"
    marker = actions[0]
    assert marker.get("_bt_selector") is True
    assert len(marker["branches"]) == 2
    # Each branch should be a flat list of leaf actions
    assert len(marker["branches"][0]) == 2  # grasp + release
    assert len(marker["branches"][1]) == 2
    assert marker["branches"][0][0]["type"] == "grasp"
    assert marker["branches"][1][0]["type"] == "grasp"
    print("PASS: flatten_tree_preserves_selector")


def test_flatten_tree_fallback_alias():
    """
    'fallback' is treated as synonym for 'selector'.
    """
    exe = _make_executor()
    tree = {
        "type": "fallback",
        "children": [
            {"type": "grasp", "params": {"force": 50}},
            {"type": "grasp", "params": {"force": 80}},
        ],
    }
    actions = exe._flatten_tree(tree)
    assert len(actions) == 1
    assert actions[0].get("_bt_selector") is True
    print("PASS: flatten_tree_fallback_alias")


def test_selector_first_branch_succeeds():
    """
    When branch 1 succeeds, branch 2 never runs.
    """
    exe = _make_executor()
    score = {
        "tree": {
            "type": "selector",
            "children": [
                {
                    "type": "sequence",
                    "children": [
                        {"type": "grasp", "params": {"force": 50}},
                    ],
                },
                {
                    "type": "sequence",
                    "children": [
                        {"type": "release", "params": {}},
                    ],
                },
            ],
        }
    }
    results = exe.execute_score(score)
    dispatched = exe._dispatch_call_counts
    assert (
        dispatched.get("grasp", 0) == 1
    ), f"grasp should be called once, got {dispatched.get('grasp', 0)}"
    assert (
        dispatched.get("release", 0) == 0
    ), f"release should NOT be called (branch 2 skipped), got {dispatched.get('release', 0)}"
    # The selector itself should report success
    selector_results = [r for r in results if r.action_type == "selector"]
    assert len(selector_results) == 1
    assert selector_results[0].success is True
    print("PASS: selector_first_branch_succeeds")


def test_selector_fallback_on_failure():
    """
    When branch 1 fails, branch 2 runs (after recovery).
    """
    exe = _make_executor(
        fail_actions={("grasp", 1)},  # first grasp call fails
    )
    score = {
        "tree": {
            "type": "selector",
            "children": [
                {
                    "type": "sequence",
                    "children": [
                        {"type": "grasp", "params": {"force": 50}},
                        {"type": "release", "params": {}},
                    ],
                },
                {
                    "type": "sequence",
                    "children": [
                        {"type": "grasp", "params": {"force": 80}},
                        {"type": "release", "params": {}},
                    ],
                },
            ],
        }
    }
    results = exe.execute_score(score)
    dispatched = exe._dispatch_call_counts
    # First grasp fails, second grasp (branch 2) should succeed
    assert (
        dispatched.get("grasp", 0) == 2
    ), f"grasp should be called twice (fail + retry), got {dispatched.get('grasp', 0)}"
    # Release should only run once (in branch 2, after successful grasp)
    assert (
        dispatched.get("release", 0) == 1
    ), f"release should be called once (branch 2 only), got {dispatched.get('release', 0)}"
    # Overall selector should succeed
    selector_results = [r for r in results if r.action_type == "selector"]
    assert len(selector_results) == 1
    assert selector_results[0].success is True
    # Recovery should have been triggered (open_gripper in robot log)
    assert (
        "open_gripper" in exe.robot._log
    ), f"Expected recovery open_gripper, got {exe.robot._log}"
    assert (
        "go_home" in exe.robot._log
    ), f"Expected recovery go_home, got {exe.robot._log}"
    print("PASS: selector_fallback_on_failure")


def test_selector_all_branches_fail():
    """
    When all branches fail, the selector fails.
    """
    exe = _make_executor(
        fail_actions={("grasp", 1), ("grasp", 2)},  # both grasps fail
    )
    score = {
        "tree": {
            "type": "selector",
            "children": [
                {
                    "type": "sequence",
                    "children": [
                        {"type": "grasp", "params": {"force": 50}},
                    ],
                },
                {
                    "type": "sequence",
                    "children": [
                        {"type": "grasp", "params": {"force": 80}},
                    ],
                },
            ],
        }
    }
    results = exe.execute_score(score)
    selector_results = [r for r in results if r.action_type == "selector"]
    assert len(selector_results) == 1
    assert selector_results[0].success is False
    assert "All 2" in selector_results[0].message
    print("PASS: selector_all_branches_fail")


def test_selector_with_redetect():
    """
    Between failed branch and fallback, re-detect fires.
    """
    pipeline = _FakePipeline()
    # Provide a detection that the re-detect can update
    det = type(
        "D",
        (),
        {
            "label": "brush",
            "position_3d": [0.35, 0.05, 0.04],
            "orientation_angle": 0.0,
            "aspect_ratio": 1.0,
        },
    )()
    pipeline._detect_results = [det]

    exe = _make_executor(
        detections={"brush": {"position_3d": [0.4, 0.1, 0.05]}},
        pipeline=pipeline,
        fail_actions={("grasp_se3", 1)},
    )
    score = {
        "tree": {
            "type": "selector",
            "children": [
                {
                    "type": "sequence",
                    "children": [
                        {
                            "type": "grasp_se3",
                            "params": {
                                "keypoint_label": "brush",
                                "strategy": "top_down",
                                "force": 60,
                            },
                        },
                    ],
                },
                {
                    "type": "sequence",
                    "children": [
                        {
                            "type": "grasp_se3",
                            "params": {
                                "keypoint_label": "brush",
                                "strategy": "horizontal",
                                "force": 80,
                            },
                        },
                    ],
                },
            ],
        }
    }
    results = exe.execute_score(score)
    # Branch 1 fails, re-detect fires, branch 2 succeeds
    dispatched = exe._dispatch_call_counts
    assert dispatched.get("grasp_se3", 0) == 2
    selector_results = [r for r in results if r.action_type == "selector"]
    assert selector_results[0].success is True
    # Detection map should have been updated by re-detect
    updated = exe.detection_map.get("brush", {})
    assert updated.get("position_3d") == [
        0.35,
        0.05,
        0.04,
    ], f"Expected re-detected position, got {updated}"
    print("PASS: selector_with_redetect")


def test_selector_inside_sequence():
    """
    A selector nested inside a sequence works correctly.
    """
    exe = _make_executor(
        fail_actions={("grasp", 1)},  # first grasp fails
    )
    score = {
        "tree": {
            "type": "sequence",
            "children": [
                {"type": "move_to_keypoint", "params": {"keypoint_label": "obj"}},
                {
                    "type": "selector",
                    "children": [
                        {"type": "grasp", "params": {"force": 50}},
                        {"type": "grasp", "params": {"force": 80}},
                    ],
                },
                {"type": "release", "params": {}},
            ],
        }
    }
    results = exe.execute_score(score)
    dispatched = exe._dispatch_call_counts
    assert dispatched.get("move_to_keypoint", 0) == 1
    assert dispatched.get("grasp", 0) == 2  # first fails, second succeeds
    assert dispatched.get("release", 0) == 1
    print("PASS: selector_inside_sequence")


def test_example_bt_from_spec():
    """
    The example BT from the task spec runs correctly.
    """
    exe = _make_executor(
        detections={
            "brush": {"position_3d": [0.4, 0.1, 0.05]},
            "dustpan": {"position_3d": [0.4, -0.1, 0.05]},
        },
        fail_actions={("grasp_se3", 1)},  # top-down fails
    )
    score = {
        "tree": {
            "type": "selector",
            "children": [
                {
                    "type": "sequence",
                    "children": [
                        {
                            "type": "grasp_se3",
                            "params": {
                                "keypoint_label": "brush",
                                "strategy": "top_down",
                                "force": 60,
                            },
                        },
                        {
                            "type": "move_to_keypoint",
                            "params": {"keypoint_label": "dustpan"},
                        },
                    ],
                },
                {
                    "type": "sequence",
                    "children": [
                        {
                            "type": "move_to_keypoint",
                            "params": {"keypoint_label": "brush"},
                        },
                        {
                            "type": "grasp_se3",
                            "params": {
                                "keypoint_label": "brush",
                                "strategy": "horizontal",
                                "force": 80,
                            },
                        },
                        {
                            "type": "move_to_keypoint",
                            "params": {"keypoint_label": "dustpan"},
                        },
                    ],
                },
            ],
        }
    }
    results = exe.execute_score(score)
    dispatched = exe._dispatch_call_counts
    # Branch 1: grasp_se3 fails, sweep_to_container never runs
    # Branch 2: search + grasp_se3 + sweep all run
    assert dispatched.get("grasp_se3", 0) == 2
    # move_to_keypoint called 2x in branch 2 (brush + dustpan)
    assert dispatched.get("move_to_keypoint", 0) == 2
    selector_results = [r for r in results if r.action_type == "selector"]
    assert selector_results[0].success is True
    print("PASS: example_bt_from_spec")


# Runner


def main():
    print("Running selector semantics tests...")
    test_flat_sequence_backward_compat()
    test_flatten_tree_preserves_selector()
    test_flatten_tree_fallback_alias()
    test_selector_first_branch_succeeds()
    test_selector_fallback_on_failure()
    test_selector_all_branches_fail()
    test_selector_with_redetect()
    test_selector_inside_sequence()
    test_example_bt_from_spec()
    print("All selector semantics tests PASSED.")


if __name__ == "__main__":
    main()

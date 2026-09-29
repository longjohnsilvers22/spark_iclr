"""
Tests for spark_dsl.executor.
"""
from __future__ import annotations

import pathlib
import sys

import pytest

# Ensure the parent ``src`` dir is on sys.path so ``spark_dsl`` resolves.
_SRC = pathlib.Path(__file__).resolve().parents[2]
if str(_SRC) not in sys.path:
    sys.path.insert(0, str(_SRC))

from spark_dsl.executor import BTExecutor, ExecutionResult, StepLog  # noqa: E402
from spark_dsl.skill_library import SkillLibrary  # noqa: E402


# Helpers

_NUM = (int, float)
_STR = (str,)


def _fresh_library() -> SkillLibrary:
    """
    Each test gets its own library so macro registrations don't leak.
    """
    return SkillLibrary()


def _record_runtime(log: list, name: str, *, raise_exc: Exception = None):
    def _fn(env, slots):
        log.append((name, dict(slots)))
        if raise_exc is not None:
            raise raise_exc
    return _fn


# 1. Three-step BT, all succeed

def test_three_step_sequence_all_succeed():
    lib = _fresh_library()
    executor = BTExecutor(library=lib)
    log: list = []
    executor.register_runtime("move_to_keypoint",
                              _record_runtime(log, "move_to_keypoint"))
    executor.register_runtime("grasp", _record_runtime(log, "grasp"))
    executor.register_runtime("release", _record_runtime(log, "release"))

    bt = {
        "tree": {
            "type": "sequence",
            "children": [
                {"type": "move_to_keypoint",
                 "params": {"keypoint_label": "red block", "offset_z": 0.0}},
                {"type": "grasp", "params": {"force": 100}},
                {"type": "release", "params": {}},
            ],
        }
    }

    env = object()
    result = executor.execute(bt, env=env)

    assert isinstance(result, ExecutionResult)
    assert result.success is True
    assert result.errors == []
    assert len(result.steps) == 3
    for step in result.steps:
        assert isinstance(step, StepLog)
        assert step.success is True
        assert step.error is None
        assert step.duration_ms >= 0.0
    assert [s.primitive for s in result.steps] == [
        "move_to_keypoint", "grasp", "release",
    ]
    # runtime callbacks were invoked in order with the right slot args
    assert [name for name, _ in log] == [
        "move_to_keypoint", "grasp", "release",
    ]
    assert log[0][1] == {"keypoint_label": "red block", "offset_z": 0.0}
    assert log[1][1] == {"force": 100}


# 2. Macro expansion

def test_macro_expansion_substitutes_free_slots():
    lib = _fresh_library()
    pick_place_body = {
        "type": "sequence",
        "children": [
            {"type": "move_to_keypoint",
             "params": {"keypoint_label": "<pick_label>", "offset_z": 0.0}},
            {"type": "grasp", "params": {"force": 100}},
            {"type": "move_relative", "params": {"dz": 0.2}},
            {"type": "move_to_keypoint",
             "params": {"keypoint_label": "<place_label>", "offset_z": 0.05}},
            {"type": "release", "params": {}},
        ],
    }
    lib.register_macro(
        "pick_and_place",
        signature={
            "pick_label": (_STR, True, "object to pick"),
            "place_label": (_STR, True, "destination"),
        },
        body_subtree=pick_place_body,
    )

    executor = BTExecutor(library=lib)
    log: list = []
    for prim in ("move_to_keypoint", "grasp", "move_relative", "release"):
        executor.register_runtime(prim, _record_runtime(log, prim))

    bt = {
        "tree": {
            "type": "sequence",
            "children": [
                {"type": "pick_and_place",
                 "params": {"pick_label": "red block",
                            "place_label": "white plate"}},
            ],
        }
    }

    result = executor.execute(bt, env=None)

    assert result.success is True, result.errors
    # macro should have expanded to 5 base-primitive calls
    assert [name for name, _ in log] == [
        "move_to_keypoint", "grasp", "move_relative",
        "move_to_keypoint", "release",
    ]
    # free slots substituted with caller's args
    assert log[0][1]["keypoint_label"] == "red block"
    assert log[3][1]["keypoint_label"] == "white plate"
    # bound slot preserved
    assert log[1][1] == {"force": 100}
    # step log records the *expanded* primitives, not the macro
    assert [s.primitive for s in result.steps] == [
        "move_to_keypoint", "grasp", "move_relative",
        "move_to_keypoint", "release",
    ]
    # macro itself does not appear in the step log
    assert all(s.primitive != "pick_and_place" for s in result.steps)


def test_macro_expansion_handles_dotted_legacy_placeholders():
    """
    macro_mining produces ``<node0.keypoint_label>``-style placeholders.
    """
    lib = _fresh_library()
    body = {
        "type": "sequence",
        "children": [
            {"type": "move_to_keypoint",
             "params": {"keypoint_label": "<node0.keypoint_label>",
                        "offset_z": 0.0}},
        ],
    }
    lib.register_macro(
        "approach",
        signature={"keypoint_label": (_STR, True, "target")},
        body_subtree=body,
    )
    executor = BTExecutor(library=lib)
    log: list = []
    executor.register_runtime("move_to_keypoint",
                              _record_runtime(log, "move_to_keypoint"))

    bt = {"tree": {"type": "sequence", "children": [
        {"type": "approach", "params": {"keypoint_label": "mug"}},
    ]}}
    result = executor.execute(bt)
    assert result.success, result.errors
    assert log[0][1]["keypoint_label"] == "mug"


# 3. Validation failure -> clean errors, no callbacks invoked

def test_validation_failure_returns_clean_errors_no_callbacks():
    lib = _fresh_library()
    executor = BTExecutor(library=lib)
    log: list = []
    executor.register_runtime("grasp", _record_runtime(log, "grasp"))

    bt = {
        "tree": {
            "type": "sequence",
            "children": [
                # totally unknown primitive -> should fail validation
                {"type": "do_a_barrel_roll", "params": {}},
                {"type": "grasp", "params": {"force": 100}},
            ],
        }
    }

    result = executor.execute(bt)
    assert isinstance(result, ExecutionResult)
    assert result.success is False
    assert result.errors, "validation errors expected"
    assert any("do_a_barrel_roll" in e for e in result.errors)
    # no runtime callback was invoked
    assert log == []
    assert result.steps == []


def test_validation_failure_missing_required_slot():
    lib = _fresh_library()
    executor = BTExecutor(library=lib)
    log: list = []
    executor.register_runtime("move_to_keypoint",
                              _record_runtime(log, "move_to_keypoint"))

    # move_to_keypoint requires keypoint_label
    bt = {"tree": {"type": "sequence", "children": [
        {"type": "move_to_keypoint", "params": {"offset_z": 0.0}},
    ]}}
    result = executor.execute(bt)
    assert result.success is False
    assert any("keypoint_label" in e for e in result.errors)
    assert log == []


# 4. Runtime exception is caught + sequence aborts

def test_runtime_exception_caught_and_aborts_sequence():
    lib = _fresh_library()
    executor = BTExecutor(library=lib)
    log: list = []
    executor.register_runtime(
        "move_to_keypoint",
        _record_runtime(log, "move_to_keypoint"),
    )
    executor.register_runtime(
        "grasp",
        _record_runtime(log, "grasp",
                        raise_exc=RuntimeError("gripper jammed")),
    )
    # Should never be invoked because grasp aborts the sequence.
    executor.register_runtime("release", _record_runtime(log, "release"))

    bt = {"tree": {"type": "sequence", "children": [
        {"type": "move_to_keypoint",
         "params": {"keypoint_label": "block"}},
        {"type": "grasp", "params": {"force": 100}},
        {"type": "release", "params": {}},
    ]}}

    result = executor.execute(bt)

    assert result.success is False
    assert any("gripper jammed" in e for e in result.errors)
    assert any("RuntimeError" in e for e in result.errors)
    # only the first two primitives ran (move + the failing grasp),
    # release was never dispatched
    invoked = [name for name, _ in log]
    assert invoked == ["move_to_keypoint", "grasp"]
    # step log: first succeeded, second failed, third never appended
    assert len(result.steps) == 2
    assert result.steps[0].success is True
    assert result.steps[1].success is False
    assert "gripper jammed" in result.steps[1].error


def test_missing_runtime_emits_clean_error():
    lib = _fresh_library()
    executor = BTExecutor(library=lib)
    # deliberately do NOT register move_to_keypoint
    bt = {"tree": {"type": "sequence", "children": [
        {"type": "move_to_keypoint",
         "params": {"keypoint_label": "block"}},
    ]}}
    result = executor.execute(bt)
    assert result.success is False
    assert any("no runtime registered" in e for e in result.errors)
    assert len(result.steps) == 1
    assert result.steps[0].success is False


def test_register_runtime_rejects_unknown_and_macros():
    lib = _fresh_library()
    lib.register_macro(
        "noop_macro",
        signature={},
        body_subtree={"type": "sequence", "children": [
            {"type": "wait", "params": {"duration": 0.1}},
        ]},
    )
    executor = BTExecutor(library=lib)
    with pytest.raises(ValueError):
        executor.register_runtime("not_a_thing", lambda env, s: None)
    with pytest.raises(ValueError):
        executor.register_runtime("noop_macro", lambda env, s: None)

"""
Unit tests for the SPARK BT-DSL grammar + skill library.

5 valid BTs (pick-place, cube stack, wipe, push, drawer-with-check) must
validate cleanly; 3 malformed BTs (missing slot, wrong type, unknown
primitive) must each produce a specific error. Tests use the production
primitive names (move_to_keypoint, grasp, release, ...) that the runners
in spark_bench/run_spark_*.py and the bt_library/ cache emit.
"""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

import pytest  # noqa: E402

from spark_dsl import SkillLibrary  # noqa: E402


def _seq(*children):
    return {"tree": {"type": "sequence", "children": list(children)}}


def _a(t, **params):
    return {"type": t, "params": params}


VALID_BTS = [
    ("pick_place_bowl", _seq(
        _a("move_to_keypoint", keypoint_label="black bowl", offset_z=0.0),
        _a("grasp", force=100),
        _a("move_relative", dz=0.20),
        _a("move_to_keypoint", keypoint_label="dinner plate", offset_z=0.05),
        _a("release"),
    )),
    ("cube_stack", _seq(
        _a("move_to_keypoint", keypoint_label="red cube", offset_z=0.10),
        _a("move_to_keypoint", keypoint_label="red cube"),
        _a("grasp", force=120),
        _a("move_relative", dz=0.15),
        _a("move_to_keypoint", keypoint_label="blue cube", offset_z=0.06),
        _a("release"),
    )),
    ("wipe", _seq(
        _a("move_to_keypoint", keypoint_label="sponge"),
        _a("grasp", force=80),
        _a("wipe", keypoint_label="dirt region", sweep_width=0.55, sweep_length=0.65),
        _a("release"),
    )),
    ("push", _seq(
        _a("push_object", keypoint_label="can",
           push_direction=[0, 1, 0], push_distance=0.1),
    )),
]

INVALID_MISSING_SLOT = _seq(
    _a("push_object", push_direction=[1, 0, 0], push_distance=0.1))  # no keypoint_label

INVALID_WRONG_TYPE = _seq(_a("grasp", force="hard"))

INVALID_UNKNOWN_PRIM = _seq(_a("teleport_to_mars", target="olympus_mons"))


@pytest.fixture(scope="module")
def lib():
    return SkillLibrary()


@pytest.mark.parametrize("name,bt", VALID_BTS)
def test_valid_bts(lib, name, bt):
    ast, errors = lib.validate_bt(bt)
    assert errors == [], f"{name}: {errors}"
    assert ast is not None


def test_invalid_missing_slot(lib):
    ast, errors = lib.validate_bt(INVALID_MISSING_SLOT)
    assert ast is None
    assert any("keypoint_label" in e and "missing" in e for e in errors), errors


def test_invalid_wrong_type(lib):
    ast, errors = lib.validate_bt(INVALID_WRONG_TYPE)
    assert ast is None
    assert any("force" in e and "expected" in e for e in errors), errors


def test_invalid_unknown_primitive(lib):
    ast, errors = lib.validate_bt(INVALID_UNKNOWN_PRIM)
    assert ast is None
    assert any("teleport_to_mars" in e and "unknown" in e for e in errors), errors


def test_macro_registration_and_prompt(lib):
    lib.register_macro("pick_and_place", signature={
        "pick_label": (str, True, "object to pick"),
        "place_label": (str, True, "place target"),
    }, body_subtree={"type": "sequence", "children": [
        _a("move_to_keypoint", keypoint_label="pick_label"),
        _a("grasp", force=100),
        _a("move_to_keypoint", keypoint_label="place_label"),
        _a("release"),
    ]})
    fragment = lib.available_primitives_for_prompt()
    assert "MACRO pick_and_place" in fragment
    for p in ("move_to_keypoint", "grasp", "release", "move_relative",
              "push_object", "turn_knob", "wait", "wipe", "insert"):
        assert p in fragment


if __name__ == "__main__":
    sys.exit(pytest.main([__file__, "-v"]))

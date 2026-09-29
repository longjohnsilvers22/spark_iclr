"""
Tests for spark_dsl.macro_mining.
"""
from __future__ import annotations

import json
import pathlib
import sys

import pytest

# Ensure the parent ``src`` dir is on sys.path so ``spark_dsl`` resolves.
_SRC = pathlib.Path(__file__).resolve().parents[2]
if str(_SRC) not in sys.path:
    sys.path.insert(0, str(_SRC))

from spark_dsl import macro_mining  # noqa: E402


# Fixtures

def _pick_place_tree(target: str, dest: str, lift_dz: float = 0.2,
                     drop_z: float = 0.05) -> dict:
    return {
        "type": "sequence",
        "children": [
            {"type": "move_to_keypoint", "params": {"keypoint_label": target, "offset_z": 0}},
            {"type": "grasp", "params": {"force": 100}},
            {"type": "move_relative", "params": {"dz": lift_dz}},
            {"type": "move_to_keypoint", "params": {"keypoint_label": dest, "offset_z": drop_z}},
            {"type": "release"},
        ],
    }


@pytest.fixture
def logs_dir(tmp_path: pathlib.Path) -> pathlib.Path:
    samples = [
        ("t1", "pick the red block and place on plate",
         _pick_place_tree("red block", "white plate"), True),
        ("t2", "pick the tomato sauce and place in basket",
         _pick_place_tree("tomato sauce", "mesh basket"), True),
        ("t3", "pick the mug and place on table",
         _pick_place_tree("mug", "wood table"), True),
        ("t4", "stack blocks", {  # different shape -> shouldn't dominate
            "type": "sequence",
            "children": [
                {"type": "move_to_keypoint", "params": {"keypoint_label": "block", "offset_z": 0}},
                {"type": "grasp", "params": {"force": 50}},
            ],
        }, True),
        ("t5", "failed pick", _pick_place_tree("nope", "nope"), False),
    ]
    for trial, task, tree, success in samples:
        path = tmp_path / f"{trial}.json"
        path.write_text(json.dumps({
            "trial_id": trial,
            "task_string": task,
            "bt": tree,
            "success": success,
            "slot_bindings": {},
        }))
    return tmp_path


# Tests

def test_load_logs_skips_failed_and_handles_dir(logs_dir: pathlib.Path) -> None:
    logs = macro_mining.load_logs(logs_dir)
    assert len(logs) == 5
    successful = [l for l in logs if l.get("success")]
    assert len(successful) == 4


def test_mine_macros_emits_pick_place_with_min_support(logs_dir: pathlib.Path) -> None:
    logs = macro_mining.load_logs(logs_dir)
    macros = macro_mining.mine_macros(logs, min_support=3)
    assert macros, "at least one candidate must be mined"

    # Top candidate should be the longest sequence (pick-and-place, 5 nodes)
    top = macros[0]
    assert len(top["body"]) == 5
    assert top["distinct_tasks"] >= 3
    assert top["name"] == "pick_and_place"

    # All node types present in expected order
    expected_types = ["move_to_keypoint", "grasp", "move_relative",
                       "move_to_keypoint", "release"]
    assert [n["type"] for n in top["body"]] == expected_types


def test_slot_inference_marks_keypoint_label_as_free(logs_dir: pathlib.Path) -> None:
    logs = macro_mining.load_logs(logs_dir)
    macros = macro_mining.mine_macros(logs, min_support=3)
    top = macros[0]

    # keypoint_label varies -> free; force=100 is bound across occurrences.
    assert any("keypoint_label" in k for k in top["free_params"])
    assert any(k.endswith(".force") and v == 100
               for k, v in top["bound_params"].items())

    # The body should have placeholders where free, literals where bound.
    grasp_node = top["body"][1]
    assert grasp_node["params"]["force"] == 100
    move_node = top["body"][0]
    assert isinstance(move_node["params"]["keypoint_label"], str)
    assert move_node["params"]["keypoint_label"].startswith("<")


def test_min_support_filters_uncommon_patterns(logs_dir: pathlib.Path) -> None:
    logs = macro_mining.load_logs(logs_dir)
    high_support = macro_mining.mine_macros(logs, min_support=10)
    assert high_support == []


def test_cli_writes_yaml(tmp_path: pathlib.Path, logs_dir: pathlib.Path) -> None:
    out = tmp_path / "macros.yaml"
    rc = macro_mining._cli([
        "--logs-dir", str(logs_dir),
        "--out", str(out),
        "--min-support", "3",
    ])
    assert rc == 0
    assert out.exists()
    import yaml
    parsed = yaml.safe_load(out.read_text())
    assert "macros" in parsed
    assert isinstance(parsed["macros"], list)
    assert len(parsed["macros"]) >= 1

"""
Unit tests for ``spark_dsl.prompt_builder.PromptBuilder``.

Verifies the prompt:
- contains every base primitive name + every dynamically-loaded macro name,
- correctly formats few-shot (task, BT) pairs as YAML,
- loads ``spark_dsl/macros.yaml`` and registers at least one macro on the
  supplied :class:`SkillLibrary`.
"""

from __future__ import annotations

import sys
from pathlib import Path

import pytest
import yaml

_SRC = Path(__file__).resolve().parents[2]
if str(_SRC) not in sys.path:
    sys.path.insert(0, str(_SRC))

from spark_dsl import SkillLibrary  # noqa: E402
from spark_dsl.prompt_builder import PromptBuilder  # noqa: E402

BASE_PRIMITIVES = (
    "move_to_keypoint", "grasp", "release", "move_relative",
    "push_object", "turn_knob", "wait", "wipe", "insert",
)

MACROS_YAML = _SRC / "spark_dsl" / "macros.yaml"


@pytest.fixture
def library() -> SkillLibrary:
    return SkillLibrary()


@pytest.fixture
def builder(library: SkillLibrary) -> PromptBuilder:
    return PromptBuilder(library=library, macros_yaml=str(MACROS_YAML))


# Macro loading

def test_macros_yaml_loads_at_least_one(library: SkillLibrary) -> None:
    """
    The shipped macros.yaml registers >=1 macro on the library.
    """
    assert MACROS_YAML.exists(), "macros.yaml fixture missing"
    builder = PromptBuilder(library=library, macros_yaml=str(MACROS_YAML))
    assert len(builder.loaded_macro_names) >= 1
    prims = library.primitives()
    for name in builder.loaded_macro_names:
        assert name in prims
        assert prims[name].is_macro is True


def test_macros_yaml_picks_clean_signature(library: SkillLibrary) -> None:
    """
    Free `keypoint_label` slots map to pick_label / place_label.
    """
    builder = PromptBuilder(library=library, macros_yaml=str(MACROS_YAML))
    prims = library.primitives()
    macro = next((p for n, p in prims.items() if p.is_macro and "pick_and_place" in n), None)
    assert macro is not None, "expected a pick_and_place macro to load"
    slot_names = list(macro.slots.keys())
    assert "pick_label" in slot_names
    assert "place_label" in slot_names
    # No raw nodeI.slot keys leaked into the signature.
    for s in slot_names:
        assert "node" not in s or not s.startswith("node")
    # And the body has been re-written to reference the clean names.
    body_str = yaml.safe_dump(macro.body)
    assert "<node0." not in body_str
    assert "pick_label" in body_str
    assert "place_label" in body_str


def test_missing_macros_yaml_is_silent(library: SkillLibrary, tmp_path: Path) -> None:
    """
    Builder should not raise if macros.yaml is missing.
    """
    missing = tmp_path / "no_such.yaml"
    pb = PromptBuilder(library=library, macros_yaml=str(missing))
    assert pb.loaded_macro_names == []


# Prompt content

def test_prompt_contains_all_base_primitives(builder: PromptBuilder) -> None:
    prompt = builder.build(task_instruction="put the bowl on the plate",
                           detected_objects=["black bowl", "white plate"])
    for name in BASE_PRIMITIVES:
        assert name in prompt, f"missing base primitive {name!r} in prompt"


def test_prompt_contains_all_loaded_macros(library: SkillLibrary,
                                            builder: PromptBuilder) -> None:
    prompt = builder.build(task_instruction="put the bowl on the plate",
                           detected_objects=["black bowl", "white plate"])
    assert builder.loaded_macro_names, "no macros loaded - test fixture broken"
    for name in builder.loaded_macro_names:
        assert name in prompt, f"missing macro {name!r} in prompt"


def test_prompt_includes_task_and_objects(builder: PromptBuilder) -> None:
    prompt = builder.build(task_instruction="put the bowl on the plate",
                           detected_objects=["black bowl", "white plate"])
    assert "put the bowl on the plate" in prompt
    assert "black bowl" in prompt
    assert "white plate" in prompt
    assert "Output: a single typed YAML BT, no commentary." in prompt
    assert "Detected objects:" in prompt


def test_prompt_handles_no_detected_objects(library: SkillLibrary) -> None:
    pb = PromptBuilder(library=library, macros_yaml=None)
    prompt = pb.build(task_instruction="do nothing")
    assert "Detected objects" in prompt
    assert "do nothing" in prompt


def test_extra_hints_appear_in_prompt(library: SkillLibrary) -> None:
    pb = PromptBuilder(library=library, macros_yaml=None)
    prompt = pb.build(task_instruction="x", detected_objects=["a"],
                      extra_hints="be quick")
    assert "Hints:" in prompt
    assert "be quick" in prompt


# Few-shot formatting

_FEWSHOT_BT = {
    "tree": {
        "type": "sequence",
        "children": [
            {"type": "move_to_keypoint",
             "params": {"keypoint_label": "red block", "offset_z": 0}},
            {"type": "grasp", "params": {"force": 100}},
            {"type": "move_relative", "params": {"dz": 0.2}},
            {"type": "move_to_keypoint",
             "params": {"keypoint_label": "white plate", "offset_z": 0.05}},
            {"type": "release", "params": {}},
        ],
    },
}


def test_fewshot_examples_formatted_as_yaml(library: SkillLibrary) -> None:
    pb = PromptBuilder(
        library=library, macros_yaml=None,
        fewshot_examples=[("pick the red block and place on white plate",
                            _FEWSHOT_BT)],
    )
    prompt = pb.build(task_instruction="put the bowl on the plate",
                      detected_objects=["bowl", "plate"])
    assert "Few-shot examples:" in prompt
    assert "Example 1 task: pick the red block and place on white plate" in prompt
    # The YAML body should round-trip through safe_load. Find the fenced
    # block AFTER the "Few-shot examples:" header so we don't grab the
    # syntax-example block earlier in the prompt.
    fewshot_anchor = prompt.index("Few-shot examples:")
    start = prompt.index("```yaml", fewshot_anchor)
    end = prompt.index("```", start + 7)
    yaml_body = prompt[start + len("```yaml"):end].strip()
    parsed = yaml.safe_load(yaml_body)
    assert parsed == _FEWSHOT_BT


def test_fewshot_optional(library: SkillLibrary) -> None:
    pb = PromptBuilder(library=library, macros_yaml=None)
    prompt = pb.build(task_instruction="x", detected_objects=["a"])
    assert "Few-shot examples:" not in prompt


if __name__ == "__main__":  # pragma: no cover
    sys.exit(pytest.main([__file__, "-v"]))

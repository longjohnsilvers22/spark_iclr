"""
Offline tests for the planner's extended plan schema.

No Gemini, no network, no robot, no camera. Every LLM response here is a
fabricated string in tests/fixtures/planner/, parsed through exactly the code
path generate_score() uses (parse_plan_yaml -> sanitize_score).

Covers:
  - an old-style plan (no new fields) parses to the identical dict it always
    did, and hashes to the identical BT cache key;
  - each new optional field carries through;
  - malformed new fields are rejected loudly (strict) or dropped with a
    WARNING (lenient), never silently honoured;
  - a malformed verify block is rejected WHOLE, never partially;
  - unparseable YAML raises PlanSchemaError, not a raw YAMLError or a 500.
"""

from __future__ import annotations

import copy
import json
import logging
from pathlib import Path

import pytest
import yaml

from spark_real.bt_library import _hash_entry
from spark_real.planning.spark_planner import (
    EXTENSION_DEFAULTS,
    GRASP_STRATEGIES,
    PlanSchemaError,
    SPARKPlanner,
    _VERIFY_LABEL_FIELDS,
    _VERIFY_SPEC,
    normalize_plan_for_hash,
    parse_plan_yaml,
    sanitize_score,
    validate_verify_block,
)

FIXTURES = Path(__file__).parent / "fixtures" / "planner"
BT_LIBRARY = Path(__file__).resolve().parents[3] / "bt_library"

# Labels the scene is assumed to have detected, for label validation.
LABELS = ["knife handle 1", "tray", "plushie", "bowl"]


def load(name: str) -> str:
    return (FIXTURES / name).read_text()


def parse(name: str, labels=LABELS) -> dict:
    return SPARKPlanner.parse_response(load(name), labels)


# --------------------------------------------------------------------------
# 1. Old-style plans are untouched.
# --------------------------------------------------------------------------


def test_old_style_plan_parses_to_exactly_the_old_dict():
    score = parse("old_style.yaml")
    assert score == {
        "task": "put the knife in the tray",
        "tree": {
            "type": "sequence",
            "children": [
                {
                    "type": "move_to_keypoint",
                    "params": {"keypoint_label": "knife handle 1", "offset_z": 0},
                },
                {"type": "grasp", "params": {"force": 50, "target_width": 0.008}},
                {"type": "move_relative", "params": {"dx": 0, "dy": 0, "dz": 0.20}},
                {
                    "type": "place_in_slot",
                    "params": {"container_label": "tray", "slot_idx": 0},
                },
            ],
        },
    }
    assert "verify" not in score
    for child in score["tree"]["children"]:
        assert not set(child["params"]) & set(EXTENSION_DEFAULTS)


def test_old_style_plan_produces_no_issues():
    score = yaml.safe_load(load("old_style.yaml").replace("```yaml", "").replace("```", ""))
    clean, issues = sanitize_score(score, LABELS)
    assert issues == []
    assert clean is score  # untouched object, not a rewritten copy


# --------------------------------------------------------------------------
# 2. New fields carry through.
# --------------------------------------------------------------------------


def test_new_style_fields_carry_through():
    score = parse("new_style.yaml")
    move, grasp = score["tree"]["children"][0], score["tree"]["children"][1]
    assert move["params"]["grasp_strategy"] == "obb"
    assert move["params"]["grasp_yaw_deg"] == 30.0
    assert grasp["params"]["grasp_strategy"] == "obb"
    assert score["verify"]["all"] == [
        {
            "pred": "inside",
            "obj": "knife handle 1",
            "container": "tray",
            "xy_margin_m": -0.02,
            "z_tol_m": 0.06,
        },
        {"pred": "held", "obj": "knife handle 1", "value": False},
    ]
    assert score["verify"]["min_conf"] == 0.35
    assert score["verify"]["require_two_views"] is False


def test_plushie_plan_can_ask_for_topdown():
    # The operator's case: the planner overrides the aspect-ratio heuristic.
    score = parse("plushie_topdown.yaml")
    strategies = [c["params"].get("grasp_strategy") for c in score["tree"]["children"]]
    assert strategies[:2] == ["topdown", "topdown"]
    assert score["verify"]["all"][1] == {
        "pred": "held",
        "obj": "plushie",
        "value": False,
    }


@pytest.mark.parametrize("strategy", GRASP_STRATEGIES)
def test_every_documented_strategy_is_accepted(strategy):
    score = {
        "task": "t",
        "tree": {
            "type": "sequence",
            "children": [{"type": "grasp", "params": {"grasp_strategy": strategy}}],
        },
    }
    clean, issues = sanitize_score(score, LABELS)
    assert issues == []
    assert clean["tree"]["children"][0]["params"]["grasp_strategy"] == strategy


# --------------------------------------------------------------------------
# 3. Malformed new fields are rejected loudly.
# --------------------------------------------------------------------------


@pytest.mark.parametrize(
    "fixture, needle",
    [
        ("bad_strategy.yaml", "unknown grasp_strategy"),
        ("bad_yaw.yaml", "out of range"),
        ("bad_pred.yaml", "unknown pred"),
        ("near_no_dist.yaml", "requires 'max_dist_m'"),
        ("unknown_label.yaml", "not in the detected keypoints"),
    ],
)
def test_malformed_extension_raises_in_strict_mode(fixture, needle):
    raw = parse_plan_yaml(load(fixture))
    with pytest.raises(PlanSchemaError) as exc:
        sanitize_score(raw, LABELS, strict=True)
    assert needle in str(exc.value)


@pytest.mark.parametrize("fixture", ["bad_strategy.yaml", "bad_yaw.yaml", "bad_pred.yaml"])
def test_malformed_extension_is_logged_not_silently_dropped(fixture, caplog):
    raw = parse_plan_yaml(load(fixture))
    with caplog.at_level(logging.WARNING):
        _clean, issues = sanitize_score(raw, LABELS)
    assert issues
    assert any("[plan-schema]" in r.message for r in caplog.records)


def test_bad_strategy_falls_back_to_todays_behaviour():
    score = parse("bad_strategy.yaml")
    params = score["tree"]["children"][0]["params"]
    assert "grasp_strategy" not in params  # dropped -> executor's `auto`
    assert params["force"] == 50  # the rest of the node survives


def test_yaw_is_only_read_under_obb():
    score = {
        "task": "t",
        "tree": {
            "type": "sequence",
            "children": [
                {"type": "grasp", "params": {"grasp_yaw_deg": 30.0}},
            ],
        },
    }
    _clean, issues = sanitize_score(score, LABELS)
    assert any("only read when grasp_strategy is 'obb'" in i for i in issues)


def test_strategy_on_a_node_that_cannot_use_it_is_an_issue():
    score = {
        "task": "t",
        "tree": {
            "type": "sequence",
            "children": [{"type": "move_relative", "params": {"grasp_strategy": "obb"}}],
        },
    }
    _clean, issues = sanitize_score(score, LABELS)
    assert any("only meaningful on" in i for i in issues)


# --------------------------------------------------------------------------
# 4. A malformed verify block is rejected WHOLE.
# --------------------------------------------------------------------------


def test_one_bad_predicate_rejects_the_entire_block():
    # bad_pred.yaml carries a valid `inside` AND a nonsense `overlapping`.
    raw = parse_plan_yaml(load("bad_pred.yaml"))
    assert len(raw["verify"]["all"]) == 2
    score = SPARKPlanner.parse_response(load("bad_pred.yaml"), LABELS)
    assert "verify" not in score, "a partially honoured predicate is worse than none"
    assert score["tree"]["children"][0]["type"] == "place_in_slot"


def test_verify_rejections_are_specific():
    assert validate_verify_block({"all": []}, LABELS)
    assert validate_verify_block({"min_conf": 0.3}, LABELS)  # missing `all`
    assert validate_verify_block("inside(a, b)", LABELS)
    assert validate_verify_block({"all": [{"pred": "held", "obj": "tray", "value": "no"}]}, LABELS)
    assert validate_verify_block(
        {"all": [{"pred": "inside", "obj": "tray", "container": "tray"}], "min_conf": 5},
        LABELS,
    )
    assert validate_verify_block(
        {"all": [{"pred": "inside", "obj": "tray", "container": "tray", "wat": 1}]},
        LABELS,
    )
    # ... and a good one is silent.
    assert (
        validate_verify_block(
            {
                "all": [
                    {"pred": "inside", "obj": "knife handle 1", "container": "tray"},
                    {"pred": "near", "obj": "tray", "target": "bowl", "max_dist_m": 0.3},
                ]
            },
            LABELS,
        )
        == []
    )


def test_instance_numbering_resolves_against_bare_labels():
    # SAM3 re-detects "fork"; the plan says "fork 1". Not an unknown label.
    assert (
        validate_verify_block(
            {"all": [{"pred": "inside", "obj": "fork 1", "container": "tray"}]},
            ["fork", "tray"],
        )
        == []
    )


# --------------------------------------------------------------------------
# 5. YAML hardening.
# --------------------------------------------------------------------------


def test_tab_indented_response_recovers_on_the_repair_retry(caplog):
    with caplog.at_level(logging.WARNING):
        score = parse_plan_yaml(load("malformed.yaml"))
    assert score["tree"]["type"] == "sequence"
    assert any("repair" in r.message for r in caplog.records)


def test_toplevel_list_is_rejected_cleanly():
    with pytest.raises(PlanSchemaError) as exc:
        parse_plan_yaml(load("toplevel_list.yaml"))
    assert "mapping" in str(exc.value) or "tree" in str(exc.value)


def test_unparseable_yaml_raises_plan_schema_error_not_yamlerror():
    with pytest.raises(PlanSchemaError):
        parse_plan_yaml(load("unparseable.yaml"))


def test_empty_response_is_rejected():
    with pytest.raises(PlanSchemaError):
        parse_plan_yaml("   ")


def test_plan_without_tree_is_rejected():
    with pytest.raises(PlanSchemaError):
        parse_plan_yaml("task: do a thing\nnotes: none\n")


# --------------------------------------------------------------------------
# 6. Hash stability over the REAL cached library.
# --------------------------------------------------------------------------


def _cached_entries():
    if not BT_LIBRARY.is_dir():
        return []
    out = []
    for path in sorted(BT_LIBRARY.glob("*.json")):
        data = json.loads(path.read_text())
        if isinstance(data.get("score"), dict) and data.get("instruction"):
            out.append((path.name, data["instruction"], data["score"]))
    return out


CACHED = _cached_entries()
requires_library = pytest.mark.skipif(not CACHED, reason="no bt_library/ checkout")


def _decorate(score):
    """Add default-valued new fields + a verify block to a cached score."""
    new = copy.deepcopy(score)
    new["verify"] = {
        "all": [{"pred": "inside", "obj": "x", "container": "y"}],
        "min_conf": 0.35,
    }

    def walk(node):
        if not isinstance(node, dict):
            return
        if node.get("type") in ("move_to_keypoint", "grasp"):
            params = node.setdefault("params", {})
            # Fill in defaults only where the plan is silent. Overwriting
            # an explicit value (the library now accumulates entries from
            # live runs, and those carry real grasp_strategy choices such
            # as "topdown") would delete semantics, not spell out a
            # default, and the resulting hash difference is correct.
            params.setdefault("grasp_strategy", "auto")
            params.setdefault("grasp_yaw_deg", None)
        for child in node.get("children") or []:
            walk(child)

    walk(new.get("tree"))
    return new


@requires_library
def test_new_schema_defaults_do_not_change_the_cache_key():
    assert len(CACHED) >= 40, f"expected the full library, got {len(CACHED)}"
    for name, instruction, score in CACHED:
        base = _hash_entry(instruction, score)
        decorated = _decorate(score)
        assert _hash_entry(instruction, normalize_plan_for_hash(decorated)) == base, (
            f"{name}: a plan that spells out the documented defaults must hash "
            f"to the cached tree"
        )


@requires_library
def test_raw_bt_library_hash_absorbs_the_new_schema_without_normalisation():
    # bt_library._strip_nonsemantic now drops `verify` and any param equal to
    # its documented default, so a plan carrying the new schema hashes to the
    # cached tree WITHOUT going through normalize_plan_for_hash first. Before
    # that landed this missed on all 42 entries, i.e. every cached tree.
    misses = [
        name
        for name, instruction, score in CACHED
        if _hash_entry(instruction, _decorate(score)) != _hash_entry(instruction, score)
    ]
    assert misses == [], f"new-schema plans became cache misses: {misses}"


@requires_library
def test_a_nondefault_strategy_is_semantic_and_changes_the_key():
    _name, instruction, score = CACHED[0]
    changed = copy.deepcopy(score)

    def walk(node):
        if not isinstance(node, dict):
            return
        if node.get("type") == "grasp":
            node.setdefault("params", {})["grasp_strategy"] = "topdown"
        for child in node.get("children") or []:
            walk(child)

    walk(changed.get("tree"))
    assert _hash_entry(instruction, normalize_plan_for_hash(changed)) != _hash_entry(
        instruction, score
    )


@requires_library
def test_cached_entries_still_validate_under_the_extended_validator():
    planner = SPARKPlanner.__new__(SPARKPlanner)  # no API key, no client
    for name, _instruction, score in CACHED:
        stripped = {k: v for k, v in score.items() if not str(k).startswith("__")}
        issues = [
            i
            for i in planner.validate_score(stripped)
            if "unknown type" not in i  # sim-only primitives not in this registry
        ]
        assert issues == [], f"{name}: {issues}"


# --------------------------------------------------------------------------
# 7. What the model is shown.
# --------------------------------------------------------------------------


def test_geometry_is_rendered_into_the_planner_context():
    detail = {
        "label": "plushie",
        "confidence": 0.24,
        "aspect_ratio": 3.14,
        "obb_minor_mm": 41.0,
        "obb_angle_deg": -68.9,
        "obb_confidence": 0.18,
        "mask_area_frac": 0.031,
        "camera": "birdview",
        "low_quality": True,
        "reprompt_attempts": 2,
    }
    text = SPARKPlanner._format_geometry(detail)
    for needle in (
        "aspect_ratio=3.14",
        "obb_confidence=0.18",
        "obb_angle=-69deg",
        "cam=birdview",
        "LOW_QUALITY_MASK",
    ):
        assert needle in text


def test_geometry_rendering_tolerates_an_old_detection_detail():
    assert SPARKPlanner._format_geometry({"label": "tray", "confidence": 0.9}) == ""


def test_prompt_documents_both_new_decisions():
    planner = SPARKPlanner.__new__(SPARKPlanner)
    planner.robot_family = "ur10e"
    prompt = planner._build_system_prompt()
    assert "grasp_strategy" in prompt
    assert "obb_confidence" in prompt
    assert "verify:" in prompt
    assert "max_dist_m" in prompt
    for strategy in GRASP_STRATEGIES:
        assert f'"{strategy}"' in prompt


def test_extract_yaml_keeps_the_task_line_when_there_is_no_code_fence():
    planner = SPARKPlanner.__new__(SPARKPlanner)
    raw = "Sure, here you go:\ntask: put the knife in the tray\ntree:\n  type: sequence\n"
    score = parse_plan_yaml(planner._extract_yaml(raw))
    assert score["task"] == "put the knife in the tray"
    assert score["tree"]["type"] == "sequence"


def test_planner_mirror_agrees_with_the_runtime_predicate_schema():
    # The planner validates what it emits; control/success_predicates.py is the
    # runtime authority and cannot import from here (numpy+stdlib only). Assert
    # the two vocabularies have not drifted.
    sp = pytest.importorskip("spark_real.control.success_predicates")
    runtime = getattr(sp, "_SCHEMA", None)
    if runtime is None:
        pytest.skip("runtime predicate schema not exposed")
    assert set(runtime) == set(_VERIFY_SPEC)
    for name, (required, optional) in runtime.items():
        assert set(required) == set(_VERIFY_SPEC[name]["required"]), name
        assert set(optional) == set(_VERIFY_SPEC[name]["optional"]), name
    assert set(sp._LABEL_KEYS) == set(_VERIFY_LABEL_FIELDS)

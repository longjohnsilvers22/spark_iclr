"""Offline proof that the planner can emit -- and the executor can run -- a
behaviour tree with BOUNDED recovery structure.

Motivating failure (real UR10e run, cached BT ``baf971e34606``): Gemini
planned a flat five-leaf sequence for "pick up the stuffed animal and place
in the bowl". The plushie landed outside the bowl and the tree had no branch
that could notice, let alone react.

Four things have to hold for that to stop happening, and each has a test here:

  1. The planner's grammar and system prompt teach acquire-then-verify and
     place-then-verify with a recovery branch, naming ONLY skills that are
     actually in the SkillRegistry.
  2. ``generate_score`` -> ``parse_response`` (driven by a FAKE Gemini client,
     no network) keeps that structure intact through sanitize + validate.
  3. Recovery is BOUNDED -- attempt counts are clamped at emit time and the
     executor enforces a global budget at run time.
  4. The resulting tree actually EXECUTES: the real ScoreExecutor walks it,
     the acquire step is forced to fail once, the recovery branch runs, and
     the tree then succeeds. Every previously cached FLAT tree still loads,
     validates, hashes and runs exactly as before.

No robot, no camera, no network.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import List

import numpy as np
import pytest
import yaml

from spark_real.bt_library import _hash_entry
from spark_real.control.score_executor import ExecutionResult, ScoreExecutor
from spark_real.planning import bt_grammar
from spark_real.planning.spark_planner import SPARKPlanner, sanitize_score
from spark_real.skills import registry as skill_registry

# The exact tree that failed on hardware. Embedded rather than read from
# output/bt_library (gitignored) so the backward-compatibility guarantee is
# pinned in the repo and does not depend on a developer's local library.
CACHED_FLAT_BT = {
    "task": "pick up the stuffed animal and place in the bowl",
    "tree": {
        "type": "sequence",
        "children": [
            {
                "type": "move_to_keypoint",
                "params": {"keypoint_label": "stuffed animal", "offset_z": 0},
            },
            {
                "type": "grasp",
                "params": {
                    "force": 20,
                    "target_width": 0.03,
                    "grasp_strategy": "topdown",
                },
            },
            {"type": "move_relative", "params": {"dx": 0, "dy": 0, "dz": 0.2}},
            {
                "type": "move_to_keypoint",
                "params": {"keypoint_label": "blue bowl", "offset_z": 0.15},
            },
            {"type": "release", "params": {"tilt_angle": 0}},
        ],
    },
    "verify": {
        "all": [
            {"pred": "inside", "obj": "stuffed animal", "container": "blue bowl"},
            {"pred": "held", "obj": "stuffed animal", "value": False},
        ]
    },
}

CACHED_FLAT_BT_HASH = "baf971e34606"

LIVE_BT_LIBRARY = Path(__file__).resolve().parents[2] / "output" / "bt_library"


# A plan in the shape the new prompt asks for: acquire-then-verify wrapped in
# a bounded retry, place-then-verify wrapped in a fallback whose recovery
# branch re-approaches and re-releases.
RECOVERY_PLAN_YAML = """
task: pick up the stuffed animal and place in the bowl
tree:
  type: sequence
  children:
    - type: retry
      params:
        max_attempts: 2
      children:
        - type: sequence
          children:
            - type: move_to_keypoint
              params:
                keypoint_label: "stuffed animal"
                offset_z: 0
            - type: grasp
              params:
                force: 20
                target_width: 0.03
            - type: verify_grasp
              params: {}
    - type: move_relative
      params:
        dz: 0.2
    - type: fallback
      children:
        - type: sequence
          children:
            - type: move_to_keypoint
              params:
                keypoint_label: "blue bowl"
                offset_z: 0.15
            - type: release
              params:
                tilt_angle: 0
            - type: verify_placed
              params:
                obj: "stuffed animal"
                container: "blue bowl"
        - type: sequence
          children:
            - type: retract_retry
              params:
                keypoint_label: "stuffed animal"
            - type: move_to_keypoint
              params:
                keypoint_label: "blue bowl"
                offset_z: 0.15
            - type: release
              params:
                tilt_angle: 0
verify:
  all:
    - pred: inside
      obj: "stuffed animal"
      container: "blue bowl"
"""

KEYPOINTS = ["stuffed animal", "blue bowl"]


# fake LLM plumbing


class _FakeGeminiResponse:
    def __init__(self, text: str):
        self.text = text


class _FakeGeminiModels:
    def __init__(self, text: str):
        self._text = text
        self.calls: List[dict] = []

    def generate_content(self, model=None, contents=None, config=None):
        self.calls.append({"model": model, "contents": contents, "config": config})
        return _FakeGeminiResponse(self._text)


class _FakeGeminiClient:
    """Stands in for google.genai.Client. Never touches the network."""

    def __init__(self, text: str):
        self.models = _FakeGeminiModels(text)


def _planner(reply: str) -> SPARKPlanner:
    planner = SPARKPlanner(llm_backend="gemini", api_key="offline-test")
    planner._client = _FakeGeminiClient(reply)
    return planner


# mock robot / executor plumbing


class _MockRobot:
    robot_family = "ur10e"
    GRIPPER_TYPE = "robotiq_2f85"
    SUPPORTS_URSCRIPT = False
    HOME_CONFIG = [0.0, -1.57, 1.57, -1.57, -1.57, 0.0]

    def __init__(self):
        self._tcp = np.array([0.4, 0.0, 0.35])
        self._open = True
        self.log: List[str] = []

    def get_tcp_pose(self):
        return np.concatenate([self._tcp, [2.22, -2.22, 0.0]])

    def get_joint_positions(self):
        return np.array(self.HOME_CONFIG, dtype=float)

    def open_gripper(self):
        self._open = True
        self.log.append("open_gripper")

    def close_gripper(self):
        self._open = False
        self.log.append("close_gripper")

    def set_gripper_position(self, pos, speed=50, force=50):
        self._open = pos < 0.5
        self.log.append(f"set_gripper_position({pos})")

    def get_gripper_position(self):
        return 0 if self._open else 200

    def get_gripper_width(self):
        return 0.085 if self._open else 0.02

    def is_object_detected(self):
        return not self._open

    def get_tcp_force(self):
        return np.zeros(6)

    def move_linear(self, pose, velocity=0.1, asynchronous=False):
        self._tcp = np.array(pose[:3])
        self.log.append("move_linear")

    def stop(self):
        self.log.append("stop")

    def go_home(self):
        self._tcp = np.array([0.4, 0.0, 0.35])
        self.log.append("go_home")

    def grasp_to_width(self, width, force=20, speed=0.1):
        self._open = False
        self.log.append("grasp_to_width")


class _FakeRecorder:
    def set_action_label(self, label):
        pass

    def mark_transition(self, label):
        pass

    def record(self):
        pass

    def save(self):
        pass


def _make_executor(detections=None, fail_actions=None):
    """Real ScoreExecutor with motion stubbed at the dispatch boundary.

    ``fail_actions`` maps action_type -> set of 1-based call indices that
    should return failure, so a test can force "the acquire fails exactly
    once" without any robot.
    """
    robot = _MockRobot()
    exe = ScoreExecutor(robot, detection_map=detections or {}, velocity=0.1)
    exe._recorder = _FakeRecorder()
    exe._servo = type(
        "S",
        (),
        {
            "abort": lambda self: None,
            "max_vel_linear": 0.1,
            "move_to_pose": lambda self, *a, **kw: True,
            "_get_tcp_pose": lambda self: np.concatenate(
                [robot._tcp, [2.22, -2.22, 0.0]]
            ),
        },
    )()

    exe.dispatch_counts = {}
    fail_actions = fail_actions or {}

    def _patched_dispatch(action_type, params):
        exe.dispatch_counts[action_type] = exe.dispatch_counts.get(action_type, 0) + 1
        n = exe.dispatch_counts[action_type]
        if n in fail_actions.get(action_type, set()):
            return ExecutionResult(
                action_type=action_type,
                success=False,
                message=f"forced failure #{n}",
            )
        return ExecutionResult(action_type=action_type, success=True, message=f"ok #{n}")

    exe._dispatch_action = _patched_dispatch
    return exe


# 1. grammar + prompt


def test_recovery_skill_names_are_registered():
    """Every skill the recovery grammar advertises must really exist.

    A prompt that names a skill the SkillRegistry does not have produces a
    tree that dies at dispatch -- strictly worse than a flat sequence.
    """
    missing = [n for n in bt_grammar.RECOVERY_SKILLS if skill_registry.get(n) is None]
    assert not missing, f"recovery grammar names unregistered skills: {missing}"


def test_system_prompt_teaches_recovery_grammar():
    planner = _planner("")
    prompt = planner._build_system_prompt()
    assert "fallback" in prompt
    assert "retry" in prompt
    for name in ("verify_grasp", "verify_placed"):
        assert name in prompt, f"prompt never mentions the {name} condition"
    # The worked example must be a real acquire-then-verify, not prose.
    assert "max_attempts" in prompt


# 2. the planner keeps the structure


def test_generate_score_emits_and_keeps_a_recovery_branch():
    planner = _planner(RECOVERY_PLAN_YAML)
    score = planner.generate_score(
        "pick up the stuffed animal and place in the bowl",
        keypoint_labels=KEYPOINTS,
    )
    assert planner._client.models.calls, "fake client was never called"
    assert bt_grammar.has_recovery(score["tree"]), "recovery structure was stripped"
    kinds = bt_grammar.recovery_node_types(score["tree"])
    assert "retry" in kinds and "fallback" in kinds


def test_validate_score_accepts_the_recovery_grammar():
    planner = _planner(RECOVERY_PLAN_YAML)
    score = planner.parse_response(RECOVERY_PLAN_YAML, KEYPOINTS)
    issues = planner.validate_score(score, KEYPOINTS)
    assert issues == [], f"validator rejected the recovery grammar: {issues}"


# 3. recovery is bounded


def test_retry_attempts_are_clamped_at_emit_time():
    runaway = RECOVERY_PLAN_YAML.replace("max_attempts: 2", "max_attempts: 99")
    planner = _planner(runaway)
    score = planner.parse_response(runaway, KEYPOINTS)
    for node in bt_grammar.walk(score["tree"]):
        if bt_grammar.is_retry(node):
            got = (node.get("params") or {}).get("max_attempts")
            assert got == bt_grammar.MAX_RETRY_ATTEMPTS, (
                f"unbounded retry survived sanitisation: max_attempts={got}"
            )


def test_selector_branches_are_capped():
    branches = "\n".join(
        f"        - type: grasp\n          params: {{force: {40 + i}}}"
        for i in range(bt_grammar.MAX_SELECTOR_BRANCHES + 3)
    )
    text = (
        "task: t\ntree:\n  type: sequence\n  children:\n"
        "    - type: fallback\n      children:\n" + branches + "\n"
    )
    planner = _planner(text)
    score = planner.parse_response(text)
    node = score["tree"]["children"][0]
    assert len(node["children"]) == bt_grammar.MAX_SELECTOR_BRANCHES


def test_global_recovery_budget_stops_a_pathological_tree():
    """Nesting must not multiply out: retry(N) inside fallback(M) is capped."""
    exe = _make_executor(fail_actions={"grasp": set(range(1, 100))})
    score = {
        "tree": {
            "type": "fallback",
            "children": [
                {
                    "type": "retry",
                    "params": {"max_attempts": bt_grammar.MAX_RETRY_ATTEMPTS},
                    "children": [{"type": "grasp", "params": {"force": 50}}],
                }
                for _ in range(bt_grammar.MAX_SELECTOR_BRANCHES)
            ],
        }
    }
    exe.execute_score(score)
    assert exe.dispatch_counts["grasp"] <= bt_grammar.MAX_RECOVERY_ATTEMPTS_PER_SCORE + 1


# 4. the tree executes


def test_recovery_tree_runs_after_a_forced_acquire_failure():
    """Force the FIRST grasp to fail; the retry branch must re-run it and win."""
    exe = _make_executor(
        detections={
            "stuffed animal": {"position_3d": [0.45, 0.10, 0.05]},
            "blue bowl": {"position_3d": [0.45, -0.15, 0.05]},
        },
        fail_actions={"grasp": {1}},
    )
    score = yaml.safe_load(RECOVERY_PLAN_YAML)
    results = exe.execute_score(score)

    assert exe.dispatch_counts.get("grasp") == 2, (
        "the retry branch never re-ran the acquire: "
        f"{exe.dispatch_counts}"
    )
    # Attempt 1 stops AT the failed grasp, so the condition only runs once --
    # on the attempt that got far enough to have something to verify.
    assert exe.dispatch_counts.get("verify_grasp") == 1
    # Place branch succeeded first time, so its recovery branch stayed asleep.
    assert exe.dispatch_counts.get("retract_retry", 0) == 0
    assert exe.dispatch_counts.get("release") == 1

    retry_results = [r for r in results if r.action_type == "retry"]
    assert retry_results and retry_results[0].success, "retry node reported failure"
    assert all(r.success for r in results if r.action_type in ("retry", "selector"))


def test_place_recovery_branch_runs_when_verification_fails():
    exe = _make_executor(
        detections={
            "stuffed animal": {"position_3d": [0.45, 0.10, 0.05]},
            "blue bowl": {"position_3d": [0.45, -0.15, 0.05]},
        },
        fail_actions={"verify_placed": {1}},
    )
    score = yaml.safe_load(RECOVERY_PLAN_YAML)
    results = exe.execute_score(score)

    assert exe.dispatch_counts.get("retract_retry") == 1, (
        "the place fallback never fired its recovery branch: "
        f"{exe.dispatch_counts}"
    )
    assert exe.dispatch_counts.get("release") == 2
    selectors = [r for r in results if r.action_type == "selector"]
    assert selectors and selectors[0].success


# 5. backward compatibility


def test_cached_flat_bt_still_validates_hashes_and_runs():
    planner = _planner("")
    assert planner.validate_score(CACHED_FLAT_BT, KEYPOINTS) == []
    assert (
        _hash_entry(CACHED_FLAT_BT["task"], CACHED_FLAT_BT) == CACHED_FLAT_BT_HASH
    ), "grammar change shifted the cache key of a stored tree"

    exe = _make_executor(
        detections={
            "stuffed animal": {"position_3d": [0.45, 0.10, 0.05]},
            "blue bowl": {"position_3d": [0.45, -0.15, 0.05]},
        }
    )
    flat = exe._flatten_tree(CACHED_FLAT_BT["tree"])
    assert [a["type"] for a in flat] == [
        "move_to_keypoint",
        "grasp",
        "move_relative",
        "move_to_keypoint",
        "release",
    ], "a flat cached tree no longer flattens to the same action list"

    results = exe.execute_score(CACHED_FLAT_BT)
    assert all(r.success for r in results)
    assert exe.dispatch_counts == {
        "move_to_keypoint": 2,
        "grasp": 1,
        "move_relative": 1,
        "release": 1,
    }


@pytest.mark.skipif(
    not LIVE_BT_LIBRARY.is_dir(), reason="no local output/bt_library to sweep"
)
def test_every_locally_cached_bt_still_validates():
    planner = _planner("")
    bad = []
    for path in sorted(LIVE_BT_LIBRARY.glob("*.json")):
        entry = json.loads(path.read_text())
        if not isinstance(entry, dict):
            continue  # sidecar state files live in the same directory
        score = entry.get("score")
        if not isinstance(score, dict) or "tree" not in score:
            continue
        issues = planner.validate_score(score, entry.get("objects") or None)
        # Label issues depend on the objects list stored with the entry and
        # are not what this test is about; only grammar issues count.
        issues = [i for i in issues if "unknown type" in i or "max_attempts" in i]
        if issues:
            bad.append((path.name, issues))
        # The stored hash of a few old entries predates unrelated changes, so
        # compare against what SANITISATION does rather than against the file:
        # the guarantee under test is that the grammar change cannot move a
        # stored tree's key, not that every historical file is self-consistent.
        instruction = entry.get("instruction", "")
        clean, _ = sanitize_score(score, entry.get("objects") or None)
        if _hash_entry(instruction, clean) != _hash_entry(instruction, score):
            bad.append((path.name, ["sanitisation moved the hash"]))
    assert not bad, f"cached BTs broken by the grammar change: {bad}"

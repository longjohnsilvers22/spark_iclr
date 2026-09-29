"""Offline tests for the post-task verification retry policy.

No robot, no camera, no SAM3, no Gemini: a fake pipeline and a robot stub that
records EVERY call it receives. The real VerifyMixin drives the run, so the
motion asserted here is the motion that would be commanded.

The rules under test (control/executor_verify.py):
  * `unverified` is an abstain and must never move the robot.
  * `verification.enabled: false` is a kill switch.
  * a replay needs an explicit `fail` AND `verification.retry_on_fail: true`.
"""

import numpy as np
import pytest

from spark_real.control import executor_verify
from spark_real.control.executor_types import ExecutionResult
from spark_real.control.executor_verify import VerifyMixin
from spark_real.control.success_predicates import FAIL, PASS, UNVERIFIED, VerifyOutcome

# Anything on this list moving the robot after an abstain is the safety blocker.
MOTION_METHODS = ("open_gripper", "close_gripper", "moveL", "moveJ", "movej", "speedl")


class RecordingRobot:
    """Records every attribute the executor touches; nothing is a silent no-op."""

    def __init__(self):
        self.calls = []

    def __getattr__(self, name):
        def _record(*args, **kwargs):
            self.calls.append(name)
            return None

        self.calls.append(f"attr:{name}")
        return _record


class _Recorder:
    def set_action_label(self, label):
        pass


class _Profile:
    def __init__(self, raw):
        self.raw = raw


class _Config:
    def __init__(self, output_dir):
        self.output_dir = str(output_dir)


class FakePipeline:
    def __init__(self, output_dir, verification=None):
        raw = {"trace": {"enabled": False}}
        if verification is not None:
            raw["verification"] = verification
        self.profile = _Profile(raw)
        self.config = _Config(output_dir)


class FakeVerifier:
    """Stands in for SuccessVerifier: hands back canned outcomes in order.

    A canned FAIL carries a failed gate, because every fail the real verifier
    emits from the vision path carries either a failed gate or a camera voting
    fail. A fail with NEITHER is the synthesised `on_missing: fail` case, which
    must not move the robot -- see test_on_missing_fail_does_not_replay.
    """

    outcomes = []
    constructed = 0

    def __init__(self, executor=None, pipeline=None, config=None):
        type(self).constructed += 1

    def verify(self, score, actions):
        status = self.outcomes.pop(0) if self.outcomes else UNVERIFIED
        gates = {"grasp": True, "transport": status != FAIL, "release": True}
        return VerifyOutcome(status=status, gates=gates, reason=f"canned {status}")


class FakeExec(VerifyMixin):
    """Minimal executor: every motion helper records instead of moving."""

    GRASP_ORIENTATION = [2.3038, 2.0802, -0.0048]

    def __init__(self, pipeline, strict=False):
        self.robot = RecordingRobot()
        self._pipeline = pipeline
        self._results = []
        self._abort = False
        self._holding = True
        self._strict_placement_verify = strict
        self._placed_labels = set()
        self._last_pick_label = "fork"
        self._recorder = _Recorder()
        # The replay re-perceives before it moves and refuses to move when the
        # re-detection found nothing, so the stub needs a detection map to
        # refresh. See test_verify_retry_closed_loop.py for the tests that own
        # that behaviour; here it is fixture, not assertion.
        self.detection_map = {
            "fork": {"position_3d": [-0.90, 0.00, -0.25]},
            "tray": {"position_3d": [-0.70, 0.00, -0.24]},
        }
        self.motion = []
        self.dispatched = []

    # motion / perception surface

    def _check_abort(self):
        pass

    def _abort_sleep(self, duration, tick=0.05):
        self.motion.append(("abort_sleep", duration))

    def _get_current_position(self):
        self.motion.append(("get_current_position",))
        return np.array([-0.9, 0.0, 0.10])

    def _move_to(self, pos, orient, **kwargs):
        self.motion.append(("move_to", [float(v) for v in pos]))

    def _dispatch_action(self, action_type, params):
        self.dispatched.append(action_type)
        return ExecutionResult(action_type=action_type, success=True, message="ok")

    def _redetect_single(self, label):
        self.motion.append(("redetect_single", label))
        self.detection_map[label] = {"position_3d": [-0.88, 0.02, -0.25]}

    def _attempt_recovery(self, *args, **kwargs):
        return None


ACTIONS = [
    {"type": "move_to_keypoint", "params": {"keypoint_label": "fork"}},
    {"type": "grasp", "params": {}},
    {"type": "move_to_keypoint", "params": {"keypoint_label": "tray"}},
    {"type": "release", "params": {}},
]
SCORE = {"task": "put the fork in the tray"}


@pytest.fixture(autouse=True)
def _patch_verifier(monkeypatch):
    FakeVerifier.outcomes = []
    FakeVerifier.constructed = 0
    monkeypatch.setattr(executor_verify, "SuccessVerifier", FakeVerifier)


def assert_no_motion(ex):
    """No REPLAY motion. One exception, narrowly allowed: the straight-up
    unocclude lift executor_verify._unocclude_for_verify runs BEFORE looking,
    because the arm parks over the placement it is about to photograph and
    would otherwise hide it (added 2026-08-20). That lift is vertical-only,
    touches no gripper, and dispatches nothing, so the invariant this function
    exists to protect -- a verdict must never re-run the task -- is unchanged.
    Anything lateral, downward, gripper-ward, or re-dispatched still fails.
    """
    moved = [c for c in ex.robot.calls if c.split(":")[-1] in MOTION_METHODS]
    assert moved == [], f"robot was commanded: {moved}"
    assert ex.robot.calls == [], f"robot was touched at all: {ex.robot.calls}"

    here = np.asarray(ex._get_current_position(), dtype=float)
    unexpected = []
    lifts = 0
    for entry in ex.motion:
        if entry[0] in ("get_current_position", "abort_sleep"):
            continue  # reads and waits are not motion
        if entry[0] == "move_to":
            target = np.asarray(entry[1], dtype=float)
            straight_up = (
                np.allclose(target[:2], here[:2], atol=1e-6)
                and target[2] >= here[2] - 1e-6
            )
            if straight_up and lifts == 0:
                lifts += 1
                continue
        unexpected.append(entry)
    assert unexpected == [], f"executor issued motion beyond the lift: {unexpected}"
    assert ex.dispatched == [], f"actions were replayed: {ex.dispatched}"


# --- BLOCKER 1: an abstain must never move -----------------------------------


def test_unverified_issues_zero_motion_even_with_retry_opted_in(tmp_path):
    """`unverified` = no verdict reached. Not even opt-in retry may replay it."""
    pipe = FakePipeline(tmp_path, {"enabled": True, "retry_on_fail": True})
    ex = FakeExec(pipe)
    FakeVerifier.outcomes = [UNVERIFIED, UNVERIFIED, UNVERIFIED]

    ex._run_post_task_verification(SCORE, ACTIONS)

    assert_no_motion(ex)
    assert ex.verify_outcome.status == UNVERIFIED
    assert FakeVerifier.constructed == 1  # verified once, never re-verified


def test_unverified_issues_zero_motion_with_default_config(tmp_path):
    pipe = FakePipeline(tmp_path)  # no verification block at all
    ex = FakeExec(pipe)
    FakeVerifier.outcomes = [UNVERIFIED]

    ex._run_post_task_verification(SCORE, ACTIONS)

    assert_no_motion(ex)
    assert ex.verify_outcome.status == UNVERIFIED


def test_unverified_in_strict_mode_issues_zero_motion(tmp_path):
    pipe = FakePipeline(tmp_path, {"enabled": True, "retry_on_fail": True})
    ex = FakeExec(pipe, strict=True)
    ex._placed_labels.add("fork")
    FakeVerifier.outcomes = [UNVERIFIED]

    ex._run_post_task_verification(SCORE, ACTIONS)

    assert_no_motion(ex)
    # The placed-label bookkeeping is retry scaffolding, so it must not fire.
    assert ex._placed_labels == {"fork"}


# --- BLOCKER 1: enabled:false is a real kill switch --------------------------


def test_disabled_verification_is_a_kill_switch(tmp_path):
    """c17c73c parity: no verifier, no capture, no retry, no motion."""
    pipe = FakePipeline(tmp_path, {"enabled": False, "retry_on_fail": True})
    ex = FakeExec(pipe)
    FakeVerifier.outcomes = [FAIL, FAIL, FAIL]  # would replay if it ran

    ex._run_post_task_verification(SCORE, ACTIONS)

    assert_no_motion(ex)
    assert FakeVerifier.constructed == 0, "verifier ran despite the kill switch"
    assert ex._results == [], "a verify row was appended with verification off"
    assert ex.verify_outcome.status == UNVERIFIED
    assert "disabled" in ex.verify_outcome.reason


def test_disabled_accepts_the_yaml_string_form(tmp_path):
    pipe = FakePipeline(tmp_path, {"enabled": "false"})
    ex = FakeExec(pipe)
    FakeVerifier.outcomes = [FAIL]

    ex._run_post_task_verification(SCORE, ACTIONS)

    assert_no_motion(ex)
    assert FakeVerifier.constructed == 0


# --- retry policy ------------------------------------------------------------


def test_fail_does_not_replay_by_default(tmp_path):
    """Retry is opt-in: an explicit fail alone must not move the robot."""
    pipe = FakePipeline(tmp_path, {"enabled": True})
    ex = FakeExec(pipe)
    FakeVerifier.outcomes = [FAIL, FAIL, FAIL]

    ex._run_post_task_verification(SCORE, ACTIONS)

    assert_no_motion(ex)
    assert ex.verify_outcome.status == FAIL


def test_fail_replays_only_when_opted_in(tmp_path):
    pipe = FakePipeline(tmp_path, {"enabled": True, "retry_on_fail": True})
    ex = FakeExec(pipe)
    FakeVerifier.outcomes = [FAIL, PASS]

    ex._run_post_task_verification(SCORE, ACTIONS)

    assert "open_gripper" in ex.robot.calls
    assert ex.dispatched == ["move_to_keypoint", "grasp", "move_to_keypoint", "release"]
    assert ex.verify_outcome.status == PASS


def test_replay_stops_when_the_verdict_becomes_an_abstain(tmp_path):
    """A fail then an abstain: the abstain ends it, no second replay."""
    pipe = FakePipeline(
        tmp_path, {"enabled": True, "retry_on_fail": True, "retry_max_attempts": 2}
    )
    ex = FakeExec(pipe)
    FakeVerifier.outcomes = [FAIL, UNVERIFIED, FAIL]

    ex._run_post_task_verification(SCORE, ACTIONS)

    assert ex.robot.calls.count("open_gripper") == 1
    assert ex.verify_outcome.status == UNVERIFIED


def test_on_missing_fail_does_not_replay(tmp_path, monkeypatch):
    """A fail with no failing gate and no failing vote is not evidence.

    `verification.on_missing: fail` re-labels "no predicate derivable" -- a
    structural abstain, no capture, no vote -- as a fail. Replaying off that
    opens the gripper on the strength of nothing, which is the same bug the
    abstain rule exists to stop.
    """
    pipe = FakePipeline(
        tmp_path,
        {"enabled": True, "retry_on_fail": True, "on_missing": FAIL},
    )
    ex = FakeExec(pipe)

    class _NoPredicate(FakeVerifier):
        def verify(self, score, actions):
            # Exactly what SuccessVerifier emits when no spec is derivable.
            return VerifyOutcome(
                status=FAIL,
                gates={"grasp": True, "transport": True, "release": True},
                reason="no predicate available",
            )

    monkeypatch.setattr(executor_verify, "SuccessVerifier", _NoPredicate)
    ex._run_post_task_verification(SCORE, ACTIONS)

    assert_no_motion(ex)
    assert ex.verify_outcome.status == FAIL  # still reported, just not acted on


def test_retry_max_attempts_zero_disables_the_replay(tmp_path):
    pipe = FakePipeline(
        tmp_path, {"enabled": True, "retry_on_fail": True, "retry_max_attempts": 0}
    )
    ex = FakeExec(pipe)
    FakeVerifier.outcomes = [FAIL]

    ex._run_post_task_verification(SCORE, ACTIONS)

    assert_no_motion(ex)


def test_pass_never_replays(tmp_path):
    pipe = FakePipeline(tmp_path, {"enabled": True, "retry_on_fail": True})
    ex = FakeExec(pipe)
    FakeVerifier.outcomes = [PASS]

    ex._run_post_task_verification(SCORE, ACTIONS)

    assert_no_motion(ex)
    assert ex.verify_outcome.status == PASS


def test_every_verify_attempt_keeps_its_own_trace_file(tmp_path):
    """Retries and reruns must not erase each other's evidence.

    TraceWriter is rebuilt per call (from_pipeline) and output_dir has no
    per-run component, so a counter starting at 0 wrote 001_verify.json every
    time -- the transport-drop evidence from attempt 1, and the whole of the
    previous run, were unrecoverable.
    """
    import json

    from spark_real.control.primitive_trace import TraceWriter

    pipe = FakePipeline(tmp_path)
    pipe.profile.raw["trace"] = {"enabled": True}
    reasons = ["attempt 1 transport drop", "attempt 2 clean", "run 2 grasp fail"]
    for reason in reasons:
        TraceWriter.from_pipeline(pipe).write_verify(
            VerifyOutcome(status=FAIL, reason=reason), 0.1
        )

    trace = tmp_path / "trace"
    files = sorted(p.name for p in trace.glob("*.json"))
    assert files == ["001_verify.json", "002_verify.json", "003_verify.json"]
    got = [json.loads((trace / n).read_text())["reason"] for n in files]
    assert got == reasons

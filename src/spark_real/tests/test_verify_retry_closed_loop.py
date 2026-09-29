"""Closed-loop retry: a verified failure must be CORRECTED, not just reported.

Reconstructs the hardware failure of 2026-08-04 (plushie -> blue bowl):

    Verify [birdview] inside(stuffed animal, blue bowl) -> fail (3d, conf=0.91)
                      3d xy_ok=False dz=+5.1cm
    Verify [birdview] held(stuffed animal, False)       -> pass
    Verify [sideview] inside(...) -> abstain ('blue bowl' unbound)
    Verification fail; retry is off (verification.retry_on_fail), no replay

The plushie was on the table next to the bowl, the verifier said so at 0.91,
and the run ended. Three things have to be true for that to become a
correction rather than a report:

  1. the shipped profile opts the mechanism in, BOUNDED;
  2. the replay RE-PERCEIVES the pick target before it moves -- the object is
     somewhere new, that is the whole reason the place failed. A replay against
     the stale plan-time detection map repeats the same miss;
  3. the replay only ever starts from an established state: gripper open,
     confirmed empty, arm lifted clear.

And two things must stay true:

  4. an ``unverified`` / abstain NEVER moves the arm (retrying an unobserved
     outcome can undo a success);
  5. the attempt count is capped -- this runs on a real UR10e.

No robot, no camera, no SAM3, no Gemini. The real VerifyMixin drives, so the
motion asserted here is the motion that would be commanded. Verdicts come from
the real ``fuse_votes`` over real ``CameraVote``s, so the fail under test is the
fail the fusion actually produces from those per-camera votes.
"""

from pathlib import Path

import numpy as np
import pytest
import yaml

from spark_real.control import executor_verify
from spark_real.control.executor_types import ExecutionResult
from spark_real.control.executor_verify import VerifyMixin
from spark_real.control.success_predicates import (
    ABSTAIN,
    FAIL,
    PASS,
    UNVERIFIED,
    CameraVote,
    VerifyOutcome,
)
from spark_real.control.success_verifier import fuse_votes

SHIPPED_YAML = (
    Path(__file__).resolve().parents[1] / "configs" / "ur10e_default.yaml"
)

OBJ = "stuffed animal"
BOWL = "blue bowl"

MOTION_METHODS = ("open_gripper", "close_gripper", "moveL", "moveJ", "movej", "speedl")


def shipped_verification_block() -> dict:
    raw = yaml.safe_load(SHIPPED_YAML.read_text())
    block = (raw or {}).get("verification")
    assert isinstance(block, dict), "ur10e_default.yaml has no verification block"
    return block


# --- the measured per-camera votes -------------------------------------------


def birdview_fail() -> CameraVote:
    """Birdview, 3D, conf 0.91: the plushie is NOT in the bowl."""
    return CameraVote(
        camera="birdview",
        predicate=f"inside({OBJ}, {BOWL})",
        vote=FAIL,
        mode="3d",
        confidence=0.91,
        detail="3d xy_ok=False dz=+5.1cm",
    )


def birdview_held_pass() -> CameraVote:
    return CameraVote(
        camera="birdview",
        predicate=f"held({OBJ}, False)",
        vote=PASS,
        mode="3d",
        confidence=0.91,
        detail="gripper empty",
    )


def sideview_abstain() -> CameraVote:
    return CameraVote(
        camera="sideview",
        predicate=f"inside({OBJ}, {BOWL})",
        vote=ABSTAIN,
        mode="none",
        confidence=0.0,
        detail=f"'{BOWL}' unbound",
    )


THE_FAILURE = (birdview_fail(), birdview_held_pass(), sideview_abstain())
THE_ABSTAIN = (sideview_abstain(),)


def outcome_from(votes) -> VerifyOutcome:
    """Fuse real votes exactly as SuccessVerifier.verify would."""
    status, reason = fuse_votes(list(votes))
    return VerifyOutcome(
        status=status,
        predicates=[f"inside({OBJ}, {BOWL})"],
        votes=list(votes),
        gates={"grasp": True, "transport": True, "release": True},
        reason=reason,
        depth_source="hardware",
    )


def test_the_reconstructed_failure_really_is_a_fail():
    """Guard the premise: these votes fuse to `fail`, the abstain alone does not."""
    assert outcome_from(THE_FAILURE).status == FAIL
    assert outcome_from(THE_ABSTAIN).status == UNVERIFIED


# --- harness -----------------------------------------------------------------


class RecordingRobot:
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


class ScriptedVerifier:
    """Hands back canned VerifyOutcomes in order; counts constructions."""

    outcomes = []
    constructed = 0

    def __init__(self, executor=None, pipeline=None, config=None):
        type(self).constructed += 1

    def verify(self, score, actions):
        if type(self).outcomes:
            votes = type(self).outcomes.pop(0)
        else:
            votes = THE_FAILURE  # never runs dry: proves the CAP stops the loop
        return outcome_from(votes)


class FakeExec(VerifyMixin):
    """Post-release executor state, exactly as _run_actions leaves it.

    After a SUCCESSFUL release ``_run_actions`` adds the pick label to
    ``_placed_labels`` and clears ``_last_pick_label``. That pairing is what
    made the old replay path blind: the strict-mode un-place guard keys off
    ``_last_pick_label`` (now "") and the in-loop re-detect is skipped for any
    label in ``_placed_labels``.
    """

    GRASP_ORIENTATION = [2.3038, 2.0802, -0.0048]
    TABLE_Z_FLOOR = -0.276

    def __init__(self, pipeline, strict=False, redetect_finds=True, still_holding=False):
        self.robot = RecordingRobot()
        self._pipeline = pipeline
        self._results = []
        self._abort = False
        self._holding = False          # the release succeeded; jaws are open
        self._strict_placement_verify = strict
        self._placed_labels = {OBJ}    # ...and the plushie was marked placed
        self._last_pick_label = ""     # ...and the pick label was cleared
        self._recorder = _Recorder()
        self.detection_map = {
            OBJ: {"position_3d": [-0.90, 0.05, -0.25]},   # PLAN-TIME (stale)
            BOWL: {"position_3d": [-0.70, 0.05, -0.24]},
        }
        self.motion = []
        self.dispatched = []
        self.log = []                  # ordered (kind, payload) trace
        self.placed_at_dispatch = []   # (action_type, _placed_labels snapshot)
        self._redetect_finds = redetect_finds
        self._still_holding = still_holding
        self.grasp_verify_calls = 0

    def _check_abort(self):
        pass

    def _abort_sleep(self, duration, tick=0.05):
        self.motion.append(("abort_sleep", duration))

    def _get_current_position(self):
        self.motion.append(("get_current_position",))
        return np.array([-0.70, 0.05, -0.20])

    def _move_to(self, pos, orient, **kwargs):
        p = [float(v) for v in pos]
        self.motion.append(("move_to", p))
        self.log.append(("move_to", p))

    def _dispatch_action(self, action_type, params):
        self.dispatched.append(action_type)
        self.log.append(("dispatch", action_type))
        self.placed_at_dispatch.append((action_type, set(self._placed_labels)))
        return ExecutionResult(action_type=action_type, success=True, message="ok")

    def _redetect_single(self, label):
        self.motion.append(("redetect_single", label))
        self.log.append(("redetect", label))
        if not self._redetect_finds:
            return  # SAM3 saw nothing: the map keeps its stale entry, untouched
        # A real re-detection writes a NEW entry (execution_recovery.redetect_single).
        self.detection_map[label] = {"position_3d": [-0.86, 0.09, -0.25]}

    def _verify_grasp(self):
        self.grasp_verify_calls += 1
        return self._still_holding

    def _attempt_recovery(self, *args, **kwargs):
        return None


ACTIONS = [
    {"type": "move_to_keypoint", "params": {"keypoint_label": OBJ}},
    {"type": "grasp", "params": {"force": 100}},
    {"type": "move_to_keypoint", "params": {"keypoint_label": BOWL, "offset_z": 0.05}},
    {"type": "release", "params": {}},
]
SCORE = {"task": "put the stuffed animal in the blue bowl"}


@pytest.fixture(autouse=True)
def _patch_verifier(monkeypatch):
    ScriptedVerifier.outcomes = []
    ScriptedVerifier.constructed = 0
    monkeypatch.setattr(executor_verify, "SuccessVerifier", ScriptedVerifier)


def enabled_block(**over):
    block = {"enabled": True, "retry_on_fail": True}
    block.update(over)
    return block


# One definition of the invariant, shared with test_verify_retry_policy: no
# verdict may replay the task, with the pre-look unocclude lift narrowly
# allowed (see that module's docstring for why). Imported rather than copied
# so the two files cannot drift apart on what "no motion" means.
from test_verify_retry_policy import assert_no_motion  # noqa: E402


def replay_count(ex):
    return ex.robot.calls.count("open_gripper")


# --- 1. the shipped profile opts in, bounded ---------------------------------


def test_shipped_yaml_turns_the_loop_on_with_a_bounded_cap():
    """The mechanism existed; the shipped profile never enabled it."""
    block = shipped_verification_block()
    assert block.get("enabled") is True
    assert block.get("retry_on_fail") is True, (
        "verification.retry_on_fail is absent/false in the shipped profile: "
        "the verifier is a reporter, a caught failure changes nothing"
    )
    cap = block.get("retry_max_attempts")
    assert isinstance(cap, int) and not isinstance(cap, bool), (
        "retry_max_attempts must be an explicit int in the shipped profile -- "
        "an unpinned cap is an unbounded loop on a real arm"
    )
    assert 1 <= cap <= 3, f"cap {cap} is not a sane bound for a real UR10e"


def test_shipped_yaml_is_what_the_mixin_reads(tmp_path):
    """No drift between the YAML keys and _verify_settings' key names."""
    block = shipped_verification_block()
    ex = FakeExec(FakePipeline(tmp_path, block))
    enabled, retry, cap = ex._verify_settings()
    assert (enabled, retry) == (True, True)
    assert cap == block["retry_max_attempts"]


# --- 2. the retry must RE-PERCEIVE -------------------------------------------


def test_the_dropped_plushie_triggers_a_retry_that_reperceives_first(tmp_path):
    """THE failure: birdview fails at 0.91, sideview abstains -> retry.

    And the retry must ask the cameras where the plushie is NOW before it
    drives anywhere. The object is 5.1cm off its intended pose; the plan-time
    entry in detection_map is where it was BEFORE the pick, which is not where
    it is now.
    """
    ex = FakeExec(FakePipeline(tmp_path, enabled_block(retry_max_attempts=1)))
    ScriptedVerifier.outcomes = [THE_FAILURE, (birdview_held_pass(),)]
    stale = ex.detection_map[OBJ]["position_3d"]

    ex._run_post_task_verification(SCORE, ACTIONS)

    assert replay_count(ex) == 1, "a verified failure did not trigger a replay"
    assert ex.dispatched == ["move_to_keypoint", "grasp", "move_to_keypoint", "release"]

    kinds = [k for k, _ in ex.log]
    assert "redetect" in kinds, "the replay never re-perceived; it replayed a stale map"
    redetected = [p for k, p in ex.log if k == "redetect"]
    assert OBJ in redetected, f"the pick target was not re-detected: {redetected}"

    first_redetect = kinds.index("redetect")
    first_dispatch = kinds.index("dispatch")
    assert first_redetect < first_dispatch, (
        f"re-perception happened after the replay started moving: {ex.log}"
    )
    assert ex.detection_map[OBJ]["position_3d"] != stale, "map is still the plan-time pose"


def test_the_placed_label_is_cleared_so_the_repick_is_not_skipped(tmp_path):
    """`_placed_labels` gates BOTH the re-detect and the pick approach itself.

    executor_motion._approach_target short-circuits move_to_keypoint with
    "already in container, skipping pick approach" for any label in
    `_placed_labels`. A replay that leaves the label there re-runs a tree whose
    pick is a no-op and whose grasp closes on air.
    """
    ex = FakeExec(FakePipeline(tmp_path, enabled_block(retry_max_attempts=1)))
    ScriptedVerifier.outcomes = [THE_FAILURE, (birdview_held_pass(),)]

    ex._run_post_task_verification(SCORE, ACTIONS)

    # The label must be gone from _placed_labels at the moment the replayed
    # pick approach is dispatched -- not merely at the end of the run.
    at_pick = [placed for atype, placed in ex.placed_at_dispatch if atype == "grasp"]
    assert at_pick, "the replay never reached the grasp"
    assert OBJ not in at_pick[0], (
        f"'{OBJ}' was still marked placed when the pick replayed: {at_pick[0]}"
    )
    # ...and the un-placing must not have depended on _last_pick_label, which
    # the successful release already cleared to "".
    assert ex._last_pick_label == ""


def test_strict_mode_also_reperceives(tmp_path):
    """Strict placement verify must not change the re-perception guarantee."""
    ex = FakeExec(FakePipeline(tmp_path, enabled_block()), strict=True)
    ScriptedVerifier.outcomes = [THE_FAILURE, (birdview_held_pass(),)]

    ex._run_post_task_verification(SCORE, ACTIONS)

    assert OBJ in [p for k, p in ex.log if k == "redetect"]


def test_no_replay_when_the_object_cannot_be_reperceived(tmp_path):
    """Re-perception that finds nothing leaves a stale map -- do not move.

    Driving to the plan-time pose "because we have nothing better" is exactly
    the miss that failed. Fail closed instead.
    """
    ex = FakeExec(
        FakePipeline(tmp_path, enabled_block(retry_max_attempts=2)),
        redetect_finds=False,
    )
    ScriptedVerifier.outcomes = [THE_FAILURE]

    ex._run_post_task_verification(SCORE, ACTIONS)

    assert ex.dispatched == [], f"replayed against a stale map: {ex.dispatched}"
    assert ex.verify_outcome.status == FAIL  # still reported


# --- 3. the retry must be SAFE from the current state ------------------------


def test_retry_establishes_an_empty_gripper_before_it_transits(tmp_path):
    """The tree starts with approach+grasp: the jaws must be open and empty."""
    ex = FakeExec(FakePipeline(tmp_path, enabled_block(retry_max_attempts=1)))
    ScriptedVerifier.outcomes = [THE_FAILURE, (birdview_held_pass(),)]

    ex._run_post_task_verification(SCORE, ACTIONS)

    assert "open_gripper" in ex.robot.calls
    assert ex._holding is False
    lifts = [p for k, p in ex.log if k == "move_to"]
    assert lifts, "no clearance lift before the replay"
    assert lifts[0][2] > -0.20, f"replay started without lifting clear: {lifts[0]}"
    kinds = [k for k, _ in ex.log]
    assert kinds.index("move_to") < kinds.index("dispatch"), (
        f"the arm transited before it was lifted clear: {ex.log}"
    )


def test_no_replay_while_the_object_is_still_in_the_jaws(tmp_path):
    """If the gripper will not let go, a re-approach drags the object through it."""
    ex = FakeExec(
        FakePipeline(tmp_path, enabled_block(retry_max_attempts=2)),
        still_holding=True,
    )
    ex._holding = True
    ScriptedVerifier.outcomes = [THE_FAILURE]

    ex._run_post_task_verification(SCORE, ACTIONS)

    assert ex.dispatched == [], "replayed a pick tree while still holding"
    assert ex.grasp_verify_calls > 0, "the grip state was never established"


# --- 4. an abstain must never move -------------------------------------------


def test_sideview_abstain_alone_never_retries(tmp_path):
    """`unverified` = no verdict. Retrying could undo an unobserved success."""
    ex = FakeExec(FakePipeline(tmp_path, enabled_block(retry_max_attempts=3)))
    ScriptedVerifier.outcomes = [THE_ABSTAIN, THE_FAILURE, THE_FAILURE]

    ex._run_post_task_verification(SCORE, ACTIONS)

    assert_no_motion(ex)
    assert ex.verify_outcome.status == UNVERIFIED
    assert ScriptedVerifier.constructed == 1
    assert ex._placed_labels == {OBJ}, "retry bookkeeping fired on an abstain"


def test_an_abstain_mid_loop_ends_the_loop(tmp_path):
    """fail -> replay -> abstain: stop. The second replay would be blind."""
    ex = FakeExec(FakePipeline(tmp_path, enabled_block(retry_max_attempts=3)))
    ScriptedVerifier.outcomes = [THE_FAILURE, THE_ABSTAIN, THE_FAILURE]

    ex._run_post_task_verification(SCORE, ACTIONS)

    assert replay_count(ex) == 1
    assert ex.verify_outcome.status == UNVERIFIED


# --- 5. the cap ---------------------------------------------------------------


@pytest.mark.parametrize("cap", [1, 2, 3])
def test_a_permanent_failure_stops_at_the_cap(tmp_path, cap):
    """The verifier never stops saying fail; the loop must still terminate."""
    ex = FakeExec(FakePipeline(tmp_path, enabled_block(retry_max_attempts=cap)))
    ScriptedVerifier.outcomes = []  # every verify returns THE_FAILURE

    ex._run_post_task_verification(SCORE, ACTIONS)

    assert replay_count(ex) == cap, f"expected {cap} replays, got {replay_count(ex)}"
    assert ScriptedVerifier.constructed == cap + 1  # initial verify + one per replay
    assert ex.dispatched.count("grasp") == cap


def test_the_shipped_cap_is_what_a_permanent_failure_gets(tmp_path):
    """Bind the loop bound to the SHIPPED profile, not to a test-local number."""
    block = shipped_verification_block()
    ex = FakeExec(FakePipeline(tmp_path, block))
    ScriptedVerifier.outcomes = []

    ex._run_post_task_verification(SCORE, ACTIONS)

    assert replay_count(ex) == block["retry_max_attempts"]


def test_a_successful_retry_stops_immediately(tmp_path):
    ex = FakeExec(FakePipeline(tmp_path, enabled_block(retry_max_attempts=3)))
    ScriptedVerifier.outcomes = [THE_FAILURE, (birdview_held_pass(),)]

    ex._run_post_task_verification(SCORE, ACTIONS)

    assert replay_count(ex) == 1
    assert ex.verify_outcome.status == PASS


def test_a_successful_retry_re_marks_the_object_as_placed(tmp_path):
    """Un-placing is retry scaffolding; a completed retry must restore it.

    pipeline_run's closed loop reads `_placed_labels` to decide what is still
    outstanding. Leaving the label cleared makes the next pass re-pick an
    object that is already in the bowl.
    """
    ex = FakeExec(FakePipeline(tmp_path, enabled_block(retry_max_attempts=1)))
    ScriptedVerifier.outcomes = [THE_FAILURE, (birdview_held_pass(),)]

    ex._run_post_task_verification(SCORE, ACTIONS)

    assert ex.verify_outcome.status == PASS
    assert OBJ in ex._placed_labels


def test_abort_during_the_loop_stops_it(tmp_path):
    ex = FakeExec(FakePipeline(tmp_path, enabled_block(retry_max_attempts=3)))
    ScriptedVerifier.outcomes = []

    real_dispatch = ex._dispatch_action

    def _abort_after_grasp(action_type, params):
        result = real_dispatch(action_type, params)
        if action_type == "grasp":
            ex._abort = True
        return result

    ex._dispatch_action = _abort_after_grasp
    ex._run_post_task_verification(SCORE, ACTIONS)

    assert replay_count(ex) == 1
    assert ex.dispatched == ["move_to_keypoint", "grasp"]

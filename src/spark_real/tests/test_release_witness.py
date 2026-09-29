"""The G3 release gate must fail a release that did not happen -- and only that.

Three consecutive rig runs logged

    Verify gate 'release' FAILED: held_before=True held_after=True

and all three physically succeeded (photographed twice). The cause was not a
short settle or a stale register: ``held_after`` was read with ``_grip_intact``,
which is a CLOSED-jaw drop check. Its position clause is "pos <
GRIPPER_FULLY_CLOSED (250) => holding", and a released 2F-85 reads pos ~0. So it
answered "holding" for every gripper that had successfully opened, and the gate
failed exactly when the release worked.

The fake driver below is the real register contract (0=open .. 255=closed plus
the gObj flag); ``_grip_intact`` and ``_read_gripper_state`` under test are the
real implementations.
"""

from __future__ import annotations

import numpy as np

from spark_real.control import success_verifier
from spark_real.control.score_executor import ScoreExecutor


class FakeGripper:
    """Robotiq-shaped registers. ``open_to`` is where the jaws actually land."""

    GRIPPER_TYPE = "robotiq_2f85"
    robot_family = "ur10e"

    def __init__(self, jaw=180.0, obj=True, open_to=0.0, obj_after=False, stalls=0):
        self.tcp = np.array([-0.75, 0.20, -0.15, 2.3038, 2.0802, -0.0048])
        self.jaw = jaw
        self.obj = obj
        self._open_to = open_to
        self._obj_after = obj_after
        self._stalls = stalls  # samples that still read the pre-open position
        self.reads = 0

    def get_tcp_pose(self):
        return self.tcp.copy()

    def get_gripper_position(self, publish=True):
        self.reads += 1
        if self._opened and self._stalls > 0:
            self._stalls -= 1
            return self._pre_open
        return self.jaw

    def is_object_detected(self, publish=True):
        return self.obj

    def _publish_gripper_state(self, force=False):
        return None

    _opened = False
    _pre_open = 0.0

    def open_gripper(self):
        self._pre_open = self.jaw
        self._opened = True
        self.jaw = self._open_to
        self.obj = self._obj_after


def _executor(robot):
    ex = ScoreExecutor(robot, detection_map={}, velocity=0.2)
    ex.RELEASE_CONFIRM_TIMEOUT_S = 0.3  # keep the failure path quick
    return ex


def _release(ex):
    """The executor_release sequence, minus the arm motion."""
    state = success_verifier.begin_release_witness(ex)
    ex.robot.open_gripper()
    ex._holding = False
    return success_verifier.finish_release_witness(ex, state, "bowl")


def _gate(ex):
    return success_verifier.collect_gates(ex)[0]["release"]


def test_the_old_predicate_really_is_a_false_negative():
    """The structural cause, in one assertion: _grip_intact says HOLDING on a
    gripper that is wide open and empty."""
    ex = _executor(FakeGripper(jaw=180.0, obj=True))
    ex.robot.open_gripper()
    assert ex.robot.jaw == 0.0
    assert ex._grip_intact() is True  # <- what held_after used to read


def test_a_successful_release_passes_the_gate():
    ex = _executor(FakeGripper(jaw=180.0, obj=True, open_to=0.0, obj_after=False))
    witness = _release(ex)

    assert witness.held_before is True
    assert witness.released is True
    assert witness.held_after is False
    assert witness.jaw_pos_before == 180.0
    assert witness.jaw_pos_after == 0.0
    assert _gate(ex) is True


def test_an_object_still_clamped_in_the_jaws_fails_the_gate():
    """The failure the gate exists for: the open command did not free the object,
    so the jaws never travelled and gObj still reports contact."""
    ex = _executor(FakeGripper(jaw=180.0, obj=True, open_to=180.0, obj_after=True))
    witness = _release(ex)

    assert witness.held_before is True
    assert witness.released is False
    assert witness.held_after is True
    assert witness.jaw_pos_after == 180.0
    assert _gate(ex) is False


def test_a_partial_open_that_travelled_far_enough_counts_as_released():
    """Thin object: the jaws stop short of the open stop but move 90 counts,
    which no clamped object survives. The absolute threshold alone would
    false-fail this."""
    ex = _executor(FakeGripper(jaw=240.0, obj=True, open_to=150.0, obj_after=False))
    witness = _release(ex)

    assert witness.released is True
    assert witness.jaw_pos_after > ex.GRIPPER_RELEASED_MAX_POS
    assert _gate(ex) is True


def test_a_partial_open_halted_on_contact_is_not_a_release():
    """Same travel as the test above, but gObj says the jaws were STOPPED by
    something. Travel alone must not outvote that."""
    ex = _executor(FakeGripper(jaw=240.0, obj=True, open_to=150.0, obj_after=True))
    witness = _release(ex)

    assert witness.released is False
    assert witness.obj_after is True
    assert _gate(ex) is False


def test_a_partial_open_that_barely_moved_does_not_count():
    ex = _executor(FakeGripper(jaw=240.0, obj=True, open_to=215.0, obj_after=True))
    witness = _release(ex)

    assert witness.released is False
    assert _gate(ex) is False


def test_an_object_dropped_before_the_release_fails_the_gate():
    """held_before is still the mid-transport drop catch: empty jaws reach the
    fully-closed stop, so nothing was there to release."""
    ex = _executor(FakeGripper(jaw=255.0, obj=False, open_to=0.0, obj_after=False))
    witness = _release(ex)

    assert witness.held_before is False
    assert witness.released is True  # the jaws did open
    assert _gate(ex) is False


def test_the_witness_polls_for_a_slow_open_instead_of_sleeping_longer():
    """Two samples still read the pre-open position; the third shows open."""
    ex = _executor(FakeGripper(jaw=180.0, obj=True, open_to=0.0, stalls=2))
    witness = _release(ex)

    assert witness.released is True
    assert witness.confirm_s > 0.0
    assert ex.robot.reads >= 4  # 1 before the open + 3 confirming


def test_an_unreadable_jaw_register_abstains_rather_than_guessing():
    robot = FakeGripper(jaw=180.0, obj=True)

    def broken(publish=True):
        raise RuntimeError("register 12 not surfaced by this ur_rtde build")

    ex = _executor(robot)
    robot.get_gripper_position = broken
    witness = _release(ex)

    assert witness.released is None
    assert "release" not in success_verifier._gates(ex)
    assert _gate(ex) is True  # abstain reads as "the primitive never reported"


def test_collect_gates_never_re_derives_the_release_verdict():
    """A stored witness must not be re-scored from held_after by a second
    reader; that is where the false negative used to come back."""
    ex = _executor(FakeGripper(jaw=180.0, obj=True, open_to=0.0))
    _release(ex)
    success_verifier._gates(ex).pop("release")

    assert _gate(ex) is True

"""
Port step 3: typed GraspOutcome attached to GraspVerdict via
_record_grasp_verdict, and the SECURED -> SLIP upgrade on a transport
drop. Hardware-free (mixin exercised on a bare host object).
"""

from spark_real.control import success_verifier as sv
from spark_real.control.executor_grasp import GraspMixin, GraspVerdict
from spark_real.control.grasp_outcome import GraspOutcome


class _Host(GraspMixin):
    GRIPPER_FULLY_CLOSED = 250

    def __init__(self):
        self._verify_gates = {}


def _record(**kw):
    host = _Host()
    v = GraspVerdict(**kw)
    host._record_grasp_verdict(v)
    return host, v


def test_gobj_true_stamps_secured():
    _, v = _record(held=True, source="gobj_fast", gobj=True, gripper_pos=147)
    assert v.outcome == GraspOutcome.SECURED.value


def test_jaws_fully_closed_stamps_empty_close():
    _, v = _record(
        held=False, source="jaws_fully_closed", gobj=False, gripper_pos=252
    )
    assert v.outcome == GraspOutcome.EMPTY_CLOSE.value


def test_force_corroborated_stamps_secured():
    _, v = _record(held=True, source="force", gobj=None, gripper_pos=147)
    assert v.outcome == GraspOutcome.SECURED.value


def test_unreadable_registers_stamp_unknown():
    _, v = _record(held=False, source="empty", gobj=None, gripper_pos=None)
    assert v.outcome == GraspOutcome.UNKNOWN.value


def test_producer_supplied_outcome_is_not_overwritten():
    _, v = _record(
        held=True, source="gobj", gobj=True, outcome=GraspOutcome.SLIP.value
    )
    assert v.outcome == GraspOutcome.SLIP.value


def test_transport_drop_upgrades_secured_to_slip():
    host, v = _record(held=True, source="gobj", gobj=True, gripper_pos=147)
    assert v.outcome == GraspOutcome.SECURED.value
    assert host._last_grasp_verdict is v
    sv.note_transport_drop(host, "lift")
    assert v.outcome == GraspOutcome.SLIP.value
    assert host._verify_gates.get("transport") is False


def test_transport_drop_does_not_touch_empty_close():
    host, v = _record(held=False, source="empty", gobj=False, gripper_pos=252)
    sv.note_transport_drop(host, "lift")
    assert v.outcome == GraspOutcome.EMPTY_CLOSE.value

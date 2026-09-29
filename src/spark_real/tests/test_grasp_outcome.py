"""
Endpoint grasp-outcome classifier: synthetic (gObj, pos, force) tuples.
No driver needed (pure module).
"""

import pytest

from spark_real.control.grasp_outcome import (
    GraspOutcome,
    classify_grasp_endpoint,
    slip_after_secured,
)


@pytest.mark.parametrize(
    "gobj,pos,force_holding,expected",
    [
        # gObj True is authoritative (no position veto: the compressed
        # plushie reads pos ~252 while genuinely held).
        (True, 147, None, GraspOutcome.SECURED),
        (True, 252, None, GraspOutcome.SECURED),
        (True, None, None, GraspOutcome.SECURED),
        # Closed stop with no object flag: empty close.
        (False, 255, None, GraspOutcome.EMPTY_CLOSE),
        (False, 250, True, GraspOutcome.EMPTY_CLOSE),  # flat-plate veto
        (None, 255, True, GraspOutcome.EMPTY_CLOSE),
        # Force-corroborated hold, jaws short of the stop.
        (False, 147, True, GraspOutcome.SECURED),
        (None, 147, True, GraspOutcome.SECURED),
        # gObj False, short of stop, no force: empty.
        (False, 147, False, GraspOutcome.EMPTY_CLOSE),
        (False, 147, None, GraspOutcome.EMPTY_CLOSE),
        # Registers unreadable: UNKNOWN (never retries).
        (None, None, None, GraspOutcome.UNKNOWN),
        # gObj unreadable, pos short of stop, no force signal: inconclusive.
        (None, 147, None, GraspOutcome.UNKNOWN),
        (None, 147, False, GraspOutcome.UNKNOWN),
    ],
)
def test_classify_endpoint(gobj, pos, force_holding, expected):
    got = classify_grasp_endpoint(
        gobj, pos, closed_pos=250, force_holding=force_holding
    )
    assert got == expected


def test_slip_transition_only_after_secured():
    assert (
        slip_after_secured(GraspOutcome.SECURED, False) == GraspOutcome.SLIP
    )
    assert slip_after_secured(GraspOutcome.SECURED, True) is None
    assert slip_after_secured(GraspOutcome.SECURED, None) is None
    assert slip_after_secured(GraspOutcome.EMPTY_CLOSE, False) is None
    assert slip_after_secured(None, False) is None


def test_shared_enum_is_the_sim_vocabulary():
    assert GraspOutcome.SECURED.value == "secured"
    assert GraspOutcome.EMPTY_CLOSE.value == "empty_close"
    assert GraspOutcome.SLIP.value == "slip"
    assert GraspOutcome.UNKNOWN.value == "unknown"

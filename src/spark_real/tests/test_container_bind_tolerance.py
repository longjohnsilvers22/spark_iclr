"""A container that RECEIVED an object must still bind to itself.

Measured on the plushie-into-bowl run (episode_0002, 12:52), which physically
succeeded and was scored a failure:

    Verify [sideview] on(plushie, bowl) -> abstain
        'bowl' unbound: nearest 'bowl' is 7.3cm from its plan-time position

The bowl did not move. The plushie landed in it, so SAM3's bowl mask went from
fill=0.75 solid=1.00 to fill=0.37 solid=0.50 (sideview) and fill=0.22
solid=0.26 cc=0.60 (birdview) -- the interior is occluded by the very object
whose placement we are verifying, and the remaining rim mask has a centroid
7.3cm off. _bind_static's flat 6cm tolerance is calibrated for "an object the
tree never touched", and a container is not that: a successful place is
exactly what deforms its mask.

Fix: scale the tolerance by the container's own plan-time footprint. A bowl's
centroid cannot shift further than its own radius while it sits still, and a
distinct bowl elsewhere on the table is still rejected.

This must NOT become a blanket loosening -- the gate exists to catch silverware
dropped BESIDE the tray. Objects with no measured extent keep the flat 6cm.
"""

import numpy as np
import pytest

from spark_real.control.success_verifier import STATIC_MATCH_TOL_M, IdentityBinder


class Det:
    """The ObjectDetection surface the binder and predicates actually read."""

    def __init__(self, xyz, label="bowl", minor=None, ar=1.0, conf=0.95):
        self.label = label
        self.position_3d = np.asarray(xyz, dtype=float)
        self.confidence = conf
        self.aspect_ratio = ar
        if minor is not None:
            self.obb_minor_m = minor  # FULL short-axis length, metres


BOWL_XYZ = (-0.922, -0.023, -0.227)
BOWL_MINOR = 0.20  # a ~20cm bowl -> ~10cm radius


def _plan(minor=BOWL_MINOR):
    return {"bowl": Det(BOWL_XYZ, minor=minor)}


def _shift(xyz, dx):
    return (xyz[0] + dx, xyz[1], xyz[2])


def test_bowl_that_received_the_object_still_binds():
    """The regression: 7.3cm of mask-driven centroid shift, bowl stationary."""
    moved = _shift(BOWL_XYZ, 0.073)
    binder = IdentityBinder(_plan(), [Det(moved, minor=BOWL_MINOR)])
    assert binder.lookup("bowl") is not None, binder.explain("bowl")


def test_a_different_bowl_across_the_table_is_still_rejected():
    """The gate must keep catching a genuinely different instance."""
    far = _shift(BOWL_XYZ, 0.60)
    binder = IdentityBinder(_plan(), [Det(far, minor=BOWL_MINOR)])
    assert binder.lookup("bowl") is None, "bound to a bowl 60cm away"


def test_two_bowls_inside_tolerance_stay_ambiguous():
    """The ambiguity guard must survive the widened tolerance."""
    binder = IdentityBinder(
        _plan(),
        [
            Det(_shift(BOWL_XYZ, -0.01), minor=BOWL_MINOR),
            Det(_shift(BOWL_XYZ, 0.01), minor=BOWL_MINOR),
        ],
    )
    assert binder.lookup("bowl") is None, binder.explain("bowl")


def test_object_with_no_measured_extent_keeps_the_flat_tolerance():
    """No OBB -> no widening. A knife 7.3cm off its anchor stays unbound."""
    plan = {"knife": Det((-0.8, 0.26, -0.11), label="knife")}
    moved = _shift((-0.8, 0.26, -0.11), 0.073)
    binder = IdentityBinder(plan, [Det(moved, label="knife")])
    assert binder.lookup("knife") is None, (
        "widened tolerance leaked to an object with no measured footprint: "
        + binder.explain("knife")
    )


def test_tolerance_never_shrinks_below_the_flat_value():
    """A tiny container must not end up with a tolerance tighter than 6cm."""
    tiny = 0.01
    moved = _shift(BOWL_XYZ, STATIC_MATCH_TOL_M * 0.5)
    binder = IdentityBinder(_plan(minor=tiny), [Det(moved, minor=tiny)])
    assert binder.lookup("bowl") is not None, (
        f"tolerance shrank below the {STATIC_MATCH_TOL_M}m floor: "
        + binder.explain("bowl")
    )

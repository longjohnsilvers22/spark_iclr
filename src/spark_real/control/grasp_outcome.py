"""
Typed grasp outcome for the real (Robotiq 2F-85) pipeline.

Shares its enum with the sim (``spark_bench.libero_pro.telemetry`` imports
it from here) so trial metadata carries the same outcome vocabulary on both
embodiments. The classifier is ENDPOINT-based: the 2F-85 cannot stream an
aperture trace during a close (each register publish is a URScript upload,
which cancels the running ``rq_close_and_wait`` program; see
``ur10e_driver._publish_gripper_state``). Inputs are gObj (output register
13, "jaws stopped on an object"), the jaw position register, and the
executor's TCP-force corroboration.

* ``SECURED``     : jaws stopped on an object (gObj True, or TCP-force
                    corroboration with jaws short of the closed stop).
* ``EMPTY_CLOSE`` : jaws reached the closed stop / force reads empty.
* ``SLIP``        : a SECURED verdict followed by a lost grip
                    (``_grip_intact`` False at a transport waypoint), emitted
                    by the executor's transport checks via
                    :func:`slip_after_secured`.
* ``UNKNOWN``     : registers unreadable / inconclusive. NEVER triggers a
                    retry.

The Franka trace thresholds (empty_threshold 8 mm, slip_drop 6 mm,
FRANKA_OPEN_APERTURE 0.08, plateau windows) are not used: they describe an
aperture trace this gripper cannot produce.

Pure module: stdlib only, no driver import.
"""
from __future__ import annotations

from enum import Enum
from typing import Optional

__all__ = [
    "GraspOutcome",
    "classify_grasp_endpoint",
    "slip_after_secured",
]


class GraspOutcome(str, Enum):
    """Discrete grasp outcome (Robotiq gObj / sim-telemetry vocabulary)."""

    SECURED = "secured"
    EMPTY_CLOSE = "empty_close"
    SLIP = "slip"
    UNKNOWN = "unknown"


def classify_grasp_endpoint(
    gobj: Optional[bool],
    gripper_pos: Optional[float],
    *,
    closed_pos: float = 250.0,
    force_holding: Optional[bool] = None,
) -> GraspOutcome:
    """
    Classify one post-close endpoint read into a :class:`GraspOutcome`.

    Parameters
    ----------
    gobj:
        Robotiq object-detect flag (register 13). True = the jaws stopped
        early on an object; False = they reached their commanded position;
        None = register unreadable.
    gripper_pos:
        Jaw position register (0 open .. 255 closed), or None if
        unreadable.
    closed_pos:
        Position at/above which the jaws count as at the closed stop
        (executor's ``GRIPPER_FULLY_CLOSED``).
    force_holding:
        The executor's TCP-force corroboration (``force_verdict``), when
        available. Only consulted when gObj does NOT confirm; a force hold
        with jaws short of the stop is a SECURED (the flat-plate
        false-positive is excluded by the closed-stop check first).

    Decision table (first match wins)::

        gobj True                                  -> SECURED
        gobj None and pos None                     -> UNKNOWN  (unreadable)
        pos >= closed_pos                          -> EMPTY_CLOSE
        force_holding True (pos short of stop)     -> SECURED
        gobj False                                 -> EMPTY_CLOSE
        otherwise (gobj None, pos short, no force) -> UNKNOWN
    """
    if gobj is True:
        return GraspOutcome.SECURED
    if gobj is None and gripper_pos is None:
        return GraspOutcome.UNKNOWN
    if gripper_pos is not None and float(gripper_pos) >= float(closed_pos):
        # At the mechanical stop: nothing between the jaws. Checked BEFORE the
        # force corroboration: a large wrist force with fully closed jaws is
        # the tool pressing a plate/table, not a grasp (flat-plate veto).
        return GraspOutcome.EMPTY_CLOSE
    if force_holding:
        return GraspOutcome.SECURED
    if gobj is False:
        return GraspOutcome.EMPTY_CLOSE
    return GraspOutcome.UNKNOWN


def slip_after_secured(
    prev_outcome: Optional[GraspOutcome], grip_intact: Optional[bool]
) -> Optional[GraspOutcome]:
    """
    SLIP transition: a grasp that classified SECURED whose grip later reads
    lost (``_grip_intact`` False at a stationary transport waypoint).

    Returns ``GraspOutcome.SLIP`` when the transition fires, else None
    (including when ``grip_intact`` is None/unreadable -- an unreadable
    check never escalates, same rule as UNKNOWN never retrying).
    """
    if prev_outcome == GraspOutcome.SECURED and grip_intact is False:
        return GraspOutcome.SLIP
    return None

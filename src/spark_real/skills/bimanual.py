"""
Bimanual primitive stubs (registered for planner prompt only).

The actual execution of bimanual primitives lives in
:class:`spark_real.control.bimanual_executor.BimanualScoreExecutor`
because they need to dispatch to two arms in parallel, which the
single-arm :class:`ScoreExecutor` cannot express.

The stubs in this module exist so the skill registry can list them in
``get_prompt_section()`` for the planner. If the bimanual executor is
the one running the score, these stubs are never called (the executor
intercepts on ``_BIMANUAL_PRIMITIVES`` before falling through to the
registry). If a single-arm executor sees one of these by mistake, the
stub returns a clear ExecutionResult-shaped error.
"""

from __future__ import annotations

import logging
import time

from spark_real.skills.registry import spark_skill
from spark_real.control.executor_types import ExecutionResult as _ExecutionResult

logger = logging.getLogger(__name__)


def _execution_result(**kwargs):
    return _ExecutionResult(**kwargs)


def _wrong_executor_error(name: str):
    return _execution_result(
        action_type=name,
        success=False,
        message=(
            f"primitive {name!r} requires the bimanual executor; running it "
            "on a single-arm ScoreExecutor is a planner / family mismatch."
        ),
    )


@spark_skill(
    name="pick_with_arm",
    description=(
        "Bimanual: have the specified arm grasp a labeled keypoint using "
        "the 6-DoF grasp generator. Use this whenever the BT root is the "
        "bimanual family, never plain 'grasp_se3'."
    ),
    params={
        "arm": str,
        "keypoint_label": str,
        "target_width": float,
        "force": float,
        "prefer_angled": bool,
    },
)
def pick_with_arm(executor, params):  # noqa: D401
    return _wrong_executor_error("pick_with_arm")


@spark_skill(
    name="place_with_arm",
    description=(
        "Bimanual: have the specified arm release the held object. Equivalent "
        "to 'release' but with explicit arm routing."
    ),
    params={"arm": str, "tilt_angle": float},
)
def place_with_arm(executor, params):
    return _wrong_executor_error("place_with_arm")


@spark_skill(
    name="move_to_keypoint_arm",
    description=(
        "Bimanual: move the specified arm's EE to a labeled keypoint with "
        "optional XYZ offsets in the arm's base frame."
    ),
    params={
        "arm": str,
        "keypoint_label": str,
        "offset_x": float,
        "offset_y": float,
        "offset_z": float,
    },
)
def move_to_keypoint_arm(executor, params):
    return _wrong_executor_error("move_to_keypoint_arm")


@spark_skill(
    name="grasp_arm",
    description="Bimanual: close the specified arm's gripper with a force cap.",
    params={"arm": str, "force": float, "target_width": float},
)
def grasp_arm(executor, params):
    return _wrong_executor_error("grasp_arm")


@spark_skill(
    name="release_arm",
    description="Bimanual: open the specified arm's gripper.",
    params={"arm": str, "tilt_angle": float},
)
def release_arm(executor, params):
    return _wrong_executor_error("release_arm")


@spark_skill(
    name="handoff",
    description=(
        "Bimanual: pass a held object from 'from_arm' to 'to_arm' at "
        "'meeting_point' (xyz in world frame). The to_arm grasps before "
        "the from_arm releases, so the object is never unsupported."
    ),
    params={
        "from_arm": str,
        "to_arm": str,
        "keypoint_label": str,
        "meeting_point": list,
        "grasp_width": float,
        "force": float,
    },
)
def handoff(executor, params):
    return _wrong_executor_error("handoff")


@spark_skill(
    name="bimanual_lift",
    description=(
        "Bimanual: both arms grasp opposite sides of an object and lift "
        "together. Use for objects too heavy / large for one arm "
        "(e.g. trays, large plates, pots)."
    ),
    params={
        "keypoint_label": str,
        "object_width": float,
        "grip_width": float,
        "lift_height": float,
        "force": float,
    },
)
def bimanual_lift(executor, params):
    return _wrong_executor_error("bimanual_lift")


@spark_skill(
    name="hold_in_place",
    description=(
        "Bimanual: the specified arm holds its current TCP pose for 'dwell' "
        "seconds. Used as the stationary half of a coordinated parallel "
        "block (one arm anchors while the other manipulates)."
    ),
    params={"arm": str, "dwell": float},
)
def hold_in_place(executor, params):
    return _wrong_executor_error("hold_in_place")


@spark_skill(
    name="bimanual_place",
    description=(
        "Symmetric counterpart of bimanual_lift: both arms move the "
        "shared object down to a target placement and release together. "
        "Use AFTER bimanual_lift to deposit at a destination (e.g. tray "
        "lifted at station A and placed at station B). Both arms remain "
        "synchronized via the executor's parallel-servo path."
    ),
    params={"target_label": str, "place_offset_z": float, "release_dwell": float},
)
def bimanual_place(executor, params):  # noqa: D401
    return _wrong_executor_error("bimanual_place")


@spark_skill(
    name="bimanual_handover_sponge",
    description=(
        "Compound bimanual: left arm grips the sponge by its width, hands "
        "off to the right arm which re-grips along the length for "
        "cylindrical-glass scrubbing. Use whenever a sponge needs both a "
        "flat-surface and a curved-surface grip in one task."
    ),
    params={
        "sponge_label": str,
        "width_grip": float,
        "length_grip": float,
        "force": float,
        "meeting_point": list,
    },
)
def bimanual_handover_sponge(executor, params):
    return _wrong_executor_error("bimanual_handover_sponge")

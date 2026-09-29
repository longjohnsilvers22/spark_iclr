"""
Per-primitive wall-clock timeouts and a watchdog runner.

When a primitive overruns its budget the motion is aborted and the result is
routed into the recovery layer as a failed post-condition (it does NOT
terminate the trial). Defaults may be overridden by a ``timeouts:`` block in
the RobotProfile or family YAML.
"""

import logging
import threading
import time
from typing import Callable, Dict, Optional

from spark_real.config import family_block
from spark_real.control.executor_types import AbortRequested, ExecutionResult

logger = logging.getLogger(__name__)

# Budgets in seconds. ``wipe`` is the planner-grammar alias for
# ``constrained_scrub`` (10s) and is listed so a lookup by either name works.
# DEFAULT_TIMEOUT_S is applied to any primitive not named here.
DEFAULT_TIMEOUT_S = 8.0

PAPER_TIMEOUTS_S: Dict[str, float] = {
    "move_to_keypoint": 4.0,
    "move_relative": 2.0,
    "grasp": 1.0,
    "release": 1.0,
    "insert": 6.0,
    "push_object": 4.0,
    "open_drawer": 5.0,
    "wipe": 10.0,
    "constrained_scrub": 10.0,
}


# Primitives whose cost is perception, not motion. Kept separate so
# PAPER_TIMEOUTS_S stays exactly the paper's table.
#
# verify_placed budget, from measured cost:
#   scoped cost   1 prompt x 2 contributing cameras         = 2 SAM3 passes
#   per pass      ~ 7.5s (30s / 4 passes on sideview)
#   capture       reused inside the freshness window, else  ~ 1s
#   worst case    2 x 7.5 + 1                               ~ 16s
#   budget        16s + margin                              = 25.0s
# Shrink this as raw SAM3 latency comes down.
NON_PAPER_TIMEOUTS_S: Dict[str, float] = {
    "verify_placed": 25.0,
    # Pure proprioception: a gripper register read. Anything approaching this
    # is a wedged RTDE link, not a slow check.
    "verify_grasp": 5.0,
}


def default_timeouts() -> Dict[str, float]:
    """Paper-spec per-primitive budgets plus the perception-cost primitives."""
    out = dict(PAPER_TIMEOUTS_S)
    out.update(NON_PAPER_TIMEOUTS_S)
    return out


def load_timeouts(profile=None, family: Optional[str] = None) -> Dict[str, float]:
    """
    Resolve per-primitive timeouts, paper defaults overlaid by YAML.

    Merges any ``timeouts:`` block from the resolved RobotProfile
    (``profile.raw``) or from ``configs/<family>_default.yaml``.

    A ``default:`` key inside the YAML block overrides DEFAULT_TIMEOUT_S for
    unlisted primitives and is stored under the special key ``"default"``.
    Any malformed entry is skipped with a warning; the paper value stands.
    """
    timeouts = default_timeouts()
    timeouts["default"] = DEFAULT_TIMEOUT_S

    raw_block = family_block(profile, family or "", "timeouts")
    if isinstance(raw_block, dict):
        for name, val in raw_block.items():
            try:
                timeouts[str(name)] = float(val)
            except (TypeError, ValueError):
                logger.warning(
                    "Ignoring non-numeric timeout for '%s' (%r)", name, val
                )
    return timeouts


def timeout_for(timeouts: Dict[str, float], action_type: str) -> float:
    """
    Look up the budget for ``action_type``, falling back to the default.
    """
    if action_type in timeouts:
        return float(timeouts[action_type])
    return float(timeouts.get("default", DEFAULT_TIMEOUT_S))


def run_with_timeout(
    executor,
    action_type: str,
    budget_s: float,
    fn: Callable,
) -> "object":
    """
    Run ``fn`` under a wall-clock budget, aborting via the executor on expiry.

    A daemon watchdog timer fires after ``budget_s`` seconds and invokes
    ``executor.abort()`` (the same path /api/stop uses), which raises
    AbortRequested inside the motion helpers and stops the CartesianServo.

    Returns whatever ``fn`` returns within budget. On a timeout it returns a
    failed ExecutionResult and clears the watchdog's transient abort flag. An
    AbortRequested raised by ``fn`` itself (operator stop) is re-raised
    unchanged.
    """
    if budget_s is None or budget_s <= 0:
        return fn()

    fired = {"timeout": False, "epoch": None}

    def _on_expire():
        fired["timeout"] = True
        logger.warning(
            "[timeout] primitive '%s' exceeded %.1fs budget; aborting motion "
            "and routing to recovery",
            action_type,
            budget_s,
        )
        try:
            # Claim an epoch for THIS abort so the retraction below undoes
            # exactly this request. abort(epoch=...) skips its own bump.
            note = getattr(executor, "note_abort_requested", None)
            if callable(note):
                fired["epoch"] = note()
                executor.abort(epoch=fired["epoch"])
            else:
                executor.abort()
        except Exception as exc:
            logger.warning("[timeout] executor.abort() raised: %s", exc)

    def _retract_watchdog_abort() -> bool:
        """
        Undo the watchdog's own abort. False unless it is the ONLY
        outstanding request; an operator stop raised before or after the
        watchdog fired must survive (see retract_abort for the rule).
        """
        retract = getattr(executor, "retract_abort", None)
        if callable(retract) and fired["epoch"] is not None:
            return bool(retract(fired["epoch"]))
        executor._abort = False
        return True

    watchdog = threading.Timer(float(budget_s), _on_expire)
    watchdog.daemon = True
    t0 = time.time()
    watchdog.start()
    try:
        result = fn()
    except AbortRequested:
        watchdog.cancel()
        if fired["timeout"] and _retract_watchdog_abort():
            # The watchdog's abort, not the operator's: convert to a failed
            # post-condition. A refused retraction means a genuine stop is
            # standing and falls through to `raise`.
            return ExecutionResult(
                action_type=action_type,
                success=False,
                message=(
                    f"{action_type} timed out after {budget_s:.1f}s "
                    f"(budget exceeded)"
                ),
                duration=time.time() - t0,
            )
        raise
    finally:
        watchdog.cancel()

    if fired["timeout"]:
        # fn returned (rather than raising) after the watchdog fired, e.g. a
        # primitive that swallows AbortRequested internally. Treat as a timed-
        # out failure and retract the transient abort flag.
        if not _retract_watchdog_abort():
            # A genuine operator stop landed while the primitive unwound; it
            # outranks the timeout and must end the run.
            raise AbortRequested()
        return ExecutionResult(
            action_type=action_type,
            success=False,
            message=(
                f"{action_type} timed out after {budget_s:.1f}s (budget exceeded)"
            ),
            duration=time.time() - t0,
        )
    return result

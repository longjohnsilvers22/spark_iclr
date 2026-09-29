"""
Bimanual score executor.

Runs SPARK YAML behavior-tree scores against a :class:`BimanualSafeRobot`,
adding two structural node types on top of the single-arm grammar:

* ``parallel``: children run concurrently on independent arm threads;
  the implicit join is a sync barrier.
* ``sync_barrier``: mid-parallel rendezvous, both arms wait until
  every concurrent branch hits the barrier before proceeding.

Per-arm leaf primitives carry an ``arm: "left"|"right"`` field in
``params``. The executor maintains per-arm state dicts (``_holding``,
``_last_pick_label``, ``_last_keypoint_label``) and dispatches arm-tagged
primitives to a per-arm helper that mirrors the single-arm
``ScoreExecutor`` surface. Bimanual-only primitives (``handoff``,
``bimanual_lift``, ``hold_in_place``) are handled inline.

There is no bimanual recovery module. When one branch of a ``parallel``
block fails, the other branches run to completion (or to a broken
barrier) and the block reports failure once every branch has joined;
the enclosing sequence then stops.

The executor delegates single-arm primitives (``move_to_keypoint``,
``grasp``, ``release``, etc.) to per-arm :class:`ScoreExecutor` instances
constructed at init, so the single-arm code paths are reused verbatim.
"""

from __future__ import annotations

import logging
import threading
import time
from dataclasses import dataclass, field
from typing import Any, Callable, Dict, List, Optional, Tuple

import numpy as np

from spark_real.control.bimanual_safe_robot import BimanualSafeRobot
from spark_real.control.bimanual_servo import BimanualCartesianServo
from spark_real.control.executor_types import AbortRequested
from spark_real.control.score_executor import (
    ExecutionResult,
    ScoreExecutor,
)
from spark_real.skills.registry import SkillRegistry

logger = logging.getLogger(__name__)


# Single source of truth for which primitive names route to per-arm
# single-arm execution vs the bimanual-only handlers. Anything not in
# either set falls through to the bimanual skill registry, which is
# auto-discovered the same way the single-arm registry is.
_PER_ARM_PRIMITIVES = {
    "move_to_keypoint",
    "move_to_keypoint_arm",
    "grasp",
    "grasp_arm",
    "release",
    "release_arm",
    "move_relative",
    "wait",
    "wiggle",
    "push_object",
    "open_drawer",
    "pour",
    "push",
    "pull",
    "drag",
    "stack",
    "grasp_se3",
    "pick_with_arm",
    "place_with_arm",
    "tool_grip_pose",
    "constrained_scrub",
    "rinse",
    "sponge_wash",
}

_BIMANUAL_PRIMITIVES = {
    "handoff",
    "bimanual_lift",
    "bimanual_place",
    "hold_in_place",
    "bimanual_handover_sponge",
    # Bimanual cloth-fold primitives (spark_real/skills/bimanual_cloth.py).
    # These have no _skill_* handler; they resolve through the auto-discovered
    # bimanual skill registry with this executor passed as `executor`.
    "bimanual_grasp_points",
    "bimanual_lift_together",
    "bimanual_arc_drape",
    "bimanual_shirt_fold",
}


@dataclass
class BimanualState:
    """
    Per-arm executor state (lifted out so recovery code can peek).
    """

    holding: Dict[str, bool] = field(
        default_factory=lambda: {"left": False, "right": False}
    )
    last_pick_label: Dict[str, str] = field(
        default_factory=lambda: {"left": "", "right": ""}
    )
    last_place_label: Dict[str, str] = field(
        default_factory=lambda: {"left": "", "right": ""}
    )
    last_keypoint: Dict[str, Optional[str]] = field(
        default_factory=lambda: {"left": None, "right": None}
    )
    placed_labels: Dict[str, set] = field(
        default_factory=lambda: {"left": set(), "right": set()}
    )


class BimanualScoreExecutor:
    """
    Score executor that dispatches per-arm and parallel BT nodes.
    """

    # Mirrored from the per-arm executor that ran verification; None until a
    # score has been verified. success_verifier.reset_run_state sets it to
    # None at the task boundary, and task_success reads it.
    verify_outcome = None

    def __init__(
        self,
        safe: BimanualSafeRobot,
        detection_map: Optional[Dict] = None,
        velocity: float = 0.20,
        pipeline=None,
        strict_placement_verify: bool = False,
    ):
        self.safe = safe
        self.driver = safe.driver
        self.detection_map = detection_map or {}
        self.velocity = velocity
        self._pipeline = pipeline
        self.state = BimanualState()
        self._results: List[ExecutionResult] = []
        self._abort = False
        self._running = False
        self._barriers: Dict[str, threading.Barrier] = {}
        # Arm that dispatched the most recent per-arm leaf; verification
        # runs on that arm's executor.
        self._last_arm = "right"

        # Per-arm single-arm executors, each wrapping the corresponding
        # SafeRobot, so the full single-arm primitive library is reused by
        # picking which arm to dispatch to.
        self._arm_executors: Dict[str, ScoreExecutor] = {
            "left": ScoreExecutor(
                safe.left,
                detection_map=self.detection_map,
                velocity=velocity,
                pipeline=pipeline,
                strict_placement_verify=strict_placement_verify,
            ),
            "right": ScoreExecutor(
                safe.right,
                detection_map=self.detection_map,
                velocity=velocity,
                pipeline=pipeline,
                strict_placement_verify=strict_placement_verify,
            ),
        }

        for ex in self._arm_executors.values():
            # _verify borrows an arm executor for the verdict. The opt-in
            # pick-cycle replay stays single-arm only: the bimanual leaves
            # still carry params['arm'] and the other arm's steps, and the
            # arm executor has no recorder (its execute_score never runs).
            ex._replay_last_pick_cycle = self._skip_replay

        # Bimanual cartesian servo. The inter-arm CBF scale is applied once
        # per servo_to call as a velocity cap, not re-evaluated per tick.
        self._servo = BimanualCartesianServo(safe)

        # Auto-discovered skill registry for bimanual-only primitives that are
        # implemented as @spark_skill functions (e.g. the bimanual cloth fold)
        # rather than inline _skill_* handlers. Lazily built on first use.
        self._skill_registry = None

    # public abort / running
    def update_detections(self, detection_map: Dict) -> None:
        self.detection_map = detection_map
        for ex in self._arm_executors.values():
            ex.detection_map = detection_map

    def request_abort(self) -> None:
        self._abort = True
        for ex in self._arm_executors.values():
            # Through note_abort_requested where the per-arm executor has it:
            # a bare `_abort = True` does not advance that executor's abort
            # epoch, so its execute_score would wipe the request when the run
            # starts (see ScoreExecutor.note_abort_requested).
            note = getattr(ex, "note_abort_requested", None)
            if callable(note):
                note()
            else:
                ex._abort = True

    def note_abort_requested(self) -> None:
        self.request_abort()

    def reset_task_state(self) -> None:
        """Task boundary (success_verifier.reset_run_state): re-arm the
        per-arm abort epochs and drop per-task state on both arms."""
        self.state = BimanualState()
        for ex in self._arm_executors.values():
            ex.reset_task_state()

    @staticmethod
    def _skip_replay(score: Dict, actions: List[Dict], max_attempts: int) -> None:
        logger.warning(
            "verification.retry_on_fail: replay is single-arm only; "
            "not replaying a bimanual score"
        )

    def abort(self) -> bool:
        """
        Request a stop and brake both arms. True only if both arms braked.
        """
        self._abort = True
        for arm in ("left", "right"):
            self._servo.for_arm(arm).abort()
        # A list, not a generator: all() must not short-circuit past an arm.
        return all([ex.abort() for ex in self._arm_executors.values()])

    @property
    def is_running(self) -> bool:
        return self._running

    # entry point
    def execute_score(self, score: Dict) -> List[ExecutionResult]:
        """
        Run a bimanual YAML score; returns the accumulated results.
        """
        self._results = []
        self._abort = False
        self._running = True
        try:
            tree = score.get("tree")
            if tree is None:
                self._results.append(
                    ExecutionResult(
                        action_type="<root>",
                        success=False,
                        message="score missing 'tree'",
                    )
                )
                return self._results
            self._run_node(tree)
            if self._pipeline is not None and not self._abort:
                self._verify(score, tree)
        finally:
            self._running = False
        return self._results

    def _verify(self, score: Dict, tree: Dict) -> None:
        """Run single-arm post-task verification on the arm that dispatched the last leaf."""
        ex = self._arm_executors[self._last_arm]
        # The verify row lands in this run's results, as it does single-arm.
        ex._results = self._results
        leaves = ex._leaf_actions(ex._flatten_tree(tree))
        try:
            ex._run_post_task_verification(score, leaves)
        except AbortRequested:
            logger.warning("Post-task verification aborted by user")
        self.verify_outcome = ex.verify_outcome

    # tree walker (recursive, handles parallel/sync_barrier explicitly)
    def _run_node(self, node: Dict) -> bool:
        if self._abort:
            return False
        ntype = (node or {}).get("type", "sequence")
        children = node.get("children") or []

        if ntype in ("sequence", "selector"):
            # Selector semantics: succeed on first success, fall through
            # on failure. Sequence: all must succeed.
            for child in children:
                ok = self._run_node(child)
                if not ok and ntype == "sequence":
                    return False
                if ok and ntype == "selector":
                    return True
            return ntype == "sequence"

        if ntype == "parallel":
            return self._run_parallel(children, node)

        # sync_barrier (and every other leaf) falls through to _run_leaf,
        # which calls _barrier_wait. If the barrier wasn't registered
        # (standalone barrier, or one with an unknown name), _barrier_wait
        # logs a warning and returns True so the standalone case still
        # behaves as a no-op without consuming a thread.
        return self._run_leaf(node)

    # parallel dispatch
    def _run_parallel(self, children: List[Dict], parent: Dict) -> bool:
        if not children:
            return True

        # Pre-compute how many threads will hit each named barrier inside
        # this parallel block. Walk recursively because branches are
        # usually `sequence` nodes that contain the barrier among their
        # children. A barrier name lives on `params.name` (matching the
        # planner's emit format) with a fallback to `node.name` and then
        # to the auto-generated "<barrier-N>" at the branch depth.
        barrier_counts: Dict[str, int] = {}

        def _count(node: Dict, depth: int) -> None:
            if (node or {}).get("type") == "sync_barrier":
                params = node.get("params") or {}
                bname = params.get("name") or node.get("name") or f"<barrier-{depth}>"
                barrier_counts[bname] = barrier_counts.get(bname, 0) + 1
                return
            for ch in node.get("children", []) or []:
                _count(ch, depth + 1)

        for branch in children:
            _count(branch, 0)

        for bname, count in barrier_counts.items():
            if count != len(children) and count > 0:
                logger.warning(
                    "parallel block has barrier %r reached by %d of %d "
                    "branches, some arms will hang. Authoring bug?",
                    bname,
                    count,
                    len(children),
                )
            # Even mismatched, set the barrier to the lower count so
            # at least the matched branches don't deadlock indefinitely.
            self._barriers[bname] = threading.Barrier(min(count, len(children)) or 1)

        results: Dict[int, bool] = {}

        def _branch(idx: int, sub: Dict) -> None:
            results[idx] = self._run_node(sub)

        threads = []
        for i, child in enumerate(children):
            t = threading.Thread(
                target=_branch, args=(i, child), name=f"bt-branch-{i}", daemon=True
            )
            t.start()
            threads.append(t)
        for t in threads:
            t.join()

        self._barriers.clear()
        return all(results.values())

    def _barrier_wait(self, name: str, timeout: float = 30.0) -> bool:
        b = self._barriers.get(name)
        if b is None:
            logger.warning("sync_barrier %r not found; continuing", name)
            return True
        try:
            b.wait(timeout=timeout)
            return True
        except threading.BrokenBarrierError:
            logger.error("sync_barrier %r broken (timeout or abort)", name)
            return False

    # leaf dispatch
    def _run_leaf(self, node: Dict) -> bool:
        t0 = time.time()
        ntype = node.get("type", "")
        params = dict(node.get("params") or {})

        if ntype == "sync_barrier":
            ok = self._barrier_wait(params.get("name", node.get("name", "<barrier-0>")))
            self._results.append(
                ExecutionResult(
                    action_type="sync_barrier", success=ok, duration=time.time() - t0
                )
            )
            return ok

        # Bimanual-only primitives.
        if ntype in _BIMANUAL_PRIMITIVES:
            handler = getattr(self, f"_skill_{ntype}", None)
            if handler is not None:
                res = handler(params, t0)
                self._results.append(res)
                return res.success
            # No inline handler: resolve through the auto-discovered bimanual
            # skill registry, passing THIS executor (so the skill sees both
            # arms via self.driver / self._arm_executors).
            res = self._dispatch_registry_skill(ntype, params, t0)
            self._results.append(res)
            return res.success

        # Per-arm primitives: dispatch to the right arm's single-arm
        # ScoreExecutor. The arm field is required for arm-tagged
        # primitive names; if missing, default to "right" and warn.
        arm = params.pop("arm", None)
        if arm is None and ntype in _PER_ARM_PRIMITIVES:
            logger.warning(
                "primitive %r in bimanual mode missing 'arm' field; "
                "defaulting to 'right'",
                ntype,
            )
            arm = "right"

        if arm not in ("left", "right"):
            self._results.append(
                ExecutionResult(
                    action_type=ntype,
                    success=False,
                    message=f"invalid arm {arm!r} for primitive {ntype!r}",
                    duration=time.time() - t0,
                )
            )
            return False

        # Strip the "_arm" suffix to map e.g. pick_with_arm -> grasp_se3.
        # The auto-discovered skill registry resolves the rest.
        clean_type = ntype
        if ntype.endswith("_arm"):
            clean_type = ntype[:-4]
        if ntype == "pick_with_arm":
            clean_type = "grasp_se3"
        elif ntype == "place_with_arm":
            clean_type = "release"

        sub_executor = self._arm_executors[arm]
        self._last_arm = arm
        # Sync per-arm state from the central dict so recovery and
        # holding-flags survive across primitives.
        sub_executor._holding = self.state.holding[arm]
        sub_executor._last_pick_label = self.state.last_pick_label[arm]
        sub_executor._last_keypoint_label = self.state.last_keypoint[arm]

        # Hand off to the single-arm executor's leaf dispatcher.
        result = sub_executor._dispatch_action(clean_type, params)

        # Roll executor state back into the central dict.
        self.state.holding[arm] = sub_executor._holding
        self.state.last_pick_label[arm] = sub_executor._last_pick_label
        self.state.last_keypoint[arm] = sub_executor._last_keypoint_label

        # Annotate result so logs / recovery can tell arms apart.
        result.action_type = f"{result.action_type}[{arm}]"
        self._results.append(result)
        return result.success

    def _dispatch_registry_skill(
        self, name: str, params: Dict, t0: float
    ) -> ExecutionResult:
        """
        Run a bimanual @spark_skill from the auto-discovered registry.

        Used for bimanual-only primitives that are implemented as registry
        skills (not inline _skill_* handlers), e.g. the bimanual cloth fold.
        The skill receives THIS executor so it can drive both arms via
        ``self.driver`` / ``self._arm_executors``.
        """
        if self._skill_registry is None:
            self._skill_registry = SkillRegistry()
        try:
            return self._skill_registry.dispatch(name, self, params)
        except Exception as exc:
            logger.exception("bimanual registry skill %r failed", name)
            return ExecutionResult(
                action_type=name,
                success=False,
                message=f"bimanual skill {name!r} raised: {exc}",
                duration=time.time() - t0,
            )

    # bimanual-only primitives (inline handlers)
    def _skill_handoff(self, params: Dict, t0: float) -> ExecutionResult:
        """
        Pass an object held in `from_arm` to `to_arm` at a meeting point.

        Sequence:
            1. Both arms move (in parallel) to safe pre-meeting offsets
               on each side of `meeting_point`.
            2. Both arms converge to the meeting point (parallel).
            3. `to_arm` closes its gripper.
            4. `from_arm` opens its gripper after a 0.3s settle.
            5. Both arms retract upward to safe height (parallel).
        """
        from_arm = params.get("from_arm")
        to_arm = params.get("to_arm")
        if {from_arm, to_arm} != {"left", "right"}:
            return ExecutionResult(
                action_type="handoff",
                success=False,
                message=f"invalid arms {from_arm}/{to_arm}",
                duration=time.time() - t0,
            )
        if not self.state.holding[from_arm]:
            return ExecutionResult(
                action_type="handoff",
                success=False,
                message=f"{from_arm} arm not holding anything",
                duration=time.time() - t0,
            )

        meeting = np.asarray(params.get("meeting_point", [0.0, 0.0, 0.30]), dtype=float)
        # Pre-meeting offsets: 8 cm to either side of the meeting point.
        offset = float(params.get("approach_offset", 0.08))
        pre_left = meeting + np.array([0.0, offset, 0.0])
        pre_right = meeting + np.array([0.0, -offset, 0.0])

        # Get current orientations to preserve.
        ori_left = self.driver.get_tcp_pose("left")[3:]
        ori_right = self.driver.get_tcp_pose("right")[3:]
        target_pre_left = np.concatenate([pre_left, ori_left])
        target_pre_right = np.concatenate([pre_right, ori_right])

        ok = self._servo.servo_to_parallel(
            target_pre_left, target_pre_right, timeout_s=6.0
        )
        if not all(ok.values()):
            return ExecutionResult(
                action_type="handoff",
                success=False,
                message=f"pre-meeting servo failed: {ok}",
                duration=time.time() - t0,
            )

        # Converge to meeting point. The `to_arm` approaches slightly
        # below the `from_arm`'s grasp so the receiver's jaws close
        # around the object below the giver's jaws.
        meet_left = np.concatenate(
            [
                meeting + np.array([0.0, 0.02, 0.0 if to_arm == "left" else 0.02]),
                ori_left,
            ]
        )
        meet_right = np.concatenate(
            [
                meeting + np.array([0.0, -0.02, 0.0 if to_arm == "right" else 0.02]),
                ori_right,
            ]
        )
        ok = self._servo.servo_to_parallel(meet_left, meet_right, timeout_s=4.0)
        if not all(ok.values()):
            return ExecutionResult(
                action_type="handoff",
                success=False,
                message=f"meeting servo failed: {ok}",
                duration=time.time() - t0,
            )

        # Receiver grasps.
        grasp_width = float(params.get("grasp_width", 0.030))
        force = float(params.get("force", 15.0))
        receiver_ok = self.driver.grasp_to_width(grasp_width, arm=to_arm, force=force)
        if not receiver_ok:
            return ExecutionResult(
                action_type="handoff",
                success=False,
                message=f"{to_arm} gripper failed to close on object",
                duration=time.time() - t0,
            )
        self.state.holding[to_arm] = True

        # Giver releases after a brief settle so the receiver has a
        # confirmed grasp before the object becomes unsupported.
        time.sleep(0.30)
        self.driver.open_gripper(arm=from_arm)
        self.state.holding[from_arm] = False

        # Retract both arms upward 8 cm.
        for arm in ("left", "right"):
            tp = self.driver.get_tcp_pose(arm)
            tp[2] += 0.08
            self._servo.servo_to(arm, tp, timeout_s=3.0)

        return ExecutionResult(
            action_type="handoff",
            success=True,
            message=f"{from_arm}->{to_arm}",
            duration=time.time() - t0,
        )

    def _skill_bimanual_lift(self, params: Dict, t0: float) -> ExecutionResult:
        """
        Both arms grasp opposite sides of a target object and lift together.
        """
        label = params.get("keypoint_label")
        if not label or label not in self.detection_map:
            return ExecutionResult(
                action_type="bimanual_lift",
                success=False,
                message=f"keypoint {label!r} not in detection_map",
                duration=time.time() - t0,
            )
        det = self.detection_map[label]
        center = np.asarray(det.get("position_3d", [0, 0, 0]), dtype=float)
        # Assume the object's longer axis is along Y; grasp from +Y and -Y.
        width = float(params.get("object_width", 0.20))
        grasp_force = float(params.get("force", 20.0))
        approach_height = 0.10

        # Pre-grasp poses (above each grasp point).
        ori_left = self.driver.get_tcp_pose("left")[3:]
        ori_right = self.driver.get_tcp_pose("right")[3:]
        pre_left = np.concatenate(
            [center + [0.0, width / 2, approach_height], ori_left]
        )
        pre_right = np.concatenate(
            [center + [0.0, -width / 2, approach_height], ori_right]
        )

        ok = self._servo.servo_to_parallel(pre_left, pre_right, timeout_s=6.0)
        if not all(ok.values()):
            return ExecutionResult(
                action_type="bimanual_lift",
                success=False,
                message="pre-grasp failed",
                duration=time.time() - t0,
            )

        # Descend to grasp height.
        grasp_left = pre_left.copy()
        grasp_left[2] = center[2]
        grasp_right = pre_right.copy()
        grasp_right[2] = center[2]
        ok = self._servo.servo_to_parallel(grasp_left, grasp_right, timeout_s=4.0)
        if not all(ok.values()):
            return ExecutionResult(
                action_type="bimanual_lift",
                success=False,
                message="descent failed",
                duration=time.time() - t0,
            )

        # Close both grippers in parallel.
        results: Dict[str, bool] = {}
        threads = []
        for arm in ("left", "right"):
            t = threading.Thread(
                target=lambda a=arm: results.__setitem__(
                    a,
                    self.driver.grasp_to_width(
                        float(params.get("grip_width", 0.020)), arm=a, force=grasp_force
                    ),
                ),
                daemon=True,
            )
            t.start()
            threads.append(t)
        for t in threads:
            t.join()
        if not all(results.values()):
            return ExecutionResult(
                action_type="bimanual_lift",
                success=False,
                message=f"grasp failed: {results}",
                duration=time.time() - t0,
            )
        self.state.holding["left"] = True
        self.state.holding["right"] = True

        # Lift together.
        lift_h = float(params.get("lift_height", 0.10))
        lift_left = grasp_left.copy()
        lift_left[2] += lift_h
        lift_right = grasp_right.copy()
        lift_right[2] += lift_h
        ok = self._servo.servo_to_parallel(lift_left, lift_right, timeout_s=4.0)
        return ExecutionResult(
            action_type="bimanual_lift",
            success=all(ok.values()),
            message="lifted" if all(ok.values()) else f"lift failed: {ok}",
            duration=time.time() - t0,
        )

    def _skill_bimanual_place(self, params: Dict, t0: float) -> ExecutionResult:
        """
        Lower a co-held object to a target and release together.

        Mirrors :meth:`_skill_bimanual_lift`: both arms descend in
        parallel to the placement, the per-arm grippers open, and both
        arms retract upward. Requires both arms to currently be holding
        (i.e. after a prior bimanual_lift).
        """
        if not (self.state.holding["left"] and self.state.holding["right"]):
            return ExecutionResult(
                action_type="bimanual_place",
                success=False,
                message="bimanual_place requires both arms holding "
                "(call bimanual_lift first)",
                duration=time.time() - t0,
            )
        target_label = params.get("target_label")
        target = self.detection_map.get(target_label)
        if target is None:
            return ExecutionResult(
                action_type="bimanual_place",
                success=False,
                message=f"target {target_label!r} not in detection map",
                duration=time.time() - t0,
            )
        place_offset_z = float(params.get("place_offset_z", 0.06))
        release_dwell = float(params.get("release_dwell", 0.3))

        target_xyz = np.asarray(target.get("position_3d"), dtype=float)
        # Current TCPs; preserve XY offsets relative to lifted-object centroid
        # so the grasp geometry does not change mid-place.
        lp = np.asarray(self.driver.get_tcp_pose("left"))
        rp = np.asarray(self.driver.get_tcp_pose("right"))
        mid_xy = 0.5 * (lp[:2] + rp[:2])
        delta_xy = np.array([target_xyz[0] - mid_xy[0], target_xyz[1] - mid_xy[1]])
        descend_z = target_xyz[2] + place_offset_z

        new_left = lp.copy()
        new_left[:2] += delta_xy
        new_left[2] = descend_z
        new_right = rp.copy()
        new_right[:2] += delta_xy
        new_right[2] = descend_z
        ok = self._servo.servo_to_parallel(new_left, new_right, timeout_s=5.0)
        if not all(ok.values()):
            return ExecutionResult(
                action_type="bimanual_place",
                success=False,
                message=f"descent failed: {ok}",
                duration=time.time() - t0,
            )

        # Release both grippers in parallel.
        threads = []
        for arm in ("left", "right"):
            t = threading.Thread(
                target=self.driver.open_gripper, kwargs={"arm": arm}, daemon=True
            )
            t.start()
            threads.append(t)
        for t in threads:
            t.join()
        self.state.holding["left"] = False
        self.state.holding["right"] = False
        time.sleep(release_dwell)

        # Retract both arms upward.
        retract_left = new_left.copy()
        retract_left[2] += 0.10
        retract_right = new_right.copy()
        retract_right[2] += 0.10
        self._servo.servo_to_parallel(retract_left, retract_right, timeout_s=3.0)
        return ExecutionResult(
            action_type="bimanual_place",
            success=True,
            message=f"placed at {target_label}",
            duration=time.time() - t0,
        )

    def _skill_hold_in_place(self, params: Dict, t0: float) -> ExecutionResult:
        """
        Servo `arm` to hold its current TCP pose for `dwell` seconds.

        Used as the stationary half of a parallel handoff: one arm
        anchors the workpiece while the other performs a manipulation.
        """
        arm = params.get("arm")
        if arm not in ("left", "right"):
            return ExecutionResult(
                action_type="hold_in_place",
                success=False,
                message=f"invalid arm {arm!r}",
                duration=time.time() - t0,
            )
        dwell = float(params.get("dwell", 3.0))
        time.sleep(dwell)
        return ExecutionResult(
            action_type="hold_in_place",
            success=True,
            message=f"{arm} held {dwell:.1f}s",
            duration=time.time() - t0,
        )

    def _skill_bimanual_handover_sponge(
        self, params: Dict, t0: float
    ) -> ExecutionResult:
        """
        Compound: left arm picks sponge by its width, hands off to right
        arm which re-grips along its length for cylindrical-glass cleaning.

        Equivalent to a sequence of pick_with_arm + handoff with sponge-
        specific defaults so the planner can express it in one line.
        """
        # Pick with left.
        pick_params = {
            "arm": "left",
            "keypoint_label": params.get("sponge_label", "sponge"),
            "target_width": float(params.get("width_grip", 0.030)),
            "force": float(params.get("force", 12.0)),
            "prefer_angled": True,
        }
        pick_result = self._arm_executors["left"]._dispatch_action(
            "grasp_se3", pick_params
        )
        if not pick_result.success:
            return ExecutionResult(
                action_type="bimanual_handover_sponge",
                success=False,
                message=f"left pick failed: {pick_result.message}",
                duration=time.time() - t0,
            )
        self.state.holding["left"] = True

        # Handoff to right with along-length re-grip.
        handoff_params = {
            "from_arm": "left",
            "to_arm": "right",
            "keypoint_label": params.get("sponge_label", "sponge"),
            "meeting_point": params.get("meeting_point", [0.0, 0.0, 0.30]),
            "grasp_width": float(params.get("length_grip", 0.020)),
            "force": float(params.get("force", 12.0)),
        }
        return self._skill_handoff(handoff_params, t0)

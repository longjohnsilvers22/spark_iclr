"""
LIBERO-PRO behavior-tree executor.

Owns the per-trial mutable state (``holding`` flag, last pick label /
position) and dispatches each BT primitive to a handler in
``libero_pro.primitives``.  Handlers are free functions with signature
``(executor, params)`` so they can stay short and live next to their
strategy variants (e.g. SE(3) / bowl / plate / top-down for picking).

The executor also covers the ``--use-dsl`` plan path: validating against
the typed ``spark_dsl`` library and macro-expanding before flattening.
Both paths converge on ``dispatch_action`` so per-primitive behaviour is
identical regardless of which path drove the runner.
"""
from __future__ import annotations

import dataclasses
import os
import time
from typing import Any, Optional

import numpy as np
import mujoco

from spark_bench.libero_pro.ik import (
    compute_ik,
    compute_ik_6dof,
    get_q,
    solve_ik_sideways,
)
from spark_bench.libero_pro.motion import (
    move_to_pose,
    find_actuator_ids,
    find_ee_site,
    find_joint_ids,
    find_robot_base_world,
    gripper_action,
    gripper_action_hold,
    joint_move,
    move_to,
    step_env,
)
from spark_bench.libero_pro.perception import redetect_agentview
from spark_real.perception.sticky_binding import (
    STATUS_MISSING,
    STATUS_MOVED,
    STATUS_SELF,
    STICKY_AMBIGUOUS,
    STICKY_ASSOCIATED,
    STICKY_HELD,
    STICKY_OCCLUDED,
    STICKY_OUT_OF_GATE,
    _labels_match,
    compute_scene_diff,
    dedup_candidate_indices,
    fuzzy_key,
    is_self_occluded,
    sticky_associate,
)
from spark_bench.libero_pro.telemetry import (
    GraspClassification,
    GraspOutcome,
    find_finger_qpos_addrs,
    read_aperture,
)
from spark_bench.libero_pro.primitives._common import fuzzy_get_det
from spark_bench.libero_pro.primitives import (
    drawer as _drawer,
    insert as _insert,
    keypoint as _keypoint,
    simple as _simple,
    wipe as _wipe,
)

# ``spark_dsl`` is optional - degraded gracefully when absent.
try:
    from spark_dsl.executor import BTExecutor
    from spark_dsl.skill_library import DEFAULT_LIBRARY as _DSL_LIBRARY
except Exception:  # pragma: no cover - DSL is optional
    BTExecutor = None  # type: ignore[assignment]
    _DSL_LIBRARY = None  # type: ignore[assignment]


__all__ = ['LiberoExecutor', 'execute_on_libero']


# Tree helpers

def _flatten_tree(tree: Any) -> list[dict]:
    """
    Depth-first flatten of a BT score into a flat list of leaf primitives.
    """
    actions: list[dict] = []

    def _walk(node: Any) -> None:
        if not isinstance(node, dict):
            return
        t = node.get('type')
        if t in ('sequence', 'selector'):
            for c in node.get('children', []) or []:
                _walk(c)
        elif t == 'retry':
            # Recovery-grammar composite. This flat open-loop executor has
            # no per-subtree success signal, so a retry inlines as one
            # attempt; trial-level retract+replan recovery supplies the
            # re-attempts.
            for c in node.get('children', []) or []:
                _walk(c)
        elif t == 'fallback':
            # Selector-with-recovery composite: children after the first
            # are contingency branches gated on the first failing. With no
            # mid-tree success signal, execute only the nominal branch.
            children = node.get('children', []) or []
            if children:
                _walk(children[0])
        else:
            actions.append(node)

    _walk(tree)
    return actions


def _dsl_strip_unknown(score: dict) -> dict:
    """
    Strip primitive params Gemini hallucinates the typed library doesn't know.

    Mirrors the legacy laxness so plans that almost-validate still run.
    """
    if _DSL_LIBRARY is None:
        return score
    prim_specs = _DSL_LIBRARY.primitives()

    def _walk(node):
        if not isinstance(node, dict):
            return node
        t = node.get('type')
        if t in ('sequence', 'selector', 'retry', 'fallback'):
            node['children'] = [_walk(c) for c in node.get('children', [])]
            return node
        spec = prim_specs.get(t)
        if spec is not None:
            p = node.get('params') or {}
            node['params'] = {k: v for k, v in p.items() if k in spec.slots}
        return node

    if 'tree' in score:
        return {'tree': _walk(score.get('tree'))}
    return _walk(score)


# Handler registry - keyed on the BT primitive ``type`` string.

def _handle_grasp_se3(executor, params: dict) -> None:
    """
    Compound primitive: pick approach + close gripper.

    Expands into ``move_to_keypoint`` + ``grasp``; the keypoint handler
    picks SE(3) / bowl rim / plate rim / top-down from
    ``executor.last_pick_label`` and the object lexicon.
    """
    keypoint_params = {
        'keypoint_label': params.get('keypoint_label', ''),
        'offset_x': params.get('offset_x', 0.0),
        'offset_y': params.get('offset_y', 0.0),
        'offset_z': params.get('offset_z', 0.0),
    }
    _keypoint.handle(executor, keypoint_params)
    grasp_params = {'force': params.get('force', 100)}
    _simple.handle_grasp(executor, grasp_params)


_HANDLERS = {
    'move_to_keypoint': _keypoint.handle,
    'wipe':             _wipe.handle,
    'constrained_scrub': _wipe.handle,
    'insert':           _insert.handle,
    'grasp':            _simple.handle_grasp,
    'grasp_se3':        _handle_grasp_se3,
    'release':          _simple.handle_release,
    'move_relative':    _simple.handle_move_relative,
    'turn_knob':        _simple.handle_turn_knob,
    'rotate':           _simple.handle_rotate,
    # The planner emits 'screw' for knob-turn tasks; alias it to rotate.
    'screw':            _simple.handle_rotate,
    'push_object':      _simple.handle_push_object,
    'wait':             _simple.handle_wait,
    # 'pull' is open_drawer under the recovery grammar's vocabulary.
    'open_drawer':      _drawer.handle_open_drawer,
    'pull':             _drawer.handle_open_drawer,
}


# Executor

class LiberoExecutor:
    """
    Executes a behaviour tree against a LIBERO ``OffScreenRenderEnv``.

    Holds joint/site/actuator indices + the per-trial mutable state.
    Per-primitive logic lives in ``libero_pro.primitives.*``; the class
    just exposes thin wrappers around motion/ik so handler bodies stay
    short.
    """

    def __init__(self, env, cfg, det_map: dict, *,
                  sam3=None, depth: Optional[np.ndarray] = None,
                  cam_pos: Optional[np.ndarray] = None,
                  cam_mat: Optional[np.ndarray] = None,
                  cam_fovy: Optional[float] = None,
                  cam_w: int = 640, cam_h: int = 480,
                  instruction: str = '',
                  prompts_for_refresh: Optional[list] = None,
                  trial_meta: Optional[dict] = None):
        self.env = env
        self.cfg = cfg
        self.det_map = det_map
        self.sam3 = sam3
        self.depth = depth
        self.cam_pos = cam_pos
        self.cam_mat = cam_mat
        self.cam_fovy = cam_fovy
        self.cam_w = cam_w
        self.cam_h = cam_h
        self.instruction = instruction
        # Prompts used when the freshness gate re-detects between primitives.
        # When None / empty, freshness gate is a no-op even if cfg.freshness_timeout_s is set.
        self.prompts_for_refresh = list(prompts_for_refresh or [])

        self.model = env.sim.model._model
        self.data = env.sim.data._data

        self.joint_ids = find_joint_ids(self.model)
        self.ee_site = find_ee_site(self.model)
        self.ndof = len(self.joint_ids)
        self.arm_actuator_ids, self.gripper_actuator_ids = find_actuator_ids(
            self.model, self.ndof)
        self.robot_base_world = find_robot_base_world(self.model, self.data)

        self._invalid = self.ee_site < 0 or self.ndof < 6

        # Mutable per-trial state.
        self.holding: bool = False
        self.last_pick_label: Optional[str] = None
        self.last_pick_det_pos: Optional[np.ndarray] = None
        self._label_history: dict = {}   # key -> list[(pos3, t)] for velocity
        self.last_place_target: Optional[np.ndarray] = None
        # Set by the open_drawer primitive: interior of the pulled-out
        # tray, so a follow-up place-into-drawer can target it.
        self.last_drawer_tray_pos: Optional[np.ndarray] = None
        # Per-trial diagnostics sink (shared with the runner; additive).
        self.trial_meta: dict = trial_meta if trial_meta is not None else {}
        # Plan-time binding snapshot for the pre-primitive scene diff:
        # label -> position_3d at plan time.  Self-caused changes
        # (retargets, releases) re-bind their labels as they happen.
        self.plan_binding: dict[str, Optional[np.ndarray]] = {
            k: (np.asarray(d.position_3d, dtype=float).copy()
                if getattr(d, 'position_3d', None) is not None else None)
            for k, d in det_map.items()}
        # Set when the scene diff routes execution to the runner's
        # recovery loop (target missing / moved beyond the retarget
        # bound).  run() stops dispatching once set.
        self.abort: Optional[dict] = None
        # Grasp-telemetry state.
        self.last_grasp_outcome: Optional[GraspClassification] = None
        self._finger_addrs: Optional[list[int]] = None
        # Per-label outcome of the last position-sticky instance
        # association (attached to SceneDiff / capture records).
        self._sticky_flags: dict[str, str] = {}
        # Same-source displacement reference: the AGENTVIEW-only centroid
        # each label was bound at (plan time or last accepted retarget).
        # Re-detections are agentview-only, while the plan binding may be
        # wrist-fused; the two cameras disagree by a systematic 1-2 cm on
        # some objects, so displacement is measured agentview-vs-agentview
        # and applied to the fused binding as a delta.
        self.agent_ref: dict[str, np.ndarray] = {}
        for k, d in det_map.items():
            p = getattr(d, 'position_agentview', None)
            if p is None:
                p = getattr(d, 'position_3d', None)
            if p is not None:
                self.agent_ref[k] = np.asarray(p, dtype=float).copy()
        # Per-capture fresh-observation stores (reset by each sticky
        # merge): in-gate associations, nearest out-of-gate candidates,
        # and the full deduped candidate list per label.  Verdict paths
        # (post-grasp second vote, post-release placement check) read
        # THESE - never the retained det_map entries.
        self._fresh_dets: dict[str, Any] = {}
        self._fresh_out_of_gate: dict[str, Any] = {}
        self._fresh_groups: dict[str, list] = {}
        # Wall-clock timestamp of the last successful perception.  Used by
        # the optional freshness gate (cfg.freshness_timeout_s) to decide
        # whether to re-run SAM3 before the next keypoint-referencing
        # action.  Initialised to "now" because det_map was just computed
        # at the call site.
        self._last_perception_t: float = time.time()

    # Thin wrappers around motion/ik so handler bodies stay short.
    def _move_to(self, target, gripper_open, steps=200) -> float:
        return move_to(self.env, self.model, self.data,
                        self.joint_ids, self.ee_site,
                        self.arm_actuator_ids, self.gripper_actuator_ids,
                        target, gripper_open, steps=steps)

    def _move_to_pose(self, target_pos, target_mat, gripper_open,
                       steps: int = 200) -> tuple:
        """Position-and-orientation OSC move; see motion.move_to_pose."""
        return move_to_pose(self.env, self.model, self.data,
                             self.joint_ids, self.ee_site,
                             self.arm_actuator_ids, self.gripper_actuator_ids,
                             target_pos, target_mat, gripper_open, steps=steps)

    def _move_to_monitored(self, label: str, target, gripper_open,
                             steps: int = 200, chunk: int = 100) -> float:
        """
        Chunked move that re-checks the scene BETWEEN chunks.

        Bounds mid-motion intervention-detection latency at roughly one
        chunk of sim time plus one verdict (~0.5 s + 0.17 s) instead of
        the full primitive duration (~2.4 s measured). Only active with
        cfg.scene_diff; otherwise falls through to the plain move. If the
        target binding moves mid-flight, the remaining waypoint shifts by
        the binding delta (in-flight retarget); a beyond-threshold move
        or a vanished target sets self.abort for recovery, and the motion
        stops where it is.
        """
        if not self._sticky_enabled() or chunk >= steps:
            return self._move_to(target, gripper_open, steps=steps)
        target = np.asarray(target, dtype=float).copy()
        sd0 = self._run_scene_diff(label, context='mid_motion')
        key = getattr(sd0, 'target_key', None) if sd0 is not None else None
        bind0 = (np.asarray(self.plan_binding.get(key), float).copy()
                 if key is not None and key in self.plan_binding else None)
        residual = 0.0
        done = 0
        while done < steps:
            n = min(chunk, steps - done)
            residual = move_to(self.env, self.model, self.data,
                                self.joint_ids, self.ee_site,
                                self.arm_actuator_ids,
                                self.gripper_actuator_ids,
                                target, gripper_open, steps=n)
            done += n
            if done >= steps:
                break
            self._maybe_refresh_perception(
                {'type': 'move_to_keypoint',
                 'params': {'keypoint_label': label}})
            sd = self._run_scene_diff(label, context='mid_motion')
            if sd is None:
                continue
            if not self._handle_target_diff(sd, context='mid_motion'):
                return residual
            if (key is not None and bind0 is not None
                    and key in self.plan_binding):
                delta = np.asarray(self.plan_binding[key], float) - bind0
                if float(np.linalg.norm(delta)) > 0.005:
                    target = target + delta
                    bind0 = np.asarray(self.plan_binding[key], float).copy()
        return residual

    def _joint_move(self, q_target, gripper_open, steps=200,
                    kp=40.0, kd=12.0) -> None:
        joint_move(self.env, self.model, self.data,
                    self.joint_ids, self.ee_site,
                    self.arm_actuator_ids, self.gripper_actuator_ids,
                    q_target, gripper_open, steps=steps, kp=kp, kd=kd)

    def _gripper(self, open_gripper, steps=60) -> None:
        gripper_action(self.env, open_gripper, steps=steps)

    def _gripper_hold(self, open_gripper, steps=60) -> None:
        """
        Open/close while OSC actively regulates EE to current pose.

        Use this in primitives where the gripper-close phase MUST not
        drift the EE off the grasp target (e.g. drawer handle pull,
        knob turn).  Costs nothing extra over ``_gripper`` other than
        a per-step site_xpos read.
        """
        gripper_action_hold(self.env, self.model, self.data, self.ee_site,
                              open_gripper, steps=steps)

    def _ik(self, target) -> np.ndarray:
        return compute_ik(self.model, self.data,
                          self.joint_ids, self.ee_site, target)

    def _ik6(self, target_pos, target_quat) -> np.ndarray:
        return compute_ik_6dof(self.model, self.data,
                                self.joint_ids, self.ee_site,
                                target_pos, target_quat)

    def _ik_sideways(self, target_pos, approach_dir, finger_close):
        if not getattr(self.cfg, 'use_pyroki', False):
            return None
        return solve_ik_sideways(
            self.joint_ids, target_pos, approach_dir, finger_close,
            self.robot_base_world, verbose=getattr(self.cfg, 'verbose', False))

    def _ee(self) -> np.ndarray:
        mujoco.mj_forward(self.model, self.data)
        return (self.data.site_xpos[self.ee_site].copy()
                if self.ee_site >= 0 else np.zeros(3))

    def _q_now(self) -> np.ndarray:
        return get_q(self.model, self.data, self.joint_ids)

    def _goal_satisfied(self) -> bool:
        """
        Best-effort BDDL goal check.  Returns False on any failure so a
        broken check_success never aborts execution prematurely.

        SPARK_NO_ORACLE=1 makes this always False: the executor's pre-action
        short-circuit, the push primitive's stop-poll, and the drawer pull's
        completion poll all lose the simulator's success signal, which a real
        robot does not have. Episode-end scoring is unaffected. Exists to
        measure the contribution of mid-episode predicate access.
        """
        if os.environ.get('SPARK_NO_ORACLE', '').strip() in ('1', 'true', 'yes'):
            return False
        try:
            return bool(self.env.check_success())
        except Exception:
            return False

    # Top-level entry point.
    def run(self, score: dict) -> None:
        if self._invalid:
            return
        actions = _flatten_tree(score.get('tree', {}))

        if self.cfg.verbose:
            print(f"[Plan] {len(actions)} actions: "
                  f"{[a.get('type', '?') for a in actions]}")
            for a in actions:
                if a.get('params', {}).get('keypoint_label'):
                    print(f"{a['type']}: "
                          f"keypoint_label={a['params']['keypoint_label']}")

        if getattr(self.cfg, 'use_dsl', False):
            actions = self._dsl_expand_or_fallback(score, actions)

        for idx, act in enumerate(actions):
            # Short-circuit: if the BDDL goal predicate is already
            # satisfied, do not run the remaining primitives (an extra
            # push after the plate lands on the goal region knocks it
            # back out).
            if self._goal_satisfied():
                if self.cfg.verbose:
                    print(f"[Exec] goal already satisfied before "
                          f"action {idx} ({act.get('type')}); short-circuit")
                break
            if not self.dispatch_action(idx, act) or self.abort is not None:
                # Scene diff aborted (target missing / moved beyond the
                # retarget bound): stop dispatching and let the runner's
                # recovery loop take over with a fresh detect.
                if self.cfg.verbose and self.abort is not None:
                    print(f"[SceneDiff] abort at action {idx}: {self.abort}")
                break

        # Publish the end-of-execution binding (self-caused changes like
        # the placed object are already re-bound) so the recovery loop's
        # scene diff compares against the end state, not the plan-time
        # state; otherwise the just-placed object reads as an external
        # 'moved' forever.
        try:
            self.trial_meta['final_binding'] = {
                k: ([float(x) for x in v] if v is not None else None)
                for k, v in self.plan_binding.items()}
        except Exception:
            pass

    def _dsl_expand_or_fallback(self, score: dict,
                                  legacy_actions: list[dict]) -> list[dict]:
        """
        Validate + macro-expand via spark_dsl; fall back to legacy on error.
        """
        if BTExecutor is None or _DSL_LIBRARY is None:
            return legacy_actions
        try:
            score = _dsl_strip_unknown(score)
            executor = BTExecutor(library=_DSL_LIBRARY)
            _ast, val_errs = _DSL_LIBRARY.validate_bt(score)
            expanded, exp_err = executor.expand_macros(score)
            if val_errs or exp_err:
                if self.cfg.verbose:
                    print(f"[DSL] validate/expand issues: "
                          f"{val_errs[:1]} {exp_err}; using legacy plan")
                return legacy_actions
            expanded_actions = _flatten_tree(expanded.get('tree', {}))
            if not expanded_actions:
                return legacy_actions
            if self.cfg.verbose:
                print(f"[DSL] expanded to {len(expanded_actions)} "
                      f"base primitive actions")
            return expanded_actions
        except Exception as e:
            print(f"[DSL] executor failed: {e}; falling back to legacy dispatch")
            return legacy_actions

    # Dispatcher - routes each action dict to its handler.
    def dispatch_action(self, act_idx: int, act: dict) -> bool:
        """
        Dispatch one leaf primitive.  Returns False when the pre-primitive
        scene diff aborted into recovery (caller stops the action loop).
        """
        atype = act.get('type')
        params = act.get('params', {}) or {}
        # Freshness gate (opt-in via cfg.freshness_timeout_s): before any
        # action that references a detected object, re-run SAM3 if the cached
        # detections are older than the timeout.
        self._maybe_refresh_perception(act)
        # Pre-primitive scene diff (opt-in via cfg.scene_diff): diff the
        # (just refreshed) det_map against the plan-time binding, retarget
        # small target moves in place, abort large / missing into recovery.
        if not self._pre_primitive_gate(act_idx, act):
            return False
        if self.cfg.verbose:
            ee = self._ee()
            print(f"[Exec {act_idx}] {atype} params={params} "
                  f"ee=({ee[0]:.3f},{ee[1]:.3f},{ee[2]:.3f}) "
                  f"holding={self.holding}")

        handler = _HANDLERS.get(atype)
        if handler is not None:
            handler(self, params)
        return self.abort is None

    # Freshness gate (DynamicVLA LAAS analog, symbolic-pipeline scope).
    @staticmethod
    def _action_uses_keypoint(act: dict) -> bool:
        params = act.get('params', {}) or {}
        return bool(params.get('keypoint_label') or params.get('label')
                    or params.get('target_label') or params.get('pick_label')
                    or params.get('place_label'))

    def _maybe_refresh_perception(self, act: dict) -> None:
        timeout = getattr(self.cfg, 'freshness_timeout_s', None)
        scene_diff_on = bool(getattr(self.cfg, 'scene_diff', False))
        if timeout is None and not scene_diff_on:
            return
        if self.sam3 is None or not self.prompts_for_refresh:
            return
        if not self._action_uses_keypoint(act):
            return
        age = time.time() - self._last_perception_t
        # scene_diff needs a fresh frame per gated primitive for the diff
        # to be meaningful (bounded below so back-to-back leaves don't
        # double-pay); otherwise the plain wall-clock gate applies.
        if scene_diff_on:
            if age < 0.05:
                return
        elif age < float(timeout):
            return
        try:
            fresh = redetect_agentview(self.env, self.sam3,
                                        self.prompts_for_refresh, self.cfg,
                                        all_instances=self._sticky_enabled())
        except Exception as e:
            if self.cfg.verbose:
                print(f"[Freshness] redetect failed: {e}")
            return
        n_labels, n_updated = self._merge_fresh_detections(fresh)
        self._last_perception_t = time.time()
        if self.cfg.verbose and n_labels:
            print(f"[Freshness] re-detected {n_labels} "
                  f"({n_updated} updates, {n_labels-n_updated} new)")

    # Position-sticky instance binding (always on with cfg.scene_diff)

    def _sticky_enabled(self) -> bool:
        return bool(getattr(self.cfg, 'scene_diff', False))

    def _sticky_gate_m(self) -> float:
        return max(float(getattr(self.cfg, 'retarget_max_cm', 10.0)),
                   12.0) / 100.0

    def _merge_fresh_detections(self, fresh) -> tuple[int, int]:
        """
        Merge a fresh DetectionResult into the executor's binding state.

        Legacy mode (scene_diff off): freshest-per-label wins, only keys
        the fresh frame produced are overwritten (conservative LAAS
        analog - other cached entries stay authoritative until re-seen).

        Sticky mode (scene_diff on): BIND-PRESERVING.  The fresh frame
        never rewrites ``det_map`` directly; it only populates the
        per-capture observation stores (``_fresh_dets`` /
        ``_fresh_out_of_gate`` / ``_fresh_groups``).  The scene diff then
        measures displacement SAME-SOURCE (fresh agentview centroid vs
        the ``agent_ref`` agentview centroid the label was bound at) and
        the explicit retarget / self-caused / recovery paths decide
        whether the authoritative (possibly wrist-fused) binding moves,
        always by the measured DELTA, never by adopting the fresh
        absolute position (the two cameras disagree by a systematic
        1-2 cm, which would read as phantom 'moved' verdicts).

        Association rules per bound label:
        * HELD label: skipped entirely; the camera cannot relocate the
          object in the gripper.
        * nearest in-gate instance -> observation recorded;
        * ambiguous / occluded-out-of-gate -> hold, no observation;
        * out-of-gate -> recorded separately; the diff reports the large
          displacement and routes to recovery WITHOUT poisoning det_map.

        Returns ``(n_labels_seen, n_labels_updated)`` where updated
        counts labels with a fresh in-gate association this capture.
        """
        self._sticky_flags = {}
        self._fresh_dets = {}
        self._fresh_out_of_gate = {}
        self._fresh_groups = {}
        if fresh is None:
            return 0, 0
        if not self._sticky_enabled() or not getattr(fresh, 'dets', None):
            n_updated = 0
            det_map = fresh.det_map or {}
            for k, v in det_map.items():
                if k in self.det_map:
                    n_updated += 1
                self.det_map[k] = v
            return len(det_map), n_updated

        rename = getattr(self.cfg, '_mp_phrase_to_concept', None) or {}
        groups: dict[str, list] = {}
        for d in fresh.dets:
            if getattr(d, 'position_3d', None) is None:
                continue
            groups.setdefault(rename.get(d.label, d.label), []).append(d)

        n_updated = 0
        for label, cands in groups.items():
            # Collapse SAM3 duplicate masks of the same object first;
            # otherwise nearest-to-previous association favours the
            # duplicate that moved least and under-measures real motion.
            if len(cands) > 1:
                keep = dedup_candidate_indices(
                    [c.position_3d for c in cands],
                    [float(getattr(c, 'confidence', 0.0)) for c in cands])
                cands = [cands[i] for i in keep]
            key = (label if label in self.det_map
                   else (fuzzy_key(self.det_map, label) or label))
            self._fresh_groups[key] = list(cands)
            prev = self.det_map.get(key)
            if prev is None:
                # Never seen before: adopt confidence top-1 (agentview-
                # only detection, so the ref IS the position).
                best = max(cands,
                           key=lambda c: float(getattr(c, 'confidence', 0.0)))
                self.det_map[key] = best
                self.agent_ref[key] = np.asarray(
                    best.position_3d, dtype=float).copy()
                if self.plan_binding.get(key) is None:
                    self.plan_binding[key] = np.asarray(
                        best.position_3d, dtype=float).copy()
                continue
            if (self.holding and self.last_pick_label
                    and _labels_match(key, self.last_pick_label)):
                # The object is in the gripper: its position is wherever
                # the hand is, and any mask SAM3 produces for it is
                # either the object-in-hand or a phantom.  Suppress
                # association for the duration of the hold.
                self._sticky_flags[key] = STICKY_HELD
                continue
            ref = self.agent_ref.get(key)
            if ref is None:
                p = getattr(prev, 'position_3d', None)
                ref = (np.asarray(p, dtype=float).copy() if p is not None
                       else self.plan_binding.get(key))
            if ref is None:
                continue
            res = sticky_associate(
                ref, [c.position_3d for c in cands],
                gate_m=self._sticky_gate_m(), ambiguity_sep_m=0.03)
            self._sticky_flags[key] = res.status
            if res.status == STICKY_OUT_OF_GATE:
                if is_self_occluded(ref, self._ee()):
                    # The arm is hovering on top of this object: it
                    # cannot be seen, and the far-away re-association is
                    # a lookalike, not a scene change.  Hold the binding;
                    # the grasp telemetry disambiguates milliseconds
                    # later.
                    self._sticky_flags[key] = STICKY_OCCLUDED
                    if self.cfg.verbose:
                        print(f"[Sticky] {key!r}: out-of-gate "
                              f"re-association while EE hovers over it - "
                              f"holding binding (self-occlusion)")
                else:
                    # Record the far candidate for the diff (it will read
                    # as moved-beyond-retarget and route to recovery) but
                    # do NOT rebind det_map to a possible lookalike.
                    self._fresh_out_of_gate[key] = cands[res.chosen_index]
                continue
            if res.status == STICKY_AMBIGUOUS or res.chosen_index is None:
                if self.cfg.verbose and res.status == STICKY_AMBIGUOUS:
                    print(f"[Sticky] {key!r}: {len(cands)} instances "
                          f"in-gate and inseparable; keeping previous "
                          f"binding")
                continue
            fresh_det = cands[res.chosen_index]
            self._fresh_dets[key] = fresh_det
            n_updated += 1
            if self.cfg.verbose:
                newp = np.asarray(fresh_det.position_3d, float).reshape(-1)
                pv = np.asarray(ref, float).reshape(-1)[:3]
                d_cm = float(np.linalg.norm(newp[:3] - pv)) * 100.0
                if d_cm > 0.5:
                    print(f"[StickyObs] {key!r} ref="
                          f"({pv[0]:.4f},{pv[1]:.4f},{pv[2]:.4f}) fresh="
                          f"({newp[0]:.4f},{newp[1]:.4f},{newp[2]:.4f}) "
                          f"delta={d_cm:.1f}cm "
                          f"conf={float(getattr(fresh_det, 'confidence', 0.0)):.3f}")
                if len(cands) > 1:
                    print(f"[Sticky] {key!r}: associated instance "
                          f"{res.chosen_index}/{len(cands)} at "
                          f"{(res.chosen_dist_m or 0)*100:.1f}cm "
                          f"({res.status})")
        return len(groups), n_updated

    def _shift_binding(self, key: str, fresh_pos, fresh_det=None) -> None:
        """
        Move the authoritative binding of ``key`` by the SAME-SOURCE delta
        implied by a fresh agentview observation.

        ``plan_binding`` (and the det_map entry's position) shifts by
        ``fresh_pos - agent_ref[key]`` so a wrist-fused bind keeps its
        cross-camera correction; ``agent_ref`` re-anchors at the fresh
        agentview centroid.  When no ref exists the fresh position is
        adopted outright.
        """
        fresh_pos = np.asarray(fresh_pos, dtype=float).reshape(-1)[:3].copy()
        ref = self.agent_ref.get(key)
        bound = self.plan_binding.get(key)
        if ref is not None and bound is not None:
            new_fused = np.asarray(bound, dtype=float) + (fresh_pos - ref)
        else:
            new_fused = fresh_pos.copy()
        self.plan_binding[key] = new_fused
        self.agent_ref[key] = fresh_pos.copy()
        hist = self._label_history.setdefault(key, [])
        hist.append((fresh_pos.copy(), time.time()))
        if len(hist) > 5:
            del hist[0]
        prev = self.det_map.get(key)
        base = fresh_det if fresh_det is not None else prev
        if base is not None:
            try:
                self.det_map[key] = dataclasses.replace(
                    base, position_3d=new_fused.copy(),
                    position_agentview=fresh_pos.copy())
            except Exception:
                self.det_map[key] = base

    def _object_velocity(self, key):
        """XY velocity (m/s) from the last two logged detections, or None."""
        hist = self._label_history.get(key)
        if not hist or len(hist) < 2:
            return None
        (p0, t0), (p1, t1) = hist[-2], hist[-1]
        dt = t1 - t0
        if dt <= 1e-3:
            return None
        v = (np.asarray(p1, float)[:2] - np.asarray(p0, float)[:2]) / dt
        if float(np.linalg.norm(v)) < 0.01:   # below 1 cm/s: treat as static
            return None
        return v

    def _track_and_grasp(self, label, hover_h=0.08, chunk=6, max_ticks=400,
                          xy_tol=0.012, retries=2):
        """
        Running node: reach and grasp wherever ``label`` is RIGHT NOW.

        The plain pick resolves the label to one point and then moves
        open-loop; this node re-resolves it every ``chunk`` sim steps,
        keeps a smoothed velocity in sim time, dead-reckons through the
        moments the gripper occludes the camera, aims at position plus
        velocity times the command horizon, and keeps servoing xy while
        the jaws close. Returns True with the object held (jaws closed on
        it, aperture stalled open), False when it gives up, in which case
        the caller falls back to the ordinary pick. Gated by
        cfg.velocity_lead_s > 0 or a ``track: true`` node param.
        """
        cfg = self.cfg
        if self.sam3 is None or not self.prompts_for_refresh:
            return False
        from spark_bench.libero_pro.telemetry import (
            find_finger_qpos_addrs, read_aperture)
        lead_s = float(getattr(cfg, 'velocity_lead_s', 0.0) or 0.0)
        lead_s = lead_s if lead_s > 0 else 0.3
        key = fuzzy_key(self.det_map, label) or label
        addrs = find_finger_qpos_addrs(self.model)
        step_dt = float(getattr(self.env, 'control_timestep', 0.05) or 0.05)
        horizon = lead_s + chunk * step_dt

        from spark_bench.libero_pro.perception import redetect_wrist
        stats = self.trial_meta.setdefault('track_obs', {
            'agent': 0, 'wrist': 0, 'none': 0,
            'no_fresh': 0, 'no_label': 0, 'blacklisted': 0, 'gated': 0})
        self.trial_meta['track_key'] = key
        self.trial_meta['track_binding'] = (
            [round(float(x), 3) for x in self.plan_binding[key]]
            if key in self.plan_binding else None)

        def _fresh_pos():
            d = self._fresh_dets.get(key) or self._fresh_out_of_gate.get(key)
            if d is None or getattr(d, 'position_3d', None) is None:
                return None
            return np.asarray(d.position_3d, dtype=float).copy()

        def observe():
            # Agentview first; when the gripper hides the target from it,
            # the eye-in-hand camera is looking straight at the object.
            for name, fn in (('agent', redetect_agentview),
                             ('wrist', redetect_wrist)):
                try:
                    fresh = fn(self.env, self.sam3, self.prompts_for_refresh,
                               cfg, all_instances=self._sticky_enabled())
                    if fresh is None:
                        continue
                    self._merge_fresh_detections(fresh)
                except Exception:
                    continue
                p = _fresh_pos()
                if p is not None:
                    stats[name] += 1
                    return p
            stats['none'] += 1
            return None

        def go(target, gripper_open):
            return move_to(self.env, self.model, self.data, self.joint_ids,
                           self.ee_site, self.arm_actuator_ids,
                           self.gripper_actuator_ids,
                           np.asarray(target, float), gripper_open,
                           steps=chunk)

        # Ambush, not chase: park the open jaws at the predicted intercept
        # point ahead of the object, let it come, close as it enters.
        # Velocity comes from agentview observations only (same source);
        # the wrist camera supplies position when the jaws hide the object.
        last_pos = None; last_t = None; v = np.zeros(2); grasp_z = None
        src_prev = None
        phase = 'intercept'; close_ticks = 0; ap_prev = None; stalled = 0
        occluded = 0; d_prev = None; away = 0
        T_amb = 1.0; z_grasp = None; last_agent = None
        rename = getattr(cfg, '_mp_phrase_to_concept', None) or {}
        wait_t0 = None
        blacklist = []
        search_i = 0
        log = self.trial_meta.setdefault('track_events', [])
        for tick in range(int(max_ticks)):
            t = float(self.data.time)
            src = None
            pos = None
            pred = None
            if last_pos is not None:
                pred = last_pos[:2] + v * (t - last_t)
            for name, fn in (('agent', redetect_agentview),
                             ('wrist', redetect_wrist)):
                if name == 'wrist' and (phase == 'intercept' or pred is None):
                    # A wrist frame taken while the arm flies is not a
                    # measurement (moving camera, changing pose); use the
                    # wrist only from a parked gripper, and only to refine
                    # a position the agentview already predicted.
                    continue
                try:
                    fresh = fn(self.env, self.sam3, self.prompts_for_refresh,
                               cfg, all_instances=True)
                except Exception:
                    fresh = None
                if fresh is None:
                    stats['no_fresh'] += 1
                    continue
                if tick == 0 and 'track_seen_labels' not in self.trial_meta:
                    self.trial_meta['track_seen_labels'] = sorted({
                        rename.get(d.label, d.label) for d in (fresh.dets or [])})
                # Associate by the node's own prediction, over EVERY
                # instance of the label: the plan-time sticky gate rejects
                # a target that has travelled, and picks a twin instance
                # (the other can) when the true one is far from the bind.
                cands = [d for d in (fresh.dets or [])
                         if rename.get(d.label, d.label) == key
                         and getattr(d, 'position_3d', None) is not None]
                if not cands:
                    stats['no_label'] += 1
                    continue
                n_before = len(cands)
                cands = [d for d in cands if not any(
                    float(np.linalg.norm(np.asarray(d.position_3d, float)[:2]
                                         - b)) < 0.03 for b in blacklist)]
                if not cands:
                    stats['blacklisted'] += 1
                    continue
                if pred is None:
                    # No track yet: the target starts where the plan bound
                    # it; a twin instance elsewhere must not win on
                    # confidence alone.
                    b0 = self.plan_binding.get(key)
                    if b0 is not None:
                        b0 = np.asarray(b0, float)[:2]
                        d_best = min(cands, key=lambda d: float(np.linalg.norm(
                            np.asarray(d.position_3d, float)[:2] - b0)))
                    else:
                        d_best = max(cands, key=lambda d: float(
                            getattr(d, 'confidence', 0.0) or 0.0))
                else:
                    # The prediction drifts while unobserved; widen the gate
                    # with the time since the last observation.
                    lost_now = max(0.0, t - last_t)
                    gate = (min(0.06 + 0.05 * lost_now, 0.22)
                            if name == 'agent' else 0.04)
                    d_best = min(cands, key=lambda d: float(np.linalg.norm(
                        np.asarray(d.position_3d, float)[:2] - pred)))
                    if float(np.linalg.norm(
                            np.asarray(d_best.position_3d, float)[:2]
                            - pred)) > gate:
                        stats['gated'] += 1
                        continue
                src, pos = name, np.asarray(d_best.position_3d, float).copy()
                break
            stats[src or 'none'] += 1
            if pos is not None:
                if src == 'agent':
                    # Velocity only from consecutive agentview pairs (same
                    # source, stationary camera). A wrist pair or a
                    # cross-camera pair reads backprojection offsets as
                    # motion.
                    if last_agent is not None and t - last_agent[1] > 1e-6:
                        v_new = (pos[:2] - last_agent[0][:2]) / (t - last_agent[1])
                        if float(np.linalg.norm(v_new)) < 0.15:
                            v = 0.6 * v + 0.4 * v_new
                    last_agent = (pos.copy(), t)
                    if grasp_z is None:
                        grasp_z = float(pos[2]); z_grasp = grasp_z - 0.03
                else:
                    if grasp_z is None:
                        grasp_z = float(pos[2]); z_grasp = grasp_z - 0.03
                last_pos = np.array([pos[0], pos[1], grasp_z])
                last_t = t
                src_prev = src
                occluded = 0
                search_i = 0
                est = last_pos.copy()
            else:
                if last_pos is None:
                    go(self._ee(), True)
                    continue
                occluded += 1
                lost_s = occluded * chunk * step_dt
                if lost_s > 8.0:
                    log.append({'tick': tick, 'event': 'lost_target'})
                    return False
                # Parked or closing: the object is arriving under the jaws
                # and the agentview cannot see it; dead-reckon, never
                # abandon the ambush to go looking. Search only while
                # still choosing an intercept, and only after 3 s.
                if lost_s > 3.0 and phase == 'intercept':
                    # Active search: hover the wrist camera over the last
                    # known position and sweep a square around it.
                    sq = [(0, 0), (0.08, 0), (0.08, 0.08), (-0.08, 0.08),
                          (-0.08, -0.08), (0.08, -0.08), (0, 0)]
                    if search_i == 0:
                        log.append({'tick': tick, 'event': 'search'})
                    ox, oy = sq[(search_i // 3) % len(sq)]
                    search_i += 1
                    go(np.array([last_pos[0] + ox, last_pos[1] + oy,
                                 (grasp_z if grasp_z is not None
                                  else last_pos[2]) + 0.16]), True)
                    continue
                est = last_pos.copy()
                est[:2] = last_pos[:2] + v * (t - last_t)
            ee = self._ee()
            d = float(np.linalg.norm(ee[:2] - est[:2]))
            speed = float(np.linalg.norm(v))
            if phase != 'intercept' or tick % 5 == 0:
                tr = self.trial_meta.setdefault('track_trace', [])
                tr.append({'k': tick, 'ph': phase[0], 'src': src or '-',
                           'est': [round(float(x), 3) for x in est[:2]],
                           'ee': [round(float(x), 3) for x in ee],
                           'd': round(d, 3),
                           'v': [round(float(x) * 100, 1) for x in v],
                           'ap': (round(read_aperture(self.data, addrs), 4)
                                  if addrs else None)})
                if len(tr) > 600:
                    del tr[0]
            if phase == 'intercept':
                aim = est[:2] + v * T_amb
                tgt = np.array([aim[0], aim[1], z_grasp])
                go(tgt, True)
                if (float(np.linalg.norm(ee[:2] - aim)) < 0.012
                        and abs(ee[2] - z_grasp) < 0.02):
                    phase = 'wait'; d_prev = None; away = 0; wait_t0 = None
                    log.append({'tick': tick, 'event': 'parked',
                                'v_cm_s': [round(float(x) * 100, 1) for x in v],
                                'd_cm': round(d * 100, 1)})
            elif phase == 'wait':
                if wait_t0 is None:
                    wait_t0 = t
                go(np.array([ee[0], ee[1], z_grasp]), True)
                tta = d / speed if speed > 0.005 else 99.0
                if t - wait_t0 > 6.0:
                    phase = 'intercept'; wait_t0 = None
                    if speed < 0.005:
                        # Parked beside something that never moves: a twin
                        # instance. Drop it and re-acquire.
                        blacklist.append(est[:2].copy())
                        last_pos = None; last_t = None; v = np.zeros(2)
                        src_prev = None; last_agent = None
                        log.append({'tick': tick, 'event': 'stale_lock',
                                    'd_cm': round(d * 100, 1)})
                    else:
                        log.append({'tick': tick, 'event': 'wait_timeout',
                                    'd_cm': round(d * 100, 1)})
                    d_prev = d
                    continue
                if d < 0.012 or tta < 0.35:
                    phase = 'close'; close_ticks = 0; ap_prev = None; stalled = 0
                    log.append({'tick': tick, 'event': 'close',
                                'd_cm': round(d * 100, 1), 'tta_s': round(tta, 2)})
                elif d_prev is not None and d > d_prev + 0.004:
                    away += 1
                    if away >= 3 and d > 0.04:
                        phase = 'intercept'
                        log.append({'tick': tick, 'event': 'missed_pass',
                                    'd_cm': round(d * 100, 1)})
                else:
                    away = 0
                d_prev = d
            else:  # close: jaws shut while xy keeps following the object
                aim = est[:2] + v * 0.1
                go(np.array([aim[0], aim[1], z_grasp]), False)
                close_ticks += 1
                ap = read_aperture(self.data, addrs)
                if ap is not None:
                    if ap_prev is not None and abs(ap - ap_prev) < 0.0005:
                        stalled += 1
                    else:
                        stalled = 0
                    ap_prev = ap
                    if stalled >= 2 and ap > 0.006:
                        log.append({'tick': tick, 'event': 'held',
                                    'aperture_m': round(ap, 4)})
                        self.holding = True
                        self.last_pick_label = label
                        self._tracked_hold = True
                        try:
                            self._shift_binding(key, est)
                        except Exception:
                            pass
                        return True
                    if ap < 0.004 or close_ticks > 12:
                        log.append({'tick': tick, 'event': 'empty_close',
                                    'aperture_m': round(ap, 4)})
                        if retries <= 0:
                            return False
                        retries -= 1
                        for _ in range(3):
                            go(np.array([ee[0], ee[1], z_grasp + hover_h]), True)
                        phase = 'intercept'
        log.append({'tick': max_ticks, 'event': 'timeout', 'phase': phase})
        return False

    # Pre-primitive scene diff + in-flight retarget (cfg.scene_diff)

    @staticmethod
    def _action_target_label(act: dict) -> str:
        params = act.get('params', {}) or {}
        for k in ('keypoint_label', 'target_label', 'label',
                  'pick_label', 'place_label'):
            v = params.get(k)
            if v:
                return str(v)
        return ''

    def _current_positions(self) -> dict:
        """
        Same-source 'current' map for the scene diff.

        Fresh agentview observations from the last merge where available;
        the agentview REFERENCE (not the fused binding) where the capture
        produced no evidence for a label - no evidence means HOLD (delta
        0), never 'missing': a label the camera cannot see right now is
        the telemetry layer's problem, not grounds to abort (mirrors the
        self-occlusion rule).
        """
        out: dict = {}
        for k in self.det_map:
            if k in self._fresh_dets:
                out[k] = np.asarray(self._fresh_dets[k].position_3d,
                                     dtype=float)
            elif k in self._fresh_out_of_gate:
                out[k] = np.asarray(
                    self._fresh_out_of_gate[k].position_3d, dtype=float)
            else:
                ref = self.agent_ref.get(k)
                if ref is None:
                    d = self.det_map.get(k)
                    p = (getattr(d, 'position_3d', None)
                         if d is not None else None)
                    ref = np.asarray(p, dtype=float) if p is not None else None
                out[k] = ref
        return out

    def _reference_positions(self) -> dict:
        """
        Same-source 'bound' map for the scene diff: agentview refs,
        falling back to the fused binding for wrist-only labels.
        """
        out: dict = {}
        for k in self.det_map:
            ref = self.agent_ref.get(k)
            if ref is None:
                ref = self.plan_binding.get(k)
            out[k] = (np.asarray(ref, dtype=float)
                      if ref is not None else None)
        return out

    def _run_scene_diff(self, target_label: str, *, context: str,
                          action_idx: Optional[int] = None):
        """
        Compute + record one scene diff, SAME-SOURCE.

        Both sides of the diff are agentview centroids (the ref each
        label was bound at vs the fresh observation), so a systematic
        wrist-vs-agentview offset can never read as motion.
        Labels the diff attributed to the robot itself have their
        binding delta-shifted (their motion is expected, so the binding
        tracks it).  Returns the SceneDiff.
        """
        cfg = self.cfg
        sd = compute_scene_diff(
            self._reference_positions(), self._current_positions(),
            target_label=target_label or None,
            held_label=self.last_pick_label if self.holding else None,
            gripper_pos=self._ee(),
            moved_threshold_m=float(getattr(
                cfg, 'scene_diff_moved_cm', 1.5)) / 100.0,
            self_radius_m=float(getattr(
                cfg, 'scene_diff_self_radius_cm', 12.0)) / 100.0)
        rec = sd.to_meta()
        rec['context'] = context
        if action_idx is not None:
            rec['action_idx'] = action_idx
        if self._sticky_flags:
            # Instance-association outcomes from the most recent merge
            # (e.g. 'ambiguous': twin instances kept on previous binding).
            rec['sticky'] = dict(self._sticky_flags)
        self.trial_meta.setdefault('scene_diffs', []).append(rec)
        for lbl, ld in sd.labels.items():
            # Self-caused motion with actual fresh evidence: track it by
            # delta-shifting the binding.  Evidence-free SELF verdicts
            # (held / occluded labels) leave the binding alone.
            if (ld.status == STATUS_SELF and ld.current_pos is not None
                    and lbl in self._fresh_dets):
                self._shift_binding(lbl, ld.current_pos)
        return sd

    def _handle_target_diff(self, sd, *, context: str) -> bool:
        """
        Retarget-or-abort decision for a diff whose target moved/vanished.

        Returns True to continue execution (ok / retargeted), False when
        the executor must fall through to recovery (self.abort set).
        """
        cfg = self.cfg
        if sd.target_status == STATUS_MOVED:
            delta_cm = sd.target_delta_m * 100.0
            # Pre-close occlusion floor: with the gripper over the target,
            # the agentview centroid shifts 2.3 to 3.8 cm on UNPERTURBED
            # trials (measured: phantom retargets at pre_close collapsed
            # the adaptive arm's 0 cm controls to 9/45 against the
            # baseline's 27/45). Below the floor at pre_close, trust the
            # standing bind; a true move at closure is caught by the
            # telemetry EMPTY_CLOSE vote and its regrasp. The wrist camera
            # is the principled fix on hardware.
            if (context == 'pre_close'
                    and delta_cm < float(getattr(
                        cfg, 'pre_close_retarget_min_cm', 4.0))):
                if cfg.verbose:
                    print(f"[Retarget] suppressed at pre_close: "
                          f"{sd.target_key} delta {delta_cm:.1f}cm is "
                          f"below the occlusion floor")
                return True
            if delta_cm < float(getattr(cfg, 'retarget_max_cm', 10.0)):
                # In-flight retarget: shift the authoritative binding
                # (and the det_map entry the handler reads at dispatch)
                # by the SAME-SOURCE delta so a wrist-fused bind keeps
                # its cross-camera correction.
                cur = sd.labels[sd.target_key].current_pos
                if cur is not None:
                    self._shift_binding(sd.target_key, cur)
                self.trial_meta.setdefault('retarget_events', []).append({
                    'label': sd.target_key, 'delta_cm': round(delta_cm, 2),
                    'context': context, 't_flag': sd.t,
                    't_resume': time.time()})
                if cfg.verbose:
                    print(f"[Retarget] {sd.target_key} moved "
                          f"{delta_cm:.1f}cm (< {cfg.retarget_max_cm}cm); "
                          f"re-aimed at fresh detection ({context})")
                return True
            self.abort = {'reason': 'target_moved_beyond_retarget',
                            'label': sd.target_key,
                            'delta_cm': round(delta_cm, 2),
                            'context': context, 't_flag': sd.t,
                            't': time.time()}
        elif sd.target_status == STATUS_MISSING:
            self.abort = {'reason': 'target_missing',
                            'label': sd.target_label, 'context': context,
                            't_flag': sd.t, 't': time.time()}
        if self.abort is not None:
            self.trial_meta['scene_abort'] = self.abort
            return False
        return True

    def _pre_primitive_gate(self, act_idx: int, act: dict) -> bool:
        """
        cfg.scene_diff gate run before every keypoint-referencing leaf.
        """
        if not getattr(self.cfg, 'scene_diff', False):
            return True
        if not self._action_uses_keypoint(act):
            return True
        label = self._action_target_label(act)
        sd = self._run_scene_diff(label, context='pre_primitive',
                                    action_idx=act_idx)
        return self._handle_target_diff(sd, context='pre_primitive')

    # Schedule-triggered captures (cfg.event_captures)

    def event_capture(self, event: str) -> Optional[dict]:
        """
        Capture + SAM3 verdict at a primitive-declared informative moment.

        Issues the SAM3 call immediately at the event (the point is
        verification latency measured from event to verdict), merges the
        fresh detections into det_map like the freshness gate, and
        records {event, t_event, t_capture, t_verdict} in
        trial_meta['capture_events'].  Returns the record (mutable - the
        caller may attach a verdict), or None when disabled/unavailable.
        """
        if not getattr(self.cfg, 'event_captures', False):
            return None
        if self.sam3 is None or not self.prompts_for_refresh:
            return None
        t_event = time.time()
        timing: dict = {}
        try:
            fresh = redetect_agentview(self.env, self.sam3,
                                         self.prompts_for_refresh, self.cfg,
                                         timing=timing,
                                         all_instances=self._sticky_enabled())
        except Exception as e:
            if self.cfg.verbose:
                print(f"[EventCapture:{event}] redetect failed: {e}")
            return None
        t_verdict = time.time()
        n_labels, n_updated = self._merge_fresh_detections(fresh)
        self._last_perception_t = t_verdict
        rec = {
            'event': event,
            't_event': t_event,
            't_capture': timing.get('t_capture', t_event),
            't_verdict': t_verdict,
            'latency_capture_s': round(
                timing.get('t_capture', t_event) - t_event, 4),
            'latency_verdict_s': round(t_verdict - t_event, 4),
            'n_labels': n_labels,
            'n_updated': n_updated,
        }
        if self._sticky_flags:
            rec['sticky'] = dict(self._sticky_flags)
        self.trial_meta.setdefault('capture_events', []).append(rec)
        if self.cfg.verbose:
            print(f"[EventCapture:{event}] verdict in "
                  f"{rec['latency_verdict_s']*1000:.0f}ms "
                  f"({rec['n_labels']} labels, {n_updated} updated)")
        return rec

    def pre_close_capture(self) -> None:
        """
        Pre-close informative moment: capture right before gripper
        closure, then (with cfg.scene_diff) re-check the pick target.

        A small move since binding re-approaches the fresh detection and
        continues (retarget); a large move / missing target sets
        self.abort so the grasp never closes on empty air and the runner
        recovers.
        """
        if getattr(self, '_tracked_hold', False):
            # The tracking node already closed on the object; a diff
            # against the plan-time binding would only re-approach it.
            self._tracked_hold = False
            return
        rec = self.event_capture('pre_close')
        if rec is None or not getattr(self.cfg, 'scene_diff', False):
            return
        label = self.last_pick_label
        if not label:
            return
        sd = self._run_scene_diff(label, context='pre_close')
        if not self._handle_target_diff(sd, context='pre_close'):
            return
        if sd.target_status == STATUS_MOVED:
            # Retargeted: the arm is hovering at the STALE grasp pose;
            # re-run the approach against the fresh detection before the
            # caller closes the jaws.
            _keypoint.handle(self, {'keypoint_label': label})
            # Stamp the actual resume time (the re-approach is part of
            # adaptation, not detection).
            evs = self.trial_meta.get('retarget_events') or []
            if evs:
                evs[-1]['t_resume'] = time.time()

    def post_release_capture(self) -> bool:
        """
        Post-release informative moment: capture right after release +
        small settle, verdict = did the released object land near the
        place target.  Returns True when the capture ran (caller then
        skips the blunt perception-invalidate).

        The verdict uses ONLY detections from THIS capture, never the
        retained det_map entries (the placed object is routinely occluded
        by its container right after release).

        Identity is decided by POSITIVE EVIDENCE, in two places only,
        because distance to the place target alone cannot tell a genuine
        failure from a lookalike:

        * a fresh instance within the sticky gate of the intended
          landing spot IS the placed object -> report its xy error;
        * else a fresh instance within the sticky gate of the pre-place
          ORIGIN is evidence the object never left -> ``ok: False``,
          reason ``still_at_origin``;
        * anything else (occluded in the container, or only far-away
          lookalikes) is not evidence about this object -> ABSTAIN.

        The failure branch keeps the verdict falsifiable: it can still
        report a real miss, but only from an observation it can place.
        """
        rec = self.event_capture('post_release')
        if rec is None:
            return False
        label = self.last_pick_label
        verdict: Optional[dict] = None
        if label:
            pos = None
            fresh_det = None
            at_origin = False
            if self._sticky_enabled():
                key = fuzzy_key(self.det_map, label) or label
                cands = self._fresh_groups.get(key) or []
                gate = self._sticky_gate_m()

                def _nearest(anchor):
                    if anchor is None or not cands:
                        return None, None
                    a = np.asarray(anchor, dtype=float)[:2]
                    best = min(cands, key=lambda c: float(np.linalg.norm(
                        np.asarray(c.position_3d, float)[:2] - a)))
                    d = float(np.linalg.norm(
                        np.asarray(best.position_3d, float)[:2] - a))
                    return best, d

                # BINDING (separate concern from the verdict): the
                # placed label's motion during this primitive is
                # SELF-CAUSED, so a large displacement is expected and
                # the bind-preserving out-of-gate rule does not apply.
                # Track the nearest fresh instance to the release point
                # whatever its distance, so recovery can still find an
                # object that slipped mid-transport.
                rebind_det, _ = _nearest(self.last_place_target)

                # VERDICT: assert only from an observation that can be placed.
                # 1. landed at the aim point?
                cand, dist = _nearest(self.last_place_target)
                if cand is not None and dist <= gate:
                    fresh_det, pos = cand, np.asarray(
                        cand.position_3d, dtype=float)
                else:
                    # 2. still sitting at the pre-place origin?  The hold
                    # suppressed association for this label, so the
                    # standing binding is exactly where it was picked up.
                    origin = self.plan_binding.get(key)
                    cand, dist = _nearest(origin)
                    if cand is not None and dist <= gate:
                        fresh_det = cand
                        pos = np.asarray(cand.position_3d, dtype=float)
                        at_origin = True
                if rebind_det is not None:
                    p = np.asarray(rebind_det.position_3d, dtype=float)
                    self.plan_binding[key] = p.copy()
                    self.agent_ref[key] = p.copy()
                    self.det_map[key] = rebind_det
            else:
                det = fuzzy_get_det(self.det_map, label)
                pos = (getattr(det, 'position_3d', None)
                       if det is not None else None)
                if pos is not None:
                    key = fuzzy_key(self.det_map, label) or label
                    self.plan_binding[key] = np.asarray(
                        pos, dtype=float).copy()
            if pos is not None:
                verdict = {'placed_label': label}
                if at_origin:
                    verdict['reason'] = 'still_at_origin'
                if self.last_place_target is not None:
                    err = float(np.linalg.norm(
                        np.asarray(pos, float)[:2]
                        - np.asarray(self.last_place_target, float)[:2]))
                    verdict['xy_error_m'] = round(err, 4)
                    verdict['ok'] = bool(err < 0.10)
            else:
                # Not freshly observed in this capture: the placed object
                # is typically occluded by its container, or the only
                # fresh instances are lookalikes that cannot be placed.
                # Abstain; a verdict without a located observation of
                # THIS object is not evidence.
                verdict = {'placed_label': label, 'ok': None,
                             'abstain': True,
                             'reason': 'not_freshly_detected'}
        rec['verdict'] = verdict
        return True

    # Gripper-telemetry grasp outcome (Watchdog / gObj analog)

    def close_gripper_with_telemetry(self, steps: int = 300) -> list[float]:
        """
        Close the gripper while sampling the finger aperture every step.

        Same actuation as ``_gripper(False, steps)`` but returns the
        aperture trace for outcome classification.
        """
        if self._finger_addrs is None:
            self._finger_addrs = find_finger_qpos_addrs(self.model)
        trace: list[float] = []
        for _ in range(steps):
            a = np.zeros(7)
            a[6] = 1.0
            if step_env(self.env, a):
                break
            w = read_aperture(self.data, self._finger_addrs)
            if w is not None:
                trace.append(w)
        return trace

    def record_grasp_outcome(self, cls: GraspClassification,
                               *, is_retry: bool = False) -> None:
        self.last_grasp_outcome = cls
        rec = cls.to_meta()
        rec['t'] = time.time()
        rec['label'] = self.last_pick_label
        rec['is_retry'] = bool(is_retry)
        self.trial_meta.setdefault('grasp_outcomes', []).append(rec)
        if self.cfg.verbose:
            print(f"[GraspTelemetry] {cls.outcome.value} "
                  f"(final={cls.final_aperture:.4f}m "
                  f"plateau={cls.plateau_aperture:.4f}m "
                  f"n={cls.n_samples}{' retry' if is_retry else ''})")

    def post_grasp_second_vote(self) -> None:
        """
        Camera second vote behind a telemetry EMPTY_CLOSE first vote.

        Captures immediately (event machinery), checks whether the pick
        target is still detected away from the hand; if so the grasp
        provably missed and one execution-layer local retry runs:
        reopen -> re-approach the fresh detection -> re-close, with the
        retry's telemetry outcome recorded alongside.
        Only active with cfg.event_captures (the vote IS a capture).
        """
        label = self.last_pick_label
        if not label:
            return
        # The vote must see the scene: drop the holding flag BEFORE the
        # capture so the sticky merge doesn't suppress the pick label as
        # held (telemetry already said the hand is empty).
        was_holding = self.holding
        self.holding = False
        rec = self.event_capture('post_grasp_verify')
        if rec is None:
            self.holding = was_holding
            return
        fresh_det = None
        if self._sticky_enabled():
            # Fresh observations from THIS capture only; the retained
            # det_map entry is the stale bind that just missed.
            key = fuzzy_key(self.det_map, label) or label
            fresh_det = (self._fresh_dets.get(key)
                         or self._fresh_out_of_gate.get(key))
            pos = (np.asarray(fresh_det.position_3d, dtype=float)
                   if fresh_det is not None else None)
        else:
            det = fuzzy_get_det(self.det_map, label)
            pos = getattr(det, 'position_3d', None) if det is not None else None
        missed = False
        if pos is not None:
            dist_to_hand = float(np.linalg.norm(
                np.asarray(pos, float) - self._ee()))
            missed = dist_to_hand > 0.10
            rec['verdict'] = {'label': label,
                                'dist_to_hand_m': round(dist_to_hand, 4),
                                'confirms_empty': missed}
        else:
            rec['verdict'] = {'label': label, 'confirms_empty': False,
                                'reason': 'not_detected'}
        if not missed:
            self.holding = was_holding
            return
        if self.cfg.verbose:
            print(f"[GraspTelemetry] camera confirms empty close on "
                  f"{label!r} - local regrasp retry")
        # The stale bind provably missed AND the camera sees the object
        # elsewhere: adopt the fresh observation outright (mask included)
        # before re-approaching; it is the only grounded position left.
        if self._sticky_enabled() and fresh_det is not None:
            key = fuzzy_key(self.det_map, label) or label
            p = np.asarray(fresh_det.position_3d, dtype=float).copy()
            self.plan_binding[key] = p
            self.agent_ref[key] = p.copy()
            self.det_map[key] = fresh_det
        from spark_bench.libero_pro.telemetry import classify_grasp_outcome
        self._gripper(True, steps=60)
        self.holding = False
        _keypoint.handle(self, {'keypoint_label': label})
        trace = self.close_gripper_with_telemetry(steps=300)
        cls2 = classify_grasp_outcome(trace)
        self.holding = True
        self.record_grasp_outcome(cls2, is_retry=True)


def execute_on_libero(env, score: dict, det_map: dict, cfg, *,
                       sam3=None, depth=None,
                       cam_pos=None, cam_mat=None, cam_fovy=None,
                       cam_w: int = 640, cam_h: int = 480,
                       instruction: str = '',
                       prompts_for_refresh: Optional[list] = None,
                       trial_meta: Optional[dict] = None) -> None:
    """
    Build a ``LiberoExecutor`` and run ``score`` against ``env``.

    ``prompts_for_refresh`` is consulted only when ``cfg.freshness_timeout_s``
    is set; it carries the same prompt list the runner used for the initial
    SAM3 call so the freshness gate can re-detect with the same vocabulary.

    ``trial_meta`` (when given) receives the executor's per-trial event
    records: grasp_outcomes, capture_events, scene_diffs,
    retarget_events, scene_abort.
    """
    executor = LiberoExecutor(env, cfg, det_map,
                               sam3=sam3, depth=depth,
                               cam_pos=cam_pos, cam_mat=cam_mat,
                               cam_fovy=cam_fovy,
                               cam_w=cam_w, cam_h=cam_h,
                               instruction=instruction,
                               prompts_for_refresh=prompts_for_refresh,
                               trial_meta=trial_meta)
    if executor._invalid:
        return
    executor.run(score)

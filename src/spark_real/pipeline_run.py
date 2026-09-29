"""
Full task-run orchestration for the SPARK real-robot pipeline.

Houses the closed-loop ``run_task`` driver that ties capture, SAM3
detection, Gemini planning, and robot execution together. RunMixin is
mixed into SPARKRealPipeline alongside ExecutionMixin, so the cross-mixin
self.plan() / self.execute() / self.evaluate_scene_progress() calls
resolve against the assembled class via MRO.
"""

import copy
import logging
import os
import re
import time
from datetime import datetime
from typing import List, Optional

import numpy as np
import yaml

from spark_real.control.success_predicates import UNVERIFIED
from spark_real.control.success_verifier import reset_run_state, task_success
from spark_real.pipeline_execution import (
    CACHE_PLAN_SOURCES,
    build_detection_details,
    verify_block_labels,
)
from spark_real.planning.plan_annotate import annotate_for_planner
from spark_real.pipeline_io import user_aborted
from spark_real.pipeline_types import TaskResult
from spark_real.routes import state as _routes_state

logger = logging.getLogger(__name__)


# Full capture -> detect -> plan -> execute orchestration.
# Mixed into the pipeline class.
class RunMixin:

    def run_task(
        self,
        instruction: str,
        prompts: List[str] = None,
        execute: bool = True,
        save_to_library: Optional[bool] = None,
    ) -> TaskResult:
        """
        Full pipeline: capture -> detect -> plan -> execute.

        When ``config.max_task_passes`` > 1 the pipeline runs closed-loop:
        after each behavior tree it re-perceives the scene, decides which
        targets still need handling (generic, any task), and re-plans over
        the fresh detections until the goal holds or the pass budget is
        spent. Single-pass behavior (max_task_passes == 1) is unchanged.
        """
        t0 = time.time()
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        logger.info("Task: %s", instruction)

        # Clear the previous run's verdict HERE, not in execute(). A dry run,
        # a no-robot run or a capture failure never reaches execute(), and the
        # verdict read at the end of this method would then be the LAST run's
        # -- reporting a stale `pass` as this run's success.
        reset_run_state(getattr(self, "_executor", None))

        # Capture
        logger.info("[1/5] Capturing images...")
        with self.activity("capturing"):
            captures = self.capture()
        if not captures:
            return TaskResult(
                instruction=instruction,
                timestamp=timestamp,
                detections=[],
                plan={},
                execution_results=[],
                success=False,
                duration=time.time() - t0,
            )

        saved_paths = {}
        if self.config.save_captures:
            saved_paths = self._save_captures(captures, timestamp)

        # Detect: two-phase approach:
        #   Phase 1: If no prompts given, ask Gemini to look at the scene
        #            and decide what to detect.
        #   Phase 2: Run SAM3 with those prompts, then send annotated
        #            results back to Gemini for planning.
        # Collect EVERY fixed view, not the first one that exists. A single
        # camera foreshortens: an elongated tool lying away from that image
        # plane reads far rounder than it is. The API takes several images in
        # one request, so send all and let the model cross-check.
        # scene_cam stays the primary (first) view, because the
        # RoboInter box space and the annotation overlay are defined against
        # exactly one camera.
        scene_img_raw = None
        scene_cam = None
        scene_raw_views = []
        for cam_name in ["birdview", "sideview"]:
            cam_data = captures.get(cam_name)
            if cam_data and cam_data.get("rgb") is not None:
                scene_raw_views.append((cam_name, cam_data["rgb"]))
                if scene_img_raw is None:
                    scene_img_raw = cam_data["rgb"]
                    scene_cam = cam_name

        use_multi_instance = False

        # Resolve the cached BT ONCE, up front. Two things ride on it:
        #   - the SAM3 detection prompts are harvested from the tree's own
        #     labels, which skips the Gemini prompt-generation call so a
        #     cache-only run is genuinely offline, and
        #   - plan() below is handed the SAME match, so the harvest and the
        #     executed tree can never disagree about which BT this run is.
        # In cache_only mode a miss raises BTCacheMiss out of run_task
        # rather than silently falling through to the planner.
        _plan_mode = self.plan_mode()
        cached_match = None
        if self._bt_library is not None:
            cached_match = self.resolve_cached_bt(instruction)
        if prompts is None and cached_match is not None:
            cached_instr = cached_match.entry.instruction
            cached_score = cached_match.score
            if cached_score is not None:
                # BTs carry instance-suffixed labels like "knife handle 1",
                # "knife handle 2". SAM3 wants the de-instanced prompt
                # ("knife handle") and produces those numbered labels
                # itself in multi-instance mode. Strip trailing " <N>"
                # and dedupe while preserving insertion order so the
                # prompt list is deterministic.
                seen = set()
                cached_labels = []

                def _harvest(node):
                    if not isinstance(node, dict):
                        return
                    p = node.get("params") or {}
                    # Pull any label-shaped param so the harvested
                    # prompt set covers every label the BT will look up
                    # at runtime (place_in_slot's container_label, e.g.
                    # "tray"), not just the grasp's keypoint_label.
                    for k in (
                        "keypoint_label",
                        "container_label",
                        "target_label",
                        "area_label",
                        "source_label",
                    ):
                        lbl = p.get(k)
                        if not lbl:
                            continue
                        base = re.sub(r"\s+\d+$", "", lbl).strip()
                        if base and base not in seen:
                            seen.add(base)
                            cached_labels.append(base)
                    for child in node.get("children") or []:
                        _harvest(child)

                _harvest((cached_score or {}).get("tree", {}))
                # Predicate labels sit outside the tree. Without them a
                # cache-served run detects only what it manipulates, and the
                # verifier abstains in every camera -> `unverified` forever.
                for lbl in sorted(verify_block_labels(cached_score)):
                    base = re.sub(r"\s+\d+$", "", lbl).strip()
                    if base and base not in seen:
                        seen.add(base)
                        cached_labels.append(base)
                if cached_labels:
                    prompts = cached_labels
                    use_multi_instance = True  # treat as batch (matches Gemini path)
                    logger.info(
                        "[2/5] cache %s: harvested prompts %s from cached BT "
                        "(matched %r) -- no prompt-generation LLM call",
                        cached_match.source,
                        prompts,
                        cached_instr,
                    )
        # Task-prompt registry (configs/tasks/*.yaml): the task's frozen
        # perception contract -- pinned vocabulary, declared object counts,
        # and a deterministic geometric instance ordering. Sits ahead of BOTH
        # the regex extractor and Gemini because it is the only prompt source
        # that is identical run to run; generate_prompts runs at temperature
        # 0.3, so the prompt VOCABULARY itself drifts, which is exactly the
        # non-repeatability a demo corpus must not have.
        #
        # A miss is not an error: unregistered instructions fall through to
        # the paths below unchanged.
        task_spec = self.task_spec(instruction)
        # Tier-2 success override (spec section 1.5). Set even when the task supplies no
        # prompts, and CLEARED otherwise so the previous task's predicate does
        # not leak into this one. The planner's own `verify:` still wins.
        if self._executor is not None:
            self._executor._task_verify_block = (
                dict(task_spec.verify) if task_spec is not None and task_spec.verify else None
            )
            if task_spec is not None and task_spec.verify:
                logger.info(
                    "[2/5] task registry %r: verify block from %s",
                    task_spec.task,
                    task_spec.source_path,
                )
        if prompts is None and task_spec is not None:
            prompts = list(task_spec.prompts)
            use_multi_instance = True
            logger.info(
                "[2/5] task registry %r: prompts %s (frozen in version "
                "control -- no LLM call)",
                task_spec.task,
                prompts,
            )
        if prompts is None and _plan_mode == "cache_only":
            # Strict offline: never fall through to the planner for prompts.
            # The regex extractor is a legitimate offline fallback; Gemini is
            # not. The task registry above covers the demo tasks without
            # either; this is the backstop for an unregistered instruction.
            prompts = self._extract_prompts(instruction)
            logger.info(
                "[2/5] plan_mode=cache_only: no cached BT labels to harvest; "
                "using regex prompts %s (no LLM call)",
                prompts,
            )
        if prompts is None:
            if self._planner is not None and scene_img_raw is not None:
                logger.info("[2/5] Asking Gemini for detection prompts...")
                try:
                    with self.activity("planning"):
                        prompts = self._planner.generate_prompts(instruction, scene_img_raw)
                    use_multi_instance = True  # Gemini prompts -> batch task -> multi-instance
                    logger.info("  Gemini prompts: %s", prompts)
                except Exception as e:
                    logger.warning(
                        "  Gemini prompt generation failed: %s, falling back to regex",
                        e,
                    )
                    prompts = self._extract_prompts(instruction)
            else:
                prompts = self._extract_prompts(instruction)

        # SAM3 detection: multi-instance only for Gemini-generated prompts (batch tasks).
        # User-provided prompts get top-1 per prompt (specific objects).
        logger.info(
            "[3/5] Detecting objects: %s (multi_instance=%s)",
            prompts,
            use_multi_instance,
        )
        with self.activity("detecting"):
            if task_spec is not None:
                # Registered task: detect under its contract. Adds the count
                # gate (with the relax-then-alt-prompts escalation) and the
                # declared instance ordering, so "blue block 1" is the same
                # physical block every run and a cached BT keeps binding.
                # Raises PromptCountMismatch when the gate fails and the
                # task's policy is `abort` -- deliberately fatal, because a
                # mislabeled episode poisons a training set silently.
                (
                    detections,
                    all_detections,
                    _resolution,
                    task_spec,
                    prompts,
                ) = self.detect_for_task(
                    captures, instruction, spec=task_spec, prompts=prompts
                )
            else:
                all_detections = self.detect(
                    captures, prompts, multi_instance=use_multi_instance
                )
                detections = self.merge_detections(all_detections)
            # Enrich container-like detections with slot poses + world-
            # frame major-axis orientation so the place_in_slot primitive
            # can drop heterogeneous items into distinct slots with
            # the gripper yaw aligned to the container's long axis.
            try:
                self._enrich_with_slots(detections, captures)
            except Exception as _e:
                logger.warning("slot enrichment failed (continuing without slots): %s", _e)
        logger.info(
            "  Found %d unique objects (%d total across cameras)",
            len(detections),
            len(all_detections),
        )
        for d in detections:
            pos_str = (
                f"({d.position_3d[0]:.3f}, {d.position_3d[1]:.3f}, {d.position_3d[2]:.3f})"
                if d.position_3d is not None
                else "N/A"
            )
            logger.info(
                "  - %s: conf=%.3f, pos=%s, cam=%s",
                d.label,
                d.confidence,
                pos_str,
                getattr(d, "camera", "?"),
            )

        # Plan: send Gemini the annotated image + detection details
        # so it can filter false positives and generate the plan.
        logger.info("[4/5] Generating plan...")
        scene_img_annotated = None
        if scene_img_raw is not None:
            cam_dets = [d for d in all_detections if getattr(d, "camera", None) == scene_cam]
            # Planner-specific overlay: adds the OBB rectangle and major-axis
            # arrow on top of the tint+contour, so the model can SEE whether an
            # object has a meaningful long axis. _annotate_image stays as-is
            # for the UI and the detection routes.
            scene_img_annotated = annotate_for_planner(
                scene_img_raw, cam_dets if cam_dets else detections
            )

        # Annotate each view against ITS OWN camera's detections -- an overlay
        # drawn from another camera's pixels would be worse than no overlay.
        scene_views = []
        for _cam, _raw in scene_raw_views:
            _dets = [d for d in all_detections if getattr(d, "camera", None) == _cam]
            if not _dets:
                continue
            try:
                scene_views.append((_cam, annotate_for_planner(_raw, _dets)))
            except Exception as _exc:  # noqa: BLE001
                logger.warning("[plan] could not annotate %s: %s", _cam, _exc)
        if len(scene_views) > 1:
            logger.info(
                "[plan] sending %d camera views to the planner: %s",
                len(scene_views),
                ", ".join(c for c, _ in scene_views),
            )
        elif scene_img_annotated is not None:
            scene_views = [(scene_cam, scene_img_annotated)]

        detection_details = build_detection_details(detections)

        with self.activity("planning"):
            score = self.plan(
                instruction,
                detections,
                scene_image=(scene_views if scene_views else scene_img_annotated),
                detection_details=detection_details,
                scene_camera=scene_cam,
            )
        plan_source = getattr(self, "_last_plan_source", None) or "llm"
        bt_hash = getattr(self, "_last_bt_hash", None)
        logger.info(
            "  Plan (source=%s, bt=%s):\n%s",
            plan_source,
            bt_hash or "-",
            yaml.dump(score, default_flow_style=False),
        )

        # Execute. Normally we require self._robot to be present.
        # EXCEPTION: when SPARK_EXECUTOR_BACKEND=osc, the OSC subprocess
        # owns the FCI and self._robot is None (set by
        # /api/control/osc/start). The executor's move_linear is routed
        # through OSC via HTTP in that mode, so execution should still
        # proceed even without a franky driver; skipping it would silently
        # no-op and report success with 0 steps.
        _server_state = _routes_state

        _osc_mode = (
            getattr(_server_state, "osc_executor_proc", None) is not None
            and _server_state.osc_executor_proc.poll() is None
            and os.environ.get("SPARK_EXECUTOR_BACKEND", "").lower() == "osc"
        )
        exec_results = []
        if execute and (self._robot is not None or _osc_mode):
            if _osc_mode:
                logger.info("[5/5] Executing via OSC backend (panda-py subprocess)...")
            else:
                logger.info("[5/5] Executing on robot...")
            exec_results = self.execute(score, detections)
        else:
            logger.info("[5/5] Skipping execution (dry run or no robot)")

        # Closed-loop outer passes (generic, all families). Re-perceive the
        # scene, decide what targets remain unhandled, re-plan over the
        # fresh detections, and re-execute until the goal holds or the pass
        # budget is spent. Already-handled objects stay closed: the
        # destination filter in execute() drops items inside the receptacle
        # and the executor's _placed_labels persist across passes, so the
        # re-plan never re-visits a node already taken care of. Newly
        # perturbed objects (added utensil, swapped receptacle) are picked
        # up because each pass detects fresh.
        all_exec_results = list(exec_results)
        max_passes = max(1, int(getattr(self.config, "max_task_passes", 1)))
        did_execute = bool(exec_results) and execute and (self._robot is not None or _osc_mode)
        last_score = score
        prev_unhandled = None
        stale = 0
        pass_idx = 1
        while did_execute and pass_idx < max_passes:
            if getattr(self._executor, "_abort", False):
                logger.info("[closed-loop] abort flag set; stopping passes")
                break
            settle = float(getattr(self.config, "closed_loop_settle_s", 0.8))
            if settle > 0:
                time.sleep(settle)
            handled = set(getattr(self._executor, "_placed_labels", set()) or set())
            with self.activity("detecting"):
                progress = self.evaluate_scene_progress(last_score, prompts, handled_labels=handled)
            unhandled = progress["unhandled"]
            if not unhandled:
                logger.info(
                    "[closed-loop] goal holds after pass %d " "(handled=%s, destinations=%s)",
                    pass_idx,
                    progress["handled"],
                    progress["destinations"],
                )
                break
            # No-progress guard: if the unhandled count is not shrinking,
            # stop rather than loop on an unreachable or mis-detected target.
            n = len(unhandled)
            if prev_unhandled is not None and n >= prev_unhandled:
                stale += 1
            else:
                stale = 0
            prev_unhandled = n
            if stale >= 2:
                logger.warning(
                    "[closed-loop] no progress over 2 passes " "(%d unhandled: %s); stopping",
                    n,
                    unhandled,
                )
                break
            logger.info(
                "[closed-loop] pass %d/%d: %d target(s) still "
                "unhandled %s -> re-planning over fresh scene",
                pass_idx + 1,
                max_passes,
                n,
                unhandled,
            )
            dets2 = progress["detections"]
            if not dets2:
                break
            details2 = build_detection_details(dets2)
            # Re-plan. When the first pass was cache-served, reuse the SAME
            # tree rather than re-resolving: the single-shot UI pin was
            # already consumed by pass 1, so a re-resolve here would fall
            # through to the LLM mid-run.
            if bt_hash and plan_source in CACHE_PLAN_SOURCES:
                score2 = copy.deepcopy(self._bt_library.get_score(bt_hash) or last_score)
                logger.info(
                    "[closed-loop] reusing cache-served BT %s for pass %d "
                    "(no re-resolve, no LLM call)",
                    bt_hash,
                    pass_idx + 1,
                )
            else:
                with self.activity("planning"):
                    score2 = self.plan(instruction, dets2, detection_details=details2)
            logger.info(
                "[closed-loop] re-plan:\n%s",
                yaml.dump(score2, default_flow_style=False),
            )
            results2 = self.execute(score2, dets2)
            all_exec_results.extend(results2)
            last_score = score2
            pass_idx += 1
            if not results2:
                break

        exec_results = all_exec_results
        goal_unhandled = None
        if max_passes > 1 and did_execute:
            logger.info(
                "[closed-loop] completed %d pass(es), %d total actions",
                pass_idx,
                len(exec_results),
            )
            # Final ground-truth check: the last pass executed actions that were
            # never re-perceived, so confirm the goal physically holds before
            # reporting success. A target still outside the destination means
            # the task failed regardless of what the per-action verify claimed.
            settle = float(getattr(self.config, "closed_loop_settle_s", 0.8))
            if settle > 0:
                time.sleep(settle)
            handled = set(getattr(self._executor, "_placed_labels", set()) or set())
            with self.activity("detecting"):
                final_progress = self.evaluate_scene_progress(
                    last_score, prompts, handled_labels=handled
                )
            goal_unhandled = final_progress["unhandled"]
            if goal_unhandled:
                logger.warning(
                    "[closed-loop] goal NOT met: %d target(s) still outside " "destination %s",
                    len(goal_unhandled),
                    goal_unhandled,
                )
            else:
                logger.info("[closed-loop] goal met: all targets handled")

        # Task success is the verifier's verdict and nothing else. An
        # ExecutionResult says a primitive RAN, not that the task was
        # achieved: a "last action succeeded" fallback banks a fork dropped
        # beside the tray as a success.
        outcome = getattr(self._executor, "verify_outcome", None) if self._executor else None
        success = task_success(outcome)
        verify_status = outcome.status if outcome is not None else UNVERIFIED
        # Closed-loop ground truth can only ever REMOVE a pass: a target still
        # outside the destination means the task did not succeed, whatever the
        # predicate said.
        if goal_unhandled:
            if success:
                logger.warning(
                    "Verifier said pass but %d target(s) are still outside the "
                    "destination %s; downgrading to fail",
                    len(goal_unhandled),
                    goal_unhandled,
                )
            success = False
            verify_status = "fail"

        # LLM REPLAN -- THE TERMINAL RUNG, at most once per task. The recovery
        # ladder below this point is deliberately LLM-free (re-bind ->
        # search_keypoint -> predicate-gated replay); this rung exists because
        # a ladder that ends in "give up" wastes the one planner call that
        # could have fixed a genuinely wrong PLAN. The affordable place for an
        # in-loop VLM call (cf. ReKep, 2409.01652) is HERE, after everything
        # cheap has been exhausted, with the failure evidence in hand.
        # Gated on real evidence (verify FAIL or the
        # closed-loop goal check), never on an abstain: replanning on no
        # evidence is how a held object gets dropped on a guess.
        replan_used = False
        if (
            not success
            and bool(getattr(self.config, "llm_last_resort", True))
            and (verify_status == "fail" or goal_unhandled)
            and self._planner is not None
        ):
            try:
                replan_used = True
                why = (
                    f"verify={verify_status}"
                    + (f"; unplaced targets: {sorted(goal_unhandled)}"
                       if goal_unhandled else "")
                    + (f"; reason: {outcome.reason}"
                       if outcome is not None and outcome.reason else "")
                )
                logger.warning(
                    "[llm-last-resort] all LLM-free recovery exhausted (%s); "
                    "consulting the planner ONCE with the failure evidence",
                    why,
                )
                retry_instruction = (
                    f"{instruction}\n\nIMPORTANT: a previous attempt at this "
                    f"task just FAILED ({why}). The scene below is the CURRENT "
                    "state -- some steps may already be partially done. Plan "
                    "only what remains, and prefer a DIFFERENT approach to "
                    "whatever failed."
                )
                with self.activity("detecting"):
                    captures2 = self.capture()
                    dets2 = self.merge_detections(
                        self.detect(captures2, prompts)
                    )
                cam2 = "birdview" if "birdview" in captures2 else next(iter(captures2))
                raw_views = [
                    (c, captures2[c].get("rgb"))
                    for c in ("birdview", "sideview")
                    if c in captures2 and captures2[c].get("rgb") is not None
                ] or [(cam2, captures2[cam2].get("rgb"))]

                # STEP 1 -- ADJUDICATE before assuming failure. The verifier's
                # false negatives are a first-class failure mode (a correct
                # placement can be reported failed), so the terminal rung's
                # first question is "is this actually
                # fine?", asked on CLEAN frames -- overlay paint sits exactly
                # on the objects whose state is in dispute. A confident YES
                # here means no motion at all: the worst outcome of replanning
                # a completed task is the robot un-completing it.
                proprio = getattr(self._executor, "_release_proprio", None) or {}
                proprio_hard_fail = proprio.get("verdict") == "fail"
                judged = self._planner.assess_completion(
                    instruction, raw_views, evidence=why
                )
                if (
                    judged["complete"]
                    and judged["confidence"] >= 0.7
                    and not proprio_hard_fail
                ):
                    # The arm's own evidence outranks the LLM: a hard proprio
                    # fail (jaws never opened / commanded tilt dropped) means
                    # the pretty picture is of a task that did not execute.
                    success = True
                    verify_status = "pass"
                    logger.warning(
                        "[llm-last-resort] Gemini adjudicated COMPLETE "
                        "(conf %.2f): %s -- overriding verify=%s, no replan",
                        judged["confidence"], judged["reason"], why,
                    )
                else:
                    if judged["complete"]:
                        logger.info(
                            "[llm-last-resort] adjudication said complete but "
                            "was overruled (conf %.2f, proprio_fail=%s)",
                            judged["confidence"], proprio_hard_fail,
                        )
                    # STEP 2 -- REPLAN. Both image forms go to the planner:
                    # the CLEAN frame so nothing is painted over, and the
                    # ANNOTATED frame because the plan must reference the
                    # numbered keypoints -- seeing the failure and naming the
                    # fix are different needs.
                    scene_views3 = list(raw_views)
                    if raw_views[0][1] is not None:
                        scene_views3.append(
                            (raw_views[0][0] + " (annotated keypoints)",
                             annotate_for_planner(raw_views[0][1], dets2))
                        )
                    with self.activity("planning"):
                        score3 = self.plan(
                            retry_instruction, dets2, scene_image=scene_views3,
                            scene_camera=raw_views[0][0],
                            # a fresh sample, never the cached tree that failed
                            temperature=0.4,
                        )
                    results3 = self.execute(score3, dets2)
                    outcome = getattr(self._executor, "verify_outcome", None)
                    success = task_success(outcome)
                    verify_status = (
                        outcome.status if outcome is not None else UNVERIFIED
                    )
                    logger.info(
                        "[llm-last-resort] replan executed: verify=%s",
                        verify_status,
                    )
                    exec_results.extend(results3 or [])
            except Exception as exc:  # noqa: BLE001 - the rung must not mask the original failure
                logger.warning("[llm-last-resort] replan failed: %s", exc)
        duration = time.time() - t0

        result = TaskResult(
            instruction=instruction,
            timestamp=timestamp,
            detections=[
                {
                    "label": d.label,
                    "confidence": d.confidence,
                    "position_3d": (
                        d.position_3d.tolist()
                        if isinstance(d.position_3d, np.ndarray)
                        else d.position_3d
                    ),
                    "centroid_2d": d.centroid_2d,
                    "depth_meters": d.depth_meters,
                    "orientation_angle": d.orientation_angle,
                    "aspect_ratio": d.aspect_ratio,
                    "obb_minor_m": float(getattr(d, "obb_minor_m", 0.0) or 0.0),
                }
                for d in detections
            ],
            plan=score,
            execution_results=[
                {
                    "action": r.action_type,
                    "success": r.success,
                    "message": r.message,
                    "duration": r.duration,
                }
                for r in exec_results
            ],
            success=success,
            duration=duration,
            captures=saved_paths,
        )

        # Provenance for the recorded-episode metadata (the demo recorder
        # reads these into metadata.spark) and for the /scores page.
        result.plan_source = plan_source
        result.bt_hash = bt_hash
        result.label_resolutions = dict(getattr(self, "_last_label_resolutions", None) or {})
        # What the planner proposed in image space and what survived the
        # bounded-correction guard. Empty list when RoboInter is off.
        result.robointer = list(getattr(self, "_last_robointer", None) or [])
        # The verdict itself, so /scores and the recorded episode can show WHY
        # a run passed or failed rather than just a bare boolean.
        result.verify = outcome.to_dict() if outcome is not None else None
        result.verify_status = verify_status

        self._task_history.append(result)
        self._save_result(result, timestamp)
        # Automatic success feedback. ``executed`` is the guard that keeps a
        # dry run out of the cache; ``plan_source``/``bt_hash`` route a
        # cache-served run to bump()+alias instead of minting a duplicate.
        self._maybe_save_bt(
            instruction,
            score,
            detections,
            success,
            save_to_library=save_to_library,
            executed=bool(exec_results),
            plan_source=plan_source,
            bt_hash=bt_hash,
            user_aborted=user_aborted(exec_results),
            verify_status=verify_status,
        )
        logger.info(
            "Task complete (%.1fs, verify=%s, success=%s, plan_source=%s)",
            duration,
            verify_status,
            success,
            plan_source,
        )
        return result

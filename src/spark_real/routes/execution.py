# Execution routes: execute, execute_approved, run_bt, plan.

import asyncio
import logging
import os
import time
from pathlib import Path

import yaml
from fastapi import APIRouter
from fastapi.responses import JSONResponse

from spark_real.control.success_predicates import UNVERIFIED
from spark_real.control.success_verifier import reset_run_state, task_success
from spark_real.episode_recorder import EpisodeRecorder
from spark_real.perception.prompt_registry import PromptCountMismatch
from spark_real.pipeline_execution import build_detection_details
from spark_real.pipeline_io import user_aborted
from spark_real.planning.plan_annotate import annotate_for_planner
from spark_real.recording.schema import MODE_AUTONOMOUS
from spark_real.routes import state
from spark_real.routes.models import ExecuteRequest, PlanRequest
from spark_real.routes.vla_record import start_recorder, stop_recorder
from spark_real.routes.visualization import serialize_detections

logger = logging.getLogger("spark_server")
router = APIRouter()


def _pretask_home(pipeline):
    """
    Open gripper, clear reflex, go home. Best-effort.
    """
    try:
        pipeline._robot.open_gripper()
    except Exception as exc:
        logger.warning("pre-task gripper open failed: %s", exc)
    try:
        pipeline._robot.recover_from_errors()
    except Exception:
        pass
    for method in ("go_home", "move_home", "home"):
        if hasattr(pipeline._robot, method):
            try:
                getattr(pipeline._robot, method)()
                logger.info("Pre-task home complete (via %s)", method)
                return
            except Exception as exc:
                logger.warning("pre-task home (%s) failed: %s", method, exc)


def _robot_ready(pipeline):
    """
    Return (ok, error_msg). Verifies FCI is up and joint state readable.
    """
    if state.executor_running:
        return False, "an execution is already in progress"
    if state.viser_teleop_proc is not None and state.viser_teleop_proc.poll() is None:
        return False, (
            "viser teleop subprocess owns the FCI; stop it via "
            "POST /api/teleop/viser/stop before executing"
        )
    _osc_active = (
        state.osc_executor_proc is not None
        and state.osc_executor_proc.poll() is None
        and os.environ.get("SPARK_EXECUTOR_BACKEND", "").lower() == "osc"
    )
    if (
        state.osc_executor_proc is not None
        and state.osc_executor_proc.poll() is None
        and not _osc_active
    ):
        return False, ("OSC executor owns FCI but SPARK_EXECUTOR_BACKEND is not 'osc'")
    if pipeline is None:
        return False, "Pipeline not initialized"
    robot = getattr(pipeline, "_robot", None)
    if robot is None and not _osc_active:
        return False, "Robot driver not constructed yet"
    if robot is None:
        return True, None
    drv = robot
    for _ in range(4):
        if hasattr(drv, "_connected"):
            if not getattr(drv, "_connected", False):
                return False, "Robot driver reports DISCONNECTED"
            # Probe the ACTUAL link, not the flag: after a controller reboot
            # or network drop the flag stays True while every RTDE read
            # raises deep inside the executor. health_check clears the flag
            # on failure so the state is consistent afterwards.
            health = getattr(drv, "health_check", None)
            if callable(health) and not health():
                return False, (
                    "Robot RTDE link is down (health check failed); "
                    "reconnect via /api/connect_robot"
                )
            break
        drv = getattr(drv, "_robot", None)
        if drv is None:
            break
    try:
        if hasattr(robot, "get_joint_positions"):
            q = robot.get_joint_positions()
            if q is None or len(q) < 6:
                return False, "Robot joint state unavailable"
    except Exception as exc:
        return False, f"Robot read failed: {exc}"
    return True, None


@router.post("/api/execute_approved")
async def execute_approved(req: ExecuteRequest):
    """
    Execute with previously approved detections.
    """
    pipeline = state.pipeline
    if pipeline is None:
        return JSONResponse(
            status_code=400, content={"error": "Pipeline not initialized"}
        )
    ok, why = _robot_ready(pipeline)
    if not ok:
        return JSONResponse(
            status_code=400, content={"error": f"Cannot execute: {why}."}
        )
    if state.pending_detections is None:
        return JSONResponse(
            status_code=400,
            content={"error": "No pending detections; run detect_approve first"},
        )
    if not state.execute_lock.acquire(blocking=False):
        return JSONResponse(
            status_code=409, content={"error": "execution already in progress"}
        )

    with state.progress_lock:
        state.progress_log.clear()
    loop = asyncio.get_event_loop()

    def _exec_sync():
        t0 = time.time()
        instruction = req.instruction or state.pending_instruction
        detections = state.pending_detections
        captures = state.pending_captures
        # Only detect_approve fills pending_captures. A run whose detections
        # came from click/box prompts (the "Execute (keep detections)" path)
        # arrives with detections but NO frames.
        # The detections are the operator's work and must be kept; the frames
        # are just planner context, so grab fresh ones. capture() only reads
        # from already-streaming cameras -- no device open, no USB cycle.
        if captures is None:
            try:
                captures = pipeline.capture()
            except Exception as exc:  # noqa: BLE001 - planner can run blind
                logger.warning(
                    "execute_approved: no pending captures and a fresh "
                    "capture failed (%s); planning without a scene image",
                    exc,
                )
                captures = {}

        # This endpoint bypasses run_task, and a dry run never reaches
        # execute() either -- without this the verdict read below is the
        # PREVIOUS run's, so a dry run that moved nothing reports its pass.
        reset_run_state(getattr(pipeline, "_executor", None))

        scene_img = None
        scene_cam = None
        for cam_name in ["birdview", "sideview"]:
            cam_data = captures.get(cam_name)
            if cam_data and cam_data.get("rgb") is not None:
                scene_img = cam_data["rgb"]
                scene_cam = cam_name
                break
        if scene_img is not None:
            scene_img = annotate_for_planner(scene_img, detections)

        detection_details = build_detection_details(detections)

        # Tier-2 success override. This endpoint bypasses run_task, so without
        # this the task YAML's `verify:` block would apply on /api/execute but
        # silently not here. Cleared when the task has none, so the previous
        # task's predicate cannot leak in.
        _spec = pipeline.task_spec(instruction)
        if pipeline._executor is not None:
            pipeline._executor._task_verify_block = (
                dict(_spec.verify) if _spec is not None and _spec.verify else None
            )

        with pipeline.activity("planning"):
            score = pipeline.plan(
                instruction,
                detections,
                scene_image=scene_img,
                detection_details=detection_details,
                temperature=req.temperature,
                scene_camera=scene_cam,
            )

        if state.current_episode is not None:
            try:
                state.current_episode.discard()
            except Exception:
                pass
            state.current_episode = None

        rec = None
        video_paths = {}
        do_execute = not req.dry_run and pipeline._robot is not None
        # Silent-success guard: a real run that expected a robot but whose
        # driver/executor never constructed would otherwise return
        # exec_results=[] and report success having moved nothing. Gate on
        # the RobotProfile.no_robot flag (server.py sets it from --no-robot)
        # so genuine plan-only --no-robot runs stay valid.
        _profile = getattr(pipeline, "profile", None)
        _expected_robot = not bool(getattr(_profile, "no_robot", False))
        _driver_missing = pipeline._robot is None or pipeline._executor is None
        if not req.dry_run and _driver_missing and _expected_robot:
            return {
                "success": False,
                "error": "robot driver unavailable (perception-only boot)",
                "instruction": instruction,
                "detections": detection_details,
                "plan": score,
                "plan_yaml": yaml.dump(score, default_flow_style=False),
                "execution_results": [],
            }
        _episode_enabled = os.environ.get("SPARK_ENABLE_EPISODE_RECORDER", "0") == "1"
        if do_execute and _episode_enabled:
            try:
                rec = EpisodeRecorder(
                    root=Path(pipeline.config.output_dir).parent,
                    instruction=instruction,
                )
                rec.begin(pipeline._robot)
                state.current_episode = rec
            except Exception:
                rec = None
            try:
                pipeline.start_video_recording(label=instruction)
            except Exception:
                pass

        # Demonstration recording. Unlike the video recorder above this is a
        # TRAINING artifact, so a failure to start must not be swallowed --
        # a run that silently recorded nothing costs a full scene reset.
        if do_execute and req.record_demo:
            start_recorder(
                pipeline,
                task=instruction,
                prompt=instruction,
                mode=MODE_AUTONOMOUS,
            )

        if do_execute:
            _pretask_home(pipeline)
            state.executor_running = True
            try:
                exec_results = pipeline.execute(score, detections)
            except BaseException:
                # Never leave the recorder thread running past a failed
                # execute: it would keep appending frames to an episode
                # nobody closes, and block the next start with a 409.
                stop_recorder(success=False, discard=True)
                raise
            finally:
                state.executor_running = False
        else:
            exec_results = []

        if do_execute:
            try:
                vp = pipeline.stop_video_recording()
                if isinstance(vp, dict):
                    video_paths = {k: Path(v) for k, v in vp.items() if v is not None}
                elif vp:
                    video_paths = {"main": Path(vp)}
            except Exception:
                pass

        duration = time.time() - t0
        # The verifier's verdict, not "the last primitive returned True".
        _executor = getattr(pipeline, "_executor", None)
        outcome = getattr(_executor, "verify_outcome", None) if _executor else None
        success = task_success(outcome)
        verify_status = outcome.status if outcome is not None else UNVERIFIED
        verify_dict = outcome.to_dict() if outcome is not None else None

        # Close out the demonstration episode. Unconditional and last: it
        # no-ops when nothing is registered, and the episode must be finalized
        # even if the library bookkeeping below throws.
        demo_summary = stop_recorder(
            success=bool(success),
            spark={
                "bt_hash": getattr(pipeline, "_last_bt_hash", None),
                "plan_source": getattr(pipeline, "_last_plan_source", None),
            },
            verify=verify_dict,
        )

        pipeline._maybe_save_bt(
            instruction,
            score,
            detections,
            success,
            # Honor the request instead of hardcoding True. None falls through
            # to config.save_to_library (True by default).
            save_to_library=req.save_to_library,
            executed=bool(exec_results),
            plan_source=getattr(pipeline, "_last_plan_source", None),
            bt_hash=getattr(pipeline, "_last_bt_hash", None),
            user_aborted=user_aborted(exec_results),
            verify_status=verify_status,
        )

        if rec is not None:
            result_payload = {
                "success": success,
                "verify": verify_dict,
                "verify_status": verify_status,
                "duration": duration,
                "execution_results": [
                    {
                        "action": r.action_type,
                        "success": r.success,
                        "message": r.message,
                        "duration": r.duration,
                    }
                    for r in exec_results
                ],
            }
            try:
                rec.end(
                    score=score,
                    detections=detections,
                    result=result_payload,
                    video_paths=video_paths,
                )
            except Exception:
                pass

        return {
            "instruction": instruction,
            "success": success,
            # Tri-state verdict. `success` is exactly verify_status == "pass";
            # "unverified" is NOT a failure and must be shown as its own state.
            "verify_status": verify_status,
            "verify": verify_dict,
            "duration": duration,
            "detections": detection_details,
            "plan": score,
            "plan_yaml": yaml.dump(score, default_flow_style=False),
            "execution_results": [
                {
                    "action": r.action_type,
                    "success": r.success,
                    "message": r.message,
                    "duration": r.duration,
                }
                for r in exec_results
            ],
            "episode": rec.summary() if rec is not None else None,
            # Where the executed tree came from, and where the recorded
            # demonstration landed. None when record_demo was not set.
            "plan_source": getattr(pipeline, "_last_plan_source", None),
            "bt_hash": getattr(pipeline, "_last_bt_hash", None),
            "demo_episode": demo_summary,
        }

    try:
        return await loop.run_in_executor(None, _exec_sync)
    finally:
        state.execute_lock.release()


@router.post("/api/run_bt")
async def run_bt_yaml(body: dict):
    """
    Run a pre-cooked BT (YAML or dict) directly through the executor.
    """
    pipeline = state.pipeline
    if pipeline is None:
        return JSONResponse(
            status_code=400, content={"error": "Pipeline not initialized"}
        )
    plan = body.get("plan")
    if plan is None and body.get("yaml"):
        plan = yaml.safe_load(body["yaml"])
    if not plan:
        return JSONResponse(
            status_code=400, content={"error": "Provide 'yaml' or 'plan'"}
        )

    if not state.execute_lock.acquire(blocking=False):
        return JSONResponse(
            status_code=409, content={"error": "execution already in progress"}
        )

    detections = state.pending_detections or []
    loop = asyncio.get_event_loop()
    skip_prelude = bool(body.get("skip_prelude", False))

    def _run():
        with state.progress_lock:
            state.progress_log.clear()
        if not skip_prelude:
            _pretask_home(pipeline)
        state.executor_running = True
        try:
            exec_results = pipeline.execute(plan, detections)
        finally:
            state.executor_running = False
        return [
            {
                "action": r.action_type,
                "success": r.success,
                "message": r.message,
                "duration": r.duration,
            }
            for r in exec_results
        ]

    try:
        results = await loop.run_in_executor(None, _run)
    finally:
        state.execute_lock.release()
    return {"execution_results": results}


@router.post("/api/plan")
def generate_plan(req: PlanRequest):
    # Sync (def) handler: FastAPI runs it in the anyio threadpool. The body
    # does capture + SAM3 + Gemini inline -- seconds to minutes of blocking
    # work that would freeze the event loop (and with it /api/stop and the
    # camera stream) for the whole call.
    pipeline = state.pipeline
    if pipeline is None:
        return JSONResponse(
            status_code=400, content={"error": "Pipeline not initialized"}
        )
    with pipeline.activity("capturing"):
        captures = pipeline.capture()
    # The camera name travels with the image: the RoboInter extension states
    # each detection's observed 2D box against THIS frame, and a box read off
    # a different camera describes a different image.
    scene_cam, scene_img = next(
        ((name, c["rgb"]) for name, c in captures.items() if c.get("rgb") is not None),
        (None, None),
    )
    # Prompt source, in order of preference:
    #   1. whatever the caller asked for
    #   2. the task's frozen contract in configs/tasks/*.yaml -- pinned
    #      vocabulary, declared object counts, deterministic instance
    #      ordering, no LLM
    #   3. Gemini vision prompt-gen (temperature 0.3, so the vocabulary itself
    #      drifts run to run -- fine for a preview, not for collection)
    #   4. the regex extractor
    # Same source as detect_approve, so a plan preview detects with the same
    # vocabulary an actual run would; otherwise the preview's labels diverge
    # from the BT's and the preview is unexecutable.
    prompts = req.prompts
    spec = pipeline.task_spec(req.instruction) if prompts is None else None
    if spec is None:
        if prompts is None and pipeline._planner is not None and scene_img is not None:
            try:
                with pipeline.activity("planning"):
                    prompts = pipeline._planner.generate_prompts(
                        req.instruction,
                        scene_img,
                    )
            except Exception:
                prompts = pipeline._extract_prompts(req.instruction)
        elif prompts is None:
            prompts = pipeline._extract_prompts(req.instruction)

    resolution = None
    with pipeline.activity("detecting"):
        if spec is not None:
            try:
                detections, _all, resolution, spec, prompts = pipeline.detect_for_task(
                    captures, req.instruction, spec=spec
                )
            except PromptCountMismatch as exc:
                return JSONResponse(
                    status_code=422,
                    content={
                        "error": str(exc),
                        "instruction": req.instruction,
                        "reason": "prompt_count_mismatch",
                    },
                )
        else:
            detections = pipeline.detect(captures, prompts, multi_instance=True)
    # Parity with /api/execute: the model gets the SAME annotated image and the
    # SAME geometry, so a plan previewed here can make the same grasp-strategy
    # call.
    scene_img_annotated = (
        annotate_for_planner(scene_img, detections) if scene_img is not None else None
    )
    with pipeline.activity("planning"):
        score = pipeline.plan(
            req.instruction,
            detections,
            scene_image=scene_img_annotated,
            detection_details=build_detection_details(detections),
            temperature=req.temperature,
            scene_camera=scene_cam,
        )
    return {
        "instruction": req.instruction,
        "detections": serialize_detections(detections),
        "plan": score,
        "plan_yaml": yaml.dump(score, default_flow_style=False),
        "prompts": list(prompts or []),
        "task_spec": None if spec is None else spec.task,
        "detection_counts": None if resolution is None else resolution.counts,
        "plan_source": getattr(pipeline, "_last_plan_source", None),
        "bt_hash": getattr(pipeline, "_last_bt_hash", None),
    }


@router.post("/api/execute")
async def execute_task(req: ExecuteRequest):
    pipeline = state.pipeline
    if pipeline is None:
        return JSONResponse(
            status_code=400, content={"error": "Pipeline not initialized"}
        )
    ok, why = _robot_ready(pipeline)
    if not ok:
        return JSONResponse(
            status_code=400, content={"error": f"Cannot execute: {why}."}
        )
    if not state.execute_lock.acquire(blocking=False):
        return JSONResponse(
            status_code=409, content={"error": "execution already in progress"}
        )
    try:
        return await _execute_task_locked(pipeline, req)
    finally:
        state.execute_lock.release()


async def _execute_task_locked(pipeline, req: ExecuteRequest):
    """Body of /api/execute; caller holds state.execute_lock."""
    with state.progress_lock:
        state.progress_log.clear()
    loop = asyncio.get_event_loop()

    if req.record_video:
        try:
            await loop.run_in_executor(
                None,
                lambda: pipeline.start_video_recording(label=req.instruction),
            )
        except Exception:
            pass

    if not req.dry_run:
        await loop.run_in_executor(None, lambda: _pretask_home(pipeline))

    # Demonstration recording. This one call is the whole "SPARK generates
    # its own training demos" path: the recorder streams synchronized
    # image + proprio + commanded-action frames while the executor drives,
    # into the same schema the human teleop corpus uses.
    #
    # NOT wrapped in try/except: a training run that silently recorded
    # nothing costs a full manual scene reset to discover. Fail loudly and
    # early, before the arm moves.
    if req.record_demo and not req.dry_run:
        await loop.run_in_executor(
            None,
            lambda: start_recorder(
                pipeline,
                task=req.instruction,
                prompt=req.instruction,
                mode=MODE_AUTONOMOUS,
            ),
        )

    # Per-run closed-loop override: temporarily raise max_task_passes so a
    # single execute can re-perceive and re-plan over what remains. Restored
    # after the run so the server default is untouched.
    _prev_passes = getattr(pipeline.config, "max_task_passes", 1)
    if req.closed_loop_passes is not None:
        pipeline.config.max_task_passes = max(1, int(req.closed_loop_passes))
    _mismatch = None
    result = None
    if not req.dry_run:
        state.executor_running = True
    try:
        result = await loop.run_in_executor(
            None,
            lambda: pipeline.run_task(
                instruction=req.instruction,
                prompts=req.prompts,
                execute=not req.dry_run,
                save_to_library=req.save_to_library,
            ),
        )
    except PromptCountMismatch as exc:
        # A registered task saw the wrong number of objects and its policy is
        # `abort`. Deliberately fatal: a mislabeled episode poisons the
        # training set far worse than losing one episode to a re-run.
        _mismatch = exc
    finally:
        pipeline.config.max_task_passes = _prev_passes
        state.executor_running = False
        # Close the episode out inside the finally so neither a count-gate
        # abort nor an executor exception can strand the recorder thread.
        # Off the event loop: the flush writes the npz and shells out to
        # ffmpeg for episode_video.mp4, which would otherwise stall every
        # other request (including the UI's progress poll) for seconds.
        _demo_spark = (
            {
                "bt_hash": getattr(result, "bt_hash", None),
                "plan_source": getattr(result, "plan_source", None),
                "label_resolutions": dict(getattr(result, "label_resolutions", None) or {}),
            }
            if result is not None
            else None
        )
        _demo_ok = bool(result is not None and result.success)
        _demo_discard = _mismatch is not None
        # The verdict, not just the boolean. Without it a `fail` and an
        # `unverified` write byte-identical episode metadata, so a run nobody
        # could judge is indistinguishable from one judged bad -- and the
        # dataset's success_only filter pools them together.
        _demo_verify = getattr(result, "verify", None) if result is not None else None
        _demo_summary = await loop.run_in_executor(
            None,
            lambda: stop_recorder(
                success=_demo_ok,
                discard=_demo_discard,
                spark=_demo_spark,
                verify=_demo_verify,
            ),
        )

    if _mismatch is not None:
        return JSONResponse(
            status_code=422,
            content={
                "error": str(_mismatch),
                "instruction": req.instruction,
                "reason": "prompt_count_mismatch",
            },
        )

    _video_path = None
    if req.record_video:
        try:
            _video_path = await loop.run_in_executor(
                None,
                lambda: pipeline.stop_video_recording(),
            )
        except Exception:
            pass

    if not req.dry_run:
        try:
            await loop.run_in_executor(None, lambda: _pretask_home(pipeline))
        except Exception:
            pass

    return {
        "instruction": result.instruction,
        "success": result.success,
        # Tri-state verdict, same as /api/execute_approved. `success` is
        # exactly verify_status == "pass"; "unverified" is not a failure.
        "verify_status": getattr(result, "verify_status", UNVERIFIED),
        "verify": getattr(result, "verify", None),
        "duration": result.duration,
        "detections": result.detections,
        "plan": result.plan,
        "execution_results": result.execution_results,
        "video_path": str(_video_path) if _video_path else None,
        # Plan provenance: which cached tree ran and how it was matched.
        # "llm" here during a collection session means the cache missed.
        "plan_source": result.plan_source,
        "bt_hash": result.bt_hash,
        "label_resolutions": result.label_resolutions,
        # Recorded demonstration (path, frame count, commanded-action
        # fraction). None when record_demo was not set.
        "demo_episode": _demo_summary,
    }

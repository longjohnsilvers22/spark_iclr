"""
Trial runner: automated multi-trial execution for benchmark experiments.

Modes:
  auto     fully automatic: run, home, re-detect, run. No pause.
  semi     run, home, wait N seconds for user intervention, auto-run.
             User can POST /api/trials/intervene to extend the window.
  manual   run, home, wait indefinitely until POST /api/trials/next.

Endpoints:
  POST /api/trials/start   begin a trial session
  POST /api/trials/stop    abort the session after current trial finishes
  POST /api/trials/next    advance from waiting state (manual mode)
  POST /api/trials/intervene signal that user is intervening (semi mode)
  GET  /api/trials/status  current session state
  GET  /api/trials/results per-trial results log
"""

from __future__ import annotations

import asyncio
import json
import logging
import threading
import time
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import List, Optional

import yaml
from fastapi import APIRouter
from fastapi.responses import JSONResponse

from spark_real.control.success_predicates import UNVERIFIED
from spark_real.control.success_verifier import task_success
from spark_real.recording.schema import MODE_AUTONOMOUS
from spark_real.routes import state
from spark_real.routes.vla_record import start_recorder, stop_recorder

logger = logging.getLogger("spark_trials")
router = APIRouter()

# Data types


@dataclass
class TrialResult:
    trial_idx: int
    started_at: float
    finished_at: float
    success: bool
    actions: list  # per-action results from executor
    detection_count: int = 0
    notes: str = ""
    # pass | fail | unverified. `success` is exactly (verify_status == "pass");
    # an unverified trial is one nobody could check, not a failed one, and a
    # trial table that collapses the two hides the verifier's blind spots.
    verify_status: str = "unverified"


@dataclass
class TrialSession:
    task_name: str
    mode: str  # "auto" | "semi" | "manual"
    bt_yaml: str
    prompts: List[str]
    n_trials: int
    inter_trial_delay_s: float  # for semi mode
    # Write each rollout as a VLA training episode under recording.data_dir,
    # in the same schema as the human teleop corpus.
    record_demo: bool = False
    current_trial: int = 0
    results: List[TrialResult] = field(default_factory=list)
    state: str = "idle"  # idle | running | waiting | intervening | done | stopped
    started_at: float = 0.0
    stop_requested: bool = False
    next_requested: bool = False
    intervene_requested: bool = False
    _thread: Optional[threading.Thread] = field(default=None, repr=False)


_session: Optional[TrialSession] = None
_session_lock = threading.Lock()

# Trial execution (runs in background thread)


def _run_trial_session(session: TrialSession):
    """
    Execute the full trial loop in a background thread.
    """
    pipeline = state.pipeline
    if pipeline is None:
        session.state = "stopped"
        logger.error("Pipeline not initialized")
        return

    score = yaml.safe_load(session.bt_yaml)
    logger.info(
        "Trial session started: task=%s mode=%s n=%d",
        session.task_name,
        session.mode,
        session.n_trials,
    )

    results_dir = Path(pipeline.output_dir) / "trials" / session.task_name
    results_dir.mkdir(parents=True, exist_ok=True)

    for trial_idx in range(session.n_trials):
        if session.stop_requested:
            logger.info("Stop requested, ending after trial %d", trial_idx)
            break

        session.current_trial = trial_idx + 1
        session.state = "running"
        logger.info(
            "Trial %d/%d (%s)",
            trial_idx + 1,
            session.n_trials,
            session.task_name,
        )

        t0 = time.time()

        # Phase 1: detect
        detection_count = 0
        try:
            with pipeline.activity("capturing"):
                captures = pipeline.capture()
            with pipeline.activity("detecting"):
                detections = pipeline.detect(
                    captures, session.prompts, instruction=session.task_name
                )
            detection_count = len(detections)
            state.pending_detections = detections
            logger.info("  Detected %d objects", detection_count)
        except Exception as e:
            logger.warning("  Detection failed: %s", e)
            detections = state.pending_detections or []
            detection_count = len(detections)

        # Phase 2: execute BT
        action_results = []
        success = False
        verify_status = UNVERIFIED
        try:
            # Home before trial (unless skip_prelude style)
            try:
                pipeline._robot.recover_from_errors()
                for m in ("go_home", "move_home", "home"):
                    if hasattr(pipeline._robot, m):
                        getattr(pipeline._robot, m)()
                        break
            except Exception as e:
                logger.warning("  Pre-trial home failed: %s", e)

            # Record this rollout as a VLA demonstration. Started AFTER the
            # pre-trial home so the episode contains the task, not the journey
            # back to home; stopped in a finally so an aborted trial still
            # finalizes (or discards) its episode instead of leaking a recorder.
            rec = None
            if session.record_demo:
                try:
                    rec = start_recorder(
                        pipeline,
                        task=session.task_name,
                        prompt=session.task_name,
                        mode=MODE_AUTONOMOUS,
                    )
                except Exception as e:
                    logger.warning("  Demo recording failed to start: %s", e)

            # Bound BEFORE the try: the finally reads it, and pipeline.execute
            # raising is exactly the case the finally exists for. Leaving it
            # unbound would raise UnboundLocalError there, skip stop_recorder,
            # and leak a running recorder thread into the next trial -- whose
            # episode would then never be finalized.
            exec_results = []
            state.executor_running = True
            try:
                exec_results = pipeline.execute(score, detections)
            finally:
                state.executor_running = False
                # The verdict, not "every primitive returned True". A trial
                # whose fork landed beside the tray ran cleanly and failed.
                _executor = getattr(pipeline, "_executor", None)
                outcome = getattr(_executor, "verify_outcome", None) if _executor else None
                success = task_success(outcome)
                verify_status = outcome.status if outcome is not None else UNVERIFIED
                if rec is not None:
                    try:
                        summary = stop_recorder(
                            success=bool(success),
                            spark={"trial_idx": trial_idx + 1,
                                   "session": session.task_name},
                            verify=outcome.to_dict() if outcome is not None else None,
                        )
                        if summary:
                            logger.info("  Episode -> %s", summary.get("episode_dir"))
                    except Exception as e:
                        logger.warning("  Demo recording failed to stop: %s", e)

            action_results = [
                {
                    "action": r.action_type,
                    "success": r.success,
                    "message": r.message,
                    "duration": r.duration,
                }
                for r in exec_results
            ]
            logger.info("  Trial verdict: %s (success=%s)", verify_status, success)
        except Exception as e:
            logger.error("  Execution error: %s", e)
            action_results = [
                {
                    "action": "error",
                    "success": False,
                    "message": str(e),
                    "duration": 0.0,
                }
            ]

        t1 = time.time()

        # Phase 3: home after trial
        try:
            pipeline._robot.open_gripper()
            for m in ("go_home", "move_home", "home"):
                if hasattr(pipeline._robot, m):
                    getattr(pipeline._robot, m)()
                    break
        except Exception as e:
            logger.warning("  Post-trial home failed: %s", e)

        # Record result
        result = TrialResult(
            trial_idx=trial_idx + 1,
            started_at=t0,
            finished_at=t1,
            success=success,
            actions=action_results,
            detection_count=detection_count,
            verify_status=verify_status,
        )
        session.results.append(result)
        logger.info(
            "  Trial %d: success=%s duration=%.1fs detections=%d",
            trial_idx + 1,
            success,
            t1 - t0,
            detection_count,
        )

        # Save incremental results
        _save_results(session, results_dir)

        # Phase 4: inter-trial wait (mode-dependent)
        if trial_idx < session.n_trials - 1 and not session.stop_requested:
            if session.mode == "manual":
                session.state = "waiting"
                session.next_requested = False
                logger.info("  Waiting for /api/trials/next ...")
                while not session.next_requested and not session.stop_requested:
                    time.sleep(0.5)
                session.next_requested = False

            elif session.mode == "semi":
                session.state = "waiting"
                session.intervene_requested = False
                deadline = time.time() + session.inter_trial_delay_s
                logger.info(
                    "  %.0fs window, POST /api/trials/intervene to pause",
                    session.inter_trial_delay_s,
                )
                while time.time() < deadline and not session.stop_requested:
                    if session.intervene_requested:
                        session.state = "intervening"
                        logger.info("  User intervening, waiting for /api/trials/next")
                        session.next_requested = False
                        while not session.next_requested and not session.stop_requested:
                            time.sleep(0.5)
                        session.next_requested = False
                        session.intervene_requested = False
                        break
                    time.sleep(0.5)

            else:  # auto
                session.state = "waiting"
                time.sleep(2.0)  # brief pause for camera settle

    # Done
    session.state = "done" if not session.stop_requested else "stopped"
    _save_results(session, results_dir)

    n_success = sum(1 for r in session.results if r.success)
    logger.info(
        "Trial session complete: %d/%d success (%.0f%%)",
        n_success,
        len(session.results),
        100 * n_success / max(len(session.results), 1),
    )


def _save_results(session: TrialSession, results_dir: Path):
    """
    Write incremental results JSON.
    """
    data = {
        "task_name": session.task_name,
        "mode": session.mode,
        "n_trials": session.n_trials,
        "completed": len(session.results),
        "successes": sum(1 for r in session.results if r.success),
        "results": [asdict(r) for r in session.results],
    }
    out = results_dir / "results.json"
    out.write_text(json.dumps(data, indent=2))


# Endpoints


@router.post("/api/trials/start")
async def start_trials(body: dict):
    """
    Start a trial session.

    Body:
        task_name:   str, identifier for logging/output directory
        mode:        "auto" | "semi" | "manual"
        bt_yaml:     str, YAML string of the BT to run each trial
        bt_file:     str, OR path to a YAML file (relative to tests/corl/scores/)
        prompts:     list[str], SAM3 detection prompts
        n_trials:    int, number of trials (default 20)
        delay_s:     float, inter-trial delay for semi mode (default 30)
        record_demo: bool, write each rollout as a VLA training episode
                     (default false)
    """
    global _session
    with _session_lock:
        if _session is not None and _session.state in (
            "running",
            "waiting",
            "intervening",
        ):
            return JSONResponse(
                status_code=409,
                content={"error": "Session already running", "state": _session.state},
            )

    task_name = body.get("task_name", "unnamed")
    mode = body.get("mode", "manual")
    if mode not in ("auto", "semi", "manual"):
        return JSONResponse(status_code=400, content={"error": f"Invalid mode: {mode}"})

    bt_yaml = body.get("bt_yaml")
    if not bt_yaml and body.get("bt_file"):
        scores_dir = (
            Path(__file__).resolve().parent.parent / "tests" / "corl" / "scores"
        )
        bt_path = scores_dir / body["bt_file"]
        if not bt_path.exists():
            return JSONResponse(
                status_code=400, content={"error": f"BT file not found: {bt_path}"}
            )
        bt_yaml = bt_path.read_text()

    if not bt_yaml:
        return JSONResponse(
            status_code=400, content={"error": "Provide bt_yaml or bt_file"}
        )

    prompts = body.get("prompts", [])
    n_trials = int(body.get("n_trials", 20))
    delay_s = float(body.get("delay_s", 30.0))
    record_demo = bool(body.get("record_demo", False))

    session = TrialSession(
        task_name=task_name,
        mode=mode,
        bt_yaml=bt_yaml,
        prompts=prompts,
        n_trials=n_trials,
        inter_trial_delay_s=delay_s,
        record_demo=record_demo,
        started_at=time.time(),
    )

    t = threading.Thread(
        target=_run_trial_session,
        args=(session,),
        daemon=True,
        name=f"trials-{task_name}",
    )
    session._thread = t

    with _session_lock:
        _session = session
    t.start()

    return {
        "status": "started",
        "task_name": task_name,
        "mode": mode,
        "n_trials": n_trials,
    }


@router.post("/api/trials/stop")
async def stop_trials():
    """
    Stop the current session after the current trial finishes.
    """
    with _session_lock:
        if _session is None:
            return {"status": "no_session"}
        _session.stop_requested = True
        _session.next_requested = True  # unblock any wait
    return {"status": "stop_requested"}


@router.post("/api/trials/next")
async def next_trial():
    """
    Advance from waiting state (manual mode or after intervention).
    """
    with _session_lock:
        if _session is None:
            return {"status": "no_session"}
        _session.next_requested = True
    return {"status": "next_signaled"}


@router.post("/api/trials/intervene")
async def intervene():
    """
    Signal user intervention during semi-auto countdown.
    """
    with _session_lock:
        if _session is None:
            return {"status": "no_session"}
        _session.intervene_requested = True
    return {
        "status": "intervene_signaled",
        "hint": "POST /api/trials/next when ready to continue",
    }


@router.get("/api/trials/status")
async def trial_status():
    """
    Get current session state.
    """
    with _session_lock:
        if _session is None:
            return {"state": "no_session"}
        return {
            "state": _session.state,
            "task_name": _session.task_name,
            "mode": _session.mode,
            "current_trial": _session.current_trial,
            "n_trials": _session.n_trials,
            "completed": len(_session.results),
            "successes": sum(1 for r in _session.results if r.success),
            "elapsed_s": round(time.time() - _session.started_at, 1),
        }


@router.get("/api/trials/results")
async def trial_results():
    """
    Get per-trial results.
    """
    with _session_lock:
        if _session is None:
            return {"results": []}
        return {
            "task_name": _session.task_name,
            "completed": len(_session.results),
            "successes": sum(1 for r in _session.results if r.success),
            "results": [asdict(r) for r in _session.results],
        }

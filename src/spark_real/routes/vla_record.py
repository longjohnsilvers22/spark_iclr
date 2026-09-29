"""
Demonstration recording routes.

Records synchronized image + trajectory episodes in the human teleop schema
(see ``spark_real/recording/``) so autonomous SPARK runs and human teleop
demonstrations land in the same on-disk format. This is the training-data
counterpart to the BT-run bundle in ``routes/episode.py``: where that logs a
trajectory beside a behavior tree, this streams per-timestep RGB + proprio +
commanded action at a fixed 15 Hz.

Endpoints
* ``POST /api/vla_record/start``  : ``{task?, prompt?, mode?, overrides?}``
* ``POST /api/vla_record/stop``   : ``{success?, discard?, spark?}``
* ``GET  /api/vla_record/status`` : recording flag + frame count + episode dir
* ``GET  /api/vla_record/list``   : enumerate recorded episodes under the root

Mode handshake
The recorder only *reads* robot state (RTDE receive) and camera frames; it
never commands the arm, so it is safe alongside either driver:

* ``mode="teleop"``     the operator drives. Refuses to start while the
  autonomous executor is running or an external teleop subprocess owns the
  arm, matching the borrow/release discipline the rest of the server follows.
* ``mode="autonomous"`` SPARK's executor drives. The executor running is the
  *expected* state here, not a conflict; this is the mode the demo-parity
  effort exists for. Started from ``routes/execution.py`` when a request sets
  ``record_demo``.
"""

from __future__ import annotations

import logging
import tempfile
from pathlib import Path
from typing import Any, Dict, List, Optional

from fastapi import APIRouter, HTTPException
from fastapi.responses import JSONResponse
from pydantic import BaseModel

from spark_real.recording.naming import safe_task_name
from spark_real.recording.schema import MODES, MODE_AUTONOMOUS, MODE_TELEOP
from spark_real.recording.settings import resolve_settings
from spark_real.recording.vla_dataset import ALL_CAMERAS, iter_episodes
from spark_real.recording.vla_recorder import DemoRecorder
from spark_real.routes import state

logger = logging.getLogger("spark_server")
router = APIRouter()


class VLARecordStartRequest(BaseModel):
    # The task string IS the episode directory name, verbatim. Falls back to
    # the pending instruction so the UI does not have to repeat itself.
    task: Optional[str] = None
    prompt: Optional[str] = None
    mode: str = MODE_TELEOP
    # Per-session overrides of the recording: config block (e.g. demo_mode).
    overrides: Optional[Dict[str, Any]] = None
    # Accepted for compatibility with the previous request shape.
    task_name: Optional[str] = None
    instruction: Optional[str] = None


class VLARecordStopRequest(BaseModel):
    success: bool = True
    discard: bool = False
    # Provenance merged into metadata["spark"] (bt_hash, plan_source, ...).
    spark: Optional[Dict[str, Any]] = None
    # VerifyOutcome dict for an autonomous episode, when the caller has one.
    # Left None for teleop, where the operator IS the judge; not guessed from
    # the executor, whose last verdict may belong to a different run.
    verify: Optional[Dict[str, Any]] = None


# Config key an operator has to change when the data root is unusable. Named
# in the error body so the 4xx is actionable without reading this file.
DATA_DIR_CONFIG_KEY = "recording.data_dir"


def _bad_request(error: str, **extra: Any) -> HTTPException:
    """
    A 4xx carrying the same ``{"error": ...}`` body shape the routes return.

    Raised (not returned) so it also surfaces cleanly from the /api/execute
    path, where ``start_recorder`` is called outside any try/except on
    purpose: FastAPI turns an HTTPException into a response wherever it is
    raised, so a misconfigured data root fails the request with a readable
    message instead of a bare PermissionError traceback and a 500.
    """
    detail: Dict[str, Any] = {"error": error}
    detail.update(extra)
    return HTTPException(status_code=400, detail=detail)


def _unwritable(root: Path, exc: BaseException) -> HTTPException:
    return _bad_request(
        f"Recording data root is not writable: {root} ({exc.__class__.__name__}: {exc}). "
        f"Set {DATA_DIR_CONFIG_KEY} in configs/<family>_default.yaml to a writable "
        f"directory, or pass overrides={{'data_dir': ...}}.",
        data_dir=str(root),
        config_key=DATA_DIR_CONFIG_KEY,
    )


def _ensure_writable_data_root(root: Path) -> None:
    """
    Fail fast, and legibly, when the configured data root cannot be written.

    Otherwise a clone with no writable data root reaches ``rec.begin()`` and
    throws a bare PermissionError out of the executor future, killing an
    ``/api/execute?record_demo=true`` with a 500 before the run starts.
    """
    try:
        root.mkdir(parents=True, exist_ok=True)
    except OSError as exc:  # PermissionError, NotADirectoryError, ENOSPC, ...
        raise _unwritable(root, exc) from exc
    try:
        # Real write probe: os.access lies under ACLs, read-only mounts and
        # root, and a recorder that fails on its first frame is worse than
        # one that never started.
        with tempfile.NamedTemporaryFile(dir=root, prefix=".spark_write_probe"):
            pass
    except OSError as exc:
        raise _unwritable(root, exc) from exc


def start_recorder(
    pipeline,
    *,
    task: str,
    prompt: Optional[str] = None,
    mode: str = MODE_AUTONOMOUS,
    overrides: Optional[Dict[str, Any]] = None,
) -> DemoRecorder:
    """
    Construct, start and register a recorder. Raises on failure.

    Importable so ``routes/execution.py`` can begin an autonomous recording
    beside ``pipeline.start_video_recording`` without duplicating any of this.

    Failures that are the caller's or the config's fault are raised as a 4xx
    ``HTTPException`` with an actionable body: an unwritable data root, or a
    task name that is not a single path component (``recording.naming`` keeps
    the episode directory confined to the data root, so ``../../etc/evil``
    raises rather than escaping -- do not "fix" that by sanitising the name,
    which would silently merge two tasks into one directory).
    """
    settings = resolve_settings(getattr(pipeline, "config", None), overrides)
    root = Path(settings.data_dir).expanduser()

    try:
        safe_task_name(task)
    except ValueError as exc:
        raise _bad_request(
            f"Invalid task name for an episode directory: {exc}. The task string is "
            "used verbatim as a single directory name under the data root.",
            task=task,
            data_dir=str(root),
        ) from exc

    _ensure_writable_data_root(root)

    try:
        rec = DemoRecorder(
            settings,
            task=task,
            prompt=prompt,
            mode=mode if mode in MODES else MODE_AUTONOMOUS,
        )
        rec.begin(capture_fn=pipeline.capture, robot=pipeline._robot)
    except ValueError as exc:
        # Path-confinement rejection from recording.naming.
        raise _bad_request(str(exc), task=task, data_dir=str(root)) from exc
    except OSError as exc:
        raise _unwritable(root, exc) from exc

    state.vla_recorder = rec
    state.vla_recording = True
    return rec


def stop_recorder(
    *,
    success: bool = True,
    discard: bool = False,
    spark: Optional[Dict[str, Any]] = None,
    verify: Optional[Dict[str, Any]] = None,
) -> Optional[Dict[str, Any]]:
    """
    Stop whatever recorder is registered. Returns its summary, or None.

    Safe to call unconditionally at the end of an execute request.

    ``verify`` is the VerifyOutcome dict for the run; ``success`` must be
    ``verify["status"] == "pass"``. Passing a bare success computed from
    ExecutionResults is how a mis-labelled demonstration enters the corpus.
    """
    rec = state.vla_recorder
    if rec is None:
        return None
    state.vla_recording = False
    state.vla_recorder = None
    if discard:
        rec.discard()
        return rec.summary()
    rec.end(success=success, spark=spark, verify=verify)
    return rec.summary()


def _data_root(pipeline) -> Path:
    settings = resolve_settings(getattr(pipeline, "config", None), None)
    return Path(settings.data_dir).expanduser()


@router.post("/api/vla_record/start")
async def vla_record_start(req: VLARecordStartRequest):
    """Begin a synchronized image+trajectory recording."""
    if state.vla_recording and state.vla_recorder is not None:
        return JSONResponse(
            status_code=409,
            content={
                "error": "A recording is already in progress; stop it first",
                "episode_dir": str(getattr(state.vla_recorder, "episode_dir", "")),
            },
        )

    pipeline = state.pipeline
    if pipeline is None or getattr(pipeline, "_robot", None) is None:
        return JSONResponse(status_code=400, content={"error": "Robot not connected"})

    mode = req.mode if req.mode in MODES else MODE_TELEOP
    if mode == MODE_TELEOP:
        if state.executor_running:
            return JSONResponse(
                status_code=409,
                content={"error": "Executor is driving the arm; use mode='autonomous'"},
            )
        if state.viser_teleop_proc is not None and state.viser_teleop_proc.poll() is None:
            return JSONResponse(
                status_code=409,
                content={"error": "External teleop subprocess owns the robot"},
            )
    elif not state.executor_running:
        # Not fatal: an operator may arm the recorder a beat before /api/execute.
        logger.info("vla_record: autonomous mode armed before the executor started")

    task = req.task or req.task_name or state.pending_instruction or "teleop demo"
    prompt = req.prompt or req.instruction or task

    try:
        rec = start_recorder(pipeline, task=task, prompt=prompt, mode=mode, overrides=req.overrides)
    except HTTPException as exc:
        # Operator-fixable (unwritable data root, bad task name): keep the 4xx
        # and its body instead of collapsing everything into a 500.
        logger.warning("vla_record start rejected: %s", exc.detail)
        detail = exc.detail if isinstance(exc.detail, dict) else {"error": str(exc.detail)}
        return JSONResponse(status_code=exc.status_code, content=detail)
    except Exception as exc:
        logger.error("vla_record start failed: %s", exc, exc_info=True)
        return JSONResponse(status_code=500, content={"error": str(exc)})

    logger.info(
        "vla_record: started episode %04d (task=%r, mode=%s, %.2f Hz) -> %s",
        rec.episode_id,
        task,
        mode,
        rec.settings.record_hz,
        rec.episode_dir,
    )
    return {
        "started": True,
        "episode_id": rec.episode_id,
        "episode_dir": str(rec.episode_dir),
        "task": task,
        "prompt": prompt,
        "mode": mode,
        "record_hz": rec.settings.record_hz,
        "camera_map": dict(rec.settings.camera_map),
        "train_cameras": list(rec.settings.train_cameras),
    }


@router.post("/api/vla_record/stop")
async def vla_record_stop(req: VLARecordStopRequest):
    """
    Stop the recording: flush ``trajectory.npz`` + ``metadata.json``, or
    delete the episode when ``discard`` is set.
    """
    if state.vla_recorder is None:
        return JSONResponse(status_code=404, content={"error": "No active recording"})
    try:
        summary = stop_recorder(
            success=bool(req.success),
            discard=bool(req.discard),
            spark=req.spark,
            verify=req.verify,
        )
    except Exception as exc:
        logger.error("vla_record stop failed: %s", exc, exc_info=True)
        return JSONResponse(status_code=500, content={"error": str(exc)})
    if req.discard:
        return {"stopped": True, "discarded": True, "episode": summary}
    logger.info(
        "vla_record: saved episode %s (%s frames) -> %s",
        (summary or {}).get("episode_id"),
        (summary or {}).get("num_frames"),
        (summary or {}).get("episode_dir"),
    )
    return {"stopped": True, "saved": True, "episode": summary}


@router.get("/api/vla_record/status")
async def vla_record_status():
    rec = state.vla_recorder
    if rec is None:
        return {"recording": False, "episode_dir": None, "num_frames": 0}
    summary = rec.summary()
    summary["recording"] = bool(state.vla_recording)
    return summary


@router.get("/api/vla_record/list")
async def vla_record_list():
    """Enumerate recorded episodes under the configured data root."""
    root = _data_root(state.pipeline)
    episodes: List[Dict[str, Any]] = []
    for ep in iter_episodes(root, cameras=ALL_CAMERAS):
        episodes.append(
            {
                "episode_dir": str(ep.path),
                "episode_id": ep.meta.get("episode_id"),
                "task": ep.task,
                "prompt": ep.prompt,
                "num_frames": ep.num_frames,
                "cameras": ep.available_cameras,
                "success": ep.success,
                "actual_fps": ep.meta.get("actual_fps"),
                "source": ep.meta.get("source", "human"),
            }
        )
    return {"data_dir": str(root), "count": len(episodes), "episodes": episodes}

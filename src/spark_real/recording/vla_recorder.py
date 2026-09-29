"""
Demonstration recorder: one episode, in the human teleop schema.

``DemoRecorder`` runs a sleep-pinned loop at ``record_hz`` (15 Hz, matching
the human corpus to within 0.6 ms of jitter). Every tick it grabs one frame
from each camera, reads the robot's proprioceptive state, derives the action
label, writes the JPEGs straight to disk and buffers the (tiny) state row. On
``end()`` it flushes ``trajectory.npz`` + ``metadata.json`` and, if enabled,
the two QA artifacts.

The same recorder serves both modes:

* ``teleop``: the operator drives; the action comes from the velocity
  latch that ``/api/velocity`` stamps.
* ``autonomous``: SPARK's executor drives; the action comes from the same
  latch, stamped by ``CartesianServo`` / ``send_velocity``.

Design rules it keeps:

* **Reuses spark_real.** Images come from ``pipeline.capture``; proprio from
  ``recording.proprio``; the action from ``recording.action_source``; every
  byte written by ``recording.writer``. Nothing is re-implemented here.
* **Row alignment is non-negotiable.** A camera that fails a tick gets a black
  placeholder frame written, exactly as the human producer did, so frame ``i``
  of every slot stays aligned with npz row ``i``. Black frames are counted, and
  an episode exceeding ``max_black_frames`` is marked ``success=false``.
* **A required camera is never silently absent.** A slot in
  ``recording.required_cameras`` that produced nothing all episode (the
  sideview Kinect is routinely unplugged) is black-filled at save time to the
  episode's exact length, because the training converter hard-requires all
  three camera directories and silently SKIPS an episode missing one. The
  substitution is recorded in ``metadata.spark.synthetic_cameras`` and logged.
* **NaN never reaches disk unflagged.** A short RTDE hiccup is carry-filled by
  the writer; anything still NaN in a core npz array fails the episode, since
  one NaN aborts the entire downstream conversion run.
* **Never raises out of the loop.** A transient camera or RTDE failure costs
  one channel for one tick, never the demonstration.

Lifecycle::

    rec = DemoRecorder(settings, task="put the knife in the tray")
    rec.begin(capture_fn=pipeline.capture, robot=pipeline._robot)
    ...                                     # operator or executor drives
    rec.end(success=True, spark={"bt_hash": ...})
    rec.discard()                           # OR: delete the episode dir
"""

from __future__ import annotations

import logging
import shutil
import threading
import time
from datetime import datetime
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Tuple

import numpy as np

from spark_real.recording.action_source import ActionSampler
from spark_real.recording.naming import episode_dir, next_episode_number
from spark_real.recording.proprio import ProprioReader
from spark_real.recording.schema import (
    CAMERA_SLOTS,
    COMMANDED_SOURCES,
    MODES,
    MODE_TELEOP,
    NPZ_CORE_KEYS,
    SPARK_SOURCE_VALUE,
)
from spark_real.recording.settings import DATA_DIR_ENV, RecordingSettings
from spark_real.recording.writer import (
    TrajectoryBuffer,
    black_frame,
    build_metadata,
    save_metadata,
    save_plot,
    save_trajectory,
    save_video,
    write_frame,
)

logger = logging.getLogger(__name__)

CaptureFn = Callable[[], Dict[str, Dict]]

# Below this share of genuinely-commanded actions the episode is still saved,
# but loudly flagged: the rest of the rows are post-hoc reconstructions.
MIN_COMMANDED_FRACTION = 0.9


def _has_nan(array: Optional[np.ndarray]) -> bool:
    """True when a numeric npz array carries a NaN. Missing counts as bad."""
    if array is None:
        return True
    arr = np.asarray(array)
    if not np.issubdtype(arr.dtype, np.floating):
        return False
    return bool(np.isnan(arr).any())


class DemoRecorder:
    """One demonstration episode, written in the human teleop schema."""

    def __init__(
        self,
        settings: RecordingSettings,
        task: str,
        *,
        prompt: Optional[str] = None,
        mode: str = MODE_TELEOP,
        data_root: Optional[str] = None,
    ):
        self.settings = settings
        self.task = task or "task"
        self.prompt = prompt if prompt is not None else self.task
        self.mode = mode if mode in MODES else MODE_TELEOP
        self.data_root = Path(data_root or settings.data_dir).expanduser()

        # Episode numbering is max(existing)+1 within this task's directory,
        # allocated at construction so the status route can report the path
        # before the first frame lands.
        self.episode_id = next_episode_number(self.data_root, self.task)
        self.episode_dir = episode_dir(self.data_root, self.task, self.episode_id)

        self.period = 1.0 / max(1e-3, float(settings.record_hz))
        self.buffer = TrajectoryBuffer()
        self.black_frames = 0
        self.slots_written: List[str] = []

        self._capture_fn: Optional[CaptureFn] = None
        self._robot: Any = None
        self._proprio: Optional[ProprioReader] = None
        self._actions: Optional[ActionSampler] = None

        self._stop = threading.Event()
        # Makes ONE recorded tick atomic: the JPEG writes, the npz row and the
        # step counter all move together. TrajectoryBuffer is nine parallel
        # lists appended one at a time, so a save that reads them mid-tick
        # would see RAGGED COLUMNS (n timestamps against n+1 joint_positions).
        # Also stops _fill_required_cameras(n) writing black frames 0..n-1
        # while the loop writes frame n.
        self._row_lock = threading.Lock()
        self._thread: Optional[threading.Thread] = None
        self._start_dt: Optional[datetime] = None
        self._finished = False
        self._step = 0
        self._final: Dict[str, Any] = {}
        self._warned: Dict[str, bool] = {}

    # lifecycle

    def begin(self, capture_fn: CaptureFn, robot: Any, start_thread: bool = True) -> None:
        """
        Wire up the sources and start the recording thread. Idempotent.

        ``start_thread=False`` arms everything but leaves the loop to the
        caller via :meth:`step_once`, used by the offline parity test, which
        needs a deterministic clock and no real 15 Hz wall time.
        """
        if self._thread is not None or self._proprio is not None:
            return
        self._capture_fn = capture_fn
        self._robot = robot
        self._proprio = ProprioReader(robot)
        self._actions = ActionSampler(self.settings, robot)
        try:
            self.episode_dir.mkdir(parents=True, exist_ok=True)
        except OSError as exc:
            # The data root is a config value and may well not exist (or not be
            # writable) on this machine. Say so in the one place that knows
            # which knob to turn, rather than surfacing a bare PermissionError.
            raise RuntimeError(
                f"cannot create the episode directory {self.episode_dir}: {exc}. "
                f"Set recording.data_dir in configs/<family>_default.yaml or "
                f"${DATA_DIR_ENV} to a writable path."
            ) from exc
        self._start_dt = datetime.now()
        self._stop.clear()
        if not start_thread:
            return
        self._thread = threading.Thread(
            target=self._loop, daemon=True, name=f"episode-rec-{self.episode_id:04d}"
        )
        self._thread.start()
        logger.info(
            "DemoRecorder: started %r @ %.2f Hz (mode=%s) -> %s",
            self.task,
            self.settings.record_hz,
            self.mode,
            self.episode_dir,
        )

    JOIN_TIMEOUT_S = 5.0

    def _quiesce(self) -> "TrajectoryBuffer":
        """
        Stop the loop, wait for it, and DETACH the row buffer.

        The loop's capture_fn can block on a camera staleness deadline and
        its proprio read goes over RTDE, so a thread still in flight at stop
        time is ordinary, and it would keep appending to the buffer _save is
        serialising. Detaching under _row_lock hands _save a private,
        rectangular snapshot and leaves a thread that will not die appending
        into a buffer nobody reads.
        """
        self._stop.set()
        thread = self._thread
        if thread is not None:
            thread.join(timeout=self.JOIN_TIMEOUT_S)
            if thread.is_alive():
                logger.error(
                    "DemoRecorder: recording thread did not exit within %.0fs "
                    "(camera or RTDE read blocked?); detaching its buffer and "
                    "saving what was recorded. It is a daemon and exits with "
                    "the process.",
                    self.JOIN_TIMEOUT_S,
                )
            self._thread = None
        with self._row_lock:
            buffer = self.buffer
            self.buffer = TrajectoryBuffer()
            return buffer

    def end(
        self,
        *,
        success: bool = True,
        spark: Optional[Dict[str, Any]] = None,
        verify: Optional[Dict[str, Any]] = None,
    ) -> Dict[str, Any]:
        """Stop, flush, and return the metadata dict. Idempotent.

        ``verify`` is the VerifyOutcome dict; ``success`` must already be
        ``verify["status"] == "pass"``.
        """
        if self._finished:
            return self._final
        self._finished = True
        buffer = self._quiesce()
        self._final = self._save(
            success=success, spark=spark or {}, verify=verify, buffer=buffer
        )
        return self._final

    def discard(self) -> None:
        """Stop and delete the episode directory. Idempotent."""
        self._quiesce()
        self._finished = True
        if self.episode_dir.exists():
            try:
                shutil.rmtree(self.episode_dir)
            except OSError as exc:
                logger.warning("DemoRecorder: discard failed: %s", exc)
        logger.info("DemoRecorder: discarded episode %04d", self.episode_id)

    # loop

    def _loop(self) -> None:
        while not self._stop.is_set():
            t0 = time.time()
            self.step_once(t0)
            # Sleep-pinned, not free-running: the human corpus sits at
            # 14.968-15.000 Hz with 0.58 ms of std, and the comparison should
            # not have to correct for a rate difference.
            elapsed = time.time() - t0
            self._stop.wait(timeout=max(0.0, self.period - elapsed))

    def step_once(self, wall_time: Optional[float] = None) -> None:
        """
        Record exactly one aligned frame + state row.

        Public so an offline harness can drive the recorder deterministically
        without a thread; the loop above is the only production caller.
        """
        t0 = time.time() if wall_time is None else float(wall_time)
        frames = self._grab()
        proprio = self._proprio.read() if self._proprio is not None else {}
        action, source = self._actions.sample(proprio)
        # One tick, indivisible: frames, row and counter. See _row_lock.
        with self._row_lock:
            self._write_frames(frames)
            self.buffer.append(
                wall_time=t0, proprio=proprio, action=action, action_source=source
            )
            self._step += 1

    def _grab(self) -> Dict[str, Optional[np.ndarray]]:
        """``{human slot: rgb or None}`` for the mapped cameras this tick."""
        out: Dict[str, Optional[np.ndarray]] = {}
        if self._capture_fn is None:
            return out
        try:
            caps = self._capture_fn() or {}
        except Exception as exc:
            self._warn("capture", "capture() failed: %s", exc)
            return out
        for cam_key, data in caps.items():
            slot = self.settings.camera_map.get(str(cam_key))
            if slot is None:
                continue
            out[slot] = data.get("rgb") if isinstance(data, dict) else None
        return out

    def _write_frames(self, frames: Dict[str, Optional[np.ndarray]]) -> None:
        """
        Persist this tick's frames.

        A slot that has produced frames before but returned nothing now gets a
        black placeholder so its frame index never drifts from the npz row. A
        slot that has never produced anything (e.g. no sideview on this rig) is
        left alone here and settled once, at save time, by
        :meth:`_fill_required_cameras`.

        A camera that only comes up mid-episode is back-filled with black
        frames for the ticks it missed, for the same reason: frame ``i`` must
        be row ``i`` for every slot the episode declares, always.
        """
        for slot in CAMERA_SLOTS:
            rgb = frames.get(slot)
            known = slot in self.slots_written
            if rgb is None and not known:
                continue
            if rgb is None:
                rgb = black_frame(self.settings.image_size)
                self.black_frames += 1
                self._warn(f"black_{slot}", "camera slot %s dropped a frame", slot)
            elif not known and self._step > 0:
                self._backfill(slot)
            ok = self._write(slot, self._step, rgb)
            if ok and not known:
                self.slots_written.append(slot)

    def _backfill(self, slot: str) -> None:
        """Black-fill ticks 0..step-1 for a camera that appeared late."""
        self._warn(
            f"late_{slot}",
            "camera slot %s appeared at tick %d; back-filling %d black frames",
            slot,
            self._step,
            self._step,
        )
        blank = black_frame(self.settings.image_size)
        for i in range(self._step):
            self._write(slot, i, blank)
            self.black_frames += 1

    def _write(self, slot: str, index: int, rgb: np.ndarray) -> bool:
        return write_frame(
            self.episode_dir,
            slot,
            index,
            rgb,
            image_size=self.settings.image_size,
            jpeg_quality=self.settings.jpeg_quality,
        )

    def _warn(self, key: str, msg: str, *args) -> None:
        if not self._warned.get(key):
            logger.warning("DemoRecorder: " + msg, *args)
            self._warned[key] = True

    # save

    def _fill_required_cameras(self, n: int) -> Tuple[List[str], List[str]]:
        """
        Settle required slots that produced nothing all episode.

        Returns ``(synthetic, missing)``: the slots black-filled to ``n``
        frames, and the ones left absent (policy ``fail``, or the fill itself
        failed). A slot filled here is a REAL directory of black JPEGs at the
        episode's own resolution and quality, which is what the training
        converter needs to ingest the episode at all (DROID does the same for
        an absent third camera). It is not counted against ``max_black_frames``:
        that budget is about a camera de-aligning mid-episode, whereas this is
        a camera the operator knowingly did not plug in.
        """
        missing = [s for s in self.settings.required_slots if s not in self.slots_written]
        if not missing:
            return [], []
        if not self.settings.fill_missing_cameras:
            logger.error(
                "DemoRecorder: episode %04d is missing required camera slot(s) %s and "
                "recording.missing_camera_policy is %r - marking it unsuccessful; the "
                "training converter SKIPS an episode without every camera directory",
                self.episode_id,
                missing,
                self.settings.missing_camera_policy,
            )
            return [], missing
        blank = black_frame(self.settings.image_size)
        synthetic: List[str] = []
        failed: List[str] = []
        for slot in missing:
            written = 0
            for i in range(n):
                written += int(self._write(slot, i, blank))
            if written == n:
                self.slots_written.append(slot)
                synthetic.append(slot)
            else:
                failed.append(slot)
        if synthetic:
            logger.warning(
                "DemoRecorder: episode %04d recorded no frames for required camera "
                "slot(s) %s; wrote %d black frame(s) per slot so the episode still "
                "converts. metadata.spark.synthetic_cameras records the substitution "
                "- those streams carry NO image information",
                self.episode_id,
                synthetic,
                n,
            )
        if failed:
            logger.error(
                "DemoRecorder: could not write black placeholder frames for %s",
                failed,
            )
        return synthetic, failed

    def _save(
        self,
        *,
        success: bool,
        spark: Dict[str, Any],
        verify: Optional[Dict[str, Any]] = None,
        buffer: Optional["TrajectoryBuffer"] = None,
    ) -> Dict[str, Any]:
        # `buffer` is the detached snapshot _quiesce hands over; the default
        # keeps the signature usable from the offline harness.
        arrays = (buffer if buffer is not None else self.buffer).to_arrays()
        n = int(len(arrays["timestamps"]))
        if n == 0:
            logger.warning("DemoRecorder: no frames recorded; nothing saved")
            return {"episode_id": self.episode_id, "num_frames": 0, "saved": False}

        ok = bool(success)
        synthetic, missing = self._fill_required_cameras(n)
        cameras = [s for s in CAMERA_SLOTS if s in self.slots_written]
        if missing:
            ok = False
        nan_keys = [k for k in NPZ_CORE_KEYS if _has_nan(arrays.get(k))]
        if nan_keys:
            logger.error(
                "DemoRecorder: episode %04d has NaN in core npz array(s) %s that "
                "no neighbouring row could fill - marking it unsuccessful; one "
                "NaN makes the LeRobot converter raise and abort the whole run",
                self.episode_id,
                nan_keys,
            )
            ok = False
        if self.black_frames > int(self.settings.max_black_frames):
            logger.error(
                "DemoRecorder: %d black frames (limit %d) - marking episode %04d "
                "unsuccessful; a de-aligned camera poisons the training set",
                self.black_frames,
                self.settings.max_black_frames,
                self.episode_id,
            )
            ok = False

        commanded = self._actions.commanded_fraction() if self._actions is not None else 0.0
        counts = dict(self._actions.counts) if self._actions is not None else {}
        provenance = dict(spark)
        provenance.setdefault("mode", self.mode)
        provenance.setdefault("action_profile", self.settings.action_profile)
        provenance.setdefault("demo_mode", bool(self.settings.demo_mode))
        provenance["commanded_action_fraction"] = round(float(commanded), 4)
        provenance["black_frames"] = int(self.black_frames)
        provenance["action_source_counts"] = {k: int(v) for k, v in counts.items() if v}
        if synthetic:
            provenance["synthetic_cameras"] = synthetic
        if missing:
            provenance["missing_cameras"] = missing
        if nan_keys:
            provenance["nan_arrays"] = nan_keys

        meta = build_metadata(
            episode_id=self.episode_id,
            task=self.task,
            success=ok,
            arrays=arrays,
            start_time=self._start_dt or datetime.now(),
            end_time=datetime.now(),
            robot_ip=self.settings.robot_ip,
            cameras=cameras,
            prompt=self.prompt,
            source=SPARK_SOURCE_VALUE,
            spark=provenance,
            verify=verify,
        )
        save_trajectory(self.episode_dir, arrays)
        save_metadata(self.episode_dir, meta)
        if self.settings.emit_plot:
            save_plot(self.episode_dir, arrays)
        if self.settings.emit_video:
            save_video(self.episode_dir, cameras, n, self.settings.record_hz)

        logger.info(
            "DemoRecorder: saved %r (%d frames, %.2f fps, %.0f%% commanded "
            "actions, cameras=%s) -> %s",
            self.task,
            n,
            meta.actual_fps,
            100.0 * commanded,
            cameras,
            self.episode_dir,
        )
        if commanded < MIN_COMMANDED_FRACTION and n > 1:
            logger.warning(
                "DemoRecorder: only %.0f%% of actions came from a command "
                "latch (%s); the rest are post-hoc reconstructions. Enable "
                "recording.demo_mode to force the servo path.",
                100.0 * commanded,
                ", ".join(sorted(COMMANDED_SOURCES)),
            )
        return meta.to_dict()

    # introspection

    def summary(self) -> Dict[str, Any]:
        out: Dict[str, Any] = {
            "episode_id": self.episode_id,
            "task": self.task,
            "prompt": self.prompt,
            "mode": self.mode,
            "episode_dir": str(self.episode_dir),
            "recording": self._thread is not None and not self._finished,
            "num_frames": self._step,
            "record_hz": self.settings.record_hz,
            "cameras": list(self.slots_written),
            "black_frames": self.black_frames,
        }
        out.update(self._final)
        return out


# Deliberately NOT aliased to ``EpisodeRecorder``: that name belongs to
# spark_real.episode_recorder (the BT-run bundle).
__all__ = ["DemoRecorder", "MIN_COMMANDED_FRACTION"]

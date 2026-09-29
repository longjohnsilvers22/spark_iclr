"""
Per-episode artifact recorder for spark_real.

One ``EpisodeRecorder`` owns a cache folder for the currently-running
task. It bundles:

  * ``bt.yaml``         : the planner's score (the BT we executed)
  * ``trajectory.npz``  : robot state @ ~30 Hz (TCP pose, joints,
                            wrench, gripper width, timestamps)
  * ``log.txt``         : slice of the executor log captured during
                            the episode (mirrors what the user saw
                            stream in the progress panel)
  * ``result.json``     : final outcome (success/fail, duration,
                            execution_results, detection summary)
  * ``video_<cam>.<ext>`` : per-camera videos owned by the pipeline's
                             ``VideoRecorder`` (not written here, just
                             copied into the bundle on save())

The cache lives at ``<output_dir>/episodes_cache/<episode_id>/``.
``save_as(task_name)`` renames it to ``<output_dir>/episodes_kept/
<task_name>_<episode_id>/``. ``discard()`` deletes it. Starting a new
episode automatically clears any prior cache (kept bundles are not
touched).

Thread model:
  * The trajectory logger runs as a daemon thread polling the robot
    driver at ~30 Hz. Started by ``begin()``, stopped by ``end()``.
  * Reads are best-effort: any driver call that raises is logged once
    and skipped on the next tick. The recorder never raises out.
"""

from __future__ import annotations

import json
import logging
import shutil
import threading
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional

import numpy as np
import yaml
from scipy.spatial.transform import Rotation

from spark_real.recording.naming import slug
from spark_real.recording.writer import pad_rows

logger = logging.getLogger(__name__)


@dataclass
class _TrajBuffer:
    """
    Single-arm trajectory buffer.
    """

    ts: List[float] = field(default_factory=list)
    tcp_pose: List[List[float]] = field(default_factory=list)  # x,y,z,rx,ry,rz
    joints: List[List[float]] = field(default_factory=list)
    wrench: List[List[float]] = field(default_factory=list)  # fx..fz tx..tz
    gripper_width: List[float] = field(default_factory=list)
    gripper_pos: List[float] = field(default_factory=list)  # 0..255 scale

    def add(
        self, ts: float, *, tcp_pose=None, joints=None, wrench=None, gw=None, gp=None
    ) -> None:
        self.ts.append(ts)
        self.tcp_pose.append(list(tcp_pose) if tcp_pose is not None else [])
        self.joints.append(list(joints) if joints is not None else [])
        self.wrench.append(list(wrench) if wrench is not None else [])
        self.gripper_width.append(float(gw) if gw is not None else float("nan"))
        self.gripper_pos.append(float(gp) if gp is not None else float("nan"))

    def to_npz_dict(self) -> Dict[str, np.ndarray]:
        # NaN-padding is the recording package's; one implementation only.
        pad = pad_rows
        return {
            "ts": np.asarray(self.ts, dtype=np.float64),
            "tcp_pose": pad(self.tcp_pose, 6),
            "joints": pad(self.joints, 7),
            "wrench": pad(self.wrench, 6),
            "gripper_width": np.asarray(self.gripper_width, dtype=np.float64),
            "gripper_pos": np.asarray(self.gripper_pos, dtype=np.float64),
        }


@dataclass
class _BimanualTrajBuffer:
    """
    Per-arm trajectory buffers + shared timestamps.

    On disk the schema is the single-arm one (``tcp_pose``, ``joints``,
    etc.) duplicated per arm with ``_left`` / ``_right`` suffixes so a
    loader that knows about bimanual can pull either arm's stream and a
    legacy loader still reads ``ts`` correctly.
    """

    ts: List[float] = field(default_factory=list)
    left: _TrajBuffer = field(default_factory=_TrajBuffer)
    right: _TrajBuffer = field(default_factory=_TrajBuffer)
    # Inter-arm distance per tick (handy for handoff post-mortems).
    inter_arm_dist: List[float] = field(default_factory=list)

    def add(
        self,
        ts: float,
        *,
        left_kw: Optional[Dict[str, Any]] = None,
        right_kw: Optional[Dict[str, Any]] = None,
        inter_dist: Optional[float] = None,
    ) -> None:
        self.ts.append(ts)
        # Per-arm sub-buffers carry their own ts list; we tolerate the
        # duplication so they remain self-contained npz-arrays.
        if left_kw is None:
            left_kw = {}
        if right_kw is None:
            right_kw = {}
        self.left.add(ts, **left_kw)
        self.right.add(ts, **right_kw)
        self.inter_arm_dist.append(
            float(inter_dist) if inter_dist is not None else float("nan")
        )

    def to_npz_dict(self) -> Dict[str, np.ndarray]:
        left_d = self.left.to_npz_dict()
        right_d = self.right.to_npz_dict()
        out: Dict[str, np.ndarray] = {
            "ts": np.asarray(self.ts, dtype=np.float64),
            "inter_arm_dist": np.asarray(self.inter_arm_dist, dtype=np.float64),
        }
        for k, v in left_d.items():
            if k == "ts":
                continue  # avoid colliding with the shared ts above
            out[f"{k}_left"] = v
        for k, v in right_d.items():
            if k == "ts":
                continue
            out[f"{k}_right"] = v
        return out


class EpisodeRecorder:
    """
    Cache + trajectory thread for a single episode.

    Lifecycle:
        rec = EpisodeRecorder(root, instruction)
        rec.begin(robot)
        ... task runs ...
        rec.end(score=..., detections=..., result=...)
        rec.save_as("pick_the_plushie")   OR   rec.discard()
    """

    POLL_HZ = 30.0

    def __init__(self, root: Path, instruction: str):
        self.root = Path(root).expanduser()
        self.root.mkdir(parents=True, exist_ok=True)
        self.episode_id = time.strftime("%Y%m%d_%H%M%S")
        self.cache_dir = self.root / "episodes_cache" / self.episode_id
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        self.instruction = instruction
        # Buffer type is chosen lazily in begin() based on the driver
        # interface. Default to the single-arm one so type stays stable
        # if begin() is never called.
        self._buf: Any = _TrajBuffer()
        self._bimanual = False
        self._stop = threading.Event()
        self._paused = False  # set True via pause() during model.sample
        self._thread: Optional[threading.Thread] = None
        self._t_start: Optional[float] = None
        self._robot: Any = None
        self._video_paths: Dict[str, Path] = {}
        self._final: Dict[str, Any] = {}
        self._exhausted = False

    # lifecycle

    def begin(self, robot: Any) -> None:
        """
        Start trajectory logging. Idempotent.

        Inspects the driver to decide single-arm vs bimanual buffering:
        a driver that exposes ``for_arm("left")`` / ``for_arm("right")``
        (BimanualFrankaDriver or BimanualSafeRobot) is treated as
        bimanual and gets per-arm trajectory streams.
        """
        if self._thread is not None:
            return
        self._robot = robot
        # Detect bimanual by duck-typing, works for both the bare driver
        # and its safe-wrapper view because both expose for_arm().
        is_bimanual = hasattr(robot, "for_arm") and callable(getattr(robot, "for_arm"))
        if is_bimanual:
            self._buf = _BimanualTrajBuffer()
            self._bimanual = True
        else:
            self._buf = _TrajBuffer()
            self._bimanual = False
        self._t_start = time.time()
        self._stop.clear()
        self._thread = threading.Thread(
            target=self._loop, daemon=True, name=f"episode-rec-{self.episode_id}"
        )
        self._thread.start()
        logger.info(
            "EpisodeRecorder: started episode %s in %s (mode=%s)",
            self.episode_id,
            self.cache_dir,
            "bimanual" if is_bimanual else "single-arm",
        )

    def end(
        self,
        *,
        score: Optional[Dict[str, Any]] = None,
        detections: Optional[List[Any]] = None,
        result: Optional[Dict[str, Any]] = None,
        video_paths: Optional[Dict[str, Path]] = None,
        log_text: Optional[str] = None,
    ) -> None:
        """
        Stop the trajectory thread and flush all artifacts to cache.
        """
        if self._exhausted:
            return
        self._exhausted = True
        self._stop.set()
        if self._thread is not None:
            self._thread.join(timeout=2.0)
            self._thread = None

        # BT YAML
        if score is not None:
            try:
                (self.cache_dir / "bt.yaml").write_text(
                    yaml.safe_dump(score, sort_keys=False)
                )
            except Exception as exc:
                logger.warning("EpisodeRecorder: bt.yaml write failed: %s", exc)

        # Trajectory
        try:
            np.savez_compressed(
                self.cache_dir / "trajectory.npz", **self._buf.to_npz_dict()
            )
        except Exception as exc:
            logger.warning("EpisodeRecorder: trajectory.npz failed: %s", exc)

        # Video paths; the actual files are owned by VideoRecorder
        # in pipeline.py; we just record where they live so the save()
        # step can copy them in.
        self._video_paths = {
            k: Path(v) for k, v in (video_paths or {}).items() if v is not None
        }

        # Result
        try:
            payload = {
                "episode_id": self.episode_id,
                "instruction": self.instruction,
                "started_at": self._t_start,
                "ended_at": time.time(),
                "duration_s": (time.time() - (self._t_start or time.time())),
                "n_samples": len(self._buf.ts),
                "video_files": {k: str(v) for k, v in self._video_paths.items()},
            }
            if detections:
                payload["detections"] = [
                    {
                        "label": getattr(d, "label", None) or d.get("label"),
                        "confidence": float(
                            getattr(d, "confidence", 0.0)
                            if not isinstance(d, dict)
                            else d.get("confidence", 0.0)
                        ),
                    }
                    for d in detections
                ]
            if result is not None:
                payload.update(result)
            (self.cache_dir / "result.json").write_text(
                json.dumps(payload, indent=2, default=str)
            )
            self._final = payload
        except Exception as exc:
            logger.warning("EpisodeRecorder: result.json failed: %s", exc)

        # Log slice
        if log_text:
            try:
                (self.cache_dir / "log.txt").write_text(log_text)
            except Exception as exc:
                logger.warning("EpisodeRecorder: log.txt failed: %s", exc)

        logger.info(
            "EpisodeRecorder: ended episode %s (%d samples, " "cache %s)",
            self.episode_id,
            len(self._buf.ts),
            self.cache_dir,
        )

    # traj poll

    def pause(self) -> None:
        """
        Pause polling without stopping the recorder thread.

        Used by EquiGraspFlow during ``model.sample`` to free the GIL so
        the inference thread can dispatch torch ops without contention.
        """
        self._paused = True

    def resume(self) -> None:
        self._paused = False

    def _loop(self) -> None:
        period = 1.0 / max(1.0, self.POLL_HZ)
        warned: Dict[str, bool] = {}
        while not self._stop.is_set():
            if getattr(self, "_paused", False):
                # When paused, wait on the stop event (rechecking every 200ms)
                # instead of busy-polling the paused flag.
                self._stop.wait(timeout=0.2)
                continue
            t0 = time.time()
            if self._bimanual:
                left_kw = self._read_arm_kw("left", warned)
                right_kw = self._read_arm_kw("right", warned)
                inter = None
                try:
                    # Inter-arm distance is a property of the safe-robot
                    # wrapper; fall back to computing from per-arm TCPs.
                    if hasattr(self._robot, "get_inter_arm_distance"):
                        inter = float(self._robot.get_inter_arm_distance())
                    else:
                        lp = left_kw.get("tcp_pose")
                        rp = right_kw.get("tcp_pose")
                        if lp and rp:
                            inter = float(
                                np.linalg.norm(np.array(lp[:3]) - np.array(rp[:3]))
                            )
                except Exception:
                    pass
                self._buf.add(t0, left_kw=left_kw, right_kw=right_kw, inter_dist=inter)
            else:
                kw = self._read_arm_kw(None, warned)
                self._buf.add(t0, **kw)
            sleep_for = max(0.0, period - (time.time() - t0))
            self._stop.wait(timeout=sleep_for)

    def _read_arm_kw(
        self, arm: Optional[str], warned: Dict[str, bool]
    ) -> Dict[str, Any]:
        """
        Poll one arm's state. ``arm=None`` for single-arm path.

        Per-arm calls pass ``arm=<side>`` as a kwarg if the driver
        supports it (BimanualFrankaDriver shim). Single-arm drivers
        ignore unknown kwargs through duck typing, but ``arm=None``
        skips the kwarg entirely so legacy drivers never see it.
        """
        kw: Dict[str, Any] = {}
        # Build the kwargs we pass into the per-getter calls.
        get_kw = {"arm": arm} if arm is not None else {}
        try:
            if hasattr(self._robot, "get_tcp_pose"):
                p = self._robot.get_tcp_pose(**get_kw)
                if isinstance(p, np.ndarray) and p.shape == (4, 4):
                    pos = p[:3, 3]
                    rv = Rotation.from_matrix(p[:3, :3]).as_rotvec()
                    kw["tcp_pose"] = [pos[0], pos[1], pos[2], rv[0], rv[1], rv[2]]
                elif p is not None:
                    kw["tcp_pose"] = list(p)[:6]
        except Exception as exc:
            tag = f"tcp_{arm or 'single'}"
            if not warned.get(tag):
                logger.warning(
                    "EpisodeRecorder: tcp_pose(%s) read failed: %s",
                    arm or "<single>",
                    exc,
                )
                warned[tag] = True
        try:
            if hasattr(self._robot, "get_joint_positions"):
                j = self._robot.get_joint_positions(**get_kw)
                if j is not None:
                    kw["joints"] = list(j)
        except Exception:
            pass
        try:
            if hasattr(self._robot, "get_tcp_force"):
                f = self._robot.get_tcp_force(**get_kw)
                if f is not None:
                    kw["wrench"] = list(f)[:6]
        except Exception:
            pass
        try:
            if hasattr(self._robot, "get_gripper_width"):
                kw["gw"] = float(self._robot.get_gripper_width(**get_kw))
        except Exception:
            pass
        try:
            if hasattr(self._robot, "get_gripper_position"):
                kw["gp"] = float(self._robot.get_gripper_position(**get_kw))
        except Exception:
            pass
        return kw

    # save / discard

    def save_as(
        self,
        task_name: str,
        kept_root: Optional[Path] = None,
        outcome: Optional[str] = None,
    ) -> Path:
        """
        Move the cache folder to ``episodes_kept/<task_name>_<id>/``
        and copy in any referenced videos. Returns the kept-folder path.
        """
        safe = slug(task_name) if str(task_name).strip() else "episode"
        # The label goes in the FOLDER NAME as well as the metadata, so a
        # corpus can be split with a glob and never depends on a reader
        # parsing json correctly.
        if outcome in ("pass", "fail"):
            safe = f"{safe}__{outcome}"
        kept = (kept_root or self.root) / "episodes_kept" / f"{safe}_{self.episode_id}"
        kept.parent.mkdir(parents=True, exist_ok=True)
        shutil.move(str(self.cache_dir), str(kept))

        for cam, src in self._video_paths.items():
            try:
                if src and src.exists():
                    shutil.copy2(src, kept / src.name)
            except Exception as exc:
                logger.warning("EpisodeRecorder: copy video %s failed: %s", src, exc)
        self.cache_dir = kept
        if outcome in ("pass", "fail"):
            try:
                meta_path = kept / "metadata.json"
                meta = {}
                if meta_path.exists():
                    meta = json.loads(meta_path.read_text())
                meta["outcome"] = outcome
                meta["outcome_source"] = "operator"
                meta_path.write_text(json.dumps(meta, indent=2))
            except Exception as exc:  # noqa: BLE001 - never lose the episode
                logger.warning("EpisodeRecorder: could not stamp outcome: %s", exc)
        logger.info(
            "EpisodeRecorder: saved episode to %s (outcome=%s)", kept, outcome
        )
        return kept

    def discard(self) -> None:
        """
        Wipe the cache folder. Idempotent.
        """
        if self.cache_dir.exists():
            try:
                shutil.rmtree(self.cache_dir)
            except Exception as exc:
                logger.warning("EpisodeRecorder: discard failed: %s", exc)
        logger.info("EpisodeRecorder: discarded episode %s", self.episode_id)

    def summary(self) -> Dict[str, Any]:
        out = {
            "episode_id": self.episode_id,
            "instruction": self.instruction,
            "cache_dir": str(self.cache_dir),
            "exhausted": self._exhausted,
            "n_samples": len(self._buf.ts),
        }
        out.update(self._final)
        return out


__all__ = ["EpisodeRecorder"]

"""
Lightweight per-task video recorder for spark_real.

Spawns a daemon thread that grabs JPEG frames from the active camera stream
at ~target_fps and encodes an mp4 with imageio_ffmpeg when stopped. Frames
are kept in memory and the file is only written on ``stop()``; for
crash-resilience swap to per-frame disk write later.

Usage::

    rec = VideoRecorder(out_dir="output/videos", fps=10)
    rec.start("pick_up_red_block")
    pipeline.run_task(...)
    path = rec.stop()  # -> Path of the resulting .mp4
"""

from __future__ import annotations

import logging
import os
import subprocess
import threading
import time
from collections import deque
from datetime import datetime
from pathlib import Path
from typing import Callable, Optional

import imageio.v3 as iio  # noqa: F401  (kept for potential future fallback)
import numpy as np

try:
    import imageio_ffmpeg
except ImportError:
    imageio_ffmpeg = None

logger = logging.getLogger(__name__)


def _resolve_ffmpeg_exe() -> str:
    """
    Return an absolute path to the ffmpeg binary.

    The posix_spawn fast path requires an absolute executable
    (``os.path.dirname(executable)`` truthy in subprocess.Popen). Prefer
    imageio_ffmpeg's bundled binary, fall back to system ffmpeg, last
    resort 'ffmpeg' on PATH (Popen rejects that for posix_spawn but exec()
    can still find it if imageio is missing).
    """
    try:
        if imageio_ffmpeg is None:
            raise ImportError("imageio_ffmpeg not available")
        return imageio_ffmpeg.get_ffmpeg_exe()
    except Exception:
        for cand in ("/usr/bin/ffmpeg", "/usr/local/bin/ffmpeg"):
            if os.path.exists(cand):
                return cand
        return "ffmpeg"


def _encode_h264_posix_spawn(frames, out_path: Path, fps: int) -> None:
    """
    Pipe raw RGB frames into ffmpeg for H.264 encoding.

    Why this exists: imageio's ``iio.imwrite`` spawns ffmpeg with a
    ``preexec_fn``, which forces Python onto the fork()+exec() path.
    fork() in a multithreaded process is unsafe per POSIX, and the
    server has JAX worker threads (pyroki) live by the first task, so
    the fork path can deadlock the encoder and the event loop.

    Instead we call ffmpeg via subprocess.Popen with arguments chosen to
    hit Python's posix_spawn fast path (subprocess.py _USE_POSIX_SPAWN):
      * absolute executable path (imageio's bundled ffmpeg)
      * preexec_fn=None
      * close_fds=False (posix_spawn cannot close arbitrary inherited fds)
      * no start_new_session, gid/uid manipulation, etc.

    Trade-off: posix_spawn cannot scrub inherited fds, so ffmpeg starts
    with the parent's open fds (Kinect handles, sockets). ffmpeg only
    reads stdin and writes the output path and never references them;
    they are freed when ffmpeg exits.
    """
    arr = np.ascontiguousarray(np.asarray(frames, dtype=np.uint8))
    n, h, w, c = arr.shape
    assert c == 3, f"expected RGB frames, got {arr.shape}"

    ffmpeg = _resolve_ffmpeg_exe()
    cmd = [
        ffmpeg,
        "-y",
        "-f",
        "rawvideo",
        "-vcodec",
        "rawvideo",
        "-s",
        f"{w}x{h}",
        "-pix_fmt",
        "rgb24",
        "-r",
        str(fps),
        "-i",
        "-",
        "-an",
        "-vcodec",
        "libx264",
        "-pix_fmt",
        "yuv420p",
        "-crf",
        "15",
        # H.264 requires even dimensions; mirror imageio's scale rescue.
        "-vf",
        f"scale={w if w % 2 == 0 else w + 1}:{h if h % 2 == 0 else h + 1}",
        "-v",
        "warning",
        str(out_path),
    ]

    # Open output as a write fd so it doesn't depend on cwd. Letting
    # ffmpeg open() it directly is fine; we just pass the path in argv.
    proc = subprocess.Popen(
        cmd,
        stdin=subprocess.PIPE,
        stdout=subprocess.DEVNULL,
        stderr=subprocess.PIPE,
        close_fds=False,  # << posix_spawn requires this
        preexec_fn=None,  # << posix_spawn requires this
        start_new_session=False,
        env=os.environ.copy(),
    )

    # Stream frames in. ffmpeg buffers ~30MB internally; for a 2-minute
    # 1080p clip the kernel pipe (1MB) will fill up and we'll block on
    # write(), which is fine, that's flow control.
    try:
        proc.stdin.write(arr.tobytes())
        proc.stdin.close()
    except BrokenPipeError as exc:
        # ffmpeg died early, drain stderr for the actual reason.
        err = proc.stderr.read().decode(errors="replace") if proc.stderr else ""
        raise RuntimeError(f"ffmpeg pipe broken before encode: {err}") from exc

    err_bytes = proc.stderr.read() if proc.stderr else b""
    rc = proc.wait()
    if rc != 0:
        raise RuntimeError(
            f"ffmpeg exited rc={rc}: {err_bytes.decode(errors='replace')}"
        )


class VideoRecorder:
    """
    Per-task recorder.  Frames captured by a ``frame_provider`` callback
    that returns an HxWx3 uint8 array, called from a background thread.
    """

    def __init__(
        self,
        frame_provider: Callable[[], Optional[np.ndarray]],
        out_dir: Path,
        fps: int = 10,
        max_seconds: float = 600.0,
    ):
        self.frame_provider = frame_provider
        self.out_dir = Path(out_dir)
        self.out_dir.mkdir(parents=True, exist_ok=True)
        self.fps = int(fps)
        self.max_frames = int(self.fps * max_seconds)

        self._frames: deque = deque(maxlen=self.max_frames)
        # Guards the buffer against a capture thread that outlived stop()'s
        # join and is still appending while the encode reads.
        self._frames_lock = threading.Lock()
        self._stop_event = threading.Event()
        self._thread: Optional[threading.Thread] = None
        self._label: str = ""

    # lifecycle

    @property
    def is_recording(self) -> bool:
        return self._thread is not None and self._thread.is_alive()

    def start(self, label: str = "") -> None:
        if self.is_recording:
            logger.warning("VideoRecorder.start: already running, ignoring")
            return
        self._label = label or "task"
        with self._frames_lock:
            self._frames.clear()
        self._stop_event.clear()
        self._thread = threading.Thread(
            target=self._loop, daemon=True, name="VideoRecorder"
        )
        self._thread.start()
        logger.info("VideoRecorder started: label=%r fps=%d", label, self.fps)

    # How long stop() waits for the capture thread. The frame provider can
    # block for a full staleness deadline inside a camera read (2 s), so a
    # 2 s join lands exactly on that boundary.
    JOIN_TIMEOUT_S = 5.0

    def stop(self) -> Optional[Path]:
        """
        Stop the capture thread and write the mp4.  Returns the file path,
        or None if no frames were captured.

        The join result is checked: the frame provider blocks up to a
        camera's 2 s staleness deadline, so a join timeout is the normal
        outcome on a wedged camera. The buffer is detached under a lock, so a
        thread that will not die appends into a deque nobody reads and cannot
        contaminate the encode ("deque mutated during iteration") or the next
        recording.
        """
        if not self.is_recording:
            return None
        self._stop_event.set()
        thread = self._thread
        if thread is not None:
            thread.join(timeout=self.JOIN_TIMEOUT_S)
            if thread.is_alive():
                logger.error(
                    "VideoRecorder capture thread did not exit within %.0fs "
                    "(frame provider blocked?); detaching its buffer and "
                    "encoding what was captured. The thread is a daemon and "
                    "will exit with the process.",
                    self.JOIN_TIMEOUT_S,
                )
        # Detach: the zombie loop keeps a live deque to append into, and the
        # encode gets a private snapshot nobody mutates.
        with self._frames_lock:
            frames = list(self._frames)
            self._frames = deque(maxlen=self.max_frames)
        self._thread = None
        return self._encode(frames)

    # internals

    def _loop(self) -> None:
        period = 1.0 / max(1, self.fps)
        next_t = time.time()
        consecutive_failures = 0
        while not self._stop_event.is_set():
            try:
                frame = self.frame_provider()
                if frame is not None and frame.size > 0:
                    with self._frames_lock:
                        self._frames.append(frame)
                    consecutive_failures = 0
                else:
                    consecutive_failures += 1
            except Exception as e:
                consecutive_failures += 1
                if consecutive_failures % 20 == 1:  # don't spam the log
                    logger.warning("VideoRecorder frame grab failed: %s", e)
            next_t += period
            sleep = max(0.0, next_t - time.time())
            self._stop_event.wait(timeout=sleep)
            # fall-behind guard: if we're chronically late, skip ahead
            if time.time() - next_t > period * 5:
                next_t = time.time()

    def _encode(self, frames=None) -> Optional[Path]:
        # `frames` is the detached snapshot stop() hands over; the default
        # keeps the historical signature usable from a test or a REPL.
        if frames is None:
            with self._frames_lock:
                frames = list(self._frames)
        if len(frames) < 2:
            logger.info("VideoRecorder: only %d frame(s), not encoding", len(frames))
            return None
        ts = datetime.now().strftime("%Y%m%d_%H%M%S")
        safe_label = "".join(c for c in self._label if c.isalnum() or c in "_-")[:48]
        path = self.out_dir / f"{ts}_{safe_label}.mp4"
        # Coerce to a contiguous (N, H, W, 3) uint8 array.
        # All frames must share H, W, channels.  Resize-on-mismatch is not
        # implemented; drop mismatched ones (rare) so encoding doesn't error.
        h, w = frames[0].shape[:2]
        clean = [f for f in frames if f.shape[:2] == (h, w) and f.dtype == np.uint8]
        if len(clean) < len(frames):
            logger.info(
                "VideoRecorder: dropped %d size-mismatched frames",
                len(frames) - len(clean),
            )
        if len(clean) < 2:
            return None
        try:
            _encode_h264_posix_spawn(clean, path, fps=self.fps)
        except Exception as e:
            logger.error("VideoRecorder encode failed: %s", e)
            return None
        logger.info(
            "VideoRecorder: wrote %s (%d frames @ %d fps)", path, len(clean), self.fps
        )
        return path

    def __len__(self) -> int:
        with self._frames_lock:
            return len(self._frames)


__all__ = ["VideoRecorder"]

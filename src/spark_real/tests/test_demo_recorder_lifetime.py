"""
DemoRecorder: one tick is indivisible, and stopping detaches the buffer.

Two defects, both on the same seam.

1. ``step_once`` wrote the JPEGs, appended the npz row and bumped ``_step``
   with no mutual exclusion against ``_save``. ``TrajectoryBuffer`` is NINE
   parallel lists appended one at a time, so a save that serialised them
   while the loop was mid-tick produced RAGGED COLUMNS -- n timestamps
   against n+1 joint_positions. Nothing raises; the episode is written
   misaligned and poisons the training set. In the same window
   ``_fill_required_cameras(n)`` black-fills frames 0..n-1 for a missing
   camera while the loop writes frame n, de-aligning that slot too.

2. ``end()`` and ``discard()`` ignored the join result. The loop's
   ``capture_fn`` reaches a camera read that blocks for a full staleness
   deadline and its proprio read goes over RTDE, so a thread still in flight
   at stop time is ordinary. It then kept appending to the very buffer
   ``_save`` was serialising.

No hardware; the capture and proprio sources are fakes.
"""

import threading
import time

import numpy as np
import pytest

from spark_real.recording.schema import MODE_TELEOP
from spark_real.recording.settings import RecordingSettings
from spark_real.recording.vla_recorder import DemoRecorder
from spark_real.recording.writer import TrajectoryBuffer


class _Robot:
    def get_observation(self):
        return {
            "joint_positions": np.zeros(6),
            "joint_velocities": np.zeros(6),
            "tcp_pose": np.zeros(6),
            "tcp_velocity": np.zeros(6),
            "gripper_position": 0.0,
        }

    def get_last_command(self):
        return None


def _settings(tmp_path):
    s = RecordingSettings()
    s.data_dir = str(tmp_path)
    s.record_hz = 200.0
    s.emit_plot = False
    s.emit_video = False
    return s


def _recorder(tmp_path, capture_fn=None, start_thread=True):
    rec = DemoRecorder(_settings(tmp_path), task="audit", mode=MODE_TELEOP)
    rec.begin(
        capture_fn=capture_fn or (lambda: {}),
        robot=_Robot(),
        start_thread=start_thread,
    )
    return rec


# 1. A tick is indivisible


def test_a_tick_never_leaves_the_buffer_columns_ragged(tmp_path):
    """
    A reader interposed exactly between two of TrajectoryBuffer's nine column
    appends must not be able to serialise a ragged buffer.
    """
    rec = _recorder(tmp_path, start_thread=False)
    reader_in = threading.Event()
    lengths = {}

    def _read():
        reader_in.set()
        # Same lock the save path takes when it detaches.
        with rec._row_lock:
            b = rec.buffer
            lengths["cols"] = {
                "wall_time": len(b.wall_time),
                "joint_positions": len(b.joint_positions),
                "actions": len(b.actions),
                "action_source": len(b.action_source),
            }

    rec.step_once(1.0)
    t = threading.Thread(target=_read, daemon=True)

    # Pin the reader inside the tick: it starts while _row_lock is held.
    with rec._row_lock:
        t.start()
        assert reader_in.wait(2.0)
        # A half-written row, which is what an unlocked tick exposes.
        rec.buffer.wall_time.append(2.0)
        time.sleep(0.05)
        assert "cols" not in lengths, "reader serialised a half-written row"
        rec.buffer.joint_positions.append(np.zeros(6))
        rec.buffer.joint_velocities.append(np.zeros(6))
        rec.buffer.tcp_poses.append(np.zeros(6))
        rec.buffer.tcp_velocity.append(np.zeros(6))
        rec.buffer.wrench.append(np.zeros(6))
        rec.buffer.gripper_positions.append(0.0)
        rec.buffer.actions.append(np.zeros(7))
        rec.buffer.action_source.append("commanded")
    t.join(timeout=3.0)

    cols = lengths["cols"]
    assert len(set(cols.values())) == 1, f"ragged buffer serialised: {cols}"


def test_step_once_holds_the_row_lock(tmp_path):
    """The frames, the row and the counter move together or not at all."""
    rec = _recorder(tmp_path, start_thread=False)
    observed = {}
    real_append = rec.buffer.append

    def _append(**kwargs):
        observed["locked"] = rec._row_lock.locked()
        return real_append(**kwargs)

    rec.buffer.append = _append
    rec.step_once(1.0)
    assert observed["locked"] is True


def test_concurrent_ticks_and_saves_never_produce_ragged_columns(tmp_path):
    """Live version: a loop ticking while saves detach, repeatedly."""
    rec = _recorder(tmp_path, start_thread=False)
    stop = threading.Event()
    ragged = []
    crashed = []

    def _guard(fn):
        def _run():
            try:
                fn()
            except BaseException as exc:  # a daemon thread dying silently
                crashed.append(repr(exc))  # would fake a pass
                stop.set()

        return _run

    def _tick():
        while not stop.is_set():
            rec.step_once()

    def _detach():
        while not stop.is_set():
            with rec._row_lock:
                b = rec.buffer
                rec.buffer = TrajectoryBuffer()
            arr = b.to_arrays()
            n = len(arr["timestamps"])
            for key in ("joint_positions", "actions", "action_source"):
                if len(arr[key]) != n:
                    ragged.append((key, n, len(arr[key])))
                    return

    threads = [
        threading.Thread(target=_guard(_tick), daemon=True) for _ in range(2)
    ]
    threads.append(threading.Thread(target=_guard(_detach), daemon=True))
    for t in threads:
        t.start()
    stop.wait(1.0)
    stop.set()
    for t in threads:
        t.join(timeout=3.0)
    assert crashed == [], crashed
    assert ragged == [], ragged


# 2. Stopping detaches the buffer from a thread that will not exit


def _wedged_recorder(tmp_path, blocker):
    ticks = []

    def _capture():
        ticks.append(1)
        if len(ticks) > 2:
            blocker.wait(timeout=10.0)
        return {}

    rec = _recorder(tmp_path, capture_fn=_capture)
    rec.JOIN_TIMEOUT_S = 0.2
    return rec, ticks


def test_quiesce_detaches_the_buffer_from_a_wedged_loop(tmp_path):
    blocker = threading.Event()
    rec, _ = _wedged_recorder(tmp_path, blocker)
    time.sleep(0.15)
    zombie = rec._thread
    detached = rec._quiesce()
    assert zombie is not None and zombie.is_alive(), (
        "test needs the recording thread to still be in flight"
    )
    n_before = len(detached)
    # The zombie now appends into rec.buffer, which nobody will save.
    blocker.set()
    zombie.join(timeout=5.0)
    assert len(detached) == n_before, "the detached snapshot kept growing"
    assert rec.buffer is not detached


def test_end_is_idempotent_and_saves_the_detached_snapshot(tmp_path):
    rec = _recorder(tmp_path)
    time.sleep(0.05)
    first = rec.end(success=True)
    second = rec.end(success=True)
    assert second is first
    assert rec._thread is None


def test_discard_stops_the_thread(tmp_path):
    rec = _recorder(tmp_path)
    time.sleep(0.05)
    rec.discard()
    assert rec._thread is None
    assert rec._finished is True
    assert not rec.episode_dir.exists()


if __name__ == "__main__":  # pragma: no cover
    pytest.main([__file__, "-v"])

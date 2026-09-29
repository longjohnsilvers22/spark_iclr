"""
Thin wrapper around the official ``flexitac`` Python package.

The FlexiTac serial protocol is not reimplemented here: the upstream
package (https://github.com/WT-MM/PyFlexiTac, pip name ``flexitac``)
owns that. This module:

  * auto-detects up to N FlexiTac sensors on serial ports
  * runs a background thread per sensor calling ``read_latest()``
    so the latest frame is always cheap to query
  * caches the latest ``FlexiTacFrame`` per sensor
  * exposes a small API: ``available()``, ``snapshot()``, ``is_in_contact()``,
    ``contact_centroid()`` that the executor + HTTP routes consume

Strictly optional. When the ``flexitac`` package is missing OR no
sensor is plugged in, ``TactileManager.available()`` returns False
and the rest of the pipeline runs unchanged.

Single-arm parallel-jaw mounting: one sensor pad per fingertip. The
``side`` argument identifies which one (``"left"`` / ``"right"``) when
two are present. With one sensor only, use ``side="left"`` everywhere
(it's just a label).

Upstream documentation:
    https://flexitac.github.io/
    https://github.com/WT-MM/PyFlexiTac
"""

from __future__ import annotations

import glob
import logging
import threading
import time
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

import numpy as np

logger = logging.getLogger(__name__)

try:
    from flexitac import FlexiTacSensor, FlexiTacFrame  # type: ignore

    _HAS_FLEXITAC = True
except ImportError:
    FlexiTacSensor = None  # type: ignore
    FlexiTacFrame = None  # type: ignore
    _HAS_FLEXITAC = False


@dataclass
class TactileSnapshot:
    """
    Result of ``TactileManager.snapshot(side)``.

    All fields are filled even when no sensor is attached (zero arrays,
    zero counts) so callers can branch on ``connected`` only and avoid
    None-checking shape attributes.
    """

    side: str
    connected: bool
    rows: int
    cols: int
    normalized: np.ndarray  # (rows, cols) float32; self-scaled display value in [0, 1]
    raw: np.ndarray  # (rows, cols) uint8
    delta: np.ndarray  # (rows, cols) float32; raw - baseline, the true signal in counts
    seq: int
    timestamp_s: float
    contact_cell_count: int  # cells where (raw - baseline) > CONTACT_DELTA
    max_response: float  # peak (raw - baseline) delta in sensor counts
    centroid_uv: Optional[
        Tuple[float, float]
    ]  # (col, row) centroid of contact, or None


class TactileManager:
    """
    Owns 0..N FlexiTac sensors and a polling thread per sensor.

    Lifecycle::

        mgr = TactileManager()
        mgr.start()           # auto-detects on /dev/ttyUSB* + /dev/ttyACM*
        snap = mgr.snapshot('left')
        ...
        mgr.stop()
    """

    POLL_HZ = 100.0  # FlexiTac streams at 100 Hz; match it
    # Contact is detected on the ABSOLUTE signal (raw - baseline) in sensor
    # counts, NOT on `normalized`: the upstream normalize() self-scales to
    # the current peak and hard-clips to [0, 1], so it cannot be thresholded.
    # Measured baseline: resting noise floor <=28 counts, firm presses read
    # 85-170. 50 sits comfortably between the two.
    CONTACT_DELTA = 50.0
    MIN_CONTACT_CELLS = 1  # one small pad taped to a fingertip -> tiny patch

    def __init__(self, ports: Optional[List[str]] = None, max_sensors: int = 2):
        """
        Args:
            ports: Explicit serial paths to open. None = auto-detect on
                ``/dev/ttyUSB*`` and ``/dev/ttyACM*``.
            max_sensors: Stop trying after this many sensors come up
                (one per fingertip = 2 for a parallel-jaw gripper).
        """
        self._explicit_ports = ports
        self._max = int(max_sensors)
        # per-side state, keyed "left"|"right"
        self._sensors: Dict[str, FlexiTacSensor] = {}
        self._latest: Dict[str, FlexiTacFrame] = {}
        self._lock = threading.Lock()
        self._stop_flag = threading.Event()
        self._threads: List[threading.Thread] = []
        self._port_for_side: Dict[str, str] = {}
        # Rebaseline handshake: the poll thread owns the serial port, so
        # recalibration must run inside it. rebaseline() sets `req`; the
        # poll loop calibrates and sets `done`.
        self._rebaseline_req: Dict[str, threading.Event] = {}
        self._rebaseline_done: Dict[str, threading.Event] = {}
        self._rebaseline_ok: Dict[str, bool] = {}

    # discovery

    def _candidate_ports(self) -> List[str]:
        if self._explicit_ports is not None:
            return list(self._explicit_ports)
        candidates = sorted(glob.glob("/dev/ttyUSB*") + glob.glob("/dev/ttyACM*"))
        return candidates

    def _try_open(self, port: str) -> Optional[FlexiTacSensor]:
        if not _HAS_FLEXITAC:
            return None
        try:
            s = FlexiTacSensor(port)
            s.open()
            # Prime the baseline so subsequent reads have a sensible
            # zero. calibrate() samples ``init_frames`` frames; if the
            # port is real but not a FlexiTac the read inside will
            # raise quickly via the upstream framing check.
            s.calibrate()
            return s
        except Exception as exc:
            logger.debug("FlexiTac probe failed on %s: %s", port, exc)
            try:
                s.close()  # type: ignore[possibly-unbound]
            except Exception:
                pass
            return None

    def start(self) -> None:
        """
        Probe ports + spawn polling threads. Idempotent.
        """
        if not _HAS_FLEXITAC:
            logger.info(
                "TactileManager: flexitac package not installed; "
                "tactile sensing disabled. `pip install flexitac` to enable."
            )
            return
        if self._sensors:
            return  # already running

        for port in self._candidate_ports():
            if len(self._sensors) >= self._max:
                break
            sensor = self._try_open(port)
            if sensor is None:
                continue
            side = "left" if "left" not in self._sensors else "right"
            self._sensors[side] = sensor
            self._port_for_side[side] = port
            self._rebaseline_req[side] = threading.Event()
            self._rebaseline_done[side] = threading.Event()
            t = threading.Thread(
                target=self._poll_loop,
                args=(side,),
                daemon=True,
                name=f"tactile-{side}",
            )
            self._threads.append(t)
            t.start()
            logger.info(
                "TactileManager: opened FlexiTac %s on %s " "(%dx%d cells)",
                side,
                port,
                sensor.rows,
                sensor.cols,
            )

        if not self._sensors:
            logger.info(
                "TactileManager: no FlexiTac sensors detected on serial "
                "ports, tactile sensing disabled. Plug in a sensor + "
                "restart the server to enable."
            )

    def stop(self) -> None:
        self._stop_flag.set()
        for t in self._threads:
            t.join(timeout=1.0)
        for side, s in self._sensors.items():
            try:
                s.close()
            except Exception:
                pass
        self._sensors.clear()
        self._latest.clear()
        self._threads.clear()

    # polling

    def _poll_loop(self, side: str) -> None:
        sensor = self._sensors[side]
        req = self._rebaseline_req[side]
        while not self._stop_flag.is_set():
            if req.is_set():
                try:
                    sensor.calibrate()
                    logger.info(
                        "TactileManager(%s): baseline recaptured "
                        "(resting max now %.1f counts)",
                        side,
                        float(np.max(sensor.baseline))
                        if sensor.baseline is not None
                        else -1.0,
                    )
                    self._rebaseline_ok[side] = True
                except Exception as exc:
                    logger.warning(
                        "TactileManager(%s): rebaseline failed: %s", side, exc
                    )
                    self._rebaseline_ok[side] = False
                finally:
                    req.clear()
                    self._rebaseline_done[side].set()
                continue
            try:
                frame = sensor.read_latest()
            except Exception as exc:
                logger.warning("TactileManager(%s): read_latest failed: %s", side, exc)
                # Brief sleep to avoid hot-loop on persistent failure.
                self._stop_flag.wait(timeout=0.2)
                continue
            if frame is not None:
                with self._lock:
                    self._latest[side] = frame
            # No sleep needed; read_latest blocks at sensor rate (100 Hz).

    # public API

    def rebaseline(
        self, side: Optional[str] = None, timeout_s: float = 5.0
    ) -> Dict[str, bool]:
        """
        Recapture per-cell baselines (median of ~30 raw frames, <1 s per
        sensor). The pads MUST be untouched while this runs - anything
        pressing gets baked into the new zero. Use after replug, re-tape,
        remount, or when resting deltas have drifted.

        Args:
            side: One side, or None for all connected sides.

        Returns:
            {side: ok} - False means the poll thread never acknowledged
            (sensor wedged) or its calibrate() raised.
        """
        targets = [side] if side is not None else self.sides()
        results: Dict[str, bool] = {}
        for sd in targets:
            if sd not in self._sensors:
                results[sd] = False
                continue
            self._rebaseline_done[sd].clear()
            self._rebaseline_req[sd].set()
        for sd in targets:
            if sd in self._sensors:
                results[sd] = self._rebaseline_done[sd].wait(
                    timeout=timeout_s
                ) and self._rebaseline_ok.get(sd, False)
        return results

    def available(self, side: Optional[str] = None) -> bool:
        if side is None:
            return bool(self._sensors)
        return side in self._sensors

    def sides(self) -> List[str]:
        return list(self._sensors.keys())

    def snapshot(self, side: str = "left") -> TactileSnapshot:
        """
        Return the latest frame (or a zero snapshot if disconnected).
        """
        sensor = self._sensors.get(side)
        if sensor is None:
            return TactileSnapshot(
                side=side,
                connected=False,
                rows=0,
                cols=0,
                normalized=np.zeros((0, 0), dtype=np.float32),
                raw=np.zeros((0, 0), dtype=np.uint8),
                delta=np.zeros((0, 0), dtype=np.float32),
                seq=0,
                timestamp_s=0.0,
                contact_cell_count=0,
                max_response=0.0,
                centroid_uv=None,
            )
        with self._lock:
            frame = self._latest.get(side)
        if frame is None:
            return TactileSnapshot(
                side=side,
                connected=True,
                rows=sensor.rows,
                cols=sensor.cols,
                normalized=np.zeros((sensor.rows, sensor.cols), dtype=np.float32),
                raw=np.zeros((sensor.rows, sensor.cols), dtype=np.uint8),
                delta=np.zeros((sensor.rows, sensor.cols), dtype=np.float32),
                seq=0,
                timestamp_s=0.0,
                contact_cell_count=0,
                max_response=0.0,
                centroid_uv=None,
            )
        norm = np.asarray(frame.normalized)
        raw = np.asarray(frame.raw)
        # Absolute per-cell signal in sensor counts. `normalized` self-scales
        # to the current peak and clips to [0, 1], so it can't be thresholded
        # for contact; the learned baseline gives the true delta.
        base = getattr(sensor, "baseline", None)
        if base is not None:
            delta = raw.astype(np.float32) - np.asarray(base, dtype=np.float32)
        else:
            delta = np.zeros_like(raw, dtype=np.float32)
        contact = delta > self.CONTACT_DELTA
        cnt = int(contact.sum())
        centroid = None
        if cnt > 0:
            ys, xs = np.where(contact)
            centroid = (float(xs.mean()), float(ys.mean()))
        return TactileSnapshot(
            side=side,
            connected=True,
            rows=int(sensor.rows),
            cols=int(sensor.cols),
            normalized=norm,
            raw=raw,
            delta=delta,
            seq=int(getattr(frame, "seq", 0)),
            timestamp_s=float(getattr(frame, "timestamp_s", time.time())),
            contact_cell_count=cnt,
            max_response=float(delta.max()),  # peak delta in counts
            centroid_uv=centroid,
        )

    def is_in_contact(
        self, side: str = "left", min_cells: Optional[int] = None
    ) -> Optional[bool]:
        """
        True if at least ``min_cells`` cells are above contact
        threshold. Returns ``None`` if the sensor isn't connected
        (so callers can treat 'no data' as different from 'no contact').
        """
        if min_cells is None:
            min_cells = self.MIN_CONTACT_CELLS
        s = self.snapshot(side)
        if not s.connected:
            return None
        return s.contact_cell_count >= int(min_cells)

    def any_in_contact(self, min_cells: Optional[int] = None) -> Optional[bool]:
        """
        OR-reduction across all connected sensors.
        """
        if not self._sensors:
            return None
        return any(self.is_in_contact(side, min_cells) for side in self._sensors)


__all__ = ["TactileManager", "TactileSnapshot"]


if __name__ == "__main__":
    # Self-check for the rebaseline handshake with a fake sensor; no hardware.
    # Run: PYTHONPATH=src python -m spark_real.sensors.tactile

    class _FakeSensor:
        rows, cols = 2, 3
        baseline = np.zeros((2, 3))
        calibrations = 0

        def calibrate(self):
            self.calibrations += 1

        def read_latest(self):
            time.sleep(0.005)
            return None

    m = TactileManager(ports=[])
    m._sensors["left"] = _FakeSensor()
    m._rebaseline_req["left"] = threading.Event()
    m._rebaseline_done["left"] = threading.Event()
    threading.Thread(target=m._poll_loop, args=("left",), daemon=True).start()
    assert m.rebaseline() == {"left": True}
    assert m.rebaseline("left") == {"left": True}
    assert m._sensors["left"].calibrations == 2
    assert m.rebaseline("right") == {"right": False}
    m.stop()

    class _BrokenSensor(_FakeSensor):
        def calibrate(self):
            raise RuntimeError("serial timeout")

    m = TactileManager(ports=[])
    m._sensors["left"] = _BrokenSensor()
    m._rebaseline_req["left"] = threading.Event()
    m._rebaseline_done["left"] = threading.Event()
    threading.Thread(target=m._poll_loop, args=("left",), daemon=True).start()
    assert m.rebaseline("left") == {"left": False}
    m.stop()
    print("tactile rebaseline self-check ok")

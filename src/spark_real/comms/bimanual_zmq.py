"""
ZMQ pub/sub bridge for the bimanual rig.

Two services are exposed:

* :class:`BimanualStateBroadcaster` runs in a daemon thread, polls
  the bimanual driver at the configured rate, and publishes the
  full :class:`BimanualObservation` (as JSON) on ``tcp://*:5601``
  under topic ``b'bimanual_state'``.

* :class:`BimanualCommandBus` is a pull socket on ``tcp://*:5602`` for
  external nodes (a VLA controller, a teleop pendant) to push
  per-arm command dicts. The bus is intentionally pull-style so the
  bimanual executor is the single source of motion authority.

Both services need ``pyzmq``. If it is missing, the broadcaster
degrades to a no-op and the command bus refuses to start.
"""

from __future__ import annotations

import json
import logging
import threading
import time
from typing import Any, Callable, Dict, Optional

try:
    import zmq  # type: ignore
except ImportError:
    zmq = None

logger = logging.getLogger(__name__)


# State broadcaster
class BimanualStateBroadcaster:
    """
    Daemon thread that periodically publishes the bimanual observation.
    """

    def __init__(self, driver, port: int = 5601, rate_hz: float = 30.0):
        self.driver = driver
        self.port = int(port)
        self.rate_hz = float(rate_hz)
        self._thread: Optional[threading.Thread] = None
        self._stop = threading.Event()
        self._sock = None
        self._enabled = False

    def start(self) -> bool:
        if zmq is None:
            logger.info("pyzmq not installed; BimanualStateBroadcaster disabled")
            return False
        ctx = zmq.Context.instance()
        self._sock = ctx.socket(zmq.PUB)
        self._sock.bind(f"tcp://*:{self.port}")
        self._enabled = True
        self._stop.clear()
        self._thread = threading.Thread(
            target=self._loop, name="bimanual-state-pub", daemon=True
        )
        self._thread.start()
        logger.info(
            "BimanualStateBroadcaster started on tcp://*:%d at %.1f Hz",
            self.port,
            self.rate_hz,
        )
        return True

    def stop(self) -> None:
        self._stop.set()
        if self._thread is not None:
            self._thread.join(timeout=1.0)
        if self._sock is not None:
            try:
                self._sock.close(linger=100)
            except Exception:
                pass
        self._enabled = False

    def _loop(self) -> None:
        period = 1.0 / max(1.0, self.rate_hz)
        while not self._stop.is_set():
            t0 = time.time()
            try:
                obs = self.driver.get_observation()
                payload = obs.to_dict() if hasattr(obs, "to_dict") else dict(obs)
                # Numpy arrays don't serialize directly; convert.
                payload = _jsonable(payload)
                self._sock.send_multipart(
                    [
                        b"bimanual_state",
                        json.dumps(payload).encode("utf-8"),
                    ]
                )
            except Exception:
                logger.exception("state broadcast tick failed")
            elapsed = time.time() - t0
            if elapsed < period:
                self._stop.wait(period - elapsed)


# Command bus
class BimanualCommandBus:
    """
    Pull socket for external nodes to push per-arm command dicts.
    """

    def __init__(self, port: int = 5602):
        self.port = int(port)
        self._thread: Optional[threading.Thread] = None
        self._stop = threading.Event()
        self._sock = None
        self._handler: Optional[Callable[[Dict[str, Any]], None]] = None

    def start(self, handler: Callable[[Dict[str, Any]], None]) -> bool:
        if zmq is None:
            logger.info("pyzmq not installed; BimanualCommandBus disabled")
            return False
        ctx = zmq.Context.instance()
        self._sock = ctx.socket(zmq.PULL)
        self._sock.bind(f"tcp://*:{self.port}")
        self._handler = handler
        self._stop.clear()
        self._thread = threading.Thread(
            target=self._loop, name="bimanual-cmd-pull", daemon=True
        )
        self._thread.start()
        logger.info("BimanualCommandBus started on tcp://*:%d", self.port)
        return True

    def stop(self) -> None:
        self._stop.set()
        if self._thread is not None:
            self._thread.join(timeout=1.0)
        if self._sock is not None:
            try:
                self._sock.close(linger=100)
            except Exception:
                pass

    def _loop(self) -> None:
        poller = zmq.Poller()
        poller.register(self._sock, zmq.POLLIN)
        while not self._stop.is_set():
            events = dict(poller.poll(timeout=250))
            if self._sock in events:
                raw = self._sock.recv()
                try:
                    cmd = json.loads(raw.decode("utf-8"))
                except Exception:
                    logger.exception("invalid command payload")
                    continue
                try:
                    if self._handler:
                        self._handler(cmd)
                except Exception:
                    logger.exception("command handler raised")


def _jsonable(obj):
    """
    Recursively convert numpy / dataclass / unknown objects to JSON types.
    """
    try:
        import numpy as np
    except ImportError:
        return obj
    if isinstance(obj, dict):
        return {k: _jsonable(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [_jsonable(v) for v in obj]
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    if hasattr(obj, "to_dict"):
        return _jsonable(obj.to_dict())
    if isinstance(obj, (np.floating,)):
        return float(obj)
    if isinstance(obj, (np.integer,)):
        return int(obj)
    return obj

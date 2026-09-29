"""Per-subsystem liveness, fsync'd, for attributing a silent host death.

The host's hard deaths leave no panic, no oops, no MCE, and an empty
pstore; kdump is armed and has never fired. Host telemetry 15 s before one
end showed 45 C, 28 W, load 1.5, 15 of 63 GB used: a perfectly healthy
machine stopped executing.

30 s granularity cannot answer the one question that would narrow this: WHICH
subsystem stopped first. Every camera interface on this host has
Driver=[none] -- uvcvideo is blacklisted and pyk4a/librealsense drive the
xHCI controller from userspace over usbfs -- so our own threads are the USB
driver, and their relative order of death is real evidence.

Each watched loop calls tick(). A daemon thread writes every name's age plus
the kernel's USB error counters once a second and fsyncs, so the last line on
disk is at most one second before the machine stopped.

Deliberately cheap: tick() is a single dict store (atomic under the GIL, no
lock), so it adds nothing measurable to the loops it watches -- which are the
loops under suspicion.
"""

import json
import logging
import os
import threading
import time
from pathlib import Path

_BEATS = {}
_STARTED = False
SOURCE = "server"
_LOCK = threading.Lock()

DEFAULT_PATH = Path(__file__).parent / "output" / "logs" / "liveness.jsonl"
PERIOD_S = 1.0
# Keep the file bounded; a week of 1 Hz lines is ~50 MB, and only the tail
# ever matters.
MAX_BYTES = 32 * 1024 * 1024


def tick(name: str) -> None:
    """Record that ``name`` is alive. Called from the watched loop itself."""
    _BEATS[name] = time.time()


def _usb_error_counts():
    """xHCI/usbfs error counters the kernel keeps, if this host exposes them."""
    out = {}
    try:
        for dev in Path("/sys/bus/usb/devices").glob("*/"):
            for f in ("urbnum",):
                p = dev / f
                if p.exists():
                    try:
                        out[dev.name] = int(p.read_text().strip())
                    except (OSError, ValueError):
                        pass
    except OSError:
        pass
    return out


# xHCI ring desync: the controller reports a completion for a DMA address the
# driver does not recognise. Measured across this host's boots, it appears ONLY
# in boots that later died. It precedes death by minutes to days, which
# makes it the one early warning available -- worth surfacing loudly rather
# than leaving in the kernel log for a post-mortem.
_XHCI_PATTERNS = ("not part of TD", "Set TR Deq", "CLEAR_HALT")
_XHCI_CHECK_EVERY = 30  # seconds


def _xhci_desync_since(seconds):
    """Count xHCI ring-desync lines in the last ``seconds`` of kernel log."""
    try:
        import subprocess

        out = subprocess.run(
            ["journalctl", "-k", "--since", f"-{int(seconds)}s", "--no-pager", "-q"],
            capture_output=True, text=True, timeout=5,
        ).stdout
        return sum(1 for ln in out.splitlines() if any(p in ln for p in _XHCI_PATTERNS))
    except Exception:  # noqa: BLE001
        return 0


def _writer(path: Path):
    path.parent.mkdir(parents=True, exist_ok=True)
    _since_check = 0.0
    _desync = 0
    while True:
        try:
            if path.exists() and path.stat().st_size > MAX_BYTES:
                path.rename(path.with_suffix(".jsonl.1"))
            now = time.time()
            _since_check += PERIOD_S
            if _since_check >= _XHCI_CHECK_EVERY:
                _since_check = 0.0
                _desync = _xhci_desync_since(_XHCI_CHECK_EVERY + 5)
                if _desync:
                    # Loud on purpose: this is the operator's cue to stop and
                    # power-cycle deliberately rather than be surprised later.
                    logging.getLogger("spark_server").critical(
                        "[xhci] %d ring-desync line(s) in the last %ds -- the "
                        "USB controller and driver have diverged. On this host "
                        "that has only ever appeared in boots that went on to "
                        "hang. Finish the run and reboot cleanly.",
                        _desync, _XHCI_CHECK_EVERY,
                    )
            row = {
                "t": round(now, 3),
                "xhci_desync": _desync,
                # Which process wrote this line. The standalone service runs
                # ALWAYS (systemd, Restart=always) so a death with the server
                # down is still witnessed; the server process contributes the
                # per-subsystem ticks that only exist inside it. Both append
                # to the same file -- single small writes under O_APPEND are
                # atomic, so the lines interleave safely.
                "src": SOURCE,
                # Age, not timestamp: a stalled subsystem shows up as a
                # growing number instead of a value you have to subtract.
                "age": {k: round(now - v, 2) for k, v in list(_BEATS.items())},
                "urb": _usb_error_counts(),
            }
            with open(path, "a") as fh:
                fh.write(json.dumps(row) + "\n")
                fh.flush()
                os.fsync(fh.fileno())  # the whole point: survive a power-off
        except Exception:  # noqa: BLE001 - a health logger must never raise
            pass
        time.sleep(PERIOD_S)


def start(path=None) -> None:
    """Idempotent. Safe to call from every subsystem that ticks."""
    global _STARTED
    with _LOCK:
        if _STARTED:
            return
        _STARTED = True
    t = threading.Thread(
        target=_writer, args=(Path(path or DEFAULT_PATH),),
        name="spark-liveness", daemon=True,
    )
    t.start()


if __name__ == "__main__":
    # Standalone mode: the restart-durable half. Records URB counters and its
    # own heartbeat once a second whether or not the SPARK server is running:
    # one death followed 3.7 days during which the host was never idle, which
    # an instrument that only runs with the server would have missed.
    import sys

    SOURCE = "standalone"
    tick("standalone")
    start(sys.argv[1] if len(sys.argv) > 1 else None)
    while True:
        tick("standalone")
        time.sleep(0.5)

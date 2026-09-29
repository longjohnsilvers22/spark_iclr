"""
Lifetime of the UR10e URScript socket under concurrent callers.

``_send_script`` resolves ``self._urscript_socket`` and then calls
``sendall()`` on it, both inside ``_script_lock``. Every other toucher of
that attribute skipped the lock, so the descriptor could be freed between
those two steps::

    executor thread                     shutdown / abort thread
    ---------------                     -----------------------
    with _script_lock:
      sock = self._urscript_socket
                                        sock.close()        # fd N freed
                                        self._urscript_socket = None
                                        (any thread that open()s now gets N)
      sock.sendall(script)              # multi-KB URScript program written
                                        # into fd N -- now an unrelated
                                        # socket, log file or device node

close() frees the number immediately and the kernel hands back the lowest
free descriptor, so this is a cross-descriptor write, not just a failed
send. Both closers were reachable during any movel/movej: /api/stop and
/api/abort are async handlers running on the event loop while the executor
drives the arm from its own thread, and pipeline.shutdown() -> disconnect()
runs from the SIGTERM handler.

disconnect() had a second ordering hole: it cleared ``_connected`` LAST, so
for the whole teardown window ``_check_connected()`` still said yes and a
concurrent reader (the demo recorder polls get_observation at 15 Hz) went
straight into ``self._rtde_r.getActualQ()`` on an interface that
``rtde_r.disconnect()`` was tearing down.

All SDK objects are fakes; nothing touches a robot or a real socket.
"""

import threading

import pytest

from spark_real.control.command_latch import CommandLatch
from spark_real.control.ur10e_driver import UR10eDriver


class _FakeSocket:
    """Records whether it was already closed when bytes were written to it."""

    def __init__(self, on_send=None):
        self.closed = False
        self.close_calls = 0
        self.sent = []
        self.closed_at_write = None
        self._on_send = on_send

    def sendall(self, payload):
        if self._on_send is not None:
            self._on_send()
        # The observation that matters: was this descriptor still ours?
        self.closed_at_write = self.closed
        self.sent.append(payload)

    def close(self):
        self.close_calls += 1
        self.closed = True


class _FakeRtdeC:
    def __init__(self, log):
        self._log = log

    def stopScript(self):
        self._log.append("rtde_c.stopScript")

    def disconnect(self):
        self._log.append("rtde_c.disconnect")


class _FakeRtdeR:
    def __init__(self, log):
        self._log = log
        self.disconnected = False

    def disconnect(self):
        self._log.append("rtde_r.disconnect")
        self.disconnected = True

    def getActualQ(self):
        if self.disconnected:
            # ur_rtde reads its state buffer through a pointer disconnect()
            # drops; in C++ that is a segfault, not an exception. Raising is
            # the loudest thing a fake can do.
            raise AssertionError(
                "getActualQ() reached a DISCONNECTED RTDE receive interface"
            )
        self._log.append("rtde_r.getActualQ")
        return [0.0] * 6


def _driver(sock=None, rtde_log=None):
    """A UR10eDriver with every SDK handle faked. __init__ needs ur-rtde
    installed, which the audit does not, so the fields are set directly."""
    d = object.__new__(UR10eDriver)
    d.robot_ip = "0.0.0.0"
    d.frequency = 500.0
    d._script_lock = threading.Lock()
    d._urscript_socket = sock
    d._connected = True
    d._urscript_down = False
    d._gripper_script_header = None
    d._gripper_initialized = False
    d._command_latch = CommandLatch()
    d._motion_lease_until = -1.0
    d._last_motion_kind = None
    d._last_joint_vel = 0.0
    d._last_linear_vel = 0.0
    d._rtde_c = None
    d._rtde_r = None
    d._rtde_io = None
    if rtde_log is not None:
        d._rtde_c = _FakeRtdeC(rtde_log)
        d._rtde_r = _FakeRtdeR(rtde_log)
    return d


def _run(fn):
    t = threading.Thread(target=fn, daemon=True)
    t.start()
    return t


# 1. No closer may free the descriptor while a sender holds _script_lock


def _sender_vs_closer(closer):
    """
    Pin a sender inside sendall(), fire ``closer`` from another thread, and
    report whether the socket had already been closed when the bytes landed.

    ``proceed`` is released on a fixed budget from the main thread, so the
    closer gets a real window in which to do damage; nothing in the ordering
    depends on a sleep racing another sleep.
    """
    in_send = threading.Event()
    proceed = threading.Event()

    def _on_send():
        in_send.set()
        proceed.wait(timeout=2.0)

    sock = _FakeSocket(on_send=_on_send)
    rtde_log = []
    d = _driver(sock, rtde_log)
    # The brake at the top of disconnect() has ALREADY completed and released
    # _script_lock -- that is what makes the window reachable. Stubbing it
    # models exactly that, and stops the test from accidentally proving
    # nothing more than "brake blocks on the lock it takes itself".
    d.brake = lambda *a, **k: True

    sent_ok = []
    sender = _run(lambda: sent_ok.append(d._send_script("movej([0,0,0,0,0,0])")))
    assert in_send.wait(2.0), "sender never reached sendall"

    closer_done = threading.Event()

    def _close():
        closer(d)
        closer_done.set()

    closer_thread = _run(_close)
    # Give the closer a generous window to free the fd if it is unguarded.
    closer_done.wait(0.25)
    closed_early = sock.closed
    proceed.set()
    sender.join(timeout=3.0)
    closer_thread.join(timeout=3.0)
    return sock, d, sent_ok, closed_early


def test_disconnect_does_not_close_the_socket_under_a_live_sender():
    sock, d, sent_ok, closed_early = _sender_vs_closer(lambda drv: drv.disconnect())
    assert closed_early is False, (
        "disconnect() freed the URScript descriptor while another thread was "
        "inside sendall() on it"
    )
    assert sock.closed_at_write is False, (
        "URScript bytes were written to an already-closed descriptor; the fd "
        "number is reusable the instant close() returns"
    )
    assert sent_ok == [True]
    # The teardown still happens, just after the sender is out.
    assert sock.closed is True
    assert d._urscript_socket is None


def test_emergency_stop_stale_socket_drop_respects_the_send_lock():
    """
    emergency_stop() drops a stale socket between brake attempts. That drop
    ran unlocked, and /api/stop reaches it from the event loop while teleop,
    the gripper publisher or the primitive-timeout watchdog may hold the lock.
    """
    d = _driver(_FakeSocket())
    dropped = threading.Event()
    assert d._script_lock.acquire(timeout=1.0)
    try:
        t = _run(lambda: (d._drop_urscript_socket(), dropped.set()))
        assert dropped.wait(0.25) is False, (
            "_drop_urscript_socket closed the socket without _script_lock"
        )
    finally:
        d._script_lock.release()
    assert dropped.wait(2.0) is True
    t.join(timeout=2.0)
    assert d._urscript_socket is None


def test_drop_is_idempotent_and_closes_once():
    sock = _FakeSocket()
    d = _driver(sock)
    d._drop_urscript_socket()
    d._drop_urscript_socket()
    assert sock.close_calls == 1
    assert d._urscript_socket is None


# 2. disconnect() must close the door before demolishing the room


def test_disconnect_clears_connected_before_tearing_down_rtde():
    """
    A reader that passes _check_connected() must not then land in a
    half-disconnected RTDE interface. Clearing the flag first turns the
    call into a clean RuntimeError instead.
    """
    rtde_log = []
    d = _driver(_FakeSocket(), rtde_log)
    observed = {}

    def _watch_stop_script():
        # Runs at the first teardown step; by now _connected must be False.
        observed["connected_at_teardown"] = d._connected
        with pytest.raises(RuntimeError, match="Not connected"):
            d.get_joint_positions()

    d._rtde_c.stopScript = _watch_stop_script
    d.brake = lambda *a, **k: True
    d.disconnect()

    assert observed["connected_at_teardown"] is False
    assert d._connected is False


def test_reader_arriving_during_teardown_is_rejected_at_the_check():
    """
    A reader that reaches _check_connected() at or after the first teardown
    step must be turned away.

    The demo recorder polls get_observation() at 15 Hz from its own thread
    for a whole episode, and pipeline.shutdown() -> disconnect() lands in the
    middle of it. Pinned here at the exact instant teardown begins.

    This is what the ordering change buys and no more: a reader that had
    ALREADY passed the check before teardown started is still on its way into
    the interface. Closing that window needs a reader-writer lease over the
    RTDE handles -- see docs/CONCURRENCY_AUDIT.md.
    """
    rtde_log = []
    d = _driver(_FakeSocket(), rtde_log)
    d.brake = lambda *a, **k: True
    teardown_started = threading.Event()
    reader_done = threading.Event()
    outcome = {}

    real_stop_script = d._rtde_c.stopScript

    def _stop_script():
        teardown_started.set()
        reader_done.wait(timeout=2.0)
        real_stop_script()

    d._rtde_c.stopScript = _stop_script

    def _poll():
        teardown_started.wait(2.0)
        try:
            d.get_joint_positions()
            outcome["result"] = "accepted"
        except RuntimeError:
            outcome["result"] = "rejected"
        except AssertionError as exc:  # reached a torn-down interface
            outcome["result"] = f"reached-dead-interface: {exc}"
        reader_done.set()

    poller = _run(_poll)
    d.disconnect()
    poller.join(timeout=3.0)

    assert outcome["result"] == "rejected", (
        "a reader was let through _check_connected() after teardown began: "
        "_connected was still True while the RTDE handles were being "
        "destroyed"
    )
    assert "rtde_r.disconnect" in rtde_log


# 3. The brake still reaches the arm through a healthy socket


def test_send_script_still_sends_on_a_healthy_socket():
    sock = _FakeSocket()
    d = _driver(sock)
    assert d._send_script("stopj(2.000)") is True
    assert sock.sent == [b"stopj(2.000)\n"]
    assert sock.closed_at_write is False


if __name__ == "__main__":  # pragma: no cover
    pytest.main([__file__, "-v"])

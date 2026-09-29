"""Fake-socket brake harness: the REAL UR10eDriver -> a fake wire -> the plant.

Why the driver has to be in the loop. ``UrPlant.stop()`` sends ``stopj`` over
its own wire, while ``UR10eDriver.stop()`` was ``rtde_c.stopJ`` -- a register
write nobody reads. So any brake test that calls ``plant.stop()`` passes
whether the driver is fixed or not. Here the plant only supplies physics: the
socket, the send path, the escalation and the decel arithmetic are the driver's
own code, exercised end to end.

Two fidelity additions over ``UrPlant``:

  * ``stopj(a)`` / ``stopl(a)`` DECELERATE at ``a`` instead of teleporting the
    speed to zero. Overshoot (v**2/2a) and stopping time (v/a) become
    measurable, which is what separates a real brake from a flag.
  * a program replaced before its start lag has elapsed never moves at all,
    which is what a stop during the ~1 s URScript upload actually does.
"""

from __future__ import annotations

import re
import time
from typing import List

import numpy as np

from spark_real.control.ur10e_driver import UR10eDriver
from spark_real.tests.blend_plant import UrPlant

_STOP_RE = re.compile(r"^stop([jl])\(\s*([-\d.eE+]+)")


class DecelPlant(UrPlant):
    """UrPlant whose stops decelerate, and whose ``stop()`` is HEAD-faithful."""

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self._braking = False
        self._brake_a = 0.0
        # Times UR10eDriver.stop()'s OLD body (rtde_c.stopJ) was invoked.
        self.rtde_stopj_calls = 0

    # driver-faithful no-op stop

    def stop(self):
        """What ``rtde_c.stopJ(2.0)`` does on this rig: nothing.

        stopJ writes an RTDE input register that ur_rtde's control script
        polls, and _send_script kills that control script before every move.
        Counted, not applied -- an offline test must not be able to pass by
        calling this.
        """
        self.rtde_stopj_calls += 1
        return None

    # measurement surface

    def joint_speed(self) -> float:
        """Leading-joint speed, rad/s (0.0 in Cartesian velocity mode)."""
        self._integrate()
        return 0.0 if self._cart is not None else float(self._speed)

    def tcp_speed_vec(self) -> np.ndarray:
        self._integrate()
        if self._cart is not None:
            return np.asarray(self._cart_v, dtype=float).copy()
        return np.zeros(6)

    def is_program_running(self) -> bool:
        self._integrate()
        return bool(self._rows) or self._cart is not None

    # physics

    def _send_script(self, script: str) -> bool:
        m = _STOP_RE.match(script.strip())
        if m is None:
            return super()._send_script(script)
        self._integrate()
        with self._lock:
            self.sent.append(script)
            a = max(float(m.group(2)), 1e-6)
            starting = (
                self._rows and self._start_at is not None and time.monotonic() < self._start_at
            )
            if starting:
                # Replaced during the upload lag: the arm never moved.
                self._rows = []
                self._idx = -1
                self._speed = 0.0
                self._start_at = None
                self._braking = False
                return True
            self._braking = True
            self._brake_a = a
            if self._cart is not None:
                self._cart_cmd = np.zeros(6)
                self._cart_acc = a
            return True

    def _step(self, dt):
        if not self._braking:
            return super()._step(dt)
        if self._cart is not None:
            # Base class ramps _cart_v toward _cart_cmd (zero) at _cart_acc.
            super()._step(dt)
            if float(np.linalg.norm(self._cart_v)) <= 1e-9:
                self._braking = False
            return
        row = self._rows[self._idx] if 0 <= self._idx < len(self._rows) else None
        if row is None or self._speed <= 0.0:
            self._speed = 0.0
            self._rows = []
            self._idx = -1
            self._braking = False
            return
        delta = row.q - self._q_from
        L = float(np.max(np.abs(delta)))
        if L < 1e-9:
            self._speed = 0.0
            self._rows = []
            self._idx = -1
            self._braking = False
            return
        self._speed = max(0.0, self._speed - self._brake_a * dt)
        # _u may pass L: that IS the overshoot past the waypoint.
        self._u += self._speed * dt
        self.q = self._q_from + delta * (self._u / L)
        if self._speed <= 0.0:
            self._rows = []
            self._idx = -1
            self._braking = False


class WireSocket:
    """Stand-in for the driver's port-30002 socket."""

    def __init__(self, plant, log: List[str], fail: bool = False):
        self.plant = plant
        self.log = log
        self.fail = fail
        self.closed = False

    def send(self, data: bytes) -> int:
        if self.fail:
            raise BrokenPipeError("test: URScript socket down")
        text = data.decode()
        self.log.append(text)
        self.plant._send_script(text)
        return len(data)

    def sendall(self, data: bytes) -> None:
        # The driver uses sendall (partial sends truncate multi-KB gripper
        # scripts); mirror the real socket API on the test double.
        self.send(data)

    def close(self):
        self.closed = True

    def settimeout(self, _t):
        pass


class PlantRtdeR:
    """The rtde_receive surface the driver reads, backed by the plant."""

    def __init__(self, plant):
        self.plant = plant

    def getActualQ(self):
        return list(self.plant.get_joint_positions())

    def getActualQd(self):
        return [0.0] * 6

    def getActualTCPPose(self):
        return list(self.plant.get_tcp_pose())

    def getActualTCPSpeed(self):
        return list(self.plant.tcp_speed_vec())

    def getActualTCPForce(self):
        return [0.0] * 6

    def getRobotMode(self):
        return 7

    def isSteady(self):
        return self.plant.joint_speed() <= 1e-9

    def disconnect(self):
        pass


class RecordingRtdeC:
    """rtde_control stand-in that records the calls HEAD relied on."""

    def __init__(self):
        self.stopJ_calls = []
        self.stopScript_calls = 0

    def stopJ(self, a=2.0):
        self.stopJ_calls.append(float(a))

    def stopL(self, a=2.0):
        self.stopJ_calls.append(float(a))

    def stopScript(self):
        self.stopScript_calls += 1

    def servoStop(self):
        pass

    def disconnect(self):
        pass


class BrakeRig:
    """A real UR10eDriver bolted to a DecelPlant through a fake socket."""

    def __init__(self, q0=None, start_lag: float = 0.05, rtde_c=None):
        self.plant = DecelPlant(q0=q0)
        self.plant.START_LAG_S = start_lag
        self.frames: List[str] = []
        self.reopens = 0
        self.reopen_ok = True
        # Number of leading _open_urscript_socket calls that fail regardless of
        # reopen_ok, so the driver's own retry can be starved without making
        # the socket permanently unrecoverable.
        self.reopen_fail_first = 0
        self.wire = WireSocket(self.plant, self.frames)
        self.driver = self._make_driver(rtde_c)

    def _make_driver(self, rtde_c):
        import spark_real.control.ur10e_driver as mod

        # __init__ only *guards* on HAS_RTDE; it opens nothing. Flipping the
        # flag keeps the real constructor (and every field it sets) in play on
        # a machine without ur_rtde installed.
        had = mod.HAS_RTDE
        mod.HAS_RTDE = True
        try:
            d = UR10eDriver("127.0.0.1")
        finally:
            mod.HAS_RTDE = had
        d._connected = True
        d._urscript_socket = self.wire
        d._rtde_r = PlantRtdeR(self.plant)
        # None is the steady state on this rig: the first _send_script stops
        # the control script and it is never re-uploaded.
        d._rtde_c = rtde_c
        d._tcp_offset_cache = np.asarray(self.plant.tcp_offset, dtype=float)
        d._open_urscript_socket = self._reopen
        return d

    def _reopen(self):
        """What the driver calls when its socket is gone."""
        self.reopens += 1
        if not self.reopen_ok or self.reopens <= self.reopen_fail_first:
            self.driver._urscript_socket = None
            return
        self.wire = WireSocket(self.plant, self.frames)
        self.driver._urscript_socket = self.wire

    # helpers

    def brake_frames(self):
        return [f for f in self.frames if f.strip().startswith(("stopj", "stopl"))]

    def q(self):
        return self.plant.get_joint_positions()

    def spin(self, seconds: float, hz: float = 200.0):
        """Let the plant integrate for `seconds` of wall time."""
        t_end = time.monotonic() + seconds
        while time.monotonic() < t_end:
            self.plant.get_joint_positions()
            time.sleep(1.0 / hz)

    def spin_until_moving(self, joint: int = 0, timeout: float = 3.0) -> float:
        """Block until the arm is measurably moving; return its speed."""
        q0 = self.q()
        t_end = time.monotonic() + timeout
        while time.monotonic() < t_end:
            time.sleep(0.005)
            if abs(self.q()[joint] - q0[joint]) > 1e-4 and self.plant.joint_speed() > 0:
                return self.plant.joint_speed()
        raise AssertionError("plant never started moving")


def target_away(q0, joint: int = 0, delta: float = 1.1) -> List[float]:
    """A joint target `delta` rad away on `joint` -- long enough to reach v."""
    q = list(np.asarray(q0, dtype=float))
    q[joint] += delta
    return q


def moving_rig(velocity: float = 1.05, acceleration: float = 1.4, delta: float = 3.0):
    """A rig whose arm is already moving at `velocity` under a movej."""
    rig = BrakeRig(start_lag=0.05)
    q_target = target_away(rig.q(), delta=delta)
    rig.driver.move_to_joint_config_urscript(q_target, velocity=velocity, acceleration=acceleration)
    rig.spin_until_moving()
    # Let it reach cruise: v/a seconds of ramp, plus a margin.
    rig.spin(velocity / acceleration + 0.05)
    return rig, np.asarray(q_target, dtype=float)

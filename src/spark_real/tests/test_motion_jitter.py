"""
Regressions for the "jackhammer" slowdown (plushie pick-place, 50-75 s).

Four independent faults conspired, and each one gets a test here:

1. The demo recorder's proprio read reached UR10eDriver.get_gripper_position,
   which uploads a URScript program. On UR that REPLACES the running motion
   program, so a 15 Hz recorder cancelled the in-flight move several times a
   second and the arm re-accelerated on the next tick.
2. The only guard against that was measured TCP speed > 0.01 m/s, which is
   blind both to a servo converging (speed -> 0 while still commanding) and to
   a fire-and-forget movej inside its ~1 s upload lag.
3. Every gripper publish paid a flat 0.2 s sleep, and one logical read paid it
   up to three times.
4. CartesianServo hammered an unreachable target for its full 15 s timeout.

Plus the guard for the premature-arrival bug in _wait_for_motion, which must
never come back: stillness may only mean "arrived" if motion was observed.
"""

import time

import numpy as np
import pytest

from spark_real.control.cartesian_servo import CartesianServo
from spark_real.control.command_latch import CommandLatch
from spark_real.control.executor_grasp import GraspMixin
from spark_real.control.executor_motion import MotionMixin
from spark_real.control.ur10e_driver import UR10eDriver
from spark_real.recording.proprio import ProprioReader

ORIENT = [2.3038, 2.0802, -0.0048]


class _FakeRTDER:
    def __init__(self):
        self.speed = [0.0] * 6
        self.regs = {12: 40, 13: 1, 14: 0}

    def getActualTCPSpeed(self):
        return list(self.speed)

    def getOutputIntRegister(self, i):
        return self.regs.get(i, 0)

    def getActualQ(self):
        return [0.0] * 6

    def getActualQd(self):
        return [0.0] * 6

    def getActualTCPPose(self):
        return [0.0] * 6


def _driver():
    """A UR10eDriver with the sockets stubbed out. Counts scripts by kind."""
    d = UR10eDriver.__new__(UR10eDriver)
    d._connected = True
    d._rtde_c = None
    d._rtde_io = None
    d._gripper_script_header = "  # header"
    d._motion_lease_until = -1.0
    d._command_latch = CommandLatch()
    d._rtde_r = _FakeRTDER()
    d._check_connected = lambda: None
    d.sent = []

    def send(script):
        d.sent.append(script)
        d._note_motion_script(script)
        # Emulate the controller acknowledging the publish immediately.
        if "_pub_grip" in script:
            d._rtde_r.regs[14] = d._publish_seq
        return True

    d._send_script = send
    return d


def _publishes(d):
    return sum("_pub_grip" in s for s in d.sent)


# 1. the recorder must never upload URScript


def test_recorder_proprio_read_uploads_no_urscript():
    """The demo recorder samples state; it must not command the robot."""
    d = _driver()
    row = ProprioReader(d).read()
    assert _publishes(d) == 0, (
        "ProprioReader uploaded a URScript program. On UR this replaces the "
        "running movej/speedl and stops the arm mid-motion."
    )
    # The diagnostic channel still arrives, from the last-published register.
    assert row["gripper_measured"] == pytest.approx(40 * 2.55 / 255.0)


def test_get_observation_is_passive():
    d = _driver()
    obs = d.get_observation()
    assert _publishes(d) == 0
    assert obs["gripper_position"] == pytest.approx(40 * 2.55)


# 2. the motion lease


def test_publish_refused_while_servo_converging():
    """The hole the old speed-only guard could not see.

    A converging servo drops below 0.01 m/s while still emitting speedl every
    8 ms. A publish landing here cancels it.
    """
    d = _driver()
    d._rtde_r.speed = [0.001, 0, 0, 0, 0, 0]  # essentially stopped, but...
    d._send_script("speedl([0.002,0,0,0,0,0], 1.0, 0.028)")  # ...still commanding
    d.sent.clear()
    d._publish_gripper_state()
    assert _publishes(d) == 0


def test_publish_refused_during_movej_upload_lag():
    """A fire-and-forget movej is STATIONARY for ~1 s after the send."""
    d = _driver()
    d._rtde_r.speed = [0.0] * 6  # not moving yet: upload lag
    d._send_script("movej([0,0,0,0,0,0], 1.0, 1.4)")
    d.sent.clear()
    d._publish_gripper_state()
    assert _publishes(d) == 0


def test_publish_allowed_once_motion_is_done():
    """The lease must self-expire, or gripper reads would never refresh."""
    d = _driver()
    d._send_script("speedl([0.1,0,0,0,0,0], 1.0, 0.028)")
    d.sent.clear()
    d._motion_lease_until = time.monotonic() - 0.001  # lease lapsed
    d._rtde_r.speed = [0.0] * 6
    d._publish_gripper_state()
    assert _publishes(d) == 1


def test_forced_publish_beats_a_stale_lease_but_not_real_motion():
    """A drop check at a parked waypoint must get FRESH registers.

    The lease is sized for the worst-case upload lag, so a movej that lands
    early leaves it running. Without the force path, the very next
    _grip_intact would silently read a stale register and could invent a drop.
    """
    d = _driver()
    d._send_script("movej([0,0,0,0,0,0], 1.0, 1.4)")  # lease now held
    d.sent.clear()
    d._rtde_r.speed = [0.0] * 6  # but the arm has actually parked
    d._publish_gripper_state(force=True)
    assert _publishes(d) == 1

    # force must NOT be able to cancel a move that is genuinely running.
    d.sent.clear()
    d._rtde_r.speed = [0.2, 0, 0, 0, 0, 0]
    d._publish_gripper_state(force=True)
    assert _publishes(d) == 0


def test_confirmed_arrival_retires_the_lease():
    d = _driver()
    d._send_script("movej([0,0,0,0,0,0], 1.0, 1.4)")
    assert d._motion_in_flight()
    d.clear_motion_lease()
    assert not d._motion_in_flight()


def test_wait_for_motion_retires_the_lease_on_arrival():
    """End-to-end: the executor's arrival signal reaches the driver."""
    d = _driver()
    d._send_script("movej([0,0,0,0,0,0], 1.0, 1.4)")
    target = [0.5, 0.0, 0.13]
    arm = _Arm([0.0, 0.0, 0.35], target, lag=0.2, q_end=[0.3] * 6)
    w = _Waiter(arm)
    w.robot = arm
    arm.clear_motion_lease = d.clear_motion_lease
    w._wait_for_motion(np.array(target), timeout=10.0, q_target=[0.3] * 6)
    assert not d._motion_in_flight()


def test_a_gripper_script_does_not_take_a_motion_lease():
    """Only arm motion takes the lease; the gripper does not move the arm."""
    d = _driver()
    d._send_script("def rq_close():\n  # gripper\nend\nrq_close()\n")
    assert not d._motion_in_flight()


# 3. publish cost


def test_publish_waits_for_acknowledgement_not_a_fixed_sleep():
    d = _driver()
    t0 = time.perf_counter()
    d._publish_gripper_state()
    assert time.perf_counter() - t0 < 0.1, "still paying the flat 0.2 s sleep"


def test_publish_degrades_to_the_old_budget_without_register_14():
    """A controller that does not surface the token must not be FASTER-but-wrong."""
    d = _driver()
    d._rtde_r.getOutputIntRegister = lambda i: (_ for _ in ()).throw(RuntimeError())
    t0 = time.perf_counter()
    d._publish_gripper_state()
    assert time.perf_counter() - t0 >= d._PUBLISH_TIMEOUT_S * 0.9


class _GripStub(GraspMixin):
    GRIPPER_FULLY_CLOSED = 250
    GRIPPER_EMPTY_THRESHOLD = 5

    def __init__(self, gobj):
        self.publishes = 0
        self._gobj = gobj
        self.robot = self

    def _publish_gripper_state(self):
        self.publishes += 1

    def get_gripper_position(self, publish=True):
        self.publishes += publish
        return 200

    def is_object_detected(self, publish=True):
        self.publishes += publish
        return self._gobj

    def _gripper_type(self):
        return "robotiq_2f85"


@pytest.mark.parametrize("gobj", [True, False])
def test_grip_intact_costs_exactly_one_publish(gobj):
    """gObj and jaw position share ONE publish; it used to be up to three."""
    s = _GripStub(gobj)
    s._grip_intact()
    assert s.publishes == 1


def test_read_gripper_state_tolerates_a_no_kwarg_override():
    """Subclasses/doubles may override the getters with the old signature."""

    class Old(_GripStub):
        def _get_gripper_position(self):  # no publish kwarg, on purpose
            return 123

    s = Old(True)
    gobj, pos = s._read_gripper_state()
    assert pos == 123 and gobj is True


# 4. the servo must not hammer an unreachable target


class _Plant:
    """Accel-limited integrator with a hard floor at ``stop_z``."""

    SUPPORTS_URSCRIPT = True

    def __init__(self, start_z, stop_z):
        self.p = np.array([0.5, 0.0, start_z])
        self.v = np.zeros(6)
        self.cmd = np.zeros(6)
        self.stop_z = stop_z
        self.t = time.monotonic()
        self.ticks = 0

    def get_tcp_pose(self):
        now = time.monotonic()
        dt = min(now - self.t, 0.05)
        self.t = now
        dv = self.cmd - self.v
        step = 1.0 * dt
        n = np.linalg.norm(dv)
        self.v = self.cmd.copy() if n <= step else self.v + dv * (step / n)
        self.p = self.p + self.v[:3] * dt
        if self.p[2] < self.stop_z:
            self.p[2] = self.stop_z
            self.v[2] = 0.0
        return np.concatenate([self.p, ORIENT])

    def _send_script(self, script):
        self.ticks += 1
        inner = script[script.index("[") + 1 : script.index("]")]
        self.cmd = np.array([float(x) for x in inner.split(",")])
        return True


def _descend(stop_z):
    plant = _Plant(0.22, stop_z)
    servo = CartesianServo(plant, rate_hz=125.0)
    servo.max_vel_linear = 0.125
    t0 = time.time()
    ok = servo.move_to_pose([0.5, 0.0, 0.13] + ORIENT, velocity=0.125)
    return ok, time.time() - t0, plant.ticks, servo


def test_servo_gives_up_on_an_unreachable_target():
    """4 mm of unreachability used to cost 15.01 s of continuous speedl."""
    ok, elapsed, ticks, servo = _descend(0.134)
    assert ok is False
    assert servo.last_exit == "stalled"
    assert elapsed < 8.0, f"still hammering: {elapsed:.1f}s, {ticks} speedl"


def test_a_reachable_descent_is_unaffected():
    """The stall exit must not fire on a healthy, converging move."""
    ok, elapsed, _, servo = _descend(-1.0)
    assert ok is True
    assert servo.last_exit == "converged"
    assert servo.last_pos_err < servo.pos_threshold


# 5. the premature-arrival bug must never come back


class _Arm:
    def __init__(self, start, end, lag, travel=1.0, q_end=None):
        self.start = np.array(start, float)
        self.end = np.array(end, float)
        self.lag = lag
        self.travel = travel
        self.q_end = np.zeros(6) if q_end is None else np.array(q_end, float)
        self.t0 = time.time()

    def _frac(self):
        el = time.time() - self.t0
        return 0.0 if el < self.lag else min(1.0, (el - self.lag) / self.travel)

    def pos(self):
        return self.start + (self.end - self.start) * self._frac()

    def get_joint_positions(self):
        return self.q_end * self._frac()


class _Waiter(MotionMixin):
    def __init__(self, arm):
        self.robot = arm
        self.arm = arm
        self._recorder = None

    def _get_current_position(self):
        return self.arm.pos()

    def _check_abort(self):
        pass


def test_stillness_during_a_long_upload_lag_is_not_arrival():
    """The bug: returned ~20 cm short, and the gripper closed above the object.

    The lag here (2.5 s) deliberately exceeds URSCRIPT_START_LAG_S, so the arm
    is still stationary when the start-lag window closes. "Still" there means
    "not started yet".
    """
    target = [0.5, 0.0, 0.13]
    arm = _Arm([0.0, 0.0, 0.35], target, lag=2.5)
    _Waiter(arm)._wait_for_motion(np.array(target), timeout=8.0)
    assert np.linalg.norm(arm.pos() - np.array(target)) < 0.01


def test_joint_target_is_a_motion_history_free_exit():
    """The only sanctioned early exit: measured joints match the commanded."""
    target = [0.5, 0.0, 0.13]
    arm = _Arm([0.0, 0.0, 0.35], target, lag=0.3, q_end=[0.3] * 6)
    t0 = time.time()
    _Waiter(arm)._wait_for_motion(np.array(target), timeout=15.0, q_target=[0.3] * 6)
    assert time.time() - t0 < 4.0

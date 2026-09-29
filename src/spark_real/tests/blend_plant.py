"""Accel-limited UR10e plant for offline motion timing (no hardware).

Same idea as the ``_Plant`` in test_motion_jitter.py that a prior agent used to
measure the servo stall exit, scaled up: this one holds the ground truth for a
6-DOF arm and is driven by the ACTUAL URScript this codebase emits over
``_send_script``, so the executor under test is unmodified.

What is modelled, and why each piece is here:

  * ``movej`` as a trapezoidal profile on the LEADING joint (that is what UR's
    ``v``/``a`` apply to), so an isolated move costs
    ``dq/v + v/a`` -- exactly what waypoints.joint_move_seconds predicts.
  * ``r=`` blending: a row with r>0 does NOT decelerate. When the remaining
    Cartesian distance to that row's TCP target drops below r, the plant
    switches to the next row carrying its current speed. A junction that
    blends therefore saves the decel ramp plus the next accel ramp; a junction
    with r=0 pays both. That difference is the whole point of the exercise.
  * ONE program at a time. A new ``_send_script`` REPLACES the running program,
    which is the real controller behaviour and the reason blending has to be a
    single send -- and the reason a gripper publish mid-path is fatal. Gripper
    scripts are modelled as exactly that: they cancel the running motion.
  * ``speedl`` velocity mode with an acceleration limit, so the real
    CartesianServo (and its 4mm stall exit) runs against it unchanged.
  * The ~1 s URScript upload/start lag, during which the arm is stationary.
    This is what makes stillness-means-arrived unsafe.
  * Output integer register 15, written between rows, so the executor's
    progress/stall detection has something real to read.
  * The Robotiq 2F-85 jaws as an ASYNCHRONOUS travel on the controller: a jaw
    position that ramps at the commanded speed_norm, an object the jaws stop
    on, gObj that only reads True once they have stopped ON it, and the
    driver's own post-send sleep. This is what makes a settle constant
    falsifiable rather than a matter of opinion -- shorten a settle too far and
    the next program cancels the travel mid-stroke, the jaws freeze part-closed,
    gObj reads False and the grasp FAILS here, offline. Without this the plant
    could only ever agree that a shorter sleep is shorter.

Kinematics are the same pyroki UR10e model the executor solves against, with
the RTDE-frame bridge (Rz(180) base + pendant TCP offset) applied so
``get_tcp_pose`` returns gripper-TIP poses in the frame the executor targets.
"""

from __future__ import annotations

import re
import threading
import time

import numpy as np
from scipy.spatial.transform import Rotation

from spark_real.control import ur10e_ik_pyroki as ur_ik

# Robotiq 2F-85 on the UR10e wrist: tip 172.5 mm along tool0 Z.
TCP_OFFSET = [0.0, 0.0, 0.1725, 0.0, 0.0, 0.0]

_MOVEJ_RE = re.compile(
    r"movej\(\[([^\]]*)\]\s*,\s*a=([-\d.eE+]+)\s*,\s*v=([-\d.eE+]+)"
    r"(?:\s*,\s*r=([-\d.eE+]+))?\s*\)"
)
_REG_RE = re.compile(r"write_output_integer_register\(\s*(\d+)\s*,\s*(-?\d+)\s*\)")
_SPEEDL_RE = re.compile(r"speedl\(\[([^\]]*)\]")
_STOP_RE = re.compile(r"(stopj|stopl)\(\s*([-\d.eE+]+)?")
_MOVEL_RE = re.compile(r"move[jl]\(p\[([^\]]*)\]")
# One pass over a program, in emission order: register writes and movej rows.
_PROGRAM_RE = re.compile(
    r"write_output_integer_register\(\s*(?P<reg>\d+)\s*,\s*(?P<val>-?\d+)\s*\)"
    r"|movej\(\[(?P<q>[^\]]*)\]\s*,\s*a=(?P<a>[-\d.eE+]+)\s*,\s*v=(?P<v>[-\d.eE+]+)"
    r"(?:\s*,\s*r=(?P<r>[-\d.eE+]+))?\s*\)"
)


def fk_rtde(q, tcp_offset=TCP_OFFSET):
    """Gripper-TIP pose in the RTDE base frame: (pos[3], R[3x3])."""
    p_urdf, R_urdf = ur_ik.fk(np.asarray(q, dtype=float))
    Rz = ur_ik._rz180()
    p_t0 = Rz @ p_urdf
    R_t0 = Rz @ R_urdf
    tcp = np.asarray(tcp_offset, dtype=float).reshape(6)
    R_off = Rotation.from_rotvec(tcp[3:]).as_matrix()
    return p_t0 + R_t0 @ tcp[:3], R_t0 @ R_off


class _Row:
    __slots__ = ("q", "v", "a", "r", "reg_value")

    def __init__(self, q, v, a, r, reg_value=None):
        self.q = np.asarray(q, dtype=float)
        self.v = float(v)
        self.a = float(a)
        self.r = float(r)
        self.reg_value = reg_value


class UrPlant:
    """UR10e stand-in: ground truth joints, driven by real URScript."""

    SUPPORTS_URSCRIPT = True
    robot_family = "ur10e"
    HOME_CONFIG = [3.2070, -1.8788, -1.7903, 5.2496, 1.5762, 0.0916]

    # Program upload + start lag on port 30002. The arm does not move for this
    # long after the send, which is why URSCRIPT_START_LAG_S exists.
    START_LAG_S = 1.0
    # Synchronous cost of getting a gripper / register-publish program onto the
    # controller: the 11.8 kB URCap header has to be uploaded and compiled. The
    # jaw TRAVEL it starts is asynchronous (see the jaw model below), so this is
    # upload only -- it used to be 0.30 s standing in for both.
    GRIPPER_SCRIPT_S = 0.05
    # UR10eDriver.close_gripper / open_gripper each sleep this after the send
    # (ur10e_driver.py ~867 / ~934). Modelled here because the executor's settle
    # constants are sized against it.
    GRIPPER_DRIVER_SLEEP_S = 0.5
    SUBSTEP_S = 0.002

    # Robotiq 2F-85 jaw model. Counts 0 (open) .. 255 (closed), datasheet stroke
    # 85 mm at 20-150 mm/s over the speed_norm range.
    JAW_OPEN = 0.0
    JAW_CLOSED = 255.0
    JAW_STROKE_MM = 85.0
    JAW_SPEED_MIN_MM_S = 20.0
    JAW_SPEED_MAX_MM_S = 150.0
    # Jaw count at which a closing stroke meets the object and stops. None = no
    # object in the jaws, so a close runs all the way to 255 and gObj stays
    # False. 150 is roughly a 36 mm block; a soft plushie reads higher.
    OBJECT_AT_COUNTS = 150.0
    GRIPPER_TYPE = "robotiq_2f85"
    # Wrench noise (N, 1 sigma per axis) so _wait_force_settled has something to
    # settle ON rather than an exact constant.
    FORCE_NOISE_N = 0.05

    def __init__(self, q0=None, tcp_offset=TCP_OFFSET, floor_z=None, seed=0):
        self._rng = np.random.default_rng(seed)
        self.q = np.asarray(q0 if q0 is not None else self.HOME_CONFIG, dtype=float)
        self.tcp_offset = list(tcp_offset)
        # Hard mechanical stop for the TCP in velocity mode, so the servo's
        # unreachable-target stall exit can be exercised (as _Plant does in
        # test_motion_jitter.py). None = no floor.
        self.floor_z = floor_z
        self._lock = threading.RLock()
        self._t = time.monotonic()
        # movej program state
        self._rows: list[_Row] = []
        self._idx = -1
        self._q_from = self.q.copy()
        self._u = 0.0  # progress along the active row [rad, leading joint]
        self._speed = 0.0  # rad/s, leading joint
        self._start_at = None  # monotonic time the program may begin moving
        self._completion_reg = None
        self._row_len_cart = 0.0
        self._braking = False
        self._brake_decel = None
        self._last_joint_vel = self.DEFAULT_JOINT_VEL
        # speedl state (Cartesian velocity mode)
        self._cart = None  # (pos[3], R[3x3]) while in speedl mode
        self._cart_v = np.zeros(6)
        self._cart_cmd = np.zeros(6)
        self._cart_acc = 1.0
        self.registers = {}
        self.sent: list[str] = []
        self.gripper_cancels = 0
        self.script_time = 0.0  # wall time paid inside gripper scripts
        self._motion_lease_until = -1.0
        # jaw state (asynchronous, runs on the "controller")
        self.jaw = self.JAW_OPEN
        self._jaw_target = self.JAW_OPEN
        self._jaw_goal = self.JAW_OPEN  # where this stroke will actually stop
        self._jaw_rate = 0.0  # counts/s, 0 = parked
        self._jaw_obj = False  # gObj: stopped ON the object
        self._jaw_t = time.monotonic()
        self.jaw_cancels = 0  # strokes killed mid-travel by a new program

    # kinematics helpers

    def _tcp(self):
        if self._cart is not None:
            return np.asarray(self._cart[0]), np.asarray(self._cart[1])
        return fk_rtde(self.q, self.tcp_offset)

    # integration

    def _advance_row(self, keep_speed: bool):
        self._q_from = self.q.copy()
        self._idx += 1
        self._u = 0.0
        if not keep_speed:
            self._speed = 0.0
        if self._idx >= len(self._rows):
            if getattr(self, "_completion_reg", None) is not None:
                self.registers[self._completion_reg[0]] = self._completion_reg[1]
                self._completion_reg = None
            self._rows = []
            self._idx = -1
            self._speed = 0.0
            return
        rv = self._rows[self._idx].reg_value
        if rv is not None:
            self.registers[rv[0]] = rv[1]
        # Cache the row's straight-line TCP length once: the blend switch test
        # runs every 2 ms substep, and FK goes through JAX.
        p0, _ = fk_rtde(self._q_from, self.tcp_offset)
        p1, _ = fk_rtde(self._rows[self._idx].q, self.tcp_offset)
        self._row_len_cart = float(np.linalg.norm(p1 - p0))

    def _step(self, dt):
        if self._cart is not None:
            dv = self._cart_cmd - self._cart_v
            lim = self._cart_acc * dt
            n = float(np.linalg.norm(dv))
            self._cart_v = self._cart_cmd.copy() if n <= lim else self._cart_v + dv * (lim / n)
            p, R = self._cart
            # Rotation integrates on the manifold. Adding w*dt to a rotvec is
            # only valid for small rotvecs, and GRASP_ORIENTATION has magnitude
            # ~3.1 rad, where it is wrong enough to fake a servo stall.
            dR = Rotation.from_rotvec(self._cart_v[3:] * dt).as_matrix()
            p = p + self._cart_v[:3] * dt
            if self.floor_z is not None and p[2] < self.floor_z:
                p[2] = self.floor_z
                self._cart_v[2] = 0.0
            self._cart = (p, dR @ R)
            return
        if not self._rows:
            return
        if self._start_at is not None and time.monotonic() < self._start_at:
            return  # upload/start lag: stationary, nothing written yet
        if self._idx < 0:
            self._advance_row(keep_speed=False)
            if not self._rows:
                return
        row = self._rows[self._idx]
        delta = row.q - self._q_from
        L = float(np.max(np.abs(delta)))
        if L < 1e-9:
            self._advance_row(keep_speed=row.r > 0)
            return
        if self._braking:
            # Coast to a halt along the active row at the commanded decel. This
            # is the stopping distance a fixed vs velocity-sized brake buys or
            # costs, integrated rather than assumed.
            self._speed = max(0.0, self._speed - self._brake_decel * dt)
            self._u = min(L, self._u + self._speed * dt)
            self.q = self._q_from + delta * (self._u / L)
            if self._speed <= 1e-6 or self._u >= L - 1e-9:
                self._rows = []
                self._idx = -1
                self._speed = 0.0
                self._braking = False
            return
        remaining = L - self._u
        if row.r > 0.0:
            cap = row.v  # blended: no decel ramp
        else:
            cap = min(row.v, float(np.sqrt(max(2.0 * row.a * remaining, 0.0))))
        if self._speed < cap:
            self._speed = min(cap, self._speed + row.a * dt)
        elif self._speed > cap:
            self._speed = max(cap, self._speed - row.a * dt)
        self._u = min(L, self._u + self._speed * dt)
        self.q = self._q_from + delta * (self._u / L)
        if row.r > 0.0:
            cart_left = self._row_len_cart * (1.0 - self._u / L)
            if cart_left <= row.r or self._u >= L - 1e-9:
                self._advance_row(keep_speed=True)
        elif self._u >= L - 1e-9 and self._speed <= 1e-3:
            self.q = row.q.copy()
            self._advance_row(keep_speed=False)

    def _integrate(self):
        with self._lock:
            now = time.monotonic()
            left = min(now - self._t, 0.5)
            self._t = now
            while left > 1e-9:
                dt = min(self.SUBSTEP_S, left)
                self._step(dt)
                left -= dt

    # jaw integration (asynchronous: the stroke runs on the controller while
    # Python is off doing something else, exactly like rq_close_and_wait)

    def _jaw_speed_counts_s(self, speed_norm):
        frac = min(max(float(speed_norm) / 100.0, 0.0), 1.0)
        mm_s = self.JAW_SPEED_MIN_MM_S + frac * (
            self.JAW_SPEED_MAX_MM_S - self.JAW_SPEED_MIN_MM_S
        )
        return mm_s / self.JAW_STROKE_MM * (self.JAW_CLOSED - self.JAW_OPEN)

    def _jaw_integrate(self):
        now = time.monotonic()
        dt = now - self._jaw_t
        self._jaw_t = now
        if self._jaw_rate <= 0.0 or dt <= 0.0:
            return
        step = self._jaw_rate * dt
        if self._jaw_goal >= self.jaw:
            self.jaw = min(self._jaw_goal, self.jaw + step)
        else:
            self.jaw = max(self._jaw_goal, self.jaw - step)
        if abs(self.jaw - self._jaw_goal) < 1e-9:
            self._jaw_rate = 0.0
            # gObj latches only when a CLOSING stroke was stopped short of its
            # commanded target by the object.
            self._jaw_obj = self._jaw_goal < self._jaw_target - 1e-9

    def _jaw_command(self, target, speed_norm):
        self._jaw_integrate()
        self._jaw_target = float(target)
        closing = self._jaw_target > self.jaw
        stop = self.OBJECT_AT_COUNTS
        if closing and stop is not None and stop < self._jaw_target:
            self._jaw_goal = max(stop, self.jaw)
        else:
            self._jaw_goal = self._jaw_target
        self._jaw_rate = self._jaw_speed_counts_s(speed_norm)
        self._jaw_obj = False

    def _jaw_cancel(self):
        """A new program replaced the one running the stroke: jaws freeze."""
        self._jaw_integrate()
        if self._jaw_rate > 0.0:
            self._jaw_rate = 0.0
            self._jaw_obj = False
            self.jaw_cancels += 1

    # driver surface

    def get_joint_positions(self):
        self._integrate()
        return self.q.copy()

    def get_tcp_pose(self):
        self._integrate()
        p, R = self._tcp()
        return np.concatenate([p, Rotation.from_matrix(R).as_rotvec()])

    def get_tcp_offset(self):
        return list(self.tcp_offset)

    def get_tcp_force(self):
        """Quiet wrench with a little sensor noise.

        Exists so _wait_force_settled runs its real condition loop instead of
        falling through to the full-timeout branch. Seeded: the timing numbers
        this plant produces have to be reproducible.
        """
        return self._rng.normal(0.0, self.FORCE_NOISE_N, 6)

    def send_velocity(self, velocity, acceleration=0.5, duration=0.1):
        """Same signature and same wire form as UR10eDriver.send_velocity.

        Present so the grasp's force-guarded descent takes the speedl path it
        takes on the rig, rather than hasattr-failing into the servo fallback.
        """
        v = ", ".join("%.6f" % float(x) for x in velocity)
        return self._send_script(
            "speedl([%s], a=%.4f, t=%.4f)" % (v, float(acceleration), float(duration))
        )

    DEFAULT_JOINT_VEL = 1.05
    DEFAULT_JOINT_ACC = 1.4

    def move_to_joint_config_urscript(self, q, velocity=None, acceleration=None):
        """Verbatim copy of the driver's emitter: one movej, no blend, no wait."""
        vel = velocity or self.DEFAULT_JOINT_VEL
        acc = acceleration or self.DEFAULT_JOINT_ACC
        q_str = ", ".join(f"{float(x):.6f}" for x in q)
        self._send_script(f"movej([{q_str}], a={acc}, v={vel})")

    def move_to_joint_config(self, q, velocity=None, acceleration=None, asynchronous=False):
        self.move_to_joint_config_urscript(q, velocity, acceleration)

    def get_output_int_register(self, reg):
        self._integrate()
        return int(self.registers.get(int(reg), 0))

    def _note_motion_script(self, script):
        if "movej(" in script or "movel(" in script:
            self._motion_lease_until = max(self._motion_lease_until, time.monotonic() + 1.5)

    def clear_motion_lease(self):
        self._motion_lease_until = -1.0

    def _sync_from_cart(self):
        """Leave speedl mode: adopt the Cartesian state as joints."""
        p, R = self._cart[0], np.asarray(self._cart[1])
        q = ur_ik.solve_ik_rtde(p, R, q_seed=self.q, tcp_offset=self.tcp_offset)
        if q is not None:
            self.q = np.asarray(q, dtype=float)
        self._cart = None
        self._cart_v = np.zeros(6)
        self._cart_cmd = np.zeros(6)

    def _send_script(self, script: str) -> bool:
        self._integrate()
        with self._lock:
            self.sent.append(script)
            self._note_motion_script(script)
            # A new program REPLACES the running one. If that one was driving a
            # jaw stroke, the stroke stops where it is -- which is how a settle
            # that is too short becomes an observable grasp failure rather than
            # just a shorter sleep. rq_open/rq_close are the strokes themselves.
            if "rq_close" in script:
                self._jaw_command(self.JAW_CLOSED, self._pending_jaw_speed)
            elif "rq_open" in script:
                self._jaw_command(self.JAW_OPEN, self._pending_jaw_speed)
            else:
                self._jaw_cancel()
            if "speedl(" in script:
                if self._cart is None:
                    p, R = self._tcp()
                    self._cart = (np.asarray(p, float), np.asarray(R, float))
                    self._cart_v = np.zeros(6)
                self._rows = []
                self._idx = -1
                m = _SPEEDL_RE.search(script)
                self._cart_cmd = np.array([float(x) for x in m.group(1).split(",")], dtype=float)
                if "stopl" in script:
                    self._cart_cmd = np.zeros(6)
                return True
            m_stop = _STOP_RE.match(script.lstrip())
            if m_stop is not None:
                # A new send replaces the running program: brake and kill it.
                # The brake is NOT instantaneous. Modelling it as such was the
                # plant's second fidelity gap: it could show WHETHER a stop
                # happened but not how far the arm travelled afterwards, which
                # is the whole safety question once the commanded velocity goes
                # up (overshoot = v^2 / 2a). The arm now keeps following its
                # active row with the speed decaying at the commanded decel, so
                # stopping distance is measurable here.
                self._brake_decel = abs(float(m_stop.group(2) or 0.0)) or None
                self._braking = True
                self._cart_cmd = np.zeros(6)
                if self._brake_decel is None:
                    self._rows = []
                    self._idx = -1
                    self._speed = 0.0
                    self._cart_v = np.zeros(6)
                return True
            ml = _MOVEL_RE.search(script)
            if ml is not None:
                # Legacy fallback path (_movej_to_pose). Reduce it to the joint
                # target it implies so it MOVES here; a silently ignored movel
                # would look like a 15 s arrival timeout and read as a blending
                # result when it is a plant gap.
                pose = [float(x) for x in ml.group(1).split(",")]
                q = ur_ik.solve_ik_rtde(
                    pose[:3], pose[3:6], q_seed=self.q, tcp_offset=self.tcp_offset
                )
                if q is None:
                    return True
                script = "movej([%s], a=1.2, v=1.05)" % ", ".join("%.6f" % v for v in q)
            rows = _MOVEJ_RE.findall(script)
            if not rows:
                # Gripper / register-only script: cancels any running motion.
                if self._rows:
                    self.gripper_cancels += 1
                self._rows = []
                self._idx = -1
                self._speed = 0.0
                for reg, val in _REG_RE.findall(script):
                    self.registers[int(reg)] = int(val)
                t0 = time.monotonic()
                time.sleep(self.GRIPPER_SCRIPT_S)
                self.script_time += time.monotonic() - t0
                self._t = time.monotonic()
                return True
            if self._cart is not None:
                self._sync_from_cart()
            # Walk the program in order; a register write is attached to the
            # movej that FOLLOWS it (that is how the executor emits progress),
            # and a trailing write with no movej after it is the completion
            # token, applied when the program runs off the end.
            pending = None
            parsed: list[_Row] = []
            self._completion_reg = None
            for tok in _PROGRAM_RE.finditer(script):
                if tok.group("reg") is not None:
                    pending = (int(tok.group("reg")), int(tok.group("val")))
                    continue
                q = [float(x) for x in tok.group("q").split(",")]
                parsed.append(
                    _Row(q, tok.group("v"), tok.group("a"), tok.group("r") or 0.0, pending)
                )
                pending = None
            self._completion_reg = pending
            self._rows = parsed
            self._idx = -1
            self._q_from = self.q.copy()
            self._speed = 0.0
            self._braking = False
            self._brake_decel = None
            if parsed:
                self._last_joint_vel = max(r.v for r in parsed)
            self._start_at = time.monotonic() + self.START_LAG_S
            return True

    # gripper surface. Mirrors UR10eDriver: the send starts an asynchronous
    # controller-side stroke, then the driver sleeps GRIPPER_DRIVER_SLEEP_S and
    # returns while the jaws may still be moving.

    _pending_jaw_speed = 100.0

    def open_gripper(self, speed=None, force=None):
        self._pending_jaw_speed = 100.0 if speed is None else float(speed)
        self._send_script("def rq_open():\n  rq_open_and_wait()\nend\nrq_open()\n")
        self._driver_gripper_sleep()

    def close_gripper(self, speed=None, force=None):
        self._pending_jaw_speed = 50.0 if speed is None else float(speed)
        self._send_script("def rq_close():\n  rq_close_and_wait()\nend\nrq_close()\n")
        self._driver_gripper_sleep()

    def _driver_gripper_sleep(self):
        t0 = time.monotonic()
        time.sleep(self.GRIPPER_DRIVER_SLEEP_S)
        self.script_time += time.monotonic() - t0
        with self._lock:
            self._t = time.monotonic()

    def _publish_gripper_state(self, force=False):
        """Refresh the jaw registers -- by uploading a program, as on the rig.

        Modelled as a program precisely because it is one: it cancels whatever
        was running, so polling a jaw stroke to find out when it finished
        destroys the stroke. That is the constraint the settle constants exist
        to respect.
        """
        self._send_script(
            "def rq_pub():\n  write_output_integer_register(16, 1)\nend\nrq_pub()\n"
        )
        return True

    def get_gripper_position(self, publish=True):
        if publish:
            self._publish_gripper_state()
        self._jaw_integrate()
        return int(round(self.jaw))

    def is_object_detected(self, publish=True):
        if publish:
            self._publish_gripper_state()
        self._jaw_integrate()
        return bool(self._jaw_obj)

    # Overshoot budget the velocity-sized brake aims for (rad on the leading
    # joint). Mirrors UR10eDriver.BRAKE_OVERSHOOT_BUDGET_RAD's role: a = v^2/2s.
    BRAKE_OVERSHOOT_BUDGET_RAD = 0.10
    BRAKE_DECEL_MIN = 1.0
    BRAKE_DECEL_MAX = 8.0

    def brake_decel(self, kind=None):
        v = float(self._last_joint_vel)
        a = v * v / (2.0 * self.BRAKE_OVERSHOOT_BUDGET_RAD)
        return float(min(max(a, self.BRAKE_DECEL_MIN), self.BRAKE_DECEL_MAX))

    def brake(self, decel=None, kind=None):
        """Velocity-sized URScript brake, as UR10eDriver.brake does.

        A FIXED decel is what the plant is here to argue against: overshoot goes
        as v^2, so the same stopj that was adequate at 1.05 rad/s is not at 1.78.
        """
        a = self.brake_decel(kind) if decel is None else float(decel)
        return self._send_script("stopj(%.3f)" % a)

    def stop(self):
        """DELIBERATELY the broken rig behaviour: rtde_c.stopJ, i.e. a no-op.

        On this rig ``UR10eDriver.stop`` is ``rtde_c.stopJ(2.0)``, which does
        nothing once _send_script has killed the control script -- which is
        always. Modelling it as a real brake is how an offline brake test passes
        spuriously, so it does not brake here either. Anything that needs to
        stop this plant must call brake()/_send_script, exactly as on the rig.
        """
        return None

    def stop_velocity(self):
        self._send_script("stopl(1.0)")

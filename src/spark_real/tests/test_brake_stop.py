"""THE STOP BUTTON MUST BRAKE THE ARM. Regressions for every stop path.

What was broken, measured at 52fc6a7: ``ScoreExecutor._stop_robot`` walked
("stop_motion", "stop", "servo_stop") and returned on the FIRST hit, which on
the UR10e was ``UR10eDriver.stop`` -> ``rtde_c.stopJ(2.0)``. stopJ writes an
RTDE input register that ur_rtde's control script polls, and ``_send_script``
deliberately stops that control script before every move -- so nobody read the
register. URScript ``movej`` is fire-and-forget, so an abort stopped this
process from issuing commands AND LEFT THE ARM RUNNING TO ITS TARGET, while the
no-op satisfied the loop and made the failure invisible. Same for ``/api/stop``
(its real ``stopl`` was followed by the same masking loop), for the primitive
budget watchdog, and for server Ctrl-C, which emitted nothing at all.

Every test here drives the REAL driver -- its socket, its send path, its
escalation -- against the accel-limited plant, so a passing test means the
braking bytes left the process and the modelled arm decelerated. The mutation
tests restore HEAD's code and assert the arm runs on: they are what makes the
rest of the file mean something.
"""

import threading
import time
import types

import numpy as np
import pytest

from spark_real.control.executor_core import is_braking_script
from spark_real.control.score_executor import ScoreExecutor
from spark_real.control.ur10e_driver import UR10eDriver
from spark_real.tests.brake_plant import BrakeRig, RecordingRtdeC, moving_rig, target_away

BUDGET_RAD = UR10eDriver.BRAKE_OVERSHOOT_BUDGET_RAD


@pytest.fixture(autouse=True)
def _restore_route_state():
    """The route handlers read a module-global pipeline; put it back."""
    from spark_real.routes import state as route_state

    before = route_state.pipeline
    yield
    route_state.pipeline = before


# ---------------------------------------------------------------- 1. the brake


def test_brake_reaches_the_wire_and_the_arm_decelerates():
    rig, q_target = moving_rig()
    q_at_stop = rig.q()[0]
    assert rig.plant.joint_speed() > 0.5, "arm was not moving; test is vacuous"

    assert rig.driver.brake() is True, "brake did not reach the wire"

    assert rig.brake_frames(), f"no stopj/stopl on the wire: {rig.frames}"
    assert rig.brake_frames()[0].startswith("stopj("), rig.brake_frames()
    rig.spin(0.8)
    assert rig.plant.joint_speed() == 0.0, "arm still moving after the brake"
    overshoot = abs(rig.q()[0] - q_at_stop)
    assert overshoot < BUDGET_RAD * 1.4, f"overshoot {overshoot:.3f} rad"
    # And it did NOT reach the target it was commanded to.
    assert abs(rig.q()[0] - q_target[0]) > 1.0, "arm ran on to its target"


def test_brake_decel_is_sized_from_the_commanded_velocity():
    """overshoot = v^2/2a, so a constant decel means a v^2 stopping distance.

    The hardcoded stopj(2.0) at 1.8 rad/s would coast 0.81 rad (~0.8 m of TCP
    at reach) after the operator presses stop. The decel has to track v.
    """
    seen = {}
    for v in (0.5, 1.05, 1.8):
        rig, _ = moving_rig(velocity=v, acceleration=1.4, delta=4.5)
        q_at_stop = rig.q()[0]
        speed = rig.plant.joint_speed()
        rig.driver.brake()
        decel = float(rig.brake_frames()[0].split("(")[1].rstrip(")\n"))
        rig.spin(1.2)
        seen[v] = (decel, abs(rig.q()[0] - q_at_stop), speed)

    assert seen[1.8][0] > seen[1.05][0] > seen[0.5][0], seen
    assert seen[1.8][0] <= UR10eDriver.BRAKE_DECEL_MAX, "decel above the C153 limit"
    assert seen[0.5][0] >= UR10eDriver.BRAKE_DECEL_MIN, "decel below a real stop"
    for v, (decel, overshoot, speed) in seen.items():
        predicted = speed * speed / (2.0 * decel)
        assert overshoot <= predicted + 0.02, (v, overshoot, predicted)
        # Every stop stays inside the budget the decel was chosen for, except
        # where the decel had to be clamped at BRAKE_DECEL_MAX.
        cap = BUDGET_RAD if decel < UR10eDriver.BRAKE_DECEL_MAX else 0.25
        assert overshoot <= cap + 0.02, (v, overshoot, cap)


def test_brake_after_a_movej_uses_stopj_and_after_a_speedl_uses_stopl():
    """stopl constrains TCP decel in m/s^2; that is the wrong constraint on a
    joint-space movej, and near a singularity it can demand large joint rates.
    """
    rig, _ = moving_rig()
    rig.driver.brake()
    assert rig.brake_frames()[-1].startswith("stopj(")

    rig2 = BrakeRig()
    rig2.driver.send_velocity([0.12, 0.0, 0.0, 0.0, 0.0, 0.0], 1.0, 0.2)
    rig2.spin(0.2)
    rig2.driver.brake()
    assert rig2.brake_frames()[-1].startswith("stopl("), rig2.frames


def test_stop_no_longer_calls_the_dead_rtde_register():
    """UR10eDriver.stop() must not resolve to rtde_c.stopJ: it cannot brake and
    its success hides the failure of the stop above it in every caller."""
    rtde_c = RecordingRtdeC()
    rig = BrakeRig(rtde_c=rtde_c)
    rig.driver.move_to_joint_config_urscript(target_away(rig.q()), velocity=1.05)
    rig.spin_until_moving()

    assert rig.driver.stop() is True
    assert rtde_c.stopJ_calls == [], "stop() still writes the dead RTDE register"
    assert rig.brake_frames(), "stop() emitted no URScript brake"
    rig.spin(0.8)
    assert rig.plant.joint_speed() == 0.0


def test_brake_during_the_upload_lag_cancels_the_program():
    """A stop pressed inside the ~1 s program start lag must leave the arm
    stationary, not let the queued move start afterwards."""
    rig = BrakeRig(start_lag=0.6)
    q0 = rig.q().copy()
    rig.driver.move_to_joint_config_urscript(target_away(rig.q()), velocity=1.05)
    rig.spin(0.1)  # still inside the lag
    assert rig.driver.brake() is True
    rig.spin(1.2)
    assert np.allclose(rig.q(), q0, atol=1e-9), "queued move ran after the stop"


# ------------------------------------------------- 2. the executor abort paths


def _executor(rig):
    ex = ScoreExecutor(rig.driver, detection_map={}, velocity=0.25)
    ex._running = True
    return ex


def test_executor_abort_brakes_the_arm_and_reports_it():
    rig, q_target = moving_rig()
    ex = _executor(rig)
    q_at_stop = rig.q()[0]

    assert ex.abort() is True, "abort() did not brake"

    assert rig.brake_frames(), "abort() set a flag and nothing reached the wire"
    rig.spin(0.8)
    assert rig.plant.joint_speed() == 0.0
    assert abs(rig.q()[0] - q_at_stop) < BUDGET_RAD * 1.4
    assert abs(rig.q()[0] - q_target[0]) > 1.0


def test_check_abort_brakes_before_it_raises():
    """_check_abort is the abort hook every primitive polls, and the
    primitive-timeout watchdog reaches the arm through this same path."""
    from spark_real.control.executor_types import AbortRequested

    rig, _ = moving_rig()
    ex = _executor(rig)
    ex._abort = True
    with pytest.raises(AbortRequested):
        ex._check_abort()
    assert rig.brake_frames(), "_check_abort raised without braking"
    rig.spin(0.8)
    assert rig.plant.joint_speed() == 0.0


# HEAD's implementation, verbatim (executor_core.py:598-609 at 52fc6a7).
def _legacy_stop_robot(self):
    for name in ("stop_motion", "stop", "servo_stop"):
        fn = getattr(self.robot, name, None)
        if callable(fn):
            try:
                fn()
                return
            except Exception:
                continue


def test_MUTATION_head_stop_lets_the_arm_run_on():
    """Mutation witness: put HEAD's stop back and the arm does NOT brake.

    Both halves of the fix are reverted -- the getattr loop in the executor and
    the driver's URScript brake (back to rtde_c.stopJ against a dead control
    script). If this test ever fails, the tests above have stopped proving
    anything: they would pass on the broken code too.
    """
    rtde_c = RecordingRtdeC()
    rig, q_target = moving_rig(delta=1.4)
    rig.driver._rtde_c = rtde_c
    rig.driver.stop = types.MethodType(lambda self: self._rtde_c.stopJ(2.0), rig.driver)
    rig.driver.emergency_stop = None  # did not exist at HEAD
    ex = _executor(rig)
    ex._stop_robot = types.MethodType(_legacy_stop_robot, ex)

    ex.abort()

    assert rtde_c.stopJ_calls == [2.0], "the legacy path was not exercised"
    assert rig.brake_frames() == [], "a brake escaped the mutation"
    rig.spin(1.6)
    assert abs(rig.q()[0] - q_target[0]) < 0.01, (
        "with HEAD's stop the arm should run all the way to its target; "
        f"joint0={rig.q()[0]:.4f} target={q_target[0]:.4f}"
    )


def test_MUTATION_flag_only_abort_lets_the_arm_run_on():
    """The other shape of the bug: abort() that only sets flags."""
    rig, q_target = moving_rig(delta=1.4)
    ex = _executor(rig)
    ex._stop_robot = types.MethodType(lambda self: False, ex)
    ex.abort()
    assert rig.brake_frames() == []
    rig.spin(1.6)
    assert abs(rig.q()[0] - q_target[0]) < 0.01


# ------------------------------------------- 3. the next motion must still work


def test_a_move_after_a_stop_works_without_a_reconnect():
    rig, _ = moving_rig()
    rig.driver.brake()
    rig.spin(0.6)
    reopens_before = rig.reopens

    q_next = target_away(rig.q(), delta=-0.6)
    rig.driver.move_to_joint_config_urscript(q_next, velocity=1.05, acceleration=1.4)
    rig.spin_until_moving()
    rig.spin(1.6)

    assert np.allclose(rig.q(), q_next, atol=0.02), (rig.q(), q_next)
    assert rig.reopens == reopens_before, "the stop wedged the socket"
    assert rig.driver._connected is True


def test_the_abort_flag_does_not_outlive_the_stop_on_the_send_path():
    """After a stop the executor must not keep rejecting motion.

    The abort-aware _send_script wrapper raises while (_running and _abort);
    /api/stop clears _running and execute_score clears _abort, and a move has
    to pass again once either is cleared.
    """
    from spark_real.control.executor_types import AbortRequested

    rig, _ = moving_rig()
    ex = _executor(rig)
    ex.abort()
    rig.spin(0.5)

    with pytest.raises(AbortRequested):
        rig.driver._send_script("movej([0,0,0,0,0,0], a=1.4, v=1.05)")

    ex._running = False  # what /api/stop does
    assert rig.driver._send_script("movej([0,0,0,0,0,0], a=1.4, v=1.05)") is True
    ex._running, ex._abort = True, False  # what execute_score does
    assert rig.driver._send_script("movej([0,0,0,0,0,0], a=1.4, v=1.05)") is True


def test_the_brake_itself_is_never_blocked_by_the_abort_flag():
    rig, _ = moving_rig()
    ex = _executor(rig)
    ex._abort = True  # wrapper now rejects ordinary motion
    assert rig.driver.brake() is True
    assert rig.brake_frames()


# ------------------------------------------------------------ 4. the servo path


def test_servo_abort_brakes_instead_of_raising():
    """The servo's own stop is ``speedl([0...])`` + ``stopl``, which the old
    startswith("stopl","stopj") whitelist REJECTED -- so an aborted servo
    emitted nothing and the arm kept its last commanded twist."""
    rig = BrakeRig()
    ex = _executor(rig)
    pose = np.asarray(rig.driver.get_tcp_pose(), dtype=float)
    target = pose.copy()
    target[0] += 0.25

    done = {}

    def _run():
        done["ok"] = ex._servo.move_to_pose(target.tolist(), timeout=4.0)

    th = threading.Thread(target=_run, daemon=True)
    th.start()
    time.sleep(0.4)
    assert np.linalg.norm(rig.plant.tcp_speed_vec()[:3]) > 0.01, "servo never moved"

    ex.abort()
    th.join(timeout=3.0)
    assert not th.is_alive(), "servo abort deadlocked"
    assert done["ok"] is False
    assert ex._servo.last_exit == "abort"
    assert rig.brake_frames(), "the servo abort emitted no brake"
    rig.spin(1.5)
    assert np.linalg.norm(rig.plant.tcp_speed_vec()[:3]) <= 1e-6, "servo still moving"


def test_is_braking_script_admits_only_scripts_that_can_slow_the_arm():
    assert is_braking_script("stopj(2.0)")
    assert is_braking_script(" stopl(1.0)\n")
    assert is_braking_script("speedl([0,0,0,0,0,0], 1.0, 0.1)\nstopl(1.0)")
    # A non-zero speedl before the stop would ACCELERATE first.
    assert not is_braking_script("speedl([0.2,0,0,0,0,0], 1.0, 0.1)\nstopl(1.0)")
    assert not is_braking_script("movej([0,0,0,0,0,0], a=1.4, v=1.05)")
    assert not is_braking_script("movej([0,0,0,0,0,0], a=1.4, v=1.05)\nstopj(2.0)")
    assert not is_braking_script("speedj([0.1,0,0,0,0,0], 1.0, 0.1)\nstopj(2.0)")
    assert not is_braking_script("def rq_close():\n  rq_close_and_wait()\nend\n")
    assert not is_braking_script("")


# -------------------------------------------------- 5. the blended-path abort


def test_blended_path_abort_still_brakes_mid_path():
    """Do not regress the one path that already worked: MotionMixin._halt_arm
    emits stopj on the same socket, and the wrapper must still pass it."""
    from spark_real.control.waypoints import JointRow, build_ur_blend_program, resolve_blend_radii

    rig = BrakeRig(start_lag=0.05)
    ex = _executor(rig)
    q0 = rig.q()
    rows = []
    for i, d in enumerate((0.4, 0.8, 1.2)):
        q = np.asarray(q0, dtype=float).copy()
        q[0] += d
        rows.append(
            JointRow(
                q=q,
                position=np.array([d, 0.0, 0.0]),
                velocity=1.05,
                acceleration=1.4,
                label="r%d" % i,
            )
        )
    kept, _ = resolve_blend_radii([0.0, 0, 0], rows, radius_m=0.05)
    rig.driver._send_script(build_ur_blend_program(kept, epoch=1))
    rig.spin_until_moving()
    rig.spin(0.4)
    q_at_stop = rig.q()[0]

    ex._abort = True
    assert ex._halt_arm() is True
    assert rig.brake_frames()
    rig.spin(1.0)
    assert rig.plant.joint_speed() == 0.0
    assert abs(rig.q()[0] - q_at_stop) < 0.35, "blended path ran on"
    assert abs(rig.q()[0] - (q0[0] + 1.2)) > 0.3, "reached the last waypoint anyway"


def test_a_blended_program_brake_is_sized_from_the_row_velocity():
    rig = BrakeRig(start_lag=0.05)
    rig.driver._send_script(
        "movej([0.1,0,0,0,0,0], a=1.4, v=1.60, r=0.05)\n"
        "movej([0.9,0,0,0,0,0], a=1.4, v=1.60, r=0.0)\n"
    )
    assert rig.driver._last_joint_vel == pytest.approx(1.60)
    assert rig.driver.brake_decel("joint") == pytest.approx(
        min(1.60**2 / (2 * BUDGET_RAD), UR10eDriver.BRAKE_DECEL_MAX)
    )


# ---------------------------------------------------------- 6. the escalation


class _FakeDashboard:
    def __init__(self, block=None):
        self.calls = []
        self.block = block

    def stop(self):
        if self.block is not None:
            self.block.wait()
        self.calls.append("stop")

    def disconnect(self):
        pass


def test_emergency_stop_escalates_to_the_dashboard_when_the_socket_is_gone():
    rig, _ = moving_rig()
    rig.wire.fail = True
    rig.reopen_ok = False
    dash = _FakeDashboard()
    rig.driver._dashboard = lambda: dash

    result = rig.driver.emergency_stop()

    assert result["braked"] is True, result
    assert result["steps"][-1] == "dashboard", result
    assert dash.calls == ["stop"]


def test_the_dashboard_escalation_is_bounded():
    """An escalation that hangs is not an escalation: DashboardClient.connect()
    blocks on an unreachable or Local-mode controller."""
    rig = BrakeRig()
    block = threading.Event()
    rig.driver._dashboard = lambda: _FakeDashboard(block=block)
    t0 = time.monotonic()
    try:
        assert rig.driver._send_dashboard_command("stop", timeout_s=0.2) is False
        assert time.monotonic() - t0 < 1.0
    finally:
        block.set()


def test_emergency_stop_reports_failure_when_nothing_reaches_the_arm():
    """The one thing that must never be overstated."""
    rig, _ = moving_rig()
    rig.wire.fail = True
    rig.reopen_ok = False
    rig.driver._dashboard = lambda: None

    result = rig.driver.emergency_stop()
    assert result["braked"] is False
    assert result["steps"] == [
        "urscript:failed",
        "urscript-reconnect:failed",
        "dashboard:failed",
    ]
    ex = _executor(rig)
    assert ex.abort() is False, "abort claimed a brake that never happened"


def test_a_dead_socket_does_not_stop_the_brake():
    """_send_script's own retry reopens the socket, so one dead socket costs a
    reconnect, not the brake."""
    rig, _ = moving_rig()
    rig.wire.fail = True  # the live socket is dead; a fresh one works

    result = rig.driver.emergency_stop()

    assert result == {"braked": True, "steps": ["urscript"]}, result
    assert rig.reopens == 1
    rig.spin(0.8)
    assert rig.plant.joint_speed() == 0.0


def test_emergency_stop_forces_a_fresh_socket_when_the_drivers_retry_fails():
    """Second line of defence: the driver's own reconnect can lose the race
    (the reopen fails once), so the escalation drops the socket and retries."""
    rig, _ = moving_rig()
    rig.wire.fail = True
    rig.reopen_fail_first = 1  # starve _send_script's internal retry

    result = rig.driver.emergency_stop()

    assert result == {
        "braked": True,
        "steps": ["urscript:failed", "urscript-reconnect"],
    }, result
    rig.spin(0.8)
    assert rig.plant.joint_speed() == 0.0


def test_disconnect_brakes_before_it_closes_the_socket():
    """server Ctrl-C -> pipeline.shutdown() -> robot.disconnect(). Closing the
    socket does not stop a program already running on the controller, which is
    the operator's "Ctrl-C made the arm just go high"."""
    rig, q_target = moving_rig()
    q_at_stop = rig.q()[0]

    rig.driver.disconnect()

    assert rig.brake_frames(), "disconnect emitted no brake"
    assert rig.wire.closed is True
    rig.spin(0.8)
    assert rig.plant.joint_speed() == 0.0
    assert abs(rig.q()[0] - q_at_stop) < BUDGET_RAD * 1.4
    assert abs(rig.q()[0] - q_target[0]) > 1.0


def test_MUTATION_the_old_whitelist_swallows_the_servo_brake():
    """Mutation witness for the servo path: with HEAD's prefix-only whitelist
    the servo's brake raises instead of reaching the wire."""
    from spark_real.control.executor_types import AbortRequested

    rig = BrakeRig()
    ex = _executor(rig)
    rig.driver.send_velocity([0.12, 0.0, 0.0, 0.0, 0.0, 0.0], 1.0, 0.2)
    rig.spin(0.2)
    ex._abort = True

    orig = rig.driver._send_script

    def legacy_send(script):
        if ex._running and ex._abort and not script.lstrip().startswith(("stopl", "stopj")):
            raise AbortRequested()
        return orig(script)

    rig.driver._send_script = legacy_send
    before = len(rig.frames)
    with pytest.raises(AbortRequested):
        ex._servo._stop()
    assert len(rig.frames) == before, "something reached the wire under HEAD"

    # With the fix in place the same call brakes.
    rig.driver._send_script = orig
    ex._servo._stop()
    assert is_braking_script(rig.frames[-1]), rig.frames[-1]
    rig.spin(1.5)
    assert np.linalg.norm(rig.plant.tcp_speed_vec()[:3]) <= 1e-6


# ------------------------------------------------------- 7. teardown / Ctrl-C


def test_servo_stop_brakes_even_though_the_rtde_call_is_dead():
    rtde_c = RecordingRtdeC()
    rig = BrakeRig(rtde_c=rtde_c)
    rig.driver.move_to_joint_config_urscript(target_away(rig.q()), velocity=1.05)
    rig.spin_until_moving()
    rig.driver.servo_stop()
    assert rig.brake_frames()
    rig.spin(0.8)
    assert rig.plant.joint_speed() == 0.0


# ---------------------------------------------------------------- 8. /api/stop


def _stop_route_client(rig, executor=None):
    from fastapi import FastAPI
    from fastapi.testclient import TestClient

    from spark_real.routes import state as route_state
    from spark_real.routes.control import router

    class _FakePipeline:
        def __init__(self):
            self._robot = rig.driver
            self._executor = executor
            self._activity_lock = threading.Lock()
            self._activity_stack = []

    app = FastAPI()
    app.include_router(router)
    route_state.pipeline = _FakePipeline()
    return TestClient(app)


def test_api_stop_brakes_the_arm():
    rig, q_target = moving_rig()
    ex = _executor(rig)
    client = _stop_route_client(rig, ex)
    q_at_stop = rig.q()[0]

    body = client.post("/api/stop").json()

    assert body["braked"] is True, body
    assert body["errors"] == [], body
    assert "executor.abort" in body["steps"], body
    assert rig.brake_frames(), "/api/stop set flags and nothing reached the wire"
    rig.spin(0.8)
    assert rig.plant.joint_speed() == 0.0, "arm still moving after /api/stop"
    assert abs(rig.q()[0] - q_at_stop) < BUDGET_RAD * 1.4
    assert abs(rig.q()[0] - q_target[0]) > 1.0, "arm ran on to its target"
    assert ex._abort is True and ex._running is False


def test_api_stop_brakes_with_no_score_running():
    """The operator can hit stop during a manual jog, with no executor at all."""
    rig, _ = moving_rig()
    client = _stop_route_client(rig, executor=None)

    body = client.post("/api/stop").json()

    assert body["braked"] is True, body
    assert body["steps"] == ["urscript"], body
    rig.spin(0.8)
    assert rig.plant.joint_speed() == 0.0


def test_api_stop_admits_when_it_did_not_brake():
    rig, _ = moving_rig()
    rig.wire.fail = True
    rig.reopen_ok = False
    rig.driver._dashboard = lambda: None
    client = _stop_route_client(rig, _executor(rig))

    body = client.post("/api/stop").json()

    assert body["braked"] is False, body
    assert "dashboard:failed" in body["steps"], body


def test_stop_velocity_brakes_a_movej_too():
    """/api/abort and /api/recover reach for stop_velocity generically, so it
    has to brake a joint move as well -- in joint space, not Cartesian."""
    rig, q_target = moving_rig()
    assert rig.driver.stop_velocity() is True
    assert rig.brake_frames()[-1].startswith("stopj("), rig.brake_frames()
    rig.spin(0.8)
    assert rig.plant.joint_speed() == 0.0
    assert abs(rig.q()[0] - q_target[0]) > 1.0

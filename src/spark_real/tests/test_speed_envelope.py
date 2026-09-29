"""The speed work: envelope, settles, program count -- all offline.

Four things are being defended here, in order of how bad it would be to lose
them:

  1. No commanded joint velocity may exceed that joint's UR10e rating, on ANY
     move shape. The leading-axis cap is the whole reason it is safe to raise
     `v` above the old 1.05, so it gets a property test, not an example.
  2. The premature-arrival guard survives the faster arrival poll. Polling 4x
     faster with the same displacement threshold would let sensor noise latch
     `ever_moved`, which is the exact input the stillness exit needs.
  3. A shortened gripper settle is FALSIFIABLE. The plant runs the jaw stroke
     asynchronously and a new program cancels it, so a settle that is too short
     leaves the jaws part-closed and fails the grasp. The test that matters is
     the one showing the plant would have caught it.
  4. Work that was removed was redundant work, not a check: the conditional
     re-squeeze and the verify-first transport check must still re-squeeze and
     must still fail on a real drop.
"""

from __future__ import annotations

import time

import numpy as np
import pytest

from spark_real.control import executor_ik as ik
from spark_real.control.executor_motion import MotionMixin, _StillnessTracker

# ---------------------------------------------------------------------------
# 1. envelope
# ---------------------------------------------------------------------------


def test_no_joint_is_ever_commanded_past_its_rating():
    """Property test over random move shapes, including near-degenerate ones."""
    rng = np.random.default_rng(0)
    rated = np.asarray(ik.UR10E_JOINT_VEL_RATED_RAD_S)
    worst_headroom = float("inf")
    for _ in range(3000):
        dq = rng.random(6) ** rng.integers(1, 6)  # skewed: sometimes one axis dominates
        if rng.random() < 0.2:
            dq[rng.integers(0, 6)] = 0.0
        if dq.max() <= 0:
            continue
        v, a = ik.resolve_joint_limits(0.25, dq=dq)
        per_joint = v * np.abs(dq) / np.abs(dq).max()
        assert np.all(per_joint <= rated + 1e-9), (
            f"dq={dq} v={v} put joint {int(np.argmax(per_joint - rated))} "
            f"at {per_joint.max():.3f} rad/s over rating"
        )
        worst_headroom = min(worst_headroom, float(np.min(rated / np.maximum(per_joint, 1e-9))))
        assert a <= ik.JOINT_ACC_CAP_RAD_S2 + 1e-9
        assert v >= ik.JOINT_VEL_FLOOR_RAD_S - 1e-9
    # The stated safety margin is real: nothing ever got closer than 1/0.85.
    assert worst_headroom >= 1.0 / ik.JOINT_VEL_MARGIN - 1e-6, worst_headroom


def test_the_raise_is_actually_a_raise():
    """A base-led move must now be materially faster than movej's own default."""
    dq = np.array([1.0, 0.2, 0.1, 0.05, 0.0, 0.0])
    v, a = ik.resolve_joint_limits(0.25, dq=dq)
    assert v > 1.05 * 1.5, v  # >= 1.6 rad/s vs the old hard cap of 1.05
    assert a > 1.4 * 2.0, a  # acceleration is the dominant lever on short moves
    assert v <= ik.JOINT_VEL_MARGIN * 2.094 + 1e-9


def test_a_wrist_led_move_gets_the_wrist_rating_and_a_mixed_one_does_not():
    wrist_led = ik.resolve_joint_limits(0.25, dq=np.array([0.02, 0.02, 0.05, 0.1, 0.2, 1.0]))[0]
    base_led = ik.resolve_joint_limits(0.25, dq=np.array([1.0, 0.0, 0, 0, 0, 0]))[0]
    # Wrist-dominated: 3.142 rating is reachable, so this must beat the base cap.
    assert wrist_led > base_led * 1.4, (wrist_led, base_led)
    # Base almost as far as the wrist: the base rating binds again.
    mixed = ik.resolve_joint_limits(0.25, dq=np.array([0.99, 0, 0, 0, 0, 1.0]))[0]
    assert mixed < base_led * 1.05, (mixed, base_led)


def test_unknown_dq_falls_back_to_the_slowest_joint():
    """_movej_to_pose has no solved joint target; it must assume the worst."""
    assert ik.resolve_joint_limits(0.25)[0] == pytest.approx(
        ik.JOINT_VEL_MARGIN * min(ik.UR10E_JOINT_VEL_RATED_RAD_S), rel=1e-6
    )


def test_legacy_env_restores_the_exact_old_clamps(monkeypatch):
    monkeypatch.setenv("SPARK_LEGACY_JOINT_LIMITS", "1")
    assert ik.resolve_joint_limits(0.25, dq=np.array([1.0, 0, 0, 0, 0, 0])) == (1.05, 1.4)
    assert ik.resolve_joint_limits(1.0, dq=np.array([1.0, 0, 0, 0, 0, 0]))[0] == 1.05
    assert ik.resolve_linear_limits(0.25)[0] == 0.25


def test_speed_fraction_ramps_the_whole_envelope(monkeypatch):
    dq = np.array([1.0, 0, 0, 0, 0, 0])
    full_v, full_a = ik.resolve_joint_limits(0.25, dq=dq)
    monkeypatch.setenv("SPARK_JOINT_SPEED_FRACTION", "0.5")
    half_v, half_a = ik.resolve_joint_limits(0.25, dq=dq)
    assert half_v == pytest.approx(full_v * 0.5, rel=1e-6)
    assert half_a == pytest.approx(full_a * 0.5, rel=1e-6)
    monkeypatch.setenv("SPARK_JOINT_SPEED_FRACTION", "9.0")  # clamped, never above 1
    assert ik.resolve_joint_limits(0.25, dq=dq)[0] == pytest.approx(full_v, rel=1e-6)
    monkeypatch.setenv("SPARK_JOINT_SPEED_FRACTION", "not-a-number")
    assert ik.resolve_joint_limits(0.25, dq=dq)[0] == pytest.approx(full_v, rel=1e-6)


def test_a_slow_caller_still_gets_a_slow_move():
    """The grasp descent / lift pass velocity*0.15 and must NOT be sped up."""
    dq = np.array([0.3, 0, 0, 0, 0, 0])
    slow = ik.resolve_joint_limits(0.25 * 0.15, dq=dq)[0]
    fast = ik.resolve_joint_limits(0.25, dq=dq)[0]
    assert slow < fast / 3.0, (slow, fast)


# ---------------------------------------------------------------------------
# 2. the faster arrival poll must not weaken the premature-arrival guard
# ---------------------------------------------------------------------------


def test_sub_sample_noise_cannot_latch_ever_moved():
    """4 polls inside one 100 ms window must be ONE displacement decision.

    Feeding 25 ms of noise-sized jitter to the tracker must not report motion:
    if it did, the stillness exit could fire without the arm having moved, which
    is the shipped-once premature-arrival regression.
    """
    rng = np.random.default_rng(1)
    t = 0.0
    tr = _StillnessTracker(np.zeros(3), t, sample_s=0.1)
    for _ in range(80):  # 2 s of polls at 25 ms
        t += 0.025
        tr.update(rng.normal(0.0, 5e-5, 3), t)  # 50 micron sensor noise
    assert tr.ever_moved is False, "noise latched ever_moved"
    assert tr.stopped() is True


def test_real_motion_is_still_detected_at_the_sample_cadence():
    t = 0.0
    tr = _StillnessTracker(np.zeros(3), t, sample_s=0.1)
    p = np.zeros(3)
    for _ in range(20):
        t += 0.025
        p = p + np.array([0.005, 0.0, 0.0])  # 0.2 m/s
        tr.update(p, t)
    assert tr.ever_moved is True
    assert tr.stopped() is False


def test_stillness_needs_whole_windows_not_poll_counts():
    """MOTION_STILL_SAMPLES must mean 3 sample WINDOWS, not 3 polls."""
    t = 0.0
    tr = _StillnessTracker(np.zeros(3), t, sample_s=0.1, still_samples=3)
    for _ in range(8):  # 200 ms of polls = at most 2 windows
        t += 0.025
        tr.update(np.zeros(3), t)
    assert tr.stopped() is False, "3 fast polls were counted as 3 still samples"
    for _ in range(6):
        t += 0.025
        tr.update(np.zeros(3), t)
    assert tr.stopped() is True


def test_the_poll_is_faster_than_the_sample_window():
    assert MotionMixin.MOTION_POLL_S < MotionMixin.MOTION_SAMPLE_S
    assert MotionMixin.MOTION_SAMPLE_S == 0.1, "the displacement window is calibrated"


def test_the_arrival_budget_floor_clears_the_program_upload_lag():
    """A raised envelope shrinks `ideal`, so the FLOOR is what fast moves get."""
    m = MotionMixin()
    assert m._min_motion_budget_s() > m.URSCRIPT_START_LAG_S + 1.0
    m.URSCRIPT_START_LAG_S = 3.0
    assert m._min_motion_budget_s() > 3.0, "floor must track the lag, not be a constant"


# ---------------------------------------------------------------------------
# 3 + 4. gripper settles and the work that was removed
# ---------------------------------------------------------------------------


def _plant_executor(monkeypatch, blend=False):
    pytest.importorskip("pyroki")
    from spark_real.control.score_executor import ScoreExecutor
    from spark_real.tests.blend_plant import UrPlant, fk_rtde

    monkeypatch.setenv("SPARK_UR_BLEND", "1" if blend else "0")
    plant = UrPlant()
    plant.START_LAG_S = 0.2
    home, _ = fk_rtde(UrPlant.HOME_CONFIG)
    pick = np.array([home[0] - 0.03, home[1] + 0.10, -0.245])
    ex = ScoreExecutor(
        plant,
        detection_map={"plushie": {"position_3d": pick.tolist(), "confidence": 0.24}},
        velocity=0.25,
    )
    return ex, plant, pick


def _closes(plant):
    return sum(1 for s in plant.sent if "rq_close" in s)


def _opens(plant):
    return sum(1 for s in plant.sent if "rq_open" in s)


def test_the_settle_is_derived_from_the_datasheet_not_guessed():
    from spark_real.control.executor_grasp import GraspMixin as G

    m = G()
    # Full stroke at speed_norm 60 -> 98 mm/s -> 0.867 s, less the driver's own
    # 0.5 s sleep, plus margin.
    assert m._robotiq_travel_settle_s(60) == pytest.approx(0.517, abs=0.01)
    # A stroke that has (almost) no travel left costs only the margin.
    assert m._robotiq_travel_settle_s(50, from_pos=250, to_pos=255) == pytest.approx(
        m.ROBOTIQ_SETTLE_MARGIN_S, abs=1e-6
    )
    # Slower speed_norm -> longer settle. Monotone, which a constant is not.
    assert m._robotiq_travel_settle_s(20) > m._robotiq_travel_settle_s(90)
    # Never negative, never below the margin.
    assert m._robotiq_travel_settle_s(100) >= m.ROBOTIQ_SETTLE_MARGIN_S


def test_the_settle_covers_the_stroke_and_a_shorter_one_would_be_caught(monkeypatch):
    """THE test that makes the settle claim non-vacuous.

    Same grasp, twice, against a plant whose jaws travel asynchronously and are
    cancelled by the next program. With the datasheet settle the stroke
    completes and gObj confirms. With the settle forced to zero the following
    register publish lands mid-stroke, the jaws freeze part-closed and the grasp
    does NOT get its fast-path confirmation.
    """
    ex, plant, pick = _plant_executor(monkeypatch)
    ex._move_to_keypoint({"keypoint_label": "plushie"}, time.time())
    r = ex._grasp({}, time.time())
    assert r.success and "gObj" in r.message, r.message
    assert plant.jaw_cancels == 0, "the datasheet settle did not cover the stroke"
    assert plant.jaw >= plant.OBJECT_AT_COUNTS - 1

    ex2, plant2, _ = _plant_executor(monkeypatch)
    monkeypatch.setattr(
        type(ex2), "_robotiq_travel_settle_s", lambda self, *a, **k: 0.0, raising=False
    )
    # Slow jaws so the driver's own 0.5 s sleep cannot cover the stroke either.
    plant2.JAW_SPEED_MIN_MM_S = 5.0
    plant2.JAW_SPEED_MAX_MM_S = 12.0
    ex2._move_to_keypoint({"keypoint_label": "plushie"}, time.time())
    ex2._grasp({}, time.time())
    assert plant2.jaw_cancels >= 1, (
        "a zero settle went undetected: the plant is not modelling the stroke, "
        "so it cannot validate the real settle either"
    )
    assert plant2.jaw < plant2.OBJECT_AT_COUNTS, "jaws should have frozen part-closed"


def test_the_re_squeeze_is_skipped_only_when_gobj_confirms(monkeypatch):
    """One close when the jaws are provably on the object, two when not."""
    ex, plant, _ = _plant_executor(monkeypatch)
    ex._move_to_keypoint({"keypoint_label": "plushie"}, time.time())
    ex._grasp({}, time.time())
    assert _closes(plant) == 1, [s[:24] for s in plant.sent if "rq_" in s]

    # Empty jaws: the close runs to the stop, gObj stays False -> the slack
    # take-up re-squeeze MUST still happen.
    ex2, plant2, _ = _plant_executor(monkeypatch)
    plant2.OBJECT_AT_COUNTS = None
    ex2._move_to_keypoint({"keypoint_label": "plushie"}, time.time())
    ex2._grasp({}, time.time())
    assert _closes(plant2) >= 2, "the re-squeeze was dropped on an unconfirmed grip"


def test_jaws_already_open_costs_no_gripper_program(monkeypatch):
    ex, plant, _ = _plant_executor(monkeypatch)
    assert plant.jaw == plant.JAW_OPEN
    assert ex._ensure_jaws_open("test") is False
    assert _opens(plant) == 0

    # Closed jaws must still be opened -- descending part-closed is a collision.
    plant.close_gripper(speed=100)
    time.sleep(0.4)
    assert ex._ensure_jaws_open("test") is True
    assert _opens(plant) == 1

    # An unreadable register must NOT be read as "already open".
    plant.close_gripper(speed=100)
    time.sleep(0.4)
    before = _opens(plant)
    monkeypatch.setattr(plant, "get_gripper_position", lambda publish=True: None)
    assert ex._ensure_jaws_open("test") is True
    assert _opens(plant) == before + 1


def test_transport_check_verifies_first_and_still_re_squeezes(monkeypatch):
    ex, plant, _ = _plant_executor(monkeypatch)
    plant.close_gripper(speed=100)
    time.sleep(0.6)
    assert plant.is_object_detected(publish=False) is True
    n = _closes(plant)
    assert ex._transport_grip_ok("lift") is True
    assert _closes(plant) == n, "an intact grip paid for a re-squeeze it did not need"

    # gObj False and jaws at the closed stop = a real drop. The re-squeeze runs,
    # and the verdict is still DROPPED -- the check was not traded away.
    ex2, plant2, _ = _plant_executor(monkeypatch)
    plant2.OBJECT_AT_COUNTS = None
    plant2.close_gripper(speed=100)
    time.sleep(1.2)
    n2 = _closes(plant2)
    assert ex2._transport_grip_ok("lift") is False
    assert _closes(plant2) > n2, "a suspected drop must still get the re-squeeze"


def test_force_settle_is_a_condition_not_a_sleep(monkeypatch):
    ex, plant, _ = _plant_executor(monkeypatch)
    t0 = time.time()
    assert ex._wait_force_settled(timeout=0.3) is True
    assert time.time() - t0 < 0.15, "settled but still slept out the timeout"

    # No force channel: fall back to waiting exactly as long as the old sleep.
    monkeypatch.setattr(
        plant, "get_tcp_force", lambda: (_ for _ in ()).throw(RuntimeError("no rtde_r"))
    )
    t0 = time.time()
    assert ex._wait_force_settled(timeout=0.2) is False
    assert time.time() - t0 >= 0.19


# ---------------------------------------------------------------------------
# the place descent keeps the servo where it matters
# ---------------------------------------------------------------------------


def test_the_servo_still_owns_the_contact_centimetres_of_the_place(monkeypatch):
    """movej covers free space; the servo keeps every mm that can touch."""
    pytest.importorskip("pyroki")
    from spark_real.control.score_executor import ScoreExecutor
    from spark_real.tests.blend_plant import UrPlant, fk_rtde

    monkeypatch.setenv("SPARK_UR_BLEND", "1")
    plant = UrPlant()
    plant.START_LAG_S = 0.2
    home, _ = fk_rtde(UrPlant.HOME_CONFIG)
    place = np.array([home[0] + 0.08, home[1] - 0.20, -0.20])
    ex = ScoreExecutor(plant, detection_map={}, velocity=0.25)
    ex._transport_grip_ok = lambda w: True
    ex._holding = True
    seen = {}
    servo_to = ex._servo_to

    def spy(position, orientation, velocity=None, **kw):
        seen["from"] = ex._get_current_position().copy()
        seen["to"] = np.asarray(position, dtype=float).copy()
        return servo_to(position, orientation, velocity=velocity, **kw)

    ex._servo_to = spy
    ex._transport_to(place, target_label="bowl")
    drop = float(seen["from"][2] - seen["to"][2])
    assert drop <= ex.PLACE_SERVO_STANDOFF_M + 0.006, (
        f"servo was handed {drop*1000:.0f} mm of descent; the whole point is "
        "that movej covers the free-space part"
    )
    assert drop > 0.005, "the servo must still own the contact approach"
    assert any("spark_blend_path" in s for s in plant.sent)


# ---------------------------------------------------------------------------
# the brake has to scale with the raised velocity, or the raise is a regression
# ---------------------------------------------------------------------------


def _overshoot_after_brake(commanded_v, decel=None):
    """Leading-joint travel AFTER the brake bytes leave, on the plant."""
    pytest.importorskip("pyroki")
    from spark_real.tests.blend_plant import UrPlant

    plant = UrPlant()
    plant.START_LAG_S = 0.0
    q0 = np.array(UrPlant.HOME_CONFIG, dtype=float)
    q1 = q0.copy()
    q1[0] += 1.6  # long enough that the brake lands mid-move at full speed
    plant._send_script(
        "movej([%s], a=%.3f, v=%.3f)"
        % (", ".join("%.6f" % v for v in q1), commanded_v * 2.0, commanded_v)
    )
    # Run up to the commanded speed.
    t_end = time.time() + 0.6
    while time.time() < t_end:
        plant.get_joint_positions()
        time.sleep(0.005)
    at_stop = plant.get_joint_positions()[0]
    if decel is None:
        plant.brake()  # velocity-sized
    else:
        plant._send_script("stopj(%.3f)" % decel)
    t_end = time.time() + 1.5
    while time.time() < t_end:
        plant.get_joint_positions()
        time.sleep(0.005)
    return abs(float(plant.get_joint_positions()[0] - at_stop))


def test_a_fixed_brake_decel_gets_worse_as_the_envelope_rises():
    """Overshoot goes as v^2/2a. This is why the raise needs the sized brake."""
    slow = _overshoot_after_brake(1.05, decel=2.0)
    fast = _overshoot_after_brake(1.78, decel=2.0)
    assert fast > slow * 1.8, (slow, fast)
    # And the velocity-sized brake holds the overshoot roughly constant instead.
    sized_slow = _overshoot_after_brake(1.05)
    sized_fast = _overshoot_after_brake(1.78)
    assert sized_fast < fast * 0.6, (fast, sized_fast)
    assert sized_fast < sized_slow * 2.0, (sized_slow, sized_fast)


def test_halt_arm_does_not_emit_a_hardcoded_decel(monkeypatch):
    """_halt_arm must go through the velocity-sized brake, not stopj(2.0)."""
    pytest.importorskip("pyroki")
    from spark_real.control.score_executor import ScoreExecutor
    from spark_real.tests.blend_plant import UrPlant

    plant = UrPlant()
    ex = ScoreExecutor(plant, detection_map={}, velocity=0.25)
    q1 = np.array(UrPlant.HOME_CONFIG, dtype=float)
    q1[0] += 1.0
    plant.move_to_joint_config_urscript(q1.tolist(), velocity=1.78, acceleration=3.5)
    assert ex._halt_arm() is True
    stops = [s for s in plant.sent if s.lstrip().startswith("stopj")]
    assert stops, plant.sent[-3:]
    decel = float(stops[-1].split("(")[1].rstrip(")"))
    assert decel > 2.0, f"brake decel {decel} was not sized to the 1.78 rad/s move"


def test_the_plants_stop_is_the_broken_rig_stop(monkeypatch):
    """Guard: an offline brake test must not be able to pass via robot.stop()."""
    pytest.importorskip("pyroki")
    from spark_real.tests.blend_plant import UrPlant

    plant = UrPlant()
    plant.START_LAG_S = 0.0
    q1 = np.array(UrPlant.HOME_CONFIG, dtype=float)
    q1[0] += 1.0
    plant.move_to_joint_config_urscript(q1.tolist(), velocity=1.05, acceleration=1.4)
    time.sleep(0.2)
    plant.get_joint_positions()
    plant.stop()
    before = plant.get_joint_positions()[0]
    time.sleep(0.5)
    after = plant.get_joint_positions()[0]
    assert abs(after - before) > 0.05, (
        "UrPlant.stop() braked; it must model rtde_c.stopJ's no-op or every "
        "offline brake test passes spuriously"
    )


def test_legacy_rollback_is_byte_exact_on_BOTH_movej_paths(monkeypatch):
    """The two paths had DIFFERENT old clamps; the rollback must honour both.

    _movej_via_ik was clip(v*6, 0.3, 1.05) / min(v*2, 1.4); the legacy
    Cartesian fallback _movej_to_pose was min(v*3, 1.0) / min(v*2, 1.4), i.e.
    0.75 rad/s at the default velocity, not 1.05. A single rollback formula
    would quietly make the fallback FASTER than the code
    SPARK_LEGACY_JOINT_LIMITS claims to restore.
    """
    monkeypatch.setenv("SPARK_LEGACY_JOINT_LIMITS", "1")
    dq = np.array([1.0, 0, 0, 0, 0, 0])
    assert ik.resolve_joint_limits(0.25, dq=dq) == (1.05, 1.4)
    assert ik.resolve_joint_limits(0.25, pose_fallback=True) == (0.75, 1.4)
    # Outside legacy mode the flag changes nothing: both take the no-dq cap.
    monkeypatch.delenv("SPARK_LEGACY_JOINT_LIMITS")
    assert ik.resolve_joint_limits(0.25) == ik.resolve_joint_limits(0.25, pose_fallback=True)

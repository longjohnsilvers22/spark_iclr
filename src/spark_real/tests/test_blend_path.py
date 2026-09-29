"""Regressions for blended multi-waypoint motion on the UR10e.

The arm used to decelerate to a full stop at every waypoint of a path that was
known in advance. Blending issues that path as ONE URScript program of
``movej(..., r=)`` rows so it never stops between transit waypoints.

Five things must hold and each one gets a test here:

1. The wire format is ur_rtde's, not ours, and the LAST row is always r=0 --
   the terminal waypoint stays a true stop so the exact joint-arrival test
   still applies.
2. Arrival is the FINAL waypoint's joint target. Intermediate waypoints are
   deliberately never reached (the arm cuts the corner at distance r), so a
   per-waypoint test would never fire.
3. Abort reaches the arm MID-PATH. An unabortable multi-second path is worse
   than the staccato version it replaces.
4. Stillness never means arrived without observed motion. That regression
   shipped once: it returned ~20 cm short and the gripper closed above the
   object. A path makes it worse, not better -- a program's upload lag is
   longer than one move's.
5. With blending off, the motion path is unchanged.
"""

import time

import numpy as np
import pytest

from spark_real.control.executor_ik import IkMixin
from spark_real.control.executor_motion import MotionMixin
from spark_real.control.executor_types import AbortRequested
from spark_real.control.waypoints import (
    UR_BLEND_SEG_FRAC,
    JointRow,
    blend_progress_value,
    build_ur_blend_program,
    decode_blend_progress,
    format_movej_row,
    joint_move_seconds,
    resolve_blend_radii,
)

ORIENT = [2.3038, 2.0802, -0.0048]


def _rows(spec, velocity=1.05, acceleration=1.4):
    """spec: list of (q_scalar, x_position). Joints differ on axis 0 only."""
    out = []
    for i, (qs, x) in enumerate(spec):
        out.append(
            JointRow(
                q=np.array([qs, 0, 0, 0, 0, 0], dtype=float),
                position=np.array([x, 0.0, 0.0]),
                velocity=velocity,
                acceleration=acceleration,
                label="r%d" % i,
            )
        )
    return out


# 1. wire format and blend geometry


def test_movej_row_is_byte_identical_to_ur_rtde():
    """The row text is ur_rtde's own PathEntry emitter, minus its leading tab."""
    rc = pytest.importorskip("rtde_control")
    q = [0.7855995302773087, 2.494502670088934, 1.73, -1.7257, -1.25, 2.3459]
    for v, a, r in ((1.05, 1.4, 0.05), (0.7, 1.4, 0.0), (0.303948978, 1.239105, 1e-3)):
        theirs = (
            rc.PathEntry(rc.PathEntry.MoveJ, rc.PathEntry.PositionJoints, list(q) + [v, a, r])
            .toScriptCode()
            .strip("\t\n")
        )
        assert format_movej_row(q, v, a, r) == theirs


def test_last_row_is_always_a_full_stop():
    rows = _rows([(0.3, 0.3), (0.6, 0.6), (0.9, 0.9)])
    kept, _ = resolve_blend_radii([0.0, 0, 0], rows, radius_m=0.05)
    assert kept[-1].blend == 0.0, (
        "the terminal waypoint must decelerate to a stop; the arrival test and "
        "every hard stop at contact/grasp/release depend on it"
    )
    assert all(r.blend > 0 for r in kept[:-1])


def test_blend_never_exceeds_half_a_segment_and_regions_never_overlap():
    # 0.30 m, then a 0.04 m segment: the requested 0.05 must be cut down.
    rows = _rows([(0.3, 0.30), (0.34, 0.34), (0.9, 0.90)])
    kept, _ = resolve_blend_radii([0.0, 0, 0], rows, radius_m=0.05)
    pts = [np.zeros(3)] + [r.position for r in kept]
    seg = [np.linalg.norm(pts[i + 1] - pts[i]) for i in range(len(kept))]
    for i, r in enumerate(kept[:-1]):
        assert r.blend <= UR_BLEND_SEG_FRAC * seg[i] + 1e-12
        assert r.blend <= UR_BLEND_SEG_FRAC * seg[i + 1] + 1e-12
    for i in range(len(kept) - 1):
        assert kept[i].blend + kept[i + 1].blend < seg[i + 1] + 1e-12, (
            "consecutive blend regions overlap; PolyScope does not validate " "this for us"
        )


def test_a_no_op_row_is_dropped_not_blended():
    """_transport_to's first lift is ~0 m long right after move_relative(+0.20)."""
    rows = _rows([(0.500, 0.50), (0.5005, 0.5005), (1.0, 1.00)])
    kept, dropped = resolve_blend_radii([0.0, 0, 0], rows, radius_m=0.05, start_q=np.zeros(6))
    assert [r.label for r in kept] == ["r0", "r2"]
    assert dropped == ["r1"]


def test_a_no_op_row_is_never_dropped_when_it_is_the_terminal_stop():
    rows = _rows([(0.5, 0.5), (0.5, 0.5)])
    kept, dropped = resolve_blend_radii([0.0, 0, 0], rows, radius_m=0.05, start_q=np.zeros(6))
    assert dropped == [] and len(kept) == 2 and kept[-1].blend == 0.0


def test_a_blend_too_small_to_be_worth_it_becomes_a_full_stop():
    rows = _rows([(0.01, 0.01), (0.02, 0.02), (1.0, 1.0)])
    kept, _ = resolve_blend_radii(
        [0.0, 0, 0], rows, radius_m=0.05, min_radius_m=0.01, start_q=np.zeros(6)
    )
    assert kept[0].blend == 0.0  # 0.45 * 0.01 m is below min_radius_m


def test_descent_radius_caps_only_the_junction_into_the_last_row():
    rows = _rows([(0.3, 0.30), (0.6, 0.60), (0.9, 0.90)])
    kept, _ = resolve_blend_radii([0.0, 0, 0], rows, radius_m=0.05, final_radius_m=0.02)
    assert kept[0].blend == pytest.approx(0.05)
    assert kept[1].blend == pytest.approx(
        0.02
    ), "a 5 cm corner-cut into a 15 cm grasp descent is not wanted"


def test_program_is_one_uploadable_block_with_progress_writes():
    rows = _rows([(0.3, 0.30), (0.9, 0.90)])
    resolve_blend_radii([0.0, 0, 0], rows, radius_m=0.05)
    prog = build_ur_blend_program(rows, epoch=7, progress_register=15)
    lines = prog.strip().split("\n")
    assert lines[0] == "def spark_blend_path():"
    assert lines[-2] == "end" and lines[-1] == "spark_blend_path()"
    assert prog.count("movej(") == 2
    # A write before every row, plus the completion token after the last.
    assert prog.count("write_output_integer_register(15,") == 3
    assert "write_output_integer_register(15, %d)" % blend_progress_value(7, 0) in prog
    assert "write_output_integer_register(15, %d)" % blend_progress_value(7, 2) in prog


def test_progress_decoding_ignores_another_epoch():
    assert decode_blend_progress(blend_progress_value(7, 2), 7, 3) == 2
    assert decode_blend_progress(blend_progress_value(6, 2), 7, 3) is None
    assert decode_blend_progress(blend_progress_value(7, 4), 7, 3) is None
    assert decode_blend_progress(None, 7, 3) is None


def test_joint_move_seconds_matches_the_trapezoid():
    # cruise reached: dq/v + v/a
    assert joint_move_seconds(1.0, 1.05, 1.4) == pytest.approx(1.0 / 1.05 + 1.05 / 1.4)
    # triangular: 2*sqrt(dq/a)
    assert joint_move_seconds(0.05, 1.05, 1.4) == pytest.approx(2 * np.sqrt(0.05 / 1.4))


# 2-4. arrival, abort, premature arrival


class _PathArm:
    """Fake arm that traverses a list of joint waypoints on a schedule.

    Intermediate waypoints ARE passed through exactly (harsher than a real
    blended path, which cuts the corner) so a per-waypoint arrival test would
    fire and be caught.
    """

    SUPPORTS_URSCRIPT = True
    robot_family = "ur10e"

    def __init__(self, qs, lag=0.2, seg_s=0.4, epoch=1, register=True, freeze=None):
        self.qs = [np.asarray(q, dtype=float) for q in qs]
        self.lag = lag
        self.seg_s = seg_s
        self.epoch = epoch
        self.register = register
        self.freeze = freeze  # stop moving after this many seconds of motion
        self.t0 = time.time()
        self.sent = []
        self.lease_cleared = 0
        self.lease_notes = 0

    def _elapsed(self):
        el = max(0.0, time.time() - self.t0 - self.lag)
        if self.freeze is not None:
            el = min(el, self.freeze)
        return el

    def get_joint_positions(self):
        el = self._elapsed()
        n = len(self.qs)
        k = min(int(el // self.seg_s), n - 1)
        frac = min(1.0, (el - k * self.seg_s) / self.seg_s)
        prev = np.zeros(6) if k == 0 else self.qs[k - 1]
        return prev + (self.qs[k] - prev) * frac

    def get_tcp_pose(self):
        q = self.get_joint_positions()
        return np.concatenate([[q[0], 0.0, 0.0], ORIENT])

    def get_output_int_register(self, reg):
        if not self.register:
            raise RuntimeError("register %d not surfaced by this build" % reg)
        el = self._elapsed()
        if time.time() - self.t0 < self.lag:
            return 0  # program not started: nothing written yet
        k = min(int(el // self.seg_s), len(self.qs))
        return blend_progress_value(self.epoch, k)

    def _send_script(self, s):
        self.sent.append(s)
        return True

    def _note_motion_script(self, s):
        self.lease_notes += 1

    def clear_motion_lease(self):
        self.lease_cleared += 1


class _Waiter(MotionMixin, IkMixin):
    def __init__(self, robot):
        self.robot = robot
        self._recorder = None
        self._abort = False
        self._pipeline = None
        self.velocity = 0.25
        self._demo_mode_cached = False
        self._blend_cfg_cached = {
            "enabled": True,
            "radius_m": 0.05,
            "descent_radius_m": 0.02,
            "min_radius_m": 0.01,
            "max_rows": 8,
            "register": 15,
        }

    def _get_current_position(self):
        return np.asarray(self.robot.get_tcp_pose()[:3], dtype=float)

    def _check_abort(self):
        if self._abort:
            raise AbortRequested()


def _three_row_path(**arm_kw):
    qs = [
        np.array([0.3, 0, 0, 0, 0, 0]),
        np.array([0.6, 0, 0, 0, 0, 0]),
        np.array([0.9, 0, 0, 0, 0, 0]),
    ]
    arm = _PathArm(qs, **arm_kw)
    rows = _rows([(0.3, 0.3), (0.6, 0.6), (0.9, 0.9)])
    resolve_blend_radii([0.0, 0, 0], rows, radius_m=0.05)
    return arm, rows


def test_arrival_is_the_final_joint_target():
    arm, rows = _three_row_path(lag=0.2, seg_s=0.3)
    w = _Waiter(arm)
    t0 = time.time()
    assert w._wait_for_blended_path(rows, epoch=1, start_pos=np.zeros(3)) is True
    assert time.time() - t0 < 4.0
    assert np.max(np.abs(arm.get_joint_positions() - rows[-1].q)) < 0.02
    assert arm.lease_cleared >= 1, "the driver's motion lease must be retired"


def test_no_intermediate_waypoint_is_mistaken_for_arrival():
    """The arm passes exactly through rows 0 and 1; only row 2 ends the wait."""
    arm, rows = _three_row_path(lag=0.2, seg_s=0.5)
    w = _Waiter(arm)
    assert w._wait_for_blended_path(rows, epoch=1, start_pos=np.zeros(3)) is True
    q = arm.get_joint_positions()
    assert q[0] > 0.85, f"returned at an intermediate waypoint (q0={q[0]:.3f})"


def test_stillness_during_a_long_program_upload_lag_is_not_arrival():
    """THE shipped-once regression: it returned ~20 cm short and closed on air.

    The lag exceeds URSCRIPT_START_LAG_S, so the arm is still stationary when
    the start-lag window closes; "still" there means "not started yet". The
    register is unavailable, so the ONLY remaining exits are the joint match,
    Cartesian proximity, and stillness -- and stillness must require having
    observed motion. Delete `and ever_moved` and this test fails.
    """
    arm, rows = _three_row_path(lag=2.5, seg_s=0.3, register=False)
    w = _Waiter(arm)
    ok = w._wait_for_blended_path(rows, epoch=1, start_pos=np.zeros(3))
    q = arm.get_joint_positions()
    assert ok is True
    assert q[0] > 0.85, (
        f"exited during the upload lag at q0={q[0]:.3f}: stillness was treated "
        "as arrival without ever observing motion"
    )


def test_a_stale_register_from_an_earlier_path_cannot_report_completion():
    """The register survives paths and process restarts; the epoch must step off it."""
    arm, rows = _three_row_path(lag=0.2, seg_s=0.3, epoch=5)
    w = _Waiter(arm)
    # Register currently holds "epoch 5, all 3 rows done" -- a full completion
    # token for a path that has not been sent yet.
    arm.get_output_int_register = lambda reg: blend_progress_value(5, 3)
    epoch = w._next_blend_epoch(3)
    assert epoch != 5
    assert decode_blend_progress(blend_progress_value(5, 3), epoch, 3) is None


def test_a_frozen_progress_register_with_a_still_arm_is_a_stall():
    """The 15.0 s / 1817-speedl grind must not come back as a silent timeout."""
    arm, rows = _three_row_path(lag=0.2, seg_s=0.3, freeze=0.35)
    w = _Waiter(arm)
    w.BLEND_STALL_S = 0.6
    t0 = time.time()
    assert w._wait_for_blended_path(rows, epoch=1, start_pos=np.zeros(3)) is False
    assert time.time() - t0 < 4.0, "stall detection did not fire; burned the budget"
    assert any(
        s.startswith("stopj") for s in arm.sent
    ), "a stalled path must be stopped, not waited out"


def test_a_stopped_arm_on_the_LAST_row_is_the_ordinary_end_of_a_move():
    """Parity with the per-move wait: stall detection covers intermediate rows.

    Once the final movej is executing there is no next waypoint to be stuck
    before, so a stopped arm there means the move ended -- which is exactly what
    _wait_for_motion concludes for a single movej today. Anything wrong with
    where it ended is the caller's verification's job, as it always was.
    """
    arm, rows = _three_row_path(lag=0.2, seg_s=0.25, freeze=0.55)
    w = _Waiter(arm)
    w.BLEND_STALL_S = 0.6
    t0 = time.time()
    assert w._wait_for_blended_path(rows, epoch=1, start_pos=np.zeros(3)) is True
    assert time.time() - t0 < 4.0
    assert not any(s.startswith("stopj") for s in arm.sent)


def test_abort_mid_path_stops_the_arm_and_raises():
    """Requirement: an unabortable blended path is a blocker, not a speedup."""
    arm, rows = _three_row_path(lag=0.2, seg_s=2.0)
    w = _Waiter(arm)

    import threading

    def kill():
        time.sleep(0.6)
        w._abort = True

    threading.Thread(target=kill, daemon=True).start()
    t0 = time.time()
    with pytest.raises(AbortRequested):
        w._wait_for_blended_path(rows, epoch=1, start_pos=np.zeros(3))
    elapsed = time.time() - t0
    assert elapsed < 2.0, f"abort took {elapsed:.2f}s to unwind"
    # The arm was still mid-path (q0 ~ 0.15 of 0.9) and MUST have been braked.
    assert any(s.startswith("stopj") for s in arm.sent), (
        "abort never reached the arm: _stop_robot's rtde_c.stopJ no-ops once "
        "_send_script has stopped the control script, i.e. always"
    )
    assert arm.get_joint_positions()[0] < 0.9


def test_the_motion_lease_is_refreshed_for_the_whole_path():
    """_MOVEJ_LEASE_S is 1.5 s; a lapsed lease lets a gripper publish cancel the path."""
    arm, rows = _three_row_path(lag=0.2, seg_s=0.6)
    w = _Waiter(arm)
    w._wait_for_blended_path(rows, epoch=1, start_pos=np.zeros(3))
    assert arm.lease_notes >= 5, (
        "the lease was not kept alive across the path (%d notes)" % arm.lease_notes
    )


# 5. blending off changes nothing; blending on removes the stops


class _FakeIkArm(_PathArm):
    def get_tcp_offset(self):
        return [0.0, 0.0, 0.1725, 0.0, 0.0, 0.0]


def _executor(blend, monkeypatch, cross_node=True):
    from spark_real.control.score_executor import ScoreExecutor

    monkeypatch.setenv("SPARK_UR_BLEND", "1" if blend else "0")
    monkeypatch.setenv("SPARK_UR_BLEND_CROSS_NODE", "1" if cross_node else "0")
    arm = _FakeIkArm([np.zeros(6)], lag=0.0, seg_s=0.01)
    ex = ScoreExecutor(arm, detection_map={}, velocity=0.25)
    ex.calls = []
    ex._demo_mode_cached = False
    ex._blend_cfg_cached = None
    ex._move_to = lambda p, o, velocity=None: ex.calls.append(("move_to", tuple(np.round(p, 4))))
    ex._servo_to = lambda p, o, velocity=None: ex.calls.append(("servo_to", tuple(np.round(p, 4))))
    ex._transport_grip_ok = lambda w: ex.calls.append(("grip_ok", w)) or True
    return ex


PLACE = np.array([-0.7, -0.19, -0.2])


def _servo_descent(ex):
    """Vertical drop the servo was handed, in metres."""
    for kind, pos in ex.calls:
        if kind == "servo_to":
            prev = [p for k, p in ex.calls if k == "move_to"]
            return float(prev[-1][2] - pos[2]) if prev else None
    return None


def test_blend_off_leaves_the_transport_sequence_untouched(monkeypatch):
    ex = _executor(False, monkeypatch)
    ex._blend_transit = lambda *a, **k: pytest.fail("_blend_transit ran with blend off")
    assert ex._transport_to(PLACE, target_label="bowl") is True
    kinds = [c[0] for c in ex.calls]
    # lift -> check -> above_target -> check -> place_standoff -> servo.
    # The third move_to is the standoff: the servo used to be handed the WHOLE
    # drop from safe_z and spend its slowest seconds crawling through free space.
    assert kinds == [
        "move_to", "grip_ok", "move_to", "grip_ok", "move_to", "servo_to"
    ], kinds
    assert _servo_descent(ex) == pytest.approx(ex.PLACE_SERVO_STANDOFF_M, abs=1e-6)


def test_blend_on_transport_is_one_phrase_one_grip_check_then_the_servo(monkeypatch):
    ex = _executor(True, monkeypatch)
    seen = {}

    def fake_blend(waypoints, final_radius_m=None, phrase=""):
        seen["rows"] = [tuple(np.round(w[0], 4)) for w in waypoints]
        seen["final_radius_m"] = final_radius_m
        return True

    ex._blend_transit = fake_blend
    assert ex._transport_to(PLACE, target_label="bowl") is True
    kinds = [c[0] for c in ex.calls]
    assert kinds == ["grip_ok", "servo_to"], kinds
    # lift, above_target, place_standoff -- the standoff is a row of the SAME
    # program, so folding the descent in costs no extra upload.
    assert len(seen["rows"]) == 3, seen["rows"]
    assert seen["rows"][-1][2] == pytest.approx(
        PLACE[2] + ex.PLACE_SERVO_STANDOFF_M, abs=1e-4
    )
    # A transport ends in a servo descent, so its last row must be a hard stop.
    assert seen["final_radius_m"] == 0.0


def test_a_refused_phrase_falls_back_to_the_old_per_move_sequence(monkeypatch):
    ex = _executor(True, monkeypatch)
    ex._blend_transit = lambda *a, **k: False
    assert ex._transport_to(PLACE, target_label="bowl") is True
    kinds = [c[0] for c in ex.calls]
    assert kinds == [
        "move_to", "grip_ok", "move_to", "grip_ok", "move_to", "servo_to"
    ], kinds


def test_a_place_below_the_standoff_is_left_entirely_to_the_servo(monkeypatch):
    """No standoff row when it would be at or above the clearance height.

    Guards against inserting a degenerate movej (or a row that inverts the
    descent) when the container sits right at the clearance plane.
    """
    ex = _executor(False, monkeypatch)
    ex._safe_clearance_z = lambda a, b: float(PLACE[2]) + 0.005
    assert ex._transport_to(PLACE, target_label="bowl") is True
    kinds = [c[0] for c in ex.calls]
    assert kinds == ["move_to", "grip_ok", "move_to", "grip_ok", "servo_to"], kinds


def test_a_lead_in_is_flown_per_move_when_the_phrase_is_refused(monkeypatch):
    ex = _executor(True, monkeypatch)
    ex._blend_transit = lambda *a, **k: False
    lead = [(np.array([-0.85, 0.31, -0.05]), ORIENT, None, "rel")]
    ex._transport_to(np.array([-0.7, -0.19, -0.2]), target_label="bowl", lead=lead)
    assert [c[0] for c in ex.calls][:2] == ["move_to", "move_to"]
    assert ex.calls[0][1] == tuple(np.round(lead[0][0], 4))


def test_a_drop_at_the_end_of_a_blended_phrase_still_fails_the_transport(monkeypatch):
    ex = _executor(True, monkeypatch)
    ex._blend_transit = lambda *a, **k: True
    ex._transport_grip_ok = lambda w: False
    ex._holding = True
    assert ex._transport_to(np.array([-0.7, -0.19, -0.2]), target_label="bowl") is False
    assert ex._holding is False
    assert [c[0] for c in ex.calls] == [], "released into thin air after a drop"


def test_the_legato_phrase_refuses_shapes_it_does_not_understand(monkeypatch):
    ex = _executor(True, monkeypatch)
    ex._blend_transit = lambda *a, **k: pytest.fail("should not have dispatched")
    ex._holding = True
    # a grasp is a barrier, never a note
    assert (
        ex._execute_ur_blend_phrase(
            [{"type": "move_relative", "params": {"dz": 0.2}}, {"type": "grasp"}]
        )
        is False
    )
    # a non-terminal move_to_keypoint is not a transport
    assert (
        ex._execute_ur_blend_phrase(
            [
                {"type": "move_to_keypoint", "params": {"keypoint_label": "a"}},
                {"type": "move_relative", "params": {"dz": 0.2}},
            ]
        )
        is False
    )
    # unknown keypoint
    assert (
        ex._execute_ur_blend_phrase(
            [
                {"type": "move_relative", "params": {"dz": 0.2}},
                {"type": "move_to_keypoint", "params": {"keypoint_label": "nope"}},
            ]
        )
        is False
    )


def test_cross_node_folding_is_off_by_default(monkeypatch):
    """A phrase that consumes the transport cannot report a drop to the core loop.

    executor_core._run_actions skips its failure/recovery block entirely when a
    legato phrase returns True, so a drop found inside the phrase would be
    reported and then ignored -- no _attempt_recovery, no re-grasp, and the
    following release opens on nothing. Worth ~1 s; not worth that.
    """
    ex = _executor(True, monkeypatch, cross_node=False)
    ex.detection_map = {"bowl": {"position_3d": [-0.7, -0.19, -0.2]}}
    ex._holding = True
    ex._transport_to = lambda *a, **k: pytest.fail("cross-node fold ran while off")
    assert (
        ex._execute_ur_blend_phrase(
            [
                {"type": "move_relative", "params": {"dz": 0.20}},
                {"type": "move_to_keypoint", "params": {"keypoint_label": "bowl"}},
            ]
        )
        is False
    )


def test_the_legato_phrase_hands_the_relative_lift_to_the_transport(monkeypatch):
    ex = _executor(True, monkeypatch)
    ex.detection_map = {"bowl": {"position_3d": [-0.7, -0.19, -0.2]}}
    ex._holding = True
    seen = {}
    ex._transport_to = lambda *a, **k: seen.update(k) or True
    ok = ex._execute_ur_blend_phrase(
        [
            {"type": "move_relative", "params": {"dz": 0.20}},
            {"type": "move_to_keypoint", "params": {"keypoint_label": "bowl"}},
        ]
    )
    assert ok is True
    assert len(seen["lead"]) == 1
    assert seen["lead"][0][0][2] == pytest.approx(0.20)  # dz applied to TCP z
    assert len(ex._results) == 2 and all(r.success for r in ex._results)


def test_blending_off_refuses_the_phrase_outright(monkeypatch):
    ex = _executor(False, monkeypatch)
    ex._holding = True
    assert (
        ex._execute_ur_blend_phrase(
            [
                {"type": "move_relative", "params": {"dz": 0.2}},
                {"type": "move_relative", "params": {"dz": 0.1}},
            ]
        )
        is False
    )


def test_ik_failure_on_any_row_rejects_the_whole_phrase_without_sending(monkeypatch):
    import spark_real.control.executor_ik as ik

    ex = _executor(True, monkeypatch)
    calls = {"n": 0}

    def flaky(pos, orient, q_seed=None, tcp_offset=None, **kw):
        calls["n"] += 1
        return None if calls["n"] == 2 else np.zeros(6)

    monkeypatch.setattr(ik, "solve_ik_ur10e_rtde", flaky)
    ok = ex._blend_transit(
        [
            (np.array([-0.8, 0.1, 0.1]), ORIENT),
            (np.array([-0.8, 0.2, 0.1]), ORIENT),
            (np.array([-0.8, 0.3, 0.1]), ORIENT),
        ]
    )
    assert ok is False
    assert not any(
        "spark_blend_path" in s for s in ex.robot.sent
    ), "a partially-validated path must never reach the socket"


def test_an_ik_branch_flip_refuses_to_blend(monkeypatch):
    import spark_real.control.executor_ik as ik

    ex = _executor(True, monkeypatch)
    seq = [np.zeros(6), np.array([3.0, 0, 0, 0, 0, 0])]

    monkeypatch.setattr(
        ik,
        "solve_ik_ur10e_rtde",
        lambda *a, **k: seq.pop(0) if seq else np.zeros(6),
    )
    assert (
        ex._blend_transit(
            [(np.array([-0.8, 0.1, 0.1]), ORIENT), (np.array([-0.8, 0.2, 0.1]), ORIENT)]
        )
        is False
    )


def test_a_single_row_phrase_is_left_to_the_normal_move_path(monkeypatch):
    import spark_real.control.executor_ik as ik

    ex = _executor(True, monkeypatch)
    monkeypatch.setattr(ik, "solve_ik_ur10e_rtde", lambda *a, **k: np.zeros(6))
    # Both rows resolve to the same joint config: one is dropped as a no-op and
    # a one-row "path" is not a path.
    assert (
        ex._blend_transit(
            [(np.array([-0.8, 0.1, 0.1]), ORIENT), (np.array([-0.8, 0.1, 0.1]), ORIENT)]
        )
        is False
    )
    assert not any("spark_blend_path" in s for s in ex.robot.sent)


def test_env_switch_overrides_the_config_default(monkeypatch):
    ex = _executor(False, monkeypatch)
    ex._blend_cfg_cached = None
    monkeypatch.setenv("SPARK_UR_BLEND", "1")
    assert ex._blend_cfg()["enabled"] is True
    ex._blend_cfg_cached = None
    monkeypatch.setenv("SPARK_UR_BLEND", "0")
    assert ex._blend_cfg()["enabled"] is False


def test_demo_mode_never_blends(monkeypatch):
    ex = _executor(True, monkeypatch)
    ex._demo_mode_cached = True
    assert ex._blend_enabled() is False, (
        "demo_mode exists so every recorded tick carries a commanded velocity; "
        "a blended program emits one setpoint for a multi-second motion"
    )


# The 4mm servo stall must survive a blended prefix (plant-backed).


def test_servo_stall_exit_still_applies_after_a_blended_path(monkeypatch):
    """A blended path that ends in a servo descent keeps the stall exit.

    4 mm of unreachability used to cost 15.01 s of continuous speedl. Blending
    must not resurrect that, so this drives the REAL blended path into the REAL
    CartesianServo against a plant with a hard floor 4 mm above the target.
    """
    pytest.importorskip("pyroki")
    from spark_real.control.score_executor import ScoreExecutor
    from spark_real.tests.blend_plant import UrPlant, fk_rtde

    monkeypatch.setenv("SPARK_UR_BLEND", "1")
    plant = UrPlant()
    plant.START_LAG_S = 0.2
    home, _ = fk_rtde(UrPlant.HOME_CONFIG)
    place = np.array([home[0] + 0.08, home[1] - 0.20, -0.20])
    plant.floor_z = place[2] + 0.004  # unreachable by 4 mm

    ex = ScoreExecutor(plant, detection_map={}, velocity=0.25)
    ex._transport_grip_ok = lambda w: True
    ex._holding = True
    t0 = time.time()
    ex._transport_to(place, target_label="bowl")
    elapsed = time.time() - t0
    assert ex._servo.last_exit == "stalled", ex._servo.last_exit
    assert elapsed < 15.0, f"still hammering after a blended prefix: {elapsed:.1f}s"
    assert any("spark_blend_path" in s for s in plant.sent), "phrase never blended"


# Safety parity: a phrase must not bypass the checks a single move gets


def test_every_row_is_workspace_clipped_like_a_single_move(monkeypatch):
    import spark_real.control.executor_ik as ik

    ex = _executor(True, monkeypatch)
    seen = []
    monkeypatch.setattr(
        ik,
        "solve_ik_ur10e_rtde",
        lambda pos, *a, **k: seen.append(np.array(pos)) or np.array([0.1, 0, 0, 0, 0, 0]),
    )
    # Second row is 2 m above the box ceiling.
    ex._blend_transit(
        [
            (np.array([-0.8, 0.1, 0.1]), ORIENT),
            (np.array([-0.8, 0.1, 2.5]), ORIENT),
        ]
    )
    assert len(seen) == 2
    assert (
        seen[1][2] <= ex.WORKSPACE_MAX[2] + 1e-9
    ), "a phrase row escaped the workspace clip that _move_to applies"


def test_an_oversize_segment_refuses_the_phrase(monkeypatch):
    import spark_real.control.executor_ik as ik

    ex = _executor(True, monkeypatch)
    monkeypatch.setattr(
        ik, "solve_ik_ur10e_rtde", lambda *a, **k: pytest.fail("should not reach IK")
    )
    ex.WORKSPACE_MIN = np.array([-5.0, -5.0, -5.0])
    ex.WORKSPACE_MAX = np.array([5.0, 5.0, 5.0])
    ex.MAX_REACH = 100.0
    assert (
        ex._blend_transit(
            [(np.array([0.0, 0.0, 0.0]), ORIENT), (np.array([0.0, 0.0, 3.0]), ORIENT)]
        )
        is False
    )


def test_every_row_goes_through_the_wrappers_joint_filter(monkeypatch):
    """SafeRobot._send_script only inspects the FIRST movej in a script."""
    import spark_real.control.executor_ik as ik

    ex = _executor(True, monkeypatch)
    qs = [np.array([0.1, 0, 0.2, 0, 0, 0]), np.array([0.1, 0, 0.3, 0, 0, 0])]
    monkeypatch.setattr(ik, "solve_ik_ur10e_rtde", lambda *a, **k: qs.pop(0) if qs else np.zeros(6))
    checked = []
    ex.robot.check_joint_target_safe = lambda q: checked.append(q) or (q[2] < 0.25)
    ok = ex._blend_transit(
        [(np.array([-0.8, 0.1, 0.1]), ORIENT), (np.array([-0.8, 0.2, 0.1]), ORIENT)]
    )
    assert ok is False
    assert len(checked) == 2, "only the first row was checked"
    assert not any("spark_blend_path" in s for s in ex.robot.sent)


def test_the_wrappers_own_epsilon_is_used_when_there_is_no_predicate(monkeypatch):
    """One source for eps_singularity: it is read off the wrapper, not inlined."""
    import spark_real.control.executor_ik as ik

    ex = _executor(True, monkeypatch)
    # q2 == pi is exactly the elbow singularity SafeRobot guards.
    qs = [np.array([0.1, 0, 0.5, 0, 0, 0]), np.array([0.1, 0, np.pi, 0, 0, 0])]
    monkeypatch.setattr(ik, "solve_ik_ur10e_rtde", lambda *a, **k: qs.pop(0) if qs else np.zeros(6))
    ex.robot._cfg = type("_Cfg", (), {"eps_singularity": 0.15})()
    assert (
        ex._blend_transit(
            [(np.array([-0.8, 0.1, 0.1]), ORIENT), (np.array([-0.8, 0.2, 0.1]), ORIENT)]
        )
        is False
    )

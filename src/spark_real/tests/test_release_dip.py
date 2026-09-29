"""The pen/spoon dip: deep tilts land on the negative-pitch side, chosen early.

Ground truth is the human teleop corpus this port copies (UR10e, ~15Hz,
/data/teleop_episodes/put the pen in the bin/episode_0000..0003): the carry is
near-vertical (1-17 deg), the dip is a pure rotation about the tool's own X
adding 38-43 deg to reach 52-64 deg from vertical AT RIM HEIGHT, the yaw is
unchanged by the dip (<=4 deg), the tilt ramps over ~2.5-3s at z=0.11-0.21m,
and the jaws open motionless 0.5-1.5s after max tilt.

The machine constraints the port must respect (measured 2026-08-20):

  * MAX_POS_PITCH_RAD = 40 deg is a PHYSICAL clearance limit (wrist-mounted
    RealSense) -- deep dips must not raise it;
  * the at-release 180 wrist flip that used to buy the 90 deg negative
    budget was removed at 13:03 after a near-360 wrist spin blew the 20s
    release budget -- it must not come back;
  * so the ONLY route to a 52-64 deg dip is choosing the place-approach YAW,
    before the transport arrives, so the dip falls on the negative-pitch
    side -- folded into the approach orientation, zero extra motion.

Everything here runs against a synthetic arm; no camera, no robot, no server.
"""

from __future__ import annotations

import time

import numpy as np
from scipy.spatial.transform import Rotation

from spark_real.control.grasp_strategy import tool_z_tilt_deg
from spark_real.control.score_executor import ScoreExecutor

# Two carry orientations, one per pitch side. NEG_SIDE has tool Y -> world -X
# (the release derives pitch_sign=-1 -> the 90 deg budget); POS_SIDE is the
# rig's own grasp_orientation from configs/ur10e_default.yaml, whose tool Y
# points at world +X (pitch_sign=+1 -> the 40 deg clamp). That the rig's BASE
# carry sits on the clamped side is exactly why the dip-side yaw choice exists.
NEG_SIDE = [2.2214, -2.2214, 0.0]
POS_SIDE = [2.3038, 2.0802, -0.0048]

START_XYZ = (-0.75, 0.10, -0.13)  # a resolved release z, rim+~9cm territory


def _tool_y_x(orient) -> float:
    """World-X component of tool Y: the release's own pitch_sign test."""
    return float(Rotation.from_rotvec(list(orient)).apply([0.0, 1.0, 0.0])[0])


def _heading(orient, axis) -> float:
    """World-XY heading (rad) of a tool axis."""
    v = Rotation.from_rotvec(list(orient)).apply(list(axis))
    return float(np.arctan2(v[1], v[0]))


def _rot_between(a, b) -> float:
    """Magnitude (rad) of the rotation from orientation a to b."""
    return float(
        (Rotation.from_rotvec(list(a)).inv() * Rotation.from_rotvec(list(b))).magnitude()
    )


class FakeArm:
    """Robotiq-shaped driver whose pose is whatever was last commanded."""

    GRIPPER_TYPE = "robotiq_2f85"
    robot_family = "ur10e"
    SUPPORTS_URSCRIPT = True

    def __init__(self, xyz, orient):
        self.tcp = np.array([*xyz, *orient], dtype=float)
        self.jaw = 216.7
        self.obj = True
        self.opens = 0
        self.events = []  # ordered log shared with the executor stubs

    def get_tcp_pose(self):
        return self.tcp.copy()

    def get_observation(self):
        return {"tcp_pose": self.tcp.copy()}

    def set_pose(self, pose6):
        self.tcp = np.asarray(pose6, dtype=float).copy()

    def open_gripper(self):
        self.opens += 1
        self.events.append(("open",))
        self.jaw = 2.5
        self.obj = False

    def get_gripper_position(self, publish=True):
        return self.jaw

    def is_object_detected(self, publish=True):
        return self.obj

    def _publish_gripper_state(self, force=False):
        return None


class FakeServo:
    """Servo that lands exactly where it is told, orientation included."""

    def __init__(self, arm):
        self.arm = arm
        self.max_vel_linear = 0.15
        self.last_exit = "none"
        self.last_pos_err = 0.0
        self.last_ori_err = 0.0
        self.calls = []

    def move_to_pose(self, target_pose, velocity=None):
        self.calls.append(list(target_pose))
        self.arm.set_pose(target_pose)
        self.last_exit = "converged"
        return True

    def _get_tcp_pose(self):
        return self.arm.get_tcp_pose()

    def abort(self):
        return None


def make_executor(carry_orient, start=START_XYZ, detection_map=None):
    """Executor mid-carry over the container, all motion stubs recording."""
    arm = FakeArm(start, carry_orient)
    ex = ScoreExecutor(arm, detection_map=dict(detection_map or {}), velocity=0.2)
    ex._servo = FakeServo(arm)
    ex._holding = True
    ex.RELEASE_CONFIRM_TIMEOUT_S = 0.3
    ex.RELEASE_OPEN_SETTLE_S = 0.0
    ex._blend_enabled = lambda: False
    ex._transport_grip_ok = lambda waypoint: True

    sleeps = []

    def fake_abort_sleep(duration, tick=0.05):
        sleeps.append(float(duration))
        arm.events.append(("sleep", float(duration)))

    ex._abort_sleep = fake_abort_sleep
    ex.sleeps = sleeps

    linear_moves = []

    def fake_move_to_linear(position, orientation, velocity=None):
        linear_moves.append(
            (np.asarray(position, dtype=float).copy(), list(orientation), velocity)
        )
        arm.set_pose(list(np.asarray(position, dtype=float)[:3]) + list(orientation))
        arm.events.append(("movel",))

    ex._move_to_linear = fake_move_to_linear
    ex.linear_moves = linear_moves

    moves = []

    def fake_move_to(position, orientation, velocity=None):
        moves.append(
            (np.asarray(position, dtype=float).copy(), list(orientation), velocity)
        )
        arm.set_pose(list(np.asarray(position, dtype=float)[:3]) + list(orientation))

    ex._move_to = fake_move_to
    ex.moves = moves
    return ex


def release(ex, tilt_angle, pitch_sign=1.0):
    return ex._release(
        {"tilt_angle": float(tilt_angle), "pitch_sign": float(pitch_sign)},
        time.time(),
    )


# --- (a) a requested 46-60 deg dip is COMMANDED, not undershot --------------


def test_deep_dip_commands_matching_tilt_from_vertical():
    """On the negative-pitch side the human's 46-60 deg band passes through
    1:1: the commanded orientation's tilt-from-vertical IS the request."""
    for req in (0.80, 0.90, 1.05):  # 45.8, 51.6, 60.2 deg
        ex = make_executor(NEG_SIDE)
        result = release(ex, req)
        assert result.success is True
        tilt_cmd_orient = ex.linear_moves[0][1]
        assert abs(tool_z_tilt_deg(tilt_cmd_orient) - np.rad2deg(req)) < 0.5


def test_the_46deg_regression_is_fixed_on_the_negative_side():
    """2026-08-20: Gemini asked 46 deg, the clamp gave 40. With the dip on the
    negative side the same request now commands the full 46."""
    ex = make_executor(NEG_SIDE)
    release(ex, 0.803)  # 46 deg
    assert abs(tool_z_tilt_deg(ex.linear_moves[0][1]) - 46.0) < 0.5


def test_positive_side_still_clamps_at_40():
    """The 40 deg wrist-camera clearance limit is physical and stays: a deep
    request that somehow arrives on the positive side is clamped, not flipped."""
    ex = make_executor(POS_SIDE)
    release(ex, 0.90)
    # The dip ADDED to the carry is the clamped 40 deg exactly; the absolute
    # tilt-from-vertical differs by the rig base's own ~2 deg lean.
    assert abs(np.rad2deg(_rot_between(POS_SIDE, ex.linear_moves[0][1])) - 40.0) < 0.5
    assert 37.0 < tool_z_tilt_deg(ex.linear_moves[0][1]) < 43.0


# --- (b) no 180 flip motion is commanded at the release ---------------------


def test_no_flip_motion_at_release_on_either_side():
    """Every orientation the release commands stays within the dip itself of
    the carry orientation -- nothing approaching the removed 180 flip (which
    read as a near-360 wrist spin on the rig)."""
    for carry, req in ((POS_SIDE, 0.90), (NEG_SIDE, 0.90)):
        ex = make_executor(carry)
        release(ex, req)
        assert len(ex.linear_moves) == 2  # tilt ramp + tilted insert, only
        for _pos, orient_cmd, _v in ex.linear_moves:
            assert _rot_between(carry, orient_cmd) < np.pi / 2


# --- (c) the placement yaw survives the tilt --------------------------------


def test_placement_yaw_survives_the_tilt():
    """The dip composes with the yaw the place aligned: it rotates about the
    CURRENT tool X, so tool X's world heading -- the aligned yaw -- is
    untouched, while tool Z tips by exactly the request."""
    yaw = np.deg2rad(35.0)
    carry = (
        Rotation.from_euler("z", yaw) * Rotation.from_rotvec(NEG_SIDE)
    ).as_rotvec().tolist()
    ex = make_executor(carry)
    release(ex, 0.80)
    tilt_cmd_orient = ex.linear_moves[0][1]
    assert abs(_heading(tilt_cmd_orient, [1, 0, 0]) - _heading(carry, [1, 0, 0])) < 1e-6
    assert abs(tool_z_tilt_deg(tilt_cmd_orient) - np.rad2deg(0.80)) < 0.5


# --- the ported human profile: high ramp, descend to resolved z, hold -------


def test_tilt_ramps_high_and_insert_descends_to_resolved_z():
    """Teleop: tilt at z=0.11-0.21m, then descend with tilt at max, open at
    the low point. So the tilt movel RAISES by TILT_RAMP_RAISE_M and the
    insert returns exactly to the resolved release z -- not the old fixed
    6cm below it, which punched through the rim guard release_height set."""
    ex = make_executor(NEG_SIDE)
    release(ex, 0.90)
    (tilt_pos, _o1, _v1), (insert_pos, _o2, _v2) = ex.linear_moves
    assert abs(tilt_pos[2] - (START_XYZ[2] + ex.TILT_RAMP_RAISE_M)) < 1e-9
    assert abs(insert_pos[2] - START_XYZ[2]) < 1e-9


def test_hold_at_max_tilt_before_the_jaws_open():
    """Teleop: max tilt is reached 0.5-1.5s BEFORE the release and the jaws
    open motionless. The settle must land between the insert and the open."""
    ex = make_executor(NEG_SIDE)
    release(ex, 0.90)
    events = ex.robot.events
    settle_i = events.index(("sleep", ex.TILT_SETTLE_S))
    open_i = events.index(("open",))
    last_movel_i = max(i for i, e in enumerate(events) if e == ("movel",))
    assert last_movel_i < settle_i < open_i


def test_a_straight_release_has_no_ramp_and_no_settle():
    """tilt_angle 0 keeps the plain release: no movels, no tilt settle."""
    ex = make_executor(NEG_SIDE)
    result = release(ex, 0.0)
    assert result.success is True
    assert ex.linear_moves == []
    assert ex.TILT_SETTLE_S not in ex.sleeps


# --- the transport-time side choice -----------------------------------------


def test_peek_release_tilt_finds_the_upcoming_release():
    actions = [
        {"type": "move_to_keypoint", "params": {"keypoint_label": "pen"}},
        {"type": "grasp", "params": {}},
        {"type": "move_to_keypoint", "params": {"keypoint_label": "bin"}},
        {
            "_bt_retry": True,
            "branch": [{"type": "release", "params": {"tilt_angle": 0.9}}],
        },
    ]
    assert ScoreExecutor._peek_release_tilt(actions) == {"tilt_angle": 0.9}
    assert ScoreExecutor._peek_release_tilt(actions[:2]) is None
    assert ScoreExecutor._peek_release_tilt([]) is None


def test_free_yaw_is_steered_onto_the_negative_side():
    """A bin place has no yaw alignment, so the approach yaw is free -- and a
    pending deep dip picks one that puts tool Y past the boundary with
    margin, still vertical, within the wrist-3 cable limit."""
    ex = make_executor(POS_SIDE)
    ex._pending_release_tilt = {"tilt_angle": 0.90}
    picked = ex._dip_side_orient(list(ex.GRASP_ORIENTATION), yaw_aligned=False)
    # 2e-4 slack: base tool Y is ~1.5 deg off horizontal, so the heading
    # arithmetic hits the margin to within its out-of-plane component.
    assert _tool_y_x(picked) < -np.sin(ex.DIP_SIDE_MARGIN_RAD) + 2e-4
    # The carry stays as vertical as the base is: a world-Z yaw cannot change
    # tilt-from-vertical, and the rig base itself leans ~2.1 deg.
    assert tool_z_tilt_deg(picked) < 3.0
    from spark_real.control.grasp_strategy import base_orientation, measured_yaw_offset

    assert abs(measured_yaw_offset(picked, base_orientation(ex))) <= ex.MAX_YAW_OFFSET + 1e-9


def test_no_pending_or_shallow_tilt_leaves_the_approach_alone():
    ex = make_executor(POS_SIDE)
    orient = list(ex.GRASP_ORIENTATION)
    ex._pending_release_tilt = None
    assert ex._dip_side_orient(orient, yaw_aligned=False) is orient
    ex._pending_release_tilt = {"tilt_angle": 0.55}  # 31.5 deg: inside 40
    assert ex._dip_side_orient(orient, yaw_aligned=False) is orient


def test_an_approach_already_on_the_negative_side_is_untouched():
    ex = make_executor(NEG_SIDE)
    ex._pending_release_tilt = {"tilt_angle": 0.90}
    assert ex._dip_side_orient(list(NEG_SIDE), yaw_aligned=True) == list(NEG_SIDE)


def test_aligned_yaw_trades_only_for_its_180_twin():
    """An aligned yaw near the cable limit: its grasp-symmetric twin is in
    range, flips the pitch side, and keeps the SAME axis line (mod 180)."""
    ex = make_executor(POS_SIDE)
    ex._pending_release_tilt = {"tilt_angle": 0.90}
    aligned = ex._oriented_grasp(np.deg2rad(95.0))
    assert _tool_y_x(aligned) > 0  # on the rig base, +95 is (just) positive-side
    picked = ex._dip_side_orient(aligned, yaw_aligned=True)
    assert _tool_y_x(picked) < 0
    d = abs(_heading(picked, [1, 0, 0]) - _heading(aligned, [1, 0, 0]))
    assert abs(d - np.pi) < 1e-6  # same axis LINE: exactly the 180 twin


def test_aligned_yaw_whose_twin_breaks_the_cable_limit_is_kept():
    """A 30 deg aligned yaw cannot flip (both twins ~150-210 deg of travel):
    the alignment wins, the release clamps to 40 -- a shallow dip beats a
    crooked seat."""
    ex = make_executor(POS_SIDE)
    ex._pending_release_tilt = {"tilt_angle": 0.90}
    aligned = ex._oriented_grasp(np.deg2rad(30.0))
    assert ex._dip_side_orient(aligned, yaw_aligned=True) == aligned


def test_transport_folds_the_dip_side_into_the_approach_and_the_dip_lands():
    """End to end, minus hardware: a plain bin place with a pending 52 deg
    release approaches on the negative-pitch side (no extra motion at the
    release) and the release then commands the FULL 52 deg."""
    bin_det = {
        "label": "bin",
        "position_3d": [-0.78, 0.05, -0.23],
        "confidence": 0.9,
    }
    ex = make_executor(POS_SIDE, detection_map={"bin": bin_det})
    ex._pending_release_tilt = {"tilt_angle": 0.90, "pitch_sign": 1.0}

    result = ex._move_to_keypoint(
        {"keypoint_label": "bin", "offset_z": 0.15}, time.time()
    )
    assert result.success is True

    # The approach itself delivered the negative-pitch side...
    carried = list(ex.robot.get_tcp_pose()[3:6])
    assert _tool_y_x(carried) < 0

    # ...so the release dips the FULL request (the added rotation is exactly
    # 0.90 rad; absolute tilt-from-vertical differs only by the base's ~2 deg
    # lean), with no flip-scale motion.
    release_result = release(ex, 0.90)
    assert release_result.success is True
    assert abs(_rot_between(carried, ex.linear_moves[0][1]) - 0.90) < 0.01
    assert abs(tool_z_tilt_deg(ex.linear_moves[0][1]) - np.rad2deg(0.90)) < 2.5
    for _pos, orient_cmd, _v in ex.linear_moves:
        assert _rot_between(carried, orient_cmd) < np.pi / 2

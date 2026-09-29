"""``release`` must refuse to open the jaws when the transport did not arrive.

Measured on the rig, run 20260804_150848, "pick up the stuffed animal and place
in the bowl" (bt_library/baf971e34606.json)::

    [4/5] move_to_keypoint {'keypoint_label': 'blue bowl', 'offset_z': 0.15}
    executor_motion WARNING: Servo stalled at pos_err=0.0049 ori_err=0.0002;
                             NOT retrying with movel (same unreachable target)
    release: OK - Released at z=0.000
    Verify [birdview] inside(stuffed animal, blue bowl) -> fail
        3d xy_ok=False dz=+5.1cm obb=obb half=(6.9,1.9)cm

The servo gave up short, logged a WARNING, and the sequence opened the jaws
anyway. ``release`` never asked whether the move that was supposed to put the
object over the bowl had actually arrived.

The numbers in this file are that run's, not invented:

  * the bowl's OBB half-extents ``(6.9, 1.9) cm`` come from the verify trace
    (output/real_runs/trace/008_verify.json), i.e. ``obb_minor_m`` 3.8 cm at
    aspect ratio 3.63;
  * the place waypoint ``(-0.8325, -0.0443)`` and the TCP the jaws opened at,
    ``(-0.8283, -0.0497)``, are consecutive samples of the recorded trajectory
    (output/trajectories/20260804_150848/trajectory.npz);
  * the plan's only offset is ``offset_z: 0.15``, so the commanded place XY IS
    the bowl's plan-time centroid XY.

The gate is the CONTAINER'S OWN EXTENT, not the servo's 3 mm convergence
threshold: the verifier judges containment against the container OBB in XY, so
millimetres of Z are harmless and centimetres of XY are not.
"""

from __future__ import annotations

import time

import numpy as np

from spark_real.control import success_verifier
from spark_real.control.score_executor import ScoreExecutor
from spark_real.control.success_predicates import EvalConfig

# --- the measured run ------------------------------------------------------

# Bowl OBB from the verify trace: half=(6.9, 1.9) cm, i.e. a 3.8 cm-wide mask
# stretched 3.63:1. theta=0 puts the MAJOR axis along world X, so the tight
# direction (1.9 cm half-extent) is world Y.
BOWL_XY = (-0.8325, -0.0443)
BOWL_Z = -0.2287  # place waypoint z (-0.0787) minus the plan's offset_z 0.15
BOWL_OBB_MINOR_M = 0.038
BOWL_ASPECT = 6.9 / 1.9
BOWL_HALF_MINOR = BOWL_OBB_MINOR_M / 2.0  # 1.9 cm

# What the servo reported when it gave up.
STALL_POS_ERR = 0.0049
STALL_ORI_ERR = 0.0002

# The XY the jaws actually opened at, relative to the place waypoint.
MEASURED_DRIFT_XY = (-0.8283 - BOWL_XY[0], -0.0497 - BOWL_XY[1])


def bowl_detection(with_extent: bool = True) -> dict:
    det = {
        "label": "blue bowl",
        "position_3d": [BOWL_XY[0], BOWL_XY[1], BOWL_Z],
        "confidence": 0.914,
        "world_major_axis_rad": 0.0,
        "aspect_ratio": BOWL_ASPECT,
    }
    if with_extent:
        det["obb_minor_m"] = BOWL_OBB_MINOR_M
    return det


# --- plant -----------------------------------------------------------------


class FakeArm:
    """Robotiq-shaped driver whose TCP is wherever the test puts it."""

    GRIPPER_TYPE = "robotiq_2f85"
    robot_family = "ur10e"
    SUPPORTS_URSCRIPT = True

    def __init__(self, xyz):
        self.tcp = np.array([*xyz, 2.2214, -2.2214, 0.0], dtype=float)
        self.jaw = 216.7  # the recorded closed-on-plushie jaw position
        self.obj = True
        self.opens = 0

    # pose
    def get_tcp_pose(self):
        return self.tcp.copy()

    def get_observation(self):
        return {"tcp_pose": self.tcp.copy()}

    def set_xyz(self, xyz):
        self.tcp[:3] = np.asarray(xyz, dtype=float)

    # gripper
    def open_gripper(self):
        self.opens += 1
        self.jaw = 2.5
        self.obj = False

    def get_gripper_position(self, publish=True):
        return self.jaw

    def is_object_detected(self, publish=True):
        return self.obj

    def _publish_gripper_state(self, force=False):
        return None


class FakeServo:
    """A servo that lands the arm where the PLANT says, not where it was told.

    That divergence is the whole point: ``last_pos_err`` is the controller's
    residual against its own commanded pose and says nothing about where the
    TCP ended up relative to the bowl.
    """

    def __init__(self, arm, land_xy, exit_reason="stalled", pos_err=STALL_POS_ERR):
        self.arm = arm
        self.land_xy = land_xy
        self.exit_reason = exit_reason
        self.max_vel_linear = 0.15
        self.last_exit = "none"
        self.last_pos_err = float("nan")
        self.last_ori_err = float("nan")
        self.calls = []
        self._pos_err = pos_err

    def move_to_pose(self, target_pose, velocity=None):
        self.calls.append(list(target_pose))
        self.arm.set_xyz([self.land_xy[0], self.land_xy[1], target_pose[2]])
        self.last_exit = self.exit_reason
        self.last_pos_err = self._pos_err
        self.last_ori_err = STALL_ORI_ERR
        return self.exit_reason == "converged"

    def _get_tcp_pose(self):
        return self.arm.get_tcp_pose()

    def abort(self):
        return None


def make_executor(land_xy, exit_reason="stalled", with_extent=True, start=None):
    """Executor holding the plushie, mid-air over the pick, bowl detected."""
    start = start or (-0.7942, 0.4138, 0.0572)  # recorded post-lift TCP
    arm = FakeArm(start)
    ex = ScoreExecutor(
        arm, detection_map={"blue bowl": bowl_detection(with_extent)}, velocity=0.2
    )
    ex._servo = FakeServo(arm, land_xy, exit_reason=exit_reason)
    ex._holding = True
    ex.RELEASE_CONFIRM_TIMEOUT_S = 0.3
    ex.RELEASE_OPEN_SETTLE_S = 0.0

    # The plant, minus every motion the test is not about.
    ex._blend_enabled = lambda: False
    ex._transport_grip_ok = lambda waypoint: True
    ex._abort_sleep = lambda duration, tick=0.05: None
    moves = []

    def fake_move_to(position, orientation, velocity=None):
        moves.append(np.asarray(position, dtype=float).copy())
        arm.set_xyz(np.asarray(position, dtype=float)[:3])

    ex._move_to = fake_move_to
    ex.moves = moves
    return ex


def transport(ex):
    return ex._move_to_keypoint(
        {"keypoint_label": "blue bowl", "offset_z": 0.15}, time.time()
    )


def release(ex):
    return ex._release({"tilt_angle": 0}, time.time())


# --- the failure ------------------------------------------------------------


def test_the_measured_stall_is_invisible_to_release_today():
    """Structural cause, in one assertion: the servo's own residual is not a
    statement about where the TCP is relative to the bowl."""
    ex = make_executor(land_xy=(BOWL_XY[0], BOWL_XY[1] - 0.06))
    transport(ex)
    assert ex._servo.last_pos_err == STALL_POS_ERR  # "converged to 4.9 mm"
    tcp = ex._get_current_position()
    assert abs(tcp[1] - BOWL_XY[1]) > BOWL_HALF_MINOR  # ... 6 cm outside the bowl


def test_a_stalled_transport_that_ends_outside_the_bowl_fails_the_action():
    ex = make_executor(land_xy=(BOWL_XY[0], BOWL_XY[1] - 0.06))
    result = transport(ex)
    assert result.success is False
    assert "blue bowl" in result.message


def test_release_refuses_after_a_transport_that_did_not_arrive():
    """THE run: stalled at pos_err=4.9 mm, TCP laterally outside the bowl OBB."""
    ex = make_executor(land_xy=(BOWL_XY[0], BOWL_XY[1] - 0.06))
    transport(ex)

    result = release(ex)

    assert result.success is False
    assert ex.robot.opens == 0, "the jaws must not open"
    assert ex._holding is True, "the object is still held"
    # Reported, not silently swallowed: the run can no longer pass.
    assert success_verifier.collect_gates(ex)[0]["transport"] is False


# --- the complementary case: do not break working tasks ---------------------


def test_a_transport_that_arrives_releases_exactly_as_today():
    ex = make_executor(land_xy=BOWL_XY, exit_reason="converged")
    assert transport(ex).success is True

    result = release(ex)

    assert result.success is True
    assert ex.robot.opens == 1
    assert ex._holding is False
    assert success_verifier.collect_gates(ex)[0]["transport"] is True


def test_the_measured_5mm_stall_over_the_bowl_still_releases():
    """What the gate would have done on THIS run's recorded geometry.

    The jaws opened 6.8 mm from the place waypoint -- a stall, but well inside
    the bowl's own footprint. A gate that refused here would break a task the
    perception said was on target.
    """
    land = (BOWL_XY[0] + MEASURED_DRIFT_XY[0], BOWL_XY[1] + MEASURED_DRIFT_XY[1])
    assert np.hypot(*MEASURED_DRIFT_XY) < 0.007
    ex = make_executor(land_xy=land)  # exit_reason="stalled", pos_err 4.9 mm

    assert transport(ex).success is True
    result = release(ex)

    assert result.success is True
    assert ex.robot.opens == 1


# --- the tolerance boundary -------------------------------------------------


def test_just_inside_the_tolerance_boundary_releases():
    """Boundary = the bowl's own half-extent plus the gate margin."""
    ex = make_executor(land_xy=(BOWL_XY[0], BOWL_XY[1] - 0.038))
    edge = BOWL_HALF_MINOR + ScoreExecutor.PLACE_ARRIVAL_XY_MARGIN_M
    assert 0.038 < edge

    transport(ex)
    assert release(ex).success is True
    assert ex.robot.opens == 1


def test_just_outside_the_tolerance_boundary_refuses():
    ex = make_executor(land_xy=(BOWL_XY[0], BOWL_XY[1] - 0.040))
    edge = BOWL_HALF_MINOR + ScoreExecutor.PLACE_ARRIVAL_XY_MARGIN_M
    assert 0.040 > edge

    transport(ex)
    assert release(ex).success is False
    assert ex.robot.opens == 0


def test_the_gate_is_looser_than_the_verifier_it_serves():
    """The margin is chosen so the gate cannot refuse a release the verifier
    would have passed: the verifier SHRINKS the OBB by 2 cm (inside_xy_margin_m
    = -0.02), the gate GROWS it by 2 cm."""
    assert ScoreExecutor.PLACE_ARRIVAL_XY_MARGIN_M == -EvalConfig().inside_xy_margin_m


# --- never block on missing information ------------------------------------


def test_a_container_with_no_measured_extent_releases_as_today():
    """No slots and no obb_minor_m: the extent is UNKNOWN, so there is nothing
    to judge arrival against. Refusing here would strand the object on a
    perception gap."""
    ex = make_executor(land_xy=(BOWL_XY[0], BOWL_XY[1] - 0.30), with_extent=False)

    assert transport(ex).success is True
    assert release(ex).success is True
    assert ex.robot.opens == 1


def test_a_recovery_that_fixes_the_approach_clears_the_refusal():
    """The gate is judged at the jaws, not at the transport.

    executor_core answers a failed move_to_keypoint with retract_retry. If the
    refusal were latched on the transport's stored verdict, a recovery that
    actually put the arm back over the bowl would still be refused, and the
    object would be held forever -- the outcome the gate is supposed to be
    better than.
    """
    ex = make_executor(land_xy=(BOWL_XY[0], BOWL_XY[1] - 0.06))
    assert transport(ex).success is False

    ex._move_to([BOWL_XY[0], BOWL_XY[1], -0.0787], ex.GRASP_ORIENTATION)  # recovered
    # The transport's own verdict is still "did not arrive" -- the release must
    # not be reading it.
    assert ex._place_arrival_record["arrived"] is False

    assert release(ex).success is True
    assert ex.robot.opens == 1


def test_drifting_off_the_container_after_a_clean_arrival_still_refuses():
    """Symmetric to the above: an arrival does not license a release later."""
    ex = make_executor(land_xy=BOWL_XY, exit_reason="converged")
    assert transport(ex).success is True

    ex._move_to([BOWL_XY[0], BOWL_XY[1] - 0.06, -0.0787], ex.GRASP_ORIENTATION)

    assert release(ex).success is False
    assert ex.robot.opens == 0


def test_a_release_with_no_preceding_transport_is_untouched():
    ex = make_executor(land_xy=BOWL_XY)
    ex._holding = True
    assert release(ex).success is True
    assert ex.robot.opens == 1

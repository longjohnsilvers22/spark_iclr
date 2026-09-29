"""Offline tests for the _grasp_v2 force verdict on light (silverware) objects.

No robot, no camera: a fake driver replays recorded-scale TCP force readings
through the real GraspMixin._grasp_v2, so the verdict asserted here is the
verdict the rig would reach.

The regression under test: the confirming lift was AND-gated at 0.5 N. A fork
weighs ~0.2 N, so every grasp that Robotiq gObj did not confirm reported EMPTY.
"""

import time

import numpy as np
import pytest

from spark_real.control.executor_grasp import GraspMixin, force_verdict

# Force signatures in newtons at the wrist, relative to the jaws-open baseline.
# clamp = squeeze reaction, lift = same channel after the 2cm confirming lift.
FORK_CLAMP_FZ = -4.0
FORK_LIFT_FZ = -4.2  # +0.2 N of fork weight: BELOW the 0.5 N corroboration bar
AIR_CLAMP_FZ = -0.6  # the documented close-on-air transient
AIR_LIFT_FZ = -0.6


class _FakeRobot:
    """Only what _grasp_v2 touches. No send_velocity -> the servo descent path.

    The wrench is served by PHASE (baseline -> clamp -> lift), advanced by the
    executor's own squeeze and lift, NOT by counting get_tcp_force calls. The
    call-count version broke the moment the baseline stopped being taken after
    a fixed sleep and became a settle condition that samples the channel until
    it is quiet -- which is a legitimate thing for the code to do, so the
    fixture is what had to change. A phase model also matches the rig: the
    wrench does not change because someone read it.
    """

    def __init__(self, forces, gobj=False):
        self._forces = list(forces)
        self._phase = 0
        self._gobj = gobj
        self.calls = []
        self.samples = 0

    def open_gripper(self):
        self.calls.append("open_gripper")

    def advance_phase(self):
        self._phase = min(self._phase + 1, len(self._forces) - 1)

    def get_tcp_force(self):
        self.samples += 1
        f = self._forces[self._phase]
        return np.array([0.0, 0.0, float(f), 0.0, 0.0, 0.0])

    def get_tcp_pose(self):
        return [-0.9, 0.0, 0.10, 2.3038, 2.0802, -0.0048]

    def is_object_detected(self):
        return self._gobj


class _Exec(GraspMixin):
    GRASP_ORIENTATION = [2.3038, 2.0802, -0.0048]
    GRIPPER_FULLY_CLOSED = 250
    GRASP_DEPTH_M = 0.022
    GRASP_LIFT_CHECK_M = 0.02
    GRASP_FORCE_EMPTY_N = 2.0
    GRASP_LIFT_DELTA_MIN_N = 0.5
    TABLE_Z_FLOOR = -0.276
    velocity = 0.25
    _pipeline = None

    def __init__(self, robot, gripper_pos=212):
        self.robot = robot
        self._holding = False
        self._abort = False
        self._gripper_pos = gripper_pos
        self._last_keypoint_label = "fork"
        self._active_grasp_orient = None
        self._active_grasp_strategy = "topdown"
        self._z = 0.10

    # motion / gripper surface, stubbed

    def _check_abort(self):
        pass

    def _abort_sleep(self, duration, tick=0.05):
        pass

    def _get_current_position(self):
        return np.array([-0.9, 0.0, self._z])

    def _move_to(self, pos, orient, **kwargs):
        # The only _move_to on this path is the 2 cm confirming lift.
        if float(pos[2]) > self._z + 1e-9:
            self.robot.advance_phase()  # -> lift wrench
        self._z = float(pos[2])

    def _servo_to(self, pos, orient, **kwargs):
        self._z = float(pos[2])

    def _gripper_squeeze(self, force, speed=50, settle=0.3):
        self.robot.advance_phase()  # -> clamp wrench

    def _robotiq_resqueeze(self, speed=50, force=60, settle=1.0):
        pass

    def _force_gripper_publish(self):
        pass

    def _get_gripper_position(self, publish=True):
        return self._gripper_pos


@pytest.fixture(autouse=True)
def _clean_env(monkeypatch):
    for key in (
        "SPARK_GRASP_FORCE_EMPTY_N",
        "SPARK_GRASP_LIFT_DELTA_MIN_N",
        "SPARK_GRASP_DEPTH_M",
        "SPARK_GRASP_SKIP_LIFT_VERIFY",
        "SPARK_GRASP_VERIFY_SETTLE_S",
    ):
        monkeypatch.delenv(key, raising=False)


def run_grasp(clamp_fz, lift_fz, gobj=False, gripper_pos=212):
    robot = _FakeRobot([0.0, clamp_fz, lift_fz], gobj=gobj)
    ex = _Exec(robot, gripper_pos=gripper_pos)
    return ex, ex._grasp_v2({}, time.time())


# --- BLOCKER 2: a light object still verifies as HELD ------------------------


def test_light_object_verifies_as_held_when_gobj_misses(caplog):
    """Fork: clamp -4.0 N, lift -4.2 N => 0.2 N lift delta, peak 4.2 N.

    0.2 N is far below the 0.5 N corroboration bar and no silverware-class
    object can ever reach it. The clamp reaction (4.2 N >= 2.0 N) is the
    signal, so the grasp is HELD.
    """
    ex, result = run_grasp(FORK_CLAMP_FZ, FORK_LIFT_FZ, gobj=False)
    assert result.success is True, result.message
    assert ex._holding is True

    verdict = ex._last_grasp_verdict
    assert verdict.held is True
    assert verdict.source == "force"
    assert verdict.force_peak_n == pytest.approx(4.2)
    assert verdict.lift_delta_n == pytest.approx(0.2)
    # The lift is recorded as NOT corroborating, and does not change the verdict.
    assert verdict.lift_corroborates is False


def test_closed_on_air_still_reports_not_held():
    """Empty air: clamp -0.6 N, lift -0.6 N => peak 0.6 N < 2.0 N => EMPTY."""
    ex, result = run_grasp(AIR_CLAMP_FZ, AIR_LIFT_FZ, gobj=False)
    assert result.success is False, result.message
    assert ex._holding is False
    assert ex._last_grasp_verdict.held is False
    assert ex._last_grasp_verdict.force_peak_n == pytest.approx(0.6)


def test_flat_plate_veto_survives():
    """Real force but the jaws shut completely: still empty (edge miss)."""
    ex, result = run_grasp(FORK_CLAMP_FZ, FORK_LIFT_FZ, gobj=False, gripper_pos=252)
    assert result.success is False
    assert ex._last_grasp_verdict.source == "jaws_fully_closed"


# --- the pure decision, with the numbers spelled out -------------------------


def test_force_verdict_light_object():
    fv = force_verdict(
        clamp_mag=4.0,
        clamp_dz=FORK_CLAMP_FZ,
        lift_mag=4.2,
        lift_dz=FORK_LIFT_FZ,
        empty_n=2.0,
        lift_delta_min_n=0.5,
    )
    assert fv.peak_n == pytest.approx(4.2)
    assert fv.lift_delta_n == pytest.approx(0.2)
    assert fv.lift_corroborates is False  # 0.2 N < 0.5 N
    assert fv.holding is True  # ... and it cannot veto


def test_force_verdict_heavy_object_corroborates():
    """A 0.9 N object clears the bar, and the verdict is unchanged: still held."""
    fv = force_verdict(4.0, -4.0, 4.9, -4.9, empty_n=2.0, lift_delta_min_n=0.5)
    assert fv.lift_delta_n == pytest.approx(0.9)
    assert fv.lift_corroborates is True
    assert fv.holding is True


def test_force_verdict_closed_on_air():
    fv = force_verdict(0.6, AIR_CLAMP_FZ, 0.6, AIR_LIFT_FZ, 2.0, 0.5)
    assert fv.peak_n == pytest.approx(0.6)
    assert fv.holding is False


# --- config authority over the force threshold -------------------------------


def test_config_force_empty_n_is_authoritative(monkeypatch):
    """No hidden floor: a configured 1.0 N is the threshold that is used."""
    robot = _FakeRobot([0.0, 0.0, 0.0])
    ex = _Exec(robot)
    ex.GRASP_FORCE_EMPTY_N = 1.0
    assert ex._force_empty_n() == pytest.approx(1.0)
    ex.GRASP_FORCE_EMPTY_N = 3.0
    assert ex._force_empty_n() == pytest.approx(3.0)


def test_env_overrides_the_config_force_threshold(monkeypatch):
    monkeypatch.setenv("SPARK_GRASP_FORCE_EMPTY_N", "0.8")
    ex = _Exec(_FakeRobot([0.0, 0.0, 0.0]))
    assert ex._force_empty_n() == pytest.approx(0.8)


def test_lowered_force_threshold_admits_a_weaker_clamp():
    """A 1.5 N clamp is empty at the 2.0 N default and held at a configured 1.0."""
    robot = _FakeRobot([0.0, -1.5, -1.5])
    ex = _Exec(robot)
    assert ex._grasp_v2({}, time.time()).success is False

    robot = _FakeRobot([0.0, -1.5, -1.5])
    ex = _Exec(robot)
    ex.GRASP_FORCE_EMPTY_N = 1.0
    assert ex._grasp_v2({}, time.time()).success is True

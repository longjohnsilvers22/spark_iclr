"""Offline tests for planner-directed grasp strategy dispatch.

No robot, no camera, no SAM3, no Gemini: synthetic detections and a fake
driver only. The real MotionMixin/GraspMixin code is exercised, so the
orientations asserted here are the ones that would be commanded.
"""

import logging

import numpy as np
import pytest
from scipy.spatial.transform import Rotation

from spark_real.control import grasp_strategy as gs
from spark_real.control.executor_grasp import GraspMixin
from spark_real.control.executor_motion import MotionMixin

# UR10e values from control/executor_core.py + configs/ur10e_default.yaml.
GRASP_ORIENTATION = [2.3038, 2.0802, -0.0048]
MAX_YAW_OFFSET = np.deg2rad(90.0)
AR_GATE = 1.6


class _FakeRobot:
    """Only what _nearest_symmetric_yaw touches."""

    def __init__(self, tcp_rotvec=None):
        self._rot = list(tcp_rotvec or GRASP_ORIENTATION)

    def get_tcp_pose(self):
        return [-0.9, 0.0, 0.1] + list(self._rot)


class _Exec(MotionMixin, GraspMixin):
    GRASP_ORIENTATION = GRASP_ORIENTATION
    MAX_YAW_OFFSET = MAX_YAW_OFFSET
    GRASP_YAW_AR_GATE = AR_GATE
    GRIPPER_FULLY_CLOSED = 250
    GRASP_FORCE_EMPTY_N = 1.0
    _pipeline = None
    _holding = False

    def __init__(self, detections=None, tcp_rotvec=None):
        self.robot = self._robot = _FakeRobot(tcp_rotvec)
        self.detection_map = dict(detections or {})
        self._last_keypoint_label = ""
        self._active_grasp_orient = None
        self._active_grasp_strategy = None

    def _robot_family(self):
        return "ur10e"


def det(label="obj", ar=1.0, angle_deg=0.0, conf=0.9, **extra):
    d = {
        "position_3d": [-0.9, 0.0, -0.24],
        "aspect_ratio": ar,
        "orientation_angle": float(np.deg2rad(angle_deg)),
        "confidence": conf,
    }
    d.update(extra)
    return d


def yaw_of(orient):
    """World-Z yaw of a commanded orientation, in degrees, vs GRASP_ORIENTATION."""
    return float(np.rad2deg(gs.measured_yaw_offset(orient, GRASP_ORIENTATION)))


# --- the operator's two misfires -------------------------------------------


def test_plushie_gets_no_yaw():
    """ar 3.14 @ 24% confidence earned -68.9 deg. It must now earn 0.0."""
    ex = _Exec()
    d = det("plushie", ar=3.14, angle_deg=111.1, conf=0.24)
    orient, strategy = gs.resolve_grasp_orientation({}, d, ex)
    assert strategy == "topdown"
    assert abs(yaw_of(orient)) < 1e-6
    assert orient == list(GRASP_ORIENTATION)


def test_round_bowl_gets_no_yaw_when_obb_confidence_is_low():
    """ar 2.06 earned -36.0 deg. A round mask reports low obb_confidence."""
    ex = _Exec()
    d = det("bowl", ar=2.06, angle_deg=144.0, conf=0.72, obb_confidence=0.12)
    orient, strategy = gs.resolve_grasp_orientation({}, d, ex)
    assert strategy == "topdown"
    assert abs(yaw_of(orient)) < 1e-6


# --- the genuine utensil still gets its yaw --------------------------------


def test_real_utensil_keeps_its_yaw():
    ex = _Exec()
    d = det("knife handle", ar=3.0, angle_deg=30.0, conf=0.9, obb_confidence=0.8)
    orient, strategy = gs.resolve_grasp_orientation({}, d, ex)
    assert strategy == "obb"
    assert yaw_of(orient) == pytest.approx(30.0, abs=0.5)


def test_utensil_at_174_deg_reduces_to_minus_6():
    """Symmetric reduction: a jaw grasp at yaw and yaw+180 is the same grasp."""
    ex = _Exec()
    d = det("fork", ar=4.0, angle_deg=174.0, conf=0.9, obb_confidence=0.8)
    orient, strategy = gs.resolve_grasp_orientation({}, d, ex)
    assert strategy == "obb"
    assert yaw_of(orient) == pytest.approx(-6.0, abs=0.5)


# --- planner override, both directions -------------------------------------


def test_planner_topdown_overrides_a_high_ar_detection():
    ex = _Exec()
    d = det("ruler", ar=5.0, angle_deg=45.0, conf=0.95, obb_confidence=0.9)
    orient, strategy = gs.resolve_grasp_orientation(
        {"grasp_strategy": "topdown"}, d, ex
    )
    assert strategy == "topdown"
    assert abs(yaw_of(orient)) < 1e-6


def test_planner_obb_overrides_the_auto_veto():
    """The plushie detection, but the planner insists on obb."""
    ex = _Exec()
    d = det("plushie", ar=3.14, angle_deg=111.1, conf=0.24)
    orient, strategy = gs.resolve_grasp_orientation({"grasp_strategy": "obb"}, d, ex)
    assert strategy == "obb"
    assert yaw_of(orient) == pytest.approx(-68.9, abs=0.5)


def test_planner_yaw_deg_overrides_the_mask():
    ex = _Exec()
    d = det("spatula", ar=4.0, angle_deg=80.0, conf=0.9, obb_confidence=0.9)
    orient, strategy = gs.resolve_grasp_orientation(
        {"grasp_strategy": "obb", "grasp_yaw_deg": 30.0}, d, ex
    )
    assert strategy == "obb"
    assert yaw_of(orient) == pytest.approx(30.0, abs=0.5)


def test_omitting_the_fields_is_todays_behaviour():
    """No strategy on the node -> the old AR gate, unchanged."""
    ex = _Exec()
    below = det("cube", ar=1.2, angle_deg=40.0, conf=0.9)
    above = det("pen", ar=2.5, angle_deg=40.0, conf=0.9)
    assert gs.resolve_grasp_orientation({}, below, ex)[1] == "topdown"
    assert gs.resolve_grasp_orientation({}, above, ex)[1] == "obb"


# --- 6-DOF backends ---------------------------------------------------------


def test_cgn_with_the_env_gate_off_falls_back_loudly(monkeypatch, caplog):
    monkeypatch.delenv("SPARK_GRASP_CGN", raising=False)
    ex = _Exec()
    d = det("mug", ar=3.0, angle_deg=20.0, conf=0.9, obb_confidence=0.8)
    with caplog.at_level(logging.WARNING):
        _, strategy = gs.resolve_grasp_orientation({"grasp_strategy": "cgn"}, d, ex)
    # Falls through to auto (which routes to obb here) and SAYS SO.
    assert strategy == "obb"
    assert any("cgn" in r.message.lower() for r in caplog.records)


def test_se3_gate_reports_the_missing_piece():
    ok, why = gs.se3_gate_open(None)
    assert isinstance(ok, bool) and isinstance(why, str) and why


def test_unknown_strategy_falls_back_to_auto(caplog):
    ex = _Exec()
    d = det("thing", ar=1.1, conf=0.9)
    with caplog.at_level(logging.WARNING):
        _, strategy = gs.resolve_grasp_orientation(
            {"grasp_strategy": "teleport"}, d, ex
        )
    assert strategy == "topdown"
    assert any("unknown grasp_strategy" in r.message for r in caplog.records)


# --- cable limit, measured on the composed rotation -------------------------


@pytest.mark.parametrize("angle_deg", list(range(0, 360, 7)))
def test_every_produced_orientation_is_inside_the_cable_limit(angle_deg):
    ex = _Exec()
    d = det("bar", ar=6.0, angle_deg=angle_deg, conf=0.95, obb_confidence=0.9)
    orient, _ = gs.resolve_grasp_orientation({}, d, ex)
    assert abs(np.deg2rad(yaw_of(orient))) <= MAX_YAW_OFFSET + 1e-6


@pytest.mark.parametrize("yaw_deg", [-179.0, -120.0, -91.0, 91.0, 120.0, 179.0])
def test_planner_yaw_beyond_the_limit_is_reduced_not_commanded_raw(yaw_deg):
    ex = _Exec()
    d = det("bar", ar=6.0, angle_deg=0.0, conf=0.95, obb_confidence=0.9)
    orient, _ = gs.resolve_grasp_orientation(
        {"grasp_strategy": "obb", "grasp_yaw_deg": yaw_deg}, d, ex
    )
    assert abs(np.deg2rad(yaw_of(orient))) <= MAX_YAW_OFFSET + 1e-6


def test_a_tighter_cable_limit_refuses_rather_than_clipping():
    """A clipped yaw is a crooked grasp; a defined top-down is better."""
    ex = _Exec()
    ex.MAX_YAW_OFFSET = np.deg2rad(30.0)
    d = det("bar", ar=6.0, angle_deg=80.0, conf=0.95, obb_confidence=0.9)
    orient, strategy = gs.resolve_grasp_orientation({}, d, ex)
    assert strategy == "topdown"
    assert abs(np.deg2rad(yaw_of(orient))) <= np.deg2rad(30.0) + 1e-6

    # ...but a yaw INSIDE the tighter limit is still honoured.
    d20 = det("bar", ar=6.0, angle_deg=20.0, conf=0.95, obb_confidence=0.9)
    orient, strategy = gs.resolve_grasp_orientation({}, d20, ex)
    assert strategy == "obb"
    assert yaw_of(orient) == pytest.approx(20.0, abs=0.5)


@pytest.mark.parametrize("angle_deg", [0.0, 30.0, 89.0, 111.1, 174.0])
def test_tool_z_stays_vertical(angle_deg):
    """A yaw about world Z must not tilt the tool off straight down."""
    ex = _Exec()
    d = det("bar", ar=6.0, angle_deg=angle_deg, conf=0.95, obb_confidence=0.9)
    orient, _ = gs.resolve_grasp_orientation({}, d, ex)
    assert gs.tool_z_tilt_deg(orient) < gs.TOPDOWN_TILT_TOL_DEG


def test_base_orientation_is_itself_vertical():
    assert gs.tool_z_tilt_deg(GRASP_ORIENTATION) < gs.TOPDOWN_TILT_TOL_DEG


def test_closing_axis_is_perpendicular_to_the_major_axis():
    """The jaws must close ACROSS the object, not along it."""
    ex = _Exec()
    d = det("knife", ar=5.0, angle_deg=25.0, conf=0.95, obb_confidence=0.9)
    orient, _ = gs.resolve_grasp_orientation({}, d, ex)
    R = Rotation.from_rotvec(orient).as_matrix()
    R0 = Rotation.from_rotvec(GRASP_ORIENTATION).as_matrix()
    # Whatever the base closing axis is, the yaw rotates it by exactly 25 deg.
    for col in (0, 1):
        a, b = R0[:2, col], R[:2, col]
        if np.linalg.norm(a) > 0.5:
            turned = np.rad2deg(
                np.arctan2(b[1], b[0]) - np.arctan2(a[1], a[0])
            )
            assert ((turned + 180) % 360) - 180 == pytest.approx(25.0, abs=0.5)


# --- _active_grasp_orient must not leak between objects ---------------------


def test_active_grasp_orient_is_cleared_between_grasps():
    ex = _Exec(
        {
            "knife": det("knife", ar=5.0, angle_deg=60.0, conf=0.95, obb_confidence=0.9),
            "cube": det("cube", ar=1.1, angle_deg=60.0, conf=0.95),
        }
    )
    ex._last_keypoint_label = "knife"
    assert ex._resolve_grasp_node_strategy({}) == "obb"
    first = list(ex._active_grasp_orient)
    assert yaw_of(first) == pytest.approx(60.0, abs=0.5)

    # A grasp reached WITHOUT a fresh approach must not inherit that yaw.
    ex._last_keypoint_label = "cube"
    assert ex._resolve_grasp_node_strategy({}) == "topdown"
    assert ex._active_grasp_orient is None


def test_grasp_node_params_beat_the_approach():
    ex = _Exec({"knife": det("knife", ar=5.0, angle_deg=60.0, conf=0.95)})
    ex._last_keypoint_label = "knife"
    ex._active_grasp_strategy = "obb"
    ex._active_grasp_orient = list(GRASP_ORIENTATION)
    assert ex._resolve_grasp_node_strategy({"grasp_strategy": "topdown"}) == "topdown"
    assert ex._active_grasp_orient is None


def test_low_quality_mask_forces_topdown():
    ex = _Exec()
    d = det("blob", ar=4.0, angle_deg=50.0, conf=0.99, low_quality=True)
    assert gs.resolve_grasp_orientation({}, d, ex)[1] == "topdown"


def test_missing_obb_confidence_is_unknown_not_zero():
    """Perception has not shipped the field yet; that must not veto everything."""
    ex = _Exec()
    d = det("fork", ar=3.0, angle_deg=10.0, conf=0.9)
    assert "obb_confidence" not in d
    assert gs.resolve_grasp_orientation({}, d, ex)[1] == "obb"


def test_no_detection_is_topdown_not_a_crash():
    ex = _Exec()
    orient, strategy = gs.resolve_grasp_orientation({}, None, ex)
    assert strategy == "topdown" and orient == list(GRASP_ORIENTATION)


def test_env_kill_switch_forces_topdown(monkeypatch):
    monkeypatch.setenv("SPARK_GRASP_STRATEGY", "topdown")
    ex = _Exec()
    d = det("knife", ar=5.0, angle_deg=40.0, conf=0.95, obb_confidence=0.9)
    assert gs.resolve_grasp_orientation({"grasp_strategy": "obb"}, d, ex)[1] == "topdown"


def test_yaw_min_conf_is_config_driven(monkeypatch):
    ex = _Exec()
    d = det("plushie", ar=3.14, angle_deg=111.1, conf=0.24)
    monkeypatch.setenv("SPARK_YAW_MIN_CONF", "0.0")
    assert gs.resolve_grasp_orientation({}, d, ex)[1] == "obb"


# --- force-verify thresholds ------------------------------------------------


def test_config_force_empty_n_is_authoritative(monkeypatch):
    """No floor: grasp.force_empty_n is used as configured, up or down."""
    monkeypatch.delenv("SPARK_GRASP_FORCE_EMPTY_N", raising=False)
    ex = _Exec()
    ex.GRASP_FORCE_EMPTY_N = 1.0
    assert ex._force_empty_n() == pytest.approx(1.0)
    ex.GRASP_FORCE_EMPTY_N = 3.0
    assert ex._force_empty_n() == pytest.approx(3.0)


def test_env_overrides_the_configured_force_threshold(monkeypatch):
    monkeypatch.setenv("SPARK_GRASP_FORCE_EMPTY_N", "0.8")
    ex = _Exec()
    assert ex._force_empty_n() == pytest.approx(0.8)

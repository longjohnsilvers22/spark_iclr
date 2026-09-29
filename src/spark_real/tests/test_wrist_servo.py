"""Offline tests for the wrist IBVS core. No robot, no camera, no GPU."""

import numpy as np
import pytest

from spark_real.control.wrist_servo import (
    ServoOutcome,
    WristIBVS,
    pixel_error,
)

# Wrist D435i colour intrinsics, used here only as realistic test numbers.
FX, FY = 608.26, 607.97
CX, CY = 327.96, 245.63


class FakeCamera:
    """
    Pinhole wrist camera whose object stays put while the camera moves.

    Applying a tool-frame lateral velocity for dt shifts the object's pixel by
    du = -(fx / Z) * vx * dt, the same interaction model the servo inverts --
    but at the TRUE depth, which the servo does not know.
    """

    def __init__(self, u, v, z_true=0.20, dt=0.1):
        self.u = float(u)
        self.v = float(v)
        self.z_true = float(z_true)
        self.dt = float(dt)
        self.frames = 0

    def read(self):
        self.frames += 1
        return (self.u, self.v)

    def apply(self, velocity):
        self.u -= FX * velocity[0] * self.dt / self.z_true
        self.v -= FY * velocity[1] * self.dt / self.z_true


def make_servo(**kwargs):
    kwargs.setdefault("standoff_m", 0.15)
    kwargs.setdefault("rate_hz", 10.0)
    return WristIBVS(fx=FX, fy=FY, **kwargs)


def test_pixel_error_signs():
    e_u, e_v, err = pixel_error((110.0, 190.0), (100.0, 200.0))
    assert e_u == pytest.approx(10.0)
    assert e_v == pytest.approx(-10.0)
    assert err == pytest.approx(np.hypot(10.0, 10.0))


def test_plus_u_error_gives_plus_x_velocity_only():
    servo = make_servo()
    vel, converged, err = servo.step((CX + 60.0, CY), (CX, CY))
    assert not converged
    assert err == pytest.approx(60.0)
    assert vel[0] > 0.0  # +u error -> move camera +x to pull the object left
    assert vel[1] == pytest.approx(0.0)
    assert vel[2] == pytest.approx(0.0)
    assert np.allclose(vel[3:], 0.0)  # no angular authority


def test_plus_v_error_gives_plus_y_velocity_only():
    servo = make_servo()
    vel, _, _ = servo.step((CX, CY + 40.0), (CX, CY))
    assert vel[1] > 0.0
    assert vel[0] == pytest.approx(0.0)


def test_sign_flips_with_error():
    servo = make_servo()
    pos, _, _ = servo.step((CX + 50.0, CY + 50.0), (CX, CY))
    neg, _, _ = servo.step((CX - 50.0, CY - 50.0), (CX, CY))
    assert np.allclose(pos[:3], -neg[:3])


def test_zero_error_is_converged_and_zero_velocity():
    servo = make_servo()
    vel, converged, err = servo.step((CX, CY), (CX, CY))
    assert converged
    assert err == pytest.approx(0.0)
    assert np.allclose(vel, 0.0)


def test_inside_deadband_is_converged():
    servo = make_servo(deadband_px=6.0)
    vel, converged, err = servo.step((CX + 4.0, CY + 3.0), (CX, CY))
    assert err == pytest.approx(5.0)
    assert converged
    assert np.allclose(vel, 0.0)
    # Just outside the deadband it must command motion again.
    _, converged_out, _ = servo.step((CX + 8.0, CY), (CX, CY))
    assert not converged_out


def test_velocity_clamp_respected_for_huge_error():
    servo = make_servo(kp=50.0, max_linear_vel=0.05)
    vel, _, _ = servo.step((CX + 3000.0, CY + 3000.0), (CX, CY))
    assert np.linalg.norm(vel[:3]) <= 0.05 + 1e-9
    assert np.linalg.norm(vel[:3]) == pytest.approx(0.05, abs=1e-9)


def test_rotated_mount_maps_into_tool_frame():
    # Camera rotated +90 deg about Z relative to the tool: cam +x -> tool +y.
    R = np.array([[0.0, -1.0, 0.0], [1.0, 0.0, 0.0], [0.0, 0.0, 1.0]])
    servo = make_servo(rotation_cam_to_tool=R)
    vel, _, _ = servo.step((CX + 60.0, CY), (CX, CY))
    assert vel[1] > 0.0
    assert vel[0] == pytest.approx(0.0)


def test_from_calibration_reads_intrinsics():
    from spark_real.calibration.model import CameraCalibration

    cal = CameraCalibration(name="wrist", width=640, height=480, fx=FX, fy=FY, cx=CX, cy=CY)
    servo = WristIBVS.from_calibration(cal)
    assert servo.fx == pytest.approx(FX)
    assert servo.fy == pytest.approx(FY)


def test_rejects_bad_parameters():
    with pytest.raises(ValueError):
        WristIBVS(fx=0.0, fy=FY)
    with pytest.raises(ValueError):
        make_servo(kp=0.0)
    with pytest.raises(ValueError):
        make_servo(standoff_m=-0.1)
    with pytest.raises(ValueError):
        make_servo(rotation_cam_to_tool=np.eye(2))


def _run_against_camera(servo, cam, target, **kwargs):
    return servo.run(
        observe=cam.read,
        command=cam.apply,
        target=target,
        sleep=lambda _dt: None,
        now=_FakeClock(cam),
        **kwargs,
    )


class _FakeClock:
    """Monotonic clock advancing one dt per read, driven by the fake camera."""

    def __init__(self, cam):
        self.cam = cam

    def __call__(self):
        return self.cam.frames * self.cam.dt


@pytest.mark.parametrize(
    "offset",
    [(80.0, 0.0), (0.0, -60.0), (120.0, 95.0), (-45.0, 130.0)],
)
def test_closed_loop_converges_within_bounded_steps(offset):
    target = (CX, CY)
    cam = FakeCamera(CX + offset[0], CY + offset[1], z_true=0.20, dt=0.1)
    servo = make_servo(deadband_px=6.0, max_iters=60, timeout_s=1e6)
    result = _run_against_camera(servo, cam, target)

    assert result.outcome is ServoOutcome.CONVERGED
    assert result.converged
    assert result.err_px <= servo.deadband_px
    assert result.iterations < servo.max_iters
    # Error must shrink monotonically -- no overshoot, no oscillation.
    assert all(b < a + 1e-9 for a, b in zip(result.err_history, result.err_history[1:]))
    assert np.hypot(cam.u - target[0], cam.v - target[1]) <= servo.deadband_px


def test_closed_loop_converges_despite_wrong_depth_assumption():
    # Servo assumes 0.15 m; the object is really at 0.35 m. Slower, still exact.
    cam = FakeCamera(CX + 150.0, CY - 110.0, z_true=0.35, dt=0.1)
    servo = make_servo(max_iters=200, timeout_s=1e6)
    result = _run_against_camera(servo, cam, (CX, CY))
    assert result.converged
    assert result.err_px <= servo.deadband_px


def test_non_convergence_is_reported_not_looped():
    # One iteration is nowhere near enough for a 300 px error.
    cam = FakeCamera(CX + 300.0, CY, z_true=0.20, dt=0.1)
    servo = make_servo(max_iters=1, timeout_s=1e6)
    result = _run_against_camera(servo, cam, (CX, CY))
    assert result.outcome is ServoOutcome.MAX_ITERATIONS
    assert not result.converged
    assert result.iterations == 1
    assert result.err_px > servo.deadband_px


def test_timeout_is_reported():
    cam = FakeCamera(CX + 300.0, CY, z_true=0.20, dt=0.1)
    servo = make_servo(max_iters=10_000, timeout_s=0.25)
    result = _run_against_camera(servo, cam, (CX, CY))
    assert result.outcome is ServoOutcome.TIMEOUT
    assert not result.converged


def test_lost_target_is_reported():
    sent = []
    servo = make_servo(max_lost_frames=3)
    result = servo.run(
        observe=lambda: None,
        command=sent.append,
        target=(CX, CY),
        sleep=lambda _dt: None,
        now=lambda: 0.0,
    )
    assert result.outcome is ServoOutcome.TARGET_LOST
    assert not result.converged
    assert all(np.allclose(v, 0.0) for v in sent)


def test_abort_is_reported_and_motion_stopped():
    cam = FakeCamera(CX + 300.0, CY, z_true=0.20, dt=0.1)
    servo = make_servo(max_iters=100, timeout_s=1e6)
    result = _run_against_camera(servo, cam, (CX, CY), should_abort=lambda: True)
    assert result.outcome is ServoOutcome.ABORTED
    assert result.iterations == 0


def test_loop_always_sends_a_final_zero_velocity():
    sent = []
    cam = FakeCamera(CX + 90.0, CY + 90.0, z_true=0.20, dt=0.1)

    def command(v):
        sent.append(np.asarray(v).copy())
        cam.apply(v)

    servo = make_servo(max_iters=60, timeout_s=1e6)
    result = servo.run(
        observe=cam.read,
        command=command,
        target=(CX, CY),
        sleep=lambda _dt: None,
        now=_FakeClock(cam),
    )
    assert result.converged
    assert np.allclose(sent[-1], 0.0)


def test_callable_target_is_re_read_each_iteration():
    # Target pixel drifts (as a live tag detector's would); loop still lands.
    cam = FakeCamera(CX + 100.0, CY, z_true=0.20, dt=0.1)
    calls = {"n": 0}

    def moving_target():
        calls["n"] += 1
        return (CX + min(calls["n"], 20) * 0.5, CY)

    servo = make_servo(max_iters=120, timeout_s=1e6)
    result = _run_against_camera(servo, cam, moving_target)
    assert result.converged
    assert calls["n"] > 1

"""The servo's approach must not decay to a crawl near the target.

_compute_velocity is pure proportional control (v = kp * err), so commanded
speed shrinks with the remaining error and the last centimetre takes longer
than the first ten. Measured on the plushie-into-bowl run
(/data/spark_episodes/.../episode_0002), descending over the bowl:

    t=13.07  z=+0.070  vz=-0.385     12.7 cm in 0.4 s
    t=13.40  z=-0.052  vz=-0.037      2.2 cm in 2.2 s
    t=15.53  z=-0.074  vz=-0.001

Five times longer for a sixth of the distance. The operator sees a fast drop
followed by a visible crawl and reads it as jitter.

The fix is a floor on the commanded approach speed while outside the arrival
tolerance, which turns the exponential tail into a linear one. The floor must
NOT apply once inside the tolerance -- that would command motion at the target
and buzz around it, which is the actual jitter this is often mistaken for.
"""

import numpy as np
import pytest

from spark_real.control.cartesian_servo import CartesianServo


@pytest.fixture
def servo():
    s = CartesianServo.__new__(CartesianServo)
    s.kp_pos, s.kp_ori = 2.0, 1.5
    s.kd_pos, s.kd_ori = 0.5, 0.4
    s.max_vel_linear, s.max_vel_angular = 0.15, 0.6
    s.pos_threshold, s.ori_threshold = 0.003, 0.05
    s.min_vel_linear = getattr(CartesianServo, "MIN_APPROACH_VEL_LINEAR", 0.03)
    s._prev_vel = np.zeros(6)
    return s


def _pose(z):
    return np.array([-0.9, 0.0, z, 2.3, 2.08, 0.0])


def _speed(servo, err_m):
    """Commanded linear speed with `err_m` of pure -Z error remaining."""
    v = servo._compute_velocity(_pose(err_m), _pose(0.0))
    return float(np.linalg.norm(v[:3]))


def test_far_approach_is_unchanged(servo):
    """The floor must not alter the fast part of the move."""
    assert _speed(servo, 0.127) == pytest.approx(0.15, abs=1e-6)  # at the cap


def test_near_target_does_not_crawl(servo):
    """The regression: 2.2 cm out, the old law commands 0.044 m/s."""
    v = _speed(servo, 0.022)
    assert v >= 0.03 - 1e-9, f"still crawling at 2.2cm: {v:.4f} m/s"


def test_last_millimetres_do_not_crawl(servo):
    """Just outside tolerance is where the old law was slowest."""
    v = _speed(servo, 0.004)
    assert v >= 0.03 - 1e-9, f"still crawling at 4mm: {v:.4f} m/s"


def test_inside_tolerance_is_not_floored(servo):
    """
    Critical: flooring INSIDE the arrival tolerance would command motion at
    the target forever and produce real buzz. Must decay to ~0 there.
    """
    v = _speed(servo, 0.001)
    assert v < 0.01, f"floor applied inside tolerance -> buzz: {v:.4f} m/s"


def test_zero_error_commands_zero(servo):
    assert _speed(servo, 0.0) == pytest.approx(0.0, abs=1e-9)


def test_approach_time_from_2cm_is_bounded(servo):
    """
    Integrate the control law and require the 2.2 cm -> 3 mm descent to take
    under 1.0 s. The measured run took 2.2 s.
    """
    err, t, dt = 0.022, 0.0, 0.008
    while err > servo.pos_threshold and t < 5.0:
        err -= _speed(servo, err) * dt
        t += dt
    assert t < 1.0, f"approach still takes {t:.2f}s"

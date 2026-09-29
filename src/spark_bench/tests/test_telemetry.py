"""
Unit tests for the gripper-telemetry grasp-outcome classifier
(spark_bench.libero_pro.telemetry) on synthetic aperture traces.

Pure CPU: no env, no GPU.
"""
import numpy as np
import pytest

from spark_bench.libero_pro.telemetry import (
    FRANKA_OPEN_APERTURE,
    GraspOutcome,
    classify_grasp_outcome,
    read_aperture,
)


def _ramp(a: float, b: float, n: int) -> list:
    return list(np.linspace(a, b, n))


def _hold(v: float, n: int) -> list:
    return [v] * n


def test_secured_object_stops_fingers_early():
    # Close from open, stop on a 3 cm object, hold stable.
    trace = _ramp(FRANKA_OPEN_APERTURE, 0.030, 60) + _hold(0.030, 120)
    cls = classify_grasp_outcome(trace)
    assert cls.outcome == GraspOutcome.SECURED
    assert cls.final_aperture == pytest.approx(0.030, abs=1e-3)
    assert cls.plateau_aperture == pytest.approx(0.030, abs=1e-3)


def test_empty_close_reaches_commanded_width():
    # Close from open all the way to ~0: jaws met nothing.
    trace = _ramp(FRANKA_OPEN_APERTURE, 0.0, 100) + _hold(0.0, 100)
    cls = classify_grasp_outcome(trace)
    assert cls.outcome == GraspOutcome.EMPTY_CLOSE


def test_slip_full_collapse_after_contact():
    # Contact plateau at 3 cm, then the object escapes and the jaws
    # close the rest of the way.
    trace = (_ramp(FRANKA_OPEN_APERTURE, 0.030, 50) + _hold(0.030, 80)
             + _ramp(0.030, 0.0, 40) + _hold(0.0, 40))
    cls = classify_grasp_outcome(trace)
    assert cls.outcome == GraspOutcome.SLIP


def test_slip_partial_decay_still_flags():
    # Plateau 3 cm decaying to 2.2 cm (> slip_drop=6 mm): partial slip.
    trace = (_ramp(FRANKA_OPEN_APERTURE, 0.030, 50) + _hold(0.030, 80)
             + _ramp(0.030, 0.022, 40) + _hold(0.022, 40))
    cls = classify_grasp_outcome(trace)
    assert cls.outcome == GraspOutcome.SLIP


def test_small_settling_noise_is_not_slip():
    # 2 mm of compliance settle stays under slip_drop -> SECURED.
    trace = (_ramp(FRANKA_OPEN_APERTURE, 0.030, 50) + _hold(0.030, 80)
             + _ramp(0.030, 0.028, 40) + _hold(0.028, 40))
    cls = classify_grasp_outcome(trace)
    assert cls.outcome == GraspOutcome.SECURED


def test_noisy_secured_trace():
    rng = np.random.default_rng(0)
    base = _ramp(FRANKA_OPEN_APERTURE, 0.025, 60) + _hold(0.025, 140)
    trace = list(np.asarray(base) + rng.normal(0, 1e-4, len(base)))
    cls = classify_grasp_outcome(trace)
    assert cls.outcome == GraspOutcome.SECURED


def test_empty_trace_is_unknown():
    cls = classify_grasp_outcome([])
    assert cls.outcome == GraspOutcome.UNKNOWN
    assert cls.n_samples == 0


def test_fingers_never_moved_is_unknown():
    # Aperture pinned open (e.g. actuator failure): inconclusive, must
    # never trigger a retry.
    cls = classify_grasp_outcome(_hold(FRANKA_OPEN_APERTURE, 200))
    assert cls.outcome == GraspOutcome.UNKNOWN


def test_commanded_partial_close():
    # A close commanded to 2 cm that reaches 2 cm exactly = empty
    # relative to the command; an early stop at 3.5 cm = secured.
    partial_empty = _ramp(FRANKA_OPEN_APERTURE, 0.020, 80) + _hold(0.020, 80)
    cls = classify_grasp_outcome(partial_empty, commanded_aperture=0.020)
    assert cls.outcome == GraspOutcome.EMPTY_CLOSE

    early_stop = _ramp(FRANKA_OPEN_APERTURE, 0.035, 80) + _hold(0.035, 80)
    cls2 = classify_grasp_outcome(early_stop, commanded_aperture=0.020)
    assert cls2.outcome == GraspOutcome.SECURED


def test_to_meta_is_json_safe():
    import json
    trace = _ramp(FRANKA_OPEN_APERTURE, 0.030, 60) + _hold(0.030, 120)
    meta = classify_grasp_outcome(trace).to_meta()
    json.dumps(meta)
    assert meta['outcome'] == 'secured'
    assert meta['n_samples'] == 180


def test_read_aperture_mirrored_joints():
    class _D:
        qpos = np.array([0.5, 0.02, -0.018])
    assert read_aperture(_D(), [1, 2]) == pytest.approx(0.038)
    assert read_aperture(_D(), []) is None
    assert read_aperture(_D(), [1]) is None

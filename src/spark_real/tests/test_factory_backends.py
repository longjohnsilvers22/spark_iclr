"""
make_robot_driver builds, or raises RuntimeError for a missing backend, for
every family and Franka backend. Never TypeError: a kwarg the driver does
not take, or a value landing in the wrong positional slot, is a factory bug
(SMOKE-01/02). TEST-NET IPs only; nothing here connects.

Usage:
    PYTHONPATH=src python -m pytest src/spark_real/tests/test_factory_backends.py -q
"""

from __future__ import annotations

import pytest

from spark_real.robots.driver_protocol import RobotDriver
from spark_real.robots.factory import KNOWN_FAMILIES, make_robot_driver

IP = "192.0.2.1"

# (family, SPARK_FRANKA_BACKEND or None, kwargs the real caller passes).
# pipeline_init.py passes gripper=/collision= for the franka family only;
# run.py passes (family, ip, frequency) for every family.
CASES = [
    ("ur10e", None, {}),
    ("g1", None, {}),
    ("bimanual_franka", None, {}),
    ("franka", "franky", {"gripper": {}, "collision": {}}),
    ("franka", "torque", {"gripper": {}, "collision": {}}),
    ("franka", "bamboo", {"gripper": {}, "collision": {}}),
]


def _build(monkeypatch, family, backend, **kwargs):
    if backend is None:
        monkeypatch.delenv("SPARK_FRANKA_BACKEND", raising=False)
    else:
        monkeypatch.setenv("SPARK_FRANKA_BACKEND", backend)
    try:
        return make_robot_driver(family, IP, **kwargs)
    except RuntimeError as exc:
        # The driver's own "backend not installed" guard; the only allowed
        # failure. TypeError and everything else propagate and fail the test.
        assert "install" in str(exc).lower(), exc
        return None


@pytest.mark.parametrize("family,backend,kwargs", CASES)
def test_builds_or_raises_runtime_error(monkeypatch, family, backend, kwargs):
    driver = _build(monkeypatch, family, backend, **kwargs)
    if driver is None:
        pytest.skip(f"{family}/{backend or '-'} backend not installed here")
    assert isinstance(driver, RobotDriver)


@pytest.mark.parametrize("family,backend,kwargs", CASES)
def test_frequency_kwarg_is_accepted(monkeypatch, family, backend, kwargs):
    # run.py passes frequency positionally; it must not shift into another slot.
    _build(monkeypatch, family, backend, frequency=1000.0, **kwargs)


def test_bamboo_port_default_and_override(monkeypatch):
    monkeypatch.setenv("SPARK_FRANKA_BACKEND", "bamboo")
    d = make_robot_driver("franka", IP, frequency=1000.0, gripper={}, collision={})
    assert d._port == 5555, d._port  # SMOKE-02: freq used to land here
    assert d._ip == IP
    assert make_robot_driver("franka", IP, port=6000)._port == 6000


def test_known_families_are_all_covered():
    assert set(KNOWN_FAMILIES) == {c[0] for c in CASES}


def test_unknown_family_is_value_error():
    with pytest.raises(ValueError):
        make_robot_driver("kuka", IP)

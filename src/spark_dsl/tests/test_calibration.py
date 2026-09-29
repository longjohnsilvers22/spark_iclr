"""
Tests for spark_dsl.calibration.
"""
from __future__ import annotations

import pathlib
import sys

import pytest

_SRC = pathlib.Path(__file__).resolve().parents[2]
if str(_SRC) not in sys.path:
    sys.path.insert(0, str(_SRC))

from spark_dsl import calibration  # noqa: E402


def test_get_returns_empty_for_unknown_pair() -> None:
    db = calibration.CalibrationDB()
    assert db.get("mug", "flat_table") == {}
    assert db.surfaces_for_object("mug") == []


def test_running_mean_matches_arithmetic_mean() -> None:
    db = calibration.CalibrationDB()
    samples = [0.010, 0.014, 0.012, 0.011, 0.013]
    for s in samples:
        db.update(object="mug", surface="flat_table",
                  observed_offset={"z": s}, success=True)
    expected = sum(samples) / len(samples)
    got = db.get("mug", "flat_table")["z"]
    assert got == pytest.approx(expected, rel=0, abs=1e-9)

    entry = db.get_entry("mug", "flat_table")
    assert entry is not None
    assert entry.n_obs == 5
    assert entry.n_success == 5
    assert 0.0 < entry.confidence(kappa=5) < 1.0
    # Welford std should be close to numpy std for these samples
    import statistics
    expected_std = statistics.stdev(samples)
    assert entry.stddev()["z"] == pytest.approx(expected_std, rel=1e-6)


def test_failure_counts_against_success_rate_but_still_updates_mean() -> None:
    db = calibration.CalibrationDB()
    db.update(object="mug", surface="flat_table",
              observed_offset={"z": 0.01}, success=True)
    db.update(object="mug", surface="flat_table",
              observed_offset={"z": 0.03}, success=False)
    entry = db.get_entry("mug", "flat_table")
    assert entry.n_obs == 2
    assert entry.n_success == 1
    assert entry.success_rate() == 0.5
    assert entry.offset["z"] == pytest.approx(0.02, abs=1e-9)


def test_round_trip_persistence(tmp_path: pathlib.Path) -> None:
    path = tmp_path / "calib.yaml"
    db = calibration.CalibrationDB(path)
    for s in (0.01, 0.012, 0.014):
        db.update(object="mug", surface="flat_table",
                  observed_offset={"z": s}, success=True)
    db.update(object="bowl", surface="counter",
              observed_offset={"x": 0.001, "z": 0.020}, success=True)
    db.save()

    db2 = calibration.CalibrationDB.load(path)
    assert len(db2) == 2
    assert sorted(db2.surfaces_for_object("mug")) == ["flat_table"]
    e = db2.get_entry("mug", "flat_table")
    assert e.n_obs == 3
    assert e.offset["z"] == pytest.approx(0.012, abs=1e-9)


def test_dump_pretty_contains_rows(tmp_path: pathlib.Path) -> None:
    db = calibration.CalibrationDB()
    db.update(object="mug", surface="flat_table",
              observed_offset={"z": 0.012}, success=True)
    db.update(object="bowl", surface="counter",
              observed_offset={"x": 0.001, "z": 0.020}, success=False)
    text = db.dump_pretty()
    assert "mug" in text and "bowl" in text
    assert "flat_table" in text and "counter" in text
    # confidence column header should appear
    assert "conf" in text
    # mm conversion: 12.0 should show up for z=0.012
    assert "+12.0" in text


def test_kappa_smoothing_pushes_confidence_toward_one() -> None:
    db = calibration.CalibrationDB(kappa=5)
    for _ in range(45):  # n_obs=45 -> conf = 45/50 = 0.9
        db.update(object="mug", surface="flat_table",
                  observed_offset={"z": 0.012}, success=True)
    e = db.get_entry("mug", "flat_table")
    assert e.confidence(kappa=5) == pytest.approx(0.9, abs=1e-3)

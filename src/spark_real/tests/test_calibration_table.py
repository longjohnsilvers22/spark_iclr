"""Offline test for the table-plane calibration loader/saver. No robot."""

import json

import spark_real.calibration_table as ct
from spark_real.calibration_table import load_table_plane, save_table_plane


def test_load_returns_floor_from_surface_plus_margin(tmp_path, monkeypatch):
    path = tmp_path / "table_plane.json"
    monkeypatch.setattr(ct, "TABLE_PLANE_PATH", path)

    save_table_plane("ur10e", -0.2801, 0.003)
    data = load_table_plane("ur10e")
    assert data is not None
    floor = data["surface_z"] + data["safety_margin_m"]
    assert abs(floor - (-0.2771)) < 1e-9
    # Floor sits ABOVE (less negative than) the surface.
    assert floor > data["surface_z"]


def test_missing_file_returns_none(tmp_path, monkeypatch):
    monkeypatch.setattr(ct, "TABLE_PLANE_PATH", tmp_path / "nope.json")
    assert load_table_plane("ur10e") is None


def test_wrong_family_returns_none(tmp_path, monkeypatch):
    path = tmp_path / "table_plane.json"
    monkeypatch.setattr(ct, "TABLE_PLANE_PATH", path)
    save_table_plane("ur10e", -0.2801, 0.003)
    assert load_table_plane("franka") is None


def test_shipped_ur10e_cal_resolves_expected_floor():
    # output/calibrations/table_plane.json is machine-local (gitignored),
    # recorded via POST /api/calibrate_table; absent on a fresh checkout.
    import pytest

    data = load_table_plane("ur10e")
    if data is None:
        pytest.skip("no local table_plane.json recorded on this machine")
    floor = data["surface_z"] + data["safety_margin_m"]
    assert floor > data["surface_z"]


if __name__ == "__main__":
    import tempfile
    from pathlib import Path

    tmp = Path(tempfile.mkdtemp())
    orig = ct.TABLE_PLANE_PATH

    ct.TABLE_PLANE_PATH = tmp / "table_plane.json"
    p = save_table_plane("ur10e", -0.2801, 0.003)
    d = load_table_plane("ur10e")
    floor = d["surface_z"] + d["safety_margin_m"]
    print("saved:", p)
    print("loaded:", json.dumps(d))
    print("floor = surface_z + margin =", floor)
    assert abs(floor - (-0.2771)) < 1e-9, floor
    assert floor > d["surface_z"]

    ct.TABLE_PLANE_PATH = tmp / "missing.json"
    assert load_table_plane("ur10e") is None
    print("missing file -> None OK")

    ct.TABLE_PLANE_PATH = orig
    shipped = load_table_plane("ur10e")
    if shipped is None:
        print("no local table_plane.json recorded (ok on fresh checkout)")
    else:
        print("local ur10e cal:", json.dumps(shipped))
        assert shipped["surface_z"] + shipped["safety_margin_m"] > shipped["surface_z"]
    print("ALL OFFLINE CHECKS PASSED")

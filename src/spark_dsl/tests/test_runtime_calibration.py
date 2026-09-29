"""
Tests for spark_dsl.runtime_calibration.
"""
from __future__ import annotations

import pathlib
import sys

import pytest

_SRC = pathlib.Path(__file__).resolve().parents[2]
if str(_SRC) not in sys.path:
    sys.path.insert(0, str(_SRC))

from spark_dsl import runtime_calibration as rtcal  # noqa: E402
from spark_dsl.calibration import CalibrationDB  # noqa: E402


# Helpers

def _seed_db(path: pathlib.Path, *, obj: str = "mug", surface: str = "flat_table",
             z: float = 0.012, n: int = 6) -> None:
    db = CalibrationDB(path)
    for _ in range(n):
        db.update(object=obj, surface=surface,
                  observed_offset={"z": z}, success=True)
    db.save()


# normalize_object / infer_surface_class

def test_normalize_object_strips_modifiers() -> None:
    assert rtcal.normalize_object("akita black bowl") == "bowl"
    assert rtcal.normalize_object("Red Mug") == "mug"
    assert rtcal.normalize_object("plate") == "plate"
    assert rtcal.normalize_object("") == ""
    assert rtcal.normalize_object(None) == ""


def test_infer_surface_class_basic() -> None:
    assert rtcal.infer_surface_class(None) == "flat_table"
    assert rtcal.infer_surface_class({}) == "flat_table"
    assert rtcal.infer_surface_class({"surface": "shelf"}) == "shelf"
    assert rtcal.infer_surface_class(
        {"objects": ["akita black bowl", "red mug"]}) == "bowl_interior"
    assert rtcal.infer_surface_class(
        {"detections": [{"label": "drawer_handle"}]}) == "drawer_interior"


# Preprocess

def test_preprocess_adds_learned_offset(tmp_path: pathlib.Path) -> None:
    db_path = tmp_path / "calib.yaml"
    _seed_db(db_path, obj="mug", surface="flat_table", z=0.012, n=6)
    wrapper = rtcal.CalibrationWrapper(db_path=db_path)

    out = wrapper.preprocess(
        primitive_name="move_to_keypoint",
        slots={"keypoint_label": "mug", "offset_z": 0},
        surface_hint="flat_table",
    )
    assert out["offset_z"] == pytest.approx(0.012, abs=1e-9)
    assert out["keypoint_label"] == "mug"


def test_preprocess_no_entry_returns_unchanged(tmp_path: pathlib.Path) -> None:
    db_path = tmp_path / "calib.yaml"  # never created
    wrapper = rtcal.CalibrationWrapper(db_path=db_path)

    slots = {"keypoint_label": "mug", "offset_z": 0.005}
    out = wrapper.preprocess(
        primitive_name="move_to_keypoint",
        slots=slots,
        surface_hint="flat_table",
    )
    assert out == slots
    # Should be a copy, not the same object.
    assert out is not slots


def test_preprocess_normalizes_color_modifiers(tmp_path: pathlib.Path) -> None:
    db_path = tmp_path / "calib.yaml"
    _seed_db(db_path, obj="bowl", surface="flat_table", z=0.020, n=4)
    wrapper = rtcal.CalibrationWrapper(db_path=db_path)

    out = wrapper.preprocess(
        primitive_name="move_to_keypoint",
        slots={"keypoint_label": "akita black bowl", "offset_z": 0.001},
        surface_hint="flat_table",
    )
    # 0.001 (existing) + 0.020 (learned) = 0.021
    assert out["offset_z"] == pytest.approx(0.021, abs=1e-9)


def test_preprocess_passthrough_for_non_spatial_primitives(tmp_path: pathlib.Path) -> None:
    db_path = tmp_path / "calib.yaml"
    _seed_db(db_path, obj="mug", surface="flat_table", z=0.012, n=6)
    wrapper = rtcal.CalibrationWrapper(db_path=db_path)

    for prim in ("grasp", "release", "move_relative", "wait",
                 "push_object", "turn_knob"):
        slots = {"keypoint_label": "mug", "offset_z": 0}
        out = wrapper.preprocess(
            primitive_name=prim,
            slots=slots,
            surface_hint="flat_table",
        )
        assert out == slots, f"primitive {prim} should pass through unchanged"


def test_preprocess_works_for_insert_and_wipe(tmp_path: pathlib.Path) -> None:
    db_path = tmp_path / "calib.yaml"
    _seed_db(db_path, obj="peg", surface="flat_table", z=0.005, n=3)
    wrapper = rtcal.CalibrationWrapper(db_path=db_path)

    for prim in ("insert", "wipe"):
        out = wrapper.preprocess(
            primitive_name=prim,
            slots={"keypoint_label": "peg"},
            surface_hint="flat_table",
        )
        assert out["offset_z"] == pytest.approx(0.005, abs=1e-9)


# Observe + persistence

def test_observe_persists_to_disk(tmp_path: pathlib.Path) -> None:
    db_path = tmp_path / "calib.yaml"
    wrapper = rtcal.CalibrationWrapper(db_path=db_path)

    wrapper.observe(
        primitive_name="move_to_keypoint",
        slots={"keypoint_label": "mug", "offset_z": 0.012},
        surface_hint="flat_table",
        success=True,
        observed_correction={"z": 0.014},
    )
    wrapper.observe(
        primitive_name="move_to_keypoint",
        slots={"keypoint_label": "mug", "offset_z": 0.012},
        surface_hint="flat_table",
        success=True,
        observed_correction={"z": 0.010},
    )
    wrapper.save()
    assert db_path.exists()

    reloaded = CalibrationDB.load(db_path)
    entry = reloaded.get_entry("mug", "flat_table")
    assert entry is not None
    assert entry.n_obs == 2
    assert entry.n_success == 2
    assert entry.offset["z"] == pytest.approx(0.012, abs=1e-9)


def test_observe_ignored_for_non_spatial_primitives(tmp_path: pathlib.Path) -> None:
    db_path = tmp_path / "calib.yaml"
    wrapper = rtcal.CalibrationWrapper(db_path=db_path)

    wrapper.observe(
        primitive_name="grasp",
        slots={"force": 200},
        surface_hint="flat_table",
        object_hint="mug",
        success=True,
        observed_correction={"z": 0.014},
    )
    assert len(wrapper.db) == 0


def test_observe_failure_recorded(tmp_path: pathlib.Path) -> None:
    db_path = tmp_path / "calib.yaml"
    wrapper = rtcal.CalibrationWrapper(db_path=db_path)

    wrapper.observe(
        primitive_name="move_to_keypoint",
        slots={"keypoint_label": "mug"},
        surface_hint="flat_table",
        success=False,
        observed_correction={"z": 0.020},
    )
    entry = wrapper.db.get_entry("mug", "flat_table")
    assert entry.n_obs == 1
    assert entry.n_success == 0
    assert entry.success_rate() == 0.0


def test_full_round_trip_preprocess_observe_reload(tmp_path: pathlib.Path) -> None:
    db_path = tmp_path / "calib.yaml"
    wrapper = rtcal.CalibrationWrapper(db_path=db_path)

    # Initially no offset injected.
    out0 = wrapper.preprocess(
        primitive_name="move_to_keypoint",
        slots={"keypoint_label": "mug", "offset_z": 0.0},
        surface_hint="flat_table",
    )
    assert out0["offset_z"] == 0.0

    # Observe several corrections.
    for z in (0.010, 0.012, 0.014):
        wrapper.observe(
            primitive_name="move_to_keypoint",
            slots={"keypoint_label": "mug"},
            surface_hint="flat_table",
            success=True,
            observed_correction={"z": z},
        )
    wrapper.save()

    # Reload from disk into a new wrapper and verify the learned offset is applied.
    wrapper2 = rtcal.CalibrationWrapper(db_path=db_path)
    out1 = wrapper2.preprocess(
        primitive_name="move_to_keypoint",
        slots={"keypoint_label": "mug", "offset_z": 0.0},
        surface_hint="flat_table",
    )
    assert out1["offset_z"] == pytest.approx(0.012, abs=1e-9)

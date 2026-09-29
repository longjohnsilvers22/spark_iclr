"""The grasp branch must be chosen knowing where the object is going.

Pins the 2026-08-21 14:15 run exactly: two equivalent grasp yaws (+84.9 /
-95.1 deg), shape-fit rotation -81.0 deg, wrist limit +-100 deg. Least-travel
took -95.1 for 0.2 deg less travel; the place then needed -176.1 deg, which is
unreachable, so it seated the screwdriver end-for-end after a 180 deg swing the
operator emergency-stopped. The other branch needed +3.9 deg.
"""

import types

import numpy as np
import pytest

from spark_real.control import grasp_strategy


class _Exec:
    MAX_YAW_OFFSET = np.deg2rad(100.0)
    JIGSAW_MIN_MARGIN = 0.05
    JIGSAW_MIN_IOU = 0.20

    def __init__(self, fit_deg=-81.0, margin=0.070, iou=0.743, label="screwdriver slot"):
        self._pending_place_label = label
        self._held_mask = np.ones((8, 8), dtype=bool)
        self.detection_map = {label: {"_mask": np.ones((8, 8), dtype=bool)}}
        self._fit = (np.deg2rad(fit_deg), iou, margin)


@pytest.fixture(autouse=True)
def _stub_fit(monkeypatch):
    import spark_real.perception.mask_geometry as mg

    def _fake(held, target, _ex=None):
        return _stub_fit.value

    monkeypatch.setattr(mg, "best_fit_rotation", lambda h, t: _stub_fit.value)
    yield


def _run(ex, yaw_deg):
    _stub_fit.value = ex._fit
    out = grasp_strategy.place_aware_yaw(ex, np.deg2rad(yaw_deg), "grasp")
    return round(float(np.rad2deg(out)), 1)


def test_the_measured_run_switches_to_the_reachable_branch():
    ex = _Exec()
    # least-travel handed us -95.1; its place would need -176.1 (unreachable)
    assert _run(ex, -95.1) == pytest.approx(84.9, abs=0.2)


def test_a_branch_whose_place_is_already_reachable_is_left_alone():
    ex = _Exec()
    assert _run(ex, 84.9) == pytest.approx(84.9, abs=0.2)


def test_an_undecided_fit_does_not_steer_the_grasp():
    ex = _Exec(margin=0.01)  # below JIGSAW_MIN_MARGIN
    assert _run(ex, -95.1) == pytest.approx(-95.1, abs=0.2)


def test_no_pending_place_leaves_the_grasp_untouched():
    ex = _Exec()
    ex._pending_place_label = None
    assert _run(ex, -95.1) == pytest.approx(-95.1, abs=0.2)


def test_non_grasp_contexts_are_never_steered():
    ex = _Exec()
    _stub_fit.value = ex._fit
    out = grasp_strategy.place_aware_yaw(ex, np.deg2rad(-95.1), "place")
    assert float(np.rad2deg(out)) == pytest.approx(-95.1, abs=0.2)


def test_when_neither_branch_is_reachable_the_choice_is_unchanged():
    # fit of 170 deg strands both branches past the +-100 limit
    ex = _Exec(fit_deg=170.0)
    assert _run(ex, -95.1) == pytest.approx(-95.1, abs=0.2)

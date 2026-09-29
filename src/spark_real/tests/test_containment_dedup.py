"""Cross-label dedup must not delete an object out of the container it is in.

The rig case, 2026-08-18 10:50:01: the plushie was released into the bowl, the
verify re-detect found both, and the merge dropped the plushie because its
centroid was 3cm from the bowl's::

    Cross-label dedup: dropping plushie (69%), kept bowl (82%)
    verify_placed(plushie, bowl) -> abstain [none] 'plushie' unbound

The success condition destroyed its own evidence, so inside(obj, container) was
unverifiable no matter how fast or how many times detection ran.
"""

import numpy as np
import pytest

from spark_real.perception import dedup
from spark_real.perception.spark_perception import ObjectDetection
from spark_real.pipeline_perception import PerceptionMixin


def _det(label, camera, pos, conf=0.9, area=100, centroid=(10.0, 10.0), **kw):
    d = ObjectDetection(
        label=label,
        confidence=conf,
        centroid_2d=centroid,
        mask_area=area,
        depth_meters=0.5,
        position_3d=np.array(pos, dtype=float),
    )
    d.camera = camera
    for k, v in kw.items():
        setattr(d, k, v)
    return d


def _bowl(camera="birdview", pos=(-0.98, -0.12, -0.19), conf=0.82, **kw):
    """A bowl the way mask_geometry stamps one: rim above a measured floor."""
    kw.setdefault("rim_z_m", -0.224)
    kw.setdefault("interior_z_m", -0.268)
    kw.setdefault("height_samples", 400)
    return _det("bowl", camera, pos, conf=conf, **kw)


class _MergeOnly(PerceptionMixin):
    """Bare host for merge_detections (no pipeline state needed)."""


def _merge(dets):
    return _MergeOnly().merge_detections.__func__(_MergeOnly(), dets)


def _labels(dets):
    return sorted(d.label for d in dets)


# --- the regression itself -------------------------------------------------


def test_plushie_in_the_bowl_survives_the_merge():
    """The logged pair: same XY within 3cm, 6.9cm apart in Z, one a container."""
    plushie = _det("plushie", "sideview", [-0.984, -0.121, -0.122], conf=0.69)
    merged = _merge([_bowl(), plushie])
    assert _labels(merged) == ["bowl", "plushie"], (
        "the placed object was deduped out of its own container"
    )


def test_the_old_rule_is_still_reachable(monkeypatch):
    monkeypatch.setenv(dedup.ENV_FLAG, "0")
    plushie = _det("plushie", "sideview", [-0.984, -0.121, -0.122], conf=0.69)
    merged = _merge([_bowl(), plushie])
    assert _labels(merged) == ["bowl"]


def test_a_genuine_duplicate_is_still_deduped():
    """Two prompts on one spoon: same XY, same Z, neither a container."""
    spoon = _det("spoon", "birdview", [0.10, 0.20, -0.25], conf=0.90)
    tool = _det("utensil", "birdview", [0.11, 0.20, -0.25], conf=0.40)
    merged = _merge([spoon, tool])
    assert _labels(merged) == ["spoon"]


# --- same_object_3d ---------------------------------------------------------


def test_3d_colocated_and_level_is_one_object():
    a = _det("a", "c", [0.0, 0.0, 0.0])
    b = _det("b", "c", [0.01, 0.0, 0.005])
    assert dedup.same_object_3d(a, b, 0.05) is True


def test_3d_separated_in_z_is_two_objects():
    a = _det("a", "c", [0.0, 0.0, 0.0])
    b = _det("b", "c", [0.01, 0.0, 0.069])
    assert dedup.same_object_3d(a, b, 0.05) is False
    assert dedup.same_object_3d(a, b, 0.05, aware=False) is True


def test_3d_far_apart_is_two_objects_under_either_rule():
    a = _det("a", "c", [0.0, 0.0, 0.0])
    b = _det("b", "c", [0.30, 0.0, 0.0])
    assert dedup.same_object_3d(a, b, 0.05) is False
    assert dedup.same_object_3d(a, b, 0.05, aware=False) is False


def test_a_container_shields_the_pair_even_when_z_agrees():
    """Level with the rim is still IN it; the container guard runs first."""
    bowl = _bowl(pos=(0.0, 0.0, -0.20))
    obj = _det("plushie", "birdview", [0.01, 0.0, -0.203])
    assert dedup.same_object_3d(obj, bowl, 0.05) is False


def test_a_container_with_no_measured_floor_still_shields_the_pair():
    bowl = _det("bowl", "birdview", [0.0, 0.0, -0.20], slots=[{"xy": (0, 0)}])
    obj = _det("plushie", "birdview", [0.01, 0.0, -0.201])
    assert dedup.is_container(bowl) is True
    assert dedup.same_object_3d(obj, bowl, 0.05) is False


def test_is_container_needs_measured_geometry():
    assert dedup.is_container(_det("bowl", "c", [0, 0, 0])) is False
    assert dedup.is_container(_bowl()) is True
    # rim/interior present but nothing behind them
    assert (
        dedup.is_container(
            _det("bowl", "c", [0, 0, 0], rim_z_m=-0.2, interior_z_m=-0.3, height_samples=0)
        )
        is False
    )


# --- same_object_2d ---------------------------------------------------------


def _mask(shape, box):
    m = np.zeros(shape, bool)
    y0, y1, x0, x1 = box
    m[y0:y1, x0:x1] = True
    return m


def test_2d_same_mask_under_two_names_is_one_object():
    shape = (100, 100)
    big = _mask(shape, (20, 80, 20, 80))
    a = _det("spoon", "c", [0, 0, 0], centroid=(50.0, 50.0), mask=big)
    b = _det("utensil", "c", [0, 0, 0], centroid=(51.0, 50.0), mask=big.copy())
    assert dedup.same_object_2d(a, b, 30) is True


def test_2d_small_mask_inside_a_big_one_is_two_objects():
    shape = (100, 100)
    bowl = _det(
        "bowl", "c", [0, 0, 0], centroid=(50.0, 50.0), mask=_mask(shape, (20, 80, 20, 80))
    )
    plushie = _det(
        "plushie", "c", [0, 0, 0], centroid=(52.0, 50.0), mask=_mask(shape, (45, 55, 45, 55))
    )
    assert dedup.mask_iou(bowl, plushie) < dedup.CROSS_LABEL_DEDUP_MIN_IOU
    assert dedup.same_object_2d(bowl, plushie, 30) is False
    # ...and the historical rule would have merged them.
    assert dedup.same_object_2d(bowl, plushie, 30, aware=False) is True


def test_2d_without_masks_falls_back_to_the_historical_rule():
    a = _det("a", "c", [0, 0, 0], centroid=(50.0, 50.0))
    b = _det("b", "c", [0, 0, 0], centroid=(55.0, 50.0))
    assert dedup.mask_iou(a, b) is None
    assert dedup.same_object_2d(a, b, 30) is True


def test_2d_far_apart_centroids_are_never_one_object():
    a = _det("a", "c", [0, 0, 0], centroid=(10.0, 10.0))
    b = _det("b", "c", [0, 0, 0], centroid=(200.0, 200.0))
    assert dedup.same_object_2d(a, b, 30) is False


@pytest.mark.parametrize("value,expected", [("0", False), ("1", True), ("off", False)])
def test_env_flag_toggles_the_rule(monkeypatch, value, expected):
    monkeypatch.setenv(dedup.ENV_FLAG, value)
    assert dedup.containment_aware() is expected

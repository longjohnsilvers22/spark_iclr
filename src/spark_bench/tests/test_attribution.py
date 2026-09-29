"""
Decision-table tests for lowest-responsible-layer recovery attribution
(spark_real.control.attribution.attribute_failure).
"""
import pytest

from spark_real.control.attribution import Layer, attribute_failure
from spark_real.perception.sticky_binding import compute_scene_diff


def _fail(**kw):
    base = {'predicate_ok': False}
    base.update(kw)
    return base


def test_missing_detection_is_perception():
    assert attribute_failure(_fail(detection_missing=True), None, 0) \
        is Layer.PERCEPTION


def test_low_confidence_is_perception():
    assert attribute_failure(_fail(detection_confidence=0.05), None, 0) \
        is Layer.PERCEPTION


def test_moved_target_is_perception():
    sd = compute_scene_diff({'bowl': [0, 0, 0.9]}, {'bowl': [0.06, 0, 0.9]},
                              target_label='bowl')
    assert attribute_failure(_fail(detection_confidence=0.9), sd, 0) \
        is Layer.PERCEPTION


def test_moved_target_dict_form_is_perception():
    # attribute_failure accepts the to_meta() dict form too.
    assert attribute_failure(_fail(), {'target_status': 'moved'}, 0) \
        is Layer.PERCEPTION


def test_stable_detection_failed_predicate_is_execution():
    sd = compute_scene_diff({'bowl': [0, 0, 0.9]}, {'bowl': [0, 0, 0.9]},
                              target_label='bowl')
    assert attribute_failure(_fail(detection_confidence=0.9), sd, 0) \
        is Layer.EXECUTION


def test_empty_close_with_stable_scene_is_execution():
    sd = compute_scene_diff({'bowl': [0, 0, 0.9]}, {'bowl': [0, 0, 0.9]},
                              target_label='bowl')
    v = _fail(detection_confidence=0.9, grasp_outcome='empty_close')
    assert attribute_failure(v, sd, 0) is Layer.EXECUTION


def test_retries_exhausted_is_plan():
    sd = compute_scene_diff({'bowl': [0, 0, 0.9]}, {'bowl': [0.06, 0, 0.9]},
                              target_label='bowl')
    # Even with a perception-looking diff, exhausted retries escalate.
    assert attribute_failure(_fail(), sd, retry_count=1,
                               max_local_retries=1) is Layer.PLAN
    assert attribute_failure(_fail(), sd, retry_count=2,
                               max_local_retries=3) is Layer.PERCEPTION


def test_semantic_failure_is_plan():
    assert attribute_failure(_fail(semantic_failure=True), None, 0) \
        is Layer.PLAN
    # Semantic beats everything, including perception evidence.
    assert attribute_failure(
        _fail(semantic_failure=True, detection_missing=True), None, 0) \
        is Layer.PLAN


def test_no_diff_no_signals_defaults_to_execution():
    assert attribute_failure(_fail(), None, 0) is Layer.EXECUTION


def test_confidence_none_is_not_low_confidence():
    assert attribute_failure(_fail(detection_confidence=None), None, 0) \
        is Layer.EXECUTION


@pytest.mark.parametrize('status,expected', [
    ('moved', Layer.PERCEPTION),
    ('missing', Layer.PERCEPTION),
    ('ok', Layer.EXECUTION),
    ('self', Layer.EXECUTION),
    ('unbound', Layer.EXECUTION),
])
def test_diff_status_table(status, expected):
    assert attribute_failure(_fail(), {'target_status': status}, 0) \
        is expected

"""
Table-driven test of the shared attribution decision table, from the real
side (the module moved to spark_real.control.attribution; the sim imports
it back through a shim).
"""

import pytest

from spark_real.control.attribution import Layer, attribute_failure


@pytest.mark.parametrize(
    "verify,retries,expected",
    [
        ({"semantic_failure": True}, 0, Layer.PLAN),
        ({}, 1, Layer.PLAN),  # retries exhausted (max_local_retries=1)
        ({"detection_missing": True}, 0, Layer.PERCEPTION),
        ({"detection_confidence": 0.05}, 0, Layer.PERCEPTION),
        ({"detection_confidence": 0.8}, 0, Layer.EXECUTION),
        ({"grasp_outcome": "empty_close"}, 0, Layer.EXECUTION),
        ({}, 0, Layer.EXECUTION),
    ],
)
def test_attribution_table(verify, retries, expected):
    assert (
        attribute_failure(verify, None, retries, max_local_retries=1)
        == expected
    )


def test_scene_diff_moved_routes_to_perception():
    class _Diff:
        target_status = "moved"

    assert (
        attribute_failure({}, _Diff(), 0, max_local_retries=2)
        == Layer.PERCEPTION
    )
    assert (
        attribute_failure({}, {"target_status": "missing"}, 0)
        == Layer.PERCEPTION
    )

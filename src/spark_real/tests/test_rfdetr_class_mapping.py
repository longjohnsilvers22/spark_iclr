"""RF-DETR class ids must map through COCO ids, not list positions.

rfdetr.assets.coco_classes.COCO_CLASSES is a dict keyed by COCO category id,
which runs 1..90 with gaps and holds only 80 entries. RFDETRProposer._class_name
flattened it with .values() and then indexed POSITIONALLY, so every label was
shifted and ids past the 80th fell off the end entirely:

    id 88 == "teddy bear"  -> positional index 88 -> out of range -> "88"
    id 52 == "banana"      -> positional index 52 -> "hot dog"

Observed live on the plushie scene: the proposer reported `88` at 0.18 and
`carrot` at 0.91. It had in fact detected the teddy bear -- the label was
destroyed on the way out, so the fusion gate could never match it to the
"plushie" prompt.

These tests read the real COCO table, so they fail if the mapping regresses to
positional indexing regardless of which ids happen to be present.
"""

import pytest

rfdetr = pytest.importorskip("rfdetr", reason="rfdetr not installed")

from rfdetr.assets.coco_classes import COCO_CLASSES  # noqa: E402

from spark_real.perception.box_proposals import RFDETRProposer  # noqa: E402


def _name(cid):
    """_class_name without constructing the model (no weights, no download)."""
    return RFDETRProposer._class_name(RFDETRProposer, cid)


def test_coco_table_is_id_keyed_with_gaps():
    """Guards the premise: if upstream flattens this, the fix must be revisited."""
    assert isinstance(COCO_CLASSES, dict)
    keys = sorted(COCO_CLASSES)
    assert len(COCO_CLASSES) == 80
    assert keys[-1] > len(COCO_CLASSES), "no id/position gap -- premise changed"


def test_teddy_bear_id_maps_to_teddy_bear():
    """The exact failure seen on the plushie run."""
    tid = next(k for k, v in COCO_CLASSES.items() if str(v).lower() == "teddy bear")
    assert _name(tid) == "teddy bear", f"id {tid} mapped to {_name(tid)!r}"


def test_every_coco_id_round_trips():
    for cid, expected in COCO_CLASSES.items():
        assert _name(cid) == str(expected).lower(), f"id {cid} mismapped"


def test_no_label_is_a_bare_number():
    """A numeric label means the lookup fell through, which is the old bug."""
    for cid in COCO_CLASSES:
        assert not _name(cid).isdigit(), f"id {cid} produced a numeric label"


def test_unknown_id_does_not_raise():
    """Out-of-table ids must degrade, not crash perception."""
    assert isinstance(_name(9999), str)

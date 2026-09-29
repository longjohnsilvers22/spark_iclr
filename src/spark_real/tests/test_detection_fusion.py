"""Offline tests for the box-proposal fusion gate and the ASPIRE repair loop.

No robot, no camera, no network. Masks are synthetic. One test runs a REAL
COCO detector and skips itself unless the weights are already in the local
torch cache, so the suite never downloads anything.

The real-data end-to-end demonstration lives in tests/fusion_plushie_demo.py.
"""

import os
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

import cv2
import numpy as np
import pytest

from spark_real.control import grasp_strategy as gs
from spark_real.perception.box_proposals import (
    BoxProposal,
    NullProposer,
    ProposalSet,
    RFDETRProposer,
    StaticProposer,
    TorchvisionCOCOProposer,
    build_proposer,
    default_alt_prompts,
)
from spark_real.perception.detection_fusion import (
    ABSTAIN,
    AGREE_BOX,
    AGREE_CLASS,
    MISS,
    DetectionGate,
    FusionConfig,
    box_iou,
    fuse_one,
    mask_bbox,
    orphan_boxes,
)
from spark_real.perception.mask_quality import measure_mask

COCO = ("teddy bear", "knife", "spoon", "bowl", "fork", "cup", "person")


@dataclass
class FakeDet:
    """Duck-types ObjectDetection over exactly the fields the gate touches."""

    label: str
    confidence: float
    mask: Optional[np.ndarray] = None
    bbox: Optional[tuple] = None
    aspect_ratio: float = 1.0
    orientation_angle: float = 0.0
    obb_confidence: float = 0.0
    centroid_2d: tuple = (0.0, 0.0)
    mask_area: int = 0
    obb_minor_m: float = 0.0
    position_3d: Optional[np.ndarray] = None
    depth_meters: float = 0.0
    low_quality: bool = False
    reprompt_attempts: int = 0


def knife_mask(angle_deg=-31.0, shape=(480, 640)):
    m = np.zeros(shape, np.uint8)
    box = cv2.boxPoints(((320, 240), (220, 26), angle_deg)).astype(np.int32)
    cv2.fillPoly(m, [box], 1)
    return m


def plushie_mask(shape=(480, 640)):
    """Round blob with two limb lumps: no real major axis."""
    m = np.zeros(shape, np.uint8)
    cv2.ellipse(m, (320, 240), (70, 58), 0, 0, 360, 1, -1)
    cv2.circle(m, (250, 215), 22, 1, -1)
    cv2.circle(m, (392, 262), 20, 1, -1)
    return m


def bleeding_plushie_mask(shape=(480, 640)):
    """Blob whose mask bleeds down a shadow, faking elongation."""
    m = plushie_mask(shape)
    cv2.line(m, (320, 240), (470, 300), 1, 9)
    return m


def on_cfg(**kw):
    base = dict(enabled=True)
    base.update(kw)
    return FusionConfig.from_dict(base)


def det_dict(label, conf, mask=None, ar=1.0, angle_deg=0.0, **extra):
    """The dict shape the executor actually reads (pipeline detection_map)."""
    d = {
        "label": label,
        "confidence": conf,
        "mask": mask,
        "aspect_ratio": ar,
        "orientation_angle": float(np.deg2rad(angle_deg)),
        "position_3d": [-0.9, 0.0, -0.24],
    }
    d.update(extra)
    return d


# --------------------------------------------------------------------------
# mask quality, no second model involved
# --------------------------------------------------------------------------


def test_knife_axis_is_trustworthy():
    q = measure_mask(knife_mask(), reported_aspect_ratio=4.8)
    assert q.measured and q.angle_trustworthy
    assert q.angle_swing_deg < 2.0
    assert q.mask_aspect_ratio > 4.0


def test_plushie_axis_rejected_because_mask_is_round():
    q = measure_mask(plushie_mask(), reported_aspect_ratio=3.14)
    assert not q.angle_trustworthy
    assert q.ar_inflation > 1.6
    assert q.geometry_trust == 0.0


def test_bleeding_mask_rejected_because_axis_swings():
    q = measure_mask(bleeding_plushie_mask(), reported_aspect_ratio=3.14)
    assert not q.angle_trustworthy
    assert q.angle_swing_deg > 12.0


def test_ar_check_is_directional():
    """Mask longer than the world OBB is foreshortening, not a fault."""
    q = measure_mask(knife_mask(), reported_aspect_ratio=2.0)
    assert q.ar_inflation < 1.0
    assert q.angle_trustworthy


def test_degenerate_masks_never_raise():
    for m in (None, np.zeros((10, 10), np.uint8), np.ones((3, 3), np.uint8)):
        q = measure_mask(m, reported_aspect_ratio=3.0)
        assert not q.measured and not q.angle_trustworthy


# --------------------------------------------------------------------------
# fusion rule: the four agreement cases
# --------------------------------------------------------------------------


def test_agree_class_raises_confidence():
    det = FakeDet("plushie", 0.24, mask=plushie_mask(), aspect_ratio=3.14)
    props = ProposalSet(
        [BoxProposal("teddy bear", 0.91, mask_bbox(det.mask))], frozenset(COCO), True, "static"
    )
    r = fuse_one(det, props, on_cfg())
    assert r.agreement == AGREE_CLASS
    assert r.fused_confidence > 0.24


def test_agree_box_only_is_capped_and_weaker_than_class_agreement():
    det = FakeDet("plushie", 0.24, mask=plushie_mask(), aspect_ratio=3.14)
    bb = mask_bbox(det.mask)
    box_only = fuse_one(
        det, ProposalSet([BoxProposal("person", 0.95, bb)], frozenset(COCO), True, "s"), on_cfg()
    )
    same = fuse_one(
        det,
        ProposalSet([BoxProposal("teddy bear", 0.95, bb)], frozenset(COCO), True, "s"),
        on_cfg(),
    )
    assert box_only.agreement == AGREE_BOX
    assert box_only.fused_confidence < same.fused_confidence
    assert box_only.fused_confidence <= on_cfg().box_only_ceiling


def test_miss_penalises_only_in_vocabulary_labels():
    det = FakeDet("knife", 0.55, mask=knife_mask(), aspect_ratio=4.8)
    far = BoxProposal("bowl", 0.9, (0.0, 0.0, 30.0, 30.0))
    r = fuse_one(det, ProposalSet([far], frozenset(COCO), True, "s"), on_cfg())
    assert r.agreement == MISS
    assert r.fused_confidence < 0.55


def test_out_of_vocabulary_label_abstains_and_never_penalises():
    """The correctness case: RF-DETR is closed-set COCO. 'tray' is not COCO,
    so silence about a tray is no evidence at all."""
    det = FakeDet("tray", 0.42, mask=knife_mask(), aspect_ratio=4.8)
    far = BoxProposal("bowl", 0.9, (0.0, 0.0, 30.0, 30.0))
    r = fuse_one(det, ProposalSet([far], frozenset(COCO), True, "s"), on_cfg())
    assert r.agreement == ABSTAIN
    assert r.fused_confidence == pytest.approx(0.42)


def test_unavailable_proposer_abstains():
    det = FakeDet("knife", 0.55, mask=knife_mask(), aspect_ratio=4.8)
    r = fuse_one(det, ProposalSet(available=False, note="not installed"), on_cfg())
    assert r.agreement == ABSTAIN
    assert r.fused_confidence == pytest.approx(0.55)


def test_orphan_box_is_reported_not_turned_into_a_detection():
    det = FakeDet("knife", 0.55, mask=knife_mask())
    orphan = BoxProposal("bowl", 0.88, (10.0, 10.0, 90.0, 90.0))
    props = ProposalSet(
        [orphan, BoxProposal("knife", 0.8, mask_bbox(det.mask))], frozenset(COCO), True, "s"
    )
    o = orphan_boxes([det], props, on_cfg())
    assert [p.label for p in o] == ["bowl"]


def test_box_iou_basics():
    assert box_iou((0, 0, 10, 10), (0, 0, 10, 10)) == pytest.approx(1.0)
    assert box_iou((0, 0, 10, 10), (20, 20, 30, 30)) == 0.0


# --------------------------------------------------------------------------
# the operator's plushie case, end to end
# --------------------------------------------------------------------------


def test_plushie_cannot_drive_a_yaw_even_with_no_second_model():
    """SAM3 conf 0.24, aspect_ratio 3.14. No proposer available at all."""
    det = FakeDet("plushie", 0.24, mask=plushie_mask(), aspect_ratio=3.14)
    gate = DetectionGate(on_cfg())
    (r,) = gate.apply([det], rgb=None)
    assert r.agreement == ABSTAIN
    assert r.fused_confidence == pytest.approx(0.24)
    assert not r.angle_trustworthy
    assert det.low_quality is True


def test_a_confident_second_opinion_still_cannot_rescue_a_round_mask():
    """Identity confidence and geometry trust are separate axes."""
    det = FakeDet("plushie", 0.24, mask=plushie_mask(), aspect_ratio=3.14)
    props = StaticProposer([BoxProposal("teddy bear", 0.93, mask_bbox(det.mask))], COCO)
    gate = DetectionGate(on_cfg(), proposer=props)
    (r,) = gate.apply([det], rgb=np.zeros((480, 640, 3), np.uint8))
    assert r.agreement == AGREE_CLASS
    assert r.fused_confidence > 2 * r.sam3_confidence  # identity is now settled
    assert not r.angle_trustworthy  # but the shape still has no axis
    assert det.low_quality is True


def test_knife_is_left_alone():
    det = FakeDet("knife", 0.72, mask=knife_mask(), aspect_ratio=4.8)
    gate = DetectionGate(on_cfg())
    (r,) = gate.apply([det], rgb=None)
    assert r.angle_trustworthy
    assert det.low_quality is False


# --------------------------------------------------------------------------
# ASPIRE
# --------------------------------------------------------------------------


def test_aspire_accepts_a_strictly_better_reprompt():
    det = FakeDet("plushie", 0.24, mask=plushie_mask(), aspect_ratio=3.14)
    calls = []

    def redetect(prompt, box):
        calls.append((prompt, box))
        return FakeDet(prompt, 0.81, mask=knife_mask(), aspect_ratio=4.6)

    gate = DetectionGate(on_cfg())
    (r,) = gate.apply(
        [det], rgb=None, redetect=redetect, alt_prompts={"plushie": ["stuffed animal"]}
    )
    assert calls == [("stuffed animal", None)]
    assert r.reprompt_attempts == 1
    assert det.low_quality is False
    assert det.confidence == pytest.approx(0.81)  # geometry adopted


def test_aspire_rejects_a_worse_reprompt_and_recovers():
    det = FakeDet("plushie", 0.24, mask=plushie_mask(), aspect_ratio=3.14)

    def redetect(prompt, box):
        return FakeDet(prompt, 0.05, mask=plushie_mask(), aspect_ratio=3.14)

    gate = DetectionGate(on_cfg())
    (r,) = gate.apply(
        [det], rgb=None, redetect=redetect, alt_prompts={"plushie": ["stuffed animal"]}
    )
    assert r.reprompt_attempts == 1
    assert det.confidence == pytest.approx(0.24)  # incumbent kept
    assert det.low_quality is True
    assert any("recovered as low_quality" in n for n in r.notes)


def test_aspire_is_capped_by_reprompt_max_attempts():
    det = FakeDet("plushie", 0.24, mask=plushie_mask(), aspect_ratio=3.14)
    calls = []

    def redetect(prompt, box):
        calls.append(prompt)
        return None

    gate = DetectionGate(on_cfg(reprompt_max_attempts=2))
    gate.apply([det], rgb=None, redetect=redetect, alt_prompts={"plushie": ["a", "b", "c", "d"]})
    assert len(calls) == 2


def test_aspire_uses_the_partner_box_as_a_geometric_prompt_first():
    """An ASSOCIATED box is an independent observation of THIS object, so it
    goes ahead of blind text alternates."""
    det = FakeDet("plushie", 0.24, mask=plushie_mask(), aspect_ratio=3.14)
    partner = BoxProposal("teddy bear", 0.9, mask_bbox(det.mask))
    props = StaticProposer([partner], COCO)
    seen = []

    def redetect(prompt, box):
        seen.append((prompt, box))
        return None

    gate = DetectionGate(on_cfg(), proposer=props)
    gate.apply([det], rgb=np.zeros((480, 640, 3), np.uint8), redetect=redetect)
    assert seen[0] == ("plushie", mask_bbox(det.mask))


def test_aspire_never_prompts_with_an_orphan_box():
    """An orphan matched no mask BY DEFINITION, so prompting with it returns a
    different object. orphan_boxes() reports them for the caller instead."""
    det = FakeDet("plushie", 0.24, mask=plushie_mask(), aspect_ratio=3.14)
    orphan = BoxProposal("teddy bear", 0.9, (500.0, 300.0, 600.0, 420.0))
    props = StaticProposer([orphan], COCO)
    seen = []

    def redetect(prompt, box):
        seen.append((prompt, box))
        return None

    gate = DetectionGate(on_cfg(), proposer=props)
    gate.apply([det], rgb=np.zeros((480, 640, 3), np.uint8), redetect=redetect)
    assert seen and all(box is None for _, box in seen)
    assert [p.label for p in orphan_boxes([det], props.propose(None), on_cfg())] == ["teddy bear"]


def test_a_raising_redetect_does_not_break_the_gate():
    det = FakeDet("plushie", 0.24, mask=plushie_mask(), aspect_ratio=3.14)

    def redetect(prompt, box):
        raise RuntimeError("sam3 exploded")

    gate = DetectionGate(on_cfg())
    (r,) = gate.apply(
        [det], rgb=None, redetect=redetect, alt_prompts={"plushie": ["stuffed animal"]}
    )
    assert det.low_quality is True


def test_a_raising_proposer_degrades_to_abstain():
    class Boom(NullProposer):
        def propose(self, rgb, labels=()):
            raise RuntimeError("cuda oom")

    det = FakeDet("knife", 0.55, mask=knife_mask(), aspect_ratio=4.8)
    gate = DetectionGate(on_cfg(), proposer=Boom())
    (r,) = gate.apply([det], rgb=np.zeros((4, 4, 3), np.uint8))
    assert r.agreement == ABSTAIN
    assert r.fused_confidence == pytest.approx(0.55)


# --------------------------------------------------------------------------
# config gating: OFF unless asked
# --------------------------------------------------------------------------


def test_gate_is_off_by_default_and_touches_nothing():
    det = FakeDet("plushie", 0.24, mask=plushie_mask(), aspect_ratio=3.14)
    assert DetectionGate().apply([det], rgb=None) == []
    assert det.low_quality is False
    assert det.reprompt_attempts == 0
    assert det.confidence == pytest.approx(0.24)


def test_confidence_is_annotated_not_rewritten_by_default():
    det = FakeDet("knife", 0.55, mask=knife_mask(), aspect_ratio=4.8)
    props = StaticProposer([BoxProposal("knife", 0.95, mask_bbox(det.mask))], COCO)
    (r,) = DetectionGate(on_cfg(), proposer=props).apply([det], rgb=np.zeros((4, 4, 3), np.uint8))
    assert r.fused_confidence > 0.55
    assert det.confidence == pytest.approx(0.55)


def test_write_confidence_opt_in():
    det = FakeDet("knife", 0.55, mask=knife_mask(), aspect_ratio=4.8)
    props = StaticProposer([BoxProposal("knife", 0.95, mask_bbox(det.mask))], COCO)
    DetectionGate(on_cfg(write_confidence=True), proposer=props).apply(
        [det], rgb=np.zeros((4, 4, 3), np.uint8)
    )
    assert det.confidence > 0.55


def test_build_proposer_defaults_to_null():
    assert isinstance(build_proposer(None), NullProposer)
    assert isinstance(build_proposer({}), NullProposer)
    # named but never switched on: not an error
    assert isinstance(build_proposer({"backend": "rfdetr"}), NullProposer)
    assert build_proposer(None).propose(None).available is False


def test_an_unknown_backend_that_was_asked_for_fails_loudly():
    with pytest.raises(RuntimeError, match="unknown box proposer backend"):
        build_proposer({"enabled": True, "backend": "bogus"})
    assert isinstance(
        build_proposer({"enabled": True, "backend": "bogus", "strict": False}), NullProposer
    )


def test_unavailable_rfdetr_fails_loudly_rather_than_silently_abstaining():
    """The operator turning fusion on and getting no protection AND no error is
    the failure that matters. strict:false restores degrade-to-null."""
    try:
        import rfdetr  # noqa: F401
    except ImportError:
        with pytest.raises(RuntimeError, match="rfdetr.*requested but is unusable"):
            build_proposer({"enabled": True, "backend": "rfdetr"})
        assert isinstance(
            build_proposer({"enabled": True, "backend": "rfdetr", "strict": False}), NullProposer
        )
    else:  # pragma: no cover - the day someone installs it
        assert build_proposer({"enabled": True, "backend": "rfdetr"}).vocabulary()


def test_rfdetr_constructor_fails_loudly_when_absent():
    try:
        import rfdetr  # noqa: F401
    except ImportError:
        with pytest.raises(RuntimeError, match="pip install rfdetr"):
            RFDETRProposer()


# --------------------------------------------------------------------------
# the second detector that DOES run here
# --------------------------------------------------------------------------


def _coco_weights_cached() -> bool:
    """True when the COCO checkpoint is already local. Keeps the suite offline."""
    try:
        from torchvision.models.detection import FasterRCNN_ResNet50_FPN_Weights as W
    except ImportError:
        return False
    name = W.COCO_V1.url.rsplit("/", 1)[-1]
    root = Path(os.environ.get("TORCH_HOME", Path.home() / ".cache" / "torch"))
    return (root / "hub" / "checkpoints" / name).exists()


@pytest.mark.skipif(not _coco_weights_cached(), reason="COCO weights not in the local torch cache")
def test_torchvision_proposer_really_runs_on_cpu():
    """A real second detector, real weights, CPU so the live server keeps the GPU."""
    p = build_proposer({"enabled": True, "backend": "torchvision", "device": "cpu"})
    assert isinstance(p, TorchvisionCOCOProposer)
    assert {"teddy bear", "bowl", "knife"} <= p.vocabulary()

    rgb = np.full((240, 320, 3), 200, np.uint8)
    cv2.rectangle(rgb, (120, 90), (200, 170), (40, 40, 160), -1)
    pset = p.propose(rgb)
    assert pset.available and pset.source == "torchvision"
    assert all(0.0 <= q.confidence <= 1.0 and len(q.box) == 4 for q in pset.proposals)


@pytest.mark.skipif(not _coco_weights_cached(), reason="COCO weights not in the local torch cache")
def test_torchvision_proposer_rejects_an_unknown_variant():
    with pytest.raises(RuntimeError, match="unknown torchvision variant"):
        TorchvisionCOCOProposer(variant="not-a-model")


def test_torchvision_proposer_abstains_on_a_bad_image():
    """Never raises out of propose(); a broken frame is an abstain."""

    class Fake(TorchvisionCOCOProposer):
        def __init__(self):
            self._categories, self._vocab, self.variant = ["bowl"], frozenset({"bowl"}), "fake"
            self.threshold, self.device, self._torch, self._model = 0.5, "cpu", None, None

    assert Fake().propose(np.zeros((4, 4), np.uint8)).available is False
    assert Fake().propose("not an image").available is False


# --------------------------------------------------------------------------
# the fused number reaches the SAME gate that guards the oriented grasp
# --------------------------------------------------------------------------


class _Exec:
    """Only what resolve_strategy touches. Values from configs/ur10e_default.yaml."""

    GRASP_YAW_AR_GATE = 1.6
    GRASP_YAW_MIN_CONF = 0.45
    GRASP_YAW_MIN_OBB_CONF = 0.40


def test_fused_confidence_gates_yaw_min_conf():
    """SAM3 0.55 clears the 0.45 gate; a second detector that MISSES it does not."""
    d = det_dict("knife", 0.55, mask=knife_mask(), ar=4.8, angle_deg=30.0)
    assert gs.resolve_strategy({}, d, _Exec()) == ("obb", "ar 4.80 >= gate 1.60, conf ok")

    far = BoxProposal("bowl", 0.9, (0.0, 0.0, 30.0, 30.0))
    props = StaticProposer([far], COCO)
    # low_quality_conf out of the way, so this asserts the CONFIDENCE gate and
    # not the low_quality one: they are independent vetoes.
    (r,) = DetectionGate(on_cfg(miss_penalty=1.0, low_quality_conf=0.05), proposer=props).apply(
        [d], rgb=np.zeros((480, 640, 3), np.uint8)
    )
    assert r.agreement == MISS and d["low_quality"] is False
    assert d["fused_confidence"] < 0.45 < d["confidence"]
    strategy, reason = gs.resolve_strategy({}, d, _Exec())
    assert strategy == "topdown"
    assert reason.startswith("fused_conf")


def test_axis_trust_gates_yaw_min_obb_conf():
    """The mask-measured axis number vetoes on the obb_confidence gate, and it
    is a min() with perception's own so either measurement can refuse."""
    d = det_dict("plushie", 0.99, mask=plushie_mask(), ar=3.14, angle_deg=111.1)
    del d["mask"]  # keep the geometry, drop low_quality's usual trigger source
    d["mask"] = plushie_mask()
    DetectionGate(on_cfg()).apply([d], rgb=None)
    assert d["axis_trust"] == 0.0
    d["low_quality"] = False  # isolate the axis_trust path from low_quality
    strategy, reason = gs.resolve_strategy({}, d, _Exec())
    assert strategy == "topdown"
    assert reason.startswith("axis_trust")


def test_a_detection_without_fusion_fields_behaves_exactly_as_before():
    """Backward compatibility: absent fused_confidence/axis_trust means the
    gate never ran, so the raw SAM3 confidence is used, as it always was."""
    d = det_dict("knife", 0.55, mask=knife_mask(), ar=4.8, angle_deg=30.0)
    assert "fused_confidence" not in d and "axis_trust" not in d
    assert gs.resolve_strategy({}, d, _Exec())[0] == "obb"
    d["confidence"] = 0.30
    assert gs.resolve_strategy({}, d, _Exec()) == ("topdown", "conf 0.30 < 0.45 (ar 4.80)")


def test_the_plushie_earns_no_wrist_rotation_end_to_end():
    """The operator's case, through the real resolver: SAM3 0.24 + ar 3.14.
    Also asserted at the high SAM3 score, where the OLD gate DID rotate."""
    for conf in (0.24, 0.93):
        d = det_dict("plushie", conf, mask=plushie_mask(), ar=3.14, angle_deg=111.1)
        (r,) = DetectionGate(on_cfg()).apply([d], rgb=None)
        assert not r.angle_trustworthy
        assert d["low_quality"] is True and d["axis_trust"] == 0.0
        assert gs.resolve_strategy({}, d, _Exec()) == ("topdown", "mask flagged low_quality")


def test_the_gate_stamps_dict_detections():
    """The executor reads a dict, not the dataclass. A hasattr-guarded setattr
    would have made the whole gate a silent no-op there."""
    d = det_dict("plushie", 0.24, mask=plushie_mask(), ar=3.14)
    (r,) = DetectionGate(on_cfg()).apply([d], rgb=None)
    assert d["low_quality"] is True
    assert d["fused_confidence"] == pytest.approx(r.fused_confidence)
    assert d["axis_trust"] == pytest.approx(r.quality.axis_trust)
    assert d["confidence"] == pytest.approx(0.24)  # untouched


def test_axis_trust_is_positive_for_a_real_axis():
    d = det_dict("knife", 0.72, mask=knife_mask(), ar=4.8)
    (r,) = DetectionGate(on_cfg()).apply([d], rgb=None)
    assert d["axis_trust"] > 0.4 and r.angle_trustworthy
    assert d["low_quality"] is False


# --------------------------------------------------------------------------
# ASPIRE with NO second model at all
# --------------------------------------------------------------------------


def test_default_alt_prompts_need_no_model_and_order_by_the_failure():
    """Elongation claimed -> sub-parts first; blob -> synonyms first."""
    assert default_alt_prompts("plushie", elongated=False)[0] == "teddy bear"
    assert default_alt_prompts("knife", elongated=True)[0] == "knife handle"
    assert "knife handle" in default_alt_prompts("knife 2", elongated=True)
    assert default_alt_prompts("") == []
    # out-of-synonym labels still get a usable sub-part vocabulary
    assert default_alt_prompts("widget") == ["widget handle", "widget grip"]


def test_aspire_reprompts_with_no_proposer_and_no_caller_prompts():
    """The whole repair path with no second model and no alt_prompts dict."""
    det = FakeDet("plushie", 0.24, mask=plushie_mask(), aspect_ratio=3.14)
    seen = []

    def redetect(prompt, box):
        seen.append((prompt, box))
        return FakeDet(prompt, 0.80, mask=knife_mask(), aspect_ratio=4.6)

    (r,) = DetectionGate(on_cfg()).apply([det], rgb=None, redetect=redetect)
    assert seen[0] == ("teddy bear", None)  # from DEFAULT_SYNONYMS, no model
    assert r.reprompt_attempts == 1
    assert det.low_quality is False and det.confidence == pytest.approx(0.80)


def test_aspire_refuses_a_better_mask_of_a_different_object():
    """Retry rule 3. A higher-scoring mask somewhere else is the worst outcome."""
    det = FakeDet("plushie", 0.24, mask=plushie_mask(), aspect_ratio=3.14)
    elsewhere = np.zeros((480, 640), np.uint8)
    cv2.rectangle(elsewhere, (10, 10), (120, 40), 1, -1)

    def redetect(prompt, box):
        return FakeDet(prompt, 0.95, mask=elsewhere, aspect_ratio=3.6)

    (r,) = DetectionGate(on_cfg()).apply([det], rgb=None, redetect=redetect)
    assert det.confidence == pytest.approx(0.24)  # incumbent kept
    assert det.low_quality is True
    assert any("different object" in n for n in r.notes)


def test_aspire_stops_as_soon_as_the_trigger_clears():
    """Retry rule 4: it does not keep shopping for a bigger number."""
    det = FakeDet("plushie", 0.24, mask=plushie_mask(), aspect_ratio=3.14)
    calls = []

    def redetect(prompt, box):
        calls.append(prompt)
        return FakeDet(prompt, 0.9, mask=knife_mask(), aspect_ratio=4.6)

    DetectionGate(on_cfg(reprompt_max_attempts=4)).apply([det], rgb=None, redetect=redetect)
    assert len(calls) == 1


def test_quality_thresholds_come_from_config():
    cfg = FusionConfig.from_dict({"enabled": True, "quality": {"max_angle_swing_deg": 90.0}})
    assert cfg.quality.max_angle_swing_deg == 90.0
    q = measure_mask(bleeding_plushie_mask(), reported_aspect_ratio=2.2, th=cfg.quality)
    assert q.angle_swing_deg > 12.0  # still measured
    assert q.angle_trustworthy  # but no longer vetoed

"""The fusion gate ON the live detect path. Offline: no robot, no GPU, no SAM3.

Inertness was the last failure of this feature: detection_fusion / box_proposals
/ mask_quality were fully unit-tested and sat on no live code path at all. So
these tests do not call the gate. They call the REAL
``PerceptionMixin.detect()`` -> ``merge_detections()`` -> the executor's
``detection_map`` -> ``grasp_strategy.resolve_strategy()``, with only SAM3 and
the camera replaced, and assert the gate's numbers arrive at the far end.

The plushie case is the headline: SAM3 confidence 0.24 with a reported
aspect_ratio of 3.14 used to clear the yaw gate and earn a -68.9 deg wrist
rotation off an OBB that meant nothing.
"""

import inspect
import math
from pathlib import Path
from types import SimpleNamespace

import cv2
import numpy as np
import pytest
import yaml

from spark_real.control import grasp_strategy as gs
from spark_real.pipeline_execution import ExecutionMixin
from spark_real.pipeline_perception import PerceptionMixin
from spark_real.perception import box_proposals as bp
from spark_real.perception import detection_fusion as df
from spark_real.perception.detection_fusion import fusion_map_fields
from spark_real.perception.spark_perception import ObjectDetection
from spark_real.tests.test_grasp_strategy import _Exec

SHAPE = (480, 640)


# --- fakes: exactly the camera + SAM3 surface the real detect path touches ---


class FakeCal:
    """CameraCalibration surface used by detect()/_detect_camera."""

    width, height = SHAPE[1], SHAPE[0]
    fovy_degrees = 60.0
    position = np.array([0.0, 0.0, 1.2])
    rotation_matrix = np.eye(3)

    def __init__(self, calibrated=True):
        self.extrinsic = np.eye(4)
        if calibrated:
            self.extrinsic[2, 3] = 1.2
        self.intrinsic_matrix = np.array(
            [[500.0, 0.0, 320.0], [0.0, 500.0, 240.0], [0.0, 0.0, 1.0]]
        )


def plushie_mask():
    """Round blob with two limb lumps: no real major axis anywhere in it."""
    m = np.zeros(SHAPE, np.uint8)
    cv2.ellipse(m, (320, 240), (70, 58), 0, 0, 360, 1, -1)
    cv2.circle(m, (250, 215), 22, 1, -1)
    cv2.circle(m, (392, 262), 20, 1, -1)
    return m


def knife_mask(angle_deg=-31.0):
    m = np.zeros(SHAPE, np.uint8)
    cv2.fillPoly(m, [cv2.boxPoints(((320, 240), (220, 26), angle_deg)).astype(np.int32)], 1)
    return m


def make_det(label, conf, mask, ar, angle_deg, pos):
    ys, xs = np.nonzero(mask)
    d = ObjectDetection(
        label=label,
        confidence=conf,
        centroid_2d=(float(xs.mean()), float(ys.mean())),
        mask_area=int(mask.sum()),
        depth_meters=1.0,
        position_3d=np.array(pos, dtype=float),
        bbox=(float(xs.min()), float(ys.min()), float(xs.max()), float(ys.max())),
        aspect_ratio=ar,
        orientation_angle=float(np.deg2rad(angle_deg)),
        obb_minor_m=0.04,
    )
    d.mask = mask
    return d


class FakeSAM3:
    """Stands in for SPARKPerception. Records every prompt list it is asked."""

    def __init__(self, per_prompt):
        self.per_prompt = per_prompt  # prompt -> list of ObjectDetection factories
        self.calls = []

    def _detect_with_rendered_depth(self, rgb=None, depth=None, prompts=(), **kw):
        self.calls.append(list(prompts))
        out = []
        for p in prompts:
            for factory in self.per_prompt.get(p, ()):
                out.append(factory())
        return out

    def detect(self, **kw):  # no-hardware-depth branch
        return self._detect_with_rendered_depth(**kw)


class LivePipeline(PerceptionMixin):
    """The real PerceptionMixin over the fakes above. Nothing else replaced."""

    def __init__(self, perception, config, cams=("birdview",)):
        self._perception = perception
        self.config = config
        self.profile = None
        self._cams = cams

    def captures(self):
        rgb = np.zeros((*SHAPE, 3), np.uint8)
        rgb[:] = 40
        return {
            c: {"rgb": rgb, "depth": np.full(SHAPE, 1.0, np.float32), "calibration": FakeCal()}
            for c in self._cams
        }


class Config:
    robot_family = "ur10e"
    use_hardware_depth = True
    table_height = -0.08


@pytest.fixture
def fusion_on(monkeypatch):
    monkeypatch.setenv("SPARK_DETECTION_FUSION", "1")
    monkeypatch.setenv("SPARK_BOX_PROPOSER", "null")  # no model needed
    return True


def executor_map(pipeline_detections):
    """Build the dict the executor consumes, the way execute() builds it.

    The fusion keys come from the same fusion_map_fields() call that
    pipeline_execution.execute() makes -- see test_execute_publishes_fusion_keys,
    which fails if that call is ever removed again.
    """
    out = {}
    for det in pipeline_detections:
        out[det.label] = {
            "position_3d": det.position_3d.tolist(),
            "confidence": det.confidence,
            "orientation_angle": det.orientation_angle,
            "aspect_ratio": det.aspect_ratio,
            "obb_minor_m": float(det.obb_minor_m or 0.0),
            "low_quality": bool(getattr(det, "low_quality", False)),
            "reprompt_attempts": int(getattr(det, "reprompt_attempts", 0) or 0),
            "_mask": det.mask,
            "_camera": getattr(det, "camera", None),
        }
        out[det.label].update(fusion_map_fields(det))
    return out


# --- 1. the gate is wired into the shipped dict builder ---------------------


def test_execute_publishes_fusion_keys():
    """Anti-inertness guard on the shared seam: execute() must call the emitter."""
    src = inspect.getsource(ExecutionMixin.execute)
    assert "fusion_map_fields(det)" in src, (
        "pipeline_execution.execute() no longer publishes the fusion stamps; "
        "grasp_strategy's yaw gate is blind again"
    )


# --- 2. the live detect path stamps, and the stamps reach the executor ------


def test_detect_stamps_fused_confidence(fusion_on):
    """REAL detect() -> the detections it returns carry the gate's verdict."""
    sam3 = FakeSAM3(
        {
            "plushie": (
                lambda: make_det("plushie", 0.24, plushie_mask(), 3.14, -68.9, (-0.9, 0.0, -0.20)),
            )
        }
    )
    pipe = LivePipeline(sam3, Config())
    dets = pipe.detect(pipe.captures(), ["plushie"])

    assert len(dets) == 1
    d = dets[0]
    assert hasattr(d, "fused_confidence"), "the gate never ran on the live path"
    assert d.fused_confidence == pytest.approx(0.24, abs=1e-6)  # null proposer abstains
    assert d.axis_trust == 0.0
    assert d.low_quality is True


def test_detection_map_carries_fusion_fields(fusion_on):
    """detect() -> merge_detections() -> the dict the executor reads."""
    sam3 = FakeSAM3(
        {
            "plushie": (
                lambda: make_det("plushie", 0.24, plushie_mask(), 3.14, -68.9, (-0.9, 0.0, -0.20)),
            ),
            "knife": (
                lambda: make_det("knife", 0.72, knife_mask(), 3.9, -31.0, (-0.7, 0.2, -0.22)),
            ),
        }
    )
    pipe = LivePipeline(sam3, Config())
    merged = pipe.merge_detections(pipe.detect(pipe.captures(), ["plushie", "knife"]))
    dmap = executor_map(merged)

    assert "fused_confidence" in dmap["plushie"]
    assert "axis_trust" in dmap["plushie"]
    assert dmap["plushie"]["axis_trust"] == 0.0
    # A real elongated tool keeps a usable axis: the gate is a filter, not a veto.
    assert dmap["knife"]["axis_trust"] > 0.0
    assert dmap["knife"]["low_quality"] is False


def test_fusion_off_publishes_nothing(monkeypatch):
    """OFF is byte-identical: no stamps, no keys, no behaviour change.

    The off-state is pinned via the env var rather than inherited from the
    shipped YAML. _fusion_settings() falls back to reading
    configs/ur10e_default.yaml when no profile is loaded, so this test used to
    pass only because that file happened to ship `enabled: false` -- flipping
    the shipped default to true broke it. What is under test here is the
    behaviour when the gate is off, not what the default happens to be; the
    default is pinned separately in test_shipped_default_enables_fusion.
    """
    monkeypatch.setenv("SPARK_DETECTION_FUSION", "0")
    sam3 = FakeSAM3(
        {
            "plushie": (
                lambda: make_det("plushie", 0.24, plushie_mask(), 3.14, -68.9, (-0.9, 0.0, -0.20)),
            )
        }
    )
    pipe = LivePipeline(sam3, Config())
    assert pipe.detection_gate() is None
    dmap = executor_map(pipe.merge_detections(pipe.detect(pipe.captures(), ["plushie"])))
    assert "fused_confidence" not in dmap["plushie"]
    assert "axis_trust" not in dmap["plushie"]
    assert dmap["plushie"]["low_quality"] is False


# --- 3. the plushie earns no wrist rotation, end to end --------------------


def _plushie_verdict(fusion_enabled, monkeypatch):
    monkeypatch.setenv("SPARK_DETECTION_FUSION", "1" if fusion_enabled else "0")
    monkeypatch.setenv("SPARK_BOX_PROPOSER", "null")
    sam3 = FakeSAM3(
        {
            "plushie": (
                lambda: make_det("plushie", 0.24, plushie_mask(), 3.14, -68.9, (-0.9, 0.0, -0.20)),
            )
        }
    )
    pipe = LivePipeline(sam3, Config())
    merged = pipe.merge_detections(pipe.detect(pipe.captures(), ["plushie"]))
    dmap = executor_map(merged)
    ex = _Exec(dmap)
    ex._pipeline = pipe  # what the executor is really handed
    orient, strategy = gs.resolve_grasp_orientation({}, dmap["plushie"], ex)
    yaw = float(np.rad2deg(gs.measured_yaw_offset(orient, ex.GRASP_ORIENTATION)))
    return dmap["plushie"], strategy, yaw


def test_plushie_earns_no_yaw_through_the_live_path(monkeypatch):
    det, strategy, yaw = _plushie_verdict(True, monkeypatch)
    assert strategy == "topdown"
    assert abs(yaw) < 1e-6, f"a 0.24-confidence blob still rotated the wrist {yaw:.1f} deg"
    # ...and for the right reason, not by accident of the AR gate.
    assert det["aspect_ratio"] > _Exec.GRASP_YAW_AR_GATE
    assert det["low_quality"] is True


def test_knife_keeps_its_yaw_through_the_live_path(fusion_on):
    """The gate must not cost a real tool its oriented grasp."""
    sam3 = FakeSAM3(
        {
            "knife": (
                lambda: make_det("knife", 0.72, knife_mask(-31.0), 3.9, -31.0, (-0.7, 0.2, -0.22)),
            )
        }
    )
    pipe = LivePipeline(sam3, Config())
    dmap = executor_map(pipe.merge_detections(pipe.detect(pipe.captures(), ["knife"])))
    ex = _Exec(dmap)
    ex._pipeline = pipe
    orient, strategy = gs.resolve_grasp_orientation({}, dmap["knife"], ex)
    yaw = float(np.rad2deg(gs.measured_yaw_offset(orient, ex.GRASP_ORIENTATION)))
    assert strategy == "obb"
    assert abs(yaw - (-31.0)) < 2.0, yaw


# --- 4. ASPIRE reprompt runs through the REAL detect code ------------------


def test_aspire_reprompts_through_real_sam3_path(fusion_on):
    """A low-quality mask triggers a real second SAM3 pass on an alt prompt."""
    sam3 = FakeSAM3(
        {
            "plushie": (
                lambda: make_det("plushie", 0.24, plushie_mask(), 3.14, -68.9, (-0.9, 0.0, -0.20)),
            ),
            # a much better mask of the SAME object, under a synonym
            "teddy bear": (
                lambda: make_det(
                    "teddy bear", 0.88, knife_mask(-10.0), 1.9, -10.0, (-0.9, 0.0, -0.20)
                ),
            ),
        }
    )
    pipe = LivePipeline(sam3, Config())
    dets = pipe.detect(pipe.captures(), ["plushie"])

    assert sam3.calls[0] == ["plushie"]
    assert ["teddy bear"] in sam3.calls[1:], sam3.calls
    d = dets[0]
    assert d.reprompt_attempts >= 1
    assert d.confidence == pytest.approx(0.88)  # the repaired mask was adopted
    assert d.axis_trust > 0.0 and d.low_quality is False


# --- 5. a configured-but-missing detector fails LOUDLY --------------------


def test_missing_detector_raises_not_abstains(monkeypatch):
    """Never silently pretend to have a detector.

    Absence is SIMULATED rather than borrowed from the environment. This used
    to name `rfdetr` on the assumption it was not installed; once it was
    actually installed the backend built fine, the test stopped raising, and
    what it guards -- that a configured-but-unbuildable detector fails loudly
    instead of abstaining -- silently stopped being checked.
    """
    import sys

    monkeypatch.setenv("SPARK_DETECTION_FUSION", "1")
    monkeypatch.setenv("SPARK_BOX_PROPOSER", "rfdetr")
    # Make `import rfdetr` raise ImportError regardless of what is installed.
    monkeypatch.setitem(sys.modules, "rfdetr", None)
    pipe = LivePipeline(FakeSAM3({}), Config())
    with pytest.raises(RuntimeError, match="box proposer is unusable"):
        pipe.detect(pipe.captures(), ["plushie"])
    # and it keeps raising rather than quietly running unprotected
    with pytest.raises(RuntimeError, match="box proposer is unusable"):
        pipe.detection_gate()


# --- 6. seam B: axis_trust travels with the axis it scored ----------------


def test_obb_enrichment_carries_donor_axis_trust(fusion_on, monkeypatch):
    """A donor camera's angle must not arrive wearing the base's trust score."""
    # Base = sideview, OBB donor = birdview (the only enrichment-trusted camera,
    # which has to be the non-primary one for enrichment to run at all).
    monkeypatch.setenv("SPARK_PRIMARY_CAM", "sideview")
    sam3 = FakeSAM3(
        {
            # sideview: honest round blob. birdview: same object, reported long.
            "plushie": (
                lambda: make_det("plushie", 0.60, plushie_mask(), 1.1, 0.0, (-0.9, 0.0, -0.20)),
            ),
        }
    )
    pipe = LivePipeline(sam3, Config(), cams=("sideview", "birdview"))
    all_dets = pipe.detect(pipe.captures(), ["plushie"])
    # Make birdview the OBB donor with a fabricated long axis, as the rig does.
    for d in all_dets:
        if d.camera == "birdview":
            d.aspect_ratio, d.orientation_angle = 3.14, math.radians(-68.9)
            d.axis_trust = 0.0
        else:
            d.axis_trust = 0.9
    merged = pipe.merge_detections(all_dets)
    assert len(merged) == 1
    assert merged[0].camera == "sideview"
    assert merged[0].aspect_ratio == pytest.approx(3.14)  # donor's axis adopted
    assert merged[0].axis_trust == 0.0, "the base's trust survived onto the donor's axis"


# --- 7. the mask backstop: a detection that never passed the perception seam --


def test_click_route_detection_still_vetoed_by_mask(fusion_on):
    """A click/box-route detection carries a mask but no gate stamps.

    grasp_strategy measures that mask itself when fusion is on, so the veto
    does not depend on which route produced the detection. Note low_quality is
    False here: this is the axis_trust channel vetoing on its own.
    """
    sam3 = FakeSAM3({})
    pipe = LivePipeline(sam3, Config())
    assert pipe.detection_gate() is not None

    clicked = {
        "position_3d": [-0.9, 0.0, -0.20],
        "confidence": 0.60,  # comfortably over yaw_min_conf
        "aspect_ratio": 3.14,  # over the AR gate
        "orientation_angle": math.radians(-68.9),
        "low_quality": False,  # never stamped: no gate ran on this route
        "_mask": plushie_mask(),
    }
    ex = _Exec({"plushie": clicked})
    ex._pipeline = pipe
    strategy, reason = gs.resolve_strategy({}, clicked, ex)
    assert strategy == "topdown"
    assert "axis_trust(mask)" in reason, reason


def test_click_route_backstop_is_off_when_fusion_is_off(monkeypatch):
    """No gate on the pipeline -> no mask measurement, no behaviour change."""
    monkeypatch.setenv("SPARK_DETECTION_FUSION", "0")
    pipe = LivePipeline(FakeSAM3({}), Config())
    assert pipe.detection_gate() is None
    clicked = {
        "position_3d": [-0.9, 0.0, -0.20],
        "confidence": 0.60,
        "aspect_ratio": 3.14,
        "orientation_angle": math.radians(-68.9),
        "low_quality": False,
        "_mask": plushie_mask(),
    }
    ex = _Exec({"plushie": clicked})
    ex._pipeline = pipe
    assert gs.resolve_strategy({}, clicked, ex)[0] == "obb"


def test_shipped_default_enables_fusion(monkeypatch):
    """The shipped ur10e YAML must leave the fusion gate ON.

    This is the knob that decides whether RF-DETR is actually consulted on the
    live path. It shipped wired-but-disabled once already, which made the
    plushie fix (SAM3 conf 0.24, aspect_ratio 3.14 -> a 69 deg wrist rotation
    off a meaningless OBB) inert for anyone launching the server normally.
    Pinned here so it cannot silently regress to false again.
    """
    monkeypatch.delenv("SPARK_DETECTION_FUSION", raising=False)
    monkeypatch.setenv("SPARK_BOX_PROPOSER", "null")
    pipe = LivePipeline(FakeSAM3({}), Config())
    assert pipe._fusion_settings().get("enabled") is True
    assert pipe.detection_gate() is not None


# --- 8. per-label gating: the proposer only votes where it is competent -----
#
# Every number below is MEASURED, not invented. Source:
#   /data/spark_episodes/"pick up the plushie and place in the bowl"/episode_0002
# with the shipped proposer config (rfdetr medium, cpu, threshold 0.35) run on
# the real JPEGs, and the SAM3 masks taken from the offline harvest in
# output/pvh_mask_cache. Boxes and mask bboxes were computed in that harvest's
# 240x320 mask space and are doubled here into the 640x480 frame these fakes
# use, so every IoU below is the one the rig computed.
#
# THE FAILURE THIS SECTION PINS. "plushie" is not a COCO class, but it resolves
# into COCO through box_proposals.DEFAULT_SYNONYMS["teddy bear"], so
# ProposalSet.covers() calls it in-vocabulary. RF-DETR then boxed no teddy bear
# on either camera, the gate read that silence as DISAGREEMENT, and a correct
# SAM3 mask was docked miss_penalty: 0.95 -> 0.92 on sideview and 0.20 -> 0.14
# on birdview (under the verify gate's 0.35 min_conf). The bowl, on the very
# same frames, is the case the feature exists for: 0.92 -> 0.98, 0.96 -> 0.99.

# The real closed set. rfdetr.assets.coco_classes.COCO_CLASSES, lowercased.
COCO80 = frozenset(
    [
        "person", "bicycle", "car", "motorcycle", "airplane", "bus", "train",
        "truck", "boat", "traffic light", "fire hydrant", "stop sign",
        "parking meter", "bench", "bird", "cat", "dog", "horse", "sheep", "cow",
        "elephant", "bear", "zebra", "giraffe", "backpack", "umbrella",
        "handbag", "tie", "suitcase", "frisbee", "skis", "snowboard",
        "sports ball", "kite", "baseball bat", "baseball glove", "skateboard",
        "surfboard", "tennis racket", "bottle", "wine glass", "cup", "fork",
        "knife", "spoon", "bowl", "banana", "apple", "sandwich", "orange",
        "broccoli", "carrot", "hot dog", "pizza", "donut", "cake", "chair",
        "couch", "potted plant", "bed", "dining table", "toilet", "tv",
        "laptop", "mouse", "remote", "keyboard", "cell phone", "microwave",
        "oven", "toaster", "sink", "refrigerator", "book", "clock", "vase",
        "scissors", "teddy bear", "hair drier", "toothbrush",
    ]
)

# What RF-DETR really returned, per frame. camera_0 == sideview here.
BOXES_SIDE_F60 = [
    ("bowl", 0.939, (319.8, 288.2, 382.2, 342.6)),
    ("dining table", 0.736, (109.6, 229.0, 514.2, 471.0)),
]
BOXES_BIRD_F60 = [
    ("bowl", 0.913, (298.6, 300.4, 359.2, 377.2)),
    ("dining table", 0.595, (46.4, 187.6, 617.4, 471.4)),
]
# NOTHING on this frame. Not a weak teddy bear, not a fire hydrant: no box.
BOXES_BIRD_F188: list = []

# tight bbox of the REAL SAM3 mask for each (camera, object)
MASKBOX_SIDE_PLUSHIE = (138, 358, 250, 480)  # camera_0 frame_0060
MASKBOX_SIDE_BOWL = (330, 290, 392, 346)  # camera_0 frame_0060
MASKBOX_BIRD_PLUSHIE = (290, 296, 348, 384)  # camera_1 frame_0188
MASKBOX_BIRD_BOWL = (290, 318, 352, 400)  # camera_1 frame_0060

# The four live SAM3 confidences the operator reported today.
PLUSHIE_SIDE, PLUSHIE_BIRD = 0.95, 0.20
BOWL_SIDE, BOWL_BIRD = 0.92, 0.96
# ...and what the miss penalty did to the two plushie numbers.
PLUSHIE_SIDE_PENALISED, PLUSHIE_BIRD_PENALISED = 0.9223, 0.1351
# ...and what agree_class correctly does for the two bowl numbers.
BOWL_SIDE_FUSED, BOWL_BIRD_FUSED = 0.9833, 0.9899


class ReplayProposer(bp.BoxProposer):
    """The real RF-DETR pass, replayed box for box, over the real COCO vocab.

    Only the MODEL is replaced -- exactly what FakeSAM3 does for SAM3. The
    gate, its config resolution, its vocabulary check and its arithmetic are
    all the shipped ones.
    """

    name = "replay"

    def __init__(self, state):
        self._state = state  # {"boxes": [...]}, read at propose() time

    def vocabulary(self):
        return COCO80

    def propose(self, rgb, labels=()):
        return bp.ProposalSet(
            [bp.BoxProposal(*b, self.name) for b in self._state["boxes"]],
            COCO80,
            True,
            self.name,
        )


def rect_mask(bbox):
    """A mask whose tight bbox is exactly `bbox` -- all fuse_one associates on.

    The gate matches boxes against mask_bbox(), so a rectangle carrying the
    real mask's extent reproduces the real IoU exactly. Mask SHAPE only feeds
    axis_trust, which section 3 already covers and this section does not test.
    """
    x1, y1, x2, y2 = (int(v) for v in bbox)
    m = np.zeros(SHAPE, np.uint8)
    m[y1:y2, x1:x2] = 1
    return m


def _live_fuse(monkeypatch, camera, specs, boxes, profile_raw=None):
    """REAL detect() on one camera with the RF-DETR pass replayed.

    `specs` is [(label, sam3_conf, mask_bbox)]. Returns the detections by label
    plus the proposer config block the gate was actually built with.
    """
    monkeypatch.setenv("SPARK_DETECTION_FUSION", "1")
    monkeypatch.delenv("SPARK_BOX_PROPOSER", raising=False)
    state = {"boxes": boxes, "cfg": None}

    def _build(cfg):
        state["cfg"] = dict(cfg or {})
        return ReplayProposer(state)

    monkeypatch.setattr(df, "build_proposer", _build)

    per_prompt = {
        label: (
            lambda label=label, conf=conf, bbox=bbox: make_det(
                label, conf, rect_mask(bbox), 1.1, 0.0, (-0.9, 0.0, -0.20)
            ),
        )
        for label, conf, bbox in specs
    }
    pipe = LivePipeline(FakeSAM3(per_prompt), Config(), cams=(camera,))
    if profile_raw is not None:
        pipe.profile = SimpleNamespace(raw=profile_raw)
    dets = pipe.detect(pipe.captures(), [s[0] for s in specs])
    return {d.label: d for d in dets}, state["cfg"]


def _shipped_raw():
    path = Path(df.__file__).resolve().parents[1] / "configs" / "ur10e_default.yaml"
    return yaml.safe_load(path.read_text())


def _with_labels(labels):
    """The shipped config with the allowlist overridden, through the REAL
    overlay path the server uses (profile.raw). Nothing here edits the YAML."""
    raw = _shipped_raw()
    prop = raw["perception"]["fusion"].setdefault("proposer", {})
    if labels is None:
        prop.pop("labels", None)
    else:
        prop["labels"] = list(labels)
    return raw


def test_shipped_yaml_gates_the_proposer_to_competent_labels():
    """The allowlist ships, and says the right thing.

    Pinned because an allowlist that is not in the shipped YAML is an
    allowlist that protects nobody -- the same inertness this whole file
    exists to prevent.
    """
    prop = _shipped_raw()["perception"]["fusion"]["proposer"]
    labels = {str(x).lower() for x in (prop.get("labels") or [])}
    assert labels, "perception.fusion.proposer.labels is missing from the shipped YAML"
    assert {"fork", "knife", "spoon", "bowl", "cup", "bottle", "wine glass", "scissors"} <= labels
    # the measured failure must NOT be on it
    assert "plushie" not in labels and "teddy bear" not in labels
    # nor the whole-table box RF-DETR emits on every frame
    assert "dining table" not in labels


@pytest.mark.parametrize(
    "camera,conf,maskbox,boxes",
    [
        ("sideview", PLUSHIE_SIDE, MASKBOX_SIDE_PLUSHIE, BOXES_SIDE_F60),
        ("birdview", PLUSHIE_BIRD, MASKBOX_BIRD_PLUSHIE, BOXES_BIRD_F188),
    ],
)
def test_plushie_abstains_on_the_live_path(monkeypatch, camera, conf, maskbox, boxes):
    """THE HEADLINE. RF-DETR is not consulted about plushies, so its silence
    costs SAM3 nothing: fused == sam3, bit for bit."""
    dets, cfg = _live_fuse(monkeypatch, camera, [("plushie", conf, maskbox)], boxes)
    d = dets["plushie"]
    assert cfg.get("labels"), "the gate was built without the allowlist"
    assert d.confidence == conf
    assert d.fused_confidence == conf  # exact, not approx: no penalty, no bonus
    assert d.fused_confidence == d.confidence


@pytest.mark.parametrize(
    "camera,conf,maskbox,boxes,expected",
    [
        ("sideview", BOWL_SIDE, MASKBOX_SIDE_BOWL, BOXES_SIDE_F60, BOWL_SIDE_FUSED),
        ("birdview", BOWL_BIRD, MASKBOX_BIRD_BOWL, BOXES_BIRD_F60, BOWL_BIRD_FUSED),
    ],
)
def test_bowl_still_earns_its_corroboration(monkeypatch, camera, conf, maskbox, boxes, expected):
    """Gating must not turn the feature off: a label the detector IS good at
    keeps the whole agree_class bonus."""
    dets, _ = _live_fuse(monkeypatch, camera, [("bowl", conf, maskbox)], boxes)
    d = dets["bowl"]
    assert d.fused_confidence == pytest.approx(expected, abs=5e-4)
    assert d.fused_confidence > d.confidence


@pytest.mark.parametrize(
    "camera,conf,maskbox,boxes,penalised",
    [
        ("sideview", PLUSHIE_SIDE, MASKBOX_SIDE_PLUSHIE, BOXES_SIDE_F60, PLUSHIE_SIDE_PENALISED),
        ("birdview", PLUSHIE_BIRD, MASKBOX_BIRD_PLUSHIE, BOXES_BIRD_F188, PLUSHIE_BIRD_PENALISED),
    ],
)
def test_empty_allowlist_keeps_todays_behaviour(
    monkeypatch, camera, conf, maskbox, boxes, penalised
):
    """Opt-in: absent or empty allowlist == exactly what shipped before."""
    for labels in (None, []):
        dets, _ = _live_fuse(
            monkeypatch,
            camera,
            [("plushie", conf, maskbox)],
            boxes,
            profile_raw=_with_labels(labels),
        )
        assert dets["plushie"].fused_confidence == pytest.approx(penalised, abs=5e-4), labels


@pytest.mark.parametrize("prompt", ["bowl", "dish", "bowls", "bowl 2"])
def test_allowlist_matches_synonyms_plurals_and_instances(monkeypatch, prompt):
    """The single entry 'bowl' must cover every phrasing SAM3 is prompted with.

    'dish' -> bowl through DEFAULT_SYNONYMS, 'bowls' -> bowl through the
    vocabulary lookup's substring rule, 'bowl 2' -> bowl through
    _strip_instance. All three are labels this rig really produces, and all
    three must stay CONSULTED (i.e. still earn the agree_class bonus).
    """
    dets, _ = _live_fuse(
        monkeypatch,
        "sideview",
        [(prompt, BOWL_SIDE, MASKBOX_SIDE_BOWL)],
        BOXES_SIDE_F60,
        profile_raw=_with_labels(["bowl"]),
    )
    assert dets[prompt].fused_confidence == pytest.approx(BOWL_SIDE_FUSED, abs=5e-4)


def test_subpart_prompt_is_gated_by_its_base_class(monkeypatch):
    """'knife handle' is what SAM3 gets asked for; it must gate as 'knife'."""
    dets, _ = _live_fuse(
        monkeypatch,
        "sideview",
        [("knife handle", PLUSHIE_SIDE, MASKBOX_SIDE_PLUSHIE)],
        BOXES_SIDE_F60,
        profile_raw=_with_labels(["knife"]),
    )
    # consulted -> RF-DETR's silence about a knife IS a disagreement here
    assert dets["knife handle"].fused_confidence == pytest.approx(
        PLUSHIE_SIDE_PENALISED, abs=5e-4
    )
    # and the same detection under a label NOT on that allowlist is untouched
    dets, _ = _live_fuse(
        monkeypatch,
        "sideview",
        [("plushie", PLUSHIE_SIDE, MASKBOX_SIDE_PLUSHIE)],
        BOXES_SIDE_F60,
        profile_raw=_with_labels(["knife"]),
    )
    assert dets["plushie"].fused_confidence == PLUSHIE_SIDE


def test_gating_survives_into_the_executor_map(monkeypatch):
    """The abstained number is the one the executor reads, not a local."""
    dets, _ = _live_fuse(
        monkeypatch,
        "birdview",
        [("plushie", PLUSHIE_BIRD, MASKBOX_BIRD_PLUSHIE)],
        BOXES_BIRD_F188,
    )
    dmap = executor_map(list(dets.values()))
    assert dmap["plushie"]["fused_confidence"] == PLUSHIE_BIRD

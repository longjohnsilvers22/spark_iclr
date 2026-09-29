"""The task-spec escalation ladder ON the UI's detect route. Offline.

THE FAILURE THIS RECONSTRUCTS
-----------------------------
Live run, "pick up the plushie and place in the bowl":

    [fusion] sideview plushie: sam3=0.94 -> fused=0.94 [abstain]
    [fusion] birdview plushie: sam3=0.09 -> fused=0.09 [abstain]
    Merge: primary=birdview (2 dets), secondary cameras: ['sideview(2)']
    merge_detections: count gate FAILED   plushie: 0 (expect 1..1)
    move_to_keypoint: FAIL - Object 'plushie' not found in detections

Top-down the plushie reads as a purple hat, so SAM3 scores it 0.09 on
birdview; the merge keeps birdview as primary, so the merged plushie carries
0.09 and the task's ``min_conf: 0.15`` drops it. The registry entry declares
``relax_secondary_score: 0.30`` and ``alt_prompts: [stuffed animal, ...]``
and NONE of it ran: the UI's three-step flow calls ``pipeline.detect()`` +
``merge_detections()`` raw, and the escalation lives in ``detect_for_task``.

So these tests do not call ``detect_for_task``. They call the REAL FastAPI
route handlers in ``spark_real.routes.detection`` with only SAM3 and the
cameras replaced, and assert the plushie survives -- plus an anti-inertness
guard (``test_detect_route_calls_detect_for_task``) that fails the moment the
route stops going through the shared ladder, which is the exact regression
that produced the log above.
"""

import asyncio
import inspect

import numpy as np
import pytest

from spark_real.perception.prompt_registry import PromptCountMismatch
from spark_real.perception.spark_perception import ObjectDetection
from spark_real.pipeline_perception import PerceptionMixin
from spark_real.routes import detection as detection_routes
from spark_real.routes import state as route_state
from spark_real.routes.models import DetectRequest

SHAPE = (240, 320)

# Each fake camera paints a distinct constant into its RGB so the fake SAM3
# can tell which view it was handed -- the whole point of the reconstruction
# is that the two views disagree about the same object.
CAM_TAG = {"birdview": 40, "sideview": 41}
TAG_CAM = {v: k for k, v in CAM_TAG.items()}


class FakeCal:
    """The CameraCalibration surface ``_detect_camera`` touches."""

    width, height = SHAPE[1], SHAPE[0]
    fovy_degrees = 60.0
    position = np.array([0.0, 0.0, 1.2])
    rotation_matrix = np.eye(3)
    fx = fy = 250.0
    cx, cy = 160.0, 120.0

    def __init__(self):
        self.extrinsic = np.eye(4)
        self.extrinsic[2, 3] = 1.2
        self.intrinsic_matrix = np.array(
            [[250.0, 0.0, 160.0], [0.0, 250.0, 120.0], [0.0, 0.0, 1.0]]
        )


def _blob(cx, cy, r=28):
    m = np.zeros(SHAPE, np.uint8)
    ys, xs = np.ogrid[: SHAPE[0], : SHAPE[1]]
    m[(xs - cx) ** 2 + (ys - cy) ** 2 <= r * r] = 1
    return m


def _det(label, conf, pos, px):
    mask = _blob(*px)
    ys, xs = np.nonzero(mask)
    d = ObjectDetection(
        label=label,
        confidence=conf,
        centroid_2d=(float(xs.mean()), float(ys.mean())),
        mask_area=int(mask.sum()),
        depth_meters=1.0,
        position_3d=np.array(pos, dtype=float),
        bbox=(float(xs.min()), float(ys.min()), float(xs.max()), float(ys.max())),
    )
    d.mask = mask
    return d


class FakeSAM3:
    """Stands in for SPARKPerception. ``table`` is {(camera, prompt): [rows]}."""

    def __init__(self, table):
        self.table = table
        self.calls = []  # (camera, prompts, secondary_score)

    def _detect_with_rendered_depth(self, rgb=None, depth=None, prompts=(), **kw):
        cam = TAG_CAM[int(rgb[0, 0, 0])]
        self.calls.append((cam, list(prompts), kw.get("secondary_score")))
        out = []
        for p in prompts:
            for conf, pos, px in self.table.get((cam, p), ()):
                out.append(_det(p, conf, pos, px))
        return out

    def detect(self, **kw):
        return self._detect_with_rendered_depth(**kw)


class Config:
    robot_family = "ur10e"
    use_hardware_depth = True
    table_height = -0.08
    task_prompts_dir = None


class LivePipeline(PerceptionMixin):
    """The real PerceptionMixin over fake cameras + fake SAM3."""

    _planner = None

    def __init__(self, perception, cams=("birdview", "sideview")):
        self._perception = perception
        self.config = Config()
        self.profile = None
        self._cams = cams

    def capture(self):
        out = {}
        for c in self._cams:
            rgb = np.full((*SHAPE, 3), CAM_TAG[c], np.uint8)
            out[c] = {
                "rgb": rgb,
                "depth": np.full(SHAPE, 1.0, np.float32),
                "calibration": FakeCal(),
            }
        return out

    class _Activity:
        def __enter__(self):
            return self

        def __exit__(self, *a):
            return False

    def activity(self, name):
        return self._Activity()


PLUSHIE_PX = (110, 120)
BOWL_PX = (215, 130)
PLUSHIE_XY = (-0.90, 0.00, -0.20)
BOWL_XY = (-0.60, 0.00, -0.20)


def plushie_scene(alt_conf=0.62):
    """The live scene: birdview 0.09 / sideview 0.94 on 'plushie'.

    ``alt_conf`` is what birdview scores the alt prompt "stuffed animal" --
    above the task's 0.15 min_conf, which is why rung 2 rescues the run.
    """
    return {
        ("birdview", "plushie"): [(0.09, PLUSHIE_XY, PLUSHIE_PX)],
        ("sideview", "plushie"): [(0.94, PLUSHIE_XY, PLUSHIE_PX)],
        ("birdview", "bowl"): [(0.88, BOWL_XY, BOWL_PX)],
        ("sideview", "bowl"): [(0.71, BOWL_XY, BOWL_PX)],
        ("birdview", "stuffed animal"): [(alt_conf, PLUSHIE_XY, PLUSHIE_PX)],
        ("sideview", "stuffed animal"): [(0.90, PLUSHIE_XY, PLUSHIE_PX)],
    }


@pytest.fixture
def wired(monkeypatch):
    """Install a LivePipeline as the server's pipeline, restore afterwards.

    The detection-fusion gate is forced OFF: it loads RF-DETR and issues its
    own ASPIRE reprompt passes, neither of which this file is about. On the
    live run it abstained on both plushie views anyway (``[abstain]`` in the
    log), so the escalation ladder is what was left to do the work.
    """
    monkeypatch.setenv("SPARK_DETECTION_FUSION", "0")

    def _install(table):
        sam3 = FakeSAM3(table)
        pipe = LivePipeline(sam3)
        monkeypatch.setattr(route_state, "pipeline", pipe, raising=False)
        return pipe, sam3

    return _install


INSTRUCTION = "pick up the plushie and place it in the bowl"


# --- 1. anti-inertness: the route must go through the shared ladder --------


def test_detect_route_calls_detect_for_task():
    """
    Guard on the seam. ``/api/detect`` calling detect()+merge_detections()
    directly is precisely how the escalation went missing on the live run;
    a second copy of the ladder would also fail this on review.
    """
    src = inspect.getsource(detection_routes.detect_objects)
    assert "detect_for_task" in src, (
        "/api/detect no longer routes registered tasks through "
        "PerceptionMixin.detect_for_task; the relax / alt_prompts / LLM "
        "escalation is dead on the UI path again"
    )


def test_detect_approve_route_calls_detect_for_task():
    src = inspect.getsource(detection_routes.detect_for_approval)
    assert "detect_for_task" in src, (
        "/api/detect_approve no longer routes registered tasks through "
        "PerceptionMixin.detect_for_task"
    )


# --- 2. the live failure, driven through the real route -------------------


def test_detect_route_rescues_the_plushie(wired):
    """birdview 0.09 / sideview 0.94 / min_conf 0.15 -> plushie survives."""
    pipe, sam3 = wired(plushie_scene())
    resp = detection_routes.detect_objects(
            DetectRequest(prompts=[], instruction=INSTRUCTION)
        )
    assert not isinstance(resp, dict) or "error" not in resp, resp
    labels = sorted(d["label"] for d in resp["detections"])
    assert "plushie" in labels, (
        f"plushie lost on the detect route (labels={labels}); the escalation "
        "did not run"
    )
    assert "bowl" in labels

    # The ladder actually climbed: pass 1 default score, pass 2 relaxed to
    # 0.30, pass 3 with the registered alt prompts appended.
    per_pass = [c for c in sam3.calls if c[0] == "birdview"]
    assert len(per_pass) == 3, [c[1:] for c in sam3.calls]
    assert per_pass[0][2] is None
    assert per_pass[1][2] == pytest.approx(0.30)
    assert "stuffed animal" in per_pass[2][1]


def test_rescued_plushie_keeps_the_canonical_label(wired):
    """The alt prompt must not leak its own name downstream."""
    pipe, sam3 = wired(plushie_scene())
    resp = detection_routes.detect_objects(
            DetectRequest(prompts=[], instruction=INSTRUCTION)
        )
    labels = {d["label"] for d in resp["detections"]}
    assert "stuffed animal" not in labels, labels
    assert route_state.pending_detections is not None
    assert "plushie" in {d.label for d in route_state.pending_detections}


def test_detect_route_reports_count_mismatch_instead_of_silently_passing(wired):
    """
    When the whole ladder fails on an `abort` task the route must say so.
    Today it returned 200 with the object missing and the executor's search
    recovery swept the workspace at z=0.35.
    """
    table = plushie_scene()
    # Neither the canonical prompt nor the alt prompt clears min_conf.
    table[("birdview", "stuffed animal")] = [(0.05, PLUSHIE_XY, PLUSHIE_PX)]
    table[("sideview", "stuffed animal")] = [(0.05, PLUSHIE_XY, PLUSHIE_PX)]
    pipe, sam3 = wired(table)
    resp = detection_routes.detect_objects(
            DetectRequest(prompts=[], instruction=INSTRUCTION)
        )
    assert getattr(resp, "status_code", 200) == 422, resp
    import json

    body = json.loads(resp.body)
    assert body["reason"] == "prompt_count_mismatch"


# --- 3. free-form /api/detect is untouched --------------------------------


FREEFORM = {
    ("birdview", "red widget"): [(0.42, (-0.7, 0.1, -0.15), (90, 100))],
    ("sideview", "red widget"): [(0.55, (-0.7, 0.1, -0.15), (90, 100))],
}


def test_freeform_detect_never_enters_the_ladder(wired, monkeypatch):
    """Operator-supplied prompts + unregistered instruction: today's path."""
    pipe, sam3 = wired(FREEFORM)

    def _boom(*a, **kw):
        raise AssertionError("free-form /api/detect must not use detect_for_task")

    monkeypatch.setattr(type(pipe), "detect_for_task", _boom, raising=False)

    resp = detection_routes.detect_objects(
            DetectRequest(
                prompts=["red widget"], instruction="do something unregistered"
            )
        )
    assert resp["count"] == 1
    assert resp["detections"][0]["label"] == "red widget"
    # Exactly one SAM3 pass per camera, default secondary score, and only the
    # operator's prompt list.
    assert [c[1] for c in sam3.calls] == [["red widget"], ["red widget"]]
    assert {c[2] for c in sam3.calls} == {None}


def test_freeform_detect_response_is_unchanged(wired, monkeypatch):
    """
    Byte-for-byte parity with the pre-change handler: same keys, same values,
    computed here from the same primitives the old handler called.
    """
    pipe, sam3 = wired(FREEFORM)
    req = DetectRequest(prompts=["red widget"], instruction=None)
    got = detection_routes.detect_objects(req)

    # Reference = the old handler body, verbatim.
    captures = pipe.capture()
    all_detections = pipe.detect(captures, ["red widget"], multi_instance=False)
    merged = pipe.merge_detections(all_detections, spec=None)
    annotated, tiled = detection_routes.create_tiled_detection_overlay(
        captures, all_detections
    )
    want_serialized = detection_routes.serialize_detections(merged)
    for s in want_serialized:
        s["tiled_bbox"] = tiled.get(s["label"])

    assert set(got) == {"detections", "all_detections", "annotated_image", "count"}
    assert got["count"] == len(merged)
    assert got["detections"] == want_serialized
    assert got["all_detections"] == detection_routes.serialize_detections(
        all_detections
    )


def test_registered_task_with_operator_prompts_stays_free_form(wired, monkeypatch):
    """
    A registered instruction whose prompts the operator overrode is still an
    operator experiment, not the frozen contract: it must not abort on the
    contract's counts.
    """
    pipe, sam3 = wired(plushie_scene())

    def _boom(*a, **kw):
        raise AssertionError("operator-supplied prompts must not use detect_for_task")

    monkeypatch.setattr(type(pipe), "detect_for_task", _boom, raising=False)

    resp = detection_routes.detect_objects(
            DetectRequest(prompts=["bowl"], instruction=INSTRUCTION)
        )
    assert [d["label"] for d in resp["detections"]] == ["bowl"]


# --- 4. the /api/detect_approve flow --------------------------------------


def test_detect_approve_rescues_the_plushie(wired):
    from spark_real.routes.models import DetectApproveRequest

    pipe, sam3 = wired(plushie_scene())
    resp = asyncio.run(
        detection_routes.detect_for_approval(
            DetectApproveRequest(instruction=INSTRUCTION)
        )
    )
    assert getattr(resp, "status_code", 200) == 200, resp
    labels = sorted(d["label"] for d in resp["detections"])
    assert "plushie" in labels, labels
    assert "stuffed animal" in resp["prompts"], resp["prompts"]


def test_detect_approve_reports_count_mismatch(wired):
    from spark_real.routes.models import DetectApproveRequest

    table = plushie_scene()
    table[("birdview", "stuffed animal")] = [(0.05, PLUSHIE_XY, PLUSHIE_PX)]
    table[("sideview", "stuffed animal")] = [(0.05, PLUSHIE_XY, PLUSHIE_PX)]
    pipe, sam3 = wired(table)
    resp = asyncio.run(
        detection_routes.detect_for_approval(
            DetectApproveRequest(instruction=INSTRUCTION)
        )
    )
    assert getattr(resp, "status_code", 200) == 422, resp


def test_unregistered_instruction_still_reaches_the_planner(wired, monkeypatch):
    """detect_approve's LLM prompt-gen path must survive the rewiring."""
    from spark_real.routes.models import DetectApproveRequest

    pipe, sam3 = wired(FREEFORM)

    class FakePlanner:
        calls = []

        def generate_prompts(self, instruction, scene_image=None):
            FakePlanner.calls.append(instruction)
            return ["red widget"]

    monkeypatch.setattr(type(pipe), "_planner", FakePlanner(), raising=False)
    resp = asyncio.run(
        detection_routes.detect_for_approval(
            DetectApproveRequest(instruction="do something unregistered")
        )
    )
    assert FakePlanner.calls == ["do something unregistered"]
    assert resp["prompts"] == ["red widget"]


def test_unregistered_detect_approve_does_not_raise_prompt_mismatch(wired):
    """No spec means no count gate, exactly as before."""
    from spark_real.routes.models import DetectApproveRequest

    pipe, sam3 = wired(FREEFORM)
    try:
        resp = asyncio.run(
            detection_routes.detect_for_approval(
                DetectApproveRequest(
                    instruction="do something unregistered", prompts=["red widget"]
                )
            )
        )
    except PromptCountMismatch as exc:  # pragma: no cover - regression guard
        pytest.fail(f"unregistered task hit the count gate: {exc}")
    assert resp["merged_count"] == 1

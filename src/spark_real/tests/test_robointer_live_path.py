"""The RoboInter plan extension ON the live plan/execute path.

Offline: no robot, no GPU, no SAM3, no network. Only the LLM client and the
camera calibrations are faked.

Inertness was this feature's failure mode, exactly as it was the detection
gate's: ``planning/robointer*.py`` shipped with four unit-test files and

    grep -rn robointer src/spark_real/ --include=*.py \\
        | grep -vE 'planning/robointer|tests/'

returned NOTHING. The representations could not reach Gemini, the reply could
not reach the executor, and the shipped YAML had no switch to turn any of it
on. So these tests do not call the RoboInter modules in isolation -- that is
what let them sit dead. They call the REAL ``ExecutionMixin.plan()`` ->
``SPARKPlanner.generate_score()`` -> ``parse_response()`` and the REAL
``ExecutionMixin.execute()`` -> ``detection_map`` ->
``MotionMixin._move_to_keypoint()``, and assert the geometry arrives at both
ends.

The headline case is the handle-vs-centroid problem: SAM3's mask centroid for
a knife sits mid-blade, so the jaws close on the blade. A ``contact_point``
read off the image moves the grasp 5 cm onto the handle -- and a contact point
that lands on a DIFFERENT object 40 cm away is refused, because an LLM pixel
is only ever accepted as a bounded correction to a measured detection.
"""

from __future__ import annotations

import inspect
from types import SimpleNamespace

import numpy as np
import pytest
import yaml

from spark_real.calibration import CameraCalibration
from spark_real.control import executor_motion
from spark_real.pipeline_execution import ExecutionMixin
from spark_real.planning import robointer_gate
from spark_real.planning.robointer import ANNOTATION_KEY, Point2D, RoboInterSchemaError
from spark_real.planning.robointer_prompt import ROBOINTER_PROMPT_SECTION
from spark_real.planning.spark_planner import SPARKPlanner, parse_plan_yaml
from spark_real.tests.test_robointer_geometry import TABLE_Z, birdview

SHAPE = (480, 640)  # h, w

KNIFE_CENTROID = np.array([-0.80, 0.10, TABLE_Z])  # what SAM3 measured
KNIFE_HANDLE = np.array([-0.85, 0.10, TABLE_Z])  # 5 cm along -X
OTHER_OBJECT = np.array([-0.50, 0.35, TABLE_Z])  # 40 cm away
TRAY = np.array([-0.60, -0.20, TABLE_Z])


# --- fakes: the LLM client and the two camera calibrations, nothing else ----


def cal_from_model(model) -> CameraCalibration:
    """The rig's CameraCalibration for one of the test CameraModels."""
    return CameraCalibration(
        name=model.name,
        width=model.width,
        height=model.height,
        fx=model.fx,
        fy=model.fy,
        cx=model.cx,
        cy=model.cy,
        extrinsic=model.extrinsic.copy(),
    )


class FakeGemini:
    """Stands in for genai.Client. Records every prompt it is handed."""

    def __init__(self, reply_yaml: str):
        self.reply_yaml = reply_yaml
        self.prompts = []
        self.images = []
        outer = self

        class _Models:
            def generate_content(self, model=None, contents=None, config=None):
                if isinstance(contents, (list, tuple)):
                    outer.prompts.append(contents[0])
                    outer.images.extend(contents[1:])
                else:
                    outer.prompts.append(contents)
                return SimpleNamespace(text=outer.reply_yaml)

        self.models = _Models()

    @property
    def prompt(self) -> str:
        assert self.prompts, "the planner never called the LLM"
        return self.prompts[-1]


class FakeExecutor:
    """The executor surface ExecutionMixin.execute() actually touches."""

    def __init__(self):
        self.detection_map = None
        self.scores = []
        self._placed_labels = set()
        self._holding = False

    def update_detections(self, dmap):
        self.detection_map = dmap

    def execute_score(self, score):
        self.scores.append(score)
        return []


def det(label, xyz, camera="birdview", bbox=(300, 200, 420, 260)):
    """A duck-typed ObjectDetection, the shape both seams read."""
    return SimpleNamespace(
        label=label,
        confidence=0.82,
        position_3d=np.asarray(xyz, dtype=float),
        centroid_2d=((bbox[0] + bbox[2]) / 2.0, (bbox[1] + bbox[3]) / 2.0),
        bbox=bbox,
        camera=camera,
        mask=None,
        mask_area=1200,
        depth_meters=1.2,
        orientation_angle=0.0,
        aspect_ratio=3.4,
        obb_minor_m=0.02,
        obb_confidence=0.7,
        low_quality=False,
        reprompt_attempts=0,
        slots=None,
        world_major_axis_rad=None,
        role_labels=None,
    )


class Config:
    robot_family = "ur10e"
    bt_plan_mode = "llm"
    use_hardware_depth = True
    table_height = TABLE_Z


class LivePipeline(ExecutionMixin):
    """The real ExecutionMixin over fake cameras and a fake LLM."""

    def __init__(self, planner=None, calibrated=True):
        self.config = Config()
        self.profile = None
        self._planner = planner
        self._bt_library = None
        self._executor = FakeExecutor()
        self._robot = None
        bv, sv = birdview(), birdview()
        self._kinect2_cal = cal_from_model(bv)  # birdview
        self._kinect_cal = cal_from_model(sv)  # sideview
        self._kinect_cal.name = "sideview"
        self._realsense_cal = None
        if not calibrated:
            self._kinect2_cal.extrinsic = np.eye(4)
            self._kinect_cal.extrinsic = np.eye(4)


def point_at(xyz) -> list:
    """A base-frame point as the 0..1000 pixel pair the planner emits."""
    cam = birdview()
    u, v = cam.base_to_pixel(xyz)
    p = Point2D.from_pixels(u, v, cam.image_size)
    return p.to_permille()


def annotated_reply(contact=None, placement=None) -> str:
    """A Gemini reply carrying RoboInter blocks, as the YAML text it arrives as."""
    pick = {
        "type": "move_to_keypoint",
        "params": {"keypoint_label": "knife handle 1", "offset_z": 0},
    }
    if contact is not None:
        pick[ANNOTATION_KEY] = {
            "subtask": "grip the knife by the handle",
            "primitive_skill": "pick",
            "label": "knife handle 1",
            "camera": "birdview",
            "contact_point": point_at(contact),
        }
    place = {
        "type": "move_to_keypoint",
        "params": {"keypoint_label": "tray", "offset_z": 0.1},
    }
    if placement is not None:
        x, y = point_at(placement)
        pad = 40
        place[ANNOTATION_KEY] = {
            "primitive_skill": "place",
            "label": "tray",
            "camera": "birdview",
            "placement_proposal": [[x - pad, y - pad], [x + pad, y + pad]],
        }
    return yaml.safe_dump(
        {
            "task": "put the knife in the tray",
            "tree": {
                "type": "sequence",
                "children": [
                    pick,
                    {"type": "grasp", "params": {"force": 60}},
                    place,
                    {"type": "release", "params": {}},
                ],
            },
        },
        sort_keys=False,
    )


PLAIN_REPLY = (
    "task: put the knife in the tray\n"
    "tree:\n"
    "  type: sequence\n"
    "  children:\n"
    '  - type: move_to_keypoint\n    params: { keypoint_label: "knife handle 1" }\n'
    "  - type: grasp\n    params: { force: 60 }\n"
)


def make_planner(reply: str):
    planner = SPARKPlanner(llm_backend="gemini", api_key="offline", robot_family="ur10e")
    planner._client = FakeGemini(reply)
    return planner


@pytest.fixture
def robointer_on(monkeypatch):
    monkeypatch.setenv(robointer_gate.ROBOINTER_ENV, "1")
    return True


@pytest.fixture
def robointer_off(monkeypatch):
    monkeypatch.setenv(robointer_gate.ROBOINTER_ENV, "0")
    return True


# --- 1. anti-inertness: the seams are IN the live functions ----------------


def test_plan_calls_the_robointer_prompt_seam():
    """ExecutionMixin.plan() must build the RoboInter planner context.

    Without this call the schema section and the observed 2D geometry never
    reach Gemini, and every RoboInter representation is dead code again.
    """
    src = inspect.getsource(ExecutionMixin.plan)
    assert "robointer_config()" in src
    assert "robointer_gate.planner_context(" in src, (
        "pipeline_execution.plan() no longer builds the RoboInter context; "
        "the planner cannot answer with geometry it was never shown"
    )


def test_execute_calls_the_robointer_consume_seam():
    """ExecutionMixin.execute() must publish the resolved corrections."""
    src = inspect.getsource(ExecutionMixin.execute)
    assert "robointer_gate.apply_to_detection_map(" in src, (
        "pipeline_execution.execute() no longer consumes RoboInter "
        "annotations; the planner's contact points reach nothing"
    )


def test_move_to_keypoint_reads_the_published_key():
    """The executor must read what execute() publishes."""
    src = inspect.getsource(executor_motion.MotionMixin._move_to_keypoint)
    assert robointer_gate.CONTACT_KEY in src, (
        "executor_motion no longer reads the RoboInter contact override; "
        "the correction is published to nobody"
    )
    assert robointer_gate.PLACEMENT_KEY in src


def test_generate_score_appends_the_schema_section():
    """The planner must be able to ASK for annotations."""
    assert "robointer_context" in inspect.getsource(SPARKPlanner.generate_score)
    assert "ROBOINTER_PROMPT_SECTION" in inspect.getsource(SPARKPlanner._build_system_prompt)


# --- 2. default OFF, and OFF is byte-identical -----------------------------


def test_shipped_default_leaves_robointer_off(monkeypatch):
    """No env, packaged YAML: the extension must be OFF."""
    monkeypatch.delenv(robointer_gate.ROBOINTER_ENV, raising=False)
    assert LivePipeline().robointer_config().enabled is False


def test_unset_env_changes_nothing_end_to_end(monkeypatch):
    """The literal shipping state: no env var, no YAML block, no difference.

    Both halves in one test on purpose -- "unchanged" has to mean the prompt
    AND the detection map, since those are the two things the extension
    touches.
    """
    monkeypatch.delenv(robointer_gate.ROBOINTER_ENV, raising=False)
    planner = make_planner(PLAIN_REPLY)
    pipe = LivePipeline(planner)
    dets = [det("knife handle 1", KNIFE_CENTROID)]
    img = np.zeros((*SHAPE, 3), np.uint8)

    pipe.plan("put the knife in the tray", dets, scene_image=img)
    prompt = planner._client.prompt
    assert "SPATIAL ANNOTATIONS" not in prompt and "observed_box" not in prompt
    assert planner._client.images[-1] is not None
    # The planner image is the object it was handed, not a re-drawn copy.
    assert robointer_gate.annotate_planner_image(pipe.robointer_config(), img) is img

    pipe.execute(parse_plan_yaml(annotated_reply(contact=KNIFE_HANDLE)), dets)
    entry = pipe._executor.detection_map["knife handle 1"]
    assert not [k for k in entry if k.startswith("robointer")], entry
    assert pipe._last_robointer == []


def test_plan_prompt_is_unchanged_when_off(robointer_off):
    planner = make_planner(PLAIN_REPLY)
    pipe = LivePipeline(planner)
    dets = [det("knife handle 1", KNIFE_CENTROID)]
    pipe.plan("put the knife in the tray", dets, scene_image=np.zeros((*SHAPE, 3), np.uint8))

    prompt = planner._client.prompt
    assert "SPATIAL ANNOTATIONS" not in prompt
    assert "observed_box" not in prompt
    assert "Scene image:" not in prompt
    assert ANNOTATION_KEY not in prompt


def test_the_prompt_extension_is_purely_additive():
    """ON must be OFF plus the schema section, nothing removed or reordered."""
    planner = SPARKPlanner(llm_backend="gemini", api_key="offline", robot_family="ur10e")
    off = planner._build_system_prompt()
    assert off == planner._build_system_prompt(robointer=False)
    assert planner._build_system_prompt(robointer=True) == off + ROBOINTER_PROMPT_SECTION


def test_the_planner_image_is_the_same_object_when_off(robointer_off):
    """No grid, no copy: the planner gets the exact bytes it gets today."""
    pipe = LivePipeline()
    img = np.zeros((*SHAPE, 3), np.uint8)
    assert robointer_gate.annotate_planner_image(pipe.robointer_config(), img) is img


def test_the_planner_image_carries_the_grid_when_on(robointer_on):
    """A coordinate request needs a coordinate frame on the image."""
    pipe = LivePipeline()
    img = np.zeros((*SHAPE, 3), np.uint8)
    gridded = robointer_gate.annotate_planner_image(pipe.robointer_config(), img)
    assert gridded is not img
    assert gridded.shape == img.shape
    assert gridded.any(), "the 0..1000 grid was never drawn"


def test_execute_publishes_no_robointer_keys_when_off(robointer_off):
    pipe = LivePipeline()
    dets = [det("knife handle 1", KNIFE_CENTROID), det("tray", TRAY)]
    score = parse_plan_yaml(annotated_reply(contact=KNIFE_HANDLE, placement=TRAY))
    pipe.execute(score, dets)

    dmap = pipe._executor.detection_map
    for entry in dmap.values():
        assert not [k for k in entry if k.startswith("robointer")], entry


# --- 3. ON: the geometry reaches Gemini ------------------------------------


def test_plan_shows_the_image_size_and_the_observed_boxes(robointer_on):
    planner = make_planner(annotated_reply(contact=KNIFE_HANDLE))
    pipe = LivePipeline(planner)
    dets = [det("knife handle 1", KNIFE_CENTROID), det("tray", TRAY)]
    score = pipe.plan(
        "put the knife in the tray",
        dets,
        scene_image=np.zeros((*SHAPE, 3), np.uint8),
        scene_camera="birdview",
    )

    prompt = planner._client.prompt
    assert "SPATIAL ANNOTATIONS" in prompt, "the schema was never asked for"
    assert "Scene image: 640x480 px" in prompt
    assert "knife handle 1: observed_box=" in prompt, (
        "the model was asked for pixel coordinates without being shown one"
    )
    # ...and the annotated reply survives parsing.
    node = score["tree"]["children"][0]
    assert ANNOTATION_KEY in node
    assert node[ANNOTATION_KEY]["contact_point"]


# --- 4. ON: the reply reaches the detection map, bounded -------------------


def test_execute_publishes_the_bounded_contact_correction(robointer_on):
    pipe = LivePipeline()
    dets = [det("knife handle 1", KNIFE_CENTROID), det("tray", TRAY)]
    score = parse_plan_yaml(annotated_reply(contact=KNIFE_HANDLE))
    pipe.execute(score, dets)

    entry = pipe._executor.detection_map["knife handle 1"]
    assert robointer_gate.CONTACT_KEY in entry, "the correction reached nothing"
    got = np.asarray(entry[robointer_gate.CONTACT_KEY], dtype=float)
    assert np.allclose(got[:2], KNIFE_HANDLE[:2], atol=5e-3), got
    assert got[2] == pytest.approx(TABLE_Z), "Z must still come from perception"
    # The measured position is never overwritten: the override is additive.
    assert np.allclose(entry["position_3d"], KNIFE_CENTROID)


def test_a_contact_point_on_another_object_is_refused(robointer_on):
    pipe = LivePipeline()
    dets = [det("knife handle 1", KNIFE_CENTROID)]
    score = parse_plan_yaml(annotated_reply(contact=OTHER_OBJECT))
    pipe.execute(score, dets)

    entry = pipe._executor.detection_map["knife handle 1"]
    assert robointer_gate.CONTACT_KEY not in entry, (
        "a 40 cm 'correction' was accepted; the guard is not running"
    )


def test_a_drifted_label_still_gets_its_correction(robointer_on):
    """A cached tree says "knife handle 1"; this run's SAM3 said "knife handle".

    LabelResolvingDetectionMap already re-binds that, so the annotation must
    follow the detection it resolved to instead of abstaining on the name.
    """
    pipe = LivePipeline()
    dets = [det("knife handle", KNIFE_CENTROID)]
    pipe.execute(parse_plan_yaml(annotated_reply(contact=KNIFE_HANDLE)), dets)

    entry = pipe._executor.detection_map["knife handle 1"]
    got = np.asarray(entry[robointer_gate.CONTACT_KEY], dtype=float)
    assert np.allclose(got[:2], KNIFE_HANDLE[:2], atol=5e-3), got


def test_placement_proposal_reaches_the_container_entry(robointer_on):
    pipe = LivePipeline()
    place = TRAY + np.array([0.06, 0.0, 0.0])
    dets = [det("knife handle 1", KNIFE_CENTROID), det("tray", TRAY)]
    score = parse_plan_yaml(annotated_reply(contact=KNIFE_HANDLE, placement=place))
    pipe.execute(score, dets)

    entry = pipe._executor.detection_map["tray"]
    got = np.asarray(entry[robointer_gate.PLACEMENT_KEY], dtype=float)
    assert np.allclose(got[:2], place[:2], atol=5e-3), got


# --- 5. ON: the executor moves to the corrected point ----------------------


class _Motion(executor_motion.MotionMixin):
    """MotionMixin over stubs: the real _move_to_keypoint, nothing else."""

    TABLE_Z_FLOOR = -0.40

    def __init__(self, detection_map, holding=False):
        self.detection_map = detection_map
        self._holding = holding
        self._placed_labels = set()
        self.target = None

    def _check_abort(self):
        pass

    def _approach_target(self, target, detection=None, params=None):
        self.target = np.asarray(target, dtype=float)

    def _transport_to(self, target, **kwargs):
        self.target = np.asarray(target, dtype=float)


def test_the_executor_grasps_the_handle_not_the_centroid(robointer_on):
    pipe = LivePipeline()
    dets = [det("knife handle 1", KNIFE_CENTROID)]
    pipe.execute(parse_plan_yaml(annotated_reply(contact=KNIFE_HANDLE)), dets)

    motion = _Motion(pipe._executor.detection_map)
    result = motion._move_to_keypoint({"keypoint_label": "knife handle 1"}, 0.0)
    assert result.success, result.message
    assert np.allclose(motion.target[:2], KNIFE_HANDLE[:2], atol=5e-3), motion.target


def test_the_executor_is_untouched_when_robointer_is_off(robointer_off):
    pipe = LivePipeline()
    dets = [det("knife handle 1", KNIFE_CENTROID)]
    pipe.execute(parse_plan_yaml(annotated_reply(contact=KNIFE_HANDLE)), dets)

    motion = _Motion(pipe._executor.detection_map)
    motion._move_to_keypoint({"keypoint_label": "knife handle 1"}, 0.0)
    assert np.allclose(motion.target[:2], KNIFE_CENTROID[:2])


# --- 6. the profile overlay is a real source, and strict really raises -----


BAD_REPLY = (
    "task: put the knife in the tray\n"
    "tree:\n"
    "  type: sequence\n"
    "  children:\n"
    '  - type: move_to_keypoint\n    params: { keypoint_label: "knife handle 1" }\n'
    "    __robointer:\n"
    "      contact_point: [4200, 300]\n"  # off the right-hand edge of the image
    "  - type: grasp\n    params: { force: 60 }\n"
)


def test_a_loaded_profile_can_switch_it_on_and_make_it_strict(monkeypatch):
    """The server's overlay path, and the knob that refuses to degrade.

    Lenient (the default) drops the offending field and the node behaves as it
    does today; `strict: true` raises instead of quietly planning around a
    coordinate the model got wrong.
    """
    monkeypatch.delenv(robointer_gate.ROBOINTER_ENV, raising=False)
    block = {"enabled": True, "consume": False, "strict": True}
    planner = make_planner(BAD_REPLY)
    pipe = LivePipeline(planner)
    pipe.profile = SimpleNamespace(raw={"planning": {"robointer": block}})

    cfg = pipe.robointer_config()
    assert cfg.enabled is True and cfg.strict is True and cfg.consume is False

    dets = [det("knife handle 1", KNIFE_CENTROID)]
    with pytest.raises(RoboInterSchemaError, match="outside the image"):
        pipe.plan("put the knife in the tray", dets, scene_image=np.zeros((*SHAPE, 3), np.uint8))

    # Lenient is the shipped behaviour: the bad field is dropped, not fatal.
    block["strict"] = False
    score = pipe.plan(
        "put the knife in the tray", dets, scene_image=np.zeros((*SHAPE, 3), np.uint8)
    )
    assert ANNOTATION_KEY not in score["tree"]["children"][0]


# --- 7. asked for but unbuildable RAISES ----------------------------------


def test_uncalibrated_cameras_raise_instead_of_abstaining(robointer_on):
    """An operator who switches this on and gets silence has no protection.

    Same convention as perception/box_proposals.build_proposer: a feature that
    was ASKED FOR and cannot be built fails loudly.
    """
    pipe = LivePipeline(calibrated=False)
    dets = [det("knife handle 1", KNIFE_CENTROID)]
    score = parse_plan_yaml(annotated_reply(contact=KNIFE_HANDLE))
    with pytest.raises(RuntimeError, match="robointer"):
        pipe.execute(score, dets)

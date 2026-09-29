"""How much perception a verification is allowed to ask for.

Every assertion here is about a SAM3 inference that no longer happens, or about
a piece of evidence that is deliberately given up to stop it happening. Offline:
no robot, no camera, no SAM3.
"""

import time

import numpy as np
import pytest

from spark_real.control import verify_scope
from spark_real.control.verify_scope import ScopeConfig


# --- fakes -----------------------------------------------------------------


class FakeCal:
    """The CameraCalibration surface verify_scope._projects_into touches."""

    def __init__(self, position=(0.0, 0.0, 1.5), calibrated=True, width=640, height=480):
        self.width, self.height = width, height
        self.intrinsic_matrix = np.array(
            [[500.0, 0.0, width / 2], [0.0, 500.0, height / 2], [0.0, 0.0, 1.0]]
        )
        self.extrinsic = np.eye(4)
        if calibrated:
            # Camera at `position` looking straight down (+Z of the camera
            # frame pointing at -Z of the world).
            self.extrinsic = np.array(
                [
                    [1.0, 0.0, 0.0, position[0]],
                    [0.0, -1.0, 0.0, position[1]],
                    [0.0, 0.0, -1.0, position[2]],
                    [0.0, 0.0, 0.0, 1.0],
                ]
            )


class FakePipeline:
    """Counts capture() and detect() calls; records the kwargs detect saw."""

    def __init__(self, cams=("birdview",), detections=None, reject=()):
        self.profile = None
        self._cams = cams
        self._detections = list(detections or [])
        self.captures = 0
        self.detects = 0
        self.detect_kwargs = []
        self._reject = set(reject)
        self._last_captures = None
        self._last_captures_t = None

    def capture(self):
        self.captures += 1
        out = {c: {"rgb": object(), "depth": object(), "calibration": FakeCal()} for c in self._cams}
        self._last_captures = out
        self._last_captures_t = time.time()
        return out

    def detect(self, captures, **kwargs):
        bad = self._reject & set(kwargs)
        if bad:
            raise TypeError(f"detect() got an unexpected keyword argument {bad.pop()!r}")
        self.detects += 1
        self.detect_kwargs.append(kwargs)
        return list(self._detections)


class FakeExecutor:
    def __init__(self, verify=None, holding=None, raises=False):
        self._holding = holding
        self._verify = verify
        self._raises = raises
        if verify is None and not raises:
            self._verify_grasp = None

    def _verify_grasp(self):  # noqa: D401 - matches the executor surface
        if self._raises:
            raise RuntimeError("gripper offline")
        return self._verify


def _det(label, camera):
    return type("D", (), {"label": label, "camera": camera})()


# --- config ----------------------------------------------------------------


def test_defaults_keep_the_task_verdicts_presence_check():
    cfg = ScopeConfig()
    assert cfg.enabled is True
    # Reuse is the one cut that can be WRONG rather than blind (a frame from
    # before the jaws opened), so it is opt-in.
    assert cfg.reuse_capture_s == 0.0
    # The SuccessVerifier still re-detects its reference; only the leaf skips it.
    assert cfg.anchor_references is False
    assert cfg.condition_anchor_references is True
    assert cfg.fusion is False


def test_config_can_restore_todays_behaviour():
    cfg = ScopeConfig.from_raw({"verification": {"scope": {"enabled": False}}})
    assert cfg.enabled is False
    assert "scope OFF" in cfg.describe()


def test_config_parses_the_block():
    cfg = ScopeConfig.from_raw(
        {
            "verification": {
                "scope": {
                    "telemetry_short_circuit": "off",
                    "reuse_capture_s": "1.25",
                    "anchor_references": "yes",
                    "fusion": True,
                    "cameras": "all",
                }
            }
        }
    )
    assert cfg.telemetry_short_circuit is False
    assert cfg.reuse_capture_s == pytest.approx(1.25)
    assert cfg.anchor_references is True
    assert cfg.fusion is True
    assert cfg.cameras == "all"


def test_reuse_window_is_clamped_to_the_hard_ceiling():
    cfg = ScopeConfig.from_raw({"verification": {"scope": {"reuse_capture_s": 999}}})
    assert cfg.reuse_capture_s == verify_scope.REUSE_CAPTURE_MAX_S


def test_a_garbled_block_does_not_take_verification_down():
    cfg = ScopeConfig.from_raw({"verification": {"scope": {"reuse_capture_s": "soon",
                                                          "cameras": "wrist-only"}}})
    assert cfg.reuse_capture_s == ScopeConfig().reuse_capture_s
    assert cfg.cameras == "auto"


# --- 1. telemetry ----------------------------------------------------------


def test_still_holding_reads_the_gripper():
    assert verify_scope.still_holding(FakeExecutor(verify=True)) is True
    assert verify_scope.still_holding(FakeExecutor(verify=False)) is False


def test_still_holding_is_unknown_when_the_gripper_raises():
    assert verify_scope.still_holding(FakeExecutor(raises=True)) is None


def test_still_holding_falls_back_to_the_executor_flag():
    exe = FakeExecutor(holding=True)
    exe._verify_grasp = None
    assert verify_scope.still_holding(exe) is True


# --- 2. captures -----------------------------------------------------------


def test_a_recent_capture_is_reused_and_costs_no_frames():
    pipe = FakePipeline()
    cfg = ScopeConfig(reuse_capture_s=2.0)
    pipe.capture()
    assert pipe.captures == 1
    caps, stamp, reused = verify_scope.captures_for_verify(pipe, cfg)
    assert reused is True and caps and stamp > 0
    assert pipe.captures == 1, "the verify took its own frames anyway"


def test_a_recent_capture_is_not_reused_by_default():
    """The default must take fresh frames: the arm just moved."""
    pipe = FakePipeline()
    pipe.capture()
    _caps, _stamp, reused = verify_scope.captures_for_verify(pipe, ScopeConfig())
    assert reused is False
    assert pipe.captures == 2


def test_a_stale_capture_is_not_reused():
    pipe = FakePipeline()
    pipe.capture()
    pipe._last_captures_t = time.time() - 60.0
    caps, _stamp, reused = verify_scope.captures_for_verify(
        pipe, ScopeConfig(reuse_capture_s=2.0)
    )
    assert reused is False
    assert pipe.captures == 2


def test_reuse_is_off_when_scoping_is_off():
    pipe = FakePipeline()
    pipe.capture()
    _caps, _stamp, reused = verify_scope.captures_for_verify(
        pipe, ScopeConfig(enabled=False, reuse_capture_s=2.0)
    )
    assert reused is False
    assert pipe.captures == 2


# --- 3. labels -------------------------------------------------------------


def _base(label):
    label = str(label or "").strip()
    if label and label[-1].isdigit() and " " in label:
        return label.rsplit(" ", 1)[0]
    return label


def test_an_anchored_reference_is_not_re_detected():
    plan = verify_scope.plan_prompts(["plushie", "bowl"], {"bowl"}, ScopeConfig(), _base)
    assert plan.prompts == ["plushie"]
    assert plan.dropped == ["bowl"]
    assert plan.saved == 1


def test_instance_suffixes_are_stripped_before_the_comparison():
    plan = verify_scope.plan_prompts(["fork 2", "tray 1"], {"tray 1"}, ScopeConfig(), _base)
    assert plan.prompts == ["fork"]


def test_the_last_prompt_is_never_dropped():
    """Detecting nothing is not the cheaper answer; every predicate abstains."""
    plan = verify_scope.plan_prompts(["bowl"], {"bowl"}, ScopeConfig(), _base)
    assert plan.prompts == ["bowl"]
    assert plan.dropped == []


def test_nothing_is_dropped_when_scoping_is_off():
    plan = verify_scope.plan_prompts(
        ["plushie", "bowl"], {"bowl"}, ScopeConfig(enabled=False), _base
    )
    assert plan.prompts == ["plushie", "bowl"]


def test_duplicate_labels_collapse_to_one_prompt():
    plan = verify_scope.plan_prompts(["fork 1", "fork 2", "tray"], set(), ScopeConfig(), _base)
    assert plan.prompts == ["fork", "tray"]


# --- 4. cameras ------------------------------------------------------------


def _caps(**cams):
    return {name: {"calibration": cal} for name, cal in cams.items()}


def test_a_camera_the_object_does_not_project_into_is_skipped():
    near = FakeCal(position=(0.0, 0.0, 1.5))
    far = FakeCal(position=(10.0, 10.0, 1.5))
    keep, dropped = verify_scope.plan_cameras(
        _caps(birdview=near, elsewhere=far), [np.array([0.0, 0.0, 0.0])], ScopeConfig()
    )
    assert keep == ["birdview"]
    assert dropped == ["elsewhere"]


def test_an_uncalibrated_camera_is_kept_because_it_cannot_be_tested():
    keep, dropped = verify_scope.plan_cameras(
        _caps(birdview=FakeCal(), wrist=FakeCal(calibrated=False)),
        [np.array([0.0, 0.0, 0.0])],
        ScopeConfig(),
    )
    assert set(keep) == {"birdview", "wrist"}
    assert dropped == []


def test_every_camera_is_kept_when_the_filter_would_empty_the_set():
    far = FakeCal(position=(10.0, 10.0, 1.5))
    keep, dropped = verify_scope.plan_cameras(
        _caps(a=far, b=far), [np.array([0.0, 0.0, 0.0])], ScopeConfig()
    )
    assert set(keep) == {"a", "b"}
    assert dropped == []


def test_no_plan_time_anchor_means_no_camera_scoping():
    keep, dropped = verify_scope.plan_cameras(
        _caps(a=FakeCal(), b=FakeCal(position=(10.0, 10.0, 1.5))), [None], ScopeConfig()
    )
    assert set(keep) == {"a", "b"}
    assert dropped == []


def test_cameras_all_disables_the_frustum_test():
    far = FakeCal(position=(10.0, 10.0, 1.5))
    keep, dropped = verify_scope.plan_cameras(
        _caps(a=FakeCal(), b=far), [np.array([0.0, 0.0, 0.0])], ScopeConfig(cameras="all")
    )
    assert set(keep) == {"a", "b"}
    assert dropped == []


# --- the detection call ----------------------------------------------------


def test_the_fusion_gate_is_off_on_the_verify_path():
    pipe = FakePipeline(detections=[_det("plushie", "birdview")])
    verify_scope.detect_scoped(pipe, {}, ["plushie"], ScopeConfig())
    assert pipe.detect_kwargs[0]["use_fusion"] is False


def test_fusion_stays_on_when_asked_for():
    pipe = FakePipeline()
    verify_scope.detect_scoped(pipe, {}, ["plushie"], ScopeConfig(fusion=True))
    assert "use_fusion" not in pipe.detect_kwargs[0]


def test_a_second_verification_in_the_window_costs_no_inferences():
    pipe = FakePipeline(detections=[_det("plushie", "birdview")])
    cfg = ScopeConfig(reuse_capture_s=2.0)
    stamp = time.time()
    first = verify_scope.detect_scoped(pipe, {}, ["plushie"], cfg, capture_time=stamp)
    second = verify_scope.detect_scoped(pipe, {}, ["plushie"], cfg, capture_time=stamp)
    assert pipe.detects == 1, "the second verify re-ran SAM3 on the same frames"
    assert [d.label for d in second] == [d.label for d in first]


def test_a_different_prompt_set_is_not_served_from_the_memo():
    pipe = FakePipeline()
    cfg, stamp = ScopeConfig(reuse_capture_s=2.0), time.time()
    verify_scope.detect_scoped(pipe, {}, ["plushie"], cfg, capture_time=stamp)
    verify_scope.detect_scoped(pipe, {}, ["plushie", "bowl"], cfg, capture_time=stamp)
    assert pipe.detects == 2


def test_a_different_capture_is_not_served_from_the_memo():
    pipe = FakePipeline()
    cfg = ScopeConfig(reuse_capture_s=2.0)
    verify_scope.detect_scoped(pipe, {}, ["plushie"], cfg, capture_time=100.0)
    verify_scope.detect_scoped(pipe, {}, ["plushie"], cfg, capture_time=200.0)
    assert pipe.detects == 2


def test_nothing_is_memoised_when_reuse_is_off():
    pipe = FakePipeline()
    cfg, stamp = ScopeConfig(), time.time()
    verify_scope.detect_scoped(pipe, {}, ["plushie"], cfg, capture_time=stamp)
    verify_scope.detect_scoped(pipe, {}, ["plushie"], cfg, capture_time=stamp)
    assert pipe.detects == 2


def test_nothing_is_memoised_when_scoping_is_off():
    pipe = FakePipeline()
    cfg, stamp = ScopeConfig(enabled=False, reuse_capture_s=2.0), time.time()
    verify_scope.detect_scoped(pipe, {}, ["plushie"], cfg, capture_time=stamp)
    verify_scope.detect_scoped(pipe, {}, ["plushie"], cfg, capture_time=stamp)
    assert pipe.detects == 2


def test_an_older_backend_without_use_fusion_still_detects():
    pipe = FakePipeline(detections=[_det("plushie", "birdview")], reject={"use_fusion"})
    out = verify_scope.detect_scoped(pipe, {}, ["plushie"], ScopeConfig())
    assert [d.label for d in out] == ["plushie"]
    assert "use_fusion" not in pipe.detect_kwargs[-1]


def test_an_older_backend_without_multi_instance_prompts_still_detects():
    pipe = FakePipeline(
        detections=[_det("fork", "birdview")], reject={"multi_instance_prompts"}
    )
    out = verify_scope.detect_scoped(
        pipe, {}, ["fork"], ScopeConfig(), multi_instance_prompts={"fork"}
    )
    assert [d.label for d in out] == ["fork"]


def test_no_prompts_means_no_detect_call_at_all():
    pipe = FakePipeline()
    assert verify_scope.detect_scoped(pipe, {}, [], ScopeConfig()) == []
    assert pipe.detects == 0


def test_per_camera_groups_without_merging():
    dets = [_det("fork", "birdview"), _det("fork", "sideview"), _det("tray", "birdview")]
    grouped = verify_scope.per_camera(dets)
    assert sorted(grouped) == ["birdview", "sideview"]
    assert len(grouped["birdview"]) == 2

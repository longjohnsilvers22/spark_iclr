"""
Port step 2: gated, held-suppressed re-detect adoption
(execution_recovery.sticky_adopt + the redetect_single / redetect_all
wiring). Hardware-free: fake pipeline + fake executor.

The invariant under test (dyn journal it-6): a fresh frame NEVER rewrites
detection_map except through an explicit, gated adoption -- held /
ambiguous / out-of-gate / no-candidate frames leave the map UNTOUCHED.
"""

import numpy as np
import pytest

from spark_real.control import execution_recovery as er


class _Det:
    def __init__(self, label, pos, conf=0.9, agent=None):
        self.label = label
        self.position_3d = None if pos is None else np.array(pos, float)
        self.confidence = conf
        self.orientation_angle = 0.3
        self.aspect_ratio = 2.0
        self.position_agentview = (
            None if agent is None else np.array(agent, float)
        )


class _FakePipeline:
    def __init__(self, dets):
        self.dets = dets
        self.profile = type(
            "P", (), {"raw": {"perception": {"sticky": {"enabled": True}}}}
        )()

    def capture(self):
        return {"birdview": {"rgb": object(), "depth": None, "calibration": None}}

    def detect(self, captures, prompts=None, **kw):
        return list(self.dets)

    def merge_detections(self, dets, **kw):
        return list(dets)


class _FakeExecutor:
    def __init__(self, dets, det_map, holding=False, pick_label=""):
        self._pipeline = _FakePipeline(dets)
        self.detection_map = det_map
        self._holding = holding
        self._last_pick_label = pick_label
        self._active_grasp_label = pick_label
        self._placed_labels = set()
        self._destination_labels = set()
        self._ee = np.array([0.9, 0.9, 0.3])

    def _get_current_position(self):
        return np.array(self._ee, float)


def _entry(pos):
    return {
        "position_3d": list(pos),
        "orientation_angle": 0.0,
        "aspect_ratio": 1.0,
    }


def test_in_gate_candidate_adopted_as_delta():
    old = _entry([0.10, 0.20, -0.10])  # fused (e.g. sideview z won)
    old["position_agentview"] = [0.10, 0.20, -0.25]
    ex = _FakeExecutor(
        [_Det("block", [0.13, 0.20, -0.24], agent=[0.13, 0.20, -0.24])],
        {"block": old},
    )
    er.redetect_single(ex, "block")
    got = np.array(ex.detection_map["block"]["position_3d"])
    # Delta (+3cm x, +1cm z same-source) applied to the FUSED position:
    # the sideview Z correction survives adoption.
    assert np.allclose(got, [0.13, 0.20, -0.09], atol=1e-9)


def test_held_label_frame_leaves_map_untouched_and_skips_detect():
    old = _entry([0.1, 0.2, -0.2])
    ex = _FakeExecutor(
        [_Det("plushie", [0.5, 0.5, -0.2])],
        {"plushie": old},
        holding=True,
        pick_label="plushie",
    )
    calls = {"n": 0}
    orig = ex._pipeline.capture

    def counting_capture():
        calls["n"] += 1
        return orig()

    ex._pipeline.capture = counting_capture
    er.redetect_single(ex, "plushie")
    assert ex.detection_map["plushie"] is old  # untouched
    assert calls["n"] == 0  # held: no capture at all


def test_twin_candidates_ambiguous_holds_binding():
    old = _entry([0.10, 0.20, -0.20])
    ex = _FakeExecutor(
        [
            # 2.5 cm apart: past the 2 cm dedup radius (two real objects)
            # but within the 3 cm ambiguity separation (undecidable).
            _Det("bowl", [0.14, 0.20, -0.20]),
            _Det("bowl", [0.165, 0.20, -0.20]),
        ],
        {"bowl": old},
    )
    er.redetect_single(ex, "bowl")
    assert ex.detection_map["bowl"] is old


def test_duplicate_masks_deduped_before_association():
    # Two slightly-offset masks of the SAME object must not read as an
    # ambiguous twin pair.
    old = _entry([0.10, 0.20, -0.20])
    ex = _FakeExecutor(
        [
            _Det("bowl", [0.14, 0.20, -0.20], conf=0.9),
            _Det("bowl", [0.145, 0.20, -0.20], conf=0.4),  # dup of the first
        ],
        {"bowl": old},
    )
    er.redetect_single(ex, "bowl")
    got = np.array(ex.detection_map["bowl"]["position_3d"])
    assert np.allclose(got[:2], [0.14, 0.20], atol=1e-9)


def test_out_of_gate_never_adopted():
    old = _entry([0.10, 0.20, -0.20])
    ex = _FakeExecutor([_Det("block", [0.60, 0.20, -0.20])], {"block": old})
    er.redetect_single(ex, "block")
    assert ex.detection_map["block"] is old


def test_out_of_gate_with_ee_hover_is_self_occlusion_hold():
    old = _entry([0.10, 0.20, -0.20])
    ex = _FakeExecutor([_Det("block", [0.60, 0.20, -0.20])], {"block": old})
    ex._ee = np.array([0.11, 0.20, -0.15])  # hovering over the binding
    er.redetect_single(ex, "block")
    assert ex.detection_map["block"] is old


def test_no_candidates_keeps_binding():
    old = _entry([0.10, 0.20, -0.20])
    ex = _FakeExecutor([], {"block": old})
    er.redetect_single(ex, "block")
    assert ex.detection_map["block"] is old


def test_static_scene_noise_adopts_without_phantom_jump():
    old = _entry([0.10, 0.20, -0.20])
    old["position_agentview"] = [0.10, 0.20, -0.20]
    ex = _FakeExecutor(
        [_Det("block", [0.108, 0.195, -0.20], agent=[0.108, 0.195, -0.20])],
        {"block": old},
    )
    er.redetect_single(ex, "block")
    got = np.array(ex.detection_map["block"]["position_3d"])
    assert np.linalg.norm(got[:2] - [0.10, 0.20]) < 0.012


def test_disabled_flag_keeps_legacy_unconditional_adoption():
    old = _entry([0.10, 0.20, -0.20])
    ex = _FakeExecutor([_Det("block", [0.60, 0.20, -0.20])], {"block": old})
    ex._pipeline.profile.raw["perception"]["sticky"]["enabled"] = False
    er.redetect_single(ex, "block")
    # Legacy behavior preserved exactly: nearest candidate adopted at ANY
    # distance (this is the default until the flag is turned on).
    got = np.array(ex.detection_map["block"]["position_3d"])
    assert np.allclose(got[:2], [0.60, 0.20])


def test_redetect_all_respects_destination_zones_and_gate():
    old = _entry([0.10, 0.20, -0.20])
    tray = _entry([0.50, 0.50, -0.20])
    tray["obb_minor_m"] = 0.2
    dets = [
        _Det("block", [0.50, 0.50, -0.20]),  # inside the tray zone (placed)
        _Det("block", [0.13, 0.20, -0.20]),  # genuine next pick, in gate
    ]
    ex = _FakeExecutor(dets, {"block": old, "tray": tray})
    ex._destination_labels = {"tray"}
    ex.robot = type("R", (), {"HOME_CONFIG": None})()
    actions = [
        {"type": "release"},
        {"type": "move_to_keypoint", "params": {"keypoint_label": "block"}},
        {"type": "grasp", "params": {}},
    ]
    er.redetect_all(ex, actions, 0)
    got = np.array(ex.detection_map["block"]["position_3d"])
    assert np.allclose(got[:2], [0.13, 0.20], atol=1e-9)


def test_env_override_wins(monkeypatch):
    old = _entry([0.10, 0.20, -0.20])
    ex = _FakeExecutor([_Det("block", [0.60, 0.20, -0.20])], {"block": old})
    ex._pipeline.profile.raw["perception"]["sticky"]["enabled"] = False
    monkeypatch.setenv("SPARK_STICKY", "1")
    er.redetect_single(ex, "block")
    assert ex.detection_map["block"] is old  # sticky forced on: out-of-gate

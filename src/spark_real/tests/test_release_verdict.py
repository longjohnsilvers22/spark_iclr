"""
Port step 6: evidence-graded per-release verdict with verdict/binding
separation (execution_recovery.verify_placement + _release_verdict).

Three verdict branches (ok / still_at_origin / abstain), and -- the it-4
lesson -- the rebind to the nearest fresh instance at ANY distance fires
even when the verdict abstains.
"""

import numpy as np

from spark_real.control import execution_recovery as er


class _Det:
    def __init__(self, label, pos, conf=0.9):
        self.label = label
        self.position_3d = np.array(pos, float)
        self.confidence = conf
        self.orientation_angle = 0.0
        self.aspect_ratio = 1.0
        self.position_agentview = None
        self.camera = "birdview"
        self.centroid_2d = (1.0, 1.0)
        self.mask_area = 10
        self.depth_meters = 0.5


class _Witness:
    def __init__(self, xyz):
        self.tcp_xyz = tuple(xyz)


class _FakePipeline:
    def __init__(self, dets):
        self.dets = dets
        self.profile = type(
            "P", (), {"raw": {"verification": {"release_verdict": True}}}
        )()

    def capture(self):
        return {"birdview": {"rgb": object(), "depth": None, "calibration": None}}

    def detect(self, captures, prompts=None, **kw):
        return list(self.dets)


class _FakeExecutor:
    def __init__(self, dets, det_map, release_xyz):
        self._pipeline = _FakePipeline(dets)
        self.detection_map = det_map
        self._release_witness = _Witness(release_xyz)
        self._holding = False
        self._last_pick_label = ""
        self._verify_gates = {}

    def _get_current_position(self):
        return np.array(self._release_witness.tcp_xyz, float)


ORIGIN = [0.10, 0.20, -0.20]
RELEASE = [0.50, 0.50, -0.05]


def _run(dets):
    ex = _FakeExecutor(
        dets, {"plushie": {"position_3d": list(ORIGIN)}}, RELEASE
    )
    verdict = er._release_verdict(ex, "plushie", dets)
    return ex, verdict


def test_ok_branch_lands_near_release_point():
    ex, v = _run([_Det("plushie", [0.52, 0.51, -0.25])])
    assert v["verdict"] == "ok"
    assert v["xy_error_m"] < 0.03
    got = np.array(ex.detection_map["plushie"]["position_3d"])
    assert np.allclose(got[:2], [0.52, 0.51])
    assert ex._verify_gates.get("release") is not False


def test_still_at_origin_branch_sets_release_fail_evidence():
    ex, v = _run([_Det("plushie", [0.11, 0.21, -0.20])])
    assert v["verdict"] == "still_at_origin"
    # Binding follows the fresh observation (self-caused displacement).
    got = np.array(ex.detection_map["plushie"]["position_3d"])
    assert np.allclose(got[:2], [0.11, 0.21])
    # Evidence for _fail_has_evidence: the release gate voted fail.
    assert ex._verify_gates.get("release") is False


def test_abstain_branch_still_rebinds_at_any_distance():
    # Neither near the release point nor the origin: verdict abstains,
    # but the binding STILL follows the nearest fresh instance (it-4:
    # gating the rebind on the verdict's evidence test took t2 6/15 -> 0/15).
    ex, v = _run([_Det("plushie", [0.80, 0.90, -0.20])])
    assert v["verdict"] == "abstain"
    got = np.array(ex.detection_map["plushie"]["position_3d"])
    assert np.allclose(got[:2], [0.80, 0.90])
    assert ex._verify_gates.get("release") is not False


def test_no_candidates_abstains_and_keeps_binding():
    ex, v = _run([])
    assert v["verdict"] == "abstain"
    assert np.allclose(
        np.array(ex.detection_map["plushie"]["position_3d"]), ORIGIN
    )


def test_verify_placement_records_verdict_in_trace(monkeypatch):
    traces = []

    class _FakeWriter:
        def next_index(self):
            return 0

        def write(self, trace):
            traces.append(trace)

    monkeypatch.setattr(
        er.TraceWriter, "from_pipeline", staticmethod(lambda p: _FakeWriter())
    )
    ex = _FakeExecutor(
        [_Det("plushie", [0.52, 0.51, -0.25])],
        {"plushie": {"position_3d": list(ORIGIN)}},
        RELEASE,
    )
    er.verify_placement(ex, "plushie", "bowl")
    assert len(traces) == 1
    v = traces[0].params.get("release_verdict")
    assert v is not None and v["verdict"] == "ok"


def test_flag_off_keeps_trace_only_behavior(monkeypatch):
    traces = []

    class _FakeWriter:
        def next_index(self):
            return 0

        def write(self, trace):
            traces.append(trace)

    monkeypatch.setattr(
        er.TraceWriter, "from_pipeline", staticmethod(lambda p: _FakeWriter())
    )
    ex = _FakeExecutor(
        [_Det("plushie", [0.52, 0.51, -0.25])],
        {"plushie": {"position_3d": list(ORIGIN)}},
        RELEASE,
    )
    ex._pipeline.profile.raw["verification"]["release_verdict"] = False
    old_entry = ex.detection_map["plushie"]
    er.verify_placement(ex, "plushie", "bowl")
    assert traces[0].params.get("release_verdict") is None
    assert ex.detection_map["plushie"] is old_entry  # no rebind

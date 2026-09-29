"""The operator's RF-DETR switch, on the REAL resolution path.

Three modes, and nothing else is exposed:

  auto    (default, = the shipped YAML) the proposer is consulted only for the
          labels on `perception.fusion.proposer.labels`.
  always  the allowlist is dropped: every in-vocabulary label is scored again,
          which is the behaviour that penalised a correct plushie mask.
  off     nobody is consulted, but the gate itself STAYS UP -- the mask-quality
          channel is what killed the bogus 69 deg wrist rotation and it must
          not be collateral damage of turning the second detector off.

These tests do not call the mode helpers in isolation. They call the REAL
``PerceptionMixin.detect()`` with only SAM3, the camera and the RF-DETR MODEL
replaced, and read the fused numbers off the detections that come out -- the
same harness (and the same MEASURED boxes and confidences) as
test_fusion_live_path, so a mode that resolves correctly but never reaches the
gate fails here.

The cache is the other half. ``detection_gate()`` memoises the gate on the
pipeline, so a toggle that does not invalidate it does nothing at all until
the next server restart. Every runtime test below therefore detects ONCE
first, to force the gate to be built and cached, and only then toggles.
"""

import asyncio

import pytest

from spark_real.perception import box_proposals as bp
from spark_real.perception import detection_fusion as df
from spark_real.pipeline_perception import PROPOSER_MODES
from spark_real.routes import perception as perception_routes
from spark_real.routes import state
from spark_real.tests.test_fusion_live_path import (
    BOXES_SIDE_F60,
    BOWL_SIDE,
    BOWL_SIDE_FUSED,
    Config,
    FakeSAM3,
    LivePipeline,
    MASKBOX_SIDE_BOWL,
    MASKBOX_SIDE_PLUSHIE,
    PLUSHIE_SIDE,
    PLUSHIE_SIDE_PENALISED,
    ReplayProposer,
    make_det,
    plushie_mask,
    rect_mask,
)

SPECS = [
    ("plushie", PLUSHIE_SIDE, MASKBOX_SIDE_PLUSHIE),
    ("bowl", BOWL_SIDE, MASKBOX_SIDE_BOWL),
]


class SpyProposer(ReplayProposer):
    """The replayed RF-DETR pass, counting how often it is actually asked."""

    def __init__(self, state_):
        super().__init__(state_)
        self.calls = []

    def propose(self, rgb, labels=()):
        self.calls.append(list(labels))
        return super().propose(rgb, labels)


class Rig:
    """One pipeline over the fakes, plus the proposers that were built for it."""

    def __init__(self, pipe, built, proposers):
        self.pipe = pipe
        self.built = built  # one proposer-config dict per gate construction
        self.proposers = proposers

    def detect(self, labels=("plushie", "bowl")):
        dets = self.pipe.detect(self.pipe.captures(), list(labels))
        return {d.label: d for d in dets}

    @property
    def consulted(self):
        return sum(len(p.calls) for p in self.proposers if isinstance(p, SpyProposer))


@pytest.fixture(autouse=True)
def clean_runtime_state(monkeypatch):
    """No leaked override, in either direction."""
    monkeypatch.setattr(state, "proposer_mode", None, raising=False)
    monkeypatch.setattr(state, "pipeline", None, raising=False)
    monkeypatch.delenv("SPARK_PROPOSER_MODE", raising=False)
    yield


@pytest.fixture
def rig(monkeypatch):
    """The live detect path with the RF-DETR pass replayed box for box.

    ``_build`` reproduces the real ``box_proposals.build_proposer`` contract:
    a null/disabled backend yields a NullProposer and no model is loaded.
    """
    monkeypatch.setenv("SPARK_DETECTION_FUSION", "1")
    monkeypatch.delenv("SPARK_BOX_PROPOSER", raising=False)
    built, proposers = [], []

    def _build(cfg):
        cfg = dict(cfg or {})
        built.append(cfg)
        backend = str(cfg.get("backend", "null")).lower()
        if backend in ("null", "off", "none", "") or not cfg.get("enabled", False):
            p = bp.NullProposer()
        else:
            p = SpyProposer({"boxes": BOXES_SIDE_F60})
        proposers.append(p)
        return p

    monkeypatch.setattr(df, "build_proposer", _build)

    per_prompt = {
        label: (
            lambda label=label, conf=conf, bbox=bbox: make_det(
                label, conf, rect_mask(bbox), 1.1, 0.0, (-0.9, 0.0, -0.20)
            ),
        )
        for label, conf, bbox in SPECS
    }
    pipe = LivePipeline(FakeSAM3(per_prompt), Config(), cams=("sideview",))
    return Rig(pipe, built, proposers)


def _set_mode(pipe, mode):
    """Toggle THROUGH THE ROUTE, which is the thing under test."""
    state.pipeline = pipe
    return asyncio.run(
        perception_routes.set_proposer_mode(perception_routes.ProposerModeBody(mode=mode))
    )


def _status(pipe):
    state.pipeline = pipe
    return asyncio.run(perception_routes.fusion_status())


# --- 1. the three modes, on the live detect path ---------------------------


def test_auto_is_the_default(rig):
    """Untouched == today: plushie abstains bit-for-bit, bowl is corroborated."""
    dets = rig.detect()
    assert dets["plushie"].fused_confidence == PLUSHIE_SIDE  # exact: no penalty
    assert dets["bowl"].fused_confidence == pytest.approx(BOWL_SIDE_FUSED, abs=5e-4)
    assert _status(rig.pipe)["mode"] == "auto"


def test_always_consults_the_proposer_on_non_allowlisted_labels(rig):
    """'always' drops the allowlist, so RF-DETR's silence costs the plushie again."""
    rig.detect()  # gate built and cached under 'auto' first
    _set_mode(rig.pipe, "always")
    dets = rig.detect()
    assert dets["plushie"].fused_confidence == pytest.approx(
        PLUSHIE_SIDE_PENALISED, abs=5e-4
    )
    assert dets["bowl"].fused_confidence == pytest.approx(BOWL_SIDE_FUSED, abs=5e-4)


def test_off_keeps_the_mask_quality_gate_up(rig):
    """No proposer at all -- and the plushie still earns no wrist rotation."""
    rig.detect()
    _set_mode(rig.pipe, "off")
    dets = rig.detect()

    gate = rig.pipe.detection_gate()
    assert gate is not None, "turning the proposer off took the whole gate down"
    assert isinstance(gate.proposer, bp.NullProposer)
    assert rig.consulted == 1, "the proposer was consulted with the mode off"
    # confidences untouched, bit for bit, on BOTH labels
    assert dets["plushie"].fused_confidence == PLUSHIE_SIDE
    assert dets["bowl"].fused_confidence == BOWL_SIDE
    # ...and the mask-quality channel is still measuring, still stamping and
    # still firing: this plushie mask fails the quality thresholds and gets
    # flagged with no second detector in sight.
    assert dets["plushie"].low_quality is True
    # But it is NOT reprompted. The blob is round, and round is a true fact
    # about the object that no synonym changes -- so the retry could only ever
    # end where it began, at a top-down grasp. Measured on a live run
    # (2026-08-18) this loop burned 18s of every 30s camera pass and rejected
    # all 24 of its attempts. The flag stays up; only the futile retry goes.
    assert dets["plushie"].reprompt_attempts == 0
    assert hasattr(dets["bowl"], "axis_trust")


def test_off_still_vetoes_a_meaningless_axis(monkeypatch, rig):
    """The 69 deg regression, with the proposer switched off.

    The round plushie blob has no real major axis; axis_trust must still be 0
    so grasp_strategy refuses the OBB yaw.
    """
    rig.pipe._perception = FakeSAM3(
        {
            "plushie": (
                lambda: make_det(
                    "plushie", 0.24, plushie_mask(), 3.14, -68.9, (-0.9, 0.0, -0.20)
                ),
            )
        }
    )
    _set_mode(rig.pipe, "off")
    d = rig.detect(("plushie",))["plushie"]
    assert d.axis_trust == 0.0
    assert d.low_quality is True


def test_off_survives_a_force_built_backend(monkeypatch, rig):
    """Belt and braces: even if a backend gets built anyway (SPARK_BOX_PROPOSER
    is a debug hook that bypasses the backend key inside build_proposer), 'off'
    means the proposer's opinion is never applied to any label."""
    spies = []

    def _build(cfg):
        rig.built.append(dict(cfg or {}))
        p = SpyProposer({"boxes": BOXES_SIDE_F60})
        spies.append(p)
        rig.proposers.append(p)
        return p

    monkeypatch.setattr(df, "build_proposer", _build)
    _set_mode(rig.pipe, "off")
    dets = rig.detect()
    assert spies, "this test is pointless unless a real backend was built"
    assert dets["bowl"].fused_confidence == BOWL_SIDE  # exact: no agree bonus
    assert dets["plushie"].fused_confidence == PLUSHIE_SIDE


# --- 2. the cache: a toggle that does not invalidate it is a no-op ---------


def test_runtime_toggle_beats_the_cached_gate(rig):
    """THE POINT. Detect first so the gate is built and memoised on the
    pipeline, THEN toggle, THEN detect again on the SAME pipeline object.

    With a stale ``_fusion_gate`` the second detect reuses the allowlisted gate
    and the plushie stays at 0.95, so this assertion is what proves the route
    resets the cache. No restart, no new pipeline.
    """
    first = rig.detect()
    assert first["plushie"].fused_confidence == PLUSHIE_SIDE
    assert len(rig.built) == 1, "the gate was not cached; this test proves nothing"

    _set_mode(rig.pipe, "always")

    second = rig.detect()
    assert len(rig.built) == 2, "the cached gate was reused: the toggle is inert"
    assert second["plushie"].fused_confidence < PLUSHIE_SIDE
    assert second["plushie"].fused_confidence == pytest.approx(
        PLUSHIE_SIDE_PENALISED, abs=5e-4
    )


def test_toggle_back_to_auto_restores_the_allowlist(rig):
    rig.detect()
    _set_mode(rig.pipe, "always")
    assert rig.detect()["plushie"].fused_confidence < PLUSHIE_SIDE
    _set_mode(rig.pipe, "auto")
    assert rig.detect()["plushie"].fused_confidence == PLUSHIE_SIDE


def test_clearing_the_override_falls_back_to_config(rig):
    """None = fall back to config, the same idiom as state.use_cached_bt."""
    _set_mode(rig.pipe, "off")
    assert _status(rig.pipe)["mode"] == "off"
    body = _set_mode(rig.pipe, None)
    assert state.proposer_mode is None
    assert body["mode"] == "auto"
    assert body["source"] == "config"
    assert rig.detect()["bowl"].fused_confidence == pytest.approx(
        BOWL_SIDE_FUSED, abs=5e-4
    )


# --- 3. resolution order: YAML -> env -> runtime override ------------------


def test_env_sets_the_mode_and_the_runtime_override_wins(monkeypatch, rig):
    monkeypatch.setenv("SPARK_PROPOSER_MODE", "always")
    assert _status(rig.pipe)["mode"] == "always"
    assert _status(rig.pipe)["source"] == "env"
    assert rig.detect()["plushie"].fused_confidence < PLUSHIE_SIDE

    _set_mode(rig.pipe, "off")  # runtime is the most specific: it wins
    st = _status(rig.pipe)
    assert st["mode"] == "off"
    assert st["source"] == "runtime"
    assert rig.detect()["plushie"].fused_confidence == PLUSHIE_SIDE


def test_garbage_env_is_ignored_not_fatal(monkeypatch, rig):
    monkeypatch.setenv("SPARK_PROPOSER_MODE", "sometimes")
    assert _status(rig.pipe)["mode"] == "auto"
    assert rig.detect()["plushie"].fused_confidence == PLUSHIE_SIDE


# --- 4. the read-only surface the UI shows --------------------------------


def test_status_surfaces_the_gate_and_the_backend(rig):
    st = _status(rig.pipe)
    assert st["modes"] == list(PROPOSER_MODES) == ["auto", "always", "off"]
    assert st["fusion_enabled"] is True
    assert st["backend"] == "rfdetr"  # shipped ur10e YAML
    assert "bowl" in st["labels"] and "plushie" not in st["labels"]
    assert st["loaded_backend"] is None  # status must not build the gate
    rig.detect()
    assert _status(rig.pipe)["loaded_backend"] == "replay"


def test_bad_mode_is_rejected_and_changes_nothing(rig):
    rig.detect()
    resp = asyncio.run(
        perception_routes.set_proposer_mode(
            perception_routes.ProposerModeBody(mode="turbo")
        )
    )
    assert getattr(resp, "status_code", 200) == 400
    assert state.proposer_mode is None
    assert len(rig.built) == 1, "a rejected mode still dropped the cached gate"

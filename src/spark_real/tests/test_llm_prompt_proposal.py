"""The LLM label-proposal rung of the detect escalation ladder. Offline.

The registry's cheap rungs -- relax the secondary-mask score, then append the
group's hand-written ``alt_prompts`` -- only help when the author guessed the
right synonym in advance. When they both miss, the only thing left today is
``fallback.on_mismatch``, i.e. losing the episode. This rung asks the planner
LLM to LOOK at the frame and name the thing before that happens: on the live
plushie run the top-down view reads as a purple hat, which no author would
have written into ``alt_prompts``.

No network here. The planner is a fake that records what it was asked and
replays a canned answer, so what is tested is the control flow: that the rung
fires only after the cheaper rungs, that its proposals come back under the
canonical group name, and that an unavailable / broken / garbage-returning
LLM degrades to ``on_mismatch`` instead of becoming a failure of its own.
"""

import numpy as np
import pytest

from spark_real.perception.prompt_registry import (
    PromptCountMismatch,
    load_registry,
)
from spark_real.perception.spark_perception import ObjectDetection
from spark_real.pipeline_perception import PerceptionMixin

TASK = "find the plushie"


def write_spec(tmp_path, fallback_extra=""):
    (tmp_path / "find_the_plushie.yaml").write_text(
        "task: \"find the plushie\"\n"
        "groups:\n"
        "  - text: \"plushie\"\n"
        "    expect: {min: 1, max: 1}\n"
        "    min_conf: 0.15\n"
        "    alt_prompts: [\"stuffed animal\"]\n"
        "fallback:\n"
        "  relax_secondary_score: 0.30\n"
        "  on_mismatch: abort\n" + fallback_extra
    )
    return tmp_path


def _od(label, conf, pos=(0.5, 0.0, 0.0)):
    return ObjectDetection(
        label=label,
        confidence=conf,
        centroid_2d=(1.0, 1.0),
        mask_area=900,
        depth_meters=1.0,
        position_3d=np.array(pos, dtype=float),
        camera="birdview",
    )


class FakePlanner:
    """Records every proposal request; replays ``answers`` in order."""

    def __init__(self, answers=(), raises=None):
        self.answers = list(answers)
        self.raises = raises
        self.calls = []

    def propose_alt_prompts(
        self, target_label, tried_prompts, scene_image=None, max_prompts=4
    ):
        self.calls.append(
            {
                "target": target_label,
                "tried": list(tried_prompts),
                "has_image": scene_image is not None,
                "max_prompts": max_prompts,
            }
        )
        if self.raises is not None:
            raise self.raises
        return self.answers.pop(0) if self.answers else []


class ScriptedPipeline(PerceptionMixin):
    """Real detect_for_task; ``detect()`` replaced by a script of results."""

    def __init__(self, script, prompts_dir, planner=None):
        self.script = list(script)
        self.calls = []
        self._planner = planner
        self.config = type("C", (), {"task_prompts_dir": str(prompts_dir)})()

    def detect(
        self,
        captures,
        prompts,
        multi_instance=False,
        multi_instance_prompts=None,
        secondary_score=None,
    ):
        self.calls.append((list(prompts), secondary_score))
        return self.script.pop(0) if self.script else []


def captures_with_image():
    rgb = np.zeros((16, 16, 3), np.uint8)
    return {"birdview": {"rgb": rgb, "depth": None, "calibration": None}}


# --- default ---------------------------------------------------------------


def test_rung_is_on_by_default(tmp_path):
    """A task that says nothing about it still gets the rescue attempt."""
    reg = load_registry(write_spec(tmp_path))
    assert reg.get(TASK).fallback.llm_propose_prompts is True


def test_rung_can_be_switched_off_per_task(tmp_path):
    reg = load_registry(write_spec(tmp_path, "  llm_propose_prompts: false\n"))
    assert reg.get(TASK).fallback.llm_propose_prompts is False


# --- ordering: cheapest first ---------------------------------------------


def test_llm_not_consulted_when_first_pass_passes(tmp_path):
    planner = FakePlanner([["purple hat"]])
    p = ScriptedPipeline([[_od("plushie", 0.9)]], write_spec(tmp_path), planner)
    _, _, res, _, _ = p.detect_for_task(captures_with_image(), TASK)
    assert res.ok
    assert planner.calls == [], "the LLM was billed for a scene that detected fine"


def test_llm_not_consulted_when_alt_prompts_rescue(tmp_path):
    """Rung 2 succeeding must end the ladder before the API call."""
    planner = FakePlanner([["purple hat"]])
    p = ScriptedPipeline(
        [
            [_od("plushie", 0.09)],
            [_od("plushie", 0.09)],
            [_od("plushie", 0.09), _od("stuffed animal", 0.62)],
        ],
        write_spec(tmp_path),
        planner,
    )
    _, _, res, _, _ = p.detect_for_task(captures_with_image(), TASK)
    assert res.ok, res.describe()
    assert planner.calls == []


def test_llm_rung_fires_after_the_cheap_rungs(tmp_path):
    """Only when relax AND alt_prompts have both already run."""
    planner = FakePlanner([["purple hat"]])
    p = ScriptedPipeline(
        [
            [_od("plushie", 0.09)],  # pass 1: default score
            [_od("plushie", 0.09)],  # pass 2: relaxed to 0.30
            [_od("plushie", 0.09)],  # pass 3: alt prompts, still nothing
            [_od("plushie", 0.09), _od("purple hat", 0.81)],  # pass 4: proposal
        ],
        write_spec(tmp_path),
        planner,
    )
    _, _, res, _, used = p.detect_for_task(captures_with_image(), TASK)

    assert len(planner.calls) == 1
    call = planner.calls[0]
    assert call["target"] == "plushie", "the LLM must be asked for the canonical group"
    assert "plushie" in call["tried"] and "stuffed animal" in call["tried"]
    assert call["has_image"], "the LLM must see the actual frame"

    assert len(p.calls) == 4, p.calls
    assert p.calls[1][1] == pytest.approx(0.30)
    assert "stuffed animal" in p.calls[2][0]
    assert "purple hat" in p.calls[3][0]
    assert "purple hat" in used
    assert res.ok, res.describe()


def test_proposal_is_relabelled_to_the_canonical_name(tmp_path):
    """Nothing downstream may ever see 'purple hat'."""
    planner = FakePlanner([["purple hat"]])
    p = ScriptedPipeline(
        [
            [_od("plushie", 0.09)],
            [_od("plushie", 0.09)],
            [_od("plushie", 0.09)],
            [_od("plushie", 0.09), _od("purple hat", 0.81)],
        ],
        write_spec(tmp_path),
        planner,
    )
    merged, _, res, spec, _ = p.detect_for_task(captures_with_image(), TASK)
    assert res.labels == ["plushie"], res.labels
    assert "purple hat" not in {d.label for d in merged}
    # The returned spec must accept the proposal too, or a caller that
    # re-resolves (a re-detect during recovery) would drop it again.
    assert spec.group_for("purple hat") is not None


# --- degradation -----------------------------------------------------------


def test_llm_error_falls_through_to_on_mismatch(tmp_path):
    """A recovery must never become a failure mode of its own."""
    planner = FakePlanner(raises=RuntimeError("429 quota exhausted"))
    p = ScriptedPipeline([[_od("plushie", 0.09)]] * 3, write_spec(tmp_path), planner)
    with pytest.raises(PromptCountMismatch):
        p.detect_for_task(captures_with_image(), TASK)
    assert len(planner.calls) == 1
    assert len(p.calls) == 3, "a failed proposal must not cost an extra SAM3 pass"


@pytest.mark.parametrize(
    "junk",
    [
        None,
        "purple hat",  # a bare string, not a list
        [],
        ["", "   "],
        [17, {"a": 1}],
        ["plushie", "stuffed animal"],  # only prompts already tried
    ],
)
def test_junk_proposals_fall_through_to_on_mismatch(tmp_path, junk):
    planner = FakePlanner([junk])
    p = ScriptedPipeline([[_od("plushie", 0.09)]] * 3, write_spec(tmp_path), planner)
    with pytest.raises(PromptCountMismatch):
        p.detect_for_task(captures_with_image(), TASK)
    assert len(p.calls) == 3


def test_no_planner_is_not_an_error(tmp_path):
    """A pipeline built with --no-llm still aborts cleanly, not with AttributeError."""
    p = ScriptedPipeline([[_od("plushie", 0.09)]] * 3, write_spec(tmp_path), None)
    with pytest.raises(PromptCountMismatch):
        p.detect_for_task(captures_with_image(), TASK)


def test_disabled_rung_never_calls_the_planner(tmp_path):
    planner = FakePlanner([["purple hat"]])
    p = ScriptedPipeline(
        [[_od("plushie", 0.09)]] * 3,
        write_spec(tmp_path, "  llm_propose_prompts: false\n"),
        planner,
    )
    with pytest.raises(PromptCountMismatch):
        p.detect_for_task(captures_with_image(), TASK)
    assert planner.calls == []


def test_proposal_is_bounded(tmp_path):
    """A chatty model must not turn one retry into twenty SAM3 passes."""
    planner = FakePlanner([[f"widget{i}" for i in range(50)]])
    p = ScriptedPipeline(
        [
            [_od("plushie", 0.09)],
            [_od("plushie", 0.09)],
            [_od("plushie", 0.09)],
            [_od("plushie", 0.09), _od("widget0", 0.7)],
        ],
        write_spec(tmp_path),
        planner,
    )
    p.detect_for_task(captures_with_image(), TASK)
    added = [x for x in p.calls[3][0] if x.startswith("widget")]
    assert 0 < len(added) <= planner.calls[0]["max_prompts"] <= 6, added


# --- the rung on the operator-facing route --------------------------------


def test_llm_rung_reaches_the_detect_route(tmp_path, monkeypatch):
    """Same rung, driven through the real /api/detect handler."""
    from spark_real.routes import detection as detection_routes
    from spark_real.routes import state as route_state
    from spark_real.routes.models import DetectRequest
    from spark_real.tests import test_detect_route_escalation as helpers
    import asyncio

    monkeypatch.setenv("SPARK_DETECTION_FUSION", "0")
    table = helpers.plushie_scene()
    # Both the canonical prompt and the registered alt prompt fail; only the
    # LLM's proposal is visible from the top-down view.
    table[("birdview", "stuffed animal")] = [(0.05, helpers.PLUSHIE_XY, helpers.PLUSHIE_PX)]
    table[("sideview", "stuffed animal")] = [(0.05, helpers.PLUSHIE_XY, helpers.PLUSHIE_PX)]
    table[("birdview", "purple hat")] = [(0.77, helpers.PLUSHIE_XY, helpers.PLUSHIE_PX)]
    table[("sideview", "purple hat")] = [(0.80, helpers.PLUSHIE_XY, helpers.PLUSHIE_PX)]

    sam3 = helpers.FakeSAM3(table)
    pipe = helpers.LivePipeline(sam3)
    planner = FakePlanner([["purple hat"]])
    pipe._planner = planner
    monkeypatch.setattr(route_state, "pipeline", pipe, raising=False)

    resp = detection_routes.detect_objects(
            DetectRequest(prompts=[], instruction=helpers.INSTRUCTION)
        )
    assert getattr(resp, "status_code", 200) == 200, resp
    labels = sorted(d["label"] for d in resp["detections"])
    assert "plushie" in labels, labels
    assert planner.calls and planner.calls[0]["target"] == "plushie"

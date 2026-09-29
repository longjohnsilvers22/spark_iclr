"""The learned-prompts tier: spark_bench's tuned-prompts idea on the live path.

spark_bench/libero_pro/perception.py runs THREE prompt tiers: tuned (a
persistent best-phrase-per-object JSON from offline sweeps), Gemini phase-1,
and adaptive self-consistency. The detect-path port brought over the Gemini
rung but dropped persistence entirely -- _llm_propose_prompts documented it
outright: "a proposal that works is therefore paid for again next run".

This file pins the missing tier: an accepted proposal (one that made the
count gate pass) is recorded per task+group in learned_prompts.json (sibling
of grasp_calibration.json, the repo's existing pattern for small learned
state), and the NEXT failing run tries the cached prompts BEFORE spending a
Gemini call. Cached prompts are still validated by the count gate every run
-- the cache can never force a detection, only skip an API call.
"""

import json

import numpy as np
import pytest

from spark_real.perception.prompt_cache import LearnedPrompts
from spark_real.tests.test_llm_prompt_proposal import (
    TASK,
    FakePlanner,
    ScriptedPipeline,
    _od,
    captures_with_image,
    write_spec,
)

PLUSHIE_MISS = [_od("plushie", 0.09)]          # under min_conf -> count gate fails
HAT_HIT = [_od("purple hat", 0.94)]            # the proposal that rescues


def _pipeline(tmp_path, script, planner, cache=None):
    p = ScriptedPipeline(script, write_spec(tmp_path), planner=planner)
    if cache is not None:
        p._learned_prompts = cache
    return p


# --- unit: the store itself -------------------------------------------------


def test_round_trip_survives_reinstantiation(tmp_path):
    path = tmp_path / "learned_prompts.json"
    LearnedPrompts(path).record(TASK, "plushie", ["purple hat"])
    assert LearnedPrompts(path).lookup(TASK, "plushie") == ["purple hat"]
    assert LearnedPrompts(path).lookup(TASK, "bowl") == []
    assert LearnedPrompts(path).lookup("other task", "plushie") == []


def test_corrupt_file_tolerated_not_fatal(tmp_path):
    path = tmp_path / "learned_prompts.json"
    path.write_text("{not json !!")
    store = LearnedPrompts(path)          # must not raise
    assert store.lookup(TASK, "plushie") == []
    store.record(TASK, "plushie", ["purple hat"])   # and must recover
    assert LearnedPrompts(path).lookup(TASK, "plushie") == ["purple hat"]


# --- the tier on the real detect_for_task ladder ----------------------------


def test_accepted_proposal_is_recorded(tmp_path):
    """LLM rescue -> the winning prompts land in the cache, per task+group."""
    cache = LearnedPrompts(tmp_path / "learned_prompts.json")
    # passes: default miss, relax miss, alt miss, LLM-proposal pass
    script = [PLUSHIE_MISS, PLUSHIE_MISS, PLUSHIE_MISS, HAT_HIT]
    p = _pipeline(tmp_path, script, FakePlanner([["purple hat"]]), cache)
    _, _, res, _, _ = p.detect_for_task(captures_with_image(), TASK)
    assert res.ok
    assert cache.lookup(TASK, "plushie") == ["purple hat"]


def test_cache_rung_rescues_without_a_planner_call(tmp_path):
    """Second run: cached prompts fire BEFORE the LLM; Gemini is not consulted."""
    cache = LearnedPrompts(tmp_path / "learned_prompts.json")
    cache.record(TASK, "plushie", ["purple hat"])
    planner = FakePlanner(raises=AssertionError("planner must not be called"))
    # passes: default miss, relax miss, alt miss, cached-prompt pass
    script = [PLUSHIE_MISS, PLUSHIE_MISS, PLUSHIE_MISS, HAT_HIT]
    p = _pipeline(tmp_path, script, planner, cache)
    merged, _, res, _, used = p.detect_for_task(captures_with_image(), TASK)
    assert res.ok
    assert "purple hat" in used
    # relabelled to the canonical group name, same as alt_prompts
    assert any(d.label == "plushie" for d in merged)


def test_failed_proposals_are_not_recorded(tmp_path):
    """A proposal the count gate rejects must not poison the cache."""
    cache = LearnedPrompts(tmp_path / "learned_prompts.json")
    script = [PLUSHIE_MISS] * 5                      # nothing ever rescues
    p = _pipeline(tmp_path, script, FakePlanner([["purple hat"]]), cache)
    with pytest.raises(Exception):
        p.detect_for_task(captures_with_image(), TASK)
    assert cache.lookup(TASK, "plushie") == []


def test_stale_cache_falls_through_to_llm(tmp_path):
    """Cached prompts that stopped working -> the LLM rung still runs."""
    cache = LearnedPrompts(tmp_path / "learned_prompts.json")
    cache.record(TASK, "plushie", ["stale phrase"])
    planner = FakePlanner([["purple hat"]])
    # passes: default, relax, alt, cached(stale) miss, LLM pass
    script = [PLUSHIE_MISS, PLUSHIE_MISS, PLUSHIE_MISS, PLUSHIE_MISS, HAT_HIT]
    p = _pipeline(tmp_path, script, planner, cache)
    _, _, res, _, _ = p.detect_for_task(captures_with_image(), TASK)
    assert res.ok
    assert planner.calls, "LLM rung should have fired after the stale cache"
    # and the cache is refreshed with the phrase that actually worked
    assert "purple hat" in cache.lookup(TASK, "plushie")


def test_no_cache_attr_behaves_as_before(tmp_path):
    """A pipeline without _learned_prompts is byte-identical to today."""
    script = [PLUSHIE_MISS, PLUSHIE_MISS, PLUSHIE_MISS, HAT_HIT]
    p = _pipeline(tmp_path, script, FakePlanner([["purple hat"]]), cache=None)
    _, _, res, _, _ = p.detect_for_task(captures_with_image(), TASK)
    assert res.ok

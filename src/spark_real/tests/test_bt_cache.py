"""
Offline tests for the BT cache: keying, the capture-once-reuse-forever
loop, restart survival, and label-drift tolerance.

Everything here runs against a temporary library directory with no robot,
no camera, no network and no LLM. The planner is never constructed; the
test asserts on the resolver and the write path directly.
"""

from __future__ import annotations

import pytest
import yaml

from spark_real.bt_label_resolver import LabelResolvingDetectionMap, canon_label
from spark_real.bt_library import (
    BTCacheMiss,
    BTLibrary,
    _near_exact_key,
    _norm,
)
from spark_real.pipeline_execution import ExecutionMixin
from spark_real.planning.bt_seeds import (
    dump_seed,
    ensure_seeded,
    load_seeds,
    parse_seed,
    seed_library,
)

# The seven tasks with human episodes, plus the eighth (empty) directory.
DEMO_TASKS = [
    "put the knife in the tray",
    "put the pen in the bin",
    "put the spoon in the tray",
    "stack the blue block on the gray block",
    "stack the gray block on the blue block",
    "stack the blocks of same color",
    "pick up the plushie and place in bowl",
    "pick up the silverware",
]


def _tree(*labels):
    return {
        "task": " ".join(labels),
        "tree": {
            "type": "sequence",
            "children": [
                {"type": "move_to_keypoint", "params": {"keypoint_label": labels[0]}},
                {"type": "grasp", "params": {"force": 50}},
                {"type": "move_to_keypoint", "params": {"keypoint_label": labels[-1]}},
                {"type": "release", "params": {}},
            ],
        },
    }


@pytest.fixture
def lib(tmp_path):
    return BTLibrary(tmp_path / "bt_library")


# keying


def test_packaged_seeds_cover_the_demo_tasks(tmp_path):
    """
    Every demo task except the documented gap resolves from the packaged
    seeds alone, with the LLM unavailable.
    """
    library = BTLibrary(tmp_path / "lib")
    report = seed_library(library)
    assert not report.errors, report.errors
    assert report.installed, "no seeds installed from configs/bt_seeds"

    resolved = {t: library.lookup(t) for t in DEMO_TASKS}
    missing = [t for t, m in resolved.items() if m is None]
    # "put the pen in the bin" ships as todo: true on purpose -- no pen/bin
    # tree has ever run green, so there is nothing honest to seed.
    assert missing == ["put the pen in the bin"], missing


def test_exact_beats_similarity(lib):
    lib.add("put the knife in the tray", _tree("knife handle", "tray"))
    lib.add("put the spoon in the tray", _tree("spoon handle", "tray"))
    match = lib.lookup("put the knife in the tray")
    assert match.source == "exact"
    assert match.entry.instruction == "put the knife in the tray"


def test_similarity_floor_blocks_a_wrong_plan(lib):
    """
    The regression that made this necessary: "put the pen in the bin"
    matched a tray plan at jaccard 0.20 and executed it.
    """
    lib.add("put the knife in the tray", _tree("knife handle", "tray"))
    assert lib.lookup("put the pen in the bin") is None


def test_the_two_stack_tasks_never_cross_resolve(lib):
    """
    These two tokenise to the SAME set -- jaccard 1.0 -- and mean opposite
    things. Exact match separates them; the order gate protects the fuzzy
    tier for any other phrasing.
    """
    a = "stack the blue block on the gray block"
    b = "stack the gray block on the blue block"
    lib.add(a, _tree("blue block", "gray block"))
    lib.add(b, _tree("gray block", "blue block"))

    assert lib.lookup(a).entry.instruction == a
    assert lib.lookup(b).entry.instruction == b
    assert _near_exact_key(a) != _near_exact_key(b)

    # A phrasing that matches neither exactly must not be silently served
    # the mirror-image tree.
    match = lib.lookup("stack blue on top of gray")
    assert match is None or match.entry.instruction == a


def test_near_exact_absorbs_spelling_and_plurals(lib):
    lib.add("put the spoon in the tray", _tree("spoon handle", "tray"))
    lib.add("stack the gray block on the blue block", _tree("gray block", "blue block"))

    for phrasing in [
        "put the spoons in the tray",
        "pick up the spoon and place it in the tray",
        "PUT THE SPOON IN THE TRAY!",
    ]:
        match = lib.lookup(phrasing)
        assert match is not None, phrasing
        assert match.entry.instruction == "put the spoon in the tray", phrasing

    match = lib.lookup("stack the grey block on a blue block")
    assert match is not None
    assert match.entry.instruction == "stack the gray block on the blue block"


# the capture-once-reuse-forever loop


def test_capture_once_then_served_forever(lib):
    """
    The core loop. One planner-on run that verifies successful, then every
    later run of the task resolves from cache with no LLM.
    """
    score = _tree("knife handle", "tray")
    entry = lib.add("put the knife in the tray", score, success=True)
    assert entry.success == 1

    for _ in range(10):
        match = lib.lookup("put the knife in the tray")
        assert match is not None
        assert match.hash == entry.hash


def test_serving_a_hit_bumps_and_aliases_instead_of_duplicating(lib):
    """
    A cache hit under a new phrasing must reinforce the entry that ran, not
    mint a near-duplicate keyed on the new phrasing. That duplication is
    how the live library reached 88 rows for ~20 tasks.
    """
    entry = lib.add("put the knife in the tray", _tree("knife handle", "tray"))
    lib.bump(entry.hash, True, alias="pick up the knife and put it in the tray")
    assert len(lib) == 1
    assert lib.get(entry.hash).success == 2
    # The new phrasing is now an EXACT hit rather than a fuzzy one.
    match = lib.lookup("pick up the knife and put it in the tray")
    assert match.source == "alias"
    assert match.hash == entry.hash


def test_auto_promotes_after_three_clean_successes(tmp_path):
    lib = BTLibrary(tmp_path / "lib", auto_promote_after=3)
    entry = lib.add("put the knife in the tray", _tree("knife handle", "tray"))
    assert not entry.promoted
    lib.bump(entry.hash, True)
    assert not lib.get(entry.hash).promoted
    lib.bump(entry.hash, True)
    assert lib.get(entry.hash).promoted
    assert lib.promoted_entries()


def test_a_failure_blocks_auto_promotion(tmp_path):
    lib = BTLibrary(tmp_path / "lib", auto_promote_after=3)
    entry = lib.add("put the knife in the tray", _tree("knife handle", "tray"))
    lib.bump(entry.hash, False)
    for _ in range(5):
        lib.bump(entry.hash, True)
    assert not lib.get(entry.hash).promoted


def test_state_survives_a_restart(tmp_path):
    """
    Entries, the per-task pin map and the cache-only toggle all reload from
    disk. The old in-process globals reverted to the env default on every
    server bounce.
    """
    root = tmp_path / "lib"
    lib = BTLibrary(root)
    a = lib.add("put the knife in the tray", _tree("knife handle", "tray"))
    b = lib.add("put the knife in the tray", _tree("knife", "tray"))
    lib.pin("put the knife in the tray", b.hash)
    lib.set_use_cached_bt(True)

    reopened = BTLibrary(root)
    assert len(reopened) == 2
    assert reopened.use_cached_bt is True
    match = reopened.lookup("put the knife in the tray")
    assert match.source == "pin"
    assert match.hash == b.hash != a.hash


def test_pin_is_not_consumed(lib):
    a = lib.add("stack the blocks of same color", _tree("blue block 1", "blue block 2"))
    lib.pin("stack the blocks of same color", a.hash)
    for _ in range(30):
        assert lib.lookup("stack the blocks of same color").hash == a.hash


def test_explicit_temperature_bypasses_the_cache(lib, monkeypatch):
    """
    The /scores page re-plans at a nonzero temperature to generate
    ALTERNATIVE trees for a task. Serving the cache there would hand back
    the very tree the operator is trying to find an alternative to.
    """

    class Cfg:
        save_to_library = True
        bt_plan_mode = "auto"
        output_dir = "output/real_runs"

    class Pipe(ExecutionMixin):
        def __init__(self):
            self._bt_library = lib
            self._planner = None
            self.config = Cfg()

    lib.add("put the knife in the tray", _tree("knife handle", "tray"))
    pipe = Pipe()

    # No temperature -> cache hit, no planner needed.
    pipe.plan("put the knife in the tray", detections=[])
    assert pipe._last_plan_source == "exact"

    # Explicit temperature -> bypasses the cache and reaches for the
    # planner (absent here, so it raises rather than serving the cache).
    with pytest.raises(RuntimeError, match="no planner is configured"):
        pipe.plan("put the knife in the tray", detections=[], temperature=0.7)


def test_cache_only_raises_instead_of_falling_through(lib):
    lib.add("put the knife in the tray", _tree("knife handle", "tray"))
    assert lib.resolve("put the knife in the tray", require_cached=True) is not None
    with pytest.raises(BTCacheMiss):
        lib.resolve("assemble the engine", require_cached=True)


def test_label_corrections_do_not_shift_the_hash(lib):
    """
    execute() pops label_corrections off the score, so the tree written back
    after a run differs from the one served. If that key were hashed, the
    served hash and the stored hash would disagree and every run would mint
    a new row.
    """
    plain = _tree("knife handle", "tray")
    with_corr = dict(plain, label_corrections={"knife": "knife handle"})
    a = lib.add("put the knife in the tray", plain)
    b = lib.add("put the knife in the tray", with_corr)
    assert a.hash == b.hash
    assert len(lib) == 1


# seeds


def test_seed_round_trip(tmp_path):
    lib = BTLibrary(tmp_path / "lib")
    entry = lib.add(
        "put the knife in the tray",
        _tree("knife handle", "tray"),
        objects=["knife handle", "tray"],
    )
    text = dump_seed(entry)
    seed = parse_seed(yaml.safe_load(text))
    assert seed.instruction == "put the knife in the tray"

    fresh = BTLibrary(tmp_path / "lib2")
    report = seed_library(fresh, seeds=[seed])
    assert report.installed
    assert fresh.lookup("put the knife in the tray") is not None


def test_seeding_is_idempotent(tmp_path):
    lib = BTLibrary(tmp_path / "lib")
    ensure_seeded(lib)
    n = len(lib)
    counts = {e.hash: e.success for e in lib.entries()}
    for _ in range(3):
        ensure_seeded(lib)
        seed_library(lib)
    assert len(lib) == n
    assert {e.hash: e.success for e in lib.entries()} == counts


def test_todo_seeds_are_reported_but_not_installed(tmp_path):
    lib = BTLibrary(tmp_path / "lib")
    report = seed_library(lib)
    assert "put_pen_in_bin" in report.todo
    assert lib.lookup("put the pen in the bin") is None


def test_every_seeded_action_exists_in_the_skill_registry():
    """
    A seed that references a primitive the registry does not have would
    fail at execute time, on the rig, mid-episode. Catch it here instead.

    Skipped in the light CI environment: the registry pulls in the
    perception stack.
    """
    registry = pytest.importorskip("spark_real.skills").registry
    live = set(registry.names())
    for seed in load_seeds():
        missing = [a for a in seed.action_types() if a not in live]
        assert not missing, f"{seed.slug} references unknown skills {missing}"


def test_every_packaged_seed_parses():
    seeds = load_seeds()
    assert seeds, "no packaged seeds found"
    for seed in seeds:
        assert _norm(seed.instruction)
        if seed.todo:
            continue
        assert seed.score.get("tree")
        assert seed.keypoint_labels(), seed.slug
        assert seed.action_types(), seed.slug


# label drift


def test_resolver_binds_across_arity_and_spelling():
    scene = LabelResolvingDetectionMap(
        {
            "gray block 1": {"position_3d": [0, 0, 0]},
            "gray block 2": {"position_3d": [1, 0, 0]},
            "blue block": {"position_3d": [2, 0, 0]},
            "knife handle": {"position_3d": [3, 0, 0]},
        }
    )
    # spelling
    assert scene.resolve_label("grey block 1") == "gray block 1"
    # N -> 1: a tree cached in a two-block scene, replayed with one
    assert scene.resolve_label("blue block 2") == "blue block"
    # 1 -> N: a tree cached in a one-block scene, replayed with two
    assert scene.resolve_label("gray block") == "gray block 1"
    # sub-part prompt drift
    assert scene.resolve_label("knife") == "knife handle"
    # genuinely absent stays absent -- never guess
    assert scene.resolve_label("tray") is None
    assert "tray" not in scene
    assert scene.get("tray") is None


def test_resolver_is_transparent_for_exact_hits():
    scene = LabelResolvingDetectionMap({"tray": {"position_3d": [0, 0, 0]}})
    assert scene.get("tray") == {"position_3d": [0, 0, 0]}
    assert "tray" in scene
    assert scene.resolutions == {}
    # Iteration still exposes the real scene, not synonyms.
    assert list(scene.keys()) == ["tray"]


def test_resolver_refuses_an_ambiguous_bind():
    scene = LabelResolvingDetectionMap({"blue block": {}, "gray block": {}})
    assert scene.resolve_label("block") is None


def test_resolver_records_what_it_rebound():
    scene = LabelResolvingDetectionMap({"gray block 1": {"position_3d": [0, 0, 0]}})
    scene.get("grey block")
    assert scene.resolutions == {"grey block": "gray block 1"}


def test_canon_label_folds_variants():
    assert canon_label("grey block 2") == "gray block"
    assert canon_label("spoons 1") == "spoon"
    assert canon_label("gray cube") == "gray block"


def test_unverified_mints_a_countable_row_that_is_not_servable(lib):
    """An abstaining verifier must leave a trace without warming the cache.

    note_unverified used to bump only entries that already existed by hash,
    but an LLM-planned run has no hash yet -- so a task the verifier never
    resolves minted nothing, was re-planned by the LLM next run, abstained
    again, and stayed invisible forever.
    """
    score = _tree("knife handle", "tray")
    for i in range(3):
        entry = lib.note_unverified(None, instruction="put the knife in the tray", score=score)
        assert entry is not None
        assert (entry.success, entry.fail, entry.unverified) == (0, 0, i + 1)

    assert len(lib) == 1
    # success=0 is below lookup()'s min_success floor, so it is countable but
    # never served as a cached plan.
    assert lib.lookup("put the knife in the tray") is None
    assert lib.lookup("put the knife in the tray", min_success=0) is not None

    # A later run that DOES verify promotes the same row, not a duplicate.
    promoted = lib.add("put the knife in the tray", score, success=True)
    assert len(lib) == 1
    assert (promoted.success, promoted.unverified) == (1, 3)
    assert lib.lookup("put the knife in the tray").hash == promoted.hash


def test_note_unverified_without_material_still_returns_none(lib):
    """No hash and nothing to mint from: nothing is invented."""
    assert lib.note_unverified(None) is None
    assert lib.note_unverified("nosuchhash") is None
    assert len(lib) == 0

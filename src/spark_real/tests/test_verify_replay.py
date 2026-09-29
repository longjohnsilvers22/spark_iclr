"""Replay of the recorded human corpus through the verifier. READ-ONLY.

Scope, stated up front: the human corpus is RGB-only. There is no depth and no
camera calibration in it, so no detection can carry a ``position_3d``. This
replay therefore exercises the **2D fallback, the abstain paths and the fusion
plumbing only** -- the 3D predicate path is proved by
``test_success_predicates.py``. A consequence worth stating: under §1.3 (a 2D
pass corroborates but never carries a verdict) NO episode here can fuse to
``pass``, including the true positives. That is the design working, and the
test asserts it.

WHAT THIS REPLAY DOES AND DOES NOT ESTABLISH (read before quoting a number):

*   Under the SHIPPED config (``fallback_2d_containment=1.01``) every camera in
    a depth-less scene abstains, so every episode -- positive, temporal
    negative and cross-task -- fuses to ``unverified``. "0 fused passes on the
    negatives" is therefore TRUE BUT NOT DISCRIMINATIVE: positives score the
    same 0. The shipped-config replay validates fail-closed plumbing, not the
    verifier's ability to tell success from failure.
*   To stop that assertion being vacuous, ``test_canary_*`` below push
    synthetic records through the SAME ``_replay`` code path under the SAME
    shipped config and DO produce a fused pass / a fused negative pass. If the
    harness ever stopped being able to observe a pass, those canaries fail
    first. They are pure-synthetic and run on a fresh clone with no corpus.
*   The discriminative measurement lives under the ``2d_090`` config, and it is
    a NEGATIVE result: see ``test_2d_containment_does_not_discriminate``.

``gripper_positions`` is the *commanded* trigger, bit-identical to
``actions[:,6]``. It is used here to index phases and never as grasp evidence.

Masks come from an offline SAM3 pass over the recorded JPEGs (no hardware);
see ``verify_replay_harvest.py``. The corpus root comes from
``$SPARK_HUMAN_EPISODES`` -- there is no default, so no rig-specific absolute
path is baked into a package that ships publicly. Without the env var or
without the mask cache the corpus-backed tests SKIP with instructions; set
``SPARK_REQUIRE_REPLAY=1`` (release gate) to turn that skip into a failure so a
missing input can never be read as a clean run.
"""

import json
import os
import sys
import types
from collections import Counter
from pathlib import Path

import numpy as np
import pytest

from spark_real.control.success_predicates import (
    ABSTAIN,
    FAIL,
    PASS,
    UNVERIFIED,
    EvalConfig,
    Predicate,
    SceneView,
    evaluate_predicate,
)
from spark_real.control.success_verifier import fuse_votes

CORPUS_ENV = "SPARK_HUMAN_EPISODES"
CACHE_ENV = "SPARK_VERIFY_REPLAY_CACHE"
REQUIRE_ENV = "SPARK_REQUIRE_REPLAY"

_CORPUS_RAW = os.environ.get(CORPUS_ENV, "").strip()
CORPUS = Path(_CORPUS_RAW) if _CORPUS_RAW else None
CACHE = Path(os.environ.get(CACHE_ENV, "output/verify_replay_cache"))
CROSS = Path(str(CACHE) + "_cross")

# The shipped config. Kept as a named constant so the canaries provably run
# against the same numbers success_verifier.VerifyConfig ships.
SHIPPED = EvalConfig(min_conf=0.35, fallback_2d_containment=1.01)
TWO_D_090 = EvalConfig(min_conf=0.35, fallback_2d_containment=0.90)


def _missing_inputs():
    """Names of the inputs this module needs and cannot find."""
    missing = []
    if CORPUS is None:
        missing.append(f"${CORPUS_ENV} is unset (no default: the corpus is rig-local)")
    elif not CORPUS.exists():
        missing.append(f"${CORPUS_ENV}={CORPUS} does not exist")
    if not (CACHE / "index.json").exists():
        missing.append(f"offline SAM3 mask cache {CACHE / 'index.json'} is missing")
    if not (CROSS / "index.json").exists():
        missing.append(f"cross-task mask cache {CROSS / 'index.json'} is missing")
    return missing


MISSING = _missing_inputs()

SKIP_REASON = (
    "corpus replay NOT RUN -- this is a SKIP, not a pass.\n"
    + "".join(f"    missing: {m}\n" for m in MISSING)
    + "  To produce the inputs (READ-ONLY over the corpus, no robot, no camera):\n"
    f"    export {CORPUS_ENV}=/path/to/teleop_episodes\n"
    "    conda run -n sam3 python src/spark_real/tests/verify_replay_harvest.py \\\n"
    f"        --episodes-per-task 9 --out {CACHE}\n"
    "    conda run -n sam3 python src/spark_real/tests/verify_replay_harvest.py \\\n"
    f"        --cross-task --episodes-per-task 9 --out {CROSS}\n"
    f"  Set {REQUIRE_ENV}=1 to make this a hard failure instead (release gate)."
)

if MISSING:
    # Loud at collection time: a silent skip is how "validated" claims rot.
    print(
        "\n" + "=" * 78 + f"\nWARNING  {SKIP_REASON}\n" + "=" * 78,
        file=sys.stderr,
    )


def _require_inputs():
    if not MISSING:
        return
    if os.environ.get(REQUIRE_ENV, "").strip().lower() in ("1", "true", "yes", "on"):
        pytest.fail(SKIP_REASON)
    pytest.skip(SKIP_REASON)


def _load(cache: Path):
    if not (cache / "index.json").exists():
        return []
    index = json.loads((cache / "index.json").read_text())
    packed = np.load(cache / "masks.npz")
    for rec in index:
        for role in ("obj", "container"):
            key = rec.get(f"{role}_key")
            if key is None:
                rec[f"{role}_mask"] = None
                continue
            shape = tuple(rec["shape"])
            bits = np.unpackbits(packed[key])[: shape[0] * shape[1]]
            rec[f"{role}_mask"] = bits.reshape(shape).astype(bool)
    return index


def _det(label, mask, score, xyz=None, obb_minor_m=0.0):
    # position_3d is None for every CORPUS record (no depth); the canaries
    # supply one so the same code path can reach a 3d vote.
    return types.SimpleNamespace(
        label=label,
        confidence=score,
        position_3d=None if xyz is None else np.asarray(xyz, dtype=float),
        mask=mask,
        obb_minor_m=obb_minor_m,
        aspect_ratio=1.0,
        world_major_axis_rad=None,
        slots=None,
    )


def _vote_for(rec, cfg):
    obj = _det(
        "obj",
        rec.get("obj_mask"),
        rec.get("obj_score", 0.0),
        rec.get("obj_xyz"),
    )
    cont = _det(
        "container",
        rec.get("container_mask"),
        rec.get("container_score", 0.0),
        rec.get("container_xyz"),
        rec.get("container_obb_minor_m", 0.0),
    )
    table = {"obj": obj, "container": cont}
    pred = Predicate(
        rec["predicate"],
        {"obj": "obj", ("surface" if rec["predicate"] == "on" else "container"): "container"},
    )
    view = SceneView(camera=rec["camera"], lookup=table.get, held=False)
    return evaluate_predicate(pred, view, cfg)


def _replay(records, cfg):
    """Returns (per-camera vote counter, fused status counter, episode rows)."""
    votes, fused, rows = Counter(), Counter(), []
    episodes = {}
    for rec in records:
        episodes.setdefault((rec["task"], rec["episode"], rec["phase"]), []).append(rec)
    for key, recs in sorted(episodes.items()):
        cam_votes = []
        for rec in recs:
            v = _vote_for(rec, cfg)
            votes[(v.vote, v.mode)] += 1
            cam_votes.append(v)
        status, reason = fuse_votes(cam_votes)
        fused[status] += 1
        rows.append((key, status, [v.vote for v in cam_votes], reason))
    return votes, fused, rows


@pytest.fixture(scope="module")
def replayed():
    """Replay under two configs: the shipped one (2D off) and 2D at 0.90."""
    _require_inputs()
    out = {}
    main, cross = _load(CACHE), _load(CROSS)
    for tag, cfg in (("shipped", SHIPPED), ("2d_090", TWO_D_090)):
        for phase in ("positive", "negative"):
            out[(tag, phase)] = _replay([r for r in main if r["phase"] == phase], cfg)
        out[(tag, "cross")] = _replay([r for r in cross if r["phase"] == "cross"], cfg)
    return out


# ---------------------------------------------------------------------------
# canaries: synthetic records that DO trip the corpus assertions.
# No corpus, no cache, no GPU -- these run on a fresh clone, and they are the
# reason the corpus assertions below are not vacuous.
# ---------------------------------------------------------------------------


def _synth(phase, predicate="inside", **kw):
    rec = {
        "task": "canary",
        "episode": "ep",
        "phase": phase,
        "camera": "camera_0",
        "predicate": predicate,
        "obj_mask": None,
        "obj_score": 0.9,
        "container_mask": None,
        "container_score": 0.9,
    }
    rec.update(kw)
    return rec


def test_canary_shipped_config_can_observe_a_fused_pass():
    """The shipped-config assertion is falsifiable: here is an input that trips it.

    Same ``_replay`` + ``fuse_votes`` path, same ``SHIPPED`` EvalConfig as the
    corpus replay. A depth-carrying detection inside its container fuses to
    ``pass``, so ``assert rows-with-PASS == []`` is a real check, not a tautology.
    """
    recs = [
        _synth(
            "negative",
            obj_xyz=(0.50, 0.00, 0.02),
            container_xyz=(0.50, 0.00, 0.00),
            container_obb_minor_m=0.20,
        )
    ]
    _, fused, rows = _replay(recs, SHIPPED)
    assert fused[PASS] == 1, rows
    # ...and this is exactly the shape test_negatives_never_fuse_to_pass rejects.
    assert [r for r in rows if r[1] == PASS] != []


def test_canary_shipped_config_can_observe_a_fused_fail():
    """Same, for the other verdict: the harness is not stuck on one status."""
    recs = [
        _synth(
            "negative",
            obj_xyz=(1.00, 0.00, 0.02),  # far outside the container OBB
            container_xyz=(0.50, 0.00, 0.00),
            container_obb_minor_m=0.20,
        )
    ]
    _, fused, _ = _replay(recs, SHIPPED)
    assert fused[FAIL] == 1


def test_canary_2d_config_can_observe_a_2d_pass_vote():
    """Under 2d_090 a fully-contained mask pair produces a ``pass/2d`` vote.

    This is what ``test_shipped_config_emits_no_2d_vote`` asserts CANNOT happen
    under the shipped config -- so that assertion has a witness too.
    """
    mask = np.zeros((8, 8), bool)
    mask[2:5, 2:5] = True
    recs = [_synth("positive", obj_mask=mask, container_mask=np.ones((8, 8), bool))]
    votes, fused, _ = _replay(recs, TWO_D_090)
    assert votes[(PASS, "2d")] == 1
    # A 2D pass alone must NOT carry a verdict (§1.3).
    assert fused[UNVERIFIED] == 1
    # And the shipped config must convert that same input into an abstain.
    votes_shipped, fused_shipped, _ = _replay(recs, SHIPPED)
    assert votes_shipped[(ABSTAIN, "none")] == 1
    assert fused_shipped[UNVERIFIED] == 1


def test_abstains_when_a_mask_is_missing():
    """No detection in a camera contributes an abstain, never a pass."""
    rec = _synth("positive", obj_score=0.0, container_mask=np.ones((8, 8), bool))
    assert _vote_for(rec, EvalConfig()).vote == ABSTAIN


# ---------------------------------------------------------------------------
# corpus-backed replay
# ---------------------------------------------------------------------------


def _counts(votes, vote):
    return sum(c for (v, _), c in votes.items() if v == vote)


def test_report(replayed, capsys):
    """The measured numbers, both sides, printed. Read this before quoting one."""
    with capsys.disabled():
        print("\n  corpus replay -- RGB-only corpus, so the 2D path is all there is")
        for tag in ("shipped", "2d_090"):
            print(f"  config={tag}")
            for phase in ("positive", "negative", "cross"):
                votes, fused, rows = replayed[(tag, phase)]
                n = sum(fused.values())
                if not n:
                    print(f"    {phase:9} no records")
                    continue
                vd = {f"{v}/{m}": c for (v, m), c in sorted(votes.items())}
                print(f"    {phase:9} episodes={n:4d} fused={dict(fused)}")
                print(f"              camera votes: {vd}")

        pos = replayed[("shipped", "positive")][1]
        n_pos = sum(pos.values())
        print(
            "\n  POSITIVE SIDE (the half that was previously not reported):\n"
            f"    true-success episodes replayed : {n_pos}\n"
            f"    fused pass (verifier confirms) : {pos[PASS]} "
            f"({100.0 * pos[PASS] / max(n_pos, 1):.0f}% recall)\n"
            f"    fused unverified               : {pos[UNVERIFIED]}\n"
            f"    fused fail (false negatives)   : {pos[FAIL]}\n"
            "    -> on an RGB-only corpus the verifier confirms NO true success.\n"
            "       Recall here is 0 BY CONSTRUCTION (§1.3 needs a 3d pass vote),\n"
            "       not evidence that the verifier works on a depth-carrying rig.\n"
        )
        neg = replayed[("shipped", "negative")][1]
        print(
            "  DISCRIMINATION UNDER THE SHIPPED CONFIG:\n"
            f"    positive fused histogram = {dict(pos)}\n"
            f"    negative fused histogram = {dict(neg)}\n"
            "    -> identical. The shipped-config replay measures fail-closed\n"
            "       plumbing ONLY; it carries zero discriminative evidence.\n"
        )


def test_negative_assertion_is_falsifiable_on_real_records():
    """Mutation check on the REAL negatives, under the SHIPPED config.

    Takes the harvested temporal negatives, gives ONE camera record of ONE
    episode a depth reading that puts the object inside the container, and
    shows the assertion in ``test_negatives_never_fuse_to_pass`` fires. Without
    this the "0 fused passes" result is unfalsifiable on this corpus, because
    no corpus record carries depth at all.
    """
    _require_inputs()
    negatives = [r for r in _load(CACHE) if r["phase"] == "negative"]
    assert negatives, "no negatives harvested"

    _, clean, clean_rows = _replay(negatives, SHIPPED)
    assert [r for r in clean_rows if r[1] == PASS] == []

    poisoned = [dict(negatives[0]), *negatives[1:]]
    poisoned[0].update(
        obj_xyz=(0.50, 0.00, 0.02),
        obj_score=0.9,
        container_xyz=(0.50, 0.00, 0.00),
        container_score=0.9,
        container_obb_minor_m=0.20,
    )
    _, dirty, dirty_rows = _replay(poisoned, SHIPPED)
    assert dirty[PASS] == 1, dict(dirty)
    assert sum(dirty.values()) == sum(clean.values())


@pytest.mark.parametrize("tag", ["shipped", "2d_090"])
@pytest.mark.parametrize("phase", ["negative", "cross"])
def test_negatives_never_fuse_to_pass(replayed, tag, phase):
    """Frames before the first grip (and other tasks' goals) must never pass.

    Falsifiability witness: ``test_canary_shipped_config_can_observe_a_fused_pass``.
    """
    _, fused, rows = replayed[(tag, phase)]
    if not sum(fused.values()):
        pytest.fail(f"{phase} cache not harvested -- an empty replay is not a pass")
    assert [r for r in rows if r[1] == PASS] == []


def test_positives_cannot_pass_without_depth(replayed):
    """Corroborating-only: an all-2D corpus can never reach a fused pass."""
    for tag in ("shipped", "2d_090"):
        _, fused, _ = replayed[(tag, "positive")]
        assert sum(fused.values()) > 0, "no positives in the cache"
        assert fused[PASS] == 0
        assert fused[UNVERIFIED] + fused[FAIL] == sum(fused.values())


def test_shipped_config_is_uniformly_abstaining_so_report_it_that_way(replayed):
    """Pin the vacuity, so nobody re-reads the negative result as discrimination.

    Under the shipped config the positive and negative fused histograms are
    IDENTICAL. Any future change that makes them differ should break this test
    and force the docstrings above to be rewritten.
    """
    pos_votes, pos_fused, _ = replayed[("shipped", "positive")]
    neg_votes, neg_fused, _ = replayed[("shipped", "negative")]
    assert set(pos_fused) == {UNVERIFIED}, dict(pos_fused)
    assert set(neg_fused) == {UNVERIFIED}, dict(neg_fused)
    assert set(pos_votes) == set(neg_votes) == {(ABSTAIN, "none")}


def test_shipped_config_emits_no_2d_vote(replayed):
    """With the 2D fallback off, a depth-less camera abstains and cannot veto.

    Falsifiability witness: ``test_canary_2d_config_can_observe_a_2d_pass_vote``.
    """
    for phase in ("positive", "negative", "cross"):
        votes, _, _ = replayed[("shipped", phase)]
        assert not [m for (_, m) in votes if m == "2d"], (phase, votes)


def test_2d_containment_does_not_discriminate(replayed):
    """Why the 2D fallback ships disabled -- measured, not assumed.

    At 0.90 containment the image-space test passes MORE temporal negatives
    than true positives (the placed object is small/occluded, while an object
    lying on the table in front of a tray is 100% "contained" in the view),
    and it 2D-fails a large share of the true positives, where a fail is a veto.
    """
    pos, _, _ = replayed[("2d_090", "positive")]
    neg, _, _ = replayed[("2d_090", "negative")]
    assert _counts(neg, PASS) >= _counts(pos, PASS), "corpus behaviour changed; re-tune"
    assert _counts(pos, FAIL) > 0, "corpus behaviour changed; re-tune"

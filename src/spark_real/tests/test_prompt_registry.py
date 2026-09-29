"""
Offline tests for the per-task SAM3 prompt registry.

Everything here runs on synthetic detections: no SAM3, no GPU, no camera,
no robot. The property under test is the one a cached behaviour tree depends
on -- that a given physical object gets the same label on every run,
regardless of the order SAM3 happened to report its masks in.
"""

import itertools
import random

import numpy as np
import pytest

from spark_real.perception.mask_geometry import mask_color_stats
from spark_real.perception.prompt_registry import (
    PromptCountMismatch,
    load_registry,
    resolve_labels,
)
from spark_real.perception.spark_perception import ObjectDetection
from spark_real.pipeline_perception import PerceptionMixin


class FakeDetection:
    """
    Duck-typed stand-in for ObjectDetection.

    The registry reads label/confidence/centroid_2d/position_3d/mask_area/
    camera/hsv_median and writes label/role_labels, so those are all it needs.
    """

    def __init__(
        self,
        label,
        confidence=0.5,
        centroid=(0.0, 0.0),
        position=(0.0, 0.0, 0.0),
        mask_area=1000,
        camera="birdview",
        hsv_median=None,
    ):
        self.label = label
        self.confidence = confidence
        self.centroid_2d = tuple(centroid)
        self.position_3d = np.array(position, dtype=float)
        self.mask_area = mask_area
        self.camera = camera
        self.hsv_median = hsv_median
        self.role_labels = None

    def __repr__(self):
        return f"<{self.label} conf={self.confidence:.2f} x={self.position_3d[0]:.2f}>"


@pytest.fixture(scope="module")
def registry():
    return load_registry()


def test_all_eight_corpus_tasks_are_registered(registry):
    """Every task directory in the human corpus resolves to a spec."""
    corpus = [
        "put the knife in the tray",
        "put the pen in the bin",
        "put the spoon in the tray",
        "stack the blue block on the gray block",
        "stack the gray block on the blue block",
        "stack the blocks of same color",
        "pick up the plushie and place in bowl",
        "pick up the silverware",
    ]
    for task in corpus:
        assert registry.lookup(task) is not None, task


def test_mirror_stack_tasks_do_not_collide(registry):
    """
    The two directional stack instructions are bag-of-words identical, so
    only an exact/alias hit can separate them. If this ever regresses, a
    cached BT would stack the blocks in the wrong order.
    """
    a = registry.lookup("stack the blue block on the gray block")
    b = registry.lookup("stack the gray block on the blue block")
    assert a.task != b.task


def test_unknown_instruction_misses(registry):
    """A miss must be None, not a low-similarity false positive."""
    assert registry.lookup("make me a sandwich") is None
    assert registry.lookup("water the plants") is None


def _two_blue_blocks():
    """
    Two blue cubes, deliberately adversarial: the LEFT block (world x=0.35)
    has the LOWER SAM3 confidence, so confidence-rank numbering and geometric
    numbering disagree.
    """
    left = FakeDetection(
        "blue block", confidence=0.31, centroid=(180, 350), position=(0.35, -0.12, 0.02)
    )
    right = FakeDetection(
        "blue block", confidence=0.88, centroid=(280, 310), position=(0.62, 0.05, 0.02)
    )
    return left, right


def test_same_color_labels_are_stable_across_shuffles(registry):
    """
    THE core property. Permute the input order every possible way; the
    physical block at world x=0.35 must always come out as instance 1.
    """
    spec = registry.lookup("stack the blocks of same color")
    seen = set()
    for perm in itertools.permutations(range(2)):
        left, right = _two_blue_blocks()
        dets = [(left, right)[i] for i in perm]
        res = resolve_labels(spec, dets)
        assert res.ok, res.describe()
        seen.add((left.label, right.label))
    assert seen == {("blue block 1", "blue block 2")}, seen


def test_same_color_labels_stable_under_random_shuffles(registry):
    """Same property, with confidences jittered as they would be run to run."""
    spec = registry.lookup("stack the blocks of same color")
    rng = random.Random(0)
    for _ in range(50):
        left, right = _two_blue_blocks()
        left.confidence = rng.uniform(0.2, 0.95)
        right.confidence = rng.uniform(0.2, 0.95)
        dets = [left, right]
        rng.shuffle(dets)
        resolve_labels(spec, dets)
        assert left.label == "blue block 1"
        assert right.label == "blue block 2"


def test_same_color_publishes_color_free_role_aliases(registry):
    """
    The staged pair colour changes between episodes, so the BT addresses
    "same color block N". Both a blue pair and a gray pair must produce the
    same two role names.
    """
    spec = registry.lookup("stack the blocks of same color")

    for color in ("blue block", "gray block"):
        a = FakeDetection(color, confidence=0.4, position=(0.35, -0.1, 0.02))
        b = FakeDetection(color, confidence=0.9, position=(0.60, 0.05, 0.02))
        res = resolve_labels(spec, [b, a])
        assert res.ok, res.describe()
        assert res.acting_group == color
        assert a.role_labels == ["same color block 1"]
        assert b.role_labels == ["same color block 2"]


def test_same_color_gate_fails_when_no_pair(registry):
    """One block of each colour is not a same-colour pair; abort."""
    spec = registry.lookup("stack the blocks of same color")
    dets = [
        FakeDetection("blue block", position=(0.35, 0.0, 0.02)),
        FakeDetection("gray block", position=(0.60, 0.0, 0.02)),
    ]
    res = resolve_labels(spec, dets)
    assert not res.ok
    with pytest.raises(PromptCountMismatch):
        res.raise_if_abort()


def test_single_instance_task_keeps_bare_labels(registry):
    """One instance means no numeric suffix -- what the seeded BTs address."""
    spec = registry.lookup("put the knife in the tray")
    knife = FakeDetection("knife handle", confidence=0.6, position=(0.5, 0.1, 0.0))
    tray = FakeDetection("tray", confidence=0.7, position=(0.4, -0.2, 0.0))
    res = resolve_labels(spec, [tray, knife])
    assert res.ok, res.describe()
    assert sorted(res.labels) == ["knife handle", "tray"]


def test_missing_object_aborts(registry):
    """A missing tray must abort rather than silently record a bad episode."""
    spec = registry.lookup("put the knife in the tray")
    res = resolve_labels(spec, [FakeDetection("knife handle", confidence=0.6)])
    assert not res.ok
    assert "tray" in res.mismatched_groups
    with pytest.raises(PromptCountMismatch):
        res.raise_if_abort()


def test_alt_prompt_detections_are_relabelled_canonically(registry):
    """
    A retry that detects "white block" must come back labelled "gray block",
    so the cached BT's keypoint name still resolves.
    """
    spec = registry.lookup("stack the blue block on the gray block")
    dets = [
        FakeDetection("blue cube", confidence=0.6, position=(0.4, 0.0, 0.02)),
        FakeDetection("white block", confidence=0.5, position=(0.6, 0.0, 0.02)),
    ]
    res = resolve_labels(spec, dets)
    assert res.ok, res.describe()
    assert sorted(res.labels) == ["blue block", "gray block"]


def test_retry_prompts_offered_for_the_failing_group(registry):
    """The escalation path needs to know what to re-prompt with."""
    spec = registry.lookup("stack the blue block on the gray block")
    res = resolve_labels(spec, [FakeDetection("blue block", confidence=0.6)])
    assert not res.ok
    assert "white block" in res.retry_prompts


def test_extra_instances_are_trimmed_geometrically(registry):
    """
    Three candidates for a max-2 group: keep two, and keep the two the
    geometry picks rather than the two SAM3 scored highest.
    """
    spec = registry.lookup("stack the blocks of same color")
    a = FakeDetection("blue block", confidence=0.30, position=(0.30, 0.0, 0.02))
    b = FakeDetection("blue block", confidence=0.35, position=(0.50, 0.0, 0.02))
    c = FakeDetection("blue block", confidence=0.99, position=(0.90, 0.0, 0.02))
    res = resolve_labels(spec, [c, b, a])
    assert [d.label for d in res.detections] == ["blue block 1", "blue block 2"]
    assert res.detections == [a, b]
    assert c in res.extras


def test_resolution_is_idempotent(registry):
    """Re-resolving an already-numbered set must not compound suffixes."""
    spec = registry.lookup("stack the blocks of same color")
    left, right = _two_blue_blocks()
    dets = [left, right]
    for _ in range(3):
        res = resolve_labels(spec, dets)
        dets = res.detections
    assert [d.label for d in dets] == ["blue block 1", "blue block 2"]


def test_per_instance_z_flag_is_set_for_stack_tasks(registry):
    """
    Median-Z flattening across instances makes stacking impossible. Every
    stack task must opt out.
    """
    for task in (
        "stack the blocks of same color",
        "stack the blue block on the gray block",
        "stack the gray block on the blue block",
    ):
        assert registry.lookup(task).per_instance_z is True, task


def test_multi_instance_declared_only_where_needed(registry):
    """
    multi_instance defaults False everywhere in spark_real, so a task that
    needs a second mask must say so; one that does not must stay quiet.
    """
    assert registry.lookup("stack the blocks of same color").multi_instance_prompts == {
        "blue block",
        "gray block",
    }
    assert registry.lookup("put the knife in the tray").multi_instance_prompts == set()


def test_mask_color_stats_matches_known_colors():
    """Hue/saturation on synthetic patches, including the real block colours."""

    def patch(rgb):
        img = np.zeros((40, 40, 3), np.uint8)
        img[:, :] = rgb
        mask = np.zeros((40, 40), bool)
        mask[5:35, 5:35] = True
        return mask_color_stats(img, mask)

    assert patch((255, 0, 0))[0] == pytest.approx(0.0, abs=1e-6)
    assert patch((0, 255, 0))[0] == pytest.approx(120.0, abs=1e-6)
    assert patch((0, 0, 255))[0] == pytest.approx(240.0, abs=1e-6)

    # Grey has no hue and must report zero confidence, so the clustering path
    # cannot reject a gray block for "not matching" a hue.
    assert patch((128, 128, 128))[3] == pytest.approx(0.0, abs=1e-9)

    # The rig's actual cubes: the blue one clears the default saturation
    # floor of 40, the off-white one does not and is treated as colourless.
    assert patch((150, 170, 215))[1] > 40.0
    assert patch((225, 228, 232))[1] < 40.0


def test_mask_color_stats_empty_mask_is_none():
    img = np.zeros((10, 10, 3), np.uint8)
    assert mask_color_stats(img, np.zeros((10, 10), bool)) is None


# ---------------------------------------------------------------------------
# detect_for_task escalation. No SAM3, no camera: detect() is replaced by a
# script of canned per-pass results, so the control flow is what is tested.
# ---------------------------------------------------------------------------


class ScriptedPipeline(PerceptionMixin):
    def __init__(self, script):
        self.script = list(script)
        self.calls = []

    def detect(
        self,
        captures,
        prompts,
        multi_instance=False,
        multi_instance_prompts=None,
        secondary_score=None,
    ):
        self.calls.append((list(prompts), secondary_score))
        return self.script.pop(0)


def _od(label, conf, pos):
    return ObjectDetection(
        label=label,
        confidence=conf,
        centroid_2d=(1.0, 1.0),
        mask_area=900,
        depth_meters=1.0,
        position_3d=np.array(pos, dtype=float),
        camera="birdview",
    )


def test_detect_for_task_succeeds_without_escalating():
    """A clean first pass must not spend extra SAM3 passes."""
    p = ScriptedPipeline(
        [[_od("knife handle", 0.7, (0.5, 0.1, 0.0)), _od("tray", 0.8, (0.4, -0.2, 0.0))]]
    )
    _, _, res, spec, _ = p.detect_for_task({}, "put the knife in the tray")
    assert res.ok
    assert len(p.calls) == 1
    assert p.calls[0][1] is None


def test_detect_for_task_escalates_then_recovers():
    """
    Pass 1 misses the gray block, pass 2 relaxes the secondary-mask score,
    pass 3 adds alt prompts and finds it as "white block" -- which must come
    back under the canonical name the cached BT addresses.
    """
    blue = _od("blue block", 0.8, (0.4, 0.0, 0.02))
    p = ScriptedPipeline(
        [
            [blue],
            [blue],
            [blue, _od("white block", 0.4, (0.6, 0.0, 0.02))],
        ]
    )
    _, _, res, _, used = p.detect_for_task({}, "stack the blue block on the gray block")
    assert res.ok, res.describe()
    assert sorted(res.labels) == ["blue block", "gray block"]

    assert len(p.calls) == 3
    assert p.calls[0][1] is None  # pass 1: default score
    assert p.calls[1][1] == 0.30  # pass 2: relaxed score
    assert "white block" in p.calls[2][0]  # pass 3: alt prompts appended
    assert "white block" in used


def test_detect_for_task_aborts_when_escalation_fails():
    """An object that never appears must abort, not record a bad episode."""
    p = ScriptedPipeline([[_od("knife handle", 0.7, (0.5, 0.0, 0.0))]] * 3)
    with pytest.raises(PromptCountMismatch):
        p.detect_for_task({}, "put the knife in the tray")


def test_detect_for_task_unregistered_task_is_pass_through():
    """An unregistered instruction gets the legacy behaviour and no gate."""
    p = ScriptedPipeline([[_od("sandwich", 0.5, (0.5, 0.0, 0.0))]])
    merged, _, res, spec, _ = p.detect_for_task({}, "make me a sandwich")
    assert res is None and spec is None
    assert len(merged) == 1


def test_merge_detections_legacy_path_still_flattens_z():
    """
    Without a spec, same-label instances keep the historical median-Z
    rewrite. Changing this would silently alter every unregistered caller.
    """
    p = ScriptedPipeline([])
    merged = p.merge_detections(
        [_od("cup", 0.9, (0.4, 0.0, 0.10)), _od("cup", 0.5, (0.7, 0.0, 0.30))]
    )
    zs = {round(float(d.position_3d[2]), 6) for d in merged}
    assert zs == {0.20}
    assert sorted(d.label for d in merged) == ["cup 1", "cup 2"]


def test_merge_detections_spec_path_preserves_z_and_orders_geometrically():
    """A stack task keeps distinct heights and numbers left-to-right."""
    p = ScriptedPipeline([])
    spec = load_registry().lookup("stack the blocks of same color")
    high = _od("blue block", 0.9, (0.62, 0.0, 0.02))
    low = _od("blue block", 0.5, (0.35, 0.0, 0.08))
    merged = p.merge_detections([high, low], spec=spec)
    assert {round(float(d.position_3d[2]), 6) for d in merged} == {0.02, 0.08}

    # merge_detections returns copies, so read the labels off the result.
    by_label = {d.label: d for d in merged}
    assert set(by_label) == {"blue block 1", "blue block 2"}
    # Leftmost wins instance 1 despite having the LOWER confidence.
    assert by_label["blue block 1"].position_3d[0] == pytest.approx(0.35)
    assert by_label["blue block 2"].position_3d[0] == pytest.approx(0.62)

"""Pure tests for the shared sticky-binding module. Real-rig-scale noise cases included."""

import numpy as np

from spark_real.perception.sticky_binding import (
    STICKY_AMBIGUOUS,
    STICKY_ASSOCIATED,
    STICKY_NO_CANDIDATES,
    STICKY_OUT_OF_GATE,
    STATUS_MOVED,
    STATUS_OK,
    STATUS_SELF,
    _labels_match,
    compute_scene_diff,
    dedup_candidate_indices,
    is_self_occluded,
    sticky_associate,
)


def test_sticky_nearest_in_gate_wins():
    r = sticky_associate(
        [0.0, 0.0, 0.0],
        [[0.30, 0.0, 0.0], [0.03, 0.0, 0.0]],
        gate_m=0.12,
    )
    assert r.status == STICKY_ASSOCIATED
    assert r.chosen_index == 1


def test_sticky_out_of_gate_reports_but_caller_keeps_old():
    r = sticky_associate([0.0, 0.0, 0.0], [[0.5, 0.0, 0.0]], gate_m=0.12)
    assert r.status == STICKY_OUT_OF_GATE
    assert r.chosen_index == 0  # reported, never silently adopted


def test_sticky_twin_ambiguity_holds_binding():
    # Two in-gate candidates closer to each other than ambiguity_sep.
    r = sticky_associate(
        [0.0, 0.0, 0.0],
        [[0.05, 0.0, 0.0], [0.06, 0.0, 0.0]],
        gate_m=0.12,
        ambiguity_sep_m=0.03,
    )
    assert r.status == STICKY_AMBIGUOUS
    assert r.chosen_index is None


def test_sticky_no_candidates():
    assert sticky_associate([0, 0, 0], []).status == STICKY_NO_CANDIDATES
    assert sticky_associate(None, [[0, 0, 0]]).status == STICKY_NO_CANDIDATES


def test_dedup_keeps_highest_confidence_of_duplicates():
    pos = [[0.0, 0.0, 0.0], [0.005, 0.0, 0.0], [0.3, 0.0, 0.0]]
    conf = [0.4, 0.9, 0.5]
    kept = dedup_candidate_indices(pos, conf, merge_radius_m=0.02)
    assert kept == [1, 2]


def test_self_occlusion_hold():
    # EE hovering directly over the object at grasp depth: occluded.
    assert is_self_occluded([0.0, 0.0, 0.10], [0.02, 0.0, 0.05])
    # EE far away in XY: not occluded.
    assert not is_self_occluded([0.0, 0.0, 0.10], [0.5, 0.0, 0.05])


def test_labels_match_loose():
    assert _labels_match("blue block 1", "blue block")
    assert _labels_match("plushie", "the plushie")
    assert not _labels_match("bowl", "spoon")


def test_scene_diff_static_scene_with_real_rig_noise():
    # Per-camera disagreement on this rig is 1-2 cm; with a re-measured
    # threshold above that, a static scene must not read as 'moved'.
    rng = np.random.default_rng(0)
    bound = {f"obj{i}": np.array([i * 0.2, 0.0, 0.0]) for i in range(4)}
    current = {
        k: v + rng.normal(0.0, 0.006, size=3) for k, v in bound.items()
    }
    diff = compute_scene_diff(
        bound, current, target_label="obj1", moved_threshold_m=0.03
    )
    assert diff.target_status == STATUS_OK
    assert all(ld.status == STATUS_OK for ld in diff.labels.values())


def test_scene_diff_held_label_is_self_caused():
    bound = {"plushie": np.array([0.0, 0.0, 0.0])}
    current = {"plushie": np.array([0.3, 0.0, 0.2])}
    diff = compute_scene_diff(
        bound, current, target_label="bowl", held_label="plushie"
    )
    assert diff.labels["plushie"].status == STATUS_SELF


def test_scene_diff_genuine_move_flags_target():
    bound = {"block": np.array([0.0, 0.0, 0.0])}
    current = {"block": np.array([0.10, 0.0, 0.0])}
    diff = compute_scene_diff(
        bound, current, target_label="block", moved_threshold_m=0.015
    )
    assert diff.target_status == STATUS_MOVED
    assert not diff.ok

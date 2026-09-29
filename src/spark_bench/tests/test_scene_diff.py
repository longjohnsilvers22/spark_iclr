"""
Unit tests for the pre-primitive scene diff + self-caused-change
subtraction (spark_real.perception.sticky_binding).
"""
import numpy as np
import pytest

from spark_real.perception.sticky_binding import (
    STATUS_MISSING,
    STATUS_MOVED,
    STATUS_OK,
    STATUS_SELF,
    STATUS_UNBOUND,
    STICKY_AMBIGUOUS,
    STICKY_ASSOCIATED,
    STICKY_NO_CANDIDATES,
    STICKY_OUT_OF_GATE,
    compute_scene_diff,
    fuzzy_key,
    sticky_associate,
)


def _p(x, y, z=0.9):
    return np.array([x, y, z])


def test_unmoved_scene_is_ok():
    bound = {'bowl': _p(0.0, 0.0), 'plate': _p(0.2, 0.1)}
    current = {'bowl': _p(0.001, -0.002), 'plate': _p(0.2, 0.1)}
    sd = compute_scene_diff(bound, current, target_label='bowl')
    assert sd.target_status == STATUS_OK
    assert sd.ok
    assert sd.labels['plate'].status == STATUS_OK


def test_moved_target_flagged_with_delta():
    bound = {'bowl': _p(0.0, 0.0)}
    current = {'bowl': _p(0.05, 0.0)}
    sd = compute_scene_diff(bound, current, target_label='bowl',
                              moved_threshold_m=0.015)
    assert sd.target_status == STATUS_MOVED
    assert sd.target_delta_m == pytest.approx(0.05)
    assert not sd.ok


def test_missing_target_flagged():
    bound = {'bowl': _p(0.0, 0.0), 'plate': _p(0.3, 0.0)}
    current = {'plate': _p(0.3, 0.0)}
    sd = compute_scene_diff(bound, current, target_label='bowl')
    assert sd.target_status == STATUS_MISSING


def test_held_object_is_self_caused():
    # The held bowl moved 30 cm (it is in the gripper) - not a scene change.
    bound = {'black bowl': _p(0.0, 0.0)}
    current = {'black bowl': _p(0.3, 0.0, 1.1)}
    sd = compute_scene_diff(bound, current, target_label='plate',
                              held_label='black bowl')
    assert sd.labels['black bowl'].status == STATUS_SELF


def test_held_label_matches_fuzzily():
    # Held label from the BT ('bowl') vs det_map key ('akita black bowl').
    bound = {'akita black bowl': _p(0.0, 0.0)}
    current = {'akita black bowl': _p(0.25, 0.0)}
    sd = compute_scene_diff(bound, current, held_label='bowl')
    assert sd.labels['akita black bowl'].status == STATUS_SELF


def test_near_gripper_change_is_self_caused():
    # A non-target object 5 cm from the gripper that moved: robot bumped
    # it - subtract.
    bound = {'mug': _p(0.0, 0.0), 'plate': _p(0.5, 0.5)}
    current = {'mug': _p(0.04, 0.0), 'plate': _p(0.5, 0.5)}
    sd = compute_scene_diff(bound, current, target_label='plate',
                              gripper_pos=_p(0.02, 0.0, 0.95),
                              self_radius_m=0.12)
    assert sd.labels['mug'].status == STATUS_SELF
    assert sd.target_status == STATUS_OK


def test_target_exempt_from_gripper_radius_subtraction():
    # The primitive's own target sits right under the gripper (pre-close
    # moment) and moved 4 cm: that movement MUST be flagged, not
    # subtracted - it is exactly the perturbation we care about.
    bound = {'bowl': _p(0.0, 0.0)}
    current = {'bowl': _p(0.04, 0.0)}
    sd = compute_scene_diff(bound, current, target_label='bowl',
                              gripper_pos=_p(0.0, 0.0, 0.95),
                              self_radius_m=0.12)
    assert sd.target_status == STATUS_MOVED
    assert sd.target_delta_m == pytest.approx(0.04)


def test_missing_label_under_arm_is_self_occlusion():
    # A non-target label that vanished while the arm hovers over it:
    # attributed to occlusion, not the scene.
    bound = {'mug': _p(0.0, 0.0), 'plate': _p(0.5, 0.5)}
    current = {'plate': _p(0.5, 0.5)}
    sd = compute_scene_diff(bound, current, target_label='plate',
                              gripper_pos=_p(0.0, 0.0, 0.95))
    assert sd.labels['mug'].status == STATUS_SELF


def test_target_resolved_fuzzily():
    bound = {'akita black bowl': _p(0.0, 0.0)}
    current = {'akita black bowl': _p(0.05, 0.0)}
    sd = compute_scene_diff(bound, current, target_label='black bowl')
    assert sd.target_key == 'akita black bowl'
    assert sd.target_status == STATUS_MOVED


def test_unbound_target():
    sd = compute_scene_diff({'bowl': _p(0, 0)}, {'bowl': _p(0, 0)},
                              target_label='xyzzy')
    assert sd.target_status == STATUS_UNBOUND
    assert sd.ok  # unbound is not a scene change verdict


def test_to_meta_json_safe():
    import json
    bound = {'bowl': _p(0.0, 0.0), 'plate': _p(0.3, 0.0)}
    current = {'bowl': _p(0.05, 0.0)}
    sd = compute_scene_diff(bound, current, target_label='bowl', now=123.0)
    meta = sd.to_meta()
    json.dumps(meta)
    assert meta['t'] == 123.0
    assert meta['target_delta_cm'] == pytest.approx(5.0)
    assert meta['labels']['plate']['status'] == STATUS_MISSING


def test_fuzzy_key_priorities():
    d = {'akita black bowl': 1, 'white plate': 2}
    assert fuzzy_key(d, 'akita black bowl') == 'akita black bowl'
    assert fuzzy_key(d, 'black bowl') == 'akita black bowl'   # substring
    assert fuzzy_key(d, 'bowl akita') == 'akita black bowl'   # word overlap
    assert fuzzy_key(d, '') is None


# Position-sticky instance association

def test_sticky_flip_scenario_confidence_order_is_irrelevant():
    # Two identical bowls at (0,0) and (0.3,0).  Bound to the one at
    # (0,0).  Between frames SAM3's confidence order flips, so the FAR
    # instance is listed first (top-1).  Sticky binding must hold the
    # near instance regardless of list order.
    bound = _p(0.0, 0.0)
    frame_a = [_p(0.0, 0.001), _p(0.3, 0.0)]   # ours first
    frame_b = [_p(0.3, 0.0), _p(0.002, 0.0)]   # top-1 flipped
    ra = sticky_associate(bound, frame_a, gate_m=0.12)
    rb = sticky_associate(bound, frame_b, gate_m=0.12)
    assert ra.status == STICKY_ASSOCIATED and ra.chosen_index == 0
    assert rb.status == STICKY_ASSOCIATED and rb.chosen_index == 1
    assert rb.chosen_dist_m < 0.01  # never the 30cm 'flip'


def test_sticky_genuine_move_follows_bound_instance():
    # The bound instance moved 5cm; the twin sits 30cm away.  Nearest
    # is still ours - the 5cm displacement stays visible to the diff.
    bound = _p(0.0, 0.0)
    r = sticky_associate(bound, [_p(0.05, 0.0), _p(0.3, 0.0)], gate_m=0.12)
    assert r.status == STICKY_ASSOCIATED
    assert r.chosen_index == 0
    assert r.chosen_dist_m == pytest.approx(0.05)


def test_sticky_swap_scenario_binding_follows_position():
    # The instances trade places: physically instance B now occupies the
    # old bound position.  Identity-by-position picks it - which is the
    # CORRECT behaviour for the perturbation protocol (the keypoint the
    # plan bound is a position, and physically interchangeable twins
    # make position the only observable identity).
    bound = _p(0.0, 0.0)
    swapped = [_p(0.3, 0.0), _p(0.005, 0.0)]  # 'A' went far, 'B' arrived
    r = sticky_associate(bound, swapped, gate_m=0.12)
    assert r.status == STICKY_ASSOCIATED
    assert r.chosen_index == 1


def test_sticky_ambiguous_keeps_previous_binding():
    # Two in-gate instances 2cm apart: undecidable - do not switch.
    bound = _p(0.0, 0.0)
    r = sticky_associate(bound, [_p(0.04, 0.0), _p(0.06, 0.0)],
                           gate_m=0.12, ambiguity_sep_m=0.03)
    assert r.status == STICKY_AMBIGUOUS
    assert r.chosen_index is None


def test_sticky_two_ingate_but_separable_associates():
    # Both in gate but 8cm apart: nearest wins normally.
    bound = _p(0.0, 0.0)
    r = sticky_associate(bound, [_p(0.03, 0.0), _p(0.11, 0.0)],
                           gate_m=0.12, ambiguity_sep_m=0.03)
    assert r.status == STICKY_ASSOCIATED and r.chosen_index == 0


def test_sticky_out_of_gate_returns_nearest_for_recovery():
    # Nothing within the gate (true large displacement or occluded
    # instance): surface the nearest candidate so the scene diff sees
    # the real delta and routes to recovery.
    bound = _p(0.0, 0.0)
    r = sticky_associate(bound, [_p(0.20, 0.0), _p(0.35, 0.0)], gate_m=0.12)
    assert r.status == STICKY_OUT_OF_GATE
    assert r.chosen_index == 0
    assert r.chosen_dist_m == pytest.approx(0.20)


def test_sticky_degenerate_inputs():
    assert sticky_associate(None, [_p(0, 0)]).status == STICKY_NO_CANDIDATES
    assert sticky_associate(_p(0, 0), []).status == STICKY_NO_CANDIDATES
    assert sticky_associate(_p(0, 0), [None]).status == STICKY_NO_CANDIDATES


def test_dedup_collapses_duplicate_masks():
    from spark_real.perception.sticky_binding import dedup_candidate_indices
    # Two masks of the SAME displaced bowl (1cm apart) + the twin 30cm
    # away.  Dedup keeps the higher-confidence duplicate + the twin, so
    # sticky association measures the true displacement instead of the
    # least-moved duplicate.
    pos = [_p(0.022, 0.0), _p(0.013, 0.0), _p(0.30, 0.0)]
    conf = [0.9, 0.4, 0.8]
    keep = dedup_candidate_indices(pos, conf, merge_radius_m=0.02)
    assert keep == [0, 2]
    # After dedup, a 2.2cm shift is measured as 2.2cm, not 1.3cm.
    r = sticky_associate(_p(0, 0), [pos[i] for i in keep], gate_m=0.12)
    assert r.chosen_dist_m == pytest.approx(0.022)


def test_is_self_occluded_rule():
    from spark_real.perception.sticky_binding import is_self_occluded
    bowl = _p(0.0, 0.0, 0.94)
    # EE hovering 2cm over the bowl: occluded.
    assert is_self_occluded(bowl, _p(0.01, 0.01, 0.96))
    # EE 20cm away in XY: not occluded (free view).
    assert not is_self_occluded(bowl, _p(0.20, 0.0, 0.96))
    # EE at rim-pinch depth (3cm BELOW the detected top surface) still
    # blocks the camera - it-3b regression: this must count as occluded.
    assert is_self_occluded(bowl, _p(0.003, 0.025, 0.91))
    # EE far below the object: camera not blocked from above.
    assert not is_self_occluded(bowl, _p(0.01, 0.0, 0.80))
    assert not is_self_occluded(None, _p(0, 0, 1))
    assert not is_self_occluded(bowl, None)


def test_dedup_preserves_distinct_instances_and_none():
    from spark_real.perception.sticky_binding import dedup_candidate_indices
    pos = [_p(0.0, 0.0), None, _p(0.10, 0.0)]
    keep = dedup_candidate_indices(pos, [0.5, 0.9, 0.6])
    assert keep == [0, 2]
    assert dedup_candidate_indices([_p(0, 0)], [0.5]) == [0]
    assert dedup_candidate_indices([], []) == []

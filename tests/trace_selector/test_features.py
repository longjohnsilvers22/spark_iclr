"""Unit tests for trace_selector feature extraction."""
import math

import pytest

from spark_bench.trace_selector.features import (
    FEATURE_NAMES, PLAN_FEATURES, GROUNDING_FEATURES, TASK_FEATURES,
    extract_features,
)

BT_YAML = """
task: Pick the akita black bowl and place it on the plate
tree:
  type: sequence
  children:
  - type: retry
    params: {max_attempts: 2}
    children:
    - type: sequence
      children:
      - type: move_to_keypoint
        params: {keypoint_label: bowl, offset_z: 0}
      - type: grasp
        params: {force: 60, target_width: 0.04}
      - type: verify_grasp
        params: {}
  - type: move_relative
    params: {dx: 0, dy: 0, dz: 0.2}
  - type: fallback
    children:
    - type: sequence
      children:
      - type: move_to_keypoint
        params: {keypoint_label: plate, offset_z: 0.04}
      - type: release
        params: {tilt_angle: 0}
      - type: verify_placed
        params: {obj: bowl, container: plate, relation: 'on'}
    - type: sequence
      children:
      - type: search_keypoint
        params: {keypoint_label: bowl}
      - type: move_to_keypoint
        params: {keypoint_label: bowl, offset_z: 0}
      - type: grasp
        params: {force: 60, target_width: 0.04}
      - type: move_to_keypoint
        params: {keypoint_label: plate, offset_z: 0.04}
      - type: release
        params: {tilt_angle: 0}
verify:
  all:
  - {pred: held, obj: bowl}
  - {pred: on, obj: bowl, container: plate}
"""

PROMPTS = ['bowl', 'black bowl', 'plate', 'white plate', 'ramekin']
INSTR = 'Pick the akita black bowl and place it on the plate'


def test_feature_names_partition():
    assert FEATURE_NAMES == PLAN_FEATURES + GROUNDING_FEATURES + TASK_FEATURES
    assert len(set(FEATURE_NAMES)) == len(FEATURE_NAMES)


def test_key_order_and_completeness():
    f = extract_features(BT_YAML, prompts=PROMPTS, instruction=INSTR)
    assert list(f.keys()) == FEATURE_NAMES


def test_plan_structure_counts():
    f = extract_features(BT_YAML, prompts=PROMPTS, instruction=INSTR,
                         pick_hint='akita_black_bowl', place_hint='plate')
    assert f['n_move_to_keypoint'] == 4
    assert f['n_grasp'] == 2
    assert f['n_release'] == 2
    assert f['n_move_relative'] == 1
    assert f['n_verify_grasp'] == 1
    assert f['n_verify_placed'] == 1
    assert f['n_search_keypoint'] == 1
    assert f['n_actions'] == 12
    assert f['n_retry'] == 1
    assert f['n_fallback'] == 1
    assert f['retry_max_attempts'] == 2
    assert f['tree_depth'] >= 3


def test_offsets_and_params():
    f = extract_features(BT_YAML, prompts=PROMPTS, instruction=INSTR)
    assert f['first_offset_z'] == 0.0
    assert f['max_offset_z'] == pytest.approx(0.04)
    assert f['place_offset_z'] == pytest.approx(0.04)
    assert f['mr_total_abs_dz'] == pytest.approx(0.2)
    assert f['grasp_force'] == 60
    assert f['grasp_target_width'] == pytest.approx(0.04)


def test_verify_block():
    f = extract_features(BT_YAML, prompts=PROMPTS, instruction=INSTR)
    assert f['has_verify_block'] == 1.0
    assert f['n_verify_clauses'] == 2
    assert f['verify_has_held'] == 1.0
    assert f['verify_has_on'] == 1.0
    assert f['verify_has_inside'] == 0.0


def test_grounding_with_prompts():
    f = extract_features(BT_YAML, prompts=PROMPTS, instruction=INSTR,
                         pick_hint='akita_black_bowl', place_hint='plate')
    assert f['frac_labels_in_vocab'] == 1.0
    assert f['n_labels_missing'] == 0.0
    assert f['place_hint_sim'] == 1.0
    assert 0.0 < f['pick_hint_sim'] < 1.0
    # det_* features are NaN without a det_map
    assert math.isnan(f['det_mean_conf'])


def test_grounding_with_det_map():
    det_map = {
        'bowl': {'confidence': 0.9, 'position_3d': [0.1, 0.0, 0.9]},
        'plate': {'confidence': 0.8, 'position_3d': [0.4, 0.0, 0.9]},
    }
    f = extract_features(BT_YAML, det_map=det_map, instruction=INSTR)
    assert f['det_pick_conf'] == pytest.approx(0.9)
    assert f['det_place_conf'] == pytest.approx(0.8)
    assert f['det_pick_place_dist'] == pytest.approx(0.3)
    assert f['det_mean_conf'] == pytest.approx(0.85)


def test_unknown_label_lowers_vocab_frac():
    yaml_bad = BT_YAML.replace('keypoint_label: bowl',
                               'keypoint_label: zebra')
    f = extract_features(yaml_bad, prompts=PROMPTS, instruction=INSTR)
    assert f['frac_labels_in_vocab'] < 1.0
    assert f['n_labels_missing'] >= 1.0


def test_validator_features():
    va = [{'event': 'invalid_labels_detected', 'invalid': ['a', 'b']},
          {'event': 'snap', 'from': 'a', 'to': 'bowl'}]
    f = extract_features(BT_YAML, prompts=PROMPTS, instruction=INSTR,
                         validator_actions=va)
    assert f['n_validator_snaps'] == 1
    assert f['n_validator_invalid'] == 2
    f2 = extract_features(BT_YAML, prompts=PROMPTS, instruction=INSTR,
                          validator_actions=None)
    assert math.isnan(f2['n_validator_snaps'])


def test_determinism():
    a = extract_features(BT_YAML, prompts=PROMPTS, instruction=INSTR)
    b = extract_features(BT_YAML, prompts=PROMPTS, instruction=INSTR)
    for k in FEATURE_NAMES:
        va, vb = a[k], b[k]
        assert (va == vb) or (math.isnan(va) and math.isnan(vb))


def test_unparseable_plan_yields_nan_structure():
    f = extract_features(':::not yaml {{{', instruction=INSTR)
    assert math.isnan(f['n_actions'])
    assert f['instr_n_words'] > 0  # task features still populated

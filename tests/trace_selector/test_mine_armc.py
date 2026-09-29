"""Unit tests for arm C [gate-candidates] ingestion in the miner."""
import json
import math
import os

from spark_bench.trace_selector import mine as M
from spark_bench.trace_selector.features import extract_features

PLAN_A = ("task: pick the bowl\ntree:\n  type: sequence\n  children:\n"
          "  - type: move_to_keypoint\n    params:\n"
          "      keypoint_label: bowl\n      offset_z: 0\n"
          "  - type: grasp\n    params:\n      force: 60\n")
PLAN_B = ("task: pick the bowl\ntree:\n  type: sequence\n  children:\n"
          "  - type: move_to_keypoint\n    params:\n"
          "      keypoint_label: plate\n      offset_z: 0.04\n"
          "  - type: grasp\n    params:\n      force: 60\n")


def _write_fixture(tmp_path):
    """One arm C run: 2 trials, K=2 candidates each."""
    data_root = tmp_path / 'data'
    log_root = tmp_path / 'logs'
    dirname, suite, arm, logname = M.ARMC_RUNS[0]
    jdir = data_root / dirname
    os.makedirs(jdir)
    os.makedirs(log_root)

    det = {'bowl': {'conf': 0.9, 'pos': [0.1, 0.0, 0.9]},
           'plate': {'conf': 0.8, 'pos': [0.4, 0.0, 0.9]}}
    trials = [
        # trial 0: chosen candidate 0, success, no recovery
        {'trial': 0, 'success': True, 'bt_yaml': PLAN_A,
         'planner': 'gemini', 'det_summary': det},
        # trial 1: all candidates vetoed -> no bt_yaml, recovery only
        {'trial': 1, 'success': False, 'det_summary': det,
         'bt_yaml_recovery': [{'planner': 'gemini', 'bt_yaml': PLAN_B}]},
    ]
    j = {'suite': suite, 'num_trials': 2, 'perturbations': {'position': {
        'per_task': [0.5], 'average': 0.5, 'task_details': [{
            'task_name': 'demo_task', 'instruction': 'pick the bowl',
            'prompts': ['bowl', 'plate'], 'pick': 'bowl', 'place': 'plate',
            'success_rate': 0.5, 'trials': [True, False],
            'trial_meta': trials,
        }]}}}
    with open(jdir / f'{suite}.json', 'w') as fh:
        json.dump(j, fh)

    recs = [
        {'chosen_idx': 0, 'candidates': [
            {'yaml': PLAN_A, 'safe': True, 'unsafe_reason': ''},
            {'yaml': PLAN_B, 'safe': True, 'unsafe_reason': ''}]},
        {'chosen_idx': -1, 'candidates': [
            {'yaml': PLAN_A, 'safe': False, 'unsafe_reason': 'z below'},
            {'yaml': PLAN_B, 'safe': False, 'unsafe_reason': 'z below'}]},
    ]
    with open(log_root / logname, 'w') as fh:
        fh.write('noise line\n')
        for r in recs:
            fh.write(M.GATE_TAG + json.dumps(r) + '\n')
    return str(data_root), str(log_root)


def test_parse_gate_candidates(tmp_path):
    data_root, log_root = _write_fixture(tmp_path)
    _, _, _, logname = M.ARMC_RUNS[0]
    recs = M.parse_gate_candidates(os.path.join(log_root, logname))
    assert len(recs) == 2
    assert recs[0]['chosen_idx'] == 0
    assert len(recs[0]['candidates']) == 2


def test_armc_rows_labeling_and_context(tmp_path):
    data_root, log_root = _write_fixture(tmp_path)
    rows, plans, pairs, sheet = M.mine(data_root, log_root)
    _, _, arm, _ = M.ARMC_RUNS[0]
    armc = [r for r in rows if r['arm'] == arm]
    kinds = sorted(r['row_kind'] for r in armc)
    # trial 0: 1 chosen + 1 unchosen; trial 1 (vetoed): BOTH candidates
    # unchosen context + 1 recovery row
    assert kinds == ['gate_cand_unchosen'] * 3 + ['gate_chosen', 'recovery']
    chosen = next(r for r in armc if r['row_kind'] == 'gate_chosen')
    assert chosen['outcome'] == 1
    assert chosen['shield_safe'] == 1
    assert chosen['has_det_summary'] == 1
    # real det grounding: pick label 'bowl' -> conf 0.9
    assert chosen['det_pick_conf'] == 0.9
    # validator features intentionally NaN (gate ranks pre-validation)
    assert math.isnan(chosen['n_validator_snaps'])
    unchosen = [r for r in armc if r['row_kind'] == 'gate_cand_unchosen']
    assert all(r['outcome'] == '' for r in unchosen)
    # trial 1 (vetoed): only unchosen + recovery rows, no labeled primary
    assert sheet['armc_join'][arm]['status'] == 'joined'
    assert sheet['armc_join'][arm]['veto_mismatch'] == 0
    rec = next(r for r in armc if r['row_kind'] == 'recovery')
    assert rec['outcome'] == 0  # episode failed


def test_armc_join_count_mismatch_skips(tmp_path):
    data_root, log_root = _write_fixture(tmp_path)
    _, _, arm, logname = M.ARMC_RUNS[0]
    # Truncate the log to 1 record -> count mismatch -> arm skipped
    path = os.path.join(log_root, logname)
    lines = [l for l in open(path) if l.startswith(M.GATE_TAG)][:1]
    with open(path, 'w') as fh:
        fh.writelines(lines)
    rows, _, _, sheet = M.mine(data_root, log_root)
    assert 'SKIPPED' in sheet['armc_join'][arm]['status']
    assert not [r for r in rows if r['arm'] == arm
                and r['row_kind'].startswith('gate')]


def test_det_summary_alias_keys():
    det = {'bowl': {'conf': 0.7, 'pos': [0.0, 0.0, 1.0]},
           'plate': {'conf': 0.5, 'pos': [0.3, 0.4, 1.0]}}
    f = extract_features(PLAN_A, det_map=det, instruction='pick the bowl')
    assert f['det_pick_conf'] == 0.7
    assert f['det_mean_conf'] == 0.6
    assert f['det_min_conf'] == 0.5


def test_unlabeled_rows_round_trip_csv(tmp_path):
    data_root, log_root = _write_fixture(tmp_path)
    rows, plans, pairs, sheet = M.mine(data_root, log_root)
    out = tmp_path / 'out'
    ds = M.write_outputs(str(out), rows, plans, pairs, sheet)
    assert ds['n_unlabeled_context'] == 3
    assert ds['n_labeled'] == ds['n_rows'] - 3
    import csv
    back = list(csv.DictReader(open(out / 'dataset.csv')))
    assert sum(1 for r in back if r['outcome'] == '') == 3

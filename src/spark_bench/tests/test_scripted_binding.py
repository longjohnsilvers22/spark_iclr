"""
Regression tests for scripted-plan label binding (_scripted_plan pick /
place selection), encoding the twin-bowl debug scenarios from
docs/debug_journal_spatial_t0.md iterations 1-2.
"""
from types import SimpleNamespace

from spark_bench.libero_pro.planning import _scripted_plan


def _dm(pairs):
    """{label: conf} -> det_map of objects with confidence + position.

    All entries carry a (truthy) agentview mask by default; tests that
    exercise the wrist-only aliasing override ``mask``/``position_3d``.
    """
    return {k: SimpleNamespace(confidence=v, position_3d=[0.0, 0.0, 0.9],
                                 mask='m')
            for k, v in pairs.items()}


def _pick_of(score):
    return score['tree']['children'][0]['params']['keypoint_label']


def test_prompt_priority_beats_later_phrasing():
    # it-0 root cause: 'dark bowl' (5th phrasing, wrong twin) must not
    # shadow 'small round bowl' (1st phrasing, correct instance).
    det = _dm({'small round bowl': 0.93, 'dark bowl': 0.82, 'plate': 0.89})
    s = _scripted_plan(list(det), 'akita_black_bowl', 'plate', 'pick...',
                        det_map=det)
    assert _pick_of(s) == 'small round bowl'


def test_low_confidence_ghost_is_ineligible():
    # it-2 (spatial task 5): 'black bowl' at conf 0.25 matches TWO hint
    # words but is a SAM3 ghost - the eligibility gate must drop it in
    # favour of the confident first-priority phrasing.
    det = _dm({'small round bowl': 0.89, 'black bowl': 0.25,
                'bowl': 0.88, 'plate': 0.85})
    s = _scripted_plan(list(det), 'akita_black_bowl', 'plate', 'pick...',
                        det_map=det)
    assert _pick_of(s) == 'small round bowl'


def test_confidence_is_not_a_tiebreak():
    # it-2 (object task 0): 'Campbell soup can' at conf 0.95 is a
    # high-confidence false positive on the tomato-sauce can; the more
    # discriminative first-priority phrasing must win the tie.
    det = _dm({'can with alphabet letters': 0.88,
                'Campbell soup can': 0.95, 'basket': 0.87})
    s = _scripted_plan(list(det), 'alphabet_soup', 'basket', 'pick...',
                        det_map=det)
    assert _pick_of(s) == 'can with alphabet letters'


def test_all_below_gate_falls_back_to_unfiltered():
    # A scene where every match is weak must still bind something.
    det = _dm({'small round bowl': 0.12, 'plate': 0.15})
    s = _scripted_plan(list(det), 'akita_black_bowl', 'plate', 'pick...',
                        det_map=det)
    assert _pick_of(s) == 'small round bowl'


def test_more_hits_still_beats_priority():
    # Overlap quality outranks priority: a confident 2-hit label beats a
    # confident 1-hit label listed earlier.
    det = _dm({'bowl': 0.9, 'black bowl': 0.9, 'plate': 0.9})
    s = _scripted_plan(list(det), 'akita_black_bowl', 'plate', 'pick...',
                        det_map=det)
    assert _pick_of(s) == 'black bowl'


def test_priority_from_prompt_list_not_detmap_order():
    # it-2b (object task 0): wrist-only detections are APPENDED by the
    # camera merge, so det_map order had 'Campbell soup can' FIRST even
    # though 'can with alphabet letters' is the higher-priority prompt.
    # Priority must come from the ordered prompt list.
    det = _dm({'Campbell soup can': 0.95, 'basket': 0.87,
                'can with alphabet letters': 0.88})  # wrist-appended last
    prompts = ['can with alphabet letters', 'Campbell soup can', 'basket']
    s = _scripted_plan(list(det), 'alphabet_soup', 'basket', 'pick...',
                        det_map=det, prompts=prompts)
    assert _pick_of(s) == 'can with alphabet letters'


def test_wrist_only_binding_aliases_to_colocated_trackable_label():
    # it-6: 'can with alphabet letters' is wrist-only (mask=None) so the
    # agentview refresh loop can never re-detect it; a co-located (2cm)
    # agentview-tracked phrasing of the same object must carry the
    # binding instead.
    det = _dm({'can with alphabet letters': 0.88,
                'Campbell soup can': 0.95, 'basket': 0.87})
    det['can with alphabet letters'].mask = None
    det['can with alphabet letters'].position_3d = [0.10, 0.20, 0.06]
    det['Campbell soup can'].position_3d = [0.11, 0.21, 0.07]
    prompts = ['can with alphabet letters', 'Campbell soup can', 'basket']
    s = _scripted_plan(list(det), 'alphabet_soup', 'basket', 'pick...',
                        det_map=det, prompts=prompts)
    assert _pick_of(s) == 'Campbell soup can'


def test_wrist_only_binding_keeps_label_when_alias_far_away():
    # The lookalike prompt latched a DIFFERENT object 30cm away: no
    # co-location, no alias - keep the discriminative binding.
    det = _dm({'can with alphabet letters': 0.88,
                'Campbell soup can': 0.95, 'basket': 0.87})
    det['can with alphabet letters'].mask = None
    det['can with alphabet letters'].position_3d = [-0.12, -0.23, 0.07]
    det['Campbell soup can'].position_3d = [0.17, 0.03, 0.06]
    prompts = ['can with alphabet letters', 'Campbell soup can', 'basket']
    s = _scripted_plan(list(det), 'alphabet_soup', 'basket', 'pick...',
                        det_map=det, prompts=prompts)
    assert _pick_of(s) == 'can with alphabet letters'


def test_no_det_map_degrades_gracefully():
    labels = ['small round bowl', 'dark bowl', 'plate']
    s = _scripted_plan(labels, 'akita_black_bowl', 'plate', 'pick...')
    assert _pick_of(s) == 'small round bowl'

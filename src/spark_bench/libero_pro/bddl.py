"""
BDDL parsing helpers for LIBERO-PRO tasks.

These functions read LIBERO BDDL goal files to recover language
instructions, object names, and pick/place targets; no shared mutable
state, no closure dependencies on the runner.
"""
from __future__ import annotations

import re
from pathlib import Path  # noqa: F401  -- re-exported for downstream callers


def parse_bddl(bddl_path: str) -> dict:
    """
    Parse a BDDL file to extract language, objects, obj_of_interest, and goal.

    Returns dict with keys: language, objects (list of type names),
    obj_of_interest (list), goal_type ('on'|'open'|'other'), goal_args
    (first predicate args, for back-compat), and goal_pairs (list of
    (pick, place) for *every* On/In predicate joined by And - used by
    multi-step / "put both X and Y in Z" tasks in libero_10).
    """
    result = {'language': None, 'objects': [], 'obj_of_interest': [],
              'goal_type': 'other', 'goal_args': [], 'goal_pairs': []}
    try:
        with open(bddl_path, 'r') as f:
            content = f.read()
    except Exception:
        return result

    # Language
    m = re.search(r'\(:language\s+(.*?)\)', content, re.S)
    if m:
        result['language'] = m.group(1).strip()

    # Objects section: "name1 name2 - type_name"
    obj_block = re.search(r'\(:objects\s+(.*?)\)', content, re.S)
    if obj_block:
        for line in obj_block.group(1).strip().splitlines():
            line = line.strip()
            if ' - ' in line:
                obj_type = line.split(' - ')[-1].strip()
                result['objects'].append(obj_type)

    # Fixtures section
    fix_block = re.search(r'\(:fixtures\s+(.*?)\)', content, re.S)
    if fix_block:
        for line in fix_block.group(1).strip().splitlines():
            line = line.strip()
            if ' - ' in line:
                fix_type = line.split(' - ')[-1].strip()
                result['objects'].append(fix_type)

    # Obj of interest
    ooi_block = re.search(r'\(:obj_of_interest\s+(.*?)\)', content, re.S)
    if ooi_block:
        result['obj_of_interest'] = [
            w.strip() for w in ooi_block.group(1).strip().split('\n')
            if w.strip()
        ]

    # Goal: extract the *entire* (:goal ...) block with balanced-paren
    # matching so compound (And (In a x) (In b x)) forms in libero_10
    # keep every predicate.
    goal_text = _extract_goal_block(content)
    if goal_text:
        # Collect *all* On/In predicates in declaration order (covers
        # both single-predicate and compound And/Or forms).
        on_in_pairs = re.findall(
            r'\((?:On|In)\s+([\w]+)\s+([\w]+)\)', goal_text)
        open_match = re.search(r'\(Open\s+([\w]+)\)', goal_text)
        if on_in_pairs:
            result['goal_type'] = 'on'
            result['goal_args'] = list(on_in_pairs[0])
            result['goal_pairs'] = [list(p) for p in on_in_pairs]
        elif open_match:
            result['goal_type'] = 'open'
            result['goal_args'] = [open_match.group(1)]

    return result


def _extract_goal_block(content: str) -> str:
    """
    Return the body of (:goal ...) using balanced paren matching, so
    compound goals like ``(:goal (And (In a x) (In b x)))`` are kept whole.
    """
    m = re.search(r'\(:goal\s', content)
    if not m:
        return ''
    start = m.end()
    depth = 1
    i = start
    while i < len(content):
        ch = content[i]
        if ch == '(':
            depth += 1
        elif ch == ')':
            depth -= 1
            if depth == 0:
                return content[start:i]
        i += 1
    return content[start:]


# Object type -> SAM3 visual prompts (what it actually looks like)
_OBJ_VISUAL_PROMPTS = {
    'akita_black_bowl': ['small round bowl', 'black bowl', 'bowl', 'round black container', 'dark bowl'],
    'plate': ['plate', 'white plate'],
    'cookies': ['cookie box', 'box of cookies'],
    'glazed_rim_porcelain_ramekin': ['ramekin', 'small white cup'],
    'wooden_cabinet': ['wooden cabinet', 'cabinet'],
    'flat_stove': ['stove', 'flat stove', 'stove burner'],
    'wine_bottle': ['wine bottle', 'green glass bottle'],
    'wine_rack': ['wine rack', 'rack'],
    # Object suite: maximize visual distinctiveness between similar items
    'cream_cheese': ['cream cheese box', 'small white rectangular box'],
    'alphabet_soup': ['can with alphabet letters', 'Campbell soup can'],
    'bbq_sauce': ['dark brown sauce bottle', 'bbq sauce bottle'],
    'butter': ['yellow butter box', 'butter package'],
    'chocolate_pudding': ['chocolate pudding cup', 'brown pudding container'],
    'ketchup': ['red ketchup bottle', 'squeeze bottle with red cap'],
    'milk': ['milk carton', 'tall white carton'],
    'orange_juice': ['orange juice carton', 'carton with orange label'],
    'salad_dressing': ['salad dressing bottle', 'bottle with green label'],
    'tomato_sauce': ['small tomato sauce can', 'short red cylinder'],
    'basket': ['basket', 'wire basket'],
    'table': ['table'],
    'bowl_drainer': ['bowl drainer', 'drying rack'],
    'moka_pot': ['moka pot', 'coffee pot'],
    'porcelain_mug': ['white mug', 'porcelain mug'],
    'white_yellow_mug': ['yellow mug', 'yellow and white mug'],
}


def _strip_instance(name: str) -> str:
    """
    Strip instance suffix and region suffix: 'flat_stove_1_cook_region' -> 'flat_stove'
    """
    m = re.search(r'^(.+?)_(\d+)(?:_.*)?$', name)
    if m:
        return m.group(1)
    return re.sub(r'_(?:contain_)?region$', '', name)


# Object-perturbation prefixes: BDDL variant names like 'bigger_alphabet_soup'
# or 'red_basket' won't match base keys in _OBJ_VISUAL_PROMPTS. Strip these
# to recover the base concept.
_VARIANT_PREFIXES = (
    'bigger', 'larger', 'smaller', 'small', 'big', 'tall', 'short',
    'red', 'blue', 'green', 'yellow', 'white', 'black', 'dark', 'orange',
    'pink', 'brown', 'gray', 'grey', 'purple',
)


def _strip_variant_prefix(name: str) -> str:
    """
    Strip a color/size variant prefix if the stripped base is in
    _OBJ_VISUAL_PROMPTS. 'bigger_alphabet_soup' -> 'alphabet_soup'. If the
    stripped form isn't known, return the input unchanged so legitimate
    multi-token names ('akita_black_bowl', 'wine_bottle') aren't damaged.
    """
    for prefix in _VARIANT_PREFIXES:
        token = prefix + '_'
        if name.startswith(token):
            base = name[len(token):]
            if base in _OBJ_VISUAL_PROMPTS:
                return base
    return name


# Spatial adjectives that may appear in instructions and need to propagate
# into SAM3 prompts when paired with a keyword. e.g. "middle drawer" should
# become a SAM3 prompt, not just "drawer".
_SPATIAL_ADJS = (
    'middle', 'top', 'bottom', 'front', 'back', 'left', 'right',
    'upper', 'lower', 'central',
)


def get_task_prompts_for_suite(suite_name: str, task_id: int, task,
                                bddl_path: str = None) -> dict:
    """
    Get SAM3 prompts by parsing the BDDL file directly.

    Extracts language, objects, goal from BDDL - no hardcoded per-task config.
    Builds focused prompt list: pick + place + spatial landmarks only.
    """
    # Parse BDDL
    bddl = parse_bddl(bddl_path) if bddl_path else {}
    instruction = bddl.get('language') or getattr(task, 'language', task.name.replace('_', ' '))
    ooi = bddl.get('obj_of_interest', [])
    goal_type = bddl.get('goal_type', 'other')
    goal_args = bddl.get('goal_args', [])

    # Goal pairs: list of (pick, place) for *every* On/In predicate.  Used
    # by libero_10 "put both X and Y in Z" compound goals.  When the BDDL
    # has a single predicate (every other suite) this is a 1-element list
    # and is equivalent to (pick, place).
    goal_pairs_raw = bddl.get('goal_pairs') or []
    goal_pairs: list[tuple[str, str]] = []
    for raw_pick, raw_place in goal_pairs_raw:
        goal_pairs.append((_strip_instance(raw_pick),
                            _strip_instance(raw_place)))

    # Derive pick/place from goal or obj_of_interest
    pick = 'none'
    place = 'none'
    if goal_pairs:
        pick, place = goal_pairs[0]
    elif goal_type == 'on' and len(goal_args) >= 2:
        pick = _strip_instance(goal_args[0])
        place = _strip_instance(goal_args[1])
    elif goal_type == 'open' and goal_args:
        pick = 'none'
        place = 'none'
    elif len(ooi) >= 2:
        pick = _strip_instance(ooi[0])
        place = _strip_instance(ooi[1])
    elif len(ooi) == 1:
        pick = _strip_instance(ooi[0])

    # Multi-pick/place lists (de-duped, ordered).  Used by the planner to
    # emit a compound BT chaining each (pick_i, place_i) pair.
    picks: list[str] = []
    places: list[str] = []
    if goal_pairs:
        seen_p: set[str] = set()
        for p, q in goal_pairs:
            if p not in seen_p:
                picks.append(p)
                seen_p.add(p)
        seen_q: set[str] = set()
        for p, q in goal_pairs:
            if q not in seen_q:
                places.append(q)
                seen_q.add(q)
    elif pick != 'none':
        picks = [pick]
        if place != 'none':
            places = [place]

    # Build FOCUSED SAM3 prompts: ALL pick targets + place targets + spatial
    # landmarks.  For compound goals (libero_10) every pick object must be
    # SAM3-detected, not only the first.
    prompts = []
    seen = set()

    # Priority 1: Pick object prompts for EVERY pick target.  Fall back to
    # variant-stripped base (object perturbation: 'bigger_alphabet_soup' ->
    # 'alphabet_soup').  Cap each object at 2 prompts when there are >1
    # picks so the total SAM3 budget stays manageable.
    pick_budget = 5 if len(picks) <= 1 else 2
    for p in picks:
        if p == 'none':
            continue
        p_key = p if p in _OBJ_VISUAL_PROMPTS else _strip_variant_prefix(p)
        if p_key in _OBJ_VISUAL_PROMPTS and p_key not in seen:
            prompts.extend(_OBJ_VISUAL_PROMPTS[p_key][:pick_budget])
            seen.add(p_key)

    # Priority 2: Place target prompts (max 2 each)
    for q in places:
        if q == 'none':
            continue
        q_key = q if q in _OBJ_VISUAL_PROMPTS else _strip_variant_prefix(q)
        if q_key in _OBJ_VISUAL_PROMPTS and q_key not in seen:
            prompts.extend(_OBJ_VISUAL_PROMPTS[q_key][:2])
            seen.add(q_key)

    # Priority 3: Spatial landmarks from instruction
    # If instruction mentions objects by name for spatial context, detect them too
    inst_lower = instruction.lower()
    spatial_kws = ['between', 'next to', 'on the', 'from', 'near', 'behind', 'in front']
    has_spatial = any(kw in inst_lower for kw in spatial_kws)
    if has_spatial:
        # Add landmarks mentioned in instruction for spatial reasoning
        for obj_type, vis_prompts in _OBJ_VISUAL_PROMPTS.items():
            if obj_type in seen:
                continue
            # Check if any visual prompt word appears in instruction
            obj_words = obj_type.replace('_', ' ').lower()
            if obj_words in inst_lower or any(vp.lower() in inst_lower for vp in vis_prompts[:1]):
                prompts.extend(vis_prompts[:1])  # Just 1 prompt per landmark
                seen.add(obj_type)

    # Priority 4: Task-relevant keywords.
    # Always include the stove itself (cook region) - not only the knob - so
    # place targets that reference "front of the stove" can be grounded.
    # If a spatial adjective precedes the keyword (e.g. "middle drawer"),
    # emit hybrid prompts so SAM3 can match the qualified phrase too.
    for kw, vp in [('basket', ['basket']), ('drawer', ['drawer handle', 'drawer']),
                    ('stove', ['stove knob', 'stove', 'flat stove']),
                    ('rack', ['wine rack', 'rack']),
                    ('microwave', ['microwave'])]:
        if kw in inst_lower and kw not in seen:
            for adj in _SPATIAL_ADJS:
                if f'{adj} {kw}' in inst_lower:
                    for base in vp:
                        prompts.append(f'{adj} {base}')
            prompts.extend(vp)
            seen.add(kw)

    if not prompts:
        prompts = [w for w in inst_lower.split() if len(w) > 3][:6]
    elif len(prompts) > 10:
        prompts = prompts[:10]

    # Per-pick-instance counts: when the BDDL goal references two instances
    # of the same type ("put both moka pots on the stove" -> moka_pot_1 +
    # moka_pot_2), pick *deduplicates* to a single entry but the task still
    # needs two grasps.  Preserve the raw instance count so the planner can
    # emit two move/grasp pairs (using e.g. an N-th instance keypoint).
    pick_counts: list[int] = []
    for p in picks:
        n = sum(1 for raw_pick, _ in goal_pairs if raw_pick == p)
        pick_counts.append(max(n, 1))

    return {
        'prompts': prompts,
        'instruction': instruction,
        'pick': pick,
        'place': place,
        # Compound-goal extras - single-pair tasks have len==1 lists.
        'picks': picks,
        'places': places,
        'pick_counts': pick_counts,
        'goal_pairs': goal_pairs,
    }


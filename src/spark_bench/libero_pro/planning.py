"""
LIBERO-PRO planning helpers - Gemini, scripted fallback, DSL-typed planner.

Each planner returns a dict ``score`` (a YAML-shaped behaviour tree) for
downstream execution; failures fall through to lower-fidelity planners so
the pipeline never aborts mid-trial because the LLM was rate-limited.
"""
from __future__ import annotations

import os
import logging
from typing import Optional

import numpy as np
import yaml
from PIL import Image, ImageDraw, ImageFont

from google.genai import types as gtypes

from spark_real.perception.spark_perception import SPARKPerception
from spark_real.planning.spark_planner import SPARKPlanner
from spark_dsl import DEFAULT_LIBRARY
from spark_dsl.prompt_builder import PromptBuilder
from spark_real.utils.env_flags import env_flag

# BT library (few-shot retrieval + replay) is optional - planning never blocks
# on it being absent.
try:
    from spark_real.planning.bt_library import get_library
except Exception:  # pragma: no cover - bt_library is optional
    get_library = None  # type: ignore[assignment]

logger = logging.getLogger(__name__)


__all__ = [
    '_get_sam3',
    '_annotate_masks',
    '_gemini_plan',
    '_gemini_plan_via_dsl',
    '_bt_frozen',
    '_library_replay_plan',
    '_scripted_plan',
    'bt_replay_enabled',
]


import copy as _copy
import hashlib as _hashlib

_GEMINI_PLAN_CACHE: dict[str, dict] = {}
_GEMINI_CACHE_HITS = 0
_GEMINI_CACHE_MISSES = 0


def _gemini_cache_key(instruction: str, labels: list,
                       picks: Optional[list],
                       places: Optional[list]) -> str:
    raw = (instruction.strip() + '|' + ','.join(sorted(labels or [])) + '|'
           + ','.join(sorted(picks or [])) + '|'
           + ','.join(sorted(places or [])))
    return _hashlib.sha1(raw.encode('utf-8')).hexdigest()[:16]


def _gemini_cache_enabled() -> bool:
    return env_flag('SPARK_GEMINI_CACHE', True)


def _bt_frozen() -> bool:
    """True when the no-adaptation control is active.

    ``SPARK_BT_FROZEN=1`` freezes the BT library for a whole run: no few-shot
    examples are retrieved into the planner prompt and no successful score is
    written back. This is the control for "does SPARK improve within a suite
    because it accumulates its own successes?".

    Read here because the LIBERO-PRO path does not import
    ``spark_real.pipeline_execution``.
    """
    return env_flag('SPARK_BT_FROZEN')


def _bt_library_examples_hint(instruction: str, k: int = 3,
                                *, is_compound: bool = False
                                ) -> Optional[str]:
    """
    Voyager-style few-shot: top-k similar successful BTs as Gemini context.

    The on-disk ``bt_library`` is populated by ``_store_in_bt_library``
    after every successful libero_pro trial. Toggle off via
    ``SPARK_BT_LIBRARY_FEWSHOT=0`` for ablation.

    ``is_compound=True`` skips retrieval: single pick-place templates as
    few-shot for a compound instruction ("put both X and Y in Z") bias Gemini
    toward dropping the second pick.
    """
    if is_compound:
        return None
    if os.environ.get('SPARK_BT_LIBRARY_FEWSHOT', '1').lower() in (
            '0', 'false', 'no', 'off'):
        return None
    if _bt_frozen():
        return None
    if get_library is None:
        return None
    try:
        lib = get_library()
        examples = lib.get_examples_prompt(instruction, k=k)
        return examples or None
    except Exception as e:  # noqa: BLE001 - never block planning on library errors
        logger.debug("BT library few-shot skipped: %s", e)
        return None


def bt_replay_enabled() -> bool:
    """
    ``SPARK_BT_REPLAY=1`` enables the zero-LLM library-replay planner.
    """
    return os.environ.get('SPARK_BT_REPLAY', '0').lower() in (
        '1', 'true', 'yes', 'on')


def _library_replay_plan(instruction: str, det_map: Optional[dict],
                          *, k: int = 1, min_overlap: float = 0.25,
                          picks: Optional[list] = None,
                          places: Optional[list] = None,
                          pick_counts: Optional[list] = None
                          ) -> Optional[dict]:
    """
    Zero-LLM planner: retrieve top-1 BT from ``bt_library`` and replay.

    Returns a BT score dict (with ``__planner='replay'`` set) when a
    library match clears the overlap threshold, ``None`` otherwise so the
    caller can fall through to Gemini.

    When ``picks`` indicates a multi-pair compound, the retrieved BT must
    also be compound (action_count > 5).

    Stored keypoint labels ("akita black bowl") may not appear in the
    current ``det_map`` ("bowl"); labels are not rewritten here. The
    downstream ``_run_label_validator`` fuzzy-snaps them with at most one
    retry.
    """
    if get_library is None:
        return None
    if not instruction or not instruction.strip():
        return None
    is_compound = _is_compound_task(picks, places, pick_counts)
    try:
        lib = get_library()
        examples = lib.retrieve(instruction, k=max(k, 5))
    except Exception as e:
        logger.debug("BT-replay: retrieve failed: %s", e)
        return None
    if not examples:
        return None

    query_words = set(instruction.lower().split())
    if not query_words:
        return None

    def _action_count(score: dict) -> int:
        tree = score.get('tree', {}) if isinstance(score, dict) else {}
        n = [0]
        def _walk(node):
            if not isinstance(node, dict):
                return
            t = node.get('type')
            if t in ('sequence', 'selector'):
                for c in node.get('children', []) or []:
                    _walk(c)
            elif t is not None:
                n[0] += 1
        _walk(tree)
        return n[0]

    def _overlap(frag) -> float:
        fw = set(frag.instruction.lower().split())
        if not fw:
            return 0.0
        return len(query_words & fw) / max(len(query_words), len(fw))

    # Rank candidates: word overlap + compound bonus when needed.
    scored = []
    for frag in examples:
        ov = _overlap(frag)
        ac = _action_count(frag.score)
        is_frag_compound = ac > 5
        if is_compound and not is_frag_compound:
            continue  # skip single pick-place when query is compound
        scored.append((ov, ac, frag))

    if not scored:
        return None

    scored.sort(key=lambda x: (-x[0], -x[1]))
    best_ov, best_ac, best_frag = scored[0]
    if best_ov < min_overlap:
        return None

    score = _copy.deepcopy(best_frag.score)
    if isinstance(score, dict):
        score.setdefault('__raw_yaml',
                          yaml.safe_dump(score, sort_keys=False))
        score['__planner'] = 'replay'
        score['__replay_source'] = best_frag.instruction
        score['__replay_overlap'] = round(best_ov, 3)
    print(f"[bt-replay HIT] overlap={best_ov:.2f} actions={best_ac} "
          f"src={best_frag.instruction[:70]!r}", flush=True)
    return score


def _is_compound_task(picks, places, pick_counts) -> bool:
    """
    Match _build_hint_extras's compound-goal detector.

    Compound = multiple unique picks OR pick_counts indicating multi-pick.
    Used to gate few-shot retrieval and shadow-sim diversity, both of
    which corrupt compound BTs (see _bt_library_examples_hint docstring).
    """
    if picks is None or places is None:
        return False
    if len(picks) > 1:
        return True
    if pick_counts and any(c > 1 for c in pick_counts):
        return True
    return False


_DEJAVU_FONT = "/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf"
_GEMINI_KEY_FILE = "~/spark/src/.gemini_api_key"
_DSL_MACROS_YAML = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "..", "spark_dsl", "macros.yaml")

_MASK_PALETTE = [
    (255, 50, 50), (50, 200, 50), (50, 120, 255), (255, 200, 40),
    (255, 80, 200), (80, 220, 220), (255, 140, 40), (160, 80, 255),
    (40, 255, 140), (200, 200, 80),
]

_sam3_instance: Optional[SPARKPerception] = None


# Singleton + small utilities

def _get_sam3() -> SPARKPerception:
    """
    Singleton SAM3 perception loader - first call pays the model-load cost.
    """
    global _sam3_instance
    if _sam3_instance is None:
        print("[SAM3] Loading...")
        _sam3_instance = SPARKPerception(sam3_threshold=0.03)
    return _sam3_instance


def _ensure_gemini_key() -> None:
    """
    Populate ``GEMINI_API_KEY`` from the on-disk key file if missing.
    """
    if 'GEMINI_API_KEY' in os.environ:
        return
    key_file = os.path.expanduser(os.environ.get("SPARK_GEMINI_KEY_FILE", _GEMINI_KEY_FILE))
    with open(key_file) as f:
        lines = [l.strip() for l in f.readlines() if l.strip()]
    if lines:
        os.environ['GEMINI_API_KEY'] = lines[0]


def _to_pil(image) -> Optional[Image.Image]:
    """
    Coerce a numpy array / PIL image / None to PIL.  ``None`` passes through.
    """
    if image is None:
        return None
    return image if isinstance(image, Image.Image) else Image.fromarray(image)


def _build_hint_extras(pick_hint: str, place_hint: str, instruction: str,
                        det_map: Optional[dict],
                        no_bddl_hints: bool,
                        picks: Optional[list] = None,
                        places: Optional[list] = None,
                        pick_counts: Optional[list] = None) -> list[str]:
    """
    Compose BDDL pick/place hints + 3D position hints into a list of lines.

    When ``picks`` / ``places`` are supplied with >1 entry (libero_10
    "put both X and Y in Z" compound goals), emit explicit per-pair
    instructions so the planner chains move_to_keypoint -> grasp ->
    move_to_keypoint -> release once per pair.  Falls back to the legacy
    single pick/place hint when only one pair is present.
    """
    lines: list[str] = []
    # Multi-pair compound goal - supersedes the singular pick/place hint.
    has_compound = (picks is not None and places is not None
                     and (len(picks) > 1 or (pick_counts
                          and any(c > 1 for c in pick_counts))))
    if not no_bddl_hints and has_compound:
        # Build (pick, place, count) triples covering each unique pick.
        pcs = pick_counts or [1] * len(picks)
        place_for_pick = (places[0] if len(places) == 1
                           else None)  # broadcast common place
        steps: list[str] = []
        for i, p in enumerate(picks):
            cnt = pcs[i] if i < len(pcs) else 1
            q = place_for_pick if place_for_pick is not None else (
                places[i] if i < len(places) else (places[-1] if places
                else 'none'))
            p_clean = p.replace('_', ' ')
            q_clean = q.replace('_', ' ') if q != 'none' else 'destination'
            if cnt > 1:
                steps.append(
                    f'  {i + 1}. {p_clean} (x{cnt}) -> {q_clean}')
            else:
                steps.append(
                    f'  {i + 1}. {p_clean} -> {q_clean}')
        lines.append(
            'COMPOUND GOAL - emit a sequence covering EVERY pair below.')
        lines.extend(steps)
        lines.append(
            'For each pair: move_to_keypoint(pick) -> grasp -> '
            'move_to_keypoint(place, offset_z=0.10 first, vary z_offset '
            'across pairs to avoid stacking) -> release.  Do NOT stop '
            'after the first pair - chain them all in a single sequence.')
    elif not no_bddl_hints:
        if pick_hint and pick_hint != 'none':
            pick_clean = pick_hint.replace('_', ' ')
            if pick_clean not in instruction.lower():
                lines.append(f'Pick object: {pick_clean}')
        if place_hint and place_hint != 'none':
            place_clean = place_hint.replace('_', ' ')
            if place_clean not in instruction.lower():
                lines.append(f'Place target: {place_clean}')
    if det_map:
        pos_lines = []
        for label, det in det_map.items():
            p = det.position_3d
            if p is None:
                continue
            pos_lines.append(f'  {label}: ({p[0]:.3f}, {p[1]:.3f}, {p[2]:.3f})')
        if pos_lines:
            lines.append('Detected object positions (x,y,z meters):')
            lines.extend(pos_lines)
    # push_object guidance: source (keypoint_label) + target (target_label)
    # so the primitive computes direction/distance from perception.
    instr_lower = instruction.lower()
    if 'push' in instr_lower:
        lines.append(
            'If the task is to push one object TO/TOWARD another (e.g. '
            '"push the plate to the front of the stove"), emit push_object '
            'with BOTH keypoint_label (the source object to push) and '
            'target_label (the destination region/object). Example:\n'
            '  - type: push_object\n'
            '    params:\n'
            '      keypoint_label: "plate"\n'
            '      target_label: "stove_front"\n'
            'Only use push_direction/push_distance when no destination '
            'object is detectable.'
        )
    return lines


# Mask annotation overlay

def _annotate_masks(rgb: np.ndarray, det_map: dict) -> np.ndarray:
    """
    Draw SAM3 mask overlays + labels on the agentview RGB image.

    Each detection's mask is colored from a fixed palette and labelled at
    its centroid.  Gemini sees this overlay alongside the raw RGB so it
    can cross-check whether each label actually covers the right object.
    """
    pil = Image.fromarray(rgb).convert('RGBA')
    overlay = Image.new('RGBA', pil.size, (0, 0, 0, 0))
    try:
        font = ImageFont.truetype(_DEJAVU_FONT, 14)
    except Exception:
        font = ImageFont.load_default()
    H, W = rgb.shape[:2]
    for i, (label, det) in enumerate(det_map.items()):
        mask = getattr(det, 'mask', None)
        if mask is None:
            continue
        color = _MASK_PALETTE[i % len(_MASK_PALETTE)]
        mask_rgba = np.zeros((H, W, 4), dtype=np.uint8)
        mask_bool = mask > 0
        mask_rgba[mask_bool] = (*color, 110)
        overlay = Image.alpha_composite(overlay,
                                          Image.fromarray(mask_rgba, mode='RGBA'))
        ys, xs = np.where(mask_bool)
        if len(xs) == 0:
            continue
        cx, cy = int(xs.mean()), int(ys.mean())
        draw = ImageDraw.Draw(overlay)
        text = f"{label} ({det.confidence:.2f})"
        bbox = draw.textbbox((cx + 2, cy - 8), text, font=font)
        draw.rectangle(bbox, fill=(0, 0, 0, 180))
        draw.text((cx + 2, cy - 8), text, fill=(*color, 255), font=font)
    return np.array(Image.alpha_composite(pil, overlay).convert('RGB'))


# Gemini plan (DSL-typed prompt)

def _gemini_plan_via_dsl(instruction: str, labels: list, pick_hint: str = '',
                          place_hint: str = '', det_map: Optional[dict] = None,
                          rgb_image=None, wrist_image=None,
                          no_bddl_hints: bool = False,
                          picks: Optional[list] = None,
                          places: Optional[list] = None,
                          pick_counts: Optional[list] = None,
                          temperature: float = 0.3) -> dict:
    """
    DSL-typed Gemini planner - falls back to ``_gemini_plan`` on any failure.

    Constructs the prompt via ``spark_dsl.PromptBuilder`` (typed primitive
    list, auto-loaded macros, BDDL/position hints).  Validates the result
    against the typed library before returning so an invalid plan never
    reaches the executor.
    """
    try:
        _ensure_gemini_key()
        builder = PromptBuilder(library=DEFAULT_LIBRARY,
                                  macros_yaml=_DSL_MACROS_YAML)
        extras = _build_hint_extras(pick_hint, place_hint, instruction,
                                      det_map, no_bddl_hints,
                                      picks=picks, places=places,
                                      pick_counts=pick_counts)
        # Suppress few-shot during diversity sampling (see _gemini_plan).
        # Also suppress for compound tasks where library templates corrupt
        # the chain (see _bt_library_examples_hint docstring). Few-shot stays on
        # at the default planning temperature (0.3); only higher diversity-
        # sampling temperatures suppress it.
        if temperature <= 0.3:
            examples_text = _bt_library_examples_hint(
                instruction, k=3,
                is_compound=_is_compound_task(picks, places, pick_counts))
            if examples_text:
                extras.append(examples_text)
        prompt = builder.build(
            task_instruction=instruction,
            detected_objects=labels,
            extra_hints='\n'.join(extras) if extras else None,
        )

        # Reuse SPARKPlanner's client for retry/backoff/key rotation.
        planner = SPARKPlanner(llm_backend='gemini')
        client = planner._get_client()
        contents: list = [prompt]
        scene = _to_pil(rgb_image)
        if scene is not None:
            contents.append(scene)
            if det_map:
                try:
                    contents.append(Image.fromarray(
                        _annotate_masks(np.array(scene), det_map)))
                except Exception:
                    pass
        wrist = _to_pil(wrist_image)
        if wrist is not None:
            contents.append(wrist)

        resp = client.models.generate_content(
            model=planner.model, contents=contents,
            config=gtypes.GenerateContentConfig(temperature=float(temperature)),
        )
        raw_text = (resp.text or '').strip()
        text = raw_text
        if '```' in text:
            text = text.split('```', 2)[1]
            if text.startswith('yaml'):
                text = text[4:]
            elif text.startswith('json'):
                text = text[4:]
        score = yaml.safe_load(text.strip()) or {}

        # Auto-wrap a bare sequence in {tree: ...} so the response shape
        # matches the legacy planner output.
        if (isinstance(score, dict) and 'tree' not in score
                and score.get('type') == 'sequence'):
            score = {'tree': score}

        _ast, errors = DEFAULT_LIBRARY.validate_bt(score)
        if errors:
            raise RuntimeError(f"DSL validation: {errors[:2]}")
        # Stash raw model output for per-trial logging (additive, ignored
        # by the executor which only looks at score['tree']).
        if isinstance(score, dict):
            score.setdefault('__raw_yaml', raw_text)
            score.setdefault('__planner', 'gemini_dsl')
        return score
    except Exception as e:
        print(f"[DSL Gemini] {e}; falling back to legacy planner")
        return _gemini_plan(instruction, labels, pick_hint, place_hint,
                             det_map=det_map, rgb_image=rgb_image,
                             wrist_image=wrist_image,
                             no_bddl_hints=no_bddl_hints,
                             picks=picks, places=places,
                             pick_counts=pick_counts,
                             temperature=temperature)


# Gemini plan (legacy free-form prompt via SPARKPlanner)

def _gemini_plan(instruction: str, labels: list, pick_hint: str = '',
                 place_hint: str = '', det_map: Optional[dict] = None,
                 rgb_image=None, wrist_image=None,
                 no_bddl_hints: bool = False,
                 picks: Optional[list] = None,
                 places: Optional[list] = None,
                 pick_counts: Optional[list] = None,
                 temperature: float = 0.0) -> dict:
    """
    Generate YAML BT via Gemini, with RGB image + 3D positions for
    spatial reasoning.  Returns a scripted fallback plan on any failure.

    ``picks`` / ``places`` / ``pick_counts`` (optional, libero_10 compound
    goals): when supplied with >1 entry the prompt instructs Gemini to
    chain a move/grasp/place sequence per (pick, place) pair instead of
    emitting a single pick-place plan.
    """
    global _GEMINI_CACHE_HITS, _GEMINI_CACHE_MISSES
    cache_key = _gemini_cache_key(instruction, labels, picks, places)
    # Temperature > 0 = diversity sampling for shadow-sim best-of-N. The cache
    # would otherwise replay the same deterministic plan, defeating diversity.
    use_cache = _gemini_cache_enabled() and temperature == 0.0
    if use_cache and cache_key in _GEMINI_PLAN_CACHE:
        _GEMINI_CACHE_HITS += 1
        cached = _copy.deepcopy(_GEMINI_PLAN_CACHE[cache_key])
        cached.setdefault('__cache_hit', True)
        print(f"[bt-cache HIT] key={cache_key} hits={_GEMINI_CACHE_HITS} "
              f"misses={_GEMINI_CACHE_MISSES}", flush=True)
        return cached
    try:
        _ensure_gemini_key()
        planner = SPARKPlanner(llm_backend='gemini')

        # Hint instruction: append BDDL pick/place inline, then 3D positions.
        hint_instruction = instruction
        has_compound = (picks is not None and places is not None
                         and (len(picks) > 1 or (pick_counts
                              and any(c > 1 for c in pick_counts))))
        if not no_bddl_hints and has_compound:
            pcs = pick_counts or [1] * len(picks)
            place_for_pick = places[0] if len(places) == 1 else None
            step_lines = []
            for i, p in enumerate(picks):
                cnt = pcs[i] if i < len(pcs) else 1
                q = place_for_pick if place_for_pick is not None else (
                    places[i] if i < len(places) else (places[-1] if places
                    else 'none'))
                p_clean = p.replace('_', ' ')
                q_clean = (q.replace('_', ' ') if q != 'none'
                            else 'destination')
                step_lines.append(
                    f'  {i + 1}. {p_clean}{f" (x{cnt})" if cnt > 1 else ""}'
                    f' -> {q_clean}')
            hint_instruction += (
                '\nCOMPOUND GOAL - emit a sequence covering EVERY pair:\n'
                + '\n'.join(step_lines)
                + '\nFor each pair: move_to_keypoint(pick) -> grasp -> '
                'move_to_keypoint(place, offset_z>=0.10, vary z_offset '
                'across pairs to avoid stacking) -> release. Chain ALL '
                'pairs in a single sequence (do NOT stop at the first).')
        elif not no_bddl_hints:
            if pick_hint and pick_hint != 'none':
                pick_clean = pick_hint.replace('_', ' ')
                if pick_clean not in instruction.lower():
                    hint_instruction += f'. Pick object: {pick_clean}'
            if place_hint and place_hint != 'none':
                place_clean = place_hint.replace('_', ' ')
                if place_clean not in instruction.lower():
                    hint_instruction += f'. Place target: {place_clean}'
        if det_map:
            pos_lines = [f'  {lbl}: ({d.position_3d[0]:.3f},'
                          f' {d.position_3d[1]:.3f},'
                          f' {d.position_3d[2]:.3f})'
                          for lbl, d in det_map.items()
                          if d.position_3d is not None]
            if pos_lines:
                hint_instruction += ('\nDetected object positions '
                                       '(x,y,z meters):\n'
                                       + '\n'.join(pos_lines))
        # Compound clause-count hint (text-derived from instruction only,
        # NOT a BDDL leak), added even with no_bddl_hints=True. Without it,
        # perception flakes can drop one object from labels and the planner
        # emits only the visible-object phase.
        instr_lower = instruction.lower()
        compound_markers = (' and ', ' then ', ' both ', ' two ',
                             ' three ', ' each ', 'first ')
        n_markers = sum(1 for m in compound_markers if m in instr_lower)
        if n_markers >= 1 or has_compound:
            n_clauses = max(n_markers + 1, 2)
            hint_instruction += (
                f'\nCOMPOUND TASK ({n_clauses}+ clauses detected from '
                'instruction text). You MUST emit ONE sequence containing '
                'EVERY clause as its own sub-chain. Even if perception '
                'currently shows only one of the mentioned objects, the BT '
                'must include ALL clauses - the executor re-perceives '
                'between primitives. Emitting only one phase (e.g. just '
                'pick, or just close) guarantees BDDL failure.')
        # Suppress few-shot during diversity sampling: temperature > 0 means
        # shadow-sim is asking for K different BTs. Always-the-same prior
        # examples bias every sample toward the same plan. Also suppress
        # for compound tasks where library templates corrupt the chain.
        if temperature == 0.0:
            examples_text = _bt_library_examples_hint(
                instruction, k=3,
                is_compound=_is_compound_task(picks, places, pick_counts))
            if examples_text:
                hint_instruction += '\n' + examples_text
        # push_object trajectory-aware hint: prefer emitting target_label
        # (a SAM3 detection label) over hardcoding push_direction /
        # push_distance.  Source = keypoint_label, target = target_label.
        if 'push' in instruction.lower():
            hint_instruction += (
                '\nFor push_object: when the task is to push one object TO/'
                'TOWARD another (e.g. "push the plate to the front of the '
                'stove"), emit BOTH keypoint_label (the source to push) and '
                'target_label (a SAM3 detection label of the destination). '
                'The primitive infers push_direction and push_distance from '
                'the two 3D positions, so push_direction/push_distance are '
                'optional. Example:\n'
                '  - type: push_object\n'
                '    params:\n'
                '      keypoint_label: "plate"\n'
                '      target_label: "stove"\n'
                'Only fall back to push_direction/push_distance when no '
                'destination label exists in the detected objects.'
            )

        pil_image = _to_pil(rgb_image)
        pil_annotated: Optional[Image.Image] = None
        if pil_image is not None and det_map:
            try:
                pil_annotated = Image.fromarray(
                    _annotate_masks(np.array(pil_image), det_map))
            except Exception:
                pil_annotated = None

        # Detection details for SPARKPlanner Phase 2.
        detection_details: list[dict] = []
        if det_map:
            for lbl, det in det_map.items():
                dd: dict = {'label': lbl, 'confidence': det.confidence}
                if det.position_3d is not None:
                    p = det.position_3d
                    dd['position_3d'] = (p.tolist() if hasattr(p, 'tolist')
                                          else list(p))
                detection_details.append(dd)

        score = planner.generate_score(
            hint_instruction, annotated_image=pil_image,
            keypoint_labels=labels,
            detection_details=detection_details,
            mask_overlay_image=pil_annotated,
            wrist_image=_to_pil(wrist_image),
            temperature=temperature)

        # Cost accounting: one greppable line per Gemini call. The planner
        # instance is per-call, so its usage_log would otherwise be dropped.
        for u in getattr(planner, 'usage_log', []):
            print(f"[gemini-usage] {u}", flush=True)

        # Empty plan -> scripted fallback.
        tree = score.get('tree', {})

        def count_actions(node):
            if node.get('type') in ('sequence', 'selector'):
                return sum(count_actions(c) for c in node.get('children', []))
            return 1

        if count_actions(tree) == 0:
            return _scripted_plan(labels, pick_hint, place_hint, instruction,
                                    picks=picks, places=places,
                                    det_map=det_map)
        try:
            raw_yaml = yaml.safe_dump(score, sort_keys=False)
            score.setdefault('__raw_yaml', raw_yaml)
            score.setdefault('__planner', 'gemini')
            n_acts = count_actions(tree)
            print(f"[bt-emit] actions={n_acts} compound={has_compound} "
                  f"picks={picks} places={places} "
                  f"yaml={raw_yaml[:240].replace(chr(10), ' | ')}",
                  flush=True)
        except Exception:
            pass
        if use_cache:
            _GEMINI_CACHE_MISSES += 1
            _GEMINI_PLAN_CACHE[cache_key] = _copy.deepcopy(score)
            print(f"[bt-cache MISS-store] key={cache_key} "
                  f"hits={_GEMINI_CACHE_HITS} misses={_GEMINI_CACHE_MISSES}",
                  flush=True)
        return score
    except Exception as e:
        print(f"[Gemini error] {e}")
        return _scripted_plan(
            labels,
            pick_hint or (labels[0] if labels else ''),
            place_hint or (labels[-1] if labels else ''),
            instruction, picks=picks, places=places, det_map=det_map)


# Scripted fallback

_PLACE_KEYWORDS = {'basket', 'plate', 'table', 'cabinet', 'stove', 'rack', 'bowl'}


def _scripted_plan(labels, pick_hint, place_hint, instruction,
                    picks=None, places=None, det_map=None, prompts=None):
    """
    Simple scripted pick-place plan when Gemini is unavailable.

    When ``picks`` is supplied with >1 entry (libero_10 compound goals),
    emit a chained pick/place sequence - one pick->grasp->place->release
    cycle per pick.  The shared ``place`` is reused with a small
    increasing z-offset so successive drops don't stack.
    """
    # Wipe/scrub tasks are not pick-place: emit a hand-authored move-to +
    # constrained_scrub over the detected dirt region (acts as the planner with
    # no LLM). The scrub primitive runs the observation-gated raster.
    il = (instruction or '').lower()
    if any(kw in il for kw in ('wipe', 'scrub', 'dirt')):
        dirt = None
        for l in labels:
            if any(kw in l.lower() for kw in ('dirt', 'marks', 'stain', 'spill', 'smudge')):
                dirt = l
                break
        if dirt is None and labels:
            dirt = labels[0]
        if dirt is not None:
            return {'tree': {'type': 'sequence', 'children': [
                {'type': 'move_to_keypoint',
                 'params': {'keypoint_label': dirt, 'offset_z': 0.05}},
                {'type': 'constrained_scrub',
                 'params': {'workpiece_label': dirt, 'duration': 10.0,
                            'amplitude': 0.15, 'frequency': 2.0,
                            'normal_force': 15.0, 'axis': 'x'}},
            ]}}

    # Push tasks are not pick-place: emit push_object with source +
    # destination labels so the primitive derives direction/distance from
    # perception (same shape Gemini emits on the eval path).  Generic:
    # source = BDDL pick hint (or the first non-place label), target =
    # BDDL place hint matched against the detected labels.
    # Best-overlap label binding: argmax on (hint-word hits, prompt-table
    # priority).
    def _conf_of(l: str) -> float:
        if not det_map or l not in det_map:
            return 1.0
        try:
            return float(getattr(det_map[l], 'confidence', 1.0))
        except Exception:
            return 1.0

    def _kw_match(hint: str) -> Optional[str]:
        kws = [w for w in (hint or '').lower().replace('_', ' ').split()
                if len(w) > 2]
        if not kws:
            return None
        # Two-pass eligibility gate: a sub-noise detection must not win the
        # binding on word overlap alone. If nothing passes the gate, fall
        # back to the full list.
        for min_conf in (0.2, 0.0):
            best = None
            best_key = (0, 0.0, 0)
            for idx, l in enumerate(labels):
                if _conf_of(l) < min_conf:
                    continue
                lw = [w for w in l.lower().split() if w]
                hits = sum(1 for w in kws if w in l.lower())
                # Coverage: fraction of the LABEL's words the hint accounts
                # for.  Breaks hits-ties toward the label that is fully
                # explained by the hint ('stove' beats 'stove knob' for
                # hint 'flat_stove' - the knob is a different part).
                cover = (sum(1 for w in lw
                              if any(w in k or k in w for k in kws))
                         / max(len(lw), 1))
                key = (hits, cover, -idx)
                if hits and key > best_key:
                    best, best_key = l, key
            if best is not None:
                return best
        return None

    # Drawer-opening tasks: emit open_drawer (executor derives level +
    # pull direction from the instruction and perception).  A compound
    # "open ... and put X inside" additionally chains the pick + a place
    # bound to the drawer handle label - the executor's drawer-aware
    # placement retargets it to the pulled-out tray.
    if 'open' in il and any(k in il for k in ('drawer', 'layer', 'cabinet')):
        handle_lbl = next((l for l in labels if 'handle' in l.lower()),
                           (labels[0] if labels else None))
        children: list = [{'type': 'open_drawer',
                            'params': ({'keypoint_label': handle_lbl}
                                        if handle_lbl else {})}]
        pick_lbl = _kw_match(pick_hint) if pick_hint else None
        if pick_lbl is not None and handle_lbl is not None:
            children.extend([
                {'type': 'move_to_keypoint',
                 'params': {'keypoint_label': pick_lbl}},
                {'type': 'grasp', 'params': {'force': 100}},
                {'type': 'move_relative', 'params': {'dz': 0.20}},
                {'type': 'move_to_keypoint',
                 'params': {'keypoint_label': handle_lbl, 'offset_z': 0.10}},
                {'type': 'release'},
            ])
        return {'tree': {'type': 'sequence', 'children': children}}

    if 'push' in il:
        src = _kw_match(pick_hint)
        if src is None:
            for l in labels:
                if not any(kw in l.lower() for kw in _PLACE_KEYWORDS):
                    src = l
                    break
        tgt = _kw_match(place_hint)
        if src is not None:
            p: dict = {'keypoint_label': src}
            if tgt is not None:
                p['target_label'] = tgt
            return {'tree': {'type': 'sequence', 'children': [
                {'type': 'push_object', 'params': p},
            ]}}

    # Multi-pick chained plan.
    if picks and len(picks) > 1:
        # Match each pick keyword to the best label in the detection set.
        def _match(hint: str) -> Optional[str]:
            kws = [w for w in hint.lower().replace('_', ' ').split()
                    if len(w) > 2]
            if not kws:
                return None
            best = None
            best_hits = 0
            for l in labels:
                ll = l.lower()
                hits = sum(1 for w in kws if w in ll)
                if hits > best_hits:
                    best = l
                    best_hits = hits
            return best if best_hits else None

        pick_lbls = [_match(p) for p in picks]
        place_lbl = (_match(places[0]) if places else None) or (
            _match(place_hint) if place_hint else None)
        if not place_lbl:
            for l in labels:
                if any(kw in l.lower() for kw in _PLACE_KEYWORDS):
                    place_lbl = l
                    break
        children: list = []
        for i, pl in enumerate(pick_lbls):
            if pl is None or place_lbl is None:
                continue
            # Stagger z-offset across pairs so drops don't stack on top
            # of each other (basket interior is ~5cm deep - 0.05/0.08).
            z = 0.05 + 0.03 * i
            children.extend([
                {'type': 'move_to_keypoint',
                 'params': {'keypoint_label': pl}},
                {'type': 'grasp', 'params': {'force': 100}},
                {'type': 'move_relative', 'params': {'dz': 0.20}},
                {'type': 'move_to_keypoint',
                 'params': {'keypoint_label': place_lbl, 'offset_z': z}},
                {'type': 'release'},
            ])
        if children:
            return {'tree': {'type': 'sequence', 'children': children}}

    # Best-overlap label binding with a confidence-eligibility gate and
    # prompt-priority tie-break:
    #
    # * a low-confidence detection is not binding-grade evidence - a
    #   2-hit label at conf 0.25 (SAM3 ghost on a lookalike) must not
    #   outrank a 1-hit label at conf 0.89 that latched the real object,
    #   so labels below _BIND_MIN_CONF are ineligible while any eligible
    #   match exists;
    # * confidence is NOT a valid tie-break either - visually confusable
    #   distractors produce HIGH-confidence false positives ('Campbell
    #   soup can' at 0.95 on the tomato-sauce can).  The prompt table is
    #   curated most-discriminative-first, so among equal-hit labels the
    #   one EARLIEST in the emitted prompt list wins.  Priority comes
    #   from ``prompts`` (the ordered list the runner detected with),
    #   NOT det_map insertion order - wrist-camera-only detections are
    #   APPENDED by the merge, which silently demotes the most
    #   discriminative phrasing whenever only the wrist saw it.
    _BIND_MIN_CONF = 0.3
    # A word hit on a near-threshold detection is not evidence of an
    # object: a 0.03-confidence ghost phrase bound the scripted plan to
    # empty table (0 cm control, object task 2) while the real bottle sat
    # under a phrase with no hint words. The confidence-free pass keeps
    # a floor so the caller's own fallback wins over a ghost.
    _BIND_FALLBACK_MIN_CONF = 0.1
    prompt_rank = {p: i for i, p in enumerate(prompts or [])}

    def _rank(i, l):
        # Lower is better: prompt-table index, falling back to det_map
        # position for labels not in the prompt list.
        return prompt_rank.get(l, len(prompt_rank) + i)

    def _best_label(hint, exclude=()):
        if not hint:
            return None
        words = [w for w in hint.lower().replace('_', ' ').split()
                 if len(w) > 2]
        if not words:
            return None

        def _conf(l):
            if det_map is None:
                return None
            return float(getattr(det_map.get(l), 'confidence', 0.0) or 0.0)

        def _pass(min_conf):
            best, best_key = None, None
            for i, l in enumerate(labels):
                if l in exclude:
                    continue
                c = _conf(l)
                if min_conf is not None and c is not None and c < min_conf:
                    continue
                hits = sum(1 for w in words if w in l.lower())
                if hits == 0:
                    continue
                key = (-hits, _rank(i, l))
                if best_key is None or key < best_key:
                    best_key, best = key, l
            return best

        return _pass(_BIND_MIN_CONF) or _pass(_BIND_FALLBACK_MIN_CONF)

    def _hits(hint, l):
        words = [w for w in hint.lower().replace('_', ' ').split()
                 if len(w) > 2]
        return sum(1 for w in words if w in l.lower())

    def _trackable_alias(hint, l):
        # Adaptation-observability guard: binding
        # to a WRIST-ONLY detection (no agentview mask) makes the target
        # invisible to the agentview-based refresh loop - every pre-close
        # retarget silently disappears and only blind recovery remains.
        # If a CO-LOCATED (<=5 cm) hint-matching label exists WITH an
        # agentview mask, bind that label instead: same physical object,
        # chosen via the discriminative phrasing's position, tracked via
        # the phrasing the verification camera can actually re-detect.
        if det_map is None or l is None:
            return l
        d = det_map.get(l)
        if d is None or getattr(d, 'mask', None) is not None:
            return l
        p = getattr(d, 'position_3d', None)
        if p is None:
            return l
        p = np.asarray(p, dtype=float)[:3]
        best_alt, best_r = None, None
        for i, l2 in enumerate(labels):
            if l2 == l or _hits(hint, l2) == 0:
                continue
            d2 = det_map.get(l2)
            if d2 is None or getattr(d2, 'mask', None) is None:
                continue
            p2 = getattr(d2, 'position_3d', None)
            if p2 is None:
                continue
            if float(np.linalg.norm(
                    np.asarray(p2, dtype=float)[:3] - p)) > 0.05:
                continue
            r = _rank(i, l2)
            if best_r is None or r < best_r:
                best_r, best_alt = r, l2
        return best_alt or l

    pick = _trackable_alias(pick_hint, _best_label(pick_hint))
    place = _trackable_alias(place_hint, _best_label(place_hint))
    # A single label must not serve as both pick and place: prefer the pick
    # binding (BDDL pick hints are more specific) and rebind the place to
    # the next-best distinct label with the same ranking.
    if pick is not None and place == pick:
        alt = _best_label(place_hint, exclude={pick})
        place = _trackable_alias(place_hint, alt)
        if place == pick:
            place = alt

    if pick is None:
        for l in labels:
            if not any(kw in l.lower() for kw in _PLACE_KEYWORDS):
                pick = l; break
    if place is None:
        for l in labels:
            if l != pick:
                place = l; break

    if not pick or not place:
        return None

    return {
        'tree': {'type': 'sequence', 'children': [
            {'type': 'move_to_keypoint', 'params': {'keypoint_label': pick}},
            {'type': 'grasp', 'params': {'force': 100}},
            {'type': 'move_relative', 'params': {'dz': 0.20}},
            {'type': 'move_to_keypoint',
             'params': {'keypoint_label': place, 'offset_z': 0.05}},
            {'type': 'release'},
        ]}
    }

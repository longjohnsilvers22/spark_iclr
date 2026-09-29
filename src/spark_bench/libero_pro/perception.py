"""
Perception helpers for the LIBERO-PRO fair runner.

Covers tuned-prompt / Gemini-prompt / adaptive-prompt selection,
dual-camera SAM3 detection + agent/wrist confidence merge, and
re-detection during the post-failure recovery loop.  The runner calls a
single ``detect_scene`` and gets back a ``DetectionResult``.

Multi-phase adaptive perception (``cfg.multiphase_adaptive=True``):

* Phase 1+1.5 (one combined call)  ->  Gemini sees the agentview RGB
  (and wrist if available), names the concepts the instruction grounds,
  and proposes K phrasing variants for each in a single JSON response.
* Phase 2  ->  SAM3 runs every variant; we keep the best mask per
  variant and surface ALL of them (one entry per variant) as candidate
  detections per concept.
* Phase 3  ->  We rasterise the candidate masks onto the RGB with
  colour-coded outlines and per-mask IDs, then ask Gemini to pick a
  ``mask_id`` per concept or return a single (x, y) click point if no
  candidate is correct.
* Phase 4  ->  For every concept where Gemini returned a click, refine
  via SAM3 point-prompt, falling back to the candidate mask whose
  centroid is closest to the click.
* Phase 5  ->  The chosen variant phrases are returned to the caller
  (``select_prompts``); ``detect_scene`` then runs SAM3 on those phrases
  with the existing depth-aware backprojection path.

``cfg.multiphase_adaptive`` defaults to ``False``; the ``tuned`` /
``gemini-prompts`` / ``adaptive`` flow runs otherwise.
"""
from __future__ import annotations

import json
import re
from dataclasses import dataclass, field
from typing import Any, Optional

import cv2
import numpy as np
import mujoco
from PIL import Image
import torch

from google.genai import types as gtypes
from robosuite.utils.camera_utils import get_real_depth_map

from spark_bench.libero_pro.planning import _get_sam3
from spark_bench.libero_pro.sam3_client import (
    Sam3ServiceClient,
    init_sam3_client,
)
from spark_bench.prompt_sweep import get_bddl_objects_of_interest
from spark_real.perception.spark_perception import ObjectDetection
from spark_real.planning.spark_planner import SPARKPlanner
from spark_real.utils.json_fence import strip_json_fence


# Optional SAM3 service client (cached per process, built on first use)

_sam3_service_client: Optional[Sam3ServiceClient] = None


def _service_client_if_enabled(cfg) -> Optional[Sam3ServiceClient]:
    """
    Return a SAM3 service client when ``cfg.use_sam3_service=True``.

    Cached per-process. Returns ``None`` (in-process path) when the config
    flag isn't set or the field is missing.
    """
    if not getattr(cfg, 'use_sam3_service', False):
        return None
    global _sam3_service_client
    if _sam3_service_client is None:
        url = getattr(cfg, 'sam3_service_url',
                       'http://127.0.0.1:8115')
        _sam3_service_client = init_sam3_client(service_url=url)
        if getattr(cfg, 'verbose', False):
            alive = _sam3_service_client.is_alive()
            print(f"[SAM3-service] using {url!r} (alive={alive})")
    return _sam3_service_client


__all__ = [
    'CameraParams',
    'DetectionResult',
    'select_prompts',
    'get_camera_params',
    'detect_scene',
    'redetect_agentview',
    'redetect_wrist',
    'disambiguate_pick_instance',
    'annotation_rescue',
]


# Phase 3 Pro model resolution (cached per process)

_PHASE3_FALLBACKS = ('gemini-3.5-flash', 'gemini-3-flash-preview', 'gemini-2.5-flash')
_resolved_phase3_model: Optional[str] = None


def _resolve_phase3_model(cfg, client) -> str:
    """
    Return the first available Pro model name, checking the SDK once.

    Tries ``cfg.phase3_model`` first, then falls back through the
    ``_PHASE3_FALLBACKS`` list.  Caches the result so subsequent Phase 3
    calls skip the ``models.list()`` round-trip.  On any SDK error returns
    ``cfg.phase3_model`` and lets the caller's try/except handle the failure.
    """
    global _resolved_phase3_model
    if _resolved_phase3_model is not None:
        return _resolved_phase3_model
    requested = getattr(cfg, 'phase3_model', None) or _PHASE3_FALLBACKS[0]
    candidates = [requested] + [m for m in _PHASE3_FALLBACKS if m != requested]
    try:
        available = {m.name.split('/', 1)[-1] for m in client.models.list()}
    except Exception:
        _resolved_phase3_model = requested
        return _resolved_phase3_model
    for name in candidates:
        if name in available:
            _resolved_phase3_model = name
            if getattr(cfg, 'verbose', False):
                print(f"[MP Phase 3] using Pro model {name!r}")
            return name
    _resolved_phase3_model = requested
    if getattr(cfg, 'verbose', False):
        print(f"[MP Phase 3] none of {candidates!r} listed; falling "
              f"back to {requested!r}")
    return requested


# Lightweight value types

@dataclass
class CameraParams:
    """
    Pose + intrinsic FOV for a MuJoCo camera.
    """
    pos: np.ndarray
    mat: np.ndarray
    fovy: float


@dataclass
class DetectionResult:
    rgb: np.ndarray
    depth: Optional[np.ndarray]
    wrist_rgb: Optional[np.ndarray]
    cam: CameraParams
    det_map: dict[str, Any] = field(default_factory=dict)
    dets: list = field(default_factory=list)


# Prompt selection: tuned / Gemini Phase 1 / adaptive self-consistency

def _load_tuned_prompts(env, cfg, prompts: list[str]) -> list[str]:
    try:
        with open(cfg.tuned_prompts_path) as f:
            tuned = json.load(f)
        best = tuned.get('best_per_object', {})
        objs: list[str] = []
        bddl = (getattr(env, 'bddl_file_name', None)
                or getattr(getattr(env, 'env', None), 'bddl_file_name', None))
        if bddl:
            objs = get_bddl_objects_of_interest(bddl)
        tp: list[str] = []
        for o in objs:
            info = best.get(o)
            if info and 'best_phrase' in info:
                tp.append(info['best_phrase'])
        if tp:
            if cfg.verbose:
                print(f"[Tuned Prompts] objs={objs} prompts={tp}")
            return tp[:10]
    except Exception as e:
        if cfg.verbose:
            print(f"[Tuned Prompts] failed: {e}; falling through")
    return prompts


def _gemini_phase1_prompts(rgb: np.ndarray,
                            wrist_rgb: Optional[np.ndarray],
                            instruction: str, cfg,
                            prompts: list[str]) -> list[str]:
    try:
        planner = SPARKPlanner(llm_backend='gemini')
        gp = planner.generate_prompts(
            instruction,
            scene_image=Image.fromarray(rgb),
            wrist_image=Image.fromarray(wrist_rgb) if wrist_rgb is not None else None,
        )
        if gp and isinstance(gp, list) and len(gp) >= 2:
            if cfg.verbose:
                print(f"[Gemini Phase 1] prompts={gp}")
            return gp[:10]
    except Exception as e:
        if cfg.verbose:
            print(f"[Gemini Phase 1] failed: {e}; using static prompts")
    return prompts


def _adaptive_variants(rgb: np.ndarray, instruction: str, cfg,
                        prompts: list[str]) -> list[str]:
    """
    Self-consistency: ask Gemini for K variants per concept, pick the
    variant whose SAM3 score (cardinality * confidence) is highest.
    """
    if not prompts:
        return prompts
    try:
        k = int(getattr(cfg, 'adaptive_k', 3))
        planner = SPARKPlanner(llm_backend='gemini')
        client = planner._get_client()
        variants_prompt = (
            f"You are helping an open-vocabulary object detector. For each "
            f"concept below (one per line), output {k} distinct short "
            f"detection phrases (1-4 words each). Vary colors, shapes, "
            f"materials. Task context: \"{instruction}\"\n\n"
            "Concepts:\n" + "\n".join(f"- {p}" for p in prompts[:10]) +
            "\n\nOutput JSON only, no code fences:\n"
            "{\"concept1\": [\"phrase1\", \"phrase2\", ...], ...}"
        )
        try:
            resp = client.models.generate_content(
                model='gemini-3-flash-preview',
                contents=variants_prompt,
                config=gtypes.GenerateContentConfig(temperature=0),
            )
            txt = resp.text.strip()
            if '```' in txt:
                txt = txt.split('```', 2)[1]
                if txt.startswith('json'):
                    txt = txt[4:]
            variants_by_concept = json.loads(txt.strip())
        except Exception as e:
            if cfg.verbose:
                print(f"[Adaptive] variant-gen failed: {e}; "
                      f"skipping self-consistency")
            return prompts

        if not variants_by_concept:
            return prompts
        chosen = _score_variants(rgb, variants_by_concept, k)
        chosen_phrases = _dedupe_chosen(chosen)
        if cfg.verbose:
            dropped = len(chosen) - len(chosen_phrases)
            print(f"[Adaptive] chose: {chosen_phrases} (deduped {dropped})")
        return chosen_phrases[:10] if chosen_phrases else prompts
    except Exception as e:
        if cfg.verbose:
            print(f"[Adaptive] failed: {e}; falling through to non-adaptive")
        return prompts


def _score_variants(rgb: np.ndarray, variants_by_concept: dict, k: int):
    """
    Return [(score, phrase, centroid)] one entry per concept.
    """
    preflight = _get_sam3()
    preflight.load_models(load_da3=False)
    state = preflight._sam3.set_image(Image.fromarray(rgb))

    chosen = []
    for concept, variants in variants_by_concept.items():
        if not isinstance(variants, list):
            continue
        best = (-1.0, concept, None)
        for v in variants[:k]:
            if not isinstance(v, str):
                continue
            st = preflight._sam3.set_text_prompt(prompt=v, state=state)
            m = st.get('masks', torch.tensor([]))
            s = st.get('scores', torch.tensor([]))
            if m.numel() == 0:
                score, centroid = 0.0, None
            else:
                n = m.shape[0]
                conf = float(s.max().item()) if s.numel() else 0.0
                # Prefer exactly 1 detection; penalise >2 (e.g. "knob" -> many).
                card_mult = 1.0 if n == 1 else (0.6 if n == 2 else 0.1)
                score = card_mult * conf
                best_idx = int(s.argmax().item())
                msk = m[best_idx].cpu().numpy().squeeze().astype(bool)
                if msk.any():
                    ys, xs = np.where(msk)
                    centroid = (float(xs.mean()), float(ys.mean()))
                else:
                    centroid = None
            if score > best[0]:
                best = (score, v, centroid)
        chosen.append(best)
    return chosen


def _dedupe_chosen(chosen: list, px_tol: float = 30.0) -> list[str]:
    """
    Drop duplicates whose mask centroids are within ``px_tol`` of each other.
    """
    deduped: list[tuple[float, str, Optional[tuple[float, float]]]] = []
    for s_i, v_i, c_i in chosen:
        if c_i is None:
            deduped.append((s_i, v_i, c_i))
            continue
        is_dup = False
        for j, (s_j, v_j, c_j) in enumerate(deduped):
            if c_j is not None and np.hypot(c_i[0] - c_j[0],
                                              c_i[1] - c_j[1]) < px_tol:
                is_dup = True
                if s_i > s_j:
                    deduped[j] = (s_i, v_i, c_i)
                break
        if not is_dup:
            deduped.append((s_i, v_i, c_i))
    return [v for _, v, _ in deduped]


# Multi-phase adaptive perception: Gemini sees SAM3 masks and refines

# Eight visually-distinct overlay colours (BGR for OpenCV).
_OVERLAY_COLORS_BGR = [
    (  0,   0, 255),  # red
    (  0, 255,   0),  # green
    (255,   0,   0),  # blue
    (  0, 255, 255),  # yellow
    (255,   0, 255),  # magenta
    (255, 255,   0),  # cyan
    (  0, 165, 255),  # orange
    (203, 192, 255),  # pink
]


def _gemini_combined_phase1(rgb: np.ndarray,
                              wrist_rgb: Optional[np.ndarray],
                              instruction: str, cfg,
                              prompts: list[str]) -> dict[str, list[str]]:
    """
    Combined Phase 1 + 1.5: one Gemini call returns K variants per concept.

    The model is given the agentview RGB (and wrist when available) plus
    the task instruction.  It is asked to (a) name every distinct object
    concept the instruction needs grounded, (b) propose ``k`` short
    detection phrases per concept that vary along colour / shape /
    material.  Returns ``{concept: [variant, ...]}``.  On any failure
    (Gemini error, JSON parse error, empty output) falls back to the
    static ``prompts`` list mapped to a single-variant dict.
    """
    fallback = {p: [p] for p in prompts if isinstance(p, str)} or {
        'object': ['object']}
    try:
        # K=6 variants so LIBERO-PRO object-perturbation variants
        # ("bigger_alphabet_soup", "red_*", "*_with_*") have more chances
        # of matching an open-vocab detector phrasing; cap at 8 to keep
        # prompt length and the Phase 3 overlay sane.
        _k_raw = int(getattr(cfg, 'adaptive_k', 6))
        # adaptive_k == 3 (the runner default) means "auto" -> 6; any other
        # explicit value passes through, then clamped to [1, 8].
        k = 6 if _k_raw == 3 else _k_raw
        k = max(1, min(k, 8))
        planner = SPARKPlanner(llm_backend='gemini')
        client = planner._get_client()
        ask = (
            "You are helping ground a robotic manipulation instruction. "
            "Look at the scene image(s) and the task instruction. "
            f"Task: \"{instruction}\"\n\n"
            "Step 1: name every distinct object concept the instruction "
            "requires grounding (e.g. \"red mug\", \"cabinet handle\"). "
            "Use short canonical phrases (1-4 words). 2-5 concepts is "
            f"typical.\nStep 2: for each concept, propose exactly {k} "
            "short open-vocabulary detection phrases (1-4 words each) "
            "that an open-vocabulary detector might use. The scene may "
            "contain a PERTURBED visual variant of the canonical object "
            "(bigger/smaller, recoloured, retextured), so MIX three kinds "
            f"of phrasings across the {k} variants:\n"
            " (a) the canonical name + 1-2 synonyms (e.g. \"alphabet "
            "soup\", \"soup can\", \"Campbell soup can\");\n"
            " (b) SIZE-modified phrasings -- pick from {\"big\", "
            "\"larger\", \"smaller\"} prefixed onto the canonical (e.g. "
            "\"big soup can\", \"larger alphabet soup\");\n"
            " (c) COLOUR/TEXTURE-modified phrasings that match what you "
            "actually see in the image -- pick from {\"red\", \"green\", "
            "\"yellow\", \"white\", \"black\", \"matte\", \"shiny\", "
            "\"darker\", \"lighter\"} prefixed onto the canonical (e.g. "
            "\"red soup can\", \"darker can\").\n"
            "Avoid plurals and duplicates.\n\n"
            "HARD RULES FOR CONCEPT NAMING:\n"
            "1. Each concept name MUST be derivable from the instruction "
            "itself (a noun, a synonym, or a near-paraphrase). Pull the "
            "most specific noun from the instruction verbatim when "
            "possible.\n"
            "2. If the instruction uses generic terms (\"red can\", "
            "\"small bottle\"), look at the scene image and choose the "
            "concept name that VISUALLY matches the generic description "
            "-- do NOT guess based on category similarity with other "
            "scene objects.\n"
            "3. NEVER substitute a concept name that doesn't appear (as "
            "noun or synonym) in the instruction. Example: if the "
            "instruction says \"tomato sauce\", concept must be "
            "\"tomato_sauce\" or \"tomato sauce can\" -- NOT \"ketchup\", "
            "NOT \"salad dressing\", even if those are visually similar. "
            "Size/colour/texture modifiers (rule 2 of Step 2) are "
            "ALLOWED on detection phrases even if not in the instruction "
            "-- they cover the perturbation distribution.\n"
            "4. If unsure between two candidates, emit BOTH as separate "
            "concepts so SAM3 can detect them and a later phase can "
            "pick.\n"
            "5. COMPOUND INSTRUCTIONS: if the instruction contains "
            "\"and\", \"both\", \"two\", \"three\", \"each\", or names "
            "TWO OR MORE distinct objects, you MUST enumerate every "
            "object as its own concept. \"put both X and Y in Z\" -> "
            "THREE concepts (X, Y, Z), never collapse X+Y into one. "
            "\"put X on A and Y on B\" -> FOUR concepts (X, A, Y, B). "
            "\"turn on X and put Y on it\" -> TWO concepts (X, Y). "
            "Missing any compound object guarantees the trial fails -- "
            "err on the side of MORE concepts.\n\n"
            "Step 3 (point hint, OPTIONAL but helps disambiguate "
            "look-alike instances): for each concept, also return a "
            "pixel coordinate `[x, y]` on the agentview image (0..W, "
            "0..H) where the target object is CENTERED. Image is "
            f"{rgb.shape[1]} wide x {rgb.shape[0]} tall, origin "
            "top-left. If you cannot tell visually, omit hint_point.\n\n"
            "Respond with JSON ONLY (no prose, no code fences). Use "
            "this schema (hint_point is optional per concept):\n"
            "{\n"
            "  \"concept_name_1\": {\n"
            "    \"variants\": [\"variant1\", \"variant2\", ...],\n"
            "    \"hint_point\": [x, y]\n"
            "  },\n"
            "  \"concept_name_2\": {...}\n"
            "}\n"
            "OR (legacy, also accepted):\n"
            "{\"concept_name_1\": [\"variant1\", ...], ...}")
        contents: list[Any] = [ask, Image.fromarray(rgb)]
        if wrist_rgb is not None:
            contents.append(Image.fromarray(wrist_rgb))
        resp = client.models.generate_content(
            model=planner.PHASE1_MODEL,
            contents=contents,
            config=gtypes.GenerateContentConfig(temperature=0),
        )
        txt = strip_json_fence(resp.text)
        data = json.loads(txt)
        cleaned: dict[str, list[str]] = {}
        hints: dict[str, tuple[int, int]] = {}
        H, W = rgb.shape[:2]
        for concept, val in data.items():
            if not isinstance(concept, str):
                continue
            # Accept both new schema {variants, hint_point} and legacy list.
            if isinstance(val, dict):
                variants = val.get('variants', [])
                hp = val.get('hint_point')
                if (isinstance(hp, (list, tuple)) and len(hp) == 2
                        and all(isinstance(v, (int, float)) for v in hp)):
                    x = int(np.clip(hp[0], 0, W - 1))
                    y = int(np.clip(hp[1], 0, H - 1))
                    hints[concept.strip()] = (x, y)
            elif isinstance(val, list):
                variants = val
            else:
                continue
            vs = [v.strip() for v in variants if isinstance(v, str)
                  and v.strip()]
            if vs:
                cleaned[concept.strip()] = vs[:max(1, k)]
        if not cleaned:
            return fallback
        # Stash hint points on cfg so _sam3_candidates_per_concept can pick
        # them up without changing the return-type contract.
        setattr(cfg, '_mp_phase1_hint_points', hints)
        if cfg.verbose:
            preview = {c: vs for c, vs in list(cleaned.items())[:4]}
            print(f"[MP Phase 1+1.5] concepts+variants={preview}")
            if hints:
                print(f"[MP Phase 1+1.5] hint_points={hints}")
        return cleaned
    except Exception as e:
        if cfg.verbose:
            print(f"[MP Phase 1+1.5] failed: {e}; falling back to "
                  f"static prompts")
        return fallback


def _mask_to_cand(phrase: str, mask: np.ndarray, score: float) -> dict:
    """
    Build a candidate dict (phrase, mask, score, centroid, bbox) from a mask.
    """
    ys, xs = np.where(mask)
    return {
        'phrase': phrase,
        'mask': mask,
        'score': score,
        'centroid': (float(xs.mean()), float(ys.mean())),
        'bbox': (int(xs.min()), int(ys.min()), int(xs.max()), int(ys.max())),
    }


def _sam3_candidates_per_concept(rgb: np.ndarray,
                                   variants_by_concept: dict[str, list[str]],
                                   service: Optional[Sam3ServiceClient] = None,
                                   hint_points: Optional[dict[str, tuple[int, int]]] = None,
                                   ) -> dict[str, list[dict]]:
    """
    Phase 2: run SAM3 on every variant; return best mask per variant.

    Returns ``{concept: [{phrase, mask, score, centroid, bbox}, ...]}``
    sorted by descending score, one entry per variant that produced any
    mask above SAM3's internal threshold. When ``service`` is provided,
    talks to the SAM3 FastAPI service instead of loading SAM3 in-process.
    When ``hint_points`` includes a (x, y) for a concept, ALSO calls the
    service's point-prompt endpoint and adds the resulting mask as an
    extra candidate alongside text variants. This lets Phase 1 disambiguate
    lookalike instances (e.g. ketchup vs tomato sauce) before SAM3 voting.
    """
    hint_points = hint_points or {}
    if service is not None:
        out_svc: dict[str, list[dict]] = {}
        for concept, variants in variants_by_concept.items():
            cands: list[dict] = []
            for v in variants:
                try:
                    results = service.segment_text(rgb, v)
                except Exception:
                    continue
                if not results:
                    continue
                # service returns list sorted by score descending; keep best.
                best = results[0]
                msk = best.get('mask')
                if msk is None or not msk.any():
                    continue
                score = float(best.get('score', 0.0))
                cands.append(_mask_to_cand(v, msk, score))
            # Phase 1 point-prompt: ask SAM3 to segment from the (x, y)
            # Gemini provided. Adds an additional "<phase1_click>" candidate.
            hp = hint_points.get(concept)
            if hp is not None:
                try:
                    pr = service.segment_point(rgb, hp)
                    msk_p = pr.get('mask') if pr else None
                    if msk_p is not None and msk_p.any():
                        score_p = float(pr.get('score', 0.0))
                        cands.append(_mask_to_cand(
                            f"<phase1_click@{hp[0]},{hp[1]}>", msk_p, score_p))
                except Exception:
                    pass
            cands.sort(key=lambda c: -c['score'])
            if cands:
                out_svc[concept] = cands
        return out_svc

    preflight = _get_sam3()
    preflight.load_models(load_da3=False)
    state = preflight._sam3.set_image(Image.fromarray(rgb))

    out: dict[str, list[dict]] = {}
    for concept, variants in variants_by_concept.items():
        cands: list[dict] = []
        for v in variants:
            try:
                st = preflight._sam3.set_text_prompt(prompt=v, state=state)
            except Exception:
                continue
            m = st.get('masks', torch.tensor([]))
            s = st.get('scores', torch.tensor([]))
            if m.numel() == 0 or s.numel() == 0:
                continue
            best_idx = int(s.argmax().item())
            score = float(s[best_idx].item())
            msk = m[best_idx].cpu().numpy().squeeze().astype(bool)
            if not msk.any():
                continue
            cands.append(_mask_to_cand(v, msk, score))
        cands.sort(key=lambda c: -c['score'])
        if cands:
            out[concept] = cands
    return out


def _draw_candidate_overlay(rgb: np.ndarray,
                              cand_by_concept: dict[str, list[dict]]
                              ) -> tuple[np.ndarray, dict[str, list[str]]]:
    """
    Rasterise per-candidate mask outlines + ID labels onto ``rgb``.

    Returns ``(annotated_rgb, id_to_concept_map)`` where
    ``id_to_concept_map`` maps ``concept -> [mask_id_0, mask_id_1, ...]``
    in the same order Gemini sees in the legend.
    """
    canvas = rgb.copy()
    # OpenCV draws in BGR; convert then back.
    canvas_bgr = cv2.cvtColor(canvas, cv2.COLOR_RGB2BGR)
    legend_rows: list[str] = []
    id_map: dict[str, list[str]] = {}
    color_idx = 0
    mask_counter = 0
    for concept, cands in cand_by_concept.items():
        ids_for_concept: list[str] = []
        for c in cands:
            mask_id = f"m{mask_counter}"
            mask_counter += 1
            ids_for_concept.append(mask_id)
            color = _OVERLAY_COLORS_BGR[color_idx % len(_OVERLAY_COLORS_BGR)]
            color_idx += 1
            msk = c['mask'].astype(np.uint8) * 255
            contours, _ = cv2.findContours(
                msk, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            cv2.drawContours(canvas_bgr, contours, -1, color, 2)
            cx, cy = int(c['centroid'][0]), int(c['centroid'][1])
            cv2.circle(canvas_bgr, (cx, cy), 4, color, -1)
            cv2.putText(canvas_bgr, mask_id, (cx + 6, cy - 6),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2,
                        cv2.LINE_AA)
            legend_rows.append(
                f"{mask_id}: concept=\"{concept}\" phrase=\"{c['phrase']}\" "
                f"conf={c['score']:.2f} centroid=({cx},{cy})")
        id_map[concept] = ids_for_concept
    annotated = cv2.cvtColor(canvas_bgr, cv2.COLOR_BGR2RGB)
    return annotated, id_map


def _gemini_phase3_refine(rgb: np.ndarray,
                            cand_by_concept: dict[str, list[dict]],
                            instruction: str, cfg
                            ) -> dict[str, dict]:
    """
    Phase 3: Gemini sees overlaid masks, picks or clicks per concept.

    Returns ``{concept: {'choice': mask_id} | {'click': (x, y)} | {}}``.
    Empty dict means undecided; the caller keeps the best-by-score
    fallback.  On Gemini failure every concept gets an empty dict.
    """
    if not cand_by_concept:
        return {}
    try:
        annotated_rgb, id_map = _draw_candidate_overlay(rgb, cand_by_concept)
        # Build a compact legend the model can read alongside the image.
        legend_lines: list[str] = []
        for concept, cands in cand_by_concept.items():
            ids = id_map[concept]
            for mid, c in zip(ids, cands):
                cx, cy = int(c['centroid'][0]), int(c['centroid'][1])
                legend_lines.append(
                    f"  {mid}: concept=\"{concept}\" "
                    f"phrase=\"{c['phrase']}\" conf={c['score']:.2f} "
                    f"centroid=({cx},{cy})")
        legend = "\n".join(legend_lines)
        h, w = rgb.shape[:2]
        concepts = list(cand_by_concept.keys())
        ask = (
            "You are checking SAM3 candidate detections for a robotic "
            "grounding task.  The image attached shows the scene with "
            "SAM3 mask outlines colour-coded and labelled (m0, m1, ...). "
            f"Image size: {w} wide x {h} tall (top-left origin).\n\n"
            f"ORIGINAL TASK INSTRUCTION: \"{instruction}\"\n\n"
            "The concept list below was derived from this instruction by "
            "an earlier perception step, BUT it may contain mistakes "
            "(e.g., a similar-looking object such as ketchup vs. tomato "
            "sauce, both red cans). Your job: for each concept, pick the "
            "mask_id that ACTUALLY matches what the instruction asks for. "
            "If none of the masks match the instruction's target object, "
            "return click coordinates instead.\n\n"
            "HARD RULE: ground your mask choice in the ORIGINAL "
            "INSTRUCTION above, not the concept name. If the instruction "
            "says \"tomato sauce\" but the concept list contains "
            "\"ketchup\", and you see both a tomato_sauce-looking mask "
            "and a ketchup-looking mask in the scene, choose the "
            "tomato_sauce mask (or click on it if it isn't in the "
            "candidate list). The TARGET OBJECT in the instruction (the "
            "noun being picked, moved, or manipulated) is authoritative; "
            "the concept name is only a hint and may be wrong.\n\n"
            f"Concepts to ground: {concepts}\n\n"
            f"Candidate legend:\n{legend}\n\n"
            "For EACH concept, output one of:\n"
            "  (a) {\"choice\": \"<mask_id>\"} when one of the listed "
            "candidates correctly matches the instruction's target.\n"
            "  (b) {\"click\": [x, y]} (pixel coords, ints) when NONE of "
            "the candidates correctly matches the instruction's target "
            "and a single click on the true object would help "
            "redetection.\n\n"
            "Respond with JSON ONLY (no prose, no code fences):\n"
            "{\"<concept_name>\": {\"choice\": \"m2\"}, "
            "\"<other_concept>\": {\"click\": [314, 207]}, ...}")
        planner = SPARKPlanner(llm_backend='gemini')
        client = planner._get_client()
        phase3_model = _resolve_phase3_model(cfg, client)
        resp = client.models.generate_content(
            model=phase3_model,
            contents=[ask, Image.fromarray(annotated_rgb)],
            config=gtypes.GenerateContentConfig(temperature=0),
        )
        txt = strip_json_fence(resp.text)
        data = json.loads(txt)
        decisions: dict[str, dict] = {}
        for concept in cand_by_concept:
            d = data.get(concept) if isinstance(data, dict) else None
            if not isinstance(d, dict):
                decisions[concept] = {}
                continue
            if 'choice' in d and isinstance(d['choice'], str):
                decisions[concept] = {'choice': d['choice'].strip()}
            elif 'click' in d and isinstance(d['click'], (list, tuple)) \
                    and len(d['click']) == 2:
                try:
                    x = int(d['click'][0]); y = int(d['click'][1])
                    x = max(0, min(w - 1, x)); y = max(0, min(h - 1, y))
                    decisions[concept] = {'click': (x, y)}
                except (TypeError, ValueError):
                    decisions[concept] = {}
            else:
                decisions[concept] = {}
        if cfg.verbose:
            print(f"[MP Phase 3] decisions={decisions}")
        return decisions
    except Exception as e:
        if cfg.verbose:
            print(f"[MP Phase 3] failed: {e}; keeping best-score "
                  f"candidates")
        return {}


def _refine_from_click(sam3_state: dict, rgb: np.ndarray,
                        click: tuple[int, int], concept_label: str,
                        cands: Optional[list[dict]] = None,
                        service: Optional[Sam3ServiceClient] = None,
                        ) -> Optional[dict]:
    """
    Phase 4: ask SAM3 for a mask at the clicked pixel via point-prompt.

    Calls ``Sam3Processor.set_point_prompt`` (which wraps the underlying
    ``SAM3InteractiveImagePredictor.predict``) using the click as a
    foreground point.  Returns a candidate dict in the same shape as the
    text-prompt candidates so the downstream pipeline can treat it
    uniformly.  Falls back to the closest-centroid candidate from
    ``cands`` if the point-prompt call fails.

    Args:
      sam3_state: state dict returned by ``Sam3Processor.set_image``
        (cached backbone features).
      rgb: the original image (only used for bounds-checking the click).
      click: ``(x, y)`` pixel coordinates of Gemini's click.
      concept_label: concept name Gemini was grounding; used to build
        the synthetic phrase tag.
      cands: optional list of existing text-prompt candidates used as a
        degenerate fallback if the point-prompt path raises.

    Returns:
      A candidate dict ``{'phrase', 'mask', 'score', 'centroid',
      'bbox'}`` or ``None`` if neither path produces a mask.
    """
    h, w = rgb.shape[:2]
    cx_click = int(max(0, min(w - 1, click[0])))
    cy_click = int(max(0, min(h - 1, click[1])))

    # Service path: skips the in-process Sam3Processor.set_point_prompt
    # call and asks the FastAPI service for the same single best-IoU mask.
    if service is not None:
        try:
            svc_res = service.segment_point(rgb, (cx_click, cy_click))
        except Exception:
            svc_res = None
        if svc_res is not None and svc_res.get('mask') is not None:
            msk = svc_res['mask']
            if msk.any():
                ys, xs = np.where(msk)
                centroid = (float(xs.mean()), float(ys.mean()))
                bbox = (int(xs.min()), int(ys.min()),
                        int(xs.max()), int(ys.max()))
                return {
                    'phrase': f"<click@{cx_click},{cy_click}>",
                    'mask': msk,
                    'score': float(svc_res.get('score', 0.0)),
                    'centroid': centroid,
                    'bbox': bbox,
                }
        # Service unreachable / empty result -- fall through to centroid snap.
        if cands:
            best = min(cands, key=lambda c: (c['centroid'][0] - cx_click) ** 2
                                             + (c['centroid'][1] - cy_click) ** 2)
            return best
        return None

    preflight = _get_sam3()
    try:
        # set_point_prompt mutates state in-place - pass it through.
        preflight._sam3.set_point_prompt(
            state=sam3_state, point_xy=(cx_click, cy_click), label=1,
            multimask_output=True)
        m = sam3_state.get('masks', torch.tensor([]))
        s = sam3_state.get('scores', torch.tensor([]))
        if m.numel() > 0:
            msk = m[0].cpu().numpy().squeeze().astype(bool)
            if msk.any():
                ys, xs = np.where(msk)
                centroid = (float(xs.mean()), float(ys.mean()))
                bbox = (int(xs.min()), int(ys.min()),
                        int(xs.max()), int(ys.max()))
                score = float(s[0].item()) if s.numel() else 0.0
                return {
                    'phrase': f"<click@{cx_click},{cy_click}>",
                    'mask': msk,
                    'score': score,
                    'centroid': centroid,
                    'bbox': bbox,
                }
    except Exception:
        # Fall through to centroid-snap fallback below.
        pass

    if cands:
        best = min(cands, key=lambda c: (c['centroid'][0] - cx_click) ** 2
                                         + (c['centroid'][1] - cy_click) ** 2)
        return best
    return None


def _multiphase_adaptive(rgb: np.ndarray, wrist_rgb: Optional[np.ndarray],
                          instruction: str, cfg, prompts: list[str]
                          ) -> list[str]:
    """
    Top-level orchestration for ``--multiphase-adaptive``.

    Returns the final list of SAM3 phrases to use for the depth-aware
    ``detect_scene`` call.  Falls through to the input ``prompts`` on any
    upstream failure so the runner never crashes from this code path.
    """
    # Clear any prior trial's rename map so a fallthrough doesn't carry
    # stale variant->concept mappings into _build_det_map.
    setattr(cfg, '_mp_phrase_to_concept', {})
    variants_by_concept = _gemini_combined_phase1(
        rgb, wrist_rgb, instruction, cfg, prompts)
    if not variants_by_concept:
        return prompts

    # Drawer-level enforcement: if the instruction names a drawer level
    # (top/middle/bottom), emit all three level handles as candidate
    # concepts so Phase 3 can disambiguate by Y-overlay and _pick_level
    # works downstream.
    if re.search(r'\b(top|middle|bottom)\s+drawer\b', instruction.lower()):
        for lvl in ('top', 'middle', 'bottom'):
            concept = f"{lvl} drawer handle"
            if concept not in variants_by_concept:
                variants_by_concept[concept] = [concept,
                                                 f"{lvl} cabinet drawer handle",
                                                 f"{lvl} drawer pull"]
        if cfg.verbose:
            print(f"[MP] drawer-level enforce: ensured all 3 level "
                  f"handles in concepts={list(variants_by_concept.keys())}")

    # Spatial-anchor enforcement: Phase 1 catches direct objects and
    # compound conjuncts but drops relational anchors ("between the plate
    # and the ramekin", "next to the cookie box", "on the stove"), which
    # the BT executor needs as masks to resolve the spatial relation.
    _instr_lower = instruction.lower()
    # Noun phrases: 1-3 words, but stop on conjunctions / prepositions /
    # punctuation so "plate and the ramekin" is captured as just "plate".
    _stop_words = r'and|or|then|to|on|in|of|at|for|with|by|but|while|near'
    _noun_chunk = (r'((?:\w+(?:\s+(?!(?:' + _stop_words + r')\b)\w+){0,2}))')
    _binary_relations = r'\bbetween\s+(?:the\s+)?' + _noun_chunk + \
                         r'\s+and\s+(?:the\s+)?' + _noun_chunk
    _unary_relations = (r'\b(?:next\s+to|beside|on(?:\s+top\s+of)?'
                         r'|in(?:side)?|from|under|behind|inside)\s+'
                         r'(?:the\s+)?' + _noun_chunk)
    _added: list[str] = []
    candidates: list[str] = []
    for m in re.finditer(_binary_relations, _instr_lower):
        candidates.extend([m.group(1).strip(), m.group(2).strip()])
    for m in re.finditer(_unary_relations, _instr_lower):
        candidates.append(m.group(1).strip())
    for anchor in candidates:
        if not anchor:
            continue
        # Skip pronouns / generics.
        if anchor in ('it', 'them', 'one', 'thing', 'side', 'top', 'bottom',
                      'middle', 'front', 'back', 'left', 'right'):
            continue
        # Already present (concept name or any variant)?
        if any(anchor in concept.lower() or
               any(anchor in v.lower() for v in variants)
               for concept, variants in variants_by_concept.items()):
            continue
        variants_by_concept[anchor] = [anchor, f"the {anchor}"]
        _added.append(anchor)
    if _added and cfg.verbose:
        print(f"[MP] spatial-anchor enforce: added concepts={_added} "
              f"from relational phrases in instruction")

    service = _service_client_if_enabled(cfg)
    hint_points = getattr(cfg, '_mp_phase1_hint_points', None) or {}
    cand_by_concept = _sam3_candidates_per_concept(
        rgb, variants_by_concept, service=service,
        hint_points=hint_points)
    if not cand_by_concept:
        if cfg.verbose:
            print("[MP] SAM3 returned no candidates for any variant; "
                  "falling back to flat variant list")
        flat = []
        for variants in variants_by_concept.values():
            flat.extend(variants[:1])
        return (flat or prompts)[:10]

    decisions = _gemini_phase3_refine(rgb, cand_by_concept, instruction, cfg)

    chosen_phrases: list[str] = []
    chosen_centroids: list[tuple[float, float]] = []
    # Track phrase -> concept rename so det_map keys collapse back to
    # canonical concept names; otherwise Phase 2 BT-gen emits the variant
    # phrase ("ceramic dish") as keypoint_label and it can match a
    # non-target mask under perturbation.
    phrase_to_concept: dict[str, str] = {}
    # Flatten cand_by_concept -> mask_id -> candidate for choice resolution.
    annotated_rgb, id_map = _draw_candidate_overlay(rgb, cand_by_concept)
    flat_lookup: dict[str, dict] = {}
    for concept, ids in id_map.items():
        for mid, c in zip(ids, cand_by_concept[concept]):
            flat_lookup[mid] = c

    # Fresh SAM3 image state on demand for click decisions: set_text_prompt
    # mutates the Phase 2 state across variants, so it is not reused.
    click_state: Optional[dict] = None

    def _ensure_click_state() -> Optional[dict]:
        nonlocal click_state
        if click_state is not None:
            return click_state
        # Service path doesn't need an in-process backbone cache - point-prompt
        # calls into the FastAPI service re-encode the image once per call.
        if service is not None:
            click_state = {}
            return click_state
        try:
            preflight = _get_sam3()
            preflight.load_models(load_da3=False)
            click_state = preflight._sam3.set_image(Image.fromarray(rgb))
        except Exception as e:
            if cfg.verbose:
                print(f"[MP Phase 4] set_image failed: {e}; using "
                      f"centroid-snap fallback")
            click_state = None
        return click_state

    for concept, cands in cand_by_concept.items():
        decision = decisions.get(concept, {}) if decisions else {}
        chosen: Optional[dict] = None
        if 'choice' in decision:
            chosen = flat_lookup.get(decision['choice'])
        elif 'click' in decision:
            st = _ensure_click_state()
            if st is not None:
                chosen = _refine_from_click(
                    st, rgb, decision['click'], concept, cands=cands,
                    service=service)
            else:
                # No state - fall back to nearest-centroid.
                chosen = _refine_from_click(
                    {}, rgb, decision['click'], concept, cands=cands,
                    service=service)
            if cfg.verbose and chosen is not None:
                ph = chosen.get('phrase', '?')
                print(f"[MP Phase 4] concept={concept!r} click="
                      f"{decision['click']} -> phrase={ph!r} "
                      f"score={chosen.get('score', 0.0):.2f}")
        if chosen is None:
            chosen = cands[0]  # best-by-score fallback
        # Dedupe by centroid distance (as in _dedupe_chosen).
        ccx, ccy = chosen['centroid']
        is_dup = any(
            (ccx - px) ** 2 + (ccy - py) ** 2 < 30.0 ** 2
            for px, py in chosen_centroids)
        if is_dup:
            continue
        chosen_phrases.append(chosen['phrase'])
        chosen_centroids.append(chosen['centroid'])
        phrase_to_concept[chosen['phrase']] = concept

    # Stash rename map on cfg so detect_scene's _build_det_map can collapse
    # variant phrases back to canonical concept names downstream.
    setattr(cfg, '_mp_phrase_to_concept', phrase_to_concept)
    if cfg.verbose:
        print(f"[MP] final phrases={chosen_phrases}")
        print(f"[MP] rename map={phrase_to_concept}")
    return chosen_phrases[:10] if chosen_phrases else prompts


def select_prompts(env, rgb: np.ndarray, wrist_rgb: Optional[np.ndarray],
                    instruction: str, cfg, prompts: list[str]) -> list[str]:
    """
    Apply tuned/Gemini/adaptive prompt-selection layers in fairness order.
    """
    if getattr(cfg, 'multiphase_adaptive', False):
        return _multiphase_adaptive(rgb, wrist_rgb, instruction, cfg, prompts)

    if getattr(cfg, 'use_tuned_prompts', False):
        prompts = _load_tuned_prompts(env, cfg, prompts)

    if (getattr(cfg, 'use_gemini_prompts', False)
            and not getattr(cfg, 'use_tuned_prompts', False)):
        prompts = _gemini_phase1_prompts(rgb, wrist_rgb, instruction, cfg,
                                          prompts)

    if getattr(cfg, 'adaptive_prompts', False):
        prompts = _adaptive_variants(rgb, instruction, cfg, prompts)

    return prompts


# Camera params + render utilities

def get_camera_params(env, cam_name: str = 'agentview') -> CameraParams:
    """
    Read pose + FOV for a named MuJoCo camera.
    """
    model = env.sim.model._model
    data = env.sim.data._data
    cam_id = env.sim.model.camera_name2id(cam_name)
    return CameraParams(
        pos=data.cam_xpos[cam_id].copy(),
        mat=data.cam_xmat[cam_id].reshape(3, 3).copy(),
        fovy=float(model.cam_fovy[cam_id]),
    )


def _flip_render(arr: np.ndarray) -> np.ndarray:
    """
    Flip vertically - robosuite renders are upside down vs MuJoCo cam frame.
    """
    return arr[::-1].copy()


def _render_depth(env, raw_depth: Optional[np.ndarray]) -> Optional[np.ndarray]:
    if raw_depth is None:
        return None
    depth = get_real_depth_map(env.sim, _flip_render(raw_depth))
    if depth.ndim == 3:
        depth = depth.squeeze(-1)
    return depth


# Detection (single + dual-camera merge)

def _detect_via_service(service: Sam3ServiceClient,
                          rgb: np.ndarray, depth: np.ndarray,
                          prompts: list[str],
                          cam_pos: np.ndarray, cam_mat: np.ndarray,
                          cam_fovy: float, w: int, h: int,
                          multi_instance_prompts: Optional[set[str]] = None,
                          sam3_threshold: float = 0.03):
    """
    Service equivalent of Sam3Processor._detect_with_rendered_depth.

    The FastAPI service supplies SAM3 masks per prompt; the MuJoCo depth
    backprojection runs locally in numpy. With this path the worker never
    loads SAM3 in-process (Sam3Processor.load_models is the only path that
    pins ~3-4 GB of GPU memory in the runner).
    """
    multi_instance_prompts = multi_instance_prompts or set()
    f = h / (2 * np.tan(np.deg2rad(cam_fovy) / 2))
    detections = []

    for prompt in prompts:
        try:
            results = service.segment_text(rgb, prompt)
        except Exception:
            continue
        if not results:
            continue

        if prompt in multi_instance_prompts:
            picked = [r for r in results
                       if float(r.get('score', 0.0)) > sam3_threshold]
            if not picked:
                picked = [results[0]]
        else:
            picked = [results[0]]

        for r in picked:
            score = float(r.get('score', 0.0))
            mask = r.get('mask')
            if mask is None:
                continue
            ys, xs = np.where(mask > 0)
            if len(xs) == 0:
                continue

            cx, cy = float(xs.mean()), float(ys.mean())

            mask_depths = depth[mask > 0]
            valid = (mask_depths > 0.01) & (mask_depths < 10)
            if valid.sum() < 3:
                ui, vi = int(cx), int(cy)
                if 0 <= vi < h and 0 <= ui < w and depth[vi, ui] > 0.01:
                    d_val = float(depth[vi, ui])
                    x_cam = (cx - w / 2) * d_val / f
                    y_cam = -(cy - h / 2) * d_val / f
                    z_cam = -d_val
                    point_world = cam_mat @ np.array(
                        [x_cam, y_cam, z_cam]) + cam_pos
                else:
                    continue
            else:
                vys = ys[valid]
                vxs = xs[valid]
                vds = mask_depths[valid]
                x_cams = (vxs.astype(np.float64) - w / 2) * vds / f
                y_cams = -(vys.astype(np.float64) - h / 2) * vds / f
                z_cams = -vds
                pts_cam = np.stack([x_cams, y_cams, z_cams], axis=1)
                pts_world = (cam_mat @ pts_cam.T).T + cam_pos
                point_world = np.median(pts_world, axis=0)
                d_val = float(np.percentile(vds, 25))

            det = ObjectDetection(
                label=prompt,
                confidence=score,
                centroid_2d=(cx, cy),
                mask_area=len(xs),
                depth_meters=d_val,
                position_3d=point_world,
            )
            det.mask = mask
            detections.append(det)

    return detections


def _detect_one_cam(sam3, rgb: np.ndarray, depth: Optional[np.ndarray],
                     prompts: list[str], cam: CameraParams,
                     cam_w: int, cam_h: int,
                     multi_instance: set[str],
                     service: Optional[Sam3ServiceClient] = None,
                     sam3_threshold: float = 0.03):
    """
    One-camera SAM3 detection with depth-aware backproject when available.

    When ``service`` is provided, routes segmentation through the FastAPI
    service and does the backprojection locally - the worker never loads
    SAM3 in-process. Falls back to ``sam3._detect_with_rendered_depth`` /
    ``sam3.detect`` when no service is supplied.
    """
    if depth is not None:
        if service is not None:
            return _detect_via_service(
                service, rgb, depth, prompts, cam.pos, cam.mat, cam.fovy,
                cam_w, cam_h, multi_instance_prompts=multi_instance,
                sam3_threshold=sam3_threshold)
        return sam3._detect_with_rendered_depth(
            rgb, depth, prompts, cam.pos, cam.mat, cam.fovy,
            cam_w, cam_h, multi_instance_prompts=multi_instance)
    return sam3.detect(rgb, prompts, cam.pos, cam.mat, cam.fovy)


def _merge_wrist_dets(dets: list, dets_w: list, verbose: bool = False) -> list:
    """
    Use wrist position when its confidence is higher; keep agentview mask.

    SE(3) grasp backprojects with agentview depth/cam params, so a
    wrist-camera mask paired with agentview cam params would produce a
    wrong point cloud.  Wrist-only detections have their mask cleared.

    Every detection that survives keeps ``position_agentview`` - the
    agentview-only backprojected centroid (or None for wrist-only
    detections).  Scene-diff / sticky comparisons against later
    agentview-only re-detections must use THIS position, not the fused
    one: the two cameras disagree by a systematic 1-2 cm on some objects
    (1.7 cm in x on one object), which otherwise reads as a phantom
    'moved' verdict on unperturbed trials.
    """
    wrist_map = {d.label: d for d in dets_w if d.position_3d is not None}
    agent_map = {d.label: d for d in dets if d.position_3d is not None}
    for d in dets:
        if d.position_3d is not None:
            d.position_agentview = d.position_3d.copy()
    for label, wd in wrist_map.items():
        if label in agent_map:
            ad = agent_map[label]
            if verbose and ad.position_3d is not None:
                delta = float(np.linalg.norm(
                    np.asarray(wd.position_3d) - np.asarray(ad.position_3d)))
                print(f"[WristFuse] {label!r}: agent conf={ad.confidence:.3f}"
                      f" pos=({ad.position_3d[0]:.4f},{ad.position_3d[1]:.4f},"
                      f"{ad.position_3d[2]:.4f}) wrist conf="
                      f"{wd.confidence:.3f} pos=({wd.position_3d[0]:.4f},"
                      f"{wd.position_3d[1]:.4f},{wd.position_3d[2]:.4f}) "
                      f"delta={delta*100:.1f}cm "
                      f"winner={'wrist' if wd.confidence > ad.confidence else 'agent'}")
            if wd.confidence > ad.confidence:
                for i, d in enumerate(dets):
                    if d.label == label:
                        dets[i].position_3d = wd.position_3d
                        dets[i].confidence = wd.confidence
                        break
        else:
            wd.mask = None
            wd.position_agentview = None
            dets.append(wd)
    return dets


# Landmarks (fixed singletons in the scene): never multi-instance even
# when the instruction has a compound trigger. "put both X and Y in Z"
# has multiple PICK targets but one shared PLACE landmark.
_LANDMARK_TOKENS = ('basket', 'stove', 'cabinet', 'drawer', 'table',
                     'caddy', 'rack', 'microwave', 'tray', 'shelf',
                     'sink', 'counter', 'wooden cabinet', 'compartment')


def _is_landmark_prompt(p: str) -> bool:
    pl = p.lower()
    return any(tok in pl for tok in _LANDMARK_TOKENS)


def detect_scene_privileged(env, prompts: list[str], obs: dict, cfg,
                              cam: Optional[CameraParams] = None) -> DetectionResult:
    """
    CaP-X-style privileged detection: bypass SAM3, use MuJoCo GT body positions.

    Reads the env's BDDL :objects-of-interest list to know which bodies are
    semantically relevant, then matches BDDL object names to MuJoCo body names
    by substring overlap. Each match becomes an ObjectDetection with the
    body's world position, perfect confidence, and the BDDL/prompt label.

    Gemini still runs downstream on det_map (label -> position_3d); only
    SAM3 is bypassed.  The agentview RGB is still captured for the
    planner's annotated image.
    """
    model = env.sim.model._model
    data = env.sim.data._data
    mujoco.mj_forward(model, data)
    rgb = obs.get('agentview_image')
    if rgb is None:
        return DetectionResult(rgb=np.zeros((1, 1, 3), np.uint8),
                                depth=None, wrist_rgb=None,
                                cam=cam or get_camera_params(env))
    rgb = _flip_render(rgb)
    depth = _render_depth(env, obs.get('agentview_depth'))
    wrist_rgb_raw = obs.get('robot0_eye_in_hand_image')
    wrist_rgb_disp = _flip_render(wrist_rgb_raw) if wrist_rgb_raw is not None else None
    if cam is None:
        cam = get_camera_params(env)

    # Pull BDDL :objects-of-interest as the canonical object list.
    bddl = (getattr(env, 'bddl_file_name', None)
            or getattr(getattr(env, 'env', None), 'bddl_file_name', None))
    bddl_objs: list[str] = []
    if bddl:
        try:
            bddl_objs = get_bddl_objects_of_interest(bddl)
        except Exception:
            bddl_objs = []
    # Use Gemini-emitted prompts as the ONLY labels so BT keypoint lookups
    # match what Gemini emits. BDDL :objects-of-interest determines which
    # MuJoCo bodies/sites are eligible candidate positions, but the label
    # stored in det_map (and seen by Gemini via image annotations) is the
    # colloquial prompt string.
    labels = list(prompts)

    dets: list = []
    # Build candidate pools: MuJoCo bodies (free joints / physical objects) AND
    # sites (target regions like ``flat_stove_1_cook_region``). LIBERO BDDL
    # :objects-of-interest mixes both - bowls and bottles are bodies, while
    # cook/front/cabinet regions are sites.
    nbody = model.nbody
    body_names = []
    for bi in range(nbody):
        bn = mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_BODY, bi) or ''
        body_names.append(bn.lower())
    nsite = model.nsite
    site_names = []
    for si in range(nsite):
        sn = mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_SITE, si) or ''
        site_names.append(sn.lower())

    def _match_tokens(label_low: str, name: str) -> bool:
        tokens = [t for t in label_low.split('_') if len(t) >= 3]
        if not tokens:
            return False
        return all(t in name for t in tokens) or label_low in name

    for label in labels:
        label_low = label.lower().replace(' ', '_')
        # Try bodies first (physical objects).
        body_cands = [bi for bi, bn in enumerate(body_names)
                      if not any(skip in bn for skip in ('robot0', 'gripper0', 'mount0', 'world'))
                      and _match_tokens(label_low, bn)]
        pos = None
        if body_cands:
            best_bid = min(body_cands, key=lambda b: len(body_names[b]))
            pos = data.xpos[best_bid].copy()
        else:
            # Fall back to sites (region targets).
            site_cands = [si for si, sn in enumerate(site_names)
                          if _match_tokens(label_low, sn)]
            if site_cands:
                best_sid = min(site_cands, key=lambda s: len(site_names[s]))
                pos = data.site_xpos[best_sid].copy()
        if pos is None:
            continue
        dets.append(ObjectDetection(
            label=label, confidence=1.0, centroid_2d=(0, 0), mask_area=0,
            depth_meters=float(pos[2]), position_3d=np.asarray(pos, dtype=float),
        ))

    det_map = _build_det_map(dets, cfg)
    return DetectionResult(rgb=rgb, depth=depth, wrist_rgb=wrist_rgb_disp,
                             cam=cam, det_map=det_map, dets=dets)


def detect_scene(env, sam3, prompts: list[str], obs: dict, cfg,
                  cam: Optional[CameraParams] = None) -> DetectionResult:
    """
    Render + detect from agentview, optionally merge with wrist camera.

    Returns a fully-populated ``DetectionResult`` (rgb, depth, det_map,
    cam params).  When ``cam`` is omitted we resolve agentview ourselves.
    """
    model = env.sim.model._model
    data = env.sim.data._data
    mujoco.mj_forward(model, data)

    rgb = obs.get('agentview_image')
    if rgb is None:
        return DetectionResult(rgb=np.zeros((1, 1, 3), np.uint8),
                                depth=None, wrist_rgb=None,
                                cam=cam or get_camera_params(env))
    rgb = _flip_render(rgb)
    depth = _render_depth(env, obs.get('agentview_depth'))
    wrist_rgb_raw = obs.get('robot0_eye_in_hand_image')
    wrist_rgb_disp = _flip_render(wrist_rgb_raw) if wrist_rgb_raw is not None else None
    if cam is None:
        cam = get_camera_params(env)

    service = _service_client_if_enabled(cfg)
    sam3_thr = float(getattr(sam3, 'sam3_threshold', 0.03))

    # Multi-instance detection: SAM3 returns ALL masks (not just top-1) for
    # prompts that may have several instances. Handles/knobs are always
    # multi-instance (drawer pulls, stove knobs); any prompt referenced by a
    # compound instruction ("both", "two"/"three") is flagged so the planner
    # can chain a pick-place per instance.
    instr = (getattr(cfg, '_current_instruction', '') or '').lower()
    compound_triggers = ('both', ' two ', ' three ', 'all the ', 'each ')
    has_compound = any(t in instr for t in compound_triggers)
    multi_instance = {p for p in prompts
                       if 'handle' in p or 'knob' in p
                       or (has_compound and not _is_landmark_prompt(p))}
    if cfg.verbose:
        print(f"[multi-instance] has_compound={has_compound} "
              f"engaged={sorted(multi_instance)} all_prompts={prompts}",
              flush=True)
    dets = _detect_one_cam(sam3, rgb, depth, prompts, cam,
                            cfg.cam_width, cfg.cam_height, multi_instance,
                            service=service, sam3_threshold=sam3_thr)

    # Optional wrist camera fusion.
    wrist_depth_raw = obs.get('robot0_eye_in_hand_depth')
    if wrist_rgb_disp is not None and wrist_depth_raw is not None:
        wrist_depth = _render_depth(env, wrist_depth_raw)
        if wrist_depth is not None:
            mujoco.mj_forward(model, data)
            cam_w_id = env.sim.model.camera_name2id('robot0_eye_in_hand')
            if cam_w_id >= 0:
                wrist_cam = CameraParams(
                    pos=data.cam_xpos[cam_w_id].copy(),
                    mat=data.cam_xmat[cam_w_id].reshape(3, 3).copy(),
                    fovy=float(model.cam_fovy[cam_w_id]))
                dets_w = _detect_one_cam(
                    sam3, wrist_rgb_disp, wrist_depth, prompts, wrist_cam,
                    cfg.cam_width, cfg.cam_height, multi_instance,
                    service=service, sam3_threshold=sam3_thr)
                dets = _merge_wrist_dets(dets, dets_w,
                                          verbose=bool(getattr(cfg, 'verbose',
                                                                False)))

    det_map = _build_det_map(dets, cfg)
    return DetectionResult(rgb=rgb, depth=depth, wrist_rgb=wrist_rgb_disp,
                             cam=cam, det_map=det_map, dets=dets)


def annotation_rescue(det: DetectionResult, required_labels: list[str],
                        cfg, sam3=None,
                        trial_meta: Optional[dict] = None) -> None:
    """
    Detect-rescue rung (cfg.annotation_rescue): point -> SAM3 click seed.

    For every entry of ``required_labels`` (the active prompt list) that
    produced NO detection after the existing prompt-selection layers, ask
    the configured annotation provider to POINT at it, seed SAM3's click
    head with that pixel (the same :func:`_refine_from_click` machinery
    Phase 4 uses), backproject the rescued mask on the object's depth
    exactly like the text-prompt path, and bind the result under the
    missing label.  Pointing sidesteps SAM3's text vocabulary, targeting
    the viewpoint-dependent miss where no phrasing variant lands.

    Mutates ``det.det_map`` / ``det.dets`` in place; each attempt is
    recorded in ``trial_meta['annotation_rescue']`` (label, provider,
    point, outcome).  Fail-open at every step: no provider, no point, no
    mask, or no valid depth just leaves the label missing, exactly as if
    the rung were off.
    """
    import time as _time
    if not getattr(cfg, 'annotation_rescue', False):
        return
    rename = getattr(cfg, '_mp_phrase_to_concept', None) or {}
    missing = [p for p in (required_labels or [])
               if p not in det.det_map and rename.get(p, p) not in det.det_map]
    if not missing or det.depth is None:
        return

    from spark_real.perception.annotations import get_provider
    provider = get_provider(getattr(cfg, 'annotation_provider', 'er2'))
    if provider is None or not provider.available():
        if cfg.verbose:
            print(f"[AnnRescue] provider "
                  f"{getattr(cfg, 'annotation_provider', 'er2')!r} unavailable")
        return

    h, w = det.rgb.shape[:2]
    service = _service_client_if_enabled(cfg)
    click_state: Optional[dict] = None
    meta = (trial_meta.setdefault('annotation_rescue', [])
            if trial_meta is not None else [])

    for label in missing:
        rec = {'label': label, 'provider': provider.name, 'ok': False,
                't': _time.time()}
        meta.append(rec)
        try:
            anns = provider.annotate(det.rgb, label, kind='point')
        except Exception as e:  # noqa: BLE001 - rescue must never break a trial
            rec['error'] = f'annotate: {e}'
            continue
        if not anns:
            rec['error'] = 'no_point'
            continue
        u, v = anns[0].to_pixels((w, h))[0]
        rec['point_px'] = [round(u, 1), round(v, 1)]

        # Seed SAM3's click head. The service path re-encodes per call so
        # an empty state dict suffices; in-process needs set_image once.
        if click_state is None and service is None:
            try:
                preflight = sam3 or _get_sam3()
                preflight.load_models(load_da3=False)
                click_state = preflight._sam3.set_image(Image.fromarray(det.rgb))
            except Exception as e:  # noqa: BLE001
                rec['error'] = f'set_image: {e}'
                continue
        cand = _refine_from_click(click_state or {}, det.rgb,
                                    (int(u), int(v)), label, cands=None,
                                    service=service)
        if cand is None or cand.get('mask') is None:
            rec['error'] = 'no_mask'
            continue
        mask = cand['mask']

        # Depth-aware backprojection, same as the text-prompt path
        # (_detect_via_service).
        f = h / (2 * np.tan(np.deg2rad(det.cam.fovy) / 2))
        ys, xs = np.where(mask > 0)
        if len(xs) == 0:
            rec['error'] = 'empty_mask'
            continue
        mask_depths = det.depth[mask > 0]
        valid = (mask_depths > 0.01) & (mask_depths < 10)
        if valid.sum() < 3:
            rec['error'] = 'no_depth'
            continue
        vds = mask_depths[valid]
        x_cams = (xs[valid].astype(np.float64) - w / 2) * vds / f
        y_cams = -(ys[valid].astype(np.float64) - h / 2) * vds / f
        pts_cam = np.stack([x_cams, y_cams, -vds], axis=1)
        pts_world = (det.cam.mat @ pts_cam.T).T + det.cam.pos
        point_world = np.median(pts_world, axis=0)

        rescued = ObjectDetection(
            label=label,
            confidence=float(cand.get('score', 0.0)),
            centroid_2d=(float(xs.mean()), float(ys.mean())),
            mask_area=len(xs),
            depth_meters=float(np.percentile(vds, 25)),
            position_3d=point_world,
        )
        rescued.mask = mask
        det.dets.append(rescued)
        det.det_map[rename.get(label, label)] = rescued
        rec['ok'] = True
        rec['mask_area'] = int(len(xs))
        rec['pos'] = [round(float(c), 4) for c in point_world]
        if cfg.verbose:
            print(f"[AnnRescue] {label!r}: {provider.name} point "
                  f"({u:.0f},{v:.0f}) -> mask {len(xs)}px "
                  f"pos=({point_world[0]:.3f},{point_world[1]:.3f},"
                  f"{point_world[2]:.3f})")


def disambiguate_pick_instance(det: DetectionResult, target_hint: str,
                                 instruction: str, cfg,
                                 sam3=None,
                                 trial_meta: Optional[dict] = None) -> None:
    """
    Optional bind-time instance disambiguation (cfg.instance_disambig).

    When more than one instance of the pick target's label is visible
    (e.g. the spatial suite's twin black bowls), position-sticky binding
    can hold an identity but cannot know WHICH instance satisfies the
    instruction's spatial relation ("the black bowl between the plate
    and the ramekin").  This hook reuses the existing multiphase Phase-3
    machinery - :func:`_gemini_phase3_refine` renders the candidate
    masks with IDs and asks the flash-tier ``cfg.phase3_model`` to pick
    the instance matching the instruction, or return a click point
    (resolved to the nearest candidate centroid, the Phase-4 fallback).

    Fires only when >1 instance is actually detected: if the initial
    ``det.dets`` carries a single mask for the label, SAM3 is re-run on
    that one prompt in multi-instance mode against the already-captured
    frame.  On any failure (no key, one instance, Gemini error) the
    binding is left untouched.  Mutates ``det.det_map[key]`` in place;
    records a ``trial_meta['instance_disambig']`` entry when it acts.
    """
    import time as _time
    from spark_real.perception.sticky_binding import fuzzy_key
    if not getattr(cfg, 'instance_disambig', False) or not target_hint:
        return
    key = fuzzy_key(det.det_map, target_hint)
    if key is None:
        # BDDL hints ('akita_black_bowl_1') often share no substring with
        # prompt phrases; retry on normalized token overlap.
        toks = {t for t in target_hint.lower().replace('_', ' ').split()
                if len(t) >= 3}
        best, best_n = None, 0
        for k in det.det_map:
            n = len(toks & set(str(k).lower().split()))
            if n > best_n:
                best, best_n = k, n
        key = best
    if key is None:
        return
    label = getattr(det.det_map[key], 'label', key)

    instances = [d for d in det.dets
                 if d.label == label and d.position_3d is not None
                 and getattr(d, 'mask', None) is not None]
    if len(instances) < 2 and det.depth is not None:
        # Initial detection kept only top-1: re-run this ONE prompt in
        # multi-instance mode against the frame already in hand.
        try:
            service = _service_client_if_enabled(cfg)
            sam3_thr = float(getattr(sam3, 'sam3_threshold', 0.03))
            dets2 = _detect_one_cam(
                sam3, det.rgb, det.depth, [label], det.cam,
                cfg.cam_width, cfg.cam_height, {label},
                service=service, sam3_threshold=sam3_thr)
            instances = [d for d in dets2
                         if d.position_3d is not None
                         and getattr(d, 'mask', None) is not None]
        except Exception as e:
            if cfg.verbose:
                print(f"[InstanceDisambig] multi-instance redetect "
                      f"failed: {e}")
            return
    if len(instances) < 2:
        return  # nothing to disambiguate

    cands = [{'phrase': label, 'mask': d.mask,
                'score': float(d.confidence),
                'centroid': (float(d.centroid_2d[0]),
                              float(d.centroid_2d[1])),
                'bbox': getattr(d, 'bbox', None)}
             for d in instances]
    decisions = _gemini_phase3_refine(det.rgb, {key: cands},
                                        instruction, cfg)
    d = decisions.get(key) or {}
    chosen, how = None, None
    if 'choice' in d:
        # Single concept: mask ids are m0..m{n-1} in candidate order.
        try:
            idx = int(str(d['choice']).lstrip('m'))
            if 0 <= idx < len(instances):
                chosen, how = instances[idx], 'choice'
        except (TypeError, ValueError):
            pass
    elif 'click' in d:
        cx, cy = d['click']
        chosen = min(instances,
                     key=lambda i: ((i.centroid_2d[0] - cx) ** 2
                                     + (i.centroid_2d[1] - cy) ** 2))
        how = 'click'
    if chosen is None:
        return
    det.det_map[key] = chosen
    if trial_meta is not None:
        trial_meta.setdefault('instance_disambig', []).append({
            'label': key, 'n_instances': len(instances), 'how': how,
            'chosen_conf': round(float(chosen.confidence), 3),
            't': _time.time()})
    if cfg.verbose:
        p = chosen.position_3d
        print(f"[InstanceDisambig] {key!r}: {len(instances)} instances, "
              f"Phase-3 picked via {how} -> "
              f"({p[0]:.3f},{p[1]:.3f},{p[2]:.3f})")


def _build_det_map(dets: list, cfg) -> dict:
    """
    Index detections by label, suffixing duplicates (e.g. ``bowl_2``).

    When ``--multiphase-adaptive`` populated ``cfg._mp_phrase_to_concept``,
    variant phrases (e.g. ``"ceramic dish"``) are renamed back to the
    canonical concept (e.g. ``"plate"``) so downstream Phase 2 BT-gen and
    the executor's ``fuzzy_get_det`` see stable labels.
    """
    rename = getattr(cfg, '_mp_phrase_to_concept', None) or {}
    det_map: dict = {}
    label_counts: dict[str, int] = {}
    for d in dets:
        if d.position_3d is None:
            continue
        base = rename.get(d.label, d.label)
        if base in det_map:
            label_counts[base] = label_counts.get(base, 1) + 1
            key = f"{base}_{label_counts[base]}"
        else:
            key = base
        det_map[key] = d
        if cfg.verbose:
            p = d.position_3d
            tag = f"{key}" if base == d.label else f"{key}<-{d.label}"
            print(f"[{tag}] conf={d.confidence:.3f} "
                  f"pos=({p[0]:.3f},{p[1]:.3f},{p[2]:.3f})")
    return det_map


def redetect_wrist(env, sam3, prompts: list[str], cfg,
                    all_instances: bool = False) -> Optional[DetectionResult]:
    """
    Re-render + re-detect from the eye-in-hand camera only.

    The agentview loses a target the moment the gripper hovers over it;
    the wrist camera sees exactly that region.  Same contract as
    redetect_agentview: None on render failure or no detections.
    """
    model = env.sim.model._model
    data = env.sim.data._data
    obs2, _, _, _ = env.step(np.zeros(7))
    rgb2 = obs2.get('robot0_eye_in_hand_image')
    depth_raw = obs2.get('robot0_eye_in_hand_depth')
    if rgb2 is None or depth_raw is None:
        return None
    rgb2 = _flip_render(rgb2)
    depth2 = _render_depth(env, depth_raw)
    if depth2 is None:
        return None
    mujoco.mj_forward(model, data)
    cam_id = env.sim.model.camera_name2id('robot0_eye_in_hand')
    if cam_id < 0:
        return None
    cam = CameraParams(pos=data.cam_xpos[cam_id].copy(),
                       mat=data.cam_xmat[cam_id].reshape(3, 3).copy(),
                       fovy=float(model.cam_fovy[cam_id]))
    service = _service_client_if_enabled(cfg)
    multi = set(prompts) if all_instances else set()
    if service is not None:
        sam3_thr = float(getattr(sam3, 'sam3_threshold', 0.03))
        dets = _detect_via_service(
            service, rgb2, depth2, prompts, cam.pos, cam.mat, cam.fovy,
            cfg.cam_width, cfg.cam_height, sam3_threshold=sam3_thr,
            multi_instance_prompts=multi)
    else:
        dets = sam3._detect_with_rendered_depth(
            rgb2, depth2, prompts, cam.pos, cam.mat, cam.fovy,
            cfg.cam_width, cfg.cam_height,
            multi_instance_prompts=multi)
    if not dets:
        return None
    rename = getattr(cfg, '_mp_phrase_to_concept', None) or {}
    det_map = {rename.get(d.label, d.label): d for d in dets
               if d.position_3d is not None}
    return DetectionResult(rgb=rgb2, depth=depth2, wrist_rgb=None,
                             cam=cam, det_map=det_map, dets=dets)


def redetect_agentview(env, sam3, prompts: list[str], cfg,
                        timing: Optional[dict] = None,
                        all_instances: bool = False
                        ) -> Optional[DetectionResult]:
    """
    Re-render + re-detect from agentview only (recovery-loop helper).

    No wrist fusion (the recovery scenario doesn't need it).  Returns
    ``None`` if rendering fails or there are no detections.

    ``timing`` (when supplied) receives ``t_capture`` - the wall-clock
    time right after the RGB/depth frame is in hand and before the SAM3
    call - so event-capture callers can split capture latency from
    verdict latency.

    ``all_instances=True`` flags every prompt multi-instance so SAM3
    returns EVERY mask above threshold, not just top-1.  The
    ``DetectionResult.dets`` list then carries all instances (several
    entries may share a label) for position-sticky association; the
    ``det_map`` keeps its legacy one-entry-per-label shape.
    """
    import time as _time
    model = env.sim.model._model
    data = env.sim.data._data
    obs2, _, _, _ = env.step(np.zeros(7))
    rgb2 = obs2.get('agentview_image')
    if rgb2 is None:
        return None
    rgb2 = _flip_render(rgb2)
    depth2 = _render_depth(env, obs2.get('agentview_depth'))
    if depth2 is None:
        return None
    if timing is not None:
        timing['t_capture'] = _time.time()
    mujoco.mj_forward(model, data)
    cam = get_camera_params(env)
    service = _service_client_if_enabled(cfg)
    multi = set(prompts) if all_instances else set()
    if service is not None:
        sam3_thr = float(getattr(sam3, 'sam3_threshold', 0.03))
        dets = _detect_via_service(
            service, rgb2, depth2, prompts, cam.pos, cam.mat, cam.fovy,
            cfg.cam_width, cfg.cam_height, sam3_threshold=sam3_thr,
            multi_instance_prompts=multi)
    else:
        dets = sam3._detect_with_rendered_depth(
            rgb2, depth2, prompts, cam.pos, cam.mat, cam.fovy,
            cfg.cam_width, cfg.cam_height,
            multi_instance_prompts=multi)
    if not dets:
        return None
    rename = getattr(cfg, '_mp_phrase_to_concept', None) or {}
    det_map = {rename.get(d.label, d.label): d for d in dets
               if d.position_3d is not None}
    return DetectionResult(rgb=rgb2, depth=depth2, wrist_rgb=None,
                             cam=cam, det_map=det_map, dets=dets)

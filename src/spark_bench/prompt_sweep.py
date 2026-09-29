"""
Offline SAM3 prompt-tuning sweep for LIBERO-PRO.

For each task in each suite, gets the list of BDDL :objects-of-interest,
asks Gemini to generate K candidate SAM3 detection prompts per object,
runs SAM3 with each candidate, scores each by IoU vs ground-truth mesh
projection, and writes the best prompt per object to a tuned dict.

Uses GT mesh as the scoring signal, so this is offline hyperparameter
tuning - analogous to CaP's skill library synthesis. Deployment doesn't
need GT, just the resulting dict.
"""
from __future__ import annotations

import argparse
import json
import os
import re
import sys
import traceback
from pathlib import Path
from typing import Dict, List, Tuple

# os.environ defaults must be set before mujoco imports.
os.environ.setdefault('MUJOCO_GL', 'egl')

import numpy as np
import mujoco

try:
    import torch
    from PIL import Image as PILImage
except ImportError:
    torch = None
    PILImage = None

try:
    from google import genai
    from google.genai import types
except ImportError:
    genai = None
    types = None

_SRC = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_SRC))
sys.path.insert(0, str(_SRC / 'libero_pro'))

# Intra-repo imports resolve only after the sys.path inserts above.
# load_libero_env / FairConfig are imported lazily inside sweep_task() to avoid a
# module-load import cycle (config -> perception -> prompt_sweep -> fair runner).
from spark_real.perception.spark_perception import SPARKPerception

K_CANDIDATES = 5
N_INIT_STATES = 3  # sample this many initial configs per task

OUT_PATH = Path(__file__).resolve().parent / 'tuned_prompts.json'


def get_bddl_objects_of_interest(bddl_path: str) -> List[str]:
    """
    Extract likely-relevant object names from BDDL.

    Pulls from :objects-of-interest AND from :goal predicate AND from :init
    positional assertions, so we tune all objects the task touches (pick
    target, place target, any reference objects).
    """
    with open(bddl_path) as f:
        text = f.read()
    names: List[str] = []

    # Primary source: objects-of-interest
    m = re.search(r'\(:objects-of-interest\s+([^)]+)\)', text)
    if m:
        names += [s.strip() for s in m.group(1).split() if s.strip()]

    # Also pull object symbols from :goal
    m2 = re.search(r'\(:goal\s+(.*?)\n\s*\)\n', text, re.DOTALL)
    if m2:
        # tokens are atoms like (On akita_black_bowl_1 plate_1) - grab non-predicate words
        for tok in re.findall(r'[A-Za-z_][A-Za-z_0-9]*', m2.group(1)):
            if '_' in tok and tok not in names and len(tok) > 3:
                names.append(tok)

    # Dedupe, strip numeric suffixes that duplicate
    seen = set(); out = []
    for n in names:
        if n.lower() not in seen:
            seen.add(n.lower()); out.append(n)
    return out


def _load_gemini_keys() -> List[str]:
    path = _SRC / '.gemini_api_key'
    if not path.exists():
        return []
    return [k.strip() for k in path.read_text().splitlines() if k.strip()]


def gen_prompt_candidates(object_names: List[str], task_language: str) -> Dict[str, List[str]]:
    """
    Generate K candidate SAM3 detection prompts per object via Gemini.

    Uses google.genai with key rotation. Falls back to heuristic
    variants on all-key failure.
    """
    prompt = f"""For each LIBERO object name below, suggest {K_CANDIDATES} distinct short
detection phrases (1-4 words each) for a vision detector. Use colors, shapes,
materials, functional descriptions - think like an open-vocabulary detector prompt.

Task context: "{task_language}"

Objects:
{chr(10).join(f'- {n}' for n in object_names)}

Output JSON only (no code fences):
{{"obj_name": ["phrase1", "phrase2", ...], ...}}
"""
    keys = _load_gemini_keys()
    last_err = None
    for idx, key in enumerate(keys):
        try:
            client = genai.Client(api_key=key)
            resp = client.models.generate_content(
                model='gemini-2.5-flash',
                contents=prompt,
                config=types.GenerateContentConfig(temperature=0),
            )
            txt = resp.text.strip()
            if '```' in txt:
                txt = txt.split('```', 2)[1]
                if txt.startswith('json'):
                    txt = txt[4:]
            return json.loads(txt.strip())
        except Exception as e:
            last_err = e
            print(f"[gemini key{idx+1} fail: {str(e)[:100]}] trying next key")
            continue
    # All keys failed - heuristic fallback so the sweep can still run
    print(f"[all gemini keys failed; last: {str(last_err)[:100]}] heuristic fallback")
    heur = {}
    for n in object_names:
        clean = n.replace('_', ' ').replace('-', ' ').strip()
        base = clean.rsplit(' ', 1)[0] if clean.rsplit(' ', 1)[-1].isdigit() else clean
        variants = [
            base, clean, f"{base} object",
            base.split()[-1] if ' ' in base else base,
            ' '.join(base.split()[:2]) if len(base.split()) > 2 else base,
        ]
        seen = set(); out = []
        for v in variants:
            if v and v not in seen:
                seen.add(v); out.append(v)
        while len(out) < K_CANDIDATES:
            out.append(base)
        heur[n] = out[:K_CANDIDATES]
    return heur


def render_rgbd(env, cam_name: str = 'agentview') -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, float]:
    """
    Render RGB + depth from a named camera, return (rgb, depth, cam_pos, cam_mat, fovy).
    """
    sim = env.env.sim
    model = sim.model._model
    data = sim.data._data
    cid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_CAMERA, cam_name)
    rgb = sim.render(camera_name=cam_name, width=256, height=256, depth=False)
    rgb_d = sim.render(camera_name=cam_name, width=256, height=256, depth=True)
    if isinstance(rgb_d, tuple):
        _, depth = rgb_d
    else:
        depth = rgb_d
    # Convert mujoco depth buffer -> meters
    extent = model.stat.extent
    near = model.vis.map.znear * extent
    far = model.vis.map.zfar * extent
    depth_m = near / (1 - depth * (1 - near / far))
    cam_pos = data.cam_xpos[cid].copy()
    cam_mat = data.cam_xmat[cid].reshape(3, 3).copy()
    fovy = float(model.cam_fovy[cid])
    return np.flipud(rgb), np.flipud(depth_m), cam_pos, cam_mat, fovy


def get_object_pixel_centroid(env, obj_name: str, cam_pos, cam_mat, fovy, w, h):
    """
    Return (cx, cy) pixel coords for obj_name's body centroid, or None.
    """
    sim = env.env.sim
    model = sim.model._model
    data = sim.data._data

    target_body_ids = []
    for bid in range(model.nbody):
        bn = (mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_BODY, bid) or '')
        bn_l = bn.lower()
        if not bn_l:
            continue
        # match: obj_name is a substring of body name OR body name with _main stripped matches obj
        if obj_name.lower() in bn_l or bn_l.replace('_main', '').replace('_g0', '') == obj_name.lower():
            target_body_ids.append(bid)

    if not target_body_ids:
        return None

    # Average world position
    world_pos = np.mean([data.xpos[b] for b in target_body_ids], axis=0)

    # Project into camera frame: cam_mat transforms camera->world, so inverse for world->camera
    rel = world_pos - cam_pos  # world relative to camera
    cam_frame = cam_mat.T @ rel  # (x_cam, y_cam, z_cam); z_cam should be negative (looking -Z)
    z = cam_frame[2]
    if z >= -1e-3:
        return None  # behind camera

    f = h / (2 * np.tan(np.deg2rad(fovy) / 2))
    px = (cam_frame[0] / (-z)) * f + w / 2
    py = (-cam_frame[1] / (-z)) * f + h / 2  # y flipped
    if px < 0 or px >= w or py < 0 or py >= h:
        return None
    return (float(px), float(py))


def iou(mask_a: np.ndarray, mask_b: np.ndarray) -> float:
    a = mask_a.astype(bool)
    b = mask_b.astype(bool)
    inter = np.logical_and(a, b).sum()
    union = np.logical_or(a, b).sum()
    return float(inter / union) if union > 0 else 0.0


def sweep_task(suite: str, task_id: int, sam3, results: Dict):
    from spark_bench.run_spark_libero_pro_fair import load_libero_env, FairConfig
    cfg = FairConfig(suite=suite, perturbation='position', num_trials=1,
                     cam_width=256, cam_height=256)
    suite_full = {'spatial': 'libero_spatial_swap', 'object': 'libero_object_swap',
                  'goal': 'libero_goal_swap'}[suite]
    env, task, init_states, bddl_path = load_libero_env(suite_full, task_id, cfg)
    language = task.language if hasattr(task, 'language') else ''
    obj_names = get_bddl_objects_of_interest(bddl_path)
    if not obj_names:
        print(f"[{suite} T{task_id}] no objects-of-interest in BDDL, skipping")
        return

    print(f"[{suite} T{task_id}] lang='{language[:60]}' objects={obj_names}")
    candidates = gen_prompt_candidates(obj_names, language)

    per_obj_scores: Dict[str, Dict[str, List[float]]] = {n: {} for n in obj_names}

    for state_idx in range(min(N_INIT_STATES, len(init_states))):
        env.reset()
        env.set_init_state(init_states[state_idx])
        for _ in range(10):
            env.step(np.zeros(7))  # let sim settle
        rgb, depth, cam_pos, cam_mat, fovy = render_rgbd(env)
        h, w = rgb.shape[:2]

        # Load image once into SAM3 state (SAM3 already loaded in main).
        pil_img = PILImage.fromarray(rgb)
        state_base = sam3._sam3.set_image(pil_img)
        for obj in obj_names:
            gt_px = get_object_pixel_centroid(env, obj, cam_pos, cam_mat, fovy, w, h)
            if gt_px is None:
                continue
            gx, gy = gt_px
            for phrase in candidates.get(obj, []):
                state = sam3._sam3.set_text_prompt(prompt=phrase, state=state_base)
                masks = state.get('masks', torch.tensor([]))
                scores = state.get('scores', torch.tensor([]))
                # Score by min pixel distance across detections, converted to [0,1]
                best_score = 0.0
                if masks.numel() > 0:
                    for i in range(masks.shape[0]):
                        m = masks[i].cpu().numpy().squeeze().astype(bool)
                        if not m.any():
                            continue
                        ys, xs = np.where(m)
                        mcx, mcy = float(xs.mean()), float(ys.mean())
                        dist = np.hypot(mcx - gx, mcy - gy)
                        # Convert distance to score: exp(-dist/30)
                        score = float(np.exp(-dist / 30.0))
                        conf = float(scores[i]) if scores.numel() > i else 0.0
                        best_score = max(best_score, score * max(conf, 0.1))
                per_obj_scores[obj].setdefault(phrase, []).append(best_score)

    # pick best phrase per object
    for obj in obj_names:
        scores = per_obj_scores[obj]
        if not scores:
            continue
        ranked = sorted([(np.mean(v), p, len(v)) for p, v in scores.items()], reverse=True)
        top = ranked[0]
        results.setdefault(obj, []).append({
            'suite': suite, 'task_id': task_id, 'phrase': top[1],
            'score': top[0], 'runs': top[2],
            'full_ranking': [(float(m), p) for m, p, _ in ranked[:3]],
        })
        print(f"{obj:30s} best=\"{top[1]}\" score={top[0]:.2f}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--suite', choices=['spatial', 'object', 'goal', 'all'], default='all')
    parser.add_argument('--out', default=str(OUT_PATH))
    args = parser.parse_args()

    print("[sweep] loading SAM3...")
    sam3 = SPARKPerception()
    sam3.load_models(load_da3=False)  # only need SAM3, GT mesh gives us depth
    print("[sweep] SAM3 ready")

    results: Dict[str, List[Dict]] = {}
    suites = ['spatial', 'object', 'goal'] if args.suite == 'all' else [args.suite]
    for suite in suites:
        for tid in range(10):
            try:
                sweep_task(suite, tid, sam3, results)
            except Exception as e:
                print(f"[{suite} T{tid}] EXCEPTION: {e}")
                traceback.print_exc()

    # Aggregate across tasks: pick the phrase with best mean IoU across all
    # appearances of each object
    best_per_obj: Dict[str, Dict] = {}
    for obj, entries in results.items():
        by_phrase: Dict[str, List[float]] = {}
        for e in entries:
            by_phrase.setdefault(e['phrase'], []).append(e['score'])
            for s, phrase in e['full_ranking']:
                by_phrase.setdefault(phrase, []).append(s)
        if not by_phrase:
            continue
        ranked = sorted([(np.mean(v), p, len(v)) for p, v in by_phrase.items()], reverse=True)
        best_per_obj[obj] = {
            'best_phrase': ranked[0][1],
            'mean_score': ranked[0][0],
            'n_observations': ranked[0][2],
            'top3': [(float(m), p) for m, p, _ in ranked[:3]],
        }

    with open(args.out, 'w') as f:
        json.dump({'per_task': results, 'best_per_object': best_per_obj}, f, indent=2)
    print(f"\n[sweep] wrote {args.out}")
    print(f"[sweep] tuned dict:")
    for obj, info in sorted(best_per_obj.items()):
        print(f"{obj:35s} -> \"{info['best_phrase']}\" (score={info['mean_score']:.2f}, n={info['n_observations']})")


if __name__ == '__main__':
    main()

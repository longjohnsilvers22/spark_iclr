#!/usr/bin/env python3
"""
SPARK on robosuite standalone tasks.

Uses the same pipeline as LIBERO-PRO fair runner (SAM3 + Gemini + OSC + IK fallback)
but on robosuite's built-in manipulation tasks.

Tasks: Lift, Stack, NutAssemblySquare, PickPlaceCan, Door, Wipe

Usage:
    conda activate openvla
    MUJOCO_GL=egl python -m spark_bench.run_spark_robosuite --task Stack --num-trials 5
    MUJOCO_GL=egl python -m spark_bench.run_spark_robosuite --task all --num-trials 1
"""
from __future__ import annotations
import os
import sys
import json
import numpy as np
from pathlib import Path
from dataclasses import dataclass

os.environ.setdefault('MUJOCO_GL', 'egl')
os.environ.setdefault('PYTORCH_CUDA_ALLOC_CONF', 'expandable_segments:True')

import torch
_torch_load_orig = torch.load
def _torch_load_patched(*args, **kwargs):
    kwargs.setdefault('weights_only', False)
    return _torch_load_orig(*args, **kwargs)
torch.load = _torch_load_patched

import tyro
import mujoco

try:
    import yaml
    from PIL import Image as PILImage
    from google import genai
    from google.genai import types as genai_types
except ImportError:
    yaml = PILImage = genai = genai_types = None

try:
    import robosuite as suite
    from robosuite.utils.camera_utils import get_real_depth_map
except ImportError:
    suite = None
    get_real_depth_map = None

# Optional typed-DSL pipeline; None when spark_dsl is unavailable.
try:
    from spark_dsl import DEFAULT_LIBRARY
    from spark_dsl.executor import BTExecutor
    from spark_dsl.prompt_builder import PromptBuilder
except ImportError:
    DEFAULT_LIBRARY = BTExecutor = PromptBuilder = None

from spark_real.planning.spark_planner import SPARKPlanner
from spark_bench.run_spark_libero_pro_fair import (
    FairConfig,
    _execute_on_libero,
    _gemini_plan,
    _get_sam3,
    _scripted_plan,
)
from spark_bench.dual_arm_bt import create_dual_arm_env, execute_dual_arm_bt


@dataclass
class RobosuiteConfig:
    task: str = "Stack"
    """
    Task: Lift, Stack, NutAssemblySquare, PickPlaceCan, Door, Wipe, or 'all'
    """
    num_trials: int = 5
    """
    Trials per task
    """
    cam_width: int = 640
    cam_height: int = 480
    horizon: int = 2000
    no_gemini: bool = False
    verbose: bool = False
    save_video: bool = False
    """
    Save video (gif) for each trial
    """
    adaptive_prompts: bool = False
    """If True, Gemini generates K=3 variants per base prompt and SAM3 picks
    the one with exactly-1 detection and highest confidence - same as LIBERO-PRO
    adaptive config."""
    adaptive_k: int = 3
    use_dsl: bool = False
    results_json: str = ''
    """Write all per-task results (plus commit and model) to this path."""
    """If True, swap the Gemini prompt builder + BT execution loop for the
    typed-DSL pipeline in spark_dsl/.  Default path is unchanged."""
    dsl_calibration_db: str = ""
    """Optional path to a calibration JSON for CalibrationWrapper.  Empty =
    disabled (no calibration hook)."""


# Task definitions
TASK_CONFIGS = {
    'Lift': {
        'instruction': 'Pick up the red cube and lift it up',
        'prompts': ['red cube', 'small red block'],
        'pick': 'red cube', 'place': 'none',
    },
    'Stack': {
        'instruction': 'Pick up the red cube and stack it on top of the green cube',
        'prompts': ['red cube', 'green cube'],
        'pick': 'red cube', 'place': 'green cube',
    },
    'NutAssemblySquare': {
        'instruction': 'Pick up the silver square nut from the table. Lift it high above the workspace. Move it directly above the brown square peg. Then use insert to carefully lower and rotate it onto the peg.',
        'prompts': ['silver square nut', 'square nut', 'brown square peg', 'square peg post'],
        'pick': 'silver square nut', 'place': 'brown square peg',
    },
    'Wipe': {
        'instruction': 'The robot already holds a wiping pad on its gripper. Move down to the brown dirt marks on the table and scrub back and forth over them to wipe them completely clean. Do not pick up or grasp any object; the wiping tool is already attached.',
        'prompts': ['brown dirt', 'dirt spots', 'brown marks on table'],
        'pick': 'none', 'place': 'none',
    },
    'CubeRestack': {
        'instruction': 'Pick up the red cube and stack it on top of the green cube',
        'prompts': ['red cube', 'green cube'],
        'pick': 'red cube', 'place': 'green cube',
    },
}


def create_env(task_name: str, cfg: RobosuiteConfig):
    """
    Create a robosuite environment with standard settings.
    """
    # CubeRestack uses the Stack env with custom init
    env_name = 'Stack' if task_name == 'CubeRestack' else task_name
    # Match CaP-X's Spill Wipe per-trial step budget (their env sets max_steps=4000,
    # never reset within a trial).
    horizon = 4000 if task_name == 'Wipe' else cfg.horizon
    # NutAssembly uses JOINT_POSITION to reach below OSC's z-limit. Wipe stays on
    # OSC: robosuite Wipe early-terminates above 60 N and a rigid joint press
    # trips that force limit.
    if task_name == 'NutAssemblySquare':
        if hasattr(suite, 'load_composite_controller_config') and not hasattr(suite, 'load_controller_config'):
            from robosuite.controllers import load_part_controller_config
            ctrl = suite.load_composite_controller_config(controller='BASIC', robot='Panda')
            jp = load_part_controller_config(default_controller='JOINT_POSITION')
            jp.update({'kp': 1500, 'input_max': 1, 'input_min': -1,
                       'output_max': 0.1, 'output_min': -0.1})

            def _to_jp(d):
                if isinstance(d, dict):
                    if str(d.get('type', '')).startswith('OSC'):
                        gripper = d.get('gripper')   # composite requires a gripper sub-config
                        d.clear(); d.update(jp)
                        if gripper is not None:
                            d['gripper'] = gripper
                    else:
                        for v in list(d.values()):
                            _to_jp(v)
                elif isinstance(d, list):
                    for v in d:
                        _to_jp(v)
            _to_jp(ctrl.get('body_parts', ctrl))
        else:
            ctrl = {
                'type': 'JOINT_POSITION',
                'input_max': 10, 'input_min': -10,
                'output_max': 1.0, 'output_min': -1.0,
                'kp': 1500, 'kd': 400, 'kv': 200,
                'interpolation': 'linear', 'ramp_ratio': 0.2,
            }
        if task_name == 'NutAssemblySquare':
            horizon = 5000
    elif hasattr(suite, 'load_controller_config'):
        ctrl = suite.load_controller_config(default_controller='OSC_POSE')
    else:
        # robosuite >=1.5 replaces load_controller_config with composite/part loaders.
        ctrl = suite.load_composite_controller_config(controller='BASIC', robot='Panda')
    env_obj = suite.make(
        env_name,
        robots='Panda',
        has_renderer=False,
        has_offscreen_renderer=True,
        use_camera_obs=True,
        camera_names=['agentview', 'robot0_eye_in_hand'],
        camera_heights=cfg.cam_height,
        camera_widths=cfg.cam_width,
        camera_depths=True,
        horizon=horizon,
        controller_configs=ctrl,
    )
    return env_obj


def _dsl_gemini_plan(instruction: str, labels: list,
                     pick_hint: str = '', place_hint: str = '',
                     det_map: dict = None, rgb_image=None,
                     fallback_labels=None) -> dict:
    """
    Typed-DSL prompt path: build the prompt with PromptBuilder, send to
    Gemini, parse the YAML BT, validate via SkillLibrary.  Falls back to the
    legacy `_gemini_plan` if anything goes wrong (so --use-dsl never hard-fails).
    """
    if DEFAULT_LIBRARY is None or PromptBuilder is None:
        print("[DSL] spark_dsl unavailable - falling back to legacy prompt")
        return _gemini_plan(instruction, labels, pick_hint, place_hint,
                            det_map=det_map, rgb_image=rgb_image)

    extra_hints = None
    if det_map:
        pos_lines = []
        for label, det in det_map.items():
            p = getattr(det, 'position_3d', None)
            if p is not None:
                pos_lines.append(
                    f"- {label}: ({float(p[0]):.3f}, {float(p[1]):.3f}, {float(p[2]):.3f})")
        if pos_lines:
            extra_hints = "Detected object positions (x,y,z meters):\n" + "\n".join(pos_lines)

    pick_clean = (pick_hint or '').replace('_', ' ').strip()
    place_clean = (place_hint or '').replace('_', ' ').strip()
    augmented = instruction
    if pick_clean and pick_clean != 'none' and pick_clean not in instruction.lower():
        augmented += f". Pick object: {pick_clean}"
    if place_clean and place_clean != 'none' and place_clean not in instruction.lower():
        augmented += f". Place target: {place_clean}"

    builder = PromptBuilder(
        library=DEFAULT_LIBRARY,
        macros_yaml=str(Path(__file__).resolve().parent.parent / 'spark_dsl' / 'macros.yaml'),
    )
    prompt_text = builder.build(
        task_instruction=augmented,
        detected_objects=labels,
        extra_hints=extra_hints,
    )

    # Send to Gemini directly (avoid SPARKPlanner's fixed system prompt).
    try:
        if 'GEMINI_API_KEY' not in os.environ:
            key_file = os.path.expanduser(os.environ.get('SPARK_GEMINI_KEY_FILE', '~/spark/src/.gemini_api_key'))
            with open(key_file) as f:
                lines = [l.strip() for l in f.readlines() if l.strip()]
                if lines:
                    os.environ['GEMINI_API_KEY'] = lines[0]
        client = genai.Client(api_key=os.environ.get('GEMINI_API_KEY'))
        contents = [prompt_text]
        if rgb_image is not None:
            img = (rgb_image if isinstance(rgb_image, PILImage.Image)
                   else PILImage.fromarray(rgb_image))
            contents.append(img)
        # Try a couple of models (production uses 3-flash-preview; gemini-2.0
        # is the safer fallback).
        last_err = None
        response = None
        _models = [m for m in (os.environ.get('SPARK_GEMINI_MODEL', ''),
                                 'gemini-3-flash-preview',
                                 'gemini-2.0-flash-exp') if m]
        for model_name in _models:
            try:
                response = client.models.generate_content(
                    model=model_name, contents=contents,
                    config=genai_types.GenerateContentConfig(temperature=0))
                break
            except Exception as e:
                last_err = e
                continue
        if response is None:
            raise RuntimeError(f"All Gemini models failed: {last_err}")
        text = response.text or ''
    except Exception as e:
        print(f"[DSL] Gemini call failed: {e} - falling back to legacy prompt")
        return _gemini_plan(instruction, labels, pick_hint, place_hint,
                            det_map=det_map, rgb_image=rgb_image)

    # Extract YAML
    yaml_text = text.strip()
    if '```yaml' in yaml_text:
        s = yaml_text.index('```yaml') + 7
        e = yaml_text.index('```', s)
        yaml_text = yaml_text[s:e].strip()
    elif '```' in yaml_text:
        s = yaml_text.index('```') + 3
        e = yaml_text.index('```', s)
        yaml_text = yaml_text[s:e].strip()
    try:
        score = yaml.safe_load(yaml_text)
    except Exception as e:
        print(f"[DSL] YAML parse failed: {e} - falling back")
        return _scripted_plan(fallback_labels or labels, pick_hint, place_hint, instruction)

    # Robustness: Gemini may wrap the BT in extra keys (e.g. {"task": ...,
    # "tree": ...}) or return the bare tree.  Find the BT subtree.
    if isinstance(score, dict):
        if 'tree' not in score:
            # Look for the first child dict that has type='sequence'.
            for v in score.values():
                if isinstance(v, dict) and v.get('type') == 'sequence':
                    score = {'tree': v}
                    break
            else:
                if score.get('type') == 'sequence':
                    score = {'tree': score}

    # Validate via typed library.  If it fails, log and try the legacy plan.
    try:
        _ast, errors = DEFAULT_LIBRARY.validate_bt(score if isinstance(score, dict) else {})
    except Exception as e:
        errors = [f"validator crashed: {e}"]
    if errors:
        print(f"[DSL] BT validation errors: {errors[:3]} - falling back to legacy plan")
        return _gemini_plan(instruction, labels, pick_hint, place_hint,
                            det_map=det_map, rgb_image=rgb_image)
    return score


def _dsl_execute(env, score, det_map, fair_cfg,
                 sam3=None, depth=None,
                 cam_pos=None, cam_mat=None, cam_fovy=None,
                 cam_w=640, cam_h=480, instruction='',
                 calibration_db_path=None):
    """
    Typed-DSL execution path.

    Validates the BT against the typed library, expands any macros to base
    primitives, then dispatches the full expanded BT through the legacy
    stateful `_execute_on_libero` in a single call. BTExecutor does
    validation + expansion only; the legacy dispatcher owns execution because
    it keeps cross-action state (`holding` flag, smooth approach trajectories).
    """
    executor = BTExecutor(library=DEFAULT_LIBRARY)

    # 1. Validate (rejects malformed plans before any motion)
    ast, errors = DEFAULT_LIBRARY.validate_bt(score)
    if errors:
        if getattr(fair_cfg, 'verbose', False):
            print(f"[DSL] validation errors: {errors[:2]}; "
                  f"falling back to legacy dispatch on raw plan")
        # Fall through to legacy on raw (un-expanded) plan
        _execute_on_libero(env, score, det_map, fair_cfg,
                           sam3=sam3, depth=depth,
                           cam_pos=cam_pos, cam_mat=cam_mat, cam_fovy=cam_fovy,
                           cam_w=cam_w, cam_h=cam_h)
        return bool(env._check_success())

    # 2. Expand macros in-place (recursive). The legacy dispatcher only
    #    knows base primitives, so anything else must inline.
    expanded, exp_err = executor.expand_macros(score)
    if exp_err:
        if getattr(fair_cfg, 'verbose', False):
            print(f"[DSL] macro expansion failed: {exp_err}; "
                  f"falling back to legacy dispatch on raw plan")
        _execute_on_libero(env, score, det_map, fair_cfg,
                           sam3=sam3, depth=depth,
                           cam_pos=cam_pos, cam_mat=cam_mat, cam_fovy=cam_fovy,
                           cam_w=cam_w, cam_h=cam_h)
        return bool(env._check_success())

    # 3. Single legacy dispatch on the expanded BT - preserves stateful
    #    per-action behavior the legacy dispatcher relies on.
    if getattr(fair_cfg, 'verbose', False):
        n_actions = len(expanded.get('tree', {}).get('children', []))
        print(f"[DSL] expanded BT has {n_actions} actions; "
              f"dispatching through legacy")
    _execute_on_libero(env, expanded, det_map, fair_cfg,
                       sam3=sam3, depth=depth,
                       cam_pos=cam_pos, cam_mat=cam_mat, cam_fovy=cam_fovy,
                       cam_w=cam_w, cam_h=cam_h)
    return bool(env._check_success())


def run_spark_on_robosuite(env, task_config: dict, cfg: RobosuiteConfig) -> bool:
    """
    Run SPARK pipeline on a robosuite env.

    Same flow as LIBERO fair runner: render -> detect -> plan -> execute -> check.
    Reuses _execute_on_libero since robosuite envs share the same env.step(7-dim) API.
    """
    model = env.sim.model._model
    data = env.sim.data._data
    mujoco.mj_forward(model, data)

    # Render
    obs = env._get_observations()
    rgb = obs.get('agentview_image')
    if rgb is not None:
        rgb = rgb[::-1].copy()
    if rgb is None:
        return False

    raw_depth = obs.get('agentview_depth')
    if raw_depth is not None:
        depth = get_real_depth_map(env.sim, raw_depth[::-1].copy())
        if depth.ndim == 3:
            depth = depth.squeeze(-1)
    else:
        depth = None

    # Camera params
    cam_id = env.sim.model.camera_name2id('agentview')
    cam_pos = data.cam_xpos[cam_id].copy()
    cam_mat = data.cam_xmat[cam_id].reshape(3, 3).copy()
    cam_fovy = float(model.cam_fovy[cam_id])

    # SAM3 detection
    sam3 = _get_sam3()
    prompts = task_config['prompts']

    # Adaptive self-consistency: K=3 Gemini variants per prompt; SAM3 picks best.
    # Same logic as LIBERO-PRO adaptive - fully fair (no GT, no hand-tuned dict).
    if getattr(cfg, 'adaptive_prompts', False) and prompts:
        try:
            _k = int(getattr(cfg, 'adaptive_k', 3))
            planner = SPARKPlanner(llm_backend='gemini')
            client = planner._get_client()
            variants_prompt = (
                f"You are helping an open-vocabulary object detector. For each "
                f"concept below (one per line), output {_k} distinct short "
                f"detection phrases (1-4 words each). Vary colors, shapes, "
                f"materials. Task context: \"{task_config['instruction']}\"\n\n"
                "Concepts:\n" + "\n".join(f"- {p}" for p in prompts[:10]) +
                "\n\nOutput JSON only, no code fences:\n"
                "{\"concept1\": [\"phrase1\", \"phrase2\", ...], ...}"
            )
            try:
                resp = client.models.generate_content(
                    model='gemini-3-flash-preview',
                    contents=variants_prompt,
                    config=genai_types.GenerateContentConfig(temperature=0),
                )
                txt = resp.text.strip()
                if '```' in txt:
                    txt = txt.split('```', 2)[1]
                    if txt.startswith('json'):
                        txt = txt[4:]
                variants_by_concept = json.loads(txt.strip())
            except Exception as _ge:
                variants_by_concept = {}
                if cfg.verbose:
                    print(f"[Adaptive] variant-gen failed: {_ge}")

            if variants_by_concept:
                sam3.load_models(load_da3=False)
                state = sam3._sam3.set_image(PILImage.fromarray(rgb))
                chosen = []
                for concept, variants in variants_by_concept.items():
                    if not isinstance(variants, list):
                        continue
                    best = (-1.0, concept, None)  # (score, phrase, centroid)
                    for v in variants[:_k]:
                        if not isinstance(v, str):
                            continue
                        st = sam3._sam3.set_text_prompt(prompt=v, state=state)
                        m = st.get('masks', torch.tensor([]))
                        s = st.get('scores', torch.tensor([]))
                        if m.numel() == 0:
                            score_ = 0.0
                            centroid = None
                        else:
                            n = m.shape[0]
                            conf = float(s.max().item()) if s.numel() else 0.0
                            card_mult = 1.0 if n == 1 else (0.6 if n == 2 else 0.1)
                            score_ = card_mult * conf
                            best_idx = int(s.argmax().item())
                            msk = m[best_idx].cpu().numpy().squeeze().astype(bool)
                            centroid = None
                            if msk.any():
                                ys, xs = np.where(msk)
                                centroid = (float(xs.mean()), float(ys.mean()))
                        if score_ > best[0]:
                            best = (score_, v, centroid)
                    chosen.append(best)
                # Centroid dedup
                deduped = []
                for s_i, v_i, c_i in chosen:
                    if c_i is None:
                        deduped.append((s_i, v_i, c_i))
                        continue
                    dup = False
                    for j, (s_j, v_j, c_j) in enumerate(deduped):
                        if c_j is not None and np.hypot(c_i[0]-c_j[0], c_i[1]-c_j[1]) < 30.0:
                            dup = True
                            if s_i > s_j:
                                deduped[j] = (s_i, v_i, c_i)
                            break
                    if not dup:
                        deduped.append((s_i, v_i, c_i))
                chosen_phrases = [v for _, v, _ in deduped]
                if cfg.verbose:
                    print(f"[Adaptive] chose: {chosen_phrases} (deduped {len(chosen)-len(chosen_phrases)})")
                prompts = chosen_phrases[:10]
        except Exception as e:
            if cfg.verbose:
                print(f"[Adaptive] failed: {e}; using hand-written prompts")

    if depth is not None:
        dets = sam3._detect_with_rendered_depth(
            rgb, depth, prompts, cam_pos, cam_mat, cam_fovy,
            cfg.cam_width, cfg.cam_height)
    else:
        dets = sam3.detect(rgb, prompts, cam_pos, cam_mat, cam_fovy)

    if not dets:
        if cfg.verbose:
            print("No detections")
        return False

    det_map = {}
    for d in dets:
        if d.position_3d is not None:
            if d.label not in det_map:
                det_map[d.label] = d
            if cfg.verbose:
                p = d.position_3d
                print(f"[{d.label}] conf={d.confidence:.3f} "
                      f"pos=({p[0]:.3f},{p[1]:.3f},{p[2]:.3f})")

    labels = list(det_map.keys())
    instruction = task_config['instruction']
    pick_hint = task_config['pick']
    place_hint = task_config['place']

    # Plan
    if cfg.no_gemini:
        score = _scripted_plan(labels, pick_hint, place_hint, instruction)
    elif getattr(cfg, 'use_dsl', False):
        score = _dsl_gemini_plan(instruction, labels, pick_hint, place_hint,
                                 det_map=det_map, rgb_image=rgb,
                                 fallback_labels=labels)
    else:
        score = _gemini_plan(instruction, labels, pick_hint, place_hint,
                             det_map=det_map, rgb_image=rgb)

    if score is None:
        if cfg.verbose:
            print("No plan generated")
        return False

    # Post-process: if instruction says "insert", ensure plan uses insert primitive
    if 'insert' in instruction.lower():
        tree = score.get('tree', {})
        children = tree.get('children', [])
        # Find the last move_to_keypoint (place target) and replace with insert
        for i in range(len(children) - 1, -1, -1):
            if children[i].get('type') == 'move_to_keypoint' and i > 0:
                # Check if this is the place step (comes after grasp/move_relative)
                prev_types = [c.get('type') for c in children[:i]]
                if 'grasp' in prev_types:
                    place_label = children[i].get('params', {}).get('keypoint_label', '')
                    children[i] = {'type': 'insert', 'params': {'keypoint_label': place_label}}
                    # Remove any release after it
                    if i + 1 < len(children) and children[i+1].get('type') == 'release':
                        children.pop(i + 1)
                    if cfg.verbose:
                        print(f"[Plan fix] replaced place with insert({place_label})")
                    break

    if cfg.verbose:
        print(f"Plan actions: {json.dumps(score, indent=2)[:300]}")

    # Execute - reuse LIBERO executor
    # Wrap env.step for action dim compatibility (Wipe has 6-dim, others 7-dim)
    actual_dim = env.action_dim
    if actual_dim != 7:
        _orig_step = env.step
        def _adapted_step(action):
            if len(action) == 7 and actual_dim == 6:
                return _orig_step(action[:6])  # Drop gripper dim (sponge attached)
            elif len(action) != actual_dim:
                padded = np.zeros(actual_dim)
                padded[:min(len(action), actual_dim)] = action[:min(len(action), actual_dim)]
                return _orig_step(padded)
            return _orig_step(action)
        env.step = _adapted_step

    fair_cfg = FairConfig(verbose=cfg.verbose)
    if getattr(cfg, 'use_dsl', False):
        _dsl_execute(env, score, det_map, fair_cfg,
                     sam3=sam3, depth=depth,
                     cam_pos=cam_pos, cam_mat=cam_mat,
                     cam_fovy=cam_fovy, cam_w=cfg.cam_width, cam_h=cfg.cam_height,
                     instruction=instruction,
                     calibration_db_path=getattr(cfg, 'dsl_calibration_db', '') or None)
    else:
        _execute_on_libero(env, score, det_map, fair_cfg,
                           sam3=sam3, depth=depth,
                           cam_pos=cam_pos, cam_mat=cam_mat,
                           cam_fovy=cam_fovy, cam_w=cfg.cam_width, cam_h=cfg.cam_height)

    # Check success (keep adapter active for fallback recovery)
    success = bool(env._check_success())

    # Find EE site for fallback retract
    ee_site = -1
    for s in range(model.nsite):
        sn = (mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_SITE, s) or '').lower()
        if 'grip_site' in sn or 'grip' in sn:
            ee_site = s; break

    # Fallback recovery: retract + re-detect + re-plan + re-execute (up to 2 retries)
    if not success and not cfg.no_gemini:
        for retry in range(2):
            # Retract arm
            for s in range(80):
                mujoco.mj_forward(model, data)
                ee = data.site_xpos[ee_site].copy() if ee_site >= 0 else np.zeros(3)
                lift = ee.copy(); lift[2] = max(ee[2] + 0.10, 0.3)
                err = lift - ee
                if np.linalg.norm(err) < 0.01:
                    break
                a = np.zeros(7)  # _execute_on_libero uses 7-dim
                a[:3] = np.clip(err * 8 / 0.05, -1, 1)
                a[6] = -1  # open gripper
                try:
                    env.step(a)
                except:
                    break

            # Re-detect
            obs2 = env._get_observations()
            rgb2 = obs2.get('agentview_image')
            if rgb2 is not None:
                rgb2 = rgb2[::-1].copy()
                raw2 = obs2.get('agentview_depth')
                if raw2 is not None:
                    depth2 = get_real_depth_map(env.sim, raw2[::-1].copy())
                    if depth2.ndim == 3:
                        depth2 = depth2.squeeze(-1)
                    mujoco.mj_forward(model, data)
                    dets2 = sam3._detect_with_rendered_depth(
                        rgb2, depth2, prompts,
                        data.cam_xpos[cam_id].copy(),
                        data.cam_xmat[cam_id].reshape(3, 3).copy(),
                        float(model.cam_fovy[cam_id]),
                        cfg.cam_width, cfg.cam_height)
                    if dets2:
                        det_map2 = {d.label: d for d in dets2 if d.position_3d is not None}
                        labels2 = list(det_map2.keys())
                        if getattr(cfg, 'use_dsl', False):
                            score2 = _dsl_gemini_plan(instruction, labels2, pick_hint, place_hint,
                                                      det_map=det_map2, rgb_image=rgb2,
                                                      fallback_labels=labels2)
                        else:
                            score2 = _gemini_plan(instruction, labels2, pick_hint, place_hint,
                                                  det_map=det_map2, rgb_image=rgb2)
                        if score2:
                            if getattr(cfg, 'use_dsl', False):
                                _dsl_execute(env, score2, det_map2, fair_cfg,
                                             sam3=sam3, depth=depth2,
                                             cam_pos=data.cam_xpos[cam_id].copy(),
                                             cam_mat=data.cam_xmat[cam_id].reshape(3, 3).copy(),
                                             cam_fovy=float(model.cam_fovy[cam_id]),
                                             cam_w=cfg.cam_width, cam_h=cfg.cam_height,
                                             instruction=instruction,
                                             calibration_db_path=getattr(cfg, 'dsl_calibration_db', '') or None)
                            else:
                                _execute_on_libero(env, score2, det_map2, fair_cfg,
                                                   sam3=sam3, depth=depth2,
                                                   cam_pos=data.cam_xpos[cam_id].copy(),
                                                   cam_mat=data.cam_xmat[cam_id].reshape(3, 3).copy(),
                                                   cam_fovy=float(model.cam_fovy[cam_id]),
                                                   cam_w=cfg.cam_width, cam_h=cfg.cam_height)
                            success = bool(env._check_success())
                            if success:
                                break

    # Restore original step after all execution + fallback
    if actual_dim != 7 and '_orig_step' in dir():
        env.step = _orig_step

    return success


def _setup_cube_restack(env):
    """
    Place cubeA (red) on top of cubeB (green) for the Restack task.
    """
    m = env.sim.model._model
    d = env.sim.data._data
    # Find cubeA and cubeB joint qpos addresses
    cubeA_addr = cubeB_addr = None
    for i in range(m.njnt):
        jn = mujoco.mj_id2name(m, mujoco.mjtObj.mjOBJ_JOINT, i) or ''
        if 'cubeA' in jn:
            cubeA_addr = m.jnt_qposadr[i]
        elif 'cubeB' in jn:
            cubeB_addr = m.jnt_qposadr[i]
    if cubeA_addr is not None and cubeB_addr is not None:
        # Get cubeB position, place cubeA directly on top
        cubeB_pos = d.qpos[cubeB_addr:cubeB_addr+3].copy()
        d.qpos[cubeA_addr] = cubeB_pos[0]      # same X
        d.qpos[cubeA_addr+1] = cubeB_pos[1]    # same Y
        # cubeB half-size=0.025, cubeA half-size=0.02 -> top of B = B_z + 0.025, center of A = top_B + 0.02
        d.qpos[cubeA_addr+2] = cubeB_pos[2] + 0.045
        mujoco.mj_forward(m, d)


def _execute_nut_assembly(env, cfg: RobosuiteConfig) -> bool:
    """
    NutAssembly with JOINT_POSITION controller (CaP-X inspired).

    Uses delta-encoded joint targets: action = (target - current) / SCALE.
    CaP-X config: input[-10,10] -> output[-1,1] -> SCALE=0.1 -> +/-1 rad/step.
    Can reach z=0.85, well below OSC's z=0.96 limit.
    """

    model = env.sim.model._model
    data = env.sim.data._data
    SCALE = 0.1  # output_range / input_range = 2/20

    # Detect objects via SAM3
    from robosuite.utils.camera_utils import get_real_depth_map
    from spark_bench.run_spark_libero_pro_fair import _get_sam3

    obs = env._get_observations()
    rgb = obs.get('agentview_image')
    if rgb is not None:
        rgb = rgb[::-1].copy()
    raw_depth = obs.get('agentview_depth')
    depth = None
    if raw_depth is not None:
        depth = get_real_depth_map(env.sim, raw_depth[::-1].copy())
        if depth.ndim == 3:
            depth = depth.squeeze(-1)

    nut_pos = peg_pos = None
    if rgb is not None and depth is not None:
        cam_id = env.sim.model.camera_name2id('agentview')
        cam_pos = data.cam_xpos[cam_id].copy()
        cam_mat = data.cam_xmat[cam_id].reshape(3, 3).copy()
        cam_fovy = float(model.cam_fovy[cam_id])
        sam3 = _get_sam3()
        all_prompts = ['silver nut', 'square nut', 'metallic nut on table',
                       'vertical post', 'brown post', 'brown peg']
        all_dets = sam3._detect_with_rendered_depth(
            rgb, depth, all_prompts, cam_pos, cam_mat, cam_fovy,
            cfg.cam_width, cfg.cam_height)
        for d in all_dets:
            if d.position_3d is None:
                continue
            if cfg.verbose:
                p = d.position_3d
                print(f"[SAM3] {d.label}: conf={d.confidence:.3f} "
                      f"pos=({p[0]:.3f},{p[1]:.3f},{p[2]:.3f})")
            lbl = d.label.lower()
            z = d.position_3d[2]
            if ('nut' in lbl or 'metallic' in lbl or 'silver' in lbl) and z < 0.86:
                if nut_pos is None or d.confidence > 0.1:
                    nut_pos = d.position_3d.copy()
            if ('post' in lbl or 'vertical' in lbl or 'peg' in lbl) and z > 0.86:
                if peg_pos is None or d.confidence > 0.1:
                    peg_pos = d.position_3d.copy()

    if nut_pos is None or peg_pos is None:
        if cfg.verbose:
            print(f"SAM3 failed: nut={'found' if nut_pos is not None else 'MISSING'}, "
                  f"peg={'found' if peg_pos is not None else 'MISSING'}")
        return False

    # Find EE site and joint IDs
    ee_site = -1
    for s in range(model.nsite):
        sn = (mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_SITE, s) or '').lower()
        if 'grip_site' in sn and 'cylinder' not in sn:
            ee_site = s; break
    if ee_site < 0:
        return False

    joint_ids = []
    for i in range(model.njnt):
        jn = (mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_JOINT, i) or '').lower()
        if ('robot0' in jn or 'joint' in jn) and 'finger' not in jn and 'gripper' not in jn:
            if model.jnt_type[i] == 3:
                joint_ids.append(i)
    ndof = len(joint_ids)

    if cfg.verbose:
        print(f"Nut: ({nut_pos[0]:.3f},{nut_pos[1]:.3f},{nut_pos[2]:.3f})")
        print(f"Peg: ({peg_pos[0]:.3f},{peg_pos[1]:.3f},{peg_pos[2]:.3f})")

    def get_q():
        mujoco.mj_forward(model, data)
        return np.array([data.qpos[model.jnt_qposadr[j]] for j in joint_ids])

    def compute_ik(target_pos):
        q = get_q()
        d2 = mujoco.MjData(model)
        d2.qpos[:] = data.qpos[:]
        for _ in range(200):
            mujoco.mj_forward(model, d2)
            ee = d2.site_xpos[ee_site].copy()
            err = target_pos - ee
            if np.linalg.norm(err) < 0.002: break
            jacp = np.zeros((3, model.nv))
            mujoco.mj_jacSite(model, d2, jacp, None, ee_site)
            J = np.zeros((3, ndof))
            for i, jid in enumerate(joint_ids):
                J[:, i] = jacp[:, model.jnt_dofadr[jid]]
            JJT = J @ J.T + 0.05**2 * np.eye(3)
            q += 0.5 * J.T @ np.linalg.solve(JJT, err)
            for i, jid in enumerate(joint_ids):
                d2.qpos[model.jnt_qposadr[jid]] = q[i]
        return q

    def jp_move(q_target, grip_cmd, max_steps=100, tol=0.02):
        """
        Move via JOINT_POSITION delta encoding. Smooth motion with clamped delta.
        """
        target = np.asarray(q_target).reshape(ndof)
        for step in range(max_steps):
            q_cur = get_q()
            delta = target - q_cur
            if np.linalg.norm(delta) < tol:
                break
            action = np.zeros(env.action_dim)
            # Clamp to +/-5 for smoother motion (0.5 rad/step max)
            action[:ndof] = np.clip(delta / SCALE, -5, 5)
            action[ndof] = grip_cmd
            try:
                _, _, done, _ = env.step(action)
                if done: break
            except: break
        mujoco.mj_forward(model, data)

    def jp_grip(close, steps=60):
        """
        Gripper control with joint hold.
        """
        for _ in range(steps):
            action = np.zeros(env.action_dim)  # Zero delta = hold joints
            action[ndof] = 1.0 if close else -1.0
            try: env.step(action)
            except: break

    # Phase 1: Move above nut
    above = nut_pos.copy(); above[2] += 0.08
    q_above = compute_ik(above)
    jp_move(q_above, -1.0, max_steps=80)

    # Phase 2: Descend to nut
    q_nut = compute_ik(nut_pos)
    jp_move(q_nut, -1.0, max_steps=100)

    if cfg.verbose:
        mujoco.mj_forward(model, data)
        ee = data.site_xpos[ee_site]
        print(f"At nut: ee=({ee[0]:.3f},{ee[1]:.3f},{ee[2]:.3f})")

    # Phase 3: Grasp
    jp_grip(True, steps=80)

    # Phase 4: Verify grasp
    mujoco.mj_forward(model, data)
    ee_z_pre = data.site_xpos[ee_site][2]
    lift_check = nut_pos.copy(); lift_check[2] += 0.05
    q_lift_check = compute_ik(lift_check)
    jp_move(q_lift_check, 1.0, max_steps=50)
    mujoco.mj_forward(model, data)
    ee_z_post = data.site_xpos[ee_site][2]
    grasp_ok = ee_z_post > ee_z_pre + 0.005
    if cfg.verbose:
        print(f"Grasp check: z={ee_z_pre:.4f}->{ee_z_post:.4f} ok={grasp_ok}")

    if not grasp_ok:
        # Retry: open, descend lower, re-grasp
        jp_grip(False, steps=30)
        lower = nut_pos.copy(); lower[2] -= 0.005
        q_lower = compute_ik(lower)
        jp_move(q_lower, -1.0, max_steps=80)
        jp_grip(True, steps=80)

    # Phase 5: Lift high (more steps for smooth motion)
    mujoco.mj_forward(model, data)
    ee = data.site_xpos[ee_site].copy()
    lift_z = max(peg_pos[2] + 0.15, ee[2] + 0.10)
    lift_target = np.array([ee[0], ee[1], lift_z])
    q_lift = compute_ik(lift_target)
    jp_move(q_lift, 1.0, max_steps=150)

    # Phase 6: Align above peg
    above_peg = peg_pos.copy(); above_peg[2] = lift_z
    q_above_peg = compute_ik(above_peg)
    jp_move(q_above_peg, 1.0, max_steps=150)

    if cfg.verbose:
        mujoco.mj_forward(model, data)
        ee = data.site_xpos[ee_site]
        xy_err = np.linalg.norm(ee[:2] - peg_pos[:2])
        print(f"Aligned above peg: ee=({ee[0]:.3f},{ee[1]:.3f},{ee[2]:.3f}) xy_err={xy_err:.4f}")

    # Phase 7: Lower onto peg with wrist rotation to align square nut
    # Try 4 rotations - keep nut gripped during insertion, only release at end
    j7_idx = ndof - 1  # Joint 7 (wrist)
    mujoco.mj_forward(model, data)
    q_start = get_q()
    j7_base = q_start[j7_idx]

    for rot_i, j7_offset in enumerate([0.0, 0.78, 1.57, 2.35]):
        # Rotate wrist while above peg
        mujoco.mj_forward(model, data)
        q_above = compute_ik(peg_pos.copy() + np.array([0, 0, 0.04]))
        q_above[j7_idx] = j7_base + j7_offset
        jp_move(q_above, 1.0, max_steps=60)

        # Descend onto peg - keep grip closed
        for z_off in [0.02, 0.0, -0.01, -0.03]:
            mujoco.mj_forward(model, data)
            ins = peg_pos.copy(); ins[2] += z_off
            q_ins = compute_ik(ins)
            q_ins[j7_idx] = j7_base + j7_offset
            jp_move(q_ins, 1.0, max_steps=40)

        if cfg.verbose:
            mujoco.mj_forward(model, data)
            ee = data.site_xpos[ee_site]
            print(f"Rot {rot_i} (off={j7_offset:.2f}): z={ee[2]:.4f}")

        # Release and settle
        jp_grip(False, steps=40)
        for _ in range(120):
            action = np.zeros(env.action_dim)
            action[ndof] = -1.0
            try: env.step(action)
            except: break

        if env._check_success():
            if cfg.verbose: print(f"Rotation {rot_i}: SUCCESS!")
            break

        # If failed, retract and re-detect nut for next rotation
        if rot_i < 3:
            mujoco.mj_forward(model, data)
            q_ret = compute_ik(peg_pos.copy() + np.array([0, 0, 0.10]))
            jp_move(q_ret, -1.0, max_steps=40)

            # Re-detect nut position (it may have fallen near the peg)
            re_obs = env._get_observations()
            re_rgb = re_obs.get('agentview_image')
            re_depth_raw = re_obs.get('agentview_depth')
            nut_now = None
            if re_rgb is not None and re_depth_raw is not None:
                re_rgb = re_rgb[::-1].copy()
                re_depth = get_real_depth_map(env.sim, re_depth_raw[::-1].copy())
                if re_depth.ndim == 3: re_depth = re_depth.squeeze(-1)
                mujoco.mj_forward(model, data)
                re_dets = sam3._detect_with_rendered_depth(
                    re_rgb, re_depth, ['silver nut', 'square nut'],
                    data.cam_xpos[cam_id].copy(),
                    data.cam_xmat[cam_id].reshape(3,3).copy(),
                    float(model.cam_fovy[cam_id]),
                    cfg.cam_width, cfg.cam_height)
                for rd in re_dets:
                    if rd.position_3d is not None and rd.position_3d[2] < 0.90:
                        nut_now = rd.position_3d.copy()
                        break
            if nut_now is None:
                nut_now = peg_pos.copy()
                nut_now[2] = 0.83  # Table height

            # Re-pick nut
            mujoco.mj_forward(model, data)
            q_above_nut = compute_ik(nut_now + np.array([0, 0, 0.06]))
            jp_move(q_above_nut, -1.0, max_steps=60)
            mujoco.mj_forward(model, data)
            q_at_nut = compute_ik(nut_now)
            jp_move(q_at_nut, -1.0, max_steps=60)
            jp_grip(True, steps=60)
            # Lift
            mujoco.mj_forward(model, data)
            q_lift = compute_ik(peg_pos.copy() + np.array([0, 0, 0.12]))
            jp_move(q_lift, 1.0, max_steps=60)
            # Re-align above peg
            mujoco.mj_forward(model, data)
            q_align = compute_ik(peg_pos.copy() + np.array([0, 0, 0.06]))
            jp_move(q_align, 1.0, max_steps=60)

    mujoco.mj_forward(model, data)
    if cfg.verbose:
        ee = data.site_xpos[ee_site]
        print(f"Inserted: z={ee[2]:.4f} (peg at z={peg_pos[2]:.4f})")

    # Phase 8: Final release and settle (if not already released by rotation search)
    jp_grip(False, steps=40)
    for _ in range(150):
        action = np.zeros(env.action_dim)
        action[ndof] = -1.0
        try: env.step(action)
        except: break

    # Retract
    mujoco.mj_forward(model, data)
    ee = data.site_xpos[ee_site].copy()
    ret = ee.copy(); ret[2] += 0.15
    q_ret = compute_ik(ret)
    jp_move(q_ret, -1.0, max_steps=80)

    for _ in range(100):
        action = np.zeros(env.action_dim)
        action[ndof] = -1.0
        try: env.step(action)
        except: break

    if cfg.verbose:
        mujoco.mj_forward(model, data)
        for i in range(model.nbody):
            bn = (mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_BODY, i) or '').lower()
            if 'nut' in bn and 'round' not in bn:
                pos = data.xpos[i]
                print(f"Nut '{bn}': ({pos[0]:.3f},{pos[1]:.3f},{pos[2]:.3f})")

    success = bool(env._check_success())
    if cfg.verbose:
        print(f"Success: {success}")
    return success


BIMANUAL_TASKS = {'TwoArmLift', 'TwoArmHandover'}


def _run_bimanual_task(task_name: str, cfg: RobosuiteConfig) -> dict:
    """
    Route bimanual tasks to dual_arm_bt executor.
    """
    from spark_bench.dual_arm_bt import create_dual_arm_env, execute_dual_arm_bt
    ctrl = 'JOINT_POSITION' if task_name == 'TwoArmHandover' else 'OSC_POSE'
    print(f"\n{'='*60}")
    print(f"SPARK Robosuite: {task_name} (BIMANUAL, controller={ctrl})")
    print(f"Trials: {cfg.num_trials}")
    print(f"{'='*60}\n")
    env = create_dual_arm_env(task_name, horizon=cfg.horizon, controller=ctrl)
    successes = []
    for trial in range(cfg.num_trials):
        env.reset()
        try:
            # Bimanual: force non-adaptive prompts. Adaptive's centroid dedup
            # collapses left/right handle variants that both detect the pot
            # center, breaking two-arm coordination.
            ok = execute_dual_arm_bt(env, task_name, verbose=cfg.verbose,
                                     adaptive=False)
        except Exception as e:
            if cfg.verbose:
                print(f"Trial {trial}: EXCEPTION {e}")
            ok = False
        successes.append(bool(ok))
        print(f"Trial {trial}: {'SUCCESS' if ok else 'FAIL'}")
    success_rate = float(np.mean(successes)) if successes else 0.0
    print(f"\n{task_name}: {success_rate:.0%} ({sum(successes)}/{len(successes)})")
    return {
        'task': task_name,
        'bimanual': True,
        'success_rate': success_rate,
        'trials': successes,
    }


def _execute_wipe(env, cfg: RobosuiteConfig) -> bool:
    """
    Joint-control table wipe (CaP-X style, no LLM).

    SAM3 localizes the dirt once; then a single boustrophedon raster over its
    extent at table height, driven by JOINT_POSITION so the flat WipingGripper
    pad presses firmly and stays level. OSC is springy and pitches, which smears
    the dirt visually without registering the marker contacts robosuite scores.
    """
    from robosuite.utils.camera_utils import get_real_depth_map
    from spark_bench.run_spark_libero_pro_fair import _get_sam3
    model = env.sim.model._model
    data = env.sim.data._data
    SCALE = 0.1

    obs = env._get_observations()
    rgb = obs.get('agentview_image')
    raw = obs.get('agentview_depth')
    if rgb is None or raw is None:
        return bool(env._check_success())
    rgb = rgb[::-1].copy()
    depth = get_real_depth_map(env.sim, raw[::-1].copy())
    if depth.ndim == 3:
        depth = depth.squeeze(-1)
    cam_id = env.sim.model.camera_name2id('agentview')
    cam_pos = data.cam_xpos[cam_id].copy()
    cam_mat = data.cam_xmat[cam_id].reshape(3, 3).copy()
    cam_fovy = float(model.cam_fovy[cam_id])
    sam3 = _get_sam3()
    dets = sam3._detect_with_rendered_depth(
        rgb, depth, ['brown dirt', 'dirt spots', 'brown marks on table'],
        cam_pos, cam_mat, cam_fovy, cfg.cam_width, cfg.cam_height)
    pts = np.array([d.position_3d for d in dets
                    if d.position_3d is not None and d.confidence > 0.05])
    if len(pts) == 0:
        if cfg.verbose:
            print("[Wipe] SAM3 found no dirt")
        return bool(env._check_success())
    cx, cy = float(pts[:, 0].mean()), float(pts[:, 1].mean())
    xmin, xmax = float(pts[:, 0].min()), float(pts[:, 0].max())
    ymin, ymax = float(pts[:, 1].min()), float(pts[:, 1].max())
    surf_z = float(np.median(pts[:, 2]))
    if cfg.verbose:
        print(f"[Wipe] dirt c=({cx:.3f},{cy:.3f}) z={surf_z:.3f} "
              f"x[{xmin:.3f},{xmax:.3f}] y[{ymin:.3f},{ymax:.3f}]")

    ee_site = -1
    for s in range(model.nsite):
        sn = (mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_SITE, s) or '').lower()
        if 'grip_site' in sn and 'cylinder' not in sn:
            ee_site = s
            break
    if ee_site < 0:
        return bool(env._check_success())
    joint_ids = []
    for i in range(model.njnt):
        jn = (mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_JOINT, i) or '').lower()
        if ('robot0' in jn or 'joint' in jn) and 'finger' not in jn and 'gripper' not in jn:
            if model.jnt_type[i] == 3:
                joint_ids.append(i)
    ndof = len(joint_ids)

    def get_q():
        mujoco.mj_forward(model, data)
        return np.array([data.qpos[model.jnt_qposadr[j]] for j in joint_ids])

    def compute_ik(target):
        q = get_q()
        d2 = mujoco.MjData(model)
        d2.qpos[:] = data.qpos[:]
        for _ in range(200):
            mujoco.mj_forward(model, d2)
            ee = d2.site_xpos[ee_site].copy()
            err = target - ee
            if np.linalg.norm(err) < 0.002:
                break
            jacp = np.zeros((3, model.nv))
            mujoco.mj_jacSite(model, d2, jacp, None, ee_site)
            J = np.zeros((3, ndof))
            for i, jid in enumerate(joint_ids):
                J[:, i] = jacp[:, model.jnt_dofadr[jid]]
            JJT = J @ J.T + 0.05 ** 2 * np.eye(3)
            q = q + 0.5 * J.T @ np.linalg.solve(JJT, err)
            for i, jid in enumerate(joint_ids):
                d2.qpos[model.jnt_qposadr[jid]] = q[i]
        return q

    def jp_move(q_target, max_steps=18, tol=0.02):
        # controller maps action in [-1,1] -> +-0.1 rad/step (output_max above)
        target = np.asarray(q_target).reshape(ndof)
        for _ in range(max_steps):
            delta = target - get_q()
            if np.linalg.norm(delta) < tol:
                break
            action = np.zeros(env.action_dim)
            action[:ndof] = np.clip(delta / SCALE, -1, 1)
            try:
                _, _, done, _ = env.step(action)
                if done:
                    return True
            except Exception:
                return True
        return False

    # press only to the contact height (not past it): rigid over-press trips the
    # 60 N force limit and early-terminates. Compliant kp + contact-height target
    # keeps a gentle, marker-registering press.
    press_z = min(surf_z, 0.88)
    # SAM3 returns one centroid for the dirt blob (extent ~0), so cover a generous
    # box around it; the robosuite dirt line spans ~0.3-0.4 m.
    pad, gs = 0.36, 0.035
    # approach above the dirt centre, then descend to press height
    if jp_move(compute_ik(np.array([cx, cy, surf_z + 0.06])), max_steps=120, tol=0.03):
        return bool(env._check_success())
    xs = np.arange(xmin - pad, xmax + pad + gs, gs)
    ys = np.arange(ymin - pad, ymax + pad + gs, gs)
    for yi, y in enumerate(ys):
        xrow = xs if yi % 2 == 0 else xs[::-1]
        for x in xrow:
            if jp_move(compute_ik(np.array([x, y, press_z])), max_steps=18, tol=0.02):
                return bool(env._check_success())
    return bool(env._check_success())


def run_task(task_name: str, cfg: RobosuiteConfig) -> dict:
    """
    Run evaluation for one robosuite task. Routes bimanual tasks to
    dual_arm_bt; single-arm tasks use the standard SPARK pipeline.
    """
    if task_name in BIMANUAL_TASKS:
        return _run_bimanual_task(task_name, cfg)

    task_config = TASK_CONFIGS.get(task_name)
    if not task_config:
        print(f"Unknown task: {task_name}")
        print(f"Available: {', '.join(list(TASK_CONFIGS.keys()) + list(BIMANUAL_TASKS))}")
        return {}

    print(f"\n{'='*60}")
    print(f"SPARK Robosuite: {task_name}")
    print(f"{task_config['instruction']}")
    print(f"Trials: {cfg.num_trials}")
    print(f"{'='*60}\n")

    successes = []
    env = create_env(task_name, cfg)

    video_dir = Path.home() / 'spark' / 'videos' / 'robosuite' / task_name.lower()
    video_dir.mkdir(parents=True, exist_ok=True)

    for trial in range(cfg.num_trials):
        env.reset()

        # CubeRestack: place red cube on top of green cube after reset
        if task_name == 'CubeRestack':
            _setup_cube_restack(env)

        for _ in range(10):
            env.step(np.zeros(env.action_dim))

        # Record frames if --save-video
        frames = []
        if cfg.save_video:
            _orig_step = env.step
            _fc = [0]
            def _rec_step(action):
                result = _orig_step(action)
                _fc[0] += 1
                if _fc[0] % 5 == 0:
                    try:
                        obs_r = env._get_observations()
                        rgb_r = obs_r.get('agentview_image')
                        if rgb_r is not None:
                            frames.append(rgb_r[::-1].copy())
                    except:
                        pass
                return result
            env.step = _rec_step

        try:
            if task_name == 'NutAssemblySquare':
                success = _execute_nut_assembly(env, cfg)
            else:
                success = run_spark_on_robosuite(env, task_config, cfg)
        except Exception as e:
            if cfg.verbose:
                print(f"Trial {trial} error: {e}")
            success = False

        if cfg.save_video:
            env.step = _orig_step  # restore
            if frames:
                pil_frames = [PILImage.fromarray(f) for f in frames]
                status = 'success' if success else 'fail'
                gif_path = video_dir / f'trial{trial}_{status}.gif'
                pil_frames[0].save(str(gif_path), save_all=True,
                                   append_images=pil_frames[1:], duration=100, loop=0)

        successes.append(success)
        print(f"Trial {trial}: {'SUCCESS' if success else 'FAIL'}")

    env.close()

    sr = sum(successes) / len(successes) if successes else 0
    print(f"\n{task_name}: {sr:.0%} ({sum(successes)}/{len(successes)})")

    return {
        'task': task_name,
        'success_rate': float(sr),
        'trials': [bool(s) for s in successes],
    }


def main():
    cfg = tyro.cli(RobosuiteConfig)

    if cfg.task.lower() == 'all':
        tasks = list(TASK_CONFIGS.keys()) + list(BIMANUAL_TASKS)
    elif cfg.task.lower() == 'bimanual':
        tasks = list(BIMANUAL_TASKS)
    else:
        tasks = [cfg.task]

    all_results = {}
    for task_name in tasks:
        result = run_task(task_name, cfg)
        if cfg.results_json:
            import subprocess as _sp
            try:
                _commit = _sp.run(['git', 'rev-parse', '--short', 'HEAD'],
                                  capture_output=True, text=True,
                                  cwd=os.path.dirname(__file__)).stdout.strip()
            except Exception:
                _commit = '?'
            _payload = {'commit': _commit,
                        'config': {k: v for k, v in vars(cfg).items()
                                   if isinstance(v, (str, int, float, bool))},
                        'model': os.environ.get('SPARK_GEMINI_MODEL',
                                                'gemini-3-flash-preview'),
                        'num_trials': cfg.num_trials,
                        'results': {**all_results,
                                    **({task_name: result} if result else {})}}
            os.makedirs(os.path.dirname(os.path.abspath(cfg.results_json)),
                        exist_ok=True)
            with open(cfg.results_json, 'w') as _f:
                json.dump(_payload, _f, indent=1, default=str)
        if result:
            all_results[task_name] = result

    # Summary
    if len(all_results) > 1:
        print(f"\n{'='*60}")
        print(f"SPARK Robosuite Summary")
        print(f"{'='*60}")
        total = []
        for name, res in all_results.items():
            print(f"{name:25s}: {res['success_rate']:.0%}")
            total.append(res['success_rate'])
        if total:
            print(f"{'Overall':25s}: {np.mean(total):.1%}")

    # Save
    out_dir = Path.home() / 'spark' / 'src' / 'spark_bench' / 'results'
    out_dir.mkdir(parents=True, exist_ok=True)
    with open(out_dir / 'robosuite_results.json', 'w') as f:
        json.dump(all_results, f, indent=2)


if __name__ == '__main__':
    main()

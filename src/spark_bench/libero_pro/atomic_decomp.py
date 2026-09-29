"""
Atomic sub-BT decomposition for LIBERO-PRO.

Parses the BDDL ``:goal`` into a list of predicate sub-goals and asks
Gemini for a sub-BT per predicate, with full re-perception between
sub-BTs.  Decomposing the goal one predicate at a time keeps each Gemini
call focused on a single achievable relation rather than the whole task.

The orchestrator preserves the existing perception / planning / executor
plumbing - it just drives them in a loop, one predicate at a time:

    for predicate in goal_predicates:
        re-render + SAM3 detect (with predicate-scoped prompts)
        ask Gemini for a sub-BT achieving only this predicate
        execute the sub-BT
        if predicate already-satisfied or env-success: stop
        if predicate-level failure: retract + retry once

On any total failure of a predicate (after retry), execution continues to
the next predicate - LIBERO ``check_success()`` is the final arbiter.

Behavior is opt-in via ``cfg.atomic_decomp``.
"""

from __future__ import annotations

from typing import List, Optional
import time

import numpy as np
import mujoco

from spark_dsl.bddl_goal_parser import (
    Predicate,
    parse_bddl_goal,
    predicate_to_natural_language,
)

from spark_bench.libero_pro.bddl import _strip_instance, _OBJ_VISUAL_PROMPTS
from spark_bench.libero_pro.executor import execute_on_libero
from spark_bench.libero_pro.perception import detect_scene, select_prompts
from spark_bench.libero_pro.planning import (
    _get_sam3,
    _gemini_plan,
    _gemini_plan_via_dsl,
    _scripted_plan,
)


__all__ = [
    'run_atomic_decomp',
    'generate_sub_bts',
    'build_subgoal_prompts',
    'predicate_pick_place',
]


# Predicate -> pick/place hint + SAM3 prompts

def predicate_pick_place(p: Predicate) -> tuple[str, str]:
    """
    Map a Predicate to (pick_hint, place_hint) for the LLM.

    Matches the keys used by the planner's _build_hint_extras() so the
    legacy and DSL planners both accept them.
    """
    op = p.op
    args = p.args or []
    if op in ('On', 'In') and len(args) >= 2:
        return _strip_instance(args[0]), _strip_instance(args[1])
    if op in ('Open', 'Close') and args:
        # No object to pick - place hint carries the joint target.
        return 'none', _strip_instance(args[0])
    if op in ('Turnon', 'TurnOn', 'Turnoff', 'TurnOff') and args:
        return 'none', _strip_instance(args[0])
    return 'none', 'none'


def build_subgoal_prompts(p: Predicate, base_prompts: List[str]) -> List[str]:
    """
    Build a focused SAM3 prompt list for the predicate's two arguments.

    Falls back to the trial-level prompts when the BDDL-derived list ends
    up empty (e.g. an unknown predicate or args that aren't in the visual
    prompt dict).
    """
    pick, place = predicate_pick_place(p)
    prompts: List[str] = []
    seen: set[str] = set()
    for key in (pick, place):
        if key == 'none' or key in seen:
            continue
        seen.add(key)
        if key in _OBJ_VISUAL_PROMPTS:
            prompts.extend(_OBJ_VISUAL_PROMPTS[key])
        else:
            # Unknown type -> use the cleaned arg as a single prompt.
            prompts.append(key.replace('_', ' '))
    # Drawer / cabinet predicates need the handle prompt for the open primitive.
    op_low = p.op.lower()
    if op_low in ('open', 'close'):
        for h in ('drawer handle', 'cabinet handle'):
            if h not in prompts:
                prompts.append(h)
    # Stove predicates: include the knob.
    if op_low in ('turnon', 'turnoff'):
        for h in ('stove knob', 'flat stove'):
            if h not in prompts:
                prompts.append(h)
    if not prompts:
        return list(base_prompts)
    # De-dupe while preserving order, cap at 10 for SAM3 budget.
    out: List[str] = []
    seen_p: set[str] = set()
    for q in prompts + list(base_prompts):
        if q in seen_p:
            continue
        seen_p.add(q)
        out.append(q)
        if len(out) >= 10:
            break
    return out


# Sub-BT generation - wraps the existing planner with predicate framing

def _build_sub_instruction(top_instruction: str, p: Predicate,
                            idx: int, total: int) -> str:
    """
    Produce a one-line LLM-facing sub-instruction for predicate ``p``.

    The top-level instruction is included as context so the LLM sees the
    whole task; the predicate text is the *current* sub-goal it should
    plan for.

    For single-predicate goals the raw user instruction is passed through
    so the planner sees identical phrasing to the baseline.  The
    predicate render is appended only as a hint, parenthesised.
    """
    sub = predicate_to_natural_language(p)
    if total == 1:
        # Identical phrasing to the baseline planner - only difference
        # is the parenthesised predicate-level hint.
        return f"{top_instruction} (sub-goal: {sub})"
    return (f"Sub-goal {idx + 1} of {total}: {sub}. "
            f"(Full task instruction: {top_instruction}.) "
            f"Generate ONLY the steps needed to achieve this sub-goal. "
            f"Assume previous sub-goals are already done.")


def _plan_one_subbt(cfg, sub_instruction: str, det_map: dict, rgb,
                    wrist_rgb, pick_hint: str, place_hint: str) -> Optional[dict]:
    """
    Call the configured Gemini planner for a single predicate's sub-BT.
    """
    labels = list(det_map.keys())
    if cfg.no_gemini:
        return _scripted_plan(labels, pick_hint, place_hint, sub_instruction)
    planner_fn = (_gemini_plan_via_dsl
                  if getattr(cfg, 'use_dsl', False) else _gemini_plan)
    try:
        return planner_fn(
            sub_instruction, labels, pick_hint, place_hint,
            det_map=det_map, rgb_image=rgb, wrist_image=wrist_rgb,
            no_bddl_hints=getattr(cfg, 'no_bddl_hints', False))
    except Exception as e:
        if cfg.verbose:
            print(f"[Atomic] planner failed for predicate: {e}")
        return _scripted_plan(labels, pick_hint, place_hint, sub_instruction)


def generate_sub_bts(cfg, top_instruction: str,
                       predicates: List[Predicate], det_map: dict,
                       rgb, wrist_rgb) -> List[dict]:
    """
    Convenience helper: generate sub-BTs for every predicate up-front.

    NB: ``run_atomic_decomp`` does NOT use this - it interleaves planning
    with re-perception so each sub-BT sees fresh detections.  This helper
    is exposed mainly for the smoke test that prints sub-BT YAML samples.
    """
    sub_bts: List[dict] = []
    total = len(predicates)
    for i, p in enumerate(predicates):
        pick, place = predicate_pick_place(p)
        sub_inst = _build_sub_instruction(top_instruction, p, i, total)
        bt = _plan_one_subbt(cfg, sub_inst, det_map, rgb, wrist_rgb,
                              pick, place)
        if bt is not None:
            sub_bts.append(bt)
    return sub_bts


# Orchestrator - full atomic-decomp pipeline for one trial

def _find_ee_site(model):
    for s in range(model.nsite):
        sn = (mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_SITE, s) or '').lower()
        if 'grip_site' in sn:
            return s
    return -1


def _retract_arm(env, model, data, ee_site: int) -> None:
    """
    Lift the arm 10 cm before re-detecting (used between sub-BTs).
    """
    mujoco.mj_forward(model, data)
    ee = (data.site_xpos[ee_site].copy() if ee_site >= 0
          else np.array([0.0, 0.0, 0.3]))
    lift = ee.copy(); lift[2] = max(ee[2] + 0.10, 0.3)
    for _ in range(80):
        try:
            curr = env.env.robots[0].controller.ee_pos
        except Exception:
            break
        err = lift - curr
        if np.linalg.norm(err) < 0.01:
            break
        a = np.zeros(7); a[:3] = np.clip(err * 8 / 0.05, -1, 1); a[6] = -1
        try:
            env.step(a)
        except Exception:
            break


def _reperceive(env, sam3, prompts: List[str], cfg, instruction: str):
    """
    Step the env once and run a full agentview (+wrist) detection.
    """
    obs, _, _, _ = env.step(np.zeros(7))
    rgb_raw = obs.get('agentview_image')
    if rgb_raw is None:
        return None, None, None
    rgb = rgb_raw[::-1].copy()
    wrist_rgb_raw = obs.get('robot0_eye_in_hand_image')
    wrist_rgb = (wrist_rgb_raw[::-1].copy()
                  if wrist_rgb_raw is not None else None)
    # Apply prompt-selection layers (tuned / Gemini / adaptive) before SAM3.
    prompts_eff = select_prompts(env, rgb, wrist_rgb, instruction, cfg, prompts)
    det = detect_scene(env, sam3, prompts_eff, obs, cfg)
    return det, rgb, wrist_rgb


def run_atomic_decomp(env, prompts: List[str], instruction: str,
                       cfg, bddl_path: str) -> bool:
    """
    Atomic sub-BT decomposition loop.

    Returns True iff ``env.check_success()`` is True at the end.

    The flow:

    1. Parse ``bddl_path`` -> list of predicates.  If empty, fall through
       to a single sub-BT containing the full instruction.
    2. For each predicate (in declaration order):
         a. Re-perceive (agentview + wrist).
         b. Skip if env already reports success.
         c. Build a predicate-scoped SAM3 prompt list.
         d. Ask Gemini for the sub-BT.
         e. Execute.  If it fails, retract + redetect + retry once.
    3. After the last predicate, return ``env.check_success()``.
    """
    model = env.sim.model._model
    data = env.sim.data._data
    mujoco.mj_forward(model, data)

    # Early-out: goal already satisfied (rare but happens for ``Turnoff``).
    try:
        if env.check_success():
            return True
    except Exception:
        pass

    predicates = parse_bddl_goal(bddl_path) if bddl_path else []
    if not predicates:
        # Fall back to a single "predicate" that's just the raw instruction.
        predicates = [Predicate(op='_TopLevel', args=[])]

    if cfg.verbose:
        print(f"[Atomic] parsed {len(predicates)} predicates:")
        for p in predicates:
            print(f"- {p.op}({', '.join(p.args)})")

    sam3 = _get_sam3()
    ee_site = _find_ee_site(model)
    overall_success = False
    total = len(predicates)

    for i, pred in enumerate(predicates):
        # Bail early if the previous sub-BT already satisfied the whole task.
        try:
            if env.check_success():
                overall_success = True
                break
        except Exception:
            pass

        sub_prompts = build_subgoal_prompts(pred, prompts)
        sub_instruction = _build_sub_instruction(instruction, pred, i, total)

        if cfg.verbose:
            print(f"[Atomic {i + 1}/{total}] '{sub_instruction[:120]}'")
            print(f"prompts: {sub_prompts}")

        # Plan + execute, with one retry on failure
        attempt_succeeded = False
        for attempt in range(2):
            det, rgb, wrist_rgb = _reperceive(env, sam3, sub_prompts, cfg,
                                                instruction)
            if det is None or not det.det_map:
                if cfg.verbose:
                    print(f"[Atomic {i + 1}] no detections "
                          f"(attempt {attempt + 1})")
                if attempt == 0:
                    _retract_arm(env, model, data, ee_site)
                    continue
                break

            pick_h, place_h = predicate_pick_place(pred)
            t0 = time.time()
            sub_bt = _plan_one_subbt(cfg, sub_instruction, det.det_map,
                                       rgb, wrist_rgb, pick_h, place_h)
            if cfg.verbose:
                print(f"[Atomic {i + 1}] planner took "
                      f"{time.time() - t0:.1f}s -> "
                      f"{'OK' if sub_bt else 'EMPTY'}")
            if not sub_bt:
                if attempt == 0:
                    _retract_arm(env, model, data, ee_site)
                    continue
                break

            try:
                execute_on_libero(
                    env, sub_bt, det.det_map, cfg,
                    sam3=sam3, depth=det.depth,
                    cam_pos=det.cam.pos, cam_mat=det.cam.mat,
                    cam_fovy=det.cam.fovy,
                    cam_w=cfg.cam_width, cam_h=cfg.cam_height,
                    instruction=sub_instruction)
            except Exception as e:
                if cfg.verbose:
                    print(f"[Atomic {i + 1}] executor exception: {e}")

            # Check if the overall task is now satisfied (cheapest predicate
            # success proxy without re-parsing predicate-by-predicate).
            try:
                if env.check_success():
                    overall_success = True
                    attempt_succeeded = True
                    break
            except Exception:
                pass

            # If this is the last predicate, no point retrying further -
            # success will be decided in the final check.
            if i == total - 1:
                attempt_succeeded = True  # don't loop, but not necessarily success
                break
            # Sub-predicate not yet success - retract and retry.
            if attempt == 0:
                _retract_arm(env, model, data, ee_site)

        if overall_success:
            break
        if not attempt_succeeded and cfg.verbose:
            print(f"[Atomic {i + 1}/{total}] failed after retry; "
                  f"continuing to next predicate")

    # Final authoritative success check.
    try:
        return bool(env.check_success())
    except Exception:
        return overall_success

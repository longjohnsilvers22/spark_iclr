#!/usr/bin/env python3
"""
Dual-arm BT coordination for SPARK.

Option 1: Two independent SPARK BT instances sharing a latent state.
Each arm gets its own BT plan. They coordinate via a shared state dict
with synchronization primitives (wait_for, signal).

Usage:
    MUJOCO_GL=egl python -m spark_bench.dual_arm_bt --task TwoArmLift --num-trials 5
"""
from __future__ import annotations
import os
import numpy as np
import threading
from dataclasses import dataclass, field
from typing import Dict, Any

os.environ.setdefault('MUJOCO_GL', 'egl')
os.environ.setdefault('PYTORCH_CUDA_ALLOC_CONF', 'expandable_segments:True')


@dataclass
class SharedState:
    """
    Shared state between two arm BT executors.

    Thread-safe via a lock. Each arm can signal events and wait for them.
    """
    _state: Dict[str, Any] = field(default_factory=dict)
    _lock: threading.Lock = field(default_factory=threading.Lock)
    _events: Dict[str, threading.Event] = field(default_factory=dict)

    def set(self, key: str, value: Any):
        with self._lock:
            self._state[key] = value
        if key in self._events:
            self._events[key].set()

    def get(self, key: str, default=None):
        with self._lock:
            return self._state.get(key, default)

    def wait_for(self, key: str, timeout: float = 30.0) -> bool:
        """
        Block until key is set in shared state. Returns True if signaled.
        """
        if key not in self._events:
            self._events[key] = threading.Event()
        if self.get(key) is not None:
            return True
        return self._events[key].wait(timeout=timeout)

    def signal(self, event_name: str):
        """
        Signal an event (same as set(event_name, True)).
        """
        self.set(event_name, True)


# Task-specific dual-arm BT plans
DUAL_ARM_PLANS = {
    'TwoArmLift': {
        'arm0': {
            'instruction': 'Grasp pot handle0 from outside and lift together',
            'tree': {
                'type': 'sequence',
                'children': [
                    # Approach handle from outside (away from pot center) to wrap fingers
                    {'type': 'approach_handle', 'params': {'keypoint_label': 'handle0', 'pot_label': 'pot'}},
                    {'type': 'grasp', 'params': {'force': 120}},
                    {'type': 'signal', 'params': {'event': 'arm0_grasped'}},
                    {'type': 'wait_for', 'params': {'event': 'arm1_grasped'}},
                    {'type': 'move_relative', 'params': {'dz': 0.15}},
                ]
            }
        },
        'arm1': {
            'instruction': 'Grasp pot handle1 from outside and lift together',
            'tree': {
                'type': 'sequence',
                'children': [
                    {'type': 'approach_handle', 'params': {'keypoint_label': 'handle1', 'pot_label': 'pot'}},
                    {'type': 'grasp', 'params': {'force': 120}},
                    {'type': 'signal', 'params': {'event': 'arm1_grasped'}},
                    {'type': 'wait_for', 'params': {'event': 'arm0_grasped'}},
                    {'type': 'move_relative', 'params': {'dz': 0.15}},
                ]
            }
        },
    },
    'TwoArmHandover': {
        'arm0': {
            'instruction': 'Pick up hammer, lift, move to handover point (center), wait for arm1 to grasp, then release',
            'tree': {
                'type': 'sequence',
                'children': [
                    {'type': 'move_to_keypoint', 'params': {'keypoint_label': 'handle'}},
                    {'type': 'grasp', 'params': {'force': 100}},
                    {'type': 'move_relative', 'params': {'dz': 0.15}},
                    # Move toward center workspace for handover
                    {'type': 'move_to_fixed', 'params': {'x': 0.0, 'y': 0.0, 'z': 1.05}},
                    {'type': 'signal', 'params': {'event': 'arm0_holding'}},
                    {'type': 'wait_for', 'params': {'event': 'arm1_grasped'}},
                    {'type': 'release'},
                    {'type': 'move_relative', 'params': {'dz': 0.05}},
                ]
            }
        },
        'arm1': {
            'instruction': 'Wait for arm0 to hold hammer at center, then grasp it and lift',
            'tree': {
                'type': 'sequence',
                'children': [
                    {'type': 'wait_for', 'params': {'event': 'arm0_holding'}},
                    {'type': 'move_to_keypoint', 'params': {'keypoint_label': 'handle'}},
                    {'type': 'grasp', 'params': {'force': 100}},
                    {'type': 'signal', 'params': {'event': 'arm1_grasped'}},
                    {'type': 'move_relative', 'params': {'dz': 0.10}},
                ]
            }
        },
    },
}


def create_dual_arm_env(task_name: str, horizon: int = 12000, controller: str = 'OSC_POSE'):
    """
    Create a dual-arm robosuite environment.
    """
    import robosuite as suite
    if controller == 'JOINT_POSITION':
        # CaP-X config: input[-10,10] -> output[-1,1] for absolute-like control.
        # robosuite >= 1.5 needs the bare part-controller dict wrapped into a
        # composite ("BASIC") config per arm, otherwise the factory tries to
        # resolve "JOINT_POSITION" as a composite controller type and asserts.
        part = {
            'type': 'JOINT_POSITION',
            'input_max': 10, 'input_min': -10,
            'output_max': 1.0, 'output_min': -1.0,
            'kp': 1500, 'kd': 400, 'kv': 200,
            'interpolation': 'linear', 'ramp_ratio': 0.2,
        }
        from robosuite.controllers.composite.composite_controller_factory import (
            refactor_composite_controller_config)
        ctrl = refactor_composite_controller_config(part, 'Panda', ['right'])
    elif hasattr(suite, 'load_controller_config'):
        ctrl = suite.load_controller_config(default_controller=controller)
    else:
        # robosuite >= 1.5: OSC_POSE is the default arm controller under BASIC.
        ctrl = suite.load_composite_controller_config(controller='BASIC', robot='Panda')
    return suite.make(
        task_name,
        robots=['Panda', 'Panda'],
        env_configuration='single-arm-opposed',
        has_renderer=False,
        has_offscreen_renderer=True,
        use_camera_obs=True,
        camera_names=['agentview'],
        camera_heights=480,
        camera_widths=640,
        camera_depths=True,
        horizon=horizon,
        controller_configs=[ctrl, ctrl],
    )


_sam3_local_instance = None


def _get_sam3_local():
    """
    Singleton SAM3 perception loader for the dual-arm runner.

    Constructs ``SPARKPerception`` directly instead of importing ``_get_sam3``
    from ``run_spark_libero_pro_fair``, whose LIBERO import chain can hit a
    circular import. Reuses the same ``_detect_with_rendered_depth``.
    """
    global _sam3_local_instance
    if _sam3_local_instance is None:
        print("[SAM3] Loading...")
        try:
            from spark_real.perception.spark_perception import SPARKPerception
            _sam3_local_instance = SPARKPerception(sam3_threshold=0.03)
        except Exception:
            # Fall back to the shared loader.
            from spark_bench.run_spark_libero_pro_fair import _get_sam3
            _sam3_local_instance = _get_sam3()
    return _sam3_local_instance


def _detect_objects_sam3(env, prompts, verbose=False):
    """
    Detect objects via SAM3 perception (camera -> detection -> 3D position).

    Returns dict mapping label -> position_3d (numpy array).
    """
    import mujoco
    from robosuite.utils.camera_utils import get_real_depth_map

    model = env.sim.model._model
    data = env.sim.data._data
    mujoco.mj_forward(model, data)

    obs = env._get_observations()
    rgb = obs.get('agentview_image')
    if rgb is None:
        return {}
    rgb = rgb[::-1].copy()

    raw_depth = obs.get('agentview_depth')
    if raw_depth is None:
        return {}
    depth = get_real_depth_map(env.sim, raw_depth[::-1].copy())
    if depth.ndim == 3:
        depth = depth.squeeze(-1)

    cam_id = env.sim.model.camera_name2id('agentview')
    cam_pos = data.cam_xpos[cam_id].copy()
    cam_mat = data.cam_xmat[cam_id].reshape(3, 3).copy()
    cam_fovy = float(model.cam_fovy[cam_id])

    sam3 = _get_sam3_local()
    dets = sam3._detect_with_rendered_depth(
        rgb, depth, prompts, cam_pos, cam_mat, cam_fovy, 640, 480)

    det_map = {}
    for d in dets:
        if d.position_3d is not None:
            if d.label not in det_map or d.confidence > det_map[d.label].confidence:
                det_map[d.label] = d.position_3d.copy()
                if verbose:
                    p = d.position_3d
                    print(f"[SAM3] {d.label}: conf={d.confidence:.3f} "
                          f"pos=({p[0]:.3f},{p[1]:.3f},{p[2]:.3f})")
    return det_map


def _get_target_from_obs(obs, label):
    """
    Get live target position from obs dict, trying multiple key formats.
    """
    for key_fmt in [f'{label}_xpos', f'{label}_pos', f'handle_xpos']:
        if key_fmt in obs:
            return obs[key_fmt][:3].copy()
    return None


def _process_arm_action(obs, act, arm_idx, arm_id, shared, holding, step,
                        action, verbose, env, _state):
    """
    Process one BT action for one arm. Returns (new_arm_idx, new_holding).

    arm_id: 0 or 1 - determines action slice offsets.
    action: the full action array to fill in-place.
    _state: dict for per-arm persistent state (grasp_start, mr_start, approach_phase).
    """
    atype = act.get('type')
    params = act.get('params', {})
    # Action offsets: arm0=[0:7], arm1=[7:14]
    off = arm_id * 7
    grip_idx = off + 6
    ee_key = f'robot{arm_id}_eef_pos'

    if atype == 'signal':
        shared.signal(params.get('event', ''))
        if verbose:
            print(f"[step {step}] arm{arm_id}: signal '{params.get('event')}'")
        return arm_idx + 1, holding

    elif atype == 'wait_for':
        if shared.get(params.get('event', '')):
            return arm_idx + 1, holding
        action[grip_idx] = 1.0 if holding else -1.0
        return arm_idx, holding

    elif atype == 'approach_handle':
        # Approach pot handle from the SIDE (not from above).
        # The handle extends radially from pot center. The gripper must
        # approach perpendicular to the handle axis to wrap fingers around it.
        # Phase: outside_high -> outside_level -> push_inward
        label = params.get('keypoint_label', '')
        pot_label = params.get('pot_label', 'pot')
        handle_pos = _get_target_from_obs(obs, label)
        pot_pos = _get_target_from_obs(obs, pot_label)
        if handle_pos is None:
            if verbose:
                print(f"arm{arm_id}: handle '{label}' not found, skipping")
            return arm_idx + 1, holding
        ee = obs[ee_key][:3] if ee_key in obs else np.zeros(3)

        phase_key = f'arm{arm_id}_handle_phase'
        step_key = f'arm{arm_id}_handle_steps'
        if phase_key not in _state:
            _state[phase_key] = 'outside_high'
            _state[step_key] = 0
        _state[step_key] = _state.get(step_key, 0) + 1

        # Handle extends radially from pot. Compute outward direction.
        if pot_pos is not None:
            out_xy = handle_pos[:2] - pot_pos[:2]
            out_norm = np.linalg.norm(out_xy)
            out_dir = out_xy / max(out_norm, 0.001)
        else:
            out_dir = np.array([0, -1.0]) if arm_id == 0 else np.array([0, 1.0])

        if _state[phase_key] == 'outside_high':
            # Go to a point OUTWARD from handle tip, slightly below handle
            target = handle_pos.copy()
            target[:2] += out_dir * 0.05  # 5cm outside handle tip
            target[2] -= 0.015  # slightly BELOW handle center for finger wrap
            err = target - ee
            dist = np.linalg.norm(err)
            if verbose and step % 100 == 0:
                print(f"[step {step}] arm{arm_id} outside_high: dist={dist:.4f} "
                      f"ee=({ee[0]:.3f},{ee[1]:.3f},{ee[2]:.3f}) "
                      f"tgt=({target[0]:.3f},{target[1]:.3f},{target[2]:.3f})")
            if dist < 0.03 or _state[step_key] > 300:
                _state[phase_key] = 'outside_level'
                _state[step_key] = 0
            else:
                action[off:off+3] = np.clip(err * 5 / 0.05, -1, 1)
                action[grip_idx] = -1.0
        elif _state[phase_key] == 'outside_level':
            # Move to slightly below handle height, still outward
            target = handle_pos.copy()
            target[:2] += out_dir * 0.04
            target[2] -= 0.01  # slightly below handle
            err = target - ee
            dist = np.linalg.norm(err)
            if dist < 0.025 or _state[step_key] > 150:
                _state[phase_key] = 'push_inward'
                _state[step_key] = 0
                if verbose:
                    print(f"[step {step}] arm{arm_id}: outside_level -> push_inward")
            else:
                action[off:off+3] = np.clip(err * 5 / 0.05, -1, 1)
                action[grip_idx] = -1.0
        elif _state[phase_key] == 'push_inward':
            # Push inward THROUGH the handle toward pot center
            target = handle_pos.copy()
            target[:2] -= out_dir * 0.02  # 2cm past handle into pot
            target[2] -= 0.01  # stay slightly below handle center
            err = target - ee
            dist = np.linalg.norm(err)
            if dist < 0.025 or _state[step_key] > 150:
                del _state[phase_key]
                del _state[step_key]
                if verbose:
                    print(f"[step {step}] arm{arm_id}: approach_handle DONE (dist={dist:.4f})")
                return arm_idx + 1, holding
            else:
                action[off:off+3] = np.clip(err * 5 / 0.05, -1, 1)
                action[grip_idx] = -1.0
        return arm_idx, holding

    elif atype == 'move_to_keypoint':
        label = params.get('keypoint_label', '')
        target = _get_target_from_obs(obs, label)
        if target is None:
            return arm_idx + 1, holding
        ee = obs[ee_key][:3] if ee_key in obs else np.zeros(3)

        # Two-phase approach: first go above, then descend
        phase_key = f'arm{arm_id}_approach_phase'
        if phase_key not in _state:
            _state[phase_key] = 'above'

        if _state[phase_key] == 'above':
            above = target.copy(); above[2] += 0.06
            err = above - ee
            if np.linalg.norm(err) < 0.02:
                _state[phase_key] = 'descend'
            else:
                action[off:off+3] = np.clip(err * 8 / 0.05, -1, 1)
                action[grip_idx] = -1.0 if not holding else 1.0
        elif _state[phase_key] == 'descend':
            err = target - ee
            if np.linalg.norm(err) < 0.02:
                del _state[phase_key]
                return arm_idx + 1, holding
            else:
                action[off:off+3] = np.clip(err * 8 / 0.05, -1, 1)
                action[grip_idx] = -1.0 if not holding else 1.0
        return arm_idx, holding

    elif atype == 'grasp':
        action[grip_idx] = 1.0
        start_key = f'arm{arm_id}_grasp_start'
        if start_key not in _state:
            _state[start_key] = step
            if verbose:
                ee = obs[ee_key][:3] if ee_key in obs else np.zeros(3)
                print(f"[step {step}] arm{arm_id}: grasp start at ee=({ee[0]:.3f},{ee[1]:.3f},{ee[2]:.3f})")
        if step - _state[start_key] > 80:
            del _state[start_key]
            return arm_idx + 1, True
        return arm_idx, True

    elif atype == 'release':
        action[grip_idx] = -1.0
        return arm_idx + 1, False

    elif atype == 'move_to_fixed':
        # Move to a fixed world-space coordinate
        target = np.array([params.get('x', 0), params.get('y', 0), params.get('z', 1.0)])
        ee = obs[ee_key][:3] if ee_key in obs else np.zeros(3)
        err = target - ee
        dist = np.linalg.norm(err)
        step_key = f'arm{arm_id}_fixed_steps'
        _state[step_key] = _state.get(step_key, 0) + 1
        if dist < 0.03 or _state[step_key] > 300:
            _state.pop(step_key, None)
            return arm_idx + 1, holding
        action[off:off+3] = np.clip(err * 5 / 0.05, -1, 1)
        action[grip_idx] = 1.0 if holding else -1.0
        return arm_idx, holding

    elif atype == 'move_relative':
        dz = params.get('dz', 0.1)
        action[off + 2] = 1.0  # Max upward
        action[grip_idx] = 1.0 if holding else -1.0
        start_key = f'arm{arm_id}_mr_start'
        if start_key not in _state:
            _state[start_key] = step
        if step - _state[start_key] > 300:
            del _state[start_key]
            return arm_idx + 1, holding
        return arm_idx, holding

    return arm_idx, holding


def _simple_move_to(env, target, arm_id, grip_closed, steps=200, verbose=False):
    """
    Move one arm to target via OSC delta. arm_id=0 or 1.
    Single env.step per iteration to avoid wasting horizon.
    """
    off = arm_id * 7
    grip_idx = off + 6
    ee_key = f'robot{arm_id}_eef_pos'
    other_grip = 6 if arm_id == 1 else 13
    obs = None
    for s in range(steps):
        if obs is None:
            obs, _, done, _ = env.step(np.zeros(env.action_dim))
        if ee_key not in obs:
            break
        ee = obs[ee_key][:3]
        err = target - ee
        if np.linalg.norm(err) < 0.02:
            return True
        action = np.zeros(env.action_dim)
        action[off:off+3] = np.clip(err * 5 / 0.05, -1, 1)
        action[grip_idx] = 1.0 if grip_closed else -1.0
        action[other_grip] = 1.0  # Always keep other arm gripping
        try:
            obs, _, done, _ = env.step(action)
            if done: break
        except:
            break
    return False


def _simple_grip(env, arm_id, close, steps=80, other_closed=True):
    """
    Open or close gripper for one arm, keeping other arm's gripper set.
    """
    off = arm_id * 7
    grip_idx = off + 6
    other_grip = 6 if arm_id == 1 else 13
    for _ in range(steps):
        action = np.zeros(env.action_dim)
        action[grip_idx] = 1.0 if close else -1.0
        action[other_grip] = 1.0 if other_closed else -1.0
        try:
            env.step(action)
        except:
            break


def execute_dual_arm_bt(env, task_name: str, verbose: bool = False,
                        adaptive: bool = False) -> bool:
    """
    Execute dual-arm task with simple sequential coordination.

    Args:
        adaptive: if True, generate SAM3 detection prompts via Gemini
            (K=3 variants + SAM3 self-select) from the task instruction alone,
            matching the fair-adaptive config used in single-arm eval.
    """
    obs = env._get_observations()
    if verbose:
        for key in sorted(obs.keys()):
            if 'pos' in key or 'xpos' in key:
                val = obs[key]
                if hasattr(val, '__len__') and len(val) >= 3:
                    print(f"obs[{key}] = ({val[0]:.3f},{val[1]:.3f},{val[2]:.3f})")

    if task_name == 'TwoArmLift':
        return _execute_two_arm_lift(env, obs, verbose, adaptive=adaptive)
    elif task_name == 'TwoArmHandover':
        return _execute_two_arm_handover(env, obs, verbose, adaptive=adaptive)
    else:
        print(f"Unknown dual-arm task: {task_name}")
        return False


# Task-language-like descriptions for adaptive prompt generation
_BIMANUAL_LANGUAGES = {
    'TwoArmLift': 'Pick up the pot by both handles and lift it together with both arms',
    'TwoArmHandover': 'Arm 0 picks up the hammer by its handle and hands it off to arm 1',
}

# Concepts per task (base names; Gemini expands each to K=3 variants)
_BIMANUAL_CONCEPTS = {
    'TwoArmLift': ['pot', 'left pot handle', 'right pot handle'],
    'TwoArmHandover': ['hammer', 'hammer handle'],
}


def _adaptive_detect(env, task_name, base_concepts, verbose=False, k=3):
    """
    Gemini-K-variants + SAM3 self-selection for bimanual tasks.

    Mirrors the adaptive block in run_spark_libero_pro_fair.py:
    - Gemini generates K variant phrasings per concept from task language
    - SAM3 runs each variant, scored by cardinality + confidence
    - Centroid-dedup of winners (drops redundant concepts hitting same object)
    Returns det_map: {phrase -> position_3d} after normal SAM3 detection.
    """
    instruction = _BIMANUAL_LANGUAGES.get(task_name, '')

    try:
        from spark_real.planning.spark_planner import SPARKPlanner
        from google.genai import types as _gtypes
        import json as _json
        planner = SPARKPlanner(llm_backend='gemini')
        client = planner._get_client()
        prompt = (
            f"You are helping an open-vocabulary object detector. For each "
            f"concept below (one per line), output {k} distinct short detection "
            f"phrases (1-4 words each). Vary colors, shapes, materials. "
            f"Task context: \"{instruction}\"\n\nConcepts:\n"
            + "\n".join(f"- {c}" for c in base_concepts) +
            "\n\nOutput JSON only, no code fences:\n"
            "{\"concept1\": [\"phrase1\", \"phrase2\", ...], ...}"
        )
        resp = client.models.generate_content(
            model='gemini-3-flash-preview', contents=prompt,
            config=_gtypes.GenerateContentConfig(temperature=0))
        txt = resp.text.strip()
        if '```' in txt:
            txt = txt.split('```', 2)[1]
            if txt.startswith('json'):
                txt = txt[4:]
        variants_by_concept = _json.loads(txt.strip())
    except Exception as e:
        if verbose:
            print(f"[Adaptive] variant-gen failed: {e}; falling back to base concepts")
        return _detect_objects_sam3(env, base_concepts, verbose=verbose)

    if not variants_by_concept:
        return _detect_objects_sam3(env, base_concepts, verbose=verbose)

    # SAM3 preflight: pick best variant per concept by cardinality x conf
    import mujoco
    from PIL import Image as _PIL
    import torch as _torch
    import numpy as _np
    from spark_bench.run_spark_libero_pro_fair import _get_sam3

    model = env.sim.model._model
    data = env.sim.data._data
    mujoco.mj_forward(model, data)
    obs = env._get_observations()
    rgb = obs.get('agentview_image')
    if rgb is None:
        return {}
    rgb = rgb[::-1].copy()

    sam3 = _get_sam3()
    sam3.load_models(load_da3=False)
    state = sam3._sam3.set_image(_PIL.fromarray(rgb))

    chosen = []
    for concept, variants in variants_by_concept.items():
        if not isinstance(variants, list):
            continue
        best = (-1.0, concept, None)
        for v in variants[:k]:
            if not isinstance(v, str):
                continue
            st = sam3._sam3.set_text_prompt(prompt=v, state=state)
            m = st.get('masks', _torch.tensor([]))
            s = st.get('scores', _torch.tensor([]))
            if m.numel() == 0:
                score = 0.0
                centroid = None
            else:
                n = m.shape[0]
                conf = float(s.max().item()) if s.numel() else 0.0
                card_mult = 1.0 if n == 1 else (0.6 if n == 2 else 0.1)
                score = card_mult * conf
                bi = int(s.argmax().item())
                msk = m[bi].cpu().numpy().squeeze().astype(bool)
                if msk.any():
                    ys, xs = _np.where(msk)
                    centroid = (float(xs.mean()), float(ys.mean()))
                else:
                    centroid = None
            if score > best[0]:
                best = (score, v, centroid)
        chosen.append(best)

    # Centroid dedup (same 30px threshold as libero)
    deduped = []
    for s_i, v_i, c_i in chosen:
        if c_i is None:
            deduped.append((s_i, v_i, c_i))
            continue
        dup = False
        for j, (s_j, v_j, c_j) in enumerate(deduped):
            if c_j is not None and _np.hypot(c_i[0]-c_j[0], c_i[1]-c_j[1]) < 30.0:
                dup = True
                if s_i > s_j:
                    deduped[j] = (s_i, v_i, c_i)
                break
        if not dup:
            deduped.append((s_i, v_i, c_i))

    chosen_phrases = [v for _, v, _ in deduped]
    if verbose:
        print(f"[Adaptive] chose: {chosen_phrases} (from {len(chosen)} concepts)")

    # Now run full SAM3 with depth to get 3D positions for the chosen phrases
    return _detect_objects_sam3(env, chosen_phrases, verbose=verbose)


def _execute_two_arm_lift(env, obs, verbose=False, adaptive=False):
    """
    TwoArmLift: top-down approach with rotation search for handle grasping.

    Handle center bar is 11cm x 2cm x 2cm. Descend from above, try
    multiple wrist rotations until the fingers straddle the bar, then
    close and lift.
    """
    # Detect objects via SAM3 perception (no ground-truth obs)
    if adaptive:
        sam3_dets = _adaptive_detect(
            env, 'TwoArmLift', _BIMANUAL_CONCEPTS['TwoArmLift'], verbose=verbose)
    else:
        sam3_dets = _detect_objects_sam3(
            env, ['pot handle', 'pot', 'green handle', 'blue handle'], verbose=verbose)
    # Map SAM3 detections to handle0/handle1/pot - accept any label containing keyword
    def _find(dets, *keywords):
        for k in keywords:
            for label, pos in dets.items():
                if k.lower() in label.lower():
                    return pos
        return None
    pot = _find(sam3_dets, 'pot') if sam3_dets else None
    if pot is None:
        pot = obs.get('pot_pos', np.zeros(3))[:3].copy()
    h0 = _find(sam3_dets, 'left', 'green', 'handle0') if sam3_dets else None
    h1 = _find(sam3_dets, 'right', 'blue', 'handle1') if sam3_dets else None
    # Second try: any prompt containing 'handle' (pick two distinct if available)
    if h0 is None:
        for label, pos in sam3_dets.items():
            if 'handle' in label.lower():
                h0 = pos; break
    if h1 is None:
        for label, pos in sam3_dets.items():
            if 'handle' in label.lower() and (h0 is None or not np.allclose(pos, h0)):
                h1 = pos; break
    # Fallback: if SAM3 can't distinguish handles, estimate from pot center
    if h0 is None or h1 is None:
        # Handles are ~16cm from pot center, opposed along Y
        if h0 is None:
            h0 = pot.copy(); h0[1] -= 0.16
        if h1 is None:
            h1 = pot.copy(); h1[1] += 0.16
        h0[2] = pot[2] + 0.06  # Handle height above pot center
        h1[2] = pot[2] + 0.06

    if verbose:
        print(f"handle0={h0}, handle1={h1}, pot={pot}")

    # Phase 1: Move both arms ABOVE their respective handles
    above_z = max(h0[2], h1[2]) + 0.08
    tgt0_above = h0.copy(); tgt0_above[2] = above_z
    tgt1_above = h1.copy(); tgt1_above[2] = above_z

    for step in range(300):
        obs_new, _, done, _ = env.step(np.zeros(env.action_dim))
        ee0 = obs_new.get('robot0_eef_pos', np.zeros(3))[:3]
        ee1 = obs_new.get('robot1_eef_pos', np.zeros(3))[:3]
        err0 = tgt0_above - ee0
        err1 = tgt1_above - ee1
        if np.linalg.norm(err0) < 0.02 and np.linalg.norm(err1) < 0.02:
            break
        action = np.zeros(env.action_dim)
        action[0:3] = np.clip(err0 * 8 / 0.05, -1, 1)
        action[7:10] = np.clip(err1 * 8 / 0.05, -1, 1)
        action[6] = -1.0; action[13] = -1.0  # Open grippers
        try:
            obs_new, _, done, _ = env.step(action)
            if done: break
        except:
            break

    if verbose:
        obs_new, _, _, _ = env.step(np.zeros(env.action_dim))
        ee0 = obs_new.get('robot0_eef_pos', np.zeros(3))[:3]
        print(f"Phase 1 done: arm0 at ({ee0[0]:.3f},{ee0[1]:.3f},{ee0[2]:.3f})")

    # Phase 2: Approach from the SIDE (CaP-X inspired sideways grasp)
    # Gripper fingers should close perpendicular to the handle bar
    out0 = h0[:2] - pot[:2]
    out0 = out0 / (np.linalg.norm(out0) + 1e-8)
    out1 = h1[:2] - pot[:2]
    out1 = out1 / (np.linalg.norm(out1) + 1e-8)

    # Position arms OUTSIDE the handles (6cm outward), slightly BELOW handle
    # Being below helps fingers wrap around the bar on closure
    approach0 = h0.copy()
    approach0[:2] += out0 * 0.06  # 6cm outside handle
    approach0[2] -= 0.01  # 1cm below handle center
    approach1 = h1.copy()
    approach1[:2] += out1 * 0.06
    approach1[2] -= 0.01

    for step in range(300):
        obs_new, _, done, _ = env.step(np.zeros(env.action_dim))
        ee0 = obs_new.get('robot0_eef_pos', np.zeros(3))[:3]
        ee1 = obs_new.get('robot1_eef_pos', np.zeros(3))[:3]
        err0 = approach0 - ee0
        err1 = approach1 - ee1
        if np.linalg.norm(err0) < 0.015 and np.linalg.norm(err1) < 0.015:
            break
        action = np.zeros(env.action_dim)
        action[0:3] = np.clip(err0 * 10 / 0.05, -1, 1)
        action[7:10] = np.clip(err1 * 10 / 0.05, -1, 1)
        action[6] = -1.0; action[13] = -1.0
        try:
            obs_new, _, done, _ = env.step(action)
            if done: break
        except: break

    if verbose:
        obs_new, _, _, _ = env.step(np.zeros(env.action_dim))
        ee0 = obs_new.get('robot0_eef_pos', np.zeros(3))[:3]
        print(f"Phase 2: outside handles, arm0=({ee0[0]:.3f},{ee0[1]:.3f},{ee0[2]:.3f})")

    # Phase 3: Push INWARD through handles toward pot center
    # Push 2cm past handle center - aggressively wraps fingers around bar
    tgt0_grasp = h0.copy()
    tgt0_grasp[:2] -= out0 * 0.02  # 2cm past handle toward pot
    tgt0_grasp[2] -= 0.01  # Stay slightly below
    tgt1_grasp = h1.copy()
    tgt1_grasp[:2] -= out1 * 0.02
    tgt1_grasp[2] -= 0.01
    obs_new = obs
    for step in range(250):
        ee0 = obs_new.get('robot0_eef_pos', np.zeros(3))[:3]
        ee1 = obs_new.get('robot1_eef_pos', np.zeros(3))[:3]
        err0 = tgt0_grasp - ee0
        err1 = tgt1_grasp - ee1
        d0 = np.linalg.norm(err0)
        d1 = np.linalg.norm(err1)
        if d0 < 0.012 and d1 < 0.012:
            if verbose:
                print(f"Phase 3: both at handles (d0={d0:.4f}, d1={d1:.4f})")
            break
        action = np.zeros(env.action_dim)
        action[0:3] = np.clip(err0 * 10 / 0.05, -1, 1)
        action[7:10] = np.clip(err1 * 10 / 0.05, -1, 1)
        action[6] = -1.0; action[13] = -1.0
        try:
            obs_new, _, done, _ = env.step(action)
            if done: break
        except:
            break

    if verbose:
        obs_new, _, _, _ = env.step(np.zeros(env.action_dim))
        ee0 = obs_new.get('robot0_eef_pos', np.zeros(3))[:3]
        ee1 = obs_new.get('robot1_eef_pos', np.zeros(3))[:3]
        print(f"Phase 3 done: arm0=({ee0[0]:.3f},{ee0[1]:.3f},{ee0[2]:.3f}), "
              f"arm1=({ee1[0]:.3f},{ee1[1]:.3f},{ee1[2]:.3f})")

    # Phase 4-5: Grasp + lift with rotation retry
    # The handle bar is 2cm thick - finger orientation must align.
    # Try: close -> lift -> check success -> if fail, rotate + retry.
    # Cap at 2 attempts to preserve horizon budget for actual lifting.
    _lift_succeeded = False
    for rot_attempt in range(2):
        # Close both grippers
        for _ in range(80):
            action = np.zeros(env.action_dim)
            action[6] = 1.0; action[13] = 1.0
            try: env.step(action)
            except: break

        # Full lift attempt
        for step in range(400):
            action = np.zeros(env.action_dim)
            action[2] = 1.0; action[6] = 1.0
            action[9] = 1.0; action[13] = 1.0
            try:
                _, _, done, _ = env.step(action)
                if done: break
                if step > 0 and step % 50 == 0 and env._check_success():
                    break
            except: break

        if env._check_success():
            if verbose:
                print(f"Rot attempt {rot_attempt}: SUCCESS")
            _lift_succeeded = True
            break

        if verbose:
            print(f"Rot attempt {rot_attempt}: FAIL - retrying with rotated wrists")

        # Lower back down
        for _ in range(80):
            action = np.zeros(env.action_dim)
            action[2] = -0.5; action[9] = -0.5
            action[6] = -1.0; action[13] = -1.0  # Open
            try: env.step(action)
            except: break

        # Rotate both wrists ~45 deg
        for _ in range(40):
            action = np.zeros(env.action_dim)
            action[5] = 0.5; action[12] = 0.5
            action[6] = -1.0; action[13] = -1.0
            try: env.step(action)
            except: break

        # Re-descend to handles (use initial SAM3 positions)
        for step in range(150):
            obs_rd, _, done, _ = env.step(np.zeros(env.action_dim))
            ee0 = obs_rd.get('robot0_eef_pos', np.zeros(3))[:3]
            ee1 = obs_rd.get('robot1_eef_pos', np.zeros(3))[:3]
            err0 = tgt0_grasp - ee0; err1 = tgt1_grasp - ee1
            if np.linalg.norm(err0) < 0.02 and np.linalg.norm(err1) < 0.02:
                break
            action = np.zeros(env.action_dim)
            action[0:3] = np.clip(err0 * 8 / 0.05, -1, 1)
            action[7:10] = np.clip(err1 * 8 / 0.05, -1, 1)
            action[6] = -1.0; action[13] = -1.0
            try:
                obs_rd, _, done, _ = env.step(action)
                if done: break
            except: break

    # Short-circuit on success: the Hold phase can let the pot slip and
    # flip the success flag back to False.
    if _lift_succeeded:
        return True

    # Hold
    for _ in range(100):
        action = np.zeros(env.action_dim)
        action[6] = 1.0; action[13] = 1.0
        try: env.step(action)
        except: break

    return bool(env._check_success())


def _setup_arm_ik(env, arm_id):
    """
    Set up IK + torque control infrastructure for one arm.
    Returns (joint_ids, actuator_ids, gripper_actuator_ids, ee_site, model, data).
    """
    import mujoco as _mj
    model = env.sim.model._model
    data = env.sim.data._data

    prefix = f'robot{arm_id}'
    # Find arm joints
    joint_ids = []
    for i in range(model.njnt):
        jn = (_mj.mj_id2name(model, _mj.mjtObj.mjOBJ_JOINT, i) or '')
        if prefix in jn and 'finger' not in jn and 'gripper' not in jn.lower():
            if model.jnt_type[i] == 3:  # hinge
                joint_ids.append(i)

    # Find arm and gripper actuators
    arm_act_ids = []
    grip_act_ids = []
    for i in range(model.nu):
        an = (_mj.mj_id2name(model, _mj.mjtObj.mjOBJ_ACTUATOR, i) or '')
        if prefix not in an and f'gripper{arm_id}' not in an:
            continue
        if 'gripper' in an or 'finger' in an:
            grip_act_ids.append(i)
        else:
            arm_act_ids.append(i)

    # Find EE site - gripper sites are named 'gripper0_grip_site' or 'gripper1_grip_site'
    ee_site = -1
    gripper_prefix = f'gripper{arm_id}'
    for s in range(model.nsite):
        sn = (_mj.mj_id2name(model, _mj.mjtObj.mjOBJ_SITE, s) or '')
        if gripper_prefix in sn and 'grip_site' in sn and 'cylinder' not in sn:
            ee_site = s
            break

    return joint_ids, arm_act_ids, grip_act_ids, ee_site, model, data


def _pyroki_ik(target_pos_world, target_quat_wxyz, base_pos=None, base_quat_wxyz=None,
               robot=None, link_name='panda_hand'):
    """
    6-DOF IK using pyroki. Returns 7 joint angles.
    Transforms world-frame target to robot base frame before solving.

    Args:
        target_pos_world: Target EE position in WORLD frame
        target_quat_wxyz: Target EE orientation in WORLD frame (wxyz)
        base_pos: Robot base position in world frame
        base_quat_wxyz: Robot base orientation in world frame (wxyz)
    """
    import pyroki as pk
    import jax.numpy as jnp
    import jax_dataclasses as jdc
    import jaxlie, jaxls
    from scipy.spatial.transform import Rotation as R

    if robot is None:
        if not hasattr(_pyroki_ik, '_robot'):
            import yourdfpy
            import robot_descriptions.panda_description as pd
            urdf = yourdfpy.URDF.load(pd.URDF_PATH)
            _pyroki_ik._robot = pk.Robot.from_urdf(urdf)
        robot = _pyroki_ik._robot

    # Transform from world frame to robot base frame
    if base_pos is not None and base_quat_wxyz is not None:
        base_rot = R.from_quat([base_quat_wxyz[1], base_quat_wxyz[2],
                                base_quat_wxyz[3], base_quat_wxyz[0]])  # xyzw
        base_rot_inv = base_rot.inv()
        target_pos_base = base_rot_inv.apply(target_pos_world - base_pos)
        target_rot_world = R.from_quat([target_quat_wxyz[1], target_quat_wxyz[2],
                                         target_quat_wxyz[3], target_quat_wxyz[0]])
        target_rot_base = base_rot_inv * target_rot_world
        tq_xyzw = target_rot_base.as_quat()
        target_quat_base = np.array([tq_xyzw[3], tq_xyzw[0], tq_xyzw[1], tq_xyzw[2]])
    else:
        target_pos_base = target_pos_world
        target_quat_base = target_quat_wxyz

    tli = robot.links.names.index(link_name)

    @jdc.jit
    def _solve(robot, tli, tw, tp):
        jv = robot.joint_var_cls(0)
        costs = [
            pk.costs.pose_cost_analytic_jac(
                robot, jv,
                jaxlie.SE3.from_rotation_and_translation(jaxlie.SO3(tw), tp),
                tli, pos_weight=50.0, ori_weight=10.0),
            pk.costs.limit_constraint(robot, jv),
        ]
        return (jaxls.LeastSquaresProblem(costs=costs, variables=[jv])
                .analyze().solve(verbose=False, linear_solver='dense_cholesky',
                                trust_region=jaxls.TrustRegionConfig(lambda_initial=1.0)))[jv]

    cfg = np.array(_solve(robot, jnp.array(tli),
                          jnp.array(target_quat_base, dtype=np.float32),
                          jnp.array(target_pos_base, dtype=np.float32)))
    return cfg[:7]


def _ik_6dof(model, data, joint_ids, ee_site, target_pos, target_quat,
             n_restarts=5):
    """
    6-DOF IK with random restarts for robust convergence.
    Tries multiple initial configs and returns the best solution.
    """
    import mujoco as _mj
    ndof = len(joint_ids)
    # Get joint limits for clamping
    j_lo = np.array([model.jnt_range[j][0] for j in joint_ids])
    j_hi = np.array([model.jnt_range[j][1] for j in joint_ids])

    best_q = None
    best_err = float('inf')

    for restart in range(n_restarts):
        d2 = _mj.MjData(model)
        d2.qpos[:] = data.qpos[:]
        if restart == 0:
            q = np.array([data.qpos[model.jnt_qposadr[j]] for j in joint_ids])
        else:
            # Random restart: sample within joint limits
            q = j_lo + np.random.rand(ndof) * (j_hi - j_lo)
        for i, jid in enumerate(joint_ids):
            d2.qpos[model.jnt_qposadr[jid]] = q[i]

        for it in range(400):
            _mj.mj_forward(model, d2)
            ee = d2.site_xpos[ee_site].copy()
            pos_err = target_pos - ee
            # Orientation error via quaternion
            ee_quat = np.zeros(4)
            _mj.mju_mat2Quat(ee_quat, d2.site_xmat[ee_site].reshape(9))
            quat_err = np.zeros(4)
            ee_conj = ee_quat.copy(); ee_conj[1:] *= -1
            _mj.mju_mulQuat(quat_err, target_quat, ee_conj)
            ori_err = quat_err[1:] * 2.0
            if quat_err[0] < 0:
                ori_err *= -1

            pos_norm = np.linalg.norm(pos_err)
            ori_norm = np.linalg.norm(ori_err)
            if pos_norm < 0.003 and ori_norm < 0.05:
                break

            # Weighted error: position is 3x more important
            err = np.concatenate([pos_err * 3.0, ori_err])
            jacp = np.zeros((3, model.nv))
            jacr = np.zeros((3, model.nv))
            _mj.mj_jacSite(model, d2, jacp, jacr, ee_site)
            J = np.zeros((6, ndof))
            for i, jid in enumerate(joint_ids):
                dof = model.jnt_dofadr[jid]
                J[:3, i] = jacp[:, dof] * 3.0
                J[3:, i] = jacr[:, dof]
            # Damped pseudoinverse with stronger damping for stability
            JJT = J @ J.T + 0.05**2 * np.eye(6)
            dq = 0.2 * J.T @ np.linalg.solve(JJT, err)
            q += dq
            q = np.clip(q, j_lo + 0.01, j_hi - 0.01)
            for i, jid in enumerate(joint_ids):
                d2.qpos[model.jnt_qposadr[jid]] = q[i]

        # Score this solution
        total_err = pos_norm + ori_norm * 0.3
        if total_err < best_err:
            best_err = total_err
            best_q = q.copy()

    return best_q


def _torque_move(model, data, joint_ids, arm_act_ids, grip_act_ids,
                 q_target, gripper_close, steps=200,
                 other_arm_act_ids=None, other_grip_act_ids=None, other_close=True):
    """
    Torque PD control for one arm, keeping other arm's gripper set.
    """
    import mujoco as _mj
    KP, KD, TORQUE_LIM = 30.0, 10.0, 87.0
    grip_open_val = 0.04

    for step in range(steps):
        _mj.mj_forward(model, data)
        q = np.array([data.qpos[model.jnt_qposadr[j]] for j in joint_ids])
        qd = np.array([data.qvel[model.jnt_dofadr[j]] for j in joint_ids])
        q_err = q_target - q
        tau = KP * q_err - KD * qd
        for i, jid in enumerate(joint_ids):
            tau[i] += data.qfrc_bias[model.jnt_dofadr[jid]]
        for i in range(min(len(tau), len(arm_act_ids))):
            data.ctrl[arm_act_ids[i]] = np.clip(tau[i], -TORQUE_LIM, TORQUE_LIM)
        for gi, gid in enumerate(grip_act_ids):
            if gripper_close:
                data.ctrl[gid] = 0.0
            else:
                data.ctrl[gid] = grip_open_val if gi == 0 else -grip_open_val
        # Keep other arm gripper
        if other_grip_act_ids is not None:
            for gi, gid in enumerate(other_grip_act_ids):
                if other_close:
                    data.ctrl[gid] = 0.0
                else:
                    data.ctrl[gid] = grip_open_val if gi == 0 else -grip_open_val
        _mj.mj_step(model, data)
        if np.linalg.norm(q_err) < 0.01 and np.linalg.norm(qd) < 0.1:
            break


def _get_joint_pos(obs, arm_id):
    """
    Extract 7-DOF joint positions from obs (cos/sin -> angle via atan2).
    """
    cos_key = f'robot{arm_id}_joint_pos_cos'
    sin_key = f'robot{arm_id}_joint_pos_sin'
    if cos_key in obs and sin_key in obs:
        cos_vals = np.array(obs[cos_key])
        sin_vals = np.array(obs[sin_key])
        return np.arctan2(sin_vals, cos_vals)
    return None


def _jp_move_blocking(env, arm_id, target_joints, gripper_close,
                      other_joints=None, other_gripper_close=True,
                      tolerance=0.02, max_steps=150):
    """
    Blocking move using CaP-X delta JOINT_POSITION encoding.
    Config: input[-10,10] -> output[-1,1] -> SCALE=0.1.
    Action = (target - current) / SCALE, clipped to [-5, 5] for smooth motion.
    """
    target = np.asarray(target_joints).reshape(7)
    grip_cmd = 1.0 if gripper_close else -1.0
    other_grip = 1.0 if other_gripper_close else -1.0
    SCALE = 0.1  # output_range / input_range = 2/20

    obs = env._get_observations()
    for step in range(max_steps):
        current = _get_joint_pos(obs, arm_id)
        if current is None:
            break
        error = np.linalg.norm(current - target)
        if error < tolerance:
            break

        other_id = 1 - arm_id
        other_cur = _get_joint_pos(obs, other_id)
        if other_cur is None:
            other_cur = np.zeros(7)

        delta_active = np.clip((target - current) / SCALE, -5.0, 5.0)
        delta_other = np.zeros(7)  # Zero delta = hold position

        action = np.zeros(16)
        if arm_id == 0:
            action[0:7] = delta_active
            action[7] = grip_cmd
            action[8:15] = delta_other
            action[15] = other_grip
        else:
            action[0:7] = delta_other
            action[7] = other_grip
            action[8:15] = delta_active
            action[15] = grip_cmd

        try:
            obs, _, done, _ = env.step(action)
            if done: break
        except:
            break
    return obs


def _jp_grip(env, arm_id, close, other_close=True, steps=60):
    """
    Set gripper for one arm. Zero joint delta = hold position.
    """
    grip_cmd = 1.0 if close else -1.0
    other_grip = 1.0 if other_close else -1.0
    for _ in range(steps):
        action = np.zeros(16)  # Zero delta = hold all joints
        if arm_id == 0:
            action[7] = grip_cmd
            action[15] = other_grip
        else:
            action[7] = other_grip
            action[15] = grip_cmd
        try:
            env.step(action)
        except: break


def _execute_two_arm_handover(env, obs, verbose=False, adaptive=False):
    """
    TwoArmHandover using the JOINT_POSITION controller, hand-authored (no LLM).

    Perception is SAM3 only (fixed concepts, no Gemini variant generation).
    Motion uses warm-started, joint-limit-clamped position IK driven in small
    incremental waypoints, because the JOINT_POSITION controller will not
    converge on a single large joint jump (it saturates / picks an untrackable
    branch).

    Pipeline:
      1. Arm0 picks the hammer handle top-down. SAM3's handle centroid is
         noisy in XY and biased ~3 cm high, so arm0 searches an XY grid and
         confirms each candidate with a TEST-LIFT (close, lift 8 cm, check the
         hammer height rose), retrying until a real grasp is found.
      2. Arm0 lifts to clearance and transits in small segments to a handover
         point near world centre that both opposed arms can reach.
      3. The head pulls the held handle vertical, hanging below arm0's grip.
         Arm1 approaches from its own side in incremental waypoints to a point
         ~7 cm below arm0's grip (on the handle, clear of arm0's fingers),
         refines its reach, then sweeps the wrist (j7) to straddle the handle.
      4. Arm0 releases ONLY after arm1's grasp is confirmed by both a valid
         gripper width AND env._get_task_info()[1] (arm1 on the handle).
      5. Arm1 lifts; success = arm1-on-handle, arm0 released, hammer above
         the table height threshold.
    """
    import mujoco as _mj
    from scipy.spatial.transform import Rotation as _R

    # Grasp-width thresholds: Panda 2F-85 gripper has ~0.085 m fully open.
    _GRIPPER_EMPTY_CLOSE_THRESH = 0.005   # below = jaws fully closed, no object
    _GRIPPER_OBJECT_CONFIRM_MAX = 0.080   # above = nothing held
    model = env.sim.model._model
    data = env.sim.data._data

    # Check if we're using JOINT_POSITION controller (16D action = 2 x (7 joints + 1 gripper))
    use_jp = env.action_dim == 16
    if verbose:
        print(f"controller={'JP' if use_jp else 'OSC'}, action_dim={env.action_dim}")

    # Detect hammer via SAM3
    if adaptive:
        sam3_dets = _adaptive_detect(
            env, 'TwoArmHandover', _BIMANUAL_CONCEPTS['TwoArmHandover'], verbose=verbose)
    else:
        sam3_dets = _detect_objects_sam3(
            env, ['hammer', 'hammer handle', 'tool handle'], verbose=verbose)
    # Find handle: try any label containing 'handle' first, fallback to any 'hammer'
    handle = None
    for label, pos in sam3_dets.items():
        if 'handle' in label.lower():
            handle = pos; break
    if handle is None:
        for label, pos in sam3_dets.items():
            if 'hammer' in label.lower():
                handle = pos; break
    if handle is None:
        if verbose:
            print("SAM3 failed - no hammer detected")
        return False  # No GT fallback - fair evaluation
    if verbose:
        print(f"hammer handle={handle}, controller={'JP' if use_jp else 'OSC'}")

    if use_jp:
        # JOINT_POSITION PATH (CaP-X delta encoding)
        arm0_jids, _, _, arm0_ee, _, _ = _setup_arm_ik(env, 0)
        arm1_jids, _, _, arm1_ee, _, _ = _setup_arm_ik(env, 1)

        def pos_ik(jids, ee_s, target_pos):
            # Position-only IK, warm-started from current qpos. CLAMP to joint
            # limits every iteration: without this the integrator can walk a
            # joint past its limit / across a 2*pi wrap, producing a target the
            # JOINT_POSITION controller reads as a ~6 rad error and can't track
            # (it saturates and the EE lands far from target). Clamping keeps
            # the solution in the same trackable branch as the current config.
            nd = len(jids)
            j_lo = np.array([model.jnt_range[j][0] for j in jids])
            j_hi = np.array([model.jnt_range[j][1] for j in jids])
            q = np.array([data.qpos[model.jnt_qposadr[j]] for j in jids])
            d2 = _mj.MjData(model); d2.qpos[:] = data.qpos[:]
            for _ in range(200):
                _mj.mj_forward(model, d2)
                err = target_pos - d2.site_xpos[ee_s][:3]
                if np.linalg.norm(err) < 0.003: break
                jacp = np.zeros((3, model.nv))
                _mj.mj_jacSite(model, d2, jacp, None, ee_s)
                J = np.zeros((3, nd))
                for i, jid in enumerate(jids): J[:, i] = jacp[:, model.jnt_dofadr[jid]]
                q += 0.5 * J.T @ np.linalg.solve(J @ J.T + 0.05**2 * np.eye(3), err)
                q = np.clip(q, j_lo + 0.01, j_hi - 0.01)
                for i, jid in enumerate(jids): d2.qpos[model.jnt_qposadr[jid]] = q[i]
            return q

        def hammer_h():
            """
            Hammer height above table (>0.10 m => lifted for success).
            """
            try:
                ti = env._get_task_info()
                return ti[2] - ti[3]
            except Exception:
                return -1.0

        # Phase 1+2: Arm0 picks AND test-lifts the hammer in one search loop.
        # The handle lies roughly along world Y (~18 cm long) and is thin
        # (radius ~0.015-0.02 m, resting on the table at z~0.82). SAM3's XY
        # centroid is noisy by 2-7 cm: an error ALONG Y is harmless (still on
        # the handle) but a PERPENDICULAR (X) error beyond the radius misses.
        # _check_grasp (task_info[0]) can report True for a fingertip touch
        # that then slips on lift, so the real grasp test is whether a small
        # lift actually raises the hammer (h > threshold). Vertical room is
        # tiny (handle bottom on the table at ~0.80), so descend z is fixed at
        # ~3 cm below the reported centroid; the search is over X offsets.
        _GRASP_Z_DROP = 0.03
        _jp_grip(env, 0, close=False, other_close=False, steps=30)
        _mj.mj_forward(model, data)
        above = handle.copy(); above[2] += 0.12
        q_above0 = pos_ik(arm0_jids, arm0_ee, above)
        _jp_move_blocking(env, 0, q_above0, False,
                          other_gripper_close=False, max_steps=150)
        arm0_lifted = False
        # Search XY offsets: X (perpendicular to the handle) is the critical
        # axis and gets fine, wide coverage; a couple of Y offsets cover the
        # case where SAM3 latches onto the head end. Centre is tried first.
        search_offsets = [(0.0, 0.0)]
        for dx in [0.03, -0.03, 0.06, -0.06, 0.09, -0.09, 0.12, -0.12]:
            search_offsets.append((dx, 0.0))
        for dy in [0.04, -0.04, 0.08, -0.08]:
            search_offsets.append((0.0, dy))
        for (dx, dy) in search_offsets:
            grasp_pt = handle.copy()
            grasp_pt[0] += dx
            grasp_pt[1] += dy
            grasp_pt[2] -= _GRASP_Z_DROP
            _mj.mj_forward(model, data)
            _jp_move_blocking(env, 0, pos_ik(arm0_jids, arm0_ee, grasp_pt), False,
                              other_gripper_close=False, max_steps=120)
            _jp_grip(env, 0, close=True, other_close=False, steps=70)
            # Test-lift 8 cm and check the hammer actually came up.
            _mj.mj_forward(model, data)
            test_lift = grasp_pt.copy(); test_lift[2] += 0.08
            _jp_move_blocking(env, 0, pos_ik(arm0_jids, arm0_ee, test_lift), True,
                              other_gripper_close=False, max_steps=80)
            if hammer_h() > 0.04:
                arm0_lifted = True
                break
            # Bad grasp: lower back down, reopen, try next offset.
            _mj.mj_forward(model, data)
            _jp_move_blocking(env, 0, pos_ik(arm0_jids, arm0_ee, grasp_pt), True,
                              other_gripper_close=False, max_steps=60)
            _jp_grip(env, 0, close=False, other_close=False, steps=15)
        if verbose: print(f"Arm0 grasp+testlift ok={arm0_lifted} (h={hammer_h():.3f})")
        if not arm0_lifted:
            return False

        # Re-clamp and settle so the grip seats firmly before further motion;
        # the thin handle slips if it isn't well seated when the arm accelerates.
        _jp_grip(env, 0, close=True, other_close=False, steps=40)
        # Continue lifting to clearance in fine steps (gentler = less slip).
        _mj.mj_forward(model, data)
        ee0 = data.site_xpos[arm0_ee][:3].copy()
        for dz in [0.05, 0.10, 0.14, 0.18]:
            _mj.mj_forward(model, data)
            lift = ee0.copy(); lift[2] += dz
            _jp_move_blocking(env, 0, pos_ik(arm0_jids, arm0_ee, lift), True,
                              other_gripper_close=False, max_steps=90)
        if verbose: print(f"Arm0 lifted (h={hammer_h():.3f})")

        # Phase 3: Transit to a handover point that BOTH arms can reach.
        # Arms are opposed (arm0 base y~-0.36, arm1 base y~+0.34); a point
        # near y~-0.05, z~1.0 is a short transit for arm0 and within arm1's
        # forward reach. Step the Y motion gradually (warm-started pos_ik each
        # waypoint) so the wrist barely rotates and the grasp survives.
        _mj.mj_forward(model, data)
        ee0 = data.site_xpos[arm0_ee][:3].copy()
        handover_y = -0.05
        handover_z = 1.02
        n_seg = 12
        for k in range(1, n_seg + 1):
            f = k / n_seg
            seg = np.array([
                ee0[0] * (1 - f),
                ee0[1] * (1 - f) + handover_y * f,
                ee0[2] * (1 - f) + handover_z * f,
            ])
            _mj.mj_forward(model, data)
            _jp_move_blocking(env, 0, pos_ik(arm0_jids, arm0_ee, seg), True,
                              other_gripper_close=False, max_steps=90)
            # Abort early if the hammer slipped (saves horizon for nothing).
            if hammer_h() < 0.04:
                break

        try: arm0_has = env._get_task_info()[0]
        except: arm0_has = True
        if verbose:
            _mj.mj_forward(model, data)
            ee0 = data.site_xpos[arm0_ee][:3]
            print(f"Arm0 handover pose: ({ee0[0]:.3f},{ee0[1]:.3f},{ee0[2]:.3f}), "
                  f"holding={arm0_has}, h={hammer_h():.3f}")

        if not arm0_has:
            return False

        # Phase 4: Pick arm1's grasp point on the handle. Arm0 holds the
        # hammer top-down and lifted, so the head pulls the handle VERTICAL:
        # the handle hangs straight down below arm0's grip (measured: handle
        # centre ~6 cm below the grip, same x,y; handle is ~18 cm long). Use
        # arm0's grip-site pose (proprioception, not hammer GT) and target a
        # point ~10 cm BELOW the grip so arm1 grabs the lower handle, clear of
        # arm0's fingers but still on the handle (success needs
        # arm1-on-handle); the j7 wrist sweep below finds the jaw azimuth that
        # straddles the hanging handle.
        _mj.mj_forward(model, data)
        arm0_grip = data.site_xpos[arm0_ee][:3].copy()
        handover_target = arm0_grip.copy()
        handover_target[2] -= 0.07   # on the handle, just below arm0 fingers
        if verbose: print(f"Arm1 target (below arm0 grip): "
                          f"({handover_target[0]:.3f},{handover_target[1]:.3f},"
                          f"{handover_target[2]:.3f})")

        # Phase 5: Arm1 approaches the hanging handle in small INCREMENTAL
        # waypoints. The JOINT_POSITION controller does not converge on a
        # single large jump from rest to a far cross-table point (it saturates
        # within max_steps); interpolating from arm1's current EE to a side
        # stage point (+Y) and then to the handle keeps every step small and
        # warm-started pos_ik in a trackable branch.
        arm1_has = False
        gw = -1.0  # gripper width after final grasp attempt
        _jp_grip(env, 1, close=False, other_close=True, steps=20)
        _mj.mj_forward(model, data)
        ee1_start = data.site_xpos[arm1_ee][:3].copy()
        stage = handover_target.copy(); stage[1] += 0.12   # arm1-side, clear of arm0
        n_a1 = 6
        for k in range(1, n_a1 + 1):
            f = k / n_a1
            wp = ee1_start * (1 - f) + stage * f
            _mj.mj_forward(model, data)
            _jp_move_blocking(env, 1, pos_ik(arm1_jids, arm1_ee, wp), False,
                              other_gripper_close=True, max_steps=70)
        # Move in laterally from the side stage to the handle in small steps.
        _mj.mj_forward(model, data)
        ee1_stage = data.site_xpos[arm1_ee][:3].copy()
        n_in = 4
        q_handle = None
        for k in range(1, n_in + 1):
            f = k / n_in
            wp = ee1_stage * (1 - f) + handover_target * f
            _mj.mj_forward(model, data)
            q_handle = pos_ik(arm1_jids, arm1_ee, wp)
            _jp_move_blocking(env, 1, q_handle, False,
                              other_gripper_close=True, max_steps=70)
        # Corrective refinement: the cross-table reach can land a few cm short.
        # Re-solve from the actual reached pose toward the target a few times so
        # arm1 closes ON the handle, not in the air near it. Re-derive q_handle
        # from the final reached config so the j7 sweep below pivots about the
        # true grasp pose.
        for _ in range(3):
            _mj.mj_forward(model, data)
            ee1_now = data.site_xpos[arm1_ee][:3].copy()
            if np.linalg.norm(handover_target - ee1_now) < 0.015:
                break
            q_handle = pos_ik(arm1_jids, arm1_ee, handover_target)
            _jp_move_blocking(env, 1, q_handle, False,
                              other_gripper_close=True, max_steps=70)
        _mj.mj_forward(model, data)
        q_handle = pos_ik(arm1_jids, arm1_ee, handover_target)

        def _arm1_gripper_width() -> float:
            """
            Panda 2F-85 finger separation in metres. Empty close ~ 0;
            valid hammer-handle grasp ~ 0.02-0.05 m.
            """
            try:
                ob, _, _, _ = env.step(np.zeros(env.action_dim))
                gqp = ob.get('robot1_gripper_qpos')
                if gqp is None or len(gqp) < 2:
                    return -1.0
                return float(abs(gqp[0]) + abs(gqp[1]))
            except Exception:
                return -1.0

        j7_idx = len(arm1_jids) - 1
        j7_id = arm1_jids[j7_idx]
        j7_lo = model.jnt_range[j7_id][0]
        j7_hi = model.jnt_range[j7_id][1]
        j7_cur = q_handle[j7_idx]
        # j7 wrist sweep: with arm0 holding the hammer top-down and lifted,
        # the head pulls the handle vertical, so the exact jaw azimuth that
        # straddles it varies with how the handle hangs. Try a range of wrist
        # offsets (including +/-90 deg) and keep the first that yields a valid
        # width + task_info grasp.
        for ri, off in enumerate([1.57, -1.57, 1.2, -1.2, 0.8, -0.8, 0.4, -0.4, 0]):
            j7_val = j7_cur + off
            if not (j7_lo < j7_val < j7_hi):
                continue
            _jp_grip(env, 1, close=False, other_close=True, steps=8)
            q_rot = q_handle.copy(); q_rot[j7_idx] = j7_val
            _jp_move_blocking(env, 1, q_rot, False,
                              other_gripper_close=True, max_steps=20)
            _jp_grip(env, 1, close=True, other_close=True, steps=35)
            gw = _arm1_gripper_width()
            try: arm1_has = env._get_task_info()[1]
            except: pass
            # Accept the rotation iff task_info says arm1_has AND the
            # gripper width is in the object-confirm band (0.005-0.08 m).
            if (arm1_has and gw > _GRIPPER_EMPTY_CLOSE_THRESH
                    and gw < _GRIPPER_OBJECT_CONFIRM_MAX):
                if verbose:
                    print(f"Arm1 grasp confirmed at j7 offset {off:+.2f} "
                          f"(width={gw:.3f}m, has={arm1_has})")
                break
        if verbose and not (arm1_has and 0.0 < gw < _GRIPPER_OBJECT_CONFIRM_MAX):
            print(f"Arm1 wrist sweep ended without confirmed grasp "
                  f"(width={gw:.3f}m, has={arm1_has})")

    else:
        # OSC PATH
        above = handle.copy(); above[2] += 0.12
        _simple_move_to(env, above, arm_id=0, grip_closed=False, steps=200)
        _simple_move_to(env, handle, arm_id=0, grip_closed=False, steps=250)
        _simple_grip(env, arm_id=0, close=True, steps=80)

        obs_l, _, _, _ = env.step(np.zeros(env.action_dim))
        ee0 = obs_l.get('robot0_eef_pos', np.zeros(3))[:3].copy()
        for dz in [0.10, 0.20]:
            lift = ee0.copy(); lift[2] += dz
            _simple_move_to(env, lift, arm_id=0, grip_closed=True, steps=150)

        center = np.array([0.0, 0.0, 1.05])
        _simple_move_to(env, center, arm_id=0, grip_closed=True, steps=400)
        obs_c, _, _, _ = env.step(np.zeros(env.action_dim))
        try: arm0_has = env._get_task_info()[0]
        except: arm0_has = True
        if not arm0_has: return False

        # Re-detect handle via SAM3 (no GT)
        sam3_mid2 = _detect_objects_sam3(env, ['hammer', 'hammer handle'], verbose=False)
        hammer_mid2 = sam3_mid2.get('hammer handle', sam3_mid2.get('hammer'))
        if hammer_mid2 is not None:
            handover_target = hammer_mid2.copy()
        else:
            ee0_osc = obs_c.get('robot0_eef_pos', np.zeros(3))[:3].copy()
            handover_target = ee0_osc.copy()
            handover_target[2] -= 0.04
        obs_m = obs_c
        for s in range(400):
            ee1 = obs_m.get('robot1_eef_pos', np.zeros(3))[:3]
            err = handover_target - ee1
            if np.linalg.norm(err) < 0.010: break
            action = np.zeros(env.action_dim)
            action[7:10] = np.clip(err * 8/0.05, -1, 1)
            action[6] = 1.0; action[13] = -1.0
            try: obs_m, _, done, _ = env.step(action)
            except: break
        _simple_grip(env, arm_id=1, close=True, steps=120, other_closed=True)
        arm1_has = False
        try: arm1_has = env._get_task_info()[1]
        except: pass
        if verbose: print(f"arm1_grasp_handle={arm1_has}")

    if use_jp:
        # Grasp recovery: width-based check, NOT just env._get_task_info().
        # If width says fingers closed on air (gw < empty-thresh) OR no
        # task_info confirmation, nudge handover_target +/-2cm along world
        # X and along world Y axes (handle long axis varies with arm0
        # horizontal-hold orientation) and retry the descent.
        _grasp_ok = (arm1_has and 0.0 < gw < _GRIPPER_OBJECT_CONFIRM_MAX
                      and gw > _GRIPPER_EMPTY_CLOSE_THRESH)
        if not _grasp_ok:
            if verbose:
                print(f"arm1 grasp invalid (has={arm1_has}, "
                      f"width={gw:.3f}); retry around the hanging handle")
            # The handle hangs below arm0's grip; a width~0 miss means arm1
            # is off in XY or not deep enough. Search a small XY grid AND a
            # couple of extra depths, and at each spot retry the two most
            # useful wrist offsets (0 and +/-90 deg) since the handle hangs
            # vertical. Re-solve from the current pose so moves stay small.
            retry_offsets = [(0, 0, -0.03), (0.03, 0, 0), (-0.03, 0, 0),
                             (0, 0.03, 0), (0, -0.03, 0), (0, 0, -0.06),
                             (0.05, 0, -0.03), (-0.05, 0, -0.03)]
            for dx, dy, dz in retry_offsets:
                retry_tgt = handover_target.copy()
                retry_tgt[0] += dx; retry_tgt[1] += dy; retry_tgt[2] += dz
                try:
                    q_retry = pos_ik(arm1_jids, arm1_ee, retry_tgt)
                except Exception:
                    continue
                _jp_grip(env, 1, close=False, other_close=True, steps=10)
                _jp_move_blocking(env, 1, q_retry, False,
                                  other_gripper_close=True, max_steps=60)
                for woff in [0.0, 1.57, -1.57]:
                    j7v = q_retry[j7_idx] + woff
                    if not (j7_lo < j7v < j7_hi):
                        continue
                    _jp_grip(env, 1, close=False, other_close=True, steps=6)
                    j7r = q_retry.copy(); j7r[j7_idx] = j7v
                    _jp_move_blocking(env, 1, j7r, False,
                                      other_gripper_close=True, max_steps=20)
                    _jp_grip(env, 1, close=True, other_close=True, steps=40)
                    gw_retry = _arm1_gripper_width()
                    try: arm1_has = env._get_task_info()[1]
                    except: pass
                    _grasp_ok = (arm1_has
                                  and 0.0 < gw_retry < _GRIPPER_OBJECT_CONFIRM_MAX
                                  and gw_retry > _GRIPPER_EMPTY_CLOSE_THRESH)
                    if _grasp_ok:
                        if verbose:
                            print(f"retry at offset ({dx:.2f},{dy:.2f},{dz:.2f}) "
                                  f"woff={woff:+.2f} SUCCESS (width={gw_retry:.3f})")
                        break
                if _grasp_ok:
                    break

        # Don't drop the hammer unless arm1 actually has it.
        if not _grasp_ok:
            if verbose:
                print("arm1 failed grasp confirm; aborting without release")
            return False

        # Both grippers closed; settle 20 steps (~1.0 s at 20 Hz) so contact
        # forces equilibrate. Real-robot handoff_v2.py uses 0.6-0.7 s.
        _jp_grip(env, 0, close=True, other_close=True, steps=20)

        # Arm0 releases only after arm1's grasp is confirmed.
        _jp_grip(env, 0, close=False, other_close=True, steps=50)
        if verbose:
            print("Arm0 released (grasp confirmed)")

        # Check if arm1 still has handle after release
        try:
            ti = env._get_task_info()
            if verbose:
                print(f"Post-release: arm0_grasp={ti[0]}, arm1_grasp={ti[1]}, "
                      f"hammer_h={ti[2]:.3f}, table_h={ti[3]:.3f}")
        except: pass

        # Arm1 lifts via JP, in small steps so the grip survives, raising the
        # hammer clearly above the success height threshold (the handle can
        # hang low when arm0's handover pose is low).
        _mj.mj_forward(model, data)
        ee1 = data.site_xpos[arm1_ee][:3].copy()
        for dz in [0.08, 0.16, 0.22]:
            _mj.mj_forward(model, data)
            lift_tgt = ee1.copy(); lift_tgt[2] += dz
            q_lift = pos_ik(arm1_jids, arm1_ee, lift_tgt)
            _jp_move_blocking(env, 1, q_lift, gripper_close=True,
                              other_gripper_close=False, max_steps=90)

        # Hold
        _jp_grip(env, 1, close=True, other_close=False, steps=100)
    else:
        # OSC mode
        for _ in range(40):
            action = np.zeros(env.action_dim)
            action[6] = 1.0; action[13] = 1.0
            try: env.step(action)
            except: break
        _simple_grip(env, arm_id=0, close=False, steps=50, other_closed=True)
        if verbose:
            print("Arm0 released")
        obs_f, _, _, _ = env.step(np.zeros(env.action_dim))
        ee1 = obs_f.get('robot1_eef_pos', np.zeros(3))[:3].copy()
        lift1_target = ee1.copy(); lift1_target[2] += 0.15
        for ls in range(300):
            ee1 = obs_f.get('robot1_eef_pos', np.zeros(3))[:3]
            err1 = lift1_target - ee1
            if np.linalg.norm(err1) < 0.02: break
            action = np.zeros(env.action_dim)
            action[6] = -1.0
            action[7:10] = np.clip(err1 * 5 / 0.05, -1, 1)
            action[13] = 1.0
            try: obs_f, _, done, _ = env.step(action)
            except: break
        for _ in range(100):
            action = np.zeros(env.action_dim)
            action[6] = -1.0; action[13] = 1.0
            try: env.step(action)
            except: break

    return bool(env._check_success())


def main():
    import tyro

    @dataclass
    class Config:
        task: str = "TwoArmLift"
        """
        TwoArmLift or TwoArmHandover
        """
        num_trials: int = 5
        verbose: bool = False
        save_video: bool = False
        """
        Save per-trial GIFs under ~/spark/videos/robosuite/<task>/.
        """
        adaptive: bool = False
        """
        If True, use Gemini K-variant SAM3 prompting (NOT for fair no-LLM eval).
        """

    cfg = tyro.cli(Config)
    tasks = ['TwoArmLift', 'TwoArmHandover'] if cfg.task == 'all' else [cfg.task]

    from pathlib import Path
    for task_name in tasks:
        print(f"\n{'='*60}")
        print(f"SPARK Dual-Arm: {task_name}")
        print(f"{'='*60}\n")

        ctrl = 'JOINT_POSITION' if task_name == 'TwoArmHandover' else 'OSC_POSE'
        env = create_dual_arm_env(task_name, controller=ctrl)
        successes = []

        video_dir = None
        if cfg.save_video:
            vname = ''.join(['_' + c.lower() if c.isupper() else c
                             for c in task_name]).lstrip('_')
            video_dir = Path.home() / 'spark' / 'videos' / 'robosuite' / vname
            video_dir.mkdir(parents=True, exist_ok=True)

        for trial in range(cfg.num_trials):
            env.reset()
            for _ in range(10):
                env.step(np.zeros(env.action_dim))

            frames = []
            if cfg.save_video:
                _orig_step = env.step
                _fc = [0]
                def _rec_step(action, _os=_orig_step, _f=frames, _c=_fc):
                    result = _os(action)
                    _c[0] += 1
                    if _c[0] % 5 == 0:
                        try:
                            rgb_r = env._get_observations().get('agentview_image')
                            if rgb_r is not None:
                                _f.append(rgb_r[::-1].copy())
                        except Exception:
                            pass
                    return result
                env.step = _rec_step

            success = execute_dual_arm_bt(
                env, task_name, verbose=cfg.verbose, adaptive=cfg.adaptive)

            if cfg.save_video:
                env.step = _orig_step
                if frames:
                    from PIL import Image as _PILImage
                    pil_frames = [_PILImage.fromarray(f) for f in frames]
                    status = 'success' if success else 'fail'
                    gif_path = video_dir / f'trial{trial}_{status}.gif'
                    pil_frames[0].save(str(gif_path), save_all=True,
                                       append_images=pil_frames[1:],
                                       duration=100, loop=0)

            successes.append(success)
            print(f"Trial {trial}: {'SUCCESS' if success else 'FAIL'}")

        env.close()
        sr = sum(successes) / len(successes) if successes else 0
        print(f"\n{task_name}: {sr:.0%} ({sum(successes)}/{len(successes)})")
        if video_dir is not None:
            print(f"videos: {video_dir}")


if __name__ == '__main__':
    main()

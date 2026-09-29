"""
SPARK runner for BEHAVIOR-1K (R1Pro humanoid, mobile manipulation).

Compares vs CaP-Agent0 Table 3 (turning_on_radio, picking_up_trash).

Features:
  - SAM3 confidence floor (>=0.20)
  - Scan-then-drive: rotate in place if no detection in first N steps
  - Reach-right primitive: extend right arm forward+down with hand-coded delta
  - Pick sequence: when target mask gets large, stop nav + reach + close gripper
  - Per-trial debug image saved on first successful detection

Architecture:
  obs (head ZED RGB) -> SAM3 detection -> proprio (eef/base) ->
    bearing-following nav -> approach -> reach + grip -> info["task_progress"]

Run via:
  ~/vla_interp/Isaac-GR00T/.venv/bin/python -u \
      spark_bench/run_spark_behavior.py --task turning_on_radio --num-trials 3
"""
from __future__ import annotations

import os
import sys
import threading
import time
import traceback
import json
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import tyro

try:
    import torch
    from PIL import Image, ImageDraw
except ImportError:
    torch = None
    Image = ImageDraw = None

try:
    import gymnasium as gym
except ImportError:
    gym = None

os.environ.setdefault('OMNIGIBSON_HEADLESS', '1')

# The Isaac Sim / gr00t / omnigibson stack and SAM3 (SPARKPerception) are
# imported inside functions on purpose: Isaac Sim must initialize before SAM3
# loads (pre-loading SAM3 crashes Isaac), and Gemini must run before the gr00t
# import (Isaac's bundled openssl breaks google-genai's cryptography backend).

# Make spark_real / spark_bench importable when running under gr00t .venv
SPARK_SRC = Path(__file__).resolve().parent.parent
if str(SPARK_SRC) not in sys.path:
    sys.path.insert(0, str(SPARK_SRC))


@dataclass
class BehaviorConfig:
    task: str = 'turning_on_radio'
    num_trials: int = 3
    max_steps_per_trial: int = 800  # half of human-2x default; long enough for nav+pick
    verbose: bool = False
    use_sam3: bool = True            # set False to skip perception (debug nav only)
    log_dir: str = str(Path(__file__).resolve().parent / 'results' / 'behavior_logs')

    # Perception / scan knobs
    # SAM3 on small distant objects (radio, soda can) tends to give low conf
    # (0.05-0.15). 0.10 keeps obvious noise out while letting real targets through.
    sam3_min_conf: float = 0.10
    scan_after_steps: int = 30       # if no detection by this step, start rotating
    scan_max_steps: int = 120        # rotate for up to this many steps before giving up
    detect_every: int = 12           # re-run SAM3 every K steps when in nav loop
    approach_mask_area: int = 4000   # mask area in pixels (default 256x256 head res)
    head_res: int = 256              # head ZED resolution; bumping breaks obs_space (kept default)
    reach_steps: int = 30            # how many steps to extend the arm
    grip_close_steps: int = 18       # how many steps to hold gripper close
    log_subthreshold: bool = True    # log SAM3 detections even if below conf threshold
    K: int = 3                       # number of Gemini variant prompts
    model: str = 'gemini-3-flash-preview'  # variant-gen model; try 'gemini-3-pro-preview'
    reach_preset: str = 'default'    # default/extended/fwd_tilt/deep/huge/floor/forward
    out_tag: str = ''                # suffix appended to results_<task>_<tag>.json
    torso_lean: float = 0.0          # rad to lean torso forward during reach (0..0.5)
    creep_vx: float = 0.0            # m/s forward base creep during first reach steps
    creep_steps: int = 0             # how many reach steps to creep forward


# R1Pro action helpers
#
# IMPORTANT: BEHAVIOR-1K R1Pro arm controller is JointController with
# motor_type=position, use_delta_commands=False. So action.right_arm is the
# *absolute target qpos*, NOT a delta. Sending zeros every step would drive
# the arm to qpos=0 (dead pose). All actions below should preserve the
# current arm pose by default; reach primitives explicitly set new targets.
# Source: gr00t/eval/sim/BEHAVIOR/og_teleop_cfg.py (R1_CONTROLLER_CONFIG).
def hold_arms_action(env, obs: dict | None) -> dict:
    """
    Action that holds both arms + torso at their current qpos.

    Critical: R1Pro arm_left/right and trunk are JointController with
    motor_type=position, use_delta_commands=False, so an action of zeros
    drives them to qpos=0 (dead pose). We must explicitly send current qpos
    every step to keep them stationary.

    If obs is None, falls back to zeros (used only before first reset()).
    """
    a = {k: np.zeros(s.shape, dtype=np.float32) for k, s in env.action_space.items()}
    if obs is not None:
        if 'state.arm_right_qpos' in obs:
            a['action.right_arm'] = np.asarray(obs['state.arm_right_qpos'],
                                               dtype=np.float32)
        if 'state.arm_left_qpos' in obs:
            a['action.left_arm'] = np.asarray(obs['state.arm_left_qpos'],
                                              dtype=np.float32)
        if 'state.trunk_qpos' in obs and 'action.torso' in a:
            a['action.torso'] = np.asarray(obs['state.trunk_qpos'],
                                           dtype=np.float32)
    return a


def zero_action(env) -> dict:
    """
    Backwards-compat: zeros for everything (used pre-reset only).
    """
    return {k: np.zeros(s.shape, dtype=np.float32) for k, s in env.action_space.items()}


def base_velocity_action(env, vx: float, vy: float, vw: float, *,
                         obs: dict | None = None) -> dict:
    """
    Base velocity action that holds arms at current qpos.
    """
    a = hold_arms_action(env, obs)
    a['action.base'] = np.array([vx, vy, vw], dtype=np.float32)
    return a


def arm_target_action(env, side: str, target_qpos: np.ndarray, *,
                      grip: float = 0.0, obs: dict | None = None) -> dict:
    """
    Action with one arm absolute-position target and a gripper command
    (-1 close, +1 open). Other arm held at its current qpos.
    """
    assert side in ('left', 'right')
    a = hold_arms_action(env, obs)
    a[f'action.{side}_arm'] = target_qpos.astype(np.float32)
    a[f'action.{side}_gripper'] = np.array([grip], dtype=np.float32)
    return a


# Primitives
def open_gripper(env, side: str = 'right', *, steps: int = 8,
                 obs: dict | None = None) -> dict:
    last_obs = obs
    for _ in range(steps):
        a = arm_target_action(
            env, side,
            np.asarray(last_obs[f'state.arm_{side}_qpos'], dtype=np.float32),
            grip=+1.0, obs=last_obs)
        last_obs, *_ = env.step(a)
    return last_obs


def close_gripper(env, side: str = 'right', *, steps: int = 12,
                  obs: dict | None = None) -> tuple:
    """
    Hold arm at its current qpos and close gripper.
    """
    last_obs = obs
    info: dict = {}
    for _ in range(steps):
        a = arm_target_action(
            env, side,
            np.asarray(last_obs[f'state.arm_{side}_qpos'], dtype=np.float32),
            grip=-1.0, obs=last_obs)
        last_obs, _, term, trunc, info = env.step(a)
        if term or trunc:
            break
    return last_obs, info


def hold(env, *, steps: int = 5, obs: dict | None = None) -> dict:
    last_obs = obs
    for _ in range(steps):
        a = hold_arms_action(env, last_obs)
        last_obs, *_ = env.step(a)
    return last_obs


# Hand-coded right-arm pre-grasp delta (forward + down).
# R1Pro right_arm has 7 joints; without IK, a small positive delta on
# shoulder pitch + elbow extends the arm forward+down.
# Joint order convention (typical R1Pro right_arm, shoulder->wrist):
#   [shoulder_pitch, shoulder_roll, shoulder_yaw, elbow, wrist_yaw, wrist_pitch, wrist_roll]
PREGRASP_DELTA_RIGHT = np.array(
    [+0.06,   # shoulder pitch (lower arm down/forward)
     -0.04,   # shoulder roll (close to torso)
     0.00,
     +0.05,   # elbow (bend slightly to reach)
     0.00,
     +0.03,   # wrist pitch (point forward)
     0.00], dtype=np.float32)


REACH_PRESETS = {
    # baseline (tiny; ineffective with absolute-position controller)
    'default': PREGRASP_DELTA_RIGHT,
    # bigger shoulder pitch + bigger elbow -> arm extends further forward+down
    'extended': np.array(
        [+0.10, -0.04, 0.00, +0.08, 0.00, +0.05, 0.00], dtype=np.float32),
    # forward tilt: bigger shoulder pitch, neutral elbow, more wrist pitch
    'fwd_tilt': np.array(
        [+0.12, -0.03, 0.00, +0.03, 0.00, +0.06, 0.00], dtype=np.float32),
    # deep: heavy on shoulder pitch + elbow, drives hand low+forward (for floor pickup)
    'deep': np.array(
        [+0.14, -0.02, 0.00, +0.10, 0.00, +0.04, 0.00], dtype=np.float32),
    # huge: ~30deg shoulder pitch + ~30deg elbow, big swing forward+down
    'huge': np.array(
        [+0.55, -0.10, 0.00, +0.50, 0.00, +0.20, 0.00], dtype=np.float32),
    # floor: aggressively reach to floor (radio/trash on ground)
    'floor': np.array(
        [+0.80, -0.05, 0.00, +0.40, 0.00, +0.15, 0.00], dtype=np.float32),
    # forward: arm extended straight forward (table-height objects)
    'forward': np.array(
        [+0.30, -0.15, 0.00, +0.15, 0.00, +0.10, 0.00], dtype=np.float32),
}


def reach_right(env, *, steps: int = 30, grip: float = +1.0,
                obs: dict | None = None,
                delta: np.ndarray | None = None,
                torso_lean: float = 0.0,
                creep_vx: float = 0.0,
                creep_steps: int = 0) -> tuple[dict, dict]:
    """
    Extend right arm forward+down by setting an absolute qpos target.

    target_qpos = current_arm_right_qpos + delta. This delta is "what we want
    the arm to move by, total" - NOT a per-step delta (the controller is
    absolute-position). Returns (last_obs, last_info).

    `torso_lean` (rad): pitches the torso forward by this amount on the first
    torso joint. Bigger torso_lean -> further reach forward.
    `creep_vx`: forward base velocity m/s during the first `creep_steps` steps
    (helps close the last 0.2-0.5m gap to satisfy the 'near' progress check).
    """
    if delta is None:
        delta = PREGRASP_DELTA_RIGHT
    if obs is None:
        raise ValueError('reach_right requires obs to read current arm qpos')
    current = np.asarray(obs['state.arm_right_qpos'], dtype=np.float32)
    target = current + delta.astype(np.float32)
    torso_current = np.asarray(obs.get('state.trunk_qpos',
                                       np.zeros(4, dtype=np.float32)),
                               dtype=np.float32)
    torso_target = torso_current.copy()
    if abs(torso_lean) > 1e-6 and len(torso_target) > 0:
        torso_target[0] = torso_current[0] + float(torso_lean)
    last_info = {}
    last_obs = obs
    for i in range(steps):
        a = arm_target_action(env, 'right', target, grip=grip, obs=last_obs)
        a['action.torso'] = torso_target.astype(np.float32)
        if i < creep_steps and abs(creep_vx) > 1e-6:
            a['action.base'] = np.array([creep_vx, 0.0, 0.0], dtype=np.float32)
        last_obs, _, term, trunc, info = env.step(a)
        last_info = info or {}
        if term or trunc:
            break
    return last_obs, last_info


# Perception: SAM3 on head camera
_SAM3 = None
def _get_sam3():
    global _SAM3
    # Re-init if we cached an instance whose underlying model failed to load
    # (e.g. CUDA OOM on the first attempt).
    if _SAM3 is not None and getattr(_SAM3, '_sam3', None) is None:
        _SAM3 = None
    if _SAM3 is None:
        # SPARKPerception import deferred until after Isaac Sim init (module head).
        from spark_real.perception.spark_perception import SPARKPerception
        sp = SPARKPerception()
        sp.load_models(load_da3=False)
        if getattr(sp, '_sam3', None) is None:
            raise RuntimeError('SAM3 load_models did not populate _sam3 (likely OOM)')
        _SAM3 = sp
    return _SAM3


def detect_object_2d(obs: dict, prompt: str, *, min_conf: float = 0.10,
                     return_subthreshold: bool = False) -> dict | None:
    """
    Run SAM3 on head ZED RGB; return 2D detection (no DA3 depth).

    Drops detections below `min_conf` unless `return_subthreshold` is True
    (in which case the result dict includes 'subthreshold': True).
    """
    rgb = obs['video.observation.images.rgb.head_256_256']  # name is legacy; actual size depends on sensor
    sam3 = _get_sam3()
    pil_img = Image.fromarray(rgb)
    img_h, img_w = rgb.shape[:2]
    state = sam3._sam3.set_image(pil_img)
    state = sam3._sam3.set_text_prompt(prompt=prompt, state=state)
    masks = state.get('masks', torch.tensor([]))
    scores = state.get('scores', torch.tensor([]))
    if masks.numel() == 0:
        return None
    best_idx = int(scores.argmax())
    conf = float(scores[best_idx])
    subthreshold = conf < min_conf
    if subthreshold and not return_subthreshold:
        return None
    mask = masks[best_idx].cpu().numpy().squeeze()
    ys, xs = np.where(mask > 0)
    if len(xs) == 0:
        return None
    cx, cy = float(xs.mean()), float(ys.mean())
    return {
        'mask': mask,
        'centroid_uv': (cx, cy),
        'confidence': conf,
        'mask_area': int(len(xs)),
        'mask_area_frac': float(len(xs)) / float(img_w * img_h),
        'phrase': prompt,
        'rgb': rgb,
        'img_w': img_w,
        'img_h': img_h,
        'subthreshold': subthreshold,
    }


def bump_head_resolution(env, target_res: int = 1024) -> int:
    """
    Bump head ZED + wrist sensors AND gym observation_space to target_res.

    gr00t's RGBLowResWrapper caps cameras at 256x256 for VLA training. SAM3
    needs higher res to detect small objects (radio conf 0.0 -> 0.36 from
    256 -> 1024 in our resolution sweep).

    Two-step patch: (1) set sensor.image_height/width on every camera sensor;
    (2) replace the corresponding gym Box subspaces in env.observation_space
    so env.step() doesn't raise 'Observation space does not match'.

    Returns the actual resolution applied.
    """
    if target_res == 256:
        return 256
    robot = env.env.robots[0] if hasattr(env, 'env') else env.robots[0]
    bumped = []
    for name, sensor in robot.sensors.items():
        if any(k in name.lower() for k in ('zed', 'realsense')):
            sensor.image_height = target_res
            sensor.image_width = target_res
            bumped.append(name)
    if not bumped:
        raise RuntimeError('no zed/realsense sensors found in robot.sensors')
    print(f'[res-bump] sensors {bumped} -> {target_res}x{target_res}', flush=True)
    # 1. Tell the INNER og.Environment to rebuild its observation_space from
    #    the sensor configs (new image_height/width). Without this,
    #    og.Environment.reset() raises "Observation space does not match".
    inner = env.env if hasattr(env, 'env') else env
    if hasattr(inner, 'load_observation_space'):
        inner.load_observation_space()
        print(f'[res-bump] inner env load_observation_space() called', flush=True)
    # 2. Also patch the gr00t wrapper's observation_space (legacy 256_256 keys)
    if hasattr(env, 'observation_space') and hasattr(env.observation_space, 'spaces'):
        for k in ('video.observation.images.rgb.head_256_256',
                  'video.observation.images.rgb.left_wrist_256_256',
                  'video.observation.images.rgb.right_wrist_256_256'):
            if k in env.observation_space.spaces:
                env.observation_space.spaces[k] = gym.spaces.Box(
                    low=0, high=255, shape=(target_res, target_res, 3),
                    dtype=np.uint8)
        print(f'[res-bump] patched gr00t wrapper obs_space for 3 RGB cameras', flush=True)
    return target_res


def enable_seg_instance(env) -> None:
    """
    Add seg_instance modality to head ZED so we can resolve SAM3 mask
    centroids to OmniGibson scene objects. Same approach CaP-Agent0 uses.
    """
    robot = env.env.robots[0] if hasattr(env, 'env') else env.robots[0]
    for name, sensor in robot.sensors.items():
        if 'zed' in name.lower():
            try:
                cur = set(sensor.modalities) if hasattr(sensor, 'modalities') else set()
                cur.add('seg_instance')
                sensor.add_modality('seg_instance')
                print(f'[seg] head sensor {name} modalities: {sorted(sensor.modalities)}',
                      flush=True)
            except Exception as e:
                print(f'[seg] failed to add seg_instance to {name}: {e}', flush=True)


def resolve_det_to_scene_object(env, det: dict, *, allow_keywords=None):
    """
    Look up the OmniGibson scene object at the SAM3 mask centroid via the
    head camera's seg_instance map. Returns the object handle, or None if no
    valid object can be resolved.

    `allow_keywords`: if given, the resolved instance_name must contain at
    least one of these substrings (case-insensitive). Used to reject obvious
    SAM3 false positives like the radio variants matching a TV/fireplace.
    """
    # omnigibson import deferred until after Isaac Sim init (see module head).
    from omnigibson.sensors.vision_sensor import VisionSensor
    robot = env.env.robots[0] if hasattr(env, 'env') else env.robots[0]
    head_sensor = next((s for n, s in robot.sensors.items() if 'zed' in n.lower()), None)
    if head_sensor is None:
        return None
    sensor_obs, _ = head_sensor.get_obs()
    seg = sensor_obs.get('seg_instance')
    if seg is None:
        return None
    seg_np = seg.cpu().numpy() if hasattr(seg, 'cpu') else np.asarray(seg)
    cx, cy = det['centroid_uv']
    h, w = seg_np.shape[:2]
    cx_i = max(0, min(int(cx), w - 1))
    cy_i = max(0, min(int(cy), h - 1))
    inst_id = int(seg_np[cy_i, cx_i])
    inst_name = VisionSensor.INSTANCE_REGISTRY.get(inst_id)
    if inst_name is None or inst_name in ('background', 'unlabelled'):
        return None
    if allow_keywords:
        name_lower = inst_name.lower()
        # Reject if instance is a structural element (wall, floor, ceiling, etc.)
        # OR if no positive keyword overlap.
        STRUCTURAL = ('wall_', 'floor_', 'ceiling_', 'window_', 'door_',
                      'baseboard_', 'roof_', 'driveway_', 'lawn_')
        if any(s in name_lower for s in STRUCTURAL):
            return None
        # Otherwise accept if any keyword matches the instance name
        if not any(kw.lower() in name_lower for kw in allow_keywords):
            return None  # name mismatch, reject
    scene = env.env.scene if hasattr(env, 'env') else env.scene
    obj = scene.object_registry('name', inst_name)
    return obj


_OG_CONTROLLER = None
def get_og_controller(env):
    """
    Build StarterSemanticActionPrimitives on first use - same controller CaP-X
    uses. Provides cuRobo-based GRASP / NAVIGATE_TO / TOGGLE_ON primitives.
    """
    global _OG_CONTROLLER
    if _OG_CONTROLLER is None:
        # omnigibson import deferred until after Isaac Sim init (see module head).
        from omnigibson.action_primitives.starter_semantic_action_primitives import (
            StarterSemanticActionPrimitives,
        )
        inner = env.env if hasattr(env, 'env') else env
        robot = inner.robots[0]
        _OG_CONTROLLER = StarterSemanticActionPrimitives(
            inner, robot, enable_head_tracking=False)
        print('[og-ctrl] StarterSemanticActionPrimitives ready (cuRobo-backed)',
              flush=True)
    return _OG_CONTROLLER


def _torch_action_to_dict(th_action) -> dict:
    """
    Inverse of gr00t preprocess_action: torch tensor -> action.* dict.
    """
    arr = th_action.cpu().numpy() if hasattr(th_action, 'cpu') else np.asarray(th_action)
    return {
        'action.base': arr[0:3].astype(np.float32),
        'action.torso': arr[3:7].astype(np.float32),
        'action.left_arm': arr[7:14].astype(np.float32),
        'action.left_gripper': arr[14:15].astype(np.float32),
        'action.right_arm': arr[15:22].astype(np.float32),
        'action.right_gripper': arr[22:23].astype(np.float32),
    }


def grasp_via_omnigibson(env, target_obj, *, max_steps: int = 400) -> tuple:
    """
    Run StarterSemanticActionPrimitives.GRASP on `target_obj`, stepping
    actions through the gr00t wrapper so task_progress is tracked.
    Returns (last_obs, last_info).
    """
    # omnigibson import deferred until after Isaac Sim init (see module head).
    from omnigibson.action_primitives.starter_semantic_action_primitives import (
        StarterSemanticActionPrimitiveSet,
    )
    controller = get_og_controller(env)
    last_obs, last_info = None, {}
    n = 0
    try:
        for action in controller.apply_ref(
                StarterSemanticActionPrimitiveSet.GRASP, target_obj, attempts=3):
            if action is None or n >= max_steps:
                break
            d = _torch_action_to_dict(action)
            last_obs, _, term, trunc, last_info = env.step(d)
            n += 1
            if term or trunc:
                break
    except Exception as e:
        print(f'[og-grasp] primitive raised: {type(e).__name__}: {e}',
              flush=True)
    print(f'[og-grasp] {n} steps consumed', flush=True)
    return last_obs, last_info


def head_camera_direction_in_base_frame(obs: dict, det: dict) -> tuple[float, float]:
    """
    Convert SAM3 pixel centroid to (forward, lateral) direction in base frame.

    Head ZED at horizontal_aperture=40 (set by RGBLowResWrapper) -> ~67 deg HFOV.
    """
    cx, _ = det['centroid_uv']
    # Image width from the detection when present, else the 1024 head res.
    img_w = det.get('img_w', 1024)
    H_FOV_DEG = 67.0
    bearing = (cx - img_w / 2.0) / (img_w / 2.0) * np.deg2rad(H_FOV_DEG / 2)
    forward = float(np.cos(bearing))
    lateral = float(-np.sin(bearing))
    return forward, lateral


def save_debug_image(rgb: np.ndarray, det: dict, path: str) -> None:
    """
    Save head RGB with detection overlay as PNG.
    """
    try:
        rgb_u8 = np.asarray(rgb, dtype=np.uint8)
        img = Image.fromarray(rgb_u8).convert('RGB')
        # overlay mask in red
        mask = det['mask'] > 0
        if mask.any():
            arr = np.array(img)
            arr[mask] = (0.5 * arr[mask] + 0.5 * np.array([255, 0, 0])).astype(np.uint8)
            img = Image.fromarray(arr)
        draw = ImageDraw.Draw(img)
        cx, cy = det['centroid_uv']
        draw.ellipse([cx - 4, cy - 4, cx + 4, cy + 4], outline='yellow', width=2)
        draw.text((4, 4),
                  f"{det['phrase']} c={det['confidence']:.2f} a={det['mask_area']}",
                  fill='yellow')
        img.save(path)
    except Exception as e:
        print(f'[debug] failed to save {path}: {e}', flush=True)


# Trial loop
def _ensure_gemini_key():
    if 'GEMINI_API_KEY' not in os.environ:
        key_file = os.path.expanduser(os.environ.get('SPARK_GEMINI_KEY_FILE', '~/spark/src/.gemini_api_key'))
        if os.path.exists(key_file):
            with open(key_file) as f:
                lines = [ln.strip() for ln in f.readlines() if ln.strip()]
            if lines:
                os.environ['GEMINI_API_KEY'] = lines[0]


def derive_target_variants_via_gemini(task_instruction: str, K: int = 3,
                                      timeout_s: float = 30.0,
                                      model: str = 'gemini-3-flash-preview'
                                      ) -> list[str]:
    """
    Adaptive: K visually-distinct phrasings of the target object.

    Two phases on top of perception (Phase 1 = variant gen, Phase 2 = SAM3
    select), the same recipe as the LIBERO-PRO adaptive runs. The ONLY
    input is the task instruction - no per-task lookup, no scene state.
    Wraps Gemini call in a thread with `timeout_s` (default 30s; Pro models
    take longer than flash); if the call hangs, falls back to the
    instruction-only heuristic.  Returns list of length K.
    """
    _ensure_gemini_key()

    result = {'value': None, 'err': None}

    def _do_gemini():
        try:
            from spark_real.planning.spark_planner import SPARKPlanner
            planner = SPARKPlanner(llm_backend='gemini')
            client = planner._get_client()
            prompt = (
                f"You are picking SAM3 detection phrases for a robot task.\n"
                f"Task: \"{task_instruction}\"\n"
                f"Output exactly {K} short noun phrases (1-3 words each), one per "
                f"line, no numbering, no quotes, no extra text. Each phrase should "
                f"be a *different visual description* of the same primary target "
                f"object (color, shape, function, common name, brand). "
                f"Order: most-specific first, most-generic last."
            )
            try:
                from google.genai import types as _gt
                resp = client.models.generate_content(
                    model=model, contents=prompt,
                    config=_gt.GenerateContentConfig(temperature=0))
            except Exception:
                resp = client.models.generate_content(
                    model=model, contents=prompt,
                    config={'temperature': 0})
            raw = (resp.text or '').strip()
            lines = [ln.strip().strip('"\'`-*-').rstrip('.').strip()
                     for ln in raw.splitlines() if ln.strip()]
            seen, out = set(), []
            for ln in lines:
                if ln and ln.lower() not in seen:
                    seen.add(ln.lower()); out.append(ln)
            if not out:
                result['value'] = None
                return
            while len(out) < K:
                out.append(out[0])
            result['value'] = out[:K]
        except Exception as e:
            result['err'] = e

    th = threading.Thread(target=_do_gemini, daemon=True)
    th.start()
    th.join(timeout=timeout_s)
    if th.is_alive():
        print(f'[variant-derive] Gemini timed out after {timeout_s}s; '
              f'using instruction fallback', flush=True)
        return _instruction_only_fallback(task_instruction, K)
    if result['err'] is not None:
        print(f'[variant-derive] Gemini failed: {result["err"]}; '
              f'using instruction fallback', flush=True)
        return _instruction_only_fallback(task_instruction, K)
    if result['value'] is None:
        return _instruction_only_fallback(task_instruction, K)
    return result['value']


def _instruction_only_fallback(task_instruction: str, K: int) -> list[str]:
    """
    Fallback when Gemini is unavailable: extract candidate phrases from the
    natural-language task instruction *only* (no scene/BDDL access).

    Heuristic: take the last noun-like content word, plus generic synonyms.
    This is still 'fair' under the anti-cheat rule because the same code path
    runs for any task.
    """
    txt = (task_instruction or '').strip().rstrip('.').lower()
    stops = {
        'a', 'an', 'the', 'on', 'in', 'to', 'of', 'and', 'or', 'with', 'into',
        'pick', 'up', 'put', 'turn', 'turning', 'place', 'move', 'set', 'get',
        'is', 'are', 'be', 'do', 'does', 'has', 'have',
    }
    words = [w for w in txt.replace(',', ' ').split() if w and w not in stops]
    primary = words[-1] if words else 'object'
    # Two-word fallback (e.g. "soda can")
    if len(words) >= 2:
        bigram = ' '.join(words[-2:])
    else:
        bigram = primary
    out = [primary, bigram, 'object']
    # dedup preserving order
    seen, dedup = set(), []
    for v in out:
        if v not in seen:
            seen.add(v); dedup.append(v)
    while len(dedup) < K:
        dedup.append('object')
    return dedup[:K]


def detect_object_adaptive(obs: dict, variants: list[str], *,
                           min_conf: float = 0.20,
                           dedup_px: float = 30.0,
                           verbose: bool = False) -> dict | None:
    """
    Phase 2: run SAM3 for each variant on the SAME head RGB; select winner
    by mask_area * confidence, with 30px centroid dedup so two variants that
    pick the same physical region get their areas summed (boost cardinality).
    """
    cands = []
    for v in variants:
        d = detect_object_2d(obs, v, min_conf=min_conf)
        if d is not None:
            cands.append(d)
            if verbose:
                print(f'[variant {v!r}] conf={d["confidence"]:.2f} '
                      f'area={d["mask_area"]} centroid=({d["centroid_uv"][0]:.0f},'
                      f'{d["centroid_uv"][1]:.0f})', flush=True)
    if not cands:
        return None
    used = [False] * len(cands)
    best, best_score = None, -1.0
    for i, ci in enumerate(cands):
        if used[i]:
            continue
        cluster_area = ci['mask_area']
        cluster_conf = ci['confidence']
        cx, cy = ci['centroid_uv']
        for j in range(i + 1, len(cands)):
            if used[j]:
                continue
            cj = cands[j]
            dx = cj['centroid_uv'][0] - cx
            dy = cj['centroid_uv'][1] - cy
            if (dx * dx + dy * dy) ** 0.5 < dedup_px:
                cluster_area += cj['mask_area']
                cluster_conf = max(cluster_conf, cj['confidence'])
                used[j] = True
        score = cluster_area * cluster_conf
        if score > best_score:
            best_score = score
            best = dict(ci)
            best['mask_area'] = cluster_area
            best['confidence'] = cluster_conf
    return best


def run_trial(env, cfg: BehaviorConfig, trial_idx: int,
              variants: list[str],
              reach_delta: np.ndarray | None = None,
              torso_lean: float = 0.0,
              creep_vx: float = 0.0,
              creep_steps: int = 0) -> dict:
    """
    One trial: reset, run adaptive SAM3 K=len(variants) -> nav + reach + grasp.

    `variants` is the K-phrase list from Phase 1 (Gemini variant-gen),
    passed in from main so the Gemini call is paid once per task.
    `reach_delta` overrides the default PREGRASP_DELTA_RIGHT.
    """
    print(f'\ntrial {trial_idx}', flush=True)
    obs, info = env.reset()
    print(f'task: {env.task_instruction}', flush=True)
    print(f'starting base: {obs["state.base_qpos"]}', flush=True)
    print(f'variants (K={len(variants)}): {variants}', flush=True)

    last_bearing = None  # (forward, lateral)
    last_mask_area = 0
    last_det = None       # most recent SAM3 detection dict (for object resolve)
    saved_debug = False
    success = False
    info = info or {}
    phase = 'nav'  # nav -> approach -> reach -> done

    # Track scan state: if no detection by `scan_after_steps`, rotate base.
    first_det_step = None
    scan_started = None  # step when scanning started

    for step in range(cfg.max_steps_per_trial):
        # perception every K steps (when in nav/approach phase)
        if cfg.use_sam3 and phase in ('nav', 'approach') and (step % cfg.detect_every == 0):
            det = detect_object_adaptive(obs, variants,
                                         min_conf=cfg.sam3_min_conf,
                                         verbose=cfg.verbose)
            if det is not None:
                sub = False  # adaptive doesn't return subthreshold
                if cfg.verbose:
                    print(f'[t={step}] WIN conf={det["confidence"]:.2f} '
                          f'centroid=({det["centroid_uv"][0]:.0f},{det["centroid_uv"][1]:.0f}) '
                          f'area={det["mask_area"]}',
                          flush=True)
                if not sub:
                    if first_det_step is None:
                        first_det_step = step
                    forward, lateral = head_camera_direction_in_base_frame(obs, det)
                    last_bearing = (forward, lateral)
                    last_mask_area = det['mask_area']
                    last_det = det
                    if not saved_debug:
                        out = Path(cfg.log_dir) / f'dbg_{cfg.task}_t{trial_idx}.png'
                        save_debug_image(det['rgb'], det, str(out))
                        saved_debug = True
                    # Switch to approach if mask is big enough
                    if last_mask_area >= cfg.approach_mask_area:
                        phase = 'approach'

        # decide action based on phase
        if phase == 'nav':
            if last_bearing is None:
                # Scan-then-drive: stay still for a bit, then rotate
                if step < cfg.scan_after_steps:
                    a = hold_arms_action(env, obs)
                else:
                    if scan_started is None:
                        scan_started = step
                    if step - scan_started > cfg.scan_max_steps:
                        # gave up - try driving forward slowly to see if it helps
                        a = base_velocity_action(env, 0.2, 0.0, 0.0, obs=obs)
                    else:
                        # rotate in place to scan
                        a = base_velocity_action(env, 0.0, 0.0, 0.4, obs=obs)
            else:
                forward, lateral = last_bearing
                vx = float(np.clip(0.4 * forward, 0.0, 0.5))
                vw = float(np.clip(-1.5 * lateral, -1.0, 1.0))
                a = base_velocity_action(env, vx, 0.0, vw, obs=obs)
            obs, _, term, trunc, info = env.step(a)
            if term or trunc:
                success = bool(info.get('success', False))
                break

        elif phase == 'approach':
            # Resolve SAM3 detection -> OmniGibson scene object via instance
            # segmentation, then run the GRASP primitive (cuRobo IK +
            # collision-aware planning). Same primitive CaP-Agent0 uses.
            print(f'[t={step}] APPROACH: mask_area={last_mask_area}, '
                  f'resolving SAM3 -> scene object', flush=True)
            # Extract last-token nouns from variants for sanity-check filter
            allow_kw = []
            for v in variants:
                for tok in v.split():
                    t = tok.strip(' .,;:').lower()
                    if len(t) >= 4 and not t.startswith(('the', 'a', 'an', 'small', 'large', 'black', 'red', 'blue', 'green', 'yellow', 'white', 'silver', 'metal', 'plastic')):
                        allow_kw.append(t)
            allow_kw = list(set(allow_kw))
            target_obj = resolve_det_to_scene_object(env, last_det,
                                                    allow_keywords=allow_kw)
            if target_obj is None:
                print(f'[t={step}] resolve rejected (instance name not in '
                      f'{allow_kw}); skipping', flush=True)
                phase = 'nav'  # try again with another detection
                last_bearing = None
                continue
            if target_obj is None:
                print(f'[t={step}] could not resolve SAM3 mask to scene '
                      f'object; falling back to hand-coded reach', flush=True)
                obs, info = reach_right(env, steps=cfg.reach_steps, grip=+1.0,
                                        delta=reach_delta, obs=obs,
                                        torso_lean=torso_lean,
                                        creep_vx=creep_vx,
                                        creep_steps=creep_steps)
                phase = 'pick'
                continue
            print(f'[t={step}] resolved -> {target_obj.name!r}; running PyRoki grasp',
                  flush=True)
            try:
                from pyroki_r1pro import grasp_via_pyroki
                obs, info = grasp_via_pyroki(env, target_obj, verbose=cfg.verbose)
            except Exception as e:
                print(f'[t={step}] PyRoki grasp raised: {type(e).__name__}: {e}',
                      flush=True)
                traceback.print_exc()
                info = {}
            success = bool((info or {}).get('success', False))
            phase = 'done'
            continue

        elif phase == 'pick':
            # Reached only on the fallback hand-coded path
            print(f'[t={step}] PICK: closing gripper (fallback)', flush=True)
            obs, info = close_gripper(env, 'right', steps=cfg.grip_close_steps,
                                      obs=obs)
            for _ in range(8):
                a = arm_target_action(
                    env, 'right',
                    np.asarray(obs['state.arm_right_qpos'], dtype=np.float32),
                    grip=-1.0, obs=obs)
                obs, _, term, trunc, info = env.step(a)
                if term or trunc:
                    success = bool(info.get('success', False))
                    break
            phase = 'done'
            continue

        else:  # done - just hold
            obs, _, term, trunc, info = env.step(hold_arms_action(env, obs))
            if term or trunc:
                success = bool(info.get('success', False))
                break

    progress = info.get('task_progress', 0.0)
    q_score = info.get('q_score', 0.0)
    print(f'result: success={success}, task_progress={progress:.1f}%, '
          f'q_score={q_score:.2f}, first_det_step={first_det_step}, '
          f'phase_reached={phase}',
          flush=True)
    return {
        'trial': trial_idx,
        'success': bool(success),
        'task_progress': float(progress),
        'q_score': float(q_score),
        'valid': info.get('valid', True),
        'first_det_step': first_det_step,
        'phase_reached': phase,
    }


def main():
    cfg = tyro.cli(BehaviorConfig)
    Path(cfg.log_dir).mkdir(parents=True, exist_ok=True)

    print(f'[SPARK-BEHAVIOR] task={cfg.task} trials={cfg.num_trials} v0.3', flush=True)

    # Phase 1: Gemini variant gen, BEFORE Isaac Sim init
    # Isaac Sim loads its own openssl which breaks the cryptography lib used by
    # google-genai (cffi: "_openssl has no function Cryptography_HAS_ED448"),
    # so Gemini runs first, while the venv is still pristine.
    # Use the BEHAVIOR-1K instruction string (e.g. "Turning on radio.") looked
    # up from a public table - no scene/BDDL access.
    from gr00t.eval.sim.BEHAVIOR.behavior_env import TASK_NAMES_TO_INSTRUCTIONS
    task_instr = TASK_NAMES_TO_INSTRUCTIONS.get(
        cfg.task, cfg.task.replace('_', ' ').capitalize() + '.')
    print(f'[SPARK-BEHAVIOR] task_instruction={task_instr!r}', flush=True)
    print(f'[SPARK-BEHAVIOR] variant model={cfg.model}', flush=True)
    variants = derive_target_variants_via_gemini(task_instr, K=cfg.K,
                                                 model=cfg.model)
    print(f'[SPARK-BEHAVIOR] Phase-1 variants (K={cfg.K}): {variants}',
          flush=True)
    if cfg.reach_preset not in REACH_PRESETS:
        raise SystemExit(
            f'unknown --reach-preset {cfg.reach_preset!r}; choose from '
            f'{list(REACH_PRESETS.keys())}')
    reach_delta = REACH_PRESETS[cfg.reach_preset]
    print(f'[SPARK-BEHAVIOR] reach_preset={cfg.reach_preset} delta={reach_delta.tolist()}',
          flush=True)
    print(f'[SPARK-BEHAVIOR] approach_mask_area={cfg.approach_mask_area}',
          flush=True)

    # SAM3 loads on the first detect call (post Isaac Sim init). Pre-loading
    # SAM3 makes Isaac Sim crash with "random_device could not be read"
    # (CUDA driver state conflict).

    print('[SPARK-BEHAVIOR] importing BEHAVIORGr00tEnv (Isaac Sim ~1-3 min)...',
          flush=True)
    t0 = time.time()
    from gr00t.eval.sim.BEHAVIOR.behavior_env import BEHAVIORGr00tEnv
    env = BEHAVIORGr00tEnv(task_name=cfg.task, env_idx=0, total_n_envs=1)
    print(f'[SPARK-BEHAVIOR] env ready in {time.time()-t0:.1f}s', flush=True)
    # Bump head ZED to 1024x1024: SAM3 conf on the radio scales 0.0 -> 0.36
    # from 256 -> 1024.
    bump_head_resolution(env, target_res=cfg.head_res)
    enable_seg_instance(env)
    # The seg_instance modality on the head sensor + the obs_space rebuild
    # require a second load_observation_space pass after both have been
    # configured. bump_head_resolution already did the first; do it again here
    # so the new modality keys are registered with og.Environment.
    inner = env.env if hasattr(env, 'env') else env
    if hasattr(inner, 'load_observation_space'):
        inner.load_observation_space()

    results: list[dict] = []
    for i in range(cfg.num_trials):
        try:
            r = run_trial(env, cfg, i, variants, reach_delta=reach_delta,
                          torso_lean=cfg.torso_lean,
                          creep_vx=cfg.creep_vx,
                          creep_steps=cfg.creep_steps)
            results.append(r)
        except Exception as e:
            traceback.print_exc()
            results.append({'trial': i, 'success': False, 'error': str(e),
                            'task_progress': 0.0, 'q_score': 0.0})

    succ = sum(int(r.get('success', False)) for r in results)
    progs = [r.get('task_progress', 0.0) for r in results]
    qs = [r.get('q_score', 0.0) for r in results]
    avg_prog = float(np.mean(progs)) if progs else 0.0
    avg_q = float(np.mean(qs)) if qs else 0.0
    print(f'\n[SPARK-BEHAVIOR] {cfg.task}: '
          f'success {succ}/{len(results)}, '
          f'mean task_progress {avg_prog:.1f}%, '
          f'mean q_score {avg_q:.2f}',
          flush=True)

    # Persist results to JSON for downstream aggregation
    if cfg.out_tag:
        out_json = Path(cfg.log_dir) / f'results_{cfg.task}_{cfg.out_tag}.json'
    else:
        out_json = Path(cfg.log_dir) / f'results_{cfg.task}.json'
    payload = {
        'task': cfg.task,
        'task_instruction': task_instr,
        'config_tag': cfg.out_tag,
        'gemini_model': cfg.model,
        'gemini_variants': variants,
        'reach_preset': cfg.reach_preset,
        'reach_delta': reach_delta.tolist(),
        'torso_lean': cfg.torso_lean,
        'approach_mask_area': cfg.approach_mask_area,
        'K': cfg.K,
        'num_trials': cfg.num_trials,
        'success_count': succ,
        'mean_task_progress': avg_prog,
        'mean_q_score': avg_q,
        'trials': results,
    }
    with open(out_json, 'w') as f:
        json.dump(payload, f, indent=2, default=str)
    print(f'[SPARK-BEHAVIOR] wrote {out_json}', flush=True)


if __name__ == '__main__':
    main()

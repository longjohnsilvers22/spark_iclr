"""
LIBERO-Dyn VLA entrants: LIBERO-finetuned policy checkpoints under the
same mid-episode displacement injector as SPARK (see libero_dyn.py).

The claim under test: a policy finetuned to high success on static LIBERO neither detects nor
adapts to mid-episode displacement.  This runner steps the LIBERO env
THROUGH ``spark_bench.libero_dyn.PerturbationInjector`` with the
identical per-(task, trial) crc32 schedules, runs the policy's normal
closed-loop action loop, and records per-trial success / perturb_fired
into the same JSON cell format as ``libero_dyn.run_dyn``.

Detection / adaptation latency are STRUCTURALLY ABSENT for an
end-to-end VLA (there is no flag, no replan, no recovery channel), so
those fields are recorded as null and ``recovery_used`` as False.

Protocol parity with the SPARK arms (libero_dyn.run_dyn):
  * same suite/task indexing (``libero_<suite>`` via the LIBERO
    benchmark dict), same ``task.name`` fed to ``schedule_seed``;
  * same per-trial env sequence: ``env.seed(trial); env.reset();
    env.set_init_state(init_states[trial % N]); env.reset();
    10 x env.step(zeros(7))`` settle;
  * same displacement vectors (``displacement_vector``), same
    ``displace_body_qpos`` body matching on the BDDL pick hint;
  * same phase semantics: POST_PLAN fires after the settle and before
    the first policy query ("after planning, before execution" - a VLA
    has no plan step, so this is displacement at execution start);
    MID_APPROACH fires when the EE has covered 50 percent of its
    initial distance to the pick target (target position read from the
    ground-truth free-joint qpos at settle time - method-agnostic
    geometric definition, no policy introspection); POST_GRASP_PLAN
    fires on the first close command on the action channel
    (``action[6] > 0.5``; the OpenVLA gripper conversion below maps
    close to +1, so the stock trigger applies unchanged).

Policy: OpenVLA (openvla/openvla-7b-finetuned-libero-object), 7-DoF
delta EE actions from a single 256x256 agentview render.  Conversions
ported from openvla/experiments/robot (run_libero_eval.py,
libero_utils.py, robot_utils.py, openvla_utils.py):
  * image: rotate 180 deg (img[::-1, ::-1]), JPEG encode/decode
    round-trip, lanczos resize to 224 (PIL port of the tf pipeline),
    optional 0.9-area center crop
    (the released LIBERO finetunes were trained with random-crop
    augmentation, so center crop is ON by default);
  * prompt: "In: What action should the robot take to <task>?\\nOut:";
  * gripper: [0, 1] -> [-1, +1], binarize, then sign-flip (LIBERO
    expects -1 open / +1 close).

Usage (conda env ``openvla``)::

    MUJOCO_GL=egl PYTHONPATH=$HOME/dyn_final/src:$HOME/spark/src/libero_pro \
        python -u -m spark_bench.libero_dyn_vla --suite object --task-id 0 \
        --num-trials 5 --phases post_plan,mid_approach,post_grasp_plan \
        --magnitudes-cm 0,2,5,10 --output-dir results_vla

    # static sanity probe (must reproduce published static success):
    ... --phases post_plan --magnitudes-cm 0 --num-trials 2 --tag probe
"""
from __future__ import annotations

import argparse
import json
import math
import os
import time
from io import BytesIO
from pathlib import Path
from typing import Optional

os.environ.setdefault('MUJOCO_GL', 'egl')
os.environ.setdefault('TOKENIZERS_PARALLELISM', 'false')

import numpy as np
import torch
from PIL import Image

from spark_bench.libero_dyn import (
    Phase, PerturbationInjector, _default_ee_reader, displace_body_qpos,
    displacement_vector, find_free_joint_for_hint)
from spark_bench.libero_pro.bddl import get_task_prompts_for_suite

from libero.libero import benchmark, get_libero_path
from libero.libero.envs import OffScreenRenderEnv

# LIBERO init-state files are numpy pickles; newer torch defaults to
# weights_only=True and refuses them (same patch as libero_dyn's stack).
_torch_load_orig = torch.load


def _torch_load_patched(*args, **kwargs):
    kwargs.setdefault('weights_only', False)
    return _torch_load_orig(*args, **kwargs)


torch.load = _torch_load_patched

DEVICE = 'cuda:0'


# Image preprocessing (PIL port of openvla/experiments/robot tf pipeline)

def preprocess_agentview(obs: dict, resize_size: int = 224) -> np.ndarray:
    """
    openvla libero_utils.get_libero_image, minus tensorflow.

    Rotate 180 deg, JPEG round-trip (the RLDS dataset builder stored
    JPEGs; tf.image.encode_jpeg default quality is 95), lanczos resize.
    """
    img = np.asarray(obs['agentview_image'])
    img = img[::-1, ::-1]
    im = Image.fromarray(img)
    buf = BytesIO()
    im.save(buf, format='JPEG', quality=95)
    buf.seek(0)
    im = Image.open(buf).convert('RGB')
    im = im.resize((resize_size, resize_size), Image.LANCZOS)
    return np.asarray(im, dtype=np.uint8)


def center_crop_resize(img: np.ndarray, crop_scale: float = 0.9,
                       out_size: int = 224) -> np.ndarray:
    """
    openvla_utils.crop_and_resize: center-crop to ``crop_scale`` x area
    (side scale = sqrt(crop_scale)), bilinear resize back to 224.
    """
    h, w = img.shape[:2]
    s = math.sqrt(crop_scale)
    ch, cw = round(h * s), round(w * s)
    top, left = (h - ch) // 2, (w - cw) // 2
    im = Image.fromarray(img[top:top + ch, left:left + cw])
    im = im.resize((out_size, out_size), Image.BILINEAR)
    return np.asarray(im, dtype=np.uint8)


# Gripper conventions (openvla robot_utils.py)

def convert_gripper(action: np.ndarray) -> np.ndarray:
    """normalize_gripper_action(binarize=True) + invert_gripper_action."""
    action = np.asarray(action, dtype=np.float64).copy()
    action[-1] = 2.0 * action[-1] - 1.0       # [0,1] -> [-1,1]
    action[-1] = np.sign(action[-1])          # binarize
    action[-1] = -action[-1]                  # RLDS 1=open -> LIBERO +1=close
    return action


# Policy wrapper

class OpenVLAPolicy:
    name = 'openvla'

    def __init__(self, checkpoint: str, unnorm_key: str,
                 center_crop: bool = True):
        from transformers import AutoModelForVision2Seq, AutoProcessor
        print(f'[dyn_vla] loading {checkpoint} ...', flush=True)
        self.processor = AutoProcessor.from_pretrained(
            checkpoint, trust_remote_code=True)
        attn = 'sdpa'
        try:
            import flash_attn  # noqa: F401
            attn = 'flash_attention_2'
        except ImportError:
            pass
        self.model = AutoModelForVision2Seq.from_pretrained(
            checkpoint, attn_implementation=attn,
            torch_dtype=torch.bfloat16, low_cpu_mem_usage=True,
            trust_remote_code=True).to(DEVICE).eval()
        if (unnorm_key not in self.model.norm_stats
                and f'{unnorm_key}_no_noops' in self.model.norm_stats):
            unnorm_key = f'{unnorm_key}_no_noops'
        assert unnorm_key in self.model.norm_stats, (
            f'unnorm key {unnorm_key} not in norm_stats '
            f'{list(self.model.norm_stats)}')
        self.unnorm_key = unnorm_key
        self.center_crop = center_crop
        self.checkpoint = checkpoint
        print(f'[dyn_vla] loaded ({attn}); unnorm_key={unnorm_key} '
              f'center_crop={center_crop}', flush=True)

    def reset(self) -> None:  # OpenVLA is stateless across steps
        pass

    def act(self, obs: dict, instruction: str, env=None) -> np.ndarray:
        img = preprocess_agentview(obs)
        if self.center_crop:
            img = center_crop_resize(img)
        image = Image.fromarray(img).convert('RGB')
        prompt = (f'In: What action should the robot take to '
                  f'{instruction.lower()}?\nOut:')
        inputs = self.processor(prompt, image).to(DEVICE,
                                                  dtype=torch.bfloat16)
        with torch.inference_mode():
            action = self.model.predict_action(
                **inputs, unnorm_key=self.unnorm_key, do_sample=False)
        return convert_gripper(np.asarray(action, dtype=np.float64))


class Pi05Policy:
    """
    Pi0.5 (lerobot/pi05_libero_finetuned) via the lerobot 0.4.x stack.

    Requires a lerobot >= 0.4 source checkout on PYTHONPATH (the pi05
    conda env's site-packages lerobot is a version stub).  The obs is
    built exactly like ``lerobot.envs.libero.LiberoEnv._format_raw_obs``
    and run through the same env/policy processor pipelines the stock
    lerobot eval uses (LiberoProcessorStep does the 180-deg image flip
    and the state packing internally); the raw ``OffScreenRenderEnv``
    stays in hand so the injector and the SPARK reset sequence apply
    unchanged.
    """
    name = 'pi05'

    def __init__(self, checkpoint: str, suite: str):
        from lerobot.policies.pi05.modeling_pi05 import PI05Policy as _P
        from lerobot.policies.factory import make_pre_post_processors
        from lerobot.envs.factory import make_env_pre_post_processors
        from lerobot.envs.configs import LiberoEnv as LiberoEnvConfig
        from lerobot.envs.utils import preprocess_observation
        print(f'[dyn_vla] loading {checkpoint} ...', flush=True)
        self._preprocess_observation = preprocess_observation
        self.policy = _P.from_pretrained(checkpoint).eval().to(DEVICE)
        self.policy.config.device = DEVICE
        self.preprocessor, self.postprocessor = make_pre_post_processors(
            policy_cfg=self.policy.config, pretrained_path=checkpoint,
            preprocessor_overrides={'device_processor': {'device': DEVICE}})
        env_cfg = LiberoEnvConfig(task=f'libero_{suite}',
                                  observation_height=256,
                                  observation_width=256)
        self.env_preprocessor, self.env_postprocessor = (
            make_env_pre_post_processors(env_cfg=env_cfg,
                                         policy_cfg=self.policy.config))
        self.checkpoint = checkpoint
        self.unnorm_key = None
        self.center_crop = None
        print('[dyn_vla] pi05 loaded', flush=True)

    def reset(self) -> None:
        self.policy.reset()  # clears the action-chunk queue

    def act(self, obs: dict, instruction: str, env=None) -> np.ndarray:
        mat = None
        try:
            mat = env.robots[0].controller.ee_ori_mat
        except Exception:
            pass
        def _b(x):  # batch dim: the processors expect vector-env (B, ...)
            return None if x is None else np.asarray(x)[None]

        obs_l = {
            'pixels': {
                'image': np.asarray(obs['agentview_image']),
                'image2': np.asarray(obs['robot0_eye_in_hand_image']),
            },
            'robot_state': {
                'eef': {'pos': _b(obs.get('robot0_eef_pos')),
                        'quat': _b(obs.get('robot0_eef_quat')),
                        'mat': _b(mat)},
                'gripper': {'qpos': _b(obs.get('robot0_gripper_qpos')),
                            'qvel': _b(obs.get('robot0_gripper_qvel'))},
                'joints': {'pos': _b(obs.get('robot0_joint_pos')),
                           'vel': _b(obs.get('robot0_joint_vel'))},
            },
        }
        op = self._preprocess_observation(obs_l)
        op['task'] = [instruction]
        op = self.env_preprocessor(op)
        op = self.preprocessor(op)
        with torch.inference_mode():
            action = self.policy.select_action(op)
        action = self.postprocessor(action)
        at = self.env_postprocessor({'action': action})
        a = at['action']
        if hasattr(a, 'cpu'):
            a = a.cpu().numpy()
        a = np.asarray(a, dtype=np.float64)
        if a.ndim > 1:
            a = a.squeeze(0)
        return a


def build_policy(name: str, checkpoint: str, suite: str,
                 center_crop: bool):
    if name == 'openvla':
        return OpenVLAPolicy(checkpoint or
                             f'openvla/openvla-7b-finetuned-libero-{suite}',
                             unnorm_key=f'libero_{suite}',
                             center_crop=center_crop)
    if name == 'pi05':
        return Pi05Policy(checkpoint or 'lerobot/pi05_libero_finetuned',
                          suite=suite)
    raise SystemExit(f'unknown --policy {name!r} (supported: openvla, pi05)')


# Env loading (mirror of fair.config.load_libero_env, minus its heavy deps)

def load_env(suite_name: str, task_id: int, cam: int = 256):
    benchmark_dict = benchmark.get_benchmark_dict()
    task_suite = benchmark_dict[suite_name]()
    task = task_suite.get_task(task_id)
    bddl_path = os.path.join(get_libero_path('bddl_files'),
                             task.problem_folder, task.bddl_file)
    if not os.path.exists(bddl_path):
        parts = task.problem_folder.split('_')
        base_folder = ('_'.join(parts[:2]) if len(parts) > 2
                       else task.problem_folder)
        bddl_path = os.path.join(get_libero_path('bddl_files'),
                                 base_folder, task.bddl_file)
    env = OffScreenRenderEnv(
        bddl_file_name=bddl_path,
        camera_heights=cam,
        camera_widths=cam,
        camera_depths=False,
        horizon=2000,
    )
    init_states = task_suite.get_task_init_states(task_id)
    return env, task, init_states, bddl_path


def _gt_pick_pos(env, pick_hint: str) -> Optional[np.ndarray]:
    """
    Ground-truth pick-target position from the free-joint qpos.

    MID_APPROACH's 50-percent-of-initial-distance trigger is defined
    geometrically; the SPARK arms read the target from their det_map,
    a VLA has none, so the GT body position (read once at settle time,
    never fed to the policy) anchors the same geometric definition.
    """
    import mujoco
    model = env.sim.model._model
    data = env.sim.data._data
    adr, _name = find_free_joint_for_hint(model, pick_hint, mj=mujoco)
    if adr is None:
        return None
    return np.array(data.qpos[adr:adr + 3], dtype=float)


# Trial loop

def run_trial(env, policy, instruction: str, pick_hint: str,
              task_name: str, phase: Phase, mag_cm: float, trial: int,
              init_states, max_steps: int) -> dict:
    trial_meta: dict = {'trial': trial, 'phase': phase.value,
                        'magnitude_cm': mag_cm}
    injector: Optional[PerturbationInjector] = None
    success = False
    steps = 0
    t0 = time.time()
    try:
        # Identical sequence to libero_dyn.run_dyn.
        env.seed(trial)
        env.reset()
        if init_states is not None and len(init_states) > 0:
            env.set_init_state(init_states[trial % len(init_states)])
            env.reset()
        obs = None
        for _ in range(10):
            out = env.step(np.zeros(7))
            obs = out[0]

        delta = displacement_vector(task_name, trial, mag_cm)
        trial_meta['direction'] = [round(float(x), 4) for x in delta[:2]]

        if float(np.linalg.norm(delta)) >= 1e-9:
            def displace():
                return displace_body_qpos(env, pick_hint, delta)
            if phase == Phase.POST_PLAN:
                t = time.time()
                body = displace()
                trial_meta['t_perturb'] = t
                trial_meta['perturb_phase'] = phase.value
                if body:
                    trial_meta['perturbed_body'] = body
            else:
                injector = PerturbationInjector(
                    env, phase, displace_fn=displace,
                    pick_target_pos=_gt_pick_pos(env, pick_hint),
                    get_ee_pos=_default_ee_reader(env),
                    meta=trial_meta)
                injector.install()

        policy.reset()
        for _ in range(max_steps):
            action = policy.act(obs, instruction, env=env)
            obs, _reward, _done, _info = env.step(action)
            steps += 1
            # NOT `_done`: robosuite's done also fires at the horizon,
            # which is not a success.  The oracle predicate decides.
            if env.check_success():
                success = True
                break
    except Exception as e:
        trial_meta['error'] = str(e)
        import traceback
        traceback.print_exc()
    finally:
        if injector is not None:
            injector.uninstall()

    trial_meta['success'] = bool(success)
    trial_meta['perturb_fired'] = 't_perturb' in trial_meta
    trial_meta['steps'] = steps
    trial_meta['wall_clock_s'] = round(time.time() - t0, 2)
    # Structurally absent for an end-to-end VLA: no detector, no flag,
    # no recovery channel.  Null by construction, not "not observed".
    trial_meta['detection_latency_s'] = None
    trial_meta['adaptation_latency_s'] = None
    trial_meta['detected_by'] = None
    trial_meta['adapted_by'] = None
    trial_meta['recovery_used'] = False
    return trial_meta


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--suite', default='object')
    p.add_argument('--task-id', type=int, default=0)
    p.add_argument('--policy', default='openvla')
    p.add_argument('--checkpoint', default='',
                   help='HF id or local path (default: per-suite openvla '
                        'LIBERO finetune)')
    p.add_argument('--num-trials', type=int, default=5)
    p.add_argument('--trial-offset', type=int, default=0)
    p.add_argument('--phases',
                   default='post_plan,mid_approach,post_grasp_plan')
    p.add_argument('--magnitudes-cm', default='0,2,5,10')
    p.add_argument('--max-steps', type=int, default=280,
                   help='openvla run_libero_eval uses 280 for libero_object')
    p.add_argument('--no-center-crop', action='store_true')
    p.add_argument('--output-dir', default='results_vla')
    p.add_argument('--tag', default='')
    args = p.parse_args()

    base_suite = f'libero_{args.suite}'
    env, task, init_states, bddl_path = load_env(base_suite, args.task_id)
    task_info = get_task_prompts_for_suite(base_suite, args.task_id, task,
                                           bddl_path=bddl_path)
    pick_hint = task_info.get('pick', '') or ''
    # The policy prompt must be the exact training-time task string
    # (task.language, e.g. "pick up the alphabet soup and place it in
    # the basket").  The BDDL-derived phrasing ("Pick the alphabet soup
    # ...") is SPARK's planner instruction, not the VLA's; the BDDL
    # parse is still used for the displacement pick hint.
    instruction = getattr(task, 'language', None) or task_info['instruction']

    phases = [Phase(s.strip()) for s in args.phases.split(',') if s.strip()]
    mags = [float(m) for m in args.magnitudes_cm.split(',') if m.strip()]

    policy = build_policy(args.policy, args.checkpoint, args.suite,
                          center_crop=not args.no_center_crop)

    print(f'LIBERO-Dyn-VLA: {task.name} | policy={policy.name} | '
          f'phases={[p.value for p in phases]} mags={mags}cm '
          f'trials/cell={args.num_trials} pick_hint={pick_hint!r}',
          flush=True)

    cells: list[dict] = []
    for phase in phases:
        for mag in mags:
            cell = {'phase': phase.value, 'magnitude_cm': mag, 'trials': []}
            for trial_i in range(args.num_trials):
                trial = trial_i + args.trial_offset
                rec = run_trial(env, policy, instruction, pick_hint,
                                task.name, phase, mag, trial,
                                init_states, args.max_steps)
                cell['trials'].append(rec)
                print(f"  [{phase.value:>15s} {mag:4.0f}cm T{trial}] "
                      f"success={rec['success']} "
                      f"fired={rec['perturb_fired']} "
                      f"steps={rec['steps']} "
                      f"({rec['wall_clock_s']:.0f}s)"
                      + (f" err={rec['error']}" if 'error' in rec else ''),
                      flush=True)
            n = len(cell['trials'])
            cell['success_rate'] = (sum(t['success'] for t in cell['trials'])
                                    / n if n else 0.0)
            cells.append(cell)

    try:
        env.close()
    except Exception:
        pass

    result = {
        'protocol': 'libero_dyn',
        'arm': f'vla_{policy.name}',
        'suite': args.suite,
        'task_id': args.task_id,
        'task_name': task.name,
        'instruction': instruction,
        'num_trials': args.num_trials,
        'config': {
            'policy': policy.name,
            'checkpoint': policy.checkpoint,
            'unnorm_key': policy.unnorm_key,
            'center_crop': policy.center_crop,
            'max_steps': args.max_steps,
            'latency_metrics': 'structurally_absent_for_vla',
        },
        'cells': cells,
    }
    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    tag = f'_{args.tag}' if args.tag else ''
    out_path = (out_dir / f'dyn_vla_{policy.name}_{args.suite}'
                          f'_task{args.task_id}{tag}.json')
    with open(out_path, 'w') as f:
        json.dump(result, f, indent=2, default=str)
    print(f'[dyn_vla] wrote {out_path}', flush=True)


if __name__ == '__main__':
    main()

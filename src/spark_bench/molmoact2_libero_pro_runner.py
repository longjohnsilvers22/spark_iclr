"""
MolmoAct2-LIBERO baseline runner on SPARK's LIBERO-PRO env.

Drives ``libero.libero.envs.OffScreenRenderEnv`` with actions from
``allenai/MolmoAct2-LIBERO`` (HF model card; see ``REVISION``).

Citations:
  - HF model card: https://huggingface.co/allenai/MolmoAct2-LIBERO
    "Continuous Actions" section.  Sample state is 8-D
    ``[eef_pos(3), axis_angle(3), gripper_qpos(2)]``.  Camera order is
    front/agentview followed by wrist.
  - LeRobot ``src/lerobot/processor/env_processor.py``
    confirms quat (x, y, z, w) -> axis_angle conversion and gripper_qpos
    is the raw 2-D pair.
  - LIBERO action signature: 7-D in ``Box(-1, 1)`` = 6-D EE delta + 1-D
    gripper.  LIBERO-PRO suite ``libero_goal_swap`` task index 6 maps to
    ``put_the_cream_cheese_in_the_bowl`` (verified via
    spark/src/libero_pro/libero/libero/benchmark/libero_suite_task_map.py
    line 1252).

Env strategy: run inside conda env ``molmoact2`` (Python 3.12, torch 2.11
+ cu130, transformers 5.8.0).  LIBERO is loaded via PYTHONPATH from
``~/spark/src/libero_pro/`` (no pip install needed).

Usage::

    conda activate molmoact2
    cd ~/spark/src
    MUJOCO_GL=egl PYTHONPATH=src/libero_pro:src:$PYTHONPATH \\
        python -m spark_bench.molmoact2_libero_pro_runner \\
            --suite goal --perturbation position --task-id 6 --n-trials 1
"""
from __future__ import annotations

import argparse
import json
import os
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

# Set MUJOCO before any mujoco / libero / torch import.
os.environ.setdefault("MUJOCO_GL", "egl")
os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")

import numpy as np
import torch
from PIL import Image

try:
    from huggingface_hub import snapshot_download
    from transformers import AutoModelForImageTextToText, AutoProcessor
except ImportError:
    snapshot_download = None
    AutoModelForImageTextToText = None
    AutoProcessor = None

# LIBERO via PYTHONPATH (libero_pro vendored under src/libero_pro).
try:
    from libero.libero import benchmark
    from libero.libero.envs import OffScreenRenderEnv
    from libero.libero.utils import get_libero_path
except ImportError:
    benchmark = None
    OffScreenRenderEnv = None
    get_libero_path = None


# torch.load patch for LIBERO init state files (numpy pickles)
_torch_load_orig = torch.load


def _torch_load_patched(*args, **kwargs):
    kwargs.setdefault("weights_only", False)
    return _torch_load_orig(*args, **kwargs)


torch.load = _torch_load_patched


REPO_ID = "allenai/MolmoAct2-LIBERO"
# Pinned checkpoint snapshot.  Set to ``None`` to always use ``main``; the
# norm_stats.json layout / predict_action signature may change in newer
# revisions.
REVISION: Optional[str] = "56c75eaeac063c3adb3734d77e7458bb47adc918"


# Config

@dataclass
class RunConfig:
    suite: str = "goal"           # one of: object / spatial / goal / 10
    perturbation: str = "position"  # position -> _swap suite, task -> _task suite
    task_id: int = 6
    n_trials: int = 1
    max_steps: int = 600          # CaP-X / SPARK use 600
    num_steps_wait: int = 50      # let scene settle (matches lerobot/envs/libero.py num_steps_wait=50)
    cam_height: int = 256
    cam_width: int = 256
    dtype: str = "bfloat16"        # "bfloat16" (~11GB) or "float32" (~24GB)
    n_action_steps: int = 10       # MolmoAct2 LIBERO action chunk size
    num_flow_steps: int = 10       # default from checkpoint config
    enable_cuda_graph: bool = False  # off for first call; expensive warm-up
    output_dir: str = str(Path(__file__).resolve().parents[2] / "output" / "molmoact2_libero_pro")
    verbose: bool = True


# Helpers

def _perturbation_suite(base_suite: str, perturbation: str) -> str:
    # "vanilla" / "none" => unperturbed base suite (sanity-check mode).
    if perturbation in ("vanilla", "none"):
        return f"libero_{base_suite}"
    ptype_map = {
        "position": "swap",
        "task": "task",
        "language": "lan",
        "object": "object",
        "environment": "env",
    }
    suffix = ptype_map.get(perturbation, perturbation)
    return f"libero_{base_suite}_{suffix}"


def _resolve_bddl(task) -> str:
    bddl_path = os.path.join(
        get_libero_path("bddl_files"), task.problem_folder, task.bddl_file
    )
    if os.path.exists(bddl_path):
        return bddl_path
    parts = task.problem_folder.split("_")
    base_folder = "_".join(parts[:2]) if len(parts) > 2 else task.problem_folder
    return os.path.join(get_libero_path("bddl_files"), base_folder, task.bddl_file)


def _quat_xyzw_to_axis_angle(quat: np.ndarray) -> np.ndarray:
    """
    Robosuite quat is (x, y, z, w); LeRobot expects (x, y, z, w) too.

    Verbatim port of LeRobot's ``_quat2axisangle`` from
    ``lerobot/src/lerobot/processor/env_processor.py:114``.
    """
    quat = np.asarray(quat, dtype=np.float32).reshape(4)
    w = float(np.clip(quat[3], -1.0, 1.0))
    den = float(np.sqrt(max(0.0, 1.0 - w * w)))
    if den < 1e-10:
        return np.zeros(3, dtype=np.float32)
    angle = 2.0 * np.arccos(w)
    axis = quat[:3] / den
    return (axis * angle).astype(np.float32)


def _build_state(raw_obs: dict) -> np.ndarray:
    """
    Convert LIBERO raw obs to MolmoAct2-LIBERO's 8-D state.

    Schema (from LeRobot env_processor.py, line 104):
        [eef_pos(3), axis_angle(3), gripper_qpos(2)]
    """
    eef_pos = np.asarray(raw_obs["robot0_eef_pos"], dtype=np.float32).reshape(3)
    eef_quat = np.asarray(raw_obs["robot0_eef_quat"], dtype=np.float32).reshape(4)
    gripper_qpos = np.asarray(raw_obs["robot0_gripper_qpos"], dtype=np.float32).reshape(2)
    axis_angle = _quat_xyzw_to_axis_angle(eef_quat)
    return np.concatenate([eef_pos, axis_angle, gripper_qpos]).astype(np.float32)


def _build_images(raw_obs: dict) -> tuple[Image.Image, Image.Image]:
    """
    LIBERO obs returns HWC uint8.  LIBERO renders images upside-down
    (LIBERO-PRO fair runner does ``rgb[::-1]``), mirrored here.
    """
    agentview = np.asarray(raw_obs["agentview_image"])
    wrist = np.asarray(raw_obs["robot0_eye_in_hand_image"])
    if agentview.dtype != np.uint8:
        agentview = (agentview * 255).clip(0, 255).astype(np.uint8)
        wrist = (wrist * 255).clip(0, 255).astype(np.uint8)
    # LIBERO renders need a 180 deg rotation (flip H AND W) per LeRobot
    # env_processor.py line 59: torch.flip(img, dims=[2, 3]).  Comment:
    # "Rotates images by 180 degrees ... HuggingFaceVLA/libero camera
    # orientation convention."  Vertical flip alone is NOT enough.
    # See run_spark_libero_pro_fair.py line 297.
    agentview = agentview[::-1, ::-1].copy()
    wrist = wrist[::-1, ::-1].copy()
    return Image.fromarray(agentview).convert("RGB"), Image.fromarray(wrist).convert("RGB")


# Model loading

def load_molmoact2(dtype: str):
    """
    Snapshot the MolmoAct2-LIBERO checkpoint and return (processor, model).
    """
    print(f"[molmoact2] snapshotting {REPO_ID}@{REVISION or 'main'} ...")
    local_dir = snapshot_download(repo_id=REPO_ID, revision=REVISION)
    print(f"[molmoact2]   local_dir={local_dir}")

    torch_dtype = {"bfloat16": torch.bfloat16, "float32": torch.float32}[dtype]

    processor = AutoProcessor.from_pretrained(local_dir, trust_remote_code=True)
    model = (
        AutoModelForImageTextToText.from_pretrained(
            local_dir,
            trust_remote_code=True,
            dtype=torch_dtype,
        )
        .to("cuda")
        .eval()
    )
    print(f"[molmoact2] loaded; cuda mem allocated="
          f"{torch.cuda.memory_allocated() / 1e9:.2f} GB")
    return processor, model


# Rollout

def rollout_one_trial(
    env,
    processor,
    model,
    instruction: str,
    cfg: RunConfig,
    init_state: Optional[np.ndarray],
    trial_idx: int,
) -> dict:
    """
    Run one closed-loop episode and return a result record.
    """
    env.seed(trial_idx)
    env.reset()
    if init_state is not None:
        env.set_init_state(init_state)
        env.reset()
    # Settle the scene.  Matches lerobot get_libero_dummy_action: 6 zeros + gripper -1 (open).
    dummy_action = np.array([0.0, 0.0, 0.0, 0.0, 0.0, 0.0, -1.0], dtype=np.float32)
    for _ in range(cfg.num_steps_wait):
        raw_obs, *_ = env.step(dummy_action)

    dtype = {"bfloat16": torch.bfloat16, "float32": torch.float32}[cfg.dtype]

    t0 = time.time()
    step = 0
    success = False
    trace: list[dict] = []
    while step < cfg.max_steps:
        try:
            if env.check_success():
                success = True
                break
        except Exception:
            pass

        # Build observation: 2 images + 8-D state.
        images = _build_images(raw_obs)
        state = _build_state(raw_obs)

        # Predict an action chunk of length n_action_steps.
        with torch.inference_mode(), torch.autocast("cuda", dtype=dtype):
            out = model.predict_action(
                processor=processor,
                images=list(images),
                task=instruction,
                state=state,
                norm_tag="libero",
                action_mode="continuous",
                enable_depth_reasoning=False,
                num_steps=cfg.num_flow_steps,
                normalize_language=True,
                enable_cuda_graph=cfg.enable_cuda_graph,
            )
        actions = out.actions
        arr = (
            actions.detach().cpu().float().numpy()
            if torch.is_tensor(actions) else np.asarray(actions, dtype=np.float32)
        )
        # predict_action returns (1, n_action_steps, 7); drop batch dim.
        if arr.ndim == 3 and arr.shape[0] == 1:
            arr = arr[0]
        if arr.ndim == 1:
            arr = arr[None, :]
        # arr is (n_action_steps, 7).  Execute open-loop until the chunk
        # is exhausted (matches the LIBERO ``n_action_steps`` convention).
        for a_idx in range(min(arr.shape[0], cfg.n_action_steps)):
            if step >= cfg.max_steps:
                break
            action = arr[a_idx].astype(np.float32)
            # LIBERO expects (7,) in Box(-1, 1).  Clip defensively.
            action = np.clip(action, -1.0, 1.0)
            try:
                raw_obs, reward, done, info = env.step(action)
            except Exception as e:
                trace.append({"step": step, "error": str(e)})
                return {
                    "success": False,
                    "steps": step,
                    "wall_clock_s": time.time() - t0,
                    "error": str(e),
                    "trace_tail": trace[-5:],
                }
            try:
                done_flag = bool(env.check_success())
            except Exception:
                done_flag = False
            if cfg.verbose and step % 20 == 0:
                trace.append({
                    "step": step,
                    "action": action.round(3).tolist(),
                    "eef": np.asarray(raw_obs["robot0_eef_pos"]).round(3).tolist(),
                    "gripper": float(np.asarray(raw_obs["robot0_gripper_qpos"]).sum()),
                })
            step += 1
            if done_flag:
                success = True
                break
        if success:
            break

    return {
        "success": success,
        "steps": step,
        "wall_clock_s": time.time() - t0,
        "instruction": instruction,
        "trace_tail": trace[-10:],
    }


# Main

def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--suite", default="goal", choices=["object", "spatial", "goal", "10"])
    p.add_argument("--perturbation", default="position",
                   choices=["position", "task", "language", "object",
                            "environment", "vanilla", "none"])
    p.add_argument("--task-id", type=int, default=6,
                   help="Task index in suite (goal/swap T6 = cream cheese in bowl)")
    p.add_argument("--n-trials", type=int, default=1)
    p.add_argument("--max-steps", type=int, default=600)
    p.add_argument("--dtype", default="bfloat16", choices=["bfloat16", "float32"])
    p.add_argument("--n-action-steps", type=int, default=10)
    p.add_argument("--num-flow-steps", type=int, default=10)
    p.add_argument("--enable-cuda-graph", action="store_true")
    p.add_argument("--output-dir", default=str(Path(__file__).resolve().parents[2] / "output" / "molmoact2_libero_pro"))
    args = p.parse_args()

    cfg = RunConfig(
        suite=args.suite,
        perturbation=args.perturbation,
        task_id=args.task_id,
        n_trials=args.n_trials,
        max_steps=args.max_steps,
        dtype=args.dtype,
        n_action_steps=args.n_action_steps,
        num_flow_steps=args.num_flow_steps,
        enable_cuda_graph=args.enable_cuda_graph,
        output_dir=args.output_dir,
    )

    suite_name = _perturbation_suite(cfg.suite, cfg.perturbation)
    print(f"[molmoact2] suite={suite_name} task_id={cfg.task_id} "
          f"trials={cfg.n_trials} dtype={cfg.dtype}")

    benchmark_dict = benchmark.get_benchmark_dict()
    if suite_name not in benchmark_dict:
        raise SystemExit(f"Suite '{suite_name}' not registered. Available: "
                         f"{sorted(k for k in benchmark_dict if 'goal' in k)}")
    task_suite = benchmark_dict[suite_name]()
    if cfg.task_id < 0 or cfg.task_id >= task_suite.n_tasks:
        raise SystemExit(f"task_id={cfg.task_id} out of range [0, {task_suite.n_tasks})")

    task = task_suite.get_task(cfg.task_id)
    bddl_path = _resolve_bddl(task)
    instruction = task.language
    print(f"[molmoact2] task.name={task.name}")
    print(f"[molmoact2] task.language={instruction!r}")
    print(f"[molmoact2] bddl_path={bddl_path}")
    sidecar = os.environ.get("MOLMOACT2_BDDL_LOG")
    if sidecar:
        with open(sidecar, "a") as _fh:
            _fh.write(f"{suite_name}\tT{cfg.task_id}\t{task.name}\t{bddl_path}\n")

    try:
        init_states = task_suite.get_task_init_states(cfg.task_id)
    except Exception as e:
        print(f"[molmoact2] WARN: no init_states ({e}); using env default")
        init_states = None

    processor, model = load_molmoact2(cfg.dtype)

    env = OffScreenRenderEnv(
        bddl_file_name=bddl_path,
        camera_heights=cfg.cam_height,
        camera_widths=cfg.cam_width,
        camera_depths=False,
        horizon=2000,
    )

    results = []
    for trial in range(cfg.n_trials):
        init_state = None
        if init_states is not None and len(init_states) > 0:
            init_state = init_states[trial % len(init_states)]
        print(f"\n[molmoact2]trial {trial}")
        rec = rollout_one_trial(env, processor, model, instruction, cfg,
                                init_state, trial)
        rec["trial"] = trial
        rec["suite"] = suite_name
        rec["task_id"] = cfg.task_id
        rec["task_name"] = task.name
        print(f"[molmoact2] trial {trial}: success={rec['success']} "
              f"steps={rec['steps']} wall_clock={rec['wall_clock_s']:.1f}s")
        for entry in rec.get("trace_tail", []):
            print(f"{entry}")
        results.append(rec)

    try:
        env.close()
    except Exception:
        pass

    out_dir = Path(cfg.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    stamp = time.strftime("%Y%m%d_%H%M%S")
    out_path = out_dir / f"{suite_name}_T{cfg.task_id}_n{cfg.n_trials}_{stamp}.json"
    payload = {
        "config": {
            "suite": suite_name,
            "task_id": cfg.task_id,
            "task_name": task.name,
            "instruction": instruction,
            "n_trials": cfg.n_trials,
            "max_steps": cfg.max_steps,
            "dtype": cfg.dtype,
            "n_action_steps": cfg.n_action_steps,
            "num_flow_steps": cfg.num_flow_steps,
            "enable_cuda_graph": cfg.enable_cuda_graph,
        },
        "checkpoint": {
            "repo_id": REPO_ID,
            "revision": REVISION,
        },
        "cuda_peak_gb": float(torch.cuda.max_memory_reserved() / 1e9),
        "n_success": sum(int(r["success"]) for r in results),
        "n_trials": len(results),
        "results": results,
    }
    out_path.write_text(json.dumps(payload, indent=2))
    print(f"\n[molmoact2] wrote {out_path}")
    print(f"[molmoact2] {payload['n_success']}/{payload['n_trials']} successes")
    print(f"[molmoact2] cuda peak: {payload['cuda_peak_gb']:.2f} GB reserved")


if __name__ == "__main__":
    main()

"""
Frozen pi0.5 on LIBERO-PRO, under SPARK's own trial protocol.

A frozen pi0.5 is driven closed-loop through the *same* env objects, the
*same* task ordering, and the *same* initial states that
``spark_bench.run_spark_libero_pro_fair`` gives the SPARK arm, scored by the
*same* ``env.check_success()``.

PROTOCOL PARITY
---------------
Reproduced exactly:

* Suite names.  ``libero_{goal,object,spatial}`` crossed with the LIBERO-PRO
  suffix map ``{'position': 'swap', 'task': 'task'}``, i.e. ``libero_goal_swap``
  ... ``libero_spatial_task``.  Same ``get_perturbation_suite`` helper, imported
  from ``spark_bench.fair.config`` rather than re-implemented, so the two arms
  cannot drift.
* Task ordering.  ``for task_id in range(task_suite.n_tasks)`` on the registered
  benchmark object -- no sorting, no filtering.
* Env construction.  ``OffScreenRenderEnv(bddl_file_name=..., horizon=2000)``
  with the same BDDL-resolution fallback chain (perturbation suites reuse the
  base suite's BDDL folder).
* Per-trial reset.  ``env.seed(trial + trial_offset)`` -> ``env.reset()`` ->
  ``env.set_init_state(init_states[(trial + trial_offset) % n_states])`` ->
  ``env.reset()``.  The second reset is in the SPARK arm and is reproduced here
  verbatim; it is load-bearing for state parity.
* Settle.  Ten steps of ``np.zeros(7)``.  The SPARK arm settles for **10**
  steps, not 20, and with an all-zero action (gripper command 0), not the
  ``[0]*6 + [-1]`` open-gripper dummy that openpi's own libero example and the
  MolmoAct2 runner use.  ``--settle-steps`` and ``--settle-gripper`` exist to
  quantify what that choice costs the VLA; the defaults are the parity values.
* Scoring.  ``bool(env.check_success())``.

Deliberately NOT reproduced:

* Mid-episode ``check_success()``.  The SPARK arm polls the predicate during
  execution because its behaviour tree uses it as a termination signal.  A VLA
  gets no oracle: this runner evaluates the predicate **once, after the last
  step**.  That is strictly harder than the poll-and-break convention used by
  openpi's example and by the MolmoAct2 runner.  ``--poll-success`` restores
  the lenient convention for the openpi-comparable number; it is off by default
  and is recorded in the output JSON so the two can never be confused.
* Camera resolution.  The SPARK arm renders 640x480 because SAM3 needs the
  pixels; pi0.5 was trained on 256x256 renders resized to 224.  This runner
  renders 256x256 (``--cam-size``).  Resolution changes only what the cameras
  see -- not the physics, not the init states, not the BDDL predicate -- so
  success is still measured on an identical world.

HORIZON CHOICE
--------------
Default ``--max-steps 600``, the SPARK arm's budget, on the same
``horizon=2000`` env.  openpi's ``examples/libero/main.py`` instead uses a
per-suite budget: 220 spatial / 280 object / 300 goal / 520 libero_10, sized
to the longest training demo in each suite.  600 dominates every one of
openpi's per-suite budgets, so a failure here cannot be explained by a starved
step budget.  ``--budget openpi`` switches to openpi's per-suite table for a
direct comparison against their reported numbers.

CHECKPOINT / BACKEND
--------------------
``--backend lerobot`` (DEFAULT).  LeRobot's ``PI05Policy`` on
``lerobot/pi05_libero_finetuned`` (7.47 GB).  Needs a lerobot >= 0.4 checkout
on PYTHONPATH.

``--backend local``.  openpi in-process on
``gs://openpi-assets/checkpoints/pi05_libero`` with train config ``pi05_libero``
-- the official Physical Intelligence pi0.5 LIBERO checkpoint, 12.44 GB, public
bucket, ships its own ``assets/physical-intelligence/libero/norm_stats.json``.
A confirmatory second checkpoint: if two independently trained frozen pi0.5
LIBERO checkpoints agree, the row is about pi0.5 and not about one release.

``--backend websocket``.  Same openpi model, held in openpi's own
``serve_policy.py`` process, so the env and the policy can live in different
conda envs.

USAGE
-----
Default backend::

    conda activate pi05
    cd ~/spark/src
    MUJOCO_GL=egl \\
    PYTHONPATH=src/libero_pro:src:<lerobot checkout>/src \\
        python -m spark_bench.vla_baseline.run_pi05_libero_pro \\
            --suite goal --perturbation all --num-trials 50

openpi in-process::

    MUJOCO_GL=egl \\
    PYTHONPATH=src/libero_pro:src:$OPENPI/src:$OPENPI/packages/openpi-client/src \\
        python -m spark_bench.vla_baseline.run_pi05_libero_pro \\
            --backend local --suite goal --perturbation all --num-trials 50

openpi over the wire::

    # terminal 1 (openpi env)
    uv run scripts/serve_policy.py --port 8000 \\
        policy:checkpoint --policy.config pi05_libero \\
        --policy.dir gs://openpi-assets/checkpoints/pi05_libero
    # terminal 2 (env with LIBERO)
    python -m spark_bench.vla_baseline.run_pi05_libero_pro \\
        --backend websocket --host 127.0.0.1 --port 8000 --suite goal

Offline checks (no GPU, no env, no network)::

    python -m spark_bench.vla_baseline.run_pi05_libero_pro --self-test
    python -m spark_bench.vla_baseline.run_pi05_libero_pro --dry-run --suite goal

Output: ``{output_dir}/{suite}.json`` in the schema
``{suite, num_trials, perturbations: {ptype: {per_task, average, task_details}}}``
consumed unchanged by ``spark_bench.stats_intervals.load_arm``.
"""
from __future__ import annotations

import argparse
import collections
import json
import math
import os
import sys
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Optional

# MuJoCo backend must be chosen before mujoco / libero / torch land.
os.environ.setdefault("MUJOCO_GL", "egl")
os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")

import numpy as np

def _force_fork_libero(fork_root=None):
    """Make ``import libero`` resolve to the LIBERO-PRO fork, always.

    ``fork_root`` defaults to ``<repo>/src/libero_pro/libero``.

    The fork's top-level ``libero/`` has no ``__init__.py``, so it is a
    NAMESPACE package, and Python resolves any regular ``libero`` package on
    sys.path (e.g. a stock pip install in site-packages) ahead of every
    namespace portion regardless of path order. Pre-seeding sys.modules with
    a module whose __path__ is the fork directory wins unconditionally and
    touches no environment. Must run before the first ``import libero``.
    """
    if fork_root is None:
        fork_root = os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', '..', 'libero_pro', 'libero')
    import types, sys
    if 'libero' in sys.modules:
        return
    m = types.ModuleType('libero')
    m.__path__ = [fork_root]
    sys.modules['libero'] = m


# Module-import time, deliberately: any later ``import libero`` in this file or
# its imports must already see the fork.
_force_fork_libero()


# --------------------------------------------------------------------------
# Optional imports.  Every one of these is deferred-tolerant so that
# ``--dry-run`` and ``--self-test`` work on a machine with no GPU, no LIBERO
# and no openpi.
# --------------------------------------------------------------------------

try:
    from libero.libero import benchmark as _libero_benchmark
    from libero.libero import get_libero_path
    from libero.libero.envs import OffScreenRenderEnv
except Exception as _e:  # pragma: no cover - import guard
    _libero_benchmark = None
    get_libero_path = None
    OffScreenRenderEnv = None
    _LIBERO_IMPORT_ERROR = _e
else:
    _LIBERO_IMPORT_ERROR = None

try:
    from openpi_client import image_tools as _image_tools
except Exception as _e:  # pragma: no cover - import guard
    _image_tools = None
    _CLIENT_IMPORT_ERROR = _e
else:
    _CLIENT_IMPORT_ERROR = None


# LIBERO's ``.pruned_init`` files are numpy pickles; torch>=2.6 defaults to
# ``weights_only=True`` and refuses them.  Same patch as the MolmoAct2 runner.


def _patch_torch_load() -> None:
    try:
        import torch
    except Exception:
        return
    if getattr(torch.load, "_spark_patched", False):
        return
    _orig = torch.load

    def _patched(*args, **kwargs):
        kwargs.setdefault("weights_only", False)
        return _orig(*args, **kwargs)

    _patched._spark_patched = True  # type: ignore[attr-defined]
    torch.load = _patched


# --------------------------------------------------------------------------
# Protocol constants
# --------------------------------------------------------------------------

#: LIBERO-PRO suffix map.  Mirrors ``spark_bench.fair.config.get_perturbation_suite``.
#: ``_perturbation_suite`` prefers the imported function and falls back to
#: this table only when spark_bench is not importable (dry-run on a bare box).
PTYPE_SUFFIX = {
    "position": "swap",
    "task": "task",
    "language": "lan",
    "object": "object",
    "environment": "env",
}

#: The six cells this baseline owes the paper.
DEFAULT_SUITES = ("goal", "object", "spatial")
DEFAULT_PERTURBATIONS = ("position", "task")

#: openpi ``examples/libero/main.py`` per-suite step budgets, sized to the
#: longest training demo in each suite.  Used only under ``--budget openpi``.
OPENPI_MAX_STEPS = {
    "libero_spatial": 220,
    "libero_object": 280,
    "libero_goal": 300,
    "libero_10": 520,
    "libero_90": 400,
}

#: SPARK arm's budget (fair/config.py).
SPARK_MAX_STEPS = 600

#: openpi backends.  Official Physical Intelligence pi0.5 LIBERO checkpoint,
#: 12.44 GB, public GCS bucket, ships its own norm stats.
DEFAULT_CHECKPOINT = "gs://openpi-assets/checkpoints/pi05_libero"
DEFAULT_CONFIG_NAME = "pi05_libero"

#: lerobot backend checkpoint (7.47 GB).
LEROBOT_CHECKPOINT = "lerobot/pi05_libero_finetuned"


@dataclass
class Pi05Config:
    # --- cell selection -------------------------------------------------
    suite: str = "goal"                    # goal | object | spatial | 10
    perturbation: str = "all"              # position | task | all | vanilla
    num_trials: int = 50
    trial_offset: int = 0
    only_task_id: int = -1
    start_task_id: int = 0
    end_task_id: int = -1
    skip_task_ids: str = ""

    # --- policy ---------------------------------------------------------
    backend: str = "lerobot"               # lerobot | local | websocket
    config_name: str = DEFAULT_CONFIG_NAME
    checkpoint: str = LEROBOT_CHECKPOINT
    host: str = "127.0.0.1"
    port: int = 8000
    replan_steps: int = 5
    resize_size: int = 224
    quantize: str = "none"                 # none | w8a8  (see _apply_quantize)

    # --- rollout --------------------------------------------------------
    budget: str = "spark"                  # spark | openpi | fixed
    max_steps: int = SPARK_MAX_STEPS
    settle_steps: int = 10                 # SPARK parity value
    settle_gripper: float = 0.0            # SPARK parity value (0, not -1)
    cam_size: int = 256
    env_horizon: int = 2000
    poll_success: bool = False             # False == episode-end scoring only

    # --- io -------------------------------------------------------------
    output_dir: str = ""
    tag: str = ""
    verbose: bool = True
    dry_run: bool = False
    self_test: bool = False


# --------------------------------------------------------------------------
# Suite / task plumbing.  Prefer the SPARK harness helpers so the two arms
# share one definition of "which suite is this".
# --------------------------------------------------------------------------

def _perturbation_suite(base_suite: str, perturbation: str) -> str:
    """
    ``('libero_goal', 'position') -> 'libero_goal_swap'``.

    Delegates to ``spark_bench.fair.config.get_perturbation_suite`` when the
    SPARK harness is importable, so a change there cannot silently desync the
    baseline.  The local table is a dry-run fallback only.
    """
    if perturbation in ("vanilla", "none"):
        return base_suite
    try:
        from spark_bench.fair.config import get_perturbation_suite
    except Exception:
        return f"{base_suite}_{PTYPE_SUFFIX.get(perturbation, perturbation)}"
    return get_perturbation_suite(base_suite, perturbation)


def _resolve_bddl(task) -> str:
    """
    BDDL path with the SPARK arm's fallback chain.

    Perturbation suites keep their own ``problem_folder`` but reuse the base
    suite's BDDL directory, so ``libero_goal_swap`` files live under
    ``libero_goal``.  Mirrors fair/config.py.
    """
    root = get_libero_path("bddl_files")
    cand = [os.path.join(root, task.problem_folder, task.bddl_file)]
    parts = task.problem_folder.split("_")
    if len(parts) > 2:
        cand.append(os.path.join(root, "_".join(parts[:2]), task.bddl_file))
    # libero_pro vendored copy, if the installed LIBERO does not carry it.
    try:
        from spark_bench.fair.config import LIBERO_PRO_ROOT
        pro = Path(LIBERO_PRO_ROOT) / "libero" / "libero" / "bddl_files"
        cand.append(str(pro / task.problem_folder / task.bddl_file))
        if len(parts) > 2:
            cand.append(str(pro / "_".join(parts[:2]) / task.bddl_file))
    except Exception:
        pass
    for c in cand:
        if os.path.exists(c):
            return c
    raise FileNotFoundError(
        f"No BDDL for {task.name}; tried:\n  " + "\n  ".join(cand))


def _init_states_for(task_suite, task_id: int, suite_name: str, task):
    """
    Init states, with the SPARK arm's ``.pruned_init`` fallback.

    Returns ``None`` when nothing is found, which makes the trial loop fall
    back to a plain ``env.reset()`` -- the same degradation the SPARK arm
    accepts, so both arms degrade identically or not at all.
    """
    try:
        return task_suite.get_task_init_states(task_id)
    except Exception:
        pass
    try:
        from spark_bench.fair.config import _load_init_states_fallback
        return _load_init_states_fallback(suite_name, task)
    except Exception:
        return None


def _max_steps_for(cfg: Pi05Config, suite_name: str) -> int:
    """Step budget for one suite.  See the HORIZON CHOICE note in the module docstring."""
    if cfg.budget == "openpi":
        base = "_".join(suite_name.split("_")[:2])
        if base not in OPENPI_MAX_STEPS:
            raise SystemExit(
                f"--budget openpi has no published budget for {base!r}; "
                f"known: {sorted(OPENPI_MAX_STEPS)}")
        return OPENPI_MAX_STEPS[base]
    if cfg.budget == "spark":
        return SPARK_MAX_STEPS
    return cfg.max_steps


# --------------------------------------------------------------------------
# Observation encoding.  Must match openpi's libero example exactly; a
# mismatch here silently degrades the baseline.
# --------------------------------------------------------------------------

def _quat2axisangle(quat) -> np.ndarray:
    """
    Robosuite (x, y, z, w) quaternion -> axis-angle.

    Verbatim port of ``examples/libero/main.py::_quat2axisangle``, itself copied
    from robosuite ``transform_utils``.  Kept byte-for-byte rather than
    "improved" so the state vector the policy sees is the one it was trained on.
    """
    quat = np.asarray(quat, dtype=np.float64).copy()
    if quat[3] > 1.0:
        quat[3] = 1.0
    elif quat[3] < -1.0:
        quat[3] = -1.0
    den = np.sqrt(1.0 - quat[3] * quat[3])
    if math.isclose(den, 0.0):
        return np.zeros(3)
    return (quat[:3] * 2.0 * math.acos(quat[3])) / den


def _build_element(obs: dict, instruction: str, resize_size: int) -> dict:
    """
    LIBERO raw obs -> openpi ``libero_policy.LiberoInputs`` payload.

    Two things here are easy to get wrong and both are load-bearing:

    1. The 180-degree rotation ``[::-1, ::-1]``.  LIBERO renders upside-down
       relative to the training data.  A vertical flip alone is NOT enough --
       the width axis must flip too.
    2. ``resize_with_pad`` rather than a plain resize, so aspect ratio is
       preserved the way it was during training.

    State is 8-D ``[eef_pos(3), axis_angle(3), gripper_qpos(2)]``.
    """
    if _image_tools is None:
        raise RuntimeError(
            f"openpi_client is not importable: {_CLIENT_IMPORT_ERROR}")
    img = np.ascontiguousarray(obs["agentview_image"][::-1, ::-1])
    wrist = np.ascontiguousarray(obs["robot0_eye_in_hand_image"][::-1, ::-1])
    img = _image_tools.convert_to_uint8(
        _image_tools.resize_with_pad(img, resize_size, resize_size))
    wrist = _image_tools.convert_to_uint8(
        _image_tools.resize_with_pad(wrist, resize_size, resize_size))
    state = np.concatenate((
        np.asarray(obs["robot0_eef_pos"], dtype=np.float64).reshape(3),
        _quat2axisangle(obs["robot0_eef_quat"]),
        np.asarray(obs["robot0_gripper_qpos"], dtype=np.float64).reshape(2),
    ))
    return {
        "observation/image": img,
        "observation/wrist_image": wrist,
        "observation/state": state,
        "prompt": str(instruction),
    }


# --------------------------------------------------------------------------
# Policy backends
# --------------------------------------------------------------------------

class _LocalPolicy:
    """In-process openpi policy.  Holds the model on this process's GPU."""

    mode = "chunk"

    def __init__(self, config_name: str, checkpoint: str, quantize: str):
        from openpi.policies import policy_config as _policy_config
        from openpi.training import config as _config

        _apply_quantize(quantize)
        train_config = _config.get_config(config_name)
        t0 = time.time()
        self._policy = _policy_config.create_trained_policy(
            train_config, checkpoint)
        self.load_seconds = time.time() - t0
        self.meta = {
            "backend": "local",
            "config_name": config_name,
            "checkpoint": checkpoint,
            "action_horizon": int(getattr(train_config.model,
                                          "action_horizon", -1)),
            "load_seconds": round(self.load_seconds, 1),
        }

    def infer(self, element: dict) -> np.ndarray:
        return np.asarray(self._policy.infer(element)["actions"])

    def reset(self) -> None:
        pass


class _WebsocketPolicy:
    """openpi ``serve_policy.py`` over the wire.  Model lives elsewhere."""

    mode = "chunk"

    def __init__(self, host: str, port: int):
        from openpi_client import websocket_client_policy as _wsc
        self._policy = _wsc.WebsocketClientPolicy(host, port)
        self.meta = {
            "backend": "websocket",
            "host": host,
            "port": port,
            "server_metadata": _safe(lambda: self._policy.get_server_metadata()),
        }

    def infer(self, element: dict) -> np.ndarray:
        return np.asarray(self._policy.infer(element)["actions"])

    def reset(self) -> None:
        pass


class _LeRobotPolicy:
    """
    LeRobot's own PI05Policy on ``lerobot/pi05_libero_finetuned``.

    Reuses ``spark_bench.libero_dyn_vla.Pi05Policy``: its observation
    packing goes through LeRobot's own env/policy processor pipeline, so the
    180-degree flip and the state layout are LeRobot's.

    Interface note: LeRobot's ``select_action`` returns ONE action and queues
    the rest of the chunk internally, so this backend is ``mode == 'step'``
    and ``--replan-steps`` does not apply.  The chunk cadence is the
    checkpoint's own (``n_action_steps``, 50 for this checkpoint).
    """

    mode = "step"

    def __init__(self, checkpoint: str, suite: str):
        from spark_bench.libero_dyn_vla import Pi05Policy
        t0 = time.time()
        self._policy = Pi05Policy(checkpoint, suite=suite)
        self.load_seconds = time.time() - t0
        cfg = getattr(self._policy.policy, "config", None)
        self.meta = {
            "backend": "lerobot",
            "checkpoint": checkpoint,
            "suite": suite,
            "n_action_steps": int(getattr(cfg, "n_action_steps", -1)),
            "chunk_size": int(getattr(cfg, "chunk_size", -1)),
            "num_inference_steps": int(getattr(cfg, "num_inference_steps", -1)),
            "load_seconds": round(self.load_seconds, 1),
            "via": "spark_bench.libero_dyn_vla.Pi05Policy",
        }

    def act(self, obs: dict, instruction: str, env) -> np.ndarray:
        return self._policy.act(obs, instruction, env=env)

    def reset(self) -> None:
        self._policy.reset()


def _safe(fn):
    try:
        return fn()
    except Exception as e:
        return f"<unavailable: {e}>"


def _apply_quantize(mode: str) -> None:
    """
    ``--quantize`` stub.

    openpi ships no quantization path, so any mode other than ``none``
    errors loudly rather than silently no-op.

    VRAM for the unquantized model: params are ~12.4 GB of bf16 on disk for
    the JAX checkpoint (~7.5 GB for the PyTorch ``model.safetensors`` export
    of the same ~3.5B-parameter net).  Budget ~16 GB of VRAM for weights +
    activations at batch 1, which fits a 24 GB GPU with room for the MuJoCo
    EGL context.  W8A8 is a throughput optimisation here, not a fit
    requirement.
    """
    if mode in ("none", "", None):
        return
    raise SystemExit(
        "--quantize is not supported: openpi has no quantization path "
        "(verified by grep over src/openpi). The unquantized model needs "
        "~16 GB VRAM at batch 1, which fits a 24 GB RTX 4090 or any A100. "
        "See docs/PI05_BASELINE_PLAN.md if you need to reproduce Zetta's "
        "W8A8 configuration specifically.")


def _make_policy(cfg: Pi05Config):
    if cfg.backend == "websocket":
        return _WebsocketPolicy(cfg.host, cfg.port)
    if cfg.backend == "local":
        ckpt = (cfg.checkpoint if cfg.checkpoint != LEROBOT_CHECKPOINT
                else DEFAULT_CHECKPOINT)
        return _LocalPolicy(cfg.config_name, ckpt, cfg.quantize)
    if cfg.backend == "lerobot":
        ckpt = (cfg.checkpoint if cfg.checkpoint != DEFAULT_CHECKPOINT
                else LEROBOT_CHECKPOINT)
        return _LeRobotPolicy(ckpt, cfg.suite)
    raise SystemExit(f"unknown --backend {cfg.backend!r}")


# --------------------------------------------------------------------------
# Rollout
# --------------------------------------------------------------------------

def rollout_trial(env, policy, instruction: str, cfg: Pi05Config,
                  init_state, trial_idx: int, max_steps: int) -> dict:
    """
    One episode.  Returns a ``trial_meta`` dict; ``['success']`` is the verdict.

    Reset order is the SPARK arm's, including the second ``env.reset()`` after
    ``set_init_state`` (run_spark_libero_pro_fair.py).

    Success is read ONCE, after the loop, unless ``--poll-success``.  The policy
    therefore never gets an early-exit signal it would not have on a real robot.
    """
    seed = trial_idx + cfg.trial_offset
    env.seed(seed)
    env.reset()
    if init_state is not None:
        env.set_init_state(init_state)
        env.reset()

    settle = np.zeros(7, dtype=np.float64)
    settle[6] = cfg.settle_gripper
    obs = None
    for _ in range(cfg.settle_steps):
        obs, _r, _d, _i = env.step(settle)
    if obs is None:  # settle_steps == 0
        obs, _r, _d, _i = env.step(settle)

    policy.reset()          # clear any queued action chunk from the last episode
    plan: collections.deque = collections.deque()
    step_mode = getattr(policy, "mode", "chunk") == "step"
    t0 = time.time()
    step = 0
    n_infer = 0
    infer_seconds = 0.0
    early = False
    err: Optional[str] = None

    while step < max_steps:
        try:
            if step_mode:
                # LeRobot queues the chunk itself; one call == one action.
                ti = time.time()
                action = np.asarray(policy.act(obs, instruction, env),
                                    dtype=np.float64)
                infer_seconds += time.time() - ti
                n_infer += 1
            else:
                if not plan:
                    element = _build_element(obs, instruction, cfg.resize_size)
                    ti = time.time()
                    chunk = policy.infer(element)
                    infer_seconds += time.time() - ti
                    n_infer += 1
                    if chunk.ndim == 3 and chunk.shape[0] == 1:
                        chunk = chunk[0]
                    if chunk.ndim == 1:
                        chunk = chunk[None, :]
                    take = min(cfg.replan_steps, chunk.shape[0])
                    if take < cfg.replan_steps and n_infer == 1:
                        print(f"    [warn] replan_steps={cfg.replan_steps} but "
                              f"the policy returns {chunk.shape[0]} steps; "
                              f"using {take}", flush=True)
                    plan.extend(chunk[:take])
                action = np.asarray(plan.popleft(), dtype=np.float64)
            action = np.clip(action.reshape(-1), -1.0, 1.0)
            obs, _r, _d, _i = env.step(action.tolist())
            step += 1
            if cfg.poll_success and bool(env.check_success()):
                early = True
                break
        except Exception as e:  # a crashed episode is a failed episode
            err = f"{type(e).__name__}: {e}"
            break

    # The arbiter.  One call, at the end.
    try:
        success = bool(env.check_success())
    except Exception as e:
        success = False
        err = err or f"check_success failed: {type(e).__name__}: {e}"

    meta = {
        "trial": trial_idx,
        "seed": seed,
        "success": success,
        "steps": step,
        "max_steps": max_steps,
        "n_infer": n_infer,
        "infer_seconds": round(infer_seconds, 2),
        "wall_clock_s": round(time.time() - t0, 2),
        "early_exit": early,
    }
    if err:
        meta["error"] = err
    return meta


# --------------------------------------------------------------------------
# Cell / suite drivers
# --------------------------------------------------------------------------

def run_cell(cfg: Pi05Config, policy, base_suite: str, ptype: str) -> dict:
    """
    One LIBERO-PRO cell: ten tasks x ``num_trials`` init states.

    Returns the ``perturbations[ptype]`` value of the output schema:
    ``{'per_task': [...], 'average': float, 'task_details': [...]}``.
    """
    if _libero_benchmark is None:
        raise RuntimeError(f"LIBERO is not importable: {_LIBERO_IMPORT_ERROR}")
    _patch_torch_load()

    benchmark_dict = _libero_benchmark.get_benchmark_dict()
    suite_name = _perturbation_suite(base_suite, ptype)
    if suite_name not in benchmark_dict:
        print(f"Note: {suite_name} not registered, using base suite {base_suite}")
        suite_name = base_suite
    task_suite = benchmark_dict[suite_name]()
    num_tasks = task_suite.n_tasks
    max_steps = _max_steps_for(cfg, suite_name)

    print(f"\n{'=' * 60}")
    print(f"pi0.5 LIBERO-PRO: {base_suite}, perturbation={ptype} "
          f"({suite_name})")
    print(f"Tasks: {num_tasks}, Trials per task: {cfg.num_trials}, "
          f"max_steps: {max_steps} ({cfg.budget})")
    print(f"Scoring: check_success at episode end"
          f"{' + mid-episode poll' if cfg.poll_success else ' ONLY'}")
    print(f"{'=' * 60}\n")

    skip = {int(s) for s in cfg.skip_task_ids.split(",") if s.strip().isdigit()}
    per_task: list[float] = []
    task_details: list[dict] = []

    for task_id in range(num_tasks):
        if cfg.only_task_id >= 0 and task_id != cfg.only_task_id:
            continue
        if task_id < cfg.start_task_id:
            continue
        if cfg.end_task_id >= 0 and task_id >= cfg.end_task_id:
            continue
        if task_id in skip:
            print(f"[skip] task_id={task_id}")
            continue

        task = task_suite.get_task(task_id)
        instruction = task.language
        print(f"{task.name[:50]:50s}", end=" ", flush=True)

        env = None
        try:
            bddl_path = _resolve_bddl(task)
            env = OffScreenRenderEnv(
                bddl_file_name=bddl_path,
                camera_heights=cfg.cam_size,
                camera_widths=cfg.cam_size,
                camera_depths=False,
                horizon=cfg.env_horizon,
            )
        except Exception as e:
            print(f"ENV LOAD FAILED: {e}")
            per_task.append(0.0)
            task_details.append({
                "task_name": task.name,
                "instruction": instruction,
                "prompts": [],
                "pick": "",
                "place": "",
                "success_rate": 0.0,
                "trials": [],
                "trial_meta": [],
                "error": str(e),
            })
            continue

        init_states = _init_states_for(task_suite, task_id, suite_name, task)
        if init_states is None and task_id == 0:
            print("\n  WARNING: no init states; falling back to env.reset()",
                  end=" ")

        trials: list[bool] = []
        metas: list[dict] = []
        for trial in range(cfg.num_trials):
            init_state = None
            if init_states is not None and len(init_states) > 0:
                init_state = init_states[
                    (trial + cfg.trial_offset) % len(init_states)]
            try:
                meta = rollout_trial(env, policy, instruction, cfg,
                                     init_state, trial, max_steps)
            except Exception as e:
                meta = {"trial": trial, "success": False,
                        "error": f"{type(e).__name__}: {e}"}
            trials.append(bool(meta["success"]))
            metas.append(meta)
            print("ok" if trials[-1] else "*", end="", flush=True)
            if (trial + 1) % 10 == 0:
                print(f"[{sum(trials)}/{len(trials)}]", end="", flush=True)

        try:
            env.close()
        except Exception:
            pass

        sr = sum(trials) / len(trials) if trials else 0.0
        per_task.append(sr)
        task_details.append({
            "task_name": task.name,
            "instruction": instruction,
            "prompts": [],       # schema parity: SPARK's SAM3 prompt list
            "pick": "",          # schema parity: SPARK's pick label
            "place": "",         # schema parity: SPARK's place label
            "success_rate": sr,
            "trials": trials,
            "trial_meta": metas,
        })
        print(f" {sr:.0%} ({sum(trials)}/{len(trials)})")

    average = float(np.mean(per_task)) if per_task else 0.0
    print(f"\n{base_suite} ({ptype}): {average:.1%}")
    return {"per_task": per_task, "average": average,
            "task_details": task_details}


def run_suite(cfg: Pi05Config) -> dict:
    """Run every requested perturbation for one suite and write the JSON."""
    base_suite = f"libero_{cfg.suite}"
    ptypes = (list(DEFAULT_PERTURBATIONS) if cfg.perturbation == "all"
              else [cfg.perturbation])

    if cfg.dry_run:
        return _dry_run(cfg, base_suite, ptypes)

    policy = _make_policy(cfg)
    print(f"[pi05] policy ready: {json.dumps(policy.meta, default=str)}")

    all_results: dict[str, Any] = {}
    for ptype in ptypes:
        all_results[ptype] = run_cell(cfg, policy, base_suite, ptype)

    print(f"\n{'=' * 60}")
    print(f"pi0.5 LIBERO-PRO Summary ({cfg.suite})")
    print(f"{'=' * 60}")
    for ptype, res in all_results.items():
        print(f"{ptype:12s}: {res['average']:.1%}")
    if all_results:
        print(f"{'Overall':12s}: "
              f"{np.mean([r['average'] for r in all_results.values()]):.1%}")

    payload = {
        "suite": cfg.suite,
        "num_trials": cfg.num_trials,
        "perturbations": all_results,
        # Provenance beyond the shared schema; stats_intervals reads only
        # 'perturbations'.
        "method": "pi05_frozen",
        "policy": policy.meta,
        "protocol": _protocol_record(cfg),
    }
    out_dir = Path(cfg.output_dir) if cfg.output_dir else _default_out_dir(cfg)
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / f"{cfg.suite}.json"
    out_path.write_text(json.dumps(payload, indent=2, default=str))
    print(f"\n[pi05] wrote {out_path}")
    return payload


def _default_out_dir(cfg: Pi05Config) -> Path:
    name = "pi05_libero_pro" + (f"_{cfg.tag}" if cfg.tag else "")
    return Path.home() / "spark" / "videos" / name


def _protocol_record(cfg: Pi05Config) -> dict:
    """
    Protocol provenance: SPARK's protocol, not openpi's.

    Written into the JSON so a number can never be reinterpreted later as
    having come from different settings.
    """
    d = asdict(cfg)
    d["settle_action"] = f"np.zeros(7) with a[6]={cfg.settle_gripper}"
    d["success_scoring"] = ("episode-end check_success only"
                            if not cfg.poll_success
                            else "mid-episode poll + episode end")
    d["reset_order"] = ("seed(trial+offset) -> reset -> set_init_state("
                        "(trial+offset) % n) -> reset -> settle")
    d["matches"] = "spark_bench.run_spark_libero_pro_fair"
    return d


# --------------------------------------------------------------------------
# Offline checks.  Neither touches CUDA, MuJoCo, or the network.
# --------------------------------------------------------------------------

def _dry_run(cfg: Pi05Config, base_suite: str, ptypes: list[str]) -> dict:
    """Print the exact plan -- suites, tasks, trials, budget -- and load nothing."""
    print("[dry-run] no model loaded, no env stepped\n")
    total = 0
    for ptype in ptypes:
        suite_name = _perturbation_suite(base_suite, ptype)
        max_steps = _max_steps_for(cfg, suite_name)
        n_tasks = "?"
        if _libero_benchmark is not None:
            bd = _libero_benchmark.get_benchmark_dict()
            if suite_name in bd:
                n_tasks = bd[suite_name]().n_tasks
            else:
                n_tasks = f"NOT REGISTERED (have: {len(bd)} suites)"
        print(f"  {base_suite:16s} {ptype:9s} -> {suite_name:22s} "
              f"tasks={n_tasks} trials={cfg.num_trials} max_steps={max_steps}")
        if isinstance(n_tasks, int):
            total += n_tasks * cfg.num_trials
    print(f"\n  episodes for this invocation: {total}")
    print(f"  libero import: "
          f"{'OK' if _libero_benchmark is not None else _LIBERO_IMPORT_ERROR}")
    print(f"  openpi_client import: "
          f"{'OK' if _image_tools is not None else _CLIENT_IMPORT_ERROR}")
    try:
        from openpi.training import config as _c
        _c.get_config(cfg.config_name)
        print(f"  openpi config {cfg.config_name!r}: OK "
              f"(backends local/websocket)")
    except Exception as e:
        print(f"  openpi config {cfg.config_name!r}: {type(e).__name__}: {e}")
    try:
        import lerobot
        from lerobot.policies.pi05.modeling_pi05 import PI05Policy  # noqa: F401
        print(f"  lerobot PI05Policy: OK (lerobot {lerobot.__version__}) "
              f"(backend lerobot)")
    except Exception as e:
        print(f"  lerobot PI05Policy: {type(e).__name__}: {e}")
    print(f"  protocol: {json.dumps(_protocol_record(cfg), indent=2)}")
    return {"dry_run": True}


def _self_test() -> int:
    """
    Pure-numpy checks of the parts that do not need a GPU, an env, or a model.

    Covers the quaternion convention and the suite-name mapping, the two
    things least likely to announce themselves at runtime.
    """
    fails = []

    def check(name, cond, detail=""):
        print(f"  {'PASS' if cond else 'FAIL'}  {name}"
              f"{'' if cond else '  ' + detail}")
        if not cond:
            fails.append(name)

    # Identity quaternion (x,y,z,w) = (0,0,0,1) -> zero rotation.
    check("quat2axisangle(identity) == 0",
          np.allclose(_quat2axisangle([0.0, 0.0, 0.0, 1.0]), 0.0))
    # 180 deg about +x -> (pi, 0, 0).
    aa = _quat2axisangle([1.0, 0.0, 0.0, 0.0])
    check("quat2axisangle(180deg about x) == (pi,0,0)",
          np.allclose(aa, [math.pi, 0.0, 0.0], atol=1e-6), f"got {aa}")
    # 90 deg about +z.
    s = math.sqrt(0.5)
    aa = _quat2axisangle([0.0, 0.0, s, s])
    check("quat2axisangle(90deg about z) == (0,0,pi/2)",
          np.allclose(aa, [0.0, 0.0, math.pi / 2], atol=1e-6), f"got {aa}")
    # Out-of-range w must be clipped, not NaN.
    check("quat2axisangle clips w>1",
          not np.any(np.isnan(_quat2axisangle([0.0, 0.0, 0.0, 1.0000001]))))

    # Suite naming, the thing that would quietly evaluate the wrong cell.
    for base, ptype, want in [
        ("libero_goal", "position", "libero_goal_swap"),
        ("libero_goal", "task", "libero_goal_task"),
        ("libero_object", "position", "libero_object_swap"),
        ("libero_object", "task", "libero_object_task"),
        ("libero_spatial", "position", "libero_spatial_swap"),
        ("libero_spatial", "task", "libero_spatial_task"),
        ("libero_goal", "vanilla", "libero_goal"),
    ]:
        got = _perturbation_suite(base, ptype)
        check(f"suite {base}+{ptype} -> {want}", got == want, f"got {got}")

    # Budget table.
    c = Pi05Config(budget="spark")
    check("spark budget == 600", _max_steps_for(c, "libero_goal_swap") == 600)
    c = Pi05Config(budget="openpi")
    check("openpi goal budget == 300",
          _max_steps_for(c, "libero_goal_swap") == 300)
    check("openpi spatial budget == 220",
          _max_steps_for(c, "libero_spatial_task") == 220)
    c = Pi05Config(budget="fixed", max_steps=520)
    check("fixed budget honours --max-steps",
          _max_steps_for(c, "libero_goal_swap") == 520)

    # Quantize stub must refuse rather than silently no-op.
    try:
        _apply_quantize("w8a8")
        check("--quantize w8a8 refuses", False, "did not raise")
    except SystemExit:
        check("--quantize w8a8 refuses", True)
    _apply_quantize("none")
    check("--quantize none is a no-op", True)

    # Output schema is exactly what stats_intervals.load_arm reads.
    cell = {"per_task": [0.5], "average": 0.5,
            "task_details": [{"task_name": "t", "trials": [True, False]}]}
    payload = {"suite": "goal", "num_trials": 2,
               "perturbations": {"position": cell}}
    try:
        rows = [list(td["trials"])
                for td in payload["perturbations"]["position"]["task_details"]]
        check("stats_intervals schema readable",
              rows == [[True, False]], f"got {rows}")
    except Exception as e:
        check("stats_intervals schema readable", False, str(e))

    print(f"\n  {len(fails)} failure(s)"
          + (": " + ", ".join(fails) if fails else ""))
    return 1 if fails else 0


# --------------------------------------------------------------------------
# CLI
# --------------------------------------------------------------------------

def _parse() -> Pi05Config:
    p = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter)
    d = Pi05Config()

    g = p.add_argument_group("cell")
    g.add_argument("--suite", default=d.suite,
                   choices=["goal", "object", "spatial", "10"])
    g.add_argument("--perturbation", default=d.perturbation,
                   choices=["position", "task", "all", "vanilla"])
    g.add_argument("--num-trials", type=int, default=d.num_trials)
    g.add_argument("--trial-offset", type=int, default=d.trial_offset)
    g.add_argument("--only-task-id", type=int, default=d.only_task_id)
    g.add_argument("--start-task-id", type=int, default=d.start_task_id)
    g.add_argument("--end-task-id", type=int, default=d.end_task_id)
    g.add_argument("--skip-task-ids", default=d.skip_task_ids)

    g = p.add_argument_group("policy")
    g.add_argument("--backend", default=d.backend,
                   choices=["lerobot", "local", "websocket"],
                   help="lerobot=LeRobot PI05Policy on the already-cached "
                        "lerobot/pi05_libero_finetuned (no download); "
                        "local=openpi in-process; websocket=openpi serve_policy")
    g.add_argument("--config-name", default=d.config_name)
    g.add_argument("--checkpoint", default=d.checkpoint)
    g.add_argument("--host", default=d.host)
    g.add_argument("--port", type=int, default=d.port)
    g.add_argument("--replan-steps", type=int, default=d.replan_steps)
    g.add_argument("--resize-size", type=int, default=d.resize_size)
    g.add_argument("--quantize", default=d.quantize, choices=["none", "w8a8"])

    g = p.add_argument_group("rollout")
    g.add_argument("--budget", default=d.budget,
                   choices=["spark", "openpi", "fixed"],
                   help="spark=600 (our arm); openpi=per-suite table; "
                        "fixed=--max-steps")
    g.add_argument("--max-steps", type=int, default=d.max_steps)
    g.add_argument("--settle-steps", type=int, default=d.settle_steps)
    g.add_argument("--settle-gripper", type=float, default=d.settle_gripper)
    g.add_argument("--cam-size", type=int, default=d.cam_size)
    g.add_argument("--env-horizon", type=int, default=d.env_horizon)
    g.add_argument("--poll-success", action="store_true",
                   help="lenient openpi-style mid-episode success poll; OFF by "
                        "default because a VLA gets no oracle")

    g = p.add_argument_group("io")
    g.add_argument("--output-dir", default=d.output_dir)
    g.add_argument("--tag", default=d.tag)
    g.add_argument("--quiet", action="store_true")
    g.add_argument("--dry-run", action="store_true",
                   help="print the plan and resolve imports; load nothing")
    g.add_argument("--self-test", action="store_true",
                   help="offline unit checks; no GPU, no env, no network")

    a = p.parse_args()
    return Pi05Config(
        suite=a.suite, perturbation=a.perturbation, num_trials=a.num_trials,
        trial_offset=a.trial_offset, only_task_id=a.only_task_id,
        start_task_id=a.start_task_id, end_task_id=a.end_task_id,
        skip_task_ids=a.skip_task_ids, backend=a.backend,
        config_name=a.config_name, checkpoint=a.checkpoint, host=a.host,
        port=a.port, replan_steps=a.replan_steps, resize_size=a.resize_size,
        quantize=a.quantize, budget=a.budget, max_steps=a.max_steps,
        settle_steps=a.settle_steps, settle_gripper=a.settle_gripper,
        cam_size=a.cam_size, env_horizon=a.env_horizon,
        poll_success=a.poll_success, output_dir=a.output_dir, tag=a.tag,
        verbose=not a.quiet, dry_run=a.dry_run, self_test=a.self_test)


def main() -> int:
    cfg = _parse()
    if cfg.self_test:
        print("pi05 LIBERO-PRO runner self-test")
        return _self_test()
    run_suite(cfg)
    return 0


if __name__ == "__main__":
    sys.exit(main())

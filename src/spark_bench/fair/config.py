"""
Imports, config (FairConfig), env loading, and module side-effects for the fair runner.
"""
from __future__ import annotations

# os.environ defaults must be set before mujoco / torch imports.
import os
os.environ.setdefault('MUJOCO_GL', 'egl')
os.environ.setdefault('PYTORCH_CUDA_ALLOC_CONF', 'expandable_segments:True')

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

import numpy as np
import torch
import tyro
import mujoco

try:
    from libero.libero import benchmark
    from libero.libero.envs import OffScreenRenderEnv
    from libero.libero.utils import get_libero_path
except Exception:
    benchmark = None
    OffScreenRenderEnv = None
    get_libero_path = None

from spark_bench.libero_pro.bddl import (
    parse_bddl,
    _strip_instance,
    get_task_prompts_for_suite,
)
from spark_bench.libero_pro.planning import (
    _annotate_masks,
    _gemini_plan,
    _gemini_plan_via_dsl,
    _get_sam3,
    _library_replay_plan,
    _scripted_plan,
    bt_replay_enabled,
)
from spark_bench.libero_pro.perception import detect_scene_privileged  # noqa
from spark_bench.libero_pro.perception import (
    detect_scene,
    redetect_agentview,
    select_prompts,
)
from spark_bench.libero_pro.executor import execute_on_libero
from spark_bench.libero_pro.atomic_decomp import run_atomic_decomp
from spark_bench.libero_pro.wm_safety_gate import WMSafetyGate
from spark_real.planning.bt_label_validator import validate_and_repair_bt

# Voyager-style BT library is optional - if the import fails (e.g. in
# environments without spark_real), `get_library` is left None and the
# library write is skipped silently.
try:
    from spark_real.planning.bt_library import get_library
except Exception:  # pragma: no cover
    get_library = None  # type: ignore[assignment]

# Shadow-sim WM: best-of-N candidate-BT selection via MuJoCo rollout.
# Optional; falls through to legacy single-plan flow when import fails.
try:
    from spark_real.world_model.shadow import (
        shadow_select_and_execute, shadow_enabled, shadow_k)
except Exception:  # pragma: no cover
    shadow_select_and_execute = None  # type: ignore[assignment]
    shadow_enabled = lambda: False  # type: ignore[assignment]
    shadow_k = lambda default=3: default  # type: ignore[assignment]


# Patch torch.load for LIBERO init state files (numpy arrays in older format).
_torch_load_orig = torch.load


def _torch_load_patched(*args, **kwargs):
    kwargs.setdefault('weights_only', False)
    return _torch_load_orig(*args, **kwargs)


torch.load = _torch_load_patched


LIBERO_PRO_ROOT = Path(__file__).parent.parent / 'libero_pro'


def _init_states_search_dirs() -> list[Path]:
    """
    Search dirs for LIBERO-PRO ``.pruned_init`` files.

    Probes (in order): the LIBERO_INIT_STATES_DIR env var, the in-repo
    libero_pro mirror, and the /tmp/libero_pro_hf cache.  The first hit
    per (suite, file) wins.
    """
    out: list[Path] = []
    env = os.environ.get('LIBERO_INIT_STATES_DIR')
    if env:
        out.append(Path(env))
    out.extend([
        LIBERO_PRO_ROOT / 'libero' / 'libero' / 'init_files',
        Path('/tmp/libero_pro_hf/init_files'),
    ])
    return [d for d in out if d.exists()]


def _load_init_states_fallback(suite_name: str, task) -> Optional[list]:
    """
    Locate ``<suite>/<task>.pruned_init`` outside the libero config dir.

    LIBERO resolves init states from ``~/.libero/config.yaml``'s
    ``init_states`` path.  Perturbation suites like ``libero_10_swap``
    may be absent there; fall back to the in-repo ``libero_pro/libero/
    libero/init_files/`` mirror.  Returns ``None`` if no candidate is found.
    """
    candidates: list[Path] = []
    for base in _init_states_search_dirs():
        candidates.append(base / suite_name / task.init_states_file)
        # Try the base-suite name too - some libero_pro variants reuse the
        # base suite's init states (e.g. swap suite reuses original inits).
        base_suite = '_'.join(suite_name.split('_')[:2])
        if base_suite != suite_name:
            candidates.append(base / base_suite / task.init_states_file)
    for c in candidates:
        if c.exists():
            try:
                return torch.load(str(c))
            except Exception:
                continue
    return None


# Config

@dataclass
class FairConfig:
    suite: str = "object"
    """
    Suite: object, goal, spatial
    """
    perturbation: str = "position"
    """
    Perturbation: position, task, all
    """
    num_trials: int = 5
    """
    Trials per task (each with different init state). CaP-X uses 50.
    """
    max_steps: int = 600
    """
    Max sim steps per episode
    """
    cam_width: int = 640
    cam_height: int = 480
    no_gemini: bool = False
    privileged: bool = False
    """CaP-X S1 mode: bypass SAM3 and use MuJoCo GT body positions for the
    Gemini-planner's det_map. Gemini still runs; only perception is privileged.
    Use for S1 ceiling measurement on libero_pro 50T. Default False = S2."""
    verbose: bool = False
    output_dir: str = ""
    start_task_id: int = 0
    """
    Skip tasks before this index (resume / partial re-run).
    """
    end_task_id: int = -1
    """If >= 0, exclusive upper bound on task index (parallel task-split).
    Pairs with start_task_id to form a [start, end) slice, e.g. start=0
    end=5 runs task IDs 0..4. -1 disables (run through num_tasks)."""
    only_task_id: int = -1
    """If >= 0, run only this one task."""
    trial_offset: int = 0
    """Debug-only: offset added to the trial index when selecting the env
    seed and init state (state_idx = (trial + trial_offset) % n_states).
    Lets debugging iterate on held-out init states without touching the
    eval grid (trial 0 / default states).  0 = eval behaviour."""
    skip_task_ids: str = ""
    """
    Comma-separated task IDs to skip (e.g. ``"0,3,7"`` to skip drawer/turn tasks).
    """
    use_pyroki: bool = False
    """If True, use pyroki 6-DOF IK for side-grasp primitives (drawer, plate).
    Falls back to hand-rolled compute_ik when pyroki errors."""
    use_gemini_prompts: bool = False
    """If True, Gemini generates SAM3 detection prompts at runtime from the
    task instruction + scene image, bypassing the hand-curated
    _OBJ_VISUAL_PROMPTS dict.  Cleaner fairness comparison vs CaP-Agent0."""
    no_bddl_hints: bool = False
    """If True, don't pass BDDL-parsed pick/place target strings to the
    Gemini BT planner - only the raw :language instruction.  Matches
    CaP-Agent0's setup exactly (they read only the language field)."""
    use_graspgen: bool = False
    """If True, use NVIDIA GraspGen (diffusion) for SE(3) grasps instead of
    EquiGraspFlow.  Runs in the `graspgen` conda env subprocess."""
    use_sam3_service: bool = False
    """If True, route the multi-phase adaptive perception text- and
    point-prompt calls through the SAM3 FastAPI service (default
    http://127.0.0.1:8115) instead of loading SAM3 in-process. Default
    False so existing sweeps are untouched. Start the service once per
    machine with ``bash scripts/launch_sam3_service.sh``. Saves ~5 GB
    GPU per worker."""
    sam3_service_url: str = "http://127.0.0.1:8115"
    """
    Base URL for the SAM3 service when ``use_sam3_service`` is True.
    """
    use_tuned_prompts: bool = False
    """If True, use the offline-tuned prompts in tuned_prompts.json (pulled
    from BDDL :objects-of-interest + :goal).  Fairness middle-ground: dict
    is auto-generated from a sweep using GT centroid distance - no runtime
    GT."""
    tuned_prompts_path: str = str(Path(__file__).resolve().parent.parent / "tuned_prompts.json")
    adaptive_prompts: bool = False
    """If True, Gemini generates K=3 variants per mentioned object from
    :language alone; SAM3 self-selects the variant with exactly-1-detection
    and highest confidence.  Fully fair vs CaP (no BDDL enumeration, no GT)."""
    adaptive_k: int = 3
    """
    Variants per object for --adaptive-prompts.
    """
    multiphase_adaptive: bool = False
    """If True, run true multi-phase adaptive perception:
    Phase 1+1.5 (single Gemini call that SEES the scene + names concepts +
    proposes K variants per concept), Phase 2 (SAM3 runs every variant),
    Phase 3 (Gemini sees the mask overlays + picks the right mask OR
    returns a click point), Phase 4 (refine via click fallback to closest
    candidate; SAM3 here doesn't expose a point-prompt API).  Replaces the
    legacy --use-gemini-prompts + --adaptive-prompts pair with a single
    image-grounded flow.  Default False preserves the existing path."""
    phase3_model: str = "gemini-3.5-flash"
    """Gemini model used for the Phase 3 visual-reasoning step (mask-pick /
    click decision).  Default Flash to keep API spend down; pass
    'gemini-3.1-pro-preview' explicitly when an ablation/final run wants
    the Pro variant.  If the requested model isn't available in this SDK
    build, perception.py falls back to 'gemini-3-pro' then 'gemini-2.5-pro'."""
    use_dsl: bool = False
    """If True, route Gemini prompt construction through PromptBuilder
    (spark_dsl/prompt_builder.py) and BT execution through BTExecutor
    (spark_dsl/executor.py).  Default False preserves the legacy path.
    Adds typed validation + auto-loaded macros for free."""
    atomic_decomp: bool = False
    """If True, parse the BDDL :goal into atomic predicates and emit ONE
    sub-BT per predicate with full SAM3 + Gemini re-perception between
    each sub-BT.  Implements option (c) from atomvla_deep_read.md.
    Default False preserves the existing single-BT-per-trial baseline."""
    pre_close_retarget_min_cm: float = 4.0
    velocity_lead_s: float = 0.0
    """Occlusion floor for retargeting at the pre_close capture: with the
    gripper over the target the agentview centroid shifts 2.3 to 3.8 cm on
    unperturbed trials, so apparent moves below this bound at pre_close are
    treated as occlusion bias and the standing bind is kept (a true move at
    closure is caught by the telemetry EMPTY_CLOSE vote and regrasp)."""
    freshness_timeout_s: Optional[float] = 0.5
    """If set, the executor re-runs SAM3 between primitives when the cached
    detections are older than this many seconds.  DynamicVLA-style
    latent-aware staleness gate, adapted to a symbolic pipeline: bounded
    perception freshness, not 25 Hz continuous control.  Default 0.5s
    is aggressive: with the SAM3 service (~50-100ms/call) the per-trial
    cost is small compared to the SR gain from fresh keypoints after a
    primitive has moved objects.  Set to None to disable entirely (this
    restores the legacy 'cache forever within a trial' behaviour).
    Object-moving primitives (handle_grasp, handle_release,
    handle_push_object, handle_move_relative >= 10cm)
    additionally force-invalidate by zeroing _last_perception_t."""
    wm_safety_gate: bool = False
    """If True, sample K=wm_safety_gate_k candidate BTs at temperature 0.3
    and route them through a kinematic + V-JEPA safety gate
    (spark_bench.libero_pro.wm_safety_gate.WMSafetyGate) before execution.
    Stage 1 rejects BTs whose compiled FK terminal/intermediate EE pose
    leaves the workspace envelope [wm_unsafe_workspace_z_min, ..._max].
    Stage 2 (only if a V-JEPA checkpoint is loadable) ranks the passing
    candidates by L1 distance from predicted terminal latent to goal latent.
    Opt-in; default off so the 44.1% baseline stays untouched."""
    wm_safety_gate_k: int = 4
    """Number of candidate BTs to sample when --wm-safety-gate is enabled.
    Reuses the existing Gemini planner; K=1 reduces to the baseline path."""
    wm_unsafe_workspace_z_min: float = 0.78
    """Floor of the safe Z envelope (meters, world frame).  LIBERO tabletops
    sit ~0.86 m so 0.78 m gives ~8 cm of slack for descend-to-grasp."""
    wm_unsafe_workspace_z_max: float = 1.30
    """Ceiling of the safe Z envelope (meters, world frame).  Above ~1.3 m
    the EE is approaching joint limits / camera mount on LIBERO Franka."""
    wm_unsafe_workspace_xy_radius: float = 1.5
    """Max horizontal reach from base (meters).  Permissive bound to catch
    runaway move_relative deltas; tabletop tasks live within ~0.8 m."""
    wm_load_vjepa: bool = False
    """If True and --wm-safety-gate is on, also load V-JEPA 2-AC for Stage 2
    latent goal-similarity ranking.  Default False: Stage 1 kinematic shield
    only (cheaper, no GPU, no goal-frame requirement)."""
    selection: str = 'shield'
    """Candidate-selection mode for the gated path (--wm-safety-gate).
    'shield' (default): arm C v2 behaviour - the kinematic shield only
    VETOES; planner order among passing candidates selects (the temp-0
    primary wins unless the shield kills it).  'trace_model': after the
    kinematic shield filters, rank the passing candidates by the
    predicate-trace plan-success model (spark_bench.trace_selector) - a
    logistic model mined from the P0-a verifier trial traces, CPU-only,
    deterministic.  Fail-open: if the artifact is missing or scoring
    raises, selection falls back to planner order among passers."""
    trace_model_path: str = str(Path(__file__).resolve().parent.parent
                                / 'trace_selector' / 'artifacts'
                                / 'trace_lr.json')
    """Artifact path for selection='trace_model'."""
    save_rgb_frames: bool = False
    """If True, save per-step (RGB_agentview, RGB_wrist, action, ee_pose,
    gripper) tuples to HDF5 for successful trials. Output:
    ``{output_dir}/rgb_streams/{task_name}_T{trial}.h5``. Used downstream to
    train a V-JEPA adapter on the SPARK BT distribution. Opt-in; failed
    trials are skipped (only successful demonstrations are saved)."""
    event_captures: bool = False
    """If True, primitives trigger captures at their informative moments
    (pre-close: right before gripper closure; post-release: right after
    release + settle) instead of relying only on the periodic freshness
    gate.  The SAM3 call is issued immediately at the event so
    verification latency is measured from event to verdict; per-event
    timestamps (t_event, t_capture, t_verdict) land in
    trial_meta['capture_events'].  With this on, the freshness poll can
    be relaxed (e.g. --freshness-timeout-s 2.0 or None) since event
    captures keep the det_map fresh at exactly the moments that matter.
    Also enables the camera second vote + single local retry behind the
    grasp-telemetry EMPTY_CLOSE verdict.  Default off: baseline runs are
    byte-identical."""
    scene_diff: bool = False
    """If True, diff the current det_map against the plan-time binding
    before dispatching each keypoint-referencing primitive (per-label
    position delta, presence, target re-check), with self-caused change
    subtracted (held object + anything within scene_diff_self_radius_cm
    of the gripper).  Records land in trial_meta['scene_diffs'].  Targets
    that moved less than retarget_max_cm are retargeted in place (see
    retarget_max_cm); larger moves or a missing target abort into the
    recovery loop.  Forces a perception refresh before each gated
    primitive.  Default off."""
    retarget_max_cm: float = 10.0
    """In-flight retarget bound (U-LAG pattern): when scene_diff reports
    the current primitive's target moved by LESS than this since binding,
    the primitive's goal is updated to the fresh detection and execution
    continues (trial_meta['retarget_events']).  At or above this bound,
    or when the target is missing, execution falls through to recovery."""
    scene_diff_moved_cm: float = 1.5
    """Per-label displacement (cm) above which the scene diff flags a
    label as moved.  1.5 cm sits above SAM3 centroid + depth
    backprojection noise in sim while catching a 2 cm scripted shift."""
    scene_diff_self_radius_cm: float = 12.0
    """Self-caused-change subtraction radius: non-target labels within
    this distance of the gripper are attributed to the robot (bumped /
    occluded), never to an external scene change."""
    layered_recovery: bool = False
    """If True, replace the blind retract->replan->re-execute escalation
    with lowest-responsible-layer attribution (ANCHOR pattern):
    perception layer -> re-detect and re-bind (same plan), execution
    layer -> local retry of the same score, plan layer -> tier-3 replan
    (skipped under --no-gemini).  Attribution decisions land in
    trial_meta['recovery_attributions'].  Default off so the published
    baseline recovery numbers are untouched."""
    max_local_retries: int = 1
    """Execution-layer local retries before attribution escalates to the
    plan layer."""
    annotation_rescue: bool = False
    """If True, when a prompt still has NO detection after the existing
    prompt-selection layers have run (tuned/Gemini/adaptive/multiphase),
    ask the configured annotation provider (--annotation-provider) to
    POINT at the missing label and seed SAM3's click head with the point
    (spark_real.perception.annotations). The rescued mask goes through
    the same depth-aware backprojection as any text-prompt detection and
    the event lands in trial_meta['annotation_rescue']. Fail-open: a
    provider miss just leaves the label missing, exactly as today.
    Default off so existing sweeps are byte-identical."""
    annotation_provider: str = "er2"
    """Provider name for --annotation-rescue: er2 (Gemini Robotics-ER 2,
    flash-tier API call) | molmo (local molmoact2 conda env) | human
    (no-op outside the web UI). See spark_real/perception/providers/."""
    instance_disambig: bool = False
    """If True, when >1 instance of the pick target's label is detected
    at bind time (or at a recovery re-bind), reuse the multiphase
    Phase-3 machinery (flash-tier phase3_model sees the candidate mask
    overlays) to pick the instance satisfying the instruction's spatial
    relation, or resolve a returned click to the nearest candidate.
    LLM-free position-sticky binding then holds that identity for the
    rest of the trial.  Requires the Gemini key; default off."""


# Env loading helpers

def load_libero_env(suite_name: str, task_id: int, cfg: FairConfig):
    """
    Load a LIBERO task using ``OffScreenRenderEnv`` (same as CaP-X).
    """
    benchmark_dict = benchmark.get_benchmark_dict()
    task_suite = benchmark_dict[suite_name]()
    task = task_suite.get_task(task_id)

    # Resolve BDDL path - perturbation suites reuse base suite BDDLs.
    bddl_path = os.path.join(get_libero_path("bddl_files"),
                              task.problem_folder, task.bddl_file)
    if not os.path.exists(bddl_path):
        parts = task.problem_folder.split('_')
        base_folder = '_'.join(parts[:2]) if len(parts) > 2 else task.problem_folder
        bddl_path = os.path.join(get_libero_path("bddl_files"),
                                  base_folder, task.bddl_file)
    if not os.path.exists(bddl_path):
        bddl_path = str(LIBERO_PRO_ROOT / 'libero' / 'libero' / 'bddl_files'
                         / task.problem_folder / task.bddl_file)
    if not os.path.exists(bddl_path):
        parts = task.problem_folder.split('_')
        base_folder = '_'.join(parts[:2])
        bddl_path = str(LIBERO_PRO_ROOT / 'libero' / 'libero' / 'bddl_files'
                         / base_folder / task.bddl_file)

    env = OffScreenRenderEnv(
        bddl_file_name=bddl_path,
        camera_heights=cfg.cam_height,
        camera_widths=cfg.cam_width,
        camera_depths=True,
        horizon=2000,  # default 500 is too low for multi-step pick-place
    )
    try:
        init_states = task_suite.get_task_init_states(task_id)
    except (FileNotFoundError, OSError):
        init_states = _load_init_states_fallback(suite_name, task)
    return env, task, init_states, bddl_path


def get_perturbation_suite(base_suite: str, perturbation: str) -> str:
    """
    Map base suite + perturbation type to the LIBERO-PRO suite name.
    """
    ptype_map = {
        'position': 'swap',
        'task': 'task',
        'language': 'lan',
        'object': 'object',
        'environment': 'env',
    }
    suffix = ptype_map.get(perturbation, perturbation)
    return f"{base_suite}_{suffix}"


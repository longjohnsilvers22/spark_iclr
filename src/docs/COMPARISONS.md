# SPARK vs CaP-X: Consolidated Architecture and Code-Level Comparison

> Consolidated from architecture overview and dense code review (2026-04-08).

---

## Executive Summary

| Dimension | SPARK | CaP-X (CaP-Agent0) |
|---|---|---|
| **Paradigm** | Neurosymbolic: structured YAML behavior trees with formal fallback semantics | Code generation: LLM writes executable Python |
| **Planning LLM** | Gemini 2.0 Flash, temperature=0, 1 call per task (~$0.001) | GPT-5.4 / Claude / DeepSeek, multi-turn up to 10 iterations, optional 9-candidate ensemble (~$0.05-0.50/trial) |
| **Perception** | SAM3 (in-process singleton) + DA3 monocular depth | SAM3 (FastAPI server) + Contact-GraspNet + Molmo + rendered/stereo depth |
| **Depth** | DA3 metric monocular (any camera) or MuJoCo GT | Sim-rendered GT depth or ZED stereo (real) |
| **Control** | IK-guided mass-matrix OSC (torque level) with nullspace optimization | PyRoKi differential IK to joint position commands |
| **Recovery** | 8 deterministic physical fallback strategies (zero LLM cost) | LLM-based code regeneration with visual differencing (1-10 LLM calls per retry) |
| **Grasp planning** | Top-down with object-specific heuristics; SAM 3D Objects mesh-based planned | Contact-GraspNet: full 6-DOF SE(3) grasp from point clouds |
| **Trials** | 1 trial, deterministic (temperature=0) | 50 trials (3-candidate ensemble at temps 0.1/0.5/0.9) |
| **Parallelism** | Sequential, single process | Ray workers + shared FastAPI perception servers |
| **Artifacts** | Pass/fail + optional video | Full execution trace: code, logs, video, visual feedback per trial |
| **LIBERO score** | 36/40 = 90% zero-shot | Reported numbers require 50-trial aggregation |

---

## 1. Perception

### SAM3 Architecture

SPARK loads SAM3 as a Python singleton, in-process:

```python
_shared_sam3 = None
def get_shared_sam3():
    global _shared_sam3
    if _shared_sam3 is None:
        _shared_sam3 = SPARKPerception(sam3_threshold=0.03)
    return _shared_sam3
```

Detection runs at fixed pipeline stages: once at task start, between multi-step actions (world changed), and during fallback recovery (arm retracted, different viewpoint). Total calls per task: 1 (single-step) to 1 + steps + retries (multi-step with fallback). Prompts are batched -- all task prompts go in one call.

CaP-X runs SAM3 as a persistent FastAPI service on port 8114, auto-launched at benchmark startup. All Ray workers share it via HTTP. Detection happens only when LLM-generated code calls it:

```python
segment_sam3_text_prompt(rgb, "basket")   # masks + scores
segment_sam3_point_prompt(rgb, [[x, y]])  # masks from point prompt
```

The LLM decides when and how many times to invoke detection. Frequency is variable and depends entirely on the generated code.

| Aspect | SPARK | CaP-X |
|---|---|---|
| **Loading** | Python singleton (in-process) | FastAPI server (port 8114) |
| **Invocation** | Fixed pipeline stages | LLM-generated code decides |
| **Prompts** | Batch: all task prompts in one call | One-at-a-time per code call |
| **Frequency** | Predictable: 1-3 per task | Variable: depends on generated code |
| **Multi-worker** | Single process only | Shared server, multiple clients |
| **Latency** | ~500ms (in-process, GPU) | ~500ms + HTTP overhead per call |
| **Fallback on miss** | GT body injection (non-DA3 mode) | Molmo point prompt to SAM3 |

### Depth Estimation

SPARK with DA3 (`--use-da3`): renders RGB from MuJoCo, runs DA3 (`DA3METRIC-LARGE`) to estimate metric depth, then backprojects SAM3 masks into 3D. No GT body matching or GT position injection. The same pipeline transfers directly to the real robot (camera RGB to DA3 to SAM3 to 3D). Works with any monocular camera -- phone, webcam, industrial camera -- no stereo or depth sensor required.

CaP-X uses sim-rendered GT depth in simulation. SAM3 masks + rendered depth yield 3D point clouds and oriented bounding boxes. Contact-GraspNet operates on these point clouds for 6-DOF grasp regression. On the real robot, CaP-X requires a ZED stereo camera for metric-scale depth. Documentation (`docs/real-franka.md`) explicitly states this requirement.

---

## 2. Planning

### SPARK: Constrained YAML Behavior Trees

Input: instruction + detected keypoint labels + few-shot examples from BT library.
Output: YAML behavior tree (Gemini 2.0 Flash, temperature=0).

```yaml
tree:
  type: sequence
  children:
    - type: move_to_keypoint
      params: {keypoint_label: "red mug", offset_z: 0}
    - type: grasp
      params: {force: 100}
    - type: move_relative
      params: {dz: 0.20}
    - type: move_to_keypoint
      params: {keypoint_label: "plate", offset_z: 0.05}
    - type: release
```

Primitive set: `move_to_keypoint`, `grasp`, `release`, `move_relative`, `open_drawer`, `push_object`, `turn_knob`, `wait`.

Few-shot retrieval ranks stored skills by keyword overlap + object match, provides top-3 as examples. Voyager-style skill accumulation after successful execution means the BT library grows over time.

Benefits: inspectable before execution, verifiable (parse YAML, check action types), composable with formal BT Selector nodes (try children until one succeeds).

### CaP-X: Unconstrained Python Code

Input: task prompt + API documentation + rendered scene image.
Output: free-form Python.

```python
obs = get_observation()
masks = segment_sam3_text_prompt(obs["rgb_agentview"], "basket")
pts = get_object_3d_points_and_masks_from_language("basket")
target_pos = np.mean(pts, axis=0)
goto_pose(target_pos + [0, 0, 0.1], [0, 1, 0, 0])
close_gripper()
```

Available APIs: `get_observation`, `segment_sam3_text_prompt`, `segment_sam3_point_prompt`, `get_object_3d_points_and_masks_from_language`, `goto_pose`, `goto_joints`, `open_gripper`, `close_gripper`, `get_robot_state`, plus full numpy/scipy.

Multi-turn: after execution, a VLM describes what changed (visual differencing), and the LLM decides to REGENERATE or FINISH. Up to 10 turns. Ensemble mode: 9 candidates (3 temperatures x 3 models), synthesis model merges.

Key operational difference: CaP-X's `env.step(code_string)` calls `exec(code, globals)` where the code itself invokes control APIs. SPARK calls control primitives from a deterministic BT executor.

| Aspect | SPARK | CaP-X |
|---|---|---|
| **Output format** | Constrained YAML (8 primitives) | Unconstrained Python |
| **Verifiability** | Parse YAML, check action types | Cannot verify before execution |
| **Expressiveness** | Limited to primitive set | Arbitrary computation |
| **LLM calls per task** | 1 (temperature=0, deterministic) | 1-10+ (multi-turn + ensemble) |
| **Cost per task** | ~$0.001 (Gemini Flash) | ~$0.05-0.50 (GPT-5.4 x10 turns) |
| **Failure mode** | Bad plan to wrong action to fallback | Code crash to sandbox_rc=1 to retry |
| **Adaptability** | BT library grows with successes | Each trial is independent |

### BDDL / Task Language

SPARK (LIBERO eval): hardcoded per-task in `get_task_prompts()` with hand-written SAM3 prompts and pick/place hints. This is evaluation scaffolding, not the real pipeline.

CaP-X: extracts task language directly from BDDL:
```python
language = re.search(r'"([^"]*)"', raw_bddl).group(1)
```
Plus a `{libero_environment_goal}` template variable in config YAML that auto-populates. CaP-X's approach is cleaner and generalizes to any BDDL task without per-task configuration.

---

## 3. Control and Execution

### SPARK: Torque-Level Operational Space Control

```
For each move_to_keypoint:
  1. Compute IK solution (4 candidates from random seeds)
  2. PD in Cartesian space: F = Kp * (x_des - x) + Kd * (dx_des - dx)
  3. Task-space inertia: tau = J^T * Lambda * F + nullspace * (q_ik - q)
     where Lambda = (J * M^-1 * J^T)^-1
  4. Gravity compensation: tau += g(q)
  5. Stall detection: if error not decreasing for 200 steps, switch to joint-space PD fallback
  6. 600 sim steps per action (tuned for convergence)
```

Parameters: KP=150, damping=1.0, 4 redundant DOF for nullspace optimization.

### CaP-X: Position-Level Differential IK

```
For each goto_pose:
  1. PyRoKi differential IK to compute target joint positions
  2. Interpolate current to target in joint space
  3. Step sim with joint position commands (0.01 rad tolerance)
  4. Gripper: binary open/close via position command
```

SPARK operates at the torque level with impedance/compliance, handling contact forces naturally. CaP-X operates at the joint position level, relying on position tracking through contact. SPARK's controller is more sophisticated for articulated object manipulation (drawers, knobs).

### Grasp Planning

| Aspect | SPARK | CaP-X |
|---|---|---|
| **Strategy** | Top-down with object-specific heuristics (bowl rim pinch, bottle midpoint) | Contact-GraspNet: 6-DOF from depth point cloud |
| **Orientation** | Fixed top-down (z-axis) | Full SE(3) grasp poses |
| **Force control** | Configurable per-object (80-120N typical) | Binary open/close |
| **Z adjustment** | Heuristic: lower by 2-4cm based on object type | GraspNet handles depth from point cloud |

---

## 4. Recovery and Error Handling

### SPARK: Deterministic Physical Recovery

The fallback module (`spark_fallback/`) implements 8 physical recovery strategies that operate without any LLM calls:

| Strategy | Behavior | Trigger |
|---|---|---|
| `wiggle` | Sinusoidal lateral motion to break static friction | Grasp failed, object stuck |
| `retract_retry` | Lift + random XY offset + retry | Missed grasp, try different angle |
| `adjust_grip` | Try different gripper widths (0.08, 0.06, 0.04m) | Object slipping |
| `search_keypoint` | Expanding spiral to find lost object | SAM3 missed detection |
| `compliant_push` | Force-limited push with F/T feedback | Insertion/contact tasks |
| `spiral_search` | Multi-loop expanding pattern | Object not where expected |
| `force_guided` | Proportional force feedback for insertion | Peg-in-hole, drawer insertion |
| `rotate_align` | Rotational sweep for alignment | Orientation-sensitive tasks |

Full retry flow (code-level):

```
if not success and cfg.fallback:
    for retry in range(max_retries):  # default: 2
        retract_arm(safe_height=0.5)
        open_gripper()
        move_to_neutral([0, 0, 0.5])
        detections = sam3.detect(rgb, depth, prompts)
        bt = gemini.plan(instruction, detections)
        execute(bt)
        if check_success(): break
```

Additional within-execution recovery: grasp verification (check finger gap; if closed on nothing, lower 2.5cm and retry), OSC stall detection (switch to joint-space PD after 200 non-converging steps), and object-specific strategies (bowl rim approach, plate wrench slide).

### CaP-X: LLM-Based Code Regeneration

```
for turn in range(MULTITURN_LIMIT):  # default: 10
    obs, reward, term, trunc, info = env.step(code)
    if reward == 1.0: break
    feedback = vlm.describe_difference(before_img, after_img)
    decision = llm.query(executed_code + stdout + stderr + feedback)
    if decision == "REGENERATE":
        code = extract_code(decision_response)
    elif decision == "FINISH":
        break
```

Error handling: code exceptions caught by sandbox (traceback in stderr), `sandbox_rc = 1` if code crashed (success = False regardless of reward), SIGALRM timeout at 1000 seconds with partial artifacts saved, and up to 3 retries on timeout at the trial level.

### Comparison

| Aspect | SPARK | CaP-X |
|---|---|---|
| **Recovery cost** | 0 LLM calls (physical only) | 1-10 LLM calls per retry |
| **Recovery latency** | ~2-5 seconds (sim steps) | ~5-30 seconds (LLM round-trip) |
| **Adaptability** | Fixed strategy set | Can write entirely new approach |
| **Failure visibility** | No feedback loop (blind retry) | Visual + text feedback to LLM |
| **Determinism** | Same recovery every time | Different code each time |

---

## 5. LIBERO Benchmark Execution

### Runner Flow

SPARK (`spark_bench/run_spark_libero_pro_fair.py`):

```
for task_id in tasks:
    env = load_libero_env(suite, task_id, cfg)     # One OffScreenRenderEnv per task
    for trial in range(num_trials):
        env.seed(trial)
        env.reset()
        env.set_init_state(init_states[trial % N])  # Deterministic init
        success = run_spark_on_libero_env(env, ...)
    env.close()                                     # Destroy after all trials
```

Per trial: render RGB + depth from agentview (+ optional wrist cam), SAM3 detection (singleton), Gemini planning (temperature=0) or scripted fallback, execute via IK-guided OSC torque control, check success with `env.check_success()` (BDDL predicates), optional fallback recovery (retract, re-detect, re-plan, re-execute, up to 2 retries).

CaP-X (`capx/envs/launch.py` to `runner.py` to `trial.py`):

```
# Sequential mode (num_workers=1):
env = instantiate(env_factory)
for trial in trial_ids:
    obs, _ = env.reset(options={"trial": trial}, seed=trial)
    code = query_model(prompt + image)
    obs, reward, terminated, truncated, info = env.step(code)
    while not done:
        decision = query_model(decision_prompt + visual_feedback)
        if REGENERATE: code = extract_new_code(); env.step(code)
        if FINISH: break
    success = (reward == 1.0 and sandbox_rc == 0)

# Parallel mode (num_workers>1): Ray workers, each with own env instance
```

### Environment Lifecycle

| Aspect | SPARK | CaP-X |
|---|---|---|
| **Creation** | One `OffScreenRenderEnv` per task | One env per worker (shared across trials) |
| **Between trials** | `env.reset()` + `env.set_init_state()` | `env.reset(seed=trial)` (init state by seed) |
| **Between tasks** | `env.close()` + create new env | Same env, just `reset()` with different config |
| **Settling** | 20 MuJoCo steps post-init | 10 sim steps in `reset()` |
| **Parallel** | Sequential only (single process) | Ray workers, each with own env instance |
| **Cleanup** | Explicit `env.close()` | Garbage collected |

Neither kills envs between trials -- both reuse the environment and call `reset()`. SPARK creates a fresh env per task (different XML/BDDL), while CaP-X keeps the same env object.

### Init State Handling

Both cycle through LIBERO's pre-computed init states, but differently:

SPARK (fair runner): `state_idx = trial % len(init_states)`, then `env.set_init_state()`.

CaP-X does a double-reset: first `reset()` to clear state, then `set_init_state()` + second `reset()` to apply. The double-reset may be more robust to residual physics state.

### Success Checking

SPARK: primary method uses distance-based thresholds (XY < 3cm for pick-on-place, joint > 0.05 rad for drawers); secondary method uses BDDL predicate evaluation via subprocess (`libero_success_checker.py`); fair mode uses direct `env.check_success()` (same as CaP-X).

CaP-X: single method via `env.compute_reward()` delegated to LIBERO, returns float [0,1]. Success = `reward == 1.0 and sandbox_rc == 0`.

Both ultimately use the same LIBERO BDDL predicates. SPARK's non-fair runners have custom distance checks as a faster approximation.

### Evaluation Modes

SPARK supports four configurations:

| Mode | Command | GT Used | What It Measures |
|---|---|---|---|
| **Full GT** (v1 baseline) | `--no-sam3` | GT positions, GT prompts | Controller + planning ceiling |
| **SAM3 + GT depth** (default) | (default flags) | GT depth, GT body matching | Perception with oracle localization |
| **SAM3 + DA3** (pure perception) | `--use-da3` | None | True zero-shot perception |
| **SAM3 + DA3 + fallback** | `--use-da3 --fallback` | None | Full neurosymbolic pipeline |

CaP-X has two modes:

- **Privileged** (`privileged: true`): GT object poses from sim (ablations only)
- **Non-privileged** (default): SAM3 + depth deprojection (reported numbers)

### What Is Hardcoded in SPARK's LIBERO Evaluation

The LIBERO-PRO benchmark runner (`run_spark_libero_pro_fair.py`) uses per-task configs. These are evaluation scaffolding, not part of the core SPARK architecture:

| Component | Hardcoded in LIBERO eval | In real pipeline |
|---|---|---|
| SAM3 prompts | Per-task lists in `get_task_prompts_for_suite()` | Gemini determines from instruction |
| Pick/place objects | Per-task body names | Gemini parses from instruction |
| Multi-step sequences | Hardcoded per-task | Gemini decomposes instruction |
| Pre-actions (drawers) | Hardcoded per-task | BT Selector nodes with fallback |
| Place positions | GT site positions (non-DA3 mode) | DA3 backprojection |
| Grasp strategy | Top-down (bowl rim heuristic) | SAM 3D Objects mesh to grasp (planned) |

---

## 6. Parallelism and Scale

SPARK is sequential only: single process, one task at a time. SAM3 runs as an in-process singleton that cannot easily share across processes. GPU utilization: SAM3 + DA3 (if enabled) on one GPU, MuJoCo on CPU. Throughput: ~40 tasks/hour depending on retry count.

CaP-X uses Ray-based parallelism with a configurable `num_workers` parameter (default 10). SAM3, GraspNet, and PyRoKi run as shared FastAPI services. GPU utilization: perception servers on GPU, MuJoCo workers on CPU. Throughput roughly 10x SPARK with 10 workers, though each task runs 50 trials vs 1.

CaP-X scales better for large evaluations. The server architecture amortizes perception GPU cost across workers.

---

## 7. Configuration and Reproducibility

SPARK: dataclass config with CLI flags (`--suite`, `--no-sam3`, `--use-da3`, `--fallback`). Reproducibility: temperature=0, deterministic IK seeds, fixed prompts per task. Output: pass/fail per task, optional video.

CaP-X: YAML files with Hydra-style `_target_` instantiation + CLI overrides. Reproducibility: variable (multi-temperature ensemble, stochastic LLM). Output: per-trial directory with generated code, all model responses, execution videos, visual feedback images, and JSON summary.

CaP-X's output structure is considerably richer -- every trial preserves the full execution trace for post-hoc analysis.

---

## 8. Real Robot Deployment and Camera Calibration

### CaP-X

Real Franka deployment exists (`capx/envs/simulators/franka_real.py`) but no camera calibration code is included in-repo. All calibration is delegated to an external package (`robots_realtime`):

```
robots_realtime (external)               cap_gym (in-repo)
+-- Camera driver (ZED stereo)           +-- FrankaRealLowLevel
+-- Calibration YAML files               |   +-- msgpack server on port 9000
|   +-- configs/camera_extrinsics/       +-- Receives obs via msgpack:
|       +-- autolab_franka_zed_top.yaml  |   +-- rgb, depth
|       (position + RPY, per-setup)      |   +-- intrinsics_matrix (3x3)
+-- Streams obs via msgpack client       |   +-- pose (translation + quat)
                                         +-- Uses intrinsics for backprojection
```

Flow: `robots_realtime` loads per-setup YAML with hand-measured extrinsics, intrinsics from ZED SDK (factory calibrated), both packed into msgpack and sent to cap_gym's server. No ArUco, no checkerboard, no hand-eye calibration code. Extrinsics YAML files "must be created manually for each setup."

In simulation, camera parameters are computed from MuJoCo state (exact, no calibration error):

```python
cam_id = self.robosuite_env.sim.model.camera_name2id(camera_name)
fovy = self.robosuite_env.sim.model.cam_fovy[cam_id]
f = 0.5 * height / np.tan(fovy * np.pi / 360.0)
K = np.array([[f, 0, 0.5 * width], [0, f, 0.5 * height], [0, 0, 1]])
```

The R1Pro integration supports 3 cameras (ego ZED stereo, left/right wrist RealSense) with intrinsics from OmniGibson sensor objects.

### SPARK

SPARK's real robot path (`spark_real/`) includes a full calibration pipeline. `calibration.py` provides a `CameraCalibration` dataclass with an SVD-based rigid transform solver, and `calibrate.py` is an interactive script for collecting 6 TCP-to-pixel correspondences per camera. Calibrated extrinsics (4x4 matrices) are stored in `da3_anchor.json` with per-camera error metrics. Intrinsics are auto-extracted from Azure Kinect (pyk4a) and RealSense (pyrealsense2) device APIs. The wrist camera extrinsic is computed dynamically from robot FK. Achieved accuracy: 3-8 mm mean reprojection error.

### Comparison

| Aspect | SPARK | CaP-X |
|---|---|---|
| **Calibration in repo** | Yes (`calibration.py`, `calibrate.py`) | No (external `robots_realtime`) |
| **Calibration method** | 6-point SVD rigid transform per camera | External `robots_realtime` YAML |
| **Intrinsics source (real)** | pyk4a / pyrealsense2 device API | ZED SDK via msgpack |
| **Extrinsics source (real)** | SVD from TCP-to-pixel correspondences | Hand-measured YAML (position + RPY) |
| **Extrinsics persistence** | `da3_anchor.json` with error metrics | Per-setup YAML in external repo |
| **Intrinsics source (sim)** | MuJoCo camera FOV | MuJoCo camera FOV (same math) |
| **Depth sensor required** | No (DA3 monocular fallback) | Yes (ZED stereo) |
| **Calibration-free option** | DA3 + any RGB camera | None -- stereo required |
| **Wrist camera** | Dynamic extrinsic from robot FK | N/A |
| **Achieved accuracy** | 3-8 mm mean reprojection error | Not reported |

---

## 9. Strengths and Weaknesses

### SPARK Advantages

1. **Deterministic and reproducible**: temperature=0, same result every time. CaP-X needs 50 trials to report meaningful statistics.
2. **Low LLM cost**: 1 Gemini Flash call per task (~$0.001). CaP-X: 1-10+ GPT-5.4 calls per trial x 50 trials (~$2.50-25 per task).
3. **Physical recovery without LLM**: retract-retry is fast (2-5 sec sim) and costs nothing.
4. **Torque-level control**: OSC with nullspace optimization handles contact forces naturally, better for articulated objects.
5. **DA3 monocular depth**: works with any camera, no depth sensor required.
6. **Skill accumulation**: BT library grows over time (Voyager-style), future tasks benefit from past successes.
7. **Inspectable plans**: YAML BTs can be verified before execution.

### SPARK Disadvantages

1. **No parallelism**: single-process sequential evaluation.
2. **Per-task configuration**: hardcoded prompts in `get_task_prompts()` for LIBERO eval.
3. **Limited primitive set**: 8 primitives may not cover all manipulation strategies.
4. **No visual feedback loop**: does not observe execution results to adapt.
5. **Single trial**: no confidence intervals from 1-trial reporting.
6. **Sparse output artifacts**: pass/fail only.
7. **GT body injection**: non-fair runners inject GT positions when SAM3 misses, inflating perception numbers.

### CaP-X Advantages

1. **Maximum flexibility**: LLM can write arbitrary Python, no primitive constraints.
2. **Multi-turn adaptation**: visual feedback + code regeneration allows observation and correction.
3. **Parallel evaluation**: Ray workers + shared perception servers scale linearly.
4. **Rich artifacts**: every trial preserves full execution trace.
5. **6-DOF grasping**: Contact-GraspNet produces full SE(3) grasp poses, not just top-down.
6. **No task-specific config**: uses LIBERO's built-in task language and BDDL directly.
7. **Ensemble planning**: multiple temperatures + synthesis model improves robustness.

### CaP-X Disadvantages

1. **High LLM cost**: multi-turn + ensemble makes each task expensive.
2. **Non-deterministic**: stochastic LLM outputs require many trials for statistical significance.
3. **Unverifiable plans**: generated Python can crash, loop, or produce nonsensical actions.
4. **LLM-dependent recovery**: every retry requires a full model round-trip.
5. **No skill accumulation**: each trial starts from scratch.
6. **Position-level control**: joint position commands handle contact forces less gracefully than torque control.
7. **Complex infrastructure**: requires 4 servers (LLM proxy, SAM3, GraspNet, PyRoKi) running simultaneously.

---

## 10. Cross-Pollination Opportunities

### Ideas to adopt from CaP-X into SPARK

| Idea | Benefit | Effort |
|---|---|---|
| Ray parallelism | 10x eval throughput | Medium: extract SAM3 to FastAPI server |
| SAM3 as shared server | Multi-worker + cross-process sharing | Low: FastAPI wrapper around existing singleton |
| Rich trial artifacts | Better debugging + paper figures | Low: save detections, plans, videos per trial |
| Visual feedback for recovery | Smarter retry (not blind) | Medium: VLM call during fallback |
| Task language from BDDL | Remove hardcoded `get_task_prompts()` | Low: parse BDDL file directly |
| Multi-trial statistics | Confidence intervals for paper | Low: loop + aggregate |

### Ideas to adopt from SPARK into CaP-X

| Idea | Benefit | Effort |
|---|---|---|
| OSC torque control | Better contact handling | High: replace PyRoKi pipeline |
| BT library / skill accumulation | Improve over time without retraining | Medium: add storage + retrieval |
| DA3 monocular depth | Remove depth sensor requirement | Low: drop-in replacement |
| Deterministic single-shot | Reproducible results, cheap eval | Low: set temperature=0, 1 trial |
| Physical fallback strategies | Recovery without LLM cost | Medium: implement primitives |

---

## 11. Novel Contributions (SPARK vs CaP-X)

1. **Neurosymbolic BTs with formal fallback** -- structured, inspectable, verifiable plans with 8 physical recovery strategies. CaP-X has no equivalent.
2. **DA3 monocular depth** -- works with any camera, no depth sensor. Enables real-world deployment on UR10e with a standard webcam.
3. **OSC with nullspace optimization** -- more compliant, better contact handling, stall detection with automatic fallback to joint-space control.
4. **Deterministic single-shot** -- 1 trial, temperature=0, reproducible. CaP-X needs 50 trials with ensemble for stable numbers.
5. **SAM 3D Objects** (in progress) -- single-image 3D mesh reconstruction for grasp planning, a different approach from Contact-GraspNet (reconstruct then plan vs point cloud then regress).

---

## 12. Roadmap (from NVIDIA Grant Proposal)

- **Phase 1 (now - April 30, 2026)**: SPARK v1 benchmark + real UR10e + DA3 + fallback
- **Phase 2 (Jan-June 2026)**: VLA compiler (trained model generates constrained YAML scores), Sparse World-VLA for proactive safety, SAE interpretability
- **Phase 3 (July-Dec 2026)**: MoE-VLA with vision + force/torque experts, TensorRT on Jetson Orin, protocol specification

Target venues: CoRL 2026, ICRA 2027, RSS 2027, IEEE RA-L

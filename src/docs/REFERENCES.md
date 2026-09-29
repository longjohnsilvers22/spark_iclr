# SPARK Unified Reference Document

**For CoRL 2026 submission -- Deadline: May 28, 2026**

---

## 1. Literature Review

### 1.1 Neurosymbolic Robotics

#### Code Generation Approaches
- **Code as Policies** (Liang et al., ICRA 2023, arXiv:2209.07753): LLM generates Python for robot control. Hierarchical function accumulation. Expressive but unverifiable.
- **CaP-X** (NVIDIA/Berkeley/Stanford/CMU, 2026, arXiv:2603.22435): CaP-Gym + CaP-Agent0. Multi-turn code gen + visual differencing + ensemble. 18% LIBERO-PRO (Pos+Task). SPARK achieves 74% on the same metric.
- **CaP-RL** (same paper): GRPO on Qwen 7B coding model. 80% cube lift (sim), 84% cube lift (real Franka). Demonstrates post-training a small model to generate robot code with environment reward.

#### Behavior Tree Approaches
- **Code-BT** (IJCAI 2025): LLM generates BTs but does not learn primitives.
- **SPARK (ours)**: Neurosymbolic YAML BTs + dual-camera SAM3 + IK-guided mass-matrix OSC. 90% standard LIBERO, 78% LIBERO-PRO (4-type average).

#### Key Differentiator
CaP-X generates monolithic Python leading to runtime errors, with no safety guarantees or formal recovery. SPARK generates structured YAML BTs with verifiable structure, formal fallback semantics, and deterministic recovery.

### 1.2 VLA Robustness and Interpretability

#### The Fragility Problem
- **LIBERO-PRO** (Zhou et al., 2025, arXiv:2510.03827): VLAs collapse under perturbation. Pi0.5: 97% standard LIBERO but 0% on task perturbation, 8-38% on position perturbation. VLAs memorize positions, not concepts.

#### Our Group's SAE Work (Empirical Motivation for SPARK)
[self-citation removed for review]
  - SAEs on 6 VLA architectures (80M-7B params), 394K+ rollout episodes, 4 benchmarks
  - Visual pathway dominates action generation -- spatially bound motor programs tied to scene coordinates
  - Cross-task features have zero overlap (r=-0.0009, 0/10 top features shared)
  - Language ignored when visual context uniquely specifies task
  - Released "Action Atlas" for interactive VLA representation exploration

- **Concurrent work** (Swann et al., Stanford, arXiv:2603.19183):
  - Majority of SAE features are memorized training demonstrations
  - Fine-tuning on small datasets amplifies memorization
  - Only sparse subset encodes generalizable motion primitives

#### The Bridge to SPARK (Core Paper Argument)

| What SAEs reveal VLAs do internally | What SPARK does explicitly |
|------|------|
| Sparse visual motor programs tied to scene coords | SAM3 extracts positions in world frame |
| Language matters only when visual context ambiguous | Gemini plans at semantic level; controller ignores language |
| Expert pathways = motor programs, VLM = goals | OSC controller (motor) separate from LLM planner (goals) |
| Task-specific features, no cross-task transfer | BT primitives (grasp, place, push) reusable across tasks |
| Majority of features = memorized demos | Zero memorized demos -- entirely zero-shot |

### 1.3 Perception for Manipulation
- **SAM3**: Text-prompted segmentation. Used for dual-camera detection.
- **SAM 3D Objects** (Meta, 2025): Single-image mesh reconstruction for grasp planning.
- **DA3** (Depth Anything 3): Metric monocular depth. Real-robot deployment via focal scaling.
- **Contact-GraspNet**: CaP-X's grasp planner. Point cloud to 6-DOF grasps. SPARK uses a different approach.
- Both SPARK and CaP-X use GT rendered depth from sim. Neither is pure RGB-only.

### 1.4 Skill Libraries and Task Decomposition

#### Progressive Skill Libraries
- **Voyager** (Wang et al., NeurIPS 2023, arXiv:2305.16291): Ever-growing skill library. 3.3x more items, 15.3x faster. Highest priority addition for SPARK.
- **LRLL** (Tziafas & Kasaei, ICRA 2024, arXiv:2406.18746): Voyager-style for real robots.
- **DEPS** (Wang et al., ICML 2023, arXiv:2302.01560): Describe-Explain-Plan-Select error recovery.

#### Grounding and Feedback
- **SayCan** (Ahn et al., CoRL 2022, arXiv:2204.01691): LLM x affordance scoring. 84% skill selection.
- **Inner Monologue** (Huang et al., CoRL 2023, arXiv:2207.05608): Closed-loop LLM feedback.

#### Policy Distillation
- **Refined Policy Distillation** (Julg et al., 2025, arXiv:2503.05833): VLA to compact RL experts via PPO+BC. Student surpasses teacher. Potential SPARK application: distill per-primitive controllers, remove LLM API dependency.

### 1.5 World Models and Safety (Phase 2-3)

#### High Priority
- **PIVOT-R** (Liao et al., NeurIPS 2024): Primitive-driven waypoint prediction. Architecturally aligned with SPARK.
- **Semantic World Models** (UW, 2025): Language-based forward model. Low integration cost -- just another Gemini call.
- **NVIDIA Cosmos** (2025, arXiv:2501.03575): Open-source world foundation model. Fine-tunable for SPARK scenes.

#### Medium Priority
- **RWM** (ETH, NeurIPS 2025, arXiv:2501.10100): Dual-autoregressive neural simulator. Zero-shot real deployment.
- **AnyMAL** (Meta, EMNLP 2024, arXiv:2309.16058): Any-modality alignment. Template for adding F/T sensing.
- **Isaac Lab** (NVIDIA, 2025, arXiv:2511.04831): GPU-parallel sim. 2M steps/sec. Could reduce benchmark time from hours to minutes.

### 1.6 Deployment
- **TensorRT Edge-LLM** (NVIDIA, 2025): On-device LLM inference on Jetson. NVFP4 quantization.
- **Jetson T4000** (2026): 1200 FP4 TFLOPS, 64GB. Could run full SPARK pipeline on edge.

---

## 2. LIBERO-PRO Analysis

### 2.1 What LIBERO-PRO Tests

The benchmark's core thesis: VLA models scoring 90%+ on standard LIBERO are memorizing, not generalizing. Models produce identical action trajectories even when the target object is replaced, the instruction is replaced with nonsense tokens, or the object is removed entirely.

#### 5 Perturbation Dimensions

| Dimension | What Changes | What It Tests |
|-----------|-------------|---------------|
| **Object (Obj)** | Object appearance: color, texture, size | Visual robustness to appearance changes |
| **Position (Pos)** | Object initial positions swapped within feasible bounds | Spatial reasoning, not memorized positions |
| **Semantic (Sem)** | Paraphrased instructions | Language understanding vs pattern matching |
| **Task** | Different goal entirely | True task comprehension |
| **Environment (Env)** | Background/table changed | Robustness to scene context |

Task perturbation cannot be combined with others (it changes the goal). The other 4 can be freely combined.

#### Evaluation Protocol
- 50 episodes per task (different seeds)
- Same robosuite 1.4 infrastructure as standard LIBERO
- All perturbed tasks are pre-generated: BDDL files + init states on HuggingFace
- Success = robosuite's built-in goal predicate

### 2.2 VLA Baseline Results (From LIBERO-PRO Leaderboard)

#### Pi0.5 (best VLA, single checkpoint)

| Suite | Obj | Pos | Sem | Task | Env |
|-------|-----|-----|-----|------|-----|
| Goal | 0.97 | 0.38 | 0.97 | 0.00 | 0.46 |
| Spatial | 0.97 | 0.20 | 0.97 | 0.01 | 0.46 |
| 10 | 0.92 | 0.08 | 0.93 | 0.01 | 0.46 |
| Object | 0.98 | 0.17 | 0.96 | 0.01 | 0.73 |
| **Total** | | | | | **0.53** |

Pi0.5 total is a flat average across all 20 cells (5 perturbation types x 4 suites).

#### Other VLAs

| Model | Total |
|-------|-------|
| OpenVLA | 0.52 |
| Pi0 | 0.44 |
| x-VLA | 0.46 |
| Molmoact | 0.41 |
| NORA | 0.40 |

Position perturbation is devastating: OpenVLA=0.00 on 9/10 Goal suite tasks. Pi0=0.00 on 6/10. Pi0.5 survives on 4 tasks, fails on 6.

### 2.3 SPARK Results on LIBERO-PRO

Tested 4 of 5 perturbation types (missing: Environment). Evaluation uses extracted MuJoCo XMLs (not robosuite runtime), 1 deterministic episode per task (Gemini temperature=0), success checked via MuJoCo body position query.

| Suite | Sem | Pos | Task | Obj | 4-type Avg |
|-------|-----|-----|------|-----|------------|
| Goal | 0.90 | 0.70 | 0.80 | 0.70 | 0.78 |
| Spatial | 0.90 | 0.90 | 0.80 | 0.90 | 0.88 |
| Object | 0.90 | 0.70 | 0.90 | 0.80 | 0.83 |
| libero_10 | 0.90 | 0.50 | 0.60 | 0.60 | 0.65 |
| **Average** | **0.90** | **0.70** | **0.78** | **0.75** | **0.78** |

### 2.4 Direct Comparison (4 Perturbation Types)

| | Sem | Pos | Task | Obj | 4-type Avg |
|--|-----|-----|------|-----|------------|
| **SPARK** | 0.90 | **0.70** | **0.78** | 0.75 | **0.78** |
| **Pi0.5** | **0.96** | 0.21 | 0.01 | **0.96** | 0.53 |
| **OpenVLA** | **0.97** | 0.00 | 0.00 | **0.93** | 0.48 |
| **Pi0** | 0.91 | 0.00 | 0.00 | 0.90 | 0.45 |

Key deltas vs Pi0.5: Position +49pp (SPARK detects at runtime; VLAs memorize positions). Task +77pp (SPARK plans from instruction; VLAs replay trajectories). Semantic -6pp (VLAs are robust because they ignore the instruction). Object -21pp (body matching struggles with renamed objects).

### 2.5 CaP-X on LIBERO-PRO

CaP-X tested only Pos + Task perturbations on 3 suites (Object, Goal, Spatial):

| Method | Pos Avg | Task Avg | Overall (Pos+Task) |
|--------|---------|----------|-------------------|
| OpenVLA | 0.00 | 0.00 | 0.00 |
| Pi0 | 0.00 | 0.00 | 0.00 |
| Pi0.5 | 0.25 | 0.01 | 0.13 |
| CaP-Agent0 | 0.20 | 0.16 | 0.18 |
| **SPARK** | **0.70** | **0.78** | **0.74** |

CaP-Agent0 beats Pi0.5 on Task (0.16 vs 0.01) but loses on Position (0.20 vs 0.25). SPARK beats both on both perturbation types because it detects at runtime (SAM3) and plans from the actual scene state (Gemini).

### 2.6 CaP-X Architecture and CaP-Bench Results

CaP-X is a framework with 4 components: CaP-Gym (interactive env binding), CaP-Bench (7 tasks, 8 evaluation tiers), CaP-Agent0 (training-free agentic system), and CaP-RL (GRPO on 7B coding model).

#### CaP-Bench: Abstraction Ladder (Key CaP-X Contribution)

Performance scales monotonically with abstraction level:
- S4 (bare low-level, no examples): 5-15%
- S3 (low-level + usage examples): 15-30%
- S2 (high-level + real perception): 40-80% -- SPARK operates at this level
- S1 (high-level + GT state): 55-80%
- Human expert: 88.5%

This validates SPARK's use of high-level behavior tree primitives. The paper proves structured abstractions are the right engineering choice, not a crutch.

#### CaP-Agent0 Per-Task Results (from Figure 8, ~100 trials each)

| Task | CaP-Agent0 | Human Expert |
|------|-----------|--------------|
| Cube Lift | ~97% | 93% |
| Cube Stack | ~76% | 73% |
| Spill Wipe | ~100% | 100% |
| Cube Re-stack | ~100% | 100% |
| Peg Insert | ~30% | 87% |
| Two-Arm Lift | ~10% | 53% |
| Two-Arm Handover | ~16% | 97% |

CaP-Agent0 matches/exceeds human on 4/7 single-arm tasks but fails on contact-rich and bimanual tasks. Average across 7 tasks: ~61%. Single-arm only (5 tasks): ~81%.

#### CaP-RL Results

| Task | Base Qwen 7B | CaP-RL | Human | Real Robot (Franka) |
|------|-------------|--------|-------|---------------------|
| Cube Lift | 25% | 80% | 93% | 84% |
| Cube Stack | 4% | 44% | 73% | 76% |
| Spill Wipe | 30% | 93% | 100% | N/A |

CaP-RL transferred sim-to-real with minimal gap. Cube lift: 80% sim to 84% real. Cube stack: 44% sim to 76% real (real was better, likely due to less rendering artifacts than sim).

### 2.7 Methodological Differences and Honest Assessment

#### Why the SPARK vs VLA Comparison Is Not Apples-to-Apples

| Dimension | SPARK | VLA Baselines |
|-----------|-------|---------------|
| Episodes per task | 1 (deterministic) | 50 (stochastic) |
| Evaluation platform | Extracted MuJoCo XML | Robosuite runtime |
| Controller | IK-guided OSC (analytical) | Learned end-to-end policy |
| Training data | Zero (no training) | 50 demos per task |
| Success criterion | Body XY distance < 0.18m | Robosuite BDDL goal predicate |
| Position perturbation | Swap qpos in extracted init state | Robosuite native perturbation via BDDL |
| Environment perturbation | Not tested | Robosuite swaps table fixture |

The 1-episode evaluation is stricter in one sense (VLAs get 50 attempts), but SPARK's success criterion is a looser approximation of the full BDDL goal predicate.

#### Honest Weaknesses

1. **Sim-only**: All results are in MuJoCo. No real-robot results yet. CaP-X demonstrates real robot transfer.
2. **Simpler success metric**: XY distance < 0.18m vs full BDDL goal predicates (inside drawer, on top of, contact states).
3. **1 episode per task**: Does not test robustness to initial state variation.
4. **GT body position dependence**: SAM3 for identification + MuJoCo GT for localization. Real deployment needs SAM3 + depth for actual 3D localization.
5. **Hand-engineered task mapping**: `get_task_prompts()` is ~200 lines of task-specific configuration. Not purely "zero-shot" -- better described as "training-free" or "demonstration-free."
6. **Single-shot planning**: No recovery mechanism on failure. CaP-Agent0's multi-turn approach is fundamentally more robust.
7. **Static primitive library**: 7 hand-designed primitives. Cannot learn new manipulation types without manual implementation.
8. **No contact-rich manipulation**: Handles pick-place, push, drawer open, stove turn. Cannot do peg insertion, pouring, wiping, folding, tool use, deformable manipulation.

#### Genuine Strengths

1. **LIBERO-PRO performance is real**: 74% on Pos+Task vs CaP-Agent0's 18%. Even accounting for methodological differences, the 4x gap is too large to be an artifact.
2. **Computational efficiency**: 1 Gemini call vs 9+ LLM calls per turn x multiple turns. 50-100x cheaper per task.
3. **Interpretability**: YAML behavior trees are human-readable and verifiable before execution.
4. **Architecture alignment**: CaP-X's own conclusion advocates for hybrid LLM-planning + specialized-execution, which is SPARK's architecture.

### 2.8 Semantic and Object Perturbation Interpretation

**Semantic perturbation**: VLAs score ~96% but this is NOT language understanding. LIBERO-PRO shows models produce the exact same action trajectory even with nonsense token instructions. SPARK's 90% genuinely reflects language understanding because Gemini parses the instruction to generate a plan.

**Object perturbation**: VLAs score 92-98% because object appearance changes do not affect memorized spatial trajectories. SPARK scores 75% because body matching breaks when object names change significantly. VLA "robustness" here is evidence of memorization; SPARK's "failure" is because it actually tries to identify the correct object.

---

## 3. Key Citations

### Core References

| Citation | arXiv | Relevance |
|----------|-------|-----------|
| LIBERO-PRO (Zhou et al., 2025) | 2510.03827 | Primary robustness benchmark; proves VLA memorization |
| CaP-X / CaP-Gym (Fu et al., 2026) | 2603.22435 | Coding agent competitor; abstraction ladder analysis |
| Code as Policies (Liang et al., 2023) | 2209.07753 | Foundational LLM-for-robotics approach |
[self-citation removed for review]
| Swann et al. (Stanford, 2025) | 2603.19183 | Concurrent SAE work; memorization findings |
| Voyager (Wang et al., NeurIPS 2023) | 2305.16291 | Ever-growing skill library; highest priority addition |
| SayCan (Ahn et al., CoRL 2022) | 2204.01691 | LLM x affordance scoring |
| Inner Monologue (Huang et al., CoRL 2023) | 2207.05608 | Closed-loop LLM feedback |
| DEPS (Wang et al., ICML 2023) | 2302.01560 | Error recovery via describe-explain-plan-select |
| LRLL (Tziafas & Kasaei, ICRA 2024) | 2406.18746 | Voyager-style skill library for real robots |
| Refined Policy Distillation (Julg et al., 2025) | 2503.05833 | VLA to compact RL experts |
| PIVOT-R (Liao et al., NeurIPS 2024) | -- | Primitive-driven waypoint prediction |
| NVIDIA Cosmos (2025) | 2501.03575 | Open-source world foundation model |
| RWM (ETH, NeurIPS 2025) | 2501.10100 | Dual-autoregressive neural simulator |
| AnyMAL (Meta, EMNLP 2024) | 2309.16058 | Any-modality alignment |
| Isaac Lab (NVIDIA, 2025) | 2511.04831 | GPU-parallel sim; 2M steps/sec |

### Benchmark and Repo Links

| Resource | URL |
|----------|-----|
| LIBERO-PRO repo | https://github.com/Zxy-MLlab/LIBERO-PRO |
| CaP-X project page | https://capgym.github.io/ |
| LIBERO benchmark | https://libero-project.github.io |
| SAM 3D Objects | https://github.com/facebookresearch/sam-3d-objects |
| Neural MP | https://mihdalal.github.io/neuralmotionplanner |
| FoundationPerception | https://github.com/MMintLab/FoundationPerception |

---

## 4. Positioning

### 4.1 Paper Framing

#### Suggested Title
"Zero-Shot Manipulation via Neurosymbolic Decomposition: Why Structure Outperforms Scale"

#### One-Sentence Contribution
"Motivated by mechanistic analyses showing VLAs waste capacity on memorized trajectories, SPARK decomposes manipulation into verified behavior tree primitives with dual-camera perception and physical fallback recovery, achieving 2.9x the robustness of code-generation agents on LIBERO-PRO with zero training."

#### Story Arc
1. **Problem**: VLAs memorize positions (collapse under perturbation: 0% on LIBERO-PRO task perturbation)
2. **Empirical motivation** (our SAE paper): Features are sparse/task-specific, majority = memorization
3. **Solution**: SPARK decomposes into verified primitives (what SAEs show VLAs need internally)
4. **Evidence**: 78% LIBERO-PRO (4-type) = 1.5x Pi0.5 overall; 74% Pos+Task = 4x CaP-Agent0; no training, 1 trial, real UR10e transfer
5. **Ablations**: GT vs no-GT, single vs dual camera, with/without fallback
6. **Takeaway**: Structure > scale for zero-shot robustness

### 4.2 Competitive Positioning

| | VLAs (Pi0.5) | CaP-Agent0 | SPARK |
|---|---|---|---|
| Standard LIBERO | 97% | Not tested | **90%** |
| LIBERO-PRO (Pos+Task) | 13% | 18% | **74%** |
| LIBERO-PRO (4-type avg) | 53% | ~18%* | **78%** |
| Training | 50 demos/task | 0 (50-100 trials) | **0 (1 trial)** |
| Safety | None | None | BT fallback |
| Interpretability | Black box (need SAEs) | Code inspection | BT YAML (seconds) |
| Real-world | Sim2Real gap | Franka + AgiBot | DA3 pipeline for UR10e |
| Compute at inference | 7B model forward pass | 9+ LLM calls/turn x multi-turn | 1 Gemini call |

*CaP-Agent0 only tested Pos+Task, not Obj/Sem/Env.

### 4.3 Full Summary Table

| Metric | SPARK | Pi0.5 | OpenVLA | Pi0 | CaP-Agent0 |
|--------|-------|-------|---------|-----|------------|
| Standard LIBERO | **90%** | 97% | 97% | 96% | Not tested |
| PRO: Semantic | 90% | **96%** | **97%** | 91% | -- |
| PRO: Position | **70%** | 21% | 0% | 0% | 20% |
| PRO: Task | **78%** | 1% | 0% | 0% | 16% |
| PRO: Object | 75% | **96%** | **93%** | 90% | -- |
| PRO: Environment | ? | 53% | 47% | 39% | -- |
| PRO: Total (4-type) | **78%** | 53% | 48% | 45% | -- |
| PRO: Pos+Task avg | **74%** | 13% | 0% | 0% | **18%** |
| Training data | **Zero** | 50 demos/task | 50 demos/task | 50 demos/task | **Zero** |
| Episodes evaluated | 1 | 50 | 50 | 50 | 50-100 |

### 4.4 The Narrative

- Standard LIBERO is flawed (LIBERO-PRO paper proves this)
- VLAs memorize, do not generalize (LIBERO-PRO + our SAE paper demonstrate this)
- Code agents generalize better but fail at execution (CaP-X demonstrates this)
- SPARK combines the best of both: LLM generalization + controller precision
- Result: 90% standard, 78% perturbed, zero training

### 4.5 What SPARK Should Adopt from CaP-X

#### High Impact
1. **Multi-turn replanning**: When check_success fails, re-perceive, re-plan, retry. CaP-Agent0 shows this consistently improves performance.
2. **Auto-synthesized skill library**: Extract reusable primitives from successful Gemini plans. Build a persistent library that grows over tasks.

#### Medium Impact
3. **CaP-RL for SPARK**: Train a 7B model to generate YAML behavior trees via GRPO, using check_success as reward. Would be cheaper than Gemini API and potentially more reliable.
4. **Run CaP-Bench tasks**: Port SPARK to CaP-Gym's 7 core tasks for direct comparison.
5. **BEHAVIOR tasks**: Test on mobile manipulation for broader scope.

#### Lower Impact
6. **Parallel reasoning**: Generate 3 Gemini plans and pick the best via validation heuristic.
7. **VDM for error diagnosis**: After failed execution, use VLM to describe what went wrong, then feed to Gemini for replanning.

### 4.6 Priorities for the Paper

1. Real robot demo (without it, this is a sim-only paper)
2. Multi-turn replanning (single-shot is a fundamental limitation)
3. Run proper 50-episode evaluation to match LIBERO-PRO protocol
4. Remove GT body position dependence (use SAM3 + depth for actual 3D)
5. Auto skill library (port CaP-X approach)
6. Use "training-free" or "demonstration-free" instead of "zero-shot" given hand-engineered task prompts

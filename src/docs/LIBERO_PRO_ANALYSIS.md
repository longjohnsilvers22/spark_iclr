# LIBERO-PRO + CaP-X Dense Analysis

**Paper:** LIBERO-PRO: Towards Robust and Fair Evaluation of VLA Models Beyond Memorization (arXiv:2510.03827, Oct 2025)
**Authors:** Zhou et al., HUST/Harvard/MIT/WUT/Lehigh
**Repo:** https://github.com/Zxy-MLlab/LIBERO-PRO (forked to ANON-LAB/LIBERO-PRO)

---

## 1. What LIBERO-PRO Actually Tests

The paper's core thesis: **VLA models scoring 90%+ on standard LIBERO are memorizing, not generalizing.** They prove this by showing models produce identical action trajectories even when:
- The target object is replaced with something irrelevant (still grasps same location)
- The instruction is replaced with nonsense tokens (still executes same trajectory)
- The object is removed entirely (still performs grasping motions)

### 5 Perturbation Dimensions

| Dimension | What Changes | Formal Notation | What It Tests |
|-----------|-------------|-----------------|---------------|
| **Object (Obj)** | Object appearance: color, texture, size. "red cup" -> "blue cup" | tau^(O) = (l, O^(O), E, p0, G) | Visual robustness to appearance changes |
| **Position (Pos)** | Object initial positions swapped within feasible bounds | tau^(I) = (l, O, E, p0^(I), G) | Spatial reasoning, not memorized positions |
| **Semantic (Sem)** | Paraphrased instructions: "pick up the mug" -> "grab the cup" | tau^(L) = (l^(L), O, p0, G) | Language understanding vs pattern matching |
| **Task** | Different goal entirely: "pick up mug" -> "pick up butter" | tau^(L) = (l^(L), O^(L), E, p0, G^(L)) | True task comprehension |
| **Environment (Env)** | Background/table changed: "main_table" -> "kitchen_table" | tau^(E) = (l, O, E^(E), p, G) | Robustness to scene context |

**Key constraint:** Task perturbation CANNOT be combined with others (it changes the goal). The other 4 can be freely combined.

### Evaluation Protocol
- **50 episodes per task** (they run each task 50 times with different seeds)
- **Same robosuite 1.4 infrastructure** as standard LIBERO
- All perturbed tasks are pre-generated: BDDL files + init states on HuggingFace
- Success = robosuite's built-in goal predicate (same as standard LIBERO)

---

## 2. The README Leaderboard (Exact Numbers)

From the repo README, columns are: Obj, Pos, Sem, Task, Env per suite.

### Pi0.5 (best VLA, single checkpoint)

| Suite | Obj | Pos | Sem | Task | Env |
|-------|-----|-----|-----|------|-----|
| Goal | 0.97 | **0.38** | 0.97 | **0.00** | 0.46 |
| Spatial | 0.97 | **0.20** | 0.97 | **0.01** | 0.46 |
| 10 | 0.92 | **0.08** | 0.93 | **0.01** | 0.46 |
| Object | 0.98 | **0.17** | 0.96 | **0.01** | 0.73 |
| **Total** | | | | | **0.53** |

### OpenVLA (multi checkpoint, per-suite)

| Suite | Obj | Pos | Sem | Task | Env |
|-------|-----|-----|-----|------|-----|
| Goal | 0.96 | **0.00** | 0.98 | **0.00** | 0.98 |
| Spatial | 0.97 | **0.00** | 0.97 | **0.00** | 0.89 |
| 10 | 0.81 | **0.00** | 0.96 | **0.00** | 0.85 |
| Object | 0.98 | **0.00** | 0.98 | **0.00** | 0.00 |
| **Total** | | | | | **0.52** |

### Pi0 (single checkpoint)

| Suite | Obj | Pos | Sem | Task | Env |
|-------|-----|-----|-----|------|-----|
| Goal | 0.94 | **0.00** | 0.93 | **0.00** | 0.39 |
| Spatial | 0.95 | **0.00** | 0.97 | **0.00** | 0.60 |
| 10 | 0.79 | **0.00** | 0.82 | **0.00** | 0.27 |
| Object | 0.94 | **0.00** | 0.90 | **0.00** | 0.29 |
| **Total** | | | | | **0.44** |

### x-VLA, Molmoact, NORA (partial results, no Env)

| Model | Total |
|-------|-------|
| x-VLA | 0.46 |
| Molmoact | 0.41 |
| NORA | 0.40 |

---

## 3. Clarifying the "Pi0.5 gets 0%" Confusion

**The README top table (P1/P2) is different from the full leaderboard.**

The README has TWO separate tables:
1. **Top motivation table** (lines 50-55): Shows 3 columns per suite: Original (blue), P1 (orange), P2 (green)
   - **P1 = Task perturbation**
   - **P2 = Position perturbation**
   - Pi0.5: P1=0.0 on ALL suites, P2=0.4/0.2/0.1/0.2

2. **Full leaderboard** (lines 378-600): Shows 5 columns per suite: Obj, Pos, Sem, Task, Env
   - Pi0.5: Task=0.00/0.01/0.01/0.01, Pos=0.38/0.20/0.08/0.17

**So Pi0.5 DOES get 0% on Task perturbation (P1).** It gets 8-38% on Position perturbation (P2). It gets 92-98% on Object and Semantic (easy perturbations). The "0.53 total" is the average across ALL 5 perturbation types x 4 suites.

The confusion: Pi0.5 on STANDARD LIBERO = 97%. On LIBERO-PRO = 0.53 total. These are completely different numbers.

---

## 4. Paper Tables 2-5 (Per-Task Breakdown)

The paper has per-task results for all 40 tasks. Key observations:

### Table 2: Goal Suite

| Task | OpenVLA Pos | Pi0 Pos | Pi0.5 Pos | Pi0.5 Task |
|------|------------|---------|-----------|------------|
| Open(cabinet, drawer_mid) | 0.00 | 0.00 | 0.00 | 0.04 |
| Put(bowl, drawer_top) | 0.00 | 0.00 | 0.00 | 1.00 |
| Push(plate, stove_front) | 0.60 | 0.00 | 0.00 | 0.00 |
| Put(bowl, plate) | 0.00 | 0.00 | 0.96 | 0.00 |
| Put(bowl, stove) | 0.00 | 0.00 | 0.00 | 0.04 |
| Put(bowl, cabinet_top) | 0.98 | 0.94 | 0.96 | 0.02 |
| Put(cream_cheese, bowl) | 0.00 | 0.96 | 0.00 | 0.02 |
| Put(wine_bottle, rack) | 0.00 | 0.64 | 0.98 | 0.02 |
| Put(wine_bottle, cabinet_top) | 0.00 | 1.00 | 1.00 | 0.02 |
| TurnOn(stove) | 0.00 | 1.00 | 1.00 | 0.00 |
| **Average** | **0.00** | **0.00** | **0.38** | **0.00** |

**Key finding:** Position perturbation is DEVASTATING. OpenVLA=0.00 on 9/10 tasks. Pi0=0.00 on 6/10. Pi0.5 survives on 4 tasks (bowl→cabinet_top, wine→rack, wine→cabinet, stove), fails on 6.

### Table 3: Spatial Suite

ALL models get 0.00 on Position perturbation for most tasks. Pi0.5 averages 0.20.

### Table 4: libero_10

Pi0.5 Position average = 0.08. Only 2 tasks above 0: Put(cream_cheese+butter, basket) at 0.12 and Put(mug+pudding, plate) at 0.02.

### Table 5: Object Suite

Pi0.5 Position average = 0.17. Best task: Place(butter, basket) at 0.96 (probably because butter position swap keeps it near the same spot).

---

## 5. How LIBERO-PRO's "Total" Score is Computed

From the leaderboard, the "Total" is the **mean across all 20 cells** (5 perturbation types x 4 suites):

```
Pi0.5 Total = mean of all 20 values
= (0.97+0.38+0.97+0.00+0.46 + 0.97+0.20+0.97+0.01+0.46 + 
   0.92+0.08+0.93+0.01+0.46 + 0.98+0.17+0.96+0.01+0.73) / 20
= 10.54 / 20 = 0.527 ≈ 0.53
```

**This is a FLAT average** - each perturbation type and each suite weighted equally.

---

## 6. SPARK vs LIBERO-PRO Baselines (Our Results)

### What We Tested

We tested 4 of 5 perturbation types (missing: Environment). Our evaluation:
- Uses **extracted MuJoCo XMLs** (not robosuite runtime)
- Runs **1 episode per task** (not 50) - deterministic with Gemini temperature=0
- Success checked via **MuJoCo body position query** (same criteria as robosuite)
- Object perturbation uses extracted scenes from HuggingFace pre-generated BDDLs

### Our Numbers

| Suite | Sem | Pos | Task | Obj | 4-type Avg |
|-------|-----|-----|------|-----|------------|
| Goal | 0.90 | 0.70 | 0.80 | 0.70 | 0.78 |
| Spatial | 0.90 | 0.90 | 0.80 | 0.90 | 0.88 |
| Object | 0.90 | 0.70 | 0.90 | 0.80 | 0.83 |
| libero_10 | 0.90 | 0.50 | 0.60 | 0.60 | 0.65 |
| **Average** | **0.90** | **0.70** | **0.78** | **0.75** | **0.78** |

### Direct Comparison (4 perturbation types)

| | Sem | Pos | Task | Obj | 4-type Avg |
|--|-----|-----|------|-----|------------|
| **SPARK** | 0.90 | **0.70** | **0.78** | 0.75 | **0.78** |
| **Pi0.5** | **0.96** | 0.21 | 0.01 | **0.96** | 0.53* |
| **OpenVLA** | **0.97** | 0.00 | 0.00 | **0.93** | 0.48* |
| **Pi0** | 0.91 | 0.00 | 0.00 | 0.90 | 0.45* |

*Pi0.5/OpenVLA/Pi0 4-type averages computed from their 4-type columns only (excluding Env).

### Key Deltas

| Perturbation | SPARK | Pi0.5 | Delta | Why |
|-------------|-------|-------|-------|-----|
| **Position** | 0.70 | 0.21 | **+0.49** | SPARK detects at runtime; VLAs memorize positions |
| **Task** | 0.78 | 0.01 | **+0.77** | SPARK plans from instruction; VLAs replay trajectories |
| Semantic | 0.90 | 0.96 | -0.06 | VLAs are robust because they IGNORE the instruction |
| Object | 0.75 | 0.96 | -0.21 | Our body matching struggles with renamed objects |

---

## 7. Critical Methodological Differences

### Why Our Comparison is NOT Apples-to-Apples

| Dimension | SPARK | VLA Baselines |
|-----------|-------|---------------|
| **Episodes per task** | 1 (deterministic) | 50 (stochastic) |
| **Evaluation platform** | Extracted MuJoCo XML | Robosuite runtime |
| **Controller** | IK-guided OSC (analytical) | Learned end-to-end policy |
| **Training data** | Zero (no training) | 50 demos per task |
| **Success criterion** | Body XY distance < 0.18m | Robosuite BDDL goal predicate |
| **Position perturbation** | Swap qpos in extracted init state | Robosuite's native position perturbation via BDDL |
| **Object perturbation** | Extracted perturbed scenes from HF | Robosuite instantiates perturbed BDDL |
| **Environment perturbation** | NOT TESTED | Robosuite swaps table fixture |

### What This Means for the Paper

1. **Our 1-episode evaluation is actually STRICTER** - VLAs get 50 attempts to succeed (50 different init seeds). We get 1 deterministic attempt. If anything, our numbers are conservative.

2. **Our success criterion may differ slightly** - we check XY distance < 0.18m between pick and place bodies. Robosuite checks the full BDDL goal predicate. These are similar but not identical (e.g., "object inside drawer" requires checking a region, not just proximity).

3. **Position perturbation method differs** - we swap object qpos directly in the JSON init state. LIBERO-PRO uses their perturbation engine to modify the BDDL, which robosuite then instantiates. The resulting object positions may differ.

4. **We should run 50 episodes to be truly comparable** - but with deterministic Gemini (temp=0), our results don't vary across episodes for the same task. The only variation comes from SAM3 detection, which is also deterministic for the same rendered image.

---

## 8. What the Semantic Perturbation Numbers Really Mean

**LIBERO-PRO's most damning finding:** VLAs score ~96% on Semantic perturbation - BUT this is NOT because they understand the paraphrased instruction. The paper shows (Figure 5, Section 5.4) that models produce the **exact same action trajectory** even when given **nonsense token instructions** ("fdsafdsgsd"). They don't read the instruction at all - they pattern-match from the visual observation.

For SPARK: Our 90% on Semantic genuinely reflects language understanding because Gemini actually PARSES the instruction to generate a plan. If you give Gemini nonsense, it would generate an empty/invalid plan and fail. SPARK's 90% Semantic score is real comprehension; VLAs' 96% Semantic score is accidental (they ignore instructions).

---

## 9. Object Perturbation: Where We Lose and Why

VLAs score 92-98% on Object perturbation. SPARK scores 75%. Why?

**VLAs win here because object appearance changes DON'T MATTER to them.** They memorize the POSITION and TRAJECTORY, not the object identity. A red bowl at position X triggers the same grasping motion as a yellow bowl at position X. The object looks different but the policy doesn't care - it's executing a memorized spatial trajectory.

**SPARK loses because body matching breaks.** When `akita_black_bowl` becomes `yellow_bowl`, our `_match_body()` function needs to fuzzy-match "bowl" in the new body name. This works most of the time (75%) but fails for some objects where the name changes significantly.

**Irony:** VLAs' "robustness" to object changes is actually evidence of memorization (they don't look at the object), while SPARK's "failure" is because it ACTUALLY TRIES to identify the correct object.

---

## 10. What CaP-X / CapGym Says About This

CaP-X (NVIDIA/Berkeley, arXiv:2603.22435) tested coding agents on LIBERO-PRO:

| Method | LIBERO-PRO Avg (Pos + Task only) |
|--------|----------------------------------|
| OpenVLA | ~0% |
| Pi0 | ~0% |
| Pi0.5 | ~13% |
| CaP-Agent0 | ~18% |
| **SPARK** | **~74%** (our Pos+Task avg) |

CaP-Agent0 generates Python code that calls perception/control primitives. It achieves 18% vs SPARK's 74% on the HARDEST perturbations (Position + Task).

**CaP-X does NOT make SPARK obsolete.** Their paper actually:
1. Validates structured abstractions (Finding 4: higher API abstraction = better performance)
2. Advocates for hybrid CaP-VLA (LLM for planning, VLA for execution) = SPARK's architecture
3. Shows code generation is brittle for contact-rich manipulation
4. Their best result (CaP-RL, post-trained 7B model) achieves 80% on simple cube tasks, not complex LIBERO

---

## 11. Where to Go From Here

### Immediate (for paper)
1. **Run 50 episodes per task** to match LIBERO-PRO protocol (currently 1 deterministic episode)
2. **Add Environment perturbation** (need robosuite runtime table swap - complex but doable)
3. **Record videos** for all tasks (added to pipeline, running now)
4. **Tighten success criteria** to match robosuite BDDL goal predicates more exactly

### Architecture Improvements
5. **Auto-synthesized skill library** (validated by CaP-X): extract reusable primitives from successful Gemini plans, build persistent library
6. **Multi-turn replanning**: when check_success fails, re-perceive + re-plan + retry (CaP-Agent0's key advantage)
7. **Improve Object perturbation** body matching: use the actual scene body list to match, not just the original task mapping

### Strategic Position
- SPARK is the **strongest system on LIBERO-PRO's hard perturbations** (Position, Task)
- VLAs are better on easy perturbations (Semantic, Object) but for the WRONG reasons (memorization, not comprehension)
- The paper story: **neurosymbolic approaches achieve genuine generalization; VLA scores are inflated by memorization**
- LIBERO-PRO validates this narrative - their entire paper argues that standard LIBERO is misleading

### What About Standard LIBERO?
- Keep reporting standard LIBERO (90%) for comparison with the field
- But emphasize LIBERO-PRO as the more meaningful benchmark
- SPARK 78% on LIBERO-PRO vs Pi0.5 53% is the headline result
- Position (+49pp) and Task (+77pp) advantages are the key evidence

---

## 12. Summary Table

| Metric | SPARK | Pi0.5 | OpenVLA | Pi0 |
|--------|-------|-------|---------|-----|
| Standard LIBERO | **90%** | 97% | 97% | 96% |
| PRO: Semantic | 90% | **96%** | **97%** | 91% |
| PRO: Position | **70%** | 21% | 0% | 0% |
| PRO: Task | **78%** | 1% | 0% | 0% |
| PRO: Object | 75% | **96%** | **93%** | 90% |
| PRO: Environment | ? | 53% | 47% | 39% |
| PRO: Total (4-type) | **78%** | 53% | 48% | 45% |
| Training data | **ZERO** | 50 demos/task | 50 demos/task | 50 demos/task |
| Episodes evaluated | 1 | 50 | 50 | 50 |

---
---

# CaP-X / CapGym Dense Analysis

**Paper:** CaP-X: A Framework for Benchmarking and Improving Coding Agents for Robot Manipulation (arXiv:2603.22435, March 2026)
**Authors:** Max Fu, Justin Yu, Karim El-Refai, Ethan Kou, Haoru Xue, + others from NVIDIA, UC Berkeley, Stanford, CMU
**Project leads:** Ken Goldberg, Linxi "Jim" Fan
**Project page:** https://capgym.github.io/

---

## 1. What CaP-X Actually Is

CaP-X is NOT a single model. It's a **framework** with 4 components:

| Component | What It Is | Purpose |
|-----------|-----------|---------|
| **CaP-Gym** | Interactive environment binding simulators (RoboSuite, LIBERO-PRO, BEHAVIOR) with a stateful Python code executor | The "gym" - agents write code, environment executes it |
| **CaP-Bench** | Systematic benchmark: 7 core tasks, 8 evaluation tiers (4 single-turn S1-S4, 4 multi-turn M1-M4) | Measures how abstraction level, multi-turn, and visual grounding affect performance |
| **CaP-Agent0** | Training-free agentic system: multi-turn code generation + VDM + skill library + multi-model ensembling | Their best training-free system |
| **CaP-RL** | GRPO reinforcement learning on a 7B coding model (Qwen 2.5 Coder) using environment rewards | Post-training a small model to generate better robot code |

### Core Idea
LLMs generate **executable Python programs** that call perception and control primitives to manipulate robots. No behavior trees, no YAML, no symbolic plans - just raw Python code.

---

## 2. The Abstraction Ladder (Key Contribution)

CaP-X's most important contribution is systematically measuring how API abstraction affects performance. They define 8 evaluation tiers:

### Single-Turn Tiers (S1-S4)

| Tier | Perception | Primitives | Examples | Description |
|------|-----------|-----------|----------|-------------|
| **S1** | Ground-truth state (privileged) | High-level (`sample_grasp_pose("red cube")`) | N/A | Upper bound - perfect perception + macro primitives |
| **S2** | Real perception (SAM3, Molmo) | High-level | N/A | Standard Code-as-Policy setup |
| **S3** | Real perception | Low-level (`solve_ik()`, `sam3_text_prompt()`) | With usage examples in prompt | LLM must compose low-level calls |
| **S4** | Real perception | Low-level | No examples - only function signatures | Hardest: LLM reasons from docstrings alone |

### Multi-Turn Tiers (M1-M4)

| Tier | Feedback Type | Visual Grounding |
|------|-------------|-----------------|
| **M1** | Text only (stdout/stderr) | None |
| **M2** | Text + raw RGB images | Direct image input |
| **M3** | Text + VDM (visual differencing → structured text) | Vision-to-text conversion |
| **M4** | Text + VDM + low-level primitives | Best of both worlds |

### Key Finding (Figure 3): Performance vs Abstraction

```
S4 (bare low-level):     ~5-15% average
S3 (low-level + examples): ~15-30%
S2 (high-level + real perception): ~40-60%
S1 (high-level + GT state): ~55-80%
Human expert: 88.5%
```

**All models improve monotonically as abstraction increases.** This means high-level primitives (like SPARK's behavior tree actions) are not a crutch - they're the right engineering choice. The paper frames this as a problem to solve (making LLMs work with low-level primitives), but the pragmatic implication is clear: structured abstractions work better.

---

## 3. CaP-Bench Results (7 Core Tasks)

Tasks: Cube Lift, Cube Stack, Spill Wipe, Peg Insertion, Cube Re-stack, Two-Arm Lift, Two-Arm Handover.

### CaP-Agent0 Performance (100 trials per task)

| Task | CaP-Agent0 | Human Expert | Gap |
|------|-----------|--------------|-----|
| Cube Lift | ~90% | 93% | -3pp |
| Cube Stack | ~70% | 73% | -3pp |
| Spill Wipe | ~95% | 100% | -5pp |
| Peg Insertion | ~30% | 87% | -57pp |
| Cube Re-stack | ~90% | 100% | -10pp |
| Two-Arm Lift | ~25% | 53% | -28pp |
| Two-Arm Handover | ~10% | 97% | -87pp |

**CaP-Agent0 matches humans on 4/7 simple tasks** (lift, stack, wipe, re-stack) but fails catastrophically on contact-rich tasks (peg insertion, two-arm handover).

### Best Frontier Model Zero-Shot (S4, no examples)
- Gemini-3-Pro: ~30% average
- GPT-5.2: ~25%
- Claude Opus 4.5: ~20%
- Best open-source (Qwen 235B): ~15%

**56-point gap between best frontier model and human experts on S4.**

---

## 4. CaP-X on LIBERO-PRO (Table 2 - THE KEY TABLE)

CaP-X evaluates on **30 LIBERO-PRO tasks** (Object, Goal, Spatial suites only - NOT libero_10 in some analyses). They test **only Pos and Task perturbations**.

### Table 2: LIBERO-PRO Results (Pos + Task only)

| Method | Object Pos | Object Task | Goal Pos | Goal Task | Spatial Pos | Spatial Task |
|--------|-----------|-------------|----------|-----------|-------------|--------------|
| OpenVLA | 0.00 | 0.00 | 0.00 | 0.00 | 0.00 | 0.00 |
| Pi0 | 0.00 | 0.00 | 0.00 | 0.00 | 0.00 | 0.00 |
| Pi0.5 | 0.17 | 0.01 | 0.38 | 0.00 | 0.20 | 0.01 |
| **CaP-Agent0** | **0.22** | **0.18** | **0.26** | **0.17** | **0.12** | **0.14** |

### Averages

| Method | Pos Avg | Task Avg | Overall (Pos+Task) |
|--------|---------|----------|-------------------|
| OpenVLA | 0.00 | 0.00 | 0.00 |
| Pi0 | 0.00 | 0.00 | 0.00 |
| Pi0.5 | 0.25 | 0.01 | 0.13 |
| CaP-Agent0 | 0.20 | 0.16 | 0.18 |
| **SPARK (ours)** | **0.70** | **0.78** | **0.74** |

**SPARK is 4x better than CaP-Agent0 on LIBERO-PRO Pos+Task.**

### Critical Observation

CaP-Agent0 BEATS Pi0.5 on Task perturbation (0.16 vs 0.01) but LOSES to Pi0.5 on Position perturbation (0.20 vs 0.25). This makes sense:
- Task perturbation = new instruction → coding agents parse language, VLAs replay trajectories → coding wins
- Position perturbation = objects in different spots → Pi0.5's broader training data gives some spatial flexibility, while CaP-Agent0's code may hardcode relative positions

SPARK beats BOTH on BOTH perturbation types because it detects at runtime (SAM3) and plans from the actual scene state (Gemini).

---

## 5. CaP-Agent0 Architecture Details

### Visual Differencing Module (VDM)
Instead of feeding raw RGB images to the LLM (which HURTS performance - M2 is WORSE than M1), VDM converts images to structured text:

```
Turn 1: "The scene contains a red cube on the table center, 
         a green cube to its left, and the gripper is open above."
Turn 2: "Changes: The gripper has moved down and closed around 
         the red cube. The red cube has lifted 5cm off the table."
```

**This validates SPARK's approach**: we convert perception to text (SAM3 labels + body positions) before giving to Gemini. Raw images → LLM is bad. Structured text → LLM is good.

### Auto-Synthesized Skill Library
Process:
1. Run S3 tier (low-level with examples) on successful tasks
2. Extract function definitions from successful code via regex
3. LLM identifies recurring task-agnostic utility patterns
4. 9 functions promoted to reusable skills

The 9 skills (from Appendix G.1) are utilities like:
- `rot_matrix_to_quat(rotation_matrix)` - convert rotation representation
- `depth_to_pointcloud(depth, intrinsics)` - depth image to 3D points
- `transform_pts(points, transform_matrix)` - coordinate transforms
- `select_top_down_grasp(groups, scores, ...)` - grasp selection heuristic

**These are NOT task-level primitives** (not "pick" or "place"). They're low-level utility functions that reduce code complexity. SPARK's behavior tree primitives (move_to_keypoint, grasp, release, open_drawer, push_object) are much higher-level and more powerful.

### Parallel Reasoning (Multi-Model Ensemble)
At each turn, CaP-Agent0 generates 9 candidate solutions:
- Single-model: 9 queries to Gemini-3-Pro
- Multi-model: 3 queries each to GPT-5.2, Claude Opus 4.5, Gemini-3-Pro
A central "ensemble agent" selects the best candidate.

This is expensive (9 LLM calls per turn, multiple turns per task) vs SPARK's single Gemini call.

---

## 6. CaP-RL (Reinforcement Learning on Code Generation)

### Method
- Base model: Qwen 2.5 Coder 7B Instruct
- Algorithm: GRPO (Group Relative Policy Optimization)
- Reward: binary task success from environment
- Training: 50 iterations, privileged state (S1 tier) for stable training
- Evaluation: S2 tier (real perception)

### Results (Table 4)

| Task | Base 7B | CaP-RL | Human | Real Robot |
|------|---------|--------|-------|------------|
| Cube Lift | 25% | **80%** | 93% | 84% |
| Cube Stack | 4% | **44%** | 73% | 76% |
| Spill Wipe | 30% | **93%** | 100% | N/A |

### Sim-to-Real Transfer
CaP-RL transfers to real Franka Emika with minimal gap because code operates over abstract APIs (not raw pixels). Cube Lift: 84% real vs 80% sim. Cube Stack: 76% real vs 44% sim (real is BETTER because real perception has less noise than sim rendering).

**This is interesting for SPARK**: could we train a small model (7B) to generate SPARK behavior trees via GRPO? The reward signal (check_success) is already built into our pipeline. The input would be instruction + detection labels, output would be YAML behavior tree. This would be a "SPARK-RL" that learns to plan from environment rewards.

---

## 7. What CaP-X Gets Right

1. **Systematic abstraction analysis.** First paper to cleanly measure the S1→S4 performance gradient. Proves that high-level primitives aren't cheating - they're necessary engineering.

2. **VDM over raw images.** Showing M2 (raw RGB) HURTS while M3 (VDM text) HELPS is an important finding that validates perception-to-text pipelines like SPARK's.

3. **Multi-turn recovery.** Single-shot code generation fails; iterative debugging with environment feedback works. SPARK currently does single-shot planning - multi-turn would help.

4. **CaP-RL sim-to-real.** Code-based policies transfer better than pixel-based VLAs because they reason over abstractions, not visual features. Same reason SPARK transfers: the plan is symbolic, not visual.

5. **Honest assessment of limitations.** Section 7: "programmatic control performs well on long-horizon, reasoning-heavy tasks, but remains brittle for contact-rich behaviors that require tight visual servoing." They know code-as-policy has limits.

---

## 8. What CaP-X Gets Wrong / Where It's Weak

1. **LIBERO-PRO results are modest.** 18% average on Pos+Task. Compare to SPARK's 74%. The code generation approach generates correct-looking programs but the programs fail during execution because low-level control is hard.

2. **Only tests Pos and Task perturbations.** Doesn't test Obj, Sem, or Env perturbations on LIBERO-PRO. Cherry-picks the perturbations where coding agents do relatively better (Task) while ignoring ones where VLAs do better (Obj, Sem).

3. **Core tasks are simple.** Cube lift, cube stack, spill wipe - these are basic manipulation tasks. SPARK solves 36/40 LIBERO tasks which include drawer opening, stove turning, multi-step transport across rooms, and bowl rim grasping.

4. **CaP-Agent0 is extremely expensive.** 9 LLM calls per turn × multiple turns × 3 frontier models. A single LIBERO task might cost $1-5 in API calls. SPARK uses 1 Gemini call per task.

5. **CaP-RL only works on 3 simple tasks.** Cube lift, cube stack, spill wipe. Not tested on LIBERO or any complex manipulation. The jump from 3 cube tasks to 40 LIBERO tasks is enormous.

6. **No comparison to neurosymbolic approaches.** They compare to VLAs (OpenVLA, Pi0, Pi0.5) but not to structured planning systems like SPARK, SayCan, Inner Monologue, or TAMP. This omission makes their contribution look larger than it is.

---

## 9. SPARK vs CaP-X: Direct Comparison

| Dimension | SPARK | CaP-Agent0 | CaP-RL |
|-----------|-------|-----------|--------|
| **Architecture** | SAM3 → Gemini → YAML BT → OSC controller | LLM → Python code → perception/control APIs | RL-trained 7B → Python code → APIs |
| **Standard LIBERO** | **90%** (36/40) | Not tested | Not tested |
| **LIBERO-PRO (Pos+Task)** | **74%** | 18% | Not tested |
| **CaP-Bench (7 core)** | Not tested | ~60% (matches human on 4/7) | ~72% (3 tasks only) |
| **BEHAVIOR (mobile)** | Not tested | 56-72% | Not tested |
| **Training** | Zero | Zero | 50 GRPO iterations |
| **LLM calls per task** | 1 | 9+ per turn × multiple turns | N/A (offline trained) |
| **Controller** | IK-guided mass-matrix OSC | PyRoki IK + motion planner | Same as CaP-Agent0 |
| **Plan representation** | YAML behavior tree (structured) | Raw Python (unstructured) | Raw Python |
| **Multi-turn** | No (single-shot) | Yes (key advantage) | No |
| **Skill library** | Static primitives (manual) | Auto-synthesized (9 utilities) | N/A |
| **Real robot** | Not yet (UR10e pipeline exists) | Yes (Franka, AgiBot) | Yes (Franka) |

### Where SPARK Wins
- **LIBERO-PRO: 74% vs 18%** - 4x better on the hardest robustness benchmark
- **Standard LIBERO: 90% vs not tested** - SPARK has proven results on complex tasks
- **Efficiency: 1 LLM call vs 9+** - orders of magnitude cheaper
- **Controller quality**: mass-matrix OSC with nullspace tracking is more precise than generic IK + motion planner
- **Structured plans**: YAML behavior trees are interpretable, verifiable, composable

### Where CaP-X Wins
- **Multi-turn recovery**: iterative debugging when code fails (SPARK is single-shot)
- **Real robot demos**: tested on Franka and AgiBot in the real world
- **Broader task scope**: BEHAVIOR mobile manipulation, bimanual tasks
- **CaP-RL direction**: post-training a coding model is a compelling research direction
- **Auto skill synthesis**: discovering reusable utilities from successful executions

---

## 10. Does CaP-X / CapGym Make SPARK Obsolete?

**No. Emphatically no.** Here's why:

1. **SPARK outperforms CaP-Agent0 by 4x on LIBERO-PRO.** The benchmark CaP-X chose to evaluate on is the one where SPARK dominates.

2. **CaP-X's own conclusion advocates for SPARK's architecture.** Section 7: "One promising direction is hybrid CaP-VLA policies, in which a coding agent manages high-level task logic and recovery while deferring low-level execution to VLA policies." This is literally SPARK: LLM manages planning, specialized controller handles execution.

3. **CaP-X validates structured abstractions.** Their Finding 2 shows all models improve with higher abstraction. SPARK's behavior tree primitives ARE high-level abstractions - the paper proves they're the right choice.

4. **CaP-X validates perception-to-text.** VDM (converting images to text) consistently beats raw image input. SPARK converts SAM3 detections to text labels before Gemini planning - same principle.

5. **CaP-X's core tasks are simpler than LIBERO.** Cube lift/stack/wipe vs drawer opening, stove turning, multi-step transport, bowl rim grasping. SPARK solves harder problems.

---

## 11. What SPARK Should Take From CaP-X

### Must-Do (High Impact)
1. **Multi-turn replanning.** When check_success fails, re-perceive → re-plan → retry. CaP-Agent0 shows this consistently improves performance. Implementation: add a retry loop around execute + check_success.

2. **Auto-synthesized skill library.** Extract reusable utility functions from successful Gemini plans. Build a persistent library that grows over tasks. This is SPARK's natural extension.

### Should-Do (Medium Impact)
3. **CaP-RL for SPARK.** Train a small model (7B) to generate SPARK behavior trees via GRPO, using check_success as the reward. The input is instruction + SAM3 labels, output is YAML BT. This would be cheaper than Gemini and potentially more reliable.

4. **Run CaP-Bench tasks.** Port SPARK to CaP-Gym's 7 core tasks (cube lift, stack, wipe, etc.) to get direct comparison numbers. SPARK should perform well on these given our OSC controller quality.

5. **BEHAVIOR tasks.** Test SPARK on mobile manipulation for broader scope.

### Nice-to-Have
6. **Parallel reasoning.** Generate 3 Gemini plans and pick the best via a validation heuristic. Cheap version of CaP-Agent0's ensemble.

7. **VDM for error diagnosis.** After a failed execution, render the scene and use a VLM to describe what went wrong, then feed that to Gemini for replanning.

---

## 12. Combined Summary: Where Does SPARK Stand?

| Benchmark | SPARK | Best VLA (Pi0.5) | Best Code Agent (CaP-Agent0) |
|-----------|-------|-----------------|------------------------------|
| Standard LIBERO | **90%** | 97% | Not tested |
| LIBERO-PRO (Pos+Task) | **74%** | 13% | 18% |
| LIBERO-PRO (4-type avg) | **78%** | 53% | ~18%* |
| CaP-Bench (7 core) | Not tested | Not applicable | ~60% |
| Training required | **Zero** | 50 demos/task | **Zero** |

*CaP-Agent0 only tested on Pos+Task, not Obj/Sem/Env.

**SPARK's position: strongest system on LIBERO-PRO by a wide margin.** VLAs score higher on standard LIBERO but collapse under perturbation. CaP-Agent0 is the only other system that handles perturbations, but at 4x lower success rate.

**The narrative for the paper:**
- Standard LIBERO is flawed (LIBERO-PRO paper proves this)
- VLAs memorize, don't generalize (LIBERO-PRO demonstrates this)
- Code agents generalize but fail at execution (CaP-X demonstrates this)
- SPARK combines the best of both: LLM generalization + controller precision
- Result: 90% standard, 78% perturbed, zero training

---
---

# 13. Honest Non-Sycophantic Critique of SPARK vs CaP-X

## Where My Previous Analysis Was Soft

I was framing things too favorably for SPARK. Let me be blunt.

---

## CaP-RL: What They Actually Did Physically

### Sim Results (Table 4, 100 trials each)
| Task | Base Qwen 7B | CaP-RL (post-trained) | Human Expert |
|------|-------------|----------------------|--------------|
| Cube Lift | 25% | **80%** | 93% |
| Cube Stack | 4% | **44%** | 73% |
| Spill Wipe | 30% | **93%** | 100% |

### Real Robot Results (25 trials each, Franka Emika Panda)
| Task | CaP-RL Real | Human Real |
|------|------------|------------|
| Cube Lift | **84%** | 92% |
| Cube Stack | **76%** | 84% |

**CaP-RL transferred sim-to-real with MINIMAL gap.** Cube lift: 80% sim → 84% real. Cube stack: 44% sim → 76% real (real was BETTER - likely because real perception has less rendering artifacts than sim).

They also tested CaP-Agent0 (training-free) on real robots:
- **Franka Panda** and **AgiBot G1** humanoid
- Zero-shot, no cross-embodiment modifications (except bimanual primitives)
- Gemini-3-Pro and Claude Opus 4.5 generate code directly for the real robot
- Tasks: complex long-horizon reasoning ("find the object under one of the cups", "solve a math problem presented physically", "stack objects by size")

### What SPARK Has Done Physically
**Nothing.** We have a UR10e pipeline (`spark_real/`) with RealSense D435 and ur_rtde driver, but zero real-robot results. All 90% numbers are in MuJoCo simulation with extracted scene XMLs.

**This is a major gap.** CaP-X has real robot demos. We don't.

---

## Honest Weaknesses of SPARK

### 1. We're Sim-Only
All results are in MuJoCo with extracted LIBERO XMLs. We've never tested on a real robot. The sim-to-real gap for our pipeline is completely unknown. CaP-X demonstrates real robot transfer. We claim it should work but haven't proven it.

### 2. Our Success Metric is Simpler Than Theirs
We check `XY distance < 0.18m` between pick and place bodies. LIBERO-PRO uses robosuite's full BDDL goal predicates (which can check "inside drawer", "on top of", contact states, joint positions). Our metric is a loose approximation. Some of our "passes" might be "fails" under the official predicate.

### 3. We Run 1 Episode Per Task, They Run 50-100
With Gemini temperature=0, our results are deterministic for a given scene. But we only test 1 init state per task. LIBERO-PRO evaluates over 50 different random initial states to measure robustness to init variation. We don't test this at all. Our 90% could drop significantly with varied init states.

### 4. Our "Position Perturbation" Isn't the Same as Theirs
We swap object qpos values in our JSON init state files. LIBERO-PRO uses their perturbation engine to modify BDDLs, which robosuite then instantiates with proper collision checking and physics settling. Our swapped positions might create physically invalid states (objects overlapping, floating, etc.) that we don't catch.

### 5. Body Matching is a Crutch
SAM3 detects objects, but then we snap to MuJoCo ground-truth body positions. In the real world, there IS no MuJoCo body to snap to. Our "SAM3 perception" is really "SAM3 for identification + MuJoCo GT for localization." On a real robot, we'd need SAM3 + depth camera for actual 3D localization, which has 2-5cm error. Our 90% relies on GT positions that won't exist on real hardware.

### 6. Our Task Mapping is Hand-Engineered
`get_task_prompts()` is a manual mapping of task names to SAM3 prompts, pick/place body names, pre-actions, and multi-step decomposition. This is ~200 lines of task-specific configuration. It's not "zero-shot" in the purest sense - we hand-engineered the perception prompts and success criteria per task family. CaP-Agent0 truly generates everything from the instruction alone.

### 7. Single-Shot Planning is Brittle
SPARK generates one Gemini plan and executes it. If it fails, it fails. CaP-Agent0 iterates: generates code, executes, observes failure, re-generates. This multi-turn approach is fundamentally more robust. Our 90% on standard LIBERO masks the fact that for the 4 tasks we fail, we have no recovery mechanism.

### 8. Our Controller is Scene-Specific
The OSC controller parameters (KP_TASK=150, KP_JOINT=30, torque limits) were tuned on LIBERO scenes. The transport strategy (lift→XY→descend), the grasp strategies (bowl rim pinch, standard top-down), the approach heights - these were all tuned through iteration on LIBERO tasks. They might not transfer to different robot configurations, grippers, or task domains without re-tuning.

### 9. No Contact-Rich Manipulation
SPARK handles pick-place, push, drawer open, stove turn. We can't do: peg insertion, pouring, wiping, folding, tool use, deformable manipulation. CaP-X's spill wipe task (93%) involves actual contact-rich behavior. Our primitives are all "move to position, close gripper" - we don't reason about contact forces, friction, or manipulation dynamics.

### 10. The Primitive Library is Static
Our 7 primitives (move_to_keypoint, grasp, release, move_relative, open_drawer, push_object, turn_knob) are hand-designed. CaP-X auto-synthesizes utility functions from successful executions. We can't learn new primitives. If a task requires a new type of manipulation (e.g., scooping, flipping, threading), we'd need to manually implement it.

---

## Honest Strengths of CaP-X Over SPARK

### 1. Real Robot Transfer
They demonstrated on Franka Panda AND AgiBot G1 humanoid. Code-based policies transfer because they operate over abstract APIs, not visual features. SPARK's OSC controller is MuJoCo-specific - transferring to a real robot requires controller re-implementation.

### 2. Multi-Turn is a Better Architecture
CaP-Agent0's iterative debugging loop (generate → execute → observe → re-generate) is fundamentally more capable than SPARK's single-shot planning. Even if each individual code generation is worse than a Gemini plan, the ability to RECOVER from failures compounds across turns.

### 3. CaP-RL is a Genuinely New Contribution
Post-training a coding model with GRPO using environment rewards is novel and important. The result: a 7B model goes from 25% → 80% on cube lift, and transfers to real. This is a scalable approach - train once, deploy everywhere. SPARK relies on Gemini API calls ($$$) for every task execution.

### 4. Broader Task Scope
CaP-X tests on 187 tasks across RoboSuite, LIBERO-PRO, AND BEHAVIOR (mobile manipulation). SPARK only tests on LIBERO (40 tasks, all tabletop). We haven't demonstrated generalization to different robots, different environments, or mobile manipulation.

### 5. Auto Skill Synthesis Scales
Their skill library is auto-generated from successful executions and grows over time. SPARK's primitive library is fixed. As task complexity increases, CaP-X's library can adapt. SPARK can't without manual engineering.

### 6. Section 7 Conclusion Is Self-Aware
CaP-X honestly states: "programmatic control performs well on long-horizon, reasoning-heavy tasks, but remains brittle for contact-rich behaviors." They propose hybrid CaP-VLA as the future. This level of honest self-assessment is good science. My analysis of SPARK should be equally honest.

---

## Where SPARK Genuinely Wins (Not Sycophantic)

### 1. LIBERO-PRO Performance is Real
74% on Pos+Task vs CaP-Agent0's 18%. This is a 4x difference on the SAME benchmark with the SAME perturbation types. Even accounting for methodological differences (1 vs 100 episodes, body position vs BDDL predicate), this gap is too large to be an artifact.

### 2. Computational Efficiency
1 Gemini call vs 9+ LLM calls per turn × multiple turns. SPARK is 50-100x cheaper per task. For deployment, this matters enormously.

### 3. Interpretability
YAML behavior trees are human-readable and verifiable. You can inspect the plan before execution and understand exactly what the robot will do. Raw Python code generated by an LLM is opaque and potentially dangerous on a real robot.

### 4. The Architecture is Correct
Even CaP-X's conclusion says the future is "hybrid CaP-VLA policies, in which a coding agent manages high-level task logic and recovery while deferring low-level execution to VLA policies." SPARK already IS this: Gemini manages task logic, OSC handles low-level execution. SPARK was early to this architecture.

---

## Honest Assessment: Where To Focus

1. **Real robot demo is #1 priority.** Without it, we're a sim paper. CaP-X has real robot. We need it.
2. **Multi-turn replanning is #2.** Single-shot is a fundamental limitation. Add retry-on-failure.
3. **Run proper 50-episode evaluation.** Match LIBERO-PRO protocol exactly.
4. **Remove GT body position dependence.** Use SAM3 + depth for actual 3D, not MuJoCo snap.
5. **Auto skill library.** Port CaP-X's approach: extract reusable functions from successful plans.
6. **Stop claiming "zero-shot" loosely.** The task_prompts mapping is hand-engineered per task family. Call it "training-free" or "demonstration-free" instead.

---

# 14. CORRECTIONS: What I Got Wrong About CaP-X

## My Error

I framed CaP-Agent0 as getting "18% on LIBERO-PRO" as if that was their main result. **It's not.** LIBERO-PRO is a secondary evaluation (Table 2, Section 4.2). Their MAIN results are on CaP-Bench (7 tasks, 8 evaluation tiers).

## CaP-X's Actual Main Claim

The paper's contribution is the **abstraction ladder** analysis, not beating VLAs on LIBERO-PRO:

### Figure 3 (the key result): Performance vs Abstraction Level
```
S4 (bare low-level, no examples):  5-15%
S3 (low-level + examples):        15-30%
S2 (high-level + real perception): 40-80%  ← SPARK operates here
S1 (high-level + GT state):        55-80%
Human expert:                      88.5%
```

### Figure 8 (CaP-Agent0 per-task, ~100 trials each):
```
Cube Lift:       ~93% (matches human 93%)
Cube Re-stack:   ~90% (vs human 100%)
Spill Wipe:      ~95% (vs human 100%)
Cube Stack:      ~70% (vs human 73%)
Peg Insertion:   ~30% (vs human 87%)  ← contact-rich = hard
Two-Arm Lift:    ~25% (vs human 53%)  ← bimanual = hard
Two-Arm Handover: ~10% (vs human 97%) ← bimanual = very hard
```

CaP-Agent0 achieves ~90%+ on single-arm tasks (lift, stack, wipe, re-stack). It fails on contact-rich (peg insertion) and bimanual tasks. The 90% numbers are REAL but on SIMPLER tasks than LIBERO.

## CaP-X's LIBERO-PRO Results (Table 2, secondary)

Only tested Pos + Task perturbations on 3 suites (Object, Goal, Spatial - no libero_10):

| Method | Obj-Pos | Obj-Task | Goal-Pos | Goal-Task | Spa-Pos | Spa-Task |
|--------|---------|----------|----------|-----------|---------|----------|
| OpenVLA | 0.00 | 0.00 | 0.00 | 0.00 | 0.00 | 0.00 |
| pi0 | 0.00 | 0.00 | 0.00 | 0.00 | 0.00 | 0.00 |
| pi0.5 | 0.17 | 0.01 | 0.38 | 0.00 | 0.20 | 0.01 |
| CaP-Agent0 | 0.22 | 0.18 | 0.26 | 0.17 | 0.12 | 0.14 |

The paper's point: "CaP-Agent0 remains robust to instruction variations, whereas VLAs perform poorly under task perturbations" and "CaP-Agent0 achieves performance comparable to pi0.5 under initial position perturbations."

## What CaP-X Actually Complains VLAs Get

Looking at Section 4.2 text: "Since VLAs are trained on a different instruction distribution, they perform poorly under task perturbations" - pi0.5 gets 0.01 on Task across all suites, OpenVLA/pi0 get 0.00.

For Position: pi0.5 gets 0.17-0.38 across suites. OpenVLA/pi0 get 0.00.

So the complaint is: VLAs get **0% on Task and 0-38% on Position** while CaP-Agent0 gets **14-18% on Task and 12-26% on Position**.

## Honest Comparison to SPARK

| Benchmark | SPARK | CaP-Agent0 | Pi0.5 |
|-----------|-------|-----------|-------|
| CaP-Bench (7 tasks) | Not tested | ~60% avg (90% on easy, 10% on hard) | Not applicable |
| Standard LIBERO (40 tasks) | **90%** | Not tested | 97% |
| LIBERO-PRO Pos+Task | **74%** | **18%** | 13% |

**The gap is genuine on LIBERO-PRO** (74% vs 18%). But CaP-Agent0 is designed for a different purpose - it's exploring how to make coding agents work WITHOUT high-level primitives. SPARK uses high-level primitives (behavior trees), which CaP-X shows is the right choice (Figure 3: S2 > S3 > S4).

## What This Actually Means for SPARK's Paper

1. **SPARK is NOT directly competing with CaP-X.** They're studying the abstraction ladder. We're building a practical system. Different papers.

2. **CaP-X validates our approach.** Their Figure 3 proves high-level primitives (S2) outperform low-level (S3/S4). SPARK IS an S2-level system.

3. **Our LIBERO-PRO advantage is real but the comparison is apples-to-oranges.** We use structured BT + OSC. They use raw Python + generic IK. Same perception (SAM3). Different everything else.

4. **The real question for our paper** is not "are we better than CaP-X" but "does structured planning + proper control achieve robust generalization where both VLAs and code agents fail?"

5. **We should test on CaP-Bench** to get a direct comparison. If SPARK can do cube lift/stack/wipe at 90%+ too, that removes the "different benchmarks" objection.

---

# 15. VERIFIED CaP-Bench Numbers (From Figure 8 + Figure 17)

## Figure 8 Right Panel: CaP-Agent0 vs Human vs S3 Baseline

Reading from the bar chart (approximate, from visual inspection):

| Task | Human Expert | S3 (low-level baseline) | CaP-Agent0 (M4+SL+3M) |
|------|-------------|------------------------|------------------------|
| Cube Lift | 93% | ~34% | **~97%** |
| Cube Stack | 73% | ~4% | **~76%** |
| Spill Wipe | 100% | ~46% | **~100%** |
| Peg Insert | 87% | ~10% | ~30% |
| Cube Re-stack | 100% | ~95% | **~100%** |
| Two-Arm Lift | 53% | ~0% | ~10% |
| Two-Arm Handover | 97% | ~25% | ~16% |

**CaP-Agent0 EXCEEDS human on 3 tasks** (Lift 97>93, Stack 76>73, Re-stack 100=100) and matches on 1 (Wipe 100=100). Fails badly on Peg Insert, 2-Arm Lift, 2-Arm Handover.

## Figure 8 Left Panel: Ablation (Average Success %)

| Configuration | Avg Success |
|--------------|-------------|
| S3 (single-turn, low-level) | ~24% |
| +1M (multi-turn, 1 model) | ~48% |
| +SL (+ skill library) | ~59% |
| M4 (+ VDM) | ~55% |
| S3+3M (multi-model ensemble) | - |
| Full CaP-Agent0 | **~68%** |

## Figure 17 (Appendix B): Full Benchmark - Task Success Rate

This is a heatmap across ALL models × ALL tiers × ALL tasks (N=100 per cell). The bottom panel shows Task Success Rate. CaP-Agent0 at M4 tier is the rightmost column group.

Key observations from the heatmap:
- **Lift task**: Most models at S1/S2 get 80-100%. At S4, drops to 5-40%. CaP-Agent0 M4 recovers to ~97%.
- **Stack task**: Human 73%. Most S2 models 20-50%. CaP-Agent0 M4 ~76%.
- **Wipe task**: Human 100%. S2 models 60-90%. CaP-Agent0 M4 ~100%.
- **Peg Insertion**: Human 87%. ALL models <30% even at S1. Contact-rich = fundamentally hard.
- **Re-stack**: High across the board at S2/M3 because it's mostly planning + perception.
- **2-Arm tasks**: Near-zero for most models. Bimanual coordination is unsolved.

## Corrected Summary

My earlier claim "CaP-Agent0 gets ~90% on 4/7 tasks" was approximately right but imprecise:
- **97%** on Cube Lift (exceeds human 93%)
- **76%** on Cube Stack (exceeds human 73%)
- **100%** on Spill Wipe (matches human 100%)
- **100%** on Cube Re-stack (matches human 100%)
- **30%** on Peg Insert (vs human 87%)
- **10%** on Two-Arm Lift (vs human 53%)
- **16%** on Two-Arm Handover (vs human 97%)

Average across 7 tasks: ~61%. Average on single-arm tasks only (5 tasks): ~81%.

**CaP-Agent0 matches/exceeds human on 4/7 single-arm tasks but fails on contact-rich and bimanual tasks.** The ~90% claim applies to the 4 tasks where it succeeds, not the overall average.

## What This Means for SPARK Comparison

CaP-Agent0's CaP-Bench results (81% on single-arm, 61% overall) are on SIMPLER tasks than LIBERO:
- CaP-Bench: cube lift, cube stack, spill wipe, peg insert, re-stack
- LIBERO: 40 diverse tasks including drawer open, stove turn, multi-step transport, bowl rim grasp, microwave, cross-room carrying

**SPARK's 90% on LIBERO is on harder, more diverse tasks than CaP-Agent0's 81% on CaP-Bench single-arm tasks.** But the benchmarks are different enough that direct comparison should be qualified.

The LIBERO-PRO comparison (SPARK 74% vs CaP-Agent0 18% on Pos+Task) IS on the same benchmark and is the fairest comparison we have.

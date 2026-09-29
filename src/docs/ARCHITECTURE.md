# SPARK Architecture

**Sequential Planning via Anchored Robotic Keypoints**

## Overview

SPARK is a zero-shot robotic manipulation framework that combines open-vocabulary perception, LLM-based task planning, and model-based control. Given a natural language instruction and an RGB-D scene, SPARK detects objects, generates an executable behavior tree, and carries out the task -- all without task-specific training or demonstrations.

Key results:
- LIBERO benchmark: 36/40 = 90% zero-shot (SAM3 + Gemini, no training)
- Real robot: UR10e + Robotiq 2F-85 at 192.168.56.101
- Standalone Python pipeline (no ROS 2 required for real robot)

```
Instruction: "pick up the black bowl and place it on the plate"
                    |
                    v
    +-------------------------------------------+
    |  1. SAM3 Perception                       |
    |     Capture RGB-D -> text-prompted masks   |
    |     -> depth backprojection -> 3D world    |
    |     -> detection map with positions        |
    +---------------------+---------------------+
                          |
                          v
    +-------------------------------------------+
    |  2. Gemini Planner (temperature=0)        |
    |     Instruction + labels -> YAML behavior |
    |     tree with primitives: move, grasp,    |
    |     release, push, open_drawer, turn_knob |
    +---------------------+---------------------+
                          |
                          v
    +-------------------------------------------+
    |  3. Score Executor                        |
    |     Flatten tree -> dispatch actions       |
    |     -> CBF safety filter -> robot motion  |
    +---------------------+---------------------+
                          |
                          v
    +-------------------------------------------+
    |  4. Controller                            |
    |     Sim: IK-guided mass-matrix OSC        |
    |     Real: ur_rtde moveL/servoJ            |
    +-------------------------------------------+
```

---

## System Architecture

### Package Structure

```
spark/
+-- src/
|   +-- spark_perception/      # SAM3 vision system (ROS 2 node)
|   +-- spark_sequencer/       # LLM planning + BT execution (ROS 2)
|   +-- spark_controller/      # Cartesian impedance control (ROS 2)
|   +-- spark_sim/             # MuJoCo simulation + scene generation
|   +-- spark_bringup/         # Launch files
|   +-- spark_interfaces/      # ROS 2 message/service/action definitions
|   +-- spark_moveit_config/   # MoveIt2 configuration (UR10e + Robotiq)
|   +-- spark_grasp_gen/       # Grasp generation (MuJoCo-based)
|   +-- spark_bench/           # LIBERO benchmarking harness
|   +-- spark_real/            # Standalone real-robot pipeline (no ROS 2)
|   +-- spark_fallback/        # Fallback controllers
+-- paper/                     # CoRL / conference materials
+-- spark.repos                # VCStool dependency manifest
```

### Two Execution Paths

**Simulation (ROS 2):** Full ROS 2 Jazzy node graph with MuJoCo physics.

```
spark_sim -> /camera{1,2,3}/* -> spark_perception -> /spark/keypoints
  -> spark_sequencer -> /bt_executor/action_goals -> spark_controller
  -> /joint_trajectory_controller/* -> spark_sim
```

**Real Robot (Standalone):** Single-process Python pipeline. No ROS 2 dependency. Communicates with UR10e via ur_rtde at 500 Hz.

```
spark_real/
+-- pipeline.py                # End-to-end orchestrator
+-- run.py                     # CLI entry point (test, planner, VLA modes)
+-- server.py                  # FastAPI web server (localhost:8888)
+-- sim_pipeline.py            # Standalone MuJoCo simulation (no ROS 2)
+-- frontend/index.html        # Web UI
+-- routes/                    # FastAPI route modules
|   +-- core.py                # Status, capture, detect, plan, execute
|   +-- control.py             # Gripper, move, velocity, home, stop
|   +-- calibration.py         # Anchor, DA3, auto-calibrate
|   +-- streaming.py           # WebSocket, camera capture
|   +-- visualization.py       # Drawing helpers, overlays
|   +-- models.py              # Pydantic request models
|   +-- state.py               # Shared server state
+-- perception/
|   +-- camera.py              # Azure Kinect + RealSense D435I + USB
|   +-- spark_perception.py    # SAM3 detection + depth backprojection
|   +-- sam3_detector.py       # SAM3 model wrapper
|   +-- da3_pose.py            # DA3 multi-view pose + metric depth
|   +-- equigrasp.py           # EquiGraspFlow SE(3) grasp generation
|   +-- equigraspflow.py       # EquiGraspFlow subprocess wrapper
+-- planning/
|   +-- spark_planner.py       # Gemini/OpenAI -> YAML behavior tree
+-- control/
|   +-- score_executor.py      # BT primitive -> robot command dispatch
|   +-- execution_recovery.py  # Grasp recovery, task verification, re-detection
|   +-- trajectory.py          # Trajectory recording + visualization
|   +-- safe_robot.py          # CBF safety filter wrapper
|   +-- barriers.py            # Barrier function definitions
|   +-- obstacle_map.py        # Depth-based obstacle detection thread
|   +-- ur10e_driver.py        # UR10e RTDE + Robotiq 2F-85 control
|   +-- vla_controller.py      # VLA policy loop (pi0.5, OpenVLA)
+-- skills/
|   +-- registry.py            # @spark_skill decorator + auto-discovery
|   +-- primitives.py          # Core 5: move_to_keypoint, grasp, release, move_relative, wait
|   +-- manipulation.py        # Extended: wiggle, push_object, open_drawer, screw
|   +-- grasping.py            # SE(3) 6-DOF grasp via EquiGraspFlow
|   +-- recovery.py            # Error recovery skills
+-- calibration.py             # CameraCalibration dataclass + SVD solver
+-- calibrate.py               # Interactive calibration script (6-point SVD)
+-- da3_anchor.json            # Saved camera extrinsics (per-camera SVD)
+-- configs/
|   +-- ur10e_default.yaml     # Robot, gripper, workspace, camera config
```

---

## Perception (SAM3)

### Detection Pipeline

SAM3 (Segment Anything Model 3) takes an RGB image and text prompts, returning per-object segmentation masks with bounding boxes and confidence scores. It is open-vocabulary: any object description works as a prompt.

```python
prompts = ["black bowl", "plate", "stove", "cabinet"]
# SAM3 returns: masks, bounding boxes, confidence scores per prompt
```

### 3D Backprojection

Each mask centroid (cx, cy) is combined with depth to produce a 3D world position:

```python
# Camera intrinsics
f = image_height / (2 * tan(fovy / 2))

# Pixel -> camera frame
x_cam = (cx - width/2) * depth / f
y_cam = -(cy - height/2) * depth / f
z_cam = -depth

# Camera frame -> world frame (via calibrated extrinsic)
point_world = cam_rotation @ [x_cam, y_cam, z_cam] + cam_position
```

Depth sources:
- **Simulation**: MuJoCo rendered depth (exact)
- **Real robot**: Azure Kinect NFOV depth or RealSense D435I stereo
- **Fallback**: DA3 monocular depth estimation (learned, less accurate)

### Body Matching (Simulation Only)

In simulation, SAM3's approximate 3D position is refined by matching detected labels to MuJoCo body names and using exact `data.xpos` positions:

1. Substring match: `"bowl"` in `"akita_black_bowl_1_main"`
2. Word match (>3 chars): match individual words
3. Short word match: last resort for 3-char words
4. Robot body filter: skip `robot0`, `gripper0`, `mount0` prefixes

### Detection Map

The final output maps labels to 3D positions:

```python
det_map = {
    "bowl":  {"position_3d": [0.052, -0.074, 0.970], "confidence": 0.86},
    "plate": {"position_3d": [0.010,  0.260, 0.962], "confidence": 0.72},
}
```

### Multi-Camera Fusion (Real Robot)

On the real robot, detections from multiple cameras are merged with priority:
1. Calibrated fixed cameras over wrist camera
2. Detections with valid 3D positions (within workspace bounds) preferred
3. Higher confidence breaks ties

### Part-Affordance Detection

For tools and complex objects, the perception system detects semantic parts (handle, blade, rim, lid) and assigns grasp priorities. The gripper targets the best-graspable part rather than the object centroid.

---

## Planning (Gemini)

### Input / Output

Gemini receives:
- System prompt with available primitives, rules, output format
- Natural language instruction
- Available keypoint labels from perception
- Annotated scene image (enables visual grounding)

Gemini outputs a YAML behavior tree ("score"):

```yaml
task: pick up the black bowl and place it on the plate
tree:
  type: sequence
  children:
    - type: move_to_keypoint
      params:
        keypoint_label: "bowl"
        offset_z: 0
    - type: grasp
      params:
        force: 100
    - type: move_relative
      params:
        dz: 0.20
    - type: move_to_keypoint
      params:
        keypoint_label: "plate"
        offset_z: 0.05
    - type: release
```

### Available Primitives

| Primitive | Parameters | Description |
|-----------|-----------|-------------|
| `move_to_keypoint` | label, offset_x/y/z | Move EE to detected object position |
| `grasp` | force | Close gripper |
| `release` | - | Open gripper |
| `move_relative` | dx, dy, dz | Move relative to current position |
| `open_drawer` | joint_name, pull_direction/distance | Open a slide-joint drawer |
| `push_object` | label, direction, distance | Push object toward target |
| `turn_knob` | label | Rotate a stove knob |

### Determinism

Gemini is called with `temperature=0`. The same instruction + labels produce the same plan.

### Validation and Fallback

If Gemini returns an empty plan or misses the pick object, the pipeline falls back to a scripted planner using pick/place hints directly.

---

## Control and Execution

### Score Executor

The executor walks the behavior tree, flattens nested sequences, and dispatches each action:

- **move_to_keypoint**: Looks up label in detection map, applies offset. If not holding: 3-step approach (above -> mid -> target). If holding: lift -> XY at safe height -> descend.
- **grasp**: Close gripper, hold with gravity compensation, set holding state.
- **move_relative**: Compute target from current EE position + delta.
- **release**: Open gripper, retract upward.
- **Pre-actions**: Some tasks require setup (open_drawer via force, turn_knob via joint rotation) before the main plan.

### Transport Strategy

Held objects follow a lift-then-move pattern to avoid collisions:

```
1. LIFT to safe_height = max(current_z, place_z + 0.10, 0.35)
2. XY move at safe height to above placement target
3. DESCEND to final placement height
```

### Simulation Controller: IK-guided Mass-Matrix OSC

In simulation, SPARK uses Operational Space Control with mass-matrix weighting for precise Cartesian tracking:

```
Lambda = (J * M^-1 * J^T)^-1       # Task-space inertia
F = Lambda * (Kp * error - Kd * vel) # Inertia-shaped PD
tau = J^T * F + tau_nullspace + gravity_comp
```

The nullspace (4 DOF for 7-joint Panda) tracks a pre-computed IK solution, keeping the arm away from singularities and joint limits without affecting end-effector motion.

Stall detection triggers a joint-space computed-torque fallback if task-space error stops decreasing for 200+ simulation steps.

| Parameter | Value | Notes |
|-----------|-------|-------|
| KP_TASK | 150.0 | Cartesian stiffness (200 causes oscillations) |
| DAMPING | 1.0 | Critical damping ratio |
| KP_JOINT | 30.0 | Nullspace stiffness |
| TORQUE_LIM | 87 Nm | Panda joints 1-5 (6-7 boosted to 80 Nm) |

### Real Robot Controller

On the UR10e, the executor uses ur_rtde directly:
- `moveL` for Cartesian linear moves
- `servoJ` for VLA policy control loops
- Velocity and acceleration limits enforced per-command
- Context manager pattern for safe connect/disconnect

---

## Real Robot Deployment

### Hardware

- **Arm**: UR10e (6 DOF) at 192.168.56.101
- **Gripper**: Robotiq 2F-85 parallel gripper
- **Cameras**:
  - 2x Azure Kinect DK (sideview + birdview), NFOV_UNBINNED depth, 720P color
  - 1x Intel RealSense D435I (wrist-mounted, 3D-printed bracket)
- **Compute**: Workstation with CUDA GPU for SAM3 inference

### USB Bandwidth Management

Azure Kinects are high-bandwidth USB3 devices. To avoid contention:
- Cameras are opened sequentially with 3-second stagger
- RGB-only streaming during live view; depth captured on-demand during detection
- Camera serials are mapped to device IDs at startup for consistent sideview/birdview assignment

### Web Frontend

A FastAPI server at `localhost:8888` provides the operator interface:

```bash
cd ~/spark/src
python -m spark_real.server --port 8888
```

Features:
- Live tiled camera view (sideview + birdview + wrist) with camera selector
- Zoom and pan on camera feeds
- Click-to-detect and box-prompt detection modes
- Text-prompted SAM3 detection with 3D position overlay
- Gemini plan generation and review
- Robot teleop controls (Cartesian jog, velocity mode, gripper open/close)
- Pipeline status monitoring (camera, robot, perception, planner states)
- RGB/depth stream toggle

### Operating Modes

**Server mode** (`python -m spark_real.server`): Web UI for interactive operation.

**Pipeline mode** (`python -m spark_real.pipeline`): Headless end-to-end execution.

**Test mode** (`python -m spark_real.run --mode test`): Interactive REPL for robot teleoperation.

**VLA mode** (`python -m spark_real.run --mode vla`): Closed-loop VLA policy execution (pi0.5, OpenVLA) at ~10 Hz with safety limits (max joint delta 0.1 rad/step, max TCP delta 0.02 m/step, workspace bounding box).

---

## CBF Safety Filter

### Design

A Control Barrier Function (CBF) safety filter sits between the executor and the robot as a `SafeRobot` wrapper. At every control tick it solves a QP:

```
min_u  ||u - u_des||^2
s.t.   A_cbf @ u >= b_cbf          (barrier constraints)
       u_min <= u <= u_max          (velocity limits)
```

where `u` is the 6-D TCP velocity `[vx, vy, vz, wrx, wry, wrz]`.

### Barrier Functions

Five barrier classes enforce `h(x) >= 0` as the safe-set invariant:

| Barrier | Constraint | Default Parameters |
|---------|-----------|-------------------|
| SingularityBarrier | Elbow joint away from 0 and +/-pi | eps = 0.20 rad |
| ReachBarrier | TCP inside max-reach sphere | r_max = 1.15 m |
| WorkspaceBarrier | TCP inside axis-aligned box (6 half-planes) | x: [-1.1, -0.5], y: [-0.5, 0.7], z: [-0.15, 0.50] |
| ObstacleBarrier | TCP outside safety sphere per obstacle | r_safe = 0.10 m |
| ForceBarrier | Measured TCP force below limit | f_max = 30 N, threshold = 10 N |

The `BarrierSet` aggregates all active barriers into `(A_cbf, b_cbf)` constraint matrices. Each barrier provides `(h, grad_h)` where `grad_h` is the 6-D gradient with respect to TCP velocity. The QP constraint for each barrier is:

```
grad_h @ u >= -eta * h
```

Convergence rates (`eta`) are tuned per barrier type:
- Kinematic (singularity, reach, workspace): eta = 0.3
- Obstacle: eta = 0.5
- Force: eta = 0.8

### QP Solver

OSQP (Operator Splitting QP) solves the constrained optimization. The solver is warm-started between ticks for real-time performance.

### Script Validation

For `moveL` commands, `SafeRobot` samples intermediate points along the linear path and checks all barrier constraints before forwarding the command. If any intermediate point violates a barrier, the move is rejected or clipped.

### Current Status

The `SafeRobot` wrapper is implemented and designed but currently commented out in the pipeline pending completion of calibration validation. The barrier functions and obstacle map are fully implemented and tested independently.

---

## Calibration

### Camera-to-Robot Transform

Each fixed camera needs a 4x4 extrinsic matrix mapping camera frame to robot base frame. SPARK uses a 6-point SVD calibration procedure per camera:

1. User jogs the robot TCP to 6 known positions on a foam surface, recording each robot-frame coordinate
2. For each position, the user clicks the corresponding pixel in the camera view
3. Clicks are backprojected to 3D using hardware depth (Azure Kinect NFOV or RealSense)
4. An SVD-based rigid transform is computed from the 6 point correspondences (camera frame to robot frame)
5. The resulting 4x4 extrinsic matrix is stored per camera in `da3_anchor.json`

Achieved accuracy: 3-8 mm mean reprojection error per camera (verified via back-projection of anchor points).

The calibration is implemented in `spark_real/calibration.py` (CameraCalibration dataclass + SVD solver) and `spark_real/calibrate.py` (interactive calibration script).

### Intrinsics

Camera intrinsics (fx, fy, cx, cy) are extracted automatically:
- Azure Kinect: from pyk4a calibration API
- RealSense D435I: from pyrealsense2 intrinsics
- MuJoCo cameras: computed from field of view

### Wrist Camera

The wrist-mounted RealSense D435I has a known tool offset (camera-to-TCP transform). Its extrinsic is computed dynamically from robot FK:

```python
T_cam_to_world = T_tcp_to_world @ T_cam_to_tcp
```

### Persistence

Calibrations are saved/loaded as JSON via `da3_anchor.json`, which stores per-camera extrinsic matrices, calibration error metrics, and anchor point pairs for verification. The `CameraCalibration` dataclass also supports YAML serialization.

---

## Skill Registry

### Decorator-Based Registration

SPARK uses a decorator-based skill system (`spark_real/skills/registry.py`) that allows new manipulation primitives to be added without modifying the executor or planner:

```python
@spark_skill(
    name="wiggle",
    description="Oscillate EE to break static friction during insertion",
    params={"amplitude": float, "frequency": float, "duration": float, "axis": str},
)
def wiggle(executor, params: dict) -> ExecutionResult:
    ...
```

At import time, the decorator registers each skill in a global registry. `SkillRegistry.auto_discover()` finds all decorated functions in `spark_real.skills`. The planner dynamically injects registered skill descriptions into its system prompt, so Gemini can use new skills without code changes.

### Skill Packages

| Package | Skills | Description |
|---------|--------|-------------|
| `primitives.py` | move_to_keypoint, grasp, release, move_relative, wait | Core 5 battle-tested primitives |
| `manipulation.py` | wiggle, push_object, open_drawer, screw | Extended manipulation skills |
| `grasping.py` | grasp_se3 | SE(3) 6-DOF grasp via EquiGraspFlow |
| `recovery.py` | (error recovery skills) | Fault recovery primitives |

### Dispatch Flow

The score executor checks the skill registry first for each action type. If a registered skill exists, it is called with the executor instance and YAML params. Otherwise, the executor falls back to its built-in handlers.

---

## Simulation

### MuJoCo Physics

- **Robot**: UR10e + Robotiq 2F-85 (from mujoco_menagerie) or Franka Panda (LIBERO)
- **Cameras**: 3 views (overhead, front-side, back-side), 640x480 RGB + depth
- **Control**: Actuator names (shoulder_pan, shoulder_lift, etc.), gripper 255 = open, 20 = closed

### Procedural Scene Generation (Neural MP-inspired)

```
PartNeXt GLB --> obj2mjcf --> MuJoCo XML with textures
                    |
PyBullet collision checking --> Collision-normal placement
                    |
Declarative scene definitions --> Generated scenes
```

- **obj2mjcf**: Converts OBJ to MuJoCo XML, splits meshes by material
- **Collision-normal placement**: Shifts objects along collision normals instead of rejection sampling. Scales to 10+ objects per scene.
- **Hierarchical placement**: Objects on containers (tables, shelves)
- **PartNeXt dataset**: 23,519 GLB models with part annotations

Scene generator CLI:

```bash
cd ~/spark/src/spark_sim/spark_sim
python scene_generator.py --preset tabletop_small --num-objects 3 --textured
python scene_generator.py --hierarchical --containers "Table" --num-objects 5
python scene_generator.py --list-presets
```

### Realistic Scene Templates

Built from LIBERO, robosuite, object_sim, and furniture_sim assets:

- `perception_test_scene.xml` -- basic lab environment
- `kitchen_scene.xml` -- actuated hinge cabinets, marble counter, fridge, wood table
- `living_room_scene.xml` -- sofa, coffee table, bookshelf, side table

Actuated furniture includes working hinge/slide joints with position sensors.

### LIBERO-PRO Benchmark

The surviving runner is the LIBERO-PRO fair protocol (`run_spark_libero_pro_fair.py`).
The plain-LIBERO `run_spark_libero_full.py` runner was removed.

```bash
cd ~/spark
conda activate openvla_env
MUJOCO_GL=egl PYTHONPATH=src/libero_pro:src/sam3:src \
    python -m spark_bench.run_spark_libero_pro_fair \
    --suite object --perturbation position --num-trials 10
```

Multi-step tasks use per-step Gemini calls with GT body injection, home reset between steps, and SAM3 re-detection after each step.

---

## ROS 2 Integration

### Topic Graph (Simulation Path)

```
spark_sim (MuJoCo)
    +-> /camera{1,2,3}/color/image_raw --> spark_perception
    +-> /camera{1,2,3}/depth/image_raw --> spark_perception

spark_perception (SAM3)
    +-> /spark/keypoints ------------> spark_sequencer
    +-> /spark/labeled_keypoints ----> bt_executor
    +-> /spark/object_masks ---------> spark_moveit_config (scene sync)

spark_sequencer (Gemini LLM)
    +-> Behavior Tree YAML ----------> bt_executor

bt_executor (py_trees)
    +-> /bt_executor/action_goals ---> spark_controller

spark_controller
    +-> /joint_trajectory_controller/* -> spark_sim
```

### Launch

```bash
# Full pipeline (LLM generates BTs from natural language)
ros2 launch spark_bringup mujoco_full_pipeline.launch.py llm_mode:=auto

# Manual mode (provide BT YAML manually)
ros2 launch spark_bringup mujoco_full_pipeline.launch.py llm_mode:=manual

# With specific scene
ros2 launch spark_bringup mujoco_full_pipeline.launch.py scene:=kitchen_scene.xml
```

Staggered startup: sim (t=0s) -> perception (t=2s) -> sequencer + executor (t=5s) -> task commander (t=8s).

### Custom Messages

- `LabeledKeypoint` / `LabeledKeypointArray` -- semantic keypoints with poses and confidence
- `ObjectMask` / `ObjectMaskArray` -- mask data for MoveIt scene sync
- Actions: `ProposeKeypoints`, `InitializeTracking`, `ExecuteInstruction`
- Services: `ExecuteScore`, `ExecuteBehaviorTree`

### MoveIt2 (Optional)

Collision-aware motion planning for the UR10e + Robotiq 2F-85:
- `scene_sync_node` bridges `/spark/object_masks` into the MoveIt planning scene
- Requires sourcing `ws_robotiq2` for the URDF
- OMPL, Pilz, and custom planner configurations

### Behavior Tree Framework

py_trees-based executor with 18 custom behaviors:
- Movement: MoveToKeypoint, MoveRelative, WiggleMotion
- Gripper: GraspObject, ReleaseObject, AdjustGripStrength
- Perception: EnsureKeypointVisible, SearchForKeypoint
- Conditions: CheckGraspQuality, CheckInsertionComplete
- Composites: Sequence, Selector (fallback), Parallel

---

## Key Technologies

| Component | Technology | Notes |
|-----------|-----------|-------|
| Perception | SAM3 (Meta) | Open-vocabulary detection + video tracking |
| Planning | Gemini (Google) | Vision + language, temperature=0 |
| BT Execution | py_trees | Industry-standard behavior trees |
| Physics | MuJoCo | Simulation + scene generation |
| Real Control | ur_rtde | 500 Hz RTDE, no ROS 2 overhead |
| Safety | OSQP | CBF-QP solver for barrier constraints |
| Web UI | FastAPI | REST + WebSocket, localhost:8888 |
| Motion Planning | MoveIt2 | Optional, collision-aware (sim only) |
| Robot | UR10e + Robotiq 2F-85 | 6-DOF arm + parallel gripper |
| Cameras | Azure Kinect DK, RealSense D435I | RGB-D, 2 fixed + 1 wrist |
| Language | Python 3.12+ | Primary language |
| GPU | CUDA 12.6+ | SAM3 inference |
| Environment | Conda (spark_conda) | Hooks auto-source ROS 2, API keys |

---

## Design Decisions

**SAM3 over SAM2 + DINO + CLIP**: Single unified model replaces three separate models, reducing perception code from ~1650 to ~450 lines.

**Standalone real pipeline**: Real-robot control uses ur_rtde directly rather than ROS 2 topics, keeping latency minimal for VLA policy loops that need consistent ~10 Hz control.

**Mask-based grasp offsets**: Perception analyzes mask geometry and part-level affordances to compute grasp points (handle of a mug, not its centroid).

**Collision-normal placement**: Scene generation uses Neural MP's Algorithm 1 instead of rejection sampling. Objects shift along collision normals, scaling to dense scenes without exponential retry cost.

**CBF safety filter as a wrapper**: SafeRobot wraps the robot driver transparently. The executor does not need to know about safety constraints -- they are enforced at the velocity command level via QP projection.

**6-point SVD calibration**: Lightweight and repeatable. No fiducial markers or checkerboards needed. Six TCP-to-pixel correspondences on a foam surface yield 3-8 mm accuracy with SVD-based rigid transform estimation.

**Decorator-based skill registry**: New skills are auto-discovered and injected into the planner's system prompt. Adding a manipulation primitive requires only a decorated function -- no executor or planner modifications needed.

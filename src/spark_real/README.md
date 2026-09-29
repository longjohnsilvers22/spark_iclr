# spark_real

Pure-Python (no ROS 2) FastAPI server wrapping a perception -> planning ->
control pipeline for real robot manipulation. One codebase drives three
embodiments, selected by a single launch flag:

- Franka FR3 via franky/libfranka (`robots/franka/franka_driver.py`, 1 kHz)
- UR10e via ur-rtde (`control/ur10e_driver.py`)
- Bimanual Panda + FR3 via the Bamboo torque controller
  (`robots/bimanual_franka/bimanual_franka_driver.py`)

Everything embodiment-specific (IP, home pose, workspace box, gripper,
camera roster, control block) lives in `configs/<family>_default.yaml` plus
optional per-machine overlays in `configs/machines/<machine>.yaml`. The
family is chosen at launch with `--robot {franka,ur10e,bimanual_franka,g1}`;
the rest of the stack is robot-agnostic.

## Architecture and data flow

```
RGB + Depth (Azure Kinect / RealSense)
    |  SAM3 open-vocab segmentation + DA3 metric depth      perception/spark_perception.py
    v
3D keypoints (label -> position_3d, orientation, geometry)
    |  one Gemini call writes a typed YAML behavior tree    planning/spark_planner.py
    v
SPARK score (behavior tree)
    |  ScoreExecutor flattens the tree and dispatches nodes control/score_executor.py
    v
Robot driver (FrankaDriver / UR10eDriver / Bimanual...)   robots/, control/
    |  on failure: re-perceive + re-detect + retry          control/execution_recovery.py
```

The pipeline (`pipeline.py`, `SPARKRealPipeline`) is mixin-composed from
`InitMixin`, `BimanualInitMixin`, `PerceptionMixin`, `ExecutionMixin`,
`RunMixin`, `IOMixin`. The three entry points a caller uses:

- `pipeline.detect(...)` (`pipeline_perception.py`) -> grounds text prompts
  into a detection map.
- `pipeline.plan(...)` (`pipeline_execution.py`) -> calls the LLM, returns
  a SPARK score.
- `pipeline.run_task(...)` (`pipeline_run.py`) -> full capture/detect/plan/
  execute loop, with closed-loop re-perception passes (`max_task_passes`).

## Directory layout

```
spark_real/
  server.py            FastAPI app + main() tyro CLI; early Kinect open, IK warmup
  run.py               Headless CLI (test / planner / vla subcommands)
  config.py            SparkConfig (CLI) + RobotProfile + YAML loader/merger
  pipeline.py          SPARKRealPipeline (mixin-composed orchestrator)
  pipeline_init.py     Driver + camera + executor bring-up (single arm)
  pipeline_init_bimanual.py   Bimanual bring-up
  pipeline_perception.py / _execution.py / _run.py / _io.py   pipeline mixins

  perception/
    spark_perception.py   SPARKPerception: SAM3 masks + DA3 depth -> 3D detections
    sam3_detector.py      SAM3 model wrapper (open-vocab text/box/point prompts)
    da3_pose.py           DA3 metric depth estimator
    camera.py             Azure Kinect + RealSense capture
    camera_registry.py / bimanual_camera_registry.py   role-keyed camera roster
    mask_geometry.py      OBB / orientation / aspect-ratio from masks
    equigrasp.py, equigraspflow.py   EquiGraspFlow SE(3) grasp candidates

  planning/
    spark_planner.py      SPARKPlanner: Gemini/OpenAI -> YAML behavior tree

  control/
    score_executor.py     ScoreExecutor = mixin composition of the below
    executor_core.py      ScoreExecutorCore: init, dispatch, workspace bounds, abort
    executor_motion.py    MotionMixin: move_to, approach, transport, servo
    executor_ik.py        IkMixin: movej via IK, movej_to_pose, legato blend
    executor_grasp.py     GraspMixin: gripper open/close/grasp + verify
    executor_release.py   ReleaseMixin: release with tilt
    executor_verify.py    VerifyMixin: task verification + snapshot recovery
    execution_recovery.py grasp recovery + re-detection on failure
    safe_robot.py         SafeRobot: CBF QP safety filter (workspace/reach/obstacle/force)
    fr3_ik_pyroki.py      FR3 IK via pyroki/jaxls (solve_ik, fk, jacobian, warmup)
    fr3_ik.py             FR3 IK Pinocchio fallback (SPARK_IK=pinocchio)
    ur10e_driver.py       UR10eDriver: ur-rtde + Robotiq 2F-85
    cartesian_servo.py    UR10e velocity (speedl) Cartesian servo, bypasses UR IK
    vla_controller.py     optional VLA policy loop (pi0.5 / OpenVLA)

  robots/
    franka/
      franka_base.py        FrankaDriverBase: abstract interface + shared gripper/consts
      franka_driver.py      FrankaDriver: franky/libfranka, 1 kHz motion generators
      franka_bamboo_driver.py  FrankaBambooDriver: torque-impedance over ZMQ to a C++ node
      franka_torque_driver.py  external-torque servo driver
      desk_session.py       auto-unlock + enable FCI via the Desk web interface
      urdf/                 fr3_franka_hand.urdf (used by IK)
    ur10e/                grippy.script (Robotiq URScript functions)
    bimanual_franka/
      bimanual_franka_driver.py  BimanualFrankaDriver: aggregates two arms (franky or bamboo)
      dynamixel_gripper.py, ssg48_gripper.py   dual jaw backends
    g1/                   Unitree G1 driver (g1_driver.py)

  skills/
    registry.py           @spark_skill decorator + SkillRegistry auto-discovery
    primitives.py         move_to_keypoint, grasp, release, move_relative, wait
    grasping.py / grasp_se3.py / grasp_top_down.py / grasp_horizontal.py   grasp strategies
    manipulation.py, tool_use.py, sweep.py, scrub.py, pour.py, place_in_slot.py
    cloth.py, cloth_fold.py, bimanual_cloth.py   cloth / bimanual primitives
    recovery.py           recovery primitives

  routes/                FastAPI route modules (one per concern)
    core.py, control.py, control_franka.py, control_teleop.py, detection.py,
    execution.py, calibration*.py, streaming.py, visualization.py, bimanual.py,
    osc_executor.py, recording.py, episode.py, state.py, models.py, dry_run.py

  calibration/
    solver.py             hand-eye / anchor solver (Procrustes / SVD)
    model.py              calibration dataclasses

  configs/
    franka_default.yaml, ur10e_default.yaml, bimanual_franka_*.yaml, g1_default.yaml
    machines/<machine>.yaml   per-host overlays (e.g. ANON-LAB.yaml)
  frontend/              single-page web UI
```

## The control stack

The execution path layers three levels. Read them outer to inner.

1. `ScoreExecutor` (`control/score_executor.py`) is the behavior-tree
   executor. It is composed from `ScoreExecutorCore` plus `MotionMixin`,
   `IkMixin`, `GraspMixin`, `ReleaseMixin`, `VerifyMixin`. `execute_score()`
   flattens the typed tree, normalizes params, and dispatches each node to
   the matching method or to a `@spark_skill` from the registry. It owns
   workspace-bound checks (`_check_workspace`), per-primitive timeouts
   (`control/primitive_timeouts.py`), and abort handling
   (`abort()` -> `/api/stop`). On a failed post-condition it routes into
   `execution_recovery.py`, which re-detects the target and retries.

2. `SafeRobot` (`control/safe_robot.py`) optionally wraps the driver. It
   intercepts velocity and URScript commands and solves a Control Barrier
   Function QP (osqp) every tick to enforce reach, workspace, singularity,
   obstacle, and contact-force barriers (`BarrierSet`, `SafetyConfig`),
   forwarding the possibly clamped command on. Methods it does not override
   pass through via `__getattr__`, so the executor sees a normal driver.

3. The driver speaks to the hardware. All Franka drivers implement the
   abstract `FrankaDriverBase` (`robots/franka/franka_base.py`), which
   defines the contract (`connect`, `move_to_joint_config`, `move_linear`,
   `get_tcp_pose`, gripper hooks) and supplies the shared constants
   (`HOME_CONFIG`, `JOINT_LIMITS`, `FLANGE_T_TCP`) and gripper logic.
   `UR10eDriver` is a separate class exposing the same shim surface
   (`SUPPORTS_URSCRIPT`, `GRIPPER_TYPE`, the same method names); the
   executor degrades cleanly across arms via `hasattr` guards.

Drivers:

- `FrankaDriver` (`robots/franka/franka_driver.py`): franky v1.1.3 motion
  generators (libfranka, 1 kHz realtime callback). Delegates motion to
  `CartesianMotion` / `JointMotion` / velocity motions; Ruckig handles
  smoothing.
- `FrankaBambooDriver` (`robots/franka/franka_bamboo_driver.py`): 1 kHz C++
  joint-impedance controller reached over a ZMQ client
  (`execute_joint_impedance_path`). It eliminates franky motion-generator
  reflex errors and manages the C++ subprocess lifecycle
  (`_start_bamboo` / `_stop_bamboo` / `_restart_bamboo`).
- `UR10eDriver` (`control/ur10e_driver.py`): motion/state over ur-rtde
  (`RTDEControlInterface` / `RTDEReceiveInterface` / `RTDEIOInterface`); raw
  URScript (`speedl`/`speedj`) goes through a TCP socket on port 30002 for
  the Cartesian servo and VLA velocity streaming. Robotiq 2F-85 gripper.
- `BimanualFrankaDriver` (`robots/bimanual_franka/bimanual_franka_driver.py`):
  composes two single-arm sub-drivers (`.left`, `.right`) by aggregation,
  selected by `backend: "franky"` or `"bamboo"`. Every shim method takes an
  `arm: "left"|"right"` kwarg; aggregate queries return a 14-vector.

IK:

- FR3 is 7-DOF (redundant), so it uses optimization-based IK in
  `control/fr3_ik_pyroki.py`. `solve_ik(target_pos, target_orient, q_seed)`
  builds a jaxls factor graph and runs Levenberg-Marquardt over costs:
  `pose_cost`, `limit_cost`, `rest_cost`, `manipulability_cost` (plus
  optional self/world collision costs gated by env vars). JIT-compiled:
  first call ~7-10 s, then <100 ms; `warmup()` pays the compile off the
  critical path at server start. JAX is pinned to CPU on this host.
  `fk()` and `jacobian()` are also exported. Pinocchio fallback in
  `fr3_ik.py` via `SPARK_IK=pinocchio`.
- UR10e is 6-DOF and uses the controller's onboard analytic IK through
  ur-rtde; no SPARK-side solver is involved for the UR.

## Running it

Launch from `~/spark/src` (so `spark_real` is importable). The CLI is tyro
over `SparkConfig` (`config.py`); `--family` is the canonical name, `--robot`
is an alias.

```bash
conda activate spark_conda
cd ~/spark/src

# Franka FR3 (default), UI at http://localhost:8888
DISPLAY=:1 XAUTHORITY=/run/user/1000/gdm/Xauthority \
  python -m spark_real.server --robot franka --auto-unlock --port 8888

python -m spark_real.server --robot ur10e --port 8888
python -m spark_real.server --robot bimanual_franka --machine ANON-LAB --port 8888

# Hardware-free boot (no robot, no pipeline init, no cameras)
python -m spark_real.server --no-robot --no-init --no-kinect

# Headless CLI (no web UI)
python -m spark_real.run planner --instruction "pick up the red block"
python -m spark_real.run test
python -m spark_real.run vla
```

Flags (from `SparkConfig`): `--family/--robot`, `--machine`, `--host`,
`--port`, `--ip`, `--no-robot`, `--no-init`, `--no-kinect`, `--auto-unlock`
(Franka only: unlocks the Desk brakes and enables FCI before connect),
`--strict-verify`, `--closed-loop-passes`.

Status: `curl http://localhost:8888/api/status` reports `robot_connected`
and per-Kinect connection flags. `--auto-unlock` is ignored for non-Franka
families.

## Key classes and functions

| Symbol | File | What it does |
|---|---|---|
| `FrankaDriverBase` | `robots/franka/franka_base.py` | Abstract Franka interface + shared constants and gripper logic |
| `FrankaDriver` | `robots/franka/franka_driver.py` | FR3 via franky/libfranka, 1 kHz motion generators |
| `FrankaBambooDriver` | `robots/franka/franka_bamboo_driver.py` | FR3 joint-impedance (torque) over ZMQ to a C++ node |
| `UR10eDriver` | `control/ur10e_driver.py` | UR10e via ur-rtde + Robotiq 2F-85, onboard IK |
| `BimanualFrankaDriver` | `robots/bimanual_franka/bimanual_franka_driver.py` | Aggregates two arms, per-arm `arm=` kwarg |
| `SafeRobot` | `control/safe_robot.py` | CBF QP safety filter wrapping any driver |
| `ScoreExecutor` | `control/score_executor.py` | Behavior-tree executor (mixin-composed) |
| `ScoreExecutorCore` | `control/executor_core.py` | Executor init, dispatch, workspace bounds, abort |
| `solve_ik` | `control/fr3_ik_pyroki.py` | FR3 IK via jaxls factor graph (pose/limit/rest/manipulability) |
| `SPARKPerception` | `perception/spark_perception.py` | SAM3 + DA3 -> 3D detection map |
| `SPARKPlanner` | `planning/spark_planner.py` | Gemini/OpenAI -> YAML behavior tree |
| `SPARKRealPipeline` | `pipeline.py` | End-to-end orchestrator (detect/plan/run_task) |
| `spark_skill` | `skills/registry.py` | Decorator registering a primitive callable by name |
| `SkillRegistry` | `skills/registry.py` | Auto-discovers all `@spark_skill` functions |

## Configs

Each family has `configs/<family>_default.yaml`; a `--machine X` overlay in
`configs/machines/X.yaml` is deep-merged on top (`load_family_yaml`,
`_deep_merge`). `load_profile()` resolves these into a `RobotProfile`, whose
accessors feed the pipeline:

- `robot`: `ip`, `frequency`, `home_config` (single-arm joint list, or
  `home_config_left`/`_right` for bimanual), `variant`, `urdf`, joint/linear
  velocity and acceleration limits.
- `gripper` (or `grippers` for bimanual): `type`, `speed`, `force`,
  `epsilon_inner/outer`, `open_width`, `closed_width`.
- `workspace`: CBF safety box (`x/y/z_min`, `x/y/z_max`) fed to
  `SafetyConfig`. `control`: executor/SafeRobot constants
  (grasp orientation, table floor, `wrist_refine`). The two are distinct
  boxes by design.
- `cameras`: role-keyed roster (role, type, serial/device, offsets);
  `kinect_*` keys (resolution, depth_mode, fps, hw_sync, master_serial).
- `timeouts`: per-primitive wall-clock budgets, loaded into
  `control/primitive_timeouts.py`.
- `cloth`: fold tuning (`sleeve_lift_h`, `arc_peak`, ...).
- `table_height`, `table_z_floor`, `max_task_passes` (closed-loop passes),
  `collision_behavior` (per-joint / Cartesian thresholds).

Example machine overlay (`configs/machines/ANON-LAB.example.yaml`; copy to
`configs/machines/ANON-LAB.yaml`, which is gitignored, and fill in the IPs) holds
only the delta over `bimanual_franka_default.yaml`: per-arm
`left_ip`/`right_ip`, `backend: bamboo`, the Panda URDF, the SSG-48 gripper
bus, webcam wrists, collision thresholds and fold tuning.

## Where to start reading

1. `robots/franka/franka_base.py` (the driver contract and shared constants)
2. `robots/franka/franka_driver.py` (how a real arm is commanded)
3. `control/fr3_ik_pyroki.py` (how a target pose becomes joint angles)
4. `control/executor_core.py` (how a behavior tree is dispatched)
5. `control/safe_robot.py` (how every command is filtered for safety)

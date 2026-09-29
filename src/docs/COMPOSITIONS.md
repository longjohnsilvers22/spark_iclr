# SPARK Compositions

SPARK chains four stages to go from a natural language instruction to robot execution.

```
Instruction + RGB-D
       |
  [Perception] SAM3 + DA3 depth --> detection map {label: position_3d}
       |
  [Compiler]   Gemini (temperature=0) --> YAML behavior tree ("SPARK score")
       |
  [Sequencer]  Score Executor --> dispatches to skill registry
       |
  [Controller] UR10e via ur-rtde + Robotiq 2F-85 --> robot motion
```

**Perception** combines SAM3 for open-vocabulary text-prompted segmentation with Depth Anything v3 for metric depth estimation. Two depth models are used: DA3 Nested Giant-Large (`DA3NESTED-GIANT-LARGE`) for multi-view pose + depth in `da3_pose.py`, and DA3 Metric Large (`DA3METRIC-LARGE`) for monocular metric depth in `spark_perception.py`. Given RGB images and text prompts, the system produces per-object masks and 3D world positions via depth backprojection through calibrated camera extrinsics. Cameras are 2x Azure Kinect DK (sideview + birdview) with a wrist-mounted RealSense D435I.

**Compiler** is Gemini (`gemini-3.1-pro-preview`, temperature=0). It receives the user instruction, a scene image with annotated detection boxes, and the detected keypoint labels, then emits a YAML behavior tree ("SPARK score") composed of primitives like move_to_keypoint, grasp, release, and move_relative, plus extended skills such as wiggle, screw, and push_object that are dynamically injected from the skill registry.

**Sequencer** is the Score Executor (`control/score_executor.py`). It flattens the nested YAML behavior tree into a linear action sequence and dispatches each action to either a built-in primitive handler or a registered `@spark_skill` function. The skill registry (`skills/registry.py`) auto-discovers decorated skills in the `spark_real.skills` package.

**Controller** drives a UR10e via ur-rtde with a force-controlled Robotiq 2F-85 gripper. Motion is handled through moveL (Cartesian linear), moveJ (joint space), and servoJ (real-time servo at ~10 Hz for VLA policies). For objects requiring non-top-down grasps, EquiGraspFlow generates SE(3) 6-DOF grasp candidates from RGB-D and object masks, which the executor selects and approaches via the `grasp_se3` skill. Default velocity is 0.15 m/s with a 0.35 m safe transport height.

**Safety** uses Control Barrier Functions (CBF) implemented in `control/safe_robot.py`. The `SafeRobot` wrapper sits between the executor and the robot driver, solving a QP at every control tick to enforce workspace bounds, singularity avoidance, reach limits, obstacle clearance, and contact force limits.

# UR10e Production Deployment Guide

Current as of: 2026-05-25, branch `franka-deploy`.

## Architecture Overview

SPARK on the UR10e is a standalone Python application. There is no ROS
dependency at runtime. The deprecated ROS 2 directories (`spark_perception`,
`spark_sequencer`, `spark_controller`, `spark_bringup`, `spark_moveit_config`)
remain in the tree but are unused -- do not source a ROS workspace, do not
launch via `ros2 launch`, do not build with `colcon` or `catkin`.

```
FastAPI server (spark_real/server.py)
  |
  +-- SAM3 perception (open-vocabulary detection + depth + 3D lifting)
  +-- Gemini planner  (YAML behaviour-tree generation, temperature=0)
  +-- ScoreExecutor   (BT interpreter -> skill dispatch)
  +-- UR10eDriver     (RTDE protocol, URScript gripper commands)
  +-- Azure Kinect    (birdview + sideview via pyk4a / libk4a)
  +-- [optional] RealSense D435I (wrist camera for grasp refinement)
  +-- [optional] EquiGraspFlow   (SE(3) grasp synthesis)
  +-- [optional] FlexiTac        (tactile sensing)
```

### Key source files

| Path | Role |
|------|------|
| `spark_real/server.py` | FastAPI app, CLI entry point, signal/shutdown handlers |
| `spark_real/pipeline.py` | `SPARKRealPipeline` -- orchestrates init, capture, detect, plan, execute |
| `spark_real/control/ur10e_driver.py` | `UR10eDriver` -- RTDE joint/Cartesian motion + Robotiq 2F-85 |
| `spark_real/robots/factory.py` | `make_robot_driver()` -- multi-embodiment factory (UR10e, Franka, G1) |
| `spark_real/control/score_executor.py` | `ScoreExecutor` -- BT interpreter, skill dispatch, retry logic |
| `spark_real/control/safe_robot.py` | `SafeRobot` -- CBF workspace/singularity safety filter |
| `spark_real/perception/spark_perception.py` | `SPARKPerception` -- SAM3 + depth + 3D lifting |
| `spark_real/planning/spark_planner.py` | `SPARKPlanner` -- Gemini LLM BT generation |
| `spark_real/configs/ur10e_default.yaml` | UR10e hardware config (IP, limits, workspace, gripper) |
| `spark_real/skills/primitives.py` | Core skills: move_to_keypoint, grasp, release, pour, sweep, etc. |
| `spark_real/skills/tool_use.py` | Extended skills: pour_to_level, sweep_to_container, sponge_wash, etc. |
| `spark_real/skills/grasping.py` | grasp_se3 (EquiGraspFlow-backed 6-DOF grasp) |
| `spark_real/skills/manipulation.py` | wiggle, push, pull, drag, stack, open_drawer, screw, etc. |

---

## 1. Environment Setup

### Conda

Always use `spark_conda`. Never `/opt/conda` or a ROS overlay workspace.

```bash
conda activate spark_conda
```

### Required Python packages

```bash
# Core
pip install numpy fastapi uvicorn pyyaml pillow opencv-python-headless

# UR10e control
pip install ur-rtde       # provides rtde_control, rtde_receive, rtde_io

# Perception
pip install pyk4a         # Azure Kinect (requires libk4a system package)
pip install pyrealsense2  # wrist RealSense (optional)

# Planning
pip install google-generativeai

# EquiGraspFlow dependencies
pip install roma omegaconf torch

# Hot-reload (optional, quality of life)
pip install jurigged
```

Verify:

```bash
python -c "import rtde_control, rtde_receive, pyk4a, google.generativeai; print('OK')"
```

---

## 2. UR10e Driver Details

**File:** `spark_real/control/ur10e_driver.py`

The `UR10eDriver` communicates with the UR10e controller over RTDE at 500 Hz
(configurable). It uses three RTDE interfaces:

- `rtde_control.RTDEControlInterface` -- joint moves (`moveJ`), linear moves
  (`moveL`), servo mode (`servoJ`), URScript execution
- `rtde_receive.RTDEReceiveInterface` -- joint/TCP state, force/torque,
  robot mode
- `rtde_io.RTDEIOInterface` -- digital/analog I/O

### Robotiq 2F-85 gripper

The gripper is controlled via URScript functions injected through
`sendCustomScriptFunction()`. The driver loads Robotiq URCap function
definitions from the vendored `grippy.script` at
`spark_real/robots/ur10e/grippy.script` (override with the
`SPARK_GRIPPER_SCRIPT` env var). If not found, a fallback using raw
`rq_*` calls is used.

Gripper position scale: **0 = fully open, 255 = fully closed** (raw Robotiq
scale). The `set_gripper_position()` method accepts 0.0-1.0 (normalised)
and converts internally.

### Motion parameters (defaults)

| Parameter | Value | Unit |
|-----------|-------|------|
| Linear velocity | 0.25 | m/s |
| Linear acceleration | 0.5 | m/s^2 |
| Joint velocity | 1.05 | rad/s |
| Joint acceleration | 1.4 | rad/s^2 |
| RTDE frequency | 500 | Hz |

### Workspace bounds (base frame)

From `configs/ur10e_default.yaml`:

```
x: [-0.9, 0.9]   y: [-0.9, 0.9]   z: [-0.020, 1.2]
```

The driver's `_check_workspace()` enforces these before every `moveL`.

---

## 3. Configuration

**File:** `spark_real/configs/ur10e_default.yaml`

Key fields:

```yaml
robot:
  ip: "192.168.56.101"   # UR10e controller IP -- verify with ping
  frequency: 500         # RTDE Hz

gripper:
  type: "robotiq_2f85"
  speed: 255
  force: 50

workspace:              # safety bounds in robot base frame (meters)
  x_min: -0.9
  x_max: 0.9
  y_min: -0.9
  y_max: 0.9
  z_min: -0.020
  z_max: 1.2
  eta_kinematic: 5.0

table_z_floor: -0.005   # hard descent stop = table_z_floor + 2mm

workspace_extras:
  eps_singularity: 0.15  # CBF elbow barrier (UR10e joint 2)

kinect_use_hw_sync: false
```

The `--ip` CLI flag overrides `robot.ip`. The `--robot ur10e` flag
selects this config file.

---

## 4. Starting the Server

### Manual start (foreground, for development)

```bash
cd ~/spark/src
conda activate spark_conda
python -m spark_real.server --robot ur10e --port 8888
```

If Kinects need `DISPLAY`/`XAUTHORITY` for the depth MCU's OpenGL context:

```bash
DISPLAY=:1 XAUTHORITY=/run/user/1000/gdm/Xauthority \
  python -m spark_real.server --robot ur10e --port 8888
```

### Lifecycle script (background, for production)

```bash
~/spark/scripts/spark_server.sh start    # background, logs to output/logs/
~/spark/scripts/spark_server.sh stop     # graceful: /api/shutdown -> SIGTERM -> SIGINT
~/spark/scripts/spark_server.sh restart
~/spark/scripts/spark_server.sh status   # PID + /api/status JSON
```

### CLI flags

| Flag | Default | Description |
|------|---------|-------------|
| `--robot` | `ur10e` | Robot family (`ur10e`, `franka`, `g1`, `bimanual_franka`) |
| `--port` | `8888` | HTTP port |
| `--ip` | from YAML | Override robot controller IP |
| `--no-robot` | off | Perception-only mode (no RTDE connection) |
| `--no-kinect` | off | Skip Kinect init (for fault recovery or headless dev) |
| `--no-init` | off | Skip all hardware init (serve API skeleton only) |
| `--strict-verify` | off | Stricter placement containment check (>= 50% mask overlap) |
| `--auto-unlock` | off | Franka only; ignored for UR10e |

### What happens at startup

1. JAX GPU allocation is capped (15%) to avoid starving SAM3/PyTorch
2. jurigged hot-reload watcher starts (disable with `SPARK_HOT_RELOAD=0`)
3. FastAPI app registers 13 route modules
4. Signal handlers registered (SIGTERM, SIGINT, SIGHUP) + atexit hook
5. `PipelineConfig` is built from `configs/ur10e_default.yaml` + CLI flags
6. `SPARKRealPipeline.initialize()` runs:
   - Azure Kinect(s) opened (pyk4a)
   - RealSense auto-detected (if plugged in)
   - Hand-eye calibrations loaded from `output/calibrations/`
   - SAM3 perception loaded (GPU)
   - Gemini planner initialised
   - EquiGraspFlow checkpoint loaded (GPU, ~30-40s)
   - FlexiTac auto-detected (optional)
   - UR10e connected via RTDE, wrapped in SafeRobot + ScoreExecutor
7. Uvicorn serves on `0.0.0.0:<port>`

### Readiness check

```bash
curl -s http://localhost:8888/api/status | python3 -m json.tool
```

All three must be `true` before running tasks:

- `robot_connected`
- `kinect_connected` (birdview)
- `kinect2_connected` (sideview, if applicable)

---

## 5. How UR10e Differs from Franka

| Aspect | UR10e | Franka FR3 |
|--------|-------|------------|
| Driver | `UR10eDriver` via `ur-rtde` | `FrankaDriver` via `franky` / `pylibfranka` |
| Protocol | RTDE (500 Hz) | FCI (1 kHz) |
| Gripper | Robotiq 2F-85 (0-255 scale, URScript) | Franka Hand (0-0.08m width, native) |
| Servo mode | `servoJ` via RTDE | CartesianServo velocity-based (avoids FCI reflex) |
| Fast moves | URScript `movej`/`movel` | franky motion generators |
| Singularity barrier | Enabled (`eps_singularity: 0.15`, elbow at J2) | Disabled (different kinematics) |
| SafeRobot CBF | Enabled, wraps driver | Disabled (CBF interferes with direct joint moves) |
| IK backend | N/A (UR10e uses native `moveL`/`moveJ`) | Pinocchio DLS or pyroki (JAX) |
| Auto-unlock | N/A (pendant Remote mode) | Desk RobotWebSession (`--auto-unlock`) |

### Factory routing

`robots/factory.py` contains `make_robot_driver()`. When `--robot ur10e`:

```python
from spark_real.control.ur10e_driver import UR10eDriver
return UR10eDriver(robot_ip, frequency=500.0)
```

For the UR10e, `pipeline.py` has a **legacy path** that also supports
the older `UR10eRobot` class from the external teleop deploy tree (if
available on `sys.path`). Otherwise it falls through to the factory.

---

## 6. Registered Skills

All skills are available regardless of robot family. The ScoreExecutor
dispatches by skill name from the BT YAML.

### Core skills (`skills/primitives.py`)

`move_to_keypoint`, `grasp`, `release`, `move_relative`, `wait`,
`tilt_wrist`, `jog_joint`, `pour`, `sweep`, `constrained_scrub`,
`place_in_slot`

### Grasping (`skills/grasping.py`)

`grasp_se3` -- EquiGraspFlow 6-DOF grasp synthesis. Accepts
`strategy: "top_down"` (OBB yaw, skips EquiGraspFlow) or
`strategy: "se3"` (full SE(3) optimisation). Top-down is faster
for flat objects like silverware.

### Manipulation (`skills/manipulation.py`)

`wiggle`, `push_object`, `open_drawer`, `screw`,
`pour_legacy_urscript`, `pull`, `push`, `drag`, `stack`

### Tool use (`skills/tool_use.py`)

`pour_to_level`, `sweep_to_container`, `tool_grip_pose`,
`constrained_scrub_oscillating`, `rinse`, `sponge_wash`, `throw_to`,
`compliant_insert`, `pen_write`

---

## 7. REST API

The server exposes all endpoints under `/api/`. Key ones:

| Endpoint | Method | Purpose |
|----------|--------|---------|
| `/api/status` | GET | Pipeline + hardware status |
| `/api/robot` | GET | Robot state (joints, TCP, gripper) |
| `/api/capture` | GET | Camera frame (birdview / sideview / wrist) |
| `/api/detect` | POST | SAM3 open-vocabulary detection |
| `/api/detect_click` | POST | Point-prompted detection |
| `/api/detect_box` | POST | Box-prompted detection |
| `/api/detect_approve` | POST | Detect + approve for execution |
| `/api/plan` | POST | Gemini BT generation |
| `/api/execute` | POST | Full pipeline: detect -> plan -> execute |
| `/api/execute_approved` | POST | Execute from approved detections |
| `/api/run_bt` | POST | Execute a raw BT YAML |
| `/api/grasp_candidates` | POST | EquiGraspFlow SE(3) grasp proposals |
| `/api/abort` | POST | E-stop current execution |
| `/api/shutdown` | POST | Graceful server shutdown |
| `/api/progress` | GET | Streaming execution progress log |
| `/api/auto_scene` | POST | Generate MuJoCo XML from live detections |
| `/api/initialize` | POST | Re-initialise pipeline (reconnect hardware) |

Frontend UI served at `http://localhost:8888/` (if `frontend/` is built).

---

## 8. Calibration

Hand-eye calibrations live in `spark_real/output/calibrations/`:

- `handeye_birdview.json` -- birdview Kinect to robot base
- `handeye_sideview.json` -- sideview Kinect to robot base
- `handeye_wrist.json` -- wrist RealSense to TCP (optional)

To recalibrate:

```bash
cd ~/spark/src
# Start server in perception-only mode
python -m spark_real.server --robot ur10e --no-robot --port 8888 &

# Run calibration (requires checkerboard visible to camera + robot powered on)
python scripts/handeye_calibrate.py --robot ur10e --ip <UR10E_IP>
```

Single-arm UR10e: the robot base frame IS the world frame (no separate
world frame transform needed).

### Depth scale

Azure Kinect has a systematic ~15% depth bias. The calibration JSON stores
a `depth_scale` correction factor (typical values: 0.84-0.86). This is
applied automatically during 3D lifting.

---

## 9. Perception Pipeline

1. **Camera capture** -- Azure Kinect birdview (1080P colour + WFOV_2X2BINNED
   depth at 30 fps). Optional sideview and wrist RealSense.
2. **SAM3 detection** -- open-vocabulary text prompts, point prompts, or box
   prompts. Returns 2D masks + confidence scores.
3. **3D lifting** -- mask centroid projected through calibrated depth map to
   get `position_3d` in robot base frame. Punch-through Z detection handles
   translucent/concave objects (gated by `obb_minor_m > 0.040m` to exclude
   silverware).
4. **OBB analysis** -- oriented bounding box from mask geometry, used for
   yaw estimation in top-down grasps.

### Troubleshooting perception

If SAM3 returns 0 detections, confidence < 0.2, or 3D positions with Z far
below the table: **perception is broken, not motion**. Check Kinect health
first (see Camera Safety below).

---

## 10. Camera / Kinect Safety

### NEVER do these

- `kill -9` the server -- bricks the depth MCU (needs physical 12V cycle)
- `xhci_hcd unbind/bind` -- damages the colour MCU
- `pkill -9` anything touching pyk4a or libk4a

### Safe shutdown methods

```bash
~/spark/scripts/spark_server.sh stop          # /api/shutdown -> SIGTERM -> SIGINT
curl -sf -X POST http://localhost:8888/api/shutdown
pkill -TERM -f spark_real.server
Ctrl-C in the terminal
```

### uvcvideo blacklist

The `uvcvideo` kernel module races with `libusb` for the Kinect 4K colour
interface (045e:097d). This has caused kernel crashes. Fix:

```bash
sudo tee /etc/modprobe.d/blacklist-uvcvideo.conf << 'EOF'
blacklist uvcvideo
EOF
sudo rmmod uvcvideo 2>/dev/null
sudo update-initramfs -u -k "$(uname -r)"
```

Neither Azure Kinect (libk4a -> libusb) nor RealSense (pyrealsense2 rsusb
backend) needs `uvcvideo`.

### Kinect recovery ladder

```bash
# 1. uhubctl Vbus cycle
sudo scripts/kinect_software_reset.sh

# 2. sysfs authorized toggle
sudo scripts/kinect_authorize_reset.sh

# 3. Probe
python -c "
import pyk4a
for i in range(pyk4a.connected_device_count()):
    k = pyk4a.PyK4A(device_id=i)
    try: k.open(); print(f'  {i}: {k.serial}'); k.close()
    except Exception as e: print(f'  {i}: FAILED')
"

# 4. Physical 12V barrel-jack unplug for ~5s (last resort)
```

---

## 11. Functional Verification

### Perception

Place 2-3 objects on the table:

```bash
curl -s -X POST http://localhost:8888/api/detect \
  -H "Content-Type: application/json" \
  -d '{"prompts": ["mug", "spoon"]}' | python3 -m json.tool
```

Check that `position_3d` has a sensible Z (above table, not negative or
far below).

### Planning

```bash
curl -s -X POST http://localhost:8888/api/plan \
  -H "Content-Type: application/json" \
  -d '{"instruction": "pick up the mug"}' | python3 -m json.tool
```

Should return a YAML BT with `move_to_keypoint` + `grasp`/`grasp_se3`.

### End-to-end execution

```bash
curl -s -X POST http://localhost:8888/api/execute \
  -H "Content-Type: application/json" \
  -d '{"instruction": "pick up the mug and place it on the plate"}' | python3 -m json.tool
```

### Validated tasks

- **Silverware sort:** "Sort the silverware into the tray" -- `grasp_se3`
  (top_down) + `place_in_slot`
- **Stack blocks:** "Stack the red block on the blue block" --
  `move_to_keypoint` + `grasp` + `release`
- **Plushie pickup:** "Pick up the plushie and put it in the box" --
  `grasp` (compressible) + `release`

---

## 12. Simulation Validation

UR10e sim scenes and BT score YAMLs for offline validation:

```
spark_sim/scenes/corl/corl_dustpan_ur10e.xml
spark_real/tests/corl/run_dustpan_ur10e.py
spark_real/tests/corl/scores/dustpan_ur10e.yaml
spark_real/tests/corl/scores/ur10e_pour.yaml
spark_real/tests/corl/scores/ur10e_silverware.yaml
spark_real/tests/corl/scores/ur10e_pens.yaml
spark_real/tests/corl/scores/ur10e_plushie.yaml
spark_real/tests/corl/scores/ur10e_sponge.yaml
```

Run a sim validation:

```bash
cd ~/spark/src
python -m spark_real.sim_pipeline \
  --scene corl_ur10e_silverware.xml \
  --score-yaml tests/corl/scores/ur10e_silverware.yaml \
  --headless --no-sam3
```

### Auto-MJCF from live perception

Generate a MuJoCo scene from current detections for sim-before-real gating:

```bash
curl -X POST http://localhost:8888/api/detect_approve \
  -H 'Content-Type: application/json' \
  -d '{"instruction": "pick up the knife", "prompts": ["knife", "tray"]}'

curl -X POST http://localhost:8888/api/auto_scene \
  -H 'Content-Type: application/json' \
  -d '{"render_preview": true}'
```

---

## 13. Environment Variables

| Variable | Default | Description |
|----------|---------|-------------|
| `SPARK_HOT_RELOAD` | `1` | Set `0` to disable jurigged live-patching |
| `SPARK_VELOCITY_OVERRIDE` | (none) | Cap Cartesian velocity (m/s) |
| `SPARK_EQUIGRASP_DISABLE` | (none) | Set `1` to disable EquiGraspFlow (fall back to simple grasp) |
| `SPARK_DISABLE_LLM` | (none) | Skip Gemini calls, use BT library cache only |
| `SPARK_IK` | (none) | IK backend override (e.g. `pyroki`); Franka only |
| `XLA_PYTHON_CLIENT_PREALLOCATE` | `false` | JAX GPU preallocation (set by server.py) |
| `XLA_PYTHON_CLIENT_MEM_FRACTION` | `0.15` | JAX GPU memory cap (set by server.py) |
| `DISPLAY` / `XAUTHORITY` | (system) | Required for Kinect depth MCU OpenGL context |

---

## 14. Troubleshooting

| Symptom | Cause | Fix |
|---------|-------|-----|
| `robot_connected: false` | Wrong IP, ur-rtde missing, pendant not in Remote mode | `ping <IP>`, `pip install ur-rtde`, switch pendant to Remote |
| `kinect_connected: false` | USB issue, depth MCU bricked | Run Kinect recovery ladder (Section 10) |
| `ModuleNotFoundError: rtde_control` | ur-rtde not installed | `pip install ur-rtde` |
| `ModuleNotFoundError: roma` | EquiGraspFlow dep missing | `pip install roma` |
| `ModuleNotFoundError: omegaconf` | EquiGraspFlow dep missing | `pip install omegaconf` |
| Server exits silently on `--no-robot` | Wrong cwd | Must start from `~/spark/src` |
| 3D positions below table | Bad calibration or Kinect depth MCU fault | Re-calibrate or run Kinect recovery |
| `joint_velocity_violation` reflexes | Wrist near singularity at high speed | Set `SPARK_VELOCITY_OVERRIDE=0.15` |
| Gripper not responding | Robotiq URCap not installed, or `grippy.script` not found | Install URCap on pendant, verify script path |
| Server startup takes >2 min | SAM3 + EquiGraspFlow GPU loading | Normal; subsequent calls are fast |

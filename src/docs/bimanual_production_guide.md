# Bimanual Franka Production Deployment Guide

Host: the bimanual Franka workstation. Branch: `reconcile-jul07` (the integration branch).
Goal: bring up the bimanual Franka rig on the SPARK standalone pipeline,
validate with sim dry-runs, then run real-hardware end-to-end bimanual
tasks.

This guide is self-contained. An agent or operator reading it should be
able to deploy bimanual from scratch.

---

## 0. Controller architecture: why Bamboo, not franky

### The problem with franky motion generators

The SPARK single-arm path currently uses franky (Python bindings around
libfranka motion generators). This is **wrong for production** and will
eventually be replaced for single-arm too. The fundamental issues:

1. **Velocity/acceleration discontinuity reflexes.** franky's motion
   generators (JointMotion, CartesianMotion) plan complete trajectories
   up front. When the SPARK pipeline chains motions (IK solve, then
   send joint target open-loop), the transition between successive
   motions triggers `joint_motion_generator_velocity_discontinuity` and
   `cartesian_motion_generator_acceleration_discontinuity` reflexes.
   The driver in `franka_driver.py` has ~200 lines of workarounds:
   pre-motion JointStopMotion, post-motion velocity polling, automatic
   error recovery, RelativeDynamicsFactor throttling. These are band-aids.

2. **No task-space awareness during joint interpolation.** We solve IK
   to get a joint target, then send it as a JointMotion. The joint
   interpolation has no knowledge of the Cartesian path, so the TCP
   tilts, dips, or oscillates mid-trajectory. This is acceptable for
   slow single-arm pick-place but dangerous for bimanual coordination
   where two arms must converge on a meeting point.

3. **Every serious Franka research lab avoids motion generators.**
   Deoxys (UT Austin RPL), DROID, R2D2 all use impedance or torque
   control at 1 kHz. Bamboo (chsahit/bamboo) is explicitly built on
   this insight, drawing from deoxys_control and drake-franka-driver.

### Bamboo C++ joint impedance controller

Bamboo runs libfranka's torque control loop at 1 kHz with joint
impedance. A ZMQ bridge exposes the robot to Python clients. The
architecture:

```
Python (SPARK server)                  C++ (Bamboo control node)
     |                                      |
     |  ZMQ (tcp://localhost:5555)           |
     +------------------------------------->|
     |  execute_joint_impedance_path(q, dq) |
     |  get_joint_positions()               |--- libfranka torque
     |  get_joint_states()                  |    control at 1 kHz
     |  open_gripper() / close_gripper()    |
     |<-------------------------------------+
     |  joint states, ee_pose, gripper_state|
```

Key properties:
- **No reflexes.** Impedance control tracks the commanded trajectory
  smoothly. No discontinuity detection because there are no
  discontinuities.
- **Built against libfranka 0.18.2** (FR3 compatible). The local build
  lives at `~/spark/src/external_controllers/bamboo/install/lib/`.
- **Collision thresholds: 100 Nm / 100 N** (Deoxys standard). The
  franky default of 20 Nm triggers protective stops on normal motion.
  Bamboo sets these internally in the C++ node.
- **Gripper managed by Bamboo** when using Franka Hand (flag `-g franka`).
  For the bimanual rig with SSG48 grippers, pass `-g none` and manage
  grippers independently from Python.

### panda-py as an alternative

panda-py is another impedance-control option. However:
- The pip wheel bundles libfranka 0.9.2, which is **Panda only**.
- FR3 requires the 0.13.3 wheel from GitHub releases, or a source build.
- Bamboo is preferred because it already works on this host and matches
  the Deoxys/DROID control architecture.

### SafeRobot status

SafeRobot wrapping for Franka was disabled on single-arm because it
interfered with direct moves via franky. Once the controller is fully
impedance-based (Bamboo), SafeRobot should be re-enabled -- impedance
control is inherently more tolerant of the velocity-gating that
SafeRobot applies.

### NO ROS NEEDED

The entire deployment is pure Python (FastAPI server) + C++ (Bamboo
control node). No ROS launch files, no ROS2 middleware. The
`SPARK_ENABLE_ROS` env var should remain unset.

---

## 1. Hardware prerequisites

| Component | Qty | Notes |
|-----------|-----|-------|
| Franka Emika Panda (left arm) | 1 | FCI at `172.16.0.101` |
| Franka Research 3 (right arm) | 1 | FCI at `172.16.0.102` |
| SSG48 grippers | 2 | Independent, managed from Python |
| ZED Mini stereoscopic camera | 1 | Center tripod (external / primary view) |
| CyberPower 1500VA UPS | 1 | Protects Franka controllers |
| Multi-port PCIe NIC or dual NICs | 1 | Dedicated ports for dual FCI |
| ChArUco board (DICT_4X4_100, 5x7, 30 mm squares) | 1 | For anchor-tap calibration |

### Bimanual vs single-arm hardware differences

- **Grippers:** Bimanual uses SSG48 grippers, not the Franka Hand or
  Dynamixel ALOHA jaws. SSG48s are physically connected to the bimanual
  host and managed entirely in Python
  (`spark_real/robots/bimanual_franka/ssg48_gripper.py`), NOT by
  Bamboo. Pass `-g none` to the Bamboo control node.
- **Cameras:** Bimanual does NOT use wrist RealSense cameras. It does
  NOT use Azure Kinects. Perception comes from the ZED Mini on the
  center tripod.
- **uvcvideo unbind udev rule:** NOT needed for bimanual. That rule
  (`/etc/udev/rules.d/99-k4a-unbind-uvcvideo.rules`) exists to prevent
  `uvcvideo` from racing with `libusb` for the Kinect 4K color
  interface. Bimanual does not use Kinects.

### Verify physical connections

```bash
# Both Frankas reachable
ping -c1 -W2 172.16.0.101 && echo "Panda (left) OK"
ping -c1 -W2 172.16.0.102 && echo "FR3 (right) OK"

# ZED Mini on USB 3.0
lsusb | grep "2b03:" && echo "ZED Mini OK"

# SSG48 grippers (verify connectivity per your gripper interface)
ls /dev/ttyUSB* 2>/dev/null
```

---

## 2. Build prerequisites (Bamboo)

### 2a. System packages

```bash
sudo apt install libzmq3-dev libmsgpack-dev libpoco-dev build-essential cmake
```

### 2b. Pinocchio (required for libfranka >= 0.14)

libfranka 0.14+ depends on Pinocchio for its internal dynamics model.
Install before building Bamboo:

```bash
# Via conda (recommended)
mamba install -c conda-forge pinocchio

# OR via apt (if not using conda for C++ deps)
sudo apt install robotpkg-pinocchio
```

Pinocchio headers must be findable by CMake. The conda install puts them
in `$CONDA_PREFIX/include/pinocchio/`.

### 2c. Build libfranka 0.18.2 + Bamboo

Bamboo bundles its own libfranka build (does NOT override system installs):

```bash
cd ~/spark/src/external_controllers/bamboo
bash InstallBambooController
```

The script prompts for:
- libfranka version (enter `0.18.2` for FR3 compatibility)
- sudo password (for adding user to `realtime`, `dialout`, `tty` groups)
- package installation

If groups are added, **log out and log back in** before running the
controller.

Verify the build:

```bash
ls ~/spark/src/external_controllers/bamboo/controller/build/bamboo_control_node
# Should exist and be executable

ls ~/spark/src/external_controllers/bamboo/install/lib/libfranka.so.0.18.2
# Should exist
```

### 2d. Install Bamboo Python client

```bash
conda activate spark_conda
pip install bamboo-franka-client
```

Or from the local checkout:

```bash
cd ~/spark/src/external_controllers/bamboo
pip install -e .
```

### 2e. Verify libfranka version compatibility

Check the FCI version in Franka Desk (Settings > Dashboard > Control)
and consult the
[FCI Compatibility Table](https://frankarobotics.github.io/docs/compatibility.html).
FR3 requires libfranka >= 0.13; the build on this host uses 0.18.2.

**Important:** The Panda (left arm) and FR3 (right arm) have different
FCI versions. libfranka 0.18.2 is backward compatible with both, but
verify that both arms' firmware is updated to a compatible version.

---

## 3. Environment setup

### 3a. Conda environment

```bash
conda activate spark_conda
python --version   # need 3.10+
```

### 3b. Install bimanual-specific dependencies

```bash
# Core deps (if not already present)
pip install roma omegaconf pyzmq

# ZED Mini SDK Python bindings
# (requires ZED SDK installed system-wide first: https://www.stereolabs.com/developers/release)
pip install pyzed

# Bamboo client (if not done in 2d)
pip install bamboo-franka-client

```

### 3c. Verify all deps

```bash
cd ~/spark/src
python -c "
import numpy, fastapi, pyzmq, roma, omegaconf
print('core deps OK')
try:
    import bamboo; print('bamboo client OK')
except ImportError: print('WARNING: bamboo-franka-client missing')
try:
    import pyzed.sl; print('pyzed OK')
except ImportError: print('WARNING: pyzed missing (ZED Mini will sim)')
"
```

### 3d. NIC configuration for dual FCI

Each Franka FCI session requires a dedicated Ethernet link at 1 kHz.
Both arms must have independent paths to avoid jitter.

```bash
# Check available NICs
ip addr show | grep -E "^[0-9]+:" | awk '{print $2}'

# Typical setup: two ports on a PCIe NIC
# Port 1 -> Panda (left):  172.16.0.1/24
# Port 2 -> FR3 (right):   172.16.0.1/24 on a separate subnet, or
#                           both on 172.16.0.0/24 if using a single subnet

# Verify route to each arm
ip route get 172.16.0.101
ip route get 172.16.0.102
```

**Important:** Direct connections are used (no switch) to avoid
switch-induced jitter. If using a managed switch, configure jumbo
frames and QoS to prioritize FCI traffic.

### 3e. Franka Desk credentials

Each arm may use a different Desk account. Create credential files:

```bash
# Left arm (Panda)
cat > ~/.franka_desk_left.json << 'EOF'
{"hostname": "172.16.0.101", "username": "admin", "password": "YOUR_PASSWORD"}
EOF

# Right arm (FR3)
cat > ~/.franka_desk_right.json << 'EOF'
{"hostname": "172.16.0.102", "username": "admin", "password": "YOUR_PASSWORD"}
EOF

chmod 600 ~/.franka_desk_left.json ~/.franka_desk_right.json
```

---

## 4. Startup sequence

The bimanual stack has three independent processes. Start them in this
order.

### 4a. Start Bamboo control nodes (one per arm)

Each arm needs its own Bamboo C++ process. They run independently and
can be on the same or different machines (as long as the ZMQ ports are
reachable).

**Left arm (Panda) on port 5555:**

```bash
LD_LIBRARY_PATH="/opt/openrobots/lib:$HOME/spark/src/external_controllers/bamboo/install/lib" \
  ~/spark/src/external_controllers/bamboo/controller/build/bamboo_control_node \
  -r 172.16.0.101 -p 5555 -g none &
```

**Right arm (FR3) on port 5556:**

```bash
LD_LIBRARY_PATH="/opt/openrobots/lib:$HOME/spark/src/external_controllers/bamboo/install/lib" \
  ~/spark/src/external_controllers/bamboo/controller/build/bamboo_control_node \
  -r 172.16.0.102 -p 5556 -g none &
```

Key flags:
- `-r`: Robot IP address (required)
- `-p`: ZMQ port (required, must be unique per arm)
- `-g none`: Do NOT manage grippers from C++ (SSG48s stay in Python)
- `-l`: Listen address (default `*` = all interfaces)
- `-m`: Use min-jerk interpolation (default: linear)

**Why `-g none`:** The SSG48 grippers are independent devices physically
connected to the bimanual host. They are not Franka Hands and they are
not Robotiq grippers. Bamboo's built-in gripper management (which
expects one of those two) does not apply.

Wait for both nodes to print `Server listening on port XXXX` before
starting the SPARK server.

**Using the RunBambooController wrapper** (alternative, manages tmux):

```bash
cd ~/spark/src/external_controllers/bamboo
# Left arm
bash RunBambooController start --robot_ip 172.16.0.101 --control_port 5555 \
  --gripper_type none --conda_env spark_conda

# Right arm (separate terminal or tmux pane)
bash RunBambooController start --robot_ip 172.16.0.102 --control_port 5556 \
  --gripper_type none --conda_env spark_conda
```

### 4b. Start the SPARK server

```bash
cd ~/spark/src
SPARK_FRANKA_BACKEND=bamboo \
  python -m spark_real.server --robot bimanual_franka --port 8888
```

The `SPARK_FRANKA_BACKEND=bamboo` env var tells `robots/factory.py` to
instantiate `FrankaBambooDriver` instead of `FrankaDriver` (franky) for
each arm. The `--robot bimanual_franka` flag triggers:

1. `robots/factory.py` dispatches to `BimanualFrankaDriver`.
2. `BimanualFrankaDriver` creates two `FrankaBambooDriver` instances
   (port 5555 for left, 5556 for right).
3. Each `FrankaBambooDriver.connect()` starts/connects to its Bamboo
   C++ node via ZMQ.
4. SSG48 grippers connect independently after FCI.
5. Camera registry builds from the YAML: external (ZED Mini).
6. ZMQ state broadcaster starts on `tcp://*:5601` at 30 Hz.
7. Bimanual routes mount at `/api/bimanual/*`.

**Note:** Because bimanual does not use Kinects, you do NOT need the
`DISPLAY` and `XAUTHORITY` env vars that the single-arm server requires
(those are for Kinect depth which uses OpenGL). If the ZED Mini also
needs a display context, add them; otherwise they are unnecessary.

### 4c. FCI connection architecture

Each arm is a separate FCI connection. The two connections cannot be
held by the same process in libfranka's default mode. Bamboo solves
this by running each arm in its own C++ process. The
`BimanualFrankaDriver` aggregates them from the Python side.

If you need both arms in a single process (e.g., for tighter
synchronization), use the `BimanualFrankaDriver` which handles the
threading internally. But for the Bamboo backend, two separate Bamboo
processes is the correct architecture.

### 4d. Verify startup

After 10-15 seconds:

```bash
curl -s http://localhost:8888/api/status | python -m json.tool
```

Check for:
- `robot_connected: true`
- `robot_family: "bimanual_franka"`

---

## 5. Graceful shutdown

### 5a. NEVER do these things

- **`kill -9` the SPARK server** -- SIGKILL prevents the atexit hook
  from releasing camera handles and ZMQ sockets. On the single-arm rig
  this bricks the Kinect depth MCU (requires 12V barrel unplug).
  On bimanual it leaves the ZED Mini handle leaked and the Bamboo
  control nodes orphaned.
- **`xhci_hcd unbind/bind`** -- damages camera MCUs. Only relevant if
  Kinects are also attached to this host.
- **`pkill -9` anything touching pyk4a, pyzed, or pyrealsense2.**

### 5b. Safe ways to stop

```bash
# Preferred: lifecycle script (see below)
~/spark/scripts/spark_bimanual.sh stop

# HTTP graceful shutdown
curl -sf -X POST http://localhost:8888/api/shutdown

# SIGTERM (signal handler runs atexit)
pkill -TERM -f spark_real.server

# Ctrl-C in the terminal (SIGINT, atexit runs)
```

### 5c. Stopping Bamboo control nodes

After the SPARK server is stopped, stop the Bamboo nodes:

```bash
# If started manually with &
kill %1 %2   # or by PID

# If started with RunBambooController
cd ~/spark/src/external_controllers/bamboo
bash RunBambooController stop
```

Bamboo control nodes handle SIGTERM gracefully -- they release the FCI
connection and exit cleanly.

### 5d. Background server via lifecycle script

```bash
mkdir -p ~/spark/scripts
cat > ~/spark/scripts/spark_bimanual.sh << 'SHEOF'
#!/bin/bash
set -u
PORT="${SPARK_PORT:-8888}"
PATTERN='python -m spark_real\.server'
SRC_DIR="$HOME/spark/src"
LOG_DIR="$HOME/spark/src/spark_real/output/logs"
LOG_FILE="$LOG_DIR/spark_bimanual.log"
if [ -n "${CONDA_PREFIX:-}" ]; then PY="$CONDA_PREFIX/bin/python"
elif [ -x "$HOME/spark/venv/bin/python" ]; then PY="$HOME/spark/venv/bin/python"
else PY="$(which python3)"; fi
cmd="${1:-status}"
is_running() { pgrep -f "$PATTERN" >/dev/null 2>&1; }
pids() { pgrep -f "$PATTERN" | tr '\n' ' '; }
wait_dead() { local t=$1 e=0; while is_running; do sleep 1; e=$((e+1)); [ "$e" -ge "$t" ] && return 1; done; }
case "$cmd" in
start)
    is_running && { echo "Already running ($(pids))" >&2; exit 1; }
    mkdir -p "$LOG_DIR"; shift
    echo "Starting SPARK (Bimanual Franka) on :$PORT ..."
    cd "$SRC_DIR"
    SPARK_FRANKA_BACKEND=bamboo \
    nohup "$PY" -m spark_real.server --robot bimanual_franka --port "$PORT" "$@" >> "$LOG_FILE" 2>&1 &
    sleep 4; is_running && echo "OK ($(pids)). Log: $LOG_FILE" || { echo "FAILED"; tail -30 "$LOG_FILE"; exit 1; } ;;
stop)
    is_running || { echo "Not running."; exit 0; }
    echo "Stopping ($(pids)) ..."
    curl -sf -X POST "http://localhost:$PORT/api/shutdown" >/dev/null 2>&1
    wait_dead 8 && { echo "Stopped (shutdown)."; exit 0; }
    pkill -TERM -f "$PATTERN"; wait_dead 6 && { echo "Stopped (TERM)."; exit 0; }
    pkill -INT -f "$PATTERN"; wait_dead 4 && { echo "Stopped (INT)."; exit 0; }
    [ "${2:-}" = "--force" ] && { pkill -9 -f "$PATTERN"; echo "KILLED (cameras may need power cycle)."; } \
      || { echo "Use 'stop --force' for SIGKILL." >&2; exit 1; } ;;
restart) "$0" stop; sleep 2; "$0" start "${@:2}" ;;
status) is_running && { echo "Running ($(pids))"; curl -sf "http://localhost:$PORT/api/status" | python3 -m json.tool 2>/dev/null; } || echo "Not running." ;;
*) echo "Usage: $0 {start|stop|restart|status}" >&2; exit 1 ;;
esac
SHEOF
chmod +x ~/spark/scripts/spark_bimanual.sh
```

---

## 6. Configuration

### 6a. Verify per-arm IPs in the YAML

```bash
grep -A2 "left_ip\|right_ip" ~/spark/src/spark_real/configs/bimanual_franka_default.yaml
```

Expected:

```
left_ip:  "172.16.0.101"   # Franka Emika Panda
right_ip: "172.16.0.102"   # Franka Research 3
```

### 6b. Bamboo port configuration

The bimanual config YAML should include the Bamboo ports:

```yaml
bamboo:
  left_port: 5555
  right_port: 5556
```

These must match the `-p` flags passed to the Bamboo control nodes.

### 6c. SSG48 gripper configuration

SSG48 grippers are managed in Python via
`robots/bimanual_franka/ssg48_gripper.py`. Key config:

```yaml
grippers:
  type: "ssg48"
  # SSG48s connect via their own interface, not through the Bamboo node
  # or the Franka Hand interface. Configure the device path here.
```

### 6d. Inter-arm safety parameters

The default config ships with conservative CBF values:

| Parameter | Value | Meaning |
|-----------|-------|---------|
| `dist_hard_m` | 0.12 | Motions bringing TCPs closer than 12 cm are **rejected** |
| `dist_soft_m` | 0.20 | Between 12-20 cm, velocity is linearly scaled down |
| `eta_kinematic` | 8.0 | CBF barrier gain |

These are in `configs/bimanual_franka_default.yaml` under `inter_arm:`.
Do not relax `dist_hard_m` below 0.10 without extensive testing.

---

## 7. Calibration

The bimanual calibration is anchor-tap against ChArUco corners measured by
ZED stereo. Full procedure and output schema: `src/docs/bimanual_calibration.md`.
Both producers live under `spark_real.calibration`.

### 7a. Measure the board (server stopped, ZED is exclusive)

```bash
cd ~/spark/src
pkill -f spark_real.server
python -m spark_real.calibration.bimanual_charuco_stereo
```

Writes `~/.spark_real/ba_frames/zed_charuco_stereo.json`.

### 7b. Tap the corners (server running)

```bash
python -m spark_real.server --robot bimanual_franka --machine ANON-LAB --auto-unlock --port 8888
python -m spark_real.calibration.bimanual_anchor_tap --corners 0,3,20,23,11 --grip-offset 0.1275
```

Per arm, pilot-drive the closed jaw tip onto the named corner and press
ENTER. `--grip-offset` must equal `grippers.jaw_offset_m` in the machine
overlay. Use `--positions N --reuse-existing` to add board positions; in
that mode the script pulls stereo frames from `GET /api/capture/stereo`.

**Success criteria:** mean residual < 5 mm per arm, max < 10 mm. The
script warns above 10 mm mean.

### 7c. Verify calibration output

```bash
python -m json.tool ~/.spark_real/calibration_bimanual.json | head -40
python -m json.tool ~/.spark_real/T_right_to_left.json
```

`calibration_bimanual.json` holds `arms.<arm>.T_zed_to_base`, residuals
and the taps; `T_right_to_left.json` is read by `BimanualPyrokiPlanner`.
Both are required: `skills/bimanual_cloth.py` and `src/scripts/detect.py`
raise when the file is missing.

### 7d. When to recalibrate

- After moving either arm's base bolts or the ZED tripod.
- After changing the jaws (re-measure `jaw_offset_m` first).
- If the same detected point lands more than 5 mm apart in the two arm frames.

---

## 8. File deployment

All bimanual files live on `reconcile-jul07`. Verify the
critical file set exists:

```bash
cd ~/spark/src/spark_real

# Robot driver
ls robots/bimanual_franka/__init__.py \
   robots/bimanual_franka/bimanual_franka_driver.py

# Bamboo driver (shared with single-arm)
ls robots/franka/franka_bamboo_driver.py

# Control layer
ls control/bimanual_executor.py \
   control/bimanual_safe_robot.py \
   control/bimanual_servo.py

# Skills
ls skills/bimanual.py

# Routes
ls routes/bimanual.py

# ZMQ comms
ls comms/bimanual_zmq.py

# Perception
ls perception/bimanual_camera_registry.py

# Calibration
ls calibration/bimanual_anchor_tap.py calibration/bimanual_charuco_stereo.py

# Config
ls configs/bimanual_franka_default.yaml

# Factory (must include bimanual_franka in KNOWN_FAMILIES)
grep "bimanual_franka" robots/factory.py
```

Expected output from the last grep:
`KNOWN_FAMILIES = ("ur10e", "franka", "g1", "bimanual_franka")`

---

## 9. Smoke tests

### 9a. Bamboo connectivity test (no SPARK server)

Test the Bamboo client directly before starting the full server:

```bash
cd ~/spark/src
python -c "
from bamboo.client import BambooFrankaClient

# Left arm
left = BambooFrankaClient(control_port=5555, server_ip='localhost')
q_left = left.get_joint_positions()
print(f'Left arm joints: {[round(x,3) for x in q_left]}')
left.close()

# Right arm
right = BambooFrankaClient(control_port=5556, server_ip='localhost')
q_right = right.get_joint_positions()
print(f'Right arm joints: {[round(x,3) for x in q_right]}')
right.close()
print('Both arms connected via Bamboo OK')
"
```

### 9b. Import check

```bash
cd ~/spark/src
python -c "
from spark_real.skills import registry
names = sorted(registry._skills)
print(f'Skills: {len(names)}')
bimanual = [n for n in names if n in (
    'pick_with_arm', 'place_with_arm', 'move_to_keypoint_arm',
    'grasp_arm', 'release_arm', 'handoff', 'bimanual_lift',
    'bimanual_place', 'hold_in_place', 'bimanual_handover_sponge')]
print(f'Bimanual skills: {len(bimanual)}')
for n in bimanual: print(f'  {n}')
"
```

Expected: 10 bimanual skills registered.

### 9c. Driver import check

```bash
cd ~/spark/src
python -c "
from spark_real.robots.bimanual_franka import BimanualFrankaDriver, ARMS
print(f'ARMS: {ARMS}')
print(f'DOF_TOTAL: {BimanualFrankaDriver.DOF_TOTAL}')
print(f'GRIPPER_TYPE: {BimanualFrankaDriver.GRIPPER_TYPE}')
"
```

Expected: `ARMS: ('left', 'right')`, `DOF_TOTAL: 14`.

### 9d. Bamboo driver check

```bash
cd ~/spark/src
python -c "
from spark_real.robots.franka.franka_bamboo_driver import FrankaBambooDriver
print(f'SUPPORTS_VELOCITY_STREAMING: {FrankaBambooDriver.SUPPORTS_VELOCITY_STREAMING}')
# Should be False -- Bamboo uses joint impedance paths, not velocity streaming
"
```

### 9e. REST API endpoint smoke test

```bash
# Bimanual state
curl -s http://localhost:8888/api/bimanual/state | python -m json.tool

# Inter-arm distance
curl -s http://localhost:8888/api/bimanual/inter_arm_distance | python -m json.tool

# Per-arm home
curl -s -X POST http://localhost:8888/api/bimanual/left/home
curl -s -X POST http://localhost:8888/api/bimanual/right/home

# Coordinated dual home
curl -s -X POST http://localhost:8888/api/bimanual/home_both

# Per-arm gripper (SSG48)
curl -s -X POST http://localhost:8888/api/bimanual/left/gripper \
  -H "Content-Type: application/json" \
  -d '{"action": "open"}'

curl -s -X POST http://localhost:8888/api/bimanual/right/gripper \
  -H "Content-Type: application/json" \
  -d '{"action": "close", "force": 10.0}'

# Primitive library
curl -s http://localhost:8888/api/bimanual/primitives | python -m json.tool
```

Common failures:

| Symptom | Cause | Fix |
|---------|-------|-----|
| `ImportError: bamboo` | Bamboo client not installed | `pip install bamboo-franka-client` |
| `ConnectionRefusedError` on port 5555/5556 | Bamboo control nodes not running | Start them per section 4a |
| `"active robot is not the bimanual family"` | Wrong `--robot` flag | Restart with `--robot bimanual_franka` |
| `"robot not connected"` | FCI sessions failed | Check arm IPs, Desk credentials, pendant in Remote |
| Bamboo exits immediately | libfranka version mismatch | Rebuild with correct libfranka version |
| Bamboo exits with `realtime` error | Not in `realtime` group | `sudo usermod -aG realtime $USER`, re-login |

---

## 10. Bimanual primitives reference

### 10a. Per-arm primitives (arm-tagged versions of single-arm skills)

| Primitive | Description | Key params |
|-----------|-------------|------------|
| `pick_with_arm` | 6-DOF grasp with specified arm | `arm`, `keypoint_label`, `target_width`, `force` |
| `place_with_arm` | Release held object with specified arm | `arm`, `tilt_angle` |
| `move_to_keypoint_arm` | Move EE to a labeled keypoint | `arm`, `keypoint_label`, `offset_x/y/z` |
| `grasp_arm` | Close specified arm's gripper | `arm`, `force`, `target_width` |
| `release_arm` | Open specified arm's gripper | `arm`, `tilt_angle` |

### 10b. Bimanual-only primitives

| Primitive | Description | Key params |
|-----------|-------------|------------|
| `handoff` | Pass object from one arm to the other | `from_arm`, `to_arm`, `keypoint_label`, `meeting_point`, `grasp_width`, `force` |
| `bimanual_lift` | Both arms grasp opposite sides and lift together | `keypoint_label`, `object_width`, `grip_width`, `lift_height`, `force` |
| `bimanual_place` | Both arms lower a co-held object and release | `target_label`, `place_offset_z`, `release_dwell` |
| `hold_in_place` | One arm holds its TCP pose while the other works | `arm`, `dwell` |
| `bimanual_handover_sponge` | Compound: left picks sponge by width, hands off to right with length-grip for cylindrical scrubbing | `sponge_label`, `width_grip`, `length_grip`, `force`, `meeting_point` |

### 10c. Handoff sequence (internal)

The `handoff` primitive executes this sequence:

1. Both arms move in parallel to pre-meeting offsets (8 cm to each side
   of `meeting_point`).
2. Both arms converge to the meeting point (parallel servo).
3. `to_arm` closes its gripper on the object.
4. 0.3s settle delay.
5. `from_arm` opens its gripper (object is never unsupported).
6. Both arms retract upward 8 cm (parallel).

---

## 11. BT grammar: parallel nodes, sync barriers, arm tags

### 11a. Arm tags

Every arm-taking BT leaf in bimanual mode **must** carry an
`arm: "left"` or `arm: "right"` field in its `params`. The executor
defaults missing `arm` fields to `"right"` with a warning.

```yaml
- type: pick_with_arm
  params: {arm: left, keypoint_label: sponge, target_width: 0.04}
```

### 11b. Parallel node

The `parallel` structural node runs its children on independent arm
threads. The implicit join is a sync barrier -- all children must
complete before the parent `sequence` continues.

```yaml
- type: parallel
  children:
    - type: hold_in_place
      params: {arm: left, dwell: 6.0}
    - type: constrained_scrub
      params: {arm: right, workpiece_label: bowl, duration: 5.0}
```

### 11c. Explicit sync barrier

For mid-parallel rendezvous, use `sync_barrier` with a matching `name`
across both branches:

```yaml
- type: parallel
  children:
    - type: sequence
      children:
        - type: move_to_keypoint_arm
          params: {arm: left, keypoint_label: tray}
        - type: sync_barrier
          params: {name: "rendezvous_1"}
        - type: grasp_arm
          params: {arm: left, force: 15}
    - type: sequence
      children:
        - type: move_to_keypoint_arm
          params: {arm: right, keypoint_label: tray}
        - type: sync_barrier
          params: {name: "rendezvous_1"}
        - type: grasp_arm
          params: {arm: right, force: 15}
```

**Warning:** Every barrier name must be reached by the same number of
branches, otherwise the non-matching branches will hang. The executor
logs a warning when it detects mismatched barrier counts.

### 11d. Full bimanual BT example

```yaml
task: hand the sponge from left to right and scrub the bowl
tree:
  type: sequence
  children:
    - type: pick_with_arm
      params: {arm: left, keypoint_label: sponge, target_width: 0.04}
    - type: handoff
      params: {from_arm: left, to_arm: right, keypoint_label: sponge,
               meeting_point: [0.0, 0.0, 0.30], grasp_width: 0.030, force: 15}
    - type: parallel
      children:
        - type: hold_in_place
          params: {arm: left, dwell: 6.0}
        - type: constrained_scrub
          params: {arm: right, workpiece_label: bowl, duration: 5.0}
    - type: place_with_arm
      params: {arm: right, target_label: sink}
```

### 11e. Planner prompt injection

The bimanual prompt (`planning/bimanual_prompt.md`) is injected into the
planner only when `pipeline.config.robot_family == "bimanual_franka"`.
Key rules it enforces:

- Use arm-tagged primitives instead of single-arm versions.
- Default to the right arm (FR3) for single-object tasks.
- Plan meeting points at least 18 cm above the table for handoffs.
- The inter-arm CBF rejects motions closer than 12 cm between TCPs.

---

## 12. Inter-arm CBF safety

The bimanual safety system is layered:

### 12a. Per-arm safety (delegated to single-arm SafeRobot)

Each arm has its own workspace cuboid, reach sphere, joint limits, and
force barrier -- all handled by the existing `SafeRobot` class.

| Parameter | Left arm | Right arm |
|-----------|----------|-----------|
| X range | -0.20 to 0.85 | -0.20 to 0.85 |
| Y range | -0.05 to 0.85 | -0.85 to 0.05 |
| Z range | 0.00 to 1.10 | 0.00 to 1.10 |
| Max reach | 1.15 m | 1.15 m |

The two boxes overlap only in the central handoff column
(|y| < 0.18, z > 0.20) where the inter-arm CBF is the primary
constraint.

### 12b. Inter-arm distance barrier (BimanualSafeRobot)

| Zone | TCP-to-TCP distance | Behavior |
|------|---------------------|----------|
| Safe | > 20 cm | Full commanded velocity |
| Soft barrier | 12-20 cm | Velocity linearly scaled: `scale = (d - 0.12) / (0.20 - 0.12)` |
| Hard floor | < 12 cm | Motion **rejected**; controller holds position |

### 12c. Velocity gating (BimanualCartesianServo)

The `BimanualCartesianServo` wraps two per-arm servo instances. On
every `servo_to()` call:

1. `BimanualSafeRobot.is_motion_safe(arm, target_xyz)` -- reject if
   target would violate hard floor.
2. `BimanualSafeRobot.velocity_scale_for(arm, target_xyz)` -- compute
   the [0, 1] multiplier.
3. Per-arm servo `max_vel_linear` is temporarily scaled down.
4. After the servo completes, the original max is restored.

For parallel bimanual motion (`servo_to_parallel`), both arms are
servoed from independent threads -- each applies the CBF independently.

### 12d. Monitoring the inter-arm distance

```bash
# Live readout
curl -s http://localhost:8888/api/bimanual/inter_arm_distance
# {"distance_m": 0.342}

# Watch in a loop (for debugging)
watch -n1 'curl -s http://localhost:8888/api/bimanual/inter_arm_distance'
```

---

## 13. Sim validation workflow

Before running on real hardware, validate behavior trees in MuJoCo
simulation.

### 13a. BT YAML dry-run (structure validation)

```bash
curl -s -X POST http://localhost:8888/api/plan \
  -H "Content-Type: application/json" \
  -d '{"instruction": "pick up the sponge with the left arm and hand it to the right arm"}' \
  | python -m json.tool
```

Verify the planner emits arm-tagged primitives with `parallel` nodes
where appropriate.

### 13b. Single-arm subset validation in sim

```bash
cd ~/spark/src
python -m spark_real.sim_pipeline \
  --instruction "pick up the sponge" \
  --headless \
  --video-out /tmp/bimanual_sim_proof/pick_sponge.mp4
```

**Note:** Full bimanual sim (two-arm MuJoCo model with dual Panda/FR3)
is not yet implemented in `sim_executor.py`. The current sim path
validates individual arm primitives and the BT structure. For bimanual
coordination (handoff, bimanual_lift), use the real hardware.

### 13c. Trial runner (automated multi-trial execution)

The server includes a trial runner for batch experiments. Three modes:

| Mode | Behavior | Use for |
|------|----------|---------|
| `auto` | Run -> home -> re-detect -> run. No pause. | Sponge/soap, pen draw |
| `semi` | Run -> home -> wait N seconds -> auto-run | Pour (varying liquid) |
| `manual` | Run -> home -> wait for user signal | Silverware, cloth fold |

---

## 14. Functional verification

### 14a. Per-arm perception

Place 2-3 objects on the table:

```bash
curl -s -X POST http://localhost:8888/api/detect \
  -H "Content-Type: application/json" \
  -d '{"prompts": ["sponge", "bowl", "mug"]}' | python -m json.tool
```

Check `position_3d` has sensible Z (above table, not negative). Verify
the external (ZED Mini) camera is providing the primary detections.

### 14b. Planning with arm tags

```bash
# Single-object (should default to right arm)
curl -s -X POST http://localhost:8888/api/plan \
  -H "Content-Type: application/json" \
  -d '{"instruction": "pick up the mug"}' | python -m json.tool

# Bimanual task (should emit arm tags + parallel nodes)
curl -s -X POST http://localhost:8888/api/plan \
  -H "Content-Type: application/json" \
  -d '{"instruction": "pick up the sponge with the left arm and scrub the bowl held by the right arm"}' \
  | python -m json.tool
```

Verify:
- Single-object plan uses `pick_with_arm` with `arm: "right"`.
- Bimanual plan uses `parallel` nodes with distinct arm tags.

### 14c. End-to-end bimanual tasks (B1-B5)

Run each task with increasing complexity:

**B1: Single-arm pick-place (one arm only)**

```bash
curl -s -X POST http://localhost:8888/api/run \
  -H "Content-Type: application/json" \
  -d '{"instruction": "pick up the mug with the right arm and place it on the tray"}'
```

**B2: Handoff (cross-arm transfer)**

```bash
curl -s -X POST http://localhost:8888/api/run \
  -H "Content-Type: application/json" \
  -d '{"instruction": "pick up the sponge with the left arm and hand it to the right arm"}'
```

**B3: Parallel hold + scrub**

```bash
curl -s -X POST http://localhost:8888/api/run \
  -H "Content-Type: application/json" \
  -d '{"instruction": "hold the bowl with the left arm and scrub it with the sponge using the right arm"}'
```

**B4: Bimanual lift (large object)**

```bash
curl -s -X POST http://localhost:8888/api/run \
  -H "Content-Type: application/json" \
  -d '{"instruction": "lift the tray with both arms"}'
```

**B5: Full compound (handover sponge + scrub)**

```bash
curl -s -X POST http://localhost:8888/api/run \
  -H "Content-Type: application/json" \
  -d '{"instruction": "pick up the sponge, hand it off for scrubbing, and clean the bowl"}'
```

---

## 15. ZMQ state broadcast

### 15a. Architecture

The bimanual rig broadcasts state on two ZMQ sockets:

| Socket | Port | Type | Purpose |
|--------|------|------|---------|
| State broadcast | `tcp://*:5601` | PUB | Full bimanual observation at 30 Hz |
| Command bus | `tcp://*:5602` | PULL | External nodes push per-arm commands |

These are SPARK application-level sockets, separate from the Bamboo
control ports (5555/5556).

### 15b. Subscribing to state broadcast

```python
import zmq, json

ctx = zmq.Context()
sub = ctx.socket(zmq.SUB)
sub.connect("tcp://localhost:5601")
sub.setsockopt(zmq.SUBSCRIBE, b"bimanual_state")

while True:
    topic, payload = sub.recv_multipart()
    data = json.loads(payload)
    print(f"ts={data['ts']:.3f}  "
          f"left_tcp={data['left']['tcp_pose'][:3]}  "
          f"right_tcp={data['right']['tcp_pose'][:3]}")
```

### 15c. Port map summary

| Port | Process | Protocol | Purpose |
|------|---------|----------|---------|
| 5555 | Bamboo control node (left) | ZMQ REQ/REP | Joint impedance commands, left arm |
| 5556 | Bamboo control node (right) | ZMQ REQ/REP | Joint impedance commands, right arm |
| 5601 | SPARK server | ZMQ PUB | Bimanual state broadcast (30 Hz) |
| 5602 | SPARK server | ZMQ PULL | External command bus |
| 8888 | SPARK server | HTTP | FastAPI REST API + UI |

---

## 16. Rollback

### 16a. Full rollback to single-arm

```bash
cd ~/spark/src
# Start the server in single-arm Franka mode instead
python -m spark_real.server --robot franka --auto-unlock --port 8888
```

The bimanual files are inert when `--robot franka` is used. All
bimanual routes return HTTP 400 ("active robot is not the bimanual
family") and bimanual skill stubs return clear errors if somehow
dispatched.

### 16b. Rollback to franky (single-arm, no Bamboo)

If Bamboo is causing issues and you need to fall back to the franky
motion-generator backend for single-arm:

```bash
# Stop Bamboo control nodes
# Start without SPARK_FRANKA_BACKEND (defaults to franky)
cd ~/spark/src
DISPLAY=:1 XAUTHORITY=/run/user/1000/gdm/Xauthority \
  python -m spark_real.server --robot franka --auto-unlock --port 8888
```

Be aware that franky will have the velocity/acceleration discontinuity
reflex issues documented in section 0.

---

## Appendix A: Bamboo driver internals

### A.1 FrankaBambooDriver (`robots/franka/franka_bamboo_driver.py`)

The Bamboo driver implements the `FrankaDriverBase` interface:

- **`connect()`**: Starts the Bamboo C++ subprocess (if not already
  running), waits for `Server listening`, then creates a
  `BambooFrankaClient` over ZMQ.
- **`move_to_joint_config(q)`**: Converts to a single-waypoint
  trajectory, estimates duration from max joint delta, calls
  `execute_joint_impedance_path()`.
- **`move_linear(pose)`**: Runs IK (pyroki) to get joint target, then
  delegates to `move_to_joint_config()`. This is the key architectural
  difference from franky: the IK is done in Python, and the C++ side
  only sees joint targets.
- **`set_collision_behavior()`**: No-op. Bamboo C++ sets 100 Nm / 100 N
  internally (matching Deoxys).
- **`SUPPORTS_VELOCITY_STREAMING = False`**: Bamboo does not (yet)
  support Cartesian velocity streaming. Use joint impedance paths.

### A.2 LD_LIBRARY_PATH

The Bamboo binary needs to find `libfranka.so.0.18.2` and Pinocchio
libraries at runtime:

```bash
LD_LIBRARY_PATH="/opt/openrobots/lib:$HOME/spark/src/external_controllers/bamboo/install/lib"
```

The driver sets this automatically when spawning the subprocess. If you
start Bamboo manually, you must set it yourself.

### A.3 Bamboo log file

The Bamboo subprocess logs to `/tmp/bamboo.log`. Check this file when
debugging connection failures.

---

## Appendix B: G1 Humanoid (Unitree G1)

The G1 humanoid is a separate bimanual platform supported by the same
`robots/factory.py` dispatch. It uses a fundamentally different control
path from the Franka bimanual rig.

### B.1 Hardware overview

| Component | Details |
|-----------|---------|
| Platform | Unitree G1-EDU 29-DOF |
| Arms | 7-DOF each (shoulder pitch/roll/yaw, elbow, wrist roll/pitch/yaw) |
| Hands | Dex3-1 multi-finger (7 motors per hand) or 1-DOF parallel gripper |
| Control interface | DDS over Ethernet (`unitree_sdk2_python`) |
| Control rate | 50 Hz (upstream SDK default) |
| Locomotion | `LocoClient` RPC (BalanceStand, Damp, Move) |
| Cameras | Binocular head cameras + per-arm wrist cameras |

### B.2 Network setup

```bash
sudo ip addr add 192.168.123.99/24 dev <iface>
sudo ip link set <iface> up
sudo ip link set <iface> multicast on
ping -c1 192.168.123.161
```

### B.3 Server startup (G1)

```bash
cd ~/spark/src
python -m spark_real.server --robot g1 --ip eth0 --port 8888
```

### B.4 What is missing for G1

1. IK pipeline (no Cartesian endpoint in SDK)
2. Hand support (stub-only)
3. Camera bring-up (YAML topic names TODO)
4. URDF + meshes on this workstation

---

## Appendix C: Files summary

| File | What it does |
|------|-------------|
| `robots/bimanual_franka/__init__.py` | Package exports |
| `robots/bimanual_franka/bimanual_franka_driver.py` | 14-DOF aggregate driver, per-arm sub-drivers, ZMQ publisher |
| `robots/bimanual_franka/dynamixel_gripper.py` | Dynamixel parallel-jaw gripper driver (ALOHA-style, USB-CAN) |
| `robots/franka/franka_bamboo_driver.py` | Bamboo C++ joint impedance backend (single-arm, used by bimanual) |
| `robots/franka/franka_base.py` | Shared constants, gripper logic, abstract interface for all Franka drivers |
| `robots/factory.py` | `bimanual_franka` in `KNOWN_FAMILIES`, `SPARK_FRANKA_BACKEND=bamboo` dispatch |
| `control/bimanual_executor.py` | BT executor with parallel nodes, sync barriers, per-arm dispatch |
| `control/bimanual_safe_robot.py` | Inter-arm CBF (12cm hard / 20cm soft / velocity gating) |
| `control/bimanual_servo.py` | Dual CartesianServo with inter-arm velocity scaling |
| `skills/bimanual.py` | 10 bimanual primitive stubs (planner prompt registration) |
| `routes/bimanual.py` | REST endpoints under `/api/bimanual/*` |
| `comms/bimanual_zmq.py` | ZMQ state broadcaster (30 Hz) + command bus |
| `perception/bimanual_camera_registry.py` | Role-keyed camera registry |
| `calibration/bimanual_charuco_stereo.py` | ZED stereo ChArUco corner measurement (step 1 of calibration) |
| `calibration/bimanual_anchor_tap.py` | Anchor-tap SVD per arm; writes `calibration_bimanual.json`, `T_right_to_left.json` |
| `routes/streaming.py` | `GET /api/capture/stereo` stereo pair + intrinsics for the tap script |
| `configs/bimanual_franka_default.yaml` | Full bimanual config (IPs, grippers, workspaces, cameras, CBF) |
| `server.py` | `--robot bimanual_franka` choice, bimanual router mount |

---

## Appendix D: Troubleshooting reference

### D.1 Bamboo control node failures

| Symptom | Diagnosis | Resolution |
|---------|-----------|------------|
| `bamboo_control_node` exits immediately | libfranka.so not found | Set `LD_LIBRARY_PATH` per section 4a |
| `bamboo_control_node` exits with "realtime" | Not in `realtime` group | `sudo usermod -aG realtime $USER`; re-login |
| `bamboo_control_node` exits with "connection refused" | Wrong robot IP or Desk locked | `ping` the arm; verify Desk is unlocked |
| `bamboo_control_node` exits with Pinocchio error | Pinocchio not installed | `mamba install -c conda-forge pinocchio` |
| Bamboo starts but Python client times out | Wrong port | Verify `-p` flag matches the port in your client code |
| `Server listening` never appears in log | FCI connection hanging | Check NIC config, direct cable, pendant in Remote |

### D.2 FCI connection failures

| Symptom | Diagnosis | Resolution |
|---------|-----------|------------|
| `"failed to connect left arm"` | Left arm IP unreachable | `ping 172.16.0.101`; check NIC cable |
| `"failed to connect right arm"` | Right arm IP unreachable | `ping 172.16.0.102`; check NIC cable |
| Both arms fail | NIC not configured | `ip addr show`; assign static IP in 172.16.0.0/24 |
| One arm connects, other times out | Desk not unlocked | Verify Desk credentials; check pendant is in Remote mode |

### D.3 Gripper failures (SSG48)

| Symptom | Diagnosis | Resolution |
|---------|-----------|------------|
| SSG48 not responding | Device not detected | Check physical USB connection; `ls /dev/ttyUSB*` |
| Permission denied on device | No udev rule / group | `sudo usermod -aG dialout $USER`; re-login |
| Gripper homing fails | Mechanical jam | Manually free the jaws; re-home |

### D.4 Calibration failures

| Symptom | Diagnosis | Resolution |
|---------|-----------|------------|
| Mean tap residual > 10 mm | Mistapped corner or wrong `--grip-offset` | Re-tap; check `grippers.jaw_offset_m` |
| Consistent 5+ mm offset in handoffs | Calibration stale | Recalibrate; check if base bolts were disturbed |
| Stereo capture finds no corners | Board outside ZED view or glare | Move the board; re-run bimanual_charuco_stereo |

### D.5 Inter-arm CBF issues

| Symptom | Diagnosis | Resolution |
|---------|-----------|------------|
| `"rejecting arm motion"` | Arms too close for planned motion | Plan meeting points with more clearance |
| Arm moves very slowly near midline | Soft barrier active (12-20 cm) | Expected behavior |
| Deadlock in parallel node | Barrier name mismatch | Check BT YAML: every `sync_barrier` name must appear in all branches |

### D.6 ZMQ issues

| Symptom | Diagnosis | Resolution |
|---------|-----------|------------|
| State broadcast disabled | pyzmq not installed | `pip install pyzmq` |
| `"Address already in use"` on port 5601 | Previous server not cleaned up | `lsof -i :5601`; kill the stale process |
| Subscriber receives nothing | Topic mismatch | Subscribe to `b"bimanual_state"` (exact bytes) |
| Port 5555/5556 conflicts | Other process using Bamboo ports | `lsof -i :5555 :5556`; stop conflicting processes |

---

## Appendix E: Quick-start cheat sheet

For operators who have already deployed once and just need the commands:

```bash
# 1. Activate environment
conda activate spark_conda

# 2. Start Bamboo left arm
LD_LIBRARY_PATH="/opt/openrobots/lib:$HOME/spark/src/external_controllers/bamboo/install/lib" \
  ~/spark/src/external_controllers/bamboo/controller/build/bamboo_control_node \
  -r 172.16.0.101 -p 5555 -g none &

# 3. Start Bamboo right arm
LD_LIBRARY_PATH="/opt/openrobots/lib:$HOME/spark/src/external_controllers/bamboo/install/lib" \
  ~/spark/src/external_controllers/bamboo/controller/build/bamboo_control_node \
  -r 172.16.0.102 -p 5556 -g none &

# 4. Wait for both to print "Server listening"

# 5. Start SPARK server
cd ~/spark/src
SPARK_FRANKA_BACKEND=bamboo \
  python -m spark_real.server --robot bimanual_franka --port 8888

# 6. Verify
curl -s http://localhost:8888/api/status | python -m json.tool

# 7. Stop (reverse order)
curl -sf -X POST http://localhost:8888/api/shutdown
kill %1 %2   # Bamboo processes
```

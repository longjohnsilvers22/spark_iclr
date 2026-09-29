# Franka FR3 / Panda driver shim (SPARK)

This package adapts the `franky` Python library
([TimSchneider42/franky](https://github.com/TimSchneider42/franky), v1.1.3,
released 2025-11-24) to the public surface of
`spark_real.control.ur10e_driver.UR10eDriver`.  The goal is that *every other
SPARK module* - perception, BT executor, primitives, pipeline - can target a
Franka without changes: instantiate `FrankaDriver(...)` where `UR10eDriver`
would be used.

The shim is intentionally thin.  No control logic is reinvented here.  Every
public call is delegated to franky/libfranka and is annotated with the
upstream symbol it forwards to (see `franka_driver.py`).  If a UR10e
capability is absent from franky (e.g. one-shot `servoJ`), the method either
emits a clear warning and falls back to the closest synchronous equivalent
or is marked `# TODO: not in upstream`.

## Why franky (and not the alternatives)

Surveyed May 2026:

| Library | Last release | Python support | Verdict |
|---|---|---|---|
| **franky** (TimSchneider42) | **v1.1.3, 2025-11-24** | py 3.7-3.13, libfranka 0.7-0.18 | **chosen** |
| panda-py (JeanElsner) | v0.8.1, 2024-07-09 | py 3.7+, libfranka up to 0.13.x | rejected - no release in 18 months, lacks FR3 0.16+ wheels |
| franka_ros2 (frankarobotics) | v2.4.0, 2026-05-04 | ROS 2 Humble (not Jazzy) | rejected - Humble-only on that release; SPARK is Jazzy / pure-Python preferred |
| libfranka C++ (frankarobotics) | rolling | none directly | rejected - no first-party Python bindings; franky wraps it |

`franky` is the right pick because (a) it ships pip-installable wheels for
libfranka 0.7-0.18 across Python 3.7-3.13, (b) supports both FR3 (the lab's
current robot family) and the older Panda from the same API, (c) exposes
joint-position, joint-velocity, Cartesian-position and Cartesian-velocity
impedance control plus Ruckig time-optimal trajectories, and (d) is actively
maintained (v1.1.3 in Nov 2025; v1.1.2 in Oct 2025; v1.0.0 in Apr 2025).

## Upstream repo

Cloned (not vendored, not a submodule) to:

```
~/spark/external/franka_upstream
```

Tagged at `v1.1.3` (commit `f847b7c29b466d5bc29d6d810c08f9844c4ec5ce`).

## Install on a deployment box

> **Lab-specific (2026-05):** the lab's FR3 runs system image **5.9.2**, which
> has a known compatibility bug with current libfranka - `Get Robot Model`
> returns an unparseable response and stock `franky.Robot(...)` hangs in
> the constructor. The lab uses **patched libfranka 0.18.2 + franky 1.1.3**
> wheels built locally. Tracking: https://github.com/frankarobotics/franka_ros2/issues/204
>
> **Prebuilt patched wheels** live at `~/franka/wheels/` and
> are already installed in the `spark_conda` (py 3.12) env. The
> `FrankaDriver.connect()` passes `realtime_config=franky.RealtimeConfig.Ignore`
> because the workstation kernel is `PREEMPT_DYNAMIC`, not `PREEMPT_RT`.
> Drop the patched build (`pip install franky-control` from PyPI) when
> Franka ships the upstream fix.

```bash
# Lab path: install the prebuilt patched wheel.
conda activate spark_conda    # Python 3.12
pip install --force-reinstall \
    ~/franka/wheels/franky_control-1.1.3-cp312-cp312-manylinux_2_34_x86_64.whl

# Fresh deployment box without our patches (after Franka fixes the bug):
pip install franky-control    # ships libfranka 0.18.0 default

# A specific libfranka version from upstream releases:
VERSION=0-17-0
wget https://github.com/TimSchneider42/franky/releases/latest/download/libfranka_${VERSION}_wheels.zip
unzip libfranka_${VERSION}_wheels.zip
pip install --no-index --find-links=./dist franky-control

# Smoke-test the import path (no robot required):
python -c "from spark_real.robots.franka import FrankaDriver; print('OK')"
```

### Caveats of the patched build

- `robot.model` is `None` and `robot.model_urdf` is `""` - **Pinocchio
  kinematics unavailable.** franky's Ruckig motion still works (uses
  libfranka's internal kinematics, not Pinocchio).
- Caller must invoke `robot.set_collision_behavior(...)` after clearing
  safety state if custom thresholds are needed (the in-constructor call
  is wrapped in try/catch so it silently fails when a safety function
  is active).
- The 589 reflex bug is version sensitive: pin a matching libfranka and
  franky release pair (see the franky changelog for compatible versions).

## What's wrapped vs. what's missing

The driver advertises three class-level capability flags so SafeRobot /
ScoreExecutor can branch without `hasattr` guessing:
`SUPPORTS_VELOCITY_STREAMING = True`, `SUPPORTS_URSCRIPT = False`,
`GRIPPER_TYPE = "franka_hand"` (UR side reports `"robotiq_2f85"`).

The shim exposes the UR10eDriver method names verbatim:

| UR10eDriver method | Franka shim implementation | Status |
|---|---|---|
| `connect()` / `disconnect()` | `franky.Robot(ip)` + `franky.Gripper(ip)` | done |
| `get_joint_positions()` | `robot.current_joint_state.position` | done |
| `get_joint_velocities()` | `robot.current_joint_state.velocity` | done |
| `get_tcp_pose()` | `robot.current_pose` -> `[xyz, axis-angle]` (UR convention) | done |
| `get_tcp_force()` | `robot.state.O_F_ext_hat_K` | done (untested on hardware) |
| `get_robot_mode()` | `robot.state.robot_mode` (enum int) | done |
| `is_steady()` | `robot.poll_motion()` | done |
| `move_to_joint_config(q, vel, acc, async)` | `JointMotion(q, relative_dynamics_factor=v/vmax)` | done - but `acc` is folded into the same factor as `vel` (franky design) |
| `move_linear(pose, vel, acc, async)` | `CartesianMotion(Affine(xyz, quat))` | done - Ruckig-shaped, not pure straight-line |
| `move_linear_relative(delta)` | `CartesianMotion(..., ReferenceType.Relative)` | done |
| `servo_joint(q, dt, ...)` | Streams `JointWaypointMotion` with `asynchronous=True`; franky's Ruckig replanner preempts on each call (see `servo_joint` docstring for the architecture). | done |
| `servo_stop()` | `JointStopMotion` | done |
| `send_velocity(linear, angular, acc, dur)` | Streams `CartesianVelocityMotion` with `asynchronous=True`; same preempt-and-replan idiom as `servo_joint`. Accepts the UR-style 6-vector single-arg form too (for `SafeRobot._forward_velocity` compatibility). | done |
| `stop_velocity()` | `CartesianVelocityStopMotion` (mode-matching stop, not `CartesianStopMotion`) | done |
| `set_collision_behavior(torque, force, ...)` | `robot.set_collision_behavior(...)` with both scalar-broadcast and per-axis upper/lower overloads; auto-called from `connect()` with defaults (20 Nm / 30 N); catches libfranka's safety-function-rejection cleanly so it can be retried after clearing safety state. | done |
| `stop()` | `Robot.stop()` | done |
| `go_home(vel)` | calls `move_to_joint_config` with the FR3 ready pose | done |
| `activate_gripper()` | `Gripper.homing()` | done |
| `open_gripper(speed)` | `Gripper.open(speed)` | done - `force` arg ignored (Franka open is unforced) |
| `close_gripper(speed, force)` | `Gripper.grasp(width=0, speed, force, eps)` | done |
| `set_gripper_position(pos[0..1], spd, frc)` | mapped to `Gripper.move` (no-force) or `Gripper.grasp` (force) | done - 0..1 scale preserved |
| `get_gripper_position()` | `Gripper.width` mapped back to 0..255 Robotiq scale | done |
| `is_object_detected()` | `Gripper.is_grasped` | done |

### Honest gaps for a real-robot demo

1. **`servo_joint` (2026-05-13: implemented).**  The driver maintains a
   "servo session" that streams successive `JointWaypointMotion` commands
   with `asynchronous=True`; franky preempts the in-flight motion on every
   call and Ruckig re-plans on the fly (this is the upstream-documented
   real-time pattern, see `README.md` ⏱️ Real-Time Motions in the franky
   repo and `examples/asynchronous.py`).  Notes / quirks:
   - franky's Python API does **not** expose a lambda-based reaction motion
     generator (only C++ does), so the "fixed Reaction + mutable shared
     target" pattern is not feasible from Python - the preempt-and-replan
     pattern is what franky actually supports.
   - `dt`, `lookahead_time`, and `gain` from the UR10e signature are
     **ignored** by design: libfranka's 1 kHz callback and Ruckig handle
     timing internally.  Callers (e.g. `vla_controller.py`) drive cadence
     by calling `servo_joint` at the desired Python-side rate.
   - Other motion methods (`move_to_joint_config`, `move_linear`,
     `move_linear_relative`) call `_preempt_servo_if_active` first; this
     issues a `JointStopMotion` and waits for the joint-position control
     mode to finalize before changing modes, because franky raises if a
     new motion would change control mode mid-stream.
2. **Orientation convention.**  Downstream SPARK code uses
   `[x, y, z, rx, ry, rz]` axis-angle (UR style).  The shim converts at the
   boundary (`_axis_angle_to_quat` / `_quat_to_axis_angle`).  This is fine,
   but be aware franky natively uses [x, y, z, w] quaternions, and tasks that
   wrap-around 180° may need to be expressed in quaternion form to avoid
   axis-flip discontinuities.
3. **Gripper choice.**  This shim assumes the **Franka Hand** (parallel jaw,
   max width 80 mm).  If the lab fits a Robotiq 2F-85 via a flange adapter
   the shim must be split: the Franka arm goes through franky, but the
   gripper would still need a separate driver (likely
   `robotiq_modbus_rtu`).  Not yet implemented.
4. **Collision behavior thresholds (2026-05-13: implemented).**  The
   driver now applies default scalar thresholds (`DEFAULT_COLLISION_TORQUE_THRESHOLD`
   = 20 Nm, `DEFAULT_COLLISION_FORCE_THRESHOLD` = 30 N) at the tail of
   `connect()` via `FrankaDriver.set_collision_behavior(...)`, which
   wraps both the 2-arg scalar-broadcast overload and the 4-arg
   per-axis upper/lower overload of `robot.set_collision_behavior(...)`.
   libfranka rejects the call when a safety function is active - the
   wrapper catches that exception and logs cleanly so `connect()`
   succeeds and the caller can retry after clearing safety state.
   Upstream: `bind_robot.cpp .def("set_collision_behavior", ...)`.
5. **External force / torque.**  `get_tcp_force()` reads
   `state.O_F_ext_hat_K`.  Franka's filter is heavily biased and noisy
   without payload calibration; expect ~1-2 N of residual.  Trust it for
   contact triggers, not for fine force control.
6. **Workspace bounds.**  The defaults are inherited from UR10e (~0.9 m
   cube).  FR3 has a ~855 mm reach so 0.85 is safer; tune for your cell.

## What needs to happen for a first physical trial

1. **Hardware**: physical FR3 (or Panda) with FCI license, dedicated NIC,
   matching libfranka version.  Real-time kernel patched into the
   workstation (`uname -a` should include `PREEMPT_RT`).
2. **Network**: low-latency direct link to controller, default 172.16.0.2.
3. **Bring-up sequence**:
   - Unlock brakes via Desk (web UI).
   - Activate FCI.
   - Run a `franky.Robot(ip)` echo test (the `examples/read.py` from the
     upstream clone is the simplest verification).
   - Then `python -c "from spark_real.robots.franka import FrankaDriver; \
       d = FrankaDriver('172.16.0.2'); d.connect(); print(d.get_tcp_pose())"`.
4. **Configure collision behavior** before any pipeline run.
5. **Run a SPARK pick-and-place primitive** with `relative_dynamics_factor`
   pinned to 0.05 for the first trial.  Compare wrist-cam scene to the
   expected planning frame; the URDF camera mount differs from the UR10e
   wrist setup.
6. **`servo_joint` is wired** (as of 2026-05-13 on the PREEMPT_RT lab box).
   For a first VLA rollout: pin `relative_dynamics_factor` low (≤0.1 via
   the `velocity` arg) and verify the Python-side cadence with a logging
   wrapper before driving the arm.

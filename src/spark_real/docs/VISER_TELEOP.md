# Viser-IK Teleop (robots_realtime integration)

Wires the external [`uynitsuj/robots_realtime`](https://github.com/uynitsuj/robots_realtime)
viser-IK teleop into SPARK as a **subprocess** managed from the FastAPI
server. Gives the operator a browser-based SE(3) gizmo (drag the frame
in 3D, pyroki solves IK at 100 Hz, panda-py OSC streams joint targets).
Coexists with the existing gamepad/keyboard teleop - but only one mode
can hold the FCI at a time.

## Why a subprocess (not in-process)

| Constraint                        | spark_real (current)         | robots_realtime          |
|-----------------------------------|------------------------------|--------------------------|
| Python                            | 3.10 / 3.12 (`spark_conda`)  | 3.11 only                |
| libfranka client                  | franky 1.1.3 (libfranka 0.13)| panda-py 0.7.x-0.8.x     |
| FCI session                       | exclusive while connected    | exclusive while connected|

Same FCI cannot be held by two libfranka clients simultaneously. The
versions also disagree, so they can't share a Python process. A
subprocess with its own uv venv is the only clean answer.

## Install (one-time)

This is what actually works on the SPARK rig - three things diverge from
the upstream README:

1. **`[franka_panda]` extra is required** (panda-py is optional in
   the upstream `pyproject.toml`).
2. **numpy must be ≥2** - pyproject pins it to 1.26.4, but pyroki/jaxls
   need `numpy.dtypes.StringDType` (numpy 2.x only). Use the venv's
   pip-like `uv pip install --no-deps --force-reinstall` to bypass the
   override.
3. **panda-py must be 0.8.x / libfranka 0.13.x** - the bundled
   `panda-py 0.7.5 / libfranka 0.10.0` wheel is for the original Panda
   and gets `IncompatibleVersionException` against an FR3 on the
   libfranka 0.13.x line (which is what SPARK's franky 1.1.3 already
   uses).

```bash
# uv (if missing)
curl -LsSf https://astral.sh/uv/install.sh | sh
export PATH="$HOME/.local/bin:$PATH"

# Clone if not already
git clone --recurse-submodules https://github.com/uynitsuj/robots_realtime ~/robots_realtime

# Venv + deps (with franka_panda extra)
cd ~/robots_realtime
uv venv --python 3.11
uv pip install -e ".[franka_panda]"

# Patch 1: numpy 2.x (bypasses the pyproject override pin)
uv pip install --python ~/robots_realtime/.venv/bin/python --no-deps --force-reinstall \
  "numpy>=2.0,<3"

# Patch 2: FR3-compatible panda-py (libfranka 0.13.3)
cd /tmp
gh release download v0.8.1 -R JeanElsner/panda-py \
    --pattern "panda_py_0.8.1_libfranka_0.13.3.zip"
unzip -o panda_py_0.8.1_libfranka_0.13.3.zip \
    "panda_python-0.8.1+libfranka.0.13.3-cp311-cp311-manylinux_2_17_x86_64.manylinux2014_x86_64.whl"
uv pip install --python ~/robots_realtime/.venv/bin/python --no-deps --force-reinstall \
  "/tmp/panda_python-0.8.1+libfranka.0.13.3-cp311-cp311-manylinux_2_17_x86_64.manylinux2014_x86_64.whl"
```

Verify with the helper script:

```bash
~/spark/scripts/viser_teleop_check.sh
```

Hardware smoke test (do this with SPARK server stopped and FR3 unlocked):

```bash
~/robots_realtime/.venv/bin/python -c \
  "import panda_py; p = panda_py.Panda('172.16.0.2'); print(p.get_state())"
```

If you get `IncompatibleVersionException`, your FR3 firmware is on a
libfranka line newer than 0.13.3 - check the [panda-py releases page](https://github.com/JeanElsner/panda-py/releases)
for a newer wheel and repeat patch 2.

## Lifecycle

The integration is mutually exclusive with SPARK's own teleop and
executor:

```
gamepad teleop / executor  ──┐
                             │  (start)            (stop)
                             ▼                     │
            SPARK driver connected   ──►   SPARK driver disconnected   ──►   SPARK driver reconnected
            FCI held by SPARK              FCI held by rr-session             FCI held by SPARK
                                           viser at :8765
```

While the rr-session subprocess is alive:
- `/api/velocity` returns `{"skipped": "viser teleop active"}`
- `/api/execute*` returns 400 with a clear message
- `/api/robot_state` may stale-read or 400 (driver is `None`)

## Endpoints

| Verb | Path                              | Returns |
|------|-----------------------------------|---------|
| POST | `/api/teleop/viser/start`         | `{success, pid, viser_url}` |
| POST | `/api/teleop/viser/stop`          | `{success, subprocess_exit_code, reclaimed}` |
| GET  | `/api/teleop/viser/status`        | `{running, viser_url?, pid?}` |
| GET  | `/api/teleop/viser/log?tail_bytes=N` | rr-session subprocess log tail |

UI: green "Viser IK Teleop" button below the gamepad block (Teleop card,
Franka family only). Auto-opens the viser URL in a new tab.

## SPARK-specific config

The teleop config that ships with this integration is at
`~/robots_realtime/configs/franka/franka_spark_viser_teleop.yaml`. Diff vs
upstream `franka_robotiq_viser_teleop.yaml`:

- `host_name: 172.16.0.2` (SPARK FCI, not the upstream default `.0.1`)
- `robotiq_gripper: false` (SPARK uses the default Franka parallel-jaw)
- No `CameraNode` (SPARK owns the Kinects + RealSense; do not contend
  for USB enumeration)
- Recordings → `/tmp/spark_viser_recordings` so they don't pollute SPARK
  episode output

## Failure modes & recovery

| Symptom | Likely cause | Fix |
|---------|--------------|-----|
| Start returns `400 "robots_realtime venv not found"` | uv venv not built | Run install steps above |
| Start returns 500 with `subprocess exited immediately` | FCI still held / Desk locked / wrong IP / panda-py version mismatch | Check `log_tail` in response; check `/api/teleop/viser/log` |
| Viser URL loads but gizmo doesn't move arm | rr-session connected but motion thread rejected motion | Operator hit a safety reflex; press the rest pose button in the viser GUI |
| Stop returns `FCI reclaim failed` | rr-session didn't fully release | Wait 5 s; call stop again; worst case `~/spark/scripts/spark_server.sh restart` |

## Source layout

- `~/spark/src/spark_real/routes/viser_teleop.py` - start/stop/status/log routes
- `~/spark/src/spark_real/routes/state.py` - `viser_teleop_proc` handle
- `~/spark/src/spark_real/routes/control.py` - `/api/velocity` guard
- `~/spark/src/spark_real/routes/core.py` - `_robot_ready()` guard
- `~/spark/src/spark_real/frontend/index.html` - Viser IK button + JS
- `~/robots_realtime/configs/franka/franka_spark_viser_teleop.yaml` - session config
- `~/robots_realtime/robot_configs/franka/franka_spark_fr3.yaml` - robot config

## IK cost toggles

`spark_real/control/fr3_ik_pyroki.py` carries two opt-in cost terms
lifted from the robots_realtime pyroki snippets, gated by env vars
so they can be A/B tested against the historical default without
disturbing anything. **Both default OFF.**

| Env var | Effect | Source in robots_realtime |
|---------|--------|---------------------------|
| `SPARK_IK_VEL_SMOOTHING=1` | Adds a per-step `limit_velocity_cost(prev_cfg, dt=0.01, weight=0.1)` that penalizes Δq exceeding `robot.joints.velocity_limits`. `prev_cfg` is the cached last successful solution (or the seed if there's no cached q yet). | `robots_realtime/robots/inverse_kinematics/pyroki_snippets/_solve_ik_vel_cost.py` |
| `SPARK_IK_COLLISION=1` | Adds `pk.costs.self_collision_cost` (margin=0.02 m, weight=5) + `pk.costs.world_collision_cost` against a `pk.collision.HalfSpace` at z=-0.025 m (SPARK table floor). | `robots_realtime/robots/inverse_kinematics/pyroki_snippets/_solve_ik_with_collision.py` |

Internally the module lazy-builds four JIT solvers - `_jit_solve`
(default), `_jit_solve_smooth`, `_jit_solve_collision`,
`_jit_solve_smooth_collision` - and picks one per call based on the
two env vars. The default path is byte-identical to the historical
factor list (`pose / limit / rest / manipulability`), so leaving both
env vars unset has zero effect.

Order-of-magnitude cost on this host (CPU JAX, FR3, single solve):

| Mode | First call (incl. JIT) | Warm call |
|------|------------------------|-----------|
| default | ~8 s | 10-40 ms |
| `SPARK_IK_VEL_SMOOTHING=1` | ~8 s | 15-30 ms |
| `SPARK_IK_COLLISION=1` | ~12 s | 20-70 ms |
| both | ~12 s | 30-50 ms |

The collision variant pays its extra ~4 s of compile time because
`pk.collision.RobotCollision.from_urdf` adds 300 active self-pair
residuals + 1 world-pair residual to the factor graph. Warm calls
are still well inside the planner's per-step budget.

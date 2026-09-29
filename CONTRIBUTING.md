# Contributing to SPARK

Thanks for your interest. SPARK is a training-free manipulation stack: perception
grounds an instruction, one LLM call writes a typed behavior tree, and IK-driven
control executes it across three real embodiments. This guide covers the dev
setup, the workflow, and — the most common contribution — adding a new robot.

## Development setup

The full environment (with CUDA torch, SAM3, and the grasp stack) is built by
the conda script; see [SETUP.md](SETUP.md) for the complete rig setup:

```bash
conda env create -f environment.yml      # creates the `spark_conda` env
conda activate spark_conda
bash src/setup_spark_conda.bash           # cu130 torch, SAM3, franky wheel, ocnn
```

For code-only work (no hardware, no perception weights), an editable install of
the package with dev tooling is enough:

```bash
pip install -e ".[dev,sim]"               # core + pytest/ruff/black
```

torch/torchvision are intentionally excluded from `pyproject.toml` — install
them from the pytorch.org CUDA wheel index first (the pip resolver otherwise
pulls a CPU build). Per-robot drivers are optional extras: `.[ur10e]` for the
UR10e (ur-rtde); the Franka driver (`franky-control`) is a local wheel, not on
PyPI (see SETUP.md).

## Workflow

- Branch from the active integration branch, not `main` (`main` is the
  paper/website line).
- Format and lint before committing: `black .` then `ruff check .`.
- Run the hardware-free tests: `pytest`. They mock the robot and cameras, so
  they run without a rig. The `tests/corl/` and `tests/viser_*` files are
  hardware/GPU demos, not unit tests, and are excluded from the default run.
- Keep comments load-bearing: explain a real constraint, not what the next line
  obviously does. No commit-message narration left inline.
- Robot-specific values (IPs, serials, workspace bounds, home config, gripper,
  timeouts, grasp/stack tuning) belong in `configs/<family>_default.yaml`, never
  hardcoded in shared `.py`. New tuning knobs go in YAML, not `os.environ`.

## Adding a new robot family

A robot family is selected at launch with `--robot <family>` and resolves to a
driver + a YAML config. Adding one (e.g. `kinova`) is four steps:

1. **Driver** — `src/spark_real/robots/kinova/kinova_driver.py`. Implement the
   driver interface (connect/disconnect, `get_tcp_pose`, `get_joint_positions`,
   `get_observation`, `move_to_joint_config`, `send_velocity`/`stop_velocity`,
   `open_gripper`/`close_gripper`, `get_gripper_position`, `is_object_detected`,
   …). Crib `control/ur10e_driver.py` for the full method list. Set the class
   attributes `robot_family`, `SUPPORTS_URSCRIPT` (False unless you speak
   URScript), and `GRIPPER_TYPE`. Capability flags keep robot-specific paths
   (URScript, gripper style) out of the shared executor — set them honestly and
   the executor dispatches correctly.

2. **Config** — `src/spark_real/configs/kinova_default.yaml`. Provide `robot.ip`,
   `home_config`, `gripper`, `cameras`, `workspace`, `control`, and
   `collision_behavior`. Everything embodiment-specific lives here.

3. **Register** — add the family to `robots/factory.py` (the `KNOWN_FAMILIES`
   tuple and the dispatch branch) and to the `Family` literal in `config.py`.

4. **Verify** — bring it up perception-only first (`--robot kinova --no-robot`
   won't connect the arm), then on hardware. Run `pytest` to confirm nothing
   family-shared regressed.

> **Known gotchas (being cleaned up):** some shared executor code still branches
> on `if family == "ur10e"/"franka"` for reach limits, grasp orientation, and
> safety bounds (`control/executor_core.py`, `control/executor_ik.py`,
> `pipeline_init.py`). A new family falls through to a default there; check those
> sites if behavior looks off. The driver interface is currently duck-typed
> rather than an enforced base class, so a missing method surfaces at runtime,
> not at construction — test every primitive path once.

## Reporting issues

Include the robot family, the instruction, the relevant `output/logs/` excerpt,
and whether it reproduces in perception-only mode (`--no-robot`).

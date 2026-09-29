# SPARK: Sequential Planning via Anchored Robotic Keypoints

Training-free neurosymbolic manipulation: open-vocabulary perception grounds an
instruction, one LLM call writes a typed behavior tree over a fixed primitive
library, and IK-driven control executes it with perception re-grounding on
failure. No training, no demonstrations.

```
Instruction + RGB-D
   |  SAM3 segmentation + metric depth  -> 3D keypoints per object
   v
   one LLM call -> typed YAML behavior tree
   v
   ScoreExecutor dispatches primitives -> robot driver
   |  on failed post-condition: re-perceive, re-detect, retry
```

One codebase drives three real embodiments, chosen by a launch flag: Franka FR3
(franky), UR10e (ur-rtde), and a bimanual Panda + FR3 workcell, with all
embodiment-specific values in per-family YAML configs. The typed BT-DSL
(`spark_dsl`) makes the space of valid plans formal; `spark_bench` runs the
LIBERO / LIBERO-PRO and robosuite benchmarks. See the paper for results.

## Structure

```
src/
  spark_real/   real-robot pipeline: FastAPI server, perception, planning,
                control, skills, drivers, calibration
  spark_bench/  benchmark runners (LIBERO-PRO, robosuite, BEHAVIOR)
  spark_dsl/    typed BT-DSL: grammar, primitive library, macros
  sam3_service/ SAM3 serving helper
  scripts/      standalone perception/fold helpers
  docs/         architecture, comparisons, calibration, references
```

## Install

See [SETUP.md](SETUP.md).

## Run

```bash
cd src

# hardware-free boot (no robot/cameras/API key); UI at :8888
python -m spark_real.server --no-robot --no-init --no-kinect

# real robot: franka (default), ur10e, or bimanual_franka
python -m spark_real.server --robot franka --auto-unlock

# headless planner
python -m spark_real.run planner --instruction "pick up the red block"

# benchmark
python -m spark_bench.run_spark_libero_pro_fair --suite goal --perturbation position --use-dsl
```

## Docs

- [spark_real](src/spark_real/README.md) - pipeline, control stack, drivers, IK, configs
- [spark_bench](src/spark_bench/README.md) - benchmark runners and results
- [spark_dsl](src/spark_dsl/README.md) - typed BT-DSL
- [architecture](src/docs/ARCHITECTURE.md), [comparisons](src/docs/COMPARISONS.md)

## License

Apache-2.0.

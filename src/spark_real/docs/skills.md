# spark_real/skills

Decorator-based skill registry. Every primitive that the SPARK behaviour tree
can call is a function decorated with `@spark_skill`; the registry
auto-discovers them at import time and the planner's prompt is built
dynamically from the registered metadata.

## Adding a new skill

```python
# spark_real/skills/my_skill.py
from spark_real.skills.registry import spark_skill

@spark_skill(
    name="poke",
    description="Briefly contact a target then retract along approach axis",
    params={"target_label": str, "depth": float, "duration": float},
)
def poke(executor, params):
    target = params["target_label"]
    depth = float(params.get("depth", 0.005))
    # ... use executor.robot, executor.detection_map, etc.
    return _result("poke", success=True, message=f"poked {target}")
```

Drop the file in this directory; the `__init__.py` walker imports every
sibling module so `@spark_skill` runs at startup. The planner's
`_build_system_prompt()` then includes `poke` in the primitive list it shows
Gemini, and the score executor's `_dispatch` finds it via
`registry.dispatch("poke", executor, params)`.

## Layout

| File | Purpose |
|---|---|
| `registry.py` | `@spark_skill`, `SkillEntry`, `SkillRegistry` (auto-discover + dispatch) |
| `primitives.py` | The 5 base primitives the BT-DSL grammar treats as atomic: `move_to_keypoint`, `grasp`, `release`, `move_relative`, `wait` |
| `manipulation.py` | Compound primitives: `wiggle`, `push_object`, `open_drawer`, `screw` |
| `grasping.py` | SE(3) 6-DOF grasping via EquiGraspFlow (used inside `grasp` when enabled) |
| `recovery.py` | Automatic recovery skills triggered by `execution_recovery.py` after a failure |

## Conventions

- **Signature**: `def fn(executor, params: dict) -> ExecutionResult`
- **Return** `_result(action_type, success, message="", duration=0.0)` - never raise out of a skill (let the executor wrap exceptions)
- **Idempotence**: skills should be safe to call twice in a row (the recovery layer may retry)
- **Workspace bounds**: every motion goes through `executor.robot` which enforces TCP/joint deltas; skills don't need to re-check
- **Logging**: `logger = logging.getLogger(__name__)` per file, prefix messages so they're searchable in trial logs

## Used by

- `spark_real.control.score_executor.ScoreExecutor._dispatch` - looks up primitives by name
- `spark_real.planning.spark_planner._build_system_prompt` - injects skill list + descriptions into the LLM prompt
- `spark_real.control.execution_recovery` - calls recovery-tagged skills automatically on partial failure

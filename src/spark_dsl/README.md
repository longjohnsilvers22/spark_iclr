# spark_dsl

Typed BT-DSL for SPARK behavior trees: a Lark grammar, a typed primitive
library, dynamically-mined macros, and a runtime that validates, expands, and
dispatches plans against the legacy execution stack.

## Why

Free-form YAML BTs from the LLM occasionally invent primitives, drop required
slots, or wrap nodes in shorthand the executor can't parse. This module makes
the surface area of acceptable plans formal:

- Every primitive has a typed slot signature.
- Macros (e.g. `pick_and_place`) are expanded to base primitives at runtime.
- A failed validation is a clean error message, not a silent no-op.

The pipeline is identical between LIBERO simulation, robosuite, and
real-robot - same grammar, same library, same executor.

## Layout

```
spark_dsl/
  grammar.lark                Lark EBNF grammar for BT YAML
  skill_library.py            Typed Primitive registry + DEFAULT_LIBRARY
  prompt_builder.py           Build the Gemini prompt from the library
  executor.py                 BTExecutor: validate -> expand_macros -> dispatch
  macro_mining.py             N-gram subtree miner (extracts macros from logs)
  macros.yaml                 22 mined macros (auto-generated)
  calibration.py              Welford per-(object, surface) offset DB
  runtime_calibration.py      Live calibration updates from successful trials
  tests/                      Pytest suite (49 tests, parity-validated)
```

## How it integrates

The runners (`spark_bench/run_spark_libero_pro_fair.py`,
`spark_bench/run_spark_robosuite.py`) accept `--use-dsl`. With the flag set:

1. Plan -> `_gemini_plan_via_dsl` builds a typed prompt and parses Gemini's
   YAML against the grammar.
2. Validate -> `DEFAULT_LIBRARY.validate_bt(score)` returns errors if any.
3. Expand macros -> `BTExecutor.expand_macros(score)` recursively inlines
   macro bodies to base primitives.
4. Flatten -> the runner's existing `flatten()` walks nested
   `sequence`/`selector` nodes and emits a flat action list.
5. Dispatch -> the legacy stateful `_dispatch_action` closure executes each
   base primitive, preserving cross-action state (`holding`, smooth approach).

## Adding a primitive

1. Edit `skill_library.py:_BASE_PRIMITIVES` - add a new `Primitive` with its
   typed `SlotSig` (`name -> (type, required, doc)`).
2. Implement the runtime in the dispatcher (one of `_dispatch_action` in
   `run_spark_libero_pro_fair.py` for sim, `_execute_on_libero` for robosuite,
   or a `@spark_skill` in `spark_real/skills/` for real-robot).
3. Optionally register a macro in `macros.yaml` that uses the new primitive.

## Adding a macro

Append an entry to `macros.yaml`:

```yaml
- name: pick_then_lift
  free_params:
    node0.keypoint_label: { example: "red block" }
  body:
    - type: move_to_keypoint
      params: { keypoint_label: "<node0.keypoint_label>" }
    - type: grasp
      params: { force: 100 }
    - type: move_relative
      params: { dz: 0.20 }
```

Free-slot placeholders are translated to clean signature names by
`prompt_builder._clean_signature` (`<node0.keypoint_label>` becomes
`<pick_label>` in the prompt + macro body).

## Testing

```bash
cd ~/spark/src
pytest spark_dsl/tests/ -v
```

49 tests cover grammar parsing, library validation, macro expansion (incl.
cycle detection), prompt construction, executor dispatch, and substitution
edge cases.

## Validation status

50-trial LIBERO-PRO benchmark (`results/dsl_benchmarks/RESULTS.md`):

| Suite | DSL | Reference (sv63) |
|---|---|---|
| goal-pos (skip drawer) | 42.0% | 40.0% |
| robosuite Lift | 100% | 100% |

DSL is at parity with the legacy pipeline. The grammar-constrained surface is
what Phase 2 (Gemma 4 SFT + GRPO) will fine-tune against.

# Intent-Aware "Skip Pick If In Container" — Design Spec

Status: proposal (design only; no code changed)
Scope: `spark_real` UR10e pipeline
Author: investigation of the plushie-in-bowl regression

---

## 1. Problem

The pipeline has a guard that, when the arm is **not holding** anything and it is
about to pick object `X`, checks whether `X`'s XY currently lies inside a known
container region. If so it **skips the pick descent** and returns success.

- **Intended use (must preserve):** "silverware / clear-the-table" tasks. After a
  utensil is placed in a tray, a re-detect finds it "in the container"; the guard
  stops the robot from re-picking what it just placed.
- **Regression it caused:** a plushie ("waddle dee") that *starts inside a bowl*
  and must be taken **out**. The guard saw it inside the bowl and skipped the
  descent, so the gripper closed on air ~55 cm up at home height.

A stopgap was added to one of the three guard sites that gates the skip on
`label in self._placed_labels` (skip only objects **we** placed this run). That
fixes the immediate case *for that one path* but is still position-based, not
intent-based, and — critically — **does not cover the path the take-out prompt
actually emits** (see §3).

### 1.1 Why position alone can never decide this

For both the silverware case and the take-out case the geometric fact is
**identical**: "`X`'s XY is inside container `Y`." The only thing that differs is
the *role of the container in the task*:

| Task              | Container `Y` is the object's… | Correct action     |
|-------------------|--------------------------------|--------------------|
| Silverware re-pick| **goal** (we just placed it)   | skip (leave it)    |
| Take-out          | **source** (it started there)  | pick (take it out) |

Source-vs-goal is **task intent**. It is not recoverable from a single frame of
geometry. It has to come from the instruction, i.e. from Gemini's plan.

---

## 2. Where the guard lives today (three divergent copies)

There is not one guard, there are **three**, and they disagree:

1. **`_move_to_keypoint`** — `spark_real/control/executor_motion.py:118-147`.
   Has the stopgap: skips only if `_in is not None and label in self._placed_labels`
   (`executor_motion.py:134`). Correct for the plushie *only if* the plan reaches
   the object through a top-down `move_to_keypoint`.

2. **`grasp_se3`** — `spark_real/skills/grasping.py:127-148`.
   **No stopgap.** Skips unconditionally whenever `grasp_target_in_container(...)`
   is truthy (`grasping.py:138`). This is the important one: the take-out prompt
   explicitly instructs Gemini to pick with `grasp_se3(X)`
   (`spark_real/planning/spark_planner.py:226`), so **the plushie bug is still
   live through this path** — the stopgap never touched it.

3. **`recover_grasp`** — `spark_real/control/execution_recovery.py:243-257`.
   During grasp recovery, a re-detect that lands inside a container marks the
   label placed and skips the recovery grasp. No intent gate.

All three call the same geometry helper `grasp_target_in_container(executor, label)`
(`execution_recovery.py:106-147`), which tests `X`'s XY against every container in
`executor._destination_labels` via `xy_inside_container` /
`_is_container_like` (`execution_recovery.py:66-103`).

`_destination_labels` is populated from the plan's place targets in
`pipeline_execution.py:374`. For a take-out task the plan places the object next
to the source container (`move_to_keypoint(Y, offset_x: 0.18, ...) -> release`,
`spark_planner.py:226`), so **`Y` is simultaneously the source and a
destination label** — which is exactly what makes the position guard fire on the
object we are trying to remove.

> Note: the plan-time destination filter that *drops* detections inside a
> destination (`pipeline_execution.py:338-366`) does **not** drop the plushie,
> because `X` is a referenced pick label and is protected by `referenced_labels`
> (`pipeline_execution.py:347`). So the object survives in `detection_map`; the
> failure is purely the pick-skip guard, not detection loss.

---

## 3. Design goal

Rewrite the skip decision so it consults **intent**:

- **Skip** only when the object is at its **intended goal** — i.e. we placed it
  there this run (`_placed_labels`), and the plan is not asking to move it again.
- **Never skip** when the plan's active intent for that label is a **pick**
  (normal pick, or an explicit take-**out**).
- Do this **once, in one shared predicate**, so all three sites agree.
- Preserve the silverware re-pick guard **exactly**.

---

## 4. Candidate mechanisms (ranked)

### 4.1 Look-ahead-for-grasp (executor infers "this is a pick") — REJECTED as primary

Idea: in the guard, look at the flattened action list; if a `grasp` follows this
`move_to_keypoint` for the same label, it's a pick, so never skip.

**Why it is not sufficient (decisive):** the silverware re-pick we must preserve
is *also* a pick — `move_to_keypoint(fork) -> grasp` (or `grasp_se3(fork)`). A
"grasp follows ⇒ never skip" rule would refuse to skip the just-placed fork too,
**reintroducing the re-pick the guard exists to prevent** — violating the hard
constraint. So look-ahead alone cannot be the deciding signal.

Also: the guard already only runs when `not self._holding`
(`executor_motion.py:118`), and a not-holding approach that is followed by a
grasp *is* a pick — so "a grasp follows" is nearly the same bit as
`not self._holding` and carries almost no new information here. And it would need
new plumbing: neither `_move_to_keypoint(params, t0)` nor the `grasp_se3` skill
(dispatched as `registry.dispatch(type, executor, params)`) currently receives
the action list or index (`executor_core.py:837-877`), so the executor would have
to stash `self._current_actions` / `self._current_action_index` in `_run_actions`
before each dispatch.

**Verdict:** keep as an *optional* assertion only (see §6), not the mechanism.

### 4.2 `_placed_labels` goal-membership gate — REQUIRED, but not new

This is the existing stopgap (`executor_motion.py:134`). It *is* the signal that
preserves silverware: `_placed_labels` is populated only after a successful
`release` (`executor_core.py:677-679`) or an explicit place/skip
(`grasping.py:139`, `execution_recovery.py:250`). "We placed it ⇒ it's at its
goal ⇒ don't re-pick" is precisely the silverware rule.

It also *already* fixes the plushie for the `move_to_keypoint` path: the plushie
was never placed by us, so it's not in `_placed_labels`. The remaining defect is
only that **`grasp_se3` and `recover_grasp` never got this gate**.

**Verdict:** promote this gate into the shared predicate so *all three* sites use
it. This alone closes the live `grasp_se3` bug.

### 4.3 Planner intent param — RECOMMENDED (the actual "intent" layer)

Have Gemini mark the pick with an explicit flag when the task is to take the
object **out** of / **remove** it **from** / **empty** a container:

```yaml
- type: grasp_se3            # or move_to_keypoint on the top-down path
  params:
    keypoint_label: "waddle dee"
    from_container: true      # <-- authoritative "this is a pick-OUT"
```

Rides the params dict that **both** `_move_to_keypoint(params, ...)` and the
`grasp_se3` skill already receive — **zero new plumbing**. It is the only signal
that carries genuinely new information: the source-vs-goal role of the container,
which geometry cannot supply.

It is authoritative for the one case `_placed_labels` cannot handle: a plan that
places `X` in a tray and *then, in the same run,* is asked to take `X` back out.
There `X ∈ _placed_labels` (goal-membership says "skip") but intent says "pick".
`from_container: true` overrides the skip.

Cheap for Gemini: it already parsed "take … out of …". Temperature is 0, so the
flag is deterministic.

**Verdict:** primary intent mechanism. Combined with §4.2 it is complete.

### 4.4 Recommendation

Adopt **§4.2 + §4.3**: a single shared predicate gated by `_placed_labels`
(preserves silverware, closes the `grasp_se3` bug) with a `from_container` /
`intent: pick_out` **override** (handles the goal-then-remove edge case and makes
intent explicit). Treat §4.1 look-ahead as an optional debug assertion only.

---

## 5. Exact new skip condition

Add one shared helper (suggested home: `execution_recovery.py`, next to
`grasp_target_in_container`, so all three sites import from one place):

```python
def _plan_says_pick_out(params: dict) -> bool:
    """True when the plan explicitly marks this pick as a take-OUT."""
    if not params:
        return False
    if params.get("from_container") in (True, "true", "True", 1):
        return True
    if str(params.get("intent", "")).lower() in ("pick_out", "take_out", "remove"):
        return True
    return False


def container_skip_target(executor, label: str, params: dict) -> Optional[str]:
    """Return the container label to skip into, or None to proceed with the pick.

    Skip ONLY when the object is at its intended goal:
      * it currently sits inside a known container, AND
      * WE placed it there this run (silverware re-pick guard), AND
      * the plan is NOT explicitly asking to take it back out.
    """
    if _plan_says_pick_out(params):          # intent override: always pick
        return None
    _in = grasp_target_in_container(executor, label)   # position gate
    if _in is None:
        return None
    if label not in getattr(executor, "_placed_labels", set()):  # goal gate
        return None                          # started-in-container -> pick it
    return _in                               # placed-by-us + still inside -> skip
```

Then every guard site collapses to the same three lines. E.g. in
`_move_to_keypoint` (`executor_motion.py:118-147`):

```python
if not self._holding:
    _in = container_skip_target(self, label, params)
    if _in is not None:
        self._placed_labels.add(label)
        return ExecutionResult(action_type="move_to_keypoint", success=True,
                               message=f"'{label}' already in container '{_in}', skipping",
                               duration=time.time() - t0)
```

And identically in `grasp_se3` (`grasping.py:127-148`) — this is the change that
actually closes the live plushie bug on the take-out path — and in
`recover_grasp` (`execution_recovery.py:243-257`), where `params` is the grasp
params already in scope so the `from_container` flag is honored during recovery
too.

**Behavior table under the new predicate:**

| Scenario                                   | in container | `_placed_labels` | `from_container` | Result |
|--------------------------------------------|:---:|:---:|:---:|--------|
| Silverware: re-pick just-placed fork       | yes | **yes** | no  | **skip** (preserved) |
| Plushie: starts in bowl, take out          | yes | no  | yes | **pick** (fixed) |
| Plushie via `grasp_se3` (take-out prompt)  | yes | no  | yes | **pick** (fixed — was broken) |
| Normal pick of a free object on the table  | no  | no  | no  | pick   |
| Placed X, then asked to take X back out    | yes | yes | **yes** | **pick** (override) |

The silverware column is unchanged from today's stopgap, so the re-pick guard is
preserved exactly (§7).

---

## 6. Optional look-ahead assertion (belt-and-suspenders, not the mechanism)

If desired, stash the flattened actions + index on the executor in `_run_actions`
(`executor_core.py:581-654`) before dispatch, then have `container_skip_target`
*log a warning* (not change behavior) if it is about to skip a label for which a
`grasp`/`grasp_se3` immediately follows and `from_container` was **not** set —
that combination means the plan wanted a pick but the planner forgot the flag.
This surfaces missing-intent plans without risking the silverware guard. Keep it
log-only; do **not** let it force a pick, or it re-creates the §4.1 hazard.

---

## 7. Planner-prompt addition

In `spark_real/planning/spark_planner.py` (the take-out rule at line 226, and the
silverware "only pick objects not already in the container" rule at lines
158-160), add:

> **Container-relative intent (REQUIRED).** When the task is to take an object
> OUT of, remove it FROM, or empty a container ("take the waddle dee out of the
> bowl", "remove the fork from the tray", "empty the basket"), you MUST add
> `from_container: true` to the params of the pick action (the `grasp_se3`, or the
> `move_to_keypoint` that precedes the `grasp`). This tells the executor the
> object starts at its **source** container and must be picked even though it
> looks "inside" a container. Example:
> ```yaml
> - type: grasp_se3
>   params: { keypoint_label: "waddle dee", from_container: true }
> - type: move_relative
>   params: { dz: 0.15 }
> - type: move_to_keypoint
>   params: { keypoint_label: "bowl", offset_x: 0.18, offset_z: -0.04 }
> - type: release
> ```
> Do NOT set `from_container` on a normal pick from the open table, and do NOT set
> it on the place action. For "put / place X in Y" tasks the container is the
> **goal**: never set `from_container`, and never emit a pick for an item that is
> already inside the target container (unchanged from the silverware rule above).

This makes the source-vs-goal role explicit in the plan and is deterministic at
temperature 0. The prompt is read fresh on every `generate_score` call
(`spark_planner.py:37-47`), so no server restart is needed to roll it out.

---

## 8. Silverware-preservation argument

The re-pick guard the silverware task depends on is: *"after we place a utensil in
the tray, do not pick it again."* Under the recommended predicate that behavior is
driven entirely by the `_placed_labels` goal gate, which is unchanged in meaning
from the current stopgap (`executor_motion.py:134`):

- A utensil we placed enters `_placed_labels` on `release`
  (`executor_core.py:677-679`) or on a prior skip (`grasping.py:139`).
- On any later approach it is *still inside* the tray → `grasp_target_in_container`
  truthy → `label in _placed_labels` true → `from_container` not set (place/clear
  tasks never set it, per §7) → **skip**. Identical to today.

The change only *adds* skips-that-should-not-happen removals:

1. It extends the `_placed_labels` gate to `grasp_se3` and `recover_grasp`, which
   previously skipped **unconditionally** — so a plushie that starts in a bowl is
   no longer skipped (the fix), while a placed fork (in `_placed_labels`) is still
   skipped (preserved).
2. It adds the `from_container` override, which only ever *reduces* skipping and
   only when the plan explicitly asks to take an object out — it can never cause a
   placed utensil (where the flag is absent) to be re-picked.

Therefore no silverware/clear-table plan changes behavior: the objects those
plans re-detect "in the tray" are exactly the ones in `_placed_labels` with no
`from_container` flag, which still skip.

---

## 9. Change summary (for a future implementation PR)

1. Add `_plan_says_pick_out` + `container_skip_target` to `execution_recovery.py`.
2. Replace the guard bodies at `executor_motion.py:118-147`,
   `grasping.py:127-148`, and `execution_recovery.py:243-257` with a call to the
   shared predicate (passing the in-scope `params`).
3. Add the container-relative-intent block to the planner prompt
   (`spark_planner.py`, take-out rule ~line 226 and silverware rule ~lines 158-160).
4. (Optional) Stash `_current_actions`/`_current_action_index` in
   `executor_core.py:_run_actions` and add the log-only look-ahead assertion (§6).

No robot, server, or config changes are required; the prompt is hot-reloaded.

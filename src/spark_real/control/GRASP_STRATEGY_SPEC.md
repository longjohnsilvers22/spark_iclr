# Grasp Strategy Selection — Design Spec

Status: DESIGN (no code in this document). Owning agent for `_grasp` /
`executor_grasp.py` / `executor_motion.py` implements the wiring later.

All file:line citations are against the repo at the time of writing.

---

## 1. Motivation & the revision

A prior design proposed making the **OBB-yaw** grasp the default. That is
wrong for the general case: OBB yaw is auto-triggered today purely by a mask
**aspect-ratio gate** (`GRASP_YAW_AR_GATE`, default `1.8` —
`spark_real/control/executor_core.py:66`), evaluated in the approach path at
`spark_real/control/executor_motion.py:339` (birdview) and `:399` (wrist). That
gate is unreliable: a plushie or any blob whose 2D mask happens to be elongated
trips `aspect_ratio >= 1.8` and yaws the wrist for no reason (the operator
already reaches for the `SPARK_YAW_AR_GATE` env escape hatch —
`executor_core.py:181-183` — to force plain top-down). So **shape-based
auto-routing must NOT default to OBB.**

Revised model: **three first-class, Gemini-selectable strategies**, with a
plain top-down grasp as the default and terminal fallback.

| strategy  | what it does | when |
|-----------|--------------|------|
| `topdown` | plain straight-down grasp at `GRASP_ORIENTATION` (`executor_core.py:46`, `[2.103,-2.329,0.059]`). No yaw. | **DEFAULT.** Compact/simple objects (blocks, balls, cups, bottles standing). |
| `obb`     | yaw the wrist to the object's mask OBB major axis before closing, so the jaws close across the **minor** axis. | Opt-in, for elongated **tools / objects with a long straight edge** (knife handle, screwdriver shaft, wrench, pen, ruler) where the mask OBB is a trustworthy grasp axis. |
| `cgn`     | 6-DOF Contact-GraspNet: generate free grasps on the object point cloud, select best, command the full pose. Supports **non-top-down** (tilted / horizontal) approaches. | Complex / irregular geometry, cluttered contact, or when the planner explicitly wants a non-top-down approach. |

`topdown` and `obb` are the two existing top-down code paths already unified
inside `_grasp` → `_grasp_v2` (Robotiq) via the persisted approach orientation
`_active_grasp_orient` (`executor_motion.py:419`, consumed at
`executor_grasp.py:285`). `cgn` is new selection logic that already exists as a
library (`spark_real/perception/cgn_grasp_select.py`) but is **not yet wired
into the `grasp` node**.

Note: a **separate** primitive `grasp_se3` already exists for pure side /
angled cylinder picks (`strategy: horizontal|angled`, documented in the planner
prompt at `spark_real/planning/spark_planner.py:132`). That is intentionally
left as its own node. `cgn` here is for the general 6-DOF / complex-geometry
case surfaced transparently through the ordinary `grasp` node — it is NOT a
replacement for `grasp_se3`'s hand-specified cylinder geometry.

---

## 2. First-class selection: a `strategy` param on the `grasp` node

Add one optional param to the existing `grasp` leaf (no new primitive, no new
node type — keeps `BUILTIN_PRIMITIVES` unchanged, `executor_types.py:24-25`):

```yaml
- type: grasp
  params:
    strategy: topdown        # topdown | obb | cgn   (optional; default auto-route)
    target_width: 0.03       # unchanged, still honored
    force: 60                # unchanged, still honored
```

`strategy` is precedent-compatible with the `strategy` param already used by
`grasp_se3` (`spark_planner.py:132`), so Gemini sees a consistent vocabulary.

### Planner-prompt wording (to add near `spark_planner.py:127-132`)

- Default: omit `strategy` (or `strategy: topdown`) for simple/compact objects
  — blocks, balls, cups, standing bottles, plush toys. Plain top-down.
- `strategy: obb` ONLY for **elongated tools with a clear straight long edge**
  whose grasp axis is the mask's long axis: knife handle, screwdriver shaft,
  wrench, pen, ruler, spatula handle. Prefer prompting SAM3 for the sub-part
  ("knife handle") so the mask/OBB is the graspable region.
- `strategy: cgn` for **irregular / complex geometry** (odd 3D shapes, handles,
  clutter) or when a **non-top-down approach** is desired but the object is not
  a clean upright cylinder (for clean cylinders keep using `grasp_se3`).
- When unsure, omit `strategy`; the executor auto-routes (Section 3) and every
  path degrades safely to plain top-down.

The planner must keep emitting `target_width`/`force` per the existing
per-category guidance (`spark_planner.py:117-124`); `strategy` is orthogonal.

---

## 3. Auto-route heuristic (when `strategy` unspecified) + fallbacks

Auto-route lives at the `grasp` boundary and is **conservative — it defaults to
`topdown` and only escalates on strong signals.** It must NOT reproduce the
current bare aspect-ratio auto-OBB behavior.

```
resolve_strategy(params, detection):
    if params.strategy in {topdown, obb, cgn}:
        return params.strategy                      # explicit wins

    # --- auto-route (no explicit choice) ---
    ar   = detection.aspect_ratio      (default 1.0)   # executor_grasp.py:218
    obb_minor_m = detection.obb_minor_m (default 0.0)  # executor_grasp.py:201

    # OBB is opt-in by SHAPE CONFIDENCE, not a bare AR gate:
    if ar >= OBB_AR_CONFIDENT (>= 3.0)      # much stricter than the 1.8 yaw gate
       and 0 < obb_minor_m <= OBB_MINOR_MAX (~0.035 m)   # thin enough to be a tool
       and detection.obb_confidence ok (elongated, non-degenerate mask):
        return obb

    # CGN for high-AR-but-not-tool / complex geometry, or planner asked non-topdown:
    if params.approach in {tilted, non_topdown} or detection.complex_geometry:
        return cgn

    return topdown                                    # DEFAULT
```

`OBB_AR_CONFIDENT >= 3.0` deliberately matches the existing "definitely a thin
tool" band already used for width derivation (`executor_grasp.py:223`,
`aspect_ratio >= 3.0`), and is far stricter than the `1.8` yaw gate that
misfires on plushies. If a mask is elongated but not thin (a plushie: high AR,
large `obb_minor_m`), it fails the `obb_minor_m <= OBB_MINOR_MAX` test and stays
`topdown`. Tunable via config/env (mirror the `SPARK_YAW_AR_GATE` override
mechanism, `executor_core.py:181-183`).

### Per-strategy viability & terminal fallback

Every strategy **degrades to plain top-down** (`GRASP_ORIENTATION`,
`executor_core.py:46`) as the terminal fallback. No strategy is allowed to
abort the grasp outright just because its own machinery failed.

- **topdown** — always viable; it IS the fallback. Uses existing
  `_grasp`/`_grasp_v2` descent + force-verify + descent-retry ladder
  (`executor_grasp.py:469`, `:233`, retries at `:589-652`).

- **obb** — viable only if the detection carries a usable
  `orientation_angle` + confident elongation. If the OBB is degenerate/missing,
  **fall back to topdown** (do not yaw). This is a stricter version of the
  existing gate at `executor_motion.py:339`. Everything downstream of the yaw is
  the same top-down descent/clamp/lift, reusing `_active_grasp_orient`
  (`executor_motion.py:419` → `executor_grasp.py:285`).

- **cgn** — call `generate_and_select(...)`
  (`spark_real/perception/cgn_grasp_select.py:154`) on the object's world-frame
  point cloud + segment id. **Veto → fall back to topdown** on any of:
  - no grasp returned (CGN produced nothing / segment empty) — `sel is None`
    (`cgn_grasp_select.py:193`);
  - low score below a min-confidence threshold (`sel["score"]`, sorted
    best-first in `_grasps_to_records`, `contact_graspnet_infer.py`);
  - **IK-unreachable**: selected 6-DOF pose fails the executor's IK / workspace
    check (reuse the same reachability path the executor already uses for
    oriented descent — pyroki IK in `executor_ik.py` / `_check_workspace`);
  - **xy-sanity**: selected grasp center is > a small radius (e.g. 5 cm) from
    the detection centroid in xy (guards against CGN latching onto a neighbor;
    mirrors the wrist-refine xy guard at `executor_motion.py:390-397`).

  Note `cgn_grasp_select` already has an internal top-down *override* mode
  (`mode == "override"`, `cgn_grasp_select.py:11-14`) that keeps the best CGN
  position but forces `grasp_orientation` when no sufficiently top-down grasp
  passes `TOPDOWN_ALIGN_THRESH` (`cgn_grasp_select.py:43`). For the `cgn`
  strategy we WANT the full 6-DOF pose (`mode == "cgn_topdown"` or a genuine
  tilted grasp), so the caller may pass a permissive `align_thresh` to allow
  non-top-down poses; the internal override remains the intra-CGN safety net,
  and the executor-level veto above is the outer net that lands on plain
  top-down.

Ordering of nets (outermost last): `cgn 6-DOF pose` → (CGN internal
top-down override) → (executor veto) → **plain topdown**.

---

## 4. Integration point — transparent at the `grasp` boundary

The grasp node dispatches to `self._grasp` at
`spark_real/control/executor_core.py:878-879` (and the `action`/meta path map at
`:866-874`). Robotiq goes `_grasp` → `_grasp_v2` (`executor_grasp.py:483`).

Because selection is resolved **inside the grasp boundary**, both consumers
benefit with **no new primitive**:

- **Hand-written / base-primitive BTs** (e.g. the pre-baked `bt_library`
  scores) get better grasps for free: omit `strategy` and rely on auto-route, or
  add `strategy:` to a specific node.
- **Gemini** gets an explicit, first-class knob it can set per grasp.

The approach orientation is already the single hand-off used by both top-down
paths: `_approach_target` (`executor_motion.py:327`) persists
`_active_grasp_orient` (`:419`) and `_grasp_v2` consumes it (`:285`). `cgn`
plugs into this same seam by writing the selected 6-DOF orientation (and target
xyz) into that hand-off, so the descent/clamp/lift machinery is unchanged.

---

## 5. Minimal wiring plan for `_grasp` (SPEC — for the owning agent)

Do NOT implement here. This is the intended minimal change set for the agent
who owns `executor_grasp.py` / `executor_motion.py`:

1. **Resolve strategy once, early.** In the approach entry (`_approach_target`,
   `executor_motion.py:327`) or at the top of `_grasp` (`executor_grasp.py:469`),
   call a small `resolve_strategy(params, detection)` helper (Section 3). Store
   `self._active_grasp_strategy`.

2. **topdown** — current behavior with the OBB auto-yaw at
   `executor_motion.py:339` / `:399` **disabled unless strategy == obb**. That
   is the key behavioral fix: gate the yaw on `strategy == "obb"` (explicit or
   auto-routed), NOT on the bare `aspect_ratio >= GRASP_YAW_AR_GATE` test alone.
   Keep `GRASP_YAW_AR_GATE` as a secondary confidence guard within the `obb`
   branch.

3. **obb** — keep the existing oriented-grasp code (`_oriented_grasp`,
   `executor_motion.py:169`; persist to `_active_grasp_orient`, `:419`). Add the
   degenerate-OBB → topdown fallback (Section 3).

4. **cgn** — new branch, invoked BEFORE the top-down descent commits:
   - build the object world cloud + segment id via
     `spark_real/perception/scene_cloud.py:build_scene_cloud` (`:25`) and
     `resolve_segment_id` (`:112`) — cloud is already in base frame;
   - `sel = generate_and_select(points, segment_labels, target_seg_id=...,
     grasp_orientation=self.GRASP_ORIENTATION, align_thresh=<permissive>)`
     (`cgn_grasp_select.py:154`);
   - run the vetoes (Section 3: None / low-score / IK-unreachable / xy-sanity);
   - on pass: set the approach target xyz = `sel["xyz"]` and
     `self._active_grasp_orient = sel["orient_rotvec"]` (the same hand-off
     `_grasp_v2` reads at `executor_grasp.py:285`), then let the existing
     oriented descent/clamp/lift run;
   - on veto: fall through to **topdown**.

5. **Config/env knobs** (mirror `executor_core.py:181-187`): `OBB_AR_CONFIDENT`,
   `OBB_MINOR_MAX`, CGN min-score, CGN xy-sanity radius, plus a global kill
   switch (e.g. `SPARK_GRASP_STRATEGY=topdown`) to force plain top-down for a
   run — analogous to `SPARK_GRASP_V2=0` (`executor_grasp.py:480`).

6. **Non-Robotiq / Franka** grippers keep the legacy `_grasp` path
   (`executor_grasp.py:484+`); `cgn`/`obb` routing should degrade to topdown
   there unless separately validated, matching the existing `_grasp_v2`
   Robotiq-only gating (`executor_grasp.py:479-483`).

### Invariants the wiring must preserve
- `strategy` is optional; absence == auto-route == usually topdown.
- No strategy aborts a grasp; the terminal fallback is always plain top-down.
- No new leaf primitive / no change to `BUILTIN_PRIMITIVES`
  (`executor_types.py:24-25`) — `strategy` is just a `grasp` param.
- `grasp_se3` (side/angled cylinders) is untouched and remains a separate node.

---

## 6. Key current-code references

| Concern | Location |
|---|---|
| grasp node → `_grasp` dispatch | `executor_core.py:878-879`, map `:866-874` |
| `_grasp` / Robotiq `_grasp_v2` | `executor_grasp.py:469` / `:233`, gate `:479-483` |
| top-down descent-retry ladder | `executor_grasp.py:589-652` |
| base `GRASP_ORIENTATION` | `executor_core.py:46` |
| current OBB auto-yaw (unreliable) | `executor_motion.py:339`, `:399` |
| yaw gate `GRASP_YAW_AR_GATE` (default 1.8) | `executor_core.py:66`; env override `:181-183` |
| `_oriented_grasp` (yaw helper) | `executor_motion.py:169` |
| approach orientation hand-off | `executor_motion.py:419` → `executor_grasp.py:285` |
| aspect_ratio / obb_minor_m in detection | `executor_grasp.py:201`, `:218`, `:223` |
| CGN selection library | `cgn_grasp_select.py:53` (`select_grasp`), `:154` (`generate_and_select`) |
| CGN modes / thresholds | `cgn_grasp_select.py:11-14`, `:41` (`CGN_TCP_OFFSET_M`), `:43` (`TOPDOWN_ALIGN_THRESH`) |
| CGN inference (6-DOF) | `spark_real/perception/contact_graspnet_infer.py` (`predict_grasps`) |
| world cloud + segment id builder | `scene_cloud.py:25` (`build_scene_cloud`), `:112` (`resolve_segment_id`) |
| planner grasp guidance / `strategy` precedent | `spark_planner.py:117-132` |
| builtin leaf primitives (single source) | `executor_types.py:24-25` |

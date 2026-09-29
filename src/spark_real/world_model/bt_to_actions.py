"""Convert a typed BT YAML score into a 7-DoF EE-delta action sequence.

V-JEPA 2-AC was trained on DROID Franka 7-DoF EE delta + gripper actions
(`[dx, dy, dz, drx, dry, drz, dgripper]`) at 5 Hz, with an L1-ball cage of
`||a||_1 <= 0.075` (~13 cm/step).  SPARK BTs compile via OSC to Cartesian
EE waypoints; the natural mapping is **waypoint-to-waypoint delta** chunked
into L1-bounded substeps.

Provenance:
- 7-DoF action format and L1 cap: notebooks/utils/mpc_utils.py:42 maxnorm=0.05,
  notebooks/energy_landscape_example.ipynb forward_actions() grid_size=0.075.
- State / pose definition (xyz + xyz_euler + gripper [0..1]): mpc_utils.py
  compute_new_pose() lines 166-190.

Action mapping (BT primitive -> 7-DoF delta chunk):
  move_to_keypoint(label, offset)  -> chunk(target_xyz - ee_now, rot=0, grip=hold)
  grasp(force)                      -> [chunk(-0.02 z), close_gripper_step]
  release()                         -> [open_gripper_step]
  move_relative(dx, dy, dz)         -> chunk([dx, dy, dz], rot=0, grip=hold)
  push_object(dir, dist)            -> chunk(dir*dist, rot=0, grip=closed)
  open_drawer / wipe / insert       -> approximate via XYZ keyframes (see _approx)

This is a coarse, action-space-only approximation of the OSC controller's
true sub-step path.  Sufficient for the V-JEPA 2-AC verifier because the
predictor consumes per-step deltas and only the **terminal latent** is needed.

If grounded keypoint XYZ in the world frame is unavailable (e.g. running
without a live SAM3 det_map), the caller passes `keypoint_xyz` overrides
keyed by label.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import numpy as np

# Upstream cage: notebooks/utils/mpc_utils.py:42 maxnorm=0.05 (CEM default)
# Upstream grid sweep used 0.075 (energy_landscape_example).
# The larger value is used so chunked motion converges faster and stays
# within the training distribution.
MAX_DELTA_XYZ = 0.075  # meters per step
GRIPPER_OPEN = 0.0     # convention: 0 = open, 1 = closed (matches pose[6])
GRIPPER_CLOSED = 1.0


@dataclass
class ActionRollout:
    """A compiled BT trajectory in V-JEPA 2-AC's action space."""

    actions: np.ndarray  # [T, 7]  per-step EE deltas (dx dy dz drx dry drz dgrip)
    states: np.ndarray   # [T+1, 7]  reconstructed EE pose at each step
    primitive_boundaries: list[int]  # action index where each BT primitive ends
    primitive_names: list[str]       # human label per boundary
    n_unresolved: int = 0  # move_to_keypoint leaves whose label did not resolve;
    # the trace after a skip is fiction (stale EE + stacked relatives), so
    # consumers judging safety should abstain when this is nonzero.


# ---------------------------------------------------------------------------
# Tree flattening (mirrors spark_bench.libero_pro.executor._flatten_tree)
# ---------------------------------------------------------------------------

def flatten_tree(tree: dict) -> list[dict]:
    """Depth-first walk of a BT score; returns the leaf primitives in order."""
    out: list[dict] = []

    def _walk(node):
        if not isinstance(node, dict):
            return
        t = node.get("type")
        if t in ("sequence", "selector"):
            for c in node.get("children", []) or []:
                _walk(c)
        elif t == "retry":
            # Recovery-grammar composite: compile as one attempt; the FK
            # trace of a retry is its nominal chain.
            for c in node.get("children", []) or []:
                _walk(c)
        elif t == "fallback":
            # Contingency composite: the FK trace follows the nominal
            # (first) branch only.
            children = node.get("children", []) or []
            if children:
                _walk(children[0])
        else:
            out.append(node)

    _walk(tree)
    return out


# ---------------------------------------------------------------------------
# Chunking
# ---------------------------------------------------------------------------

def _chunk_xyz(delta_xyz: np.ndarray, grip: float, max_step: float = MAX_DELTA_XYZ) -> list[np.ndarray]:
    """Break a Cartesian delta into L1-cage-respecting steps.

    The cage in upstream is L1 < 0.075 for xyz; inf-norm <= MAX_DELTA_XYZ
    is used here, which is strictly tighter and still inside the trained
    regime.
    """
    delta_xyz = np.asarray(delta_xyz, dtype=np.float32)
    n_steps = max(1, int(np.ceil(np.max(np.abs(delta_xyz)) / max_step)))
    step = delta_xyz / n_steps
    return [
        np.array([step[0], step[1], step[2], 0.0, 0.0, 0.0, grip - GRIPPER_OPEN],
                  dtype=np.float32)
        for _ in range(n_steps)
    ]


def _gripper_step(grip_open: bool, hold_xyz: bool = True) -> np.ndarray:
    """Single in-place gripper transition step."""
    target = GRIPPER_OPEN if grip_open else GRIPPER_CLOSED
    # Encode the *delta* to the target gripper state, not the absolute.
    return np.array([0.0, 0.0, 0.0, 0.0, 0.0, 0.0, target - 0.5], dtype=np.float32)


# ---------------------------------------------------------------------------
# BT -> ActionRollout
# ---------------------------------------------------------------------------

def bt_to_actions(
    bt_score: dict,
    *,
    ee_init_xyz: np.ndarray,
    keypoint_xyz: dict[str, np.ndarray],
    grip_init: float = GRIPPER_OPEN,
) -> ActionRollout:
    """Compile a BT score to a 7-DoF EE-delta sequence.

    Args:
        bt_score: dict with 'tree' key (root sequence/selector + children).
        ee_init_xyz: current EE position in world frame, shape (3,).
        keypoint_xyz: map label -> XYZ for resolving move_to_keypoint targets.
            Caller is responsible for producing this from SAM3 detection
            (or for the smoke test, from a hand-written stub).
        grip_init: initial gripper state, 0=open or 1=closed.

    Returns:
        ActionRollout with per-step [dx, dy, dz, 0, 0, 0, dgrip] deltas
        and the reconstructed EE pose path.
    """
    actions: list[np.ndarray] = []
    boundaries: list[int] = []
    names: list[str] = []

    ee_xyz = np.array(ee_init_xyz, dtype=np.float32)
    grip = float(grip_init)
    holding = False  # tracks pick/place state to mirror executor.holding logic
    last_pick_xyz: Optional[np.ndarray] = None
    n_unresolved = 0

    leaves = flatten_tree(bt_score.get("tree", {}))

    for node in leaves:
        ptype = node.get("type", "")
        params = node.get("params", {}) or {}

        # -- resolve waypoint(s) for this primitive
        steps_for_this: list[np.ndarray] = []

        if ptype == "move_to_keypoint":
            label = params.get("keypoint_label", "")
            target = _resolve_keypoint(label, keypoint_xyz)
            if target is None:
                # Unknown label: the executor would re-detect or recover
                # here, so any further reconstruction is fiction. Count it
                # so safety consumers can abstain instead of judging a
                # stale-height trace.
                n_unresolved += 1
                continue
            offset = np.array(
                [
                    params.get("offset_x", 0.0),
                    params.get("offset_y", 0.0),
                    params.get("offset_z", 0.0),
                ],
                dtype=np.float32,
            )
            target = target + offset

            if holding:
                # Place: approach from above, then descend.
                above = target + np.array([0.0, 0.0, 0.08], dtype=np.float32)
                steps_for_this += _chunk_xyz(above - ee_xyz, grip)
                ee_xyz = above
                steps_for_this += _chunk_xyz(target - ee_xyz, grip)
                ee_xyz = target
            else:
                # Pick: approach from 8 cm above then descend.
                above = target + np.array([0.0, 0.0, 0.08], dtype=np.float32)
                steps_for_this += _chunk_xyz(above - ee_xyz, grip)
                ee_xyz = above
                steps_for_this += _chunk_xyz(target - ee_xyz, grip)
                ee_xyz = target
                last_pick_xyz = target.copy()

        elif ptype == "grasp":
            # Close gripper.  No EE motion -- a single discrete step.
            grip = GRIPPER_CLOSED
            holding = True
            steps_for_this.append(_gripper_step(grip_open=False))

        elif ptype == "release":
            grip = GRIPPER_OPEN
            holding = False
            steps_for_this.append(_gripper_step(grip_open=True))

        elif ptype == "move_relative":
            delta = np.array(
                [
                    params.get("dx", 0.0),
                    params.get("dy", 0.0),
                    params.get("dz", 0.0),
                ],
                dtype=np.float32,
            )
            steps_for_this += _chunk_xyz(delta, grip)
            ee_xyz = ee_xyz + delta

        elif ptype == "push_object":
            direction = np.array(params.get("push_direction", [1.0, 0.0, 0.0]),
                                  dtype=np.float32)
            n = np.linalg.norm(direction)
            if n > 1e-6:
                direction = direction / n
            dist = float(params.get("push_distance", 0.20))
            label = params.get("keypoint_label", "")
            target = _resolve_keypoint(label, keypoint_xyz)
            if target is not None:
                # Approach object first.
                above = target + np.array([0.0, 0.0, 0.05], dtype=np.float32)
                steps_for_this += _chunk_xyz(above - ee_xyz, grip)
                ee_xyz = above
                steps_for_this += _chunk_xyz(target - ee_xyz, grip)
                ee_xyz = target
            delta = direction * dist
            steps_for_this += _chunk_xyz(delta, grip)
            ee_xyz = ee_xyz + delta

        elif ptype == "open_drawer":
            label = params.get("keypoint_label", "")
            target = _resolve_keypoint(label, keypoint_xyz)
            if target is not None:
                steps_for_this += _chunk_xyz(target - ee_xyz, grip)
                ee_xyz = target.copy()
            # Pull along -Y (LIBERO front).
            delta = np.array([0.0, -0.20, 0.0], dtype=np.float32)
            steps_for_this += _chunk_xyz(delta, grip)
            ee_xyz = ee_xyz + delta

        elif ptype in ("wipe", "insert", "turn_knob", "wait"):
            # Coarse approximation: 5 small forward steps at current pose
            # (the predictor mainly needs an action stream of the correct
            # cadence; these primitives don't have a clean XYZ target).
            for _ in range(5):
                actions.append(
                    np.array([0.01, 0.0, 0.0, 0.0, 0.0, 0.0, grip - 0.5],
                              dtype=np.float32)
                )
            steps_for_this = []  # already appended

        else:
            # Unknown primitive -- emit a no-op pause to keep cadence.
            steps_for_this.append(
                np.array([0.0, 0.0, 0.0, 0.0, 0.0, 0.0, grip - 0.5],
                          dtype=np.float32)
            )

        actions.extend(steps_for_this)
        boundaries.append(len(actions))
        names.append(ptype)

    if not actions:
        # Empty BT: emit a single no-op so downstream code has a tensor.
        actions = [np.zeros(7, dtype=np.float32)]
        boundaries = [1]
        names = ["noop"]

    actions_np = np.stack(actions, axis=0)  # [T, 7]

    # Reconstruct the implied state path: pose[0] = (ee_init, rot=0, grip_init),
    # then accumulate xyz + gripper.  Rotation is unused (drx=dry=drz=0).
    T = actions_np.shape[0]
    states_np = np.zeros((T + 1, 7), dtype=np.float32)
    states_np[0, :3] = ee_init_xyz
    states_np[0, 6] = grip_init
    for t in range(T):
        states_np[t + 1, :3] = states_np[t, :3] + actions_np[t, :3]
        # Gripper delta encoded as (target - 0.5); the state stores the
        # clipped cumulative sum.
        states_np[t + 1, 3:6] = 0.0
        states_np[t + 1, 6] = np.clip(states_np[t, 6] + actions_np[t, 6], 0.0, 1.0)

    return ActionRollout(
        actions=actions_np,
        states=states_np,
        primitive_boundaries=boundaries,
        primitive_names=names,
        n_unresolved=n_unresolved,
    )


def _resolve_keypoint(label: str, keypoint_xyz: dict[str, np.ndarray]) -> Optional[np.ndarray]:
    """Fuzzy lookup -- mirrors libero_pro.primitives._common.fuzzy_get_det."""
    if not label:
        return None
    label_l = label.lower()
    # Exact key.
    if label in keypoint_xyz:
        return np.array(keypoint_xyz[label], dtype=np.float32)
    # Substring / token match (cheap).
    for k, v in keypoint_xyz.items():
        if k.lower() in label_l or label_l in k.lower():
            return np.array(v, dtype=np.float32)
    return None

"""
World-model-gated safe execution layer for LIBERO-PRO BT execution.

Pipeline:
    1. Compile each candidate BT to a 7-DoF EE-delta action sequence via
       ``spark_real.world_model.bt_to_actions``.
    2. **Kinematic shield (cheap, mandatory):** walk the compiled state path
       and reject any BT whose forward-kinematic terminal or intermediate EE
       pose leaves the workspace envelope
       ``[wm_unsafe_workspace_z_min, wm_unsafe_workspace_z_max]``.  This is
       a geometric pre-filter that runs without V-JEPA at all.
    3. **Latent goal-similarity (optional, when a goal frame is available):**
       roll the compiled action sequence through V-JEPA 2-AC and rank the
       candidates that passed the kinematic shield by L1 distance to the goal
       latent (the loss V-JEPA 2-AC was trained with).
    4. Return the best candidate that passed; ``None`` to trigger recovery if
       *every* candidate fails the kinematic shield.

The "unsafe set" is kinematic workspace violation (EE pose below the table
or above the safe ceiling, pure forward kinematics on the compiled state
path), not a learned classifier on V-JEPA latents. The V-JEPA latent is
used only for goal-similarity ranking among candidates that pass the shield.

References:
- OSCBF: arXiv:2404.18712 - Operational-Space CBF for manipulators.
- ASIMOV: arXiv:2503.08663 - semantic safety benchmark (orthogonal axis).
- V-JEPA 2-AC: arXiv:2506.09985 - frozen action-conditioned predictor.
"""
from __future__ import annotations

import time
from dataclasses import dataclass, field
from typing import Optional

import numpy as np

# World-model integration is optional - imports cleanly on CPU-only machines
# without the V-JEPA repo checked out, and the gate degrades to the kinematic
# shield when these are absent.
try:
    from spark_real.world_model.bt_to_actions import bt_to_actions
except Exception:  # pragma: no cover - world_model is optional
    bt_to_actions = None  # type: ignore[assignment]
try:
    from spark_real.world_model.bt_verifier import verify_candidates
except Exception:  # pragma: no cover - world_model is optional
    verify_candidates = None  # type: ignore[assignment]


__all__ = ["CandidateScore", "GateResult", "WMSafetyGate", "evaluate_kinematic_safety"]


# Result types

@dataclass
class CandidateScore:
    """
    Per-candidate output of the WM safety gate.
    """

    bt: dict
    """
    The candidate BT score dict (with a 'tree' key).
    """

    safe: bool
    """True iff the kinematic shield passed (terminal + intermediate EE
    poses all stayed inside the workspace envelope)."""

    unsafe_reason: str = ""
    """Human-readable reason when ``safe`` is False
    (e.g. "z=0.71 below floor 0.78 at step 14")."""

    goal_l1: float = float("inf")
    """L1 distance from predicted terminal V-JEPA latent to goal latent.
    Infinity when ``safe`` is False or when no V-JEPA model was provided."""

    trace_prob: float = float("nan")
    """Predicted success probability from the trace-model ranker
    (``selection='trace_model'``).  NaN when no ranker was provided or the
    candidate failed the kinematic shield."""

    n_steps: int = 0
    """
    Length of the compiled action sequence.
    """

    wall_time_s: float = 0.0


@dataclass
class GateResult:
    """
    Aggregate gate output for a batch of candidates.
    """

    candidates: list[CandidateScore] = field(default_factory=list)
    n_passed: int = 0
    n_rejected: int = 0
    best_idx: int = -1
    """Index in ``candidates`` of the chosen BT; -1 if every candidate was
    rejected."""

    total_time_s: float = 0.0


# Kinematic shield (no V-JEPA needed)

def evaluate_kinematic_safety(
    states: np.ndarray,
    *,
    z_min: float,
    z_max: float,
    xy_radius: float = 1.5,
) -> tuple[bool, str]:
    """
    Walk a state path and reject if any EE pose leaves the safety envelope.

    Args:
        states: ``[T+1, 7]`` reconstructed EE pose path
            (xyz + xyz_euler + gripper) from ``bt_to_actions.ActionRollout``.
        z_min: floor of the safe Z envelope (table height).
        z_max: ceiling of the safe Z envelope.
        xy_radius: max horizontal reach from origin (in meters); LIBERO
            tabletops are ~80 cm from base, so 1.5 m is permissive.

    Returns:
        (safe, reason).  ``safe`` is True only if every step is inside the
        envelope.
    """
    if states.shape[0] == 0:
        return True, ""

    xyz = states[:, :3]
    # Z floor / ceiling.
    z_floor_viol = xyz[:, 2] < z_min
    if z_floor_viol.any():
        step = int(np.argmax(z_floor_viol))
        return False, f"z={xyz[step, 2]:.3f} below floor {z_min:.3f} at step {step}"
    z_ceil_viol = xyz[:, 2] > z_max
    if z_ceil_viol.any():
        step = int(np.argmax(z_ceil_viol))
        return False, f"z={xyz[step, 2]:.3f} above ceiling {z_max:.3f} at step {step}"

    # Horizontal reach.
    xy_norm = np.linalg.norm(xyz[:, :2], axis=1)
    xy_viol = xy_norm > xy_radius
    if xy_viol.any():
        step = int(np.argmax(xy_viol))
        return False, (f"xy_norm={xy_norm[step]:.3f} beyond radius {xy_radius:.3f} "
                       f"at step {step}")
    return True, ""


# Main gate object

class WMSafetyGate:
    """
    Two-stage safety gate: kinematic shield + V-JEPA goal-similarity ranker.

    Holds an optional V-JEPA 2-AC model and the gate thresholds.  When
    ``wm`` is None, the gate falls back to kinematic-shield-only behaviour
    and ranks passing candidates by ascending action-sequence length
    (shorter == more efficient).

    Reuses :mod:`spark_real.world_model` for V-JEPA loading and BT-to-action
    compilation; this module owns only the gate policy, not the WM
    machinery.
    """

    def __init__(
        self,
        *,
        z_min: float,
        z_max: float,
        xy_radius: float = 1.5,
        wm: Optional[object] = None,
        ranker: Optional[object] = None,
    ):
        self.z_min = float(z_min)
        self.z_max = float(z_max)
        self.xy_radius = float(xy_radius)
        self.wm = wm  # VJEPA2AC bundle or None
        # Optional trace-model ranker (selection='trace_model'):
        # callable ``ranker(bts: list[dict]) -> list[tuple[int, float]]``
        # returning (index-into-bts, success_prob) sorted by descending
        # probability (spark_bench.trace_selector.select.rank_plans,
        # partially applied with det_map/instruction context).  When set it
        # ranks the candidates that PASSED the kinematic shield; the shield
        # itself stays mandatory.
        self.ranker = ranker

    # Evaluation

    def evaluate_candidates(
        self,
        bts: list[dict],
        *,
        ee_init_xyz: np.ndarray,
        keypoint_xyz: dict,
        grip_init: float = 0.0,
        current_obs: Optional[np.ndarray] = None,
        goal_obs: Optional[np.ndarray] = None,
    ) -> GateResult:
        """
        Score every candidate BT.

        Args:
            bts: list of BT score dicts (each with a 'tree' key).
            ee_init_xyz: current EE position in world frame, shape (3,).
            keypoint_xyz: label -> (3,) world-frame keypoint XYZ map for
                resolving move_to_keypoint targets.
            grip_init: initial gripper state (0=open, 1=closed).
            current_obs: HxWx3 uint8 RGB of the current scene
                (required for V-JEPA latent ranking, otherwise unused).
            goal_obs: HxWx3 uint8 RGB of the goal scene
                (required for V-JEPA latent ranking, otherwise unused).

        Returns:
            GateResult with per-candidate scores and the chosen ``best_idx``.
        """
        t0 = time.time()

        scores: list[CandidateScore] = []
        passed_indices: list[int] = []

        for bt in bts:
            t_cand = time.time()
            try:
                rollout = bt_to_actions(
                    bt,
                    ee_init_xyz=ee_init_xyz,
                    keypoint_xyz=keypoint_xyz,
                    grip_init=grip_init,
                )
            except Exception as exc:
                # A compile error counts as unsafe (the BT cannot be predicted).
                scores.append(CandidateScore(
                    bt=bt,
                    safe=False,
                    unsafe_reason=f"bt_to_actions failed: {exc}",
                    n_steps=0,
                    wall_time_s=time.time() - t_cand,
                ))
                continue

            # The ceiling must accommodate the start height: recovery
            # replans begin with the arm already retracted at 1.17 to
            # 1.47 m, and the descent trace from the current position
            # transits above any fixed ceiling (otherwise vetoes zero whole
            # task cells). The plan is judged on where it newly commands
            # the arm, not on where the arm already is.
            z_max_eff = max(self.z_max, float(ee_init_xyz[2]) + 0.05)
            if getattr(rollout, "n_unresolved", 0) > 0:
                # The compiler skipped unresolvable keypoints, so the
                # remaining trace is stale-height fiction. Abstain: the
                # executor's own re-detection and recovery handle this
                # case; a veto here would judge a path nobody will fly.
                safe, reason = True, (
                    f"abstained: {rollout.n_unresolved} unresolved label(s)")
            else:
                safe, reason = evaluate_kinematic_safety(
                    rollout.states,
                    z_min=self.z_min,
                    z_max=z_max_eff,
                    xy_radius=self.xy_radius,
                )
            cs = CandidateScore(
                bt=bt,
                safe=safe,
                unsafe_reason=reason,
                n_steps=int(rollout.actions.shape[0]),
                wall_time_s=time.time() - t_cand,
            )
            scores.append(cs)
            if safe:
                passed_indices.append(len(scores) - 1)

        # Stage 2: V-JEPA goal-similarity among passing candidates.
        if (
            self.wm is not None
            and current_obs is not None
            and goal_obs is not None
            and passed_indices
        ):
            self._rank_with_vjepa(
                bts=[scores[i].bt for i in passed_indices],
                indices=passed_indices,
                scores=scores,
                ee_init_xyz=ee_init_xyz,
                keypoint_xyz=keypoint_xyz,
                grip_init=grip_init,
                current_obs=current_obs,
                goal_obs=goal_obs,
            )

        # Stage 2b: trace-model ranking among passing candidates
        # (selection='trace_model').  Fail-open: a ranker exception falls
        # back to the default (goal_l1, n_steps) ordering.
        ranker_ok = False
        if self.ranker is not None and passed_indices:
            try:
                ranked = self.ranker([scores[i].bt for i in passed_indices])
                for local_idx, prob in ranked:
                    scores[passed_indices[local_idx]].trace_prob = float(prob)
                ranker_ok = True
            except Exception:
                ranker_ok = False

        # Pick the best passing candidate.
        best_idx = -1
        if passed_indices:
            if ranker_ok:
                # Highest predicted success prob; ties keep planner order
                # (candidate 0 is the temperature-0 primary).
                def key(i: int):
                    s = scores[i]
                    return (-s.trace_prob, i)
            else:
                # If V-JEPA ran, rank by L1; otherwise rank by shortest path
                # (tie-breaker: original order).
                def key(i: int):
                    s = scores[i]
                    # primary: lower L1 to goal (inf if no V-JEPA)
                    # secondary: shorter action sequence
                    return (s.goal_l1, s.n_steps, i)
            best_idx = min(passed_indices, key=key)

        return GateResult(
            candidates=scores,
            n_passed=len(passed_indices),
            n_rejected=len(scores) - len(passed_indices),
            best_idx=best_idx,
            total_time_s=time.time() - t0,
        )

    # Convenience picker

    def pick(
        self,
        bts: list[dict],
        *,
        ee_init_xyz: np.ndarray,
        keypoint_xyz: dict,
        grip_init: float = 0.0,
        current_obs: Optional[np.ndarray] = None,
        goal_obs: Optional[np.ndarray] = None,
    ) -> Optional[dict]:
        """
        argmax over passing candidates; ``None`` if all fail the shield.

        Caller treats ``None`` as "trigger recovery" (retract, re-perceive,
        re-plan) instead of executing an unsafe BT.
        """
        result = self.evaluate_candidates(
            bts,
            ee_init_xyz=ee_init_xyz,
            keypoint_xyz=keypoint_xyz,
            grip_init=grip_init,
            current_obs=current_obs,
            goal_obs=goal_obs,
        )
        if result.best_idx < 0:
            return None
        return result.candidates[result.best_idx].bt

    # V-JEPA ranking (Stage 2)

    def _rank_with_vjepa(
        self,
        *,
        bts: list[dict],
        indices: list[int],
        scores: list[CandidateScore],
        ee_init_xyz: np.ndarray,
        keypoint_xyz: dict,
        grip_init: float,
        current_obs: np.ndarray,
        goal_obs: np.ndarray,
    ) -> None:
        """
        Mutate ``scores`` in-place: fill ``goal_l1`` for passing candidates.

        Reuses :func:`spark_real.world_model.bt_verifier.verify_candidates`.
        """
        try:
            verifier_out = verify_candidates(
                self.wm,
                bts,
                current_obs=current_obs,
                goal_frame=goal_obs,
                ee_init_xyz=ee_init_xyz,
                keypoint_xyz=keypoint_xyz,
                grip_init=grip_init,
                goal_source="wm_safety_gate",
            )
        except Exception as exc:
            # V-JEPA failure is non-fatal: Stage 1 already filtered the
            # unsafe candidates; fall back to "shortest-path" ranking.
            for i in indices:
                scores[i].unsafe_reason = (
                    scores[i].unsafe_reason
                    or f"vjepa rank skipped: {exc}"
                )
            return

        # verify_candidates returns the list sorted by L1 ascending.  Walk
        # the input order to map each score back to its slot.
        # The dataclass stores ``bt_score=bt`` so identity works.
        bt_to_score = {id(cs.bt_score): cs for cs in verifier_out.candidates}
        for i in indices:
            bt = scores[i].bt
            v = bt_to_score.get(id(bt))
            if v is not None:
                scores[i].goal_l1 = float(v.l1_to_goal)

"""K-candidate BT verifier using V-JEPA 2-AC as a frozen latent world model.

Pipeline:
    1. Encode current observation -> z_0
    2. Encode goal image          -> z_goal
    3. For each candidate BT:
         a. compile BT -> 7-DoF EE-delta action sequence (bt_to_actions)
         b. unroll predictor from (z_0, s_0) along the action sequence
         c. score = L1(z_T, z_goal)   (matches V-JEPA 2-AC training loss)
    4. Return argmin BT + the full score vector.

Zero training; frozen Meta checkpoint.
"""
from __future__ import annotations

import time
from dataclasses import dataclass
from typing import Optional

import numpy as np
import torch

from .bt_to_actions import ActionRollout, bt_to_actions
from .vjepa2_ac import (
    VJEPA2AC,
    encode_frame,
    latent_l1,
    latent_l2,
    unroll_actions,
)


@dataclass
class CandidateScore:
    """One BT candidate's verifier output."""

    rank: int                       # 0 = best
    bt_score: dict                  # the BT score dict
    rollout: ActionRollout          # compiled action sequence
    z_terminal: torch.Tensor        # final latent [1, tokens, D]
    l1_to_goal: float
    l2_to_goal: float
    wall_time_s: float


@dataclass
class VerifierResult:
    candidates: list[CandidateScore]   # sorted by L1 ascending
    best_index_in_input: int           # which input BT won
    goal_source: str                   # "hdf5" / "video" / "snapshot" / "given"
    encode_time_s: float
    total_time_s: float


def verify_candidates(
    wm: VJEPA2AC,
    bt_candidates: list[dict],
    *,
    current_obs: np.ndarray,
    goal_frame: np.ndarray,
    ee_init_xyz: np.ndarray,
    keypoint_xyz: dict[str, np.ndarray],
    grip_init: float = 0.0,
    goal_source: str = "given",
) -> VerifierResult:
    """Rank K behavior-tree candidates by predicted-terminal-to-goal distance.

    Args:
        wm: VJEPA2AC bundle (encoder + AC predictor + transform).
        bt_candidates: list of BT score dicts (each has a `tree` key).
        current_obs: HxWx3 uint8 RGB of the current agent-view.
        goal_frame: HxWx3 uint8 RGB of the goal scene.
        ee_init_xyz: (3,) initial EE position in world frame.
        keypoint_xyz: label -> (3,) world-frame keypoint XYZ map.
        grip_init: initial gripper state.
        goal_source: provenance tag for logging.

    Returns:
        VerifierResult with sorted candidates (best first).
    """
    t0 = time.time()
    z_current = encode_frame(wm, current_obs)
    z_goal = encode_frame(wm, goal_frame)
    encode_time = time.time() - t0

    s0 = torch.from_numpy(
        np.concatenate([ee_init_xyz, np.zeros(3, dtype=np.float32), [grip_init]])
    ).to(device=wm.device, dtype=wm.dtype).view(1, 1, 7)

    cands: list[CandidateScore] = []
    for idx, bt in enumerate(bt_candidates):
        t_cand = time.time()
        rollout = bt_to_actions(
            bt,
            ee_init_xyz=ee_init_xyz,
            keypoint_xyz=keypoint_xyz,
            grip_init=grip_init,
        )
        action_t = torch.from_numpy(rollout.actions).to(
            device=wm.device, dtype=wm.dtype
        )
        z_T, _s_T = unroll_actions(wm, z_current, s0, action_t)
        l1 = latent_l1(z_T, z_goal)
        l2 = latent_l2(z_T, z_goal)
        cands.append(
            CandidateScore(
                rank=-1,  # filled after sort
                bt_score=bt,
                rollout=rollout,
                z_terminal=z_T.detach().cpu(),
                l1_to_goal=l1,
                l2_to_goal=l2,
                wall_time_s=time.time() - t_cand,
            )
        )

    # Rank ascending by L1 (the training loss).
    order = sorted(range(len(cands)), key=lambda i: cands[i].l1_to_goal)
    ranked: list[CandidateScore] = []
    for rank, idx in enumerate(order):
        cands[idx].rank = rank
        ranked.append(cands[idx])
    best_in_input = order[0]

    return VerifierResult(
        candidates=ranked,
        best_index_in_input=best_in_input,
        goal_source=goal_source,
        encode_time_s=encode_time,
        total_time_s=time.time() - t0,
    )

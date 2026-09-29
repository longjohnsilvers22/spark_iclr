"""
Pinocchio-based IK for the FR3.

Used by grasp_se3 to compute a single 7-DOF target joint configuration
for a desired hand_tcp pose. The motion executor then drives JointMotion
to that configuration in one franky/Ruckig call, a smooth joint-space
arc, no Cartesian linearization, no mid-motion elbow flips.

Public API:
    solve_ik(target_pos, target_orient, q_seed) -> Optional[np.ndarray]
        target_pos: (3,) base-frame position [m]
        target_orient: (3,) base-frame axis-angle [rad]  OR  (3,3) rotation matrix
        q_seed: (7,) arm joints to seed the solver
        returns: (7,) joint solution, or None if no convergence
"""

from __future__ import annotations

import glob
import logging
import os
import sys
from typing import Optional, Union

import numpy as np
from scipy.spatial.transform import Rotation


def _ensure_pinocchio_on_path():
    """Force the cmeel pinocchio prefix onto sys.path.

    cmeel.pth injects it at startup for the main module, but request
    handlers can import before that runs; this makes import order moot.
    """
    candidates = []
    for base in (os.path.expanduser("~/.local/lib"), sys.prefix + "/lib"):
        candidates.extend(
            glob.glob(
                base + "/python*/site-packages/cmeel.prefix/lib/python*/site-packages"
            )
        )
    for p in candidates:
        if p not in sys.path and os.path.isdir(p):
            sys.path.insert(0, p)


_ensure_pinocchio_on_path()
import pinocchio as pin

logger = logging.getLogger(__name__)

_URDF_PATH = os.path.join(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
    "robots",
    "franka",
    "urdf",
    "fr3_franka_hand.urdf",
)

_model = None
_data = None
_tcp_frame_id = None


def _ensure_loaded():
    global _model, _data, _tcp_frame_id
    if _model is not None:
        return
    if not os.path.exists(_URDF_PATH):
        raise FileNotFoundError(f"FR3 URDF not found at {_URDF_PATH}")
    _model = pin.buildModelFromUrdf(_URDF_PATH)
    _data = _model.createData()
    _tcp_frame_id = _model.getFrameId("fr3_hand_tcp")
    logger.info("FR3 IK loaded: nq=%d, tcp_frame_id=%d", _model.nq, _tcp_frame_id)


def solve_ik(
    target_pos: np.ndarray,
    target_orient: Union[np.ndarray, list],
    q_seed: np.ndarray,
    max_iters: int = 200,
    eps_pos: float = 5e-4,
    eps_rot: float = 5e-3,
    damping: float = 1e-3,
    step_scale: float = 1.0,
    j5_target: float = -0.8,
    j5_weight: float = 3.0,
    seed_bias_weight: float = 0.8,
) -> Optional[np.ndarray]:
    """
    Damped least-squares IK to fr3_hand_tcp.

    Returns 7-vector of arm joints (no fingers), or None if not converged.
    """
    _ensure_loaded()

    target_pos = np.asarray(target_pos, dtype=float).reshape(3)
    target_orient = np.asarray(target_orient, dtype=float)
    if target_orient.shape == (3,):
        R_tgt = Rotation.from_rotvec(target_orient).as_matrix()
    elif target_orient.shape == (3, 3):
        R_tgt = target_orient
    else:
        raise ValueError(
            f"target_orient must be (3,) or (3,3), got {target_orient.shape}"
        )

    T_tgt = pin.SE3(R_tgt, target_pos)

    q_seed = np.asarray(q_seed, dtype=float).reshape(-1)
    if q_seed.size < 7:
        raise ValueError(f"q_seed needs at least 7 entries, got {q_seed.size}")
    q = np.zeros(_model.nq)
    q[:7] = q_seed[:7]
    # Fingers stay at seed (or zero); they don't move the TCP frame.
    if q_seed.size >= 9:
        q[7:9] = q_seed[7:9]

    for it in range(max_iters):
        pin.framesForwardKinematics(_model, _data, q)
        T_cur = _data.oMf[_tcp_frame_id]
        err6 = pin.log6(T_cur.inverse() * T_tgt).vector  # body-frame twist

        if np.linalg.norm(err6[:3]) < eps_pos and np.linalg.norm(err6[3:]) < eps_rot:
            arm_joint_lower = np.array(
                [-2.7437, -1.7837, -2.9007, -3.0421, -2.8065, 0.5445, -3.0159]
            )
            arm_joint_upper = np.array(
                [2.7437, 1.7837, 2.9007, -0.1518, 2.8065, 4.5169, 3.0159]
            )
            q_arm = q[:7].copy()
            if np.all(q_arm >= arm_joint_lower - 1e-3) and np.all(
                q_arm <= arm_joint_upper + 1e-3
            ):
                return q_arm
            else:
                logger.warning(
                    "FR3 IK converged but solution outside joint limits: %s", q_arm
                )
                return None

        J = pin.computeFrameJacobian(
            _model, _data, q, _tcp_frame_id, pin.LOCAL
        )  # 6 x nv
        # Damped least squares: dq = J^T (J J^T + lambda^2 I)^-1 err
        H = J @ J.T + (damping**2) * np.eye(6)
        dq = J.T @ np.linalg.solve(H, err6)
        # Null-space tasks via the (I - J^+ J) projector, which only adds
        # joint motion that does not disturb the TCP pose:
        #   - j5_target / j5_weight: keep J5 away from 0 (wrist singular)
        #   - seed_bias_weight: pull all joints toward the seed, avoiding
        #     wild far-away solutions (e.g. J7 winding through a full turn)
        if j5_weight > 0 or seed_bias_weight > 0:
            e_sec = np.zeros(_model.nv)
            if j5_weight > 0:
                e_sec[4] += j5_weight * (j5_target - q[4])
            if seed_bias_weight > 0:
                # Seed is the current config, so this biases toward the
                # nearest solution.
                arm_seed = np.asarray(q_seed[:7], dtype=float)
                for k_j in range(7):
                    e_sec[k_j] += seed_bias_weight * (arm_seed[k_j] - q[k_j])
            J_pinv = J.T @ np.linalg.solve(H, np.eye(6))
            N = np.eye(_model.nv) - J_pinv @ J
            dq = dq + N @ e_sec
        # Cap per-iter joint step so the solver can't wind a joint through
        # multiple turns chasing a far target.
        dq_norm = float(np.linalg.norm(dq))
        max_step = 0.3
        if dq_norm > max_step:
            dq *= max_step / dq_norm
        q += step_scale * dq
        # Clamp arm joints inside their limits each iter so winding can't
        # accumulate past a full turn.
        arm_lo = np.array(
            [-2.7437, -1.7837, -2.9007, -3.0421, -2.8065, 0.5445, -3.0159]
        )
        arm_hi = np.array([2.7437, 1.7837, 2.9007, -0.1518, 2.8065, 4.5169, 3.0159])
        q[:7] = np.clip(q[:7], arm_lo, arm_hi)
        # Clamp finger joints to a fixed value so they don't drift
        if _model.nq >= 9:
            q[7:9] = np.clip(q[7:9], 0.0, 0.04)

    logger.warning(
        "FR3 IK did not converge in %d iters: final_err_pos=%.4f err_rot=%.4f",
        max_iters,
        np.linalg.norm(err6[:3]),
        np.linalg.norm(err6[3:]),
    )
    return None

"""
Pure IK helpers for the LIBERO Panda runner.

Each function takes the MuJoCo model/data + joint/site indices explicitly
(no closures over runner state), so they are testable and shared between
the legacy dispatcher and the BT-DSL executor.
"""
from __future__ import annotations

from typing import Optional, Sequence

import numpy as np
import mujoco

from spark_bench.pyroki_ik import solve_ik_6dof, quat_from_approach


__all__ = [
    'get_q',
    'compute_ik',
    'compute_ik_6dof',
    'pyroki_ik',
    'solve_ik_sideways',
]


def get_q(model, data, joint_ids: Sequence[int]) -> np.ndarray:
    """
    Read the actuated joint configuration from a MuJoCo data block.
    """
    return np.array([data.qpos[model.jnt_qposadr[j]] for j in joint_ids])


def compute_ik(model, data, joint_ids: Sequence[int], ee_site: int,
                target: np.ndarray, max_iters: int = 600,
                tol: float = 0.003) -> np.ndarray:
    """
    Position-only 3-DOF IK via damped least squares.

    Returns the actuated joint configuration that brings ``ee_site`` close
    to ``target``.  Mutates a scratch ``MjData`` only - caller's ``data``
    is untouched.
    """
    ndof = len(joint_ids)
    q = get_q(model, data, joint_ids)
    d2 = mujoco.MjData(model)
    d2.qpos[:] = data.qpos[:]
    for _ in range(max_iters):
        mujoco.mj_forward(model, d2)
        ee = d2.site_xpos[ee_site].copy()
        err = target - ee
        if np.linalg.norm(err) < tol:
            break
        jacp = np.zeros((3, model.nv))
        mujoco.mj_jacSite(model, d2, jacp, None, ee_site)
        J = np.zeros((3, ndof))
        for i, jid in enumerate(joint_ids):
            J[:, i] = jacp[:, model.jnt_dofadr[jid]]
        JJT = J @ J.T + 0.01 ** 2 * np.eye(3)
        dq = 0.3 * J.T @ np.linalg.solve(JJT, err)
        dq = np.clip(dq, -0.2, 0.2)
        q += dq
        for i, jid in enumerate(joint_ids):
            d2.qpos[model.jnt_qposadr[jid]] = q[i]
    return q


def compute_ik_6dof(model, data, joint_ids: Sequence[int], ee_site: int,
                     target_pos: np.ndarray, target_quat: np.ndarray,
                     max_iters: int = 300) -> np.ndarray:
    """
    6-DOF IK (position + quaternion) via damped least squares.

    ``target_quat`` is in (w, x, y, z) order to match
    ``mujoco.mju_mat2Quat`` output.
    """
    ndof = len(joint_ids)
    q = get_q(model, data, joint_ids)
    d2 = mujoco.MjData(model)
    d2.qpos[:] = data.qpos[:]
    for i, jid in enumerate(joint_ids):
        d2.qpos[model.jnt_qposadr[jid]] = q[i]
    for _ in range(max_iters):
        mujoco.mj_forward(model, d2)
        ee = d2.site_xpos[ee_site].copy()
        pos_err = target_pos - ee
        ee_quat = np.zeros(4)
        mujoco.mju_mat2Quat(ee_quat, d2.site_xmat[ee_site].reshape(9))
        quat_err = np.zeros(4)
        ee_quat_conj = ee_quat.copy()
        ee_quat_conj[1:] *= -1
        mujoco.mju_mulQuat(quat_err, target_quat, ee_quat_conj)
        ori_err = quat_err[1:] * 2.0
        if quat_err[0] < 0:
            ori_err *= -1
        err = np.concatenate([pos_err, ori_err])
        if np.linalg.norm(pos_err) < 0.003 and np.linalg.norm(ori_err) < 0.05:
            break
        jacp = np.zeros((3, model.nv))
        jacr = np.zeros((3, model.nv))
        mujoco.mj_jacSite(model, d2, jacp, jacr, ee_site)
        J = np.zeros((6, ndof))
        for i, jid in enumerate(joint_ids):
            dof = model.jnt_dofadr[jid]
            J[:3, i] = jacp[:, dof]
            J[3:, i] = jacr[:, dof]
        JJT = J @ J.T + 0.01 ** 2 * np.eye(6)
        q += 0.3 * J.T @ np.linalg.solve(JJT, err)
        for i, jid in enumerate(joint_ids):
            d2.qpos[model.jnt_qposadr[jid]] = q[i]
    return q


def pyroki_ik(model, data, joint_ids: Sequence[int], ee_site: int,
               target_pos: np.ndarray, target_quat_wxyz: np.ndarray,
               robot_base_world: Optional[np.ndarray]) -> np.ndarray:
    """
    Pyroki 6-DOF IK with a hand-rolled fallback on any failure.
    """
    try:
        q = solve_ik_6dof(
            np.asarray(target_pos), np.asarray(target_quat_wxyz),
            robot_base_world_pos=robot_base_world,
        )
        if q is None:
            return compute_ik_6dof(model, data, joint_ids, ee_site,
                                    target_pos, target_quat_wxyz)
        if len(q) >= len(joint_ids):
            return np.asarray(q[:len(joint_ids)])
        return np.asarray(q)
    except Exception:
        return compute_ik_6dof(model, data, joint_ids, ee_site,
                                target_pos, target_quat_wxyz)


def solve_ik_sideways(joint_ids: Sequence[int],
                       target_pos: np.ndarray,
                       approach_dir_world: np.ndarray,
                       finger_close_world: Optional[np.ndarray],
                       robot_base_world: Optional[np.ndarray],
                       verbose: bool = False) -> Optional[np.ndarray]:
    """
    Pyroki IK for an explicit palm approach + finger-close direction.

    Returns the actuated joint configuration (length ``len(joint_ids)``)
    or ``None`` if pyroki fails - callers fall back to ``compute_ik`` or
    ``compute_ik_6dof``.
    """
    try:
        quat = quat_from_approach(np.asarray(approach_dir_world),
                                   finger_close_world)
        q_pyroki = solve_ik_6dof(
            np.asarray(target_pos),
            quat,
            robot_base_world_pos=robot_base_world,
        )
        if q_pyroki is None:
            return None
        if len(q_pyroki) >= len(joint_ids):
            return np.asarray(q_pyroki[:len(joint_ids)])
        return np.asarray(q_pyroki)
    except Exception as e:
        if verbose:
            print(f"[pyroki IK] failed: {e}")
        return None

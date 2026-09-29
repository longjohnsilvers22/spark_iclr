"""
Cartesian and joint-space impedance controllers for Franka FR3.

Used by FrankaTorqueDriver; imports no pylibfranka so the math is reusable.
"""

from __future__ import annotations

from typing import Optional

import numpy as np
from scipy.spatial.transform import Rotation

from spark_real.utils.rotations import orientation_error as _orientation_error


class ImpedanceController:
    """Cartesian + joint-space impedance with reference limiting.

    Computes joint torques from a target pose/joint config and the
    current robot state. Gravity is assumed to be handled externally
    (e.g. by libfranka's internal compensation).
    """

    # Reference limiting per 1 ms cycle (SERL technique)
    MAX_POS_STEP = 0.001  # 1 mm/ms = 1 m/s max
    MAX_ORI_STEP = 0.002  # ~0.1 rad/s max (per axis)
    MAX_JOINT_STEP = 0.001  # rad per 1ms cycle = 1 rad/s max

    # Gain ramp-up (SERL: filter_params = 0.005)
    GAIN_RAMP_ALPHA = 0.005

    # Torque rate saturation (SERL/panda-py: max 1.0 Nm change per tick)
    MAX_DTAU = 1.0

    # Joint torque absolute limits (Bamboo: conservative software clamp)
    TAU_LIMITS = np.array([60.0, 60.0, 60.0, 60.0, 30.0, 15.0, 15.0])

    # FR3 datasheet joint torque limits (clamp for the direct torque paths)
    TAU_MAX_DATASHEET = np.array([87.0, 87.0, 87.0, 87.0, 12.0, 12.0, 12.0])

    # Joint limit avoidance (panda-py virtual walls)
    JOINT_LIMIT_LOWER = np.array(
        [-2.7437, -1.7837, -2.9007, -3.0421, -2.8065, 0.5445, -3.0159]
    )
    JOINT_LIMIT_UPPER = np.array(
        [2.7437, 1.7837, 2.9007, -0.1518, 2.8065, 4.5169, 3.0159]
    )
    JOINT_WALL_ZONE = 0.15
    JOINT_WALL_KP = 200.0
    JOINT_WALL_KD = 10.0

    def __init__(
        self,
        K_pos: np.ndarray = np.array([600.0, 600.0, 600.0]),
        K_ori: np.ndarray = np.array([30.0, 30.0, 30.0]),
        damping_ratio: float = 1.0,
        K_nullspace: float = 5.0,
    ):
        self.K_pos = K_pos.copy()
        self.K_ori = K_ori.copy()
        self.damping_ratio = damping_ratio
        self.K_nullspace = K_nullspace

    def cartesian_torques(
        self,
        target_pose: np.ndarray,
        p_ee: np.ndarray,
        R_ee: np.ndarray,
        v_ee: np.ndarray,
        J: np.ndarray,
        q: np.ndarray,
        dq: np.ndarray,
        coriolis: np.ndarray,
        q_nullspace: np.ndarray,
        velocity_ff: Optional[np.ndarray] = None,
    ) -> np.ndarray:
        """Compute Cartesian impedance control torques.

        tau = J^T @ (K @ error + D @ vel_error) + coriolis + nullspace
        """
        pos_error = target_pose[:3, 3] - p_ee
        ori_error = _orientation_error(target_pose[:3, :3], R_ee)

        # Stiffness wrench
        K = np.diag(np.concatenate([self.K_pos, self.K_ori]))
        error_6 = np.concatenate([pos_error, ori_error])
        F_stiffness = K @ error_6

        # Damping wrench (critically damped)
        D_diag = (
            2.0 * self.damping_ratio * np.sqrt(np.concatenate([self.K_pos, self.K_ori]))
        )
        D = np.diag(D_diag)

        vel_error = (velocity_ff - v_ee) if velocity_ff is not None else -v_ee
        F_damping = D @ vel_error

        tau_task = J.T @ (F_stiffness + F_damping)

        # Nullspace torque
        lam2 = 0.01
        Jpinv = J.T @ np.linalg.inv(J @ J.T + lam2 * np.eye(6))
        N = np.eye(7) - Jpinv @ J
        D_null = 2.0 * np.sqrt(self.K_nullspace)
        tau_nullspace = N @ (self.K_nullspace * (q_nullspace - q) - D_null * dq)

        tau = tau_task + tau_nullspace + coriolis
        return np.clip(tau, -self.TAU_MAX_DATASHEET, self.TAU_MAX_DATASHEET)

    def joint_torques(
        self,
        q_d: np.ndarray,
        q: np.ndarray,
        dq: np.ndarray,
        coriolis: np.ndarray,
    ) -> np.ndarray:
        """
        Compute joint-space impedance control torques.
        """
        K_joint = np.array([600.0, 600.0, 600.0, 600.0, 250.0, 150.0, 50.0])
        D_joint = 2.0 * self.damping_ratio * np.sqrt(K_joint)
        tau = K_joint * (q_d - q) - D_joint * dq + coriolis
        return np.clip(tau, -self.TAU_MAX_DATASHEET, self.TAU_MAX_DATASHEET)

    def apply_reference_limiting(
        self,
        reference: np.ndarray,
        target: np.ndarray,
    ) -> np.ndarray:
        """
        SERL-style reference limiting for Cartesian pose.
        """
        ref = reference.copy()

        # Position
        pos_delta = target[:3, 3] - ref[:3, 3]
        pos_norm = np.linalg.norm(pos_delta)
        if pos_norm > self.MAX_POS_STEP:
            pos_delta = pos_delta * (self.MAX_POS_STEP / pos_norm)
        ref[:3, 3] += pos_delta

        # Orientation
        R_ref = ref[:3, :3]
        R_target = target[:3, :3]
        ori_delta = _orientation_error(R_target, R_ref)
        ori_norm = np.linalg.norm(ori_delta)
        if ori_norm > self.MAX_ORI_STEP:
            ori_delta = ori_delta * (self.MAX_ORI_STEP / ori_norm)
        if ori_norm > 1e-10:
            dR = Rotation.from_rotvec(ori_delta).as_matrix()
            ref[:3, :3] = dR @ R_ref

        return ref

    def apply_joint_reference_limiting(
        self,
        reference: np.ndarray,
        target: np.ndarray,
    ) -> np.ndarray:
        """
        Per-joint reference limiting.
        """
        delta = np.clip(target - reference, -self.MAX_JOINT_STEP, self.MAX_JOINT_STEP)
        return reference + delta

    def saturate_torque_rate(
        self, tau_cmd: np.ndarray, tau_J_d: np.ndarray
    ) -> np.ndarray:
        """
        Clamp torque change per tick (max 1.0 Nm/tick).
        """
        delta = np.clip(tau_cmd - tau_J_d, -self.MAX_DTAU, self.MAX_DTAU)
        return tau_J_d + delta

    def joint_limit_avoidance(self, q: np.ndarray, dq: np.ndarray) -> np.ndarray:
        """
        Virtual wall torques near joint limits.
        """
        tau = np.zeros(7)
        for i in range(7):
            dist_lower = q[i] - self.JOINT_LIMIT_LOWER[i]
            dist_upper = self.JOINT_LIMIT_UPPER[i] - q[i]
            if dist_lower < self.JOINT_WALL_ZONE:
                tau[i] += (
                    self.JOINT_WALL_KP * (self.JOINT_WALL_ZONE - dist_lower)
                    - self.JOINT_WALL_KD * dq[i]
                )
            if dist_upper < self.JOINT_WALL_ZONE:
                tau[i] -= (
                    self.JOINT_WALL_KP * (self.JOINT_WALL_ZONE - dist_upper)
                    + self.JOINT_WALL_KD * dq[i]
                )
        return tau

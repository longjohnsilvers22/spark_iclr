"""
Motion primitives for the LIBERO Panda runner.

Factored out of the runner.  These take the env + MuJoCo handles +
joint/actuator indices explicitly and carry no state - the executor calls
them directly.
"""
from __future__ import annotations

from typing import Sequence

import numpy as np
import mujoco

from spark_bench.libero_pro.ik import compute_ik, get_q

# robosuite's controller factory is optional - the joint-position pull path
# falls back to OSC when it is unavailable.
try:
    from robosuite.controllers import controller_factory
except Exception:  # pragma: no cover - robosuite is optional
    controller_factory = None  # type: ignore[assignment]


__all__ = [
    'GRIPPER_OPEN_VAL',
    'TORQUE_LIM',
    'find_joint_ids',
    'find_ee_site',
    'find_actuator_ids',
    'find_robot_base_world',
    'step_env',
    'gripper_action',
    'gripper_action_hold',
    'joint_move',
    'move_to',
]


GRIPPER_OPEN_VAL = 0.04   # Franka finger open position
TORQUE_LIM = 87.0         # Franka per-joint torque limit


# Robot/site discovery - small helpers that scan the MuJoCo model once.

def find_joint_ids(model) -> list[int]:
    """
    Robot arm joint IDs (excludes gripper joints).
    """
    joint_ids = []
    for i in range(model.njnt):
        jname = (mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_JOINT, i) or '').lower()
        if 'robot0_joint' in jname and 'gripper' not in jname:
            joint_ids.append(i)
    return joint_ids


def find_ee_site(model) -> int:
    """
    Find the gripper site (``grip_site``).  Returns -1 if missing.
    """
    for s in range(model.nsite):
        sn = (mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_SITE, s) or '').lower()
        if 'grip_site' in sn:
            return s
    return -1


def find_actuator_ids(model, ndof: int) -> tuple[list[int], list[int]]:
    """
    Split actuators into (arm, gripper) by name; falls back to first ndof.
    """
    arm_ids: list[int] = []
    grip_ids: list[int] = []
    for i in range(model.nu):
        aname = (mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_ACTUATOR, i) or '').lower()
        if 'gripper' in aname or 'finger' in aname:
            grip_ids.append(i)
        elif 'robot0' in aname or 'joint' in aname:
            arm_ids.append(i)
    if not arm_ids:
        arm_ids = list(range(min(ndof, model.nu)))
    if not grip_ids:
        grip_ids = [i for i in range(model.nu) if i not in arm_ids]
    return arm_ids, grip_ids


def find_robot_base_world(model, data) -> np.ndarray | None:
    """
    Locate the Panda base body's world-frame position (for pyroki transform).
    """
    for bid in range(model.nbody):
        bn = (mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_BODY, bid) or '').lower()
        if 'robot0_base' in bn or bn == 'robot_base' or 'panda_link0' in bn:
            return data.xpos[bid].copy()
    return None


# Env-stepping helpers

def step_env(env, action: np.ndarray) -> bool:
    """
    env.step that swallows ValueError and returns done=True on failure.

    Many primitives loop ``env.step(...)`` until either a max iter count or
    the environment terminates - wrapping this in a single helper keeps
    those loops clean.
    """
    try:
        _, _, done, _ = env.step(action)
        return bool(done)
    except (ValueError, Exception):
        return True


def gripper_action(env, open_gripper: bool, steps: int = 60) -> None:
    """
    Open or close the gripper for a fixed number of OSC steps.
    """
    cmd = -1.0 if open_gripper else 1.0
    for _ in range(steps):
        action = np.zeros(7)
        action[6] = cmd
        if step_env(env, action):
            break


def gripper_action_hold(env, model, data, ee_site: int,
                          open_gripper: bool, steps: int = 60) -> None:
    """
    Open/close the gripper while actively holding the current EE pose.

    Sending OSC delta=0 in ``gripper_action`` lets the underlying OSC
    controller drift toward whatever neutral target it last latched, which
    can yank the fingers out of the handle before contact.  This variant
    re-computes the position error each step and feeds it back as a small
    bounded delta so the controller keeps regulating to the target pose for
    the entire gripper close.
    """
    cmd = -1.0 if open_gripper else 1.0
    if ee_site < 0:
        gripper_action(env, open_gripper, steps=steps)
        return
    mujoco.mj_forward(model, data)
    target = data.site_xpos[ee_site].copy()
    for _ in range(steps):
        mujoco.mj_forward(model, data)
        ee = data.site_xpos[ee_site].copy()
        err = target - ee
        action = np.zeros(7)
        # Same delta-scale used in move_to so OSC tracking matches.
        action[:3] = np.clip(err * 10.0 / 0.05, -1.0, 1.0)
        action[6] = cmd
        if step_env(env, action):
            break


# Joint-space PD torque control (bypasses robosuite OSC)

def joint_move(env, model, data,
               joint_ids: Sequence[int], ee_site: int,
               arm_actuator_ids: Sequence[int],
               gripper_actuator_ids: Sequence[int],
               q_target: np.ndarray, gripper_open: bool,
               steps: int = 200, kp: float = 40.0, kd: float = 12.0) -> None:
    """
    Direct torque joint-space PD control with gravity comp + gripper hold.

    Much more capable than robosuite's OSC_POSE at workspace edges and kinematically
    challenging poses (e.g. drawer-pull horizontal palm).
    """
    for _step in range(steps):
        mujoco.mj_forward(model, data)

        q = get_q(model, data, joint_ids)
        qd = np.array([data.qvel[model.jnt_dofadr[j]] for j in joint_ids])

        q_err = q_target - q
        tau = kp * q_err - kd * qd

        for i, jid in enumerate(joint_ids):
            tau[i] += data.qfrc_bias[model.jnt_dofadr[jid]]

        for i in range(min(len(tau), len(arm_actuator_ids))):
            data.ctrl[arm_actuator_ids[i]] = np.clip(tau[i], -TORQUE_LIM, TORQUE_LIM)

        for gi, gid in enumerate(gripper_actuator_ids):
            if gripper_open:
                data.ctrl[gid] = GRIPPER_OPEN_VAL if gi == 0 else -GRIPPER_OPEN_VAL
            else:
                data.ctrl[gid] = 0.0

        mujoco.mj_step(model, data)

        if np.linalg.norm(q_err) < 0.01 and np.linalg.norm(qd) < 0.1:
            break


# Cartesian OSC delta with joint-space PD fallback

def move_to(env, model, data,
            joint_ids: Sequence[int], ee_site: int,
            arm_actuator_ids: Sequence[int],
            gripper_actuator_ids: Sequence[int],
            target: np.ndarray, gripper_open: bool,
            steps: int = 200) -> float:
    """
    Move EE to a Cartesian target with stall detection + PD fallback.

    Returns the final distance from EE to target.
    """
    gripper_cmd = -1.0 if gripper_open else 1.0
    prev_dist = float('inf')
    stall_count = 0
    osc_stalled = False
    for step in range(steps):
        mujoco.mj_forward(model, data)
        ee = data.site_xpos[ee_site].copy()
        err = target - ee
        dist = np.linalg.norm(err)
        if dist < 0.005 and step > 0:
            break
        if step > 0 and step % 30 == 0:
            if dist > prev_dist * 0.95:
                stall_count += 1
                if stall_count >= 2:
                    osc_stalled = True
                    break
            else:
                stall_count = 0
            prev_dist = dist
        action = np.zeros(7)
        action[:3] = np.clip(err * 10.0 / 0.05, -1.0, 1.0)
        action[6] = gripper_cmd
        if step_env(env, action):
            break

    if osc_stalled:
        mujoco.mj_forward(model, data)
        final_dist = np.linalg.norm(target - data.site_xpos[ee_site])
        if final_dist > 0.01:
            q_ik = compute_ik(model, data, joint_ids, ee_site, target)
            joint_move(env, model, data, joint_ids, ee_site,
                        arm_actuator_ids, gripper_actuator_ids,
                        q_ik, gripper_open, steps=200)

    mujoco.mj_forward(model, data)
    return float(np.linalg.norm(target - data.site_xpos[ee_site]))


# CaP-X-style JOINT_POSITION pull via env.step (mid-trial controller swap)

def _swap_to_joint_position_controller(env):
    """
    Temporarily swap the env's active controller to JOINT_POSITION.

    Returns ``(robot, original_controller, control_freq)`` so caller can
    restore via ``robot.controller = original_controller``.  This mirrors
    CaP-X's ``move_to_joints_blocking`` path where the env is built with
    ``controller="JOINT_POSITION"`` from the start.  Swapping mid-trial keeps
    the OSC-based primitives (move_to / grasp approach) working everywhere
    else.

    Returns ``None`` on failure (e.g. controller_factory missing); caller
    should fall back to the OSC pull path.
    """
    if controller_factory is None:
        return None
    # Locate the underlying robosuite env + robot.
    rs_env = env.env if hasattr(env, 'env') else env
    if not hasattr(rs_env, 'robots') or not rs_env.robots:
        return None
    robot = rs_env.robots[0]
    original = robot.controller
    if original is None:
        return None
    # Build a JOINT_POSITION config by mirroring the existing controller's
    # sim-side wiring (joint indexes, eef name, actuator range, control freq).
    cfg = {
        'type': 'JOINT_POSITION',
        'input_max': 1,
        'input_min': -1,
        'output_max': 0.05,
        'output_min': -0.05,
        'kp': 50,
        'damping_ratio': 1,
        'impedance_mode': 'fixed',
        'kp_limits': [0, 300],
        'damping_ratio_limits': [0, 10],
        'qpos_limits': None,
        'interpolation': None,
        'ramp_ratio': 0.2,
        # Sim wiring (copied from the OSC controller's config).
        'robot_name': robot.name,
        'sim': rs_env.sim,
        'eef_name': robot.gripper.important_sites['grip_site'],
        'eef_rot_offset': robot.eef_rot_offset,
        'joint_indexes': {
            'joints': robot.joint_indexes,
            'qpos': robot._ref_joint_pos_indexes,
            'qvel': robot._ref_joint_vel_indexes,
        },
        'actuator_range': robot.torque_limits,
        'policy_freq': robot.control_freq,
        'ndim': len(robot.robot_joints),
    }
    try:
        new_ctrl = controller_factory('JOINT_POSITION', cfg)
    except Exception:
        return None
    robot.controller = new_ctrl
    return (robot, original, int(robot.control_freq))


def _restore_controller(swap_info) -> None:
    """
    Restore the original (OSC) controller saved by ``_swap_to_joint_position_controller``.
    """
    if swap_info is None:
        return
    robot, original, _ = swap_info
    try:
        robot.controller = original
    except Exception:
        pass


def move_to_pose(env, model, data,
                 joint_ids, ee_site: int,
                 arm_actuator_ids, gripper_actuator_ids,
                 target_pos, target_mat, gripper_open: bool,
                 steps: int = 200, pos_tol: float = 0.005,
                 rot_tol: float = 0.08) -> tuple:
    """
    Drive the EE to a position AND an orientation through OSC_POSE.

    ``move_to`` writes only ``action[:3]``, so every grasp inherits whatever
    wrist pose the arm happens to hold. The LIBERO action is seven wide,
    ``[dx, dy, dz, drx, dry, drz, gripper]``, and the middle three are an
    axis-angle delta the controller already accepts. Commanding them is what
    makes a frontal grasp reachable: a drawer handle is a vertical bar, and the
    jaws have to close across it rather than down onto it.

    ``target_mat`` is a 3x3 world rotation for the gripper site.
    Returns ``(pos_err_m, rot_err_rad)``.
    """
    gripper_cmd = -1.0 if gripper_open else 1.0
    pos_err = rot_err = float('inf')
    R_des = np.asarray(target_mat, dtype=float).reshape(3, 3)
    for step in range(steps):
        mujoco.mj_forward(model, data)
        ee = data.site_xpos[ee_site].copy()
        R_cur = data.site_xmat[ee_site].reshape(3, 3).copy()
        perr = np.asarray(target_pos, dtype=float) - ee
        pos_err = float(np.linalg.norm(perr))
        # Orientation error as an axis-angle vector in the world frame.
        R_err = R_des @ R_cur.T
        aa = np.zeros(3)
        quat = np.zeros(4)
        mujoco.mju_mat2Quat(quat, R_err.flatten())
        mujoco.mju_quat2Vel(aa, quat, 1.0)
        rot_err = float(np.linalg.norm(aa))
        if pos_err < pos_tol and rot_err < rot_tol and step > 0:
            break
        action = np.zeros(7)
        action[:3] = np.clip(perr * 10.0 / 0.05, -1.0, 1.0)
        action[3:6] = np.clip(aa * 2.0, -1.0, 1.0)
        action[6] = gripper_cmd
        if step_env(env, action):
            break
    return pos_err, rot_err

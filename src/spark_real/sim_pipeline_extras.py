"""SPARK sim-side trajectory dispatcher.

Lives BESIDE sim_pipeline.py and provides:
  * pyroki IK for the Panda (matches the LIBERO horizontal-grasp recipe)
  * Joint-space trajectory interpolation (the same pattern used by the real
    spark_real top-down / horizontal grasps, see
    skills/grasping.py::_joint_motion_to). The sim has no libfranka reflexes
    so joints are lerped over N waypoints, keeping the multi-phase
    trajectory shape so the test mirrors the real primitive.
  * Smooth gripper actuation (no instant teleport).
  * Frame capture + video record helpers.

This module is standalone: it does NOT import skills/, control/score_executor
or any real-robot driver. It only consumes the geometric trajectories those
skills would have built, mirroring the trajectory math from
skills/grasping.py::_horizontal_grasp_orientation and ::_grasp_horizontal.

Used by tests/corl_<task>_sim.py for the four paper-relevant Franka tasks:
F1 pour, F2 silverware sort, F3 sponge scrub, F4 fold cloth.
"""

from __future__ import annotations

import os
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Optional, Sequence

import numpy as np
from scipy.spatial.transform import Rotation as SciRot

# Env caps must be set before importing mujoco.
os.environ.setdefault("MUJOCO_GL", "egl")
os.environ.setdefault("JAX_PLATFORMS", "cpu")
os.environ.setdefault("TORCH_FORCE_NO_WEIGHTS_ONLY_LOAD", "1")

import mujoco  # noqa: E402
import imageio  # noqa: E402


# pyroki IK (lazy load)
_PK_ROBOT = None
_PK_TARGET_LINK_NAME = "panda_hand"
_PK_TARGET_LINK_IDX = None
_PK_SOLVE_FN = None


def _pyroki_init():
    global _PK_ROBOT, _PK_TARGET_LINK_IDX, _PK_SOLVE_FN
    if _PK_ROBOT is not None:
        return
    import pyroki as pk
    import jax_dataclasses as jdc
    import jaxlie
    import jaxls
    import jax.numpy as jnp
    from robot_descriptions.loaders.yourdfpy import load_robot_description

    urdf = load_robot_description("panda_description")
    _PK_ROBOT = pk.Robot.from_urdf(urdf)
    _PK_TARGET_LINK_IDX = _PK_ROBOT.links.names.index(_PK_TARGET_LINK_NAME)

    @jdc.jit
    def _solve(robot, target_link_index, target_wxyz, target_position,
               rest_pose, rest_weight):
        joint_var = robot.joint_var_cls(0)
        variables = [joint_var]
        costs = [
            pk.costs.pose_cost_analytic_jac(
                robot,
                joint_var,
                jaxlie.SE3.from_rotation_and_translation(
                    jaxlie.SO3(target_wxyz), target_position
                ),
                target_link_index,
                pos_weight=50.0,
                ori_weight=10.0,
            ),
            pk.costs.limit_constraint(robot, joint_var),
            # Soft rest-pose pull. Keeps IK in a basin near the current q
            # so motion-to-motion transitions don't pick wildly different
            # IK branches (which manifests as the elbow swinging across
            # the workspace and knocking objects over).
            pk.costs.rest_cost(joint_var, rest_pose, rest_weight),
        ]
        sol = (
            jaxls.LeastSquaresProblem(costs=costs, variables=variables)
            .analyze()
            .solve(
                verbose=False,
                linear_solver="dense_cholesky",
                trust_region=jaxls.TrustRegionConfig(lambda_initial=1.0),
            )
        )
        return sol[joint_var]

    _PK_SOLVE_FN = _solve


# panda_hand_tcp URDF offset (matches the LIBERO recipe).
_TCP_OFFSET_HAND = np.array([0.0, 0.0, -0.1034], dtype=np.float64)


def quat_xyzw_to_wxyz(q):
    return np.array([q[3], q[0], q[1], q[2]], dtype=np.float64)


def quat_wxyz_to_xyzw(q):
    return np.array([q[1], q[2], q[3], q[0]], dtype=np.float64)


def solve_ik_pyroki(target_pos: np.ndarray, target_wxyz: np.ndarray,
                     rest_pose_q7: Optional[np.ndarray] = None,
                     rest_weight: float = 0.5) -> np.ndarray:
    """Returns 7-DOF Panda joint solution. target_pos is the panda_hand_tcp
    position (between the fingertips); we shift it by the TCP offset before
    handing off to pyroki, which solves for panda_hand.

    rest_pose_q7: optional 7-DOF seed (current robot config). Pyroki's
    rest_cost pulls the solution toward this pose, so successive IK calls
    don't pick discontinuous branches.
    """
    _pyroki_init()
    import jax.numpy as jnp
    rot = SciRot.from_quat(quat_wxyz_to_xyzw(target_wxyz))
    panda_hand_pos = target_pos + rot.apply(_TCP_OFFSET_HAND)
    # Pyroki Panda URDF has 9 joints (7 arm + 2 finger). Pad seed to 9.
    n_joints = _PK_ROBOT.joints.num_actuated_joints
    if rest_pose_q7 is None:
        # Use the panda HOME pose (matches LIBERO + spark_real defaults).
        rest_full = np.array([0.0, -0.785, 0.0, -2.356, -0.5, 1.571, 0.785,
                              0.04, 0.04], dtype=np.float64)
    else:
        rest_full = np.zeros(n_joints, dtype=np.float64)
        rest_full[:7] = np.asarray(rest_pose_q7, dtype=np.float64)[:7]
        if n_joints > 7:
            rest_full[7:] = 0.04  # fingers open
    rest_full = rest_full[:n_joints]
    cfg = _PK_SOLVE_FN(
        _PK_ROBOT,
        jnp.array(_PK_TARGET_LINK_IDX),
        jnp.array(target_wxyz),
        jnp.array(panda_hand_pos),
        jnp.array(rest_full),
        jnp.array(rest_weight),
    )
    return np.asarray(cfg[:7], dtype=np.float64)


# Orientation helpers
def horizontal_grasp_quat_wxyz(approach_xy: np.ndarray,
                                pitch_rad: float = 0.0) -> np.ndarray:
    """Mirror of skills/grasping.py::_horizontal_grasp_orientation.

    approach_xy: 2-vector pointing FROM hover TOWARD object (unit not required).
    pitch_rad: 0 = pure horizontal side. pi/4 = 45 deg from above. pi/2 = top-down.

    Returns quat as wxyz (pyroki convention).
    """
    a = np.asarray(approach_xy, dtype=float).reshape(-1)
    if a.size == 2:
        a = np.array([a[0], a[1], 0.0], dtype=float)
    a = np.array([a[0], a[1], 0.0], dtype=float)
    n = float(np.linalg.norm(a))
    if n < 1e-6:
        a = np.array([1.0, 0.0, 0.0], dtype=float)
    else:
        a = a / n
    cos_p, sin_p = float(np.cos(pitch_rad)), float(np.sin(pitch_rad))
    z_axis = cos_p * a + np.array([0.0, 0.0, -sin_p])
    z_axis = z_axis / (np.linalg.norm(z_axis) + 1e-12)
    world_z = np.array([0.0, 0.0, 1.0])
    y_axis = np.cross(world_z, z_axis)
    yn = float(np.linalg.norm(y_axis))
    if yn < 1e-6:
        y_axis = np.array([0.0, 1.0, 0.0])
    else:
        y_axis = y_axis / yn
    x_axis = np.cross(y_axis, z_axis)
    x_axis = x_axis / (np.linalg.norm(x_axis) + 1e-12)
    R = np.column_stack([x_axis, y_axis, z_axis])
    return quat_xyzw_to_wxyz(SciRot.from_matrix(R).as_quat())


def top_down_quat_wxyz(yaw_rad: float = 0.0) -> np.ndarray:
    """TCP-Z pointing -Z world (gripper points down), yaw about world Z.

    This matches the real-robot GRASP_ORIENTATION = [pi, 0, 0] rotvec when
    yaw=0 (which is panda_hand pointing -Z).
    """
    # Start from a rotation that flips +Z hand to -Z world.
    # panda_hand frame: +Z = approach. We want approach = -world_Z.
    # The rotation that maps +Z -> -Z is rotation by pi about either X or Y.
    base = SciRot.from_euler("x", np.pi)  # equivalent to rotvec [pi, 0, 0]
    yawed = SciRot.from_euler("z", yaw_rad) * base
    return quat_xyzw_to_wxyz(yawed.as_quat())


# Sim wrapper
class PandaSimWorld:
    """Minimal MuJoCo wrapper for Panda + scene. 7 arm + 1 gripper actuator
    (gripper is the panda menagerie tendon-split actuator: 0=closed, 255=open).
    """

    PANDA_HOME = np.array([0.0, -0.785, 0.0, -2.356, -0.5, 1.571, 0.785])

    def __init__(self, scene_path: str, base_pos: np.ndarray | None = None,
                 base_quat_wxyz: np.ndarray | None = None,
                 width: int = 640, height: int = 480,
                 record_camera: str = "agentview"):
        self.scene_path = scene_path
        self.model = mujoco.MjModel.from_xml_path(scene_path)
        self.data = mujoco.MjData(self.model)
        mujoco.mj_forward(self.model, self.data)

        # Default base = identity if not provided. Scenes that mount panda
        # at non-origin (e.g. pedestal) must pass base_pos/base_quat_wxyz.
        self.base_pos = (np.array(base_pos, dtype=float) if base_pos is not None
                         else np.zeros(3))
        self.base_quat_wxyz = (np.array(base_quat_wxyz, dtype=float)
                               if base_quat_wxyz is not None
                               else np.array([1.0, 0.0, 0.0, 0.0]))
        self._base_R = SciRot.from_quat(quat_wxyz_to_xyzw(self.base_quat_wxyz)).as_matrix()
        self._base_R_inv = self._base_R.T

        # Joint qpos addresses for the 7 panda arm joints.
        # Try menagerie names first, fall back to LIBERO-style names.
        self._arm_qpos_addrs = []
        self._arm_qvel_addrs = []
        names_attempts = [
            [f"joint{i}" for i in range(1, 8)],
            [f"robot0_joint{i}" for i in range(1, 8)],
            [f"fr3_joint{i}" for i in range(1, 8)],
        ]
        for names in names_attempts:
            ok = True
            qpos_addrs = []
            qvel_addrs = []
            for jn in names:
                jid = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_JOINT, jn)
                if jid < 0:
                    ok = False
                    break
                qpos_addrs.append(int(self.model.jnt_qposadr[jid]))
                qvel_addrs.append(int(self.model.jnt_dofadr[jid]))
            if ok:
                self._arm_qpos_addrs = qpos_addrs
                self._arm_qvel_addrs = qvel_addrs
                self._arm_joint_names = names
                break
        if not self._arm_qpos_addrs:
            raise RuntimeError(f"No panda arm joints found in {scene_path}")

        # Actuator IDs: first 7 are arm, 8th is gripper tendon
        self._arm_act_ids = list(range(min(7, self.model.nu)))
        self._gripper_act_id = 7 if self.model.nu > 7 else None

        # Cameras
        self._cam_id = mujoco.mj_name2id(
            self.model, mujoco.mjtObj.mjOBJ_CAMERA, record_camera)
        # Renderer
        self.width = width
        self.height = height
        self._renderer = mujoco.Renderer(self.model, height, width)
        self.frames: list[np.ndarray] = []

        # Reset to home
        self.set_arm_qpos(self.PANDA_HOME, hold=False)
        self.set_gripper_ctrl(255)  # open
        mujoco.mj_forward(self.model, self.data)

    # joint state
    def get_arm_qpos(self) -> np.ndarray:
        return np.array(self.data.qpos[self._arm_qpos_addrs], dtype=float)

    def set_arm_qpos(self, q: np.ndarray, hold: bool = True):
        q = np.asarray(q, dtype=float).reshape(7)
        for i, addr in enumerate(self._arm_qpos_addrs):
            self.data.qpos[addr] = q[i]
        for addr in self._arm_qvel_addrs:
            self.data.qvel[addr] = 0.0
        if hold:
            for i in range(7):
                if i < len(self._arm_act_ids):
                    self.data.ctrl[self._arm_act_ids[i]] = q[i]
        mujoco.mj_forward(self.model, self.data)

    def set_arm_ctrl(self, q: np.ndarray):
        q = np.asarray(q, dtype=float).reshape(7)
        for i in range(7):
            if i < len(self._arm_act_ids):
                self.data.ctrl[self._arm_act_ids[i]] = q[i]

    def set_gripper_ctrl(self, value: float):
        """0=closed, 255=open (matches panda menagerie convention)."""
        if self._gripper_act_id is not None:
            self.data.ctrl[self._gripper_act_id] = float(np.clip(value, 0, 255))

    def step(self, n: int = 1, snap_every: int = 0):
        for i in range(n):
            mujoco.mj_step(self.model, self.data)
            if snap_every > 0 and (i % snap_every == 0):
                self.snap()

    # IK to joint trajectory
    def ik_world(self, target_pos_world: np.ndarray,
                 target_quat_wxyz_world: np.ndarray,
                 seed_from_current: bool = True,
                 rest_weight: float = 3.0) -> np.ndarray:
        """Solve IK with target in world frame. Returns 7-DOF arm config.

        seed_from_current: if True (default), use the robot's current joint
        config as the IK rest pose. Crucial for continuity: successive IK
        calls will land on the same kinematic branch, so the elbow doesn't
        flip across the workspace between waypoints.
        """
        target_pos_base = self._base_R_inv @ (
            np.asarray(target_pos_world, dtype=float) - self.base_pos)
        R_world = SciRot.from_quat(
            quat_wxyz_to_xyzw(target_quat_wxyz_world)).as_matrix()
        R_base = self._base_R_inv @ R_world
        target_quat_base_wxyz = quat_xyzw_to_wxyz(
            SciRot.from_matrix(R_base).as_quat())
        seed = self.get_arm_qpos() if seed_from_current else None
        return solve_ik_pyroki(target_pos_base, target_quat_base_wxyz,
                                 rest_pose_q7=seed, rest_weight=rest_weight)

    def move_to_joints_lerp(self, target: np.ndarray, *, n_waypoints: int = 50,
                              steps_per_wp: int = 8, snap_every: int = 4):
        """Linear-interp joint trajectory with controller tracking each
        waypoint. Mirrors LIBERO recipe's move_to_joints_interp.
        """
        target = np.asarray(target, dtype=float).reshape(7)
        start = self.get_arm_qpos()
        for k in range(1, n_waypoints + 1):
            t = k / n_waypoints
            wp = (1 - t) * start + t * target
            self.set_arm_ctrl(wp)
            for s in range(steps_per_wp):
                mujoco.mj_step(self.model, self.data)
                if snap_every > 0 and (s % snap_every == 0):
                    self.snap()
        # Final hold
        self.set_arm_ctrl(target)
        for s in range(40):
            mujoco.mj_step(self.model, self.data)
            if snap_every > 0 and (s % snap_every == 0):
                self.snap()

    def move_to_pose(self, target_pos_world, target_quat_wxyz_world, *,
                     n_waypoints: int = 50, steps_per_wp: int = 8,
                     snap_every: int = 4, label: str = "",
                     n_cart_subdiv: int = 1):
        """Solve IK, then lerp. Returns the achieved arm config.

        n_cart_subdiv: when >1, divide the CARTESIAN path from current TCP
        to target into this many sub-segments and solve IK at each. This
        eliminates the "elbow flies wildly" failure mode where joint-space
        lerp between two IK solutions takes the arm through configs that
        aren't anywhere near the straight Cartesian line. For long
        transit moves over a cluttered table, set n_cart_subdiv=4-8.
        """
        target_quat_wxyz_world = np.asarray(target_quat_wxyz_world, dtype=float)
        target_pos_world = np.asarray(target_pos_world, dtype=float)
        if n_cart_subdiv <= 1:
            q_target = self.ik_world(target_pos_world, target_quat_wxyz_world)
            if label:
                print(f"  [move/{label}] dq_max={float(np.max(np.abs(self.get_arm_qpos() - q_target))):.2f}rad q_target={q_target.round(2).tolist()}")
            self.move_to_joints_lerp(q_target, n_waypoints=n_waypoints,
                                       steps_per_wp=steps_per_wp,
                                       snap_every=snap_every)
            return q_target
        # Multi-segment Cartesian: get start TCP from forward kinematics
        # of current q via a pyroki query.
        start_pos = self._read_tcp_pos_world()
        start_quat = self._read_tcp_quat_wxyz_world()
        wp_per_seg = max(int(n_waypoints / n_cart_subdiv), 6)
        from scipy.spatial.transform import Slerp
        rotations = SciRot.from_quat([
            quat_wxyz_to_xyzw(start_quat),
            quat_wxyz_to_xyzw(target_quat_wxyz_world),
        ])
        slerp = Slerp([0.0, 1.0], rotations)
        q_target = None
        for k in range(1, n_cart_subdiv + 1):
            t = k / n_cart_subdiv
            seg_pos = (1 - t) * start_pos + t * target_pos_world
            seg_R = slerp([t])[0]
            seg_quat_wxyz = quat_xyzw_to_wxyz(seg_R.as_quat())
            q_target = self.ik_world(seg_pos, seg_quat_wxyz)
            if label:
                print(f"  [move/{label}/seg{k}] dq_max={float(np.max(np.abs(self.get_arm_qpos() - q_target))):.2f}rad seg_pos={seg_pos.round(3).tolist()}")
            self.move_to_joints_lerp(q_target, n_waypoints=wp_per_seg,
                                       steps_per_wp=steps_per_wp,
                                       snap_every=snap_every)
        return q_target

    def _read_tcp_pos_world(self) -> np.ndarray:
        """TCP (pinch site) position in world. Falls back to hand body
        pos + offset if no 'pinch' site present."""
        sid = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_SITE, "pinch")
        if sid >= 0:
            return self.data.site_xpos[sid].copy()
        bid = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_BODY, "hand")
        if bid >= 0:
            return self.data.xpos[bid].copy()
        return np.zeros(3)

    def _read_tcp_quat_wxyz_world(self) -> np.ndarray:
        """TCP orientation as quat wxyz, read from MuJoCo's current state."""
        sid = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_SITE, "pinch")
        if sid >= 0:
            site_xmat = self.data.site_xmat[sid].reshape(3, 3).copy()
        else:
            bid = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_BODY, "hand")
            if bid >= 0:
                site_xmat = self.data.xmat[bid].reshape(3, 3).copy()
            else:
                site_xmat = np.eye(3)
        return quat_xyzw_to_wxyz(SciRot.from_matrix(site_xmat).as_quat())

    # gripper
    def smooth_gripper(self, target_ctrl: float, steps: int = 60,
                       snap_every: int = 6):
        """Linearly ramp gripper ctrl from current to target over `steps`
        sim steps. Avoids an instantaneous open/close that flings objects
        across the scene.
        """
        start = float(self.data.ctrl[self._gripper_act_id]) if self._gripper_act_id is not None else 0.0
        for k in range(1, steps + 1):
            t = k / steps
            self.set_gripper_ctrl((1 - t) * start + t * target_ctrl)
            mujoco.mj_step(self.model, self.data)
            if snap_every > 0 and (k % snap_every == 0):
                self.snap()

    def gripper_open(self, **kw):
        self.smooth_gripper(255, **kw)

    def gripper_close(self, **kw):
        self.smooth_gripper(0, **kw)

    # queries
    def get_body_pos(self, name: str) -> Optional[np.ndarray]:
        bid = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_BODY, name)
        if bid < 0:
            return None
        return self.data.xpos[bid].copy()

    def get_site_pos(self, name: str) -> Optional[np.ndarray]:
        sid = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_SITE, name)
        if sid < 0:
            return None
        return self.data.site_xpos[sid].copy()

    # rendering
    def render(self, camera: Optional[str] = None) -> np.ndarray:
        if camera is None:
            cam_id = self._cam_id
        else:
            cam_id = mujoco.mj_name2id(
                self.model, mujoco.mjtObj.mjOBJ_CAMERA, camera)
        if cam_id < 0:
            cam_id = -1
        self._renderer.update_scene(self.data, camera=cam_id)
        return self._renderer.render().copy()

    def snap(self, camera: Optional[str] = None):
        self.frames.append(self.render(camera))

    def save_png(self, path: str, camera: Optional[str] = None):
        os.makedirs(os.path.dirname(path), exist_ok=True)
        imageio.imwrite(path, self.render(camera))

    def save_video(self, path: str, fps: int = 30):
        if not self.frames:
            return
        os.makedirs(os.path.dirname(path), exist_ok=True)
        imageio.mimsave(path, self.frames, fps=fps)

    def settle(self, n: int = 200, snap_every: int = 0):
        # Hold current ctrl at current joint state (avoid free-fall)
        cur = self.get_arm_qpos()
        self.set_arm_ctrl(cur)
        for s in range(n):
            mujoco.mj_step(self.model, self.data)
            if snap_every > 0 and (s % snap_every == 0):
                self.snap()


def world_target_from_object(obj_pos: np.ndarray, *, dx=0.0, dy=0.0, dz=0.0):
    return np.array([obj_pos[0] + dx, obj_pos[1] + dy, obj_pos[2] + dz],
                     dtype=float)


__all__ = [
    "PandaSimWorld",
    "horizontal_grasp_quat_wxyz",
    "top_down_quat_wxyz",
    "quat_wxyz_to_xyzw",
    "quat_xyzw_to_wxyz",
    "solve_ik_pyroki",
    "world_target_from_object",
]

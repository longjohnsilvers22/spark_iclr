"""
Bimanual collision-aware IK planner backed by PyRoki.

Glue only: calls pyroki's public API (``pk.Robot.from_urdf``,
``pk.collision.RobotCollision.from_urdf``, ``pk.collision.Sphere``,
``pk.costs.{pose_cost, rest_cost, self_collision_cost, world_collision_cost,
limit_cost}``) plus ``jaxls.LeastSquaresProblem`` for the solve, and
translates inputs/outputs.

Source provenance:
* PyRoki (chungmin99 fork): https://github.com/chungmin99/pyroki
* Reference integration: uynitsuj/robots_realtime, especially
  ``robots/inverse_kinematics/pyroki_snippets/_solve_ik_with_collision.py``
  and ``franka_pyroki.py``.

:class:`BimanualPyrokiPlanner`: given a target end-effector pose for one
arm and the CURRENT joint configuration of the OTHER arm, solves IK with
pose, rest-pose, self-collision, and other-arm avoidance costs (the other
arm's TCP modeled as a sphere obstacle in the moving arm's base frame). No
motion is issued here; the result goes through
:class:`BimanualFrankaDriver.move_to_joint_config`.

The first call JIT-compiles the JAX solver (~5-10 s); subsequent calls
take 30-80 ms on CPU for the 7-DOF Franka.
"""

from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Optional, Sequence

import numpy as np

try:
    import jax
    import jax.numpy as jnp
    import jaxlie
    import jaxls
    import pyroki as pk
    import yourdfpy
    from robot_descriptions.loaders.yourdfpy import load_robot_description
except ImportError:
    jax = jnp = jaxlie = jaxls = pk = yourdfpy = None
    load_robot_description = None

logger = logging.getLogger("spark_real.pyroki_planner")


def _try_import_pyroki():
    if pk is None:
        logger.error("PyRoki / JAX not installed in this env")
        return None, None
    return pk, yourdfpy


class BimanualPyrokiPlanner:
    """
    Holds per-arm PyRoki Robot + RobotCollision objects.
    """

    def __init__(
        self,
        *,
        left_description: str = "panda_description",
        right_description: str = "fr3_description",
        left_target_link: str = "panda_link8",
        right_target_link: str = "fr3_link8",
        T_right_to_left: Optional[np.ndarray] = None,
        other_arm_sphere_radius: float = 0.08,
    ):
        """
        Args:
        left_description / right_description: robot_descriptions
            package names (e.g. "panda_description", "fr3_description").
            These are the same names robots_realtime uses, so URDFs
            stay in lock-step with the upstream reference.
        T_right_to_left: 4x4 SE(3); a point in the RIGHT base frame,
            pre-multiplied by this, lands in the LEFT base frame.
        other_arm_sphere_radius: radius of the keep-out sphere around
            the other arm's TCP. 8 cm is a reasonable buffer for the
            SSG-48 gripper geometry.
        """
        pk, yourdfpy = _try_import_pyroki()
        if pk is None:
            raise RuntimeError("PyRoki / JAX not installed in this env")
        self._pk = pk
        self._yourdfpy = yourdfpy

        self.left_urdf = load_robot_description(left_description)
        self.right_urdf = load_robot_description(right_description)
        self.left_robot = pk.Robot.from_urdf(self.left_urdf)
        self.right_robot = pk.Robot.from_urdf(self.right_urdf)
        self.left_coll = pk.collision.RobotCollision.from_urdf(self.left_urdf)
        self.right_coll = pk.collision.RobotCollision.from_urdf(self.right_urdf)

        if T_right_to_left is None:
            T_right_to_left = np.eye(4)
        self.T_right_to_left = np.asarray(T_right_to_left, dtype=float)
        self.T_left_to_right = np.linalg.inv(self.T_right_to_left)
        self.other_arm_radius = float(other_arm_sphere_radius)

        self.left_target_link = left_target_link
        self.right_target_link = right_target_link
        # Pre-build per-arm JIT-compiled IK solvers for the common case (no
        # inter-arm collision sphere) so steady-state calls reuse the compiled
        # graph instead of rebuilding the jaxls problem each call. The
        # world-collision path (other_arm_tcp given) still builds per call.
        self._fast_solve = {
            "left": self._make_fast_solver(
                self.left_robot, self.left_coll, self.left_target_link
            ),
            "right": self._make_fast_solver(
                self.right_robot, self.right_coll, self.right_target_link
            ),
        }
        logger.info(
            "BimanualPyrokiPlanner ready (left=%s, right=%s)",
            left_description,
            right_description,
        )

    def _make_fast_solver(self, robot, coll, link):
        """
        Return a jax.jit IK solver with robot/collision/target-link baked
        in (static), so only the target pose and seed vary between calls.
        Cost stack is identical to _solve_ik_jit.
        """
        pk = self._pk
        link_idx = robot.links.names.index(link)

        @jax.jit
        def _solve(T_target, prev_cfg):
            jv = robot.joint_var_cls(0)
            costs = [
                pk.costs.pose_cost(
                    robot,
                    jv,
                    target_pose=T_target,
                    target_link_index=link_idx,
                    pos_weight=20.0,
                    ori_weight=20.0,
                ),
                pk.costs.rest_cost(jv, rest_pose=prev_cfg, weight=0.1),
                pk.costs.self_collision_cost(robot, coll, jv, margin=0.01, weight=2.0),
                pk.costs.limit_cost(robot, jv, weight=100.0),
            ]
            sol = (
                jaxls.LeastSquaresProblem(costs, [jv])
                .analyze()
                .solve(initial_vals=jaxls.VarValues.make([jv.with_value(prev_cfg)]))
            )
            return sol[jv]

        return _solve

    @classmethod
    def from_default(
        cls,
        *,
        T_right_to_left_json: Optional[Path] = None,
        other_arm_sphere_radius: float = 0.08,
    ) -> "BimanualPyrokiPlanner":
        """
        Build the ANON-LAB planner: panda_description + fr3_description,
        with the cross-arm transform loaded from ~/.spark_real/.
        """
        if T_right_to_left_json is None:
            T_right_to_left_json = Path.home() / ".spark_real" / "T_right_to_left.json"
        T_r_to_l = None
        if T_right_to_left_json and Path(T_right_to_left_json).exists():
            data = json.loads(Path(T_right_to_left_json).read_text())
            T_r_to_l = np.asarray(data.get("T_right_to_left"), dtype=float)
        return cls(
            T_right_to_left=T_r_to_l, other_arm_sphere_radius=other_arm_sphere_radius
        )

    def solve(
        self,
        arm: str,
        target_position_base: np.ndarray,
        target_wxyz_base: np.ndarray,
        prev_cfg: np.ndarray,
        other_arm_tcp_in_other_base: Optional[np.ndarray] = None,
    ) -> np.ndarray:
        """
        Return target joint config for ``arm`` that reaches the goal pose
        in its base frame while avoiding self-collision and the other
        arm's current TCP (modeled as a sphere).
        """
        pk = self._pk

        if arm == "left":
            robot = self.left_robot
            coll = self.left_coll
            link = self.left_target_link
            T_other_to_this = self.T_right_to_left
        elif arm == "right":
            robot = self.right_robot
            coll = self.right_coll
            link = self.right_target_link
            T_other_to_this = self.T_left_to_right
        else:
            raise ValueError(f"arm must be left|right, got {arm!r}")

        target_position_base = np.asarray(target_position_base, dtype=float).reshape(3)
        target_wxyz_base = np.asarray(target_wxyz_base, dtype=float).reshape(4)
        prev_cfg = np.asarray(prev_cfg, dtype=float).reshape(-1)
        # The Panda/FR3 robot_descriptions URDFs include a finger joint
        # (8 actuated), but the driver only commands the 7 arm joints.
        # Pad incoming prev_cfg if needed; the result will be sliced to 7
        # before returning.
        n_act = robot.joints.num_actuated_joints
        if prev_cfg.size < n_act:
            prev_cfg_padded = np.zeros(n_act)
            prev_cfg_padded[: prev_cfg.size] = prev_cfg
            # Default finger to mid-open
            if n_act > prev_cfg.size:
                prev_cfg_padded[7:] = 0.02
            prev_cfg = prev_cfg_padded
        elif prev_cfg.size > n_act:
            prev_cfg = prev_cfg[:n_act]

        # Project the other arm's TCP into THIS arm's base frame and
        # model it as a sphere obstacle.
        world_coll: Sequence = ()
        if other_arm_tcp_in_other_base is not None:
            p_other = np.asarray(other_arm_tcp_in_other_base, dtype=float).reshape(3)
            p_in_this = (T_other_to_this @ np.append(p_other, 1.0))[:3]
            obstacle = pk.collision.Sphere.from_center_and_radius(
                center=jnp.array(p_in_this),
                radius=jnp.array(self.other_arm_radius),
            )
            world_coll = (obstacle,)

        target_link_idx = robot.links.names.index(link)
        # pyroki's pose_residual asserts target_link_index.dtype == jnp.int32
        # (some non-keep_orientation paths land here with int64 default).
        T_target = jaxlie.SE3(
            jnp.concatenate(
                [
                    jnp.array(target_wxyz_base, dtype=jnp.float32),
                    jnp.array(target_position_base, dtype=jnp.float32),
                ]
            )
        )
        if not world_coll:
            # Fast path: cached jitted solver (no inter-arm collision sphere).
            cfg = self._fast_solve[arm](
                T_target, jnp.array(prev_cfg, dtype=jnp.float32)
            )
        else:
            cfg = _solve_ik_jit(
                robot,
                coll,
                world_coll,
                T_target,
                jnp.array(target_link_idx, dtype=jnp.int32),
                jnp.array(prev_cfg, dtype=jnp.float32),
            )
        # Strip the gripper joint(s) so callers get the 7 arm joints
        # the franky driver expects.
        return np.asarray(cfg)[:7]


def _solve_ik_jit(robot, coll, world_coll, T_target, target_link_idx, prev_cfg):
    """
    Module-level JIT entry point (so multiple planner instances share
    the same compiled kernel).
    """
    joint_var = robot.joint_var_cls(0)
    costs = [
        pk.costs.pose_cost(
            robot,
            joint_var,
            target_pose=T_target,
            target_link_index=target_link_idx,
            pos_weight=20.0,
            ori_weight=20.0,
        ),
        pk.costs.rest_cost(
            joint_var,
            rest_pose=prev_cfg,
            weight=0.1,
        ),
        pk.costs.self_collision_cost(
            robot,
            coll,
            joint_var,
            margin=0.01,
            weight=2.0,
        ),
        pk.costs.limit_cost(robot, joint_var, weight=100.0),
    ]
    for w in world_coll:
        costs.append(
            pk.costs.world_collision_cost(
                robot, coll, joint_var, w, margin=0.02, weight=20.0
            )
        )

    sol = (
        jaxls.LeastSquaresProblem(costs, [joint_var])
        .analyze()
        .solve(initial_vals=jaxls.VarValues.make([joint_var.with_value(prev_cfg)]))
    )
    return sol[joint_var]


__all__ = ["BimanualPyrokiPlanner"]

"""
Pyroki-based 6-DOF IK for LIBERO Panda.

Wraps pyroki's pose_cost_analytic_jac solver to get joint configs that reach
a target position + orientation reliably.  Replaces the hand-rolled
compute_ik_6dof, which diverges on horizontal palm orientations.

The first call JIT-compiles the solver (~5s); subsequent calls are fast.
"""
from __future__ import annotations

from functools import lru_cache
from typing import Optional

import numpy as np

try:
    import jax
    import jax.numpy as jnp
    import jaxlie
    import jaxls
    import pyroki as pk
    from robot_descriptions.loaders.yourdfpy import load_robot_description
except ImportError:
    jax = jnp = jaxlie = jaxls = pk = load_robot_description = None


_LIBERO_TARGET_LINK = "panda_hand_tcp"  # fingertip frame - matches LIBERO grip_site


@lru_cache(maxsize=1)
def _get_robot():
    """
    Load Panda URDF once per process.
    """
    urdf = load_robot_description("panda_description")
    robot = pk.Robot.from_urdf(urdf)
    return robot


def _try_solve(robot, link_index, target_wxyz, target_position):
    """
    Inner JAX solver; broken out so it can be cache-compiled.
    """
    joint_var = robot.joint_var_cls(0)
    costs = [
        pk.costs.pose_cost_analytic_jac(
            robot,
            joint_var,
            jaxlie.SE3.from_rotation_and_translation(
                jaxlie.SO3(target_wxyz), target_position
            ),
            link_index,
            pos_weight=50.0,
            ori_weight=10.0,
        ),
        pk.costs.limit_constraint(robot, joint_var),
    ]
    sol = (
        jaxls.LeastSquaresProblem(costs=costs, variables=[joint_var])
        .analyze()
        .solve(
            verbose=False,
            linear_solver="dense_cholesky",
            trust_region=jaxls.TrustRegionConfig(lambda_initial=1.0),
        )
    )
    return sol[joint_var]


def quat_from_approach(approach_dir_world: np.ndarray,
                       finger_close_world: Optional[np.ndarray] = None) -> np.ndarray:
    """
    Build a gripper orientation quaternion from an approach direction.

    The gripper's Z-axis (palm forward) is aligned with `approach_dir_world`
    (the direction the gripper moves into the object).  Its X-axis (one
    finger closing direction) is aligned with `finger_close_world` if given;
    otherwise a reasonable default is picked perpendicular to approach.

    Args:
        approach_dir_world: (3,) world-frame unit vector - the direction
            the gripper's palm faces (e.g. (0, -1, 0) for a horizontal grab
            facing -Y in world).
        finger_close_world: optional (3,) world vector - the direction one
            finger closes.  Defaults to world +Z (vertical close) for a
            horizontal approach, or world +X for a vertical approach.

    Returns:
        quat in (w, x, y, z) order.
    """
    z = np.asarray(approach_dir_world, dtype=np.float64)
    z = z / (np.linalg.norm(z) + 1e-9)
    if finger_close_world is None:
        # Pick a default perpendicular axis
        if abs(z[2]) < 0.9:
            x = np.array([0.0, 0.0, 1.0])  # world Z (vertical close)
        else:
            x = np.array([1.0, 0.0, 0.0])
    else:
        x = np.asarray(finger_close_world, dtype=np.float64)
    # Orthonormalise X against Z
    x = x - np.dot(x, z) * z
    x = x / (np.linalg.norm(x) + 1e-9)
    y = np.cross(z, x)
    R = np.column_stack([x, y, z])
    # Convert rotation matrix to (w, x, y, z) quaternion
    trace = R[0, 0] + R[1, 1] + R[2, 2]
    if trace > 0:
        s = np.sqrt(trace + 1.0) * 2
        w = 0.25 * s
        qx = (R[2, 1] - R[1, 2]) / s
        qy = (R[0, 2] - R[2, 0]) / s
        qz = (R[1, 0] - R[0, 1]) / s
    elif R[0, 0] > R[1, 1] and R[0, 0] > R[2, 2]:
        s = np.sqrt(1.0 + R[0, 0] - R[1, 1] - R[2, 2]) * 2
        w = (R[2, 1] - R[1, 2]) / s
        qx = 0.25 * s
        qy = (R[0, 1] + R[1, 0]) / s
        qz = (R[0, 2] + R[2, 0]) / s
    elif R[1, 1] > R[2, 2]:
        s = np.sqrt(1.0 + R[1, 1] - R[0, 0] - R[2, 2]) * 2
        w = (R[0, 2] - R[2, 0]) / s
        qx = (R[0, 1] + R[1, 0]) / s
        qy = 0.25 * s
        qz = (R[1, 2] + R[2, 1]) / s
    else:
        s = np.sqrt(1.0 + R[2, 2] - R[0, 0] - R[1, 1]) * 2
        w = (R[1, 0] - R[0, 1]) / s
        qx = (R[0, 2] + R[2, 0]) / s
        qy = (R[1, 2] + R[2, 1]) / s
        qz = 0.25 * s
    return np.array([w, qx, qy, qz])


def solve_ik_6dof(
    target_pos: np.ndarray,
    target_quat_wxyz: np.ndarray,
    target_link: str = _LIBERO_TARGET_LINK,
    robot_base_world_pos: Optional[np.ndarray] = None,
) -> Optional[np.ndarray]:
    """
    Solve 6-DOF IK, returning the actuated joint configuration.

    Args:
        target_pos: (3,) world-frame position.
        target_quat_wxyz: (4,) world-frame quaternion in (w, x, y, z) order
            - matches mujoco.mju_mat2Quat output.
        target_link: URDF link name to place at the target pose. For LIBERO
            Panda this is ``panda_hand`` (the body the grip_site hangs off).

    Returns:
        (7,) actuated joint config, or None if pyroki/jax missing.
    """
    if jnp is None:
        return None

    robot = _get_robot()
    try:
        link_index = robot.links.names.index(target_link)
    except ValueError:
        # Fall back to a hand-adjacent link if the exact name isn't present.
        for cand in ("panda_hand_tcp", "panda_link8", "panda_hand", "hand", "tool0"):
            if cand in robot.links.names:
                link_index = robot.links.names.index(cand)
                break
        else:
            return None

    # Pyroki URDF has robot base at origin; shift the world target into the
    # robot's base frame if the base isn't at the world origin.
    pos = np.asarray(target_pos, dtype=np.float64).copy()
    if robot_base_world_pos is not None:
        pos = pos - np.asarray(robot_base_world_pos, dtype=np.float64)

    cfg = _try_solve(
        robot,
        jnp.array(link_index),
        jnp.array(target_quat_wxyz, dtype=jnp.float32),
        jnp.array(pos, dtype=jnp.float32),
    )
    return np.array(cfg)

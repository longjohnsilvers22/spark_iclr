"""
UR10e inverse kinematics using pyroki (JAX-based).

Companion to fr3_ik_pyroki for the 6-DOF UR10e. Solves a jaxls
Levenberg-Marquardt factor graph (pose + limit + rest costs) for the
`tool0` flange frame. JIT-compiled: first call ~1s, subsequent calls ~2ms.

Public interface mirrors fr3_ik_pyroki.solve_ik so executor_ik can dispatch
by robot_family:
    solve_ik(target_pos, target_orient, q_seed, ...) -> q[6] | None

URDF: robot_descriptions "ur10e_description" (ur_description ur.urdf.xacro,
flattened by xacrodoc). TCP link = "tool0" (bare flange, NO Robotiq 2F-85).
The real robot's RTDE TCP is offset to the gripper tip; callers that target
the gripper tip use solve_ik_rtde, which removes the pendant TCP offset.
"""

from __future__ import annotations
import logging, os, time
from typing import Optional, Union

# CPU pinned: GPU JAX linalg solves fail here on a mixed nvidia wheel set
# (same rationale as fr3_ik_pyroki).
os.environ.setdefault("JAX_PLATFORMS", "cpu")
os.environ.setdefault("XLA_PYTHON_CLIENT_PREALLOCATE", "false")
os.environ.setdefault("XLA_PYTHON_CLIENT_MEM_FRACTION", "0.15")

import numpy as np
from scipy.spatial.transform import Rotation

try:
    import jax
    import jax.numpy as jnp
    import jaxlie
    import jaxls
    import pyroki
    from robot_descriptions.loaders.yourdfpy import load_robot_description
except ImportError:
    jax = jnp = jaxlie = jaxls = pyroki = None
    load_robot_description = None

logger = logging.getLogger(__name__)

# UR10e IK seed / rest pose (elbow-up wrist-down grasp posture). Kept distinct
# from ur10e_driver.HOME_CONFIG so the solver rests toward a downward-facing
# grasp configuration rather than the mechanical home.
_REST = np.array([1.5708, -1.5708, 1.5708, -1.5708, -1.5708, 0.0], dtype=np.float32)

_ROBOT_DESC = "ur10e_description"
_TCP_LINK = "tool0"

_robot = None
_tli: Optional[int] = None
_nact: Optional[int] = None
_solver = None
_last_q: Optional[np.ndarray] = None


def _load():
    global _robot, _tli, _nact
    if _robot is not None:
        return
    if pyroki is None or load_robot_description is None:
        raise RuntimeError(
            "pyroki/JAX/robot_descriptions not installed; UR10e pyroki IK unavailable"
        )
    urdf = load_robot_description(_ROBOT_DESC)
    _robot = pyroki.Robot.from_urdf(urdf)
    names = list(_robot.links.names)
    if _TCP_LINK not in names:
        raise RuntimeError(f"No '{_TCP_LINK}' link in UR10e URDF. Available: {names}")
    _tli = names.index(_TCP_LINK)
    _nact = int(_robot.joints.num_actuated_joints)
    logger.info("UR10e IK (pyroki): %d links, %d joints, tcp=%s", len(names), _nact, _TCP_LINK)


def _build():
    global _solver
    if _solver is not None:
        return
    pk = pyroki.costs
    tli = jnp.array(_tli, dtype=jnp.int32)

    def _solve(tp, tq, sd, pw, ow, rw, lw):
        pose = jaxlie.SE3.from_rotation_and_translation(jaxlie.SO3(tq), tp)
        jv = _robot.joint_var_cls(0)
        fs = [
            pk.pose_cost(_robot, jv, pose, tli, pos_weight=pw, ori_weight=ow),
            pk.limit_cost(_robot, jv, weight=lw),
            pk.rest_cost(jv, rest_pose=sd, weight=rw),
        ]
        prob = jaxls.LeastSquaresProblem(fs, [jv]).analyze()
        sol = prob.solve(
            initial_vals=jaxls.VarValues.make([jv.with_value(sd)]),
            linear_solver="dense_cholesky",
            termination=jaxls.TerminationConfig(max_iterations=150),
            verbose=False,
        )
        qf = sol[jv]
        return qf, jaxlie.SE3(_robot.forward_kinematics(qf)[_tli]).translation()

    _solver = jax.jit(_solve)


def _seed(q_seed: np.ndarray) -> np.ndarray:
    """Coerce an arbitrary seed to a 6-vec float32 (pad/truncate with rest)."""
    s = _REST.copy()
    q = np.asarray(q_seed, dtype=np.float32).ravel()
    n = min(6, q.size)
    s[:n] = q[:n]
    return s


def solve_ik(
    target_pos: np.ndarray,
    target_orient: Union[np.ndarray, list],
    q_seed: np.ndarray,
    pos_weight: float = 300.0,
    ori_weight: float = 100.0,
    rest_weight: float = 1.5,
    limit_weight: float = 200.0,
    **_kw,
) -> Optional[np.ndarray]:
    """
    Solve UR10e IK for the tool0 frame.

    Args:
        target_pos: (3,) world position of tool0 (meters, robot base frame).
        target_orient: (3,) axis-angle rotvec or (3,3) rotation matrix.
        q_seed: current joint config (>=6 used); seeds the solve + tie-break.

    Returns:
        (6,) joint positions, or None on failure / >5cm residual.
    """
    global _last_q
    _load()
    _build()

    target_pos = np.asarray(target_pos, dtype=float).reshape(3)
    to = np.asarray(target_orient, dtype=float)
    if to.shape == (3,):
        Rt = Rotation.from_rotvec(to).as_matrix()
    elif to.shape == (3, 3):
        Rt = to
    else:
        raise ValueError(f"target_orient must be (3,) or (3,3), got {to.shape}")

    sd = _seed(q_seed)
    q_ref = np.asarray(q_seed, dtype=float).ravel()

    def _one(R):
        qx = Rotation.from_matrix(R).as_quat()  # xyzw
        qw = np.array([qx[3], qx[0], qx[1], qx[2]], dtype=np.float32)  # wxyz
        try:
            qj, ap = _solver(
                jnp.asarray(target_pos, jnp.float32),
                jnp.asarray(qw, jnp.float32),
                jnp.asarray(sd, jnp.float32),
                jnp.float32(pos_weight),
                jnp.float32(ori_weight),
                jnp.float32(rest_weight),
                jnp.float32(limit_weight),
            )
        except Exception as exc:  # noqa: BLE001
            logger.debug("[ur pyroki IK] solve raised: %s", exc)
            return None, None
        # numpy conversion OUTSIDE the jit (asarray inside traces->error).
        return np.asarray(qj, dtype=float).ravel(), np.asarray(ap, dtype=float)

    # Parallel-jaw yaw symmetry: try the requested wrist yaw and yaw+pi.
    Ra = Rt @ Rotation.from_euler("z", np.pi).as_matrix()
    cands = []
    for R in (Rt, Ra):
        q, ap = _one(R)
        if q is None or q.size < 6:
            continue
        err = float(np.linalg.norm(ap - target_pos))
        dq = float(np.max(np.abs(q[:6] - _seed(q_seed)[:6])))
        cands.append((q, ap, dq, err))

    cands = [c for c in cands if c[3] <= 0.05]
    if not cands:
        logger.warning("[ur pyroki IK] no candidate within 5cm")
        return None
    cands.sort(key=lambda c: c[2])  # smallest joint swing
    qb, ab, dqb, eb = cands[0]

    qb = np.asarray(qb, dtype=float).copy()
    logger.info("[ur pyroki IK] solved dq=%.2f pos_err=%.4fm", dqb, eb)
    _last_q = np.asarray(qb, dtype=np.float32).copy()
    return qb[:6]


# Hardware frame handling (RTDE base <-> URDF base_link).
# The UR RTDE base frame is a pure Rz(180) from the URDF base_link frame pyroki
# uses. getTCPOffset() gives the pendant TCP (gripper tip relative to tool0).
# solve_ik_rtde converts an RTDE-frame gripper-TIP target into a URDF-frame
# tool0 target and solves.


def _se3(p, rv_or_R):
    T = np.eye(4)
    a = np.asarray(rv_or_R, float)
    T[:3, :3] = a if a.shape == (3, 3) else Rotation.from_rotvec(a).as_matrix()
    T[:3, 3] = np.asarray(p, float).reshape(3)
    return T


def _inv(T):
    Ti = np.eye(4)
    Ti[:3, :3] = T[:3, :3].T
    Ti[:3, 3] = -T[:3, :3].T @ T[:3, 3]
    return Ti


def _rz180():
    return Rotation.from_euler("z", np.pi).as_matrix()


def solve_ik_rtde(
    tip_pos: np.ndarray,
    tip_orient: Union[np.ndarray, list],
    q_seed: np.ndarray,
    tcp_offset: Union[np.ndarray, list],
    **kw,
) -> Optional[np.ndarray]:
    """
    Solve UR10e IK for a gripper-TIP target expressed in the RTDE base frame.

    Handles the two hardware frame facts bare solve_ik does not:
      1. RTDE base <-> URDF base_link differ by Rz(180 deg).
      2. The pendant TCP (getTCPOffset) offsets the gripper tip from tool0.

    Args:
        tip_pos: (3,) gripper-tip target, RTDE base frame.
        tip_orient: (3,) rotvec or (3,3) matrix, tip orientation, RTDE frame.
        q_seed: current joints (seed + tie-break).
        tcp_offset: (6,) [x,y,z,rx,ry,rz] from RTDEControl.getTCPOffset()
            (tip relative to tool0, tool0 frame).
    Returns:
        (6,) joints, or None on failure.
    """
    to = np.asarray(tip_orient, float)
    R_tip = to if to.shape == (3, 3) else Rotation.from_rotvec(to.reshape(3)).as_matrix()
    tcp = np.asarray(tcp_offset, float).reshape(6)
    T_base_tip = _se3(np.asarray(tip_pos, float).reshape(3), R_tip)
    T_tool0_tip = _se3(tcp[:3], tcp[3:])
    T_base_tool0 = T_base_tip @ _inv(T_tool0_tip)   # RTDE-frame tool0 target
    Rz = _rz180()                                    # URDF <- RTDE (== its own inverse)
    pos_urdf = Rz @ T_base_tool0[:3, 3]
    R_urdf = Rz @ T_base_tool0[:3, :3]
    return solve_ik(pos_urdf, R_urdf, q_seed, **kw)


def fk(q: np.ndarray) -> tuple:
    """Forward kinematics for tool0. Returns (pos[3], R[3x3])."""
    _load()
    qf = _seed(q)
    T = jaxlie.SE3(_robot.forward_kinematics(jnp.asarray(qf))[_tli])
    return (
        np.asarray(T.translation(), dtype=float),
        np.asarray(T.rotation().as_matrix(), dtype=float),
    )


def warmup():
    """Pre-compile the IK solver."""
    _load()
    p, _ = fk(_REST)
    solve_ik(p, Rotation.from_euler("xyz", [np.pi, 0, 0]).as_matrix(), _REST)


def benchmark():
    _load()
    tp, _ = fk(_REST)
    R = Rotation.from_euler("xyz", [np.pi, 0, 0]).as_matrix()
    t0 = time.time()
    q = solve_ik(tp, R, _REST)
    print(f"first: {(time.time()-t0)*1000:.0f}ms  q={None if q is None else q.round(3)}")
    for i in range(5):
        t0 = time.time()
        solve_ik(tp + [0.01 * i, 0, 0], R, _REST)
        print(f"call {i+2}: {(time.time()-t0)*1000:.2f}ms")


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    benchmark()

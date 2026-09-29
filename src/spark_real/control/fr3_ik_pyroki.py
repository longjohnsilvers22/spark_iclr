"""
FR3 inverse kinematics using pyroki (JAX-based).

Drop-in replacement for fr3_ik.solve_ik. Uses Levenberg-Marquardt over
a jaxls factor graph with pose, limit, rest, and manipulability costs.
JIT-compiled: first call ~7-10s, subsequent calls <100ms.
"""

from __future__ import annotations
import logging, os, time
from typing import Optional, Union

# CPU pinned: GPU JAX linalg solves fail here on a mixed nvidia wheel set.
# Clean the nvidia-* wheels before retrying JAX_PLATFORMS=cuda.
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
    import yourdfpy
    import pyroki
    from jaxls import Cost, Var, VarValues
    from jax import Array
except ImportError:
    jax = jnp = jaxlie = jaxls = yourdfpy = pyroki = None
    Cost = Var = VarValues = Array = None

logger = logging.getLogger(__name__)
_URDF = os.path.join(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
    "robots",
    "franka",
    "urdf",
    "fr3_franka_hand.urdf",
)
_robot = _rcoll = _ths = None
_tli = _nact = None
_solvers: dict = {}
_last_q: Optional[np.ndarray] = None
_VEL_DT, _VEL_W = 0.01, 0.1
_SC_M, _SC_W, _WC_M, _WC_W, _TZ = 0.02, 5.0, 0.02, 5.0, -0.025


def _load():
    global _robot, _tli, _nact
    if _robot is not None:
        return
    if not os.path.exists(_URDF):
        raise FileNotFoundError(f"FR3 URDF not found at {_URDF}")
    if pyroki is None:
        raise RuntimeError("pyroki/JAX not installed; FR3 pyroki IK unavailable")
    _robot = pyroki.Robot.from_urdf(yourdfpy.URDF.load(_URDF))
    names = list(_robot.links.names)
    for c in ("fr3_hand_tcp", "fr3_hand", "panda_hand_tcp"):
        if c in names:
            _tli = names.index(c)
            logger.info("FR3 IK (pyroki): %d links, tcp=%s", len(names), c)
            break
    if _tli is None:
        raise RuntimeError(f"No TCP link in URDF. Available: {names[:20]}...")
    _nact = int(_robot.joints.num_actuated_joints)


def _load_coll():
    global _rcoll, _ths
    if _rcoll is not None:
        return
    _rcoll = pyroki.collision.RobotCollision.from_urdf(yourdfpy.URDF.load(_URDF))
    _ths = pyroki.collision.HalfSpace.from_point_and_normal(
        point=jnp.array([0.0, 0.0, _TZ], dtype=jnp.float32),
        normal=jnp.array([0.0, 0.0, 1.0], dtype=jnp.float32),
    )


def _solver(sm: bool, co: bool):
    k = (sm, co)
    if k not in _solvers:
        _load()
        if co:
            _load_coll()
        _solvers[k] = _build(sm, co)
    return _solvers[k]


def _build(sm: bool, co: bool):
    pk = pyroki.costs
    tli = jnp.array(_tli, dtype=jnp.int32)
    if sm:

        @Cost.create_factory
        def _vc(v: VarValues, r, jv: Var[Array], p: Array, dt: float, w) -> Array:
            return (
                jnp.maximum(0.0, jnp.abs((v[jv] - p) / dt) - r.joints.velocity_limits)
                * w
            ).flatten()

    def _solve(tp, tq, sd, pw, ow, rw, lw, mw, *ex):
        ex = iter(ex)
        if sm:
            prev, vw = next(ex), next(ex)
        if co:
            scw, wcw, tg = next(ex), next(ex), next(ex)
        pose = jaxlie.SE3.from_rotation_and_translation(jaxlie.SO3(tq), tp)
        jv = _robot.joint_var_cls(0)
        fs = [
            pk.pose_cost(_robot, jv, pose, tli, pos_weight=pw, ori_weight=ow),
            pk.limit_cost(_robot, jv, weight=lw),
            pk.rest_cost(jv, rest_pose=sd, weight=rw),
            pk.manipulability_cost(_robot, jv, tli, weight=mw),
        ]
        if sm:
            fs.append(_vc(_robot, jv, prev, _VEL_DT, vw))
        if co:
            fs.append(
                pk.self_collision_cost(
                    _robot, robot_coll=_rcoll, joint_var=jv, margin=_SC_M, weight=scw
                )
            )
            fs.append(
                pk.world_collision_cost(
                    _robot,
                    robot_coll=_rcoll,
                    joint_var=jv,
                    world_geom=tg,
                    margin=_WC_M,
                    weight=wcw,
                )
            )
        prob = jaxls.LeastSquaresProblem(fs, [jv]).analyze()
        sol = prob.solve(
            initial_vals=jaxls.VarValues.make([jv.with_value(sd)]),
            linear_solver="dense_cholesky",
            termination=jaxls.TerminationConfig(max_iterations=150),
            verbose=False,
        )
        qf = sol[jv]
        return qf, jaxlie.SE3(_robot.forward_kinematics(qf)[_tli]).translation()

    return jax.jit(_solve)


def _seed(q7):
    s = np.zeros(_nact, dtype=np.float32)
    n = min(7, _nact)
    s[:n] = q7[:n]
    if abs(s[4]) < 0.30:
        s[4] = -0.50 if s[4] <= 0 else 0.50
    if _nact > 7:
        s[7:] = 0.02
    return s


def _wj7(q, ref):
    q7 = q[6]
    best, bok, bd = q7, abs(q7) <= 2.85, abs(q7 - ref)
    for d in (-np.pi, np.pi):
        c, ok, di = q7 + d, abs(q7 + d) <= 2.85, abs(q7 + d - ref)
        if (ok and not bok) or (ok == bok and di < bd):
            best, bok, bd = c, ok, di
    q[6] = best


def solve_ik(
    target_pos: np.ndarray,
    target_orient: Union[np.ndarray, list],
    q_seed: np.ndarray,
    pos_weight: float = 300.0,
    ori_weight: float = 100.0,
    rest_weight: float = 1.5,
    limit_weight: float = 200.0,
    manip_weight: float = 1.0,
    **_kw,
) -> Optional[np.ndarray]:
    """
    Solve FR3 IK. Returns 7-vec joint positions or None on failure.
    """
    global _last_q
    _load()
    uv = os.environ.get("SPARK_IK_VEL_SMOOTHING", "0") == "1"
    uc = os.environ.get("SPARK_IK_COLLISION", "0") == "1"
    slv = _solver(uv, uc)

    target_pos = np.asarray(target_pos, dtype=float).reshape(3)
    to = np.asarray(target_orient, dtype=float)
    if to.shape == (3,):
        Rt = Rotation.from_rotvec(to).as_matrix()
    elif to.shape == (3, 3):
        Rt = to
    else:
        raise ValueError(f"target_orient must be (3,) or (3,3), got {to.shape}")

    q_seed = np.asarray(q_seed, dtype=float).ravel()
    if q_seed.size < 7:
        raise ValueError(f"q_seed needs >= 7 entries, got {q_seed.size}")
    sd = _seed(q_seed)
    pv = _last_q if _last_q is not None else sd

    def _one(R, rw):
        qx = Rotation.from_matrix(R).as_quat()
        qw = np.array([qx[3], qx[0], qx[1], qx[2]], dtype=np.float32)
        a = (
            jnp.asarray(target_pos, jnp.float32),
            jnp.asarray(qw, jnp.float32),
            jnp.asarray(sd, jnp.float32),
            jnp.float32(pos_weight),
            jnp.float32(ori_weight),
            jnp.float32(rw),
            jnp.float32(limit_weight),
            jnp.float32(manip_weight),
        )
        ex = ()
        if uv:
            ex += (jnp.asarray(pv, jnp.float32), jnp.float32(_VEL_W))
        if uc:
            ex += (jnp.float32(_SC_W), jnp.float32(_WC_W), _ths)
        try:
            qj, ap = slv(*a, *ex)
        except Exception:
            return None, None
        return np.asarray(qj, dtype=float).ravel(), np.asarray(ap, dtype=float)

    # Solve both yaw orientations (gripper symmetry: yaw and yaw+pi).
    Ra = Rt @ Rotation.from_euler("z", np.pi).as_matrix()
    cands = []
    for R, tag in ((Rt, "primary"), (Ra, "yaw+pi")):
        q, ap = _one(R, rest_weight)
        if q is None or q.size < 7:
            continue
        _wj7(q, q_seed[6])
        cands.append(
            (
                q,
                ap,
                np.max(np.abs(q[:7] - q_seed[:7])),
                float(np.linalg.norm(ap - target_pos)),
                tag,
            )
        )

    if not cands:
        logger.warning("[pyroki IK] no candidate solved")
        return None
    cands = [c for c in cands if c[3] <= 0.05]
    if not cands:
        logger.warning("[pyroki IK] all candidates >5cm off")
        return None
    # Prefer J5 away from the wrist singularity; do not add an upper band
    # (it breaks candidate continuity across chained waypoints). The hard
    # clamp below handles the limit case.
    ns = [c for c in cands if abs(c[0][4]) >= 0.25]
    if ns:
        cands = ns
    cands.sort(key=lambda c: c[2])
    qb, ab, dqb, eb, tb = cands[0]

    # Heavy-rest retry for large swing or J5 near singularity.
    if dqb > 1.0 or abs(qb[4]) < 0.25:
        logger.info("[pyroki IK] dq=%.2f/J5=%.2f; heavy-rest retry", dqb, qb[4])
        q2, a2 = _one(Rt if tb == "primary" else Ra, rest_weight * 4)
        if q2 is not None and q2.size >= 7:
            _wj7(q2, q_seed[6])
            d2, e2 = np.max(np.abs(q2[:7] - q_seed[:7])), float(
                np.linalg.norm(a2 - target_pos)
            )
            if e2 <= 0.05 and d2 < dqb:
                qb, ab, dqb, eb = q2, a2, d2, e2

    if eb > 0.05:
        return None
    # Hard-clamp to joint limits minus a margin: the soft limit_cost can
    # let a solution land just past a bound, which libfranka rejects with
    # joint_position_limits_violation and a slow fallback cascade.
    lo = np.asarray(_robot.joints.lower_limits, dtype=float)[:7] + 0.02
    hi = np.asarray(_robot.joints.upper_limits, dtype=float)[:7] - 0.02
    qb = np.asarray(qb, dtype=float).copy()
    qb[:7] = np.clip(qb[:7], lo, hi)
    logger.info("[pyroki IK] solved dq=%.2f pos_err=%.4fm", dqb, eb)
    _last_q = np.asarray(qb, dtype=np.float32).copy()
    return qb[:7]


def fk(q: np.ndarray) -> tuple:
    """
    Forward kinematics for TCP. Returns (pos[3], R[3x3]).
    """
    _load()
    qf = np.zeros(_nact, dtype=np.float32)
    q = np.asarray(q, dtype=float).ravel()
    qf[: min(7, _nact)] = q[: min(7, _nact)]
    if _nact > 7:
        qf[7:] = 0.02
    T = jaxlie.SE3(_robot.forward_kinematics(jnp.asarray(qf))[_tli])
    return np.asarray(T.translation(), dtype=float), np.asarray(
        T.rotation().as_matrix(), dtype=float
    )


def jacobian(q: np.ndarray) -> np.ndarray:
    """
    6x7 geometric Jacobian at the TCP frame, in world coordinates.
    """
    _load()
    q = np.asarray(q, dtype=float).ravel()
    qf = np.zeros(_nact, dtype=np.float32)
    qf[: min(7, _nact)] = q[: min(7, _nact)]
    if _nact > 7:
        qf[7:] = 0.02
    jq = jnp.asarray(qf)

    # Numerical Jacobian via finite differences (fast enough for teleop).
    eps = 1e-4
    T0 = jaxlie.SE3(_robot.forward_kinematics(jq)[_tli])
    p0 = np.asarray(T0.translation(), dtype=float)
    R0 = np.asarray(T0.rotation().as_matrix(), dtype=float)
    J = np.zeros((6, 7), dtype=float)
    for i in range(7):
        qp = qf.copy()
        qp[i] += eps
        Tp = jaxlie.SE3(_robot.forward_kinematics(jnp.asarray(qp))[_tli])
        pp = np.asarray(Tp.translation(), dtype=float)
        Rp = np.asarray(Tp.rotation().as_matrix(), dtype=float)
        J[:3, i] = (pp - p0) / eps
        dR = Rp @ R0.T
        cos_a = np.clip((np.trace(dR) - 1) / 2, -1, 1)  # axis-angle from matrix
        angle = np.arccos(cos_a)
        if abs(angle) > 1e-8:
            axis = np.array(
                [dR[2, 1] - dR[1, 2], dR[0, 2] - dR[2, 0], dR[1, 0] - dR[0, 1]]
            )
            axis = axis / (2 * np.sin(angle) + 1e-12)
            J[3:, i] = axis * angle / eps
    return J


def warmup():
    """
    Pre-compile the IK solver.
    """
    _load()
    # IK seed only (not the ready pose): rounded J5=0 variant, intentionally
    # differs from franka_base.HOME_CONFIG / franka_default.yaml home_config.
    solve_ik(
        np.array([0.5, 0, 0.3]),
        Rotation.from_euler("xyz", [np.pi, 0, 0]).as_matrix(),
        np.array([0, -0.785, 0, -2.356, 0, 1.571, 0.785]),
    )


def benchmark():
    # IK seed only (not the ready pose): rounded J5=0 variant, intentionally
    # differs from franka_base.HOME_CONFIG / franka_default.yaml home_config.
    q0, tp = np.array([0, -0.785, 0, -2.356, 0, 1.571, 0.785]), np.array([0.5, 0, 0.3])
    R = Rotation.from_euler("xyz", [np.pi, 0, 0]).as_matrix()
    t0 = time.time()
    q = solve_ik(tp, R, q0)
    print(f"first: {(time.time()-t0)*1000:.0f}ms  q={q}")
    for i in range(5):
        t0 = time.time()
        q = solve_ik(tp + [0.01 * i, 0, 0], R, q0)
        print(f"call {i+2}: {(time.time()-t0)*1000:.1f}ms")


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    benchmark()

# Shared utilities for grasp skills: joint motion, orientation helpers.

import logging
import time

import numpy as np
from typing import Optional

from spark_real.skills.primitives import _result
from spark_real.utils.rotations import rotmat_to_rotvec as _rotation_matrix_to_rotvec
from scipy.spatial.transform import Rotation as _R

try:
    from spark_real.control.fr3_ik_pyroki import solve_ik as _solve_ik
except ImportError:
    _solve_ik = None

try:
    import franky
except ImportError:
    franky = None

logger = logging.getLogger(__name__)

# Substrings that always mark a benign franky abort that callers may
# absorb (controller reflex, command aborted, rejected motion, or a
# near-singular pose). Two contexts add one extra keyword each via the
# `extra_keywords` argument: JointMotion fallbacks also treat "motion
# finished commanded" as benign, while servo descents also treat
# "singular" as benign.
_BENIGN_FRANKY_REFLEX_KEYWORDS = (
    "reflex",
    "aborted",
    "rejected",
    "discontinuity",
)


def is_benign_franky_reflex(exc: Exception, extra_keywords=()) -> bool:
    """Return True when an exception is a benign franky controller abort.

    Args:
        exc: The exception raised by a motion command.
        extra_keywords: Additional context-specific substrings to treat as
            benign, e.g. ("motion finished commanded",) for JointMotion
            fallbacks or ("singular",) for servo descents.
    """
    msg = str(exc).lower()
    keywords = _BENIGN_FRANKY_REFLEX_KEYWORDS + tuple(extra_keywords)
    return any(k in msg for k in keywords)


def canonical_symmetric_yaw(yaw: float) -> float:
    """Fold a yaw into the two-jaw canonical range via modulo pi.

    A two-jaw gripper grasps identically at yaw and yaw+pi, so any angle
    can be reduced modulo pi and then folded into (-pi/2, pi/2]. This is
    the modulo-pi form used by the OBB-derived grasp yaws; it differs
    from the no-modulo single-fold form in place_in_slot / sweep, which
    assume an already-bounded input, so those sites are intentionally not
    routed here. Distinct from `_nearest_symmetric_yaw`, which instead
    minimizes wrist travel against the current pose after this fold.
    """
    yaw = yaw % np.pi
    if yaw > np.pi / 2:
        yaw -= np.pi
    return yaw


def validate_grasp_orientation(rot: np.ndarray) -> np.ndarray:
    """
    Re-orthogonalize a rotation matrix via SVD to fix numerical drift.
    """
    U, _, Vt = np.linalg.svd(rot)
    R = U @ Vt
    if np.linalg.det(R) < 0:
        U[:, -1] *= -1
        R = U @ Vt
    return R


def orient_diag(tag: str, target_rotvec, achieved_R=None) -> None:
    """
    Log TCP-Z direction and tilt error for orientation diagnostics.
    """
    try:
        target_R = _R.from_rotvec(np.asarray(target_rotvec, dtype=float)).as_matrix()
        tz = target_R[:, 2]
        tilt_deg = float(np.degrees(np.arccos(np.clip(-tz[2], -1.0, 1.0))))
        msg = (
            "[diag/%s] target TCP-Z=(%+.3f,%+.3f,%+.3f) "
            "tilt_from_down=%.2f deg" % (tag, tz[0], tz[1], tz[2], tilt_deg)
        )
        if achieved_R is not None:
            az = np.asarray(achieved_R)[:, 2]
            dot = float(np.clip(np.dot(tz, az), -1.0, 1.0))
            err_deg = float(np.degrees(np.arccos(dot)))
            ach_tilt_deg = float(np.degrees(np.arccos(np.clip(-az[2], -1.0, 1.0))))
            msg += (
                " | achieved TCP-Z=(%+.3f,%+.3f,%+.3f) "
                "achieved_tilt_from_down=%.2f deg orient_err=%.2f deg"
                % (az[0], az[1], az[2], ach_tilt_deg, err_deg)
            )
        logger.info(msg)
    except Exception as exc:
        logger.debug("[diag/%s] orient log failed: %s", tag, exc)


def joint_motion_to(
    executor,
    position,
    orientation_rotvec,
    velocity_factor: float = 0.6,
    label: str = "joint_move",
) -> bool:
    """
    Solve IK + dispatch a franky.JointMotion to (position, orient).

    Joint-space alternative to executor._move_to. Bypasses libfranka's
    Cartesian planner which on FR3 can reject with "cannot start at
    singular pose" or trip joint_velocity_discontinuity.

    Returns True on success; False on any failure (callers should fall
    back to Cartesian).
    """
    if not hasattr(executor.robot, "get_joint_positions"):
        return False

    # Joint-space IK here is FR3-specific (fr3_ik_pyroki solves a 7-DOF Franka
    # chain and raises ValueError on a <7 seed). On a 6-DOF arm or any
    # non-franka family, skip it and signal the caller to use its Cartesian path.
    if not _franka_joint_chain(executor):
        logger.info(
            "[top_down/%s] non-FR3 joint chain (<7 joints); cartesian fallback",
            label,
        )
        return False

    if _solve_ik is None:
        logger.info(
            "[top_down/%s] pyroki IK unavailable; cartesian fallback",
            label,
        )
        return False

    q_seed = np.asarray(executor.robot.get_joint_positions(), dtype=float)[:7].copy()
    q_target = _solve_ik(
        np.asarray(position, dtype=float),
        np.asarray(orientation_rotvec, dtype=float),
        q_seed,
    )
    if q_target is None:
        logger.info("[top_down/%s] IK returned None; cartesian fallback", label)
        return False

    dq_max = float(np.max(np.abs(np.asarray(q_target) - q_seed)))

    # Walk wrapper chain to find a franky.Robot with .move()
    inner = _resolve_franky_robot(executor.robot)

    if inner is None or not hasattr(inner, "move"):
        try:
            executor.robot.move_to_joint_config(
                q_target.tolist(),
                velocity=executor.velocity * velocity_factor,
            )
            logger.info(
                "[top_down/%s] via move_to_joint_config dq_max=%.2f rad",
                label,
                dq_max,
            )
            return True
        except Exception as exc:
            logger.warning("[top_down/%s] move_to_joint_config failed: %s", label, exc)
            return False

    if franky is None:
        logger.info("[top_down/%s] franky unavailable; cartesian fallback", label)
        return False
    try:
        if getattr(inner, "has_errors", False):
            try:
                inner.recover_from_errors()
            except Exception:
                pass
        # Command the JointMotion only once residual velocity from the previous
        # motion has bled off, so the velocity/acceleration discontinuity reflex
        # never fires.
        try:
            gate_joint_stationary(executor, inner)
        except Exception as exc:
            logger.debug("[top_down/%s] gate skipped: %s", label, exc)

        rdf_val = float(np.clip(executor.velocity * velocity_factor / 0.25, 0.05, 0.25))
        rdf = franky.RelativeDynamicsFactor(rdf_val)
        motion = franky.JointMotion(q_target.tolist(), relative_dynamics_factor=rdf)
        logger.info(
            "[top_down/%s] JointMotion dq_max=%.2f rad rdf=%.2f",
            label,
            dq_max,
            rdf_val,
        )
        inner.move(motion, asynchronous=False)
        return True
    except Exception as exc:
        if is_benign_franky_reflex(exc, ("motion finished commanded",)):
            logger.info(
                "[top_down/%s] JointMotion absorbed reflex (%s); driver fallback",
                label,
                exc,
            )
            try:
                executor.robot.move_to_joint_config(
                    q_target.tolist(),
                    velocity=executor.velocity * velocity_factor * 0.7,
                )
                return True
            except Exception as exc2:
                logger.warning("[top_down/%s] driver fallback failed: %s", label, exc2)
                return False
        logger.warning("[top_down/%s] JointMotion raised: %s", label, exc)
        return False


def _franka_joint_chain(executor) -> bool:
    """
    True only for a 7-DOF franka-derived arm (FR3/Panda).

    Guards the fr3_ik_pyroki joint-space path: it solves a 7-DOF Franka
    chain and raises ValueError on a seed with <7 entries, so a 6-DOF
    UR10e (or any non-franka family) must skip it and fall back to the
    caller's Cartesian path. Checks the live joint count first (the
    authoritative signal) and the resolved family as a backstop.
    """
    robot = getattr(executor, "robot", None)
    if robot is None:
        return False
    # Live joint count: <7 is decisive (UR10e reports 6).
    try:
        q = robot.get_joint_positions()
        if q is not None and len(q) < 7:
            return False
    except Exception:
        # Unreadable joint state: fall through to the family check.
        pass
    # Family backstop: walk the wrapper chain like executor_core does, then
    # the pipeline config. Anything not franka-derived is not this path.
    family = getattr(robot, "robot_family", None) or getattr(
        getattr(robot, "_robot", None), "robot_family", None
    )
    if family is None:
        cfg = getattr(getattr(executor, "_pipeline", None), "config", None)
        family = getattr(cfg, "robot_family", None)
    if family is None:
        # No family signal and joints were readable as >=7: allow the FR3
        # path (preserves single-arm Franka behavior when family is unset).
        return True
    return str(family).lower() in ("franka", "bimanual_franka")


def _resolve_franky_robot(robot):
    """
    Walk the SafeRobot/FrankaDriver wrapper chain to find a franky.Robot.
    """
    _cur = robot
    for _ in range(4):
        if _cur is None:
            break
        if hasattr(_cur, "move") and hasattr(_cur, "state"):
            return _cur
        _cur = getattr(_cur, "_robot", None)
    return None


def _resolve_driver(robot):
    """
    Walk the wrapper chain to find the FrankaDriver (has gate_joint_stationary).
    """
    _cur = robot
    for _ in range(4):
        if _cur is None:
            break
        if hasattr(_cur, "gate_joint_stationary"):
            return _cur
        _cur = getattr(_cur, "_robot", None)
    return None


def gate_joint_stationary(
    executor, inner, vel_tol: float = 0.01, timeout: float = 0.5
) -> float:
    """
    Fast stationarity gate before a direct franky.JointMotion.

    Prefers the driver's shared gate (FrankaDriver.gate_joint_stationary).
    Falls back to a minimal inline poll on the raw franky robot's joint
    velocities. Single state read when already still (no sleep); waits at
    most `timeout` otherwise. Returns wait seconds (0.0 on the fast path).
    """
    drv = _resolve_driver(getattr(executor, "robot", None))
    if drv is not None:
        try:
            return float(drv.gate_joint_stationary(vel_tol=vel_tol, timeout=timeout))
        except Exception as exc:
            logger.debug("driver gate unavailable (%s); inline fallback", exc)

    def _dq_max():
        try:
            v = np.asarray(inner.current_joint_velocities, dtype=float)
        except Exception:
            try:
                v = np.asarray(inner.state.dq, dtype=float)
            except Exception:
                return None
        return float(np.max(np.abs(v)))

    dqm = _dq_max()
    if dqm is None or dqm < vel_tol:
        return 0.0

    t0 = time.monotonic()
    deadline = t0 + timeout
    while time.monotonic() < deadline:
        time.sleep(0.01)
        dqm = _dq_max()
        if dqm is None or dqm < vel_tol:
            break
    waited = time.monotonic() - t0
    logger.info(
        "[top_down] joint stationarity gate waited %.0f ms "
        "(dq_max=%.4f rad/s) before JointMotion",
        waited * 1e3,
        dqm if dqm is not None else float("nan"),
    )
    return waited


def horizontal_grasp_orientation(
    approach_xy: np.ndarray, pitch_rad: float = 0.0
) -> list:
    """
    Build a TCP rotvec for a horizontal / angled side grasp.

    URDF panda_hand frame convention:
      - TCP-Z is the gripper approach axis (toward the object)
      - TCP-Y is the jaw closure axis
      - TCP-X = cross(Y, Z), approximately +world_Z

    pitch_rad tilts the approach DOWN from horizontal.
    approach_xy is the 2-vector pointing from pre-grasp toward the object.
    """

    a = np.asarray(approach_xy, dtype=float).reshape(-1)
    if a.size == 2:
        a = np.array([a[0], a[1], 0.0], dtype=float)
    else:
        a = np.array([a[0], a[1], 0.0], dtype=float)
    n = float(np.linalg.norm(a))
    if n < 1e-6:
        a = np.array([-1.0, 0.0, 0.0], dtype=float)
    else:
        a = a / n

    cos_p = float(np.cos(pitch_rad))
    sin_p = float(np.sin(pitch_rad))
    z_axis = cos_p * a + np.array([0.0, 0.0, -sin_p], dtype=float)
    z_axis = z_axis / (np.linalg.norm(z_axis) + 1e-12)

    world_z = np.array([0.0, 0.0, 1.0], dtype=float)
    y_axis = np.cross(world_z, z_axis)
    yn = float(np.linalg.norm(y_axis))
    if yn < 1e-6:
        y_axis = np.array([0.0, 1.0, 0.0], dtype=float)
    else:
        y_axis = y_axis / yn

    x_axis = np.cross(y_axis, z_axis)
    x_axis = x_axis / (np.linalg.norm(x_axis) + 1e-12)
    R = np.column_stack([x_axis, y_axis, z_axis])
    R = validate_grasp_orientation(R)
    return _R.from_matrix(R).as_rotvec().tolist()

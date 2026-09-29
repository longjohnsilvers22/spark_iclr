"""
Spatio-temporal waypoint graph for SPARK motion.

Leaves emit ``Waypoint`` notes onto a shared ``WaypointBuffer``, and the buffer
flushes them as a single Ruckig-blended motion through all waypoints
(``hold_target_duration=0`` between them) until a barrier breaks the
phrase.

A barrier is anything that semantically requires "stop, do something
that depends on the robot being at rest, then resume": gripper
open/close, fresh perception capture, IK re-detect, force/contact
checks, BT subtree boundaries.

Public API:
    Waypoint(position, orientation, hold_s=0.0, dynamics_factor=0.5,
             min_time_s=0.0, label="")
    WaypointBuffer():
        .add(wp) -> None
        .flush(franky_robot) -> bool       # builds + executes one motion
        .barrier(franky_robot) -> bool     # flush + clear (semantic sync)
        .clear() -> None
        .pending -> int

The franky half above is FRANKA-ONLY and inert on the UR10e (franky is
not installed and ``resolve_franky_robot`` can never match a
``UR10eDriver`` chain). The UR10e blending transport lives in the
``JointRow`` section at the bottom of this file: it emits ONE URScript
program of ``movej(..., r=)`` rows over the primary socket, which is the
only blend mechanism immune to the rtde_c control-script lifecycle.
"""

from __future__ import annotations

import logging
import math
from dataclasses import dataclass, field
from typing import List, Optional, Sequence

import numpy as np
from scipy.spatial.transform import Rotation

from spark_real.utils.rotations import quat_xyzw_to_rotvec

try:
    import franky
except ImportError:
    franky = None

logger = logging.getLogger(__name__)


@dataclass
class Waypoint:
    """
    One vertex of the spatio-temporal graph: a TCP pose + temporal hints.

    Attributes:
        position: (3,) base-frame target translation [m].
        orientation: (3,) axis-angle rotation vector [rad] OR a length-4
            xyzw quaternion. Stored as quaternion internally.
        hold_s: dwell duration at the target. 0 = blend straight through
            (the "legato" case; default). Positive = hold for that many
            seconds before the next waypoint.
        dynamics_factor: per-waypoint scaling against the robot's
            absolute velocity / acceleration / jerk limits. franky
            interprets this as ``relative_dynamics_factor``; 1.0 is full
            speed. Default 0.5 keeps motions visibly smooth.
        min_time_s: minimum time the segment leading INTO this waypoint
            should take. Useful to slow down the final approach without
            slowing the whole phrase.
        label: free-text annotation; shows up in logs and is helpful
            when debugging which note a phrase tripped on.
    """

    position: np.ndarray
    orientation: Sequence[float]
    hold_s: float = 0.0
    dynamics_factor: float = 0.5
    min_time_s: float = 0.0
    label: str = ""

    def __post_init__(self):
        self.position = np.asarray(self.position, dtype=float).reshape(3)
        ori = np.asarray(self.orientation, dtype=float)
        if ori.shape == (3,):
            quat_xyzw = Rotation.from_rotvec(ori).as_quat()
        elif ori.shape == (4,):
            quat_xyzw = ori / max(np.linalg.norm(ori), 1e-12)
        else:
            raise ValueError(f"orientation must be (3,) rotvec or (4,) quat, got {ori.shape}")
        self._quat_xyzw = quat_xyzw

    def _quat_xyzw_to_rotvec(self) -> np.ndarray:
        """Return the orientation as a 3-vec axis-angle (rotvec).

        Used by the joint-space IK path that wants rotvec, while the
        cartesian motion path uses the quaternion directly.
        """
        return quat_xyzw_to_rotvec(self._quat_xyzw)


def _build_franky_waypoint(wp: Waypoint):
    """
    Convert a Waypoint into a franky.CartesianWaypoint.
    """
    if franky is None:
        raise RuntimeError("franky SDK not installed; cannot build waypoint")
    target = franky.RobotPose(franky.Affine(wp.position.tolist(), wp._quat_xyzw.tolist()))
    kwargs = {
        "target": target,
        "reference_type": franky.ReferenceType.Absolute,
        "relative_dynamics_factor": float(np.clip(wp.dynamics_factor, 0.05, 1.0)),
    }
    # franky.Duration takes microseconds; this API exposes seconds
    if wp.hold_s > 0:
        kwargs["hold_target_duration"] = franky.Duration(int(wp.hold_s * 1000))
    if wp.min_time_s > 0:
        kwargs["minimum_time"] = franky.Duration(int(wp.min_time_s * 1000))
    return franky.CartesianWaypoint(**kwargs)


class WaypointBuffer:
    """
    Owns the pending waypoints in the current motion phrase.

    Two flush modes:
      .flush_cartesian(franky_robot): CartesianWaypointMotion (simple,
        but franky's Cartesian planner will reject if the start config
        is near singular or the trajectory implies joint-velocity
        discontinuity).
      .flush_joint(franky_robot, seed_q, ik_solver): JointWaypointMotion
        with per-waypoint IK. Bypasses the Cartesian-planner check and
        plans a single Ruckig joint-space trajectory through all the
        waypoints. This is the "legato through singularity" path used
        by FR3 with top-down grasps.

    .flush() picks joint mode when ik_solver is provided, else Cartesian.
    """

    def __init__(self):
        self._wps: List[Waypoint] = []

    def add(self, wp: Waypoint) -> None:
        self._wps.append(wp)
        logger.debug(
            "[waypoints] add %s pos=%s hold=%.2fs dyn=%.2f (n=%d)",
            wp.label,
            wp.position.tolist(),
            wp.hold_s,
            wp.dynamics_factor,
            len(self._wps),
        )

    def clear(self) -> None:
        self._wps.clear()

    @property
    def pending(self) -> int:
        return len(self._wps)

    def flush(
        self,
        franky_robot,
        seed_q: Optional[np.ndarray] = None,
        ik_solver=None,
    ) -> bool:
        if seed_q is not None and ik_solver is not None:
            return self.flush_joint(franky_robot, seed_q, ik_solver)
        return self.flush_cartesian(franky_robot)

    @staticmethod
    def resolve_franky_robot(maybe_wrapper):
        """Walk a driver-wrapper chain (SafeRobot -> FrankaDriver ->
        franky.Robot) to find the actual franky.Robot instance, or None.

        Lets the executor and skills get the handle that
        CartesianWaypointMotion.move() needs without knowing the wrap depth.
        """
        if franky is None:
            return None
        obj = maybe_wrapper
        for _ in range(5):
            if obj is None:
                return None
            if isinstance(obj, franky.Robot):
                return obj
            obj = getattr(obj, "_robot", None)
        return None

    def flush_cartesian(self, franky_robot) -> bool:
        """
        Execute the accumulated waypoints as a single blended motion.

        Returns True if the motion completed without exception, False
        otherwise. Buffer is cleared either way.
        """
        if not self._wps:
            return True
        if franky is None:
            logger.warning("[waypoints] franky unavailable; cannot flush")
            self.clear()
            return False

        franky_wps = [_build_franky_waypoint(wp) for wp in self._wps]
        labels = [wp.label or f"wp{i}" for i, wp in enumerate(self._wps)]
        logger.info(
            "[waypoints] flushing phrase: %s",
            " -> ".join(labels),
        )

        motion = franky.CartesianWaypointMotion(
            franky_wps,
            return_when_finished=True,
        )
        success = True
        try:
            franky_robot.move(motion, asynchronous=False)
        except Exception as exc:
            msg = str(exc).lower()
            recoverable = any(
                k in msg
                for k in (
                    "reflex",
                    "aborted",
                    "discontinuity",
                    "singular",
                    "rejected",
                )
            )
            if recoverable:
                logger.warning(
                    "[waypoints] phrase rejected/aborted (%s); buffer cleared, "
                    "caller decides whether to fall back",
                    exc,
                )
            else:
                logger.error("[waypoints] phrase failed: %s", exc)
            success = False
        finally:
            self.clear()
        return success

    def flush_joint(self, franky_robot, seed_q: np.ndarray, ik_solver) -> bool:
        """
        Execute the phrase as ONE JointWaypointMotion.

        Pre-computes IK at each waypoint (each one seeded by the previous
        solution) so the Cartesian-planner discontinuity check doesn't
        fire, franky just gets a list of joint targets and runs Ruckig
        in joint space.

        Args:
            franky_robot: the franky.Robot instance
            seed_q: 7-vector, current arm joints (the trajectory starts here)
            ik_solver: callable(position, orientation, q_seed) -> q_target or None
        """
        if not self._wps:
            return True
        if franky is None:
            logger.warning("[waypoints] franky unavailable; cannot flush_joint")
            self.clear()
            return False

        q_seed = np.asarray(seed_q, dtype=float).reshape(-1)[:7].copy()
        franky_wps = []
        labels = []
        for i, wp in enumerate(self._wps):
            q_target = ik_solver(wp.position, wp._quat_xyzw_to_rotvec(), q_seed)
            if q_target is None:
                logger.warning(
                    "[waypoints] IK failed at %s; aborting phrase",
                    wp.label or f"wp{i}",
                )
                self.clear()
                return False
            js = franky.JointState(np.asarray(q_target, dtype=float))
            kwargs = {
                "target": js,
                "reference_type": franky.ReferenceType.Absolute,
                "relative_dynamics_factor": float(np.clip(wp.dynamics_factor, 0.05, 1.0)),
            }
            if wp.hold_s > 0:
                kwargs["hold_target_duration"] = franky.Duration(int(wp.hold_s * 1000))
            if wp.min_time_s > 0:
                kwargs["minimum_time"] = franky.Duration(int(wp.min_time_s * 1000))
            franky_wps.append(franky.JointWaypoint(**kwargs))
            labels.append(wp.label or f"wp{i}")
            q_seed = q_target  # seed next IK from this solution

        logger.info(
            "[waypoints] flushing JOINT phrase: %s",
            " -> ".join(labels),
        )
        motion = franky.JointWaypointMotion(
            franky_wps,
            return_when_finished=True,
        )
        success = True
        try:
            franky_robot.move(motion, asynchronous=False)
        except Exception as exc:
            msg = str(exc).lower()
            recoverable = any(
                k in msg
                for k in (
                    "reflex",
                    "aborted",
                    "discontinuity",
                    "singular",
                    "rejected",
                )
            )
            if recoverable:
                logger.warning(
                    "[waypoints] joint phrase aborted (%s); caller decides fallback",
                    exc,
                )
            else:
                logger.error("[waypoints] joint phrase failed: %s", exc)
            success = False
        finally:
            self.clear()
        return success

    def barrier(self, franky_robot, seed_q=None, ik_solver=None) -> bool:
        """
        Semantic sync point: flush any pending phrase and break legato.
        """
        return self.flush(franky_robot, seed_q=seed_q, ik_solver=ik_solver)


# =====================================================================
# UR10e blended joint path (one URScript program of movej(..., r=) rows)
# =====================================================================
#
# Why a second transport instead of rtde_c.moveJ(path):
#   * every move on this rig already goes out over the raw URScript socket
#     (ur10e_driver._send_script, port 30002) because rtde_c.moveJ silently
#     no-ops once a speedl / gripper script has stopped its control script,
#     and _send_script itself calls rtde_c.stopScript() before every move.
#   * consecutive _send_script("movej(...)") calls do NOT blend: each send
#     REPLACES the running program, so send #2 cancels move #1 mid-flight.
#     Blending therefore requires all rows inside ONE program in ONE send.
#     That shape is already proven here by the gripper publisher, which
#     uploads a multi-line def/end block plus its call in a single send.
#
# Emitted shape (last row always r=0, so the terminal waypoint is a true
# stop and the executor's exact joint-arrival test still applies):
#
#   def spark_blend_path():
#     write_output_integer_register(15, 1000)
#     movej([...],a=1.4,v=1.05,r=0.05)
#     write_output_integer_register(15, 1001)
#     movej([...],a=1.4,v=1.05,r=0)
#     write_output_integer_register(15, 1002)
#   end
#   spark_blend_path()
#
# The register writes are a mid-path progress signal. They are inside the
# same program, so they cannot cancel anything (unlike a fresh send), and
# they need no control script. Without them a stall inside a multi-second
# path would only surface as a silent full-path timeout (the measured 15.0 s /
# 1817-speedl grind into a mechanical stop). "Register not advancing AND TCP
# still" is the stall signature.

# 12/13/14 belong to the gripper publisher (_POS/_OBJ/_SEQ_REGISTER).
UR_PROGRESS_REGISTER = 15
# value = epoch * STRIDE + row_index, so a register left over from the
# PREVIOUS path can never be mistaken for progress on this one.
UR_PROGRESS_STRIDE = 1000
UR_MAX_EPOCH = 200
UR_BLEND_FUNC = "spark_blend_path"

# A junction blend radius may not exceed this fraction of either adjoining
# segment. r_i <= 0.45*min(L_i, L_i+1) also guarantees non-overlap of
# consecutive blend regions (r_i + r_i+1 <= 0.9 * L_i+1). PolyScope does no
# validation (rtde_control.Path.toScriptCode() accepted 0.05 blends on
# ~0.1 rad segments), so the policing is here.
UR_BLEND_SEG_FRAC = 0.45
# Below this a blend is not worth the corner-cut; emit a full stop instead.
UR_BLEND_MIN_M = 0.01
# Two rows whose joint targets agree this closely are the same pose: the row
# is a no-op and is dropped rather than blended (this is _transport_to's
# first lift, which is ~0 m long right after a move_relative(dz=+0.20)).
UR_ROW_MERGE_RAD = 0.01
# Reject a phrase whose IK jumps this far on one row: an elbow/wrist branch
# flip mid-path becomes one continuous blended wrench instead of a discrete,
# visible reconfiguration between two stopped moves.
UR_MAX_ROW_DQ_RAD = 2.6


@dataclass
class JointRow:
    """One ``movej`` row of a blended URScript path.

    Attributes:
        q: (6,) joint target [rad].
        position: (3,) TCP position this row commands, base frame [m]. Only
            used to size blend radii (``r`` is metres of TCP path even for
            movej) and to budget the wait; never sent.
        velocity: joint velocity [rad/s] -> ``v=``.
        acceleration: joint acceleration [rad/s^2] -> ``a=``.
        blend: blend radius [m] -> ``r=``. 0 means decelerate to a stop.
        label: free text for logs.
    """

    q: np.ndarray
    position: np.ndarray
    velocity: float
    acceleration: float
    blend: float = 0.0
    label: str = ""

    def __post_init__(self):
        self.q = np.asarray(self.q, dtype=float).reshape(-1)
        self.position = np.asarray(self.position, dtype=float).reshape(3)


def _fmt_num(x: float) -> str:
    """Format like ur_rtde's own PathEntry emitter (C++ ostream default: %g)."""
    return f"{float(x):g}"


def format_movej_row(q: Sequence[float], velocity: float, acceleration: float, blend: float) -> str:
    """One ``movej([...],a=,v=,r=)`` line.

    Byte-identical to ``rtde_control.PathEntry(MoveJ, PositionJoints,
    [*q, v, a, r]).toScriptCode()`` minus its leading tab (asserted by
    tests/test_blend_path.py), so the wire contract is ur_rtde's.
    """
    qs = ",".join(_fmt_num(v) for v in list(q)[:6])
    return (
        f"movej([{qs}],a={_fmt_num(acceleration)}," f"v={_fmt_num(velocity)},r={_fmt_num(blend)})"
    )


def resolve_blend_radii(
    start_position: Sequence[float],
    rows: List[JointRow],
    radius_m: float,
    final_radius_m: float = 0.0,
    min_radius_m: float = UR_BLEND_MIN_M,
    merge_rad: float = UR_ROW_MERGE_RAD,
    start_q: Optional[Sequence[float]] = None,
) -> tuple:
    """Drop no-op rows and size every junction blend. Mutates ``rows``' blends.

    ``radius_m`` is the requested junction radius; each junction is clamped to
    ``UR_BLEND_SEG_FRAC`` of BOTH adjoining segments, and anything below
    ``min_radius_m`` becomes a full stop. The LAST row is always r=0;
    ur_rtde's own emitter does the same, and the executor's arrival test
    depends on that terminal waypoint being a true stop.

    ``final_radius_m`` caps the junction leading INTO the last row separately,
    so a phrase that ends in a grasp descent can keep that descent essentially
    vertical (a 5 cm corner-cut into a 15 cm descent is not wanted) while the
    transit junctions before it stay generous.

    Returns (kept_rows, dropped_labels).
    """
    kept: List[JointRow] = []
    dropped: List[str] = []
    prev_q = None if start_q is None else np.asarray(start_q, dtype=float).reshape(-1)
    for i, row in enumerate(rows):
        last = i == len(rows) - 1
        if (
            prev_q is not None
            and not last
            and prev_q.size >= row.q.size
            and float(np.max(np.abs(row.q - prev_q[: row.q.size]))) < merge_rad
        ):
            # Same pose as the current or previous target: a no-op row.
            dropped.append(row.label or f"row{i}")
            continue
        kept.append(row)
        prev_q = row.q
    if not kept:
        return kept, dropped

    pts = [np.asarray(start_position, dtype=float).reshape(3)] + [r.position for r in kept]
    seg = [float(np.linalg.norm(pts[i + 1] - pts[i])) for i in range(len(kept))]
    for i, row in enumerate(kept):
        if i == len(kept) - 1:
            row.blend = 0.0  # terminal waypoint is a true stop
            continue
        want = radius_m
        if i == len(kept) - 2:
            want = min(want, final_radius_m) if final_radius_m > 0 else want
        r = min(want, UR_BLEND_SEG_FRAC * seg[i], UR_BLEND_SEG_FRAC * seg[i + 1])
        row.blend = float(r) if r >= min_radius_m else 0.0
    return kept, dropped


def build_ur_blend_program(
    rows: List[JointRow],
    epoch: int,
    progress_register: int = UR_PROGRESS_REGISTER,
    func_name: str = UR_BLEND_FUNC,
) -> str:
    """Wrap ``rows`` into one uploadable URScript program.

    ``epoch`` namespaces the progress register so a value left by the previous
    path is not read as progress on this one. Row k is *entered* when the
    register reads ``epoch*STRIDE + k``; ``+ len(rows)`` means the program ran
    to the end.
    """
    if not rows:
        raise ValueError("build_ur_blend_program: no rows")
    base = int(epoch) * UR_PROGRESS_STRIDE
    lines = [f"def {func_name}():"]
    for k, row in enumerate(rows):
        if progress_register is not None:
            lines.append(f"  write_output_integer_register({progress_register}, {base + k})")
        lines.append("  " + format_movej_row(row.q, row.velocity, row.acceleration, row.blend))
    if progress_register is not None:
        lines.append(f"  write_output_integer_register({progress_register}, {base + len(rows)})")
    lines.append("end")
    lines.append(f"{func_name}()")
    return "\n".join(lines) + "\n"


def blend_progress_value(epoch: int, row_index: int) -> int:
    """The register value written on entering ``row_index`` of ``epoch``."""
    return int(epoch) * UR_PROGRESS_STRIDE + int(row_index)


def decode_blend_progress(value, epoch: int, n_rows: int):
    """Row index from a register read, or None if it is not this path's epoch."""
    try:
        v = int(value)
    except (TypeError, ValueError):
        return None
    base = int(epoch) * UR_PROGRESS_STRIDE
    if base <= v <= base + n_rows:
        return v - base
    return None


def joint_move_seconds(dq_max: float, velocity: float, acceleration: float) -> float:
    """Trapezoidal duration of one movej, from the leading joint's travel.

    Used only to BUDGET the wait (3x this plus the start lag), never to
    decide arrival.
    """
    d = abs(float(dq_max))
    v = max(float(velocity), 1e-6)
    a = max(float(acceleration), 1e-6)
    if d <= v * v / a:
        return 2.0 * math.sqrt(d / a)
    return d / v + v / a

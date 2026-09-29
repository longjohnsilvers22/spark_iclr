"""
IkMixin: IK-backed joint-space motion primitives for ScoreExecutor.

Splits the inverse-kinematics motion path out of MotionMixin so the
solver-bound helpers live next to their lazy IK imports. Provides
_movej_via_ik, _movej_to_pose, _wait_stationary_after_servo, and
_execute_legato_phrase. These are mixed into ScoreExecutor alongside
MotionMixin and rely on the same instance attributes (self.robot,
self._servo, self.detection_map, ...) that the rest of the executor
sets up.
"""

import logging
import os
import time

import numpy as np
from scipy.spatial.transform import Rotation

from spark_real.control.executor_types import (
    ExecutionResult,
    maybe_osc_move_linear,
)
from spark_real.control.waypoints import (
    UR_MAX_EPOCH,
    UR_MAX_ROW_DQ_RAD,
    JointRow,
    build_ur_blend_program,
    decode_blend_progress,
    resolve_blend_radii,
)

try:
    import franky
except ImportError:
    franky = None

try:
    from spark_real.control.fr3_ik import solve_ik as solve_ik_dls
except ImportError:
    solve_ik_dls = None

try:
    from spark_real.control.fr3_ik_pyroki import solve_ik as solve_ik_pyroki
except ImportError:
    solve_ik_pyroki = None

try:
    from spark_real.control.ur10e_ik_pyroki import solve_ik as solve_ik_ur10e
    from spark_real.control.ur10e_ik_pyroki import solve_ik_rtde as solve_ik_ur10e_rtde
except ImportError:
    solve_ik_ur10e = None
    solve_ik_ur10e_rtde = None

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Joint / Cartesian speed envelope: ONE source for every movej and movel this
# executor emits. The URScript defaults movej(q, a=1.4, v=1.05) and
# movel(pose, a=1.2, v=0.25) are 50% of the base/shoulder rating and 33% of
# the wrist rating, not UR10e capabilities.
#
# Rated maxima, UR10e tech sheet (base + shoulder 120 deg/s, elbow + wrist
# 1/2/3 180 deg/s):
#   https://www.universal-robots.com/manuals/EN/HTML/SW5_25/Content/
#   prod-usr-man/complianceUR10e/H_g5_sections/appendix_g5/tech_spec_sheet.htm
UR10E_JOINT_VEL_RATED_RAD_S = (2.094, 2.094, 3.142, 3.142, 3.142, 3.142)

# SAFETY MARGIN, stated explicitly: command at most 85% of the rated joint
# speed (15% headroom). The headroom is not decoration: the installed pendant
# safety configuration has its own Joint Speed Limit, and exceeding THAT is a
# protective stop (C150/C152), not a faster move. 85% also leaves room for the
# pendant Stopping Time / Stopping Distance limits, which throttle commanded
# speed so the arm can always stop inside them.
JOINT_VEL_MARGIN = 0.85

# UR publishes NO per-joint acceleration rating: movej "scales torques to be
# within hardware limits" itself, so this is a REQUEST ceiling, not a hardware
# number. 3.5 rad/s^2 is 2.5x the URScript default of 1.4. Acceleration is the
# dominant lever here (a trapezoid is triangular for any move shorter than
# v^2/a, and most primitives on this task are short), but very high acceleration combined with
# blends is UR's named trigger for C173 joint-torque-overload, so this is
# deliberately not pushed further without rig evidence.
JOINT_ACC_CAP_RAD_S2 = 3.5
# Never crawl: a tiny commanded velocity makes large reconfigurations take
# forever.
JOINT_VEL_FLOOR_RAD_S = 0.3
# The Cartesian `velocity` knob (PipelineConfig.velocity, default 0.25 m/s) is
# what the whole executor is parameterised by. This is the value of that knob
# which maps to the FULL leading-axis joint cap; below it the joint cap scales
# down proportionally.
JOINT_VEL_REF_LINEAR_M_S = 0.25

# movel is Cartesian. UR10e tool speed: 4 m/s max, "approx 1 m/s" typical in
# the tech-sheet table. Cap at half the typical figure.
LINEAR_VEL_CAP_M_S = 0.5
LINEAR_ACC_CAP_M_S2 = 1.5

# Ramp-in knob: SPARK_JOINT_SPEED_FRACTION scales the whole envelope (v and a
# together). 1.0 = the envelope above; ~0.6 reproduces a 1.05 rad/s velocity.
_SPEED_FRACTION_ENV = "SPARK_JOINT_SPEED_FRACTION"
# SPARK_LEGACY_JOINT_LIMITS restores the legacy clamps below exactly
# (one-env-var rollback).
_LEGACY_LIMITS_ENV = "SPARK_LEGACY_JOINT_LIMITS"
_LEGACY = {
    "vel_scale": 6.0,
    "vel_cap": 1.05,
    "acc_cap": 1.4,
    "pose_vel_scale": 3.0,
    "pose_vel_cap": 1.0,
    "lin_vel_cap": 0.25,
    "lin_acc_cap": 1.2,
}


def legacy_joint_limits() -> bool:
    """True when SPARK_LEGACY_JOINT_LIMITS restores the legacy clamps."""
    return os.environ.get(_LEGACY_LIMITS_ENV, "0") not in ("0", "false", "False", "")


def joint_speed_fraction() -> float:
    """Fraction of the whole speed envelope to command (0 < f <= 1)."""
    raw = os.environ.get(_SPEED_FRACTION_ENV)
    if raw is None:
        return 1.0
    try:
        return float(np.clip(float(raw), 0.05, 1.0))
    except ValueError:
        logger.warning("%s=%r ignored", _SPEED_FRACTION_ENV, raw)
        return 1.0


def leading_axis_vel_cap(dq) -> float:
    """Highest movej ``v`` that keeps EVERY joint inside its own rating.

    URScript movej applies ``v``/``a`` to the LEADING axis (the largest
    |dq|); every other joint is time-scaled to finish together, so joint i
    actually runs at ``v * dq_i / L``. A single global cap therefore has to be
    the slowest joint's rating (2.094 rad/s), but on a wrist-dominated move
    the base is barely turning and there is 50% more headroom available. The
    exact condition is ``v * dq_i / L <= rated_i`` for all i, i.e.
    ``v <= min_i(rated_i * L / dq_i)``.

    Falls back to the most conservative rating when dq is unknown or zero.
    """
    dq = np.abs(np.asarray(dq, dtype=float).reshape(-1))
    rated = np.asarray(UR10E_JOINT_VEL_RATED_RAD_S[: dq.size], dtype=float)
    if dq.size == 0 or rated.size != dq.size:
        return JOINT_VEL_MARGIN * min(UR10E_JOINT_VEL_RATED_RAD_S)
    lead = float(dq.max())
    if lead <= 1e-9:
        return JOINT_VEL_MARGIN * min(UR10E_JOINT_VEL_RATED_RAD_S)
    moving = dq > 1e-9
    cap = float(np.min(rated[moving] * lead / dq[moving]))
    return JOINT_VEL_MARGIN * cap


def resolve_joint_limits(velocity, dq=None, pose_fallback=False):
    """Map the Cartesian ``velocity`` knob to one movej ``(v, a)`` in rad, s.

    ``dq`` is the per-joint travel of the move about to be issued; passing it
    unlocks the wrist headroom described in leading_axis_vel_cap. Omit it and
    the cap collapses to the slowest joint's rating, which is what a move with
    no solved joint target has to assume.

    ``pose_fallback`` marks the legacy Cartesian path (_movej_to_pose), whose
    legacy clamps (velocity*3 capped at 1.0) differ from the IK path's
    (velocity*6 capped at 1.05). It only changes anything under
    SPARK_LEGACY_JOINT_LIMITS.
    """
    if legacy_joint_limits():
        if pose_fallback:
            vel_j = float(
                min(float(velocity) * _LEGACY["pose_vel_scale"], _LEGACY["pose_vel_cap"])
            )
        else:
            vel_j = float(
                np.clip(
                    float(velocity) * _LEGACY["vel_scale"],
                    JOINT_VEL_FLOOR_RAD_S,
                    _LEGACY["vel_cap"],
                )
            )
        return vel_j, float(min(vel_j * 2.0, _LEGACY["acc_cap"]))
    frac = joint_speed_fraction()
    cap = leading_axis_vel_cap(dq if dq is not None else []) * frac
    ref = max(JOINT_VEL_REF_LINEAR_M_S, 1e-6)
    vel_j = float(np.clip(float(velocity) / ref * cap, JOINT_VEL_FLOOR_RAD_S, cap))
    acc_j = float(min(vel_j * 2.0, JOINT_ACC_CAP_RAD_S2 * frac))
    return vel_j, acc_j


def resolve_linear_limits(velocity):
    """movel ``(v, a)`` in m/s, m/s^2 for the legacy Cartesian fallback."""
    if legacy_joint_limits():
        vel_l = float(min(float(velocity), _LEGACY["lin_vel_cap"]))
        return vel_l, float(min(max(vel_l, 0.05) * 4.0, _LEGACY["lin_acc_cap"]))
    frac = joint_speed_fraction()
    vel_l = float(min(float(velocity), LINEAR_VEL_CAP_M_S * frac))
    acc_l = float(min(max(vel_l, 0.05) * 4.0, LINEAR_ACC_CAP_M_S2 * frac))
    return vel_l, acc_l


class IkMixin:
    """
    IK-backed joint-space motion primitives mixed into ScoreExecutor.
    """

    def _robot_family(self):
        """Resolve the robot family ('franka'/'ur10e'/...) off the driver."""
        fam = getattr(self.robot, "robot_family", None) or getattr(
            getattr(self.robot, "_robot", None), "robot_family", None
        )
        return str(fam or "").lower()

    def _movej_via_ik(self, position, orientation, velocity):
        """
        Solve IK in-process, then issue a joint-space move (no firmware IK).

        Dispatch by robot_family:
          - franka -> fr3_ik_pyroki, execute via move_to_joint_config (franky)
          - ur10e  -> ur10e_ik_pyroki, execute via movej over _send_script
                      (move_to_joint_config_urscript). Gated by
                      SPARK_UR_PYROKI_IK (default "1"); =0 raises so the caller
                      falls back to _movej_to_pose (the legacy Cartesian path).
        """
        self._check_abort()
        family = self._robot_family()
        pos = np.array(position, dtype=float)
        orient = list(orientation)

        ur_tcp_off = None
        if family == "ur10e":
            if os.environ.get("SPARK_UR_PYROKI_IK", "1") != "1":
                raise RuntimeError("SPARK_UR_PYROKI_IK disabled; using legacy path")
            if solve_ik_ur10e_rtde is None:
                raise RuntimeError("UR10e pyroki IK not available")
            # The executor targets are RTDE-frame gripper-TIP poses; pyroki
            # solves URDF-frame tool0. solve_ik_rtde bridges the Rz(180) base
            # rotation + the pendant TCP offset. Without the offset that bridge
            # is unsafe, so fall back to the (frame-safe) legacy path.
            for src in (self.robot, getattr(self.robot, "_robot", None)):
                if src is not None and hasattr(src, "get_tcp_offset"):
                    try:
                        ur_tcp_off = np.asarray(src.get_tcp_offset(), dtype=float)
                        break
                    except Exception as exc:
                        logger.warning("[movej_via_ik:ur10e] get_tcp_offset failed: %s", exc)
            if ur_tcp_off is None:
                raise RuntimeError("UR10e TCP offset unavailable; using legacy path")
            solve_ik = solve_ik_ur10e
            default_seed = np.array(
                getattr(self.robot, "HOME_CONFIG",
                        [1.5708, -1.5708, 1.5708, -1.5708, -1.5708, 0.0]),
                dtype=float,
            )
        else:
            if solve_ik_pyroki is None:
                raise RuntimeError("pyroki IK not available")
            if not hasattr(self.robot, "move_to_joint_config"):
                raise RuntimeError("Robot has no move_to_joint_config")
            solve_ik = solve_ik_pyroki
            # IK seed only (not the ready pose): rounded J5=0 variant, intentionally
            # differs from franka_base.HOME_CONFIG / franka_default.yaml home_config.
            default_seed = np.array([0, -0.785, 0, -2.356, 0, 1.571, 0.785])

        try:
            q_current = np.array(self.robot.get_joint_positions(), dtype=float)
        except Exception:
            q_current = default_seed

        if family == "ur10e":
            q_target = solve_ik_ur10e_rtde(
                pos, orient, q_seed=q_current, tcp_offset=ur_tcp_off
            )
        else:
            q_target = solve_ik(pos, orient, q_seed=q_current)
        if q_target is None:
            raise RuntimeError(
                f"pyroki IK failed for pos={pos.round(3)} orient={np.array(orient).round(3)}"
            )

        dq = np.abs(q_target - q_current[: q_target.size])
        logger.info(
            "[movej_via_ik:%s] pos=(%.3f,%.3f,%.3f) dq_max=%.2frad j%d",
            family or "?",
            pos[0],
            pos[1],
            pos[2],
            dq.max(),
            dq.argmax(),
        )

        # velocity is Cartesian m/s but joint moves want joint velocity (rad/s).
        # resolve_joint_limits owns that mapping AND the envelope; dq is passed
        # so a wrist-led move gets the wrist rating rather than the base's.
        vel_j, acc_j = resolve_joint_limits(velocity, dq=dq)
        logger.info(
            "[movej_via_ik:%s] v=%.2f rad/s a=%.2f rad/s^2 (lead j%d, cap %.2f)",
            family or "?",
            vel_j,
            acc_j,
            int(dq.argmax()),
            leading_axis_vel_cap(dq),
        )

        if family == "ur10e":
            # movej over the URScript primary channel (_send_script): rtde_c.moveJ
            # silently no-ops after any speedl/gripper _send_script, so route the
            # joint target through the conflict-free URScript socket.
            # Fire-and-forget, then block on _wait_for_motion.
            self.robot.move_to_joint_config_urscript(
                q_target.tolist(), velocity=vel_j, acceleration=acc_j
            )
            # Hand the wait the joint target it just commanded: that is the
            # exact arrival test. Without it the wait falls back to Cartesian
            # proximity plus a velocity heuristic and burns the full timeout on
            # any move short enough to finish inside the URScript start lag.
            self._wait_for_motion(pos, q_target=q_target)
        else:
            self.robot.move_to_joint_config(q_target.tolist(), velocity=vel_j)

    # Blended multi-waypoint motion (UR10e only)

    def _ur_tcp_offset(self):
        """Pendant TCP offset (gripper tip relative to tool0), or None.

        Cached at connect by the driver; solve_ik_rtde cannot bridge the
        RTDE-frame tip target to a URDF tool0 target without it.
        """
        for src in (self.robot, getattr(self.robot, "_robot", None)):
            if src is not None and hasattr(src, "get_tcp_offset"):
                try:
                    return np.asarray(src.get_tcp_offset(), dtype=float)
                except Exception as exc:  # noqa: BLE001
                    logger.warning("[blend] get_tcp_offset failed: %s", exc)
        return None

    def _next_blend_epoch(self, n_rows: int) -> int:
        """Pick a progress-register epoch the CURRENT register value is not in.

        The register survives across paths and across process restarts, so a
        stale value inside this path's window would read as progress, and a
        stale terminal value would read as "path complete" before the program
        had written a byte. Reading first and stepping off the held value
        closes that: never infer from a signal this path has not been observed
        to produce.
        """
        held = self._read_blend_progress()
        ep = getattr(self, "_blend_epoch", 0) % UR_MAX_EPOCH + 1
        for _ in range(UR_MAX_EPOCH):
            if decode_blend_progress(held, ep, n_rows) is None:
                break
            ep = ep % UR_MAX_EPOCH + 1
        self._blend_epoch = ep
        return ep

    def _solve_blend_rows(self, waypoints, q_start):
        """IK every waypoint with seed chaining; None rejects the whole phrase.

        Pre-validating the ENTIRE path before the first byte is sent is the
        point: a mid-path IK failure cannot be recovered once the program is
        running, and an IK branch flip on one row becomes a single continuous
        blended wrench instead of a discrete reconfiguration between two
        stopped moves.
        """
        if solve_ik_ur10e_rtde is None:
            logger.info("[blend] UR10e pyroki IK unavailable; no blended path")
            return None
        if os.environ.get("SPARK_UR_PYROKI_IK", "1") != "1":
            return None
        tcp_off = self._ur_tcp_offset()
        if tcp_off is None:
            logger.info("[blend] TCP offset unavailable; no blended path")
            return None

        q_seed = np.asarray(q_start, dtype=float).reshape(-1)
        rows = []
        for i, wp in enumerate(waypoints):
            pos = np.asarray(wp[0], dtype=float).reshape(3)
            orient = list(wp[1])
            vel = float(wp[2]) if len(wp) > 2 and wp[2] else self.velocity
            label = str(wp[3]) if len(wp) > 3 else f"row{i}"
            q = solve_ik_ur10e_rtde(pos, orient, q_seed=q_seed, tcp_offset=tcp_off)
            if q is None:
                logger.info(
                    "[blend] IK failed at %s pos=%s; falling back to per-move",
                    label,
                    pos.round(3).tolist(),
                )
                return None
            dq = float(np.max(np.abs(np.asarray(q, dtype=float) - q_seed[: q.size])))
            if dq > UR_MAX_ROW_DQ_RAD:
                logger.info(
                    "[blend] %s needs dq_max=%.2f rad (> %.2f): IK branch flip, "
                    "refusing to blend through it",
                    label,
                    dq,
                    UR_MAX_ROW_DQ_RAD,
                )
                return None
            vel_j, acc_j = resolve_joint_limits(
                vel, dq=np.asarray(q, dtype=float) - q_seed[: q.size]
            )
            rows.append(
                JointRow(
                    q=q,
                    position=pos,
                    velocity=vel_j,
                    acceleration=acc_j,
                    label=label,
                )
            )
            q_seed = np.asarray(q, dtype=float)
        return rows

    def _clip_blend_waypoints(self, waypoints, start_pos, phrase):
        """Apply _move_to's workspace clip + oversize refusal to every row.

        Per-move dispatch gets both of these inside _move_to; a phrase must not
        bypass them just because it takes one send instead of N. Returns the
        clipped waypoint list, or None to refuse the phrase.
        """
        out = []
        prev = np.asarray(start_pos, dtype=float)
        for wp in waypoints:
            raw = np.asarray(wp[0], dtype=float).reshape(3)
            pos = raw
            if not self._check_workspace(pos):
                pos = np.clip(raw, self.WORKSPACE_MIN, self.WORKSPACE_MAX)
                logger.info(
                    "[blend] %s row CLIPPED by workspace: (%.3f,%.3f,%.3f) -> "
                    "(%.3f,%.3f,%.3f)",
                    phrase,
                    raw[0],
                    raw[1],
                    raw[2],
                    pos[0],
                    pos[1],
                    pos[2],
                )
            seg = float(np.linalg.norm(pos - prev))
            if seg > 1.0:
                logger.warning(
                    "[blend] %s segment %.3fm too large; per-move dispatch",
                    phrase,
                    seg,
                )
                return None
            out.append((pos,) + tuple(wp[1:]))
            prev = pos
        return out

    def _blend_rows_pass_safety_filter(self, rows, phrase) -> bool:
        """Run the driver wrapper's movej joint filter over EVERY row.

        Deliberately redundant with SafeRobot's own per-row movej scan, for
        two reasons:

          - It degrades to per-move dispatch (return False) instead of
            refusing the motion outright, so a single near-singular waypoint
            costs the blend, not the task.
          - It is the only joint check in the chain when self.robot is a bare
            driver with no SafeRobot wrapper.

        Blending must not buy speed with safety, and a second independent
        evaluation of the same barrier is cheap.

        Preference order: a public per-target predicate on the wrapper if one
        exists, else the wrapper's OWN epsilon applied with the same predicate
        shape SafeRobot uses (so there is still one source for the number). A
        chain with no filter in it (a bare driver) has nothing to enforce.
        """
        node = self.robot
        for _ in range(4):
            if node is None:
                return True
            check = getattr(node, "check_joint_target_safe", None)
            if callable(check):
                for row in rows:
                    if not check(row.q.tolist()):
                        logger.warning(
                            "[blend] %s row %s refused by the safety filter; "
                            "per-move dispatch",
                            phrase,
                            row.label,
                        )
                        return False
                return True
            eps = getattr(getattr(node, "_cfg", None), "eps_singularity", None)
            if eps is not None:
                for row in rows:
                    q2 = float(row.q[2])
                    if (q2**2) * ((q2 - np.pi) ** 2) - float(eps) ** 2 < 0:
                        logger.warning(
                            "[blend] %s row %s near the elbow singularity "
                            "(q2=%.3f); per-move dispatch",
                            phrase,
                            row.label,
                            q2,
                        )
                        return False
                return True
            node = getattr(node, "_robot", None)
        return True

    def _blend_transit(self, waypoints, final_radius_m=None, phrase="phrase") -> bool:
        """Fly ``waypoints`` as ONE blended URScript program. True = arrived.

        ``waypoints`` is a list of ``(position, orientation[, velocity[,
        label]])``. Every row but the last carries a blend radius, so the arm
        cuts the corner at each intermediate waypoint instead of decelerating
        to a stop; the last row is always ``r=0`` and is therefore a true stop
        the exact joint-arrival test still applies to.

        Returns False (having commanded nothing, or having stopped the arm)
        whenever the phrase cannot be trusted (blending disabled, wrong family,
        IK rejection, nothing left to blend after dropping no-op rows, a failed
        send, a stall, or a timeout). Callers must treat False as "do it the
        old way from the current pose".
        """
        if not self._blend_enabled():
            return False
        if self._robot_family() != "ur10e":
            return False
        if not hasattr(self.robot, "_send_script"):
            return False
        self._check_abort()

        cfg = self._blend_cfg()
        max_rows = int(cfg["max_rows"])
        if len(waypoints) < 2:
            return False
        if len(waypoints) > max_rows:
            logger.info(
                "[blend] %s has %d rows (> blend_max_rows=%d); per-move dispatch",
                phrase,
                len(waypoints),
                max_rows,
            )
            return False

        try:
            q_start = np.asarray(self.robot.get_joint_positions(), dtype=float)
        except Exception as exc:  # noqa: BLE001
            logger.info("[blend] cannot read joints (%s); per-move dispatch", exc)
            return False
        start_pos = self._get_current_position()

        waypoints = self._clip_blend_waypoints(waypoints, start_pos, phrase)
        if waypoints is None:
            return False

        rows = self._solve_blend_rows(waypoints, q_start)
        if not rows:
            return False
        if not self._blend_rows_pass_safety_filter(rows, phrase):
            return False

        kept, dropped = resolve_blend_radii(
            start_pos,
            rows,
            radius_m=float(cfg["radius_m"]),
            final_radius_m=(
                float(cfg["descent_radius_m"])
                if final_radius_m is None
                else float(final_radius_m)
            ),
            min_radius_m=float(cfg["min_radius_m"]),
            start_q=q_start,
        )
        if dropped:
            logger.info("[blend] %s: dropped no-op rows %s", phrase, dropped)
        if len(kept) < 2:
            # Nothing left to blend: one real move is exactly what the
            # unblended path would issue, so let the caller issue it.
            return False

        epoch = self._next_blend_epoch(len(kept))
        program = build_ur_blend_program(kept, epoch, self._blend_register())
        logger.info(
            "[blend] %s: %d rows, r=[%s], epoch=%d",
            phrase,
            len(kept),
            ", ".join(f"{r.blend:.3f}" for r in kept),
            epoch,
        )
        if not self.robot._send_script(program):
            logger.warning("[blend] program send failed; per-move dispatch")
            self._retire_motion_lease()
            return False
        return self._wait_for_blended_path(
            kept, epoch, start_pos=start_pos, q_start=q_start
        )

    def _movej_to_pose(self, pose, velocity):
        """
        Joint-space move to a Cartesian pose (legacy fallback).

        UR fast path: emit movej(p[...]) URScript. Non-UR drivers fall back
        to move_linear or CartesianServo PD.
        """
        self._check_abort()
        # No solved joint target on this path (the controller runs its own IK),
        # so dq is unknown and the cap collapses to the slowest joint's rating.
        vel_j, acc_j = resolve_joint_limits(velocity, pose_fallback=True)
        if getattr(self.robot, "SUPPORTS_URSCRIPT", False) and hasattr(
            self.robot, "_send_script"
        ):
            ps = f"p[{pose[0]},{pose[1]},{pose[2]}," f"{pose[3]},{pose[4]},{pose[5]}]"
            if os.environ.get("SPARK_UR_MOVEL", "1") == "1":
                # Straight-line movel via the PRIMARY interface (_send_script),
                # NOT rtde_c.moveL: the gripper/speedl _send_script calls stop
                # the rtde_c control script, so rtde_c.moveL no-ops after a grasp
                # (arm never moves -> stack guard bails at full xy_err). movel
                # over _send_script works script-stopped AND keeps the TCP on a
                # straight line, killing the joint-space "wrenching" of movej.
                # movel v/a are Cartesian (m/s, m/s^2).
                vel_l, acc_l = resolve_linear_limits(velocity)
                sent = self.robot._send_script(f"movel({ps}, a={acc_l}, v={vel_l})")
            else:
                sent = self.robot._send_script(f"movej({ps}, a={acc_j}, v={vel_j})")
            if not sent:
                # Dropped command = failed primitive, never a silent pass
                # while _wait_for_motion times out on an arm that was never
                # told to move.
                raise RuntimeError(
                    "URScript move dropped (channel down); primitive failed"
                )
            self._wait_for_motion(np.array(pose[:3]))
            return

        if hasattr(self.robot, "move_linear"):
            try:
                if maybe_osc_move_linear(list(pose), velocity):
                    return
                self.robot.move_linear(list(pose), velocity=velocity)
                return
            except Exception as exc:
                logger.warning("move_linear failed (%s); trying CartesianServo PD", exc)
        if self._servo is not None and hasattr(self._servo, "move_to_pose"):
            try:
                self._servo.move_to_pose(list(pose), velocity=velocity)
                self._wait_stationary_after_servo()
                return
            except Exception as exc:
                logger.warning(
                    "CartesianServo PD fallback failed (%s); " "trying move_to_pose",
                    exc,
                )

        if hasattr(self.robot, "move_to_pose"):
            self.robot.move_to_pose(pose, velocity=velocity, wait=False)
            self._wait_for_motion(np.array(pose[:3]))
        elif hasattr(self.robot, "move_linear"):
            try:
                self.robot.move_linear(list(pose), velocity=velocity, asynchronous=True)
            except TypeError:
                self.robot.move_linear(list(pose), velocity=velocity)
            self._wait_for_motion(np.array(pose[:3]))

    def _wait_stationary_after_servo(
        self, max_wait: float = 2.0, vel_tol: float = 2e-3
    ):
        """
        Block until FR3 has fully decelerated after velocity-mode motion.
        """
        raw = getattr(self.robot, "_robot", self.robot)
        if hasattr(raw, "_robot"):
            raw = raw._robot
        try:
            t0 = time.time()
            prev_ok = False
            while time.time() - t0 < max_wait:
                try:
                    qdot = np.asarray(raw.state.dq, dtype=float)
                except Exception:
                    time.sleep(0.20)
                    return
                if np.max(np.abs(qdot)) < vel_tol:
                    if prev_ok:
                        return
                    prev_ok = True
                else:
                    prev_ok = False
                time.sleep(0.01)
        except Exception:
            time.sleep(0.20)

    def _execute_ur_blend_phrase(self, run) -> bool:
        """UR10e backend for the legato look-ahead: one blended path per phrase.

        Accepts only one shape: a run of
        ``move_relative`` notes, optionally closed by ONE terminal
        ``move_to_keypoint`` while holding. That terminal note is the transport,
        so it is handed to ``_transport_to`` as a lead-in rather than reduced to
        a bare "go to the detection", which is what the franky path above does
        and which would skip the clearance lift and the place descent entirely.

        Returns False without commanding anything on any shape it does not
        recognise, so the caller falls back to per-action dispatch.
        """
        if not self._blend_enabled():
            return False

        lead = []
        running = np.asarray(self._get_current_position(), dtype=float).copy()
        terminal = None
        for k, note in enumerate(run):
            atype = note.get("type", note.get("name", ""))
            params = note.get("params", {}) or {}
            if atype == "move_relative":
                running = running + np.array(
                    [
                        params.get("dx", 0.0),
                        params.get("dy", 0.0),
                        params.get("dz", 0.0),
                    ],
                    dtype=float,
                )
                lead.append(
                    (
                        running.copy(),
                        list(self.GRASP_ORIENTATION),
                        self.velocity,
                        "rel%d(dz=%.2f)" % (k, params.get("dz", 0.0)),
                    )
                )
            elif atype == "move_to_keypoint" and k == len(run) - 1 and self._holding:
                if not self._blend_cfg()["cross_node"]:
                    # OFF BY DEFAULT, and not for want of speed.
                    # executor_core._run_actions skips its whole failure/recovery
                    # block when a legato phrase returns True (`i = j; continue`),
                    # so a drop detected inside the phrase would be reported and
                    # then ignored: no _attempt_recovery, no re-grasp, and the
                    # following release would open on nothing. Folding the
                    # preceding move_relative into the transport is worth ~1 s,
                    # which is not worth silently losing drop recovery. Turning
                    # this on needs _run_actions to consume a phrase's per-note
                    # results; until then the
                    # transport still blends internally, inside _transport_to.
                    return False
                terminal = note
            else:
                return False

        t0 = time.time()
        if terminal is None:
            if not self._blend_transit(lead, final_radius_m=0.0, phrase="move_rel"):
                return False
        else:
            params = terminal.get("params", {}) or {}
            label = params.get("keypoint_label", "")
            det = self.detection_map.get(label)
            if det is None:
                return False
            pos = np.asarray(det["position_3d"], dtype=float).copy()
            if pos[2] < self.TABLE_Z_FLOOR:
                pos[2] = self.TABLE_Z_FLOOR
            target = pos + np.array(
                [
                    params.get("offset_x", 0.0),
                    params.get("offset_y", 0.0),
                    params.get("offset_z", 0.0),
                ],
                dtype=float,
            )
            self._last_keypoint_label = label
            ok = self._transport_to(
                target,
                target_detection=det,
                target_label=label,
                params=params,
                lead=lead,
            )
            if not ok or not self._holding:
                # Drop detected mid-transport. The notes ARE consumed (the arm
                # moved), so report them here exactly as _move_to_keypoint would
                # rather than replaying the phrase.
                self._append_phrase_results(run, time.time() - t0, failed_last=True)
                return True

        self._append_phrase_results(run, time.time() - t0)
        return True

    def _append_phrase_results(self, run, elapsed: float, failed_last: bool = False):
        """One ExecutionResult per note of a phrase that played as one motion."""
        per = elapsed / max(len(run), 1)
        for k, note in enumerate(run):
            atype = note.get("type", note.get("name", ""))
            last = k == len(run) - 1
            bad = failed_last and last
            self._results.append(
                ExecutionResult(
                    action_type=atype,
                    success=not bad,
                    message=(
                        "Object lost during transport"
                        if bad
                        else "Played as one blended phrase (%d notes)" % len(run)
                    ),
                    duration=per,
                )
            )

    def _execute_legato_phrase(self, run, run_start_idx: int, total: int):
        """
        Play a sequence of blendable actions as one joint-space phrase.

        UR10e goes to _execute_ur_blend_phrase (one URScript blended path).
        Everything else keeps the franky/Ruckig backend below: each note
        becomes a Waypoint, IK'd at each step, and dispatched as a single
        franky.JointWaypointMotion.
        """
        if self._robot_family() == "ur10e":
            return self._execute_ur_blend_phrase(run)
        if os.environ.get("SPARK_IK") == "pinocchio":
            _ik = solve_ik_dls
        else:
            _ik = solve_ik_pyroki
        if _ik is None:
            logger.info("[legato] IK modules unavailable; falling back")
            return False
        franky_robot = getattr(getattr(self.robot, "_robot", None), "_robot", None)
        if franky_robot is None or not hasattr(franky_robot, "move"):
            return False
        if not hasattr(self.robot, "get_joint_positions"):
            return False

        cur_pose = self._get_current_position()
        cur_R = None
        try:
            obs = self.robot.get_observation()
            tcp = obs.get("tcp_pose")
            if tcp is not None and hasattr(tcp, "shape") and tcp.shape == (4, 4):
                cur_R = np.array(tcp[:3, :3], dtype=float)
        except Exception:
            pass
        if cur_R is None:
            cur_R = Rotation.from_rotvec(self.GRASP_ORIENTATION).as_matrix()
        cur_rotvec = Rotation.from_matrix(cur_R).as_rotvec()

        targets = []
        labels = []
        running_pos = np.array(cur_pose, dtype=float).copy()
        for note in run:
            atype = note.get("type", "")
            params = note.get("params", {})
            if atype == "move_relative":
                running_pos = running_pos + np.array(
                    [
                        params.get("dx", 0.0),
                        params.get("dy", 0.0),
                        params.get("dz", 0.0),
                    ],
                    dtype=float,
                )
                targets.append(running_pos.copy())
                labels.append(f"rel(dz={params.get('dz', 0):.2f})")
            elif atype == "move_to_keypoint":
                kp = params.get("keypoint_label", "")
                det = self.detection_map.get(kp)
                if det is None:
                    logger.info(
                        "[legato] '%s' not in detection_map; "
                        "falling back to per-action dispatch",
                        kp,
                    )
                    return False
                pos = np.array(det["position_3d"], dtype=float)
                pos = pos + np.array(
                    [
                        params.get("offset_x", 0.0),
                        params.get("offset_y", 0.0),
                        params.get("offset_z", 0.0),
                    ],
                    dtype=float,
                )
                if pos[2] < self.TABLE_Z_FLOOR:
                    pos[2] = self.TABLE_Z_FLOOR
                running_pos = pos.copy()
                targets.append(pos.copy())
                labels.append(f"kp({kp})")
            else:
                return False

        q_seed = np.asarray(self.robot.get_joint_positions(), dtype=float)[:7].copy()
        q_solutions = []
        for tgt, lbl in zip(targets, labels):
            q_sol = _ik(tgt, cur_rotvec, q_seed)
            if q_sol is None:
                logger.info("[legato] IK failed at %s; falling back", lbl)
                return False
            q_solutions.append(q_sol)
            q_seed = q_sol

        if franky is None:
            return False

        rdf_val = 0.4
        rdf = franky.RelativeDynamicsFactor(rdf_val)
        franky_wps = []
        for q in q_solutions:
            js = franky.JointState(np.asarray(q, dtype=float))
            franky_wps.append(
                franky.JointWaypoint(
                    target=js,
                    reference_type=franky.ReferenceType.Absolute,
                    relative_dynamics_factor=rdf,
                )
            )
        motion = franky.JointWaypointMotion(franky_wps, return_when_finished=True)
        logger.info("[legato] phrase: %s (%d notes)", " -> ".join(labels), len(labels))

        # Pre-legato barrier: settle residual velocity
        try:
            franka_drv = getattr(self.robot, "_robot", None)
            inner = (
                getattr(franka_drv, "_robot", None) if franka_drv is not None else None
            )
            joint_vel = None
            if inner is not None and hasattr(inner, "state"):
                state = inner.state
                qd = getattr(state, "q_d", None) or getattr(state, "dq", None)
                if qd is not None:
                    joint_vel = float(np.max(np.abs(np.asarray(qd))))
            if joint_vel is None or joint_vel > 0.02:
                self._wait_stationary_after_servo()
        except Exception:
            try:
                self._wait_stationary_after_servo()
            except Exception:
                pass

        # Recover leftover error state
        try:
            franka_drv = getattr(self.robot, "_robot", None)
            if franka_drv is not None and hasattr(franka_drv, "recover_from_errors"):
                if getattr(franka_drv, "has_errors", False):
                    franka_drv.recover_from_errors()
        except Exception:
            pass

        t0 = time.time()
        legato_succeeded = False
        try:
            franky_robot.move(motion, asynchronous=False)
            legato_succeeded = True
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
                    "motion finished commanded",
                )
            )
            if recoverable and "discontinuity" in msg:
                logger.info(
                    "[legato] discontinuity reflex; retrying once at half rdf "
                    "after settle (was rdf=%.2f)",
                    rdf_val,
                )
                try:
                    if hasattr(franky_robot, "recover_from_errors"):
                        franky_robot.recover_from_errors()
                except Exception:
                    pass
                try:
                    self._wait_stationary_after_servo()
                except Exception:
                    pass
                rdf_slow = franky.RelativeDynamicsFactor(rdf_val * 0.5)
                franky_wps_slow = [
                    franky.JointWaypoint(
                        target=franky.JointState(np.asarray(q, dtype=float)),
                        reference_type=franky.ReferenceType.Absolute,
                        relative_dynamics_factor=rdf_slow,
                    )
                    for q in q_solutions
                ]
                motion_slow = franky.JointWaypointMotion(
                    franky_wps_slow, return_when_finished=True
                )
                try:
                    franky_robot.move(motion_slow, asynchronous=False)
                    legato_succeeded = True
                except Exception as exc2:
                    logger.warning(
                        "[legato] retry at rdf=%.2f also failed (%s); "
                        "falling back to per-action",
                        rdf_val * 0.5,
                        exc2,
                    )
                    return False
            elif recoverable:
                logger.warning("[legato] phrase aborted (%s); falling back", exc)
                return False
            else:
                logger.error("[legato] phrase failed: %s", exc)
                return False
        if not legato_succeeded:
            return False

        try:
            self._wait_stationary_after_servo()
        except Exception:
            pass

        elapsed = time.time() - t0
        per_action_dur = elapsed / max(len(run), 1)
        for note, lbl, tgt in zip(run, labels, targets):
            atype = note.get("type", "")
            self._results.append(
                ExecutionResult(
                    action_type=atype,
                    success=True,
                    message=f"Played as legato note in phrase: {lbl} -> {tgt.tolist()}",
                    duration=per_action_dur,
                )
            )
            if atype == "move_to_keypoint":
                self._last_place_label = note.get("params", {}).get(
                    "keypoint_label", ""
                )
        return True

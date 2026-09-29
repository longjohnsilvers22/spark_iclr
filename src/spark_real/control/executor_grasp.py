"""
GraspMixin: gripper control and grasp verification for ScoreExecutor.
"""

import logging
import os
import time
from dataclasses import dataclass, field
from typing import Optional, Tuple

import numpy as np
from scipy.spatial.transform import Rotation

from spark_real.control import success_verifier
from spark_real.control.executor_types import ExecutionResult
from spark_real.control.grasp_outcome import GraspOutcome
from spark_real.control.grasp_strategy import (
    resolve_grasp_orientation,
    resolve_strategy,
)
from spark_real.skills import registry as skill_registry

logger = logging.getLogger(__name__)


def _call_force_publish(fn):
    """Ask for a forced publish, tolerating drivers without the kwarg."""
    try:
        return fn(force=True)
    except TypeError:
        return fn()


def _call_maybe_passive(fn, publish: bool):
    """
    Call a gripper getter, honouring ``publish`` when the driver supports it.

    Drivers without a ``publish`` kwarg get the plain call.
    """
    if publish:
        return fn()
    try:
        return fn(publish=False)
    except TypeError:
        return fn()


# Force-verify default (N) when neither config nor env sets one. The
# close-on-air transient on this rig is ~0.6 N. A DEFAULT, not a floor:
# grasp.force_empty_n is authoritative and may lower it.
GRASP_FORCE_EMPTY_DEFAULT_N = 2.0
# Extra downward pull (N) at which the confirming lift CORROBORATES "holding".
# Corroboration only, it can never veto. Config: grasp.lift_delta_min_n.
GRASP_LIFT_DELTA_MIN_N = 0.5


@dataclass
class ForceVerdict:
    """What TCP force says about the jaws, and the evidence behind it."""

    peak_n: float
    lift_delta_n: float
    lift_corroborates: bool
    holding: bool


def force_verdict(
    clamp_mag: float,
    clamp_dz: float,
    lift_mag: float,
    lift_dz: float,
    empty_n: float,
    lift_delta_min_n: float,
) -> ForceVerdict:
    """Decide "force says holding" from the clamp + confirming-lift readings.

    Pure and module-level so the decision is testable without a robot.

    ``peak`` folds in BOTH readings: a light object (silverware is ~0.2 N)
    shows up in the clamp reaction, not in its own weight, so lift_delta
    CORROBORATES but never vetoes. Closed-on-air is caught by ``peak``: the
    ~0.6 N transient stays below empty_n.
    """
    peak = max(abs(clamp_mag), abs(lift_mag), abs(clamp_dz), abs(lift_dz))
    delta = float(clamp_dz - lift_dz)
    return ForceVerdict(
        peak_n=float(peak),
        lift_delta_n=delta,
        lift_corroborates=delta >= float(lift_delta_min_n),
        holding=float(peak) >= float(empty_n),
    )


@dataclass
class GraspVerdict:
    """Structured result of the G1 grasp gate, for the success verifier.

    `held` is the executor's verdict; the rest is the evidence behind it.
    """

    held: bool
    source: str = ""
    gobj: Optional[bool] = None
    gripper_pos: Optional[int] = None
    force_peak_n: float = 0.0
    force_empty_n: float = 0.0
    lift_delta_n: Optional[float] = None
    lift_corroborates: Optional[bool] = None
    label: str = ""
    strategy: str = ""
    # Typed outcome (control.grasp_outcome vocabulary): 'secured' /
    # 'empty_close' / 'slip' / 'unknown'. Stamped by _record_grasp_verdict
    # when the producer did not set it; upgraded to 'slip' by
    # note_transport_drop when a SECURED grip later reads lost.
    outcome: Optional[str] = None
    t: float = field(default_factory=time.time)



# Sub-part words a planner appends when it wants the jaws on a specific part.
# "screwdriver handle" -> "screwdriver"; the parent carries the shape that says
# which way the tool points.
_SUBPART_WORDS = ("handle", "grip", "blade", "tip", "head", "shaft", "prongs", "tines")


def _parent_object_label(label, detection_map):
    """The whole-object label for a grasped sub-part, if it was also detected."""
    if not label or not detection_map:
        return None
    parts = str(label).split()
    if len(parts) < 2 or parts[-1].lower() not in _SUBPART_WORDS:
        return None
    parent = " ".join(parts[:-1])
    return parent if parent in detection_map else None

class GraspMixin:
    """
    Gripper squeeze, grasp execution, and grasp verification.
    """

    _GRASP_WIDTH_MIN_M = 0.004
    _GRASP_WIDTH_MAX_M = 0.080

    # Robotiq close / re-squeeze force (0-100 percent). The closed POSITION
    # (pos 1.0) takes up slack on a compressible object; this force CAP
    # protects it from over-crush (100 crushed soft plushies). A BT `force`
    # param still wins; env SPARK_GRASP_CLOSE_FORCE overrides here.
    GRASP_CLOSE_FORCE = 60
    # Settle pause (s) after the clamp / at a stationary transport waypoint,
    # BEFORE the forced gObj publish+read, so the read is fresh rather than a
    # mid-move stale value. Env: SPARK_GRASP_VERIFY_SETTLE_S.
    GRASP_VERIFY_SETTLE_S = 0.4

    # --- Robotiq 2F-85 jaw-travel model -------------------------------------
    # The settle must be a sleep, not a condition wait: reading the jaw
    # registers uploads a publish program, and a new program REPLACES the
    # rq_close_and_wait still running on the controller, so polling the close
    # would cancel the close. The sleep is sized from the datasheet upper
    # bound on travel time.
    #
    # 2F-85: 85 mm stroke, closing speed 20-150 mm/s over the speed_norm range.
    ROBOTIQ_STROKE_MM = 85.0
    ROBOTIQ_SPEED_MIN_MM_S = 20.0
    ROBOTIQ_SPEED_MAX_MM_S = 150.0
    # UR10eDriver.close_gripper / open_gripper already sleep this much after the
    # send, so the caller's settle only has to cover what is left.
    ROBOTIQ_DRIVER_SETTLE_S = 0.5
    # Margin on top of the datasheet travel time (send + compile + the
    # controller finishing rq_close_and_wait's own bookkeeping).
    ROBOTIQ_SETTLE_MARGIN_S = 0.15
    # Jaw counts, 0 (open) .. 255 (closed).
    ROBOTIQ_POS_MAX = 255.0
    # UNMEASURED ASSUMPTION: UR10eDriver.open_gripper passes no speed_norm, so
    # the URCap's own open speed is unknown on this rig. The open settle is
    # sized as if it ran at this speed_norm (the one the grasp closes at).
    # Assuming the datasheet's slowest (20 mm/s) would mean a 3.9 s settle per
    # open; at 60 the open pays 0.5 s (driver) + ~0.52 s.
    ROBOTIQ_OPEN_SPEED_NORM_ASSUMED = 60
    # Jaw position at/below which the jaws count as ALREADY OPEN, so
    # _ensure_jaws_open may skip commanding an open. Much tighter than the
    # release witness's GRIPPER_RELEASED_MAX_POS (60): this one gates whether
    # it is safe to descend, so it demands the jaws be within ~5 mm of the
    # open stop (15/255 of an 85 mm stroke). Too permissive here means
    # descending with part-closed jaws.
    JAWS_OPEN_MAX_POS = 15

    def _force_empty_n(self) -> float:
        """Resolve the force-verify empty threshold (N).

        env > grasp.force_empty_n > module default. No floor: the configured
        value is authoritative in both directions.
        """
        env = os.environ.get("SPARK_GRASP_FORCE_EMPTY_N")
        if env is not None:
            try:
                return float(env)
            except ValueError:
                logger.warning("SPARK_GRASP_FORCE_EMPTY_N=%r ignored", env)
        return float(getattr(self, "GRASP_FORCE_EMPTY_N", GRASP_FORCE_EMPTY_DEFAULT_N))

    def _lift_delta_min_n(self) -> float:
        return float(
            os.environ.get(
                "SPARK_GRASP_LIFT_DELTA_MIN_N",
                getattr(self, "GRASP_LIFT_DELTA_MIN_N", GRASP_LIFT_DELTA_MIN_N),
            )
        )

    def _record_grasp_verdict(self, verdict: "GraspVerdict") -> "GraspVerdict":
        """Publish the G1 verdict for the success verifier's gate.

        Routed through success_verifier so the gate flag is set and a
        successful re-grasp clears an earlier transport drop; the structured
        verdict itself is stored verbatim for the trace. Also stamps the typed
        GraspOutcome (gObj / closed-stop position / force corroboration) when
        the producer did not set one.
        """
        if verdict.outcome is None:
            try:
                from spark_real.control.grasp_outcome import (
                    classify_grasp_endpoint,
                )

                verdict.outcome = classify_grasp_endpoint(
                    verdict.gobj,
                    verdict.gripper_pos,
                    closed_pos=float(
                        getattr(self, "GRIPPER_FULLY_CLOSED", 250)
                    ),
                    force_holding=(
                        verdict.held if verdict.source == "force" else None
                    ),
                ).value
            except Exception:  # noqa: BLE001 - annotation must never break a grasp
                pass
        success_verifier.record_grasp_verdict(
            self, verdict.held, verdict.source, verdict=verdict
        )
        return verdict

    def _gripper_type(self) -> str:
        """
        Return the underlying gripper hardware class.

        Walks through SafeRobot's __getattr__ chain so wrappers are
        transparent. Defaults to "robotiq_2f85" for backward compat.
        """
        target = self.robot
        seen = set()
        while target is not None and id(target) not in seen:
            seen.add(id(target))
            gt = getattr(target, "GRIPPER_TYPE", None)
            if isinstance(gt, str):
                return gt
            target = getattr(target, "_robot", None)
        return "robotiq_2f85"

    def _robotiq_travel_settle_s(
        self, speed_norm: int, from_pos: Optional[int] = None, to_pos: float = None
    ) -> float:
        """Settle a jaw move needs, from the datasheet.

        ``speed_norm`` is the 0-100 knob the executor passes to the driver;
        ``from_pos``/``to_pos`` are jaw counts (0 open .. 255 closed). Unknown
        ``from_pos`` means "assume the full stroke", which is the conservative
        direction. The driver's own post-send sleep is subtracted, so the return
        value is only the part the caller still has to wait out; it is never
        negative and never below the margin.
        """
        frac = float(np.clip(float(speed_norm) / 100.0, 0.0, 1.0))
        mm_s = self.ROBOTIQ_SPEED_MIN_MM_S + frac * (
            self.ROBOTIQ_SPEED_MAX_MM_S - self.ROBOTIQ_SPEED_MIN_MM_S
        )
        if from_pos is None or to_pos is None:
            travel_frac = 1.0
        else:
            travel_frac = abs(float(to_pos) - float(from_pos)) / self.ROBOTIQ_POS_MAX
            travel_frac = float(np.clip(travel_frac, 0.0, 1.0))
        travel_s = self.ROBOTIQ_STROKE_MM * travel_frac / max(mm_s, 1e-6)
        return float(
            max(
                self.ROBOTIQ_SETTLE_MARGIN_S,
                travel_s + self.ROBOTIQ_SETTLE_MARGIN_S - self.ROBOTIQ_DRIVER_SETTLE_S,
            )
        )

    def _ensure_jaws_open(self, context: str = "") -> bool:
        """Open the jaws only if they are not already open. True if commanded.

        Requires a stationary arm (precondition for the register publish).
        Conservative on every unknown: a failed read, a non-Robotiq gripper,
        or a position at/above the open threshold all fall through to
        commanding the open. Descending with part-closed jaws is a collision,
        so the open is only SKIPPED when the register proves the jaws open.
        """
        pos = None
        if self._gripper_type() == "robotiq_2f85":
            try:
                _obj, pos = self._read_gripper_state()
            except Exception as exc:  # noqa: BLE001
                logger.debug("_ensure_jaws_open read failed: %s", exc)
                pos = None
        if pos is not None and pos <= self.JAWS_OPEN_MAX_POS:
            logger.info(
                "%s: jaws already open (pos=%s <= %d); skipping open_gripper",
                context or "gripper",
                pos,
                self.JAWS_OPEN_MAX_POS,
            )
            return False
        try:
            self.robot.open_gripper()
        except Exception as exc:  # noqa: BLE001
            logger.warning("%s open_gripper failed: %s", context or "gripper", exc)
            return False
        speed = self.ROBOTIQ_OPEN_SPEED_NORM_ASSUMED
        self._abort_sleep(
            self._robotiq_travel_settle_s(speed, from_pos=pos, to_pos=0.0)
            if pos is not None
            else self._robotiq_travel_settle_s(speed)
        )
        return True

    def _wait_force_settled(
        self, timeout: float = 0.3, tol_n: float = 0.5, poll: float = 0.02
    ) -> bool:
        """Block until the TCP force channel is quiet enough to baseline off.

        Settled means two consecutive get_tcp_force samples agree within
        ``tol_n``. The rtde_r channel is independent of the URScript socket,
        so this costs no program upload and cannot cancel a gripper program.
        Returns True if it settled, False if it ran out the timeout.
        """
        t_end = time.time() + float(timeout)
        prev = None
        while True:
            self._check_abort()
            try:
                ft = np.asarray(self.robot.get_tcp_force(), dtype=float)
                cur = ft[:3].copy() if ft.size >= 3 else None
            except Exception:  # noqa: BLE001
                cur = None
            if cur is None:
                self._abort_sleep(max(0.0, t_end - time.time()))
                return False
            if prev is not None and float(np.linalg.norm(cur - prev)) < tol_n:
                return True
            prev = cur
            remaining = t_end - time.time()
            if remaining <= 0:
                return False
            time.sleep(min(poll, remaining))

    def _gripper_squeeze(self, force: int, speed: int = 50, settle: float = 0.3):
        """
        Close gripper around an object.

        Dispatches via GRIPPER_TYPE: robotiq_2f85 uses the UR fast path,
        franka_hand uses the shim set_gripper_position, others fall back
        to close_gripper.
        """
        gt = self._gripper_type()

        # Width-targeted path
        tw = getattr(self, "_last_grasp_target_width", None)
        if tw is not None and hasattr(self.robot, "grasp_to_width"):
            try:
                self.robot.grasp_to_width(
                    width=float(tw),
                    force=min(force, 100),
                    speed=float(speed) / 100.0 if speed > 1 else float(speed),
                )
                self._abort_sleep(settle)
                return
            except Exception as exc:
                logger.warning(
                    "grasp_to_width(%.3f) raised: %s; falling back to "
                    "force-only close",
                    tw,
                    exc,
                )

        if gt == "robotiq_2f85" and hasattr(self.robot, "close_gripper"):
            # Full close to the mechanical stop (rq_close_and_wait), not the
            # norm-scaled move that halts at rPR~226 leaving a ~29/255 gap.
            self.robot.close_gripper(speed=speed, force=min(force, 100))
            self._abort_sleep(settle)
            return
        if hasattr(self.robot, "set_gripper_position"):
            try:
                self.robot.set_gripper_position(1.0, speed=speed, force=min(force, 100))
                self._abort_sleep(settle)
                return
            except Exception:
                pass
        self.robot.close_gripper()
        self._abort_sleep(settle)

    def _robotiq_resqueeze(
        self, speed: int = 50, force: int = 100, settle: float = 1.0
    ) -> bool:
        """
        Robotiq-only partial-close re-squeeze. Returns True if issued.
        """
        gt = self._gripper_type()
        if gt == "robotiq_2f85":
            if hasattr(self.robot, "close_gripper"):
                self.robot.close_gripper(speed=speed, force=min(force, 100))
                self._abort_sleep(settle)
                return True
            return False
        if gt == "franka_hand":
            return False
        logger.warning(
            "Unknown GRIPPER_TYPE=%r; skipping partial-close "
            "re-squeeze (driver-specific safety unknown)",
            gt,
        )
        return False

    def _get_gripper_position(self, publish: bool = True) -> Optional[int]:
        """
        Jaw position on the Robotiq count scale, 0 (open) .. 255 (closed).

        Robotiq: the register read. ``publish=False`` reads the register a
        PRECEDING _force_gripper_publish already refreshed, instead of paying
        for a second publish of the same values. See _read_gripper_state.

        Other drivers: their get_gripper_position (already 0..255), else
        get_gripper_width against _GRASP_WIDTH_MAX_M. None only when there is
        no position source.
        """
        try:
            if hasattr(self.robot, "get_gripper_position"):
                return _call_maybe_passive(
                    self.robot.get_gripper_position, publish
                )
            if hasattr(self.robot, "get_gripper_width"):
                frac = 1.0 - float(self.robot.get_gripper_width()) / self._GRASP_WIDTH_MAX_M
                return float(np.clip(frac, 0.0, 1.0)) * 255.0
        except Exception:
            pass
        return None

    # A gripper position at/below this means the jaws are essentially open
    # (open reads ~2.5, fully closed 250+). gObj has read True at pos=5 with
    # the jaws never closed; open jaws cannot hold anything, whatever a flag
    # says.
    JAWS_OPEN_POS_MAX = 60

    def _jaws_contradict_holding(self, pos) -> bool:
        """True when the gripper position is physically incompatible with
        holding an object, RE-READ once before judging.

        The gObj/position registers go stale on the contended script path, so
        a single open-looking sample is not enough to call a grasp failed, but
        two in a row against a flag that claims contact means the flag is the
        stale one. Never fabricates a pass.
        """
        try:
            if pos is None or float(pos) > self.JAWS_OPEN_POS_MAX:
                return False
        except (TypeError, ValueError):
            return False
        try:
            self._abort_sleep(0.15)
            gobj2, pos2 = self._read_gripper_state()
            if pos2 is None or float(pos2) > self.JAWS_OPEN_POS_MAX:
                logger.info(
                    "[grasp-check] first position sample %s was stale; "
                    "re-read %s is consistent with holding", pos, pos2,
                )
                return False
            logger.warning(
                "[grasp-check] jaws read OPEN twice (pos=%s then %s, open<=%d) "
                "while the object-detect flag claims contact -- the flag is "
                "stale, not the geometry",
                pos, pos2, self.JAWS_OPEN_POS_MAX,
            )
            return True
        except Exception:  # noqa: BLE001 - unknown stays non-contradictory
            return False

    def _vision_says_still_on_table(self, label: str):
        """Did the object stay put? True = it never left the table.

        Compares a fresh detection against the position the object had when
        the grasp was planned; an object still within VISION_MOVED_TOL_M of
        where it started was not carried.

        Returns True (still there -> NOT grasped), False (moved or gone ->
        consistent with grasped), or None (no evidence).
        """
        try:
            label = label or getattr(self, "_last_keypoint_label", "") or ""
            if not label:
                return None
            prior = ((self.detection_map or {}).get(label) or {}).get("position_3d")
            if prior is None:
                return None
            prior = np.asarray(prior, dtype=float)
            pipe = getattr(self, "_pipeline", None)
            if pipe is None:
                return None
            base_prompt = label.rsplit(" ", 1)[0] if label[-1:].isdigit() else label
            caps = pipe.capture()
            merged = pipe.merge_detections(pipe.detect(caps, prompts=[base_prompt]))
            best = None
            for d in merged or []:
                pos = getattr(d, "position_3d", None)
                if pos is None:
                    continue
                dist = float(np.linalg.norm(np.asarray(pos, float)[:2] - prior[:2]))
                if best is None or dist < best:
                    best = dist
            if best is None:
                # Not re-detected is NOT evidence of a grasp: the gripper
                # occludes it just as effectively as carrying it.
                logger.info("[grasp-check] '%s' not re-detected; no verdict", label)
                return None
            still = best <= self.VISION_MOVED_TOL_M
            logger.info(
                "[grasp-check] '%s' nearest detection is %.1f cm from where it "
                "was before the close -> %s",
                label, best * 100,
                "STILL ON THE TABLE" if still else "it moved with us",
            )
            return still
        except Exception as exc:  # noqa: BLE001 - a failed look is not a verdict
            logger.warning("[grasp-check] vision cross-check failed: %s", exc)
            return None

    # An object that has not moved further than this since the grasp was
    # planned is still sitting where it was.
    VISION_MOVED_TOL_M = 0.04

    def _is_object_detected(self, publish: bool = True) -> Optional[bool]:
        """Robotiq gObj flag, or None if unreadable. See _get_gripper_position."""
        try:
            if hasattr(self.robot, "is_object_detected"):
                return bool(
                    _call_maybe_passive(self.robot.is_object_detected, publish)
                )
        except Exception as exc:
            logger.warning("object-detect read failed: %s", exc)
        return None

    def _read_gripper_state(self):
        """
        One publish, then both register reads. Returns ``(gObj, position)``.

        gObj and jaw position live in two registers that a SINGLE
        _publish_gripper_state refreshes together; each extra publish uploads
        a URScript program and waits for the controller to run it.
        """
        self._force_gripper_publish()
        # _call_maybe_passive, not a direct publish=False call: subclasses and
        # test doubles may override these with a no-kwarg signature.
        return (
            _call_maybe_passive(self._is_object_detected, False),
            _call_maybe_passive(self._get_gripper_position, False),
        )

    def _force_gripper_publish(self):
        """Force ONE fresh Robotiq state publish before a stationary read.

        The driver only refreshes rq_current_pos_norm / rq_is_object_detected
        into the RTDE registers via _publish_gripper_state, and REFUSES to do
        so while the arm is moving (TCP-speed guard). A reliable gObj/position
        read requires the arm STATIONARY and a forced publish first. No-op for
        non-robotiq drivers or on any read error.
        """
        try:
            if hasattr(self.robot, "_publish_gripper_state"):
                # force=True: the caller guarantees a parked arm, so the
                # driver's motion lease must not withhold the refresh. The
                # driver still checks measured motion before it publishes.
                _call_force_publish(self.robot._publish_gripper_state)
        except Exception as exc:
            logger.debug("_force_gripper_publish failed: %s", exc)

    def _grip_intact(self) -> bool:
        """Cheap per-waypoint drop check at a STATIONARY arm pose.

        Primary signal is the driver's object-detect flag (Robotiq gObj,
        Franka is_grasped: True = jaws stopped on an object). If it does not
        confirm, a still-spread jaw position (pos < GRIPPER_FULLY_CLOSED, on
        the 0..255 scale _get_gripper_position gives every driver) also counts
        as holding: a soft object keeps the jaws off the fully-closed stop.
        Only when the flag is not True AND the jaws have reached the
        fully-closed stop is it a DROP. Conservative: returns True for a
        driver with no jaw state at all and on any read failure (never
        fabricate a drop).
        """
        # Arm must be stationary here (caller's responsibility).
        obj, pos = self._read_gripper_state()
        if obj is None and pos is None and self._gripper_type() != "robotiq_2f85":
            return True
        if obj is True:
            logger.info("_grip_intact: HOLDING (object-detect flag)")
            return True
        if pos is not None and pos < self.GRIPPER_FULLY_CLOSED:
            logger.info(
                "_grip_intact: HOLDING (jaws still spread on object, "
                "pos=%d < closed=%d, gObj=%s)",
                pos,
                self.GRIPPER_FULLY_CLOSED,
                obj,
            )
            return True
        logger.warning(
            "_grip_intact: DROPPED (gObj=%s, pos=%s >= closed=%d)",
            obj,
            pos,
            self.GRIPPER_FULLY_CLOSED,
        )
        return False

    @staticmethod
    def _derive_target_width_from_detection(
        detection: dict,
    ) -> Optional[Tuple[float, float, str]]:
        """
        Synthesize a Franka Hand grasp width + force from mask geometry.

        Uses 0.6x the OBB minor axis (depth-backprojected to meters) as the
        target width. Returns (target_width_m, force_hint_N, source) or None.
        """
        if detection is None:
            return None
        try:
            obb_minor_m = float(detection.get("obb_minor_m", 0.0) or 0.0)
        except (TypeError, ValueError):
            return None
        if obb_minor_m <= 0.0:
            return None

        raw_width = obb_minor_m * 0.6
        target_width = float(
            np.clip(
                raw_width,
                GraspMixin._GRASP_WIDTH_MIN_M,
                min(0.060, GraspMixin._GRASP_WIDTH_MAX_M),
            )
        )

        force_hint_N = 35.0
        try:
            aspect_ratio = float(detection.get("aspect_ratio", 1.0) or 1.0)
        except (TypeError, ValueError):
            aspect_ratio = 1.0
        if obb_minor_m < 0.010:
            force_hint_N = 20.0
        elif aspect_ratio >= 3.0:
            force_hint_N = 25.0
        elif obb_minor_m >= 0.050:
            force_hint_N = 45.0

        source = "mask_obb_minor_x0.6"
        if raw_width != target_width:
            source += "_clamped"
        return (target_width, force_hint_N, source)

    def _grasp_v2(self, params: dict, t0: float) -> ExecutionResult:
        """Mid-height grasp with lift-based verify (gated by SPARK_GRASP_V2=1).

        (1) descend the OPEN jaws to block MID-height so they straddle the
        block, (2) clamp with force, (3) verify by LIFTING 2cm and checking
        the jaws stayed partly open on an object. Hard-clamped to TABLE_Z_FLOOR;
        gentle descent; abort-safe.

        Tunables (env, restart to apply): SPARK_GRASP_DEPTH_M (default 0.022,
        how far below the perceived top to aim, ~half a 48mm block) and
        SPARK_GRASP_EMPTY_POS (default 205, gripper pos at/above which the jaws
        are considered fully shut = empty; a held 36mm block reads ~147).
        """
        self._check_abort()
        # Grasp force (0-100): GRASP_CLOSE_FORCE default, BT `force` param wins.
        # A FORCE cap only: the close still commands the fully-closed POSITION.
        force = float(params.get("force", self.GRASP_CLOSE_FORCE))
        # No target_width param -> full close (pos 1.0), not a partial width
        # setpoint. Only an explicit BT target_width narrows it.
        tw = params.get("target_width", None)
        self._last_grasp_target_width = float(tw) if tw is not None else None
        # Class default (overridden by the `grasp:` YAML block); env var wins.
        depth = float(os.environ.get("SPARK_GRASP_DEPTH_M", self.GRASP_DEPTH_M))

        # 1. Descend OPEN jaws to block mid-height, clamped above the table.
        self._ensure_jaws_open("grasp")
        self._wait_force_settled()
        # Baseline TCP force (rtde_r channel) with the jaws open. The gripper
        # position/OBJ REGISTER reads ride the contended _send_script path and
        # come back STALE, so they cannot verify a grasp alone.
        try:
            _bft = np.asarray(self.robot.get_tcp_force(), dtype=float)
            base_ft = _bft[:6].copy() if _bft.size >= 6 else np.zeros(6)
        except Exception:
            base_ft = np.zeros(6)
        start = self._get_current_position()
        # Reuse the approach's (possibly OBB-yawed) orientation so the jaws stay
        # aligned to the object through the descent + clamp.
        grasp_orient = getattr(self, "_active_grasp_orient", None) or self.GRASP_ORIENTATION

        # Direct descent to the grasp height computed from the detection. depth
        # is how far below the approach pose to aim; clamp to the safe
        # closed-fingertip floor. Force is only a GUARD: if the jaws feel contact
        # before reaching the target (a thick block, a mis-detected top), stop
        # early, back off a couple mm, and clamp there instead of ramming.
        target_z = max(float(start[2]) - depth, self.TABLE_Z_FLOOR + 0.002)
        contact_n = float(os.environ.get("SPARK_GRASP_CONTACT_N", "8.0"))
        backoff = float(os.environ.get("SPARK_GRASP_BACKOFF_M", "0.003"))

        def _fz_delta():
            try:
                ft = np.asarray(self.robot.get_tcp_force(), dtype=float)
                return abs(float(ft[2])) - abs(float(base_ft[2])) if ft.size >= 3 else 0.0
            except Exception:
                return 0.0

        contact_z = None
        if hasattr(self.robot, "send_velocity"):
            desc_v = max(0.015, self.velocity * 0.15)  # normal descent, not a crawl
            logger.info(
                "Grasp v2: descend from z=%.3f to target z=%.3f (v=%.3f, "
                "contact-guard>%.1fN)",
                float(start[2]), target_z, desc_v, contact_n,
            )
            t_start = time.time()
            while not self._abort:
                cur = self._get_current_position()
                if cur[2] <= target_z:            # reached computed height, no contact
                    break
                if time.time() - t_start > 15.0:
                    logger.warning("Grasp v2: descent time cap at z=%.3f", cur[2])
                    break
                d = _fz_delta()
                if abs(d) > contact_n:            # GUARD: unexpected early contact
                    contact_z = cur[2]
                    logger.info("Grasp v2: early contact at z=%.3f (|dFz|=%.1fN)", cur[2], d)
                    break
                self.robot.send_velocity([0.0, 0.0, -desc_v, 0.0, 0.0, 0.0], 0.5, 0.06)
                time.sleep(0.03)
            try:
                self.robot.send_velocity([0.0] * 6, 1.0, 0.05)
            except Exception:
                pass
            if contact_z is not None:
                # Backed off a couple mm above the contact point, then clamp there.
                back = self._get_current_position()
                back[2] = min(contact_z + backoff, float(start[2]))
                self._move_to(back, grasp_orient, velocity=self.velocity * 0.15)
                logger.info(
                    "Grasp v2: contact z=%.3f -> backed off %.0fmm to z=%.3f, clamping",
                    contact_z, backoff * 1000, back[2],
                )
        else:
            # Fallback (driver without velocity control): direct servo to target.
            descend = start.copy()
            descend[2] = target_z
            self._servo_to(descend, grasp_orient, velocity=self.velocity * 0.15)

        # 2. Clamp to the fully-closed POSITION at the capped `force`: the
        # closed position takes up slack on a soft object while the force cap
        # protects it from over-crush. The re-squeeze is CONDITIONAL: gObj
        # True at this stationary pose means the jaws stopped on an object and
        # there is no slack to take up; otherwise it runs with a settle sized
        # to the real travel from the measured position.
        self._check_abort()
        # Record the ACTUAL grasp TCP z at jaw close: the object hangs from
        # THIS height, not from perceived_top + GRIPPER_OPEN_Z_OFFSET. The
        # release computation reads it back (behind place.release_giveback).
        try:
            self._actual_grasp_tcp_z = float(self._get_current_position()[2])
        except Exception:  # noqa: BLE001 - bookkeeping must not stop a grasp
            self._actual_grasp_tcp_z = None
        self._gripper_squeeze(
            force=force, speed=60, settle=self._robotiq_travel_settle_s(60)
        )
        fast_ok = os.environ.get("SPARK_GRASP_SKIP_LIFT_VERIFY", "1") == "1"
        gobj_fast, gpos_fast = (None, None)
        # The squeeze settle above already covers rq_close_and_wait's full
        # stroke, so this read is fresh.
        if fast_ok:
            gobj_fast, gpos_fast = self._read_gripper_state()
        if gobj_fast is not True:
            self._robotiq_resqueeze(
                speed=50,
                force=int(force),
                settle=self._robotiq_travel_settle_s(
                    50, from_pos=gpos_fast, to_pos=self.ROBOTIQ_POS_MAX
                ),
            )
            if fast_ok:
                self._abort_sleep(
                    float(os.environ.get("SPARK_GRASP_VERIFY_SETTLE_S",
                                         self.GRASP_VERIFY_SETTLE_S))
                )
                gobj_fast, gpos_fast = self._read_gripper_state()

        # Fast path: a gObj flag confirming the jaws stopped on an object at
        # the stationary clamped pose skips the ~10s lift-based force verify
        # below (the fallback when gObj is unavailable or does NOT confirm).
        # SPARK_GRASP_SKIP_LIFT_VERIFY=0 forces the full lift verify.
        if fast_ok:
            # The flag alone is not enough: gObj has read True with the jaws
            # open (pos=5). A contradiction runs the full lift verify instead.
            if gobj_fast is True and self._jaws_contradict_holding(gpos_fast):
                logger.warning(
                    "Grasp v2: object-detect flag says held but the jaws are "
                    "open; NOT taking the fast path -- running the full lift "
                    "verify"
                )
                gobj_fast = None
            if gobj_fast is True:
                self._holding = True
                logger.info(
                    "Grasp v2: GRASPED (gObj fast-path, pos=%s; lift verify skipped)",
                    gpos_fast,
                )
                self._record_grasp_verdict(
                    GraspVerdict(
                        held=True,
                        source="gobj_fast",
                        gobj=True,
                        gripper_pos=gpos_fast,
                        force_empty_n=self._force_empty_n(),
                        label=self._last_keypoint_label or "",
                        strategy=getattr(self, "_active_grasp_strategy", "") or "",
                    )
                )
                return ExecutionResult(
                    action_type="grasp",
                    success=True,
                    message=f"Grasped (gObj object-detect, pos={gpos_fast})",
                    duration=time.time() - t0,
                )
            # gObj did not confirm -> fall through to the lift + force verify.

        # 3. Verify by TCP FORCE + a confirming lift, NOT the (stale) gripper
        # position register. A real object shows a clear wrist-force delta from
        # the squeeze reaction and/or its weight on the lift; empty air stays
        # near zero through both.
        def _fdelta():
            try:
                ft = np.asarray(self.robot.get_tcp_force(), dtype=float)
                if ft.size >= 6:
                    return (
                        float(np.linalg.norm(ft[:3] - base_ft[:3])),
                        float(ft[2] - base_ft[2]),
                    )
            except Exception:
                pass
            return 0.0, 0.0

        clamp_mag, clamp_dz = _fdelta()
        # Lift 2cm; a held block adds a downward pull (its weight) at the wrist.
        self._check_abort()
        lift = self._get_current_position()
        lift[2] += self.GRASP_LIFT_CHECK_M
        self._move_to(lift, grasp_orient, velocity=self.velocity * 0.3)
        self._abort_sleep(0.3)
        lift_mag, lift_dz = _fdelta()
        empty_n = self._force_empty_n()
        lift_min = self._lift_delta_min_n()
        fv = force_verdict(clamp_mag, clamp_dz, lift_mag, lift_dz, empty_n, lift_min)
        peak = fv.peak_n
        lift_delta_n = fv.lift_delta_n
        lift_holds = fv.lift_corroborates
        force_says_holding = fv.holding

        # PRIMARY signal: the Robotiq gObj object-detect flag, read at a
        # STATIONARY pose (settled after the 2cm lift) after a forced fresh
        # publish. gObj True = the jaws stopped on an object. Valid on a
        # COMPRESSIBLE object too: a soft plushie squeezes the jaws PAST the
        # "fully closed" position (pos ~252 >= 250) while genuinely held, so
        # the position veto must never override a True gObj.
        settle_verify = float(
            os.environ.get("SPARK_GRASP_VERIFY_SETTLE_S", self.GRASP_VERIFY_SETTLE_S)
        )
        self._abort_sleep(settle_verify)
        obj_detected, gpos = self._read_gripper_state()
        logger.info(
            "Grasp v2 verify: gObj=%s pos=%s force_peak=%.2fN (empty<%.2fN) "
            "clamp|df|=%.2f lift|df|=%.2f lift_delta=%.2fN (corroborates >= %.2f: %s)",
            obj_detected, gpos, peak, empty_n, clamp_mag, lift_mag,
            lift_delta_n, lift_min, lift_holds,
        )

        def _verdict(held: bool, source: str) -> None:
            self._record_grasp_verdict(
                GraspVerdict(
                    held=held,
                    source=source,
                    gobj=obj_detected,
                    gripper_pos=gpos,
                    force_peak_n=float(peak),
                    force_empty_n=float(empty_n),
                    lift_delta_n=lift_delta_n,
                    lift_corroborates=lift_holds,
                    label=self._last_keypoint_label or "",
                    strategy=getattr(self, "_active_grasp_strategy", "") or "",
                )
            )

        # 1. Positive gObj -> GRASPED. No position veto (plushie false-negative).
        if obj_detected is True:
            self._holding = True
            logger.info("Grasp v2: GRASPED (Robotiq object-detect, pos=%s)", gpos)
            _verdict(True, "gobj")
            return ExecutionResult(
                action_type="grasp",
                success=True,
                message=f"Grasped (gObj object-detect, pos={gpos})",
                duration=time.time() - t0,
            )

        # 2. gObj did NOT confirm (False or unavailable). Corroborate with TCP
        #    force. Force alone false-positives when the tool presses a flat
        #    plate / the table (large |dF| but jaws close FULLY on nothing), so
        #    here, and only where gObj has NOT confirmed, keep the pos>=CLOSED
        #    veto to reject that flat-plate/edge-miss case.
        if force_says_holding:
            if gpos is not None and gpos >= self.GRIPPER_FULLY_CLOSED:
                self._holding = False
                logger.info(
                    "Grasp v2: force peak %.2fN but jaws fully closed "
                    "(pos=%d >= %d) and gObj=%s -> empty (flat-plate/miss)",
                    peak, gpos, self.GRIPPER_FULLY_CLOSED, obj_detected,
                )
                _verdict(False, "jaws_fully_closed")
                return ExecutionResult(
                    action_type="grasp",
                    success=False,
                    message=f"Empty (jaws fully closed, pos={gpos})",
                    duration=time.time() - t0,
                )
            self._holding = True
            logger.info(
                "Grasp v2: GRASPED (force-corroborated, peak %.2fN, pos=%s, "
                "gObj=%s)",
                peak, gpos, obj_detected,
            )
            _verdict(True, "force")
            return ExecutionResult(
                action_type="grasp",
                success=True,
                message=f"Grasped (force-corroborated, {peak:.1f}N)",
                duration=time.time() - t0,
            )

        # 3. gObj not True AND force ~empty -> genuinely empty.
        self._holding = False
        logger.info(
            "Grasp v2: NOT grasped (gObj=%s, force peak %.2fN < %.2fN, "
            "lift_delta %.2fN)",
            obj_detected, peak, empty_n, lift_delta_n,
        )
        _verdict(False, "empty")
        return ExecutionResult(
            action_type="grasp",
            success=False,
            message=f"Empty (gObj={obj_detected}, force peak {peak:.1f}N)",
            duration=time.time() - t0,
        )

    def _resolve_grasp_node_strategy(self, params: dict) -> str:
        """Settle the strategy for THIS grasp and set _active_grasp_orient.

        `_active_grasp_orient` is cleared first and then set explicitly, so a
        grasp reached without a preceding approach (recovery re-entry,
        grasp_top_down, grasp_se3_flow) cannot inherit the PREVIOUS object's yaw.
        """
        approach_orient = getattr(self, "_active_grasp_orient", None)
        approach_strategy = getattr(self, "_active_grasp_strategy", None)
        approach_label = getattr(self, "_active_grasp_label", None)
        self._active_grasp_orient = None

        label = self._last_keypoint_label or ""
        det = self.detection_map.get(label) if label else None
        # An approach for a DIFFERENT object (or none at all) is stale.
        stale = approach_label != label
        node_asked = bool(params) and (
            "grasp_strategy" in params or "grasp_yaw_deg" in params
        )
        if node_asked or stale or not approach_strategy:
            strategy, reason = resolve_strategy(params or {}, det, self)
        else:
            strategy, reason = approach_strategy, "from approach"

        if strategy == "obb":
            # Reuse the approach's composed yaw when it is fresh (already
            # cable-limit checked there); otherwise re-derive from the detection.
            orient = None if (stale or node_asked) else approach_orient
            if orient is None:
                orient, used = resolve_grasp_orientation(
                    params or {}, det, self, context="grasp"
                )
                strategy = used
                orient = orient if used == "obb" else None
            self._active_grasp_orient = list(orient) if orient is not None else None
        self._active_grasp_strategy = strategy
        self._active_grasp_label = label
        # Record the held object's MASK on the path the grasp actually takes;
        # without it the downstream direction tests fall back to symmetric_yaw's
        # arbitrary mod-180 choice, which seats a tool end-for-end.
        self._held_mask = None
        self._held_axis_img = None
        self._held_axis_sign = None
        try:
            # Direction must be measured on the WHOLE tool, not the grasped
            # sub-part: a handle is symmetric end-for-end, the asymmetry lives
            # in the shaft. Grasp 'knife handle', orient by 'knife'.
            _m = det.get("_mask") if isinstance(det, dict) else None
            _src = label or "?"
            _parent = _parent_object_label(label, getattr(self, "detection_map", None))
            if _parent is not None:
                _pm = (self.detection_map.get(_parent) or {}).get("_mask")
                if _pm is not None:
                    _m, _src = _pm, _parent
            if _m is not None:
                from spark_real.perception.mask_geometry import (
                    _pca_obb,
                    heavy_end_sign,
                )
                _a, _ar, _, _ = _pca_obb(np.asarray(_m))
                self._held_mask = np.asarray(_m)
                self._held_axis_img = float(_a)
                self._held_axis_sign = float(heavy_end_sign(np.asarray(_m), _a))
                logger.info(
                    "[held-axis] '%s' image axis %.1f deg, ar %.2f, heavy end %s",
                    _src, np.rad2deg(_a), _ar,
                    {1.0: "+", -1.0: "-", 0.0: "UNKNOWN"}.get(
                        self._held_axis_sign, "?"),
                )
        except Exception as exc:  # noqa: BLE001 - never let this stop a grasp
            logger.warning("[held-axis] could not record the held mask: %s", exc)
        logger.info("[grasp-strategy] grasp node: %s (%s)", strategy, reason)
        return strategy

    def _grasp(self, params: dict, t0: float) -> ExecutionResult:
        """
        Close gripper and verify an object was grasped.
        """
        self._check_abort()
        strategy = self._resolve_grasp_node_strategy(params or {})
        if strategy in ("cgn", "se3"):
            skill = "grasp_cgn" if strategy == "cgn" else "grasp_se3"
            fwd = dict(params or {})
            fwd.setdefault("keypoint_label", self._last_keypoint_label or "")
            logger.info("[grasp-strategy] dispatching grasp -> %s", skill)
            if skill_registry.get(skill) is None:
                logger.warning(
                    "[grasp-strategy] %s is not registered; top-down instead", skill
                )
            else:
                return skill_registry.dispatch(skill, self, fwd)
        # Grasp v2 (mid-height grasp + force verify) is ROBOTIQ-ONLY: it exists
        # for the UR/Robotiq rig whose gripper position/OBJ register reads are
        # stale. Franka/other grippers keep the legacy path below.
        # SPARK_GRASP_V2=0 also reverts.
        if (
            os.environ.get("SPARK_GRASP_V2", "1") == "1"
            and self._gripper_type() == "robotiq_2f85"
        ):
            return self._grasp_v2(params, t0)
        label = self._last_keypoint_label or ""
        det = self.detection_map.get(label) if label else None

        # Priority chain: BT > mask-derived > legacy default.
        target_width = None
        force = None
        source = None

        bt_tw = params.get("target_width", None)
        if bt_tw is not None:
            target_width = float(bt_tw)
            bt_f = params.get("force", None)
            if bt_f is not None:
                force = float(bt_f)
            source = "bt"

        if target_width is None and det is not None:
            try:
                mask_prior = self._derive_target_width_from_detection(det)
            except Exception as exc:
                logger.warning("Mask-derived prior raised: %s", exc)
                mask_prior = None
            if mask_prior is not None:
                target_width = float(mask_prior[0])
                bt_f = params.get("force", None)
                force = float(bt_f) if bt_f is not None else float(mask_prior[1])
                source = f"mask:{mask_prior[2]}"

        if force is None:
            force = float(params.get("force", 50))
        if source is None:
            source = "legacy"

        logger.info(
            "Grasp params: target_width=%s force=%.1f source=%s label=%r",
            f"{target_width:.4f}m" if target_width is not None else "none",
            force,
            source,
            label,
        )

        self._last_grasp_target_width = (
            float(target_width) if target_width is not None else None
        )

        # Capture pre-grasp wrench baseline
        try:
            if hasattr(self.robot, "get_tcp_force"):
                ft0 = np.asarray(self.robot.get_tcp_force(), dtype=float)
                self._pre_grasp_force = ft0[:6].copy() if ft0.size >= 6 else np.zeros(6)
                self._pre_grasp_force_magnitude = float(np.linalg.norm(ft0[:3]))
            else:
                self._pre_grasp_force = np.zeros(6)
                self._pre_grasp_force_magnitude = 0.0
        except Exception:
            self._pre_grasp_force = np.zeros(6)
            self._pre_grasp_force_magnitude = 0.0
        logger.info(
            "Pre-grasp wrench: F=(%.2f,%.2f,%.2f) N, |F|=%.2f",
            self._pre_grasp_force[0],
            self._pre_grasp_force[1],
            self._pre_grasp_force[2],
            self._pre_grasp_force_magnitude,
        )

        try:
            tcp_pose = self._servo._get_tcp_pose()
            R_tcp = Rotation.from_rotvec(tcp_pose[3:6]).as_matrix()
            tx = R_tcp @ [1, 0, 0]
            ty = R_tcp @ [0, 1, 0]
            logger.info(
                "Pre-grasp orientation: tool_X=%.1f deg, tool_Y=%.1f deg",
                np.rad2deg(np.arctan2(tx[1], tx[0])),
                np.rad2deg(np.arctan2(ty[1], ty[0])),
            )
        except Exception:
            pass

        self._gripper_squeeze(force=force, speed=80, settle=1.5)
        self._robotiq_resqueeze(speed=50, force=100, settle=1.0)

        def _verdict(held: bool, source: str, pos, outcome: GraspOutcome) -> None:
            # G1 record for the success verifier, same shape as _grasp_v2's;
            # no extra register read on the exit path.
            self._record_grasp_verdict(
                GraspVerdict(
                    held=held,
                    source=source,
                    gripper_pos=pos,
                    force_empty_n=self._force_empty_n(),
                    label=self._last_keypoint_label or "",
                    strategy=getattr(self, "_active_grasp_strategy", "") or "",
                    outcome=outcome.value,
                )
            )

        grasp_tcp = self._get_current_position()
        logger.info(
            "Grasp TCP: [%.3f, %.3f, %.3f] (z=%.3f)",
            grasp_tcp[0],
            grasp_tcp[1],
            grasp_tcp[2],
            grasp_tcp[2],
        )

        if self._verify_grasp():
            if self.grasp_calibration is not None and self._last_keypoint_label:
                _final_z = self._get_current_position()[2]
                _delta_z = _final_z - self._grasp_perception_target_z
                self.grasp_calibration.record_success(
                    self._last_keypoint_label, _delta_z, n_retries=0
                )
        else:
            # Descent retry: try going lower in 1.5cm steps
            DESCENT_RETRIES = 5
            DESCENT_STEP = 0.015
            FLOOR_MARGIN = 0.002
            floor_z = self.TABLE_Z_FLOOR + FLOOR_MARGIN
            grasped = False
            hit_floor = False
            for retry in range(DESCENT_RETRIES):
                self._check_abort()
                self.robot.open_gripper()
                self._abort_sleep(0.3)
                current = self._get_current_position()
                target_z = current[2] - DESCENT_STEP
                if target_z <= floor_z:
                    target_z = floor_z
                    hit_floor = True
                    logger.warning(
                        "Grasp retry %d/%d: descent clamped to floor "
                        "z=%.3f (table=%.3f, margin=%.0fmm)",
                        retry + 1,
                        DESCENT_RETRIES,
                        target_z,
                        self.TABLE_Z_FLOOR,
                        FLOOR_MARGIN * 1000,
                    )
                current[2] = target_z
                logger.info(
                    "Grasp retry %d/%d: descending to z=%.3f",
                    retry + 1,
                    DESCENT_RETRIES,
                    current[2],
                )
                self._servo_to(
                    current, self.GRASP_ORIENTATION, velocity=self.velocity * 0.3
                )
                self._gripper_squeeze(force=force, speed=80, settle=1.0)
                # Force re-squeeze: the width-targeted _gripper_squeeze only
                # POSITIONS the jaws to the target width (never clamps), so a
                # retry would stop ~3.7cm wide on the block without gripping it.
                self._robotiq_resqueeze(speed=50, force=100, settle=1.0)
                if self._verify_grasp():
                    grasped = True
                    if self.grasp_calibration is not None and self._last_keypoint_label:
                        _final_z = self._get_current_position()[2]
                        _delta_z = _final_z - self._grasp_perception_target_z
                        self.grasp_calibration.record_success(
                            self._last_keypoint_label, _delta_z, n_retries=retry + 1
                        )
                    break
                if hit_floor:
                    break
            if not grasped:
                logger.warning(
                    "Grasp failed after %d retries at z=%.3f%s",
                    DESCENT_RETRIES,
                    self._get_current_position()[2],
                    " (reached floor)" if hit_floor else "",
                )
                self._holding = False
                _verdict(False, "legacy_retries_exhausted", None, GraspOutcome.EMPTY_CLOSE)
                return ExecutionResult(
                    action_type="grasp",
                    success=False,
                    message=(
                        f"Grasp failed after {DESCENT_RETRIES} "
                        f"descent retries" + (" (reached floor)" if hit_floor else "")
                    ),
                    duration=time.time() - t0,
                )

        # Post-lift verification
        logger.info("Grasp check passed, lifting 3cm to verify hold")
        self._holding = True
        current = self._get_current_position()
        lift_pos = current.copy()
        lift_pos[2] += 0.03
        self._move_to(lift_pos, self.GRASP_ORIENTATION, velocity=self.velocity * 0.3)
        self._abort_sleep(0.3)

        pre_pos = self._get_gripper_position()
        if self._gripper_type() == "robotiq_2f85":
            self._gripper_squeeze(force=100, speed=50, settle=0.5)
        else:
            self._abort_sleep(0.5)
        post_pos = self._get_gripper_position()

        if pre_pos is not None and post_pos is not None:
            logger.info("Post-lift grip check: pre=%d post=%d", pre_pos, post_pos)
            if post_pos >= self.GRIPPER_FULLY_CLOSED:
                logger.warning("Object lost during lift (pos=%d)", post_pos)
                self._holding = False
                _verdict(False, "legacy_lift_lost", post_pos, GraspOutcome.EMPTY_CLOSE)
                return ExecutionResult(
                    action_type="grasp",
                    success=False,
                    message=f"Object lost during lift (pos {post_pos})",
                    duration=time.time() - t0,
                )

        _verdict(True, "legacy_verify", post_pos, GraspOutcome.SECURED)
        return ExecutionResult(
            action_type="grasp",
            success=True,
            message=f"Grasped with force={force} (verified, pos={post_pos})",
            duration=time.time() - t0,
        )

    def _verify_grasp(self) -> bool:
        """
        Did the last grasp actually pick up an object?

        Reads four independent physical signals: TCP external force delta,
        gripper width position, driver-specific flag, and FlexiTac tactile.
        Returns True if evidence indicates an object is held.
        """
        # Width-fast-path
        tw = getattr(self, "_last_grasp_target_width", None)
        if tw is not None and hasattr(self.robot, "get_gripper_width"):
            try:
                achieved_w = float(self.robot.get_gripper_width())
                if achieved_w < 0.001:
                    logger.info(
                        "Verify width: target=%.3fm achieved=%.3fm "
                        "(jaws fully closed) -> NOT grasped",
                        tw,
                        achieved_w,
                    )
                    return False
                if (tw - 0.005) <= achieved_w <= (tw + 0.015):
                    logger.info(
                        "Verify width: target=%.3fm achieved=%.3fm -> HOLDING",
                        tw,
                        achieved_w,
                    )
                    return True
                logger.info(
                    "Verify width: target=%.3fm achieved=%.3fm "
                    "out of band, falling back to force/pos verify",
                    tw,
                    achieved_w,
                )
            except Exception as exc:
                logger.warning("Verify width read failed: %s", exc)

        # One publish for both registers; the gObj half is consumed by the
        # "Driver flag" block below.
        gobj_sample, pos = self._read_gripper_state()
        if pos is not None and pos >= self.GRIPPER_FULLY_CLOSED:
            logger.info(
                "Verify: gripper fully closed (pos=%d >= %d) -> NOT grasped",
                pos,
                self.GRIPPER_FULLY_CLOSED,
            )
            return False

        # Force feedback
        force_says_holding = None
        baseline_vec = getattr(self, "_pre_grasp_force", None)
        try:
            if baseline_vec is None:
                # No pre-grasp sample -> NO DELTA EXISTS, so this channel must
                # abstain. Substituting zeros would turn "unknown" into
                # "holding": the raw reading carries the tool's own weight
                # (~78 N in z). Position and object-detect decide instead.
                logger.info(
                    "Verify force: no pre-grasp baseline; force channel "
                    "abstains (position/object-detect decide)"
                )
            elif hasattr(self.robot, "get_tcp_force"):
                ft = np.asarray(self.robot.get_tcp_force(), dtype=float)
                if ft.size >= 6:
                    f_now_vec = ft[:6].copy()
                    delta_xyz = f_now_vec[:3] - baseline_vec[:3]
                    delta_mag = float(np.linalg.norm(delta_xyz))
                    delta_z = float(delta_xyz[2])
                    # One threshold for one decision, shared with _grasp_v2 via
                    # _force_empty_n(). A real grasp shows a clear delta (~10 N
                    # of contact); the ~0.6 N close-on-air transient must NOT
                    # pass, or it false-confirms and skips the descent-retry.
                    _empty_n = self._force_empty_n()
                    if delta_mag > _empty_n or delta_z < -_empty_n:
                        force_says_holding = True
                    elif delta_mag < 0.5 and abs(delta_z) < 0.5:
                        force_says_holding = False
                    logger.info(
                        "Verify force: now=(%.2f,%.2f,%.2f) baseline=(%.2f,%.2f,%.2f) "
                        "delta=(%.2f,%.2f,%.2f) |delta|=%.2f -> %s",
                        f_now_vec[0],
                        f_now_vec[1],
                        f_now_vec[2],
                        baseline_vec[0] if baseline_vec is not None else 0.0,
                        baseline_vec[1] if baseline_vec is not None else 0.0,
                        baseline_vec[2] if baseline_vec is not None else 0.0,
                        delta_xyz[0],
                        delta_xyz[1],
                        delta_xyz[2],
                        delta_mag,
                        {True: "HOLDING", False: "EMPTY", None: "ambiguous"}[
                            force_says_holding
                        ],
                    )
        except Exception as exc:
            logger.warning("Verify force read failed: %s", exc)

        # Position in the holding band
        pos_says_holding = None
        if pos is not None:
            if self.GRIPPER_EMPTY_THRESHOLD <= pos < self.GRIPPER_FULLY_CLOSED:
                pos_says_holding = True
            elif pos < self.GRIPPER_EMPTY_THRESHOLD:
                pos_says_holding = False
            logger.info(
                "Verify position: pos=%d empty<=%d closed>=%d -> %s",
                pos,
                self.GRIPPER_EMPTY_THRESHOLD,
                self.GRIPPER_FULLY_CLOSED,
                {True: "HOLDING", False: "EMPTY", None: "ambiguous"}[pos_says_holding],
            )

        # Driver flag
        flag_says_holding = gobj_sample
        if flag_says_holding is not None:
            logger.info(
                "Verify flag (driver=%s): is_object_detected=%s",
                self._gripper_type(),
                flag_says_holding,
            )

        # FlexiTac tactile
        tactile_says_holding = None
        try:
            tac = (
                getattr(self._pipeline, "_tactile", None)
                if getattr(self, "_pipeline", None) is not None
                else None
            )
            if tac is not None and tac.available():
                # min_cells=2 here (sensor default is 1): tactile-True outranks
                # a force-says-empty veto below, so a single drifted/taped cell
                # must not be able to fake a GRASPED verdict on an empty close.
                tactile_says_holding = bool(tac.any_in_contact(min_cells=2))
                snaps = [tac.snapshot(side) for side in tac.sides()]
                cells = [s.contact_cell_count for s in snaps]
                logger.info(
                    "Verify tactile: contact_cells=%s sides=%s -> %s",
                    cells,
                    [s.side for s in snaps],
                    tactile_says_holding,
                )
        except Exception as exc:
            logger.warning("Verify tactile read failed: %s", exc)

        # Trust a POSITIVE Robotiq gObj object-detect flag outright: force alone
        # reads ~empty for a light object held statically, so it must not veto a
        # confirmed hardware grasp.
        if flag_says_holding is True and pos_says_holding is False:
            # Position disagrees with the flag; ask the cameras to break the
            # tie. Vision abstains freely (the gripper occludes), so an unknown
            # answer falls through to the remaining channels.
            logger.warning(
                "Verify: flag says held but position says EMPTY (pos=%s); "
                "cross-checking with vision", pos,
            )
            still_there = self._vision_says_still_on_table(
                self._last_keypoint_label or ""
            )
            if still_there is True:
                logger.warning(
                    "Verify: NOT GRASPED -- the object is still on the table "
                    "and the jaws are open; the object-detect flag was stale"
                )
                return False
            if still_there is False:
                logger.info(
                    "Verify: GRASPED (object left its place; flag agrees)"
                )
                return True
        elif flag_says_holding is True:
            logger.info("Verify: GRASPED (Robotiq object-detect flag)")
            return True
        # A False gObj flag is NOT trusted as a veto: the flag and gripper
        # position ride the contended _send_script register path and read
        # stale/False on real grasps. Fall through to TCP force (rtde_r channel)
        # instead of hard-rejecting.
        if flag_says_holding is False:
            logger.info(
                "Verify: object-detect flag False (stale), "
                "deferring to force (force=%s pos=%s)",
                force_says_holding,
                pos_says_holding,
            )
        if force_says_holding is True:
            logger.info("Verify: GRASPED (force confirmed)")
            return True
        if tactile_says_holding is True:
            logger.info(
                "Verify: GRASPED (tactile contact, force=%s pos=%s)",
                force_says_holding,
                pos_says_holding,
            )
            return True
        if force_says_holding is False:
            logger.info(
                "Verify: NOT grasped (force says empty even though "
                "pos=%s flag=%s tactile=%s)",
                pos_says_holding,
                flag_says_holding,
                tactile_says_holding,
            )
            return False
        if pos_says_holding is True and flag_says_holding is True:
            logger.info("Verify: GRASPED (pos+flag both positive, force ambiguous)")
            return True
        if pos_says_holding is True and flag_says_holding is None:
            logger.info("Verify: GRASPED (pos positive, flag unavailable)")
            return True
        logger.info(
            "Verify: NOT grasped (force=%s pos=%s flag=%s tactile=%s)",
            force_says_holding,
            pos_says_holding,
            flag_says_holding,
            tactile_says_holding,
        )
        return False

    def _verify_holding_during_transport(self) -> bool:
        """
        Is the object still held during a transport move?

        Force-feedback first, then position check. Conservative: returns
        True unless both force and position confirm loss.
        """
        self._check_abort()
        gt = self._gripper_type()

        force_says_holding = None
        try:
            if hasattr(self.robot, "get_tcp_force"):
                ft = np.asarray(self.robot.get_tcp_force(), dtype=float)
                if ft.size >= 6:
                    baseline = getattr(self, "_pre_grasp_force", None)
                    if baseline is not None and baseline.size >= 3:
                        delta_xyz = ft[:3] - baseline[:3]
                        delta_mag = float(np.linalg.norm(delta_xyz))
                        delta_z = float(delta_xyz[2])
                        if delta_mag > 0.25 or delta_z < -0.25:
                            force_says_holding = True
                        elif delta_mag < 0.10 and abs(delta_z) < 0.10:
                            force_says_holding = False
                        logger.info(
                            "Transport force: |delta|=%.2f F_z_delta=%.2f -> %s",
                            delta_mag,
                            delta_z,
                            {True: "HOLDING", False: "LOST", None: "ambiguous"}[
                                force_says_holding
                            ],
                        )
        except Exception as exc:
            logger.warning("Transport force read failed: %s", exc)

        if gt == "robotiq_2f85":
            self._robotiq_resqueeze(speed=50, force=100, settle=0.3)

        pos = self._get_gripper_position()
        pos_says_holding = None
        if pos is not None:
            pos_says_holding = (
                pos < self.GRIPPER_FULLY_CLOSED and pos < self.GRIPPER_EMPTY_THRESHOLD
            )
            logger.info(
                "Transport position: pos=%d empty=%d closed=%d -> %s",
                pos,
                self.GRIPPER_EMPTY_THRESHOLD,
                self.GRIPPER_FULLY_CLOSED,
                {True: "HOLDING", False: "LOST", None: "ambiguous"}[pos_says_holding],
            )

        if force_says_holding is True:
            return True
        if force_says_holding is None and pos_says_holding is True:
            return True
        if force_says_holding is False:
            logger.info("Transport: object LOST (force evidence)")
            return False
        return True

"""
ScoreExecutorCore: init, orchestration, dispatch, abort, workspace.
"""

import json
import logging
import math
import os
import re
import threading
import time
import urllib.request
from typing import Dict, List, Optional

import numpy as np
from scipy.spatial.transform import Rotation

from spark_real.config import family_block
from spark_real.control import execution_recovery
from spark_real.control import primitive_timeouts
from spark_real.control.cartesian_servo import CartesianServo
from spark_real.control.executor_types import (
    BUILTIN_PRIMITIVES,
    AbortRequested,
    ExecutionResult,
    osc_backend_enabled,
)
from spark_real.control.trajectory import TrajectoryRecorder
from spark_real.control.waypoints import WaypointBuffer
from spark_real.calibration_table import load_table_plane
from spark_real.grasp_depth_memory import GraspCalibration
from spark_real.planning import bt_grammar
from spark_real.skills import registry as skill_registry

logger = logging.getLogger(__name__)

# Statements that can only ADD speed. A script containing any of them is not a
# brake, whatever else it contains.
_ACCELERATING_STATEMENTS = ("movej(", "movel(", "movep(", "movec(", "servoj(", "speedj(")
_SPEEDL_VEC_RE = re.compile(r"speedl\(\[([^\]]*)\]")

# Families whose arms are Franka Panda/FR3 (tool-Z down grasp, 0.855 m reach).
_FRANKA_FAMILIES = frozenset({"franka", "bimanual_franka"})


def is_braking_script(script: str) -> bool:
    """True if every statement in ``script`` can only reduce arm speed.

    The abort-aware _send_script wrapper lets the brake through after the
    abort flag is set. A prefix test on "stopj"/"stopl" is too narrow: the
    Cartesian servo brakes with ``speedl([0,...])`` followed by ``stopl(1.0)``.
    """
    s = script.strip()
    if not s or not ("stopj(" in s or "stopl(" in s):
        return False
    if any(tok in s for tok in _ACCELERATING_STATEMENTS):
        return False
    for m in _SPEEDL_VEC_RE.finditer(s):
        try:
            vec = [float(x) for x in m.group(1).split(",")]
        except ValueError:
            return False  # unparseable speed: assume it accelerates
        if any(abs(v) > 1e-9 for v in vec):
            return False
    return True


class ScoreExecutorCore:
    """Init, orchestration loop, dispatch, abort, and workspace logic.

    Base mixin for ScoreExecutor: __init__, the execute_score loop, action
    dispatch, abort machinery, and workspace bounds.
    """

    SAFE_HEIGHT_Z = 0.35
    APPROACH_OFFSET_Z = 0.0
    GRIPPER_OPEN_Z_OFFSET = -0.005
    # Same grasp as [2.103, -2.329, 0.059], expressed in the half-turn nearest
    # HOME_CONFIG (a parallel jaw is symmetric under 180 deg about the tool
    # axis); that value sat -177.8 deg in yaw from home and wrapped the
    # wrist-camera cable. Must match configs/ur10e_default.yaml.
    GRASP_ORIENTATION = [2.3038, 2.0802, -0.0048]
    WRIST_CAM_OFFSET_DEG = 81.0
    GRIPPER_EMPTY_THRESHOLD = 235
    GRIPPER_FULLY_CLOSED = 250
    # Release-side jaw thresholds (0=open .. 255=closed). GRIPPER_FULLY_CLOSED
    # answers "are the jaws still spread on something" AFTER a close; it says
    # nothing after an OPEN, where a released gripper reads ~0. These two are
    # the open-side pair: jaws at/near the open stop, or enough travel toward
    # open to prove the object let go. Bounds, not calibrated values; the
    # release witness logs the measured numbers so a rig run can tighten them.
    GRIPPER_RELEASED_MAX_POS = 60
    GRIPPER_RELEASE_MIN_TRAVEL = 40
    # Extra window the release witness may poll for the jaws to read open,
    # AFTER the driver's own open settle. Zero cost on a normal release (the
    # first sample already reads open).
    RELEASE_CONFIRM_TIMEOUT_S = 0.6
    # Brake decel for a URScript-capable driver that has no brake() of its own
    # (rad/s^2). UR10eDriver sizes its own from the commanded velocity; this is
    # only the last-resort literal.
    ABORT_DECEL_FALLBACK = 2.0
    MAX_GRASP_RETRIES = 3
    TABLE_Z_FLOOR = -0.276
    WORKSPACE_MIN = np.array([-1.1, -0.5, -0.27])
    WORKSPACE_MAX = np.array([-0.5, 0.7, 0.50])
    MAX_REACH = 1.15
    RECOVERY_Z_BIAS = [0.010, 0.020, 0.030]

    # Grasp / stack tuning. Overridable per-family via the `grasp:` and `stack:`
    # YAML blocks (see _apply_family_tuning); a family that omits them keeps
    # these defaults. Env vars remain as optional debug overrides.
    GRASP_DEPTH_M = 0.022            # descend this far below the perceived top
                                     # before clamping (TCP open->closed offset).
    GRASP_FORCE_EMPTY_N = 2.0        # force-verify: peak |dF| below this = empty.
                                     # gObj still leads.
    GRASP_LIFT_CHECK_M = 0.02        # confirming lift after clamp.
    GRASP_LIFT_DELTA_MIN_N = 0.5     # extra pull the confirming lift must add
                                     # for it to corroborate "holding".
    GRASP_YAW_AR_GATE = 1.8          # only do an oriented (yaw) grasp when the
                                     # object's aspect ratio exceeds this. Cubes
                                     # read ~1.1-1.5 from mask noise; real tools
                                     # ~2+ (1.4 fired 85 deg yaws on cubes).
    # Input-validity floors on the SAME oriented-grasp decision. AR cannot tell
    # a tool from a soft blob: the plushie read AR 3.14 at conf 0.24.
    GRASP_YAW_MIN_CONF = 0.45        # detection confidence floor for an OBB yaw.
    GRASP_YAW_MIN_OBB_CONF = 0.40    # OBB-axis confidence floor. Absent = unknown.
    STACK_CLEARANCE_M = 0.12         # begin place descent this far above target top.
    STACK_MAX_DESCENT_M = 0.15       # descent safety cap.
    STACK_CONTACT_FORCE_N = 5.0      # force-DELTA that counts as contact.
    STACK_HARD_STOP_DELTA_N = 25.0   # force-DELTA emergency stop.
    STACK_STEP_M = 0.004             # slow final-approach step size.
    STACK_APPROACH_GAP_M = 0.02      # fast approach stops this far above contact.
    STACK_DESCENT_VEL_FRAC = 0.10    # slow final-approach velocity fraction.

    def __init__(
        self,
        robot,
        detection_map=None,
        velocity=0.25,
        grasp_calibration: "GraspCalibration" = None,
        safe_height=None,
        pipeline=None,
        strict_placement_verify: bool = False,
    ):
        self.robot = robot
        self.detection_map = detection_map or {}
        try:
            from spark_real.control.spatial_verify import remember_targets
            remember_targets(self)
        except Exception:  # noqa: BLE001
            pass
        family = getattr(robot, "robot_family", None) or getattr(
            getattr(robot, "_robot", None), "robot_family", None
        )
        if str(family).lower() == "franka" and velocity > 0.10:
            logger.info(
                "ScoreExecutor: clamping velocity %.2f -> 0.10 "
                "for Franka FR3 (CartesianServo PD stability)",
                velocity,
            )
            velocity = 0.10
        self.velocity = velocity
        self.safe_height = safe_height or self.SAFE_HEIGHT_Z
        self._results = []
        self._abort = False
        # Abort epochs. _abort_epoch is bumped by every abort request;
        # _armed_epoch is stamped at the TASK boundary (reset_task_state).
        # execute_score clears the flag only when the two agree, so a stop
        # pressed during capture -> detect -> plan survives into the run.
        self._abort_epoch = 0
        self._armed_epoch = 0
        # Serialises bump-and-read against retract-if-unchanged, so the
        # watchdog cannot retract an operator stop that landed between its
        # own request and its retraction.
        self._abort_lock = threading.Lock()
        self._running = False
        self._wrist_servo_enabled = True
        # Wrist-camera refinement during approach/place/search; off by
        # default because the FR3 wrist RealSense is video-only. Resolved
        # from control.wrist_refine below.
        self._wrist_refine_enabled = False
        self._pipeline = pipeline
        self._recorder = None
        self._servo = CartesianServo(robot, rate_hz=125.0)
        self.waypoint_buffer = WaypointBuffer()
        # Everything scoped to ONE task lives in reset_task_state.
        self.reset_task_state()
        self._strict_placement_verify = strict_placement_verify
        self.grasp_calibration = grasp_calibration
        self._install_abort_aware_send_script()
        self._apply_family_workspace_bounds()
        self._resolve_wrist_refine_flag()
        self._apply_family_tuning()
        self._load_primitive_timeouts()

    def _apply_family_tuning(self):
        """Override grasp/stack tuning constants from the `grasp:`/`stack:` YAML
        blocks (profile.raw preferred, else <family>_default.yaml). Any
        missing key keeps the class default.
        """
        cfg = getattr(self._pipeline, "config", None) if self._pipeline else None
        if cfg is None:
            return
        family = (getattr(cfg, "robot_family", "ur10e") or "ur10e").lower()
        try:
            profile = getattr(self._pipeline, "profile", None)
            grasp = family_block(profile, family, "grasp")
            stack = family_block(profile, family, "stack")
            mapping = [
                (grasp, "depth_m", "GRASP_DEPTH_M"),
                (grasp, "force_empty_n", "GRASP_FORCE_EMPTY_N"),
                (grasp, "lift_check_m", "GRASP_LIFT_CHECK_M"),
                (grasp, "lift_delta_min_n", "GRASP_LIFT_DELTA_MIN_N"),
                (grasp, "yaw_aspect_ratio_gate", "GRASP_YAW_AR_GATE"),
                (grasp, "yaw_min_conf", "GRASP_YAW_MIN_CONF"),
                (grasp, "yaw_min_obb_conf", "GRASP_YAW_MIN_OBB_CONF"),
                (grasp, "released_max_pos", "GRIPPER_RELEASED_MAX_POS"),
                (grasp, "release_min_travel", "GRIPPER_RELEASE_MIN_TRAVEL"),
                (grasp, "release_confirm_timeout_s", "RELEASE_CONFIRM_TIMEOUT_S"),
                (grasp, "max_yaw_offset_deg", "_MAX_YAW_OFFSET_DEG"),
                (stack, "clearance_m", "STACK_CLEARANCE_M"),
                (stack, "max_descent_m", "STACK_MAX_DESCENT_M"),
                (stack, "contact_force_n", "STACK_CONTACT_FORCE_N"),
                (stack, "hard_stop_delta_n", "STACK_HARD_STOP_DELTA_N"),
                (stack, "step_m", "STACK_STEP_M"),
                (stack, "approach_gap_m", "STACK_APPROACH_GAP_M"),
                (stack, "descent_vel_frac", "STACK_DESCENT_VEL_FRAC"),
            ]
            applied = {}
            for block, key, attr in mapping:
                if isinstance(block, dict) and block.get(key) is not None:
                    setattr(self, attr, float(block[key]))
                    applied[attr] = getattr(self, attr)
            if applied:
                logger.info(
                    "ScoreExecutor grasp/stack tuning from config (%s): %s",
                    family, applied,
                )
            # Env override (quick experiment): SPARK_YAW_AR_GATE sets the oriented
            # (OBB-yaw) grasp gate; set it very high to force a plain top-down
            # grasp on objects whose aspect ratio would otherwise trip the yaw.
            _g = os.environ.get("SPARK_YAW_AR_GATE")
            if _g is not None:
                self.GRASP_YAW_AR_GATE = float(_g)
                logger.info(
                    "GRASP_YAW_AR_GATE=%.2f (SPARK_YAW_AR_GATE override)",
                    self.GRASP_YAW_AR_GATE,
                )
            # Cable limit for the oriented (OBB-yaw) grasp, in degrees.
            _y = os.environ.get("SPARK_MAX_YAW_OFFSET_DEG") or getattr(
                self, "_MAX_YAW_OFFSET_DEG", None
            )
            if _y is not None:
                self.MAX_YAW_OFFSET = math.radians(float(_y))
                logger.info(
                    "MAX_YAW_OFFSET=%.0f deg (oriented-grasp wrist-3 cable limit)",
                    float(_y),
                )
        except Exception as exc:
            logger.warning(
                "Could not load grasp/stack tuning: %s; keeping defaults", exc
            )

    def _resolve_wrist_refine_flag(self):
        """Read control.wrist_refine from the profile (or family YAML).

        Mirrors how _apply_family_workspace_bounds sources its values;
        defaults to False on any missing key or read error.
        """
        cfg = getattr(self._pipeline, "config", None) if self._pipeline else None
        if cfg is None:
            return
        family = (getattr(cfg, "robot_family", "ur10e") or "ur10e").lower()
        try:
            profile = getattr(self._pipeline, "profile", None)
            control = family_block(profile, family, "control")
            if "wrist_refine" in control:
                self._wrist_refine_enabled = bool(control["wrist_refine"])
                logger.info(
                    "ScoreExecutor wrist_refine=%s (control.wrist_refine)",
                    self._wrist_refine_enabled,
                )
        except Exception as exc:
            logger.warning(
                "Could not load control.wrist_refine: %s; "
                "keeping wrist refine disabled",
                exc,
            )
        # Env override: SPARK_WRIST_REFINE=1 forces wrist refinement on.
        _env = os.environ.get("SPARK_WRIST_REFINE")
        if _env is not None:
            self._wrist_refine_enabled = _env not in ("0", "false", "False", "")
            logger.info(
                "ScoreExecutor wrist_refine=%s (SPARK_WRIST_REFINE override)",
                self._wrist_refine_enabled,
            )

    def _load_primitive_timeouts(self):
        """Resolve the per-primitive wall-clock budgets.

        Starts from the defaults and overlays any timeouts: block in the
        resolved RobotProfile (preferred) or family YAML. Re-run from
        pipeline_init once _profile is wired so the profile override wins.
        """
        cfg = getattr(self._pipeline, "config", None) if self._pipeline else None
        family = (
            (getattr(cfg, "robot_family", None) or "").lower() if cfg is not None
            else ""
        )
        profile = getattr(self, "_profile", None)
        try:
            self._primitive_timeouts = primitive_timeouts.load_timeouts(
                profile=profile, family=family or None
            )
            logger.info(
                "ScoreExecutor primitive timeouts loaded (%d entries, default=%.1fs)",
                len(self._primitive_timeouts),
                self._primitive_timeouts.get(
                    "default", primitive_timeouts.DEFAULT_TIMEOUT_S
                ),
            )
        except Exception as exc:
            logger.warning(
                "Could not load primitive timeouts (%s); using paper defaults", exc
            )
            self._primitive_timeouts = primitive_timeouts.default_timeouts()
            self._primitive_timeouts["default"] = primitive_timeouts.DEFAULT_TIMEOUT_S

    def _apply_family_workspace_bounds(self):
        """Replace UR-default workspace bounds from the resolved profile.

        Sources configs/<family>_default.yaml plus any per-machine overlay.
        Falls back to class defaults if no pipeline/config is present, and
        re-reads the family YAML directly when the profile is missing/empty.
        """
        cfg = getattr(self._pipeline, "config", None) if self._pipeline else None
        if cfg is None:
            return
        family = (getattr(cfg, "robot_family", "ur10e") or "ur10e").lower()
        if family == "ur10e":
            # UR10e keeps its box, reach, table floor, and grasp rotvec in
            # the family YAML control: block.
            self._apply_ur10e_control_bounds()
            return
        try:
            # Source the workspace dict and raw table_z_floor; for franka the
            # profile mirrors the family YAML (workspace block + table_z_floor).
            profile = getattr(self, "_profile", None)
            raw = family_block(profile, family)
            cfg_label = (
                "RobotProfile" if getattr(profile, "raw", None) else f"{family}_default.yaml"
            )
            # A per-arm executor (bimanual) reads its own workspace_<arm> box;
            # arm None keeps the single-arm top-level workspace block.
            arm = getattr(self, "arm", None)
            ws = raw.get(f"workspace_{arm}" if arm else "workspace") or {}
            table_z_floor_raw = raw.get("table_z_floor")
            if not ws:
                return
            ws_min = np.array(
                [
                    float(ws.get("x_min", -0.85)),
                    float(ws.get("y_min", -0.85)),
                    float(ws.get("z_min", -0.05)),
                ],
                dtype=float,
            )
            ws_max = np.array(
                [
                    float(ws.get("x_max", 0.85)),
                    float(ws.get("y_max", 0.85)),
                    float(ws.get("z_max", 1.20)),
                ],
                dtype=float,
            )
            self.WORKSPACE_MIN = ws_min
            self.WORKSPACE_MAX = ws_max
            self.MAX_REACH = float(
                ws.get("reach_max", 0.855 if family in _FRANKA_FAMILIES else 1.15)
            )
            if table_z_floor_raw is not None:
                self.TABLE_Z_FLOOR = float(table_z_floor_raw)
            logger.info(
                "ScoreExecutor workspace overridden from %s: "
                "X=[%.2f,%.2f] Y=[%.2f,%.2f] Z=[%.2f,%.2f] reach=%.2f "
                "table_z_floor=%.3f",
                cfg_label,
                ws_min[0],
                ws_max[0],
                ws_min[1],
                ws_max[1],
                ws_min[2],
                ws_max[2],
                self.MAX_REACH,
                self.TABLE_Z_FLOOR,
            )
            if family in _FRANKA_FAMILIES:
                # control.grasp_orientation when the YAML declares it, else
                # the Franka tool-Z-down rotvec (pi, 0, 0).
                grasp = (raw.get("control") or {}).get("grasp_orientation")
                self.GRASP_ORIENTATION = (
                    [float(v) for v in grasp] if grasp else [math.pi, 0.0, 0.0]
                )
                logger.info(
                    "ScoreExecutor GRASP_ORIENTATION set to %s (Franka tool-Z down)",
                    self.GRASP_ORIENTATION,
                )
        except Exception as exc:
            logger.warning(
                "Could not load family workspace bounds: %s; " "keeping UR defaults",
                exc,
            )

    def _apply_ur10e_control_bounds(self):
        """Load the UR10e executor box from the control: block.

        Sources the resolved profile (or ur10e_default.yaml directly);
        every key defaults to the class constant. Reads control.* (the
        executor/SafeRobot box), not the workspace: block, which feeds the
        CBF _family_safety_bounds path the live UR10e branch never reaches.
        """
        try:
            control = family_block(getattr(self, "_profile", None), "ur10e", "control")
            if not control:
                return
            grasp = control.get("grasp_orientation")
            if grasp is not None:
                self.GRASP_ORIENTATION = [float(v) for v in grasp]
            ws_min = control.get("workspace_min")
            if ws_min is not None:
                self.WORKSPACE_MIN = np.array([float(v) for v in ws_min], dtype=float)
            ws_max = control.get("workspace_max")
            if ws_max is not None:
                self.WORKSPACE_MAX = np.array([float(v) for v in ws_max], dtype=float)
            if control.get("max_reach") is not None:
                self.MAX_REACH = float(control["max_reach"])
            # Table floor precedence: calibration file > YAML control.table_z_floor
            # > class default. The YAML value / class const stay as the FALLBACK.
            if control.get("table_z_floor") is not None:
                self.TABLE_Z_FLOOR = float(control["table_z_floor"])
                floor_src = "config: %.4f" % self.TABLE_Z_FLOOR
            else:
                floor_src = "class default: %.4f" % self.TABLE_Z_FLOOR
            cal = load_table_plane("ur10e")
            if cal is not None:
                surface_z = float(cal["surface_z"])
                # Margin is added so the floor sits a few mm ABOVE (less negative
                # than) the measured surface. Prefer the cal file's own margin.
                margin = float(
                    cal.get(
                        "safety_margin_m",
                        control.get("table_safety_margin_m", 0.003),
                    )
                )
                cal_floor = surface_z + margin
                # Sanity clamp: a cal recorded with a bad TCP reading must not
                # LOWER the safety floor below the config/class fallback by more
                # than 2 cm (that is how a bumped calibration drives the gripper
                # into the table). Raising the floor is always accepted.
                if cal_floor < self.TABLE_Z_FLOOR - 0.02:
                    logger.warning(
                        "ScoreExecutor REJECTING table calibration floor %.4f: "
                        "more than 2cm below fallback %.4f (suspect bad cal); "
                        "keeping fallback",
                        cal_floor,
                        self.TABLE_Z_FLOOR,
                    )
                else:
                    self.TABLE_Z_FLOOR = cal_floor
                    floor_src = "calibration: surface=%.4f +%dmm -> %.4f" % (
                        surface_z,
                        round(margin * 1000),
                        self.TABLE_Z_FLOOR,
                    )
            logger.info("ScoreExecutor table floor from %s", floor_src)
            logger.info(
                "ScoreExecutor UR10e control bounds from config: "
                "X=[%.2f,%.2f] Y=[%.2f,%.2f] Z=[%.2f,%.2f] reach=%.2f "
                "table_z_floor=%.3f grasp_rotvec=%s",
                self.WORKSPACE_MIN[0],
                self.WORKSPACE_MAX[0],
                self.WORKSPACE_MIN[1],
                self.WORKSPACE_MAX[1],
                self.WORKSPACE_MIN[2],
                self.WORKSPACE_MAX[2],
                self.MAX_REACH,
                self.TABLE_Z_FLOOR,
                self.GRASP_ORIENTATION,
            )
        except Exception as exc:
            logger.warning(
                "Could not load UR10e control bounds: %s; " "keeping class defaults",
                exc,
            )

    def _install_abort_aware_send_script(self):
        """
        Wrap robot._send_script (UR-only) to respect the abort flag.
        """
        executor_ref = self

        def make_wrapper(orig):
            def abort_aware_send(script):
                if (
                    executor_ref._running
                    and executor_ref._abort
                    and not is_braking_script(script)
                ):
                    raise AbortRequested()
                return orig(script)

            abort_aware_send._spark_abort_wrapped = True
            return abort_aware_send

        target = self.robot
        seen = set()
        while target is not None and id(target) not in seen:
            seen.add(id(target))
            if hasattr(target, "_send_script"):
                original_send = target._send_script
                if not getattr(original_send, "_spark_abort_wrapped", False):
                    try:
                        target._send_script = make_wrapper(original_send)
                    except AttributeError:
                        pass
            target = getattr(target, "_robot", None)

    def update_detections(self, detection_map):
        self.detection_map = detection_map
        # Snapshot the approved scene for the end-of-task spatial rung here,
        # not at verify time: mid-run re-detections mutate detection_map in
        # place.
        try:
            from spark_real.control.spatial_verify import remember_targets
            remember_targets(self)
        except Exception:  # noqa: BLE001 - memory is an aid, never a blocker
            pass

    def _join_redetect(self, timeout: float = 5.0):
        """Bounded wait for a live post-release re-detect thread.

        Called before the executor starts any new detect so two detection
        passes never run concurrently against the stateful SAM3 predictor
        or mutate detection_map from two threads. Bounded so a wedged detect
        cannot hang a run.
        """
        thread = getattr(self, "_redetect_thread", None)
        if thread is None or not thread.is_alive():
            return
        try:
            thread.join(timeout=timeout)
            if thread.is_alive():
                logger.warning(
                    "_join_redetect: re-detect thread still running after "
                    "%.1fs; proceeding (perception lock serializes SAM3)",
                    timeout,
                )
            else:
                self._redetect_thread = None
        except Exception as exc:
            logger.warning("_join_redetect failed: %s", exc)

    # Fields whose lifetime is ONE task. Attribute -> factory for a fresh value.
    # Anything a run writes and a later run reads belongs here.
    _TASK_SCOPED_STATE = {
        "_holding": bool,
        "_placed_labels": set,
        "_last_pick_label": str,
        "_last_place_label": str,
        "_last_keypoint_label": lambda: None,
        "_last_action_failed": bool,
        "_grasp_retry_count": dict,
        "_destination_labels": set,
        # Depth memory written by _approach_target and read by the grasp's
        # calibration record.
        "_grasp_perception_target_z": lambda: None,
        # Actual TCP z at the moment the jaws closed (grasp v2 / recovery
        # re-grasp). Read by the release give-back. Reset per pick in
        # _approach_target, per task here.
        "_actual_grasp_tcp_z": lambda: None,
        "_last_grasp_target_width": lambda: None,
        "_active_grasp_label": lambda: None,
        "_active_grasp_orient": lambda: None,
        "_active_grasp_strategy": lambda: None,
        # Release look-ahead written per-action by _run_actions/_run_branch
        # and read by the place transport's deep-dip side choice
        # (ReleaseMixin._dip_side_orient).
        "_pending_release_tilt": lambda: None,
    }

    def reset_task_state(self):
        """Drop every field scoped to ONE task.

        The executor is a server-lifetime singleton: one instance runs every
        task. Called from __init__ and from success_verifier.reset_run_state
        (the task boundary). NOT from execute_score: the closed loop calls
        that once per pass and _placed_labels must survive the passes of one
        task.
        """
        for attr, factory in self._TASK_SCOPED_STATE.items():
            setattr(self, attr, factory())
        # The task boundary is the ONE place an abort is legitimately
        # forgotten. Arming here lets execute_score distinguish a stale flag
        # from a stop pressed while this task was being planned.
        self._armed_epoch = getattr(self, "_abort_epoch", 0)
        self._abort = False
        # A post-release re-detection thread mutates detection_map in place,
        # so an unjoined one would rewrite the NEXT task's detections. Daemon
        # threads cannot be killed; wait briefly and say so if still running.
        thread = getattr(self, "_redetect_thread", None)
        if thread is not None:
            try:
                thread.join(timeout=2.0)
                if thread.is_alive():
                    logger.warning(
                        "reset_task_state: re-detect thread from the previous "
                        "task is still running; it may overwrite detections"
                    )
            except Exception as exc:
                logger.warning("reset_task_state: re-detect join failed: %s", exc)
            self._redetect_thread = None
        # An abandoned motion phrase must not replay into the next task.
        try:
            self.waypoint_buffer.clear()
        except Exception as exc:
            logger.debug("reset_task_state: waypoint buffer clear failed: %s", exc)

    def note_abort_requested(self) -> int:
        """
        Raise the abort flag, advance the epoch, and return the new epoch.

        Every stop path must come through here rather than assigning
        ``_abort`` directly, or the request is invisible to execute_score's
        arming check. The epoch goes up FIRST, so a reader that sees
        ``_abort`` True has already seen the epoch move.

        The returned epoch identifies THIS request, so the primitive-timeout
        watchdog can retract its own abort without stepping on an operator
        stop that arrived in the same window.
        """
        with self._abort_lock:
            self._abort_epoch += 1
            self._abort = True
            return self._abort_epoch

    def retract_abort(self, epoch: int) -> bool:
        """
        Withdraw abort request ``epoch``. False unless it is the ONLY
        outstanding one.

        For the primitive-timeout watchdog, whose abort is a transient it must
        undo so recovery can run. TWO conditions, both needed:

        * nothing NEWER has landed (``_abort_epoch == epoch``): an operator
          stop raised while the primitive unwound;
        * nothing OLDER is still outstanding (``epoch - 1 == _armed_epoch``):
          an operator stop raised BEFORE the watchdog fired.

        Together: this request is the only thing between the task's arming
        point and now, so undoing it restores exactly the pre-timeout state.
        """
        with self._abort_lock:
            if self._abort_epoch != epoch:
                logger.warning(
                    "retract_abort(%d) refused: a newer stop (epoch %d) is "
                    "standing; leaving the abort in place",
                    epoch,
                    self._abort_epoch,
                )
                return False
            if epoch - 1 != self._armed_epoch:
                logger.warning(
                    "retract_abort(%d) refused: an earlier stop is still "
                    "outstanding (armed at epoch %d); leaving the abort in "
                    "place",
                    epoch,
                    self._armed_epoch,
                )
                return False
            self._armed_epoch = epoch
            self._abort = False
            return True

    def abort(self, epoch: Optional[int] = None) -> bool:
        """Request a stop AND brake the arm. True if the brake reached the arm.

        Setting the flag only stops this process from issuing new commands.
        URScript motion is fire-and-forget, so an unbraked abort leaves the arm
        running to its target while the executor unwinds. Brake here, at the
        single entry point every caller uses (/api/stop, /api/abort, the
        primitive-timeout watchdog).

        ``epoch`` is for a caller that has ALREADY called
        note_abort_requested() and is holding its epoch so it can retract
        exactly that request later (the watchdog). Passing it skips the bump
        so one stop is one epoch.
        """
        if epoch is None:
            self.note_abort_requested()
        self._servo.abort()
        braked = self._stop_robot()
        if braked:
            logger.warning("Abort requested; arm braked")
        else:
            logger.critical(
                "Abort requested but THE ARM WAS NOT BRAKED -- it may still be "
                "moving. Use the physical E-stop."
            )
        return braked

    def _stop_robot(self) -> bool:
        """Brake the arm. True only if a stop actually reached the driver.

        Order matters. The first branch is the UR escalation (URScript stopj on
        the primary socket, then a reopened socket, then Dashboard); it is the
        only thing that stops a fire-and-forget movej. rtde_c.stopJ is a no-op
        against a dead control script, so it must not be the first and only try.
        """
        emergency = getattr(self.robot, "emergency_stop", None)
        if callable(emergency):
            try:
                result = emergency()
                if isinstance(result, dict) and result.get("braked"):
                    return True
                logger.error("emergency_stop did not reach the arm: %s", result)
            except Exception as exc:  # noqa: BLE001 - fall through to the rest
                logger.warning("emergency_stop raised: %s", exc)
        # Any other URScript-capable driver: emit the brake directly. The
        # abort-aware _send_script wrapper lets braking scripts through.
        if getattr(self.robot, "SUPPORTS_URSCRIPT", False):
            send = getattr(self.robot, "_send_script", None)
            if callable(send):
                try:
                    if bool(send("stopj(%.2f)" % self.ABORT_DECEL_FALLBACK)):
                        return True
                except Exception as exc:  # noqa: BLE001
                    logger.warning("stopj send failed: %s", exc)
        # Non-UR drivers (Franka, bimanual, OSC proxy): these DO brake, and
        # unlike the UR they have no URScript channel. Try them all, in order,
        # and do not stop at the first one that merely fails to raise.
        braked = False
        for name in ("stop_motion", "stop", "servo_stop"):
            fn = getattr(self.robot, name, None)
            if callable(fn):
                try:
                    # A driver that reports failure (False) has not braked.
                    braked = fn() is not False or braked
                except Exception as exc:  # noqa: BLE001
                    logger.warning("%s failed: %s", name, exc)
        return braked

    def _check_abort(self):
        """
        Raise AbortRequested if stop has been requested.
        """
        if self._abort:
            self._stop_robot()
            raise AbortRequested()

    def _abort_sleep(self, duration: float, tick: float = 0.05):
        """
        Abort-aware sleep. Polls the abort flag every tick seconds.
        """
        t_end = time.time() + duration
        while True:
            self._check_abort()
            remaining = t_end - time.time()
            if remaining <= 0:
                return
            time.sleep(min(tick, remaining))

    def _check_workspace(self, pos):
        if np.linalg.norm(pos) > self.MAX_REACH:
            return False
        return bool(
            np.all(pos >= self.WORKSPACE_MIN) and np.all(pos <= self.WORKSPACE_MAX)
        )

    def _get_current_position(self):
        if osc_backend_enabled():
            try:
                with urllib.request.urlopen(
                    "http://127.0.0.1:9009/state",
                    timeout=1.5,
                ) as r:
                    st = json.loads(r.read().decode())
                xyz = st.get("tcp_xyz")
                if xyz is not None:
                    return np.asarray(xyz[:3], dtype=float)
            except Exception as exc:
                logger.warning(
                    "OSC /state read failed (%s); falling back to "
                    "franky get_tcp_pose if available",
                    exc,
                )
        if hasattr(self.robot, "get_tcp_pose"):
            pose = self.robot.get_tcp_pose()
            if isinstance(pose, np.ndarray) and pose.shape == (4, 4):
                return pose[:3, 3].copy()
            return np.array(pose[:3])
        return np.zeros(3)

    # Orchestration

    def execute_score(self, score: dict) -> List[ExecutionResult]:
        self._results = []
        # NOT an unconditional `self._abort = False`: a stop pressed after this
        # task was armed (during capture -> detect -> plan) must survive into
        # the run rather than be wiped by it.
        self._abort = self._abort_epoch != self._armed_epoch
        if self._abort:
            logger.warning(
                "execute_score: a stop was requested after this task was "
                "armed (epoch %d != %d); honouring it instead of starting",
                self._abort_epoch,
                self._armed_epoch,
            )
        self._running = True
        self._recorder = TrajectoryRecorder(self.robot)

        tree = score.get("tree", score)
        actions = self._flatten_tree(tree)
        # Recovery budget is scoped to ONE execute_score call: a closed-loop
        # pass gets its own six re-attempts, but a single pass can never spend
        # more however deeply the plan nests retry inside fallback.
        self._recovery_attempts_used = 0
        # Post-task verification and the replay path both want a plain list of
        # leaves: every leaf of every branch.
        leaf_actions = self._leaf_actions(actions)
        logger.info(
            "Executing %d actions (%d leaves)", len(actions), len(leaf_actions)
        )

        aborted = False
        try:
            try:
                self._run_actions(actions)
            except AbortRequested:
                logger.warning("Execution aborted by user")
                self._results.append(
                    ExecutionResult(
                        action_type="abort", success=False, message="Aborted by user"
                    )
                )
                aborted = True
                self._stop_robot()

            if self._pipeline is not None and not aborted:
                try:
                    self._run_post_task_verification(score, leaf_actions)
                except AbortRequested:
                    logger.warning("Post-task verification aborted by user")
                    aborted = True

            try:
                self._recorder.save()
            except Exception as e:
                logger.warning("Failed to save trajectory: %s", e)

            return self._results
        finally:
            self._running = False

    _LEGATO_BLENDABLE = {"move_relative", "move_to_keypoint"}

    def _is_legato_blendable(self, action: dict) -> bool:
        """
        Return True if this action can be a legato note (vs barrier).
        """
        if action is None:
            return False
        atype = action.get("type", action.get("name", ""))
        if atype not in self._LEGATO_BLENDABLE:
            return False
        if atype == "move_to_keypoint" and not self._holding:
            return False
        return True

    @staticmethod
    def _peek_release_tilt(actions):
        """Params of the first ``release`` leaf in ``actions``, else None.

        Depth-first through selector/retry markers, in execution order. Lets
        the PLACE TRANSPORT know what dip the release will ask for: a tilt
        past the positive pitch budget (the wrist-camera clearance, 40 deg) is
        only achievable when the approach yaw already put the dip on the
        negative-pitch side (90 deg budget), see ReleaseMixin._dip_side_orient.
        """
        for node in actions or []:
            if not isinstance(node, dict):
                continue
            if node.get("_bt_selector"):
                for branch in node.get("branches") or []:
                    hit = ScoreExecutorCore._peek_release_tilt(branch)
                    if hit is not None:
                        return hit
                continue
            if node.get("_bt_retry"):
                hit = ScoreExecutorCore._peek_release_tilt(node.get("branch"))
                if hit is not None:
                    return hit
                continue
            if node.get("type", node.get("name", "")) == "release":
                return node.get("params", {}) or {}
        return None

    @staticmethod
    def _peek_place_label(actions):
        """Label of the next PLACE destination in ``actions``, else None.

        Sibling of _peek_release_tilt: it lets the grasp choose between the
        two equivalent grasp yaws knowing the destination. A parallel gripper
        closes identically at yaw and yaw+180, but the choice fixes which way
        the held object points, and therefore the yaw the place will need
        (which can land past the +-100 deg wrist limit and seat the tool
        END-FOR-END).
        """
        for node in actions or []:
            if not isinstance(node, dict):
                continue
            if node.get("_bt_selector"):
                for branch in node.get("branches") or []:
                    hit = ScoreExecutorCore._peek_place_label(branch)
                    if hit is not None:
                        return hit
                continue
            if node.get("_bt_retry"):
                hit = ScoreExecutorCore._peek_place_label(node.get("branch"))
                if hit is not None:
                    return hit
                continue
            params = node.get("params", {}) or {}
            kind = node.get("type", node.get("name", ""))
            if kind == "place_in_slot":
                return params.get("container_label")
            if kind == "move_to_keypoint":
                # The transport that delivers to the place is the FIRST
                # keypoint move after the grasp; the grasp's own approach has
                # already been consumed by the time this look-ahead runs.
                return params.get("keypoint_label")
        return None

    def _run_actions(self, actions):
        i = 0
        while i < len(actions):
            action = actions[i]
            # Refresh the release look-ahead every step so it can neither go
            # stale (a consumed release must stop steering later transports)
            # nor be missed (legato phrases and recovery both re-enter here).
            self._pending_release_tilt = self._peek_release_tilt(actions[i:])
            # Skip the current node: the NEXT destination is what matters.
            self._pending_place_label = self._peek_place_label(actions[i + 1:])
            if self._abort:
                logger.warning("Execution aborted by user")
                self._results.append(
                    ExecutionResult(
                        action_type="abort", success=False, message="Aborted by user"
                    )
                )
                break

            # Control-flow markers are handled first: they are not
            # dispatchable leaves, they own their children's failure handling,
            # and the flat path's recovery must not double up on theirs.
            if action.get("_bt_selector") or action.get("_bt_retry"):
                if action.get("_bt_selector"):
                    result = self._run_selector(action)
                else:
                    result = self._run_retry(action)
                self._results.append(result)
                self._last_action_failed = not bool(result.success)
                if not result.success:
                    i = self._skip_failed_cycle(actions, i)
                    continue
                i += 1
                continue

            # Legato look-ahead
            if self._is_legato_blendable(action):
                run = [action]
                j = i + 1
                while j < len(actions) and self._is_legato_blendable(actions[j]):
                    run.append(actions[j])
                    j += 1
                if len(run) >= 2:
                    logger.info(
                        "[%d-%d/%d] legato phrase: %s",
                        i + 1,
                        j,
                        len(actions),
                        [a.get("type") for a in run],
                    )
                    ok = self._execute_legato_phrase(run, i, len(actions))
                    if ok:
                        i = j
                        continue

            action_type = action.get("type", action.get("name", "unknown"))
            params = action.get("params", {})
            logger.info("[%d/%d] %s %s", i + 1, len(actions), action_type, params)

            if action_type == "move_to_keypoint":
                kp = params.get("keypoint_label", "")
                label = f"move_to {kp}" if kp else action_type
            elif action_type == "move_relative":
                label = "move_rel"
            else:
                label = action_type
            self._recorder.set_action_label(label)
            self._recorder.mark_transition(label)
            self._recorder.record()

            # Re-detect before approaching when previous action failed
            if (
                action_type in ("move_to_keypoint", "grasp_se3")
                and i > 0
                and not self._holding
                and self._pipeline is not None
                and getattr(self, "_last_action_failed", False)
            ):
                kp_label = params.get("keypoint_label", "")
                if kp_label and kp_label not in self._placed_labels:
                    self._join_redetect()
                    self._redetect_single(kp_label)

            if action_type == "move_to_keypoint" and not self._holding:
                self._last_pick_label = params.get("keypoint_label", "")
            elif action_type == "grasp_se3" and not self._holding:
                self._last_pick_label = params.get("keypoint_label", "")

            result = self._dispatch_action(action_type, params)
            self._results.append(result)
            self._last_action_failed = not bool(result.success)

            if result.success and action_type == "release":
                if (
                    self._last_pick_label
                    and self._pipeline is not None
                    and (
                        getattr(self, "strict_placement_verify", False)
                        # The per-release two-anchor verdict + rebind also
                        # rides this capture (verification.release_verdict).
                        or execution_recovery.release_verdict_enabled(self)
                    )
                ):
                    try:
                        execution_recovery.verify_placement(
                            self, self._last_pick_label, self._last_place_label
                        )
                    except Exception as e:
                        logger.warning("Post-release verification error: %s", e)
                if self._last_pick_label:
                    self._placed_labels.add(self._last_pick_label)
                    logger.info("Marked '%s' as placed", self._last_pick_label)
                    self._last_pick_label = ""
                if self._pipeline is not None:
                    remaining = actions[i + 1 :]
                    has_next_pick = any(
                        a.get("type") in ("move_to_keypoint", "grasp_se3")
                        for a in remaining
                    )
                    if has_next_pick:
                        # Never two re-detect threads at once.
                        self._join_redetect()
                        self._redetect_thread = threading.Thread(
                            target=self._redetect_all,
                            args=(actions, i),
                            daemon=True,
                        )
                        self._redetect_thread.start()
                        logger.info(
                            "Inter-grasp: skipping home, going direct to next pick"
                        )

            if not result.success:
                logger.warning("Action %d failed: %s", i + 1, result.message)
                msg_lower = (result.message or "").lower()
                if any(
                    k in msg_lower
                    for k in ("reflex", "singular", "discontinuity", "control")
                ):
                    try:
                        self._snapshot_failure_state(action_type, params, result)
                    except Exception as _snap_e:
                        logger.warning("Failure snapshot skipped: %s", _snap_e)
                recovery = self._attempt_recovery(
                    action_type, params, result, actions, i
                )
                if recovery is not None and recovery.success:
                    self._results[-1] = recovery
                    i += 1
                    continue
                logger.warning("Action failed, resetting")
                self._check_abort()
                still_holding = False
                if self._holding:
                    try:
                        still_holding = self._verify_grasp()
                        logger.info(
                            "Recovery: force-verify says %s; " "%s open the gripper",
                            "HOLDING" if still_holding else "EMPTY",
                            "skipping" if still_holding else "going to",
                        )
                    except Exception as _vexc:
                        logger.warning("Recovery verify failed: %s", _vexc)
                if not still_holding:
                    self.robot.open_gripper()
                    self._abort_sleep(0.3)
                    self._holding = False
                cur = self._get_current_position()
                if cur[2] < 0.45:
                    cur[2] = min(cur[2] + 0.10, 0.45)
                    self._move_to(cur, self.current_orientation())
                # Post-failure J5 escape from singularity
                try:
                    q_now = np.asarray(self.robot.get_joint_positions(), dtype=float)
                    if abs(q_now[4]) < 0.5:
                        q_escape = q_now[:7].copy()
                        q_escape[4] = -0.8
                        logger.info(
                            "Recovery: J5=%.3f near singular; "
                            "escape-nudge to -0.8 before next cycle",
                            float(q_now[4]),
                        )
                        self.robot.move_to_joint_config(
                            q_escape.tolist(), velocity=self.velocity * 0.5
                        )
                        try:
                            self._wait_stationary_after_servo()
                        except Exception:
                            pass
                except Exception as _e:
                    logger.debug("Recovery J5 escape skipped: %s", _e)
                # Retry-once on grasp failure
                if action_type in ("grasp", "grasp_se3"):
                    kp = params.get("keypoint_label", "")
                    if not hasattr(self, "_grasp_retry_count"):
                        self._grasp_retry_count = {}
                    retries = self._grasp_retry_count.get(kp, 0) if kp else 0
                    if kp and retries < 1:
                        self._grasp_retry_count[kp] = retries + 1
                        logger.info(
                            "[retry] grasp '%s' failed; retrying once with "
                            "fresh detection (attempt %d)",
                            kp,
                            retries + 2,
                        )
                        continue
                    if kp:
                        logger.info(
                            "[retry] grasp '%s' failed twice; giving up, "
                            "skipping cycle",
                            kp,
                        )
                    while i + 1 < len(actions):
                        next_type = actions[i + 1].get("type", "")
                        if next_type in ("grasp", "grasp_se3"):
                            break
                        i += 1
                        self._results.append(
                            ExecutionResult(
                                action_type=next_type,
                                success=False,
                                message="Skipped (prior grasp failed)",
                            )
                        )
                    logger.info("Skipped to next grasp cycle")

            i += 1

    _LEAF_TYPES = {
        "move_to_keypoint",
        "grasp",
        "release",
        "move_relative",
        "open_drawer",
        "push_object",
        "turn_knob",
        "wait",
        "pour",
    }

    def _flatten_tree(self, node: dict) -> List[dict]:
        """Flatten a score tree into the action list ``_run_actions`` walks.

        Sequences collapse. A ``selector``/``fallback`` or ``retry`` node does
        NOT: it becomes ONE opaque marker action carrying its already-flattened
        branches, otherwise "try A, else B" would flatten to "do A, then do B".
        """
        node_type = node.get("type", "")

        if node_type in bt_grammar.SELECTOR_TYPES:
            branches = []
            for child in node.get("children") or []:
                branch = self._flatten_tree(child)
                if branch:
                    branches.append(branch)
            if len(branches) > bt_grammar.MAX_SELECTOR_BRANCHES:
                logger.warning(
                    "selector has %d branches; running the first %d",
                    len(branches),
                    bt_grammar.MAX_SELECTOR_BRANCHES,
                )
                branches = branches[: bt_grammar.MAX_SELECTOR_BRANCHES]
            return [
                {
                    "type": "selector",
                    "_bt_selector": True,
                    "branches": branches,
                    "params": dict(node.get("params") or {}),
                }
            ]

        if node_type in bt_grammar.RETRY_TYPES:
            body = []
            for child in node.get("children") or []:
                body.extend(self._flatten_tree(child))
            attempts = bt_grammar.clamp_retry_attempts(
                (node.get("params") or {}).get("max_attempts")
            )
            return [
                {
                    "type": "retry",
                    "_bt_retry": True,
                    "branch": body,
                    "max_attempts": attempts,
                    "params": dict(node.get("params") or {}),
                }
            ]

        if node_type == "sequence" or (
            "children" in node and node_type not in self._LEAF_TYPES
        ):
            actions = []
            for child in node.get("children", []):
                actions.extend(self._flatten_tree(child))
            return actions
        return [node]

    def _leaf_actions(self, actions: List[dict]) -> List[dict]:
        """Expand control-flow markers into the leaves they can run.

        Downstream consumers (post-task verification, the pick-cycle replay)
        reason about "which labels did this plan touch", not about control
        flow.
        """
        out: List[dict] = []
        for action in actions:
            if action.get("_bt_selector"):
                for branch in action.get("branches") or []:
                    out.extend(self._leaf_actions(branch))
            elif action.get("_bt_retry"):
                out.extend(self._leaf_actions(action.get("branch") or []))
            else:
                out.append(action)
        return out

    # ---- bounded recovery -------------------------------------------------

    def _spend_recovery_budget(self, what: str) -> bool:
        """Charge one recovery re-attempt against the per-score budget.

        Returns False once the budget is gone, which makes the enclosing node
        fail cleanly. The per-node caps alone are not enough: retry(3) nested
        inside fallback(3) multiplies to nine attempts, and nine blind
        re-approaches on a real arm is not recovery, it is thrashing.
        """
        used = getattr(self, "_recovery_attempts_used", 0)
        if used >= bt_grammar.MAX_RECOVERY_ATTEMPTS_PER_SCORE:
            logger.warning(
                "recovery budget exhausted (%d/%d); refusing %s",
                used,
                bt_grammar.MAX_RECOVERY_ATTEMPTS_PER_SCORE,
                what,
            )
            return False
        self._recovery_attempts_used = used + 1
        logger.info(
            "recovery attempt %d/%d: %s",
            self._recovery_attempts_used,
            bt_grammar.MAX_RECOVERY_ATTEMPTS_PER_SCORE,
            what,
        )
        return True

    @staticmethod
    def _first_label_of(actions: List[dict]) -> str:
        for action in actions:
            label = (action.get("params") or {}).get("keypoint_label")
            if label:
                return str(label)
        return ""

    def _reset_before_recovery(self, next_actions: List[dict]) -> None:
        """Put the arm in a known state before a recovery branch runs.

        Mirrors the reset the flat path already does on a failed action: drop
        nothing that is still gripped, clear the workspace vertically, return
        to home, and re-ground the label the next branch is about to reach for.
        Every step is best-effort: a reset that raises must not mask the
        failure that triggered it.
        """
        self._check_abort()

        still_holding = False
        if self._holding:
            try:
                still_holding = self._verify_grasp()
            except Exception as exc:
                logger.warning("Recovery reset: grasp verify failed: %s", exc)
        if not still_holding:
            try:
                self.robot.open_gripper()
                self._abort_sleep(0.3)
            except AbortRequested:
                raise
            except Exception as exc:
                logger.warning("Recovery reset: open_gripper failed: %s", exc)
            self._holding = False

        try:
            cur = self._get_current_position()
            if cur[2] < 0.45:
                cur[2] = min(cur[2] + 0.10, 0.45)
                # A reset LIFT is a translation; re-commanding GRASP_ORIENTATION
                # here would unwind the task's yaw at v=2.67 rad/s.
                self._move_to(cur, self.current_orientation())
        except AbortRequested:
            raise
        except Exception as exc:
            logger.warning("Recovery reset: lift failed: %s", exc)

        go_home = getattr(self.robot, "go_home", None)
        if callable(go_home):
            try:
                go_home()
            except AbortRequested:
                raise
            except Exception as exc:
                logger.warning("Recovery reset: go_home failed: %s", exc)

        if self._pipeline is not None:
            label = self._first_label_of(self._leaf_actions(next_actions))
            if label and label not in self._placed_labels:
                try:
                    self._redetect_single(label)
                except AbortRequested:
                    raise
                except Exception as exc:
                    logger.warning("Recovery reset: re-detect failed: %s", exc)

    def _skip_failed_cycle(self, actions: List[dict], index: int) -> int:
        """Index of the next action worth running after a control-flow node
        exhausted its recovery.

        Once the acquire has genuinely failed, the transport and the place
        that follow it would move an empty gripper to a container and open it.
        Skip to the next action that can start a NEW pick cycle (one whose
        leaves contain a grasp), which for a single-object plan is the end,
        and for a multi-object plan is the next object.
        """
        j = index + 1
        while j < len(actions):
            leaves = self._leaf_actions([actions[j]])
            if any(a.get("type") in ("grasp", "grasp_se3") for a in leaves):
                return j
            self._results.append(
                ExecutionResult(
                    action_type=actions[j].get("type", "unknown"),
                    success=False,
                    message="Skipped (recovery for the preceding step failed)",
                )
            )
            j += 1
        return j

    def _run_branch(self, actions: List[dict]) -> bool:
        """Run one branch under STRICT sequence semantics.

        Strict means: stop at the first failure and report it. The enclosing
        selector's next branch, or the enclosing retry's next attempt, IS the
        reaction to the failure.
        """
        for k, action in enumerate(actions):
            self._check_abort()
            # Same release look-ahead as _run_actions: a recovery re-run must
            # steer its approach yaw by the same upcoming dip the first try did.
            self._pending_release_tilt = self._peek_release_tilt(actions[k:])
            self._pending_place_label = self._peek_place_label(actions[k + 1:])
            if action.get("_bt_selector"):
                result = self._run_selector(action)
            elif action.get("_bt_retry"):
                result = self._run_retry(action)
            else:
                result = self._run_leaf(action)
            self._results.append(result)
            self._last_action_failed = not bool(result.success)
            if not result.success:
                return False
        return True

    def _run_leaf(self, action: dict) -> ExecutionResult:
        action_type = action.get("type", action.get("name", "unknown"))
        params = action.get("params", {}) or {}
        logger.info("[branch] %s %s", action_type, params)
        try:
            self._recorder.set_action_label(action_type)
            self._recorder.mark_transition(action_type)
            self._recorder.record()
        except Exception as exc:
            logger.debug("Recorder skipped for %s: %s", action_type, exc)

        if action_type in ("move_to_keypoint", "grasp_se3") and not self._holding:
            self._last_pick_label = params.get("keypoint_label", "")

        result = self._dispatch_action(action_type, params)
        if result.success and action_type == "release" and self._last_pick_label:
            self._placed_labels.add(self._last_pick_label)
            self._last_pick_label = ""
        return result

    def _run_selector(self, marker: dict) -> ExecutionResult:
        branches = marker.get("branches") or []
        total = len(branches)
        if total == 0:
            return ExecutionResult(
                action_type="selector", success=True, message="empty selector"
            )
        for idx, branch in enumerate(branches):
            if idx > 0:
                if not self._spend_recovery_budget(
                    f"selector branch {idx + 1}/{total}"
                ):
                    return ExecutionResult(
                        action_type="selector",
                        success=False,
                        message=(
                            f"All {total} selector branches failed "
                            f"(recovery budget exhausted after {idx})"
                        ),
                    )
                self._reset_before_recovery(branch)
            if self._run_branch(branch):
                return ExecutionResult(
                    action_type="selector",
                    success=True,
                    message=f"branch {idx + 1}/{total} succeeded",
                )
            logger.warning("selector branch %d/%d failed", idx + 1, total)
        return ExecutionResult(
            action_type="selector",
            success=False,
            message=f"All {total} selector branches failed",
        )

    def _run_retry(self, marker: dict) -> ExecutionResult:
        branch = marker.get("branch") or []
        if not branch:
            return ExecutionResult(
                action_type="retry", success=True, message="empty retry"
            )
        attempts = bt_grammar.clamp_retry_attempts(marker.get("max_attempts"))
        for k in range(attempts):
            if k > 0:
                if not self._spend_recovery_budget(f"retry attempt {k + 1}/{attempts}"):
                    return ExecutionResult(
                        action_type="retry",
                        success=False,
                        message=(
                            f"retry gave up after {k}/{attempts} attempts "
                            "(recovery budget exhausted)"
                        ),
                    )
                self._reset_before_recovery(branch)
            if self._run_branch(branch):
                return ExecutionResult(
                    action_type="retry",
                    success=True,
                    message=f"succeeded on attempt {k + 1}/{attempts}",
                )
            logger.warning("retry attempt %d/%d failed", k + 1, attempts)
        return ExecutionResult(
            action_type="retry",
            success=False,
            message=f"all {attempts} retry attempts failed",
        )

    # Grammar slot aliases accepted everywhere the canonical name is expected.
    # The canonical name wins if both are present. Applied once at the dispatch
    # choke point so both builtin and skill primitives see normalized params.
    _PARAM_ALIASES = {
        "label": "keypoint_label",
        "target": "target_label",
    }

    def _normalize_params(self, params: dict) -> dict:
        if not params:
            return params
        if not any(a in params for a in self._PARAM_ALIASES):
            return params
        norm = dict(params)
        for alias, canonical in self._PARAM_ALIASES.items():
            if alias in norm and canonical not in norm:
                norm[canonical] = norm[alias]
        return norm

    def _dispatch_action(self, action_type: str, params: dict) -> ExecutionResult:
        """Normalize grammar aliases, then run under the primitive's budget.

        On a budget overrun the watchdog aborts the motion and returns a
        failed result so the caller's recovery path handles it like any
        failed post-condition.
        """
        params = self._normalize_params(params)
        budget = primitive_timeouts.timeout_for(
            getattr(self, "_primitive_timeouts", None) or {}, action_type
        )
        return primitive_timeouts.run_with_timeout(
            self,
            action_type,
            budget,
            lambda: self._dispatch_action_body(action_type, params),
        )

    _last_safety_reconcile_ts = 0.0

    def _reconcile_after_protective_stop(self, action_type: str, params: dict):
        """One world-model check at the first action boundary after an
        auto-cleared protective stop.

        The unlock path (ur10e_driver._check_safety_and_recover) restores
        CONTROL, but a protective stop means a COLLISION happened: the held
        object may have been knocked from the jaws (ask the jaws) and the
        involved detection may have moved (re-bind).
        """
        ts = float(getattr(self.robot, "last_protective_recovery_ts", 0.0) or 0.0)
        if ts <= self._last_safety_reconcile_ts:
            return
        self._last_safety_reconcile_ts = ts
        logger.warning(
            "[post-stop] reconciling before '%s': the last protective stop "
            "was auto-cleared, so the collision's effects are checked now",
            action_type,
        )
        # 1. Still holding? A drop here fails the CURRENT action so recovery
        #    re-grasps instead of transporting empty jaws to the target.
        if getattr(self, "_holding", False):
            try:
                if not self._grip_intact():
                    logger.warning(
                        "[post-stop] the jaws are EMPTY; the collision cost "
                        "us the object -- marking not-holding so recovery "
                        "re-grasps"
                    )
                    self._holding = False
            except Exception:  # noqa: BLE001 - unknown grip stays assumed-held
                pass
        # 2. Re-bind the label this action is about, if it names one. The
        #    collision may have moved it (or been CAUSED by it moving).
        label = (params or {}).get("keypoint_label") or (params or {}).get(
            "container_label"
        )
        if label:
            try:
                from spark_real.control.execution_recovery import (
                    rebind_before_approach,
                )

                rebind_before_approach(self, label)
            except Exception:  # noqa: BLE001
                pass

    def _dispatch_action_body(
        self, action_type: str, params: dict
    ) -> ExecutionResult:
        t0 = time.time()
        try:
            self._reconcile_after_protective_stop(action_type, params)
            builtin = set(BUILTIN_PRIMITIVES) | {"action"}
            if action_type not in builtin and skill_registry.get(action_type) is not None:
                return skill_registry.dispatch(action_type, self, params)

            if action_type in ("move_to_keypoint", "action"):
                name = params.get("name", action_type)
                dispatch_map = {
                    "move_to_keypoint": self._move_to_keypoint,
                    "grasp": self._grasp,
                    "release": self._release,
                    "move_relative": self._move_relative,
                }
                handler = dispatch_map.get(name)
                if handler:
                    return handler(params, t0)
                if action_type == "action" and name != action_type:
                    return self._dispatch_action_body(name, params)
                return self._move_to_keypoint(params, t0)
            elif action_type == "grasp":
                return self._grasp(params, t0)
            elif action_type == "release":
                return self._release(params, t0)
            elif action_type == "move_relative":
                return self._move_relative(params, t0)
            elif action_type == "wait":
                return self._wait(params, t0)
            else:
                return ExecutionResult(
                    action_type=action_type,
                    success=False,
                    message=f"Unknown action type: {action_type}",
                    duration=time.time() - t0,
                )
        except AbortRequested:
            raise
        except Exception as e:
            return ExecutionResult(
                action_type=action_type,
                success=False,
                message=str(e),
                duration=time.time() - t0,
            )

    def _move_relative(self, params: dict, t0: float) -> ExecutionResult:
        self._check_abort()
        dx = params.get("dx", 0.0)
        dy = params.get("dy", 0.0)
        dz = params.get("dz", 0.0)
        current = self._get_current_position()
        self._move_to(current + np.array([dx, dy, dz]), self.current_orientation())
        return ExecutionResult(
            action_type="move_relative",
            success=True,
            message=f"Moved by ({dx}, {dy}, {dz})",
            duration=time.time() - t0,
        )

    def _wait(self, params: dict, t0: float) -> ExecutionResult:
        # duration defaults to 0.5s when the plan omits it.
        duration = params.get("duration", 0.5)
        self._abort_sleep(duration)
        return ExecutionResult(
            action_type="wait",
            success=True,
            message=f"Waited {duration}s",
            duration=time.time() - t0,
        )

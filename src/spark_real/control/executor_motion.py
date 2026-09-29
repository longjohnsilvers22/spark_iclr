"""MotionMixin: movement primitives for ScoreExecutor.

Handles _move_to, _servo_to, _approach_target, _transport_to,
_safe_clearance_z, _oriented_grasp, and low-level motion helpers. The
IK-backed joint-space helpers (_movej_via_ik, _movej_to_pose, legato
phrasing) live in executor_ik.IkMixin.
"""

import logging
import os
import time
from typing import Optional

import cv2
import numpy as np
from PIL import Image as PILImage
from scipy.spatial.transform import Rotation

from spark_real.config import family_block
from spark_real.calibration import compute_wrist_camera_extrinsic
from spark_real.calibration_table import load_table_plane
from spark_real.control import success_verifier
from spark_real.control.execution_recovery import grasp_target_in_container
from spark_real.control.executor_types import (
    AbortRequested,
    ExecutionResult,
    maybe_osc_move_linear,
)
from spark_real.control.grasp_strategy import (
    compose_yaw,
    resolve_grasp_orientation,
)
from spark_real.control.release_height import resolve_release_z
from spark_real.control.success_predicates import container_obb_xy, obb_contains
from spark_real.control.waypoints import (
    UR_PROGRESS_REGISTER,
    Waypoint,
    WaypointBuffer,
    decode_blend_progress,
    joint_move_seconds,
)
from spark_real.perception.mask_geometry import resolve_slot_direction
from spark_real.perception.wrist_ray import intersect_ray_plane, pixel_ray_base
from spark_real.recording.settings import resolve_settings

try:
    import torch
except ImportError:
    torch = None

logger = logging.getLogger(__name__)


class _StillnessTracker:
    """Motion / stillness detector on a FIXED sample cadence.

    The poll rate and the motion test are independent: displacement is
    evaluated over its own ``sample_s`` window. ``speed_eps`` is 3 mm/s, i.e.
    0.3 mm of travel per 100 ms sample; over a 25 ms window it would be
    0.075 mm, inside TCP measurement noise, and a spuriously latched
    ``ever_moved`` lets the stillness exit fire early.
    """

    __slots__ = ("_sample_s", "_eps", "_need", "_prev", "_prev_t", "ever_moved", "still")

    def __init__(self, tcp, now, sample_s=0.1, speed_eps=0.003, still_samples=3):
        self._sample_s = float(sample_s)
        self._eps = float(speed_eps)
        self._need = int(still_samples)
        self._prev = np.asarray(tcp, dtype=float).copy()
        self._prev_t = float(now)
        self.ever_moved = False
        self.still = 0

    def update(self, tcp, now) -> None:
        """Fold one poll in; only whole sample windows change the verdict."""
        dt = float(now) - self._prev_t
        if dt < self._sample_s:
            return
        step_speed = float(np.linalg.norm(np.asarray(tcp, dtype=float) - self._prev)) / dt
        if step_speed > self._eps:
            self.ever_moved = True
            self.still = 0
        else:
            self.still += 1
        self._prev = np.asarray(tcp, dtype=float).copy()
        self._prev_t = float(now)

    def stopped(self) -> bool:
        """Enough consecutive still windows to call the arm stopped."""
        return self.still >= self._need


class MotionMixin:
    """
    Movement primitives mixed into ScoreExecutor.
    """

    # Hard wrist-3 cable limit, not a tuning knob: the wrist-camera USB cable
    # will not take more travel than this. Config: grasp.max_yaw_offset_deg
    # (env SPARK_MAX_YAW_OFFSET_DEG). Enforced on the COMPOSED rotation by
    # grasp_strategy.check_yaw_limit, not just on the requested angle.
    # 100, not 90: a two-jaw gripper is symmetric mod 180, and at a 90 cap the
    # nearer equivalent of a yaw near +-90 (e.g. +92.9 for -87.1) is refused
    # by a few degrees, so the wrist takes the LONGER path. 10 deg of headroom.
    MAX_YAW_OFFSET = np.deg2rad(100)
    CLEARANCE_MARGIN = 0.12
    OBSTACLE_XY_RADIUS = 0.12

    # How much of the place descent the Cartesian servo keeps (m). Everything
    # above this is flown as a movej/blend row; the servo owns every millimetre
    # where the held object can contact the container.
    PLACE_SERVO_STANDOFF_M = 0.02
    # Extra clearance above the first possible rim contact, covering the
    # held-extent estimate's own error. See _place_standoff_m.
    PLACE_CONTACT_MARGIN_M = 0.015
    # Ceiling on the slow phase. The servo crawls near convergence, so an
    # unbounded standoff would turn a bad held-extent number into a 10 s place.
    PLACE_SERVO_STANDOFF_MAX_M = 0.12

    WRIST_REFINE_MODES = ("off", "depth", "ray_plane")
    WRIST_REFINE_MODE_DEFAULT = "ray_plane"
    # A plane prior nearer the camera than this is rejected: the ray grazes
    # the plane and the intersection runs away laterally.
    WRIST_RAY_MIN_STANDOFF_M = 0.03

    # How far OUTSIDE a container's own planar footprint the TCP may sit and
    # still count as having arrived over it. The ONLY tolerance the place path
    # judges arrival with (see _place_arrival); not the servo's pos_threshold
    # (3 mm), which says nothing about whether the object is over the container.
    # +2 cm mirrors the verifier's containment margin
    # (success_predicates.EvalConfig.inside_xy_margin_m = -0.02): anything
    # refused here is at least 4 cm outside a region that could have passed
    # verify, so the gate is never the reason a good task refuses to let go.
    PLACE_ARRIVAL_XY_MARGIN_M = 0.02

    def _demo_mode(self) -> bool:
        """
        True when ``recording.demo_mode`` asks motion to stay on a path that
        emits a per-timestep COMMANDED velocity.

        Read through RecordingSettings (one place for the knob's name, default
        and aliases). Cached per executor. Missing pipeline or config resolves
        to the default, False.
        """
        cached = getattr(self, "_demo_mode_cached", None)
        if cached is None:
            config = getattr(self._pipeline, "config", None)
            try:
                cached = bool(resolve_settings(config).demo_mode)
            except Exception:  # noqa: BLE001 - a config quirk must not stop motion
                cached = False
            self._demo_mode_cached = cached
        return cached

    def current_orientation(self, fallback=None):
        """The orientation the arm is actually holding, or `fallback`.

        Translations must not re-pick a wrist angle: passing the fixed
        GRASP_ORIENTATION into a pure move UNWINDS whatever yaw the task had
        established.

        `obs.get("tcp_pose") or obs.get(...)` raises on a numpy array, so the
        None comparisons are explicit.
        """
        try:
            obs = self.robot.get_observation()
            tcp = obs.get("tcp_pose")
            if tcp is None:
                tcp = obs.get("tcp_pos")
            if tcp is not None and len(tcp) >= 6:
                return list(np.asarray(tcp[3:6], dtype=float))
        except Exception as exc:  # noqa: BLE001
            logger.warning("current_orientation: read failed (%s)", exc)
        return fallback if fallback is not None else list(self.GRASP_ORIENTATION)

    def _nearest_symmetric_yaw(self, yaw: float) -> float:
        """
        Pick yaw or yaw+-pi, whichever needs the least wrist travel.

        A two-jaw gripper grasps identically at yaw and yaw+pi, so <=90 deg
        of travel always suffices. Candidates stay inside MAX_YAW_OFFSET; on
        any read failure the canonical yaw passes through unchanged.
        """
        try:
            tcp = np.asarray(self.robot.get_tcp_pose(), dtype=float)
            R_cur = Rotation.from_rotvec(tcp[3:6])
            R_base = Rotation.from_rotvec(np.asarray(self.GRASP_ORIENTATION, dtype=float))
            rel = (R_cur * R_base.inv()).as_matrix()
            cur_yaw = float(np.arctan2(rel[1, 0], rel[0, 0]))
        except Exception:
            return yaw
        best = yaw
        for cand in (yaw - np.pi, yaw + np.pi):
            if abs(cand) <= self.MAX_YAW_OFFSET and abs(cand - cur_yaw) < abs(best - cur_yaw):
                best = cand
        if best != yaw:
            logger.info(
                "[yaw-symmetry] %.1f deg -> %.1f deg (current %.1f, travel "
                "%.1f instead of %.1f deg)",
                np.rad2deg(yaw),
                np.rad2deg(best),
                np.rad2deg(cur_yaw),
                np.rad2deg(abs(best - cur_yaw)),
                np.rad2deg(abs(yaw - cur_yaw)),
            )
        return best

    def _move_to_keypoint(self, params: dict, t0: float) -> ExecutionResult:
        self._check_abort()
        label = params.get("keypoint_label", "")
        self._last_keypoint_label = label
        # offset_z default 0.0: move_to_keypoint takes the gripper TO the
        # keypoint so the following grasp closes on it (the grasp descent-retry
        # only spans ~7.5cm). The planner emits explicit offsets (0 picks,
        # 0.04+ places); this only governs plans/BTs that omit the field.
        offset = np.array(
            [
                params.get("offset_x", 0.0),
                params.get("offset_y", 0.0),
                params.get("offset_z", 0.0),
            ]
        )

        det = self.detection_map.get(label)
        if det is None:
            return ExecutionResult(
                action_type="move_to_keypoint",
                success=False,
                message=f"Object '{label}' not found in detections",
                duration=time.time() - t0,
            )

        # Hard guard for pick approaches: if this object already sits inside
        # the tray / container, skip the pick entirely (it is done). Only
        # applies when not holding so placement moves into the container are
        # never blocked.
        if not self._holding:
            try:
                _in = grasp_target_in_container(self, label)
            except Exception:
                _in = None
            # Only skip the pick if this run placed the object into the
            # container (a re-pick guard for stacking recovery). An object that
            # STARTED inside a container is not in _placed_labels and must be
            # picked normally.
            if _in is not None and label in self._placed_labels:
                self._placed_labels.add(label)
                logger.info(
                    "move_to_keypoint '%s': already in container " "'%s', skipping pick approach",
                    label,
                    _in,
                )
                return ExecutionResult(
                    action_type="move_to_keypoint",
                    success=True,
                    message=f"'{label}' already in container '{_in}', " f"skipping",
                    duration=time.time() - t0,
                )

        pos = np.array(det["position_3d"])
        # RoboInter override, default OFF. pipeline_execution.execute()
        # publishes these keys only when a spatial annotation exists for this
        # node AND it survived the bounded-correction guard (<= 8 cm for a
        # contact point, <= 20 cm for a placement, XY only; Z is always the
        # perceived value). Absent key -> untouched behaviour. Source-agnostic:
        # inline planner block or out-of-band provider point, both routed
        # through robointer_gate.apply_to_detection_map.
        override = det.get("robointer_placement_xyz" if self._holding else "robointer_contact_xyz")
        if override is not None:
            corrected = np.array(override, dtype=float)
            logger.info(
                "move_to_keypoint '%s': RoboInter %s correction %.1f cm " "(%s -> %s)",
                label,
                "placement" if self._holding else "contact",
                float(np.linalg.norm(corrected[:2] - pos[:2])) * 100.0,
                np.round(pos, 3).tolist(),
                np.round(corrected, 3).tolist(),
            )
            pos = corrected
        if pos[2] < self.TABLE_Z_FLOOR:
            logger.warning("Detection '%s' Z=%.3f below floor, clamping", label, pos[2])
            pos[2] = self.TABLE_Z_FLOOR
        target = pos + offset

        if self._holding:
            # PLACE. Recompute the Z from the container's MEASURED rim and
            # interior floor plus the held object's overhang; the plan's
            # offset_z survives as an upper bound, and an unmeasurable
            # container leaves it untouched (see control/release_height.py).
            # XY is never touched here.
            target[2] = self._place_release_z(float(target[2]), det, params)
            delivered = self._transport_to(
                target, target_detection=det, target_label=label, params=params
            )
            if not self._holding:
                return ExecutionResult(
                    action_type="move_to_keypoint",
                    success=False,
                    message=f"Object lost during transport to '{label}'",
                    duration=time.time() - t0,
                )
            if not delivered:
                # Still holding, but not over the target. A failed action:
                # executor_core runs recovery, and the reset path does NOT open
                # the gripper while it is still holding.
                record = getattr(self, "_place_arrival_record", None) or {}
                return ExecutionResult(
                    action_type="move_to_keypoint",
                    success=False,
                    message=(
                        f"Did not arrive over '{label}': "
                        f"{record.get('detail', 'transport stalled short')}"
                    ),
                    duration=time.time() - t0,
                )
        else:
            self._approach_target(target, detection=det, params=params)

        return ExecutionResult(
            action_type="move_to_keypoint",
            success=True,
            message=f"Moved to '{label}' at {target.tolist()}",
            duration=time.time() - t0,
        )

    def _place_release_z(self, plan_z: float, container, params: dict) -> float:
        """Resolve the Z a place transport flies to; see control/release_height.

        Binds executor state (held detection, table plane, grasp Z offset) to
        the pure resolver and logs the verdict. Every branch returns a number;
        an unmeasurable container returns ``plan_z`` unchanged.
        """
        held_label = getattr(self, "_active_grasp_label", "") or ""
        held = self.detection_map.get(held_label) if held_label else None
        strict = bool(params.get("strict_offset_z", False))
        # Effective grasp-z offset for held_drop. The static constant assumes
        # the TCP closed at perceived_top + GRIPPER_OPEN_Z_OFFSET; the real
        # grasp descends further (SPARK_GRASP_DEPTH_M, contact guard, learned
        # per-label offsets, recovery Z bias), so gripping N cm lower means the
        # object hangs N cm LESS below the TCP. Behind place.release_giveback
        # (default off); plan-strict and plan-no-geometry bypass the drop
        # model either way.
        gz = float(self.GRIPPER_OPEN_Z_OFFSET)
        if self._release_giveback_enabled():
            actual = getattr(self, "_actual_grasp_tcp_z", None)
            ptz = getattr(self, "_grasp_perception_target_z", None)
            if actual is not None and ptz is not None:
                gz = float(actual) - float(ptz)
                logger.info(
                    "Release give-back: effective grasp_z_offset=%.3f "
                    "(actual close TCP %.3f - perceived top %.3f; constant "
                    "was %.3f)",
                    gz,
                    float(actual),
                    float(ptz),
                    float(self.GRIPPER_OPEN_Z_OFFSET),
                )
        try:
            out = resolve_release_z(
                plan_z=plan_z,
                container=container,
                held=held,
                table_z=float(self.TABLE_Z_FLOOR),
                strict=strict,
                grasp_z_offset=gz,
            )
        except Exception as exc:  # noqa: BLE001 - a height quirk must not stop a place
            logger.warning(
                "Release height: could not resolve (%s); keeping the plan's " "z=%.3f",
                exc,
                plan_z,
            )
            return float(plan_z)
        if abs(out.z - plan_z) < 1e-6:
            logger.info("Release height: %s (unchanged)", out.describe())
        else:
            logger.info(
                "Release height: plan z=%.3f -> %.3f (%+.1f cm) %s",
                plan_z,
                out.z,
                (out.z - plan_z) * 100.0,
                out.describe(),
            )
        self._last_release_height = out
        return float(out.z)

    def _release_giveback_enabled(self) -> bool:
        """place.release_giveback in the family YAML (default OFF = the static
        GRIPPER_OPEN_Z_OFFSET). SPARK_RELEASE_GIVEBACK env overrides."""
        env = os.environ.get("SPARK_RELEASE_GIVEBACK")
        if env is not None:
            return env.strip() not in ("", "0", "false", "no")
        try:
            profile = getattr(self._pipeline, "profile", None)
            raw = getattr(profile, "raw", None) or {}
            return bool((raw.get("place") or {}).get("release_giveback", False))
        except Exception:  # noqa: BLE001
            return False

    def _oriented_grasp(self, yaw_offset: float = 0.0) -> list:
        """Post-rotate GRASP_ORIENTATION by Rz(yaw) about world Z.

        The closing direction ends up perpendicular to the OBB major axis.
        """
        clamped = np.clip(yaw_offset, -self.MAX_YAW_OFFSET, self.MAX_YAW_OFFSET)
        if abs(clamped - yaw_offset) > 0.01:
            logger.warning(
                "_oriented_grasp: yaw clamped from %.1f to %.1f deg",
                np.rad2deg(yaw_offset),
                np.rad2deg(clamped),
            )
        logger.info("_oriented_grasp: yaw_offset=%.1f deg", np.rad2deg(clamped))
        base = Rotation.from_rotvec(self.GRASP_ORIENTATION)
        yaw = Rotation.from_euler("z", clamped)
        return (yaw * base).as_rotvec().tolist()

    def _safe_clearance_z(self, start: np.ndarray, end: np.ndarray) -> float:
        """
        Safety lift 15 cm above the object/end Z.
        """
        base_z = end[2] + 0.15
        if not self.detection_map:
            return base_z
        max_obstacle_z = base_z
        for label, det in self.detection_map.items():
            pos = det.get("position_3d")
            if pos is None:
                continue
            obs = np.array(pos)
            path_dir = end[:2] - start[:2]
            path_len = np.linalg.norm(path_dir)
            if path_len < 0.01:
                continue
            path_unit = path_dir / path_len
            to_obs = obs[:2] - start[:2]
            proj = np.clip(np.dot(to_obs, path_unit), 0, path_len)
            closest = start[:2] + proj * path_unit
            dist_xy = np.linalg.norm(obs[:2] - closest)
            if dist_xy < self.OBSTACLE_XY_RADIUS:
                obstacle_top = obs[2] + self.CLEARANCE_MARGIN
                if obstacle_top > max_obstacle_z:
                    max_obstacle_z = obstacle_top
                    logger.info(
                        "Clearance: '%s' at z=%.3f, clearing at z=%.3f",
                        label,
                        obs[2],
                        obstacle_top,
                    )
        return max(max_obstacle_z, base_z)

    def _control_config(self) -> dict:
        """The resolved `control:` block (profile.raw preferred, else family YAML).

        Returns {} on any missing pipeline/config/file.
        """
        cfg = getattr(self._pipeline, "config", None) if self._pipeline else None
        if cfg is None:
            return {}
        family = (getattr(cfg, "robot_family", "ur10e") or "ur10e").lower()
        return family_block(getattr(self._pipeline, "profile", None), family, "control")

    def _read_wrist_refine_mode_config(self) -> Optional[str]:
        """control.wrist_refine_mode from the profile, else the family YAML."""
        return self._control_config().get("wrist_refine_mode")

    def _wrist_refine_mode(self) -> str:
        """Resolve the wrist refinement mode: off | depth | ray_plane.

        ray_plane (default) needs RGB only, so it works with the wrist D435i
        opened color-only (streaming color+depth+IMU wedges the USB event
        ring on this rig). depth uses mask-median-depth backprojection. Cached
        per executor; SPARK_WRIST_REFINE_MODE overrides.
        """
        cached = getattr(self, "_wrist_refine_mode_cached", None)
        if cached is not None:
            return cached
        # _wrist_refine_mode_cfg is set by executor_core; fall back to reading
        # it here so this works standalone.
        mode = getattr(self, "_wrist_refine_mode_cfg", None)
        if mode is None:
            mode = self._read_wrist_refine_mode_config()
        mode = os.environ.get("SPARK_WRIST_REFINE_MODE") or mode
        mode = str(mode or self.WRIST_REFINE_MODE_DEFAULT).strip().lower()
        if mode not in self.WRIST_REFINE_MODES:
            logger.warning(
                "Unknown wrist_refine_mode %r; using %s",
                mode,
                self.WRIST_REFINE_MODE_DEFAULT,
            )
            mode = self.WRIST_REFINE_MODE_DEFAULT
        self._wrist_refine_mode_cached = mode
        logger.info("Wrist refine mode: %s", mode)
        return mode

    def _wrist_plane_z(
        self,
        coarse_target: np.ndarray,
        cam_z: float,
        z_plane: Optional[float] = None,
        tcp_z: Optional[float] = None,
    ) -> tuple:
        """Pick the horizontal plane the wrist ray is intersected with.

        Priority: explicit object-Z prior > the coarse target's own Z (the
        Kinect Z is reliable here; the wrist is wanted for lateral fixes) >
        the calibrated table surface > TABLE_Z_FLOOR (itself config-fed via
        control.table_z_floor).

        A prior is only taken if it clears WRIST_RAY_MIN_STANDOFF_M below BOTH
        the camera (a grazing ray runs away laterally) and the TCP (a "target"
        at the gripper's own height is not an object: search_keypoint passes
        its TCP waypoint as the coarse target, and the wrist cam sits above
        the TCP, so the camera test alone would wave it through).

        Returns (z, source).
        """
        coarse = np.asarray(coarse_target, dtype=float)
        for z, src in ((z_plane, "object_prior"), (coarse[2], "coarse_target")):
            if z is None:
                continue
            z = float(z)
            if cam_z - z <= self.WRIST_RAY_MIN_STANDOFF_M:
                continue
            if tcp_z is not None and tcp_z - z <= self.WRIST_RAY_MIN_STANDOFF_M:
                continue
            return z, src
        cal = load_table_plane(self._robot_family())
        if cal is not None:
            return float(cal["surface_z"]), "table_cal"
        return float(self.TABLE_Z_FLOOR), "table_z_floor"

    def _wrist_pos_from_depth(self, rs_cal, depth, mask, cx_det, cy_det, label):
        """Depth path: median depth inside the mask, backprojected."""
        if depth is None:
            logger.info(
                "Wrist refine: no depth stream (camera is color-only); "
                "use control.wrist_refine_mode: ray_plane"
            )
            return None, None
        mask_depths = depth[mask > 0]
        valid = mask_depths[(mask_depths > 0.01) & (mask_depths < 1.0)]
        if len(valid) < 5:
            logger.info("Wrist refine: no valid depth for '%s'", label)
            return None, None
        d_val = float(np.median(valid))
        x_cam = (cx_det - rs_cal.cx) * d_val / rs_cal.fx
        y_cam = (cy_det - rs_cal.cy) * d_val / rs_cal.fy
        pos = rs_cal.rotation_matrix @ np.array([x_cam, y_cam, d_val]) + rs_cal.position
        return pos, "depth=%.3f" % d_val

    def _wrist_pos_from_ray(
        self, rs_cal, cx_det, cy_det, coarse_target, z_plane, label, tcp_z=None
    ):
        """RGB-only path: mask-centroid ray x horizontal plane."""
        origin, direction = pixel_ray_base(cx_det, cy_det, rs_cal, rs_cal.extrinsic)
        plane_z, src = self._wrist_plane_z(coarse_target, float(origin[2]), z_plane, tcp_z)
        pos = intersect_ray_plane(origin, direction, plane_z)
        if pos is None:
            logger.info(
                "Wrist refine: ray for '%s' misses plane z=%.3f (%s) -- "
                "parallel or pointing away",
                label,
                plane_z,
                src,
            )
            return None, None
        return pos, "plane_z=%.3f (%s)" % (plane_z, src)

    def _refine_with_wrist(
        self,
        label: str,
        coarse_target: np.ndarray,
        z_plane: Optional[float] = None,
    ) -> Optional[dict]:
        """Capture wrist image, run SAM3, compute OBB orientation in tool frame.

        Returns a dict with refined position_3d, orientation_angle,
        aspect_ratio and the source mode, or None on failure. Gated off by
        default via control.wrist_refine; when off this returns None and never
        touches the camera. control.wrist_refine_mode picks how the 3D point
        is recovered (see _wrist_refine_mode). z_plane is the caller's
        object-height prior for ray_plane.
        """
        if not self._wrist_refine_enabled:
            return None
        if self._pipeline is None:
            return None
        mode = self._wrist_refine_mode()
        if mode == "off":
            return None
        rs = self._pipeline._realsense
        rs_cal = self._pipeline._realsense_cal
        perception = self._pipeline._perception
        if rs is None or rs_cal is None or perception is None:
            return None

        try:
            if torch is None:
                return None

            rgb, depth = rs.read()
            if rgb is None:
                return None

            tcp_pose = self._servo._get_tcp_pose()
            rs_cal.extrinsic = compute_wrist_camera_extrinsic(
                tcp_pose=tcp_pose,
                tool_offset=self._pipeline._wrist_tool_offset,
            )

            sam3 = perception._sam3
            if sam3 is None:
                return None
            pil_img = PILImage.fromarray(rgb)
            state = sam3.set_image(pil_img)
            state = sam3.set_text_prompt(prompt=label, state=state)
            masks = state.get("masks", torch.tensor([]))
            scores = state.get("scores", torch.tensor([]))
            if masks.numel() == 0:
                logger.info("Wrist refine: '%s' not found", label)
                return None

            best_idx = int(scores.argmax())
            mask = masks[best_idx].cpu().numpy().squeeze()
            ys, xs = np.where(mask > 0)
            if len(xs) < 20:
                return None

            contours, _ = cv2.findContours(
                mask.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
            )
            wrist_angle = 0.0
            aspect_ratio = 1.0
            if contours:
                largest = max(contours, key=cv2.contourArea)
                if len(largest) >= 5:
                    rect = cv2.minAreaRect(largest)
                    box = cv2.boxPoints(rect)
                    s1 = box[1] - box[0]
                    s2 = box[2] - box[1]
                    l1, l2 = np.linalg.norm(s1), np.linalg.norm(s2)
                    major = s1 if l1 >= l2 else s2
                    long, short = max(l1, l2), min(l1, l2)
                    if short > 0:
                        aspect_ratio = float(long / short)
                    wrist_angle = float(np.arctan2(major[1], major[0]))

            cx_det = float(xs.mean())
            cy_det = float(ys.mean())
            if mode == "depth":
                pos_world, detail = self._wrist_pos_from_depth(
                    rs_cal, depth, mask, cx_det, cy_det, label
                )
            else:
                pos_world, detail = self._wrist_pos_from_ray(
                    rs_cal,
                    cx_det,
                    cy_det,
                    coarse_target,
                    z_plane,
                    label,
                    tcp_z=float(tcp_pose[2]),
                )
            if pos_world is None:
                return None

            logger.info(
                "Wrist refine '%s' [%s]: img_angle=%.1f deg, ar=%.1f, %s, "
                "pos=(%.3f, %.3f, %.3f)",
                label,
                mode,
                np.rad2deg(wrist_angle),
                aspect_ratio,
                detail,
                pos_world[0],
                pos_world[1],
                pos_world[2],
            )
            return {
                "position_3d": pos_world,
                "orientation_angle": wrist_angle,
                "aspect_ratio": aspect_ratio,
                "source": mode,
            }
        except Exception as e:
            logger.warning("Wrist refine failed: %s", e)
            return None

    def _approach_target(self, target: np.ndarray, detection: dict = None, params: dict = None):
        """
        Approach target: lift, position above, wrist refine, servo descend.

        `params` are the plan node's params; only grasp_strategy /
        grasp_yaw_deg are read. Omitting them is the auto route.
        """
        self._check_abort()
        if not self._holding:
            self._ensure_jaws_open("pre-approach")
        # Orientation source is the planner's `grasp_strategy` when it named
        # one, otherwise the auto route (AR gate + input-validity floors).
        orient, strategy = resolve_grasp_orientation(
            params or {}, detection, self, context="approach"
        )
        self._active_grasp_strategy = strategy
        self._active_grasp_label = self._last_keypoint_label or ""
        adjusted = target.copy()
        _gc_label = self._last_keypoint_label or ""
        _gc_learned = (
            self.grasp_calibration.learned_offset(_gc_label)
            if self.grasp_calibration is not None
            else None
        )
        _gc_initial = _gc_learned if _gc_learned is not None else self.GRIPPER_OPEN_Z_OFFSET
        adjusted[2] += _gc_initial
        if _gc_learned is not None:
            logger.info(
                "Grasp learned-offset for %r: %.3f m (%d samples)",
                _gc_label,
                _gc_learned,
                self.grasp_calibration._stats.get(
                    _gc_label.lower().rstrip("0123456789 "), type("_", (), {"n": 0})()
                ).n,
            )
        self._grasp_perception_target_z = float(target[2])
        # Per-pick reset: a stale close-height from the PREVIOUS pick must
        # never feed this pick's release give-back.
        self._actual_grasp_tcp_z = None

        current = self._get_current_position()
        safe_z = self._safe_clearance_z(current, adjusted)
        transit = []
        if np.linalg.norm(adjusted - current) > 0.5:
            lift = current.copy()
            lift[2] = safe_z
            transit.append((lift, list(self.GRASP_ORIENTATION), None, "lift"))
            above = adjusted.copy()
            above[2] = safe_z
            transit.append((above, list(self.GRASP_ORIENTATION), None, "above"))
        else:
            above = adjusted.copy()
            above[2] = max(adjusted[2] + 0.10, safe_z)
            transit.append((above, list(self.GRASP_ORIENTATION), None, "above"))

        label = self._last_keypoint_label or ""
        if self._blend_enabled():
            # The descent target only exists AFTER a wrist capture, so a phrase
            # can only absorb it when refinement is off (the default here).
            refine_pending = bool(
                self._wrist_refine_enabled and label and self._wrist_refine_mode() != "off"
            )
            phrase = list(transit)
            with_descent = not refine_pending
            if with_descent:
                phrase.append(
                    (
                        adjusted.copy(),
                        list(orient),
                        self.velocity * 0.9,
                        "grasp_descent",
                    )
                )
            if self._blend_transit(phrase, phrase="approach"):
                if with_descent:
                    # Terminal row was r=0: the arm IS stopped at the grasp pose.
                    return
                transit = []
        for pos_i, ori_i, _v, _lbl in transit:
            self._move_to(pos_i, ori_i)

        # Wrist-camera refinement, gated by control.wrist_refine (off by
        # default); when off the servo goes straight to the birdview/sideview
        # target.
        if label:
            refined = self._refine_with_wrist(
                label, adjusted, z_plane=self._grasp_perception_target_z
            )
            if refined is not None:
                new_pos = np.array(refined["position_3d"])
                xy_diff = np.linalg.norm(new_pos[:2] - adjusted[:2])
                if xy_diff < 0.05:
                    adjusted[:2] = new_pos[:2]
                    # ray_plane's Z IS the prior it was handed; re-applying it
                    # would add the gripper offset twice. Only the depth path
                    # can correct Z.
                    if refined.get("source") != "ray_plane" and (
                        abs(new_pos[2] - adjusted[2]) < 0.03
                    ):
                        adjusted[2] = new_pos[2] + self.GRIPPER_OPEN_Z_OFFSET
                    logger.info("Wrist XY refine: delta=%.3fm", xy_diff)
                else:
                    logger.info("Wrist XY too far (%.3fm), keeping birdview", xy_diff)

                # Re-resolve with the wrist's own AR/angle, keeping the source
                # detection's confidence: same strategy rules, fresher geometry.
                img_deg = np.rad2deg(refined["orientation_angle"])
                yaw_deg = (img_deg % 180) - self.WRIST_CAM_OFFSET_DEG
                if yaw_deg > 90:
                    yaw_deg -= 180
                if yaw_deg < -90:
                    yaw_deg += 180
                wrist_det = dict(detection or {})
                wrist_det["aspect_ratio"] = refined["aspect_ratio"]
                wrist_det["orientation_angle"] = refined["orientation_angle"]
                wrist_orient, wrist_strategy = resolve_grasp_orientation(
                    params or {},
                    wrist_det,
                    self,
                    yaw_override_rad=np.deg2rad(yaw_deg),
                    context="wrist-refine",
                )
                if wrist_strategy == "obb":
                    orient = wrist_orient
                    self._active_grasp_strategy = wrist_strategy
                    logger.info(
                        "Wrist grasp: img=%.1f deg -> yaw=%.1f deg (ar=%.1f)",
                        img_deg,
                        yaw_deg,
                        refined["aspect_ratio"],
                    )

        # Persist the (possibly OBB-yawed) orientation so the grasp primitive's
        # descent/clamp/lift reuse it instead of reverting to GRASP_ORIENTATION
        # right before the jaws close. See _grasp_v2.
        self._active_grasp_orient = list(orient)
        # Remember which WAY the object points, so the place can put its heavy
        # end where the receptacle's heavy end is (a PCA axis is only a line).
        self._held_axis_sign = None
        self._held_axis_img = None
        # Clear the mask WITH the sign, or an approach without a detection
        # (recovery's retract_retry) registers the PREVIOUS pick's silhouette.
        self._held_mask = None
        try:
            _m = detection.get("_mask") if isinstance(detection, dict) else None
            if _m is not None:
                from spark_real.perception.mask_geometry import (
                    _pca_obb,
                    heavy_end_sign,
                )

                _a, _ar, _, _ = _pca_obb(np.asarray(_m))
                _sgn = heavy_end_sign(np.asarray(_m), _a)
                self._held_axis_img = float(_a)
                self._held_axis_sign = float(_sgn)
                self._held_mask = np.asarray(_m)
                logger.info(
                    "[held-axis] image axis %.1f deg, heavy end %s (ar %.2f)",
                    np.rad2deg(_a),
                    {1.0: "+", -1.0: "-", 0.0: "UNKNOWN"}.get(_sgn, "?"),
                    _ar,
                )
        except Exception as exc:  # noqa: BLE001 - never let this stop a grasp
            logger.warning("[held-axis] could not measure: %s", exc)
        # Final oriented descent as a JOINT move (pyroki IK -> movej): smooth,
        # fast, hits the yaw exactly. The Cartesian PD servo is jittery near
        # convergence and slow; fall back to it only if the joint-IK path is
        # unavailable or fails.
        #
        # EXCEPT under recording.demo_mode: movej emits NO per-timestep
        # commanded velocity for the demo recorder, while the servo issues one
        # speedl per tick. Off by default; check
        # metadata.spark.commanded_action_fraction on a supervised run.
        if self._demo_mode():
            logger.info(
                "[approach] recording.demo_mode: descending via Cartesian "
                "servo so every tick carries a commanded velocity"
            )
            self._servo_to(adjusted, orient, velocity=self.velocity * 0.9)
            return
        if self._robot_family() == "ur10e":
            try:
                self._movej_via_ik(adjusted, orient, self.velocity * 0.9)
                return
            except Exception as exc:  # noqa: BLE001
                logger.warning(
                    "[approach] joint-IK descend failed (%s); Cartesian-servo fallback",
                    exc,
                )
        self._servo_to(adjusted, orient, velocity=self.velocity * 0.9)

    def _servo_to(self, position: np.ndarray, orientation: list, velocity: float = 0.15) -> bool:
        """
        Servo to a target pose with precise orientation control.

        Returns False when the servo STALLED (gave up against an unreachable
        target, no movel retry issued, TCP parked where it got stuck). True
        otherwise (converged, or a timeout the movel fallback finished). The
        place descent must not treat a stall as an arrival.
        """
        self._check_abort()
        target_pose = list(position) + list(orientation)
        self._servo.max_vel_linear = min(velocity, 0.25)
        converged = self._servo.move_to_pose(target_pose, velocity=velocity)
        self._check_abort()
        stalled = False
        if not converged:
            # A STALL means the servo held a command into an unreachable
            # target. A movel to that same pose would drive into the identical
            # obstruction, risking a protective stop. Accept the current pose
            # and let the caller's verification decide. A TIMEOUT is different:
            # the servo was still closing, so movel can finish the job.
            if getattr(self._servo, "last_exit", "") == "stalled":
                stalled = True
                logger.warning(
                    "Servo stalled at pos_err=%.4f ori_err=%.4f; NOT retrying "
                    "with movel (same unreachable target)",
                    self._servo.last_pos_err,
                    self._servo.last_ori_err,
                )
            else:
                logger.warning("Servo did not converge, falling back to movel")
                self._move_to_linear(position, orientation, velocity=velocity)
        # Hand the controller back before anything else uploads a program.
        # The servo streams servoL commands and OWNS the controller until it is
        # stopped; a gripper command is a URScript upload, and uploading one
        # while the servo still holds the channel silently loses it (jaw
        # register 175.95 -> 175.95, zero travel).
        try:
            if hasattr(self.robot, "servo_stop"):
                self.robot.servo_stop()
        except Exception as exc:  # noqa: BLE001 - never let cleanup stop motion
            logger.warning("servo_stop after servo failed: %s", exc)

        actual = self._servo._get_tcp_pose()
        actual_rotvec = actual[3:6]
        R_actual = Rotation.from_rotvec(actual_rotvec).as_matrix()
        tool_x_world = R_actual @ np.array([1, 0, 0])
        tool_y_world = R_actual @ np.array([0, 1, 0])
        logger.info(
            "Servo result: tool_X=%.1f deg, tool_Y=%.1f deg (closing axis)",
            np.rad2deg(np.arctan2(tool_x_world[1], tool_x_world[0])),
            np.rad2deg(np.arctan2(tool_y_world[1], tool_y_world[0])),
        )
        return not stalled

    def _place_arrival(self, label: str, detection) -> tuple:
        """Did the arm end up OVER container ``label``? -> (ok, detail).

        ``ok`` is True whenever the question cannot be answered: a container
        with no slots and no measured width has an UNKNOWN extent
        (success_predicates._container_obb returns None). Missing perception
        never blocks a release.

        The test is the container's own OBB in XY, widened by
        PLACE_ARRIVAL_XY_MARGIN_M, evaluated against the CURRENT TCP, not
        against the pose the servo was commanding.
        """
        obb = container_obb_xy(detection) if detection is not None else None
        if obb is None:
            return True, ""
        tcp = self._get_current_position()
        if obb_contains(tcp[:2], obb, self.PLACE_ARRIVAL_XY_MARGIN_M):
            return True, ""
        centre, half_a, half_b, theta = obb
        d = np.asarray(tcp[:2], dtype=float) - np.asarray(centre, dtype=float)
        ca, sa = np.cos(theta), np.sin(theta)
        along = abs(d[0] * ca + d[1] * sa)
        across = abs(-d[0] * sa + d[1] * ca)
        detail = (
            f"TCP ({tcp[0]:.3f},{tcp[1]:.3f}) is outside '{label or '?'}': "
            f"{along * 100:.1f}/{across * 100:.1f} cm from centre vs "
            f"half=({half_a * 100:.1f},{half_b * 100:.1f})cm "
            f"+{self.PLACE_ARRIVAL_XY_MARGIN_M * 100:.0f}cm margin"
        )
        return False, detail

    def _note_place_arrival(self, label: str, arrived: bool, detail: str) -> None:
        """Record that a place transport ran, and how it went.

        Read by ReleaseMixin._transport_delivered, which uses the LABEL to know
        which container to judge against and recomputes the verdict from the
        live TCP; ``arrived``/``detail`` feed the failed-action message and
        the log. The release ignores a record whose label no longer matches
        ``_last_place_label``.
        """
        self._place_arrival_record = {
            "label": label,
            "arrived": bool(arrived),
            "detail": detail,
        }

    def _transport_to(
        self,
        target: np.ndarray,
        target_detection: dict = None,
        target_label: str = "",
        params: dict = None,
        lead=None,
    ):
        """
        Lift, horizontal move, servo descend with orientation matching.

        `params` are the plan node's params; only grasp_strategy /
        grasp_yaw_deg are read.

        `lead` is an optional list of ``(position, orientation, velocity,
        label)`` transit waypoints to fly BEFORE the lift, as part of the same
        blended phrase (the legato look-ahead folds a preceding
        ``move_relative`` into the transport). Ignored when blending is off.

        Returns True if the transport reached its place pose. False when:

          * the grip check found a DROP (``_holding`` is already cleared), or
          * the arm did not ARRIVE inside the target container's planar
            footprint (``_place_arrival``). The object is still held; the
            caller reports the action failed so the tree can recover.
        """
        self._check_abort()
        self._last_place_label = target_label
        self._place_arrival_record = None

        orient = self.GRASP_ORIENTATION
        # Does `orient` end up carrying a slot/fit/OBB alignment? Read by the
        # deep-dip side choice below: an aligned yaw may only be traded for
        # its grasp-symmetric 180 twin, a free yaw may be picked outright.
        yaw_aligned = False
        # Slotted-container safety net for a generic place (move_to_keypoint
        # + release). For a slotted tray the raw container OBB major axis is
        # PERPENDICULAR to the slots (world-PCA axis-swap on a near-square
        # tray), so derive the slot direction the SAME way place_in_slot does
        # and align to that instead of the raw OBB.
        slots = target_detection.get("slots") if target_detection else None
        if slots:
            slot_dir, slot_src = resolve_slot_direction(target_detection, target_label)
            yaw = (slot_dir + np.pi / 2.0) % np.pi - np.pi / 2.0
            yaw = self._nearest_symmetric_yaw(yaw)
            # Slot direction is a validated source and bypasses the strategy
            # gates, but the cable limit still applies to the composed rotation.
            slot_orient = compose_yaw(self, yaw, context="place-slots")
            if slot_orient is not None:
                # Same directed head/tail decision as the obb place path and
                # place_in_slot; axis alignment alone (mod pi) is a coin flip
                # on which end of the slot the tool's tip lands in.
                orient = self._orient_head_to_head(slot_orient, target_detection)
                yaw_aligned = True
            logger.info(
                "[place_orient] slotted container '%s': aligning to slots "
                "(dir=%.1f deg src=%s) -> yaw=%.1f deg, NOT raw OBB",
                target_label or "?",
                np.rad2deg(slot_dir),
                slot_src,
                np.rad2deg(yaw),
            )
        elif target_detection:
            target_orient = target_detection.get("orientation_angle")

            # SHAPE FIT FIRST; it decides on its own whether orientation
            # matters. A slot or tool cut-out is shaped like the thing that
            # goes in it, so one rotation fits with a large margin; for a bowl
            # the margin collapses and the fit says "do not rotate".
            fitted = self._fit_place_yaw(target_detection)
            if fitted is not None:
                orient = fitted
                yaw_aligned = True
                logger.info("[place_orient] using the shape fit for the place")
                target_orient = None  # the fit is the only source for this target

            # The axis path gives the right LINE; the fit is also used for the
            # half it answers without any image-to-world frame conversion:
            # whether that line should be flipped 180 deg. See
            # _orient_head_to_head, which compares both in one image frame.

            if target_orient is not None and abs(float(target_orient)) > 1e-3:
                place_orient, place_strategy = resolve_grasp_orientation(
                    params or {}, target_detection, self, context="place"
                )
                # Adopt ANY resolved yaw ("obb", "plane", "target_axis",
                # "caller_yaw"). "cgn"/"se3" are excluded on purpose: those
                # dispatch to a 6-DoF backend and keep the approach top-down.
                if place_strategy not in ("topdown", "cgn", "se3") and (
                    place_orient is not None
                ):
                    orient = place_orient
                    place_orient = self._orient_head_to_head(
                        place_orient, target_detection
                    )
                    orient = place_orient
                    yaw_aligned = True
                    logger.info(
                        "[place_orient] adopting '%s' yaw for the place",
                        place_strategy,
                    )
            else:
                logger.info(
                    "[place_orient] skipping yaw-align for '%s' " "(orient<=1mrad or missing)",
                    target_label or "?",
                )

        # DEEP-DIP SIDE CHOICE, decided BEFORE the transport flies and folded
        # into the approach orientation. When the upcoming release asks for a
        # dip past the positive pitch budget (wrist-camera clearance, 40 deg),
        # rotate the approach yaw so the dip lands on the NEGATIVE side (90 deg
        # budget). See ReleaseMixin._dip_side_orient for the rules.
        orient = self._dip_side_orient(orient, yaw_aligned=yaw_aligned)

        place_target = target.copy()
        logger.info("Place '%s': target_z=%.3f", target_label, place_target[2])

        current_pose = self._get_current_position()
        # An operator-drawn path, if one is armed for THIS destination, becomes
        # additional lead rows in the same blended phrase (the "trace"
        # annotation kind in perception/annotations.py). Stand-in for the
        # lateral motion planning SPARK does not have.
        lead = self._consume_drawn_trace(target_label, orient, lead)
        clearance_from = np.asarray(lead[-1][0], dtype=float) if lead else current_pose
        safe_z = self._safe_clearance_z(clearance_from, place_target)
        lift = clearance_from.copy()
        lift[2] = safe_z
        above_target = place_target.copy()
        above_target[2] = safe_z
        # Hand the servo only the last centimetres of the descent. The servo
        # is a PD loop whose gain falls with the error (kp_pos=2.0,
        # pos_threshold=3 mm), so it crawls near convergence (3.2 s for the
        # full descent). A movej covers the free-space part; the servo keeps
        # the part that can touch something, with its compliance and its
        # unreachable-target stall exit intact.
        standoff = place_target.copy()
        standoff[2] = min(
            float(place_target[2]) + self._place_standoff_m(place_target), safe_z
        )

        if self._blend_enabled():
            phrase = list(lead or [])
            # Lift at the orientation the arm is ALREADY holding, not the base
            # constant: snapping to GRASP_ORIENTATION would rotate the wrist
            # to zero on the way up and out to the place yaw on the way
            # across. Holding the carry angle leaves a SINGLE rotation during
            # the horizontal transit.
            carry_orient = getattr(self, "_active_grasp_orient", None)
            if not carry_orient:
                try:
                    carry_orient = list(
                        np.asarray(self.robot.get_tcp_pose(), dtype=float)[3:6]
                    )
                except Exception:  # noqa: BLE001
                    carry_orient = list(self.GRASP_ORIENTATION)
            phrase.append((lift, list(carry_orient), None, "lift"))
            phrase.append((above_target, list(orient), None, "above_target"))
            if standoff[2] < above_target[2] - 1e-4:
                # Terminal row of the SAME program: no extra upload; the r=0
                # stop is the one the exact joint-arrival test applies to.
                phrase.append((standoff, list(orient), None, "place_standoff"))
            if self._blend_transit(phrase, final_radius_m=0.0, phrase="transport"):
                # ONE grip check, at the parked end of the phrase. A
                # mid-transport check sends a gripper URScript, which REPLACES
                # the running program and would cancel the blend; a passive
                # register read cannot see a drop since the last publish.
                if not self._transport_grip_ok("above_target"):
                    self._holding = False
                    return False
                return self._place_descent(place_target, orient, target_label, target_detection)
            for pos_i, ori_i, _v, _lbl in lead or []:
                self._move_to(pos_i, ori_i)

        self._move_to(lift, self.GRASP_ORIENTATION)

        # Per-waypoint drop detection at EACH stationary transport arrival
        # (lift, then above-target). On a confirmed drop, clear _holding and
        # return; the caller fails the action and recovery re-grasps.
        if not self._transport_grip_ok("lift"):
            self._holding = False
            return False

        self._move_to(above_target, orient)
        if not self._transport_grip_ok("above_target"):
            self._holding = False
            return False

        # Same standoff as the blended phrase, at the cost of one extra
        # program upload.
        if standoff[2] < above_target[2] - 1e-4:
            self._move_to(standoff, orient)
        return self._place_descent(place_target, orient, target_label, target_detection)

    def _fit_place_yaw(self, target_detection):
        """Place yaw from SHAPE AGREEMENT, or None when shape cannot decide.

        Registers the held object's mask against the receptacle's over every
        rotation and keeps the best. Returns a composed orientation when the
        winner beats every orientation >=90 deg away by `JIGSAW_MIN_MARGIN`,
        and None otherwise (a round container).

        The angle is applied as a DELTA on the orientation the arm is already
        carrying, so it never depends on an image-to-world sign convention.
        """
        try:
            held = getattr(self, "_held_mask", None)
            tgt = (
                target_detection.get("_mask")
                if isinstance(target_detection, dict)
                else None
            )
            if held is None or tgt is None:
                return None
            from spark_real.perception.mask_geometry import best_fit_rotation

            ang, iou, margin = best_fit_rotation(held, tgt)
            logger.info(
                "[fit] rotation %.1f deg  IoU %.3f  margin %.3f "
                "(need margin>=%.3f, IoU>=%.3f)",
                np.rad2deg(ang), iou, margin,
                self.JIGSAW_MIN_MARGIN, self.JIGSAW_MIN_IOU,
            )
            if margin < self.JIGSAW_MIN_MARGIN or iou < self.JIGSAW_MIN_IOU:
                logger.info(
                    "[fit] shape does not decide the orientation here "
                    "(margin %.3f); leaving the yaw to the axis path",
                    margin,
                )
                return None

            from spark_real.control.grasp_strategy import (
                base_orientation,
                compose_yaw,
                measured_yaw_offset,
            )

            cur = measured_yaw_offset(
                self._active_grasp_orient or base_orientation(self),
                base_orientation(self),
            )
            # want = cur + ang: the fit reports the signed rotation that
            # carries the held shape onto the target, ADDED to the carry yaw,
            # no extra 90 and no negation (verified: cur -5.7, ang 273 = -87
            # -> want -92.7 -> tool_X -8.6 against a measured -9). `cur - ang`
            # and `cur - ang - 90` were each wrong by a clean 90 or 180.
            want = cur + float(ang)
            want = (want + np.pi) % (2 * np.pi) - np.pi
            if abs(want) > self.MAX_YAW_OFFSET + 1e-9:
                alt = want + (np.pi if want < 0 else -np.pi)
                if abs(alt) <= self.MAX_YAW_OFFSET + 1e-9:
                    # Substitute only when the shapes cannot tell the two ends
                    # apart. A DECISIVE fit means the 180-equivalent is the
                    # orientation the fit rejected: refusing to orient beats
                    # orienting backwards (grasp_strategy.place_aware_yaw
                    # prevents most of these upstream).
                    if margin >= 2.0 * self.JIGSAW_MIN_MARGIN:
                        logger.warning(
                            "[fit] wanted %.1f deg (past the wrist limit) and "
                            "the fit is DECISIVE (margin %.3f), so the "
                            "180-equivalent %.1f deg would seat the object "
                            "end-for-end; refusing to orient rather than "
                            "placing it backwards",
                            np.rad2deg(want), margin, np.rad2deg(alt),
                        )
                        return None
                    logger.warning(
                        "[fit] wanted %.1f deg (past the wrist limit); using "
                        "the 180-equivalent %.1f deg -- margin %.3f says the "
                        "ends are interchangeable here",
                        np.rad2deg(want), np.rad2deg(alt), margin,
                    )
                    want = alt
                else:
                    logger.warning(
                        "[fit] wanted %.1f deg, beyond the wrist limit and no "
                        "reachable equivalent; leaving the yaw to the axis path",
                        np.rad2deg(want),
                    )
                    return None
            return compose_yaw(self, want, context="place")
        except Exception as exc:  # noqa: BLE001 - never let this stop a place
            logger.warning("[fit] failed (%s); falling back to the axis path", exc)
            return None

    def _orient_head_to_head(self, place_orient, target_detection):
        """Flip the place yaw by 180 deg when the object would go in backwards.

        The yaw resolved so far aligns the two AXES; a PCA axis is a line,
        identical under a 180 deg flip, and the wrong half is a tool seated
        end-for-end.

        Both masks break the tie by being fatter at one end
        (mask_geometry.heavy_end_sign). The comparison stays in ONE image
        frame, so no image-to-world sign convention can invert it.

        Requires BOTH heavy ends to be known. Either coming back 0.0 means the
        shape is too symmetric to call, and the yaw passes through untouched.
        """
        # FIRST: jigsaw. Which rotation makes the two SHAPES coincide answers
        # direction and angle together. Only fall through to the heavy-end
        # heuristic when the shapes cannot decide it.
        try:
            mask_t0 = (
                target_detection.get("_mask")
                if isinstance(target_detection, dict)
                else None
            )
            mask_h0 = getattr(self, "_held_mask", None)
            if mask_t0 is not None and mask_h0 is not None:
                from spark_real.perception.mask_geometry import best_fit_rotation

                _ang, _iou, _margin = best_fit_rotation(mask_h0, mask_t0)
                logger.info(
                    "[jigsaw] best-fit rotation %.1f deg, IoU %.3f, margin %.3f "
                    "(need margin >= %.3f)",
                    np.rad2deg(_ang), _iou, _margin, self.JIGSAW_MIN_MARGIN,
                )
                if _margin >= self.JIGSAW_MIN_MARGIN and _iou >= self.JIGSAW_MIN_IOU:
                    ang_h = getattr(self, "_held_axis_img", None)
                    mt, _, _, _ = __import__(
                        "spark_real.perception.mask_geometry",
                        fromlist=["_pca_obb"],
                    )._pca_obb(np.asarray(mask_t0))
                    if ang_h is not None:
                        # Does the shape fit agree with the axis alignment, or
                        # with its 180 flip? Same image frame throughout.
                        def _w(a):
                            return (a + np.pi) % (2 * np.pi) - np.pi

                        axis_rot = _w(mt - ang_h)
                        # Reduce into [-pi/2, pi/2] BEFORE comparing. A PCA
                        # eigenvector's sign is arbitrary, so mt - ang_h is
                        # only defined mod pi (36/50 vs 50/50 correct flips on
                        # synthetic trials). Same reduction as the heavy-end
                        # rung below.
                        if axis_rot > np.pi / 2:
                            axis_rot -= np.pi
                        elif axis_rot < -np.pi / 2:
                            axis_rot += np.pi
                        if abs(_w(_ang - axis_rot)) > np.pi / 2:
                            flipped = self._flip_yaw_180(place_orient)
                            if flipped is not None:
                                logger.info(
                                    "[jigsaw] shape fit disagrees with the axis "
                                    "alignment by ~180; flipping the place yaw"
                                )
                                return flipped
                            logger.warning(
                                "[jigsaw] shape fit wants a 180 flip but it "
                                "exceeds the wrist limit; seating end-for-end"
                            )
                        else:
                            logger.info("[jigsaw] shape fit agrees; no flip")
                        return place_orient
        except Exception as exc:  # noqa: BLE001
            logger.warning("[jigsaw] failed (%s); falling back to heavy-end", exc)

        try:
            sgn_h = getattr(self, "_held_axis_sign", None)
            ang_h = getattr(self, "_held_axis_img", None)
            mask_t = (
                target_detection.get("_mask")
                if isinstance(target_detection, dict)
                else None
            )
            if sgn_h in (None, 0.0) or ang_h is None or mask_t is None:
                logger.info(
                    "[head-to-head] not decidable (held sign=%s, target mask=%s); "
                    "leaving the yaw as resolved",
                    sgn_h,
                    "yes" if mask_t is not None else "no",
                )
                return place_orient

            from spark_real.perception.mask_geometry import _pca_obb, heavy_end_sign

            ang_t, ar_t, _, _ = _pca_obb(np.asarray(mask_t))
            sgn_t = heavy_end_sign(np.asarray(mask_t), ang_t)
            if sgn_t == 0.0:
                logger.info(
                    "[head-to-head] target heavy end UNKNOWN (ar %.2f); leaving "
                    "the yaw as resolved",
                    ar_t,
                )
                return place_orient

            def _wrap(a):
                return (a + np.pi) % (2 * np.pi) - np.pi

            dir_h = ang_h + (0.0 if sgn_h > 0 else np.pi)
            dir_t = ang_t + (0.0 if sgn_t > 0 else np.pi)
            # The rotation the pipeline effectively applied, reduced mod 180,
            # which is all an axis alignment can express.
            rot = _wrap(ang_t - ang_h)
            if rot > np.pi / 2:
                rot -= np.pi
            elif rot < -np.pi / 2:
                rot += np.pi
            residual = _wrap(dir_t - (dir_h + rot))
            if abs(residual) <= np.pi / 2:
                logger.info(
                    "[head-to-head] heads agree (residual %.0f deg); no flip",
                    np.rad2deg(residual),
                )
                return place_orient

            flipped = self._flip_yaw_180(place_orient)
            if flipped is None:
                logger.warning(
                    "[head-to-head] object would seat BACKWARDS (residual "
                    "%.0f deg) but the 180 flip exceeds the wrist limit; "
                    "placing anyway, end-for-end",
                    np.rad2deg(residual),
                )
                return place_orient
            logger.info(
                "[head-to-head] object would seat BACKWARDS (residual %.0f deg); "
                "flipping the place yaw 180",
                np.rad2deg(residual),
            )
            return flipped
        except Exception as exc:  # noqa: BLE001 - never let this stop a place
            logger.warning("[head-to-head] failed (%s); yaw unchanged", exc)
            return place_orient

    # A shape fit must beat every orientation >=90 deg away by this much
    # before it is allowed to decide direction; below it the object is too
    # symmetric to call and the heavy-end test gets a turn.
    JIGSAW_MIN_MARGIN = 0.05
    JIGSAW_MIN_IOU = 0.20

    def _flip_yaw_180(self, orient):
        """`orient` rotated 180 deg about world Z, or None if out of limit."""
        from spark_real.control.grasp_strategy import (
            base_orientation,
            compose_yaw,
            measured_yaw_offset,
        )

        cur = measured_yaw_offset(orient, base_orientation(self))
        for cand in (cur + np.pi, cur - np.pi):
            if abs(cand) <= self.MAX_YAW_OFFSET + 1e-9:
                return compose_yaw(self, cand, context="place")
        return None

    def _place_standoff_m(self, place_target) -> float:
        """How much of the descent the slow servo must own, in metres.

        A held object first meets the container rim when its BOTTOM reaches
        rim height, i.e. at TCP z = rim_z + held_drop. For anything taller
        than the container is deep that point is ABOVE the release height,
        and contact would land in the fast blended move (no compliance, no
        stall exit). Example: rim -0.220, held_drop 10.3 cm, release z -0.139:
        contact begins at -0.117, but a fixed 2 cm standoff takes over at
        -0.119.

        So: start the slow phase strictly ABOVE the first possible contact,
        with a margin for the held-extent estimate. Short objects keep the
        fixed 2 cm because their computed value is smaller.
        """
        base = float(self.PLACE_SERVO_STANDOFF_M)
        geom = getattr(self, "_last_release_height", None)
        rim = getattr(geom, "rim_z", None)
        drop = getattr(geom, "held_drop_m", None)
        if rim is None or drop is None:
            return base
        try:
            contact_z = float(rim) + float(drop)
            need = (contact_z - float(place_target[2])) + self.PLACE_CONTACT_MARGIN_M
        except (TypeError, ValueError):
            return base
        if not np.isfinite(need) or need <= base:
            return base
        need = min(need, self.PLACE_SERVO_STANDOFF_MAX_M)
        logger.info(
            "Place standoff: %.1f cm (rim=%.3f + held_drop=%.3f -> first "
            "contact at z=%.3f, %.1f cm above the release height); the slow "
            "servo owns the whole contact zone instead of the blend",
            need * 100.0,
            float(rim),
            float(drop),
            contact_z,
            (contact_z - float(place_target[2])) * 100.0,
        )
        return need

    # Above this residual orientation error, the wrist is corrected at
    # clearance height before any contact-zone motion (5 deg: well above
    # servo convergence residuals ~0.06 deg, well below anything that sweeps
    # a held tool into a rim).
    PLACE_ORI_FIX_GATE_RAD = np.deg2rad(5.0)
    # Clearance above the place target for wrist rotations with a held
    # object: longer than the tool extent below the jaws (~10 cm screwdriver)
    # plus margin.
    ROTATE_CLEARANCE_M = 0.15

    def _consume_drawn_trace(self, target_label: str, orient, lead):
        """Fold an armed operator trace into this transport's lead rows.

        Consumed ONCE and only by a transport heading to the label it was
        drawn for; a stale path must never steer an unrelated motion.
        """
        try:
            from spark_real.perception.trace_path import waypoints_to_lead
            from spark_real.routes import state as _state

            trace = getattr(_state, "pending_trace", None)
            if not isinstance(trace, dict):
                return lead
            want = str(trace.get("label") or "")
            if not want or want != str(target_label or ""):
                return lead
            rows = waypoints_to_lead(trace.get("waypoints") or [], orient)
            _state.pending_trace = None  # one move, one path
            if not rows:
                logger.info("[trace] armed path for '%s' had no usable rows", want)
                return lead
            logger.warning(
                "[trace] flying the operator's drawn path to '%s': %d "
                "waypoint(s) ahead of the transport's own rows",
                want, len(rows),
            )
            return list(lead or []) + rows
        except Exception as exc:  # noqa: BLE001 - a bad path must not stop the move
            logger.warning("[trace] could not apply the drawn path: %s", exc)
            return lead

    def _place_descent(self, place_target, orient, target_label: str, target_detection) -> bool:
        """Final servo of a transport, plus the arrival question it answers.

        A stall alone does NOT fail the transport: a 4.9 mm stall in the
        middle of a bowl has delivered the object. What fails it is ending up
        outside the container's own footprint.
        """
        # ORIENTATION IS SETTLED AT HEIGHT, NEVER IN THE CONTACT ZONE. A held
        # tool extends far below the TCP, so a wrist rotation near the rim
        # sweeps a volume the planner never cleared (a held screwdriver swept
        # into the tool bed and the controller went into PROTECTIVE_STOP). If
        # orientation error remains here, rise to a clearance height, fix the
        # wrist THERE, and come back down; the contact-zone servo then owns
        # position only.
        try:
            cur_o = self.current_orientation()
            if cur_o is not None and orient is not None:
                from scipy.spatial.transform import Rotation as _R

                ori_err = float(
                    (_R.from_rotvec(list(orient)).inv()
                     * _R.from_rotvec(list(cur_o))).magnitude()
                )
                if ori_err > self.PLACE_ORI_FIX_GATE_RAD:
                    fix = np.array(place_target, dtype=float).copy()
                    fix[2] = float(place_target[2]) + self.ROTATE_CLEARANCE_M
                    logger.warning(
                        "[place] %.1f deg of orientation left at the "
                        "standoff; correcting at z=%.3f (+%.0f cm clearance) "
                        "before the contact-zone descent",
                        np.rad2deg(ori_err), fix[2],
                        self.ROTATE_CLEARANCE_M * 100,
                    )
                    self._move_to(fix.tolist(), list(orient))
        except Exception as exc:  # noqa: BLE001 - the descent must still run
            logger.warning("[place] orientation pre-check failed: %s", exc)
        # 0.25, not 0.5: this servo owns the rim-contact zone (see
        # _place_standoff_m), so its speed sets the impulse delivered to the
        # container on contact. Keeps a light bowl from being knocked over.
        landed = self._servo_to(place_target, orient, velocity=self.velocity * 0.25)
        arrived, detail = self._place_arrival(target_label, target_detection)
        self._note_place_arrival(target_label, arrived, detail)
        if arrived:
            if not landed:
                logger.info(
                    "Place '%s': servo stalled but the TCP is over the "
                    "container; releasing here is still correct",
                    target_label or "?",
                )
            return True
        logger.warning(
            "Place '%s' DID NOT ARRIVE (%s): %s",
            target_label or "?",
            "servo stalled" if not landed else "servo converged",
            detail,
        )
        return False

    def _transport_grip_ok(self, waypoint: str) -> bool:
        """Maintain + verify the grip at a STATIONARY transport waypoint.

        Robotiq path: VERIFY FIRST via the gObj-primary _grip_intact() (one
        forced publish + two register reads, ~30-200 ms, no jaw motion). Only
        if that does not confirm, re-squeeze to the fully-closed POSITION at
        GRASP_CLOSE_FORCE to take up slow slip, then re-verify. A False after
        that is a real drop.

        Non-robotiq path (FR3/bimanual): delegate to the force-based
        _verify_holding_during_transport().
        """
        if self._gripper_type() != "robotiq_2f85":
            return self._verify_holding_during_transport()
        if self._grip_intact():
            return True
        close_force = int(os.environ.get("SPARK_GRASP_CLOSE_FORCE", self.GRASP_CLOSE_FORCE))
        logger.info(
            "Transport '%s': grip unconfirmed, re-squeezing to take up slack",
            waypoint,
        )
        self._robotiq_resqueeze(speed=50, force=close_force, settle=self.GRASP_VERIFY_SETTLE_S)
        intact = self._grip_intact()
        if not intact:
            logger.warning("Transport: object DROPPED at waypoint '%s'", waypoint)
            # G2: the run can no longer pass without a later successful re-grasp.
            success_verifier.note_transport_drop(self, waypoint)
        return intact

    # Low-level motion helpers

    def _move_to(self, position, orientation, velocity=None):
        """
        Move with workspace protection; uses movej for large moves.
        """
        self._check_abort()
        vel = velocity or self.velocity
        target_raw = np.array(position, dtype=float)
        target = target_raw.copy()
        if not self._check_workspace(target):
            target = np.clip(target, self.WORKSPACE_MIN, self.WORKSPACE_MAX)
            logger.info(
                "Move target CLIPPED by workspace: requested (%.3f,%.3f,%.3f) "
                "-> commanded (%.3f,%.3f,%.3f)",
                target_raw[0],
                target_raw[1],
                target_raw[2],
                target[0],
                target[1],
                target[2],
            )

        # Phrase chaining via waypoint buffer
        buf = getattr(self, "waypoint_buffer", None)
        franky_handle = None
        if buf is not None and hasattr(buf, "pending") and buf.pending > 0:
            try:
                franky_handle = WaypointBuffer.resolve_franky_robot(self.robot)
            except Exception:
                franky_handle = None
        if (
            buf is not None
            and franky_handle is not None
            and hasattr(buf, "pending")
            and buf.pending > 0
        ):
            try:
                buf.add(Waypoint(position=target, orientation=orientation, label="move_to"))
                logger.info("MOVE_TO chained: flushing %d-waypoint phrase", buf.pending)
                ok = buf.flush_cartesian(franky_handle)
                if ok:
                    return
                logger.warning("Chained phrase rejected; falling back to direct move")
            except Exception as exc:
                logger.warning("Chained move failed (%s); falling back", exc)
                try:
                    buf.clear()
                except Exception:
                    pass

        pose = list(target) + list(orientation)
        current = self._get_current_position()
        dist = np.linalg.norm(target - current)
        logger.info(
            "MOVE_TO target=(%.3f,%.3f,%.3f) orient=(%.2f,%.2f,%.2f) "
            "from=(%.3f,%.3f,%.3f) dist=%.3fm v=%.2f",
            target[0],
            target[1],
            target[2],
            orientation[0],
            orientation[1],
            orientation[2],
            current[0],
            current[1],
            current[2],
            dist,
            vel,
        )
        if dist > 1.0:
            logger.warning("Move %.3fm too large, skipping", dist)
            return
        try:
            self._movej_via_ik(target, orientation, vel)
        except AbortRequested:
            raise
        except Exception as e:
            logger.warning("_movej_via_ik failed (%s); trying _movej_to_pose", e)
            try:
                # Full velocity: on UR10e _movej_to_pose IS the primary path
                # (controller firmware IK); halving it would make every UR
                # move crawl at ~0.375 rad/s.
                self._movej_to_pose(pose, vel)
            except Exception:
                raise

    def _issue_linear(self, pose, vel):
        """
        Dispatch a Cartesian move to whichever driver method exists.
        """
        if maybe_osc_move_linear(list(pose), vel):
            return
        if hasattr(self.robot, "move_to_pose"):
            self.robot.move_to_pose(pose, velocity=vel, wait=False)
        elif hasattr(self.robot, "move_linear"):
            try:
                self.robot.move_linear(pose, velocity=vel, asynchronous=True)
            except TypeError:
                self.robot.move_linear(pose, velocity=vel)

    def _move_to_linear(self, position, orientation, velocity=None):
        """
        Force linear (Cartesian) move to preserve TCP orientation.
        """
        self._check_abort()
        vel = velocity or self.velocity
        target = np.array(position)
        if not self._check_workspace(target):
            target = np.clip(target, self.WORKSPACE_MIN, self.WORKSPACE_MAX)
        pose = list(target) + list(orientation)
        dist = np.linalg.norm(target - self._get_current_position())
        if dist > 1.0:
            logger.warning("Move %.3fm too large, skipping", dist)
            return
        try:
            self._issue_linear(pose, vel)
            # Budget the wait to the move actually commanded. There is no
            # q_target on this path (the driver runs its own Cartesian IK), so
            # the only exits are proximity and stillness-after-motion, and a
            # movel short enough to finish inside the URScript start lag hits
            # neither. 3x the ideal duration plus the start lag and a second
            # of slack, capped at 15 s.
            ideal = dist / max(vel, 1e-3)
            budget = float(
                np.clip(
                    3.0 * ideal + self.URSCRIPT_START_LAG_S + 1.0,
                    self._min_motion_budget_s(),
                    15.0,
                )
            )
            arrived = self._wait_for_motion(target, timeout=budget)
            if arrived is False:
                # The arm never moved and is not at the target: the command
                # was dropped (dead RTDE control script, refused URScript).
                # Raising lets callers with a fallback (e.g. the tilt release)
                # take it instead of proceeding as if the pose was reached.
                raise RuntimeError(
                    f"linear move to {list(np.round(target, 3))} produced no "
                    "motion (command dropped)"
                )
        except AbortRequested:
            raise
        except Exception as e:
            if any(k in str(e).lower() for k in ("singular", "protective", "accel")):
                try:
                    self._movej_to_pose(pose, vel * 0.5)
                except Exception:
                    raise
            else:
                raise

    # Joint tolerance counting a URScript movej as arrived (rad, per joint).
    JOINT_ARRIVE_TOL = 0.01
    # URScript upload/start lag on the primary socket: the arm is stationary for
    # up to ~1 s after the send, so stillness before this means "not started".
    URSCRIPT_START_LAG_S = 1.0
    # Arrival-poll cadence. The exact tests (joint target, Cartesian proximity)
    # are cheap register reads, so polling at 25 ms rather than 100 ms removes
    # up to 75 ms of dead time per program (~0.5 s over a pick-and-place).
    # MOTION_SAMPLE_S stays the window the displacement test is evaluated
    # over; see _StillnessTracker.
    MOTION_POLL_S = 0.025
    MOTION_SAMPLE_S = 0.1
    MOTION_STILL_SAMPLES = 3
    # Slack added on top of 3x the ideal move duration when budgeting a wait.
    MOTION_BUDGET_SLACK_S = 1.5

    # Blended-path defaults. Config: control.blend_motion / blend_radius_m /
    # blend_descent_radius_m / blend_min_radius_m / blend_max_rows; env
    # SPARK_UR_BLEND overrides the on/off switch. See _blend_cfg.
    BLEND_MOTION_DEFAULT = True
    BLEND_RADIUS_M = 0.05
    # A phrase that ends in a grasp/place descent keeps that descent
    # essentially vertical: 2 cm of corner-cut into a 10-15 cm descent, not 5.
    BLEND_DESCENT_RADIUS_M = 0.02
    BLEND_MIN_RADIUS_M = 0.01
    BLEND_MAX_ROWS = 8
    BLEND_TIMEOUT_CAP_S = 45.0
    # No progress-register advance AND a still TCP for this long, mid-path, is
    # the stall signature (vs a 15.0 s grind into a mechanical stop ending as
    # a silent full-path timeout).
    BLEND_STALL_S = 3.0
    # Last-resort brake decel (rad/s^2) for a driver with no velocity-sized
    # brake() of its own. Only reached by bare-mixin test doubles; the real
    # driver computes this from the speed it last commanded.
    BLEND_ABORT_DECEL = 4.0

    def _min_motion_budget_s(self) -> float:
        """Floor for any arrival budget, tied to the program start lag.

        A fast move has a small `ideal` and collapses onto the floor; a floor
        that does not clear the upload lag by a real margin times out before
        the arm has moved and reads as a motion failure.
        """
        return float(self.URSCRIPT_START_LAG_S + self.MOTION_BUDGET_SLACK_S)

    def _wait_for_motion(self, target, timeout=15.0, q_target=None):
        """
        Block until a fire-and-forget URScript movej lands.

        ``q_target`` is the authoritative signal: the controller is done when
        the measured joints equal the commanded ones. Cartesian proximity
        alone is not, because the commanded pose and the achieved TCP
        routinely differ by more than the 5 mm gate (IK rounding, TCP offset),
        and the stillness exit cannot fire on a move that finished inside the
        start lag.
        """
        t0 = time.time()
        motion = _StillnessTracker(
            self._get_current_position(),
            t0,
            sample_s=self.MOTION_SAMPLE_S,
            still_samples=self.MOTION_STILL_SAMPLES,
        )
        next_record = t0

        def _arrived():
            """Retire the driver's motion lease: this move is provably done."""
            clear = getattr(self.robot, "clear_motion_lease", None)
            if callable(clear):
                try:
                    clear()
                except Exception:  # noqa: BLE001 - never block on bookkeeping
                    pass

        while time.time() - t0 < timeout:
            self._check_abort()
            time.sleep(self.MOTION_POLL_S)
            now = time.time()
            tcp = self._get_current_position()
            # The recorder keeps its own ~10 Hz cadence: the faster poll exists
            # to cut arrival latency, not to change what gets logged.
            if self._recorder and now >= next_record:
                self._recorder.record()
                next_record = now + self.MOTION_SAMPLE_S
            # 1. Joints match the command: unambiguously arrived.
            if q_target is not None:
                try:
                    q = np.asarray(self.robot.get_joint_positions(), dtype=float)
                    if np.max(np.abs(q - np.asarray(q_target, dtype=float))) < (
                        self.JOINT_ARRIVE_TOL
                    ):
                        _arrived()
                        return True
                except Exception:  # noqa: BLE001 - fall through to the other tests
                    pass
            # 2. Cartesian proximity (kept: works when q_target is unavailable).
            if np.linalg.norm(tcp - target) < 0.005:
                _arrived()
                return True
            # 3. Stillness, past the start lag AND only after the arm was seen
            #    to move: if the URScript upload lag runs past
            #    URSCRIPT_START_LAG_S, "still" means "not started yet", not
            #    "arrived" (returned ~20 cm short, grasp closed above the
            #    object). The joint-target test above is exact and needs no
            #    motion history.
            motion.update(tcp, now)
            settled = motion.stopped() and now - t0 > self.URSCRIPT_START_LAG_S
            if settled and motion.ever_moved:
                _arrived()
                return True
        dist = float(np.linalg.norm(self._get_current_position() - target))
        logger.warning(
            "[_wait_for_motion] %.1fs timeout with no arrival signal " "(moved=%s, dist=%.3fm)",
            timeout,
            motion.ever_moved,
            dist,
        )
        # A timeout where the arm NEVER MOVED and is still away from the
        # target is a dropped command, not a slow one. False lets callers
        # distinguish "arm is basically there" from "nothing happened".
        return bool(motion.ever_moved or dist < 0.02)

    # Blended multi-waypoint motion: config, abort, progress, arrival

    def _blend_cfg(self) -> dict:
        """Resolve the blended-path knobs once per executor.

        Precedence: env SPARK_UR_BLEND (on/off only) > control.blend_* in the
        resolved profile / family YAML > the class defaults above.

        Default ON for the transit chains (approach + transport). Safety:
          * abort: a blended path emits URScript stopj on the same socket and
            provably brakes (_halt_arm).
          * every row goes through the workspace clip (_clip_blend_waypoints)
            and the joint/singularity filter (_blend_rows_pass_safety_filter).
          * the terminal row is forced to r=0 and a descent row keeps
            BLEND_DESCENT_RADIUS_M = 2 cm, so an over-large radius cannot
            corner-cut into the object.
        PolyScope behaviour on an over-large blend radius (silent reduction vs
        runtime error) must be read off the pendant log on the first rig run.
        control.blend_motion: false (or SPARK_UR_BLEND=0) restores per-move
        dispatch.
        """
        cached = getattr(self, "_blend_cfg_cached", None)
        if cached is not None:
            return cached
        control = self._control_config()
        enabled = bool(control.get("blend_motion", self.BLEND_MOTION_DEFAULT))
        env = os.environ.get("SPARK_UR_BLEND")
        if env is not None:
            enabled = env not in ("0", "false", "False", "")
        cfg = {
            "enabled": enabled,
            "radius_m": float(control.get("blend_radius_m", self.BLEND_RADIUS_M)),
            "descent_radius_m": float(
                control.get("blend_descent_radius_m", self.BLEND_DESCENT_RADIUS_M)
            ),
            "min_radius_m": float(control.get("blend_min_radius_m", self.BLEND_MIN_RADIUS_M)),
            "max_rows": int(control.get("blend_max_rows", self.BLEND_MAX_ROWS)),
            "register": int(control.get("blend_progress_register", UR_PROGRESS_REGISTER)),
            # Fold a preceding move_relative into the transport phrase, i.e.
            # blend ACROSS a BT node boundary. Separate switch, default off;
            # see _execute_ur_blend_phrase.
            "cross_node": bool(control.get("blend_cross_node", False)),
        }
        env_x = os.environ.get("SPARK_UR_BLEND_CROSS_NODE")
        if env_x is not None:
            cfg["cross_node"] = env_x not in ("0", "false", "False", "")
        self._blend_cfg_cached = cfg
        logger.info(
            "Blended motion: %s (r=%.3fm, descent r=%.3fm, max_rows=%d, reg=%d, " "cross_node=%s)",
            "ON" if enabled else "off",
            cfg["radius_m"],
            cfg["descent_radius_m"],
            cfg["max_rows"],
            cfg["register"],
            cfg["cross_node"],
        )
        return cfg

    def _blend_enabled(self) -> bool:
        """True when transit chains may be issued as one blended path.

        demo_mode is excluded on purpose: it exists so every recorded tick
        carries a COMMANDED velocity, and a blended movej program emits one
        setpoint for a multi-second motion.
        """
        if not self._blend_cfg()["enabled"]:
            return False
        if self._demo_mode():
            return False
        return True

    def _blend_register(self) -> int:
        return self._blend_cfg()["register"]

    def _rtde_receive(self):
        """The rtde_receive handle, wherever it sits in the wrapper chain."""
        obj = self.robot
        for _ in range(4):
            if obj is None:
                return None
            r = getattr(obj, "_rtde_r", None)
            if r is not None:
                return r
            obj = getattr(obj, "_robot", None)
        return None

    def _read_blend_progress(self):
        """Current value of the blend progress register, or None if unreadable.

        A pure rtde_receive read: it uploads nothing, so unlike a gripper
        publish it can never replace the running program.
        """
        reg = self._blend_register()
        fn = getattr(self.robot, "get_output_int_register", None)
        if callable(fn):
            try:
                return int(fn(reg))
            except Exception:  # noqa: BLE001
                return None
        rtde_r = self._rtde_receive()
        if rtde_r is None:
            return None
        try:
            return int(rtde_r.getOutputIntRegister(reg))
        except Exception:  # noqa: BLE001 - build may not surface this register
            return None

    def _refresh_motion_lease(self):
        """Keep the driver's motion lease alive for the length of a path.

        _MOVEJ_LEASE_S is 1.5 s, sized for ONE move's upload lag. If the lease
        lapses mid-path a single _publish_gripper_state replaces the running
        program and cancels the whole path. Re-noting a movej pushes the lease
        out by another 1.5 s; _wait_for_blended_path retires it on arrival.
        """
        note = getattr(self.robot, "_note_motion_script", None)
        if callable(note):
            try:
                note("movej(")
            except Exception:  # noqa: BLE001 - never block on bookkeeping
                pass

    def _retire_motion_lease(self):
        clear = getattr(self.robot, "clear_motion_lease", None)
        if callable(clear):
            try:
                clear()
            except Exception:  # noqa: BLE001
                pass

    def _halt_arm(self) -> bool:
        """Decelerate the arm NOW. True only if the brake left this process.

        A new send REPLACES the running program, so a URScript stop both brakes
        the arm and kills a blended path. No hardcoded ``stopj(2.0)`` here:
        overshoot is ``v**2 / 2a``, so a FIXED decel gets quadratically worse
        as commanded velocity rises (at 1.78 rad/s, ``stopj(2.0)`` is 0.79 rad
        of travel AFTER the stop, roughly 0.8 m of TCP at full reach). The
        driver sizes its decel from the speed it last commanded
        (UR10eDriver.brake_decel).

        Order:
          1. the driver's own ``brake()``, the only branch that knows what
             velocity was commanded;
          2. the executor's ``_stop_robot()``, the shared escalation ladder
             (reopened socket, then Dashboard); its URScript branch emits a
             FIXED ABORT_DECEL_FALLBACK, hence (1) first;
          3. a literal stopj, for a bare MotionMixin with neither.
        """
        ok = False
        brake = getattr(self.robot, "brake", None)
        if callable(brake):
            try:
                ok = bool(brake())
            except Exception as exc:  # noqa: BLE001 - a stop never raises
                logger.warning("[blend] driver brake failed: %s", exc)
        if not ok:
            stop = getattr(self, "_stop_robot", None)
            if callable(stop):
                try:
                    ok = bool(stop())
                except Exception as exc:  # noqa: BLE001
                    logger.warning("[blend] brake escalation failed: %s", exc)
        if not ok:
            send = getattr(self.robot, "_send_script", None)
            if callable(send):
                try:
                    ok = bool(send("stopj(%.2f)" % self.BLEND_ABORT_DECEL))
                except Exception as exc:  # noqa: BLE001
                    logger.warning("[blend] stopj send failed: %s", exc)
        self._retire_motion_lease()
        return ok

    def _wait_for_blended_path(self, rows, epoch, start_pos=None, q_start=None) -> bool:
        """Block until a blended URScript path lands. True only on arrival.

        Arrival is the FINAL waypoint's joint target, nothing else.
        Intermediate waypoints are never reached (the arm cuts each corner at
        distance r), so a per-waypoint joint match would never fire and a
        per-waypoint Cartesian test would never come within 5 mm. The last row
        is forced to r=0 (waypoints.resolve_blend_radii) so the terminal
        waypoint is a true stop.

        Stillness is evaluated over the WHOLE path, never per waypoint, and
        may only mean "arrived" if motion has been observed; a path's upload
        lag is longer than one move's.
        """
        n = len(rows)
        final = rows[-1]
        target = final.position
        q_target = np.asarray(final.q, dtype=float)
        register_ok = self._read_blend_progress() is not None
        if not register_ok:
            logger.warning(
                "[blend] progress register %d unreadable: no mid-path stall "
                "detection, only the whole-path budget",
                self._blend_register(),
            )

        # Budget the WHOLE path, not one move: 15 s is both too long for a
        # 2 cm reposition and too short for a 5-row transport. Same shape as
        # _move_to_linear's per-move budget (3x ideal + start lag + slack).
        # Without q_start the first segment is budgeted off a half-turn rather
        # than under-budgeted.
        ideal = 0.0
        q_prev = None if q_start is None else np.asarray(q_start, dtype=float)
        for row in rows:
            dq = (
                float(np.max(np.abs(row.q - q_prev[: row.q.size]))) if q_prev is not None else np.pi
            )
            ideal += joint_move_seconds(dq, row.velocity, row.acceleration)
            q_prev = row.q
        timeout = float(
            np.clip(
                3.0 * ideal + self.URSCRIPT_START_LAG_S + 1.0,
                self._min_motion_budget_s(),
                self.BLEND_TIMEOUT_CAP_S,
            )
        )

        t0 = time.time()
        motion = _StillnessTracker(
            self._get_current_position(),
            t0,
            sample_s=self.MOTION_SAMPLE_S,
            still_samples=self.MOTION_STILL_SAMPLES,
        )
        next_record = t0
        progress = None
        progress_at = t0

        def _arrived(why: str) -> bool:
            self._retire_motion_lease()
            logger.info(
                "[blend] path complete after %.2fs (%s, %d rows)",
                time.time() - t0,
                why,
                n,
            )
            return True

        while time.time() - t0 < timeout:
            if self._abort:
                # Reach the arm BEFORE raising (see _halt_arm); otherwise an
                # aborted blended path keeps running to its last waypoint
                # after the executor has unwound.
                self._halt_arm()
            self._check_abort()
            time.sleep(self.MOTION_POLL_S)
            now = time.time()
            self._refresh_motion_lease()
            tcp = self._get_current_position()
            if self._recorder and now >= next_record:
                self._recorder.record()
                next_record = now + self.MOTION_SAMPLE_S

            # 1. Joints match the FINAL commanded row: unambiguously arrived.
            try:
                q = np.asarray(self.robot.get_joint_positions(), dtype=float)
                if np.max(np.abs(q[: q_target.size] - q_target)) < self.JOINT_ARRIVE_TOL:
                    return _arrived("joint target")
            except Exception:  # noqa: BLE001 - fall through to the other tests
                pass
            # 2. Cartesian proximity to the final waypoint.
            if np.linalg.norm(tcp - target) < 0.005:
                return _arrived("cartesian proximity")

            motion.update(tcp, now)

            # 3. Progress register. `n` means the program ran past its last
            #    movej, so the controller itself reports the path finished.
            #    Trustworthy only because _next_blend_epoch stepped off
            #    whatever value the register was already holding.
            k = decode_blend_progress(self._read_blend_progress(), epoch, n)
            if k is not None and (progress is None or k > progress):
                progress = k
                progress_at = now
                if k >= n:
                    return _arrived("progress register")
            # 4. Stall: started, still before the LAST row, register frozen,
            #    TCP still. Intermediate rows only: a stopped arm on the final
            #    row is the ordinary end of a move, handled by (5).
            if (
                progress is not None
                and progress < n - 1
                and now - progress_at > self.BLEND_STALL_S
                and motion.stopped()
            ):
                logger.warning(
                    "[blend] STALLED at row %d/%d after %.1fs (register frozen, "
                    "TCP still); stopping the path",
                    progress,
                    n,
                    now - t0,
                )
                self._halt_arm()
                return False
            # 5. Whole-path stillness, gated on observed motion (never an exit
            #    without ever_moved). Also refused while the register says the
            #    program is still inside its rows: a mid-path stop looks like a
            #    finished path from the outside. On the LAST row, or with a
            #    mute register, this is the per-move wait's test.
            settled = motion.stopped() and now - t0 > self.URSCRIPT_START_LAG_S
            if settled and motion.ever_moved and (progress is None or progress >= n - 1):
                return _arrived("settled after observed motion")

        logger.warning(
            "[blend] %.1fs timeout, no arrival signal (moved=%s, row=%s/%d, "
            "dist=%.3fm); stopping the path",
            timeout,
            motion.ever_moved,
            progress,
            n,
            float(np.linalg.norm(self._get_current_position() - target)),
        )
        self._halt_arm()
        return False

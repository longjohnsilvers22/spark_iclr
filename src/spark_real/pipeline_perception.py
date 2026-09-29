import copy
import logging
import os
import re
import time
from collections import defaultdict
from typing import Dict, List

import cv2
import numpy as np
from PIL import Image, ImageDraw, ImageFont

from spark_real.utils.env_flags import as_bool
from spark_real.config import family_block
from spark_real.perception.detection_fusion import (
    DetectionGate,
    FusionConfig,
    box_iou,
    mask_bbox,
)
from spark_real.perception import dedup
from spark_real.perception.spark_perception import ObjectDetection
from spark_real.perception.slot_detector import detect_slots
from spark_real.perception.prompt_registry import (
    TaskSpec,
    base_label,
    load_registry,
    resolve_labels,
)
from spark_real.calibration import compute_wrist_camera_extrinsic
from spark_real.routes import state as _routes_state

logger = logging.getLogger(__name__)


# Operator-facing modes of the second-opinion box proposer, a view of the
# `perception.fusion.proposer` block: `auto` is whatever that block says,
# `always` drops its allowlist, `off` swaps in the null backend. The master
# `perception.fusion.enabled` switch is untouched by all three, so the
# mask-quality channel (which vetoed a bogus 69 deg wrist rotation off a
# meaningless plushie OBB) stays up even in `off`.
PROPOSER_MODE_AUTO = "auto"
PROPOSER_MODE_ALWAYS = "always"
PROPOSER_MODE_OFF = "off"
PROPOSER_MODES = (PROPOSER_MODE_AUTO, PROPOSER_MODE_ALWAYS, PROPOSER_MODE_OFF)

# Backend names box_proposals.build_proposer treats as "no detector".
_NULL_BACKENDS = ("null", "off", "none", "")

# An allowlist entry no real label can canonicalise to, so `off` holds even if
# a backend gets built anyway (SPARK_BOX_PROPOSER bypasses the backend key
# inside build_proposer): proposer_trusts() is False for every label and every
# detection takes the ABSTAIN path, confidence bit-identical.
_PROPOSER_OFF_LABEL = "__proposer_off__"


def normalize_proposer_mode(mode):
    """A member of PROPOSER_MODES, or None for "no override".

    Raises ValueError on anything else.
    """
    if mode is None:
        return None
    m = str(mode).strip().lower()
    if not m:
        return None
    if m not in PROPOSER_MODES:
        raise ValueError(f"unknown proposer mode {mode!r}; known: {list(PROPOSER_MODES)}")
    return m


# Capture, SAM3 detection, multi-view merge, and annotation.
# Mixed into the pipeline class.
# Nothing resting on the table stands taller than this. A detection above
# it is a depth failure, not a tall object, and is worth correcting from
# the other camera even on weak evidence.
MAX_ON_TABLE_HEIGHT_M = 0.35


class PerceptionMixin:

    # Count-gate outcome of the most recent spec-driven merge_detections.
    # Class-level default because this is a mixin with no __init__ of its own.
    _last_resolution = None
    _last_captures = None
    _last_captures_t = None
    _prompt_registry = None

    # --- Detector fusion / ASPIRE quality gate (config-gated, default OFF) ---
    # `None` = not resolved yet, `False` = resolved and OFF, otherwise the gate.
    _fusion_gate = None
    _fusion_gate_error = None

    FUSION_ENV = "SPARK_DETECTION_FUSION"
    PROPOSER_MODE_ENV = "SPARK_PROPOSER_MODE"

    @staticmethod
    def _config_proposer_mode(proposer: Dict) -> str:
        """Which of the three modes a `proposer` block already describes.

        Derived, never stored (the YAML has no `mode:` key). A null/disabled
        backend is `off`, an allowlist is `auto`, no allowlist is `always`.
        """
        backend = str(proposer.get("backend", "null")).lower()
        if backend in _NULL_BACKENDS or not proposer.get("enabled", False):
            return PROPOSER_MODE_OFF
        return PROPOSER_MODE_AUTO if proposer.get("labels") else PROPOSER_MODE_ALWAYS

    def proposer_mode_override(self):
        """`(mode, source)` for a mode somebody forced, else `(None, "config")`.

        Resolution order is YAML -> env -> runtime override, override last.
        `None` = fall back to config.
        """
        for value, source in (
            (getattr(_routes_state, "proposer_mode", None), "runtime"),
            (os.environ.get(self.PROPOSER_MODE_ENV), "env"),
        ):
            try:
                mode = normalize_proposer_mode(value)
            except ValueError as exc:
                # A typo in a shell env var must not take detection down.
                logger.warning("[fusion] ignoring %s proposer mode (%s)", source, exc)
                continue
            if mode is not None:
                return mode, source
        return None, "config"

    def _apply_proposer_mode(self, cfg: Dict, mode: str) -> Dict:
        """Rewrite the `proposer` block so `mode` is what the gate is built with."""
        if mode == PROPOSER_MODE_AUTO:
            # By definition the config's own behaviour: touch nothing.
            return cfg
        proposer = dict(cfg.get("proposer") or {})
        if mode == PROPOSER_MODE_OFF:
            proposer["enabled"] = False
            proposer["backend"] = "null"
            proposer["labels"] = [_PROPOSER_OFF_LABEL]
        else:  # PROPOSER_MODE_ALWAYS: pre-allowlist behaviour
            proposer["enabled"] = True
            proposer["labels"] = []
        cfg = dict(cfg)
        cfg["proposer"] = proposer
        return cfg

    def fusion_status(self) -> Dict:
        """Everything the UI shows about the gate. Never builds it.

        `loaded_backend` comes off the cached gate and stays None until a
        detect has actually built one; a status poll must not pull an RF-DETR
        checkpoint onto the GPU.
        """
        settings = self._fusion_settings()
        cfg = FusionConfig.from_dict(settings)
        proposer = dict(settings.get("proposer") or {})
        override, source = self.proposer_mode_override()
        gate = self._fusion_gate or None

        if self._fusion_gate_error is not None:
            gate_state = "error"
        elif not cfg.enabled:
            gate_state = "off"
        elif gate is not None:
            gate_state = "on"
        else:
            gate_state = "unbuilt"

        return {
            "mode": self._config_proposer_mode(proposer),
            "override": override,
            "source": source,
            "modes": list(PROPOSER_MODES),
            # read-only: the whole fusion gate, and what is behind it
            "fusion_enabled": bool(cfg.enabled),
            "gate_state": gate_state,
            "backend": str(proposer.get("backend", "null")).lower(),
            "loaded_backend": getattr(getattr(gate, "proposer", None), "name", None),
            "labels": [lbl for lbl in cfg.proposer_labels() if lbl != _PROPOSER_OFF_LABEL],
            "error": (
                str(self._fusion_gate_error) if self._fusion_gate_error is not None else None
            ),
        }

    def reset_detection_gate(self) -> None:
        """Drop the memoised gate so the next detect() rebuilds it.

        detection_gate() caches on first use (including the cached-OFF and
        cached-ERROR states), so ANY runtime change to the fusion settings that
        does not come through here is inert until the server restarts.
        """
        self._fusion_gate = None
        self._fusion_gate_error = None
        logger.info("[fusion] gate cache cleared; next detect rebuilds it")

    def _fusion_settings(self) -> Dict:
        """Resolve the `perception.fusion` block, env last.

        Same resolution order ScoreExecutor._apply_family_tuning uses:
        profile.raw when the server loaded an overlay, else the packaged
        configs/<family>_default.yaml. `perception.reprompt_min_conf` and
        `reprompt_max_attempts` are this gate's knobs.
        """
        family = (getattr(self.config, "robot_family", "ur10e") or "ur10e").lower()
        per = dict(family_block(getattr(self, "profile", None), family, "perception"))
        cfg = dict((per.get("fusion") or {}))
        if per.get("reprompt_min_conf") is not None:
            cfg.setdefault("low_quality_conf", float(per["reprompt_min_conf"]))
        if per.get("reprompt_max_attempts") is not None:
            cfg.setdefault("reprompt_max_attempts", int(per["reprompt_max_attempts"]))

        env = os.environ.get(self.FUSION_ENV)
        if env is not None:
            cfg["enabled"] = as_bool(env, False)

        # Last, so it wins: the operator's proposer-mode override. It only ever
        # rewrites the `proposer` sub-block, so `enabled` -- the mask-quality
        # gate itself -- is not something this switch can take down.
        mode, _source = self.proposer_mode_override()
        if mode is not None:
            cfg = self._apply_proposer_mode(cfg, mode)
        return cfg

    def detection_gate(self):
        """The live fusion gate, or None when it is switched off.

        Turned on by `perception.fusion.enabled: true` in the family YAML or by
        SPARK_DETECTION_FUSION=1. A backend that was asked for and cannot be
        built raises here and keeps raising.
        """
        if self._fusion_gate_error is not None:
            raise self._fusion_gate_error
        if self._fusion_gate is not None:
            return self._fusion_gate or None

        cfg = FusionConfig.from_dict(self._fusion_settings())
        if not cfg.enabled:
            self._fusion_gate = False
            return None
        try:
            gate = DetectionGate(cfg)
        except Exception as exc:  # noqa: BLE001 - re-raised, never swallowed
            self._fusion_gate_error = RuntimeError(
                f"detection fusion is enabled but its box proposer is unusable: {exc}"
            )
            logger.error("[fusion] %s", self._fusion_gate_error)
            raise self._fusion_gate_error from exc

        self._fusion_gate = gate
        logger.info(
            "[fusion] gate ON: proposer=%s, low_quality_conf=%.2f, "
            "reprompt_max_attempts=%d, write_confidence=%s",
            getattr(gate.proposer, "name", "?"),
            cfg.low_quality_conf,
            cfg.reprompt_max_attempts,
            cfg.write_confidence,
        )
        return gate

    def table_z_floor(self) -> float:
        """Work-surface Z in the base frame for perception plausibility checks.
        config.table_height > executor TABLE_Z_FLOOR > profile YAML > class literal.
        """
        floor = getattr(getattr(self, "config", None), "table_height", None)
        if floor is None:
            floor = getattr(getattr(self, "_executor", None), "TABLE_Z_FLOOR", None)
        if floor is None:
            profile = getattr(self, "profile", None)
            if profile is not None:
                floor = profile.control().get("table_z_floor", profile.raw.get("table_z_floor"))
        if floor is None:
            from spark_real.control.executor_core import ScoreExecutorCore

            floor = ScoreExecutorCore.TABLE_Z_FLOOR
        return float(floor)

    def _detect_camera(
        self, rgb, depth, cal, prompts, multi_instance_set, secondary_score
    ) -> List[ObjectDetection]:
        """One camera, one SAM3 pass. Extracted so the ASPIRE reprompt can run
        the SAME detection code the first pass ran."""
        if depth is not None and self.config.use_hardware_depth:
            K = None
            table_z = None
            if not np.allclose(cal.extrinsic, np.eye(4)):
                K = cal.intrinsic_matrix
                table_z = self.table_z_floor()
            return self._perception._detect_with_rendered_depth(
                rgb=rgb,
                depth=depth,
                prompts=prompts,
                cam_pos=cal.position,
                cam_mat=cal.rotation_matrix,
                cam_fovy=cal.fovy_degrees,
                w=cal.width,
                h=cal.height,
                multi_instance_prompts=multi_instance_set,
                intrinsic_matrix=K,
                table_height=table_z,
                secondary_score=secondary_score,
            )
        return self._perception.detect(
            rgb=rgb,
            prompts=prompts,
            cam_pos=cal.position,
            cam_mat=cal.rotation_matrix,
            cam_fovy=cal.fovy_degrees,
        )

    def _fusion_redetect(self, gate, rgb, depth, cal, secondary_score):
        """Build the ASPIRE `redetect(prompt, box)` callback for one camera.

        The alternate prompt is a real SAM3 pass through _detect_camera, so a
        reprompt result carries a real mask and a real backprojected position.
        A `box` (a second detector's partner box) is honoured by asking for
        every instance of the prompt and keeping the one that overlaps the box.
        """

        def redetect(prompt, box=None):
            multi = {prompt} if box is not None else set()
            cands = self._detect_camera(rgb, depth, cal, [prompt], multi, secondary_score)
            if not cands:
                return None
            if box is not None:
                overlapping = [
                    c
                    for c in cands
                    if box_iou(mask_bbox(c.mask) or c.bbox or (0, 0, 0, 0), box)
                    >= gate.cfg.assoc_iou
                ]
                cands = overlapping or []
            if not cands:
                return None
            return max(cands, key=lambda c: float(c.confidence or 0.0))

        return redetect

    def _apply_fusion(self, gate, detections, rgb, depth, cal, cam_name, secondary_score) -> None:
        """Fuse + ASPIRE-repair one camera's detections, in place."""
        try:
            results = gate.apply(
                detections,
                rgb=rgb,
                redetect=self._fusion_redetect(gate, rgb, depth, cal, secondary_score),
            )
        except Exception as exc:  # noqa: BLE001 - the gate must never stop perception
            logger.warning(
                "[fusion] gate pass failed on %s (%s); detections unchanged",
                cam_name,
                exc,
            )
            return
        for res in results:
            logger.info("[fusion] %s %s", cam_name, res.describe())

    def capture(self) -> Dict[str, Dict]:
        """
        Capture RGB + depth from all active cameras.

        Returns dict of camera_name -> {"rgb": ndarray, "depth": ndarray, "calibration": CameraCalibration}
        """
        captures = {}
        # Per-camera device-read and lock-wait costs. A stream client holding
        # a Kinect read lock shows up here as *_lock, distinct from a slow
        # sensor (*_read); without the split the two are indistinguishable.
        _cap_waits = {}
        _t_capture = time.perf_counter()

        def _scaled(depth, cal):
            """
            Apply the per-camera depth correction z = z*scale + offset
            (sensor bias baked in by the RGB-D Procrustes hand-eye). Depth is
            in meters here, so depth_offset is in meters. The math lives on
            CameraCalibration.correct_depth so the streaming capture path
            (routes/streaming.py) applies the identical correction.
            """
            if depth is None or cal is None:
                return depth
            return cal.correct_depth(depth)

        if self._kinect is not None:
            _t_lock = time.perf_counter()
            with self._kinect_read_lock:
                _cap_waits["sideview_lock"] = time.perf_counter() - _t_lock
                _t_read = time.perf_counter()
                rgb, depth = self._kinect.read()
                _cap_waits["sideview_read"] = time.perf_counter() - _t_read
            if rgb is not None:
                depth = _scaled(depth, self._kinect_cal)
                captures["sideview"] = {
                    "rgb": rgb,
                    "depth": depth,
                    "calibration": self._kinect_cal,
                }

        if self._kinect2 is not None:
            _t_lock = time.perf_counter()
            with self._kinect2_read_lock:
                _cap_waits["birdview_lock"] = time.perf_counter() - _t_lock
                _t_read = time.perf_counter()
                rgb, depth = self._kinect2.read()
                _cap_waits["birdview_read"] = time.perf_counter() - _t_read
            if rgb is not None:
                depth = _scaled(depth, self._kinect2_cal)
                captures["birdview"] = {
                    "rgb": rgb,
                    "depth": depth,
                    "calibration": self._kinect2_cal,
                }

        if self._realsense is not None:
            _t_lock = time.perf_counter()
            with self._realsense_read_lock:
                _cap_waits["wrist_lock"] = time.perf_counter() - _t_lock
                _t_read = time.perf_counter()
                rgb, depth = self._realsense.read()
                _cap_waits["wrist_read"] = time.perf_counter() - _t_read
            if rgb is not None:
                depth = _scaled(depth, self._realsense_cal)
                # Update wrist cam extrinsic from robot FK
                if self._robot is not None and self._realsense_cal is not None:
                    try:
                        obs = self._robot.get_observation()
                        tcp = obs["tcp_pose"]
                        self._realsense_cal.extrinsic = compute_wrist_camera_extrinsic(
                            tcp_pose=tcp,
                            tool_offset=self._wrist_tool_offset,
                        )
                    except Exception:
                        pass
                captures["wrist"] = {
                    "rgb": rgb,
                    "depth": depth,
                    "calibration": self._realsense_cal,
                }

        # Stash the frames for consumers that run after perception but need
        # this run's pixels (robointer_gate's out-of-band correction source,
        # er2/molmo). Read-only, fail-open: absent stash means no correction.
        self._last_captures = captures
        # When these frames were taken. Verification reuses them inside a
        # configured freshness window instead of re-grabbing every camera; see
        # control/verify_scope.captures_for_verify.
        self._last_captures_t = time.time()

        _cap_total = time.perf_counter() - _t_capture
        logger.info(
            "[detect-timing] capture: %d camera(s) in %.2fs | %s",
            len(captures),
            _cap_total,
            " ".join(f"{k}={v:.2f}s" for k, v in sorted(_cap_waits.items()))
            or "no cameras",
        )

        return captures

    def detect(
        self,
        captures: Dict,
        prompts: List[str],
        multi_instance: bool = False,
        multi_instance_prompts=None,
        secondary_score=None,
        use_fusion: bool = True,
    ) -> List[ObjectDetection]:
        """
        Run SAM3 detection on captured images with hardware depth.

        Supports multi-instance detection: if SAM3 finds multiple masks
        for a prompt (e.g., 3 forks), they are numbered ("fork 1", "fork 2",
        "fork 3") so the planner and executor can address each one.

        `multi_instance` promotes every prompt; `multi_instance_prompts`
        promotes only the named ones, which is what a task spec supplies so
        that a container prompt does not start returning phantom instances
        just because a utensil prompt needs to. `secondary_score` overrides
        the score a non-best mask must clear to count as another instance.

        Returns all per-camera detections (for visualization across all tiles).
        Use detect_merged() for the executor (one best per label).
        """
        all_detections = []
        detect_cams = {k: v for k, v in captures.items()}
        # Resolved once per detect: an enabled-but-unusable proposer raises out
        # of here rather than degrading to a detector that sees nothing.
        # `use_fusion=False` skips the gate entirely, ASPIRE reprompts included
        # (extra SAM3 passes per detection); the verify path asks for this
        # because no predicate reads a fusion field. See control/verify_scope.
        gate = self.detection_gate() if use_fusion else None

        # Sort: calibrated cameras first (better 3D positions)
        cam_order = sorted(
            detect_cams.keys(),
            key=lambda c: (
                0 if not np.allclose(detect_cams[c]["calibration"].extrinsic, np.eye(4)) else 1
            ),
        )

        # Multi-instance: return all confident masks per prompt (batch tasks).
        # Single-instance: top-1 per prompt (user-specified objects).
        if multi_instance_prompts is not None:
            multi_instance_set = set(multi_instance_prompts)
        else:
            multi_instance_set = set(prompts) if multi_instance else set()

        for cam_name in cam_order:
            data = detect_cams[cam_name]
            rgb = data["rgb"]
            depth = data["depth"]
            cal = data["calibration"]

            # A camera without depth (e.g. wrist that opened on a USB2
            # link) must not fall through to the monocular-depth path:
            # the DepthAnything3 fallback raises when it is not installed
            # and kills the whole detect.
            if depth is None and self.config.use_hardware_depth:
                logger.warning("detect: skipping '%s' (no hardware depth)", cam_name)
                continue

            _t_cam = time.perf_counter()
            detections = self._detect_camera(
                rgb, depth, cal, prompts, multi_instance_set, secondary_score
            )
            _t_detect = time.perf_counter() - _t_cam

            for det in detections:
                det.camera = cam_name
            # THE FUSION SEAM. Stamps fused_confidence / axis_trust /
            # low_quality onto every detection before it can reach the merge,
            # the planner or the executor.
            _t_fuse = time.perf_counter()
            if gate is not None and detections:
                self._apply_fusion(gate, detections, rgb, depth, cal, cam_name, secondary_score)
            _t_fuse = time.perf_counter() - _t_fuse
            # One line per camera, always. The fusion gate's own lines are
            # emitted only after gate.apply() returns, so without this a slow
            # gate pass (31 s per camera has been seen) is indistinguishable
            # from a slow detect.
            logger.info(
                "[detect-timing] camera '%s': detect=%.2fs fusion=%.2fs "
                "total=%.2fs (%d det)",
                cam_name,
                _t_detect,
                _t_fuse,
                _t_detect + _t_fuse,
                len(detections),
            )
            all_detections.extend(detections)

        # Filter out physically impossible detections (below table surface)
        TABLE_Z_MIN = -0.28
        before = len(all_detections)
        all_detections = [
            d for d in all_detections if d.position_3d is None or d.position_3d[2] > TABLE_Z_MIN
        ]
        if len(all_detections) < before:
            logger.warning(
                "Filtered %d detections below table (Z < %.2f)",
                before - len(all_detections),
                TABLE_Z_MIN,
            )

        # Don't number yet; numbering happens after cross-camera merge
        logger.info("Detected %d total across %d cameras", len(all_detections), len(captures))
        return all_detections

    MERGE_PROXIMITY_M = 0.15  # 15cm, loose enough for cross-camera calibration drift

    def merge_detections(
        self,
        detections: List[ObjectDetection],
        spec: "TaskSpec" = None,
    ) -> List[ObjectDetection]:
        """
        Merge detections across cameras. Birdview is the primary authority.

        1. Start with birdview (top-down, sees everything) as the base set
        2. For each sideview detection, find closest same-label birdview match
        3. Matched: keep birdview detection (correct pixel coords for overlay)
        4. Unmatched: add only if birdview has no instance of that label
        5. Number multi-instance, normalize Z
        6. Uses copy.copy so original detections aren't mutated

        `spec` is an optional :class:`TaskSpec` from the prompt registry. When
        given it replaces confidence-rank instance numbering with the task's
        declared geometric order and honours its per-instance-Z setting, so a
        cached behaviour tree's "<label> 1" refers to the same physical object
        on every run. Omitting it preserves the legacy behaviour exactly.
        """
        by_camera = defaultdict(list)
        for det in detections:
            by_camera[getattr(det, "camera", "unknown")].append(det)

        # Primary camera: birdview > sideview (birdview sees entire table).
        # SPARK_PRIMARY_CAM overrides the preference order at runtime (e.g.
        # =sideview to trust sideview's hand-eye during a calibration test);
        # unset restores the birdview-first default. Purely a debug toggle.
        _pref = os.environ.get("SPARK_PRIMARY_CAM", "").strip().lower()
        cam_pref = [_pref, "birdview", "sideview"] if _pref else ["birdview", "sideview"]
        primary_cam = None
        for cam in cam_pref:
            if cam in by_camera and by_camera[cam]:
                primary_cam = cam
                break
        if primary_cam is None and by_camera:
            primary_cam = next(iter(by_camera))
        if not primary_cam:
            return []

        # Base = copies of primary camera detections.
        # position_agentview keeps the primary camera's own backprojection
        # even after the sideview Z-override / wrist fusion rewrites
        # position_3d. Any later same-camera displacement measurement (sticky
        # gate, redetect teleport gate, scene diff) must compare against this,
        # not the fused position: the cameras disagree by a systematic 1-2 cm,
        # which otherwise reads as a phantom 'moved'. position_3d is
        # re-materialized as a fresh array because copy.copy is shallow.
        merged = []
        for d in by_camera[primary_cam]:
            if d.position_3d is None:
                continue
            c = copy.copy(d)
            c.position_3d = np.array(d.position_3d, dtype=float)
            c.position_agentview = np.array(d.position_3d, dtype=float)
            merged.append(c)
        logger.info(
            "Merge: primary=%s (%d dets), secondary cameras: %s",
            primary_cam,
            len(merged),
            [f"{c}({len(ds)})" for c, ds in by_camera.items() if c != primary_cam],
        )

        # Add sideview-only detections that birdview missed entirely.
        primary_labels = {d.label for d in merged}
        for cam, dets_c in by_camera.items():
            if cam == primary_cam:
                continue
            for d in dets_c:
                if d.label not in primary_labels and d.position_3d is not None:
                    c = copy.copy(d)
                    c.position_3d = np.array(d.position_3d, dtype=float)
                    # Secondary-only detection: its "same-source" anchor is
                    # its own camera's backprojection at merge time.
                    c.position_agentview = np.array(d.position_3d, dtype=float)
                    merged.append(c)
                    primary_labels.add(d.label)
                    logger.info(
                        "Merge: added %s-only detection '%s' " "(conf=%.2f, z=%.1fmm)",
                        cam,
                        d.label,
                        d.confidence,
                        d.position_3d[2] * 1000 if d.position_3d is not None else 0,
                    )

        # Birdview owns POSITION: it has the most accurate hand-eye.
        #
        # OBB enrichment from secondary cameras: birdview's mask of a
        # flat container (tray, plate) viewed straight-down can be roughly
        # square (low AR) or noisy because the container's vertical sides
        # project to a thin rim and SAM3's mask edges wobble at low
        # contrast. Sideview sees the footprint at an angle and gets a
        # cleaner OBB. For PLACEMENT we want yaw alignment, so keep
        # birdview position but enrich orientation_angle/aspect_ratio/
        # obb_minor_m from whichever camera has the strongest AR for the
        # same label.
        # Minimum SAM3 confidence for a secondary-camera detection to be
        # trusted for OBB enrichment; below this is treated as a noise mask.
        OBB_ENRICH_MIN_CONF = 0.30

        # Cameras eligible to PROVIDE OBB info. Wrist is excluded: its
        # hand-eye is the noisiest so its world-frame orientation can't be
        # trusted even at high SAM3 confidence. OBB enrichment is restricted
        # to a single camera (birdview) because instance numbers are
        # per-camera, so the same label string can refer to a different
        # object across cameras and cross-wire the grasp. Single-camera OBB
        # keeps the instance IDs self-consistent.
        OBB_ENRICH_TRUSTED_CAMS = {"birdview"}

        # Max world-XY distance (meters) between a merged instance and a
        # trusted-camera detection for that detection's OBB to enrich the
        # instance. Prevents applying one birdview OBB to a different
        # same-label instance across the table (two spoons at different
        # angles each get their own nearest birdview OBB). Env-overridable.
        OBB_ENRICH_MAX_XY_M = float(
            os.environ.get("OBB_ENRICH_MAX_XY_M", "0.12")  # 12 cm default gate
        )

        # Per-instance OBB enrichment via greedy nearest-neighbor matching:
        # for each trusted donor camera, match each merged instance to the
        # nearest same-label donor detection by world XY (closest pair first,
        # each donor consumed by at most one instance). Instances with no
        # donor inside the gate keep their own OBB; the AR-improvement gate
        # below means a donor only wins when its OBB is stronger.
        for cam, dets_c in by_camera.items():
            if cam == primary_cam or cam not in OBB_ENRICH_TRUSTED_CAMS:
                continue
            donors = [
                x
                for x in dets_c
                if x.position_3d is not None and float(x.confidence or 0.0) >= OBB_ENRICH_MIN_CONF
            ]
            if not donors:
                continue
            # Candidate (distance, merged_idx, donor_idx) pairs within gate,
            # same label only.
            pairs = []
            for mi, d in enumerate(merged):
                if d.position_3d is None:
                    continue
                m_xy = np.array(d.position_3d[:2], dtype=float)
                for di, cand in enumerate(donors):
                    if cand.label != d.label:
                        continue
                    dist = float(np.linalg.norm(m_xy - np.array(cand.position_3d[:2], dtype=float)))
                    if dist <= OBB_ENRICH_MAX_XY_M:
                        pairs.append((dist, mi, di))
            # Greedy nearest-first assignment; each merged instance and each
            # donor used at most once.
            pairs.sort(key=lambda p: p[0])
            used_m = set()
            used_d = set()
            for dist, mi, di in pairs:
                if mi in used_m or di in used_d:
                    continue
                used_m.add(mi)
                used_d.add(di)
                d = merged[mi]
                cand = donors[di]
                cand_ar = float(cand.aspect_ratio or 1.0)
                # Only enrich when the donor's OBB is stronger (higher AR)
                # than what the instance already has.
                if cand_ar <= float(d.aspect_ratio or 1.0):
                    continue
                logger.info(
                    "  OBB-enrich '%s': orient %.3f->%.3f rad, AR %.2f->%.2f "
                    "(from %s @ %.1fcm; position stays %s)",
                    d.label,
                    float(d.orientation_angle or 0.0),
                    float(cand.orientation_angle or 0.0),
                    float(d.aspect_ratio or 1.0),
                    cand_ar,
                    cam,
                    dist * 100.0,
                    d.camera,
                )
                d.orientation_angle = cand.orientation_angle
                d.aspect_ratio = cand_ar
                # The fusion gate's axis_trust scored the DONOR's mask axis, so
                # it has to travel with the axis it scored. Without this the
                # instance wears one camera's angle and another camera's trust,
                # and a wrist can be aimed off an axis nothing vouched for.
                # Donor unmeasured -> clear it: absent means unknown, and the
                # base's own trust no longer describes the axis being carried.
                d.axis_trust = getattr(cand, "axis_trust", None)
                # world-frame major axis follows the donor's OBB when present.
                if getattr(cand, "world_major_axis_rad", None) is not None:
                    d.world_major_axis_rad = cand.world_major_axis_rad
                # Keep the primary's width if the enriching camera has none:
                # sideview often reports obb_minor_m=0.
                cand_obb_minor = float(cand.obb_minor_m or 0.0)
                if cand_obb_minor > 0:
                    d.obb_minor_m = cand_obb_minor

        # Z override from secondary camera: when the two cameras disagree on
        # height, the primary's depth is wrong (Kinect IR averages table +
        # object pixels for small/shiny/translucent surfaces from above,
        # producing Z anywhere from table level to mid-object) and sideview
        # resolves height much better. No "near table" gate; compare the two
        # cameras directly. Bidirectional: birdview over-reads as easily as it
        # under-reads (a tray measured at z=+0.071 against a table at ~-0.27,
        # 34 cm in the air).
        Z_ELEVATION_THRESH = 0.05  # cameras must disagree by >=50mm
        for d in merged:
            if d.position_3d is None:
                continue
            pri_z = float(d.position_3d[2])
            for cam, dets_c in by_camera.items():
                if cam == primary_cam:
                    continue
                # A detection implausibly far above the table cannot be right
                # for an object resting on it, so widen the search: accept a
                # base-label match at any confidence (sideview has called a
                # tray 'purple tray grip' at 0.12, which an exact-label 0.20
                # gate rejected).
                implausible = pri_z > self.table_z_floor() + MAX_ON_TABLE_HEIGHT_M
                same = [
                    x
                    for x in dets_c
                    if x.label == d.label
                    and x.position_3d is not None
                    and float(x.confidence or 0) >= 0.20
                ]
                if not same and implausible:
                    base = str(d.label).split()[0].lower()
                    same = [
                        x
                        for x in dets_c
                        if x.position_3d is not None
                        and str(x.label).lower().startswith(base)
                    ]
                    if same:
                        logger.warning(
                            "  '%s' at z=%.0fmm is >%.0fmm above the table; "
                            "accepting a loose '%s' match from %s to correct it",
                            d.label, pri_z * 1000,
                            MAX_ON_TABLE_HEIGHT_M * 1000, base, cam,
                        )
                if not same:
                    continue
                sec = max(same, key=lambda x: float(x.confidence or 0))
                sec_z = float(sec.position_3d[2])
                if abs(sec_z - pri_z) > Z_ELEVATION_THRESH:
                    logger.info(
                        "  Z-override '%s': %s z=%.0fmm -> %s z=%.0fmm "
                        "(%+.0fmm, sideview wins)",
                        d.label,
                        primary_cam,
                        pri_z * 1000,
                        cam,
                        sec_z * 1000,
                        (sec_z - pri_z) * 1000,
                    )
                    d.position_3d[2] = sec_z

        # Per-camera detection counts (kept for diagnostics).
        for cam in list(by_camera.keys()):
            if cam != primary_cam:
                logger.info(
                    "  Secondary camera %s contributed %d detection(s) "
                    "(used for OBB enrichment + Z override; position XY from %s)",
                    cam,
                    len(by_camera[cam]),
                    primary_cam,
                )

        # Filter low confidence. The floor is low because cloth sub-parts
        # (shirt collar, hem) ground at very low SAM3 confidence on plain
        # fabric while still landing in the correct region of the body
        # mask. The 3D backprojection + downstream verify catch any
        # mislocalised low-conf detections, so the low floor buys
        # part-grounding on fabric without producing false placements.
        MIN_MERGED_CONFIDENCE = 0.05
        before_filter = len(merged)
        merged = [d for d in merged if d.confidence >= MIN_MERGED_CONFIDENCE]
        if len(merged) < before_filter:
            logger.info(
                "Dropped %d merged detections below %.0f%% confidence",
                before_filter - len(merged),
                MIN_MERGED_CONFIDENCE * 100,
            )

        # Cross-label spatial dedup: SAM3 often detects the same object for
        # multiple prompts (e.g., "knife handle" mask on a spoon). If two
        # detections from different labels are the same physical object,
        # keep the higher-confidence one.
        #
        # "Same physical object" is decided by perception.dedup, not by XY
        # proximity alone: an object resting in a container is centimetres from
        # the container's centroid, and deleting it here makes
        # inside(obj, container) unverifiable. See perception/dedup.py.
        CROSS_LABEL_DEDUP_M = 0.05  # 5cm; same physical object
        aware = dedup.containment_aware()
        before_dedup = len(merged)
        deduped = []
        for d in merged:
            if d.position_3d is None:
                deduped.append(d)
                continue
            duplicate = False
            for i, existing in enumerate(deduped):
                if existing.position_3d is None:
                    continue
                # Part-of guard: never dedup a part against its parent
                # ('mug handle' vs 'mug'). The parent mask includes the
                # part so their centroids collide, which would otherwise
                # delete the very label the planner needs to grasp.
                if d.label in existing.label or existing.label in d.label:
                    continue
                if dedup.same_object_3d(d, existing, CROSS_LABEL_DEDUP_M, aware=aware):
                    if d.confidence > existing.confidence:
                        logger.info(
                            "  Cross-label dedup: %s (%.0f%%) replaces %s (%.0f%%) at same pos",
                            d.label,
                            d.confidence * 100,
                            existing.label,
                            existing.confidence * 100,
                        )
                        deduped[i] = d
                    else:
                        logger.info(
                            "  Cross-label dedup: dropping %s (%.0f%%), kept %s (%.0f%%)",
                            d.label,
                            d.confidence * 100,
                            existing.label,
                            existing.confidence * 100,
                        )
                    duplicate = True
                    break
            if not duplicate:
                deduped.append(d)
        merged = deduped
        if len(merged) < before_dedup:
            logger.info("Cross-label dedup removed %d duplicate(s)", before_dedup - len(merged))

        # Number multi-instance labels and normalize Z.
        # With a task spec the numbering follows the registry's declared
        # geometric order (a total order, so "<label> 1" is the same physical
        # object on every run). Without one it falls back to confidence rank,
        # which is arbitrary for identical objects, so a cached BT must always
        # run under a spec.
        if spec is not None:
            resolution = resolve_labels(spec, merged)
            # Published for detect_for_task, which needs the count gate but
            # cannot get it from the return value without breaking every other
            # caller's signature. Re-resolving instead would be wrong once Z
            # flattening below has run, since order_by may reference world_z.
            self._last_resolution = resolution
            merged = resolution.detections + resolution.extras
            if not resolution.ok:
                logger.warning("merge_detections: %s", resolution.describe())
        else:
            label_counts = defaultdict(list)
            for d in merged:
                label_counts[d.label].append(d)

            for label, dets in label_counts.items():
                if len(dets) > 1:
                    dets.sort(key=lambda d: d.confidence, reverse=True)
                    for i, d in enumerate(dets):
                        d.label = f"{label} {i + 1}"

        # Z flattening across same-label instances: two views of one flat
        # object can disagree on height, but flattening is fatal for a stack.
        # per_instance_z opts a task out; the default applies to callers with
        # no spec.
        if spec is not None and spec.per_instance_z:
            logger.debug("merge_detections: per-instance Z kept for task %r", spec.task)
        else:
            flatten_groups = defaultdict(list)
            for d in merged:
                flatten_groups[base_label(d.label)].append(d)
            for label, dets in flatten_groups.items():
                if len(dets) < 2:
                    continue
                zs = [d.position_3d[2] for d in dets if d.position_3d is not None]
                if not zs:
                    continue
                median_z = float(np.median(zs))
                for d in dets:
                    if d.position_3d is not None:
                        d.position_3d[2] = median_z

        for d in merged:
            pos_str = (
                f"({d.position_3d[0]:.3f},{d.position_3d[1]:.3f},{d.position_3d[2]:.3f})"
                if d.position_3d is not None
                else "N/A"
            )
            logger.info(
                "  Merged: %s -> cam=%s conf=%.2f pos=%s",
                d.label,
                getattr(d, "camera", "?"),
                d.confidence,
                pos_str,
            )
        return merged

    # Container labels whose detections should receive slot poses +
    # world-frame major-axis orientation. Slot detection runs in birdview
    # using surface-normal clustering and falls back to stripe-along-axis
    # when no compartments are separable. Matched case-insensitively
    # against the detection's base label (digit suffix from multi-
    # instance numbering stripped).
    SLOT_CONTAINER_LABELS = frozenset(
        {
            "tray",
            "grey tray",
            "silverware tray",
            "plate",
            "dish",
            "bowl",
            "dustpan",
            "pan",
            "drawer",
            "container",
            "compartment",
            "box",
        }
    )

    def _enrich_with_slots(self, merged, captures):
        """
        For container-like merged detections, compute slot poses +
        world-frame major-axis orientation. Mutates merged in place.

        Slot detection requires the source camera's depth + intrinsics
        + hand-eye extrinsic. Birdview is preferred (top-down view +
        best calibration); if a container was merged from another
        camera, we'll still try slot detection from THAT camera's
        depth, but the result is most reliable on birdview.
        """
        for d in merged:
            base_label = re.sub(r"\s+\d+$", "", str(d.label or "")).strip().lower()
            if base_label not in self.SLOT_CONTAINER_LABELS:
                continue
            cam_name = getattr(d, "camera", None)
            cap = captures.get(cam_name) if cam_name else None
            if cap is None:
                continue
            mask = d.mask
            depth = cap.get("depth")
            cal = cap.get("calibration")
            if mask is None or depth is None or cal is None:
                continue
            T_cam_to_base = getattr(cal, "extrinsic", None)
            try:
                K = cal.intrinsic_matrix
            except Exception:
                continue
            try:
                # depth here is cap["depth"], already corrected by capture()'s
                # _scaled (z*scale+offset), so pass depth_scale=1.0 to avoid
                # double-applying the per-camera depth correction.
                slots = detect_slots(
                    rgb=cap.get("rgb"),
                    depth_m=depth,
                    tray_mask=mask,
                    K=K,
                    T_cam_to_base=T_cam_to_base,
                    depth_scale=1.0,
                )
            except Exception as exc:
                logger.warning("detect_slots(%s) raised: %s", d.label, exc)
                continue
            d.slots = slots
            # World-frame major-axis orientation from the existing
            # world-XY PCA OBB code already populated d.orientation_angle.
            # For container labels, orientation_angle from
            # _detect_with_rendered_depth is in WORLD frame (radians
            # from world +X) when the world-XY PCA path fires, which
            # it does for top-down containers on the table.
            d.world_major_axis_rad = float(d.orientation_angle or 0.0)
            n_slots = len(slots) if slots else 0
            modes = sorted({s.get("mode", "?") for s in (slots or [])})
            logger.info(
                "  Slots(%s): %d slot(s) via %s, world_axis=%.3f rad",
                d.label,
                n_slots,
                "+".join(modes) if modes else "none",
                d.world_major_axis_rad or 0.0,
            )

    def _is_camera_calibrated(self, cam_name: str) -> bool:
        """
        Check if a camera has been calibrated (non-identity extrinsic).
        """
        cal = None
        if cam_name == "sideview":
            cal = self._kinect_cal
        elif cam_name == "birdview":
            cal = self._kinect2_cal
        elif cam_name == "wrist":
            cal = self._realsense_cal
        if cal is None:
            return False
        return not np.allclose(cal.extrinsic, np.eye(4))

    def prompt_registry(self):
        """
        Lazily-built, process-wide task prompt registry.

        Built once and cached on the pipeline instance. The directory is
        `perception.task_prompts_dir` from the family YAML when set, else
        $SPARK_TASK_PROMPTS, else the packaged configs/tasks.
        """
        registry = getattr(self, "_prompt_registry", None)
        if registry is None:
            config = getattr(self, "config", None)
            registry = load_registry(getattr(config, "task_prompts_dir", None))
            self._prompt_registry = registry
        return registry

    def task_spec(self, instruction: str):
        """
        Resolve an instruction to its frozen perception contract, or None.

        A hit means the prompts, the expected object counts and the instance
        ordering for this task are all pinned in version control, so no LLM
        is consulted and the labelling is reproducible run to run. A miss is
        not an error: the caller falls through to its existing prompt path.
        """
        if not instruction:
            return None
        return self.prompt_registry().lookup(instruction)

    def detect_for_task(
        self,
        captures: Dict,
        instruction: str,
        spec: "TaskSpec" = None,
        prompts: List[str] = None,
    ):
        """
        Detect + merge under a task's registered perception contract.

        Returns ``(merged, all_detections, resolution, spec, prompts_used)``.
        ``resolution`` is None when the instruction is not registered, in which
        case this is exactly the old detect-then-merge with no count checking.

        Escalation when the declared object counts are not met, cheapest
        first, stopping as soon as the gate passes:

        1. Re-run with the task's relaxed secondary-mask score. The usual
           cause of a missing second instance is the global 0.50 floor, and
           this costs one more SAM3 pass over an already-encoded image.
        2. Re-run with the group's alt_prompts appended. Covers the case where
           the vocabulary, not the threshold, is wrong. Alt prompts are
           relabelled to their canonical group name during resolution, so
           downstream names are unchanged.
        3. Ask the planner LLM to look at the frame and propose prompts for
           the still-missing groups (``fallback.llm_propose_prompts``). Covers
           the viewpoint-dependent case the task author could not anticipate.
           Proposals are folded into the spec as alt_prompts, so they are
           relabelled to the canonical name exactly like rung 2's.
        4. Apply ``fallback.on_mismatch``: ``abort`` raises
           :class:`PromptCountMismatch`, ``operator_click`` and
           ``best_effort`` return with ``resolution.ok`` False for the caller
           to route to a point prompt or to proceed anyway.

        ``abort`` is the default for the demo task set: a mislabelled episode
        silently poisons a training set.
        """
        if spec is None:
            spec = self.task_spec(instruction)

        if spec is None:
            used = prompts or self._extract_prompts(instruction)
            all_dets = self.detect(captures, used, multi_instance=True)
            return self.merge_detections(all_dets), all_dets, None, None, used

        base_prompts = list(prompts or spec.prompts)
        multi = spec.multi_instance_prompts

        def _pass(prompt_list, secondary):
            all_dets = self.detect(
                captures,
                prompt_list,
                multi_instance_prompts=multi | (set(prompt_list) & multi),
                secondary_score=secondary,
            )
            merged = self.merge_detections(all_dets, spec=spec)
            return merged, all_dets, self._last_resolution

        merged, all_dets, resolution = _pass(base_prompts, None)
        used = base_prompts

        relax = spec.fallback.relax_secondary_score
        if not resolution.ok and relax is not None:
            logger.warning(
                "task %r: count gate failed, retrying with secondary score %.2f",
                spec.task,
                relax,
            )
            merged, all_dets, resolution = _pass(base_prompts, relax)

        if not resolution.ok and resolution.retry_prompts:
            retry = base_prompts + [p for p in resolution.retry_prompts if p not in base_prompts]
            logger.warning(
                "task %r: count gate still failed, retrying with alt prompts %s",
                spec.task,
                resolution.retry_prompts,
            )
            # Alt prompts can themselves be the multi-instance ones.
            multi = multi | {alt for g in spec.groups if g.multi_instance for alt in g.alt_prompts}
            merged, all_dets, resolution = _pass(retry, relax)
            used = retry

        # Learned-prompts rung: phrases that rescued THIS task+group on an
        # earlier run (spark_bench's tuned-prompts tier; see
        # perception/prompt_cache.py). Costs one SAM3 pass and no API call,
        # so it sits between alt_prompts and the LLM. Cached phrases are
        # re-validated by the count gate like any alt prompt -- a stale one
        # simply fails through to the LLM rung below.
        if not resolution.ok:
            cached = self._cached_prompt_proposals(spec, resolution, used)
            if cached:
                spec = spec.with_alt_prompts(cached)
                extra = [p for ps in cached.values() for p in ps if p not in used]
                retry = used + extra
                logger.warning(
                    "task %r: count gate still failed, retrying with " "learned prompts %s",
                    spec.task,
                    extra,
                )
                multi = multi | {
                    alt for g in spec.groups if g.multi_instance for alt in g.alt_prompts
                }
                merged, all_dets, resolution = _pass(retry, relax)
                used = retry

        if not resolution.ok and spec.fallback.llm_propose_prompts:
            proposed = self._llm_propose_prompts(captures, spec, resolution, used)
            if proposed:
                # Folding the proposals in as alt_prompts is what makes them
                # relabel to the canonical group name -- the same mechanism
                # rung 2 uses, not a parallel one. `_pass` closes over `spec`,
                # so rebinding it here is what the retry merges against.
                spec = spec.with_alt_prompts(proposed)
                extra = [p for ps in proposed.values() for p in ps if p not in used]
                retry = used + extra
                logger.warning(
                    "task %r: count gate still failed, retrying with " "LLM-proposed prompts %s",
                    spec.task,
                    extra,
                )
                multi = multi | {
                    alt for g in spec.groups if g.multi_instance for alt in g.alt_prompts
                }
                merged, all_dets, resolution = _pass(retry, relax)
                used = retry
                # Persist ONLY what the gate just accepted: each proposed
                # group that is no longer mismatched earned its phrases a
                # place in the cache. A rejected proposal never touches it.
                self._record_accepted_prompts(spec, resolution, proposed)

        # Annotation-rescue rung (config annotation_rescue or
        # $SPARK_ANNOTATION_RESCUE): a pointing provider (Gemini Robotics-ER 2
        # / MolmoAct) puts a pixel on each still-missing group, which seeds
        # SAM3's click head (the same head the operator's UI click uses).
        # Last rung before on_mismatch: most expensive, least cacheable. The
        # rescued mask is re-validated by the count gate like every other rung.
        if not resolution.ok and self._annotation_rescue_provider() is not None:
            rescued = self._annotation_point_rescue(captures, spec, resolution, all_dets)
            if rescued:
                merged = self.merge_detections(all_dets, spec=spec)
                resolution = self._last_resolution

        if not resolution.ok:
            logger.error("task %r: %s", spec.task, resolution.describe())
            resolution.raise_if_abort()

        return merged, all_dets, resolution, spec, used

    def _annotation_rescue_provider(self):
        """
        The configured annotation provider, or None when the rung is off.

        $SPARK_ANNOTATION_RESCUE overrides the config: "0"/"off" disables,
        "1" enables with the config's provider, any other value is taken
        as the provider name itself. Fail-open: an unavailable provider
        reads as rung-off.
        """
        env = os.environ.get("SPARK_ANNOTATION_RESCUE", "").strip().lower()
        cfg = getattr(self, "config", None)
        enabled = bool(getattr(cfg, "annotation_rescue", False))
        name = getattr(cfg, "annotation_provider", "er2")
        if env in ("0", "off", "false"):
            return None
        if env and env not in ("1", "on", "true"):
            name, enabled = env, True
        elif env:
            enabled = True
        if not enabled:
            return None
        from spark_real.perception.annotations import get_provider

        provider = get_provider(name)
        if provider is None or not provider.available():
            logger.warning("annotation rescue: provider %r unavailable", name)
            return None
        return provider

    def _annotation_point_rescue(self, captures, spec, resolution, all_dets) -> bool:
        """
        Rung body: point at each missing group, click-seed SAM3, append.

        Appends rescued ObjectDetections (labelled with the canonical
        group text, stamped with their camera) to ``all_dets`` in place;
        returns True when at least one landed so the caller re-merges and
        re-runs the count gate. Every failure mode returns without
        raising -- a rescue that can hard-fail is worse than none.
        """
        provider = self._annotation_rescue_provider()
        if provider is None:
            return False
        rescued_any = False
        for missing in resolution.mismatched_groups:
            group = spec.group_for(missing)
            query = group.text if group is not None else missing
            for cam_name in self._PROPOSAL_CAM_ORDER:
                data = (captures or {}).get(cam_name)
                cal = self._annotation_cal_for(cam_name)
                if data is None or data.get("rgb") is None or cal is None:
                    continue
                try:
                    anns = provider.annotate(data["rgb"], query, kind="point")
                except Exception as exc:  # noqa: BLE001 - rescue is best-effort
                    logger.warning(
                        "annotation rescue: %s.annotate(%r) failed: %s",
                        provider.name,
                        query,
                        exc,
                    )
                    anns = []
                if not anns:
                    continue
                h, w = data["rgb"].shape[:2]
                u, v = anns[0].to_pixels((w, h))[0]
                det = self._point_seeded_detection(data["rgb"], data.get("depth"), cal, u, v, query)
                if det is None:
                    logger.warning(
                        "annotation rescue: %s pointed at %r on %s "
                        "(%.0f, %.0f) but SAM3 gave no mask",
                        provider.name,
                        query,
                        cam_name,
                        u,
                        v,
                    )
                    continue
                det.camera = cam_name
                det.rescued_by = provider.name
                all_dets.append(det)
                rescued_any = True
                logger.warning(
                    "annotation rescue: %r found via %s point on %s " "(%.0f, %.0f), conf %.2f",
                    query,
                    provider.name,
                    cam_name,
                    u,
                    v,
                    det.confidence,
                )
                break  # next missing group
        return rescued_any

    def _annotation_cal_for(self, cam_name: str):
        """Calibration for a camera IFF it is actually calibrated."""
        if not self._is_camera_calibrated(cam_name):
            return None
        return {
            "sideview": self._kinect_cal,
            "birdview": self._kinect2_cal,
            "wrist": self._realsense_cal,
        }.get(cam_name)

    def _point_seeded_detection(self, rgb, depth, cal, u, v, label):
        """
        SAM3 point prompt at pixel (u, v) -> a backprojected ObjectDetection.

        The same click head the UI's /api/detect_click uses
        (routes/detection._run_point_prompt), taking the calibration
        directly instead of reading server state. Returns None when the
        prompt yields no mask.
        """
        try:
            perception = self._perception
            perception.load_models(load_da3=False)
            sam3_state = perception._sam3.set_image(Image.fromarray(rgb))
            sam3_state = perception.set_point_prompt(u, v, sam3_state, label=1)
        except Exception as exc:  # noqa: BLE001 - no mask is a soft miss
            logger.warning("annotation rescue: point prompt failed: %s", exc)
            return None
        import torch

        masks = sam3_state.get("masks", torch.tensor([]))
        scores = sam3_state.get("scores", torch.tensor([]))
        if masks.numel() == 0:
            return None
        best = int(scores.argmax())
        mask = masks[best].cpu().numpy().squeeze()
        ys, xs = np.where(mask > 0)
        if len(xs) == 0:
            return None

        position_3d, d_val = None, 0.0
        if depth is not None:
            h, w = rgb.shape[:2]
            mask_depths = depth[mask > 0]
            valid = (mask_depths > 0.01) & (mask_depths < 10)
            if valid.sum() > 3:
                fx = cal.fx if cal.fx > 0 else h / (2 * np.tan(np.deg2rad(cal.fovy_degrees) / 2))
                fy = cal.fy if cal.fy > 0 else fx
                cx_i = cal.cx if cal.cx > 0 else w / 2
                cy_i = cal.cy if cal.cy > 0 else h / 2
                vds = mask_depths[valid]
                x_c = (xs[valid].astype(np.float64) - cx_i) * vds / fx
                y_c = -(ys[valid].astype(np.float64) - cy_i) * vds / fy
                pts_cam = np.stack([x_c, y_c, -vds], axis=1)
                pts_world = (cal.rotation_matrix @ pts_cam.T).T + cal.position
                position_3d = np.median(pts_world, axis=0)
                d_val = float(np.percentile(vds, 25))

        det = ObjectDetection(
            label=label,
            confidence=float(scores[best]),
            centroid_2d=(float(xs.mean()), float(ys.mean())),
            mask_area=len(xs),
            depth_meters=d_val,
            position_3d=position_3d,
        )
        det.mask = mask
        return det

    def _cached_prompt_proposals(self, spec, resolution, tried) -> Dict[str, List[str]]:
        """Learned prompts for the still-missing groups, minus anything tried.

        Fail-open on absence: a pipeline built without a LearnedPrompts store
        (tests, sim, bimanual) simply has an empty cache rung.
        """
        store = getattr(self, "_learned_prompts", None)
        if store is None:
            return {}
        out: Dict[str, List[str]] = {}
        tried_set = set(tried or [])
        for missing in resolution.mismatched_groups:
            group = spec.group_for(missing)
            key = group.text if group is not None else missing
            fresh = [p for p in store.lookup(spec.task, key) if p not in tried_set]
            if fresh:
                out[key] = fresh
        return out

    def _record_accepted_prompts(self, spec, resolution, proposed) -> None:
        """Write each proposed group's phrases to the store iff its gate now passes."""
        store = getattr(self, "_learned_prompts", None)
        if store is None or not proposed:
            return
        still_missing = set(resolution.mismatched_groups) if not resolution.ok else set()
        for key, phrases in proposed.items():
            if key in still_missing:
                continue
            try:
                store.record(spec.task, key, phrases)
            except Exception as exc:  # noqa: BLE001 - cache loss must not fail detect
                logger.warning("learned-prompt record failed for %r: %s", key, exc)

    # Camera preference for the frame the LLM is shown. Birdview first for the
    # same reason merge_detections trusts it first: it is the view the merged
    # detection's XY comes from, so it is the view whose vocabulary has to work.
    _PROPOSAL_CAM_ORDER = ("birdview", "sideview", "wrist")

    def _llm_propose_prompts(self, captures, spec, resolution, tried) -> Dict[str, List[str]]:
        """
        Rung 3: ask the planner LLM to name the groups SAM3 could not find.

        Returns ``{group_text: [prompt, ...]}``, or ``{}`` for every failure
        mode (no planner, no frame, the call raised, junk answer, or every
        suggestion already tried). Nothing here propagates: the caller sees an
        empty dict and falls through to ``fallback.on_mismatch``.

        Accepted proposals persist: _record_accepted_prompts writes gate-
        passing phrases into the pipeline's LearnedPrompts store, and the
        cached rung above tries them on the next failing run before this one
        spends an API call. Pasting a winner into the task's ``alt_prompts:``
        YAML is the stronger fix (rung 2 instead of rung 3.5).
        """
        planner = getattr(self, "_planner", None)
        if planner is None or not hasattr(planner, "propose_alt_prompts"):
            logger.warning(
                "task %r: no planner available for prompt proposal; "
                "falling through to on_mismatch",
                spec.task,
            )
            return {}

        scene_img = None
        for cam in list(self._PROPOSAL_CAM_ORDER) + list(captures or {}):
            data = (captures or {}).get(cam)
            if data is not None and data.get("rgb") is not None:
                scene_img = data["rgb"]
                break
        if scene_img is None:
            logger.warning("task %r: no frame to show the planner", spec.task)
            return {}

        tried_norm = {base_label(p) for p in (tried or [])}
        out: Dict[str, List[str]] = {}
        for missing in resolution.mismatched_groups:
            group = spec.group_for(missing)
            known = set(tried_norm)
            if group is not None:
                known |= {base_label(group.text)}
                known |= {base_label(a) for a in group.alt_prompts}
            try:
                raw = planner.propose_alt_prompts(
                    missing,
                    sorted(known),
                    scene_image=scene_img,
                    max_prompts=spec.fallback.llm_max_prompts,
                )
            except Exception as exc:  # noqa: BLE001 - never fail the recovery
                logger.warning(
                    "task %r: prompt proposal for %r failed (%s); "
                    "falling through to on_mismatch",
                    spec.task,
                    missing,
                    exc,
                )
                return {}
            if isinstance(raw, str) or not isinstance(raw, (list, tuple)):
                logger.warning(
                    "task %r: planner returned %s for %r, not a list of prompts",
                    spec.task,
                    type(raw).__name__,
                    missing,
                )
                continue
            clean: List[str] = []
            for item in raw:
                if not isinstance(item, str):
                    continue
                text = item.strip()
                if not text or base_label(text) in known:
                    continue
                known.add(base_label(text))
                clean.append(text)
                if len(clean) >= spec.fallback.llm_max_prompts:
                    break
            if clean:
                out[missing] = clean
                logger.warning(
                    "task %r: planner proposes %s for missing %r -- add the "
                    "one that works to alt_prompts in %s",
                    spec.task,
                    clean,
                    missing,
                    spec.source_path or "the task YAML",
                )
        return out

    def _extract_prompts(self, instruction: str) -> List[str]:
        """
        Extract object names from instruction for SAM3 detection.

        Fold tasks expand to sub-part keypoints: SAM3 is open-vocab and
        can ground "shirt sleeve" / "shirt hem" / "shirt collar"
        directly, and the fold planner sequence operates over those
        parts (pinch a sleeve, arc to centre, release, repeat).
        """
        instr_l = instruction.lower()
        instr_words = instr_l.replace(",", " ").split()

        # Fold-task expansion: keypoints are garment-specific parts.
        fold_verbs = ("fold", "folding")
        if any(v in instr_words for v in fold_verbs):
            if "t-shirt" in instr_l or "tshirt" in instr_l or "shirt" in instr_l:
                return [
                    "shirt left sleeve",
                    "shirt right sleeve",
                    "shirt hem",
                    "shirt collar",
                    "shirt body",
                ]
            if "towel" in instr_l:
                return [
                    "towel left edge",
                    "towel right edge",
                    "towel top edge",
                    "towel bottom edge",
                    "towel",
                ]
            if "napkin" in instr_l:
                return ["napkin corner", "napkin edge", "napkin"]
            if "pants" in instr_l or "trousers" in instr_l:
                return ["pants left leg", "pants right leg", "pants waistband", "pants"]
            # Generic cloth fallback: corners + edges of the named item.
            # The word(s) after "fold" usually name the cloth.
            try:
                idx = next(i for i, w in enumerate(instr_words) if w in fold_verbs)
                name = " ".join(instr_words[idx + 1 : idx + 4])
                name = name.replace("the ", "").strip()
                if name:
                    return [f"{name} corner", f"{name} edge", name]
            except StopIteration:
                pass

        stop_words = {
            "pick",
            "up",
            "the",
            "and",
            "place",
            "it",
            "on",
            "put",
            "in",
            "into",
            "a",
            "an",
            "to",
            "from",
            "move",
            "grab",
            "take",
            "get",
            "set",
            "drop",
            "both",
            "all",
            "each",
        }
        prompts = []
        current = []
        for w in instr_words:
            if w in stop_words:
                if current:
                    prompts.append(" ".join(current))
                    current = []
            else:
                current.append(w)
        if current:
            prompts.append(" ".join(current))
        return prompts if prompts else [instruction]

    @staticmethod
    def _annotate_image(
        rgb: np.ndarray, detections: list, depth: np.ndarray = None, calibration=None
    ) -> np.ndarray:
        """
        Draw clean detection mask overlays and labels on an image.

        Shows ONLY:
          - Semi-transparent colored mask with contour outline
          - Label text with black outline for readability

        No bounding boxes, OBBs, axes, arrows, crosshairs, or coordinates.
        The ``depth`` and ``calibration`` parameters are accepted for
        API compatibility but no longer affect the mask overlay.
        """
        vis = rgb.copy()
        colors = [
            (255, 60, 60),
            (60, 220, 120),
            (60, 120, 255),
            (255, 200, 40),
            (220, 60, 220),
            (60, 220, 220),
            (255, 140, 40),
            (140, 255, 40),
            (40, 140, 255),
        ]

        for i, det in enumerate(detections):
            c = np.array(colors[i % len(colors)])
            c_tuple = tuple(int(x) for x in c)
            mask = getattr(det, "mask", None)
            if mask is not None and mask.shape == vis.shape[:2]:
                # Semi-transparent mask tint
                vis[mask > 0] = (vis[mask > 0] * 0.55 + c * 0.45).astype(np.uint8)
                # Contour outline
                contours, _ = cv2.findContours(
                    mask.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
                )
                cv2.drawContours(vis, contours, -1, c_tuple, 2)

        pil = Image.fromarray(vis)
        draw = ImageDraw.Draw(pil)
        try:
            font = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSansMono.ttf", 16)
        except (OSError, IOError):
            font = ImageFont.load_default()

        for i, det in enumerate(detections):
            cx, cy = det.centroid_2d if det.centroid_2d else (0, 0)
            c = colors[i % len(colors)]
            label_text = det.label
            tx, ty = int(cx) + 14, int(cy) - 10
            # Black outline for readability against busy backgrounds
            for dx in (-1, 0, 1):
                for dy in (-1, 0, 1):
                    if dx == 0 and dy == 0:
                        continue
                    draw.text((tx + dx, ty + dy), label_text, fill=(0, 0, 0), font=font)
            draw.text((tx, ty), label_text, fill=c, font=font)

        return np.array(pil)

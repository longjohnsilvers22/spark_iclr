"""
Water-level estimation from the sideview Azure Kinect.

During a pour primitive the executor needs a real-time signal saying
"the target container is now ~70% full, stop tilting". This module
provides ``WaterLevelTracker``, a sideview-camera-driven estimator that:

1. At pour-start, captures one sideview RGB+depth frame, runs SAM3 on
   the container label, and computes the **rim plane Z** + the
   **container floor Z** in robot/world coordinates. The container's
   total interior depth = rim_z - floor_z.
2. During pouring, captures fresh sideview frames at the rate the
   executor polls ``update()``. For each frame it finds the
   **shallowest stable surface** visible inside the container mask
   (= the water surface; water reflects/refracts IR but its meniscus
   produces a usable depth gradient at the container interior), and
   converts that to a fill_fraction in [0, 1]:

       fill_fraction = (current_surface_z - floor_z) / (rim_z - floor_z)

3. As a secondary signal, computes vertical optical flow inside the
   container ROI between consecutive frames (Farneback dense flow).
   Upward flow magnitude correlates with fill rate; it is exposed as
   ``"flow_rate"`` so the caller can detect "pouring is happening"
   independently of the absolute level estimate.

The estimator is a best-effort surface tracker, NOT a metrology tool.
It works best on opaque matte mugs/cups under good lighting and is
substantially worse on clear glass, reflective rims, or in dim light
(depth from IR ToF refracts/absorbs at water surfaces). The optical-flow
rate signal is more robust than the absolute level for the "stop now"
decision.

Integration sketch (executor pseudo-code)::

    from spark_real.perception.water_level import WaterLevelTracker

    def pour_with_level_feedback(pipeline, target_label, fill_frac=0.7):
        tracker = WaterLevelTracker(pipeline, target_label=target_label)
        tracker.start(target_fill_fraction=fill_frac)
        # Begin the tilt motion (async / non-blocking).
        executor.start_tilt(...)
        try:
            while not executor.tilt_done():
                state = tracker.update()
                if state["should_stop"]:
                    executor.cancel_tilt()
                    executor.return_to_neutral()
                    break
                time.sleep(0.15)  # ~6-7 Hz polling
        finally:
            tracker.stop()

Limitations:
- Sideview Kinect must see the container interior from the rim down;
  if the camera is below the rim the rim plane fit will be wrong.
- Container shape is assumed roughly cylindrical / convex from the
  top: the floor profile is not modelled, just its 90th-pct depth in
  the SAM3 mask at start time as ``floor_z``.
- Glass / transparent containers: depth is unreliable through the
  wall, so the water surface inside may register as the OPPOSITE
  wall instead. For glass, the optical-flow signal is more useful.
- Calibration drift aliases directly into fill_fraction error; re-run
  handeye_calibrate if the sideview RMSE creeps up.
- "Stable" surface filter: the shallowest depth percentile must hold
  steady across the last few frames before it is trusted, which rejects
  splash/spray transients at the cost of some lag near the end of a pour.
"""

from __future__ import annotations

import logging
import time
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

try:
    import cv2

    _HAS_CV2 = True
except ImportError:  # pragma: no cover, cv2 is a hard dep elsewhere
    _HAS_CV2 = False


logger = logging.getLogger(__name__)


# Tuning constants. Conservative defaults; tweak per-container as needed.

# Depth percentile used to pick the "water surface": shallow side of the
# in-mask depth distribution. A percentile rather than min, because raw min
# is dominated by single noisy ToF pixels.
SURFACE_PCTL: float = 5.0

# Depth percentile used at start() to pick the container floor: deep side.
FLOOR_PCTL: float = 90.0

# Depth percentile used at start() to pick the rim plane (shallow slice of
# the in-mask depth distribution).
RIM_PCTL: float = 10.0

# Minimum number of valid depth pixels inside the container mask required
# to trust a fill estimate.
MIN_VALID_PIXELS: int = 30

# Minimum SAM3 confidence for the container detection at start().
MIN_CONTAINER_CONFIDENCE: float = 0.10

# Surface-stability window: number of consecutive updates that must agree
# (within STABLE_TOL_M) before ``stable: true`` is reported.
STABLE_WINDOW: int = 3
STABLE_TOL_M: float = 0.002  # 2 mm

# Optical flow tuning (Farneback). Off-the-shelf defaults sized for the
# sideview crop of the container.
FLOW_PYR_SCALE: float = 0.5
FLOW_LEVELS: int = 3
FLOW_WIN: int = 15
FLOW_ITERATIONS: int = 3
FLOW_POLY_N: int = 5
FLOW_POLY_SIGMA: float = 1.2

# Floor on start(target_fill_fraction=...): the target is never taken
# below this regardless of the caller's request.
SAFETY_MIN_FRAC: float = 0.05


@dataclass
class _ContainerState:
    """
    Geometry baseline captured at start().
    """

    mask_init: np.ndarray  # binary (H, W), the SAM3 container mask
    bbox: Tuple[int, int, int, int]  # x1, y1, x2, y2 in sideview pixels
    rim_z_world: float
    floor_z_world: float
    height_m: float  # rim_z - floor_z (positive when rim is above floor)
    init_surface_z: float  # surface Z at start (~floor_z for empty cup)
    init_gray: Optional[np.ndarray]  # ROI gray image for optical-flow seed
    label: str
    confidence: float


class WaterLevelTracker:
    """
    Sideview-camera water level tracker for the ``pour`` primitive.

    Construct, call ``start()`` once just before the executor begins
    tilting, poll ``update()`` at 5-10 Hz, call ``stop()`` to release
    held references.

    Either pass ``pipeline`` (a ``SPARKRealPipeline`` instance, uses
    its sideview Kinect + perception) OR pass ``perception`` and
    ``camera`` directly (for the standalone probe script). One of the
    two paths must be wired.
    """

    def __init__(
        self,
        pipeline: Optional[Any] = None,
        target_label: str = "cup",
        *,
        perception: Optional[Any] = None,
        camera: Optional[Any] = None,
        calibration: Optional[Any] = None,
    ):
        """
        Args:
            pipeline: SPARKRealPipeline instance. If provided, its
                sideview camera + SAM3 perception are re-used.
            target_label: SAM3 text prompt for the container
                ("cup", "glass", "mug", "white bowl", ...).
            perception: explicit SPARKPerception instance (used by
                the standalone probe). Ignored if ``pipeline`` is set.
            camera: explicit AzureKinectCamera (sideview). Ignored if
                ``pipeline`` is set.
            calibration: explicit CameraCalibration for ``camera``.
                Ignored if ``pipeline`` is set.
        """
        self._pipeline = pipeline
        self._target_label = target_label
        self._perception_override = perception
        self._camera_override = camera
        self._calibration_override = calibration

        self._state: Optional[_ContainerState] = None
        self._target_fill: float = 0.70
        self._history: List[float] = []
        self._prev_gray: Optional[np.ndarray] = None
        self._last_update_ts: float = 0.0

    # Plumbing helpers: abstract over pipeline vs. standalone wiring.

    def _capture_sideview(self) -> Optional[Dict[str, Any]]:
        """
        Return a dict with keys ``rgb`` (H,W,3 uint8), ``depth``
        (H,W float32 meters, may be None), ``calibration``
        (CameraCalibration). None if no frame is available.
        """
        if self._pipeline is not None:
            captures = self._pipeline.capture()
            data = captures.get("sideview")
            if data is None:
                logger.warning("water_level: pipeline has no sideview capture")
                return None
            return data

        if self._camera_override is None or self._calibration_override is None:
            raise RuntimeError(
                "WaterLevelTracker needs either a pipeline or an explicit "
                "(perception, camera, calibration) trio."
            )
        rgb, depth = self._camera_override.read()
        if rgb is None:
            return None
        # Apply the per-camera depth_scale just like pipeline.capture() does.
        scale = float(getattr(self._calibration_override, "depth_scale", 1.0) or 1.0)
        if depth is not None and scale != 1.0:
            depth = (depth.astype(np.float32) * scale).astype(depth.dtype)
        return {
            "rgb": rgb,
            "depth": depth,
            "calibration": self._calibration_override,
        }

    def _get_perception(self):
        if self._pipeline is not None:
            perc = getattr(self._pipeline, "_perception", None)
            if perc is None:
                raise RuntimeError("pipeline has no SPARKPerception attached")
            return perc
        if self._perception_override is None:
            raise RuntimeError(
                "WaterLevelTracker needs a perception instance in standalone mode."
            )
        return self._perception_override

    # 3D backprojection: mirrors spark_perception._cloud_median_world
    # (OpenCV convention; z forward, x right, y down).

    def _world_z_from_mask_depth_pctl(
        self,
        mask: np.ndarray,
        depth: np.ndarray,
        cal: Any,
        depth_pctl: float,
        depth_layer_m: float = 0.01,
    ) -> Optional[float]:
        """
        Find the depth value at ``depth_pctl`` percentile inside
        ``mask``, then return the WORLD-FRAME Z of the centroid of
        pixels within ``depth_layer_m`` of that depth.

        Equivalent to "world Z of the surface visible at this slice
        of the depth distribution". Used to extract rim Z (shallow
        slice), floor Z (deep slice), and water-surface Z (shallowest
        stable slice) from the same container mask.
        """
        if depth is None:
            return None
        ys, xs = np.where(mask > 0)
        if len(xs) < MIN_VALID_PIXELS:
            return None
        depths = depth[ys, xs].astype(np.float64)
        valid = (depths > 0.01) & (depths < 10.0)
        if int(valid.sum()) < MIN_VALID_PIXELS:
            return None
        xs_v = xs[valid].astype(np.float64)
        ys_v = ys[valid].astype(np.float64)
        d_v = depths[valid]

        target_d = float(np.percentile(d_v, depth_pctl))
        slab = np.abs(d_v - target_d) <= depth_layer_m
        if int(slab.sum()) < max(MIN_VALID_PIXELS // 3, 5):
            # Fallback: just use the percentile-band pixels, even sparse.
            slab = (d_v <= target_d + depth_layer_m) & (d_v >= target_d - depth_layer_m)
            if int(slab.sum()) < 5:
                return None
        xs_s = xs_v[slab]
        ys_s = ys_v[slab]
        d_s = d_v[slab]

        fx = float(cal.fx)
        fy = float(cal.fy)
        cx_k = float(cal.cx)
        cy_k = float(cal.cy)
        # OpenCV convention (matches spark_perception when it uses
        # intrinsic_matrix; matches the standalone SAM3 probe).
        x_cam = (xs_s - cx_k) * d_s / fx
        y_cam = (ys_s - cy_k) * d_s / fy
        z_cam = d_s
        pts_cam = np.column_stack([x_cam, y_cam, z_cam])
        R = cal.extrinsic[:3, :3]
        t = cal.extrinsic[:3, 3]
        pts_world = (R @ pts_cam.T).T + t
        return float(np.median(pts_world[:, 2]))

    # Container detection (called once at start).

    def _detect_container(
        self,
        rgb: np.ndarray,
        depth: Optional[np.ndarray],
        cal: Any,
    ) -> Optional[Any]:
        """
        Run SAM3 on the target label; return the best ObjectDetection
        (or None if SAM3 finds nothing above MIN_CONTAINER_CONFIDENCE).
        """
        perc = self._get_perception()
        if depth is not None:
            dets = perc._detect_with_rendered_depth(
                rgb=rgb,
                depth=depth,
                prompts=[self._target_label],
                cam_pos=cal.position,
                cam_mat=cal.rotation_matrix,
                cam_fovy=cal.fovy_degrees,
                w=int(cal.width),
                h=int(cal.height),
                intrinsic_matrix=cal.intrinsic_matrix,
            )
        else:
            dets = perc.detect(
                rgb=rgb,
                prompts=[self._target_label],
                cam_pos=cal.position,
                cam_mat=cal.rotation_matrix,
                cam_fovy=cal.fovy_degrees,
            )
        if not dets:
            return None
        # Pick the highest-confidence detection; only ONE container
        # matters during a pour.
        best = max(dets, key=lambda d: float(d.confidence or 0.0))
        if float(best.confidence or 0.0) < MIN_CONTAINER_CONFIDENCE:
            return None
        return best

    # Public API.

    def start(self, target_fill_fraction: float = 0.70) -> Dict[str, Any]:
        """
        Capture sideview, lock the container geometry, return baseline.

        Returns a dict::

            {
                "ok": bool,
                "reason": str,            # populated if ok=False
                "label": str,             # container label
                "confidence": float,
                "rim_z": float,           # world Z (meters)
                "floor_z": float,
                "height_m": float,
                "target_fill_fraction": float,
                "bbox": (x1, y1, x2, y2),
            }
        """
        self._target_fill = float(max(SAFETY_MIN_FRAC, min(1.0, target_fill_fraction)))
        self._history.clear()
        self._prev_gray = None
        self._state = None

        cap = self._capture_sideview()
        if cap is None:
            return {"ok": False, "reason": "no_sideview_capture"}
        rgb = cap["rgb"]
        depth = cap.get("depth")
        cal = cap.get("calibration")
        if rgb is None or cal is None:
            return {"ok": False, "reason": "missing_rgb_or_calibration"}
        if depth is None:
            return {
                "ok": False,
                "reason": "no_depth_from_sideview (estimator requires depth)",
            }

        det = self._detect_container(rgb, depth, cal)
        if det is None or det.mask is None:
            return {
                "ok": False,
                "reason": f"sam3_no_detection_for_label={self._target_label!r}",
            }

        mask = det.mask.astype(bool)
        # Rim = shallow slice, floor = deep slice. These are the two
        # extremes of the in-mask depth distribution. For an empty cup
        # viewed from the side this matches the literal rim + literal
        # interior floor.
        rim_z = self._world_z_from_mask_depth_pctl(
            mask, depth, cal, depth_pctl=RIM_PCTL
        )
        floor_z = self._world_z_from_mask_depth_pctl(
            mask, depth, cal, depth_pctl=FLOOR_PCTL
        )
        if rim_z is None or floor_z is None:
            return {
                "ok": False,
                "reason": "insufficient_valid_depth_in_container_mask",
            }

        # Sideview cameras typically look slightly down at the table, so
        # rim_z > floor_z in robot frame (Z is up). If inverted (e.g. the
        # mask captures the table behind the cup, not the cup interior),
        # swap the labels and warn.
        if rim_z < floor_z:
            logger.warning(
                "water_level: rim_z (%.3f) < floor_z (%.3f); "
                "swapping. Container mask may be including table.",
                rim_z,
                floor_z,
            )
            rim_z, floor_z = floor_z, rim_z

        height_m = float(rim_z - floor_z)
        if height_m < 0.005:
            return {
                "ok": False,
                "reason": (
                    f"container_height_too_small ({height_m * 1000:.1f} mm), "
                    "likely a 2D mask or wrong SAM3 prompt"
                ),
            }

        # Initial surface estimate (assume the cup is empty at start;
        # the shallowest stable surface inside the container should be
        # at floor level for the empty case).
        init_surface_z = floor_z

        # ROI gray for optical-flow init.
        bbox = (
            tuple(int(v) for v in det.bbox)
            if det.bbox is not None
            else (0, 0, rgb.shape[1], rgb.shape[0])
        )
        init_gray = self._roi_gray(rgb, bbox)

        self._state = _ContainerState(
            mask_init=mask,
            bbox=bbox,
            rim_z_world=float(rim_z),
            floor_z_world=float(floor_z),
            height_m=height_m,
            init_surface_z=float(init_surface_z),
            init_gray=init_gray,
            label=self._target_label,
            confidence=float(det.confidence or 0.0),
        )
        self._prev_gray = init_gray
        self._last_update_ts = time.time()
        logger.info(
            "water_level.start: label=%s conf=%.2f rim_z=%.3fm floor_z=%.3fm "
            "height=%.3fm target_fill=%.2f",
            self._target_label,
            float(det.confidence or 0.0),
            rim_z,
            floor_z,
            height_m,
            self._target_fill,
        )
        return {
            "ok": True,
            "label": self._target_label,
            "confidence": float(det.confidence or 0.0),
            "rim_z": float(rim_z),
            "floor_z": float(floor_z),
            "height_m": height_m,
            "target_fill_fraction": self._target_fill,
            "bbox": bbox,
        }

    def update(self) -> Dict[str, Any]:
        """
        Capture a fresh sideview, return current fill estimate.

        Returns a dict::

            {
                "ok": bool,
                "reason": str,            # populated if ok=False
                "fill_fraction": float,   # 0=empty, 1=at rim
                "surface_z": float,       # world Z (m)
                "stable": bool,           # last STABLE_WINDOW agree within tol
                "should_stop": bool,      # fill_fraction >= target AND stable
                "flow_rate": float,       # mean upward vertical flow (px/frame)
                "elapsed_s": float,       # seconds since last update
            }
        """
        if self._state is None:
            return {"ok": False, "reason": "tracker_not_started"}

        cap = self._capture_sideview()
        if cap is None:
            return {"ok": False, "reason": "no_sideview_capture"}
        rgb = cap["rgb"]
        depth = cap.get("depth")
        cal = cap.get("calibration")
        if rgb is None or depth is None or cal is None:
            return {"ok": False, "reason": "missing_rgb_depth_or_calibration"}

        # Reuse the container mask from start(). The container shouldn't have
        # moved during the pour; if it did, the whole pour is in trouble
        # anyway. Re-running SAM3 every update would also be far more expensive
        # than slicing depth against the cached mask.
        mask = self._state.mask_init
        if mask.shape[:2] != depth.shape[:2]:
            # Defensive resize: sideview shouldn't change resolution
            # mid-stream, but if it does, nearest-neighbor preserves
            # the boolean mask.
            mh, mw = mask.shape
            dh, dw = depth.shape
            if _HAS_CV2:
                mask = cv2.resize(
                    mask.astype(np.uint8),
                    (dw, dh),
                    interpolation=cv2.INTER_NEAREST,
                ).astype(bool)
            else:
                return {
                    "ok": False,
                    "reason": f"mask_shape_mismatch ({mh}x{mw} vs {dh}x{dw}) and no cv2",
                }

        # Shallowest stable surface inside the container = water surface.
        surface_z = self._world_z_from_mask_depth_pctl(
            mask,
            depth,
            cal,
            depth_pctl=SURFACE_PCTL,
        )
        if surface_z is None:
            return {"ok": False, "reason": "no_valid_depth_in_container_mask"}

        # Clamp to [floor_z, rim_z + small slack]. Above-rim surface_z is
        # physically possible (overflow); clamp at 1.05 to flag it
        # rather than report fill_fraction > 1 silently.
        height = self._state.height_m
        fill_raw = (surface_z - self._state.floor_z_world) / max(height, 1e-6)
        fill_fraction = float(np.clip(fill_raw, 0.0, 1.05))

        # Track recent surface Z for stability.
        self._history.append(surface_z)
        if len(self._history) > STABLE_WINDOW:
            self._history.pop(0)
        if len(self._history) >= STABLE_WINDOW:
            recent_arr = np.asarray(self._history, dtype=np.float64)
            stable = bool((recent_arr.max() - recent_arr.min()) <= STABLE_TOL_M)
        else:
            stable = False

        # Optical flow (secondary fill-rate signal).
        flow_rate = self._compute_flow_rate(rgb)

        # Stop condition: target reached AND stable. Stability is required
        # so a single splash transient doesn't trigger an early stop.
        # Also stop unconditionally if fill > 0.98 (about to overflow)
        # even without stability; safety beats accuracy.
        should_stop = (
            fill_fraction >= self._target_fill and stable
        ) or fill_fraction >= 0.98

        now = time.time()
        elapsed = float(now - self._last_update_ts)
        self._last_update_ts = now

        return {
            "ok": True,
            "fill_fraction": fill_fraction,
            "surface_z": float(surface_z),
            "stable": stable,
            "should_stop": should_stop,
            "flow_rate": float(flow_rate),
            "elapsed_s": elapsed,
        }

    def stop(self) -> None:
        """
        Release held references. Safe to call multiple times.
        """
        self._state = None
        self._history.clear()
        self._prev_gray = None
        logger.info("water_level.stop: tracker released")

    # Optical flow helpers.

    def _roi_gray(
        self, rgb: np.ndarray, bbox: Tuple[int, int, int, int]
    ) -> Optional[np.ndarray]:
        if not _HAS_CV2 or rgb is None:
            return None
        x1, y1, x2, y2 = bbox
        h, w = rgb.shape[:2]
        x1 = max(0, min(int(x1), w - 1))
        y1 = max(0, min(int(y1), h - 1))
        x2 = max(x1 + 1, min(int(x2), w))
        y2 = max(y1 + 1, min(int(y2), h))
        roi = rgb[y1:y2, x1:x2]
        if roi.size == 0:
            return None
        return cv2.cvtColor(roi, cv2.COLOR_RGB2GRAY)

    def _compute_flow_rate(self, rgb: np.ndarray) -> float:
        """
        Mean **upward** vertical flow (image-y is positive downward,
        so upward motion = negative dy). Returns 0.0 when flow can't
        be computed (cv2 missing, ROI empty, etc.).
        """
        if not _HAS_CV2 or self._state is None:
            return 0.0
        gray = self._roi_gray(rgb, self._state.bbox)
        if gray is None:
            self._prev_gray = None
            return 0.0
        if self._prev_gray is None or self._prev_gray.shape != gray.shape:
            self._prev_gray = gray
            return 0.0
        try:
            flow = cv2.calcOpticalFlowFarneback(
                self._prev_gray,
                gray,
                None,
                FLOW_PYR_SCALE,
                FLOW_LEVELS,
                FLOW_WIN,
                FLOW_ITERATIONS,
                FLOW_POLY_N,
                FLOW_POLY_SIGMA,
                0,
            )
        except cv2.error as exc:  # pragma: no cover, defensive
            logger.debug("optical-flow failure: %s", exc)
            self._prev_gray = gray
            return 0.0
        # flow[..., 1] is dy in pixels. Negative dy = upward (rising
        # water surface). Average |dy| over upward-moving pixels gives
        # a magnitude that tracks pour rate without sign cancellation.
        dy = flow[..., 1]
        upward = dy < 0
        if not upward.any():
            self._prev_gray = gray
            return 0.0
        rate = float(np.mean(-dy[upward]))
        self._prev_gray = gray
        return rate

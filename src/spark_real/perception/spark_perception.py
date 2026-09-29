"""
SPARK Perception: SAM3 + DA3 for object detection with metric 3D positions.

SAM3: text-prompted segmentation masks
DA3: metric monocular depth estimation
Combined: detect objects and localize them in 3D world coordinates.
"""

import contextlib
import functools
import logging
import os
import sys
import threading
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

import cv2
import numpy as np
import torch
from PIL import Image

try:
    import mujoco as _mujoco
except ImportError:
    _mujoco = None

try:
    from scipy.spatial import cKDTree as _cKDTree
except ImportError:
    _cKDTree = None

# Mask-to-3D geometry helpers, re-exported at module level for out-of-tree
# imports.
from spark_real.perception import dedup
from spark_real.perception.mask_geometry import (  # noqa: F401
    _sigma_n,
    _cloud_median_world,
    _top_layer_mask,
    _world_xy_pca_obb,
    _pca_obb,
    _ray_intersect_table,
    _compute_orientation,
    mask_color_stats,
    mask_height_profile,
)

logger = logging.getLogger(__name__)

_SAM3_PATHS = [
    Path(os.environ["SPARK_SAM3_PATH"]) if os.environ.get("SPARK_SAM3_PATH") else Path.home() / "mv_sam3" / "sam3",
]
for _p in _SAM3_PATHS:
    if _p.exists():
        sys.path.insert(0, str(_p))
        break

try:
    from sam3.model_builder import build_sam3_image_model
    from sam3.model.sam3_image_processor import Sam3Processor
except ImportError:
    build_sam3_image_model = None
    Sam3Processor = None

try:
    from depth_anything_3.api import DepthAnything3
except ImportError:
    DepthAnything3 = None


# Image-space centroid distance below which two masks in ONE camera are
# CANDIDATES for being the same object; dedup.same_object_2d then decides with
# mask IoU whether they actually are.
_DEDUP_PX = 30


@dataclass
class ObjectDetection:
    """
    Detected object with 3D position.
    """

    label: str
    confidence: float
    centroid_2d: tuple
    mask_area: int
    depth_meters: float = 0.0
    position_3d: Optional[np.ndarray] = None
    bbox: Optional[tuple] = None
    mask: Optional[np.ndarray] = None
    camera: Optional[str] = None
    orientation_angle: float = 0.0
    aspect_ratio: float = 1.0
    obb_minor_m: float = 0.0  # OBB short-axis length in meters (drives grasp width prior)
    slots: Optional[list] = None  # slot poses for container detections (tray, plate, etc.)
    world_major_axis_rad: Optional[float] = None  # major-axis angle (radians from world +X)
    # Ray-plane orientation: the MASK's major axis intersected with the table
    # plane, in world frame. Kept alongside orientation_angle because the two
    # fail in opposite conditions. This one needs a crisp silhouette and ONE
    # centroid depth; the point-cloud OBB that usually overwrites
    # orientation_angle needs good depth over the WHOLE object. On thin
    # specular things -- silverware, tools -- the silhouette is excellent and
    # the point cloud is noise, so this is the axis to grasp by.
    plane_orientation_angle: Optional[float] = None
    plane_aspect_ratio: Optional[float] = None
    # (hue_deg, saturation, value, hue_conf) sampled over the eroded mask.
    # Populated opportunistically; None when the source RGB was unavailable.
    # Consumed by the prompt registry's optional hsv_cluster disambiguation.
    hsv_median: Optional[tuple] = None
    # Colour-free role aliases assigned by the prompt registry, e.g.
    # "same color block 1". A cached behaviour tree addresses these instead of
    # a literal colour when the staged object identity varies between episodes.
    role_labels: Optional[list] = None
    # How well-defined the OBB axes are, [0,1]. A round or soft object scores
    # near 0 even when its aspect_ratio reads high off backprojection noise, so
    # the grasp resolver can refuse a meaningless yaw. 0.0 means NOT MEASURED
    # (no depth), which the resolver treats as unknown, not as a veto.
    obb_confidence: float = 0.0
    # World-Z top and bottom of this mask's depth cloud, and how many valid
    # depth pixels backed them (see mask_geometry.mask_height_profile). For a
    # CONTAINER these are its rim and its interior floor; for anything else,
    # simply the object's top and bottom. None / 0 means NOT MEASURED -- the
    # release-height computation falls back to the plan rather than guess.
    rim_z_m: Optional[float] = None
    interior_z_m: Optional[float] = None
    height_samples: int = 0
    # Agentview-only backprojected centroid, kept alongside the (possibly
    # wrist-fused) position_3d.  Same-source displacement measurement: a
    # later agentview-only re-detection must be compared against THIS, not
    # against the fused position - the two cameras disagree by a systematic
    # 1-2 cm on some objects, which otherwise reads as a phantom 'moved'.
    # None when the detection never had an agentview backprojection.
    position_agentview: Optional[np.ndarray] = None
    # ASPIRE reprompt bookkeeping (perception/detection conditioning).
    reprompt_attempts: int = 0
    # True when the surviving mask still trips a quality trigger. Forces the
    # `auto` grasp strategy to top-down and is reported to the planner.
    low_quality: bool = False


# torch.cuda.OutOfMemoryError (torch >= 1.13). None on older builds, in
# which case OOM containment is skipped rather than mis-catching.
_CUDA_OOM = getattr(torch.cuda, "OutOfMemoryError", None)


# A lock wait longer than this is a real contention event worth a log line.
_LOCK_WAIT_WARN_S = 0.25


class StageTimer:
    """Accumulates per-stage wall time for one detect, for a single log line.

    CUDA is async, so any stage that ends on GPU work synchronizes before the
    clock stops; otherwise the backbone's cost lands on whichever later stage
    first reads a tensor back to the host.
    """

    __slots__ = ("totals", "counts", "_sync")

    def __init__(self, sync_cuda=True):
        self.totals = {}
        self.counts = {}
        # SPARK_DETECT_TIMING_SYNC=0 keeps the stage lines and drops the CUDA
        # syncs, at the cost of async GPU work being attributed to a later
        # stage.
        if os.environ.get("SPARK_DETECT_TIMING_SYNC", "1").strip() == "0":
            sync_cuda = False
        self._sync = sync_cuda and torch.cuda.is_available()

    @contextlib.contextmanager
    def stage(self, name):
        if self._sync:
            torch.cuda.synchronize()
        t0 = time.perf_counter()
        try:
            yield
        finally:
            if self._sync:
                torch.cuda.synchronize()
            dt = time.perf_counter() - t0
            self.totals[name] = self.totals.get(name, 0.0) + dt
            self.counts[name] = self.counts.get(name, 0) + 1

    def total(self):
        return sum(self.totals.values())

    def describe(self):
        """"stage=total_s(xN)" for every stage, slowest first."""
        if not self.totals:
            return "no stages timed"
        parts = [
            f"{k}={v:.2f}s(x{self.counts[k]})"
            for k, v in sorted(self.totals.items(), key=lambda kv: -kv[1])
        ]
        return " ".join(parts)


def _serialized(fn):
    """Hold the instance's _detect_lock for the whole call + contain CUDA OOM.

    Serialization: the SAM3 predictor is STATEFUL (set_image then predict);
    two threads interleaving set_image/predict compute masks against the
    wrong image (the arm then servos to a phantom pose), and concurrent
    access to one CUDA model context can raise device-side asserts that
    poison the CUDA context. The executor's post-release re-detect thread
    runs concurrently with foreground detects, so every SAM3 entry point
    serializes here.

    OOM containment: the server shares its GPU with other jobs (and the
    Azure Kinect depth engine). empty_cache + one retry, then a CLEAR
    failure the caller can report.
    """

    @functools.wraps(fn)
    def wrapper(self, *args, **kwargs):
        # Time the WAIT separately from the work: a detect slow because it sat
        # behind another thread's lock looks the same from outside as one slow
        # on the GPU. Only a wait worth acting on is logged.
        _t_wait = time.perf_counter()
        with self._detect_lock:
            _waited = time.perf_counter() - _t_wait
            if _waited > _LOCK_WAIT_WARN_S:
                logger.warning(
                    "[detect-timing] %s waited %.2fs for _detect_lock "
                    "(another detect/redetect thread held it)",
                    fn.__name__,
                    _waited,
                )
            if _CUDA_OOM is None:
                return fn(self, *args, **kwargs)
            try:
                return fn(self, *args, **kwargs)
            except _CUDA_OOM:
                logger.warning(
                    "CUDA OOM in %s; empty_cache + one retry "
                    "(is another GPU job running beside the server?)",
                    fn.__name__,
                )
                torch.cuda.empty_cache()
                try:
                    return fn(self, *args, **kwargs)
                except _CUDA_OOM as exc:
                    torch.cuda.empty_cache()
                    raise RuntimeError(
                        f"SAM3/DA3 {fn.__name__} failed: CUDA out of memory "
                        "even after empty_cache + retry. Free GPU memory "
                        "(stop competing GPU jobs) and re-run the detect."
                    ) from exc

    return wrapper


class SPARKPerception:
    """
    SAM3 + DA3 perception module.

    Detects objects from text prompts and estimates their 3D world positions
    using monocular metric depth.
    """

    def __init__(
        self,
        sam3_threshold: float = 0.05,
        da3_model: str = "depth-anything/DA3METRIC-LARGE",
    ):
        self.sam3_threshold = sam3_threshold
        self.da3_model_name = da3_model
        self._sam3 = None
        self._da3 = None
        # Serializes every SAM3/DA3 forward pass (see _serialized). RLock so
        # a locked entry point may call another locked helper.
        self._detect_lock = threading.RLock()
        self.device = "cuda" if torch.cuda.is_available() else "cpu"

    def load_models(self, load_da3: bool = True):
        """
        Load SAM3 and optionally DA3 models.
        """
        if self._sam3 is None:
            logger.info("Loading SAM3...")
            model = build_sam3_image_model()
            self._sam3 = Sam3Processor(model)
            self._sam3.set_confidence_threshold(self.sam3_threshold)
            logger.info("SAM3 ready")
        if load_da3 and self._da3 is None:
            logger.info("Loading DA3 metric depth...")
            self._da3 = DepthAnything3.from_pretrained(self.da3_model_name)
            self._da3 = self._da3.to(self.device).eval()
            logger.info("DA3 ready")

    def set_point_prompt(self, x: float, y: float, state: dict, label: int = 1) -> dict:
        """
        Segment an object via a single point click using SAM2 decoder.

        Args:
            x: Pixel x-coordinate.
            y: Pixel y-coordinate.
            state: State dict previously returned by ``_sam3.set_image()``.
            label: 1 = foreground, 0 = background.

        Returns:
            Updated state dict with masks, scores, boxes (same format as
            ``set_text_prompt``).
        """
        self.load_models(load_da3=False)
        with self._detect_lock:
            return self._sam3.set_point_prompt(x, y, state, label=label)

    @_serialized
    def detect(
        self,
        rgb: np.ndarray,
        prompts: list,
        cam_pos=None,
        cam_mat=None,
        cam_fovy: float = 45.0,
    ) -> list:
        """
        Detect objects and estimate 3D positions using DA3 depth.
        """
        self.load_models()
        pil_img = Image.fromarray(rgb)
        h, w = rgb.shape[:2]

        # Run DA3 with focal scaling
        process_res = 504
        timer = StageTimer()
        _t_all = time.perf_counter()
        with timer.stage("da3_depth"):
            with torch.no_grad():
                da3_pred = self._da3.inference(
                    image=[pil_img], process_res=process_res
                )
                depth_map = da3_pred.depth[0]
            actual_f = h / (2 * np.tan(np.deg2rad(cam_fovy) / 2))
            da3_scale = process_res / max(w, h)
            depth_map = depth_map * (actual_f * da3_scale / 300.0)
            depth_resized = np.array(
                Image.fromarray(depth_map).resize((w, h), Image.BILINEAR)
            )

        detections = []
        # Encode the image ONCE and reuse it for every prompt: set_image() runs
        # the full vision backbone (1008x1008); set_text_prompt() only runs the
        # text encoder + grounding head against the cached backbone_out.
        # set_image() is deterministic and carries no prompt state.
        #
        # SAM3 needs bf16 autocast here, else the vision neck mixes
        # BFloat16 activations with Float weights.
        with timer.stage("encode"):
            with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
                base_state = self._sam3.set_image(pil_img)

        for prompt in prompts:
            # Shallow copy per prompt so each prompt's masks/scores land on its
            # own dict while the (expensive, image-only) backbone_out is shared.
            with timer.stage("grounding"):
                with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
                    state = self._sam3.set_text_prompt(
                        prompt=prompt, state={**base_state}
                    )
            masks = state.get("masks", torch.tensor([]))
            scores = state.get("scores", torch.tensor([]))
            if masks.numel() == 0:
                continue

            n_confident = int((scores > 0.30).sum())
            sigma = _sigma_n(n_confident)
            best_idx = int(scores.argmax())
            score = float(scores[best_idx]) * sigma
            mask = masks[best_idx].cpu().numpy().squeeze()
            ys, xs = np.where(mask > 0)
            if len(xs) == 0:
                continue

            cx, cy = float(xs.mean()), float(ys.mean())
            bbox = (float(xs.min()), float(ys.min()), float(xs.max()), float(ys.max()))
            obb_mask = _top_layer_mask(mask, depth_resized)
            img_angle_rad, aspect_ratio, short_px, long_px = _pca_obb(obb_mask)
            _wxy_orient, _wxy_ar, _wxy_mj_m, _wxy_mn_m, _wxy_conf = _world_xy_pca_obb(
                mask,
                depth_resized,
                cam_pos,
                cam_mat,
                fovy_deg=cam_fovy,
                use_opencv=False,
                return_confidence=True,
            )

            mask_depths = depth_resized[mask > 0]
            valid = mask_depths[mask_depths > 0.01]
            metric_depth = (
                float(np.median(valid))
                if len(valid) > 0
                else float(depth_resized[int(cy), int(cx)])
            )

            orientation_angle = _compute_orientation(
                mask,
                depth_resized,
                cam_mat,
                cam_pos,
                cam_fovy,
                cx,
                cy,
                h,
                w,
                img_angle_rad,
                long_px,
                metric_depth,
            )

            obb_minor_m = 0.0
            if short_px > 0 and metric_depth > 0.01:
                f_minor = h / (2 * np.tan(np.deg2rad(cam_fovy) / 2)) if cam_fovy else h
                if f_minor > 0:
                    obb_minor_m = float(short_px * metric_depth / f_minor)

            # Preserve the ray-plane answer BEFORE the point-cloud OBB
            # overwrites it: the override ignores _wxy_conf, and grasp_strategy
            # falls back to these when the point-cloud OBB is untrustworthy.
            plane_orientation_angle = float(orientation_angle)
            plane_aspect_ratio = float(aspect_ratio)

            if _wxy_orient is not None:
                orientation_angle = _wxy_orient
                aspect_ratio = float(_wxy_ar)
                obb_minor_m = float(_wxy_mn_m)

            det = ObjectDetection(
                label=prompt,
                confidence=score,
                centroid_2d=(cx, cy),
                mask_area=len(xs),
                depth_meters=metric_depth,
                bbox=bbox,
                mask=mask,
                orientation_angle=orientation_angle,
                plane_orientation_angle=plane_orientation_angle,
                plane_aspect_ratio=plane_aspect_ratio,
                aspect_ratio=aspect_ratio,
                obb_minor_m=obb_minor_m,
                obb_confidence=float(_wxy_conf or 0.0),
            )

            if cam_pos is not None and cam_mat is not None:
                centroid_w, _ = _cloud_median_world(
                    mask,
                    depth_resized,
                    cam_mat,
                    cam_pos,
                    fovy_deg=cam_fovy,
                    use_opencv=False,
                )
                if centroid_w is not None:
                    det.position_3d = centroid_w
                else:
                    f = h / (2 * np.tan(np.deg2rad(cam_fovy) / 2))
                    x_cam = (cx - w / 2) * metric_depth / f
                    y_cam = -(cy - h / 2) * metric_depth / f
                    z_cam = -metric_depth
                    det.position_3d = (
                        cam_mat @ np.array([x_cam, y_cam, z_cam]) + cam_pos
                    )
                # Same rim/interior profile as the hardware-depth path. DA3 is
                # monocular metric depth, so the absolute scale is softer here
                # -- the release-height clamps are what keep that honest.
                _profile = mask_height_profile(
                    mask,
                    depth_resized,
                    cam_pos,
                    cam_mat,
                    fovy_deg=cam_fovy,
                    use_opencv=False,
                )
                if _profile is not None:
                    det.rim_z_m = _profile["rim_z"]
                    det.interior_z_m = _profile["interior_z"]
                    det.height_samples = _profile["n_valid"]
            detections.append(det)

        # "geometry" is the residual: total wall time minus the stages timed
        # above, so nothing unaccounted for disappears.
        _elapsed = time.perf_counter() - _t_all
        logger.info(
            "[detect-timing] da3 %dx%d: %d prompt(s) -> %d det(s) in %.2fs | "
            "%s geometry+other=%.2fs",
            w,
            h,
            len(prompts),
            len(detections),
            _elapsed,
            timer.describe(),
            max(_elapsed - timer.total(), 0.0),
        )
        return detections

    @_serialized
    def detect_in_mujoco(
        self,
        model,
        data,
        prompts: list,
        cam_name: str = "robot0_eye_in_hand",
        width: int = 640,
        height: int = 480,
        use_rendered_depth: bool = True,
        dual_camera: bool = False,
    ) -> list:
        """
        Detect objects in a MuJoCo scene using SAM3 + depth.
        """
        if dual_camera and use_rendered_depth:
            return self._detect_dual_camera(model, data, prompts, width, height)
        cam_id = _mujoco.mj_name2id(model, _mujoco.mjtObj.mjOBJ_CAMERA, cam_name)
        if cam_id < 0:
            for alt in ["agentview", "frontview", "birdview"]:
                cam_id = _mujoco.mj_name2id(model, _mujoco.mjtObj.mjOBJ_CAMERA, alt)
                if cam_id >= 0:
                    break
        renderer = _mujoco.Renderer(model, height, width)
        renderer.update_scene(data, camera=cam_id)
        rgb = renderer.render().copy()
        mj_depth = None
        if use_rendered_depth:
            renderer.enable_depth_rendering()
            mj_depth = renderer.render().copy()
            renderer.disable_depth_rendering()
            if mj_depth.ndim == 3:
                mj_depth = mj_depth[:, :, 0]
        renderer.close()
        cam_pos = data.cam_xpos[cam_id].copy()
        cam_mat = data.cam_xmat[cam_id].reshape(3, 3).copy()
        cam_fovy = float(model.cam_fovy[cam_id])
        if use_rendered_depth and mj_depth is not None:
            return self._detect_with_rendered_depth(
                rgb, mj_depth, prompts, cam_pos, cam_mat, cam_fovy, width, height
            )
        return self.detect(rgb, prompts, cam_pos, cam_mat, cam_fovy)

    def _render_cam(self, model, data, cam_id, width, height):

        renderer = _mujoco.Renderer(model, height, width)
        renderer.update_scene(data, camera=cam_id)
        rgb = renderer.render().copy()
        renderer.enable_depth_rendering()
        depth = renderer.render().copy()
        renderer.disable_depth_rendering()
        if depth.ndim == 3:
            depth = depth[:, :, 0]
        renderer.close()
        return (
            rgb,
            depth,
            data.cam_xpos[cam_id].copy(),
            data.cam_xmat[cam_id].reshape(3, 3).copy(),
            float(model.cam_fovy[cam_id]),
        )

    def _backproject_mask(self, mask, depth, cam_pos, cam_mat, fovy, w, h):
        ys, xs = np.where(mask > 0)
        if len(xs) == 0:
            return np.empty((0, 3))
        f = h / (2 * np.tan(np.deg2rad(fovy) / 2))
        mask_depths = depth[mask > 0]
        valid = (mask_depths > 0.01) & (mask_depths < 10)
        if valid.sum() == 0:
            return np.empty((0, 3))
        vx, vy, vd = (
            xs[valid].astype(np.float64),
            ys[valid].astype(np.float64),
            mask_depths[valid],
        )
        pts_cam = np.stack([(vx - w / 2) * vd / f, -(vy - h / 2) * vd / f, -vd], axis=1)
        return (cam_mat @ pts_cam.T).T + cam_pos

    def _detect_dual_camera(self, model, data, prompts, width, height):

        self.load_models(load_da3=False)
        cam_names = ["agentview", "robot0_eye_in_hand"]
        cam_ids = []
        for cn in cam_names:
            cid = _mujoco.mj_name2id(model, _mujoco.mjtObj.mjOBJ_CAMERA, cn)
            if cid >= 0:
                cam_ids.append((cn, cid))
        if not cam_ids:
            for alt in ["frontview", "birdview"]:
                cid = _mujoco.mj_name2id(model, _mujoco.mjtObj.mjOBJ_CAMERA, alt)
                if cid >= 0:
                    cam_ids.append((alt, cid))
                    break
        cam_data = []
        for cn, cid in cam_ids:
            rgb, depth, cam_pos, cam_mat, fovy = self._render_cam(
                model, data, cid, width, height
            )
            cam_data.append(
                {
                    "name": cn,
                    "rgb": rgb,
                    "depth": depth,
                    "cam_pos": cam_pos,
                    "cam_mat": cam_mat,
                    "fovy": fovy,
                }
            )
        cam_pil_imgs = [Image.fromarray(cd["rgb"]) for cd in cam_data]

        detections = []
        # One backbone encode per CAMERA, not per (prompt, camera) pair; the
        # loops below are prompt-major.
        with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
            cam_base_states = [self._sam3.set_image(img) for img in cam_pil_imgs]

        for prompt in prompts:
            per_cam = []
            for ci, cd in enumerate(cam_data):
                with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
                    state = self._sam3.set_text_prompt(
                        prompt=prompt, state={**cam_base_states[ci]}
                    )
                masks = state.get("masks", torch.tensor([]))
                scores = state.get("scores", torch.tensor([]))
                if masks.numel() == 0:
                    continue
                best_idx = int(scores.argmax())
                score = float(scores[best_idx]) * _sigma_n(int((scores > 0.30).sum()))
                mask = masks[best_idx].cpu().numpy().squeeze()
                ys, xs = np.where(mask > 0)
                cx = float(xs.mean()) if len(xs) > 0 else 0.0
                cy = float(ys.mean()) if len(ys) > 0 else 0.0
                pts = self._backproject_mask(
                    mask,
                    cd["depth"],
                    cd["cam_pos"],
                    cd["cam_mat"],
                    cd["fovy"],
                    width,
                    height,
                )
                if len(pts) > 0:
                    per_cam.append(
                        {
                            "pts": pts,
                            "score": score,
                            "cx": cx,
                            "cy": cy,
                            "centroid": np.median(pts, axis=0),
                            "mask": mask,
                            "cam": cd["name"],
                        }
                    )
            if not per_cam:
                continue
            if len(per_cam) == 2:
                dist = np.linalg.norm(per_cam[0]["centroid"] - per_cam[1]["centroid"])
                if dist < 0.20:
                    merged = np.vstack([r["pts"] for r in per_cam])
                else:
                    agent = [r for r in per_cam if r["cam"] == "agentview"]
                    merged = (
                        agent[0]["pts"]
                        if agent
                        else max(per_cam, key=lambda r: r["score"])["pts"]
                    )
            else:
                merged = per_cam[0]["pts"]
            if len(merged) > 20 and _cKDTree is not None:
                counts = _cKDTree(merged).query_ball_point(
                    merged, r=0.02, return_length=True
                )
                dense = counts >= 5
                if dense.sum() > 5:
                    merged = merged[dense]
            point_world = np.median(merged, axis=0)
            best = max(per_cam, key=lambda r: r["score"])
            det = ObjectDetection(
                label=prompt,
                confidence=best["score"],
                centroid_2d=(best["cx"], best["cy"]),
                mask_area=len(merged),
                depth_meters=float(
                    np.linalg.norm(point_world - cam_data[0]["cam_pos"])
                ),
                position_3d=point_world,
                mask=best["mask"],
            )
            detections.append(det)
        return detections

    @_serialized
    def _detect_with_rendered_depth(
        self,
        rgb,
        depth,
        prompts,
        cam_pos,
        cam_mat,
        cam_fovy,
        w,
        h,
        multi_instance_prompts=None,
        intrinsic_matrix=None,
        table_height=None,
        secondary_score=None,
    ):
        """
        SAM3 + hardware/rendered depth for exact 3D positions.
        """
        self.load_models(load_da3=False)
        multi_instance_prompts = multi_instance_prompts or set()
        pil_img = Image.fromarray(rgb)

        use_opencv = intrinsic_matrix is not None
        if use_opencv:
            fx = intrinsic_matrix[0, 0]
            fy = intrinsic_matrix[1, 1]
            cx_k = intrinsic_matrix[0, 2]
            cy_k = intrinsic_matrix[1, 2]
        else:
            fx = fy = cx_k = cy_k = None
        f = h / (2 * np.tan(np.deg2rad(cam_fovy) / 2))

        detections = []
        timer = StageTimer()
        # Encode the image once, reuse for every prompt. bf16 autocast is
        # required by SAM3 here (mixed-dtype crash without it).
        with timer.stage("encode"):
            with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
                base_state = self._sam3.set_image(pil_img)

        for prompt in prompts:
            with timer.stage("grounding"):
                with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
                    state = self._sam3.set_text_prompt(
                        prompt=prompt, state={**base_state}
                    )
            masks = state.get("masks", torch.tensor([]))
            scores = state.get("scores", torch.tensor([]))
            if masks.numel() == 0:
                continue

            sigma = _sigma_n(int((scores > 0.30).sum()))
            best_idx = int(scores.argmax())
            indices = self._select_mask_indices(
                prompt,
                masks,
                scores,
                best_idx,
                multi_instance_prompts,
                secondary_score=secondary_score,
            )

            for mask_idx in indices:
                with timer.stage("geometry"):
                    det = self._process_single_mask(
                        prompt,
                        masks[mask_idx],
                        float(scores[mask_idx]) * sigma,
                        depth,
                        cam_pos,
                        cam_mat,
                        cam_fovy,
                        w,
                        h,
                        fx,
                        fy,
                        cx_k,
                        cy_k,
                        f,
                        use_opencv,
                        intrinsic_matrix,
                        table_height,
                        rgb=rgb,
                    )
                if det is not None:
                    detections.append(det)

        logger.info(
            "[detect-timing] hw-depth %dx%d: %d prompt(s) -> %d mask(s) in "
            "%.2fs | %s",
            w,
            h,
            len(prompts),
            len(detections),
            timer.total(),
            timer.describe(),
        )

        # Cross-label dedup. Two prompts that resolved to the SAME mask are
        # one object; two prompts whose masks merely overlap because one thing
        # is sitting inside the other are two. dedup.same_object_2d draws that
        # line with mask IoU -- centroid distance alone deletes the placed
        # object out of its container. See perception/dedup.py.
        aware = dedup.containment_aware()
        deduped = []
        for det in detections:
            duplicate = False
            for i, existing in enumerate(deduped):
                if det.label == existing.label:
                    continue
                # Part-of guard: never dedup a part against its parent
                # ('mug handle' vs 'mug'). The parent mask can contain the
                # part, so their centroids collide and the part is lost.
                if det.label in existing.label or existing.label in det.label:
                    continue
                if dedup.same_object_2d(det, existing, _DEDUP_PX, aware=aware):
                    if det.confidence > existing.confidence:
                        deduped[i] = det
                    duplicate = True
                    break
            if not duplicate:
                deduped.append(det)
        return deduped

    # Score a non-best mask must clear to be considered a second instance of
    # the same prompt. A task whose second object is reliably dimmer or more
    # occluded than the first can relax this via the prompt registry's
    # fallback.relax_secondary_score, rather than lowering it globally and
    # admitting spurious instances into every other task.
    SECONDARY_MASK_SCORE = 0.50

    def _select_mask_indices(
        self,
        prompt,
        masks,
        scores,
        best_idx,
        multi_instance_prompts,
        secondary_score=None,
    ):
        """
        Select which mask indices to process (NMS for multi-instance).
        """
        if prompt not in multi_instance_prompts:
            return [best_idx]
        floor = (
            self.SECONDARY_MASK_SCORE if secondary_score is None else float(secondary_score)
        )
        candidates = [best_idx] + sorted(
            [
                i
                for i in range(len(scores))
                if i != best_idx and float(scores[i]) > floor
            ],
            key=lambda i: float(scores[i]),
            reverse=True,
        )
        kept_masks = []
        indices = []
        for ci in candidates:
            m = masks[ci].cpu().numpy().squeeze().astype(bool)
            m_ys, m_xs = np.where(m)
            m_cx, m_cy = float(m_xs.mean()), float(m_ys.mean())
            suppress = False
            for km in kept_masks:
                km_ys, km_xs = np.where(km)
                if (
                    np.sqrt(
                        (m_cx - float(km_xs.mean())) ** 2
                        + (m_cy - float(km_ys.mean())) ** 2
                    )
                    < 30
                ):
                    suppress = True
                    break
                inter = np.logical_and(m, km).sum()
                union = np.logical_or(m, km).sum()
                iou = inter / union if union > 0 else 0
                iomin = (
                    inter / min(m.sum(), km.sum()) if min(m.sum(), km.sum()) > 0 else 0
                )
                if iou > 0.3 or iomin > 0.5:
                    suppress = True
                    break
            if not suppress:
                indices.append(ci)
                kept_masks.append(m)
        return indices

    def _process_single_mask(
        self,
        prompt,
        mask_tensor,
        score,
        depth,
        cam_pos,
        cam_mat,
        cam_fovy,
        w,
        h,
        fx,
        fy,
        cx_k,
        cy_k,
        f,
        use_opencv,
        intrinsic_matrix,
        table_height,
        rgb=None,
    ):
        """
        Process a single SAM3 mask into an ObjectDetection.

        `rgb` is optional and used only to sample the mask's median colour;
        omitting it leaves `hsv_median` None and costs nothing else.
        """
        mask = mask_tensor.cpu().numpy().squeeze()
        ys, xs = np.where(mask > 0)
        if len(xs) == 0:
            return None

        cx, cy = float(xs.mean()), float(ys.mean())
        point_world, d_val = _cloud_median_world(
            mask,
            depth,
            cam_mat,
            cam_pos,
            fx=fx,
            fy=fy,
            cx_k=cx_k,
            cy_k=cy_k,
            fovy_deg=cam_fovy,
            use_opencv=use_opencv,
        )

        if point_world is None:
            ui, vi = int(cx), int(cy)
            if 0 <= vi < h and 0 <= ui < w and depth[vi, ui] > 0.01:
                d_val = float(depth[vi, ui])
                if use_opencv:
                    x_cam = (cx - cx_k) * d_val / fx
                    y_cam = (cy - cy_k) * d_val / fy
                    z_cam = d_val
                else:
                    x_cam = (cx - w / 2) * d_val / f
                    y_cam = -(cy - h / 2) * d_val / f
                    z_cam = -d_val
                point_world = cam_mat @ np.array([x_cam, y_cam, z_cam]) + cam_pos
            else:
                return None

        # Punch-through correction for translucent/concave objects
        point_world = self._punch_through_correction(
            prompt,
            mask,
            depth,
            point_world,
            d_val,
            xs,
            ys,
            cam_pos,
            cam_mat,
            cam_fovy,
            w,
            h,
            fx,
            fy,
            cx_k,
            cy_k,
            f,
            use_opencv,
            table_height,
        )

        # Orientation + OBB
        obb_mask_e = _top_layer_mask(mask, depth)
        img_angle_rad, ar, short_px_e, long_px_e = _pca_obb(obb_mask_e)
        orient_angle = img_angle_rad

        _wxy = _world_xy_pca_obb(
            mask,
            depth,
            cam_pos,
            cam_mat,
            return_confidence=True,
            fx=fx,
            fy=fy,
            cx_k=cx_k,
            cy_k=cy_k,
            fovy_deg=cam_fovy,
            use_opencv=use_opencv,
        )

        if long_px_e > 0 and d_val and d_val > 0.01:
            fx_b = (
                float(intrinsic_matrix[0, 0])
                if (use_opencv and intrinsic_matrix is not None)
                else (
                    h / (2 * np.tan(np.deg2rad(cam_fovy) / 2)) if cam_fovy else float(h)
                )
            )
            fy_b = (
                float(intrinsic_matrix[1, 1])
                if (use_opencv and intrinsic_matrix is not None)
                else fx_b
            )
            cx_b = (
                float(intrinsic_matrix[0, 2])
                if (use_opencv and intrinsic_matrix is not None)
                else w / 2.0
            )
            cy_b = (
                float(intrinsic_matrix[1, 2])
                if (use_opencv and intrinsic_matrix is not None)
                else h / 2.0
            )
            orient_angle = _compute_orientation(
                mask,
                depth,
                cam_mat,
                cam_pos,
                cam_fovy,
                cx,
                cy,
                h,
                w,
                img_angle_rad,
                long_px_e,
                d_val,
                use_opencv,
                fx_b,
                fy_b,
                cx_b,
                cy_b,
            )

        obb_minor_m_e = 0.0
        if short_px_e > 0 and d_val > 0.01:
            f_for_minor = (
                float(intrinsic_matrix[1, 1])
                if (intrinsic_matrix is not None)
                else (
                    h / (2 * np.tan(np.deg2rad(cam_fovy) / 2)) if cam_fovy else float(h)
                )
            )
            if f_for_minor > 0:
                obb_minor_m_e = float(short_px_e * d_val / f_for_minor)

        # Preserve the ray-plane answer BEFORE the point-cloud OBB overwrites
        # it (same guard as detect()); the grasp fallback reads these.
        plane_orientation_angle = (
            float(orient_angle) if orient_angle is not None else None
        )
        plane_aspect_ratio = float(ar) if ar is not None else None

        obb_conf_e = 0.0
        if _wxy[0] is not None:
            orient_angle = _wxy[0]
            ar = float(_wxy[1])
            obb_minor_m_e = float(_wxy[3])
            obb_conf_e = float(_wxy[4] or 0.0)

        hsv = mask_color_stats(rgb, mask > 0) if rgb is not None else None

        # World-Z top/bottom of the same depth cloud the position came from.
        # A container's rim and interior floor; the release height is computed
        # from these instead of a planner constant (control.release_height).
        _profile = mask_height_profile(
            mask,
            depth,
            cam_pos,
            cam_mat,
            fx=fx,
            fy=fy,
            cx_k=cx_k,
            cy_k=cy_k,
            fovy_deg=cam_fovy,
            use_opencv=use_opencv,
        )

        return ObjectDetection(
            label=prompt,
            confidence=score,
            centroid_2d=(cx, cy),
            mask_area=len(xs),
            depth_meters=d_val,
            position_3d=point_world,
            bbox=(float(xs.min()), float(ys.min()), float(xs.max()), float(ys.max())),
            mask=mask,
            orientation_angle=orient_angle,
            aspect_ratio=ar,
            plane_orientation_angle=plane_orientation_angle,
            plane_aspect_ratio=plane_aspect_ratio,
            obb_minor_m=obb_minor_m_e,
            obb_confidence=obb_conf_e,
            hsv_median=hsv,
            rim_z_m=_profile["rim_z"] if _profile else None,
            interior_z_m=_profile["interior_z"] if _profile else None,
            height_samples=_profile["n_valid"] if _profile else 0,
        )

    def _punch_through_correction(
        self,
        prompt,
        mask,
        depth,
        point_world,
        d_val,
        xs,
        ys,
        cam_pos,
        cam_mat,
        cam_fovy,
        w,
        h,
        fx,
        fy,
        cx_k,
        cy_k,
        f,
        use_opencv,
        table_height,
    ):
        """
        Correct Z for translucent objects where depth punches through to table.

        table_height is the work surface Z in the same world frame as the
        camera pose; None means the caller has no table, and no correction.
        """
        if point_world is None or table_height is None or len(xs) < 30:
            return point_world
        _valid_d = mask & (depth > 0.01)
        if _valid_d.sum() < 30:
            return point_world
        _vy, _vx = np.where(_valid_d)
        _vd = depth[_vy, _vx]
        if use_opencv:
            _vxc = (_vx - cx_k) * _vd / fx
            _vyc = (_vy - cy_k) * _vd / fy
            _vzc = _vd
        else:
            _vxc = (_vx - w / 2) * _vd / f
            _vyc = -(_vy - h / 2) * _vd / f
            _vzc = -_vd
        _pts_world_v = (cam_mat @ np.column_stack([_vxc, _vyc, _vzc]).T).T + cam_pos
        _wz = _pts_world_v[:, 2]
        _wz_p05, _wz_p50, _wz_p95 = np.percentile(_wz, [5, 50, 95])
        _wz_spread = float(_wz_p95 - _wz_p05)
        _bbox_h_px = float(ys.max() - ys.min())
        _table_z = float(table_height)
        _flat = _wz_spread < 0.040
        _on_table = _wz_p50 < (_table_z + 0.030)
        _tall_bbox = _bbox_h_px > 40
        _bbox_w_px = float(xs.max() - xs.min())
        # Use the oriented minor axis, not the axis-aligned bbox: an
        # elongated utensil lying diagonally spans a wide box in both image
        # axes and would wrongly pass the chunky gate. minAreaRect gives the
        # true object width regardless of yaw, so thin cutlery fails the gate
        # while a real translucent cup still passes.
        try:
            _rect = cv2.minAreaRect(np.column_stack([xs, ys]).astype(np.float32))
            _short_px = float(min(_rect[1])) or min(_bbox_h_px, _bbox_w_px)
        except Exception:
            _short_px = min(_bbox_h_px, _bbox_w_px)
        _fy_m = fy if (use_opencv and fy is not None) else f
        _obj_minor_m = _short_px * d_val / _fy_m if _fy_m and _fy_m > 0 else 0.0
        _chunky = _obj_minor_m > 0.040
        _img_ar = _bbox_h_px / max(_bbox_w_px, 1.0)
        _tall_shape = _img_ar > 1.8
        if _flat and _on_table and _tall_bbox and _chunky and _tall_shape:
            _fy_h = fy if (use_opencv and fy is not None) else f
            _est_h_m = max(float(_bbox_h_px * d_val / _fy_h), 0.050)
            _new_z = _table_z + _est_h_m / 2.0
            logger.info(
                "[punch-through] '%s': spread=%.3fm median=%.3f "
                "table=%.3f bbox_h=%.0fpx minor=%.3fm -> Z %.3f -> %.3f",
                prompt,
                _wz_spread,
                _wz_p50,
                _table_z,
                _bbox_h_px,
                _obj_minor_m,
                point_world[2],
                _new_z,
            )
            return np.array([float(point_world[0]), float(point_world[1]), _new_z])
        return point_world

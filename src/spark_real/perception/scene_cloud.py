"""Full-scene point cloud + per-point segmentation builder.

Contact-GraspNet (see ``contact_graspnet_infer.predict_grasps``) needs a
FULL-scene cloud plus a per-point segment map (0 = background, i+1 = i-th
detected object); an object-only cloud yields 0 grasps. This module factors
that cloud/segmap construction out of the ``/api/scene_pointcloud`` route so it
can be reused from a skill (``grasp_cgn``) without going through FastAPI.

The output frame is the robot base (world) frame: the birdview depth image is
backprojected with the camera intrinsics and lifted to world with the camera
extrinsic. Grasp poses returned by CGN on this cloud are therefore already in
the base frame the executor commands.
"""

from __future__ import annotations

import logging
from typing import Dict, List, Optional

import numpy as np

logger = logging.getLogger(__name__)


def build_scene_cloud(
    pipeline,
    prompts: List[str],
    multi_instance: bool = True,
    max_background: int = 20000,
    captures: Optional[dict] = None,
    merged: Optional[list] = None,
) -> Dict:
    """Capture + detect, then build a world-frame scene cloud + segment map.

    Args:
        pipeline: the ``SPARKPipeline`` (perception + cameras).
        prompts: SAM3 text prompts (object labels) to detect.
        multi_instance: pass-through to ``pipeline.detect``.
        max_background: subsample background points down to this many (all
            object points are always kept).
        captures: optional pre-captured frames (skips capture()).
        merged: optional pre-merged detections (skips capture()+detect()).

    Returns:
        dict with keys:
            camera          str: which camera the cloud came from
            points          (N, 3) float64 world-frame points
            segment_labels  (N,)  int32 per-point labels (0 = background)
            objects         list of {label, seg_id, n_points}
            detections      the merged ObjectDetection list (for downstream
                            orientation / OBB reuse)
    """
    if captures is None or merged is None:
        with pipeline.activity("capturing"):
            captures = pipeline.capture()
        if not captures:
            raise RuntimeError("No captures from cameras")
        with pipeline.activity("detecting"):
            all_dets = pipeline.detect(captures, prompts, multi_instance=multi_instance)
            merged = pipeline.merge_detections(all_dets)

    cam = "birdview" if "birdview" in captures else next(iter(captures))
    cam_data = captures[cam]
    depth = np.asarray(cam_data["depth"], dtype=float)
    cal = cam_data["calibration"]
    K = cal.intrinsic_matrix
    cr = cal.extrinsic
    H, W = depth.shape

    ys, xs = np.where((depth > 0.05) & (depth < 3.0))
    d = depth[ys, xs]
    fx, fy, cx, cy = K[0, 0], K[1, 1], K[0, 2], K[1, 2]
    xc = (xs - cx) * d / fx
    yc = (ys - cy) * d / fy
    pts_cam = np.stack([xc, yc, d], axis=1)
    pts_world = (cr[:3, :3] @ pts_cam.T).T + cr[:3, 3]

    labels = np.zeros(len(xs), dtype=np.int32)
    objs = []
    for oi, det in enumerate(merged):
        m = getattr(det, "mask", None)
        if m is None:
            continue
        m = np.asarray(m)
        if m.shape != (H, W):
            continue
        labels[m[ys, xs] > 0] = oi + 1
        objs.append(
            {
                "label": det.label,
                "seg_id": oi + 1,
                "n_points": int((labels == oi + 1).sum()),
            }
        )

    # Keep ALL object points; subsample background to keep CGN input light.
    obj_idx = np.where(labels > 0)[0]
    bg_idx = np.where(labels == 0)[0]
    if len(bg_idx) > max_background:
        bg_idx = np.random.choice(bg_idx, max_background, replace=False)
    keep = np.concatenate([obj_idx, bg_idx])

    return {
        "camera": cam,
        "points": pts_world[keep].astype(np.float64),
        "segment_labels": labels[keep].astype(np.int32),
        "objects": objs,
        "detections": list(merged),
    }


def resolve_segment_id(objects: List[dict], target_label: str) -> Optional[int]:
    """Match a target label to a segment id from ``build_scene_cloud`` objects.

    Case-insensitive; tolerates trailing instance numbers/whitespace on either
    side (e.g. "red block" matches "red block 2"). Returns the seg_id of the
    first (highest-confidence, since detections are merged best-first) match, or
    None.
    """

    def _norm(s: str) -> str:
        return str(s).lower().strip().rstrip("0123456789 ").strip()

    tgt = _norm(target_label)
    if not tgt:
        return None
    # Exact-normalized match first, then substring either direction.
    for o in objects:
        if _norm(o["label"]) == tgt:
            return int(o["seg_id"])
    for o in objects:
        lab = _norm(o["label"])
        if tgt in lab or lab in tgt:
            return int(o["seg_id"])
    return None

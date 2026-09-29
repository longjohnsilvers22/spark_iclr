"""Standalone Contact-GraspNet (PyTorch port) inference helper.

Wraps the vendored ``contact_graspnet_pytorch`` package so the rest of
spark_real can get 6-DoF grasps from a tabletop point cloud with a single
call, without touching the FastAPI grasp server or the running robot.

The underlying model is the pure-PyTorch re-implementation of
Contact-GraspNet (Sundermeyer et al., ICRA 2021). It requires NO compiled
CUDA ops (the Pointnet++ layers are plain PyTorch), so it works anywhere a
recent torch + CUDA build is available (torch 2.11+cu130, numpy 2.5).

Typical use
-----------
    from spark_real.perception.contact_graspnet_infer import predict_grasps

    grasps = predict_grasps(points_xyz)          # full-scene point cloud, Nx3
    for g in grasps:                             # sorted best-first
        T = np.eye(4); T[:3, :3] = g["R_3x3"]; T[:3, 3] = g["xyz"]
        # T is the 6-DoF gripper pose in the SAME frame as points_xyz
        # (OpenCV camera convention: +z = approach/into the object)

For denser, object-wise, background-free grasps, pass a per-point instance
segmentation via ``segment_labels`` (same length as points_xyz). Then the
model crops local regions around each object and filters grasp contacts to
the object surface; this is the recommended mode when you have SAM3 masks.

Grasp pose convention (Contact-GraspNet / OpenCV camera frame):
  * R[:, 2] (local +z) is the gripper approach direction (points INTO object)
  * R[:, 0] (local +x) is the gripper closing/baseline direction
  * xyz is the gripper BASE frame (fingertip midpoint 0.1034 m ahead along the approach)
  * width is the predicted gripper opening in meters (<= 0.08 for a Panda)
"""

from __future__ import annotations

import importlib.util
import os
import sys
import threading
from typing import Dict, List, Optional

import numpy as np
import torch

# The vendored Contact-GraspNet ships as the installed ``contact_graspnet_pytorch``
# package, so resolve the repo root from the package location instead of a
# hardcoded path. ``__file__`` is <repo>/contact_graspnet_pytorch/__init__.py,
# whose parent's parent is the repo root that holds ``checkpoints/`` and the
# ``Pointnet_Pointnet2_pytorch/`` submodule. An env override is honored but has
# no hardcoded default.
import contact_graspnet_pytorch as _cgn_pkg

_CGN_PKG_DIR = os.path.dirname(os.path.abspath(_cgn_pkg.__file__))
_CGN_REPO = os.environ.get("CONTACT_GRASPNET_REPO") or os.path.dirname(_CGN_PKG_DIR)
_CGN_CKPT_DIR = os.environ.get(
    "CONTACT_GRASPNET_CKPT",
    os.path.join(_CGN_REPO, "checkpoints", "contact_graspnet"),
)


def _install_cgn_models_shim() -> None:
    """Resolve the ``models`` package-name collision with EquiGraspFlow.

    CGN's ``contact_graspnet.py`` does ``from models.pointnet2_utils import ...``,
    expecting the vendored ``Pointnet_Pointnet2_pytorch/models`` package. But
    EquiGraspFlow (loaded at server start) registers its OWN top-level ``models``
    package in ``sys.modules`` first, so a plain ``from models...`` resolves to
    the wrong package and raises ``No module named 'models.pointnet2_utils'``.

    Fix: pre-register ``sys.modules['models.pointnet2_utils']`` by loading the
    vendored file directly via an explicit module spec. Because the fully
    qualified submodule name is already cached, CPython's importer returns it
    without ever touching the parent ``models`` entry, so EquiGraspFlow's
    ``models`` package is left untouched and both coexist in one process. The
    vendored ``pointnet2_utils`` only imports torch/numpy (no relative imports),
    so loading it in isolation is safe.
    """
    name = "models.pointnet2_utils"
    if name in sys.modules:
        return
    src = os.path.join(_CGN_REPO, "Pointnet_Pointnet2_pytorch", "models", "pointnet2_utils.py")
    spec = importlib.util.spec_from_file_location(name, src)
    if spec is None or spec.loader is None:
        raise ImportError(f"Cannot load vendored CGN pointnet2_utils from {src}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)


# Register the pointnet2_utils shim BEFORE importing the CGN estimator (its
# import chain triggers ``from models.pointnet2_utils import ...``).
_install_cgn_models_shim()

from contact_graspnet_pytorch import config_utils
from contact_graspnet_pytorch.checkpoints import CheckpointIO
from contact_graspnet_pytorch.contact_grasp_estimator import GraspEstimator

_ESTIMATOR = None
_LOCK = threading.Lock()


def _get_estimator():
    """Build the GraspEstimator and load the pretrained checkpoint (once).

    The estimator is a process-wide singleton (thread-safe construction) so
    the model weights are only loaded onto the GPU once.
    """
    global _ESTIMATOR
    if _ESTIMATOR is not None:
        return _ESTIMATOR
    with _LOCK:
        if _ESTIMATOR is not None:
            return _ESTIMATOR
        cfg = config_utils.load_config(_CGN_CKPT_DIR, batch_size=1)
        est = GraspEstimator(cfg)
        ckpt_io = CheckpointIO(
            checkpoint_dir=os.path.join(_CGN_CKPT_DIR, "checkpoints"),
            model=est.model,
        )
        ckpt_io.load("model.pt")
        est.model.eval()
        _ESTIMATOR = est
    return _ESTIMATOR


def _grasps_to_records(pred_grasps, scores, openings) -> List[Dict]:
    """Flatten the per-segment CGN output dicts into a flat, sorted list."""
    records: List[Dict] = []
    for key in pred_grasps:
        G = pred_grasps[key]
        if G is None or len(G) == 0:
            continue
        S = np.asarray(scores[key]).reshape(-1)
        W = openings.get(key) if isinstance(openings, dict) else None
        W = np.asarray(W).reshape(-1) if W is not None and np.size(W) else None
        for i in range(len(G)):
            T = np.asarray(G[i], dtype=np.float64)  # 4x4
            records.append(
                {
                    "xyz": T[:3, 3].copy(),
                    "R_3x3": T[:3, :3].copy(),
                    "T_4x4": T.copy(),
                    "score": float(S[i]) if i < len(S) else 0.0,
                    "width": float(W[i]) if (W is not None and i < len(W)) else float("nan"),
                    "segment": int(key) if np.isscalar(key) or isinstance(key, (int, np.integer)) else key,
                }
            )
    records.sort(key=lambda r: r["score"], reverse=True)
    return records


def predict_grasps(
    points_xyz: np.ndarray,
    colors: Optional[np.ndarray] = None,
    segment_labels: Optional[np.ndarray] = None,
    z_range: Optional[List[float]] = None,
    forward_passes: int = 1,
    max_grasps: Optional[int] = None,
) -> List[Dict]:
    """Predict 6-DoF grasps for a tabletop point cloud.

    Arguments:
        points_xyz: (N, 3) float array. Full-scene point cloud in meters, in
            the camera or world frame. Grasp poses are returned in this SAME
            frame. CGN was trained with the OpenCV camera convention
            (+x right, +y down, +z forward/into scene); if you feed a world
            frame the geometry is unchanged but the "approach into object"
            interpretation still holds relative to the returned R.
        colors: optional (N, 3) array (unused by the model; accepted for API
            symmetry / future visualization).
        segment_labels: optional (N,) int array of per-point instance ids
            (0 = background/ignored). When provided, the model crops local 3D
            regions around each object and filters grasp contacts to the
            object surface; denser and cleaner than full-scene mode.
        z_range: optional [zmin, zmax] to crop points by depth before
            inference (helps drop table/background). Applied to points_xyz[:, 2].
        forward_passes: number of batched forward passes; increase to sample
            more grasp contacts (default 1).
        max_grasps: if set, return only the top-N grasps by score.

    Returns:
        List of dicts sorted best-first, each with:
            xyz    (3,)   grasp center in the input frame
            R_3x3  (3, 3) rotation; column 2 is the approach axis
            T_4x4  (4, 4) homogeneous grasp pose
            score  float  contact confidence in [0, 1]
            width  float  predicted gripper opening (m), NaN if unavailable
            segment int   source segment id (-1 for full-scene mode)
    """
    pts = np.asarray(points_xyz, dtype=np.float32).reshape(-1, 3)
    if z_range is not None:
        m = (pts[:, 2] > z_range[0]) & (pts[:, 2] < z_range[1])
        seg_z = segment_labels[m] if segment_labels is not None else None
        pts = pts[m]
        segment_labels = seg_z
    if pts.shape[0] < 100:
        return []

    est = _get_estimator()

    pc_segments = {}
    local_regions = False
    filter_grasps = False
    if segment_labels is not None:
        segment_labels = np.asarray(segment_labels).reshape(-1)
        for sid in np.unique(segment_labels[segment_labels > 0]):
            seg_pts = pts[segment_labels == sid]
            if seg_pts.shape[0] >= 50:
                pc_segments[int(sid)] = seg_pts
        if pc_segments:
            local_regions = True
            filter_grasps = True

    with torch.no_grad():
        pred_grasps, scores, contact_pts, openings = est.predict_scene_grasps(
            pts,
            pc_segments=pc_segments,
            local_regions=local_regions,
            filter_grasps=filter_grasps,
            forward_passes=forward_passes,
        )

    records = _grasps_to_records(pred_grasps, scores, openings)
    if max_grasps is not None:
        records = records[:max_grasps]
    return records


if __name__ == "__main__":
    # Smoke test on a bundled example scene (depth+segmap -> point cloud).
    import argparse

    parser = argparse.ArgumentParser(description="Contact-GraspNet standalone smoke test")
    parser.add_argument("--np_path", default=os.path.join(_CGN_REPO, "test_data", "7.npy"))
    parser.add_argument("--use_segmap", action="store_true", help="use instance segmap (dense, object-wise)")
    parser.add_argument("--max", type=int, default=10)
    args = parser.parse_args()

    from contact_graspnet_pytorch.data import load_available_input_data

    segmap, rgb, depth, cam_K, pc_full, pc_colors = load_available_input_data(args.np_path, K=None)
    est = _get_estimator()
    if pc_full is None:
        pc_full, pc_segments, pc_colors = est.extract_point_clouds(
            depth, cam_K, segmap=segmap, rgb=rgb, z_range=[0.2, 1.8]
        )
        seg_labels = None
        if args.use_segmap:
            # rebuild a per-point label array aligned with pc_full is nontrivial
            # here; the segment_labels path is meant for live SAM3 point clouds.
            # This file-based test runs full-scene mode.
            pass
    print(f"Point cloud: {pc_full.shape}")
    grasps = predict_grasps(pc_full, max_grasps=args.max)
    print(f"Predicted {len(grasps)} grasps (showing top {min(args.max, len(grasps))}):")
    for g in grasps:
        print(f"  score={g['score']:.3f} width={g['width']:.3f} xyz={np.round(g['xyz'], 3)}")

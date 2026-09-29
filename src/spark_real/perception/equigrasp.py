"""
EquiGraspFlow SE(3) grasp generation (Lim et al., CoRL 2024).

Supports in-process and subprocess modes. Generates 6-DOF grasp pose
candidates from point clouds as 4x4 homogeneous transforms.
"""

from __future__ import annotations

import json
import logging
import os
import subprocess
import sys
import tempfile
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import List, Optional

import numpy as np

from spark_real.routes import state as _route_state

try:
    import torch
except ImportError:
    torch = None

try:
    import roma
except ImportError:
    roma = None

try:
    from omegaconf import OmegaConf
except ImportError:
    OmegaConf = None

logger = logging.getLogger(__name__)

_EQUIGRASP_ROOT = Path(
    os.environ.get(
        "EQUIGRASPFLOW_ROOT",
        Path(__file__).resolve().parent.parent.parent / "EquiGraspFlow",
    )
)
_PRETRAINED_PARTIAL = (
    _EQUIGRASP_ROOT / "train_results" / "pretrained_models" / "equigraspflow_partial"
)
_PRETRAINED_FULL = (
    _EQUIGRASP_ROOT / "train_results" / "pretrained_models" / "equigraspflow_full"
)
_CONDA_ENV = os.environ.get("EQUIGRASPFLOW_CONDA_ENV", "equigraspflow")

_SCALE = 8  # EquiGraspFlow training scale factor
# ACRONYM/NVIDIA convention: the pose translation is the gripper BASE link,
# with the fingertip midpoint one fixed depth ahead along +Z (Panda: 0.1034 m).
# Every SPARK consumer commands a fingertip/TCP frame, so positions returned
# from this module are TCP-corrected; raw base positions ship alongside.
_EGF_GRIPPER_DEPTH_M = float(os.environ.get("SPARK_EGF_GRIPPER_DEPTH_M", "0.1034"))


def _workspace_bounds() -> dict:
    """
    Return XYZ workspace bounds from SafeRobot config or FR3 defaults.
    """
    try:
        pipe = getattr(_route_state, "pipeline", None)
        if pipe is not None:
            safe = getattr(pipe, "_safe_robot", None) or getattr(pipe, "_robot", None)
            cfg = getattr(safe, "config", None)
            if cfg is not None and hasattr(cfg, "x_min"):
                return {
                    "x_min": float(cfg.x_min),
                    "x_max": float(cfg.x_max),
                    "y_min": float(cfg.y_min),
                    "y_max": float(cfg.y_max),
                    "z_min": float(cfg.z_min),
                    "z_max": float(cfg.z_max),
                }
    except Exception:
        pass
    return {
        "x_min": -0.85,
        "x_max": 0.85,
        "y_min": -0.85,
        "y_max": 0.85,
        "z_min": -0.05,
        "z_max": 0.70,
    }


@dataclass
class GraspCandidate:
    """
    A single 6-DOF grasp candidate in the camera frame.
    """

    position: np.ndarray  # (3,) grasp contact point [x, y, z]
    rotation: np.ndarray  # (3, 3) rotation matrix (gripper frame)
    approach: np.ndarray  # (3,) approach direction unit vector
    score: float  # grasp quality score (higher = better)
    width: float  # estimated gripper opening width (meters)

    def to_dict(self) -> dict:
        return {
            "position": self.position.tolist(),
            "rotation": self.rotation.tolist(),
            "approach": self.approach.tolist(),
            "score": self.score,
            "width": self.width,
        }

    @classmethod
    def from_dict(cls, d: dict) -> GraspCandidate:
        return cls(
            position=np.array(d["position"]),
            rotation=np.array(d["rotation"]),
            approach=np.array(d["approach"]),
            score=d["score"],
            width=d["width"],
        )

    def __repr__(self) -> str:
        p = self.position
        return (
            f"GraspCandidate(pos=[{p[0]:.3f},{p[1]:.3f},{p[2]:.3f}], "
            f"score={self.score:.3f}, width={self.width:.4f})"
        )


class EquiGraspGenerator:
    """
    SE(3) 6-DOF grasp generator. Modes: auto, inprocess, subprocess.
    """

    def __init__(
        self,
        model_path: Optional[str] = None,
        mode: str = "auto",
        device: str = "cuda:0",
        conda_env: Optional[str] = None,
    ):
        self._model_path = Path(model_path) if model_path else _PRETRAINED_PARTIAL
        self._device = device
        self._conda_env = conda_env or _CONDA_ENV
        self._model = None  # loaded on first use

        if mode == "auto":
            self._mode = "inprocess" if self._check_inprocess() else "subprocess"
        else:
            self._mode = mode

        logger.info(
            "EquiGraspGenerator: mode=%s, weights=%s, device=%s",
            self._mode,
            self._model_path.name,
            self._device,
        )

    @staticmethod
    def _check_inprocess() -> bool:
        """
        Return True if EquiGraspFlow can run in the current env.
        """
        if roma is None or OmegaConf is None or torch is None:
            return False
        # Check that the repo is importable
        egf_root = _EQUIGRASP_ROOT
        if not (egf_root / "models" / "equi_grasp_flow.py").exists():
            return False
        return True

    def _load_model(self):
        """
        Load model weights (called once on first use).
        """
        if self._model is not None:
            return

        if torch is None or OmegaConf is None:
            raise RuntimeError("torch/omegaconf not installed; EquiGraspFlow unavailable")

        egf_str = str(_EQUIGRASP_ROOT)
        if egf_str not in sys.path:
            sys.path.insert(0, egf_str)

        # get_model lives in the EquiGraspFlow repo, importable only after
        # the sys.path insert above.
        from models import get_model  # noqa: E402

        config_files = list(self._model_path.glob("*.yml"))
        if not config_files:
            raise FileNotFoundError(f"No .yml config found in {self._model_path}")
        cfg = OmegaConf.load(str(config_files[0]))
        ckpt_path = self._model_path / "model_best_val_loss.pkl"
        if not ckpt_path.exists():
            pkls = list(self._model_path.glob("*.pkl"))
            if not pkls:
                raise FileNotFoundError(
                    f"No checkpoint (.pkl) found in {self._model_path}"
                )
            ckpt_path = pkls[0]

        cfg.model.checkpoint = str(ckpt_path)

        self._model = get_model(cfg.model).to(self._device)
        self._model.eval()
        logger.info(
            "EquiGraspFlow model loaded from %s (%s)",
            ckpt_path.name,
            self._device,
        )

    @staticmethod
    def backproject_mask(
        depth: np.ndarray,
        mask: np.ndarray,
        intrinsics: np.ndarray,
        min_depth: float = 0.01,
        max_depth: float = 3.0,
    ) -> np.ndarray:
        """
        Back-project masked depth pixels to an (N,3) cloud (OpenCV convention).
        """
        ys, xs = np.where(mask > 0)
        if len(xs) == 0:
            return np.empty((0, 3), dtype=np.float32)

        depths = depth[ys, xs].astype(np.float64)
        valid = (depths > min_depth) & (depths < max_depth)
        if valid.sum() == 0:
            return np.empty((0, 3), dtype=np.float32)

        xs = xs[valid].astype(np.float64)
        ys = ys[valid].astype(np.float64)
        depths = depths[valid]

        fx, fy = intrinsics[0, 0], intrinsics[1, 1]
        cx, cy = intrinsics[0, 2], intrinsics[1, 2]
        x = (xs - cx) * depths / fx
        y = (ys - cy) * depths / fy
        z = depths

        return np.stack([x, y, z], axis=1).astype(np.float32)

    def generate_grasps(
        self,
        rgb: np.ndarray,
        depth: np.ndarray,
        mask: np.ndarray,
        intrinsics: np.ndarray,
        n_candidates: int = 10,
        cam_to_robot: Optional[np.ndarray] = None,
        max_approach_angle: float = 45.0,
        grip_end: str = "center",
        detection: Optional[object] = None,
    ) -> List[GraspCandidate]:
        """
        Generate SE(3) grasp candidates from depth+mask, sorted by score.
        """
        pts_cam = self.backproject_mask(depth, mask, intrinsics)
        if len(pts_cam) < 50:
            logger.warning(
                "EquiGraspFlow: insufficient points (%d < 50), skipping",
                len(pts_cam),
            )
            return []

        logger.info("EquiGraspFlow: %d mask points", len(pts_cam))
        if cam_to_robot is not None:
            R_cr = cam_to_robot[:3, :3]
            t_cr = cam_to_robot[:3, 3]
            pts_work = (R_cr @ pts_cam.T).T + t_cr
        else:
            pts_work = pts_cam

        pts = self._subsample(pts_work, n_points=1024)
        centroid = pts.mean(axis=0)
        pts_centered = pts - centroid
        if self._mode == "inprocess":
            Ts = self._generate_inprocess(pts_centered, n_candidates)
        else:
            Ts = self._generate_subprocess(pts_centered, n_candidates)

        if Ts is None or len(Ts) == 0:
            logger.warning("EquiGraspFlow: model returned no transforms")
            return []

        for T in Ts:
            T[:3, 3] = T[:3, 3] / _SCALE + centroid

        # Object major axis: prefer SAM3 OBB angle for elongated objects,
        # fall back to PCA on the point cloud for round objects.
        major_source = "pca"
        sam3_angle = (
            getattr(detection, "orientation_angle", None)
            if detection is not None
            else None
        )
        sam3_ar = (
            getattr(detection, "aspect_ratio", None) if detection is not None else None
        )
        if sam3_angle is not None and sam3_ar is not None and float(sam3_ar) >= 1.3:
            major_angle = float(sam3_angle)
            major_source = "sam3_obb"
        else:
            pts_xy = pts_work[:, :2]
            cov = np.cov(pts_xy.T)
            eigvals, eigvecs = np.linalg.eigh(cov)
            major_axis_xy = eigvecs[:, -1]
            major_angle = np.arctan2(major_axis_xy[1], major_axis_xy[0])
        logger.info(
            "EquiGraspFlow: major axis=%.1f deg (source=%s)",
            np.rad2deg(major_angle),
            major_source,
        )

        candidates = []
        n_angle_reject = 0
        n_workspace_reject = 0
        n_flipped = 0
        for i, T in enumerate(Ts):
            rot = T[:3, :3]
            pos = T[:3, 3]
            approach = rot[:, 2]
            # Flip upside-down twins about closing axis
            if approach[2] > 0.0:
                closing = rot[:, 0]
                R_flip = 2.0 * np.outer(closing, closing) - np.eye(3)
                # Contact-preserving flip: the reflection sends approach to
                # -approach about the base origin, which moves the fingertip
                # midpoint by 2*depth. Translate so the contacts stay put.
                pos = pos + 2.0 * _EGF_GRIPPER_DEPTH_M * approach
                rot = R_flip @ rot
                approach = rot[:, 2]
                n_flipped += 1

            down_component = -approach[2]
            if down_component < np.cos(np.deg2rad(max_approach_angle)):
                n_angle_reject += 1
                continue

            ws = _workspace_bounds()
            # Grasp z-floor is lower than the motion z_min: a block resting on
            # the table has valid top-down grasps whose TCP sits at mid-block,
            # BELOW the conservative motion z_min (e.g. flat plates on a table at
            # ~-0.278 give grasp TCPs ~-0.27..-0.278, which the -0.27 motion floor
            # would wrongly reject). The execution descent hard-stop
            # (table_z_floor+2mm) still bounds the actual motion, so relaxing
            # the *filter* here is safe.
            grasp_z_margin = float(os.environ.get("SPARK_GRASP_Z_MARGIN", "0.020"))
            if (
                pos[2] < ws["z_min"] - grasp_z_margin
                or pos[2] > ws["z_max"]
                or not (
                    ws["x_min"] < pos[0] < ws["x_max"]
                    and ws["y_min"] < pos[1] < ws["y_max"]
                )
            ):
                n_workspace_reject += 1
                continue

            closing = rot[:, 0]
            closing_angle = np.arctan2(closing[1], closing[0])
            angle_diff = abs(closing_angle - major_angle)
            angle_diff = angle_diff % np.pi
            if angle_diff > np.pi / 2:
                angle_diff = np.pi - angle_diff
            perp_score = angle_diff / (np.pi / 2)

            score = 0.4 * down_component + 0.6 * perp_score

            if grip_end != "center":
                centroid = pts_work.mean(axis=0)
                major_dir = np.array([np.cos(major_angle), np.sin(major_angle), 0])
                proj = np.dot(pos[:2] - centroid[:2], major_dir[:2])
                projs_all = pts_work[:, :2] @ major_dir[:2]
                extent = projs_all.max() - projs_all.min()
                if extent > 0.01:
                    norm_proj = proj / (extent / 2)
                    if grip_end == "back":
                        end_score = max(norm_proj, 0)
                    else:
                        end_score = max(-norm_proj, 0)
                    score = 0.3 * down_component + 0.4 * perp_score + 0.3 * end_score

            width = self._estimate_width(pts_work, pos, rot)

            candidates.append(
                GraspCandidate(
                    # EGF translation is the gripper base; SPARK consumers
                    # command TCP frames.
                    position=pos + _EGF_GRIPPER_DEPTH_M * approach,
                    rotation=rot,
                    approach=approach / (np.linalg.norm(approach) + 1e-8),
                    score=score,
                    width=width,
                )
            )

        logger.info(
            "EquiGraspFlow: %d/%d passed filter "
            "(%d angle rejected, %d workspace rejected, %d flipped to top-down)",
            len(candidates),
            len(Ts),
            n_angle_reject,
            n_workspace_reject,
            n_flipped,
        )
        if candidates:
            best = max(candidates, key=lambda c: c.score)
            logger.info(
                "  Best: pos=(%.3f,%.3f,%.3f) approach=(%.2f,%.2f,%.2f) score=%.3f",
                *best.position,
                *best.approach,
                best.score,
            )

        # Sort by score descending
        candidates.sort(key=lambda c: c.score, reverse=True)
        return candidates[:n_candidates]

    def _generate_inprocess(
        self, pts_centered: np.ndarray, n_candidates: int
    ) -> Optional[List[np.ndarray]]:
        """
        Run inference in-process with GIL-contention mitigations.
        """
        if torch is None:
            raise RuntimeError("torch not installed; EquiGraspFlow unavailable")

        _paused_recorder = None
        try:
            _ep = getattr(_route_state, "current_episode", None)
            if _ep is not None and hasattr(_ep, "_stop"):
                if hasattr(_ep, "pause"):
                    _ep.pause()
                    _paused_recorder = _ep
        except Exception:
            pass

        _prev_switch = sys.getswitchinterval()
        sys.setswitchinterval(0.05)

        try:
            self._load_model()

            pc = (
                torch.from_numpy(pts_centered * _SCALE)
                .float()
                .unsqueeze(0)
                .to(self._device)
            )
            pc = pc.transpose(1, 2)
            nums = torch.tensor([n_candidates], device=self._device)

            t0 = time.time()
            with torch.no_grad():
                results = self._model.sample(pc, nums)
            elapsed = time.time() - t0
            logger.info("EquiGraspFlow: model.sample completed in %.2fs", elapsed)

            Ts_batch = results[0]  # (n_candidates, 4, 4)
            Ts = [Ts_batch[i].cpu().numpy() for i in range(len(Ts_batch))]
            return Ts
        finally:
            sys.setswitchinterval(_prev_switch)
            if _paused_recorder is not None and hasattr(_paused_recorder, "resume"):
                try:
                    _paused_recorder.resume()
                except Exception:
                    pass

    def _generate_subprocess(
        self, pts_centered: np.ndarray, n_candidates: int
    ) -> Optional[List[np.ndarray]]:
        """
        Run EquiGraspFlow inference in an isolated conda environment.
        """
        with tempfile.TemporaryDirectory(prefix="equigrasp_") as tmpdir:
            # Save input point cloud
            pts_path = os.path.join(tmpdir, "points.npy")
            np.save(pts_path, pts_centered)

            out_path = os.path.join(tmpdir, "grasps.npy")

            script = f"""
import sys
sys.path.insert(0, '{_EQUIGRASP_ROOT}')

import numpy as np
import torch
from omegaconf import OmegaConf
from models import get_model

# Load config + checkpoint
import glob
model_dir = '{self._model_path}'
cfg_files = glob.glob(model_dir + '/*.yml')
cfg = OmegaConf.load(cfg_files[0])

ckpt_files = glob.glob(model_dir + '/*.pkl')
best = [f for f in ckpt_files if 'best' in f]
ckpt = best[0] if best else ckpt_files[0]
cfg.model.checkpoint = ckpt

device = '{self._device}'
model = get_model(cfg.model).to(device)
model.eval()

# Load point cloud
pts = np.load('{pts_path}')
pc = torch.from_numpy(pts * {_SCALE}).float().unsqueeze(0).to(device)
nums = torch.tensor([{n_candidates}], device=device)

with torch.no_grad():
    results = model.sample(pc, nums)

Ts = results[0].cpu().numpy()  # (n_candidates, 4, 4)
np.save('{out_path}', Ts)
print('SUCCESS')
"""
            script_path = os.path.join(tmpdir, "run_equigrasp.py")
            with open(script_path, "w") as f:
                f.write(script)

            result = subprocess.run(
                ["conda", "run", "-n", self._conda_env, "python", script_path],
                capture_output=True,
                text=True,
                timeout=120,
            )

            if "SUCCESS" not in result.stdout:
                logger.error(
                    "EquiGraspFlow subprocess failed:\nstdout: %s\nstderr: %s",
                    result.stdout[-500:],
                    result.stderr[-500:],
                )
                return None

            Ts = np.load(out_path)  # (n_candidates, 4, 4)
            return [Ts[i] for i in range(len(Ts))]

    def select_best_grasp(
        self,
        candidates: List[GraspCandidate],
        current_tcp: Optional[np.ndarray] = None,
        prefer_side: bool = False,
    ) -> Optional[GraspCandidate]:
        """
        Select best grasp by score + reachability + side preference.
        """
        if not candidates:
            return None

        scored = []
        for c in candidates:
            s = c.score

            if current_tcp is not None:
                dist = np.linalg.norm(c.position - current_tcp)
                s += max(0.0, 0.2 * (1.0 - dist / 0.5))
            if prefer_side:
                horizontality = 1.0 - abs(c.approach[2])
                s += 0.3 * horizontality

            scored.append((s, c))

        scored.sort(key=lambda x: x[0], reverse=True)
        best_score, best = scored[0]
        logger.info(
            "Selected grasp: pos=(%.3f,%.3f,%.3f) approach=(%.2f,%.2f,%.2f) "
            "score=%.3f width=%.4f",
            *best.position,
            *best.approach,
            best_score,
            best.width,
        )
        return best

    @staticmethod
    def _subsample(pts: np.ndarray, n_points: int = 1024) -> np.ndarray:
        """
        Randomly subsample or upsample to exactly n_points.
        """
        n = len(pts)
        if n == n_points:
            return pts
        if n > n_points:
            idx = np.random.choice(n, n_points, replace=False)
        else:
            # Upsample by repeating + adding small noise
            idx = np.concatenate(
                [
                    np.arange(n),
                    np.random.choice(n, n_points - n, replace=True),
                ]
            )
        return pts[idx]

    @staticmethod
    def _estimate_width(
        pts: np.ndarray,
        grasp_pos: np.ndarray,
        rot: np.ndarray,
        radius: float = 0.02,
    ) -> float:
        """
        Estimate gripper width from local point spread along x-axis.
        """
        diffs = pts - grasp_pos
        dists = np.linalg.norm(diffs, axis=1)
        near = diffs[dists < radius]
        if len(near) < 3:
            return 0.04
        x_axis = rot[:, 0]
        projections = near @ x_axis
        width = float(projections.max() - projections.min())
        return np.clip(width, 0.005, 0.085)


_generator: Optional[EquiGraspGenerator] = None


def _get_generator() -> EquiGraspGenerator:
    global _generator
    if _generator is None:
        _generator = EquiGraspGenerator()
    return _generator


def generate_grasps_subprocess(
    pts: np.ndarray,
    num_grasps: int = 20,
    prefer_side: bool = False,
) -> List[dict]:
    """
    Generate SE(3) grasps from world-frame points (backward-compat API).
    """
    gen = _get_generator()
    pts = pts.astype(np.float32)
    if len(pts) < 50:
        return []

    pts_sub = gen._subsample(pts, n_points=1024)
    centroid = pts_sub.mean(axis=0)
    pts_centered = pts_sub - centroid

    if gen._mode == "inprocess":
        Ts = gen._generate_inprocess(pts_centered, num_grasps)
    else:
        Ts = gen._generate_subprocess(pts_centered, num_grasps)

    if Ts is None or len(Ts) == 0:
        return []

    results = []
    for T in Ts:
        T[:3, 3] = T[:3, 3] / _SCALE + centroid
        rot = T[:3, :3]
        pos = T[:3, 3]
        # Upstream convention: +Z is the approach, INTO the object. Negating
        # it would top-rank underside grasps and invert the too-vertical veto.
        approach = rot[:, 2]
        approach = approach / (np.linalg.norm(approach) + 1e-8)

        # Heuristic preference, NOT a model score: EquiGraspFlow emits SE(3)
        # transforms only. Penalize approaches pointing up (underside grasps).
        z_penalty = max(0.0, approach[2])
        score = 1.0 - 0.5 * z_penalty

        # Translation is the gripper BASE; consumers command TCP frames, so
        # return the fingertip-midpoint position.
        tcp_pos = pos + _EGF_GRIPPER_DEPTH_M * approach
        results.append(
            {
                "position": tcp_pos,
                "base_position": pos,
                "rotation": rot,
                "approach": approach,
                "score": score,
                "score_is_heuristic": True,
                "width": 0.04,
            }
        )

    results.sort(key=lambda r: r["score"], reverse=True)
    return results


def select_best_grasp(
    grasps: List[dict],
    prefer_side: bool = False,
) -> Optional[dict]:
    """
    Select best grasp dict by score (backward-compat API).
    """
    if not grasps:
        return None

    scored = []
    for g in grasps:
        s = g["score"]
        if prefer_side:
            horizontality = 1.0 - abs(g["approach"][2])
            s += 0.3 * horizontality
        scored.append((s, g))

    scored.sort(key=lambda x: x[0], reverse=True)
    return scored[0][1]


def generate_grasps_from_detection(
    detection,
    camera_data: dict,
    n_candidates: int = 10,
    generator: Optional[EquiGraspGenerator] = None,
    max_approach_angle: float = 45.0,
    grip_end: str = "center",
) -> List[GraspCandidate]:
    """
    Generate SE(3) grasps from an ObjectDetection + camera capture.
    """
    if generator is None:
        generator = _get_generator()

    rgb = camera_data["rgb"]
    depth = camera_data["depth"]
    cal = camera_data["calibration"]

    mask = getattr(detection, "mask", None)
    if mask is None:
        logger.warning(
            "Detection '%s' has no mask, cannot generate grasps", detection.label
        )
        return []

    intrinsics = cal.intrinsic_matrix
    cam_to_robot = cal.extrinsic  # 4x4 camera -> robot base

    return generator.generate_grasps(
        rgb=rgb,
        depth=depth,
        mask=mask,
        intrinsics=intrinsics,
        n_candidates=n_candidates,
        cam_to_robot=cam_to_robot,
        max_approach_angle=max_approach_angle,
        grip_end=grip_end,
        detection=detection,
    )

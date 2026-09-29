"""NVIDIA GraspGen integration for SPARK.

Diffusion-based 6-DOF grasp generation.

Two backends, selected at call time:

1. **ZMQ-server (preferred)**: when ``SPARK_GRASPGEN_SERVER_URL`` is set
   (e.g. ``tcp://127.0.0.1:5556``), point clouds are sent to a long-running
   GraspGen ZMQ server (upstream canonical pattern, see
   ``client-server/README.md`` in the NVlabs/GraspGen repo). One persistent
   CUDA context, no per-trial driver init. This is what every parallel
   sweep should use.

2. **Subprocess fallback (legacy)**: spawns a fresh ``graspgen`` conda
   env per call. Reliable in single-process settings but blows up with
   ``handle_0 INTERNAL ASSERT FAILED`` when N>1 workers all try to init
   CUDA on shared GPUs. Only kept for environments where the server isn't
   running.

Mirrors equigrasp.py's API.

Usage:
    from spark_real.perception.graspgen import generate_grasps
    grasps = generate_grasps(point_cloud, num_grasps=50)
    # grasps: list of dicts with 'position', 'rotation', 'approach',
    #         'width', 'confidence'
"""

import os
import subprocess
import tempfile
import logging
from typing import Optional

import numpy as np

_GRASPGEN_GRIPPER_DEPTH_M = float(os.environ.get("SPARK_GRASPGEN_GRIPPER_DEPTH_M", "0.10527"))

logger = logging.getLogger(__name__)

GRASPGEN_ROOT = os.path.expanduser('~/spark/src/graspgen')
GRIPPER_CONFIG = os.path.expanduser(
    '~/spark/datasets/graspgen_models/checkpoints/graspgen_franka_panda.yml'
)
CONDA_ENV = 'graspgen'


# ---------------------------------------------------------------------------
# ZMQ-server backend (preferred for multi-process sweeps)
# ---------------------------------------------------------------------------

_ZMQ_CLIENT = None


def _get_zmq_client():
    """Lazy-load the ``GraspGenClient`` ZMQ client, cached per process."""
    global _ZMQ_CLIENT
    if _ZMQ_CLIENT is not None:
        return _ZMQ_CLIENT
    url = os.environ.get('SPARK_GRASPGEN_SERVER_URL')
    if not url:
        return None
    try:
        # The upstream zmq_client only depends on pyzmq + msgpack + numpy
        # (no torch / CUDA), so it imports cleanly from any env that has
        # those three.
        import sys
        if GRASPGEN_ROOT not in sys.path:
            sys.path.insert(0, GRASPGEN_ROOT)
        from grasp_gen.serving.zmq_client import GraspGenClient
    except Exception as e:
        logger.warning("GraspGen ZMQ client unavailable (%s); "
                       "falling back to subprocess", e)
        return None
    # Parse tcp://host:port
    host, port = '127.0.0.1', 5556
    if url.startswith('tcp://'):
        rest = url[len('tcp://'):]
        if ':' in rest:
            h, p = rest.rsplit(':', 1)
            host, port = h, int(p)
    try:
        _ZMQ_CLIENT = GraspGenClient(host=host, port=port,
                                       wait_for_server=False)
        if not _ZMQ_CLIENT.health_check():
            logger.warning("GraspGen server at %s:%d failed health check; "
                           "falling back to subprocess", host, port)
            _ZMQ_CLIENT = None
            return None
        logger.info("GraspGen ZMQ client connected to %s:%d", host, port)
        return _ZMQ_CLIENT
    except Exception as e:
        logger.warning("GraspGen ZMQ connect failed (%s); falling back", e)
        return None


def _generate_grasps_zmq(point_cloud: np.ndarray,
                          num_grasps: int = 50) -> Optional[list]:
    """Call the GraspGen ZMQ server, return list-of-dicts or None on failure."""
    client = _get_zmq_client()
    if client is None:
        return None
    try:
        # Server expects object-centered point cloud; restore world frame
        # after by shifting translations back by the centroid (same logic
        # as the legacy subprocess path).
        pc = np.asarray(point_cloud, dtype=np.float32)
        if len(pc) > 2048:
            idx = np.random.choice(len(pc), 2048, replace=False)
            pc = pc[idx]
        center = pc.mean(axis=0)
        pc_centered = pc - center
        poses, confs = client.infer(
            pc_centered,
            num_grasps=num_grasps,
            topk_num_grasps=-1,
            min_grasps=min(10, num_grasps),
            max_tries=3,
            remove_outliers=True,
        )
        if len(poses) == 0:
            return []
        # poses are (N, 4, 4) homogeneous transforms in object-centered frame;
        # shift translations back to world frame.
        poses = np.asarray(poses, dtype=np.float64).copy()
        poses[:, :3, 3] = poses[:, :3, 3] + center
        grasps = []
        for i in range(poses.shape[0]):
            grasps.append({
                'position': poses[i, :3, 3].astype(np.float64),
                'rotation': poses[i, :3, :3].astype(np.float64),
                'approach': poses[i, :3, 2].astype(np.float64),
                'width': 0.08,
                'confidence': float(confs[i]),
            })
        return grasps
    except Exception as e:
        logger.warning("GraspGen ZMQ inference failed (%s); falling back", e)
        return None


def generate_grasps(point_cloud: np.ndarray, num_grasps: int = 50,
                     gripper_config: str = None) -> list:
    """Dispatch to ZMQ server when configured, else subprocess.

    Single entry point that callers should use. Selection happens via
    ``SPARK_GRASPGEN_SERVER_URL`` env var.
    """
    zmq_result = _generate_grasps_zmq(point_cloud, num_grasps=num_grasps)
    if zmq_result is not None:
        return zmq_result
    return generate_grasps_subprocess(point_cloud, num_grasps=num_grasps,
                                       gripper_config=gripper_config)


def generate_grasps_subprocess(point_cloud: np.ndarray,
                               num_grasps: int = 50,
                               gripper_config: str = None) -> list:
    """Generate 6-DOF grasps from a point cloud via GraspGen subprocess.

    Args:
        point_cloud: (N, 3) point cloud in world coordinates.
        num_grasps: number of grasp candidates to sample.
        gripper_config: path to gripper YAML (default: franka_panda).

    Returns:
        list of dicts with keys: 'position' (3,), 'rotation' (3,3),
            'approach' (3,) world-frame, 'width' float, 'confidence' float.
    """
    if gripper_config is None:
        gripper_config = GRIPPER_CONFIG

    # GraspGen wants a centered point cloud in canonical object frame.
    pc = np.asarray(point_cloud, dtype=np.float32)
    # Downsample to 2048 (franka_panda.yml num_points=2048).
    if len(pc) > 2048:
        idx = np.random.choice(len(pc), 2048, replace=False)
        pc = pc[idx]

    with tempfile.TemporaryDirectory(prefix='graspgen_') as tmpdir:
        pc_path = os.path.join(tmpdir, 'pc.npy')
        out_path = os.path.join(tmpdir, 'grasps.npz')
        np.save(pc_path, pc)

        script = f"""
import sys, numpy as np, torch
sys.path.insert(0, '{GRASPGEN_ROOT}')
from grasp_gen.grasp_server import GraspGenSampler, load_grasp_cfg

cfg = load_grasp_cfg('{gripper_config}')
sampler = GraspGenSampler(cfg)

pc_world = np.load('{pc_path}')
center = pc_world.mean(axis=0)
pc_centered = pc_world - center  # GraspGen expects object-centered coords

grasps, conf = GraspGenSampler.run_inference(
    object_pc=pc_centered,
    grasp_sampler=sampler,
    num_grasps={num_grasps},
    topk_num_grasps=-1,
    min_grasps=10,
    max_tries=3,
    remove_outliers=True,
)

if len(grasps) == 0:
    np.savez('{out_path}', poses=np.zeros((0, 4, 4)), conf=np.zeros(0), center=center)
    print('EMPTY')
else:
    poses = grasps.cpu().numpy() if torch.is_tensor(grasps) else np.asarray(grasps)
    # Shift translations back into world frame.
    poses[:, :3, 3] = poses[:, :3, 3] + center
    conf_np = conf.cpu().numpy() if torch.is_tensor(conf) else np.asarray(conf)
    np.savez('{out_path}', poses=poses, conf=conf_np, center=center)
    print(f'SUCCESS {{len(poses)}}')
"""
        script_path = os.path.join(tmpdir, 'run_gg.py')
        with open(script_path, 'w') as f:
            f.write(script)

        result = subprocess.run(
            ['bash', '-c',
             f'eval "$(conda shell.bash hook 2>/dev/null)" && '
             f'conda activate {CONDA_ENV} && '
             f'export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True && '
             f'python {script_path}'],
            capture_output=True, text=True, timeout=120,
        )

        if 'SUCCESS' not in result.stdout and 'EMPTY' not in result.stdout:
            logger.warning(f"GraspGen failed: {result.stderr[-400:]}")
            return []

        data = np.load(out_path)
        poses = data['poses']
        confs = data['conf']
        grasps = []
        for i in range(poses.shape[0]):
            pose = poses[i]
            approach = pose[:3, 2].astype(np.float64)  # Z = approach
            base_pos = pose[:3, 3].astype(np.float64)
            grasps.append({
                # GraspGen anchors the pose at the gripper BASE link, with the
                # contact midpoint gripper_depth ahead along +Z (franka_panda:
                # 0.10527 m). Consumers command TCP frames, so 'position' is
                # TCP-corrected; the raw base pose ships alongside. Returning
                # the raw base pose closes the fingers ~10 cm short of the
                # object.
                'position': base_pos + _GRASPGEN_GRIPPER_DEPTH_M * approach,
                'base_position': base_pos,
                'rotation': pose[:3, :3].astype(np.float64),
                'approach': approach,
                'width': 0.08,
                'confidence': float(confs[i]),
            })
        return grasps


def select_best_grasp(grasps: list, prefer_side: bool = False) -> dict:
    """Select best grasp from candidates, ranked by confidence + optional side-bias."""
    if not grasps:
        return None
    scored = []
    for g in grasps:
        s = g.get('confidence', 0.0)
        if prefer_side:
            # Penalize top-down for flat objects (plate/nut)
            s += (1.0 - abs(g['approach'][2])) * 0.5
        scored.append((s, g))
    scored.sort(key=lambda x: x[0], reverse=True)
    return scored[0][1]

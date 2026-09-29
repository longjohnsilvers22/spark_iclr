"""Contact-GraspNet grasp generation + top-down selection (CaP-X recipe).

Contact-GraspNet emits FREE 6-DoF grasps. For the UR10e + Robotiq top-down
rig a POST-HOC selection is done (the CaP-X approach):

  1. Run ``predict_grasps`` on the full-scene world cloud + segment map.
  2. For each grasp, approach = R[:, 2] (points into the object). Compute the
     top-down ``alignment = -dot(approach, [0,0,1])`` (=1 for a perfectly
     straight-down grasp). Keep grasps with ``alignment > TOPDOWN_ALIGN_THRESH``
     and pick the highest score  -> mode "cgn_topdown".
  3. If NONE pass (flat / simple objects where CGN only proposes side grasps),
     FALL BACK to a top-down OVERRIDE: keep the best-score grasp's POSITION but
     force our downward ``grasp_orientation`` (optionally yawed to the object's
     OBB major axis)  -> mode "override".

CGN -> TCP offset: CGN returns the gripper BASE frame, NOT the fingertip
midpoint. Upstream constructs grasp_t = contact + (thickness/2)*base_dir
- 0.1034*approach with center_to_tip = 0.0, so the fingertip baseline sits
0.1034 m AHEAD of the returned xyz along the approach.
The TCP is the gripper TIP (pendant getTCPOffset). The returned base point is
translated by ``CGN_TCP_OFFSET_M`` along the approach to land the tool where
CGN intends: 0.1034 m of that recovers the contact plane, and the remainder
(~1.7 cm at the 0.12 default) is Robotiq pad engagement past it. Tune via
``SPARK_CGN_TCP_OFFSET_M``. Sign convention: +offset moves ALONG approach
(deeper toward/into the object). For the override case the CGN approach is
discarded, so no along-approach shift is applied; the raw grasp center is used
and the skill backs off along +Z for the pregrasp.

The cloud passed in is already in the robot base (world) frame, so returned
poses are directly commandable.
"""

from __future__ import annotations

import logging
import os
from typing import Dict, List, Optional

import numpy as np
from scipy.spatial.transform import Rotation

logger = logging.getLogger(__name__)

# Default CGN grasp-center -> gripper-TIP translation along the approach axis (m).
CGN_TCP_OFFSET_M = 0.12
# Top-down alignment gate: alignment = -dot(approach, +Z). 0.8 ~= <36.9 deg tilt.
TOPDOWN_ALIGN_THRESH = 0.8


def _tcp_offset_m() -> float:
    try:
        return float(os.environ.get("SPARK_CGN_TCP_OFFSET_M", CGN_TCP_OFFSET_M))
    except (TypeError, ValueError):
        return CGN_TCP_OFFSET_M


def select_grasp(
    grasps: List[Dict],
    target_seg_id: Optional[int] = None,
    grasp_orientation=(2.103, -2.329, 0.059),
    tcp_offset_m: Optional[float] = None,
    align_thresh: float = TOPDOWN_ALIGN_THRESH,
    obb_yaw_rad: Optional[float] = None,
) -> Optional[Dict]:
    """Select a top-down grasp from a CGN record list (pure, no GPU/robot).

    Args:
        grasps: list of dicts from ``predict_grasps`` (xyz, R_3x3, score,
            width, segment). Poses are assumed to be in the WORLD frame.
        target_seg_id: if given, only consider grasps whose ``segment`` matches
            (falls back to all grasps if the target has none).
        grasp_orientation: rotvec (3,) of our downward TCP orientation, used for
            the override mode (and optionally yawed by ``obb_yaw_rad``).
        tcp_offset_m: CGN center -> TCP translation along approach; defaults to
            env ``SPARK_CGN_TCP_OFFSET_M`` / ``CGN_TCP_OFFSET_M``.
        align_thresh: top-down alignment gate.
        obb_yaw_rad: optional world-Z yaw (rad) applied to the override
            orientation to match the object's OBB major axis.

    Returns:
        dict {xyz, orient_rotvec, R_3x3, width, score, mode, alignment,
        approach, seg_id, n_candidates} or None if there were no grasps.
    """
    if not grasps:
        return None
    if tcp_offset_m is None:
        tcp_offset_m = _tcp_offset_m()

    # Restrict to the target segment when it has matching grasps.
    pool = grasps
    if target_seg_id is not None:
        seg_pool = [g for g in grasps if int(g.get("segment", -1)) == int(target_seg_id)]
        if seg_pool:
            pool = seg_pool
        else:
            logger.warning(
                "cgn_select: no grasps for seg_id=%s; considering all %d grasps",
                target_seg_id,
                len(grasps),
            )

    z_axis = np.array([0.0, 0.0, 1.0])

    def _approach(g):
        R = np.asarray(g["R_3x3"], dtype=float).reshape(3, 3)
        return R[:, 2]

    def _alignment(g):
        return float(-np.dot(_approach(g), z_axis))

    # (a) Best top-down grasp: alignment > thresh, max score.
    topdown = [(g, _alignment(g)) for g in pool]
    topdown = [(g, a) for (g, a) in topdown if a > align_thresh]
    if topdown:
        g, align = max(topdown, key=lambda ga: ga[0].get("score", 0.0))
        R = np.asarray(g["R_3x3"], dtype=float).reshape(3, 3)
        approach = R[:, 2]
        center = np.asarray(g["xyz"], dtype=float).reshape(3)
        tcp_xyz = center + tcp_offset_m * approach
        return {
            "xyz": tcp_xyz.tolist(),
            "orient_rotvec": Rotation.from_matrix(R).as_rotvec().tolist(),
            "R_3x3": R.tolist(),
            "width": float(g.get("width", float("nan"))),
            "score": float(g.get("score", 0.0)),
            "mode": "cgn_topdown",
            "alignment": align,
            "approach": approach.tolist(),
            "seg_id": int(g.get("segment", -1)),
            "n_candidates": len(pool),
            "tcp_offset_m": tcp_offset_m,
        }

    # (b) Override: best-score grasp CONTACT POINT + forced top-down
    # orientation. CGN's xyz is the gripper BASE, set back tcp_offset_m along
    # the (tilted) approach; using it raw puts the descent 6-10 cm laterally
    # off the object for the tilted approaches that trigger this branch.
    # Recover the contact midpoint first.
    g = max(pool, key=lambda gg: gg.get("score", 0.0))
    center = (np.asarray(g["xyz"], dtype=float).reshape(3)
              + tcp_offset_m * _approach(g))
    base = Rotation.from_rotvec(np.asarray(grasp_orientation, dtype=float))
    if obb_yaw_rad is not None:
        base = Rotation.from_rotvec([0.0, 0.0, float(obb_yaw_rad)]) * base
    R_over = base.as_matrix()
    return {
        # Discard CGN approach in override; use the raw grasp center. The skill
        # backs off along +Z for the pregrasp and descends straight down.
        "xyz": center.tolist(),
        "orient_rotvec": base.as_rotvec().tolist(),
        "R_3x3": R_over.tolist(),
        "width": float(g.get("width", float("nan"))),
        "score": float(g.get("score", 0.0)),
        "mode": "override",
        "alignment": _alignment(g),
        "approach": _approach(g).tolist(),
        "seg_id": int(g.get("segment", -1)),
        "n_candidates": len(pool),
        "tcp_offset_m": tcp_offset_m,
    }


def generate_and_select(
    points_xyz: np.ndarray,
    segment_labels: np.ndarray,
    target_seg_id: Optional[int] = None,
    grasp_orientation=(2.103, -2.329, 0.059),
    tcp_offset_m: Optional[float] = None,
    align_thresh: float = TOPDOWN_ALIGN_THRESH,
    obb_yaw_rad: Optional[float] = None,
    forward_passes: int = 1,
    max_grasps: Optional[int] = None,
) -> Optional[Dict]:
    """Run Contact-GraspNet on a world cloud, then select a top-down grasp.

    Import-guarded: CGN/torch are imported lazily so this module stays
    importable without them. Returns the ``select_grasp`` dict (with an added
    ``total_grasps`` field), or None if CGN produced no grasps.
    """
    # Lazy import so the module (and the skill) load without torch/CGN.
    from spark_real.perception.contact_graspnet_infer import predict_grasps

    grasps = predict_grasps(
        np.asarray(points_xyz, dtype=np.float32),
        segment_labels=np.asarray(segment_labels),
        forward_passes=forward_passes,
        max_grasps=max_grasps,
    )
    logger.info(
        "cgn: predicted %d grasps (segments=%s)",
        len(grasps),
        sorted({int(g.get("segment", -1)) for g in grasps}),
    )
    sel = select_grasp(
        grasps,
        target_seg_id=target_seg_id,
        grasp_orientation=grasp_orientation,
        tcp_offset_m=tcp_offset_m,
        align_thresh=align_thresh,
        obb_yaw_rad=obb_yaw_rad,
    )
    if sel is not None:
        sel["total_grasps"] = len(grasps)
    return sel

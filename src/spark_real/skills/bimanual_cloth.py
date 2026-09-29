"""
Bimanual cloth-fold orchestrator + shared helpers for the ANON-LAB rig.

This module holds the registry-discoverable orchestrator skill plus the
helpers shared with the composable primitives, which live in
skills/bimanual_cloth_primitives.py:

  - bimanual_shirt_fold    : orchestrator. Sleeves (grasp both, synced lift,
    arc each to garment-center y), RE-PERCEIVE, then hem (pinch both corners,
    synced lift, simultaneous drape to the collar). Re-perceives between
    phases instead of replaying t=0 coordinates.

The composable leaf chains (bimanual_grasp_points, bimanual_lift_together,
bimanual_arc_drape) are the registry-discoverable counterparts of the
standalone fold (scripts/fold_tshirt_v3.py) and live in
bimanual_cloth_primitives.py; they import the shared helpers (_p, _cloth_cfg,
_require_bimanual, _load_cross_arm_transforms, GRASP_ORIENT_0DEG, DEFAULTS)
from this module.

Execution surface: these run on the BimanualScoreExecutor. Motion is driven
through the per-arm single-arm executors (executor._arm_executors[arm], which
expose _move_to(position, orientation_rotvec, velocity) with per-arm IK and
workspace protection) so the existing single-arm motion path is reused
verbatim. Grippers and the SSG-48 raw readout go through the bimanual driver
(executor.driver.grippers.for_arm(arm)). Synchronization mirrors the script's
threaded both()/synced() helpers.

Fold params resolve with precedence: BT param > family profile cloth() block
> hardcoded default, exactly like skills/cloth_fold.py. The ANON-LAB cloth block
supplies hem_x_offset_right_m / hem_x_offset_left_m / left_jaw_yaw_flip_deg.

Untested: the standalone fold runs on the bamboo driver plus a
BimanualPyrokiPlanner with per-arm base offsets and the left-z correction,
whereas the server's per-arm executors use the single-arm fr3 IK backend. The
geometry and sequencing are faithful ports, but the per-arm-IK motion path has
not been exercised. See docs/bimanual_fold_reference_bt.yaml.
"""

from __future__ import annotations

import json
import logging
import os
import time
from typing import Dict, List, Optional, Tuple

import numpy as np

from spark_real.skills.primitives import _result
from spark_real.skills.registry import spark_skill
from spark_real.skills import cloth_geometry as geom

logger = logging.getLogger(__name__)

# Cross-arm calibration written by the guided web calibrator. Same file
# scripts/detect.py:load_calibration reads.
# Schema: cal["arms"]["right"|"left"]["T_zed_to_base"] (4x4, row-major).
_CALIB_PATH = os.path.expanduser("~/.spark_real/calibration_bimanual.json")


# Top-down jaw orientation (rotvec), matching skills/cloth_fold.py.
GRASP_ORIENT_0DEG = [np.pi, 0.0, 0.0]

# Defaults ported from the standalone fold. Overridable per-call
# and (for the offsets) via the family profile cloth() block.
DEFAULTS = {
    "hover_z": 0.05,
    "grasp_z": -0.035,
    "lift_z": 0.130,
    "land_z": 0.025,
    "arc_peak": 0.040,
    "hem_arc_peak_scale": 1.5,
    "land_x": 0.74,
    "sleeve_y_offset_right_m": -0.015,
    "sleeve_drape_fraction": 0.80,
    "hem_x_offset_right_m": 0.030,
    "hem_x_offset_left_m": 0.020,
    "left_jaw_yaw_flip_deg": 180.0,
    # The LEFT tool plate is rotated ~90 deg from the RIGHT, so the left jaw
    # yaw carries a +90 deg mount offset on top of the camera flip.
    "left_jaw_yaw_mount_deg": 90.0,
    "move_vel": 0.22,
    "arc_vel": 0.16,
    "approach_vel": 0.08,
    "force": 15.0,
    # SSG-48 raw position >= this means the jaw closed on air (abort signal).
    "empty_raw_threshold": 250,
    "sleeve_arc_steps": 8,
    "hem_arc_steps": 10,
}


def _cloth_cfg(executor) -> dict:
    """
    Family profile cloth() block, or {} when no pipeline/profile.
    """
    pipeline = getattr(executor, "_pipeline", None)
    profile = getattr(pipeline, "profile", None) if pipeline is not None else None
    if profile is not None and hasattr(profile, "cloth"):
        try:
            return dict(profile.cloth() or {})
        except Exception:  # pragma: no cover - defensive
            return {}
    return {}


def _p(params: dict, cfg: dict, key: str):
    """
    Resolve a fold param: BT param > profile cloth block > DEFAULTS.
    """
    if key in params and params[key] is not None:
        return params[key]
    if key in cfg and cfg[key] is not None:
        return cfg[key]
    return DEFAULTS[key]


def _require_bimanual(executor, name: str):
    """
    Return (ok, error_result_or_None). Bimanual skills need two arms.
    """
    if (
        getattr(executor, "_arm_executors", None) is None
        or getattr(executor, "driver", None) is None
    ):
        return False, _result(
            name, False, f"{name} requires the bimanual executor (two arms)", 0.0
        )
    return True, None


def _load_cross_arm_transforms() -> Optional[Tuple[np.ndarray, np.ndarray]]:
    """
    Load (T_zed_to_base_right, T_zed_to_base_left) 4x4 matrices.

    Reads ``~/.spark_real/calibration_bimanual.json`` (the file
    scripts/detect.py:load_calibration uses). Returns None if the file or
    the expected keys are missing, so callers fail their skill cleanly.
    """
    try:
        cal = json.loads(open(_CALIB_PATH).read())
        T_r = np.asarray(cal["arms"]["right"]["T_zed_to_base"], dtype=float).reshape(
            4, 4
        )
        T_l = np.asarray(cal["arms"]["left"]["T_zed_to_base"], dtype=float).reshape(
            4, 4
        )
        return T_r, T_l
    except (OSError, KeyError, ValueError) as exc:
        logger.warning(
            "[bimanual_cloth] cross-arm calibration load failed " "(%s): %s",
            _CALIB_PATH,
            exc,
        )
        return None


def _external_camera_entry(executor):
    """
    Return the ZED external CameraEntry from the bimanual pipeline's
    camera_registry, or None.
    """
    pipeline = getattr(executor, "_pipeline", None)
    registry = getattr(pipeline, "camera_registry", None) if pipeline else None
    if registry is None:
        return None
    try:
        return registry.get("external")
    except Exception:
        return None


def _redetect(
    executor, prompts: List[str]
) -> Tuple[Optional[Dict[str, dict]], Optional[dict]]:
    """
    Re-run perception on the external (ZED) camera, in the RIGHT base frame.

    Captures RGB+depth from the bimanual pipeline's ``camera_registry``
    ``external`` role, loads the cross-arm transforms from the bimanual
    calibration JSON, and runs SAM3 via the pipeline's perception backend
    with the right-arm ``T_zed_to_base`` as the camera extrinsic (so
    every ``position_3d`` lands in the RIGHT arm base frame, matching
    scripts/detect.py).

    Returns ``(detection_map, bundle)`` where:
      * detection_map: label -> {"position_3d": (right-frame xyz),
        "_mask": HxW mask, "orientation_angle", "aspect_ratio"}.
      * bundle: {"depth", "K", "T_r", "T_l"} reused by hem geometry.
    Either element is None on failure; the orchestrator handles that.
    """
    pipeline = getattr(executor, "_pipeline", None)
    perception = getattr(pipeline, "_perception", None) if pipeline else None
    entry = _external_camera_entry(executor)
    tfm = _load_cross_arm_transforms()
    if pipeline is None or perception is None or entry is None or tfm is None:
        logger.warning(
            "[bimanual_shirt_fold] redetect prerequisites missing "
            "(perception=%s external_cam=%s calib=%s)",
            perception is not None,
            entry is not None,
            tfm is not None,
        )
        return None, None
    T_r, T_l = tfm

    try:
        rgb, depth = entry.device.read(depth=True)
    except Exception as exc:
        logger.warning("[bimanual_shirt_fold] external camera read failed: %s", exc)
        return None, None
    if rgb is None or depth is None:
        logger.warning("[bimanual_shirt_fold] external camera returned no frame")
        return None, None

    K = np.asarray(entry.calibration.intrinsic_matrix, dtype=float)
    h, w = rgb.shape[:2]
    fovy = float(2 * np.degrees(np.arctan(h / (2 * K[1, 1]))))
    try:
        dets = perception._detect_with_rendered_depth(
            rgb=rgb,
            depth=depth,
            prompts=prompts,
            cam_pos=T_r[:3, 3],
            cam_mat=T_r[:3, :3],
            cam_fovy=fovy,
            w=w,
            h=h,
            intrinsic_matrix=K,
            table_height=-0.08,
        )
    except Exception as exc:
        logger.warning("[bimanual_shirt_fold] SAM3 detect failed: %s", exc)
        return None, None

    out: Dict[str, dict] = {}
    for d in dets or []:
        out[getattr(d, "label", "")] = {
            "position_3d": getattr(d, "position_3d", None),
            "_mask": getattr(d, "mask", None),
            "orientation_angle": getattr(d, "orientation_angle", 0.0),
            "aspect_ratio": getattr(d, "aspect_ratio", 1.0),
        }
    bundle = {"depth": np.asarray(depth), "K": K, "T_r": T_r, "T_l": T_l}
    return (out or None), bundle


@spark_skill(
    name="bimanual_shirt_fold",
    description=(
        "Bimanual t-shirt fold orchestrator (Panda left + FR3 right + SSG-48). "
        "Chains the cloth primitives end to end: (1) detect garment+sleeves, "
        "grasp both sleeve tips with per-sleeve OBB-minor-axis yaw, synced "
        "lift, arc-drape each sleeve to garment-center y; (2) RE-PERCEIVE the "
        "garment; (3) pinch both hem corners (from the depth-valid cloud, "
        "lowest-x band) simultaneously, synced lift, simultaneous arc-drape to "
        "the collar (land_x). Re-perceives between phases rather than replaying "
        "t=0 coordinates. Offsets and the left-jaw yaw flip come from the "
        "family profile cloth() block. Requires the bimanual executor; abort "
        "signal is an SSG-48 raw >= empty_raw_threshold on grasp."
    ),
    params={
        "instruction": "natural-language task (for logging)",
        "skip_sleeves": "skip the sleeve phase (default false)",
        "skip_hem": "skip the hem phase (default false)",
        "land_x": "hem drape landing x toward the collar (default 0.74)",
        "force": "cloth pinch force (default 15N)",
    },
)
def bimanual_shirt_fold(executor, params: dict):
    """
    Orchestrator. Re-perceives between phases; each phase reuses the
    composable primitives in bimanual_cloth_primitives so the planner can also
    chain them directly.
    """
    # imported here to break a circular import: bimanual_cloth_primitives
    # imports the shared helpers from this module at its top level.
    from spark_real.skills.bimanual_cloth_primitives import (
        bimanual_arc_drape,
        bimanual_grasp_points,
        bimanual_lift_together,
    )

    t0 = time.time()
    ok, err = _require_bimanual(executor, "bimanual_shirt_fold")
    if not ok:
        return err
    cfg = _cloth_cfg(executor)
    do_sleeves = not bool(params.get("skip_sleeves", False))
    do_hem = not bool(params.get("skip_hem", False))
    force = float(_p(params, cfg, "force"))

    notes: List[str] = []

    # PHASE 1: sleeves.
    if do_sleeves:
        det, bundle = _redetect(executor, ["garment", "left sleeve", "right sleeve"])
        if det is None:
            det = dict(executor.detection_map or {})
        ls = det.get("left sleeve")
        rs = det.get("right sleeve")
        garment = det.get("garment") or det.get("shirt")
        if ls is None or rs is None or garment is None:
            return _result(
                "bimanual_shirt_fold",
                False,
                "sleeve phase needs 'garment','left sleeve',"
                "'right sleeve' detections",
                time.time() - t0,
            )

        # Per-arm sleeve grasp points + yaw from the outer-edge tip and
        # OBB-minor-axis jaw yaw. RIGHT arm grabs the camera-LEFT sleeve, LEFT
        # arm grabs the camera-RIGHT sleeve. r_grab is in the RIGHT base frame;
        # l_grab in the LEFT base frame.
        try:
            r_grab, yaw_r, l_grab, yaw_l = _sleeve_targets(ls, rs, garment, bundle)
        except Exception as exc:
            logger.warning("[bimanual_shirt_fold] sleeve geometry failed: %s", exc)
            r_grab = yaw_r = l_grab = yaw_l = None
        if r_grab is None or l_grab is None:
            # Fall back to mask-centroid position_3d + plain top-down yaw.
            r_grab = _pick_point(ls, "right")
            l_grab = _pick_point(rs, "left")
            yaw_r = _field(ls, "jaw_yaw_right")
            yaw_l = _field(rs, "jaw_yaw_left")
            if l_grab is not None and bundle is not None:
                # position_3d is in the RIGHT frame; the LEFT arm needs it in
                # the LEFT frame.
                l_grab = geom.right_to_left(l_grab, bundle["T_r"], bundle["T_l"])
        if r_grab is None or l_grab is None:
            return _result(
                "bimanual_shirt_fold",
                False,
                "could not resolve sleeve grasp points",
                time.time() - t0,
            )
        r_grab = np.asarray(r_grab, dtype=float).copy()
        l_grab = np.asarray(l_grab, dtype=float).copy()
        r_grab[1] += float(_p(params, cfg, "sleeve_y_offset_right_m"))

        g_yr = _field(garment, "center_y_right")
        g_yl = _field(garment, "center_y_left")
        frac = float(_p(params, cfg, "sleeve_drape_fraction"))
        # Drape each sleeve toward the garment center y in its own frame.
        r_center = (
            geom.drape_to_center_y(r_grab[1], g_yr, frac)
            if g_yr is not None
            else r_grab[1]
        )
        l_center = (
            geom.drape_to_center_y(l_grab[1], g_yl, frac)
            if g_yl is not None
            else l_grab[1]
        )

        grasp_res = bimanual_grasp_points(
            executor,
            {
                "left_point": l_grab.tolist(),
                "right_point": r_grab.tolist(),
                "left_yaw_rad": yaw_l,
                "right_yaw_rad": yaw_r,
                "force": force,
            },
        )
        if not grasp_res.success:
            return _result(
                "bimanual_shirt_fold",
                False,
                f"sleeve grasp: {grasp_res.message}",
                time.time() - t0,
            )
        lift_res = bimanual_lift_together(executor, {})
        if not lift_res.success:
            return _result(
                "bimanual_shirt_fold",
                False,
                f"sleeve lift: {lift_res.message}",
                time.time() - t0,
            )
        drape_res = bimanual_arc_drape(
            executor,
            {
                "both_arms": True,
                "steps": int(DEFAULTS["sleeve_arc_steps"]),
                "left_land_xy": [l_grab[0], l_center],
                "right_land_xy": [r_grab[0], r_center],
                "left_yaw_rad": yaw_l,
                "right_yaw_rad": yaw_r,
            },
        )
        if not drape_res.success:
            return _result(
                "bimanual_shirt_fold",
                False,
                f"sleeve drape: {drape_res.message}",
                time.time() - t0,
            )
        executor.driver.open_gripper()
        executor.state.holding["left"] = False
        executor.state.holding["right"] = False
        notes.append("sleeves folded")

    # PHASE 2: RE-PERCEIVE, then hem.
    if do_hem:
        det, bundle = _redetect(executor, ["garment"])
        garment = (det or {}).get("garment") or (det or {}).get("shirt")
        hem_R = hem_L = None
        yaw_r = yaw_l = None
        if garment is not None and bundle is not None:
            try:
                hem_R, hem_L, yaw_r, yaw_l = _hem_targets(garment, bundle, cfg, params)
            except Exception as exc:
                logger.warning("[bimanual_shirt_fold] hem geometry failed: %s", exc)
        if hem_R is None or hem_L is None:
            # Fall back to explicit hem points if the planner supplied them.
            if params.get("right_hem_point") and params.get("left_hem_point"):
                hem_R = np.asarray(params["right_hem_point"], dtype=float)
                hem_L = np.asarray(params["left_hem_point"], dtype=float)
            else:
                return _result(
                    "bimanual_shirt_fold",
                    False,
                    "hem phase needs a re-detected garment + external ZED "
                    "capture + cross-arm calibration "
                    "(~/.spark_real/calibration_bimanual.json), or explicit "
                    "*_hem_point params",
                    time.time() - t0,
                )

        grasp_res = bimanual_grasp_points(
            executor,
            {
                "left_point": hem_L.tolist(),
                "right_point": hem_R.tolist(),
                "left_yaw_rad": yaw_l,
                "right_yaw_rad": yaw_r,
                "force": force,
            },
        )
        if not grasp_res.success:
            return _result(
                "bimanual_shirt_fold",
                False,
                f"hem grasp: {grasp_res.message}",
                time.time() - t0,
            )
        lift_res = bimanual_lift_together(executor, {})
        if not lift_res.success:
            return _result(
                "bimanual_shirt_fold",
                False,
                f"hem lift: {lift_res.message}",
                time.time() - t0,
            )
        drape_res = bimanual_arc_drape(
            executor,
            {
                "both_arms": True,
                "steps": int(DEFAULTS["hem_arc_steps"]),
                "arc_peak": float(_p(params, cfg, "arc_peak"))
                * float(DEFAULTS["hem_arc_peak_scale"]),
                "land_x": float(_p(params, cfg, "land_x")),
                "left_yaw_rad": yaw_l,
                "right_yaw_rad": yaw_r,
            },
        )
        if not drape_res.success:
            return _result(
                "bimanual_shirt_fold",
                False,
                f"hem drape: {drape_res.message}",
                time.time() - t0,
            )
        executor.driver.open_gripper()
        executor.state.holding["left"] = False
        executor.state.holding["right"] = False
        notes.append("hem folded")

    return _result(
        "bimanual_shirt_fold",
        True,
        f"bimanual t-shirt fold complete ({', '.join(notes)})",
        time.time() - t0,
    )


def _field(det: dict, key: str):
    """
    Read a numeric field from a detection dict, or None.
    """
    if not isinstance(det, dict):
        return None
    v = det.get(key)
    return float(v) if v is not None else None


def _pick_point(det: dict, arm: str) -> Optional[np.ndarray]:
    """
    Resolve a per-arm grasp point for a sleeve detection.

    Prefers the detect.py-style outer-edge / per-frame fields; falls back to
    position_3d. RIGHT arm uses the *_right fields, LEFT arm the *_left.
    """
    if not isinstance(det, dict):
        return None
    suffix = "right" if arm == "right" else "left"
    for key in (f"outer_edge_{suffix}", f"{suffix}_frame", "position_3d"):
        v = det.get(key)
        if v is not None:
            return np.asarray(v, dtype=float)
    return None


def _body_ref_px(garment: dict, bundle: dict) -> Optional[np.ndarray]:
    """
    Shirt-body centroid in PIXELS, the reference for the sleeve tip band.

    Mirrors scripts/detect.py: the centroid of the garment-body mask, with an
    image-center fallback when the mask is unavailable.
    """
    mask = garment.get("_mask") if isinstance(garment, dict) else None
    if mask is not None:
        m = np.asarray(mask) > 0
        bys, bxs = np.where(m)
        if len(bxs) > 0:
            return np.array([float(bxs.mean()), float(bys.mean())])
    depth = bundle.get("depth") if bundle else None
    if depth is not None:
        h, w = np.asarray(depth).shape[:2]
        return np.array([w / 2.0, h / 2.0])
    return None


def _sleeve_targets(
    ls: dict, rs: dict, garment: dict, bundle: Optional[dict]
) -> Tuple[
    Optional[np.ndarray], Optional[float], Optional[np.ndarray], Optional[float]
]:
    """
    Per-arm sleeve grasp tips + OBB-minor jaw yaws.

    Computes, from each sleeve mask + depth + intrinsics + cross-arm
    extrinsics (carried in ``bundle``):
      * RIGHT arm (grasps the camera-LEFT sleeve ``ls``): outer-edge tip in
        the RIGHT base frame + OBB-minor yaw in the right frame.
      * LEFT arm (grasps the camera-RIGHT sleeve ``rs``): outer-edge tip
        transformed into the LEFT base frame + OBB-minor yaw in the left
        frame (the +90 mount + 180 flip are added downstream in
        _orient_rotvec, exactly like the hem phase).

    Returns ``(r_grab, yaw_r, l_grab, yaw_l)``; raises / returns Nones when
    the masks, depth, intrinsics, or extrinsics are missing.
    """
    if bundle is None:
        return None, None, None, None
    depth = bundle.get("depth")
    K = bundle.get("K")
    T_r = bundle.get("T_r")
    T_l = bundle.get("T_l")
    ls_mask = ls.get("_mask") if isinstance(ls, dict) else None
    rs_mask = rs.get("_mask") if isinstance(rs, dict) else None
    if any(v is None for v in (depth, K, T_r, T_l, ls_mask, rs_mask)):
        return None, None, None, None
    depth = np.asarray(depth)
    K = np.asarray(K)
    T_r = np.asarray(T_r)
    T_l = np.asarray(T_l)
    ref_px = _body_ref_px(garment, bundle)
    if ref_px is None:
        return None, None, None, None

    # RIGHT arm: camera-left sleeve tip, primary (right) frame.
    r_tip, _ = geom.compute_sleeve_outer_edge(
        np.asarray(ls_mask), ref_px, depth, K, T_r
    )
    yaw_r, _ = geom.compute_jaw_yaw_minor(np.asarray(ls_mask), depth, K, T_r, T_l, T_r)

    # LEFT arm: camera-right sleeve tip; computed in the primary (right) frame
    # then mapped into the LEFT base frame for the left executor.
    l_tip_r, _ = geom.compute_sleeve_outer_edge(
        np.asarray(rs_mask), ref_px, depth, K, T_r
    )
    _, yaw_l = geom.compute_jaw_yaw_minor(np.asarray(rs_mask), depth, K, T_r, T_l, T_r)
    if r_tip is None or l_tip_r is None:
        return None, None, None, None
    l_tip = geom.right_to_left(l_tip_r, T_r, T_l)
    return (
        np.asarray(r_tip, dtype=float),
        yaw_r,
        np.asarray(l_tip, dtype=float),
        yaw_l,
    )


def _hem_targets(
    garment: dict, bundle: dict, cfg: dict, params: dict
) -> Tuple[np.ndarray, np.ndarray, float, float]:
    """
    Compute per-arm hem grasp points + yaw from a re-detected garment.

    Uses the garment mask + depth + intrinsics + cross-arm extrinsics
    (carried in the ``bundle`` returned by :func:`_redetect`) to build the
    depth-valid cloud, pick the two hem corners, apply the ANON-LAB x offsets,
    and compute the OBB-major hem yaw with the left 180 deg flip. Pure
    delegation to cloth_geometry. Raises on missing inputs.
    """
    mask = garment.get("_mask")
    depth = bundle.get("depth")
    K = bundle.get("K")
    T_r = bundle.get("T_r")
    T_l = bundle.get("T_l")
    if any(v is None for v in (mask, depth, K, T_r, T_l)):
        raise ValueError("hem geometry needs mask/depth/intrinsics/extrinsics")
    mask = np.asarray(mask)
    depth = np.asarray(depth)
    K = np.asarray(K)
    T_r = np.asarray(T_r)
    T_l = np.asarray(T_l)

    cloud = geom.backproject_cloud(mask, depth, K, T_r)
    right_c, left_c = geom.hem_corners_from_cloud(cloud)
    hem_R, hem_L = geom.apply_hem_x_offsets(
        right_c,
        left_c,
        T_r,
        T_l,
        float(_p(params, cfg, "hem_x_offset_right_m")),
        float(_p(params, cfg, "hem_x_offset_left_m")),
    )
    yaw_r = geom.yaw_from_major_axis(mask, depth, K, T_r)
    # Left flip is applied inside _orient_rotvec; pass the same yaw_r and let
    # the orientation builder add the mount + flip. Returning yaw_r for both
    # keeps the per-arm flip in one place.
    return hem_R, hem_L, yaw_r, yaw_r

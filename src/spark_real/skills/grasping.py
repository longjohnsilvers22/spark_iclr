"""
SE(3) 6-DOF grasping skill dispatcher.

Delegates to strategy-specific modules:
  grasp_top_down.py  - OBB-aligned top-down grasp
  grasp_horizontal.py - side/angled grasp for cylinders
  grasp_se3.py       - EquiGraspFlow 6-DOF grasp
  grasp_utils.py     - shared helpers (joint motion, orient, servo)
"""

import logging
import time

import numpy as np
from typing import Optional

from spark_real.skills.registry import spark_skill
from spark_real.skills.primitives import _result
from spark_real.skills.grasp_top_down import grasp_top_down as _grasp_top_down
from spark_real.skills.grasp_horizontal import grasp_horizontal as _grasp_horizontal
from spark_real.skills.grasp_se3 import grasp_se3_flow as _grasp_se3_flow

logger = logging.getLogger(__name__)


@spark_skill(
    name="grasp_se3",
    description=(
        "SE(3) 6-DOF grasp using EquiGraspFlow, for objects that need "
        "non-top-down grasps (plates, tools, oddly shaped objects)"
    ),
    params={
        "keypoint_label": str,
        "n_candidates": int,
        "prefer_side": bool,
        "prefer_angled": bool,
        "grip_end": str,
        "pre_grasp_distance": float,
        "force": float,
        "target_width": float,
        "strategy": str,
        "top_down": bool,
        "object_height_m": float,
        "approach_dir_x": float,
        "approach_dir_y": float,
        "approach_pitch_rad": float,
        "pitch_rad": float,
        "angle": float,
        "grasp_height_fraction": float,
        "offset_x": float,
        "offset_y": float,
        "offset_z": float,
    },
)
def grasp_se3(executor, params: dict):
    """
    Perform an SE(3) 6-DOF grasp on a detected object.

    Dispatches to the appropriate strategy:
      - "top_down": OBB-aligned vertical grasp (silverware, pens)
      - "horizontal"/"angled": side grasp for cylinders (bottles, cups)
      - "se3" (default): EquiGraspFlow candidate generation
    """
    t0 = time.time()

    # Wait for any background redetection to finish
    _redetect_thread = getattr(executor, "_redetect_thread", None)
    if _redetect_thread is not None and _redetect_thread.is_alive():
        _redetect_thread.join(timeout=5.0)
        executor._redetect_thread = None

    label = params.get("keypoint_label", "")
    n_candidates = max(params.get("n_candidates", 50), 50)
    prefer_side = params.get("prefer_side", False)
    prefer_angled = params.get("prefer_angled", False)
    grip_end = params.get("grip_end", "center")
    pre_grasp_dist = params.get("pre_grasp_distance", 0.10)
    force = params.get("force", 60)
    target_width = params.get("target_width", None)
    max_angle = 70.0 if prefer_angled else 45.0

    strategy = params.get("strategy", None)
    if strategy is None:
        strategy = "top_down" if params.get("top_down", False) else "se3"

    # Paper-grammar `angle` slot: an optional approach-orientation hint (rad).
    # The horizontal/angled strategies feed it into their approach-pitch slot
    # (when no explicit pitch is given). top_down and se3 derive yaw from the
    # OBB / EquiGraspFlow candidates, so there `angle` is advisory only.
    angle_hint = params.get("angle", None)
    if angle_hint is not None:
        if strategy in ("horizontal", "angled"):
            if params.get("approach_pitch_rad") is None and (
                params.get("pitch_rad") is None
            ):
                params = dict(params)
                params["approach_pitch_rad"] = float(angle_hint)
                logger.info(
                    "grasp_se3 '%s': mapping angle=%.3f rad -> approach_pitch_rad",
                    label,
                    float(angle_hint),
                )
        else:
            logger.info(
                "grasp_se3 '%s': angle=%.3f rad accepted as advisory only "
                "(strategy '%s' derives orientation from OBB/EquiGraspFlow)",
                label,
                float(angle_hint),
                strategy,
            )

    logger.info("grasp_se3 '%s': looking up detection...", label)
    det = executor.detection_map.get(label)
    if det is None:
        return _result(
            "grasp_se3",
            False,
            f"Object '{label}' not found in detections",
            time.time() - t0,
        )

    # Hard guard: never grasp an object that already sits inside the tray /
    # container, on first-pass grasps and any closed-loop repass. The target
    # xy is read from the live detection_map here.
    try:
        # Imported here to break a circular import: execution_recovery
        # imports spark_real.skills.registry at module load.
        from spark_real.control.execution_recovery import grasp_target_in_container

        _in = grasp_target_in_container(executor, label)
    except Exception:
        _in = None
    if _in is not None:
        executor._placed_labels.add(label)
        logger.info(
            "grasp_se3 '%s': already in container '%s', skipping grasp", label, _in
        )
        return _result(
            "grasp_se3",
            True,
            f"'{label}' already in container '{_in}', skipping",
            time.time() - t0,
        )

    # Optional explicit grasp-point offset from the detection centroid, in
    # base frame meters. Lets a BT grasp a specific spot on a large object
    # (e.g. a brush handle's midpoint) while keeping the validated hover ->
    # OBB yaw -> gated descend -> squeeze pipeline. Applied to a copy so
    # the live detection_map keeps the true centroid.
    _off = np.array(
        [
            float(params.get("offset_x", 0.0) or 0.0),
            float(params.get("offset_y", 0.0) or 0.0),
            float(params.get("offset_z", 0.0) or 0.0),
        ]
    )
    if np.any(_off != 0.0):
        det = dict(det)
        _p = np.asarray(det.get("position_3d"), dtype=float) + _off
        det["position_3d"] = _p.tolist()
        logger.info(
            "grasp_se3 '%s': grasp offset (%+.3f,%+.3f,%+.3f) -> " "(%.3f,%.3f,%.3f)",
            label,
            *_off,
            *_p,
        )

    if strategy == "top_down":
        return _grasp_top_down(
            executor,
            label,
            det,
            force,
            target_width,
            t0,
            grip_mode=params.get("grip_mode", "force"),
        )

    if strategy in ("horizontal", "angled"):
        ax = params.get("approach_dir_x", None)
        ay = params.get("approach_dir_y", None)
        approach_dir_xy = None
        if ax is not None or ay is not None:
            approach_dir_xy = np.array(
                [
                    float(ax) if ax is not None else 0.0,
                    float(ay) if ay is not None else 0.0,
                ],
                dtype=float,
            )
        object_height_m = float(params.get("object_height_m", 0.15))
        _default_pitch = (np.pi / 4.0) if strategy == "angled" else 0.0
        _pitch_val = params.get(
            "approach_pitch_rad", params.get("pitch_rad", _default_pitch)
        )
        _gz = params.get("grasp_z_m", None)
        _ghf = params.get("grasp_height_fraction", 0.5)
        return _grasp_horizontal(
            executor,
            label,
            det,
            force,
            target_width,
            object_height_m=object_height_m,
            pre_grasp_distance=float(pre_grasp_dist),
            approach_dir_xy=approach_dir_xy,
            approach_pitch_rad=float(_pitch_val),
            t0=t0,
            grasp_z_m=float(_gz) if _gz is not None else None,
            grasp_height_fraction=float(_ghf) if _ghf is not None else 0.5,
        )

    # Default: full SE(3) EquiGraspFlow path
    return _grasp_se3_flow(
        executor,
        label,
        det,
        force,
        target_width,
        n_candidates=n_candidates,
        prefer_side=prefer_side,
        prefer_angled=prefer_angled,
        grip_end=grip_end,
        pre_grasp_dist=float(pre_grasp_dist),
        max_angle=max_angle,
        t0=t0,
    )

"""Contact-GraspNet-backed SE(3) grasp skill for the UR10e.

``grasp_cgn(keypoint_label="red block")`` runs the full CaP-X-style pipeline:

    capture + SAM3 detect  ->  full-scene world cloud + segment map
        ->  Contact-GraspNet 6-DoF grasps
        ->  top-down selection (or top-down override for flat objects)
        ->  open jaws -> pregrasp -> descend -> force-verify clamp -> lift

Everything heavy (torch / CGN) is import-guarded and only touched at call time,
so importing this module never pulls in the model. The skill is GATED: it is a
no-op unless ``SPARK_GRASP_CGN=1`` AND the robot family is ur10e, so it can ship
disabled and be turned on for a guarded live test.

Selection + generation live in ``perception.cgn_grasp_select``; the cloud
builder lives in ``perception.scene_cloud``. This file is only the executor
glue (motion + clamp), reusing the executor's force-verify grasp path
(``_grasp_v2``) so grasp verification/floor-clamping behavior matches the
regular ``grasp`` primitive.
"""

import logging
import time

import numpy as np

from spark_real.control.grasp_strategy import cgn_gate_open, resolve_strategy
from spark_real.skills.registry import spark_skill
from spark_real.skills.primitives import _result

logger = logging.getLogger(__name__)




@spark_skill(
    name="grasp_cgn",
    description=(
        "Contact-GraspNet SE(3) grasp: detect the keypoint object, generate "
        "6-DoF grasps on the full-scene cloud, select the best top-down grasp "
        "(or a top-down override for flat objects), then approach + clamp + "
        "lift. UR10e only; gated by SPARK_GRASP_CGN=1."
    ),
    params={
        "keypoint_label": str,
        "force": float,
        "width": float,
        "pregrasp_backoff": float,
        "lift": float,
    },
)
def grasp_cgn(executor, params: dict):
    """Generate + execute a Contact-GraspNet grasp for a labeled object."""
    t0 = time.time()
    label = params.get("keypoint_label", "")
    force = float(params.get("force", 60))
    width_override = params.get("width", None)
    backoff = float(params.get("pregrasp_backoff", 0.10))
    lift = float(params.get("lift", 0.15))

    gate_ok, gate_why = cgn_gate_open(executor)
    if not gate_ok:
        # Loud: a disabled backend must never look like a quiet skip.
        logger.warning("[grasp_cgn] refusing to run: %s", gate_why)
        return _result(
            "grasp_cgn", False, f"grasp_cgn unavailable: {gate_why}", time.time() - t0
        )
    if not label:
        return _result("grasp_cgn", False, "keypoint_label required", time.time() - t0)
    pipeline = getattr(executor, "_pipeline", None)
    if pipeline is None:
        return _result(
            "grasp_cgn", False, "no pipeline on executor", time.time() - t0
        )

    try:
        from spark_real.perception.scene_cloud import (
            build_scene_cloud,
            resolve_segment_id,
        )
        from spark_real.perception.cgn_grasp_select import generate_and_select
    except Exception as exc:  # noqa: BLE001
        return _result(
            "grasp_cgn", False, f"CGN modules unavailable: {exc}", time.time() - t0
        )

    try:
        cloud = build_scene_cloud(pipeline, prompts=[label], multi_instance=True)
    except Exception as exc:  # noqa: BLE001
        return _result(
            "grasp_cgn", False, f"scene cloud build failed: {exc}", time.time() - t0
        )
    seg_id = resolve_segment_id(cloud["objects"], label)
    if seg_id is None:
        return _result(
            "grasp_cgn",
            False,
            f"'{label}' not detected (objects={[o['label'] for o in cloud['objects']]})",
            time.time() - t0,
        )

    # OBB yaw for the override case. Same resolver as the ordinary grasp path,
    # so a low-confidence blob does not yaw the override either.
    obb_yaw = None
    for det in cloud.get("detections", []):
        if det.label == cloud["objects"][seg_id - 1]["label"]:
            strat, why = resolve_strategy({"grasp_strategy": "auto"}, det, executor)
            if strat == "obb":
                obb_yaw = float(getattr(det, "orientation_angle", 0.0))
            logger.info("[grasp_cgn] override yaw source: %s (%s)", strat, why)
            break

    try:
        sel = generate_and_select(
            cloud["points"],
            cloud["segment_labels"],
            target_seg_id=seg_id,
            grasp_orientation=executor.GRASP_ORIENTATION,
            obb_yaw_rad=obb_yaw,
        )
    except Exception as exc:  # noqa: BLE001
        return _result(
            "grasp_cgn", False, f"CGN inference failed: {exc}", time.time() - t0
        )
    if sel is None:
        return _result(
            "grasp_cgn",
            False,
            f"no grasps produced for '{label}' (seg {seg_id})",
            time.time() - t0,
        )

    grasp_xyz = np.asarray(sel["xyz"], dtype=float)
    orient = list(sel["orient_rotvec"])
    # Never command below the table floor.
    grasp_xyz[2] = max(float(grasp_xyz[2]), executor.TABLE_Z_FLOOR + 0.002)
    logger.info(
        "[grasp_cgn] '%s' seg=%d mode=%s score=%.3f align=%.3f width=%.3f "
        "xyz=(%.3f,%.3f,%.3f) tcp_off=%.3f",
        label, seg_id, sel["mode"], sel["score"], sel["alignment"],
        sel["width"], grasp_xyz[0], grasp_xyz[1], grasp_xyz[2], sel["tcp_offset_m"],
    )

    # Reuse the executor's force-verify grasp path (_grasp_v2): it descends
    # GRASP_DEPTH_M with a contact guard, clamps, verifies via TCP force + a
    # confirming lift, and floor-clamps; identical semantics to `grasp`.
    try:
        executor.robot.open_gripper()
        executor._abort_sleep(0.3)

        # Pregrasp: back off straight up along +Z from the grasp point.
        pregrasp = grasp_xyz.copy()
        pregrasp[2] = grasp_xyz[2] + backoff
        executor._move_to(pregrasp, orient, velocity=executor.velocity * 0.6)

        # Descend to GRASP_DEPTH_M above the grasp point; _grasp_v2 then
        # descends the final GRASP_DEPTH_M and clamps at the grasp point.
        pre_clamp = grasp_xyz.copy()
        pre_clamp[2] = grasp_xyz[2] + executor.GRASP_DEPTH_M
        executor._move_to(pre_clamp, orient, velocity=executor.velocity * 0.4)

        # Hand the CGN orientation + label to the clamp path.
        executor._active_grasp_orient = list(orient)
        executor._last_keypoint_label = label
        # Use the CGN-predicted width if valid so the width-targeted squeeze +
        # verify engages; otherwise force-only close.
        w = width_override if width_override is not None else sel.get("width")
        if w is not None and np.isfinite(w) and 0.004 <= float(w) <= 0.080:
            executor._last_grasp_target_width = float(w)
        else:
            executor._last_grasp_target_width = None

        clamp_res = executor._grasp_v2({"force": force}, t0)
    except Exception as exc:  # noqa: BLE001
        return _result(
            "grasp_cgn", False, f"grasp motion failed: {exc}", time.time() - t0
        )

    if not clamp_res.success:
        return _result(
            "grasp_cgn",
            False,
            f"clamp failed ({sel['mode']}): {clamp_res.message}",
            time.time() - t0,
        )

    # Lift for transport clearance.
    executor._holding = True
    try:
        cur = executor._get_current_position()
        up = cur.copy()
        up[2] += lift
        executor._move_to(up, orient, velocity=executor.velocity * 0.4)
    except Exception as exc:  # noqa: BLE001
        logger.warning("[grasp_cgn] post-grasp lift failed: %s", exc)

    return _result(
        "grasp_cgn",
        True,
        f"CGN grasp ok (mode={sel['mode']}, score={sel['score']:.2f}, "
        f"align={sel['alignment']:.2f}, seg={seg_id})",
        time.time() - t0,
    )

"""
Recovery and re-detection logic for score execution.

All functions take the executor instance as their first argument. Success
verdicts live in control.success_verifier.
"""

import os
import time
import logging
import concurrent.futures as _cf
import numpy as np
from typing import Optional

from spark_real.skills import registry as skill_registry
from spark_real.control.executor_types import ExecutionResult
from spark_real.control.primitive_trace import (
    PrimitiveTrace,
    TraceWriter,
    detection_summary,
)
from spark_real.perception.sticky_binding import (
    STICKY_AMBIGUOUS,
    STICKY_ASSOCIATED,
    STICKY_NO_CANDIDATES,
    STICKY_OUT_OF_GATE,
    _labels_match,
    dedup_candidate_indices,
    is_self_occluded,
    sticky_associate,
)
from spark_real.utils.det_fields import det_field

logger = logging.getLogger(__name__)


# --- Sticky (gated, held-suppressed) re-detect adoption ---------------------
#
# A fresh frame never rewrites detection_map directly:
#   * a HELD label is skipped entirely (birdview cannot relocate an object
#     inside the Robotiq jaws; the mask is the object-in-hand or a phantom);
#   * candidates are deduped (SAM3 multi-instance emits offset twins);
#   * the nearest candidate within the gate wins; two in-gate candidates
#     closer than ambiguity_sep to each other are undecidable -> hold;
#   * out-of-gate is never adopted (keep-old-position fallback);
#   * displacement is measured same-source (position_agentview) and applied
#     as a delta to the fused position, so the sideview Z-override survives
#     adoption.
#
# Config: perception.sticky in the family YAML (enabled defaults False).
# The 3 cm ambiguity_sep and 12 cm gate are sim-tuned and must be re-measured
# on the real table. SPARK_STICKY=0/1 overrides the enable bit.

_STICKY_DEFAULTS = {
    "enabled": False,
    "gate_m": 0.12,
    "ambiguity_sep_m": 0.03,
    "dedup_radius_m": 0.02,
}


def sticky_settings(executor) -> dict:
    """The perception.sticky block with defaults; env SPARK_STICKY overrides
    just the enable bit."""
    out = dict(_STICKY_DEFAULTS)
    try:
        profile = getattr(getattr(executor, "_pipeline", None), "profile", None)
        raw = getattr(profile, "raw", None) or {}
        block = (raw.get("perception") or {}).get("sticky") or {}
        if isinstance(block, dict):
            for k in out:
                if block.get(k) is not None:
                    out[k] = block[k]
    except Exception:  # noqa: BLE001 - config read must never break recovery
        pass
    env = os.environ.get("SPARK_STICKY")
    if env is not None:
        out["enabled"] = env.strip() not in ("", "0", "false", "no")
    out["enabled"] = bool(out["enabled"])
    for k in ("gate_m", "ambiguity_sep_m", "dedup_radius_m"):
        out[k] = float(out[k])
    return out


def _measure_pos(det):
    """Same-source measurement position: the primary-camera backprojection
    when stamped, else the fused position."""
    pos = det_field(det, "position_agentview")
    if pos is None:
        pos = det_field(det, "position_3d")
    return None if pos is None else np.asarray(pos, dtype=float).reshape(-1)[:3]


def _held_label_of(executor) -> str:
    if not getattr(executor, "_holding", False):
        return ""
    return getattr(executor, "_last_pick_label", "") or getattr(
        executor, "_active_grasp_label", ""
    ) or ""


def sticky_adopt(executor, label, candidates, *, context="") -> bool:
    """Gate-checked adoption of fresh ``candidates`` for ``label`` into
    executor.detection_map. Returns True iff the map entry was updated.

    Every non-adopt branch leaves detection_map UNTOUCHED (the existing
    keep-old-position behavior) and says why.
    """
    cfg = sticky_settings(executor)
    det_map = getattr(executor, "detection_map", None)
    if det_map is None:
        return False

    held = _held_label_of(executor)
    if held and _labels_match(label, held):
        logger.info(
            "Sticky[%s]: '%s' is HELD; camera evidence cannot relocate an "
            "object inside the jaws -- binding kept", context, label
        )
        return False

    entry = det_map.get(label) or {}
    anchor = _measure_pos(entry)
    fused_old = det_field(entry, "position_3d")
    fused_old = (
        None if fused_old is None
        else np.asarray(fused_old, dtype=float).reshape(-1)[:3]
    )

    positions = [_measure_pos(c) for c in candidates]
    confs = [float(det_field(c, "confidence", 0.0) or 0.0) for c in candidates]
    kept = dedup_candidate_indices(
        positions, confs, merge_radius_m=cfg["dedup_radius_m"]
    )
    cands = [candidates[i] for i in kept]
    cand_pos = [positions[i] for i in kept]

    if anchor is None:
        # No prior position: nothing to gate against; adopt the highest-
        # confidence deduped candidate (first bind).
        if not cands:
            logger.info("Sticky[%s]: '%s' no candidates; binding kept",
                        context, label)
            return False
        best_i = int(np.argmax([float(det_field(c, "confidence", 0.0) or 0.0)
                                for c in cands]))
        _write_adopted(executor, label, cands[best_i], cand_pos[best_i],
                       cand_pos[best_i], context, note="first-bind")
        return True

    res = sticky_associate(
        anchor, cand_pos,
        gate_m=cfg["gate_m"], ambiguity_sep_m=cfg["ambiguity_sep_m"],
    )
    if res.status == STICKY_ASSOCIATED:
        c = cands[res.chosen_index]
        fresh = cand_pos[res.chosen_index]
        base = fused_old if fused_old is not None else anchor
        new_pos = base + (fresh - anchor)  # delta application, same-source
        _write_adopted(executor, label, c, new_pos, fresh, context,
                       note=f"d={res.chosen_dist_m * 100:.1f}cm")
        return True
    if res.status == STICKY_AMBIGUOUS:
        logger.info(
            "Sticky[%s]: '%s' AMBIGUOUS (two in-gate candidates within "
            "%.0fmm of each other); binding kept",
            context, label, cfg["ambiguity_sep_m"] * 1000,
        )
        return False
    if res.status == STICKY_OUT_OF_GATE:
        # Self-occlusion hold: the EE hovering over the last bound position
        # blinds birdview, so an out-of-gate re-association at that moment is
        # occlusion, not a scene change. Radii are sim-tuned pending rig
        # re-measurement.
        ee = None
        try:
            ee = executor._get_current_position()
        except Exception:  # noqa: BLE001
            pass
        if ee is not None and is_self_occluded(anchor, ee):
            logger.info(
                "Sticky[%s]: '%s' out-of-gate while EE hovers over it "
                "(self-occlusion); binding kept", context, label
            )
        else:
            logger.info(
                "Sticky[%s]: '%s' nearest candidate %.1fcm away "
                "(> gate %.0fcm); NOT adopted, binding kept",
                context, label,
                (res.chosen_dist_m or 0.0) * 100, cfg["gate_m"] * 100,
            )
        return False
    logger.info("Sticky[%s]: '%s' %s; binding kept",
                context, label,
                "no candidates" if res.status == STICKY_NO_CANDIDATES
                else res.status)
    return False


def _write_adopted(executor, label, det, fused_pos, measured_pos, context,
                   note=""):
    old = executor.detection_map.get(label, {}) or {}
    executor.detection_map[label] = {
        "position_3d": [float(v) for v in np.asarray(fused_pos).reshape(-1)[:3]],
        "position_agentview": [
            float(v) for v in np.asarray(measured_pos).reshape(-1)[:3]
        ],
        "orientation_angle": det_field(
            det, "orientation_angle", det_field(old, "orientation_angle", 0.0)
        ),
        "aspect_ratio": det_field(
            det, "aspect_ratio", det_field(old, "aspect_ratio", 1.0)
        ),
    }
    logger.info(
        "Sticky[%s]: '%s' adopted -> [%.3f,%.3f,%.3f] %s",
        context, label, *executor.detection_map[label]["position_3d"], note,
    )


# container_region / xy_inside_container / TRAY_FALLBACK_HALF_EXTENT_M live in
# the leaf control.container_geometry (numpy-only) so skills can use them
# without a circular import through skills.registry. Re-exported here.
from spark_real.control.container_geometry import (  # noqa: E402,F401
    TRAY_FALLBACK_HALF_EXTENT_M,
    container_region,
    xy_inside_container,
)


# A destination only acts as a "container" for the in-tray grasp guard if it is
# plausibly a receptacle: detected slots, or a label that reads like one.
# Otherwise the fallback radius (TRAY_FALLBACK_HALF_EXTENT_M = 10cm) makes a
# bare stack target look like a bin and the grasp is falsely skipped. A block
# is placed ON, not IN.
_CONTAINER_LABEL_HINTS = (
    "tray", "bin", "bowl", "dustpan", "pan", "box", "cup", "container",
    "basket", "mug", "pot", "plate", "drawer", "tin", "crate", "caddy",
    "holder", "sink", "bucket", "cart", "jar", "can",
)


def _is_container_like(det, label: str) -> bool:
    """True only for genuine receptacles (detected slots or a bin-like label)."""
    if det is not None:
        get = det.get if hasattr(det, "get") else (lambda k, d=None: getattr(det, k, d))
        if get("slots", None):
            return True
    lab = (label or "").lower()
    return any(h in lab for h in _CONTAINER_LABEL_HINTS)


def grasp_target_in_container(executor, label: str) -> Optional[str]:
    """
    If the grasp target ``label`` currently sits inside any known
    destination/container region, return that container's label; else None.

    Containers come from the executor's ``_destination_labels`` (derived
    from the plan's place targets in pipeline_execution). The target's xy
    is read from the live detection_map at call time so re-detection moves
    are honored. Only genuine receptacles count (see ``_is_container_like``);
    a bare stack target (another block) is placed ON, not IN, so it never
    triggers the guard.
    """
    if not label or executor is None:
        return None
    det_map = getattr(executor, "detection_map", None) or {}
    dest_labels = getattr(executor, "_destination_labels", None) or set()
    if not dest_labels:
        return None
    target_det = det_map.get(label)
    if target_det is None:
        return None
    get = (
        target_det.get
        if hasattr(target_det, "get")
        else (lambda k, d=None: getattr(target_det, k, d))
    )
    pos = get("position_3d", None)
    if pos is None:
        return None
    try:
        target_xy = np.asarray(pos[:2], dtype=float)
    except Exception:
        return None
    for dest_label in dest_labels:
        if dest_label == label:
            continue
        dest_det = det_map.get(dest_label)
        if not _is_container_like(dest_det, dest_label):
            continue
        if xy_inside_container(target_xy, dest_det):
            return dest_label
    return None


def find_preceding_label(actions: list, index: int) -> Optional[str]:
    for j in range(index - 1, -1, -1):
        if actions[j].get("type") == "move_to_keypoint":
            return actions[j].get("params", {}).get("keypoint_label", "")
    return None


def attribution_enabled(executor) -> bool:
    """verification.attribution in the family YAML (default OFF = current
    dispatch). SPARK_ATTRIBUTION env overrides for a quick A/B."""
    env = os.environ.get("SPARK_ATTRIBUTION")
    if env is not None:
        return env.strip() not in ("", "0", "false", "no")
    try:
        profile = getattr(getattr(executor, "_pipeline", None), "profile", None)
        raw = getattr(profile, "raw", None) or {}
        block = raw.get("verification") or {}
        return bool(block.get("attribution", False))
    except Exception:  # noqa: BLE001
        return False


def build_verify_result(executor, label, failed_result=None) -> dict:
    """The attribute_failure input, from live executor state: typed
    GraspOutcome + detection_map presence/confidence for ``label``."""
    det = executor.detection_map.get(label) if label else None
    v = getattr(executor, "_last_grasp_verdict", None)
    outcome = (
        v.get("outcome") if isinstance(v, dict)
        else getattr(v, "outcome", None)
    )
    missing = bool(label) and (
        det is None or det_field(det, "position_3d") is None
    )
    out = {
        "semantic_failure": False,
        "detection_missing": missing,
        "grasp_outcome": outcome,
    }
    conf = det_field(det, "confidence")
    if conf is not None:
        out["detection_confidence"] = float(conf)
    return out


def attempt_recovery(
    executor, action_type, params, failed_result, actions, action_index
):
    """
    Dispatch to the right recovery strategy based on failure type.

    With verification.attribution enabled, the lowest-responsible-layer
    table (control.attribution) picks the tier: PERCEPTION -> re-detect and
    re-bind (skip the tier-1 in-place perturb); EXECUTION -> the local
    ladder; PLAN -> stop and surface to the operator, never an autonomous
    LLM replan.
    """
    layer = None
    if attribution_enabled(executor):
        from spark_real.control.attribution import Layer, attribute_failure

        label = (params or {}).get("keypoint_label", "") or find_preceding_label(
            actions, action_index
        )
        _enabled, _retry, max_attempts = (True, False, 1)
        try:
            _enabled, _retry, max_attempts = executor._verify_settings()
        except Exception:  # noqa: BLE001
            pass
        # Local retries already spent on this label (executor_core's
        # per-label grasp retry counter).
        retry_count = 0
        try:
            retry_count = int(
                (getattr(executor, "_grasp_retry_count", {}) or {}).get(
                    label, 0
                )
            )
        except Exception:  # noqa: BLE001
            pass
        layer = attribute_failure(
            build_verify_result(executor, label, failed_result),
            None,  # scene_diff: wired in once the pre-primitive diff lands
            retry_count,
            max_local_retries=max(1, max_attempts),
        )
        logger.info(
            "Attribution: %s failure on %s attributed to %s",
            action_type, label or "?", layer.value,
        )
        if layer == Layer.PLAN:
            logger.warning(
                "Attribution: PLAN layer -> stopping recovery and surfacing "
                "to the operator (no autonomous replan)"
            )
            return None

    if action_type == "grasp":
        skip_tier1 = False
        if layer is not None:
            from spark_real.control.attribution import Layer

            skip_tier1 = layer == Layer.PERCEPTION
        return recover_grasp(
            executor, params, actions, action_index, skip_tier1=skip_tier1
        )
    if action_type == "move_to_keypoint":
        label = params.get("keypoint_label", "")
        if "not found" in failed_result.message:
            return recover_object_not_found(executor, label)
        return recover_approach_failed(executor, label)
    return None


# Recover from a failed grasp with progressive Z bias
def recover_grasp(executor, grasp_params, actions, action_index, skip_tier1=False):
    target_label = find_preceding_label(actions, action_index)
    force = grasp_params.get("force", 50)
    logger.info(
        "Grasp recovery for '%s': up to %d retries",
        target_label or "unknown",
        executor.MAX_GRASP_RETRIES,
    )

    # Config table_height overrides only when explicitly set; None falls
    # back to the executor's per-family TABLE_Z_FLOOR (config-fed).
    table_z = executor.TABLE_Z_FLOOR
    if executor._pipeline is not None:
        _th = getattr(executor._pipeline.config, "table_height", None)
        if _th is not None:
            table_z = _th

    failed_z = executor._get_current_position()[2]
    recovery_z_floor = -0.27

    # Tier-1: in-place reseat before the expensive perception re-grounding. A
    # barely-missed grasp is often fixed by a small nudge (down / lateral) and
    # a re-close, with no re-detect and no LLM call. Only if these cheap
    # perturbations fail does recovery drop to tier-2 (re-ground).
    # skip_tier1: attribution said PERCEPTION, so the detection itself is the
    # suspect and an in-place perturb around a wrong position is wasted motion.
    if skip_tier1:
        logger.info(
            "Grasp recovery: tier-1 perturb skipped (attributed to PERCEPTION); "
            "going straight to re-grounding"
        )
    else:
        try:
            from spark_real.skills.recovery import grasp_perturb as _grasp_perturb
            for _t1 in range(2):
                if executor._abort:
                    break
                r1 = _grasp_perturb(executor, {"attempt": _t1, "force": force})
                if getattr(r1, "success", False):
                    logger.info(
                        "Grasp recovery: tier-1 perturb reseated the grasp "
                        "(attempt %d)", _t1,
                    )
                    return r1
            logger.info(
                "Grasp recovery: tier-1 perturb did not reseat; "
                "falling back to perception re-grounding"
            )
        except Exception as exc:
            logger.warning("Grasp recovery: tier-1 perturb skipped (%s)", exc)

    for attempt in range(1, executor.MAX_GRASP_RETRIES + 1):
        z_bias = executor.RECOVERY_Z_BIAS[
            min(attempt - 1, len(executor.RECOVERY_Z_BIAS) - 1)
        ]
        logger.info(
            "Recovery attempt %d/%d (Z bias: -%.1fmm)",
            attempt,
            executor.MAX_GRASP_RETRIES,
            z_bias * 1000,
        )

        executor.robot.open_gripper()
        time.sleep(0.5)

        # Lift before re-perceiving: the failed descent may have nudged
        # the object, so always clear+redetect rather than re-aim at a
        # stale XY.
        current = executor._get_current_position()
        clear_pos = current.copy()
        clear_pos[2] = max(current[2] + 0.05, failed_z + 0.03)
        executor._move_to(clear_pos, executor.GRASP_ORIENTATION)

        new_target = _redetect_for_recovery(executor, target_label)
        # Guard: a recovery re-detect that lands inside the tray / container
        # means the object is already placed (or got knocked in). Never
        # re-grasp from inside the container; report success-with-note.
        if new_target is not None and target_label:
            existing = executor.detection_map.get(target_label, {}) or {}
            probe = dict(existing)
            probe["position_3d"] = new_target.tolist()
            executor.detection_map[target_label] = probe
            _in = grasp_target_in_container(executor, target_label)
            if _in is not None:
                executor._placed_labels.add(target_label)
                executor._holding = False
                msg = (
                    f"'{target_label}' already in container '{_in}', "
                    f"skipping recovery grasp"
                )
                logger.info(msg)
                return ExecutionResult(action_type="grasp", success=True, message=msg)
        det = executor.detection_map.get(target_label)
        if new_target is None:
            if det is None:
                continue
            new_target = np.array(det["position_3d"])
            logger.info(
                "Attempt %d: redetect missed, falling back to " "stale XY=[%.3f,%.3f]",
                attempt,
                new_target[0],
                new_target[1],
            )
        else:
            logger.info(
                "Attempt %d: redetected at [%.3f,%.3f]",
                attempt,
                new_target[0],
                new_target[1],
            )

        new_target[2] = max(min(new_target[2], failed_z) - z_bias, recovery_z_floor)
        logger.info("Attempt %d: approaching [%.3f,%.3f,%.3f]", attempt, *new_target)
        existing = executor.detection_map.get(target_label, {}) or {}
        executor.detection_map[target_label] = {
            "position_3d": new_target.tolist(),
            "orientation_angle": existing.get("orientation_angle", 0.0),
            "aspect_ratio": existing.get("aspect_ratio", 1.0),
        }
        executor._approach_target(
            new_target, detection=executor.detection_map[target_label]
        )

        # Compliant descent
        if skill_registry.get("compliant_push") is not None:
            skill_registry.dispatch(
                "compliant_push",
                executor,
                {
                    "direction": [0, 0, -1],
                    "max_distance": 0.03,
                    "force_threshold": 10.0,
                    "step_size": 0.002,
                },
            )

        # Re-grasp
        try:
            if hasattr(executor.robot, "get_tcp_force"):
                ft = executor.robot.get_tcp_force()
                executor._pre_grasp_force_magnitude = np.linalg.norm(ft[:3])
        except Exception:
            executor._pre_grasp_force_magnitude = 0.0

        if hasattr(executor.robot, "_send_gripper_command"):
            executor.robot._send_gripper_command(1.0, speed=80, force=min(force, 100))
            time.sleep(1.5)
            executor.robot._send_gripper_command(1.0, speed=50, force=100)
            time.sleep(1.0)
        else:
            executor.robot.close_gripper()
            time.sleep(1.5)

        failed_z = min(failed_z, executor._get_current_position()[2])

        if executor._verify_grasp():
            executor._holding = True
            # The recovery re-grasp closed at this height (progressive
            # RECOVERY_Z_BIAS included); record it so the release give-back
            # accounts for the extra descent.
            try:
                executor._actual_grasp_tcp_z = float(
                    executor._get_current_position()[2]
                )
            except Exception:  # noqa: BLE001
                pass
            msg = f"Grasp recovery succeeded on attempt {attempt}"
            logger.info(msg)
            return ExecutionResult(action_type="grasp", success=True, message=msg)

        logger.info("Recovery attempt %d: verification failed", attempt)

    logger.warning(
        "Grasp recovery failed after %d attempts", executor.MAX_GRASP_RETRIES
    )
    return None


def _redetect_for_recovery(executor, label):
    """
    Re-detect during recovery; returns new 3D position or None.

    Logs each substep so silent stalls in capture/detect are visible. Runs
    the SAM3 detect on a threadpool with a 30 s timeout; on timeout returns
    None and the caller falls back to the stale XY.
    """
    if executor._pipeline is None or not label:
        return None
    try:
        t0 = time.time()
        logger.info("Re-detect '%s': capture...", label)
        captures = executor._pipeline.capture()
        logger.info(
            "Re-detect '%s': capture done (%.2fs, %d cams) -> detect...",
            label,
            time.time() - t0,
            len(captures),
        )

        def _do_detect():
            return executor._pipeline.detect(captures, prompts=[label])

        with _cf.ThreadPoolExecutor(
            max_workers=1, thread_name_prefix="redetect"
        ) as pool:
            fut = pool.submit(_do_detect)
            try:
                new_dets = fut.result(timeout=30.0)
            except _cf.TimeoutError:
                logger.warning(
                    "Re-detect '%s': detect timed out after 30s; "
                    "aborting redetect (fallback to stale XY)",
                    label,
                )
                return None
        logger.info(
            "Re-detect '%s': detect done (%.2fs total, %d dets)",
            label,
            time.time() - t0,
            len(new_dets),
        )

        # Anchor to the detection nearest the label's prior known position, not
        # the first re-assigned instance label: a fresh multi-instance detect
        # can renumber "blue block 1" / "blue block 2", and recovery would
        # grasp the wrong one.
        prior = executor.detection_map.get(label, {}) or {}
        prior_pos = prior.get("position_3d") if hasattr(prior, "get") else None
        best, best_d = None, float("inf")
        for det in new_dets:
            det_label = det.label if hasattr(det, "label") else det.get("label", "")
            pos = (
                det.position_3d
                if hasattr(det, "position_3d")
                else det.get("position_3d")
            )
            if pos is None:
                continue
            if prior_pos is not None:
                d = float(
                    np.linalg.norm(
                        np.asarray(pos[:2], float) - np.asarray(prior_pos[:2], float)
                    )
                )
            else:
                # No prior anchor: fall back to exact-label match.
                d = 0.0 if det_label == label else float("inf")
            if d < best_d:
                best, best_d = np.array(pos), d
        # 0.15 m gate: a genuine re-detect of the same block barely moves; a jump
        # to the other block is ~0.3 m and must not be accepted as "the same one".
        if best is not None and best_d <= 0.15:
            logger.info(
                "Re-detected '%s' at [%.3f,%.3f,%.3f] (nearest prior, d=%.3fm)",
                label, *best, best_d,
            )
            return best
        logger.info(
            "Re-detect '%s': no detection within 0.15m of prior (best_d=%.3fm, %d dets)",
            label, best_d, len(new_dets),
        )
    except Exception as e:
        logger.warning("Re-detection failed: %s", e)
    return None


def recover_object_not_found(executor, label):
    if skill_registry.get("search_keypoint") is None:
        return None
    logger.info("Search recovery for '%s'", label)
    result = skill_registry.dispatch(
        "search_keypoint",
        executor,
        {"keypoint_label": label, "search_radius": 0.15, "num_loops": 2},
    )
    if not result.success:
        return None
    det = executor.detection_map.get(label)
    if det is None:
        return None
    executor._approach_target(np.array(det["position_3d"]), detection=det)
    return ExecutionResult(
        action_type="move_to_keypoint",
        success=True,
        message=f"Found '{label}' via search",
    )


def recover_approach_failed(executor, label):
    if skill_registry.get("retract_retry") is None:
        return None
    logger.info("Retract-retry recovery for '%s'", label)
    result = skill_registry.dispatch(
        "retract_retry",
        executor,
        {
            "keypoint_label": label,
            "retract_distance": 0.08,
            "offset_range": 0.015,
            "num_retries": 3,
        },
    )
    if result.success:
        return ExecutionResult(
            action_type="move_to_keypoint",
            success=True,
            message=f"Re-approached '{label}' via retract_retry",
        )
    return None


def redetect_all(executor, actions, current_index):
    """Re-detect the next pick label so the next fold/grasp uses a fresh pose.

    Handles two pick shapes:
      * Pick-place:  move_to_keypoint(X) -> grasp(...) ...
      * SE(3) fold:  grasp_se3(keypoint_label=X) directly (the skill does
        the approach internally, so there's no move_to_keypoint before it).
    The cloth-fold sequence shifts the cloth state on every release, so
    re-detecting before the next grasp_se3 is the high-value path for folds.
    """
    remaining = actions[current_index + 1 :]
    next_label = None
    for j, a in enumerate(remaining):
        atype = a.get("type")
        params = a.get("params", {})
        label = params.get("keypoint_label", "")
        if atype == "grasp_se3" and label and label not in executor._placed_labels:
            next_label = label
            break
        if atype == "move_to_keypoint":
            next_actions = remaining[j + 1 : j + 3]
            is_pick = any(na.get("type") == "grasp" for na in next_actions)
            if label and is_pick and label not in executor._placed_labels:
                next_label = label
                break

    if not next_label:
        return

    # Redetecting during inter-grasp transit lets birdview see the arm
    # mid-motion and SAM3 latch onto phantom masks. Two guards:
    #   1. wait until the arm is stationary and near home before capturing
    #   2. reject a new position that teleported > REDETECT_MAX_MOVE_M from
    #      the planner anchor (a real object move stays within a few cm).
    REDETECT_MAX_MOVE_M = 0.10
    try:
        # Best-effort stationary wait; skip silently if no such helper.
        if hasattr(executor, "_wait_stationary_after_servo"):
            executor._wait_stationary_after_servo()
        # Best-effort "arm near home?" gate so a mid-transit arm doesn't
        # occlude birdview; skipped if joints can't be read.
        try:
            home_q = getattr(executor.robot, "HOME_CONFIG", None)
            cur_q = executor.robot.get_joint_positions()
            if home_q is not None and cur_q is not None:
                home_a = np.asarray(home_q, dtype=float)[:7]
                cur_a = np.asarray(cur_q, dtype=float)[:7]
                d_home = float(np.max(np.abs(cur_a - home_a)))
                if d_home > 0.3:
                    logger.info(
                        "Re-detect deferred for '%s': arm not at home "
                        "(max |dq|=%.2f rad > 0.30); using stale "
                        "detection. Will retry from grasp_se3's own "
                        "pre-approach redetect when arm reaches hover.",
                        next_label,
                        d_home,
                    )
                    return
        except Exception:
            pass

    except Exception:
        pass

    logger.info("Re-detecting next pick target: %s", next_label)
    try:
        captures = executor._pipeline.capture()
        new_dets = executor._pipeline.detect(captures, prompts=[next_label])

        # Build destination-exclusion zones (e.g. the tray) so a detection
        # inside the destination (likely the object just placed) is not
        # retargeted as the next pickup. Mirrors pipeline.py's planning-time
        # destination filter, applied here at execution time.
        dest_zones = []
        for dest_label in getattr(executor, "_destination_labels", set()) or set():
            dest_det = executor.detection_map.get(dest_label)
            if not dest_det or not dest_det.get("position_3d"):
                continue
            dest_pos_xy = np.asarray(dest_det["position_3d"][:2], dtype=float)
            dest_minor = float(dest_det.get("obb_minor_m") or 0.0)
            dest_zones.append((dest_label, dest_pos_xy, max(0.12, dest_minor * 0.6)))

        cfg = sticky_settings(executor)
        if cfg["enabled"]:
            # Gated adoption path: keep the destination-zone exclusion (placed
            # objects must not become the next pickup), then dedup + held
            # suppression + ambiguity hold + never-adopt-out-of-gate.
            filtered = []
            for det in new_dets:
                lbl = det.label if hasattr(det, "label") else det.get("label", "")
                pos = det_field(det, "position_3d")
                if not lbl or pos is None:
                    continue
                cand_xy = np.asarray(pos[:2], dtype=float)
                if any(
                    np.linalg.norm(cand_xy - dest_xy) < dest_r
                    for _dl, dest_xy, dest_r in dest_zones
                ):
                    continue
                filtered.append(det)
            sticky_adopt(executor, next_label, filtered, context="post-release")
            return

        # Pick the candidate closest to the planner's anchor that is not
        # inside any destination zone; if none remain outside, keep the
        # existing map entry (the planner anchor is the best guess).
        old_pos = executor.detection_map.get(next_label, {}).get("position_3d")
        old_xy = np.asarray(old_pos[:2], dtype=float) if old_pos is not None else None
        best_det = None
        best_dist = float("inf")
        for det in new_dets:
            lbl = det.label if hasattr(det, "label") else det.get("label", "")
            pos = (
                det.position_3d
                if hasattr(det, "position_3d")
                else det.get("position_3d")
            )
            if not lbl or pos is None:
                continue
            cand_xy = np.asarray(pos[:2], dtype=float)
            # Skip if inside any destination zone (placed objects).
            in_dest = False
            for dest_label, dest_xy, dest_r in dest_zones:
                if np.linalg.norm(cand_xy - dest_xy) < dest_r:
                    logger.info(
                        "Re-detect dropped '%s' at [%.3f,%.3f] "
                        "(inside '%s' destination zone, likely a placed item)",
                        lbl,
                        pos[0],
                        pos[1],
                        dest_label,
                    )
                    in_dest = True
                    break
            if in_dest:
                continue
            # Prefer closest to planner's anchor; if no anchor,
            # accept first valid.
            if old_xy is not None:
                d = float(np.linalg.norm(cand_xy - old_xy))
                if d < best_dist:
                    best_dist = d
                    best_det = det
            elif best_det is None:
                best_det = det

        if best_det is None:
            logger.warning(
                "Re-detect for '%s': no candidate outside "
                "destination zones; keeping old position",
                next_label,
            )
            return
        pos = (
            best_det.position_3d
            if hasattr(best_det, "position_3d")
            else best_det.get("position_3d")
        )
        # Reject a re-detect that teleports the target: a real move between
        # trial start and inter-grasp transit is a few cm at most, so a jump
        # of hundreds of mm is a phantom/wrong-instance detection. Trust the
        # planner anchor and let grasp_se3's pre-approach redetect refine.
        if old_xy is not None and best_dist > REDETECT_MAX_MOVE_M:
            logger.warning(
                "Re-detect '%s' REJECTED: candidate at [%.3f,%.3f] is "
                "%.1fmm from planner anchor (>%.0fmm threshold). "
                "Likely a phantom / wrong-instance detection. Keeping "
                "old position.",
                next_label,
                pos[0],
                pos[1],
                best_dist * 1000,
                REDETECT_MAX_MOVE_M * 1000,
            )
            return
        orient = getattr(best_det, "orientation_angle", 0.0)
        ar = getattr(best_det, "aspect_ratio", 1.0)
        executor.detection_map[next_label] = {
            "position_3d": list(pos),
            "orientation_angle": orient,
            "aspect_ratio": ar,
        }
        logger.info(
            "Re-detect '%s' -> [%.3f,%.3f,%.3f]%s",
            next_label,
            pos[0],
            pos[1],
            pos[2],
            (
                f" (moved {best_dist * 1000:.1f}mm)"
                if old_xy is not None and best_dist != float("inf")
                else ""
            ),
        )
    except Exception as e:
        logger.warning("Re-detection failed: %s", e)


def rebind_settings(executor) -> dict:
    """perception.rebind block with defaults (enabled by default)."""
    out = {"enabled": True, "log_drift_cm": 2.0}
    try:
        profile = getattr(getattr(executor, "_pipeline", None), "profile", None)
        raw = getattr(profile, "raw", None) or {}
        block = (raw.get("perception") or {}).get("rebind") or {}
        if isinstance(block, dict):
            out.update({k: block[k] for k in out if k in block})
    except Exception:  # noqa: BLE001
        pass
    return out


def rebind_before_approach(executor, label):
    """Re-detect the bound label before every approach so a target moved
    between plan approval and the approach becomes a silent parameter update
    on the same tree: no re-plan, no LLM, no failed attempt first.

    Adoption is delegated to redetect_single (sticky gating handles held-object
    suppression, ambiguity hold, never-adopt-out-of-gate). This wrapper only
    decides whether to run it and reports the drift it corrected.
    """
    cfg = rebind_settings(executor)
    if not cfg["enabled"] or not label:
        return
    old = None
    try:
        d = (executor.detection_map or {}).get(label) or {}
        p = d.get("position_3d")
        if p is not None:
            old = np.array(p, dtype=float)
    except Exception:  # noqa: BLE001
        pass
    try:
        redetect_single(executor, label)
    except Exception as exc:  # noqa: BLE001 - a failed rebind keeps the old binding
        logger.warning("[rebind] '%s' re-detect failed (%s); keeping the "
                       "approved binding", label, exc)
        return
    try:
        d = (executor.detection_map or {}).get(label) or {}
        p = d.get("position_3d")
        if old is not None and p is not None:
            drift_cm = float(np.linalg.norm(np.array(p)[:2] - old[:2])) * 100.0
            if drift_cm >= float(cfg["log_drift_cm"]):
                logger.info(
                    "[rebind] '%s' moved %.1f cm since approval; approach "
                    "retargeted in place (no replan)", label, drift_cm,
                )
    except Exception:  # noqa: BLE001
        pass


def redetect_single(executor, label):
    logger.info("Pre-approach re-detection for '%s'", label)
    cfg = sticky_settings(executor)
    if cfg["enabled"]:
        held = _held_label_of(executor)
        if held and _labels_match(label, held):
            logger.info(
                "Sticky[pre-approach]: '%s' is HELD; skipping re-detect "
                "entirely", label
            )
            return
    try:
        base_prompt = label.rsplit(" ", 1)[0] if label[-1].isdigit() else label
        captures = executor._pipeline.capture()
        new_dets = executor._pipeline.detect(captures, prompts=[base_prompt])
        merged = executor._pipeline.merge_detections(new_dets)

        if cfg["enabled"]:
            # Gated adoption: dedup + held suppression + ambiguity hold +
            # never-adopt-out-of-gate (the legacy path below adopts the
            # nearest candidate unconditionally at any distance).
            sticky_adopt(executor, label, merged, context="pre-approach")
            return

        old_pos = executor.detection_map.get(label, {}).get("position_3d")
        best_det = None
        best_dist = float("inf")

        for det in merged:
            pos = (
                det.position_3d
                if hasattr(det, "position_3d")
                else det.get("position_3d")
            )
            if pos is None:
                continue
            if old_pos is not None:
                dist = np.linalg.norm(np.array(pos)[:2] - np.array(old_pos)[:2])
                if dist < best_dist:
                    best_dist = dist
                    best_det = det
            elif best_det is None:
                best_det = det

        if best_det is not None:
            pos = (
                best_det.position_3d
                if hasattr(best_det, "position_3d")
                else best_det.get("position_3d")
            )
            orient = getattr(best_det, "orientation_angle", 0.0)
            ar = getattr(best_det, "aspect_ratio", 1.0)
            executor.detection_map[label] = {
                "position_3d": list(pos),
                "orientation_angle": orient,
                "aspect_ratio": ar,
            }
            if old_pos is not None:
                logger.info(
                    "'%s' updated: [%.3f,%.3f,%.3f] " "(moved %.1fmm, %d candidates)",
                    label,
                    *executor.detection_map[label]["position_3d"],
                    best_dist * 1000,
                    len(merged),
                )
        else:
            logger.warning("'%s' not found, keeping old position", label)
    except Exception as e:
        logger.warning("Re-detection for '%s' failed: %s", label, e)


def release_verdict_enabled(executor) -> bool:
    """verification.release_verdict in the family YAML (default OFF = the
    trace-only behavior). SPARK_RELEASE_VERDICT env overrides."""
    env = os.environ.get("SPARK_RELEASE_VERDICT")
    if env is not None:
        return env.strip() not in ("", "0", "false", "no")
    try:
        profile = getattr(getattr(executor, "_pipeline", None), "profile", None)
        raw = getattr(profile, "raw", None) or {}
        block = raw.get("verification") or {}
        return bool(block.get("release_verdict", False))
    except Exception:  # noqa: BLE001
        return False


def _release_anchor_xy(executor):
    """The last commanded release point's XY: the ReleaseWitness TCP if one
    was recorded, else the current TCP (the arm is still at the release pose
    when verify_placement runs)."""
    witness = getattr(executor, "_release_witness", None)
    tcp = getattr(witness, "tcp_xyz", None) if witness is not None else None
    if tcp is not None:
        return np.asarray(tcp[:2], dtype=float)
    try:
        return np.asarray(executor._get_current_position()[:2], dtype=float)
    except Exception:  # noqa: BLE001
        return None


def _release_verdict(executor, pick_label, dets):
    """Two-anchor, evidence-graded per-release verdict plus unconditional
    rebind.

    Verdict (asserted only from the fresh same-capture observation):
      * candidate within gate of the release point -> ok, with xy error;
      * candidate within gate of the pre-place origin -> still_at_origin
        (the object never left; the release demonstrably failed);
      * neither -> abstain. Never a synthesized fail.

    Binding is separate from the verdict: detection_map[pick_label] is
    rebound to the nearest fresh instance to the release point at any
    distance, since its displacement is self-caused and
    _replay_last_pick_cycle must re-aim at where the object actually fell.
    The rebind fires even when the verdict abstains.
    """
    gate = sticky_settings(executor)["gate_m"]
    release_xy = _release_anchor_xy(executor)
    origin = (executor.detection_map.get(pick_label) or {})
    origin_pos = det_field(origin, "position_3d")
    origin_xy = (
        None if origin_pos is None
        else np.asarray(origin_pos[:2], dtype=float)
    )

    positions = [_measure_pos(d) for d in dets]
    confs = [float(det_field(d, "confidence", 0.0) or 0.0) for d in dets]
    kept = dedup_candidate_indices(positions, confs, merge_radius_m=0.02)
    cands = [(dets[i], positions[i]) for i in kept if positions[i] is not None]

    verdict = {"verdict": "abstain", "gate_m": gate}
    if not cands:
        verdict["detail"] = "object not visible after release"
        return verdict

    # Nearest fresh instance to the release point (any distance): binding.
    if release_xy is not None:
        d_rel, det, pos = min(
            (
                (float(np.linalg.norm(p[:2] - release_xy)), d, p)
                for d, p in cands
            ),
            key=lambda t: t[0],
        )
        _write_adopted(
            executor, pick_label, det, pos, pos, "post-release",
            note=f"rebind d_release={d_rel * 100:.1f}cm",
        )
        verdict["rebind_xy"] = [float(pos[0]), float(pos[1])]
        if d_rel <= gate:
            verdict["verdict"] = "ok"
            verdict["xy_error_m"] = round(d_rel, 4)
            return verdict
        if origin_xy is not None:
            d_org = float(np.linalg.norm(pos[:2] - origin_xy))
            if d_org <= gate:
                verdict["verdict"] = "still_at_origin"
                verdict["origin_dist_m"] = round(d_org, 4)
                # Evidence, not a synthesized fail: the object demonstrably
                # never left its pre-place position, which is exactly what
                # _fail_has_evidence wants behind a replay. The task verdict
                # stays SuccessVerifier's.
                from spark_real.control import success_verifier

                success_verifier.set_gate(
                    executor, "release", False,
                    f"post-release: '{pick_label}' still at origin "
                    f"({d_org * 100:.1f}cm from pre-place position)",
                )
                return verdict
        verdict["detail"] = (
            f"nearest instance {d_rel * 100:.1f}cm from the release point "
            "and not at the origin"
        )
    return verdict


def verify_placement(executor, pick_label, place_label):
    """Post-release trace emitter plus optional per-release verdict.

    The task verdict comes from control.success_verifier at end of task. The
    per-release verdict here (verification.release_verdict, default off) is a
    per-primitive attribution signal, never the run verdict. It records where
    the object appears right after the release and, when enabled, rebinds the
    placed label to where it actually fell so a replay re-aims correctly.
    """
    if executor._pipeline is None:
        return
    try:
        captures = executor._pipeline.capture()
        prompt = (
            pick_label.rsplit(" ", 1)[0]
            if pick_label and pick_label[-1].isdigit()
            else pick_label
        )
        dets = executor._pipeline.detect(captures, prompts=[prompt]) or []
        records = [detection_summary(d) for d in dets]
        for rec in records:
            logger.info(
                "Release trace: '%s' seen by %s conf=%.2f at %s",
                rec.get("label"),
                rec.get("camera"),
                float(rec.get("confidence") or 0.0),
                rec.get("position_3d"),
            )
        if not records:
            logger.info("Release trace: '%s' not visible after release", pick_label)

        params = {"pick_label": pick_label, "place_label": place_label}
        if release_verdict_enabled(executor):
            verdict = _release_verdict(executor, pick_label, dets)
            params["release_verdict"] = verdict
            logger.info(
                "Release verdict for '%s': %s", pick_label, verdict
            )

        writer = TraceWriter.from_pipeline(executor._pipeline)
        writer.write(
            PrimitiveTrace(
                index=writer.next_index(),
                primitive="release_placement",
                params=params,
                detections_used=records,
            )
        )
    except Exception as e:
        logger.warning("Release trace failed: %s", e)

"""Planner-directed grasp orientation.

A plan may name ``grasp_strategy`` (and an explicit ``grasp_yaw_deg``) on
``move_to_keypoint`` or ``grasp``; absent both, ``auto`` applies the
``aspect_ratio >= GRASP_YAW_AR_GATE`` test plus input-validity preconditions.

Every yaw is checked against the wrist-3 cable limit AFTER composition, on
the rotation that will actually be commanded, not on the requested angle.
"""

import functools
import logging
import math
import os
import shutil
import subprocess
from pathlib import Path

import numpy as np
from scipy.spatial.transform import Rotation

from spark_real.perception.mask_quality import measure_mask

logger = logging.getLogger(__name__)

STRATEGIES = ("auto", "topdown", "obb", "cgn", "se3")
DEFAULT_STRATEGY = "auto"

# Contact-GraspNet ships disabled; this is the one place that knows the gate.
CGN_ENV_GATE = "SPARK_GRASP_CGN"

# Input-validity floors for the `auto` route. A yaw read off a 24%-confidence
# blob is noise, not policy. Config: grasp.yaw_min_conf / grasp.yaw_min_obb_conf.
DEFAULT_YAW_MIN_CONF = 0.45
DEFAULT_YAW_MIN_OBB_CONF = 0.40

# Fallback wrist-3 cable limit when the executor carries none.
DEFAULT_MAX_YAW_OFFSET_RAD = math.radians(100.0)  # see executor_motion.MAX_YAW_OFFSET

# A "top-down" orientation must keep tool-Z this close to straight down.
TOPDOWN_TILT_TOL_DEG = 5.0

_WORLD_DOWN = np.array([0.0, 0.0, -1.0])


from spark_real.utils.det_fields import det_field as _field  # noqa: E402


def _tuned(executor, attr: str, env: str, default: float) -> float:
    """env override > executor attribute > module default."""
    raw = os.environ.get(env)
    if raw is None:
        raw = getattr(executor, attr, None)
    try:
        return float(raw) if raw is not None else float(default)
    except (TypeError, ValueError):
        logger.warning("%s=%r is not a number; using %.2f", env, raw, default)
        return float(default)


def yaw_min_conf(executor) -> float:
    return _tuned(executor, "GRASP_YAW_MIN_CONF", "SPARK_YAW_MIN_CONF",
                  DEFAULT_YAW_MIN_CONF)


def yaw_min_obb_conf(executor) -> float:
    return _tuned(executor, "GRASP_YAW_MIN_OBB_CONF", "SPARK_YAW_MIN_OBB_CONF",
                  DEFAULT_YAW_MIN_OBB_CONF)


def _live_fusion_gate(executor):
    """The pipeline's fusion gate, or None when fusion is switched off."""
    gate = getattr(getattr(executor, "_pipeline", None), "_fusion_gate", None)
    return gate if getattr(gate, "cfg", None) is not None else None


def axis_trust(detection, executor):
    """(value, source) for the mask-measured OBB axis trust, else (None, "").

    Prefers the value the perception gate already stamped. Falls back to
    measuring the detection's own `_mask` (click/box detections never pass
    the perception seam). Only runs when fusion is on.
    """
    v = _field(detection, "axis_trust", None)
    if v is not None:
        try:
            return float(v), "axis_trust"
        except (TypeError, ValueError):
            logger.warning("[grasp-strategy] axis_trust=%r is not a number", v)
            return None, ""

    gate = _live_fusion_gate(executor)
    if gate is None:
        return None, ""
    mask = _field(detection, "_mask", None)
    if mask is None:
        mask = _field(detection, "mask", None)
    if mask is None:
        return None, ""
    try:
        q = measure_mask(
            mask,
            reported_aspect_ratio=_field(detection, "aspect_ratio", None),
            th=gate.cfg.quality,
        )
    except Exception as exc:  # noqa: BLE001 - never let scoring stop a grasp
        logger.warning("[grasp-strategy] mask axis measurement failed (%s)", exc)
        return None, ""
    return float(q.axis_trust), "axis_trust(mask)"


def max_yaw_offset(executor) -> float:
    """Wrist-3 cable limit in radians (grasp.max_yaw_offset_deg)."""
    lim = getattr(executor, "MAX_YAW_OFFSET", None)
    try:
        lim = float(lim)
    except (TypeError, ValueError):
        lim = None
    return lim if lim and lim > 0 else DEFAULT_MAX_YAW_OFFSET_RAD


def base_orientation(executor) -> np.ndarray:
    return np.asarray(
        getattr(executor, "GRASP_ORIENTATION", [np.pi, 0.0, 0.0]), dtype=float
    )


def symmetric_yaw(yaw: float) -> float:
    """Reduce a two-jaw-symmetric yaw into (-pi/2, pi/2]."""
    y = float(yaw) % np.pi
    if y > np.pi / 2:
        y -= np.pi
    return y


def measured_yaw_offset(orientation, base) -> float:
    """World-Z yaw of `orientation` relative to `base`, from the matrices."""
    rel = (
        Rotation.from_rotvec(np.asarray(orientation, dtype=float))
        * Rotation.from_rotvec(np.asarray(base, dtype=float)).inv()
    ).as_matrix()
    return float(np.arctan2(rel[1, 0], rel[0, 0]))


def tool_z_tilt_deg(orientation) -> float:
    """Angle (deg) between the tool Z axis and straight down."""
    tool_z = Rotation.from_rotvec(np.asarray(orientation, dtype=float)).as_matrix()[:, 2]
    return float(np.rad2deg(np.arccos(np.clip(np.dot(tool_z, _WORLD_DOWN), -1.0, 1.0))))


def compose_yaw(executor, yaw: float, context: str = "grasp"):
    """Compose a world-Z yaw onto GRASP_ORIENTATION, cable limit enforced.

    Returns the rotvec list, or None when the COMMANDED rotation exceeds the
    limit (caller falls back to top-down). The limit is measured on the
    composed rotation, so a strategy that supplies a full pose is checked the
    same way an OBB yaw is.
    """
    orient = executor._oriented_grasp(float(yaw))
    if not check_yaw_limit(executor, orient, context):
        return None
    # _oriented_grasp CLIPS to the limit. A clipped yaw is a crooked grasp, so
    # refuse it and let the caller command a defined top-down instead.
    measured = measured_yaw_offset(orient, base_orientation(executor))
    if abs(measured - float(yaw)) > np.deg2rad(1.0):
        logger.warning(
            "[grasp-strategy] %s: requested yaw %.1f deg was clipped to %.1f "
            "by the wrist-3 cable limit -> refusing (top-down instead)",
            context, np.rad2deg(float(yaw)), np.rad2deg(measured),
        )
        return None
    return orient


def check_yaw_limit(executor, orientation, context: str = "grasp") -> bool:
    """True when `orientation` is within the wrist-3 cable envelope."""
    limit = max_yaw_offset(executor)
    measured = measured_yaw_offset(orientation, base_orientation(executor))
    if abs(measured) > limit + 1e-6:
        logger.warning(
            "[grasp-strategy] %s: composed yaw %.1f deg exceeds the wrist-3 "
            "cable limit %.1f deg -> refusing this orientation",
            context, np.rad2deg(measured), np.rad2deg(limit),
        )
        return False
    return True


@functools.lru_cache(maxsize=1)
def _conda_envs() -> frozenset:
    """Installed conda env names. Cached: this runs inside the motion path."""
    try:
        out = subprocess.run(
            ["conda", "env", "list"], capture_output=True, text=True, timeout=20
        ).stdout
    except Exception as exc:  # noqa: BLE001
        logger.warning("[grasp-strategy] could not list conda envs: %s", exc)
        return frozenset()
    return frozenset(
        line.split()[0]
        for line in out.splitlines()
        if line.strip() and not line.startswith("#")
    )


def cgn_gate_open(executor=None):
    """Can `grasp_strategy: "cgn"` actually run? Returns (ok, reason).

    Exposed so both the grasp node and the skill itself fall back LOUDLY
    instead of dispatching into a guaranteed no-op.
    """
    if os.environ.get(CGN_ENV_GATE, "0") != "1":
        return False, f"{CGN_ENV_GATE} is not 1"
    if executor is not None:
        try:
            family = executor._robot_family()
        except Exception as exc:  # noqa: BLE001
            return False, f"robot family unknown ({exc})"
        if family != "ur10e":
            return False, f"ur10e-only (family={family!r})"
        if getattr(executor, "_pipeline", None) is None:
            return False, "no pipeline on executor (needs cameras for the cloud)"
    return True, "ok"


def se3_gate_open(executor=None):
    """Can `grasp_strategy: "se3"` actually run? Returns (ok, reason).

    EquiGraspFlow runs out-of-process in its own conda env against its own
    pretrained weights. Both must exist or the dispatch is a guaranteed
    failure, and the caller must be told which piece is missing.
    """
    # Same default as perception.equigrasp, resolved without importing it
    # (that module pulls torch and the route state into the control layer).
    root = Path(
        os.environ.get(
            "EQUIGRASPFLOW_ROOT", Path(__file__).resolve().parents[2] / "EquiGraspFlow"
        )
    )
    if not (root / "train_results" / "pretrained_models").is_dir():
        return False, f"no EquiGraspFlow pretrained_models under {root}"
    env = os.environ.get("EQUIGRASPFLOW_CONDA_ENV", "equigraspflow")
    if shutil.which("conda") is None:
        return False, "conda not on PATH (EquiGraspFlow runs out-of-process)"
    if env not in _conda_envs():
        return False, f"conda env {env!r} does not exist"
    return True, "ok"


def resolve_strategy(params, detection, executor):
    """Return (strategy, reason). Never raises, never returns 'auto'."""
    params = params or {}
    requested = params.get("grasp_strategy")
    forced = os.environ.get("SPARK_GRASP_STRATEGY")
    if forced:
        requested = forced
    requested = str(requested or DEFAULT_STRATEGY).strip().lower()
    if requested not in STRATEGIES:
        logger.warning(
            "[grasp-strategy] unknown grasp_strategy %r; using auto", requested
        )
        requested = DEFAULT_STRATEGY

    # Loud, not a silent no-op: the planner asked for a 6-DOF backend and is
    # not getting it.
    if requested in ("cgn", "se3"):
        ok, why = (cgn_gate_open if requested == "cgn" else se3_gate_open)(executor)
        if not ok:
            logger.warning(
                "[grasp-strategy] planner asked for %r but the backend is "
                "unavailable (%s); falling back to auto", requested, why,
            )
            requested = DEFAULT_STRATEGY

    if requested != DEFAULT_STRATEGY:
        return requested, "planner"

    # auto: AR gate plus input-validity preconditions.
    if detection is None:
        return "topdown", "no detection"
    if bool(_field(detection, "low_quality", False)):
        # low_quality covers two claims: "mask is round" (no axis, must veto)
        # and "world OBB is inflated" (a depth claim that flips run to run on
        # shiny tools). Refuse only when the MASK says there is no axis; when
        # only the depth-derived OBB is suspect, use the ray-plane axis.
        if not bool(_field(detection, "shape_round", False)):
            plane = _plane_axis(detection, gate_for(executor))
            if plane is not None:
                return (
                    "plane",
                    f"low_quality is depth-driven (ar_inflated="
                    f"{bool(_field(detection, 'ar_inflated', False))}), mask ar "
                    f"{plane[1]:.2f} still has an axis -> ray-plane",
                )
        return "topdown", "mask flagged low_quality"

    ar = float(_field(detection, "aspect_ratio", 1.0) or 1.0)
    gate = float(getattr(executor, "GRASP_YAW_AR_GATE", 1.8))
    if ar < gate:
        # `aspect_ratio` is the WORLD OBB's, from the depth cloud, which a
        # thin specular tool can deflate (e.g. 1.40 against a 1.60 gate). Ask
        # the mask before giving up; a round object fails both.
        plane = _plane_axis(detection, gate)
        if plane is not None:
            return (
                "plane",
                f"world ar {ar:.2f} < gate {gate:.2f}, but the mask ar is "
                f"{plane[1]:.2f} -> ray-plane axis",
            )
        return "topdown", f"ar {ar:.2f} < gate {gate:.2f}"

    # Prefer the fusion gate's `fused_confidence` over SAM3's own score.
    conf, conf_src = _field(detection, "fused_confidence", None), "fused_conf"
    if conf is None:
        conf, conf_src = _field(detection, "confidence", None), "conf"
    min_conf = yaw_min_conf(executor)
    if conf is not None and float(conf) < min_conf:
        return "topdown", f"{conf_src} {float(conf):.2f} < {min_conf:.2f} (ar {ar:.2f})"

    min_obb = yaw_min_obb_conf(executor)
    # Two independent OBB-axis measurements: obb_confidence (depth) and
    # axis_trust (mask alone). Either may veto. Absent is UNKNOWN, not zero.
    obb_sources = {"obb_conf": _field(detection, "obb_confidence", None)}
    trust, trust_src = axis_trust(detection, executor)
    if trust is not None:
        obb_sources[trust_src] = trust
    for name, value in obb_sources.items():
        if value is not None and float(value) < min_obb:
            # The point-cloud OBB is untrustworthy, but that is a DEPTH claim.
            # The ray-plane axis (silhouette plus one centroid depth) survives
            # thin, shiny, low-return objects such as silverware and tools.
            plane = _plane_axis(detection, gate)
            if plane is not None:
                return (
                    "plane",
                    f"{name} {float(value):.2f} < {min_obb:.2f}, "
                    f"falling back to the ray-plane axis (ar {plane[1]:.2f})",
                )
            return "topdown", f"{name} {float(value):.2f} < {min_obb:.2f}"

    return "obb", f"ar {ar:.2f} >= gate {gate:.2f}, {conf_src} ok"


def gate_for(executor) -> float:
    """The elongation gate an oriented grasp must clear."""
    return float(getattr(executor, "GRASP_YAW_AR_GATE", 1.8))


def _plane_axis(detection, gate: float):
    """(angle_rad, mask_ar) from the ray-plane axis, or None if unusable.

    Requires the MASK's own aspect ratio to clear the same elongation gate the
    OBB path uses. A round mask has no axis to grasp by whatever the depth
    said, so this must not resurrect one.
    """
    angle = _field(detection, "plane_orientation_angle", None)
    if angle is None:
        return None
    ar = _field(detection, "plane_aspect_ratio", None)
    if ar is None:
        return None
    try:
        angle_f, ar_f = float(angle), float(ar)
    except (TypeError, ValueError):
        return None
    if not np.isfinite(angle_f) or not np.isfinite(ar_f) or ar_f < gate:
        return None
    return angle_f, ar_f


def _record_held_axis(detection, executor, params=None):
    """Remember which WAY the object about to be grasped points.

    A PCA axis is a line, so it cannot say which end of a screwdriver is the
    tip. The place compares the held object's heavy end against the
    receptacle's to avoid seating the tool end-for-end. Recorded here because
    resolve_grasp_orientation is the one funnel every grasp path passes through.
    """
    executor._held_mask = None
    executor._held_axis_img = None
    executor._held_axis_sign = None
    try:
        import numpy as _np

        from spark_real.perception.mask_geometry import _pca_obb, heavy_end_sign

        mask = _field(detection, "_mask", None)
        src = _field(detection, "label", None) or (
            (params or {}).get("keypoint_label") if isinstance(params, dict) else None
        )
        # Direction lives in the WHOLE tool, not the grasped sub-part: a
        # screwdriver handle is symmetric end-for-end, only the shaft breaks
        # the tie.
        try:
            from spark_real.control.executor_grasp import _parent_object_label

            parent = _parent_object_label(src, getattr(executor, "detection_map", None))
            if parent is not None:
                pm = (executor.detection_map.get(parent) or {}).get("_mask")
                if pm is not None:
                    mask, src = pm, parent
        except Exception:  # noqa: BLE001 - sub-part upgrade is best-effort
            pass

        if mask is None:
            logger.info(
                "[held-axis] '%s' has no mask; the place cannot tell its ends "
                "apart and will fall back to the undirected axis",
                src,
            )
            return

        arr = _np.asarray(mask)
        ang, ar, _, _ = _pca_obb(arr)
        sgn = float(heavy_end_sign(arr, ang))
        executor._held_mask = arr
        executor._held_axis_img = float(ang)
        executor._held_axis_sign = sgn
        logger.info(
            "[held-axis] '%s' image axis %.1f deg, ar %.2f, heavy end %s",
            src, _np.rad2deg(ang), ar,
            {1.0: "+", -1.0: "-", 0.0: "UNKNOWN (too symmetric to call)"}.get(sgn, "?"),
        )
    except Exception as exc:  # noqa: BLE001 - never let this stop a grasp
        logger.warning("[held-axis] could not record the held mask: %s", exc)


def place_aware_yaw(executor, yaw, context):
    """Choose between yaw and yaw+-180 so the PLACE that follows is reachable.

    A parallel gripper closes identically at both, but the choice fixes which
    way the held object points and hence the wrist angle the place will need
    (a bad branch can push the place past the +-100 deg wrist limit and seat
    the object end-for-end).

    Returns the chosen yaw, or the input unchanged whenever the preview
    cannot be computed (no pending place, no masks, fit undecided). It never
    picks an unreachable branch over a reachable one, and when both are
    reachable it defers to the caller's least-travel choice.
    """
    if context != "grasp":
        return yaw
    label = getattr(executor, "_pending_place_label", None)
    held = getattr(executor, "_held_mask", None)
    if not label or held is None:
        return yaw
    try:
        target = ((getattr(executor, "detection_map", None) or {}).get(label) or {}).get(
            "_mask"
        )
        if target is None:
            return yaw
        from spark_real.perception.mask_geometry import best_fit_rotation

        ang, iou, margin = best_fit_rotation(np.asarray(held), np.asarray(target))
        if ang is None:
            return yaw
        min_margin = float(getattr(executor, "JIGSAW_MIN_MARGIN", 0.05))
        min_iou = float(getattr(executor, "JIGSAW_MIN_IOU", 0.20))
        if margin < min_margin or iou < min_iou:
            return yaw  # the fit will not steer the place either; nothing to preview
        limit = float(gate_yaw_limit(executor))

        def _place_yaw(branch):
            w = branch + float(ang)
            return (w + np.pi) % (2 * np.pi) - np.pi

        alt = yaw + (np.pi if yaw < 0 else -np.pi)
        alt = (alt + np.pi) % (2 * np.pi) - np.pi
        here_ok = abs(_place_yaw(yaw)) <= limit + 1e-9
        alt_ok = abs(_place_yaw(alt)) <= limit + 1e-9
        if here_ok or not alt_ok:
            return yaw  # already fine, or neither branch helps
        logger.warning(
            "[place-aware grasp] grasping at %.1f deg would need %.1f deg at "
            "the place (limit %.0f) and seat the object end-for-end; taking "
            "the equivalent %.1f deg branch instead, whose place is %.1f deg",
            np.rad2deg(yaw), np.rad2deg(_place_yaw(yaw)), np.rad2deg(limit),
            np.rad2deg(alt), np.rad2deg(_place_yaw(alt)),
        )
        return alt
    except Exception as exc:  # noqa: BLE001 - a failed preview keeps the old choice
        logger.warning("[place-aware grasp] preview failed (%s); keeping %.1f deg",
                       exc, np.rad2deg(yaw))
        return yaw


def gate_yaw_limit(executor) -> float:
    """The wrist-3 yaw budget actually in force (config overrides the class)."""
    return float(getattr(executor, "MAX_YAW_OFFSET", DEFAULT_MAX_YAW_OFFSET_RAD))

def resolve_grasp_orientation(
    params, detection, executor, yaw_override_rad=None, context="grasp"
):
    """Return (orientation_rotvec, strategy_used).

    Pure except for the executor's current-TCP read inside
    _nearest_symmetric_yaw. `yaw_override_rad` is a caller-supplied world yaw
    (wrist refinement, slot direction) that replaces the detection OBB angle.
    'cgn'/'se3' return the neutral top-down orientation: the approach stays
    neutral and the grasp node dispatches to the 6-DOF backend.
    """
    if context == "grasp":
        _record_held_axis(detection, executor, params)
    base = list(base_orientation(executor))
    try:
        strategy, reason = resolve_strategy(params, detection, executor)
    except Exception as exc:  # noqa: BLE001 - never let strategy math stop motion
        logger.warning("[grasp-strategy] resolve failed (%s); top-down", exc)
        return base, "topdown"

    if strategy in ("cgn", "se3"):
        logger.info("[grasp-strategy] %s: %s (%s), approach stays top-down",
                    context, strategy, reason)
        return base, strategy

    # A caller-supplied yaw (slot geometry, wrist refinement) or the planner's
    # grasp_yaw_deg OUTRANKS every veto below: resolve_strategy's vetoes are
    # statements about the HELD OBJECT's mask and depth, and a slot's
    # direction is a property of the tool bed, not of the object.
    explicit_yaw = yaw_override_rad
    if explicit_yaw is None:
        _pyaw = (params or {}).get("grasp_yaw_deg")
        if _pyaw is not None:
            try:
                explicit_yaw = np.deg2rad(float(_pyaw))
            except (TypeError, ValueError):
                explicit_yaw = None

    # Only when the strategy would otherwise DROP the yaw; the "obb" branch
    # below consumes grasp_yaw_deg itself.
    if explicit_yaw is not None and strategy == "topdown":
        try:
            yaw_o = float(explicit_yaw)
        except (TypeError, ValueError):
            yaw_o = None
        if yaw_o is not None and np.isfinite(yaw_o):
            yaw_o = symmetric_yaw(yaw_o)
            if hasattr(executor, "_nearest_symmetric_yaw"):
                yaw_o = executor._nearest_symmetric_yaw(yaw_o)
            orient_o = compose_yaw(executor, yaw_o, context=context)
            if orient_o is not None:
                logger.info(
                    "[grasp-strategy] %s: caller yaw=%.1f deg (slot/refinement) "
                    "overrides strategy '%s' (%s)",
                    context, np.rad2deg(yaw_o), strategy, reason,
                )
                return orient_o, "caller_yaw"
            logger.info(
                "[grasp-strategy] %s: caller yaw refused by the wrist limit; "
                "falling through to '%s'", context, strategy,
            )

    if strategy == "topdown" and context == "place":
        # PLACING is not GRASPING. resolve_strategy's vetoes are about closing
        # jaws ON the object (fill, solidity, fragmentation, depth agreement),
        # not about which WAY the target lies. If the target is ELONGATED,
        # align to it, preferring the mask's aspect ratio over the
        # depth-derived one, and track which field supplied each number for
        # the log.
        _ar = _ar_src = None
        for _k in ("mask_aspect_ratio", "plane_aspect_ratio", "aspect_ratio"):
            _v = _field(detection, _k, None)
            if _v is not None:
                _ar, _ar_src = _v, _k
                break
        _axis = _axis_src = None
        for _k in ("plane_orientation_angle", "world_major_axis_rad", "orientation_angle"):
            _v = _field(detection, _k, None)
            if _v is not None:
                _axis, _axis_src = _v, _k
                break
        try:
            _ar_f = float(_ar) if _ar is not None else None
            _axis_f = float(_axis) if _axis is not None else None
        except (TypeError, ValueError):
            _ar_f = _axis_f = None
        if (
            _ar_f is not None
            and _axis_f is not None
            and np.isfinite(_ar_f)
            and np.isfinite(_axis_f)
            and _ar_f >= gate_for(executor)
        ):
            yaw_t = symmetric_yaw(_axis_f)
            if hasattr(executor, "_nearest_symmetric_yaw"):
                yaw_t = executor._nearest_symmetric_yaw(yaw_t)
            orient_t = compose_yaw(executor, yaw_t, context=context)
            if orient_t is not None:
                logger.info(
                    "[grasp-strategy] place: aligning to the TARGET's axis "
                    "yaw=%.1f deg [axis from %s] (target ar %.2f [from %s]); "
                    "'%s' was a grasp-quality veto and does not describe the "
                    "target's direction",
                    np.rad2deg(yaw_t), _axis_src, _ar_f, _ar_src, reason,
                )
                return orient_t, "target_axis"

    if strategy == "topdown":
        if context == "place":
            logger.info(
                "[grasp-strategy] place: topdown (%s); target-axis fallback "
                "declined: ar=%s [from %s] gate=%.2f, axis=%s [from %s]",
                reason,
                (f"{float(_ar):.2f}" if _ar is not None else "none"),
                _ar_src,
                gate_for(executor),
                (f"{float(_axis):.3f}rad" if _axis is not None else "none"),
                _axis_src,
            )
            return base, "topdown"
        logger.info("[grasp-strategy] %s: topdown (%s)", context, reason)
        return base, "topdown"

    if strategy == "plane":
        plane = _plane_axis(detection, float(getattr(executor, "GRASP_YAW_AR_GATE", 1.8)))
        if plane is None:
            logger.info("[grasp-strategy] %s: plane axis vanished -> topdown", context)
            return base, "topdown"
        yaw = symmetric_yaw(float(plane[0]))
        if hasattr(executor, "_nearest_symmetric_yaw"):
            yaw = executor._nearest_symmetric_yaw(yaw)
        orient = compose_yaw(executor, yaw, context=context)
        if orient is None:
            return base, "topdown"
        logger.info(
            "[grasp-strategy] %s: plane yaw=%.1f deg from the ray-plane axis (%s)",
            context, np.rad2deg(yaw), reason,
        )
        return orient, "plane"

    # obb
    yaw_deg = (params or {}).get("grasp_yaw_deg")
    if yaw_deg is not None:
        yaw = np.deg2rad(float(yaw_deg))
        src = f"grasp_yaw_deg={float(yaw_deg):.1f}"
    elif yaw_override_rad is not None:
        yaw = float(yaw_override_rad)
        src = "caller yaw"
    else:
        raw = _field(detection, "orientation_angle", None)
        if raw is None:
            logger.info(
                "[grasp-strategy] %s: obb requested but the detection has no "
                "orientation_angle -> topdown", context,
            )
            return base, "topdown"
        yaw = float(raw)
        src = "detection OBB"

    yaw = symmetric_yaw(yaw)
    if hasattr(executor, "_nearest_symmetric_yaw"):
        yaw = executor._nearest_symmetric_yaw(yaw)
    # Least-travel chose a branch knowing only the grasp; let the destination
    # override it when that would strand the place past the wrist limit.
    yaw = place_aware_yaw(executor, yaw, context)
    orient = compose_yaw(executor, yaw, context=context)
    if orient is None:
        return base, "topdown"
    logger.info(
        "[grasp-strategy] %s: obb yaw=%.1f deg from %s (%s)",
        context, np.rad2deg(yaw), src, reason,
    )
    return orient, "obb"

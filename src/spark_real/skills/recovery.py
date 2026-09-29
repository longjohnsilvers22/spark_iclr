"""
Recovery skills for the real UR10e pipeline.

Ported from spark_fallback/fallback_node.py recovery strategies.
These are used when primary manipulation fails: object not found,
grasp slipped, insertion blocked, etc.

Skills:
search_keypoint  - spiral search pattern if object not found
retract_retry    - retract up, offset XY randomly, retry approach
adjust_grip      - try different gripper widths to improve grasp
compliant_push   - slow descent with force monitoring, stop on contact
"""

import math
import time
import logging
import numpy as np

from spark_real.skills.registry import spark_skill
from spark_real.skills.primitives import _result

logger = logging.getLogger(__name__)

# How far to lift to clear the fixed cameras' line of sight. Deliberately not
# safe_height: 0.35 m is a TRANSIT clearance, and using it as a "look again"
# height turned every failed branch into a metre of pointless travel.
UNOCCLUDE_LIFT_M = 0.08


@spark_skill(
    name="grasp_perturb",
    description=(
        "Tier-1 in-place grasp reseat: open, nudge the TCP by a small offset "
        "(down / left-right / fwd-back), and re-close, for a barely-missed "
        "grasp, tried BEFORE the more expensive perception re-grounding. Pass an "
        "explicit offset [dx,dy,dz], or let the recovery loop sample the nudge "
        "ring by attempt index."
    ),
    params={
        "offset": list,      # [dx,dy,dz] meters; overrides the sampled ring
        "magnitude": float,  # ring nudge size (m); default 0.006
        "attempt": int,      # which ring direction to use when no explicit offset
        "force": int,
    },
)
def grasp_perturb(executor, params: dict):
    """In-place reseat of a barely-missed grasp (paper tier-1 recovery)."""
    t0 = time.time()
    force = int(params.get("force", 80))
    d = float(params.get("magnitude", 0.006))
    # Nudge ring: straight down first (the most common near-miss), then the four
    # lateral directions, then diagonals. `attempt` indexes it; `offset` overrides.
    ring = [
        [0.0, 0.0, -d], [d, 0.0, 0.0], [-d, 0.0, 0.0],
        [0.0, d, 0.0], [0.0, -d, 0.0], [d, d, 0.0], [-d, -d, 0.0],
    ]
    off = params.get("offset")
    if not off:
        off = ring[int(params.get("attempt", 0)) % len(ring)]
    off = [float(off[0]), float(off[1]), float(off[2])]

    # Open, nudge (clamped above the table), re-close.
    if hasattr(executor.robot, "open_gripper"):
        executor.robot.open_gripper()
    executor._abort_sleep(0.3)
    cur = executor._get_current_position()
    target = cur.copy()
    target[0] += off[0]
    target[1] += off[1]
    target[2] += off[2]
    floor = executor.TABLE_Z_FLOOR + 0.002
    if target[2] < floor:
        target[2] = floor
    executor._move_to(target, executor.GRASP_ORIENTATION, velocity=executor.velocity * 0.2)
    executor._abort_sleep(0.2)

    # Re-close and force-verify (gripper position/OBJ registers read stale on the
    # UR rig; _verify_grasp already defers to TCP force after its veto removal).
    if hasattr(executor, "_gripper_squeeze"):
        executor._gripper_squeeze(force=force, speed=60, settle=1.0)
    if hasattr(executor, "_robotiq_resqueeze"):
        executor._robotiq_resqueeze(speed=50, force=100, settle=0.8)
    held = False
    if hasattr(executor, "_verify_grasp"):
        try:
            held = bool(executor._verify_grasp())
        except Exception:
            held = False
    executor._holding = held
    return _result(
        "grasp_perturb",
        held,
        f"perturb dxyz=({off[0]:.3f},{off[1]:.3f},{off[2]:.3f}) -> "
        f"{'held' if held else 'empty'}",
        time.time() - t0,
    )


@spark_skill(
    name="search_keypoint",
    description=(
        "Execute an expanding spiral search pattern to find an object "
        "that was not detected in the current view"
    ),
    params={
        "keypoint_label": str,
        "search_radius": float,
        "num_loops": int,
    },
)
def search_keypoint(executor, params: dict):
    """
    Spiral search using the wrist camera to re-detect a lost object.

    Starting from the current TCP position, the end-effector traces
    an expanding spiral at the current height.  At each waypoint the
    wrist camera is queried via ``executor._refine_with_wrist()`` to
    see if the target object is visible.  If found, the detection map
    is updated so subsequent skills can use the new position.

    The spiral has *num_loops* full revolutions with 8 waypoints per
    revolution, expanding out to *search_radius*.
    """
    t0 = time.time()
    label = params.get("keypoint_label", "")
    radius = params.get("search_radius", 0.15)
    num_loops = params.get("num_loops", 2)
    points_per_loop = 8

    origin = executor._get_current_position()

    # The spiral exists to give the WRIST CAMERA new viewpoints. When wrist
    # refinement is off -- which is the default, control.wrist_refine --
    # _refine_with_wrist returns None without touching a camera, so every one
    # of the 16 waypoints gathers exactly nothing.
    #
    # The static cameras do not move with the arm, so the only thing arm
    # motion can buy them is UN-OCCLUSION. Do that once, cheaply, and
    # re-detect: lift just clear of the scene rather than to safe height, step
    # aside, look again.
    if not bool(getattr(executor, "_wrist_refine_enabled", False)):
        clear_z = min(float(origin[2]) + UNOCCLUDE_LIFT_M, float(executor.safe_height))
        logger.info(
            "[search_keypoint] wrist refine is off, so the spiral would learn "
            "nothing; lifting %.2f m to un-occlude the fixed cameras and "
            "re-detecting once instead",
            clear_z - float(origin[2]),
        )
        waypoint = origin.copy()
        waypoint[2] = clear_z
        try:
            executor._move_to(waypoint, executor.GRASP_ORIENTATION)
        except Exception as exc:  # noqa: BLE001
            logger.warning("[search_keypoint] un-occlude move failed: %s", exc)
        # _redetect_for_recovery returns a fresh world position or None, and
        # already carries its own 30 s watchdog.
        # Imported lazily: skills/ is imported during control/ package init,
        # so a module-level import here is circular.
        from spark_real.control.execution_recovery import _redetect_for_recovery

        found = _redetect_for_recovery(executor, label)
        ok = found is not None
        logger.info(
            "[search_keypoint] re-detect after un-occlude: %s",
            "found" if ok else "still not found",
        )
        return _result(
            "search_keypoint",
            ok,
            f"un-occlude + re-detect '{label}': {'found' if ok else 'not found'}",
            time.time() - t0,
        )

    search_height = max(origin[2], executor.safe_height)

    total_points = num_loops * points_per_loop

    for i in range(total_points):
        # Expanding spiral: radius grows linearly, angle advances
        frac = (i + 1) / total_points
        r = radius * frac
        theta = 2.0 * math.pi * (i / points_per_loop)

        waypoint = origin.copy()
        waypoint[0] += r * math.cos(theta)
        waypoint[1] += r * math.sin(theta)
        waypoint[2] = search_height

        executor._move_to(waypoint, executor.GRASP_ORIENTATION)

        # Check wrist camera at this position
        refined = executor._refine_with_wrist(label, waypoint)
        if refined is not None:
            pos = refined["position_3d"]
            pos_list = pos.tolist() if hasattr(pos, "tolist") else list(pos)
            logger.info(
                "search_keypoint: found '%s' at spiral point %d/%d: "
                "(%.3f, %.3f, %.3f)",
                label,
                i + 1,
                total_points,
                *pos_list,
            )
            executor.detection_map[label] = {
                "position_3d": pos_list,
                "orientation_angle": refined.get("orientation_angle", 0.0),
                "aspect_ratio": refined.get("aspect_ratio", 1.0),
            }
            return _result(
                "search_keypoint",
                True,
                f"Found '{label}' at {pos_list}",
                time.time() - t0,
            )

    # Return to origin after failed search
    executor._move_to(origin, executor.GRASP_ORIENTATION)

    return _result(
        "search_keypoint",
        False,
        f"Could not find '{label}' within {radius}m radius "
        f"({total_points} positions scanned)",
        time.time() - t0,
    )


@spark_skill(
    name="retract_retry",
    description=(
        "Retract upward, apply a small random XY offset, then "
        "re-approach; useful when an approach failed or was blocked"
    ),
    params={
        "retract_distance": float,
        "offset_range": float,
        "num_retries": int,
        "keypoint_label": str,
    },
)
def retract_retry(executor, params: dict):
    """
    Retract, offset randomly, re-approach the target.

    Steps per retry:
    1. Retract straight up by *retract_distance*.
    2. Apply a random XY offset within [-offset_range, +offset_range].
    3. Re-approach the original target.

    If *keypoint_label* is provided the detection map position is used
    as the re-approach target; otherwise the position from before the
    retract is used.
    """
    t0 = time.time()
    retract_dist = params.get("retract_distance", 0.08)
    offset_range = params.get("offset_range", 0.015)
    num_retries = params.get("num_retries", 3)
    label = params.get("keypoint_label", "")

    # Determine target position
    if label:
        det = executor.detection_map.get(label)
        if det is not None:
            target = np.array(det["position_3d"])
        else:
            target = executor._get_current_position()
    else:
        target = executor._get_current_position()

    for attempt in range(num_retries):
        current = executor._get_current_position()

        # 1. Retract upward
        retract_pos = current.copy()
        # CEILING, not an increment: an increment from wherever the arm
        # happens to be stacks another lift per recovery attempt
        # (retract-retries have walked the arm to z=0.478 and released the
        # object from there). Clamp to the same safe height the rest of the
        # executor transits at; a retract that is already high enough becomes
        # a no-op instead of climbing further.
        ceiling = float(getattr(executor, "SAFE_HEIGHT_Z", 0.35))
        retract_pos[2] = min(float(current[2]) + retract_dist, ceiling)
        if retract_pos[2] > float(current[2]) + 1e-4:
            executor._move_to(retract_pos, executor.GRASP_ORIENTATION)
        else:
            logger.info(
                "retract_retry: already at/above the %.2f m ceiling (z=%.3f); "
                "not lifting further", ceiling, float(current[2]),
            )
            retract_pos[2] = float(current[2])

        # 2. Random XY offset
        dx = np.random.uniform(-offset_range, offset_range)
        dy = np.random.uniform(-offset_range, offset_range)
        offset_pos = retract_pos.copy()
        offset_pos[0] += dx
        offset_pos[1] += dy
        executor._move_to(offset_pos, executor.GRASP_ORIENTATION)

        logger.info(
            "retract_retry attempt %d/%d: offset (%.3f, %.3f)",
            attempt + 1,
            num_retries,
            dx,
            dy,
        )

        # 3. Re-approach target
        retry_target = target.copy()
        retry_target[0] += dx
        retry_target[1] += dy
        executor._approach_target(retry_target)

        # Check if approach succeeded (simple distance check)
        final_pos = executor._get_current_position()
        dist_to_target = np.linalg.norm(final_pos[:2] - retry_target[:2])
        if dist_to_target < 0.02:
            return _result(
                "retract_retry",
                True,
                f"Re-approach succeeded on attempt {attempt + 1} "
                f"(offset: {dx:.3f}, {dy:.3f})",
                time.time() - t0,
            )

    return _result(
        "retract_retry",
        False,
        f"All {num_retries} retract-retry attempts failed",
        time.time() - t0,
    )


@spark_skill(
    name="adjust_grip",
    description=(
        "Try different gripper widths to find a stable grasp "
        "(25%, 50%, 75%, 100% closed)"
    ),
    params={
        "force": float,
        "widths": list,
    },
)
def adjust_grip(executor, params: dict):
    """
    Attempt multiple gripper widths to improve grasp quality.

    The gripper is commanded to each width in *widths* (as fractions
    0.0=open to 1.0=closed).  After each attempt the TCP force is
    checked; if the grip force is above a minimum threshold the
    grasp is considered stable and the skill returns success.

    Default widths: [0.25, 0.50, 0.75, 1.00]
    """
    t0 = time.time()
    force = params.get("force", 50)
    widths = params.get("widths", [0.25, 0.50, 0.75, 1.00])

    for width in widths:
        logger.info("adjust_grip: trying width=%.0f%%", width * 100)

        if hasattr(executor.robot, "_send_gripper_command"):
            executor.robot._send_gripper_command(
                width,
                speed=50,
                force=min(force, 100),
            )
        elif hasattr(executor.robot, "set_gripper_position"):
            executor.robot.set_gripper_position(width)
        else:
            # Fallback: binary open/close
            if width > 0.5:
                executor.robot.close_gripper()
            else:
                executor.robot.open_gripper()
        time.sleep(0.5)

        # Check grasp quality via force
        grasp_ok = False
        try:
            if hasattr(executor.robot, "get_tcp_force"):
                ft = executor.robot.get_tcp_force()
                fz = abs(ft[2]) if len(ft) > 2 else 0.0
                if fz > 1.0:
                    logger.info(
                        "adjust_grip: stable grasp at width=%.0f%% (fz=%.1fN)",
                        width * 100,
                        fz,
                    )
                    grasp_ok = True
        except Exception:
            pass

        if grasp_ok:
            executor._holding = True
            return _result(
                "adjust_grip",
                True,
                f"Stable grasp at width={width*100:.0f}%",
                time.time() - t0,
            )

    # No force feedback: verify the grasp by gripper width instead of
    # assuming success. A width at or below the empty-jaw threshold means
    # the jaws closed on nothing, so report FAILED and let recovery retry.
    width_m = None
    try:
        if hasattr(executor.robot, "get_observation"):
            obs = executor.robot.get_observation()
            if obs:
                width_m = obs.get("gripper_width")
        if width_m is None and hasattr(executor.robot, "get_gripper_width"):
            width_m = executor.robot.get_gripper_width()
    except Exception:
        width_m = None
    if width_m is not None and float(width_m) >= 0.001:
        executor._holding = True
        return _result(
            "adjust_grip",
            True,
            f"Grasp verified by width ({float(width_m):.3f}m; no force sensor)",
            time.time() - t0,
        )
    return _result(
        "adjust_grip",
        False,
        "adjust_grip: grasp not verified (jaws closed empty or no force/width feedback)",
        time.time() - t0,
    )


@spark_skill(
    name="compliant_push",
    description=(
        "Slow descent with force monitoring; stops when contact force "
        "exceeds threshold (for placing, insertion, or surface probing)"
    ),
    params={
        "direction": list,
        "max_distance": float,
        "force_threshold": float,
        "step_size": float,
    },
)
def compliant_push(executor, params: dict):
    """
    Descend slowly while monitoring force, stopping on contact.

    The default direction is straight down (-Z).  At each step the
    TCP force is checked; when the force magnitude along the push
    direction exceeds *force_threshold* the motion stops.

    This is the UR10e equivalent of the impedance-controlled
    compliant push from the simulation fallback node, but implemented
    with discrete position steps + force gating.
    """
    t0 = time.time()
    direction = np.array(params.get("direction", [0, 0, -1]), dtype=float)
    max_dist = params.get("max_distance", 0.10)
    force_thresh = params.get("force_threshold", 15.0)
    step_size = params.get("step_size", 0.002)  # 2 mm steps

    direction_norm = direction / (np.linalg.norm(direction) + 1e-8)
    num_steps = max(int(max_dist / step_size), 1)
    step_vec = direction_norm * step_size

    contact_detected = False
    steps_taken = 0
    current_pos = executor._get_current_position()

    # BASELINE the force before descending. get_tcp_force() is an absolute
    # reading that carries the tool's own weight, so projecting it onto a
    # downward push and comparing to a threshold fires on gravity alone:
    # "contact at step 1/30 (force=83.9N > 25.0N)" before the arm has moved,
    # 83.9 N being the payload hanging off the wrist. Grasp v2 takes the same
    # baseline (executor_grasp._fz_delta).
    base_f = np.zeros(3)
    try:
        if hasattr(executor.robot, "get_tcp_force"):
            _b = np.asarray(executor.robot.get_tcp_force(), dtype=float)
            if _b.size >= 3:
                base_f = _b[:3].copy()
                logger.info(
                    "compliant_push: force baseline (%.1f, %.1f, %.1f) N; "
                    "contact is judged on the DELTA from here",
                    base_f[0], base_f[1], base_f[2],
                )
    except Exception as exc:  # noqa: BLE001
        logger.warning("compliant_push: could not baseline force (%s); "
                       "contact detection disabled for this push", exc)
        base_f = None

    for i in range(num_steps):
        target = current_pos + step_vec
        # A push is a TRANSLATION. Handing it GRASP_ORIENTATION re-commands a
        # wrist angle on every step and unwinds the task's yaw (measured:
        # 80.8 deg of wrist_3 over ~4 s while standing still). Hold what the
        # arm already has.
        executor._move_to(
            target,
            executor.current_orientation(),
            velocity=executor.velocity * 0.15,  # very slow
        )
        current_pos = executor._get_current_position()
        steps_taken = i + 1

        # Force check
        try:
            if hasattr(executor.robot, "get_tcp_force"):
                ft = executor.robot.get_tcp_force()
                # Project the CHANGE in force onto the push direction. Without
                # a baseline this is the tool's weight and fires immediately.
                f_vec = np.array(ft[:3])
                if base_f is None:
                    f_along = 0.0          # no baseline -> never claim contact
                else:
                    f_along = abs(np.dot(f_vec - base_f, direction_norm))

                if f_along > force_thresh:
                    logger.info(
                        "compliant_push: contact at step %d/%d "
                        "(force=%.1fN > %.1fN)",
                        steps_taken,
                        num_steps,
                        f_along,
                        force_thresh,
                    )
                    contact_detected = True
                    break
        except Exception:
            pass

    total_dist = steps_taken * step_size
    msg = f"Pushed {total_dist*1000:.1f}mm along {direction.tolist()}"
    if contact_detected:
        msg += f" (contact at {force_thresh}N)"
    else:
        msg += " (max distance reached, no contact)"

    return _result("compliant_push", True, msg, time.time() - t0)

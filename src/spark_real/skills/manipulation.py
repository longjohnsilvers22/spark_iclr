"""
Manipulation skills: wiggle, push, pull, drag, screw, pour, stack.
"""

import math
import time
import logging
import numpy as np

from spark_real.control.grasp_strategy import resolve_grasp_orientation
from spark_real.skills.registry import spark_skill
from spark_real.skills.primitives import _result

logger = logging.getLogger(__name__)


@spark_skill(
    name="wiggle",
    description=(
        "Oscillate end-effector in XY/Z at configurable frequency and "
        "amplitude for insertion assistance or breaking static friction"
    ),
    params={
        "amplitude": float,
        "frequency": float,
        "duration": float,
        "axis": str,
        "force_threshold": float,
    },
)
def wiggle(executor, params: dict):
    """
    Wiggle TCP via speedl at ~30 Hz; exits early on force threshold.
    """
    t0 = time.time()
    amplitude = params.get("amplitude", 0.005)
    frequency = params.get("frequency", 2.0)
    duration = params.get("duration", 3.0)
    axis = params.get("axis", "xy").lower()
    force_thresh = params.get("force_threshold", 30.0)

    dt = 1.0 / 30.0
    omega = 2.0 * math.pi * frequency

    vel_amp = omega * amplitude

    insertion_detected = False
    elapsed = 0.0

    while elapsed < duration:
        t = elapsed

        vx = vel_amp * math.sin(omega * t) if "x" in axis else 0.0
        vy = vel_amp * math.cos(omega * t) if "y" in axis else 0.0
        vz = vel_amp * math.sin(omega * t) if "z" in axis else 0.0

        velocity_cmd = [vx, vy, vz, 0.0, 0.0, 0.0]

        acc = vel_amp * 5.0
        script = (
            f"speedl([{velocity_cmd[0]:.5f}, {velocity_cmd[1]:.5f}, "
            f"{velocity_cmd[2]:.5f}, {velocity_cmd[3]:.5f}, "
            f"{velocity_cmd[4]:.5f}, {velocity_cmd[5]:.5f}], "
            f"{acc:.3f}, {dt + 0.02:.4f})"
        )
        if hasattr(executor.robot, "_send_script"):
            executor.robot._send_script(script)
        elif hasattr(executor.robot, "send_velocity"):
            executor.robot.send_velocity(velocity_cmd, acc, dt + 0.02)

        try:
            if hasattr(executor.robot, "get_tcp_force"):
                force = executor.robot.get_tcp_force()
                fz = abs(force[2]) if len(force) > 2 else 0.0
                if fz > force_thresh:
                    logger.info(
                        "Wiggle: insertion detected (fz=%.1fN > %.1fN) at t=%.2fs",
                        fz,
                        force_thresh,
                        elapsed,
                    )
                    insertion_detected = True
                    break
        except Exception:
            pass

        time.sleep(dt)
        elapsed += dt

    _stop_speedl(executor)
    msg = (
        f"Wiggled axis={axis} amp={amplitude*1000:.1f}mm "
        f"freq={frequency}Hz for {elapsed:.1f}s"
    )
    if insertion_detected:
        msg += " (insertion detected)"

    return _result("wiggle", True, msg, time.time() - t0)


def _stop_speedl(executor):
    """
    Send zero-velocity speedl then stopl to decelerate gracefully.
    """
    stop_script = "speedl([0,0,0,0,0,0], 1.0, 0.1)\nstopl(1.0)"
    if hasattr(executor.robot, "_send_script"):
        executor.robot._send_script(stop_script)
    elif hasattr(executor.robot, "stop_motion"):
        executor.robot.stop_motion()
    time.sleep(0.15)


@spark_skill(
    name="push_object",
    description="Push an object in a specified direction by approaching and moving through it",
    params={
        "keypoint_label": str,
        "push_direction": list,
        "push_distance": float,
        "force_limit": float,
    },
)
def push_object(executor, params: dict):
    """
    Approach from behind and push object along push_direction.
    """
    t0 = time.time()
    label = params.get("keypoint_label", "")
    push_dir = np.array(params.get("push_direction", [0, -1, 0]), dtype=float)
    push_dist = params.get("push_distance", 0.10)
    force_limit = params.get("force_limit", 40.0)

    det = executor.detection_map.get(label)
    if det is None:
        return _result(
            "push_object",
            False,
            f"Object '{label}' not found in detections",
            time.time() - t0,
        )

    obj_pos = np.array(det["position_3d"])
    push_dir_norm = push_dir / (np.linalg.norm(push_dir) + 1e-8)

    approach_pos = obj_pos - push_dir_norm * 0.05
    executor._approach_target(approach_pos)

    num_steps = max(int(push_dist / 0.01), 5)
    step_vec = push_dir_norm * (push_dist / num_steps)
    current_target = approach_pos.copy()

    for i in range(num_steps):
        current_target = current_target + step_vec
        executor._move_to(
            current_target,
            executor.GRASP_ORIENTATION,
            velocity=executor.velocity * 0.3,
        )

        try:
            if hasattr(executor.robot, "get_tcp_force"):
                force = executor.robot.get_tcp_force()
                f_mag = np.linalg.norm(force[:3])
                if f_mag > force_limit:
                    logger.info(
                        "Push: force limit hit (%.1fN > %.1fN) at step %d/%d",
                        f_mag,
                        force_limit,
                        i + 1,
                        num_steps,
                    )
                    break
        except Exception:
            pass

    retract = current_target.copy()
    retract[2] += 0.05
    executor._move_to(retract, executor.GRASP_ORIENTATION)
    return _result(
        "push_object",
        True,
        f"Pushed '{label}' {push_dist}m along {push_dir.tolist()}",
        time.time() - t0,
    )


@spark_skill(
    name="open_drawer",
    description="Grasp a drawer/cabinet handle and pull it open",
    params={
        "keypoint_label": str,
        "pull_direction": list,
        "pull_distance": float,
        "grasp_force": float,
    },
)
def open_drawer(executor, params: dict):
    """
    Grasp handle and pull open by pull_distance.
    """
    t0 = time.time()
    label = params.get("keypoint_label", params.get("joint_name", ""))
    pull_dir = np.array(params.get("pull_direction", [0, 1, 0]), dtype=float)
    pull_dist = params.get("pull_distance", 0.15)
    grasp_force = params.get("grasp_force", 60)

    det = executor.detection_map.get(label)
    if det is None:
        return _result(
            "open_drawer",
            False,
            f"Handle '{label}' not found in detections",
            time.time() - t0,
        )

    handle_pos = np.array(det["position_3d"])
    executor._approach_target(handle_pos)

    if hasattr(executor.robot, "_send_gripper_command"):
        executor.robot._send_gripper_command(
            1.0,
            speed=100,
            force=min(grasp_force, 100),
        )
    else:
        executor.robot.close_gripper()
    time.sleep(0.5)
    pull_dir_norm = pull_dir / (np.linalg.norm(pull_dir) + 1e-8)
    num_steps = max(int(pull_dist / 0.02), 3)
    step_vec = pull_dir_norm * (pull_dist / num_steps)
    current_pos = handle_pos.copy()

    for _ in range(num_steps):
        current_pos = current_pos + step_vec
        executor._move_to(
            current_pos,
            executor.GRASP_ORIENTATION,
            velocity=executor.velocity * 0.4,
        )

    executor.robot.open_gripper()
    time.sleep(0.3)
    retract = current_pos.copy()
    retract[2] += 0.05
    executor._move_to(retract, executor.GRASP_ORIENTATION)

    return _result(
        "open_drawer",
        True,
        f"Opened '{label}' by {pull_dist}m",
        time.time() - t0,
    )


@spark_skill(
    name="screw",
    description=(
        "Rotate wrist joint (joint 5) by a given angle while maintaining "
        "TCP position, for screw/fastening operations"
    ),
    params={
        "angle": float,
        "linear_distance": float,
        "duration": float,
        "max_torque": float,
    },
)
def screw(executor, params: dict):
    """
    Rotate wrist (joint 5) while holding position; optional Z advance.
    """
    t0 = time.time()
    angle = params.get("angle", math.pi)
    linear_dist = params.get("linear_distance", 0.0)
    duration = params.get("duration", 2.0)
    max_torque = params.get("max_torque", 0.0)

    try:
        obs = executor.robot.get_observation()
        joints = list(obs["joint_positions"][:6])
    except Exception as e:
        return _result("screw", False, f"Cannot read joints: {e}", time.time() - t0)

    origin_pos = executor._get_current_position()
    num_steps = max(int(abs(angle) / (math.pi / 8)), 4)
    step_angle = angle / num_steps
    step_linear = linear_dist / num_steps
    vel_j = min(0.5, abs(angle) / duration)
    acc_j = min(vel_j * 2, 1.4)

    torque_hit = False
    last_step = num_steps

    for i in range(1, num_steps + 1):
        target_joints = list(joints)
        target_joints[5] += step_angle * i

        joints_str = ", ".join(f"{j:.6f}" for j in target_joints)
        script = f"movej([{joints_str}], a={acc_j:.3f}, v={vel_j:.3f})"

        if hasattr(executor.robot, "_send_script"):
            executor.robot._send_script(script)
        elif hasattr(executor.robot, "move_to_joint_config"):
            executor.robot.move_to_joint_config(
                target_joints,
                velocity=vel_j,
                acceleration=acc_j,
            )

        if abs(step_linear) > 1e-6:
            pos = origin_pos.copy()
            pos[2] += step_linear * i
            executor._move_to(
                pos,
                executor.GRASP_ORIENTATION,
                velocity=executor.velocity * 0.3,
            )

        time.sleep(duration / num_steps)

        if max_torque > 0:
            try:
                if hasattr(executor.robot, "get_tcp_force"):
                    ft = executor.robot.get_tcp_force()
                    tz = abs(ft[5]) if len(ft) > 5 else 0.0
                    if tz > max_torque:
                        logger.info(
                            "Screw: torque limit (tz=%.2fNm > %.2fNm) at step %d/%d",
                            tz,
                            max_torque,
                            i,
                            num_steps,
                        )
                        torque_hit = True
                        last_step = i
                        break
            except Exception:
                pass

    final_angle = step_angle * last_step
    msg = f"Screw angle={math.degrees(final_angle):.1f}deg linear={linear_dist}m"
    if torque_hit:
        msg += " (torque limit)"

    return _result("screw", True, msg, time.time() - t0)


@spark_skill(
    name="pull",
    description=(
        "Pull an object (drawer handle, lever) along a direction by a given distance. "
        "Grasps the handle, pulls, then releases."
    ),
    params={
        "keypoint_label": str,
        "pull_direction": list,
        "pull_distance": float,
        "grasp_force": float,
    },
)
def pull(executor, params: dict):
    """
    Grasp handle and pull along pull_direction by pull_distance.
    """
    t0 = time.time()
    label = params.get("keypoint_label", "")
    pull_dir = np.array(params.get("pull_direction", [0, 1, 0]), dtype=float)
    pull_dist = params.get("pull_distance", 0.15)
    grasp_force = params.get("grasp_force", 80)

    det = executor.detection_map.get(label)
    if det is None:
        return _result("pull", False, f"'{label}' not found", time.time() - t0)

    handle_pos = np.array(det["position_3d"])
    executor._approach_target(handle_pos)
    if hasattr(executor.robot, "_send_gripper_command"):
        executor.robot._send_gripper_command(
            1.0, speed=100, force=min(int(grasp_force), 100)
        )
    else:
        executor.robot.close_gripper()
    time.sleep(0.5)
    pull_dir_norm = pull_dir / (np.linalg.norm(pull_dir) + 1e-8)
    num_steps = max(int(pull_dist / 0.02), 3)
    step_vec = pull_dir_norm * (pull_dist / num_steps)
    current_pos = handle_pos.copy()

    logger.info(
        "Pull '%s': %.0fmm along (%.2f,%.2f,%.2f)",
        label,
        pull_dist * 1000,
        *pull_dir_norm,
    )

    for _ in range(num_steps):
        current_pos = current_pos + step_vec
        executor._move_to(
            current_pos, executor.GRASP_ORIENTATION, velocity=executor.velocity * 0.4
        )

    executor.robot.open_gripper()
    time.sleep(0.3)
    retract = current_pos.copy()
    retract[2] += 0.05
    executor._move_to(retract, executor.GRASP_ORIENTATION)
    return _result(
        "pull", True, f"Pulled '{label}' {pull_dist*1000:.0f}mm", time.time() - t0
    )


@spark_skill(
    name="push",
    description=(
        "Push an object along a direction by a given distance. "
        "Approaches from behind the object and pushes with the closed gripper."
    ),
    params={
        "keypoint_label": str,
        "push_direction": list,
        "push_distance": float,
    },
)
def push(executor, params: dict):
    """
    Push object along push_direction using closed gripper.
    """
    t0 = time.time()
    label = params.get("keypoint_label", "")
    push_dir = np.array(params.get("push_direction", [0, -1, 0]), dtype=float)
    push_dist = params.get("push_distance", 0.10)

    det = executor.detection_map.get(label)
    if det is None:
        return _result("push", False, f"'{label}' not found", time.time() - t0)

    obj_pos = np.array(det["position_3d"])
    push_dir_norm = push_dir / (np.linalg.norm(push_dir) + 1e-8)
    executor.robot.close_gripper()
    time.sleep(0.3)
    approach_pos = obj_pos - push_dir_norm * 0.03
    executor._approach_target(approach_pos)
    num_steps = max(int(push_dist / 0.01), 3)
    step_vec = push_dir_norm * (push_dist / num_steps)
    current_pos = approach_pos.copy()

    logger.info(
        "Push '%s': %.0fmm along (%.2f,%.2f,%.2f)",
        label,
        push_dist * 1000,
        *push_dir_norm,
    )

    for _ in range(num_steps):
        current_pos = current_pos + step_vec
        executor._move_to(
            current_pos, executor.GRASP_ORIENTATION, velocity=executor.velocity * 0.3
        )

    retract = current_pos.copy()
    retract[2] += 0.05
    executor._move_to(retract, executor.GRASP_ORIENTATION)
    executor.robot.open_gripper()
    return _result(
        "push", True, f"Pushed '{label}' {push_dist*1000:.0f}mm", time.time() - t0
    )


@spark_skill(
    name="drag",
    description=(
        "Drag an object across the surface to a target position. "
        "Grasps with light force and moves horizontally while maintaining contact."
    ),
    params={
        "keypoint_label": str,
        "target_label": str,
        "target_offset": list,
        "grasp_force": float,
    },
)
def drag(executor, params: dict):
    """
    Grasp with light force and drag horizontally to target.
    """
    t0 = time.time()
    label = params.get("keypoint_label", "")
    target_label = params.get("target_label", "")
    target_offset = params.get("target_offset", None)
    grasp_force = params.get("grasp_force", 40)

    det = executor.detection_map.get(label)
    if det is None:
        return _result("drag", False, f"'{label}' not found", time.time() - t0)

    obj_pos = np.array(det["position_3d"])
    if target_label:
        target_det = executor.detection_map.get(target_label)
        if target_det is None:
            return _result(
                "drag", False, f"Target '{target_label}' not found", time.time() - t0
            )
        target_pos = np.array(target_det["position_3d"])
    elif target_offset is not None:
        offset = np.array(target_offset, dtype=float)
        target_pos = obj_pos.copy()
        target_pos[0] += offset[0]
        target_pos[1] += offset[1] if len(offset) > 1 else 0
    else:
        return _result("drag", False, "No target specified", time.time() - t0)
    executor._approach_target(obj_pos)
    if hasattr(executor.robot, "_send_gripper_command"):
        executor.robot._send_gripper_command(
            1.0, speed=50, force=min(int(grasp_force), 100)
        )
    else:
        executor.robot.close_gripper()
    time.sleep(0.5)
    drag_target = target_pos.copy()
    drag_target[2] = obj_pos[2]

    drag_dist = np.linalg.norm(drag_target[:2] - obj_pos[:2])
    num_steps = max(int(drag_dist / 0.01), 5)
    step_vec = (drag_target - obj_pos) / num_steps

    logger.info(
        "Drag '%s' -> '%s': %.0fmm in %d steps",
        label,
        target_label or f"offset {target_offset}",
        drag_dist * 1000,
        num_steps,
    )

    current_pos = obj_pos.copy()
    for _ in range(num_steps):
        current_pos = current_pos + step_vec
        executor._move_to(
            current_pos, executor.GRASP_ORIENTATION, velocity=executor.velocity * 0.3
        )

    executor.robot.open_gripper()
    time.sleep(0.3)
    executor._holding = False
    retract = current_pos.copy()
    retract[2] += 0.05
    executor._move_to(retract, executor.GRASP_ORIENTATION)

    return _result(
        "drag",
        True,
        f"Dragged '{label}' {drag_dist*1000:.0f}mm",
        time.time() - t0,
    )


@spark_skill(
    name="stack",
    description=(
        "Place a held object on top of another using force-guided descent. "
        "Descends slowly until contact is detected via F/T sensor, then releases. "
        "Much more reliable than guessing offset_z for stacking."
    ),
    params={
        "target_label": str,
        "force_threshold": float,
        "max_descent": float,
        "clearance": float,
    },
)
def stack(executor, params: dict):
    """
    Force-guided descent onto target; release on contact.
    """
    t0 = time.time()
    target_label = params.get("target_label", "")
    # Config-driven tuning (control/executor_core.py class defaults, overridden
    # by the `stack:` YAML block); explicit BT params still win per-call.
    force_thresh = float(params.get("force_threshold", executor.STACK_CONTACT_FORCE_N))
    max_descent = float(params.get("max_descent", executor.STACK_MAX_DESCENT_M))
    clearance = float(params.get("clearance", executor.STACK_CLEARANCE_M))
    hard_stop_n = float(params.get("hard_stop_delta_n", executor.STACK_HARD_STOP_DELTA_N))
    desc_vel = max(0.01, executor.velocity * float(executor.STACK_DESCENT_VEL_FRAC))

    if not executor._holding:
        return _result(
            "stack", False, "Not holding anything to stack", time.time() - t0
        )

    det = executor.detection_map.get(target_label)
    if det is None:
        return _result(
            "stack", False, f"Target '{target_label}' not found", time.time() - t0
        )

    target_pos = np.array(det["position_3d"])
    # Placement orientation: orient the held object to the TARGET's OBB (as if we
    # were grasping the target) so it lands ALIGNED with it, and hold that yaw
    # through transport instead of reverting the wrist to the base orientation.
    # When the pick and place objects share an angle (e.g. two same-size blocks)
    # the wrist barely moves from pick to place.
    place_orient = executor.GRASP_ORIENTATION
    try:
        # Same resolver as the grasp/place paths: no bare AR comparison.
        _orient, _strategy = resolve_grasp_orientation(
            params, det, executor, context="stack"
        )
        if _strategy == "obb":
            place_orient = _orient
    except Exception as _e:  # noqa: BLE001
        logger.debug("Stack: oriented placement unavailable (%s)", _e)
    above = target_pos.copy()
    above[2] += clearance
    current = executor._get_current_position()
    safe_z = max(current[2], above[2] + 0.05)
    lift = current.copy()
    lift[2] = safe_z
    executor._move_to(lift, place_orient)
    if hasattr(executor, "_verify_holding_during_transport"):
        if not executor._verify_holding_during_transport():
            executor._holding = False
            return _result(
                "stack", False, "Object lost during transport", time.time() - t0
            )
    above_high = above.copy()
    above_high[2] = safe_z
    executor._move_to(above_high, place_orient)
    executor._move_to(above, place_orient)

    # GUARD: confirm the arm is actually ABOVE the target before descending.
    # The positioning moves can early-return short of the target (movej's
    # _wait_for_motion treats the slow initial creep as "stationary"). If we
    # then enter the descent loop, each "2mm step" becomes a full lateral move
    # to target_xy at the crawl descent velocity -> a 20-30cm diagonal drag
    # that times out (observed: arm at Y=0.23, target Y=-0.047, dist=0.285m).
    arrived = False
    for _ in range(3):
        executor._check_abort()
        cur = executor._get_current_position()
        if float(np.linalg.norm(cur[:2] - above[:2])) < 0.01:  # within 1cm XY
            arrived = True
            break
        executor._move_to(above, place_orient)  # full speed, retry
    if not arrived:
        cur = executor._get_current_position()
        xy_err = float(np.linalg.norm(cur[:2] - above[:2]))
        # Bail only on a REAL drag risk (>4cm). A couple-cm residual (the pyroki
        # IK sometimes leaves ~2cm on far/yawed place targets) is fine to place;
        # a slightly offset stack beats aborting (a 2cm gate tripped on a 22mm
        # residual). CRITICAL: do NOT clear _holding here; the block is still
        # physically in the jaws, and marking not-holding lets the post-task
        # cleanup open the gripper and DROP it from height. Keep holding so a
        # failure leaves the block safely gripped, not dropped.
        if xy_err > 0.04:
            return _result(
                "stack",
                False,
                f"Could not position above '{target_label}' before descent "
                f"(xy_err={xy_err * 1000:.0f}mm)",
                time.time() - t0,
            )
        logger.info(
            "Stack: %.0fmm off above '%s', within place tolerance, descending",
            xy_err * 1000, target_label,
        )

    baseline_fz = 0.0
    try:
        if hasattr(executor.robot, "get_tcp_force"):
            ft = executor.robot.get_tcp_force()
            baseline_fz = abs(ft[2]) if len(ft) > 2 else 0.0
    except Exception:
        pass
    contact_z = None
    floor_z = executor.TABLE_Z_FLOOR + 0.002

    def _fz_delta():
        try:
            if hasattr(executor.robot, "get_tcp_force"):
                ft = executor.robot.get_tcp_force()
                fz = abs(ft[2]) if len(ft) > 2 else 0.0
                return fz, fz - baseline_fz
        except Exception:
            pass
        return baseline_fz, 0.0

    # Hard stop / contact both test the baseline-SUBTRACTED delta, NOT absolute
    # fz. getActualTCPForce carries a large constant gripper/payload offset
    # (~69N); an absolute threshold tripped on step 1 and DROPPED the
    # block 12cm up. The delta cancels the offset so only a real collision spikes.
    # Continuous force-guided descent for ANY arm with velocity control (UR
    # speedl, FR3 via franky send_velocity). The stepped path below is only a
    # last-resort fallback for a driver that exposes no send_velocity at all.
    use_continuous = hasattr(executor.robot, "send_velocity")
    if use_continuous:
        # Continuous constant-velocity descent: smooth (no discrete-movel
        # crawl/under-shoot), and a steady velocity keeps the wrist force flat so
        # a real block-on-block contact spikes cleanly (discrete 2mm stepping
        # moved ~30mm in 60s and timed out).
        logger.info(
            "Stack: force-guided descent toward '%s' from z=%.3f "
            "(v=%.3f m/s, contact>%.1fN, cap=%.0fmm)",
            target_label, above[2], desc_vel, force_thresh, max_descent * 1000,
        )
        # Two-phase descent: fast until we near the target top, then CRAWL.
        # Rigid block-on-block contact builds force in <1mm, so at full descent
        # speed the ~10Hz force loop only samples every ~2.5mm and overshoots
        # the threshold hard (e.g. 4N -> 19N in one step). Crawling once within
        # slow_margin of the target's detected top keeps that final overshoot
        # small so the press is gentle, without slowing the whole descent.
        slow_margin = float(params.get(
            "slow_margin", getattr(executor, "STACK_SLOW_MARGIN_M", 0.04)))
        slow_vel = max(0.006, desc_vel * float(
            getattr(executor, "STACK_SLOW_VEL_FRAC", 0.35)))
        slow_z = target_pos[2] + slow_margin
        t_start = time.time()
        _step = 0
        _crawling = False
        while not executor._abort:
            cur = executor._get_current_position()
            if cur[2] <= floor_z or (above[2] - cur[2]) >= max_descent:
                contact_z = cur[2]
                logger.info("Stack: descent cap/floor reached at z=%.3f", cur[2])
                break
            if time.time() - t_start > 25.0:
                contact_z = cur[2]
                logger.warning("Stack: descent time cap at z=%.3f", cur[2])
                break
            fz, delta = _fz_delta()
            _step += 1
            # Continuous force readout: log the whole descent profile (~10 Hz)
            # so the fz/delta curve up to contact is inspectable in the log.
            if _step % 3 == 0:
                logger.info(
                    "Stack descent: z=%.4f  fz=%.2f  delta=%.2fN  (contact>%.1f)",
                    cur[2], fz, delta, force_thresh,
                )
            # Contact = a big CHANGE in the wrist z-force, EITHER SIGN. On a
            # block-on-block landing the held object's weight is partly borne by
            # the lower block, so abs(ft_z) DROPS (delta goes strongly negative);
            # a straight push-into would raise it. A positive-only test never
            # detects the landing (the arm rams down and holds ~42N until the
            # 25s time cap), so use |delta|.
            if abs(delta) > hard_stop_n:
                logger.warning("Stack: HARD STOP |delta|=%.1fN at z=%.3f", delta, cur[2])
                contact_z = cur[2]
                break
            if abs(delta) > force_thresh:
                contact_z = cur[2]
                logger.info("Stack: contact at z=%.3f (delta=%.1fN)", cur[2], delta)
                break
            # Crawl for the final approach near the target top (gentle landing).
            v = desc_vel
            if cur[2] <= slow_z:
                v = slow_vel
                if not _crawling:
                    _crawling = True
                    logger.info(
                        "Stack: entering slow zone at z=%.3f (v=%.3f m/s)",
                        cur[2], slow_vel,
                    )
            # Positional args: SafeRobot names the 3rd param `time_duration`,
            # the raw driver names it `duration`; positional works for both.
            # Short duration (0.06s) so a contact-break overshoots ~1.5mm, not ~3mm.
            executor.robot.send_velocity([0.0, 0.0, -v, 0.0, 0.0, 0.0], 0.5, 0.06)
            time.sleep(0.03)
        # stop residual motion
        try:
            executor.robot.send_velocity([0.0] * 6, 1.0, 0.05)
        except Exception:
            pass
        for _stop in ("stop", "stop_motion"):
            if hasattr(executor.robot, _stop):
                try:
                    getattr(executor.robot, _stop)()
                except Exception:
                    pass
                break
    else:
        # Stepped _move_to fallback (franka / drivers without send_velocity).
        step_size = float(executor.STACK_STEP_M)
        num_steps = max(int(max_descent / step_size), 1)
        descent_pos = above.copy()
        logger.info("Stack: stepped descent toward '%s' from z=%.3f", target_label, above[2])
        for i in range(num_steps):
            if executor._abort:
                logger.warning("Stack: aborted at step %d/%d", i, num_steps)
                break
            descent_pos[2] -= step_size
            descent_pos[:2] = above[:2]
            executor._move_to(descent_pos, place_orient, velocity=desc_vel)
            time.sleep(0.05)
            fz, force_delta = _fz_delta()
            if force_delta > hard_stop_n:
                logger.warning("Stack: HARD STOP delta=%.1fN at step %d", force_delta, i + 1)
                descent_pos[2] += step_size * 3
                executor._move_to(descent_pos, place_orient, velocity=desc_vel)
                contact_z = descent_pos[2]
                break
            if force_delta > force_thresh:
                contact_z = descent_pos[2]
                logger.info("Stack: contact at z=%.3f (delta=%.1fN) step %d/%d",
                            contact_z, force_delta, i + 1, num_steps)
                break

    if executor._abort:
        executor._holding = False
        return _result("stack", False, "Aborted during descent", time.time() - t0)

    if contact_z is None:
        contact_z = executor._get_current_position()[2]
        logger.warning(
            "Stack: no contact detected after %.0fmm descent, " "releasing at z=%.3f",
            max_descent * 1000,
            contact_z,
        )

    if hasattr(executor.robot, "_send_gripper_command"):
        executor.robot._send_gripper_command(0.0, speed=30, force=20)
    else:
        executor.robot.open_gripper()
    time.sleep(0.8)
    executor._holding = False
    retract = executor._get_current_position()
    retract[2] += 0.06
    executor._move_to(
        retract, executor.GRASP_ORIENTATION, velocity=executor.velocity * 0.3
    )

    descent_mm = (above[2] - contact_z) * 1000
    return _result(
        "stack",
        True,
        f"Stacked on '{target_label}' at z={contact_z:.3f} "
        f"(descended {descent_mm:.0f}mm, contact={'yes' if contact_z is not None else 'no'})",
        time.time() - t0,
    )

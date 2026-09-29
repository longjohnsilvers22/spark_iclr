"""
Single-arm cloth manipulation skills: fold, hem press, smooth crease.

Adapted from the bimanual fold on the bimanual host. Single-arm folds
one edge at a time: grip corner, lift, arc swing to mirror position
across fold axis, lower to lay. Hem pressing uses sinusoidal velocity
along the fold crease to flatten it.
"""

import math
import time
import logging
import numpy as np

from spark_real.skills.registry import spark_skill
from spark_real.skills.primitives import _result
from spark_real.control.executor_types import ExecutionResult

logger = logging.getLogger(__name__)

GRASP_ORIENT = [math.pi, 0.0, 0.0]
TABLE_Z_FLOOR = 0.002


def _get_detection(executor, label):
    det = executor.detection_map.get(label)
    if det is None:
        return None, None
    return np.asarray(det["position_3d"], dtype=float), det


def _move_to_xyz(executor, pos, orient=None, vel=0.08):
    """
    Move gripper to a world-frame position.
    """
    orient = orient or GRASP_ORIENT
    executor._move_to(pos, orient, velocity=vel)


def _arc_waypoints(start, end, peak_height, n_steps=8):
    """
    Generate arc waypoints from start to end with peak at peak_height.

    Returns list of (n_steps+1) xyz positions tracing a parabolic arc.
    """
    pts = []
    for i in range(n_steps + 1):
        t = i / n_steps
        xy = start[:2] * (1 - t) + end[:2] * t
        # parabolic z: peaks at t=0.5
        z_base = start[2] * (1 - t) + end[2] * t
        z_arc = peak_height * 4.0 * t * (1.0 - t)
        pts.append(np.array([xy[0], xy[1], z_base + z_arc]))
    return pts


@spark_skill(
    name="cloth_fold_single",
    description=(
        "Single-arm cloth fold: grip one edge/corner, lift, swing arc "
        "to mirror position across fold axis, lower to lay. Repeat for "
        "each fold step. For a full shirt fold, the planner chains "
        "multiple cloth_fold_single actions with different corner labels. "
        "For automated t-shirt folding, use the 'cloth_fold' primitive "
        "instead, which handles detection and multi-step folding internally."
    ),
    params={
        "corner_label": "detection label of the corner/edge to grip",
        "anchor_label": "detection label of the anchor edge (fold axis midpoint)",
        "anchor_midpoint": "explicit [x,y,z] of fold axis midpoint (alternative to anchor_label)",
        "lift_height": "height above cloth to lift before swing (default 0.15m)",
        "place_height": "height above table to lay the fold (default 0.01m)",
        "force": "gripper force for cloth pinch (default 15N)",
        "grip_width": "gripper width for cloth pinch (default 0.003m)",
        "arc_steps": "number of waypoints in the swing arc (default 8)",
        "approach_height": "hover height above corner before descent (default 0.08m)",
        "release_after": "release gripper after fold (default true)",
    },
)
def cloth_fold_single(executor, params: dict):
    t0 = time.time()

    corner_label = params.get("corner_label", "corner")
    corner_pos, corner_det = _get_detection(executor, corner_label)
    if corner_pos is None:
        return _result(
            "cloth_fold_single",
            False,
            f"corner '{corner_label}' not found",
            time.time() - t0,
        )

    # fold axis midpoint
    anchor_mid = params.get("anchor_midpoint")
    if anchor_mid is None:
        anchor_label = params.get("anchor_label")
        if anchor_label:
            anchor_pos, _ = _get_detection(executor, anchor_label)
            if anchor_pos is not None:
                anchor_mid = anchor_pos
    if anchor_mid is None:
        return _result(
            "cloth_fold_single",
            False,
            "need anchor_midpoint or anchor_label",
            time.time() - t0,
        )
    anchor_mid = np.asarray(anchor_mid, dtype=float)

    lift_h = float(params.get("lift_height", 0.15))
    place_h = float(params.get("place_height", 0.01))
    approach_h = float(params.get("approach_height", 0.08))
    force = float(params.get("force", 15.0))
    grip_w = float(params.get("grip_width", 0.003))
    arc_steps = int(params.get("arc_steps", 8))
    release = bool(params.get("release_after", True))

    # mirror corner across anchor midpoint in XY
    target_xy = 2.0 * anchor_mid[:2] - corner_pos[:2]
    target_z = corner_pos[2] + place_h

    logger.info(
        "[cloth_fold_single] corner=%s anchor_mid=%s target_xy=%s",
        corner_pos.round(3),
        anchor_mid.round(3),
        target_xy.round(3),
    )

    # 1. hover above corner
    hover = corner_pos.copy()
    hover[2] += approach_h
    _move_to_xyz(executor, hover, vel=0.10)

    # 2. descend to cloth surface
    pinch_pos = corner_pos.copy()
    pinch_pos[2] = max(pinch_pos[2] + 0.005, TABLE_Z_FLOOR)
    _move_to_xyz(executor, pinch_pos, vel=0.04)

    # 3. pinch grip
    executor.robot.close_gripper(force=force)
    time.sleep(0.3)
    width = executor.robot.get_gripper_width()
    if width > 0.03:
        logger.warning(
            "[cloth_fold_single] gripper too open (%.3fm), cloth not gripped", width
        )
        executor.robot.open_gripper()
        return _result(
            "cloth_fold_single",
            False,
            "cloth not gripped (gripper too open)",
            time.time() - t0,
        )

    # 4. lift
    lift_pos = corner_pos.copy()
    lift_pos[2] += lift_h
    _move_to_xyz(executor, lift_pos, vel=0.06)

    # 5. arc swing to mirror position
    end_pos = np.array([target_xy[0], target_xy[1], target_z + lift_h * 0.3])
    waypoints = _arc_waypoints(
        lift_pos, end_pos, peak_height=lift_h * 0.3, n_steps=arc_steps
    )
    for wp in waypoints[1:]:
        _move_to_xyz(executor, wp, vel=0.06)

    # 6. lower to lay the fold
    lay_pos = np.array([target_xy[0], target_xy[1], target_z])
    _move_to_xyz(executor, lay_pos, vel=0.03)

    # 7. release and retract
    if release:
        executor.robot.open_gripper()
        time.sleep(0.2)
        retract = lay_pos.copy()
        retract[2] += approach_h
        _move_to_xyz(executor, retract, vel=0.08)

    return _result(
        "cloth_fold_single",
        True,
        f"folded '{corner_label}' across {anchor_mid.round(3).tolist()}",
        time.time() - t0,
    )


@spark_skill(
    name="hem_press",
    description=(
        "Press and smooth a fold crease using sinusoidal motion along "
        "the fold line. Gripper descends onto the crease and wiggles "
        "back and forth to flatten it."
    ),
    params={
        "start_label": "detection label of one end of the crease",
        "end_label": "detection label of the other end of the crease",
        "start_pos": "explicit [x,y,z] start of crease (alternative to start_label)",
        "end_pos": "explicit [x,y,z] end of crease (alternative to end_label)",
        "force": "downward force during press (default 10N)",
        "amplitude": "lateral wiggle amplitude in meters (default 0.008)",
        "frequency": "wiggle frequency in Hz (default 2.0)",
        "passes": "number of passes along the crease (default 2)",
        "press_height": "height above table for pressing (default 0.005m)",
    },
)
def hem_press(executor, params: dict):
    t0 = time.time()

    # resolve crease endpoints
    start = params.get("start_pos")
    if start is None:
        sl = params.get("start_label")
        if sl:
            pos, _ = _get_detection(executor, sl)
            start = pos
    end = params.get("end_pos")
    if end is None:
        el = params.get("end_label")
        if el:
            pos, _ = _get_detection(executor, el)
            end = pos

    if start is None or end is None:
        return _result(
            "hem_press",
            False,
            "need start and end positions or labels",
            time.time() - t0,
        )

    start = np.asarray(start, dtype=float)
    end = np.asarray(end, dtype=float)

    force = float(params.get("force", 10.0))
    amplitude = float(params.get("amplitude", 0.008))
    freq = float(params.get("frequency", 2.0))
    passes = int(params.get("passes", 2))
    press_z = float(params.get("press_height", 0.005))

    crease_vec = end[:2] - start[:2]
    crease_len = float(np.linalg.norm(crease_vec))
    if crease_len < 0.01:
        return _result("hem_press", False, "crease too short", time.time() - t0)

    crease_dir = crease_vec / crease_len
    # perpendicular for wiggle
    perp = np.array([-crease_dir[1], crease_dir[0]])

    logger.info(
        "[hem_press] crease %.3fm from %s to %s, %d passes",
        crease_len,
        start.round(3),
        end.round(3),
        passes,
    )

    # close gripper to use as pressing tool
    executor.robot.close_gripper(force=force)
    time.sleep(0.2)

    # hover above crease start
    hover = start.copy()
    hover[2] = press_z + 0.06
    _move_to_xyz(executor, hover, vel=0.08)

    # descend to press height
    press_start = start.copy()
    press_start[2] = press_z
    _move_to_xyz(executor, press_start, vel=0.03)

    # sinusoidal press along crease
    omega = 2.0 * math.pi * freq
    dt = 1.0 / 30.0
    vel_amp = omega * amplitude

    for p in range(passes):
        direction = 1.0 if p % 2 == 0 else -1.0
        steps = int(crease_len / (0.04 * dt * 30))
        steps = max(steps, 20)

        for i in range(steps):
            t = i * dt
            frac = i / steps
            # position along crease
            base_xy = start[:2] + crease_dir * crease_len * (
                frac if direction > 0 else (1 - frac)
            )
            # sinusoidal lateral wiggle
            wiggle_offset = perp * amplitude * math.sin(omega * t)
            target_xy = base_xy + wiggle_offset

            vel_cmd = [0.0, 0.0, 0.0, 0.0, 0.0, 0.0]
            vel_cmd[0] = float(
                crease_dir[0] * 0.03 * direction
                + perp[0] * vel_amp * math.cos(omega * t)
            )
            vel_cmd[1] = float(
                crease_dir[1] * 0.03 * direction
                + perp[1] * vel_amp * math.cos(omega * t)
            )

            if hasattr(executor.robot, "send_velocity"):
                try:
                    executor.robot.send_velocity(
                        vel_cmd, acceleration=0.5, duration=dt + 0.02
                    )
                except Exception:
                    pass
            time.sleep(dt)

    # retract
    retract = end.copy() if passes % 2 == 0 else start.copy()
    retract[2] = press_z + 0.06
    _move_to_xyz(executor, retract, vel=0.06)
    executor.robot.open_gripper()

    return _result(
        "hem_press",
        True,
        f"pressed crease ({passes} passes, {crease_len*1000:.0f}mm)",
        time.time() - t0,
    )


# pants / shorts folding


def _pinch_at(executor, pos, force=15.0, approach_h=0.08, vel_down=0.04):
    """
    Hover above pos, descend, pinch cloth. Returns True if grip succeeded.
    """
    hover = pos.copy()
    hover[2] += approach_h
    _move_to_xyz(executor, hover, vel=0.10)

    pinch = pos.copy()
    pinch[2] = max(pinch[2] + 0.005, TABLE_Z_FLOOR)
    _move_to_xyz(executor, pinch, vel=vel_down)

    executor.robot.close_gripper(force=force)
    time.sleep(0.3)
    width = executor.robot.get_gripper_width()
    if width > 0.03:
        logger.warning("[_pinch_at] gripper too open (%.3fm), cloth not gripped", width)
        executor.robot.open_gripper()
        return False
    return True


def _fold_arc(executor, start_pos, end_xy, lift_h, place_h, arc_steps=8):
    """
    Lift from start_pos, arc swing to end_xy, lower to place_h above table.

    Assumes the gripper is already closed on the cloth at start_pos.
    Does not release the gripper.
    """
    lift_pos = start_pos.copy()
    lift_pos[2] += lift_h
    _move_to_xyz(executor, lift_pos, vel=0.06)

    end_z = start_pos[2] + place_h + lift_h * 0.3
    end_pos = np.array([end_xy[0], end_xy[1], end_z])
    waypoints = _arc_waypoints(
        lift_pos, end_pos, peak_height=lift_h * 0.3, n_steps=arc_steps
    )
    for wp in waypoints[1:]:
        _move_to_xyz(executor, wp, vel=0.06)

    lay_pos = np.array([end_xy[0], end_xy[1], start_pos[2] + place_h])
    _move_to_xyz(executor, lay_pos, vel=0.03)
    return lay_pos


@spark_skill(
    name="pants_fold",
    description=(
        "Single-arm pants/shorts fold. Two sequential folds: "
        "(1) grip one leg cuff and swing it across to the other leg "
        "(lengthwise fold), (2) grip the cuff end and swing to the "
        "waistband (crosswise fold). Optionally finishes with a hem "
        "press on the final crease. Requires four detection labels: "
        "the two leg cuffs (left_cuff, right_cuff) and the waistband "
        "center."
    ),
    params={
        "left_cuff_label": "detection label of the left leg cuff (default 'left cuff')",
        "right_cuff_label": "detection label of the right leg cuff (default 'right cuff')",
        "waistband_label": "detection label of the waistband center (default 'waistband')",
        "lift_height": "height to lift cloth before swing (default 0.15m)",
        "place_height": "height above table to lay the fold (default 0.01m)",
        "force": "gripper force for cloth pinch (default 15N)",
        "approach_height": "hover height above surface before descent (default 0.08m)",
        "arc_steps": "number of waypoints in each swing arc (default 8)",
        "press_crease": "run hem_press on the final fold crease (default true)",
    },
)
def pants_fold(executor, params: dict):
    """
    Two-fold pants/shorts fold: lengthwise then crosswise.
    """
    t0 = time.time()

    left_label = params.get("left_cuff_label", "left cuff")
    right_label = params.get("right_cuff_label", "right cuff")
    waist_label = params.get("waistband_label", "waistband")

    left_pos, _ = _get_detection(executor, left_label)
    right_pos, _ = _get_detection(executor, right_label)
    waist_pos, _ = _get_detection(executor, waist_label)

    if left_pos is None:
        return _result(
            "pants_fold", False, f"'{left_label}' not found", time.time() - t0
        )
    if right_pos is None:
        return _result(
            "pants_fold", False, f"'{right_label}' not found", time.time() - t0
        )
    if waist_pos is None:
        return _result(
            "pants_fold", False, f"'{waist_label}' not found", time.time() - t0
        )

    lift_h = float(params.get("lift_height", 0.15))
    place_h = float(params.get("place_height", 0.01))
    approach_h = float(params.get("approach_height", 0.08))
    force = float(params.get("force", 15.0))
    arc_steps = int(params.get("arc_steps", 8))
    press = bool(params.get("press_crease", True))

    logger.info(
        "[pants_fold] left_cuff=%s right_cuff=%s waist=%s",
        left_pos.round(3),
        right_pos.round(3),
        waist_pos.round(3),
    )

    # fold 1: lengthwise fold - swing left cuff across to right cuff.
    # Fold axis runs along the garment midline (waist to midpoint of cuffs).
    logger.info("[pants_fold] fold 1: lengthwise (left cuff -> right cuff)")
    if not _pinch_at(executor, left_pos, force=force, approach_h=approach_h):
        return _result(
            "pants_fold", False, "failed to grip left cuff for fold 1", time.time() - t0
        )

    fold1_end_xy = right_pos[:2].copy()
    lay1 = _fold_arc(
        executor,
        left_pos,
        fold1_end_xy,
        lift_h=lift_h,
        place_h=place_h,
        arc_steps=arc_steps,
    )

    # release and retract
    executor.robot.open_gripper()
    time.sleep(0.2)
    retract1 = lay1.copy()
    retract1[2] += approach_h
    _move_to_xyz(executor, retract1, vel=0.08)

    # after fold 1 both cuffs are stacked near right_pos
    # fold 2: crosswise fold - grip the stacked cuffs, swing to waistband
    logger.info("[pants_fold] fold 2: crosswise (cuffs -> waistband)")
    cuff_stack_pos = right_pos.copy()
    if not _pinch_at(executor, cuff_stack_pos, force=force, approach_h=approach_h):
        return _result(
            "pants_fold",
            False,
            "failed to grip cuff stack for fold 2",
            time.time() - t0,
        )

    fold2_end_xy = waist_pos[:2].copy()
    lay2 = _fold_arc(
        executor,
        cuff_stack_pos,
        fold2_end_xy,
        lift_h=lift_h,
        place_h=place_h,
        arc_steps=arc_steps,
    )

    executor.robot.open_gripper()
    time.sleep(0.2)
    retract2 = lay2.copy()
    retract2[2] += approach_h
    _move_to_xyz(executor, retract2, vel=0.08)

    # optional crease press on the final fold line
    if press:
        # crease runs perpendicular to the cuff-to-waist direction at the
        # midpoint between cuff_stack and waistband
        mid = 0.5 * (cuff_stack_pos[:2] + waist_pos[:2])
        fold_dir = waist_pos[:2] - cuff_stack_pos[:2]
        fold_len = float(np.linalg.norm(fold_dir))
        if fold_len > 0.02:
            fold_dir_n = fold_dir / fold_len
            perp = np.array([-fold_dir_n[1], fold_dir_n[0]])
            crease_half = fold_len * 0.4
            crease_start = np.array(
                [
                    mid[0] - perp[0] * crease_half,
                    mid[1] - perp[1] * crease_half,
                    TABLE_Z_FLOOR + 0.005,
                ]
            )
            crease_end = np.array(
                [
                    mid[0] + perp[0] * crease_half,
                    mid[1] + perp[1] * crease_half,
                    TABLE_Z_FLOOR + 0.005,
                ]
            )
            logger.info(
                "[pants_fold] pressing crease from %s to %s",
                crease_start.round(3),
                crease_end.round(3),
            )
            hem_press(
                executor,
                {
                    "start_pos": crease_start.tolist(),
                    "end_pos": crease_end.tolist(),
                    "passes": 1,
                    "force": 10.0,
                },
            )

    return _result(
        "pants_fold",
        True,
        f"folded pants: lengthwise ({left_label}->{right_label}) "
        f"then crosswise (cuffs->{waist_label})",
        time.time() - t0,
    )


@spark_skill(
    name="bimanual_pants_fold",
    description=(
        "Bimanual pants/shorts fold. Phase 1: both arms grip one leg "
        "cuff each, lift together, and swing legs to meet (lengthwise "
        "fold). Phase 2: one arm releases, the other holds while the "
        "free arm re-grips the cuff end and swings it to the waistband "
        "(crosswise fold). Phase 3: lay down and press the crease. "
        "Requires a bimanual executor; returns an error on single-arm."
    ),
    params={
        "left_cuff_label": "detection label for left leg cuff (default 'left cuff')",
        "right_cuff_label": "detection label for right leg cuff (default 'right cuff')",
        "waistband_label": "detection label for waistband center (default 'waistband')",
        "lift_height": "lift height before swing (default 0.18m)",
        "place_height": "height above table to lay fold (default 0.01m)",
        "force": "gripper pinch force (default 15N)",
        "approach_height": "hover height above surface (default 0.08m)",
        "arc_steps": "waypoints per swing arc (default 10)",
        "press_crease": "run hem_press on final crease (default true)",
    },
)
def bimanual_pants_fold(executor, params: dict):
    """
    Bimanual two-phase pants fold with coordinated arm motion.
    """
    t0 = time.time()

    # guard: only runs on bimanual executor
    if not hasattr(executor, "left") or not hasattr(executor, "right"):
        return _result(
            "bimanual_pants_fold",
            False,
            "requires bimanual executor (two arms)",
            time.time() - t0,
        )

    left_label = params.get("left_cuff_label", "left cuff")
    right_label = params.get("right_cuff_label", "right cuff")
    waist_label = params.get("waistband_label", "waistband")

    left_pos, _ = _get_detection(executor, left_label)
    right_pos, _ = _get_detection(executor, right_label)
    waist_pos, _ = _get_detection(executor, waist_label)

    if left_pos is None:
        return _result(
            "bimanual_pants_fold", False, f"'{left_label}' not found", time.time() - t0
        )
    if right_pos is None:
        return _result(
            "bimanual_pants_fold", False, f"'{right_label}' not found", time.time() - t0
        )
    if waist_pos is None:
        return _result(
            "bimanual_pants_fold", False, f"'{waist_label}' not found", time.time() - t0
        )

    lift_h = float(params.get("lift_height", 0.18))
    place_h = float(params.get("place_height", 0.01))
    approach_h = float(params.get("approach_height", 0.08))
    force = float(params.get("force", 15.0))
    arc_steps = int(params.get("arc_steps", 10))
    press = bool(params.get("press_crease", True))

    logger.info(
        "[bimanual_pants_fold] L_cuff=%s R_cuff=%s waist=%s",
        left_pos.round(3),
        right_pos.round(3),
        waist_pos.round(3),
    )

    mid_xy = 0.5 * (left_pos[:2] + right_pos[:2])  # meeting point for fold 1
    meet_z = max(left_pos[2], right_pos[2]) + lift_h  # meeting z, lift_h above table

    # phase 1: both arms grip their respective cuff
    logger.info("[bimanual_pants_fold] phase 1: grip both cuffs")
    for arm_obj, cuff_pos, label in [
        (executor.left, left_pos, left_label),
        (executor.right, right_pos, right_label),
    ]:
        hover = cuff_pos.copy()
        hover[2] += approach_h
        arm_obj._move_to(hover, GRASP_ORIENT, velocity=0.10)

        pinch = cuff_pos.copy()
        pinch[2] = max(pinch[2] + 0.005, TABLE_Z_FLOOR)
        arm_obj._move_to(pinch, GRASP_ORIENT, velocity=0.04)

        arm_obj.robot.close_gripper(force=force)
        time.sleep(0.3)
        width = arm_obj.robot.get_gripper_width()
        if width > 0.03:
            logger.warning(
                "[bimanual_pants_fold] %s not gripped (width=%.3fm)", label, width
            )
            arm_obj.robot.open_gripper()
            return _result(
                "bimanual_pants_fold",
                False,
                f"failed to grip '{label}'",
                time.time() - t0,
            )

    # phase 1b: lift both cuffs together
    logger.info("[bimanual_pants_fold] phase 1b: lift cuffs")
    for arm_obj, cuff_pos in [
        (executor.left, left_pos),
        (executor.right, right_pos),
    ]:
        lifted = cuff_pos.copy()
        lifted[2] = meet_z
        arm_obj._move_to(lifted, GRASP_ORIENT, velocity=0.06)

    # phase 1c: swing both legs to meet at midpoint (lengthwise fold)
    logger.info("[bimanual_pants_fold] phase 1c: swing legs to meet")
    meet_pos = np.array([mid_xy[0], mid_xy[1], meet_z])
    for arm_obj in [executor.left, executor.right]:
        arm_obj._move_to(meet_pos, GRASP_ORIENT, velocity=0.05)
    time.sleep(0.3)

    # phase 1d: left releases, right holds the stacked cuffs
    logger.info("[bimanual_pants_fold] phase 1d: left releases")
    executor.left.robot.open_gripper()
    time.sleep(0.2)

    # retract left arm out of the way
    left_retract = meet_pos.copy()
    left_retract[2] += 0.10
    left_retract[0] -= 0.10
    executor.left._move_to(left_retract, GRASP_ORIENT, velocity=0.08)

    # phase 1e: right arm lowers the stacked cuffs to the table
    logger.info("[bimanual_pants_fold] phase 1e: lower stacked cuffs")
    lay1 = np.array([mid_xy[0], mid_xy[1], right_pos[2] + place_h])
    executor.right._move_to(lay1, GRASP_ORIENT, velocity=0.04)
    executor.right.robot.open_gripper()
    time.sleep(0.2)
    right_retract = lay1.copy()
    right_retract[2] += approach_h
    executor.right._move_to(right_retract, GRASP_ORIENT, velocity=0.08)

    # phase 2: crosswise fold - grip cuff stack, swing to waistband
    # use whichever arm is closer to the cuff stack
    logger.info("[bimanual_pants_fold] phase 2: crosswise fold")
    cuff_stack = np.array([mid_xy[0], mid_xy[1], right_pos[2]])

    # pick the arm closer to cuff_stack for the second fold
    dist_l = float(np.linalg.norm(left_retract[:2] - cuff_stack[:2]))
    dist_r = float(np.linalg.norm(right_retract[:2] - cuff_stack[:2]))
    fold_arm = executor.right if dist_r <= dist_l else executor.left

    hover2 = cuff_stack.copy()
    hover2[2] += approach_h
    fold_arm._move_to(hover2, GRASP_ORIENT, velocity=0.10)

    pinch2 = cuff_stack.copy()
    pinch2[2] = max(pinch2[2] + 0.005, TABLE_Z_FLOOR)
    fold_arm._move_to(pinch2, GRASP_ORIENT, velocity=0.04)

    fold_arm.robot.close_gripper(force=force)
    time.sleep(0.3)
    w2 = fold_arm.robot.get_gripper_width()
    if w2 > 0.03:
        logger.warning("[bimanual_pants_fold] cuff stack not gripped (%.3fm)", w2)
        fold_arm.robot.open_gripper()
        return _result(
            "bimanual_pants_fold",
            False,
            "failed to grip cuff stack for fold 2",
            time.time() - t0,
        )

    # arc swing cuffs to waistband
    lifted2 = cuff_stack.copy()
    lifted2[2] += lift_h
    fold_arm._move_to(lifted2, GRASP_ORIENT, velocity=0.06)

    end2 = np.array(
        [waist_pos[0], waist_pos[1], cuff_stack[2] + place_h + lift_h * 0.3]
    )
    wps = _arc_waypoints(lifted2, end2, peak_height=lift_h * 0.3, n_steps=arc_steps)
    for wp in wps[1:]:
        fold_arm._move_to(wp, GRASP_ORIENT, velocity=0.06)

    lay2 = np.array([waist_pos[0], waist_pos[1], cuff_stack[2] + place_h])
    fold_arm._move_to(lay2, GRASP_ORIENT, velocity=0.03)

    fold_arm.robot.open_gripper()
    time.sleep(0.2)
    retract_final = lay2.copy()
    retract_final[2] += approach_h
    fold_arm._move_to(retract_final, GRASP_ORIENT, velocity=0.08)

    # phase 3: optional crease press
    if press:
        mid2 = 0.5 * (cuff_stack[:2] + waist_pos[:2])
        fold_dir = waist_pos[:2] - cuff_stack[:2]
        fold_len = float(np.linalg.norm(fold_dir))
        if fold_len > 0.02:
            fold_dir_n = fold_dir / fold_len
            perp = np.array([-fold_dir_n[1], fold_dir_n[0]])
            crease_half = fold_len * 0.4
            crease_start = np.array(
                [
                    mid2[0] - perp[0] * crease_half,
                    mid2[1] - perp[1] * crease_half,
                    TABLE_Z_FLOOR + 0.005,
                ]
            )
            crease_end = np.array(
                [
                    mid2[0] + perp[0] * crease_half,
                    mid2[1] + perp[1] * crease_half,
                    TABLE_Z_FLOOR + 0.005,
                ]
            )
            logger.info("[bimanual_pants_fold] pressing final crease")
            hem_press(
                executor,
                {
                    "start_pos": crease_start.tolist(),
                    "end_pos": crease_end.tolist(),
                    "passes": 1,
                    "force": 10.0,
                },
            )

    return _result(
        "bimanual_pants_fold",
        True,
        f"bimanual pants fold complete: lengthwise then crosswise "
        f"({left_label}+{right_label} -> {waist_label})",
        time.time() - t0,
    )

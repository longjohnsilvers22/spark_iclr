#!/usr/bin/env python3
"""Bimanual t-shirt fold (FrankaBambooDriver + per-step PyRoki IK).

No raw bamboo calls. Uses FrankaBambooDriver.move_to_joint_config() which
handles duration/velocity internally. Every arc step gets fresh IK from
current config as seed. Dry-run validates all IK solutions before moving.

Frame rules:
  - detect.py swaps SAM3 wearer-perspective to camera-frame naming
  - "left sleeve" = left of camera = FR3 (RIGHT robot) side
  - RIGHT robot (FR3): RIGHT frame coords
  - LEFT robot (Panda): LEFT frame coords

Pipeline (SAM3 masks only; no human annotation):
  Phase 1: RIGHT folds left sleeve inward -> lift -> release -> retract
  Phase 2: LEFT folds right sleeve inward -> lift -> release -> retract
  Phase 3: Both move to the shirt's TOP CORNERS (collar edges) + grasp + lift
  Phase 4: Arc-fold the top corners down over the body toward the hem
  Phase 5: Release + home

The body fold grasps the COLLAR/top corners (the reachable, well-conditioned
end of the shirt) rather than the hem: the hem sits at low x near the bases
where the FR3 contorts into a near-singular config. detect.py computes the
collar/top-corner grab points for any "shirt" mask automatically.

Usage:
    cd ~/spark/src
    export JAX_PLATFORMS=cpu
    PY=$(conda run -n spark_conda which python3)

    $PY scripts/fold_tshirt_v3.py                    # full pipeline
    $PY scripts/fold_tshirt_v3.py --phase sleeve     # sleeves only
    $PY scripts/fold_tshirt_v3.py --phase hem        # hem only
    $PY scripts/fold_tshirt_v3.py --skip-detect      # use last detection
    $PY scripts/fold_tshirt_v3.py --dry-run          # IK validation only
"""
import argparse
import signal
import sys, os, json, time, threading, subprocess
import numpy as np
from scipy.spatial.transform import Rotation

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
os.environ.setdefault("JAX_PLATFORMS", "cpu")

from spark_real.config import load_family_yaml
from spark_real.robots.franka.franka_bamboo_driver import FrankaBambooDriver

_frame_i = [0]


def capture_frame(label, enabled):
    """Grab a ZED RGB frame to /tmp/fold_frames so the fold can be inspected
    after the fact. Behind --frames. Read-only, no motion."""
    if not enabled:
        return
    try:
        import cv2
        from spark_real.perception.zed import ZEDMiniCamera
        os.makedirs("/tmp/fold_frames", exist_ok=True)
        cam = ZEDMiniCamera(width=1280, height=720, depth_mode="NEURAL")
        cam.open(); time.sleep(0.3)
        rgb, _ = cam.read(depth=True)
        cam.close()
        if rgb is not None:
            _frame_i[0] += 1
            path = f"/tmp/fold_frames/{_frame_i[0]:02d}_{label}.png"
            cv2.imwrite(path, cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR))
            print(f"  [frame] {path}")
    except Exception as exc:
        print(f"  [frame] failed: {exc}")

# --- Constants ---
HOME_Q = np.array([0.0, -0.785, 0.0, -2.356, 0.0, 1.571, 0.785])
PYROKI_Z_OFFSET = 0.103
R_TD = np.array([[1, 0, 0], [0, -1, 0], [0, 0, -1]], dtype=float)


def _wxyz(R):
    q = Rotation.from_matrix(R).as_quat()
    return np.array([float(q[3]), float(q[0]), float(q[1]), float(q[2])])


# Per-arm jaw yaw. Both flanges solve to the same top-down orientation in
# their own base frames (dev < 1.5 deg), yet the LEFT jaw line is ~90 deg
# rotated from the right one on the table (the bases / tool-plate mounts are
# rotated relative to each other), so the LEFT target carries a yaw offset
# about vertical. Parallel jaws are symmetric under 180 deg; the sign is
# irrelevant.
LEFT_YAW_OFFSET_DEG = 90.0
ORIENT_WXYZ = _wxyz(R_TD)  # right arm
ORIENT_WXYZ_LEFT = _wxyz(
    Rotation.from_euler("z", LEFT_YAW_OFFSET_DEG, degrees=True).as_matrix() @ R_TD)


def _orient_for(arm):
    return ORIENT_WXYZ_LEFT if arm == "left" else ORIENT_WXYZ


def sleeve_orient_for(arm, yaw_rad):
    """Top-down orientation with the jaw line rotated by ``yaw_rad`` about
    vertical (on top of the per-arm base orientation), so the closing
    direction can be aligned with the sleeve axis (tip -> shirt body).

    Branch canonicalization: the yaw comes from a detected OBB axis, which
    is 180-deg ambiguous. Parallel jaws don't care, but the wrist camera
    must face backward toward the robots, and the jaw center carries an x
    offset from the flange, so the wrong branch lands the grasp ~2x that
    offset off in x. Canonical branch per arm:
      right: sin(yaw) <= 0   (validated hem -90.3 deg, sleeve -161 deg)
      left:  sin(yaw) >= 0   (validated hem +89.7 deg, sleeve +157 deg)
    Idempotent: callers that already add the +180 left hem flip are
    re-normalized to the same branch."""
    yaw = float(np.arctan2(np.sin(yaw_rad), np.cos(yaw_rad)))
    if arm == "left":
        if np.sin(yaw) < 0.0:
            yaw += np.pi
        base = Rotation.from_euler(
            "z", LEFT_YAW_OFFSET_DEG, degrees=True).as_matrix() @ R_TD
    else:
        if np.sin(yaw) > 0.0:
            yaw -= np.pi
        base = R_TD
    return _wxyz(Rotation.from_euler("z", yaw).as_matrix() @ base)
JOINT_LIMITS = {
    "left": np.array([[-2.8973, 2.8973], [-1.7628, 1.7628], [-2.8973, 2.8973],
                      [-3.0718, -0.0698], [-2.8973, 2.8973], [-0.0175, 3.7525],
                      [-2.8973, 2.8973]]),
    "right": np.array([[-2.74, 2.74], [-1.78, 1.78], [-2.90, 2.90],
                       [-3.04, -0.15], [-2.80, 2.80], [0.54, 4.52],
                       [-3.01, 3.01]]),
}


def limit_margin_deg(arm, q):
    """
    Smallest distance (deg) of a joint config from its nearest limit.
    """
    lims = JOINT_LIMITS[arm]
    marg = np.minimum(np.asarray(q)[:7] - lims[:, 0],
                      lims[:, 1] - np.asarray(q)[:7])
    return float(np.degrees(marg.min()))


def find_hem_orient(arm, x, y, base_yaw_rad, hover_z=0.05,
                    min_margin_deg=1.0):
    """Search jaw-yaw nudges around the detected hem axis for one whose
    full IK chain (hover + grasp depth) is in-limits with real margin.

    The detected yaw can sit exactly on a joint-limit boundary (zero
    in-limits solutions at the detected yaw while +8 deg had 2.8 deg of
    margin). Parallel jaws pinch a hem corner identically under a few
    degrees of yaw, so plan the yaw at execution time instead of reflexing
    mid-descend.

    Returns (orient_wxyz, nudge_deg, q_hover) or (None, None, None).
    """
    for d in (0, 2, -2, 4, -4, 6, -6, 8, -8, 10, -10, 12, -12, 16, -16,
              20, -20):
        o = sleeve_orient_for(arm, base_yaw_rad + np.radians(d))
        qh = ik(arm, [x, y, hover_z], HOME_Q, orient_wxyz=o)
        if qh is None:
            continue
        zoff = PYROKI_Z_OFFSET + (_LEFT_Z_CAL if arm == "left" else 0.0)
        ph = fk_pos(arm, qh)
        if (np.linalg.norm(ph - np.array([x, y, hover_z + zoff]))
                > FK_REACH_TOL or limit_margin_deg(arm, qh) < min_margin_deg):
            continue
        qg = ik(arm, [x, y, GRASP_Z], qh, orient_wxyz=o)
        if qg is None:
            continue
        pg = fk_pos(arm, qg)
        if (np.linalg.norm(pg - np.array([x, y, GRASP_Z + zoff]))
                > FK_REACH_TOL or limit_margin_deg(arm, qg) < min_margin_deg):
            continue
        return o, d, qh
    return None, None, None


ARC_PEAK = 0.040
LIFT_Z = 0.130
# Sleeve-phase lift, kept lower than the hem lift: the full 130mm raise
# hoists too much shirt during the sleeve folds; 60mm gives a clean double fold.
LIFT_Z_SLEEVE = 0.060
LAND_Z = 0.025
# Workspace-aware body fold. The left (Panda) arm is well-conditioned at the
# far collar only when (a) the lateral offset is bounded (a y of 0.29 forced
# a poor reach branch with 33mm FK error mid-drape; capping |y| at 0.22 drops
# the whole continuous drape to <2mm) and (b) the drape is lifted into the
# dexterous part of the envelope rather than dragged low. Hence the cap on
# grasp lateral offset (COLLAR_MAX_ABS_Y) and the lifted drape.
SLEEVE_Y_OFFSET_R = -0.015
HEM_X_L = 0.010
HEM_X_R = 0.020
HEM_Y = 0.020
HEM_Y_OFFSET_R = -0.020  # RIGHT robot hem: -20mm y toward edge
HEM_Z_FLOOR = -0.020
GRASP_Z = -0.035  # descend to the cloth on the table (~-0.035), not above it

# Body fold targets the COLLAR, not the hem. The hem sits at low x (close to
# the bases) where the FR3 (right) arm contorts into a near-base config and
# reflexes. The body fold grasps the collar (the far, extended-posture end)
# and drapes the top half toward the hem, with the landing x clamped to a
# reflex-safe band. The dry-run refuses any right-arm waypoint below
# FR3_REFLEX_MIN_X or with FK error above FK_REACH_TOL.
FR3_REFLEX_MIN_X = 0.32     # right-arm grasps below this x trip the reflex
COLLAR_Y_INSET = 0.020      # inset from the collar edge toward the centerline
COLLAR_MAX_ABS_Y = 0.22     # cap collar grasp |y|: beyond this the arm reaches
# to the side into a poorly-conditioned branch (validated: 0.29 -> 33mm mid
# drape, 0.22 -> <2mm across the whole drape).
FOLD_LAND_X = 0.46          # drape landing x (toward hem, clamped reflex-safe)
FK_REACH_TOL = 0.030        # FK position error above this means unreachable
GRASP_FORCE = 60
# Joint-space speeds (rad/s). The bamboo driver computes an acceleration-limited
# duration (T >= sqrt(5.77*D/A_MAX), A_MAX = 1.5 rad/s^2), which caps the
# min-jerk torque spike that trips libfranka's estimated-wrench cartesian_reflex
# on short moves. Big transits are velocity-bound and quick; small arc steps are
# accel-bound and gentle by construction. APPROACH_VEL stays low: the descend
# ends in cloth contact, so gentle approach is about contact force.
MOVE_VEL = 0.22
ARC_VEL = 0.16
APPROACH_VEL = 0.08

DET_PATH = os.path.expanduser("~/.spark_real/detections/detections.json")

# --- IK helper (BimanualPyrokiPlanner for per-arm URDF + base offsets) ---
_planner = None
def _get_planner():
    global _planner
    if _planner is None:
        from spark_real.control.pyroki_planner import BimanualPyrokiPlanner
        _planner = BimanualPyrokiPlanner()
    return _planner

# Left cross-arm calibration carries a tilt: the same physical point reads
# 15-45mm higher in the left frame than the right (right hand-eye is the
# trusted one). Commanding the left arm to a z in its own frame therefore
# lands it physically lower by the local discrepancy. set_left_z_correction()
# installs the locally measured dz (left z - right z at the grasp point) and
# ik() adds it to every left z target, so both arms land at the same height.
_LEFT_Z_CAL = 0.0


def set_left_z_correction(dz):
    global _LEFT_Z_CAL
    _LEFT_Z_CAL = float(dz)
    print(f"  LEFT z calibration correction: {_LEFT_Z_CAL*1000:+.1f}mm")


def ik(arm, pos, q_seed, orient_wxyz=None):
    """PyRoki IK via BimanualPyrokiPlanner. Returns joint config or None.
    orient_wxyz overrides the per-arm default top-down orientation (used for
    sleeve-aligned jaw yaw)."""
    t = np.array(pos, dtype=float).copy()
    t[2] += PYROKI_Z_OFFSET
    if arm == "left":
        t[2] += _LEFT_Z_CAL
    if orient_wxyz is None:
        orient_wxyz = _orient_for(arm)
    try:
        q = _get_planner().solve(arm=arm, target_position_base=t,
                                  target_wxyz_base=orient_wxyz, prev_cfg=q_seed)[:7]
        return q
    except Exception:
        return None


def fk_pos(arm, q):
    """Forward-kinematics TCP position for a 7-joint config, base frame.

    pyroki returns a best-effort IK config even for unreachable targets, so
    callers compare this against the requested point to catch "solved but
    far." Returns None on failure. The returned point is in the same frame
    the ik() target uses (PYROKI_Z_OFFSET already baked into ik targets, so
    compare against pos with that offset added)."""
    if q is None:
        return None
    try:
        pl = _get_planner()
        robot = pl.left_robot if arm == "left" else pl.right_robot
        link = pl.left_target_link if arm == "left" else pl.right_target_link
        n_act = robot.joints.num_actuated_joints
        cfg = np.zeros(n_act)
        cfg[:min(len(q), n_act)] = np.asarray(q)[:n_act]
        if n_act > 7:
            cfg[7:] = 0.02
        Ts = np.asarray(robot.forward_kinematics(cfg))
        idx = robot.links.names.index(link)
        return np.asarray(Ts[idx][-3:], dtype=float)
    except Exception:
        return None


def reach_err(arm, pos, q_seed=HOME_Q):
    """
    FK position error (m) of the IK solution for pos, or None/inf.
    """
    q = ik(arm, pos.tolist() if hasattr(pos, "tolist") else list(pos), q_seed)
    p = fk_pos(arm, q)
    if p is None:
        return float("inf")
    t = np.asarray(pos, dtype=float).copy()
    t[2] += PYROKI_Z_OFFSET
    return float(np.linalg.norm(p - t))


def reachable(arm, pos, q_seed=HOME_Q, tol=None):
    """True if ik() reaches pos within tol (default FK_REACH_TOL), and (right
    arm) pos is clear of the FR3 near-base reflex zone."""
    pos = np.asarray(pos, dtype=float)
    if arm == "right" and pos[0] < FR3_REFLEX_MIN_X:
        return False
    return reach_err(arm, pos, q_seed) <= (FK_REACH_TOL if tol is None else tol)


# A grasp must land on the cloth, so it needs much tighter accuracy than a
# drape waypoint: a top-down IK that solves 25-30mm short still "reaches" by
# the loose gate but the gripper closes off the cloth. Pull the collar grasp
# to where the arm is accurate to within this.
GRASP_ACCURATE_TOL = 0.013


def clamp_collar_grab(arm, pos, q_seed=HOME_Q):
    """Pull a collar grab point to where the arm lands ACCURATELY (within
    GRASP_ACCURATE_TOL), easing y toward the centerline and x toward the base.
    This trades grabbing the exact far collar corner for a grab that actually
    closes on the cloth. Returns the adjusted point, or None if it cannot be
    made accurate."""
    pos = np.asarray(pos, dtype=float).copy()
    if abs(pos[1]) > COLLAR_MAX_ABS_Y:
        pos[1] = np.sign(pos[1]) * COLLAR_MAX_ABS_Y
    x_floor = (FR3_REFLEX_MIN_X + 0.06) if arm == "right" else 0.40
    for _ in range(14):
        if reachable(arm, pos, q_seed, tol=GRASP_ACCURATE_TOL):
            return pos
        pos[1] *= 0.88                      # ease toward centerline
        pos[0] = max(pos[0] - 0.02, x_floor)  # ease toward the base
    return pos if reachable(arm, pos, q_seed, tol=GRASP_ACCURATE_TOL) else None


def clamp_sleeve_grab(arm, gx, gy, q_seed=HOME_Q):
    """Ease a SAM3 sleeve grab (gx, gy) toward the base until the arm reaches it
    within FK_REACH_TOL at BOTH the approach (z=0.05) and grasp (GRASP_Z)
    heights. SAM3 picks the sleeve mask's interior point, which for a far
    sleeve can land past the arm's accurate envelope (a 76mm miss on the
    right sleeve). Returns the adjusted (gx, gy); unchanged when already
    reachable."""
    x, y = float(gx), float(gy)
    x_floor = (FR3_REFLEX_MIN_X + 0.06) if arm == "right" else 0.42

    def worst(xx, yy):
        return max(reach_err(arm, [xx, yy, 0.05], q_seed),
                   reach_err(arm, [xx, yy, GRASP_Z], q_seed))

    for _ in range(16):
        if worst(x, y) <= FK_REACH_TOL:
            return x, y
        x = max(x - 0.02, x_floor)   # ease toward the base
        y *= 0.92                    # ease toward the centerline
    return x, y

# --- Gripper helpers ---
def grippers(cfg):
    from spark_real.robots.bimanual_franka.bimanual_franka_driver import make_dual_gripper
    dual = make_dual_gripper(cfg.get("grippers", {}))
    dual.connect()
    time.sleep(0.5)
    return dual

def grip_raw(grip, arm):
    """Firmware position counts via a freshness-gated read.

    Returns None when no live status frame is arriving (CAN bus down):
    the stale cache must NOT be used as a fallback, since a stale 0
    is indistinguishable from 'never closed'."""
    g = grip.for_arm(arm)
    if hasattr(g, "raw_position_fresh"):
        return g.raw_position_fresh()
    return getattr(g._motor, "gripper_position", None)

def grip_open(grip, arms=("left", "right")):
    for a in arms:
        grip.for_arm(a).send_pack_locked(0, 255, 500, 1, 1, 0, 0)
    time.sleep(0.6)

def grip_close(grip, arms=("left", "right")):
    for a in arms:
        grip.for_arm(a).send_pack_locked(255, 255, 1300, 1, 1, 0, 0)
    time.sleep(1.2)
    # Verify-and-retry: arm motion can knock the CAN bus out (bus-off,
    # auto-recovered by the driver watchdog) right as the close is sent,
    # so a jaw still near open (or silent telemetry) gets one re-send.
    for a in arms:
        raw = grip_raw(grip, a)
        if raw is None or raw < 100:
            print(f"    {a}: raw={raw} after close, re-sending", flush=True)
            time.sleep(0.5)  # give the bus watchdog time to reinit
            grip.for_arm(a).send_pack_locked(255, 255, 1300, 1, 1, 0, 0)
            time.sleep(1.2)
    for a in arms:
        print(f"    {a}: raw={grip_raw(grip, a)}")

def grip_check(grip):
    for a in ["left", "right"]:
        raw = grip_raw(grip, a)
        state = "CLOSED" if raw and raw > 100 else "OPEN"
        print(f"    {a}: raw={raw} [{state}]")

# --- Dry-run validation ---
def _wp_reach_ok(arm, pos, qs):
    """FK + reflex-zone gate for one dry-run waypoint. pyroki returns a
    best-effort config even when a target is unreachable, so IK-not-None is
    not enough: verify FK actually reaches it, and keep the right arm clear
    of the FR3 near-base reflex zone. Returns (ok, reason)."""
    pos = np.asarray(pos, dtype=float)
    if arm == "right" and pos[0] < FR3_REFLEX_MIN_X:
        return False, (f"x {pos[0]:.3f} < FR3 reflex-zone limit "
                       f"{FR3_REFLEX_MIN_X:.2f} (near-base, flashes red)")
    p = fk_pos(arm, qs)
    if p is None:
        return False, "FK failed"
    t = pos.copy()
    t[2] += PYROKI_Z_OFFSET
    err = float(np.linalg.norm(p - t))
    if err > FK_REACH_TOL:
        return False, (f"FK error {err*1000:.0f}mm > {FK_REACH_TOL*1000:.0f}mm "
                       f"(IK solved but does not reach)")
    return True, ""


def dry_run_linear_transit(label, arm, start_xyz, goal_xyz, q_seed, n_steps=8, max_delta=0.8):
    """
    Validate linearly-interpolated IK transit. Returns True if all pass.
    """
    q = q_seed.copy()
    start = np.array(start_xyz)
    goal = np.array(goal_xyz)
    for i in range(1, n_steps + 1):
        t = i / n_steps
        pos = start + t * (goal - start)
        qs = ik(arm, pos.tolist(), q)
        if qs is None:
            print(f"  {label} step {i}/{n_steps}: IK FAIL at "
                  f"({pos[0]:.3f},{pos[1]:.3f},{pos[2]:.3f})")
            return False
        ok, why = _wp_reach_ok(arm, pos, qs)
        if not ok:
            print(f"  {label} step {i}/{n_steps}: {why} at "
                  f"({pos[0]:.3f},{pos[1]:.3f},{pos[2]:.3f})")
            return False
        delta = float(np.max(np.abs(qs - q[:7])))
        if delta > max_delta:
            print(f"  {label} step {i}/{n_steps}: delta {delta:.3f} > {max_delta} at "
                  f"({pos[0]:.3f},{pos[1]:.3f},{pos[2]:.3f})")
            return False
        q = qs
    print(f"  {label}: all {n_steps} transit IK OK (max per-step < {max_delta})")
    return True


def dry_run_arc(label, arm, gx, gy, fx, fy, q_seed, keep_x=True, n_arc=12):
    """
    Validate all IK solutions for a grasp+fold arc. Returns True if all pass.
    """
    q = q_seed.copy()
    wps = [
        ("approach", [gx, gy, 0.05]),
        ("grasp", [gx, gy, GRASP_Z]),
        ("lift", [gx, gy, GRASP_Z + LIFT_Z]),
    ]
    lift = GRASP_Z + LIFT_Z
    for i in range(1, n_arc + 1):
        t = i / n_arc
        ax = gx if keep_x else gx + t * (fx - gx)
        ay = gy + t * (fy - gy)
        az = lift + ARC_PEAK * np.sin(t * np.pi) if i < n_arc else LAND_Z
        wps.append((f"arc{i}", [ax, ay, az]))
    wps.append(("retract", [wps[-1][1][0], wps[-1][1][1], 0.08]))

    for name, pos in wps:
        qs = ik(arm, pos, q)
        if qs is None:
            print(f"  {label} {name}: IK FAIL at ({pos[0]:.3f},{pos[1]:.3f},{pos[2]:.3f})")
            return False
        ok, why = _wp_reach_ok(arm, pos, qs)
        if not ok:
            print(f"  {label} {name}: {why} at ({pos[0]:.3f},{pos[1]:.3f},{pos[2]:.3f})")
            return False
        q = qs
    print(f"  {label}: all IK OK (FK-verified, reflex-safe)")
    return True

# --- Single-arm grasp + fold arc ---
def pinch_drape(robot, grip, arm, label, gx, gy, fx, fy, keep_x=True, n_arc=8,
                orient_wxyz=None):
    """Pinch fabric at (gx,gy), arch-drape to (fx,fy), release, retract.
    orient_wxyz: jaw orientation for the whole phase (e.g. the sleeve's OBB
    minor-axis yaw); None keeps the per-arm default top-down orientation."""
    print(f"\n--- {label}: pinch -> arch -> drape ---")
    q = np.array(robot.get_joint_positions())

    # Approach: IK to above grasp point
    qs = ik(arm, [gx, gy, 0.05], q, orient_wxyz=orient_wxyz)
    if qs is None:
        print(f"  APPROACH IK FAIL at ({gx:.3f},{gy:.3f},0.050)")
        return False
    robot.move_to_joint_config(qs.tolist(), velocity=MOVE_VEL)
    q = qs
    time.sleep(0.2)

    # Descend to grasp z
    qs = ik(arm, [gx, gy, GRASP_Z], q, orient_wxyz=orient_wxyz)
    if qs is None:
        print(f"  DESCEND IK FAIL at ({gx:.3f},{gy:.3f},{GRASP_Z:.3f})")
        return False
    robot.move_to_joint_config(qs.tolist(), velocity=APPROACH_VEL)
    q = qs
    time.sleep(0.3)

    # Grasp
    grip_close(grip, arms=(arm,))
    tcp = robot.get_tcp_pose()
    print(f"  pinched at ({tcp[0]:.3f},{tcp[1]:.3f},{tcp[2]:.3f})")

    # Lift
    lift_z = GRASP_Z + LIFT_Z
    qs = ik(arm, [gx, gy, lift_z], q, orient_wxyz=orient_wxyz)
    if qs is None:
        print(f"  LIFT IK FAIL at ({gx:.3f},{gy:.3f},{lift_z:.3f})")
        return False
    robot.move_to_joint_config(qs.tolist(), velocity=MOVE_VEL)
    q = qs
    time.sleep(0.2)

    # Arc fold - stay at lift height (no descent), let fabric drape by gravity
    q = np.array(robot.get_joint_positions())
    for i in range(1, n_arc + 1):
        t = i / n_arc
        ax = gx if keep_x else gx + t * (fx - gx)
        ay = gy + t * (fy - gy)
        az = lift_z + ARC_PEAK * np.sin(t * np.pi)
        qs = ik(arm, [ax, ay, az], q, orient_wxyz=orient_wxyz)
        if qs is None:
            print(f"  ARC IK FAIL at step {i}/{n_arc} ({ax:.3f},{ay:.3f},{az:.3f})")
            return False
        try:
            robot.move_to_joint_config(qs.tolist(), velocity=ARC_VEL)
            q = qs
        except RuntimeError:
            print(f"  ARC step {i}/{n_arc} bamboo fail (skipping)")
            q = np.array(robot.get_joint_positions())
        time.sleep(0.02)

    tcp = robot.get_tcp_pose()
    print(f"  drape end at ({tcp[0]:.3f},{tcp[1]:.3f},{tcp[2]:.3f})")

    # Lift before release
    qs = ik(arm, [tcp[0], tcp[1], tcp[2] + 0.08], q, orient_wxyz=orient_wxyz)
    if qs is not None:
        robot.move_to_joint_config(qs.tolist(), velocity=MOVE_VEL)
        q = qs

    # Release
    grip_open(grip, arms=(arm,))
    time.sleep(0.3)

    # Retract up
    tcp = robot.get_tcp_pose()
    qs = ik(arm, [tcp[0], tcp[1], 0.10], q, orient_wxyz=orient_wxyz)
    if qs is not None:
        robot.move_to_joint_config(qs.tolist(), velocity=MOVE_VEL)
    time.sleep(0.2)
    return True

# --- Bimanual hem fold arc ---
def hem_fold(l_robot, r_robot, l_start, r_start, l_end, r_end, n_arc=10):
    """
    Both arms fold hem to shirt top simultaneously.
    """
    print(f"\n--- Hem fold ---")
    lq = np.array(l_robot.get_joint_positions())
    rq = np.array(r_robot.get_joint_positions())
    lift_z = max(l_start[2], r_start[2]) + LIFT_Z

    for i in range(1, n_arc + 1):
        t = i / n_arc
        dz = ARC_PEAK * 1.5 * np.sin(t * np.pi) if i < n_arc else LAND_Z - lift_z
        lx = l_start[0] + t * (l_end[0] - l_start[0])
        ly = l_start[1] + t * (l_end[1] - l_start[1])
        rx = r_start[0] + t * (r_end[0] - r_start[0])
        ry = r_start[1] + t * (r_end[1] - r_start[1])

        lqs = ik("left", [lx, ly, lift_z + dz], lq)
        rqs = ik("right", [rx, ry, lift_z + dz], rq)

        if lqs is not None and rqs is not None:
            def ml(): l_robot.move_to_joint_config(lqs.tolist(), velocity=ARC_VEL)
            def mr(): r_robot.move_to_joint_config(rqs.tolist(), velocity=ARC_VEL)
            tl = threading.Thread(target=ml); tr = threading.Thread(target=mr)
            tl.start(); tr.start(); tl.join(); tr.join()
            lq = lqs; rq = rqs
        time.sleep(0.02)

    lt = l_robot.get_tcp_pose(); rt = r_robot.get_tcp_pose()
    print(f"  L end: ({lt[0]:.3f},{lt[1]:.3f},{lt[2]:.3f})")
    print(f"  R end: ({rt[0]:.3f},{rt[1]:.3f},{rt[2]:.3f})")
    return True

# --- Detection ---
def load_detections():
    det = json.loads(open(DET_PATH).read())
    d = {e["label"]: e for e in det}
    # The rollout runner prompts "garment" (more reliable than "shirt" on a
    # part-folded cloth); treat it as the shirt body.
    if "shirt" not in d and "garment" in d:
        d["shirt"] = d["garment"]
    # The body fold grabs the COLLAR (from the shirt mask's top band), so the
    # hem detection is optional: useful context, not a gate.
    for needed in ["left sleeve", "right sleeve", "shirt"]:
        if needed not in d:
            print(f"  MISSING: {needed}. Found: {list(d.keys())}")
            return None
    if "hem" not in d:
        print("  (no hem detection; collar-based fold does not need it)")
    return d

# --- Main ---
def main():
    p = argparse.ArgumentParser()
    p.add_argument("--phase", choices=["sleeve", "hem", "all"], default="all",
                   help="sleeve = both sleeve folds; hem = the top-corner "
                        "(collar) body fold; all = the full fold.")
    p.add_argument("--skip-detect", action="store_true")
    p.add_argument("--dry-run", action="store_true")
    p.add_argument("--no-record", action="store_true", help="Skip recording")
    p.add_argument("--skip-dry-run", action="store_true", help="Skip IK dry run (use with known-good coords)")
    p.add_argument("--frames", action="store_true", help="Capture ZED frames at checkpoints to /tmp/fold_frames")
    p.add_argument("--machine", default="ANON-LAB",
                   help="configs/machines/<machine>.yaml overlay (arm IPs, grippers)")
    args = p.parse_args()

    do_sleeves = args.phase in ("sleeve", "all")
    do_hem = args.phase in ("hem", "all")

    # Grippers are connected AFTER the dry-run gate so a --dry-run is fully
    # hardware-free (no CAN actuation, no FCI). grip stays None until then.
    grip = None

    # Detect
    if not args.skip_detect:
        print("\n=== DETECT ===")
        subprocess.run([sys.executable, "scripts/detect.py",
                        "left sleeve", "right sleeve", "hem", "shirt",
                        "--hem-edges", "--annotate"], capture_output=True, text=True)

    d = load_detections()
    if d is None:
        if grip is not None:
            grip.disconnect()
        return

    # Parse targets. Sleeve grasps prefer the OUTER-EDGE (tip) point detect.py
    # computes from the mask band farthest from the shirt body; the centroid
    # is only the fallback (it pinches mid-sleeve, which folds poorly).
    _ls, _rs = d["left sleeve"], d["right sleeve"]
    ls_r = np.array(_ls.get("outer_edge_right") or _ls["right_frame"])
    ls_l = np.array(_ls.get("outer_edge_left") or _ls["left_frame"])
    rs_r = np.array(_rs.get("outer_edge_right") or _rs["right_frame"])
    rs_l = np.array(_rs.get("outer_edge_left") or _rs["left_frame"])
    if "outer_edge_right" in _ls and "outer_edge_left" in _rs:
        print("  Sleeve grasps: OUTER EDGE (tip)")
    else:
        print("  Sleeve grasps: centroid fallback (no outer_edge in detection)")
    shirt = d["shirt"]
    # Local left-frame z correction, measured at the LEFT arm's own grasp
    # point (the right sleeve) from its dual-frame coordinates.
    set_left_z_correction(rs_l[2] - rs_r[2])
    # Jaw yaw for the sleeve phase: the sleeve OBB minor axis from detect.py
    # (user-validated). None falls back to the per-arm default orientation.
    _yaw_r = _ls.get("jaw_yaw_right")
    _yaw_l = _rs.get("jaw_yaw_left")
    orient_sleeve_r = sleeve_orient_for("right", float(_yaw_r)) if _yaw_r is not None else None
    orient_sleeve_l = sleeve_orient_for("left", float(_yaw_l)) if _yaw_l is not None else None
    if _yaw_r is not None:
        print(f"  Sleeve jaw yaw (OBB minor axis): R {np.degrees(_yaw_r):+.0f} deg"
              f"  L {np.degrees(_yaw_l):+.0f} deg")

    # SAM3-only: ignore the grasp_annotator app (keypoints_annotated.json) and
    # use the SAM3 mask detections for everything: sleeves here, collar/top
    # corners below (annotated_path=None forces the detections path).
    print("  Using SAM3 detections (annotated keypoints ignored)")
    # Pull each SAM3 sleeve grab into the grabbing arm's accurate reach cone.
    # LEFT (Panda) grabs the right sleeve (rs_l); RIGHT (FR3) grabs the left
    # sleeve (ls_r). A no-op when the point is already reachable.
    rs_l[0], rs_l[1] = clamp_sleeve_grab("left", rs_l[0], rs_l[1])
    ls_r[0], ls_r[1] = clamp_sleeve_grab("right", ls_r[0], ls_r[1])
    center_y_r = (ls_r[1] + rs_r[1]) / 2
    center_y_l = (ls_l[1] + rs_l[1]) / 2
    # Drape the sleeve 80% to center, not all the way: the exact midpoint sits
    # at the workspace edge for a wide shirt. 80% still folds it over the body.
    center_y_r = ls_r[1] + 0.80 * (center_y_r - ls_r[1])
    center_y_l = rs_l[1] + 0.80 * (center_y_l - rs_l[1])

    # RIGHT robot sleeve target with -15mm y offset (sleeve phase unchanged)
    r_sleeve_grab = ls_r.copy()
    r_sleeve_grab[1] += SLEEVE_Y_OFFSET_R

    # Body fold grabs the COLLAR (not the hem). Use the ReKep-style keypoint
    # loader, which prefers human annotation (keypoints_annotated.json) and
    # falls back to detect.py collar edges / centroid. RIGHT arm grabs
    # collar_left (its right_frame), LEFT arm grabs collar_right (its
    # left_frame). Each grab is inset toward center and clamped into the
    # arm's reachable cone.
    from scripts.fold_keypoints import load_fold_keypoints
    kps = load_fold_keypoints(annotated_path=None)  # SAM3 only, no annotation
    by_name = kps.get("by_name", {})
    print(f"  Keypoints source: {kps.get('source')}")

    def _collar_grab(name, frame_key, centroid_key):
        kp = by_name.get(name) or {}
        p = kp.get(frame_key)
        if p is None:
            p = shirt.get(centroid_key)
        p = np.array(p, dtype=float)
        p[1] -= np.sign(p[1]) * COLLAR_Y_INSET  # inset toward centerline
        p[2] = max(p[2], HEM_Z_FLOOR)
        return p

    r_collar = _collar_grab("collar_left", "right_frame", "shirt_top_right")
    l_collar = _collar_grab("collar_right", "left_frame", "shirt_top_left")
    r_hem = clamp_collar_grab("right", r_collar)
    l_hem = clamp_collar_grab("left", l_collar)
    if r_hem is None or l_hem is None:
        print("  COLLAR grab unreachable even after clamp "
              f"(R={r_hem}, L={l_hem}). Run scripts/annotate_keypoints.py to "
              "hand-pick reachable collar points, then retry.")
        if grip is not None:
            grip.disconnect()
        return
    # Fold landing: drape toward the hem, x clamped reflex-safe so the FR3
    # never re-enters the near-base zone. y holds (fold runs along x).
    shirt_top_r = np.array([FOLD_LAND_X, r_hem[1], r_hem[2]])
    shirt_top_l = np.array([FOLD_LAND_X, l_hem[1], l_hem[2]])

    print(f"\n  LEFT  -> right sleeve LEFT:  ({rs_l[0]:.3f}, {rs_l[1]:.3f}, {rs_l[2]:.3f})")
    print(f"  RIGHT -> left sleeve RIGHT:  ({r_sleeve_grab[0]:.3f}, {r_sleeve_grab[1]:.3f}, {r_sleeve_grab[2]:.3f})")
    print(f"  RIGHT collar grab: ({r_hem[0]:.3f}, {r_hem[1]:.3f}, {r_hem[2]:.3f})")
    print(f"  LEFT  collar grab: ({l_hem[0]:.3f}, {l_hem[1]:.3f}, {l_hem[2]:.3f})")
    print(f"  Fold land (toward hem, clamped) R: ({shirt_top_r[0]:.3f}, {shirt_top_r[1]:.3f})")
    print(f"  Fold land (toward hem, clamped) L: ({shirt_top_l[0]:.3f}, {shirt_top_l[1]:.3f})")
    print(f"  Center Y: R={center_y_r:.3f} L={center_y_l:.3f}")

    # === DRY RUN ===
    if args.skip_dry_run:
        print("\n=== DRY RUN SKIPPED ===")
        ok = True
    else:
        print("\n=== DRY RUN ===")
        ok = True
    if not args.skip_dry_run and do_sleeves:
        if not dry_run_arc("R sleeve", "right", r_sleeve_grab[0], r_sleeve_grab[1],
                           r_sleeve_grab[0], center_y_r, HOME_Q):
            ok = False
        if not dry_run_arc("L sleeve", "left", rs_l[0], rs_l[1],
                           rs_l[0], center_y_l, HOME_Q):
            ok = False
    if not args.skip_dry_run and do_hem:
        # Validate collar transit: post-sleeve-retract -> collar grasp.
        # Use a simulated post-fold retract as the seed.
        r_retract_pos = [r_sleeve_grab[0], center_y_r, 0.08]
        l_retract_pos = [rs_l[0], center_y_l, 0.08]
        q_r_ret = ik("right", r_retract_pos, HOME_Q)
        q_l_ret = ik("left", l_retract_pos, HOME_Q)
        if q_r_ret is not None:
            if not dry_run_linear_transit("R collar transit", "right",
                                          r_retract_pos,
                                          [r_hem[0], r_hem[1], GRASP_Z],
                                          q_r_ret):
                ok = False
        if q_l_ret is not None:
            if not dry_run_linear_transit("L collar transit", "left",
                                          l_retract_pos,
                                          [l_hem[0], l_hem[1], GRASP_Z],
                                          q_l_ret):
                ok = False
        # Validate body fold arc (collar grasp -> drape toward hem -> land)
        if not dry_run_arc("R body fold", "right", r_hem[0], r_hem[1],
                           shirt_top_r[0], r_hem[1], HOME_Q, keep_x=False):
            ok = False
        if not dry_run_arc("L body fold", "left", l_hem[0], l_hem[1],
                           shirt_top_l[0], l_hem[1], HOME_Q, keep_x=False):
            ok = False
    if not ok:
        print("ABORT: dry run failed")
        if grip is not None:
            grip.disconnect()
        return
    print("  All dry runs passed")

    if args.dry_run:
        print("Dry run only (no hardware touched): not executing.")
        if grip is not None:
            grip.disconnect()
        return

    # Past the dry-run gate: NOW connect + open the grippers (real run only).
    cfg = load_family_yaml("bimanual_franka", args.machine)
    grip = grippers(cfg)
    print("\n=== GRIPPERS ===")
    grip_open(grip)
    grip_check(grip)

    # === RECORDER ===
    rec = None
    TaskRecorder = None
    if not args.no_record:
        try:
            from scripts.recorder import TaskRecorder
        except ImportError:
            # scripts/recorder.py is not on this branch (bimanual only).
            print("\n=== RECORDER: scripts.recorder not available; not recording ===")
    if TaskRecorder is not None:
        ts = time.strftime("%Y%m%d_%H%M%S")
        rec = TaskRecorder(f"fold_tshirt_{ts}")
        print(f"\n=== RECORDER: {rec.out_dir} ===")
        rec.capture_scene()
        rec.start_recording()
        rec.log_detections(list(d.values()))

        def _sigterm_handler(sig, frame):
            print("\n=== INTERRUPTED - saving recording ===")
            if rec:
                rec.stop_recording()
                rec.close()
                print(f"Recording saved to {rec.out_dir}")
            sys.exit(1)
        signal.signal(signal.SIGTERM, _sigterm_handler)
        signal.signal(signal.SIGINT, _sigterm_handler)

    # No LLM in the loop: the fold sequence is fixed and the grasp points come
    # straight from SAM3 geometry.

    # === CONNECT ROBOTS ===
    print("\n=== CONNECT ===")
    _robot_cfg = cfg.get("robot", {})
    left_ip = _robot_cfg.get("left_ip", "172.16.0.101")
    right_ip = _robot_cfg.get("right_ip", "172.16.0.102")
    l_robot = FrankaBambooDriver(ip=left_ip, port=5555)
    r_robot = FrankaBambooDriver(ip=right_ip, port=5556)
    l_robot.connect()
    r_robot.connect()

    # Home both
    print("Homing...")
    def hl(): l_robot.go_home(velocity=0.2)
    def hr(): r_robot.go_home(velocity=0.2)
    tl = threading.Thread(target=hl); tr = threading.Thread(target=hr)
    tl.start(); tr.start(); tl.join(); tr.join()
    print("  Both homed")
    if rec:
        rec.mark_keyframe("homed")
        rec.log_state(l_robot, r_robot, label="homed")

    try:
        # === SLEEVES (sequential: RIGHT first, then LEFT) ===
        if do_sleeves:
            print("\n=== PHASE 1: RIGHT sleeve fold ===")
            rr = pinch_drape(r_robot, grip, "right", "RIGHT: left sleeve",
                             r_sleeve_grab[0], r_sleeve_grab[1],
                             r_sleeve_grab[0], center_y_r,
                             orient_wxyz=orient_sleeve_r)
            if not rr:
                raise RuntimeError("RIGHT sleeve fold failed")
            print(f"  RIGHT sleeve: OK")
            time.sleep(0.3)

            print("\n=== PHASE 2: LEFT sleeve fold ===")
            lr = pinch_drape(l_robot, grip, "left", "LEFT: right sleeve",
                             rs_l[0], rs_l[1],
                             rs_l[0], center_y_l,
                             orient_wxyz=orient_sleeve_l)
            if not lr:
                raise RuntimeError("LEFT sleeve fold failed")
            print(f"  LEFT sleeve: OK")

            if rec:
                rec.mark_keyframe("sleeves_folded")
                rec.log_state(l_robot, r_robot, label="sleeves_folded")
            time.sleep(0.3)

        # === BODY FOLD (collar grab, drape toward hem) ===
        if do_hem:
            print("\n=== PHASE 3: Both arms move to collar (sync) ===")
            # Both approach hem edges via linearly-interpolated IK waypoints
            # from retract position. Bamboo can't handle >0.8 rad max joint
            # delta in a single step, so we subdivide into N_HEM_TRANSIT
            # evenly-spaced Cartesian waypoints with fresh IK at each step.
            # 6 waypoints keeps the worst-case delta <0.5 rad.
            N_HEM_TRANSIT = 8

            lq = np.array(l_robot.get_joint_positions())
            rq = np.array(r_robot.get_joint_positions())
            lt_start = l_robot.get_tcp_pose()
            rt_start = r_robot.get_tcp_pose()

            l_start = np.array([lt_start[0], lt_start[1], lt_start[2]])
            r_start = np.array([rt_start[0], rt_start[1], rt_start[2]])
            l_goal = np.array([l_hem[0], l_hem[1], GRASP_Z])
            r_goal = np.array([r_hem[0], r_hem[1], GRASP_Z])

            print(f"  R transit: ({r_start[0]:.3f},{r_start[1]:.3f},{r_start[2]:.3f}) "
                  f"-> ({r_goal[0]:.3f},{r_goal[1]:.3f},{r_goal[2]:.3f}) "
                  f"in {N_HEM_TRANSIT} steps")
            print(f"  L transit: ({l_start[0]:.3f},{l_start[1]:.3f},{l_start[2]:.3f}) "
                  f"-> ({l_goal[0]:.3f},{l_goal[1]:.3f},{l_goal[2]:.3f}) "
                  f"in {N_HEM_TRANSIT} steps")

            for step_i in range(1, N_HEM_TRANSIT + 1):
                t = step_i / N_HEM_TRANSIT
                l_pos = l_start + t * (l_goal - l_start)
                r_pos = r_start + t * (r_goal - r_start)

                l_wp = ik("left", l_pos.tolist(), lq)
                r_wp = ik("right", r_pos.tolist(), rq)

                if l_wp is None:
                    print(f"  ABORT: LEFT IK failed at hem transit step {step_i}/{N_HEM_TRANSIT}")
                    raise RuntimeError(f"LEFT IK failed at hem transit step {step_i}")
                if r_wp is None:
                    print(f"  ABORT: RIGHT IK failed at hem transit step {step_i}/{N_HEM_TRANSIT}")
                    raise RuntimeError(f"RIGHT IK failed at hem transit step {step_i}")

                l_delta = float(np.max(np.abs(l_wp - lq[:7])))
                r_delta = float(np.max(np.abs(r_wp - rq[:7])))
                vel = APPROACH_VEL if step_i == N_HEM_TRANSIT else MOVE_VEL
                print(f"    step {step_i}/{N_HEM_TRANSIT}: "
                      f"L({l_pos[0]:.2f},{l_pos[1]:.2f},{l_pos[2]:.2f}) d={l_delta:.3f}  "
                      f"R({r_pos[0]:.2f},{r_pos[1]:.2f},{r_pos[2]:.2f}) d={r_delta:.3f}")

                l_wp_list = l_wp.tolist()
                r_wp_list = r_wp.tolist()
                def _ml(q=l_wp_list, v=vel): l_robot.move_to_joint_config(q, velocity=v)
                def _mr(q=r_wp_list, v=vel): r_robot.move_to_joint_config(q, velocity=v)
                tl = threading.Thread(target=_ml); tr = threading.Thread(target=_mr)
                tl.start(); tr.start(); tl.join(); tr.join()
                time.sleep(0.1)

                lq = l_wp.copy()
                rq = r_wp.copy()

            time.sleep(0.3)

            # Verify both arms reached the target z
            lt = l_robot.get_tcp_pose(); rt = r_robot.get_tcp_pose()
            print(f"  L at hem: ({lt[0]:.3f},{lt[1]:.3f},{lt[2]:.3f})")
            print(f"  R at hem: ({rt[0]:.3f},{rt[1]:.3f},{rt[2]:.3f})")
            HEM_Z_TOL = 0.030  # 30mm tolerance
            if abs(lt[2] - GRASP_Z) > HEM_Z_TOL:
                print(f"  ABORT: LEFT arm z={lt[2]:.3f} too far from target {GRASP_Z:.3f}")
                raise RuntimeError(f"LEFT arm failed to reach hem z (at {lt[2]:.3f}, want {GRASP_Z:.3f})")
            if abs(rt[2] - GRASP_Z) > HEM_Z_TOL:
                print(f"  ABORT: RIGHT arm z={rt[2]:.3f} too far from target {GRASP_Z:.3f}")
                raise RuntimeError(f"RIGHT arm failed to reach hem z (at {rt[2]:.3f}, want {GRASP_Z:.3f})")

            # Grasp hem - sync barrier: both arms must be at target before grasping
            print("  === SYNC BARRIER: verifying both arms at hem ===")
            lt = l_robot.get_tcp_pose(); rt = r_robot.get_tcp_pose()
            print(f"    L: ({lt[0]:.3f},{lt[1]:.3f},{lt[2]:.3f}) target z={GRASP_Z:.3f}")
            print(f"    R: ({rt[0]:.3f},{rt[1]:.3f},{rt[2]:.3f}) target z={GRASP_Z:.3f}")
            if abs(lt[2] - GRASP_Z) > HEM_Z_TOL or abs(rt[2] - GRASP_Z) > HEM_Z_TOL:
                print("  ABORT: one or both arms not at hem z")
                raise RuntimeError("Sync barrier failed - arms not at hem")
            print("  Both arms at hem - grasping...")
            if rec:
                rec.mark_keyframe("at_hem")
                rec.log_state(l_robot, r_robot, label="at_hem")
            grip_close(grip)
            time.sleep(0.4)
            capture_frame("collar_grasp", args.frames)
            print("  Verifying grasp...")
            for a in ["left", "right"]:
                raw = getattr(grip.for_arm(a)._motor, "gripper_position", None)
                empty = raw is not None and raw >= 250
                print(f"    {a}: raw={raw} {'EMPTY' if empty else 'HOLDING'}")

            # Lift, incremental steps to avoid bamboo table stall
            N_LIFT = 3
            lq = np.array(l_robot.get_joint_positions())
            rq = np.array(r_robot.get_joint_positions())
            lt2 = l_robot.get_tcp_pose(); rt2 = r_robot.get_tcp_pose()
            for li in range(1, N_LIFT + 1):
                frac = li / N_LIFT
                lz = lt2[2] + frac * LIFT_Z
                rz = rt2[2] + frac * LIFT_Z
                l_s = ik("left", [lt2[0], lt2[1], lz], lq)
                r_s = ik("right", [rt2[0], rt2[1], rz], rq)
                if l_s is not None and r_s is not None:
                    l_list = l_s.tolist(); r_list = r_s.tolist()
                    def _ml(q=l_list): l_robot.move_to_joint_config(q, velocity=APPROACH_VEL)
                    def _mr(q=r_list): r_robot.move_to_joint_config(q, velocity=APPROACH_VEL)
                    tl = threading.Thread(target=_ml); tr = threading.Thread(target=_mr)
                    tl.start(); tr.start(); tl.join(); tr.join()
                    lq = l_s; rq = r_s
                time.sleep(0.1)
            print("  Lifted")

            # Hem fold arc
            print("\n=== PHASE 4: Body fold (drape collar toward hem) ===")
            lt3 = l_robot.get_tcp_pose(); rt3 = r_robot.get_tcp_pose()
            l_start = np.array(lt3[:3]); r_start = np.array(rt3[:3])
            l_end = np.array([shirt_top_l[0], lt3[1], lt3[2]])
            r_end = np.array([shirt_top_r[0], rt3[1], rt3[2]])
            print(f"  L: x={l_start[0]:.3f} -> {l_end[0]:.3f}")
            print(f"  R: x={r_start[0]:.3f} -> {r_end[0]:.3f}")
            hem_fold(l_robot, r_robot, l_start, r_start, l_end, r_end)
            capture_frame("after_drape", args.frames)
            if rec:
                rec.mark_keyframe("hem_folded")
                rec.log_state(l_robot, r_robot, label="hem_folded")

            # Release + home
            print("\n=== PHASE 5: Release + home ===")
            grip_open(grip)

            # Retract up
            lq = np.array(l_robot.get_joint_positions())
            rq = np.array(r_robot.get_joint_positions())
            lt4 = l_robot.get_tcp_pose(); rt4 = r_robot.get_tcp_pose()
            l_up = ik("left", [lt4[0], lt4[1], 0.10], lq)
            r_up = ik("right", [rt4[0], rt4[1], 0.10], rq)
            if l_up is not None and r_up is not None:
                def ml4(): l_robot.move_to_joint_config(l_up.tolist(), velocity=MOVE_VEL)
                def mr4(): r_robot.move_to_joint_config(r_up.tolist(), velocity=MOVE_VEL)
                tl = threading.Thread(target=ml4); tr = threading.Thread(target=mr4)
                tl.start(); tr.start(); tl.join(); tr.join()

            def hl2(): l_robot.go_home(velocity=0.2)
            def hr2(): r_robot.go_home(velocity=0.2)
            tl = threading.Thread(target=hl2); tr = threading.Thread(target=hr2)
            tl.start(); tr.start(); tl.join(); tr.join()
            if rec:
                rec.mark_keyframe("complete")
                rec.log_state(l_robot, r_robot, label="complete")
            print("  DONE")

    except Exception as e:
        print(f"\nFAILED: {e}")
        import traceback; traceback.print_exc()
        if rec:
            rec.mark_keyframe("failed")
        try: grip_open(grip)
        except: pass
        try:
            l_robot.go_home(velocity=0.2)
            r_robot.go_home(velocity=0.2)
        except: pass

    if rec:
        rec.stop_recording()
        rec.close()
        print(f"\nRecording saved to {rec.out_dir}")

    l_robot.disconnect()
    r_robot.disconnect()
    grip.disconnect()


if __name__ == "__main__":
    main()

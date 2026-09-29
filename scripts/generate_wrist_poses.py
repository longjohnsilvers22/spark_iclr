#!/usr/bin/env python3
"""Generate a 100-pose set for the wrist hand-eye calibration.

Design:
- ArUco/AprilTag is laid on the table at world ~(0.45, 0.0) (Z = table top).
- The wrist RealSense is on the gripper; with the TCP at home orientation
  (rotvec [pi, 0, 0]) the camera looks straight DOWN.
- Tag-image-rotation diversity, so the handeye solver isn't conditioned to
  a single tag orientation: TCP yaw rotations about world Z (the
  tag-on-table rotates in the image plane).
- Depth diversity (TCP Z from ~0.12m, just above the table, to ~0.50m).
  Low Z is critical for constraining the cam-to-TCP translation; an
  under-constrained t_z shows up as a Z bias of order 17mm.
- Lever-arm diversity (tilted views, offset XY) to constrain the rotation
  portion of the cam-to-TCP transform.

Output: scripts/calibration_poses/poses_wrist.json (overwrites).
"""

import json
from pathlib import Path

import numpy as np
from scipy.spatial.transform import Rotation as R

OUT = (Path(__file__).resolve().parent
       / "calibration_poses" / "poses_wrist.json")

# Tag world XY (table-mounted). Z is the table; the TCP never goes there.
TAG_X = 0.45
TAG_Y = 0.00

# TCP "look straight down" rotvec.
DOWN_RV = np.array([np.pi, 0.0, 0.0])
R_DOWN = R.from_rotvec(DOWN_RV)


def pose_yaw_tilt(x, y, z, yaw_deg=0.0, pitch_rad=0.0, roll_rad=0.0):
    """
    Build a TCP pose [x, y, z, rx, ry, rz] with the gripper pointing
    down, then yaw'd about world Z and optionally tilted in TCP's own
    pitch/roll. Yaw rotates the cam image plane; pitch/roll create
    oblique views that constrain the handeye rotation.
    """
    r = R_DOWN
    if pitch_rad or roll_rad:
        # Local-frame tilt: apply pitch (about TCP-y) then roll (about TCP-x)
        # AFTER the down flip. This keeps the camera approximately
        # looking at the tag region.
        r = r * R.from_rotvec([roll_rad, pitch_rad, 0.0])
    if yaw_deg:
        # World-frame yaw: pre-multiply so the gripper spins about world Z.
        r = R.from_rotvec([0.0, 0.0, np.deg2rad(yaw_deg)]) * r
    rv = r.as_rotvec()
    return [round(x, 3), round(y, 3), round(z, 3),
            round(float(rv[0]), 4), round(float(rv[1]), 4),
            round(float(rv[2]), 4)]


def main():
    poses = []

    # Group A: 30 "directly above tag" poses, Z sweep
    # Pure down-look from TCP directly above the tag.  Z varies from
    # 0.12 (just above the table) to 0.50.  Several at each extreme to
    # anchor t_z.
    for z in [0.12, 0.14, 0.16, 0.18, 0.20, 0.22, 0.25, 0.28,
              0.32, 0.36, 0.40, 0.44, 0.48]:
        poses.append(pose_yaw_tilt(TAG_X, TAG_Y, z))
    # Extra coverage at the extremes for t_z conditioning.
    for z in [0.13, 0.15, 0.17, 0.46, 0.50, 0.45]:
        poses.append(pose_yaw_tilt(TAG_X, TAG_Y, z))
    # Slight XY jitter at mid Z to break degeneracy from a static
    # camera optical axis.
    for dx, dy in [(0.02, 0.0), (-0.02, 0.0), (0.0, 0.02), (0.0, -0.02),
                   (0.02, 0.02), (-0.02, -0.02),
                   (0.03, -0.02), (-0.03, 0.02),
                   (0.04, 0.0), (-0.04, 0.0), (0.0, 0.04)]:
        poses.append(pose_yaw_tilt(TAG_X + dx, TAG_Y + dy, 0.25))

    # Group B: 25 "yaw-rotated" poses
    # Spin the gripper about world Z so the tag rotates in the cam image
    # and the solve is not locked to one tag orientation.
    for yaw in [30, 60, 90, 120, 150, 180, 210, 240, 270, 300, 330]:
        # At a mid hover Z, varying yaw across the full circle.
        poses.append(pose_yaw_tilt(TAG_X, TAG_Y, 0.28, yaw_deg=yaw))
    # Some at lower Z (close to the table), bigger image rotation effect.
    for yaw in [45, 135, 225, 315]:
        poses.append(pose_yaw_tilt(TAG_X, TAG_Y, 0.18, yaw_deg=yaw))
    # Mid-Z with XY offset + yaw.
    for yaw, dx, dy in [
        (45,  0.03, 0.02), (90,  -0.02, 0.03),
        (135, -0.03, -0.02), (180, 0.02, -0.03),
        (225, 0.03, 0.0), (270, 0.0, 0.03),
        (315, -0.03, 0.0), (0,   0.0, -0.03),
        (60,  0.04, 0.04), (240, -0.04, -0.04),
    ]:
        poses.append(pose_yaw_tilt(TAG_X + dx, TAG_Y + dy, 0.30, yaw_deg=yaw))

    # Group C: 25 "tilted oblique" poses
    # Tilt the gripper so the cam looks at the tag from an angle.  These
    # constrain the rotation portion of T_cam_to_tcp.  Keep tilts <=0.30
    # rad (~17 deg) so the tag stays in the wrist's narrow FOV.
    for tilt in [0.15, -0.15, 0.25, -0.25]:
        # Pitch only (cam looks forward/back)
        poses.append(pose_yaw_tilt(TAG_X, TAG_Y, 0.28, pitch_rad=tilt))
        # Roll only (cam looks left/right)
        poses.append(pose_yaw_tilt(TAG_X, TAG_Y, 0.28, roll_rad=tilt))
    for tilt_p, tilt_r in [
        (0.20, 0.20), (-0.20, 0.20), (0.20, -0.20), (-0.20, -0.20),
        (0.15, 0.25), (-0.15, -0.25),
    ]:
        poses.append(pose_yaw_tilt(TAG_X, TAG_Y, 0.30,
                                    pitch_rad=tilt_p, roll_rad=tilt_r))
    # Tilts at low Z (close-up obliques)
    for tilt in [0.18, -0.18]:
        poses.append(pose_yaw_tilt(TAG_X, TAG_Y, 0.20, pitch_rad=tilt))
        poses.append(pose_yaw_tilt(TAG_X, TAG_Y, 0.20, roll_rad=tilt))
    # Tilts at high Z
    for tilt in [0.20, -0.20]:
        poses.append(pose_yaw_tilt(TAG_X, TAG_Y, 0.42, pitch_rad=tilt))
        poses.append(pose_yaw_tilt(TAG_X, TAG_Y, 0.42, roll_rad=tilt))
    # Tilt + slight XY offset
    poses.append(pose_yaw_tilt(TAG_X + 0.04, TAG_Y, 0.25, pitch_rad=-0.20))
    poses.append(pose_yaw_tilt(TAG_X - 0.04, TAG_Y, 0.25, pitch_rad= 0.20))
    poses.append(pose_yaw_tilt(TAG_X, TAG_Y + 0.04, 0.25, roll_rad=-0.20))

    # Group D: 20 "fully diverse" poses
    # Combined yaw + tilt + Z + XY, these are the highest-information
    # poses for the solver.
    diverse = [
        (TAG_X + 0.03, TAG_Y + 0.02, 0.16,  45, 0.10, 0.10),
        (TAG_X - 0.03, TAG_Y - 0.02, 0.20, 135, -0.15, 0.10),
        (TAG_X + 0.03, TAG_Y - 0.03, 0.22, 225, 0.15, -0.10),
        (TAG_X - 0.03, TAG_Y + 0.03, 0.26, 315, -0.10, -0.15),
        (TAG_X + 0.04, TAG_Y + 0.04, 0.32,  60, -0.10, 0.10),
        (TAG_X - 0.04, TAG_Y - 0.04, 0.34, 120, 0.10, -0.10),
        (TAG_X + 0.04, TAG_Y - 0.04, 0.38, 200, -0.10, -0.10),
        (TAG_X - 0.04, TAG_Y + 0.04, 0.42, 280, 0.10, 0.10),
        (TAG_X + 0.05, TAG_Y, 0.22, 90, 0.0, -0.20),
        (TAG_X - 0.05, TAG_Y, 0.22, 270, 0.0, 0.20),
        (TAG_X, TAG_Y + 0.05, 0.24, 180, -0.20, 0.0),
        (TAG_X, TAG_Y - 0.05, 0.24, 0, 0.20, 0.0),
        (TAG_X + 0.03, TAG_Y + 0.03, 0.45, 45, 0.0, 0.0),
        (TAG_X - 0.03, TAG_Y - 0.03, 0.45, 225, 0.0, 0.0),
        (TAG_X + 0.02, TAG_Y - 0.02, 0.14, 60, 0.0, 0.0),
        (TAG_X - 0.02, TAG_Y + 0.02, 0.14, 300, 0.0, 0.0),
        (TAG_X, TAG_Y, 0.36, 75, 0.15, 0.15),
        (TAG_X, TAG_Y, 0.36, 255, -0.15, -0.15),
        (TAG_X, TAG_Y, 0.19, 105, 0.10, -0.05),
        (TAG_X, TAG_Y, 0.19, 195, -0.10, 0.05),
    ]
    for (x, y, z, yaw, pitch, roll) in diverse:
        poses.append(pose_yaw_tilt(x, y, z,
                                    yaw_deg=yaw,
                                    pitch_rad=pitch,
                                    roll_rad=roll))

    # Sanity: keep going past 100 if the padding put us over.
    print(f"Generated {len(poses)} poses (target: 100)")
    z_vals = [p[2] for p in poses]
    print(f"  Z range: {min(z_vals):.3f} to {max(z_vals):.3f}")

    payload = {
        "comment": (
            "v5 wrist hand-eye pose set, 100 poses. Tag laid on the "
            "TABLE at ~(0.45, 0.0). Designed to constrain the residual "
            "Z translation bias and rotation in T_cam_to_tcp. Heavy "
            "yaw diversity (every 30 deg) so the ArUco appears at all "
            "image-plane rotations. Z spans 0.12-0.50m including "
            "extreme close-ups for t_z conditioning."
        ),
        "intended_camera": "wrist",
        "frame": "robot_base",
        "rotation_format": "axis_angle_rotvec_rad",
        "tag_world_xy": [TAG_X, TAG_Y],
        "poses": poses,
    }

    OUT.parent.mkdir(parents=True, exist_ok=True)
    with open(OUT, "w") as f:
        json.dump(payload, f, indent=2)
    print(f"Wrote {OUT}")


if __name__ == "__main__":
    main()

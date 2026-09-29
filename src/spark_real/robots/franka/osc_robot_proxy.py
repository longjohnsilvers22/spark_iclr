"""
Drop-in stand-in for FrankaDriver that talks to the robots_realtime
OSC subprocess via HTTP instead of holding the FCI directly.

When SPARK_EXECUTOR_BACKEND=osc, panda-py owns the FCI inside the
rr-session subprocess. SPARK's executor still expects to call methods
like ``open_gripper()``, ``get_joint_positions()``, ``move_linear()``,
``go_home()`` on ``pipeline._robot``. This class exposes that interface
and forwards each call to the bridge agent's HTTP endpoints:

  GET  /state           - joint_pos + last_target + tcp_xyz + tcp_wxyz + grip
  POST /move_to_pose    - body {position, wxyz, duration_s, gripper_width?}
  POST /open            - gripper to ~80 mm
  POST /close           - gripper to 0

Endpoints that don't exist on the bridge (joint-space motions, reflex
recovery, velocity streaming) raise ``NotImplementedError`` - the OSC
backend is only fit for Cartesian-only pick-and-place flows. Callers
should check ``_osc_backend_enabled()`` upstream when they need joint
motions.
"""

from __future__ import annotations

import json
import logging
import time
import urllib.request
import urllib.error
from typing import List, Optional, Tuple

import numpy as np
from scipy.spatial.transform import Rotation

from spark_real.robots.franka.franka_base import HOME_CONFIG
from spark_real.control.fr3_ik_pyroki import fk

logger = logging.getLogger(__name__)


def _http_get_json(url: str, timeout: float = 5.0) -> dict:
    req = urllib.request.Request(url, method="GET")
    with urllib.request.urlopen(req, timeout=timeout) as r:
        return json.loads(r.read().decode())


def _http_post_json(url: str, body: dict, timeout: float = 10.0) -> dict:
    raw = json.dumps(body).encode()
    req = urllib.request.Request(
        url,
        data=raw,
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    with urllib.request.urlopen(req, timeout=timeout) as r:
        return json.loads(r.read().decode())


class OSCRobotProxy:
    """
    FrankaDriver-shaped facade backed by the OSC subprocess.

    The executor's existing primitives (``self.robot.open_gripper()``
    etc.) work unchanged - internally each routes to an HTTP call on
    the rr-session bridge.
    """

    SUPPORTS_URSCRIPT = False  # not a UR

    # HOME_CONFIG matches FrankaDriver's HOME_CONFIG (avoid wrist singularity).
    # Single-sourced from franka_base.HOME_CONFIG (== franka_default.yaml home_config).
    HOME_CONFIG = list(HOME_CONFIG)

    def __init__(self, bridge_url: str = "http://127.0.0.1:9009"):
        self.bridge_url = bridge_url.rstrip("/")
        self._connected = True
        # Used by some executor paths that probe has_errors.
        self.has_errors = False
        logger.info("OSCRobotProxy: bound to %s", self.bridge_url)

    # Connection lifecycle (no-ops, bridge has its own lifecycle).
    def connect(self):
        self._connected = True

    def disconnect(self):
        self._connected = False

    def recover_from_errors(self):
        # The OSC controller doesn't enter Franka's Reflex mode the same
        # way (impedance soaks small disturbances). No-op so executor
        # retry paths don't crash.
        return True

    def stop(self):
        return

    def stop_velocity(self):
        return

    def stop_motion(self):
        return

    # State read
    def _state(self) -> dict:
        return _http_get_json(f"{self.bridge_url}/state", timeout=3.0)

    def get_joint_positions(self) -> List[float]:
        s = self._state()
        q = s.get("observed_joint_pos")
        if q is None:
            # Bridge hasn't received its first observation yet, short
            # sleep + retry. Without this the executor's pre-task
            # state read can fail before the first FCI tick.
            time.sleep(0.3)
            s = self._state()
            q = s.get("observed_joint_pos")
        if q is None:
            raise RuntimeError("OSC bridge has no observed_joint_pos yet")
        return list(q)

    # SPARK's franky returns TCP at the gripper tip (panda_grasptarget,
    # ~0.1034 m beyond panda_hand). The OSC bridge agent's FK targets
    # panda_hand. If we forward the bridge's TCP unchanged, every
    # downstream consumer that depends on TCP (wrist hand-eye in
    # particular) sees the robot ~10 cm higher than it actually is,
    # which drops wrist-camera detections by the same amount and pushes
    # them under the perception's -0.28 m below-table filter. Apply the
    # offset locally so the proxy's TCP matches franky's convention.
    TCP_LOCAL_Z_OFFSET = 0.1034  # m, panda_hand -> panda_grasptarget

    def get_tcp_pose(self) -> Tuple[List[float], List[float]]:
        s = self._state()
        xyz = s.get("tcp_xyz")
        wxyz = s.get("tcp_wxyz")
        if xyz is None or wxyz is None:
            raise RuntimeError("OSC bridge state has no TCP pose")
        # Rotate the local-frame +Z offset into base frame using the
        # bridge's reported orientation. wxyz is (w,x,y,z); scipy wants
        # xyzw.
        xyzw = [float(wxyz[1]), float(wxyz[2]), float(wxyz[3]), float(wxyz[0])]
        R_mat = Rotation.from_quat(xyzw).as_matrix()
        offset_base = R_mat @ np.array([0.0, 0.0, self.TCP_LOCAL_Z_OFFSET])
        xyz_corrected = [
            float(xyz[0]) + float(offset_base[0]),
            float(xyz[1]) + float(offset_base[1]),
            float(xyz[2]) + float(offset_base[2]),
        ]
        return xyz_corrected, [float(v) for v in wxyz]

    def get_observation(self) -> dict:
        s = self._state()
        return {
            "joint_pos": s.get("observed_joint_pos"),
            "tcp_pos": s.get("tcp_xyz"),
            "tcp_wxyz": s.get("tcp_wxyz"),
            "gripper_width": s.get("gripper_width"),
        }

    def get_tcp_force(self):
        # OSC controller doesn't expose external-force vector through
        # the bridge. Return zeros so grasp-verify falls back to
        # gripper-width signal instead of force.
        return np.zeros(6, dtype=float)

    # Motion
    def move_linear(self, pose, velocity: float = 0.25, **_):
        """
        ``pose`` is the SPARK executor's 6-vec [x,y,z,rx,ry,rz]
        (axis-angle). Converts to position+wxyz and POSTs to bridge.
        """
        pose = list(pose)
        pos = [float(pose[0]), float(pose[1]), float(pose[2])]
        rxyz = np.asarray(pose[3:6], dtype=float)
        q_xyzw = Rotation.from_rotvec(rxyz).as_quat()
        wxyz = [float(q_xyzw[3]), float(q_xyzw[0]), float(q_xyzw[1]), float(q_xyzw[2])]
        duration_s = max(1.5, min(4.0, 0.5 / max(float(velocity), 0.05)))
        body = {"position": pos, "wxyz": wxyz, "duration_s": duration_s}
        resp = _http_post_json(
            f"{self.bridge_url}/move_to_pose",
            body,
            timeout=duration_s + 10.0,
        )
        if not resp.get("ok"):
            raise RuntimeError(f"OSC move_to_pose ok=false: {resp.get('error')}")

    def move_to_joint_config(self, q, velocity: float = 0.25, **_):
        """
        No native joint-space motion via the bridge. Compute the FK
        of the requested joint config locally and route as a
        move_linear to that pose. This works as long as ``q`` is a
        valid IK solution for some reachable Cartesian pose, which
        it always is for any q produced by our own pyroki IK.
        """
        pos, R = fk(list(q))
        rxyz = Rotation.from_matrix(R).as_rotvec()
        pose = list(pos) + list(rxyz)
        self.move_linear(pose, velocity=velocity)

    def servo_joint(self, *args, **kwargs):
        raise NotImplementedError(
            "OSC backend does not support joint-streaming primitives. "
            "Stop the OSC subprocess and switch to franky for skills "
            "that need servo_joint."
        )

    def send_velocity(self, *args, **kwargs):
        raise NotImplementedError(
            "OSC backend does not support send_velocity. " "Stop OSC for teleop."
        )

    def go_home(self):
        # Use move_to_joint_config to drive to the home pose. Bridges
        # the OSC subprocess to /move_to_pose via FK.
        self.move_to_joint_config(self.HOME_CONFIG, velocity=0.20)

    # Gripper
    def open_gripper(self):
        resp = _http_post_json(f"{self.bridge_url}/open", {}, timeout=5.0)
        if not resp.get("ok"):
            raise RuntimeError(f"OSC /open ok=false: {resp.get('error')}")

    def close_gripper(self, force: float = 50.0):
        resp = _http_post_json(f"{self.bridge_url}/close", {}, timeout=5.0)
        if not resp.get("ok"):
            raise RuntimeError(f"OSC /close ok=false: {resp.get('error')}")

    def set_gripper_position(self, width: float, **_):
        # Bridge only has open/close. Threshold to closest end.
        if width <= 0.04:
            self.close_gripper()
        else:
            self.open_gripper()

    def _send_gripper_command(self, *args, **kwargs):
        # UR10e-style helper used by some skills. Treat the first
        # positional arg as a width signal.
        width = args[0] if args else kwargs.get("width", 0.0)
        try:
            w = float(width)
        except (TypeError, ValueError):
            w = 0.0
        # 1.0 in UR convention means CLOSE; small numbers mean open.
        if w >= 0.5:
            self.close_gripper()
        else:
            self.open_gripper()

    def get_gripper_width(self) -> float:
        s = self._state()
        return float(s.get("gripper_width") or 0.0)


__all__ = ["OSCRobotProxy"]

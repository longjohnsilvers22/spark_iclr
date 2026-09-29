"""
Shared types and OSC helpers for the ScoreExecutor family.
"""

import json
import os
import urllib.error
import urllib.request
from dataclasses import dataclass

import numpy as np
from scipy.spatial.transform import Rotation


# Leaf primitive names the executor dispatches directly (not via the skill
# registry). The planner validates emitted leaf types against this same set.
# The `action` meta-type and structural node types (sequence, selector,
# parallel, sync_barrier) are added locally at each site.
BUILTIN_PRIMITIVES = frozenset(
    {
        "move_to_keypoint",
        "grasp",
        "release",
        "move_relative",
        "wait",
    }
)


class AbortRequested(Exception):
    """Raised when the operator requests an emergency stop mid-motion.

    Propagated out of every motion-issuing helper so that no new motion
    command can overwrite the stop that was just sent to the controller.
    """

    pass


@dataclass
class ExecutionResult:
    action_type: str
    success: bool
    message: str = ""
    duration: float = 0.0


def osc_backend_enabled() -> bool:
    return os.environ.get("SPARK_EXECUTOR_BACKEND", "franky").strip().lower() == "osc"


def osc_move_linear(pose, velocity: float) -> None:
    """
    POST a Cartesian-pose move to the local OSC bridge.

    Converts the 6-vec [x,y,z,rx,ry,rz] (axis-angle) to (position, wxyz)
    and posts to the local OSC endpoint. The bridge blocks until motion
    completes.
    """
    pos = [float(pose[0]), float(pose[1]), float(pose[2])]
    rxyz = np.asarray(pose[3:6], dtype=float)
    q_xyzw = Rotation.from_rotvec(rxyz).as_quat()
    wxyz = [float(q_xyzw[3]), float(q_xyzw[0]), float(q_xyzw[1]), float(q_xyzw[2])]
    duration_s = max(1.5, min(4.0, 0.5 / max(velocity, 0.05)))
    payload = {
        "position": pos,
        "wxyz": wxyz,
        "duration_s": duration_s,
    }
    raw = json.dumps(payload).encode()
    req = urllib.request.Request(
        "http://127.0.0.1:8888/api/control/osc/move_to_pose",
        data=raw,
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    timeout = duration_s + 10.0
    try:
        with urllib.request.urlopen(req, timeout=timeout) as r:
            resp = json.loads(r.read().decode())
    except urllib.error.HTTPError as exc:
        body = ""
        try:
            body = exc.read().decode()
        except Exception:
            pass
        raise RuntimeError(
            f"OSC move_to_pose returned HTTP {exc.code}: {body}"
        ) from exc
    except Exception as exc:
        raise RuntimeError(f"OSC move_to_pose request failed: {exc!r}") from exc
    if not resp.get("ok"):
        raise RuntimeError(f"OSC move_to_pose responded ok=false: {resp.get('error')}")


def maybe_osc_move_linear(pose, velocity: float) -> bool:
    """
    Route through OSC if the env var is set. Returns True if handled.
    """
    if not osc_backend_enabled():
        return False
    osc_move_linear(pose, velocity)
    return True

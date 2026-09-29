"""
Smoke test: exercise grasp_se3 with strategy=top_down using a mock
executor. Catches signature mismatches, import errors, and basic
control-flow bugs before they hit hardware.

Usage:
    python -m spark_real.tests.test_grasp_top_down_smoke
Exit 0 = passed; non-zero = failed.
"""

from __future__ import annotations
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
import numpy as np
from spark_real.skills import grasping


class _MockRobot:
    def get_joint_positions(self):
        return np.array([0.01, -0.54, 0.0, -2.20, -0.8, 1.67, 0.80])

    def open_gripper(self):
        pass

    def close_gripper(self):
        pass

    def grasp_to_width(self, **kw):
        self._last_grasp_to_width_kwargs = kw

    _last_grasp_to_width_kwargs = None


class _MockExecutor:
    """
    Minimal interface used by grasp_se3 + _grasp_top_down.
    """

    velocity = 0.10
    TABLE_Z_FLOOR = -0.040
    MAX_YAW_OFFSET = np.pi
    GRASP_ORIENTATION = [np.pi, 0.0, 0.0]
    GRIPPER_FULLY_CLOSED = 245
    _holding = False
    _last_keypoint_label = ""
    _last_grasp_target_width = None
    _redetect_thread = None

    def __init__(self, det):
        self.robot = _MockRobot()
        self.detection_map = {"spoon handle 1": det}
        self._motion_log = []
        self._gripper_log = []
        # Track the commanded TCP so _get_current_position() reflects motion,
        # like a real arm ending at the target after a move (the descent-reached
        # check reads this; a static value makes it look like the arm never moved).
        self._cur_pos = [0.40, 0.0, 0.48]

    def _get_current_position(self):
        return np.array(self._cur_pos)

    def _safe_clearance_z(self, start, end):
        return max(start[2], end[2] + 0.15)

    def _oriented_grasp(self, yaw_offset=0.0):
        from scipy.spatial.transform import Rotation

        base = Rotation.from_rotvec(self.GRASP_ORIENTATION)
        yaw = Rotation.from_euler("z", yaw_offset)
        return (yaw * base).as_rotvec().tolist()

    def _move_to(self, pos, orient, *a, **kw):
        self._cur_pos = list(map(float, pos))
        self._motion_log.append(("move_to", list(map(float, pos))))

    def _servo_to(self, pos, orient, **kw):
        self._cur_pos = list(map(float, pos))
        self._motion_log.append(("servo_to", list(map(float, pos)), kw.get("velocity")))

    def _gripper_squeeze(self, **kw):
        self._gripper_log.append(("squeeze", kw))

    def _get_gripper_position(self):
        return 100

    def _verify_grasp(self):
        return True


def main():
    det = {
        "position_3d": [0.5, 0.2, -0.04],
        "orientation_angle": np.radians(85.1),
        "aspect_ratio": 6.7,
        "obb_minor_m": 0.018,
        "confidence": 0.91,
        "_mask": None,
        "_camera": "birdview",
    }
    exe = _MockExecutor(det)

    params = {
        "keypoint_label": "spoon handle 1",
        "strategy": "top_down",
        "force": 50,
        "target_width": 0.010,
    }
    result = grasping.grasp_se3(exe, params)
    assert result.success, f"top_down grasp failed: {result.message}"
    gripper_issued = (
        bool(exe._gripper_log) or exe.robot._last_grasp_to_width_kwargs is not None
    )
    assert gripper_issued, "no gripper command issued"

    motion_types = [m[0] for m in exe._motion_log]
    # All three phases (hover, descent, lift) now use planned _move_to.
    # The PD servo was dropped after the FR3 acceleration-discontinuity
    # reflex storm on knife grasps. See _grasp_top_down comments.
    assert (
        motion_types.count("move_to") >= 2
    ), f"expected at least 2 move_to in {motion_types}"

    print("smoke test: top_down strategy OK")
    print(f"result.message: {result.message}")
    print(f"motions: {len(exe._motion_log)} ({motion_types})")
    print(f"gripper:  {exe._gripper_log}")


def test_grasp_top_down_smoke():
    """pytest entry point for the top-down grasp smoke test."""
    main()


if __name__ == "__main__":
    main()

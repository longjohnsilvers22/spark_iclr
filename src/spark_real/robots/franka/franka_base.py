'''
FrankaDriverBase - shared constants, gripper logic, and interface for all Franka drivers
'''
from __future__ import annotations

import math
import logging
from abc import ABC, abstractmethod
from typing import Optional, Sequence

import numpy as np

logger = logging.getLogger(__name__)

# Shared constants

# Canonical FR3 ready pose. Must match configs/franka_default.yaml robot.home_config byte-for-byte.
HOME_CONFIG = [0.0, -0.785398, 0.0, -2.356194, -0.15, 1.570796, 0.785398]

JOINT_LIMITS = [
    (-2.7437, 2.7437),
    (-1.7837, 1.7837),
    (-2.9007, 2.9007),
    (-3.0421, -0.1518),
    (-2.8065, 2.8065),
    (0.5445, 4.5169),
    (-3.0159, 3.0159),
]

GRIPPER_OPEN_WIDTH = 0.08
GRIPPER_CLOSED_WIDTH = 0.0
GRIPPER_DEFAULT_SPEED = 0.1
GRIPPER_DEFAULT_FORCE = 70.0
GRIPPER_EPSILON = 0.005

DEFAULT_VELOCITY = 0.25
DEFAULT_JOINT_VEL = 1.05


def build_flange_T_tcp() -> np.ndarray:
    # URDF: fr3_link8 -> fr3_hand (Rz(-pi/4)) -> fr3_hand_tcp (z+=0.1034).
    c, s = math.cos(-math.pi / 4), math.sin(-math.pi / 4)
    T = np.array([[c, -s, 0, 0], [s, c, 0, 0], [0, 0, 1, 0], [0, 0, 0, 1]])
    T[2, 3] = 0.1034  # hand -> tcp offset
    # Rz(-pi/4) already applied; the translation is in the rotated frame
    T_hand = np.array([[c, -s, 0, 0], [s, c, 0, 0], [0, 0, 1, 0], [0, 0, 0, 1]])
    T_tcp = np.eye(4)
    T_tcp[2, 3] = 0.1034
    return T_hand @ T_tcp


FLANGE_T_TCP = build_flange_T_tcp()


class FrankaDriverBase(ABC):
    """
    Abstract base for Franka FR3 drivers (franky, torque, bamboo).

    Subclasses implement connect/disconnect/motion. Gripper logic,
    constants, and observation synthesis are shared here.
    """

    robot_family = "franka"
    SUPPORTS_URSCRIPT = False
    GRIPPER_TYPE = "franka_hand"

    HOME_CONFIG = list(HOME_CONFIG)
    JOINT_LIMITS = list(JOINT_LIMITS)
    DEFAULT_VELOCITY = DEFAULT_VELOCITY
    DEFAULT_JOINT_VEL = DEFAULT_JOINT_VEL

    def __init__(self, robot_ip: str, frequency: float = 1000.0):
        self.robot_ip = robot_ip
        self.frequency = frequency
        self._connected = False
        self._has_errors = False

    @property
    def connected(self) -> bool:
        return self._connected

    # Abstract: subclasses must implement

    @abstractmethod
    def connect(self): ...

    @abstractmethod
    def disconnect(self): ...

    @abstractmethod
    def get_joint_positions(self) -> np.ndarray: ...

    @abstractmethod
    def get_tcp_pose(self) -> np.ndarray: ...

    @abstractmethod
    def move_to_joint_config(
        self,
        q: Sequence[float],
        velocity: Optional[float] = None,
        acceleration: Optional[float] = None,
        asynchronous: bool = False,
    ): ...

    @abstractmethod
    def move_linear(
        self,
        pose: Sequence[float],
        velocity: Optional[float] = None,
        acceleration: Optional[float] = None,
        asynchronous: bool = False,
    ): ...

    @abstractmethod
    def stop(self): ...

    # Abstract: gripper hardware access (differs per backend)

    @abstractmethod
    def _gripper_open(self, speed: float): ...

    @abstractmethod
    def _gripper_grasp(
        self,
        width: float,
        speed: float,
        force: float,
        epsilon_inner: float,
        epsilon_outer: float,
    ) -> bool: ...

    @abstractmethod
    def _gripper_width(self) -> float: ...

    @abstractmethod
    def _gripper_is_grasped(self) -> bool: ...

    @abstractmethod
    def _gripper_max_width(self) -> float: ...

    @abstractmethod
    def _gripper_homing(self): ...

    # Shared implementations

    def go_home(self, velocity: Optional[float] = None):
        self.move_to_joint_config(self.HOME_CONFIG, velocity=velocity)

    def recover_from_errors(self):
        self._has_errors = False
        return True

    def get_observation(self) -> dict:
        obs = {}
        for fn_name, key in [
            ("get_joint_positions", "joint_positions"),
            ("get_tcp_pose", "tcp_pose"),
            ("get_gripper_position", "gripper_position"),
        ]:
            fn = getattr(self, fn_name, None)
            if callable(fn):
                try:
                    obs[key] = fn()
                except Exception:
                    pass
        obs.setdefault("gripper_position", 0.0)
        return obs

    def get_tcp_force(self) -> np.ndarray:
        return np.zeros(6)

    def activate_gripper(self):
        self._gripper_homing()
        logger.info("Gripper homed")

    def _cfg_gripper(self, key: str, constant: float) -> float:
        """
        Return the RobotProfile gripper value for key, else the module constant.
        """
        cfg = getattr(self, "_gripper_cfg", None) or {}
        val = cfg.get(key)
        if val is None:
            return float(constant)
        return float(val)

    def open_gripper(
        self, speed: Optional[float] = None, force: Optional[float] = None
    ):
        spd = (
            speed
            if speed is not None
            else self._cfg_gripper("speed", GRIPPER_DEFAULT_SPEED)
        )
        self._gripper_open(spd)

    def close_gripper(
        self, speed: Optional[float] = None, force: Optional[float] = None
    ):
        self._send_grasp(0.0, speed, force)

    def set_gripper_position(
        self,
        position: float,
        speed: Optional[float] = None,
        force: Optional[float] = None,
    ):
        pos = float(np.clip(position, 0.0, 1.0))
        if pos < 0.5:
            self.open_gripper(speed=speed)
        else:
            self.close_gripper(speed=speed, force=force)

    def _send_grasp(self, width: float, speed: Optional[float], force: Optional[float]):
        spd = (
            speed
            if speed is not None
            else self._cfg_gripper("speed", GRIPPER_DEFAULT_SPEED)
        )
        frc = (
            force
            if force is not None
            else self._cfg_gripper("force", GRIPPER_DEFAULT_FORCE)
        )
        max_w = self._gripper_max_width()
        self._gripper_grasp(
            float(width), float(spd), float(frc), float(max_w), float(max_w)
        )

    def grasp_to_width(
        self,
        width: float,
        force: Optional[float] = None,
        speed: Optional[float] = None,
        epsilon_inner: Optional[float] = None,
        epsilon_outer: Optional[float] = None,
    ) -> bool:
        spd = (
            speed
            if speed is not None
            else self._cfg_gripper("speed", GRIPPER_DEFAULT_SPEED)
        )
        frc = (
            force
            if force is not None
            else self._cfg_gripper("force", GRIPPER_DEFAULT_FORCE)
        )
        max_w = self._gripper_max_width()
        # Profile epsilon (mm tolerance) when set; else default to full jaw width.
        eps_in = (
            epsilon_inner
            if epsilon_inner is not None
            else self._cfg_gripper("epsilon_inner", max_w)
        )
        eps_out = (
            epsilon_outer
            if epsilon_outer is not None
            else self._cfg_gripper("epsilon_outer", max_w)
        )
        try:
            return self._gripper_grasp(
                float(width), float(spd), float(frc), float(eps_in), float(eps_out)
            )
        except Exception as e:
            logger.warning("grasp_to_width(%.3f) failed: %s", width, e)
            return False

    def get_gripper_width(self) -> float:
        try:
            return self._gripper_width()
        except Exception:
            return 0.0

    def get_gripper_position(self) -> float:
        w = self.get_gripper_width()
        max_w = self._gripper_max_width()
        if max_w < 1e-6:
            return 0.0
        return float(np.clip((1.0 - w / max_w) * 255.0, 0.0, 255.0))

    def is_object_detected(self) -> bool:
        try:
            return self._gripper_is_grasped()
        except Exception:
            return False

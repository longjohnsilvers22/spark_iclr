"""
VLA Policy Controller for real-time robot control.

Runs a VLA policy (e.g., pi0.5) in a control loop, reading camera
observations and outputting joint commands via ur_rtde servoJ.
"""

import logging
import time
from typing import Optional
from pathlib import Path

import cv2
import numpy as np

try:
    from openpi.models import load_policy
except ImportError:
    load_policy = None

try:
    from transformers import AutoModelForVision2Seq, AutoProcessor
except ImportError:
    AutoModelForVision2Seq = AutoProcessor = None

logger = logging.getLogger(__name__)


class VLAController:
    """
    Runs VLA policies in real-time control loop on UR10e.
    """

    # Safety limits
    MAX_JOINT_DELTA = 0.1  # rad per step
    MAX_TCP_DELTA = 0.02  # m per step
    WORKSPACE_BOUNDS = {
        "x": (-0.9, 0.9),
        "y": (-0.9, 0.9),
        "z": (0.01, 1.2),
    }

    def __init__(
        self,
        robot,
        policy_path: str = None,
        camera_ids: list = None,
        control_freq: float = 10.0,
        action_horizon: int = 16,
    ):
        """
        Args:
            robot: UR10eDriver instance (connected)
            policy_path: Path to VLA checkpoint
            camera_ids: List of USB camera device indices
            control_freq: Policy execution frequency (Hz)
            action_horizon: Number of future actions to predict (chunk size)
        """
        self.robot = robot
        self.policy_path = policy_path
        self.camera_ids = camera_ids or [0]
        self.control_freq = control_freq
        self.action_horizon = action_horizon
        self.dt = 1.0 / control_freq

        self._policy = None
        self._cameras = []
        self._running = False

    def load_policy(self, policy_type: str = "pi05"):
        """
        Load VLA policy.

        Args:
            policy_type: "pi05" for pi0.5, "openvla" for OpenVLA, "custom"
        """
        if policy_type == "pi05":
            self._policy = self._load_pi05()
        elif policy_type == "openvla":
            self._policy = self._load_openvla()
        else:
            raise ValueError(f"Unknown policy type: {policy_type}")

        logger.info("Loaded %s policy from %s", policy_type, self.policy_path)

    def _load_pi05(self):
        """
        Load pi0.5 policy. Requires openpi installation.
        """
        if load_policy is None:
            logger.warning("openpi not installed. Install from ~/vla_interp/openpi/")
            return None
        return load_policy(self.policy_path)

    def _load_openvla(self):
        """
        Load OpenVLA policy.
        """
        if AutoModelForVision2Seq is None or AutoProcessor is None:
            logger.warning("transformers not installed for OpenVLA")
            return None
        processor = AutoProcessor.from_pretrained(
            self.policy_path, trust_remote_code=True
        )
        model = AutoModelForVision2Seq.from_pretrained(
            self.policy_path, trust_remote_code=True
        ).to("cuda")
        return {"model": model, "processor": processor}

    def setup_cameras(self):
        """
        Initialize USB cameras for observation.
        """
        self._cameras = []
        for cam_id in self.camera_ids:
            cap = cv2.VideoCapture(cam_id)
            if not cap.isOpened():
                logger.warning("Camera %d not available", cam_id)
                continue
            cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
            cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
            self._cameras.append(cap)
        logger.info("%d cameras initialized", len(self._cameras))

    def get_observation(self) -> dict:
        """
        Capture current observation from cameras and robot state.
        """
        obs = {
            "joint_positions": self.robot.get_joint_positions(),
            "tcp_pose": self.robot.get_tcp_pose(),
            "tcp_force": self.robot.get_tcp_force(),
            "images": [],
        }
        for cam in self._cameras:
            ret, frame = cam.read()
            if ret:
                obs["images"].append(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
        return obs

    def predict_actions(self, observation: dict, instruction: str) -> np.ndarray:
        """
        Run VLA policy to predict action chunk.

        Args:
            observation: Dict with images, joint_positions, etc.
            instruction: Language instruction

        Returns:
            Action chunk (action_horizon, action_dim) - joint deltas or absolute positions
        """
        if self._policy is None:
            raise RuntimeError("No policy loaded. Call load_policy() first.")

        # TODO: Implement actual policy inference based on policy_type
        logger.info("Predicting actions for: '%s'", instruction)
        return np.zeros((self.action_horizon, 7))  # 6 joints + 1 gripper

    def _check_safety(self, current_q: np.ndarray, target_q: np.ndarray) -> bool:
        """
        Check if proposed action is within safety limits.
        """
        delta = np.abs(target_q - current_q)
        if np.any(delta > self.MAX_JOINT_DELTA):
            logger.warning(
                "Joint delta %.4f exceeds limit %.4f", delta.max(), self.MAX_JOINT_DELTA
            )
            return False
        return True

    def run(self, instruction: str, max_steps: int = 1000):
        """
        Execute VLA policy for given instruction.

        Args:
            instruction: Natural language task instruction
            max_steps: Maximum control steps before timeout
        """
        logger.info("Executing: '%s'", instruction)
        self._running = True

        try:
            for step in range(max_steps):
                if not self._running:
                    break

                t_start = time.time()

                obs = self.get_observation()
                actions = self.predict_actions(obs, instruction)

                # Execute action chunk with interpolation
                for action in actions:
                    if not self._running:
                        break

                    joint_target = action[:6]
                    gripper_target = action[6] if len(action) > 6 else None

                    current_q = self.robot.get_joint_positions()
                    if not self._check_safety(current_q, joint_target):
                        logger.warning("Safety limit reached, stopping")
                        self._running = False
                        break

                    self.robot.servo_joint(
                        joint_target.tolist(), dt=self.dt, lookahead_time=0.1, gain=300
                    )

                    if gripper_target is not None:
                        gripper_pos = np.clip(gripper_target, 0, 255)
                        self.robot.set_gripper_position(float(gripper_pos))

                    # Maintain control frequency
                    elapsed = time.time() - t_start
                    if elapsed < self.dt:
                        time.sleep(self.dt - elapsed)

        except KeyboardInterrupt:
            logger.info("Interrupted by user")
        finally:
            self.robot.servo_stop()
            self._running = False
            logger.info("Execution complete")

    def stop(self):
        """
        Stop the control loop.
        """
        self._running = False

    def cleanup(self):
        """
        Release cameras and stop robot.
        """
        self.stop()
        for cam in self._cameras:
            cam.release()
        self._cameras = []

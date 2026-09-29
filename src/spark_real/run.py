#!/usr/bin/env python3
"""
SPARK Real - Main entry point for real-world robot control.

Usage (tyro names the subcommands after the mode classes; --robot is required):
    python -m spark_real.run test-mode --robot ur10e --robot.ip 192.168.1.100
    python -m spark_real.run planner-mode --robot ur10e --robot.ip 192.168.1.100 \
        --instruction "pick up the red block"
    python -m spark_real.run vla-mode --robot franka --policy-path /path/to/checkpoint
    python -m spark_real.run planner-mode --robot franka --dry-run \
        --instruction "pick up the red block"
"""

from __future__ import annotations

import logging
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Annotated

import numpy as np
import tyro
import yaml

from spark_real.config import SparkConfig, load_profile
from spark_real.control.vla_controller import VLAController
from spark_real.planning.spark_planner import SPARKPlanner
from spark_real.robots.factory import make_robot_driver

logger = logging.getLogger(__name__)

# Control-loop rate (Hz) is a run-mode concern that isn't part of
# SparkConfig (which selects the robot family + deployment). None lets
# make_robot_driver pick the per-family default: UR10e 500, Franka 1000,
# G1 from the SDK.
_DEFAULT_FREQUENCY: float | None = None


def _open_driver(cfg: SparkConfig, frequency: float | None):
    # Resolve the per-family IP (YAML value, then fallback, with --ip as an
    # override) through the shared profile loader so run.py and server.py
    # agree on addresses instead of carrying a separate hardcoded default.
    robot_ip = load_profile(cfg).resolved_ip()
    return make_robot_driver(cfg.family, robot_ip, frequency)


@dataclass
class TestMode:
    """
    Interactive test mode - move robot, test gripper, read state.
    """

    robot: SparkConfig
    frequency: float | None = _DEFAULT_FREQUENCY
    """
    Control loop rate (Hz). None uses the per-family driver default.
    """

    def run(self):
        with _open_driver(self.robot, self.frequency) as robot:
            _interactive_loop(robot)


@dataclass
class PlannerMode:
    """
    SPARK planner mode - LLM generates behavior tree, execute primitives.
    """

    robot: SparkConfig
    instruction: str = ""
    """
    Natural language task instruction.
    """
    dry_run: bool = False
    """
    Plan only, don't execute on robot.
    """
    llm_backend: str = "gemini"
    """
    LLM backend: gemini or openai.
    """
    frequency: float | None = _DEFAULT_FREQUENCY
    """
    Control loop rate (Hz). None uses the per-family driver default.
    """

    def run(self):
        # The planner prompt has per-family sections; without the family it
        # would plan for its own default (ur10e) whatever robot was launched.
        planner = SPARKPlanner(
            llm_backend=self.llm_backend, robot_family=self.robot.family
        )
        score = planner.generate_score(self.instruction)
        issues = planner.validate_score(score)
        if issues:
            logger.warning("Score validation issues: %s", issues)
            return

        logger.info("Generated score:\n%s", yaml.dump(score, default_flow_style=False))

        if self.dry_run:
            logger.info("Dry run; would execute the above score on the robot")
            return

        with _open_driver(self.robot, self.frequency) as robot:
            _execute_score(robot, score)


@dataclass
class VLAMode:
    """
    VLA policy mode - end-to-end learned control.
    """

    robot: SparkConfig
    instruction: str = ""
    """
    Task instruction for the policy.
    """
    policy_path: str = ""
    """
    Path to VLA checkpoint.
    """
    policy_type: str = "pi05"
    """
    Policy type: pi05, openvla, or custom.
    """
    control_freq: float = 10.0
    """
    Policy execution frequency (Hz).
    """
    camera_ids: list[int] = field(default_factory=lambda: [0])
    """
    USB camera device indices.
    """
    frequency: float | None = _DEFAULT_FREQUENCY
    """
    Control loop rate (Hz). None uses the per-family driver default.
    """

    def run(self):
        with _open_driver(self.robot, self.frequency) as robot:
            controller = VLAController(
                robot=robot,
                policy_path=self.policy_path,
                camera_ids=self.camera_ids,
                control_freq=self.control_freq,
            )
            controller.load_policy(self.policy_type)
            controller.setup_cameras()
            controller.run(self.instruction)


def _interactive_loop(robot):
    """
    Interactive test mode control loop.
    """
    print("\nSPARK Real - Test Mode")
    print("Commands: home, pose, joints, open, close, move <x> <y> <z>, force, quit")

    while True:
        try:
            cmd = input("\n> ").strip().lower()
        except (EOFError, KeyboardInterrupt):
            break

        if cmd == "quit":
            break
        elif cmd == "home":
            robot.go_home()
            print("Moved to home position")
        elif cmd == "pose":
            pose = robot.get_tcp_pose()
            print(f"TCP: x={pose[0]:.4f} y={pose[1]:.4f} z={pose[2]:.4f}")
            print(f"rx={pose[3]:.4f} ry={pose[4]:.4f} rz={pose[5]:.4f}")
        elif cmd == "joints":
            q = robot.get_joint_positions()
            for i, val in enumerate(q):
                print(f"J{i}: {val:.4f} rad ({val*180/np.pi:.1f} deg)")
        elif cmd == "open":
            robot.open_gripper()
        elif cmd == "close":
            robot.close_gripper()
        elif cmd.startswith("move"):
            parts = cmd.split()
            if len(parts) == 4:
                dx, dy, dz = float(parts[1]), float(parts[2]), float(parts[3])
                robot.move_linear_relative([dx, dy, dz, 0, 0, 0])
            else:
                print("Usage: move <dx> <dy> <dz>")
        elif cmd == "force":
            ft = robot.get_tcp_force()
            print(f"Force:  Fx={ft[0]:.2f} Fy={ft[1]:.2f} Fz={ft[2]:.2f}")
            print(f"Torque: Tx={ft[3]:.2f} Ty={ft[4]:.2f} Tz={ft[5]:.2f}")


def _execute_score(robot, score: dict):
    """
    Execute a SPARK score on real robot.
    """
    tree = score.get("tree", {})
    for child in tree.get("children", []):
        action_type = child.get("type")
        params = child.get("params", {})

        if action_type == "grasp":
            robot.close_gripper(force=params.get("force", 100))
        elif action_type == "release":
            robot.open_gripper()
        elif action_type == "move_relative":
            dx, dy, dz = params.get("dx", 0), params.get("dy", 0), params.get("dz", 0)
            robot.move_linear_relative([dx, dy, dz, 0, 0, 0])
        elif action_type == "wait":
            time.sleep(params.get("duration", 1.0))
        elif action_type == "move_to_keypoint":
            logger.info(
                "move_to_keypoint: %s (needs perception)", params.get("keypoint_label")
            )


def main():
    mode = tyro.cli(TestMode | PlannerMode | VLAMode)
    mode.run()


if __name__ == "__main__":
    main()

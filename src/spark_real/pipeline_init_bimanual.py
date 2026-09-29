"""
Bimanual Franka bring-up for SPARKRealPipeline.

This module holds the self-contained bimanual initialization split out
of pipeline_init.py to keep the single-arm bring-up small. The mixin is
added to the SPARKRealPipeline base list alongside InitMixin, so the
family=='bimanual_franka' branch in _init_robot resolves
self._init_bimanual_franka() via the method resolution order.

The bring-up wires together the BimanualFrankaDriver, the
BimanualSafeRobot safety wrapper, the BimanualScoreExecutor, the
role-keyed bimanual camera registry, and an optional ZMQ state
broadcaster. Configuration comes from the threaded RobotProfile
(family default deep-merged with an optional machine overlay) so per-arm
IPs, gripper backend, workspace boxes, inter-arm CBF gains, and the
camera roster all read from one source. See docs/bimanual_design.md for
the contract.
"""

import logging

import numpy as np

from spark_real.config import family_block
from spark_real.control.bimanual_executor import BimanualScoreExecutor
from spark_real.control.bimanual_safe_robot import (
    BaseFrames,
    BimanualSafeRobot,
    InterArmConfig,
)
from spark_real.control.safe_robot import SafetyConfig
from spark_real.perception.bimanual_camera_registry import (
    build_bimanual_camera_registry,
)
from spark_real.robots.bimanual_franka import (
    BimanualFrankaDriver,
    DEFAULT_LEFT_IP,
    DEFAULT_RIGHT_IP,
    make_dual_gripper,
)

try:
    from spark_real.comms.bimanual_zmq import BimanualStateBroadcaster
except ImportError:
    BimanualStateBroadcaster = None

logger = logging.getLogger(__name__)


# Bimanual Franka init, kept on its own mixin so the legacy single-arm
# _init_robot stays small and the bimanual branch can grow without
# scrolling. Mixed into the pipeline class; uses attributes set in
# __init__.
class BimanualInitMixin:

    def _init_bimanual_franka(self):
        """
        Bring up the bimanual stack: driver + safe wrapper + executor +
        camera registry + optional ZMQ state broadcast.

        Config comes from the threaded RobotProfile (family default deep-
        merged with the optional configs/machines/<machine>.yaml overlay,
        e.g. --machine ANON-LAB for the Panda+FR3 SSG-48 rig) for per-arm IPs,
        gripper backend, workspace boxes, inter-arm CBF gains, and camera
        roster. See docs/bimanual_design.md for the contract.
        """
        # The profile's merged yaml (default + machine overlay), else the
        # packaged default. The camera registry reads the same mapping.
        cfg = family_block(self.profile, "bimanual_franka")
        robot_cfg = cfg.get("robot", {}) or {}
        gripper_cfg = cfg.get("grippers", {}) or {}
        bcast_cfg = cfg.get("state_broadcast", {}) or {}

        # Gripper backend dispatched from grippers.type ("dynamixel" = UCLA
        # ALOHA jaws, "ssg48" = ANON-LAB Source Robotics jaws on dual slcan
        # buses). No type declared keeps the legacy Dynamixel kwargs below.
        dual_gripper = None
        if gripper_cfg.get("type"):
            dual_gripper = make_dual_gripper(gripper_cfg)

        # Construct + connect the driver.
        left_ip = str(robot_cfg.get("left_ip", DEFAULT_LEFT_IP))
        right_ip = str(robot_cfg.get("right_ip", DEFAULT_RIGHT_IP))
        # Arm-motion backend: "franky" (libfranka motion generators) or
        # "bamboo" (the validated ANON-LAB C++ joint-impedance controller).
        # Selectable from the machine overlay's robot.backend key; defaults
        # to franky so the UCLA default rig is unchanged.
        backend = str(robot_cfg.get("backend", "franky")).lower()
        driver = BimanualFrankaDriver(
            left_ip=left_ip,
            right_ip=right_ip,
            backend=backend,
            frequency=float(robot_cfg.get("frequency", 1000.0)),
            left_gripper_id=int(gripper_cfg.get("left_id", 1)),
            right_gripper_id=int(gripper_cfg.get("right_id", 2)),
            gripper_port=str(gripper_cfg.get("port", "/dev/ttyUSB0")),
            gripper_baud=int(gripper_cfg.get("baudrate", 1_000_000)),
            dual_gripper=dual_gripper,
            # The standalone broadcaster below owns the state port; letting
            # the driver also bind it raises "Address already in use".
            publish_state=False,
            publish_port=int(bcast_cfg.get("port", 5601)),
        )
        driver.connect()
        self._raw_driver = driver  # bare driver (no safety wrapper)

        # Per-arm SafetyConfig from the per-arm workspace cuboids.
        def _make_arm_safety(ws_dict):
            return SafetyConfig(
                vel_lin_max=float(robot_cfg.get("max_linear_velocity", 0.15)) + 0.05,
                ws_min=np.array(
                    [
                        ws_dict.get("x_min", -0.85),
                        ws_dict.get("y_min", -0.85),
                        ws_dict.get("z_min", 0.0),
                    ],
                    dtype=float,
                ),
                ws_max=np.array(
                    [
                        ws_dict.get("x_max", 0.85),
                        ws_dict.get("y_max", 0.85),
                        ws_dict.get("z_max", 1.20),
                    ],
                    dtype=float,
                ),
                reach_max=float(ws_dict.get("reach_max", 0.855)),
                eps_singularity=0.0,  # disable UR-tuned check on FR3
                eta_kinematic=float(ws_dict.get("eta_kinematic", 5.0)),
            )

        left_sc = _make_arm_safety(cfg.get("workspace_left", {}) or {})
        right_sc = _make_arm_safety(cfg.get("workspace_right", {}) or {})
        inter_cfg_dict = cfg.get("inter_arm", {}) or {}
        inter_cfg = InterArmConfig(
            dist_hard_m=float(inter_cfg_dict.get("dist_hard_m", 0.12)),
            dist_soft_m=float(inter_cfg_dict.get("dist_soft_m", 0.20)),
            eta_kinematic=float(inter_cfg_dict.get("eta_kinematic", 8.0)),
        )
        # Per-arm base SE3 offsets from the rig YAML (robot.base_left /
        # base_right). Without these BimanualSafeRobot defaults to identity
        # frames, so the inter-arm distance treats both bases as the origin
        # and the CBF margins are wrong by the inter-arm base offset. Wire
        # the real geometry in so the distance gate is meaningful; fall back
        # to the warn-and-identity path only if the YAML omits the base
        # blocks entirely.
        try:
            bases = BaseFrames.from_yaml_block(robot_cfg)
        except ValueError as exc:
            logger.warning(
                "bimanual base offsets missing from config (%s); inter-arm "
                "distance will treat both bases as the origin",
                exc,
            )
            bases = None
        self._safe = BimanualSafeRobot(
            driver, left_sc, right_sc, inter_cfg, bases=bases
        )
        # Expose the safe wrapper as the canonical _robot attr so the
        # routes that read pipeline._robot.something don't break. The
        # SafeRobot interface is duck-typed; consumers that ask for
        # `get_tcp_pose()` without an arm will hit BimanualFrankaDriver's
        # error path, which is the correct behaviour for the bimanual
        # family.
        self._robot = self._safe

        self._executor = BimanualScoreExecutor(
            self._safe,
            velocity=self.config.velocity,
            pipeline=self,
            strict_placement_verify=self.config.strict_placement_verify,
        )
        # Hand the resolved RobotProfile to the bimanual executor for
        # consistency with the single-arm path.
        self._executor._profile = self.profile
        # The per-arm executors applied workspace bounds in __init__ before
        # _profile existed and with no arm, so they kept the UR10e box. Name
        # the arm, wire the profile, and re-apply so each reads its own
        # workspace_<arm> box, table_z_floor and grasp orientation.
        for arm, ex in self._executor._arm_executors.items():
            ex.arm = arm
            ex._profile = self.profile
            ex._apply_family_workspace_bounds()

        # Camera registry replaces the three named slots for the
        # bimanual pipeline. Streaming/visualization consult
        # `pipeline.camera_registry` before falling back to the legacy
        # slots, so this is the single integration point.
        try:
            self.camera_registry = build_bimanual_camera_registry(cfg, auto_open=True)
            # Map wrist-camera roles back into the slot-style attributes
            # any single-arm code path still queries (e.g. perception's
            # SAM3 wrist refinement). Keep these as references into the
            # registry rather than new opens so close() is centralized.
            ext = self.camera_registry.get("external")
            wl = self.camera_registry.get("wrist_left")
            wr = self.camera_registry.get("wrist_right")
            self._kinect = ext.device if ext else None
            self._kinect_cal = ext.calibration if ext else None
            self._kinect2 = None  # bimanual has no "birdview" slot
            self._kinect2_cal = None
            # Default the legacy `_realsense` slot to the right wrist so
            # downstream code that hardcodes wrist refinement still works
            # in single-arm-style operation.
            self._realsense = wr.device if wr else (wl.device if wl else None)
            self._realsense_cal = (
                wr.calibration if wr else (wl.calibration if wl else None)
            )
            # Convenience: per-arm wrist resolver.
            self._wrist_for_arm = dict(self.camera_registry.wrist_for_arm)
        except Exception:
            logger.exception(
                "bimanual camera registry build failed; "
                "continuing without external cameras"
            )
            self.camera_registry = None
            self._wrist_for_arm = {}

        # Optional ZMQ state broadcaster. The driver also opens a PUB
        # socket when constructed with publish_state=True, so this is
        # the more flexible standalone publisher used by external nodes.
        if bcast_cfg.get("enabled"):
            try:
                if BimanualStateBroadcaster is None:
                    raise ImportError("BimanualStateBroadcaster unavailable")
                self._state_broadcaster = BimanualStateBroadcaster(
                    driver,
                    port=int(bcast_cfg.get("port", 5601)),
                    rate_hz=float(bcast_cfg.get("rate_hz", 30.0)),
                )
                self._state_broadcaster.start()
            except Exception:
                logger.exception("bimanual state broadcaster failed to start")

        logger.info(
            "Bimanual Franka connected (backend=%s left=%s right=%s); "
            "BimanualSafeRobot + BimanualScoreExecutor ready",
            backend,
            left_ip,
            right_ip,
        )

import json
import logging
import os
import sys
import time
from pathlib import Path

import numpy as np

from spark_real.config import family_block
from spark_real.perception.camera import (
    AzureKinectCamera,
)
from spark_real.perception.camera_registry import registry_from_open_devices
from spark_real.perception.bimanual_camera_registry import CameraEntry
from spark_real.robots.factory import make_robot_driver
from spark_real.control.score_executor import ScoreExecutor
from spark_real.control.safe_robot import SafeRobot, SafetyConfig
from spark_real.calibration import CameraCalibration

logger = logging.getLogger(__name__)

# External UR10e teleop deploy tree (provides cameras.*, UR10eRobot, etc.).
# Override with SPARK_TELEOP_DEPLOY; defaults to a neutral home subpath that
# the rig host symlinks to the actual deploy checkout.
_TELEOP_DEPLOY_BASE = Path(
    os.environ.get(
        "SPARK_TELEOP_DEPLOY", str(Path.home() / "external_deploy/teleop/ur10e_deploy")
    )
)
_teleop_path = str(_TELEOP_DEPLOY_BASE)
if Path(_teleop_path).exists():
    sys.path.insert(0, _teleop_path)
try:
    from cameras.azure_kinect_camera import (
        MultiAzureKinectManager,
        AzureKinectCamera as TeleopK4A,
    )

    HAS_MULTI_K4A = True
except ImportError:
    HAS_MULTI_K4A = False

try:
    import pyk4a

    HAS_PYK4A = True
except ImportError:
    HAS_PYK4A = False

try:
    from robot.ur10e_robot import UR10eRobot, UR10eConfig

    HAS_UR10E = True
except ImportError:
    HAS_UR10E = False


# Camera, calibration, and robot bring-up for SPARKRealPipeline.
# Mixed into the pipeline class; uses attributes set in __init__.
class InitMixin:

    def _resolve_wrist_tool_offset(self) -> np.ndarray:
        """
        Build the 4x4 TCP-to-camera transform, honoring config + family.

        Priority order:
          1. Explicit ``wrist_tool_offset_xyz`` in PipelineConfig
             (translation-only override).
          2. Measured hand-eye calibration at
             ``output/calibrations/handeye_wrist.json``: uses the FULL
             4x4 (rotation + translation). Mandatory for the Franka wrist
             RealSense, whose physical mount is rotated ~90 deg about Z
             relative to the TCP, not just translated.
          3. Family placeholder defaults (translation only, identity R).
        """
        offset_xyz = self.config.wrist_tool_offset_xyz
        if offset_xyz is not None:
            T = np.eye(4)
            T[0, 3] = float(offset_xyz[0])
            T[1, 3] = float(offset_xyz[1])
            T[2, 3] = float(offset_xyz[2])
            return T

        handeye_path = (
            Path(__file__).parent / "output" / "calibrations" / "handeye_wrist.json"
        )
        if handeye_path.exists():
            try:
                data = json.loads(handeye_path.read_text())
                # Accept both old AX=XB ("transform_4x4") and new RGB-D
                # Procrustes ("T_cam_to_base_4x4") output formats.
                T_key = (
                    "T_cam_to_base_4x4"
                    if "T_cam_to_base_4x4" in data
                    else "transform_4x4"
                )
                T = np.array(data[T_key], dtype=np.float64)
                if T.shape != (4, 4):
                    raise ValueError(f"{T_key} must be 4x4, got {T.shape}")
                # Stash the per-camera depth correction so it can be applied
                # in capture() once _realsense_cal is created. We can't set
                # _realsense_cal.depth_scale/offset here because that cal
                # object doesn't exist yet at pipeline init time.
                self._wrist_depth_scale = float(data.get("depth_scale_correction", 1.0))
                self._wrist_depth_offset = float(
                    data.get("depth_offset_correction", 0.0)
                )
                logger.info(
                    "wrist tool_offset: loaded from handeye_wrist.json "
                    "(rmse=%.1fmm, depth_scale=%.4f, depth_offset=%.1fmm, "
                    "pairs/poses=%d)",
                    float(data.get("rmse_mm", data.get("residual_mm", 0.0))),
                    self._wrist_depth_scale,
                    self._wrist_depth_offset * 1000.0,
                    int(data.get("num_pairs", data.get("num_poses", 0))),
                )
                return T
            except Exception as exc:
                logger.warning(
                    "wrist tool_offset: failed to load handeye_wrist.json "
                    "(%s); using default tool offset.",
                    exc,
                )

        family_defaults = {
            "ur10e": [0.05, 0.00, -0.10],
            "franka": [0.00, 0.06, -0.05],  # default until wrist hand-eye calibrated
            "g1": [0.00, 0.00, -0.05],
        }
        offset_xyz = family_defaults.get(
            self.config.robot_family, family_defaults["ur10e"]
        )
        if (
            self.config.robot_family == "franka"
            and self.config.use_realsense is not False
        ):
            logger.info(
                "wrist hand-eye calibration not found; using default tool "
                "offset=%s for Franka. Set `wrist_tool_offset_xyz: [x, y, z]` "
                "in configs/franka_default.yaml to override.",
                offset_xyz,
            )
        T = np.eye(4)
        T[0, 3] = float(offset_xyz[0])
        T[1, 3] = float(offset_xyz[1])
        T[2, 3] = float(offset_xyz[2])
        return T

    def _init_kinects(self):
        """
        Initialize Azure Kinect cameras.

        When ``config.kinect_use_hw_sync`` is True (3.5mm sync cable
        connected between two Kinects), the **subordinate** is opened
        FIRST in SUBORDINATE mode (it then waits silently for the master
        trigger), THEN the master is opened in MASTER mode and starts
        firing pulses. This is mandatory per the K4A SDK and also keeps
        USB bandwidth manageable when both Kinects share one host
        controller: the subordinate doesn't free-run.

        Without the sync cable (standalone), both Kinects run free and
        bandwidth must come from separate USB host controllers.
        """
        if not HAS_PYK4A:
            logger.warning("pyk4a not available, skipping Kinect init")
            return

        num_kinects = pyk4a.connected_device_count()
        logger.info("Found %d Azure Kinect device(s)", num_kinects)

        # Budget preflight for the LATE open path (server.py runs the same
        # check before its early open). Refuses a configuration that puts
        # more camera traffic on one shared xHCI controller than the profile
        # that repeatedly hard-killed this host.
        # See docs/MULTICAM_CRASH_MECHANISM.md.
        from spark_real import usb_budget as _ub

        _loads = [
            _ub.kinect_load(
                f"kinect{i}",
                self.config.kinect_resolution,
                self.config.kinect_depth_mode,
                self.config.kinect_fps,
            )
            for i in range(num_kinects)
        ]
        if getattr(self.config, "use_realsense", None) is not False:
            _loads.append(
                _ub.realsense_load(
                    "wrist realsense",
                    color_only=bool(
                        getattr(self.config, "realsense_color_only", True)
                    ),
                )
            )
        _ub.preflight(_loads, log=logger)

        master_serial = self.config.kinect_master_serial
        use_hw_sync = bool(getattr(self.config, "kinect_use_hw_sync", False))
        serial_map = {}  # serial -> device_id
        if HAS_MULTI_K4A:
            try:
                found = TeleopK4A.find_cameras()
                serial_map = {c["serial_number"]: c["device_id"] for c in found}
                logger.info("Camera serials: %s", serial_map)
            except Exception:
                pass

        # Determine open order: subordinates first when hw-sync is on so
        # they're listening before the master starts triggering them.
        device_ids = list(range(num_kinects))
        if use_hw_sync and serial_map and master_serial in serial_map:
            master_dev = serial_map[master_serial]
            device_ids = [d for d in device_ids if d != master_dev] + [master_dev]
            logger.info(
                "HW-sync ENABLED. Open order (sub first -> master last): %s", device_ids
            )

        for dev_id in device_ids:
            serial = None
            for sn, did in serial_map.items():
                if did == dev_id:
                    serial = sn
                    break
            # Try getting serial directly from pyk4a if not in map
            if serial is None:
                try:
                    dev = pyk4a.PyK4A(device_id=dev_id)
                    dev.open()
                    serial = dev.serial
                    dev.close()
                except Exception:
                    pass
            name = "sideview" if serial == master_serial else "birdview"
            if serial is None:
                # Last resort: device 1 is typically master on this system
                name = "birdview" if dev_id == 0 else "sideview"

            # In hw-sync: master is the named sideview, subordinate is birdview.
            # Standalone: every device is STANDALONE.
            if use_hw_sync:
                sync_mode = "MASTER" if name == "sideview" else "SUBORDINATE"
            else:
                sync_mode = "STANDALONE"

            logger.info(
                "Opening Kinect %d as %s (SN=%s, sync=%s)...",
                dev_id,
                name,
                serial or "?",
                sync_mode,
            )
            try:
                # Subordinate delay depends on the master's depth laser
                # pulse width, which differs by depth mode. NFOV pulses
                # are ~125us so 160us margin is fine; WFOV pulses are
                # nearly an order of magnitude longer, so use 1600us to
                # avoid cross-illumination causing dropped frames and
                # blocking open() forever in warmup.
                sub_delay = 0
                if sync_mode == "SUBORDINATE":
                    if self.config.kinect_depth_mode.startswith("WFOV"):
                        sub_delay = 1600
                    else:
                        sub_delay = 160
                cam = AzureKinectCamera(
                    device_id=dev_id,
                    color_resolution=self.config.kinect_resolution,
                    depth_mode=self.config.kinect_depth_mode,
                    sync_mode=sync_mode,
                    subordinate_delay_usec=sub_delay,
                    camera_fps=self.config.kinect_fps,
                )
                for attempt in range(2):
                    try:
                        cam.open()
                        break
                    except Exception:
                        if attempt == 0:
                            logger.warning(
                                "Kinect %d failed, retrying in 2s...", dev_id
                            )
                            time.sleep(2)
                        else:
                            raise
                # From here on `cam` is open. Any exception before the
                # outer except catches it would leak the handle, so
                # close it explicitly on failure.
                try:
                    cal = CameraCalibration(
                        name=name,
                        width=cam.config.width,
                        height=cam.config.height,
                        fx=cam.config.fx,
                        fy=cam.config.fy,
                        cx=cam.config.cx,
                        cy=cam.config.cy,
                        # Carry any config-level extrinsic across instead of
                        # silently dropping it: the fallback for a camera whose
                        # config carries an extrinsic but has no hand-eye file
                        # (_load_handeye_calibrations() overwrites cal.extrinsic
                        # when one exists).
                        extrinsic=getattr(cam.config, "extrinsic", None),
                    )
                    if name == "sideview":
                        self._kinect = cam
                        self._kinect_cal = cal
                    else:
                        self._kinect2 = cam
                        self._kinect2_cal = cal
                except Exception:
                    try:
                        cam.close()
                    except Exception:
                        pass
                    raise
                logger.info(
                    "Kinect %d (%s) ready (%dx%d, fx=%.1f)",
                    dev_id,
                    name,
                    cal.width,
                    cal.height,
                    cal.fx,
                )
                # Stagger camera starts to avoid USB contention
                time.sleep(3)
            except Exception as e:
                logger.warning("Kinect %d failed: %s", dev_id, e)

    def _load_handeye_calibrations(self):
        """
        Apply saved hand-eye calibrations to the live Kinect cals.

        scripts/handeye_calibrate.py writes the cam-to-base 4x4 transforms
        to ``output/calibrations/handeye_{birdview,sideview,wrist}.json``.
        Without this loader the Kinect cals stay at identity and detect()
        backprojects every pixel ~1 m below the table (the OpenGL fallback
        path triggers because ``intrinsic_matrix`` is gated on
        ``not np.allclose(extrinsic, np.eye(4))``).

        Wrist is intentionally skipped here: its extrinsic is recomputed
        every frame from the live TCP pose in ``capture()`` so a fixed
        cam-to-base load would be wrong the instant the arm moves.
        """
        cal_dir = Path(__file__).parent / "output" / "calibrations"
        targets = [
            ("birdview", "handeye_birdview.json", self._kinect2_cal),
            ("sideview", "handeye_sideview.json", self._kinect_cal),
        ]
        for name, fname, cal in targets:
            if cal is None:
                continue
            path = cal_dir / fname
            if not path.exists():
                logger.info("hand-eye: no %s file at %s", name, path)
                continue
            try:
                data = json.loads(path.read_text())
                # Accept either format:
                #   {transform_4x4: [[...]], ...}              (legacy
                #     AX=XB hand-eye output)
                #   {T_cam_to_base_4x4: [[...]],
                #    depth_scale_correction: 0.85, ...}        (new
                #     RGB-D Procrustes output from spark_calibrate.py)
                T_key = (
                    "T_cam_to_base_4x4"
                    if "T_cam_to_base_4x4" in data
                    else "transform_4x4"
                )
                T = np.array(data[T_key], dtype=np.float64)
                if T.shape != (4, 4):
                    raise ValueError(f"{T_key} must be 4x4, got {T.shape}")
                cal.extrinsic = T
                depth_scale = float(data.get("depth_scale_correction", 1.0))
                depth_offset = float(data.get("depth_offset_correction", 0.0))
                cal.depth_scale = depth_scale
                cal.depth_offset = depth_offset
                logger.info(
                    "hand-eye: applied %s extrinsic from %s "
                    "(residual=%.1fmm, depth_scale=%.4f, depth_offset=%.1fmm, "
                    "poses/corners=%d)",
                    name,
                    fname,
                    float(data.get("rmse_mm", data.get("residual_mm", 0.0))),
                    depth_scale,
                    depth_offset * 1000.0,
                    int(data.get("num_pairs", data.get("num_poses", 0))),
                )
            except Exception as exc:
                logger.warning(
                    "hand-eye: failed to load %s from %s: %s", name, fname, exc
                )

    def _build_unified_camera_registry(self):
        # Wrap the already-open single-arm cameras in one role-keyed registry.
        # The dual-Kinect open in _init_kinects() carries master/subordinate
        # sync, depth-mode, retry, and stagger semantics that depend on
        # runtime device enumeration, so do NOT re-open through the
        # config-driven builder here: wrap the live devices and their
        # calibrated CameraCalibration objects so the registry entries and the
        # _kinect / _kinect2 / _realsense slots reference the same objects.
        # The bimanual family builds its own `camera_registry` in
        # _init_bimanual_franka; skip it here.
        if (
            getattr(self.config, "robot_family", "") or ""
        ).lower() == "bimanual_franka":
            return

        entries = []
        # sideview = master Kinect (_kinect); birdview = subordinate (_kinect2);
        # wrist = RealSense (_realsense). This is the slot<->role mapping from
        # _init_kinects (name == "sideview" iff serial == kinect_master_serial).
        slot_roles = [
            ("sideview", self._kinect, self._kinect_cal, "kinect"),
            ("birdview", self._kinect2, self._kinect2_cal, "kinect"),
            ("wrist", self._realsense, self._realsense_cal, "realsense"),
        ]
        for role, device, cal, kind in slot_roles:
            if device is None or cal is None:
                continue
            entries.append(
                CameraEntry(
                    role=role,
                    kind=kind,
                    device=device,
                    calibration=cal,
                    serial=getattr(device, "serial", "") or "",
                )
            )

        # `sideview` (master Kinect) is the workspace anchor for single-arm
        # rigs, matching camera_registry._resolve_primary_role.
        primary = (
            "sideview"
            if any(e.role == "sideview" for e in entries)
            else (entries[0].role if entries else "")
        )
        self._camera_registry = registry_from_open_devices(
            entries, primary_role=primary
        )

        # Re-point the legacy slots FROM the registry (identity reassignments;
        # the entries wrap the same device + cal objects).
        # The unified registry is stored under a PRIVATE name on purpose:
        # routes/streaming.py keys its bimanual dispatch off the public
        # `camera_registry` attribute, and single-arm streaming must keep
        # using the per-device read-lock path, so single-arm pipelines must
        # NOT expose a public `camera_registry`.
        sv = self._camera_registry.get("sideview")
        bv = self._camera_registry.get("birdview")
        wr = self._camera_registry.get("wrist")
        if sv is not None:
            self._kinect = sv.device
            self._kinect_cal = sv.calibration
        if bv is not None:
            self._kinect2 = bv.device
            self._kinect2_cal = bv.calibration
        if wr is not None:
            self._realsense = wr.device
            self._realsense_cal = wr.calibration
        logger.info(
            "Unified camera registry built: roles=%s primary=%s",
            self._camera_registry.roles,
            self._camera_registry.primary_role,
        )

    def _init_robot(self):
        """
        Initialize robot connection.

        All real families now route through the ``make_robot_driver``
        factory, which returns a driver implementing the UR10eDriver shim
        interface SafeRobot + ScoreExecutor expect. UR10e is first-class
        here: the in-repo ``UR10eDriver`` (control/ur10e_driver.py) carries
        the full surface (URScript speedl via _send_script, send_velocity,
        get_observation, grasp_to_width, GRIPPER_TYPE, SUPPORTS_URSCRIPT) so
        a fresh UR10e host deploys out-of-box with only ``pip install
        ur-rtde`` + hand-eye calibration.

        The external teleop ``UR10eRobot`` survives only as an explicit
        fallback when the factory construction raises (see
        ``_init_ur10e_legacy``), with a deprecation log line.
        """
        # Reconnect-safety: if a previous driver exists, tear it down
        # BEFORE reassigning self._robot. Otherwise the old FrankaDriver
        # gets GC'd while franky's motion thread is still joinable and
        # libfranka's std::thread dtor calls std::terminate, surfacing as
        # a silent server death mid-teleop. SafeRobot proxies disconnect()
        # to the wrapped driver via __getattr__.
        if getattr(self, "_robot", None) is not None:
            try:
                self._robot.disconnect()
            except Exception as exc:
                logger.warning(
                    "previous robot driver disconnect raised: %s "
                    "(continuing with reconnect)",
                    exc,
                )
            self._robot = None

        family = (getattr(self.config, "robot_family", "ur10e") or "ur10e").lower()

        def _family_safety_bounds(fam: str):
            """
            Per-family workspace box + reach + singularity tuning.

            UR10e is bench-mounted with the table edge as the negative-x
            workspace boundary; Franka FR3 sits in front of the user and
            its box reflects the FR3 reach sphere. Defaults come from
            configs/<family>_default.yaml's workspace section when present,
            with the per-family fallback below otherwise.
            """
            ws = family_block(self.profile, fam, "workspace")
            # Per-family fallback defaults if the YAML has no workspace.
            # Tuple: (ws_min, ws_max, reach_max, eps_singularity, eta_kinematic).
            #   eps_singularity > 0 enables the UR-tuned elbow check
            #   (formula q2(q2-pi)); for Franka/G1 this check is wrong
            #   so we pass 0 to disable it (workspace + reach barriers
            #   still active).
            #   eta_kinematic is the CBF gain on the workspace box. The
            #   barrier caps velocity toward a bound at eta * distance,
            #   so a low eta makes teleop crawl near the bound (e.g.
            #   approaching a tabletop). UR10e's roomy cell tolerates a
            #   low eta; Franka's tight z_min needs a much higher eta so
            #   teleop stays at full speed until the last centimetre or two.
            family_defaults = {
                "ur10e": ((-1.1, -0.5, -0.25), (-0.5, 0.7, 0.50), 1.30, 0.20, 0.3),
                # SafeRobot barrier floor; the executor IK clip in executor_core is -0.05 and is a different bound
                "franka": ((-0.85, -0.85, 0.01), (0.85, 0.85, 1.20), 0.855, 0.0, 5.0),
                "g1": ((-0.7, -0.7, -0.20), (0.7, 0.7, 1.50), 0.90, 0.0, 0.3),
            }
            fb = family_defaults.get(fam, family_defaults["ur10e"])
            fb_min, fb_max, fb_reach, fb_sing, fb_eta = fb
            ws_min = np.array(
                [
                    float(ws.get("x_min", fb_min[0])),
                    float(ws.get("y_min", fb_min[1])),
                    float(ws.get("z_min", fb_min[2])),
                ]
            )
            ws_max = np.array(
                [
                    float(ws.get("x_max", fb_max[0])),
                    float(ws.get("y_max", fb_max[1])),
                    float(ws.get("z_max", fb_max[2])),
                ]
            )
            reach_max = float(ws.get("reach_max", fb_reach))
            eps_sing = float(ws.get("eps_singularity", fb_sing))
            eta_kin = float(ws.get("eta_kinematic", fb_eta))
            return ws_min, ws_max, reach_max, eps_sing, eta_kin

        if family == "bimanual_franka":
            # Bimanual family: separate driver + safe-robot + executor classes
            # because the single-arm SafeRobot / ScoreExecutor pair is built
            # around a single TCP. See docs/bimanual_design.md.
            try:
                self._init_bimanual_franka()
            except Exception as e:
                logger.warning(
                    "Bimanual Franka connection failed: %s "
                    "(running perception-only)",
                    e,
                )
                self._robot = None
                self._executor = None
            return

        # Multi-embodiment path: build the driver through the factory.
        # UR10e, Franka, and G1 all flow here. SafeRobot wrapping is
        # family-specific: UR10e keeps the CBF filter ON (its elbow
        # singularity check + workspace box are real and tuned), Franka
        # keeps it OFF (the CBF clamps direct joint moves unpredictably;
        # collision thresholds + ScoreExecutor checks cover it).
        try:
            # Franka single-arm reads its gripper + collision thresholds
            # from the RobotProfile. The factory forwards these kwargs to
            # FrankaDriver; other families (e.g. ur10e, g1) ignore them so
            # we only pass them for franka.
            driver_kwargs = {}
            if family == "franka" and self.profile is not None:
                driver_kwargs["gripper"] = self.profile.gripper()
                driver_kwargs["collision"] = self.profile.collision_behavior()
            self._robot = make_robot_driver(
                family,
                self.config.robot_ip,
                frequency=None,  # let the driver pick its native rate
                **driver_kwargs,
            )
            self._robot.connect()

            if family == "ur10e":
                # Legacy-parity SafeRobot wiring for UR10e. The SafeRobot box
                # comes from the same ur10e_default.yaml control: block the
                # ScoreExecutor reads (via _apply_ur10e_control_bounds), so the
                # filter box and the executor box stay one source. Fallbacks
                # are byte-identical to the historical literals.
                _ur_ctrl = (
                    self.profile.control() if self.profile is not None else {}
                ) or {}
                _ur_ws_min = _ur_ctrl.get("workspace_min", [-1.1, -0.5, -0.27])
                _ur_ws_max = _ur_ctrl.get("workspace_max", [-0.5, 0.7, 0.50])
                safety_cfg = SafetyConfig(
                    vel_lin_max=self.config.velocity + 0.05,
                    ws_min=np.array([float(v) for v in _ur_ws_min]),
                    ws_max=np.array([float(v) for v in _ur_ws_max]),
                )
                self._robot = SafeRobot(
                    self._robot,
                    enable_obstacles=False,  # obstacles set from detection map
                    enable_force=False,  # noisy F/T causes jiggling
                    config=safety_cfg,
                )
                self._executor = ScoreExecutor(
                    robot=self._robot,
                    velocity=self.config.velocity,
                    safe_height=self.config.safe_height,
                    pipeline=self,
                    strict_placement_verify=self.config.strict_placement_verify,
                    grasp_calibration=self._grasp_calibration,
                )
                # _apply_family_workspace_bounds() dispatches ur10e to
                # _apply_ur10e_control_bounds(), which reads the control: block
                # (the executor/SafeRobot box, distinct from workspace:).
                self._executor._profile = self.profile
                self._executor._apply_family_workspace_bounds()
                logger.info(
                    "UR10e connected at %s via factory "
                    "(in-repo UR10eDriver, SafeRobot enabled, force off)",
                    self.config.robot_ip,
                )
                return

            # Franka / G1: family-aware CBF bounds from the workspace: block.
            ws_min, ws_max, reach_max, eps_sing, eta_kin = _family_safety_bounds(
                family
            )
            safety_cfg = SafetyConfig(
                vel_lin_max=self.config.velocity + 0.05,
                ws_min=ws_min,
                ws_max=ws_max,
                reach_max=reach_max,
                eps_singularity=eps_sing,
                eta_kinematic=eta_kin,
            )
            logger.info(
                "SafetyConfig for %s: ws=[%.2f..%.2f x %.2f..%.2f x %.2f..%.2f], "
                "reach=%.2f, eps_sing=%.3f, eta_kin=%.2f",
                family,
                ws_min[0],
                ws_max[0],
                ws_min[1],
                ws_max[1],
                ws_min[2],
                ws_max[2],
                reach_max,
                eps_sing,
                eta_kin,
            )
            # SafeRobot disabled for Franka: its CBF velocity
            # filter interferes with direct joint-space moves and
            # clamps targets unpredictably. The 100Nm collision
            # thresholds in FrankaDriver + workspace checks in
            # ScoreExecutor provide sufficient protection.
            self._executor = ScoreExecutor(
                robot=self._robot,
                velocity=self.config.velocity,
                safe_height=self.config.safe_height,
                pipeline=self,
                strict_placement_verify=self.config.strict_placement_verify,
                grasp_calibration=self._grasp_calibration,
            )
            # Give the executor the resolved RobotProfile and reapply the
            # workspace bounds so they read from config, not raw YAML. The
            # values are identical to the historical YAML read.
            self._executor._profile = self.profile
            self._executor._apply_family_workspace_bounds()
            logger.info(
                "%s connected at %s via factory (SafeRobot/CBF "
                "intentionally disabled for this family)",
                family,
                self.config.robot_ip,
            )
        except Exception as e:
            logger.warning(
                "Robot connection failed (%s): %s (running perception-only)",
                family,
                e,
            )
            self._robot = None
            self._executor = None
            # UR10e only: fall back to the deprecated external UR10eRobot if
            # the in-repo factory path could not be constructed (e.g. the
            # in-repo driver is unusable but a teleop deploy tree is present).
            if family == "ur10e":
                self._init_ur10e_legacy()

    def _init_ur10e_legacy(self):
        """
        DEPRECATED external-teleop UR10e bring-up (factory fallback only).

        Builds the external ``UR10eRobot`` from the teleop deploy tree and
        wires the same SafeRobot/ScoreExecutor/control-bounds assembly the
        factory path produces. Reached only when ``_init_robot``'s factory
        construction for the ur10e family raised. New UR10e hosts should use
        the in-repo ``UR10eDriver``; this path exists for continuity with
        rigs that still depend on the external teleop checkout.
        """
        if not HAS_UR10E:
            logger.warning(
                "UR10e factory path failed and no external teleop UR10eRobot "
                "available (running perception-only)"
            )
            return
        logger.warning(
            "DEPRECATED: falling back to external teleop UR10eRobot. "
            "The in-repo UR10eDriver is preferred; this fallback will be "
            "removed once all UR10e rigs migrate."
        )
        robot_deploy_paths = [
            _TELEOP_DEPLOY_BASE,
            Path(__file__).parent.parent.parent / "external_deploy/teleop/ur10e_deploy",
        ]
        for p in robot_deploy_paths:
            if p.exists():
                sys.path.insert(0, str(p))
                break
        try:
            robot_config = UR10eConfig(
                robot_ip=self.config.robot_ip,
                max_velocity=self.config.velocity,
                max_acceleration=0.3,
            )
            self._robot = UR10eRobot(robot_config)
            self._robot.connect()
            _ur_ctrl = (
                self.profile.control() if self.profile is not None else {}
            ) or {}
            _ur_ws_min = _ur_ctrl.get("workspace_min", [-1.1, -0.5, -0.27])
            _ur_ws_max = _ur_ctrl.get("workspace_max", [-0.5, 0.7, 0.50])
            safety_cfg = SafetyConfig(
                vel_lin_max=self.config.velocity + 0.05,
                ws_min=np.array([float(v) for v in _ur_ws_min]),
                ws_max=np.array([float(v) for v in _ur_ws_max]),
            )
            self._robot = SafeRobot(
                self._robot,
                enable_obstacles=False,  # obstacles set from detection map
                enable_force=False,  # force monitoring causes jiggling from noisy F/T
                config=safety_cfg,
            )
            self._executor = ScoreExecutor(
                robot=self._robot,
                velocity=self.config.velocity,
                safe_height=self.config.safe_height,
                pipeline=self,
                strict_placement_verify=self.config.strict_placement_verify,
                grasp_calibration=self._grasp_calibration,
            )
            self._executor._profile = self.profile
            logger.info(
                "UR10e connected at %s (legacy external driver, "
                "SafeRobot enabled, force off)",
                self.config.robot_ip,
            )
        except Exception as e:
            logger.warning(
                "Legacy UR10e connection failed: %s (running perception-only)", e
            )
            self._robot = None
            self._executor = None

    def _init_kinects_from_early(self, devices: dict):
        # Use pre-started Kinect devices from server.py (avoids glibc crash).
        # pyk4a is imported at module top (try/except); this path only runs
        # when early-opened devices exist, i.e. pyk4a is present.
        global HAS_PYK4A
        HAS_PYK4A = True
        master = self.config.kinect_master_serial
        use_hw_sync = bool(getattr(self.config, "kinect_use_hw_sync", False))
        for dev_id, dev in sorted(devices.items()):
            serial = dev.serial
            name = "sideview" if serial == master else "birdview"
            # Mirror the wired-sync role the early open gave the device so
            # the wrapper's warmup semantics match (subordinates defer).
            sync_mode = "STANDALONE"
            if use_hw_sync:
                sync_mode = "MASTER" if serial == master else "SUBORDINATE"
            cam = AzureKinectCamera(
                device_id=dev_id,
                color_resolution=self.config.kinect_resolution,
                depth_mode=self.config.kinect_depth_mode,
                camera_fps=self.config.kinect_fps,
                sync_mode=sync_mode,
                pre_started_device=dev,
            )
            cam.open()
            cal = CameraCalibration(
                name=name,
                width=cam.config.width,
                height=cam.config.height,
                fx=cam.config.fx,
                fy=cam.config.fy,
                cx=cam.config.cx,
                cy=cam.config.cy,
            )
            if name == "sideview":
                self._kinect = cam
                self._kinect_cal = cal
                self._kinect_serial = serial
            else:
                self._kinect2 = cam
                self._kinect2_cal = cal
                self._kinect2_serial = serial
            logger.info(
                "Kinect %d (%s) ready (%dx%d, fx=%.1f)",
                dev_id,
                name,
                cam.config.width,
                cam.config.height,
                cam.config.fx,
            )

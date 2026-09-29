"""
SPARK Real Pipeline - End-to-end orchestrator for physical robot deployment.

Coordinates: Camera capture -> SAM3 perception -> Gemini planning -> robot execution.
"""

import logging
import os
import threading
import time
from contextlib import contextmanager
from pathlib import Path
from typing import Dict, List, Optional

from spark_real.perception.annotation_anchors import AnnotationAnchors
from spark_real.perception.prompt_cache import LearnedPrompts
from spark_real.perception.camera import (
    RealSenseCamera,
)
from spark_real.perception.spark_perception import SPARKPerception
from spark_real.planning.spark_planner import SPARKPlanner
from spark_real.planning.bt_seeds import ensure_seeded
from spark_real.bt_library import BTLibrary
from spark_real.grasp_depth_memory import GraspCalibration
from spark_real.video_recorder import VideoRecorder
from spark_real.control.score_executor import ScoreExecutor
from spark_real.calibration import CameraCalibration
from spark_real.config import SparkConfig, load_profile
from spark_real.pipeline_types import PipelineConfig, TaskResult
from spark_real.pipeline_init import InitMixin
from spark_real.pipeline_init_bimanual import BimanualInitMixin
from spark_real.pipeline_perception import PerceptionMixin
from spark_real.pipeline_execution import ExecutionMixin
from spark_real.pipeline_run import RunMixin
from spark_real.pipeline_io import IOMixin

try:
    import pyrealsense2 as rs
except ImportError:
    rs = None

try:
    from spark_real.perception.equigrasp import _get_generator
except Exception:
    _get_generator = None

try:
    from spark_real.sensors.tactile import TactileManager
except Exception:
    TactileManager = None

logger = logging.getLogger(__name__)

# Anchor for RELATIVE paths coming out of the config (output_dir,
# bt.library_dir, ...). `python -m spark_real.server` is documented as being
# launched from <repo>/src, and that is where the live output/ tree already
# sits, so the directory that CONTAINS the package reproduces the documented
# layout without depending on the process cwd (launching from <repo> instead
# of <repo>/src would otherwise open a different, empty bt_library). An
# absolute output_dir (or bt.library_dir) still wins.
_CONFIG_PATH_ANCHOR = Path(__file__).resolve().parents[1]


def resolve_config_path(path_str) -> Path:
    """
    Resolve a config path to an absolute one, independent of the cwd.

    Absolute (and ``~``) paths are returned as given; relative ones are
    anchored on the directory containing the ``spark_real`` package rather
    than on the process cwd.
    """
    p = Path(str(path_str)).expanduser()
    if not p.is_absolute():
        p = _CONFIG_PATH_ANCHOR / p
    return p.resolve()


# Cached result of the RealSense availability probe. Probed ONCE per
# process: on the RSUSB librealsense backend (mandatory on this host, see
# /etc/modprobe.d/blacklist-uvcvideo.conf) every rs.context() re-inits
# libusb and re-walks the bus, and repeated enumeration is exactly the
# lifecycle churn that corrupts the fragile xHCI controller here.
_REALSENSE_PROBE_RESULT: Optional[bool] = None


def _realsense_device_available() -> bool:
    """
    Return True iff pyrealsense2 enumerates at least one camera.

    Used to default-init the wrist RealSense when one is plugged in,
    without requiring an explicit flag or hardcoded /dev path. The bus is
    probed once per process and the answer cached; a camera plugged in
    later needs a server restart to be picked up (it always did in
    practice: this is only consulted during initialize()).
    """
    global _REALSENSE_PROBE_RESULT
    if _REALSENSE_PROBE_RESULT is not None:
        return _REALSENSE_PROBE_RESULT
    if rs is None:
        return False
    try:
        _REALSENSE_PROBE_RESULT = len(rs.context().query_devices()) > 0
    except Exception:
        _REALSENSE_PROBE_RESULT = False
    return _REALSENSE_PROBE_RESULT


class SPARKRealPipeline(
    InitMixin, BimanualInitMixin, PerceptionMixin, ExecutionMixin, RunMixin, IOMixin
):
    """
    End-to-end pipeline for SPARK physical deployment.
    """

    def __init__(self, config: PipelineConfig = None, profile=None):
        self.config = config or PipelineConfig()
        # Resolved per-system config (gripper, cameras, home, workspace): the
        # single source each subsystem reads. Built from the family when not
        # supplied so non-server callers (routes, tests) still get one.
        if profile is None:
            profile = load_profile(SparkConfig(family=self.config.robot_family))
        self.profile = profile
        # SPARK_VELOCITY_OVERRIDE env var lets a single run drop the
        # Cartesian motion speed (m/s) without editing the default. Lower
        # nominal speed gives Ruckig more headroom to smooth the path near
        # a singularity and avoid joint-velocity reflexes during transit.
        try:
            _vel_env = os.environ.get("SPARK_VELOCITY_OVERRIDE")
            if _vel_env is not None and float(_vel_env) > 0.0:
                self.config.velocity = float(_vel_env)
                logger.info(
                    "SPARK_VELOCITY_OVERRIDE: capping Cartesian "
                    "velocity at %.3f m/s (default 0.25)",
                    self.config.velocity,
                )
        except (ValueError, TypeError):
            pass
        self._kinect = None  # Side view (master Kinect)
        self._kinect2 = None  # Bird view (subordinate Kinect)
        self._multi_k4a = None  # MultiAzureKinectManager for synced capture
        self._kinect_serial = None
        self._kinect2_serial = None
        self._realsense: Optional[RealSenseCamera] = None
        self._perception: Optional[SPARKPerception] = None
        self._planner: Optional[SPARKPlanner] = None
        self._robot = None
        self._executor: Optional[ScoreExecutor] = None
        self._kinect_cal: Optional[CameraCalibration] = None
        self._kinect2_cal: Optional[CameraCalibration] = None
        self._realsense_cal: Optional[CameraCalibration] = None
        # Unified role-keyed view over the single-arm cameras, built in
        # initialize() from the already-open slots. Private on purpose: see
        # _build_unified_camera_registry for why single-arm must not expose a
        # public `camera_registry` (routes/streaming.py keys off that name).
        self._camera_registry = None
        self._initialized = False
        self._task_history: List[TaskResult] = []
        self._video_recorder: Optional[VideoRecorder] = None
        # Multi-camera recording: one VideoRecorder per live camera.
        self._video_recorders: dict = {}
        # Voyager-style memory: append every successful BT score so future
        # plans can few-shot from prior real-robot wins -- and, once a task
        # has one verified run, serve that exact tree back with no LLM call.
        #
        # The root is derived from output_dir, never from the cwd: a relative
        # output_dir is anchored by resolve_config_path. bt.library_dir
        # overrides, and an absolute one is honoured verbatim.
        self._output_root = resolve_config_path(self.config.output_dir)
        self._bt_library_root = (
            resolve_config_path(self.config.bt_library_dir)
            if self.config.bt_library_dir
            else self._output_root.parent / "bt_library"
        )
        # Logged at INFO on construction so a mid-collection cwd change shows
        # up in the log as a changed root, instead of looking like data loss.
        logger.info(
            "BT library root: %s (output_dir=%r, cwd=%s)",
            self._bt_library_root,
            self.config.output_dir,
            os.getcwd(),
        )
        self._bt_library = BTLibrary(
            self._bt_library_root,
            min_similarity=self.config.bt_min_similarity,
            auto_promote_after=self.config.bt_auto_promote_after,
        )
        # Prime it from the packaged seed YAMLs (configs/bt_seeds/) so a fresh
        # checkout serves the demo tasks from cache on its very first request.
        # Idempotent; never overwrites a live entry with a higher success count.
        # Skipped while collecting a fresh corpus -- see bt_seed_on_start.
        if self.config.bt_seed_on_start:
            ensure_seeded(self._bt_library, seed_dir=self.config.bt_seed_dir)
        else:
            logger.info(
                "BT seeds: not installed (bt.seed_on_start=false); tasks without a "
                "live entry will be planned, not served from cache"
            )
        # Per-object grasp depth-correction memory. Same anchoring as the BT
        # library: this is persistent learned state, so it must not follow the
        # cwd either.
        # Learned SAM3 prompts (spark_bench's tuned-prompts tier): phrases the
        # LLM rung proposed and the count gate accepted, persisted per
        # task+group so the next run tries them before paying another API
        # call. Same anchoring rule as the BT library and grasp calibration:
        # learned state must not follow the cwd.
        self._learned_prompts = LearnedPrompts(
            self._output_root.parent / "learned_prompts.json"
        )
        # Operator click/box annotations, saved per task. Same anchoring rule.
        self._annotation_anchors = AnnotationAnchors(
            self._output_root.parent / "annotation_anchors.json"
        )
        self._grasp_calibration = (
            GraspCalibration(self._output_root.parent / "grasp_calibration.json")
            if self.config.use_grasp_calibration
            else None
        )
        # Activity tracking surfaced via /api/status for the UI status pill.
        # Stack so nested/overlapping activities don't lose state.
        self._activity_stack: List[str] = []
        self._activity_lock = threading.Lock()
        # Per-device read locks. AzureKinectCamera.read() only reads cached
        # numpy arrays under the camera's own lock; the ONLY thread that
        # touches the k4a handle is the capture thread, which these locks do
        # not gate at all. What they buy: one consumer at a time sees a
        # consistent rgb/depth/ir triple, and the deferred rectification in
        # read() is not run twice on the same frame.
        # Anything that needs real exclusion from the device must gate on
        # the capture thread (close() joins it), not on these.
        self._kinect_read_lock = threading.Lock()
        self._kinect2_read_lock = threading.Lock()
        self._realsense_read_lock = threading.Lock()
        # Wrist-camera offset from TCP, family-aware.  Use the per-family
        # default unless the config supplies an explicit override.
        self._wrist_tool_offset = self._resolve_wrist_tool_offset()

    def initialize(self):
        """
        Initialize all pipeline components.
        """
        logger.info("Initializing SPARK Real Pipeline...")

        if self.config.use_kinect:
            pre = getattr(self, "_pre_opened_kinects", {})
            if pre:
                self._init_kinects_from_early(pre)
            else:
                self._init_kinects()

        if self.config.use_realsense is None:
            self.config.use_realsense = _realsense_device_available()
            logger.info(
                "RealSense auto-detect: %s",
                "device found" if self.config.use_realsense else "no device, skipping",
            )

        if self.config.use_realsense:
            logger.info("Opening RealSense...")
            try:
                self._realsense = RealSenseCamera(
                    serial=self.config.realsense_serial or None,
                    color_only=getattr(self.config, "realsense_color_only", True),
                )
                self._realsense.open()
                self._realsense_cal = CameraCalibration(
                    name="wrist",
                    width=self._realsense.config.width,
                    height=self._realsense.config.height,
                    fx=self._realsense.config.fx,
                    fy=self._realsense.config.fy,
                    cx=self._realsense.config.cx,
                    cy=self._realsense.config.cy,
                    depth_scale=float(getattr(self, "_wrist_depth_scale", 1.0)),
                    depth_offset=float(getattr(self, "_wrist_depth_offset", 0.0)),
                )
                logger.info(
                    "RealSense ready (%dx%d, depth_scale=%.4f, " "depth_offset=%.1fmm)",
                    self._realsense_cal.width,
                    self._realsense_cal.height,
                    self._realsense_cal.depth_scale,
                    self._realsense_cal.depth_offset * 1000.0,
                )
            except Exception as e:
                # Wrist camera is optional, used only for grasp refinement.
                # Bird/sideview Kinects can drive the pipeline on their own.
                logger.warning(
                    "RealSense unavailable (%s), continuing without "
                    "wrist camera. Grasp refinement will be skipped.",
                    e,
                )
                self._realsense = None
                self._realsense_cal = None

        self._load_handeye_calibrations()

        # Unified, role-keyed view over the cameras the existing open paths
        # above just brought up. Built for every family so downstream code
        # can iterate cameras by role; the bimanual family attaches its own
        # `camera_registry` later in _init_bimanual_franka. See
        # _build_unified_camera_registry for why single-arm uses a private
        # attribute rather than `camera_registry`.
        self._build_unified_camera_registry()

        logger.info("Loading SAM3 perception...")
        self._perception = SPARKPerception(sam3_threshold=self.config.sam3_threshold)
        self._perception.load_models(load_da3=not self.config.use_hardware_depth)
        logger.info("SAM3 perception ready")

        logger.info("Initializing Gemini planner...")
        self._planner = SPARKPlanner(
            llm_backend="gemini",
            model=self.config.gemini_model,
            robot_family=self.config.robot_family,
            temperature=self.config.planner_temperature,
        )
        logger.info(
            "Planner ready (model=%s, family=%s)",
            self.config.gemini_model,
            self.config.robot_family,
        )

        # Eager-load EquiGraspFlow so the first grasp_se3 call doesn't pay
        # the cold-start (PyTorch import + checkpoint load onto GPU); that
        # latency is absorbed into server startup. Safe to fail: if the
        # model can't load (missing weights, no CUDA, env gap), log and
        # continue and grasp_se3 hits the same error on first use.
        try:
            if _get_generator is None:
                raise ImportError("EquiGraspFlow unavailable")
            _t0 = time.time()
            gen = _get_generator()
            if gen._mode == "inprocess":
                logger.info("Eager-loading EquiGraspFlow checkpoint...")
                gen._load_model()
                logger.info(
                    "EquiGraspFlow ready in %.1fs (mode=inprocess)", time.time() - _t0
                )
            else:
                logger.info(
                    "EquiGraspFlow mode=%s; skipping eager load "
                    "(subprocess starts fresh per call)",
                    gen._mode,
                )
        except Exception as _eg_exc:
            logger.warning(
                "EquiGraspFlow eager-load failed: %s (first grasp_se3 "
                "call will pay the load cost)",
                _eg_exc,
            )

        # Optional FlexiTac tactile sensing. Auto-detects on
        # /dev/ttyUSB* + /dev/ttyACM*; silently no-ops when no
        # sensor is plugged in or the `flexitac` package is missing.
        self._tactile = None
        try:
            if TactileManager is None:
                raise ImportError("TactileManager unavailable")
            self._tactile = TactileManager()
            self._tactile.start()
            if self._tactile.available():
                logger.info("FlexiTac tactile online: sides=%s", self._tactile.sides())
        except Exception as _tac_exc:
            logger.warning("Tactile init failed (continuing without): %s", _tac_exc)
            self._tactile = None

        # Robot is optional; can run perception-only
        if self.config.robot_ip:
            self._init_robot()

        Path(self.config.output_dir).mkdir(parents=True, exist_ok=True)

        self._initialized = True
        logger.info("Pipeline initialization complete")

    @property
    def task_history(self) -> List[TaskResult]:
        """
        Get history of all tasks run in this session.
        """
        return self._task_history

    @contextmanager
    def activity(self, name: str):
        """
        Context manager: mark the pipeline as busy with ``name``.

        Stacks so that nested or overlapping activities still report
        correctly. The latest pushed name wins for the status field.
        Robot execution (``executor._running``) is reported separately
        and overrides any pushed name.
        """
        with self._activity_lock:
            self._activity_stack.append(name)
        try:
            yield
        finally:
            with self._activity_lock:
                # Pop the matching entry, usually the last one, but be
                # defensive in case something else mutated the stack.
                try:
                    idx = (
                        len(self._activity_stack)
                        - 1
                        - self._activity_stack[::-1].index(name)
                    )
                    self._activity_stack.pop(idx)
                except ValueError:
                    pass

    @property
    def activity_state(self) -> str:
        """
        One of: idle | capturing | detecting | planning | executing.

        Robot execution beats any pushed activity (because it's the
        operator-visible motion). Otherwise the most recently pushed
        activity wins, falling back to ``idle``.
        """
        if self._executor is not None and self._executor._running:
            return "executing"
        with self._activity_lock:
            if self._activity_stack:
                return self._activity_stack[-1]
        return "idle"

    def get_status(self) -> Dict:
        """
        Get current pipeline status for the frontend.
        """
        # Per-camera frame age: 'connected' only says the handle exists; a
        # wedged Kinect is connected AND frozen. The UI shows a dead camera
        # instead of a frozen picture using this.
        frame_age = {}
        for name, dev in (
            ("sideview", self._kinect),
            ("birdview", self._kinect2),
            ("wrist", self._realsense),
        ):
            if dev is not None and hasattr(dev, "last_frame_age_s"):
                try:
                    frame_age[name] = dev.last_frame_age_s()
                except Exception:  # noqa: BLE001 - status must never raise
                    frame_age[name] = None
        return {
            "initialized": self._initialized,
            "kinect_connected": self._kinect is not None,
            "kinect2_connected": self._kinect2 is not None,
            "realsense_connected": self._realsense is not None,
            "camera_frame_age_s": frame_age,
            "robot_connected": self._robot is not None,
            "perception_ready": self._perception is not None
            and self._perception._sam3 is not None,
            "planner_ready": self._planner is not None,
            "tasks_completed": len(self._task_history),
            "robot_ip": self.config.robot_ip,
            "robot_family": self.config.robot_family,
            "robot_model": self.config.robot_model,
            "activity": self.activity_state,
        }

    def _stop_reader_threads(self):
        """
        Quiesce every background thread that reads a device or the robot.

        Called first from shutdown(), so no close() or disconnect() below can
        land while one of these threads still holds a handle. Each is isolated:
        a recorder that will not stop must not prevent the devices being
        released.
        """
        if getattr(self, "_video_recorders", None):
            try:
                self.stop_video_recording()
            except Exception as exc:  # noqa: BLE001
                logger.warning("stop_video_recording raised during shutdown: %s", exc)

    def shutdown(self):
        """
        Clean shutdown of all components.

        Each device close is isolated so a hang or exception in one
        (libfranka disconnect can hang) does not prevent the others
        from being released. Cameras go first because their depth MCUs
        brick if left mid-stream; the robot just sits idle if disconnect
        is skipped.
        """
        logger.info("Shutting down pipeline...")
        # STOP AND JOIN THE READERS BEFORE CLOSING WHAT THEY READ.
        # The per-camera VideoRecorder threads poll
        # streaming.capture_single_camera at 10 Hz for the whole task;
        # unjoined they would read cameras that shutdown is concurrently
        # releasing, and their mp4s would never be encoded (stop() is the only
        # thing that writes the file). Joining here is also what makes the
        # ordering assertable.
        self._stop_reader_threads()
        # Quiesce the wrist camera's recovery machinery FIRST (non-blocking):
        # closing a Kinect can take seconds, and a RealSense hardware_reset
        # firing mid-Kinect-close is exactly the overlapping-lifecycle
        # pattern that corrupts the shared xHCI controller. request_stop()
        # only sets flags; the actual close happens in the loop below, after
        # the Kinects.
        # getattr throughout: /api/initialize calls shutdown() on a pipeline
        # whose __init__ or initialize() did NOT finish, to release the
        # devices it did manage to open. A bare self._realsense on such an
        # object would raise AttributeError and abort the teardown before the
        # Kinects are closed.
        realsense = getattr(self, "_realsense", None)
        if realsense is not None and hasattr(realsense, "request_stop"):
            try:
                realsense.request_stop()
            except Exception as exc:
                logger.warning("realsense request_stop raised: %s", exc)
        for label, dev in (
            ("kinect (sideview)", getattr(self, "_kinect", None)),
            ("kinect2 (birdview)", getattr(self, "_kinect2", None)),
            ("realsense", realsense),
        ):
            if dev is None:
                continue
            try:
                dev.close()
            except Exception as exc:
                logger.warning("%s close raised: %s", label, exc)
        robot = getattr(self, "_robot", None)
        if robot is not None:
            try:
                robot.disconnect()
            except Exception as exc:
                logger.warning("robot disconnect raised: %s", exc)
        self._initialized = False
        logger.info("Pipeline shutdown complete")

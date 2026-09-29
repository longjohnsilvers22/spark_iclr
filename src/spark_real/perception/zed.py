"""
ZED Mini stereo camera wrapper for SPARK bimanual perception.

Matches the surface of :class:`RealSenseCamera` and
:class:`AzureKinectCamera` so the higher-level pipeline can swap one
device for another without branching on driver type.

Falls back to a synthetic-frame simulator when ``pyzed.sl`` is not
importable so unit tests and the frontend can be exercised without the
ZED SDK installed. The simulator returns a checker-board RGB frame and
a flat-depth frame at the configured resolution.

Public API mirrors what :mod:`spark_real.pipeline` expects:

* ``open(serial=None)``: boot the device (or sim).
* ``close()``: shut down.
* ``read(depth=True)`` -> ``(rgb_uint8, depth_meters)``: single
  synchronized capture.
* ``config: CameraConfig``: fx/fy/cx/cy/width/height populated from
  the ZED SDK calibration (or filled with synthetic values in sim).
"""

from __future__ import annotations

import configparser
import glob
import logging
import threading
import time
from dataclasses import dataclass
from typing import Optional, Tuple

import numpy as np

from spark_real.perception.camera import CameraConfig

try:
    import pyzed.sl as sl  # type: ignore  # ZED SDK optional
except ImportError:
    sl = None

logger = logging.getLogger(__name__)


_DEPTH_MODE_MAP = {
    # Strings come from the YAML config; map to pyzed.sl.DEPTH_MODE enum
    # at runtime (only if the SDK is importable).
    "NEURAL": "NEURAL",
    "ULTRA": "ULTRA",
    "QUALITY": "QUALITY",
    "PERFORMANCE": "PERFORMANCE",
    "STANDARD": "STANDARD",
}


class ZEDMiniCamera:
    """
    Stereo + depth wrapper for the ZED Mini.

    Serves as the primary localization camera for the bimanual rig,
    complementary to the two wrist RealSenses (which are occluded during
    handoff).
    """

    def __init__(
        self,
        serial: Optional[str] = None,
        width: int = 1280,
        height: int = 720,
        fps: int = 30,
        depth_mode: str = "NEURAL",
        flip: str = "auto",
    ):
        self.serial = serial or ""
        self.width = int(width)
        self.height = int(height)
        self.fps = int(fps)
        self.depth_mode = depth_mode.upper()
        # flip: "auto" (SDK picks via IMU), "on" (force 180-degree image flip),
        # "off" (no flip). Use "on" if the ZED is physically mounted inverted
        # AND the SDK's auto detect is not kicking in.
        self.flip = str(flip or "auto").lower()
        self._zed = None
        self._sim_mode = False
        self._opened = False
        self._lock = threading.Lock()
        # CameraConfig doesn't carry a name field; the camera registry
        # holds the role string. Use width*0.7 as a placeholder focal
        # length so backproject is at least well-conditioned in sim mode.
        self.config = CameraConfig(
            width=self.width,
            height=self.height,
            fx=self.width * 0.7,
            fy=self.width * 0.7,
            cx=self.width / 2,
            cy=self.height / 2,
        )

    # lifecycle
    def open(self, serial: Optional[str] = None) -> bool:
        if serial:
            self.serial = serial
        if sl is None:
            logger.warning(
                "pyzed.sl not installed; ZEDMiniCamera running in simulator "
                "mode. Install ZED SDK + Python wrapper for real hardware."
            )
            self._sim_mode = True
            self._opened = True
            return True

        zed = sl.Camera()
        init = sl.InitParameters()
        init.camera_resolution = self._resolution_for(sl)
        init.camera_fps = self.fps
        # SDK 5.x dropped DEPTH_MODE.STANDARD; resolve the default via
        # getattr too because Python evaluates the default arg eagerly and
        # referencing a missing enum member there crashes even when the
        # primary lookup would succeed.
        _default_depth_mode = getattr(
            sl.DEPTH_MODE, "PERFORMANCE", getattr(sl.DEPTH_MODE, "NEURAL", None)
        )
        init.depth_mode = getattr(
            sl.DEPTH_MODE,
            _DEPTH_MODE_MAP.get(self.depth_mode, "NEURAL"),
            _default_depth_mode,
        )
        init.coordinate_units = sl.UNIT.METER
        init.coordinate_system = sl.COORDINATE_SYSTEM.RIGHT_HANDED_Z_UP_X_FWD
        # Image flip control. SDK 5.x exposes sl.FLIP_MODE.{AUTO,ON,OFF}
        # on InitParameters.camera_image_flip. Wrap in getattr so older SDK
        # builds without the enum still work (fall back to no flip).
        _flip_mode_enum = getattr(sl, "FLIP_MODE", None)
        if _flip_mode_enum is not None:
            _flip_choice = {
                "auto": getattr(_flip_mode_enum, "AUTO", None),
                "on": getattr(_flip_mode_enum, "ON", None),
                "off": getattr(_flip_mode_enum, "OFF", None),
            }.get(self.flip, getattr(_flip_mode_enum, "AUTO", None))
            if _flip_choice is not None:
                try:
                    init.camera_image_flip = _flip_choice
                except Exception:
                    logger.warning(
                        "ZED: SDK did not accept camera_image_flip=%s", self.flip
                    )
        if self.serial:
            try:
                init.set_from_serial_number(int(self.serial))
            except Exception:
                logger.warning(
                    "invalid ZED serial %r; opening any available unit", self.serial
                )

        status = zed.open(init)
        if status != sl.ERROR_CODE.SUCCESS:
            logger.error("ZED open failed: %s, falling back to sim mode", status)
            self._sim_mode = True
            self._opened = True
            return False

        # Pull intrinsics from the SDK.
        ci = zed.get_camera_information().camera_configuration
        cp_left = ci.calibration_parameters.left_cam
        self.config = CameraConfig(
            width=self.width,
            height=self.height,
            fx=float(cp_left.fx),
            fy=float(cp_left.fy),
            cx=float(cp_left.cx),
            cy=float(cp_left.cy),
        )
        self._zed = zed
        self._opened = True
        return True

    def close(self) -> None:
        if self._zed is not None and not self._sim_mode:
            try:
                self._zed.close()
            except Exception:
                logger.exception("ZED close failed")
        self._opened = False

    # capture
    def read(self, depth: bool = True) -> Tuple[np.ndarray, Optional[np.ndarray]]:
        with self._lock:
            if self._sim_mode or self._zed is None:
                return self._sim_frame(depth=depth)

            rt = sl.RuntimeParameters()
            err = self._zed.grab(rt)
            if err != sl.ERROR_CODE.SUCCESS:
                logger.warning("ZED grab error %s; returning sim frame", err)
                return self._sim_frame(depth=depth)
            mat_rgb = sl.Mat()
            self._zed.retrieve_image(mat_rgb, sl.VIEW.LEFT)
            rgb = mat_rgb.get_data()[:, :, :3].copy()
            # ZED returns BGR; convert to RGB for SAM3.
            rgb = rgb[..., ::-1].copy()
            depth_np: Optional[np.ndarray] = None
            if depth:
                mat_d = sl.Mat()
                self._zed.retrieve_measure(mat_d, sl.MEASURE.DEPTH)
                depth_np = mat_d.get_data().astype(np.float32)
                # ZED uses NaN for invalid depth; promote to 0 so consumers
                # that expect numeric arrays don't blow up. SAM3 / DA3 paths
                # treat 0 as "no depth".
                depth_np = np.nan_to_num(depth_np, nan=0.0, posinf=0.0, neginf=0.0)
            return rgb, depth_np

    def read_stereo(
        self, hires: bool = True
    ) -> Tuple[np.ndarray, np.ndarray, Optional[np.ndarray]]:
        """
        Return (left_bgr, right_bgr, depth_f32) from the ZED stereo pair.

        When hires=True, retrieves at the camera's native resolution
        regardless of the configured width/height (for calibration).
        """
        with self._lock:
            if self._sim_mode or self._zed is None:
                f, d = self._sim_frame(depth=True)
                return f, f.copy(), d

            rt = sl.RuntimeParameters()
            err = self._zed.grab(rt)
            if err != sl.ERROR_CODE.SUCCESS:
                f, d = self._sim_frame(depth=True)
                return f, f.copy(), d
            mat_l = sl.Mat()
            mat_r = sl.Mat()
            mat_d = sl.Mat()
            self._zed.retrieve_image(mat_l, sl.VIEW.LEFT)
            self._zed.retrieve_image(mat_r, sl.VIEW.RIGHT)
            self._zed.retrieve_measure(mat_d, sl.MEASURE.DEPTH)
            left = mat_l.get_data()[:, :, :3].copy()
            right = mat_r.get_data()[:, :, :3].copy()
            depth = np.nan_to_num(
                mat_d.get_data().astype(np.float32), nan=0.0, posinf=0.0, neginf=0.0
            )
            return left, right, depth

    def get_camera_info(self) -> dict:
        """
        Return SDK intrinsics + baseline for stereo triangulation.
        """
        with self._lock:
            if self._sim_mode or self._zed is None:
                return {"sim": True}
            info = self._zed.get_camera_information()
            cp = info.camera_configuration.calibration_parameters
            return {
                "fx_l": cp.left_cam.fx,
                "fy_l": cp.left_cam.fy,
                "cx_l": cp.left_cam.cx,
                "cy_l": cp.left_cam.cy,
                "fx_r": cp.right_cam.fx,
                "fy_r": cp.right_cam.fy,
                "cx_r": cp.right_cam.cx,
                "cy_r": cp.right_cam.cy,
                "baseline_m": abs(cp.stereo_transform.get_translation().get()[0])
                or self._read_baseline_from_conf(),
            }

    def _read_baseline_from_conf(self) -> float:
        """
        Read baseline from /usr/local/zed/settings/SN*.conf.
        """
        for p in glob.glob("/usr/local/zed/settings/SN*.conf"):
            cfg = configparser.ConfigParser()
            cfg.read(p)
            for sec in cfg.sections():
                if cfg.has_option(sec, "Baseline"):
                    return float(cfg.get(sec, "Baseline")) / 1000.0
        return 0.0

    # simulator
    def _sim_frame(self, depth: bool) -> Tuple[np.ndarray, Optional[np.ndarray]]:
        h, w = self.height, self.width
        # Checker pattern for sanity-checking the frontend tile.
        rgb = np.zeros((h, w, 3), dtype=np.uint8)
        sq = 32
        for y in range(0, h, sq):
            for x in range(0, w, sq):
                v = 220 if ((x // sq) + (y // sq)) % 2 else 40
                rgb[y : y + sq, x : x + sq] = (v, v // 2, 255 - v)
        depth_np = None
        if depth:
            depth_np = np.full((h, w), 0.80, dtype=np.float32)
        return rgb, depth_np

    # helpers
    def _resolution_for(self, sl):
        """
        Pick the lowest sl.RESOLUTION >= the requested width.

        ZED Mini supports HD2K (2208x1242), HD1080, HD720, VGA;
        defaults to HD720 if nothing matches.
        """
        if self.width >= 2208:
            return sl.RESOLUTION.HD2K
        if self.width >= 1920:
            return sl.RESOLUTION.HD1080
        if self.width >= 1280:
            return sl.RESOLUTION.HD720
        return sl.RESOLUTION.VGA

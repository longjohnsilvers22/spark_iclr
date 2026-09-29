"""
Camera registry for the bimanual rig.

A role-keyed dictionary so streaming, calibration, and grasp-fusion code
can iterate cameras without embedding their names.

Roles for the bimanual rig:

* ``external``:    ZED Mini on the center tripod (primary localization).
* ``wrist_left``:  Intel RealSense D435i on the left arm's wrist mount.
* ``wrist_right``: Intel RealSense D435i on the right arm's wrist mount.

The registry is constructed from ``configs/bimanual_franka_default.yaml``
by :func:`build_bimanual_camera_registry`. Each entry stores the device
handle, the per-camera :class:`CameraCalibration`, the optional
``arm`` association (for wrist cameras), and the ``tool_offset_xyz``
used to recompute the wrist-camera extrinsic from the per-arm TCP on
every capture.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

import numpy as np

from spark_real.calibration import CameraCalibration
from spark_real.perception.camera import (
    AzureKinectCamera,
    CameraConfig,
    RealSenseCamera,
    USBCamera,
)
from spark_real.perception.zed import ZEDMiniCamera

logger = logging.getLogger(__name__)


@dataclass
class CameraEntry:
    """
    Single camera + its calibration and arm association.
    """

    role: str  # 'external' | 'wrist_left' | 'wrist_right'
    kind: str  # 'zed_mini' | 'realsense' | 'azure_kinect'
    device: Any  # the opened device handle
    calibration: CameraCalibration
    arm: Optional[str] = None  # 'left'|'right' for wrist cams; None otherwise
    tool_offset_xyz: Optional[List[float]] = None  # TCP->camera (for wrist cams)
    serial: str = ""


@dataclass
class CameraRegistry:
    """
    Role-keyed camera registry.

    Iteration order (``roles``) is fixed at construction so the frontend
    tile layout is stable across reloads.
    """

    cameras: Dict[str, CameraEntry] = field(default_factory=dict)
    roles: List[str] = field(default_factory=list)
    primary_role: str = ""  # used for global localization (e.g. external)
    wrist_for_arm: Dict[str, str] = field(default_factory=dict)

    def __getitem__(self, role: str) -> CameraEntry:
        return self.cameras[role]

    def get(self, role: str) -> Optional[CameraEntry]:
        return self.cameras.get(role)

    def __contains__(self, role: str) -> bool:
        return role in self.cameras

    def __iter__(self):
        for r in self.roles:
            yield r, self.cameras[r]


def build_camera_registry(
    cfg: Dict[str, Any],
    auto_open: bool = True,
    primary_role: Optional[str] = None,
) -> CameraRegistry:
    """
    Generic camera registry loader for ANY family's config mapping.

    Reads the ``cameras:`` list of ``cfg`` (the resolved family mapping,
    see config.family_block) and constructs one :class:`CameraEntry` per
    role. Works for single-arm (typically: ``sideview``, ``birdview``,
    ``wrist``) and bimanual (typically: ``external``, ``wrist_left``,
    ``wrist_right``) alike, plus any custom role names a per-rig config
    wants to use.

    Args:
        cfg: the family config mapping.
        auto_open: open each camera immediately. Set False in tests.
        primary_role: override the auto-detected primary role. If not
            given, the loader picks (in order) ``external``, ``sideview``,
            ``birdview``, then the first declared role.
    """
    cam_specs = cfg.get("cameras") or []
    registry = CameraRegistry()
    # Auto-pick primary role unless the caller specified one. Bimanual
    # rigs ship ``external``; single-arm UR10e rigs ship ``sideview``.
    if primary_role is None:
        for candidate in ("external", "sideview", "birdview"):
            if any((spec.get("role") == candidate) for spec in cam_specs):
                primary_role = candidate
                break
        if primary_role is None and cam_specs:
            primary_role = cam_specs[0].get("role", "")
    registry.primary_role = primary_role or ""

    return _populate_registry(registry, cam_specs, auto_open)


def build_bimanual_camera_registry(
    cfg: Dict[str, Any],
    auto_open: bool = True,
) -> CameraRegistry:
    """
    Bimanual-specific alias of :func:`build_camera_registry`.

    Kept for backward compatibility with code that imported the
    bimanual-specific name; the primary role is always ``external``.
    """
    return build_camera_registry(cfg, auto_open=auto_open, primary_role="external")


def _populate_registry(
    registry: CameraRegistry, cam_specs: list, auto_open: bool
) -> CameraRegistry:
    """
    Walk a list of camera specs and populate the registry in-place.
    """
    for spec in cam_specs:
        role = spec.get("role")
        if not role:
            logger.warning("camera spec missing 'role'; skipping: %s", spec)
            continue
        kind = (spec.get("type") or "").lower()
        device: Any = None

        try:
            if kind == "zed_mini":
                device = ZEDMiniCamera(
                    serial=spec.get("serial") or None,
                    width=int(spec.get("width", 1280)),
                    height=int(spec.get("height", 720)),
                    fps=int(spec.get("fps", 30)),
                    depth_mode=spec.get("depth_mode", "NEURAL"),
                )
            elif kind == "realsense":
                # RealSenseCamera takes a CameraConfig (not width/height kwargs);
                # build the config from the YAML spec.
                rs_config = CameraConfig(
                    width=int(spec.get("width", 640)),
                    height=int(spec.get("height", 480)),
                )
                device = RealSenseCamera(
                    serial=spec.get("serial") or None,
                    config=rs_config,
                )
            elif kind in ("kinect", "azure_kinect"):
                # `kinect` is the single-arm roster's spelling; `azure_kinect`
                # is the bimanual spelling. Same device class. Pass through
                # the optional resolution/depth/fps the roster may carry so a
                # config-driven open matches the SDK-level defaults.
                device = AzureKinectCamera(
                    color_resolution=spec.get("resolution", "1080P"),
                    depth_mode=spec.get("depth_mode", "NFOV_UNBINNED"),
                    camera_fps=int(spec.get("fps", 30)),
                )
            elif kind in ("webcam", "usb_camera", "uvc"):
                # Plain UVC USB camera, used on the ANON-LAB rig for the two
                # gripper-mounted Realtek cameras. ``device`` accepts either
                # an integer index or a /dev/v4l/by-path/... string; by-path
                # is preferred because both Realtek cams report the same USB
                # serial and by-id collides.
                cam_cfg = CameraConfig(
                    width=int(spec.get("width", 640)),
                    height=int(spec.get("height", 480)),
                )
                device = USBCamera(
                    device_id=spec.get("device", spec.get("device_id", 0)),
                    config=cam_cfg,
                )
            else:
                logger.warning(
                    "unknown camera type %r for role %r; skipping", kind, role
                )
                continue
        except ImportError as exc:
            # Hardware SDK missing: fall back to the ZED sim camera so
            # downstream code (streaming, frontend) still has something
            # to query. This is the same pattern the ZEDMiniCamera uses
            # internally when pyzed.sl is absent.
            logger.warning(
                "Camera SDK missing for %r (%s); substituting simulator. "
                "Install the matching SDK for live capture.",
                role,
                exc,
            )
            device = ZEDMiniCamera(
                width=int(spec.get("width", 640)),
                height=int(spec.get("height", 480)),
                fps=int(spec.get("fps", 30)),
            )

        if auto_open:
            try:
                device.open()
            except Exception:
                logger.exception(
                    "camera %r failed to open; keeping handle anyway", role
                )

        # Build a CameraCalibration scaffold. Extrinsic stays as identity
        # until the bimanual calibrator writes it. Intrinsics come from
        # the device after open().
        ccfg = getattr(device, "config", None)
        if ccfg is None:
            ccfg = CameraConfig(
                width=int(spec.get("width", 640)),
                height=int(spec.get("height", 480)),
                fx=600.0,
                fy=600.0,
                cx=320.0,
                cy=240.0,
            )
        # CameraCalibration is a dataclass with all-required fields, so
        # build the whole object up front rather than mutating it later.
        cal = CameraCalibration(
            name=role,
            width=int(ccfg.width),
            height=int(ccfg.height),
            fx=float(ccfg.fx),
            fy=float(ccfg.fy),
            cx=float(ccfg.cx),
            cy=float(ccfg.cy),
            extrinsic=np.eye(4),
        )

        entry = CameraEntry(
            role=role,
            kind=kind,
            device=device,
            calibration=cal,
            arm=spec.get("arm"),
            tool_offset_xyz=spec.get("tool_offset_xyz"),
            serial=spec.get("serial") or "",
        )
        registry.cameras[role] = entry
        registry.roles.append(role)
        if entry.arm in ("left", "right"):
            registry.wrist_for_arm[entry.arm] = role

    # Stable order: primary first, then wrist_left, then wrist_right,
    # then anything else. Keeps the frontend tile order predictable.
    preferred = [
        registry.primary_role,
        registry.wrist_for_arm.get("left", ""),
        registry.wrist_for_arm.get("right", ""),
    ]
    ordered = [r for r in preferred if r and r in registry.cameras]
    extras = [r for r in registry.roles if r not in ordered]
    registry.roles = ordered + extras
    logger.info(
        "bimanual camera registry built: roles=%s primary=%s wrist_map=%s",
        registry.roles,
        registry.primary_role,
        registry.wrist_for_arm,
    )
    return registry


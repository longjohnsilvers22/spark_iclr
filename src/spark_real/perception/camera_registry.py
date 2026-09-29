"""
Unified, config-driven, role-keyed camera registry for every SPARK family.

One source of truth for "which cameras does this deployment have, and how
do I reach each by role". Driven entirely by RobotProfile.cameras(), which
returns the `cameras:` list from configs/<family>_default.yaml.

Two entry points:
  spec(profile)                 -> pure dict role -> {type, serial/device,
                                    ...}. Opens no hardware, so it is safe to
                                    call in tests and at config-validation
                                    time.
  build_camera_registry(profile)-> instantiates each entry by its `type`
                                    (kinect, realsense, zed_mini) keyed by
                                    role and returns a CameraRegistry.

The heavy lifting (instantiate-by-type, open, calibration scaffold, stable
role ordering) is shared with the bimanual registry: this module reuses
CameraEntry / CameraRegistry / _populate_registry from
perception.bimanual_camera_registry rather than re-implementing the
camera-open code. The bimanual rig (external zed_mini + two wrist
realsenses) and the single-arm rigs (birdview/sideview kinects + wrist
realsense) therefore go through the exact same builder.
"""

from __future__ import annotations

import logging
from typing import Any, Dict, List, Optional

from spark_real.perception.bimanual_camera_registry import (
    CameraEntry,
    CameraRegistry,
    _populate_registry,
)

logger = logging.getLogger(__name__)

# Roles preferred as the primary localization camera, in priority order.
# The bimanual rig ships `external`; single-arm rigs ship `sideview` (the
# master Kinect) as the workspace anchor. Falls through to `birdview`, then
# the first declared role.
_PRIMARY_ROLE_PREFERENCE = ("external", "sideview", "birdview")


def spec(profile) -> Dict[str, Dict[str, Any]]:
    """
    Return a pure role -> spec mapping from profile.cameras(). No hardware
    is touched, so this is the testable contract: each value is a copy of
    the YAML entry for that role (type, serial or device, width, height,
    fps, arm, tool_offset_xyz, ...). Entries without a `role` are skipped
    with a warning, mirroring _populate_registry's behavior.
    """
    out: Dict[str, Dict[str, Any]] = {}
    for entry in profile.cameras():
        role = entry.get("role")
        if not role:
            logger.warning("camera spec missing 'role'; skipping: %s", entry)
            continue
        out[role] = dict(entry)
    return out


def _resolve_primary_role(
    cam_specs: List[Dict[str, Any]], override: Optional[str]
) -> str:
    """
    Pick the primary localization role. Honor an explicit override, else
    walk the family-agnostic preference list, else fall back to the first
    declared role. Mirrors build_camera_registry in the bimanual module so
    both paths agree on what "primary" means.
    """
    if override:
        return override
    declared = {spec.get("role") for spec in cam_specs}
    for candidate in _PRIMARY_ROLE_PREFERENCE:
        if candidate in declared:
            return candidate
    if cam_specs:
        return cam_specs[0].get("role", "")
    return ""


def build_camera_registry(
    profile, auto_open: bool = True, primary_role: Optional[str] = None
) -> CameraRegistry:
    """
    Build a role-keyed CameraRegistry from profile.cameras() for ANY family.

    Each entry's `type` selects the camera class via the shared
    _populate_registry helper:
      kinect / azure_kinect -> AzureKinectCamera
      realsense             -> RealSenseCamera
      zed_mini              -> ZEDMiniCamera
    When auto_open is True each device is opened immediately; set it False
    in tests to exercise the wiring without hardware.

    The single-arm live path differs: the dual-Kinect open in
    SPARKRealPipeline._init_kinects() carries master/subordinate sync,
    depth-mode, retry, and stagger semantics that depend on runtime device
    enumeration, none of which live in this declarative roster. The
    pipeline keeps opening the external Kinects through that path and wraps
    the already-open devices via CameraRegistry.from_open_devices(). This
    builder is used directly for families whose cameras open cleanly from
    config alone (e.g. bimanual, via build_bimanual_camera_registry) and
    as the from-config option for the wrist RealSense.
    """
    cam_specs = profile.cameras()
    registry = CameraRegistry()
    registry.primary_role = _resolve_primary_role(cam_specs, primary_role)
    return _populate_registry(registry, cam_specs, auto_open)


def registry_from_open_devices(
    entries: List[CameraEntry], primary_role: str = ""
) -> CameraRegistry:
    """
    Assemble a role-keyed registry around devices that are already open.

    Used by the single-arm pipeline so the _kinect / _kinect2 / _realsense
    slots and the unified registry share the SAME device + calibration
    objects (the slots are populated from these entries, not re-opened).
    This keeps the live capture path byte-for-byte identical while still
    giving every family one role-keyed registry.
    """
    registry = CameraRegistry()
    registry.primary_role = primary_role
    for entry in entries:
        registry.cameras[entry.role] = entry
        registry.roles.append(entry.role)
        if entry.arm in ("left", "right"):
            registry.wrist_for_arm[entry.arm] = entry.role
    # Stable order: primary first, then wrist arms, then the rest, matching
    # _populate_registry so the frontend tile layout is consistent.
    preferred = [
        registry.primary_role,
        registry.wrist_for_arm.get("left", ""),
        registry.wrist_for_arm.get("right", ""),
    ]
    ordered = [r for r in preferred if r and r in registry.cameras]
    extras = [r for r in registry.roles if r not in ordered]
    registry.roles = ordered + extras
    return registry

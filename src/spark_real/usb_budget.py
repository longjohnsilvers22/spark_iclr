"""
USB bandwidth / topology preflight for the multi-camera rig.

WHY THIS EXISTS
---------------
On the UR10e host every camera -- both Azure Kinects and the wrist
RealSense D435i -- hangs off ONE xHCI controller (``0000:00:14.0``), which
has one event ring serviced by one CPU. Microsoft's documented guidance is
one USB host controller per Azure Kinect at high resolution / high frame
rate. Running two Kinects at 1080P/30 plus a RealSense on that single
controller produced repeated xHCI transfer-ring desync in the kernel log
("Event dma ... not part of TD", "WARN Set TR Deq Ptr cmd failed") and,
downstream of that, whole-host deaths with no panic and no oops. See
``docs/MULTICAM_CRASH_MECHANISM.md``.

The failure mode we are removing is "the over-subscribed configuration
starts fine and kills the machine an hour later". This module makes it
"the over-subscribed configuration refuses to start, and says why".

HONESTY ABOUT THE NUMBERS
-------------------------
The per-controller ceiling here is a POLICY threshold, not a hardware
limit. Each Kinect sits on its own 5 Gbit/s root port, so no single link
is saturated; what is shared and provably strained is the controller's
scheduler, its single event ring, and the periodic (isochronous) schedule
that the Kinect color streams reserve for the life of the stream. The
default ceiling is set so that the profile which repeatedly killed this
host (2x 1080P/30 + WFOV_2X2BINNED/30 + RealSense, ~1.4 Gbit/s) is
refused and the tuned profile (2x 720P/15 + WFOV_2X2BINNED/15 +
RealSense, ~0.5 Gbit/s) is allowed. Treat it as a guardrail calibrated on
the crash record, and re-calibrate it if the topology changes.

Everything here is pure computation over inputs plus optional reads of
``/sys``; it never opens a device and never writes anything. Import is
cheap on purpose: ``server.py`` calls it from the early-Kinect block,
before torch/scipy load.
"""

from __future__ import annotations

import logging
import os
import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional, Sequence

logger = logging.getLogger(__name__)

# --------------------------------------------------------------------------
# Stream models
# --------------------------------------------------------------------------

COLOR_PIXELS = {
    "720P": 1280 * 720,
    "1080P": 1920 * 1080,
    "1440P": 2560 * 1440,
    "2160P": 3840 * 2160,
}

DEPTH_PIXELS = {
    "NFOV_UNBINNED": 640 * 576,
    "NFOV_2X2BINNED": 320 * 288,
    "WFOV_UNBINNED": 1024 * 1024,
    "WFOV_2X2BINNED": 512 * 512,
}

# Compressed color payload, bits per pixel. The Kinect color camera always
# ships MJPG on the wire (BGRA32 is a HOST-side conversion inside libk4a, so
# it costs CPU, not bus). 1080p30 measures 15-30 MB/s on this class of
# device, i.e. 1.9-3.9 bits/px; 3.0 is the midpoint.
COLOR_BITS_PER_PIXEL = 3.0

# Isochronous reservation overhead. Color is isoc at bInterval=1, so the
# controller RESERVES a fixed slice of every 125 us service interval for the
# life of the stream, sized by the selected alt setting rather than by the
# bytes that actually arrive. Observed on this host: ~190 Mbit/s of 1080p30
# payload sits inside a 262-524 Mbit/s reservation.
ISOC_RESERVATION_FACTOR = 2.0

# Depth mode ships depth AND active-IR, both 16-bit, at the configured fps.
DEPTH_BITS_PER_PIXEL_PER_FRAME = 2 * 16

# Default per-controller ceiling for camera traffic, Mbit/s. See the module
# docstring: policy threshold, calibrated on the crash record.
DEFAULT_CONTROLLER_BUDGET_MBIT = 1000.0

# Microsoft's documented usbfs ceiling for Azure Kinect is 32 MB for one
# device; librealsense's >2 MP guidance is where the 1000 on this host came
# from. usbfs_memory_mb is a CAP, not a reservation, so a large value
# allocates nothing -- it just removes the only kernel-side bound on how much
# DMA-mapped URB memory the camera stack can have outstanding at once.
USBFS_MEMORY_MB_MAX = 256


@dataclass(frozen=True)
class StreamLoad:
    """Estimated bus load of one camera, in Mbit/s."""

    label: str
    isoc_reserved_mbit: float
    bulk_mbit: float

    @property
    def total_mbit(self) -> float:
        return self.isoc_reserved_mbit + self.bulk_mbit


def kinect_load(
    label: str,
    color_resolution: str,
    depth_mode: str,
    fps: int,
    color_enabled: bool = True,
    depth_enabled: bool = True,
) -> StreamLoad:
    """Estimated controller load of one Azure Kinect at these settings."""
    isoc = 0.0
    if color_enabled:
        px = COLOR_PIXELS.get(str(color_resolution).upper(), COLOR_PIXELS["1080P"])
        payload = px * COLOR_BITS_PER_PIXEL * float(fps)
        isoc = payload * ISOC_RESERVATION_FACTOR / 1e6
    bulk = 0.0
    if depth_enabled:
        dpx = DEPTH_PIXELS.get(str(depth_mode).upper(), DEPTH_PIXELS["WFOV_2X2BINNED"])
        bulk = dpx * DEPTH_BITS_PER_PIXEL_PER_FRAME * float(fps) / 1e6
    return StreamLoad(label=label, isoc_reserved_mbit=isoc, bulk_mbit=bulk)


def realsense_load(
    label: str,
    width: int = 640,
    height: int = 480,
    fps: int = 15,
    color_only: bool = True,
) -> StreamLoad:
    """
    Estimated controller load of one RealSense.

    librealsense's RSUSB backend uses BULK endpoints only (verified with
    ``lsusb -v``: zero isochronous endpoints), so a RealSense reserves no
    periodic schedule -- it competes for whatever the Kinect color streams
    leave behind.
    """
    bits = width * height * 24 * float(fps)
    if not color_only:
        bits += width * height * 16 * float(fps)
    return StreamLoad(label=label, isoc_reserved_mbit=0.0, bulk_mbit=bits / 1e6)


# --------------------------------------------------------------------------
# Topology (sysfs)
# --------------------------------------------------------------------------

_PCI_IN_PATH = re.compile(r"/(\d{4}:[0-9a-f]{2}:[0-9a-f]{2}\.\d)/usb\d+/")

# 045e:097a/097b are the Kinect's internal hubs; 097c/097d the depth and 4K
# cameras. 8086:0b3a is the D435i.
KINECT_VENDOR = "045e"
KINECT_CAMERA_PRODUCTS = {"097c", "097d"}
REALSENSE_VENDOR = "8086"


@dataclass
class UsbTopology:
    """Which PCI xHCI controller each camera device enumerated behind."""

    controllers: Dict[str, List[str]] = field(default_factory=dict)
    available: bool = True

    @property
    def controller_count(self) -> int:
        return len(self.controllers)

    def busiest_controller(self) -> Optional[str]:
        if not self.controllers:
            return None
        return max(self.controllers, key=lambda k: len(self.controllers[k]))


def read_topology(sysfs_root: str = "/sys/bus/usb/devices") -> UsbTopology:
    """
    Map camera USB devices to their xHCI controller by resolving sysfs
    symlinks. Returns ``available=False`` (and no controllers) when sysfs
    is not readable, so callers degrade to warn-only instead of guessing.
    """
    root = Path(sysfs_root)
    if not root.is_dir():
        return UsbTopology(available=False)
    controllers: Dict[str, List[str]] = {}
    try:
        entries = sorted(root.iterdir())
    except OSError:
        return UsbTopology(available=False)
    for entry in entries:
        vid = _read_sysfs_id(entry / "idVendor")
        pid = _read_sysfs_id(entry / "idProduct")
        if vid is None or pid is None:
            continue
        if vid == KINECT_VENDOR and pid in KINECT_CAMERA_PRODUCTS:
            kind = "kinect"
        elif vid == REALSENSE_VENDOR:
            kind = "realsense"
        else:
            continue
        try:
            real = str(entry.resolve())
        except OSError:
            continue
        m = _PCI_IN_PATH.search(real + "/")
        if not m:
            continue
        controllers.setdefault(m.group(1), []).append(f"{entry.name} ({kind})")
    return UsbTopology(controllers=controllers, available=True)


def _read_sysfs_id(path: Path) -> Optional[str]:
    try:
        return path.read_text().strip().lower()
    except OSError:
        return None


def read_usbfs_memory_mb(
    path: str = "/sys/module/usbcore/parameters/usbfs_memory_mb",
) -> Optional[int]:
    try:
        return int(Path(path).read_text().strip())
    except (OSError, ValueError):
        return None


# --------------------------------------------------------------------------
# Verdict
# --------------------------------------------------------------------------


@dataclass
class BudgetVerdict:
    ok: bool
    level: str  # "ok" | "warn" | "error"
    message: str
    total_mbit: float
    isoc_mbit: float
    budget_mbit: float
    shared_controller: Optional[str]
    lines: List[str] = field(default_factory=list)

    def report(self) -> str:
        return "\n".join([self.message] + self.lines)


class UsbBudgetExceeded(RuntimeError):
    """Raised at startup instead of letting an over-budget rig run for an hour."""


def enforcement_mode() -> str:
    """
    ``SPARK_USB_BUDGET`` = ``error`` (default) | ``warn`` | ``off``.

    ``error`` refuses to start an over-budget camera configuration.
    ``warn`` logs and continues -- for deliberately re-running the old
    profile during the reproduction experiment.
    """
    val = os.environ.get("SPARK_USB_BUDGET", "error").strip().lower()
    return val if val in ("error", "warn", "off") else "error"


def check_camera_budget(
    loads: Sequence[StreamLoad],
    topology: Optional[UsbTopology] = None,
    budget_mbit: Optional[float] = None,
    usbfs_memory_mb: Optional[int] = None,
) -> BudgetVerdict:
    """
    Evaluate a planned camera configuration against the per-controller budget.

    Escalates to ``error`` ONLY when the estimated load exceeds the budget
    AND two or more camera devices are known to share one controller. A rig
    with one camera per controller is the arrangement Microsoft documents,
    so it is not this module's business to police it; and if topology could
    not be read we degrade to ``warn`` rather than block a rig we cannot see.
    """
    if budget_mbit is None:
        budget_mbit = float(
            os.environ.get(
                "SPARK_USB_CONTROLLER_BUDGET_MBIT", DEFAULT_CONTROLLER_BUDGET_MBIT
            )
        )
    total = sum(x.total_mbit for x in loads)
    isoc = sum(x.isoc_reserved_mbit for x in loads)

    shared = None
    if topology is not None and topology.available:
        busiest = topology.busiest_controller()
        if busiest is not None and len(topology.controllers[busiest]) >= 2:
            shared = busiest

    lines = [
        f"  {x.label}: {x.total_mbit:7.0f} Mbit/s "
        f"({x.isoc_reserved_mbit:.0f} reserved isoc + {x.bulk_mbit:.0f} bulk)"
        for x in loads
    ]
    lines.append(f"  TOTAL: {total:.0f} Mbit/s (budget {budget_mbit:.0f})")
    if shared:
        devs = ", ".join(topology.controllers[shared])
        lines.append(f"  shared xHCI controller {shared}: {devs}")
    elif topology is not None and not topology.available:
        lines.append("  USB topology unreadable (/sys); budget is advisory only")

    if usbfs_memory_mb is not None and usbfs_memory_mb > USBFS_MEMORY_MB_MAX:
        lines.append(
            f"  usbcore.usbfs_memory_mb={usbfs_memory_mb} (> {USBFS_MEMORY_MB_MAX}): "
            "no kernel-side ceiling on outstanding DMA-mapped URB memory. "
            "Microsoft's Azure Kinect guidance is 32; set 256 on the kernel "
            "cmdline (host change, not a code change)."
        )

    over = total > budget_mbit
    if not over:
        return BudgetVerdict(
            True,
            "ok",
            f"USB camera budget OK: {total:.0f}/{budget_mbit:.0f} Mbit/s",
            total,
            isoc,
            budget_mbit,
            shared,
            lines,
        )
    if shared is None:
        return BudgetVerdict(
            True,
            "warn",
            f"USB camera load {total:.0f} Mbit/s exceeds the {budget_mbit:.0f} "
            "Mbit/s per-controller budget, but the cameras are not known to "
            "share one controller; continuing.",
            total,
            isoc,
            budget_mbit,
            shared,
            lines,
        )
    return BudgetVerdict(
        False,
        "error",
        f"USB camera load {total:.0f} Mbit/s exceeds the {budget_mbit:.0f} "
        f"Mbit/s budget for a SINGLE shared xHCI controller ({shared}). "
        "This is the configuration class that hard-killed this host ~11 "
        "times. Lower kinect_fps / kinect_resolution / kinect_depth_mode, or "
        "move a camera to its own USB host controller. Set SPARK_USB_BUDGET="
        "warn to run it anyway (reproduction experiments only).",
        total,
        isoc,
        budget_mbit,
        shared,
        lines,
    )


def preflight(
    loads: Sequence[StreamLoad],
    sysfs_root: str = "/sys/bus/usb/devices",
    log: Optional[logging.Logger] = None,
) -> BudgetVerdict:
    """
    Run the budget check and act on ``SPARK_USB_BUDGET``.

    Raises :class:`UsbBudgetExceeded` in ``error`` mode. Always returns the
    verdict so callers can surface it.
    """
    log = log or logger
    mode = enforcement_mode()
    verdict = check_camera_budget(
        loads,
        topology=read_topology(sysfs_root),
        usbfs_memory_mb=read_usbfs_memory_mb(),
    )
    if verdict.level == "ok":
        log.info(verdict.report())
        return verdict
    if mode == "off":
        log.warning("SPARK_USB_BUDGET=off: %s", verdict.report())
        return verdict
    if verdict.level == "warn" or mode == "warn":
        log.warning(verdict.report())
        return verdict
    log.error(verdict.report())
    raise UsbBudgetExceeded(verdict.report())

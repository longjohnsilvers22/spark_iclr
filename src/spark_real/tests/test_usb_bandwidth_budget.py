"""
USB bandwidth-budget preflight tests.

Context: every camera on the UR10e host (both Kinects + the wrist D435i)
enumerates behind ONE xHCI controller, against Microsoft's one-controller-
per-Kinect guidance. The profile that ran there (2x 1080P/30 + WFOV/30 +
RealSense) produced repeated xHCI transfer-ring desync and whole-host
deaths. These tests pin the two things that keep that from happening
silently again: the shipped defaults are the tuned profile, and the
over-budget profile is REFUSED at startup rather than allowed to run for an
hour. Pure computation plus a fake sysfs tree -- zero hardware contact.
"""

import pytest

from spark_real import usb_budget as ub
from spark_real.pipeline_types import PipelineConfig


# 1. Shipped defaults are the tuned profile, in every place that has one.


def test_pipeline_config_defaults_are_the_tuned_profile():
    cfg = PipelineConfig()
    assert cfg.kinect_resolution == "720P"
    assert cfg.kinect_fps == 15
    # WFOV is retained deliberately: it is the only mode that keeps the whole
    # cell in one camera. Bandwidth comes out of resolution and fps instead.
    assert cfg.kinect_depth_mode == "WFOV_2X2BINNED"


def test_ur10e_yaml_pins_the_knobs_explicitly():
    """
    Leaving these keys commented out is what silently handed the rig the
    max-bandwidth defaults. They must be present, not inherited.
    """
    import yaml
    from pathlib import Path

    import spark_real

    p = Path(spark_real.__file__).parent / "configs" / "ur10e_default.yaml"
    raw = yaml.safe_load(p.read_text())
    assert raw["kinect_resolution"] == "720P"
    assert raw["kinect_fps"] == 15
    assert raw["kinect_depth_mode"] == "WFOV_2X2BINNED"


def test_server_early_path_fallbacks_match_pipeline_config():
    """
    server.py duplicates the defaults because it must not import heavy deps
    before libk4a. Duplication is fine; drift is not.
    """
    import spark_real.server as srv

    es = srv._early_kinect_settings.__doc__
    assert es  # documented duplication
    cfg = PipelineConfig()
    # Parse the literal fallbacks out of the function's own source so the
    # test fails if someone edits one side only.
    import inspect

    src = inspect.getsource(srv._early_kinect_settings)
    assert f'"kinect_resolution", "{cfg.kinect_resolution}"' in src
    assert f'"kinect_fps", {cfg.kinect_fps}' in src
    assert f'"kinect_depth_mode", "{cfg.kinect_depth_mode}"' in src


# 2. Load model: the two profiles land on the right side of the budget.


def _profile(resolution, depth_mode, fps):
    return [
        ub.kinect_load("k0", resolution, depth_mode, fps),
        ub.kinect_load("k1", resolution, depth_mode, fps),
        ub.realsense_load("wrist"),
    ]


def test_old_profile_is_over_budget_and_new_profile_is_under():
    old = sum(x.total_mbit for x in _profile("1080P", "WFOV_2X2BINNED", 30))
    new = sum(x.total_mbit for x in _profile("720P", "WFOV_2X2BINNED", 15))
    assert old > ub.DEFAULT_CONTROLLER_BUDGET_MBIT
    assert new < ub.DEFAULT_CONTROLLER_BUDGET_MBIT
    # Not a marginal call: the tuned profile is well under half the old one.
    assert new < old / 2


def test_depth_bulk_estimate_matches_the_wire_arithmetic():
    """WFOV_2X2BINNED = 512x512 depth + 512x512 IR, 16-bit, at fps."""
    load = ub.kinect_load(
        "k", "720P", "WFOV_2X2BINNED", 30, color_enabled=False
    )
    expected = 512 * 512 * 2 * 16 * 30 / 1e6  # ~252 Mbit/s
    assert load.bulk_mbit == pytest.approx(expected)
    assert load.isoc_reserved_mbit == 0.0


def test_realsense_reserves_no_periodic_schedule():
    """RSUSB backend is bulk-only: it competes for leftovers, reserves none."""
    load = ub.realsense_load("wrist", 640, 480, 15, color_only=True)
    assert load.isoc_reserved_mbit == 0.0
    assert load.bulk_mbit == pytest.approx(640 * 480 * 24 * 15 / 1e6)


# 3. Topology detection from a fake sysfs tree.


def _fake_sysfs(tmp_path, devices):
    """
    devices: {name: (vid, pid, controller)}. Builds the symlink shape the
    real tree has: /sys/bus/usb/devices/<name> -> .../<pci>/usb<N>/<name>.
    """
    root = tmp_path / "sys" / "bus" / "usb" / "devices"
    root.mkdir(parents=True)
    for name, (vid, pid, controller) in devices.items():
        real = tmp_path / "sys" / "devices" / "pci0000:00" / controller
        bus = name.split("-")[0]
        real = real / f"usb{bus}" / name
        real.mkdir(parents=True, exist_ok=True)
        (real / "idVendor").write_text(vid + "\n")
        (real / "idProduct").write_text(pid + "\n")
        (root / name).symlink_to(real)
    return str(root)


def test_topology_detects_all_cameras_on_one_controller(tmp_path):
    root = _fake_sysfs(
        tmp_path,
        {
            "2-1": ("8086", "0b3a", "0000:00:14.0"),
            "2-2.1": ("045e", "097d", "0000:00:14.0"),
            "2-2.2": ("045e", "097c", "0000:00:14.0"),
            "2-7.1": ("045e", "097d", "0000:00:14.0"),
            "2-7.2": ("045e", "097c", "0000:00:14.0"),
            "1-3": ("1d6b", "0002", "0000:00:14.0"),  # unrelated device
        },
    )
    topo = ub.read_topology(root)
    assert topo.available
    assert topo.controller_count == 1
    assert topo.busiest_controller() == "0000:00:14.0"
    assert len(topo.controllers["0000:00:14.0"]) == 5


def test_topology_detects_a_split_across_two_controllers(tmp_path):
    root = _fake_sysfs(
        tmp_path,
        {
            "2-1": ("8086", "0b3a", "0000:00:14.0"),
            "2-2.1": ("045e", "097d", "0000:00:14.0"),
            "4-1.1": ("045e", "097d", "0000:04:00.0"),
        },
    )
    topo = ub.read_topology(root)
    assert topo.controller_count == 2


def test_topology_unavailable_when_sysfs_missing(tmp_path):
    topo = ub.read_topology(str(tmp_path / "nope"))
    assert not topo.available
    assert topo.controllers == {}


# 4. Verdict policy.


def _shared_topology():
    return ub.UsbTopology(
        controllers={"0000:00:14.0": ["2-2.1 (kinect)", "2-7.1 (kinect)"]},
        available=True,
    )


def test_over_budget_on_a_shared_controller_is_an_error():
    v = ub.check_camera_budget(
        _profile("1080P", "WFOV_2X2BINNED", 30), topology=_shared_topology()
    )
    assert not v.ok and v.level == "error"
    assert "0000:00:14.0" in v.message


def test_tuned_profile_on_the_same_shared_controller_passes():
    v = ub.check_camera_budget(
        _profile("720P", "WFOV_2X2BINNED", 15), topology=_shared_topology()
    )
    assert v.ok and v.level == "ok"


def test_over_budget_without_a_shared_controller_only_warns():
    """One camera per controller is the documented arrangement; do not block it."""
    topo = ub.UsbTopology(
        controllers={
            "0000:00:14.0": ["2-2.1 (kinect)"],
            "0000:04:00.0": ["4-1.1 (kinect)"],
        },
        available=True,
    )
    v = ub.check_camera_budget(_profile("1080P", "WFOV_2X2BINNED", 30), topology=topo)
    assert v.ok and v.level == "warn"


def test_unreadable_topology_degrades_to_warn():
    v = ub.check_camera_budget(
        _profile("1080P", "WFOV_2X2BINNED", 30),
        topology=ub.UsbTopology(available=False),
    )
    assert v.ok and v.level == "warn"
    assert any("topology unreadable" in line for line in v.lines)


def test_usbfs_memory_cap_is_reported_when_oversized():
    v = ub.check_camera_budget(
        _profile("720P", "WFOV_2X2BINNED", 15),
        topology=_shared_topology(),
        usbfs_memory_mb=1000,
    )
    assert any("usbfs_memory_mb=1000" in line for line in v.lines)


# 5. preflight() enforcement modes.


def test_preflight_raises_in_default_error_mode(monkeypatch, tmp_path):
    root = _fake_sysfs(
        tmp_path,
        {
            "2-2.1": ("045e", "097d", "0000:00:14.0"),
            "2-7.1": ("045e", "097d", "0000:00:14.0"),
        },
    )
    monkeypatch.delenv("SPARK_USB_BUDGET", raising=False)
    with pytest.raises(ub.UsbBudgetExceeded):
        ub.preflight(_profile("1080P", "WFOV_2X2BINNED", 30), sysfs_root=root)


def test_preflight_warn_mode_allows_the_reproduction_run(monkeypatch, tmp_path):
    root = _fake_sysfs(
        tmp_path,
        {
            "2-2.1": ("045e", "097d", "0000:00:14.0"),
            "2-7.1": ("045e", "097d", "0000:00:14.0"),
        },
    )
    monkeypatch.setenv("SPARK_USB_BUDGET", "warn")
    v = ub.preflight(_profile("1080P", "WFOV_2X2BINNED", 30), sysfs_root=root)
    assert v.level == "error"  # verdict still says error; mode downgrades action


def test_preflight_passes_the_tuned_profile(monkeypatch, tmp_path):
    root = _fake_sysfs(
        tmp_path,
        {
            "2-2.1": ("045e", "097d", "0000:00:14.0"),
            "2-7.1": ("045e", "097d", "0000:00:14.0"),
        },
    )
    monkeypatch.delenv("SPARK_USB_BUDGET", raising=False)
    v = ub.preflight(_profile("720P", "WFOV_2X2BINNED", 15), sysfs_root=root)
    assert v.ok and v.level == "ok"

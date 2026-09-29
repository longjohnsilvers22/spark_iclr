"""
Robot driver factory for SPARK.

Centralises the embodiment switch: given a `family` string (and the same
`robot_ip` / `frequency` arguments the UR10e shim has used since day one),
returns a driver instance that implements the **UR10eDriver shim interface**
(connect/disconnect/get_joint_positions/get_tcp_pose/move_to_joint_config/
servo_joint/open_gripper/close_gripper/...).

This module exists so call sites like ``run.py`` and ``server.py`` don't have
to know which driver class maps to which family.  Each driver module guards
its own optional backend (``franky``, ``pylibfranka``, ``rtde_*``, the bamboo
client, ``unitree_sdk2_python``) behind a module-level ``try/except`` so
importing a driver class on a host that lacks its backend is harmless; the
backend's absence surfaces only when the driver is actually constructed and
connected.

If a brand-new robot family shows up later, add a branch here; no other
file in ``spark_real`` should ever need to know which driver class to
instantiate.
"""

from __future__ import annotations

import logging
import os
from typing import Any

from spark_real.control.ur10e_driver import UR10eDriver
from spark_real.robots.driver_protocol import missing_members
from spark_real.robots.franka import FrankaDriver
from spark_real.robots.franka.franka_torque_driver import FrankaTorqueDriver
from spark_real.robots.franka.franka_bamboo_driver import FrankaBambooDriver
from spark_real.robots.g1.g1_driver import G1Driver

# NOTE: BimanualFrankaDriver is imported lazily inside make_robot_driver's
# bimanual branch. Its module chains into the SSG48 gripper, which hard-imports
# Spectral_BLDC (a BLDC motor lib not installed on single-arm hosts). Importing
# it eagerly here crashes server startup for EVERY family (ur10e included);
# deferring it keeps non-bimanual bring-up independent of that dependency.

logger = logging.getLogger(__name__)


# Public registry of known families. Kept as a module-level constant so
# server.py can validate its --robot flag against the same source of truth.
KNOWN_FAMILIES = ("ur10e", "franka", "g1", "bimanual_franka")


def make_robot_driver(
    family: str,
    robot_ip: str,
    frequency: float | None = None,
    **kwargs: Any,
):
    """
    Instantiate the right driver class for a robot family.

    Args:
        family: one of "ur10e", "franka", "g1".  Case-insensitive.
        robot_ip: FCI / RTDE / network address.  For the G1 (which talks
            DDS over a NIC rather than to a controller IP), this string is
            interpreted as the network interface name (e.g. "eth0").  Pass
            an empty string to fall back to the driver's default.
        frequency: control loop hz; ``None`` means use the driver's own
            default (UR10e: 500 Hz, Franka: 1 kHz, G1: control_rate from
            the SDK).  Ignored for G1 since the SDK fixes the rate.
        **kwargs: forwarded verbatim to the driver constructor.  Use this
            for gripper config, dof selection, hand enablement, etc.
            See each driver's ``__init__`` for accepted keys.

    Returns:
        A driver instance implementing the UR10eDriver shim interface.

    Raises:
        ValueError: if ``family`` is not one of the known families.
        ImportError: if the chosen backend's Python package isn't
            installed.  (We let the original ImportError bubble up so the
            traceback points at the missing dependency.)
        TypeError: if the built driver lacks a member of the RobotDriver
            protocol (robots/driver_protocol.py), so the gap shows up here
            and not mid-episode.
    """
    driver = _build_driver(family, robot_ip, frequency, **kwargs)
    missing = missing_members(driver)
    if missing:
        raise TypeError(
            f"{type(driver).__name__} does not implement RobotDriver; "
            f"missing: {missing}"
        )
    return driver


def _build_driver(
    family: str,
    robot_ip: str,
    frequency: float | None = None,
    **kwargs: Any,
):
    fam = (family or "").strip().lower()
    if fam not in KNOWN_FAMILIES:
        raise ValueError(
            f"Unknown robot family {family!r}. "
            f"Expected one of: {', '.join(KNOWN_FAMILIES)}."
        )

    if fam == "ur10e":
        freq = 500.0 if frequency is None else float(frequency)
        logger.info("make_robot_driver: UR10e @ %s (%.1f Hz)", robot_ip, freq)
        return UR10eDriver(robot_ip, freq, **kwargs)

    if fam == "franka":
        freq = 1000.0 if frequency is None else float(frequency)

        # SPARK_FRANKA_BACKEND selects between:
        #   "franky"  - motion generators (default, may cause reflexes)
        #   "torque"  - pylibfranka impedance at 1kHz (reflex-free)
        #   "bamboo"  - Bamboo C++ joint impedance at 1kHz (reflex-free,
        #               requires RunBambooController running separately)
        backend = os.environ.get("SPARK_FRANKA_BACKEND", "franky").strip().lower()
        if backend in ("torque", "bamboo"):
            # Only FrankaDriver (franky) takes the profile's gripper= and
            # collision= kwargs that pipeline_init passes for this family;
            # the other two backends reject them with TypeError.
            kwargs.pop("gripper", None)
            kwargs.pop("collision", None)
        if backend == "torque":
            logger.info(
                "make_robot_driver: Franka FR3 @ %s (%.1f Hz, torque backend)",
                robot_ip,
                freq,
            )
            return FrankaTorqueDriver(robot_ip, freq, **kwargs)

        if backend == "bamboo":
            logger.info(
                "make_robot_driver: Franka FR3 @ %s (%.1f Hz, bamboo backend)",
                robot_ip,
                freq,
            )
            # Bamboo takes (ip, port); the rate is fixed by the C++ node, so
            # freq must not land in the port slot.
            return FrankaBambooDriver(ip=robot_ip, port=kwargs.pop("port", 5555))

        # Default: franky motion-generator backend
        logger.info("make_robot_driver: Franka FR3 @ %s (%.1f Hz)", robot_ip, freq)
        return FrankaDriver(robot_ip, freq, **kwargs)

    if fam == "g1":
        # G1Driver takes net_iface, not robot_ip; treat the ip slot as the
        # NIC name for the G1.  Empty string -> let driver use its default.
        net_iface = robot_ip or "eth0"
        logger.info("make_robot_driver: Unitree G1 on iface %s", net_iface)
        # frequency is informational on G1 (SDK sets the rate); accept and
        # ignore here so the call signature is uniform across families.
        return G1Driver(net_iface=net_iface, **kwargs)

    if fam == "bimanual_franka":
        # `robot_ip` is unused on this family; the per-arm IPs come from
        # kwargs (or the bimanual config YAML loaded by server.py).
        freq = 1000.0 if frequency is None else float(frequency)
        # If the caller passed `robot_ip` (a single string), fall back to
        # using it for the right arm so the existing --ip CLI flag still
        # has a useful effect; left/right ip override via kwargs takes
        # precedence.
        if robot_ip and "right_ip" not in kwargs:
            kwargs["right_ip"] = robot_ip
        # Lazy import: pulls in the SSG48/Spectral_BLDC chain only when a
        # bimanual rig is actually requested (see note at module top).
        from spark_real.robots.bimanual_franka import BimanualFrankaDriver

        logger.info(
            "make_robot_driver: Bimanual Franka (Panda+FR3) @ %s / %s (%.1f Hz)",
            kwargs.get("left_ip", "<default>"),
            kwargs.get("right_ip", "<default>"),
            freq,
        )
        return BimanualFrankaDriver(frequency=freq, **kwargs)

    # Unreachable: KNOWN_FAMILIES guard above.
    raise AssertionError(f"unhandled family: {fam!r}")

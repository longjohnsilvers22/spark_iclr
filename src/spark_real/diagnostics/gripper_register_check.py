"""Robotiq 2F-85 register diagnostic (UR10e).

Connects the standard UR10eDriver (which auto-activates the gripper), then
reads the Robotiq state that is surfaced through RTDE output integer
registers: position (reg 12, 0-100 -> scaled 0-255) and object-detected
(reg 13, 0/1), and exercises open/close, sampling the registers after each
move. Arm does NOT move; only the gripper actuates.

Run:  conda activate sam3
      cd ~/spark/src && python -m spark_real.gripper_register_check --ip 192.168.56.101
"""

import argparse
import logging
import time

from spark_real.control.ur10e_driver import UR10eDriver

logging.basicConfig(level=logging.INFO, format="%(levelname)s %(name)s: %(message)s")
log = logging.getLogger("gripper_check")


def connect_gripper_only(drv: UR10eDriver) -> None:
    """Bring up ONLY the gripper transport, skipping RTDEControlInterface.

    A fieldbus (EtherNet/IP / PROFINET / MODBUS) is enabled and
    reserves the RTDE *input* registers, so RTDEControlInterface fails hard
    with "RTDE input registers already in use" (the same root cause the driver
    already documents for RTDEIOInterface). The gripper needs NONE of
    RTDEControl; it is driven over the primary URScript socket (port 30002)
    and its state is read back over RTDE *receive* (output int regs 12/13). We
    set up just those two transports and leave _rtde_c = None, which also means
    no arm control script is ever uploaded, so the arm is definitively
    stationary for this test.
    """
    import rtde_receive

    drv._dashboard_preflight()  # best-effort; warns if UR is in LOCAL mode
    drv._rtde_c = None  # no arm control script -> arm cannot move
    drv._rtde_r = rtde_receive.RTDEReceiveInterface(drv.robot_ip)
    drv._rtde_io = None
    drv._open_urscript_socket()
    drv._connected = True
    drv._load_gripper_functions()
    drv.activate_gripper()
    log.info("Gripper-only bring-up complete. Robot mode: %s", drv.get_robot_mode())


def read_state(drv: UR10eDriver) -> dict:
    """Sample the gripper registers via the driver's spin-free readers."""
    drv._publish_gripper_state()  # push Robotiq state -> output int regs 12/13
    pos_reg = drv._rtde_r.getOutputIntRegister(drv._POS_REGISTER)   # 0-100
    obj_reg = drv._rtde_r.getOutputIntRegister(drv._OBJ_REGISTER)   # 0/1
    return {
        "pos_reg12_0_100": pos_reg,
        "pos_0_255": drv.get_gripper_position(),   # scaled convention
        "width_m": round(drv.get_gripper_width(), 4),
        "obj_reg13": obj_reg,
        "object_detected": drv.is_object_detected(),
    }


def report(label: str, st: dict) -> None:
    log.info(
        "%-18s reg12=%3d (0-255=%.0f, width=%.3fm)  reg13=%d object_detected=%s",
        label,
        st["pos_reg12_0_100"],
        st["pos_0_255"],
        st["width_m"],
        st["obj_reg13"],
        st["object_detected"],
    )


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--ip", default="192.168.56.101", help="UR10e controller IP")
    ap.add_argument("--cycles", type=int, default=1, help="open/close cycles")
    args = ap.parse_args()

    drv = UR10eDriver(args.ip)
    log.info("Connecting to %s (auto-activates gripper)...", args.ip)
    try:
        drv.connect()
    except RuntimeError as e:
        if "input register" in str(e).lower():
            log.warning(
                "RTDEControl unavailable (%s). A fieldbus is enabled "
                "that reserves RTDE input registers. Falling back to "
                "gripper-only bring-up (arm stays stationary).",
                e,
            )
            connect_gripper_only(drv)
        else:
            raise
    try:
        report("initial", read_state(drv))

        for i in range(args.cycles):
            log.info("--- cycle %d/%d ---", i + 1, args.cycles)

            log.info("OPEN gripper")
            drv.open_gripper()
            time.sleep(1.0)
            report("after open", read_state(drv))

            log.info("CLOSE gripper")
            drv.close_gripper()
            time.sleep(1.0)
            report("after close", read_state(drv))

        # leave it open
        log.info("OPEN gripper (final)")
        drv.open_gripper()
        time.sleep(1.0)
        report("final open", read_state(drv))
    finally:
        drv.disconnect()
        log.info("Disconnected.")


if __name__ == "__main__":
    main()

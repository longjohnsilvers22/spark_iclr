"""
Standalone Source Robotics SSG-48 reset / recalibration utility.

Ported from the external ROS2 gripper driver's ``gripper_reset.py``
so the SPARK deployment stays self-contained (no ROS workspace
dependency). Behaviour is intentionally identical: hard-reset, activate,
trigger ``Send_gripper_calibrate``, and poll until the firmware reports
``gripper_calibrated == 1``.

Run while the SPARK server is NOT connected to the gripper (the CAN bus
is exclusive). Typical use:

    cd ~/spark/src
    python -m spark_real.robots.bimanual_franka.ssg48_reset \
        --channel /dev/ttyACM0     # right gripper
    python -m spark_real.robots.bimanual_franka.ssg48_reset \
        --channel /dev/ttyACM1     # left gripper

Exits 0 on success, 1 on timeout/error.
"""

from __future__ import annotations

import argparse
import sys
import time

import Spectral_BLDC as Spectral

_BUSTYPE = "slcan"
_BITRATE = 1_000_000
_MAX_CURRENT_MA = 1300


def _recv_once(comm, motor, timeout: float = 0.05) -> bool:
    try:
        msg, uid = comm.receive_can_messages(timeout=timeout)
    except Exception:
        return False
    if msg is None or uid is None:
        return False
    try:
        motor.UnpackData(msg, uid)
        return True
    except Exception:
        return False


def _drain(comm, motor, n: int = 20, recv_timeout: float = 0.05) -> None:
    for _ in range(n):
        if not _recv_once(comm, motor, recv_timeout):
            break


def _send_gripper(
    motor,
    *,
    activate: int,
    action: int,
    position: int = 0,
    speed: int = 20,
    current: int = 500,
    release_dir: int = 0,
) -> bool:
    current = max(0, min(_MAX_CURRENT_MA, current))
    try:
        motor.Send_gripper_data_pack(
            position, speed, current, activate, action, 0, release_dir
        )
        return True
    except Exception as e:
        print(f"send error: {e}")
        return False


def _poll_until(
    comm,
    motor,
    attr: str,
    expected_value: int,
    timeout_s: float,
    send_fn,
    *,
    send_interval: float = 0.15,
    label: str = "",
) -> bool:
    deadline = time.time() + timeout_s
    last_send = 0.0
    while time.time() < deadline:
        now = time.time()
        if now - last_send >= send_interval:
            send_fn()
            last_send = now
        _recv_once(comm, motor, timeout=0.05)
        val = getattr(motor, attr, None)
        pos = getattr(motor, "gripper_position", "?")
        cal = getattr(motor, "gripper_calibrated", "?")
        act = getattr(motor, "gripper_activated", "?")
        print(
            f"  {label}  {attr}={val}  pos={pos}  " f"activated={act}  calibrated={cal}"
        )
        if val == expected_value:
            return True
    return False


def main(argv=None) -> int:
    p = argparse.ArgumentParser(
        description="Reset / recalibrate a Source Robotics SSG-48 gripper."
    )
    p.add_argument(
        "--channel",
        default="/dev/ttyACM0",
        help="slcan device (e.g. /dev/ttyACM0 right, "
        "/dev/ttyACM1 left)",
    )
    p.add_argument("--node-id", type=int, default=0)
    p.add_argument(
        "--timeout",
        type=float,
        default=60.0,
        help="Max seconds to wait for calibration to finish.",
    )
    p.add_argument(
        "--release-dir",
        type=int,
        default=0,
        choices=[0, 1],
        help="Release direction. Try 1 if calibration runs backwards.",
    )
    args = p.parse_args(argv)

    print(
        f"Connecting: {_BUSTYPE} on {args.channel} @ {_BITRATE} bps, "
        f"node_id={args.node_id}"
    )
    try:
        comm = Spectral.CanCommunication(
            bustype=_BUSTYPE, channel=args.channel, bitrate=_BITRATE
        )
        motor = Spectral.SpectralCAN(node_id=args.node_id, communication=comm)
    except Exception as e:
        print(f"ERROR: failed to connect: {e}")
        print("Make sure no other process (the SPARK server) is using the port.")
        return 1

    success = False
    try:
        print("\nStep 1: Hard reset (board reboots ~3s)...")
        motor.Send_Reset()
        time.sleep(3.0)
        _drain(comm, motor)

        print("\nStep 2: Activating (polling for gripper_activated==1)...")
        activated = _poll_until(
            comm,
            motor,
            attr="gripper_activated",
            expected_value=1,
            timeout_s=10.0,
            send_fn=lambda: _send_gripper(
                motor, activate=1, action=0, release_dir=args.release_dir
            ),
            label="activating",
        )
        if not activated:
            print("WARNING: gripper_activated never became 1, continuing anyway.")
        else:
            print("Gripper activated.")

        print("\nStep 3: Moving to fully-open before calibration sweep...")
        _send_gripper(
            motor,
            activate=1,
            action=1,
            position=0,
            speed=20,
            current=500,
            release_dir=args.release_dir,
        )
        time.sleep(2.0)
        _drain(comm, motor)

        # Send_gripper_calibrate produces no response: poll with empty
        # Send_gripper_data_pack to keep status frames flowing while the
        # firmware sweeps both limits. Never send action=1 during this
        # window: it aborts the sweep.
        print(
            f"\nStep 4: Calibrating (timeout={args.timeout}s, do not "
            f"obstruct jaws)..."
        )
        motor.Send_gripper_calibrate()
        time.sleep(0.1)

        deadline = time.time() + args.timeout
        last_poll = 0.0
        POLL_INTERVAL = 0.2
        while time.time() < deadline:
            now = time.time()
            if now - last_poll >= POLL_INTERVAL:
                try:
                    motor.Send_gripper_data_pack()  # 0-byte poll frame
                except Exception as e:
                    print(f"poll error: {e}")
                last_poll = now
            _recv_once(comm, motor, timeout=0.05)
            cal = getattr(motor, "gripper_calibrated", None)
            pos = getattr(motor, "gripper_position", "?")
            act = getattr(motor, "gripper_activated", "?")
            print(
                f"  calibrating  gripper_calibrated={cal}  pos={pos}  "
                f"activated={act}"
            )
            if cal == 1:
                success = True
                break
    finally:
        try:
            comm.bus.shutdown()
        except Exception:
            pass

    if success:
        print("\nCalibration complete: gripper is ready.")
        return 0
    print(f"\nCalibration did not complete within {args.timeout}s.")
    print("Check: power supply, CAN wiring, jaw clearance.")
    return 1


if __name__ == "__main__":
    sys.exit(main())

"""Robotiq 2F-85 position-register sweep + staleness probe (UR10e).

Two experiments, arm STATIONARY throughout (only the gripper actuates):

  A. COMMANDED SWEEP: step the commanded position 0..100% closed and read
     the position register (reg 12) back after a settle. If the register is
     "live" the read tracks the command monotonically; if it were globally
     stale it would sit at one value (the ~226/0-255 claim). Repeats each
     read N times to expose refresh noise.

  B. STALENESS / LATENCY PROBE: command a big open->close swing, then poll
     the register at increasing delays to measure how long it takes to reflect
     the true jaw position. This characterizes the exact "stale read"
     documented in executor_grasp.py (the reason grasp-verify uses TCP force +
     the gOBJ object-detect flag instead of this register).

Run:  conda activate sam3
      cd ~/spark/src && python -m spark_real.gripper_sweep --ip 192.168.56.101
"""

import argparse
import logging
import time

import numpy as np

from spark_real.control.ur10e_driver import UR10eDriver

logging.basicConfig(level=logging.INFO, format="%(levelname)s %(name)s: %(message)s")
log = logging.getLogger("gripper_sweep")


def raw_regs(drv: UR10eDriver, publish: bool = True):
    """Read reg12 (pos 0-100) + reg13 (object 0/1). publish=False skips the
    URScript write and reads the LAST-published values (shows staleness)."""
    if publish:
        drv._publish_gripper_state()
    return (
        drv._rtde_r.getOutputIntRegister(drv._POS_REGISTER),
        drv._rtde_r.getOutputIntRegister(drv._OBJ_REGISTER),
    )


def commanded_sweep(drv: UR10eDriver, steps, samples: int):
    """Experiment A: command each position, read reg12 back `samples` times."""
    rows = []
    for pct in steps:
        drv.set_gripper_position(pct / 100.0)  # 0.0 open .. 1.0 closed
        time.sleep(0.6)
        reads = []
        obj = 0
        for _ in range(samples):
            p, o = raw_regs(drv, publish=True)
            reads.append(p)
            obj = o
            time.sleep(0.1)
        arr = np.array(reads, dtype=float)
        width = round((1.0 - np.clip(arr.mean() / 100.0, 0, 1)) * drv.GRIPPER_MAX_WIDTH_M, 4)
        rows.append(
            {
                "cmd_pct_closed": pct,
                "reg12_mean": round(arr.mean(), 1),
                "reg12_min": int(arr.min()),
                "reg12_max": int(arr.max()),
                "reg12_0_255": round(arr.mean() * 2.55),
                "width_m": width,
                "reg13_obj": obj,
            }
        )
        log.info(
            "cmd=%3d%% -> reg12 mean=%5.1f [%d..%d] (0-255=%3.0f) width=%.3fm obj=%d",
            pct, arr.mean(), arr.min(), arr.max(), arr.mean() * 2.55, width, obj,
        )
    return rows


def latency_probe(drv: UR10eDriver, delays):
    """Experiment B: after a full open->close swing, poll reg12 at increasing
    post-command delays. Reads WITHOUT an intervening re-publish first (to see
    the stale last value), then WITH publish (to see the refreshed value)."""
    log.info("Latency probe: OPEN then CLOSE, sampling register refresh...")
    drv.open_gripper()
    time.sleep(1.0)
    stale_before, _ = raw_regs(drv, publish=True)  # ~1 (open) now published
    log.info("  pre-close published reg12 = %d (expect ~open)", stale_before)

    # Command close but immediately probe. close_gripper() blocks on
    # rq_move_and_wait, so by return the jaws are closed; the question is how
    # fast the READ-BACK register reflects it at various publish delays.
    drv.close_gripper()
    rows = []
    for d in delays:
        time.sleep(d)
        no_pub, _ = raw_regs(drv, publish=False)  # last-published (may lag)
        pub, obj = raw_regs(drv, publish=True)  # force refresh
        rows.append(
            {"delay_s": d, "reg12_no_publish": no_pub, "reg12_after_publish": pub, "reg13_obj": obj}
        )
        log.info("  +%.2fs: no_publish=%3d  after_publish=%3d  obj=%d", d, no_pub, pub, obj)
    return rows


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--ip", default="192.168.56.101")
    ap.add_argument("--samples", type=int, default=5, help="reads per sweep step")
    args = ap.parse_args()

    steps = [0, 10, 20, 30, 40, 50, 60, 70, 80, 90, 100, 50, 0]  # up, then back down
    delays = [0.0, 0.1, 0.2, 0.4, 0.8]

    drv = UR10eDriver(args.ip)
    log.info("Connecting to %s...", args.ip)
    drv.connect()
    try:
        log.info("==== Experiment A: commanded position sweep ====")
        a_rows = commanded_sweep(drv, steps, args.samples)
        log.info("==== Experiment B: staleness / latency probe ====")
        b_rows = latency_probe(drv, delays)

        # Machine-readable dump for the caller to lift into a report.
        import json

        print("SWEEP_JSON_A=" + json.dumps(a_rows))
        print("SWEEP_JSON_B=" + json.dumps(b_rows))

        drv.open_gripper()  # park open
    finally:
        drv.disconnect()
        log.info("Disconnected.")


if __name__ == "__main__":
    main()

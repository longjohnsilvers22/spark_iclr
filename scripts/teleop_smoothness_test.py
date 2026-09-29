#!/usr/bin/env python3
"""Direct teleop smoothness test, no gamepad, no UI. Sends Cartesian
velocity commands to /api/velocity at 30 Hz and samples TCP pose at
~50 Hz simultaneously. Computes per-tick velocity (numerical derivative
of TCP position) and reports the variance across samples. Smooth motion
= low variance; jittery = high variance.

Usage:
  python teleop_smoothness_test.py
"""
import json
import threading
import time

import numpy as np
import requests

API = "http://localhost:8888"
COMMAND_HZ = 30.0           # frontend-matching cadence
COMMAND_DT = 1.0 / COMMAND_HZ
SAMPLE_HZ = 50.0            # pose sample rate
SAMPLE_DT = 1.0 / SAMPLE_HZ
DURATION_S = 2.5            # per-direction
LIN = 0.07                  # m/s, matches franka teleop profile


def home_robot():
    print("homing...")
    r = requests.post(f"{API}/api/home", timeout=20)
    r.raise_for_status()
    time.sleep(1.0)


# Send zero velocity then call /api/stop.
def stop_robot():
    try:
        requests.post(f"{API}/api/velocity",
                      json={"vx": 0, "vy": 0, "vz": 0,
                            "wrx": 0, "wry": 0, "wrz": 0,
                            "duration": 0.05}, timeout=2)
    except Exception:
        pass
    try:
        requests.post(f"{API}/api/stop", timeout=3)
    except Exception:
        pass


# Read current TCP position as numpy [x, y, z].
def get_tcp():
    r = requests.get(f"{API}/api/robot_state", timeout=2)
    r.raise_for_status()
    return np.array(r.json()["tcp_pose"][:3])


# Send sustained velocity, sample TCP. Returns (times, positions).
def drive_axis(label, vx, vy, vz, duration_s=DURATION_S):
    stop_event = threading.Event()
    samples = []  # list of (t, x, y, z)

    def cmd_thread():
        t0 = time.time()
        while not stop_event.is_set():
            try:
                requests.post(f"{API}/api/velocity",
                              json={"vx": vx, "vy": vy, "vz": vz,
                                    "wrx": 0, "wry": 0, "wrz": 0,
                                    "duration": 0.05}, timeout=0.5)
            except Exception:
                pass
            t = time.time() - t0
            if t >= duration_s:
                stop_event.set()
                break
            target = t0 + (int((time.time() - t0) / COMMAND_DT) + 1) * COMMAND_DT
            time.sleep(max(0, target - time.time()))

    def sample_thread():
        t0 = time.time()
        while not stop_event.is_set():
            try:
                tcp = get_tcp()
                samples.append((time.time() - t0, *tcp.tolist()))
            except Exception:
                pass
            target = t0 + (len(samples) + 1) * SAMPLE_DT
            time.sleep(max(0, target - time.time()))

    print(f"  driving {label} (vx={vx:.2f}, vy={vy:.2f}, vz={vz:.2f}) "
          f"for {duration_s:.1f}s ...")
    th_cmd = threading.Thread(target=cmd_thread, daemon=True)
    th_smp = threading.Thread(target=sample_thread, daemon=True)
    th_cmd.start()
    th_smp.start()

    # Wait both threads
    th_cmd.join(timeout=duration_s + 2)
    th_smp.join(timeout=2)

    return np.array(samples)  # shape (N, 4): t, x, y, z


# Compute per-sample velocity and report smoothness.
def analyze(label, samples):
    if len(samples) < 5:
        print(f"  {label}: too few samples ({len(samples)}); skipping")
        return
    t = samples[:, 0]
    pos = samples[:, 1:]
    # Numerical velocity between consecutive samples.
    dt = np.diff(t)
    dpos = np.diff(pos, axis=0)
    vel = dpos / dt[:, None]  # m/s per axis per sample

    # Drop first 5 samples (acceleration ramp) and last 5 (deceleration).
    if len(vel) > 12:
        vel = vel[5:-5]

    mean_v = vel.mean(axis=0)
    std_v = vel.std(axis=0)
    max_v = vel.max(axis=0)
    min_v = vel.min(axis=0)
    # Smoothness = coefficient of variation (std / mean), per axis
    cv = std_v / (np.abs(mean_v) + 1e-9)

    print(f"  {label}: {len(samples)} samples, {len(vel)} velocity samples")
    print(f"    mean vel (m/s): x={mean_v[0]:+.4f}  y={mean_v[1]:+.4f}  z={mean_v[2]:+.4f}")
    print(f"    std  vel (m/s): x={std_v[0]:.4f}  y={std_v[1]:.4f}  z={std_v[2]:.4f}")
    print(f"    cv (std/|mean|): x={cv[0]:.2f}  y={cv[1]:.2f}  z={cv[2]:.2f}  (lower = smoother)")
    print(f"    range (m/s):    x=[{min_v[0]:+.3f}, {max_v[0]:+.3f}]  "
          f"y=[{min_v[1]:+.3f}, {max_v[1]:+.3f}]  z=[{min_v[2]:+.3f}, {max_v[2]:+.3f}]")


def main():
    print("teleop smoothness test")
    print()

    home_robot()
    p0 = get_tcp()
    print(f"home TCP: ({p0[0]:+.3f}, {p0[1]:+.3f}, {p0[2]:+.3f})")
    print()

    print("Test 1: +Y (left/right): user reports SMOOTH")
    s = drive_axis("+Y (left/right)", 0, LIN, 0)
    analyze("+Y", s)
    stop_robot()
    time.sleep(1)

    home_robot()
    print()
    print("Test 2: +X (forward/back): user reports JITTERY")
    s = drive_axis("+X (forward/back)", LIN, 0, 0)
    analyze("+X", s)
    stop_robot()
    time.sleep(1)

    home_robot()
    print()
    print("Test 3: +Z (up/down): user reports JITTERY")
    s = drive_axis("+Z (up/down)", 0, 0, LIN)
    analyze("+Z", s)
    stop_robot()

    print()
    print("done")
    print("If x cv ~ y cv ~ z cv (all small, < 0.1): motion equally smooth in all axes.")
    print("If x or z cv >> y cv: confirms axis-direction-dependent jitter.")


if __name__ == "__main__":
    main()

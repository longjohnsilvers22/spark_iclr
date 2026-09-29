#!/usr/bin/env python3
"""Interactive grasp calibration profile.

Runs a series of width-targeted grasps via /api/grasp_test and records
metrics for tuning the per-category target_width / force defaults
in spark_planner.py and skills/grasping.py.

Usage:
    python scripts/grasp_profile.py
    python scripts/grasp_profile.py --host localhost --port 8888
    python scripts/grasp_profile.py --csv my_run.csv  --only plushie,phone

Workflow per object: the script tells you what to place between the
jaws and the suggested target width, you place it, press Enter, the
gripper closes after a 3-second countdown, measurements are printed
and appended to a CSV.

Single-pass: each object gets one trial at the default width + force.
Pass --sweep to also try (target_width - 5mm) and (target_width + 5mm)
so you can see how forgiving each object is to width error.
"""

import argparse
import csv
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

import requests


# Default profile: per-object suggested width (m) + force (N).
# Width is the FINAL jaw separation expected on a successful grip, i.e.
# slightly less than the object's grip dimension. Force is intentionally
# low for compressible / fragile items so the jaws stop on contact
# instead of crushing through.
DEFAULT_PROFILE = [
    ("plushie",      0.025, 18.0),
    ("phone",        0.008, 50.0),
    ("block",        0.025, 60.0),
    ("spoon handle", 0.006, 40.0),
    ("bowl rim",     0.005, 35.0),
    ("can",          0.045, 25.0),   # low force: crushable
    ("pen",          0.008, 30.0),
    ("cup",          0.025, 50.0),
]


@dataclass
class TrialResult:
    label: str
    target_width: float
    force: float
    achieved_width: float
    libfranka_grasp_success: bool
    is_grasped_flag: Optional[bool]
    delta_force_mag: float
    inferred_holding: bool


def countdown(seconds: float):
    for s in range(int(seconds), 0, -1):
        sys.stdout.write(f"\r  closing in {s} ... ")
        sys.stdout.flush()
        time.sleep(1)
    print("\r  closing now            ")


def run_trial(base_url: str, label: str, target_width: float,
              force: float, settle_s: float = 1.5) -> Optional[TrialResult]:
    try:
        r = requests.post(
            f"{base_url}/api/grasp_test",
            json={
                "label": label,
                "target_width": target_width,
                "force": force,
                "settle_s": settle_s,
                "release_after": True,
            },
            timeout=30.0,
        )
    except requests.RequestException as exc:
        print(f"  ERROR: request failed: {exc}")
        return None

    if r.status_code != 200:
        print(f"  ERROR: HTTP {r.status_code}: {r.text[:200]}")
        return None

    d = r.json()
    if "error" in d:
        print(f"  ERROR: {d['error']}")
        return None

    return TrialResult(
        label=d["label"],
        target_width=d["target_width"],
        force=d["force"],
        achieved_width=d["achieved_width"],
        libfranka_grasp_success=d["libfranka_grasp_success"],
        is_grasped_flag=d.get("is_grasped_flag"),
        delta_force_mag=d["delta_force_mag"],
        inferred_holding=d["inferred_holding"],
    )


def print_result(t: TrialResult):
    verdict = "HELD" if t.inferred_holding else "miss / crushed"
    print(f"  -> achieved={t.achieved_width*1000:5.1f} mm "
          f"(target {t.target_width*1000:5.1f} mm) "
          f"|dF|={t.delta_force_mag:5.2f} N  "
          f"libfranka_ok={t.libfranka_grasp_success}  "
          f"is_grasped={t.is_grasped_flag}  "
          f"=> {verdict}")


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--host", default="localhost")
    ap.add_argument("--port", type=int, default=8888)
    ap.add_argument("--csv", default="grasp_profile_results.csv",
                    help="Output CSV path (appended to).")
    ap.add_argument("--only", default=None,
                    help="Comma-separated subset of labels to test.")
    ap.add_argument("--sweep", action="store_true",
                    help="Also try target_width - 5mm and + 5mm.")
    ap.add_argument("--settle", type=float, default=1.5,
                    help="Hold time after grasp before measuring (s).")
    args = ap.parse_args()

    base_url = f"http://{args.host}:{args.port}"
    profile = DEFAULT_PROFILE
    if args.only:
        wanted = {s.strip() for s in args.only.split(",")}
        profile = [p for p in profile if p[0] in wanted]
        if not profile:
            print(f"--only filter matched nothing in {[p[0] for p in DEFAULT_PROFILE]}")
            sys.exit(1)

    csv_path = Path(args.csv)
    new_file = not csv_path.exists()
    csv_f = csv_path.open("a", newline="")
    writer = csv.writer(csv_f)
    if new_file:
        writer.writerow([
            "ts", "label", "target_width_m", "force_N",
            "achieved_width_m", "libfranka_grasp_success",
            "is_grasped_flag", "delta_force_mag_N", "inferred_holding",
        ])

    print(f"Grasp profiling against {base_url}")
    print(f"Results will append to: {csv_path.resolve()}")
    print()

    for label, tw, f in profile:
        widths = [tw]
        if args.sweep:
            widths = [max(0.003, tw - 0.005), tw, tw + 0.005]
        for w in widths:
            print(f"\n{label}  (target_width = {w*1000:.1f} mm, "
                  f"force = {f:.0f} N)")
            print(f"  Place the {label} between the jaws.")
            try:
                input("  Press ENTER when ready  (Ctrl-C to skip)... ")
            except KeyboardInterrupt:
                print("\n  skipped.")
                continue
            countdown(3)
            t = run_trial(base_url, label, w, f, settle_s=args.settle)
            if t is None:
                continue
            print_result(t)
            writer.writerow([
                time.strftime("%Y-%m-%d %H:%M:%S"),
                t.label,
                t.target_width,
                t.force,
                t.achieved_width,
                t.libfranka_grasp_success,
                t.is_grasped_flag,
                t.delta_force_mag,
                t.inferred_holding,
            ])
            csv_f.flush()

    csv_f.close()
    print(f"\nDone. CSV saved to {csv_path.resolve()}")
    print("To tune the planner defaults, look at rows where "
          "inferred_holding=False, those need a different "
          "target_width or force.")


if __name__ == "__main__":
    main()

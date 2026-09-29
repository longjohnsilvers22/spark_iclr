"""
Offline annotation-provider probe: score provider POINTING against
recorded frames whose detections are already known. API calls (and, optionally, local
Molmo inference) over existing data only; no robot, no simulator.

Input corpus: real-pipeline run folders (``src/output/real_runs/<ts>/``),
each holding ``result.json`` (SAM3 detections with pixel ``centroid_2d``)
plus the captured ``sideview_rgb.jpg`` / ``birdview_rgb.jpg``.  The
legacy result schema does not record WHICH camera a merged detection's
centroid came from, so the probe queries every available camera frame
and reports the error per camera plus the per-label best -- a provider
that points well will be near one camera's centroid and far from the
other's, and the per-camera medians expose which.

Usage (repo root):

    PYTHONPATH=src conda run -n spark_conda python \
        scripts/probe_annotation_providers.py --providers er2 --max-runs 10

    # Molmo needs the 4090 free (bf16 ~16 GB) - run only when it is:
    PYTHONPATH=src conda run -n spark_conda python \
        scripts/probe_annotation_providers.py --providers molmo --max-runs 5

Output: a per-(provider, camera) error table on stdout and a JSON dump
next to the runs (``--out``).
"""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

import numpy as np
from PIL import Image

from spark_real.perception.annotations import get_provider

REPO_ROOT = Path(__file__).resolve().parent.parent
CAMERAS = ("sideview", "birdview")


def load_runs(runs_dir: Path, max_runs: int) -> list[dict]:
    """Newest-first run folders that carry both detections and frames."""
    runs = []
    for d in sorted(runs_dir.iterdir(), reverse=True):
        rj = d / "result.json"
        if not rj.exists():
            continue
        try:
            res = json.loads(rj.read_text())
        except json.JSONDecodeError:
            continue
        dets = [
            {"label": det["label"], "centroid": det["centroid_2d"],
             "camera": det.get("camera")}
            for det in res.get("detections", [])
            if det.get("centroid_2d") and det.get("label")
        ]
        frames = {c: d / f"{c}_rgb.jpg" for c in CAMERAS
                  if (d / f"{c}_rgb.jpg").exists()}
        if not dets or not frames:
            continue
        runs.append({"dir": d, "instruction": res.get("instruction", ""),
                      "dets": dets, "frames": frames})
        if len(runs) >= max_runs:
            break
    return runs


def probe(provider_names: list[str], runs: list[dict]) -> list[dict]:
    records = []
    providers = {}
    for name in provider_names:
        p = get_provider(name)
        if p is None or not p.available():
            print(f"[probe] provider {name!r} UNAVAILABLE - skipped")
            continue
        providers[name] = p

    # One image load per (run, camera), shared across providers/labels.
    for run in runs:
        images = {}
        for cam, path in run["frames"].items():
            images[cam] = np.asarray(Image.open(path).convert("RGB"))
        for pname, prov in providers.items():
            for det in run["dets"]:
                label = det["label"]
                gt_u, gt_v = float(det["centroid"][0]), float(det["centroid"][1])
                per_cam = {}
                for cam, img in images.items():
                    h, w = img.shape[:2]
                    anns = prov.annotate(img, label, kind="point")
                    if not anns:
                        per_cam[cam] = None
                        continue
                    u, v = anns[0].to_pixels((w, h))[0]
                    err = math.hypot(u - gt_u, v - gt_v)
                    per_cam[cam] = {
                        "point_px": [round(u, 1), round(v, 1)],
                        "err_px": round(err, 1),
                        "err_norm": round(err / math.hypot(w, h), 4),
                    }
                rec = {
                    "run": run["dir"].name,
                    "instruction": run["instruction"],
                    "provider": pname,
                    "label": label,
                    "gt_centroid_px": [round(gt_u, 1), round(gt_v, 1)],
                    "gt_camera": det["camera"],  # None in the legacy schema
                    "per_camera": per_cam,
                }
                hits = [c["err_px"] for c in per_cam.values() if c]
                rec["best_err_px"] = min(hits) if hits else None
                records.append(rec)
                best = (f"{rec['best_err_px']:.0f}px"
                        if rec["best_err_px"] is not None else "MISS")
                print(f"[probe] {pname:6s} {run['dir'].name} {label!r}: "
                      f"best {best}  "
                      + " ".join(f"{c}={v['err_px']:.0f}px" if v else f"{c}=miss"
                                  for c, v in per_cam.items()))
    return records


def summarize(records: list[dict]) -> dict:
    out = {}
    for pname in sorted({r["provider"] for r in records}):
        recs = [r for r in records if r["provider"] == pname]
        summary = {"n_queries": len(recs)}
        for cam in CAMERAS:
            errs = [r["per_camera"][cam]["err_px"] for r in recs
                    if r["per_camera"].get(cam)]
            summary[cam] = {
                "answered": len(errs),
                "median_err_px": round(float(np.median(errs)), 1) if errs else None,
                "mean_err_px": round(float(np.mean(errs)), 1) if errs else None,
            }
        best = [r["best_err_px"] for r in recs if r["best_err_px"] is not None]
        summary["per_label_best"] = {
            "answered": len(best),
            "median_err_px": round(float(np.median(best)), 1) if best else None,
            "mean_err_px": round(float(np.mean(best)), 1) if best else None,
            "within_50px": sum(e <= 50 for e in best),
            "within_100px": sum(e <= 100 for e in best),
        }
        out[pname] = summary
    return out


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[1])
    ap.add_argument("--runs-dir",
                    default=str(REPO_ROOT / "src" / "output" / "real_runs"))
    ap.add_argument("--max-runs", type=int, default=10)
    ap.add_argument("--providers", default="er2",
                    help="comma-separated: er2,molmo")
    ap.add_argument("--out", default=str(REPO_ROOT / "output"
                                          / "annotation_probe.json"))
    args = ap.parse_args()

    runs = load_runs(Path(args.runs_dir), args.max_runs)
    print(f"[probe] {len(runs)} runs with frames + detections "
          f"from {args.runs_dir}")
    if not runs:
        print("[probe] nothing to score - need real_runs folders with "
              "result.json + sideview/birdview jpgs")
        return

    records = probe([p.strip() for p in args.providers.split(",") if p.strip()],
                     runs)
    summary = summarize(records)
    print("\n=== summary (pixel error vs known SAM3 centroid) ===")
    print(json.dumps(summary, indent=2))

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(
        {"summary": summary, "records": records}, indent=2))
    print(f"[probe] wrote {out_path}")


if __name__ == "__main__":
    main()

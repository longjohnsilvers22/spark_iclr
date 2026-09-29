"""
Offline probe: planner-INLINE RoboInter pointing vs OUT-OF-BAND ER2 pointing.

Measures the RoboInter-inline vs ER2 contact-point trade over recorded
frames, with no robot:

  * INLINE (planner): one ``generate_score`` call per (run, camera) with the
    RoboInter extension on -- the 0..1000 grid drawn on the frame, the schema
    section in the system prompt -- and the reply's ``__robointer``
    contact/placement pixels scored against the recorded SAM3 centroids.
  * OUT-OF-BAND (ER2): the pointing provider asked, on the SAME frame, for
    the SAME targets the planner annotated, scored against the same centroids.

Corpus and loading pattern are probe_annotation_providers.py's: real-pipeline
run folders (``src/output/real_runs/<ts>/``) holding ``result.json`` (SAM3
detections with pixel ``centroid_2d``) plus ``sideview_rgb.jpg`` /
``birdview_rgb.jpg``.  The legacy schema does not say WHICH camera a merged
centroid came from, so both frames are probed and the per-label best per
source is reported alongside per-camera numbers.

Ground truth is the SAM3 mask CENTROID -- a PROXY.  A good contact point
deliberately deviates from a centroid (handle vs blade), so this measures
POINTING CALIBRATION (can the model put a pixel on the object it names), not
grasp quality.  Both sources are scored against the same proxy, and the
detections handed to the planner carry NO pixel geometry (legacy runs have no
bbox), so neither source can echo the answer back.

Usage (repo root; ~2 Gemini planning calls per run, capped by --max-calls):

    SPARK_GEMINI_KEY_FILE=/path/to/.gemini_api_key \\
    PYTHONPATH=src conda run -n spark_conda python \\
        scripts/probe_robointer_vs_er2.py --max-runs 7

Output: a side-by-side per-source pixel-error table on stdout and a JSON dump
(``--out``, default output/robointer_vs_er2.json).  The API key is never
printed.
"""

from __future__ import annotations

import argparse
import importlib.util
import json
import math
from pathlib import Path

import numpy as np
from PIL import Image

from spark_real.bt_label_resolver import base_label
from spark_real.perception.annotations import get_provider
from spark_real.planning.robointer import extract_annotations
from spark_real.planning.robointer_annotate import draw_coordinate_grid
from spark_real.planning.robointer_gate import _node_label, _nodes_by_path
from spark_real.planning.robointer_prompt import build_robointer_context
from spark_real.planning.spark_planner import SPARKPlanner

REPO_ROOT = Path(__file__).resolve().parent.parent
CAMERAS = ("sideview", "birdview")

# The annotation probe's loader, reused verbatim (scripts/ is not a package).
_pap_spec = importlib.util.spec_from_file_location(
    "probe_annotation_providers", REPO_ROOT / "scripts" / "probe_annotation_providers.py"
)
_pap = importlib.util.module_from_spec(_pap_spec)
_pap_spec.loader.exec_module(_pap)
load_runs = _pap.load_runs


def dedupe_detections(dets: list[dict], min_conf: float = 0.3) -> dict[str, tuple]:
    """label -> (u, v) ground-truth centroid; highest confidence per label.

    Legacy result.json rows repeat labels across cameras/false positives; a
    duplicate label makes the 'known centroid' ambiguous, so keep the most
    confident instance and drop low-confidence noise entirely.
    """
    best: dict[str, tuple] = {}
    best_conf: dict[str, float] = {}
    for det in dets:
        label = det["label"]
        conf = float(det.get("confidence", 1.0) or 1.0)
        if conf < min_conf:
            continue
        if conf > best_conf.get(label, -1.0):
            best_conf[label] = conf
            best[label] = (float(det["centroid"][0]), float(det["centroid"][1]))
    return best


def match_centroid(target: str, centroids: dict[str, tuple]):
    """Instance-suffix tolerant lookup ('knife handle 1' -> 'knife handle')."""
    if target in centroids:
        return centroids[target]
    tb = base_label(target)
    for label, uv in centroids.items():
        if base_label(label) == tb:
            return uv
    return None


def planner_points(planner, image_rgb, instruction, labels, camera):
    """One RoboInter-on planning call -> [(target_label, kind, (x, y))].

    Normalized (x, y), read from the reply's __robointer blocks: contact
    points directly, placement proposals as their box center (all the
    consumer ever reads).  The frame carries the 0..1000 grid and the context
    states the image size -- the production prompt side, minus SAM3's
    observed boxes, which the legacy corpus does not have (and which would
    let the planner echo geometry instead of reading the image).
    """
    h, w = image_rgb.shape[:2]
    ctx = build_robointer_context([], image_size=(w, h), camera=camera)
    gridded = Image.fromarray(draw_coordinate_grid(image_rgb))
    score = planner.generate_score(
        instruction=instruction,
        annotated_image=gridded,
        keypoint_labels=labels,
        robointer_context=ctx,
    )
    nodes = _nodes_by_path(score)
    out = []
    for path, ann in extract_annotations(score, labels):
        target = ann.label or _node_label(nodes.get(path))
        if not target:
            continue
        if ann.contact_point is not None:
            out.append((target, "contact", (ann.contact_point.x, ann.contact_point.y)))
        if ann.placement_proposal is not None:
            c = ann.placement_proposal.center
            out.append((target, "placement", (c.x, c.y)))
    return out


def px_err(norm_xy, gt_uv, size) -> float:
    w, h = size
    return math.hypot(norm_xy[0] * w - gt_uv[0], norm_xy[1] * h - gt_uv[1])


def summarize(records: list[dict]) -> dict:
    out: dict = {}
    for source in ("planner_inline", "er2_oob"):
        recs = [r for r in records if r["source"] == source and r["err_px"] is not None]
        errs = [r["err_px"] for r in recs]
        by_kind = {}
        for kind in ("contact", "placement"):
            ke = [r["err_px"] for r in recs if r["kind"] == kind]
            by_kind[kind] = {
                "n": len(ke),
                "median_err_px": round(float(np.median(ke)), 1) if ke else None,
                "mean_err_px": round(float(np.mean(ke)), 1) if ke else None,
            }
        out[source] = {
            "n_scored": len(errs),
            "median_err_px": round(float(np.median(errs)), 1) if errs else None,
            "mean_err_px": round(float(np.mean(errs)), 1) if errs else None,
            "within_50px": sum(e <= 50 for e in errs),
            "within_100px": sum(e <= 100 for e in errs),
            "by_kind": by_kind,
        }
    return out


def best_across_cameras(records: list[dict]) -> dict:
    """Per-source stats over per-(run, target, kind) BEST camera.

    The legacy result.json does not say which camera a merged centroid came
    from, so a per-camera error against the wrong camera's frame is inflated
    for BOTH sources.  Same remedy as probe_annotation_providers: a source
    that points well is near ONE camera's centroid, so take the per-label
    minimum across cameras.  This is the headline number.
    """
    out: dict = {}
    for source in ("planner_inline", "er2_oob"):
        keyed: dict[tuple, float] = {}
        for r in records:
            if r["source"] != source or r["err_px"] is None:
                continue
            key = (r["run"], r["target"], r["kind"])
            keyed[key] = min(keyed.get(key, math.inf), r["err_px"])
        errs = sorted(keyed.values())
        out[source] = {
            "n_targets": len(errs),
            "median_err_px": round(float(np.median(errs)), 1) if errs else None,
            "mean_err_px": round(float(np.mean(errs)), 1) if errs else None,
            "within_25px": sum(e <= 25 for e in errs),
            "within_50px": sum(e <= 50 for e in errs),
        }
    return out


def paired_summary(records: list[dict]) -> dict:
    """Per (run, camera, target, kind) pairs where BOTH sources answered."""
    keyed: dict[tuple, dict] = {}
    for r in records:
        if r["err_px"] is None:
            continue
        key = (r["run"], r["camera"], r["target"], r["kind"])
        keyed.setdefault(key, {})[r["source"]] = r["err_px"]
    pairs = [v for v in keyed.values() if len(v) == 2]
    if not pairs:
        return {"n_pairs": 0}
    deltas = [p["planner_inline"] - p["er2_oob"] for p in pairs]
    return {
        "n_pairs": len(pairs),
        "er2_better": sum(d > 0 for d in deltas),
        "planner_better": sum(d < 0 for d in deltas),
        "median_delta_px_planner_minus_er2": round(float(np.median(deltas)), 1),
    }


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[1])
    ap.add_argument("--runs-dir", default=str(REPO_ROOT / "src" / "output" / "real_runs"))
    ap.add_argument("--max-runs", type=int, default=7)
    ap.add_argument("--max-calls", type=int, default=15, help="hard cap on Gemini PLANNING calls")
    ap.add_argument("--model", default=None, help="planner model override")
    ap.add_argument("--out", default=str(REPO_ROOT / "output" / "robointer_vs_er2.json"))
    args = ap.parse_args()

    runs = load_runs(Path(args.runs_dir), args.max_runs)
    print(f"[probe] {len(runs)} runs with frames + detections from {args.runs_dir}")
    if not runs:
        return

    planner = SPARKPlanner(llm_backend="gemini", model=args.model, temperature=0.0)
    er2 = get_provider("er2")
    if er2 is None or not er2.available():
        print("[probe] ER2 provider UNAVAILABLE -- inline-only run")
        er2 = None

    records: list[dict] = []
    calls = 0
    for run in runs:
        gt = dedupe_detections(run["dets"])
        if not gt:
            continue
        labels = sorted(gt)
        for cam, path in sorted(run["frames"].items()):
            if calls >= args.max_calls:
                break
            img = np.asarray(Image.open(path).convert("RGB"))
            h, w = img.shape[:2]

            # -- inline route: one planning call ------------------------------
            calls += 1
            try:
                points = planner_points(planner, img, run["instruction"], labels, cam)
            except Exception as exc:  # noqa: BLE001 - a failed call is a data point
                print(
                    f"[probe] planner call failed on {run['dir'].name}/{cam}: "
                    f"{type(exc).__name__}"
                )
                points = []
            targets = []
            for target, kind, xy in points:
                gt_uv = match_centroid(target, gt)
                err = px_err(xy, gt_uv, (w, h)) if gt_uv else None
                records.append(
                    {
                        "run": run["dir"].name,
                        "camera": cam,
                        "source": "planner_inline",
                        "target": target,
                        "kind": kind,
                        "point_px": [round(xy[0] * w, 1), round(xy[1] * h, 1)],
                        "gt_px": list(gt_uv) if gt_uv else None,
                        "err_px": round(err, 1) if err is not None else None,
                    }
                )
                if gt_uv:
                    targets.append((target, kind, gt_uv))
                    print(
                        f"[probe] inline {run['dir'].name}/{cam} {kind:9s} "
                        f"{target!r}: {err:.0f}px"
                    )

            # -- out-of-band route: the SAME targets, same frame --------------
            if er2 is None:
                continue
            for target, kind, gt_uv in targets:
                query = target if kind == "contact" else f"empty spot inside the {target}"
                try:
                    anns = er2.annotate(img, query, kind="point")
                except Exception:  # noqa: BLE001 - provider is fail-open anyway
                    anns = []
                if anns:
                    x, y = anns[0].point
                    err = px_err((x, y), gt_uv, (w, h))
                    rec_pt = [round(x * w, 1), round(y * h, 1)]
                else:
                    err, rec_pt = None, None
                records.append(
                    {
                        "run": run["dir"].name,
                        "camera": cam,
                        "source": "er2_oob",
                        "target": target,
                        "kind": kind,
                        "point_px": rec_pt,
                        "gt_px": list(gt_uv),
                        "err_px": round(err, 1) if err is not None else None,
                    }
                )
                shown = f"{err:.0f}px" if err is not None else "MISS"
                print(f"[probe] er2    {run['dir'].name}/{cam} {kind:9s} " f"{target!r}: {shown}")

    summary = summarize(records)
    best = best_across_cameras(records)
    pairs = paired_summary(records)
    print(f"\n[probe] {calls} planning calls made")
    print("=== per-record, per-camera (GT camera unknown; inflated for both) ===")
    print(json.dumps(summary, indent=2))
    print("=== HEADLINE: per-target best across cameras ===")
    print(json.dumps(best, indent=2))
    print("=== paired same-frame (both sources answered the same target) ===")
    print(json.dumps(pairs, indent=2))

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(
        json.dumps(
            {
                "summary": summary,
                "best_across_cameras": best,
                "paired": pairs,
                "planning_calls": calls,
                "records": records,
            },
            indent=2,
        )
    )
    print(f"[probe] wrote {out_path}")


if __name__ == "__main__":
    main()

"""Detector-fusion gate on REAL data. No robot, no GPU, no SAM3 process.

What is real here and what is not, stated up front:

  REAL  every RGB frame, from the recorded teleop corpus ($SPARK_HUMAN_EPISODES).
  REAL  every SAM3 mask and SAM3 score, unpacked from an offline SAM3 harvest
        (output/pvh_mask_cache, written by tests/verify_replay_harvest.py).
  REAL  the second detector: a COCO-trained torchvision model, loaded and run
        here, on CPU, on that same frame. Nothing replayed, nothing stubbed.
  REAL  the gate, the fusion arithmetic and the grasp-strategy resolver: the
        same modules the executor calls, on the dict shape the executor reads.
  GIVEN the pair (confidence 0.24, aspect_ratio 3.14) is the MEASURED live
        failure the operator reported. The harvest scored the same object
        higher, so the first case pins the live numbers onto the real mask
        rather than pretending 0.24 came from the cache. The second case uses
        the harvest's own score and is the more damning of the two.

Run::

    PYTHONPATH=src SPARK_HUMAN_EPISODES=<corpus> conda run -n spark_conda \\
        python -m spark_real.tests.fusion_plushie_demo --proposer torchvision
"""

from __future__ import annotations

import argparse
import json
import os
import time
from collections import Counter
from pathlib import Path
from typing import Optional

import cv2
import numpy as np

from spark_real.control import grasp_strategy as gs
from spark_real.perception import box_proposals as bp
from spark_real.perception.detection_fusion import DetectionGate, FusionConfig, mask_bbox
from spark_real.perception.mask_quality import measure_mask
from spark_real.tests.human_corpus import CORPUS_ENV, corpus_root
from spark_real.tests.test_grasp_strategy import AR_GATE, _Exec

MASK_CACHE_ENV = "SPARK_MASK_CACHE"
DEFAULT_CACHE = Path(__file__).resolve().parents[3] / "output" / "pvh_mask_cache"

# The live failure, measured on the rig: SAM3 said 0.24 and perception's
# world-XY PCA said 3.14, which cleared the 1.8 AR gate then in force and
# earned a -68.9 deg wrist rotation off a meaningless OBB.
LIVE_PLUSHIE_CONF = 0.24
LIVE_PLUSHIE_AR = 3.14


def _unpack(packed, key, shape):
    bits = np.unpackbits(packed[key])[: shape[0] * shape[1]]
    return bits.reshape(shape).astype(np.uint8)


class Harvest:
    """Read-only accessor over an offline SAM3 mask harvest."""

    def __init__(self, cache: Path):
        self.index = json.loads((cache / "index.json").read_text())
        self._packed = np.load(cache / "masks.npz")

    def records(self, prompt: str, camera: str, phase: str):
        for rec in self.index:
            if prompt not in rec.get("obj_prompt", "") or rec.get("camera") != camera:
                continue
            if rec.get("phase") != phase or not rec.get("obj_key"):
                continue
            yield rec, _unpack(self._packed, rec["obj_key"], tuple(rec["shape"]))


def load_rgb(root: Path, rec: dict, camera: str) -> Optional[np.ndarray]:
    path = root / rec["task"] / rec["episode"] / "images" / camera / f"frame_{rec['frame']:04d}.jpg"
    img = cv2.imread(str(path))
    return (None, path) if img is None else (cv2.cvtColor(img, cv2.COLOR_BGR2RGB), path)


def as_detection(label, conf, mask, reported_ar) -> dict:
    """The dict shape the executor actually reads (pipeline detection_map)."""
    q = measure_mask(mask)
    return {
        "label": label,
        "confidence": float(conf),
        "mask": mask,
        "bbox": mask_bbox(mask),
        "aspect_ratio": float(reported_ar),
        # perception's OBB angle; the mask's own major axis is the honest
        # stand-in for it and is the direction a yaw would have aimed along.
        "orientation_angle": float(np.deg2rad(q.major_axis_deg)),
        "position_3d": [-0.9, 0.0, -0.24],
    }


def strategy_verdict(det: dict) -> str:
    """What the real resolver would command for this detection."""
    ex = _Exec()
    _, reason = gs.resolve_strategy({}, det, ex)
    orient, used = gs.resolve_grasp_orientation({}, det, ex)
    yaw = float(np.rad2deg(gs.measured_yaw_offset(orient, ex.GRASP_ORIENTATION)))
    return f"strategy={used:8s} wrist_yaw={yaw:+6.1f} deg   ({reason})"


class Scaled(bp.BoxProposer):
    """Runs a real proposer on the full-res frame, then reduces its boxes into
    the harvest's downsampled mask space so IoU is computed in one frame."""

    name = "scaled"

    def __init__(self, inner: bp.BoxProposer, scale: float):
        self.inner, self.scale = inner, scale
        self.last = bp.ProposalSet(available=False)
        self.seconds = 0.0

    def vocabulary(self):
        return self.inner.vocabulary()

    def propose(self, rgb, labels=()):
        t0 = time.time()
        pset = self.inner.propose(rgb, labels)
        self.seconds = time.time() - t0
        self.last = pset
        out = [
            bp.BoxProposal(p.label, p.confidence, tuple(v / self.scale for v in p.box), p.source)
            for p in pset.proposals
        ]
        return bp.ProposalSet(out, pset.vocabulary, pset.available, self.name, pset.note)


def report_availability(variant: str, device: str) -> None:
    print("== proposer availability in this env (nothing installed for this run)")
    for backend in ("rfdetr", "torchvision"):
        cfg = {"enabled": True, "backend": backend, "variant": variant, "device": device}
        t0 = time.time()
        try:
            p = bp.build_proposer(cfg)
            print(f"  {backend:12s} OK        {type(p).__name__} ({time.time() - t0:.1f}s)")
        except RuntimeError as exc:
            print(f"  {backend:12s} UNUSABLE  raised: {str(exc)[:120]}")


def run_case(title, det, rgb, proposer, cfg, note="") -> None:
    print(f"\n-- {title}")
    if note:
        print(f"   {note}")
    print(f"   BEFORE gate: {strategy_verdict(det)}")
    gate = DetectionGate(cfg, proposer=proposer)
    (res,) = gate.apply([det], rgb=rgb if proposer is not None else None)
    if proposer is not None:
        boxes = ", ".join(
            f"{p.label} {p.confidence:.2f}"
            for p in sorted(proposer.last.proposals, key=lambda x: -x.confidence)[:6]
        )
        print(
            f"   detector   : {len(proposer.last.proposals)} boxes in "
            f"{proposer.seconds:.2f}s -> {boxes}"
        )
    print(f"   fusion     : {res.describe()}")
    print(
        f"   stamped    : fused_confidence={det['fused_confidence']:.3f} "
        f"axis_trust={det['axis_trust']:.3f} low_quality={det['low_quality']} "
        f"confidence={det['confidence']:.3f} (SAM3's own, untouched)"
    )
    if res.low_quality:
        # What ASPIRE would ask SAM3 next. Printed rather than executed: no
        # SAM3 process runs here, and inventing its replies would be a fake.
        plan = gate._attempts(det, res, {})[: cfg.reprompt_max_attempts]
        shown = ", ".join(f"{p!r}{' +partner box' if b else ''}" for p, b in plan)
        print(
            f"   aspire plan: {shown or '(nothing to try)'}  "
            f"[budget {cfg.reprompt_max_attempts}, monotone, min_iou "
            f"{cfg.reprompt_min_iou}]"
        )
    print(f"   AFTER gate : {strategy_verdict(det)}")


def scan(harvest, root, proposer, cfg, camera, phase, limit) -> None:
    """How often does the second detector corroborate a real plushie mask?"""
    print(f"\n== corroboration scan: {limit} real plushie frames, detector run on each")
    agree, header = Counter(), True
    for rec, mask in list(harvest.records("plush", camera, phase))[:limit]:
        rgb, path = load_rgb(root, rec, camera)
        if rgb is None:
            continue
        det = as_detection("plushie", rec["obj_score"], mask, LIVE_PLUSHIE_AR)
        (res,) = DetectionGate(cfg, proposer=proposer).apply([det], rgb=rgb)
        agree[res.agreement] += 1
        if header:
            print(
                "   episode      sam3  fused  agreement    partner            mask_ar swing  yaw?"
            )
            header = False
        partner = (
            f"{res.partner.label} {res.partner.confidence:.2f} iou={res.iou:.2f}"
            if res.partner
            else "-"
        )
        print(
            f"   {rec['episode']:<12s} {res.sam3_confidence:.2f}  {res.fused_confidence:.2f}   "
            f"{res.agreement:<11s} {partner:<18s} {res.quality.mask_aspect_ratio:5.2f} "
            f"{res.quality.angle_swing_deg:5.1f}  {res.angle_trustworthy}"
        )
    print(f"   totals: {dict(agree)}")

    # The plushie is present in every one of these frames, so a box is a hit
    # and silence is a miss: this IS the detector's recall, and -log(1-recall)
    # is the only defensible value for cfg.miss_penalty.
    n = sum(agree.values())
    hits = agree.get("agree_class", 0) + agree.get("agree_box", 0)
    if n and hits < n:
        recall = hits / n
        print(
            f"   corroboration recall {hits}/{n} = {recall:.2f} -> calibrated "
            f"miss_penalty = -log(1-recall) = {-np.log(1.0 - recall):.2f} logits "
            f"(config carries {cfg.miss_penalty:.2f})"
        )


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--proposer", default="torchvision", choices=("none", "torchvision", "rfdetr"))
    ap.add_argument("--variant", default="fasterrcnn")
    ap.add_argument("--device", default="cpu")
    ap.add_argument("--camera", default="camera_0")
    ap.add_argument("--phase", default="negative", help="negative = before the place")
    ap.add_argument("--scan", type=int, default=8)
    ap.add_argument("--corpus", default=None)
    ap.add_argument("--mask-cache", default=os.environ.get(MASK_CACHE_ENV) or str(DEFAULT_CACHE))
    args = ap.parse_args()

    root = corpus_root(args.corpus)
    if root is None or not root.is_dir():
        print(f"no corpus: set ${CORPUS_ENV} or --corpus")
        return 2
    cache = Path(args.mask_cache)
    if not (cache / "index.json").exists():
        print(f"no SAM3 harvest at {cache}: set ${MASK_CACHE_ENV} or --mask-cache")
        return 2

    report_availability(args.variant, args.device)
    harvest = Harvest(cache)

    rec, mask = next(iter(harvest.records("plush", args.camera, args.phase)), (None, None))
    if rec is None:
        print("no plushie record with a mask in the harvest")
        return 2
    rgb, path = load_rgb(root, rec, args.camera)
    if rgb is None:
        print(f"cannot read {path}")
        return 2

    proposer = None
    if args.proposer != "none":
        inner = bp.build_proposer(
            {
                "enabled": True,
                "backend": args.proposer,
                "variant": args.variant,
                "device": args.device,
            }
        )
        proposer = Scaled(inner, rgb.shape[1] / mask.shape[1])
        print(
            f"\n== second detector: {type(inner).__name__} {args.variant} on {args.device}, "
            f"{len(inner.vocabulary())} COCO classes"
        )

    cfg = FusionConfig.from_dict({"enabled": True})
    print(
        f"\n== gate ON, all defaults: assoc_iou={cfg.assoc_iou} w_class={cfg.w_class} "
        f"w_box={cfg.w_box} miss_penalty={cfg.miss_penalty} "
        f"low_quality_conf={cfg.low_quality_conf}"
    )
    print(
        f"   grasp gates unchanged: AR gate {AR_GATE}, yaw_min_conf "
        f"{gs.DEFAULT_YAW_MIN_CONF}, yaw_min_obb_conf {gs.DEFAULT_YAW_MIN_OBB_CONF}"
    )
    print(
        f"\n== real inputs\n   frame {path}\n         {rgb.shape} rgb, mask {mask.shape} "
        f"({int(mask.sum())} px), harvest SAM3 score {rec['obj_score']:.3f}"
    )

    run_case(
        f"plushie, THE LIVE FAILURE: sam3 conf {LIVE_PLUSHIE_CONF:.2f}, "
        f"reported aspect_ratio {LIVE_PLUSHIE_AR:.2f}",
        as_detection("plushie", LIVE_PLUSHIE_CONF, mask, LIVE_PLUSHIE_AR),
        rgb,
        proposer,
        cfg,
    )
    run_case(
        f"plushie, harvest score {rec['obj_score']:.2f} on the SAME real mask "
        f"(ar {LIVE_PLUSHIE_AR:.2f})",
        as_detection("plushie", rec["obj_score"], mask, LIVE_PLUSHIE_AR),
        rgb,
        proposer,
        cfg,
        note="a high SAM3 score does NOT make a round mask's axis real",
    )

    # Control on its own real frame: a genuinely elongated real mask must
    # still earn its yaw, or the gate is just a blanket veto.
    krec, kmask = next(iter(harvest.records("knife", args.camera, args.phase)), (None, None))
    if krec is not None:
        krgb, kpath = load_rgb(root, krec, args.camera)
        kq = measure_mask(kmask)
        if krgb is not None:
            run_case(
                f"knife CONTROL on its own frame, harvest score {krec['obj_score']:.2f}",
                as_detection("knife", krec["obj_score"], kmask, kq.mask_aspect_ratio),
                krgb,
                proposer,
                cfg,
                note=f"{kpath.name} of {krec['task']!r}; reported ar = mask ar "
                f"{kq.mask_aspect_ratio:.2f} since the harvest stores no world OBB",
            )

    if proposer is not None and args.scan:
        scan(harvest, root, proposer, cfg, args.camera, args.phase, args.scan)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

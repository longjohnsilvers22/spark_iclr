"""
Second-opinion box proposers.

SAM3 is open-vocabulary and good at masks, but its score is a poor gate (a
plushie at 0.24 confidence still drove a 69 deg wrist rotation). A second,
independently-trained detector that says "yes, there is an object there" is
corroborating evidence that SAM3's own score cannot provide.

This module defines the interface only. It deliberately knows nothing about
SAM3, fusion, or grasping -- see detection_fusion.py for that.

BACKENDS:
  rfdetr        NOT INSTALLED in spark_conda; its checkpoints are a separate
                runtime download. RFDETRProposer raises at construction with
                the exact install command. It is NOT stubbed and NOT faked.
  torchvision   RUNS, no install: torchvision is already in spark_conda and
                ships COCO-trained detectors (1.86 s/frame on CPU for a
                640x480 frame). device defaults to "cpu" ON PURPOSE so a
                second opinion can never contend with the live server for
                VRAM.
  static        real, offline, for tests and for replaying a recorded detector.
  null          real, always abstains. This is the default.

A configured-but-unconstructable backend RAISES (see build_proposer). Silently
falling back to null would mean the operator turns fusion on, sees no error,
and gets none of the protection they asked for.

RF-DETR is also CLOSED-SET over 80 COCO classes with no text-prompt path.
Of the corpus vocabulary, bowl/knife/spoon/fork/cup/teddy bear are COCO
classes; tray/bin/block/pen/cube/container are NOT. That asymmetry is why
`vocabulary()` exists and why silence from a proposer on an out-of-vocabulary
label must be read as ABSTAIN, never as a veto.
"""

from __future__ import annotations

import logging
import os
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Sequence

import numpy as np

logger = logging.getLogger(__name__)

# COCO class -> the corpus/prompt words that should count as the same thing.
# Only used to decide "did the two detectors name the same object", never to
# rename anything downstream.
DEFAULT_SYNONYMS: Dict[str, tuple] = {
    "teddy bear": ("plushie", "plush", "stuffed animal", "toy", "teddy"),
    "bowl": ("bowl", "dish"),
    "knife": ("knife", "blade"),
    "spoon": ("spoon",),
    "fork": ("fork",),
    "cup": ("cup", "mug"),
    "bottle": ("bottle",),
    "scissors": ("scissors",),
    "book": ("book",),
    "cell phone": ("phone", "cellphone"),
}


# Sub-part phrasings. SAM3 answers "knife handle" much better than it answers
# "the graspable end of a knife", and a handle mask has a real major axis where
# a whole-tool mask often does not.
SUBPART_SUFFIXES: tuple = ("handle", "grip")


def default_alt_prompts(label: str, elongated: bool = False, limit: int = 4) -> List[str]:
    """Reprompt vocabulary for ASPIRE when the caller supplies none.

    Needs no second model: synonyms come from DEFAULT_SYNONYMS, sub-parts from
    SUBPART_SUFFIXES. `elongated` (the detection claimed a major axis) puts the
    sub-part phrasings first, because that is the failure where a handle mask
    is the actual repair.
    """
    base = _strip_instance(label)
    if not base:
        return []
    words: List[str] = []
    for cls, syn in DEFAULT_SYNONYMS.items():
        if base == cls or base in syn or any(w in base for w in syn):
            words.extend([cls, *syn])
    subparts = [f"{base} {s}" for s in SUBPART_SUFFIXES]
    ordered = subparts + words if elongated else words + subparts

    out: List[str] = []
    for w in ordered:
        w = w.strip().lower()
        if w and w != base and w not in out:
            out.append(w)
    return out[:limit]


@dataclass
class BoxProposal:
    """One box from a secondary detector, in pixels of the source image."""

    label: str
    confidence: float
    box: tuple  # (x1, y1, x2, y2) pixels
    source: str = "unknown"

    def area(self) -> float:
        x1, y1, x2, y2 = self.box
        return max(0.0, x2 - x1) * max(0.0, y2 - y1)


@dataclass
class ProposalSet:
    """Everything one proposer pass returns.

    `available` False means the backend could not run. Callers MUST treat that
    as abstain -- an unavailable detector is not a detector that saw nothing.
    """

    proposals: List[BoxProposal] = field(default_factory=list)
    vocabulary: frozenset = frozenset()
    available: bool = True
    source: str = "unknown"
    note: str = ""

    def covers(self, label: str, synonyms: Optional[Dict[str, tuple]] = None) -> bool:
        """True when `label` is inside this detector's closed vocabulary."""
        if not self.available or not self.vocabulary:
            return False
        return _canonical_class(label, self.vocabulary, synonyms) is not None


class BoxProposer:
    """Interface. Implementations must not raise from `propose`."""

    name = "base"

    def vocabulary(self) -> frozenset:
        raise NotImplementedError

    def propose(self, rgb: np.ndarray, labels: Sequence[str] = ()) -> ProposalSet:
        raise NotImplementedError


class NullProposer(BoxProposer):
    """Always abstains. The default, so nothing changes until opt-in."""

    name = "null"

    def vocabulary(self) -> frozenset:
        return frozenset()

    def propose(self, rgb, labels=()) -> ProposalSet:
        return ProposalSet(available=False, source=self.name, note="disabled")


class StaticProposer(BoxProposer):
    """Replays a fixed list. Real code path, used by tests and by replaying a
    detector that was run offline."""

    name = "static"

    def __init__(self, proposals: Sequence[BoxProposal], vocabulary: Sequence[str] = ()):
        self._proposals = list(proposals)
        self._vocab = frozenset(v.lower() for v in vocabulary) or frozenset(
            p.label.lower() for p in self._proposals
        )

    def vocabulary(self) -> frozenset:
        return self._vocab

    def propose(self, rgb, labels=()) -> ProposalSet:
        return ProposalSet(
            proposals=list(self._proposals),
            vocabulary=self._vocab,
            available=True,
            source=self.name,
        )


class RFDETRProposer(BoxProposer):
    """RF-DETR backend. Fails loudly when the package is missing.

    Kept as a real adapter rather than a stub so that installing the package
    is the only thing standing between here and a working second opinion.
    """

    name = "rfdetr"
    INSTALL_HINT = (
        "rfdetr is not installed in this env. "
        "conda run -n spark_conda pip install rfdetr  "
        "(~16 MB of wheels, does not touch torch/torchvision/numpy; the "
        "checkpoint is a separate 135 MB-1.5 GB runtime download). "
        "NOTE: RF-DETR is closed-set over 80 COCO classes and cannot replace "
        "SAM3's open-vocabulary role."
    )

    def __init__(self, variant: str = "medium", threshold: float = 0.35, device: str = "cuda"):
        try:
            import rfdetr  # noqa: F401
        except ImportError as exc:
            raise RuntimeError(f"RFDETRProposer unavailable: {self.INSTALL_HINT}") from exc

        from rfdetr import RFDETRMedium, RFDETRNano  # noqa: F401  (import-time check)

        self.threshold = float(threshold)
        self.variant = variant
        builder = {"nano": RFDETRNano, "medium": RFDETRMedium}.get(variant, RFDETRMedium)
        self._model = builder(device=device)
        self._vocab = frozenset(c.lower() for c in self._coco_classes())

    @staticmethod
    def _coco_classes() -> Sequence[str]:
        from rfdetr.assets.coco_classes import COCO_CLASSES

        if isinstance(COCO_CLASSES, dict):
            return [str(v) for v in COCO_CLASSES.values()]
        return [str(v) for v in COCO_CLASSES]

    def vocabulary(self) -> frozenset:
        return self._vocab

    def propose(self, rgb, labels=()) -> ProposalSet:
        try:
            from PIL import Image

            det = self._model.predict(Image.fromarray(rgb), threshold=self.threshold)
            out = []
            for xyxy, conf, cid in zip(det.xyxy, det.confidence, det.class_id):
                out.append(
                    BoxProposal(
                        label=self._class_name(int(cid)),
                        confidence=float(conf),
                        box=tuple(float(v) for v in xyxy),
                        source=self.name,
                    )
                )
            return ProposalSet(out, self._vocab, True, self.name)
        except Exception as exc:  # noqa: BLE001 - a detector must never stop perception
            logger.warning("[proposer] rfdetr pass failed (%s); abstaining", exc)
            return ProposalSet(available=False, source=self.name, note=str(exc))

    def _class_name(self, cid: int) -> str:
        """Map a model class id to its COCO name.

        COCO_CLASSES is keyed by CATEGORY ID (1..90, 80 entries, with gaps),
        so it must be looked up by key, not by position: indexing .values()
        positionally shifts every label and drops the ids past the 80th (id
        88, "teddy bear", would come back as the string "88").
        """
        from rfdetr.assets.coco_classes import COCO_CLASSES

        if isinstance(COCO_CLASSES, dict):
            name = COCO_CLASSES.get(cid)
            if name is None and 0 <= cid < len(COCO_CLASSES):
                # Some builds emit contiguous indices instead of category ids.
                name = list(COCO_CLASSES.values())[cid]
            return str(name).lower() if name is not None else str(cid)
        classes = list(COCO_CLASSES)
        return str(classes[cid]).lower() if 0 <= cid < len(classes) else str(cid)


class TorchvisionCOCOProposer(BoxProposer):
    """COCO detector from torchvision. The one second opinion that RUNS HERE.

    Same closed-set 80-class COCO signal RF-DETR would give, with no install:
    torchvision is already a hard dependency of this env. CPU by default --
    the live server owns the GPU, and 1.9 s on CPU is cheap next to one 29 s
    pick-and-place.

    Fails loudly: a missing weight cache with no network raises rather than
    quietly proposing nothing.
    """

    name = "torchvision"
    VARIANTS = ("fasterrcnn", "fasterrcnn_mobilenet", "retinanet")
    INSTALL_HINT = (
        "torchvision.models.detection is unavailable. The COCO weights are a "
        "one-time ~160 MB download into ~/.cache/torch/hub/checkpoints; with no "
        "network, pre-seed that cache on a connected machine."
    )

    def __init__(self, variant: str = "fasterrcnn", threshold: float = 0.35, device: str = "cpu"):
        try:
            import torch
            from torchvision.models import detection as tvdet
        except ImportError as exc:
            raise RuntimeError(f"TorchvisionCOCOProposer unavailable: {self.INSTALL_HINT}") from exc

        builders = {
            "fasterrcnn": (tvdet.fasterrcnn_resnet50_fpn, tvdet.FasterRCNN_ResNet50_FPN_Weights),
            "fasterrcnn_mobilenet": (
                tvdet.fasterrcnn_mobilenet_v3_large_fpn,
                tvdet.FasterRCNN_MobileNet_V3_Large_FPN_Weights,
            ),
            "retinanet": (tvdet.retinanet_resnet50_fpn, tvdet.RetinaNet_ResNet50_FPN_Weights),
        }
        if variant not in builders:
            raise RuntimeError(
                f"unknown torchvision variant {variant!r}; pick one of {self.VARIANTS}"
            )
        builder, weights_enum = builders[variant]
        weights = weights_enum.COCO_V1
        try:
            model = builder(weights=weights)
        except Exception as exc:  # noqa: BLE001 - no network + no cache lands here
            raise RuntimeError(
                f"TorchvisionCOCOProposer could not load {variant} COCO weights "
                f"({exc}). {self.INSTALL_HINT}"
            ) from exc

        self._torch = torch
        self.variant = variant
        self.threshold = float(threshold)
        self.device = str(device)
        self._model = model.eval().to(self.device)
        # "N/A" placeholders exist in the COCO category list; drop them so they
        # cannot be matched as a class name.
        self._categories = [str(c) for c in weights.meta["categories"]]
        self._vocab = frozenset(c.lower() for c in self._categories if c and c != "N/A")
        logger.info(
            "[proposer] torchvision %s on %s, %d COCO classes, threshold %.2f",
            variant,
            self.device,
            len(self._vocab),
            self.threshold,
        )

    def vocabulary(self) -> frozenset:
        return self._vocab

    def propose(self, rgb, labels=()) -> ProposalSet:
        try:
            arr = np.ascontiguousarray(np.asarray(rgb))
            if arr.ndim != 3 or arr.shape[2] < 3:
                return ProposalSet(available=False, source=self.name, note="rgb is not HxWx3")
            x = self._torch.from_numpy(arr[:, :, :3].copy()).permute(2, 0, 1).float()
            if arr.dtype == np.uint8:
                x = x / 255.0
            x = x.to(self.device)
            with self._torch.no_grad():
                out = self._model([x])[0]
            props = []
            for box, score, cid in zip(out["boxes"], out["scores"], out["labels"]):
                s = float(score)
                if s < self.threshold:
                    continue
                props.append(
                    BoxProposal(
                        label=self._class_name(int(cid)),
                        confidence=s,
                        box=tuple(float(v) for v in box.tolist()),
                        source=f"{self.name}:{self.variant}",
                    )
                )
            return ProposalSet(props, self._vocab, True, self.name)
        except Exception as exc:  # noqa: BLE001 - a detector must never stop perception
            logger.warning("[proposer] torchvision pass failed (%s); abstaining", exc)
            return ProposalSet(available=False, source=self.name, note=str(exc))

    def _class_name(self, cid: int) -> str:
        if 0 <= cid < len(self._categories):
            return self._categories[cid].lower()
        return str(cid)


_BACKENDS = {
    "null": NullProposer,
    "rfdetr": RFDETRProposer,
    "torchvision": TorchvisionCOCOProposer,
}

_PROPOSER_KWARGS = ("variant", "threshold", "device")


def build_proposer(cfg: Optional[dict]) -> BoxProposer:
    """Config-gated construction. Absent/off config -> NullProposer.

    A backend that was explicitly ASKED FOR and cannot be built RAISES. The
    operator turning fusion on and getting silence instead of protection is the
    failure mode that matters here; `strict: false` degrades to null instead.
    """
    cfg = dict(cfg or {})
    backend = str(cfg.get("backend", "null")).lower()
    env = os.environ.get("SPARK_BOX_PROPOSER")
    if env:
        backend = env.lower()

    if backend in ("null", "off", "none", ""):
        return NullProposer()
    if not cfg.get("enabled", False) and not env:
        # Backend named but never switched on. Not an error, not a warning.
        return NullProposer()

    strict = bool(cfg.get("strict", True))
    cls = _BACKENDS.get(backend)
    if cls is None:
        msg = f"unknown box proposer backend {backend!r}; known: {sorted(_BACKENDS)}"
        if strict:
            raise RuntimeError(msg)
        logger.warning("[proposer] %s; using null", msg)
        return NullProposer()
    try:
        return cls(**{k: v for k, v in cfg.items() if k in _PROPOSER_KWARGS})
    except Exception as exc:  # noqa: BLE001
        if strict:
            raise RuntimeError(f"box proposer {backend!r} was requested but is unusable: {exc}")
        logger.warning("[proposer] %s unavailable (%s); using null", backend, exc)
        return NullProposer()


def _canonical_class(label: str, vocabulary, synonyms=None) -> Optional[str]:
    """Map a SAM3 prompt onto a vocabulary class, or None if out of vocab."""
    syn = DEFAULT_SYNONYMS if synonyms is None else synonyms
    lab = _strip_instance(label)
    if lab in vocabulary:
        return lab
    for cls, words in syn.items():
        if cls not in vocabulary:
            continue
        if lab == cls or any(w in lab for w in words):
            return cls
    # last resort: a vocabulary word appearing inside a compound prompt
    for cls in vocabulary:
        if cls in lab:
            return cls
    return None


def _strip_instance(label: str) -> str:
    """'blue block 2' -> 'blue block'."""
    parts = str(label).strip().lower().split()
    if parts and parts[-1].isdigit():
        parts = parts[:-1]
    return " ".join(parts)

"""How MUCH perception a verification is allowed to ask for.

Unscoped, a verification asks for everything: a fresh capture off every
camera, a SAM3 pass for every label the predicate mentions on every one of
them, and the fusion gate's ASPIRE reprompts on top; one ``verify_placed``
measured 7 SAM3 inferences and 54 seconds against an 8-second primitive budget.

This module is the budget. It answers four questions, cheapest first, and every
answer is a config switch with the unscoped behaviour still reachable:

  1. Can telemetry already answer this?  A gripper that still reads HOLDING
     after a release is positive evidence the place failed. Zero inferences.
  2. Is there a recent enough capture?   OFF by default, and it must stay off
     unless the caller knows the scene is static; see the warning on
     ``reuse_capture_s``. When on, a second verification inside the window
     costs zero captures and zero inferences.
  3. Which labels must be re-detected?   The object moved; the container the
     tree never touched did not, and its plan-time pose is already the
     reference frame the verifier prefers (success_verifier._bind_reference
     documents why re-detecting it is actively harmful). Drop its prompt.
  4. Which cameras can see them?         A camera the labels do not project
     into, or that has no usable depth, can only abstain.

EVIDENCE GIVEN UP, stated plainly because a verifier that quietly sees less is
worse than a slow one:

  * ``anchor_references``: the container is no longer PROVEN present in each
    camera at verify time. Plan-time geometry becomes a reference frame AND a
    licence to skip seeing it, which the task verdict must not accept, so
    this one ships OFF for the SuccessVerifier and the presence check stands.
    ``condition_anchor_references`` ships ON for the ``verify_placed`` LEAF,
    which has an 8-second budget, exists only to catch a positive failure, and
    is followed by the task-level verifier that does still check presence. A
    container carried away by something other than the tree is therefore
    noticed at the end of the run rather than mid-tree.
  * ``fusion``: verify detections are not passed through the ASPIRE quality
    gate, so a badly-segmented mask is not offered an alternate prompt. No
    predicate reads any fusion field (low_quality / axis_trust /
    fused_confidence); the gate exists for grasp geometry, which verification
    does not compute.
  * ``reuse_capture_s``: the verification judges an instant up to the window
    old rather than now. THIS IS THE ONE CUT THAT CAN BE WRONG rather than
    merely blind: a frame taken before the jaws opened shows the object still
    in the gripper, and nothing here can tell that frame from a current one.
    ``pipeline.capture()`` is called by re-detection, failure snapshots and the
    detect routes (the camera STREAM does not touch it), so a pre-release
    frame inside the window is possible on a busy server. It therefore ships
    OFF (0.0) and is only for a caller who knows the arm has not moved and the
    gripper has not changed state since. The other three cuts are blind at
    worst; this one is the reason the default is not 2.0.
  * camera scoping never removes a camera that could have voted: the frustum
    test only drops cameras the labels do not project into, and it falls back
    to every camera whenever the plan-time geometry is not available.

Nothing here can turn an abstain into a pass, and nothing here removes a
camera's ability to vote FAIL on evidence it actually has.
"""

from __future__ import annotations

import logging
import time
from dataclasses import dataclass, field
from typing import Any, Dict, Iterable, List, Optional, Sequence, Set, Tuple

import numpy as np
from spark_real.utils.env_flags import as_bool

logger = logging.getLogger(__name__)

# A capture older than this is never reused, whatever the config says: past a
# couple of seconds the arm has moved and the scene may have settled further.
REUSE_CAPTURE_MAX_S = 5.0


@dataclass
class ScopeConfig:
    """The ``verification.scope`` block.

    ``enabled: false`` restores the unscoped behaviour exactly: every label,
    every camera, the fusion gate on, a fresh capture every time.
    """

    enabled: bool = True
    telemetry_short_circuit: bool = True
    # OFF: a frame from before the jaws opened is indistinguishable from a
    # current one, and only the caller knows whether the scene has moved.
    reuse_capture_s: float = 0.0
    # Skip re-detecting a reference the tree never touched. OFF for the task
    # verdict (its presence check is load-bearing), ON for the condition leaf.
    anchor_references: bool = False
    condition_anchor_references: bool = True
    fusion: bool = False
    cameras: str = "auto"  # "auto" (frustum-scoped) or "all"

    @classmethod
    def from_raw(cls, raw: Any) -> "ScopeConfig":
        cfg = cls()
        block = (raw or {}).get("verification") if hasattr(raw, "get") else None
        block = block.get("scope") if isinstance(block, dict) else None
        if not isinstance(block, dict):
            return cfg
        for key in (
            "enabled",
            "telemetry_short_circuit",
            "anchor_references",
            "condition_anchor_references",
            "fusion",
        ):
            if block.get(key) is not None:
                setattr(cfg, key, as_bool(block[key], getattr(cfg, key)))
        if block.get("reuse_capture_s") is not None:
            try:
                cfg.reuse_capture_s = float(block["reuse_capture_s"])
            except (TypeError, ValueError):
                logger.warning("verification.scope.reuse_capture_s is not a number; ignoring")
        if block.get("cameras") is not None:
            value = str(block["cameras"]).strip().lower()
            if value in ("auto", "all"):
                cfg.cameras = value
            else:
                logger.warning("verification.scope.cameras=%r is not auto|all; using auto", value)
        cfg.reuse_capture_s = max(0.0, min(cfg.reuse_capture_s, REUSE_CAPTURE_MAX_S))
        return cfg

    @classmethod
    def from_pipeline(cls, pipeline: Any) -> "ScopeConfig":
        profile = getattr(pipeline, "profile", None)
        return cls.from_raw(getattr(profile, "raw", None))

    def describe(self) -> str:
        if not self.enabled:
            return "scope OFF (every label, every camera, fusion on, fresh capture)"
        return (
            f"scope ON (reuse<={self.reuse_capture_s:.1f}s, "
            f"anchor_references={self.anchor_references}/"
            f"{self.condition_anchor_references}, fusion={self.fusion}, "
            f"cameras={self.cameras}, telemetry={self.telemetry_short_circuit})"
        )


# --- 1. telemetry ----------------------------------------------------------


def still_holding(executor) -> Optional[bool]:
    """Does the gripper read HOLDING right now? ``None`` = unreadable.

    This is the one question after a release that proprioception answers
    outright: jaws that never let go mean the object was never placed, and no
    camera is needed to say so. The converse is NOT true and is not claimed
    here: open, empty jaws over a container do not prove the object stayed in
    it (see success_verifier._maybe_occlusion_escape).
    """
    # The executor's OWN release gate outranks a live gripper read. _verify_grasp
    # trusts the Robotiq gObj flag, and gObj goes high for anything that arrests
    # the jaws, including a deliberate empty close (close_gripper +
    # compliant_push tamping a seated tool raises gObj after a correct place, and
    # a hardware read alone would then re-pick a tool already in place).
    #
    # Keyed on the witness, NOT on _holding: _holding is cleared unconditionally
    # before the witness is read (executor_release), so a `_holding is False`
    # veto would delete this rung entirely; it is only ever reached after a
    # release. A witness that abstained (released None/False) still falls through
    # to the hardware read, so a genuinely failed release still short-circuits.
    witness = getattr(executor, "_release_witness", None)
    if witness is not None:
        released = getattr(witness, "released", None)
        if released is None and isinstance(witness, dict):
            released = witness.get("released")
        if released is True:
            logger.info(
                "verify scope: the release witness says the jaws opened and the "
                "object stayed; not asking the gripper"
            )
            return False

    check = getattr(executor, "_verify_grasp", None)
    if callable(check):
        try:
            return bool(check())
        except Exception as exc:  # noqa: BLE001 - telemetry must never raise out
            logger.warning("verify scope: grip state unreadable (%s)", exc)
            return None
    held = getattr(executor, "_holding", None)
    return None if held is None else bool(held)


# --- 2. captures -----------------------------------------------------------


def captures_for_verify(pipeline, cfg: ScopeConfig) -> Tuple[Dict[str, dict], float, bool]:
    """``(captures, capture_time, reused)``.

    Reuses ``pipeline._last_captures`` when it is younger than
    ``reuse_capture_s``; otherwise takes a fresh one.
    """
    if pipeline is None:
        return {}, 0.0, False
    if cfg.enabled and cfg.reuse_capture_s > 0:
        stamp = getattr(pipeline, "_last_captures_t", None)
        last = getattr(pipeline, "_last_captures", None)
        if last and stamp:
            age = time.time() - float(stamp)
            if 0 <= age <= cfg.reuse_capture_s:
                logger.info("Verify: reusing the capture from %.2fs ago (no new frames)", age)
                return last, float(stamp), True
    captures = pipeline.capture() or {}
    return captures, float(getattr(pipeline, "_last_captures_t", time.time())), False


# --- 3. labels -------------------------------------------------------------


@dataclass
class PromptPlan:
    """Which base prompts the verify detection actually runs, and what it drops."""

    prompts: List[str] = field(default_factory=list)
    dropped: List[str] = field(default_factory=list)
    reason: str = ""

    @property
    def saved(self) -> int:
        return len(self.dropped)


def plan_prompts(
    labels: Sequence[str],
    anchored_bases: Iterable[str],
    cfg: ScopeConfig,
    base_of,
) -> PromptPlan:
    """Base prompts for the verify detection, minus the plan-anchored references.

    ``anchored_bases`` are base labels whose pose the verifier is going to take
    from the plan-time map anyway; the CALLER decides which those are, because
    the task verdict and the condition leaf answer that differently (see
    ``anchor_references`` vs ``condition_anchor_references``). Prompting for
    them buys a presence check and nothing else, which is the evidence traded
    away here.
    """
    seen: Set[str] = set()
    ordered: List[str] = []
    for lab in labels:
        base = base_of(lab)
        if base and base not in seen:
            seen.add(base)
            ordered.append(base)
    anchored = {b for b in (base_of(a) for a in anchored_bases) if b}
    if not cfg.enabled or not anchored:
        return PromptPlan(prompts=ordered, reason="every predicate label")
    # Never drop the last prompt: a detection pass with no prompts sees nothing
    # and every predicate abstains, which is worse than one extra inference.
    keep = [p for p in ordered if p not in anchored]
    if not keep:
        return PromptPlan(prompts=ordered, reason="every label is anchored; keeping all")
    dropped = [p for p in ordered if p in anchored]
    return PromptPlan(
        prompts=keep,
        dropped=dropped,
        reason=f"plan-anchored reference(s) {sorted(dropped)} not re-detected",
    )


# --- 4. cameras ------------------------------------------------------------


def _projects_into(cal, xyz) -> Optional[bool]:
    """Does world point ``xyz`` land inside this camera's image? None = unknown."""
    if cal is None or xyz is None:
        return None
    try:
        extrinsic = np.asarray(cal.extrinsic, dtype=float)
        K = np.asarray(cal.intrinsic_matrix, dtype=float)
        width = float(cal.width)
        height = float(cal.height)
    except Exception:  # noqa: BLE001 - a calibration surface we do not know
        return None
    if extrinsic.shape != (4, 4) or K.shape != (3, 3) or np.allclose(extrinsic, np.eye(4)):
        # Uncalibrated: it has no world frame to test against.
        return None
    try:
        world = np.append(np.asarray(xyz, dtype=float)[:3], 1.0)
        cam = np.linalg.inv(extrinsic) @ world
        if cam[2] <= 1e-6:
            return False  # behind the camera
        uv = K @ (cam[:3] / cam[2])
        return bool(0 <= uv[0] < width and 0 <= uv[1] < height)
    except Exception:  # noqa: BLE001
        return None


def plan_cameras(
    captures: Dict[str, dict],
    anchors: Sequence[Any],
    cfg: ScopeConfig,
) -> Tuple[List[str], List[str]]:
    """``(cameras_to_detect_on, dropped)``.

    ``anchors`` are the plan-time world positions of the labels under test. A
    camera is dropped only when EVERY anchor provably falls outside its image;
    an unknown projection keeps the camera. If the filter would empty the set,
    every camera is kept; verifying nothing is never the cheaper answer.
    """
    cams = list(captures or ())
    if not cfg.enabled or cfg.cameras != "auto" or not cams:
        return cams, []
    usable = [a for a in anchors if a is not None]
    if not usable:
        return cams, []
    keep, dropped = [], []
    for cam in cams:
        cal = (captures.get(cam) or {}).get("calibration")
        verdicts = [_projects_into(cal, a) for a in usable]
        if any(v is not False for v in verdicts):
            keep.append(cam)
        else:
            dropped.append(cam)
    if not keep:
        return cams, []
    return keep, dropped


# --- the detection call itself ---------------------------------------------


def detect_scoped(
    pipeline,
    captures: Dict[str, dict],
    prompts: Sequence[str],
    cfg: ScopeConfig,
    multi_instance_prompts: Optional[Set[str]] = None,
    capture_time: float = 0.0,
) -> List[Any]:
    """One verify detection pass, memoised inside the freshness window.

    The memo is what stops a ``verify_placed`` leaf and the task-level
    SuccessVerifier (which run within a second of each other on the same
    predicate) paying for the same inferences twice. It is keyed on the
    capture, the prompt set and the fusion flag, so it can only ever return
    detections computed from exactly the frames the caller is holding.
    """
    if pipeline is None or not prompts:
        return []
    multi = set(multi_instance_prompts or ())
    key = (round(float(capture_time), 3), tuple(sorted(prompts)), tuple(sorted(multi)), cfg.fusion)
    if cfg.enabled and cfg.reuse_capture_s > 0 and capture_time:
        cached = getattr(pipeline, "_verify_detect_memo", None)
        if cached and cached[0] == key and (time.time() - cached[1]) <= cfg.reuse_capture_s:
            logger.info(
                "Verify: reusing %d detection(s) for %s (no SAM3 pass)",
                len(cached[2]),
                sorted(prompts),
            )
            return list(cached[2])

    kwargs: Dict[str, Any] = {"prompts": list(prompts)}
    if multi:
        kwargs["multi_instance_prompts"] = multi
    if cfg.enabled and not cfg.fusion:
        kwargs["use_fusion"] = False
    dets = _call_detect(pipeline, captures, kwargs)
    if cfg.enabled and capture_time:
        pipeline._verify_detect_memo = (key, time.time(), list(dets))
    return dets


def _call_detect(pipeline, captures, kwargs: Dict[str, Any]) -> List[Any]:
    """``pipeline.detect`` with graceful degradation for older backends."""
    for drop in (None, "use_fusion", "multi_instance_prompts"):
        if drop is not None:
            if drop not in kwargs:
                continue
            kwargs = {k: v for k, v in kwargs.items() if k != drop}
            logger.warning("Verify: detect() has no %s; continuing without it", drop)
        try:
            return list(pipeline.detect(captures, **kwargs) or [])
        except TypeError as exc:
            if drop == "multi_instance_prompts" or "unexpected keyword" not in str(exc):
                raise
    return []


def per_camera(detections: Iterable[Any]) -> Dict[str, List[Any]]:
    out: Dict[str, List[Any]] = {}
    for det in detections or ():
        out.setdefault(getattr(det, "camera", None) or "unknown", []).append(det)
    return out


__all__ = [
    "REUSE_CAPTURE_MAX_S",
    "PromptPlan",
    "ScopeConfig",
    "captures_for_verify",
    "detect_scoped",
    "per_camera",
    "plan_cameras",
    "plan_prompts",
    "still_holding",
]

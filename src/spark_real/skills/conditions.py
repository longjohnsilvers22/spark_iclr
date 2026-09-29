"""Condition leaves: the only nodes in the grammar that report a FACT.

A behaviour tree can only react to a failure it can see: without these, the
tree's sole failure signal is "a motion primitive raised", so a release that
puts the plushie NEXT TO the bowl reports success and the tree walks on.

Two conditions, both cheap and both non-moving:

  ``verify_grasp``   is the gripper actually holding something right now?
  ``verify_placed``  is the object actually inside / on its target?

Both are meant to sit as the last leaf of a sequence inside a ``retry`` or
``fallback``, so their failure is what triggers the recovery branch.

ABSTENTION IS SUCCESS. When the evidence is not there -- no pipeline, no
camera, the object occluded, a container whose extent perception could not
recover -- these return success with a message saying so. On a real arm,
re-running a motion because we could not see is worse than not re-running it:
the object is probably where we put it, and a blind retry drives the gripper
back into a scene we have no model of. Failure is reported only on positive
evidence of failure.

CHEAP IS PART OF THE CONTRACT. A fresh capture off every camera plus a SAM3
pass per label per camera costs a minute (measured: 7 inferences, 54 seconds
against an 8-second primitive budget), and a condition leaf that costs a
minute is not a condition leaf. ``verify_placed`` goes through
control.verify_scope (telemetry first, then a reused capture, then only the
labels and cameras the predicate needs) and through the SAME identity binder
and per-camera evaluation the task-level SuccessVerifier uses, so a placed
object is not deduped out of its own container by the cross-camera merge.
"""

import logging
import time

from spark_real.control import success_predicates, verify_scope
from spark_real.control.success_verifier import (
    CAMERA_ORDER,
    IdentityBinder,
    VerifyConfig,
    _plan_key,
    fuse_votes,
)
from spark_real.skills.primitives import _result
from spark_real.skills.registry import spark_skill

logger = logging.getLogger(__name__)


def _base_prompt(label: str) -> str:
    """Strip the instance suffix: SAM3 is prompted with "fork", not "fork 2"."""
    label = str(label or "").strip()
    if label and label[-1].isdigit() and " " in label:
        return label.rsplit(" ", 1)[0]
    return label


def _detect(pipeline, labels):
    """Fresh capture + detect for ``labels``. Returns merged detections."""
    prompts = []
    for label in labels:
        prompt = _base_prompt(label)
        if prompt and prompt not in prompts:
            prompts.append(prompt)
    captures = pipeline.capture()
    dets = pipeline.detect(captures, prompts=prompts) or []
    merge = getattr(pipeline, "merge_detections", None)
    if callable(merge):
        merged = merge(dets)
        if merged is not None:
            dets = merged
    return list(dets)


def _det_label(det) -> str:
    value = getattr(det, "label", None)
    if value is None and hasattr(det, "get"):
        value = det.get("label")
    return str(value or "")


def _det_conf(det) -> float:
    value = getattr(det, "confidence", None)
    if value is None and hasattr(det, "get"):
        value = det.get("confidence")
    try:
        return float(value)
    except (TypeError, ValueError):
        return 0.0


def _make_lookup(detections):
    """Label -> highest-confidence detection whose label matches.

    Matching is exact first, then base-prompt, so "stuffed animal" finds
    "stuffed animal 1" when SAM3 numbered a single instance.
    """

    def lookup(label):
        want = str(label or "").strip().lower()
        want_base = _base_prompt(want)
        best, best_conf = None, -1.0
        for det in detections:
            got = _det_label(det).strip().lower()
            if got != want and _base_prompt(got) != want_base:
                continue
            conf = _det_conf(det)
            if conf > best_conf:
                best, best_conf = det, conf
        return best

    return lookup


@spark_skill(
    name="verify_grasp",
    description=(
        "CONDITION (no motion): succeeds only if the gripper is actually "
        "holding an object right now. Put it after grasp inside a retry so a "
        "missed grasp re-runs the acquire instead of transporting nothing."
    ),
    params={},
)
def verify_grasp(executor, params: dict):
    """Proprioceptive hold check. Abstains (succeeds) if unreadable."""
    t0 = time.time()
    check = getattr(executor, "_verify_grasp", None)
    if not callable(check):
        return _result(
            "verify_grasp", True, "no grasp verifier on this executor", time.time() - t0
        )
    try:
        holding = bool(check())
    except Exception as exc:  # noqa: BLE001 - a condition must never raise
        logger.warning("verify_grasp: check raised (%s); abstaining", exc)
        return _result(
            "verify_grasp", True, f"grasp check unavailable: {exc}", time.time() - t0
        )
    return _result(
        "verify_grasp",
        holding,
        "gripper is holding" if holding else "gripper is EMPTY",
        time.time() - t0,
    )


@spark_skill(
    name="verify_placed",
    description=(
        "CONDITION (no motion): re-detects and succeeds only if 'obj' is "
        "actually inside (or on, with relation: on) 'container'. Put it after "
        "release inside a fallback so a drop that missed the container "
        "triggers the recovery branch. Abstains (succeeds) when it cannot see."
    ),
    params={
        "obj": str,
        "container": str,
        "relation": str,  # "inside" (default) or "on"
    },
)
def verify_placed(executor, params: dict):
    """Vision check that the released object ended up in its target."""
    t0 = time.time()
    obj = params.get("obj") or params.get("keypoint_label") or ""
    container = params.get("container") or params.get("target_label") or ""
    relation = str(params.get("relation") or "inside").lower()
    if relation not in ("inside", "on"):
        relation = "inside"
    container_key = "container" if relation == "inside" else "surface"

    if not obj or not container:
        return _result(
            "verify_placed",
            True,
            "abstain: needs both 'obj' and 'container'",
            time.time() - t0,
        )

    pipeline = getattr(executor, "_pipeline", None)
    if pipeline is None:
        return _result(
            "verify_placed", True, "abstain: no perception pipeline", time.time() - t0
        )

    predicate = success_predicates.Predicate(
        pred=relation, params={"obj": obj, container_key: container}
    )
    scope = verify_scope.ScopeConfig.from_pipeline(pipeline)

    # RUNG 0 -- telemetry. Jaws that still read HOLDING after the release are
    # positive evidence the object was never placed, and they cost no capture,
    # no SAM3 pass and no wall clock. The converse is deliberately NOT claimed:
    # open, empty jaws over a container do not prove the object stayed in it.
    if scope.enabled and scope.telemetry_short_circuit:
        if verify_scope.still_holding(executor) is True:
            return _result(
                "verify_placed",
                False,
                f"{predicate.describe()} is FALSE: the gripper still reads "
                f"HOLDING after the release (no vision needed)",
                time.time() - t0,
            )

    try:
        vote_status, detail = _judge(executor, pipeline, predicate, scope)
    except Exception as exc:  # noqa: BLE001 - a condition must never raise
        logger.warning("verify_placed: detection failed (%s); abstaining", exc)
        return _result(
            "verify_placed", True, f"abstain: detection failed: {exc}", time.time() - t0
        )

    logger.info("verify_placed(%s, %s) -> %s %s", obj, container, vote_status, detail)
    if vote_status == success_predicates.FAIL:
        return _result(
            "verify_placed",
            False,
            f"{predicate.describe()} is FALSE: {detail}",
            time.time() - t0,
        )
    passed = vote_status == success_predicates.PASS
    return _result(
        "verify_placed",
        True,
        f"{predicate.describe()} {'confirmed' if passed else 'abstain'}: {detail}",
        time.time() - t0,
    )


def _judge(executor, pipeline, predicate, scope):
    """``(status, detail)`` for one predicate. PASS / FAIL / UNVERIFIED."""
    if not scope.enabled:
        return _judge_merged(executor, pipeline, predicate)
    return _judge_scoped(executor, pipeline, predicate, scope)


def _judge_merged(executor, pipeline, predicate):
    """Pre-scoping behaviour: fresh capture, every camera, cross-camera merge.

    Kept reachable by ``verification.scope.enabled: false``. The merge's
    cross-label dedup can drop a placed object out of its own container (see
    perception/dedup.py), so this path is weaker as well as slower.
    """
    detections = _detect(pipeline, predicate.labels)
    view = success_predicates.SceneView(
        camera="fused",
        lookup=_make_lookup(detections),
        held=bool(getattr(executor, "_holding", False)),
    )
    vote = success_predicates.evaluate_predicate(
        predicate, view, success_predicates.EvalConfig()
    )
    status = {
        success_predicates.PASS: success_predicates.PASS,
        success_predicates.FAIL: success_predicates.FAIL,
    }.get(vote.vote, success_predicates.UNVERIFIED)
    return status, f"[{vote.mode}] {vote.detail}"


def _judge_scoped(executor, pipeline, predicate, scope):
    """The scoped path: the SuccessVerifier's machinery, one predicate wide.

    Same identity binder, same per-camera independence, same fusion rule -- so
    a condition leaf and the task verdict cannot disagree about what they saw.
    What it does NOT do is merge across cameras: the merge is where a placed
    object gets deduped against the container it is sitting in.
    """
    vcfg = VerifyConfig.from_pipeline(pipeline)
    plan_map = getattr(executor, "detection_map", None) or {}

    # The container is scenery this tree never touched, so its plan-time pose
    # is the reference frame -- and re-detecting it is actively harmful once
    # the placed object occludes its interior (see
    # success_verifier.IdentityBinder._bind_reference). Drop its prompt when
    # the 2D fallback, the only consumer of a fresh container mask, is off.
    object_keys = {_plan_key(plan_map, o) for o in predicate.object_labels} - {None}
    anchored = {
        key
        for key in (_plan_key(plan_map, r) for r in predicate.reference_labels)
        if key is not None and key not in object_keys
    }
    pose_only = (
        set(anchored)
        if scope.condition_anchor_references and vcfg.fallback_2d_containment > 1.0
        else set()
    )

    captures, stamp, reused = verify_scope.captures_for_verify(pipeline, scope)
    if not captures:
        return success_predicates.UNVERIFIED, "no camera captured"

    anchor_keys = {_plan_key(plan_map, lab) for lab in predicate.labels}
    anchors = [
        _plan_xyz(plan_map, key) for key in anchor_keys if key is not None
    ]
    keep, dropped = verify_scope.plan_cameras(captures, anchors, scope)
    if dropped:
        logger.info("verify_placed: %s cannot see %s; skipped", sorted(dropped), predicate.labels)
    captures = {c: captures[c] for c in keep}

    plan = verify_scope.plan_prompts(predicate.labels, pose_only, scope, _base_prompt)
    logger.info(
        "verify_placed: %d prompt(s) %s over %d camera(s) %s (%s; %s)",
        len(plan.prompts),
        plan.prompts,
        len(captures),
        sorted(captures),
        "reused capture" if reused else "fresh capture",
        plan.reason,
    )
    dets = verify_scope.detect_scoped(
        pipeline, captures, plan.prompts, scope, capture_time=stamp
    )
    per_cam = verify_scope.per_camera(dets)

    held = getattr(executor, "_holding", None)
    cfg = success_predicates.EvalConfig(
        min_conf=vcfg.min_conf,
        fallback_2d_containment=vcfg.fallback_2d_containment,
        inside_xy_margin_m=vcfg.inside_xy_margin_m,
        inside_z_tol_m=vcfg.inside_z_tol_m,
    )
    order = [c for c in CAMERA_ORDER if c in per_cam]
    order += [c for c in per_cam if c not in CAMERA_ORDER]
    votes, silent = [], [c for c in captures if c not in per_cam]
    for cam in order:
        # The object under test is the ONE thing this tree moved, so it binds
        # by elimination against its stationary siblings rather than by
        # standing still -- which it demonstrably did not.
        binder = IdentityBinder(
            plan_map,
            per_cam[cam],
            manipulated=object_keys,
            plan_anchored=anchored,
            pose_only=pose_only,
        )
        view = success_predicates.SceneView(
            camera=cam,
            lookup=binder.lookup,
            held=None if held is None else bool(held),
            explain=binder.explain,
        )
        votes.append(success_predicates.evaluate_predicate(predicate, view, cfg))
    return fuse_votes(votes, silent_cameras=silent)


def _plan_xyz(plan_map, key):
    get = getattr(plan_map, "get", None)
    entry = get(key) if callable(get) else None
    return success_predicates.detection_xyz(entry) if entry is not None else None

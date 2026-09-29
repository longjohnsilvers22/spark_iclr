"""Task-success verification: cheap gates, per-camera votes, ENPIRE fusion.

Order of work, cheapest first:

  G1 grasp     did the jaws ever hold anything (gObj / TCP wrench / jaw pos)
  G2 transport was the object still held at every stationary waypoint
  G3 release   was it held immediately before opening and gone right after

A gate that is explicitly False short-circuits to ``fail`` with NO capture and
NO SAM3; a drop mid-transport is caught for free. Only if the gates are clean
does the vision path run: every camera detects independently (no cross-camera
merge, no shared masks) and judges the whole predicate list on its own; the
per-camera verdicts are then fused.

Predicate labels are re-bound to detections GEOMETRICALLY, never by instance
number; see :class:`IdentityBinder`. Instance numbering is not stable across
detection passes, so re-resolving ``"fork 1"`` after the verify re-detection
could bind a predicate to a different physical fork than the tree manipulated.

Fail-closed everywhere: a missing check, a throwing check, or no predicate at
all yields ``unverified``, never ``pass``. ``unverified`` must neither reinforce
nor suppress a BT-library row.
"""

from __future__ import annotations

import logging
import time
from dataclasses import dataclass
from typing import Any, Dict, Iterable, List, Mapping, Optional, Sequence, Set, Tuple

import numpy as np

from spark_real.utils.env_flags import as_bool
from spark_real.bt_label_resolver import LabelResolvingDetectionMap, base_label, canon_label
from spark_real.control import verify_scope
from spark_real.control.success_predicates import (
    ABSTAIN,
    FAIL,
    PASS,
    UNVERIFIED,
    CameraVote,
    EvalConfig,
    SceneView,
    VerifyBlockError,
    VerifyOutcome,
    VerifySpec,
    camera_verdict,
    container_obb_xy,
    derive_default_verify,
    detection_field,
    detection_xyz,
    displaced_labels,
    evaluate_predicate,
    obb_contains,
    parse_verify_block,
)

logger = logging.getLogger(__name__)

CAMERA_ORDER = ("birdview", "sideview", "wrist")
GATE_NAMES = ("grasp", "transport", "release")

# How far a re-detected object may sit from its plan-time position and still
# count as the same physical object that never moved. Covers centroid jitter
# and depth noise; well under the spacing of distinct utensils on a table.
STATIC_MATCH_TOL_M = 0.06

# Detection fields a predicate reads off a reference object (container /
# surface / base / target). These are the ones taken from the PLAN-TIME
# detection when the reference is plan-anchored; ``mask`` and ``camera`` are
# deliberately NOT here; see _plan_anchored_reference.
_REFERENCE_FIELDS = (
    "label",
    "position_3d",
    "aspect_ratio",
    "world_major_axis_rad",
    "obb_minor_m",
    "slots",
)


def _plan_anchored_reference(plan_det: Any, observed: Any) -> Dict[str, Any]:
    """A reference detection whose POSE is plan-time and whose MASK is now.

    World-frame geometry (position, OBB, slots) comes from the plan-time
    detection: that is the pose the arm actually serviced, and it predates the
    occlusion the placement itself created.

    ``mask`` and ``camera`` must NOT come from plan time. The 2D containment
    fallback compares two masks pixel-for-pixel in ONE camera's image, and the
    plan-time map is a cross-camera merge; its mask may belong to a different
    view entirely. So the image-space fields stay with the fresh detection in
    this camera, and the world-frame fields come from plan time.

    ``confidence`` likewise comes from the fresh detection: it answers "did
    this camera see the container just now", which is what licenses reusing the
    plan-time pose at all. Falls back to the plan-time value when the
    re-detection has none.
    """
    ref: Dict[str, Any] = {name: detection_field(plan_det, name) for name in _REFERENCE_FIELDS}
    conf = detection_field(observed, "confidence") if observed is not None else None
    ref["confidence"] = conf if conf is not None else detection_field(plan_det, "confidence")
    # ``observed`` is None when verify_scope dropped this reference's prompt
    # altogether (verification.scope.anchor_references). There is then no
    # image-space evidence in this camera at all, so the 2D containment
    # fallback abstains, which is why the prompt is only dropped when that
    # fallback is switched off. The world-frame fields, which is all the 3D
    # test reads, are unaffected.
    ref["mask"] = detection_field(observed, "mask") if observed is not None else None
    ref["camera"] = detection_field(observed, "camera") if observed is not None else None
    ref["plan_anchored"] = True
    return ref


@dataclass
class VerifyConfig:
    enabled: bool = True
    min_conf: float = 0.35
    require_two_views: bool = False
    inside_xy_margin_m: float = -0.02
    inside_z_tol_m: float = 0.06
    # > 1.0 disables the 2D fallback (votes abstain instead). Shipped OFF:
    # replayed over 70 corpus episodes x 3 cameras the image-space test passed
    # 6 temporal NEGATIVES and only 1 true positive, and 2D-failed 42 true
    # positives; as a veto it would suppress real successes. See
    # tests/test_verify_replay.py. Set 0.90 to re-enable.
    fallback_2d_containment: float = 1.01
    on_missing: str = UNVERIFIED

    @classmethod
    def from_raw(cls, raw: Any) -> "VerifyConfig":
        block = (raw or {}).get("verification") if hasattr(raw, "get") else None
        cfg = cls()
        if not isinstance(block, dict):
            return cfg
        for key in (
            "enabled",
            "min_conf",
            "require_two_views",
            "inside_xy_margin_m",
            "inside_z_tol_m",
            "fallback_2d_containment",
            "on_missing",
        ):
            if block.get(key) is None:
                continue
            cur, val = getattr(cfg, key), block[key]
            if isinstance(cur, bool):
                val = (
                    as_bool(val, False)
                    if isinstance(val, str)
                    else bool(val)
                )
            elif isinstance(cur, float):
                val = float(val)
            else:
                val = str(val)
            setattr(cfg, key, val)
        if cfg.on_missing == PASS:
            logger.warning("verification.on_missing=pass is not allowed; using unverified")
            cfg.on_missing = UNVERIFIED
        return cfg

    @classmethod
    def from_pipeline(cls, pipeline: Any) -> "VerifyConfig":
        profile = getattr(pipeline, "profile", None)
        return cls.from_raw(getattr(profile, "raw", None))


@dataclass
class ReleaseWitness:
    held_before: bool
    held_after: bool
    tcp_xyz: Tuple[float, float, float]
    container_label: Optional[str] = None
    inside_region: Optional[bool] = None
    tcp_dz_to_container: Optional[float] = None
    # Jaw evidence behind held_after. ``released`` is the gate's verdict:
    # True = the jaws demonstrably opened, False = they did not, None = no
    # usable read (abstain). The raw numbers are logged so the next rig run
    # calibrates GRIPPER_RELEASED_MAX_POS / GRIPPER_RELEASE_MIN_TRAVEL.
    released: Optional[bool] = None
    jaw_pos_before: Optional[float] = None
    jaw_pos_after: Optional[float] = None
    obj_after: Optional[bool] = None
    confirm_s: float = 0.0

    def to_dict(self) -> dict:
        return {
            "held_before": bool(self.held_before),
            "held_after": bool(self.held_after),
            "tcp_xyz": [float(v) for v in self.tcp_xyz],
            "container_label": self.container_label,
            "inside_region": self.inside_region,
            "tcp_dz_to_container": self.tcp_dz_to_container,
            "released": self.released,
            "jaw_pos_before": self.jaw_pos_before,
            "jaw_pos_after": self.jaw_pos_after,
            "obj_after": self.obj_after,
            "confirm_s": round(float(self.confirm_s), 3),
        }


# gate bookkeeping: executors call these; the verifier only reads


def _gates(executor) -> Dict[str, Optional[bool]]:
    flags = getattr(executor, "_verify_gates", None)
    if not isinstance(flags, dict):
        flags = {}
        try:
            executor._verify_gates = flags
        except Exception:
            pass
    return flags


def set_gate(executor, name: str, value: Optional[bool], detail: str = "") -> None:
    if executor is None or name not in GATE_NAMES:
        return
    _gates(executor)[name] = value
    if value is False:
        logger.warning("Verify gate '%s' FAILED%s", name, f": {detail}" if detail else "")


def record_grasp_verdict(executor, holding: bool, detail: str = "", verdict: Any = None) -> None:
    """G1: called from the grasp verify path with its structured verdict.

    ``verdict`` is the executor's own richer record (control.executor_grasp's
    GraspVerdict) when it has one; it is stored verbatim so the ASPIRE trace
    keeps the evidence behind the boolean.
    """
    set_gate(executor, "grasp", bool(holding), detail)
    if holding:
        # A successful re-grasp clears an earlier transport drop.
        if _gates(executor).get("transport") is False:
            set_gate(executor, "transport", True, "re-grasp after drop")
    try:
        executor._last_grasp_verdict = (
            verdict if verdict is not None else {"holding": bool(holding), "detail": detail}
        )
    except Exception:
        pass


def _verdict_holding(verdict: Any) -> Optional[bool]:
    """Read the boolean out of either verdict shape (dict or GraspVerdict)."""
    if verdict is None:
        return None
    if isinstance(verdict, dict):
        val = verdict.get("holding")
    else:
        val = getattr(verdict, "held", None)
    return None if val is None else bool(val)


def note_transport_drop(executor, waypoint: str = "") -> None:
    """G2: called when a stationary-waypoint grip check says DROPPED.

    Also upgrades the stored grasp verdict's typed outcome to SLIP when the
    grasp had classified SECURED: a secured close followed by a lost grip
    at a stationary waypoint is the Robotiq analog of the sim's
    aperture-collapse SLIP. Additive metadata only.
    """
    set_gate(executor, "transport", False, f"drop at {waypoint}")
    try:
        from spark_real.control.grasp_outcome import (
            GraspOutcome,
            slip_after_secured,
        )

        v = getattr(executor, "_last_grasp_verdict", None)
        prev_raw = (
            v.get("outcome") if isinstance(v, dict)
            else getattr(v, "outcome", None)
        )
        prev = GraspOutcome(prev_raw) if prev_raw else None
        slip = slip_after_secured(prev, False)
        if slip is not None:
            if isinstance(v, dict):
                v["outcome"] = slip.value
            else:
                v.outcome = slip.value
            logger.warning(
                "Grasp outcome: SECURED -> SLIP (drop at %s)", waypoint
            )
    except Exception:  # noqa: BLE001 - annotation must never break the gate
        pass


def reset_run_state(executor, task_scope: bool = True) -> None:
    """Drop every verdict/gate carried by the previous run.

    An executor instance outlives a task, so without this the next run would
    inherit the last run's outcome, gates and release witness, and a stale
    ``pass`` would be read as this run's success.

    ``task_scope`` also clears the executor's own per-task bookkeeping
    (_placed_labels, _holding, the grasp retry budget, ...) via
    ScoreExecutor.reset_task_state. Pass False from a CLOSED-LOOP PASS
    boundary: the passes of one task share that state (_placed_labels is what
    tells pass 3 that pass 1 already handled an object), so clearing it there
    would make the loop re-pick an object already sitting in the receptacle.
    """
    if executor is None:
        return
    for attr, value in (
        ("verify_outcome", None),
        ("_verify_gates", {}),
        ("_release_witness", None),
        ("_last_grasp_verdict", None),
    ):
        try:
            setattr(executor, attr, value)
        except Exception:
            pass
    if not task_scope:
        return
    reset = getattr(executor, "reset_task_state", None)
    if callable(reset):
        try:
            reset()
        except Exception as exc:
            logger.warning("reset_run_state: per-task reset failed: %s", exc)


def _read_holding(executor) -> Optional[bool]:
    """Best available live grip state (gObj-primary when the rig exposes it)."""
    try:
        if hasattr(executor, "_grip_intact") and executor._gripper_type() == "robotiq_2f85":
            return bool(executor._grip_intact())
    except Exception as exc:
        logger.warning("Release witness: grip read failed: %s", exc)
    held = getattr(executor, "_holding", None)
    return None if held is None else bool(held)


def _is_robotiq(executor) -> bool:
    try:
        return executor._gripper_type() == "robotiq_2f85"
    except Exception:
        return False


def _has_jaw_source(executor) -> bool:
    """Does the driver expose any jaw state (register, position or width)?"""
    if _is_robotiq(executor):
        return True
    robot = getattr(executor, "robot", None)
    return any(
        hasattr(robot, name) for name in ("get_gripper_position", "get_gripper_width")
    )


def _jaw_sample(executor) -> Tuple[Optional[bool], Optional[float]]:
    """One fresh jaw sample: ``(object_flag, jaw_pos)``, 0=open .. 255=closed.

    Any driver: the executor's _get_gripper_position rescales a non-Robotiq
    position (or width) onto the Robotiq count scale, so the open-side
    thresholds below apply to every gripper. ``(None, None)`` only when the
    driver has no position source or the read failed. On the Robotiq,
    _read_gripper_state forces a publish and waits for the controller to
    acknowledge it, so this is a current sample, not a stale register.
    """
    try:
        obj, pos = executor._read_gripper_state()
    except Exception as exc:
        logger.warning("Release witness: jaw read failed: %s", exc)
        return None, None
    return (
        None if obj is None else bool(obj),
        None if pos is None else float(pos),
    )


def _jaws_opened(
    executor,
    pos_before: Optional[float],
    pos_after: Optional[float],
    obj_after: Optional[bool],
) -> Optional[bool]:
    """Did the jaws open? None when there is no usable position read.

    Reaching the open stop is conclusive on its own: a gObj that still reads
    contact there is a stale flag, not an object. Short of the stop, gObj=1
    means the jaws were halted BY something, so only travel with gObj clear
    counts. The travel clause carries the thin object, whose jaws legitimately
    stop before the absolute threshold.
    """
    if pos_after is None:
        return None
    max_pos = float(getattr(executor, "GRIPPER_RELEASED_MAX_POS", 60.0))
    min_travel = float(getattr(executor, "GRIPPER_RELEASE_MIN_TRAVEL", 40.0))
    if pos_after <= max_pos:
        return True
    if obj_after is True:
        return False
    if pos_before is not None and (pos_before - pos_after) >= min_travel:
        return True
    return False


def _confirm_jaws_open(
    executor, pos_before: Optional[float]
) -> Tuple[Optional[bool], Optional[bool], Optional[float], float]:
    """Poll for the jaws to read open. ``(opened, gObj, pos, waited_s)``.

    Only safe once the open URScript has finished: every sample uploads a
    program of its own, which would cancel an rq_move_and_wait still running.
    The caller's settle guarantees that. Costs nothing on a normal release;
    the first sample already reads open.
    """
    timeout = float(getattr(executor, "RELEASE_CONFIRM_TIMEOUT_S", 0.6))
    t0 = time.monotonic()
    obj, pos = _jaw_sample(executor)
    opened = _jaws_opened(executor, pos_before, pos, obj)
    while opened is False and (time.monotonic() - t0) < timeout:
        time.sleep(0.05)
        obj, pos = _jaw_sample(executor)
        opened = _jaws_opened(executor, pos_before, pos, obj)
    return opened, obj, pos, time.monotonic() - t0


def begin_release_witness(executor) -> Dict[str, Any]:
    """G3 part 1: call immediately BEFORE opening the gripper."""
    state = {"held_before": _read_holding(executor)}
    _, state["jaw_pos_before"] = _jaw_sample(executor)
    try:
        state["tcp_xyz"] = [float(v) for v in executor._get_current_position()[:3]]
    except Exception:
        state["tcp_xyz"] = [0.0, 0.0, 0.0]
    return state


def finish_release_witness(
    executor, state: Dict[str, Any], container_label: Optional[str] = None
) -> Optional[ReleaseWitness]:
    """G3 part 2: call after the open + settle. Stores the witness on the executor.

    ``held_after`` must NOT be read with _grip_intact. That predicate answers
    "are the jaws still spread on something" after a CLOSE, and its position
    clause (pos < GRIPPER_FULLY_CLOSED = holding) is True for every OPEN
    gripper (pos ~0 on a released 2F-85).
    Post-open the meaningful question is the opposite one: did the jaws reach
    open, or did something stop them.
    """
    try:
        pos_before = state.get("jaw_pos_before")
        opened, obj_after, pos_after, waited = _confirm_jaws_open(executor, pos_before)
        if opened is not None:
            released = opened
        elif _has_jaw_source(executor):
            # The rig HAS a jaw source and it did not read: abstain. Falling
            # back to _grip_intact here would answer "DROPPED" from the same
            # unreadable state and pass the gate on no evidence at all.
            released = None
        else:
            # No jaw state at all: the executor's own flag is the only signal.
            held = _read_holding(executor)
            released = None if held is None else (not held)
        witness = ReleaseWitness(
            held_before=bool(state.get("held_before")),
            held_after=(released is False),
            tcp_xyz=tuple(state.get("tcp_xyz") or (0.0, 0.0, 0.0)),
            container_label=container_label,
            released=released,
            jaw_pos_before=pos_before,
            jaw_pos_after=pos_after,
            obj_after=obj_after,
            confirm_s=waited,
        )
        label = container_label or getattr(executor, "_last_place_label", "") or ""
        det = (getattr(executor, "detection_map", None) or {}).get(label)
        obb = container_obb_xy(det) if det is not None else None
        if obb is not None:
            witness.container_label = label
            witness.inside_region = bool(
                obb_contains(np.asarray(witness.tcp_xyz[:2], dtype=float), obb, 0.0)
            )
            cxyz = detection_xyz(det)
            if cxyz is not None:
                witness.tcp_dz_to_container = float(witness.tcp_xyz[2] - cxyz[2])
        executor._release_witness = witness
        open_max = getattr(executor, "GRIPPER_RELEASED_MAX_POS", 60)
        min_travel = getattr(executor, "GRIPPER_RELEASE_MIN_TRAVEL", 40)
        detail = (
            f"held_before={witness.held_before} released={released} "
            f"jaw {pos_before} -> {pos_after} "
            f"(open<={open_max}, travel>={min_travel}) "
            f"gObj_after={obj_after} confirm={waited:.2f}s"
        )
        if released is None:
            # Abstain rather than invent a verdict: collect_gates reads a
            # missing gate as "the primitive never reported", and a fabricated
            # failure here would fail every task on a rig with no jaw register.
            logger.warning("Release witness: no grip evidence, abstaining (%s)", detail)
        else:
            set_gate(executor, "release", bool(witness.held_before and released), detail)
        logger.info("Release witness: %s", witness.to_dict())
        return witness
    except Exception as exc:
        logger.warning("Release witness failed: %s", exc)
        return None


def collect_gates(executor) -> Tuple[Dict[str, bool], List[str]]:
    """Resolve the three gates to booleans plus the reasons any of them failed.

    An unknown gate reads True (the primitive never ran); only explicit
    negative evidence fails, and the vision path still has to pass.
    """
    flags = dict(_gates(executor))
    if "grasp" not in flags:
        held = _verdict_holding(getattr(executor, "_last_grasp_verdict", None))
        if held is not None:
            flags["grasp"] = held
        else:
            results = [
                r
                for r in (getattr(executor, "_results", None) or [])
                if getattr(r, "action_type", "") == "grasp"
            ]
            if results:
                flags["grasp"] = bool(results[-1].success)
    witness = getattr(executor, "_release_witness", None)
    if "release" not in flags and isinstance(witness, ReleaseWitness):
        # Same verdict finish_release_witness computed, never a re-derivation:
        # released=None means it abstained, and re-deriving from held_after
        # would resurrect the false negative it exists to avoid.
        if witness.released is not None:
            flags["release"] = bool(witness.held_before and witness.released)

    gates, failures = {}, []
    for name in GATE_NAMES:
        val = flags.get(name)
        gates[name] = True if val is None else bool(val)
        if val is False:
            failures.append(name)
    return gates, failures


# identity binding


def _plan_key(plan_map: Any, label: Any) -> Optional[str]:
    """The plan-time detection key a predicate label refers to, or None.

    NOT ``LabelResolvingDetectionMap.resolve_label``. That resolver exists to
    keep a cached tree DRIVING when labels drift, so it deliberately collapses
    ``"fork"`` (and even ``"fork 9"``) onto the lowest-numbered instance. That
    is the right call for dispatching a primitive and the wrong one here: this
    key decides WHICH physical object a success predicate is about, and a
    silent collapse binds the predicate to a fork the tree never touched.

    So: an exact key, a declared role alias, or a canonical base with exactly
    one plan-time instance. More than one instance is ambiguous and returns
    None, which the binder and the predicates read as abstain.
    """
    key = str(label or "").strip()
    if not key or not plan_map:
        return None
    # Iteration yields the real detected labels for both map types.
    keys = list(plan_map)
    if key in keys:
        return key
    # A role alias is an authoritative binding published by the task's
    # perception contract, so it outranks the ambiguity rule below.
    alias = (getattr(plan_map, "aliases", None) or {}).get(key)
    if alias in keys:
        return alias
    want = canon_label(key)
    hits = [k for k in keys if canon_label(k) == want]
    if len(hits) > 1:
        return None
    if hits:
        return hits[0]
    # No instance of this base at all: sub-part / super-part resolution
    # ("knife" -> "knife handle") is unambiguous by construction.
    if isinstance(plan_map, LabelResolvingDetectionMap):
        return plan_map.resolve_label(key)
    return None


def _xy_dist(a: Optional[np.ndarray], b: Optional[np.ndarray]) -> Optional[float]:
    if a is None or b is None:
        return None
    return float(np.linalg.norm(np.asarray(a)[:2] - np.asarray(b)[:2]))


class IdentityBinder:
    """Bind a predicate label to the ONE physical object it was written about.

    Instance numbers are assigned per detection pass and permute freely between
    passes: SAM3 orders them geometrically, so the moment an object moves the
    numbering re-sorts. Re-resolving ``"fork 1"`` against the verify-time
    detections therefore binds the predicate to whatever object happens to be
    first NOW. Identity must come from geometry, not the number.

    Two rules, both anchored on the plan-time detection map the tree was
    written against:

      * a label the tree never grasped did not move, so it binds to the
        re-detection nearest its plan-time position, and only when that
        nearest match is unique inside the tolerance;
      * a label the tree DID grasp is the only object allowed to have moved,
        so it binds to the re-detection left over once every *other* plan-time
        instance of the same base label has claimed its own, still-stationary
        re-detection.

    ``plan_anchored`` is a third, narrower rule for REFERENCE objects (the
    container/surface/base a predicate measures against, when the tree never
    touched it). Those bind to their PLAN-TIME pose and are not re-located at
    all; see :meth:`_bind_reference`.

    Every other case (a sibling that vanished, two candidates on top of each
    other, more than one unaccounted detection, no plan-time anchor) returns
    None, which the predicates read as abstain. Abstaining is a fine outcome;
    binding to the wrong object is the failure this class prevents.
    """

    def __init__(
        self,
        plan_map: Any,
        detections: Iterable[Any],
        manipulated: Optional[Set[str]] = None,
        tol: float = STATIC_MATCH_TOL_M,
        plan_anchored: Optional[Set[str]] = None,
        pose_only: Optional[Set[str]] = None,
    ):
        self._plan = plan_map if isinstance(plan_map, Mapping) else {}
        self._manipulated = set(manipulated or ())
        self._plan_anchored = set(plan_anchored or ())
        # Plan keys whose prompt was never issued (verify_scope dropped it), so
        # "no candidate in this camera" is the scope's doing and must not abstain.
        self._pose_only = set(pose_only or ())
        self._tol = float(tol)
        self._by_base: Dict[str, List[Tuple[str, Any, Optional[np.ndarray]]]] = {}
        for det in detections or ():
            label = str(getattr(det, "label", "") or "")
            self._by_base.setdefault(canon_label(label), []).append(
                (label, det, detection_xyz(det))
            )
        self._cache: Dict[str, Tuple[Optional[Any], str]] = {}

    # public surface consumed by SceneView

    def lookup(self, label: Any) -> Optional[Any]:
        return self._resolve(label)[0]

    def explain(self, label: Any) -> str:
        return self._resolve(label)[1]

    def _resolve(self, label: Any) -> Tuple[Optional[Any], str]:
        key = str(label or "")
        if key not in self._cache:
            self._cache[key] = self._bind(key)
            det, why = self._cache[key]
            if det is None:
                logger.info("Identity binder: %r unbound (%s)", key, why)
            else:
                logger.info("Identity binder: %r -> %s", key, why)
        return self._cache[key]

    # binding

    def _plan_siblings(self, base: str, exclude: Optional[str]):
        """Plan-time instances of the same base label, minus the bound one."""
        return [
            (k, detection_xyz(v))
            for k, v in self._plan.items()
            if canon_label(k) == base and k != exclude
        ]

    def _bind(self, label: str) -> Tuple[Optional[Any], str]:
        base = canon_label(label)
        cands = list(self._by_base.get(base, ()))
        plan_key = _plan_key(self._plan, label)
        if not cands:
            if plan_key is not None and plan_key in self._pose_only:
                plan_det = dict.get(self._plan, plan_key)
                if detection_xyz(plan_det) is None:
                    return None, f"reference {plan_key!r} has no plan-time 3d anchor"
                return _plan_anchored_reference(plan_det, None), (
                    f"plan-time pose of {plan_key!r} (prompt not issued: "
                    f"verification.scope.anchor_references)"
                )
            return None, "not detected in this camera"

        siblings = self._plan_siblings(base, plan_key)
        if plan_key is None:
            # No anchor to check against. A label the scene has exactly one of
            # is still the only candidate there is; anything else is ambiguous.
            if len(cands) == 1 and not siblings:
                return cands[0][1], f"sole {base!r} in view (no plan-time anchor)"
            return None, f"{len(cands)} {base!r} detections and no plan-time anchor"
        # No "sole candidate" shortcut past here. Being the only detection in
        # view is not evidence of identity: an object the tree never grasped
        # that is now 28cm from where it was planned is either a different
        # object or one that moved unobserved, and binding to it is exactly the
        # wrong-object pass this class exists to stop. Geometry decides.
        if plan_key in self._manipulated:
            return self._bind_moved(base, cands, siblings)
        if plan_key in self._plan_anchored:
            return self._bind_reference(cands, plan_key)
        return self._bind_static(base, cands, plan_key)

    def _bind_reference(self, cands, plan_key) -> Tuple[Optional[Any], str]:
        """A reference frame the tree never touched: use its plan-time pose.

        Re-detecting the container of a containment predicate is a pure loss.
        The tree does not touch it, a bowl on a table does not move on its own,
        and the object JUST PLACED occludes its interior, so the surviving
        mask is a partial rim and the depth behind those pixels is sampled off
        the placed object, which slides the backprojected centroid along the
        camera ray. Measured: 8.5cm (birdview), 13.7cm and 17.3cm (sideview) of
        apparent motion on a bowl of radius 8cm, on places that physically
        succeeded. A successful place is exactly what corrupts this measurement.

        The re-detection is still REQUIRED to exist: this method is only
        reached with a non-empty candidate list, so a container that vanished
        from a camera's view still abstains there. What the re-detection may no
        longer do is supply the pose.

        Disagreement policy: a re-detection far from the plan-time anchor is
        LOGGED (the distance goes into the bind explanation and the trace) but
        does not change the binding and does not force an abstain. There is no
        honest threshold available: the drift induced by occlusion is
        unbounded along the camera ray, so any cutoff loose enough to admit the
        17.3cm case is loose enough to admit a container that genuinely moved.
        What guards that case instead is presence (above), the exclusion of any
        container the tree itself displaced (``displaced_labels``), and the
        degenerate-extent abstain in success_predicates.
        """
        plan_det = dict.get(self._plan, plan_key)
        anchor = detection_xyz(plan_det)
        if anchor is None:
            return None, f"reference {plan_key!r} has no plan-time 3d anchor"
        scored = sorted(
            ((_xy_dist(c[2], anchor), c) for c in cands if c[2] is not None),
            key=lambda t: t[0],
        )
        if scored:
            nearest = scored[0][1]
            seen = f"re-detected {scored[0][0] * 100:.1f}cm away"
        else:
            nearest = cands[0]
            seen = "re-detected without 3d"
        ref = _plan_anchored_reference(plan_det, nearest[1])
        return ref, (
            f"plan-time pose of {plan_key!r} ({seen}; the re-detection is not "
            f"used as the reference frame)"
        )

    def _bind_moved(self, base, cands, siblings) -> Tuple[Optional[Any], str]:
        """The grasped object is whatever the stationary siblings do not claim."""
        unclaimed = list(cands)
        for name, anchor in siblings:
            if anchor is None:
                return None, f"un-manipulated {name!r} has no plan-time 3d anchor"
            near = []
            for c in unclaimed:
                d = _xy_dist(c[2], anchor)  # None (no 3d) is not a match; 0.0 is
                if d is not None and d <= self._tol:
                    near.append(c)
            if len(near) != 1:
                return None, (
                    f"{len(near)} detections within {self._tol * 100:.0f}cm of un-manipulated "
                    f"{name!r}; identity ambiguous"
                )
            unclaimed.remove(near[0])
        if len(unclaimed) != 1:
            return None, f"{len(unclaimed)} unaccounted {base!r} detections after elimination"
        if unclaimed[0][2] is None and len(cands) > 1:
            return None, f"leftover {base!r} carries no 3d; identity unconfirmable"
        return unclaimed[0][1], f"{unclaimed[0][0]!r} by elimination ({len(siblings)} sibling(s))"

    def _bind_static(self, base, cands, plan_key) -> Tuple[Optional[Any], str]:
        """An object the tree never touched must still be where it was."""
        anchor = detection_xyz(dict.get(self._plan, plan_key))
        if anchor is None:
            return None, f"{plan_key!r} has no plan-time 3d anchor and {len(cands)} candidates"
        scored = sorted(
            ((_xy_dist(c[2], anchor), c) for c in cands if c[2] is not None),
            key=lambda t: t[0],
        )
        if not scored:
            return None, f"no {base!r} candidate carries 3d"
        if len(scored) < len(cands):
            return None, f"{len(cands) - len(scored)} {base!r} candidate(s) lack 3d; ambiguous"
        tol = self._static_tol(dict.get(self._plan, plan_key))
        best_d, best = scored[0]
        if best_d > tol:
            return None, f"nearest {base!r} is {best_d * 100:.1f}cm from its plan-time position"
        if len(scored) > 1 and scored[1][0] <= tol:
            return None, f"two {base!r} candidates within {tol * 100:.0f}cm; ambiguous"
        return best[1], f"{best[0]!r} {best_d * 100:.1f}cm from its plan-time position"

    def _static_tol(self, plan_det: Any) -> float:
        """Drift a stationary object's *measured centroid* may show.

        The flat tolerance assumes the mask is stable, which holds for an
        object nothing happened to. It does NOT hold for the container just
        placed into: the object now occupying it occludes its interior, so the
        surviving mask is a partial rim whose centroid sits well off the true
        centre (measured: mask fill 0.75 -> 0.37 and a 7.3cm centroid shift on
        a bowl that did not move).

        A stationary container's centroid cannot drift beyond its own radius,
        so that radius is the physically meaningful bound. Objects with no
        measured extent (no slots, no obb_minor_m) keep the flat tolerance;
        widening those is what would let a utensil dropped BESIDE the tray
        bind to the tray's own anchor.

        Never shrinks below the flat value.
        """
        obb = container_obb_xy(plan_det) if plan_det is not None else None
        if obb is None:
            return self._tol
        _, half_a, half_b, _ = obb
        return max(self._tol, float(max(half_a, half_b)))


# fusion


def fuse_votes(
    votes: Sequence[CameraVote],
    require_two_views: bool = False,
    silent_cameras: Sequence[str] = (),
) -> Tuple[str, str]:
    """ENPIRE fusion. Returns (status, reason).

    ``silent_cameras`` captured but resolved nothing, so they emit no vote.
    They are named in the reason as ``silent`` rather than simply omitted:
    with ``require_two_views`` off a lone camera carries the whole verdict,
    and "birdview was unplugged" must not read the same as "birdview agreed".
    """
    by_cam: Dict[str, List[CameraVote]] = {}
    for v in votes:
        by_cam.setdefault(v.camera, []).append(v)
    if not by_cam:
        return UNVERIFIED, "no camera produced a vote"

    verdicts = {cam: camera_verdict(vs) for cam, vs in by_cam.items()}
    labelled = dict(verdicts)
    for cam in silent_cameras:
        labelled.setdefault(cam, "silent")
    summary = ", ".join(f"{c}={labelled[c]}" for c in sorted(labelled))

    failing = [c for c, v in verdicts.items() if v == FAIL]
    if failing:
        return FAIL, f"camera(s) {sorted(failing)} vote fail ({summary})"

    passing = [c for c, v in verdicts.items() if v == PASS]
    # A 2D pass corroborates only: a fused pass needs a 3D pass vote.
    passing_3d = [c for c in passing if any(v.mode == "3d" and v.vote == PASS for v in by_cam[c])]
    if not passing_3d:
        return UNVERIFIED, f"no 3d pass vote ({summary})"
    if require_two_views and len(passing) < 2:
        return UNVERIFIED, f"require_two_views but only {len(passing)} pass ({summary})"
    return PASS, f"{len(passing)} camera(s) pass ({summary})"


class SuccessVerifier:
    """Produces the single authoritative verdict for a run."""

    def __init__(self, executor=None, pipeline=None, config: Optional[VerifyConfig] = None):
        self.executor = executor
        self.pipeline = pipeline if pipeline is not None else getattr(executor, "_pipeline", None)
        self.config = config or VerifyConfig.from_pipeline(self.pipeline)
        # How much perception this verification may ask for. See verify_scope
        # for the four cuts and the evidence each one trades away.
        self.scope = verify_scope.ScopeConfig.from_pipeline(self.pipeline)
        # Set by _verify: the capture timestamp the detections were computed
        # from, so the memo can only serve detections of the same frames.
        self._capture_t = 0.0

    # spec resolution (planner -> task yaml -> derived)

    def resolve_spec(self, score: Any, actions: Sequence[dict]) -> Optional[VerifySpec]:
        detections = getattr(self.executor, "detection_map", None) or {}
        resolver = None
        if isinstance(detections, LabelResolvingDetectionMap):
            resolver = lambda lab: lab if detections.get(lab) is not None else None  # noqa: E731

        for block, source in (
            ((score or {}).get("verify") if isinstance(score, dict) else None, "planner"),
            (getattr(self.executor, "_task_verify_block", None), "task-yaml"),
        ):
            if not block:
                continue
            try:
                spec = parse_verify_block(
                    block,
                    label_resolver=resolver,
                    default_min_conf=self.config.min_conf,
                    source=source,
                )
                logger.info("Verify spec from %s: %s", source, spec.describe())
                return self._apply_config_floor(spec)
            except VerifyBlockError as exc:
                logger.warning("Rejecting %s verify block: %s", source, exc)

        spec = derive_default_verify(actions or [], detections, min_conf=self.config.min_conf)
        if spec is None:
            logger.warning("No verify predicate derivable from the tree")
            return None
        logger.info("Verify spec derived from tree: %s", spec.describe())
        return self._apply_config_floor(spec)

    def _apply_config_floor(self, spec: VerifySpec) -> VerifySpec:
        """Config is a floor, never a ceiling: it can only tighten a spec."""
        if self.config.require_two_views:
            spec.require_two_views = True
        spec.min_conf = max(spec.min_conf, self.config.min_conf)
        return spec

    # vision

    def _capture(self) -> Dict[str, dict]:
        if self.pipeline is None:
            return {}
        captures, stamp, _reused = verify_scope.captures_for_verify(self.pipeline, self.scope)
        self._capture_t = stamp
        return captures

    def _pose_only_keys(self, anchored: Set[str]) -> Set[str]:
        """Plan-anchored references whose prompt the scope may drop entirely.

        Only when the 2D containment fallback is OFF. That fallback is the one
        consumer of a reference's fresh MASK, and a reference bound from plan
        time alone has none. With the fallback on, the prompt is still issued
        so the image-space branch keeps its evidence.
        """
        if not (self.scope.enabled and self.scope.anchor_references):
            return set()
        if self.config.fallback_2d_containment <= 1.0:
            logger.info(
                "Verify: not dropping reference prompts (the 2D containment "
                "fallback is on and needs a fresh mask)"
            )
            return set()
        return set(anchored or ())

    def _detect_per_camera(
        self,
        captures: Dict[str, dict],
        labels: Sequence[str],
        multi_bases: Optional[Set[str]] = None,
        pose_only: Optional[Set[str]] = None,
    ) -> Dict[str, List[Any]]:
        """Detect independently in every camera. No cross-camera merge, ever.

        EVERY detection is kept, duplicates included: collapsing repeated
        labels would hide a second instance and make an ambiguous scene look
        unambiguous to the identity binder.

        ``multi_bases`` are promoted to multi-instance prompts, because the
        default top-1-per-prompt would hand the binder one fork in a two-fork
        scene; an ambiguous scene the binder cannot see is ambiguous, so it
        would bind to the survivor and pass on the wrong object. Reference
        labels (containers, surfaces) stay top-1: a phantom second tray only
        costs an abstain, and abstaining on every container is worse.

        ``pose_only`` names references the binder will serve from the plan-time
        map, so their prompt is not issued at all, one SAM3 pass per camera
        saved per reference. See verify_scope.
        """
        plan = verify_scope.plan_prompts(labels, pose_only or (), self.scope, base_label)
        if plan.dropped:
            logger.info(
                "Verify: %d prompt(s) dropped (%s); detecting %s",
                plan.saved,
                plan.reason,
                plan.prompts,
            )
        if not plan.prompts or self.pipeline is None:
            return {}
        multi = {p for p in plan.prompts if p in (multi_bases or set())}
        dets = verify_scope.detect_scoped(
            self.pipeline,
            captures,
            plan.prompts,
            self.scope,
            multi_instance_prompts=multi,
            capture_time=self._capture_t,
        )
        return verify_scope.per_camera(dets)

    @staticmethod
    def _multi_instance_bases(spec: VerifySpec, plan_map: Any, manipulated: Set[str]) -> Set[str]:
        """Base labels the verify re-detection must see EVERY instance of.

        The objects under test (predicate ``obj``, anything the tree grasped)
        and any base the plan already saw twice. Hiding a duplicate of one of
        those is what lets a predicate bind to the wrong physical object.
        """
        out = {base_label(lab) for lab in manipulated}
        for pred in spec.predicates:
            obj = pred.params.get("obj")
            if isinstance(obj, str) and obj:
                out.add(base_label(obj))
        counts: Dict[str, int] = {}
        for key in plan_map or ():
            counts[canon_label(key)] = counts.get(canon_label(key), 0) + 1
        for key in plan_map or ():
            if counts.get(canon_label(key), 0) > 1:
                out.add(base_label(key))
        return {b for b in out if b}

    @staticmethod
    def _plan_anchored_keys(spec: VerifySpec, plan_map: Any, displaced: Set[str]) -> Set[str]:
        """Plan-time detection keys whose pose is the verify-time reference frame.

        A key qualifies only when all three hold:

          * it is used as a REFERENCE (container / surface / base / target),
            never as the ``obj`` under test in any predicate; the object is
            the thing that moved, and it must always be re-detected;
          * the tree did not displace it (``displaced_labels`` covers carrying
            it and shoving it: open_drawer, push, pull, drag, sweep, ...), so
            its plan-time pose is still the truth;
          * the predicate label resolves to exactly one plan-time detection
            (``_plan_key``); an ambiguous label anchors nothing and falls
            through to the ordinary geometric binding, which abstains.
        """
        objects = {
            key
            for key in (_plan_key(plan_map, lab) for lab in spec.object_labels)
            if key is not None
        }
        out: Set[str] = set()
        for lab in spec.reference_labels:
            key = _plan_key(plan_map, lab)
            if key is None or key in objects or key in displaced:
                continue
            out.add(key)
        return out

    @staticmethod
    def _depth_source(captures: Dict[str, dict], pipeline) -> str:
        use_hw = bool(getattr(getattr(pipeline, "config", None), "use_hardware_depth", True))
        kinds = set()
        for cam in captures.values():
            has_depth = (cam or {}).get("depth") is not None
            kinds.add("hardware" if (has_depth and use_hw) else "da3")
        if not kinds:
            return "none"
        return kinds.pop() if len(kinds) == 1 else "mixed"

    # main entry

    def verify(self, score: Any, actions: Sequence[dict]) -> VerifyOutcome:
        t0 = time.time()
        try:
            return self._verify(score, actions)
        except Exception:
            logger.exception("Verification raised; reporting unverified (fail-closed)")
            gates, _ = collect_gates(self.executor)
            return VerifyOutcome(
                status=UNVERIFIED,
                gates=gates,
                reason="verification raised an exception",
            )
        finally:
            logger.info("Verification took %.2fs", time.time() - t0)

    def _verify(self, score: Any, actions: Sequence[dict]) -> VerifyOutcome:
        gates, failures = collect_gates(self.executor)
        if not self.config.enabled:
            return VerifyOutcome(status=UNVERIFIED, gates=gates, reason="verification disabled")
        if failures:
            return VerifyOutcome(
                status=FAIL,
                gates=gates,
                reason=f"gate(s) failed before any capture: {failures}",
            )

        spec = self.resolve_spec(score, actions)
        if spec is None:
            return VerifyOutcome(
                status=self.config.on_missing, gates=gates, reason="no predicate available"
            )
        names = spec.describe()

        captures = self._capture()
        if not captures:
            return VerifyOutcome(
                status=UNVERIFIED, predicates=names, gates=gates, reason="no camera captured"
            )

        # Which labels the tree moved: the only ones whose plan-time position
        # is allowed to be stale when the binder re-identifies them.
        plan_map = getattr(self.executor, "detection_map", None) or {}
        manipulated = {
            key
            for key in (_plan_key(plan_map, lab) for lab in displaced_labels(actions))
            if key is not None
        }
        anchored = self._plan_anchored_keys(spec, plan_map, manipulated)
        if anchored:
            logger.info("Verify: reference frame(s) taken from plan time: %s", sorted(anchored))
        pose_only = self._pose_only_keys(anchored)

        # Cameras the labels under test cannot project into can only abstain,
        # so they are not detected on at all. Falls back to every camera when
        # the plan-time geometry does not support the test.
        anchor_keys = {_plan_key(plan_map, lab) for lab in spec.labels}
        anchors = [
            detection_xyz(dict.get(plan_map, key)) for key in anchor_keys if key is not None
        ]
        keep, dropped_cams = verify_scope.plan_cameras(captures, anchors, self.scope)
        if dropped_cams:
            logger.info(
                "Verify: %s cannot see %s; not detecting there",
                sorted(dropped_cams),
                spec.labels,
            )
        captures = {c: captures[c] for c in keep}
        depth_source = self._depth_source(captures, self.pipeline)

        per_cam = self._detect_per_camera(
            captures,
            spec.labels,
            self._multi_instance_bases(spec, plan_map, manipulated),
            pose_only=pose_only,
        )

        witness = getattr(self.executor, "_release_witness", None)
        held = witness.held_after if isinstance(witness, ReleaseWitness) else None
        if held is None:
            held = getattr(self.executor, "_holding", None)

        cfg = EvalConfig(
            min_conf=spec.min_conf,
            fallback_2d_containment=self.config.fallback_2d_containment,
            inside_xy_margin_m=self.config.inside_xy_margin_m,
            inside_z_tol_m=self.config.inside_z_tol_m,
            z_tol_scale=2.0 if depth_source in ("da3", "mixed") else 1.0,
        )

        cams = [c for c in CAMERA_ORDER if c in captures]
        cams += [c for c in captures if c not in CAMERA_ORDER]
        votes: List[CameraVote] = []
        silent: List[str] = []
        for cam in cams:
            cam_dets = per_cam.get(cam)
            if not cam_dets:
                logger.info("Verify: %s detected nothing, contributes no vote", cam)
                silent.append(cam)
                continue
            binder = IdentityBinder(
                plan_map,
                cam_dets,
                manipulated,
                plan_anchored=anchored,
                pose_only=pose_only,
            )
            view = SceneView(
                camera=cam,
                lookup=binder.lookup,
                held=None if held is None else bool(held),
                depth_source=depth_source,
                explain=binder.explain,
            )
            for pred in spec.predicates:
                vote = evaluate_predicate(pred, view, cfg)
                votes.append(vote)
                logger.info(
                    "Verify [%s] %s -> %s (%s, conf=%.2f) %s",
                    cam,
                    vote.predicate,
                    vote.vote,
                    vote.mode,
                    vote.confidence,
                    vote.detail,
                )

        status, reason = fuse_votes(
            votes, require_two_views=spec.require_two_views, silent_cameras=silent
        )
        status, reason = self._maybe_occlusion_escape(status, reason, spec, votes, gates)

        outcome = VerifyOutcome(
            status=status,
            predicates=names,
            votes=votes,
            gates=gates,
            reason=reason,
            depth_source=depth_source,
        )
        logger.info("VERIFY %s: %s", status.upper(), reason)
        return outcome

    def _maybe_occlusion_escape(self, status, reason, spec, votes, gates):
        """``occlusion_ok`` annotates an all-abstain result. It cannot pass it.

        The release witness proves the jaws opened inside the container's
        footprint and let go. It does NOT prove the object stayed: a bounce out
        of the tray leaves the object somewhere the cameras did not resolve it,
        which produces exactly the same all-abstain vision result as an object
        genuinely swallowed by an opaque bin. Upgrading that to ``pass`` would
        manufacture success from no visual evidence.

        So the witness is recorded in the reason and the status stays
        ``unverified``. Proving placement into an opaque container needs
        evidence the object is still there (a second viewpoint into the bin, a
        wrist look-down, tactile/weight), not more proprioception.
        """
        if status != UNVERIFIED or not votes:
            return status, reason
        inside_preds = [p for p in spec.predicates if p.pred == "inside"]
        if not inside_preds or not all(bool(p.params.get("occlusion_ok")) for p in inside_preds):
            return status, reason
        if any(v.vote == FAIL for v in votes) or not all(
            v.vote == ABSTAIN for v in votes if v.mode != "none"
        ):
            return status, reason
        witness = getattr(self.executor, "_release_witness", None)
        if not isinstance(witness, ReleaseWitness):
            return status, reason
        dz = witness.tcp_dz_to_container
        if gates.get("release") and witness.inside_region and dz is not None and dz <= 0.15:
            return status, (
                f"{reason}; occlusion_ok: release witness was inside the container "
                f"(dz={dz:.3f}m) but proprioception cannot prove the object stayed"
            )
        return status, reason


def task_success(outcome: Optional[VerifyOutcome]) -> bool:
    """The ONLY definition of task success. A missing outcome is not success."""
    return outcome is not None and outcome.status == PASS


__all__ = [
    "CAMERA_ORDER",
    "STATIC_MATCH_TOL_M",
    "IdentityBinder",
    "task_success",
    "ReleaseWitness",
    "SuccessVerifier",
    "VerifyConfig",
    "begin_release_witness",
    "collect_gates",
    "finish_release_witness",
    "fuse_votes",
    "note_transport_drop",
    "record_grasp_verdict",
    "reset_run_state",
    "set_gate",
]


if __name__ == "__main__":
    # Self-check: the G3 release witness reads a non-Robotiq jaw source.
    # Run: PYTHONPATH=src python -m spark_real.control.success_verifier
    from spark_real.control.score_executor import ScoreExecutor

    class _FrankaHand:
        GRIPPER_TYPE = "franka_hand"
        robot_family = "franka"

        def __init__(self, pos, grasped, open_to, grasped_after):
            self.pos, self.grasped = pos, grasped
            self._open_to, self._grasped_after = open_to, grasped_after

        def get_tcp_pose(self):
            return np.array([0.4, 0.0, 0.2, 3.14159, 0.0, 0.0])

        def get_gripper_position(self):  # 0..255 like FrankaDriverBase
            return self.pos

        def is_object_detected(self):
            return self.grasped

        def open_gripper(self):
            self.pos, self.grasped = self._open_to, self._grasped_after

    def _release(robot):
        ex = ScoreExecutor(robot, detection_map={}, velocity=0.2)
        ex.RELEASE_CONFIRM_TIMEOUT_S = 0.1
        ex._holding = True
        state = begin_release_witness(ex)
        ex.robot.open_gripper()
        ex._holding = False
        return ex, finish_release_witness(ex, state, "bowl")

    ex, w = _release(_FrankaHand(180.0, True, 0.0, False))
    assert w.released is True and collect_gates(ex)[0]["release"] is True, w
    ex, w = _release(_FrankaHand(180.0, True, 180.0, True))
    assert w.released is False and collect_gates(ex)[0]["release"] is False, w
    ex, w = _release(_FrankaHand(180.0, True, 180.0, False))
    assert w.released is False, w  # no travel, flag clear: still not a release
    broken = _FrankaHand(180.0, True, 0.0, False)
    broken.get_gripper_position = lambda: 1 / 0
    ex, w = _release(broken)
    assert w.released is None and "release" not in _gates(ex), w
    broken.grasped = True
    assert ex._grip_intact() is True  # flag says held, position unreadable

    class _NoJaws:
        GRIPPER_TYPE = "ssg48"

        def get_tcp_pose(self):
            return np.zeros(6)

    assert ScoreExecutor(_NoJaws(), detection_map={}, velocity=0.2)._grip_intact() is True
    closed = ScoreExecutor(_FrankaHand(255.0, False, 0.0, False), detection_map={}, velocity=0.2)
    assert closed._grip_intact() is False  # jaws at the stop, flag clear: dropped
    print("success_verifier self-check OK")

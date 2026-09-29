"""
Gripper-telemetry grasp-outcome classification.

Sim analog of the Robotiq gObj object-detection flag: after the grasp
primitive commands a close, the Franka finger-joint aperture trajectory is
sampled every sim step and classified into a discrete
:class:`GraspOutcome`.  This is the Watchdog pattern from "A Physical
Agentic Loop for Language-Guided Grasping" (arXiv 2604.07395): noisy
gripper telemetry -> typed outcome enum, milliseconds after closure,
BEFORE any camera frame is needed.  The executor consumes the outcome as
the first, instant vote of its post-grasp verify; camera verification
remains the second vote.

Pure classifier - no MuJoCo / env dependency - so it unit-tests on
synthetic aperture traces.  The two small MuJoCo helpers at the bottom
(:func:`find_finger_qpos_addrs`, :func:`read_aperture`) are the only
sim-touching pieces and degrade to no-ops when the joints can't be found.
"""
from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
from typing import Optional, Sequence

import numpy as np

try:  # only needed by the two sim helpers, not the classifier
    import mujoco
except Exception:  # pragma: no cover - classifier stays importable
    mujoco = None  # type: ignore[assignment]


__all__ = [
    'GraspOutcome',
    'GraspClassification',
    'classify_grasp_outcome',
    'find_finger_qpos_addrs',
    'read_aperture',
    'FRANKA_OPEN_APERTURE',
]


# Franka Panda: two mirrored finger joints, |q1| + |q2| ~ 0.08 m fully open.
FRANKA_OPEN_APERTURE = 0.08


# Shared outcome vocabulary: the enum lives in spark_real (the real
# pipeline's endpoint classifier uses it too), so trial metadata carries the
# same values on both embodiments. Semantics for the sim's trace classifier:
#
# * ``SECURED``     - fingers stopped early on an object and the aperture
#                     held stable through the settle window.
# * ``EMPTY_CLOSE`` - fingers reached the commanded (closed) width:
#                     nothing between the jaws (gObj == 3 analog).
# * ``SLIP``        - fingers initially stopped on an object (contact
#                     plateau) but the aperture then collapsed.
# * ``UNKNOWN``     - telemetry unavailable or inconclusive.  Never
#                     triggers a retry.
from spark_real.control.grasp_outcome import GraspOutcome  # noqa: E402


@dataclass
class GraspClassification:
    """
    Classifier output: outcome + the scalars that produced it.
    """
    outcome: GraspOutcome
    final_aperture: float
    plateau_aperture: float
    commanded_aperture: float
    n_samples: int

    def to_meta(self) -> dict:
        """JSON-safe dict for trial_meta."""
        return {
            'outcome': self.outcome.value,
            'final_aperture_m': round(float(self.final_aperture), 5),
            'plateau_aperture_m': round(float(self.plateau_aperture), 5),
            'commanded_aperture_m': round(float(self.commanded_aperture), 5),
            'n_samples': int(self.n_samples),
        }


def _contact_plateau(trace: np.ndarray, *, start_idx: int,
                       window: int, eps: float) -> float:
    """
    Aperture at the FIRST stable window after closing began.

    Scans forward from ``start_idx`` for the first ``window`` consecutive
    samples whose peak-to-peak spread is below ``eps`` - that is where the
    fingers first stopped moving (either on an object or fully closed).
    Falls back to the median of the second half when no stable window
    exists (e.g. trace shorter than ``window``).
    """
    n = len(trace)
    if n >= start_idx + window:
        for i in range(start_idx, n - window + 1):
            seg = trace[i:i + window]
            if float(seg.max() - seg.min()) < eps:
                return float(np.median(seg))
    half = trace[max(0, n // 2):]
    return float(np.median(half)) if len(half) else float(trace[-1])


def classify_grasp_outcome(trace: Sequence[float], *,
                             commanded_aperture: float = 0.0,
                             empty_threshold: float = 0.008,
                             slip_drop: float = 0.006,
                             settle_frac: float = 0.15,
                             stability_window: int = 10,
                             stability_eps: float = 5e-4,
                             ) -> GraspClassification:
    """
    Classify a finger-aperture trajectory into a :class:`GraspOutcome`.

    Parameters
    ----------
    trace:
        Aperture samples (meters, jaw opening) collected during the close
        command and the trailing settle window, in time order.
    commanded_aperture:
        The width the close command drives toward (0.0 = fully closed).
        ``reached ~= commanded`` means the jaws met nothing.
    empty_threshold:
        Reached-vs-commanded margin below which the close counts as empty
        (8 mm default - just above Franka finger-pad compliance).
    slip_drop:
        Aperture decay from the contact plateau to the settle window that
        flags a slip (object escaping while the jaws keep squeezing).
    settle_frac:
        Trailing fraction of the trace used as the settle window for the
        final-aperture estimate.
    stability_window / stability_eps:
        First window of this many consecutive samples with peak-to-peak
        spread under ``eps`` defines the contact plateau.

    Decision table (final = settle-window median, plateau = first stable
    aperture after closing began):

    ==============================  ==================  ============
    condition                       plateau             outcome
    ==============================  ==================  ============
    no samples / fingers unmoved    -                   UNKNOWN
    final <= commanded + empty_thr  plateau high        SLIP
    final <= commanded + empty_thr  plateau low too     EMPTY_CLOSE
    final > empty_thr               plateau-final>drop  SLIP
    final > empty_thr               stable              SECURED
    ==============================  ==================  ============
    """
    arr = np.asarray(list(trace), dtype=float)
    n = len(arr)
    if n == 0:
        return GraspClassification(GraspOutcome.UNKNOWN, float('nan'),
                                     float('nan'), commanded_aperture, 0)

    settle_n = max(1, int(round(settle_frac * n)))
    final = float(np.median(arr[-settle_n:]))

    # Closing must actually have begun: find the first sample where the
    # aperture dropped noticeably below the initial width.  If it never
    # does, the fingers never moved -> inconclusive.
    start_width = float(arr[0])
    moved = np.nonzero(arr <= start_width - 0.005)[0]
    if len(moved) == 0:
        return GraspClassification(GraspOutcome.UNKNOWN, final,
                                     start_width, commanded_aperture, n)
    start_idx = int(moved[0])

    plateau = _contact_plateau(arr, start_idx=start_idx,
                                 window=stability_window, eps=stability_eps)

    reached_margin = final - commanded_aperture
    if reached_margin <= empty_threshold:
        # Jaws effectively reached the commanded width.  If they had first
        # stopped on something (high plateau) the object slipped out;
        # otherwise they closed on air.
        if plateau - final > slip_drop and plateau > empty_threshold:
            outcome = GraspOutcome.SLIP
        else:
            outcome = GraspOutcome.EMPTY_CLOSE
    elif plateau - final > slip_drop:
        outcome = GraspOutcome.SLIP
    else:
        outcome = GraspOutcome.SECURED

    return GraspClassification(outcome, final, plateau,
                                 commanded_aperture, n)


# MuJoCo helpers (executor-side sampling)

def find_finger_qpos_addrs(model) -> list[int]:
    """
    qpos addresses of the two Franka gripper finger joints.

    Mirrors the joint scan in ``primitives.simple._gripper_finger_widths``
    but returns the addresses once so the executor can cache them and
    sample the aperture every sim step at zero lookup cost.  Returns an
    empty list when the joints cannot be located (telemetry degrades to
    UNKNOWN, never blocks the grasp).
    """
    if mujoco is None or model is None:
        return []
    addrs: list[int] = []
    fallback: list[int] = []
    try:
        for jid in range(model.njnt):
            jn = (mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_JOINT, jid)
                  or '').lower()
            if 'finger_joint' in jn and 'gripper' in jn:
                addrs.append(int(model.jnt_qposadr[jid]))
            elif 'finger' in jn and len(fallback) < 2:
                fallback.append(int(model.jnt_qposadr[jid]))
            if len(addrs) == 2:
                return addrs
    except Exception:
        return []
    return addrs if len(addrs) == 2 else (fallback if len(fallback) == 2 else [])


def read_aperture(data, addrs: Sequence[int]) -> Optional[float]:
    """
    Jaw opening in meters: ``|q1| + |q2|`` (mirrored Franka fingers).
    """
    if not addrs or len(addrs) < 2:
        return None
    try:
        return float(abs(data.qpos[addrs[0]]) + abs(data.qpos[addrs[1]]))
    except Exception:
        return None

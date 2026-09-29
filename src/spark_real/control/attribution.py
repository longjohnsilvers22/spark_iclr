"""
Lowest-responsible-layer recovery attribution (ANCHOR pattern,
arXiv 2604.25323).

When a post-primitive / post-episode verify fails, pick the CHEAPEST layer
that can plausibly repair it instead of escalating blindly:

* ``PERCEPTION`` - the target's detection is missing / low-confidence /
  moved since binding.  Action: re-detect and re-bind, same plan,
  no replan.
* ``EXECUTION``  - detections are stable but the predicate still failed:
  the motion itself is at fault.  Action: local retry of the failed
  subtree.
* ``PLAN``       - local retries exhausted, or the failure is semantic
  (wrong object / wrong goal).  Action: tier-3 replan.

Pure decision function, shared by the sim recovery loop
(``spark_bench.fair.execution``) and the real executor's recovery seams
(``execution_recovery`` / ``executor_verify``). On the real robot PLAN means
"stop and surface to the operator", never an autonomous LLM replan.
"""
from __future__ import annotations

from enum import Enum
from typing import Any, Mapping, Optional


__all__ = ['Layer', 'attribute_failure']


class Layer(str, Enum):
    PERCEPTION = 'perception'
    EXECUTION = 'execution'
    PLAN = 'plan'


def _diff_target_status(scene_diff: Any) -> Optional[str]:
    """
    Accept a SceneDiff dataclass, its ``to_meta()`` dict, or None.
    """
    if scene_diff is None:
        return None
    status = getattr(scene_diff, 'target_status', None)
    if status is None and isinstance(scene_diff, Mapping):
        status = scene_diff.get('target_status')
    return status


def attribute_failure(verify_result: Mapping[str, Any],
                        scene_diff: Any,
                        retry_count: int,
                        *,
                        max_local_retries: int = 1,
                        low_confidence: float = 0.10) -> Layer:
    """
    Attribute a verification failure to the lowest responsible layer.

    Parameters
    ----------
    verify_result:
        Dict describing the failed verify.  Recognized keys (all
        optional):

        * ``semantic_failure`` (bool) - the predicate failed for semantic
          reasons (wrong object identity / goal mismatch), not geometry.
        * ``detection_missing`` (bool) - the target label has no current
          detection.
        * ``detection_confidence`` (float) - current confidence of the
          target detection.
        * ``grasp_outcome`` (str) - telemetry outcome ('empty_close' /
          'slip' / 'secured'), informational; an empty close with stable
          detections is an execution failure and falls through to
          EXECUTION naturally.
    scene_diff:
        The most recent SceneDiff for the target (dataclass or
        ``to_meta()`` dict) or None.
    retry_count:
        Execution-layer local retries already attempted for this failure.
    max_local_retries:
        Once ``retry_count`` reaches this, escalate to PLAN (retries
        exhausted).

    Decision table (first match wins)::

        semantic_failure                          -> PLAN
        retry_count >= max_local_retries          -> PLAN
        detection_missing                         -> PERCEPTION
        detection_confidence < low_confidence     -> PERCEPTION
        scene_diff target moved / missing         -> PERCEPTION
        otherwise (stable detection, failed pred) -> EXECUTION
    """
    if verify_result.get('semantic_failure'):
        return Layer.PLAN
    if retry_count >= max_local_retries:
        return Layer.PLAN

    if verify_result.get('detection_missing'):
        return Layer.PERCEPTION
    conf = verify_result.get('detection_confidence')
    if conf is not None and float(conf) < low_confidence:
        return Layer.PERCEPTION
    if _diff_target_status(scene_diff) in ('moved', 'missing'):
        return Layer.PERCEPTION

    return Layer.EXECUTION

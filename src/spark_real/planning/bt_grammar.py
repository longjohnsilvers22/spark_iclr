"""Control-flow grammar for SPARK behaviour trees: selector, fallback, retry.

Leaf module -- stdlib only, imports nothing from ``spark_real`` -- so both the
planner (which must reject a bad node at emit time) and the executor (which
must walk the node at run time) can share ONE definition of the grammar, so
no node type the planner accepts is mis-run by the executor.

Backward compatibility
----------------------
Every stored tree in the BT library is a flat ``sequence``. Nothing here fires
on a tree with no control-flow node: ``normalize`` returns the input object
unchanged, ``has_recovery`` returns False, and the executor's flattener takes
the same path it always did. A grammar addition must never shift a cached
tree's hash, and a flat tree contains no node this module touches.

Bounding recovery
-----------------
Every cap below is a hard ceiling enforced twice: clamped when the plan is
parsed, and re-checked by the executor as it walks. An unbounded retry loop on
a real UR10e is worse than a clean failure -- the arm keeps re-approaching an
object that is not there while the operator watches -- so a plan that asks for
more attempts than the cap gets the cap, not a rejection.
"""

from __future__ import annotations

from typing import Any, Dict, Iterator, Set

# ``selector`` is the classic BT name; ``fallback`` is what most LLM training
# data calls it. Accept both, normalise to one at execution time.
SELECTOR_TYPES = frozenset({"selector", "fallback"})
RETRY_TYPES = frozenset({"retry", "repeat_until_success"})
SEQUENCE_TYPES = frozenset({"sequence"})
CONTROL_FLOW_TYPES = SELECTOR_TYPES | RETRY_TYPES

# --- attempt caps ---------------------------------------------------------
#
# MAX_RETRY_ATTEMPTS = 3. A retry re-runs the SAME motion with the same
# parameters against a freshly re-detected pose. The first repeat buys a new
# detection and a new approach, which is where most single-shot grasp misses
# are recovered. A second repeat occasionally wins on a rolled object. Past
# three, the failure is systematic -- wrong label, unreachable pose, object not
# actually there -- and repeating only wears the gripper and risks nudging the
# object further out of reach. Three attempts of a pick cycle is roughly 90 s
# on the UR10e at the default velocity, which is still inside an operator's
# attention span.
MAX_RETRY_ATTEMPTS = 3
DEFAULT_RETRY_ATTEMPTS = 2

# MAX_SELECTOR_BRANCHES = 3. Branches are DIFFERENT strategies (top-down grasp,
# then a side grasp, then a search-and-retry), not repeats, so more of them is
# more informative than more retries -- but each one is a full acquire or place
# cycle. Three bounds the primary plus two genuinely distinct recoveries, which
# is as many as the skill library actually offers for any one failure.
MAX_SELECTOR_BRANCHES = 3

# MAX_RECOVERY_ATTEMPTS_PER_SCORE = 6. The two caps above are per NODE, and a
# retry(3) nested inside a selector(3) multiplies out to nine attempts. This is
# a run-scoped budget on recovery re-attempts (every selector branch after the
# first, every retry attempt after the first) so nesting cannot multiply past
# it. Six re-attempts is about five minutes of arm time; beyond that a human
# should be looking at the scene, not the robot trying again.
MAX_RECOVERY_ATTEMPTS_PER_SCORE = 6

# Skills the recovery grammar is allowed to name. Every one of these MUST be
# in the SkillRegistry -- a prompt that advertises a skill the dispatcher does
# not have produces a tree that dies at execution, which is strictly worse than
# the flat sequence it replaced. tests/test_planner_recovery_grammar.py checks
# this tuple against the live registry.
RECOVERY_SKILLS = (
    "verify_grasp",
    "verify_placed",
    "search_keypoint",
    "retract_retry",
    "adjust_grip",
    "grasp_perturb",
    "compliant_push",
)


def node_type(node: Any) -> str:
    if not isinstance(node, dict):
        return ""
    return str(node.get("type") or "")


def is_retry(node: Any) -> bool:
    return node_type(node) in RETRY_TYPES


def is_control_flow(node: Any) -> bool:
    return node_type(node) in CONTROL_FLOW_TYPES


def walk(node: Any) -> Iterator[Dict[str, Any]]:
    """Yield every dict node in the tree, depth first, root included."""
    if not isinstance(node, dict):
        return
    yield node
    for child in node.get("children") or []:
        yield from walk(child)


def has_recovery(node: Any) -> bool:
    """True if the tree contains any branch that can react to a failure."""
    return any(is_control_flow(n) for n in walk(node))


def recovery_node_types(node: Any) -> Set[str]:
    """The set of control-flow type names present, as written by the planner."""
    return {node_type(n) for n in walk(node) if is_control_flow(n)}


def clamp_retry_attempts(value: Any) -> int:
    """Coerce a plan's ``max_attempts`` into [1, MAX_RETRY_ATTEMPTS].

    A missing, non-numeric or absurd value becomes the documented default
    rather than an error: the tree is still executable, just bounded.
    """
    try:
        attempts = int(value)
    except (TypeError, ValueError):
        return DEFAULT_RETRY_ATTEMPTS
    if attempts < 1:
        return 1
    return min(attempts, MAX_RETRY_ATTEMPTS)


def control_flow_issues(node: Any, path: str = "tree") -> list:
    """Structural problems with the control-flow nodes of one tree.

    Returns an empty list for any tree without a control-flow node, so a
    stored flat sequence validates exactly as it did before this grammar
    existed.
    """
    issues = []
    if not isinstance(node, dict):
        return issues
    ntype = node_type(node)
    if ntype in CONTROL_FLOW_TYPES:
        children = node.get("children")
        if not isinstance(children, list) or not children:
            issues.append(f"{path}: '{ntype}' node needs a non-empty children list")
        elif ntype in SELECTOR_TYPES and len(children) > MAX_SELECTOR_BRANCHES:
            issues.append(
                f"{path}: '{ntype}' has {len(children)} branches, "
                f"cap is {MAX_SELECTOR_BRANCHES}"
            )
        if ntype in RETRY_TYPES:
            raw = (node.get("params") or {}).get("max_attempts")
            if raw is not None and clamp_retry_attempts(raw) != raw:
                issues.append(
                    f"{path}: 'max_attempts' {raw!r} is outside "
                    f"[1, {MAX_RETRY_ATTEMPTS}]"
                )
    for i, child in enumerate(node.get("children") or []):
        issues.extend(control_flow_issues(child, f"{path}/{i}"))
    return issues


def normalize(node: Any) -> Any:
    """Clamp attempt counts and truncate over-wide selectors, in place.

    Call on a copy. Returns the node for chaining. A tree with no control-flow
    node is untouched.
    """
    if not isinstance(node, dict):
        return node
    ntype = node_type(node)
    children = node.get("children")
    if ntype in SELECTOR_TYPES and isinstance(children, list):
        if len(children) > MAX_SELECTOR_BRANCHES:
            node["children"] = children[:MAX_SELECTOR_BRANCHES]
    if ntype in RETRY_TYPES:
        params = node.get("params")
        if not isinstance(params, dict):
            params = {}
            node["params"] = params
        params["max_attempts"] = clamp_retry_attempts(params.get("max_attempts"))
    for child in node.get("children") or []:
        normalize(child)
    return node

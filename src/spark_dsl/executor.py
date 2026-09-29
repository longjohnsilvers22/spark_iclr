"""
BT runtime for the SPARK BT-DSL.

Executes a parsed BT against the typed :class:`SkillLibrary`:
  1. validate the BT (grammar + per-primitive slot types),
  2. expand macros into base-primitive subtrees at execution time,
  3. dispatch each base primitive to a registered runtime callback.

A macro body may reference caller-supplied slots via string placeholders of
the form ``<slot_name>`` -- these are substituted with the caller's params at
expansion time.  Macros may call macros; cycles are detected and reported.
"""
from __future__ import annotations

import time
from dataclasses import dataclass, field
from typing import Any, Callable, Dict, List, Optional

from .skill_library import DEFAULT_LIBRARY, Primitive, SkillLibrary

RuntimeFn = Callable[[Any, Dict[str, Any]], Any]

_PLACEHOLDER_PREFIX = "<"
_PLACEHOLDER_SUFFIX = ">"
_MAX_EXPANSION_DEPTH = 32


@dataclass
class StepLog:
    primitive: str
    slot_args: Dict[str, Any]
    success: bool
    error: Optional[str] = None
    duration_ms: float = 0.0


@dataclass
class ExecutionResult:
    success: bool
    errors: List[str] = field(default_factory=list)
    steps: List[StepLog] = field(default_factory=list)


class BTExecutor:
    """
    Runtime that validates, expands, and dispatches a typed BT.
    """

    def __init__(self, library: SkillLibrary = DEFAULT_LIBRARY) -> None:
        self.library = library
        self._runtimes: Dict[str, RuntimeFn] = {}

    # -- registration

    def register_runtime(self, primitive_name: str, fn: RuntimeFn) -> None:
        """
        Register a runtime callback for a base primitive.
        """
        prims = self.library.primitives()
        if primitive_name not in prims:
            raise ValueError(f"unknown primitive {primitive_name!r}")
        if prims[primitive_name].is_macro:
            raise ValueError(
                f"cannot register runtime for macro {primitive_name!r}; "
                "macros expand to base primitives"
            )
        self._runtimes[primitive_name] = fn

    def runtimes(self) -> Dict[str, RuntimeFn]:
        return dict(self._runtimes)

    # -- execution

    def execute(self, bt_dict: Dict, env: Any = None,
                context: Optional[Dict[str, Any]] = None) -> ExecutionResult:
        """
        Validate, expand, and dispatch a BT.  Never raises.
        """
        result = ExecutionResult(success=False)
        try:
            _ast, errors = self.library.validate_bt(bt_dict)
        except Exception as exc:  # defensive: validator is third-party
            result.errors.append(f"validator crashed: {exc}")
            return result
        if errors:
            result.errors.extend(errors)
            return result
        tree = bt_dict.get("tree", bt_dict) if isinstance(bt_dict, dict) else None
        try:
            expanded = self._expand(tree, depth=0, stack=())
        except _ExpansionError as exc:
            result.errors.append(str(exc))
            return result
        ok = self._run_node(expanded, env, context or {}, result)
        result.success = ok and not result.errors
        return result

    # -- expansion

    def _expand(self, node: Dict, depth: int, stack: tuple) -> Dict:
        """
        Recursively inline macros into base-primitive subtrees.
        """
        if depth > _MAX_EXPANSION_DEPTH:
            raise _ExpansionError(
                f"macro expansion exceeded depth {_MAX_EXPANSION_DEPTH}"
            )
        ntype = node.get("type")
        if ntype in ("sequence", "selector"):
            new_children = [
                self._expand(c, depth + 1, stack)
                for c in node.get("children", [])
            ]
            return {"type": ntype, "children": new_children}
        prim = self.library.primitives().get(ntype)
        if prim is None or not prim.is_macro:
            # Base primitive: keep as-is (with shallow copy of params).
            return {"type": ntype, "params": dict(node.get("params") or {})}
        if ntype in stack:
            raise _ExpansionError(
                f"macro cycle detected: {' -> '.join(stack + (ntype,))}"
            )
        body = prim.body
        if not isinstance(body, dict):
            raise _ExpansionError(f"macro {ntype!r} has no body subtree")
        caller_params = node.get("params") or {}
        substituted = _substitute(body, caller_params)
        if substituted.get("type") not in ("sequence", "selector"):
            substituted = {"type": "sequence", "children": [substituted]}
        return self._expand(substituted, depth + 1, stack + (ntype,))

    # -- dispatch

    def _run_node(self, node: Dict, env: Any, ctx: Dict[str, Any],
                  result: ExecutionResult) -> bool:
        ntype = node.get("type")
        if ntype == "sequence":
            for child in node.get("children", []):
                if not self._run_node(child, env, ctx, result):
                    return False  # short-circuit: abort on first failure
            return True
        if ntype == "selector":
            for child in node.get("children", []):
                # selector: try each child until one succeeds.  Snapshot the
                # error count so failures from skipped branches don't poison
                # the final result.
                err_before = len(result.errors)
                if self._run_node(child, env, ctx, result):
                    # success: discard errors emitted by failed earlier branches
                    del result.errors[err_before:]
                    return True
            return False
        return self._dispatch_primitive(node, env, ctx, result)

    def expand_macros(self, bt_dict: Dict) -> Dict:
        """
        Public: return a deep copy of ``bt_dict`` with all macro nodes
        recursively expanded to base primitives, for runtimes that need the
        full expanded action sequence as a single payload.

        Wraps ``_expand`` and lifts cycle/depth errors into a clean tuple:
        returns ``(expanded_bt, error_str_or_None)``.
        """
        tree = bt_dict.get("tree", bt_dict) if isinstance(bt_dict, dict) else None
        if not isinstance(tree, dict):
            return ({}, "BT must be a dict (with 'tree' key, or be the tree)")
        try:
            expanded = self._expand(tree, depth=0, stack=tuple())
        except _ExpansionError as exc:
            return ({}, str(exc))
        return ({"tree": expanded}, None)

    def _dispatch_primitive(self, node: Dict, env: Any, ctx: Dict[str, Any],
                            result: ExecutionResult) -> bool:
        ntype = node.get("type")
        slot_args = dict(node.get("params") or {})
        fn = self._runtimes.get(ntype)
        if fn is None:
            err = f"primitive {ntype!r} has no runtime registered"
            result.steps.append(StepLog(ntype, slot_args, False, err, 0.0))
            result.errors.append(err)
            return False
        t0 = time.perf_counter()
        try:
            fn(env, slot_args)
        except Exception as exc:
            dt = (time.perf_counter() - t0) * 1000.0
            err = f"primitive {ntype!r} raised {type(exc).__name__}: {exc}"
            result.steps.append(StepLog(ntype, slot_args, False, err, dt))
            result.errors.append(err)
            return False
        dt = (time.perf_counter() - t0) * 1000.0
        result.steps.append(StepLog(ntype, slot_args, True, None, dt))
        return True


# -- substitution helpers


class _ExpansionError(Exception):
    """
    Raised internally during macro expansion; converted to a clean error.
    """


def _substitute(node: Any, caller_params: Dict[str, Any]) -> Any:
    """
    Recursively substitute ``<slot>`` placeholders with caller params.

    Strings whose entire value is ``<name>`` are replaced with the bound
    value (preserving original type, e.g. int/list).  Strings containing
    ``<name>`` as a substring are replaced via textual interpolation.
    """
    if isinstance(node, dict):
        return {k: _substitute(v, caller_params) for k, v in node.items()}
    if isinstance(node, list):
        return [_substitute(x, caller_params) for x in node]
    if isinstance(node, str):
        return _sub_string(node, caller_params)
    return node


def _sub_string(s: str, caller_params: Dict[str, Any]) -> Any:
    if not (s.startswith(_PLACEHOLDER_PREFIX) and s.endswith(_PLACEHOLDER_SUFFIX)):
        return _interp_partial(s, caller_params)
    inner = s[1:-1]
    if not inner or _PLACEHOLDER_PREFIX in inner or _PLACEHOLDER_SUFFIX in inner:
        return _interp_partial(s, caller_params)
    key = _resolve_key(inner, caller_params)
    if key is None:
        return s  # unbound placeholder: leave verbatim (validator will catch)
    return caller_params[key]


def _interp_partial(s: str, caller_params: Dict[str, Any]) -> str:
    """
    Substitute ``<name>`` substrings inside a longer string.
    """
    out: List[str] = []
    i = 0
    while i < len(s):
        if s[i] == _PLACEHOLDER_PREFIX:
            j = s.find(_PLACEHOLDER_SUFFIX, i + 1)
            if j != -1:
                inner = s[i + 1:j]
                key = _resolve_key(inner, caller_params)
                if key is not None:
                    out.append(str(caller_params[key]))
                    i = j + 1
                    continue
        out.append(s[i])
        i += 1
    return "".join(out)


def _resolve_key(inner: str, caller_params: Dict[str, Any]) -> Optional[str]:
    """
    Match a placeholder body to a caller-param key.

    Direct match (``<pick_label>`` -> ``pick_label``) wins.  Otherwise the
    last dotted segment is tried (``<node0.keypoint_label>`` ->
    ``keypoint_label``) so legacy macro bodies from macro_mining still work
    when the caller exposes a single keypoint slot.
    """
    if inner in caller_params:
        return inner
    if "." in inner:
        tail = inner.rsplit(".", 1)[-1]
        if tail in caller_params:
            return tail
    return None


__all__ = ["BTExecutor", "ExecutionResult", "StepLog", "RuntimeFn"]

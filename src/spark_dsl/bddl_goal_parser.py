"""
Parse the (:goal ...) field of a LIBERO BDDL task into atomic predicates.

This is a tiny standalone module - no LIBERO imports - used by the atomic
sub-BT decomposition path (`run_spark_libero_pro_fair --atomic-decomp`).
Given a BDDL file, it returns the list of *unit* goal predicates so a
planner can emit one sub-BT per predicate and execute them sequentially
with re-perception in between.

Predicate forms LIBERO actually uses (verified from the ``libero_goal``
and ``libero_10`` BDDL files):

* ``(On x y)``       - x is resting on/inside surface y
* ``(In x y)``       - x is inside container y
* ``(Open x)``       - joint x is open (drawer / cabinet)
* ``(Close x)``      - joint x is closed
* ``(Turnon x)`` / ``(Turnoff x)``   - switch state
* Nested under ``(And ...)`` / ``(Or ...)`` / ``(Not ...)``
* Sometimes top-level without ``And`` (single predicate).

Compound operators that take multiple predicates as arguments are flattened
into the returned list; their grouping semantics are kept in a ``mode``
field (``'and'`` is the common case; ``'or'`` is rare in LIBERO).  The
caller can decide what to do with non-And modes - the atomic-decomp loop
treats Or as best-effort sequential.
"""

from __future__ import annotations

from collections import namedtuple
from pathlib import Path
from typing import List, Optional, Union
import re

Predicate = namedtuple("Predicate", ["op", "args"])


__all__ = ["Predicate", "parse_bddl_goal", "parse_goal_string",
           "predicate_to_natural_language"]


# S-expression tokenizer and recursive-descent parser

def _tokenize(s: str) -> List[str]:
    """
    Split a parenthesised PDDL-style expression into tokens.
    """
    s = s.replace('(', ' ( ').replace(')', ' ) ')
    return s.split()


def _parse_sexpr(tokens: List[str], i: int = 0):
    """
    Recursive-descent into nested S-expressions.

    Returns ``(node, next_index)`` where ``node`` is either a list (for
    parenthesised forms) or a bare string atom.
    """
    if i >= len(tokens):
        return None, i
    tok = tokens[i]
    if tok == '(':
        node: list = []
        i += 1
        while i < len(tokens) and tokens[i] != ')':
            child, i = _parse_sexpr(tokens, i)
            if child is not None:
                node.append(child)
        if i < len(tokens):  # consume ')'
            i += 1
        return node, i
    if tok == ')':
        return None, i + 1
    return tok, i + 1


# Public: convert an S-expr node tree into a flat list of Predicates

_COMPOUND_OPS = {'and', 'or', 'not'}


def _flatten_node(node) -> List[Predicate]:
    """
    Walk an S-expr node and return the leaf predicates in order.

    ``And`` / ``Or`` are unwrapped; ``Not`` is wrapped into a Predicate
    whose ``op`` is prefixed with ``"Not"`` so downstream callers can
    notice.  Unknown compound forms fall through as-is.
    """
    if not isinstance(node, list) or len(node) == 0:
        return []
    head = node[0]
    if not isinstance(head, str):
        # Non-atom head - nothing reasonable to do; recurse on children.
        out: List[Predicate] = []
        for c in node:
            out.extend(_flatten_node(c))
        return out
    op_low = head.lower()
    if op_low in _COMPOUND_OPS:
        # Unwrap And/Or by walking children; Not flips a flag on the inner.
        if op_low == 'not':
            inner = _flatten_node(node[1]) if len(node) > 1 else []
            return [Predicate(op=f"Not{p.op}", args=p.args) for p in inner]
        out = []
        for c in node[1:]:
            out.extend(_flatten_node(c))
        return out
    # Leaf predicate: head is the op, the rest are args.
    args = [a for a in node[1:] if isinstance(a, str)]
    return [Predicate(op=head, args=args)]


def parse_goal_string(goal_text: str) -> List[Predicate]:
    """
    Parse the contents of a BDDL ``(:goal ...)`` block.

    ``goal_text`` is the raw bracketed string (with or without the surrounding
    outer paren that comes after ``:goal``).
    """
    # Defensive: strip any trailing whitespace and ensure outer paren present
    s = goal_text.strip()
    if not s.startswith('('):
        s = '(' + s + ')'
    tokens = _tokenize(s)
    if not tokens:
        return []
    tree, _ = _parse_sexpr(tokens, 0)
    if tree is None:
        return []
    return _flatten_node(tree)


# Public: full-BDDL-file entry point

def _extract_goal_text(content: str) -> Optional[str]:
    """
    Pull out the body of ``(:goal ...)`` with balanced paren matching.
    """
    # Find ":goal" - search through nested parens to find the matching close.
    m = re.search(r'\(:goal\s', content)
    if not m:
        return None
    start = m.end()
    depth = 1  # already inside the outer goal-paren
    i = start
    while i < len(content):
        ch = content[i]
        if ch == '(':
            depth += 1
        elif ch == ')':
            depth -= 1
            if depth == 0:
                return content[start:i]
        i += 1
    return content[start:]  # malformed BDDL: return the remainder


def parse_bddl_goal(bddl_path: Union[str, Path]) -> List[Predicate]:
    """
    Read a BDDL file and return its goal predicates as a flat list.

    Empty list on any failure (missing file, no ``:goal`` block, etc.).
    """
    try:
        with open(bddl_path, 'r') as f:
            content = f.read()
    except OSError:
        return []
    goal_text = _extract_goal_text(content)
    if not goal_text:
        return []
    return parse_goal_string(goal_text)


# Public: render a Predicate to a natural-language sub-instruction

def _clean_arg(arg: str) -> str:
    """
    Strip instance suffixes (``_1``) and region suffixes (``_top_region``).

    Mirrors ``spark_bench.libero_pro.bddl._strip_instance`` but kept local
    so this module has no cross-imports.
    """
    if not arg:
        return arg
    s = arg
    # Strip ``_<digit>`` followed by optional region/side suffix.
    m = re.match(r'^(.+?)_(\d+)(?:_(.+))?$', s)
    if m:
        base = m.group(1).replace('_', ' ')
        region = m.group(3)
        if region:
            return f"{base} {region.replace('_', ' ')}"
        return base
    return s.replace('_', ' ')


def predicate_to_natural_language(p: Predicate) -> str:
    """
    Render a Predicate as a one-line sub-instruction for the LLM.
    """
    op = p.op
    args = [_clean_arg(a) for a in p.args]
    if op in ('On',):
        if len(args) >= 2:
            return f"Place the {args[0]} on the {args[1]}"
    if op == 'In':
        if len(args) >= 2:
            return f"Place the {args[0]} in the {args[1]}"
    if op == 'Open':
        if args:
            return f"Open the {args[0]}"
    if op == 'Close':
        if args:
            return f"Close the {args[0]}"
    if op in ('Turnon', 'TurnOn'):
        if args:
            return f"Turn on the {args[0]}"
    if op in ('Turnoff', 'TurnOff'):
        if args:
            return f"Turn off the {args[0]}"
    if op.startswith('Not') and args:
        return f"Make sure the {args[0]} is not {op[3:].lower()}"
    # Fallback: pass through verbatim.
    return f"{op} " + " ".join(args)

"""
Typed skill registry for the SPARK BT-DSL.

Holds the production base primitives + dynamically-registered macros, exposes
a prompt fragment, and validates BT dicts against ``grammar.lark`` plus
per-primitive slot type signatures.

Production primitive set (the actions the benchmark runners dispatch):
    move_to_keypoint, grasp, release, move_relative,
    push_object, turn_knob, rotate, wait, wipe, insert

Validation: render BT dict to a canonical S-expression, parse with Lark, then
walk the dict checking slot presence and types.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

from lark import Lark, LarkError

# slot name -> (type or tuple, required, doc)
SlotSig = Dict[str, Tuple[Any, bool, str]]


@dataclass
class Primitive:
    name: str
    slots: SlotSig
    output_type: str
    doc: str
    is_macro: bool = False
    body: Optional[Dict] = None  # macro expansion (BT subtree)


_NUM = (int, float)
_STR = (str,)
_LIST = list

_BASE_PRIMITIVES: Dict[str, Primitive] = {
    "move_to_keypoint": Primitive("move_to_keypoint", {
        "keypoint_label": (_STR, True, "perception keypoint"),
        "offset_x": (_NUM, False, "x offset (m)"),
        "offset_y": (_NUM, False, "y offset (m)"),
        "offset_z": (_NUM, False, "z offset (m)"),
    }, "ee_pose", "Move EE to a labeled 3D keypoint with optional offset."),
    "grasp": Primitive("grasp", {
        "force": (_NUM, False, "grip force 0-255"),
    }, "gripper_state", "Close the gripper to grasp the held object."),
    "release": Primitive("release", {
    }, "gripper_state", "Open the gripper to release."),
    "move_relative": Primitive("move_relative", {
        "dx": (_NUM, False, "delta x (m)"),
        "dy": (_NUM, False, "delta y (m)"),
        "dz": (_NUM, False, "delta z (m)"),
    }, "ee_pose", "Move EE by a relative offset in world frame."),
    "push_object": Primitive("push_object", {
        "keypoint_label": (_STR, True, "SAM3 label of the SOURCE object to push"),
        "target_label": (_STR, False, "OPTIONAL SAM3 label of the TARGET region/object the source should be pushed toward. When provided, push direction and distance are computed from source->target 3D positions (preferred for tasks like 'push the plate to the front of the stove'). Falls back to push_direction/push_distance if omitted."),
        "push_direction": (_LIST, False, "[x,y,z] unit vector, default [0,-1,0]. Ignored when target_label is provided and detected."),
        "push_distance": (_NUM, False, "distance (m), default 0.15. Ignored when target_label is provided and detected."),
    }, "ee_pose", "Push an object toward a target region. Prefer emitting target_label (SAM3 label of the destination) so direction/distance are inferred from perception; fall back to push_direction/push_distance only when no target object is detectable."),
    "turn_knob": Primitive("turn_knob", {
        "joint_name": (_STR, False, "knob joint name (or empty for auto)"),
        "keypoint_label": (_STR, False, "SAM3 label of the knob (optional; defaults to 'stove knob')"),
    }, "ee_pose", "Rotate a stove knob via arc motion. Thin alias over `rotate` - emit `rotate` instead for non-stove knobs."),
    "rotate": Primitive("rotate", {
        "keypoint_label": (_STR, True, "SAM3 detection label of the object to rotate (e.g. 'stove knob', 'oven dial', 'faucet handle')"),
        "angle_rad": (_NUM, False, "wrist rotation magnitude in radians, default 1.0"),
        "direction": (_STR, False, "'cw' (default; on / screwdrive in) or 'ccw' (off / unscrew)"),
        "axis": (_STR, False, "rotation axis; only 'wrist' is implemented (default)"),
    }, "ee_pose", "Generic rotate: descend onto the detected object, close gripper around the stem, then rotate the wrist joint by ``angle_rad`` (sign per ``direction`` - auto-flipped for 'turn off' instructions). Use for knobs, dials, valves, screws."),
    "wait": Primitive("wait", {
        "duration": (_NUM, False, "wait duration (s)"),
    }, "none", "Hold position for a duration."),
    "wipe": Primitive("wipe", {
        "keypoint_label": (_STR, True, "dirt region keypoint"),
        "sweep_width": (_NUM, False, "sweep width (m), default 0.55"),
        "sweep_length": (_NUM, False, "sweep length (m), default 0.65"),
    }, "ee_pose", "Snake-raster sweep over a dirt region; closed-loop until clean."),
    "insert": Primitive("insert", {
        "keypoint_label": (_STR, True, "insertion target keypoint"),
        "offset_x": (_NUM, False, "x offset (m)"),
        "offset_y": (_NUM, False, "y offset (m)"),
        "offset_z": (_NUM, False, "z offset (m)"),
    }, "ee_pose", "Visual-servo insertion: position above target, refine, drop."),
}


class SkillLibrary:
    """
    Registry of base primitives + dynamically registered macros.
    """

    _GRAMMAR_PATH = Path(__file__).with_name("grammar.lark")

    def __init__(self) -> None:
        self._primitives: Dict[str, Primitive] = dict(_BASE_PRIMITIVES)
        self._parser = Lark(self._GRAMMAR_PATH.read_text(), start="start")

    # -- registration

    def register_macro(self, name: str, signature: SlotSig,
                       body_subtree: Dict) -> None:
        """
        Register a macro discovered later by macro_mining.
        """
        if name in self._primitives:
            raise ValueError(f"primitive {name!r} already registered")
        if not name.replace("_", "").isalnum() or not name.islower():
            raise ValueError(f"macro name {name!r} must be lowercase_snake")
        self._primitives[name] = Primitive(
            name=name, slots=signature, output_type="macro",
            doc=f"Macro: {len(body_subtree.get('children', []))} actions",
            is_macro=True, body=body_subtree,
        )

    def primitives(self) -> Dict[str, Primitive]:
        return dict(self._primitives)

    # -- prompt fragment

    def available_primitives_for_prompt(self) -> str:
        """
        Render a string fragment listing primitives + macros for the LLM.
        """
        lines = ["Available primitives (typed):"]
        for prim in sorted(self._primitives.values(),
                           key=lambda p: (p.is_macro, p.name)):
            tag = "MACRO " if prim.is_macro else ""
            lines.append(f"- {tag}{prim.name} -> {prim.output_type}")
            lines.append(f"    {prim.doc}")
            for sname, (stype, sreq, sdoc) in prim.slots.items():
                req = "required" if sreq else "optional"
                lines.append(f"    :{sname} ({_type_name(stype)}, {req}) -- {sdoc}")
        return "\n".join(lines)

    # -- validation

    def validate_bt(self, bt_dict: Dict) -> Tuple[Optional[Any], List[str]]:
        """
        Validate a BT dict.  Returns (ast, errors); errors is [] iff valid.
        """
        tree = bt_dict.get("tree", bt_dict) if isinstance(bt_dict, dict) else None
        if not isinstance(tree, dict):
            return None, ["BT must be a dict (with 'tree' key, or be the tree)"]
        if tree.get("type") != "sequence":
            return None, [f"BT root must be type='sequence', got {tree.get('type')!r}"]
        try:
            canonical = self.bt_to_canonical(tree)
        except ValueError as exc:
            return None, [f"Cannot canonicalize BT: {exc}"]
        try:
            ast = self._parser.parse(canonical)
        except LarkError as exc:
            return None, [f"Grammar error: {exc}"]
        errors: List[str] = []
        self._typecheck_node(tree, errors, "tree")
        return (None, errors) if errors else (ast, [])

    # -- internals

    def bt_to_canonical(self, node: Dict) -> str:
        ntype = node.get("type")
        if ntype in ("sequence", "selector"):
            children = node.get("children", [])
            if not children:
                raise ValueError(f"{ntype} must have at least one child")
            inner = " ".join(self.bt_to_canonical(c) for c in children)
            return f"({ntype} {inner})"
        if ntype is None:
            raise ValueError(f"node missing 'type' key: {node!r}")
        params = node.get("params", {}) or {}
        slots = " ".join(f":{k} {_to_canonical(v)}" for k, v in params.items())
        body = (" " + slots) if slots else ""
        return f"(action {ntype}{body})"

    def _typecheck_node(self, node: Dict, errors: List[str], path: str) -> None:
        ntype = node.get("type")
        if ntype in ("sequence", "selector"):
            for i, child in enumerate(node.get("children", [])):
                self._typecheck_node(child, errors, f"{path}.children[{i}]")
            return
        if ntype not in self._primitives:
            errors.append(f"{path}: unknown primitive {ntype!r}")
            return
        prim = self._primitives[ntype]
        params = node.get("params", {}) or {}
        for sname, (_t, sreq, _d) in prim.slots.items():
            if sreq and sname not in params:
                errors.append(f"{path}: {ntype} missing required slot {sname!r}")
        for sname, val in params.items():
            if sname not in prim.slots:
                errors.append(f"{path}: {ntype} has unknown slot {sname!r}")
                continue
            stype, _r, _d = prim.slots[sname]
            if not _check_type(val, stype):
                errors.append(
                    f"{path}: {ntype}.{sname} expected {_type_name(stype)}, "
                    f"got {type(val).__name__}"
                )


# -- helpers


def _check_type(val: Any, expected: Any) -> bool:
    if expected is list:
        return isinstance(val, list)
    if isinstance(expected, tuple):
        if expected == _NUM and isinstance(val, bool):
            return False  # bool is technically int; reject for numeric slots
        return isinstance(val, expected)
    return isinstance(val, expected)


def _type_name(t: Any) -> str:
    if t is list:
        return "list"
    if isinstance(t, tuple):
        return "|".join(x.__name__ for x in t)
    return getattr(t, "__name__", str(t))


def _to_canonical(v: Any) -> str:
    if isinstance(v, bool):
        return "true" if v else "false"
    if v is None:
        return "null"
    if isinstance(v, (int, float)):
        return repr(v)
    if isinstance(v, str):
        return json.dumps(v)
    if isinstance(v, list):
        return "[" + " ".join(_to_canonical(x) for x in v) + "]"
    raise ValueError(f"cannot canonicalize {type(v).__name__}: {v!r}")


DEFAULT_LIBRARY = SkillLibrary()

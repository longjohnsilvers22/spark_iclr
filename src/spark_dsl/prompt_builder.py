"""
Build the Gemini-Flash planning prompt from the typed skill library.

Pure-Python: no LLM calls, no I/O beyond reading the optional ``macros.yaml``.
The :class:`PromptBuilder` loads dynamically-mined macros, registers them on
the supplied :class:`SkillLibrary`, and produces a single cohesive prompt
string that Gemini-Flash can answer with one typed YAML behavior tree.

Free-slot placeholders in macro bodies (``<node0.keypoint_label>`` etc.) are
translated to clean signature names like ``pick_label`` / ``place_label``
based on positional order: the first occurrence of ``keypoint_label`` gets
``pick_label``, the second ``place_label``, the third ``via_label``, etc.
Non-keypoint slots fall back to the bare slot name with a ``_N`` suffix on
collision.
"""

from __future__ import annotations

import copy
import os
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple

import yaml

from .skill_library import SkillLibrary, SlotSig

FewShot = Tuple[str, Dict[str, Any]]


_KEYPOINT_LABEL_NAMES = ["pick_label", "place_label", "via_label",
                         "extra_label_1", "extra_label_2"]


def _clean_signature(free_params: Dict[str, Any]
                     ) -> Tuple[SlotSig, Dict[str, str]]:
    """
    Translate ``nodeI.slot`` keys into a clean ``SlotSig`` + rename map.

    Returns ``(signature, rename_map)`` where ``rename_map`` maps the original
    placeholder body-token ``"nodeI.slot"`` to the clean signature name.
    """
    sorted_keys = sorted(free_params.keys(),
                         key=lambda k: (int(k.split(".")[0][4:]), k))
    keypoint_idx = 0
    used: Dict[str, int] = {}
    sig: SlotSig = {}
    rename: Dict[str, str] = {}
    for raw in sorted_keys:
        _, slot = raw.split(".", 1)
        if slot == "keypoint_label" and keypoint_idx < len(_KEYPOINT_LABEL_NAMES):
            clean = _KEYPOINT_LABEL_NAMES[keypoint_idx]
            keypoint_idx += 1
        else:
            base = slot
            count = used.get(base, 0)
            clean = base if count == 0 else f"{base}_{count + 1}"
            used[base] = count + 1
        meta = free_params[raw] or {}
        example = meta.get("example") if isinstance(meta, dict) else None
        doc = f"free slot (e.g. {example!r})" if example is not None else "free slot"
        sig[clean] = (str, True, doc)
        rename[raw] = clean
    return sig, rename


def _substitute_body(body: Any, rename: Dict[str, str]) -> Any:
    """
    Replace ``<nodeI.slot>`` strings inside a body subtree with the
    cleanly-named placeholder (still wrapped in ``<>``), so the BTExecutor's
    macro-expansion substitution recognizes them at call time.
    """
    if isinstance(body, dict):
        return {k: _substitute_body(v, rename) for k, v in body.items()}
    if isinstance(body, list):
        return [_substitute_body(x, rename) for x in body]
    if isinstance(body, str) and body.startswith("<") and body.endswith(">"):
        token = body[1:-1]
        if token in rename:
            return f"<{rename[token]}>"
        return body
    return body


def _wrap_subtree(body_list: List[Dict[str, Any]]) -> Dict[str, Any]:
    """
    Macro bodies in macros.yaml are flat lists; wrap them as a sequence.
    """
    return {"type": "sequence", "children": body_list}


def _format_fewshot(examples: Sequence[FewShot]) -> str:
    chunks: List[str] = []
    for i, (task, bt) in enumerate(examples, start=1):
        body = yaml.safe_dump(bt, sort_keys=False, default_flow_style=False)
        chunks.append(f"Example {i} task: {task}\nExample {i} BT:\n```yaml\n{body.strip()}\n```")
    return "\n\n".join(chunks)


class PromptBuilder:
    """
    Construct the Gemini-Flash planning prompt from a typed skill library.
    """

    ROLE = (
        "You are SPARK, a robotic task planner. Output a single typed YAML "
        "behavior tree (root type=sequence) using only the primitives below. "
        "Use detected object labels verbatim. Do not invent new primitives. "
        "Choose macros when they fit; otherwise compose base primitives."
    )

    # Canonical-syntax example so Gemini emits the typed-node format
    # (`type: <name>`, `params: {...}`) rather than YAML-mapping shorthand
    # (`{<name>: {...}}`). Validator rejects the latter.
    SYNTAX_EXAMPLE = (
        "Required syntax for every action node (including macro calls):\n"
        "  - type: <primitive_or_macro_name>\n"
        "    params:\n"
        "      <slot>: <value>\n"
        "      ...\n"
        "Example BT for 'pick up the red cube and place on the blue cube':\n"
        "```yaml\n"
        "tree:\n"
        "  type: sequence\n"
        "  children:\n"
        "    - type: pick_and_place\n"
        "      params:\n"
        "        pick_label: red cube\n"
        "        place_label: blue cube\n"
        "```\n"
        "DO NOT use the shorthand `{pick_and_place: {pick_label: ...}}` form. "
        "Every node MUST have explicit `type:` and `params:` keys."
    )

    OUTPUT_INSTRUCTION = "Output: a single typed YAML BT, no commentary."

    def __init__(
        self,
        library: SkillLibrary,
        macros_yaml: Optional[str] = None,
        fewshot_examples: Optional[Sequence[FewShot]] = None,
    ) -> None:
        self.library = library
        self.fewshot_examples: List[FewShot] = list(fewshot_examples or [])
        self.loaded_macro_names: List[str] = []
        path = self._resolve_macros_path(macros_yaml)
        if path is not None and path.exists():
            self._load_macros(path)

    @staticmethod
    def _resolve_macros_path(macros_yaml: Optional[str]) -> Optional[Path]:
        """
        Resolve the macros file; default is the package-local macros.yaml.
        """
        if macros_yaml is None:
            return Path(__file__).resolve().parent / "macros.yaml"
        path = Path(macros_yaml)
        if not path.is_absolute() and not path.exists():
            alt = Path(__file__).resolve().parent.parent / macros_yaml
            if alt.exists():
                return alt
        return path

    # macros

    def _load_macros(self, path: Path) -> None:
        try:
            blob = yaml.safe_load(path.read_text())
        except (OSError, yaml.YAMLError):
            return
        if not isinstance(blob, dict):
            return
        entries = blob.get("macros") or []
        seen: Dict[str, int] = {}
        for entry in entries:
            if not isinstance(entry, dict):
                continue
            name = entry.get("name")
            body = entry.get("body")
            if not isinstance(name, str) or not isinstance(body, list):
                continue
            free = entry.get("free_params") or {}
            sig, rename = _clean_signature(free if isinstance(free, dict) else {})
            subtree = _substitute_body(_wrap_subtree(copy.deepcopy(body)), rename)
            count = seen.get(name, 0)
            seen[name] = count + 1
            register_name = name if count == 0 else f"{name}_v{count + 1}"
            try:
                self.library.register_macro(register_name, sig, subtree)
            except (ValueError, KeyError):
                continue
            self.loaded_macro_names.append(register_name)

    # build

    def build(
        self,
        task_instruction: str,
        detected_objects: Optional[Iterable[str]] = None,
        extra_hints: Optional[str] = None,
    ) -> str:
        """
        Assemble the full prompt string.
        """
        sections: List[str] = [self.ROLE, ""]
        sections.append(self.library.available_primitives_for_prompt())
        sections.append("")
        sections.append(self.SYNTAX_EXAMPLE)
        sections.append("")

        labels = list(detected_objects or [])
        if labels:
            sections.append("Detected objects:")
            sections.extend(f"- {lbl}" for lbl in labels)
            sections.append("")
            # Hard constraint mirroring SPARKPlanner.generate_score - every
            # target_label / keypoint_label / labels entry the model emits
            # must come from this list verbatim.  No fabrication.
            sections.append(
                "Available detection labels (you may only refer to these "
                "in target_label / keypoint_label / labels):"
            )
            sections.extend(f"  - {lbl}" for lbl in labels)
            sections.append(
                "Do NOT invent labels. If no detection matches a concept the "
                "task requires, pick the closest available label or emit a "
                "recovery hint - never fabricate a string outside this list."
            )
        else:
            sections.append("Detected objects: (none provided)")
        sections.append("")

        if self.fewshot_examples:
            sections.append("Few-shot examples:")
            sections.append(_format_fewshot(self.fewshot_examples))
            sections.append("")

        if extra_hints:
            sections.append("Hints:")
            sections.append(extra_hints.strip())
            sections.append("")

        sections.append(f"Task instruction: {task_instruction}")
        sections.append("")
        sections.append(self.OUTPUT_INSTRUCTION)
        return "\n".join(sections).rstrip() + "\n"


__all__ = ["PromptBuilder", "FewShot"]

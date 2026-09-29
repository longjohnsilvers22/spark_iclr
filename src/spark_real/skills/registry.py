"""
SPARK Skill Registry: decorator-based skill registration and dispatch.

Usage:
    from spark_real.skills.registry import SkillRegistry, spark_skill

    @spark_skill(
        name="wiggle",
        description="Wiggle end-effector for insertion assistance",
        params={"amplitude": float, "frequency": float, "duration": float, "axis": str},
    )
    def wiggle(executor, params):
        ...

    registry = SkillRegistry()
    registry.dispatch("wiggle", executor, {"amplitude": 0.005})
"""

import importlib
import logging
import pkgutil
from dataclasses import dataclass, field
from typing import Any, Callable, Dict, Optional, Type

logger = logging.getLogger(__name__)

_GLOBAL_REGISTRY: Dict[str, "SkillEntry"] = {}


@dataclass
class SkillEntry:
    """
    Metadata + callable for a single registered skill.
    """

    name: str
    fn: Callable
    description: str
    params: Dict[str, Type]  # param_name -> expected type (for docs only)

    def __call__(self, executor, params: dict):
        return self.fn(executor, params)


def spark_skill(
    name: str,
    description: str = "",
    params: Optional[Dict[str, Type]] = None,
):
    """
    Decorator that registers a function as a SPARK primitive skill.

    The decorated function must have the signature:
        def skill_fn(executor, params: dict) -> ExecutionResult

    Args:
        name: Unique skill name used in YAML scores (e.g. "wiggle").
        description: One-line description for LLM prompt injection.
        params: {param_name: type} dict documenting expected parameters.
    """
    params = params or {}

    def _decorator(fn: Callable) -> Callable:
        entry = SkillEntry(name=name, fn=fn, description=description, params=params)
        if name in _GLOBAL_REGISTRY:
            logger.warning("Skill '%s' registered twice; overwriting.", name)
        _GLOBAL_REGISTRY[name] = entry
        # Attach metadata to the function itself for introspection
        fn._spark_skill = entry
        return fn

    return _decorator


class SkillRegistry:
    """
    Central dispatch table for SPARK primitive skills.

    On construction, auto-discovers all ``@spark_skill`` decorated functions
    in the ``spark_real.skills`` package (i.e. every .py file in this
    directory).  Additional skills can be added via ``register()``.
    """

    def __init__(self, auto_discover: bool = True):
        # Start with a copy of whatever was populated at import time
        self._skills: Dict[str, SkillEntry] = dict(_GLOBAL_REGISTRY)

        if auto_discover:
            self._discover()

    def _discover(self):
        """
        Import every module in the ``spark_real.skills`` package so that
        their ``@spark_skill`` decorators fire and populate ``_GLOBAL_REGISTRY``.
        """
        # Imported here to break a circular import: registry is part of the
        # spark_real.skills package whose __init__ constructs this registry.
        import spark_real.skills as skills_pkg

        for importer, modname, ispkg in pkgutil.walk_packages(
            skills_pkg.__path__,
            prefix=skills_pkg.__name__ + ".",
        ):
            if modname.endswith(".registry"):
                continue  # Don't re-import ourselves
            try:
                importlib.import_module(modname)
            except Exception:
                logger.warning(
                    "Failed to import skill module %s", modname, exc_info=True
                )

        # Merge anything newly registered by the imports above
        self._skills.update(_GLOBAL_REGISTRY)

    def register(
        self,
        name: str,
        fn: Callable,
        description: str = "",
        params: Optional[Dict[str, Type]] = None,
    ):
        """
        Register a skill programmatically (without the decorator).
        """
        entry = SkillEntry(
            name=name, fn=fn, description=description, params=params or {}
        )
        self._skills[name] = entry
        _GLOBAL_REGISTRY[name] = entry
        logger.info("Skill '%s' registered manually.", name)

    def dispatch(self, name: str, executor, params: dict):
        """
        Look up *name* and call the skill function.

        Args:
            name: Skill / primitive name (e.g. ``"wiggle"``).
            executor: The ``ScoreExecutor`` instance (passed as first arg).
            params: Parameter dict from the YAML score node.

        Returns:
            ``ExecutionResult`` from the skill function.

        Raises:
            KeyError: If *name* is not in the registry.
        """
        entry = self._skills.get(name)
        if entry is None:
            raise KeyError(
                f"Unknown skill '{name}'. "
                f"Available: {', '.join(sorted(self._skills))}"
            )
        return entry(executor, params)

    def __contains__(self, name: str) -> bool:
        return name in self._skills

    def __len__(self) -> int:
        return len(self._skills)

    def names(self):
        """
        Return sorted list of registered skill names.
        """
        return sorted(self._skills)

    def get(self, name: str) -> Optional[SkillEntry]:
        return self._skills.get(name)

    def get_prompt_section(self) -> str:
        """
        Return a formatted block listing every skill, ready for injection
        into the planner's SYSTEM_PROMPT.

        Example output::

            Available primitives:
            - move_to_keypoint: Move end-effector to a labeled keypoint
              params: keypoint_label (str), offset_x (float), offset_y (float), offset_z (float)
            - grasp: Close gripper
              params: force (float)
            ...
        """
        lines = ["Available primitives:"]
        for name in sorted(self._skills):
            entry = self._skills[name]
            line = f"- {name}: {entry.description}"
            if entry.params:
                param_strs = [
                    f"{pname} ({ptype.__name__ if hasattr(ptype, '__name__') else ptype})"
                    for pname, ptype in entry.params.items()
                ]
                line += f"\n  params: {', '.join(param_strs)}"
            lines.append(line)
        return "\n".join(lines)


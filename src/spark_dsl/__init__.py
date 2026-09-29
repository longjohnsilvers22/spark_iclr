"""
SPARK BT-DSL: typed behavior-tree DSL with grammar + skill library.

Runner scripts in ``spark_bench/`` opt in by importing :class:`SkillLibrary`
to validate Gemini-produced plans before executing them.

Public API:
    - :data:`DEFAULT_LIBRARY` -- module-level singleton
    - :class:`SkillLibrary`   -- typed registry (8 primitives + macros)
    - :class:`Primitive`      -- primitive metadata dataclass
"""

from .skill_library import (
    DEFAULT_LIBRARY,
    Primitive,
    SkillLibrary,
)

__all__ = ["DEFAULT_LIBRARY", "Primitive", "SkillLibrary"]

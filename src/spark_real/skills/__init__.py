"""
SPARK Skill Registry: auto-discovered primitive skills for the real robot pipeline.

Import this package to get a ready-to-use registry:

    from spark_real.skills import registry

    registry.dispatch("wiggle", executor, {"amplitude": 0.005})
    print(registry.get_prompt_section())

Or import individual pieces:

    from spark_real.skills.registry import SkillRegistry, spark_skill
"""

from spark_real.skills.registry import SkillRegistry, spark_skill  # noqa: F401

# Module-level singleton; auto_discover walks every module in this package.
registry = SkillRegistry(auto_discover=True)

__all__ = ["registry", "SkillRegistry", "spark_skill"]

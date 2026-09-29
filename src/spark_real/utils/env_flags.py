"""One truthiness vocabulary for env flags and yaml booleans: 1/true/yes/on."""

import os

_TRUE = ("1", "true", "yes", "on")


def as_bool(value, default: bool) -> bool:
    """None -> default; str by the 1/true/yes/on vocabulary (case/space-insensitive); else bool()."""
    if value is None:
        return default
    if isinstance(value, str):
        return value.strip().lower() in _TRUE
    return bool(value)


def env_flag(name: str, default: bool = False) -> bool:
    return as_bool(os.environ.get(name), default)

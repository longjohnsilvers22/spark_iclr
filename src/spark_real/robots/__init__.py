"""
SPARK robot drivers package.

Provides per-embodiment driver classes (`UR10eDriver`, `FrankaDriver`,
`G1Driver`) and the `make_robot_driver` factory that selects the right one
based on a family string.  Each driver module guards its optional backend
(franky, unitree SDK, rtde, bamboo) behind a module-level try/except, so the
factory can import the driver classes at module load without the backend's
import-time side effects firing on hosts that lack it.
"""

from .factory import KNOWN_FAMILIES, make_robot_driver

__all__ = ["KNOWN_FAMILIES", "make_robot_driver"]

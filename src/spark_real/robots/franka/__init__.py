"""
Franka FR3 / Panda robot package for SPARK.

Two backends:
  - ``FrankaDriver``: franky (motion generators, default)
  - ``FrankaTorqueDriver``: pylibfranka torque-mode impedance (reflex-free)

Select at runtime via ``SPARK_FRANKA_BACKEND=torque`` (see robots/factory.py).
"""

from .franka_driver import FrankaDriver
from .franka_torque_driver import FrankaTorqueDriver

__all__ = ["FrankaDriver", "FrankaTorqueDriver"]

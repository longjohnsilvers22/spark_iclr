"""
Sensors for SPARK Real: optional add-ons (tactile, force, etc.).

Each module here is import-safe even when its underlying hardware
package is not installed. ``from spark_real.sensors.tactile import
TactileManager`` always succeeds; the manager just reports no-sensors
when the ``flexitac`` package is missing.
"""

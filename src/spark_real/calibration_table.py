"""
Table-plane calibration: load/save the measured table surface z.

The table floor is a safety height the executor clamps descents against. It
is NOT a hand-typed constant: an operator jogs the CLOSED gripper down until
the fingertips touch the table, records TCP z as `surface_z`, and the floor
sits `safety_margin_m` ABOVE that surface (less negative than surface_z).

File shape (output/calibrations/table_plane.json):
    {
        "surface_z": -0.2801,        # TCP z with closed fingertips on table
        "robot_family": "ur10e",
        "safety_margin_m": 0.003,    # floor = surface_z + margin (above)
        "measured_at": "2026-07-06T00:00:00"
    }
"""

import json
from datetime import datetime
from pathlib import Path
from typing import Optional

# Same anchor pattern as the hand-eye cals: parent is the spark_real package,
# so this resolves to spark_real/output/calibrations/table_plane.json.
TABLE_PLANE_PATH = (
    Path(__file__).resolve().parent / "output" / "calibrations" / "table_plane.json"
)


def load_table_plane(robot_family: str) -> Optional[dict]:
    """Load the table-plane cal for a robot family.

    Returns the parsed dict (surface_z, safety_margin_m, robot_family,
    measured_at) or None if the file is missing, unreadable, or belongs to a
    different family. Callers derive the floor as surface_z + safety_margin_m.
    """
    if not TABLE_PLANE_PATH.exists():
        return None
    try:
        data = json.loads(TABLE_PLANE_PATH.read_text())
    except (OSError, ValueError):
        return None
    if not isinstance(data, dict):
        return None
    if str(data.get("robot_family", "")).lower() != str(robot_family).lower():
        return None
    if "surface_z" not in data:
        return None
    return data


def save_table_plane(robot_family: str, surface_z: float, margin: float) -> Path:
    """Persist a measured table surface and return the written path.

    surface_z is TCP z with the CLOSED fingertips touching the table; margin
    is added at load time so the floor sits a few mm above the surface.
    """
    TABLE_PLANE_PATH.parent.mkdir(parents=True, exist_ok=True)
    data = {
        "surface_z": float(surface_z),
        "robot_family": str(robot_family).lower(),
        "safety_margin_m": float(margin),
        "measured_at": datetime.now().isoformat(timespec="seconds"),
    }
    TABLE_PLANE_PATH.write_text(json.dumps(data, indent=2))
    return TABLE_PLANE_PATH

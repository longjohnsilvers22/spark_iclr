"""
Auto-MJCF scene generator: perception detections -> MuJoCo XML.

Converts a list of SPARK ObjectDetection dicts (from /api/detect_approve)
into a complete MuJoCo XML string loadable by mujoco.MjModel.from_xml_string().

This is the core of the "perception-grounded sim verification for safe
manipulation" pipeline: real perception -> auto scene -> sim dry-run ->
safety gate -> real execution.

Usage (standalone test):
    python -m spark_real.perception.auto_scene
"""

from __future__ import annotations

import math
import tempfile
import time
import xml.etree.ElementTree as ET
from pathlib import Path
from typing import Optional

import numpy as np

from spark_real.utils.rotations import euler_to_quat_wxyz_str as _euler_to_quat

try:
    import mujoco  # optional dependency
except ImportError:
    mujoco = None

try:
    from PIL import Image
except ImportError:
    Image = None

__all__ = ["generate_scene_mjcf", "SHAPE_MAP"]

# Label -> primitive MuJoCo shape mapping (rule-based, no LLM).
# SHAPE_MAP fields:
# type: "cylinder" | "box"
#   h:              override height (metres)
#   h_scale:        multiply estimated height by this factor
#   r_field:        which detection field to use for radius
#   size_from_obb:  derive box dims from OBB fields
#   size:           fixed [hx, hy, hz] half-sizes for box
#   hollow:         True -> add a second inner cylinder for bowl-like shape
#   wall_thickness: metres, used when hollow=True
#   rgba:           override colour (otherwise auto-assigned)
#   mass:           override mass in kg
#   friction:       MuJoCo friction triple as string

SHAPE_MAP: dict[str, dict] = {
    # Cylindrical objects
    "cup": {"type": "cylinder"},
    "plastic cup": {"type": "cylinder"},
    "glass": {"type": "cylinder", "rgba": "0.8 0.9 1.0 0.5"},
    "bottle": {"type": "cylinder", "h_scale": 1.5},
    "water bottle": {"type": "cylinder", "h_scale": 1.8},
    "can": {"type": "cylinder"},
    "mug": {"type": "cylinder"},
    "jar": {"type": "cylinder"},
    "bowl": {
        "type": "cylinder",
        "hollow": True,
        "wall_thickness": 0.003,
        "rgba": "0.9 0.85 0.75 1",
    },
    "plate": {"type": "cylinder", "h": 0.012},
    "dish": {"type": "cylinder", "h": 0.015},
    # Flat / elongated objects
    "knife": {
        "type": "box",
        "size_from_obb": True,
        "rgba": "0.85 0.85 0.85 1",
        "mass": 0.10,
    },
    "fork": {
        "type": "box",
        "size_from_obb": True,
        "rgba": "0.85 0.85 0.85 1",
        "mass": 0.08,
    },
    "spoon": {
        "type": "box",
        "size_from_obb": True,
        "rgba": "0.85 0.85 0.85 1",
        "mass": 0.06,
    },
    "spatula": {"type": "box", "size_from_obb": True},
    "chopstick": {"type": "box", "size_from_obb": True, "mass": 0.02},
    "chopsticks": {"type": "box", "size_from_obb": True, "mass": 0.04},
    "pen": {"type": "box", "size_from_obb": True, "mass": 0.02},
    "marker": {"type": "box", "size_from_obb": True, "mass": 0.03},
    # Box-like objects
    "sponge": {
        "type": "box",
        "size": [0.05, 0.03, 0.015],
        "rgba": "1.0 0.9 0.2 1",
        "mass": 0.02,
    },
    "tray": {
        "type": "box",
        "size_from_obb": True,
        "h": 0.02,
        "rgba": "0.35 0.25 0.18 1",
    },
    "cutting board": {"type": "box", "size_from_obb": True, "h": 0.015},
    "book": {"type": "box", "size_from_obb": True, "h": 0.025},
    "box": {"type": "box", "size_from_obb": True},
    "block": {"type": "box"},
    "eraser": {"type": "box", "size": [0.03, 0.015, 0.01]},
    "napkin": {"type": "box", "size_from_obb": True, "h": 0.003, "mass": 0.01},
    "cloth": {"type": "box", "size_from_obb": True, "h": 0.004, "mass": 0.02},
}

# Colour palette for auto-colouring (cycled for unknown / uncoloured objects).
_COLOURS = [
    "0.85 0.3  0.3  1",  # red
    "0.3  0.7  0.3  1",  # green
    "0.3  0.4  0.85 1",  # blue
    "0.85 0.7  0.2  1",  # yellow
    "0.7  0.3  0.7  1",  # purple
    "0.2  0.75 0.75 1",  # teal
    "0.9  0.5  0.2  1",  # orange
    "0.6  0.6  0.6  1",  # grey
]

# Robot MJCF path (FR3+hand as used on the real system).
_FR3_MJCF = Path(__file__).parent.parent / "robots" / "franka" / "fr3_with_hand.xml"
# Only the FR3 robot is available here; the Panda sim base is not in this build.

# FR3 joint home (from the keyframe in fr3_with_hand.xml). Differs from the
# stock-Panda sim keyframe (J4, J7 sign) and from
# franka_base.HOME_CONFIG / franka_default.yaml home_config (do not unify).
_FR3_HOME_QPOS = "0 0 0 -1.57079 0 1.57079 -0.7853"
_FR3_HOME_CTRL = "0 0 0 -1.57079 0 1.57079 -0.7853"

# Table defaults. Real FR3 table top is at world Z ~= 0.
_TABLE_Z = 0.0  # Top surface of the table in world frame.
_TABLE_HALF_X = 0.45
_TABLE_HALF_Y = 0.40
_TABLE_THICKNESS = 0.02


# Helpers


def _safe_name(label: str, idx: int) -> str:
    """
    Convert a detection label into a valid MuJoCo body name.
    """
    name = label.lower().replace(" ", "_").replace("-", "_")
    name = "".join(c for c in name if c.isalnum() or c == "_")  # strip non-alphanumeric
    return f"obj_{name}_{idx}"


def _estimate_height(det: dict, shape_cfg: dict) -> float:
    """
    Estimate object height from detection fields + shape config.
    """
    if "h" in shape_cfg:
        return float(shape_cfg["h"])
    minor = float(det.get("obb_minor_m", 0) or 0)
    ar = float(det.get("aspect_ratio", 1.0) or 1.0)
    h_scale = float(shape_cfg.get("h_scale", 1.0))
    # For cylinders: height ~ minor (diameter ~ minor, height ~ minor).
    # For boxes with size_from_obb: the OBB already encodes it.
    if minor > 0.005:
        return max(minor * h_scale, 0.01)
    # Fallback: 5cm.
    return 0.05 * h_scale


def _estimate_radius(det: dict, shape_cfg: dict) -> float:
    """
    Estimate cylinder radius from detection fields.
    """
    minor = float(det.get("obb_minor_m", 0) or 0)
    if minor > 0.005:
        return minor / 2.0
    return 0.03  # 3cm default radius


def _obb_half_sizes(det: dict, shape_cfg: dict) -> tuple[float, float, float]:
    """
    Compute (hx, hy, hz) half-sizes for box geom from OBB fields.
    """
    minor = float(det.get("obb_minor_m", 0) or 0)
    ar = float(det.get("aspect_ratio", 1.0) or 1.0)
    if minor < 0.005:
        minor = 0.04
    major = minor * max(ar, 1.0)
    h = float(shape_cfg.get("h", minor * 0.3))
    # MuJoCo box size = half-extents; major direction is longest.
    return (major / 2.0, minor / 2.0, h / 2.0)


def _object_pos_str(det: dict, h_geom: float) -> str:
    """
    Compute body position from detected world position.

    Places the body so its bottom sits on the table surface (_TABLE_Z).
    The detection's position_3d is the centroid; Z is adjusted so the
    geom bottom = table top.
    """
    pos = det.get("position_3d")
    if pos is None:
        return "0.5 0 0.05"
    if isinstance(pos, np.ndarray):
        pos = pos.tolist()
    x, y, z = float(pos[0]), float(pos[1]), float(pos[2])
    # Place bottom on table: body centre Z = table_z + half_height.
    z_placed = _TABLE_Z + h_geom
    return f"{x:.4f} {y:.4f} {z_placed:.4f}"


def _yaw_from_det(det: dict) -> float:
    """
    World-frame yaw from world_major_axis_rad, else image-frame orientation_angle.
    """
    wma = det.get("world_major_axis_rad")
    if wma is not None:
        return float(wma)
    return float(det.get("orientation_angle", 0.0) or 0.0)


# Main generator


def generate_scene_mjcf(
    detections: list[dict],
    robot_mjcf_path: str | None = None,
    table_centre: tuple[float, float] = (0.55, 0.0),
    include_robot: bool = True,
) -> str:
    """
    Generate a complete MuJoCo XML string from perception detections.

    Parameters:
    detections : list[dict]
        Each dict has fields from serialize_detections():
        label, position_3d, obb_minor_m, aspect_ratio,
        orientation_angle, depth_meters, world_major_axis_rad.
        Also accepts ObjectDetection dataclass instances.
    robot_mjcf_path : str or None
        Path to robot MJCF. Defaults to FR3+hand. The "panda" value is
        no longer supported because the CoRL Panda sim base was removed
        with the simulation package; passing it raises RuntimeError.
    table_centre : tuple
        (x, y) world position of table centre.
    include_robot : bool
        Whether to include the robot body. Set False for object-only
        scenes (e.g. for collision checking).

    Returns:
    str
        Complete MJCF XML string loadable by
        mujoco.MjModel.from_xml_string().
    """
    t0 = time.monotonic()

    if robot_mjcf_path == "panda":
        raise RuntimeError(
            "The 'panda' robot base for auto-scene generation depended on "
            "the CoRL sim scenes from the removed simulation package, which "
            "are not part of this build. Use the default FR3 robot instead."
        )

    # Normalise detections: accept both dicts and ObjectDetection dataclasses.
    dets = []
    for d in detections:
        if hasattr(d, "label"):
            # ObjectDetection dataclass -> dict
            entry = {
                "label": d.label,
                "position_3d": (
                    d.position_3d.tolist()
                    if isinstance(d.position_3d, np.ndarray)
                    else d.position_3d
                ),
                "obb_minor_m": float(getattr(d, "obb_minor_m", 0) or 0),
                "aspect_ratio": float(getattr(d, "aspect_ratio", 1.0) or 1.0),
                "orientation_angle": float(getattr(d, "orientation_angle", 0) or 0),
                "depth_meters": float(getattr(d, "depth_meters", 0) or 0),
                "world_major_axis_rad": getattr(d, "world_major_axis_rad", None),
            }
            dets.append(entry)
        else:
            dets.append(dict(d))

    # Build XML tree
    root = ET.Element("mujoco", model="auto_scene")

    # Compiler
    if include_robot:
        rpath = robot_mjcf_path or str(_FR3_MJCF)
        ET.SubElement(
            root,
            "compiler",
            angle="radian",
            meshdir=str(
                Path(rpath).parent / "../../mujoco_menagerie/franka_fr3/assets"
            ),
        )
    else:
        ET.SubElement(root, "compiler", angle="radian", autolimits="true")

    ET.SubElement(root, "option", integrator="implicitfast")

    # Visual
    vis = ET.SubElement(root, "visual")
    ET.SubElement(
        vis, "headlight", diffuse="0.4 0.4 0.4", ambient="0.2 0.2 0.2", specular="0 0 0"
    )
    ET.SubElement(vis, "rgba", haze="0.15 0.25 0.35 1")
    ET.SubElement(
        vis, "global", azimuth="120", elevation="-20", offwidth="800", offheight="600"
    )

    # Assets
    asset = ET.SubElement(root, "asset")
    ET.SubElement(
        asset,
        "texture",
        type="skybox",
        builtin="gradient",
        rgb1="0.3 0.5 0.8",
        rgb2="0.1 0.2 0.4",
        width="512",
        height="3072",
    )
    ET.SubElement(
        asset,
        "texture",
        type="2d",
        name="lab_floor",
        builtin="checker",
        rgb1="0.35 0.4 0.45",
        rgb2="0.25 0.3 0.35",
        width="512",
        height="512",
    )
    ET.SubElement(
        asset,
        "material",
        name="groundplane",
        texture="lab_floor",
        texrepeat="20 20",
        reflectance="0.3",
        shininess="0.1",
    )
    ET.SubElement(
        asset,
        "material",
        name="table_mat",
        rgba="0.7 0.5 0.3 1",
        reflectance="0.4",
        shininess="0.3",
    )

    # Per-object materials
    for i, det in enumerate(dets):
        label = det["label"].lower()
        cfg = SHAPE_MAP.get(label, {})
        rgba = cfg.get("rgba", _COLOURS[i % len(_COLOURS)])
        mat_name = f"mat_{_safe_name(det['label'], i)}"
        ET.SubElement(
            asset, "material", name=mat_name, rgba=rgba, specular="0.4", shininess="0.3"
        )

    # Worldbody
    wb = ET.SubElement(root, "worldbody")

    # Lights
    ET.SubElement(
        wb,
        "light",
        pos="0 0 4",
        dir="0 0 -1",
        directional="true",
        diffuse="0.5 0.5 0.5",
    )
    ET.SubElement(
        wb, "light", pos="2 2 3", dir="-0.5 -0.5 -1", diffuse="0.25 0.25 0.25"
    )
    ET.SubElement(wb, "light", pos="0 -2 3", dir="0 0.5 -1", diffuse="0.2 0.2 0.2")

    # Floor
    ET.SubElement(
        wb,
        "geom",
        name="floor",
        size="10 10 0.01",
        type="plane",
        material="groundplane",
    )

    # Table
    tx, ty = table_centre
    table_body = ET.SubElement(
        wb, "body", name="table", pos=f"{tx:.3f} {ty:.3f} {_TABLE_Z - 0.01:.3f}"
    )
    ET.SubElement(
        table_body,
        "geom",
        name="table_top",
        type="box",
        size=f"{_TABLE_HALF_X} {_TABLE_HALF_Y} {_TABLE_THICKNESS/2:.3f}",
        material="table_mat",
        friction="1 0.5 0.0001",
    )

    # Robot inclusion via <include>, which keeps the XML clean but is not
    # resolved by from_xml_string(), so the result must be loaded via
    # from_xml_path of a temp file (the /api/auto_scene endpoint does this).
    if include_robot:
        rpath = robot_mjcf_path or str(_FR3_MJCF)
        ET.SubElement(root, "include", file=rpath)

    # Objects
    objects_info = []
    for i, det in enumerate(dets):
        label = det["label"]
        label_lower = label.lower()
        cfg = SHAPE_MAP.get(label_lower, {})
        body_name = _safe_name(label, i)
        mat_name = f"mat_{body_name}"
        shape_type = cfg.get("type", "box")
        used_default = label_lower not in SHAPE_MAP

        yaw = _yaw_from_det(det)
        quat = _euler_to_quat(yaw)

        if shape_type == "cylinder":
            radius = _estimate_radius(det, cfg)
            height = _estimate_height(det, cfg)
            half_h = height / 2.0
            pos_str = _object_pos_str(det, half_h)

            body = ET.SubElement(wb, "body", name=body_name, pos=pos_str, quat=quat)
            ET.SubElement(body, "freejoint", name=f"{body_name}_free")
            ET.SubElement(
                body,
                "geom",
                name=f"{body_name}_geom",
                type="cylinder",
                size=f"{radius:.4f} {half_h:.4f}",
                material=mat_name,
                condim="6",
                friction="1 0.5 0.001",
            )
            mass = cfg.get("mass", max(0.05, radius * height * 500))
            ET.SubElement(
                body,
                "inertial",
                pos="0 0 0",
                mass=f"{mass:.4f}",
                diaginertia=f"{mass*0.001:.6f} {mass*0.001:.6f} {mass*0.0005:.6f}",
            )

            # Hollow interior for bowls
            if cfg.get("hollow"):
                wall = cfg.get("wall_thickness", 0.003)
                inner_r = max(radius - wall, 0.005)
                inner_h = max(half_h - wall, 0.005)
                ET.SubElement(
                    body,
                    "geom",
                    name=f"{body_name}_inner",
                    type="cylinder",
                    size=f"{inner_r:.4f} {inner_h:.4f}",
                    pos=f"0 0 {wall:.4f}",
                    rgba="0 0 0 0",
                    contype="0",
                    conaffinity="0",
                    group="3",
                )

            objects_info.append(
                {
                    "name": body_name,
                    "label": label,
                    "shape": "cylinder",
                    "radius_m": round(radius, 4),
                    "height_m": round(height, 4),
                    "position": pos_str,
                    "default_box": used_default,
                }
            )

        else:  # box
            if "size" in cfg:
                hx, hy, hz = [s / 2.0 for s in cfg["size"]]
            elif cfg.get("size_from_obb"):
                hx, hy, hz = _obb_half_sizes(det, cfg)
            else:
                minor = float(det.get("obb_minor_m", 0) or 0)
                ar = float(det.get("aspect_ratio", 1.0) or 1.0)
                if minor < 0.005:
                    minor = 0.04
                major = minor * max(ar, 1.0)
                h = float(cfg.get("h", minor))
                hx, hy, hz = major / 2.0, minor / 2.0, h / 2.0

            pos_str = _object_pos_str(det, hz)

            body = ET.SubElement(wb, "body", name=body_name, pos=pos_str, quat=quat)
            ET.SubElement(body, "freejoint", name=f"{body_name}_free")
            ET.SubElement(
                body,
                "geom",
                name=f"{body_name}_geom",
                type="box",
                size=f"{hx:.4f} {hy:.4f} {hz:.4f}",
                material=mat_name,
                condim="6",
                friction="1 0.5 0.001",
            )
            mass = cfg.get("mass", max(0.03, hx * hy * hz * 8 * 1000))
            ET.SubElement(
                body,
                "inertial",
                pos="0 0 0",
                mass=f"{mass:.4f}",
                diaginertia=f"{mass*0.001:.6f} {mass*0.001:.6f} {mass*0.0005:.6f}",
            )

            objects_info.append(
                {
                    "name": body_name,
                    "label": label,
                    "shape": "box",
                    "half_sizes_m": [round(hx, 4), round(hy, 4), round(hz, 4)],
                    "position": pos_str,
                    "default_box": used_default,
                }
            )

    # Cameras (agentview, sideview, topview matching CoRL scenes)
    cam_target = ET.SubElement(wb, "body", name="cam_target", pos=f"{tx:.3f} 0 0.05")
    ET.SubElement(
        cam_target,
        "site",
        name="cam_target_site",
        pos="0 0 0",
        size="0.001",
        rgba="0 0 0 0",
    )

    cam_agent = ET.SubElement(wb, "body", name="cam_agent_body", pos="1.5 0.0 0.9")
    ET.SubElement(
        cam_agent,
        "camera",
        name="agentview",
        fovy="55",
        pos="0 0 0",
        mode="targetbody",
        target="cam_target",
    )

    cam_side = ET.SubElement(wb, "body", name="cam_side_body", pos=f"{tx:.3f} -1.1 0.5")
    ET.SubElement(
        cam_side,
        "camera",
        name="sideview",
        fovy="55",
        pos="0 0 0",
        mode="targetbody",
        target="cam_target",
    )

    cam_top = ET.SubElement(wb, "body", name="cam_top_body", pos=f"{tx:.3f} 0 1.5")
    ET.SubElement(
        cam_top,
        "camera",
        name="topview",
        fovy="50",
        pos="0 0 0",
        mode="targetbody",
        target="cam_target",
    )

    # Keyframe (robot home + objects at detected positions)
    if include_robot:
        kf = ET.SubElement(root, "keyframe")
        # Build qpos: 7 arm joints + 2 finger joints + per-object freejoints (7 each: 3 pos + 4 quat)
        obj_qpos_parts = []
        for i, det in enumerate(dets):
            label_lower = det["label"].lower()
            cfg = SHAPE_MAP.get(label_lower, {})
            shape_type = cfg.get("type", "box")
            if shape_type == "cylinder":
                h = _estimate_height(det, cfg) / 2.0
            elif "size" in cfg:
                h = cfg["size"][2] / 4.0
            elif cfg.get("size_from_obb"):
                _, _, h = _obb_half_sizes(det, cfg)
            else:
                minor = float(det.get("obb_minor_m", 0) or 0)
                if minor < 0.005:
                    minor = 0.04
                h_val = float(cfg.get("h", minor))
                h = h_val / 2.0

            pos = det.get("position_3d")
            if pos is None:
                px, py, pz = 0.5, 0.0, _TABLE_Z + h
            else:
                if isinstance(pos, np.ndarray):
                    pos = pos.tolist()
                px, py = float(pos[0]), float(pos[1])
                pz = _TABLE_Z + h

            yaw = _yaw_from_det(det)
            cw = math.cos(yaw / 2.0)
            sz = math.sin(yaw / 2.0)
            obj_qpos_parts.append(f"{px:.4f} {py:.4f} {pz:.4f} {cw:.6f} 0 0 {sz:.6f}")

        # FR3: 7 arm + 2 finger DOFs, then free joints
        qpos = f"{_FR3_HOME_QPOS} 0.04 0.04"
        if obj_qpos_parts:
            qpos += " " + " ".join(obj_qpos_parts)
        ctrl = f"{_FR3_HOME_CTRL} 255"
        ET.SubElement(kf, "key", name="auto_scene_home", qpos=qpos, ctrl=ctrl)

    # Serialise
    ET.indent(root, space="  ")
    xml_str = ET.tostring(root, encoding="unicode", xml_declaration=False)

    elapsed_ms = (time.monotonic() - t0) * 1000
    return xml_str, objects_info, elapsed_ms


def validate_mjcf(xml_str: str) -> tuple[bool, str]:
    """
    Try loading the MJCF string in MuJoCo. Returns (ok, error_msg).
    """
    try:
        if mujoco is None:
            raise RuntimeError("mujoco not installed")

        # from_xml_string doesn't resolve <include> file= paths, so write
        # to a temp file and use from_xml_path.
        with tempfile.NamedTemporaryFile(
            mode="w", suffix=".xml", delete=False, dir="/tmp"
        ) as f:
            f.write(xml_str)
            tmp_path = f.name
        model = mujoco.MjModel.from_xml_path(tmp_path)
        return True, f"OK: {model.nbody} bodies, {model.njnt} joints"
    except Exception as e:
        return False, str(e)


def render_scene(
    xml_path: str, cam_name: str = "agentview", width: int = 800, height: int = 600
) -> Optional[np.ndarray]:
    """
    Render a single frame from the generated scene. Returns RGB array.
    """
    try:
        if mujoco is None:
            raise RuntimeError("mujoco not installed")

        model = mujoco.MjModel.from_xml_path(xml_path)
        data = mujoco.MjData(model)
        # Reset to home keyframe if available
        if model.nkey > 0:
            mujoco.mj_resetDataKeyframe(model, data, 0)
        mujoco.mj_forward(model, data)
        # Find camera
        cam_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_CAMERA, cam_name)
        if cam_id < 0:
            cam_id = 0
        renderer = mujoco.Renderer(model, height, width)
        renderer.update_scene(data, camera=cam_id)
        rgb = renderer.render().copy()
        renderer.close()
        return rgb
    except Exception:
        return None


# CLI test

if __name__ == "__main__":
    # Synthetic detections mimicking cup + bowl on the real table.
    test_dets = [
        {
            "label": "cup",
            "position_3d": [0.45, 0.10, 0.04],
            "obb_minor_m": 0.07,
            "aspect_ratio": 1.2,
            "orientation_angle": 0.0,
            "depth_meters": 0.85,
            "world_major_axis_rad": None,
        },
        {
            "label": "bowl",
            "position_3d": [0.55, -0.12, 0.03],
            "obb_minor_m": 0.12,
            "aspect_ratio": 1.1,
            "orientation_angle": 0.0,
            "depth_meters": 0.90,
            "world_major_axis_rad": None,
        },
        {
            "label": "knife",
            "position_3d": [0.50, 0.22, 0.01],
            "obb_minor_m": 0.018,
            "aspect_ratio": 8.5,
            "orientation_angle": 0.3,
            "depth_meters": 0.82,
            "world_major_axis_rad": 0.3,
        },
    ]

    xml_str, objects, elapsed = generate_scene_mjcf(test_dets)
    print(f"Generated MJCF ({elapsed:.1f} ms), {len(objects)} objects:")
    for o in objects:
        print(
            f"  {o['label']}: {o['shape']} at {o['position']} "
            f"{'(default box)' if o['default_box'] else ''}"
        )

    # Validate
    ok, msg = validate_mjcf(xml_str)
    print(f"Validation: {msg}")

    if ok:
        with tempfile.NamedTemporaryFile(
            mode="w", suffix=".xml", delete=False, dir="/tmp", prefix="auto_scene_"
        ) as f:
            f.write(xml_str)
            tmp_path = f.name
        print(f"Saved to: {tmp_path}")

        # Try rendering
        rgb = render_scene(tmp_path)
        if rgb is not None and Image is not None:
            out_path = "/tmp/auto_scene_preview.png"
            Image.fromarray(rgb).save(out_path)
            print(f"Rendered preview: {out_path} ({rgb.shape})")

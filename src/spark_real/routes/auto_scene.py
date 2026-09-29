"""
Auto-scene generation routes: perception -> MJCF.

Endpoints:
  POST /api/auto_scene   : generate MJCF from pending detections
"""

import base64
import io
import logging
import tempfile

from fastapi import APIRouter
from fastapi.responses import JSONResponse
from PIL import Image

from spark_real.perception.auto_scene import (
    generate_scene_mjcf,
    render_scene,
    validate_mjcf,
)
from spark_real.routes import state

logger = logging.getLogger("spark_server")
router = APIRouter()


@router.post("/api/auto_scene")
async def auto_scene(req: dict = None):
    """
    Generate a MuJoCo XML scene from the latest approved detections.

    Reads ``state.pending_detections`` (set by POST /api/detect_approve)
    and produces a complete MJCF XML loadable by MuJoCo. Optionally
    renders a preview frame.

    Request body (all optional):
      {
        "robot": "fr3_with_hand" | "panda",   // default: fr3_with_hand
        "include_robot": true,                 // default: true
        "render_preview": true,                // default: true
        "camera": "agentview"                  // preview camera name
      }

    Returns:
      {
        "mjcf": "<mujoco>...</mujoco>",
        "objects": [...],
        "robot": "fr3_with_hand",
        "xml_path": "/tmp/auto_scene_*.xml",
        "generation_ms": 1.2,
        "validation": {"ok": true, "msg": "..."},
        "preview_image": "base64 PNG..."       // if render_preview=true
      }
    """
    body = req or {}
    robot_choice = body.get("robot", "fr3_with_hand")
    include_robot = body.get("include_robot", True)
    render_preview = body.get("render_preview", True)
    camera = body.get("camera", "agentview")

    # Get detections from state (set by /api/detect_approve).
    detections = state.pending_detections
    if detections is None or len(detections) == 0:
        return JSONResponse(
            status_code=400,
            content={
                "error": "No pending detections. Run POST " "/api/detect_approve first."
            },
        )

    # Map robot choice to MJCF path.
    robot_mjcf = None
    if robot_choice == "panda":
        robot_mjcf = "panda"
    # else: None -> defaults to FR3+hand in auto_scene.py

    # Serialize detections if they are ObjectDetection instances.
    # generate_scene_mjcf accepts both dicts and dataclasses.
    xml_str, objects_info, gen_ms = generate_scene_mjcf(
        detections,
        robot_mjcf_path=robot_mjcf,
        include_robot=include_robot,
    )

    # Write to temp file (MuJoCo needs file path for <include>).
    with tempfile.NamedTemporaryFile(
        mode="w", suffix=".xml", delete=False, dir="/tmp", prefix="auto_scene_"
    ) as f:
        f.write(xml_str)
        xml_path = f.name

    # Validate.
    ok, val_msg = validate_mjcf(xml_str)

    result = {
        "mjcf": xml_str,
        "objects": objects_info,
        "robot": robot_choice,
        "xml_path": xml_path,
        "generation_ms": round(gen_ms, 2),
        "validation": {"ok": ok, "msg": val_msg},
        "n_detections": len(detections),
        "mapped_labels": [o["label"] for o in objects_info if not o.get("default_box")],
        "default_boxed_labels": [
            o["label"] for o in objects_info if o.get("default_box")
        ],
    }

    # Render preview if requested and validation passed.
    if render_preview and ok:
        try:
            rgb = render_scene(xml_path, cam_name=camera)
            if rgb is not None:
                buf = io.BytesIO()
                Image.fromarray(rgb).save(buf, format="PNG")
                result["preview_image"] = base64.b64encode(buf.getvalue()).decode()
                # Also save to disk for easy viewing.
                png_path = xml_path.replace(".xml", ".png")
                Image.fromarray(rgb).save(png_path)
                result["preview_path"] = png_path
        except Exception as e:
            logger.warning("auto_scene render failed: %s", e)
            result["preview_error"] = str(e)

    logger.info(
        "auto_scene: %d objects, %.1f ms, valid=%s, path=%s",
        len(objects_info),
        gen_ms,
        ok,
        xml_path,
    )
    return result

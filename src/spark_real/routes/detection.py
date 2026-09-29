# Detection routes: detect, detect_click, detect_box, detect_approve.

import asyncio
import base64
import copy
import logging
import os
import time
from collections import defaultdict

import cv2
import numpy as np
import torch
from PIL import Image
from fastapi import APIRouter
from fastapi.responses import JSONResponse

from spark_real.perception.spark_perception import ObjectDetection
from spark_real.perception.mask_geometry import _cloud_median_world, _pca_obb
from spark_real.perception.prompt_registry import PromptCountMismatch

from spark_real.routes import state, streaming
from spark_real.routes.models import (
    BoxDetectRequest,
    ClickDetectRequest,
    DetectApproveRequest,
    DetectRequest,
)
from spark_real.routes.visualization import (
    create_detection_overlay,
    create_tiled_detection_overlay,
    draw_bbox_overlay,
    draw_grasp_preview,
    encode_image_b64,
    depth_to_colormap,
    serialize_detections,
)

logger = logging.getLogger("spark_server")
router = APIRouter()


def _merge_into_pending(dets, *, source):
    """Add hand-made detections to the pending set instead of REPLACING it.

    A re-annotation of the same label supersedes the old one; everything else
    accumulates.
    """
    prior = list(state.pending_detections or [])
    fresh_labels = {str(d.label).lower() for d in dets}
    kept = [d for d in prior if str(d.label).lower() not in fresh_labels]
    merged = kept + list(dets)
    state.pending_detections = merged
    if kept:
        logger.info(
            "%s: kept %d earlier detection(s) [%s] alongside the new one(s) [%s]",
            source, len(kept), ", ".join(str(d.label) for d in kept),
            ", ".join(str(d.label) for d in dets),
        )
    return merged


@router.post("/api/detect")
def detect_objects(req: DetectRequest):
    pipeline = state.pipeline
    if pipeline is None:
        return JSONResponse(
            status_code=400, content={"error": "Pipeline not initialized"}
        )
    with pipeline.activity("capturing"):
        captures = pipeline.capture()
    if not captures:
        return JSONResponse(status_code=500, content={"error": "No captures"})

    # A registered task with no operator override runs its frozen contract via
    # detect_for_task, the one implementation of the count gate and its
    # escalation ladder (relax the secondary-mask score, then the registered
    # alt_prompts, then an LLM prompt proposal).
    #
    # Operator-supplied prompts stay on the legacy path deliberately. Typing
    # prompts into the box is an experiment, not the contract; holding it to
    # the task's declared counts would abort a probe of a scene the operator
    # already knows is wrong.
    spec = pipeline.task_spec(req.instruction) if not req.prompts else None
    prompts = req.prompts or (spec.prompts if spec is not None else [])
    # A registered task declares which of its prompts can legitimately return
    # more than one mask. Without that, multi_instance defaults False and the
    # second of two identical blocks is never even segmented.
    multi = req.multi_instance or (spec is not None and bool(spec.multi_instance_prompts))

    with pipeline.activity("detecting"):
        if spec is not None:
            try:
                merged, all_detections, _res, spec, prompts = pipeline.detect_for_task(
                    captures, req.instruction, spec=spec
                )
            except PromptCountMismatch as exc:
                # The task's own policy is `abort`: a 200 with the object
                # missing sends the arm into a spiral search at z=0.35.
                return JSONResponse(
                    status_code=422,
                    content={
                        "error": str(exc),
                        "instruction": req.instruction,
                        "reason": "prompt_count_mismatch",
                    },
                )
        else:
            all_detections = pipeline.detect(
                captures,
                prompts,
                multi_instance=multi,
            )
            merged = pipeline.merge_detections(all_detections, spec=spec)
        annotated, tiled_bboxes = create_tiled_detection_overlay(
            captures,
            all_detections,
        )
    serialized = serialize_detections(merged)
    for i, s in enumerate(serialized):
        s["tiled_bbox"] = tiled_bboxes.get(s["label"])
    serialized_all = serialize_detections(all_detections)
    # Text prompts accumulate on the same terms as a click or a box: a text
    # detect for "spoon" must not silently erase a boxed "tan button" the
    # operator drew a minute earlier. Same-label results supersede, everything
    # else survives, and the approval list's discard control is there for
    # anything that should not have.
    # Do NOT rebind `merged`: the response must report what THIS call found.
    # Accumulation belongs to the pending set alone -- returning the union
    # would make a free-form detect claim credit for every earlier annotation.
    _merge_into_pending(merged, source="detect(text)")
    state.pending_captures = captures
    return {
        "detections": serialized,
        "all_detections": serialized_all,
        "annotated_image": annotated,
        "count": len(merged),
    }


@router.post("/api/scene_pointcloud")
def scene_pointcloud(req: DetectRequest):
    """
    CaP-X-style Contact-GraspNet input: the FULL birdview scene cloud (world
    frame) + per-point segment labels (0=background, i+1 = i-th detected object).
    CGN needs the full scene for context + a segmap to emit object grasps; an
    object-only cloud yields 0 grasps.
    """
    pipeline = state.pipeline
    if pipeline is None:
        return JSONResponse(status_code=400, content={"error": "Pipeline not initialized"})
    from spark_real.perception.scene_cloud import build_scene_cloud

    try:
        cloud = build_scene_cloud(
            pipeline, req.prompts, multi_instance=req.multi_instance
        )
    except RuntimeError as exc:
        return JSONResponse(status_code=500, content={"error": str(exc)})
    return {
        "camera": cloud["camera"],
        "n_points": int(len(cloud["points"])),
        "points": cloud["points"].tolist(),
        "segment_labels": cloud["segment_labels"].tolist(),
        "objects": cloud["objects"],
    }



def _snapshot_frame(pipeline):
    """
    One consistent ``(rgb, depth, cam_name)`` for a click/box detect.

    Returns the last published frame, or captures one and publishes it if
    nothing has been published yet. The publish also carries the camera name,
    so the very first click after a restart deprojects through the right
    extrinsics.

    Returns ``(None, None, None)`` when there is nothing to capture.
    """
    rgb, depth, cam_name = state.get_last_frame()
    if rgb is not None:
        return rgb, depth, cam_name
    captures = pipeline.capture()
    if not captures:
        return None, None, None
    cam_key = state.active_camera
    if cam_key not in captures:
        cam_key = next(iter(captures))
    cam = captures[cam_key]
    state.set_last_frame(cam["rgb"], cam["depth"], cam_key)
    return state.get_last_frame()


def _cal_for_camera(pipeline, cam_name):
    """The CameraCalibration for a slot name, or None (sim / unknown slot)."""
    if pipeline._kinect_cal and cam_name == "sideview":
        return pipeline._kinect_cal
    if pipeline._kinect2_cal and cam_name == "birdview":
        return pipeline._kinect2_cal
    if pipeline._realsense_cal and cam_name == "wrist":
        return pipeline._realsense_cal
    return None


def _backproject_prompt_mask(pipeline, mask, depth, cam_name, h, w):
    """Median world position + robust depth for a click/box mask.

    Deprojects through mask_geometry._cloud_median_world with
    use_opencv=True, the same convention and code the text-detect path
    uses (_process_single_mask on hardware depth). The hand-eye extrinsic
    is solved for the OpenCV pinhole convention (y down, z_c = +d);
    deprojecting in the OpenGL/MuJoCo convention (y up, z_c = -d) moves
    every birdview annotation from the table (z = -0.270 in base) to
    roughly the camera's slant range (z = +1.52..1.57). Sharing
    _cloud_median_world also buys the top-surface Z logic (95th percentile
    when the mask straddles table + object), so click/box and text
    detections of the same object agree in all three axes.

    Returns ``(position_3d | None, d_val, cal | None)``.
    """
    cal = _cal_for_camera(pipeline, cam_name)
    if depth is None or cal is None:
        return None, 0.0, cal
    fx = cal.fx if cal.fx > 0 else h / (2 * np.tan(np.deg2rad(cal.fovy_degrees) / 2))
    fy = cal.fy if cal.fy > 0 else fx
    cx_intr = cal.cx if cal.cx > 0 else w / 2
    cy_intr = cal.cy if cal.cy > 0 else h / 2
    position_3d, d_val = _cloud_median_world(
        mask,
        depth,
        cal.rotation_matrix,
        cal.position,
        fx=fx,
        fy=fy,
        cx_k=cx_intr,
        cy_k=cy_intr,
        use_opencv=True,
    )
    return position_3d, d_val, cal


def _run_point_prompt(pipeline, rgb, depth, px_x, px_y, label="", cam_name=None):
    """
    Run SAM3 with a true point prompt on the image.

    Uses the SAM2 instance predictor path (predict_inst) for precise
    click-to-segment.  Returns a list with 0 or 1 ObjectDetection.

    ``cam_name`` names the camera ``rgb``/``depth`` came from and selects the
    calibration to deproject through. It is a parameter, not a re-read of
    state.last_cam_name, so a concurrent stream publish cannot swap it
    mid-call (see routes/state.set_last_frame).
    """
    perception = pipeline._perception
    perception.load_models(load_da3=False)
    h, w = rgb.shape[:2]
    pil_img = Image.fromarray(rgb)
    # The SAM3 predictor is stateful; serialize set_image + prompt against
    # any concurrent pipeline.detect() (executor re-detect thread, stream
    # detects) via the shared perception lock.
    with perception._detect_lock:
        sam3_state = perception._sam3.set_image(pil_img)

        try:
            sam3_state = perception.set_point_prompt(
                px_x,
                px_y,
                sam3_state,
                label=1,
            )
        except Exception as exc:
            logger.warning("Point prompt failed (%s), will fall back to box", exc)
            return []  # caller should fall back

    masks = sam3_state.get("masks", torch.tensor([]))
    scores = sam3_state.get("scores", torch.tensor([]))
    if masks.numel() == 0:
        return []

    best_idx = int(scores.argmax())
    score = float(scores[best_idx])
    mask = masks[best_idx].cpu().numpy().squeeze()
    ys, xs = np.where(mask > 0)
    if len(xs) == 0:
        return []

    cx_det, cy_det = float(xs.mean()), float(ys.mean())

    # Backproject with depth -- shared with _run_box_prompt and, more
    # importantly, with the text-detect convention (see helper docstring).
    position_3d, d_val, cal = _backproject_prompt_mask(
        pipeline, mask, depth, cam_name, h, w
    )

    img_angle_rad, aspect_ratio, short_px, _long_px = _pca_obb(mask > 0)
    orientation_angle = float(img_angle_rad) if aspect_ratio > 1.3 else 0.0

    obb_minor_m = 0.0
    if short_px > 0 and d_val > 0.01 and cal is not None:
        f_minor = (
            cal.fx if cal.fx > 0 else h / (2 * np.tan(np.deg2rad(cal.fovy_degrees) / 2))
        )
        if f_minor > 0:
            obb_minor_m = float(short_px * d_val / f_minor)

    det = ObjectDetection(
        label=label or "selected",
        confidence=score,
        centroid_2d=(cx_det, cy_det),
        mask_area=len(xs),
        depth_meters=d_val,
        position_3d=position_3d,
        orientation_angle=orientation_angle,
        aspect_ratio=aspect_ratio,
        obb_minor_m=obb_minor_m,
    )
    det.mask = mask
    return [det]


def _run_box_prompt(pipeline, rgb, depth, x1, y1, x2, y2, label="", cam_name=None):
    """
    Run SAM3 with a box prompt on the image.

    ``cam_name`` selects the calibration; see _run_point_prompt for why it is
    a parameter rather than a re-read of state.last_cam_name.
    """
    perception = pipeline._perception
    perception.load_models(load_da3=False)
    h, w = rgb.shape[:2]
    pil_img = Image.fromarray(rgb)

    bx = (x1 + x2) / 2.0 / w
    by = (y1 + y2) / 2.0 / h
    bw = abs(x2 - x1) / w
    bh = abs(y2 - y1) / h

    # Stateful predictor: serialize set_image + geometric prompt against
    # concurrent detects (see _run_point_prompt).
    with perception._detect_lock:
        sam3_state = perception._sam3.set_image(pil_img)
        perception._sam3.reset_all_prompts(sam3_state)
        sam3_state = perception._sam3.add_geometric_prompt(
            state=sam3_state,
            box=[bx, by, bw, bh],
            label=True,
        )

    masks = sam3_state.get("masks", torch.tensor([]))
    scores = sam3_state.get("scores", torch.tensor([]))
    if masks.numel() == 0:
        return []

    best_idx = int(scores.argmax())
    score = float(scores[best_idx])
    mask = masks[best_idx].cpu().numpy().squeeze()
    ys, xs = np.where(mask > 0)
    if len(xs) == 0:
        return []

    cx_det, cy_det = float(xs.mean()), float(ys.mean())

    # Backproject with depth -- shared helper, same convention as text
    # detect (see _backproject_prompt_mask).
    position_3d, d_val, cal = _backproject_prompt_mask(
        pipeline, mask, depth, cam_name, h, w
    )

    img_angle_rad, aspect_ratio, short_px, _long_px = _pca_obb(mask > 0)
    orientation_angle = float(img_angle_rad) if aspect_ratio > 1.3 else 0.0

    obb_minor_m = 0.0
    if short_px > 0 and d_val > 0.01 and cal is not None:
        f_minor = (
            cal.fx if cal.fx > 0 else h / (2 * np.tan(np.deg2rad(cal.fovy_degrees) / 2))
        )
        if f_minor > 0:
            obb_minor_m = float(short_px * d_val / f_minor)

    det = ObjectDetection(
        label=label or "selected",
        confidence=score,
        centroid_2d=(cx_det, cy_det),
        mask_area=len(xs),
        depth_meters=d_val,
        position_3d=position_3d,
        orientation_angle=orientation_angle,
        aspect_ratio=aspect_ratio,
        obb_minor_m=obb_minor_m,
    )
    det.mask = mask
    return [det]


@router.get("/api/detections")
async def list_detections():
    """The pending set, so the UI can show and prune what will actually run."""
    dets = state.pending_detections or []
    return {
        "detections": [
            {
                "index": i,
                "label": str(d.label),
                "confidence": float(d.confidence or 0.0),
                "camera": getattr(d, "camera", None),
                "position_3d": (
                    [float(v) for v in d.position_3d]
                    if d.position_3d is not None else None
                ),
            }
            for i, d in enumerate(dets)
        ]
    }


@router.post("/api/detections/drop")
async def drop_detections(req: dict):
    """Remove detections from the pending set by index or by label."""
    dets = list(state.pending_detections or [])
    if not dets:
        return {"ok": False, "error": "no pending detections", "remaining": 0}

    idxs = req.get("indices")
    labels = req.get("labels")
    if idxs is None and labels is None:
        return {"ok": False, "error": "pass indices or labels", "remaining": len(dets)}

    drop = set()
    for i in (idxs or []):
        try:
            i = int(i)
        except (TypeError, ValueError):
            continue
        if 0 <= i < len(dets):
            drop.add(i)
    want = {str(x).lower() for x in (labels or [])}
    for i, d in enumerate(dets):
        if str(d.label).lower() in want:
            drop.add(i)

    kept = [d for i, d in enumerate(dets) if i not in drop]
    state.pending_detections = kept
    logger.info(
        "detections/drop: removed %d of %d (%s); %d remain",
        len(drop), len(dets),
        ", ".join(str(dets[i].label) for i in sorted(drop)) or "-",
        len(kept),
    )
    return {"ok": True, "dropped": len(drop), "remaining": len(kept)}


@router.post("/api/detect_click")
def detect_click(req: ClickDetectRequest):
    """
    Run SAM3 with a box prompt centred on the click point.
    """
    pipeline = state.pipeline
    if pipeline is None:
        return JSONResponse(
            status_code=400, content={"error": "Pipeline not initialized"}
        )

    # ONE snapshot of the whole triple, so a concurrent stream publish cannot
    # pair one camera's RGB with another's depth and a third's extrinsics.
    rgb, depth, cam_name = _snapshot_frame(pipeline)
    if rgb is None:
        return JSONResponse(status_code=500, content={"error": "No captures"})

    h_img, w_img = rgb.shape[:2]
    scale_x = w_img / req.img_width
    scale_y = h_img / req.img_height
    cx = req.x * scale_x
    cy = req.y * scale_y

    with pipeline.activity("detecting"):
        # Try true point prompt first (SAM2 instance predictor path)
        dets = _run_point_prompt(
            pipeline, rgb, depth, cx, cy, req.label, cam_name=cam_name
        )
        # Fall back to box prompt if point prompt returned nothing
        if not dets:
            box_half = 40 * max(scale_x, scale_y)
            x1 = max(0, cx - box_half)
            y1 = max(0, cy - box_half)
            x2 = min(w_img, cx + box_half)
            y2 = min(h_img, cy + box_half)
            dets = _run_box_prompt(
                pipeline, rgb, depth, x1, y1, x2, y2, req.label, cam_name=cam_name
            )
    dets = _merge_into_pending(dets, source="detect_click")
    annotated = create_detection_overlay(rgb, dets)
    # A click has no box, so anchor a small pad around it -- enough for SAM3 to
    # box-prompt with later, small enough not to swallow a neighbour.
    _pad_x = 0.03
    _pad_y = 0.03
    _nx, _ny = req.x / req.img_width, req.y / req.img_height
    saved = _save_anchor_if_requested(
        pipeline,
        req.save_for_task,
        req.label,
        cam_name,
        (max(0.0, _nx - _pad_x), max(0.0, _ny - _pad_y),
         min(1.0, _nx + _pad_x), min(1.0, _ny + _pad_y)),
    )
    return {
        "detections": serialize_detections(dets),
        "annotated_image": annotated,
        "count": len(dets),
        "click": {"x": cx, "y": cy},
        "anchor_saved": saved,
    }


def _save_anchor_if_requested(pipeline, task, label, camera, box_norm):
    """Persist an operator annotation for `task`, or return None.

    Stored NORMALISED so it survives a resolution change, and only when the
    operator explicitly asked -- a one-off rescue click must not silently
    become permanent state. The anchor can later only ADD a box-prompted
    candidate; it can never force a detection past the quality/count gates.
    """
    if not task or not str(task).strip():
        return None
    store = getattr(pipeline, "_annotation_anchors", None)
    if store is None:
        logger.warning("annotation anchor requested but no store on the pipeline")
        return None
    try:
        anchor = store.save(task, label or "object", camera, box_norm)
        return {"task": str(task), **anchor}
    except Exception as exc:  # noqa: BLE001 - never fail a detect over this
        logger.warning("could not save annotation anchor: %s", exc)
        return None


@router.post("/api/detect_box")
def detect_box(req: BoxDetectRequest):
    """
    Run SAM3 with a user-drawn bounding box prompt.
    """
    pipeline = state.pipeline
    if pipeline is None:
        return JSONResponse(
            status_code=400, content={"error": "Pipeline not initialized"}
        )
    rgb, depth, cam_name = _snapshot_frame(pipeline)
    if rgb is None:
        return JSONResponse(status_code=500, content={"error": "No captures"})

    h_img, w_img = rgb.shape[:2]
    sx, sy = w_img / req.img_width, h_img / req.img_height
    x1, y1 = req.x1 * sx, req.y1 * sy
    x2, y2 = req.x2 * sx, req.y2 * sy

    with pipeline.activity("detecting"):
        dets = _run_box_prompt(
            pipeline, rgb, depth, x1, y1, x2, y2, req.label, cam_name=cam_name
        )
    dets = _merge_into_pending(dets, source="detect_box")
    annotated = create_detection_overlay(rgb, dets)
    saved = _save_anchor_if_requested(
        pipeline,
        req.save_for_task,
        req.label,
        cam_name,
        (req.x1 / req.img_width, req.y1 / req.img_height,
         req.x2 / req.img_width, req.y2 / req.img_height),
    )
    return {
        "detections": serialize_detections(dets),
        "annotated_image": annotated,
        "count": len(dets),
        "anchor_saved": saved,
    }


@router.post("/api/detect_approve")
async def detect_for_approval(req: DetectApproveRequest):
    """
    Detect objects and return annotated image for user approval.
    """
    pipeline = state.pipeline
    if pipeline is None:
        return JSONResponse(
            status_code=400, content={"error": "Pipeline not initialized"}
        )
    loop = asyncio.get_event_loop()

    def _detect_sync():
        with pipeline.activity("capturing"):
            captures = pipeline.capture()
        if not captures:
            return None

        # IR swap for low-light
        if req.use_ir:
            for cam_name in ("sideview", "birdview"):
                cd = captures.get(cam_name)
                if cd is None or cd.get("rgb") is None:
                    continue
                target_hw = cd["rgb"].shape[:2]
                try:
                    ir_rgb, _ = streaming.capture_single_camera(
                        cam_name,
                        need_depth=False,
                        ir_as_rgb=True,
                    )
                    if ir_rgb is None:
                        continue
                    if ir_rgb.shape[:2] != target_hw:
                        ir_rgb = cv2.resize(
                            ir_rgb,
                            (target_hw[1], target_hw[0]),
                            interpolation=cv2.INTER_LINEAR,
                        )
                    cd["rgb"] = ir_rgb
                except Exception:
                    pass

        scene_img = None
        scene_cam = None
        for cam_name in ["birdview", "sideview"]:
            cam_data = captures.get(cam_name)
            if cam_data and cam_data.get("rgb") is not None:
                scene_img = cam_data["rgb"]
                scene_cam = cam_name
                break

        # Registry first, LLM second. A registered task has its prompts and
        # its instance ordering pinned in version control, so this both skips
        # a Gemini call and removes the run-to-run prompt drift that
        # generate_prompts introduces at temperature 0.3, which makes a
        # cached BT unsafe.
        spec = pipeline.task_spec(req.instruction)
        prompts = req.prompts
        if prompts is None and spec is not None:
            prompts = spec.prompts
        elif prompts is None and pipeline._planner is not None and scene_img is not None:
            try:
                with pipeline.activity("planning"):
                    prompts = pipeline._planner.generate_prompts(
                        req.instruction,
                        scene_img,
                    )
            except Exception:
                prompts = pipeline._extract_prompts(req.instruction)
        elif prompts is None:
            prompts = pipeline._extract_prompts(req.instruction)

        with pipeline.activity("detecting"):
            # Registered task, prompts not overridden by the operator -> the
            # frozen contract, which means the shared detect_for_task ladder
            # (relax -> alt_prompts -> LLM proposal). Anything else keeps the
            # legacy pass: an operator's own prompt list is an experiment, not
            # a contract to abort on.
            if spec is not None and req.prompts is None:
                merged, all_dets, _res, spec, prompts = pipeline.detect_for_task(
                    captures, req.instruction, spec=spec
                )
            else:
                all_dets = pipeline.detect(captures, prompts, multi_instance=True)
                merged = pipeline.merge_detections(all_dets, spec=spec)

        # Per-camera overlays
        per_cam_overlays = {}
        cam_names_found = []
        for cam_name, cam_data in captures.items():
            rgb_c = cam_data.get("rgb")
            if rgb_c is None:
                continue
            cam_d = [d for d in all_dets if getattr(d, "camera", None) == cam_name]
            if not cam_d:
                continue
            cam_names_found.append(cam_name)
            cam_d_numbered = [copy.copy(d) for d in cam_d]
            lbl_grp = defaultdict(list)
            for d in cam_d_numbered:
                lbl_grp[d.label].append(d)
            for lbl, ds in lbl_grp.items():
                if len(ds) > 1:
                    for idx, d in enumerate(ds):
                        d.label = f"{lbl} {idx + 1}"
            per_cam_overlays[cam_name] = {
                "mask": encode_image_b64(
                    pipeline._annotate_image(rgb_c, cam_d_numbered)
                ),
                "bbox": encode_image_b64(draw_bbox_overlay(rgb_c, cam_d_numbered)),
                "grasp": encode_image_b64(draw_grasp_preview(rgb_c, cam_d_numbered)),
                "count": len(cam_d_numbered),
                "detections": serialize_detections(cam_d_numbered),
            }

        # Scene camera raw detections
        cam_dets = [d for d in all_dets if getattr(d, "camera", None) == scene_cam]
        raw_dets = cam_dets if cam_dets else merged
        raw_dets_numbered = [copy.copy(d) for d in raw_dets]
        raw_grp = defaultdict(list)
        for d in raw_dets_numbered:
            raw_grp[d.label].append(d)
        for lbl, ds in raw_grp.items():
            if len(ds) > 1:
                for idx, d in enumerate(ds):
                    d.label = f"{lbl} {idx + 1}"

        mask_img = pipeline._annotate_image(scene_img, raw_dets_numbered)
        bbox_img = draw_bbox_overlay(scene_img, raw_dets_numbered)
        grasp_img = draw_grasp_preview(scene_img, raw_dets_numbered)

        # Merged overlays with cross-camera projection
        scene_cal = (
            captures.get(scene_cam, {}).get("calibration") if scene_cam else None
        )
        merged_for_overlay = _build_merged_overlay(
            merged,
            scene_cam,
            scene_cal,
            scene_img,
        )
        mask_img_merged = pipeline._annotate_image(
            scene_img,
            merged_for_overlay,
        )
        bbox_img_merged = draw_bbox_overlay(scene_img, merged_for_overlay)
        grasp_img_merged = draw_grasp_preview(scene_img, merged_for_overlay)

        # Depth diagnostics
        depth_diag = _build_depth_diag(captures, all_dets)

        return {
            "prompts": prompts,
            "detections": serialize_detections(merged),
            "raw_detections": serialize_detections(raw_dets_numbered),
            "raw_count": len(raw_dets_numbered),
            "merged_count": len(merged),
            "mask_image": encode_image_b64(mask_img),
            "bbox_image": encode_image_b64(bbox_img),
            "grasp_image": encode_image_b64(grasp_img),
            "mask_merged": encode_image_b64(mask_img_merged),
            "bbox_merged": encode_image_b64(bbox_img_merged),
            "grasp_merged": encode_image_b64(grasp_img_merged),
            "per_cam": per_cam_overlays,
            "cam_names": cam_names_found,
            "captures": captures,
            "all_detections": all_dets,
            "merged": merged,
            "depth_diag": depth_diag,
        }

    try:
        result = await loop.run_in_executor(None, _detect_sync)
    except PromptCountMismatch as exc:
        # Same contract as /api/plan: the task's own on_mismatch policy said
        # abort, so the operator gets the counts, not a green approval screen
        # with the object silently absent.
        return JSONResponse(
            status_code=422,
            content={
                "error": str(exc),
                "instruction": req.instruction,
                "reason": "prompt_count_mismatch",
            },
        )
    if result is None:
        return JSONResponse(status_code=500, content={"error": "No captures"})

    state.pending_detections = result["merged"]
    state.pending_captures = result["captures"]
    state.pending_instruction = req.instruction
    state.pending_prompts = result["prompts"]

    return {
        k: v
        for k, v in result.items()
        if k not in ("captures", "all_detections", "merged")
    }


def _build_merged_overlay(merged, scene_cam, scene_cal, scene_img):
    """
    Build merged overlay list, projecting cross-camera detections.
    """
    merged_for_overlay = []
    for d in merged:
        if getattr(d, "camera", None) == scene_cam:
            merged_for_overlay.append(d)
        elif d.position_3d is not None and scene_cal is not None:
            try:
                pos = np.array(d.position_3d[:3])
                R = scene_cal.extrinsic[:3, :3]
                t = scene_cal.extrinsic[:3, 3]
                p_cam = R.T @ (pos - t)
                if p_cam[2] > 0.01:
                    u = scene_cal.fx * p_cam[0] / p_cam[2] + scene_cal.cx
                    v = scene_cal.fy * p_cam[1] / p_cam[2] + scene_cal.cy
                    h_img, w_img = scene_img.shape[:2]
                    if 0 <= u < w_img and 0 <= v < h_img:
                        proj = copy.copy(d)
                        proj.centroid_2d = (float(u), float(v))
                        proj.mask = None
                        merged_for_overlay.append(proj)
            except Exception:
                pass
        else:
            merged_for_overlay.append(d)
    return merged_for_overlay


def _build_depth_diag(captures, all_dets):
    """
    Build per-camera depth diagnostics for the detect_approve response.
    """
    depth_diag = {}
    for cam_name, cam_data in captures.items():
        depth_arr = cam_data.get("depth")
        cal = cam_data.get("calibration")
        if depth_arr is None or cal is None:
            continue
        depth_m = depth_arr.astype(np.float32)
        if depth_m.max() > 100.0:
            depth_m = depth_m / 1000.0
        H, W = depth_m.shape
        R_cw = cal.extrinsic[:3, :3]
        t_cw = cal.extrinsic[:3, 3]

        # Table samples at canonical positions
        table_samples = []
        for uv_label, uu, vv in [
            ("center", W // 2, H // 2),
            ("upper-left", W // 4, H // 4),
            ("lower-right", 3 * W // 4, 3 * H // 4),
        ]:
            if not (0 <= vv < H and 0 <= uu < W):
                continue
            dz = float(depth_m[vv, uu])
            if not (0.05 < dz < 5.0):
                continue
            xc = (uu - cal.cx) * dz / cal.fx
            yc = (vv - cal.cy) * dz / cal.fy
            pw = R_cw @ np.array([xc, yc, dz]) + t_cw
            table_samples.append(
                {
                    "where": uv_label,
                    "u": uu,
                    "v": vv,
                    "depth_m": dz,
                    "world_z_m": float(pw[2]),
                }
            )

        per_det = []
        cam_d = [d for d in all_dets if getattr(d, "camera", None) == cam_name]
        for d in cam_d:
            m = d.mask
            if m is None or m.shape != depth_m.shape:
                continue
            valid = m & (depth_m > 0.05) & (depth_m < 5.0)
            n = int(valid.sum())
            if n < 20:
                continue
            ys, xs = np.where(valid)
            zs = depth_m[ys, xs]
            xc = (xs - cal.cx) * zs / cal.fx
            yc = (ys - cal.cy) * zs / cal.fy
            pts_cam = np.stack([xc, yc, zs], axis=1)
            pts_world = pts_cam @ cal.extrinsic[:3, :3].T + cal.extrinsic[:3, 3]
            wz = pts_world[:, 2]
            pcts = np.percentile(wz, [5, 25, 50, 75, 95]).tolist()
            per_det.append(
                {
                    "label": d.label,
                    "n_pixels": n,
                    "world_z_p25_m": pcts[1],
                    "world_z_p50_m": pcts[2],
                    "world_z_range_m": float(wz.max() - wz.min()),
                }
            )

        depth_diag[cam_name] = {
            "depth_colormap": encode_image_b64(depth_to_colormap(depth_arr)),
            "table_samples": table_samples,
            "detections": per_det,
        }
    return depth_diag

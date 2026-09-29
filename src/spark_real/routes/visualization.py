"""
Image annotation and overlay utilities for detection visualization.
"""

import base64
import cv2
import io
import logging

import numpy as np
from PIL import Image, ImageDraw
import scipy.ndimage as ndi
from spark_real.utils.fonts import load_font

try:
    import matplotlib.cm as cm
except ImportError:
    cm = None

logger = logging.getLogger("spark_server")

DETECTION_COLORS = [
    (255, 60, 60),
    (60, 220, 120),
    (60, 120, 255),
    (255, 200, 40),
    (220, 60, 220),
    (60, 220, 220),
    (255, 140, 40),
    (140, 255, 40),
    (40, 140, 255),
]


def encode_image_b64(img_array: np.ndarray, quality: int = 88) -> str:
    """
    Encode numpy RGB array to base64 JPEG string.
    """
    pil = Image.fromarray(img_array)
    buf = io.BytesIO()
    pil.save(buf, format="JPEG", quality=quality)
    return base64.b64encode(buf.getvalue()).decode()


def encode_image(arr: np.ndarray, quality: int = 85) -> str:
    """
    Encode numpy RGB array to base64 JPEG (for API responses).
    """
    return encode_image_b64(arr, quality=quality)


def depth_to_colormap(depth: np.ndarray) -> np.ndarray:
    """
    Convert depth map to turbo-colormap RGB image.
    """
    mask = depth > 0
    if not mask.any():
        return np.zeros((*depth.shape, 3), dtype=np.uint8)
    vmin, vmax = np.percentile(depth[mask], [5, 95])
    norm = np.clip((depth - vmin) / (vmax - vmin + 1e-6), 0, 1)
    norm[~mask] = 0
    colored = (cm.turbo(norm)[:, :, :3] * 255).astype(np.uint8)
    colored[~mask] = 0
    return colored


def draw_crosshair(img: np.ndarray, cx: int, cy: int, color: tuple, size: int = 10):
    """
    Draw a crosshair at (cx, cy) on an image array.
    """
    h, w = img.shape[:2]
    for d in range(-size, size + 1):
        if 0 <= cy < h and 0 <= cx + d < w:
            img[cy, cx + d] = color
        if 0 <= cy + d < h and 0 <= cx < w:
            img[cy + d, cx] = color


def draw_contour(img: np.ndarray, mask: np.ndarray, color: tuple, thickness: int = 2):
    """
    Draw contour outline around a binary mask.
    """
    eroded = ndi.binary_erosion(mask, iterations=thickness)
    border = mask & ~eroded
    img[border] = color


def _pca_obb_for_draw(mask: np.ndarray):
    """
    Image-frame PCA OBB for visualization. Mirrors the executor's
    perception._pca_obb. Returns
    (centroid_uv, major_dir_uv, minor_dir_uv, major_len_px, minor_len_px, angle_rad)
    or None on degenerate masks.
    """
    ys, xs = np.where(mask > 0)
    if len(xs) < 5:
        return None
    pts = np.column_stack([xs.astype(np.float64), ys.astype(np.float64)])
    mean = pts.mean(axis=0)
    centered = pts - mean
    cov = np.cov(centered, rowvar=False)
    if cov.ndim == 0 or cov.shape != (2, 2):
        return None
    eigvals, eigvecs = np.linalg.eigh(cov)
    major_dir = eigvecs[:, 1]
    minor_dir = eigvecs[:, 0]
    proj_major = centered @ major_dir
    proj_minor = centered @ minor_dir
    long_px = float(proj_major.max() - proj_major.min())
    short_px = float(proj_minor.max() - proj_minor.min())
    if short_px <= 0 or long_px <= 0:
        return None
    angle = float(np.arctan2(major_dir[1], major_dir[0]))
    return mean, major_dir, minor_dir, long_px, short_px, angle


def _draw_obb(
    img: np.ndarray,
    mask: np.ndarray,
    color: tuple,
    thickness: int = 2,
    label_axes: bool = True,
):
    """
    Draw the PCA-derived OBB of a mask plus labeled major/minor axes.

    Uses the same PCA the executor uses. Returns the 4 box corners.
    """
    obb = _pca_obb_for_draw(mask)
    if obb is None:
        return None
    centroid, mj, mn, mj_len, mn_len, angle = obb

    # OBB rectangle from PCA axes (matches the executor's idea of "OBB").
    box = []
    for sx in (-1, 1):
        for sy in (-1, 1):
            box.append(centroid + sx * (mj_len / 2) * mj + sy * (mn_len / 2) * mn)
    box = np.array([box[0], box[1], box[3], box[2]], dtype=np.int32)
    cv2.drawContours(img, [box], 0, color, thickness)

    if label_axes:
        # Major axis (longest direction): red arrow.
        mj_a = (centroid - (mj_len / 2) * mj).astype(int)
        mj_b = (centroid + (mj_len / 2) * mj).astype(int)
        cv2.arrowedLine(
            img, tuple(mj_a), tuple(mj_b), (255, 0, 0), thickness + 1, tipLength=0.06
        )
        # Minor axis (shortest direction): cyan arrow.
        mn_a = (centroid - (mn_len / 2) * mn).astype(int)
        mn_b = (centroid + (mn_len / 2) * mn).astype(int)
        cv2.arrowedLine(
            img, tuple(mn_a), tuple(mn_b), (0, 220, 220), thickness + 1, tipLength=0.10
        )
        ar = mj_len / mn_len if mn_len > 0 else 0.0
        ang_deg = np.degrees(angle)
        # Label: major endpoint
        text = f"major  {ang_deg:+.0f}deg  AR={ar:.2f}"
        tx, ty = int(mj_b[0]) + 6, int(mj_b[1]) - 4
        # Black halo + colored text for readability over masks
        cv2.putText(
            img,
            text,
            (tx, ty),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.55,
            (0, 0, 0),
            4,
            cv2.LINE_AA,
        )
        cv2.putText(
            img,
            text,
            (tx, ty),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.55,
            (255, 80, 80),
            1,
            cv2.LINE_AA,
        )
        # Minor label
        mtx, mty = int(mn_b[0]) + 6, int(mn_b[1]) - 4
        cv2.putText(
            img,
            "minor",
            (mtx, mty),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.5,
            (0, 0, 0),
            4,
            cv2.LINE_AA,
        )
        cv2.putText(
            img,
            "minor",
            (mtx, mty),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.5,
            (50, 230, 230),
            1,
            cv2.LINE_AA,
        )

    return box


def draw_detections_on_image(rgb: np.ndarray, detections: list) -> np.ndarray:
    """
    Draw clean mask overlays and labels on an image.

    Shows ONLY:
      - Semi-transparent colored mask with contour outline
      - Label text with black outline for readability

    No bounding boxes, axes, arrows, crosshairs, or coordinates.
    Use ``draw_bbox_overlay`` or ``draw_grasp_preview`` for those.
    """
    vis = rgb.copy()
    for i, det in enumerate(detections):
        c = np.array(DETECTION_COLORS[i % len(DETECTION_COLORS)])
        c_tuple = tuple(c.astype(int).tolist())
        mask = getattr(det, "mask", None)
        if mask is not None and mask.shape == vis.shape[:2]:
            vis[mask > 0] = (vis[mask > 0] * 0.55 + c * 0.45).astype(np.uint8)
            draw_contour(vis, mask, c_tuple, thickness=2)

    pil = Image.fromarray(vis)
    draw = ImageDraw.Draw(pil)
    font = load_font(14)
    for i, det in enumerate(detections):
        cx, cy = det.centroid_2d if det.centroid_2d else (0, 0)
        c = DETECTION_COLORS[i % len(DETECTION_COLORS)]
        label_text = f"{det.label} {det.confidence:.0%}"
        tx, ty = int(cx) + 14, int(cy) - 8
        # Black outline for readability against busy backgrounds
        for dx in (-1, 0, 1):
            for dy in (-1, 0, 1):
                if dx == 0 and dy == 0:
                    continue
                draw.text((tx + dx, ty + dy), label_text, fill=(0, 0, 0), font=font)
        draw.text((tx, ty), label_text, fill=c, font=font)
    return np.array(pil)


def create_detection_overlay(rgb: np.ndarray, detections: list) -> str:
    """
    Draw detection overlays and return base64 JPEG.
    """
    vis = draw_detections_on_image(rgb, detections)
    buf = io.BytesIO()
    Image.fromarray(vis).save(buf, format="JPEG", quality=88)
    return base64.b64encode(buf.getvalue()).decode()


def draw_bbox_overlay(rgb: np.ndarray, detections: list) -> np.ndarray:
    """
    Draw oriented bounding boxes on image.
    """
    vis = rgb.copy()
    for i, det in enumerate(detections):
        c = DETECTION_COLORS[i % len(DETECTION_COLORS)]
        mask = getattr(det, "mask", None)
        if mask is not None and mask.shape == vis.shape[:2]:
            _draw_obb(vis, mask, c, thickness=2)
        elif det.bbox:
            # Fallback to axis-aligned if no mask available
            x1, y1, x2, y2 = [int(v) for v in det.bbox]
            cv2.rectangle(vis, (x1, y1), (x2, y2), c, 2)

    pil = Image.fromarray(vis)
    draw = ImageDraw.Draw(pil)
    font = load_font(16)
    for i, det in enumerate(detections):
        cx, cy = det.centroid_2d if det.centroid_2d else (0, 0)
        c = DETECTION_COLORS[i % len(DETECTION_COLORS)]
        label_text = f"{det.label} {det.confidence:.0%}"
        tx, ty = int(cx) + 14, int(cy) - 10
        bbox_t = draw.textbbox((tx, ty), label_text, font=font)
        draw.rectangle(
            [bbox_t[0] - 2, bbox_t[1] - 1, bbox_t[2] + 2, bbox_t[3] + 1],
            fill=(0, 0, 0, 200),
        )
        draw.text((tx, ty), label_text, fill=c, font=font)
    return np.array(pil)


def draw_grasp_preview(rgb: np.ndarray, detections: list) -> np.ndarray:
    """
    Draw gripper jaw previews closing perpendicular to the OBB's longest side.
    """
    vis = rgb.copy()

    for i, det in enumerate(detections):
        cx, cy = det.centroid_2d if det.centroid_2d else (0, 0)
        cx, cy = int(cx), int(cy)
        c = DETECTION_COLORS[i % len(DETECTION_COLORS)]
        mask = getattr(det, "mask", None)

        # Compute major axis from mask OBB (image space)
        img_angle = 0.0
        ar = 1.0
        if mask is not None:
            contours, _ = cv2.findContours(
                mask.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
            )
            if contours:
                largest = max(contours, key=cv2.contourArea)
                if len(largest) >= 5:
                    rect = cv2.minAreaRect(largest)
                    box = cv2.boxPoints(rect)
                    s1 = box[1] - box[0]
                    s2 = box[2] - box[1]
                    l1, l2 = np.linalg.norm(s1), np.linalg.norm(s2)
                    major = s1 if l1 >= l2 else s2
                    long, short = max(l1, l2), min(l1, l2)
                    if short > 0:
                        ar = long / short
                    img_angle = float(np.arctan2(major[1], major[0]))

        if ar <= 1.5:
            continue  # no oriented grasp for roughly square objects

        # Major axis direction in image
        mx, my = np.cos(img_angle), np.sin(img_angle)
        # Perpendicular = gripper approach direction (jaws close across)
        px, py = -my, mx

        jaw_len = 30
        jaw_width = 8
        gap = 18

        # Draw jaws on opposite sides perpendicular to major axis
        for side in [-1, 1]:
            jx = cx + side * px * (gap + jaw_width // 2)
            jy = cy + side * py * (gap + jaw_width // 2)
            corners = []
            for sx, sy in [(-1, -1), (1, -1), (1, 1), (-1, 1)]:
                rx = jx + sx * mx * jaw_len // 2 + sy * px * jaw_width // 2
                ry = jy + sx * my * jaw_len // 2 + sy * py * jaw_width // 2
                corners.append((rx, ry))
            cv2.fillPoly(vis, [np.array(corners, dtype=np.int32)], (30, 30, 30))
            cv2.polylines(vis, [np.array(corners, dtype=np.int32)], True, c, 1)

    # Labels
    pil = Image.fromarray(vis)
    draw = ImageDraw.Draw(pil)
    font = load_font(14)
    for i, det in enumerate(detections):
        cx, cy = det.centroid_2d if det.centroid_2d else (0, 0)
        c = DETECTION_COLORS[i % len(DETECTION_COLORS)]
        label_text = f"{det.label}"
        tx, ty = int(cx) + 14, int(cy) - 10
        bbox_t = draw.textbbox((tx, ty), label_text, font=font)
        draw.rectangle(
            [bbox_t[0] - 2, bbox_t[1] - 1, bbox_t[2] + 2, bbox_t[3] + 1],
            fill=(0, 0, 0, 200),
        )
        draw.text((tx, ty), label_text, fill=c, font=font)
    return np.array(pil)


def project_3d_to_camera(pos_3d, calibration) -> tuple:
    """
    Project a 3D world point into a camera's image plane.

    Uses camera convention: x=right, y=up, z=backward (OpenGL-like).
    Returns (u, v) pixel coords or None if behind camera.
    """
    R = calibration.rotation_matrix
    t = calibration.position
    w, h = calibration.width, calibration.height
    f = h / (2 * np.tan(np.deg2rad(calibration.fovy_degrees) / 2))

    p_cam = R.T @ (np.asarray(pos_3d) - t)
    depth = -p_cam[2]
    if depth <= 0.01:
        return None

    u = (p_cam[0] / depth) * f + w / 2
    v = (-p_cam[1] / depth) * f + h / 2

    margin = 50
    if u < -margin or u > w + margin or v < -margin or v > h + margin:
        return None
    return (float(u), float(v))


def create_tiled_detection_overlay(captures: dict, detections: list) -> tuple:
    """
    Build tiled side-by-side image with detections on all camera views.

    For each detection with a 3D position, projects into every camera view
    using calibrated extrinsics. Source camera gets full mask overlay;
    other cameras get a projected crosshair + label.

    Returns (base64_jpeg, tiled_bboxes) where tiled_bboxes maps
    label -> [x1, y1, x2, y2] in tiled image pixel coords.
    """
    target_h = 480
    cam_order = ["sideview", "birdview", "wrist"]

    # Draw source-camera masks
    cam_images = {}
    for cam_name in cam_order:
        data = captures.get(cam_name)
        if data is None:
            continue
        rgb = data["rgb"].copy()
        cam_dets = [d for d in detections if getattr(d, "camera", None) == cam_name]
        if cam_dets:
            rgb = draw_detections_on_image(rgb, cam_dets)
        cam_images[cam_name] = rgb

    # Project detections into non-source cameras
    for i, det in enumerate(detections):
        if det.position_3d is None:
            continue
        pos = (
            det.position_3d
            if isinstance(det.position_3d, np.ndarray)
            else np.array(det.position_3d)
        )
        c = DETECTION_COLORS[i % len(DETECTION_COLORS)]
        source_cam = getattr(det, "camera", None)

        for cam_name in cam_order:
            if cam_name == source_cam:
                continue
            data = captures.get(cam_name)
            if data is None or cam_name not in cam_images:
                continue
            cal = data["calibration"]
            if np.allclose(cal.extrinsic, np.eye(4)):
                continue

            proj = project_3d_to_camera(pos, cal)
            if proj is None:
                continue

            u, v = int(proj[0]), int(proj[1])
            rgb = cam_images[cam_name]
            draw_crosshair(rgb, u, v, c, size=15)

            pil = Image.fromarray(rgb)
            draw = ImageDraw.Draw(pil)
            font = load_font(13)
            label_text = f"{det.label} {det.confidence:.0%}"
            tx, ty = u + 16, v - 7
            # Black outline for readability
            for dx in (-1, 0, 1):
                for dy in (-1, 0, 1):
                    if dx == 0 and dy == 0:
                        continue
                    draw.text((tx + dx, ty + dy), label_text, fill=(0, 0, 0), font=font)
            draw.text((tx, ty), label_text, fill=c, font=font)
            cam_images[cam_name] = np.array(pil)

    # Build tiles
    tiles = []
    tile_offsets = {}
    x_offset = 0
    for cam_name in cam_order:
        if cam_name in cam_images:
            rgb = cam_images[cam_name]
            orig_h, orig_w = rgb.shape[:2]
            new_w = int(orig_w * target_h / orig_h)
            tile = np.array(
                Image.fromarray(rgb).resize((new_w, target_h), Image.BILINEAR)
            )
            tiles.append(tile)
            tile_offsets[cam_name] = (
                x_offset,
                new_w / orig_w,
                target_h / orig_h,
                orig_w,
                orig_h,
            )
            x_offset += new_w
        else:
            pw = int(target_h * 4 / 3)
            tiles.append(np.zeros((target_h, pw, 3), dtype=np.uint8))
            x_offset += pw

    tiled_bboxes = {}
    for det in detections:
        cam = getattr(det, "camera", None)
        if cam and cam in tile_offsets and det.bbox:
            xo, sx, sy, _, _ = tile_offsets[cam]
            x1, y1, x2, y2 = det.bbox
            tiled_bboxes[det.label] = [xo + x1 * sx, y1 * sy, xo + x2 * sx, y2 * sy]

    combined = np.hstack(tiles)
    img = Image.fromarray(combined)
    buf = io.BytesIO()
    img.save(buf, format="JPEG", quality=88)
    return base64.b64encode(buf.getvalue()).decode(), tiled_bboxes


def serialize_detections(detections: list) -> list:
    """
    Convert ObjectDetection list to JSON-serializable dicts.
    """
    out = []
    for d in detections:
        # Serialize slots if present (container detections only).
        # Each slot's world_xyz is a numpy array, converted to list for JSON.
        raw_slots = getattr(d, "slots", None)
        slots = None
        if raw_slots:
            slots = []
            for s in raw_slots:
                slot_copy = dict(s)
                wxyz = slot_copy.get("world_xyz")
                if isinstance(wxyz, np.ndarray):
                    slot_copy["world_xyz"] = wxyz.tolist()
                slots.append(slot_copy)
        out.append(
            {
                "label": d.label,
                "confidence": float(d.confidence),
                "centroid_2d": list(d.centroid_2d) if d.centroid_2d else None,
                "position_3d": (
                    d.position_3d.tolist()
                    if isinstance(d.position_3d, np.ndarray)
                    else d.position_3d
                ),
                "depth_meters": float(d.depth_meters),
                "mask_area": int(d.mask_area),
                "bbox": list(d.bbox) if d.bbox else None,
                "camera": getattr(d, "camera", None),
                "orientation_angle": float(d.orientation_angle),
                "aspect_ratio": float(d.aspect_ratio),
                "obb_minor_m": float(getattr(d, "obb_minor_m", 0.0) or 0.0),
                # Colour-free role names published by the task prompt
                # registry (configs/tasks/*.yaml), e.g. "same color block 1".
                # A cached BT can address these instead of a colour word, so
                # exposing them lets the UI and the saved run traces show
                # which physical object a BT keypoint will actually bind to.
                "role_labels": list(getattr(d, "role_labels", None) or []) or None,
                # (hue, saturation, value) median over the eroded mask, or
                # None when colour extraction is off (the default).
                "hsv_median": (
                    list(getattr(d, "hsv_median", None))
                    if getattr(d, "hsv_median", None) is not None
                    else None
                ),
                # Slot detection fields (None for non-container detections).
                # Needed by skills.primitives.place_in_slot to rotate the
                # gripper yaw at release; without these the placement uses
                # whatever yaw the gripper had at grasp time.
                "slots": slots,
                "world_major_axis_rad": (
                    float(getattr(d, "world_major_axis_rad", None))
                    if getattr(d, "world_major_axis_rad", None) is not None
                    else None
                ),
            }
        )
    return out

#!/usr/bin/env python3
"""Standalone detection CLI. No server needed.

Captures from ZED Mini, runs SAM3 detection, outputs 3D positions in both
arm frames. Positions are in JAW-TIP frame (calibration used jaw tips).

Usage:
    cd ~/spark/src

    # Basic detection
    python src/scripts/detect.py "left sleeve" "right sleeve" "hem"

    # With annotated image output
    python src/scripts/detect.py "left sleeve" "right sleeve" --annotate

    # With grab points (20mm inset from mask edge)
    python src/scripts/detect.py "left sleeve" "right sleeve" --grab

    # Save masks as PNGs
    python src/scripts/detect.py "left sleeve" --masks

    # From a saved image instead of live ZED
    python src/scripts/detect.py "left sleeve" --image /path/to/rgb.png --depth /path/to/depth.npy

    # JSON output for piping to other scripts
    python src/scripts/detect.py "left sleeve" --json
"""
import argparse
import json
import os
import sys
import time

import cv2
import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

CAL_PATH = os.path.expanduser("~/.spark_real/calibration_bimanual.json")
OUT_DIR = os.path.expanduser("~/.spark_real/detections")


def load_calibration():
    cal = json.loads(open(CAL_PATH).read())
    T_r = np.array(cal["arms"]["right"]["T_zed_to_base"]).reshape(4, 4)
    T_l = np.array(cal["arms"]["left"]["T_zed_to_base"]).reshape(4, 4)
    return T_r, T_l


def right_to_left(pos_r, T_r, T_l):
    pt_zed = (np.linalg.inv(T_r) @ np.append(pos_r, 1.0))[:3]
    return (T_l @ np.append(pt_zed, 1.0))[:3]


def left_to_right(pos_l, T_r, T_l):
    pt_zed = (np.linalg.inv(T_l) @ np.append(pos_l, 1.0))[:3]
    return (T_r @ np.append(pt_zed, 1.0))[:3]


def capture_zed():
    from spark_real.perception.zed import ZEDMiniCamera
    # HD2K: utensil handles are ~25px at 720p, too coarse for clean
    # handle splits and minor-axis fits. Intrinsics come from the SDK for
    # the active resolution; base_T_cam is resolution-independent.
    cam = ZEDMiniCamera(width=2208, height=1242, depth_mode="NEURAL")
    cam.open()
    # HARD FAIL on the sim-mode fallback: a wedged/absent ZED otherwise
    # serves synthetic frames and the whole pipeline runs on nothing.
    if getattr(cam, "_sim_mode", False):
        cam.close()
        print("ERROR: ZED failed to open (sim-mode fallback); REFUSING to "
              "capture fake frames. Replug the camera USB if the SDK "
              "device list is empty.", file=sys.stderr)
        sys.exit(3)
    time.sleep(0.5)
    rgb, depth = cam.read(depth=True)
    config = cam.config
    cam.close()
    if rgb is None:
        print("ERROR: ZED capture failed", file=sys.stderr)
        sys.exit(1)
    return rgb, depth, config


def run_sam3(rgb, depth, prompts, intrinsic_matrix, T_zed_to_base):
    from spark_real.perception.spark_perception import SPARKPerception
    perc = SPARKPerception(sam3_threshold=0.5)
    R = T_zed_to_base[:3, :3]
    t = T_zed_to_base[:3, 3]
    h, w = rgb.shape[:2]
    fovy = float(2 * np.degrees(np.arctan(h / (2 * intrinsic_matrix[1, 1]))))
    detections = perc._detect_with_rendered_depth(
        rgb=rgb, depth=depth, prompts=prompts,
        cam_pos=t, cam_mat=R, cam_fovy=fovy, w=w, h=h,
        intrinsic_matrix=intrinsic_matrix,
        table_height=-0.08,
    )
    return detections


class _PointDetection:
    """
    Minimal detection result from a point prompt.
    """
    def __init__(self, label, mask, confidence, centroid_2d, position_3d):
        self.label = label
        self.mask = mask
        self.confidence = confidence
        self.centroid_2d = centroid_2d
        self.position_3d = position_3d


def run_sam3_points(rgb, depth, point_specs, intrinsic_matrix, T_zed_to_base):
    """
    Run SAM3 with point prompts. point_specs: list of (label, x, y).
    """
    import torch
    from PIL import Image
    from spark_real.perception.spark_perception import SPARKPerception
    perc = SPARKPerception(sam3_threshold=0.5)
    perc.load_models()
    pil_img = Image.fromarray(rgb)
    h, w = rgb.shape[:2]

    with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
        base_state = perc._sam3.set_image(pil_img)

    detections = []
    for label, px, py in point_specs:
        with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
            state = {**base_state}
            if "backbone_out" in state and "language_features" not in state["backbone_out"]:
                dummy_text = perc._sam3.model.backbone.forward_text(
                    ["visual"], device=perc._sam3.device)
                state["backbone_out"].update(dummy_text)
            if "geometric_prompt" not in state:
                state["geometric_prompt"] = perc._sam3.model._get_dummy_prompt()
            pt = torch.tensor([px / w, py / h], device=perc._sam3.device,
                              dtype=torch.float32).view(1, 1, 2)
            lbl = torch.tensor([True], device=perc._sam3.device,
                               dtype=torch.bool).view(1, 1)
            state["geometric_prompt"].append_points(pt, lbl)
            state = perc._sam3._forward_grounding(state)

        masks = state.get('masks', torch.tensor([]))
        scores = state.get('scores', torch.tensor([]))
        if masks.numel() == 0:
            continue
        best_idx = int(scores.argmax())
        mask = masks[best_idx].cpu().numpy().squeeze().astype(bool)
        conf = float(scores[best_idx])
        ys, xs = np.where(mask)
        if len(xs) == 0:
            continue
        cx, cy = float(xs.mean()), float(ys.mean())
        pos = _backproject_pixel(cx, cy, depth, intrinsic_matrix, T_zed_to_base)
        detections.append(_PointDetection(
            label=label, mask=mask, confidence=conf,
            centroid_2d=(cx, cy), position_3d=pos))
    return detections


def _backproject_pixel(u, v, depth, intrinsic_matrix, T_zed_to_base):
    """
    Backproject a single pixel to world (RIGHT arm) frame.
    """
    fx = intrinsic_matrix[0, 0]
    fy = intrinsic_matrix[1, 1]
    cx_k = intrinsic_matrix[0, 2]
    cy_k = intrinsic_matrix[1, 2]
    h, w = depth.shape[:2]
    ui, vi = int(round(u)), int(round(v))
    ui = max(0, min(ui, w - 1))
    vi = max(0, min(vi, h - 1))
    d_val = float(depth[vi, ui])
    if d_val < 0.01:
        patch = depth[max(0, vi-5):vi+5, max(0, ui-5):ui+5]
        valid = patch[(patch > 0.01) & (patch < 10)]
        if len(valid) > 0:
            d_val = float(np.median(valid))
        else:
            return None
    x_cam = (u - cx_k) * d_val / fx
    y_cam = (v - cy_k) * d_val / fy
    z_cam = d_val
    R = T_zed_to_base[:3, :3]
    t = T_zed_to_base[:3, 3]
    return R @ np.array([x_cam, y_cam, z_cam]) + t


def _backproject_with_depth(u, v, d_val, intrinsic_matrix, T_zed_to_base):
    """Backproject a pixel using an explicit depth (meters), not the
    per-pixel depth. Used where a single edge pixel can read through a gap
    (e.g. the collar neckline) and a band-median depth is more reliable."""
    fx = intrinsic_matrix[0, 0]
    fy = intrinsic_matrix[1, 1]
    cx_k = intrinsic_matrix[0, 2]
    cy_k = intrinsic_matrix[1, 2]
    x_cam = (u - cx_k) * d_val / fx
    y_cam = (v - cy_k) * d_val / fy
    R = T_zed_to_base[:3, :3]
    t = T_zed_to_base[:3, 3]
    return R @ np.array([x_cam, y_cam, d_val]) + t


def compute_grab_points(mask, depth, intrinsic_matrix, T_zed_to_base, inset_m=0.015):
    """
    Compute grab point at mask centroid, inset from nearest edge.
    """
    ys, xs = np.where(mask > 0)
    if len(xs) < 10:
        return None
    cx_px, cy_px = float(xs.mean()), float(ys.mean())
    dist_map = cv2.distanceTransform(mask.astype(np.uint8), cv2.DIST_L2, 5)
    fx = intrinsic_matrix[0, 0]
    d_val = float(depth[int(cy_px), int(cx_px)]) if depth is not None else 0.5
    if d_val < 0.01:
        valid = depth[mask > 0]
        valid = valid[(valid > 0.01) & (valid < 10)]
        d_val = float(np.median(valid)) if len(valid) > 0 else 0.5
    inset_px = inset_m * fx / d_val
    if dist_map[int(cy_px), int(cx_px)] > inset_px:
        grab_u, grab_v = cx_px, cy_px
    else:
        max_dist_idx = np.argmax(dist_map[mask > 0])
        grab_v = float(ys[max_dist_idx])
        grab_u = float(xs[max_dist_idx])
    return _backproject_pixel(grab_u, grab_v, depth, intrinsic_matrix, T_zed_to_base)


def compute_hem_edges(mask, depth, intrinsic_matrix, T_zed_to_base, inset_m=0.020):
    """Compute left and right edge grab points of a hem mask.

    In image space: left edge = min x, right edge = max x.
    Inset is applied in y of the respective arm frame (not image space).

    Returns dict with 'left_edge' and 'right_edge' as world positions,
    or None for edges that can't be computed.
    """
    ys, xs = np.where(mask > 0)
    if len(xs) < 10:
        return {"left_edge": None, "right_edge": None}

    fx = intrinsic_matrix[0, 0]
    d_val = float(depth[int(ys.mean()), int(xs.mean())]) if depth is not None else 0.5
    if d_val < 0.01:
        valid = depth[mask > 0]
        valid = valid[(valid > 0.01) & (valid < 10)]
        d_val = float(np.median(valid)) if len(valid) > 0 else 0.5
    inset_px = inset_m * fx / d_val

    x_min, x_max = int(xs.min()), int(xs.max())

    left_col = x_min + int(inset_px)
    left_col = min(left_col, x_max)
    left_rows = ys[xs == min(xs[xs >= left_col])] if np.any(xs >= left_col) else ys[xs == x_min]
    left_v = float(left_rows.mean())
    left_u = float(left_col)

    right_col = x_max - int(inset_px)
    right_col = max(right_col, x_min)
    right_rows = ys[xs == max(xs[xs <= right_col])] if np.any(xs <= right_col) else ys[xs == x_max]
    right_v = float(right_rows.mean())
    right_u = float(right_col)

    left_pt = _backproject_pixel(left_u, left_v, depth, intrinsic_matrix, T_zed_to_base)
    right_pt = _backproject_pixel(right_u, right_v, depth, intrinsic_matrix, T_zed_to_base)

    return {
        "left_edge": left_pt,
        "right_edge": right_pt,
        "left_px": (left_u, left_v),
        "right_px": (right_u, right_v),
    }


def compute_sleeve_outer_edge(mask, ref_px, depth, intrinsic_matrix,
                              T_zed_to_base, inset_m=0.030):
    """Outer-edge grab point of a sleeve mask.

    The fold pinches the sleeve TIP, not the centroid: take the band of mask
    pixels FARTHEST from ``ref_px`` (the shirt-body centroid in pixels), then
    pull the point back toward the body by ``inset_m`` so the pinch lands on
    cloth rather than the very edge. Returns (world_point, (u, v)) in the
    primary (right-arm) frame, or (None, None).
    """
    ys, xs = np.where(mask > 0)
    if len(xs) < 10:
        return None, None
    px = np.stack([xs.astype(np.float64), ys.astype(np.float64)], axis=1)
    ref = np.asarray(ref_px, dtype=np.float64)
    dist = np.linalg.norm(px - ref, axis=1)
    # Robust tip: 97th-percentile distance band rather than the raw max, so a
    # few stray mask pixels past the cloth edge cannot drag the tip outside
    # the sleeve.
    d_hi = float(np.percentile(dist, 97.0))
    band = px[dist >= d_hi]
    tip = band.mean(axis=0)
    fx = intrinsic_matrix[0, 0]
    d_val = float(depth[int(tip[1]), int(tip[0])]) if depth is not None else 0.5
    if d_val < 0.01:
        valid = depth[mask > 0]
        valid = valid[(valid > 0.01) & (valid < 10)]
        d_val = float(np.median(valid)) if len(valid) > 0 else 0.5
    direction = ref - tip
    n = float(np.linalg.norm(direction))
    if n > 1e-6:
        tip = tip + direction / n * (inset_m * fx / d_val)
    pt = _backproject_pixel(tip[0], tip[1], depth, intrinsic_matrix, T_zed_to_base)
    return pt, (float(tip[0]), float(tip[1]))


def compute_shirt_top(mask, depth, intrinsic_matrix, T_zed_to_base):
    """Compute the top edge of a shirt mask (highest x in robot frame).

    The shirt top is the collar region. In image space this corresponds to
    the rows with the smallest v (top of image). Takes the centroid of
    pixels in the top 15px band of the mask.
    """
    ys, xs = np.where(mask > 0)
    if len(xs) < 10:
        return None
    top_v = int(ys.min())
    band = 15
    top_mask = (ys >= top_v) & (ys <= top_v + band)
    top_xs = xs[top_mask]
    top_ys = ys[top_mask]
    if len(top_xs) < 3:
        return None
    u = float(top_xs.mean())
    v = float(top_ys.mean())
    return _backproject_pixel(u, v, depth, intrinsic_matrix, T_zed_to_base)


def compute_collar_edges(mask, depth, intrinsic_matrix, T_zed_to_base,
                         band=20, inset_m=0.020):
    """Left and right grab points of the collar (top band of a shirt mask).

    The collar is the reachable end of the shirt for a tabletop two-arm
    fold: the hem sits past the arms' workspace, so the body fold grabs the
    collar and pulls toward the hem instead. Takes the top band of the
    mask and returns its left-most and right-most points (image x), inset in
    x, so each arm grabs its own side of the collar. Mirrors
    compute_hem_edges but on the top band rather than the full width.

    Returns {'left_edge', 'right_edge'} world points (or None each).
    """
    ys, xs = np.where(mask > 0)
    if len(xs) < 10:
        return {"left_edge": None, "right_edge": None}
    top_v = int(ys.min())
    sel = (ys >= top_v) & (ys <= top_v + band)
    bxs, bys = xs[sel], ys[sel]
    if len(bxs) < 6:
        return {"left_edge": None, "right_edge": None}

    fx = intrinsic_matrix[0, 0]
    d_val = (float(depth[int(bys.mean()), int(bxs.mean())])
             if depth is not None else 0.5)
    if d_val < 0.01:
        valid = depth[mask > 0]
        valid = valid[(valid > 0.01) & (valid < 10)]
        d_val = float(np.median(valid)) if len(valid) > 0 else 0.5
    inset_px = inset_m * fx / d_val

    x_min, x_max = int(bxs.min()), int(bxs.max())
    left_col = min(x_min + int(inset_px), x_max)
    right_col = max(x_max - int(inset_px), x_min)
    left_rows = (bys[bxs == min(bxs[bxs >= left_col])]
                 if np.any(bxs >= left_col) else bys[bxs == x_min])
    right_rows = (bys[bxs == max(bxs[bxs <= right_col])]
                  if np.any(bxs <= right_col) else bys[bxs == x_max])
    left_u, left_v = float(left_col), float(left_rows.mean())
    right_u, right_v = float(right_col), float(right_rows.mean())
    # Backproject both edges with the band-median depth so a single edge
    # pixel reading through the neckline gap cannot throw the grab point
    # far out of reach.
    band_d = depth[bys, bxs] if depth is not None else np.array([])
    band_d = band_d[(band_d > 0.01) & (band_d < 10)]
    if len(band_d) > 0:
        d_band = float(np.median(band_d))
        left_pt = _backproject_with_depth(left_u, left_v, d_band,
                                          intrinsic_matrix, T_zed_to_base)
        right_pt = _backproject_with_depth(right_u, right_v, d_band,
                                           intrinsic_matrix, T_zed_to_base)
    else:
        left_pt = _backproject_pixel(left_u, left_v, depth,
                                     intrinsic_matrix, T_zed_to_base)
        right_pt = _backproject_pixel(right_u, right_v, depth,
                                      intrinsic_matrix, T_zed_to_base)
    return {
        "left_edge": left_pt,
        "right_edge": right_pt,
        "left_px": (left_u, left_v),
        "right_px": (right_u, right_v),
    }


def compute_obb_angle(mask, depth, intrinsic_matrix, T_zed_to_base):
    """Compute the OBB long-axis angle in the robot XY plane (radians).

    Use directly as grasp yaw: jaws will close perpendicular to the
    long axis of the masked region. Returns 0.0 on failure.
    """
    ys, xs = np.where(mask > 0)
    if len(xs) < 20:
        return 0.0
    coords = np.column_stack([xs.astype(float), ys.astype(float)])
    mean = coords.mean(axis=0)
    centered = coords - mean
    cov = np.cov(centered.T)
    eigenvalues, eigenvectors = np.linalg.eigh(cov)
    long_axis_px = eigenvectors[:, -1]

    cx, cy = float(mean[0]), float(mean[1])
    offset = 40.0
    p1 = _backproject_pixel(cx + long_axis_px[0] * offset,
                            cy + long_axis_px[1] * offset,
                            depth, intrinsic_matrix, T_zed_to_base)
    p2 = _backproject_pixel(cx - long_axis_px[0] * offset,
                            cy - long_axis_px[1] * offset,
                            depth, intrinsic_matrix, T_zed_to_base)
    if p1 is None or p2 is None:
        return 0.0
    dx = p1[0] - p2[0]
    dy = p1[1] - p2[1]
    return float(np.arctan2(dy, dx))


def compute_jaw_yaw_minor(mask, depth, K, T_r, T_l, T_primary):
    """OBB MINOR-axis jaw yaw in BOTH robot frames (radians).

    This is the gripper-yaw to command so the parallel jaws close ACROSS the
    masked region (its short dimension), not along it: the correct grasp for
    long thin objects (sleeves, utensils). Fits an OBB with cv2.minAreaRect,
    takes the minor (short) axis in image space, backprojects two points along
    it through depth to get a world direction in the primary frame, then
    transforms that direction into the left frame via the cross-arm
    calibration so each arm gets a yaw expressed in its OWN base frame.
    Returns (yaw_r, yaw_l) or (None, None) on failure.
    """
    m_pts = cv2.findNonZero((mask > 0).astype(np.uint8))
    if m_pts is None:
        return None, None
    (mcx, mcy), (mw, mh), mang = cv2.minAreaRect(m_pts)
    ma = np.radians(mang)
    dir_w = np.array([np.cos(ma), np.sin(ma)])
    dir_h = np.array([-np.sin(ma), np.cos(ma)])
    # minAreaRect width spans the dir_w edge; the MINOR axis is perpendicular to
    # the longer edge, i.e. the direction of the SHORTER side.
    minor = dir_h if mw >= mh else dir_w
    p0 = _backproject_pixel(mcx - minor[0] * 15, mcy - minor[1] * 15,
                            depth, K, T_primary)
    p1 = _backproject_pixel(mcx + minor[0] * 15, mcy + minor[1] * 15,
                            depth, K, T_primary)
    if p0 is None or p1 is None:
        return None, None
    p0 = np.asarray(p0, dtype=float)
    p1 = np.asarray(p1, dtype=float)
    yaw_r = float(np.arctan2(p1[1] - p0[1], p1[0] - p0[0]))
    q0 = right_to_left(p0, T_r, T_l)
    q1 = right_to_left(p1, T_r, T_l)
    yaw_l = float(np.arctan2(q1[1] - q0[1], q1[0] - q0[0]))
    return yaw_r, yaw_l


def draw_annotations(rgb, detections, grab_points=None):
    img = rgb.copy()
    colors = [
        (255, 0, 0), (0, 255, 0), (0, 0, 255),
        (255, 255, 0), (255, 0, 255), (0, 255, 255),
    ]
    for i, det in enumerate(detections):
        color = colors[i % len(colors)]
        if det.mask is not None:
            overlay = img.copy()
            overlay[det.mask > 0] = color
            img = cv2.addWeighted(img, 0.55, overlay, 0.45, 0)
            contours, _ = cv2.findContours(
                det.mask.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            cv2.drawContours(img, contours, -1, color, 2)

        cx, cy = int(det.centroid_2d[0]), int(det.centroid_2d[1])
        cv2.circle(img, (cx, cy), 5, color, -1)
        label = f"{det.label} ({det.confidence:.2f})"
        cv2.putText(img, label, (cx + 8, cy - 8),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1)

        if det.position_3d is not None:
            pos = det.position_3d
            coord_str = f"x={pos[0]:.3f} y={pos[1]:.3f} z={pos[2]:.3f}"
            cv2.putText(img, coord_str, (cx + 8, cy + 12),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 255, 255), 1)

    if grab_points:
        for label, gp in grab_points.items():
            if gp is None:
                continue
            for det in detections:
                if det.label == label and det.mask is not None:
                    fx = 700
                    break

    return img


def main():
    p = argparse.ArgumentParser(description="Standalone SAM3 detection CLI")
    p.add_argument("prompts", nargs="+", help="Text prompts for detection")
    p.add_argument("--image", help="Path to RGB image (skip ZED capture)")
    p.add_argument("--depth", help="Path to depth .npy (skip ZED capture)")
    p.add_argument("--annotate", action="store_true", help="Save annotated image")
    p.add_argument("--grab", action="store_true", help="Compute grab points (inset from edge)")
    p.add_argument("--hem-edges", action="store_true", help="Compute left/right edge grab points for hem")
    p.add_argument("--inset", type=float, default=0.015, help="Grab inset from edge in meters (default 0.015)")
    p.add_argument("--masks", action="store_true", help="Save individual masks as PNGs")
    p.add_argument("--json", action="store_true", dest="json_out", help="JSON output to stdout")
    p.add_argument("--outdir", default=OUT_DIR, help="Output directory")
    p.add_argument("--threshold", type=float, default=0.5, help="SAM3 confidence threshold")
    p.add_argument("--frame", default="right", choices=["right", "left"],
                   help="Primary calibration frame (default: right)")
    p.add_argument("--points", nargs="+", metavar="LABEL:X,Y",
                   help="Point prompts as label:x,y (pixel coords). "
                        "E.g. --points 'knife1:800,400' 'knife2:900,420'")
    args = p.parse_args()

    os.makedirs(args.outdir, exist_ok=True)
    T_r, T_l = load_calibration()
    T_primary = T_r if args.frame == "right" else T_l

    if args.image:
        rgb = cv2.imread(args.image)
        if rgb is None:
            print(f"ERROR: cannot read {args.image}", file=sys.stderr)
            sys.exit(1)
        rgb = cv2.cvtColor(rgb, cv2.COLOR_BGR2RGB)
        depth = None
        if args.depth:
            depth = np.load(args.depth)
        h, w = rgb.shape[:2]
        K = np.array([[w * 0.7, 0, w / 2], [0, w * 0.7, h / 2], [0, 0, 1]])
    else:
        print("Capturing from ZED...", file=sys.stderr)
        rgb, depth, cam_cfg = capture_zed()
        K = np.array([
            [cam_cfg.fx, 0, cam_cfg.cx],
            [0, cam_cfg.fy, cam_cfg.cy],
            [0, 0, 1],
        ])
        np.save(os.path.join(args.outdir, "rgb.npy"), rgb)
        if depth is not None:
            np.save(os.path.join(args.outdir, "depth.npy"), depth)
        # Intrinsics alongside the frame so downstream tools (grasp_preview)
        # can project base-frame points back into this exact capture.
        np.save(os.path.join(args.outdir, "intrinsics.npy"), K)
        cv2.imwrite(os.path.join(args.outdir, "capture.png"),
                    cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR))
        print("Saved capture to", args.outdir, file=sys.stderr)

    point_specs = []
    if args.points:
        for ps in args.points:
            label, coords = ps.rsplit(":", 1)
            x, y = coords.split(",")
            point_specs.append((label.strip(), float(x), float(y)))

    detections = []
    if args.prompts:
        print(f"Running SAM3 on {len(args.prompts)} text prompts...", file=sys.stderr)
        detections.extend(run_sam3(rgb, depth, args.prompts, K, T_primary))
    if point_specs:
        if detections:
            import torch, gc
            torch.cuda.empty_cache(); gc.collect()
        print(f"Running SAM3 with {len(point_specs)} point prompts...", file=sys.stderr)
        detections.extend(run_sam3_points(rgb, depth, point_specs, K, T_primary))
    print(f"Found {len(detections)} detections", file=sys.stderr)

    # SAM3 labels left/right from the WEARER's perspective (mirrored from
    # camera). Camera-frame naming is needed: "left" = left of image = FR3
    # side and "right" = right of image = Panda side. Fix: for any pair of
    # detections that differ only by "left"/"right", swap labels based on
    # their pixel centroid x coordinate (smaller x = left of image).
    _swap_pairs = [
        ("left sleeve", "right sleeve"),
        ("left sleeve cuff", "right sleeve cuff"),
        ("left leg opening", "right leg opening"),
        ("left pant cuff", "right pant cuff"),
    ]
    for lname, rname in _swap_pairs:
        ld = [d for d in detections if d.label == lname]
        rd = [d for d in detections if d.label == rname]
        if ld and rd:
            l_cx = ld[0].centroid_2d[0]
            r_cx = rd[0].centroid_2d[0]
            if l_cx > r_cx:
                ld[0].label = rname
                rd[0].label = lname

    grab_points = {}
    hem_edges = {}
    shirt_top = None
    if args.grab:
        for det in detections:
            if det.mask is not None:
                gp = compute_grab_points(det.mask, depth, K, T_primary, inset_m=args.inset)
                grab_points[det.label] = gp
    # Labels whose masks should get left/right hem-edge computation.
    # "hem" covers t-shirt hem; leg openings, cuffs, and folded edges
    # have the same horizontal-strip geometry.
    _HEM_EDGE_KEYWORDS = ("hem", "leg opening", "cuff", "folded edge")
    if args.hem_edges:
        for det in detections:
            if det.mask is not None and any(k in det.label.lower() for k in _HEM_EDGE_KEYWORDS):
                hem_edges[det.label] = compute_hem_edges(
                    det.mask, depth, K, T_primary, inset_m=args.inset)

    # Labels whose masks define the garment body (top edge = fold target).
    _BODY_KEYWORDS = ("shirt", "sweater", "shorts", "pants", "folded pants",
                      "garment", "cloth", "fabric")
    collar_edges = {}
    for det in detections:
        if det.mask is not None and any(k in det.label.lower() for k in _BODY_KEYWORDS):
            shirt_top = compute_shirt_top(det.mask, depth, K, T_primary)
            # Collar left/right grab points: the reachable end of the shirt
            # for a two-arm fold (each arm grabs its side, then folds toward
            # the hem). The hem itself is past the arms' workspace.
            collar_edges[det.label] = compute_collar_edges(
                det.mask, depth, K, T_primary, inset_m=args.inset)

    # Sleeve outer-edge (tip) grab points: farthest mask band from the shirt
    # body centroid, inset back onto the cloth. The fold prefers these over
    # the centroid so the pinch lands at the sleeve edge.
    sleeve_edges = {}
    _body_ref = None
    for det in detections:
        if det.mask is not None and any(k in det.label.lower() for k in _BODY_KEYWORDS):
            bys, bxs = np.where(det.mask > 0)
            if len(bxs) > 0:
                _body_ref = (float(bxs.mean()), float(bys.mean()))
            break
    if _body_ref is None and depth is not None:
        _body_ref = (depth.shape[1] / 2.0, depth.shape[0] / 2.0)
    if _body_ref is not None:
        for det in detections:
            if det.mask is not None and "sleeve" in det.label.lower():
                pt, _tip_px = compute_sleeve_outer_edge(
                    det.mask, _body_ref, depth, K, T_primary)
                if pt is None:
                    continue
                # OBB minor-axis yaw (the jaw closing direction), in both
                # robot frames.
                yaw_r, yaw_l = compute_jaw_yaw_minor(det.mask, depth, K, T_r, T_l, T_primary)
                sleeve_edges[det.label] = {"pt": pt, "yaw_r": yaw_r, "yaw_l": yaw_l}

    results = []
    print("")
    print(f"{'Label':<16} {'Conf':>5}  {'RIGHT frame (flange z)':^36}  {'LEFT frame (flange z)':^36}")
    print(f"{'':16} {'':>5}  {'x':>10} {'y':>10} {'z':>10}  {'x':>10} {'y':>10} {'z':>10}")
    print("-" * 110)

    for det in detections:
        if det.position_3d is None:
            print(f"{det.label:<16} {det.confidence:5.2f}  {'no 3D position':^36}  {'':^36}")
            continue

        if args.frame == "right":
            pos_r = det.position_3d
            pos_l = right_to_left(pos_r, T_r, T_l)
        else:
            pos_l = det.position_3d
            pos_r = left_to_right(pos_l, T_r, T_l)

        print(f"{det.label:<16} {det.confidence:5.2f}  "
              f"{pos_r[0]:10.4f} {pos_r[1]:10.4f} {pos_r[2]:10.4f}  "
              f"{pos_l[0]:10.4f} {pos_l[1]:10.4f} {pos_l[2]:10.4f}")

        entry = {
            "label": det.label,
            "confidence": float(det.confidence),
            "right_frame": [float(x) for x in pos_r],
            "left_frame": [float(x) for x in pos_l],
        }

        if args.grab and det.label in grab_points and grab_points[det.label] is not None:
            gp_r = grab_points[det.label] if args.frame == "right" else left_to_right(grab_points[det.label], T_r, T_l)
            gp_l = right_to_left(gp_r, T_r, T_l) if args.frame == "right" else grab_points[det.label]
            entry["grab_right"] = [float(x) for x in gp_r]
            entry["grab_left"] = [float(x) for x in gp_l]

        if args.grab and det.mask is not None and depth is not None:
            entry["obb_angle_right"] = compute_obb_angle(det.mask, depth, K, T_r)
            entry["obb_angle_left"] = compute_obb_angle(det.mask, depth, K, T_l)

        # Generic OBB MINOR-axis jaw yaw for ANY masked label (when --masks is
        # on). obb_angle_* above is the LONG axis (jaws close ALONG the object);
        # jaw_yaw_* here is the MINOR axis (jaws close ACROSS it), the correct
        # grasp for long thin objects like utensils. Additive: the sleeve path
        # may already have set these via sleeve_edges, so don't overwrite.
        if (args.masks and det.mask is not None and depth is not None
                and "jaw_yaw_right" not in entry):
            jy_r, jy_l = compute_jaw_yaw_minor(det.mask, depth, K, T_r, T_l, T_primary)
            if jy_r is not None:
                entry["jaw_yaw_right"] = jy_r
                entry["jaw_yaw_left"] = jy_l

        if det.label in hem_edges:
            he = hem_edges[det.label]
            if he["left_edge"] is not None:
                le_r = he["left_edge"] if args.frame == "right" else left_to_right(he["left_edge"], T_r, T_l)
                le_l = right_to_left(le_r, T_r, T_l) if args.frame == "right" else he["left_edge"]
                entry["left_edge_right"] = [float(x) for x in le_r]
                entry["left_edge_left"] = [float(x) for x in le_l]
            if he["right_edge"] is not None:
                re_r = he["right_edge"] if args.frame == "right" else left_to_right(he["right_edge"], T_r, T_l)
                re_l = right_to_left(re_r, T_r, T_l) if args.frame == "right" else he["right_edge"]
                entry["right_edge_right"] = [float(x) for x in re_r]
                entry["right_edge_left"] = [float(x) for x in re_l]

        # Collar edges: RIGHT robot grabs the collar's left edge, LEFT robot
        # grabs the collar's right edge (mirrors the hem-edge convention).
        # Key names match fold_keypoints.load_fold_keypoints (collar_left_*,
        # collar_right_*) so the keypoint loader consumes them directly.
        if det.label in collar_edges:
            ce = collar_edges[det.label]
            if ce["left_edge"] is not None:
                cle_r = ce["left_edge"] if args.frame == "right" else left_to_right(ce["left_edge"], T_r, T_l)
                cle_l = right_to_left(cle_r, T_r, T_l) if args.frame == "right" else ce["left_edge"]
                entry["collar_left_right"] = [float(x) for x in cle_r]
                entry["collar_left_left"] = [float(x) for x in cle_l]
            if ce["right_edge"] is not None:
                cre_r = ce["right_edge"] if args.frame == "right" else left_to_right(ce["right_edge"], T_r, T_l)
                cre_l = right_to_left(cre_r, T_r, T_l) if args.frame == "right" else ce["right_edge"]
                entry["collar_right_right"] = [float(x) for x in cre_r]
                entry["collar_right_left"] = [float(x) for x in cre_l]

        # Sleeve outer-edge (tip) grab, preferred by the fold over centroid,
        # plus the OBB minor-axis jaw yaw in both frames.
        if det.label in sleeve_edges:
            rec = sleeve_edges[det.label]
            se = rec["pt"]
            se_r = se if args.frame == "right" else left_to_right(se, T_r, T_l)
            se_l = right_to_left(se_r, T_r, T_l) if args.frame == "right" else se
            entry["outer_edge_right"] = [float(x) for x in se_r]
            entry["outer_edge_left"] = [float(x) for x in se_l]
            if rec["yaw_r"] is not None:
                entry["jaw_yaw_right"] = rec["yaw_r"]
                entry["jaw_yaw_left"] = rec["yaw_l"]

        results.append(entry)

    if args.grab and grab_points:
        print("")
        print(f"{'Label':<16} {'Grab RIGHT frame':^36}  {'Grab LEFT frame':^36}")
        print(f"{'':16} {'x':>10} {'y':>10} {'z':>10}  {'x':>10} {'y':>10} {'z':>10}")
        print("-" * 100)
        for r in results:
            if "grab_right" in r:
                gr = r["grab_right"]
                gl = r["grab_left"]
                print(f"{r['label']:<16} "
                      f"{gr[0]:10.4f} {gr[1]:10.4f} {gr[2]:10.4f}  "
                      f"{gl[0]:10.4f} {gl[1]:10.4f} {gl[2]:10.4f}")

    if hem_edges:
        print("")
        print("Hem edges (15mm inset, flange z):")
        print(f"  {'Edge':<16} {'RIGHT frame':^36}  {'LEFT frame':^36}")
        print(f"  {'':16} {'x':>10} {'y':>10} {'z':>10}  {'x':>10} {'y':>10} {'z':>10}")
        print("  " + "-" * 100)
        for r in results:
            if "left_edge_right" in r:
                lr = r["left_edge_right"]
                ll = r["left_edge_left"]
                print(f"  {'left edge':<16} "
                      f"{lr[0]:10.4f} {lr[1]:10.4f} {lr[2]:10.4f}  "
                      f"{ll[0]:10.4f} {ll[1]:10.4f} {ll[2]:10.4f}")
                print(f"    RIGHT robot grabs left edge:  RIGHT frame (x={lr[0]:.4f}, y={lr[1]:.4f}, z={lr[2]:.4f})")
            if "right_edge_right" in r:
                rr = r["right_edge_right"]
                rl = r["right_edge_left"]
                print(f"  {'right edge':<16} "
                      f"{rr[0]:10.4f} {rr[1]:10.4f} {rr[2]:10.4f}  "
                      f"{rl[0]:10.4f} {rl[1]:10.4f} {rl[2]:10.4f}")
                print(f"    LEFT robot grabs right edge:  LEFT frame  (x={rl[0]:.4f}, y={rl[1]:.4f}, z={rl[2]:.4f})")

    if shirt_top is not None:
        st_r = shirt_top if args.frame == "right" else left_to_right(shirt_top, T_r, T_l)
        st_l = right_to_left(st_r, T_r, T_l) if args.frame == "right" else shirt_top
        print(f"\nShirt top (hem fold target):")
        print(f"  RIGHT frame: x={st_r[0]:.4f} y={st_r[1]:.4f} z={st_r[2]:.4f}")
        print(f"  LEFT frame:  x={st_l[0]:.4f} y={st_l[1]:.4f} z={st_l[2]:.4f}")
        for r in results:
            lbl = r.get("label", "").lower()
            if any(k in lbl for k in _BODY_KEYWORDS):
                r["shirt_top_right"] = [float(x) for x in st_r]
                r["shirt_top_left"] = [float(x) for x in st_l]

    if args.annotate:
        ann = draw_annotations(rgb, detections, grab_points)
        ann_path = os.path.join(args.outdir, "annotated.png")
        cv2.imwrite(ann_path, cv2.cvtColor(ann, cv2.COLOR_RGB2BGR))
        print(f"\nAnnotated image: {ann_path}", file=sys.stderr)

    if args.masks:
        for det in detections:
            if det.mask is not None:
                mask_path = os.path.join(
                    args.outdir, f"mask_{det.label.replace(' ', '_')}.png")
                cv2.imwrite(mask_path, (det.mask * 255).astype(np.uint8))
        print(f"Masks saved to {args.outdir}", file=sys.stderr)

    if args.json_out:
        print(json.dumps(results, indent=2))

    json_path = os.path.join(args.outdir, "detections.json")
    with open(json_path, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nResults saved to {json_path}", file=sys.stderr)


if __name__ == "__main__":
    main()

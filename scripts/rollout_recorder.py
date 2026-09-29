#!/usr/bin/env python3
"""Reusable rollout recorder for CoRL paper figures.

Workflow:
  1. capture_scene()   : snap photos + depth from kinects, save, close kinects
  2. detect_scene()    : run SAM3 on saved images (no kinects needed), save annotated
  3. save_bt(yaml_str) : persist the BT plan YAML
  4. start_recording() : reopen kinects for video only (no SAM3)
  5. mark_keyframe()   : tag current frame with label (e.g. "grasp", "pour")
  6. capture_placement_shot(idx) : save overhead frame for position distribution
  7. stop_recording()  : save per-camera videos + keyframe images + meta.json
  8. set_result(success, notes) : label rollout as success/fail in meta.json

Usage:
    from rollout_recorder import RolloutRecorder
    rec = RolloutRecorder("mug_pour")
    caps, cals = rec.capture_scene()
    dets = rec.detect_scene(["mug", "mug handle", "plate"])
    rec.save_bt(yaml_plan_str)
    rec.start_recording()
    # ... execute task, call rec.mark_keyframe("grasp") at key moments ...
    rec.capture_placement_shot(0)
    rec.stop_recording()
    rec.set_result(success=True, notes="clean grasp")
"""
from __future__ import annotations

import datetime
import glob
import json
import os
import shutil
import subprocess
import sys
import tempfile
import threading
import time

import cv2
import numpy as np
import pyk4a
import yaml
from PIL import Image as PILImage

_src = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "src")
if _src not in sys.path:
    sys.path.insert(0, os.path.abspath(_src))

from spark_real.pipeline_types import PipelineConfig  # noqa: E402

ROLLOUT_ROOT = os.path.join(_src, "spark_real", "rollouts")
CAL_DIR = os.path.join(_src, "spark_real", "output", "calibrations")


def _serial_roles(family: str | None = None) -> dict:
    """Kinect serial -> role map from configs/<family>_default.yaml.

    Config-based on purpose: the same physical Kinect serial has DIFFERENT
    roles on different rigs, so a static dict silently mislabels recordings.
    """
    family = (
        family
        or os.environ.get("SPARK_ROBOT")
        or PipelineConfig.__dataclass_fields__["robot_family"].default
    )
    cfg_path = os.path.join(
        _src, "spark_real", "configs", f"{family}_default.yaml"
    )
    with open(cfg_path) as fh:
        raw = yaml.safe_load(fh) or {}
    return {
        str(c["serial"]): str(c["role"])
        for c in raw.get("cameras", [])
        if c.get("type") == "kinect" and c.get("serial")
    }


SERIALS = _serial_roles()


class RolloutRecorder:
    def __init__(self, task_name: str, fps: int = 10):
        self.task_name = task_name
        self.fps = fps
        ts = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
        self.out_dir = os.path.join(ROLLOUT_ROOT, task_name, ts)
        os.makedirs(self.out_dir, exist_ok=True)

        self._recording = False
        self._rec_thread = None
        self._frames = {"birdview": [], "sideview": []}
        self._keyframes = []
        self._kinects = None
        self._caps = None
        self._cals = None
        self._detections = {}
        self._result = None  # {"success": bool, "notes": str}

    # Step 1: Capture scene photos

    def capture_scene(self):
        kinects = {}
        for i in range(pyk4a.connected_device_count()):
            k = pyk4a.PyK4A(device_id=i)
            k.open(); k.start()
            cam = SERIALS.get(k.serial, f"unk_{k.serial}")
            kinects[cam] = k

        caps = {}
        for cam, k in kinects.items():
            cap = k.get_capture()
            rgb = cap.color[:, :, :3][:, :, ::-1].copy()
            depth = cap.transformed_depth.copy()
            intr = k.calibration.get_camera_matrix(pyk4a.CalibrationType.COLOR)
            caps[cam] = {
                "rgb": rgb, "depth": depth,
                "fx": intr[0, 0], "fy": intr[1, 1],
                "cx": intr[0, 2], "cy": intr[1, 2],
            }
            cv2.imwrite(
                os.path.join(self.out_dir, f"scene_{cam}.png"),
                cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR),
            )

            np.save(os.path.join(self.out_dir, f"depth_{cam}.npy"), depth)

            # Save colorized depth overlay
            valid_mask = depth > 0
            if valid_mask.any():
                d_norm = np.zeros_like(depth, dtype=np.float32)
                d_min = float(depth[valid_mask].min())
                d_max = float(depth[valid_mask].max())
                if d_max > d_min:
                    d_norm[valid_mask] = (
                        (depth[valid_mask].astype(np.float32) - d_min)
                        / (d_max - d_min) * 255.0
                    )
                d_u8 = d_norm.astype(np.uint8)
                depth_color = cv2.applyColorMap(d_u8, cv2.COLORMAP_JET)
                # Zero-depth pixels -> black
                depth_color[~valid_mask] = 0
                cv2.imwrite(
                    os.path.join(self.out_dir, f"depth_viz_{cam}.png"),
                    depth_color,
                )

        for k in kinects.values():
            try: k.stop()
            except: pass
            try: k.close()
            except: pass

        self._cals = _load_cals()
        self._caps = caps
        return caps, self._cals

    # Step 2: Detect with SAM3 on saved images

    def detect_scene(self, prompts: list[str], cameras: str = "both"):
        # DELIBERATELY DEFERRED IMPORT: SPARKPerception pulls in torch, and
        # capture_scene() must be able to run pyk4a captures BEFORE torch is
        # loaded in this process (torch loaded first segfaults pyk4a capture
        # -- same reason start_recording() runs the Kinects in a subprocess).
        # Do NOT move this to module top.
        from spark_real.perception.spark_perception import SPARKPerception

        perc = SPARKPerception()
        perc.load_models(load_da3=False)

        cam_list = list(self._caps.keys()) if cameras == "both" else [cameras]
        colors_cycle = [
            (0, 255, 0), (255, 0, 0), (0, 0, 255),
            (255, 255, 0), (255, 0, 255), (0, 255, 255),
        ]

        results = {}
        for cam in cam_list:
            c = self._caps[cam]
            pil = PILImage.fromarray(c["rgb"])
            intr = {k: c[k] for k in ("fx", "fy", "cx", "cy")}
            img_ann = c["rgb"].copy()

            for pi, prompt in enumerate(prompts):
                state = perc._sam3.set_image(pil)
                state = perc._sam3.set_text_prompt(prompt=prompt, state=state)
                masks = state.get("masks"); scores = state.get("scores")
                if masks is None or masks.numel() == 0:
                    continue
                best = int(scores.argmax())
                score = float(scores[best])
                mask = masks[best].cpu().numpy().squeeze()
                ys, xs = np.where(mask > 0)
                if len(xs) == 0:
                    continue

                mcx, mcy = float(xs.mean()), float(ys.mean())
                pos = _backproject(mcx, mcy, c["depth"], intr, self._cals[cam])

                n = min(200, len(ys))
                idx = np.linspace(0, len(ys) - 1, n, dtype=int)
                zs = []
                for i in idx:
                    p = _backproject(float(xs[i]), float(ys[i]),
                                     c["depth"], intr, self._cals[cam])
                    if p is not None:
                        zs.append(p[2])
                max_z = max(zs) if zs else None

                key = f"{cam}_{prompt}"
                results[key] = {
                    "pos": pos, "conf": score, "max_z": max_z,
                    "centroid_px": (mcx, mcy), "camera": cam, "label": prompt,
                }

                color = colors_cycle[pi % len(colors_cycle)]
                mask_u8 = (mask * 255).astype(np.uint8)
                contours, _ = cv2.findContours(
                    mask_u8, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
                cv2.drawContours(img_ann, contours, -1, color, 2)
                cv2.circle(img_ann, (int(mcx), int(mcy)), 6, color, -1)

                # OBB major-axis arrow via PCA on mask pixels
                if len(xs) >= 5:
                    pts = np.column_stack([xs.astype(np.float64),
                                          ys.astype(np.float64)])
                    mean = pts.mean(axis=0)
                    cov = np.cov(pts, rowvar=False)
                    eigvals, eigvecs = np.linalg.eigh(cov)
                    major = eigvecs[:, np.argmax(eigvals)]
                    # Arrow length proportional to spread, capped
                    arrow_len = min(80, int(np.sqrt(max(eigvals)) * 1.5))
                    pt1 = (int(mean[0]), int(mean[1]))
                    pt2 = (int(mean[0] + major[0] * arrow_len),
                           int(mean[1] + major[1] * arrow_len))
                    cv2.arrowedLine(img_ann, pt1, pt2, color, 2,
                                   tipLength=0.25)

                lbl = prompt
                if pos is not None:
                    lbl += f" ({pos[0]:.3f},{pos[1]:.3f},{pos[2]:.3f})"
                cv2.putText(img_ann, lbl, (int(mcx) + 10, int(mcy) - 10),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.4, color, 2)

            cv2.imwrite(
                os.path.join(self.out_dir, f"annotated_{cam}.png"),
                cv2.cvtColor(img_ann, cv2.COLOR_RGB2BGR),
            )

        self._detections = results
        return results

    # Save BT plan YAML

    def save_bt(self, yaml_str: str):
        path = os.path.join(self.out_dir, "bt_plan.yaml")
        with open(path, "w") as f:
            f.write(yaml_str)
        print(f"  BT plan saved: {path}")

    # Success / fail labeling

    def set_result(self, success: bool, notes: str = ""):
        """
        Label this rollout as success or failure.

        Can be called before or after stop_recording(). If called after,
        meta.json is re-written with the result field.
        """
        self._result = {"success": success, "notes": notes}
        # If meta.json already exists (stop_recording was called), update it
        meta_path = os.path.join(self.out_dir, "meta.json")
        if os.path.exists(meta_path):
            with open(meta_path) as f:
                meta = json.load(f)
            meta["result"] = self._result
            with open(meta_path, "w") as f:
                json.dump(meta, f, indent=2, default=_json_default)
            print(f"  Result updated in {meta_path}")

    # Placement shot for position-distribution figures

    def capture_placement_shot(self, shot_idx: int):
        """
        Save an overhead (birdview) frame as placement_{shot_idx:02d}.png.

        Must be called while recording is active (kinects running in the
        subprocess).  Reads the most recent birdview frame from the tmpdir.
        """
        if not self._recording:
            print("  Warning: capture_placement_shot called while not recording")
            return
        files = sorted(glob.glob(
            os.path.join(self._rec_tmpdir, "birdview_*.jpg")))
        if not files:
            print("  Warning: no birdview frames available yet")
            return
        src = files[-1]
        dst = os.path.join(self.out_dir, f"placement_{shot_idx:02d}.png")
        frame = cv2.imread(src)
        if frame is not None:
            cv2.imwrite(dst, frame)
            print(f"  Placement shot saved: {dst}")

    # Start video recording
    # Kinects are opened in a SUBPROCESS to avoid torch/pyk4a segfault.
    # Frames are written to a shared tmpdir and stitched into video at stop.

    def start_recording(self):
        self._rec_tmpdir = tempfile.mkdtemp(prefix="rollout_rec_")
        self._rec_flag = os.path.join(self._rec_tmpdir, "RECORDING")
        open(self._rec_flag, "w").close()
        self._frames = {"birdview": [], "sideview": []}
        self._keyframes = []

        script = f"""
import pyk4a, cv2, os, time, json
SERIALS = {json.dumps(SERIALS)}
tmpdir = "{self._rec_tmpdir}"
flag = "{self._rec_flag}"
fps = {self.fps}

kinects = {{}}
for i in range(pyk4a.connected_device_count()):
    k = pyk4a.PyK4A(device_id=i)
    k.open(); k.start()
    cam = SERIALS.get(k.serial, f"unk_{{k.serial}}")
    kinects[cam] = k

idx = 0
while os.path.exists(flag):
    for cam, k in kinects.items():
        try:
            cap = k.get_capture()
            rgb = cap.color[:,:,:3][:,:,::-1].copy()
            path = os.path.join(tmpdir, f"{{cam}}_{{idx:06d}}.jpg")
            cv2.imwrite(path, cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR))
        except: pass
    idx += 1
    time.sleep(1.0 / fps)

for k in kinects.values():
    try: k.stop()
    except: pass
    try: k.close()
    except: pass

open(os.path.join(tmpdir, "DONE"), "w").write(str(idx))
"""
        self._rec_proc = subprocess.Popen(
            [sys.executable, "-c", script],
            stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL,
        )
        time.sleep(2)  # let kinects start
        self._recording = True
        print(f"  Recording subprocess pid={self._rec_proc.pid}")

    # Step 4: Mark keyframe

    def mark_keyframe(self, label: str):
        if not self._recording:
            return
        # Find most recent frame in tmpdir for each cam
        for cam in ["birdview", "sideview"]:
            files = sorted(glob.glob(os.path.join(self._rec_tmpdir, f"{cam}_*.jpg")))
            if files:
                src = files[-1]
                dst = os.path.join(self.out_dir, f"keyframe_{label}_{cam}.png")
                frame = cv2.imread(src)
                if frame is not None:
                    cv2.imwrite(dst, frame)
                    self._keyframes.append({
                        "label": label, "camera": cam, "path": dst,
                    })

    # Step 5: Stop recording & save

    def stop_recording(self):
        self._recording = False

        # Signal subprocess to stop
        if hasattr(self, '_rec_flag') and os.path.exists(self._rec_flag):
            os.remove(self._rec_flag)

        if hasattr(self, '_rec_proc') and self._rec_proc:
            self._rec_proc.wait(timeout=15)

        # Wait for DONE file
        done = os.path.join(self._rec_tmpdir, "DONE")
        t0 = time.time()
        while not os.path.exists(done) and time.time() - t0 < 10:
            time.sleep(0.5)

        # Stitch frames into per-camera videos
        for cam in ["birdview", "sideview"]:
            files = sorted(glob.glob(
                os.path.join(self._rec_tmpdir, f"{cam}_*.jpg")))
            if not files:
                print(f"  No frames for {cam}")
                continue
            sample = cv2.imread(files[0])
            h, w = sample.shape[:2]
            path = os.path.join(self.out_dir, f"{cam}.mp4")
            out = cv2.VideoWriter(path, cv2.VideoWriter_fourcc(*"mp4v"),
                                  self.fps, (w, h))
            for f in files:
                frame = cv2.imread(f)
                if frame is not None:
                    out.write(frame)
            out.release()
            print(f"  Video: {path} ({len(files)} frames)")

        # Cleanup tmpdir
        shutil.rmtree(self._rec_tmpdir, ignore_errors=True)

        # Save metadata
        meta = {
            "task": self.task_name,
            "keyframes": self._keyframes,
            "detections": {
                k: {kk: vv for kk, vv in v.items() if kk != "mask"}
                for k, v in self._detections.items()
            },
        }
        if self._result is not None:
            meta["result"] = self._result
        with open(os.path.join(self.out_dir, "meta.json"), "w") as f:
            json.dump(meta, f, indent=2, default=_json_default)

        print(f"  Rollout saved: {self.out_dir}")

    def get(self, label, cam_pref="sideview"):
        for cp in [cam_pref, "birdview" if cam_pref == "sideview" else "sideview"]:
            d = self._detections.get(f"{cp}_{label}")
            if d and d.get("pos") is not None:
                return d
        return None


def _load_cals():
    cals = {}
    for cam in ("birdview", "sideview"):
        path = os.path.join(CAL_DIR, f"handeye_{cam}.json")
        with open(path) as f:
            d = json.load(f)
        cals[cam] = {
            "T": np.array(d["T_cam_to_base_4x4"]).reshape(4, 4),
            "depth_scale": d.get("depth_scale_correction", 1.0),
        }
    return cals


def _backproject(cx, cy, depth, intr, cal):
    iy = max(0, min(depth.shape[0] - 1, int(round(cy))))
    ix = max(0, min(depth.shape[1] - 1, int(round(cx))))
    r = 5; h, w = depth.shape
    patch = depth[max(0, iy-r):min(h, iy+r+1), max(0, ix-r):min(w, ix+r+1)]
    valid = patch[patch > 0]
    if len(valid) == 0:
        return None
    z_m = float(np.median(valid)) / 1000.0 * cal["depth_scale"]
    x_cam = (cx - intr["cx"]) * z_m / intr["fx"]
    y_cam = (cy - intr["cy"]) * z_m / intr["fy"]
    p = cal["T"] @ np.array([x_cam, y_cam, z_m, 1.0])
    return p[:3]


def _json_default(obj):
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    if isinstance(obj, (np.float32, np.float64)):
        return float(obj)
    return str(obj)

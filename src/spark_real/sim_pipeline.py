#!/usr/bin/env python3
"""
SPARK Standalone Simulation Pipeline - No ROS2.

Pure Python pipeline: MuJoCo sim + SAM3 perception + Gemini planning + IK execution.
This is the Python-first reimplementation of the ROS2 pipeline.

Usage:
    # Interactive mode with viewer
    python -m spark_real.sim_pipeline --scene perception_test_scene.xml

    # Headless with instruction
    python -m spark_real.sim_pipeline --headless --instruction "pick up the red block"

    # With video tracking
    python -m spark_real.sim_pipeline --tracking --instruction "sort the blocks by color"
"""

from __future__ import annotations
from dataclasses import dataclass
import mujoco
import mujoco.viewer
import numpy as np
import time
import os
from pathlib import Path
from typing import Optional

import tyro

# SPARK imports
from spark_real.perception.camera import CameraConfig
from spark_real.planning.spark_planner import SPARKPlanner


SCENES_DIR = Path(__file__).parent.parent / "spark_sim" / "scenes"


def _detect_robot_family(model: "mujoco.MjModel") -> str:
    """Sniff a MuJoCo model to decide which robot family it carries.

    Returns one of {"panda", "ur10e", "unknown"}. Used to switch between
    the legacy 6-DOF UR10e sim path and the production-skills 7-DOF
    Panda path. We look for known joint names rather than counting DoFs
    so non-arm joints in the scene (free joints on objects, hinges on
    napkins) don't confuse us.
    """
    for jn in (f"joint{i}" for i in range(1, 8)):
        if mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, jn) >= 0:
            return "panda"
    for jn in ("shoulder_pan_joint", "elbow_joint"):
        if mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, jn) >= 0:
            return "ur10e"
    return "unknown"


class StandaloneMuJoCoSim:
    """MuJoCo simulation without ROS2."""

    # UR10e joint names (for reference)
    JOINT_NAMES = [
        'shoulder_pan_joint', 'shoulder_lift_joint', 'elbow_joint',
        'wrist_1_joint', 'wrist_2_joint', 'wrist_3_joint'
    ]

    def __init__(self, scene_path: str, headless: bool = False):
        self.scene_path = scene_path
        self.headless = headless
        self.model = mujoco.MjModel.from_xml_path(scene_path)
        self.data = mujoco.MjData(self.model)
        self._viewer = None

        # Find TCP site
        self.tcp_site_id = mujoco.mj_name2id(
            self.model, mujoco.mjtObj.mjOBJ_SITE, 'pinch'
        )

        mujoco.mj_forward(self.model, self.data)
        print(f"[Sim] Loaded: {scene_path}")
        print(f"[Sim] Bodies: {self.model.nbody}, Joints: {self.model.njnt}, "
              f"Actuators: {self.model.nu}, Cameras: {self.model.ncam}")

    @property
    def ee_position(self) -> np.ndarray:
        if self.tcp_site_id >= 0:
            return self.data.site_xpos[self.tcp_site_id].copy()
        return np.zeros(3)

    @property
    def joint_positions(self) -> np.ndarray:
        return self.data.qpos[:6].copy()

    def step(self, n: int = 1):
        """Step simulation."""
        for _ in range(n):
            mujoco.mj_step(self.model, self.data)

    def set_joint_targets(self, q: np.ndarray):
        """Set joint position targets via actuators."""
        for i in range(min(6, len(q))):
            self.data.ctrl[i] = q[i]

    def set_gripper(self, value: float):
        """Set gripper (0=open, 255=closed). Uses actuator's native range."""
        if self.model.nu > 6:
            self.data.ctrl[6] = float(np.clip(value, 0, 255))

    def render_camera(self, cam_idx: int = 0, width: int = 640,
                      height: int = 480) -> tuple:
        """Render RGB and depth from a camera."""
        renderer = mujoco.Renderer(self.model, height, width)
        renderer.update_scene(self.data, camera=cam_idx)
        rgb = renderer.render().copy()

        renderer.enable_depth_rendering()
        depth = renderer.render().copy()
        renderer.disable_depth_rendering()
        renderer.close()

        if len(depth.shape) == 3:
            depth = depth[:, :, 0]
        return rgb, depth

    def get_camera_config(self, cam_idx: int, width: int = 640,
                          height: int = 480) -> CameraConfig:
        """Get camera intrinsics/extrinsics."""
        fovy = self.model.cam_fovy[cam_idx]
        f = height / (2.0 * np.tan(np.deg2rad(fovy) / 2.0))

        cam_pos = self.data.cam_xpos[cam_idx].copy()
        cam_mat = self.data.cam_xmat[cam_idx].reshape(3, 3).copy()
        extrinsic = np.eye(4)
        extrinsic[:3, :3] = cam_mat
        extrinsic[:3, 3] = cam_pos

        return CameraConfig(
            width=width, height=height,
            fx=f, fy=f, cx=width / 2.0, cy=height / 2.0,
            extrinsic=extrinsic,
        )

    def get_object_position(self, name: str) -> Optional[np.ndarray]:
        """Get position of a named body."""
        body_id = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_BODY, name)
        if body_id >= 0:
            return self.data.xpos[body_id].copy()
        return None

    def compute_ik(self, target_pos: np.ndarray, max_iter: int = 300,
                   tol: float = 0.001) -> Optional[np.ndarray]:
        """Damped least-squares IK to target position with orientation constraint."""
        q = self.joint_positions.copy()

        # Work on a copy of data
        data_copy = mujoco.MjData(self.model)
        for i in range(min(6, self.model.nq)):
            data_copy.qpos[i] = q[i]

        # Target orientation: gripper pointing down (z-axis = -Z world)
        target_xmat = np.array([[1, 0, 0], [0, -1, 0], [0, 0, -1]])

        for iteration in range(max_iter):
            mujoco.mj_forward(self.model, data_copy)
            current_pos = data_copy.site_xpos[self.tcp_site_id].copy()
            pos_error = target_pos - current_pos

            # Orientation error
            site_xmat = data_copy.site_xmat[self.tcp_site_id].reshape(3, 3)
            R_err = target_xmat @ site_xmat.T
            angle = np.arccos(np.clip((np.trace(R_err) - 1) / 2, -1, 1))
            if angle > 1e-6:
                axis = np.array([R_err[2,1]-R_err[1,2], R_err[0,2]-R_err[2,0], R_err[1,0]-R_err[0,1]])
                axis = axis / (2 * np.sin(angle))
                ori_error = axis * angle * 0.3
            else:
                ori_error = np.zeros(3)

            if np.linalg.norm(pos_error) < tol:
                return q

            # Jacobian
            jacp = np.zeros((3, self.model.nv))
            jacr = np.zeros((3, self.model.nv))
            mujoco.mj_jacSite(self.model, data_copy, jacp, jacr, self.tcp_site_id)
            Jp = jacp[:, :6]
            Jr = jacr[:, :6]

            # Combined position + orientation
            J = np.vstack([Jp, Jr])
            error = np.hstack([pos_error, ori_error])

            # Adaptive damping - less damping when close
            damping = 0.05 if np.linalg.norm(pos_error) < 0.05 else 0.1
            JJT = J @ J.T + damping**2 * np.eye(6)
            dq = J.T @ np.linalg.solve(JJT, error)

            # Larger step when far, smaller when close
            step_size = 0.5 if np.linalg.norm(pos_error) > 0.1 else 0.3
            q += step_size * dq
            q = np.clip(q, -2 * np.pi, 2 * np.pi)

            for i in range(6):
                data_copy.qpos[i] = q[i]

        final_err = np.linalg.norm(pos_error)
        if final_err < 0.02:
            return q
        return None

    def move_to(self, target_pos: np.ndarray, duration: float = 2.0,
                steps_per_sec: int = 500):
        """Move end-effector to target position with smooth interpolation."""
        target_q = self.compute_ik(target_pos)
        if target_q is None:
            print(f"[Sim] IK failed for target {target_pos}")
            return False

        start_q = self.joint_positions
        total_steps = int(duration * steps_per_sec)

        for step in range(total_steps):
            alpha = step / total_steps
            interp_q = start_q + alpha * (target_q - start_q)
            self.set_joint_targets(interp_q)
            self.step()

        return True

    def open_viewer(self):
        """Open interactive MuJoCo viewer."""
        if not self.headless:
            self._viewer = mujoco.viewer.launch_passive(self.model, self.data)

    def viewer_sync(self):
        """Sync viewer if open."""
        if self._viewer is not None and self._viewer.is_running():
            self._viewer.sync()


def run_pipeline(scene_path: str, instruction: str = "",
                 headless: bool = False, use_sam3: bool = True,
                 score_yaml: Optional[str] = None,
                 video_out: Optional[str] = None):
    """
    Run the standalone SPARK pipeline.

    Steps:
    1. Load MuJoCo scene; auto-detect robot family (UR10e vs Panda).
       For Panda scenes (corl_f*), construct a PandaSimWorld +
       MuJoCoSimExecutor so the score dispatches through the
       PRODUCTION skill registry (same path as real-robot).
    2. Render cameras (for SAM3 input, or fall back to MuJoCo ground truth).
    3. Run SAM3 detection if requested; else use ground-truth body positions.
    4. Either (a) load a pre-baked YAML score from disk (--score-yaml) or
       (b) call Gemini for a fresh score (--instruction).
    5. Dispatch score via skill registry (Panda path) or builtin handlers
       (UR10e path).
    """
    # Inspect the scene before constructing anything heavy.
    probe_model = mujoco.MjModel.from_xml_path(scene_path)
    family = _detect_robot_family(probe_model)
    print(f"[Pipeline] Scene: {scene_path}")
    print(f"[Pipeline] Detected robot family: {family} "
          f"(nq={probe_model.nq}, ncam={probe_model.ncam})")
    del probe_model

    sim_executor = None
    panda_sim = None
    record_camera = "agentview"

    if family == "panda":
        # Production path: PandaSimWorld + MuJoCoSimExecutor -> skill_registry.
        from spark_real.sim_pipeline_extras import PandaSimWorld
        from spark_real.sim_executor import MuJoCoSimExecutor
        panda_sim = PandaSimWorld(
            scene_path, width=800, height=600, record_camera=record_camera)
        sim_executor = MuJoCoSimExecutor(panda_sim)
        # Re-use the same sim for snapshotting + later detections.
        sim = _PandaSimShim(panda_sim)
    else:
        # Legacy UR10e path.
        sim = StandaloneMuJoCoSim(scene_path, headless=headless)
        if not headless:
            sim.open_viewer()

    # Step 1: Let sim settle.
    if panda_sim is not None:
        panda_sim.settle(80, snap_every=4)
    else:
        for _ in range(100):
            sim.step()
        if not headless:
            sim.viewer_sync()

    # Step 2: Render from all cameras (for SAM3 / annotation).
    print("[Pipeline] Rendering cameras...")
    images = {}
    ncam = sim.model.ncam
    for cam_idx in range(min(3, ncam)):
        try:
            rgb, depth = sim.render_camera(cam_idx)
            cam_config = sim.get_camera_config(cam_idx)
            images[cam_idx] = {"rgb": rgb, "depth": depth, "config": cam_config}
            print(f"  Camera {cam_idx}: {rgb.shape} RGB, {depth.shape} depth")
        except Exception as e:
            print(f"  Camera {cam_idx}: render failed ({e})")

    # Step 3: detections. SAM3 is optional; sim ground-truth is the
    # safer default for the regression-test bed (avoids GPU contention
    # with the live server).
    detections = []
    if use_sam3 and images:
        try:
            from spark_real.perception.sam3_detector import SAM3Detector
            detector = SAM3Detector()
            detector.load_model()
            prompts = _get_prompts_for_scene(sim)
            print(f"[Pipeline] Detecting with prompts: {prompts}")
            detections = detector.detect(
                images[0]["rgb"], prompts,
                images[0]["depth"], images[0]["config"],
            )
            print(f"[Pipeline] Found {len(detections)} objects:")
            for d in detections:
                pos_str = (f"3D=({d.position_3d[0]:.3f},"
                           f"{d.position_3d[1]:.3f},{d.position_3d[2]:.3f})"
                           if d.position_3d is not None else "no depth")
                print(f"  - {d.label}: conf={d.confidence:.2f}, {pos_str}")
        except Exception as e:
            print(f"[Pipeline] SAM3 detection failed: {e}")
            print("[Pipeline] Falling back to MuJoCo ground-truth positions")
            detections = _get_mujoco_detections(sim)
    else:
        detections = _get_mujoco_detections(sim)

    # Step 4: get a score (either pre-baked YAML or live planner).
    score = None
    if score_yaml is not None:
        import yaml as _yaml
        p = Path(score_yaml)
        if not p.is_absolute():
            p = Path.cwd() / p
        print(f"[Pipeline] Loading pre-baked score: {p}")
        with open(p) as f:
            score = _yaml.safe_load(f)
    elif instruction:
        print(f"[Pipeline] Planning for: '{instruction}'")
        # Map the sniffed sim family onto the planner's prompt families: a
        # Panda scene plans with the Franka section, anything else keeps the
        # planner's UR10e default.
        planner = SPARKPlanner(
            llm_backend="gemini",
            robot_family="franka" if family == "panda" else "ur10e",
        )
        keypoint_labels = [d.label for d in detections]
        annotated = (_annotate_image(images[0]["rgb"], detections)
                     if images else None)
        try:
            from PIL import Image
            pil_image = (Image.fromarray(annotated) if annotated is not None
                         else None)
            score = planner.generate_score(instruction, pil_image, keypoint_labels)
            issues = planner.validate_score(score)
            if issues:
                print(f"[Pipeline] Score issues: {issues}")
            else:
                print(f"[Pipeline] Score generated successfully")
        except Exception as e:
            print(f"[Pipeline] Planning failed: {e}")
            score = None

    # Step 5: Execute.
    if score is not None:
        _execute_score(sim, score, detections, headless,
                       sim_executor=sim_executor)
    elif not instruction and score_yaml is None:
        # Interactive mode (legacy UR10e only).
        if panda_sim is None:
            _interactive_mode(sim, detections)
        else:
            print("[Pipeline] No instruction/score, no interactive mode for Panda")

    # Step 6: dump video if requested (Panda path).
    if video_out and panda_sim is not None:
        # A few extra settle frames before save.
        panda_sim.settle(60, snap_every=4)
        os.makedirs(os.path.dirname(video_out), exist_ok=True)
        panda_sim.save_video(video_out, fps=30)
        print(f"[Pipeline] Wrote {video_out} ({len(panda_sim.frames)} frames)")


class _PandaSimShim:
    """Wraps PandaSimWorld with the legacy StandaloneMuJoCoSim interface
    used by _get_prompts_for_scene, _get_mujoco_detections, render_camera.
    Avoids duplicating the helpers between the two sim classes."""

    def __init__(self, panda_sim):
        self._panda = panda_sim
        self.model = panda_sim.model
        self.data = panda_sim.data

    def get_object_position(self, name: str):
        return self._panda.get_body_pos(name)

    def render_camera(self, cam_idx: int, width: int = 640, height: int = 480):
        renderer = mujoco.Renderer(self.model, height, width)
        renderer.update_scene(self.data, camera=cam_idx)
        rgb = renderer.render().copy()
        renderer.enable_depth_rendering()
        depth = renderer.render().copy()
        renderer.disable_depth_rendering()
        renderer.close()
        if len(depth.shape) == 3:
            depth = depth[:, :, 0]
        return rgb, depth

    def get_camera_config(self, cam_idx: int, width: int = 640, height: int = 480):
        fovy = self.model.cam_fovy[cam_idx]
        f = height / (2.0 * np.tan(np.deg2rad(fovy) / 2.0))
        cam_pos = self.data.cam_xpos[cam_idx].copy()
        cam_mat = self.data.cam_xmat[cam_idx].reshape(3, 3).copy()
        extrinsic = np.eye(4)
        extrinsic[:3, :3] = cam_mat
        extrinsic[:3, 3] = cam_pos
        return CameraConfig(
            width=width, height=height,
            fx=f, fy=f, cx=width / 2.0, cy=height / 2.0,
            extrinsic=extrinsic,
        )

    def step(self, n: int = 1):
        self._panda.step(n)

    def viewer_sync(self):
        return None


def _get_prompts_for_scene(sim) -> list:
    """Generate detection prompts from scene objects."""
    prompts = []
    for i in range(sim.model.nbody):
        name = mujoco.mj_id2name(sim.model, mujoco.mjtObj.mjOBJ_BODY, i)
        if name and name not in ('world', 'base', 'table', 'robot_pedestal',
                                  'back_wall', 'keypoint_markers') \
                and 'link' not in name and 'driver' not in name \
                and 'coupler' not in name and 'spring' not in name \
                and 'follower' not in name and 'pad' not in name \
                and 'silicone' not in name and 'mount' not in name \
                and 'camera' not in name:
            prompts.append(name.replace('_', ' '))
    return prompts


def _get_mujoco_detections(sim) -> list:
    """Get object positions directly from MuJoCo (no perception)."""
    from spark_real.perception.sam3_detector import Detection
    detections = []
    for prompt in _get_prompts_for_scene(sim):
        body_name = prompt.replace(' ', '_')
        pos = sim.get_object_position(body_name)
        if pos is not None:
            detections.append(Detection(
                label=prompt,
                mask=np.zeros((1, 1), dtype=np.uint8),
                bbox=np.zeros(4),
                confidence=1.0,
                centroid_2d=np.zeros(2),
                position_3d=pos,
            ))
    return detections


def _annotate_image(image: np.ndarray, detections: list) -> np.ndarray:
    """Draw detection labels on image."""
    import cv2
    annotated = image.copy()
    for d in detections:
        u, v = int(d.centroid_2d[0]), int(d.centroid_2d[1])
        if 0 <= u < image.shape[1] and 0 <= v < image.shape[0]:
            cv2.circle(annotated, (u, v), 5, (0, 255, 0), -1)
            cv2.putText(annotated, d.label, (u + 10, v),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)
    return annotated


def _execute_score(sim, score: dict, detections: list, headless: bool,
                   sim_executor=None):
    """Execute a SPARK score on the simulation.

    When a `sim_executor` (MuJoCoSimExecutor) is provided, ALL primitives
    dispatch through the production
    `skill_registry.dispatch(name, executor, params)` path, the same code
    path the real-robot FastAPI server uses.

    When `sim_executor` is None (legacy UR10e path), falls back to the
    5-builtin direct dispatch, for pre-Franka scenes that ship without an
    `executor.robot` proxy.
    """
    tree = score.get("tree", {})
    det_map = {d.label: d for d in detections}

    # Production dispatch via skill registry (Franka / sim_executor)
    if sim_executor is not None:
        from spark_real.skills.registry import SkillRegistry
        registry = SkillRegistry(auto_discover=True)
        # spark_real.skills.__init__ imports primitives BEFORE manipulation,
        # so `manipulation.pour` (rotates joint 5, UR-style) overrides
        # `primitives.pour` (local-X tilt, vision-tracked), which the CoRL F1
        # pour test depends on. Force-reload primitives so the LATER
        # decorator wins. TODO: fix the import order in skills/__init__ so
        # the production server gets the same pour.
        try:
            import importlib as _il
            import spark_real.skills.primitives as _prim
            _il.reload(_prim)
            registry = SkillRegistry(auto_discover=True)
        except Exception as _exc:
            print(f"[Exec] WARN: could not force-reload primitives ({_exc})")
        # Push detections in the format skills expect: dict-of-dict with
        # 'position_3d' + 'orientation_angle' + 'aspect_ratio' + 'obb_minor_m'.
        sim_executor.update_detections(
            _detections_to_skill_format(det_map, sim))
        print(f"[Exec] Skill registry loaded: {len(registry)} skills "
              f"({len(tree.get('children', []))} BT children to dispatch)")
        for child in tree.get("children", []):
            action_type = child.get("type")
            params = child.get("params", {})
            if action_type not in registry:
                print(f"[Exec] WARN: '{action_type}' not in registry; skipping")
                continue
            print(f"[Exec] >>> dispatch '{action_type}' params={params}")
            t0 = time.time()
            try:
                result = registry.dispatch(action_type, sim_executor, params)
                print(f"[Exec] <<< {action_type}: success={getattr(result, 'success', None)} "
                      f"msg={getattr(result, 'message', '')!r} "
                      f"({time.time() - t0:.1f}s)")
            except Exception as exc:
                print(f"[Exec] !!! {action_type} raised: {exc}")
                import traceback
                traceback.print_exc()
        return

    # Legacy path: 5-builtin direct dispatch (UR10e StandaloneMuJoCoSim)
    for child in tree.get("children", []):
        action_type = child.get("type")
        params = child.get("params", {})

        if action_type == "move_to_keypoint":
            label = params.get("keypoint_label", "")
            det = det_map.get(label)
            if det and det.position_3d is not None:
                offset = np.array([
                    params.get("offset_x", 0),
                    params.get("offset_y", 0),
                    params.get("offset_z", 0),
                ])
                target = det.position_3d + offset
                # Clamp z: pinch site must be at or above object center for grasp
                target[2] = max(target[2], det.position_3d[2])

                # Always approach from above first (safe motion)
                approach = target.copy()
                approach[2] = max(target[2] + 0.12, det.position_3d[2] + 0.15)
                print(f"[Exec] Approach above {label} at {approach}")
                sim.move_to(approach, duration=1.5)
                for _ in range(100):
                    sim.step()

                # Then descend to target
                print(f"[Exec] Descend to {label} at {target}")
                sim.move_to(target, duration=1.5)
                for _ in range(200):
                    sim.step()
                if not headless:
                    sim.viewer_sync()

        elif action_type == "grasp":
            force = params.get("force", 100)
            print(f"[Exec] Grasping with force={force}")
            sim.set_gripper(min(force * 2.55, 255))  # Scale 0-100 to 0-255
            for _ in range(1000):  # Enough time for fingers to close
                sim.step()
            if not headless:
                sim.viewer_sync()

        elif action_type == "release":
            print("[Exec] Releasing")
            sim.set_gripper(0)
            for _ in range(500):
                sim.step()
            if not headless:
                sim.viewer_sync()

        elif action_type == "move_relative":
            dx = params.get("dx", 0)
            dy = params.get("dy", 0)
            dz = params.get("dz", 0)
            target = sim.ee_position + np.array([dx, dy, dz])
            print(f"[Exec] Moving relative by [{dx}, {dy}, {dz}]")
            sim.move_to(target)
            if not headless:
                sim.viewer_sync()

        elif action_type == "wait":
            duration = params.get("duration", 1.0)
            print(f"[Exec] Waiting {duration}s")
            time.sleep(duration)


def _detections_to_skill_format(det_map: dict, sim) -> dict:
    """Convert a {label: Detection} map into the dict-of-dict format the
    production skills expect.

    Skills read `det['position_3d']` plus optional `orientation_angle`,
    `aspect_ratio`, `obb_minor_m`, `world_major_axis_rad`, `slots`,
    `_mask`, `_camera`. For sim ground-truth detections we synthesise
    sensible defaults from MuJoCo bounding-box geometry where possible.
    """
    out = {}
    for label, d in det_map.items():
        if d.position_3d is None:
            continue
        entry = {
            "position_3d": np.asarray(d.position_3d, dtype=float),
            # Default orientation = 0 (object aligned with world X).
            # F2 silverware overrides via test setup if needed.
            "orientation_angle": 0.0,
            "aspect_ratio": 1.0,
            "obb_minor_m": 0.05,
            "world_major_axis_rad": 0.0,
            "slots": [],
            "_mask": None,
            "_camera": None,
        }
        out[label] = entry
    return out


def _interactive_mode(sim, detections):
    """Interactive control of simulation."""
    print("\n=== SPARK Standalone Sim ===")
    print("Objects detected:")
    for d in detections:
        if d.position_3d is not None:
            print(f"  {d.label}: ({d.position_3d[0]:.3f}, {d.position_3d[1]:.3f}, {d.position_3d[2]:.3f})")
    print("\nCommands: move <object>, grasp, release, home, step <n>, quit")

    while True:
        try:
            cmd = input("\n> ").strip()
        except (EOFError, KeyboardInterrupt):
            break

        if cmd == "quit":
            break
        elif cmd == "home":
            home = np.array([0, -np.pi/2, np.pi/2, -np.pi/2, -np.pi/2, 0])
            sim.set_joint_targets(home)
            for _ in range(1000):
                sim.step()
            sim.viewer_sync()
        elif cmd.startswith("move "):
            obj_name = cmd[5:].strip()
            det = next((d for d in detections if d.label == obj_name), None)
            if det and det.position_3d is not None:
                # Approach from above
                approach = det.position_3d.copy()
                approach[2] += 0.1
                sim.move_to(approach)
                sim.viewer_sync()
            else:
                print(f"Object '{obj_name}' not found")
        elif cmd == "grasp":
            sim.set_gripper(200)
            for _ in range(200):
                sim.step()
            sim.viewer_sync()
        elif cmd == "release":
            sim.set_gripper(0)
            for _ in range(200):
                sim.step()
            sim.viewer_sync()
        elif cmd.startswith("step"):
            parts = cmd.split()
            n = int(parts[1]) if len(parts) > 1 else 100
            sim.step(n)
            sim.viewer_sync()
            print(f"EE: {sim.ee_position}")
        elif cmd == "ee":
            print(f"EE position: {sim.ee_position}")
        elif cmd == "joints":
            q = sim.joint_positions
            for i, name in enumerate(sim.JOINT_NAMES):
                print(f"  {name}: {q[i]:.4f} rad ({np.degrees(q[i]):.1f} deg)")


@dataclass
class SimPipelineConfig:
    """SPARK standalone simulation pipeline."""
    scene: str = "perception_test_scene.xml"
    """Scene XML file name or path."""
    instruction: str = ""
    """Natural language task instruction (empty = interactive mode)."""
    headless: bool = True
    """Run without viewer (EGL rendering). Default True for offscreen scenes."""
    no_sam3: bool = True
    """Skip SAM3, use MuJoCo ground truth positions. Default True for sim-only."""
    score_yaml: str = ""
    """Optional path to a pre-baked BT score YAML to bypass the planner."""
    video_out: str = ""
    """Optional path to write a video of the simulation playback."""


def main():
    cfg = tyro.cli(SimPipelineConfig)

    scene_path = cfg.scene
    if not os.path.isabs(scene_path):
        # Try scenes/ then scenes/corl/.
        candidates = [
            SCENES_DIR / scene_path,
            SCENES_DIR / "corl" / scene_path,
        ]
        for c in candidates:
            if c.exists():
                scene_path = str(c)
                break

    run_pipeline(
        scene_path=scene_path,
        instruction=cfg.instruction,
        headless=cfg.headless,
        use_sam3=not cfg.no_sam3,
        score_yaml=cfg.score_yaml or None,
        video_out=cfg.video_out or None,
    )


if __name__ == "__main__":
    main()

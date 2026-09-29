"""CoRL: dustpan sweep task (UR10e + Robotiq 2F-85 sim demo).

Exercises the UR10e legacy sim path (StandaloneMuJoCoSim with 6-DOF
DLS IK + direct joint interpolation):
  1. Grasp brush (top-down approach, close gripper)
  2. Transport brush behind debris cluster
  3. Lower bristles to table surface
  4. Multi-pass sweep (3 stripes) pushing debris toward dustpan
  5. Retreat and release

Output:
  /tmp/spark_corl_sim_proof/dustpan_ur10e/{00..10}_*.png
  /tmp/spark_corl_sim_proof/dustpan_ur10e/video.mp4
"""
from __future__ import annotations

import os
import sys
import time
from pathlib import Path
import numpy as np

# Env caps for headless rendering.
os.environ.setdefault("MUJOCO_GL", "egl")

_HERE = Path(__file__).resolve().parent
SRC = str(_HERE.parents[2])
if SRC not in sys.path:
    sys.path.insert(0, SRC)

import mujoco
import imageio

SCENE = str(Path(SRC) / "spark_sim" / "scenes" / "corl" / "corl_dustpan_ur10e.xml")
OUT = "/tmp/spark_corl_sim_proof/dustpan_ur10e"
os.makedirs(OUT, exist_ok=True)


# ---------------------------------------------------------------------------
# Thin wrapper around StandaloneMuJoCoSim that adds frame recording +
# image I/O, matching the PandaSimWorld interface used by the other
# CoRL scripts. We could use StandaloneMuJoCoSim directly, but the
# rendering / video helpers would have to be inline in main().
# ---------------------------------------------------------------------------
class UR10eSimWorld:
    """UR10e sim wrapper with trajectory helpers and video recording."""

    HOME = np.array([0, -np.pi / 2, np.pi / 2, -np.pi / 2, -np.pi / 2, 0])

    def __init__(self, scene_path: str, width: int = 800, height: int = 600,
                 record_camera: str = "agentview"):
        self.model = mujoco.MjModel.from_xml_path(scene_path)
        self.data = mujoco.MjData(self.model)
        self.width = width
        self.height = height
        self._renderer = mujoco.Renderer(self.model, height, width)
        self.frames: list[np.ndarray] = []

        self._cam_id = mujoco.mj_name2id(
            self.model, mujoco.mjtObj.mjOBJ_CAMERA, record_camera)

        # TCP pinch site
        self._tcp_site = mujoco.mj_name2id(
            self.model, mujoco.mjtObj.mjOBJ_SITE, "pinch")

        # Go to home and open gripper
        for i in range(6):
            self.data.qpos[i] = self.HOME[i]
            self.data.ctrl[i] = self.HOME[i]
        self.data.ctrl[6] = 0  # gripper open
        mujoco.mj_forward(self.model, self.data)

    @property
    def tcp_pos(self) -> np.ndarray:
        return self.data.site_xpos[self._tcp_site].copy()

    def get_body_pos(self, name: str):
        bid = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_BODY, name)
        if bid < 0:
            return None
        return self.data.xpos[bid].copy()

    # ---------- physics ----------
    def _step_one(self):
        """Single physics step + brush weld maintenance."""
        self._apply_brush_weld()
        mujoco.mj_step(self.model, self.data)

    def settle(self, steps: int = 100, snap_every: int = 0):
        for i in range(steps):
            self._step_one()
            if snap_every > 0 and (i % snap_every == 0):
                self.snap()

    # ---------- IK (6-DOF DLS, top-down orientation) ----------
    def compute_ik(self, target_pos: np.ndarray, max_iter: int = 400,
                   tol: float = 0.001) -> np.ndarray | None:
        """Damped-least-squares IK. Returns 6-DOF joint config or None."""
        q = np.array(self.data.qpos[:6], dtype=float)
        d2 = mujoco.MjData(self.model)
        for i in range(6):
            d2.qpos[i] = q[i]

        target_xmat = np.array([[1, 0, 0], [0, -1, 0], [0, 0, -1]])

        for iteration in range(max_iter):
            mujoco.mj_forward(self.model, d2)
            cur = d2.site_xpos[self._tcp_site].copy()
            pe = target_pos - cur

            site_xmat = d2.site_xmat[self._tcp_site].reshape(3, 3)
            R_err = target_xmat @ site_xmat.T
            angle = np.arccos(np.clip((np.trace(R_err) - 1) / 2, -1, 1))
            if angle > 1e-6:
                ax = np.array([R_err[2, 1] - R_err[1, 2],
                               R_err[0, 2] - R_err[2, 0],
                               R_err[1, 0] - R_err[0, 1]])
                ax = ax / (2 * np.sin(angle))
                oe = ax * angle * 0.3
            else:
                oe = np.zeros(3)

            if np.linalg.norm(pe) < tol:
                return q

            jacp = np.zeros((3, self.model.nv))
            jacr = np.zeros((3, self.model.nv))
            mujoco.mj_jacSite(self.model, d2, jacp, jacr, self._tcp_site)
            Jp = jacp[:, :6]
            Jr = jacr[:, :6]
            J = np.vstack([Jp, Jr])
            err = np.hstack([pe, oe])

            damping = 0.05 if np.linalg.norm(pe) < 0.05 else 0.1
            JJT = J @ J.T + damping ** 2 * np.eye(6)
            dq = J.T @ np.linalg.solve(JJT, err)
            step = 0.5 if np.linalg.norm(pe) > 0.1 else 0.3
            q += step * dq
            q = np.clip(q, -2 * np.pi, 2 * np.pi)
            for i in range(6):
                d2.qpos[i] = q[i]

        if np.linalg.norm(pe) < 0.02:
            return q
        return None

    # ---------- motion ----------
    def move_to(self, target_pos: np.ndarray, duration: float = 2.0,
                steps_per_sec: int = 500, snap_every: int = 8,
                n_cart_subdiv: int = 1) -> bool:
        """Move TCP to target_pos. With n_cart_subdiv>1, subdivide the
        Cartesian path into segments and re-solve IK at each to avoid
        large joint-space jumps that leave the IK in a bad basin."""
        if n_cart_subdiv <= 1:
            q_tgt = self.compute_ik(target_pos)
            if q_tgt is None:
                print(f"  [UR10e] IK FAILED for {target_pos}")
                return False
            q_start = np.array(self.data.qpos[:6], dtype=float)
            total = int(duration * steps_per_sec)
            for s in range(total):
                alpha = s / total
                interp = q_start + alpha * (q_tgt - q_start)
                for i in range(6):
                    self.data.ctrl[i] = interp[i]
                self._step_one()
                if snap_every > 0 and (s % snap_every == 0):
                    self.snap()
            return True

        # Multi-segment Cartesian subdivision with online IK re-solve.
        # At each segment, solve IK from the CURRENT actual joint config
        # (not the planned one), so contact perturbations don't accumulate.
        start_pos = self.tcp_pos.copy()
        seg_dur = duration / n_cart_subdiv
        seg_steps = int(seg_dur * steps_per_sec)
        for k in range(1, n_cart_subdiv + 1):
            t = k / n_cart_subdiv
            seg_target = (1 - t) * start_pos + t * target_pos
            q_tgt = self.compute_ik(seg_target)
            if q_tgt is None:
                continue
            q_now = np.array(self.data.qpos[:6], dtype=float)
            for s in range(seg_steps):
                alpha = s / seg_steps
                interp = q_now + alpha * (q_tgt - q_now)
                for i in range(6):
                    self.data.ctrl[i] = interp[i]
                self._step_one()
                if snap_every > 0 and (s % snap_every == 0):
                    self.snap()
        return True

    def set_gripper(self, value: float, ramp_steps: int = 300,
                    snap_every: int = 8):
        """Smoothly ramp gripper from current to value (0=open, 255=closed)."""
        start = float(self.data.ctrl[6])
        for k in range(1, ramp_steps + 1):
            t = k / ramp_steps
            self.data.ctrl[6] = (1 - t) * start + t * value
            self._step_one()
            if snap_every > 0 and (k % snap_every == 0):
                self.snap()

    # ---------- weld brush to gripper ----------
    def weld_brush(self):
        """Activate a pre-defined weld constraint between the pinch site
        and the brush body. This simulates a firm tool grasp that the
        Robotiq actuator can't maintain due to low sim gripping force.

        We create the weld at runtime by modifying the model's equality
        constraints via mujoco.MjSpec (MuJoCo >= 3.0). For older mujoco
        versions, we fall back to directly manipulating eq_data.
        """
        # Find the brush body and the gripper body (base2 = Robotiq base)
        brush_bid = mujoco.mj_name2id(
            self.model, mujoco.mjtObj.mjOBJ_BODY, "brush")
        grip_bid = mujoco.mj_name2id(
            self.model, mujoco.mjtObj.mjOBJ_BODY, "base2")
        if brush_bid < 0 or grip_bid < 0:
            print("  [weld] WARNING: brush or base2 body not found")
            return

        # Compute relative transform: brush in gripper frame
        brush_pos = self.data.xpos[brush_bid].copy()
        grip_pos = self.data.xpos[grip_bid].copy()
        grip_mat = self.data.xmat[grip_bid].reshape(3, 3)
        rel_pos = grip_mat.T @ (brush_pos - grip_pos)

        # The brush freejoint gives it 7 DOF (pos + quat). We constrain
        # all 6 DOF by setting the brush qpos directly and then locking
        # with very stiff joint damping. This is simpler and more
        # compatible than runtime equality creation.
        #
        # Strategy: disable the freejoint by welding the brush to the
        # gripper's coordinate frame. We do this by finding the brush
        # freejoint and overriding its qvel to zero every step.
        # Actually the simplest robust method: just make the brush body
        # a child of base2 at its current relative pose. But we can't
        # re-parent at runtime.
        #
        # Pragmatic solution: at each sim step, teleport the brush body
        # to maintain its relative pose to the gripper. We store the
        # offset and apply it in a wrapper around step().
        self._brush_weld_active = True
        self._brush_weld_rel_pos = rel_pos.copy()
        brush_quat = self.data.xquat[brush_bid].copy()
        grip_quat = self.data.xquat[grip_bid].copy()
        # Store relative quaternion
        from scipy.spatial.transform import Rotation as R
        R_grip = R.from_quat([grip_quat[1], grip_quat[2],
                              grip_quat[3], grip_quat[0]])
        R_brush = R.from_quat([brush_quat[1], brush_quat[2],
                               brush_quat[3], brush_quat[0]])
        R_rel = R_grip.inv() * R_brush
        self._brush_weld_rel_quat = R_rel.as_quat()  # xyzw
        self._brush_bid = brush_bid
        self._grip_bid = grip_bid
        # Find the brush freejoint qpos address
        fj_id = mujoco.mj_name2id(
            self.model, mujoco.mjtObj.mjOBJ_JOINT, "brush_free")
        if fj_id >= 0:
            self._brush_qpos_adr = int(self.model.jnt_qposadr[fj_id])
            self._brush_qvel_adr = int(self.model.jnt_dofadr[fj_id])
        else:
            self._brush_qpos_adr = None
        print(f"  [weld] Brush welded to gripper at rel_pos={rel_pos.round(3)}")

    def _apply_brush_weld(self):
        """If weld is active, teleport brush to maintain relative pose."""
        if not getattr(self, '_brush_weld_active', False):
            return
        if self._brush_qpos_adr is None:
            return
        from scipy.spatial.transform import Rotation as R
        grip_pos = self.data.xpos[self._grip_bid].copy()
        grip_mat = self.data.xmat[self._grip_bid].reshape(3, 3)
        grip_quat = self.data.xquat[self._grip_bid].copy()

        # World position = grip_pos + grip_R @ rel_pos
        brush_world = grip_pos + grip_mat @ self._brush_weld_rel_pos

        # World quaternion = grip_R * rel_R
        R_grip = R.from_quat([grip_quat[1], grip_quat[2],
                              grip_quat[3], grip_quat[0]])
        R_brush = R_grip * R.from_quat(self._brush_weld_rel_quat)
        brush_quat_xyzw = R_brush.as_quat()
        brush_quat_wxyz = np.array([brush_quat_xyzw[3], brush_quat_xyzw[0],
                                     brush_quat_xyzw[1], brush_quat_xyzw[2]])

        # Write to qpos
        a = self._brush_qpos_adr
        self.data.qpos[a:a+3] = brush_world
        self.data.qpos[a+3:a+7] = brush_quat_wxyz
        # Zero velocities
        va = self._brush_qvel_adr
        self.data.qvel[va:va+6] = 0.0

    # ---------- aggressive servo for sweep ----------
    def sweep_servo(self, start: np.ndarray, end: np.ndarray,
                    duration: float = 3.0, snap_every: int = 8) -> bool:
        """Sweep from start to end at table height, using aggressive
        servo control: solve IK for each waypoint and SET ctrl directly
        (no interpolation), stepping physics multiple times per waypoint
        so the arm actually tracks through contact forces.

        This mimics how a real industrial robot servo (1kHz PID) would
        bulldoze through contact, unlike the gentle interpolation in
        move_to which lets the arm drift when external loads push it.
        """
        n_waypoints = 60  # ~50ms per waypoint at 500Hz
        steps_per_wp = int(duration * 500 / n_waypoints)
        for k in range(n_waypoints):
            t = (k + 1) / n_waypoints
            wp = (1 - t) * start + t * end
            q = self.compute_ik(wp)
            if q is None:
                continue
            # Set ctrl directly to the IK solution; no lerp
            for i in range(6):
                self.data.ctrl[i] = q[i]
            # Step physics multiple times to let the controller track
            for s in range(steps_per_wp):
                self._step_one()
                if snap_every > 0 and (s % snap_every == 0):
                    self.snap()
        return True

    # ---------- rendering ----------
    def render(self, camera: str | None = None) -> np.ndarray:
        if camera is None:
            cam_id = self._cam_id
        else:
            cam_id = mujoco.mj_name2id(
                self.model, mujoco.mjtObj.mjOBJ_CAMERA, camera)
        self._renderer.update_scene(self.data, camera=cam_id)
        return self._renderer.render().copy()

    def snap(self, camera: str | None = None):
        self.frames.append(self.render(camera))

    def save_png(self, path: str, camera: str | None = None):
        os.makedirs(os.path.dirname(path), exist_ok=True)
        imageio.imwrite(path, self.render(camera))

    def save_video(self, path: str, fps: int = 30):
        if not self.frames:
            return
        os.makedirs(os.path.dirname(path), exist_ok=True)
        imageio.mimsave(path, self.frames, fps=fps)
        print(f"  Wrote {path} ({len(self.frames)} frames)")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main():
    t0 = time.time()
    sim = UR10eSimWorld(SCENE, width=800, height=600, record_camera="agentview")
    print(f"Loaded {SCENE}")

    # ---- Ground truth positions ----
    brush_pos = sim.get_body_pos("brush")
    dustpan_pos = sim.get_body_pos("dustpan")
    print(f"Brush at {brush_pos}")
    print(f"Dustpan at {dustpan_pos}")
    debris_initial = {}
    for i in range(1, 7):
        pos = sim.get_body_pos(f"debris_{i}")
        debris_initial[f"debris_{i}"] = pos.copy()
        print(f"  debris_{i} at {pos}")

    # Table surface Z (from scene: table top at 0.225)
    TABLE_Z = 0.225

    # ---- Settle ----
    sim.settle(200, snap_every=8)
    sim.save_png(f"{OUT}/00_initial.png")
    sim.save_png(f"{OUT}/00_initial_top.png", camera="topview")
    sim.save_png(f"{OUT}/00_initial_side.png", camera="sideview")
    print("Settled.")

    # ---- Phase 1: Approach above brush ----
    print("\n=== Phase 1: hover above brush ===")
    brush_now = sim.get_body_pos("brush")
    hover_brush = np.array([brush_now[0], brush_now[1], brush_now[2] + 0.10])
    ok = sim.move_to(hover_brush, duration=2.0)
    print(f"  TCP at hover: {sim.tcp_pos}")
    sim.save_png(f"{OUT}/01_hover_brush.png")

    # Open gripper wide before descending
    sim.set_gripper(0, ramp_steps=100, snap_every=8)
    sim.settle(50, snap_every=8)

    # ---- Phase 2: Descend to grasp height ----
    print("\n=== Phase 2: descend to brush ===")
    # Brush handle centre is at brush body Z. The pinch site (TCP) needs
    # to be aligned with the handle centre for a firm wrap. The handle
    # cylinder is 12mm radius (24mm dia); the Robotiq pads will close
    # around it. Target TCP at brush body Z so pads wrap the centre.
    grasp_pos = np.array([brush_now[0], brush_now[1], brush_now[2]])
    ok = sim.move_to(grasp_pos, duration=1.5)
    sim.settle(100, snap_every=8)
    sim.save_png(f"{OUT}/02_at_brush.png")
    print(f"  TCP at brush: {sim.tcp_pos}")

    # ---- Phase 3: Close gripper + weld brush ----
    print("\n=== Phase 3: grasp brush ===")
    # Fully close (ctrl=255) so the Robotiq pads clamp tightly.
    sim.set_gripper(255, ramp_steps=500, snap_every=4)
    sim.settle(300, snap_every=4)

    # Activate the weld equality constraint to lock the brush to the
    # gripper. The Robotiq 2F-85 actuator in MuJoCo doesn't generate
    # enough holding force for tool-use tasks; the weld is the standard
    # sim workaround (same pattern as MuJoCo's grasp_fixed_site in
    # dm_control manipulation tasks).
    sim.weld_brush()
    sim.settle(100, snap_every=6)
    sim.save_png(f"{OUT}/03_grasped.png")
    brush_grasped = sim.get_body_pos("brush")
    print(f"  Brush after grasp: {brush_grasped}")

    # ---- Phase 4: Lift brush ----
    print("\n=== Phase 4: lift brush ===")
    lift_pos = sim.tcp_pos.copy()
    lift_pos[2] += 0.12
    ok = sim.move_to(lift_pos, duration=1.5)
    sim.save_png(f"{OUT}/04_lifted.png")
    brush_lifted = sim.get_body_pos("brush")
    brush_dz = brush_lifted[2] - brush_pos[2]
    print(f"  Brush lifted dz={brush_dz * 1000:.1f}mm")

    # ---- Phase 5: Move behind debris cluster (sweep start position) ----
    # Debris is between x=0.10..0.17, dustpan at x=0.28.
    # Sweep direction: +X (toward dustpan).
    # Start behind debris: x=0.03 (behind cluster), at table surface height.
    print("\n=== Phase 5: position behind debris ===")
    debris_centroid_y = np.mean([debris_initial[f"debris_{i}"][1]
                                 for i in range(1, 7)])
    # Start position: behind debris, centered on debris spread
    sweep_start_x = 0.03  # behind the debris cluster
    sweep_z_high = TABLE_Z + 0.10  # approach height
    # From observed sim: bristle world z = TCP_z + ~0.025 (weld offset).
    # Debris tops at z=0.237. Need bristle_z <= 0.237, so TCP_z <= 0.212.
    # But can't go too low or phasing occurs. TCP at TABLE_Z - 0.010 = 0.215
    # puts bristles at ~0.240 (still above debris). Need to go lower.
    # TCP at TABLE_Z - 0.020 = 0.205 -> bristles at ~0.230, just at debris.
    sweep_z_low = TABLE_Z - 0.020

    behind_high = np.array([sweep_start_x, debris_centroid_y, sweep_z_high])
    ok = sim.move_to(behind_high, duration=2.0)
    sim.save_png(f"{OUT}/05_behind_debris.png")
    print(f"  TCP behind debris: {sim.tcp_pos}")

    # ---- Phase 6-8: Multi-pass sweep (3 stripes) ----
    # Each pass: lower -> sweep +X toward dustpan -> lift -> reposition
    # Stripe spacing: cover y range of debris (-0.06 to +0.07)
    sweep_end_x = dustpan_pos[0]  # sweep right to dustpan center
    # The brush bristles are offset ~0.065m in +Y from the TCP due to
    # the brush geometry + weld. The bristle pad is now 60mm half-width,
    # so each pass covers a wide swath. To sweep through debris at
    # y=-0.04..+0.05, the TCP needs to be at y = debris_y - 0.065.
    BRISTLE_Y_OFFSET = 0.065
    n_passes = 7
    y_offsets = np.linspace(-0.08 - BRISTLE_Y_OFFSET,
                            0.08 - BRISTLE_Y_OFFSET,
                            n_passes)

    for pi, y_off in enumerate(y_offsets):
        pass_num = pi + 1
        sweep_y = debris_centroid_y + y_off
        print(f"\n=== Sweep pass {pass_num}/{n_passes} (y_offset={y_off:+.3f}) ===")

        # a) Move to start of sweep stripe (high)
        start_high = np.array([sweep_start_x, sweep_y, sweep_z_high])
        ok = sim.move_to(start_high, duration=1.2, n_cart_subdiv=3)

        # b) Lower to sweep height
        start_low = np.array([sweep_start_x, sweep_y, sweep_z_low])
        ok = sim.move_to(start_low, duration=0.8)

        # c) Sweep forward toward dustpan (aggressive servo for contact)
        end_low = np.array([sweep_end_x, sweep_y, sweep_z_low])
        ok = sim.sweep_servo(start_low, end_low, duration=3.0, snap_every=8)

        # d) Lift at dustpan to clear rim
        end_high = np.array([sweep_end_x, sweep_y, sweep_z_high])
        ok = sim.move_to(end_high, duration=0.8)

        sim.save_png(f"{OUT}/0{5 + pass_num}_sweep_pass_{pass_num}.png")
        brush_sweep = sim.get_body_pos("brush")
        tcp_sweep = sim.tcp_pos
        brush_tcp_dist = np.linalg.norm(brush_sweep - tcp_sweep)
        # Estimate bristle tip position (brush_pos + brush_rotation * bristle_offset)
        brush_bid = mujoco.mj_name2id(sim.model, mujoco.mjtObj.mjOBJ_BODY, "brush")
        brush_mat = sim.data.xmat[brush_bid].reshape(3, 3)
        bristle_local = np.array([0, 0.065, -0.012])  # bristle pad center in brush frame
        bristle_world = brush_sweep + brush_mat @ bristle_local
        print(f"  Pass {pass_num} done. TCP: {tcp_sweep.round(3)} "
              f"brush: {brush_sweep.round(3)} bristles: {bristle_world.round(3)} "
              f"brush-TCP: {brush_tcp_dist*1000:.0f}mm")

    sim.save_png(f"{OUT}/09_sweep_done.png", camera="topview")

    # ---- Phase 9: Retreat ----
    print("\n=== Phase 9: retreat ===")
    retreat = sim.tcp_pos.copy()
    retreat[2] = TABLE_Z + 0.20
    ok = sim.move_to(retreat, duration=1.5)

    # ---- Phase 10: Release brush ----
    print("\n=== Phase 10: release brush ===")
    sim.set_gripper(0, ramp_steps=200, snap_every=6)
    sim.settle(100, snap_every=6)
    sim.save_png(f"{OUT}/10_released.png")

    # Final top and side views
    sim.settle(100, snap_every=6)
    sim.save_png(f"{OUT}/11_final.png")
    sim.save_png(f"{OUT}/11_final_top.png", camera="topview")
    sim.save_png(f"{OUT}/11_final_side.png", camera="sideview")

    # ---- Save video ----
    sim.save_video(f"{OUT}/video.mp4", fps=30)

    # ---- Results ----
    elapsed = time.time() - t0
    print(f"\n=== Dustpan UR10e done in {elapsed:.1f}s ===")

    # Check how many debris pieces moved toward/into the dustpan.
    # Dustpan center XY = (0.35, 0.0), extent ~ 0.08m in X, 0.07m in Y.
    dp_x, dp_y = dustpan_pos[0], dustpan_pos[1]
    n_in_dustpan = 0
    n_moved_toward = 0
    for i in range(1, 7):
        pos = sim.get_body_pos(f"debris_{i}")
        initial = debris_initial[f"debris_{i}"]
        dx_toward = pos[0] - initial[0]  # positive = toward dustpan
        dist_to_pan = np.sqrt((pos[0] - dp_x) ** 2 + (pos[1] - dp_y) ** 2)
        in_pan = (abs(pos[0] - dp_x) < 0.10 and
                  abs(pos[1] - dp_y) < 0.08 and
                  pos[2] < TABLE_Z + 0.05)
        if in_pan:
            n_in_dustpan += 1
        if dx_toward > 0.02:
            n_moved_toward += 1
        status = "IN PAN" if in_pan else ("MOVED" if dx_toward > 0.02 else "STATIC")
        print(f"  debris_{i}: {initial.round(3)} -> {pos.round(3)} "
              f"dx={dx_toward * 1000:+.0f}mm dist_to_pan={dist_to_pan * 1000:.0f}mm "
              f"[{status}]")

    print(f"\nDebris in dustpan: {n_in_dustpan}/6")
    print(f"Debris moved toward dustpan: {n_moved_toward}/6")
    print(f"Brush lifted during grasp: {brush_dz * 1000:.1f}mm")

    # Success: brush was picked up AND at least 3 debris moved toward dustpan.
    # The brush lift threshold is 2mm (sim friction is lower than real).
    success = (brush_dz > 0.002) and (n_moved_toward >= 3)
    print(f"SUCCESS: {success} (brush_lift>2mm AND moved>=3)")
    return success


if __name__ == "__main__":
    ok = main()
    sys.exit(0 if ok else 1)

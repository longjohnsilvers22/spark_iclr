"""SafeRobot: CBF safety filter wrapper (robot-agnostic).

Drop-in replacement that intercepts velocity and script commands, solves a
Control Barrier Function QP at every tick to enforce kinematic, obstacle,
and contact-force safety constraints, then forwards the (possibly modified)
command to the underlying robot.

Dependencies: pip install osqp scipy numpy
"""

from __future__ import annotations

import logging
import re
import threading
import time
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

import numpy as np

try:
    import osqp
    import scipy.sparse as sp
    from scipy import ndimage as _ndimage

    _OSQP_AVAILABLE = True
except ImportError:
    _OSQP_AVAILABLE = False
    _ndimage = None

logger = logging.getLogger(__name__)


@dataclass
class SafetyConfig:
    """
    Tuneable parameters for the CBF safety filter.
    """

    eta_kinematic: float = 0.3
    eta_obstacle: float = 0.5
    eta_force: float = 0.8
    vel_lin_max: float = 0.25
    vel_ang_max: float = 1.0
    # Fallback box used only when SafeRobot is built with config=None. Kept
    # as literals rather than unified with ur10e_default.yaml: this z_min is
    # -0.25 while the live UR10e SafetyConfig uses -0.27, so merging would
    # change a value.
    ws_min: np.ndarray = field(default_factory=lambda: np.array([-1.1, -0.5, -0.25]))
    ws_max: np.ndarray = field(default_factory=lambda: np.array([-0.5, 0.7, 0.50]))
    reach_max: float = 1.15
    eps_singularity: float = 0.20
    r_safe: float = 0.10
    f_threshold: float = 10.0
    f_max: float = 30.0
    force_baseline_samples: int = 20
    stale_timeout: float = 0.5
    depth_hz: float = 30.0
    movel_path_samples: int = 5


@dataclass
class Obstacle:
    """
    Single detected obstacle (sphere approximation).
    """

    position: np.ndarray
    radius: float
    timestamp: float = 0.0


class ObstacleMap:
    """
    Thread-safe obstacle list using atomic ref swap under the GIL.
    """

    def __init__(self, stale_timeout: float = 0.5):
        self._stale_timeout = stale_timeout
        self._obstacles: List[Obstacle] = []

    def update(self, obstacles: List[Obstacle]) -> None:
        self._obstacles = obstacles

    def get(self) -> List[Obstacle]:
        now = time.monotonic()
        return [o for o in self._obstacles if (now - o.timestamp) < self._stale_timeout]

    def clear(self) -> None:
        self._obstacles = []


class BarrierSet:
    """
    Collect all barrier functions and linearise them for the QP.

    Each barrier returns a row (a, b) such that the CBF constraint is
    a @ u >= b where u is the 6-D Cartesian velocity command.
    """

    def __init__(self, cfg: SafetyConfig):
        self.cfg = cfg

    def _reach_barrier(self, p_tcp: np.ndarray) -> Optional[tuple]:
        # h_reach = R_max^2 - ||p||^2
        r2 = float(np.dot(p_tcp, p_tcp))
        R2 = self.cfg.reach_max**2
        h = R2 - r2
        if h > R2:
            return None
        a = np.zeros(6)
        a[:3] = -2.0 * p_tcp
        return a, -self.cfg.eta_kinematic * h

    def _workspace_barriers(self, p_tcp: np.ndarray) -> List[tuple]:
        rows: List[tuple] = []
        eta = self.cfg.eta_kinematic
        for axis in range(3):
            h_lo = p_tcp[axis] - self.cfg.ws_min[axis]
            a_lo = np.zeros(6)
            a_lo[axis] = 1.0
            rows.append((a_lo, -eta * h_lo))
            h_hi = self.cfg.ws_max[axis] - p_tcp[axis]
            a_hi = np.zeros(6)
            a_hi[axis] = -1.0
            rows.append((a_hi, -eta * h_hi))
        return rows

    def _singularity_barrier(self, joints: np.ndarray, dt: float) -> Optional[tuple]:
        """
        UR-elbow specific; disabled when eps_singularity <= 0.
        """
        if self.cfg.eps_singularity <= 0:
            return None
        q2 = float(joints[2]) if len(joints) > 2 else 0.0
        eps2 = self.cfg.eps_singularity**2
        h = (q2**2) * ((q2 - np.pi) ** 2) - eps2
        if h > 10.0 * eps2:
            return None
        dh_dq2 = 2.0 * q2 * ((q2 - np.pi) ** 2) + (q2**2) * 2.0 * (q2 - np.pi)
        scale = max(abs(dh_dq2), 1e-6)
        alpha = max(self.cfg.eta_kinematic * max(h, 0.0) / scale, 0.01)
        rows: List[tuple] = []
        for axis in range(6):
            a_pos = np.zeros(6)
            a_pos[axis] = -1.0
            rows.append((a_pos, -alpha))
            a_neg = np.zeros(6)
            a_neg[axis] = 1.0
            rows.append((a_neg, -alpha))
        return rows

    def _obstacle_barriers(
        self, p_tcp: np.ndarray, obstacles: List[Obstacle]
    ) -> List[tuple]:
        rows: List[tuple] = []
        eta = self.cfg.eta_obstacle
        r_safe = self.cfg.r_safe
        for obs in obstacles:
            diff = p_tcp - obs.position
            dist2 = float(np.dot(diff, diff))
            margin = (r_safe + obs.radius) ** 2
            h = dist2 - margin
            a = np.zeros(6)
            a[:3] = 2.0 * diff
            rows.append((a, -eta * h))
        return rows

    def _force_barrier(
        self, f_tcp: np.ndarray, f_baseline: np.ndarray
    ) -> Optional[tuple]:
        f_net = f_tcp[:3] - f_baseline[:3]
        f_mag2 = float(np.dot(f_net, f_net))
        F2 = self.cfg.f_max**2
        h = F2 - f_mag2
        f_mag = np.sqrt(f_mag2)
        if f_mag < self.cfg.f_threshold:
            return None
        f_hat = f_net / max(f_mag, 1e-8)
        a = np.zeros(6)
        a[:3] = -f_hat
        return a, self.cfg.eta_force * h

    def evaluate(
        self, p_tcp, joints, obstacles, f_tcp, f_baseline, dt=0.1, enable_force=True
    ) -> tuple:
        """
        Return (A_cbf, b_cbf, obstacle_active) for the QP.
        """
        rows: List[tuple] = []
        obstacle_active = False

        rb = self._reach_barrier(p_tcp)
        if rb is not None:
            rows.append(rb)

        rows.extend(self._workspace_barriers(p_tcp))

        sb = self._singularity_barrier(joints, dt)
        if sb is not None:
            if isinstance(sb, list):
                rows.extend(sb)
            else:
                rows.append(sb)

        obs_rows = self._obstacle_barriers(p_tcp, obstacles)
        if obs_rows:
            obstacle_active = True
            rows.extend(obs_rows)

        if enable_force and f_tcp is not None and f_baseline is not None:
            fb = self._force_barrier(f_tcp, f_baseline)
            if fb is not None:
                rows.append(fb)

        if not rows:
            return np.zeros((0, 6)), np.zeros(0), obstacle_active
        A = np.array([r[0] for r in rows])
        b = np.array([r[1] for r in rows])
        return A, b, obstacle_active

    def check_pose_safe(
        self, p_tcp: np.ndarray, joints: Optional[np.ndarray] = None
    ) -> tuple:
        """
        Check whether a static TCP pose violates any kinematic barrier.
        """
        violations: List[str] = []
        r2 = float(np.dot(p_tcp, p_tcp))
        if r2 > self.cfg.reach_max**2:
            violations.append(
                f"reach: ||p||={np.sqrt(r2):.3f}m > {self.cfg.reach_max}m"
            )
        for axis, name in enumerate("xyz"):
            if p_tcp[axis] < self.cfg.ws_min[axis]:
                violations.append(
                    f"workspace: {name}={p_tcp[axis]:.3f} < {self.cfg.ws_min[axis]}"
                )
            if p_tcp[axis] > self.cfg.ws_max[axis]:
                violations.append(
                    f"workspace: {name}={p_tcp[axis]:.3f} > {self.cfg.ws_max[axis]}"
                )
        if joints is not None and len(joints) > 2 and self.cfg.eps_singularity > 0:
            q2 = float(joints[2])
            h = (q2**2) * ((q2 - np.pi) ** 2) - self.cfg.eps_singularity**2
            if h < 0:
                violations.append(f"singularity: h={h:.4f}, q2={q2:.3f}")
        return len(violations) == 0, violations


class DepthObstacleDetector:
    """
    Background thread that processes depth frames into obstacle spheres.
    """

    def __init__(
        self, cameras: Dict[str, Any], obstacle_map: ObstacleMap, cfg: SafetyConfig
    ):
        self._cameras = cameras
        self._map = obstacle_map
        self._cfg = cfg
        self._running = False
        self._thread: Optional[threading.Thread] = None
        self._reference_depth: Optional[np.ndarray] = None
        self._capture_reference()

    def _capture_reference(self, n_frames: int = 10):
        cam = self._cameras.get("birdview") or self._cameras.get("base")
        if cam is None:
            logger.warning(
                "DepthObstacleDetector: no camera, obstacle detection disabled"
            )
            return
        frames = []
        for _ in range(n_frames):
            try:
                capture = cam.read() if callable(getattr(cam, "read", None)) else None
                if capture is None:
                    continue
                depth = capture.get("depth") if isinstance(capture, dict) else capture
                if depth is not None:
                    frames.append(depth.astype(np.float32))
            except Exception:
                pass
            time.sleep(0.05)
        if frames:
            self._reference_depth = np.mean(frames, axis=0)
            logger.info("Depth reference captured (%d frames)", len(frames))

    def start(self):
        if self._running:
            return
        self._running = True
        self._thread = threading.Thread(
            target=self._loop, daemon=True, name="depth-obstacle-detector"
        )
        self._thread.start()

    def stop(self):
        self._running = False
        if self._thread is not None:
            self._thread.join(timeout=2.0)
            self._thread = None

    def _loop(self):
        interval = 1.0 / max(self._cfg.depth_hz, 1.0)
        cam = self._cameras.get("birdview") or self._cameras.get("base")
        while self._running:
            t0 = time.monotonic()
            try:
                if cam is not None and self._reference_depth is not None:
                    self._map.update(self._process_frame(cam))
            except Exception as exc:
                logger.debug("Depth processing error: %s", exc)
            elapsed = time.monotonic() - t0
            time.sleep(max(interval - elapsed, 0.001))

    def _process_frame(self, cam) -> List[Obstacle]:
        try:
            capture = cam.read()
        except Exception:
            return []
        depth = capture.get("depth") if isinstance(capture, dict) else capture
        if depth is None:
            return []
        depth_f = depth.astype(np.float32)
        diff = np.abs(depth_f - self._reference_depth)
        mask = diff > 0.05
        if not np.any(mask):
            return []
        if _ndimage is not None:
            labelled, n_labels = _ndimage.label(mask)
        else:
            n_labels = 1
            labelled = mask.astype(np.int32)

        obstacles: List[Obstacle] = []
        now = time.monotonic()
        h, w = depth_f.shape[:2]
        for label_id in range(1, n_labels + 1):
            ys, xs = np.where(labelled == label_id)
            if len(xs) < 20:
                continue
            cx, cy = float(xs.mean()), float(ys.mean())
            d = float(np.median(depth_f[ys, xs]))
            if d < 0.05 or d > 3.0:
                continue
            pixel_extent = max(xs.max() - xs.min(), ys.max() - ys.min())
            radius = (pixel_extent / 640.0) * d * 0.5
            fx = fy = 500.0
            ox, oy = w / 2.0, h / 2.0
            pos = np.array([(cx - ox) * d / fx, (cy - oy) * d / fy, d])
            obstacles.append(Obstacle(position=pos, radius=radius, timestamp=now))
        return obstacles


class SafeRobot:
    """
    Drop-in safety wrapper for any driver implementing the shim interface.

    Intercepts send_velocity (CBF-QP filter), move_linear/move_linear_relative
    (workspace clamp), and _send_script (UR-only URScript rewriting).
    All other methods pass through via __getattr__.
    """

    def __init__(
        self,
        robot,
        cameras: Optional[Dict[str, Any]] = None,
        enable_obstacles: bool = True,
        enable_force: bool = True,
        config: Optional[SafetyConfig] = None,
    ):
        self._robot = robot
        self._cfg = config or SafetyConfig()
        self._enable_force = enable_force
        self._barriers = BarrierSet(self._cfg)
        self._obstacle_map = ObstacleMap(stale_timeout=self._cfg.stale_timeout)

        self._depth_detector: Optional[DepthObstacleDetector] = None
        if cameras and enable_obstacles:
            self._depth_detector = DepthObstacleDetector(
                cameras, self._obstacle_map, self._cfg
            )
            self._depth_detector.start()

        self._f_baseline: Optional[np.ndarray] = None
        self._f_baseline_count: int = 0
        self._f_baseline_acc: Optional[np.ndarray] = None
        self.cbf_deviated: bool = False
        self._solver: Optional[Any] = None
        self._solver_n_constraints: int = 0

        lim_lin = self._cfg.vel_lin_max
        lim_ang = self._cfg.vel_ang_max
        self._u_min = np.array([-lim_lin] * 3 + [-lim_ang] * 3)
        self._u_max = np.array([lim_lin] * 3 + [lim_ang] * 3)

        if not _OSQP_AVAILABLE:
            logger.warning("osqp/scipy not installed; CBF-QP disabled")
        logger.info(
            "SafeRobot init (obstacles=%s, force=%s, osqp=%s)",
            enable_obstacles,
            enable_force,
            _OSQP_AVAILABLE,
        )

    # State reading helpers

    def _read_tcp_position(self) -> np.ndarray:
        try:
            obs = self._robot.get_observation()
            pose = obs.get("tcp_pose")
            if pose is not None:
                return np.array(pose[:3], dtype=np.float64)
        except Exception:
            pass
        try:
            p = self._robot.get_tcp_pose()
            if isinstance(p, np.ndarray) and p.shape == (4, 4):
                return p[:3, 3].copy()
            return np.array(p[:3], dtype=np.float64)
        except Exception:
            return np.zeros(3)

    def _read_joints(self) -> np.ndarray:
        try:
            return np.array(
                self._robot.get_observation()["joint_positions"], dtype=np.float64
            )
        except Exception:
            return np.zeros(6)

    def _read_force(self) -> Optional[np.ndarray]:
        if hasattr(self._robot, "get_tcp_force"):
            try:
                return np.array(self._robot.get_tcp_force(), dtype=np.float64)
            except Exception:
                pass
        return None

    def _update_force_baseline(self, f_tcp: np.ndarray) -> None:
        if self._f_baseline is not None:
            return
        if self._f_baseline_acc is None:
            self._f_baseline_acc = np.zeros(6)
        self._f_baseline_acc += f_tcp
        self._f_baseline_count += 1
        if self._f_baseline_count >= self._cfg.force_baseline_samples:
            self._f_baseline = self._f_baseline_acc / self._f_baseline_count
            logger.info("Force baseline calibrated: %s", self._f_baseline)

    # QP solver

    def _build_and_solve_qp(
        self, u_des: np.ndarray, A_cbf: np.ndarray, b_cbf: np.ndarray
    ) -> Optional[np.ndarray]:
        if not _OSQP_AVAILABLE:
            return None
        n = 6
        n_cbf = A_cbf.shape[0] if A_cbf.ndim == 2 else 0
        P = sp.eye(n, format="csc") * 2.0
        q = -2.0 * u_des
        A_top = A_cbf if n_cbf > 0 else np.zeros((0, n))
        A_full = np.vstack([A_top, np.eye(n)])
        A_sp = sp.csc_matrix(A_full)
        l_cbf = b_cbf if n_cbf > 0 else np.array([])
        u_cbf = np.full(n_cbf, np.inf)
        l_full = np.concatenate([l_cbf, self._u_min])
        u_full = np.concatenate([u_cbf, self._u_max])
        total = n_cbf + n

        try:
            if self._solver is not None and self._solver_n_constraints == total:
                self._solver.update(q=q, l=l_full, u=u_full, Ax=A_sp.data)
            else:
                self._solver = osqp.OSQP()
                self._solver.setup(
                    P=P,
                    q=q,
                    A=A_sp,
                    l=l_full,
                    u=u_full,
                    verbose=False,
                    warm_starting=True,
                    max_iter=200,
                    eps_abs=1e-4,
                    eps_rel=1e-4,
                    polish=False,
                    adaptive_rho=True,
                )
                self._solver_n_constraints = total
            result = self._solver.solve()
            if result.info.status in ("solved", "solved_inaccurate"):
                return np.array(result.x, dtype=np.float64)
            logger.warning("QP status: %s", result.info.status)
            return None
        except Exception as exc:
            logger.error("QP solver error: %s", exc)
            self._solver = None
            self._solver_n_constraints = 0
            return None

    # Capability checks

    def _supports_velocity_streaming(self) -> bool:
        flag = getattr(self._robot, "SUPPORTS_VELOCITY_STREAMING", None)
        if flag is not None:
            return bool(flag)
        return callable(getattr(self._robot, "send_velocity", None))

    def _supports_urscript(self) -> bool:
        flag = getattr(self._robot, "SUPPORTS_URSCRIPT", None)
        if flag is not None:
            return bool(flag)
        return callable(getattr(self._robot, "_send_script", None))

    def _forward_velocity(
        self, u: np.ndarray, acceleration: float, time_duration: float
    ) -> None:
        if self._supports_velocity_streaming():
            try:
                self._robot.send_velocity(
                    u.tolist(), acceleration=acceleration, duration=time_duration
                )
            except TypeError:
                self._robot.send_velocity(
                    u.tolist(), acceleration=acceleration, time_duration=time_duration
                )
            return
        raise NotImplementedError(
            f"{type(self._robot).__name__} does not support velocity streaming."
        )

    def _stop_underlying(self) -> None:
        for name in ("stop_motion", "stop", "servo_stop"):
            fn = getattr(self._robot, name, None)
            if callable(fn):
                try:
                    fn()
                    return
                except Exception:
                    continue

    # Velocity entry point

    def send_velocity(
        self, velocity, acceleration: float = 0.5, time_duration: float = 0.1
    ):
        """
        Filter velocity through CBF-QP then forward to robot.
        """
        u_des = np.array(velocity, dtype=np.float64).ravel()[:6]
        if u_des.shape[0] < 6:
            u_des = np.pad(u_des, (0, 6 - u_des.shape[0]))

        if not _OSQP_AVAILABLE:
            self._forward_velocity(
                np.clip(u_des, self._u_min, self._u_max), acceleration, time_duration
            )
            return

        # Pure rotation skips QP
        if np.linalg.norm(u_des[:3]) < 1e-4:
            self._forward_velocity(
                np.clip(u_des, self._u_min, self._u_max), acceleration, time_duration
            )
            return

        p_tcp = self._read_tcp_position()
        joints = self._read_joints()
        obstacles = self._obstacle_map.get()
        f_tcp = self._read_force() if self._enable_force else None
        if f_tcp is not None and self._f_baseline is None:
            self._update_force_baseline(f_tcp)

        A_cbf, b_cbf, obstacle_active = self._barriers.evaluate(
            p_tcp,
            joints,
            obstacles,
            f_tcp,
            self._f_baseline,
            dt=time_duration,
            enable_force=self._enable_force,
        )

        u_safe = self._build_and_solve_qp(u_des, A_cbf, b_cbf)
        if u_safe is None:
            logger.warning("CBF-QP infeasible, emergency stop")
            try:
                self._stop_underlying()
            except Exception:
                pass
            return

        if obstacle_active and float(np.linalg.norm(u_safe - u_des)) > 1e-3:
            self.cbf_deviated = True
            logger.info(
                "CBF obstacle deviation: %.4f m/s",
                float(np.linalg.norm(u_safe - u_des)),
            )
        self._forward_velocity(u_safe, acceleration, time_duration)

    # URScript interception (UR-only)

    _RE_MOVEJ = re.compile(r"movej\s*\(\s*p\[([^\]]+)\]", re.IGNORECASE)
    _RE_MOVEL = re.compile(r"movel\s*\(\s*p\[([^\]]+)\]", re.IGNORECASE)
    _RE_MOVEJ_JOINTS = re.compile(r"movej\s*\(\s*\[([^\]]+)\]", re.IGNORECASE)

    def _send_script(self, script: str) -> bool:
        if not self._supports_urscript():
            raise AttributeError(
                f"{type(self._robot).__name__} does not support URScript."
            )
        # EVERY row is checked, not just the first. A blended program is one
        # script holding N movej rows; .search() validated row 0 and let rows
        # 1..N-1 through unchecked, so with blending on by default most of each
        # move bypassed the singularity/workspace filter entirely.
        #
        # Checked in wire order, and the first rejection refuses the WHOLE
        # script: a blended path is not separable, so partially approving it
        # would run the arm through the rows we rejected.
        for regex, handler in (
            (self._RE_MOVEL, self._check_movel_row),
            (self._RE_MOVEJ, self._check_movej_cartesian_row),
            (self._RE_MOVEJ_JOINTS, self._check_movej_joints_row),
        ):
            for m in regex.finditer(script):
                if not handler(m.group(1)):
                    logger.warning(
                        "[CBF] refusing script: a %s row failed the check "
                        "(%d row(s) inspected)",
                        handler.__name__,
                        len(regex.findall(script)),
                    )
                    return False
        return self._robot._send_script(script)

    def _parse_pose_values(self, csv: str) -> Optional[np.ndarray]:
        try:
            return np.array([float(x.strip()) for x in csv.split(",")])
        except (ValueError, TypeError):
            return None

    # Check-only row predicates used by the multi-row scan in _send_script.
    # They must NOT clamp or rewrite: a blended program's rows are one motion,
    # so silently moving one waypoint changes the path the others were planned
    # against. Unparseable rows pass (the legacy single-row path did the same).

    def _check_pose_row(self, csv: str) -> bool:
        pose = self._parse_pose_values(csv)
        if pose is None or len(pose) < 3:
            return True
        safe, violations = self._barriers.check_pose_safe(pose[:3])
        if not safe:
            logger.warning("[CBF] pose row unsafe: %s", violations)
        return bool(safe)

    def _check_movel_row(self, csv: str) -> bool:
        return self._check_pose_row(csv)

    def _check_movej_cartesian_row(self, csv: str) -> bool:
        return self._check_pose_row(csv)

    def _check_movej_joints_row(self, csv: str) -> bool:
        q = self._parse_pose_values(csv)
        if q is None or len(q) < 6:
            return True
        # A joint row carries no TCP pose, so the workspace box cannot be
        # applied here without forward kinematics we do not have. The elbow
        # singularity barrier IS evaluable from q alone, and it is the only
        # test the previous single-row handler enforced -- so enforcing it on
        # every row is strict parity plus the rows that used to be skipped.
        #
        # Do NOT reintroduce a "no FK available -> return True" branch: that
        # is what silently turned this predicate into a no-op and left
        # movej([q1..q6]) completely unvalidated.
        if self._cfg.eps_singularity <= 0:
            return True
        q2 = float(q[2])
        h = (q2**2) * ((q2 - np.pi) ** 2) - self._cfg.eps_singularity**2
        if h < 0:
            logger.warning(
                "[CBF] movej joint row near singularity: q2=%.3f h=%.4f", q2, h
            )
            return False
        return True

    # Workspace clamping

    def _clamp_to_workspace(self, p: np.ndarray) -> np.ndarray:
        clamped = np.clip(p, self._cfg.ws_min, self._cfg.ws_max)
        r = np.linalg.norm(clamped)
        if r > self._cfg.reach_max:
            clamped = clamped * (self._cfg.reach_max / r) * 0.98
        return clamped

    def _clamp_pose_to_workspace(self, pose) -> np.ndarray:
        p = np.asarray(pose, dtype=np.float64).ravel()
        if p.shape[0] < 6:
            p = np.pad(p, (0, 6 - p.shape[0]))
        else:
            p = p[:6]
        out = p.copy()
        out[:3] = self._clamp_to_workspace(p[:3])
        return out

    # Pose-motion entry points

    def move_linear(self, pose, *args, **kwargs):
        clamped = self._clamp_pose_to_workspace(pose)
        if not np.allclose(clamped[:3], np.asarray(pose, dtype=np.float64).ravel()[:3]):
            logger.info(
                "SafeRobot: clamped move_linear xyz %s -> %s",
                np.asarray(pose).ravel()[:3].tolist(),
                clamped[:3].tolist(),
            )
        return self._robot.move_linear(clamped.tolist(), *args, **kwargs)

    def move_linear_relative(self, delta, *args, **kwargs):
        d = np.asarray(delta, dtype=np.float64).ravel()
        if d.shape[0] < 6:
            d = np.pad(d, (0, 6 - d.shape[0]))
        else:
            d = d[:6]
        p_current = self._read_tcp_position()
        p_target_clamped = self._clamp_to_workspace(p_current + d[:3])
        d_adjusted = d.copy()
        d_adjusted[:3] = p_target_clamped - p_current
        return self._robot.move_linear_relative(d_adjusted.tolist(), *args, **kwargs)

    def move_to_joint_config(self, q, *args, **kwargs):
        return self._robot.move_to_joint_config(q, *args, **kwargs)

    def move_home(self, *args, **kwargs):
        if hasattr(self._robot, "move_home"):
            return self._robot.move_home(*args, **kwargs)
        if hasattr(self._robot, "go_home"):
            return self._robot.go_home(*args, **kwargs)
        raise AttributeError("No move_home or go_home on underlying driver.")

    def open_gripper(self, *args, **kwargs):
        return self._robot.open_gripper(*args, **kwargs)

    def close_gripper(self, *args, **kwargs):
        return self._robot.close_gripper(*args, **kwargs)

    def set_gripper_position(self, position, *args, **kwargs):
        if hasattr(self._robot, "set_gripper_position"):
            return self._robot.set_gripper_position(position, *args, **kwargs)
        return (
            self._robot.close_gripper()
            if position >= 0.5
            else self._robot.open_gripper()
        )

    def get_observation(self, *args, **kwargs):
        if hasattr(self._robot, "get_observation"):
            return self._robot.get_observation(*args, **kwargs)
        obs: Dict[str, Any] = {}
        for shim_name, key in (
            ("get_joint_positions", "joint_positions"),
            ("get_joint_velocities", "joint_velocities"),
            ("get_tcp_pose", "tcp_pose"),
            ("get_tcp_force", "tcp_force"),
            ("get_gripper_position", "gripper_position"),
        ):
            fn = getattr(self._robot, shim_name, None)
            if callable(fn):
                try:
                    obs[key] = fn()
                except Exception:
                    pass
        obs.setdefault("gripper_position", 0.0)
        return obs

    def get_tcp_pose(self, *args, **kwargs):
        return self._robot.get_tcp_pose(*args, **kwargs)

    def stop_motion(self, *args, **kwargs):
        if hasattr(self._robot, "stop_motion"):
            return self._robot.stop_motion(*args, **kwargs)
        if hasattr(self._robot, "stop"):
            return self._robot.stop(*args, **kwargs)
        raise AttributeError("No stop_motion or stop on underlying driver.")

    def __getattr__(self, name: str):
        return getattr(self._robot, name)

    def shutdown(self):
        if self._depth_detector is not None:
            self._depth_detector.stop()
        self._obstacle_map.clear()
        logger.info("SafeRobot shutdown complete")

    def __repr__(self):
        return (
            f"SafeRobot(robot={self._robot.__class__.__name__}, "
            f"osqp={_OSQP_AVAILABLE}, "
            f"obstacles={self._depth_detector is not None}, "
            f"force={self._enable_force})"
        )

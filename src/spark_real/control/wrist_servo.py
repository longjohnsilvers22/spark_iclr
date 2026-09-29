"""
Image-based visual servoing (IBVS) for wrist-camera grasp alignment.

RGB only -- no depth, no 3D reconstruction. The wrist D435i and the
Robotiq fingers are both bolted to the flange, so the fingertips land on a
FIXED pixel for a given aperture. Alignment is therefore an image-space
problem: drive the object's mask centroid onto that fingertip pixel.

Interaction model (pinhole, translation only, depth Z in camera frame):

    u_dot = (-fx * vx + (u - cx) * vz) / Z
    v_dot = (-fy * vy + (v - cy) * vz) / Z

Holding standoff (vz = 0) and asking for exponential error decay
e_dot = -kp * e gives the lateral camera velocity

    vx = kp * Z * e_u / fx        vy = kp * Z * e_v / fy

with e = (u - u*, v - v*). Dividing by the focal length makes the gain
scale-free (pixels -> radians). Z is unknown on this rig (wrist depth wedges
the USB stream), so a nominal ``standoff_m`` is folded into the gain. The
DIRECTION is exact for any Z; only the loop time constant scales, by
Z_assumed / Z_true. Under-estimating the standoff is the safe side (slower
loop); over-estimating it overshoots.

This module owns NO robot and NO camera. The caller reads pixels, calls
:meth:`WristIBVS.step` (or :meth:`WristIBVS.run` with its own callbacks), and
issues the motion itself.

Usage::

    servo = WristIBVS.from_calibration(wrist_cal)
    vel, converged, err_px = servo.step((u, v), (u_star, v_star))
    # ...or a bounded closed loop the caller drives:
    result = servo.run(observe=read_object_pixel,
                       command=robot.send_velocity,
                       target=(u_star, v_star))
    if not result.converged:
        ...  # result.outcome says why
"""

import logging
import time
from dataclasses import dataclass, field
from enum import Enum
from typing import Callable, List, Optional, Sequence, Tuple

import numpy as np

logger = logging.getLogger(__name__)

# Defaults tuned conservatively: this runs open-loop-in-depth next to a table.
DEFAULT_KP = 2.0  # error-decay rate, 1/s (kp * dt < 2 for stability)
DEFAULT_DEADBAND_PX = 6.0  # converged inside this pixel radius
DEFAULT_MAX_LINEAR_VEL = 0.05  # m/s clamp on the lateral command
DEFAULT_STANDOFF_M = 0.12  # nominal object distance; use a SMALL estimate
DEFAULT_RATE_HZ = 10.0  # loop rate for run()
DEFAULT_MAX_ITERS = 60
DEFAULT_TIMEOUT_S = 8.0
DEFAULT_MAX_LOST_FRAMES = 5


class ServoOutcome(str, Enum):
    """Terminal state of a servo loop. Only CONVERGED is a success."""

    CONVERGED = "converged"
    MAX_ITERATIONS = "max_iterations"
    TIMEOUT = "timeout"
    TARGET_LOST = "target_lost"
    ABORTED = "aborted"


@dataclass
class ServoResult:
    """Outcome of a bounded servo loop."""

    outcome: ServoOutcome
    iterations: int
    err_px: float
    elapsed_s: float
    last_velocity: np.ndarray = field(default_factory=lambda: np.zeros(6))
    err_history: List[float] = field(default_factory=list)

    @property
    def converged(self) -> bool:
        return self.outcome is ServoOutcome.CONVERGED


def pixel_error(current: Sequence[float], target: Sequence[float]) -> Tuple[float, float, float]:
    """(e_u, e_v, |e|) for a current and target pixel."""
    e_u = float(current[0]) - float(target[0])
    e_v = float(current[1]) - float(target[1])
    return e_u, e_v, float(np.hypot(e_u, e_v))


class WristIBVS:
    """
    Pure IBVS core: pixel error in, tool-frame Cartesian velocity out.

    Deterministic and side-effect free. ``step`` is stateless; the iteration
    and wall-clock budgets live in ``run`` so a caller can never spin forever.
    """

    def __init__(
        self,
        fx: float,
        fy: float,
        *,
        kp: float = DEFAULT_KP,
        deadband_px: float = DEFAULT_DEADBAND_PX,
        max_linear_vel: float = DEFAULT_MAX_LINEAR_VEL,
        standoff_m: float = DEFAULT_STANDOFF_M,
        rate_hz: float = DEFAULT_RATE_HZ,
        max_iters: int = DEFAULT_MAX_ITERS,
        timeout_s: float = DEFAULT_TIMEOUT_S,
        max_lost_frames: int = DEFAULT_MAX_LOST_FRAMES,
        rotation_cam_to_tool: Optional[np.ndarray] = None,
    ):
        """
        Args:
            fx, fy: wrist colour focal lengths in pixels. Distortion is
                ignored -- the D435i colour stream is factory-rectified.
            kp: image-error decay rate in 1/s. The loop is stable while
                kp * dt * (Z_true / standoff_m) < 2.
            deadband_px: converged when |error| is below this.
            max_linear_vel: clamp on the lateral velocity magnitude (m/s).
            standoff_m: nominal camera-to-object distance folded into the
                gain. Prefer an under-estimate.
            rate_hz / max_iters / timeout_s: bounds enforced by ``run``.
            max_lost_frames: consecutive missed observations before ``run``
                gives up with TARGET_LOST.
            rotation_cam_to_tool: 3x3 rotation mapping camera-frame vectors
                into the tool frame. Default identity, i.e. the optical axis
                is parallel to the tool approach axis and image +u/+v align
                with tool +x/+y. A rotated mount (e.g. the Franka wrist, ~90
                deg about Z) passes the rotation block of its hand-eye
                offset here.
        """
        if fx <= 0 or fy <= 0:
            raise ValueError(f"focal lengths must be positive, got {fx}, {fy}")
        if kp <= 0:
            raise ValueError(f"kp must be positive, got {kp}")
        if deadband_px < 0:
            raise ValueError(f"deadband_px must be >= 0, got {deadband_px}")
        if max_linear_vel <= 0:
            raise ValueError(f"max_linear_vel must be positive, got {max_linear_vel}")
        if standoff_m <= 0:
            raise ValueError(f"standoff_m must be positive, got {standoff_m}")

        self.fx = float(fx)
        self.fy = float(fy)
        self.kp = float(kp)
        self.deadband_px = float(deadband_px)
        self.max_linear_vel = float(max_linear_vel)
        self.standoff_m = float(standoff_m)
        self.rate_hz = float(rate_hz)
        self.max_iters = int(max_iters)
        self.timeout_s = float(timeout_s)
        self.max_lost_frames = int(max_lost_frames)

        if rotation_cam_to_tool is None:
            self.rotation_cam_to_tool = np.eye(3)
        else:
            R = np.asarray(rotation_cam_to_tool, dtype=float)
            if R.shape != (3, 3):
                raise ValueError(f"rotation_cam_to_tool must be 3x3, got {R.shape}")
            self.rotation_cam_to_tool = R

    @classmethod
    def from_calibration(cls, calibration, **kwargs) -> "WristIBVS":
        """Build from a CameraCalibration (reads fx/fy; never hardcoded)."""
        return cls(fx=calibration.fx, fy=calibration.fy, **kwargs)

    @property
    def dt(self) -> float:
        return 1.0 / self.rate_hz

    def step(
        self,
        current: Sequence[float],
        target: Sequence[float],
    ) -> Tuple[np.ndarray, bool, float]:
        """
        One IBVS update.

        Args:
            current: observed object pixel (u, v).
            target: fingertip-midpoint pixel (u*, v*).

        Returns:
            (velocity_6d, converged, err_px). velocity_6d is
            [vx, vy, vz, wx, wy, wz] in the TOOL frame; vz and the angular
            terms are always zero (no depth, no rotation authority here).
            Inside the deadband the velocity is exactly zero.
        """
        e_u, e_v, err_px = pixel_error(current, target)

        if err_px <= self.deadband_px:
            return np.zeros(6), True, err_px

        # Focal-length normalisation makes kp scale-free; standoff turns the
        # angular rate into a metric lateral rate.
        gain = self.kp * self.standoff_m
        v_cam = np.array([gain * e_u / self.fx, gain * e_v / self.fy, 0.0])

        speed = float(np.linalg.norm(v_cam))
        if speed > self.max_linear_vel:
            v_cam *= self.max_linear_vel / speed

        v_tool = self.rotation_cam_to_tool @ v_cam
        velocity = np.zeros(6)
        velocity[:3] = v_tool
        return velocity, False, err_px

    def run(
        self,
        observe: Callable[[], Optional[Sequence[float]]],
        command: Callable[[np.ndarray], None],
        target,
        *,
        sleep: Callable[[float], None] = time.sleep,
        now: Callable[[], float] = time.monotonic,
        should_abort: Optional[Callable[[], bool]] = None,
        on_step: Optional[Callable[[int, float, np.ndarray], None]] = None,
    ) -> ServoResult:
        """
        Bounded closed loop. Always terminates; always stops the motion.

        Args:
            observe: returns the current object pixel, or None if the object
                was not found this frame.
            command: receives each 6-vector tool velocity. A zero velocity is
                always sent last, whatever the outcome.
            target: a fixed (u*, v*) or a callable returning one (so a tag
                detector can re-resolve the fingertip pixel each frame).
            sleep / now: injectable clocks, for deterministic tests.
            should_abort: polled each iteration; True ends with ABORTED.
            on_step: optional (iteration, err_px, velocity) observer.
        """
        target_fn = target if callable(target) else (lambda: target)
        t0 = now()
        err_px = float("inf")
        history: List[float] = []
        velocity = np.zeros(6)
        lost = 0
        i = 0
        outcome = ServoOutcome.MAX_ITERATIONS

        while i < self.max_iters:
            if should_abort is not None and should_abort():
                outcome = ServoOutcome.ABORTED
                break
            if now() - t0 >= self.timeout_s:
                outcome = ServoOutcome.TIMEOUT
                break

            observation = observe()
            if observation is None:
                lost += 1
                if lost >= self.max_lost_frames:
                    outcome = ServoOutcome.TARGET_LOST
                    break
                command(np.zeros(6))
                sleep(self.dt)
                i += 1
                continue
            lost = 0

            velocity, converged, err_px = self.step(observation, target_fn())
            history.append(err_px)
            if on_step is not None:
                on_step(i, err_px, velocity)
            if converged:
                outcome = ServoOutcome.CONVERGED
                break

            command(velocity)
            sleep(self.dt)
            i += 1

        command(np.zeros(6))
        result = ServoResult(
            outcome=outcome,
            iterations=i,
            err_px=err_px,
            elapsed_s=now() - t0,
            last_velocity=velocity,
            err_history=history,
        )
        if not result.converged:
            logger.warning(
                "wrist IBVS did not converge: %s after %d iters, err=%.1f px",
                outcome.value,
                i,
                err_px,
            )
        return result

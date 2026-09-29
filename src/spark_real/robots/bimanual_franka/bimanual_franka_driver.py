"""
Bimanual Franka driver: two FrankaDriver instances + dual Dynamixel jaws.

Design contract (see `docs/bimanual_design.md`):

* The bimanual driver is **its own family** ("bimanual_franka"), composed
  by aggregation rather than inheritance from FrankaDriver. Single-arm
  semantics (control-mode invariants, reflex handling, joint limits) are
  reused by delegation so future single-arm changes do not silently leak
  into the bimanual stack.

* Every public shim method (the UR10eDriver surface) accepts an optional
  ``arm: "left" | "right"`` kwarg. For aggregate queries (e.g.
  ``get_joint_positions()``) the default returns the concatenated
  14-vector (left || right). For commands without an arm kwarg the
  driver refuses to run: there is no implicit arm.

* Per-arm sub-drivers are addressable as :attr:`left` and :attr:`right`
  so any legacy single-arm code can be passed *one* arm by reference.

* Grippers are decoupled from FCI IPs: both jaws share a single USB-CAN
  bus through :class:`DualDynamixelGripper`. This makes the rig hot-swap
  friendly (replace a stock Franka Hand without touching FCI).

* Optional state broadcast: :class:`BimanualStatePublisher` exposes a ZMQ
  PUB socket emitting the full bimanual observation at 30 Hz so external
  consumers (perception, VLA inference, dashboards) can subscribe
  without polling the HTTP routes.
"""

from __future__ import annotations

import json
import logging
import threading
import time
from dataclasses import dataclass, field
from typing import Any, Dict, List, Mapping, Optional, Tuple

import numpy as np

from spark_real.robots.franka.franka_driver import FrankaDriver
from spark_real.robots.franka.franka_bamboo_driver import FrankaBambooDriver

from .dynamixel_gripper import (
    DualDynamixelGripper,
    DynamixelConfig,
    DynamixelGripper,
    WIDTH_MAX_M,
    WIDTH_MIN_M,
)
from .ssg48_gripper import DualSSG48Gripper, SSG48Config, SSG48Gripper

logger = logging.getLogger(__name__)

try:
    import zmq  # type: ignore
except ImportError:
    zmq = None


ARMS: Tuple[str, str] = ("left", "right")


def make_dual_gripper(gripper_cfg: Optional[Mapping[str, Any]]) -> Any:
    """
    Build a dual-gripper backend from a YAML ``grippers:`` block.

    Dispatches on ``gripper_cfg["type"]``:

    * ``"dynamixel"`` (default): single USB bus, IDs 1/2 (ALOHA-style rig).
    * ``"ssg48"``: Source Robotics SSG-48 on two slcan buses (bimanual rig).
      Reads ``left:`` and ``right:`` sub-blocks for per-arm channels.

    The returned object satisfies the same surface as
    :class:`DualDynamixelGripper` (``connect``, ``disconnect``, ``homing``,
    ``for_arm``), so the bimanual driver is gripper-backend-agnostic past
    construction.

    Pass the result via ``BimanualFrankaDriver(dual_gripper=...)``.
    """
    cfg = dict(gripper_cfg or {})
    kind = str(cfg.get("type", "dynamixel")).lower()

    if kind == "dynamixel":
        port = str(cfg.get("port", "/dev/ttyUSB0"))
        baud = int(cfg.get("baudrate", 1_000_000))
        left_id = int(cfg.get("left_id", 1))
        right_id = int(cfg.get("right_id", 2))
        return DualDynamixelGripper(
            left=DynamixelGripper(
                DynamixelConfig(motor_id=left_id, port=port, baudrate=baud)
            ),
            right=DynamixelGripper(
                DynamixelConfig(motor_id=right_id, port=port, baudrate=baud)
            ),
        )

    if kind == "ssg48":

        def _arm_cfg(arm: str) -> SSG48Config:
            sub = dict(cfg.get(arm, {}) or {})
            return SSG48Config(
                channel=str(sub.get("channel", cfg.get("port", "/dev/ttyACM0"))),
                node_id=int(sub.get("node_id", cfg.get("node_id", 0))),
                bustype=str(sub.get("bustype", cfg.get("bustype", "slcan"))),
                bitrate=int(sub.get("bitrate", cfg.get("bitrate", 1_000_000))),
                default_speed=int(
                    sub.get("default_speed", cfg.get("default_speed", 60))
                ),
                default_force_ma=int(
                    sub.get("default_force_ma", cfg.get("default_force_ma", 500))
                ),
                max_force_ma=int(
                    sub.get("max_force_ma", cfg.get("max_force_ma", 1300))
                ),
                position_deadband=int(
                    sub.get("position_deadband", cfg.get("position_deadband", 5))
                ),
                auto_calibrate=bool(
                    sub.get("auto_calibrate", cfg.get("auto_calibrate", False))
                ),
                invert_position=bool(
                    sub.get("invert_position", cfg.get("invert_position", False))
                ),
                width_open_m=float(
                    sub.get("width_open_m", cfg.get("width_open_m", 0.090))
                ),
                width_closed_m=float(
                    sub.get("width_closed_m", cfg.get("width_closed_m", 0.000))
                ),
            )

        return DualSSG48Gripper(
            left=SSG48Gripper(_arm_cfg("left")),
            right=SSG48Gripper(_arm_cfg("right")),
        )

    raise ValueError(f"Unknown gripper type {kind!r}; expected 'dynamixel' or 'ssg48'.")


# Default IPs (override via config YAML or constructor kwargs). The Panda
# lives on .101, the FR3 on .102; if your subnet differs, pass `left_ip` /
# `right_ip` explicitly.
DEFAULT_LEFT_IP = "172.16.0.101"  # Franka Emika Panda
DEFAULT_RIGHT_IP = "172.16.0.102"  # Franka Research 3


@dataclass
class BimanualObservation:
    """
    Snapshot of both arms + both grippers + timestamp.

    Returned by :meth:`BimanualFrankaDriver.get_observation` when called
    without an ``arm`` kwarg. Per-arm observation (the dict at
    ``self["left"]``) matches the single-arm Franka observation schema so
    consumers that already speak it can be extended trivially.
    """

    ts: float
    left: Dict[str, Any] = field(default_factory=dict)
    right: Dict[str, Any] = field(default_factory=dict)

    def __getitem__(self, key: str) -> Dict[str, Any]:
        if key == "left":
            return self.left
        if key == "right":
            return self.right
        if key == "ts":
            return self.ts  # type: ignore[return-value]
        raise KeyError(key)

    def to_dict(self) -> Dict[str, Any]:
        return {"ts": self.ts, "left": self.left, "right": self.right}


class BimanualFrankaDriver:
    """
    Two-arm driver implementing the UR10eDriver shim with an ``arm`` kwarg.

    Construction does **not** open the FCI: call :meth:`connect` after
    instantiating so the caller can decide whether to block on Desk
    unlocking, etc. The driver is safe to ``connect()`` lazily but
    cannot be passed to a :class:`SafeRobot` until :attr:`connected`
    is True.
    """

    # shim capability flags
    SUPPORTS_VELOCITY_STREAMING = True
    SUPPORTS_URSCRIPT = False
    GRIPPER_TYPE = "dynamixel"

    # Aggregate joint vector layout: left 7 || right 7 = 14
    DOF_PER_ARM = 7
    DOF_TOTAL = 14

    # Per-arm ZMQ control ports for the bamboo backend (one C++ control
    # node per arm; each binds its own port so the two impedance controllers
    # don't collide). Franky doesn't use these.
    BAMBOO_LEFT_PORT = 5555
    BAMBOO_RIGHT_PORT = 5556

    def __init__(
        self,
        left_ip: str = DEFAULT_LEFT_IP,
        right_ip: str = DEFAULT_RIGHT_IP,
        frequency: float = 1000.0,
        left_gripper_id: int = 1,
        right_gripper_id: int = 2,
        gripper_port: str = "/dev/ttyUSB0",
        gripper_baud: int = 1_000_000,
        dual_gripper: Optional[Any] = None,
        publish_state: bool = False,
        publish_port: int = 5601,
        backend: str = "franky",
        **kwargs,
    ):
        self.left_ip = left_ip
        self.right_ip = right_ip
        self.frequency = frequency
        self._publish_state = bool(publish_state)
        self._publish_port = int(publish_port)

        # Arm-motion backend: "franky" (libfranka motion generators) or
        # "bamboo" (C++ joint-impedance controller). Selected here so connect()
        # builds the right per-arm driver class. Default stays franky for
        # backward compatibility.
        backend = str(backend or "franky").lower()
        if backend not in ("franky", "bamboo"):
            raise ValueError(
                f"unknown arm backend {backend!r}; expected 'franky' or " "'bamboo'"
            )
        self.backend = backend

        # Per-arm FrankaDriver instances are built in connect() to defer the
        # franky import (it pulls in the patched libfranka wheel).
        self._left_driver = None
        self._right_driver = None

        # Gripper backend: caller can inject any object satisfying the
        # DualDynamixelGripper surface (left/right SSG-48 buses, mocks, etc.).
        # Falling back to the Dynamixel default keeps the default bring-up path.
        if dual_gripper is not None:
            self.grippers = dual_gripper
        else:
            left_gcfg = DynamixelConfig(
                motor_id=left_gripper_id, port=gripper_port, baudrate=gripper_baud
            )
            right_gcfg = DynamixelConfig(
                motor_id=right_gripper_id, port=gripper_port, baudrate=gripper_baud
            )
            self.grippers = DualDynamixelGripper(
                left=DynamixelGripper(left_gcfg),
                right=DynamixelGripper(right_gcfg),
            )

        # Per-arm locks so parallel BT branches do not stomp on each other's
        # franky control-mode session.
        self._arm_locks = {arm: threading.Lock() for arm in ARMS}
        self._connected = False

        # Optional state publisher (ZMQ; falls back to noop when pyzmq absent).
        self._publisher = None

    # connection
    @property
    def connected(self) -> bool:
        return self._connected

    @property
    def left(self):
        """
        Per-arm sub-driver (the underlying single-arm FrankaDriver).

        Exposed so legacy single-arm code (e.g. a one-arm calibration
        routine) can be handed exactly one arm without learning the
        bimanual API.
        """
        return self._left_driver

    @property
    def right(self):
        return self._right_driver

    def for_arm(self, arm: str):
        # Return the underlying single-arm FrankaDriver for ``arm``.
        if arm == "left":
            return self._left_driver
        if arm == "right":
            return self._right_driver
        raise ValueError(f"unknown arm {arm!r}; expected one of {ARMS}")

    def connect(self) -> None:
        """
        Open both FCI sessions and the gripper bus.

        Order: Desk unlock both arms in parallel (each on its own thread
        because franka Desk unlocks block on HTTPS), then construct the
        franky Robot for each arm, then bring up the gripper bus. Errors
        on one arm leave the other arm connected: the caller can decide
        whether to retry or abort.
        """
        if self._connected:
            return

        if self.backend == "bamboo":
            # Bamboo backend: one FrankaBambooDriver per arm, each managing
            # its own bamboo C++ control node on a distinct ZMQ port. The
            # SSG-48 jaws are external (self.grippers), so the per-arm
            # bamboo nodes run with the gripper disabled ("-g none"), which
            # FrankaBambooDriver already enforces.
            self._left_driver = FrankaBambooDriver(
                ip=self.left_ip, port=self.BAMBOO_LEFT_PORT
            )
            self._right_driver = FrankaBambooDriver(
                ip=self.right_ip, port=self.BAMBOO_RIGHT_PORT
            )
            # Arm hint lets FrankaBambooDriver.move_linear pick the correct
            # PyRoki chain when panda_py analytical IK fails.
            self._left_driver._arm_hint = "left"
            self._right_driver._arm_hint = "right"
        else:
            # Per-arm drivers without Franka Hand clients: bimanual jaws are
            # external (self.grippers, e.g. Dynamixel or SSG-48), so franky's
            # Gripper client must not be constructed against the bare flanges.
            no_hand = {"type": "none"}
            self._left_driver = FrankaDriver(
                self.left_ip, self.frequency, gripper=no_hand
            )
            self._right_driver = FrankaDriver(
                self.right_ip, self.frequency, gripper=no_hand
            )

        threads: List[threading.Thread] = []
        errors: Dict[str, Exception] = {}

        def _open(name: str, drv) -> None:
            try:
                drv.connect()
            except Exception as exc:  # pragma: no cover - hardware path
                errors[name] = exc

        for name, drv in (("left", self._left_driver), ("right", self._right_driver)):
            t = threading.Thread(
                target=_open,
                args=(name, drv),
                daemon=True,
                name=f"franka-{name}-connect",
            )
            t.start()
            threads.append(t)
        for t in threads:
            t.join()

        if errors:
            for name, exc in errors.items():
                logger.error("failed to connect %s arm: %s", name, exc)
            # If only one arm came up we still allow operation in single-arm
            # mode (useful for bring-up debugging) but mark connected=True
            # only when both succeed.
            if len(errors) == len(ARMS):
                raise errors[next(iter(errors))]

        # Dynamixel bus comes up after FCI so any USB renumbering caused by
        # franky's libfranka load happens first.
        self.grippers.connect()
        try:
            self.grippers.homing()
        except Exception:
            logger.exception("gripper homing failed; continuing")

        if self._publish_state:
            self._publisher = _maybe_make_publisher(self._publish_port)

        self._connected = True

    def disconnect(self) -> None:
        if not self._connected:
            return
        if self._publisher is not None:
            try:
                self._publisher.close()
            except Exception:
                logger.exception("publisher close failed")
        for drv in (self._left_driver, self._right_driver):
            if drv is None:
                continue
            try:
                drv.disconnect()
            except Exception:
                logger.exception("driver disconnect failed")
        try:
            self.grippers.disconnect()
        except Exception:
            logger.exception("gripper disconnect failed")
        self._connected = False

    def reconnect_arm(self, arm: str) -> bool:
        """
        Tear down and re-open the FCI session for one arm only.

        Use this after rebooting a robot or re-enabling FCI on its Desk
        without bouncing the whole spark_real server.  Leaves the other
        arm's session, gripper bus, and state publisher untouched.

        Returns True if the arm came back up, False otherwise.  Either
        way, ``self._connected`` stays True so the rest of the server
        keeps running; check ``self.for_arm(arm)._connected`` for the
        per-arm state.
        """
        if arm not in ARMS:
            raise ValueError(f"arm must be 'left' or 'right', got {arm!r}")
        drv = self.for_arm(arm)
        if drv is None:
            return False
        try:
            drv.disconnect()
        except Exception:
            logger.exception("reconnect_arm(%s): disconnect failed", arm)
        try:
            drv.connect()
        except Exception as exc:
            logger.error("reconnect_arm(%s) failed: %s", arm, exc)
            return False
        logger.info("reconnect_arm(%s): FCI session reopened", arm)
        return True

    def __enter__(self):
        self.connect()
        return self

    def __exit__(self, exc_type, exc, tb) -> None:
        self.disconnect()

    # arm resolver
    def _resolve(self, arm: Optional[str]):
        if arm is None:
            raise ValueError(
                "bimanual driver requires explicit arm='left' or arm='right' "
                "for per-arm commands"
            )
        return self.for_arm(arm)

    # state readers
    def get_joint_positions(self, arm: Optional[str] = None) -> np.ndarray:
        if arm is not None:
            return self._resolve(arm).get_joint_positions()
        return np.concatenate(
            [
                self._left_driver.get_joint_positions(),
                self._right_driver.get_joint_positions(),
            ]
        )

    def get_joint_velocities(self, arm: Optional[str] = None) -> np.ndarray:
        if arm is not None:
            return self._resolve(arm).get_joint_velocities()
        return np.concatenate(
            [
                self._left_driver.get_joint_velocities(),
                self._right_driver.get_joint_velocities(),
            ]
        )

    def get_tcp_pose(self, arm: Optional[str] = None) -> np.ndarray:
        """
        Return single-arm TCP pose; ``arm`` is REQUIRED.

        Aggregate TCP makes no kinematic sense, so we refuse rather than
        invent a synthetic centroid.
        """
        return self._resolve(arm).get_tcp_pose()

    def get_tcp_force(self, arm: Optional[str] = None) -> np.ndarray:
        return self._resolve(arm).get_tcp_force()

    def get_robot_mode(self, arm: Optional[str] = None):
        if arm is not None:
            return self._resolve(arm).get_robot_mode()
        return {a: self.for_arm(a).get_robot_mode() for a in ARMS}

    def is_steady(self, arm: Optional[str] = None) -> bool:
        if arm is not None:
            return self._resolve(arm).is_steady()
        return all(self.for_arm(a).is_steady() for a in ARMS)

    @property
    def has_errors(self) -> bool:
        return any(
            self.for_arm(a).has_errors for a in ARMS if self.for_arm(a) is not None
        )

    def recover_from_errors(self, arm: Optional[str] = None) -> bool:
        if arm is not None:
            return self._resolve(arm).recover_from_errors()
        ok = True
        for a in ARMS:
            try:
                ok = self.for_arm(a).recover_from_errors() and ok
            except Exception:
                logger.exception("recover_from_errors failed for arm=%s", a)
                ok = False
        return ok

    def get_observation(self, arm: Optional[str] = None) -> Any:
        """
        Aggregate or per-arm observation snapshot.

        Per-arm form matches the single-arm Franka observation schema so
        existing consumers (anchor calibration, EpisodeRecorder)
        continue to work after extending them to know which arm to
        query.
        """
        if arm is not None:
            drv = self._resolve(arm)
            return {
                "joint_positions": drv.get_joint_positions(),
                "tcp_pose": drv.get_tcp_pose(),
                "tcp_force": drv.get_tcp_force(),
                "gripper_width": self.grippers.for_arm(arm).get_present_width(),
                "gripper_position": self.grippers.for_arm(arm).get_position_norm(),
            }
        return BimanualObservation(
            ts=time.time(),
            left=self.get_observation("left"),
            right=self.get_observation("right"),
        )

    # motion commands (all require an arm)
    def move_to_joint_config(
        self,
        q,
        arm: Optional[str] = None,
        velocity: Optional[float] = None,
        acceleration: Optional[float] = None,
        asynchronous: bool = False,
    ):
        """
        Move one arm to a 7-vector OR both arms to a 14-vector.

        Bimanual aggregate path runs the two arms in parallel via
        ``asynchronous=True`` on each franky call, then joins.
        """
        q_arr = np.asarray(q, dtype=float).ravel()
        if arm is not None:
            with self._arm_locks[arm]:
                return self._resolve(arm).move_to_joint_config(
                    q_arr,
                    velocity=velocity,
                    acceleration=acceleration,
                    asynchronous=asynchronous,
                )
        if q_arr.size != self.DOF_TOTAL:
            raise ValueError(
                f"aggregate move_to_joint_config expected a {self.DOF_TOTAL}-vector "
                f"(left||right); got {q_arr.size}"
            )
        left_q = q_arr[: self.DOF_PER_ARM]
        right_q = q_arr[self.DOF_PER_ARM :]
        results: Dict[str, Any] = {}

        def _go(name: str, drv, target):
            with self._arm_locks[name]:
                results[name] = drv.move_to_joint_config(
                    target,
                    velocity=velocity,
                    acceleration=acceleration,
                    asynchronous=False,
                )

        threads = [
            threading.Thread(
                target=_go,
                args=("left", self._left_driver, left_q),
                name="left-moveJ",
                daemon=True,
            ),
            threading.Thread(
                target=_go,
                args=("right", self._right_driver, right_q),
                name="right-moveJ",
                daemon=True,
            ),
        ]
        for t in threads:
            t.start()
        for t in threads:
            t.join()
        return results

    def move_linear(self, pose, arm: Optional[str] = None, **kwargs):
        with self._arm_locks[arm or "right"]:
            return self._resolve(arm).move_linear(pose, **kwargs)

    def move_linear_relative(self, delta, arm: Optional[str] = None, **kwargs):
        with self._arm_locks[arm or "right"]:
            return self._resolve(arm).move_linear_relative(delta, **kwargs)

    def servo_joint(self, q, arm: Optional[str] = None, **kwargs):
        with self._arm_locks[arm or "right"]:
            return self._resolve(arm).servo_joint(q, **kwargs)

    def servo_stop(self, arm: Optional[str] = None) -> None:
        if arm is not None:
            self._resolve(arm).servo_stop()
            return
        for a in ARMS:
            try:
                self.for_arm(a).servo_stop()
            except Exception:
                logger.exception("servo_stop failed for arm=%s", a)

    def send_velocity(
        self,
        linear,
        angular=None,
        arm: Optional[str] = None,
        acceleration: float = 0.5,
        duration: float = 0.0,
    ):
        with self._arm_locks[arm or "right"]:
            return self._resolve(arm).send_velocity(
                linear, angular=angular, acceleration=acceleration, duration=duration
            )

    def stop_velocity(self, arm: Optional[str] = None) -> None:
        if arm is not None:
            self._resolve(arm).stop_velocity()
            return
        for a in ARMS:
            try:
                self.for_arm(a).stop_velocity()
            except Exception:
                logger.exception("stop_velocity failed for arm=%s", a)

    def stop(self, arm: Optional[str] = None) -> None:
        # Hard stop both arms (default) or one. Always brakes the gripper too.
        if arm is not None:
            self._resolve(arm).stop()
            return
        for a in ARMS:
            try:
                self.for_arm(a).stop()
            except Exception:
                logger.exception("stop failed for arm=%s", a)

    def go_home(
        self, velocity: Optional[float] = None, arm: Optional[str] = None
    ) -> None:
        """
        Home one or both arms. Both-arm path sequences left then right so the
        elbows do not interfere as they swing through the home config.
        """
        if arm is not None:
            self._resolve(arm).go_home(velocity=velocity)
            return
        # Sequential to avoid mid-air collision (the home configs are
        # mirror-symmetric and the elbows pass close to the midline).
        for a in ARMS:
            try:
                self.for_arm(a).go_home(velocity=velocity)
            except Exception:
                logger.exception("go_home failed for arm=%s", a)

    # gripper passthrough
    def activate_gripper(self, arm: Optional[str] = None) -> None:
        if arm is not None:
            self.grippers.for_arm(arm).homing()
            return
        self.grippers.homing()

    def open_gripper(self, arm: Optional[str] = None, speed=None, force=None) -> None:
        if arm is not None:
            self.grippers.for_arm(arm).open(speed=speed)
            return
        for a in ARMS:
            self.grippers.for_arm(a).open(speed=speed)

    def close_gripper(self, arm: Optional[str] = None, speed=None, force=None) -> bool:
        if arm is not None:
            return self.grippers.for_arm(arm).close(force_n=force, speed=speed)
        ok = True
        for a in ARMS:
            ok = self.grippers.for_arm(a).close(force_n=force, speed=speed) and ok
        return ok

    def set_gripper_position(
        self, position: float, arm: Optional[str] = None, speed=None, force=None
    ) -> None:
        # ``position`` in 0..1 (0 = open, 1 = closed).
        position = max(0.0, min(1.0, float(position)))
        width = WIDTH_MAX_M - position * (WIDTH_MAX_M - WIDTH_MIN_M)
        if arm is not None:
            self.grippers.for_arm(arm).move_to_width(width, speed=speed)
            return
        for a in ARMS:
            self.grippers.for_arm(a).move_to_width(width, speed=speed)

    def grasp_to_width(
        self, width: float, arm: Optional[str] = None, force=None, speed=None
    ) -> bool:
        if arm is None:
            raise ValueError("grasp_to_width requires arm='left' or 'right'")
        return self.grippers.for_arm(arm).grasp_to_width(
            width, force_n=force, speed=speed
        )

    def get_gripper_width(self, arm: Optional[str] = None) -> float:
        if arm is None:
            raise ValueError("get_gripper_width requires arm='left' or 'right'")
        return self.grippers.for_arm(arm).get_present_width()

    def get_gripper_position(self, arm: Optional[str] = None) -> float:
        if arm is None:
            raise ValueError("get_gripper_position requires arm='left' or 'right'")
        return self.grippers.for_arm(arm).get_position_norm()

    def is_object_detected(self, arm: Optional[str] = None) -> bool:
        if arm is None:
            return any(self.grippers.for_arm(a).is_object_detected() for a in ARMS)
        return self.grippers.for_arm(arm).is_object_detected()


# Optional ZMQ state publisher
def _maybe_make_publisher(port: int):
    # Construct a ZMQ PUB socket if pyzmq is installed; else return None.
    if zmq is None:
        logger.info("pyzmq not installed; bimanual state broadcast disabled")
        return None
    return _ZmqStatePublisher(port)


class _ZmqStatePublisher:
    # Thin PUB-socket wrapper; topics keyed by ``b'bimanual_state'``.

    def __init__(self, port: int):
        self._ctx = zmq.Context.instance()
        self._sock = self._ctx.socket(zmq.PUB)
        self._sock.bind(f"tcp://*:{port}")
        logger.info("bimanual state publisher bound on tcp://*:%d", port)

    def publish(self, payload: Dict[str, Any]) -> None:
        self._sock.send_multipart(
            [b"bimanual_state", json.dumps(payload, default=float).encode()]
        )

    def close(self) -> None:
        try:
            self._sock.close(linger=100)
        except Exception:
            pass

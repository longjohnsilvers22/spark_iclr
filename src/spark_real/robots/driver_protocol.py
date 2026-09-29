"""
The driver surface the executor relies on, as a runtime-checkable Protocol.

RobotDriver lists ONLY the methods and attributes that control/executor_*.py,
control/success_verifier.py and skills/ call on the robot without a hasattr
guard, intersected with what UR10eDriver, FrankaDriverBase (and the bamboo
sibling FrankaBambooDriver) and BimanualFrankaDriver all define today.
make_robot_driver checks every freshly built driver against it, so a driver
missing one of these fails at construction instead of mid-episode.

Deliberately NOT listed, because at least one driver lacks them and every
call site already probes with hasattr: send_velocity / stop_velocity,
set_gripper_position, grasp_to_width, get_gripper_position, robot_family
(BimanualFrankaDriver has none), recover().
"""

from __future__ import annotations

from typing import Any, List, Protocol, runtime_checkable


@runtime_checkable
class RobotDriver(Protocol):
    GRIPPER_TYPE: str
    SUPPORTS_URSCRIPT: bool

    def get_joint_positions(self) -> Any: ...

    def get_tcp_pose(self) -> Any: ...

    def get_tcp_force(self) -> Any: ...

    def get_observation(self) -> Any: ...

    def move_to_joint_config(self, *args: Any, **kwargs: Any) -> Any: ...

    def move_linear(self, *args: Any, **kwargs: Any) -> Any: ...

    def open_gripper(self, *args: Any, **kwargs: Any) -> Any: ...

    def close_gripper(self, *args: Any, **kwargs: Any) -> Any: ...

    def get_gripper_width(self) -> Any: ...

    def is_object_detected(self) -> Any: ...


def missing_members(driver: Any) -> List[str]:
    """Names from RobotDriver that ``driver`` does not have (for the error)."""
    names = list(RobotDriver.__annotations__) + [
        n for n, v in vars(RobotDriver).items()
        if callable(v) and not n.startswith("_")
    ]
    return [n for n in names if not hasattr(driver, n)]

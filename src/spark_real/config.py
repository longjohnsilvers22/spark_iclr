"""
Config selection for a SPARK real deployment.

A single tyro flag (family) plus configs/<family>_default.yaml, and an
optional per-machine overlay under configs/machines/<machine>.yaml, resolve
to a RobotProfile. The profile maps to PipelineConfig kwargs; the richer
fields (home_config, gripper, workspace, urdf) are read from profile.raw.
"""

from __future__ import annotations

import copy
import functools
import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Annotated, Any, Dict, List, Literal, Optional

import tyro
import yaml

logger = logging.getLogger(__name__)

CONFIGS_DIR = Path(__file__).parent / "configs"

Family = Literal["ur10e", "franka", "g1", "bimanual_franka"]

# Per-family fallbacks. Used only when the family yaml does not supply the
# value.
_FALLBACK_MODELS: Dict[str, str] = {
    "ur10e": "UR10e",
    "franka": "Franka FR3",
    "g1": "Unitree G1",
    "bimanual_franka": "Bimanual Franka (Panda+FR3)",
}
_FALLBACK_IPS: Dict[str, str] = {
    "ur10e": "192.168.56.101",
    "franka": "172.16.0.2",
    "g1": "192.168.123.161",
    "bimanual_franka": "172.16.0.102",
}


@dataclass
class SparkConfig:
    # Operational flags for launching the server. The robot setup itself is
    # selected by family plus the optional per-machine machine overlay.
    family: Annotated[Family, tyro.conf.arg(aliases=["--robot"])]
    """Robot family. Required: there is no default, so a launch that names no
    family fails at the CLI instead of silently building one rig's pipeline
    (grasp orientation, camera roles, workspace) on another. --family is
    derived from the field name; --robot is an alias so historical launches
    (python -m spark_real.server --robot franka) keep working."""
    machine: Optional[str] = None
    host: str = "0.0.0.0"
    port: int = 8888
    no_robot: bool = False
    no_init: bool = False
    no_kinect: bool = False
    ip: Optional[str] = None
    auto_unlock: bool = False
    strict_verify: bool = False
    closed_loop_passes: Optional[int] = None
    """Closed-loop outer-pass cap for run_task. None = use the family YAML
    (max_task_passes) or the PipelineConfig default of 1 (single shot). >1
    re-perceives and re-plans over what remains after each pass."""


def _deep_merge(base: Dict[str, Any], overlay: Dict[str, Any]) -> Dict[str, Any]:
    """
    Recursively merge overlay onto base and return a new dict.
    """
    out = dict(base)
    for key, value in overlay.items():
        if isinstance(value, dict) and isinstance(out.get(key), dict):
            out[key] = _deep_merge(out[key], value)
        else:
            out[key] = value
    return out


def load_family_yaml(family: str, machine: Optional[str] = None) -> Dict[str, Any]:
    """
    Load configs/<family>_default.yaml and deep-merge an optional
    configs/machines/<machine>.yaml overlay. Missing files yield {}.
    """
    merged: Dict[str, Any] = {}
    base_path = CONFIGS_DIR / f"{family}_default.yaml"
    if base_path.exists():
        with open(base_path) as handle:
            merged = yaml.safe_load(handle) or {}
    if machine:
        overlay_path = CONFIGS_DIR / "machines" / f"{machine}.yaml"
        if overlay_path.exists():
            with open(overlay_path) as handle:
                overlay = yaml.safe_load(handle) or {}
            merged = _deep_merge(merged, overlay)
        else:
            # Overlays are gitignored per rig, so a git-only sync to a new
            # host has none. Say so instead of silently running the
            # packaged family defaults (placeholder IPs, serials, offsets).
            logger.warning(
                "--machine %s named but %s does not exist; running on "
                "%s_default.yaml alone",
                machine, overlay_path, family,
            )
    return merged



@functools.lru_cache(maxsize=None)
def _packaged_yaml(family: str) -> Dict[str, Any]:
    """configs/<family>_default.yaml, read once; {} when missing or unreadable."""
    try:
        return load_family_yaml(family)
    except Exception as exc:  # noqa: BLE001 - a config read must not kill a run
        logger.warning("could not read %s_default.yaml (%s)", family, exc)
        return {}


def family_block(profile: Any, family: str, name: Optional[str] = None) -> Dict[str, Any]:
    """
    One top-level block of the family config, or the whole mapping when name
    is None. Reads profile.raw when a RobotProfile with content is wired
    (server path: family default deep-merged with the machine overlay), else
    the packaged configs/<family>_default.yaml, cached. The single home of
    the "profile.raw else open the family YAML" fallback.
    """
    raw = profile.raw if getattr(profile, "raw", None) else _packaged_yaml(family)
    # copies: callers mutate blocks (setdefault, pop) and the packaged mapping is cached process-wide
    if name is None:
        return copy.deepcopy(raw)
    return copy.deepcopy(raw.get(name) or {})


@dataclass
class RobotProfile:
    """
    Resolved deployment config. raw holds the full merged yaml so later
    steps can read home_config, gripper, workspace, urdf, etc. directly.
    """

    family: str
    machine: Optional[str]
    raw: Dict[str, Any]
    no_robot: bool = False
    no_kinect: bool = False
    strict_verify: bool = False
    ip_override: Optional[str] = None
    closed_loop_passes: Optional[int] = None

    def resolved_model(self) -> str:
        model = _FALLBACK_MODELS.get(self.family, self.family)
        variant = (self.raw.get("robot") or {}).get("variant")
        if variant:
            model = str(variant)
        return model

    def resolved_ip(self) -> str:
        ip = _FALLBACK_IPS.get(self.family, "192.168.56.101")
        yaml_ip = (self.raw.get("robot") or {}).get("ip")
        if yaml_ip:
            ip = str(yaml_ip)
        if self.ip_override:
            ip = self.ip_override
        return ip

    def to_pipeline_kwargs(self) -> Dict[str, Any]:
        """
        The PipelineConfig kwargs for this profile. Keys not set here fall
        back to PipelineConfig dataclass defaults.
        """
        cfg = self.raw
        use_realsense = None
        if cfg.get("use_realsense") is not None:
            use_realsense = bool(cfg["use_realsense"])
        kwargs: Dict[str, Any] = dict(
            robot_ip="" if self.no_robot else self.resolved_ip(),
            use_kinect=not self.no_kinect,
            use_realsense=use_realsense,
            strict_placement_verify=self.strict_verify,
            robot_family=self.family,
            robot_model=self.resolved_model(),
        )
        if "kinect_use_hw_sync" in cfg:
            kwargs["kinect_use_hw_sync"] = bool(cfg["kinect_use_hw_sync"])
        depth_mode = cfg.get("kinect_depth_mode")
        if depth_mode:
            kwargs["kinect_depth_mode"] = str(depth_mode)
        resolution = cfg.get("kinect_resolution")
        if resolution:
            kwargs["kinect_resolution"] = str(resolution)
        fps = cfg.get("kinect_fps")
        if fps:
            kwargs["kinect_fps"] = int(fps)
        # Master Kinect serial: also names the sideview camera in
        # pipeline_init._init_kinects_from_early. Without forwarding it,
        # PipelineConfig's dataclass default (the FR3 host's serial) wins and
        # non-FR3 rigs name both Kinects birdview. Forward "" too so an explicit
        # empty value in YAML does not resurrect the FR3 default.
        if cfg.get("kinect_master_serial") is not None:
            kwargs["kinect_master_serial"] = str(cfg["kinect_master_serial"])
        if cfg.get("table_height") is not None:
            kwargs["table_height"] = float(cfg["table_height"])
        # Closed-loop execution: YAML supplies the default; the CLI flag
        # (--closed-loop-passes) overrides it when set.
        if cfg.get("max_task_passes") is not None:
            kwargs["max_task_passes"] = int(cfg["max_task_passes"])
        if cfg.get("closed_loop_settle_s") is not None:
            kwargs["closed_loop_settle_s"] = float(cfg["closed_loop_settle_s"])
        if self.closed_loop_passes is not None:
            kwargs["max_task_passes"] = max(1, int(self.closed_loop_passes))
        # Demonstration recording knobs (configs/<family>_default.yaml
        # `recording:`). Forwarded WHOLE as a dict: RecordingSettings is the
        # single place that knows the field names and their defaults, so
        # adding a knob there needs no edit here. Unknown keys are ignored by
        # RecordingSettings rather than crashing config load.
        recording = dict(cfg.get("recording") or {})
        if recording:
            kwargs["recording"] = recording

        # Behavior-tree cache (configs/<family>_default.yaml `bt:`). Governs
        # where the disk-backed BTLibrary lives, which packaged seeds prime it,
        # and whether a cache miss may reach for the LLM at all.
        bt = dict(cfg.get("bt") or {})
        if bt.get("library_dir") is not None:
            kwargs["bt_library_dir"] = str(bt["library_dir"])
        if bt.get("seed_dir") is not None:
            kwargs["bt_seed_dir"] = str(bt["seed_dir"])
        if bt.get("seed_on_start") is not None:
            kwargs["bt_seed_on_start"] = bool(bt["seed_on_start"])
        if bt.get("plan_mode") is not None:
            kwargs["bt_plan_mode"] = str(bt["plan_mode"])
        if bt.get("min_similarity") is not None:
            kwargs["bt_min_similarity"] = float(bt["min_similarity"])
        if bt.get("auto_promote_after") is not None:
            kwargs["bt_auto_promote_after"] = int(bt["auto_promote_after"])

        # Frozen per-task SAM3 prompt contracts (configs/tasks/*.yaml).
        # Top level, or under `perception:` for symmetry with the other
        # perception knobs. None => the packaged directory.
        perception = dict(cfg.get("perception") or {})
        task_prompts_dir = cfg.get("task_prompts_dir", perception.get("task_prompts_dir"))
        if task_prompts_dir is not None:
            kwargs["task_prompts_dir"] = str(task_prompts_dir)
        return kwargs

    def gripper(self) -> Dict[str, Any]:
        # Single-arm uses `gripper`; bimanual uses `grippers` (shared dual jaw).
        return dict(self.raw.get("gripper") or self.raw.get("grippers") or {})

    def cloth(self) -> Dict[str, Any]:
        # Fold tuning for this family (sleeve_lift_h, arc_peak, etc.).
        return dict(self.raw.get("cloth") or {})

    def cameras(self) -> List[Dict[str, Any]]:
        # Role-keyed camera entries (role, type, serial or device, offsets).
        cams = self.raw.get("cameras") or []
        return [dict(c) for c in cams if isinstance(c, dict)]

    def workspace(self) -> Dict[str, Any]:
        # Single-arm `workspace`; bimanual splits into per-arm boxes.
        if self.raw.get("workspace"):
            return dict(self.raw["workspace"])
        left = self.raw.get("workspace_left")
        right = self.raw.get("workspace_right")
        if left or right:
            return {"left": left, "right": right}
        return {}

    def control(self) -> Dict[str, Any]:
        # ScoreExecutor base-frame constants (grasp_orientation, workspace_min/
        # max, max_reach, table_z_floor). Distinct from workspace(): this is the
        # executor/SafeRobot box, workspace() is the CBF _family_safety_bounds box.
        return dict(self.raw.get("control") or {})

    def collision_behavior(self) -> Dict[str, Any]:
        # Per-joint and Cartesian collision thresholds for the driver.
        return dict(self.raw.get("collision_behavior") or {})

    def calibration(self) -> Dict[str, Any]:
        # Per-machine hand-eye and anchor file paths, when set in the overlay.
        return dict(self.raw.get("calibration") or {})

    def home_config(self) -> Any:
        # Single-arm returns a joint list; bimanual returns {left, right}.
        robot = self.raw.get("robot") or {}
        if robot.get("home_config") is not None:
            return list(robot["home_config"])
        left = robot.get("home_config_left")
        right = robot.get("home_config_right")
        if left is not None or right is not None:
            return {"left": left, "right": right}
        return None


def load_profile(cfg: SparkConfig) -> RobotProfile:
    """
    Build the resolved profile from the family yaml plus machine overlay.
    A yaml read failure falls back to an empty config, which yields the
    per-family fallback model and ip.
    """
    try:
        raw = load_family_yaml(cfg.family, cfg.machine)
    except Exception:
        raw = {}
    return RobotProfile(
        family=cfg.family,
        machine=cfg.machine,
        raw=raw,
        no_robot=cfg.no_robot,
        no_kinect=cfg.no_kinect,
        strict_verify=cfg.strict_verify,
        ip_override=cfg.ip,
        closed_loop_passes=cfg.closed_loop_passes,
    )

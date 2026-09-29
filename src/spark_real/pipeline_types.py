# Shared pipeline dataclasses, split out so the mixin modules can import
# them without a cycle back through pipeline.py.

from dataclasses import dataclass, field
from typing import Dict, List, Optional


@dataclass
class PipelineConfig:
    """
    Configuration for the real robot pipeline.
    """

    # robot_ip is set by server.py from --ip / configs/<family>_default.yaml.
    # The empty default is a sentinel meaning "no robot configured"; code
    # paths that try to connect should treat "" as a "skip connect" signal.
    robot_ip: str = ""

    # Robot identity surfaced to the frontend so the UI can swap banner +
    # control panel per family. server.py overrides via --robot.
    robot_family: str = "ur10e"  # one of: ur10e | franka | g1
    robot_model: str = "UR10e"  # human-readable model string

    use_kinect: bool = True
    use_realsense: Optional[bool] = None
    """
    None = auto (init iff a RealSense is enumerated); True/False forces.
    """
    # Color resolution. 720P, not 1080P: color is the ONLY isochronous
    # stream on the Kinect, so its alt-setting reserves a fixed slice of the
    # shared xHCI periodic schedule for the life of the stream (262-852
    # Mbit/s per camera at 1080P). Two Kinects plus the wrist RealSense sit
    # on one controller on the UR10e rig, and depth (bulk) only gets what
    # the isoc reservation leaves. 720P drops the reserved band roughly by
    # half at no geometric cost: the SDK reports per-resolution factory
    # intrinsics, so fx/fy/cx/cy follow automatically and hand-eye stays
    # valid. See docs/MULTICAM_CRASH_MECHANISM.md.
    kinect_resolution: str = "720P"
    kinect_master_serial: str = "000000000000"  # Side view (master)
    kinect_use_hw_sync: bool = False  # Set True only if sync cable connected
    # Depth modes (Azure Kinect DK):
    #   NFOV_UNBINNED:  75x65 FOV,  640x576, 30 fps, range 0.5-3.86 m
    #   NFOV_2X2BINNED: 75x65 FOV,  320x288, 30 fps, range 0.5-5.46 m
    #   WFOV_UNBINNED:  120x120,    1024x1024, 15 fps, range 0.25-2.21 m
    #   WFOV_2X2BINNED: 120x120,    512x512,  30 fps, range 0.25-2.88 m
    # WFOV_2X2BINNED is the right default for tabletop manipulation:
    # full 30 fps for the VLA loop, wide enough that one Kinect sees the
    # whole cell, and close-range support down to 25 cm. Override in the
    # per-family YAML's `kinect_depth_mode` field.
    kinect_depth_mode: str = "WFOV_2X2BINNED"
    # 15, not 30. Frame rate multiplies EVERYTHING that scales with the
    # crash surface at once: bulk depth+IR bytes on the shared controller,
    # URB submission rate through usbfs, host-side color decode, and
    # depth-engine GL dispatches on the single 4090. 15 fps is what the
    # franka rig already runs and what published multi-Kinect rigs use;
    # 30 fps on two cameras was never validated on one host controller.
    kinect_fps: int = 15
    realsense_serial: str = ""
    # Stream the wrist RealSense color-only (no depth) to halve SuperSpeed USB
    # bandwidth and avoid the xHCI event-ring desync that drops the stream.
    # Depth is unused while wrist_refine is off; set False to restore it.
    realsense_color_only: bool = True

    sam3_threshold: float = 0.03
    use_hardware_depth: bool = True

    gemini_model: str = "gemini-3.5-flash"
    planner_temperature: float = 0.3

    velocity: float = 0.25
    safe_height: float = 0.35
    # Recovery search floor (robot frame, open-gripper TCP at the work
    # surface). None means: use the executor's config-fed TABLE_Z_FLOOR
    # for the active family. Set per machine in the family YAML
    # (table_height) only when the rig has a raised surface like the
    # original UR10e cell's foam mat (-0.28).
    table_height: float | None = None

    output_dir: str = "output/real_runs"
    save_captures: bool = True

    # Demonstration recording (synchronized image + proprio + commanded
    # action, written in the SAME on-disk schema as the human teleop corpus;
    # see spark_real/recording/). The data root lives OUTSIDE the code tree so
    # recorded datasets never land in git.
    #
    # Carried as an opaque dict rather than a field per knob: every default
    # already lives in recording/settings.py:RecordingSettings, which is the
    # single source of truth. Set via the `recording:` block in
    # configs/<family>_default.yaml; an empty dict means "all defaults".
    recording: Dict = field(default_factory=dict)
    """`recording:` config block, resolved by RecordingSettings.from_config."""

    # Behavior-tree cache. Once a task has one verified run in the library,
    # every later run of it is served from disk with no LLM call. Set via the
    # `bt:` block in configs/<family>_default.yaml.
    bt_library_dir: Optional[str] = None
    """Library root. None => <output_dir>/../bt_library (NOT cwd-relative)."""
    bt_seed_dir: Optional[str] = None
    """Seed YAML root. None => the packaged configs/bt_seeds."""
    bt_frozen: bool = False
    """
    Freeze the BT library: serve NOTHING as few-shot context and record
    nothing back.

    The no-adaptation control: the paper's Problem Setup defines the
    no-adaptation protocol as seeing no trial of a task before that task is
    scored, so with bt_frozen set, trial N plans with exactly the context
    trial 1 had.

    Env override: SPARK_BT_FROZEN=1.
    """

    bt_seed_on_start: bool = True
    """
    Install the packaged seed trees into the library at startup.

    Turn this OFF when collecting a fresh demonstration corpus: a seeded tree
    is cache-eligible immediately, so the task would be served from disk
    instead of being planned, and the run you meant to record fresh never
    calls the planner at all.
    """
    bt_plan_mode: str = "auto"
    # The recovery ladder's terminal rung: after every LLM-free rung fails
    # WITH evidence, consult the planner once with the failure context.
    # Off: surface the failure and stop.
    llm_last_resort: bool = True
    """auto (cache first, LLM on miss) | cache_only (miss raises) | llm."""
    bt_min_similarity: float = 0.75
    """Jaccard floor for fuzzy retrieval. Below it is a miss, not a hit."""
    bt_auto_promote_after: int = 3
    """Successes (with zero failures) before an entry auto-promotes."""

    # Frozen per-task SAM3 prompt contracts (prompts, expected object counts,
    # deterministic instance ordering). None => the packaged configs/tasks.
    # Also overridable by $SPARK_TASK_PROMPTS.
    task_prompts_dir: Optional[str] = None
    """Directory of task prompt YAMLs; see perception/prompt_registry.py."""

    # Experimental: strict placement verification.
    # When True, replaces the binary mask-intersection check with a
    # containment ratio (>= 50% of the picked object's mask must lie
    # inside the place target's mask), allows the post-task retry to
    # re-detect the placed object's current location, and caps retries
    # at 1. Catches "tumbled out next to container" failures that the
    # legacy verify reports as success.
    # Off by default, flip via --strict-verify on the server CLI.
    strict_placement_verify: bool = False

    save_to_library: bool = True
    """
    If True, every successful task appends its (instruction, score) to bt_library/.
    """

    # Closed-loop task execution (generic, all families).
    # After a behavior tree runs, the pipeline re-perceives the scene and
    # checks whether the target objects actually reached their destination
    # (in a receptacle, swept into a dustpan, lifted off the source). Any
    # target that did NOT make it, plus any object the scene was perturbed
    # with mid-task (swapped utensil, added item, moved receptacle), drives
    # another plan-and-execute pass. Already-handled objects stay closed via
    # the destination filter and the executor's placed-label set, so the
    # re-plan never re-visits a node already taken care of.
    # max_task_passes == 1 is single shot.
    max_task_passes: int = 1
    """Outer closed-loop cap. 1 = single shot (legacy). >1 re-perceives and
    re-plans over what remains until the goal holds or the budget is spent."""
    closed_loop_settle_s: float = 0.8
    """Seconds to let the scene settle (arm home, objects still) before the
    between-pass re-perception so SAM3 sees the workspace, not motion."""
    closed_loop_clear_radius_m: float = 0.12
    """A target whose centroid is within this distance of a destination
    centroid is treated as handled (inside the receptacle / swept in). Used
    as a floor; the destination's own OBB minor radius widens it per object."""

    use_grasp_calibration: bool = True
    """
    If True, ScoreExecutor uses per-object learned grasp depth offsets.
    """

    # Annotation-rescue rung for the count-gate escalation ladder
    # (pipeline_perception.detect_for_task). When every prompt rung has
    # failed for a group, the configured provider POINTS at it and the
    # point seeds SAM3's click head. $SPARK_ANNOTATION_RESCUE overrides
    # ("0"/"1"/provider name), same pattern as $SPARK_ROBOINTER.
    annotation_rescue: bool = False
    """Enable the provider-point rescue rung. Default off."""
    annotation_provider: str = "er2"
    """Provider for the rescue rung: er2 | molmo | human. See
    spark_real/perception/annotations.py."""

    # Wrist-camera mounting transform (RealSense D435I to TCP). Robot-family
    # dependent because each arm's flange geometry differs. Set to None to
    # use the per-family default below; override here (or via config YAML)
    # after measuring the actual bracket. The 4x4 transform is built as
    # `eye(4)` with translation set from `wrist_tool_offset_xyz`; orientation
    # is identity by default (camera Z aligned with TCP Z). When you change
    # the bracket geometry, update this: see spark_real/calibration.
    wrist_tool_offset_xyz: Optional[List[float]] = None  # None = family default


@dataclass
class TaskResult:
    """
    Result from a single task execution.
    """

    instruction: str
    timestamp: str
    detections: List[Dict]
    plan: Dict
    execution_results: List[Dict]
    success: bool
    duration: float
    captures: Dict[str, str] = field(default_factory=dict)

    # Plan provenance. Declared (not just set as ad-hoc attributes) because
    # the recorder copies them verbatim into metadata["spark"], so an episode
    # on disk says exactly which tree drove it and where that tree came from.
    plan_source: Optional[str] = None
    """pin | exact | alias | jaccard | seed | llm -- see BTLibrary.resolve."""
    bt_hash: Optional[str] = None
    """Content hash of the executed behavior tree, or None if unlibraried."""
    label_resolutions: Dict[str, str] = field(default_factory=dict)
    """Cached-BT label -> the differently-named detection it re-bound to."""
    robointer: List[Dict] = field(default_factory=list)
    """Per-node RoboInter resolutions: what the planner proposed in image
    space, what it resolved to in the base frame, and -- when a proposal was
    refused -- how far from the detection it was. Empty for every run with the
    extension off (planning.robointer.enabled / $SPARK_ROBOINTER)."""

    # Task-success verdict (control/success_verifier.VerifyOutcome.to_dict()).
    # ``success`` above is exactly ``verify_status == "pass"``; this carries the
    # evidence -- predicates, per-camera votes, gates, depth source.
    verify: Optional[Dict] = None
    """Full VerifyOutcome dict, or None when no verdict was produced."""
    verify_status: str = "unverified"
    """pass | fail | unverified. `unverified` must not train and must not
    reinforce a cached tree; it is not a quieter kind of failure."""

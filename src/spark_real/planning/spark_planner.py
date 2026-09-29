"""
SPARK task planner - standalone version (no ROS2).

Generates SPARK scores (YAML-based DSL) from:
- Natural language instructions
- Camera observations (annotated with SAM3 keypoints)
- Scene context

Uses Gemini or OpenAI as the LLM backend.
"""

import copy
import json
import logging
import os
import re
from pathlib import Path

import yaml
from PIL import Image as _PILImage

from spark_real.control.executor_types import BUILTIN_PRIMITIVES
from spark_real.planning import bt_grammar
from spark_real.planning.robointer import sanitize_plan_annotations
from spark_real.planning.robointer_prompt import ROBOINTER_PROMPT_SECTION
from spark_real.skills import registry as skill_registry
from spark_real.utils.score_tree import is_num, label_known, walk_nodes

logger = logging.getLogger(__name__)

# Optional LLM backends; absent installs leave the name None and the
# client setup raises a clear error when that backend is selected.
try:
    from google import genai as _genai
    from google.genai import types as _genai_types
except ImportError:
    _genai = None
    _genai_types = None

try:
    from openai import OpenAI as _OpenAI
except ImportError:
    _OpenAI = None

# Hard deadline for every Gemini HTTP request (milliseconds; google-genai
# takes ms in http_options). A stalled connection must surface as a failed
# plan, never a hang.
GEMINI_TIMEOUT_MS = int(os.environ.get("SPARK_GEMINI_TIMEOUT_MS", "120000"))


def _make_genai_client(api_key: str):
    """google-genai Client with the request deadline installed."""
    return _genai.Client(
        api_key=api_key,
        http_options={"timeout": GEMINI_TIMEOUT_MS},
    )

class PlanSchemaError(ValueError):
    """
    Raised when a plan cannot be parsed, or when a caller asks for strict
    validation and an optional field is malformed.
    """


# ---------------------------------------------------------------------------
# Optional plan extensions. A plan that omits all of them parses and executes
# unchanged; the 43 cached trees in bt_library/ depend on that.
# ---------------------------------------------------------------------------

GRASP_STRATEGIES = ("auto", "topdown", "obb", "cgn", "se3")

# Node types that may carry grasp_strategy / grasp_yaw_deg. `grasp` wins on
# conflict (resolved downstream in control/grasp_strategy.py).
STRATEGY_NODE_TYPES = ("move_to_keypoint", "grasp")

# Extension param -> the value that means "not specified". Params equal to
# their default are dropped before hashing so a plan that spells out the
# default hashes identically to a cached tree that omits it.
EXTENSION_DEFAULTS = {"grasp_strategy": "auto", "grasp_yaw_deg": None}

# Score keys that carry no plan semantics and must not shift the BT hash.
# `verify` is a success predicate, not an action -- two plans that differ only
# in how success is judged are the same tree.
PLAN_NONSEMANTIC_KEYS = ("label_corrections", "verify")

_VERIFY_TOP_KEYS = frozenset({"all", "min_conf", "require_two_views"})

# Predicate vocabulary. Mirrors control/success_predicates.py, which is the
# runtime authority; this copy exists so the planner can reject a bad block at
# emit time (that module is numpy+stdlib only and cannot import from here).
_VERIFY_SPEC = {
    "inside": {
        "required": ("obj", "container"),
        "optional": {
            "xy_margin_m": "num",
            "z_tol_m": "num",
            "occlusion_ok": "bool",
        },
    },
    "on": {
        "required": ("obj", "surface"),
        "optional": {"xy_margin_m": "num", "z_max_m": "num"},
    },
    "stacked": {
        "required": ("obj", "base"),
        "optional": {"xy_tol_m": "num", "dz_min_m": "num", "dz_max_m": "num"},
    },
    "near": {"required": ("obj", "target", "max_dist_m"), "optional": {}},
    "removed_from": {"required": ("obj", "container"), "optional": {}},
    "held": {"required": ("obj", "value"), "optional": {}},
    "absent": {"required": ("obj",), "optional": {}},
}

# Predicate fields whose value is a detection label.
_VERIFY_LABEL_FIELDS = ("obj", "container", "surface", "base", "target")

# Sanity ranges. Wide on purpose: these catch unit errors and hallucinated
# values, not borderline choices.
_RANGES = {
    "min_conf": (0.0, 1.0),
    "grasp_yaw_deg": (-360.0, 360.0),
    "xy_margin_m": (-0.5, 0.5),
    "z_tol_m": (0.0, 1.0),
    "z_max_m": (0.0, 1.0),
    "xy_tol_m": (0.0, 1.0),
    "dz_min_m": (0.0, 1.0),
    "dz_max_m": (0.0, 1.0),
    "max_dist_m": (0.0, 5.0),
}

def _range_issue(name: str, value, path: str) -> str:
    lo, hi = _RANGES.get(name, (None, None))
    if lo is None or lo <= float(value) <= hi:
        return ""
    return f"{path}: {name}={value} out of range [{lo}, {hi}]"


def validate_grasp_params(node_type: str, params: dict, path: str) -> list:
    """
    Validate the optional grasp-strategy fields on one node.

    Returns a list of human-readable issues; empty means the node is fine
    (including the common case where neither field is present).
    """
    issues = []
    if not isinstance(params, dict):
        return issues

    for key in EXTENSION_DEFAULTS:
        if key not in params:
            continue
        if node_type not in STRATEGY_NODE_TYPES:
            issues.append(
                f"{path}: '{key}' is only meaningful on "
                f"{'/'.join(STRATEGY_NODE_TYPES)}, not '{node_type}'"
            )

    if "grasp_strategy" in params:
        value = params["grasp_strategy"]
        if not isinstance(value, str) or value not in GRASP_STRATEGIES:
            issues.append(
                f"{path}: unknown grasp_strategy {value!r}; "
                f"expected one of {list(GRASP_STRATEGIES)}"
            )

    if "grasp_yaw_deg" in params:
        value = params["grasp_yaw_deg"]
        if value is not None:
            if not is_num(value):
                issues.append(f"{path}: grasp_yaw_deg must be a number, got {value!r}")
            else:
                bad = _range_issue("grasp_yaw_deg", value, path)
                if bad:
                    issues.append(bad)
        strategy = params.get("grasp_strategy", EXTENSION_DEFAULTS["grasp_strategy"])
        if value is not None and strategy != "obb":
            issues.append(
                f"{path}: grasp_yaw_deg is only read when grasp_strategy is "
                f"'obb' (got {strategy!r})"
            )
    return issues


def validate_verify_block(block, keypoint_labels=None) -> list:
    """
    Validate a top-level `verify:` block against the predicate vocabulary.

    Any issue invalidates the WHOLE block: a partially
    honoured predicate is worse than none, because it silently verifies
    something other than the goal.
    """
    issues = []
    if not isinstance(block, dict):
        return [f"verify: must be a mapping, got {type(block).__name__}"]

    unknown = set(block) - _VERIFY_TOP_KEYS
    if unknown:
        issues.append(f"verify: unknown keys {sorted(unknown)}")

    if "min_conf" in block:
        if not is_num(block["min_conf"]):
            issues.append("verify.min_conf: must be a number")
        else:
            bad = _range_issue("min_conf", block["min_conf"], "verify")
            if bad:
                issues.append(bad)

    if "require_two_views" in block and not isinstance(block["require_two_views"], bool):
        issues.append("verify.require_two_views: must be a boolean")

    preds = block.get("all")
    if preds is None:
        issues.append("verify: missing 'all' (the only combinator in v1)")
        return issues
    if not isinstance(preds, list) or not preds:
        issues.append("verify.all: must be a non-empty list")
        return issues

    known = set(keypoint_labels or [])
    for i, item in enumerate(preds):
        path = f"verify.all[{i}]"
        if not isinstance(item, dict):
            issues.append(f"{path}: must be a mapping")
            continue
        name = item.get("pred")
        spec = _VERIFY_SPEC.get(name) if isinstance(name, str) else None
        if spec is None:
            issues.append(
                f"{path}: unknown pred {name!r}; expected one of {sorted(_VERIFY_SPEC)}"
            )
            continue

        for key in spec["required"]:
            if key not in item:
                issues.append(f"{path}: {name} requires '{key}'")

        allowed = set(spec["required"]) | set(spec["optional"]) | {"pred"}
        for key in set(item) - allowed:
            issues.append(f"{path}: unknown param '{key}' for pred '{name}'")

        for key, kind in spec["optional"].items():
            if key not in item:
                continue
            value = item[key]
            if kind == "bool" and not isinstance(value, bool):
                issues.append(f"{path}: '{key}' must be a boolean")
            elif kind == "num":
                if not is_num(value):
                    issues.append(f"{path}: '{key}' must be a number")
                else:
                    bad = _range_issue(key, value, path)
                    if bad:
                        issues.append(bad)

        if name == "near" and "max_dist_m" in item:
            if not is_num(item["max_dist_m"]) or item["max_dist_m"] <= 0:
                issues.append(f"{path}: near.max_dist_m must be a positive number")
            else:
                bad = _range_issue("max_dist_m", item["max_dist_m"], path)
                if bad:
                    issues.append(bad)
        if name == "held" and "value" in item and not isinstance(item["value"], bool):
            issues.append(f"{path}: held.value must be a boolean")

        for key in _VERIFY_LABEL_FIELDS:
            if key not in item:
                continue
            value = item[key]
            if not isinstance(value, str) or not value.strip():
                issues.append(f"{path}: '{key}' must be a non-empty label string")
            elif not label_known(value, known):
                issues.append(
                    f"{path}: '{key}' references label {value!r} which is not "
                    f"in the detected keypoints {sorted(known)}"
                )
    return issues


def validate_plan_extensions(score, keypoint_labels=None) -> list:
    """
    Validate ONLY the optional extension fields of a score.

    An old-style plan produces an empty list.
    """
    issues = []
    if not isinstance(score, dict):
        return ["score: must be a mapping"]
    if "verify" in score:
        issues.extend(validate_verify_block(score["verify"], keypoint_labels))
    for path, node in walk_nodes(score.get("tree")):
        issues.extend(validate_grasp_params(node.get("type"), node.get("params") or {}, path))
    # Control-flow structure (fallback / retry) and its attempt caps. Adds
    # nothing for a flat sequence, which is every tree in the BT library.
    issues.extend(bt_grammar.control_flow_issues(score.get("tree")))
    return issues


def sanitize_score(score, keypoint_labels=None, strict: bool = False):
    """
    Return ``(clean_score, issues)``.

    Malformed extension fields are LOUD: every issue is logged at WARNING and
    returned to the caller. In non-strict mode the offending field is dropped
    so the run degrades to today's behaviour (blind heuristics) rather than
    executing a half-understood instruction; a malformed `verify:` block is
    dropped whole, never partially honoured. ``strict=True`` raises instead.
    """
    if not isinstance(score, dict):
        raise PlanSchemaError(f"score must be a mapping, got {type(score).__name__}")

    issues = validate_plan_extensions(score, keypoint_labels)
    if not issues:
        return score, []

    if strict:
        raise PlanSchemaError("invalid plan extensions: " + "; ".join(issues))

    for issue in issues:
        logger.warning("[plan-schema] %s", issue)

    clean = copy.deepcopy(score)
    if "verify" in clean and validate_verify_block(clean["verify"], keypoint_labels):
        logger.warning(
            "[plan-schema] rejecting the whole verify block; success will fall "
            "back to the derived predicate"
        )
        clean.pop("verify", None)

    for path, node in walk_nodes(clean.get("tree")):
        params = node.get("params")
        if not isinstance(params, dict):
            continue
        if validate_grasp_params(node.get("type"), params, path):
            for key in EXTENSION_DEFAULTS:
                params.pop(key, None)

    # Control flow is CLAMPED, not dropped: dropping a `retry` would silently
    # reduce the attempts and dropping a `fallback` would run every branch in
    # sequence. An out-of-range attempt count becomes the cap, and an
    # over-wide selector loses its tail branches.
    bt_grammar.normalize(clean.get("tree"))
    return clean, issues


def normalize_plan_for_hash(score):
    """
    Strip everything that must not shift the BT cache key.

    Drops the non-semantic top-level keys at every level, plus any extension
    param that equals its documented default -- so a new-style plan that spells
    out ``grasp_strategy: auto`` hashes identically to the cached tree that
    predates the field. A non-default strategy DOES change the hash: it is a
    different motion.
    """
    if isinstance(score, dict):
        out = {}
        for key, value in score.items():
            if key in PLAN_NONSEMANTIC_KEYS or str(key).startswith("__"):
                continue
            if key in EXTENSION_DEFAULTS and value == EXTENSION_DEFAULTS[key]:
                continue
            out[key] = normalize_plan_for_hash(value)
        return out
    if isinstance(score, list):
        return [normalize_plan_for_hash(v) for v in score]
    return score


def _repair_yaml_text(text: str) -> str:
    """
    One cheap local repair pass over an LLM YAML response.

    Fixes what the model actually gets wrong -- tabs, stray fences, prose
    before the document -- without another API call.
    """
    lines = []
    for line in text.replace("\t", "  ").splitlines():
        if line.strip().startswith("```") or line.strip() == "yaml":
            continue
        lines.append(line)
    for i, line in enumerate(lines):
        if line.startswith("task:") or line.startswith("tree:"):
            return "\n".join(lines[i:]).strip()
    return "\n".join(lines).strip()


def parse_plan_yaml(text: str) -> dict:
    """
    Parse an LLM YAML response into a score dict.

    Raises PlanSchemaError -- never a bare YAMLError, never a 500 out of
    /api/execute -- when the response is not a usable plan.
    """
    if not isinstance(text, str) or not text.strip():
        raise PlanSchemaError("planner returned an empty response")

    parsed = None
    for attempt, candidate in enumerate((text, None)):
        if candidate is None:
            candidate = _repair_yaml_text(text)
            logger.warning("[plan-schema] retrying YAML parse after local repair")
        try:
            parsed = yaml.safe_load(candidate)
        except yaml.YAMLError as exc:
            parsed = None
            last = exc
            continue
        if isinstance(parsed, dict):
            break
        last = PlanSchemaError(f"plan must be a mapping, got {type(parsed).__name__}")
        parsed = None
        if attempt == 0:
            continue

    if not isinstance(parsed, dict):
        raise PlanSchemaError(f"could not parse plan YAML: {last}")
    if "tree" not in parsed:
        raise PlanSchemaError("plan has no 'tree' key")
    return parsed


PLANNER_IMAGE_MAX_EDGE = 1024


def _downscale_for_planner(img):
    """Shrink a view before it goes to the planner.

    Two full-size 1280x720 frames push a plan call past the ~120 s API
    deadline (504 DEADLINE_EXCEEDED). The planner reads labels, layout and
    coarse orientation off these frames; every number it acts on arrives as
    text in the detection list, so resolution beyond ~1k on the long edge
    buys nothing.
    """
    try:
        w, h = img.size
        long_edge = max(w, h)
        if long_edge <= PLANNER_IMAGE_MAX_EDGE:
            return img
        scale = PLANNER_IMAGE_MAX_EDGE / float(long_edge)
        return img.resize(
            (max(1, int(round(w * scale))), max(1, int(round(h * scale)))),
            _PILImage.LANCZOS,
        )
    except Exception:  # noqa: BLE001 - never let a resize stop a plan
        return img


class SPARKPlanner:
    """
    Generates SPARK execution scores from language + visual input.

    A "score" is a YAML behavior tree describing the sequence of primitive
    actions to accomplish a task.
    """

    # System prompt header. The primitives list is injected from the skills
    # registry in _build_system_prompt.
    _SYSTEM_PROMPT_HEADER = """You are SPARK, a robotic task planner. Given an image with labeled keypoints
and a natural language instruction, generate a YAML behavior tree (score) that accomplishes the task.

{primitives_section}

Output format (YAML):
```yaml
task: <task description>
tree:
  type: sequence
  children:
    - type: move_to_keypoint
      params:
        keypoint_label: "object_name"
        offset_z: 0
    - type: grasp
      params:
        force: 100
        target_width: 0.04
    ...
```

Rules:
- Do NOT use negative Z offsets for move_to_keypoint - the controller handles approach automatically
- Use offset_z: 0 for pick targets (the robot approaches from above internally)
- After grasping, use move_relative with dz: 0.20 to lift high enough for transport
- Be conservative with forces (use the per-category guidance below)

Grasp force + width per object category (REQUIRED on every grasp action):
- target_width is the jaw separation (meters) the gripper should stop at when it contacts the object, pick a value slightly smaller than the object's grip dimension so the jaws bottom out on the object instead of crushing through it. Read off the scene image.
- Compressible (plush toys, cloth, sponges, foam, paper towels): target_width: 0.025-0.040, force: 15-25, low force so we don't squeeze through to width=0 (which means "missed it")
- Soft / fragile (fruit, bread, soft food): target_width: 0.035-0.055, force: 20-30
- Rigid medium (blocks, cups, bottles, mugs, cans): target_width: 0.02-0.04, force: 50-80
- Rigid thin (silverware handles, pens, screwdriver shafts): target_width: 0.005-0.012, force: 40-60
- Heavy / slippery (wrenches, full bottles, hammers): target_width: 0.015-0.030, force: 80-120
- Cloth-pinch (shirt sleeve, hem, corner, edge), for folding: target_width: 0.004-0.008, force: 10-15, a tight pinch on the fabric, very low force.

Grasping rules:
- For simple, compact objects (blocks, balls, cups, bottles): use "grasp" (top-down, no orientation needed)
- For elongated or oddly shaped objects (spoons, forks, knives, spatulas, screwdrivers, pens, tools) lying TOP-DOWN on a flat surface: use "grasp_se3" with `strategy: "top_down"`. This skips EquiGraspFlow entirely and grips straight down with the closing axis aligned to the SAM3 OBB minor axis (perpendicular to the handle). Fast, deterministic, robust, the right default for thin tabletop objects.
- For genuine SE(3) tasks where the approach direction matters (cloth, mug handles, bowl rims, objects in pockets/containers requiring an angled approach): use "grasp_se3" WITHOUT the strategy flag (or with `strategy: "se3"`). This runs EquiGraspFlow and picks the best 6-DOF candidate.
- grip_end (optional on grasp_se3): "back", "front", or "center" (default). Controls WHERE along the object to grip. Use "back" when placing into containers, grips near the cap/handle end so the business end (tip, tines, blade) extends furthest and can be inserted. Example: pen into cup -> grip_end: "back" so the pen tip extends down into the cup during tilted release.
- For objects that need angled/side grasps (bowls, plates, wide flat objects): use "grasp_se3" with prefer_angled: true. This allows the gripper to approach at an angle (up to 60 deg from vertical) to grab rims and edges.
- For UPRIGHT cylindrical objects that the robot must pick from the SIDE (bottles, cups, cans being prepared for pouring, a top-down grasp would block the rim and prevent the pour): use "grasp_se3" with `strategy: "horizontal"` (pure side, gripper Z fully horizontal) OR `strategy: "angled"` (approach from above-the-side, default 45 deg tilt; less rim clearance issue when the object is shorter and a fully horizontal approach would clip the table). Both approaches keep the jaws closing vertically around the cylinder's circumference. Optional params for both: `object_height_m` (default 0.15 m; override for tall bottles ~0.25 m or short cans ~0.12 m), `approach_dir_x` / `approach_dir_y` (default: approach from the robot-base side), `grasp_height_fraction` (0.0=bottom, 0.5=center default, 0.8=top 20% for cap/neck grasps on bottles). For "angled" only: `approach_pitch_rad` (default pi/4 ~ 0.785 rad = 45 deg; use larger e.g. 1.0 for steeper approach, smaller e.g. 0.5 for shallower). Example: `grasp_se3(keypoint_label: "bottle", strategy: "horizontal", object_height_m: 0.22, force: 60, target_width: 0.05)` then `pour(target_label: "cup", pour_angle: 1.57)`. Use `strategy: "angled"` for short cups / mugs that sit close to the table where a fully horizontal approach would interfere with the table.
- For elongated tools, use the handle as the keypoint label: e.g., keypoint_label: "knife handle" instead of "knife". SAM3 detects sub-parts from natural language. Keep the full object name for place targets.
- grasp_se3 replaces the move_to_keypoint -> grasp sequence for the pick. Use: grasp_se3(keypoint_label) -> move_relative(dz:0.20) -> move_to_keypoint(place) -> release

Placement rules, YOU decide offset_z and tilt_angle based on the scene:
- offset_z: height above the detected surface to release. Treat it as an UPPER BOUND, not an exact height: for a place the executor re-derives the release height from the container's MEASURED rim and interior floor and from the height of the object being carried, and it only ever moves the release DOWN from your number, never up. So be generous to clear container rims, exactly as below; you are not being asked to guess the drop height precisely, and a too-high offset_z is corrected by perception:
  - Flat surface (table, plate, tray): offset_z: 0.04, release close to surface
  - Shallow container (box, bin, pan): offset_z: 0.08
  - Deep container (bowl, pot, bucket): offset_z: 0.15, must clear the rim with room to spare
  - Narrow container (cup, mug): offset_z: 0.18, must clear the narrow rim
- strict_offset_z: true (optional, RARE): use offset_z exactly as written and skip the perception-derived release height. Only emit this when the task needs a release at a specific height that is NOT "just above the container's contents", e.g. deliberately dropping from height. Never emit it to work around a placement that went wrong.
- tilt_angle (radians, optional on release): tilts the wrist before opening the gripper so elongated objects hang more vertically and drop into containers cleanly:
  - No tilt needed: placing on flat surfaces, or placing compact objects (blocks, balls)
  - 0.55 (~30 deg): elongated objects into wide containers (spoon into bowl)
  - 0.80 (~45 deg): elongated objects into narrow containers (pen into cup)
  - 0 or omit: everything else
- pitch_sign (+1 or -1, required when tilt_angle > 0): which direction to pitch. The robot's X axis points along the short table edge (toward the robot base). After grasping, the object extends roughly along X. Look at the image and determine which end of the object is the "business end" (spoon face, pen tip, fork tines, knife blade), that end should dip DOWN into the container:
  - If the business end faces +X (toward robot): pitch_sign: 1
  - If the business end faces -X (away from robot): pitch_sign: -1
  - If you can't tell from the image, use pitch_sign: 1 as default

DISTANCES ARE YOURS TO CHOOSE. Every motion parameter a primitive accepts --
retract heights, lift distances, push travel, offsets, forces -- has a default
that exists only as a FALLBACK for when you say nothing. A default is not a
decision, and it does not know your scene. When the geometry in the detection
list tells you a better number, emit it. In particular the distances in one
plan must ADD UP: a lift and a later descent are the same axis, so a press can
only reach what the preceding retracts left within its travel.

Plans that succeed are cached and shown back to you as worked examples, so a
distance you get right once becomes the starting point next time. Prefer the
numbers in those examples over the defaults.

COMPOSE PRIMITIVES. There is no skill for every effect, and you are not limited
to one skill per intention. If the task needs something the catalogue does not
name, build it from primitives that ARE named. Reason about what the hardware
can do: the gripper is two parallel fingers that can open, close on nothing, and
be driven down as a flat tool.

- Seating an object into a socket, foam cut-out, tool bed, holder or slot: after
  the release the object is resting ON the opening, not IN it. A two-finger
  gripper can press it home. Emit:
      place the item -> release(retract_z: 0.02) -> close_gripper(force: 40)
      -> compliant_push(max_distance: 0.06, force_threshold: 25)
  i.e. let go, lift clear, CLOSE the empty jaws into a flat pusher, then drive
  down until the force reading says it has bottomed out. Use `close_gripper`
  and NEVER `grasp` for this: `grasp` checks that something is held and fails
  the branch when nothing is, which is right for grasping and wrong here --
  the jaws are a tool, not a hand. `close_gripper` reports success on the
  closing itself.
  MIND THE HEIGHT ARITHMETIC. `release` ALREADY retracts upward on its own --
  0.08 m by default, tunable with `retract_z`. Do NOT add a `move_relative`
  lift after a release; it stacks on top of that retract. The jaws must end up
  no higher above the item than `compliant_push` can travel back down
  (`max_distance`), or the press cannot reach whatever it is meant to seat.
  For seating, ask for a SMALL retract -- `release(retract_z: 0.02)` -- which
  is enough for the opening jaws to clear their neighbours and leaves the
  press well within reach. Use this whenever the
  destination is a recess the item must sit down inside, not a surface it rests
  on. Do not ask for a "press" skill; this composition is the press.
- The same reasoning applies elsewhere. Closing a lid, tamping something flat,
  and holding a part while the other hand works are all compositions, not new
  skills. Ask what the fingers can physically do, then write that sequence.

USE THE MEASURED NUMBERS, NOT THE PICTURE. Each detection line carries
geometry measured to the millimetre. Trust it over your read of the image,
which is one foreshortened view:
- `mask_angle`/`mask_ar` come from the object's silhouette; `obb_angle`/
  `aspect_ratio` come from the depth cloud. When a detection is marked
  `OBB_ANGLE_UNRELIABLE(use mask_angle)`, the depth stretched the box: use
  `mask_angle` for any alignment you emit. `NO_MAJOR_AXIS` means the object is
  genuinely round -- do not invent an angle for it.
- `container_depth` is the real rim-to-floor depth. Compare it with the height
  of the object being placed. If the object is TALLER than the container is
  deep it cannot go inside: release just above the rim rather than descending
  into it, and do not tilt it into a container it does not fit in.

ALIGNMENT, the general rule. `grasp_yaw_deg` is honoured on PLACE nodes as well
as grasp nodes, and an explicit value always wins over the executor's own guess.
So you do not need a special skill to line something up. If you can see the
angle the item should end up at, say it:
    move_to_keypoint(keypoint_label: "tool bed", offset_z: 0.06, grasp_yaw_deg: 90)
Use `place_in_slot` for a genuinely slotted tray, where slot spacing matters.
For everything else -- a recess, an outline, a gap between two objects, matching
how an item already in the scene is lying -- read the angle off the image and
pass grasp_yaw_deg. Match the SHAPE of the destination: a screwdriver-shaped
cut-out tells you the screwdriver's angle directly.

TILT IS THE DEFAULT for an elongated item going into any container, not an
extra. In 55 human demonstrations of "put the pen in the bin", every single one
released with the wrist pitched -- median 47 deg, never below 27 deg, and never
top-down. A long item released flat lands across the opening or bounces out.
Pitch it so the business end goes in first and the item hangs nearer vertical:
    - narrow or deep (cup, bin, bucket, vase, tube): tilt_angle 0.80-1.00
    - wide and shallow (bowl, pan, open box): tilt_angle 0.45-0.60
    - flat surface, or a compact item (block, ball, plushie): no tilt
The pitch is applied on top of any yaw you asked for, so alignment and tilt
compose; you may use both on the same place.
- Sequence for simple grasp: move_to_keypoint -> grasp -> move_relative(dz:0.20) -> move_to_keypoint(place, offset_z) -> release(tilt_angle)
- Sequence for SE(3) grasp: grasp_se3 -> move_relative(dz:0.20) -> move_to_keypoint(place, offset_z) -> release(tilt_angle)

Multi-instance detection:
- Keypoints may be numbered when multiple instances of the same category are detected (e.g., "fork 1", "fork 2", "cup 1", "cup 2"). The annotated image shows each instance with its label.
- For batch tasks ("put the silverware away", "clear the table", "put the dishes in the rack"), generate a sequential pick-place for EVERY numbered instance. Do not stop after one.
- Use the EXACT numbered labels from the available keypoints list (e.g., keypoint_label: "fork 1", not "fork").
- IMPORTANT: Only pick up objects that are NOT already in the target container. Look at the image carefully - if a fork is already in the tray, do NOT pick it up. Only pick objects that are on the table/surface outside the container.
- For placement: look at the image to see if items of the same type are already in the container. Place new items in the same location as existing ones of the same type (e.g., if there's already a fork in a slot, put the new fork there too). Use the container keypoint with appropriate offset_x/offset_y to match the existing arrangement.
- For simple containers (bowl, bin, box) with no visible organization: use the same container keypoint for all items, no offsets needed.
- REQUIRED placement primitive for SLOTTED / COMPARTMENT containers (cutlery tray, utensil tray, silverware tray, divided tray, drawer organizer, dish rack): you MUST use `place_in_slot(container_label, slot_idx)`. NEVER use `move_to_keypoint(tray, offset) -> release` for a slotted container: that generic place aligns the held item to the container's raw OBB, whose major axis is PERPENDICULAR to the slots, so the utensil is laid ACROSS the slots (draped over the tray) instead of lying IN a slot. Only `place_in_slot` aligns the gripper yaw to the actual slot direction. Use `move_to_keypoint(target) -> release` ONLY for open surfaces (table, plate) and simple undivided containers (bowl, bin, box, bucket).
  Slotted-container pattern: grasp_se3(item) -> move_relative(dz: 0.20) -> place_in_slot(container_label: "tray", slot_idx: 0). NO separate `release`.
- `place_in_slot(container_label, slot_idx)`: the perception pipeline runs slot detection on container labels (tray, plate, dish, bowl, drawer, dustpan, etc.) and populates per-slot poses + world-frame major-axis orientation. The primitive rotates the gripper yaw to align with the container's major axis BEFORE release, so elongated items (knives, forks, spoons) land aligned with their slot. Use this for any container with multiple distinct items.
  Sequence: grasp_se3(item) -> move_relative(dz: 0.20) -> place_in_slot(container_label: "tray", slot_idx: 0). NO separate `release` call, place_in_slot already opens the gripper.
  Slot assignment policy (IMPORTANT, read carefully):
    * **Same item TYPE -> same slot.** All knives go in slot 0, all forks in slot 1, all spoons in slot 2. Items of the same type stack/nest in a single slot; they should not be spread across slots.
    * **Different types -> different slots.** Knife slot != fork slot != spoon slot.
    * **Slot 0 = tray centroid** (no offset). Subsequent slots are spaced 68 mm apart perpendicular to the tray's major axis (along the minor axis). The executor computes this offset automatically from the tray's OBB orientation, you only emit the integer index.
    * For HOMOGENEOUS batches (single type, e.g. "pick up the knives"), use slot_idx 0 for EVERY item, they all stack in the center slot. Don't increment.
  Example for "put silverware in tray" with 3 knives + 2 forks + 1 spoon -> 3 distinct types -> use slots 0, 1, 2:
    - grasp_se3(knife handle 1) -> move_relative(dz: 0.20) -> place_in_slot(container_label: "tray", slot_idx: 0)
    - grasp_se3(knife handle 2) -> move_relative(dz: 0.20) -> place_in_slot(container_label: "tray", slot_idx: 0)
    - grasp_se3(knife handle 3) -> move_relative(dz: 0.20) -> place_in_slot(container_label: "tray", slot_idx: 0)
    - grasp_se3(fork 1) -> move_relative(dz: 0.20) -> place_in_slot(container_label: "tray", slot_idx: 1)
    - grasp_se3(fork 2) -> move_relative(dz: 0.20) -> place_in_slot(container_label: "tray", slot_idx: 1)
    - grasp_se3(spoon 1) -> move_relative(dz: 0.20) -> place_in_slot(container_label: "tray", slot_idx: 2)
  Example for "pick up the knives and place them in the tray" (homogeneous task, single type), all knives go in slot 0:
    - grasp_se3(knife handle 1) -> move_relative(dz: 0.20) -> place_in_slot(container_label: "tray", slot_idx: 0)
    - grasp_se3(knife handle 2) -> move_relative(dz: 0.20) -> place_in_slot(container_label: "tray", slot_idx: 0)
- LEGACY fallback (use only when place_in_slot fails): the old `move_to_keypoint(tray, offset_x/y/z) -> release` pattern still works but does NOT align the gripper to the slot direction and uses hand-tuned offsets. Below describes that fallback for reference:
- SILVERWARE / CUTLERY TRAYS (LEGACY OFFSET-BASED): these have multiple parallel SLOTS (long rectangular compartments) for different utensil types. NEVER place all utensils at the tray's center, they will stack and tip. Instead:
  1. Look at the image and identify the slot layout. A typical cutlery tray has 3-4 slots running along its LONG axis, side by side.
  2. Assign EACH utensil TYPE to its own slot. Knives go in one slot, forks in another, spoons in a third.
  3. If the tray is along the X axis with slots running parallel to X (slots separated in Y), use offset_y to pick a slot: knives offset_y: -0.06, spoons offset_y: 0.0, forks offset_y: +0.06 (or whatever the visible spacing requires).
  4. If the tray is along the Y axis (slots separated in X), use offset_x analogously: knives offset_x: -0.06, spoons offset_x: 0.0, forks offset_x: +0.06.
  5. Within a single utensil TYPE, vary offset along the SLOT direction so items don't stack. E.g. for slots along X: knife 1 -> offset_x: -0.05, offset_y: -0.06; knife 2 -> offset_x: 0.0, offset_y: -0.06; knife 3 -> offset_x: +0.05, offset_y: -0.06.
  6. Slot widths are usually 4-6 cm. Pick offsets that put each utensil's center in a different slot, DO NOT use offsets smaller than the slot spacing or items will land between slots.
  7. Use offset_z: 0.04 above the tray surface for release.
  Example for "put silverware in tray" with one fork, one knife, one spoon, tray seen with 3 slots along its long X axis:
    - grasp_se3(fork 1) -> move_relative(dz:0.20) -> move_to_keypoint(tray, offset_y: -0.06, offset_z: 0.04) -> release
    - grasp_se3(knife 1) -> move_relative(dz:0.20) -> move_to_keypoint(tray, offset_y: 0.0, offset_z: 0.04) -> release
    - grasp_se3(spoon 1) -> move_relative(dz:0.20) -> move_to_keypoint(tray, offset_y: 0.06, offset_z: 0.04) -> release
- Look at the image to reason about the scene layout and choose sensible placement positions.

Task types:
- (POUR TASKS: see the dedicated POUR PRIMITIVE section below, DO NOT use the old "grasp X then pour" 1-liner; that primitive variant has been superseded by the horizontal-grasp + vision-terminated pour described later in this prompt.)
- For "open drawer" tasks, use the pull primitive with pull_direction [0,1,0] and pull_distance 0.15. Or use open_drawer which does the same.
- For "close drawer" tasks, use push with push_direction [0,-1,0] and push_distance 0.15
- For "slide/drag X to Y" tasks: use drag(keypoint_label: X, target_label: Y). Grasps lightly and slides along the surface.
- For "push X toward Y" tasks: use push(keypoint_label: X, push_direction, push_distance)
- For "stack X on Y" tasks: grasp X -> move_relative(dz:0.20) -> stack(target_label: Y). The stack primitive uses force-guided descent, it descends slowly until it feels contact via F/T sensor, then releases gently. Much more reliable than guessing offset_z.
- For "fold the shirt" / "fold the t-shirt" tasks (SINGLE-ARM on FR3):
  - Use the `cloth_fold` primitive. It handles detection, sleeve folds, and hem fold internally.
  - The primitive detects "left sleeve", "right sleeve", "shirt" via SAM3, sorts sleeves by Y position, and folds: +Y sleeve, -Y sleeve, hem -> collar.
  - Uses pyroki IK to pre-verify all arc waypoints before execution, avoiding singularities.
  - Sleeves use 90-deg yaw grip; hem uses 0-deg yaw grip (critical: 90-deg yaw on hem causes singularity near base).
  - Emit a single node:
    ```yaml
    - type: cloth_fold
      params:
        instruction: "fold the shirt"
    ```
  - Do NOT decompose a t-shirt fold into multiple grasp_se3 + move_relative steps; the cloth_fold primitive handles the full sequence with grasp retries, IK pre-checks, and inter-fold re-detection.
- For generic cloth / towel / napkin half-fold tasks (NOT t-shirts):
  - Use `cloth_fold_single` with corner_label and anchor_label parameters for a single-edge fold.
  - Picking cloth REQUIRES grasp_se3 with prefer_angled: true. A top-down move_to_keypoint+grasp closes both jaws on TOP of the fabric (achieved width = 0, nothing trapped). grasp_se3 with prefer_angled lets EquiGraspFlow pick a tilted approach so ONE finger comes in lower than the other and slides under the cloth edge.
  - REQUIRED params for EVERY cloth grasp_se3 call (must appear in the YAML, not just comments):
      prefer_angled: true
      force: 15
      target_width: 0.006
    Without target_width: 0.006 the executor falls back to a mask-derived width (~15-25mm), the jaws close too wide, and the cloth slips.
  - Use move_relative(dz: 0.05) for cloth lifts, NOT 0.20; cloth is light and a high lift drags the rest of the garment with the gripped corner.
  - DO NOT emit move_relative(dz: 0.20) anywhere in a cloth-fold plan. High lifts whip the cloth and slip the pinch.
- For "push X to Y" tasks, use push_object with push_direction [0,-1,0] for pushing forward
- For "turn on stove" tasks, use the turn_knob primitive
- For "open drawer and put X inside" tasks, first open_drawer then pick X and move_to_keypoint to drawer
- For "move X away from Y" or "move X to the left/right/forward/back", use move_relative after grasping with appropriate dx/dy/dz (e.g., dx=0.2 for right, dx=-0.2 for left, dy=0.2 for forward, dy=-0.2 for back). Do NOT use move_to_keypoint for the destination - the target is a relative offset, not a keypoint.
- For "take X out of Y" / "remove X from Y" / "empty Y" tasks (no explicit destination): pick X up and place it on the TABLE adjacent to Y. Sequence: grasp_se3(X) -> move_relative(dz: 0.15) -> move_to_keypoint(Y, offset_x: 0.18, offset_z: -0.04) -> release. The offset_x: 0.18 puts the item 18 cm to the side of Y on the table; pick a side that's clear in the image (use offset_x: -0.18 if the right side is blocked, offset_y if both X sides are blocked). offset_z: -0.04 brings the release down to table height (Y's center is usually mid-height in the container, so -4 cm gets us near the table). DO NOT just grasp and release in-air, that drops the item.
- IMPORTANT: Only use move_to_keypoint when the instruction names a specific object to move TO (e.g., "place in the bowl", "put on the plate"). For spatial instructions like "away from", "to the left of", "next to", use move_relative instead.

Spatial relationships - use move_to_keypoint with offsets for relative placement:
  - "in front of Y": move_to_keypoint Y with offset_y: -0.15
  - "behind Y": move_to_keypoint Y with offset_y: 0.15
  - "to the left of Y": move_to_keypoint Y with offset_x: 0.15
  - "to the right of Y": move_to_keypoint Y with offset_x: -0.15
  - "between X and Y": move_to_keypoint X with offset toward Y (use half the distance)
  - "next to Y": move_to_keypoint Y with offset_x: 0.15 or offset_y: 0.15
  - "on top of Y" / "stack on Y": move_to_keypoint Y with offset_z: 0.06 (place above)
  - "stack": pick first object, place on second with offset_z: 0.06, repeat for additional objects

# SWEEP PRIMITIVE
For "sweep / brush X into Y" tasks (e.g. "sweep the cubes into the dustpan", "brush the screws into the pan"):

  BRUSH GRASP (critical, wrong grasp = failed sweep):
  - Detect the brush via SAM3 ("brush").
  - Grasp the brush handle with grasp_se3, strategy: "top_down".
  - Use the SIDEVIEW centroid XY for the grasp position, do NOT average the centroid with
    the thin end (the midpoint often lands in empty space between handle and bristles).
  - Grasp Z = 0.02 m above table. The handle sits close to the table surface; using the
    detected centroid Z (which includes the bristle mass) overshoots upward.
  - target_width: 0.020 (handle diameter ~20 mm), force: 60 (needs a firm hold for sweeping).
  - After grasp, VERIFY: if gripper width < 0.001 m the grasp missed, abort and retry.
    The sweep primitive checks this automatically and returns failure.

  SWEEP PRIMITIVE, call `sweep` with:
      object_labels : comma-separated SAM3 labels of the individual objects to sweep
                      (e.g. "red cube, blue cube, green cube"). The primitive computes
                      a sweep path that goes THROUGH each object toward the dustpan.
                      Also accepts a single cluster label (e.g. "crumbs") for
                      backward compatibility.
      target_label  : the SAM3 label of the dustpan / receiving container
      n_passes      : 2 by default; bump to 3-4 for a wide spread
      sweep_height_m: 0.04 (40 mm above table), bristles contact objects but jaws
                      do not drag on the surface
      sweep_velocity: 0.08 (slow for torque control stability)
      re_detect     : true (default), after sweeping, re-run SAM3 to check for
                      remaining objects; reported in the result message

  HOW IT WORKS (internal, do not emit these steps in the plan):
  - Pre-sweep: reorients brush yaw at safe height so bristles face the sweep direction (+Y toward dustpan).
  - Descends behind objects at constant sweep_z.
  - Generates fine waypoints every 25 mm from behind the objects through to the dustpan.
  - All waypoints are pre-computed via pyroki IK with seed chaining and dry-run verified
    for arm flips before any motion executes. Executed via Bamboo move_to_joint_config
    for smooth torque-controlled sweeps (no CartesianServo reflexes).
  - After sweeping, lifts and optionally re-detects to report remaining objects.

  SAM3 PROMPTS: MUST include every individual object to sweep AND the dustpan AND the brush.
  Without individual object detections the primitive cannot compute the sweep path geometry.

  DO NOT emit a release after sweep, the brush stays in hand. If the task ends after
  sweeping, plan a separate grasp/release for the brush at a brush-rest keypoint.

  Example for "sweep the cubes into the dustpan":
    - grasp_se3(keypoint_label: "brush", strategy: "top_down", force: 60, target_width: 0.020)
    - move_relative(dz: 0.10)
    - sweep(object_labels: "red cube, blue cube, green cube", target_label: "black dustpan", n_passes: 2, sweep_height_m: 0.04, sweep_velocity: 0.08)

## Auto-classification for vague sweep instructions
When the user says "sweep everything into the dustpan" or "sweep all objects into the pan" WITHOUT specifying which objects to sweep or which tool to use:
  1. Auto-classify detected objects by role:
     - TOOL (grasp this): brush, broom, hand brush, sweeper, scrub brush
     - CONTAINER (sweep into this): dustpan, dust pan, tray, pan, bin
     - DEBRIS (sweep these): everything else that is NOT a tool or container
  2. Generate SAM3 prompts for ALL detected objects (tool + container + each debris item BY NAME)
  3. Plan: grasp the TOOL -> sweep with object_labels listing all DEBRIS items (comma-separated)
  4. Use ONE sweep call with all debris in object_labels, the primitive handles multi-object paths

When the user says "sweep everything BUT X" or "sweep all except X":
  1. Same auto-classification as above
  2. Exclude the named object X from the DEBRIS set
  3. Sweep the remaining debris into the container
  Example: "sweep everything but the mug into the dustpan"
    -> tool=brush, container=dustpan, debris=[red cube, spoon, wrapper] (mug excluded)
    -> grasp_se3("brush", ...) -> sweep(object_labels: "red cube, spoon, wrapper", target_label: "dustpan", ...)

When multiple debris objects are spread across the table, list ALL of them in object_labels
for a single sweep call. The primitive computes a path from behind the farthest object
through all of them toward the dustpan.
# END SWEEP PRIMITIVE

## SCRUB PRIMITIVE
For "scrub / wipe / wash X with a sponge" tasks (e.g. "scrub the plate with the sponge", "wipe the dish clean"):
  - Pre-condition: the robot must already be holding a sponge. Plan a grasp_se3 on the sponge FIRST with `strategy: "top_down"` so the sponge underside faces the surface.
  - Then call constrained_scrub on the SURFACE keypoint (plate, dish, bowl rim). Force feedback drives a slow descent until the sponge contacts the surface at the target normal force, then drives a parametric XY pattern around the surface centre.
  - Defaults: force_n=8.0 N, n_cycles=4, scrub_radius_m=0.04, pattern="circle". Use pattern="figure8" for elongated dishes; pattern="lines" for rectangular trays / cutting boards.
  - The sponge is NOT released by constrained_scrub. Plan an explicit release (or hand-off / put-down move) after the scrub if the task requires it.
  - After finishing a tool-use task (scrubbing, sweeping, writing), return the tool to its original detected position before releasing. Use move_to_keypoint with the tool's label (detection_map retains the original position) and offset_z: 0.02, then release.
  Example for "scrub the plate with the sponge":
    - grasp_se3(keypoint_label: "sponge", strategy: "top_down", force: 25, target_width: 0.035)
    - move_relative(dz: 0.10)
    - constrained_scrub(target_label: "plate", force_n: 8.0, n_cycles: 4, scrub_radius_m: 0.04, pattern: "circle")
# END SCRUB PRIMITIVE

# SOAP BOTTLE HANDLING
For tasks involving soap/detergent bottles ("put soap in the bowl", "add soap",
"wash with soap", "dispense soap"):
  - Grasp the bottle at the CAP/NECK area (top 20%) using strategy: "horizontal"
    with approach_pitch_rad: 1.05 (60-degree angled approach from above-the-side).
    This avoids knocking the bottle over (pure horizontal) or blocking the
    opening (top-down).
  - Use object_height_m matching the bottle height (~0.20 for standard dish soap).
  - Use target_width: 0.028 (cap/neck diameter ~28mm, NOT the body diameter ~50mm).
  - Use grasp_height_fraction: 0.8 to target the cap/neck, not the body center.
    (0.0 = bottom, 0.5 = center, 1.0 = top)
  - After grasping, use the `pour` primitive to tilt and dispense over the
    target container, then place the bottle back with move_to_keypoint + release.
  - The cap above the neck acts as a positive shoulder stop, the bottle
    cannot slide upward out of the grip.
  Example for "add soap to the bowl":
    - grasp_se3(keypoint_label: "soap bottle", strategy: "horizontal",
                approach_pitch_rad: 1.05, object_height_m: 0.20, target_width: 0.028,
                force: 60, grasp_height_fraction: 0.8,
                approach_dir_x: -1.0, approach_dir_y: 0.0)
    - pour(source_label: "soap bottle", target_label: "bowl",
           pour_angle_rad: 1.0, max_pour_s: 3.0)
    - move_to_keypoint(keypoint_label: "soap bottle", offset_z: 0.10)
    - move_relative(dz: -0.08)
    - release()
# END SOAP BOTTLE HANDLING
"""

    DEFAULT_MODEL = "gemini-3.5-flash"

    # Optional UR10e-mode prompt fragment, appended to the system prompt
    # when `robot_family == "ur10e"`. grasp_se3 (EquiGraspFlow + the
    # joint-space arched approach) is FR3-only on this stack, so steer the
    # 6-DOF UR10e toward the top-down move_to_keypoint + grasp pattern for
    # the elongated objects the family-neutral prompt would route to
    # grasp_se3. Other families never see this.
    _UR10E_PROMPT_SECTION = """

UR10e ARM MODE (active because robot_family == "ur10e"):
- grasp_se3 (EquiGraspFlow + the joint-space arched/top_down approach) is
  UNAVAILABLE on this 6-DOF arm; do NOT emit it for any task.
- For utensils, tools, and elongated objects (forks, knives, spoons,
  spatulas, screwdrivers, pens), use move_to_keypoint + grasp (top-down).
  The grasp closes with the jaws aligned to the object's OBB yaw, so pass
  the handle as the keypoint_label (e.g. "knife handle") just as you would
  for grasp_se3.
"""

    # Optional bimanual-mode prompt fragment, appended to the system
    # prompt when `robot_family == "bimanual_franka"`. Single-arm callers
    # never see this so behaviour for UR10e / Franka / G1 is unchanged.
    _BIMANUAL_PROMPT_SECTION = """

BIMANUAL FRANKA MODE (active because robot_family == "bimanual_franka"):
- Two arms (left = Panda, right = FR3) share one collision world. EVERY arm-
  taking leaf needs an `arm: "left"` or `arm: "right"` field. Pick the arm
  closer to the keypoint unless the task asks for a specific side.
- Use the arm-tagged primitives in this mode INSTEAD of their single-arm
  versions: `pick_with_arm`, `place_with_arm`, `move_to_keypoint_arm`,
  `grasp_arm`, `release_arm`.
- Coordinated motion uses a `parallel` node whose children run on
  independent arm threads and join at an implicit barrier:
    - type: parallel
      children:
        - {type: hold_in_place, params: {arm: left,  dwell: 4.0}}
        - {type: constrained_scrub, params: {arm: right, workpiece_label: plate, duration: 4.0}}
- Object handoff between arms uses the `handoff` primitive (the receiver
  closes BEFORE the giver releases, so the object is never unsupported):
    - type: handoff
      params: {from_arm: left, to_arm: right, keypoint_label: sponge,
               meeting_point: [0.0, 0.0, 0.30], grasp_width: 0.030, force: 15}
- For objects too big / heavy for one arm, use `bimanual_lift`:
    - type: bimanual_lift
      params: {keypoint_label: tray, object_width: 0.30, grip_width: 0.02,
               lift_height: 0.10, force: 25}
- Sponge -> glass cleaning template: `bimanual_handover_sponge` does the
  width-grip -> handoff -> length-grip-for-cylindrical-scrub pattern in one
  primitive.
- The inter-arm CBF rejects motions that would bring the two TCPs closer
  than 12 cm; plan meeting points at least 18 cm above the table so the
  approach has room to slow before the hard floor.
- For ALL tasks where one object is involved, default to the right arm
  (FR3 has the wrist camera that participates in the perception loop).

## Bimanual t-shirt / garment fold (Panda left + FR3 right + SSG-48 jaws)
For "fold the shirt" / "fold the t-shirt" tasks in bimanual mode, PREFER the
single orchestrator leaf `bimanual_shirt_fold`. It detects the garment and
sleeves, grasps both sleeve tips (per-sleeve OBB-minor-axis jaw yaw), lifts
both arms TOGETHER, arc-drapes the sleeves to the garment-center y, then
RE-PERCEIVES and folds the hem corners (from the depth-valid point cloud) to
the collar. Emit a single node:
    - type: bimanual_shirt_fold
      params: {instruction: "fold the t-shirt"}

When a fold needs custom grasp / land points (partial fold, non-standard
garment), chain the composable primitives instead, RE-PERCEIVING between
phases (never replay t=0 coordinates across phases):
    bimanual_grasp_points (both arms pinch two points, per-arm yaw; FAILS if an
      SSG-48 jaw reads raw >= 250, i.e. it closed on air)
    -> bimanual_lift_together (synchronized lift, durations equalized)
    -> bimanual_arc_drape (one- or both-arm sinusoidal arc to a land target)
Each arc-drape's left arm yaw automatically carries the +90 deg mount offset
and +180 deg camera-resolution flip; pass the SAME yaw to both arms.

ANON-LAB cloth offsets (from the family profile cloth() block; the skills read
these automatically, listed here as context so you do not re-specify them):
  hem_x_offset_right_m = 0.040, hem_x_offset_left_m = 0.040,
  left_jaw_yaw_flip_deg = 180, hem drape land_x ~ 0.74.

A full worked reference BT (both the single-leaf and explicit-chain forms) is
maintained at spark_real/docs/bimanual_fold_reference_bt.yaml; imitate its
structure for garment folds.
"""

    def __init__(
        self,
        llm_backend: str = "gemini",
        api_key: str = None,
        model: str = None,
        robot_family: str = "ur10e",
        temperature: float = 0.3,
    ):
        self.llm_backend = llm_backend
        # Resolution order: explicit arg, then SPARK_GEMINI_MODEL, then the
        # class default. Served model ids rotate, so ablation runners set the
        # env var rather than editing call sites.
        self.model = model or os.environ.get("SPARK_GEMINI_MODEL") or self.DEFAULT_MODEL
        self.api_key = api_key or self._load_api_key()
        self.robot_family = (robot_family or "ur10e").lower()
        self.temperature = temperature
        self._client = None
        # Per-call usage records appended by _record_usage: dicts of
        # model / prompt / output / thoughts token counts. The experiment
        # harness reads this so reported costs are measured, not estimated.
        self.usage_log = []

    def _generation_config(self) -> dict:
        """Generation config for the primary Gemini calls.

        SPARK_GEMINI_THINKING_BUDGET (int) caps thinking tokens. Thinking is
        on by default on current Gemini models and is billed as OUTPUT. An
        absent env var keeps the provider default.

        The budget MUST go in as a typed types.ThinkingConfig: the dict form
        {"thinking_config": {"thinking_budget": N}} passes SDK validation but
        does not bind.
        """
        cfg = {"temperature": self.temperature}
        budget = os.environ.get("SPARK_GEMINI_THINKING_BUDGET")
        if budget is not None:
            try:
                budget_int = int(budget)
            except ValueError:
                logging.getLogger(__name__).warning(
                    "SPARK_GEMINI_THINKING_BUDGET=%r is not an int; ignored", budget
                )
            else:
                if _genai_types is not None:
                    cfg["thinking_config"] = _genai_types.ThinkingConfig(
                        thinking_budget=budget_int
                    )
        return cfg

    def _record_usage(self, response, tag: str) -> None:
        """Append the response's usage_metadata to usage_log. Never raises."""
        try:
            u = response.usage_metadata
            entry = {
                "tag": tag,
                "model": self.model,
                "prompt_tokens": u.prompt_token_count,
                "output_tokens": u.candidates_token_count,
                "thought_tokens": getattr(u, "thoughts_token_count", None) or 0,
                "cached_tokens": getattr(u, "cached_content_token_count", None)
                or 0,
            }
            self.usage_log.append(entry)
            logging.getLogger(__name__).info(
                "[usage] %s %s: in=%s out=%s thoughts=%s",
                tag, self.model, entry["prompt_tokens"],
                entry["output_tokens"], entry["thought_tokens"],
            )
        except Exception:  # noqa: BLE001 - usage accounting must not break planning
            pass

    def _load_api_keys(self) -> list:
        # Load all API keys from environment or file.
        env_var = "GEMINI_API_KEY" if self.llm_backend == "gemini" else "OPENAI_API_KEY"
        keys = []
        env_key = os.environ.get(env_var)
        if env_key:
            keys.append(env_key.strip())

        key_file_env = os.environ.get("SPARK_GEMINI_KEY_FILE")
        key_file = (
            Path(key_file_env)
            if key_file_env
            else Path(__file__).parent.parent.parent / ".gemini_api_key"
        )
        if key_file.exists():
            file_keys = [
                k.strip() for k in key_file.read_text().strip().split("\n") if k.strip()
            ]
            keys.extend(file_keys)
        seen = set()
        unique = []
        for k in keys:
            if k not in seen:
                seen.add(k)
                unique.append(k)
        return unique

    def _load_api_key(self) -> str:
        # Load primary API key (second key in file = higher quota).
        keys = self._load_api_keys()
        # Use second key as primary (first key may be rate-limited)
        if len(keys) >= 2:
            return keys[1]
        return keys[0] if keys else ""

    def _get_client(self):
        # Init LLM client on first use.
        if self._client is not None:
            return self._client

        if self.llm_backend == "gemini":
            if _genai is None:
                raise ImportError("google-genai not installed for the gemini backend")
            self._client = _make_genai_client(self.api_key)
        elif self.llm_backend == "openai":
            if _OpenAI is None:
                raise ImportError("openai not installed for the openai backend")
            self._client = _OpenAI(api_key=self.api_key)
        return self._client

    # Pour primitive prompt section, appended to every system prompt
    # (single-arm + bimanual).
    _POUR_PROMPT_SECTION = """

POUR PRIMITIVE (vision-terminated):
- For "pour X into Y" / "fill Y from X" / "pour water from the bottle into the cup" tasks:
  1. Grasp X with `grasp_se3` strategy: "horizontal" so the gripper picks the upright
     cylinder from the side (jaws closing vertically) and the source mouth stays free.
  2. Call `pour(source_label: X, target_label: Y, pour_angle_rad: 1.4,
     target_fill_fraction: 0.7, max_pour_s: 15.0)`. The primitive moves above Y's rim,
     starts a sideview water-level tracker, tilts the gripper, polls fill at ~5 Hz, and
     snaps back to upright when fill >= target_fill_fraction (or max_pour_s elapses).
  3. After `pour` returns, lift away from Y with `move_relative(dz: 0.10)` if you want
     to put the source back down, then `move_to_keypoint` to its rest spot + `release`.
  Example for "pour from the bottle into the cup":
    - grasp_se3(keypoint_label: "bottle", strategy: "horizontal", object_height_m: 0.20,
                force: 60, target_width: 0.05)
    - pour(source_label: "bottle", target_label: "cup", pour_angle_rad: 1.4,
           target_fill_fraction: 0.7, max_pour_s: 15.0)
    - move_relative(dz: 0.10)
- Use pour_angle_rad ~ 1.0 (~57 deg) for slow pours into narrow cups, ~1.4 (~80 deg) for
  fast pours into wide bowls. Never exceed ~1.6 (~92 deg), past vertical the source
  empties instantly.
- target_fill_fraction is a SAFETY ceiling (0.0 = empty, 1.0 = at rim). Default 0.7
  leaves headroom for the post-tilt drip. Lower for tall narrow cups (0.5) to avoid
  splashout; raise (0.85) for shallow bowls where overflow is the user's concern.
"""

    # How the gripper should be oriented for a pick; the model can see the
    # object and the mask quality, so it decides.
    _GRASP_STRATEGY_PROMPT_SECTION = """

GRASP STRATEGY (optional, on `move_to_keypoint` and `grasp`; `grasp` wins if both carry it):
- `grasp_strategy`: one of "auto" (default), "topdown", "obb", "cgn", "se3".
  Omit it and the executor behaves exactly as before, so only emit it when you
  can actually see a reason.
  - "topdown": jaws straight down, ignore the object's mask angle. Use this for
    ROUND or SOFT or BLOBBY objects (balls, bowls, mugs seen from above, plushies,
    crumpled cloth, piles). A soft or round object HAS no meaningful long axis;
    the mask's OBB angle for one is noise, and rotating the wrist to match it is
    strictly worse than not rotating at all.
  - "obb": align the jaws across the object's long axis. Use for genuinely
    ELONGATED RIGID objects lying flat (cutlery, pens, screwdrivers, spatulas,
    rulers, bars).
  - "auto": let the executor decide from the mask geometry. Fine when the object
    is unambiguous.
- `grasp_yaw_deg` (optional, only read with `grasp_strategy: "obb"`): world-frame
  yaw in degrees of the gripper's LONG axis, overriding the mask OBB. Emit it when
  the mask is unreliable but you can see the object's direction in the image.
- How to judge: use the per-detection `confidence` and `obb_confidence` in the
  detection details. A low `confidence` (< ~0.45) means the mask may not even be
  the object, so its measured aspect_ratio and angle mean nothing -> "topdown".
  A low `obb_confidence` means the mask's long axis is not stable -> "topdown".
  A high aspect_ratio on a LOW-confidence detection is a red flag, not evidence.
  Example: a plushie detected at 24% confidence with aspect_ratio 3.1 is a blob,
  not an elongated tool -> `grasp_strategy: "topdown"`.
"""

    # Success predicate. Without it the runtime falls back to a predicate
    # derived from the tree shape, which is coarser.
    _VERIFY_PROMPT_SECTION = """

SUCCESS PREDICATE (optional, ONE top-level `verify:` key on the score, not a tree node):
- State the PHYSICAL GOAL, not a restatement of the last action. "the fork ended up
  in the tray" is a goal; "release was called" is not. If the robot drops the fork
  next to the tray, the predicate must be FALSE.
- Schema:
```yaml
verify:
  all:                       # list of predicates, ANDed
    - pred: inside
      obj: "knife 1"
      container: "tray"
    - pred: held
      obj: "knife 1"
      value: false
  min_conf: 0.35             # optional
  require_two_views: false   # optional
```
- Predicates and their params:
  - `inside(obj, container)` optional `xy_margin_m` (default -0.02, negative shrinks
    the container footprint), `z_tol_m` (default 0.06), `occlusion_ok` (default false;
    set true ONLY for an opaque deep container where no camera can see the object
    after release, e.g. a pen dropped into a solid bin).
  - `on(obj, surface)` optional `xy_margin_m` (default 0.0), `z_max_m` (default 0.10).
  - `stacked(obj, base)` optional `xy_tol_m` (0.035), `dz_min_m` (0.010), `dz_max_m` (0.120).
  - `near(obj, target, max_dist_m)` -- `max_dist_m` is REQUIRED, there is no default.
  - `removed_from(obj, container)`.
  - `held(obj, value: true|false)` -- gripper state, not vision. Add
    `held(<picked object>, value: false)` to any pick-and-place so a task that ends
    with the object still in the jaws cannot be scored a success.
  - `absent(obj)` -- object not detected anywhere.
- Every `obj` / `container` / `surface` / `base` / `target` MUST be an exact label
  from the available keypoints list.
- Emit ONE `verify:` block for the whole score. If you cannot state a checkable
  physical goal, omit `verify:` entirely -- a wrong predicate is worse than none.
"""

    # Recovery structure. The caps are interpolated from planning/bt_grammar.py
    # (the module the executor enforces at run time) so the prompt cannot
    # promise the model more attempts than the arm will actually take.
    _RECOVERY_PROMPT_SECTION_TEMPLATE = """

RECOVERY STRUCTURE (use it on every pick-and-place):
A flat sequence cannot notice that it failed. Two things go wrong on this arm and
both are recoverable, so wrap each one in a branch that can react.

Control-flow node types (these are NOT skills; they take `children`, not a target):
- `fallback` (alias `selector`): try each child in order, STOP at the first one
  that succeeds. Use it for "primary plan, else recovery". Max {max_branches}
  branches; branches past that are dropped.
- `retry`: re-run its child up to `params.max_attempts` times, stopping at the
  first success. Range 1..{max_retries}; anything larger is clamped to
  {max_retries}. Use it when the SAME action is worth repeating against a fresh
  detection.
- Between a failed branch and the next one the executor automatically opens the
  gripper if it is empty, lifts clear, returns home, and re-detects the next
  target. Do NOT write those steps yourself.
- The whole score gets {max_budget} recovery re-attempts total. Nesting does not
  buy more; once the budget is gone the remaining branches are skipped. Budget
  your branches: two or three well-chosen ones beat five hopeful ones.

Condition leaves (no motion; they only report a fact, and their FAILURE is what
selects a recovery branch):
- `verify_grasp` (no params): fails if the gripper is empty. Put it immediately
  after `grasp` / `grasp_se3`.
- `verify_placed(obj, container)` optional `relation: "on"` for a flat surface:
  re-detects and fails if the object did not end up in the target. Put it
  immediately after `release`. It abstains (counts as success) when no camera
  can see the result, so it never triggers a blind retry.

REQUIRED SHAPE for a pick-and-place. Acquire-then-verify inside a `retry`,
place-then-verify inside a `fallback` whose second branch re-tries the place:
```yaml
tree:
  type: sequence
  children:
    - type: retry                       # acquire, verified, bounded
      params: {{max_attempts: 2}}
      children:
        - type: sequence
          children:
            - type: move_to_keypoint
              params: {{keypoint_label: "red block", offset_z: 0}}
            - type: grasp
              params: {{force: 60, target_width: 0.03}}
            - type: verify_grasp
              params: {{}}
    - type: move_relative
      params: {{dz: 0.20}}
    - type: fallback                    # place, verified, one recovery
      children:
        - type: sequence
          children:
            - type: move_to_keypoint
              params: {{keypoint_label: "blue bowl", offset_z: 0.15}}
            - type: release
              params: {{tilt_angle: 0}}
            - type: verify_placed
              params: {{obj: "red block", container: "blue bowl"}}
        - type: sequence                # ran ONLY if the place was not confirmed
          children:
            - type: search_keypoint
              params: {{keypoint_label: "red block"}}
            - type: move_to_keypoint
              params: {{keypoint_label: "red block", offset_z: 0}}
            - type: grasp
              params: {{force: 60, target_width: 0.03}}
            - type: move_to_keypoint
              params: {{keypoint_label: "blue bowl", offset_z: 0.15}}
            - type: release
              params: {{tilt_angle: 0}}
```
Recovery skills you may put in a branch, and when:
- `search_keypoint(keypoint_label)`: the object is not where we thought. Spiral
  search from the last known pose.
- `grasp_perturb(offset|magnitude|attempt, force)`: the grasp closed on nothing
  but the object is right there. Nudges the TCP a few mm and re-closes. Cheapest
  recovery -- prefer it as the first branch after a failed `verify_grasp`.
- `retract_retry(keypoint_label)`: the approach was blocked. Lift, jitter XY,
  re-approach.
- `adjust_grip(...)`: the object was gripped but slipped. Re-closes at a
  different width.
- `compliant_push(...)`: force-guided descent when a plain move would collide.
Do NOT invent a skill name. If none of the above fits, use a second branch built
from ordinary primitives, or emit no fallback at all.

Multi-object plans: wrap EACH object's acquire in its own `retry`. If one
object's acquire exhausts its attempts, the executor skips that object's
transport and place and moves on to the next object rather than placing air.
"""

    @classmethod
    def _recovery_prompt_section(cls) -> str:
        """Recovery grammar, with the caps read from the enforcing module.

        Interpolated rather than hardcoded so a change to bt_grammar's caps
        cannot leave the prompt advertising attempt counts the executor will
        refuse to take.
        """
        return cls._RECOVERY_PROMPT_SECTION_TEMPLATE.format(
            max_branches=bt_grammar.MAX_SELECTOR_BRANCHES,
            max_retries=bt_grammar.MAX_RETRY_ATTEMPTS,
            max_budget=bt_grammar.MAX_RECOVERY_ATTEMPTS_PER_SCORE,
        )

    def _build_system_prompt(self, robointer: bool = False) -> str:
        """
        Build the full system prompt, injecting registered skills from
        the registry. When `robot_family == "bimanual_franka"` the
        bimanual prompt section is appended so the LLM knows about the
        arm-tagged leaves, the `parallel` node, and the handoff primitive.

        ``robointer=True`` appends the optional RoboInter spatial-annotation
        schema. It is off by default and gated by the caller
        (``planning.robointer.enabled`` / $SPARK_ROBOINTER).
        """
        primitives_section = skill_registry.get_prompt_section()
        base = self._SYSTEM_PROMPT_HEADER.format(primitives_section=primitives_section)
        if self.robot_family == "ur10e":
            base = base + self._UR10E_PROMPT_SECTION
        if self.robot_family == "bimanual_franka":
            base = base + self._BIMANUAL_PROMPT_SECTION
        base = base + self._POUR_PROMPT_SECTION
        base = base + self._GRASP_STRATEGY_PROMPT_SECTION
        base = base + self._VERIFY_PROMPT_SECTION
        base = base + self._recovery_prompt_section()
        if robointer:
            base = base + ROBOINTER_PROMPT_SECTION
        return base

    _PROMPT_GENERATION_PROMPT = """You are a robotic perception assistant. Given a scene image and a task instruction, output the object detection prompts that a segmentation model (SAM3) should use to find the relevant objects.

Rules:
- Output a JSON list of short, specific object names: ["fork", "knife", "spoon", "tray"]
- Each prompt should describe ONE type of object (SAM3 handles multi-instance automatically)
- Include BOTH the objects to manipulate AND the target locations (containers, surfaces, etc.)
- Use simple, visual nouns - not abstract concepts. "red block" not "the thing on the left"
- CRITICAL: SAM3 grounds VISUAL appearance, not proper nouns. If the instruction names an object by a proper noun / character / brand name a segmentation model will NOT recognize (e.g. "dexter plushie", "waddle dee plushie", "pikachu toy"), DO NOT pass the name to SAM3. Use the generic visual category ("plushie", "toy"), and when two objects of the SAME category must be told apart, use a distinguishing VISIBLE-APPEARANCE descriptor instead (color/feature): e.g. instruction "swap the dexter and waddle dee plushies" in a scene with a brown one and a purple one -> ["brown plushie", "purple plushie", "bowl"]. The downstream planner sees the annotated image and binds the instruction's proper nouns to the labeled instances, so you only need SAM3 to find and separate them.
- For utensils, tools, and elongated objects that need to be grasped by the handle, emit BOTH the sub-part AND the whole object: "knife handle" AND "knife", "screwdriver handle" AND "screwdriver". SAM3 detects sub-parts from natural language, and the two masks answer two different questions. The sub-part says where to close the jaws. The WHOLE object says which way the thing points -- a handle on its own is symmetric end-for-end, so it fits a shaped slot either way round and the executor cannot tell a screwdriver's tip from its grip. Measured 2026-08-19: matching a handle mask (ar 2.90) to a screwdriver-shaped cutout scored IoU 0.455 with a margin of 0.002, i.e. the reversed orientation fitted just as well, and the tool went in backwards. Keep the full object name for place targets and non-graspable objects.
- For categories like "silverware" or "dishes", break them into the specific items you can see in the image
- For tasks involving reading/math/symbols, include prompts for each visible symbol or object
- Keep prompts under 4 words each
- NEVER include the table surface, mat, floor, or background as a prompt. The black mat IS the table, it is not an object to manipulate or a target location.
- For FOLDING tasks on cloth/garments, expand to the sub-parts that the robot has to address as separate grasp keypoints, do NOT return only the parent garment name. The downstream planner pinches one sub-part at a time, so each sub-part MUST be its own SAM3 prompt. Use natural-language part names; SAM3 grounds them directly against the cloth mask.
  - "fold the t-shirt" / "fold the shirt": ["shirt left sleeve", "shirt right sleeve", "shirt hem", "shirt collar", "shirt body"]
  - "fold the towel": ["towel left edge", "towel right edge", "towel top edge", "towel bottom edge", "towel"]
  - "fold the pants": ["pants left leg", "pants right leg", "pants waistband", "pants"]
  - "fold the napkin": ["napkin corner", "napkin edge", "napkin"]
  - Generic fold of an unnamed cloth: ["<cloth-name> corner", "<cloth-name> edge", "<cloth-name>"]
- Output ONLY the JSON list, nothing else
"""


    def assess_completion(self, instruction, images, evidence=""):
        """Adjudicate a disputed run: is the task ALREADY done in these images?

        The terminal recovery rung calls this BEFORE replanning, on CLEAN
        (unannotated) frames, because overlay paint sits on the objects whose
        state is in dispute. It is a second opinion against verifier false
        negatives.

        ``images`` is a list of (camera_name, HxWx3 rgb). Returns
        {"complete": bool, "confidence": float 0..1, "reason": str};
        on any error returns complete=False, confidence=0.0 (never blocks
        the replan path).
        """
        try:
            client = self._get_client()
            prompt = (
                "You are inspecting a robot workspace AFTER a manipulation "
                "attempt. Decide whether the task is ALREADY COMPLETE as "
                "photographed.\n"
                f"Task: {instruction}\n"
                + (f"The automated verifier reported: {evidence}\n" if evidence else "")
                + "Automated verifiers here have known FALSE NEGATIVES (a "
                "correctly seated tool can be reported as a failure), so "
                "judge only from the images. Answer with STRICT JSON: "
                '{"complete": true|false, "confidence": 0.0-1.0, '
                '"reason": "<one sentence naming the visual evidence>"}. '
                "Set complete=true only when the images clearly show the "
                "goal state; when unsure, complete=false with low confidence."
            )
            contents = [prompt]
            for name, img in images or []:
                contents.append(f"Camera '{name}':")
                pil = (
                    img if isinstance(img, _PILImage.Image)
                    else _PILImage.fromarray(img)
                )
                # Same 1024-edge cap as every other planner call: byte size,
                # not image count (3,600/request allowed), is what exceeds the
                # deadline.
                contents.append(_downscale_for_planner(pil))
            text = self._gemini_generate_raw(client, contents)
            match = re.search(r"\{.*\}", text, re.DOTALL)
            out = json.loads(match.group()) if match else {}
            return {
                "complete": bool(out.get("complete", False)),
                "confidence": float(out.get("confidence", 0.0) or 0.0),
                "reason": str(out.get("reason", ""))[:300],
            }
        except Exception as exc:  # noqa: BLE001 - adjudication is best-effort
            logger.warning("[adjudicate] assess_completion failed: %s", exc)
            return {"complete": False, "confidence": 0.0, "reason": f"error: {exc}"}

    def generate_prompts(self, instruction: str, scene_image=None) -> list:
        """
        Phase 1: Ask Gemini to look at the scene and decide what to detect.

        For simple instructions like "pick up the red block", the regex
        extractor works fine.  This method is for complex/ambiguous tasks
        where Gemini needs to see the scene to decide what objects to find.

        Returns:
            List of prompt strings for SAM3 detection.
        """
        client = self._get_client()

        context = f"Instruction: {instruction}\n"

        if self.llm_backend == "gemini":
            if scene_image is not None:
                contents = [
                    self._PROMPT_GENERATION_PROMPT + "\n\n" + context,
                    (
                        scene_image
                        if isinstance(scene_image, _PILImage.Image)
                        else _PILImage.fromarray(scene_image)
                    ),
                ]
            else:
                contents = self._PROMPT_GENERATION_PROMPT + "\n\n" + context

            response_text = self._gemini_generate_raw(client, contents)

            # Parse the JSON list from the response.
            match = re.search(r"\[.*?\]", response_text, re.DOTALL)
            if match:
                try:
                    prompts = json.loads(match.group())
                    if isinstance(prompts, list) and all(
                        isinstance(p, str) for p in prompts
                    ):
                        return prompts
                except json.JSONDecodeError:
                    pass

            # Fallback: split by commas/newlines
            return [
                p.strip().strip("\"'") for p in response_text.split(",") if p.strip()
            ]

        # Fallback for non-Gemini
        return [instruction]

    _ALT_PROMPT_PROPOSAL_PROMPT = """You are helping a robot's open-vocabulary segmentation model (SAM3) find ONE object it has already failed to find in this image.

You are given the object's canonical name and every text prompt that has already been tried and returned nothing (or returned a mask the system rejected as too low-confidence to be that object).

Look at the image and answer: what would SAM3 have to be asked to segment THIS object from THIS viewpoint?

Rules:
- Output a JSON list of short alternative prompts, best first: ["purple hat", "purple fabric bundle"]
- DO NOT repeat any prompt that was already tried, and do not return trivial variants of them (plural, "the ", adjective reordering) - those have effectively been tried too
- Describe what the object LOOKS LIKE from this camera, not what it is. A plushie seen from directly above with its face hidden may only read as a coloured hat or a fabric lump - say so
- Use simple visual nouns with at most one or two descriptive words (colour, material, shape). Under 4 words each
- SAM3 grounds visual appearance, not proper nouns: never output a character, brand or person name
- If a distinctive SUB-PART is easier to segment than the whole (an ear, a handle, a rim), propose that
- If the object genuinely is not visible in the image, output an empty list []
- Output ONLY the JSON list, nothing else
"""

    def propose_alt_prompts(
        self,
        target_label: str,
        tried_prompts,
        scene_image=None,
        max_prompts: int = 4,
    ) -> list:
        """
        Ask the LLM to name an object SAM3 could not find, given the frame.

        Last-resort rung of the perception escalation in
        ``PerceptionMixin.detect_for_task``, reached after the task's
        registered prompt and its ``alt_prompts`` both came back empty.

        Same client, key rotation and retry policy as :meth:`generate_prompts`.
        Returns a ranked list of new prompt strings, or ``[]``; the caller
        treats ``[]`` and an exception identically and falls through to
        ``fallback.on_mismatch``.
        """
        tried = [str(p) for p in (tried_prompts or []) if str(p).strip()]
        if self.llm_backend != "gemini":
            return []

        client = self._get_client()
        context = (
            f"Object to find: {target_label}\n"
            f"Prompts already tried and rejected: {json.dumps(tried)}\n"
            f"Return at most {int(max_prompts)} alternatives.\n"
        )
        if scene_image is not None:
            contents = [
                self._ALT_PROMPT_PROPOSAL_PROMPT + "\n\n" + context,
                (
                    scene_image
                    if isinstance(scene_image, _PILImage.Image)
                    else _PILImage.fromarray(scene_image)
                ),
            ]
        else:
            contents = self._ALT_PROMPT_PROPOSAL_PROMPT + "\n\n" + context

        response_text = self._gemini_generate_raw(client, contents)

        match = re.search(r"\[.*?\]", response_text or "", re.DOTALL)
        if not match:
            return []
        try:
            proposals = json.loads(match.group())
        except json.JSONDecodeError:
            return []
        if not isinstance(proposals, list):
            return []
        # The caller sanitises again (it must, since it also handles a raising
        # or absent planner), but a model that answers with numbers or dicts
        # should not travel any further than this.
        return [str(p).strip() for p in proposals if isinstance(p, str) and p.strip()]

    PHASE1_MODEL = "gemini-3.5-flash"  # Fast model for prompt generation (phase 1)

    def _gemini_generate_raw(self, client, contents) -> str:
        """
        Generate with Gemini flash and return raw text (no YAML extraction).

        Uses the fast flash model since phase 1 just outputs a JSON list of prompts.
        """
        try:
            response = client.models.generate_content(
                model=self.PHASE1_MODEL,
                contents=contents,
                config=self._generation_config(),
            )
            self._record_usage(response, "prompt_generation")
            return response.text
        except Exception as e:
            err_str = str(e).lower()
            if "429" in err_str or "rate" in err_str or "quota" in err_str:
                logger.warning("Rate limited on prompt generation, trying fallback...")
                all_keys = self._load_api_keys()
                for key in all_keys:
                    if key == self.api_key:
                        continue
                    try:
                        fallback_client = _make_genai_client(key)
                        response = fallback_client.models.generate_content(
                            model=self.model,
                            contents=contents,
                            config={"temperature": self.temperature},
                        )
                        return response.text
                    except Exception:
                        continue
            raise

    # Geometry fields rendered into the planner context when present. Absent
    # keys are skipped so an old caller's detection_details still works.
    _GEOMETRY_FIELDS = (
        ("aspect_ratio", "aspect_ratio={:.2f}"),
        ("obb_minor_mm", "obb_minor={:.0f}mm"),
        ("obb_angle_deg", "obb_angle={:.0f}deg"),
        ("obb_confidence", "obb_confidence={:.2f}"),
        # Mask-derived twin of the two above. Prefer these when the OBB is
        # flagged untrustworthy: they come from the silhouette, not the depth.
        ("mask_angle_deg", "mask_angle={:.0f}deg"),
        ("mask_aspect_ratio", "mask_ar={:.2f}"),
        ("container_depth_cm", "container_depth={:.1f}cm"),
        ("mask_area_frac", "mask_area={:.3%}"),
    )

    @classmethod
    def _format_geometry(cls, detail: dict) -> str:
        # ", key=value" fragments for the geometry fields present on a detection.
        parts = []
        for key, fmt in cls._GEOMETRY_FIELDS:
            value = detail.get(key)
            if value is not None:
                parts.append(fmt.format(value))
        if detail.get("camera"):
            parts.append(f"cam={detail['camera']}")
        if detail.get("low_quality"):
            parts.append("LOW_QUALITY_MASK")
        if detail.get("depth_stretched_the_obb"):
            parts.append("OBB_ANGLE_UNRELIABLE(use mask_angle)")
        if detail.get("mask_is_round"):
            parts.append("NO_MAJOR_AXIS")
        if detail.get("reprompt_attempts"):
            parts.append(f"reprompts={detail['reprompt_attempts']}")
        return (", " + ", ".join(parts)) if parts else ""

    @staticmethod
    def _downscale_for_planner(img):
        return _downscale_for_planner(img)

    def generate_score(
        self,
        instruction: str,
        annotated_image=None,
        keypoint_labels: list = None,
        detection_details: list = None,
        fewshot_examples: list = None,
        robointer_context: str = None,
        robointer_strict: bool = False,
        mask_overlay_image=None,  # accepted for bench-caller API parity; ignored
        wrist_image=None,         # accepted for bench-caller API parity; ignored
        temperature: float = None,
        scene_camera: str = None,
    ) -> dict:
        """
        Generate a SPARK score from instruction and visual context.

        Args:
            instruction: Natural language task description
            annotated_image: PIL Image with keypoint annotations (optional)
            keypoint_labels: List of detected keypoint labels
            robointer_context: When not None, the RoboInter extension is ON
                for this call: the schema section is appended to the system
                prompt, this string (image size + each detection's observed
                box, in the same 0..1000 space the answer must use) is
                appended to the user context, and the reply's `__robointer`
                blocks are schema-checked. None -> the pre-RoboInter prompt,
                unchanged. Built by planning.robointer_gate.planner_context.

        Returns:
            Parsed YAML score as dict
        """
        # Per-call temperature override: the bench's diversity-sampling
        # path (shadow best-of-N) requests temp > 0 per sample. Callers
        # there construct a fresh planner per call, so mutating the
        # instance default is contained.
        if temperature is not None:
            self.temperature = float(temperature)
        client = self._get_client()

        context = f"Instruction: {instruction}\n"
        if keypoint_labels:
            context += f"Available keypoints: {', '.join(keypoint_labels)}\n"
        if detection_details:
            context += (
                "Detection details (use these to judge which detections are real):\n"
            )
            for dd in detection_details:
                context += f"  - {dd['label']}: confidence={dd['confidence']:.0%}"
                if dd.get("position_3d"):
                    p = dd["position_3d"]
                    context += f", pos=({p[0]:.3f}, {p[1]:.3f}, {p[2]:.3f})"
                context += self._format_geometry(dd)
                context += "\n"
            context += (
                "Read confidence and obb_confidence together with the image: a "
                "high aspect_ratio on a low-confidence or low-obb_confidence "
                "mask is noise, not an elongated object.\n"
            )
            context += "IMPORTANT: Verify each detection label against what you see in the image. If a label is CLEARLY wrong (e.g., 'knife handle 2' is actually a spoon), add a 'label_corrections' field to the YAML mapping the wrong label to the correct one. Use the CORRECTED label in your plan's keypoint_label fields. Example:\n"
            context += "label_corrections:\n  knife handle 2: spoon handle\n"
            context += "Only correct labels when the object type is clearly wrong. Do NOT change color descriptions (e.g., don't rename 'yellow block' to 'orange block'), color perception varies and the original detection label is what the robot uses for tracking.\n"
            context += "Ignore detections that don't correspond to any real object in the image.\n"

        # BTLibrary few-shot: prior successful real-robot trials
        # (instruction, score) pairs.  Boosts generalisation across
        # similar instructions without retraining.  Wire-up is in
        # pipeline.py; safe to ignore when caller passes nothing.
        if fewshot_examples:
            context += "\nPrior successful plans for similar tasks:\n"
            for i, (ex_instr, ex_score) in enumerate(fewshot_examples, 1):
                context += f"\nExample {i} (task: {ex_instr}):\n"
                context += "```yaml\n"
                context += yaml.safe_dump(ex_score, sort_keys=False).strip()
                context += "\n```\n"
            context += "\nUse these as guides; emit a single new plan.\n"

        # RoboInter: the observed 2D geometry the answer's coordinates are
        # measured against. Appended LAST so it sits next to the image in the
        # payload, and skipped entirely when the extension is off.
        if robointer_context:
            context += "\n" + robointer_context

        system_prompt = self._build_system_prompt(robointer=robointer_context is not None)

        if self.llm_backend == "gemini":
            if annotated_image is not None:
                # Multimodal: image(s) + text. annotated_image may be ONE image
                # or a list of (camera_name, image) pairs; a second view
                # corrects the foreshortening of elongated objects in a single
                # fixed camera. The text names the images in order so the
                # model can refer to each.
                views = annotated_image
                if not isinstance(views, (list, tuple)):
                    views = [(scene_camera or "scene", views)]
                elif views and not isinstance(views[0], (list, tuple)):
                    views = [(scene_camera or "scene", v) for v in views]

                header = system_prompt + "\n\n" + context
                if len(views) > 1:
                    header += (
                        "\n\nYou are given " + str(len(views)) + " camera views of "
                        "the SAME scene, in this order: "
                        + ", ".join(str(n) for n, _ in views)
                        + ". They show one scene from different angles, not "
                        "different scenes. Cross-check between them: an object "
                        "that looks round or short in one view is often "
                        "foreshortened there, and the other view shows its true "
                        "extent. Where they disagree, prefer the measured "
                        "geometry in the detection list over either picture.\n"
                    )
                contents = [header]
                for _name, _img in views:
                    if _img is None:
                        continue
                    _p = (
                        _img
                        if isinstance(_img, _PILImage.Image)
                        else _PILImage.fromarray(_img)
                    )
                    contents.append(_downscale_for_planner(_p))
            else:
                contents = system_prompt + "\n\n" + context
            yaml_text = self._gemini_generate(client, contents)
        elif self.llm_backend == "openai":
            messages = [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": context},
            ]
            response = client.chat.completions.create(model="gpt-4o", messages=messages)
            yaml_text = self._extract_yaml(response.choices[0].message.content)
        else:
            raise PlanSchemaError(f"unknown llm_backend {self.llm_backend!r}")

        return self.parse_response(
            yaml_text,
            keypoint_labels,
            robointer=robointer_context is not None,
            robointer_strict=robointer_strict,
        )

    @staticmethod
    def parse_response(
        yaml_text: str,
        keypoint_labels: list = None,
        robointer: bool = False,
        robointer_strict: bool = False,
    ) -> dict:
        """
        Parse + schema-check one raw LLM response. No network; safe to call
        on a fabricated payload in tests.

        ``robointer=True`` additionally sanitizes the reply's `__robointer`
        blocks: every malformed FIELD is logged and dropped so no bad
        coordinate reaches the consumer. Off by default.
        """
        score = parse_plan_yaml(yaml_text)
        clean, _issues = sanitize_score(score, keypoint_labels)
        if robointer:
            clean, _ri_issues = sanitize_plan_annotations(
                clean, keypoint_labels, strict=robointer_strict
            )
        return clean

    FALLBACK_MODEL = "gemini-3.5-flash"

    def _gemini_generate(self, client, contents: str) -> str:
        # Generate with Gemini, falling back to thinking model on rate limit.
        try:
            response = client.models.generate_content(
                model=self.model,
                contents=contents,
                config=self._generation_config(),
            )
            self._record_usage(response, "bt_generation")
            return self._extract_yaml(response.text)
        except Exception as e:
            err_str = str(e).lower()
            # A DEADLINE on a multi-image request is usually the payload, not
            # the model. Retry ONCE with the primary view only: two views are
            # a cross-check, not a requirement.
            if (
                ("504" in err_str or "deadline" in err_str)
                and isinstance(contents, list)
                and sum(1 for c in contents if not isinstance(c, str)) > 1
            ):
                trimmed, kept_one = [], False
                for c in contents:
                    if isinstance(c, str):
                        trimmed.append(c)
                    elif not kept_one:
                        trimmed.append(c)
                        kept_one = True
                logger.warning(
                    "Planner deadline with %d images; retrying with 1 image",
                    sum(1 for c in contents if not isinstance(c, str)),
                )
                try:
                    response = client.models.generate_content(
                        model=self.model,
                        contents=trimmed,
                        config=self._generation_config(),
                    )
                    self._record_usage(response, "bt_generation")
                    return self._extract_yaml(response.text)
                except Exception as e2:  # noqa: BLE001
                    logger.warning("Single-image retry also failed: %s", e2)
                    err_str = str(e2).lower()
            if (
                "429" in err_str
                or "rate" in err_str
                or "quota" in err_str
                or "resource" in err_str
            ):
                logger.warning("Rate limited on %s, trying fallback key...", self.model)
            else:
                raise

        # Try other keys with primary model
        all_keys = self._load_api_keys()
        for key in all_keys:
            if key == self.api_key:
                continue
            try:
                fallback_client = _make_genai_client(key)
                response = fallback_client.models.generate_content(
                    model=self.model,
                    contents=contents,
                    config={"temperature": self.temperature},
                )
                logger.info("Succeeded with alternate key on %s", self.model)
                return self._extract_yaml(response.text)
            except Exception:
                continue

        # Fall back to thinking model with all keys
        logger.warning(
            "All keys exhausted on %s, falling back to %s",
            self.model,
            self.FALLBACK_MODEL,
        )
        for key in all_keys:
            try:
                fallback_client = _make_genai_client(key)
                response = fallback_client.models.generate_content(
                    model=self.FALLBACK_MODEL,
                    contents=contents,
                    # Typed ThinkingConfig: the dict form does not bind (see
                    # _generation_config).
                    config={
                        "temperature": self.temperature,
                        "thinking_config": _genai_types.ThinkingConfig(
                            thinking_budget=1024
                        ),
                    },
                )
                logger.info("Succeeded with %s", self.FALLBACK_MODEL)
                return self._extract_yaml(response.text)
            except Exception:
                continue

        raise RuntimeError("All Gemini keys and models exhausted")

    def _extract_yaml(self, text: str) -> str:
        # Extract YAML from LLM response (may be wrapped in markdown code block).
        if "```yaml" in text:
            start = text.index("```yaml") + 7
            end = text.index("```", start)
            return text[start:end].strip()
        if "```" in text:
            start = text.index("```") + 3
            end = text.index("```", start)
            return text[start:end].strip()
        # No code block: drop the prose preamble, starting at `task:` when
        # the model emitted one.
        return _repair_yaml_text(text)

    def validate_score(self, score: dict, keypoint_labels: list = None) -> list:
        """
        Validate a SPARK score for correctness.

        Returns list of issues (empty if valid). Covers the optional
        extension fields too; an old-style plan adds no issues.
        """
        issues = []
        if "tree" not in score:
            issues.append("Missing 'tree' key")
            return issues

        tree = score["tree"]
        if tree.get("type") not in {"sequence"} | bt_grammar.CONTROL_FLOW_TYPES:
            issues.append("Root tree type should be 'sequence'")

        # Structural node types (no leaf semantics; the executor branches
        # on these). `selector`/`fallback`/`retry` are run by
        # ScoreExecutorCore._run_selector / _run_retry. `parallel` and
        # `sync_barrier` are bimanual-specific; the single-arm executor
        # flattens them to sequential.
        valid_types = (
            set(BUILTIN_PRIMITIVES)
            | {"sequence", "parallel", "sync_barrier"}
            | set(bt_grammar.CONTROL_FLOW_TYPES)
        )
        valid_types.update(skill_registry.names())

        # Recurse one level so the parallel children are also checked.
        def _scan(node, path=""):
            if not isinstance(node, dict):
                return
            t = node.get("type")
            if t is not None and t not in valid_types:
                issues.append(f"{path or 'root'}: unknown type '{t}'")
            for i, child in enumerate(node.get("children", []) or []):
                _scan(child, f"{path}/{i}" if path else f"child[{i}]")

        for i, child in enumerate(tree.get("children", []) or []):
            _scan(child, f"child[{i}]")

        issues.extend(validate_plan_extensions(score, keypoint_labels))
        return issues

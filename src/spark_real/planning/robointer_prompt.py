"""
Prompt-side of the RoboInter extension: what Gemini is ASKED for, and what it
is SHOWN so it can answer.

Two halves, both additive:

  * :data:`ROBOINTER_PROMPT_SECTION` -- a system-prompt section, appended the
    same way ``_GRASP_STRATEGY_PROMPT_SECTION`` and ``_VERIFY_PROMPT_SECTION``
    already are. It teaches the schema, the coordinate space, and WHEN a
    geometric answer is better than a label.

  * :func:`build_robointer_context` -- extra user-side context: the image
    size and each detection's observed box in the SAME 0..1000 space the
    answer must use, so every number the model produces has a worked example
    beside it.

Text only. No network, no image processing (that is
``planning.robointer_annotate``).
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np

from spark_real.planning.robointer import (
    ANNOTATION_KEY,
    COORD_SCALE,
    MAX_TRACE_POINTS,
    PRIMITIVE_SKILLS,
)

ROBOINTER_PROMPT_SECTION = f"""

SPATIAL ANNOTATIONS (optional, one `{ANNOTATION_KEY}:` block per tree node):
You are shown the scene. Where a LABEL is ambiguous or a mask is untrustworthy,
say WHERE in the image, and the run stops depending on a threshold guessing for
you. Every field below is optional and every one of them is free to omit: a
node with no `{ANNOTATION_KEY}` behaves exactly as it does today.

Coordinates are INTEGERS from 0 to {int(COORD_SCALE)}, x first then y, measured on the
image you were given: x=0 is its left edge, x={int(COORD_SCALE)} its right edge, y=0 the top,
y={int(COORD_SCALE)} the bottom. The image size and every detection's observed box are given
to you in this same space, so read a coordinate off those before estimating.

```yaml
- type: move_to_keypoint
  params: {{ keypoint_label: "knife 1" }}
  {ANNOTATION_KEY}:
    subtask: "reach the knife's handle, not its blade"
    primitive_skill: pick
    label: "knife 1"          # which detected keypoint these coordinates are about
    camera: sideview          # the image you read them off; omit if only one was shown
    object_box: [[412, 300], [655, 372]]
    contact_point: [455, 337]
    trace: [[455, 337], [520, 300], [610, 250]]
```

- `object_box: [[x1,y1],[x2,y2]]` -- where the object actually is. Emit it when
  the detection's box looks wrong to you, when two instances of the same label
  could be confused, or when a mask covers two touching objects.
- `contact_point: [x,y]` -- the pixel the gripper should close on. This is the
  field that fixes the handle-vs-blade problem: for a knife, a spoon, a mug, a
  pan, the centroid of the mask is NOT where you grip. It must lie inside
  `object_box`. Emit it for anything with a handle or an obvious grip point.
- `placement_proposal: [[x1,y1],[x2,y2]]` -- the free region to release into.
  Emit it when a container is crowded, partly occluded, or already holds
  something you must not drop onto.
- `affordance_box: [[x1,y1],[x2,y2]]` -- where the open gripper jaws should sit
  at contact. Roughly the contact point widened along the grip direction.
- `trace: [[x,y], ...]` -- up to {MAX_TRACE_POINTS} waypoints the gripper should pass through,
  in order, starting at the contact point. Emit it ONLY to route around
  something: over a rim, around an obstacle, out of a drawer's swing. A
  straight-line move needs no trace.
- `state_affordance: [x, y, z, rx, ry, rz]` -- a full 6D end-effector pose in
  the ROBOT BASE FRAME, metres and rotation-vector radians. Only emit this if
  you were given base-frame coordinates precise enough to justify it; the 2D
  fields above are almost always the better answer.
- `primitive_skill` -- one of: {", ".join(PRIMITIVE_SKILLS)}.
- `subtask` -- one short sentence naming this node's goal in physical terms.

Rules:
- These annotations do NOT replace `params`. Keep `keypoint_label` correct;
  the annotation refines it.
- Do not invent coordinates for an object you cannot see. Omitting a field is
  always better than guessing one -- a wrong contact point moves the gripper to
  the wrong part of a real object.
- Read coordinates off the image, not off the label. If the detection box you
  were given and what you see disagree, emit `object_box` and say so in
  `subtask`.

REASONING TRACE (optional, one top-level `__fcot:` key, a list of short strings):
Before the tree, list the subtasks you decomposed the instruction into, one
line each, in order. Mention the geometry you used ("the tray's left half is
empty, so place there"). This is read by humans and logged with the episode;
it never changes execution.
"""


def _fmt_box_permille(box: Sequence[float], size: Tuple[int, int]) -> str:
    """Pixel (x1,y1,x2,y2) -> the 0..1000 string the model must answer in."""
    w, h = float(size[0]), float(size[1])
    x1, y1, x2, y2 = (float(v) for v in box)
    lo_x, hi_x = sorted((x1, x2))
    lo_y, hi_y = sorted((y1, y2))
    vals = (
        lo_x / w * COORD_SCALE,
        lo_y / h * COORD_SCALE,
        hi_x / w * COORD_SCALE,
        hi_y / h * COORD_SCALE,
    )
    a, b, c, d = (int(round(np.clip(v, 0.0, COORD_SCALE))) for v in vals)
    return f"[[{a}, {b}], [{c}, {d}]]"


def _fmt_point_permille(pt: Sequence[float], size: Tuple[int, int]) -> str:
    w, h = float(size[0]), float(size[1])
    x = int(round(np.clip(float(pt[0]) / w * COORD_SCALE, 0.0, COORD_SCALE)))
    y = int(round(np.clip(float(pt[1]) / h * COORD_SCALE, 0.0, COORD_SCALE)))
    return f"[{x}, {y}]"


def build_robointer_context(
    detections: Optional[Sequence[Any]] = None,
    image_size: Optional[Tuple[int, int]] = None,
    camera: Optional[str] = None,
) -> str:
    """User-side context block: image size + observed 2D geometry per detection.

    ``detections`` are duck-typed the same way ``build_detection_details`` and
    ``annotate_for_planner`` do it, so an ``ObjectDetection``, a dict-like or a
    test fake all work. Detections from a camera other than the annotated one
    are skipped: their pixel coordinates refer to a different image.

    Returns "" when there is nothing useful to say, so a caller can append it
    unconditionally.
    """
    if image_size is None:
        for det in detections or []:
            mask = getattr(det, "mask", None)
            if mask is not None and getattr(mask, "ndim", 0) == 2:
                image_size = (int(mask.shape[1]), int(mask.shape[0]))
                break
    if image_size is None:
        return ""

    lines: List[str] = [
        f"Scene image: {int(image_size[0])}x{int(image_size[1])} px"
        + (f" ({camera})" if camera else "")
        + f", addressed as 0..{int(COORD_SCALE)} on both axes.",
    ]

    rows: List[str] = []
    for det in detections or []:
        det_cam = getattr(det, "camera", None)
        if camera and det_cam and det_cam != camera:
            continue
        label = getattr(det, "label", None)
        if not label:
            continue
        bbox = getattr(det, "bbox", None)
        if bbox is None or len(tuple(bbox)) != 4:
            continue
        row = f"  - {label}: observed_box={_fmt_box_permille(bbox, image_size)}"
        centroid = getattr(det, "centroid_2d", None)
        if centroid is not None and len(tuple(centroid)) == 2:
            row += f", mask_centroid={_fmt_point_permille(centroid, image_size)}"
        rows.append(row)

    if rows:
        lines.append(
            "Observed 2D geometry, in that same 0..%d space (this is where the "
            "detector thinks each object is -- correct it if you disagree):" % int(COORD_SCALE)
        )
        lines.extend(rows)
    return "\n".join(lines) + "\n"


def robointer_fewshot() -> Tuple[str, Dict[str, Any]]:
    """One worked (instruction, score) pair for ``fewshot_examples``.

    Hand-written, not harvested: the cached library predates the schema, so
    without this the model has no in-context example of a filled-in block
    and tends to emit either nothing or a free-form paraphrase of the keys.
    """
    score: Dict[str, Any] = {
        "task": "put the knife in the tray",
        "__fcot": [
            "the knife lies handle-left, blade-right; grip the handle",
            "the tray's right half already holds a spoon, so place left",
        ],
        "tree": {
            "type": "sequence",
            "children": [
                {
                    "type": "move_to_keypoint",
                    "params": {"keypoint_label": "knife 1", "offset_z": 0},
                    ANNOTATION_KEY: {
                        "subtask": "reach the knife's handle",
                        "primitive_skill": "pick",
                        "label": "knife 1",
                        "object_box": [[380, 470], [700, 545]],
                        "contact_point": [425, 508],
                    },
                },
                {"type": "grasp", "params": {"force": 60, "target_width": 0.01}},
                {"type": "move_relative", "params": {"dx": 0, "dy": 0, "dz": 0.2}},
                {
                    "type": "move_to_keypoint",
                    "params": {"keypoint_label": "tray", "offset_z": 0.1},
                    ANNOTATION_KEY: {
                        "subtask": "release over the empty left half of the tray",
                        "primitive_skill": "place",
                        "label": "tray",
                        "placement_proposal": [[120, 300], [300, 430]],
                    },
                },
                {"type": "release"},
            ],
        },
    }
    return ("put the knife in the tray", score)

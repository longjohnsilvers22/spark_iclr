#!/usr/bin/env python3
"""ReKep/MetaFold-style fold keypoints + constraint steps for a t-shirt fold.

Geometry/annotation layer only. This assembles a structured set of named
fold KEYPOINTS (each with a 3D position in both arm frames and a role) and
an ordered list of relational CONSTRAINT STEPS from the saved detections,
inspired by ReKep's relational-keypoint-constraint representation and
MetaFold's per-stage folding actions. No torch / VLM / model weights are
required at runtime.

Sources, in priority order:
  1. ~/.spark_real/detections/keypoints_annotated.json  (human override)
  2. ~/.spark_real/detections/detections.json           (auto detection)

Usage:
    from fold_keypoints import load_fold_keypoints
    fk = load_fold_keypoints()
    for kp in fk["keypoints"]:
        print(kp["name"], kp["role"], kp["right_frame"], kp["left_frame"])
    for step in fk["steps"]:
        print(step["stage"], step["description"])
"""
import json
import os

DET_DIR = os.path.expanduser("~/.spark_real/detections")
DET_PATH = os.path.join(DET_DIR, "detections.json")
ANNOT_PATH = os.path.join(DET_DIR, "keypoints_annotated.json")

# Canonical fold keypoints. Each is (name, role). The role tells the fold
# executor what the point is for, in the spirit of ReKep grasp/align/target
# keypoints and MetaFold pick/place anchors.
KEYPOINT_SPEC = [
    ("collar_left", "grasp"),
    ("collar_right", "grasp"),
    ("hem_left", "grasp"),
    ("hem_right", "grasp"),
    ("right_sleeve_tip", "grasp"),
    ("left_sleeve_tip", "grasp"),
    ("garment_center", "reference"),
]


def _as_xyz(v):
    """
    Coerce a value to a 3-list of floats, or None if not usable.
    """
    if v is None:
        return None
    try:
        out = [float(v[0]), float(v[1]), float(v[2])]
    except (TypeError, ValueError, IndexError):
        return None
    return out


def _midpoint(a, b):
    """
    Elementwise mean of two xyz lists, or whichever is available.
    """
    a, b = _as_xyz(a), _as_xyz(b)
    if a is None:
        return b
    if b is None:
        return a
    return [(a[i] + b[i]) / 2.0 for i in range(3)]


def _make_kp(name, role, right_frame=None, left_frame=None, source="auto", pixel=None):
    """
    Build a single keypoint record with both arm frames.
    """
    return {
        "name": name,
        "role": role,
        "right_frame": _as_xyz(right_frame),
        "left_frame": _as_xyz(left_frame),
        "pixel": list(pixel) if pixel is not None else None,
        "source": source,
    }


def _index_detections(detections):
    """
    Map label -> detection entry for quick lookup.
    """
    out = {}
    for entry in detections:
        if isinstance(entry, dict) and "label" in entry:
            out[entry["label"]] = entry
    return out


def _collar_points(shirt, hem):
    """Resolve collar-left / collar-right in both frames.

    Prefers detect.py's compute_collar_edges output (collar_left_*,
    collar_right_*) if present, else falls back to the shirt-top band
    endpoints (shirt_top_*), which mark the collar region in detect.py.
    """
    cl_r = cl_l = cr_r = cr_l = None
    if shirt is not None:
        cl_r = shirt.get("collar_left_right") or shirt.get("shirt_top_right")
        cl_l = shirt.get("collar_left_left") or shirt.get("shirt_top_left")
        cr_r = shirt.get("collar_right_right") or shirt.get("shirt_top_right")
        cr_l = shirt.get("collar_right_left") or shirt.get("shirt_top_left")
    return cl_r, cl_l, cr_r, cr_l


def _build_steps(have):
    """Ordered ReKep/MetaFold-style constraint stages for a t-shirt fold.

    `have` is a set of keypoint names with a usable position. Steps whose
    required keypoints are missing are still emitted but flagged ready=False
    so a caller can skip or request annotation for them.
    """
    raw = [
        {
            "stage": 1,
            "name": "grasp_sleeves",
            "description": "RIGHT arm grasps left_sleeve_tip, LEFT arm grasps right_sleeve_tip",
            "grasp": {"right": "left_sleeve_tip", "left": "right_sleeve_tip"},
            "requires": ["left_sleeve_tip", "right_sleeve_tip"],
        },
        {
            "stage": 2,
            "name": "fold_sleeves_to_center",
            "description": "each arm drapes its sleeve tip toward garment_center, then releases",
            "target": "garment_center",
            "requires": ["left_sleeve_tip", "right_sleeve_tip", "garment_center"],
        },
        {
            "stage": 3,
            "name": "grasp_hem",
            "description": "RIGHT arm grasps hem_left, LEFT arm grasps hem_right",
            "grasp": {"right": "hem_left", "left": "hem_right"},
            "requires": ["hem_left", "hem_right"],
        },
        {
            "stage": 4,
            "name": "fold_hem_to_collar",
            "description": "both arms arc hem_left/hem_right up and over toward collar_left/collar_right, then release",
            "target": {"right": "collar_left", "left": "collar_right"},
            "requires": ["hem_left", "hem_right", "collar_left", "collar_right"],
        },
    ]
    for step in raw:
        step["ready"] = all(r in have for r in step["requires"])
    return raw


def _build_from_annotation(annot):
    """
    Build keypoints from keypoints_annotated.json (human override).
    """
    kp_in = annot.get("keypoints", {}) if isinstance(annot, dict) else {}
    keypoints = []
    by_name = {}
    for name, role in KEYPOINT_SPEC:
        rec = kp_in.get(name)
        if rec is None:
            kp = _make_kp(name, role, source="annotated")
        else:
            kp = _make_kp(
                name, role,
                right_frame=rec.get("right_frame"),
                left_frame=rec.get("left_frame"),
                source="annotated",
                pixel=rec.get("pixel"),
            )
        keypoints.append(kp)
        by_name[name] = kp

    center = by_name.get("garment_center")
    if center is not None and center["right_frame"] is None:
        center["right_frame"] = _midpoint(
            by_name["collar_left"]["right_frame"], by_name["hem_left"]["right_frame"])
        center["left_frame"] = _midpoint(
            by_name["collar_right"]["left_frame"], by_name["hem_right"]["left_frame"])
    return keypoints, by_name


def _build_from_detections(detections):
    """
    Build keypoints from the auto detect.py detections.json.
    """
    by_label = _index_detections(detections)
    left_sleeve = by_label.get("left sleeve")
    right_sleeve = by_label.get("right sleeve")
    hem = by_label.get("hem")
    shirt = by_label.get("shirt")

    cl_r, cl_l, cr_r, cr_l = _collar_points(shirt, hem)

    # detect.py names hem edges in image space: left_edge = min image x.
    hem_left_r = hem.get("left_edge_right") if hem else None
    hem_left_l = hem.get("left_edge_left") if hem else None
    hem_right_r = hem.get("right_edge_right") if hem else None
    hem_right_l = hem.get("right_edge_left") if hem else None

    rst_r = left_sleeve.get("right_frame") if left_sleeve else None
    rst_l = left_sleeve.get("left_frame") if left_sleeve else None
    lst_r = right_sleeve.get("right_frame") if right_sleeve else None
    lst_l = right_sleeve.get("left_frame") if right_sleeve else None

    # garment center: prefer shirt centroid, else mean of sleeve tips.
    if shirt is not None:
        center_r = shirt.get("right_frame")
        center_l = shirt.get("left_frame")
    else:
        center_r = _midpoint(rst_r, lst_r)
        center_l = _midpoint(rst_l, lst_l)

    vals = {
        "collar_left": (cl_r, cl_l),
        "collar_right": (cr_r, cr_l),
        "hem_left": (hem_left_r, hem_left_l),
        "hem_right": (hem_right_r, hem_right_l),
        "right_sleeve_tip": (rst_r, rst_l),
        "left_sleeve_tip": (lst_r, lst_l),
        "garment_center": (center_r, center_l),
    }
    keypoints = []
    by_name = {}
    for name, role in KEYPOINT_SPEC:
        r_frame, l_frame = vals.get(name, (None, None))
        kp = _make_kp(name, role, right_frame=r_frame, left_frame=l_frame, source="auto")
        keypoints.append(kp)
        by_name[name] = kp
    return keypoints, by_name


def load_fold_keypoints(detections_path=DET_PATH, annotated_path=ANNOT_PATH):
    """Return a structured {keypoints, steps, source} dict for a t-shirt fold.

    Prefers keypoints_annotated.json when it exists (human override), else
    falls back to the auto detections.json. Missing fields are tolerated:
    any keypoint that cannot be resolved keeps right_frame/left_frame=None,
    and steps that depend on it are flagged ready=False.
    """
    source = "none"
    keypoints, by_name = [], {}

    if annotated_path and os.path.exists(annotated_path):
        try:
            annot = json.loads(open(annotated_path).read())
            keypoints, by_name = _build_from_annotation(annot)
            source = "annotated"
        except (ValueError, OSError):
            keypoints, by_name = [], {}

    if not keypoints and detections_path and os.path.exists(detections_path):
        try:
            detections = json.loads(open(detections_path).read())
            keypoints, by_name = _build_from_detections(detections)
            source = "auto"
        except (ValueError, OSError):
            keypoints, by_name = [], {}

    if not keypoints:
        for name, role in KEYPOINT_SPEC:
            kp = _make_kp(name, role, source="none")
            keypoints.append(kp)
            by_name[name] = kp

    have = {kp["name"] for kp in keypoints
            if kp["right_frame"] is not None or kp["left_frame"] is not None}
    steps = _build_steps(have)
    return {
        "source": source,
        "keypoints": keypoints,
        "by_name": by_name,
        "steps": steps,
        "missing": sorted(n for n, _ in KEYPOINT_SPEC if n not in have),
    }


def _summary(fk):
    """
    Human-readable dump for CLI use.
    """
    lines = [f"source: {fk['source']}", ""]
    lines.append(f"{'keypoint':<18}{'role':<11}{'right_frame':<30}{'left_frame'}")
    for kp in fk["keypoints"]:
        rf = kp["right_frame"]
        lf = kp["left_frame"]
        rf_s = "None" if rf is None else "(" + ", ".join(f"{x:.3f}" for x in rf) + ")"
        lf_s = "None" if lf is None else "(" + ", ".join(f"{x:.3f}" for x in lf) + ")"
        lines.append(f"{kp['name']:<18}{kp['role']:<11}{rf_s:<30}{lf_s}")
    lines.append("")
    lines.append("steps:")
    for step in fk["steps"]:
        flag = "ok" if step["ready"] else "NOT READY"
        lines.append(f"  [{step['stage']}] {step['name']} ({flag}): {step['description']}")
    if fk["missing"]:
        lines.append("")
        lines.append("missing keypoints: " + ", ".join(fk["missing"]))
    return "\n".join(lines)


def main():
    import argparse
    p = argparse.ArgumentParser(description="Assemble ReKep-style t-shirt fold keypoints")
    p.add_argument("--detections", default=DET_PATH)
    p.add_argument("--annotated", default=ANNOT_PATH)
    p.add_argument("--json", action="store_true", dest="json_out")
    args = p.parse_args()
    fk = load_fold_keypoints(args.detections, args.annotated)
    if args.json_out:
        out = {k: fk[k] for k in ("source", "keypoints", "steps", "missing")}
        print(json.dumps(out, indent=2))
    else:
        print(_summary(fk))


if __name__ == "__main__":
    main()

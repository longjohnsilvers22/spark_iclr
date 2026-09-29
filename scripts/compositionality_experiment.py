#!/usr/bin/env python3
"""Compositionality experiment: does Gemini discover composed behaviors from
only the 4 base primitives?

Strips the planner prompt down to {move_to_keypoint, move_relative,
grasp_se3, release} and asks it to perform tasks that normally have a
dedicated composed primitive (scrub, sweep, stack, wipe, push, drag).
Each task gets N trials; the emitted BT must (a) use ONLY the 4 base
primitives and (b) contain the structural signature of the correct motion.
"""
import sys, os, json, argparse
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "src")))

from spark_real.planning.spark_planner import SPARKPlanner

BASE_PROMPT = """You are a robot task planner. Given a natural language instruction and detected objects, output a YAML behavior tree using ONLY these four primitives:

- grasp_se3: close the gripper on an object. Params: keypoint_label (str), strategy ("top_down"), force (N), target_width (m)
- release: open the gripper. No params.
- move_to_keypoint: move the TCP above a detected object. Params: keypoint_label (str), offset_x/y/z (m)
- move_relative: move the TCP by a delta from its current pose. Params: dx, dy, dz (m)

These are the ONLY four primitives. There is NO scrub, wash, wipe, sweep, push, drag, stack, pour, or fold primitive. If a task needs such a motion you MUST build it from move_to_keypoint and move_relative steps (e.g. a wiping motion = a sequence of small move_relative deltas; a sweep = move_relative deltas in one direction through the object positions; a stack = move above target then move_relative dz down then release).

Output YAML only. No markdown fences, no prose. Schema:
task: <instruction>
tree:
  type: sequence
  children:
    - type: <primitive>
      params: {...}
"""

# Each task: instruction, detections, and a validator over the flat primitive list.
# Return a flat list of (type, params) leaves from any BT dict shape.
def _flatten(tree):
    out = []
    def rec(node):
        if not isinstance(node, dict):
            return
        # children container
        kids = node.get("children")
        t = node.get("type") or node.get("primitive") or node.get("behavior") or node.get("name")
        if kids:
            for k in kids:
                rec(k)
        elif t:
            params = node.get("params", {k: v for k, v in node.items()
                                          if k not in ("type", "primitive", "behavior", "name")})
            out.append((str(t), params))
    root = tree.get("tree", tree.get("behavior_tree", tree))
    rec(root)
    return out

BASE = {"move_to_keypoint", "move_relative", "grasp_se3", "release"}

def only_base(leaves):
    types = {t for t, _ in leaves if t.lower() not in ("sequence", "action")}
    return types.issubset(BASE), types - BASE

def _rels(leaves):
    return [p for t, p in leaves if t == "move_relative"]

# A scrub/wipe needs >=3 move_relative with sign changes in x or y (oscillation).
def v_scrub(leaves):
    rels = _rels(leaves)
    if len(rels) < 3:
        return False, f"only {len(rels)} move_relative"
    xs = [float(r.get("dx", 0) or 0) for r in rels]
    ys = [float(r.get("dy", 0) or 0) for r in rels]
    x_osc = any(a * b < 0 for a, b in zip(xs, xs[1:]))
    y_osc = any(a * b < 0 for a, b in zip(ys, ys[1:]))
    if x_osc or y_osc:
        return True, "oscillating raster"
    # rectangle pattern: alternating x/y nonzero
    nonzero_axes = [("x" if abs(x) > 1e-4 else "") + ("y" if abs(y) > 1e-4 else "")
                    for x, y in zip(xs, ys)]
    if len(set(a for a in nonzero_axes if a)) >= 2:
        return True, "multi-axis raster"
    return False, "no oscillation/raster pattern"

# A sweep: descend then >=2 move_relative predominantly one lateral direction.
def v_sweep(leaves):
    rels = _rels(leaves)
    lateral = [r for r in rels if abs(float(r.get("dx", 0) or 0)) > 1e-3
               or abs(float(r.get("dy", 0) or 0)) > 1e-3]
    if len(lateral) < 1:
        return False, "no lateral motion"
    # must contact then push: at least one downward move_rel or move_to with low offset
    descended = any(float(r.get("dz", 0) or 0) < -1e-3 for r in rels) or \
                any(t == "move_to_keypoint" and float(p.get("offset_z", 1) or 0) < 0.05
                    for t, p in leaves)
    return (len(lateral) >= 1 and descended), \
        f"lateral={len(lateral)} descended={descended}"

# Stack: grasp -> move_to target -> move_relative dz<0 -> release.
def v_stack(leaves):
    seq = [t for t, _ in leaves]
    has_grasp = "grasp_se3" in seq
    has_release = "release" in seq
    has_target = any(t == "move_to_keypoint" for t in seq)
    has_descend = any(t == "move_relative" and float(p.get("dz", 0) or 0) < -1e-3
                      for t, p in leaves)
    ok = has_grasp and has_release and has_target and has_descend
    return ok, f"grasp={has_grasp} target={has_target} descend={has_descend} release={has_release}"

# Push object to a goal: move behind/above object then move_relative toward goal.
def v_push(leaves):
    rels = _rels(leaves)
    lateral = [r for r in rels if abs(float(r.get("dx", 0) or 0)) > 1e-3
               or abs(float(r.get("dy", 0) or 0)) > 1e-3]
    has_approach = any(t == "move_to_keypoint" for t, _ in leaves)
    return (len(lateral) >= 1 and has_approach), \
        f"approach={has_approach} lateral={len(lateral)}"

def v_pickplace(leaves):
    seq = [t for t, _ in leaves]
    return ("grasp_se3" in seq and "release" in seq
            and seq.count("move_to_keypoint") >= 2), str(seq.count("move_to_keypoint"))

TASKS = [
    dict(name="scrub_plate",
         instruction="Pick up the sponge and scrub the plate clean, then set the sponge down",
         dets=[("sponge", [0.46, 0.18, -0.01]), ("plate", [0.45, -0.14, -0.02])],
         validate=v_scrub),
    dict(name="wipe_table",
         instruction="Pick up the cloth and wipe the table surface, then put the cloth back",
         dets=[("cloth", [0.40, 0.20, -0.01]), ("table", [0.45, 0.0, -0.03])],
         validate=v_scrub),
    dict(name="sweep_cubes",
         instruction="Use the brush to sweep the cube into the dustpan",
         dets=[("brush", [0.33, -0.20, 0.02]), ("cube", [0.45, 0.05, -0.01]),
               ("dustpan", [0.42, 0.35, -0.02])],
         validate=v_sweep),
    dict(name="stack_blocks",
         instruction="Pick up the red block and stack it on top of the blue block",
         dets=[("red block", [0.40, 0.15, 0.0]), ("blue block", [0.50, -0.10, 0.0])],
         validate=v_stack),
    dict(name="push_can",
         instruction="Push the can toward the goal marker",
         dets=[("can", [0.40, 0.10, 0.0]), ("goal marker", [0.55, -0.20, 0.0])],
         validate=v_push),
    dict(name="pick_place_mug",
         instruction="Pick up the mug and place it on the coaster",
         dets=[("mug", [0.45, 0.15, 0.02]), ("coaster", [0.40, -0.20, -0.01])],
         validate=v_pickplace),
]


def run(n_trials, temperature, model):
    # model=None -> planner uses its own DEFAULT_MODEL (gemini-3.5-flash)
    planner = SPARKPlanner(temperature=temperature, model=model)
    planner._build_system_prompt = lambda: BASE_PROMPT  # strip to 4 base primitives
    print(f"(planner model = {planner.model})")

    results = {}
    for task in TASKS:
        det_list = [{"label": l, "confidence": 0.9, "position_3d": p}
                    for l, p in task["dets"]]
        labels = [l for l, _ in task["dets"]]
        ok_base = 0
        ok_struct = 0
        details = []
        for trial in range(n_trials):
            try:
                score = planner.generate_score(
                    instruction=task["instruction"],
                    keypoint_labels=labels,
                    detection_details=det_list,
                )
            except Exception as e:
                details.append(f"  trial {trial}: PLAN ERROR {e}")
                continue
            if not score:
                details.append(f"  trial {trial}: empty")
                continue
            leaves = _flatten(score)
            base_ok, extra = only_base(leaves)
            struct_ok, why = task["validate"](leaves)
            if base_ok:
                ok_base += 1
            if base_ok and struct_ok:
                ok_struct += 1
            flag = "OK" if (base_ok and struct_ok) else ("CHEAT" if not base_ok else "WEAK")
            details.append(f"  trial {trial}: {flag} base={base_ok}({extra}) struct={struct_ok}({why}) n_leaves={len(leaves)}")
        results[task["name"]] = dict(base=ok_base, struct=ok_struct, n=n_trials, details=details)

    print(f"\nCOMPOSITIONALITY EXPERIMENT  (temp={temperature}, model={model}, n={n_trials})")
    tot_base = tot_struct = tot = 0
    for name, r in results.items():
        print(f"\n{name}:  base-only {r['base']}/{r['n']}   correct-composition {r['struct']}/{r['n']}")
        for d in r["details"]:
            print(d)
        tot_base += r["base"]; tot_struct += r["struct"]; tot += r["n"]
    print(f"\nTOTAL: base-only {tot_base}/{tot} ({100*tot_base/tot:.0f}%)   "
          f"correct-composition {tot_struct}/{tot} ({100*tot_struct/tot:.0f}%)")
    return results


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--trials", type=int, default=5)
    ap.add_argument("--temperature", type=float, default=0.5)
    ap.add_argument("--model", type=str, default=None,
                    help="Override model; default None uses planner DEFAULT_MODEL (gemini-3.5-flash)")
    ap.add_argument("--out", type=str, default="/tmp/compositionality_results.json")
    args = ap.parse_args()
    res = run(args.trials, args.temperature, args.model)
    json.dump({k: {kk: vv for kk, vv in v.items() if kk != "details"}
               for k, v in res.items()},
              open(args.out, "w"), indent=2)
    print(f"\nSaved summary to {args.out}")

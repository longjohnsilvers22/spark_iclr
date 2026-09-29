"""
Trace-guided debugging harness for LIBERO-PRO primitives (ASPIRE-style).

Runs one task with ground-truth object poses, no SAM3 and no Gemini, and
records a per-primitive trace so a failure can be attributed to a specific
primitive call rather than to "the task failed". This is the execution
evidence an agent needs to write a targeted repair, which is what
ASPIRE's robot execution engine provides and what the benchmark runner does
not: the runner reports a success bit per trial.

Usage:
    PYTHONPATH=src python -m spark_bench.dojo --suite libero_goal --task 6
"""
from __future__ import annotations

import argparse
import json
import numpy as np

from spark_bench.fair.config import FairConfig, load_libero_env
from spark_bench.libero_pro.executor import LiberoExecutor


class GTDet:
    """Minimal stand-in for a DetectionResult entry."""
    __slots__ = ('label', 'position_3d', 'confidence', 'mask', 'bbox')

    def __init__(self, label, pos):
        self.label = label
        self.position_3d = np.asarray(pos, dtype=float)
        self.confidence = 1.0
        self.mask = None
        self.bbox = None

    def __repr__(self):
        p = self.position_3d
        return f"GTDet({self.label}, [{p[0]:+.3f} {p[1]:+.3f} {p[2]:+.3f}])"


def gt_det_map(env) -> dict:
    """Ground-truth det_map keyed on the BDDL object names MuJoCo carries."""
    sim = env.sim
    model, data = sim.model, sim.data
    out = {}
    for i in range(model.nbody):
        name = model.body_id2name(i) if hasattr(model, 'body_id2name') else None
        if not name or name in ('world', 'table'):
            continue
        if name.startswith('robot') or name.startswith('gripper'):
            continue
        pos = np.array(data.body_xpos[i], dtype=float)
        # Strip LIBERO's trailing instance index so labels read like the
        # planner's ("akita_black_bowl_1" -> "akita_black_bowl").
        label = name[:-2] if name[-2] == '_' and name[-1].isdigit() else name
        label = label.replace('_main', '')
        if label not in out:
            out[label] = GTDet(label, pos)
    return out


class TracingExecutor(LiberoExecutor):
    """LiberoExecutor that records what every primitive did."""

    def __init__(self, *a, **kw):
        super().__init__(*a, **kw)
        self.trace: list[dict] = []

    def _snapshot(self) -> dict:
        d = self.env.sim.data
        return {
            'ee': np.array(self._ee(), dtype=float).round(4).tolist(),
            'objects': {k: np.array(v.position_3d).round(4).tolist()
                        for k, v in self.det_map.items()},
            'success': self._goal_satisfied(),
        }

    def _move_to(self, target, gripper_open, steps=200):
        before = np.array(self._ee(), dtype=float)
        err = super()._move_to(target, gripper_open, steps=steps)
        after = np.array(self._ee(), dtype=float)
        self.trace.append({
            'call': '_move_to',
            'target': np.asarray(target, dtype=float).round(4).tolist(),
            'ee_before': before.round(4).tolist(),
            'ee_after': after.round(4).tolist(),
            'residual_m': float(np.linalg.norm(after[:3] - np.asarray(target)[:3])),
            'reported': float(err) if err is not None else None,
            'gripper_open': bool(gripper_open),
        })
        return err

    def _gripper(self, open_gripper, steps=60):
        out = super()._gripper(open_gripper, steps=steps)
        self.trace.append({'call': '_gripper', 'open': bool(open_gripper),
                           'ee': np.array(self._ee(), dtype=float).round(4).tolist()})
        return out


def run(suite: str, task_id: int, bt: dict, *, init_state: int = 0,
        verbose: bool = True) -> dict:
    cfg = FairConfig()
    cfg.no_gemini = True
    env, task, init_states, bddl_path = load_libero_env(suite, task_id, cfg)
    env.reset()
    if init_states is not None and len(init_states) > init_state:
        env.set_init_state(init_states[init_state])
    for _ in range(20):
        env.step([0.0] * 6 + [-1.0])

    dm = gt_det_map(env)
    if verbose:
        print(f"[dojo] {suite} task {task_id} init {init_state}")
        for k, v in sorted(dm.items()):
            print(f"   {v}")

    ex = TracingExecutor(env, cfg, dm, instruction='')
    ex.run(bt)
    ok = bool(env.check_success())
    if verbose:
        print(f"\n[dojo] success={ok}  primitives traced={len(ex.trace)}")
        for i, t in enumerate(ex.trace):
            if t['call'] == '_move_to':
                print(f"  {i:2d} move_to  tgt={t['target'][:3]} "
                      f"got={t['ee_after'][:3]} residual={t['residual_m']*1000:.1f}mm")
            else:
                print(f"  {i:2d} gripper  open={t['open']}")
    return {'success': ok, 'trace': ex.trace, 'det_map': {k: v.position_3d.tolist()
                                                          for k, v in dm.items()}}


if __name__ == '__main__':
    ap = argparse.ArgumentParser()
    ap.add_argument('--suite', default='libero_goal')
    ap.add_argument('--task', type=int, default=6)
    ap.add_argument('--init', type=int, default=0)
    ap.add_argument('--bt', default=None, help='path to a YAML/JSON BT')
    args = ap.parse_args()
    bt = json.load(open(args.bt)) if args.bt else {'tree': {'type': 'sequence', 'children': []}}
    run(args.suite, args.task, bt, init_state=args.init)

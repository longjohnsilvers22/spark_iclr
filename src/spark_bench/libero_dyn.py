"""
LIBERO-Dyn: scripted mid-episode object displacement inside LIBERO.

Perturbation protocol for the adaptive-execution ladder (see
paper/ADAPTIVE_EXECUTION_PLAN.md section 2): the pick target is displaced
via direct qpos writes at fixed phases, with a magnitude sweep and a
direction table seeded per (task, trial) so schedules are IDENTICAL
across methods (HRIBench fixed-schedule pattern, arXiv 2607.13056;
Shift-x-Lag injection timing, arXiv 2510.02526; OmniContact phase
indexing, arXiv 2606.26201).

Phases
------
* ``POST_PLAN``       - immediately after planning, before execution.
* ``MID_APPROACH``    - when the EE has covered 50% of its initial
                        distance to the pick target (geometric definition
                        of "50% duration of the first move_to_keypoint";
                        method-agnostic, no executor introspection).
* ``POST_GRASP_PLAN`` - immediately before gripper closure (first close
                        command observed on the action channel).

Metrics per trial
-----------------
* ``success``            - env.check_success() (oracle, scoring only).
* ``recovery_used``      - any recovery attribution / replan fired.
* ``detection_latency``  - perturbation -> first SceneDiff flag (or
                           telemetry EMPTY/SLIP vote when the diff never
                           fired).
* ``adaptation_latency`` - flag -> resumed nominal execution (retarget
                           resume, or recovery re-execution start).

The protocol / schedule / metric helpers at the top are pure (numpy +
stdlib) and unit-testable without an env; the runner half lazily imports
the heavy fair-runner stack.

Usage (GPU machine)::

    MUJOCO_GL=egl PYTHONPATH=src/libero_pro:src/sam3:src \\
        python -m spark_bench.libero_dyn --suite spatial --task-id 0 \\
        --no-gemini --num-trials 1 --event-captures --scene-diff \\
        --layered-recovery
"""
from __future__ import annotations

import json
import time
import zlib
from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path
from typing import Callable, Optional

import numpy as np

from spark_real.perception.sticky_binding import fuzzy_key

# FairConfig import is heavy (torch / mujoco / libero); keep the protocol
# half importable in a bare CPU env for unit tests.
try:
    from spark_bench.fair.config import FairConfig as _FairBase
    _HAVE_FAIR = True
except Exception:  # pragma: no cover - unit-test environments
    _FairBase = object  # type: ignore[assignment, misc]
    _HAVE_FAIR = False


__all__ = [
    'Phase', 'DIRECTION_TABLE', 'schedule_seed', 'displacement_direction',
    'displacement_vector', 'PerturbationInjector', 'displace_body_qpos',
    'compute_latency_metrics', 'DynConfig', 'run_dyn', 'main',
]


class Phase(str, Enum):
    POST_PLAN = 'post_plan'
    MID_APPROACH = 'mid_approach'
    POST_GRASP_PLAN = 'post_grasp_plan'


# Fixed direction table: 8 compass unit vectors in the table plane.
_S = float(np.sqrt(0.5))
DIRECTION_TABLE: tuple = (
    (1.0, 0.0), (_S, _S), (0.0, 1.0), (-_S, _S),
    (-1.0, 0.0), (-_S, -_S), (0.0, -1.0), (_S, -_S),
)


def schedule_seed(task_name: str, trial: int) -> int:
    """
    Deterministic 32-bit seed per (task, trial).

    Uses crc32, NOT Python ``hash()`` (which is salted per process), so
    the schedule is identical across runs, machines, and methods.
    """
    return zlib.crc32(f'{task_name}|{int(trial)}'.encode('utf-8')) & 0xFFFFFFFF


def displacement_direction(task_name: str, trial: int) -> np.ndarray:
    """
    Unit XY displacement direction for this (task, trial) cell.

    Depends ONLY on (task, trial) - the magnitude sweep reuses the same
    direction so cells differ in exactly one variable.
    """
    idx = schedule_seed(task_name, trial) % len(DIRECTION_TABLE)
    return np.asarray(DIRECTION_TABLE[idx], dtype=float)


def displacement_vector(task_name: str, trial: int,
                          magnitude_cm: float) -> np.ndarray:
    """
    3-vector (dx, dy, 0) in meters for this (task, trial, magnitude).
    """
    d = displacement_direction(task_name, trial) * (float(magnitude_cm) / 100.0)
    return np.array([d[0], d[1], 0.0], dtype=float)


# Displacement via env.sim qpos write

_HINT_STOPWORDS = {'the', 'and', 'left', 'right', 'front', 'back', 'top'}


def _hint_tokens(hint: str) -> list[str]:
    # Digits are kept: LIBERO instances differ only by suffix
    # (akita_black_bowl_1 vs _2) and the schedule must hit the right one.
    return [t for t in hint.lower().replace('_', ' ').split()
            if (len(t) >= 3 or t.isdigit()) and t not in _HINT_STOPWORDS]


def find_free_joint_for_hint(model, hint: str, mj=None):
    """
    Locate the free joint whose body name best matches ``hint`` tokens.

    Returns ``(qposadr, body_name)`` or ``(None, None)``.  Pass ``mj``
    (the mujoco module) explicitly in tests; defaults to importing it.
    """
    if mj is None:
        import mujoco as mj  # type: ignore[no-redef]
    tokens = _hint_tokens(hint)
    if not tokens:
        return None, None
    best = (None, None)
    best_score = 0
    for jid in range(model.njnt):
        if int(model.jnt_type[jid]) != int(mj.mjtJoint.mjJNT_FREE):
            continue
        bid = int(model.jnt_bodyid[jid])
        bn = (mj.mj_id2name(model, mj.mjtObj.mjOBJ_BODY, bid) or '').lower()
        if not bn or 'robot' in bn or 'gripper' in bn:
            continue
        score = sum(1 for t in tokens if t in bn)
        # Prefer exact full-token coverage, then shorter names.
        if score > best_score or (score == best_score and score > 0
                                    and best[1] is not None
                                    and len(bn) < len(best[1])):
            best = (int(model.jnt_qposadr[jid]), bn)
            best_score = score
    return best if best_score > 0 else (None, None)


def displace_body_qpos(env, hint: str, delta: np.ndarray) -> Optional[str]:
    """
    Displace the free-jointed body matching ``hint`` by ``delta`` (m).

    Writes directly into ``env.sim`` qpos (position part of the free
    joint) and re-forwards the model.  Returns the displaced body name,
    or None when no matching body was found.
    """
    import mujoco
    model = env.sim.model._model
    data = env.sim.data._data
    adr, name = find_free_joint_for_hint(model, hint, mj=mujoco)
    if adr is None:
        return None
    d = np.asarray(delta, dtype=float).reshape(-1)
    data.qpos[adr] += d[0]
    data.qpos[adr + 1] += d[1]
    if len(d) > 2:
        data.qpos[adr + 2] += d[2]
    mujoco.mj_forward(model, data)
    try:
        env.sim.forward()
    except Exception:
        pass
    return name


# Collision-aware multi-object displacement

def _free_object_joints(model, mj):
    """All free-joint (qposadr, body_id, body_name) for non-robot bodies."""
    out = []
    for jid in range(model.njnt):
        if int(model.jnt_type[jid]) != int(mj.mjtJoint.mjJNT_FREE):
            continue
        bid = int(model.jnt_bodyid[jid])
        bn = (mj.mj_id2name(model, mj.mjtObj.mjOBJ_BODY, bid) or '').lower()
        if not bn or 'robot' in bn or 'gripper' in bn:
            continue
        out.append((int(model.jnt_qposadr[jid]), bid, bn))
    return out


def _body_object_contact(model, data, bid, other_bids, mj):
    """True if body ``bid`` currently penetrates any body in other_bids."""
    for i in range(int(data.ncon)):
        c = data.contact[i]
        if c.dist > -1e-4:      # only count real penetration
            continue
        b1 = int(model.geom_bodyid[c.geom1])
        b2 = int(model.geom_bodyid[c.geom2])
        if (b1 == bid and b2 in other_bids) or (b2 == bid and b1 in other_bids):
            return True
    return False


class MultiSlideController:
    """
    Continuous, collision-free displacement of one or more free objects.

    Each mover slides delta/steps per env step from its start_step.  After
    each increment MuJoCo's own contacts are checked; a step that would
    drive the body into ANOTHER object body is reverted (the object bumps
    and stops), so paths never interpenetrate.  Uses the simulator's
    collision detection directly, no external engine.
    """

    def __init__(self, env, movers, steps, mj=None):
        # movers: list of dict {hint, delta(3-vec), start_step}
        import mujoco as _mj
        self.mj = mj or _mj
        self.env = env
        self.model = env.sim.model._model
        self.data = env.sim.data._data
        self.steps = int(steps)
        self.tick = 0
        self.movers = []
        objs = _free_object_joints(self.model, self.mj)
        for m in movers:
            adr = bid = None
            toks = _hint_tokens(m['hint'])
            best = 0
            for a, b, bn in objs:
                sc = sum(1 for t in toks if t in bn)
                if sc > best:
                    best, adr, bid = sc, a, b
            if adr is None:
                continue
            self.movers.append({
                'adr': adr, 'bid': bid,
                'per_step': np.asarray(m['delta'], float).reshape(-1) /
                    max(self.steps, 1),
                'start': int(m.get('start_step', 0)),
                'done': 0, 'name': m['hint']})
        self._all_obj_bids = {b for _, b, _ in objs}

    def active(self):
        return any(mv['done'] < self.steps for mv in self.movers)

    def step(self):
        """Advance one env step; apply one increment per active mover."""
        moved = False
        for mv in self.movers:
            if self.tick < mv['start'] or mv['done'] >= self.steps:
                continue
            adr = mv['adr']
            before = self.data.qpos[adr:adr + 3].copy()
            d = mv['per_step']
            self.data.qpos[adr] += d[0]
            self.data.qpos[adr + 1] += d[1]
            if len(d) > 2:
                self.data.qpos[adr + 2] += d[2]
            self.mj.mj_forward(self.model, self.data)
            others = self._all_obj_bids - {mv['bid']}
            if _body_object_contact(self.model, self.data, mv['bid'],
                                     others, self.mj):
                # revert: this object bumped another; hold it here
                self.data.qpos[adr:adr + 3] = before
                self.mj.mj_forward(self.model, self.data)
                mv['done'] = self.steps    # stop this mover
            else:
                mv['done'] += 1
                moved = True
        if moved:
            try:
                self.env.sim.forward()
            except Exception:
                pass
        self.tick += 1
        return moved


def astar_grid(occ: np.ndarray, start: tuple, goal: tuple):
    """
    8-connected A* on a boolean occupancy grid (True = blocked).

    ``start`` and ``goal`` are (row, col).  Returns the list of cells from
    start to goal inclusive, or None when no path exists.  Pure numpy +
    heapq; diagonal moves cost sqrt(2) and may not cut a blocked corner.
    """
    import heapq
    h, w = occ.shape
    sr, sc = start
    gr, gc = goal
    if not (0 <= sr < h and 0 <= sc < w and 0 <= gr < h and 0 <= gc < w):
        return None
    if occ[gr, gc] or occ[sr, sc]:
        return None
    nbrs = ((-1, 0, 1.0), (1, 0, 1.0), (0, -1, 1.0), (0, 1, 1.0),
            (-1, -1, 1.4142135), (-1, 1, 1.4142135),
            (1, -1, 1.4142135), (1, 1, 1.4142135))

    def hcost(r, c):
        return float(np.hypot(r - gr, c - gc))

    g = {start: 0.0}
    came = {}
    heap = [(hcost(sr, sc), 0.0, start)]
    closed = set()
    while heap:
        _f, gc_, cur = heapq.heappop(heap)
        if cur in closed:
            continue
        if cur == goal:
            path = [cur]
            while cur in came:
                cur = came[cur]
                path.append(cur)
            path.reverse()
            return path
        closed.add(cur)
        r, c = cur
        for dr, dc, cost in nbrs:
            nr, nc = r + dr, c + dc
            if not (0 <= nr < h and 0 <= nc < w) or occ[nr, nc]:
                continue
            if dr and dc and (occ[r + dr, c] or occ[r, c + dc]):
                continue    # no corner cutting through a blocked cell
            ng = gc_ + cost
            nxt = (nr, nc)
            if ng < g.get(nxt, float('inf')):
                g[nxt] = ng
                came[nxt] = cur
                heapq.heappush(heap, (ng + hcost(nr, nc), ng, nxt))
    return None


class LiveMotionController:
    HOLD_STEPS = 8

    """
    Keep one free object moving at constant speed until the robot holds it.

    Each env step advances the body by speed*dt.  With ``planner='astar'``
    (default) the body follows A*-planned polylines between random free
    goals on a 1 cm occupancy grid built from the other objects and the
    named fixtures (footprints inflated by the mover's own radius), so it
    travels between things instead of bouncing in a box; a step that
    would still interpenetrate (MuJoCo's own contacts) is reverted, that
    cell is marked blocked, and the path is re-planned.  ``planner='box'``
    keeps the original reflect-in-a-box patrol, which is also the fallback
    when no path exists.  Motion stops for good on a real hold (both
    finger sides in contact with the aperture stalled) or a lift.
    """

    FIXTURE_KW = ('basket', 'bin', 'tray', 'plate', 'bowl', 'stove',
                  'drawer', 'cabinet')
    CELL_M = 0.01
    MIN_GOAL_M = 0.08

    def __init__(self, env, hint, direction, speed_m_s, radius_m,
                 dt_s=0.05, mj=None, planner: str = 'astar', seed: int = 0):
        import mujoco as _mj
        self.mj = mj or _mj
        self.env = env
        self.model = env.sim.model._model
        self.data = env.sim.data._data
        objs = _free_object_joints(self.model, self.mj)
        toks = _hint_tokens(hint)
        self.adr = self.bid = None
        best = 0
        for a, b, bn in objs:
            sc = sum(1 for t in toks if t in bn)
            if sc > best:
                best, self.adr, self.bid = sc, a, b
        self._others = {b for _a, b, _bn in objs if b != self.bid}
        self.dir = np.asarray(direction, float)[:2]
        n = float(np.linalg.norm(self.dir))
        self.dir = self.dir / n if n > 0 else np.array([1.0, 0.0])
        self.step_m = float(speed_m_s) * float(dt_s)
        self.radius = float(radius_m)
        self.start = (self.data.qpos[self.adr:self.adr + 3].copy()
                      if self.adr is not None else None)
        self.stopped = None      # reason string once motion ends
        self.path_m = 0.0
        self.n_steps = 0
        # Motion planner state ('astar': planned polylines between random
        # free goals through a 1 cm occupancy grid of the other objects;
        # 'box': reflect-in-a-box patrol).
        self.planner = str(planner)
        self.rng = np.random.RandomState(int(seed) & 0x7fffffff)
        self.replans = 0
        self._blocked_cells: set = set()   # cells that produced a contact
        self._grid = None        # (occ, origin_xy) from the last build
        self._path_pts: list = []          # world xy waypoints
        self._path_idx = 0
        self.r_self = (self._planar_radius(self.bid)
                       if self.bid is not None else 0.0)
        # A graze by the arm or palm does not stop a live object (a
        # conveyor does not pause because a finger brushed the item); only
        # a grasp does: both finger bodies in contact at once, or a lift.
        self._finger_adrs = None
        self._ap_hist = []
        self._touch_hist = []
        self._hold_run = 0
        # Panda gripper: leftfinger + finger_joint1_tip on one side,
        # rightfinger + finger_joint2_tip on the other; a hold needs both
        # SIDES in contact, not two bodies of the same finger.
        self._finger_side = {}
        for b in range(self.model.nbody):
            bn = (self.mj.mj_id2name(self.model, self.mj.mjtObj.mjOBJ_BODY, b)
                  or '').lower()
            if 'finger' not in bn:
                continue
            if 'left' in bn or 'joint1' in bn:
                self._finger_side[b] = 0
            elif 'right' in bn or 'joint2' in bn:
                self._finger_side[b] = 1
        self._finger_bids = set(self._finger_side)

    def _both_fingers_touching(self):
        touching = set()
        for i in range(int(self.data.ncon)):
            c = self.data.contact[i]
            b1 = int(self.model.geom_bodyid[c.geom1])
            b2 = int(self.model.geom_bodyid[c.geom2])
            if b1 == self.bid and b2 in self._finger_bids:
                touching.add(self._finger_side[b2])
            elif b2 == self.bid and b1 in self._finger_bids:
                touching.add(self._finger_side[b1])
        return len(touching) >= 2

    def _aperture(self):
        if self._finger_adrs is None:
            try:
                from spark_bench.libero_pro.telemetry import find_finger_qpos_addrs
                self._finger_adrs = list(find_finger_qpos_addrs(self.model))
            except Exception:
                self._finger_adrs = []
        if len(self._finger_adrs) < 2:
            return None
        a, b = self._finger_adrs[:2]
        return abs(float(self.data.qpos[a]) - float(self.data.qpos[b]))

    def _grasped(self):
        """
        A hold, not a brush: over a window of HOLD_STEPS steps, both finger
        sides touch the object on most steps and the aperture has stalled
        above fully-closed.  Windowed, not consecutive: the state-write
        drive flickers a held object's contacts.  Jaws closing past a
        moving object touch it briefly and then close to zero; that is an
        empty close and the object keeps moving.
        """
        touching = self._both_fingers_touching()
        self._touch_hist.append(bool(touching))
        if len(self._touch_hist) > self.HOLD_STEPS:
            del self._touch_hist[0]
        ap = self._aperture()
        if ap is not None:
            self._ap_hist.append(ap)
            if len(self._ap_hist) > self.HOLD_STEPS:
                del self._ap_hist[0]
        if len(self._touch_hist) < self.HOLD_STEPS:
            return False
        n_touch = sum(self._touch_hist)
        if ap is None:
            return n_touch >= self.HOLD_STEPS - 2
        stalled = (len(self._ap_hist) == self.HOLD_STEPS
                   and max(self._ap_hist) - min(self._ap_hist) < 0.001
                   and ap > 0.005)
        return n_touch >= self.HOLD_STEPS - 2 and stalled

    # Motion planning over the table plane

    def _geom_planar_radius(self, g):
        """Horizontal footprint radius of one geom (xy half-diagonal of its
        local AABB when available; the bounding sphere otherwise, which
        for a tall mesh overstates the footprint by its height)."""
        aabb = getattr(self.model, 'geom_aabb', None)
        if aabb is not None:
            hx, hy = float(aabb[g][3]), float(aabb[g][4])
            if hx > 0 or hy > 0:
                return float(np.hypot(hx, hy))
        return float(self.model.geom_rbound[g])

    def _planar_radius(self, bid):
        rs = [self._geom_planar_radius(g) for g in range(self.model.ngeom)
              if int(self.model.geom_bodyid[g]) == bid]
        return max(rs) if rs else 0.03

    def _table_extents(self):
        """(xmin, xmax, ymin, ymax) of the table top, else start +/- 0.20."""
        mj = self.mj
        box_t = int(mj.mjtGeom.mjGEOM_BOX)
        best = None
        for g in range(self.model.ngeom):
            bid = int(self.model.geom_bodyid[g])
            bn = (mj.mj_id2name(self.model, mj.mjtObj.mjOBJ_BODY, bid)
                  or '').lower()
            if 'table' not in bn or int(self.model.geom_type[g]) != box_t:
                continue
            sz = self.model.geom_size[g]
            c = self.data.geom_xpos[g]
            area = float(sz[0] * sz[1])
            if best is None or area > best[0]:
                best = (area, float(c[0] - sz[0]), float(c[0] + sz[0]),
                        float(c[1] - sz[1]), float(c[1] + sz[1]))
        if best is not None:
            return best[1:]
        sx, sy = float(self.start[0]), float(self.start[1])
        return (sx - 0.20, sx + 0.20, sy - 0.20, sy + 0.20)

    def _build_grid(self):
        """Occupancy grid (True = blocked) over the workspace rectangle."""
        hw = max(self.radius, 0.15)
        sx, sy = float(self.start[0]), float(self.start[1])
        txmin, txmax, tymin, tymax = self._table_extents()
        xmin, xmax = max(sx - hw, txmin), min(sx + hw, txmax)
        ymin, ymax = max(sy - hw, tymin), min(sy + hw, tymax)
        if xmax - xmin < 4 * self.CELL_M or ymax - ymin < 4 * self.CELL_M:
            xmin, xmax, ymin, ymax = sx - hw, sx + hw, sy - hw, sy + hw
        nr = int(np.ceil((ymax - ymin) / self.CELL_M)) + 1
        nc = int(np.ceil((xmax - xmin) / self.CELL_M)) + 1
        occ = np.zeros((nr, nc), dtype=bool)
        origin = np.array([xmin, ymin], dtype=float)
        rows = (np.arange(nr) * self.CELL_M + ymin)[:, None]
        cols = (np.arange(nc) * self.CELL_M + xmin)[None, :]
        margin = self.r_self + 0.01

        def block_disc(cx, cy, r):
            occ[:] |= ((cols - cx) ** 2 + (rows - cy) ** 2) <= (r + margin) ** 2

        mj = self.mj
        # other free-jointed objects: footprint = xy + bounding radius
        for bid in self._others:
            p = self.data.xpos[bid]
            block_disc(float(p[0]), float(p[1]), self._planar_radius(bid))
        # static fixtures by name (basket, bowl, stove, ...)
        free_bids = self._others | {self.bid}
        for g in range(self.model.ngeom):
            bid = int(self.model.geom_bodyid[g])
            if bid in free_bids:
                continue
            bn = (mj.mj_id2name(self.model, mj.mjtObj.mjOBJ_BODY, bid)
                  or '').lower()
            if not any(k in bn for k in self.FIXTURE_KW):
                continue
            if 'robot' in bn or 'gripper' in bn:
                continue
            p = self.data.geom_xpos[g]
            if not (xmin - 0.1 <= p[0] <= xmax + 0.1
                    and ymin - 0.1 <= p[1] <= ymax + 0.1):
                continue
            block_disc(float(p[0]), float(p[1]), self._geom_planar_radius(g))
        for (r, c) in self._blocked_cells:
            if 0 <= r < nr and 0 <= c < nc:
                occ[r, c] = True
        self._grid = (occ, origin)
        return occ, origin

    def _cell_of(self, xy):
        occ, origin = self._grid
        c = int(round((float(xy[0]) - origin[0]) / self.CELL_M))
        r = int(round((float(xy[1]) - origin[1]) / self.CELL_M))
        r = min(max(r, 0), occ.shape[0] - 1)
        c = min(max(c, 0), occ.shape[1] - 1)
        return (r, c)

    def _xy_of(self, cell):
        _occ, origin = self._grid
        return np.array([origin[0] + cell[1] * self.CELL_M,
                         origin[1] + cell[0] * self.CELL_M], dtype=float)

    def _plan(self, cur_xy):
        """Plan a new polyline to a random free goal; True on success."""
        occ, _origin = self._build_grid()
        start = self._cell_of(cur_xy)
        if occ[start]:
            # the mover's own cell can read blocked after a contact mark;
            # free it so a plan can leave from where the object sits
            occ[start] = False
        free = np.argwhere(~occ)
        if len(free) == 0:
            return False
        min_cells = self.MIN_GOAL_M / self.CELL_M
        for _ in range(50):
            r, c = free[self.rng.randint(len(free))]
            if np.hypot(r - start[0], c - start[1]) < min_cells:
                continue
            path = astar_grid(occ, start, (int(r), int(c)))
            if path and len(path) > 1:
                self._path_pts = [self._xy_of(p) for p in path[1:]]
                self._path_idx = 0
                self.replans += 1
                return True
        return False

    def _planned_candidate(self, cur_xy):
        """Advance step_m along the current polyline; None if no path."""
        if self._path_idx >= len(self._path_pts):
            if not self._plan(cur_xy):
                return None
        remaining = self.step_m
        pos = np.asarray(cur_xy, float).copy()
        while remaining > 1e-9 and self._path_idx < len(self._path_pts):
            wp = self._path_pts[self._path_idx]
            d = float(np.linalg.norm(wp - pos))
            if d <= remaining:
                pos = wp.copy()
                remaining -= d
                self._path_idx += 1
            else:
                pos = pos + (wp - pos) * (remaining / d)
                remaining = 0.0
        if self._path_idx > 0:
            self.dir = pos - np.asarray(cur_xy, float)
            n = float(np.linalg.norm(self.dir))
            if n > 0:
                self.dir = self.dir / n
        return pos

    def step(self):
        if self.adr is None or self.stopped is not None:
            return False
        adr = self.adr
        if self._grasped():
            self.stopped = 'grasped'
            return False
        if self.data.qpos[adr + 2] > self.start[2] + 0.02:
            self.stopped = 'lifted'
            return False
        before = self.data.qpos[adr:adr + 3].copy()
        cand = None
        planned = False
        if self.planner == 'astar':
            cand = self._planned_candidate(before[:2])
            planned = cand is not None
        if cand is None:
            # box patrol (planner='box', or no path found this step)
            cand = before[:2] + self.dir * self.step_m
            for k in range(2):
                if abs(cand[k] - self.start[k]) > self.radius:
                    self.dir[k] = -self.dir[k]
                    cand[k] = before[k] + self.dir[k] * self.step_m
        self.data.qpos[adr] = cand[0]
        self.data.qpos[adr + 1] = cand[1]
        self.mj.mj_forward(self.model, self.data)
        if _body_object_contact(self.model, self.data, self.bid,
                                 self._others, self.mj):
            self.data.qpos[adr:adr + 3] = before
            self.mj.mj_forward(self.model, self.data)
            if planned and self._grid is not None:
                # remember the cell that produced a contact and re-plan
                self._blocked_cells.add(self._cell_of(cand))
                self._path_pts = []
                self._path_idx = 0
            else:
                self.dir = -self.dir
            return False
        try:
            self.env.sim.forward()
        except Exception:
            pass
        self.path_m += self.step_m
        self.n_steps += 1
        return True


def _resolve_pick_target_pos(det_map, pick_hint: str, score=None,
                               _fuzzy=fuzzy_key) -> Optional[np.ndarray]:
    """
    Position of the pick target in ``det_map``, for MID_APPROACH progress.

    Resolution order: the score's first ``move_to_keypoint`` label (it is
    a det_map key by construction), then fuzzy_key on the BDDL pick hint,
    then normalized token overlap ('akita_black_bowl_1' vs 'dark bowl'
    share no literal substring, but suite prompts usually do).
    """
    def _pos(key):
        if key is None or key not in det_map:
            return None
        p = getattr(det_map[key], 'position_3d', None)
        return None if p is None else np.asarray(p, dtype=float)

    # 1. Score's own binding.
    try:
        for act in _iter_leaves((score or {}).get('tree', {})):
            lbl = (act.get('params', {}) or {}).get('keypoint_label')
            if lbl:
                p = _pos(_fuzzy(det_map, str(lbl)))
                if p is not None:
                    return p
                break
    except Exception:
        pass
    # 2. Fuzzy key on the raw hint.
    p = _pos(_fuzzy(det_map, pick_hint))
    if p is not None:
        return p
    # 3. Normalized token overlap.
    toks = set(_hint_tokens(pick_hint))
    if toks:
        best_key, best_n = None, 0
        for k in det_map:
            n = len(toks & set(str(k).lower().replace('_', ' ').split()))
            if n > best_n:
                best_key, best_n = k, n
        p = _pos(best_key)
        if p is not None:
            return p
    return None


def _iter_leaves(tree):
    """Depth-first leaves of a BT score dict (sequence/selector aware)."""
    if not isinstance(tree, dict):
        return
    if tree.get('type') in ('sequence', 'selector'):
        for c in tree.get('children', []) or []:
            yield from _iter_leaves(c)
    else:
        yield tree


class PerturbationInjector:
    """
    Wraps ``env.step`` and fires a one-shot displacement at the scheduled
    phase.  ``POST_PLAN`` does not need the wrapper (the hook fires the
    displacement directly); ``MID_APPROACH`` and ``POST_GRASP_PLAN`` are
    step-triggered:

    * MID_APPROACH: fires when the EE has covered >= ``progress_frac`` of
      its initial distance to ``pick_target_pos``.
    * POST_GRASP_PLAN: fires on the first step whose gripper channel
      commands a close (``action[6] > 0.5``), BEFORE that step executes.

    ``displace_fn`` receives no arguments and performs the displacement
    (closure over env/hint/delta); injectable for unit tests along with
    ``get_ee_pos``.
    """

    def __init__(self, env, phase: Phase, *,
                   displace_fn: Callable[[], Optional[str]],
                   pick_target_pos: Optional[np.ndarray] = None,
                   get_ee_pos: Optional[Callable[[], Optional[np.ndarray]]] = None,
                   progress_frac: float = 0.5,
                   meta: Optional[dict] = None,
                   slide_steps: int = 0,
                   fire_on_first_step: bool = False):
        self.env = env
        self.phase = Phase(phase)
        self.displace_fn = displace_fn
        self.pick_target_pos = (None if pick_target_pos is None else
                                  np.asarray(pick_target_pos, float)[:3])
        self.get_ee_pos = get_ee_pos
        self.progress_frac = float(progress_frac)
        self.meta = meta if meta is not None else {}
        self.fired = False
        self.slide_steps = int(slide_steps)
        self.fire_on_first_step = bool(fire_on_first_step)
        self._slide_left = 0
        self._start_dist: Optional[float] = None
        self._orig_step = None

    # env.step wrapper
    def install(self) -> None:
        if self._orig_step is not None:
            return
        self._orig_step = self.env.step

        def _patched(action):
            if not self.fired and self._should_fire(action):
                self.fire()
            elif self._slide_left > 0:
                # continuous slide: one increment per env step
                self._slide_left -= 1
                try:
                    self.displace_fn()
                except Exception as e:
                    self.meta.setdefault('perturb_error', str(e))
                if self._slide_left == 0:
                    self.meta['t_perturb_end'] = time.time()
            return self._orig_step(action)

        self.env.step = _patched

    def uninstall(self) -> None:
        if self._orig_step is not None:
            try:
                self.env.step = self._orig_step
            except Exception:
                pass
            self._orig_step = None

    # firing logic
    def _should_fire(self, action) -> bool:
        if self.fire_on_first_step:
            return True
        if self.phase == Phase.POST_GRASP_PLAN:
            try:
                return float(np.asarray(action).reshape(-1)[6]) > 0.5
            except Exception:
                return False
        if self.phase == Phase.MID_APPROACH:
            if self.pick_target_pos is None or self.get_ee_pos is None:
                return False
            ee = self.get_ee_pos()
            if ee is None:
                return False
            dist = float(np.linalg.norm(
                np.asarray(ee, float)[:3] - self.pick_target_pos))
            if self._start_dist is None:
                self._start_dist = max(dist, 1e-6)
                return False
            progress = 1.0 - dist / self._start_dist
            return progress >= self.progress_frac
        return False

    def fire(self) -> None:
        """Start displacement: one-shot write, or a slide over N steps."""
        if self.fired:
            return
        self.fired = True
        t = time.time()
        body = None
        try:
            body = self.displace_fn()
        except Exception as e:  # never break the trial on injection failure
            self.meta['perturb_error'] = str(e)
        if self.slide_steps > 1:
            self._slide_left = self.slide_steps - 1
            self.meta['slide_steps'] = self.slide_steps
        self.meta['t_perturb'] = t
        self.meta['perturb_phase'] = self.phase.value
        if body is not None:
            self.meta['perturbed_body'] = body


def _default_ee_reader(env) -> Callable[[], Optional[np.ndarray]]:
    """
    EE position reader off the underlying MuJoCo grip site.
    """
    import mujoco
    model = env.sim.model._model
    data = env.sim.data._data
    site = -1
    for s in range(model.nsite):
        sn = (mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_SITE, s) or '').lower()
        if 'grip_site' in sn:
            site = s
            break

    def _read() -> Optional[np.ndarray]:
        if site < 0:
            return None
        mujoco.mj_forward(model, data)
        return data.site_xpos[site].copy()

    return _read


# Metrics

def compute_latency_metrics(trial_meta: dict) -> dict:
    """
    Detection / adaptation latency from the trial_meta event timeline.

    * detection_latency_s: t_perturb -> first SceneDiff record with
      target_status in {moved, missing} at t >= t_perturb.  Falls back to
      the first telemetry EMPTY_CLOSE / SLIP vote (``detected_by`` says
      which fired).
    * adaptation_latency_s: flag -> resumed nominal execution.  Retarget
      events carry (t_flag, t_resume); recovery attributions carry
      (t, t_resume) where t_resume is stamped just before re-execution.

    Pure dict-in / dict-out so it unit-tests on synthetic timelines.
    """
    out = {
        'detection_latency_s': None,
        'adaptation_latency_s': None,
        'detected_by': None,
        'adapted_by': None,
        'recovery_used': bool(trial_meta.get('recovery_attributions')
                                or trial_meta.get('bt_yaml_recovery')),
    }
    t_p = trial_meta.get('t_perturb')
    if t_p is None:
        return out

    # Earliest flag wins across both detectors: the scene diff (camera)
    # and the grasp-telemetry vote (which fires first when the object
    # moves as the jaws close, e.g. POST_GRASP_PLAN injections).
    candidates: list[tuple[float, str]] = []
    for rec in trial_meta.get('scene_diffs', []) or []:
        if (rec.get('target_status') in ('moved', 'missing')
                and float(rec.get('t', 0.0)) >= t_p):
            candidates.append((float(rec['t']), 'scene_diff'))
            break
    for rec in trial_meta.get('grasp_outcomes', []) or []:
        if (rec.get('outcome') in ('empty_close', 'slip')
                and float(rec.get('t', 0.0)) >= t_p):
            candidates.append((float(rec['t']), 'telemetry'))
            break
    if not candidates:
        return out
    t_flag, out['detected_by'] = min(candidates)
    out['detection_latency_s'] = t_flag - t_p

    for ev in trial_meta.get('retarget_events', []) or []:
        tf = float(ev.get('t_flag', 0.0))
        tr = ev.get('t_resume')
        if tr is not None and tf >= t_p:
            out['adaptation_latency_s'] = float(tr) - tf
            out['adapted_by'] = 'retarget'
            return out
    for rec in trial_meta.get('recovery_attributions', []) or []:
        tr = rec.get('t_resume')
        if tr is not None and float(tr) >= t_flag:
            out['adaptation_latency_s'] = float(tr) - t_flag
            out['adapted_by'] = f"recovery_{rec.get('layer', '?')}"
            return out
    return out


# Runner (heavy imports deferred; requires the fair-runner stack)

@dataclass
class DynConfig(_FairBase):  # type: ignore[misc, valid-type]
    """
    FairConfig + LIBERO-Dyn sweep parameters.

    Runs the BASE suite (perturbation is ours, injected mid-episode);
    ``num_trials`` trials per (phase, magnitude) cell.
    """
    task_id: int = 0
    """Task index within the base suite."""
    phases: str = 'post_plan,mid_approach,post_grasp_plan'
    """Comma-separated injection phases."""
    magnitudes_cm: str = '0,2,5,10'
    """Comma-separated displacement magnitudes (cm)."""
    dyn_output: str = ''
    """JSON output path (default: <output_dir or ~/libero_dyn_results>/
    dyn_<suite>_task<id>.json)."""
    trial_offset: int = 0
    """Base trial index (seed + init state + schedule all follow).  Debug
    protocol: iterate on held-out init states (e.g. --trial-offset 3)
    and NEVER tune on the eval grid's trial 0."""
    slide_steps: int = 0
    """0 = legacy instantaneous state write; N>0 spreads the displacement
    over N consecutive env steps as a continuous slide (delta/N per step),
    so the object visibly moves while the robot acts."""
    live_cm_s: float = 0.0
    """Live-scene mode: the target moves at this constant speed (cm/s) from
    the first step after planning until a robot or gripper body touches it,
    patrolling a box of half-width magnitude_cm around its start (direction
    reflects at the box edge and reverses on contact with another object).
    The object is in motion for the whole approach, so the grasp must meet
    a moving target. magnitude 0 stays the control cell."""
    live_planner: str = 'astar'
    """Object motion planner for live mode: 'astar' follows planned
    polylines between random free goals through an occupancy grid of the
    other objects and fixtures; 'box' is the reflect-in-a-box patrol."""
    distractor_count: int = 0
    """With slide_steps>0: also displace this many non-target objects on
    staggered, collision-checked schedules (MultiSlideController), so the
    whole scene is in motion. 0 = only the pick target moves."""
    debug_gt: bool = False
    """Print ground-truth free-joint body positions at trial start and
    just before scoring (diagnosis only - never feeds the pipeline)."""


def run_dyn(cfg: 'DynConfig') -> dict:
    if not _HAVE_FAIR:
        raise RuntimeError('spark_bench.fair.config failed to import - '
                             'run_dyn needs the full LIBERO stack')
    from spark_bench.fair.config import (
        benchmark, get_task_prompts_for_suite, load_libero_env)
    from spark_bench.fair.execution import run_spark_on_libero_env

    base_suite = f'libero_{cfg.suite}'
    benchmark_dict = benchmark.get_benchmark_dict()
    task_suite = benchmark_dict[base_suite]()
    task = task_suite.get_task(cfg.task_id)

    env, _task, init_states, bddl_path = load_libero_env(
        base_suite, cfg.task_id, cfg)
    # Perturbed trials legitimately spend extra steps on retargets and
    # recovery re-executions; the stock 2000-step horizon terminates the
    # robosuite episode mid-recovery ("executing action in terminated
    # episode").  Give Dyn trials headroom.
    try:
        env.env.horizon = max(int(getattr(env.env, 'horizon', 2000)), 6000)
    except Exception:
        pass
    task_info = get_task_prompts_for_suite(base_suite, cfg.task_id, task,
                                             bddl_path=bddl_path)
    pick_hint = task_info.get('pick', '') or ''
    instruction = task_info['instruction']
    prompts = task_info['prompts']

    phases = [Phase(p.strip()) for p in cfg.phases.split(',') if p.strip()]
    mags = [float(m) for m in cfg.magnitudes_cm.split(',') if m.strip()]

    print(f"LIBERO-Dyn: {task.name} | phases={[p.value for p in phases]} "
          f"mags={mags}cm trials/cell={cfg.num_trials}")

    def _print_gt(tag: str) -> None:
        if not getattr(cfg, 'debug_gt', False):
            return
        try:
            import mujoco
            model = env.sim.model._model
            data = env.sim.data._data
            mujoco.mj_forward(model, data)
            rows = []
            for jid in range(model.njnt):
                if int(model.jnt_type[jid]) != int(mujoco.mjtJoint.mjJNT_FREE):
                    continue
                bid = int(model.jnt_bodyid[jid])
                bn = mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_BODY, bid) or ''
                p = data.xpos[bid]
                rows.append(f'{bn}=({p[0]:.3f},{p[1]:.3f},{p[2]:.3f})')
            print(f'[GT:{tag}] ' + ' '.join(rows), flush=True)
        except Exception as e:
            print(f'[GT:{tag}] failed: {e}', flush=True)

    cells: list[dict] = []
    for phase in phases:
        for mag in mags:
            cell = {'phase': phase.value, 'magnitude_cm': mag, 'trials': []}
            for trial_i in range(cfg.num_trials):
                trial = trial_i + int(getattr(cfg, 'trial_offset', 0))
                trial_meta: dict = {'trial': trial, 'phase': phase.value,
                                      'magnitude_cm': mag}
                injector: Optional[PerturbationInjector] = None
                recorder = None
                success = False
                try:
                    env.seed(trial)
                    env.reset()
                    if init_states is not None and len(init_states) > 0:
                        env.set_init_state(init_states[trial % len(init_states)])
                        env.reset()
                    for _ in range(10):
                        env.step(np.zeros(7))
                    _print_gt(f'{phase.value}/{mag:g}cm/T{trial}/start')

                    delta = displacement_vector(task.name, trial, mag)
                    trial_meta['direction'] = [round(float(x), 4)
                                                 for x in delta[:2]]

                    def _hook(hook_env, score, det,
                               _phase=phase, _delta=delta,
                               _meta=trial_meta):
                        nonlocal injector
                        if float(np.linalg.norm(_delta)) < 1e-9:
                            return  # magnitude 0: control cell
                        _trial_seed = _meta.get('trial', 0)
                        _delta_norm = float(np.linalg.norm(_delta[:2]))
                        live = float(getattr(cfg, 'live_cm_s', 0.0) or 0.0)
                        if live > 0.0:
                            # Live scene: constant-speed patrol inside a box of
                            # half-width |delta| until the robot touches it.
                            lc = LiveMotionController(
                                hook_env, pick_hint, _delta[:2],
                                speed_m_s=live / 100.0, radius_m=_delta_norm,
                                planner=getattr(cfg, 'live_planner', 'astar'),
                                seed=schedule_seed(task.name, _trial_seed))
                            _meta['live_planner'] = lc.planner
                            _meta['t_perturb'] = time.time()
                            _meta['perturb_phase'] = _phase.value
                            _meta['live_cm_s'] = live
                            _meta['live_radius_cm'] = round(_delta_norm * 100, 2)
                            orig_step = hook_env.step
                            def _lstep(action, _c=lc, _o=orig_step, _m=_meta):
                                _c.step()
                                if _c.stopped and 't_perturb_end' not in _m:
                                    _m['t_perturb_end'] = time.time()
                                    _m['live_stop_reason'] = _c.stopped
                                    _m['live_path_cm'] = round(_c.path_m * 100, 1)
                                    _m['live_steps'] = _c.n_steps
                                    _m['live_replans'] = _c.replans
                                return _o(action)
                            hook_env.step = _lstep
                            _meta['_restore_step'] = orig_step
                            _meta['_live_ctrl'] = lc
                            return
                        n_slide = max(int(getattr(cfg, 'slide_steps', 0)), 0)
                        step_delta = (_delta / n_slide) if n_slide > 1 else _delta
                        displace = lambda: displace_body_qpos(  # noqa: E731
                            hook_env, pick_hint, step_delta)
                        if _phase == Phase.POST_PLAN and n_slide <= 1:
                            t = time.time()
                            body = displace()
                            _meta['t_perturb'] = t
                            _meta['perturb_phase'] = _phase.value
                            if body:
                                _meta['perturbed_body'] = body
                            return
                        if _phase == Phase.POST_PLAN:
                            ndist = int(getattr(cfg, 'distractor_count', 0))
                            if n_slide > 1 and ndist > 0:
                                import mujoco as _mj
                                mdl = hook_env.sim.model._model
                                objs = _free_object_joints(mdl, _mj)
                                names = [bn for _a, _b, bn in objs
                                         if not any(t in bn for t in
                                                    _hint_tokens(pick_hint))]
                                rng = schedule_seed(task.name, _trial_seed)
                                movers = [{'hint': pick_hint, 'delta': _delta,
                                            'start_step': 0}]
                                for j in range(min(ndist, len(names))):
                                    nm = names[(rng + j * 7) % len(names)]
                                    dirn = DIRECTION_TABLE[(rng + j * 3)
                                                            % len(DIRECTION_TABLE)]
                                    mag = _delta_norm  # same magnitude class
                                    dv = np.array([dirn[0] * mag,
                                                    dirn[1] * mag, 0.0])
                                    movers.append({'hint': nm, 'delta': dv,
                                                    'start_step': 5 + j * 8})
                                ctrl = MultiSlideController(
                                    hook_env, movers, n_slide)
                                _meta['t_perturb'] = time.time()
                                _meta['perturb_phase'] = _phase.value
                                _meta['multi_slide'] = [m['hint']
                                                         for m in movers]
                                orig_step = hook_env.step
                                def _mstep(action, _c=ctrl, _o=orig_step,
                                            _m=_meta):
                                    _c.step()
                                    if not _c.active() and                                             't_perturb_end' not in _m:
                                        _m['t_perturb_end'] = time.time()
                                    return _o(action)
                                hook_env.step = _mstep
                                _meta['_restore_step'] = orig_step
                                return
                            injector = PerturbationInjector(
                                hook_env, _phase, displace_fn=displace,
                                meta=_meta, slide_steps=n_slide,
                                fire_on_first_step=True)
                            injector.install()
                            return
                        # Step-triggered phases need the pick target pos.
                        # The BDDL pick hint ('akita_black_bowl_1') and
                        # the det_map keys (prompt phrases like 'dark
                        # bowl') share few literal tokens, so fall back
                        # from fuzzy_key to token overlap on normalized
                        # strings, then to the first move_to_keypoint
                        # label of the score.
                        tgt = _resolve_pick_target_pos(
                            det.det_map, pick_hint, score, fuzzy_key)
                        injector = PerturbationInjector(
                            hook_env, _phase, displace_fn=displace,
                            pick_target_pos=tgt,
                            get_ee_pos=_default_ee_reader(hook_env),
                            meta=_meta, slide_steps=n_slide)
                        injector.install()

                    if getattr(cfg, 'save_rgb_frames', False):
                        from spark_bench.fair.execution import _RGBFrameRecorder
                        recorder = _RGBFrameRecorder(env)
                    success = run_spark_on_libero_env(
                        env, list(prompts), instruction,
                        pick_hint, task_info.get('place', ''), cfg,
                        trial_meta=trial_meta,
                        picks=task_info.get('picks') or None,
                        places=task_info.get('places') or None,
                        pick_counts=task_info.get('pick_counts') or None,
                        post_plan_hook=_hook)
                except Exception as e:
                    trial_meta['error'] = str(e)
                    if cfg.verbose:
                        import traceback
                        traceback.print_exc()
                finally:
                    if injector is not None:
                        injector.uninstall()
                    _lc = trial_meta.pop('_live_ctrl', None)
                    if _lc is not None and 'live_stop_reason' not in trial_meta:
                        trial_meta['live_stop_reason'] = 'never'
                        trial_meta['live_path_cm'] = round(_lc.path_m * 100, 1)
                        trial_meta['live_steps'] = _lc.n_steps
                    if _lc is not None:
                        trial_meta['live_replans'] = _lc.replans
                        trial_meta['live_planner'] = _lc.planner
                    _rs = trial_meta.pop('_restore_step', None)
                    if _rs is not None:
                        try: env.step = _rs
                        except Exception: pass
                    if recorder is not None:
                        recorder.stop()
                        if recorder.rgb_a_list:
                            try:
                                import imageio.v2 as _iio
                                vp = (Path(cfg.output_dir or '.') /
                                      f'dyn_video_{phase.value}_{mag:g}cm_T{trial}.mp4')
                                vp.parent.mkdir(parents=True, exist_ok=True)
                                _iio.mimwrite(str(vp), recorder.rgb_a_list,
                                              fps=20, quality=8)
                                print(f'[libero_dyn] video -> {vp}', flush=True)
                            except Exception as e:
                                print(f'[libero_dyn] video save failed: {e}',
                                      flush=True)

                _print_gt(f'{phase.value}/{mag:g}cm/T{trial}/end')
                trial_meta['success'] = bool(success)
                trial_meta['perturb_fired'] = 't_perturb' in trial_meta
                trial_meta.update(compute_latency_metrics(trial_meta))
                cell['trials'].append(trial_meta)
                dl = trial_meta.get('detection_latency_s')
                al = trial_meta.get('adaptation_latency_s')
                print(f"  [{phase.value:>15s} {mag:4.0f}cm T{trial}] "
                      f"success={success} "
                      f"fired={trial_meta['perturb_fired']} "
                      f"det={dl if dl is None else f'{dl:.3f}s'} "
                      f"adapt={al if al is None else f'{al:.3f}s'} "
                      f"by={trial_meta.get('detected_by')}"
                      f"/{trial_meta.get('adapted_by')}", flush=True)
            n = len(cell['trials'])
            cell['success_rate'] = (sum(t['success'] for t in cell['trials'])
                                      / n if n else 0.0)
            cells.append(cell)

    try:
        env.close()
    except Exception:
        pass

    result = {
        'protocol': 'libero_dyn',
        'suite': cfg.suite,
        'task_id': cfg.task_id,
        'task_name': task.name,
        'instruction': instruction,
        'num_trials': cfg.num_trials,
        'config': {
            'event_captures': getattr(cfg, 'event_captures', False),
            'scene_diff': getattr(cfg, 'scene_diff', False),
            'layered_recovery': getattr(cfg, 'layered_recovery', False),
            'retarget_max_cm': getattr(cfg, 'retarget_max_cm', None),
            'no_gemini': cfg.no_gemini,
        },
        'cells': cells,
    }
    out_dir = Path(cfg.output_dir) if cfg.output_dir else (
        Path.home() / 'libero_dyn_results')
    out_path = (Path(cfg.dyn_output) if cfg.dyn_output else
                out_dir / f'dyn_{cfg.suite}_task{cfg.task_id}.json')
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, 'w') as f:
        json.dump(result, f, indent=2, default=str)
    print(f"[libero_dyn] wrote {out_path}")
    return result


def main() -> None:
    import tyro
    run_dyn(tyro.cli(DynConfig))


if __name__ == '__main__':
    main()

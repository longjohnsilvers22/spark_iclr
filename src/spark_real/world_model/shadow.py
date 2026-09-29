"""Shadow-sim world model: best-of-N candidate-BT selection via MuJoCo rollout.

In simulation this uses MuJoCo itself as the dynamics oracle: snapshot the
env, roll a candidate BT, check success, and restore. On real robots a
learned dynamics model
(predicate or full-state JEPA) drops in via the same ``RolloutBackend``
protocol; only the winning BT runs on the real arm.

The pattern is intentionally cap_gym-flavoured: sample N programs, score
by simulated reward, keep the best. No xfrc, no qfrc, no sim-state writes
beyond the snapshot/restore: only ``mujoco.mj_step`` via the existing
LIBERO env, satisfying the honest-physics constraint.
"""
from __future__ import annotations

import os
import time
import logging
from typing import Callable, Optional, Sequence

logger = logging.getLogger(__name__)

__all__ = [
    'shadow_select_and_execute',
    'shadow_enabled',
    'shadow_k',
]


def shadow_enabled() -> bool:
    """Return True when ``SPARK_SHADOW_SIM`` env var enables the selector."""
    return os.environ.get('SPARK_SHADOW_SIM', '0').lower() in (
        '1', 'true', 'yes', 'on')


def shadow_k(default: int = 3) -> int:
    """Number of candidate BTs to shadow-roll per trial (env ``SPARK_SHADOW_K``)."""
    try:
        return max(1, int(os.environ.get('SPARK_SHADOW_K', default)))
    except ValueError:
        return default


def _snapshot(env) -> Optional[object]:
    """Return a restorable opaque handle to the env's MuJoCo state."""
    try:
        return env.get_sim_state()
    except Exception:
        try:
            return env.sim.get_state().flatten()
        except Exception as e:
            logger.warning("shadow snapshot failed: %s", e)
            return None


def _restore(env, state) -> bool:
    """Restore env to the previously snapshotted state. Returns True on success.

    Resets the robosuite OSC controller's internal goal-pose state alongside
    the MuJoCo qpos/qvel restore; without this, the controller carries over
    goal pose from the previous candidate's last primitive call and issues
    spurious commands on the first few steps of the next rollout.
    """
    if state is None:
        return False
    try:
        env.set_state(state)
        env.sim.forward()
        # Reset robosuite OSC controller(s) so goal-pose state matches the
        # restored qpos. env.robots is a list of Manipulator wrappers.
        try:
            robots = getattr(env, 'robots', None) or getattr(env.env, 'robots', [])
            for robot in (robots or []):
                ctrl = getattr(robot, 'controller', None)
                if ctrl is not None and hasattr(ctrl, 'reset_goal'):
                    ctrl.reset_goal()
        except Exception:  # Controller API drift: fall through, set_state alone is usually enough.
            pass
        try:
            env._update_observables(force=True)
        except Exception:
            pass
        return True
    except Exception as e:
        logger.warning("shadow restore failed: %s", e)
        return False


def shadow_select_and_execute(
    env,
    candidate_bts: Sequence[dict],
    execute_fn: Callable[[object, dict], None],
    *,
    max_candidates: Optional[int] = None,
    verbose: bool = False,
) -> tuple[Optional[dict], dict]:
    """Try each candidate BT in ``env`` until one succeeds; restore between attempts.

    In simulation, this is "best-of-N with checkpoint resume": because the
    env IS the metric, the moment a rollout returns ``check_success() ==
    True`` the search is done; no separate "real" rollout is needed. Restores
    happen lazily: only when the previous candidate failed and another
    remains.

    On real robots, callers should use a separate shadow env for rollouts
    and execute only the winning BT on the real arm (a different code
    path; this function is sim-only).

    Args:
        env: LIBERO/robosuite env with ``get_sim_state``, ``set_state``,
             ``check_success``, ``sim.forward`` (env_wrapper.ControlEnv).
        candidate_bts: BT score dicts to try in order. First success wins.
        execute_fn: ``(env, score) -> None``, runs one BT against ``env``.
                    Whatever the libero_pro executor wraps; should not
                    raise on primitive failure (executor handles its own
                    fallbacks).
        max_candidates: cap on rollouts; falls back to ``shadow_k()``.

    Returns:
        ``(winning_bt | None, diagnostics_dict)``. ``winning_bt`` is the
        score that achieved ``check_success`` (env is left in the success
        state; DO NOT re-execute). On total failure it is the last BT
        tried (env restored to snapshot). Diagnostics carry per-rollout
        success and timing for logging.
    """
    if not candidate_bts:
        return None, {'rollouts': [], 'winner_idx': None, 'fell_through': True}

    n = min(len(candidate_bts), max_candidates or shadow_k())
    state0 = _snapshot(env)
    if state0 is None:
        if verbose:
            logger.info("shadow: snapshot unavailable; running first BT only")
        try:
            execute_fn(env, candidate_bts[0])
        except Exception as e:
            logger.warning("shadow primary exec failed: %s", e)
        return candidate_bts[0], {
            'rollouts': [{'idx': 0, 'success': bool_check(env)}],
            'winner_idx': 0,
            'fell_through': True,
        }

    diagnostics: dict = {'rollouts': [], 'winner_idx': None, 'fell_through': False}
    for i in range(n):
        bt = candidate_bts[i]
        t0 = time.time()
        if i > 0:
            ok = _restore(env, state0)
            if not ok:
                # Restore broke: bail to whatever state is left.
                diagnostics['rollouts'].append({
                    'idx': i, 'success': False, 'note': 'restore_failed',
                    'dt': round(time.time() - t0, 2),
                })
                diagnostics['fell_through'] = True
                break
        try:
            execute_fn(env, bt)
        except Exception as e:
            diagnostics['rollouts'].append({
                'idx': i, 'success': False, 'exception': str(e)[:200],
                'dt': round(time.time() - t0, 2),
            })
            if verbose:
                logger.info("shadow rollout %d raised: %s", i, e)
            continue
        ok = bool_check(env)
        diagnostics['rollouts'].append({
            'idx': i, 'success': ok, 'dt': round(time.time() - t0, 2),
        })
        if verbose:
            logger.info("shadow rollout %d: success=%s dt=%.1fs",
                         i, ok, time.time() - t0)
        if ok:
            diagnostics['winner_idx'] = i
            return bt, diagnostics

    # No candidate succeeded: restore to snapshot and surrender the last BT.
    _restore(env, state0)
    return candidate_bts[n - 1], diagnostics


def bool_check(env) -> bool:
    """Defensive ``env.check_success()``: returns False on any exception."""
    try:
        return bool(env.check_success())
    except Exception:
        return False

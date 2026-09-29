#!/usr/bin/env python3
"""
LIBERO-PRO success checker using exact BDDL predicates.

Runs in openvla_env (has correct robosuite version).
Called as subprocess from spark_conda-based pipeline.

Usage:
    python -m spark_bench.libero_success_checker \
        --bddl /path/to/task.bddl \
        --state /path/to/final_state.npz \
        [--render /path/to/output.mp4]

The state file contains:
    - qpos: joint positions
    - qvel: joint velocities
"""

import argparse
import json
import os
import sys
from pathlib import Path
import numpy as np

os.environ.setdefault('MUJOCO_GL', 'egl')
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / 'libero_pro'))

try:
    from libero.libero.envs import OffScreenRenderEnv
except ImportError:
    OffScreenRenderEnv = None


def check_success(bddl_path: str, state_path: str, render_path: str = None):
    """
    Create LIBERO env, load state, check success using BDDL predicates.
    """
    env = OffScreenRenderEnv(
        bddl_file_name=bddl_path,
        robots=['Panda'],
        camera_names=['agentview', 'robot0_eye_in_hand'],
        camera_heights=480,
        camera_widths=640,
        has_offscreen_renderer=True,
        use_camera_obs=True,
    )
    env.reset()

    # Load final state from the raw MuJoCo controller
    state_data = np.load(state_path)
    qpos = state_data['qpos']
    qvel = state_data['qvel']

    # Sync state: set qpos/qvel in robosuite's MjSim
    sim = env.env.sim
    n_qpos = min(len(qpos), sim.data.qpos.shape[0])
    n_qvel = min(len(qvel), sim.data.qvel.shape[0])
    sim.data.qpos[:n_qpos] = qpos[:n_qpos]
    sim.data.qvel[:n_qvel] = qvel[:n_qvel]
    sim.forward()

    # Update object states (needed for predicate evaluation)
    if hasattr(env.env, '_post_process'):
        env.env._post_process()

    # Check success using exact BDDL predicates
    success = env.check_success()

    # Render both cameras if requested
    if render_path:
        obs = env.env._get_observations()
        agentview = obs.get('agentview_image', None)
        wrist = obs.get('robot0_eye_in_hand_image', None)
        if agentview is not None:
            np.savez(render_path.replace('.mp4', '_frames.npz'),
                     agentview=agentview, wrist=wrist)

    env.close()
    return success


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--bddl', required=True, help='Path to BDDL file')
    parser.add_argument('--state', required=True, help='Path to state .npz file')
    parser.add_argument('--render', default=None, help='Path to save rendered frame')
    args = parser.parse_args()

    success = check_success(args.bddl, args.state, args.render)
    # Output result as JSON for easy parsing by caller
    print(json.dumps({'success': success}))
    sys.exit(0 if success else 1)


if __name__ == '__main__':
    main()

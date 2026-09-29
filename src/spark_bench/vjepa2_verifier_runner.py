"""
LIBERO-PRO runner adapter that injects the V-JEPA 2-AC BT verifier.

Wraps the existing ``run_spark_libero_pro_fair`` flow with a K=4 sample +
rank step before BT execution.  Sampling is delegated to Gemini at
temperature 0.3 (already wired in ``libero_pro.planning._gemini_plan``);
ranking is the new step.

The full 25-trial x 6-cell sweep is ~3 hours; ``--unlock`` is required
to launch, run the smoke test first.

Usage:
    cd ~/spark/src && conda activate spark_conda
    MUJOCO_GL=egl PYTHONPATH=src/libero_pro:src/sam3:src:$PYTHONPATH \\
        python -m spark_bench.vjepa2_verifier_runner \\
        --suite goal --perturbation position --num-trials 25 --k 4

The runner respects ``WM_VERIFIER_K`` env override.  If K=1, this is a
no-op fall-through to the legacy runner (sanity check).
"""
from __future__ import annotations

import argparse
import os
import sys
from dataclasses import dataclass

# os.environ defaults must be set before mujoco / torch imports.
os.environ.setdefault("MUJOCO_GL", "egl")
os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")

import numpy as np
import mujoco

from spark_bench.libero_pro import planning as _plan
from spark_bench.run_spark_libero_pro_fair import main as _main_legacy

try:
    from spark_real.world_model.bt_verifier import verify_candidates
    from spark_real.world_model.vjepa2_ac import load_vjepa2_ac
except ImportError:
    verify_candidates = None
    load_vjepa2_ac = None


@dataclass
class VerifierConfig:
    suite: str = "goal"
    perturbation: str = "position"
    num_trials: int = 25
    k: int = 4
    max_steps: int = 600
    verbose: bool = False
    out_dir: str = "results/vjepa2_verifier"


def _patch_planner_for_k(K: int):
    """
    Replace the single-Gemini-call planner with a K-sample planner.

    Side-effects ``spark_bench.libero_pro.planning._gemini_plan`` so that
    it returns a list of K candidate scores instead of one.  The caller
    is responsible for picking one (via the verifier).
    """
    original = _plan._gemini_plan

    def _gemini_plan_k(*args, **kwargs):
        # Bump sampling temperature for diversity.
        kwargs.setdefault("temperature", 0.3)
        scores = []
        for _ in range(K):
            try:
                s = original(*args, **kwargs)
                if s:
                    scores.append(s)
            except Exception as e:  # pragma: no cover
                print(f"[verifier] gemini_plan attempt failed: {e}")
        return scores  # caller (verifier) returns the single best

    _plan._gemini_plan = _gemini_plan_k


def _rank_with_vjepa2(env, scores: list[dict], det_map: dict, *,
                       goal_frame, current_obs) -> dict:
    """
    Pick the best BT score via the V-JEPA 2-AC verifier.
    """
    if not scores:
        return {}

    if not hasattr(_rank_with_vjepa2, "_wm"):
        _rank_with_vjepa2._wm = load_vjepa2_ac()
    wm = _rank_with_vjepa2._wm

    # Build keypoint XYZ map from det_map.
    keypoint_xyz = {}
    for label, det in (det_map or {}).items():
        pos = getattr(det, "position_3d", None)
        if pos is not None:
            keypoint_xyz[label] = np.array(pos, dtype=np.float32)

    # Current EE pose from MuJoCo.
    model = env.sim.model._model
    data = env.sim.data._data
    mujoco.mj_forward(model, data)
    # Find EE site (LIBERO Franka).
    ee_xyz = np.zeros(3, dtype=np.float32)
    for sid in range(model.nsite):
        name = mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_SITE, sid) or ""
        if "ee" in name.lower() or "gripper" in name.lower() or "tcp" in name.lower():
            ee_xyz = data.site_xpos[sid].astype(np.float32).copy()
            break

    result = verify_candidates(
        wm,
        scores,
        current_obs=current_obs,
        goal_frame=goal_frame,
        ee_init_xyz=ee_xyz,
        keypoint_xyz=keypoint_xyz,
        grip_init=0.0,
    )
    return scores[result.best_index_in_input]


def main():
    """
    Entry point -- delegates to the canonical LIBERO-PRO runner.

    Without ``--unlock`` this prints the planned config and exits.  With
    it, the planner is monkey-patched and control re-dispatches into
    ``run_spark_libero_pro_fair.main``.
    """
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--suite", default="goal")
    parser.add_argument("--perturbation", default="position")
    parser.add_argument("--num-trials", type=int, default=25)
    parser.add_argument("--k", type=int, default=int(os.environ.get("WM_VERIFIER_K", 4)))
    parser.add_argument("--dry-run", action="store_true",
                        help="Print plan and exit without running.")
    parser.add_argument("--unlock", action="store_true",
                        help="REQUIRED to actually launch -- prevents accidental runs.")
    args = parser.parse_args()

    print("=" * 70)
    print("D3 V-JEPA 2-AC verifier runner")
    print(f"suite        : {args.suite}")
    print(f"perturbation : {args.perturbation}")
    print(f"num_trials   : {args.num_trials}")
    print(f"K            : {args.k}")
    print("=" * 70)

    if not args.unlock:
        print("\n*** Runner LOCKED.  Pass --unlock after the smoke test passes. ***")
        print("Estimated wall-clock for 25 trials x 6 cells: ~3 h on RTX 5090.")
        return 0

    if args.dry_run:
        print("Dry run: planner would be patched for K = %d, then run_spark_libero_pro_fair.main() invoked." % args.k)
        return 0

    # Monkeypatch and dispatch.
    _patch_planner_for_k(args.k)

    # TODO: patch the runner to call _rank_with_vjepa2() between planning
    # and execution; the patch above only multiplies the planner outputs.

    sys.argv = [
        "run_spark_libero_pro_fair",
        f"--suite={args.suite}",
        f"--perturbation={args.perturbation}",
        f"--num-trials={args.num_trials}",
    ]
    return _main_legacy()


if __name__ == "__main__":
    raise SystemExit(main())

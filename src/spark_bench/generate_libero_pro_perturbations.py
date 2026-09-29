#!/usr/bin/env python3
"""
Generate perturbed BDDL files and init states for LIBERO-PRO evaluation.

This must be run ONCE before evaluation to create the perturbed environments.
Uses perturbation.py from LIBERO-PRO to apply swap/task perturbations to base BDDLs.

Usage:
    conda activate openvla
    PYTHONPATH="libero_pro:$PYTHONPATH" python -m spark_bench.generate_libero_pro_perturbations
"""
import argparse
import os
import pickle
import sys
import shutil
import zipfile
from pathlib import Path

# os.environ defaults must be set before libero imports.
os.environ.setdefault('MUJOCO_GL', 'egl')

import numpy as np

# libero_pro (vendored under src/libero_pro) must be on the path before its
# perturbation / libero modules import.
LIBERO_PRO_ROOT = Path(__file__).parent.parent / 'libero_pro'
sys.path.insert(0, str(LIBERO_PRO_ROOT))

from perturbation import (
    BDDLParser, SwapPerturbator, TaskPerturbator,
    PerturbFlags, BDDLCombinedPerturbator
)
from libero.libero import get_libero_path
from libero.libero.envs import OffScreenRenderEnv


SUITES = ['libero_object', 'libero_spatial', 'libero_goal']
PERTURBATION_TYPES = {
    'swap': PerturbFlags(use_swap=True),
    'task': PerturbFlags(use_task=True),
}

OOD_CONFIGS = {
    'swap': str(LIBERO_PRO_ROOT / 'libero_ood' / 'ood_spatial_relation.yaml'),
    'task': str(LIBERO_PRO_ROOT / 'libero_ood' / 'ood_task.yaml'),
    'object': str(LIBERO_PRO_ROOT / 'libero_ood' / 'ood_object.yaml'),
    'language': str(LIBERO_PRO_ROOT / 'libero_ood' / 'ood_language.yaml'),
    'environment': str(LIBERO_PRO_ROOT / 'libero_ood' / 'ood_environment.yaml'),
}

SEED = 42


def generate_perturbed_bddls():
    bddl_base = Path(get_libero_path("bddl_files"))

    for suite in SUITES:
        for ptype, flags in PERTURBATION_TYPES.items():
            suffix = ptype  # swap or task
            output_suite = f"{suite}_{suffix}"
            input_dir = bddl_base / suite
            output_dir = bddl_base / output_suite

            if not input_dir.exists():
                print(f"[SKIP] Input dir not found: {input_dir}")
                continue

            if output_dir.exists():
                shutil.rmtree(output_dir)
            output_dir.mkdir(parents=True, exist_ok=True)

            bddl_files = sorted(input_dir.glob("*.bddl"))
            print(f"\n{'='*60}")
            print(f"Generating {output_suite}: {len(bddl_files)} tasks")
            print(f"{'='*60}")

            pipeline = BDDLCombinedPerturbator(configs=OOD_CONFIGS)

            for bddl_file in bddl_files:
                task_name = bddl_file.stem
                with open(bddl_file, 'r') as f:
                    content = f.read()

                try:
                    perturbed = pipeline.perturb_content(
                        content=content,
                        task_suite_name=suite,
                        task_name=task_name,
                        flags=flags,
                        seed=SEED,
                    )

                    output_file = output_dir / bddl_file.name
                    with open(output_file, 'w') as f:
                        f.write(perturbed)

                    changed = "CHANGED" if perturbed != content else "UNCHANGED"
                    print(f"{task_name}: {changed}")

                except Exception as e:
                    print(f"{task_name}: ERROR - {e}")
                    # Copy original as fallback
                    shutil.copy2(bddl_file, output_dir / bddl_file.name)

            print(f"Output: {output_dir} ({len(list(output_dir.glob('*.bddl')))} files)")


def generate_init_states():
    """
    Generate init states for all perturbed suites.
    """
    bddl_base = Path(get_libero_path("bddl_files"))
    init_base = Path(get_libero_path("init_states"))

    NUM_INITS = 50

    for suite in SUITES:
        for ptype in PERTURBATION_TYPES:
            output_suite = f"{suite}_{ptype}"
            bddl_dir = bddl_base / output_suite
            init_dir = init_base / output_suite

            if not bddl_dir.exists():
                print(f"[SKIP] BDDL dir not found: {bddl_dir}")
                continue

            init_dir.mkdir(parents=True, exist_ok=True)
            bddl_files = sorted(bddl_dir.glob("*.bddl"))

            print(f"\n{'='*60}")
            print(f"Generating init states for {output_suite}: {len(bddl_files)} tasks, {NUM_INITS} states each")
            print(f"{'='*60}")

            for bddl_file in bddl_files:
                task_name = bddl_file.stem
                output_file = init_dir / f"{task_name}.pruned_init"

                if output_file.exists():
                    print(f"{task_name}: already exists, skipping")
                    continue

                print(f"{task_name}: generating {NUM_INITS} init states...", end=" ", flush=True)

                all_states = []
                for i in range(NUM_INITS):
                    env = None
                    try:
                        env = OffScreenRenderEnv(
                            bddl_file_name=str(bddl_file),
                            camera_heights=128,
                            camera_widths=128,
                        )
                        state = env.get_sim_state()
                        all_states.append(state)
                    except Exception as e:
                        if i == 0:
                            print(f"ERROR on first state: {e}")
                            break
                    finally:
                        if env is not None:
                            env.close()

                if all_states:
                    all_states = np.array(all_states)
                    with zipfile.ZipFile(output_file, 'w', zipfile.ZIP_DEFLATED) as zf:
                        zf.writestr("archive/data.pkl", pickle.dumps(all_states))
                        zf.writestr("archive/version", b"1")
                    print(f"OK ({len(all_states)} states)")
                else:
                    print("FAILED (0 states)")


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--bddl-only', action='store_true', help='Only generate perturbed BDDLs')
    parser.add_argument('--init-only', action='store_true', help='Only generate init states')
    args = parser.parse_args()

    if args.init_only:
        generate_init_states()
    elif args.bddl_only:
        generate_perturbed_bddls()
    else:
        generate_perturbed_bddls()
        generate_init_states()

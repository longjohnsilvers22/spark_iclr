#!/usr/bin/env bash
# Launch robots_realtime viser IK teleop for FR3.
# Usage: ./scripts/rr_teleop.sh [--no-tui]
#
# Requires: conda env 'rr_env' with robots_realtime installed.
# Opens viser at http://localhost:8765

set -euo pipefail

# Borrow spark_conda's bundled CUDA libs for JAX. Override the prefix on hosts
# where the env lives elsewhere.
SPARK_CONDA_NVIDIA=${SPARK_CONDA_NVIDIA:-"$HOME/miniconda3/envs/spark_conda/lib/python3.12/site-packages/nvidia"}
NVIDIA_LIBS=$(find "$SPARK_CONDA_NVIDIA" \
    -maxdepth 2 -name "lib" -type d 2>/dev/null | tr '\n' ':')
export LD_LIBRARY_PATH="${NVIDIA_LIBS}/usr/local/cuda/lib64:${LD_LIBRARY_PATH:-}"
export JAX_PLATFORMS=cuda
export XLA_FLAGS="--xla_gpu_deterministic_ops=false"

eval "$(conda shell.bash hook 2>/dev/null)"
conda activate rr_env

cd ~/robots_realtime
exec rr-session configs/franka/franka_spark_viser_teleop_franky.yaml "$@"

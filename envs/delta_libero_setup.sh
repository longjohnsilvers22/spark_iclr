#!/bin/bash
# Minimal LIBERO / LIBERO-PRO / LIBERO-Dyn evaluation environment for the clusters.
# The full `conda env export` of the lab machine does not resolve on a fresh box
# (incremental local pins), so this installs only what the runners import, with
# hard pins on the four packages the benchmark protocol depends on.
# Usage: bash delta_libero_setup.sh <miniforge_root> <env_name>
set -eu
ROOT=${1:-/projects/biqw/bgrant1/miniforge}
ENV=${2:-libero_env}
source "$ROOT/etc/profile.d/conda.sh"
conda config --set always_yes true
conda env remove -n "$ENV" >/dev/null 2>&1 || true
conda create -n "$ENV" python=3.12 pip
conda activate "$ENV"
# stage 1: CUDA torch from the PyTorch index (the resolver picks a CPU wheel otherwise)
pip install torch==2.9.1 torchvision==0.24.1 --index-url https://download.pytorch.org/whl/cu126
# stage 2: protocol pins
pip install mujoco==3.4.0 robosuite==1.4.1 PyOpenGL==3.1.10 "numpy==2.3.5"
# stage 3: SPARK runtime + LIBERO fork deps, loose pins
pip install scipy opencv-python pillow scikit-image pyyaml requests tqdm tyro lark \
  matplotlib imageio imageio-ffmpeg fastapi uvicorn pydantic websockets h5py \
  google-genai google-generativeai huggingface_hub "transformers>=4.50" einops \
  hydra-core easydict bddl cloudpickle gym thop termcolor
# stage 4: SAM3 from the staged checkout (its pyproject wants numpy<2; the code runs on numpy 2, so re-pin after)
SAM3_DIR=${SAM3_DIR:-/projects/biqw/bgrant1/mv_sam3/sam3}
if [ -d "$SAM3_DIR" ]; then
  pip install -e "$SAM3_DIR"
  pip install decord pycocotools psutil
fi
# robosuite and SAM3 installs may downgrade numpy; restore the pin
pip install "numpy==2.3.5"
python - <<'PY'
import torch, mujoco, robosuite, numpy, cv2, scipy, yaml, tyro, h5py
print("torch", torch.__version__, "cuda", torch.cuda.is_available())
print("mujoco", mujoco.__version__, "robosuite", robosuite.__version__, "numpy", numpy.__version__)
PY
echo "env $ENV ready at $(date)"

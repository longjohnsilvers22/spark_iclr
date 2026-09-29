#!/bin/bash
# SPARK Conda Environment Setup Script
# Sets up conda environment with SAM3 and ROS2 integration

echo "=== SPARK Conda Environment Setup ==="
echo ""

# Resolve the repo root from this script's location (src/ is the script dir).
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"

# Check if conda is available
if ! command -v conda &> /dev/null; then
    echo "Error: conda not found. Please install miniconda first."
    exit 1
fi

# Create or activate spark_conda environment
ENV_NAME="spark_conda"
if conda env list | grep -q "^${ENV_NAME} "; then
    echo "Environment '${ENV_NAME}' already exists. Activating..."
    eval "$(conda shell.bash hook)"
    conda activate ${ENV_NAME}
else
    echo "Creating new conda environment '${ENV_NAME}'..."
    conda create -n ${ENV_NAME} python=3.12 -y
    eval "$(conda shell.bash hook)"
    conda activate ${ENV_NAME}
fi

echo ""
echo "Installing PyTorch 2.11.0 + torchvision 0.26.0 with CUDA 13.0 support..."
# DO NOT downgrade to torch 2.7+cu126 - that combination hangs inside
# EquiGraspFlow's VN-DGCNN on Ada Lovelace GPUs (RTX 4090).
# See SETUP.md "EquiGraspFlow hangs inside model.sample()" for details.
pip install torch==2.11.0 torchvision==0.26.0 --index-url https://download.pytorch.org/whl/cu130

echo ""
echo "Pinning numpy to 1.26.4 (torch install pulls in numpy>=2 which is incompatible)..."
pip install numpy==1.26.4

echo ""
echo "Installing SAM3 and dependencies..."

# Navigate to SAM3 directory and install
SAM3_DIR="${SAM3_DIR:-$HOME/mv_sam3/sam3}"
if [ -d "$SAM3_DIR" ]; then
    cd "$SAM3_DIR"
    pip install -e .
    echo "SAM3 installed successfully!"
else
    echo "Warning: SAM3 directory not found at $SAM3_DIR"
    echo "Please clone SAM3 from https://github.com/facebookresearch/sam3"
fi

echo ""
echo "Installing ROS2 Python dependencies..."
# Install ROS2 packages that are pip-installable (numpy already installed via SAM3)
pip install opencv-python pandas
pip install empy lark catkin_pkg

echo ""
echo "Building SPARK ROS2 workspace..."
cd "$REPO_ROOT"

# Set Python executable for colcon
export COLCON_PYTHON_EXECUTABLE="$(which python3)"

# Build the workspace
colcon build --symlink-install

echo ""
echo "=== Setup Complete ==="
echo ""
echo "SAM3 weights location: ~/.cache/huggingface/hub/models--facebook--sam3/"
echo ""
echo "To use SPARK, activate the environment in each terminal:"
echo "  conda activate ${ENV_NAME}"
echo "  source ~/spark/install/setup.bash"
echo ""
echo "For convenience, add to your ~/.bashrc:"
echo "  alias spark='conda activate ${ENV_NAME} && source ~/spark/install/setup.bash'"

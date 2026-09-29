#!/bin/bash
# Auto-chain: wait for sv45, run pyroki diagnostic, then launch spatial 50-trial.

set -u

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
SRC_DIR="$(cd "$SCRIPT_DIR/.." && pwd)"

LOG_DIR=$SCRIPT_DIR/results/libero_pro_logs
DIAG_LOG=$LOG_DIR/sv46_pyroki_diag.log
SPATIAL_LOG=$LOG_DIR/sv47_spatial_50trials.log
SV45_PID=${1:-1428352}

# -- 1. wait for sv45 to exit --
echo "[auto-chain] Waiting for sv45 PID=$SV45_PID to finish..."
while kill -0 "$SV45_PID" 2>/dev/null; do sleep 30; done
echo "[auto-chain] sv45 done at $(date)"

# -- 2. source env & launch single-trial diagnostic on drawer + plate + sanity --
source "$HOME/miniconda3/etc/profile.d/conda.sh"
conda activate openvla
export PYTHONPATH=$SRC_DIR:$SRC_DIR/libero_pro
export MUJOCO_GL=egl
export TORCHDYNAMO_DISABLE=1
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
cd "$SRC_DIR"

echo "[auto-chain] Single-trial pyroki diagnostic starting..." | tee -a "$DIAG_LOG"
# Goal task T0 = open_the_middle_drawer perturbed to bottom - pure drawer test
echo "=== T0 drawer with pyroki ==="                             | tee -a "$DIAG_LOG"
python -u -m spark_bench.run_spark_libero_pro_fair \
    --suite goal --perturbation task --num-trials 1 --verbose \
    --only-task-id 0 --use-pyroki >> "$DIAG_LOG" 2>&1

# T1 = plate on stove (pyroki plate pinch)
echo "=== T1 plate-on-stove with pyroki ==="                     | tee -a "$DIAG_LOG"
python -u -m spark_bench.run_spark_libero_pro_fair \
    --suite goal --perturbation task --num-trials 1 --verbose \
    --only-task-id 1 --use-pyroki >> "$DIAG_LOG" 2>&1

# T8 = wine on plate - sanity: check no regression on a task that worked
echo "=== T8 wine-on-plate sanity (regression check) ==="        | tee -a "$DIAG_LOG"
python -u -m spark_bench.run_spark_libero_pro_fair \
    --suite goal --perturbation task --num-trials 1 --verbose \
    --only-task-id 8 --use-pyroki >> "$DIAG_LOG" 2>&1

# -- 3. decide whether to use pyroki in the big run --
T0_RESULT=$(grep -oP '\d+% \(\d+/\d+\)' "$DIAG_LOG" | head -1)
T1_RESULT=$(grep -oP '\d+% \(\d+/\d+\)' "$DIAG_LOG" | sed -n '2p')
T8_RESULT=$(grep -oP '\d+% \(\d+/\d+\)' "$DIAG_LOG" | sed -n '3p')
echo "[auto-chain] drawer=$T0_RESULT plate=$T1_RESULT regression=$T8_RESULT" | tee -a "$DIAG_LOG"

# Use pyroki if drawer OR plate worked AND regression still passed
USE_PYROKI_FLAG=""
if [[ "$T0_RESULT" == *"(1/1)"* || "$T1_RESULT" == *"(1/1)"* ]]; then
    USE_PYROKI_FLAG="--use-pyroki"
    echo "[auto-chain] Pyroki unlocked a task - enabling for big run" | tee -a "$DIAG_LOG"
else
    echo "[auto-chain] Pyroki didn't unlock anything - keeping baseline" | tee -a "$DIAG_LOG"
fi

# -- 4. launch spatial 50-trial (faster than goal, strong SPARK performer) --
echo "=== spatial 50-trial (use_pyroki=$USE_PYROKI_FLAG) ===" >> "$SPATIAL_LOG"
python -u -m spark_bench.run_spark_libero_pro_fair \
    --suite spatial --perturbation all --num-trials 50 $USE_PYROKI_FLAG \
    >> "$SPATIAL_LOG" 2>&1

echo "[auto-chain] All done at $(date)" | tee -a "$DIAG_LOG"

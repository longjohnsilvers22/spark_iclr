#!/bin/bash
# P0-a matched-budget ablation, one suite at a time, RATS protocol
# (10 init states per task). Runs arm A (baseline split: 1 BT call at
# temp 0 + recovery replans) then arm B (plan-sampling: shadow best-of-N
# K=2, temp>0 diversity samples, env-checkpoint selection) sequentially
# on the same suite. Usage: p0a_suite_run.sh <suite> [num_trials]
set -u

SUITE=${1:?suite required: spatial|object|goal}
TRIALS=${2:-10}

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
SRC_DIR="$(cd "$SCRIPT_DIR/.." && pwd)"
LOG_DIR=$SCRIPT_DIR/results/libero_pro_logs
OUT_BASE=$HOME/spark/videos/libero_pro_fair
mkdir -p "$LOG_DIR"

PY=$HOME/miniconda3/envs/openvla_env/bin/python
export MUJOCO_GL=egl
export TORCHDYNAMO_DISABLE=1
export PYTHONPATH=$SRC_DIR:$SRC_DIR/libero_pro
# Pin the model for both arms; the paper's sim id is dead (404) and the
# pro-tier successor keeps arms comparable to the published pro numbers.
export SPARK_GEMINI_MODEL=gemini-3.1-pro-preview
# Trial independence: no in-process BT reuse across init states. Each
# trial pays (and logs) its own calls; the API's implicit prefix caching
# still applies, which is where the savings actually come from.
export SPARK_GEMINI_CACHE=0
cd "$SRC_DIR"

run_arm () {
    local arm=$1; shift
    local log=$LOG_DIR/p0a_${SUITE}_${arm}.log
    echo "=== P0-a $SUITE arm=$arm trials=$TRIALS started $(date) ===" | tee -a "$log"
    "$@" $PY -u -m spark_bench.run_spark_libero_pro_fair \
        --suite "$SUITE" --perturbation all --num-trials "$TRIALS" \
        --output-dir "$OUT_BASE/p0a_${SUITE}_${arm}" >> "$log" 2>&1
    local rc=$?
    echo "=== arm=$arm finished rc=$rc $(date) ===" | tee -a "$log"
    return $rc
}

run_arm baseline env SPARK_SHADOW_SIM=0
run_arm shadow_k2 env SPARK_SHADOW_SIM=1 SPARK_SHADOW_K=2
# E1, the no-adaptation control. experiments.tex disclosed that the on-disk BT
# few-shot library "was active and grows within a run, so later trials plan
# with richer prompts". The paper's Problem Setup defines the no-adaptation
# protocol as seeing no trial of a task before that task is scored, so the
# baseline arm above violates the definition the paper uses to exclude Zetta
# and Harness VLA. SPARK_BT_FROZEN serves no few-shot context and records
# nothing, so trial N plans with exactly the context trial 1 had. Run this arm
# and report it as the protocol-clean number.
run_arm frozen env SPARK_SHADOW_SIM=0 SPARK_BT_FROZEN=1

echo "[p0a] suite $SUITE complete $(date)"

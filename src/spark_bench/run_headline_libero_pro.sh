#!/bin/bash
# The unassisted LIBERO-PRO headline protocol (Table 1, ICLR 2027 submission), as one command.
# Every env var here was set by hand for the September sweeps and lived nowhere in the repo;
# this script is the record. Usage:
#   src/spark_bench/run_headline_libero_pro.sh <suite> <perturbation> [num_trials] [extra runner args...]
# Requires: the LIBERO evaluation env active (openvla_env locally, libero_env on Delta),
# PYTHONPATH=src:src/libero_pro, the Gemini key file, and MUJOCO_GL=egl for headless rendering.
set -u
SUITE=${1:?suite: object|goal|spatial}
PERT=${2:?perturbation: position|task|all}
TRIALS=${3:-50}
shift 3 2>/dev/null || shift $#
export SPARK_GEMINI_MODEL=${SPARK_GEMINI_MODEL:-gemini-3.7-flash}   # planner model of the headline sweep
export SPARK_GEMINI_CACHE=0        # in-process plan cache off: every trial plans fresh
export SPARK_NO_ORACLE=1           # no mid-episode simulator success reads (the unassisted row)
export SPARK_BT_FROZEN=1           # frozen cross-run library: no few-shot retrieval, no write-back
export MUJOCO_GL=${MUJOCO_GL:-egl}
export PYTHONPATH=${PYTHONPATH:-src:src/libero_pro}
exec python -m spark_bench.run_spark_libero_pro_fair --suite "$SUITE" --perturbation "$PERT" \
  --num-trials "$TRIALS" --no-bddl-hints "$@"

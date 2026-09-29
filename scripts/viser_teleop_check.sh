#!/bin/bash
# End-to-end sanity for the robots_realtime viser-IK teleop integration.
# Runs in 4 phases. Stops at the first failure with a clear message.
# Does NOT actually touch the robot - just verifies the install, imports,
# config files, and that the SPARK routes import cleanly.
#
# Usage: ./viser_teleop_check.sh
#
# After this passes, the real smoke test is:
#   1. ~/spark/scripts/spark_server.sh start
#   2. Open SPARK UI, click "Viser IK Teleop"
#   3. Open the viser URL, drag the gizmo, watch the arm move
#   4. Click "Stop Viser"
#   5. SPARK gamepad teleop works again

set -u

RR_ROOT="$HOME/robots_realtime"
RR_PY="$RR_ROOT/.venv/bin/python"
RR_CONFIG="$RR_ROOT/configs/franka/franka_spark_viser_teleop.yaml"
SPARK_CONDA_PY="$HOME/miniconda3/envs/spark_conda/bin/python"

red() { echo -e "\033[31m$*\033[0m"; }
green() { echo -e "\033[32m$*\033[0m"; }
yellow() { echo -e "\033[33m$*\033[0m"; }

phase() { echo ""; yellow "=== $* ==="; }

phase "Phase 1: robots_realtime venv exists"
if [ ! -x "$RR_PY" ]; then
    red "FAIL: $RR_PY not found"
    echo "Run: cd $RR_ROOT && uv venv --python 3.11 && uv pip install -e ."
    exit 1
fi
green "OK: $RR_PY"
"$RR_PY" --version

phase "Phase 2: robots_realtime imports cleanly"
"$RR_PY" -c "
import sys
try:
    import robots_realtime
    import panda_py
    import pyroki
    import viser
    print('rr version:', robots_realtime.__version__ if hasattr(robots_realtime, '__version__') else '(unversioned)')
    print('panda_py path:', panda_py.__file__)
    print('libfranka via panda_py:', getattr(panda_py, '__version__', '(unknown)'))
    print('pyroki path:', pyroki.__file__)
    print('viser path:', viser.__file__)
    print('ALL_IMPORTS_OK')
except Exception as e:
    print('IMPORT_FAIL:', type(e).__name__, e, file=sys.stderr)
    sys.exit(1)
"
if [ $? -ne 0 ]; then red "FAIL: imports broken"; exit 1; fi
green "OK: imports clean"

phase "Phase 3: SPARK teleop config files present"
for f in "$RR_CONFIG" "$RR_ROOT/robot_configs/franka/franka_spark_fr3.yaml"; do
    if [ ! -f "$f" ]; then red "FAIL: missing $f"; exit 1; fi
    green "OK: $f"
done

phase "Phase 4: SPARK routes import cleanly"
"$SPARK_CONDA_PY" -c "
import sys
sys.path.insert(0, '$HOME/spark/src')
try:
    from spark_real.routes import viser_teleop, state
    assert hasattr(state, 'viser_teleop_proc'), 'state.viser_teleop_proc missing'
    assert hasattr(viser_teleop, 'router'), 'viser_teleop.router missing'
    routes = [r.path for r in viser_teleop.router.routes]
    print('mounted routes:', routes)
    print('ROUTES_OK')
except Exception as e:
    print('ROUTE_FAIL:', type(e).__name__, e, file=sys.stderr)
    sys.exit(1)
"
if [ $? -ne 0 ]; then red "FAIL: SPARK routes broken"; exit 1; fi
green "OK: SPARK routes register"

phase "Phase 5 (optional, manual): hardware smoke test"
echo "Run this ONLY when SPARK server is stopped and the FR3 is unlocked:"
echo ""
echo "  $RR_PY -c 'import panda_py; p = panda_py.Panda(\"${FRANKA_IP:-172.16.0.2}\"); print(p.get_state())'"
echo ""
echo "Expected output: a State object."
echo "If you get 'IncompatibleVersionException' your FR3 firmware needs a newer panda-py:"
echo ""
echo "  cd $RR_ROOT && uv pip install --force-reinstall \\"
echo "    https://github.com/JeanElsner/panda-py/releases/download/v0.8.1/panda_python-0.8.1+libfranka.0.13.3-cp311-cp311-manylinux_2_17_x86_64.manylinux2014_x86_64.whl"

echo ""
green "All non-hardware checks passed. Ready to wire from UI."

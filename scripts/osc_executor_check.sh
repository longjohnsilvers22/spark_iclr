#!/bin/bash
# End-to-end sanity for the robots_realtime OSC executor backend.
# Runs in 5 phases. Stops at the first failure with a clear message.
# Does NOT actually touch the robot - just verifies the install, imports,
# config files, agent loads with JIT warm-up, and SPARK routes register.
#
# Usage: ./osc_executor_check.sh
#
# After this passes, the real smoke test is:
#   1. ~/spark/scripts/spark_server.sh start
#   2. POST /api/control/osc/start (releases FCI, spawns rr-session)
#   3. POST /api/control/osc/move_to_pose {position:[..],wxyz:[..],duration_s:2}
#   4. POST /api/control/osc/stop  (reclaims FCI for SPARK)

set -u

RR_ROOT="$HOME/robots_realtime"
RR_PY="$RR_ROOT/.venv/bin/python"
RR_CONFIG="$RR_ROOT/configs/franka/franka_spark_osc_executor.yaml"
RR_ROBOT_CONFIG="$RR_ROOT/robot_configs/franka/franka_spark_fr3_with_gripper.yaml"
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

phase "Phase 2: OSC executor config files present"
for f in "$RR_CONFIG" "$RR_ROBOT_CONFIG"; do
    if [ ! -f "$f" ]; then red "FAIL: missing $f"; exit 1; fi
    green "OK: $f"
done

phase "Phase 3: rr-side agent imports + pyroki JIT loads"
# This actually constructs the agent (HTTP server bound, pyroki warm-up
# done), then tears it down immediately so the HTTP port doesn't linger.
"$RR_PY" -c "
import sys
import threading
import time
try:
    import pyroki  # noqa
    import robot_descriptions  # noqa
    from robots_realtime.agents.client.franka_osc_cartesian_target_agent import (
        FrankaOscCartesianTargetAgent,
    )
    # Bind to a throwaway port (not 9009) so this verifier doesn't
    # conflict with a real running rr-session.
    agent = FrankaOscCartesianTargetAgent(
        http_host='127.0.0.1', http_port=19009,
    )
    # Smoke-test FK from a zero-q.
    import numpy as np
    xyz, wxyz = agent._fk(np.zeros(7, dtype=np.float32))
    assert len(xyz) == 3 and len(wxyz) == 4, 'FK output bad shape'
    print('FK_OK xyz=', xyz, 'wxyz=', wxyz)
    agent.close()
    print('AGENT_OK')
except Exception as e:
    print('AGENT_FAIL:', type(e).__name__, e, file=sys.stderr)
    import traceback
    traceback.print_exc()
    sys.exit(1)
"
if [ $? -ne 0 ]; then red "FAIL: agent broken"; exit 1; fi
green "OK: agent loads + FK works"

phase "Phase 4: rr-session can parse the OSC YAML (no robot connection)"
# load_session() will instantiate node classes; the RobotNode is lazy
# (robot_config is stored, not resolved) so this doesn't try to reach
# the FR3. The AgentNode is also lazy - only setup() builds the agent.
"$RR_PY" -c "
import sys
try:
    from robots_realtime.runtime.config import load_session
    sess = load_session('$RR_CONFIG')
    print('session_loaded nodes=',
          [n.node_name if hasattr(n, 'node_name') else n._node.name
           for n in sess._hosts])
    print('SESSION_OK')
except Exception as e:
    print('SESSION_FAIL:', type(e).__name__, e, file=sys.stderr)
    import traceback
    traceback.print_exc()
    sys.exit(1)
"
if [ $? -ne 0 ]; then red "FAIL: rr-session YAML broken"; exit 1; fi
green "OK: rr-session config parses"

phase "Phase 5: SPARK routes import cleanly"
"$SPARK_CONDA_PY" -c "
import sys
sys.path.insert(0, '$HOME/spark/src')
try:
    from spark_real.routes import osc_executor, state
    assert hasattr(state, 'osc_executor_proc'), 'state.osc_executor_proc missing'
    assert hasattr(state, 'osc_executor_log_fh'), 'state.osc_executor_log_fh missing'
    assert hasattr(osc_executor, 'router'), 'osc_executor.router missing'
    routes = sorted(r.path for r in osc_executor.router.routes)
    print('mounted routes:', routes)
    expected = {
        '/api/control/osc/start',
        '/api/control/osc/stop',
        '/api/control/osc/status',
        '/api/control/osc/move_to_pose',
        '/api/control/osc/gripper',
        '/api/control/osc/log',
    }
    missing = expected - set(routes)
    assert not missing, f'missing routes: {missing}'

    # And the executor-side helper must be importable too.
    from spark_real.control.score_executor import (
        _osc_backend_enabled, _maybe_osc_move_linear, _osc_move_linear,
    )
    # Default is franky.
    import os
    os.environ.pop('SPARK_EXECUTOR_BACKEND', None)
    assert _osc_backend_enabled() is False, 'default backend should be franky'
    os.environ['SPARK_EXECUTOR_BACKEND'] = 'osc'
    assert _osc_backend_enabled() is True, 'osc env not picked up'
    os.environ.pop('SPARK_EXECUTOR_BACKEND', None)

    # _robot_ready must reject when osc proc is alive.
    from spark_real.routes import core
    class _FakeProc:
        def poll(self): return None  # alive
    state.osc_executor_proc = _FakeProc()
    ok, msg = core._robot_ready(None)  # pipeline=None reaches the proc check first
    assert ok is False, f'expected reject when osc proc alive, got ok=True'
    assert 'OSC' in msg or 'osc' in msg, f'expected OSC mention in msg, got {msg!r}'
    state.osc_executor_proc = None
    print('BACKEND_GATE_OK')

    print('ROUTES_OK')
except Exception as e:
    print('ROUTE_FAIL:', type(e).__name__, e, file=sys.stderr)
    import traceback
    traceback.print_exc()
    sys.exit(1)
"
if [ $? -ne 0 ]; then red "FAIL: SPARK routes broken"; exit 1; fi
green "OK: SPARK routes register + executor backend gate works"

echo ""
green "All non-hardware checks passed. To run the real smoke test:"
echo ""
echo "  1. ~/spark/scripts/spark_server.sh start"
echo "  2. curl -X POST http://localhost:8888/api/control/osc/start"
echo "  3. curl http://localhost:8888/api/control/osc/status"
echo "  4. curl -X POST http://localhost:8888/api/control/osc/move_to_pose \\"
echo "       -H 'Content-Type: application/json' \\"
echo "       -d '{\"position\":[0.4,0.0,0.4],\"wxyz\":[0,1,0,0],\"duration_s\":2.0}'"
echo "  5. curl -X POST http://localhost:8888/api/control/osc/stop"

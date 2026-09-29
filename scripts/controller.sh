#!/bin/bash
# Switch between franky (teleop) and bamboo (execution) controllers.
# Usage:
#   controller.sh franky   - kill bamboo, franky takes FCI
#   controller.sh bamboo   - kill franky users, start bamboo C++ node
#   controller.sh status   - show what's running
#   controller.sh stop     - kill everything

BAMBOO_BIN="$HOME/spark/src/external_controllers/bamboo/controller/build/bamboo_control_node"
BAMBOO_LD="/opt/openrobots/lib:$HOME/spark/src/external_controllers/bamboo/install/lib"
ROBOT_IP="${FRANKA_IP:-172.16.0.2}"
BAMBOO_PORT=5555

case "${1:-status}" in
  franky)
    echo "Switching to franky (teleop mode)..."
    pkill -f bamboo_control_node 2>/dev/null
    sleep 1
    echo "Bamboo stopped. FCI free for franky."
    echo "Start teleop via server or direct franky script."
    ;;
  bamboo)
    echo "Switching to bamboo (execution mode)..."
    pkill -f bamboo_control_node 2>/dev/null
    sleep 1
    LD_LIBRARY_PATH="$BAMBOO_LD" nohup "$BAMBOO_BIN" \
      -r "$ROBOT_IP" -p "$BAMBOO_PORT" -g franka \
      > /tmp/bamboo.log 2>&1 &
    echo "Bamboo PID: $!"
    sleep 5
    if grep -q "Server listening" /tmp/bamboo.log 2>/dev/null; then
      echo "Bamboo ready on port $BAMBOO_PORT"
    else
      echo "Bamboo not ready yet. Check /tmp/bamboo.log"
    fi
    ;;
  stop)
    pkill -f bamboo_control_node 2>/dev/null
    echo "All controllers stopped. FCI free."
    ;;
  status)
    if pgrep -f bamboo_control_node > /dev/null 2>&1; then
      echo "Active: BAMBOO (port $BAMBOO_PORT)"
    else
      echo "Active: NONE (FCI free for franky)"
    fi
    ;;
  *)
    echo "Usage: $0 {franky|bamboo|stop|status}"
    ;;
esac

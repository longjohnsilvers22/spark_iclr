#!/bin/bash
# SPARK server lifecycle wrapper. Use this instead of `pkill -9` - that
# bricks the Kinect depth MCU and forces a 12V cycle.
#
# Stop sequence (in order):
#   1. POST /api/shutdown - synchronous in-process cleanup (best path)
#   2. SIGTERM via pkill - fires the signal handler in server.py
#   3. SIGINT via pkill - same path as Ctrl-C
#   4. Refuse to escalate further unless --force is passed; SIGKILL
#      bypasses all cleanup and bricks the Kinects.
#
# Usage:
#   spark_server.sh start [extra-args-to-server]   e.g. start --no-robot
#   spark_server.sh stop  [--force]
#   spark_server.sh restart
#   spark_server.sh status
#   spark_server.sh reload-tact [--zero]   re-probe tactile serial and re-zero (optional) on running server, no restart.
#       Exit codes: 1 server not running
#       2 endpoint unreachable,
#       3 server up but no sensors found.

set -u

PORT="${SPARK_PORT:-8888}"
PATTERN='python -m spark_real\.server'
SRC_DIR="$HOME/spark/src"
CONDA_PY="$HOME/miniconda3/envs/spark_conda/bin/python"
LOG_DIR="$HOME/spark/src/spark_real/output/logs"
LOG_FILE="$LOG_DIR/spark_server.log"

cmd="${1:-status}"

is_running() {
    pgrep -f "$PATTERN" >/dev/null 2>&1
}

pids() {
    pgrep -f "$PATTERN" | tr '\n' ' '
}

wait_dead() {
    local timeout=$1
    local elapsed=0
    while is_running; do
        sleep 1
        elapsed=$((elapsed + 1))
        if [ "$elapsed" -ge "$timeout" ]; then
            return 1
        fi
    done
    return 0
}

case "$cmd" in
status)
    if is_running; then
        echo "spark_server running: pid(s) $(pids)"
        echo "status: $(curl -sS -m 3 "http://localhost:$PORT/api/status" 2>/dev/null | head -c 300)"
        exit 0
    fi
    echo "spark_server not running"
    exit 1
    ;;

start)
    if is_running; then
        echo "already running: pid(s) $(pids)"
        exit 0
    fi
    shift || true
    if [ "${1:-}" = "--" ]; then shift; fi  # bare -- would crash tyro at boot
    cd "$SRC_DIR" || { echo "missing $SRC_DIR" >&2; exit 1; }
    mkdir -p "$LOG_DIR"
    # Rotate the previous log so we don't lose context if launches stack.
    if [ -f "$LOG_FILE" ]; then
        mv "$LOG_FILE" "$LOG_FILE.prev"
    fi
    echo "starting spark_server on port $PORT (log: $LOG_FILE)..."
    # Family default. Override per-launch with
    # SPARK_ROBOT=franka scripts/spark_server.sh start, or pass --robot ...
    # as an extra arg (later flags win).
    ROBOT_FAMILY="${SPARK_ROBOT:-ur10e}"
    DISPLAY=:1 XAUTHORITY=/run/user/1000/gdm/Xauthority \
    nohup "$CONDA_PY" -m spark_real.server \
            --robot "$ROBOT_FAMILY" --auto-unlock --port "$PORT" "$@" \
            </dev/null >"$LOG_FILE" 2>&1 &
    echo "started: pid $!"
    ;;

stop)
    force=0
    if [ "${2:-}" = "--force" ]; then force=1; fi

    if ! is_running; then
        echo "not running"
        exit 0
    fi

    echo "stop step 1: POST /api/shutdown"
    if curl -sS -m 5 -X POST "http://localhost:$PORT/api/shutdown" \
            >/dev/null 2>&1; then
        if wait_dead 15; then
            echo "  clean exit via /api/shutdown"
            exit 0
        fi
    else
        echo "  endpoint did not respond"
    fi

    echo "stop step 2: SIGTERM (signal handler in server.py)"
    pkill -TERM -f "$PATTERN" 2>/dev/null
    if wait_dead 15; then
        echo "  clean exit via SIGTERM"
        exit 0
    fi

    echo "stop step 3: SIGINT (Ctrl-C equivalent)"
    pkill -INT -f "$PATTERN" 2>/dev/null
    if wait_dead 10; then
        echo "  clean exit via SIGINT"
        exit 0
    fi

    if [ "$force" -ne 1 ]; then
        echo ""
        echo "REFUSED: server didn't exit on TERM/INT after 40s." >&2
        echo "  SIGKILL will brick the Kinect depth MCU and force a" >&2
        echo "  12V cycle. Investigate why shutdown is hanging." >&2
        echo "  If you really must: $0 stop --force" >&2
        echo "  Stuck pid(s): $(pids)" >&2
        exit 2
    fi

    echo "stop step 4: SIGKILL (--force) - Kinects WILL likely brick"
    pkill -KILL -f "$PATTERN" 2>/dev/null
    wait_dead 5
    ;;

restart)
    "$0" stop || exit $?
    sleep 1
    exec "$0" start "${@:2}"
    ;;

reload-tact)
    if ! is_running; then
        echo "server not running. restart to load cameras and tactile"
        exit 1
    fi

    # -s quiet progress, -S still show errors; -m 10 give up after 10 s.
    # Endpoints always answer HTTP 200 with the result in the JSON body,
    # so curl's exit only covers transport -- gate on "connected" below.
    out=$(curl -sS -m 10 -X POST "http://localhost:$PORT/api/tactile/reconnect") \
    || { echo "reconnect failed. is the server on port $PORT?" >&2; exit 2; }
    echo "$out"
    if ! grep -q '"connected":true' <<<"$out"; then
        echo "server up but no tactile sensors found. check USB and" >&2
        echo "port owners (fuser -v /dev/ttyUSB*)" >&2
        exit 3
    fi
    echo "tactile sensors reloaded"

    if [ "${2:-}" = "--zero" ]; then
        echo "  zeroing tactile sensors. make sure nothing is in contact with the pads"
        sleep 2
        out=$(curl -sS -m 10 -X POST "http://localhost:$PORT/api/tactile/rebaseline") \
        || { echo "sensor zeroing failed. make sure nothing is contacting the pads" >&2; exit 2;}
        echo "  $out"
        echo "  tactile sensors zeroed"
    fi
    ;;

*)
    echo "usage: $0 {start|stop|restart|status|reload-tact} [--force] [--zero]" >&2
    exit 1
    ;;
esac

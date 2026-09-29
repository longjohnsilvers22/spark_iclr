#!/bin/bash
# Spark server supervisor - auto-restart on libfranka C++ crash
# (`terminate called without an active exception` kills the process).
# Stops thrashing by limiting restart attempts within a window.
#
# Usage:
#   ./spark_server_supervisor.sh start     # run in foreground (Ctrl-C to stop)
#   ./spark_server_supervisor.sh bgstart   # daemonize via nohup
#   ./spark_server_supervisor.sh stop      # stop both server AND supervisor

set -u

WRAPPER="$HOME/spark/scripts/spark_server.sh"
SUPERVISOR_PID_FILE="/tmp/spark_server_supervisor.pid"
LOG_FILE="$HOME/spark/src/spark_real/output/logs/supervisor.log"
MAX_RESTARTS_PER_HOUR=20
MIN_GAP_S=5

log() {
    echo "[$(date +'%H:%M:%S')] $*" | tee -a "$LOG_FILE"
}

is_server_running() {
    pgrep -f 'python -m spark_real\.server' >/dev/null 2>&1
}

stop_all() {
    log "supervisor: stop_all called"
    if [ -f "$SUPERVISOR_PID_FILE" ]; then
        local sup_pid=$(cat "$SUPERVISOR_PID_FILE")
        if [ -n "$sup_pid" ]; then
            kill -TERM "$sup_pid" 2>/dev/null
            sleep 1
        fi
        rm -f "$SUPERVISOR_PID_FILE"
    fi
    "$WRAPPER" stop
}

supervisor_loop() {
    echo $$ > "$SUPERVISOR_PID_FILE"
    log "supervisor: starting (pid=$$, max_restarts/hr=$MAX_RESTARTS_PER_HOUR)"

    local restart_times=()
    local first_iter=1

    while true; do
        if [ "$first_iter" -eq 1 ] || ! is_server_running; then
            if [ "$first_iter" -ne 1 ]; then
                log "supervisor: server NOT running - restarting"
                # Rate-limit
                local now=$(date +%s)
                restart_times+=($now)
                # Keep only restarts within the last hour
                local cutoff=$((now - 3600))
                local kept=()
                for t in "${restart_times[@]}"; do
                    if [ "$t" -gt "$cutoff" ]; then
                        kept+=("$t")
                    fi
                done
                restart_times=("${kept[@]}")
                if [ "${#restart_times[@]}" -gt "$MAX_RESTARTS_PER_HOUR" ]; then
                    log "supervisor: too many restarts in last hour (${#restart_times[@]}); waiting longer"
                    sleep 60
                fi
                sleep "$MIN_GAP_S"
            fi
            log "supervisor: starting server"
            "$WRAPPER" start >/dev/null
            first_iter=0
        fi
        sleep 5
    done
}

case "${1:-status}" in
start)
    supervisor_loop
    ;;
bgstart)
    if [ -f "$SUPERVISOR_PID_FILE" ] && kill -0 "$(cat $SUPERVISOR_PID_FILE)" 2>/dev/null; then
        echo "supervisor already running (pid $(cat $SUPERVISOR_PID_FILE))"
        exit 0
    fi
    nohup "$0" start </dev/null >/dev/null 2>&1 &
    echo "supervisor started in background (pid $!)"
    ;;
stop)
    stop_all
    ;;
status)
    if [ -f "$SUPERVISOR_PID_FILE" ] && kill -0 "$(cat $SUPERVISOR_PID_FILE)" 2>/dev/null; then
        echo "supervisor running (pid $(cat $SUPERVISOR_PID_FILE))"
    else
        echo "supervisor NOT running"
    fi
    "$WRAPPER" status
    ;;
*)
    echo "usage: $0 {start|bgstart|stop|status}"
    exit 1
    ;;
esac

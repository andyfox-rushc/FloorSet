#!/bin/bash
# Waits for the current training run to finish (or nearly finish), then
# automatically runs the 5-case eval. Runs detached, independent of any
# Claude session, so a long wait (hours) doesn't depend on the session
# staying open.
set -uo pipefail
cd "$(dirname "$0")/.."
source ../venv/bin/activate
export LD_LIBRARY_PATH=/usr/lib/x86_64-linux-gnu

LOGDIR=logs
AUTO_LOG="$LOGDIR/auto_eval.log"
NEAR_DONE_ELAPSED_S=82800   # 23h of the 24h budget

echo "$(date -Iseconds) auto-eval watcher started (PID $$)" >> "$AUTO_LOG"

while true; do
    active_log=$(ls -t "$LOGDIR"/train_algfix_run*.log "$LOGDIR"/train_autorelaunch_*.log 2>/dev/null | head -1)
    if [ -n "$active_log" ]; then
        last_iter_line=$(grep "^iter " "$active_log" | tail -1)
        elapsed=$(echo "$last_iter_line" | grep -oP 'elapsed=\K[0-9.]+')
        done_now=false
        if grep -q "time limit of" "$active_log"; then
            done_now=true
        elif echo "$last_iter_line" | grep -qE '^iter ([0-9]+)/\1 '; then
            done_now=true
        elif [ -n "${elapsed:-}" ] && awk -v e="$elapsed" -v t="$NEAR_DONE_ELAPSED_S" 'BEGIN{exit !(e>=t)}'; then
            done_now=true
        fi

        if [ "$done_now" = true ]; then
            echo "$(date -Iseconds) training near/at done ($active_log: $last_iter_line) - running eval" >> "$AUTO_LOG"
            sleep 30  # let any in-flight checkpoint save finish
            python scripts/run_first5_eval.py >> "$AUTO_LOG" 2>&1
            echo "$(date -Iseconds) EVAL_DONE - see first5_results_now.json" >> "$AUTO_LOG"
            break
        fi
    fi
    sleep 300
done

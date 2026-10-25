#!/bin/bash
# Auto-relaunches rl.train from the latest checkpoint if it's not running.
# Runs in the background (nohup+disown); checks every 3 min. Exits once the
# newest training log shows the run finished (last iteration or time limit),
# so a completed run isn't restarted forever.
# Also keeps scripts/anchor_progress_watcher.py running (images to
# $PROGRESS_DIR) and scripts/checkpoint_eval_watcher.py (results_checkpoints/). Usage: train_watchdog.sh [extra rl.train args]
set -uo pipefail
cd "$(dirname "$0")/.."
source ../venv/bin/activate
export LD_LIBRARY_PATH=/usr/lib/x86_64-linux-gnu

LOGDIR=logs
mkdir -p "$LOGDIR"
WATCHDOG_LOG="$LOGDIR/train_watchdog.log"
EXTRA_ARGS=("$@")
PROGRESS_DIR=${PROGRESS_DIR:-progress_util_curriculum}

# Only one watchdog at a time (e.g. the Windows logon task plus a manual start).
# Children are started with fd 9 closed so they don't hold the lock.
exec 9> "$LOGDIR/.watchdog.lock"
flock -n 9 || { echo "$(date -Iseconds) another watchdog is running - exiting" >> "$WATCHDOG_LOG"; exit 0; }

echo "$(date -Iseconds) watchdog started (PID $$) extra args: ${EXTRA_ARGS[*]:-none}" >> "$WATCHDOG_LOG"

run_finished() {
    local latest
    latest=$(ls -t "$LOGDIR"/train_*.log 2>/dev/null | head -1)
    [ -n "$latest" ] || return 1
    tail -5 "$latest" | grep -qE "time limit of .* reached|^iter ([0-9]+)/\1 "
}

while true; do
    if ! pgrep -f "anchor_progress_watcher.py" > /dev/null; then
        mkdir -p "$PROGRESS_DIR"
        nohup python -u scripts/anchor_progress_watcher.py --out-dir "$PROGRESS_DIR" \
            --poll-seconds 30 --time-budget 15.0 --anneal-budget 8.0 \
            >> "$LOGDIR/anchor_progress_watchdog.log" 2>&1 9>&- &
        disown
        echo "$(date -Iseconds) started anchor_progress_watcher" >> "$WATCHDOG_LOG"
    fi
    if ! pgrep -f "checkpoint_eval_watcher.py" > /dev/null; then
        nohup python -u scripts/checkpoint_eval_watcher.py --start 4000 \
            >> "$LOGDIR/checkpoint_eval_watcher.log" 2>&1 9>&- &
        disown
        echo "$(date -Iseconds) started checkpoint_eval_watcher" >> "$WATCHDOG_LOG"
    fi
    if ! pgrep -f "python -u -m rl.train" > /dev/null; then
        if run_finished; then
            echo "$(date -Iseconds) training finished - watchdog exiting" >> "$WATCHDOG_LOG"
            exit 0
        fi
        ts=$(date +%Y%m%d_%H%M%S)
        newlog="$LOGDIR/train_autorelaunch_${ts}.log"
        echo "$(date -Iseconds) rl.train not running - relaunching -> $newlog" >> "$WATCHDOG_LOG"
        nohup python -u -m rl.train --iterations 20000 --episodes-per-iter 4 --ppo-epochs 4 \
            --resume checkpoints/policy.pt --checkpoint checkpoints/policy.pt \
            --checkpoint-every 250 --threads 4 "${EXTRA_ARGS[@]}" > "$newlog" 2>&1 9>&- &
        disown
        sleep 30 9>&-
    fi
    sleep 180 9>&-
done

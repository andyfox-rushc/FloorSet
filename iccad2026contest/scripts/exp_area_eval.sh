#!/bin/bash
# Scores each area-weight arm at 13000 and 14000 (100-case greedy/sampled scan at 0.70).
cd "$(dirname "$0")/.."
source ../venv/bin/activate
export LD_LIBRARY_PATH=/usr/lib/x86_64-linux-gnu
for it in 013000 014000; do
    for w in 1 3 6; do
        until [ -f checkpoints/exp_area$w/history/policy_iter$it.pt.meta.json ]; do sleep 60; done
    done
    for w in 1 3 6; do
        python -u scripts/canvas_scan.py --configs 0.7:4:1:0:1:3 --random 0 --sampled 4 --workers 3 \
            --checkpoint checkpoints/exp_area$w/history/policy_iter$it.pt --out exp_area${w}_iter$it.json 2>&1 \
            | grep "^U=" | sed "s/^/iter $it area x$w: /"
    done
done
echo "EXP_EVAL_DONE"

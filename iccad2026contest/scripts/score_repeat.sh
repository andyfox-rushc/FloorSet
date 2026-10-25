#!/bin/bash
# 10-04: 3 repeats of the full pipeline on 100 designs for the top from-scratch
# checkpoints and the installed policy (noise check before picking the best).
cd "$(dirname "$0")/.."
source ../venv/bin/activate
export LD_LIBRARY_PATH=/usr/lib/x86_64-linux-gnu
for r in 1 2 3; do
    for it in 006000 014000 012500 005500; do
        FLOORSET_FRAME_UTIL=0.80 python -u scripts/polish_bench.py pipeline --checkpoint checkpoints/scratch/history/policy_iter$it.pt \
            --splits 2:slice+repack:1 --workers 4 --out allck/rep${r}_scratch_iter$it.json 2>&1 | grep "cost" | sed "s/^/rep $r scratch $it: /"
    done
    python -u scripts/polish_bench.py pipeline --checkpoint checkpoints/policy.pt --splits 2:slice+repack:1 --workers 4 \
        --out allck/rep${r}_installed12000.json 2>&1 | grep "cost" | sed "s/^/rep $r installed 12000: /"
done
echo "REPEAT_DONE"

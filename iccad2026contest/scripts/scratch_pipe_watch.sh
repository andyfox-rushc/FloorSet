#!/bin/bash
# Full-pipeline 100-design eval (2 s fine-tune + 1 s slice+repack, frame 0.80)
# of the from-scratch run every 2000 iterations, from 8000 to 20000.
cd "$(dirname "$0")/.."
source ../venv/bin/activate
export LD_LIBRARY_PATH=/usr/lib/x86_64-linux-gnu
H=checkpoints/scratch/history
for it in 008000 010000 012000 014000 016000 018000 020000; do
    until [ -f $H/policy_iter$it.pt.meta.json ]; do sleep 120; done
    FLOORSET_FRAME_UTIL=0.80 python -u scripts/polish_bench.py pipeline --checkpoint $H/policy_iter$it.pt \
        --splits 2:slice+repack:1 --workers 2 --out scratch_pipe_iter$it.json 2>&1 | grep "cost" | sed "s/^/pipeline iter $it: /"
done
echo "SCRATCH_PIPE_DONE"

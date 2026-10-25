#!/bin/bash
# Scores the 09-29 arms (slice-reward, edge-last) with the full current
# pipeline (2 s fine-tune + 1 s slice+repack) on the 100 validation designs:
# iter 12000 baseline once (both placement modes), then each arm at 14000 and 16000.
cd "$(dirname "$0")/.."
source ../venv/bin/activate
export LD_LIBRARY_PATH=/usr/lib/x86_64-linux-gnu
bench() {  # label checkpoint out [env assignment]
    env $4 python -u scripts/polish_bench.py pipeline --checkpoint "$2" --splits 2:slice+repack:1 \
        --workers 2 --out "$3" 2>&1 | grep "cost" | sed "s/^/$1: /"
}
bench "iter 12000 baseline" checkpoints/policy.pt exp_pack_base_iter012000.json
bench "iter 12000 baseline, edge-last" checkpoints/policy.pt exp_pack_base_edge_iter012000.json FLOORSET_EDGE_LAST=1
for it in 014000 016000; do
    for a in slice edge; do
        until [ -f checkpoints/exp_$a/history/policy_iter$it.pt.meta.json ]; do sleep 120; done
    done
    bench "iter $it slice-reward" checkpoints/exp_slice/history/policy_iter$it.pt exp_pack_slice_iter$it.json
    bench "iter $it edge-last" checkpoints/exp_edge/history/policy_iter$it.pt exp_pack_edge_iter$it.json FLOORSET_EDGE_LAST=1
done
echo "EXP_PACK_EVAL_DONE"

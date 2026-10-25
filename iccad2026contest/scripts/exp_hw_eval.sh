#!/bin/bash
# Scores the 10-02 HPWL-weight arms (FLOORSET_HPWL_WEIGHT=2, big-first 0.4 +
# frame 0.80; A = "scale" x2 ordering, B = "seed" ordering) with the full
# pipeline (2 s fine-tune + 1 s slice+repack) on the 100 validation designs:
# 22000 under training load, 24000 quiet next to installed iter 12000.
cd "$(dirname "$0")/.."
source ../venv/bin/activate
export LD_LIBRARY_PATH=/usr/lib/x86_64-linux-gnu
bench() {  # label checkpoint out workers [env assignments]
    env $5 python -u scripts/polish_bench.py pipeline --checkpoint "$2" --splits 2:slice+repack:1 \
        --workers $4 --out "$3" 2>&1 | grep "cost" | sed "s/^/$1: /"
}
A="FLOORSET_BIG_MODE=scale FLOORSET_BIG_FRAC=0.4 FLOORSET_BIG_WEIGHT=2 FLOORSET_FRAME_UTIL=0.80"
B="FLOORSET_BIG_FRAC=0.4 FLOORSET_BIG_WEIGHT=0.5 FLOORSET_FRAME_UTIL=0.80"
for it in 022000 024000; do
    for a in hwA hwB; do
        until [ -f checkpoints/exp_$a/history/policy_iter$it.pt.meta.json ]; do sleep 120; done
    done
    w=2; [ $it = 024000 ] && { while pgrep -f "rl.train" > /dev/null; do sleep 30; done; w=4; }
    bench "iter $it arm A (scale x2, hpwl x2)" checkpoints/exp_hwA/history/policy_iter$it.pt exp_hwA_iter$it.json $w "$A"
    bench "iter $it arm B (seed, hpwl x2)" checkpoints/exp_hwB/history/policy_iter$it.pt exp_hwB_iter$it.json $w "$B"
done
bench "iter 12000 installed" checkpoints/policy.pt exp_hw_ref_base12000.json 4
echo "EXP_HW_EVAL_DONE"

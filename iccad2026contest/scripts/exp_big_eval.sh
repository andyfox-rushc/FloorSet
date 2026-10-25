#!/bin/bash
# Scores the 10-02 big-first + frame arm (FLOORSET_BIG_FRAC=0.4,
# FLOORSET_FRAME_UTIL=0.80) with the full pipeline (2 s fine-tune + 1 s
# slice+repack) on the 100 validation designs, at 18000 (under training load)
# and 20000 (quiet, next to installed iter 12000 and exp_frame 16000).
cd "$(dirname "$0")/.."
source ../venv/bin/activate
export LD_LIBRARY_PATH=/usr/lib/x86_64-linux-gnu
bench() {  # label checkpoint out workers [env assignments]
    env $5 python -u scripts/polish_bench.py pipeline --checkpoint "$2" --splits 2:slice+repack:1 \
        --workers $4 --out "$3" 2>&1 | grep "cost" | sed "s/^/$1: /"
}
H=checkpoints/exp_big/history
ARM="FLOORSET_BIG_FRAC=0.4 FLOORSET_FRAME_UTIL=0.80"
until [ -f $H/policy_iter018000.pt.meta.json ]; do sleep 120; done
bench "iter 18000 big-first arm" $H/policy_iter018000.pt exp_big_iter018000.json 2 "$ARM"
until [ -f $H/policy_iter020000.pt.meta.json ]; do sleep 120; done
while pgrep -f "rl.train" > /dev/null; do sleep 30; done
bench "iter 20000 big-first arm" $H/policy_iter020000.pt exp_big_iter020000.json 4 "$ARM"
bench "exp_frame 16000, frame only" checkpoints/exp_frame/history/policy_iter016000.pt exp_big_ref_frame16000.json 4 FLOORSET_FRAME_UTIL=0.80
bench "iter 12000 installed" checkpoints/policy.pt exp_big_ref_base12000.json 4
echo "EXP_BIG_EVAL_DONE"

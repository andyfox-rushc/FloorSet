#!/bin/bash
# Scores the 09-30 live-frame retrain arm (FLOORSET_FRAME_UTIL=0.80) with the
# full pipeline (2 s fine-tune + 1 s slice+repack) on the 100 validation designs.
# 14000: arm vs exp_area1 control (no frame), under training load.
# 16000: arm vs installed iter 12000, quiet machine, both with and without frame.
cd "$(dirname "$0")/.."
source ../venv/bin/activate
export LD_LIBRARY_PATH=/usr/lib/x86_64-linux-gnu
bench() {  # label checkpoint out workers [env assignment]
    env $5 python -u scripts/polish_bench.py pipeline --checkpoint "$2" --splits 2:slice+repack:1 \
        --workers $4 --out "$3" 2>&1 | grep "cost" | sed "s/^/$1: /"
}
H=checkpoints/exp_frame/history
until [ -f $H/policy_iter014000.pt.meta.json ]; do sleep 120; done
bench "iter 14000 frame arm, frame" $H/policy_iter014000.pt exp_frame_iter014000.json 2 FLOORSET_FRAME_UTIL=0.80
bench "iter 14000 control exp_area1, no frame" checkpoints/exp_area1/history/policy_iter014000.pt exp_frame_ctrl_iter014000.json 2
until [ -f $H/policy_iter016000.pt.meta.json ]; do sleep 120; done
while pgrep -f "rl.train" > /dev/null; do sleep 30; done
bench "iter 16000 frame arm, frame" $H/policy_iter016000.pt exp_frame_iter016000.json 4 FLOORSET_FRAME_UTIL=0.80
bench "iter 16000 frame arm, no frame" $H/policy_iter016000.pt exp_frame_noframe_iter016000.json 4
bench "iter 12000 installed, no frame" checkpoints/policy.pt exp_frame_base_iter012000.json 4
echo "EXP_FRAME_EVAL_DONE"

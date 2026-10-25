#!/bin/bash
# Scores the 3 arms: greedy 100-design scan at 2000 (under load) and the full
# pipeline (2 s fine-tune + 1 s slice+repack, no best-of, so the policy is what
# differs) at 2000 (load) and 4000 (quiet). Results in logs/ab3_eval.log.
cd "$(dirname "$0")/.."
source ../venv/bin/activate
export LD_LIBRARY_PATH=/usr/lib/x86_64-linux-gnu
envof() { case $1 in A) echo "FLOORSET_POS_NET=cnn";; B) echo "FLOORSET_POS_NET=unet";;
                     C) echo "FLOORSET_POS_NET=unet FLOORSET_TARGET_MAP=1";; esac; }
for it in 002000 004000; do
    for a in A B C; do until [ -f checkpoints/ab_$a/history/policy_iter$it.pt.meta.json ]; do sleep 120; done; done
    w=2; [ $it = 004000 ] && { while pgrep -f "rl.train" > /dev/null; do sleep 30; done; w=4; }
    for a in A B C; do
        ck=checkpoints/ab_$a/history/policy_iter$it.pt
        u=$(python3 -c "import json;print(json.load(open('$ck.meta.json'))['utilization'])")
        env $(envof $a) python -u scripts/frame_scan.py --frames none --util $u --workers $w --checkpoint $ck \
            --out ab3_greedy_${a}_iter$it.json 2>&1 | grep -E "^ *None" | sed "s/^/greedy $a $it (util $u): /"
        env $(envof $a) FLOORSET_CANDIDATES= python -u scripts/polish_bench.py pipeline --checkpoint $ck \
            --splits 2:slice+repack:1 --workers $w --out ab3_pipe_${a}_iter$it.json 2>&1 | grep cost | sed "s/^/pipeline $a $it: /"
    done
done
echo "AB3_EVAL_DONE"

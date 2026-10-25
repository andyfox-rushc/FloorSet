#!/bin/bash
# 10-04: after scripts/ab3_eval.sh finishes, score each arm's 4000 checkpoint
# twice more (quiet machine, full pipeline, no best-of) and average with the
# eval's run. Winner = better of B/C if it beats control A by > MARGIN; then
# continue it to 20000 iterations on all cores. Decision -> logs/ab3_decision.log.
cd "$(dirname "$0")/.."
source ../venv/bin/activate
export LD_LIBRARY_PATH=/usr/lib/x86_64-linux-gnu
MARGIN=0.02
envof() { case $1 in A) echo "FLOORSET_POS_NET=cnn";; B) echo "FLOORSET_POS_NET=unet";;
                     C) echo "FLOORSET_POS_NET=unet FLOORSET_TARGET_MAP=1";; esac; }
until grep -q AB3_EVAL_DONE logs/ab3_eval.log 2>/dev/null; do sleep 120; done
for r in 2 3; do
    for a in A B C; do
        env $(envof $a) FLOORSET_CANDIDATES= python -u scripts/polish_bench.py pipeline \
            --checkpoint checkpoints/ab_$a/history/policy_iter004000.pt --splits 2:slice+repack:1 --workers 8 \
            --out ab3_pipe_${a}_iter004000_r$r.json 2>&1 | grep cost | sed "s/^/repeat $r $a: /"
    done
done
PICK=$(python3 - <<'PY'
import json, statistics as st
avg = {}
for a in "ABC":
    cs = []
    for f in [f"ab3_pipe_{a}_iter004000.json", f"ab3_pipe_{a}_iter004000_r2.json", f"ab3_pipe_{a}_iter004000_r3.json"]:
        try:
            d = json.load(open(f)); cs.append(d[next(iter(d))]["cost"])
        except FileNotFoundError:
            pass
    avg[a] = st.mean(cs) if cs else float("inf")
best = min("BC", key=lambda a: avg[a])
# User 10-04: keep training either way -- without a clear B/C win, continue
# whichever arm scored best on average.
win = best if avg["A"] - avg[best] > 0.02 else min("ABC", key=lambda a: avg[a]) + "-nomargin"
print(win, " ".join(f"{a}={avg[a]:.4f}" for a in "ABC"))
PY
)
echo "decision: $PICK (margin 0.02)"
W=${PICK%% *}
W=${W%-nomargin}
env $(envof $W) nohup python -u -m rl.train --iterations 20000 --episodes-per-iter 4 --ppo-epochs 4 \
    --resume checkpoints/ab_$W/policy.pt --checkpoint checkpoints/ab_$W/policy.pt --checkpoint-every 500 \
    --threads 6 --data-seed 12345 >> logs/train_ab_${W}_full.log 2>&1 &
echo "launched full run: arm $W to 20000 iterations (logs/train_ab_${W}_full.log)"
# Progress tracking (policy pipeline, no best-of, so the policy is what is measured).
sleep 60
env $(envof $W) FLOORSET_CANDIDATES= nohup python -u scripts/anchor_watch.py --history-dir checkpoints/ab_$W/history \
    --out-dir results_ab_$W --stop-after-iter 20000 > logs/anchor_watch_ab_$W.log 2>&1 &
env $(envof $W) nohup python -u scripts/greedy_watch.py --history-dir checkpoints/ab_$W/history \
    --out-dir results_ab_$W --stop-after-iter 20000 > logs/greedy_watch_ab_$W.log 2>&1 &
echo "AB3_DECIDED"

#!/bin/bash
# 10-02: after the HPWL-weight arms are scored (scripts/exp_hw_eval.sh), pick
# the setup by 100-design full-pipeline cost at iter 24000 -- arm A or B if
# it beats installed iter 12000, else frame 0.80 only -- then train a fresh
# policy for 20000 iterations (checkpoints/scratch/) and run
# scripts/anchor_watch.py (anchors every 500 iterations -> results_scratch/).
cd "$(dirname "$0")/.."
source ../venv/bin/activate
export LD_LIBRARY_PATH=/usr/lib/x86_64-linux-gnu
until grep -q EXP_HW_EVAL_DONE logs/exp_hw_eval.log 2>/dev/null; do sleep 120; done
cost() { grep "$1" logs/exp_hw_eval.log | sed 's/.*cost \([0-9.]*\).*/\1/'; }
A=$(cost "iter 024000 arm A"); B=$(cost "iter 024000 arm B"); BASE=$(cost "iter 12000 installed")
PICK=$(python3 -c "
a, b, base = float('${A:-9}'), float('${B:-9}'), float('${BASE:-0}')
print('A' if a < base and a <= b else 'B' if b < base else 'frame')")
case $PICK in
    A) PLACE="FLOORSET_BIG_MODE=scale FLOORSET_BIG_FRAC=0.4 FLOORSET_BIG_WEIGHT=2 FLOORSET_FRAME_UTIL=0.80"
       TRAIN="FLOORSET_HPWL_WEIGHT=2" ;;
    B) PLACE="FLOORSET_BIG_FRAC=0.4 FLOORSET_BIG_WEIGHT=0.5 FLOORSET_FRAME_UTIL=0.80"
       TRAIN="FLOORSET_HPWL_WEIGHT=2" ;;
    *) PLACE="FLOORSET_FRAME_UTIL=0.80"
       TRAIN="" ;;
esac
mkdir -p checkpoints/scratch results_scratch
{
    echo "picked: $PICK  (arm A $A, arm B $B, installed $BASE)"
    echo "placement env (needed to use these checkpoints): $PLACE"
    echo "training-only env: $TRAIN"
    echo "started: $(date -Iseconds)"
} > results_scratch/CONFIG.txt
cat results_scratch/CONFIG.txt
env $PLACE $TRAIN nohup python -u -m rl.train --iterations 20000 --episodes-per-iter 4 --ppo-epochs 4 \
    --checkpoint checkpoints/scratch/policy.pt --checkpoint-every 500 --threads 4 \
    > logs/train_scratch.log 2>&1 &
sleep 60
env $PLACE nohup python -u scripts/anchor_watch.py --history-dir checkpoints/scratch/history \
    --out-dir results_scratch --stop-after-iter 20000 > logs/anchor_watch_scratch.log 2>&1 &
echo "SCRATCH_LAUNCHED"

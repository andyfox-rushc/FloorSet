#!/bin/bash
# 10-11: after the candidate diagnostic, A/B the soft-block aspect cap
# (FLOORSET_MAX_ASPECT 8 = current, 3 = ground truth's range, 16) on the 10
# largest designs, one arm at a time, then RL fine-tuning of the relation placer.
cd "$(dirname "$0")/.."
while pgrep -f "scripts/cand_diag.py" > /dev/null; do sleep 60; done
for cap in 8 3 16; do
    FLOORSET_MAX_ASPECT=$cap ../venv/bin/python scripts/solver_ab.py --ids 90 91 92 93 94 95 96 97 98 99 \
        --out aspect_ab_$cap.json > logs/aspect_ab_$cap.log 2>&1
done
../venv/bin/python scripts/rl_relation.py --init checkpoints/placer_r.pt --hours 8 \
    --out checkpoints/placer_rl2.pt > logs/rl_relation.log 2>&1
echo chain done

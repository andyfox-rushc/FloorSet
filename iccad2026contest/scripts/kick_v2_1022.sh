#!/bin/bash
# 10-22 evening: kick policy v2 -- train on the layout pool, then learned vs random kicks
# at equal time on 30 held-out training designs.
cd "$(dirname "$0")/.."
until grep -q "^done:" logs/build_layout_pool.log 2>/dev/null; do sleep 30; done
FLOORSET_PARTITION_MEMO=20000 ../venv/bin/python scripts/train_kick_policy.py --pool checkpoints/layout_pool.pkl \
  --hours 1.25 --steps 6 --k 6 --lr 1e-3 --out checkpoints/kick_policy_v2.pt > logs/train_kick_policy_v2.log 2>&1
FLOORSET_PARTITION_MEMO=20000 ../venv/bin/python scripts/eval_kick_policy.py --designs 30 --per-file 3 --seconds 20 \
  --policy checkpoints/kick_policy_v2.pt > prime_results/eval_kick_policy_v2.log 2>&1
echo finished > prime_results/kick_v2.done

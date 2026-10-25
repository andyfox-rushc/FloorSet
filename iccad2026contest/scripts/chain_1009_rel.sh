#!/bin/bash
# 10-09: after relation training ends -> official run with relation-net candidates -> continue training.
cd "$(dirname "$0")/.."
until grep -q "^done:" logs/train_placer_rel.log; do sleep 30; done
cp checkpoints/placer_rel.pt checkpoints/placer_rel_run1_final.pt
../venv/bin/python iccad2026_evaluate.py --evaluate my_optimizer.py --output official_eval_1009_rel.json > logs/official_eval_1009_rel.log 2>&1
../venv/bin/python scripts/train_placer.py --hours 6 --threads 6 --seed 6 --hidden 256 --layers 8 --norm --rel \
    --resume checkpoints/placer_rel_run1_final.pt --lr 2e-4 --val-every 5000 --out checkpoints/placer_rel2.pt > logs/train_placer_rel2.log 2>&1

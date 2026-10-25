#!/bin/bash
# 10-17: defaults + second packer round from the winner with wall penalty 100
# (compare official_eval_1015_cluster.json, 1.0432 / 72 beat GT).
cd "$(dirname "$0")/.."
FLOORSET_REPACK_WINNER=1 FLOORSET_WALL_PASS=100 ../venv/bin/python iccad2026_evaluate.py --evaluate my_optimizer.py \
    --output official_eval_1017_repack.json > logs/official_eval_1017_repack.log 2>&1
echo run done

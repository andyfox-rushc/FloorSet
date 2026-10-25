#!/bin/bash
# 10-15: defaults = cap 16 + reattach_clusters + cluster-gap penalty 1 (compare official_eval_1012_cap16.json, 1.0451 / 68 beat GT).
cd "$(dirname "$0")/.."
../venv/bin/python iccad2026_evaluate.py --evaluate my_optimizer.py \
    --output official_eval_1015_cluster.json > logs/official_eval_1015_cluster.log 2>&1
echo run done

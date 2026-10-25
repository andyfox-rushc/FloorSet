#!/bin/bash
# 10-12: current defaults + soft-block aspect cap 16 (A/B on 90-99: 1.0354 vs 1.0386).
cd "$(dirname "$0")/.."
FLOORSET_MAX_ASPECT=16 ../venv/bin/python iccad2026_evaluate.py --evaluate my_optimizer.py \
    --output official_eval_1012_cap16.json > logs/official_eval_1012_cap16.log 2>&1
echo run done

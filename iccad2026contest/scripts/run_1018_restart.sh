#!/bin/bash
# 10-18: defaults (10-17: winner repack + wall pass) + up to 3 whole-solve runs while the
# best layout still has a soft violation; reruns add wall-penalty packer jobs
# (compare official_eval_1017_repack.json, 1.0404 / 77 beat GT).
cd "$(dirname "$0")/.."
FLOORSET_SOLVE_RUNS=3 FLOORSET_RESTART_VIOL=1 ../venv/bin/python iccad2026_evaluate.py --evaluate my_optimizer.py \
    --output official_eval_1018_restart.json > logs/official_eval_1018_restart.log 2>&1
echo run done

#!/bin/bash
# 10-11: 10/10 setup + relB as learned4, packer budget 200 (2x).
cd "$(dirname "$0")/.."
FLOORSET_PLACER4=$PWD/checkpoints/placer_relB.pt FLOORSET_PACK=200 \
    FLOORSET_CANDIDATES="base,scratch0.75:it600:po1.0,scratch0.85:it600:po1.0,scratch0.90:it600:po1.0,scratch0.95:it600:po1.0,scratch1.0:it600:po1.0,learned,learned2,learned3,learned4" \
    ../venv/bin/python iccad2026_evaluate.py --evaluate my_optimizer.py --output official_eval_1011_pack200.json > logs/official_eval_1011_pack200.log 2>&1
echo run done

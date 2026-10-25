#!/bin/bash
# 10-11 (replaces chain_1011_aspect_rl.sh after its first arm): A/B on the 10
# largest designs, one arm at a time, then RL fine-tuning of the relation placer.
#   aspect_ab_8   current settings (already running when this starts)
#   topk10_200    pack all 10 candidates, packer 200
#   topk10_100    pack all 10, packer 100 (about today's runtime)
#   aspect_ab_3 / aspect_ab_16   soft-block aspect cap
cd "$(dirname "$0")/.."
IDS="90 91 92 93 94 95 96 97 98 99"
while kill -0 770754 2> /dev/null; do sleep 60; done
FLOORSET_PACK_TOPK=10 ../venv/bin/python scripts/solver_ab.py --ids $IDS --out topk10_200.json > logs/topk10_200.log 2>&1
FLOORSET_PACK_TOPK=10 FLOORSET_PACK=100 ../venv/bin/python scripts/solver_ab.py --ids $IDS --out topk10_100.json > logs/topk10_100.log 2>&1
for cap in 3 16; do
    FLOORSET_MAX_ASPECT=$cap ../venv/bin/python scripts/solver_ab.py --ids $IDS --out aspect_ab_$cap.json > logs/aspect_ab_$cap.log 2>&1
done
../venv/bin/python scripts/rl_relation.py --init checkpoints/placer_r.pt --hours 8 \
    --out checkpoints/placer_rl2.pt > logs/rl_relation.log 2>&1
echo chain done

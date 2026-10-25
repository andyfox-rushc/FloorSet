#!/bin/bash
# 10-16 anchor A/B: search-only pull toward placer_r's predicted centres (FLOORSET_PACK_ANCHOR),
# fading out over the first half of the packer search; 10 designs we lose to the ground truth.
cd "$(dirname "$0")/.."
IDS="90 61 54 70 94 49 11 43 59 25"
for a in 0 1 3; do
    FLOORSET_PACK_ANCHOR=$a ../venv/bin/python scripts/solver_ab.py --ids $IDS --out anchor_ab$a.json \
        > logs/anchor_ab$a.log 2>&1
done
echo chain done

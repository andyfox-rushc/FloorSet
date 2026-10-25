#!/bin/bash
# 10-15 cluster fix A/B on the 12 designs with split clusters in the cap-16 official run.
# All arms have reattach_clusters on; FLOORSET_PACK_GAP = search-only cluster-gap penalty.
cd "$(dirname "$0")/.."
IDS="5 6 21 34 38 53 56 61 64 68 70 74"
for g in 0 1 3; do
    FLOORSET_PACK_GAP=$g ../venv/bin/python scripts/solver_ab.py --ids $IDS --out cluster_gap$g.json \
        > logs/cluster_gap$g.log 2>&1
done
echo chain done

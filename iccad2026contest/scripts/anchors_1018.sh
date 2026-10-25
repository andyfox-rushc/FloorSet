#!/bin/bash
# 10-18 report anchors: designs 0 / 49 / 98, current defaults (cap 16, cluster fix, winner repack + wall pass, violation restarts), 3 runs, sequential.
cd "$(dirname "$0")/.."
for r in 0 1 2; do
    ../venv/bin/python scripts/anchor_results.py --test-ids 0 49 98 --out-dir results_report_1018/run$r \
        --label "10/18 violation restarts" > logs/anchors_1018_run$r.log 2>&1
done
echo anchors done

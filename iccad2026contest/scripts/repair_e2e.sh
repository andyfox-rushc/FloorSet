#!/bin/bash
# 10-05: end-to-end best-of (2 s fine-tune + default candidates) on the 100
# designs with the step-1 repair changes, plus ablations. 4 workers like
# bestof_A4000.json (1.424).
cd "$(dirname "$0")/.."
source ../venv/bin/activate
export LD_LIBRARY_PATH=/usr/lib/x86_64-linux-gnu
run() {  # label out [env...]
    local label=$1 out=$2; shift 2
    env "$@" python -u scripts/polish_bench.py pipeline --checkpoint checkpoints/policy.pt \
        --splits 2:best:1 --workers 4 --out "$out" 2>&1 | grep "cost" | sed "s/^/$label: /"
}
export FLOORSET_CANDIDATES="base,scratch0.85:it250:po1.0,scratch0.95:it250:po1.0"
run "all"        repair_e2e_all.json
run "no-repair"  repair_e2e_norepair.json FLOORSET_REPAIR=0
run "no-walls"   repair_e2e_nowalls.json  FLOORSET_OUTLINE_WALLS=0
run "all-r2"     repair_e2e_all_r2.json
echo "REPAIR_E2E_DONE"

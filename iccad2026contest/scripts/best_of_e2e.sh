#!/bin/bash
# 10-04: end-to-end (2 s fine-tune + best-of candidates) on the 100 designs,
# installed policy, 4 workers, one config at a time, vs today's pipeline.
cd "$(dirname "$0")/.."
source ../venv/bin/activate
export LD_LIBRARY_PATH=/usr/lib/x86_64-linux-gnu
run() {  # label candidates out
    FLOORSET_CANDIDATES="$2" python -u scripts/polish_bench.py pipeline --checkpoint checkpoints/policy.pt \
        --splits 2:best:1 --workers 4 --out "$3" 2>&1 | grep "cost" | sed "s/^/$1: /"
}
python -u scripts/polish_bench.py pipeline --checkpoint checkpoints/policy.pt --splits 2:slice+repack:1 \
    --workers 4 --out e2e_base.json 2>&1 | grep "cost" | sed "s/^/today: /"
run "A base:po0.5+scratch0.95:it250:po1.0" "base:po0.5,scratch0.95:it250:po1.0" e2e_A.json
run "B base:po0.5+scratch0.95:it250:po0.5+scratch0.85:it250:po1.0" "base:po0.5,scratch0.95:it250:po0.5,scratch0.85:it250:po1.0" e2e_B.json
run "C base+scratch0.85:it250:po1.0+scratch0.95:it250:po1.0" "base,scratch0.85:it250:po1.0,scratch0.95:it250:po1.0" e2e_C.json
echo "E2E_DONE"

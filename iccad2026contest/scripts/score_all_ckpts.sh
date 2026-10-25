#!/bin/bash
# 10-04: full pipeline (2 s fine-tune + 1 s slice+repack) on the 100 designs for
# every from-scratch checkpoint (500..20000), one at a time, 4 workers, quiet
# machine -> allck/scratch_iter<N>.json, summary lines in logs/score_all_ckpts.log.
cd "$(dirname "$0")/.."
source ../venv/bin/activate
export LD_LIBRARY_PATH=/usr/lib/x86_64-linux-gnu
mkdir -p allck
for it in $(seq 500 500 20000); do
    n=$(printf "%06d" $it)
    out=allck/scratch_iter$n.json
    [ -f $out ] && continue
    FLOORSET_FRAME_UTIL=0.80 python -u scripts/polish_bench.py pipeline --checkpoint checkpoints/scratch/history/policy_iter$n.pt \
        --splits 2:slice+repack:1 --workers 4 --out $out 2>&1 | grep "cost" | sed "s/^/iter $it: /"
done
echo "SCORE_ALL_DONE"

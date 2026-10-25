#!/bin/bash
# Rebuild the final Prime result from scratch with the current code (about 16 h on 8 cores, plus
# about 5 h for the LNS stage): the flows behind prime_results/final_1024.json, best-of + 60 s
# polish (final.json, as final_1023.json), window LNS (final_lns.json, as final_1024.json), a re-score.
# The LNS stage stops a design once it reaches the ground truth's cost (--stop-at-gt: time
# allocation only; every keep/pick decision is ground-truth free). Timed searches: a rerun lands
# near, not exactly on, the stored results.
# A step whose output exists is skipped, so a stopped run resumes. IDS="0 49" = a subset.
#   bash scripts/reproduce_final.sh [out_dir]      (default prime_results/repro)
# LNS_ROUNDS=60,60 LNS_SEEDS=1,2 shortens the LNS stage (a quick check).
set -e
cd "$(dirname "$0")/.."
PY=../venv/bin/python
R=${1:-prime_results/repro}
mkdir -p "$R"
SEL=(); [ -n "$IDS" ] && SEL=(--ids $IDS)
run() { local out=$1; shift; [ -f "$out" ] || { echo "$(date +%H:%M) -> $out"; "$@" > "${out%.json}.log" 2>&1; }; }
EVAL=(prime_evaluate.py "${SEL[@]}" --evaluate)

run "$R/packer.json" env FLOORSET_SOLVE_RUNS=1 FLOORSET_PACK=20 $PY "${EVAL[@]}" prime_optimizer.py --output "$R/packer.json"
run "$R/packer_refined.json" $PY scripts/refine_offline.py "$R/packer.json" "$R/packer_refined.json"
run "$R/partition.json" $PY "${EVAL[@]}" prime_partition.py --output "$R/partition.json"
run "$R/partition_refined.json" $PY scripts/refine_offline.py "$R/partition.json" "$R/partition_refined.json"
run "$R/partition_refine.json" env FLOORSET_REFINE=15 $PY "${EVAL[@]}" prime_partition.py --output "$R/partition_refine.json"
run "$R/partition_refine_tree.json" env FLOORSET_REFINE=15 FLOORSET_TREE_POLICY=checkpoints/tree_policy_best_8559.pt \
    $PY "${EVAL[@]}" prime_partition.py --output "$R/partition_refine_tree.json"
run "$R/final.json" env FLOORSET_PARTITION_MEMO=20000 $PY scripts/final_polish.py \
    --flow "Packer flow=$R/packer.json" --flow "Packer + refine=$R/packer_refined.json" \
    --flow "Partition=$R/partition.json" --flow "Partition + refine=$R/partition_refined.json" \
    --flow "Partition + refine in restarts=$R/partition_refine.json" \
    --flow "Partition + refine + tree policy=$R/partition_refine_tree.json" \
    --seconds 60 --jobs 3 --out "$R/final.json"
run "$R/final_lns.json" env OMP_NUM_THREADS=1 FLOORSET_PARTITION_MEMO=20000 $PY scripts/lns_final.py \
    --base "$R/final.json" --flow "Partition=$R/partition.json" --flow "Partition + refine=$R/partition_refined.json" \
    --flow "Partition + refine in restarts=$R/partition_refine.json" \
    --flow "Partition + refine + tree policy=$R/partition_refine_tree.json" \
    --rounds "${LNS_ROUNDS:-900,1200,2400,2400,2400,2400}" --seeds "${LNS_SEEDS:-1,1,2,4,4,4}" --jobs 6 --stop-at-gt \
    --out "$R/final_lns.json"
$PY scripts/verify_results.py "$R"/packer.json "$R"/packer_refined.json "$R"/partition.json "$R"/partition_refined.json \
    "$R"/partition_refine.json "$R"/partition_refine_tree.json "$R"/final.json "$R"/final_lns.json 2>&1 | grep -v "Warn\|detach"

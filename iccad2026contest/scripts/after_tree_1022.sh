#!/bin/bash
# 10-22: finish the RL tree-builder run, rerun its crashed designs with the fixed refiner,
# merge, then best-of + polish over all flows and rebuild the report.
cd "$(dirname "$0")/.."
until grep -q "weighted score" prime_results/partition_full_refine_tree.log; do sleep 60; done
IDS=$(../venv/bin/python -c "import json; print(' '.join(str(r['id']) for r in json.load(open('prime_results/partition_full_refine_tree.json')) if not r['feasible']))")
echo "rerun ids: $IDS" > prime_results/after_tree_1022.out
if [ -n "$IDS" ]; then
  FLOORSET_REFINE=15 FLOORSET_TREE_POLICY=checkpoints/tree_policy_best_8559.pt ../venv/bin/python prime_evaluate.py \
    --evaluate prime_partition.py --ids $IDS --output prime_results/partition_full_refine_tree_rerun.json \
    > prime_results/partition_full_refine_tree_rerun.log 2>&1
  ../venv/bin/python scripts/merge_results.py prime_results/partition_full_refine_tree.json \
    prime_results/partition_full_refine_tree_rerun.json prime_results/partition_full_refine_tree_merged.json >> prime_results/after_tree_1022.out
else
  cp prime_results/partition_full_refine_tree.json prime_results/partition_full_refine_tree_merged.json
fi
F=(--flow "Packer flow=prime_results/contour_pack20.json"
   --flow "Packer + refine=prime_results/contour_pack20_ils.json"
   --flow "Partition (t80, cache fixed)=prime_results/partition_full_t80b.json"
   --flow "Partition + refine=prime_results/partition_t80b_refined.json"
   --flow "Partition + refine in restarts=prime_results/partition_full_refine.json"
   --flow "Partition + refine + tree policy=prime_results/partition_full_refine_tree_merged.json")
FLOORSET_PARTITION_MEMO=20000 ../venv/bin/python scripts/final_polish.py "${F[@]}" \
  --flow "Final 10/22 night=prime_results/final_1022.json" --seconds 60 --jobs 3 \
  --out prime_results/final_1022b.json > prime_results/final_1022b.log 2>&1
cp prime_report_1022.pdf prime_report_1022_v1.pdf
../venv/bin/python scripts/generate_prime_report_1022.py --out prime_report_1022.pdf "${F[@]}" \
  --flow "Final (best of all + polish)=prime_results/final_1022b.json" \
  --panels "Partition (t80, cache fixed)" > prime_results/report_1022b.log 2>&1
echo finished > prime_results/after_tree_1022.done

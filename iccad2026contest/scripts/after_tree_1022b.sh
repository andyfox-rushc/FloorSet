#!/bin/bash
# 10-22 evening: rerun the RL tree-builder designs that crashed or never ran (fixed refiner,
# crash-proof restarts), merge with the 80 scored ones, then best-of + polish and the report.
cd "$(dirname "$0")/.."
IDS="0 13 21 25 27 30 37 45 48 49 57 62 64 69 75 77 79 80 81 82 83 84 85 86 87 88 89 90 91 92 93 94 95 96 97 98 99"
FLOORSET_REFINE=15 FLOORSET_TREE_POLICY=checkpoints/tree_policy_best_8559.pt ../venv/bin/python prime_evaluate.py \
  --evaluate prime_partition.py --ids $IDS --output prime_results/partition_full_refine_tree_rerun.json \
  > prime_results/partition_full_refine_tree_rerun.log 2>&1
../venv/bin/python scripts/merge_results.py prime_results/partition_full_refine_tree_part80.json \
  prime_results/partition_full_refine_tree_rerun.json prime_results/partition_full_refine_tree_merged.json \
  > prime_results/after_tree_1022.out 2>&1
F=(--flow "Packer flow=prime_results/contour_pack20.json"
   --flow "Packer + refine=prime_results/contour_pack20_ils.json"
   --flow "Partition (t80, cache fixed)=prime_results/partition_full_t80b.json"
   --flow "Partition + refine=prime_results/partition_t80b_refined.json"
   --flow "Partition + refine in restarts=prime_results/partition_full_refine.json"
   --flow "Partition + refine + tree policy=prime_results/partition_full_refine_tree_merged.json")
FLOORSET_PARTITION_MEMO=20000 ../venv/bin/python scripts/final_polish.py "${F[@]}" \
  --flow "Final 10/22 night=prime_results/final_1022.json" --seconds 60 --jobs 3 \
  --out prime_results/final_1022b.json > prime_results/final_1022b.log 2>&1
[ -f prime_report_1022_v1.pdf ] || cp prime_report_1022.pdf prime_report_1022_v1.pdf
../venv/bin/python scripts/generate_prime_report_1022.py --out prime_report_1022.pdf "${F[@]}" \
  --flow "Final (best of all + polish)=prime_results/final_1022b.json" \
  --panels "Partition (t80, cache fixed)" > prime_results/report_1022b.log 2>&1
echo finished > prime_results/after_tree_1022.done

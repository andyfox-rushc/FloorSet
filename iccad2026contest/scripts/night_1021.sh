#!/bin/bash
# 10-21 night queue (after the full t80b rerun):
#  1. offline pair re-cut refinement of every t80b layout (scripts/refine_offline.py)
#  2. 12-design A/B, both with in-restart refinement: seed tree vs tree-policy start
#  3. full 100-design run with refinement + the better start
cd "$(dirname "$0")/.."
until [ -f prime_results/partition_full_t80b.done ]; do sleep 30; done
../venv/bin/python scripts/refine_offline.py prime_results/partition_full_t80b.json prime_results/partition_t80b_refined.json > prime_results/partition_t80b_refined.log 2>&1 &
IDS="3 11 19 27 35 44 52 61 69 77 86 95"
FLOORSET_REFINE=15 ../venv/bin/python prime_evaluate.py --evaluate prime_partition.py --ids $IDS \
  --output prime_results/ab_refine_s12.json > prime_results/ab_refine_s12.log 2>&1
cp checkpoints/tree_policy.pt checkpoints/tree_policy_ab1.pt
FLOORSET_REFINE=15 FLOORSET_TREE_POLICY=checkpoints/tree_policy_ab1.pt ../venv/bin/python prime_evaluate.py --evaluate prime_partition.py --ids $IDS \
  --output prime_results/ab_refine_tree_s12.json > prime_results/ab_refine_tree_s12.log 2>&1
echo finished > prime_results/night_ab.done

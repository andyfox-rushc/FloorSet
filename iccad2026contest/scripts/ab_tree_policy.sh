#!/bin/bash
# 10-21 night: partition flow started from the tree policy's best sampled tree (plan A)
# vs the t80b baseline, same 12 designs; waits for the full rerun to finish.
cd "$(dirname "$0")/.."
until [ -f prime_results/partition_full_t80b.done ]; do sleep 30; done
cp checkpoints/tree_policy.pt checkpoints/tree_policy_ab1.pt
FLOORSET_TREE_POLICY=checkpoints/tree_policy_ab1.pt ../venv/bin/python prime_evaluate.py --evaluate prime_partition.py \
  --ids 3 11 19 27 35 44 52 61 69 77 86 95 --output prime_results/ab_tree_s12.json > prime_results/ab_tree_s12.log 2>&1
echo finished > prime_results/ab_tree_s12.done

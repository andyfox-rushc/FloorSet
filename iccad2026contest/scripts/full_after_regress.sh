#!/bin/bash
# 10-21: full partition rerun (200k cache restored) once the regression check is done
cd "$(dirname "$0")/.."
until grep -q done prime_results/regress_1021.out; do sleep 30; done
../venv/bin/python prime_evaluate.py --evaluate prime_partition.py --output prime_results/partition_full_t80b.json > prime_results/partition_full_t80b.log 2>&1
echo finished > prime_results/partition_full_t80b.done

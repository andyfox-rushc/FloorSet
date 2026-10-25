#!/bin/bash
# 10-21 regression hunt: the t80 full run lost to the 10-20 run on big designs.
# Arms: old cache size; old cache + analytic seeds only; current settings.
cd "$(dirname "$0")/.."
IDS="61 62 73 77"
FLOORSET_PARTITION_MEMO=200000 ../venv/bin/python prime_evaluate.py --evaluate prime_partition.py --ids $IDS --output prime_results/regress_memo200k.json > prime_results/regress_memo200k.log 2>&1
FLOORSET_PARTITION_MEMO=200000 FLOORSET_PRIME_PLACER= ../venv/bin/python prime_evaluate.py --evaluate prime_partition.py --ids $IDS --output prime_results/regress_memo200k_analytic.json > prime_results/regress_memo200k_analytic.log 2>&1
../venv/bin/python prime_evaluate.py --evaluate prime_partition.py --ids $IDS --output prime_results/regress_current.json > prime_results/regress_current.log 2>&1
echo done

#!/bin/bash
# LNS (rl/window.py) on the designs of final_1024.json not at 1.000; 6 in parallel.
cd "$(dirname "$0")/.."
export OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 FLOORSET_PARTITION_MEMO=20000
IDS=${IDS:-"76 83 98 80 86 81 75 48 87 63 36 15 29"}
SECS=${SECS:-900}
SEED=${SEED:-0}
for i in $IDS; do echo $i; done | xargs -P 6 -I{} sh -c \
  "../venv/bin/python scripts/lns_designs.py prime_results/final_1024.json {} prime_results/lns_1024/lns_{}_s$SEED.json --seconds $SECS --seed $SEED 2>&1 | grep -v RuntimeWarning"
echo done

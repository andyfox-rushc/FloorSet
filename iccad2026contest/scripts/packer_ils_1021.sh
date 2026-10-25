#!/bin/bash
# 10-21 night: packer layouts refined with iterated local search (30 s each), after the t80b run
cd "$(dirname "$0")/.."
until [ -f prime_results/partition_full_t80b.done ]; do sleep 30; done
../venv/bin/python scripts/refine_offline.py prime_results/contour_pack20.json prime_results/contour_pack20_ils.json \
  > prime_results/contour_pack20_ils.log 2>&1

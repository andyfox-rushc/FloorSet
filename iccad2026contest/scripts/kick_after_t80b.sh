#!/bin/bash
# restart the kick-policy training once the t80b run has freed its memory (small caches)
cd "$(dirname "$0")/.."
until [ -f prime_results/partition_full_t80b.done ]; do sleep 30; done
FLOORSET_PARTITION_MEMO=20000 ../venv/bin/python scripts/train_kick_policy.py --hours 7 --out checkpoints/kick_policy.pt > logs/train_kick_policy.log 2>&1

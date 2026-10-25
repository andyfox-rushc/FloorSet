#!/bin/bash
# 10-21 night: after the 12-design A/B, the full 100-design partition run with
# in-restart refinement, plus the tree-policy start if it won the A/B.
cd "$(dirname "$0")/.."
until [ -f prime_results/night_ab.done ]; do sleep 30; done
WIN=$(../venv/bin/python - <<'PY'
import json, math
def w(f):
    rs = json.load(open(f))
    m = max(r["block_count"] for r in rs)
    ws = [math.exp((r["block_count"] - m) / 12) for r in rs]
    return sum(r.get("quality_cost", r["cost"]) * x for r, x in zip(rs, ws)) / sum(ws)
a, b = w("prime_results/ab_refine_s12.json"), w("prime_results/ab_refine_tree_s12.json")
print("tree" if b < a else "plain")
PY
)
echo "A/B winner: $WIN" > prime_results/night_full.choice
if [ "$WIN" = "tree" ]; then
  export FLOORSET_TREE_POLICY=checkpoints/tree_policy_ab1.pt
fi
FLOORSET_REFINE=15 ../venv/bin/python prime_evaluate.py --evaluate prime_partition.py \
  --output prime_results/partition_full_refine.json > prime_results/partition_full_refine.log 2>&1
echo finished > prime_results/partition_full_refine.done

#!/bin/bash
# 10-12 restarts A/B, designs 90-99, aspect cap 16 (compare aspect_ab_16.json, 1.0354):
#   restart_jobs6    6 packer jobs (top 4 + new seeds on #1, #2): idle workers, same wall time
#   restart_jobs12   12 jobs (3 seeds per candidate): 2x wall time
#   restart_pack400  4 jobs, twice the packer budget: 2x wall time, control for jobs12
# Waits for the official cap-16 run first (wall-clock budgets: never overlap).
cd "$(dirname "$0")/.."
while ps -p 1449 > /dev/null; do sleep 60; done
IDS="90 91 92 93 94 95 96 97 98 99"
export FLOORSET_MAX_ASPECT=16
FLOORSET_PACK_JOBS=6 ../venv/bin/python scripts/solver_ab.py --ids $IDS --out restart_jobs6.json > logs/restart_jobs6.log 2>&1
FLOORSET_PACK_JOBS=12 ../venv/bin/python scripts/solver_ab.py --ids $IDS --out restart_jobs12.json > logs/restart_jobs12.log 2>&1
FLOORSET_PACK=400 ../venv/bin/python scripts/solver_ab.py --ids $IDS --out restart_pack400.json > logs/restart_pack400.log 2>&1
python3 - <<'EOF' > logs/restart_summary.txt
import json, math
for f in ["aspect_ab_8", "aspect_ab_16", "restart_jobs6", "restart_jobs12", "restart_pack400"]:
    d = json.load(open(f + ".json"))
    w = {k: math.exp(v["n"] / 12) for k, v in d.items()}
    tot = sum(w[k] * v["cost"] for k, v in d.items()) / sum(w.values())
    per = " ".join(f"{k}:{v['cost']:.3f}" for k, v in sorted(d.items()))
    print(f"{f:16s} weighted {tot:.4f}  secs {sum(v['secs'] for v in d.values()) / len(d):.0f}  {per}")
EOF
echo chain done

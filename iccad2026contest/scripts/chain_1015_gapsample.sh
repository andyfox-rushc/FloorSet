#!/bin/bash
# 10-15: cluster-gap penalty winner ($1) on the clean sample 90-99; compare with the
# restart A/B arms (aspect_ab_16 1.0354, noise band 1.0345-1.0409).
cd "$(dirname "$0")/.."
G=${1:?usage: chain_1015_gapsample.sh GAP}
IDS="90 91 92 93 94 95 96 97 98 99"
FLOORSET_PACK_GAP=$G ../venv/bin/python scripts/solver_ab.py --ids $IDS --out sample_gap$G.json \
    > logs/sample_gap$G.log 2>&1
python3 - "$G" <<'PY' > logs/sample_gap$G.summary.txt
import json, math, sys
for f in ["aspect_ab_8", "aspect_ab_16", "restart_jobs6", "restart_jobs12", "restart_pack400", "sample_gap" + sys.argv[1]]:
    d = json.load(open(f + ".json"))
    w = {k: math.exp(v["n"] / 12) for k, v in d.items()}
    tot = sum(w[k] * v["cost"] for k, v in d.items()) / sum(w.values())
    per = " ".join(f"{k}:{v['cost']:.3f}" for k, v in sorted(d.items()))
    print(f"{f:16s} weighted {tot:.4f}  secs {sum(v['secs'] for v in d.values()) / len(d):.0f}  {per}")
PY
echo chain done

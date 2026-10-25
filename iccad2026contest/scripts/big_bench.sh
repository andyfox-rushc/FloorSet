#!/bin/bash
# Big-design bench (10-06): solver on the 21 designs with 100+ blocks (83% of
# the weighted score), folded into the full weighted score using
# official_eval_1006b.json for the rest.
#   bash scripts/big_bench.sh <tag> [ENV=VAL ...]  -> big_<tag>.json
tag=$1; shift
IDS=$(../venv/bin/python -c "
import json; t = json.load(open('official_eval_1006b.json'))['test_results']
print(' '.join(str(r['test_id']) for r in t if r['block_count'] >= 100))" 2>/dev/null)
env "$@" ../venv/bin/python scripts/solver_ab.py --ids $IDS --out big_$tag.json > logs/big_$tag.log 2>&1
../venv/bin/python - "$tag" <<'PY' 2>/dev/null
import json, math, sys
tag = sys.argv[1]
t = json.load(open('official_eval_1006b.json'))['test_results']
b = {int(k): v for k, v in json.load(open(f'big_{tag}.json')).items()}
W = sum(math.exp(r['block_count'] / 12) for r in t)
tot = sum(math.exp(r['block_count'] / 12) * (b[r['test_id']]['cost'] if r['test_id'] in b else r['cost']) for r in t) / W
bw = sum(math.exp(v['n'] / 12) for v in b.values())
print(f"{tag}: weighted total {tot:.4f} | big-only weighted {sum(math.exp(v['n'] / 12) * v['cost'] for v in b.values()) / bw:.4f} "
      f"mean {sum(v['cost'] for v in b.values()) / len(b):.4f} secs {sum(v['secs'] for v in b.values()) / len(b):.1f}")
PY

"""Refine chosen designs of a result file with rl/refine.py (fresh seed), keep
each only if its contest cost drops, and write the merged result file.

  python scripts/refine_designs.py in.json out.json --ids 50 77 80 98 [--seconds 60]
"""
import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
import prime_evaluate as E  # noqa: E402
from rl import partition as PT  # noqa: E402
from rl.refine import Refiner  # noqa: E402

ap = argparse.ArgumentParser()
ap.add_argument("inp")
ap.add_argument("out")
ap.add_argument("--ids", type=int, nargs="+", required=True)
ap.add_argument("--seconds", type=float, default=60.0)
ap.add_argument("--seed", type=int, default=0)
a = ap.parse_args()
q = lambda r: r.get("quality_cost", r["cost"])
F = {r["id"]: r for r in json.loads(Path(a.inp).read_text())}
for i in a.ids:
    d = E.load_designs(ids=[i])[0]
    P = PT.Problem(d["block_count"], d["area_targets"], d["b2b"], d["p2b"], d["pins_pos"], d["constraints"],
                   d["target_polygons"])
    R = Refiner(P, F[i]["layout"])
    st = R.run(a.seconds, seed=a.seed)
    lay = R.layout()
    n = E.evaluate(d, lay, runtime=F[i].get("runtime", 0.0) + st["seconds"])
    if n["feasible"] and q(n) < q(F[i]) - 1e-9:
        n["layout"] = [list(map(list, v)) for v in lay]
        n["picked"] = F[i].get("picked")
        print(f"design {i}: {q(F[i]):.4f} -> {q(n):.4f}  soft {n.get('soft')}", flush=True)
        F[i] = n
    else:
        print(f"design {i}: kept {q(F[i]):.4f} (refined {q(n):.4f}, feasible {n['feasible']})", flush=True)
out = [F[i] for i in sorted(F)]
Path(a.out).write_text(json.dumps(out))
E._summary(out, a.out)

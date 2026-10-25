"""Large-neighbourhood search (rl/window.py lns: re-partition random windows of
6-20 blocks, rest fixed) on one design of a result file; writes the layout and
prints the evaluator's verdict. Merge with scripts/merge_layout.py.

  python scripts/lns_designs.py results.json ID out_layout.json [--seconds 600] [--seed 0]
"""
import argparse
import json
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
import prime_evaluate as E  # noqa: E402
from rl import partition as PT  # noqa: E402
from rl import window as WN  # noqa: E402
from rl.refine import Refiner  # noqa: E402

ap = argparse.ArgumentParser()
ap.add_argument("results")
ap.add_argument("id", type=int)
ap.add_argument("out")
ap.add_argument("--seconds", type=float, default=600.0)
ap.add_argument("--seed", type=int, default=0)
ap.add_argument("--kmin", type=int, default=6)
ap.add_argument("--kmax", type=int, default=14)
a = ap.parse_args()
q = lambda r: r.get("quality_cost", r["cost"])
row = next(r for r in json.loads(Path(a.results).read_text()) if r["id"] == a.id)
d = E.load_designs(ids=[a.id])[0]
P = PT.Problem(d["block_count"], d["area_targets"], d["b2b"], d["p2b"], d["pins_pos"], d["constraints"],
               d["target_polygons"])
Rf = Refiner(P, row["layout"])
Rf.hp = Rf.hp_ref = max(P.hpwl(Rf.c[:, 0], Rf.c[:, 1]), 1e-9)
Rf.viol = Rf.soft_viol(range(P.n))
t0 = time.time()
st = WN.lns(Rf, t0 + a.seconds, seed=a.seed, kmin=a.kmin, kmax=a.kmax)
lay = Rf.layout()
n = E.evaluate(d, lay)
Path(a.out).write_text(json.dumps([list(map(list, v)) for v in lay]))
print(f"design {a.id}: {q(row):.4f} -> {q(n):.4f} feasible {n['feasible']} hpwl_gap {n.get('hpwl_gap', 0):.4f} "
      f"soft {n.get('soft')} | tries {st['tries']} wins {st['wins']} {time.time() - t0:.0f}s", flush=True)

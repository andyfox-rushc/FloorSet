"""Put one design's layout (a JSON list of polygons) into a result file if the
evaluator finds it feasible and cheaper; prints old -> new.

  python scripts/merge_layout.py results.json ID layout.json [--seconds S] [--picked LABEL] [--by-hpwl]

--by-hpwl: keep it if it is valid, has no more soft violations and has shorter
wirelength (the cost is floored at 1.000, so below the ground truth it can't drop).
"""
import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
import prime_evaluate as E  # noqa: E402

ap = argparse.ArgumentParser()
ap.add_argument("results")
ap.add_argument("id", type=int)
ap.add_argument("layout")
ap.add_argument("--seconds", type=float, default=0.0, help="runtime added to the design's")
ap.add_argument("--picked", default=None)
ap.add_argument("--by-hpwl", action="store_true")
a = ap.parse_args()
q = lambda r: r.get("quality_cost", r["cost"])
rows = json.loads(Path(a.results).read_text())
lay = json.loads(Path(a.layout).read_text())
d = E.load_designs(ids=[a.id])[0]
k = next(k for k, r in enumerate(rows) if r["id"] == a.id)
n = E.evaluate(d, lay, runtime=rows[k].get("runtime", 0.0) + a.seconds)
nviol = lambda r: sum(v for key, v in r["soft"].items() if key != "n_soft")
if a.by_hpwl:
    better = n["feasible"] and nviol(n) <= nviol(rows[k]) and n["hpwl"] < rows[k]["hpwl"] * (1 - 1e-9)
else:
    better = n["feasible"] and q(n) < q(rows[k]) - 1e-9
if better:
    n["layout"] = lay
    n["picked"] = a.picked or rows[k].get("picked")
    print(f"design {a.id}: {q(rows[k]):.4f} -> {q(n):.4f}, HPWL gap {rows[k]['hpwl_gap']:+.4f} -> {n['hpwl_gap']:+.4f}")
    rows[k] = n
    Path(a.results).write_text(json.dumps(rows))
else:
    print(f"design {a.id}: kept {q(rows[k]):.4f} (new {q(n):.4f}, HPWL gap {n.get('hpwl_gap', 0):+.4f}, "
          f"feasible {n['feasible']})")

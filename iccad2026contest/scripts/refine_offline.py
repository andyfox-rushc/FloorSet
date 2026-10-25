"""Pair re-cut refinement (rl/refine.py) of every layout in a prime_evaluate.py
result file, re-scored against the ground truth.

  python scripts/refine_offline.py in.json out.json [--seconds 30]
"""
import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
import prime_evaluate as E  # noqa: E402
from rl import partition as PT  # noqa: E402
from rl.refine import Refiner  # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("inp")
    ap.add_argument("out")
    ap.add_argument("--seconds", type=float, default=30.0)
    a = ap.parse_args()
    rows = json.loads(Path(a.inp).read_text())
    designs = {d["id"]: d for d in E.load_designs(ids=[r["id"] for r in rows])}
    out = []
    for r in rows:
        d = designs[r["id"]]
        if not r.get("feasible") or not r.get("layout"):
            out.append(r)
            continue
        args = (d["block_count"], d["area_targets"], d["b2b"], d["p2b"], d["pins_pos"], d["constraints"],
                d["target_polygons"])
        R = Refiner(PT.Problem(*args), r["layout"])
        st = R.run(time_budget=a.seconds)
        lay = R.layout()
        n = E.evaluate(d, lay, runtime=r.get("runtime", 0.0) + st["seconds"])
        if not n["feasible"]:                 # never worse than the input
            out.append(r)
            continue
        n["layout"] = [list(map(list, v)) for v in lay]
        n["refine"] = st
        out.append(n)
        q = lambda x: x.get("quality_cost", x["cost"])
        print(f"  design {d['id']:3d} n={d['block_count']:3d}: {q(r):.4f} -> {q(n):.4f} ({st['moves']} moves, {st['seconds']:.1f}s)", flush=True)
    Path(a.out).write_text(json.dumps(out))
    E._summary(rows, "input")
    E._summary(out, "refined")


if __name__ == "__main__":
    main()

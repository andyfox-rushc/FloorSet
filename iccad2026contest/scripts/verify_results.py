"""Re-score the layouts stored in prime_evaluate.py result files against the ground
truth, ignoring the scores saved with them; prints the summary line and the
counts the reports quote (at 1.000, HPWL under GT, within 5%).

  python scripts/verify_results.py prime_results/final_1023.json [more.json ...]
"""
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
import prime_evaluate as E  # noqa: E402

for path in sys.argv[1:]:
    rows = json.loads(Path(path).read_text())
    designs = {d["id"]: d for d in E.load_designs(ids=[r["id"] for r in rows])}
    out = [E.evaluate(designs[r["id"]], r["layout"], runtime=r.get("runtime", 0.0)) for r in rows]
    q = lambda r: r.get("quality_cost", r["cost"])
    E._summary(out, path)
    print(f"  at 1.000: {sum(r['feasible'] and q(r) <= 1 + 1e-6 for r in out)} | "
          f"HPWL under GT: {sum(r.get('hpwl_gap', 1) < 0 for r in out)} | "
          f"within 5%: {sum(q(r) <= 1.05 for r in out)}")

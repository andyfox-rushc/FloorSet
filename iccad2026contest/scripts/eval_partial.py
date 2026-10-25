"""Score the layouts of a prime_evaluate.py .partial file (id, runtime, layout) into
ordinary result rows, e.g. to merge a stopped run with a rerun.

  python scripts/eval_partial.py run.json.partial out.json
"""
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
import prime_evaluate as E  # noqa: E402

rows = json.loads(Path(sys.argv[1]).read_text())
designs = {d["id"]: d for d in E.load_designs(ids=[r["id"] for r in rows])}
out = []
for r in rows:
    n = E.evaluate(designs[r["id"]], r["layout"], runtime=r.get("runtime", 0.0))
    n["layout"] = r["layout"]
    out.append(n)
Path(sys.argv[2]).write_text(json.dumps(out))
print(f"{len(out)} rows, {sum(x['feasible'] for x in out)} feasible")

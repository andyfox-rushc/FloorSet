"""Best-of-both from two prime_evaluate.py result files, picked exactly as
prime_combined.py does at solve time (prime_evaluate.relative_cost with the
first file's layout as the reference -- no ground truth), then scored
against the ground truth. Saves re-running both flows.

  python scripts/prime_best_of.py partition.json packer.json --output combined.json
"""
import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import prime_evaluate as E  # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("first", help="partition-flow results (reference layout)")
    ap.add_argument("second", help="packer-flow results")
    ap.add_argument("--output")
    a = ap.parse_args()
    A = {r["id"]: r for r in json.loads(Path(a.first).read_text())}
    B = {r["id"]: r for r in json.loads(Path(a.second).read_text())}
    ids = sorted(set(A) & set(B))
    designs = {d["id"]: d for d in E.load_designs(ids=ids)}
    out, picked = [], {"first": 0, "second": 0}
    for i in ids:
        d = designs[i]
        args = (d["block_count"], d["area_targets"], d["b2b"], d["p2b"], d["pins_pos"], d["constraints"],
                d["target_polygons"])
        la, lb = A[i]["layout"], B[i]["layout"]
        if la is None or lb is None:
            choice = "first" if lb is None else "second"
        else:
            ca, oka = E.relative_cost(args, la, la)
            cb, okb = E.relative_cost(args, lb, la)
            choice = "first" if oka and (not okb or ca <= cb) else "second"
        picked[choice] += 1
        lay = la if choice == "first" else lb
        r = E.evaluate(d, lay, A[i]["runtime"] + B[i]["runtime"])
        r["layout"], r["picked"] = lay, choice
        out.append(r)
    print(f"picked: {picked}")
    E._summary(out, "best of both")
    if a.output:
        Path(a.output).write_text(json.dumps(out, indent=1))


if __name__ == "__main__":
    main()

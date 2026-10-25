"""Partition flow seeded from another flow's layouts (10-21 experiment): each
design's block centroids in a result file, scaled into the exact frame, seed
the partition search -- alone or alongside the usual learned + analytic seeds.

  python scripts/partition_seed_test.py prime_results/contour_pack20.json --ids 5 12 --output out.json
"""
import argparse
import json
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))


def main():
    import prime_evaluate as E
    import prime_partition as PP
    from my_optimizer import _pmap
    from rl import partition as PT
    ap = argparse.ArgumentParser()
    ap.add_argument("source")
    ap.add_argument("--ids", type=int, nargs="*")
    ap.add_argument("--only", action="store_true", help="source seeds only (else alongside the usual ones)")
    ap.add_argument("--output")
    a = ap.parse_args()
    src = {r["id"]: r for r in json.loads(Path(a.source).read_text())}
    out = []
    for d in E.load_designs(ids=a.ids):
        t0 = time.time()
        args = (d["block_count"], d["area_targets"], d["b2b"], d["p2b"], d["pins_pos"], d["constraints"],
                d["target_polygons"])
        P = PT.Problem(*args)
        s = PP.layout_seeds(src[d["id"]]["layout"], P.W, P.H)
        sets = [s]
        if not a.only:
            sets += PP.seed_sets(*args, P)
        lay = PT.solve_parallel(P, sets, max(10.0, PP.TIME * (d["block_count"] / 77) ** 2), PP.RESTARTS, pmap=_pmap)
        r = E.evaluate(d, lay)
        r["layout"], r["runtime"] = lay, time.time() - t0
        print(f"  design {d['id']:3d} n={d['block_count']:3d}: {r.get('quality_cost', r['cost']):.4f} "
              f"feasible={r['feasible']} {r['runtime']:.0f}s", flush=True)
        out.append(r)
    E._summary(out, Path(a.source).stem + " seeds")
    if a.output:
        Path(a.output).write_text(json.dumps(out, indent=1))


if __name__ == "__main__":
    main()

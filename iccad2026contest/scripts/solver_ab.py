"""Run my_optimizer.MyOptimizer.solve on validation designs one at a time
(like the contest evaluator) and print cost / runtime per design.

  FLOORSET_PROCS=1 python scripts/solver_ab.py --ids 88 89 ... --out x.json
"""
import argparse
import json
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))
sys.path.insert(0, str(Path(__file__).parent))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ids", type=int, nargs="+", required=True)
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    import polish_bench as PB
    PB._init()
    from my_optimizer import MyOptimizer
    from rl.data import from_validation_sample
    opt = MyOptimizer()
    res = {}
    for t in a.ids:
        inst = from_validation_sample(PB.EV, t)
        s = PB.EV.dataset[t]["input"]
        t0 = time.time()
        pos = opt.solve(inst.block_count, inst.area_targets, inst.b2b_connectivity, inst.p2b_connectivity,
                        inst.pins_pos, inst.constraints, inst.target_positions)
        secs = time.time() - t0
        r = PB._score(inst, pos)
        res[t] = dict(cost=r.cost, hpwl_gap=r.hpwl_gap, area_gap=r.area_gap, vrel=r.violations_relative,
                      feasible=r.is_feasible, secs=secs, n=inst.block_count, positions=[list(map(float, q)) for q in pos])
        print(f"design {t} n {inst.block_count} cost {r.cost:.3f} secs {secs:.1f}", flush=True)
    print(f"mean cost {sum(v['cost'] for v in res.values()) / len(res):.3f}")
    json.dump(res, open(a.out, "w"), indent=1)


if __name__ == "__main__":
    main()

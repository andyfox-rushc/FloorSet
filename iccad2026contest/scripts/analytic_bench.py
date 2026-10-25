"""Roadmap step 1 benchmark (10-04): on cached policy layouts, compare per
design the current polish against analytic-placement candidates, and the
best-of pick made WITHOUT ground truth (rl/repack.py contest_relative_cost,
relative to the policy layout) -- what my_optimizer could do at solve time.

  python scripts/analytic_bench.py --cache cache_installed_2s.pkl --out analytic_bench.json
"""
import argparse
import json
import pickle
import sys
import time
from multiprocessing import get_context
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))
import torch

EV = None
CANDS = None


def _init(cands):
    global EV, CANDS
    torch.set_num_threads(1)
    from iccad2026_evaluate import ContestEvaluator
    EV = ContestEvaluator("../", verbose=False)
    EV._load_dataset()
    CANDS = cands


def _one(args):
    tid, rl = args
    from rl.analytic import analytic_solve
    from rl.data import from_validation_sample
    from rl.repack import contest_relative_cost, repack_polish
    from rl.slicing import slicing_polish
    from scripts.polish_bench import _score
    inst = from_validation_sample(EV, tid)

    def pol(p, budget):
        return repack_polish(inst, slicing_polish(inst, p, time_budget=budget / 2), time_budget=budget / 2,
                             seed=tid)
    out = {}
    for name in CANDS:
        # name = kind[:itN][:polX], kind = base | refine<ov_start> | scratch<util>
        parts = name.split(":")
        kind, opts = parts[0], dict((q[:2], float(q[2:]) if q[:2] == "po" else int(q[2:])) for q in parts[1:])
        it, budget = opts.get("it", 500), opts.get("po", 1.0)
        t = time.time()
        if kind == "base":
            p = pol(rl, budget)
        elif kind.startswith("refine"):
            p = pol(analytic_solve(inst, init_positions=rl, util=0.85, iters=min(it, 300),
                                   ov_start=float(kind[6:]), seed=tid), budget)
        elif kind.startswith("scratch"):
            p = pol(analytic_solve(inst, util=float(kind[7:]), iters=it, seed=tid), budget)
        else:
            raise ValueError(name)
        secs = time.time() - t
        proxy, ok = contest_relative_cost(inst, p, rl)
        r = _score(inst, p)
        out[name] = dict(cost=r.cost, hpwl_gap=r.hpwl_gap, area_gap=r.area_gap, vrel=r.violations_relative,
                         feasible=r.is_feasible, proxy=proxy if ok else float("inf"), secs=secs)
    return tid, out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--cache", default="cache_installed_2s.pkl")
    ap.add_argument("--cands", nargs="+", default=["base", "refine0.5", "refine0.1", "scratch0.85", "scratch0.95"])
    ap.add_argument("--workers", type=int, default=8)
    ap.add_argument("--out", default="analytic_bench.json")
    a = ap.parse_args()
    cache = pickle.load(open(a.cache, "rb"))["positions"]
    with get_context("forkserver").Pool(a.workers, initializer=_init, initargs=(a.cands,)) as pool:
        res = dict(pool.map(_one, sorted(cache.items()), chunksize=1))
    json.dump(res, open(a.out, "w"), indent=1)
    ids = sorted(res)
    avg = lambda f: sum(f(res[i]) for i in ids) / len(ids)
    print(f"{'candidate':14s} {'cost':>6s} {'hpwl':>6s} {'area':>6s} {'vrel':>6s} {'secs':>5s}")
    for c in a.cands:
        print(f"{c:14s} {avg(lambda r: r[c]['cost']):6.3f} {avg(lambda r: r[c]['hpwl_gap']):6.3f} "
              f"{avg(lambda r: r[c]['area_gap']):6.3f} {avg(lambda r: r[c]['vrel']):6.3f} "
              f"{avg(lambda r: r[c]['secs']):5.2f}")
    pick = lambda r, cs: min(cs, key=lambda c: r[c]["proxy"])
    groups = [a.cands]   # subsets: analyse the saved JSON
    for cs in groups:
        print(f"proxy pick {'+'.join(cs)[:60]:60s} {avg(lambda r: r[pick(r, cs)]['cost']):6.3f}   "
              f"(oracle {avg(lambda r: min(r[c]['cost'] for c in cs)):6.3f}, "
              f"time {avg(lambda r: sum(r[c]['secs'] for c in cs)):5.2f}s)")


if __name__ == "__main__":
    main()

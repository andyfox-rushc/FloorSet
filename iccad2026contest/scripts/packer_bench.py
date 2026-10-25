"""Compact packer bench (10-05).

  python scripts/packer_bench.py cache --out cache_bestof.pkl   # best-of winners, 100 designs
  python scripts/packer_bench.py run --cache cache_bestof.pkl --budget 1.5
"""
import argparse
import json
import math
import os
import pickle
import statistics as st
import sys
import time
from multiprocessing import get_context
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))
sys.path.insert(0, str(Path(__file__).parent))
import polish_bench as PB  # noqa: E402

CANDS = os.environ.get("PB_CANDS", "base,scratch0.85:it250:po1.0,scratch0.95:it250:po1.0")


def _cache_one(args):
    tid, ckpt = args
    from rl.analytic import best_of
    from rl.checkpoint_meta import utilization_ladder
    from rl.data import from_validation_sample
    from rl.finetune import finetune_and_solve
    from rl.networks import ActorCritic, load_state
    net = ActorCritic()
    load_state(net, ckpt)
    inst = from_validation_sample(PB.EV, tid)
    pos, _ = finetune_and_solve(inst, net=net, time_budget=2.0, seed=tid, utilizations=utilization_ladder(ckpt))
    return tid, best_of(inst, pos, CANDS.split(","), seed=tid)


def _cacheall_one(args):
    """All best-of candidates (polished), plus the policy layout they're scored against."""
    tid, ckpt = args
    from rl.analytic import analytic_solve
    from rl.checkpoint_meta import utilization_ladder
    from rl.data import from_validation_sample
    from rl.finetune import finetune_and_solve
    from rl.networks import ActorCritic, load_state
    from rl.repack import repack_polish
    from rl.slicing import slicing_polish
    net = ActorCritic()
    load_state(net, ckpt)
    inst = from_validation_sample(PB.EV, tid)
    rl_pos, _ = finetune_and_solve(inst, net=net, time_budget=2.0, seed=tid, utilizations=utilization_ladder(ckpt))
    out = {}
    for name in CANDS.split(","):
        kind = name.split(":")[0]
        sd = tid + int(([q[2:] for q in name.split(":")[1:] if q.startswith("sd")] or ["0"])[0])
        it = int(([q[2:] for q in name.split(":")[1:] if q.startswith("it")] or ["250"])[0])
        p = rl_pos if kind == "base" else analytic_solve(inst, util=float(kind[7:]), iters=it, seed=sd)
        if os.environ.get("PB_RAW"):         # unpolished candidates
            out[name] = p
        else:
            po = float(os.environ.get("PB_PO", "1.0"))
            out[name] = repack_polish(inst, slicing_polish(inst, p, time_budget=po / 2), time_budget=po / 2, seed=tid)
    return tid, dict(rl=rl_pos, cands=out)


def _multi_one(args):
    """Pack every candidate for `budget` each; keep the best by relative cost."""
    tid, entry, budget, topk = args
    from rl.data import from_validation_sample
    from rl.packer import compact_pack
    from rl.repack import contest_relative_cost
    inst = from_validation_sample(PB.EV, tid)
    ref = entry["rl"]
    ranked = sorted(entry["cands"].values(), key=lambda p: contest_relative_cost(inst, p, ref)[0])[:topk]
    t = time.time()
    if os.environ.get("PB_SCALE"):      # budget x n / PB_SCALE (same total at mean n)
        pw = float(os.environ.get("PB_POW", "1"))
        budget = budget * (inst.block_count / float(os.environ["PB_SCALE"])) ** pw
    outs = [compact_pack(inst, p, time_budget=budget, seed=tid) for p in ranked]
    secs = time.time() - t
    best = min(outs, key=lambda p: contest_relative_cost(inst, p, ref)[0])
    m = lambda r: dict(cost=r.cost, hpwl_gap=r.hpwl_gap, area_gap=r.area_gap, vrel=r.violations_relative,
                       feasible=r.is_feasible)
    return tid, dict(before=m(PB._score(inst, ranked[0])), after=m(PB._score(inst, best)), secs=secs, changed=True)


def _run_one(args):
    tid, pos, budget = args
    from rl.data import from_validation_sample
    from rl.packer import compact_pack
    inst = from_validation_sample(PB.EV, tid)
    r0 = PB._score(inst, pos)
    t = time.time()
    p = compact_pack(inst, pos, time_budget=budget, seed=tid)
    secs = time.time() - t
    r = PB._score(inst, p)
    m = lambda r: dict(cost=r.cost, hpwl_gap=r.hpwl_gap, area_gap=r.area_gap, vrel=r.violations_relative,
                       feasible=r.is_feasible)
    return tid, dict(before=m(r0), after=m(r), secs=secs, changed=p is not pos)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("stage", choices=["cache", "run", "cacheall", "multi"])
    ap.add_argument("--topk", type=int, default=3)
    ap.add_argument("--checkpoint", default="checkpoints/policy.pt")
    ap.add_argument("--cache", default="cache_bestof.pkl")
    ap.add_argument("--budget", type=float, default=1.5)
    ap.add_argument("--cases", type=int, default=100)
    ap.add_argument("--workers", type=int, default=6)
    ap.add_argument("--out", default="packer_bench.json")
    a = ap.parse_args()
    with get_context("forkserver").Pool(a.workers, initializer=PB._init) as pool:
        if a.stage == "cache":
            res = dict(pool.imap_unordered(_cache_one, [(t, a.checkpoint) for t in range(a.cases)]))
            pickle.dump(res, open(a.cache, "wb"))
            print(f"cached {len(res)} layouts -> {a.cache}")
            return
        if a.stage == "cacheall":
            res = dict(pool.imap_unordered(_cacheall_one, [(t, a.checkpoint) for t in range(a.cases)]))
            pickle.dump(res, open(a.cache, "wb"))
            print(f"cached {len(res)} designs -> {a.cache}")
            return
        cache = pickle.load(open(a.cache, "rb"))
        ids = sorted(cache)[:a.cases]
        if a.stage == "multi":
            res = dict(pool.imap_unordered(_multi_one, [(t, cache[t], a.budget, a.topk) for t in ids]))
        else:
            res = dict(pool.imap_unordered(_run_one, [(t, cache[t], a.budget) for t in ids]))
    avg = lambda k, f: st.mean(res[t][k][f] for t in ids)
    for k in ("before", "after"):
        print(f"{k:7s} cost {avg(k, 'cost'):.3f} hpwl {avg(k, 'hpwl_gap'):.3f} area {avg(k, 'area_gap'):.3f} "
              f"vrel {avg(k, 'vrel'):.3f} feasible {sum(res[t][k]['feasible'] for t in ids)}")
    print(f"changed {sum(res[t]['changed'] for t in ids)}/{len(ids)}, secs {st.mean(res[t]['secs'] for t in ids):.2f}")
    N = {r["design"]: r["n"] for r in json.load(open("bestof_diag.json"))}   # block counts
    wt = {t: math.exp(N[t] / 12) for t in ids}
    for k in ("before", "after"):
        print(f"{k:7s} weighted score {sum(res[t][k]['cost'] * wt[t] for t in ids) / sum(wt.values()):.3f}")
    json.dump({str(t): res[t] for t in ids}, open(a.out, "w"), indent=1)


if __name__ == "__main__":
    main()

"""Upper bound for imitation (10-07): if a policy produced the ground truth's
ARRANGEMENT, spread out like a grid policy's layout, how close would the
packer get to the ground truth?

Each ground-truth layout is spread by scaling block positions (not sizes)
about the origin by --spread (legal: scaling only opens gaps), then packed
with the compact packer, as a candidate would be.

  python scripts/gt_arrangement_bound.py --spread 1.15 --budget 6 --out gt_bound.json
"""
import argparse
import json
import math
import sys
from multiprocessing import get_context
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))
sys.path.insert(0, str(Path(__file__).parent))
import polish_bench as PB
import os
FIT = os.environ.get("FIT", "0") == "1"


def _one(args):
    tid, spread, budget, shape_sigma = args
    from rl.data import from_validation_sample
    from rl.packer import compact_pack, pack_budget, squeeze, relax_hpwl
    inst = from_validation_sample(PB.EV, tid)
    _, gt = PB.EV._extract_baseline(tid, PB.EV.dataset[tid]["label"], inst.b2b_connectivity,
                                    inst.p2b_connectivity, inst.pins_pos, inst.block_count)
    gt = [tuple(map(float, q[:4])) for q in gt]
    pre = inst.constraints[:inst.block_count, 1] != 0
    if shape_sigma > 0:
        # keep the ground truth's sequence pair (always legal), perturb soft shapes, pack
        import numpy as np
        from rl import packer as K, repack as R
        P = R._problem(inst, gt)
        p = P["pos"]
        gp, gm = K.sequence_pair(p[:, 0], p[:, 1], p[:, 2], p[:, 3])
        rng = np.random.default_rng(tid)
        soft = ~P["fixed"] & ~P["preplaced"]
        f = np.exp(rng.normal(0, shape_sigma, P["n"]))
        w = np.where(soft, p[:, 2] * f, p[:, 2])
        h = np.where(soft, P["area"] / w, p[:, 3])
        for g in set(P["mib"][P["mib"] > 0].tolist()):      # MIB groups keep one shape
            m = np.where(P['mib'] == g)[0]
            known = m[~soft[m]]          # a fixed / preplaced member sets the group's shape
            ms = m[soft[m]]
            if len(known):
                w[ms], h[ms] = p[known[0], 2], p[known[0], 3]
            elif len(ms):
                w[ms] = w[ms[0]]
                h[ms] = P['area'][ms] / w[ms]
        Bt = K._Batch(P, 1.0, 1.0, (0.0, 0.0))
        x, y, over = Bt.pack(gm[None], np.argsort(gp)[None], np.argsort(gm)[None], w[None], h[None])
        start = [(float(x[0, i]), float(y[0, i]), float(w[i]), float(h[i])) for i in range(P["n"])]
        if FIT:     # fit shapes on the true arrangement (resolves preplaced collisions too)
            from rl.sizing import fit_shapes
            fitted = fit_shapes(inst, start, iters=1000, lr=0.02, seqs=(gp, gm))
            if fitted is not start:
                start = fitted
        if over[0] > 1e-6 and start is not None and not PB._score(inst, start).is_feasible:
            fixed = K._origin_repair(Bt, gp, gm, w, h)      # fallback: resolve collisions by moving blocks
            if fixed is not None:
                gp, gm = fixed
                x, y, over = Bt.pack(gm[None], np.argsort(gp)[None], np.argsort(gm)[None], w[None], h[None])
                start = [(float(x[0, i]), float(y[0, i]), float(w[i]), float(h[i])) for i in range(P["n"])]
    else:
        start = [q if pre[i] else (q[0] * spread, q[1] * spread, q[2], q[3]) for i, q in enumerate(gt)]
    s0 = PB._score(inst, start)
    out = compact_pack(inst, start, time_budget=pack_budget(inst.block_count, budget), seed=0)
    out = relax_hpwl(inst, squeeze(inst, out))
    s1 = PB._score(inst, out)
    return dict(design=tid, n=inst.block_count, spread_cost=s0.cost if s0.is_feasible else None,
                cost=s1.cost, feasible=s1.is_feasible, hpwl_gap=s1.hpwl_gap, area_gap=s1.area_gap,
                vrel=s1.violations_relative)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--spread", type=float, default=1.15)
    ap.add_argument("--budget", type=float, default=6)
    ap.add_argument("--shape-sigma", type=float, default=0.0,
                    help="instead of spreading: keep the GT sequence pair, perturb soft shapes by exp(N(0, s))")
    ap.add_argument("--workers", type=int, default=4)
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    with get_context("forkserver").Pool(a.workers, initializer=PB._init) as pool:
        res = pool.map(_one, [(t, a.spread, a.budget, a.shape_sigma) for t in range(100)])
    json.dump(res, open(a.out, "w"), indent=1)
    W = sum(math.exp(r["n"] / 12) for r in res)
    m = lambda k: sum(r[k] for r in res) / len(res)
    sp = [r for r in res if r["spread_cost"] is not None]
    tag = f"shapes sigma {a.shape_sigma}" if a.shape_sigma > 0 else f"spread x{a.spread}"
    print(f"{tag}: start layouts cost {sum(r['spread_cost'] for r in sp) / max(len(sp), 1):.3f} "
          f"({len(sp)} legal); after packer: cost {m('cost'):.3f} weighted "
          f"{sum(math.exp(r['n'] / 12) * r['cost'] for r in res) / W:.3f} hpwl {m('hpwl_gap'):.3f} "
          f"area {m('area_gap'):.3f} vrel {m('vrel'):.3f} feasible {sum(r['feasible'] for r in res)}/100")


if __name__ == "__main__":
    main()

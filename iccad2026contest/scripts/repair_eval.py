"""Repair net check (10-16): does repairing a layout's arrangement lower the packed cost?

  corrupted  ground-truth sequence pair with k blocks moved: packed as is vs repaired
  own        placer_r's packed prediction: as is vs repaired once / twice
Each layout -> (repair: new sequence pair from the repair net's relations on that
layout) -> shape fit -> packer (--budget at 77 blocks) -> squeeze + relax.

  python scripts/repair_eval.py --net checkpoints/repair.pt --ids 90 61 54 --out repair_eval.json
"""
import argparse
import json
import math
import sys
from multiprocessing import get_context
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))
sys.path.insert(0, str(Path(__file__).parent))
import numpy as np
import polish_bench as PB

A = None


def _legal(inst, det):
    """to_layout result -> collision-free layout (shape fit, else origin repair) or None."""
    from rl import packer as K, repack as R
    from rl.sizing import fit_shapes
    pos, (gp, gm), over = det
    if over <= 1e-6:
        return pos
    f = fit_shapes(inst, pos, iters=600, lr=0.02, seqs=(gp, gm))
    if f is not pos and PB._score(inst, f).is_feasible:
        return f
    P = R._problem(inst, pos)
    w, h = P["pos"][:, 2].copy(), P["pos"][:, 3].copy()
    Bt = K._Batch(P, 1.0, 1.0, (0.0, 0.0))
    fx = K._origin_repair(Bt, np.asarray(gp), np.asarray(gm), w, h)
    if fx is None:
        return None
    gp, gm = fx
    x, y, o = Bt.pack(gm[None], np.argsort(gp)[None], np.argsort(gm)[None], w[None], h[None])
    return [(float(x[0, i]), float(y[0, i]), float(w[i]), float(h[i])) for i in range(len(w))] if o[0] <= 1e-6 else None


def _finish(inst, pos, budget):
    from rl import packer as K
    from rl.sizing import fit_shapes
    if pos is None:
        return None
    pos = fit_shapes(inst, pos, iters=1000, lr=0.02)
    out = K.compact_pack(inst, pos, time_budget=K.pack_budget(inst.block_count, budget), seed=0)
    s = PB._score(inst, K.relax_hpwl(inst, K.squeeze(inst, out)))
    return s.cost if s.is_feasible else None


def _job(args):
    t, kind, k, seed, rounds, net_path, budget = args
    from rl.data import from_validation_sample
    from rl import learned_placer as L
    from train_repair import corrupted_layout
    import torch
    inst = from_validation_sample(PB.EV, t)
    n = inst.block_count
    _, gt = PB.EV._extract_baseline(t, PB.EV.dataset[t]["label"], inst.b2b_connectivity,
                                    inst.p2b_connectivity, inst.pins_pos, n)
    sol = torch.tensor([[float(q[2]), float(q[3]), float(q[0]), float(q[1])] for q in gt])
    if kind == "corrupted":
        lay = _legal(inst, corrupted_layout(inst, sol, np.random.default_rng(1000 * t + 10 * k + seed),
                                            sigma=0.0, k=k, full=True))
    else:
        lay = _legal(inst, L.to_layout(inst, L.load("checkpoints/placer_r.pt"), use_rel=True))
    net = L.load(net_path)
    for _ in range(rounds):
        if lay is None:
            break
        lay = _legal(inst, L.to_layout(inst, net, use_rel=True, layout=lay))
    return dict(t=t, n=n, kind=kind, k=k, seed=seed, rounds=rounds, cost=_finish(inst, lay, budget))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--net", required=True)
    ap.add_argument("--ids", type=int, nargs="+", required=True)
    ap.add_argument("--budget", type=float, default=20)
    ap.add_argument("--workers", type=int, default=4)
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    jobs = []
    for t in a.ids:
        for k, seed in ((2, 0), (5, 0)):
            jobs += [(t, "corrupted", k, seed, r, a.net, a.budget) for r in (0, 1)]
        jobs += [(t, "own", 0, 0, r, a.net, a.budget) for r in (0, 1, 2)]
    with get_context("forkserver").Pool(a.workers, initializer=PB._init) as pool:
        res = pool.map(_job, jobs, chunksize=1)
    json.dump(res, open(a.out, "w"), indent=1)
    keys = sorted({(r["kind"], r["k"], r["rounds"]) for r in res})
    for t in a.ids:
        print(t, "  ".join(f"{kd[:3]}{k}r{rd} {next((r['cost'] for r in res if r['t'] == t and (r['kind'], r['k'], r['rounds']) == (kd, k, rd)), None) or float('nan'):.3f}"
                           for kd, k, rd in keys))
    for kd, k, rd in keys:
        rr = [r for r in res if (r["kind"], r["k"], r["rounds"]) == (kd, k, rd) and r["cost"] is not None]
        W = sum(math.exp(r["n"] / 12) for r in rr)
        print(f"{kd:9s} k{k} rounds {rd}: mean {np.mean([r['cost'] for r in rr]):.4f} weighted "
              f"{sum(math.exp(r['n'] / 12) * r['cost'] for r in rr) / W:.4f} ({len(rr)} designs)")


if __name__ == "__main__":
    main()

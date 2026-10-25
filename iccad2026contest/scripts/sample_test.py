"""Sampling test (10-08): does the best of N noisy placer samples (ranked by
the cheap exact packing) beat the deterministic prediction, before and after
the normal fit + packer chain?
  python scripts/sample_test.py --ckpt checkpoints/placer_c.pt --n 16 --noise 0.02 --aspect-noise 0.2
"""
import argparse, json, math, sys, time
from multiprocessing import get_context
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent)); sys.path.insert(0, str(Path(__file__).parent))
import polish_bench as PB


def _one(args):
    tid, ckpt, n, noise, anoise, budget = args
    import numpy as np, torch
    torch.set_num_threads(1)
    from rl.data import from_validation_sample
    from rl import learned_placer as L, packer as K, repack as R
    from rl.analytic import learned_candidate
    from rl.sizing import fit_shapes
    inst = from_validation_sample(PB.EV, tid)
    net = L.load(ckpt)
    rng = np.random.default_rng(tid)
    det, seqs, over = L.to_layout(inst, net)
    ref = det
    cands = [(det, seqs, over)] + [L.to_layout(inst, net, noise, anoise, rng) for _ in range(n)]
    def cheap(c):      # exact packing as is; collisions (over > 0) rank last
        pos, _, ov = c
        if ov > 1e-6:
            return float('inf')
        cost, ok = R.contest_relative_cost(inst, pos, ref)
        return cost if ok else float('inf')
    scores = [cheap(c) for c in cands]
    k = int(np.argmin(scores))
    def finish(c):
        pos, (gp, gm), ov = c
        if ov > 1e-6:
            f = fit_shapes(inst, pos, iters=600, lr=0.02, seqs=(gp, gm))
            if f is pos:
                return None
            pos = f
        pos = fit_shapes(inst, pos, iters=1000, lr=0.02)
        out = K.compact_pack(inst, pos, time_budget=K.pack_budget(inst.block_count, budget), seed=0)
        s = PB._score(inst, K.relax_hpwl(inst, K.squeeze(inst, out)))
        return s.cost if s.is_feasible else None
    raw = lambda c: PB._score(inst, c[0]).cost if c[2] <= 1e-6 and PB._score(inst, c[0]).is_feasible else None
    return dict(design=tid, n=inst.block_count, det_raw=raw(cands[0]), best_raw=raw(cands[k]) if k else raw(cands[0]),
                picked=k, det_final=finish(cands[0]), best_final=finish(cands[k]) if k else None,
                n_ok=sum(s < float('inf') for s in scores))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", default="checkpoints/placer_c.pt")
    ap.add_argument("--n", type=int, default=16)
    ap.add_argument("--noise", type=float, default=0.02)
    ap.add_argument("--aspect-noise", type=float, default=0.2)
    ap.add_argument("--budget", type=float, default=6)
    ap.add_argument("--ids", default="0-99")
    ap.add_argument("--workers", type=int, default=3)
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    lo, _, hi = a.ids.partition("-")
    ids = list(range(int(lo), int(hi or lo) + 1))
    with get_context("forkserver").Pool(a.workers, initializer=PB._init) as pool:
        res = pool.map(_one, [(t, a.ckpt, a.n, a.noise, a.aspect_noise, a.budget) for t in ids])
    json.dump(res, open(a.out, "w"), indent=1)
    ok = [r for r in res if r["det_final"] is not None]
    for r in ok:
        if r["best_final"] is None:
            r["best_final"] = r["det_final"]
    W = sum(math.exp(r["n"] / 12) for r in ok)
    w = lambda k: sum(math.exp(r["n"] / 12) * r[k] for r in ok) / W
    print(f"noise {a.noise} aspect {a.aspect_noise} n {a.n}: picked a sample on {sum(r['picked'] > 0 for r in res)}/{len(res)}; "
          f"after fit+packer({a.budget}): deterministic {w('det_final'):.4f} vs best-of-{a.n} {w('best_final'):.4f} "
          f"(better on {sum(r['best_final'] < r['det_final'] - 1e-9 for r in ok)}, worse on {sum(r['best_final'] > r['det_final'] + 1e-9 for r in ok)}); "
          f"min of both {sum(math.exp(r['n'] / 12) * min(r['det_final'], r['best_final']) for r in ok) / W:.4f}")


if __name__ == "__main__":
    main()

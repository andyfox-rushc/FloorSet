"""Evaluate the learned placer as a candidate (10-07): predicted arrangement
-> pack from origin (collisions repaired) -> fit_shapes -> packer -> squeeze ->
relaxation, scored against ground truth.

  python scripts/placer_eval.py --ckpt checkpoints/placer.pt --budget 6 --ids 0-99
"""
import argparse, json, math, sys, time
from multiprocessing import get_context
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent)); sys.path.insert(0, str(Path(__file__).parent))
import polish_bench as PB


def _one(args):
    tid, ckpt, budget = args
    import torch
    torch.set_num_threads(1)
    from rl.data import from_validation_sample
    from rl import learned_placer as L, packer as K, repack as R
    from rl.sizing import fit_shapes
    inst = from_validation_sample(PB.EV, tid)
    net = L.load(ckpt)
    t0 = time.time()
    pos, (gp, gm), over = L.to_layout(inst, net)
    if over > 1e-6:
        P = R._problem(inst, pos)
        Bt = K._Batch(P, 1.0, 1.0, (0.0, 0.0))
        w, h = P["pos"][:, 2].copy(), P["pos"][:, 3].copy()
        fitted = fit_shapes(inst, pos, iters=600, lr=0.02, seqs=(gp, gm))
        if fitted is not pos and PB._score(inst, fitted).is_feasible:
            pos = fitted
        else:
            fx = K._origin_repair(Bt, gp, gm, w, h)
            if fx is not None:
                gp, gm = fx
                x, y, o = Bt.pack(gm[None], torch.tensor(gp).argsort()[None].numpy(), torch.tensor(gm).argsort()[None].numpy(), w[None], h[None])
                pos = [(float(x[0, i]), float(y[0, i]), float(w[i]), float(h[i])) for i in range(len(w))]
    s0 = PB._score(inst, pos)
    fit = fit_shapes(inst, pos, iters=1000, lr=0.02) if s0.is_feasible else pos
    s1 = PB._score(inst, fit)
    out = K.compact_pack(inst, fit, time_budget=K.pack_budget(inst.block_count, budget), seed=0)
    out = K.relax_hpwl(inst, K.squeeze(inst, out))
    s2 = PB._score(inst, out)
    return dict(design=tid, n=inst.block_count, raw=s0.cost if s0.is_feasible else None,
                fit=s1.cost if s1.is_feasible else None, cost=s2.cost if s2.is_feasible else None,
                hpwl=s2.hpwl_gap, area=s2.area_gap, vrel=s2.violations_relative, secs=time.time() - t0)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", default="checkpoints/placer.pt")
    ap.add_argument("--budget", type=float, default=6)
    ap.add_argument("--ids", default="0-99")
    ap.add_argument("--workers", type=int, default=3)
    ap.add_argument("--out", default=None)
    a = ap.parse_args()
    lo, _, hi = a.ids.partition("-")
    ids = list(range(int(lo), int(hi or lo) + 1))
    with get_context("forkserver").Pool(a.workers, initializer=PB._init) as pool:
        res = pool.map(_one, [(t, a.ckpt, a.budget) for t in ids])
    if a.out:
        json.dump(res, open(a.out, "w"), indent=1)
    ours = {r["test_id"]: r for r in json.load(open("official_eval_1007_tol.json"))["test_results"]}
    ok = [r for r in res if r["cost"] is not None]
    W = sum(math.exp(r["n"] / 12) for r in ok)
    wf = lambda vals: sum(math.exp(r["n"] / 12) * v for r, v in zip(ok, vals)) / max(W, 1e-9)
    print(f"learned placer + fit + packer({a.budget}): feasible {len(ok)}/{len(res)}; weighted {wf([r['cost'] for r in ok]):.4f} "
          f"(ours official on same designs {wf([ours[r['design']]['cost'] for r in ok]):.4f}); "
          f"hpwl {sum(r['hpwl'] for r in ok) / max(len(ok), 1):.3f} area {sum(r['area'] for r in ok) / max(len(ok), 1):.3f} "
          f"vrel {sum(r['vrel'] for r in ok) / max(len(ok), 1):.3f}; raw feasible {sum(r['raw'] is not None for r in res)}, "
          f"better than ours on {sum(r['cost'] < ours[r['design']]['cost'] for r in ok)}")


if __name__ == "__main__":
    main()

"""Reward alignment test (10-09): which cheap reward ranks sampled placer
arrangements the way the true final cost does?

For each design, K arrangements are sampled from the learned placer. Each gets
the TRUE final cost (as scripts/placer_eval.py: collisions resolved, fit_shapes
1000 steps, packer budget 6, squeeze, relaxation) and cheap reward variants:
  r0  raw: exact pack, collisions repaired (rl/packer.py _origin_repair)
  r1  r0 + fit_shapes 200 steps
  r2  r1 + 1 s of packer
Reported per variant: mean within-design Spearman rank correlation with the
true cost, and the true-cost regret of picking the variant's best sample.

  python scripts/reward_alignment.py --ckpt checkpoints/placer_d.pt --k 8 --ids 0-99:3
"""
import argparse
import json
import math
import sys
import time
from multiprocessing import get_context
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))
sys.path.insert(0, str(Path(__file__).parent))
import polish_bench as PB

INF = 10.0


def _resolve(inst, L, K, R, sample):
    """Sampled arrangement -> legal layout (collisions repaired), or None."""
    import numpy as np
    pos, (gp, gm), over = sample
    if over <= 1e-6:
        return pos
    P = R._problem(inst, pos)
    p = P["pos"]
    w, h = p[:, 2].copy(), p[:, 3].copy()
    Bt = K._Batch(P, 1.0, 1.0, (0.0, 0.0))
    fx = K._origin_repair(Bt, np.asarray(gp), np.asarray(gm), w, h)
    if fx is None:
        return None
    gp, gm = fx
    x, y, o = Bt.pack(gm[None], np.argsort(gp)[None], np.argsort(gm)[None], w[None], h[None])
    if o[0] > 1e-6:
        return None
    return [(float(x[0, i]), float(y[0, i]), float(w[i]), float(h[i])) for i in range(len(w))]


def _one(args):
    tid, ckpt, k, noise, anoise, seed = args
    import numpy as np
    import torch
    torch.set_num_threads(1)
    from rl.data import from_validation_sample
    from rl import learned_placer as L, packer as K, repack as R
    from rl.sizing import fit_shapes
    inst = from_validation_sample(PB.EV, tid)
    net = L.load(ckpt)
    rng = np.random.default_rng(seed + tid)
    score = lambda pos: (lambda s: s.cost if s.is_feasible else INF)(PB._score(inst, pos))
    rows = []
    t0 = time.time()
    for j in range(k):
        sample = L.to_layout(inst, net, noise, anoise, rng) if j else L.to_layout(inst, net)
        pos = _resolve(inst, L, K, R, sample)
        if pos is None:
            rows.append(dict(r0=INF, r1=INF, r2=INF, true=INF))
            continue
        r0 = score(pos)
        f1 = fit_shapes(inst, pos, iters=200, lr=0.02)
        r1 = score(f1)
        p2 = K.compact_pack(inst, f1, time_budget=1.0, seed=0)
        r2 = score(p2)
        ff = fit_shapes(inst, pos, iters=1000, lr=0.02)
        out = K.compact_pack(inst, ff, time_budget=K.pack_budget(inst.block_count, 6), seed=0)
        true = score(K.relax_hpwl(inst, K.squeeze(inst, out)))
        rows.append(dict(r0=r0, r1=r1, r2=r2, true=true))
    return dict(design=tid, n=inst.block_count, rows=rows, secs=time.time() - t0)


def spearman(a, b):
    import numpy as np
    ra = np.argsort(np.argsort(a))
    rb = np.argsort(np.argsort(b))
    if np.std(ra) == 0 or np.std(rb) == 0:
        return float("nan")
    return float(np.corrcoef(ra, rb)[0, 1])


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", default="checkpoints/placer_d.pt")
    ap.add_argument("--k", type=int, default=8)
    ap.add_argument("--noise", type=float, default=0.02)
    ap.add_argument("--aspect-noise", type=float, default=0.2)
    ap.add_argument("--ids", default="0-99:3", help="lo-hi[:step]")
    ap.add_argument("--workers", type=int, default=6)
    ap.add_argument("--seed", type=int, default=7)
    ap.add_argument("--out", default="reward_alignment.json")
    a = ap.parse_args()
    rng_part, _, step = a.ids.partition(":")
    lo, _, hi = rng_part.partition("-")
    ids = list(range(int(lo), int(hi or lo) + 1, int(step or 1)))
    with get_context("forkserver").Pool(a.workers, initializer=PB._init) as pool:
        res = pool.map(_one, [(t, a.ckpt, a.k, a.noise, a.aspect_noise, a.seed) for t in ids], chunksize=1)
    json.dump(res, open(a.out, "w"), indent=1)
    import numpy as np
    print(f"{len(res)} designs x {a.k} samples (noise {a.noise}, aspect {a.aspect_noise}), "
          f"{np.mean([r['secs'] for r in res]):.0f} s per design")
    true_spread = []
    for r in res:
        t = [x["true"] for x in r["rows"] if x["true"] < INF]
        if len(t) > 1:
            true_spread.append(max(t) - min(t))
    print(f"true-cost spread within a design: mean {np.mean(true_spread):.4f}")
    det_regret = [r["rows"][0]["true"] - min(x["true"] for x in r["rows"]) for r in res if r["rows"][0]["true"] < INF]
    print(f"deterministic prediction vs best sample (true cost): mean regret {np.mean(det_regret):.4f}, "
          f"best sample better on {sum(x > 1e-9 for x in det_regret)}/{len(det_regret)}")
    for v in ("r0", "r1", "r2"):
        rho, regret = [], []
        for r in res:
            ok = [x for x in r["rows"] if x["true"] < INF and x[v] < INF]
            if len(ok) < 3:
                continue
            rho.append(spearman([x[v] for x in ok], [x["true"] for x in ok]))
            pick = min(ok, key=lambda x: x[v])
            regret.append(pick["true"] - min(x["true"] for x in ok))
        rho = [x for x in rho if not math.isnan(x)]
        print(f"{v}: within-design Spearman {np.mean(rho):.3f} (median {np.median(rho):.3f}), "
              f"regret of its pick {np.mean(regret):.4f}")


if __name__ == "__main__":
    main()

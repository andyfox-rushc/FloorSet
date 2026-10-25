"""Fine-tune the slicing-tree builder (rl/tree_policy.py) with a reward closer to
its use: in the partition flow each sampled tree is only the START of an anneal,
so here every sampled tree is annealed for --anneal seconds and the reward is
the cost the anneal reaches (best valid state; best invalid + 1 when none).
REINFORCE against the K-sample mean per design, as scripts/train_tree_policy.py.

Designs run in parallel worker processes (--jobs); each worker computes its
design's gradient and the main process averages them.

  python scripts/train_tree_policy_anneal.py --hours 1.5 --jobs 6 \
      --resume checkpoints/tree_policy_best_8559.pt --out checkpoints/tree_policy_anneal.pt
"""
import argparse
import glob
import os
import sys
import time
from multiprocessing import get_context
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
import numpy as np
import torch

_W = {}


def _init(placer):
    import torch as T
    from rl.prime_placer import load as load_placer
    T.set_num_threads(1)
    _W["seednet"] = load_placer(placer)


def _anneal_cost(part, expr, seconds, seed):
    """seconds > 0: timed anneal; seconds < 0: -seconds is a fixed step count (deterministic)."""
    if seconds < 0:
        res = part.anneal(list(expr), 0.0, seed, max_steps=int(-seconds))
    else:
        res = part.anneal(list(expr), seconds, seed)
    if res is not None:
        return float(res[0]), True
    return float(part.last_best[0]) + 1.0, False


def _job(args):
    """One design: K sampled trees, each annealed; returns (grads, stats) or None."""
    from rl import tree_policy as TP
    from scripts.train_tree_policy import make_case
    state, design, k, seconds, seed = args
    net = TP.TreePolicy()
    net.load_state_dict(state)
    try:
        case = make_case(*design, _W["seednet"])
    except Exception:
        return None
    if case is None:
        return None
    part, ctx, c0 = case
    gen = torch.Generator().manual_seed(seed)
    costs, logps, valid = [], [], 0
    for j in range(k):
        e, lp, _ = TP.rollout(net, part, ctx, rng=gen)
        part.memo.clear()
        c, ok = _anneal_cost(part, e, seconds, seed * 31 + j)
        costs.append(c)
        logps.append(lp)
        valid += ok
    if sum(lp is not None for lp in logps) < 2:
        return None
    cs = torch.tensor(costs)
    adv = -(cs - cs.mean()) / (cs.std() + 1e-3)
    loss = -(adv * torch.stack(logps)).mean()
    net.zero_grad()
    loss.backward()
    grads = {n: p.grad.detach().clone() for n, p in net.named_parameters() if p.grad is not None}
    return grads, (float(cs.mean()), float(cs.min()), valid / k)


def _val_job(args):
    """Held-out design: greedy policy tree vs the seed tree, each annealed (same seconds, seed)."""
    from rl import partition as PT
    from rl import tree_policy as TP
    from scripts.train_tree_policy import make_case
    state, design, seconds = args
    net = TP.TreePolicy()
    net.load_state_dict(state)
    net.eval()
    case = make_case(*design, _W["seednet"])
    if case is None:
        return None
    part, ctx, _ = case
    P = part.P
    r = part.region
    e0 = PT.initial_expression(part.operands, part.seeds, P.area,
                               (r[:, 0].min(), r[:, 1].min(), r[:, 2].max(), r[:, 3].max()), P.groups)
    with torch.no_grad():
        e1, _, _ = TP.rollout(net, part, ctx, greedy=True)
    part.memo.clear()
    c0, _ = _anneal_cost(part, e0, seconds, 1)
    part.memo.clear()
    c1, _ = _anneal_cost(part, e1, seconds, 1)
    return c0, c1


def main():
    from rl import tree_policy as TP
    from scripts.train_prime_placer import _designs, held_out, iter_training
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default="../PrimeTensorData")
    ap.add_argument("--hours", type=float, default=1.5)
    ap.add_argument("--jobs", type=int, default=6)
    ap.add_argument("--k", type=int, default=8)
    ap.add_argument("--anneal", type=float, default=1.5, help="seconds; negative = a fixed number of steps")
    ap.add_argument("--lr", type=float, default=3e-4)
    ap.add_argument("--val", type=int, default=30)
    ap.add_argument("--val-every", type=int, default=20, help="updates between validations")
    ap.add_argument("--seed", type=int, default=3)
    ap.add_argument("--placer", default="checkpoints/prime_placer.pt")
    ap.add_argument("--resume", default="checkpoints/tree_policy_best_8559.pt")
    ap.add_argument("--out", default="checkpoints/tree_policy_anneal.pt")
    a = ap.parse_args()
    torch.set_num_threads(1)
    net = TP.load(a.resume)
    net.train()
    opt = torch.optim.Adam(net.parameters(), lr=a.lr)
    files = sorted(glob.glob(os.path.join(a.data, "config_*", "primedata_*.pth")))
    hold = sorted(held_out(files))
    val = []
    for f in hold:
        for k, d in enumerate(_designs(f)):
            if k >= max(1, a.val // len(hold)):
                break
            val.append(d)
    data = iter_training([f for f in files if f not in set(hold)], a.seed, buffer_files=3)
    pool = get_context("forkserver").Pool(a.jobs, initializer=_init, initargs=(a.placer,))

    def validate():
        st = {n: p.detach().clone() for n, p in net.state_dict().items()}
        res = [r for r in pool.map(_val_job, [(st, d, a.anneal) for d in val]) if r is not None]
        r = np.array(res)
        return float(np.mean(r[:, 1] / r[:, 0])), float(r[:, 0].mean()), int((r[:, 1] < r[:, 0] - 1e-9).sum()), len(r)

    v, v0, wins, nv = validate()
    best = v
    print(f"start: held-out mean cost ratio policy/seed tree {v:.4f} (seed tree {v0:.4f}; policy better on {wins}/{nv})", flush=True)
    t0, end, upd, seed = time.time(), time.time() + a.hours * 3600, 0, a.seed * 1000
    while time.time() < end:
        st = {n: p.detach().clone() for n, p in net.state_dict().items()}
        jobs = []
        for _ in range(a.jobs):
            seed += 1
            jobs.append((st, next(data), a.k, a.anneal, seed))
        res = [r for r in pool.map(_job, jobs) if r is not None]
        if not res:
            continue
        opt.zero_grad()
        for n, p in net.named_parameters():
            gs = [g[n] for g, _ in res if n in g]
            if gs:
                p.grad = torch.stack(gs).mean(0)
        torch.nn.utils.clip_grad_norm_(net.parameters(), 1.0)
        opt.step()
        upd += 1
        s = np.array([x for _, x in res])
        print(f"update {upd}, {(time.time() - t0) / 60:.0f} min: annealed cost mean {s[:, 0].mean():.3f}, "
              f"best of {a.k} {s[:, 1].mean():.3f}, valid {s[:, 2].mean():.2f}", flush=True)
        if upd % a.val_every == 0:
            v, v0, wins, nv = validate()
            tag = ""
            if v < best:
                best = v
                TP.save(net, a.out, updates=upd, val=v)
                tag = " (saved)"
            print(f"  held-out: ratio policy/seed {v:.4f} (seed tree {v0:.4f}; policy better on {wins}/{nv}){tag}", flush=True)
    pool.close()
    print(f"done: {upd} updates, best held-out {best:.4f}", flush=True)


if __name__ == "__main__":
    main()

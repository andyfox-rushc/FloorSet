"""Train the slicing-tree builder (rl/tree_policy.py) with REINFORCE on the
Prime training set (../PrimeTensorData): K sampled trees per design, reward
= -Partitioner.cost of each, advantage against the K-sample mean (scaled by
its spread), so every design counts the same whatever its cost level.

  python scripts/train_tree_policy.py --hours 6 --out checkpoints/tree_policy.pt

Validation: 40 held-out designs (same held-out files as the centroid
predictor); the checkpoint with the lowest mean greedy-tree cost is kept.
"""
import argparse
import glob
import math
import os
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
import numpy as np
import torch

from rl import partition as PT
from rl import tree_policy as TP
from rl.prime_placer import load as load_placer, predict
from scripts.train_prime_placer import _designs, held_out, iter_training


def make_case(area, b2b, p2b, pins, cons, polys, seednet):
    """(Partitioner, Context, first-tree cost) for one training design."""
    n = len(area)
    tp = []
    for i in range(n):
        v = [(float(x), float(y)) for x, y in polys[i].tolist() if not (x == -1 and y == -1)]
        if cons[i, 1]:
            tp.append(v)
        elif cons[i, 0]:
            x0, y0 = min(p[0] for p in v), min(p[1] for p in v)
            tp.append([(x - x0, y - y0) for x, y in v])
        else:
            tp.append(None)
    args = (n, area.float(), b2b.float(), p2b.float(), pins.float(), cons.float(), tp)
    P = PT.Problem(*args)
    seeds = np.clip(predict(seednet, *args), 0, [P.W, P.H])
    part = PT.Partitioner(P, seeds)
    if len(part.operands) < 3:
        return None
    r = part.region
    e0 = PT.initial_expression(part.operands, part.seeds, P.area,
                               (r[:, 0].min(), r[:, 1].min(), r[:, 2].max(), r[:, 3].max()), P.groups)
    c0 = part.cost(e0)              # also fixes part.ref, the HPWL scale of every later cost
    return part, TP.Context(part), c0[3]


def greedy_cost(net, case):
    part, ctx, _ = case
    with torch.no_grad():
        e, _, _ = TP.rollout(net, part, ctx, greedy=True)
    c = part.cost(e)
    return c[3], c[2] == 0


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default="../PrimeTensorData")
    ap.add_argument("--hours", type=float, default=6.0)
    ap.add_argument("--k", type=int, default=8, help="sampled trees per design")
    ap.add_argument("--batch", type=int, default=4, help="designs per update")
    ap.add_argument("--lr", type=float, default=3e-4)
    ap.add_argument("--val", type=int, default=40)
    ap.add_argument("--val-every", type=int, default=500)
    ap.add_argument("--seed", type=int, default=1)
    ap.add_argument("--placer", default="checkpoints/prime_placer.pt")
    ap.add_argument("--resume")
    ap.add_argument("--out", default="checkpoints/tree_policy.pt")
    a = ap.parse_args()
    torch.set_num_threads(1)
    torch.manual_seed(a.seed)
    seednet = load_placer(a.placer)
    files = sorted(glob.glob(os.path.join(a.data, "config_*", "primedata_*.pth")))
    hold = set(held_out(files))
    train = [f for f in files if f not in hold]
    val = []
    for f in sorted(hold):
        for k, d in enumerate(_designs(f)):
            if k >= max(1, a.val // len(hold)):
                break
            c = make_case(*d, seednet)
            if c is not None:
                val.append(c)
    net = TP.load(a.resume) if a.resume else TP.TreePolicy()
    net.train()
    opt = torch.optim.Adam(net.parameters(), lr=a.lr)

    def validate():
        g = [greedy_cost(net, c) for c in val]
        return float(np.mean([x for x, _ in g])), float(np.mean([v for _, v in g]))

    first = float(np.mean([c[2] for c in val]))
    best, vv = validate()
    print(f"{len(train)} training files, {len(val)} validation designs; val first tree {first:.3f}, "
          f"untrained greedy {best:.3f} (valid {vv:.2f})", flush=True)
    t0, end = time.time(), time.time() + a.hours * 3600
    data = iter_training(train, a.seed)
    seen, log = 0, []
    gen = torch.Generator().manual_seed(a.seed)
    opt.zero_grad()
    while time.time() < end:
        try:
            case = make_case(*next(data), seednet)
        except Exception as e:          # odd designs: skip, don't stop the run
            print(f"  skipped a design: {e!r}", flush=True)
            continue
        if case is None:
            continue
        part, ctx, c0 = case
        costs, logps, valid = [], [], []
        for _ in range(a.k):
            e, lp, _ = TP.rollout(net, part, ctx, rng=gen)
            c = part.cost(e)
            costs.append(c[3])
            valid.append(c[2] == 0)
            logps.append(lp)
        cs = torch.tensor(costs, dtype=torch.float32)
        adv = -(cs - cs.mean()) / (cs.std() + 1e-3)
        loss = -(adv * torch.stack(logps)).mean() / a.batch
        loss.backward()
        seen += 1
        log.append((c0, float(cs.mean()), float(cs.min()), float(np.mean(valid))))
        if seen % a.batch == 0:
            torch.nn.utils.clip_grad_norm_(net.parameters(), 1.0)
            opt.step()
            opt.zero_grad()
        if seen % 100 == 0:
            m = np.array(log)
            print(f"{seen} designs, {(time.time() - t0) / 60:.0f} min: first tree {m[:, 0].mean():.3f}, "
                  f"sampled mean {m[:, 1].mean():.3f}, best of {a.k} {m[:, 2].mean():.3f}, "
                  f"valid samples {m[:, 3].mean():.2f}, sigma {float(net.log_sigma.exp()):.3f}", flush=True)
            log = []
        if seen % a.val_every == 0:
            net.eval()
            v, vv = validate()
            net.train()
            tag = ""
            TP.save(net, a.out.replace(".pt", "_latest.pt"), seen=seen, val=v)
            if v < best:
                best = v
                TP.save(net, a.out, seen=seen, val=v)
                tag = " (saved)"
            print(f"  val greedy {v:.3f} valid {vv:.2f}{tag}", flush=True)
    print(f"done: {seen} designs, best val greedy {best:.3f}", flush=True)


if __name__ == "__main__":
    main()

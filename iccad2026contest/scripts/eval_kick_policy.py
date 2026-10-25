"""Learned vs random kicks for the refiner's iterated local search, on held-out
training designs (the files train_prime_placer.py holds out; never the 100 test
designs). Same start layout (short partition solve), same time, same seeds;
compares the refiner's contest-proxy score at the end (lower is better).

  python scripts/eval_kick_policy.py --designs 24 --seconds 20 --policy checkpoints/kick_policy.pt
"""
import argparse
import glob
import os
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
import numpy as np
import torch

from rl import kick_policy as KP
from rl import partition as PT
from rl.prime_placer import load as load_placer
from rl.refine import Refiner
from scripts.train_prime_placer import _designs, held_out
from scripts.train_tree_policy import make_case


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default="../PrimeTensorData")
    ap.add_argument("--designs", type=int, default=24)
    ap.add_argument("--seconds", type=float, default=20.0)
    ap.add_argument("--solve", type=float, default=8.0)
    ap.add_argument("--seeds", type=int, default=2)
    ap.add_argument("--per-file", type=int, default=3)
    ap.add_argument("--policy", default="checkpoints/kick_policy.pt")
    ap.add_argument("--placer", default="checkpoints/prime_placer.pt")
    a = ap.parse_args()
    torch.set_num_threads(1)
    net = KP.load(a.policy)
    seednet = load_placer(a.placer)
    files = sorted(held_out(sorted(glob.glob(os.path.join(a.data, "config_*", "primedata_*.pth")))))
    rows = []
    picks = []
    for f in files:
        for k, d in enumerate(_designs(f)):         # first per_file designs of each held-out file
            if k >= a.per_file:
                break
            picks.append(d)
    for d in picks:
        if len(rows) >= a.designs:
            break
        case = make_case(*d, seednet)
        if case is None:
            continue
        part = case[0]
        lay = PT.solve(part.P, part.seeds, a.solve, seed=0)
        if lay is None:
            continue
        rnd, lrn = [], []
        for s in range(a.seeds):
            R = Refiner(part.P, lay)
            rnd.append(R.run(time_budget=a.seconds, seed=s)["score"])
            R = Refiner(part.P, lay)
            lrn.append(KP.run_learned(R, net, a.seconds, seed=s)["score"])
        rows.append((np.mean(rnd), np.mean(lrn)))
        print(f"  design {len(rows):2d} n={part.P.n:3d}: random kicks {np.mean(rnd):.4f}, learned kicks {np.mean(lrn):.4f}",
              flush=True)
    r = np.array(rows)
    print(f"{len(r)} held-out designs: random {r[:, 0].mean():.4f}, learned {r[:, 1].mean():.4f}; learned better on "
          f"{int((r[:, 1] < r[:, 0] - 1e-9).sum())}, worse on {int((r[:, 1] > r[:, 0] + 1e-9).sum())}", flush=True)


if __name__ == "__main__":
    main()

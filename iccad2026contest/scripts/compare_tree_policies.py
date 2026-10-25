"""Two tree-builder checkpoints head to head on held-out training designs: each
policy's greedy tree annealed for --anneal seconds, same seeds; paired comparison.

  python scripts/compare_tree_policies.py old.pt new.pt --designs 30 --seeds 3 --anneal 3
"""
import argparse
import glob
import os
import sys
from multiprocessing import get_context
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
import numpy as np


def _job(args):
    import torch
    from rl import tree_policy as TP
    from scripts.train_tree_policy import make_case
    from scripts.train_tree_policy_anneal import _W, _anneal_cost
    paths, design, seconds, seeds = args
    case = make_case(*design, _W["seednet"])
    if case is None:
        return None
    part, ctx, _ = case
    out = []
    for p in paths:
        net = TP.load(p)
        with torch.no_grad():
            e, _, _ = TP.rollout(net, part, ctx, greedy=True)
        cs = []
        for s in range(seeds):
            part.memo.clear()
            cs.append(_anneal_cost(part, e, seconds, s)[0])
        out.append(float(np.mean(cs)))
    return out


def main():
    from scripts.train_prime_placer import _designs, held_out
    from scripts.train_tree_policy_anneal import _init
    ap = argparse.ArgumentParser()
    ap.add_argument("old")
    ap.add_argument("new")
    ap.add_argument("--data", default="../PrimeTensorData")
    ap.add_argument("--designs", type=int, default=30)
    ap.add_argument("--seeds", type=int, default=3)
    ap.add_argument("--anneal", type=float, default=3.0)
    ap.add_argument("--jobs", type=int, default=6)
    ap.add_argument("--skip", type=int, default=3, help="per held-out file, skip the designs used for validation")
    a = ap.parse_args()
    files = sorted(held_out(sorted(glob.glob(os.path.join(a.data, "config_*", "primedata_*.pth")))))
    per = max(1, a.designs // len(files))
    ds = []
    for f in files:
        for k, d in enumerate(_designs(f)):
            if k < a.skip:
                continue
            if k >= a.skip + per:
                break
            ds.append(d)
    with get_context("forkserver").Pool(a.jobs, initializer=_init, initargs=("checkpoints/prime_placer.pt",)) as pool:
        res = [r for r in pool.map(_job, [((a.old, a.new), d, a.anneal, a.seeds) for d in ds]) if r is not None]
    r = np.array(res)
    print(f"{len(r)} held-out designs (not the validation ones), {a.seeds} seeds x {a.anneal} s anneal: "
          f"old {r[:, 0].mean():.4f}, new {r[:, 1].mean():.4f}; new better on {int((r[:, 1] < r[:, 0] - 1e-9).sum())}, "
          f"worse on {int((r[:, 1] > r[:, 0] + 1e-9).sum())}; median ratio new/old {np.median(r[:, 1] / r[:, 0]):.3f}")


if __name__ == "__main__":
    main()

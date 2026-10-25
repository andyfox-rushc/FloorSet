"""Log packer moves on training designs for the move scorer (rl/move_policy.py).

Each design: a learned-placer start layout, then the packer for --secs x
(n / 77)^2 seconds with move logging on (and the scorer filtering when
FLOORSET_MOVE_POLICY is set, for the next RL round). One .npz per design:
f (features), d (cost change / start temperature), step.

  FLOORSET_MOVE_POLICY=checkpoints/move_policy.npz python scripts/collect_moves.py --designs 400 --out movelogs/r1
"""
import argparse
import os
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))
sys.path.insert(0, str(Path(__file__).parent))
import numpy as np


def _job(args):
    k, inst, secs, out = args
    from rl import packer as K
    from rl.analytic import learned_candidate
    start = learned_candidate(inst, None, index=2)
    if start is None:
        return k, 0
    K.MOVE_LOGGING = True
    K.MOVE_LOG.clear()
    K.compact_pack(inst, start, time_budget=K.pack_budget(inst.block_count, secs), seed=k)
    if not K.MOVE_LOG:
        return k, 0
    f = np.concatenate([e[0] for e in K.MOVE_LOG])
    d = np.concatenate([e[1] for e in K.MOVE_LOG])
    step = np.concatenate([np.full(len(e[1]), i) for i, e in enumerate(K.MOVE_LOG)])   # logged step index
    np.savez_compressed(os.path.join(out, f"d{k:05d}.npz"), f=f, d=d, step=step, n=inst.block_count)
    return k, len(d)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--designs", type=int, default=400)
    ap.add_argument("--secs", type=float, default=10.0, help="packer seconds at 77 blocks (x (n/77)^2)")
    ap.add_argument("--min-blocks", type=int, default=40)
    ap.add_argument("--procs", type=int, default=6)
    ap.add_argument("--seed", type=int, default=5)
    ap.add_argument("--data-path", default="../")
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    os.makedirs(a.out, exist_ok=True)
    import multiprocessing as mp
    from rl.packer import init_worker
    from train_placer import iter_training
    data = iter_training(a.data_path, a.seed)
    jobs = []
    while len(jobs) < a.designs:
        inst, _ = next(data)
        if inst.block_count >= a.min_blocks:
            jobs.append((len(jobs), inst, a.secs, a.out))
    with mp.get_context("forkserver").Pool(a.procs, initializer=init_worker) as pool:
        tot = 0
        for i, (k, m) in enumerate(pool.imap_unordered(_job, jobs)):
            tot += m
            if (i + 1) % 20 == 0:
                print(f"{i + 1}/{len(jobs)} designs, {tot} moves", flush=True)
    print(f"done: {tot} moves", flush=True)


if __name__ == "__main__":
    main()

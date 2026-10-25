"""A pool of valid partition layouts on Prime training designs (never the 100
test designs, never the held-out files), for training the kick policy on many
ILS episodes without a fresh solve each time.

  python scripts/build_layout_pool.py --n 150 --solve 8 --out checkpoints/layout_pool.pkl
"""
import argparse
import glob
import os
import pickle
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
import torch

from rl import partition as PT
from rl.prime_placer import load as load_placer
from scripts.train_prime_placer import _designs, held_out, iter_training
from scripts.train_tree_policy import make_case


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default="../PrimeTensorData")
    ap.add_argument("--n", type=int, default=150)
    ap.add_argument("--solve", type=float, default=8.0)
    ap.add_argument("--seed", type=int, default=5)
    ap.add_argument("--placer", default="checkpoints/prime_placer.pt")
    ap.add_argument("--out", default="checkpoints/layout_pool.pkl")
    a = ap.parse_args()
    torch.set_num_threads(1)
    seednet = load_placer(a.placer)
    files = sorted(glob.glob(os.path.join(a.data, "config_*", "primedata_*.pth")))
    hold = set(held_out(files))
    data = iter_training([f for f in files if f not in hold], a.seed, buffer_files=2)
    pool, t0 = [], time.time()
    while len(pool) < a.n:
        d = next(data)
        try:
            case = make_case(*d, seednet)
            if case is None:
                continue
            part = case[0]
            lay = PT.solve(part.P, part.seeds, a.solve, seed=0)
        except Exception as e:
            print(f"  skipped: {e!r}", flush=True)
            continue
        if lay is None:
            continue
        pool.append((d, lay))
        if len(pool) % 10 == 0:
            print(f"{len(pool)} layouts, {(time.time() - t0) / 60:.0f} min", flush=True)
            with open(a.out, "wb") as f:
                pickle.dump(pool, f)
    with open(a.out, "wb") as f:
        pickle.dump(pool, f)
    print(f"done: {len(pool)} layouts", flush=True)


if __name__ == "__main__":
    main()

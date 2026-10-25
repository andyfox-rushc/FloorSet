"""Train the refiner's kick policy (rl/kick_policy.py) on Prime training designs.

Per design: a short partition solve (one restart) gives a valid layout, the
greedy refiner takes it to a local optimum, then for STEPS iterated-local-search
steps: K policy kicks from the same state, each followed by greedy descent;
reward = relative score drop (x100); advantage against the K-kick mean;
the best kick's state is kept when it improves (ILS acceptance).

  python scripts/train_kick_policy.py --hours 6 --out checkpoints/kick_policy.pt
"""
import argparse
import glob
import os
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
import numpy as np
import torch

from rl import kick_policy as KP
from rl import partition as PT
from rl.prime_placer import load as load_placer, predict
from rl.refine import Refiner
from scripts.train_prime_placer import _designs, held_out, iter_training
from scripts.train_tree_policy import make_case


def layout_for(case, seconds):
    part, ctx, _ = case
    P = part.P
    res = PT.solve(P, part.seeds, seconds, seed=0)
    return P, res


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default="../PrimeTensorData")
    ap.add_argument("--hours", type=float, default=6.0)
    ap.add_argument("--solve", type=float, default=8.0, help="partition seconds per training design")
    ap.add_argument("--steps", type=int, default=12, help="ILS steps per design")
    ap.add_argument("--k", type=int, default=4, help="kicks tried per step")
    ap.add_argument("--moves", type=int, default=3, help="re-cuts per kick")
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--seed", type=int, default=2)
    ap.add_argument("--placer", default="checkpoints/prime_placer.pt")
    ap.add_argument("--out", default="checkpoints/kick_policy.pt")
    ap.add_argument("--pool", help="layout pool (scripts/build_layout_pool.py) instead of fresh solves")
    ap.add_argument("--resume")
    a = ap.parse_args()
    torch.set_num_threads(1)
    torch.manual_seed(a.seed)
    seednet = load_placer(a.placer)
    files = sorted(glob.glob(os.path.join(a.data, "config_*", "primedata_*.pth")))
    hold = set(held_out(files))
    train = [f for f in files if f not in hold]
    net = KP.load(a.resume) if a.resume else KP.KickPolicy()
    net.train()
    opt = torch.optim.Adam(net.parameters(), lr=a.lr)
    gen = torch.Generator().manual_seed(a.seed)
    data = iter_training(train, a.seed)
    t0, end = time.time(), time.time() + a.hours * 3600
    seen, log = 0, []
    pool = None
    if a.pool:
        import pickle
        with open(a.pool, "rb") as f:
            pool = pickle.load(f)
        print(f"layout pool: {len(pool)} designs", flush=True)
    local = {}                                    # pool index -> greedy local optimum (rects, c, hp, viol, hp_ref)
    rng = np.random.default_rng(a.seed)
    while time.time() < end:
        try:
            if pool is not None:
                k_ = int(rng.integers(len(pool)))
                d, lay = pool[k_]
                case = make_case(*d, seednet)
                if case is None:
                    continue
                P = case[0].P
            else:
                k_ = None
                case = make_case(*next(data), seednet)
                if case is None:
                    continue
                P, lay = layout_for(case, a.solve)
        except Exception as e:
            print(f"  skipped a design: {e!r}", flush=True)
            continue
        if lay is None:
            continue
        R = Refiner(P, lay)
        if k_ is not None and k_ in local:
            rects, c, R.hp, R.viol, R.hp_ref = local[k_]
            R.rects, R.c = dict(rects), c.copy()
        else:
            R.hp = R.hp_ref = max(P.hpwl(R.c[:, 0], R.c[:, 1]), 1e-9)
            R.viol = R.soft_viol(range(P.n))
            R._greedy(time.time() + 30, 20)
        s_start = R.score(R.hp, R.viol)
        for _ in range(a.steps):
            state = (dict(R.rects), R.c.copy(), R.hp, R.viol)
            s0 = R.score(R.hp, R.viol)
            rewards, logps, ends = [], [], []
            for _ in range(a.k):
                R.rects, R.c, R.hp, R.viol = dict(state[0]), state[1].copy(), state[2], state[3]
                lp = KP.policy_kick(R, net, a.moves, gen)
                R._greedy(time.time() + 10, 20)
                s1 = R.score(R.hp, R.viol)
                rewards.append(100 * (s0 - s1) / s0)
                logps.append(lp)
                ends.append((s1, dict(R.rects), R.c.copy(), R.hp, R.viol))
            ok = [i for i, lp in enumerate(logps) if lp is not None]
            if len(ok) >= 2:
                r = torch.tensor([rewards[i] for i in ok])
                adv = (r - r.mean()) / (r.std() + 1e-6)
                loss = -(adv * torch.stack([logps[i] for i in ok])).mean()
                opt.zero_grad()
                loss.backward()
                torch.nn.utils.clip_grad_norm_(net.parameters(), 1.0)
                opt.step()
            j = int(np.argmin([e[0] for e in ends]))
            if ends[j][0] < s0 - 1e-12:                   # ILS acceptance
                _, R.rects, R.c, R.hp, R.viol = ends[j]
            else:
                R.rects, R.c, R.hp, R.viol = dict(state[0]), state[1].copy(), state[2], state[3]
            log.append((float(np.mean(rewards)), float(np.max(rewards)), float(np.mean([x > 0 for x in rewards]))))
        if k_ is not None:                          # keep the improved state for the next visit
            local[k_] = (dict(R.rects), R.c.copy(), R.hp, R.viol, R.hp_ref)
        seen += 1
        log_d = (s_start - R.score(R.hp, R.viol)) / s_start * 100
        if seen % 10 == 0:
            m = np.array(log)
            print(f"{seen} designs, {(time.time() - t0) / 60:.0f} min: kick reward mean {m[:, 0].mean():.3f}, "
                  f"best-of-{a.k} {m[:, 1].mean():.3f}, improving kicks {m[:, 2].mean():.2f}, "
                  f"last design ILS gain {log_d:.2f}%", flush=True)
            log = []
            KP.save(net, a.out, seen=seen)
    KP.save(net, a.out, seen=seen)
    print(f"done: {seen} designs", flush=True)


if __name__ == "__main__":
    main()

"""Packer move A/B (10-07): pack the same start layouts with different env
settings (same seeds and budgets, arms interleaved so they share the load).

  python scripts/move_ab.py --start official_eval_1006b.json --budget 15 \
      --arm base: --arm ideal:FLOORSET_PACK_IDEAL=0.3 --out move_ab.json
"""
import argparse
import json
import math
import sys
from multiprocessing import get_context
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))
sys.path.insert(0, str(Path(__file__).parent))
import polish_bench as PB


def _one(job):
    import importlib
    import os
    arm, envs, tid, n, pos, budget, seed = job
    import torch
    torch.set_num_threads(1)
    for k, v in envs.items():
        os.environ[k] = v
    from rl import packer as K
    importlib.reload(K)
    from rl.data import from_validation_sample
    inst = from_validation_sample(PB.EV, tid)
    out = K.compact_pack(inst, [tuple(p) for p in pos], time_budget=K.pack_budget(n, budget), seed=seed)
    s = PB._score(inst, out)
    return arm, tid, n, s.cost, s.hpwl_gap, s.area_gap, s.violations_relative


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--start", required=True)
    ap.add_argument("--budget", type=float, default=15)
    ap.add_argument("--min-n", type=int, default=100)
    ap.add_argument("--seeds", type=int, default=1)
    ap.add_argument("--seed-offset", type=int, default=0)
    ap.add_argument("--arm", action="append", required=True, help="name:ENV=VAL,ENV=VAL")
    ap.add_argument("--workers", type=int, default=4)
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    arms = []
    for spec in a.arm:
        name, _, rest = spec.partition(":")
        arms.append((name, dict(kv.split("=", 1) for kv in rest.split(",") if kv)))
    T = [r for r in json.load(open(a.start))["test_results"] if r["block_count"] >= a.min_n]
    jobs = [(name, envs, r["test_id"], r["block_count"], r["positions"], a.budget, seed)
            for r in T for seed in range(a.seed_offset, a.seed_offset + a.seeds) for name, envs in arms]
    with get_context("forkserver").Pool(a.workers, initializer=PB._init, maxtasksperchild=1) as pool:
        res = pool.map(_one, jobs, chunksize=1)
    json.dump(res, open(a.out, "w"))
    for name, _ in arms:
        rs = [r for r in res if r[0] == name]
        W = sum(math.exp(r[2] / 12) for r in rs)
        f = lambda k: sum(math.exp(r[2] / 12) * r[k] for r in rs) / W
        m = lambda k: sum(r[k] for r in rs) / len(rs)
        print(f"{name}: weighted {f(3):.4f} mean {m(3):.4f} hpwl {m(4):.3f} area {m(5):.3f} vrel {m(6):.3f} ({len(rs)} runs)")


if __name__ == "__main__":
    main()

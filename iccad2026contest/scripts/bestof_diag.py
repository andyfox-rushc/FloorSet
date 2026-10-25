"""Best-of diagnosis (10-05): per design, which candidate wins and where its
soft violations (edge pins, clusters, MIB shapes) appear along the stages
(placer raw -> legalized -> slicing -> re-pack). Mirrors rl/analytic.py
best_of with the default candidates.

  python scripts/bestof_diag.py --workers 6 --out bestof_diag.json
"""
import argparse
import json
import sys
import time
from multiprocessing import get_context
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))
sys.path.insert(0, str(Path(__file__).parent))
import numpy as np
import torch

from polish_bench import _init, _score  # noqa: E402
import polish_bench as PB  # noqa: E402

CANDS = ["base", "scratch0.85:it250:po1.0", "scratch0.95:it250:po1.0"]


def breakdown(inst, pos):
    """Soft violations by kind, plus per-violation details."""
    from rl import repack as R
    P = R._problem(inst, pos)
    p = P["pos"]
    x, y, w, h = p[:, 0], p[:, 1], p[:, 2], p[:, 3]
    C = R._Cost(P, 1.0, 1.0)
    off = C.off_edge(x, y, w, h)
    mib = sum(len({(round(w[i], 4), round(h[i], 4)) for i in g}) - 1 for g in C.mibs)
    grp = C.violations(x, y, w, h) - len(off) - mib
    kinds = []
    for b in off:
        code = int(P["bnd"][b])
        kinds.append(dict(block=b, corner=bin(code).count("1") == 2, pre=bool(P["preplaced"][b]),
                          fixed=bool(P["fixed"][b]), cluster=int(P["cluster"][b] > 0), mib=int(P["mib"][b] > 0)))
    # overlap area (placer raw output may overlap)
    x1, y1 = x + w, y + h
    ov = (np.minimum(x1[:, None], x1[None, :]) - np.maximum(x[:, None], x[None, :])).clip(0) * \
         (np.minimum(y1[:, None], y1[None, :]) - np.maximum(y[:, None], y[None, :])).clip(0)
    np.fill_diagonal(ov, 0)
    return dict(bnd=len(off), grp=int(grp), mib=int(mib), n_soft=int(P["n_soft"]),
                n_pinned=int((P["bnd"] != 0).sum()), off=kinds,
                overlap=float(ov.sum() / 2 / max(P["area"].sum(), 1e-9)),
                overhang=bool(min(x.min(), y.min()) < -1e-6))


def _one(args):
    tid, ckpt = args
    from rl.analytic import analytic_solve, global_place
    from rl import repack as R
    from rl.checkpoint_meta import utilization_ladder
    from rl.data import from_validation_sample
    from rl.finetune import finetune_and_solve
    from rl.networks import ActorCritic, load_state
    from rl.repack import contest_relative_cost, repack_polish
    from rl.slicing import _legalize_grow, slicing_polish
    net = ActorCritic()
    load_state(net, ckpt)
    inst = from_validation_sample(PB.EV, tid)
    rl_pos, _ = finetune_and_solve(inst, net=net, time_budget=2.0, seed=tid, utilizations=utilization_ladder(ckpt))
    out = dict(design=tid, n=inst.block_count, cands={})
    for name in CANDS:
        t = time.time()
        parts = name.split(":")
        opts = {q[:2]: q[2:] for q in parts[1:]}
        it, budget = int(opts.get("it", 500)), float(opts.get("po", 1.0))
        st = {}
        if parts[0] == "base":
            p = rl_pos
            st["policy"] = breakdown(inst, p)
        else:
            gp = global_place(inst, util=float(parts[0][7:]), iters=it, seed=tid)
            st["placer"] = breakdown(inst, gp)
            p = analytic_solve(inst, util=float(parts[0][7:]), iters=it, seed=tid)
            st["legal"] = breakdown(inst, p)
        p = slicing_polish(inst, p, time_budget=budget / 2)
        st["slice"] = breakdown(inst, p)
        p = repack_polish(inst, p, time_budget=budget / 2, seed=tid)
        st["repack"] = breakdown(inst, p)
        rel, ok = contest_relative_cost(inst, p, rl_pos)
        r = _score(inst, p)
        out["cands"][name] = dict(stages=st, rel=rel, feasible=ok, cost=r.cost, hpwl_gap=r.hpwl_gap,
                                  area_gap=r.area_gap, vrel=r.violations_relative, secs=time.time() - t)
    ok = {k: v for k, v in out["cands"].items() if v["feasible"]}
    out["winner"] = min(ok, key=lambda k: ok[k]["rel"]) if ok else None
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--checkpoint", default="checkpoints/policy.pt")
    ap.add_argument("--cases", type=int, default=100)
    ap.add_argument("--workers", type=int, default=6)
    ap.add_argument("--out", default="bestof_diag.json")
    a = ap.parse_args()
    with get_context("forkserver").Pool(a.workers, initializer=_init) as pool:
        res = []
        for r in pool.imap_unordered(_one, [(t, a.checkpoint) for t in range(a.cases)]):
            res.append(r)
            w = r["winner"]
            print(f"design {r['design']:3d} winner {w} cost {r['cands'][w]['cost']:.3f} "
                  f"bnd {r['cands'][w]['stages']['repack']['bnd']} grp {r['cands'][w]['stages']['repack']['grp']}",
                  flush=True)
    res.sort(key=lambda r: r["design"])
    json.dump(res, open(a.out, "w"), indent=1)


if __name__ == "__main__":
    main()

"""Compare post-RL polish methods on identical inputs.

  python scripts/polish_bench.py cache --checkpoint checkpoints/history/policy_iter006000.pt
  python scripts/polish_bench.py run --methods none anneal repack

`cache` runs finetune_and_solve (RL + compaction) once per validation case
and saves the layouts; `run` applies each polish method to the cached
layouts (cases spread over --workers processes) and scores against ground
truth with the contest evaluator.
"""
import argparse
import json
import math
import pickle
import sys
import time
from multiprocessing import get_context
from pathlib import Path

ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(ROOT))
import torch

EV = None


def _init():
    global EV
    torch.set_num_threads(1)
    from iccad2026_evaluate import ContestEvaluator
    EV = ContestEvaluator("../", verbose=False)
    EV._load_dataset()


def _cache_one(args):
    tid, ckpt = args
    from rl.checkpoint_meta import utilization_ladder
    from rl.data import from_validation_sample
    from rl.finetune import finetune_and_solve
    from rl.networks import ActorCritic, load_state
    net = ActorCritic()
    load_state(net, ckpt)
    inst = from_validation_sample(EV, tid)
    pos, _ = finetune_and_solve(inst, net=net, time_budget=20.0, seed=tid,
                                utilizations=utilization_ladder(ckpt))
    return tid, pos


def _score(inst, pos):
    """Contest metrics of `pos` exactly as returned (rl.reward._evaluate
    compacts first, which would undo polish-stage boundary fixes)."""
    from iccad2026_evaluate import evaluate_solution
    from rl.reward import _target_positions_list
    return evaluate_solution({'positions': pos, 'runtime': 1.0}, inst.baseline_metrics, inst.constraints,
                             inst.b2b_connectivity, inst.p2b_connectivity, inst.pins_pos,
                             inst.area_targets, _target_positions_list(inst), median_runtime=1.0)


def _polish(method, inst, pos, budget, seed):
    if method == "none":
        return pos
    if method == "anneal":
        from rl.anneal import anneal_polish
        return anneal_polish(inst, pos, time_budget=budget, seed=seed)
    if method == "repack":
        from rl.repack import repack_polish
        return repack_polish(inst, pos, time_budget=budget, chains=1, seed=seed)
    if method == "slice":
        from rl.slicing import slicing_polish
        return slicing_polish(inst, pos, time_budget=budget)
    if method == "slice+repack":
        from rl.repack import repack_polish
        from rl.slicing import slicing_polish
        p = slicing_polish(inst, pos, time_budget=budget / 2)
        return repack_polish(inst, p, time_budget=budget / 2, chains=1, seed=seed)
    if method == "best":       # rl/analytic.py best_of, candidates from FLOORSET_CANDIDATES
        import os
        from rl.analytic import best_of
        pack = float(os.environ.get("FLOORSET_PACK", "1.5"))
        if pack <= 0:
            return best_of(inst, pos, os.environ["FLOORSET_CANDIDATES"].split(","), seed=seed)
        from rl.packer import compact_pack, pack_budget        # as in my_optimizer.py
        from rl.repack import contest_relative_cost
        ranked = best_of(inst, pos, os.environ["FLOORSET_CANDIDATES"].split(","), seed=seed, return_all=True)
        outs = [compact_pack(inst, q, time_budget=pack_budget(inst.block_count, pack), seed=seed)
                for q in ranked[:int(os.environ.get("FLOORSET_PACK_TOPK", "4"))]]
        return min(outs, key=lambda q: contest_relative_cost(inst, q, pos)[0])
    if method == "repack+anneal":
        from rl.anneal import anneal_polish
        from rl.repack import repack_polish
        p = repack_polish(inst, pos, time_budget=budget / 2, chains=1, seed=seed)
        return anneal_polish(inst, p, time_budget=budget / 2, seed=seed)
    raise ValueError(method)


def _pipeline_one(args):
    """Fine-tune for ft seconds, then polish for pol seconds (method)."""
    tid, ckpt, ft, method, pol = args
    from rl.checkpoint_meta import utilization_ladder
    from rl.data import from_validation_sample
    from rl.finetune import finetune_and_solve
    from rl.networks import ActorCritic, load_state
    net = ActorCritic()
    load_state(net, ckpt)
    inst = from_validation_sample(EV, tid)
    t = time.time()
    pos, _ = finetune_and_solve(inst, net=net, time_budget=ft, seed=tid,
                                utilizations=utilization_ladder(ckpt))
    t_ft = time.time() - t
    if pol > 0:
        pos = _polish(method, inst, pos, pol, tid)
    r = _score(inst, pos)
    return tid, dict(cost=r.cost, hpwl_gap=r.hpwl_gap, area_gap=r.area_gap, vrel=r.violations_relative,
                     feasible=r.is_feasible, secs=time.time() - t, ft_secs=t_ft)


def _run_one(args):
    tid, pos, methods, budget = args
    from rl.data import from_validation_sample
    inst = from_validation_sample(EV, tid)
    out = {}
    for m in methods:
        t = time.time()
        p = _polish(m, inst, pos, budget, tid)
        r = _score(inst, p)
        out[m] = dict(cost=r.cost, hpwl_gap=r.hpwl_gap, area_gap=r.area_gap,
                      vrel=r.violations_relative, feasible=r.is_feasible, secs=time.time() - t,
                      overhang=min(min(q[0] for q in p), min(q[1] for q in p)) < -1e-6)
    return tid, out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("stage", choices=["cache", "run", "pipeline"])
    ap.add_argument("--splits", nargs="+", default=["20:repack:8", "28:none:0"],
                    help="pipeline stage: finetune_secs:method:polish_secs per config")
    ap.add_argument("--checkpoint", default="checkpoints/history/policy_iter006000.pt")
    ap.add_argument("--cache", default="polish_bench_cache.pkl")
    ap.add_argument("--methods", nargs="+", default=["none", "anneal", "repack"])
    ap.add_argument("--budget", type=float, default=8.0)
    ap.add_argument("--cases", type=int, default=100)
    ap.add_argument("--workers", type=int, default=8)
    ap.add_argument("--out", default="polish_bench_results.json")
    a = ap.parse_args()
    ctx = get_context("forkserver")
    with ctx.Pool(a.workers, initializer=_init) as pool:
        if a.stage == "pipeline":
            summary = {}
            for split in a.splits:
                ft, method, pol = split.split(":")
                jobs = [(t, a.checkpoint, float(ft), method, float(pol)) for t in range(a.cases)]
                res = dict(pool.map(_pipeline_one, jobs))
                rows = list(res.values())
                avg = lambda k: sum(r[k] for r in rows) / len(rows)
                summary[split] = dict(per_case=res, **{k: avg(k) for k in ("cost", "hpwl_gap", "area_gap", "vrel", "secs")})
                print(f"{split:18s} cost {avg('cost'):.3f} hpwl_gap {avg('hpwl_gap'):.3f} area_gap {avg('area_gap'):.3f} "
                      f"vrel {avg('vrel'):.3f} feasible {sum(r['feasible'] for r in rows)}/{len(rows)} "
                      f"secs {avg('secs'):.1f}", flush=True)
                # official total: weights exp(n / 12) by block count
                N = {r["design"]: r["n"] for r in json.load(open(ROOT / "bestof_diag.json"))}
                wt = {t: math.exp(N[t] / 12) for t in res}
                summary[split]["weighted"] = sum(res[t]["cost"] * wt[t] for t in res) / sum(wt.values())
                print(f"{split:18s} weighted score {summary[split]['weighted']:.3f}", flush=True)
                json.dump(summary, open(a.out, "w"), indent=1)
            return
        if a.stage == "cache":
            res = dict(pool.map(_cache_one, [(t, a.checkpoint) for t in range(a.cases)]))
            pickle.dump(dict(checkpoint=a.checkpoint, positions=res), open(a.cache, "wb"))
            print(f"cached {len(res)} layouts from {a.checkpoint}")
            return
        cache = pickle.load(open(a.cache, "rb"))
        jobs = [(t, p, a.methods, a.budget) for t, p in sorted(cache["positions"].items())][:a.cases]
        res = dict(pool.map(_run_one, jobs))
    json.dump(res, open(a.out, "w"), indent=1)
    print(f"{len(res)} cases, checkpoint {cache['checkpoint']}, budget {a.budget}s")
    print(f"{'method':14s} {'cost':>7s} {'hpwl_gap':>9s} {'area_gap':>9s} {'vrel':>7s} {'feasible':>9s} {'secs':>6s}")
    for m in a.methods:
        rows = [r[m] for r in res.values()]
        avg = lambda k: sum(r[k] for r in rows) / len(rows)
        print(f"{m:14s} {avg('cost'):7.3f} {avg('hpwl_gap'):9.3f} {avg('area_gap'):9.3f} {avg('vrel'):7.3f} "
              f"{sum(r['feasible'] for r in rows):6d}/{len(rows)} {avg('secs'):6.1f}")


if __name__ == "__main__":
    main()

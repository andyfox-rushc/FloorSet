"""Which candidates win after packing? For each design: build every best-of
candidate, pack all of them (not only the solver's top k), squeeze + relax,
and print the real contest cost of each before and after packing, marking
the ones the solver's top-k cut would have kept.

  ../venv/bin/python scripts/cand_diag.py --ids 90 70 61 --out diag.json
"""
import argparse
import json
import os
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))
sys.path.insert(0, str(Path(__file__).parent))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ids", type=int, nargs="+", required=True)
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    import polish_bench as PB
    PB._init()
    import my_optimizer as M
    from rl.analytic import _candidate_job
    from rl.data import from_validation_sample
    from rl.finetune import finetune_and_solve
    from rl.packer import _pack_job, pack_budget, squeeze, relax_hpwl
    opt = M.MyOptimizer()
    res = {}
    for t in a.ids:
        inst = from_validation_sample(PB.EV, t)
        rl_pos, _ = finetune_and_solve(inst, net=opt._load_checkpoint(), time_budget=opt.time_budget,
                                       max_iterations=opt.max_iterations, episodes_per_iter=opt.episodes_per_iter,
                                       ppo_epochs=opt.ppo_epochs, verbose=False,
                                       utilizations=M.utilization_ladder(opt.checkpoint_path))
        names = opt.candidates
        built = M._pmap(_candidate_job, [(inst, rl_pos, nm, 0) for nm in names])
        budget = pack_budget(inst.block_count, opt.pack_time_budget)
        packed = M._pmap(_pack_job, [(inst, p, budget, k) for k, (_, _, p) in enumerate(built)])
        # solver's top-k: cheapest feasible relative cost, then the forced learned ones
        order = sorted([i for i, b in enumerate(built) if b[1]], key=lambda i: built[i][0])
        forced = [i for i in order[1:] if names[i].startswith("learned")]
        kept = (order[:1] + forced + [i for i in order[1:] if i not in forced])[:opt.pack_topk]
        rows = []
        for i, nm in enumerate(names):
            pre = PB._score(inst, built[i][2]).cost
            fin = relax_hpwl(inst, squeeze(inst, packed[i]))
            r = PB._score(inst, fin)
            rows.append(dict(name=nm, pre=pre, post=r.cost, kept=i in kept, hpwl=r.hpwl_gap, area=r.area_gap,
                             vrel=r.violations_relative))
        res[t] = dict(n=inst.block_count, rows=rows)
        best = min(rows, key=lambda r: r["post"])
        bestk = min((r for r in rows if r["kept"]), key=lambda r: r["post"])
        print(f"design {t} n {inst.block_count}: best {best['name']} {best['post']:.3f}, top-k best "
              f"{bestk['name']} {bestk['post']:.3f}", flush=True)
        for r in rows:
            print(f"   {r['name']:24s} pre {r['pre']:.3f} post {r['post']:.3f} {'K' if r['kept'] else ' '}"
                  f" hpwl {r['hpwl']:.3f} area {r['area']:.3f} vrel {r['vrel']:.3f}", flush=True)
        json.dump(res, open(a.out, "w"), indent=1)


if __name__ == "__main__":
    main()

"""RL on the relation placer's arrangement (10-11, scripts/rl_relation.py).

Policy: the relation head's per-block scores (rl/learned_placer.py
rel_sequences: summed probability of coming first in G+ / G-) as
Plackett-Luce logits, so a sequence pair is sampled by sorting
beta * score + Gumbel noise, and its log probability is differentiable.

Reward: the cost of the solver's own path for that arrangement (pack from
the origin, collision repair, shape fitting, short packer run), scored by
the contest evaluator against the design's baseline. The 10-08 attempt
(scripts/rl_placer.py) used the cost before fitting and packing, which
barely tracks the final cost (rank correlation 0.38 vs 0.54) and failed.
"""
import numpy as np
import torch

FAIL = 3.0          # cost of an arrangement the repair can't make legal


def scores(net, instance):
    """Differentiable per-block G+ / G- scores [n], [n] (net already in the mode wanted)."""
    from .learned_placer import features
    n = instance.block_count
    feats, pin_feats, b2b, p2b, _ = features(instance)
    net(feats, pin_feats, b2b, p2b)
    iu, ju = np.triu_indices(n, 1)
    pr = torch.softmax(net.rel_logits(net._h, torch.tensor(iu), torch.tensor(ju)), dim=1)
    bp, bm = pr[:, 0] + pr[:, 3], pr[:, 0] + pr[:, 2]
    iu_t, ju_t = torch.tensor(iu), torch.tensor(ju)
    sp = torch.zeros(n).index_add(0, iu_t, bp).index_add(0, ju_t, 1 - bp)
    sm = torch.zeros(n).index_add(0, iu_t, bm).index_add(0, ju_t, 1 - bm)
    return sp, sm


def pl_sample(u, rng):
    """Plackett-Luce order (first = highest) for logits u (numpy) via Gumbel sorting."""
    return np.argsort(-(u + rng.gumbel(size=len(u))), kind="stable")


def pl_logp(u, order):
    """log P(order | logits u): sum_t u[o_t] - logsumexp(u[o_t:])."""
    v = u[torch.as_tensor(order)]
    tail = torch.logcumsumexp(v.flip(0), 0).flip(0)
    return (v - tail).sum()


def sample_cost(args):
    """Process-pool entry: (instance, (cx, cy, la), (gp, gm), fit iters, pack s, seed)
    -> (contest cost vs the instance's baseline, feasible)."""
    from . import learned_placer as L
    from . import packer as K
    from . import repack as R
    from .sizing import fit_shapes
    from iccad2026_evaluate import evaluate_solution
    from .reward import _target_positions_list
    inst, pred, seqs, iters, pack_s, seed = args
    pos, (gp, gm), over = L.to_layout(inst, None, pred=pred, seqs=seqs)
    if over > 1e-6:
        P = R._problem(inst, pos)
        w, h = P["pos"][:, 2].copy(), P["pos"][:, 3].copy()
        Bt = K._Batch(P, 1.0, 1.0, (0.0, 0.0))
        fx = K._origin_repair(Bt, np.asarray(gp), np.asarray(gm), w, h)
        if fx is None:
            return FAIL, False
        gp, gm = fx
        x, y, o = Bt.pack(gm[None], np.argsort(gp)[None], np.argsort(gm)[None], w[None], h[None])
        if o[0] > 1e-6:
            return FAIL, False
        pos = [(float(x[0, i]), float(y[0, i]), float(w[i]), float(h[i])) for i in range(len(w))]
    pos = fit_shapes(inst, pos, iters=iters, lr=0.02)
    if pack_s > 0:
        pos = K.compact_pack(inst, pos, time_budget=pack_s, seed=seed)
    s = evaluate_solution({"positions": pos, "runtime": 1.0}, inst.baseline_metrics, inst.constraints,
                          inst.b2b_connectivity, inst.p2b_connectivity, inst.pins_pos,
                          inst.area_targets, _target_positions_list(inst), median_runtime=1.0)
    return (float(s.cost), True) if s.is_feasible else (min(float(s.cost), FAIL), False)

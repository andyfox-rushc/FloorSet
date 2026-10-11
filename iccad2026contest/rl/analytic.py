"""
Analytical global placement (roadmap step 1, 10-04): block centres by
gradient descent on smoothed weighted wirelength (b2b + p2b, the contest's
HPWL formula with |d| ~ sqrt(d^2 + eps^2)), plus penalties for pairwise
overlap (weight ramped up over the run so blocks spread gradually), leaving a
target outline (pin-box shape at utilization `util`), edge-pinned blocks off
their outline edge, and cluster members apart. Preplaced blocks stay put.
The result may still overlap; callers legalize it (rl/repack.py _legalize,
then the usual slicing / re-pack polish).

Shapes: fixed / preplaced blocks keep their given (w, h); soft blocks take
their shape from `init_positions` if given, else a square of their area;
an MIB group shares its leader's shape.
"""
import math
import os
import time
from typing import List, Optional, Tuple

import numpy as np
import torch

Position = Tuple[float, float, float, float]
LEGALIZER = os.environ.get("FLOORSET_LEGALIZER", "graph")   # "tetris": rl/legalize.py
OUTLINE_WALLS = os.environ.get("FLOORSET_OUTLINE_WALLS", "0") != "0"   # 10-05: 1.362 on vs 1.354 off (2 runs each)
# Upstream origin rule (10-06, cost 0.04 / 0.02 weighted at packer 1.5 / 15): the
# placer and best-of keep layouts at x, y >= 0; FLOORSET_ORIGIN works in the packer.
ORIGIN = os.environ.get("FLOORSET_ORIGIN_UPSTREAM", "0") != "0"


def _shapes(instance, init_positions):
    n = instance.block_count
    area = instance.area_targets[:n].clamp(min=1e-9).double()
    tp = instance.target_positions
    c = instance.constraints
    fixed = (c[:n, 0] != 0) if c.shape[1] > 0 else torch.zeros(n, dtype=torch.bool)
    pre = (c[:n, 1] != 0) if c.shape[1] > 1 else torch.zeros(n, dtype=torch.bool)
    if init_positions is not None:
        ip = torch.tensor(init_positions, dtype=torch.float64)
        w, h = ip[:, 2].clone(), ip[:, 3].clone()
    else:
        w, h = area.sqrt(), area.sqrt()
    known = fixed | pre
    w = torch.where(known, tp[:n, 2].double(), w)
    h = torch.where(known, tp[:n, 3].double(), h)
    mib = c[:n, 2].long() if c.shape[1] > 2 else torch.zeros(n, dtype=torch.long)
    for g in mib.unique():
        if g == 0:
            continue
        m = (mib == g).nonzero().flatten()
        lead = m[known[m]][0] if known[m].any() else m[area[m].argmax()]
        for i in m:
            if not known[i]:
                w[i], h[i] = w[lead], h[lead]
    return w, h, pre


def global_place(instance, init_positions: Optional[List[Position]] = None, util: float = 0.85,
                 iters: int = 500, time_budget: Optional[float] = None, seed: int = 0,
                 ov_start: float = 0.02, ov_end: float = 200.0) -> List[Position]:
    t0 = time.time()
    g = torch.Generator().manual_seed(seed)
    n = instance.block_count
    w, h, pre = _shapes(instance, init_positions)
    tp = instance.target_positions[:n].double()
    c = instance.constraints
    bnd = c[:n, 4].long() if c.shape[1] > 4 else torch.zeros(n, dtype=torch.long)
    clu = c[:n, 3].long() if c.shape[1] > 3 else torch.zeros(n, dtype=torch.long)
    tot = float((w * h).sum())
    scale = math.sqrt(tot / max(n, 1))

    # Target outline: pin-box shape (pins start at the origin), area tot / util,
    # grown to cover preplaced blocks.
    pins = instance.pins_pos.double() if instance.pins_pos is not None else torch.zeros(0, 2)
    valid = pins[(pins[:, 0] >= 0) & (pins[:, 1] >= 0)] if len(pins) else pins
    ratio = float(valid[:, 0].max() / max(valid[:, 1].max(), 1e-9)) if len(valid) > 1 else 1.0
    ratio = min(max(ratio, 0.1), 10.0)
    W = math.sqrt(tot / util * ratio)
    H = tot / util / W
    # Right/top-pinned preplaced blocks reveal the true width/height (the GT
    # bbox starts at the origin): use them as the outline (10-05).
    if OUTLINE_WALLS:
        kw = (tp[:, 0] + tp[:, 2])[pre & ((bnd & 2) != 0)]
        kh = (tp[:, 1] + tp[:, 3])[pre & ((bnd & 4) != 0)]
        if len(kw) and len(kh):
            W, H = float(kw.max()), float(kh.max())
        elif len(kw):
            W = float(kw.max())
            H = tot / util / W
        elif len(kh):
            H = float(kh.max())
            W = tot / util / H
    if pre.any():
        W = max(W, float((tp[pre, 0] + tp[pre, 2]).max()))
        H = max(H, float((tp[pre, 1] + tp[pre, 3]).max()))

    b2b = instance.b2b_connectivity
    b2b = b2b[b2b[:, 0] >= 0] if b2b is not None and b2b.numel() else torch.zeros(0, 3)
    bi, bj, bw = b2b[:, 0].long(), b2b[:, 1].long(), b2b[:, 2].double()
    p2b = instance.p2b_connectivity
    p2b = p2b[p2b[:, 0] >= 0] if p2b is not None and p2b.numel() else torch.zeros(0, 3)
    pb, pw = p2b[:, 1].long(), p2b[:, 2].double()
    ppx = pins[p2b[:, 0].long(), 0] if len(p2b) else torch.zeros(0, dtype=torch.float64)
    ppy = pins[p2b[:, 0].long(), 1] if len(p2b) else torch.zeros(0, dtype=torch.float64)

    # Initial centres: the given layout, else the weighted mean of connected
    # pins (outline centre without any), plus a little noise.
    if init_positions is not None:
        ip = torch.tensor(init_positions, dtype=torch.float64)
        cx0, cy0 = ip[:, 0] + ip[:, 2] / 2, ip[:, 1] + ip[:, 3] / 2
    else:
        sw = torch.zeros(n, dtype=torch.float64).index_add_(0, pb, pw)
        sx = torch.zeros(n, dtype=torch.float64).index_add_(0, pb, pw * ppx)
        sy = torch.zeros(n, dtype=torch.float64).index_add_(0, pb, pw * ppy)
        cx0 = torch.where(sw > 0, sx / sw.clamp(min=1e-12), torch.full((n,), W / 2, dtype=torch.float64))
        cy0 = torch.where(sw > 0, sy / sw.clamp(min=1e-12), torch.full((n,), H / 2, dtype=torch.float64))
        cx0 = cx0 + 0.05 * scale * torch.randn(n, generator=g, dtype=torch.float64)
        cy0 = cy0 + 0.05 * scale * torch.randn(n, generator=g, dtype=torch.float64)
    cx0 = torch.where(pre, tp[:, 0] + tp[:, 2] / 2, cx0)
    cy0 = torch.where(pre, tp[:, 1] + tp[:, 3] / 2, cy0)
    cx = cx0.clone().requires_grad_(True)
    cy = cy0.clone().requires_grad_(True)
    mov = (~pre).double()

    eps = 0.05 * scale
    sabs = lambda d: torch.sqrt(d * d + eps * eps)

    def wirelength(x, y):
        t = (bw * (sabs(x[bi] - x[bj]) + sabs(y[bi] - y[bj]))).sum() if len(bw) else x.sum() * 0
        if len(pw):
            t = t + (pw * (sabs(x[pb] - ppx) + sabs(y[pb] - ppy))).sum()
        return t

    with torch.no_grad():
        wl0 = float(wirelength(cx0, cy0)) + 1e-9
    iu = torch.triu_indices(n, n, 1)
    cl_pairs = [(a, b) for a, b in iu.t().tolist() if clu[a] > 0 and clu[a] == clu[b]]
    cla = torch.tensor([p[0] for p in cl_pairs], dtype=torch.long)
    clb = torch.tensor([p[1] for p in cl_pairs], dtype=torch.long)

    opt = torch.optim.Adam([cx, cy], lr=0.05 * scale)
    for k in range(iters):
        if time_budget is not None and time.time() - t0 > time_budget:
            break
        frac = k / max(iters - 1, 1)
        lam = ov_start * (ov_end / ov_start) ** frac       # geometric overlap ramp
        x = torch.where(pre, cx0, cx)
        y = torch.where(pre, cy0, cy)
        x0, x1, y0, y1 = x - w / 2, x + w / 2, y - h / 2, y + h / 2
        ox = (torch.minimum(x1[iu[0]], x1[iu[1]]) - torch.maximum(x0[iu[0]], x0[iu[1]])).clamp(min=0)
        oy = (torch.minimum(y1[iu[0]], y1[iu[1]]) - torch.maximum(y0[iu[0]], y0[iu[1]])).clamp(min=0)
        overlap = (ox * oy).sum() / tot
        out = ((-x0).clamp(min=0) ** 2 + (x1 - W).clamp(min=0) ** 2
               + (-y0).clamp(min=0) ** 2 + (y1 - H).clamp(min=0) ** 2).sum() / (scale * scale)
        edge = ((bnd & 1 > 0) * x0 ** 2 + (bnd & 2 > 0) * (W - x1) ** 2
                + (bnd & 8 > 0) * y0 ** 2 + (bnd & 4 > 0) * (H - y1) ** 2).sum() / (scale * scale)
        clus = (((x[cla] - x[clb]) ** 2 + (y[cla] - y[clb]) ** 2).sum() / (scale * scale)
                if len(cla) else x.sum() * 0)
        loss = (wirelength(x, y) / wl0 + lam * overlap + 10.0 * out
                + 0.5 * (1 + 10 * frac) * edge / max(n, 1) + 0.05 * clus / max(n, 1))
        opt.zero_grad()
        loss.backward()
        cx.grad *= mov
        cy.grad *= mov
        for gr in opt.param_groups:
            gr["lr"] = 0.05 * scale * (1 - 0.9 * frac)
        opt.step()

    with torch.no_grad():
        x = torch.where(pre, cx0, cx)
        y = torch.where(pre, cy0, cy)
        out = [(float(x[i] - w[i] / 2), float(y[i] - h[i] / 2), float(w[i]), float(h[i])) for i in range(n)]
    for i in range(n):          # preplaced exactly at their given spot
        if pre[i]:
            out[i] = tuple(float(v) for v in tp[i])
    return out


def analytic_solve(instance, init_positions=None, util: float = 0.85, iters: int = 500,
                   time_budget: Optional[float] = None, seed: int = 0, **kw) -> List[Position]:
    """global_place, then the constraint-graph legalizer from rl/repack.py
    (overlap-free, as tight as the arrangement allows)."""
    from . import repack as R
    from .slicing import _legalize_grow
    gp = global_place(instance, init_positions, util=util, iters=iters, time_budget=time_budget, seed=seed, **kw)
    P = R._problem(instance, gp)
    pos = P["pos"]
    if LEGALIZER == "tetris":   # rl/legalize.py, outline from the placer's spread
        from .legalize import tetris
        W, H = float((pos[:, 0] + pos[:, 2]).max()), float((pos[:, 1] + pos[:, 3]).max())
        lx, ly, lw, lh = tetris(P, pos[:, 0], pos[:, 1], pos[:, 2], pos[:, 3], W, H)
        return [(float(lx[i]), float(ly[i]), float(lw[i]), float(lh[i])) for i in range(P["n"])]
    # The slicer's variant lowers the floor when blocks in front of a preplaced
    # block don't fit, instead of leaving them overlapping it.
    # FLOORSET_ORIGIN_UPSTREAM (10-06): never below the origin; blocks that don't fit in
    # front of a preplaced block stay overlapping it for slicing to sort out.
    if ORIGIN:
        lx, ly, _ = R._legalize(pos[:, 0], pos[:, 1], pos[:, 2], pos[:, 3], P)
    else:
        lx, ly, _ = _legalize_grow(P, pos[:, 0], pos[:, 1], pos[:, 2], pos[:, 3])
    return [(float(lx[i]), float(ly[i]), float(pos[i, 2]), float(pos[i, 3])) for i in range(P["n"])]


def build_candidate(instance, rl_positions, name: str, seed: int = 0) -> List[Position]:
    """One best-of candidate "kind[:itN][:poX][:sdN]", polished (see best_of)."""
    from .repack import repack_polish
    from .slicing import slicing_polish
    parts = name.split(":")
    kind = parts[0]
    opts = {q[:2]: q[2:] for q in parts[1:]}
    it, budget = int(opts.get("it", 500)), float(opts.get("po", 1.0))
    sd = seed + int(opts.get("sd", 0))
    if kind == "base":
        p = rl_positions
    elif kind.startswith("refine"):
        p = analytic_solve(instance, init_positions=rl_positions, util=0.85, iters=min(it, 300),
                           ov_start=float(kind[6:]), seed=sd)
    elif kind.startswith("scratch"):
        p = analytic_solve(instance, util=float(kind[7:]), iters=it, seed=sd)
    elif kind.startswith("learned"):   # 10-08: learned placer(s) (rl/learned_placer.py) + shape fitting
        return learned_candidate(instance, rl_positions, index=int(kind[7:] or 1))
    else:
        raise ValueError(name)
    return repack_polish(instance, slicing_polish(instance, p, time_budget=budget / 2),
                         time_budget=budget / 2, seed=seed)


_PLACER = {}


# Checkpoints of the learned candidates "learned" / "learned2" / "learned3" ...
# (env FLOORSET_PLACER, FLOORSET_PLACER2, ... override them).
# 10-09: second slot = placer_d.pt (256 x 8 network with normalized message passing,
# val centre L1 0.052) instead of placer_b.pt: placers {a, c, d} min-of 1.128 vs {a, b, c} 1.142.
# 10-09 (evening): slots 2 / 3 = the relation-head network (placer_r.pt, 256 x 8, norm, rel),
# arrangement from its pairwise relations / from its centres. Packer-6 test, better-of:
# {a, r-rel, r-centre} 1.114 vs {a, d, c} 1.128; with the official results 1.0807 -> 1.0732 (est.);
# official 1.0727. 10-10: placer_r.pt = 312k-design relation run (val centre L1 0.050, relation
# accuracy 0.918): {a, r-rel, r-centre} 1.082 (was 1.114); previous file kept as placer_r_run1.pt.
# An entry is a file, or (file, use_rel).
# 10-11: slot 4 = placer_relB.pt (independent relation network from placer_d.pt, seed 31).
PLACER_FILES = ("placer_a.pt", ("placer_r.pt", True), ("placer_r.pt", False), ("placer_relB.pt", True))
# Sample-and-rank (10-09): official 1.0801 with 8 vs 1.0815 without -- noise, +20 s per
# design; off by default (the packer at budget 100 evens out starts of the same kind).
SAMPLES = int(os.environ.get("FLOORSET_PLACER_SAMPLES", "1"))     # 1 = prediction only
SAMPLE_NOISE = float(os.environ.get("FLOORSET_PLACER_NOISE", "0.02"))          # centres, pin-box units
SAMPLE_ASPECT_NOISE = float(os.environ.get("FLOORSET_PLACER_ASPECT_NOISE", "0.2"))   # log aspect


def learned_candidate(instance, fallback, index: int = 1):
    """Predicted arrangement -> collisions resolved (shape fitting, else moving
    blocks) -> shapes fitted. Falls back to `fallback` without a checkpoint.
    Two checkpoints (10-08): each alone ~1.21 with packer 6 on 100 designs, the
    better of the two per design 1.164 -- they make different mistakes."""
    import os
    from . import learned_placer as L
    from . import packer as K
    from . import repack as R
    from .sizing import fit_shapes
    ck = os.path.join(os.path.dirname(__file__), "..", "checkpoints")
    entry = PLACER_FILES[index - 1] if index <= len(PLACER_FILES) else ""
    fname, use_rel = (entry, None) if isinstance(entry, str) else entry
    default = os.path.join(ck, fname) if fname else ""
    path = os.environ.get("FLOORSET_PLACER" + ("" if index == 1 else str(index)), default)
    if path not in _PLACER:
        _PLACER[path] = L.load(path)
    net = _PLACER[path]
    if net is None:
        return fallback
    feasible = lambda q: R.contest_relative_cost(instance, q, q)[1]

    def repair(sample):
        """Sampled arrangement -> collision-free packed layout (fast repair), or None."""
        pos, (gp, gm), over = sample
        if over <= 1e-6:
            return pos
        P = R._problem(instance, pos)
        w, h = P["pos"][:, 2].copy(), P["pos"][:, 3].copy()
        Bt = K._Batch(P, 1.0, 1.0, (0.0, 0.0))
        fx = K._origin_repair(Bt, np.asarray(gp), np.asarray(gm), w, h)
        if fx is None:
            return None
        gp, gm = fx
        x, y, o = Bt.pack(gm[None], np.argsort(gp)[None], np.argsort(gm)[None], w[None], h[None])
        if o[0] > 1e-6:
            return None
        return [(float(x[0, i]), float(y[0, i]), float(w[i]), float(h[i])) for i in range(len(w))]

    det = L.to_layout(instance, net, use_rel=use_rel)
    pos, (gp, gm), over = det
    if over > 1e-6:        # deterministic prediction: shape fitting first, then the repair
        fitted = fit_shapes(instance, pos, iters=600, lr=0.02, seqs=(gp, gm))
        pos = fitted if fitted is not pos and feasible(fitted) else repair(det)
    starts = [pos] if pos is not None and feasible(pos) else []
    # Sample-and-rank (10-09): extra sampled arrangements, ranked by the honest cheap
    # reward r2 (fit 200 steps + 1 s packer; within-design rank correlation 0.54 with the
    # final cost, pick regret 0.029 vs 0.098 for the prediction alone).
    if SAMPLES > 1:
        rng = np.random.default_rng(instance.block_count * 7919 + index)
        for _ in range(SAMPLES - 1):
            q = repair(L.to_layout(instance, net, SAMPLE_NOISE, SAMPLE_ASPECT_NOISE, rng, use_rel=use_rel))
            if q is not None and feasible(q):
                starts.append(q)
    if not starts:
        return fallback
    if len(starts) > 1:
        ref = starts[0]
        r2 = []
        for q in starts:
            f = fit_shapes(instance, q, iters=200, lr=0.02)
            f = K.compact_pack(instance, f, time_budget=1.0, seed=0)
            c, ok = R.contest_relative_cost(instance, f, ref)
            r2.append(c if ok else float("inf"))
        pos = starts[int(np.argmin(r2))]
    else:
        pos = starts[0]
    return fit_shapes(instance, pos, iters=1000, lr=0.02)


def _candidate_job(args):
    """Process-pool entry: (instance, rl_positions, name, seed) -> (cost, feasible, positions)."""
    from .repack import contest_relative_cost
    instance, rl_positions, name, seed = args
    p = build_candidate(instance, rl_positions, name, seed)
    cost, ok = contest_relative_cost(instance, p, rl_positions)
    return cost, ok, p


def best_of(instance, rl_positions, cands, seed: int = 0, verbose: bool = False, return_all: bool = False,
            pmap=None, force=()):
    """Solve-time best-of (10-04): build each candidate layout and return the
    one with the lowest contest cost relative to the policy layout
    (rl/repack.py contest_relative_cost: no ground truth needed). A candidate
    is "kind[:itN][:poX]": kind = base (policy layout) | scratch<util>
    (global_place from scratch) | refine<ov_start> (global_place from the
    policy layout); itN placer iterations, poX seconds of slicing + re-pack."""
    from .repack import contest_relative_cost, repack_polish
    from .slicing import slicing_polish
    best, best_cost = rl_positions, float("inf")
    jobs = [(instance, rl_positions, name, seed) for name in cands]
    results = pmap(_candidate_job, jobs) if pmap is not None else [_candidate_job(j) for j in jobs]
    ranked, spare = [], []
    for name, (cost, ok, p) in zip(cands, results):
        if verbose:
            print(f"  candidate {name}: relative cost {cost:.4f} feasible={ok}")
        if ok and ORIGIN and min(min(q[0] for q in p), min(q[1] for q in p)) < -1e-6:
            ok = False      # origin rule: overhanging candidates only if nothing else is feasible
            spare.append((cost, p, name))
        if ok:
            ranked.append((cost, p, name))
        if ok and cost < best_cost:
            best, best_cost = p, cost
    if not ranked and spare:
        ranked = spare
        best = min(spare, key=lambda t: t[0])[1]
    if return_all:      # feasible candidates, cheapest first (falls back to the policy layout)
        order = sorted(ranked, key=lambda t: t[0])
        # `force` names go right after the cheapest one, so a top-k cut keeps them (10-08:
        # the learned placer's layouts rank low before packing but win after it)
        forced = [t for t in order[1:] if t[2] in force]
        order = order[:1] + forced + [t for t in order[1:] if t[2] not in force]
        return [t[1] for t in order] or [rl_positions]
    return best

"""
Shape fitting for a fixed arrangement (10-07, roadmap "A").

The packer (rl/packer.py) keeps a layout's arrangement as a sequence pair and
changes soft-block shapes only by random nudges. Tests on the ground truth
showed the shapes decide area and violations: the ground truth's own
arrangement with soft shapes off by ~20% packs to 1.24 (exact shapes: 1.02).

`fit_shapes` keeps the sequence pair and optimizes every soft unit's aspect
ratio (MIB groups share one) by gradient descent on a differentiable
packing: x of each block = max over the blocks left of it of their right
edge (0 if none; preplaced blocks fixed), y likewise with "below", iterated
to the longest path. The loss is the contest cost with HPWL and bbox area
relative to the start, plus edge-pin and preplaced-collision penalties.
Snapshots are packed exactly (rl/packer.py _Batch.pack) and scored with the
real evaluator; the best one is returned if it beats the input.
"""
import math
from typing import List, Optional, Tuple

import numpy as np
import torch

from . import repack as R

Position = Tuple[float, float, float, float]
LOG_ASP = math.log(8.0) / 2     # w / sqrt(area) within [1/sqrt(8), sqrt(8)]: aspect 1/8 .. 8
CLUSTER_W = 2.0                 # weight of the gap between a cluster member and its nearest clustermate
BOX_W = 5.0                     # weight of running past the pin box (when `box` is given), 10-08


def _units(P):
    soft = ~P["fixed"] & ~P["preplaced"] & (P["area"] > 0)
    units, seen = [], set()
    for i in np.where(soft)[0]:
        g = P["mib"][i]
        if g:
            if g in seen:
                continue
            seen.add(g)
            m = np.where(P["mib"] == g)[0]
            if soft[m].all():
                units.append(m)
        else:
            units.append(np.array([i]))
    return units


def _longest(rel, d, pre, pp, floor=0.0, max_iter=None):
    """Start coordinate of every block: max over rel-predecessors of their
    end, at least floor; preplaced blocks at pp. Returns (start, collision)."""
    n = len(d)
    neg = torch.tensor(-1e9, dtype=d.dtype)
    s = torch.where(pre, pp, torch.full_like(d, floor))
    for _ in range(max_iter or n):
        cand = torch.where(rel, (s + d)[:, None], neg).max(dim=0).values.clamp(min=floor)
        new = torch.where(pre, pp, cand)
        if torch.equal(new.detach(), s.detach()):
            s = new
            break
        s = new
    coll = torch.where(pre, torch.relu(cand - pp), torch.zeros_like(d)).sum()
    return s, coll


def fit_shapes(instance, positions: List[Position], iters: int = 300, lr: float = 0.05,
               check_every: int = 25, seqs=None, verbose: bool = False, box=None) -> List[Position]:
    """Optimize soft-unit shapes for the layout's sequence pair (or `seqs`)."""
    from .packer import _Batch, sequence_pair
    P = R._problem(instance, positions)
    n = P["n"]
    if n < 2:
        return positions
    p = P["pos"]
    x0, y0, w0, h0 = p[:, 0], p[:, 1], p[:, 2].copy(), p[:, 3].copy()
    gp, gm = seqs if seqs is not None else sequence_pair(x0, y0, w0, h0)
    gp, gm = np.asarray(gp), np.asarray(gm)
    rp, rm = np.empty(n, int), np.empty(n, int)
    rp[gp], rm[gm] = np.arange(n), np.arange(n)
    before = rm[:, None] < rm[None, :]
    left = torch.tensor(before & (rp[:, None] < rp[None, :]))
    below = torch.tensor(before & (rp[:, None] > rp[None, :]))
    units = _units(P)
    if not units:
        return positions
    area = torch.tensor(P["area"], dtype=torch.float64)
    pre = torch.tensor(P["preplaced"])
    prex = torch.tensor(p[:, 0], dtype=torch.float64)
    prey = torch.tensor(p[:, 1], dtype=torch.float64)
    uid = np.full(n, -1)
    for k, u in enumerate(units):
        uid[u] = k
    is_soft = torch.tensor(uid >= 0)
    uidt = torch.tensor(np.maximum(uid, 0))
    s = torch.tensor([math.log(w0[u[0]] / math.sqrt(P["area"][u[0]])) for u in units],
                     dtype=torch.float64, requires_grad=True)
    w_fix, h_fix = torch.tensor(w0), torch.tensor(h0)
    bnd = P["bnd"]
    bl, br = torch.tensor((bnd & 1) != 0), torch.tensor((bnd & 2) != 0)
    bt, bb = torch.tensor((bnd & 4) != 0), torch.tensor((bnd & 8) != 0)
    tmp = R._Cost(P, 1.0, 1.0)
    ref_h = max(float(tmp.hpwl(x0, y0, w0, h0)), 1e-9)
    ref_a = max(float(((x0 + w0).max() - x0.min()) * ((y0 + h0).max() - y0.min())), 1e-9)
    scale = math.sqrt(ref_a)
    i_b, j_b = torch.tensor(P["b2b_i"]), torch.tensor(P["b2b_j"])
    w_b = torch.tensor(P["b2b_w"], dtype=torch.float64)
    p_b = torch.tensor(P["p2b_b"])
    p_x, p_y = torch.tensor(P["p2b_px"], dtype=torch.float64), torch.tensor(P["p2b_py"], dtype=torch.float64)
    p_w = torch.tensor(P["p2b_w"], dtype=torch.float64)
    # cluster pairs (members must touch): penalize the gap to the nearest clustermate
    cl = P["cluster"]
    cl_groups = [torch.tensor(np.where(cl == g)[0]) for g in np.unique(cl[cl > 0])]
    cl_groups = [g for g in cl_groups if len(g) > 1]

    def dims():
        ws = torch.sqrt(area) * torch.exp(s.clamp(-LOG_ASP, LOG_ASP)[uidt])
        return torch.where(is_soft, ws, w_fix), torch.where(is_soft, area / ws, h_fix)

    def forward():
        w, h = dims()
        x, cx_ = _longest(left, w, pre, prex)
        y, cy_ = _longest(below, h, pre, prey)
        W, H = (x + w).max(), (y + h).max()
        cx, cy = x + w / 2, y + h / 2
        hp = (w_b * ((cx[i_b] - cx[j_b]).abs() + (cy[i_b] - cy[j_b]).abs())).sum()
        if len(p_b):
            hp = hp + (p_w * ((cx[p_b] - p_x).abs() + (cy[p_b] - p_y).abs())).sum()
        edge = (torch.where(bl, x, 0.0).sum() + torch.where(br, W - x - w, 0.0).sum()
                + torch.where(bb, y, 0.0).sum() + torch.where(bt, H - y - h, 0.0).sum())
        q = 1 + 0.5 * (hp / ref_h + W * H / ref_a)
        gap = torch.zeros((), dtype=torch.float64)
        for g in cl_groups:
            gx0, gy0, gx1, gy1 = x[g], y[g], x[g] + w[g], y[g] + h[g]
            dx = torch.relu(torch.maximum(gx0[:, None] - gx1[None, :], gx0[None, :] - gx1[:, None]))
            dy = torch.relu(torch.maximum(gy0[:, None] - gy1[None, :], gy0[None, :] - gy1[:, None]))
            d = dx + dy + torch.eye(len(g), dtype=torch.float64) * 1e9
            gap = gap + d.min(dim=1).values.sum()
        out = (torch.relu(W - box[0]) + torch.relu(H - box[1])) if box is not None else 0.0
        return (q + 2.0 * edge / scale + 10.0 * (cx_ + cy_) / scale + CLUSTER_W * gap / scale
                + BOX_W * out / scale)

    Bt = _Batch(P, ref_h, ref_a, (0.0, 0.0))
    Bt.mib_v = sum(len({(round(w0[i], 4), round(h0[i], 4)) for i in g}) - 1 for g in tmp.mibs)
    gm_b, rp_b, rm_b = gm[None], np.argsort(gp)[None], np.argsort(gm)[None]

    def exact(wn, hn):
        x, y, over = Bt.pack(gm_b, rp_b, rm_b, wn[None], hn[None])
        return [(float(x[0, i]), float(y[0, i]), float(wn[i]), float(hn[i])) for i in range(n)], float(over[0])

    from .packer import in_box, prefer_in_box
    old, _ = R.contest_relative_cost(instance, positions, positions)
    snaps = []      # (layout, cost, inside the pin box)
    opt = torch.optim.Adam([s], lr=lr)
    for it in range(iters + 1):
        if it % check_every == 0 or it == iters:
            with torch.no_grad():
                wn, hn = (t.numpy().copy() for t in dims())
            cand, over = exact(wn, hn)
            if over <= 1e-6:
                c, ok = R.contest_relative_cost(instance, cand, positions)
                if ok:
                    snaps.append((cand, c, box is None or in_box(cand, box)))
            if it == iters:
                break
        opt.zero_grad()
        loss = forward()
        loss.backward()
        opt.step()
    if not snaps:
        return positions
    k = prefer_in_box([c for _, c, _ in snaps], [i for _, _, i in snaps]) if box is not None \
        else min(range(len(snaps)), key=lambda i: snaps[i][1])
    best, best_c, best_in = snaps[k]
    if verbose:
        print(f"  fit_shapes: relative cost {old:.4f} -> {best_c:.4f}")
    if box is not None and best_in and not in_box(positions, box) and best_c <= old * (1 + 0.025):
        return best
    return best if best_c < old else positions

"""
Stage 1 of the grid flow (10-21): RPolyPlan-style global floorplan
(arXiv 2610.28276) in the exact Prime frame with zero white space.

Every soft block is a bounding box (centre, box area a >= target A, aspect
ratio ar) plus, for each other soft block, a share of their overlap: sigma_ij
= sigmoid(r_ij - r_ji) is the part block i keeps, so sigma_ij + sigma_ji = 1.
A block's kept area is its box minus the overlap it gives away and minus any
overlap with rigid / preplaced blocks; the constraints are

  phi_i = (A_i - kept_i) / A_i <= 0          soft blocks keep their area
  phi_jk = overlap_jk / A_min <= 0           rigid blocks overlap nothing
  phi_b = gap to the required frame edge / W <= 0   edge-pinned blocks

and the objective is net-weighted centroid HPWL (smoothed |.|). Augmented
Lagrangian as in RPolyPlan: Adam on all variables inside, multipliers updated
when the violation drops below a shrinking tolerance, else the penalty grows.
Boxes are projected into the frame after every step. Since the free area is
exactly the soft blocks' total, kept areas that meet their targets inside the
frame tile it; stage 2 (rl/grid_assign.py) makes that exact on the unit grid.
"""

import math
from dataclasses import dataclass
from typing import Dict, List, Tuple

import numpy as np
import torch

from . import partition as PT

EPS_OV = 1.0          # smoothing width of the overlap plateau (frame units)
SLACK0 = float(__import__("os").environ.get("FLOORSET_ALM_SLACK", "0"))   # starting white-space allowance
OV_MIN = 2.0          # cluster MST pairs: required box overlap depth (frame units)
LR_DECAY = float(__import__("os").environ.get("FLOORSET_ALM_DECAY", "0.93"))   # step size per outer iteration


@dataclass
class GlobalResult:
    box: np.ndarray            # [n, 4] x0, y0, x1, y1 of every block's box (rigid: its shape's bbox)
    sigma: np.ndarray          # [n, n] share of the (i, j) overlap block i keeps (soft pairs)
    shape: Dict[int, object]   # rigid block -> chosen G.Shape (orientation)
    hpwl: float
    violation: float


def _ov1d(d, wi, wj):
    """RPolyPlan's 1-D overlap of two intervals of widths wi, wj with centre
    distance d: smoothed where one interval sits inside the other."""
    a = ((wi - wj).abs() + EPS_OV) / 2
    b = (wi + wj) / 2
    ad = d.abs()
    inner = b - (d * d + a * a) / (2 * a)
    mid = b - ad
    return torch.where(ad <= a, inner, torch.where(ad <= b, mid, torch.zeros_like(d)))


def solve(P: PT.Problem, part: PT.Partitioner, seeds: np.ndarray, outer: int = 40, inner: int = 50,
          ar_max: float = 4.0, verbose: bool = False) -> GlobalResult:
    n = P.n
    W, H = P.W, P.H
    dt = torch.float64
    soft = np.array([i in part.operands and i not in part.options for i in range(n)])
    soft[list(part.pockets)] = False                     # pockets are already exact
    rigid = np.array([i in part.options and i in part.operands for i in range(n)])
    S = np.where(soft)[0]
    Rg = np.where(rigid)[0]
    A = torch.tensor(P.area, dtype=dt)
    # rigid shapes: the orientation whose bbox is squarest (stage 2 may still rotate)
    shape = {int(j): min(part.options[j], key=lambda s: abs(math.log(s.w / s.h))) for j in Rg}
    # fixed obstacles: preplaced pieces, pocket blocks
    obst = []
    for i, (s, x, y) in part.pre.items():
        obst += [(x + px, y + py, pw, ph) for px, py, pw, ph in s.pieces]
    for b, (s, x, y) in part.pocket_rigid.items():
        obst += [(x + px, y + py, pw, ph) for px, py, pw, ph in s.pieces]
    for b, R in part.pockets.items():
        obst += [(r[0], r[1], r[2] - r[0], r[3] - r[1]) for r in R]
    ob = torch.tensor(obst, dtype=dt).reshape(-1, 4)
    ob_c = torch.stack([ob[:, 0] + ob[:, 2] / 2, ob[:, 1] + ob[:, 3] / 2], 1)

    # variables
    c0 = np.clip(seeds, 0, [W, H]).astype(float)
    cx = torch.tensor(c0[:, 0], dtype=dt, requires_grad=True)
    cy = torch.tensor(c0[:, 1], dtype=dt, requires_grad=True)
    t = torch.zeros(n, dtype=dt, requires_grad=True)       # box area = A * exp(t), t >= 0
    q = torch.zeros(n, dtype=dt, requires_grad=True)       # aspect = exp(q) (h / w)
    r = torch.zeros(n, n, dtype=dt, requires_grad=True)
    rw = torch.tensor([shape[j].w if j in shape else 0.0 for j in range(n)], dtype=dt)
    rh = torch.tensor([shape[j].h if j in shape else 0.0 for j in range(n)], dtype=dt)
    rcx = torch.tensor([shape[j].cx if j in shape else 0.0 for j in range(n)], dtype=dt)
    rcy = torch.tensor([shape[j].cy if j in shape else 0.0 for j in range(n)], dtype=dt)
    for i, (s, x, y) in list(part.pre.items()) + list(part.pocket_rigid.items()):
        c0[i] = (x + s.cx, y + s.cy)
    for b, R in part.pockets.items():
        c0[b] = PT.K.centroid(R)
    fixed_c = torch.tensor(c0, dtype=dt)
    movable = torch.tensor(soft | rigid)
    Ss, Rs = torch.tensor(S), torch.tensor(Rg)

    bi, bj, bw = (torch.tensor(v) for v in (P.bi, P.bj, P.bw))
    # cluster groups must end up touching: each MST edge (seed distance) of a group is a
    # constraint that the two boxes overlap by OV_MIN; stage 2 then cuts that overlap
    # between them, so they share the cut
    from scipy.sparse.csgraph import minimum_spanning_tree
    ci, cj = [], []
    for grp in P.groups:
        g_ = [int(b) for b in grp if soft[int(b)] or rigid[int(b)]]
        if len(g_) < 2:
            continue
        D = np.abs(c0[g_][:, None] - c0[g_][None]).sum(-1) + 1e-6
        T = minimum_spanning_tree(D).tocoo()
        ci += [g_[a_] for a_ in T.row]
        cj += [g_[b_] for b_ in T.col]
    # pairs with a rigid member may only touch (rigid blocks overlap nothing): margin 0
    cm = torch.tensor([0.0 if (rigid[a_] or rigid[b_]) else OV_MIN for a_, b_ in zip(ci, cj)], dtype=dt)
    ci, cj = torch.tensor(ci, dtype=torch.long), torch.tensor(cj, dtype=torch.long)
    pb, pw, px, py = (torch.tensor(v) for v in (P.pb, P.pw, P.px, P.py))
    hp_scale = float((bw.sum() + pw.sum()) * (W + H) / 4) or 1.0
    code = torch.tensor(P.bnd[:n].astype(np.int64))

    def dims():
        a = A * torch.exp(t)
        ar = torch.exp(q)
        w = torch.where(torch.tensor(soft), torch.sqrt(a / ar), rw)
        h = torch.where(torch.tensor(soft), torch.sqrt(a * ar), rh)
        return w, h

    def centres(w, h):
        x = torch.where(movable, cx, fixed_c[:, 0])
        y = torch.where(movable, cy, fixed_c[:, 1])
        # rigid: (cx, cy) is the bbox centre; the HPWL pin is the shape's centroid
        gx = torch.where(torch.tensor(rigid), x - w / 2 + rcx, x)
        gy = torch.where(torch.tensor(rigid), y - h / 2 + rcy, y)
        return x, y, gx, gy

    def hpwl(gx, gy):
        sab = lambda d: torch.sqrt(d * d + 1.0)
        v = (bw * (sab(gx[bi] - gx[bj]) + sab(gy[bi] - gy[bj]))).sum()
        return v + (pw * (sab(gx[pb] - px) + sab(gy[pb] - py))).sum()

    def constraints(x, y, w, h):
        out = []
        sig = torch.sigmoid(r - r.T)
        # soft-soft overlaps and sharing
        xs, ys, ws, hs = x[Ss], y[Ss], w[Ss], h[Ss]
        o = _ov1d(xs[:, None] - xs[None], ws[:, None], ws[None]) * _ov1d(ys[:, None] - ys[None], hs[:, None], hs[None])
        o = o * (1 - torch.eye(len(S), dtype=dt))
        give = (o * (1 - sig[Ss][:, Ss])).sum(1)
        # soft vs rigid and obstacles: the soft block gives it all
        lose = torch.zeros(len(S), dtype=dt)
        if len(Rg):
            orr = _ov1d(xs[:, None] - x[Rs][None], ws[:, None], w[Rs][None]) * \
                _ov1d(ys[:, None] - y[Rs][None], hs[:, None], h[Rs][None])
            lose = lose + orr.sum(1)
        if len(ob):
            oo = _ov1d(xs[:, None] - ob_c[None, :, 0], ws[:, None], ob[None, :, 2]) * \
                _ov1d(ys[:, None] - ob_c[None, :, 1], hs[:, None], ob[None, :, 3])
            lose = lose + oo.sum(1)
        kept = A[Ss] * torch.exp(t[Ss]) - give - lose
        out.append((A[Ss] * (1 - slack[0]) - kept) / A[Ss])
        # rigid vs rigid / obstacles
        if len(Rg) > 1:
            xr, yr, wr, hr = x[Rs], y[Rs], w[Rs], h[Rs]
            orr2 = _ov1d(xr[:, None] - xr[None], wr[:, None], wr[None]) * _ov1d(yr[:, None] - yr[None], hr[:, None], hr[None])
            iu = torch.triu_indices(len(Rg), len(Rg), 1)
            out.append(orr2[iu[0], iu[1]] / float(P.area[Rg].min()))
        if len(Rg) and len(ob):
            oro = _ov1d(x[Rs][:, None] - ob_c[None, :, 0], w[Rs][:, None], ob[None, :, 2]) * \
                _ov1d(y[Rs][:, None] - ob_c[None, :, 1], h[Rs][:, None], ob[None, :, 3])
            out.append(oro.flatten() / float(P.area[Rg].min()))
        if len(ci):
            sx = (x[ci] - x[cj]).abs() - (w[ci] + w[cj]) / 2
            sy = (y[ci] - y[cj]).abs() - (h[ci] + h[cj]) / 2
            out.append((torch.maximum(sx, sy) + cm) / max(W, H))
        # edge-pinned blocks touch their edges
        mv = torch.tensor(np.where((P.bnd[:n] != 0) & (soft | rigid))[0])
        if len(mv):
            c = code[mv]
            gaps = [torch.where((c & 1) != 0, (x[mv] - w[mv] / 2) / W, torch.zeros(len(mv), dtype=dt)),
                    torch.where((c & 2) != 0, (W - x[mv] - w[mv] / 2) / W, torch.zeros(len(mv), dtype=dt)),
                    torch.where((c & 8) != 0, (y[mv] - h[mv] / 2) / H, torch.zeros(len(mv), dtype=dt)),
                    torch.where((c & 4) != 0, (H - y[mv] - h[mv] / 2) / H, torch.zeros(len(mv), dtype=dt))]
            out += gaps
        return torch.cat(out)

    def project():
        with torch.no_grad():
            t.clamp_(min=0.0, max=math.log(4.0))
            q.clamp_(-math.log(ar_max), math.log(ar_max))
            w, h = dims()
            # keep the box inside the frame: shrink the aspect where a side is too long
            a = A * torch.exp(t)
            q.copy_(torch.where(torch.tensor(soft), torch.clamp(q, torch.log(a / (W * W)), torch.log(H * H / a)), q))
            w, h = dims()
            cx.copy_(torch.clamp(cx, w / 2, torch.clamp(W - w / 2, min=w / 2)))
            cy.copy_(torch.clamp(cy, h / 2, torch.clamp(H - h / 2, min=h / 2)))

    project()
    groups = [{"params": [cx, cy], "lr": 0.005 * max(W, H)}, {"params": [t, q, r], "lr": 0.05}]
    opt = torch.optim.Adam(groups)
    mu, lam = 1.0, None
    eps_k, theta, tau, target = 1.0, 0.95, 2.0, 0.01
    wl = viol = 0.0
    slack = [SLACK0]
    for k in range(outer):
        # continuation (10-23): start with SLACK0 white space allowed, reach 0 at 70 % of the outer loop
        slack[0] = max(0.0, SLACK0 * (1 - k / max(1, 0.7 * outer)))
        for _ in range(inner):
            opt.zero_grad()
            w, h = dims()
            x, y, gx, gy = centres(w, h)
            phi = constraints(x, y, w, h)
            if lam is None:
                lam = torch.zeros_like(phi)
            hp = hpwl(gx, gy) / hp_scale
            pen = (mu / 2) * (torch.clamp(lam / mu + phi, min=0) ** 2).sum() - (mu / 2) * ((lam / mu) ** 2).sum()
            (hp + pen).backward()
            opt.step()
            project()
        with torch.no_grad():
            w, h = dims()
            x, y, gx, gy = centres(w, h)
            phi = constraints(x, y, w, h)
            psi = float(torch.linalg.norm(torch.maximum(phi, -lam / mu)))
            wl = float(hpwl(gx, gy))
            viol = float(torch.clamp(phi, min=0).max()) if len(phi) else 0.0
            if psi < eps_k:
                lam = torch.clamp(lam + mu * phi, min=0)
                eps_k = theta * psi
            else:
                mu = min(mu * tau, 1e3)        # 10-21: uncapped, mu reached 1e17 with no gain
        for g_ in opt.param_groups:
            g_["lr"] *= LR_DECAY
        if verbose and (k % 10 == 0 or k == outer - 1):
            ns = len(S)
            parts = {"area": phi[:ns]}
            rest = phi[ns:]
            parts["rigid/obst/edge"] = rest
            print(f"    alm {k}: hpwl {wl:.1f} max violation {viol:.4f} psi {psi:.4f} mu {mu:.1f} | "
                  + " ".join(f"{kk}: max {float(v.max()) if len(v) else 0:.3f} n>0.01 {int((v > 0.01).sum())}" for kk, v in parts.items()))
        if psi < target and viol < target:
            break
    with torch.no_grad():
        w, h = dims()
        x, y, _, _ = centres(w, h)
        box = torch.stack([x - w / 2, y - h / 2, x + w / 2, y + h / 2], 1).numpy()
        sig = torch.sigmoid(r - r.T).numpy()
    return GlobalResult(box=box, sigma=sig, shape=shape, hpwl=wl, violation=viol)

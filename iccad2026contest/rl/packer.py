"""
Compact packer (roadmap step 4, 10-05): sequence-pair search on the contest
cost, started from the best-of winner (rl/analytic.py best_of).

Representation: a sequence pair (G+, G-). For blocks a, b: a before b in
both -> a left of b; a after b in G+ but before in G- -> a below b. Every
pair gets exactly one relation, so the bottom-left packing (longest paths,
blocks in G- order) never overlaps. Preplaced blocks keep their spot; a
packing that would push one off it is infeasible. Right/top-pinned blocks
with nothing right of / above them slide out to the bbox edge (safe: every
other block is related to them on the other axis).

Speed without numba: each search step scores BATCH neighbours at once
(numpy over [batch, n] / [batch, n, n]), then takes the best one under an
annealing acceptance rule.

Cost: the contest formula with HPWL / bbox area relative to the start
layout and exact boundary + grouping counts (MIB groups are reshaped
together, so their count can't change). The result is re-checked with the
real evaluator and only returned if better.
"""
import math
import os
import random
import time
from typing import List, Optional, Tuple

import numpy as np

from . import repack as R

Position = Tuple[float, float, float, float]
BATCH = int(os.environ.get("FLOORSET_PACK_BATCH", "32"))
TARGETED = float(os.environ.get("FLOORSET_PACK_TARGET", "0.5"))   # share of moves next to a partner; 10-05 cached top-3: 0 1.248, 0.25 1.239, 0.5 1.234, 0.8 1.239
WALL_PEN = float(os.environ.get("FLOORSET_PACK_WALL", "0"))   # search penalty per sqrt(area) past a wall
EDGE_RESHAPE = float(os.environ.get("FLOORSET_PACK_EDGE", "0"))   # share of edge-thinning moves
T_SCALE = float(os.environ.get("FLOORSET_PACK_T", "1.0"))   # start temperature multiplier
# Shape refit halfway through the search (10-08): big designs, packer 15, 5 seeds: -0.0066 (t -1.2),
# area gap 0.113 -> 0.109; default on, confirmed by official run official_eval_1008_refit.json.
REFIT = os.environ.get("FLOORSET_PACK_REFIT", "1") != "0"
REFIT_ITERS = int(os.environ.get("FLOORSET_PACK_REFIT_ITERS", "300"))
IDEAL = float(os.environ.get("FLOORSET_PACK_IDEAL", "0.15"))   # share of swaps aimed at a block's ideal spot; 10-07 big designs, 3-5 seeds: 0.15 -0.010 (t -1.7), 0.3 -0.004 (t -1.0)
# Pack from the origin (10-06): the ground-truth bbox starts at (0, 0) and the
# pins sit on it, so nothing may go below / left of it; a start that does is
# repacked from the origin (conflicts with preplaced blocks are searched away).
# 10-07: start repaired by _origin_repair; big designs at packer 40: 1.1474 -> 1.1415.
ORIGIN = os.environ.get("FLOORSET_ORIGIN", "1") != "0"
# Pin box (10-07): pins sit on the ground-truth outline, so no block may end
# past the pins' max x / y. Search penalty per sqrt(area) past it (0 = off);
# in-box layouts are preferred wherever a best is picked.
PINBOX_PEN = float(os.environ.get("FLOORSET_PINBOX", "3"))
RELAX_CLUSTERS = os.environ.get("FLOORSET_RELAX_CLUSTERS", "1") != "0"   # see relax_hpwl
# Cluster repair of every packer result (10-15, reattach_clusters): cap-16 official layouts
# 14 -> 8 split clusters, beat GT 68 -> 69, weighted 1.04515 -> 1.04495 (never worse).
REATTACH = os.environ.get("FLOORSET_REATTACH", "1") != "0"
# Search-only penalty per sqrt(area) of distance between the pieces of a split cluster
# (10-15 A/B: 1 beat 0 and 3; split clusters 14 -> 2). Steers toward contact.
GAP_PEN = float(os.environ.get("FLOORSET_PACK_GAP", "1"))
# Search-only pull toward the relation net's predicted centres (10-16, 0 = off),
# fading to 0 over the first half of the search: its centres are ~0.04 of the box off.
ANCHOR = float(os.environ.get("FLOORSET_PACK_ANCHOR", "0"))
ANCHOR_NET = os.environ.get("FLOORSET_ANCHOR_NET", "placer_r.pt")
_ANCHOR_NETS = {}
# In-box layouts are preferred only if at most BOX_TOL dearer on the relative-cost
# scale (~2x the contest cost, so 0.025 ~ 5% of contest cost); user choice 10-07.
BOX_TOL = float(os.environ.get("FLOORSET_BOX_TOL", "0.025"))
# 10-10: right / top pinned preplaced blocks sit on the true outline, which is
# often well inside the pin box (99: x 129 vs 147); use it as the box there.
WALLBOX = os.environ.get("FLOORSET_WALLBOX", "0") != "0"
# 10-16: repack a winner that runs past a preplaced wall with this WALL_PEN (0 = off)
WALL_PASS = float(os.environ.get("FLOORSET_WALL_PASS", "100"))
# 10-17: second packer round from the winner on every design (offline: 27/100 better, none worse);
# official with WALL_PASS 100: 1.0432 / 72 beat GT -> 1.0404 / 77, 210 -> 405 s per design
REPACK_WINNER = os.environ.get("FLOORSET_REPACK_WINNER", "1") != "0"
# 10-17: aim for the preplaced walls from the first packer round on designs that have them:
# WALL_START = penalty (0 = off) on WALL_START_JOBS extra jobs (-1 = the idle worker slots),
# or on every job when WALL_START_ALL
WALL_START = float(os.environ.get("FLOORSET_WALL_START", "0"))
WALL_START_JOBS = int(os.environ.get("FLOORSET_WALL_START_JOBS", "-1"))
WALL_START_ALL = os.environ.get("FLOORSET_WALL_START_ALL", "0") != "0"
# FloorSet-Prime copy (10-20): contour packing -- a block left of another in the
# sequence pair only pushes it where their pieces (polygon blocks) or boxes share y.
CONTOUR = os.environ.get("FLOORSET_PACK_CONTOUR", "1") != "0"


def prefer_in_box(costs, inside, tol=None):
    """Index of the pick: cheapest overall, unless an in-box one is within tol."""
    tol = BOX_TOL if tol is None else tol
    k = min(range(len(costs)), key=lambda i: costs[i])
    ins = [i for i in range(len(costs)) if inside[i]]
    if ins:
        j = min(ins, key=lambda i: costs[i])
        if costs[j] <= costs[k] * (1 + tol):
            return j
    return k


def pin_box(instance):
    """(max pin x, max pin y): the top / right limit of the layout, tightened
    to the right / top edge of right / top pinned preplaced blocks (WALLBOX)."""
    pins = instance.pins_pos
    box = None
    if pins is not None and len(pins):
        v = pins[(pins[:, 0] >= 0) & (pins[:, 1] >= 0)]
        box = (float(v[:, 0].max()), float(v[:, 1].max())) if len(v) else None
    if not WALLBOX or instance.constraints is None or instance.constraints.dim() < 2 \
            or instance.constraints.shape[1] < 5:
        return box
    n = instance.block_count
    c = instance.constraints[:n].numpy()
    t = instance.target_positions[:n].numpy()
    pre, bnd = c[:, 1] != 0, c[:, 4].astype(np.int64)
    lim = list(box) if box is not None else [math.inf, math.inf]
    for ax, bit in ((0, 2), (1, 4)):
        m = pre & ((bnd & bit) != 0)
        if m.any():
            lim[ax] = min(lim[ax], float((t[m, ax] + t[m, ax + 2]).max()))
    return None if math.isinf(lim[0]) and math.isinf(lim[1]) else tuple(lim)


def in_box(positions, box, tol=1e-6):
    """Inside [0, box] (origin rule + pin box)."""
    if min(min(q[0] for q in positions), min(q[1] for q in positions)) < -tol:
        return False
    if box is None:
        return True
    return (max(q[0] + q[2] for q in positions) <= box[0] + tol
            and max(q[1] + q[3] for q in positions) <= box[1] + tol)
EPS = 1e-6


def _orders(x, y, w, h):
    """DAGs for the two sequences of a legal layout. G-: a before b if a is
    left of b with overlapping y-ranges, below b with overlapping x-ranges, or
    fully lower-left of b. G+: the same with above / upper-left. The pair
    relation the sequence pair then implies is always true of the layout, so
    packing it can only shrink it. (Picking one relation per pair by the
    larger gap made the orders cyclic: 80/100 starts packed bigger.)"""
    x1, y1 = x + w, y + h
    tol = 1e-6 * max(float(x1.max() - x.min()), float(y1.max() - y.min()), 1.0)
    left = x1[:, None] <= x[None, :] + tol           # a entirely left of b
    below = y1[:, None] <= y[None, :] + tol
    above = below.T
    yov, xov = ~below & ~above, ~left & ~left.T
    Gm = (left & yov) | (below & xov) | (left & below)
    Gp = (left & yov) | (above & xov) | (left & above)
    np.fill_diagonal(Gm, False)
    np.fill_diagonal(Gp, False)
    return Gp, Gm


def _topo(G, key):
    """Topological order of DAG G (G[a, b]: a before b), ties by key; a
    cycle (shouldn't happen) is broken at the lowest-indegree node."""
    n = len(key)
    indeg = G.sum(axis=0).astype(int)
    done = np.zeros(n, dtype=bool)
    out = []
    for _ in range(n):
        ready = np.where(~done & (indeg == 0))[0]
        if not len(ready):
            rest = np.where(~done)[0]
            ready = rest[indeg[rest] == indeg[rest].min()]
        v = ready[np.argmin(key[ready])]
        out.append(v)
        done[v] = True
        indeg -= G[v].astype(int)
    return np.array(out)


def sequence_pair(x, y, w, h):
    """(G+, G-) as block-index sequences reproducing the layout's relations."""
    Gp, Gm = _orders(x, y, w, h)
    cx, cy = x + w / 2, y + h / 2
    gm = _topo(Gm, cx + cy)
    gp = _topo(Gp, cx - cy)
    return gp, gm


class _Batch:
    """Problem data + batched packing and cost."""

    def __init__(self, P, ref_hpwl, ref_area, floors):
        self.P = P
        n = self.n = P["n"]
        self.pre = P["preplaced"]
        self.prex, self.prey = P["pos"][:, 0], P["pos"][:, 1]
        self.fx, self.fy = floors
        bnd = P["bnd"]
        self.left, self.right = (bnd & 1) != 0, (bnd & 2) != 0
        self.top, self.bottom = (bnd & 4) != 0, (bnd & 8) != 0
        self.pinned = bnd != 0
        self.ref_hpwl, self.ref_area = max(ref_hpwl, 1e-9), max(ref_area, 1e-9)
        cl = P["cluster"]
        self.groups = [np.where(cl == g)[0] for g in range(1, int(cl.max(initial=0)) + 1)]
        self.groups = [g for g in self.groups if len(g) > 1]
        mb = P["mib"]
        self.mib_v = 0
        self.n_soft = P["n_soft"]
        from .legalize import walls
        self.walls = walls(P)
        self.walls_any = any(v is not None for v in self.walls)
        self.ar = None
        self.box = None            # (max x, max y) pin box, set by compact_pack
        self.box_pen = PINBOX_PEN
        self.wall_pen = WALL_PEN
        self.anchor, self.anchor_w = None, 0.0     # (tx, ty, weight per block), current weight
        # FloorSet-Prime polygon blocks (rl/prime_adapter.py): up to NP rectangle
        # pieces per block as offsets from its box corner, and centroid offsets.
        shapes = P.get("shapes")
        self.poly = np.array([s is not None for s in shapes]) if shapes else np.zeros(n, dtype=bool)
        self.poly_any = bool(self.poly.any())
        if self.poly_any:
            NP = max(len(s.pieces) for s in shapes if s is not None)
            self.pc = np.zeros((n, NP, 4))              # (dx, dy, w, h); w = 0 marks an empty slot
            self.cxo, self.cyo = np.zeros(n), np.zeros(n)
            for i, s in enumerate(shapes):
                if s is not None:
                    self.pc[i, :len(s.pieces)] = s.pieces
                    self.cxo[i], self.cyo[i] = s.cx, s.cy
        # Wirelength terms (10-06): parallel block-block nets merged; pin nets
        # as per-block piecewise-linear sums sum_k w_k |c - p_k|, evaluated
        # from sorted pins + prefix sums (one search per block, not per pin).
        i, j = np.minimum(P["b2b_i"], P["b2b_j"]), np.maximum(P["b2b_i"], P["b2b_j"])
        key, inv = np.unique(i * n + j, return_inverse=True)
        self.bi, self.bj = key // n, key % n
        self.bw = np.bincount(inv.ravel(), weights=P["b2b_w"], minlength=len(key)) if len(key) else np.zeros(0)
        self.pins = [self._pin_axis(P["p2b_b"], P["p2b_px"], P["p2b_w"], n),
                     self._pin_axis(P["p2b_b"], P["p2b_py"], P["p2b_w"], n)] if len(P["p2b_w"]) else None

    @staticmethod
    def _pin_axis(b, p, wt, n):
        span = 4.0 * (float(np.abs(p).max(initial=0.0)) + 1e4)     # keys of block k lie in [k, k+1) x span
        o = np.lexsort((p, b))
        keys = b[o] * span + p[o]
        cw = np.concatenate([[0.0], np.cumsum(wt[o])])
        cs = np.concatenate([[0.0], np.cumsum(wt[o] * p[o])])
        start = np.searchsorted(b[o], np.arange(n), side="left")
        end = np.searchsorted(b[o], np.arange(n), side="right")
        return keys, cw, cs, start, end, span

    def _pin_hpwl(self, c, ax):
        keys, cw, cs, start, end, span = self.pins[ax]
        k = np.searchsorted(keys, np.arange(c.shape[1]) * span + c, side="right")
        wl, sl = cw[k] - cw[start], cs[k] - cs[start]
        wr, sr = cw[end] - cw[k], cs[end] - cs[k]
        return (c * (wl - wr) - sl + sr).sum(axis=1)

    def pack(self, gm, rp, rm, w, h):
        """gm [B, n] G- sequences, rp / rm [B, n] ranks in G+ / G-. Works in
        G- order: the blocks before position k are k's only possible
        predecessors, left of it if earlier in G+ too, else below it."""
        Bn, n = w.shape
        ar = np.arange(Bn)[:, None]
        rpo = rp[ar, gm]                       # G+ rank of the block at G- position k
        wo, ho = w[ar, gm], h[ar, gm]
        pre_o = self.pre[gm]
        pxo, pyo = self.prex[gm], self.prey[gm]
        pre_any = pre_o.any(axis=0)
        # Loop state in [n, batch] layout: the predecessors [:k] are one
        # contiguous block (10-06).
        rpt = np.ascontiguousarray(rpo.T)
        wt, ht_ = np.ascontiguousarray(wo.T), np.ascontiguousarray(ho.T)
        xe = np.empty((n, Bn))                 # right / top edges by position
        ye = np.empty((n, Bn))
        xo = np.empty((n, Bn))
        yo = np.empty((n, Bn))
        over = np.zeros(Bn)
        poly = CONTOUR and self.poly_any
        if poly:
            # pieces by position [n, B, NP]: polygon blocks from the table, a
            # rectangle is one piece (its box); empty slots never overlap
            isp = self.poly[gm][..., None]
            pc = self.pc[gm]
            QX0 = np.where(isp, pc[..., 0], 0.0).transpose(1, 0, 2)
            QY0 = np.where(isp, pc[..., 1], 0.0).transpose(1, 0, 2)
            qw, qh = np.where(isp, pc[..., 2], 0.0), np.where(isp, pc[..., 3], 0.0)
            qw[..., 0] = np.where(isp[..., 0], qw[..., 0], wo)
            qh[..., 0] = np.where(isp[..., 0], qh[..., 0], ho)
            EMPTY = (qw <= 0).transpose(1, 0, 2)
            QX1 = QX0 + qw.transpose(1, 0, 2)
            QY1 = QY0 + qh.transpose(1, 0, 2)
            PXe = np.empty_like(QX0)
            PYo, PYe = np.empty_like(QX0), np.empty_like(QX0)
        for k in range(n):
            if k:
                left = rpt[:k] < rpt[k]
                cy = np.where(left, self.fy, ye[:k]).max(axis=0)
            else:
                cy = np.full(Bn, self.fy)
            cy_req = cy
            if pre_any[k]:
                pj = pre_o[:, k]
                cy = np.where(pj, pyo[:, k], cy)
            if not k:
                cx = np.full(Bn, self.fx)
            elif not CONTOUR:
                cx = np.where(left, xe[:k], self.fx).max(axis=0)
            elif poly:
                # contour: a left neighbour pushes k only where their pieces share y
                lo = cy[None, :, None, None] + QY0[k][None, :, None, :]
                hi = cy[None, :, None, None] + QY1[k][None, :, None, :]
                ov = (PYo[:k, :, :, None] < hi - EPS) & (PYe[:k, :, :, None] > lo + EPS)
                need = np.where(ov, PXe[:k, :, :, None] - QX0[k][None, :, None, :], -np.inf).max(axis=(2, 3))
                cx = np.maximum(np.where(left, need, -np.inf).max(axis=0), self.fx)
            else:
                ov = (yo[:k] < cy + ht_[k] - EPS) & (ye[:k] > cy + EPS)
                cx = np.where(left & ov, xe[:k], self.fx).max(axis=0)
            if pre_any[k]:
                over += np.where(pj, np.maximum(cx - pxo[:, k], 0) + np.maximum(cy_req - pyo[:, k], 0), 0)
                cx = np.where(pj, pxo[:, k], cx)
            xo[k], yo[k] = cx, cy
            xe[k], ye[k] = cx + wt[k], cy + ht_[k]
            if poly:
                PXe[k] = cx[:, None] + QX1[k]
                PYo[k] = np.where(EMPTY[k], np.inf, cy[:, None] + QY0[k])
                PYe[k] = np.where(EMPTY[k], -np.inf, cy[:, None] + QY1[k])
        xo, yo = xo.T, yo.T
        # successors: a later position with a higher G+ rank is right of k, lower is above
        suf_max = np.maximum.accumulate(rpo[:, ::-1], axis=1)[:, ::-1]
        suf_min = np.minimum.accumulate(rpo[:, ::-1], axis=1)[:, ::-1]
        has_r = np.zeros((Bn, n), dtype=bool)
        has_t = np.zeros((Bn, n), dtype=bool)
        has_r[:, :-1] = suf_max[:, 1:] > rpo[:, :-1]
        has_t[:, :-1] = suf_min[:, 1:] < rpo[:, :-1]
        x = np.empty((Bn, n))
        y = np.empty((Bn, n))
        x[ar, gm], y[ar, gm] = xo, yo
        hr = np.empty((Bn, n), dtype=bool)
        ht = np.empty((Bn, n), dtype=bool)
        hr[ar, gm], ht[ar, gm] = has_r, has_t
        # right/top pins with nothing beyond them slide out to the edge
        X1 = (x + w).max(axis=1, keepdims=True)
        Y1 = (y + h).max(axis=1, keepdims=True)
        if not CONTOUR:
            x = np.where(~hr & self.right[None, :] & ~self.pre[None, :], X1 - w, x)
            y = np.where(~ht & self.top[None, :] & ~self.pre[None, :], Y1 - h, y)
            return x, y, over
        # Contour packing lets left/right-related blocks share x (or y), so a
        # slide is only taken where it hits nothing (checked piece by piece).
        for ax, flag, lim in ((0, ~hr & self.right[None, :], X1[:, 0]), (1, ~ht & self.top[None, :], Y1[:, 0])):
            for j in np.where((flag & ~self.pre[None, :]).any(axis=0))[0]:
                cur = x if ax == 0 else y
                size = w if ax == 0 else h
                new = lim - size[:, j]
                go = flag[:, j] & (new > cur[:, j] + EPS)
                if not go.any():
                    continue
                trial = cur.copy()
                trial[:, j] = new
                hit = self._collides(j, *((trial, y) if ax == 0 else (x, trial)), w, h)
                cur[:, j] = np.where(go & ~hit, new, cur[:, j])
        return x, y, over

    def _pieces(self, x, y, w, h):
        """Absolute pieces [B, n, NP, 4] (x0, y0, x1, y1); empty slots never overlap."""
        if not self.poly_any:
            return np.stack([x, y, x + w, y + h], axis=-1)[:, :, None, :]
        pc = self.pc[None]
        isp = self.poly[None, :, None]
        x0 = np.where(isp, x[..., None] + pc[..., 0], x[..., None])
        y0 = np.where(isp, y[..., None] + pc[..., 1], y[..., None])
        ones = np.ones(x.shape + (pc.shape[2],))
        pw = np.where(isp, pc[..., 2], 0.0) * ones
        ph = np.where(isp, pc[..., 3], 0.0) * ones
        pw[..., 0] = np.where(self.poly[None, :], pw[..., 0], w)
        ph[..., 0] = np.where(self.poly[None, :], ph[..., 0], h)
        x1 = np.where(pw > 0, x0 + pw, -np.inf)
        return np.stack([x0, y0, x1, y0 + ph], axis=-1)

    def _collides(self, j, x, y, w, h):
        """[B] bool: does block j overlap any other block?"""
        p = self._pieces(x, y, w, h)
        a = p[:, j][:, None, :, None, :]                  # [B, 1, NP, 1, 4]
        b = p[:, :, None, :, :]                           # [B, n, 1, NP, 4]
        ov = ((np.minimum(a[..., 2], b[..., 2]) - np.maximum(a[..., 0], b[..., 0]) > EPS)
              & (np.minimum(a[..., 3], b[..., 3]) - np.maximum(a[..., 1], b[..., 1]) > EPS))
        ov[:, j] = False
        return ov.any(axis=(1, 2, 3))

    def cost(self, x, y, w, h, over):
        P = self.P
        cx, cy = x + w / 2, y + h / 2
        if self.poly_any:                    # polygon blocks: true centroid
            cx = np.where(self.poly, x + self.cxo, cx)
            cy = np.where(self.poly, y + self.cyo, cy)
        hp = np.zeros(len(x))
        if len(self.bw):     # [n, batch] layout: each gather copies contiguous rows (4.5x faster)
            i, j = self.bi, self.bj
            cxt, cyt = np.ascontiguousarray(cx.T), np.ascontiguousarray(cy.T)
            hp += self.bw @ (np.abs(cxt[i] - cxt[j]) + np.abs(cyt[i] - cyt[j]))
        if self.pins is not None:
            hp += self._pin_hpwl(cx, 0) + self._pin_hpwl(cy, 1)
        x1, y1 = x + w, y + h
        X0, Y0 = x.min(axis=1, keepdims=True), y.min(axis=1, keepdims=True)
        X1, Y1 = x1.max(axis=1, keepdims=True), y1.max(axis=1, keepdims=True)
        area = ((X1 - X0) * (Y1 - Y0))[:, 0]
        off = ((self.left & (np.abs(x - X0) > EPS)) | (self.right & (np.abs(x1 - X1) > EPS))
               | (self.top & (np.abs(y1 - Y1) > EPS)) | (self.bottom & (np.abs(y - Y0) > EPS)))
        v = off.sum(axis=1).astype(float) + self.mib_v
        gap = np.zeros(len(x))
        for g in self.groups:
            gx, gy, gx1, gy1 = x[:, g], y[:, g], x1[:, g], y1[:, g]
            ovx = np.minimum(gx1[:, :, None], gx1[:, None, :]) - np.maximum(gx[:, :, None], gx[:, None, :])
            ovy = np.minimum(gy1[:, :, None], gy1[:, None, :]) - np.maximum(gy[:, :, None], gy[:, None, :])
            t = ((ovx > EPS) & (np.abs(ovy) <= EPS)) | ((ovy > EPS) & (np.abs(ovx) <= EPS))
            lab = np.broadcast_to(np.arange(len(g)), gx.shape).copy()
            for _ in range(len(g)):
                new = np.minimum(lab, np.where(t, lab[:, None, :], len(g)).min(axis=2))
                if np.array_equal(new, lab):
                    break
                lab = new
            v += (lab == np.arange(len(g))).sum(axis=1) - 1
            if GAP_PEN > 0:     # each piece's distance to the rest of its cluster (counted from both sides)
                dist = np.maximum(-ovx, 0) + np.maximum(-ovy, 0)
                other = lab[:, :, None] != lab[:, None, :]
                di = np.where(other, dist, np.inf).min(axis=2)
                for l in range(len(g)):
                    cm = np.where(lab == l, di, np.inf).min(axis=1)
                    gap += np.where(np.isfinite(cm), cm, 0.0)
        q = 1 + R.ALPHA * (hp / self.ref_hpwl + area / self.ref_area)
        c = q * np.exp(R.BETA * v / self.n_soft)
        scale = math.sqrt(self.ref_area)
        c = np.where(over > EPS, c * (10.0 + over / scale), c)
        self.true_cost = c
        self.out_box = np.zeros(len(c))
        if self.box is not None:
            self.out_box = np.maximum(X1[:, 0] - self.box[0], 0) + np.maximum(Y1[:, 0] - self.box[1], 0)
            c = c * (1.0 + self.box_pen * self.out_box / scale)
        if GAP_PEN > 0:
            c = c * (1.0 + GAP_PEN * gap / (2 * scale))
        if self.anchor is not None and self.anchor_w > 0:
            tx, ty, aw = self.anchor
            d = (aw * (np.abs(x + w / 2 - tx) + np.abs(y + h / 2 - ty))).sum(axis=1)
            c = c * (1.0 + self.anchor_w * d / scale)
        if self.wall_pen > 0 and self.walls_any:
            # search-only pull back inside the walls of pinned preplaced blocks
            x_lo, y_lo, x_hi, y_hi = self.walls
            out = np.zeros(len(c))
            if x_lo is not None:
                out += np.maximum(x_lo - X0[:, 0], 0)
            if y_lo is not None:
                out += np.maximum(y_lo - Y0[:, 0], 0)
            if x_hi is not None:
                out += np.maximum(X1[:, 0] - x_hi, 0)
            if y_hi is not None:
                out += np.maximum(Y1[:, 0] - y_hi, 0)
            c = c * (1.0 + self.wall_pen * out / scale)
        return c


def pack_budget(n: int, base: float) -> float:
    """Packer seconds per candidate for an n-block design: base x (n / 77)^2
    (77 = rms block count of the validation set, so the total matches a flat
    `base`). Big designs need more steps and weigh most in the total score
    (exp(n / 12)); 10-05 cached 6-candidate top-4 weighted score: flat 1.250,
    linear 1.236, quadratic 1.229."""
    return max(0.25, base * (n / 77.0) ** 2)


def _next_to(gp, gm, a, b, side):
    """Sequences with block a moved next to b: side 0 right of b, 1 left,
    2 above, 3 below."""
    out = []
    for seq, after in ((gp, side in (0, 3)), (gm, side in (0, 2))):
        q = [v for v in seq.tolist() if v != a]
        k = q.index(b)
        q.insert(k + 1 if after else k, a)
        out.append(np.array(q))
    return out


def _origin_repair(Bt, gp, gm, w, h, max_moves=None):
    """Make a sequence pair pack from the origin without pushing a preplaced
    block off its spot (10-07). A block whose packed end runs into a
    preplaced block it is left of / below is moved to another side of that
    block (right, left, above, below; the option with the least remaining
    conflict, then the lowest cost, wins); repeat until nothing conflicts.
    Returns (gp, gm) or None if it can't."""
    n = len(gp)
    pre = Bt.pre
    pids = np.where(pre)[0]
    max_moves = max_moves or 3 * n

    def pack1(GP, GM):
        GP, GM = np.asarray(GP), np.asarray(GM)
        rp, rm = np.argsort(GP, axis=1), np.argsort(GM, axis=1)
        x, y, over = Bt.pack(GM, rp, rm, np.repeat(w[None], len(GP), 0), np.repeat(h[None], len(GP), 0))
        Bt.cost(x, y, np.repeat(w[None], len(GP), 0), np.repeat(h[None], len(GP), 0), over)
        return x, y, over

    rng = np.random.default_rng(0)
    seen = {(gp.tobytes(), gm.tobytes())}
    x, y, over = pack1([gp], [gm])
    for _ in range(max_moves):
        if over[0] <= EPS:
            return gp, gm
        rp, rm = np.empty(n, int), np.empty(n, int)
        rp[gp], rm[gm] = np.arange(n), np.arange(n)
        xs, ys = x[0], y[0]
        offenders = []
        for p in pids:
            bm = rm < rm[p]
            lft = bm & (rp < rp[p]) & ~pre & (xs + w > Bt.prex[p] + EPS)
            blw = bm & (rp > rp[p]) & ~pre & (ys + h > Bt.prey[p] + EPS)
            for a in np.where(lft | blw)[0]:
                offenders.append((float(max(xs[a] + w[a] - Bt.prex[p], ys[a] + h[a] - Bt.prey[p])), int(a), int(p)))
        if not offenders:
            return None
        offenders.sort(reverse=True)
        # next to the preplaced block it hits, or next to a random other block
        opts = [_next_to(gp, gm, a, p, side) for _, a, p in offenders[:16] for side in range(4)]
        opts += [_next_to(gp, gm, a, int(b), side) for _, a, _ in offenders[:8]
                 for b in rng.choice(np.where(~pre)[0], 4) if b != a for side in range(4)]
        X, Y, O = pack1([o[0] for o in opts], [o[1] for o in opts])
        o = np.lexsort((Bt.true_cost, O))
        k = next((int(i) for i in o if (opts[i][0].tobytes(), opts[i][1].tobytes()) not in seen), None)
        if k is None or O[k] > over[0] * 1.5 + EPS:      # sideways moves allowed when stuck
            return None
        gp, gm = opts[k]
        seen.add((gp.tobytes(), gm.tobytes()))
        x, y, over = X[k:k + 1], Y[k:k + 1], O[k:k + 1]
    return (gp, gm) if over[0] <= EPS else None


def anchor_targets(instance, P):
    """(tx, ty, area share) from the anchor net's predicted centres; preplaced
    blocks get weight 0. None without the checkpoint."""
    from . import learned_placer as L
    path = os.path.join(os.path.dirname(__file__), "..", "checkpoints", ANCHOR_NET)
    if path not in _ANCHOR_NETS:
        _ANCHOR_NETS[path] = L.load(path)
    net = _ANCHOR_NETS[path]
    if net is None:
        return None
    cx, cy, _ = L.predict(net, instance)
    a = np.where(P["preplaced"], 0.0, np.maximum(P["area"], 0.0))
    return np.asarray(cx, float), np.asarray(cy, float), a / max(a.sum(), 1e-12)


def compact_pack(instance, positions: List[Position], time_budget: float = 1.5,
                 seed: Optional[int] = None, verbose: bool = False,
                 box_pen: Optional[float] = None, _inner: bool = False,
                 _anchor: bool = False, wall_pen: Optional[float] = None) -> List[Position]:
    if REFIT and not _inner and time_budget >= 2.0:
        # 10-08: half the search, refit all soft shapes jointly for the arrangement
        # found (rl/sizing.py), then the other half from the refitted layout
        from .sizing import fit_shapes
        t0 = time.time()
        first = compact_pack(instance, positions, time_budget / 2, seed, verbose, box_pen, _inner=True,
                             _anchor=ANCHOR > 0, wall_pen=wall_pen)
        box = pin_box(instance) if PINBOX_PEN > 0 else None
        fitted = fit_shapes(instance, first, iters=REFIT_ITERS, lr=0.02, box=box)
        rest = max(time_budget - (time.time() - t0), 0.5)
        return compact_pack(instance, fitted, rest, None if seed is None else seed + 1, verbose, box_pen, _inner=True,
                            wall_pen=wall_pen)
    rng = random.Random(seed)
    P = R._problem(instance, positions)
    n = P["n"]
    pos = P["pos"]
    x0, y0, w0, h0 = pos[:, 0], pos[:, 1], pos[:, 2].copy(), pos[:, 3].copy()
    if n < 2:
        return positions
    start = time.time()
    tmp = R._Cost(P, 1.0, 1.0)
    ref_hpwl = tmp.hpwl(x0, y0, w0, h0)
    ref_area = float(((x0 + w0).max() - x0.min()) * ((y0 + h0).max() - y0.min()))
    overhang = min(float(x0.min()), float(y0.min())) < -EPS
    floors = (0.0, 0.0) if ORIGIN else (min(0.0, float(x0.min())), min(0.0, float(y0.min())))
    Bt = _Batch(P, ref_hpwl, ref_area, floors)
    box = pin_box(instance) if PINBOX_PEN > 0 else None
    Bt.box = box
    if box_pen is not None:
        Bt.box_pen = box_pen
    if wall_pen is not None:
        Bt.wall_pen = wall_pen
    Bt.mib_v = sum(len({(round(w0[i], 4), round(h0[i], 4)) for i in g}) - 1
                   for g in tmp.mibs)
    if _anchor:
        Bt.anchor = anchor_targets(instance, P)

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
    movable = np.where(~P["preplaced"])[0]
    # Partners for targeted "move next to" moves: wired neighbours (weighted)
    # and clustermates.
    nbrs = [[] for _ in range(n)]
    for i, j, wt in zip(P["b2b_i"], P["b2b_j"], P["b2b_w"]):
        nbrs[int(i)].append((int(j), float(wt)))
        nbrs[int(j)].append((int(i), float(wt)))
    for g in range(1, int(P["cluster"].max(initial=0)) + 1):
        m = np.where(P["cluster"] == g)[0].tolist()
        tot = sum(wt for i in m for _, wt in nbrs[i]) / max(len(m), 1) or 1.0
        for i in m:
            nbrs[i] += [(j, tot) for j in m if j != i]
    nb_ids = [np.array([j for j, _ in v]) for v in nbrs]
    nb_p = [np.array([wt for _, wt in v]) / max(sum(wt for _, wt in v), 1e-12) for v in nbrs]
    # Ideal-spot swaps (10-07): where a block's nets pull it = weighted mean of
    # its wired blocks' centres and its pins; swap it with the block nearest that.
    wire = [[] for _ in range(n)]
    for i, j, wt in zip(P["b2b_i"], P["b2b_j"], P["b2b_w"]):
        wire[int(i)].append((int(j), float(wt)))
        wire[int(j)].append((int(i), float(wt)))
    w_ids = [np.array([j for j, _ in v], dtype=int) for v in wire]
    w_wt = [np.array([wt for _, wt in v]) for v in wire]
    pin_sum = np.zeros((n, 3))       # sum w*px, sum w*py, sum w per block
    for b, px, py, wt in zip(P["p2b_b"], P["p2b_px"], P["p2b_py"], P["p2b_w"]):
        pin_sum[int(b)] += (wt * px, wt * py, wt)
    cur_xy = [x0 + w0 / 2, y0 + h0 / 2]     # centres of the current SA state

    def ideal_partner(a):
        cx, cy = cur_xy
        sw = float(w_wt[a].sum()) + pin_sum[a, 2]
        if sw <= 0:
            return None
        ix = (float((w_wt[a] * cx[w_ids[a]]).sum()) + pin_sum[a, 0]) / sw
        iy = (float((w_wt[a] * cy[w_ids[a]]).sum()) + pin_sum[a, 1]) / sw
        d = np.abs(cx[movable] - ix) + np.abs(cy[movable] - iy)
        order = np.argsort(d)
        b = int(movable[order[0]])
        return b if b != a else (int(movable[order[1]]) if len(order) > 1 else None)

    gp, gm = sequence_pair(x0, y0, w0, h0)
    if ORIGIN and overhang:
        fixed = _origin_repair(Bt, gp, gm, w0, h0)
        if fixed is not None:
            gp, gm = fixed

    def ranks(seq):
        r = np.empty_like(seq)
        r[seq] = np.arange(len(seq))
        return r

    def evaluate(GP, GM, W, H):
        GP, GM, W, H = np.asarray(GP), np.asarray(GM), np.asarray(W), np.asarray(H)
        rp = np.argsort(GP, axis=1)
        rm = np.argsort(GM, axis=1)
        x, y, over = Bt.pack(GM, rp, rm, W, H)
        return Bt.cost(x, y, W, H, over), x, y, over

    def propose(gp, gm, w, h):
        gp, gm, w, h = gp.copy(), gm.copy(), w, h
        r = rng.random()
        if r < EDGE_RESHAPE and edge_units[0]:              # thin a unit that sets the bbox edge
            w, h = w.copy(), h.copy()
            side = rng.randrange(2) if edge_units[1] else 0
            u = edge_units[side][rng.randrange(len(edge_units[side]))] if edge_units[side] else None
            if u is not None:
                f = rng.uniform(0.75, 0.97) if side == 0 else 1.0 / rng.uniform(0.75, 0.97)
                for i in u:
                    a = w[i] * h[i]
                    nw = min(max(w[i] * f, math.sqrt(a / R.MAX_ASPECT)), math.sqrt(a * R.MAX_ASPECT))
                    w[i], h[i] = nw, a / nw
            return gp, gm, w, h
        if r < 0.25 and units:                                  # reshape a soft unit
            w, h = w.copy(), h.copy()
            f = math.exp(rng.gauss(0.0, 0.3))
            for i in units[rng.randrange(len(units))]:
                a = w[i] * h[i]
                nw = min(max(w[i] * f, math.sqrt(a / R.MAX_ASPECT)), math.sqrt(a * R.MAX_ASPECT))
                w[i], h[i] = nw, a / nw
        elif r < 0.6:                                           # swap a pair in G+, G- or both
            a, b = rng.sample(range(n), 2)
            which = rng.randrange(3)
            if rng.random() < IDEAL and not P["preplaced"][a]:     # a with the block at its ideal spot
                b2 = ideal_partner(a)
                if b2 is not None:
                    b, which = b2, 2
            for seq, on in ((gp, which != 1), (gm, which != 0)):
                if on:
                    ia, ib = int(np.where(seq == a)[0][0]), int(np.where(seq == b)[0][0])
                    seq[ia], seq[ib] = b, a
        else:                                                   # move a next to b (one side)
            a = int(movable[rng.randrange(len(movable))])
            if len(nb_ids[a]) and rng.random() < TARGETED:
                b = int(nb_ids[a][np.searchsorted(np.cumsum(nb_p[a]), rng.random() * nb_p[a].sum())
                                  .clip(max=len(nb_ids[a]) - 1)])
            else:
                b = rng.randrange(n)
            if a == b:
                return gp, gm, w, h
            side = rng.randrange(4)      # 0 right of b, 1 left, 2 above, 3 below
            after_p = side in (0, 3)     # a after b in G+ ?
            after_m = side in (0, 2)     # a after b in G- ?
            out = []
            for seq, after in ((gp, after_p), (gm, after_m)):
                s = [v for v in seq.tolist() if v != a]
                k = s.index(b)
                s.insert(k + 1 if after else k, a)
                out.append(np.array(s))
            gp, gm = out
        return gp, gm, w, h

    edge_units = [[], []]       # soft units touching the right / top bbox edge of the current layout

    def set_edges(x, y, w, h):
        X1, Y1 = (x + w).max(), (y + h).max()
        tol = 1e-6 * max(X1, Y1)
        edge_units[0] = [u for u in units if (x[u] + w[u] >= X1 - tol).any()]
        edge_units[1] = [u for u in units if (y[u] + h[u] >= Y1 - tol).any()]

    c, x, y, over = evaluate([gp], [gm], w0[None], h0[None])
    set_edges(x[0], y[0], w0, h0)
    cur = (float(c[0]), gp, gm, w0, h0)
    # best overall and best inside the pin box; the in-box one wins if within BOX_TOL
    feas0 = over[0] <= EPS
    best = [float(Bt.true_cost[0]) if feas0 else float("inf"), ((gp, gm, w0, h0), x[0], y[0]) if feas0 else None]
    best_in = [float(Bt.true_cost[0]) if feas0 and Bt.out_box[0] <= EPS else float("inf"),
               best[1] if feas0 and Bt.out_box[0] <= EPS else None]
    steps = 0
    t0 = None
    while time.time() - start < time_budget:
        props = [propose(cur[1], cur[2], cur[3], cur[4]) for _ in range(BATCH)]
        cs, xs, ys, ovs = evaluate([p[0] for p in props], [p[1] for p in props],
                                   [p[2] for p in props], [p[3] for p in props])
        tc, ob = Bt.true_cost, Bt.out_box
        for slot, mask in ((best, ovs <= EPS), (best_in, (ovs <= EPS) & (ob <= EPS))):
            kt = int(np.argmin(np.where(mask, tc, np.inf)))
            if mask[kt] and tc[kt] < slot[0]:
                slot[0], slot[1] = float(tc[kt]), (props[kt], xs[kt], ys[kt])
        k = int(np.argmin(cs))
        if t0 is None:          # start temperature: typical uphill step of the best neighbour
            t0 = T_SCALE * max(abs(float(cs[k]) - cur[0]), 1e-4)
        frac = (time.time() - start) / time_budget
        temp = t0 * (0.001 ** frac)
        if Bt.anchor is not None:
            Bt.anchor_w = ANCHOR * max(0.0, 1.0 - frac)
        if cs[k] <= cur[0] or rng.random() < math.exp(-(cs[k] - cur[0]) / max(temp, 1e-12)):
            cur = (float(cs[k]),) + props[k]
            cur_xy[0], cur_xy[1] = xs[k] + cur[3] / 2, ys[k] + cur[4] / 2
            if EDGE_RESHAPE > 0:
                set_edges(xs[k], ys[k], cur[3], cur[4])
        steps += 1
    if best[1] is None:
        return positions
    best_t = best_in[1] if best_in[1] is not None and best_in[0] <= best[0] * (1 + BOX_TOL) else best[1]
    (_, _, bw, bh), bx, by = best_t
    cand = [(float(bx[i]), float(by[i]), float(bw[i]), float(bh[i])) for i in range(n)]
    old, _ = R.contest_relative_cost(instance, positions, positions)
    new, ok = R.contest_relative_cost(instance, cand, positions)
    if verbose:
        print(f"  packer: {steps} steps x {BATCH}, relative cost {old:.4f} -> {new:.4f} (feasible={ok}, "
              f"fast cost start {Bt.cost(*(a[None] for a in (x0, y0, w0, h0)), np.zeros(1))[0]:.4f})")
    if ORIGIN and overhang:      # inside the box beats a cheaper overhanging start
        return cand if ok else positions
    if box is not None and ok and in_box(cand, box) != in_box(positions, box):
        inside, outside = (new, old) if in_box(cand, box) else (old, new)
        keep_cand = (inside <= outside * (1 + BOX_TOL)) == in_box(cand, box)
        return cand if keep_cand else positions
    return cand if ok and new < old else positions


def _wmedian_range(v, wt):
    """[lo, hi] of the weighted medians of v (every point in it is optimal)."""
    o = np.argsort(v)
    v, c = v[o], np.cumsum(wt[o])
    half = c[-1] / 2
    k = int(np.searchsorted(c, half - 1e-12))
    if abs(c[k] - half) <= 1e-12 * max(c[-1], 1.0) and k + 1 < len(v):
        return float(v[k]), float(v[k + 1])
    return float(v[k]), float(v[k])


_SQ_DEBUG = False


def _alap(gm, rel, d, pre, pp, T):
    """Latest start of each block (preplaced fixed) with everything ending by T."""
    st = np.zeros(len(d))
    for k in gm[::-1]:
        q = rel[k]
        lim = min(float(st[q].min()) if q.any() else T, T)
        st[k] = pp[k] if pre[k] else lim - d[k]
    return st


def squeeze(instance, positions: List[Position], rounds: int = 200,
            fracs=(1.0, 0.6, 0.3), box=None) -> List[Position]:
    """Slack-driven reshaping (10-06): keep the layout's sequence pair; each
    round, every soft unit on the height-critical path that has room to its
    right is widened into that room (so it gets shorter), and vice versa for
    the width-critical path; all variants are packed exactly and the best
    one is taken while the cost drops. Targets the white space bottom-left
    packing leaves next to critical columns. Official 10-06 layouts:
    weighted 1.2250 -> 1.2203 (area gap 0.146 -> 0.141), 0.15 s per design."""
    P = R._problem(instance, positions)
    n = P["n"]
    if n < 2:
        return positions
    p = P["pos"]
    x0, y0, w, h = p[:, 0], p[:, 1], p[:, 2].copy(), p[:, 3].copy()
    gp, gm = sequence_pair(x0, y0, w, h)
    rp, rm = np.empty(n, int), np.empty(n, int)
    rp[gp], rm[gm] = np.arange(n), np.arange(n)
    before = rm[:, None] < rm[None, :]
    left, below = before & (rp[:, None] < rp[None, :]), before & (rp[:, None] > rp[None, :])
    floors = (min(0.0, float(x0.min())), min(0.0, float(y0.min())))
    tmp = R._Cost(P, 1.0, 1.0)
    ref_area = float(((x0 + w).max() - x0.min()) * ((y0 + h).max() - y0.min()))
    Bt = _Batch(P, tmp.hpwl(x0, y0, w, h), ref_area, floors)
    Bt.mib_v = sum(len({(round(w[i], 4), round(h[i], 4)) for i in g}) - 1 for g in tmp.mibs)
    Bt.box = box
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
    pre, area = P["preplaced"], P["area"]

    def evaluate(W, H):
        W, H = np.asarray(W), np.asarray(H)
        k = len(W)
        x, y, over = Bt.pack(np.repeat(gm[None], k, 0), np.repeat(rp[None], k, 0), np.repeat(rm[None], k, 0), W, H)
        c = Bt.cost(x, y, W, H, over)     # with a box: pin-box penalty included
        return np.where(over > EPS, np.inf, c if box is not None else Bt.true_cost), x, y

    c, xs, ys = evaluate([w], [h])
    cur, x, y = float(c[0]), xs[0], ys[0]
    for _ in range(rounds):
        # room is measured to the current outline, or to the pin box when it is larger
        X1, Y1 = float((x + w).max()), float((y + h).max())
        if box is not None:
            X1, Y1 = max(X1, box[0]) if Y1 > box[1] else X1, max(Y1, box[1]) if X1 > box[0] else Y1
        sx = _alap(gm, left, w, pre, x, X1) - x
        sy = _alap(gm, below, h, pre, y, Y1) - y
        cw, ch = [], []
        joint = {(d, f): (w.copy(), h.copy()) for d in (0, 1) for f in fracs + (0.15,)}
        for u in units:
            i = u[0]
            ux, uy = float(sx[u].min()), float(sy[u].min())
            lo, hi = math.sqrt(area[i] / R.MAX_ASPECT), math.sqrt(area[i] * R.MAX_ASPECT)
            for f in fracs + (0.15,):
                if uy < 1e-6 < ux:          # height-critical: widen into the x room
                    d, nw = 0, min(w[i] + f * ux, hi)
                elif ux < 1e-6 < uy:        # width-critical: heighten into the y room
                    d, nw = 1, max(area[i] / (h[i] + f * uy), lo)
                else:
                    continue
                if abs(nw - w[i]) < 1e-6:
                    continue
                Wj, Hj = joint[(d, f)]      # every critical unit at once (parallel critical paths)
                Wj[u], Hj[u] = nw, area[u] / nw
                if f == 0.15:
                    continue
                W2, H2 = w.copy(), h.copy()
                W2[u], H2[u] = nw, area[u] / nw
                cw.append(W2)
                ch.append(H2)
        for Wj, Hj in joint.values():
            if not np.array_equal(Wj, w):
                cw.append(Wj)
                ch.append(Hj)
        if not cw:
            break
        c, xs, ys = evaluate(cw, ch)
        k = int(np.argmin(c))
        if c[k] >= cur - 1e-9:
            break
        cur, w, h, x, y = float(c[k]), cw[k], ch[k], xs[k], ys[k]
        if _SQ_DEBUG:
            print(f"    squeeze round: cost {cur:.4f} bbox {(x + w).max():.1f} x {(y + h).max():.1f} cands {len(cw)}")
    cand = [(float(x[i]), float(y[i]), float(w[i]), float(h[i])) for i in range(n)]
    old, _ = R.contest_relative_cost(instance, positions, positions)
    new, ok = R.contest_relative_cost(instance, cand, positions)
    if box is not None and ok and in_box(cand, box) and not in_box(positions, box):
        return cand
    return cand if ok and new < old else positions


def relax_hpwl(instance, positions: List[Position], sweeps: int = 30) -> List[Position]:
    """Wirelength relaxation inside the layout's own slack (10-06): keep its
    sequence pair (every pair keeps its left/below relation) and the bbox,
    and move each block, one axis at a time, toward the weighted median of
    what it connects to (the exact optimum of that axis with the others
    fixed). Preplaced, edge-pinned (on their axis) and cluster blocks stay,
    so no violation can appear. 100 official 10-06 layouts: weighted
    1.2335 -> 1.2308, ~4 ms per design."""
    P = R._problem(instance, positions)
    n = P["n"]
    if n < 2:
        return positions
    p = P["pos"]
    x, y, w, h = p[:, 0].copy(), p[:, 1].copy(), p[:, 2], p[:, 3]
    gp, gm = sequence_pair(x, y, w, h)
    rp, rm = np.empty(n, int), np.empty(n, int)
    rp[gp], rm[gm] = np.arange(n), np.arange(n)
    before = rm[:, None] < rm[None, :]
    rel = (before & (rp[:, None] < rp[None, :]), before & (rp[:, None] > rp[None, :]))   # left of, below
    bnd, pre, cl = P["bnd"], P["preplaced"], P["cluster"] > 0
    # Cluster members move too (10-07, RELAX_CLUSTERS) as long as their cluster stays
    # in one touching piece; otherwise they stay put.
    frz_cl = cl & (not RELAX_CLUSTERS)
    frozen = (pre | frz_cl | ((bnd & 3) != 0), pre | frz_cl | ((bnd & 12) != 0))
    groups = {int(g): np.where(P["cluster"] == g)[0] for g in np.unique(P["cluster"][cl])}

    def pieces(g):
        gx, gy, gx1, gy1 = x[g], y[g], x[g] + w[g], y[g] + h[g]
        ovx = np.minimum(gx1[:, None], gx1[None, :]) - np.maximum(gx[:, None], gx[None, :])
        ovy = np.minimum(gy1[:, None], gy1[None, :]) - np.maximum(gy[:, None], gy[None, :])
        t = ((ovx > EPS) & (np.abs(ovy) <= EPS)) | ((ovy > EPS) & (np.abs(ovx) <= EPS))
        lab = np.arange(len(g))
        for _ in range(len(g)):
            new = np.minimum(lab, np.where(t, lab[None, :], len(g)).min(axis=1))
            if np.array_equal(new, lab):
                break
            lab = new
        return int((lab == np.arange(len(g))).sum())
    nb = [[] for _ in range(n)]
    for i, j, wt in zip(P["b2b_i"], P["b2b_j"], P["b2b_w"]):
        nb[int(i)].append((int(j), float(wt)))
        nb[int(j)].append((int(i), float(wt)))
    pins = [[] for _ in range(n)]
    for b, px, py, wt in zip(P["p2b_b"], P["p2b_px"], P["p2b_py"], P["p2b_w"]):
        pins[int(b)].append((float(px), float(py), float(wt)))
    nets = []
    for i in range(n):
        js = np.array([j for j, _ in nb[i]], dtype=int)
        wts = np.array([wt for _, wt in nb[i]] + [q[2] for q in pins[i]])
        nets.append((js, np.array([[q[0], q[1]] for q in pins[i]]).reshape(-1, 2), wts))
    box = ((x.min(), (x + w).max()), (y.min(), (y + h).max()))
    for _ in range(sweeps):
        moved = 0.0
        for i in range(n):
            js, pp, wts = nets[i]
            if not len(wts):
                continue
            for ax, (pos, d) in enumerate(((x, w), (y, h))):
                if frozen[ax][i]:
                    continue
                c = pos + d / 2
                m0, m1 = _wmedian_range(np.concatenate([c[js], pp[:, ax]]), wts)
                pred, succ = rel[ax][:, i], rel[ax][i, :]
                lo = max(box[ax][0], float((pos[pred] + d[pred]).max(initial=-np.inf)))
                hi = min(box[ax][1], float(pos[succ].min(initial=np.inf))) - d[i]
                if hi < lo - EPS:
                    continue
                ci = pos[i] + d[i] / 2
                tgt = ci if m0 - EPS <= ci <= m1 + EPS else (m0 if ci < m0 else m1)
                nx = min(max(tgt - d[i] / 2, lo), hi)
                if abs(nx - pos[i]) < 1e-12:
                    continue
                if cl[i]:
                    g = groups[int(P["cluster"][i])]
                    before_p, old_v = pieces(g), pos[i]
                    pos[i] = nx
                    if pieces(g) > before_p:
                        pos[i] = old_v
                        continue
                moved += abs(nx - pos[i]) if not cl[i] else abs(nx - old_v)
                pos[i] = nx
        if moved < 1e-6:
            break
    cand = [(float(x[i]), float(y[i]), float(w[i]), float(h[i])) for i in range(n)]
    old, _ = R.contest_relative_cost(instance, positions, positions)
    new, ok = R.contest_relative_cost(instance, cand, positions)
    return cand if ok and new < old else positions


def _attach_spots(P, x, y, w, h, a, anchors, soft, ratios, t):
    """Candidate (x, y, w, h) arrays for block a abutting any of `anchors`:
    slides along each side (soft: several aspect ratios), and for soft a,
    stretches that keep a's far edge and end exactly on the anchor's edge."""
    ar = w[a] * h[a]
    sw = np.concatenate([[w[a]], np.sqrt(ar * ratios)]) if soft[a] else np.array([w[a]])
    sh = ar / sw
    out = []

    def add(px, py, wa, ha):
        px, py = np.atleast_1d(np.asarray(px, float)), np.atleast_1d(np.asarray(py, float))
        px, py = np.broadcast_arrays(px, py)
        out.append((px, py, np.full(px.shape, wa), np.full(px.shape, ha)))
    for b in anchors:
        for wa, ha in zip(sw, sh):
            d = 1e-3 * min(wa, ha, w[b], h[b])      # keep a real overlap along the shared edge
            ys = y[b] - ha + d + t * (h[b] + ha - 2 * d)
            xs = x[b] - wa + d + t * (w[b] + wa - 2 * d)
            add(x[b] + w[b], ys, wa, ha), add(x[b] - wa, ys, wa, ha)
            add(xs, y[b] + h[b], wa, ha), add(xs, y[b] - ha, wa, ha)
        if not soft[a]:
            continue
        for lo, hi, horiz in ((x[a], x[b], True), (x[b] + w[b], x[a] + w[a], True),
                              (y[a], y[b], False), (y[b] + h[b], y[a] + h[a], False)):
            ln = hi - lo
            if ln <= EPS or not (1.0 / R.MAX_ASPECT <= (ln * ln / ar if horiz else ar / (ln * ln)) <= R.MAX_ASPECT):
                continue
            if horiz:
                ha = ar / ln
                add(lo, [y[a], y[a] + h[a] - ha, y[b], y[b] + h[b] - ha], ln, ha)
            else:
                wa = ar / ln
                add([x[a], x[a] + w[a] - wa, x[b], x[b] + w[b] - wa], lo, wa, ln)
    return [np.concatenate([o[k] for o in out]) for k in range(4)]


def reattach_clusters(instance, positions: List[Position], rounds: int = 4,
                      offsets: int = 9, n_shapes: int = 13) -> List[Position]:
    """Cluster repair after packing (10-15): a member of one touching piece
    of a split cluster tries every free spot abutting another piece
    (_attach_spots); all spots are scored at once with the exact cost and
    the best kept while cheaper. Nothing else moves, so contact is exact."""
    from .legalize import _components
    P = R._problem(instance, positions)
    n = P["n"]
    pos = P["pos"]
    x, y, w, h = pos[:, 0].copy(), pos[:, 1].copy(), pos[:, 2].copy(), pos[:, 3].copy()
    cl = P["cluster"]
    groups = [g for g in (np.where(cl == k)[0] for k in range(1, int(cl.max(initial=0)) + 1)) if len(g) > 1]
    if n < 2 or not groups:
        return positions
    tmp = R._Cost(P, 1.0, 1.0)
    ref_area = float(((x + w).max() - x.min()) * ((y + h).max() - y.min()))
    Bt = _Batch(P, tmp.hpwl(x, y, w, h), ref_area, (0.0, 0.0))
    Bt.box = pin_box(instance) if PINBOX_PEN > 0 else None
    Bt.mib_v = sum(len({(round(w[i], 4), round(h[i], 4)) for i in g}) - 1 for g in tmp.mibs)
    soft = ~P["fixed"] & ~P["preplaced"] & (P["area"] > 0) & (P["mib"] == 0)
    ratios = np.geomspace(1.0 / R.MAX_ASPECT, R.MAX_ASPECT, n_shapes)
    t = np.linspace(0.0, 1.0, offsets)
    cur = float(Bt.cost(x[None], y[None], w[None], h[None], np.zeros(1))[0])
    for _ in range(rounds):
        improved = False
        for g in groups:
            comps = _components(P, x, y, w, h, g)
            if len(comps) < 2:
                continue
            for ci, comp in enumerate(comps):
                anchors = np.concatenate([c for cj, c in enumerate(comps) if cj != ci])
                for a in (int(v) for v in comp if not P["preplaced"][v]):
                    cx, cy, cw, ch = _attach_spots(P, x, y, w, h, a, anchors, soft, ratios, t)
                    others = np.arange(n) != a
                    ovx = np.minimum(cx[:, None] + cw[:, None], (x + w)[None, others]) - np.maximum(cx[:, None], x[None, others])
                    ovy = np.minimum(cy[:, None] + ch[:, None], (y + h)[None, others]) - np.maximum(cy[:, None], y[None, others])
                    k = np.where(~((ovx > EPS) & (ovy > EPS)).any(axis=1) & (cx >= -EPS) & (cy >= -EPS))[0]
                    if not len(k):
                        continue
                    X, Y = np.repeat(x[None], len(k), 0), np.repeat(y[None], len(k), 0)
                    W, H = np.repeat(w[None], len(k), 0), np.repeat(h[None], len(k), 0)
                    X[:, a], Y[:, a], W[:, a], H[:, a] = cx[k], cy[k], cw[k], ch[k]
                    c = Bt.cost(X, Y, W, H, np.zeros(len(k)))
                    j = int(np.argmin(c))
                    if c[j] < cur - 1e-12:
                        cur, improved = float(c[j]), True
                        x[a], y[a], w[a], h[a] = X[j, a], Y[j, a], W[j, a], H[j, a]
        if not improved:
            break
    cand = [(float(x[i]), float(y[i]), float(w[i]), float(h[i])) for i in range(n)]
    old, _ = R.contest_relative_cost(instance, positions, positions)
    new, ok = R.contest_relative_cost(instance, cand, positions)
    return cand if ok and new < old else positions


def has_walls(instance, positions: List[Position]) -> bool:
    """Does a right/top-pinned preplaced block reveal part of the outline?"""
    from .legalize import walls
    _, _, xh, yh = walls(R._problem(instance, positions))
    return xh is not None or yh is not None


def wall_violations(instance, positions: List[Position]) -> int:
    """Pinned preplaced blocks whose required edge is off the bounding box
    (the layout runs past the wall they reveal)."""
    P = R._problem(instance, positions)
    p, bnd = P["pos"], P["bnd"]
    x, y, x1, y1 = p[:, 0], p[:, 1], p[:, 0] + p[:, 2], p[:, 1] + p[:, 3]
    off = (((bnd & 1) != 0) & (np.abs(x - x.min()) > EPS)) | (((bnd & 2) != 0) & (np.abs(x1 - x1.max()) > EPS)) \
        | (((bnd & 4) != 0) & (np.abs(y1 - y1.max()) > EPS)) | (((bnd & 8) != 0) & (np.abs(y - y.min()) > EPS))
    return int((off & P["preplaced"]).sum())


def _pack_job(args):
    """Process-pool entry: (instance, positions, time_budget, seed[, box_pen[, wall_pen]]) -> positions."""
    instance, positions, budget, seed = args[:4]
    out = compact_pack(instance, positions, time_budget=budget, seed=seed,
                       box_pen=args[4] if len(args) > 4 else None, wall_pen=args[5] if len(args) > 5 else None)
    return reattach_clusters(instance, out) if REATTACH else out


def init_worker():
    """Process-pool initializer (my_optimizer.py): one torch thread per worker."""
    import torch
    torch.set_num_threads(1)

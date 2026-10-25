"""
Pair re-cut refinement (10-21 night): stage 3 of the RPolyPlan / JigsawPlanner
plan applied to a valid zero-white-space layout (the partition flow's).

Move: two touching soft blocks u, v are merged and re-split by one straight
cut at the exact area of u (rl/partition.py split, either axis, either side
first). Areas stay exact and no white space appears; the move is kept when
both blocks stay valid leaves (one piece, no hole, <= MAX_CORNERS corners),
no cluster / edge-pin violation is added and the net-weighted centroid HPWL
drops. Pairs re-cut independently of any tree, so the layout can leave the
slicing family (85 of 100 Prime ground truths are not slicing). The move set
is what a learned policy can choose from later.
"""

import math
import os
import time
from typing import Dict, List, Sequence

import numpy as np
from shapely.geometry import Polygon

from . import partition as PT

EPS = 1e-7
TRIPLES = True
MERGE = 1e-5          # output: coordinates closer than this become one
PUSH = True           # targeted pushes for edge-pin misses and split clusters
CORRIDOR = True       # edge pins: also try a corridor re-slice (10-23)
WINDOW = True         # edge pins still off: window re-partition (rl/window.py, 10-24)
# share of run()'s budget for window LNS (rl/window.py lns) before the kicks; 0 = off.
# Design 97: 1.012 -> 1.000 in 600 s of LNS alone (10-24)
LNS = float(os.environ.get("FLOORSET_REFINE_LNS", "0"))
# an edge pin needs this much contact length: Refiner.layout() merges coordinates closer
# than MERGE, which erased a 1e-6 sliver that alone met an edge (10-24, designs 23 and 73)
CONTACT = 1e-3


def edge_gap(R: np.ndarray, code: int, W: float, H: float, eps: float = EPS) -> float:
    """Summed distance of R from the frame edges in its edge-pin code (0 when it
    meets them all), counting only rectangles at least CONTACT long along each edge."""
    tall = R[R[:, 3] - R[:, 1] >= CONTACT]
    wide = R[R[:, 2] - R[:, 0] >= CONTACT]
    tall = tall if len(tall) else R
    wide = wide if len(wide) else R
    gaps = (((code & 1) and tall[:, 0].min()), ((code & 2) and W - tall[:, 2].max()),
            ((code & 8) and wide[:, 1].min()), ((code & 4) and H - wide[:, 3].max()))
    return float(sum(g for g in gaps if g > eps))


class DontLook:
    """Don't-look bits for Refiner._greedy: a move that gave no gain is skipped until
    one of its blocks changes. In LNS, cleanup ran 2.7-3.1x faster with the same wins
    (10-24); moves of net neighbours are ignored, so a few small gains are missed."""

    def __init__(self, n: int):
        self.clock = 1
        self.stamp = np.ones(n, dtype=np.int64)
        self.checked = {}

    def touch(self, blocks):
        self.clock += 1
        for b in blocks:
            self.stamp[b] = self.clock

    def need(self, key) -> bool:
        return self.checked.get(key, 0) < max(self.stamp[b] for b in key)

    def done(self, key):
        self.checked[key] = self.clock


def snap(R: np.ndarray) -> np.ndarray:
    """Round coordinates to 1e-6 so edges that should coincide do exactly: a
    float-noise gap left a cluster split for the evaluator (design 30, 10-21)."""
    if not len(R):
        return R
    R = np.round(R, 6)
    return R[(R[:, 2] - R[:, 0] > 1e-9) & (R[:, 3] - R[:, 1] > 1e-9)]


class Refiner:
    def __init__(self, P: PT.Problem, layout: Sequence[Sequence]):
        self.P = P
        n = P.n
        self.rects: Dict[int, np.ndarray] = {}
        for i, v in enumerate(layout):
            self.rects[i] = snap(PT.decompose_geom(Polygon(v)))
        self.soft = [i for i in range(n) if not P.rigid[i]]
        # edge pins are judged on the layout's own extent (the packer's exceeds the pin box)
        self.W = max(float(R[:, 2].max()) for R in self.rects.values())
        self.H = max(float(R[:, 3].max()) for R in self.rects.values())
        self.c = np.array([PT.K.centroid(self.rects[i]) for i in range(n)])
        # incident nets per block for incremental HPWL
        self.b2b = [[] for _ in range(n)]
        for k, (i, j) in enumerate(zip(P.bi, P.bj)):
            self.b2b[i].append(k)
            self.b2b[j].append(k)
        self.p2b = [[] for _ in range(n)]
        for k, b in enumerate(P.pb):
            self.p2b[b].append(k)
        # rigid blocks that may move (fixed shapes, MIB rectangles; not preplaced), rotations only
        from . import polygon as G
        self.movable_rigid = [i for i in range(n) if P.rigid[i] and not P.pre[i]]
        # MIB members keep their orientation (a rotated member counts as a different shape: 10-21, design 30)
        self.shapes = {i: ([G.shape_from_polygon(PT.to_polygon(self.rects[i]))] if P.mib[i] else
                           G.orientations(G.shape_from_polygon(PT.to_polygon(self.rects[i]))))
                       for i in self.movable_rigid}
        self.dlb = None                             # DontLook while LNS runs
        self.group_of = {}
        for g in P.groups:
            members = [int(x) for x in g]           # one list per group: counted once (was once per member)
            for b in members:
                self.group_of[b] = members

    def hp_local(self, blocks, c) -> float:
        P = self.P
        ks = sorted({k for b in blocks for k in self.b2b[b]})
        ps = sorted({k for b in blocks for k in self.p2b[b]})
        v = 0.0
        if ks:
            ks = np.array(ks)
            i, j = P.bi[ks], P.bj[ks]
            v += float((P.bw[ks] * (np.abs(c[i, 0] - c[j, 0]) + np.abs(c[i, 1] - c[j, 1]))).sum())
        if ps:
            ps = np.array(ps)
            b = P.pb[ps]
            v += float((P.pw[ps] * (np.abs(c[b, 0] - P.px[ps]) + np.abs(c[b, 1] - P.py[ps]))).sum())
        return v

    def soft_viol(self, blocks) -> int:
        P = self.P
        v = 0
        for b in blocks:
            code = int(P.bnd[b])
            if code:
                v += edge_gap(self.rects[b], code, self.W, self.H) > 0
        seen = set()
        for b in blocks:
            g = self.group_of.get(b)
            if g is None or id(g) in seen:
                continue
            seen.add(id(g))
            m = len(g)
            par = list(range(m))                    # union-find (scipy's graph setup was half the run time)

            def find(x):
                while par[x] != x:
                    par[x] = par[par[x]]
                    x = par[x]
                return x
            comps = m
            for a_ in range(m):
                for b_ in range(a_ + 1, m):
                    ra, rb = find(a_), find(b_)
                    if ra != rb and PT.K.touching_any(self.rects[g[a_]], self.rects[g[b_]]):
                        par[ra] = rb
                        comps -= 1
            v += comps - 1
        return v

    def pairs(self) -> List[tuple]:
        out = []
        S = self.soft
        for a_, u in enumerate(S):
            Ru = self.rects[u]
            bu = (Ru[:, 0].min(), Ru[:, 1].min(), Ru[:, 2].max(), Ru[:, 3].max())
            for v in S[a_ + 1:]:
                Rv = self.rects[v]
                if Rv[:, 0].min() > bu[2] + EPS or Rv[:, 2].max() < bu[0] - EPS or \
                        Rv[:, 1].min() > bu[3] + EPS or Rv[:, 3].max() < bu[1] - EPS:
                    continue
                if PT.K.touching_any(Ru, Rv):
                    out.append((u, v))
        return out

    def score(self, hp: float, viol: int) -> float:
        """Contest quality proxy: (1 + 0.5 * HPWL gap) * exp(2 * violations / N_soft),
        with the starting HPWL standing in for the unknown ground-truth HPWL."""
        return (0.5 + 0.5 * hp / self.hp_ref) * math.exp(2 * viol / self.P.n_soft)

    def recut(self, u, v) -> float:
        """Apply the best re-cut of the u/v pair that lowers the score; returns the score gain (0 if none)."""
        R = np.concatenate([self.rects[u], self.rects[v]])
        au = PT.area(self.rects[u])
        blocks = [u, v]
        hp0 = self.hp_local(blocks, self.c)
        sv0 = self.soft_viol(blocks)
        s0 = self.score(self.hp, self.viol)
        old = (self.rects[u], self.rects[v], self.c[u].copy(), self.c[v].copy())
        best = None
        for ax in (0, 1):
            for first_u in (True, False):
                a_lo = au if first_u else PT.area(R) - au
                lo, hi = PT.split(R, ax, a_lo)
                lo, hi = snap(lo), snap(hi)
                ru, rv = (lo, hi) if first_u else (hi, lo)
                if not len(ru) or not len(rv) or thin(ru) or thin(rv) or PT.badness_cached(ru) or PT.badness_cached(rv):
                    continue
                c = self.c
                c[u], c[v] = PT.K.centroid(ru), PT.K.centroid(rv)
                gain = hp0 - self.hp_local(blocks, c)
                self.rects[u], self.rects[v] = ru, rv
                sv = self.soft_viol(blocks)
                self.rects[u], self.rects[v] = old[0], old[1]
                c[u], c[v] = old[2], old[3]
                s1 = self.score(self.hp - gain, self.viol + sv - sv0)
                if s0 - s1 > 1e-12 and (best is None or s0 - s1 > best[0]):
                    best = (s0 - s1, ru, rv, gain, sv - sv0)
        if best is None:
            return 0.0
        ds, ru, rv, gain, dv = best
        if not (shape_ok(ru) and shape_ok(rv)):
            return 0.0
        self.rects[u], self.rects[v] = ru, rv
        self.c[u], self.c[v] = PT.K.centroid(ru), PT.K.centroid(rv)
        self.hp -= gain
        self.viol += dv
        return ds

    # -- rigid block beside a soft block ------------------------------------
    def rigid_pairs(self) -> List[tuple]:
        out = []
        for r in self.movable_rigid:
            Rr = self.rects[r]
            for v in self.soft:
                if PT.K.touching_any(Rr, self.rects[v]):
                    out.append((r, v))
        return out

    def move_rigid(self, r, v) -> float:
        """Re-place rigid block r (any rotation) at a corner of one of the
        rectangles of the r + v region; v takes the rest if it stays a valid leaf."""
        R = np.concatenate([self.rects[r], self.rects[v]])
        blocks = [r, v]
        hp0 = self.hp_local(blocks, self.c)
        sv0 = self.soft_viol(blocks)
        s0 = self.score(self.hp, self.viol)
        old = (self.rects[r], self.rects[v], self.c[r].copy(), self.c[v].copy())
        best = None
        for sh in self.shapes[r]:
            spots = {(q[0], q[1]) for q in R} | {(q[2] - sh.w, q[1]) for q in R} \
                | {(q[0], q[3] - sh.h) for q in R} | {(q[2] - sh.w, q[3] - sh.h) for q in R}
            for x, y in spots:
                x, y = round(x, 6), round(y, 6)        # position only: sizes stay exact
                rr = PT._rects(sh, x, y)
                if not PT.covered(rr, R):
                    continue
                rest = snap(PT.subtract(R, rr))
                if not len(rest) or thin(rest) or PT.badness_cached(rest):
                    continue
                c = self.c
                c[r] = (x + sh.cx, y + sh.cy)
                c[v] = PT.K.centroid(rest)
                gain = hp0 - self.hp_local(blocks, c)
                self.rects[r], self.rects[v] = rr, rest
                sv = self.soft_viol(blocks)
                self.rects[r], self.rects[v] = old[0], old[1]
                c[r], c[v] = old[2], old[3]
                s1 = self.score(self.hp - gain, self.viol + sv - sv0)
                if s0 - s1 > 1e-12 and (best is None or s0 - s1 > best[0]):
                    best = (s0 - s1, rr, rest, gain, sv - sv0, (x + sh.cx, y + sh.cy))
        if best is None:
            return 0.0
        ds, rr, rest, gain, dv, cr = best
        if not shape_ok(rest):
            return 0.0
        self.rects[r], self.rects[v] = rr, rest
        self.c[r], self.c[v] = cr, PT.K.centroid(rest)
        self.hp -= gain
        self.viol += dv
        return ds

    # -- three soft blocks ------------------------------------------------------
    def recut3(self, a, b, c3) -> float:
        """Best re-split of three touching soft blocks by two straight cuts."""
        blocks = [a, b, c3]
        R = np.concatenate([self.rects[x] for x in blocks])
        ar = {x: PT.area(self.rects[x]) for x in blocks}
        tot = PT.area(R)
        hp0 = self.hp_local(blocks, self.c)
        sv0 = self.soft_viol(blocks)
        s0 = self.score(self.hp, self.viol)
        old = {x: (self.rects[x], self.c[x].copy()) for x in blocks}
        best = None
        for first in blocks:
            rest_b = [x for x in blocks if x != first]
            for ax in (0, 1):
                for lo_first in (True, False):
                    lo, hi = PT.split(R, ax, ar[first] if lo_first else tot - ar[first])
                    lo, hi = snap(lo), snap(hi)
                    rf, rrest = (lo, hi) if lo_first else (hi, lo)
                    if not len(rf) or thin(rf) or PT.badness_cached(rf) or PT.components(rrest) != 1:
                        continue
                    p_, q_ = rest_b
                    for ax2 in (0, 1):
                        for lo_p in (True, False):
                            lo2, hi2 = PT.split(rrest, ax2, ar[p_] if lo_p else PT.area(rrest) - ar[p_])
                            lo2, hi2 = snap(lo2), snap(hi2)
                            rp, rq = (lo2, hi2) if lo_p else (hi2, lo2)
                            if not len(rp) or not len(rq) or thin(rp) or thin(rq) or PT.badness_cached(rp) or PT.badness_cached(rq):
                                continue
                            new = {first: rf, p_: rp, q_: rq}
                            for x in blocks:
                                self.rects[x] = new[x]
                                self.c[x] = PT.K.centroid(new[x])
                            gain = hp0 - self.hp_local(blocks, self.c)
                            sv = self.soft_viol(blocks)
                            for x in blocks:
                                self.rects[x], self.c[x] = old[x][0], old[x][1].copy()
                            s1 = self.score(self.hp - gain, self.viol + sv - sv0)
                            if s0 - s1 > 1e-12 and (best is None or s0 - s1 > best[0]):
                                best = (s0 - s1, new, gain, sv - sv0)
        if best is None:
            return 0.0
        ds, new, gain, dv = best
        if not all(shape_ok(new[x]) for x in blocks):
            return 0.0
        for x in blocks:
            self.rects[x] = new[x]
            self.c[x] = PT.K.centroid(new[x])
        self.hp -= gain
        self.viol += dv
        return ds

    def triples(self, pairs) -> List[tuple]:
        nb = {}
        for u, v in pairs:
            nb.setdefault(u, set()).add(v)
            nb.setdefault(v, set()).add(u)
        out = set()
        for b, ns in nb.items():
            ns = sorted(ns)
            for i in range(len(ns)):
                for j in range(i + 1, len(ns)):
                    out.add(tuple(sorted((b, ns[i], ns[j]))))
        return sorted(out)

    def kick(self, rng, k: int = 3) -> int:
        """k random valid re-cuts of random touching soft pairs, whatever their score."""
        pairs = self.pairs()
        done = 0
        for _ in range(4 * k):
            if done >= k or not pairs:
                break
            u, v = pairs[rng.randrange(len(pairs))]
            R = np.concatenate([self.rects[u], self.rects[v]])
            au = PT.area(self.rects[u])
            opts = []
            for ax in (0, 1):
                for first_u in (True, False):
                    lo, hi = PT.split(R, ax, au if first_u else PT.area(R) - au)
                    lo, hi = snap(lo), snap(hi)
                    ru, rv = (lo, hi) if first_u else (hi, lo)
                    if len(ru) and len(rv) and not thin(ru) and not thin(rv) and not PT.badness_cached(ru) \
                            and not PT.badness_cached(rv):
                        opts.append((ru, rv))
            if not opts:
                continue
            ru, rv = opts[rng.randrange(len(opts))]
            if not (shape_ok(ru) and shape_ok(rv)):
                continue
            blocks = [u, v]
            hp0, sv0 = self.hp_local(blocks, self.c), self.soft_viol(blocks)
            self.rects[u], self.rects[v] = ru, rv
            self.c[u], self.c[v] = PT.K.centroid(ru), PT.K.centroid(rv)
            self.hp += self.hp_local(blocks, self.c) - hp0
            self.viol += self.soft_viol(blocks) - sv0
            done += 1
        return done

    def _try(self, move, key) -> float:
        D = self.dlb
        if D is not None and not D.need(key):
            return 0.0
        g = move(*key)
        if D is not None:
            D.touch(key) if g > 0 else D.done(key)
        return g

    def _greedy(self, t_end: float, passes: int) -> int:
        moves = 0
        for _ in range(passes):
            gained = 0.0
            for u, v in self.pairs():
                if time.time() > t_end:
                    return moves
                g = self._try(self.recut, (u, v))
                if g > 0:
                    gained += g
                    moves += 1
            for r, v in self.rigid_pairs():
                if time.time() > t_end:
                    return moves
                g = self._try(self.move_rigid, (r, v))
                if g > 0:
                    gained += g
                    moves += 1
            if gained <= 0 and TRIPLES:
                for t in self.triples(self.pairs()):
                    if time.time() > t_end:
                        return moves
                    g = self._try(self.recut3, t)
                    if g > 0:
                        gained += g
                        moves += 1
            if gained <= 0:
                break
        return moves

    # -- targeted pushes for soft violations (10-22) -------------------------
    def _edge_gap(self, b) -> float:
        """Distance of block b from the frame edges its edge-pin code requires."""
        return edge_gap(self.rects[b], int(self.P.bnd[b]), self.W, self.H)

    def _mate_gap(self, b, main) -> float:
        """0 if b touches a member of `main`, else the bbox gap to the nearest one."""
        Rb = self.rects[b]
        best = math.inf
        for m in main:
            Rm = self.rects[m]
            if PT.K.touching_any(Rb, Rm):
                return 0.0
            dx = max(0.0, Rm[:, 0].min() - Rb[:, 2].max(), Rb[:, 0].min() - Rm[:, 2].max())
            dy = max(0.0, Rm[:, 1].min() - Rb[:, 3].max(), Rb[:, 1].min() - Rm[:, 3].max())
            best = min(best, dx + dy + 1e-3)
        return best

    def _options(self, b, n):
        """Re-shapes of b with neighbour n inside their joint region: (rects_b, rects_n)."""
        R = np.concatenate([self.rects[b], self.rects[n]])
        if b in self.shapes:                              # rigid b: corner placements, n takes the rest
            for sh in self.shapes[b]:
                spots = {(q[0], q[1]) for q in R} | {(q[2] - sh.w, q[1]) for q in R} \
                    | {(q[0], q[3] - sh.h) for q in R} | {(q[2] - sh.w, q[3] - sh.h) for q in R}
                for x, y in spots:
                    x, y = round(x, 6), round(y, 6)        # position only: sizes stay exact
                    rr = PT._rects(sh, x, y)
                    if not PT.covered(rr, R):
                        continue
                    rest = snap(PT.subtract(R, rr))
                    if len(rest) and not thin(rest) and not PT.badness_cached(rest):
                        yield rr, rest
            return
        ab = PT.area(self.rects[b])
        for ax in (0, 1):
            for first in (True, False):
                lo, hi = PT.split(R, ax, ab if first else PT.area(R) - ab)
                lo, hi = snap(lo), snap(hi)
                rb, rn = (lo, hi) if first else (hi, lo)
                if len(rb) and len(rn) and not thin(rb) and not thin(rn) and not PT.badness_cached(rb) \
                        and not PT.badness_cached(rn):
                    yield rb, rn

    def push(self, b, gap_fn, max_steps: int = 40) -> bool:
        """Walk block b toward its goal: each step re-shapes b with one soft
        neighbour so that gap_fn(b) drops most (HPWL breaks ties), whatever the
        score; stops at gap 0 or when no step helps. Returns True at gap 0."""
        for _ in range(max_steps):
            g0 = gap_fn(b)
            if g0 <= 0:
                return True
            best = None
            # soft neighbours re-cut with b; for a soft b also rigid neighbours, re-placed in
            # their joint region with b taking the rest (pushes were stuck behind rigid blocks)
            nbrs = self.soft + ([] if b in self.shapes else self.movable_rigid)
            for n in nbrs:
                if n == b or not PT.K.touching_any(self.rects[b], self.rects[n]):
                    continue
                old = (self.rects[b], self.rects[n], self.c[b].copy(), self.c[n].copy())
                hp0 = self.hp_local([b, n], self.c)
                opts = self._options(b, n) if n not in self.shapes else ((rb, rn) for rn, rb in self._options(n, b))
                for rb, rn in opts:
                    self.rects[b], self.rects[n] = rb, rn
                    self.c[b], self.c[n] = PT.K.centroid(rb), PT.K.centroid(rn)
                    g = gap_fn(b)
                    dh = self.hp_local([b, n], self.c) - hp0
                    if g < g0 - 1e-9 and (best is None or (g, dh) < best[:2]):
                        best = (g, dh, n, rb, rn)
                    self.rects[b], self.rects[n] = old[0], old[1]
                    self.c[b], self.c[n] = old[2], old[3]
            if best is None:
                return False
            _, dh, n, rb, rn = best
            if not (shape_ok(rb) and shape_ok(rn)):
                return False
            sv0 = self.soft_viol([b, n])
            self.rects[b], self.rects[n] = rb, rn
            self.c[b], self.c[n] = PT.K.centroid(rb), PT.K.centroid(rn)
            self.hp += dh
            self.viol += self.soft_viol([b, n]) - sv0
        return gap_fn(b) <= 0

    def _moves_of(self, b):
        """All re-shapes of b with one touching neighbour: (n, rects_b, rects_n)."""
        nbrs = self.soft + ([] if b in self.shapes else self.movable_rigid)
        for n in nbrs:
            if n == b or not PT.K.touching_any(self.rects[b], self.rects[n]):
                continue
            opts = self._options(b, n) if n not in self.shapes else ((rb, rn) for rn, rb in self._options(n, b))
            for rb, rn in opts:
                yield n, rb, rn

    def route_push(self, b, gap_fn, depth: int = 15, beam: int = 6, t_end: float = math.inf) -> bool:
        """Beam search for a sequence of b's re-shapes that reaches gap 0, ranked
        by (gap, added HPWL); steps may keep or raise the gap, so b can go around a
        preplaced or rigid wall (push() only takes gap-lowering steps). Applies the
        cheapest route found; returns whether one was."""
        start = (dict(self.rects), self.c.copy())
        frontier = [(gap_fn(b), 0.0, start)]
        seen = {self.rects[b].tobytes()}
        found = None
        for _ in range(depth):
            if time.time() > t_end:
                break
            children = []
            for g0, dh0, (rects, c) in frontier:
                self.rects, self.c = dict(rects), c.copy()
                for n, rb, rn in self._moves_of(b):
                    key = rb.tobytes()
                    if key in seen:
                        continue
                    seen.add(key)
                    hp0 = self.hp_local([b, n], self.c)
                    ob, on, cb, cn = self.rects[b], self.rects[n], self.c[b].copy(), self.c[n].copy()
                    self.rects[b], self.rects[n] = rb, rn
                    self.c[b], self.c[n] = PT.K.centroid(rb), PT.K.centroid(rn)
                    dh = dh0 + self.hp_local([b, n], self.c) - hp0
                    g = gap_fn(b)
                    if shape_ok(rb) and shape_ok(rn):
                        children.append((g, dh, (dict(self.rects), self.c.copy())))
                    self.rects[b], self.rects[n], self.c[b], self.c[n] = ob, on, cb, cn
            if not children:
                break
            children.sort(key=lambda t: (t[0], t[1]))
            done = [ch for ch in children if ch[0] <= 0]
            if done:
                found = min(done, key=lambda t: t[1])
                break
            frontier = children[:beam]
        self.rects, self.c = dict(start[0]), start[1].copy()
        if found is None:
            return False
        _, dh, (rects, c) = found
        self.rects, self.c = dict(rects), c.copy()
        self.hp += dh
        self.viol = self.soft_viol(range(self.P.n))
        return True

    # -- corridor re-slice: edge pins out of push range (10-23) -------------
    def _at_goal(self, R, code) -> bool:
        """R touches every edge in code (a corner when two are set)."""
        return edge_gap(R, code, self.W, self.H) <= 0

    def _band(self, b) -> list:
        """b plus the soft blocks overlapping the strip between b and its required
        edge(s), if their union is one piece (a more regular region than a chain)."""
        code, Rb = int(self.P.bnd[b]), self.rects[b]
        x0, y0, x1, y1 = Rb[:, 0].min(), Rb[:, 1].min(), Rb[:, 2].max(), Rb[:, 3].max()
        if code & 1: x0 = 0.0
        if code & 2: x1 = self.W
        if code & 8: y0 = 0.0
        if code & 4: y1 = self.H
        band = [b] + [s for s in self.soft if s != b and
                      (np.minimum(self.rects[s][:, 2], x1) - np.maximum(self.rects[s][:, 0], x0) > EPS).any() and
                      ((np.minimum(self.rects[s][:, 2], x1) - np.maximum(self.rects[s][:, 0], x0) > EPS) &
                       (np.minimum(self.rects[s][:, 3], y1) - np.maximum(self.rects[s][:, 1], y0) > EPS)).any()]
        if len(band) < 2 or len(band) > 12 or PT.components(np.concatenate([self.rects[x] for x in band])) != 1:
            return []
        return band

    def _chains(self, b, max_len: int, max_chains: int) -> List[list]:
        """Shortest chains of touching soft blocks from b to soft blocks on b's
        required edge (BFS over the soft adjacency; one chain per end block)."""
        code = int(self.P.bnd[b])
        soft = set(self.soft) - {b}
        par, frontier, ends = {b: None}, [b], []
        for _ in range(max_len):
            nxt = []
            for u in frontier:
                for s in soft:
                    if s not in par and PT.K.touching_any(self.rects[u], self.rects[s]):
                        par[s] = u
                        nxt.append(s)
                        if self._at_goal(self.rects[s], code):
                            ends.append(s)
            frontier = nxt
            if len(ends) >= max_chains or not frontier:
                break
        out = []
        for e in ends[:max_chains]:
            ch = [e]
            while par[ch[-1]] is not None:
                ch.append(par[ch[-1]])
            out.append(ch[::-1])                          # b first
        return out

    def _trees(self, U, blocks, rng, n: int, first=None):
        """Leaf regions of n random slicing trees over U for `blocks` (adaptive
        cuts, rl/partition.py evaluate_expression); `first` = (block, low side)
        puts that block alone on one side of the root cut. Valid leaves only."""
        k = len(blocks)
        areas = np.array([PT.area(self.rects[x]) for x in blocks])

        def rand_tree(ids):
            if len(ids) == 1:
                return [ids[0]]
            cut = rng.randrange(1, len(ids))
            return rand_tree(ids[:cut]) + rand_tree(ids[cut:]) + [rng.choice((PT.V, PT.H))]
        seen = set()
        for _ in range(n):
            ids = list(range(k))
            if first is not None:
                j, low = blocks.index(first[0]), first[1]
                rest = [i for i in ids if i != j]
                rng.shuffle(rest)
                sub = rand_tree(rest) if rest else []
                op = rng.choice((PT.V, PT.H))
                expr = ([j] + sub if low else sub + [j]) + ([op] if rest else [])
            else:
                rng.shuffle(ids)
                expr = rand_tree(ids)
            key = tuple(expr)
            if key in seen:
                continue
            seen.add(key)
            leaves = PT.evaluate_expression(expr, U, areas)
            out = {}
            for i, R in leaves.items():
                R = snap(R)
                if not len(R) or thin(R) or PT.badness_cached(R):
                    break
                out[blocks[i]] = R
            else:
                yield out

    def _corridor_options(self, b, chain, rng, n: int):
        """Re-partitions of the chain's merged region with b on its required edge: {block: rects}."""
        code = int(self.P.bnd[b])
        U = np.concatenate([self.rects[x] for x in chain])
        if b not in self.shapes:                          # soft b: a leaf of a tree over the chain
            for low in (True, False):
                yield from self._trees(U, chain, rng, n // 2, first=(b, low))
            yield from self._trees(U, chain, rng, n)
            return
        others = chain[1:]
        for sh in self.shapes[b]:                         # rigid b: placed on the edge inside U
            spots = set()
            if code & 3:
                x0 = 0.0 if code & 1 else self.W - sh.w
                spots |= {(x0, q[1]) for q in U} | {(x0, q[3] - sh.h) for q in U}
            if code & 12:
                y0 = 0.0 if code & 8 else self.H - sh.h
                spots |= {(q[0], y0) for q in U} | {(q[2] - sh.w, y0) for q in U}
            for x, y in spots:
                rr = PT._rects(sh, round(x, 6), round(y, 6))
                if not PT.covered(rr, U):
                    continue
                R2 = snap(PT.subtract(U, rr))
                for opt in self._trees(R2, others, rng, max(8, n // max(len(spots), 1))):
                    opt[b] = rr
                    yield opt

    def corridor(self, b, t_end: float, passes: int = 20, max_len: int = 8, max_chains: int = 12,
                 top: int = 3, trees: int = 200, seed: int = 0) -> bool:
        """Move b onto its required edge by re-slicing a chain of soft blocks
        between b and that edge (pair pushes can't cross a wide band of blocks:
        designs 50, 77, 80, 98). The `top` best re-slices by score each get a
        greedy descent; the best result is applied. Returns whether b got there."""
        import random
        rng = random.Random(seed)
        cands = []
        chains = self._chains(b, max_len, max_chains)
        band = self._band(b)
        for chain in ([band] if band else []) + chains:
            hp0, sv0 = self.hp_local(chain, self.c), self.soft_viol(chain)
            for opt in self._corridor_options(b, chain, rng, trees):
                if not all(shape_ok(r) for r in opt.values()):
                    continue
                old = {x: (self.rects[x], self.c[x].copy()) for x in chain}
                for x, r in opt.items():
                    self.rects[x], self.c[x] = r, PT.K.centroid(r)
                ok = self._edge_gap(b) <= 0
                dh = self.hp_local(chain, self.c) - hp0
                dv = self.soft_viol(chain) - sv0
                for x, (r, c) in old.items():
                    self.rects[x], self.c[x] = r, c
                if ok:
                    cands.append((self.score(self.hp + dh, self.viol + dv), id(opt), opt, dh, dv))
            if time.time() > t_end:
                break
        if not cands:
            return False
        cands.sort(key=lambda t: t[:2])
        start = (dict(self.rects), self.c.copy(), self.hp, self.viol)
        best = None
        for _, _, opt, dh, dv in cands[:top]:
            self.rects, self.c, self.hp, self.viol = dict(start[0]), start[1].copy(), start[2], start[3]
            for x, r in opt.items():
                self.rects[x], self.c[x] = r, PT.K.centroid(r)
            self.hp += dh
            self.viol += dv
            self._greedy(t_end, passes)
            sc = self.score(self.hp, self.viol)
            if best is None or sc < best[0]:
                best = (sc, dict(self.rects), self.c.copy(), self.hp, self.viol)
        _, rects, c, self.hp, self.viol = best
        self.rects, self.c = dict(rects), c.copy()
        return True

    def fix_violations(self, t_end: float, passes: int = 20) -> int:
        """For each edge-pin miss and split cluster: push the block (or the
        stray members) to the goal, descend greedily, keep it if the score drops."""
        fixed = 0
        targets = []
        for b in range(self.P.n):
            if self.P.bnd[b] and not self.P.pre[b] and self._edge_gap(b) > 0:
                targets.append(("edge", b))
        seen = set()
        for b, g in self.group_of.items():
            if id(g) in seen:
                continue
            seen.add(id(g))
            if self.soft_viol([b]) > 0:
                targets.append(("cluster", b))
        for kind, b in targets:
            if time.time() > t_end:
                break
            state = (dict(self.rects), self.c.copy(), self.hp, self.viol)
            s0 = self.score(self.hp, self.viol)
            if kind == "edge":
                ok = self.push(b, self._edge_gap) or self.route_push(b, self._edge_gap, t_end=t_end)
                if CORRIDOR:                              # pushes vs a corridor re-slice: keep the better
                    self._greedy(t_end, passes)
                    pushed = (self.score(self.hp, self.viol), dict(self.rects), self.c.copy(), self.hp, self.viol)
                    self.rects, self.c, self.hp, self.viol = dict(state[0]), state[1].copy(), state[2], state[3]
                    if not self.corridor(b, t_end, passes) or self.score(self.hp, self.viol) >= pushed[0]:
                        _, rects, c, self.hp, self.viol = pushed
                        self.rects, self.c = dict(rects), c.copy()
                if WINDOW and self._edge_gap(b) > 0:      # still off its edge: window re-partition
                    from .window import pocket_fix, window_fix
                    for fix in (window_fix, pocket_fix):  # pocket: goal corner walled off (design 50)
                        win = fix(self, b, t_end, passes=passes)
                        if win is not None and win[0] < self.score(self.hp, self.viol):
                            _, rects, c, self.hp, self.viol = win
                            self.rects, self.c = dict(rects), c.copy()
            else:
                g = self.group_of[b]
                comps = self._components(g)
                main = max(comps, key=lambda c: sum(PT.area(self.rects[m]) for m in c))
                ok = True
                for comp in comps:
                    if comp is main:
                        continue
                    for m in comp:
                        if m in self.soft or m in self.shapes:
                            gap = lambda x, main=main: self._mate_gap(x, main)
                            ok &= self.push(m, gap) or self.route_push(m, gap, t_end=t_end)
            self._greedy(t_end, passes)
            if self.score(self.hp, self.viol) < s0 - 1e-12:
                fixed += 1
            else:
                self.rects, self.c, self.hp, self.viol = dict(state[0]), state[1].copy(), state[2], state[3]
            if kind == "cluster" and WINDOW and self.soft_viol([b]) > 0 and time.time() < t_end:
                from .window import cluster_shift         # still split: move the stray piece (design 8)
                win = cluster_shift(self, self.group_of[b], t_end, passes=passes)
                if win is not None and win[0] < self.score(self.hp, self.viol) - 1e-12:
                    _, rects, c, self.hp, self.viol = win
                    self.rects, self.c = dict(rects), c.copy()
                    fixed += 1
        return fixed

    def _components(self, g):
        m = len(g)
        par = list(range(m))

        def find(x):
            while par[x] != x:
                par[x] = par[par[x]]
                x = par[x]
            return x
        for a_ in range(m):
            for b_ in range(a_ + 1, m):
                if PT.K.touching_any(self.rects[g[a_]], self.rects[g[b_]]):
                    par[find(a_)] = find(b_)
        out = {}
        for k in range(m):
            out.setdefault(find(k), []).append(g[k])
        return list(out.values())

    def run(self, time_budget: float = 30.0, passes: int = 20, ils: bool = True, seed: int = 0) -> dict:
        """Greedy descent, then (ils) iterated local search until the budget:
        kick with a few random re-cuts, descend again, keep only improvements."""
        import random
        t0 = time.time()
        t_end = t0 + time_budget
        self.hp = self.hp_ref = max(self.P.hpwl(self.c[:, 0], self.c[:, 1]), 1e-9)
        self.viol = self.soft_viol(range(self.P.n))
        moves = self._greedy(t_end, passes)
        if PUSH and self.viol:
            moves += self.fix_violations(t_end, passes)
        if LNS > 0:
            from .window import lns
            lns(self, time.time() + LNS * max(0.0, t_end - time.time()), passes=passes, seed=seed)
        kicks = wins = 0
        if ils:
            rng = random.Random(seed)
            best = (self.score(self.hp, self.viol), dict(self.rects), self.c.copy(), self.hp, self.viol)
            while time.time() < t_end:
                self.kick(rng, k=rng.choice((2, 3, 4)))
                moves += self._greedy(t_end, passes)
                kicks += 1
                sc = self.score(self.hp, self.viol)
                if sc < best[0] - 1e-12:
                    best = (sc, dict(self.rects), self.c.copy(), self.hp, self.viol)
                    wins += 1
                else:
                    _, rects, c, self.hp, self.viol = best
                    self.rects, self.c = dict(rects), c.copy()
        return dict(moves=moves, kicks=kicks, wins=wins, seconds=time.time() - t0, hpwl=self.hp, viol=self.viol,
                    score=self.score(self.hp, self.viol))

    def run_greedy_only(self, time_budget: float = 30.0, passes: int = 20) -> dict:
        t0 = time.time()
        self.hp = self.hp_ref = max(self.P.hpwl(self.c[:, 0], self.c[:, 1]), 1e-9)
        self.viol = self.soft_viol(range(self.P.n))
        total, moves = 0.0, 0
        for _ in range(passes):
            gained = 0.0
            pairs = self.pairs()
            for u, v in pairs:
                if time.time() - t0 > time_budget:
                    break
                g = self.recut(u, v)
                if g > 0:
                    gained += g
                    moves += 1
            for r, v in self.rigid_pairs():
                if time.time() - t0 > time_budget:
                    break
                g = self.move_rigid(r, v)
                if g > 0:
                    gained += g
                    moves += 1
            if gained <= 0 and TRIPLES:          # pairs converged: try three-block re-splits
                for t in self.triples(self.pairs()):
                    if time.time() - t0 > time_budget:
                        break
                    g = self.recut3(*t)
                    if g > 0:
                        gained += g
                        moves += 1
            total += gained
            if gained <= 0 or time.time() - t0 > time_budget:
                break
        return dict(gain=total, moves=moves, seconds=time.time() - t0, hpwl=self.hp, viol=self.viol,
                    score=self.score(self.hp, self.viol))

    def layout(self) -> List[list]:
        """Polygons with every x (and y) coordinate within MERGE of another set to one
        value across all blocks: 1e-6 jogs added a corner (design 86) and a vertex that
        failed a fixed shape check (design 77) on 10-21. When merging breaks an edge-pin or
        cluster contact (designs 25, 33, 62 on 10-24), the unmerged polygons if they pass."""
        out = self._layout(merge=True)
        n = range(self.P.n)
        v = Refiner(self.P, out).soft_viol(n)
        if v <= self.soft_viol(n):
            return out
        alt = self._layout(merge=False)
        return alt if layout_ok(self.P, alt) and Refiner(self.P, alt).soft_viol(n) < v else out

    def _layout(self, merge: bool) -> List[list]:
        if not merge:
            return self._polygons([PT.to_polygon(self.rects[i]) for i in range(self.P.n)])
        allr = np.concatenate([self.rects[i] for i in range(self.P.n)])
        rig = np.concatenate([np.full(len(self.rects[i]), bool(self.P.rigid[i])) for i in range(self.P.n)])
        maps = []
        for cols in ((0, 2), (1, 3)):
            vals = np.concatenate([allr[:, c] for c in cols])
            isr = np.concatenate([rig, rig])
            order = np.lexsort((~isr, vals))
            v, r_ = vals[order], isr[order]
            uniq, first = np.unique(v, return_index=True)
            ur = np.array([r_[first[k]:(first[k + 1] if k + 1 < len(first) else len(v))].any() for k in range(len(uniq))])
            rep = uniq.copy()
            k = 0
            while k < len(uniq):                       # runs of values within MERGE share one value,
                j = k                                  # a rigid block's own value when there is one
                while j + 1 < len(uniq) and uniq[j + 1] - uniq[k] <= MERGE:
                    j += 1
                pick = next((t for t in range(k, j + 1) if ur[t]), k)
                rep[k:j + 1] = uniq[pick]
                k = j + 1
            maps.append((uniq, rep))
        gs, raw = [], set()
        for i in range(self.P.n):
            R = self.rects[i].copy()
            for (v, rep), cols in zip(maps, ((0, 2), (1, 3))):
                for c in cols:
                    R[:, c] = rep[np.searchsorted(v, R[:, c])]
            R = R[(R[:, 2] - R[:, 0] > 1e-9) & (R[:, 3] - R[:, 1] > 1e-9)]
            g = PT.to_polygon(R) if len(R) else None
            if g is None or g.geom_type != "Polygon":      # merging closed a sliver: keep it unmerged
                g = PT.to_polygon(self.rects[i])
                raw.add(i)
            gs.append(g)
        # a merged neighbour can overlap an unmerged block by up to MERGE (10-24, design 31:
        # rigid 17 over block 41's 4e-6 strip): keep those unmerged too, until none overlaps
        todo = list(raw)
        while todo:
            i = todo.pop()
            for j in range(self.P.n):
                if j not in raw and gs[j].intersects(gs[i]) and gs[j].intersection(gs[i]).area > 1e-12:
                    gs[j] = PT.to_polygon(self.rects[j])
                    raw.add(j)
                    todo.append(j)
        return self._polygons(gs)

    def _polygons(self, gs) -> List[list]:
        out = []
        for g in gs:
            pts = strip_collinear(g.exterior.coords[:-1])
            h = Polygon(pts)
            if not h.is_valid or abs(h.area - g.area) > 1e-6 * max(g.area, 1.0):
                pts = [list(p) for p in g.exterior.coords[:-1]]   # self-touching ring (design 80): keep as is
            out.append(pts)
        return out


def shape_ok(R: np.ndarray) -> bool:
    """The block's polygon as the evaluator will read it (cleaned vertices) is valid
    and has the rectangles' area. A thin strip closing a ring around another block
    gave a self-touching ring that neither hole count saw (design 80, 10-23)."""
    from . import polygon as G
    a = PT.area(R)
    for Q in (R, np.round(R / MERGE) * MERGE):     # also as written: merging can close a 1e-5 gap
        Q = Q[(Q[:, 2] - Q[:, 0] > 1e-9) & (Q[:, 3] - Q[:, 1] > 1e-9)]
        g = PT.to_polygon(Q)
        if g.geom_type != "Polygon" or len(g.interiors):
            return False
        try:
            h = Polygon(G.clean_vertices(list(g.exterior.coords[:-1])))
        except Exception:
            return False
        if not h.is_valid or abs(h.area - a) > 1e-4 * max(a, 1.0):
            return False
    return True


def thin(R: np.ndarray, w: float = 1e-3) -> bool:
    """A piece narrower than w (a cut next to an existing edge leaves slivers that
    collapse when coordinates are merged: design 98, 10-22)."""
    return bool(((R[:, 2] - R[:, 0]) < w).any() or ((R[:, 3] - R[:, 1]) < w).any())


def strip_collinear(coords) -> list:
    """Drop vertices that lie on a straight axis-aligned run between their
    neighbours (shapely's simplify(0) can keep them: design 86 block 33 read as
    17 corners with 16 real ones, 10-21)."""
    pts = [tuple(p) for p in coords]
    changed = True
    while changed and len(pts) > 4:
        changed = False
        for k in range(len(pts)):
            a, b, c = pts[k - 1], pts[k], pts[(k + 1) % len(pts)]
            if (a[0] == b[0] == c[0]) or (a[1] == b[1] == c[1]) or a == b:
                del pts[k]
                changed = True
                break
    return [list(p) for p in pts]


def layout_ok(P: PT.Problem, layout) -> bool:
    """The evaluator's hard checks that refinement can break (prime_evaluate.evaluate):
    valid single polygons, soft areas within 1 %, soft corners within the cap, fixed shapes."""
    from . import polygon as G
    for i, v in enumerate(layout):
        try:
            g = Polygon(G.clean_vertices(v))
        except Exception:
            return False
        if not g.is_valid or g.area <= 1e-6:
            return False
        if P.pre[i]:
            continue
        if P.fixed[i]:
            if not G.same_shape(g, Polygon(P.targets[i])):
                return False
            continue
        if abs(g.area - P.area[i]) > 0.01 * P.area[i]:
            return False
        if len(g.simplify(0).exterior.coords) - 1 > PT.MAX_CORNERS:
            return False
    return True

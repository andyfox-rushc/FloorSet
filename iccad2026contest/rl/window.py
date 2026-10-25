"""
Window re-partition (10-24): the full partitioner (rl/partition.py: rigid
leaves in corners, pocket cuts, Wong-Liu annealing) run on the union of a
window's blocks while every other block stays where it is (WindowPartitioner).
Moves built on it, applied to a valid layout held by an rl/refine.py Refiner:

  window_fix     an edge-pinned block pushes and the corridor can't move: the
                 strip from it to its edge, widened (design 80)
  pocket_fix     its goal corner lies in a pocket walled off by preplaced
                 blocks: the pocket gets the block plus one partner of the
                 right area, the blocks it held move out (design 50)
  cluster_shift  a split cluster's stray piece moved as a unit until it
                 touches the main piece, the window around it re-partitioned
                 (design 8)
  lns            large-neighbourhood search for wirelength: random windows of
                 6-20 blocks re-annealed, kept when the score drops (brought 14
                 test designs from above the ground truth's cost down to 1.000)

All keep a result only if the Refiner's score (HPWL against the starting
layout, soft violations) drops; none uses the ground truth.
"""

import math
import time

import numpy as np
from shapely.ops import unary_union

from . import partition as PT
from . import polygon as G
from .refine import DontLook, edge_gap


class WindowPartitioner(PT.Partitioner):
    """Partitioner over the union of blocks S; the other blocks are fixed context."""

    def __init__(self, P: PT.Problem, Rf, S, must=None, must_w: float = 0.1, preset=None, region=None,
                 penalty=None, shaped: float = 0.0):
        """preset {block: rects}: blocks of S placed up front (fixed for the search);
        region: the rectangles the tree divides (default: the union of S);
        penalty(rects) -> search-only cost (default: block `must`'s edge gap)."""
        super().__init__(P, Rf.c.copy())
        preset = preset or {}
        self.must, self.must_w = must, must_w          # search-only penalty: block `must` off its edge
        self.penalty = penalty
        self.shaped = shaped                            # search-only weight on _gaps
        S = sorted(int(x) for x in S)
        Sset = set(S)
        self.W, self.H = Rf.W, Rf.H
        self.pre = {}                                   # unused by the overridden cost
        self.operands = [i for i in S if i not in preset]
        self.options = {i: Rf.shapes[i] for i in self.operands if i in Rf.shapes}
        self.pocket_rigid, self.pockets, self.memo = {}, {}, {}
        self.region = region if region is not None else \
            PT.decompose_geom(unary_union([PT.to_polygon(Rf.rects[i]) for i in S]))
        self.ref = Rf.hp_ref
        self.mates = {b: [x for x in m if x in Sset] for b, m in self.mates.items() if b in Sset}
        self.mates = {b: m for b, m in self.mates.items() if m}
        self.base_c = Rf.c.copy()
        self.base_rects = dict(Rf.rects)
        self.S = S
        # violations outside the window stay as they are
        self.bnd_in = [int(i) for i in np.where(P.bnd != 0)[0] if int(i) in Sset]
        self.groups_in = [g for g in P.groups if Sset.intersection(int(x) for x in g)]
        self.v_out = 0
        self.v_out = Rf.soft_viol(range(P.n)) - self._viol(Rf.rects)
        for b, R in preset.items():
            self.base_rects[b], self.base_c[b] = R, PT.K.centroid(R)
        self.mates = {b: m for b, m in self.mates.items() if b not in preset}

    def _viol(self, rects) -> int:
        """Edge-pin misses of the window's pinned blocks + split clusters touching the window (+ v_out)."""
        P = self.P
        v = self.v_out
        for i in self.bnd_in:
            v += edge_gap(rects[i], int(P.bnd[i]), self.W, self.H) > 0
        for g in self.groups_in:
            m = len(g)
            par = list(range(m))

            def find(x):
                while par[x] != x:
                    par[x] = par[par[x]]
                    x = par[x]
                return x
            comps = m
            for a_ in range(m):
                for b_ in range(a_ + 1, m):
                    ra, rb = find(a_), find(b_)
                    if ra != rb and PT.K.touching_any(rects[int(g[a_])], rects[int(g[b_])]):
                        par[ra] = rb
                        comps -= 1
            v += comps - 1
        return v

    def _gaps(self, rects) -> float:
        """Edge gaps of the window's pinned blocks + stray-piece gaps of its split
        clusters, over the frame size: a slope toward meeting all of them at once
        (design 8: block 5 needs the right edge and its cluster mate 12)."""
        P = self.P
        tot = 0.0
        for i in self.bnd_in:
            tot += edge_gap(rects[i], int(P.bnd[i]), self.W, self.H)
        for g in self.groups_in:
            tot += _stray_gap([int(x) for x in g], rects)
        return tot / max(self.W, self.H)

    def placed(self, leaves, rigid):
        out = {}
        for b, R in leaves.items():
            out[b] = R
        for b, (s, x, y) in rigid.items():
            out[b] = PT._rects(s, x, y)
        return out

    def cost(self, expr):
        P = self.P
        leaves, rigid, bad = self.evaluate(expr)
        cx, cy = self.base_c[:, 0].copy(), self.base_c[:, 1].copy()
        rects = dict(self.base_rects)
        for b, (s, x, y) in rigid.items():
            cx[b], cy[b] = x + s.cx, y + s.cy
            rects[b] = PT._rects(s, x, y)
        for b, R in leaves.items():
            cx[b], cy[b] = PT.K.centroid(R)
            bad += PT.badness_cached(R)
            rects[b] = R
        v = self._viol(rects)
        hp = P.hpwl(cx, cy)
        c = (1 + 0.5 * (hp / self.ref + 1)) * math.exp(2 * v / P.n_soft) + PT.BAD_W * bad
        self._cxy = (cx, cy)
        search = c
        if self.shaped > 0:
            search += self.shaped * self._gaps(rects)
        if self.penalty is not None:
            search += self.penalty(rects)
        elif self.must is not None:
            gap = edge_gap(rects[self.must], int(P.bnd[self.must]), self.W, self.H)
            search += self.must_w * gap / max(self.W, self.H)
        return search, (leaves, rigid), bad, c


def window_sets(Rf, b, factors=(0.25, 0.5, 1.0, 2.0, 3.0, 4.0), max_blocks: int = 45):
    """Block sets of growing windows: the strip from b's near side to its required
    edge(s), widened by f * sqrt(area b) and reaching that far back into b; only
    the blocks in b's piece of the union (a preplaced block can split it)."""
    P = Rf.P
    code, Rb = int(P.bnd[b]), Rf.rects[b]
    bx0, by0, bx1, by1 = Rb[:, 0].min(), Rb[:, 1].min(), Rb[:, 2].max(), Rb[:, 3].max()
    s = math.sqrt(float(P.area[b]))
    seen, out = set(), []
    movable = [i for i in range(P.n) if not P.pre[i]]
    for f in factors:
        m = f * s
        x0, y0, x1, y1 = bx0 - m, by0 - m, bx1 + m, by1 + m
        if code & 1: x0, x1 = 0.0, min(bx1, bx0 + m)
        if code & 2: x0, x1 = max(bx0, bx1 - m), Rf.W
        if code & 8: y0, y1 = 0.0, min(by1, by0 + m)
        if code & 4: y0, y1 = max(by0, by1 - m), Rf.H
        x0, y0, x1, y1 = max(0.0, x0), max(0.0, y0), min(Rf.W, x1), min(Rf.H, y1)
        S = []
        for i in movable:
            R = Rf.rects[i]
            ov = (np.clip(np.minimum(R[:, 2], x1) - np.maximum(R[:, 0], x0), 0, None)
                  * np.clip(np.minimum(R[:, 3], y1) - np.maximum(R[:, 1], y0), 0, None)).sum()
            if ov > 1e-9 or i == b:
                S.append(i)
        S = _piece_of(Rf, S, b)
        key = tuple(S)
        if key in seen or len(S) < 2 or len(S) > max_blocks:
            continue
        seen.add(key)
        out.append(S)
    return out


def _piece_of(Rf, S, b):
    """Blocks of S connected to b by touching (within S)."""
    keep, frontier = {b}, [b]
    rest = set(S) - {b}
    while frontier:
        u = frontier.pop()
        for v in list(rest):
            if PT.K.touching_any(Rf.rects[u], Rf.rects[v]):
                rest.discard(v)
                keep.add(v)
                frontier.append(v)
    return sorted(keep)


def window_fix(Rf, b, t_end: float, anneal_s: float = 8.0, seeds=(0, 1, 2, 3), passes: int = 20, verbose=False):
    """Best (score, state) over windows x anneal seeds with b on its edge, after a
    greedy descent; None if none. Leaves Rf in its starting state."""
    from .refine import snap, shape_ok, thin
    start = (dict(Rf.rects), Rf.c.copy(), Rf.hp, Rf.viol)
    best = None
    for S in window_sets(Rf, b):
        if time.time() > t_end:
            break
        WP = WindowPartitioner(Rf.P, Rf, S, must=b)
        x0, y0 = WP.region[:, 0].min(), WP.region[:, 1].min()
        x1, y1 = WP.region[:, 2].max(), WP.region[:, 3].max()
        expr = PT.initial_expression(S, WP.seeds, Rf.P.area, (x0, y0, x1, y1), Rf.P.groups)
        for sd in seeds:
            if time.time() > t_end:
                break
            res = WP.anneal(expr, min(anneal_s, max(0.5, t_end - time.time())), sd)
            if res is None:
                if verbose:
                    print(f"    window {len(S)} blocks seed {sd}: no valid partition", flush=True)
                continue
            new = {k: snap(R) for k, R in WP.placed(*res[2]).items()}
            if not all(len(R) and not thin(R) and shape_ok(R) for R in new.values()):
                continue
            Rf.rects, Rf.c = dict(start[0]), start[1].copy()
            for k, R in new.items():
                Rf.rects[k], Rf.c[k] = R, PT.K.centroid(R)
            if Rf._edge_gap(b) > 0:
                if verbose:
                    print(f"    window {len(S)} blocks seed {sd}: valid, b not on its edge", flush=True)
                Rf.rects, Rf.c = dict(start[0]), start[1].copy()
                continue
            Rf.hp = Rf.P.hpwl(Rf.c[:, 0], Rf.c[:, 1])
            Rf.viol = Rf.soft_viol(range(Rf.P.n))
            s0 = Rf.score(Rf.hp, Rf.viol)
            Rf._greedy(t_end, passes)
            sc = Rf.score(Rf.hp, Rf.viol)
            if verbose:
                print(f"    window {len(S)} blocks seed {sd}: score {s0:.4f} -> {sc:.4f} "
                      f"(start {Rf.score(start[2], start[3]):.4f}, viol {Rf.viol})", flush=True)
            if best is None or sc < best[0]:
                best = (sc, dict(Rf.rects), Rf.c.copy(), Rf.hp, Rf.viol)
            Rf.rects, Rf.c, Rf.hp, Rf.viol = dict(start[0]), start[1].copy(), start[2], start[3]
    Rf.rects, Rf.c, Rf.hp, Rf.viol = dict(start[0]), start[1].copy(), start[2], start[3]
    return best


# -- pocket swap: the goal corner is in a pocket walled off by preplaced blocks ----
def _free_pieces(Rf):
    """The frame minus the preplaced blocks: (rects, component label per rect)."""
    from shapely.geometry import box
    P = Rf.P
    pre = [PT.to_polygon(Rf.rects[i]) for i in range(P.n) if P.pre[i]]
    R = PT.decompose_geom(box(0, 0, Rf.W, Rf.H).difference(unary_union(pre)) if pre else box(0, 0, Rf.W, Rf.H))
    return R, PT.K.component_labels(R)


def _goal_point(Rf, code):
    if bool(code & 1) == bool(code & 2) or bool(code & 4) == bool(code & 8):
        return None                                   # one edge: no single goal point
    return (0.0 if code & 1 else Rf.W, 0.0 if code & 8 else Rf.H)


def pocket_fills(Rf, b, C, D, tol: float = 0.0099, max_opts: int = 6):
    """Ways to fill pocket C (rects) with b on its goal corner plus one partner
    (soft or rigid) from outside the pocket, areas within the 1 % tolerance:
    [(partner, {b: rects, partner: rects})], partner None when b fills it alone."""
    from .refine import snap, shape_ok, thin
    P = Rf.P
    A = PT.area(C)
    code = int(P.bnd[b])
    ok_b = lambda R: (len(R) and not thin(R) and not PT.badness_cached(R) and shape_ok(R)
                      and Rf._at_goal(R, code) and abs(PT.area(R) - P.area[b]) <= tol * P.area[b])
    out = []
    if ok_b(C):
        out.append((None, {b: C}))
    for r in range(P.n):
        if r == b or P.pre[r] or r in D:
            continue
        rest = A - P.area[b]
        if not abs(P.area[r] - rest) <= tol * max(P.area[r], P.area[b]) + 1e-9:
            continue
        if r in Rf.shapes:                            # rigid partner: a corner spot in C, b gets the rest
            for sh in Rf.shapes[r]:
                spots = {(q[0], q[1]) for q in C} | {(q[2] - sh.w, q[1]) for q in C} \
                    | {(q[0], q[3] - sh.h) for q in C} | {(q[2] - sh.w, q[3] - sh.h) for q in C}
                for x, y in sorted(spots):
                    rr = PT._rects(sh, round(x, 6), round(y, 6))
                    if not PT.covered(rr, C):
                        continue
                    Rb = snap(PT.subtract(C, rr))
                    if ok_b(Rb):
                        out.append((r, {b: Rb, r: rr}))
        elif not P.rigid[r]:                          # soft partner: one straight cut
            for ax in (0, 1):
                for first in (True, False):
                    lo, hi = PT.split(C, ax, P.area[b] / (P.area[b] + P.area[r]) * A if first
                                      else P.area[r] / (P.area[b] + P.area[r]) * A)
                    lo, hi = snap(lo), snap(hi)
                    Rb, Rr = (lo, hi) if first else (hi, lo)
                    if ok_b(Rb) and len(Rr) and not thin(Rr) and not PT.badness_cached(Rr) and shape_ok(Rr):
                        out.append((r, {b: Rb, r: Rr}))
    # nearest partners first (they leave a hole near the pocket's displaced blocks)
    gx, gy = PT.K.centroid(C)
    out.sort(key=lambda t: 0.0 if t[0] is None else abs(Rf.c[t[0], 0] - gx) + abs(Rf.c[t[0], 1] - gy))
    return out[:max_opts]


def pocket_fix(Rf, b, t_end: float, anneal_s: float = 8.0, seeds=tuple(range(12)), factors=(2.0,),
               max_blocks: int = 45, passes: int = 20, verbose=False):
    """b's goal corner lies in a pocket walled off by preplaced blocks (design 50:
    block 9, top-right corner above preplaced 41). Fill the pocket with b plus a
    partner of the right area; the blocks it held (D) go into a window of the main
    region around b's and the partner's old spots, re-partitioned by the annealer.
    Returns the first success with fewer violations and a lower score, else the best; as
    window_fix. Design 50 met its pin in 4 of 72 anneals at factor 2.0 and 0 of 101 at
    3.0-4.0 (10-24), so all seeds go there."""
    from .refine import snap, shape_ok, thin
    P = Rf.P
    g = _goal_point(Rf, int(P.bnd[b]))
    if g is None:
        return None
    R, lab = _free_pieces(Rf)
    hit = [k for k, q in enumerate(R) if q[0] - 1e-6 <= g[0] <= q[2] + 1e-6 and q[1] - 1e-6 <= g[1] <= q[3] + 1e-6]
    if not hit:
        return None
    C = snap(R[lab == lab[hit[0]]])
    if PT.covered(Rf.rects[b], C):
        return None                                   # b is in that piece already: not a pocket case
    D = [i for i in range(P.n) if not P.pre[i] and PT.covered(Rf.rects[i], C)]
    if abs(sum(PT.area(Rf.rects[i]) for i in D) - PT.area(C)) > 1e-6 * PT.area(C):
        return None                                   # the piece holds part of a block: not walled off
    fills = pocket_fills(Rf, b, C, D)
    if verbose:
        print(f"    pocket {PT.area(C):.0f} holds {D}; fills with partners {[f[0] for f in fills]}", flush=True)
    start = (dict(Rf.rects), Rf.c.copy(), Rf.hp, Rf.viol)
    best = None
    s = math.sqrt(float(P.area[b]))
    movable = [i for i in range(P.n) if not P.pre[i] and i not in D]
    for partner, preset in fills:
        moved = [b] + ([partner] if partner is not None else [])
        bx = np.concatenate([Rf.rects[x] for x in moved])
        seen = set()
        for f in factors:
            if time.time() > t_end:
                break
            m = f * s
            x0, y0 = max(0.0, bx[:, 0].min() - m), max(0.0, bx[:, 1].min() - m)
            x1, y1 = min(Rf.W, bx[:, 2].max() + m), min(Rf.H, bx[:, 3].max() + m)
            S = [i for i in movable if i in moved or (
                np.clip(np.minimum(Rf.rects[i][:, 2], x1) - np.maximum(Rf.rects[i][:, 0], x0), 0, None)
                * np.clip(np.minimum(Rf.rects[i][:, 3], y1) - np.maximum(Rf.rects[i][:, 1], y0), 0, None)).sum() > 1e-9]
            S = _piece_of(Rf, S, b)
            if partner is not None and partner not in S:
                continue
            key = tuple(S)
            if key in seen or len(S) + len(D) > max_blocks:
                continue
            seen.add(key)
            region = PT.decompose_geom(unary_union([PT.to_polygon(Rf.rects[i]) for i in S]))
            WP = WindowPartitioner(P, Rf, S + D, preset=preset, region=region)
            ops = WP.operands
            x0r, y0r = region[:, 0].min(), region[:, 1].min()
            x1r, y1r = region[:, 2].max(), region[:, 3].max()
            expr = PT.initial_expression(ops, WP.seeds, P.area, (x0r, y0r, x1r, y1r), P.groups)
            for sd in seeds:
                if time.time() > t_end:
                    break
                res = WP.anneal(expr, min(anneal_s, max(0.5, t_end - time.time())), sd)
                if res is None:
                    if verbose:
                        print(f"    pocket partner {partner} window {len(S)}+{len(D)} seed {sd}: no valid partition",
                              flush=True)
                    continue
                new = {k: snap(Rq) for k, Rq in WP.placed(*res[2]).items()}
                new.update(preset)
                if not all(len(Rq) and not thin(Rq) and shape_ok(Rq) for Rq in new.values()):
                    continue
                Rf.rects, Rf.c = dict(start[0]), start[1].copy()
                for k, Rq in new.items():
                    Rf.rects[k], Rf.c[k] = Rq, PT.K.centroid(Rq)
                Rf.hp = P.hpwl(Rf.c[:, 0], Rf.c[:, 1])
                Rf.viol = Rf.soft_viol(range(P.n))
                s0 = Rf.score(Rf.hp, Rf.viol)
                Rf._greedy(t_end, passes)
                sc = Rf.score(Rf.hp, Rf.viol)
                if verbose:
                    print(f"    pocket partner {partner} window {len(S)}+{len(D)} seed {sd}: score {s0:.4f} -> "
                          f"{sc:.4f} (start {Rf.score(start[2], start[3]):.4f}, viol {Rf.viol})", flush=True)
                if Rf._edge_gap(b) <= 0 and (best is None or sc < best[0]):
                    best = (sc, dict(Rf.rects), Rf.c.copy(), Rf.hp, Rf.viol)
                Rf.rects, Rf.c, Rf.hp, Rf.viol = dict(start[0]), start[1].copy(), start[2], start[3]
                if best is not None and best[4] < start[3] and best[0] < Rf.score(start[2], start[3]):
                    return best                       # stop at the first with fewer violations and a lower score
    Rf.rects, Rf.c, Rf.hp, Rf.viol = dict(start[0]), start[1].copy(), start[2], start[3]
    return best


# -- split clusters (designs 8, 76: zero-white-space layouts lost to one split cluster) --
def _gap(A, B) -> float:
    """0 if rect sets A and B touch, else their bbox gap (dx + dy) plus a little."""
    if PT.K.touching_any(A, B):
        return 0.0
    dx = max(0.0, B[:, 0].min() - A[:, 2].max(), A[:, 0].min() - B[:, 2].max())
    dy = max(0.0, B[:, 1].min() - A[:, 3].max(), A[:, 1].min() - B[:, 3].max())
    return dx + dy + 1e-3


def _stray_gap(g, rects) -> float:
    """Sum of bbox gaps from each stray piece of cluster g to its largest piece."""
    m = len(g)
    par = list(range(m))

    def find(x):
        while par[x] != x:
            par[x] = par[par[x]]
            x = par[x]
        return x
    for a_ in range(m):
        for b_ in range(a_ + 1, m):
            if PT.K.touching_any(rects[g[a_]], rects[g[b_]]):
                par[find(a_)] = find(b_)
    comps = {}
    for k in range(m):
        comps.setdefault(find(k), []).append(g[k])
    if len(comps) == 1:
        return 0.0
    pieces = sorted(comps.values(), key=lambda c: -sum(PT.area(rects[x]) for x in c))
    main = np.concatenate([rects[x] for x in pieces[0]])
    return sum(_gap(np.concatenate([rects[x] for x in c]), main) for c in pieces[1:])


def _shifts(Rf, piece, main, pins_ok):
    """Translations (dx, dy) of the rects of `piece` that make it touch `main`
    (each main rect, each side, a few alignments), inside the frame, off the
    preplaced blocks and with pins_ok(translated) -- nearest first."""
    P = Rf.P
    Rp = np.concatenate([Rf.rects[x] for x in piece])
    px0, py0, px1, py1 = Rp[:, 0].min(), Rp[:, 1].min(), Rp[:, 2].max(), Rp[:, 3].max()
    pre = [Rf.rects[i] for i in range(P.n) if P.pre[i] and i not in piece]
    pre = np.concatenate(pre) if pre else np.zeros((0, 4))
    out = set()
    for x in main:
        for q in Rf.rects[x]:
            xs = {0.0, q[0] - px0, q[2] - px1, (q[0] + q[2] - px0 - px1) / 2}
            ys = {0.0, q[1] - py0, q[3] - py1, (q[1] + q[3] - py0 - py1) / 2}
            out |= {(dx, q[1] - py1) for dx in xs} | {(dx, q[3] - py0) for dx in xs}
            out |= {(q[0] - px1, dy) for dy in ys} | {(q[2] - px0, dy) for dy in ys}
    Rm = np.concatenate([Rf.rects[x] for x in main])
    ok = []
    for dx, dy in out:
        dx, dy = round(float(dx), 6), round(float(dy), 6)
        if px0 + dx < -1e-9 or py0 + dy < -1e-9 or px1 + dx > Rf.W + 1e-9 or py1 + dy > Rf.H + 1e-9:
            continue
        T = {x: Rf.rects[x] + np.array([dx, dy, dx, dy]) for x in piece}
        Rt = np.concatenate(list(T.values()))
        if len(pre) and (np.clip(np.minimum(Rt[:, None, 2], pre[None, :, 2]) - np.maximum(Rt[:, None, 0], pre[None, :, 0]), 0, None)
                         * np.clip(np.minimum(Rt[:, None, 3], pre[None, :, 3]) - np.maximum(Rt[:, None, 1], pre[None, :, 1]), 0, None)).sum() > 1e-9:
            continue
        if not PT.K.touching_any(Rt, Rm) or not pins_ok(T):
            continue
        ok.append((abs(dx) + abs(dy), dx, dy, T))
    ok.sort(key=lambda t: t[:3])
    return ok


def cluster_shift(Rf, g, t_end: float, anneal_s: float = 8.0, seeds=(0, 1, 2), factors=(0.5, 1.0, 2.0),
                  max_shifts: int = 6, max_blocks: int = 45, w: float = 0.5, passes: int = 20, verbose=False):
    """Move each stray piece of split cluster g as a unit until it touches the main
    piece (design 8: rigid MIB pair 4+5 below preplaced 12, block 5 pinned to the
    right edge -- the tree's corner placement never found that spot), fix it there
    and re-partition the window around its old and new place. Same return as window_fix."""
    from .refine import snap, shape_ok, thin
    P = Rf.P
    g = [int(x) for x in g]
    comps = Rf._components(g)
    if len(comps) < 2:
        return None
    main = max(comps, key=lambda c: (any(P.pre[x] for x in c), sum(PT.area(Rf.rects[x]) for x in c)))
    strays = [c for c in comps if c is not main and not any(P.pre[x] for x in c)]
    if not strays:
        return None
    piece = min(strays, key=lambda c: sum(PT.area(Rf.rects[x]) for x in c))
    pinned = [x for x in piece if P.bnd[x] and Rf._at_goal(Rf.rects[x], int(P.bnd[x]))]
    pins_ok = lambda T: all(Rf._at_goal(T[x], int(P.bnd[x])) for x in pinned)
    shifts = _shifts(Rf, piece, main, pins_ok)[:max_shifts]
    if verbose:
        print(f"    cluster {g}: piece {piece} -> shifts {[(round(dx, 1), round(dy, 1)) for _, dx, dy, _ in shifts]}",
              flush=True)
    movable = [i for i in range(P.n) if not P.pre[i]]
    start = (dict(Rf.rects), Rf.c.copy(), Rf.hp, Rf.viol)
    best = None
    s = math.sqrt(sum(P.area[x] for x in piece))
    for _, dx, dy, T in shifts:
        Rall = np.concatenate([Rf.rects[x] for x in piece] + list(T.values()))
        seen = set()
        for f in factors:
            if time.time() > t_end:
                break
            m = f * s
            x0, y0 = max(0.0, Rall[:, 0].min() - m), max(0.0, Rall[:, 1].min() - m)
            x1, y1 = min(Rf.W, Rall[:, 2].max() + m), min(Rf.H, Rall[:, 3].max() + m)
            S = [i for i in movable if i in piece or (
                np.clip(np.minimum(Rf.rects[i][:, 2], x1) - np.maximum(Rf.rects[i][:, 0], x0), 0, None)
                * np.clip(np.minimum(Rf.rects[i][:, 3], y1) - np.maximum(Rf.rects[i][:, 1], y0), 0, None)).sum() > 1e-9]
            S = _piece_of(Rf, S, piece[0])
            key = tuple(S)
            if key in seen or len(S) > max_blocks or not set(piece) <= set(S):
                continue
            seen.add(key)
            U = PT.decompose_geom(unary_union([PT.to_polygon(Rf.rects[i]) for i in S]))
            Tn = np.concatenate(list(T.values()))
            if not PT.covered(Tn, U):
                continue
            region = snap(PT.subtract(U, Tn))
            if not len(region) or PT.components(region) != 1:
                continue
            WP = WindowPartitioner(P, Rf, S, preset=T, region=region, shaped=w)
            ops = WP.operands
            if len(ops) < 2:
                continue
            xr0, yr0 = region[:, 0].min(), region[:, 1].min()
            xr1, yr1 = region[:, 2].max(), region[:, 3].max()
            expr = PT.initial_expression(ops, WP.seeds, P.area, (xr0, yr0, xr1, yr1), P.groups)
            for sd in seeds:
                if time.time() > t_end:
                    break
                res = WP.anneal(expr, min(anneal_s, max(0.5, t_end - time.time())), sd)
                if res is None:
                    if verbose:
                        print(f"    shift ({dx:.1f},{dy:.1f}) window {len(S)} seed {sd}: no valid partition", flush=True)
                    continue
                new = {k: snap(Rq) for k, Rq in WP.placed(*res[2]).items()}
                new.update(T)
                if not all(len(Rq) and not thin(Rq) and shape_ok(Rq) for Rq in new.values()):
                    continue
                Rf.rects, Rf.c = dict(start[0]), start[1].copy()
                for k, Rq in new.items():
                    Rf.rects[k], Rf.c[k] = Rq, PT.K.centroid(Rq)
                Rf.hp = P.hpwl(Rf.c[:, 0], Rf.c[:, 1])
                Rf.viol = Rf.soft_viol(range(P.n))
                s0 = Rf.score(Rf.hp, Rf.viol)
                Rf._greedy(t_end, passes)
                sc = Rf.score(Rf.hp, Rf.viol)
                if verbose:
                    print(f"    shift ({dx:.1f},{dy:.1f}) window {len(S)} seed {sd}: score {s0:.4f} -> {sc:.4f} "
                          f"(start {Rf.score(start[2], start[3]):.4f}, viol {Rf.viol})", flush=True)
                if best is None or sc < best[0]:
                    best = (sc, dict(Rf.rects), Rf.c.copy(), Rf.hp, Rf.viol)
                Rf.rects, Rf.c, Rf.hp, Rf.viol = dict(start[0]), start[1].copy(), start[2], start[3]
    Rf.rects, Rf.c, Rf.hp, Rf.viol = dict(start[0]), start[1].copy(), start[2], start[3]
    return best


# -- large-neighbourhood search: re-partition random windows for wirelength ----
def random_window(Rf, rng, kmin: int = 6, kmax: int = 20):
    """Blocks of a window grown around a random movable block until it holds
    k in [kmin, kmax] blocks (one touching piece)."""
    P = Rf.P
    movable = [i for i in range(P.n) if not P.pre[i]]
    c = rng.choice(movable)
    k = rng.randint(kmin, kmax)
    Rc = Rf.rects[c]
    x0, y0, x1, y1 = Rc[:, 0].min(), Rc[:, 1].min(), Rc[:, 2].max(), Rc[:, 3].max()
    m = 0.25 * math.sqrt(P.area[c])
    S = [c]
    for _ in range(30):
        wx0, wy0, wx1, wy1 = x0 - m, y0 - m, x1 + m, y1 + m
        T = [i for i in movable if (
            np.clip(np.minimum(Rf.rects[i][:, 2], wx1) - np.maximum(Rf.rects[i][:, 0], wx0), 0, None)
            * np.clip(np.minimum(Rf.rects[i][:, 3], wy1) - np.maximum(Rf.rects[i][:, 1], wy0), 0, None)).sum() > 1e-9]
        T = _piece_of(Rf, T, c)
        if len(T) > kmax:
            break
        S = T
        if len(S) >= k:
            break
        m *= 1.4
    return S


def lns(Rf, t_end: float, anneal_s: float = 1.0, kmin: int = 6, kmax: int = 14, w: float = 0.5,
        passes: int = 20, seed: int = 0, verbose=False) -> dict:
    """Until t_end: re-partition a random window (rest fixed) by annealing, descend
    greedily, keep it when the score drops. Leaves Rf in the best state.
    1 s anneals and windows of at most 14 blocks (were 3 s, 20): the descent finds the
    gains, and 6-14 blocks gave twice the gain per second of 15-20 (10-24)."""
    import random
    rng = random.Random(seed)
    outer, Rf.dlb = Rf.dlb, DontLook(Rf.P.n)
    try:
        return _lns(Rf, t_end, anneal_s, kmin, kmax, w, passes, rng, verbose)
    finally:
        Rf.dlb = outer


def _lns(Rf, t_end, anneal_s, kmin, kmax, w, passes, rng, verbose) -> dict:
    from .refine import snap, shape_ok, thin
    P, D = Rf.P, Rf.dlb
    tries = wins = 0
    while time.time() < t_end:
        S = random_window(Rf, rng, kmin, kmax)
        if len(S) < 3:
            continue
        tries += 1
        start = (dict(Rf.rects), Rf.c.copy(), Rf.hp, Rf.viol)
        s_start = Rf.score(Rf.hp, Rf.viol)
        WP = WindowPartitioner(P, Rf, S, shaped=w)
        xr0, yr0 = WP.region[:, 0].min(), WP.region[:, 1].min()
        xr1, yr1 = WP.region[:, 2].max(), WP.region[:, 3].max()
        expr = PT.initial_expression(S, WP.seeds, P.area, (xr0, yr0, xr1, yr1), P.groups)
        res = WP.anneal(expr, min(anneal_s, max(0.2, t_end - time.time())), rng.randrange(1 << 30))
        if res is None:
            continue
        new = {k: snap(Rq) for k, Rq in WP.placed(*res[2]).items()}
        if not all(len(Rq) and not thin(Rq) and shape_ok(Rq) for Rq in new.values()):
            continue
        for k, Rq in new.items():
            Rf.rects[k], Rf.c[k] = Rq, PT.K.centroid(Rq)
        Rf.hp = P.hpwl(Rf.c[:, 0], Rf.c[:, 1])
        Rf.viol = Rf.soft_viol(range(P.n))
        D.touch(_changed(Rf.rects, start[0]))
        Rf._greedy(t_end, passes)
        sc = Rf.score(Rf.hp, Rf.viol)
        if sc < s_start - 1e-9:
            wins += 1
            if verbose:
                print(f"    lns try {tries}: window {len(S)} {s_start:.4f} -> {sc:.4f}", flush=True)
        else:
            D.touch(_changed(Rf.rects, start[0]))           # the revert changes them back
            Rf.rects, Rf.c, Rf.hp, Rf.viol = dict(start[0]), start[1].copy(), start[2], start[3]
    return dict(tries=tries, wins=wins, score=Rf.score(Rf.hp, Rf.viol))


def _changed(A, B) -> list:
    return [k for k in A if A[k] is not B[k] and (A[k].shape != B[k].shape or not np.array_equal(A[k], B[k]))]

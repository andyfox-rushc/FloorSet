"""
Zero-white-space partitioner for FloorSet-Prime (10-20).

Prime's ground truth tiles its frame exactly: the pins outline it (pin box ==
frame on all 100 test designs) and its area equals the total block area. So
the canvas is known and the job is to *divide* it, not pack into it:

  1. Rigid layer: preplaced polygons where given; fixed polygons (any of 4
     rotations) and MIB soft groups (identical rectangles) placed near their
     seed centroids without overlap.
  2. Free region = frame minus rigid pieces, as a set of disjoint rectangles.
  3. A slicing tree (Polish expression over the soft blocks) cuts the free
     region recursively; each cut is placed exactly where it splits off its
     subtree's area. Every soft block gets its exact area and nothing is
     left over. Where a cut meets a rigid piece, leaves wrap around it: L, T
     and notched shapes that interlock with their neighbours.
  4. Simulated annealing (Wong-Liu moves) on the expression minimizes
     centroid HPWL and soft violations; a leaf that is split, has a hole or
     more than MAX_CORNERS corners is penalized (and never returned).
"""

import math
import os
import random
import time
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np
import shapely
from shapely.geometry import LineString, Polygon, box
from scipy import ndimage
from scipy.sparse import csr_matrix
from scipy.sparse.csgraph import connected_components
from shapely.ops import unary_union

from . import partition_kernels as K
from . import polygon as G

# Corner cap per polygon (user, 10-20: up to 16, so jigsaw-like interlocking
# shapes can take up all white space; the ground truth never exceeds 8)
MAX_CORNERS = int(os.environ.get("FLOORSET_MAX_CORNERS", "16"))
EPS = 1e-7
BAD_W = 0.5          # search cost per unit of leaf badness (rl/partition.py leaf_badness)
# Search caches per worker process (entries; cleared when full). 200k is ~1 GB per worker;
# 20k (tried 10-21 to share the box with Vivado) clears too often: big designs failed or lost 0.3-0.6
MEMO_MAX = int(os.environ.get("FLOORSET_PARTITION_MEMO", "200000"))
# 10-21 wirelength work: share of ideal-spot swaps, and a search-only penalty per
# log-unit of leaf aspect ratio above ASPECT_MAX (the ground truth rarely passes 3)
IDEAL = float(os.environ.get("FLOORSET_PARTITION_IDEAL", "0"))
# share of moves putting a block next to a net partner in the expression (block-to-block
# nets are where the partition loses: +31% vs the ground truth, the packer +23%)
NET_MATE = float(os.environ.get("FLOORSET_PARTITION_NETMATE", "0"))
ASPECT_W = float(os.environ.get("FLOORSET_PARTITION_ASPECT", "0"))
ASPECT_MAX = 3.0


# -----------------------------------------------------------------------------
# Rectangle-set regions: np.ndarray [m, 4] of disjoint (x0, y0, x1, y1)
# -----------------------------------------------------------------------------
def decompose_geom(geom) -> np.ndarray:
    """Disjoint rectangles covering a rectilinear (Multi)Polygon, holes allowed:
    horizontal slabs between its distinct y's, x intervals from each slab's midline."""
    if geom.is_empty:
        return np.zeros((0, 4))
    ys = []
    for y in sorted(set(round(c[1], 9) for g in getattr(geom, "geoms", [geom])
                        for ring in [g.exterior, *g.interiors] for c in ring.coords)):
        if not ys or y - ys[-1] > EPS:
            ys.append(y)
    x0, _, x1, _ = geom.bounds
    out = []
    for ya, yb in zip(ys, ys[1:]):
        cut = geom.intersection(LineString([(x0 - 1, (ya + yb) / 2), (x1 + 1, (ya + yb) / 2)]))
        for g in getattr(cut, "geoms", [cut]):
            if g.length > EPS:
                out.append((g.bounds[0], ya, g.bounds[2], yb))
    return np.array(out) if out else np.zeros((0, 4))


def area(R: np.ndarray) -> float:
    return K.area(R)


def split(R: np.ndarray, axis: int, a_lo: float) -> Tuple[np.ndarray, np.ndarray]:
    """Cut R with the line coord = t (x if axis 0, else y) at the t where the
    part below t has area a_lo; returns (low part, high part)."""
    return K.split(R, axis, a_lo)


def _touching(A: np.ndarray, B: np.ndarray) -> np.ndarray:
    """[len(A), len(B)] rectangles sharing a boundary of positive length."""
    ox = np.minimum(A[:, None, 2], B[None, :, 2]) - np.maximum(A[:, None, 0], B[None, :, 0])
    oy = np.minimum(A[:, None, 3], B[None, :, 3]) - np.maximum(A[:, None, 1], B[None, :, 1])
    return ((ox > EPS) & (np.abs(oy) <= EPS)) | ((oy > EPS) & (np.abs(ox) <= EPS))


def components(R: np.ndarray) -> int:
    return K.components(R)


def to_polygon(R: np.ndarray):
    return unary_union([box(*r) for r in R]).simplify(0)


def leaf_badness(R: np.ndarray) -> float:
    """0 for a valid leaf (one piece, no hole, no corner-only contact, at most
    MAX_CORNERS corners), else a graded score so the search can tell a
    10-corner leaf from a 16-corner one (rl/partition_kernels.py leaf_stats)."""
    if len(R) == 0:
        return 10.0
    if len(R) == 1:
        return 0.0
    pieces, holes, pinch, corners = K.leaf_stats(R)
    return 3.0 * (pieces - 1) + 3.0 * holes + 2.0 * pinch + 0.5 * max(0, corners - MAX_CORNERS)


def leaf_ok(R: np.ndarray) -> bool:
    return leaf_badness(R) == 0.0


# -----------------------------------------------------------------------------
# Slicing trees as Polish expressions: ints are soft blocks, -1 = V (vertical
# cut: left | right), -2 = H (horizontal cut: bottom / top)
# -----------------------------------------------------------------------------
V, H = -1, -2


def initial_expression(blocks: Sequence[int], seeds: np.ndarray, areas: np.ndarray,
                       bounds: Tuple[float, float, float, float],
                       groups: Sequence[Sequence[int]] = ()) -> List[int]:
    """Recursive bisection of the seed centroids: cut the longer side of the
    current box at the area median of the items sorted along it. Each group
    (cluster) is one item until it is alone in its box, so its members end
    up in one subtree."""
    blocks = [int(b) for b in blocks]
    present = set(blocks)
    items, used = [], set()
    for g in groups:
        m = [int(b) for b in g if int(b) in present]
        if len(m) > 1:
            items.append(m)
            used.update(m)
    items += [[b] for b in blocks if b not in used]

    def build(items, bounds):
        if len(items) == 1:
            m = items[0]
            if len(m) == 1:
                return [m[0]]
            return build([[b] for b in m], bounds)
        x0, y0, x1, y1 = bounds
        axis = 0 if (x1 - x0) >= (y1 - y0) else 1
        def key(it):
            a = areas[it]
            return float((seeds[it, axis] * a).sum() / a.sum())
        order = sorted(items, key=key)
        tot = float(sum(areas[it].sum() for it in order))
        acc, k = 0.0, 1
        for k in range(1, len(order)):
            acc += float(areas[order[k - 1]].sum())
            if acc >= tot / 2:
                break
        lo, hi = order[:k], order[k:]
        frac = float(sum(areas[it].sum() for it in lo)) / tot
        if axis == 0:
            t = x0 + frac * (x1 - x0)
            b_lo, b_hi = (x0, y0, t, y1), (t, y0, x1, y1)
        else:
            t = y0 + frac * (y1 - y0)
            b_lo, b_hi = (x0, y0, x1, t), (x0, t, x1, y1)
        return build(lo, b_lo) + build(hi, b_hi) + [V if axis == 0 else H]

    return build(items, bounds)


_BAD_CACHE: Dict[bytes, float] = {}
# leaf-badness cache entries per process, cleared when full; was 10 x MEMO_MAX, i.e. 2M entries once
# MEMO_MAX went back to 200k -- a training process grew to 6.7 GB (10-21 night)
BAD_CACHE_MAX = int(os.environ.get("FLOORSET_BAD_CACHE", "300000"))


def badness_cached(R: np.ndarray) -> float:
    k = R.tobytes()
    v = _BAD_CACHE.get(k)
    if v is None:
        if len(_BAD_CACHE) > BAD_CACHE_MAX:
            _BAD_CACHE.clear()
        v = _BAD_CACHE[k] = leaf_badness(R)
    return v


def evaluate_expression(expr: Sequence[int], region: np.ndarray, areas: np.ndarray,
                        adaptive: bool = True, memo: Optional[dict] = None) -> Dict[int, np.ndarray]:
    """Leaf regions of a slicing tree over `region` (areas scaled to fill it).
    `memo` (subtree, region) -> leaves lets the search reuse every subtree a
    move left alone; a move changes only the nodes on two root paths."""
    stack = []
    for tok in expr:
        if tok >= 0:
            stack.append((int(tok), float(areas[tok]), int(tok)))
        else:
            r = stack.pop()
            l = stack.pop()
            stack.append(((tok, l, r), l[1] + r[1], hash((tok, l[2], r[2]))))
    leaves = {}

    def fits(node, R):
        """Cheap test that a child region can work: a leaf must be valid, a
        subtree's region at least one piece."""
        if isinstance(node[0], int):
            return badness_cached(R) == 0.0
        return components(R) == 1

    def assign(node, R):
        item, a, key = node
        if isinstance(item, int):
            leaves[item] = R
            return
        mk = (key, R.tobytes()) if memo is not None else None
        if mk is not None and mk in memo:
            leaves.update(memo[mk])
            return
        before = set(leaves) if mk is not None else None
        op, l, r = item
        axis = 0 if op == V else 1
        tot = area(R)
        # adaptive cut: the expression's cut first, then the other axis and the
        # swapped side order, keeping the first whose children both fit
        first = None
        for ax, (p, q) in ((axis, (l, r)), (1 - axis, (l, r)), (axis, (r, l)), (1 - axis, (r, l))):
            lo, hi = split(R, ax, tot * p[1] / a)
            if first is None:
                first = (p, q, lo, hi)
            if not adaptive or (fits(p, lo) and fits(q, hi)):
                break
        else:
            p, q, lo, hi = first
        assign(p, lo)
        assign(q, hi)
        if mk is not None:
            if len(memo) > MEMO_MAX:
                memo.clear()
            memo[mk] = {b: leaves[b] for b in leaves if b not in before}

    assign(stack[0], region)
    return leaves


# -----------------------------------------------------------------------------
# Problem
# -----------------------------------------------------------------------------
class Problem:
    """One Prime design for the partitioner (numpy, block indices as in Prime)."""

    def __init__(self, block_count, area_targets, b2b, p2b, pins_pos, constraints, target_polygons):
        self.n = n = block_count
        self.area = area_targets[:n].double().numpy()
        c = constraints[:n].long().numpy()
        self.fixed, self.pre = c[:, 0] != 0, c[:, 1] != 0
        self.mib, self.cluster, self.bnd = c[:, 2], c[:, 3], c[:, 4]
        pins = pins_pos.double().numpy()
        ok = (pins[:, 0] >= 0) & (pins[:, 1] >= 0)
        self.W, self.H = float(pins[ok, 0].max()), float(pins[ok, 1].max())
        b2b = b2b[b2b[:, 0] >= 0].double().numpy() if len(b2b) else np.zeros((0, 3))
        p2b = p2b[p2b[:, 0] >= 0].double().numpy() if len(p2b) else np.zeros((0, 3))
        self.bi, self.bj, self.bw = b2b[:, 0].astype(int), b2b[:, 1].astype(int), b2b[:, 2]
        self.pb, self.pw = p2b[:, 1].astype(int), p2b[:, 2]
        self.px, self.py = pins[p2b[:, 0].astype(int), 0], pins[p2b[:, 0].astype(int), 1]
        self.targets = target_polygons
        # MIB soft groups: one rectangle shape for the whole group, placed like a
        # fixed block (slicing leaves would all differ: 5 of 18 violations on design 0)
        self.mib_soft = np.zeros(n, dtype=bool)
        for g in set(self.mib.tolist()) - {0}:
            m = self.mib == g
            if m.sum() > 1 and not (self.fixed[m] | self.pre[m]).any():
                self.mib_soft |= m
        # MIB groups with a fixed / preplaced member: the soft members copy its shape
        self.mib_copy = {}
        for g in set(self.mib.tolist()) - {0}:
            m = np.where(self.mib == g)[0]
            known = [i for i in m if self.fixed[i] or self.pre[i]]
            if known:
                for i in m:
                    if not (self.fixed[i] or self.pre[i]):
                        self.mib_copy[int(i)] = int(known[0])
        copy = np.zeros(n, dtype=bool)
        copy[list(self.mib_copy)] = True
        self.rigid = self.fixed | self.pre | self.mib_soft | copy
        self.soft = ~self.rigid
        n_soft = int((self.bnd != 0).sum())
        for ids in (self.mib, self.cluster):
            for g in set(ids.tolist()) - {0}:
                n_soft += max(0, int((ids == g).sum()) - 1)
        self.n_soft = max(n_soft, 1)
        self.groups = [np.where(self.cluster == g)[0] for g in sorted(set(self.cluster.tolist()) - {0})]
        self.groups = [g for g in self.groups if len(g) > 1]

    def hpwl(self, cx, cy) -> float:
        hp = float((self.bw * (np.abs(cx[self.bi] - cx[self.bj]) + np.abs(cy[self.bi] - cy[self.bj]))).sum())
        return hp + float((self.pw * (np.abs(cx[self.pb] - self.px) + np.abs(cy[self.pb] - self.py))).sum())


# -----------------------------------------------------------------------------
# Rigid blocks as tree leaves
# -----------------------------------------------------------------------------
def subtract(R: np.ndarray, S: np.ndarray) -> np.ndarray:
    """Rectangle set R minus rectangle set S (still disjoint rectangles)."""
    return K.subtract(R, S)


def covered(S: np.ndarray, R: np.ndarray) -> bool:
    """Is every rectangle of S inside the union of R?"""
    return K.covered(S, R)


def _rects(s: G.Shape, x: float, y: float) -> np.ndarray:
    return np.array([(x + a, y + b, x + a + c, y + b + d) for a, b, c, d in s.pieces], dtype=np.float64)


# -----------------------------------------------------------------------------
# Search
# -----------------------------------------------------------------------------
class Partitioner:
    """Slicing tree over every block except the preplaced ones. A soft leaf
    takes whatever region its cuts leave; a rigid leaf (fixed polygon, MIB
    copy, MIB soft group rectangle) next to a sibling subtree is put in a
    corner of their shared region -- best rotation and corner by distance to
    its seed -- and the sibling gets the rest, which wraps around it."""

    def __init__(self, P: Problem, seeds: np.ndarray):
        self.P = P
        # edge-pinned blocks: seed on their edge, so the first tree and the
        # rigid corner choice put them there
        seeds = seeds.copy()
        for i in np.where(P.bnd != 0)[0]:
            code = int(P.bnd[i])
            if code & 1: seeds[i, 0] = 0.0
            if code & 2: seeds[i, 0] = P.W
            if code & 8: seeds[i, 1] = 0.0
            if code & 4: seeds[i, 1] = P.H
        self.seeds = seeds
        self.pre = {}
        for i in np.where(P.pre)[0]:
            g = Polygon(G.clean_vertices(P.targets[i]))
            self.pre[int(i)] = (G.shape_from_polygon(g), g.bounds[0], g.bounds[1])
        frame = box(0, 0, P.W, P.H)
        solid = unary_union([s.polygon(x, y) for s, x, y in self.pre.values()]) if self.pre else Polygon()
        self.region = decompose_geom(frame.difference(solid))
        self.operands = [int(i) for i in np.where(~P.pre)[0]]
        self.options = {}                       # rigid block -> candidate shapes
        mib_rect = {}
        for i in self.operands:
            if P.fixed[i] or i in P.mib_copy:
                self.options[i] = G.orientations(G.shape_from_vertices(P.targets[P.mib_copy.get(i, i)]))
            elif P.mib_soft[i]:
                g = P.mib[i]
                if g not in mib_rect:            # one shape per group; rotations keep it the same shape
                    a = float(P.area[i])
                    mib_rect[g] = G.orientations(G.rect_shape(math.sqrt(a * 1.5), math.sqrt(a / 1.5)))
                self.options[i] = mib_rect[g]
        self.pocket_rigid = {}             # rigid blocks that exactly fill a pocket
        self.pockets = self._fill_pockets(self.seeds)
        self.ref = None
        self.memo = {}
        self.code = [int(c) for c in P.bnd]
        nb = [[] for _ in range(P.n)]                   # (partner, weight) for the ideal-spot move
        for i, j, w in zip(P.bi, P.bj, P.bw):
            nb[i].append((j, w))
            nb[j].append((i, w))
        self.nbrs = [np.array(v, dtype=np.float64).reshape(-1, 2) for v in nb]
        self.pinsum = np.zeros((P.n, 3))
        for b, w, x, y in zip(P.pb, P.pw, P.px, P.py):
            self.pinsum[b] += (w * x, w * y, w)
        self.wsum = np.array([v[:, 1].sum() if len(v) else 0.0 for v in self.nbrs]) + self.pinsum[:, 2]
        ops = set(self.operands)
        self.mates = {}
        for g in P.groups:
            m = [int(b) for b in g if int(b) in ops]
            for b in m:
                if len(m) > 1:
                    self.mates[b] = [x for x in m if x != b]

    # -- evaluation --------------------------------------------------------
    def _place(self, b, R, side):
        """Best (bad, shape, x, y, rest) for rigid block b in a corner of R on
        `side` (0 left, 1 right, 2 bottom, 3 top, None any)."""
        x0, y0, x1, y1 = R[:, 0].min(), R[:, 1].min(), R[:, 2].max(), R[:, 3].max()
        sx, sy = self.seeds[b]
        best = None
        for s in self.options[b]:
            spots = {0: [(x0, y0), (x0, y1 - s.h)], 1: [(x1 - s.w, y0), (x1 - s.w, y1 - s.h)],
                     2: [(x0, y0), (x1 - s.w, y0)], 3: [(x0, y1 - s.h), (x1 - s.w, y1 - s.h)]}
            cand = spots[side] if side is not None else [p for v in spots.values() for p in v]
            # fallback: corners of every rectangle of R (the region's own corners
            # are often taken by preplaced blocks: designs 35 and 75 never fit)
            extra = [(r[0], r[1]) for r in R] + [(r[2] - s.w, r[1]) for r in R] \
                + [(r[0], r[3] - s.h) for r in R] + [(r[2] - s.w, r[3] - s.h) for r in R]
            for k, (x, y) in enumerate(cand + extra):
                if k == len(cand) and best is not None and best[0][0] == 0.0:
                    break                     # a corner on the asked side worked
                rr = _rects(s, x, y)
                ok = covered(rr, R)
                rest = subtract(R, rr) if ok else None
                bad = 0.0 if ok and (len(rest) == 0 or components(rest) == 1) else (2.0 if ok else 5.0)
                d = abs(x + s.cx - sx) + abs(y + s.cy - sy)
                code = self.code[b]          # edge pins: corners on the required edge first
                miss = (((code & 1) and x > EPS) + ((code & 2) and x + s.w < self.P.W - EPS)
                        + ((code & 8) and y > EPS) + ((code & 4) and y + s.h < self.P.H - EPS))
                key = (bad, miss, d)
                if best is None or key < best[0]:
                    best = (key, s, x, y, rest if ok else R)
        (bad, _, _), s, x, y, rest = best
        return bad, s, x, y, rest

    def _fill_pockets(self, seeds):
        """Free space walled in by preplaced blocks (and the frame) is its own
        piece of the free region; no cut can give it to a block, so each such
        pocket goes up front to the soft block (or pair of blocks) whose area
        matches it within the area tolerance, nearest seeds first (design 35's
        top pocket is one ground-truth block). The tree then works on the main
        piece only. Returns {block: region}."""
        P = self.P
        R = self.region
        lab = K.component_labels(R)
        if lab.max() == 0:
            return {}
        areas = np.array([area(R[lab == c]) for c in range(lab.max() + 1)])
        main = int(np.argmax(areas))
        soft = [b for b in self.operands if b not in self.options]
        out = {}
        for c in np.argsort(areas):
            if c == main:
                continue
            C = R[lab == c]
            A = areas[c]
            cx, cy = K.centroid(C)
            near = lambda bs: sum(abs(seeds[b, 0] - cx) + abs(seeds[b, 1] - cy) for b in bs)
            ok = lambda tot: abs(tot - A) <= 0.009 * A
            # a rigid block whose shape fills the pocket exactly (design 75: a fixed block)
            fit = None
            for b in sorted((b for b in self.options if b in self.operands and b not in self.pocket_rigid
                             and abs(P.area[b] - A) <= 1e-6 * A), key=lambda b: near([b])):
                for sh in self.options[b]:
                    x, y = C[:, 0].min(), C[:, 1].min()
                    rr = _rects(sh, x, y)
                    if covered(rr, C) and abs(area(rr) - A) <= 1e-6 * A:
                        fit = (b, sh, x, y)
                        break
                if fit:
                    break
            if fit is None and len(C) == 1:
                # a rectangular pocket the size of a soft MIB group's blocks: the group takes the
                # pocket's shape (its members only need to match each other) and one member fills
                # it (design 75: a 5 x 7 pocket, MIB blocks of area 35 -- the 1.5-aspect default never fit)
                pw, ph = C[0, 2] - C[0, 0], C[0, 3] - C[0, 1]
                for b in sorted((b for b in self.operands if P.mib_soft[b] and b not in self.pocket_rigid
                                 and abs(P.area[b] - A) <= 1e-6 * A), key=lambda b: near([b])):
                    shapes = G.orientations(G.rect_shape(pw, ph))
                    for m in np.where(P.mib == P.mib[b])[0]:
                        if int(m) in self.options:
                            self.options[int(m)] = shapes
                    fit = (b, shapes[0], C[0, 0], C[0, 1])
                    break
            if fit:
                b, sh, x, y = fit
                self.pocket_rigid[b] = (sh, x, y)
                out[b] = C
                continue
            free = [b for b in soft if b not in out]
            single = [b for b in free if ok(P.area[b]) and leaf_ok(C)]
            if single:
                b = min(single, key=lambda b: near([b]))
                out[b] = C
                continue
            pairs = [(a, b) for k, a in enumerate(free) for b in free[k + 1:] if ok(P.area[a] + P.area[b])]
            for a, b in sorted(pairs, key=near):
                for ax in (0, 1):
                    lo, hi = split(C, ax, A * P.area[a] / (P.area[a] + P.area[b]))
                    if leaf_ok(lo) and leaf_ok(hi):
                        out[a], out[b] = lo, hi
                        break
                if a in out:
                    break
        if out:      # pockets that got blocks leave the tree's region; unmatched ones stay
            rest = [c for c in range(lab.max() + 1) if c == main or not self._pocket_taken(R[lab == c], out)]
            self.region = R[np.isin(lab, rest)]
            self.operands = [b for b in self.operands if b not in out]
        return {b: C for b, C in out.items() if b not in self.pocket_rigid}

    @staticmethod
    def _pocket_taken(C, out):
        a = area(C)
        got = sum(area(v) for v in out.values() if covered(v, C))
        return abs(got - a) <= 1e-6 * max(a, 1.0)

    def _pocket_cut(self, R, ax, a_lo, p, q):
        """A cut that leaves a soft leaf in pieces (a pocket walled in by
        preplaced blocks, design 35) keeps its largest piece; the pockets go
        to the sibling and the cut moves by their area until the leaf's piece
        has exactly its area. Returns (lo, hi) or (None, None)."""
        is_soft_leaf = lambda nd: isinstance(nd[0], int) and nd[0] not in self.options
        if not (is_soft_leaf(p) or is_soft_leaf(q)):
            return None, None
        leaf_lo = is_soft_leaf(p)
        target = a_lo if leaf_lo else area(R) - a_lo
        cut = a_lo
        for _ in range(4):
            lo, hi = split(R, ax, cut)
            side = lo if leaf_lo else hi
            if len(side) == 0:
                return None, None
            lab = K.component_labels(side)
            if lab.max() == 0:
                break
            areas = np.array([area(side[lab == c]) for c in range(lab.max() + 1)])
            main = int(np.argmax(areas))
            pockets = float(areas.sum() - areas[main])
            if abs(areas[main] - target) <= 1e-6 * max(target, 1.0):
                keep, give = side[lab == main], side[lab != main]
                if leaf_lo:
                    return keep, np.concatenate([hi, give])
                return np.concatenate([lo, give]), keep
            cut = (target + pockets) if leaf_lo else (area(R) - target - pockets)
        return None, None

    def evaluate(self, expr):
        """(soft leaves {b: rects}, rigid {b: (shape, x, y)}, badness)."""
        P = self.P
        stack = []
        for tok in expr:
            if tok >= 0:
                stack.append((int(tok), float(P.area[tok]), int(tok)))
            else:
                r = stack.pop()
                l = stack.pop()
                stack.append(((tok, l, r), l[1] + r[1], hash((tok, l[2], r[2]))))
        leaves, rigid = {}, {}
        state = {"bad": 0.0}
        memo = self.memo
        is_leaf = lambda nd: isinstance(nd[0], int)
        is_rigid = lambda nd: is_leaf(nd) and nd[0] in self.options

        def fits(nd, R):
            if is_rigid(nd):
                return True
            if is_leaf(nd):
                return badness_cached(R) == 0.0
            return components(R) == 1

        def assign(nd, R):
            item, a, key = nd
            if is_leaf(nd):
                if item in self.options:            # rigid with nothing beside it: fill R exactly or fail
                    bad, s, x, y, rest = self._place(item, R, None)
                    rigid[item] = (s, x, y)
                    state["bad"] += bad + (3.0 if len(rest) and area(rest) > 1e-3 * a else 0.0)
                else:
                    leaves[item] = R
                return
            mk = (key, R.tobytes())
            hit = memo.get(mk)
            if hit is not None:
                leaves.update(hit[0])
                rigid.update(hit[1])
                state["bad"] += hit[2]
                return
            before_l, before_r, before_b = set(leaves), set(rigid), state["bad"]
            op, l, r = item
            if is_rigid(l) != is_rigid(r):          # rigid leaf beside a subtree: corner placement
                rg, other = (l, r) if is_rigid(l) else (r, l)
                side = (0 if rg is l else 1) if op == V else (2 if rg is l else 3)
                bad, s, x, y, rest = self._place(rg[0], R, side)
                if bad:                             # the other side may still fit
                    alt = self._place(rg[0], R, None)
                    if alt[0] < bad:
                        bad, s, x, y, rest = alt
                rigid[rg[0]] = (s, x, y)
                state["bad"] += bad
                assign(other, rest)
            else:
                axis = 0 if op == V else 1
                tot = area(R)
                first = None
                for ax, (p, q) in ((axis, (l, r)), (1 - axis, (l, r)), (axis, (r, l)), (1 - axis, (r, l))):
                    lo, hi = split(R, ax, tot * p[1] / a)
                    if first is None:
                        first = (p, q, lo, hi)
                    if fits(p, lo) and fits(q, hi):
                        break
                    lo2, hi2 = self._pocket_cut(R, ax, tot * p[1] / a, p, q)
                    if lo2 is not None and fits(p, lo2) and fits(q, hi2):
                        lo, hi = lo2, hi2
                        break
                else:
                    p, q, lo, hi = first
                assign(p, lo)
                assign(q, hi)
            if len(memo) > MEMO_MAX:
                memo.clear()
            memo[mk] = ({b: leaves[b] for b in leaves if b not in before_l},
                        {b: rigid[b] for b in rigid if b not in before_r}, state["bad"] - before_b)

        assign(stack[0], self.region)
        leaves.update(self.pockets)
        rigid.update(self.pocket_rigid)
        return leaves, rigid, state["bad"]

    def cost(self, expr):
        P = self.P
        leaves, rigid, bad = self.evaluate(expr)
        cx, cy = np.zeros(P.n), np.zeros(P.n)
        rects = {}
        for i, (s, x, y) in list(self.pre.items()) + list(rigid.items()):
            cx[i], cy[i] = x + s.cx, y + s.cy
            rects[i] = _rects(s, x, y)
        for b, R in leaves.items():
            cx[b], cy[b] = K.centroid(R)
            bad += badness_cached(R)
            rects[b] = R
        v = 0
        for i in np.where(P.bnd != 0)[0]:
            R, code = rects[i], int(P.bnd[i])
            v += not (((code & 1) == 0 or R[:, 0].min() <= EPS) and ((code & 2) == 0 or R[:, 2].max() >= P.W - EPS)
                      and ((code & 8) == 0 or R[:, 1].min() <= EPS) and ((code & 4) == 0 or R[:, 3].max() >= P.H - EPS))
        for g in P.groups:
            m = len(g)
            adj = np.zeros((m, m), dtype=bool)
            for a_ in range(m):
                for b_ in range(a_ + 1, m):
                    adj[a_, b_] = adj[b_, a_] = K.touching_any(rects[g[a_]], rects[g[b_]])
            v += int(connected_components(csr_matrix(adj), directed=False)[0]) - 1
        hp = P.hpwl(cx, cy)
        if self.ref is None:
            self.ref = max(hp, 1e-9)
        # additive penalty: a multiplier gave 300x cost swings early on and set
        # the annealing temperature far too hot for most of the run
        c = (1 + 0.5 * (hp / self.ref + 1)) * math.exp(2 * v / P.n_soft) + BAD_W * bad
        self._cxy = (cx, cy)
        search = c
        if ASPECT_W > 0:       # search-only: long thin leaves put centroids far from their nets
            asp = 0.0
            for R in leaves.values():
                w = R[:, 2].max() - R[:, 0].min()
                h = R[:, 3].max() - R[:, 1].min()
                asp += max(0.0, math.log(max(w / h, h / w) / ASPECT_MAX))
            search += ASPECT_W * asp
        return search, (leaves, rigid), bad, c

    def anneal(self, expr: List[int], time_budget: float, seed: int = 0, verbose: bool = False,
               max_steps: Optional[int] = None):
        """max_steps: a fixed number of moves instead of the time budget (deterministic for a
        given expression and seed: RL rewards from timed anneals followed the machine's load)."""
        rng = random.Random(seed)
        cur = list(expr)
        cur_c, cur_l, cur_bad, cur_t = self.cost(cur)
        self.cur_cxy = self._cxy
        best = (cur_t, list(cur), cur_l, cur_bad)
        best_ok = (cur_t, list(cur), cur_l) if cur_bad == 0 else None
        t0 = time.time()
        # start temperature: median cost change of sample moves from the start
        ds = []
        for _ in range(100):
            new = self._move(cur, rng)
            if new is not None:
                ds.append(abs(self.cost(new)[0] - cur_c))
        temp0 = max(float(np.median(ds)) if ds else 0.1, 1e-3)
        steps = 0
        while (steps < max_steps) if max_steps else (time.time() - t0 < time_budget):
            new = self._move(cur, rng)
            if new is None:
                continue
            c, l, bad, ct = self.cost(new)
            steps += 1
            d = c - cur_c
            frac = steps / max_steps if max_steps else (time.time() - t0) / time_budget
            temp = temp0 * 0.001 ** frac
            if d <= 0 or rng.random() < math.exp(-d / max(temp, 1e-12)):
                cur, cur_c = new, c
                self.cur_cxy = self._cxy
                if ct < best[0]:
                    best = (ct, list(new), l, bad)
                if bad == 0 and (best_ok is None or ct < best_ok[0]):
                    best_ok = (ct, list(new), l)
        self.last_best = best          # diagnostics: best state, valid or not
        if verbose:
            print(f"  partition: {steps} steps, best {best[0]:.4f} bad={best[3]}, "
                  f"best feasible {best_ok[0] if best_ok else None}")
        return best_ok

    def _move(self, expr, rng):
        e = list(expr)
        kind = rng.random()
        operands = [k for k, t in enumerate(e) if t >= 0]
        if IDEAL > 0 and rng.random() < IDEAL:          # swap a block toward where its nets pull it
            cx, cy = self.cur_cxy
            b = e[rng.choice(operands)]
            sw = self.wsum[b]
            if sw > 0:
                nb = self.nbrs[b]
                ix = (float((nb[:, 1] * cx[nb[:, 0].astype(int)]).sum()) + self.pinsum[b, 0]) / sw
                iy = (float((nb[:, 1] * cy[nb[:, 0].astype(int)]).sum()) + self.pinsum[b, 1]) / sw
                ops = [e[k] for k in operands]
                d = np.abs(cx[ops] - ix) + np.abs(cy[ops] - iy)
                c = ops[int(np.argmin(d))]
                if c != b:
                    ka, kb = e.index(b), e.index(c)
                    e[ka], e[kb] = e[kb], e[ka]
                    return e
            return None
        if NET_MATE > 0 and rng.random() < NET_MATE:    # bring a block next to a net partner
            b = e[rng.choice(operands)]
            nb = self.nbrs[b]
            if len(nb):
                pr = nb[:, 1] / nb[:, 1].sum()
                m = int(nb[rng.choices(range(len(nb)), weights=pr)[0], 0])
                pos = {t: k for k, t in enumerate(e) if t >= 0}
                if m in pos and b in pos:
                    kb, km = pos[b], pos[m]
                    j = operands.index(km)
                    tgt = (operands[j + 1] if j + 1 < len(operands) and operands[j + 1] != kb
                           else (operands[j - 1] if j > 0 else None))
                    if tgt is not None and tgt != kb:
                        e[kb], e[tgt] = e[tgt], e[kb]
                        return e
            return None
        if kind < 0.15 and self.mates:                  # bring a block next to a clustermate
            b = rng.choice(list(self.mates))
            pos = {t: k for k, t in enumerate(e) if t >= 0}
            if b in pos:
                m = rng.choice(self.mates[b])
                if m in pos:
                    kb, km = pos[b], pos[m]
                    j = operands.index(km)            # the operand next to the mate
                    tgt = (operands[j + 1] if j + 1 < len(operands) and operands[j + 1] != kb
                           else (operands[j - 1] if j > 0 else None))
                    if tgt is not None and tgt != kb:
                        e[kb], e[tgt] = e[tgt], e[kb]
                        return e
            return None
        if kind < 0.45:                                 # swap two operands
            a, b = rng.sample(operands, 2)
            e[a], e[b] = e[b], e[a]
            return e
        if kind < 0.72:                                 # complement an operator chain
            ops = [k for k, t in enumerate(e) if t < 0]
            k = rng.choice(ops)
            while k < len(e) and e[k] < 0:
                e[k] = V if e[k] == H else H
                k += 1
            return e
        k = rng.randrange(len(e) - 1)                   # swap adjacent operand / operator
        if (e[k] >= 0) == (e[k + 1] >= 0):
            return None
        e[k], e[k + 1] = e[k + 1], e[k]
        cnt = 0
        for t in e:                                     # balloting property
            cnt += 1 if t >= 0 else -1
            if cnt < 1:
                return None
        return e


# Learned tree builder (rl/tree_policy.py, plan A 10-21): the search starts from the
# cheapest of TREE_SAMPLES sampled trees (plus the policy's greedy tree and the
# seed tree) instead of the seed tree alone. Empty = off.
TREE_POLICY = os.environ.get("FLOORSET_TREE_POLICY", "")
TREE_SAMPLES = int(os.environ.get("FLOORSET_TREE_SAMPLES", "32"))
_TREE_NET = []


def _policy_start(part, expr, seed):
    import torch
    from . import tree_policy as TP
    if not _TREE_NET:
        torch.set_num_threads(1)
        _TREE_NET.append(TP.load(TREE_POLICY))
    net = _TREE_NET[0]
    if net is None:
        return expr
    best = (part.cost(expr)[3], expr)        # first cost call also fixes part.ref, as before
    ctx = TP.Context(part)
    gen = torch.Generator().manual_seed(seed)
    with torch.no_grad():
        for k in range(TREE_SAMPLES + 1):
            e, _, _ = TP.rollout(net, part, ctx, greedy=(k == 0), rng=gen)
            c = part.cost(e)[3]
            if c < best[0]:
                best = (c, e)
    return best[1]


# Pair re-cut refinement (rl/refine.py, 10-21 night) of every restart's result;
# restarts are then compared by the refined contest proxy. 0 = off.
REFINE = float(os.environ.get("FLOORSET_REFINE", "0"))       # seconds per restart


def _restart_job(args):
    """Process-pool entry: one independent search -> (search cost, polygons) or None.
    A crash returns None: an exception here made my_optimizer._pmap drop the pool and
    run every later design's restarts serially, 4x slower (10-22)."""
    try:
        return _restart(args)
    except Exception as e:
        print(f"  restart {args[3]} failed: {e!r}", flush=True)
        return None


def _restart(args):
    P, seeds, budget, seed = args
    res = solve(P, seeds, budget, seed, with_cost=True)
    if res is None or REFINE <= 0:
        return res
    from .refine import Refiner, layout_ok
    R = Refiner(P, res[1])
    st = R.run(time_budget=REFINE)
    lay = R.layout()
    if not layout_ok(P, lay):           # 10-21: two refined layouts broke hard rules unchecked
        R0 = Refiner(P, res[1])
        R0.hp = R0.hp_ref = max(P.hpwl(R0.c[:, 0], R0.c[:, 1]), 1e-9)
        return (R0.hp, R0.soft_viol(range(P.n))), res[1]
    return (st["hpwl"], st["viol"]), lay


def solve_parallel(P: Problem, seeds, time_budget: float, restarts: int = 6,
                   pmap=None, verbose: bool = False):
    """`restarts` independent searches (different random seeds), the cheapest
    valid result kept; `pmap` runs them in parallel (my_optimizer._pmap).
    `seeds`: one centroid array, or a list cycled over the restarts."""
    seed_sets = seeds if isinstance(seeds, (list, tuple)) else [seeds]
    jobs = [(P, seed_sets[k % len(seed_sets)], time_budget, k) for k in range(restarts)]
    res = (pmap or (lambda f, js: [f(j) for j in js]))(_restart_job, jobs)
    ok = [r for r in res if r is not None]
    if ok and REFINE > 0:              # refined: (hpwl, violations) -> contest proxy on a common scale
        h0 = min(c[0] for c, _ in ok)
        ok = [((0.5 + 0.5 * c[0] / h0) * math.exp(2 * c[1] / P.n_soft), l) for c, l in ok]
    if verbose:
        print(f"  partition restarts: {len(ok)}/{restarts} valid, costs "
              + ", ".join(f"{c:.3f}" for c, _ in sorted(ok, key=lambda t: t[0])))
    return min(ok, key=lambda t: t[0])[1] if ok else None


def solve(P: Problem, seeds: np.ndarray, time_budget: float = 10.0, seed: int = 0,
          verbose: bool = False, with_cost: bool = False):
    """Polygons for every block, or None if no valid partition was found
    (with_cost: (search cost, polygons))."""
    part = Partitioner(P, seeds)
    cost = 0.0
    ops = part.operands
    if not ops:
        leaves, rigid = {}, {}
    else:
        x0, y0 = part.region[:, 0].min(), part.region[:, 1].min()
        x1, y1 = part.region[:, 2].max(), part.region[:, 3].max()
        expr = initial_expression(ops, part.seeds, P.area, (x0, y0, x1, y1), P.groups)
        if len(ops) > 2 and TREE_POLICY:
            expr = _policy_start(part, expr, seed)
        if len(ops) > 1:
            res = part.anneal(expr, time_budget, seed, verbose)
            if res is None:
                return None
            cost = res[0]
            leaves, rigid = res[2]
        else:
            leaves, rigid, bad = part.evaluate(expr)
            if bad:
                return None
    out = []
    for i in range(P.n):
        if i in part.pre or i in rigid:
            s, x, y = part.pre.get(i) or rigid[i]
            out.append(list(s.polygon(x, y).exterior.coords[:-1]))
        else:
            out.append(list(to_polygon(leaves[i]).exterior.coords[:-1]))
    return (cost, out) if with_cost else out

"""
Stages 2 and 3 of the grid flow (10-21): JigsawPlanner-style legalization on
Prime's unit grid (ICCAD'24, doi 10.1145/3676536.3676707), from the stage 1
boxes of rl/global_alm.py.

Prime coordinates and areas are integers, so unit cells give every block its
exact area and the frame is tiled with no white space and no overlap:

  1. preplaced / pocket blocks are rasterized; rigid blocks (fixed shapes,
     MIB rectangles) go to the nearest free integer spot (any rotation);
  2. claims: each soft block's stage 1 box, with every pairwise overlap cut
     across its longer centre-distance axis in the ratio sigma_ij (RPolyPlan's
     shape determination);
  3. assignment: each free cell goes to argmin_i (cost_i - price_i), cost 0
     inside the claim and the taxicab distance to it outside; prices are
     adjusted until the counts are close, and a repair passes single cells
     along chains of neighbouring blocks until every count is exact;
  4. cleanup: stray fragments go to a neighbour (then re-balanced).

Moves in 3-4 change one cell's owner between neighbours and are only made when
the cell is a simple point for both blocks (no split, no hole), preferring
moves that do not add corners -- the move set the stage 3 RL policy will choose
from (cell transfers keep every block's area exact when they come in chains).
"""

import math
from typing import Dict, List, Optional, Tuple

import numpy as np
from scipy import ndimage
from shapely.geometry import box as sbox
from shapely.ops import unary_union

from . import polygon as G
from . import partition as PT

FREE = -1
CORNER_W = 4.0         # cost per added corner in repair moves (frame units of distance)
SOFT_W = 0.05          # re-cut objective per cluster piece / edge-pin miss (HPWL is relative)
CLUSTER_W = 3.0        # rigid placement: cost per cell of distance from the cluster's soft claims
TABU = 30              # corner_fix: moves before a cell may move again


# -----------------------------------------------------------------------------
# Small grid helpers
# -----------------------------------------------------------------------------
def _fill(owner, x, y, w, h, val):
    owner[int(round(y)):int(round(y + h)), int(round(x)):int(round(x + w))] = val


def _integral(occ):
    I = np.zeros((occ.shape[0] + 1, occ.shape[1] + 1), dtype=np.int64)
    I[1:, 1:] = occ.cumsum(0).cumsum(1)
    return I


def _simple(mask3: np.ndarray) -> bool:
    """(4, 8) simple-point test on a 3x3 foreground window (centre ignored):
    changing the centre keeps the foreground's 4-connectivity and the
    background's 8-connectivity -- no split, no new hole."""
    m = mask3.copy()
    m[1, 1] = False
    four = [(0, 1), (1, 0), (1, 2), (2, 1)]
    if not any(m[p] for p in four):
        return False
    lab, k = ndimage.label(m, structure=[[0, 1, 0], [1, 1, 1], [0, 1, 0]])
    if len({lab[p] for p in four if m[p]}) != 1:
        return False
    bg = ~m
    bg[1, 1] = False
    lab2, _ = ndimage.label(bg, structure=np.ones((3, 3)))
    return len({lab2[p] for p in four if bg[p]}) == 1


def _window(owner, r, c, b):
    H, W = owner.shape
    out = np.zeros((3, 3), dtype=bool)
    for dr in (-1, 0, 1):
        for dc in (-1, 0, 1):
            rr, cc = r + dr, c + dc
            if 0 <= rr < H and 0 <= cc < W:
                out[dr + 1, dc + 1] = owner[rr, cc] == b
    return out


def _window_set(owner, r, c, members):
    H, W = owner.shape
    out = np.zeros((3, 3), dtype=bool)
    for dr in (-1, 0, 1):
        for dc in (-1, 0, 1):
            rr, cc = r + dr, c + dc
            if 0 <= rr < H and 0 <= cc < W:
                out[dr + 1, dc + 1] = int(owner[rr, cc]) in members
    return out


def _corners_at(owner, r, c, b):
    """Corners block b has at the 4 grid vertices around cell (r, c)."""
    H, W = owner.shape
    tot = 0
    for vr in (r, r + 1):
        for vc in (c, c + 1):
            q = [(vr - 1, vc - 1), (vr - 1, vc), (vr, vc - 1), (vr, vc)]
            s = [0 <= a < H and 0 <= bb < W and owner[a, bb] == b for a, bb in q]
            k = sum(s)
            tot += 1 if k in (1, 3) else (2 if k == 2 and s[0] == s[3] else 0)
    return tot


def corners(mask: np.ndarray) -> int:
    p = np.pad(mask, 1).astype(np.int8)
    q = p[:-1, :-1] + p[:-1, 1:] + p[1:, :-1] + p[1:, 1:]
    diag = (p[:-1, :-1] == p[1:, 1:]) & (p[:-1, 1:] == p[1:, :-1]) & (p[:-1, :-1] != p[:-1, 1:])
    return int(((q == 1) | (q == 3)).sum() + 2 * diag.sum())


# -----------------------------------------------------------------------------
class GridLayout:
    def __init__(self, P: PT.Problem, part: PT.Partitioner):
        self.P, self.part = P, part
        self.W, self.H = int(round(P.W)), int(round(P.H))
        self.owner = np.full((self.H, self.W), FREE, dtype=np.int32)
        self.rigid_at: Dict[int, Tuple[object, int, int]] = {}
        for i, (s, x, y) in list(part.pre.items()) + list(part.pocket_rigid.items()):
            for px, py, pw, ph in s.pieces:
                _fill(self.owner, x + px, y + py, pw, ph, i)
            self.rigid_at[i] = (s, x, y)
        for b, R in part.pockets.items():
            for r in R:
                _fill(self.owner, r[0], r[1], r[2] - r[0], r[3] - r[1], b)
        self.locked = set(int(v) for v in np.unique(self.owner) if v >= 0)
        self.group_of = {}
        for grp in P.groups:
            g = frozenset(int(b) for b in grp)
            for b in g:
                self.group_of[b] = g

    # -- rigid blocks ---------------------------------------------------------
    def _int_shapes(self, b):
        """Integer-sized orientations of rigid block b (MIB rectangles rounded
        to integer sides within the area tolerance)."""
        P = self.P
        out = []
        for s in self.part.options[b]:
            if abs(s.w - round(s.w)) < 1e-6 and abs(s.h - round(s.h)) < 1e-6:
                out.append(s)
        if out:
            return out
        a = float(P.area[b])
        best = None
        for w in range(1, self.W + 1):
            h = int(round(a / w))
            if h < 1 or h > self.H or abs(w * h - a) > 0.009 * a:
                continue
            key = abs(math.log(w / h) - math.log(1.5))
            if best is None or key < best[0]:
                best = (key, w, h)
        if best is None:
            return []
        _, w, h = best
        return G.orientations(G.rect_shape(float(w), float(h)))

    def place_rigid(self, order: List[int], target: np.ndarray, near: Optional[Dict[int, np.ndarray]] = None) -> List[int]:
        """Nearest free integer spot per rigid block (largest first); returns
        the blocks that found no spot. near: block -> mask its footprint should
        touch (its cluster's soft claims), CLUSTER_W per cell of distance."""
        P = self.P
        failed = []
        mib_shape = {}
        for b in sorted(order, key=lambda b: -P.area[b]):
            g = int(P.mib[b])
            shapes = mib_shape.get(g) if g and P.mib_soft[b] else None
            shapes = shapes or self._int_shapes(b)
            if g and P.mib_soft[b]:
                mib_shape[g] = shapes
            occ = (self.owner != FREE).astype(np.int64)
            I = _integral(occ)
            best = None
            code = int(P.bnd[b])
            for s in shapes:
                w, h = int(round(s.w)), int(round(s.h))
                if w > self.W or h > self.H:
                    continue
                ok = np.ones((self.H - h + 1, self.W - w + 1), dtype=bool)
                for px, py, pw, ph in s.pieces:
                    px, py, pw, ph = (int(round(v)) for v in (px, py, pw, ph))
                    y0, x0 = py, px
                    win = I[y0 + ph:y0 + ph + ok.shape[0], x0 + pw:x0 + pw + ok.shape[1]] \
                        - I[y0:y0 + ok.shape[0], x0 + pw:x0 + pw + ok.shape[1]] \
                        - I[y0 + ph:y0 + ph + ok.shape[0], x0:x0 + ok.shape[1]] \
                        + I[y0:y0 + ok.shape[0], x0:x0 + ok.shape[1]]
                    ok &= win == 0
                if not ok.any():
                    continue
                ys, xs = np.nonzero(ok)
                d = np.abs(xs + s.cx - target[b, 0]) + np.abs(ys + s.cy - target[b, 1])
                if near is not None and b in near and near[b].any():
                    grp = near[b] | np.isin(self.owner, [i for i in self.rigid_at if near.get(i) is near[b]])
                    D = ndimage.distance_transform_cdt(~grp, metric="taxicab")
                    gap = np.full(len(xs), np.inf)
                    for ox, oy in ((0, 0), (w - 1, 0), (0, h - 1), (w - 1, h - 1), (w // 2, 0), (w // 2, h - 1),
                                   (0, h // 2), (w - 1, h // 2)):
                        gap = np.minimum(gap, D[np.clip(ys + oy, 0, self.H - 1), np.clip(xs + ox, 0, self.W - 1)])
                    d = d + CLUSTER_W * np.maximum(gap - 1, 0)
                if code:
                    d = d + 1e3 * (((code & 1) and 1) * (xs > 0) + ((code & 2) and 1) * (xs + w < self.W)
                                   + ((code & 8) and 1) * (ys > 0) + ((code & 4) and 1) * (ys + h < self.H))
                k = int(np.argmin(d))
                if best is None or d[k] < best[0]:
                    best = (d[k], s, int(xs[k]), int(ys[k]))
            if best is None:
                failed.append(b)
                continue
            _, s, x, y = best
            for px, py, pw, ph in s.pieces:
                _fill(self.owner, x + px, y + py, pw, ph, b)
            self.rigid_at[b] = (s, x, y)
            self.locked.add(b)
        return failed

    # -- soft blocks ----------------------------------------------------------
    def claims(self, soft: List[int], boxes: np.ndarray, sigma: np.ndarray) -> Dict[int, Tuple[int, int, int, int]]:
        """Integer claim rectangles minus RPolyPlan overlap cuts, as boolean masks."""
        H, W = self.H, self.W
        masks = {}
        bx = {}
        for i in soft:
            x0, y0, x1, y1 = boxes[i]
            x0, x1 = int(np.clip(round(x0), 0, W)), int(np.clip(round(x1), 0, W))
            y0, y1 = int(np.clip(round(y0), 0, H)), int(np.clip(round(y1), 0, H))
            bx[i] = (x0, y0, max(x1, x0 + 1), max(y1, y0 + 1))
            m = np.zeros((H, W), dtype=bool)
            m[bx[i][1]:bx[i][3], bx[i][0]:bx[i][2]] = True
            masks[i] = m
        for a_, i in enumerate(soft):
            for j in soft[a_ + 1:]:
                ix0, iy0, ix1, iy1 = bx[i]
                jx0, jy0, jx1, jy1 = bx[j]
                ox0, oy0, ox1, oy1 = max(ix0, jx0), max(iy0, jy0), min(ix1, jx1), min(iy1, jy1)
                if ox0 >= ox1 or oy0 >= oy1:
                    continue
                dx = ((ix0 + ix1) - (jx0 + jx1)) / 2
                dy = ((iy0 + iy1) - (jy0 + jy1)) / 2
                s = float(sigma[i, j])
                if abs(dx) / (ix1 - ix0 + jx1 - jx0) >= abs(dy) / (iy1 - iy0 + jy1 - jy0):
                    if dx <= 0:          # i left of j: i keeps the left s share
                        c = ox0 + int(round(s * (ox1 - ox0)))
                        masks[i][oy0:oy1, c:ox1] = False
                        masks[j][oy0:oy1, ox0:c] = False
                    else:
                        c = ox1 - int(round(s * (ox1 - ox0)))
                        masks[i][oy0:oy1, ox0:c] = False
                        masks[j][oy0:oy1, c:ox1] = False
                else:
                    if dy <= 0:
                        c = oy0 + int(round(s * (oy1 - oy0)))
                        masks[i][c:oy1, ox0:ox1] = False
                        masks[j][oy0:c, ox0:ox1] = False
                    else:
                        c = oy1 - int(round(s * (oy1 - oy0)))
                        masks[i][oy0:c, ox0:ox1] = False
                        masks[j][c:oy1, ox0:ox1] = False
        return masks

    def _targets(self, soft, nfree):
        """Integer cell counts per soft block summing to the free cells, each
        within 0.9 % of its area."""
        P = self.P
        target = np.array([max(1, int(round(P.area[i]))) for i in soft])
        diff = nfree - int(target.sum())
        order = np.argsort(-target)
        k = 0
        while diff != 0 and k < 20 * len(soft):
            i = order[k % len(soft)]
            step = 1 if diff > 0 else -1
            if abs(target[i] + step - P.area[soft[i]]) <= 0.009 * P.area[soft[i]]:
                target[i] += step
                diff -= step
            k += 1
        return target

    def _free_graph(self, rr, cc):
        """4-neighbour graph over the free cells (index = position in rr, cc)."""
        from scipy.sparse import coo_matrix
        idx = np.full((self.H, self.W), -1)
        idx[rr, cc] = np.arange(len(rr))
        a, b = [], []
        for dy, dx in ((0, 1), (1, 0)):
            src = idx[:self.H - dy, :self.W - dx]
            dst = idx[dy:, dx:]
            m = (src >= 0) & (dst >= 0)
            a.append(src[m]); b.append(dst[m])
        a, b = np.concatenate(a), np.concatenate(b)
        n = len(rr)
        return coo_matrix((np.ones(2 * len(a)), (np.concatenate([a, b]), np.concatenate([b, a]))), shape=(n, n)).tocsr(), idx

    def _cost(self, i, claim, centre, rr, cc, graph=None, idx=None):
        """Distance through free cells to the main piece of i's claim (0 inside),
        plus a small pull to the claim centre that keeps cut lines straight."""
        from scipy.sparse.csgraph import dijkstra
        m = claim[i] & (self.owner == FREE)
        tie = 1e-3 * np.hypot(cc + 0.5 - centre[i, 0], rr + 0.5 - centre[i, 1])     # Euclidean: |dx|+|dy| tied whole zones
        if m.any():
            lab, k = ndimage.label(m)
            if k > 1:
                sizes = ndimage.sum(m, lab, range(1, k + 1))
                m = lab == int(np.argmax(sizes)) + 1
            src = idx[m]
            d = dijkstra(graph, indices=src[src >= 0], min_only=True, unweighted=True)
            d = np.where(np.isfinite(d), d, 1e6)
        else:
            d = np.abs(cc + 0.5 - centre[i, 0]) + np.abs(rr + 0.5 - centre[i, 1])
        return d + tie

    def _cost_mask(self, m, c0, rr, cc, graph, idx):
        """_cost for an explicit claim mask m with centre c0."""
        from scipy.sparse.csgraph import dijkstra
        tie = 1e-3 * np.hypot(cc + 0.5 - c0[0], rr + 0.5 - c0[1])
        src = idx[m]
        src = src[src >= 0]
        if not len(src):
            return np.abs(cc + 0.5 - c0[0]) + np.abs(rr + 0.5 - c0[1]) + tie
        d = dijkstra(graph, indices=src, min_only=True, unweighted=True)
        return np.where(np.isfinite(d), d, 1e6) + tie

    @staticmethod
    def price_assign(cost: np.ndarray, target: np.ndarray, K: int = 8, D: float = 3.0) -> np.ndarray:
        """Exact-count assignment (transportation LP, HiGHS dual simplex: a vertex
        solution is integral). Cells whose best block beats the second by more
        than D keep it; the contested rest (K cheapest blocks each) go to the LP,
        D growing until it is feasible. 10-21: price (dual) updates left
        thousands of cells off on integer-cost ties."""
        from scipy.optimize import linprog
        from scipy.sparse import csr_matrix, vstack
        N, m = cost.shape
        if m == 1:
            return np.zeros(N, dtype=int)
        K = min(K, m)
        order = np.argsort(cost, axis=1)[:, :K]
        ar = np.arange(N)
        gap = cost[ar, order[:, 1]] - cost[ar, order[:, 0]]
        for d in (D, 2 * D, 4 * D, 8 * D, 16 * D, np.inf):
            cont = gap <= d
            a = order[:, 0].copy()
            rest = target - np.bincount(a[~cont], minlength=m)
            if (rest < 0).any():
                continue
            idx = np.nonzero(cont)[0]
            if len(idx) == 0:
                if not rest.any():
                    return a
                continue
            rows = np.repeat(np.arange(len(idx)), K)
            cols = order[idx].ravel()
            c = cost[idx[rows], cols]
            nv = len(c)
            A = vstack([csr_matrix((np.ones(nv), (rows, np.arange(nv))), shape=(len(idx), nv)),
                        csr_matrix((np.ones(nv), (cols, np.arange(nv))), shape=(m, nv))]).tocsr()
            res = linprog(c, A_eq=A, b_eq=np.concatenate([np.ones(len(idx)), rest]), bounds=(0, 1),
                          method="highs-ds")
            if res.status != 0:
                continue
            x = res.x.reshape(len(idx), K)
            a[idx] = order[idx, np.argmax(x, 1)]
            return a
        return order[:, 0]

    def assign(self, soft: List[int], claim: Dict[int, np.ndarray], centre: np.ndarray):
        """Exact-count assignment of every free cell, cluster-aware: each
        cluster's soft members first share one item (union of claims, summed
        count), so the group gets one region; members then split it."""
        P = self.P
        free = self.owner == FREE
        rr, cc = np.nonzero(free)
        soft = list(soft)
        target = self._targets(soft, len(rr))
        self.target = dict(zip(soft, target.tolist()))
        sset = set(soft)
        items, seen = [], set()
        for grp in P.groups:
            m = [int(b) for b in grp if int(b) in sset and int(b) not in seen]
            if len(m) > 1:
                items.append(m)
                seen.update(m)
        items += [[b] for b in soft if b not in seen]
        graph, gidx = self._free_graph(rr, cc)
        cost_b = {i: self._cost(i, claim, centre, rr, cc, graph, gidx) for i in soft}
        cols = []
        for it in items:
            if len(it) == 1:
                cols.append(cost_b[it[0]])
                continue
            # one basin: the connected part of the members' claims around the largest member
            big = max(it, key=lambda b: self.target[b])
            U = np.zeros((self.H, self.W), dtype=bool)
            for b in it:
                U |= claim[b]
            U &= self.owner == FREE
            lab_, _ = ndimage.label(U)
            core = claim[big] & (self.owner == FREE)
            keep = set(np.unique(lab_[core]).tolist()) - {0}
            M = np.isin(lab_, list(keep)) if keep else core
            cols.append(self._cost_mask(M, centre[big], rr, cc, graph, gidx))
        icost = np.stack(cols, 1)
        itgt = np.array([sum(self.target[b] for b in it) for it in items])
        a = self.price_assign(icost, itgt)
        lab = np.full(len(rr), -1)
        for k, it in enumerate(items):
            sel = np.nonzero(a == k)[0]
            if len(it) == 1:
                lab[sel] = it[0]
                continue
            sub = np.stack([cost_b[b][sel] for b in it], 1)
            tg = np.array([self.target[b] for b in it])
            lab[sel] = np.array(it)[self.price_assign(sub, tg)]
        self.owner[rr, cc] = np.where(lab >= 0, lab, FREE)
        self._cost_rows = (rr, cc, np.stack([cost_b[b] for b in soft], 1), soft)
        self.set_ranges()

    def grow(self, idx, rr, cc, costs, seeds, targets, allowed) -> np.ndarray:
        """Region growing with a shared priority queue: item k starts at cell
        seeds[k] and takes 4-adjacent allowed cells in order of costs[k] until it
        has targets[k]; every region is one piece. Returns item per cell (-1 left)."""
        import heapq
        lab = np.full(len(rr), -1)
        cnt = np.zeros(len(costs), dtype=int)
        heap = []
        for k, sd in enumerate(seeds):
            if allowed[sd]:
                heapq.heappush(heap, (float(costs[k][sd]), k, sd))
        H, W = self.H, self.W
        while heap:
            c, k, cell = heapq.heappop(heap)
            if lab[cell] >= 0 or cnt[k] >= targets[k]:
                continue
            lab[cell] = k
            cnt[k] += 1
            r, q = rr[cell], cc[cell]
            for y, x in ((r - 1, q), (r + 1, q), (r, q - 1), (r, q + 1)):
                if 0 <= y < H and 0 <= x < W:
                    j = idx[y, x]
                    if j >= 0 and lab[j] < 0 and allowed[j]:
                        heapq.heappush(heap, (float(costs[k][j]), k, j))
        return lab

    def fix_pinches(self) -> int:
        """Corner-only self contacts (2x2 diagonal pattern) of a soft block make an
        invalid polygon; give the block one of the two other cells when simple."""
        fixed = 0
        o = self.owner
        H, W = o.shape
        a, b, c, d = o[:-1, :-1], o[:-1, 1:], o[1:, :-1], o[1:, 1:]
        for diag in ((a == d) & (a != b) & (a != c), (b == c) & (b != a) & (b != d)):
            for r, cidx in zip(*np.nonzero(diag)):
                blk = o[r, cidx] if diag is not None and o[r, cidx] == o[r + 1, cidx + 1] else o[r, cidx + 1]
                blk = int(blk)
                if blk < 0 or blk in self.locked:
                    continue
                for (y, x) in ((r, cidx), (r, cidx + 1), (r + 1, cidx), (r + 1, cidx + 1)):
                    v = int(o[y, x])
                    if v == blk or v < 0 or v in self.locked:
                        continue
                    if _simple(_window(o, y, x, v)) and _simple(_window(o, y, x, blk)):
                        o[y, x] = blk
                        fixed += 1
                        break
        return fixed

    def cell_cost(self, i, r, c):
        rr, cc, cost, soft = self._cost_rows
        if not hasattr(self, "_cost_grid"):
            self._cost_grid = np.zeros((len(soft), self.H, self.W))
            for k in range(len(soft)):
                self._cost_grid[k, rr, cc] = cost[:, k]
            self._k = {b: k for k, b in enumerate(soft)}
        return self._cost_grid[self._k[i], r, c]

    # -- moves ----------------------------------------------------------------
    def neighbours(self, u):
        """Soft blocks 4-adjacent to block u."""
        m = self.owner == u
        sh = np.zeros_like(m)
        sh[1:] |= m[:-1]; sh[:-1] |= m[1:]; sh[:, 1:] |= m[:, :-1]; sh[:, :-1] |= m[:, 1:]
        v = np.unique(self.owner[sh & ~m])
        return [int(x) for x in v if x >= 0 and int(x) not in self.locked]

    def counts(self) -> np.ndarray:
        o = self.owner[self.owner >= 0]
        return np.bincount(o, minlength=self.P.n)

    def set_ranges(self):
        """Allowed cell counts per soft block: within 0.9 % of the area (the
        contest allows 1 %), so short blocks leave a little white space."""
        P = self.P
        self.lo = {i: int(math.ceil(P.area[i] * 0.991 - 1e-9)) for i in self.target}
        self.hi = {i: max(self.lo[i], int(math.floor(P.area[i] * 1.009 + 1e-9))) for i in self.target}

    def _move_ok(self, r, c, u, v) -> bool:
        """Cell (r, c) from u to v keeps both blocks one hole-free piece (FREE
        and locked sides are not checked)."""
        if u >= 0 and not _simple(_window(self.owner, r, c, u)):
            return False
        if v >= 0 and not _simple(_window(self.owner, r, c, v)):
            return False
        gu, gv = self.group_of.get(u), self.group_of.get(v)
        if gu is not None and gu is not gv and not _simple(_window_set(self.owner, r, c, gu)):
            return False
        if gv is not None and gv is not gu and not _simple(_window_set(self.owner, r, c, gv)):
            return False
        return True

    def _corner_delta(self, r, c, u, v) -> int:
        before = (_corners_at(self.owner, r, c, u) if u >= 0 else 0) + (_corners_at(self.owner, r, c, v) if v >= 0 else 0)
        self.owner[r, c] = v
        after = (_corners_at(self.owner, r, c, u) if u >= 0 else 0) + (_corners_at(self.owner, r, c, v) if v >= 0 else 0)
        self.owner[r, c] = u
        return after - before

    def transfer(self, u, v) -> bool:
        """Move one cell from u to v (u or v may be FREE): a u cell 4-adjacent
        to v, simple for both, fewest added corners then lowest cost change."""
        mu = self.owner == u
        mv = self.owner == v
        if v == FREE and u >= 0:          # white space can open anywhere on u's boundary
            mv = np.pad(~mu, 1, constant_values=True)
            adj = mv[:-2, 1:-1] | mv[2:, 1:-1] | mv[1:-1, :-2] | mv[1:-1, 2:]
        else:
            adj = np.zeros_like(mv)
            adj[1:] |= mv[:-1]; adj[:-1] |= mv[1:]; adj[:, 1:] |= mv[:, :-1]; adj[:, :-1] |= mv[:, 1:]
        rs, cs = np.nonzero(mu & adj)
        if u >= 0 and mu.sum() <= 1:
            return False
        best = None
        for r, c in zip(rs, cs):
            if not self._move_ok(r, c, u, v):
                continue
            key = CORNER_W * self._corner_delta(r, c, u, v)
            if v >= 0:
                key += self.cell_cost(v, r, c)
            if u >= 0:
                key -= self.cell_cost(u, r, c)
            if best is None or key < best[0]:
                best = (key, r, c)
        if best is None:
            return False
        self.owner[best[1], best[2]] = v
        return True

    def _free_neighbours(self, u):
        m = self.owner == u
        sh = np.zeros_like(m)
        sh[1:] |= m[:-1]; sh[:-1] |= m[1:]; sh[:, 1:] |= m[:, :-1]; sh[:, :-1] |= m[:, 1:]
        return bool((sh & (self.owner == FREE)).any())

    def balance(self, max_moves: int = 5000) -> int:
        """Every soft block into its count range: a short block takes a free
        cell or a cell passed along the shortest chain from a block with
        spare; a long block passes one to a neighbour with room or leaves it
        free. Returns moves made, -1 if some block stays out of range."""
        moves = 0
        banned = set()
        while moves < max_moves:
            cnt = self.counts()
            short = [i for i in self.target if cnt[i] < self.lo[i]]
            long_ = [i for i in self.target if cnt[i] > self.hi[i]]
            if not short and not long_:
                return moves
            progress = False
            for d in short:
                if self._free_neighbours(d) and self.transfer(FREE, d):
                    moves += 1
                    progress = True
                    continue
                # BFS from d to the nearest block with spare (count above its lower bound)
                # or with white space beside it (FREE: takes a free cell first)
                prev, frontier, found = {d: None}, [d], None
                while frontier and found is None:
                    nxt = []
                    for u in frontier:
                        if u != d and self._free_neighbours(u):
                            prev[FREE] = u
                            found = FREE
                            break
                        for v in self.neighbours(u):
                            if v in prev or (v, u) in banned:
                                continue
                            prev[v] = u
                            if cnt[v] > self.lo[v]:
                                found = v
                                break
                            nxt.append(v)
                        if found is not None:
                            break
                    frontier = nxt
                if found is None:
                    continue
                path = [found]
                while prev[path[-1]] is not None:
                    path.append(prev[path[-1]])
                for u, v in zip(path, path[1:]):          # spare block -> ... -> d
                    if not self.transfer(u, v):
                        banned.add((u, v))
                        break
                    moves += 1
                    progress = True
            for u in long_:
                # BFS from u to the nearest block with room; else open white space
                prev, frontier, found = {u: None}, [u], None
                while frontier and found is None:
                    nxt = []
                    for a_ in frontier:
                        for v in self.neighbours(a_):
                            if v in prev or (a_, v) in banned:
                                continue
                            prev[v] = a_
                            if cnt[v] < self.hi[v]:
                                found = v
                                break
                            nxt.append(v)
                        if found is not None:
                            break
                    frontier = nxt
                ok = False
                if found is not None:
                    path = [found]
                    while prev[path[-1]] is not None:
                        path.append(prev[path[-1]])
                    path = path[::-1]                       # u -> ... -> found
                    ok = True
                    for a_, v in zip(path, path[1:]):
                        if not self.transfer(a_, v):
                            banned.add((a_, v))
                            ok = False
                            break
                        moves += 1
                if not ok and self.transfer(u, FREE):
                    ok = True
                    moves += 1
                progress |= ok
            if not progress:
                return -1
        return -1

    def corner_fix(self, max_moves: int = 3000, plateau: int = 400) -> int:
        """Blocks over the corner cap: single-cell moves into or out of the
        block (to / from a neighbour or white space) that cut its corners,
        keeping every count in range."""
        moves = 0
        cap = PT.MAX_CORNERS
        cnt = self.counts()
        for b in list(self.target):
            tabu = {}                     # cell -> move index; no cell moves twice within TABU moves
            flat = 0
            while moves < max_moves and corners(self.owner == b) > cap:
                mb = self.owner == b
                ring = ndimage.binary_dilation(mb, structure=[[0, 1, 0], [1, 1, 1], [0, 1, 0]]) & ~mb
                inner = mb & ndimage.binary_dilation(~mb, structure=[[0, 1, 0], [1, 1, 1], [0, 1, 0]])
                best = None
                for r, c in zip(*np.nonzero(ring)):             # take a cell in
                    v = int(self.owner[r, c])
                    if v >= 0 and (v in self.locked or v not in self.target or cnt[v] - 1 < self.lo[v]):
                        continue
                    if cnt[b] + 1 > self.hi[b] or moves - tabu.get((r, c), -99) < TABU or not self._move_ok(r, c, v, b):
                        continue
                    dl = self._corner_delta(r, c, v, b) + 1e-3 * (cnt[b] + 1 - self.target[b]) ** 2
                    if best is None or dl < best[0]:
                        best = (dl, r, c, v, b)
                for r, c in zip(*np.nonzero(inner)):            # give a cell away
                    if cnt[b] - 1 < self.lo[b]:
                        break
                    for (rr, cc_) in ((r - 1, c), (r + 1, c), (r, c - 1), (r, c + 1)):
                        if not (0 <= rr < self.H and 0 <= cc_ < self.W):
                            continue
                        v = int(self.owner[rr, cc_])
                        if v == b or (v >= 0 and (v in self.locked or v not in self.target or cnt[v] + 1 > self.hi[v])):
                            continue
                        if moves - tabu.get((r, c), -99) < TABU or not self._move_ok(r, c, b, v):
                            continue
                        dl = self._corner_delta(r, c, b, v) + 1e-3 * (cnt[b] - 1 - self.target[b]) ** 2
                        if best is None or dl < best[0]:
                            best = (dl, r, c, b, v)
                # neutral moves (corner shifts along a long step) allowed up to `plateau` times
                if best is None or best[0] > 0.5 or (best[0] > -0.5 and flat >= plateau):
                    break
                flat += best[0] > -0.5
                _, r, c, u, v = best
                tabu[(r, c)] = moves
                self.owner[r, c] = v
                if u >= 0:
                    cnt[u] -= 1
                if v >= 0:
                    cnt[v] += 1
                moves += 1
        return moves

    def refill(self) -> int:
        """Stray pieces of soft blocks become white space, then blocks below
        their target grow back into adjacent white space, cheapest cells first
        (growth from a block's edge keeps it one piece). Returns cells dropped."""
        import heapq
        dropped = 0
        for i in list(self.target):
            m = self.owner == i
            lab, k = ndimage.label(m)
            if k <= 1:
                continue
            sizes = ndimage.sum(m, lab, range(1, k + 1))
            keep = int(np.argmax(sizes)) + 1
            stray = m & (lab != keep)
            self.owner[stray] = FREE
            dropped += int(stray.sum())
        cnt = self.counts()
        heap = []
        H, W = self.H, self.W

        def push_around(r, c):
            for y, x in ((r - 1, c), (r + 1, c), (r, c - 1), (r, c + 1)):
                if 0 <= y < H and 0 <= x < W and self.owner[y, x] == FREE:
                    for b in {int(self.owner[yy, xx]) for yy, xx in ((y - 1, x), (y + 1, x), (y, x - 1), (y, x + 1))
                              if 0 <= yy < H and 0 <= xx < W and int(self.owner[yy, xx]) in self.target}:
                        if cnt[b] < self.target[b]:
                            heapq.heappush(heap, (self.cell_cost(b, y, x), b, y, x))

        for r, c in zip(*np.nonzero(self.owner == FREE)):
            for b in {int(self.owner[yy, xx]) for yy, xx in ((r - 1, c), (r + 1, c), (r, c - 1), (r, c + 1))
                      if 0 <= yy < H and 0 <= xx < W and int(self.owner[yy, xx]) in self.target}:
                if cnt[b] < self.target[b]:
                    heapq.heappush(heap, (self.cell_cost(b, r, c), b, r, c))
        while heap:
            _, b, r, c = heapq.heappop(heap)
            if self.owner[r, c] != FREE or cnt[b] >= self.target[b]:
                continue
            if not self._move_ok(r, c, FREE, b):
                continue
            self.owner[r, c] = b
            cnt[b] += 1
            for y, x in ((r - 1, c), (r + 1, c), (r, c - 1), (r, c + 1)):
                if 0 <= y < H and 0 <= x < W and self.owner[y, x] == FREE:
                    heapq.heappush(heap, (self.cell_cost(b, y, x), b, y, x))
        return dropped

    def _block_ok(self, b, cnt=None) -> bool:
        m = self.owner == b
        if not m.any() or ndimage.label(m)[1] != 1 or ndimage.binary_fill_holes(m).sum() != m.sum():
            return False
        if corners(m) > PT.MAX_CORNERS:
            return False
        c = int(m.sum()) if cnt is None else cnt[b]
        return self.lo[b] <= c <= self.hi[b]

    def relegalize(self, b) -> bool:
        """Local re-solve around soft block b: b and its soft neighbours share
        their cells (plus adjacent white space); b's cost is the distance to a
        rectangle of its target area at its centroid, a neighbour's the distance
        to its current cells; exact-count assignment, kept only if every block
        involved comes out valid."""
        from scipy.sparse.csgraph import dijkstra
        nbrs = self.neighbours(b)
        blocks = [b] + [v for v in nbrs if v in self.target]
        M = np.isin(self.owner, blocks)
        M |= (self.owner == FREE) & ndimage.binary_dilation(self.owner == b)
        rr, cc = np.nonzero(M)
        if len(rr) < 2:
            return False
        cnt = self.counts()
        tgt = np.array([self.target[b]] + [int(cnt[v]) for v in blocks[1:]])
        tgt[0] = min(max(self.lo[b], len(rr) - int(tgt[1:].sum())), self.hi[b])
        if tgt.sum() > len(rr):
            return False
        mb = self.owner == b
        ys, xs = np.nonzero(mb)
        cy, cx = ys.mean() + 0.5, xs.mean() + 0.5
        bw, bh = xs.max() - xs.min() + 1, ys.max() - ys.min() + 1
        ar = min(max(bh / bw, 1 / 3), 3)
        w = math.sqrt(self.target[b] / ar)
        h = self.target[b] / w
        qx0, qx1, qy0, qy1 = cx - w / 2, cx + w / 2, cy - h / 2, cy + h / 2
        costs = [np.maximum(0, np.maximum(qx0 - (cc + 0.5), (cc + 0.5) - qx1)) +
                 np.maximum(0, np.maximum(qy0 - (rr + 0.5), (rr + 0.5) - qy1)) + 1e-3 * np.hypot(cc + 0.5 - cx, rr + 0.5 - cy)]
        graph, idx = self._free_graph_mask(M)
        for v in blocks[1:]:
            src = idx[self.owner == v]
            src = src[src >= 0]
            d = dijkstra(graph, indices=src, min_only=True, unweighted=True)
            vy, vx = np.nonzero(self.owner == v)
            costs.append(np.where(np.isfinite(d), d, 1e6)[idx[rr, cc]] + 1e-3 * np.hypot(cc + 0.5 - vx.mean(), rr + 0.5 - vy.mean()))
        free_slack = len(rr) - int(tgt.sum())
        if free_slack:
            costs.append(np.full(len(rr), 50.0))         # leftover cells stay white space
            tgt = np.append(tgt, free_slack)
        a = self.price_assign(np.stack(costs, 1), tgt)
        old = self.owner[rr, cc].copy()
        labels = np.array(blocks + ([FREE] if free_slack else []))
        self.owner[rr, cc] = labels[a]
        if all(self._block_ok(v) for v in blocks) and self.soft_viol(blocks) <= self._viol_before:
            return True
        self.owner[rr, cc] = old
        return False

    def _free_graph_mask(self, M):
        from scipy.sparse import coo_matrix
        rr, cc = np.nonzero(M)
        idx = np.full((self.H, self.W), -1)
        idx[rr, cc] = np.arange(len(rr))
        a, b = [], []
        for dy, dx in ((0, 1), (1, 0)):
            src = idx[:self.H - dy, :self.W - dx]
            dst = idx[dy:, dx:]
            m = (src >= 0) & (dst >= 0)
            a.append(src[m]); b.append(dst[m])
        a, b = np.concatenate(a), np.concatenate(b)
        n = len(rr)
        return coo_matrix((np.ones(2 * len(a)), (np.concatenate([a, b]), np.concatenate([b, a]))), shape=(n, n)).tocsr(), idx

    def fix_bad_blocks(self, rounds: int = 3) -> int:
        """relegalize every soft block that is out of range, over the corner
        cap, split or holed."""
        fixed = 0
        for _ in range(rounds):
            bad = [b for b in self.target if not self._block_ok(b)]
            if not bad:
                break
            for b in bad:
                if self._block_ok(b):
                    continue
                self._viol_before = self.soft_viol([b] + self.neighbours(b))
                fixed += self.relegalize(b)
        return fixed

    def fix_fragments(self) -> int:
        """Stray pieces of a soft block go to the neighbour sharing most of their boundary."""
        fixed = 0
        for i in list(self.target):
            m = self.owner == i
            lab, k = ndimage.label(m)
            if k <= 1:
                continue
            sizes = ndimage.sum(m, lab, range(1, k + 1))
            keep = int(np.argmax(sizes)) + 1
            for c in range(1, k + 1):
                if c == keep:
                    continue
                piece = lab == c
                ring = ndimage.binary_dilation(piece) & ~piece
                vals = self.owner[ring]
                vals = vals[(vals >= 0) & (vals != i)]
                vals = [v for v in vals if int(v) not in self.locked]
                if not vals:
                    continue
                mates = [v for v in vals if self.group_of.get(i) is not None and int(v) in self.group_of[i]]
                v = np.bincount(mates or vals).argmax()
                self.owner[piece] = v
                fixed += 1
        return fixed

    # -- stage 3: cleanup moves ----------------------------------------------
    def centroids(self):
        """Centroids [n, 2] of every block in the current layout."""
        P = self.P
        c = np.zeros((P.n, 2))
        for i, (s, x, y) in self.rigid_at.items():
            c[i] = (x + s.cx, y + s.cy)
        lab = self.owner
        ok = lab >= 0
        rr, cc = np.nonzero(ok)
        ids = lab[ok]
        cnt = np.bincount(ids, minlength=P.n)
        sx = np.bincount(ids, cc + 0.5, minlength=P.n)
        sy = np.bincount(ids, rr + 0.5, minlength=P.n)
        m = cnt > 0
        free_ids = [i for i in range(P.n) if i not in self.rigid_at]
        for i in free_ids:
            if m[i]:
                c[i] = (sx[i] / cnt[i], sy[i] / cnt[i])
        return c

    def _recut_options(self, u, v):
        """The union of u and v re-split by one straight cut (x or y order,
        either side first) with u keeping its exact count; yields (sub-owner
        window, (y0, x0)) for options where both blocks stay one hole-free piece."""
        M = (self.owner == u) | (self.owner == v)
        rs, cs = np.nonzero(M)
        y0, y1, x0, x1 = rs.min(), rs.max() + 1, cs.min(), cs.max() + 1
        win = self.owner[y0:y1, x0:x1]
        Mw = M[y0:y1, x0:x1]
        r, c = np.nonzero(Mw)
        nu = int((win == u).sum())
        for key in ((c, r), (r, c)):
            order = np.lexsort((key[1], key[0]))
            for rev in (False, True):
                o = order[::-1] if rev else order
                new = win.copy()
                new[r[o[:nu]], c[o[:nu]]] = u
                new[r[o[nu:]], c[o[nu:]]] = v
                ok = True
                for b in (u, v):
                    m = new == b
                    if ndimage.label(m)[1] != 1 or ndimage.binary_fill_holes(m).sum() != m.sum():
                        ok = False
                        break
                if ok:
                    yield new, (y0, x0)

    def recut(self, u, v, hp_ref: float, cw: float = 1.0) -> bool:
        """Best re-cut of the u/v pair if it lowers J = HPWL / hp_ref +
        cw * (excess corners over the cap) + 0.01 * corners."""
        P = self.P
        cap = PT.MAX_CORNERS

        def j_of(owner_win, y0, x0):
            sub_u, sub_v = owner_win == u, owner_win == v
            cu, cv = corners(sub_u), corners(sub_v)
            return cw * (max(0, cu - cap) + max(0, cv - cap)) + 0.01 * (cu + cv) + SOFT_W * self.soft_viol((u, v))

        M = (self.owner == u) | (self.owner == v)
        rs, cs = np.nonzero(M)
        y0, x0 = rs.min(), cs.min()
        cur_win = self.owner[y0:rs.max() + 1, x0:cs.max() + 1].copy()
        c0 = self.centroids()
        j0 = j_of(cur_win, y0, x0) + P.hpwl(c0[:, 0], c0[:, 1]) / hp_ref
        best = None
        for new, (yy, xx) in self._recut_options(u, v):
            h, w = new.shape
            old = self.owner[yy:yy + h, xx:xx + w].copy()
            self.owner[yy:yy + h, xx:xx + w] = new
            c1 = self.centroids()
            j1 = j_of(new, yy, xx) + P.hpwl(c1[:, 0], c1[:, 1]) / hp_ref
            self.owner[yy:yy + h, xx:xx + w] = old
            if j1 < j0 - 1e-9 and (best is None or j1 < best[0]):
                best = (j1, new, yy, xx)
        if best is None:
            return False
        _, new, yy, xx = best
        h, w = new.shape
        self.owner[yy:yy + h, xx:xx + w] = new
        return True

    def soft_viol(self, blocks) -> int:
        """Cluster pieces beyond one (groups containing these blocks) plus
        edge-pin misses of these blocks, in the current layout."""
        P = self.P
        v = 0
        for grp in P.groups:
            if any(b in grp for b in blocks):
                m = np.isin(self.owner, grp)
                v += ndimage.label(m)[1] - 1
        for b in blocks:
            code = int(P.bnd[b])
            if code:
                m = self.owner == b
                v += ((code & 1) and not m[:, 0].any()) + ((code & 2) and not m[:, -1].any()) \
                    + ((code & 8) and not m[0].any()) + ((code & 4) and not m[-1].any())
        return v

    def adjacent_pairs(self):
        a, b = self.owner[:, :-1], self.owner[:, 1:]
        c, d = self.owner[:-1, :], self.owner[1:, :]
        pairs = set()
        for p, q in ((a, b), (c, d)):
            m = (p != q) & (p >= 0) & (q >= 0)
            for u, v in zip(p[m].tolist(), q[m].tolist()):
                if u not in self.locked and v not in self.locked:
                    pairs.add((min(u, v), max(u, v)))
        return sorted(pairs)

    def cleanup(self, passes: int = 4, all_pairs: bool = False) -> int:
        """Re-cut passes over neighbouring pairs: first the pairs where a block
        is over the corner cap, then (all_pairs) every pair for wirelength."""
        c0 = self.centroids()
        hp_ref = max(self.P.hpwl(c0[:, 0], c0[:, 1]), 1e-9)
        done = 0
        for _ in range(passes):
            changed = 0
            over = {i for i in self.target if corners(self.owner == i) > PT.MAX_CORNERS}
            for u, v in self.adjacent_pairs():
                if all_pairs or u in over or v in over:
                    if self.recut(u, v, hp_ref):
                        changed += 1
                        over = {i for i in self.target if corners(self.owner == i) > PT.MAX_CORNERS}
            done += changed
            if not changed:
                break
        return done

    # -- output ---------------------------------------------------------------
    def polygons(self) -> List[list]:
        P = self.P
        out = []
        for i in range(P.n):
            if i in self.rigid_at:
                s, x, y = self.rigid_at[i]
                out.append(list(s.polygon(x, y).exterior.coords[:-1]))
                continue
            m = self.owner == i
            if not m.any():
                out.append([])
                continue
            rects = []
            for r in np.nonzero(m.any(1))[0]:          # row runs
                row = m[r]
                d = np.diff(np.concatenate([[0], row.astype(np.int8), [0]]))
                for a, b in zip(np.nonzero(d == 1)[0], np.nonzero(d == -1)[0]):
                    rects.append(sbox(a, r, b, r + 1))
            g = unary_union(rects).simplify(0)
            if g.geom_type != "Polygon":
                g = max(g.geoms, key=lambda q: q.area)
            out.append(list(g.exterior.coords[:-1]))
        return out

    def stats(self) -> dict:
        frag = holes = over16 = 0
        for i in self.target:
            m = self.owner == i
            lab, k = ndimage.label(m)
            frag += k > 1
            holes += int(ndimage.binary_fill_holes(m).sum() - m.sum() > 0)
            over16 += corners(m) > PT.MAX_CORNERS
        cnt = self.counts()
        miss = sum(not (self.lo[i] <= cnt[i] <= self.hi[i]) for i in self.target)
        return dict(fragmented=frag, holes=holes, over_corners=over16, wrong_count=miss,
                    free=int((self.owner == FREE).sum()))

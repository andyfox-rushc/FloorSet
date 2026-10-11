"""
Constraint-aware repair (roadmap step 2, 10-05), run after the re-pack /
slicing boundary snaps (rl/repack.py _fix_boundary).

Diagnosis on 100 best-of results (scripts/bestof_diag.py): 71 of 112 edge
violations were PREPLACED blocks -- they can't move, so they're only on
their edge if nothing sticks out past it. The ground-truth bbox starts at
(0, 0), so pinned preplaced blocks reveal walls: a left/bottom pin means
x/y >= 0, a right/top pin fixes the true width/height. The other big loss
was cluster members cut off from their cluster (100 violations).

`repair` greedily moves blocks to fix both, keeping a move only if the
fast contest cost (rl/repack.py _Cost, through the caller's `evaluate`,
which re-legalizes) drops:
  - blocks past a wall go into free holes inside the walls (soft blocks
    also try other shapes) or onto the free side of the layout;
  - cut-off cluster members go next to the cluster's largest component.
"""
import math
import time

import numpy as np

from . import repack as R

GRID = 64          # raster resolution for hole finding (cells along the longer side)
SHAPES = (1.0, 0.5, 0.7, 1.4, 2.0)


def walls(P):
    """(x_lo, y_lo, x_hi, y_hi) the layout must stay inside, None = free."""
    pos, pre, bnd = P["pos"], P["preplaced"], P["bnd"]
    right, top = pos[:, 0] + pos[:, 2], pos[:, 1] + pos[:, 3]
    has = lambda bit: pre & ((bnd & bit) != 0)
    x_lo = float(pos[has(1), 0].min()) if has(1).any() else None
    y_lo = float(pos[has(8), 1].min()) if has(8).any() else None
    x_hi = float(right[has(2)].max()) if has(2).any() else None
    y_hi = float(top[has(4)].max()) if has(4).any() else None
    return x_lo, y_lo, x_hi, y_hi


def _ideal(P, x, y, w, h, b):
    """Weighted mean of what block b connects to (its HPWL-optimal centre)."""
    cx, cy = x + w / 2, y + h / 2
    sw = sx = sy = 0.0
    i, j, wt = P["b2b_i"], P["b2b_j"], P["b2b_w"]
    for other, mask in ((j, i == b), (i, j == b)):
        if mask.any():
            o = other[mask]
            sw += float(wt[mask].sum())
            sx += float((wt[mask] * cx[o]).sum())
            sy += float((wt[mask] * cy[o]).sum())
    m = P["p2b_b"] == b
    if m.any():
        sw += float(P["p2b_w"][m].sum())
        sx += float((P["p2b_w"][m] * P["p2b_px"][m]).sum())
        sy += float((P["p2b_w"][m] * P["p2b_py"][m]).sum())
    if sw <= 0:
        return float(cx[b]), float(cy[b])
    return sx / sw, sy / sw


def _holes(x, y, w, h, b, bw, bh, box, target, k=4):
    """Up to k lower-left corners inside `box` where a bw x bh block fits
    without overlapping any block but b, nearest `target` (a centre) first."""
    X0, Y0, X1, Y1 = box
    if bw > X1 - X0 + 1e-9 or bh > Y1 - Y0 + 1e-9:
        return []
    cell = max(X1 - X0, Y1 - Y0) / GRID
    nx, ny = max(1, int(math.ceil((X1 - X0) / cell))), max(1, int(math.ceil((Y1 - Y0) / cell)))
    occ = np.zeros((ny, nx), dtype=np.int32)
    for i in range(len(x)):
        if i == b:
            continue
        # cells touched by block i (conservative: any overlap marks the cell)
        c0 = int(math.floor((x[i] - X0) / cell + 1e-9))
        c1 = int(math.ceil((x[i] + w[i] - X0) / cell - 1e-9))
        r0 = int(math.floor((y[i] - Y0) / cell + 1e-9))
        r1 = int(math.ceil((y[i] + h[i] - Y0) / cell - 1e-9))
        c0, c1, r0, r1 = max(c0, 0), min(c1, nx), max(r0, 0), min(r1, ny)
        if c0 < c1 and r0 < r1:
            occ[r0:r1, c0:c1] = 1
    kw, kh = int(math.ceil(bw / cell - 1e-9)), int(math.ceil(bh / cell - 1e-9))
    if kw > nx or kh > ny:
        return []
    S = np.zeros((ny + 1, nx + 1), dtype=np.int32)
    S[1:, 1:] = occ.cumsum(0).cumsum(1)
    tot = S[kh:, kw:] - S[:-kh, kw:] - S[kh:, :-kw] + S[:-kh, :-kw]   # [ny-kh+1, nx-kw+1]
    rr, cc = np.nonzero(tot == 0)
    if not len(rr):
        return []
    px = np.minimum(X0 + cc * cell, X1 - bw)
    py = np.minimum(Y0 + rr * cell, Y1 - bh)
    d = np.abs(px + bw / 2 - target[0]) + np.abs(py + bh / 2 - target[1])
    out = []
    for idx in np.argsort(d):
        p = (float(px[idx]), float(py[idx]))
        if all(abs(p[0] - q[0]) + abs(p[1] - q[1]) > 2 * cell for q in out):
            out.append(p)
            if len(out) >= k:
                break
    return out


def _past_walls(x, y, w, h, wl, eps=1e-6):
    x_lo, y_lo, x_hi, y_hi = wl
    bad = np.zeros(len(x), dtype=bool)
    if x_lo is not None:
        bad |= x < x_lo - eps
    if y_lo is not None:
        bad |= y < y_lo - eps
    if x_hi is not None:
        bad |= x + w > x_hi + eps
    if y_hi is not None:
        bad |= y + h > y_hi + eps
    return np.where(bad)[0]


def _box(x, y, w, h, wl):
    """Region a moved block may go: inside the walls, else the current bbox."""
    x_lo, y_lo, x_hi, y_hi = wl
    return (x.min() if x_lo is None else x_lo, y.min() if y_lo is None else y_lo,
            (x + w).max() if x_hi is None else x_hi, (y + h).max() if y_hi is None else y_hi)


def _components(P, x, y, w, h, g):
    """Connected components (edge contact) of cluster member list g."""
    x1, y1 = x + w, y + h
    gx, gy, gx1, gy1 = x[g], y[g], x1[g], y1[g]
    ov_x = np.minimum(gx1[:, None], gx1[None, :]) - np.maximum(gx[:, None], gx[None, :])
    ov_y = np.minimum(gy1[:, None], gy1[None, :]) - np.maximum(gy[:, None], gy[None, :])
    touch = ((ov_x > R.EPS) & (np.abs(ov_y) <= R.EPS)) | ((ov_y > R.EPS) & (np.abs(ov_x) <= R.EPS))
    comp = -np.ones(len(g), dtype=int)
    k = 0
    for s in range(len(g)):
        if comp[s] >= 0:
            continue
        stack, comp[s] = [s], k
        while stack:
            a = stack.pop()
            for b in np.nonzero(touch[a])[0]:
                if comp[b] < 0:
                    comp[b] = k
                    stack.append(b)
        k += 1
    return [g[comp == c] for c in range(k)]


def _beside(x, y, w, h, a, anchors):
    """Spots for block a abutting one of `anchors` on each side (3 offsets)."""
    for b in anchors:
        for t in (0.0, 0.5, 1.0):
            yield x[b] + w[b], y[b] + t * (h[b] - h[a])
            yield x[b] - w[a], y[b] + t * (h[b] - h[a])
            yield x[b] + t * (w[b] - w[a]), y[b] + h[b]
            yield x[b] + t * (w[b] - w[a]), y[b] - h[a]


def repair(best, evaluate, cost_fn, P, time_budget: float = 0.5, rounds: int = 3):
    """Greedy wall + cluster repair on best = (cost, x, y, w, h)."""
    start = time.time()
    wl = walls(P)
    soft = ~P["fixed"] & ~P["preplaced"] & (P["area"] > 0) & (P["mib"] == 0)
    movable = ~P["preplaced"]
    groups = [np.where(P["cluster"] == g)[0] for g in range(1, int(P["cluster"].max(initial=0)) + 1)]
    groups = [g for g in groups if len(g) > 1]

    def out_of_time():
        return time.time() - start > time_budget

    def try_spots(a, spots):
        nonlocal best
        improved = False
        for sx_, sy_, sw, sh in spots:
            if out_of_time():
                break
            c0, x, y, w, h = best
            if a is None:                      # whole-layout candidate
                nx, ny = sx_, sy_
            else:
                nx, ny = x.copy(), y.copy()
                nx[a], ny[a] = sx_, sy_
            c, lx, ly, feas, _ = evaluate(nx, ny, sw, sh)
            if feas and c < best[0] - 1e-12:
                best = (c, lx.copy(), ly.copy(), sw.copy(), sh.copy())
                improved = True
        return improved

    def shapes(a, w, h):
        out = [(w, h)]
        if soft[a]:
            ar = w[a] * h[a]
            for f in SHAPES[1:]:
                nw = min(max(w[a] * f, math.sqrt(ar / 8.0)), math.sqrt(ar * 8.0))
                sw, sh = w.copy(), h.copy()
                sw[a], sh[a] = nw, ar / nw
                out.append((sw, sh))
        return out

    def joint(prefer):
        """Move every block past a wall at once (largest first), each into
        the hole nearest its ideal (prefer="ideal") or current spot."""
        c0, x, y, w, h = best
        x, y, w, h = x.copy(), y.copy(), w.copy(), h.copy()
        bad = [a for a in _past_walls(x, y, w, h, wl) if movable[a]]
        if not bad:
            return False
        box = _box(x, y, w, h, wl)
        for a in sorted(bad, key=lambda i: -w[i] * h[i]):
            tgt = _ideal(P, x, y, w, h, a) if prefer == "ideal" else (x[a] + w[a] / 2, y[a] + h[a] / 2)
            opts = []
            for sw, sh in shapes(a, w, h):
                for px, py in _holes(x, y, sw, sh, a, sw[a], sh[a], box, tgt, k=1):
                    d = abs(px + sw[a] / 2 - tgt[0]) + abs(py + sh[a] / 2 - tgt[1])
                    opts.append((d, px, py, sw[a], sh[a]))
            if opts:
                _, x[a], y[a], w[a], h[a] = min(opts)
        return try_spots(None, [(x, y, w, h)])

    for _ in range(rounds):
        changed = False
        # 1. blocks past a wall: all at once, then one at a time
        if any(v is not None for v in wl):
            for prefer in ("ideal", "current"):
                if not out_of_time():
                    changed |= joint(prefer)
            c0, x, y, w, h = best
            for a in _past_walls(x, y, w, h, wl):
                if out_of_time() or not movable[a]:
                    continue
                c0, x, y, w, h = best
                box = _box(x, y, w, h, wl)
                tgt = _ideal(P, x, y, w, h, a)
                spots = []
                for sw, sh in shapes(a, w, h):
                    spots += [(px, py, sw, sh) for px, py in _holes(x, y, sw, sh, a, sw[a], sh[a], box, tgt)]
                    # free side of the layout: on top / to the right, inside the walls
                    if wl[3] is None:
                        spots.append((min(max(x[a], box[0]), box[2] - sw[a]), (y + h).max(), sw, sh))
                    if wl[2] is None:
                        spots.append(((x + w).max(), min(max(y[a], box[1]), box[3] - sh[a]), sw, sh))
                changed |= try_spots(a, spots)
        # 2. cut-off cluster members -> beside the largest component
        for g in groups:
            if out_of_time():
                break
            c0, x, y, w, h = best
            comps = _components(P, x, y, w, h, g)
            if len(comps) < 2:
                continue
            main = max(comps, key=lambda c: float((w[c] * h[c]).sum()))
            for comp in comps:
                if comp is main:
                    continue
                for a in comp:
                    if out_of_time() or not movable[a]:
                        continue
                    c0, x, y, w, h = best
                    spots = [(px, py, sw, sh) for sw, sh in shapes(a, w, h)[:3]
                             for px, py in _beside(x, y, sw, sh, a, main)]
                    changed |= try_spots(a, spots)
        if not changed or out_of_time():
            break
    return best


# ---------------------------------------------------------------------------
# Tetris legalizer for the analytic placer (replaces the constraint-graph
# legalizer there, which resolves overlaps by pushing blocks outward and so
# ignores the walls: design 2, wall at x=102, every candidate ended 105-118
# wide).
# ---------------------------------------------------------------------------
CLUSTER_PEN = 10.0


def tetris(P, x, y, w, h, W, H, order: str = "y", res: int = 64, k_near: int = 24):
    """Place blocks one at a time at the free spot nearest their (overlapping)
    position (x, y), inside the walls / the (W, H) outline from the origin.
    Edge-pinned blocks go on their edge, cluster members prefer touching a
    placed clustermate, soft blocks try a few shapes; when nothing fits, the
    outline grows on a side without a wall. Returns (x, y, w, h) arrays."""
    n = P["n"]
    pre, fixed, bnd, cl, mib = P["preplaced"], P["fixed"], P["bnd"], P["cluster"], P["mib"]
    soft = ~fixed & ~pre & (mib == 0)
    x_lo, y_lo, x_hi, y_hi = walls(P)
    Wb = x_hi if x_hi is not None else max(W, float((x + w)[pre].max(initial=0.0)))
    Hb = y_hi if y_hi is not None else max(H, float((y + h)[pre].max(initial=0.0)))
    cell = max(Wb, Hb) / res
    nx0, ny0 = int(math.ceil(Wb / cell - 1e-9)), int(math.ceil(Hb / cell - 1e-9))
    NX, NY = 3 * nx0, 3 * ny0                         # walls are soft: growing past one costs 10x
    occ = np.zeros((NY, NX), dtype=np.int32)
    cur = [nx0, ny0]                                  # current outline in cells
    ox, oy, ow, oh = x.copy(), y.copy(), w.copy(), h.copy()

    def mark(px, py, pw, ph):
        c0, c1 = int(math.floor(px / cell + 1e-9)), int(math.ceil((px + pw) / cell - 1e-9))
        r0, r1 = int(math.floor(py / cell + 1e-9)), int(math.ceil((py + ph) / cell - 1e-9))
        occ[max(r0, 0):max(min(r1, NY), 0), max(c0, 0):max(min(c1, NX), 0)] = 1

    for i in np.where(pre)[0]:
        mark(x[i], y[i], w[i], h[i])

    S = np.zeros((NY + 1, NX + 1), dtype=np.int32)

    def free_windows(kw, kh):
        """[NY-kh+1, NX-kw+1] bool: window with lower-left cell (r, c) is free
        (S = summed-area table of occ, refreshed once per block)."""
        if kw > NX or kh > NY:
            return None
        return (S[kh:, kw:] - S[:-kh, kw:] - S[kh:, :-kw] + S[:-kh, :-kw]) == 0

    def spots(b, bw, bh, code):
        """Candidate (x, y, growth) for block b with shape bw x bh."""
        kw, kh = int(math.ceil(bw / cell - 1e-9)), int(math.ceil(bh / cell - 1e-9))
        F = free_windows(kw, kh)
        if F is None:
            return []
        rr, cc = np.nonzero(F)
        px, py = cc * cell, rr * cell
        if code & 1:
            keep = cc == 0
            rr, cc, px, py = rr[keep], cc[keep], px[keep], py[keep]
        if code & 8:
            keep = rr == 0
            rr, cc, px, py = rr[keep], cc[keep], px[keep], py[keep]
        if code & 2:            # flush with the right outline edge
            xr = cur[0] * cell - bw if x_hi is None else x_hi - bw
            c0 = int(math.floor(xr / cell + 1e-9))
            keep = cc == c0
            rr, cc, py = rr[keep], cc[keep], py[keep]
            px = np.full(len(rr), xr)
        if code & 4:
            yt = cur[1] * cell - bh if y_hi is None else y_hi - bh
            r0 = int(math.floor(yt / cell + 1e-9))
            keep = rr == r0
            rr, cc, px = rr[keep], cc[keep], px[keep]
            py = np.full(len(rr), yt)
        if not len(rr):
            return []
        gx = np.maximum(cc + kw - cur[0], 0) * (10 if x_hi is not None else 1)
        gy = np.maximum(rr + kh - cur[1], 0) * (10 if y_hi is not None else 1)
        return px, py, gx * cur[1] + gy * cur[0], kw, kh

    def slide(b, X, Y, sw, sh, code):
        """Slide a free spot left, then down, until it touches a placed block
        or the origin (not away from a right/top edge pin), twice."""
        m = placed.copy()
        m[b] = False
        px_, py_, pw_, ph_ = ox[m], oy[m], ow[m], oh[m]
        tol = 1e-9 * max(Wb, Hb)
        for _ in range(2):
            if not code & 2:
                yov = (np.minimum(Y + sh, py_ + ph_) - np.maximum(Y, py_)) > tol
                left = yov & (px_ + pw_ <= X + tol)
                X = max(float((px_ + pw_)[left].max(initial=0.0)), 0.0)
            if not code & 4:
                xov = (np.minimum(X + sw, px_ + pw_) - np.maximum(X, px_)) > tol
                below = xov & (py_ + ph_ <= Y + tol)
                Y = max(float((py_ + ph_)[below].max(initial=0.0)), 0.0)
        return X, Y

    keyf = {"area": lambda i: (bnd[i] == 0, -w[i] * h[i]),
            "x": lambda i: (bnd[i] == 0, x[i] + w[i] / 2),
            "y": lambda i: (bnd[i] == 0, y[i] + h[i] / 2)}[order]
    todo, seen = [], set()
    for i in sorted((i for i in range(n) if not pre[i]), key=keyf):
        if i in seen:
            continue
        group = [i]
        if cl[i] > 0:   # the rest of the cluster right after its first member, nearest first
            rest = [j for j in range(n) if cl[j] == cl[i] and j != i and not pre[j] and j not in seen]
            ci = (x[i] + w[i] / 2, y[i] + h[i] / 2)
            group += sorted(rest, key=lambda j: abs(x[j] + w[j] / 2 - ci[0]) + abs(y[j] + h[j] / 2 - ci[1]))
        todo += group
        seen.update(group)
    placed = np.zeros(n, dtype=bool)
    placed[pre] = True
    for b in todo:
        tx, ty = x[b] + w[b] / 2, y[b] + h[b] / 2
        mates = np.where(placed & (cl == cl[b]) & (cl[b] > 0))[0]
        shp = [(w[b], h[b])]
        if soft[b]:
            a_ = w[b] * h[b]
            shp += [(min(max(w[b] * f, math.sqrt(a_ / 8)), math.sqrt(a_ * 8)), 0.0) for f in (0.7, 1.4, 0.5, 2.0)]
            shp = [(sw, a_ / sw if sh == 0.0 else sh) for sw, sh in shp]
        best = None
        S[1:, 1:] = occ.cumsum(0).cumsum(1)
        cands = [(sw, sh, int(bnd[b])) for sw, sh in shp]
        if bnd[b]:
            cands += [(sw, sh, 0) for sw, sh in shp]        # fallback: off the edge
        for sw, sh, code in cands:
            if best is not None and code == 0 and bnd[b]:
                break                                       # an edge spot was found
            sp = spots(b, sw, sh, code)
            if not len(sp):
                continue
            px, py, grow, kw, kh = sp
            d = np.abs(px + sw / 2 - tx) + np.abs(py + sh / 2 - ty)
            score = d + 1e3 * cell * grow / max(cur[0] * cur[1], 1)   # growth only as a last resort
            idx = np.argsort(score)[:k_near]
            for j in idx:
                s_ = float(score[j])
                X, Y = slide(b, float(px[j]), float(py[j]), sw, sh, code)
                if len(mates):
                    tol = 1e-6 * max(Wb, Hb)
                    ovx = np.minimum(X + sw, ox[mates] + ow[mates]) - np.maximum(X, ox[mates])
                    ovy = np.minimum(Y + sh, oy[mates] + oh[mates]) - np.maximum(Y, oy[mates])
                    touch = (((np.abs(X + sw - ox[mates]) < tol) | (np.abs(ox[mates] + ow[mates] - X) < tol)) & (ovy > tol)) | \
                            (((np.abs(Y + sh - oy[mates]) < tol) | (np.abs(oy[mates] + oh[mates] - Y) < tol)) & (ovx > tol))
                    if not touch.any():
                        s_ += CLUSTER_PEN * math.sqrt(sw * sh)   # detached from the cluster
                if best is None or s_ < best[0]:
                    best = (s_, X, Y, sw, sh, kw, kh)
        if best is None:                                     # nothing fits anywhere: leave it
            continue
        _, bx, by, sw, sh, kw, kh = best
        ox[b], oy[b], ow[b], oh[b] = bx, by, sw, sh
        mark(bx, by, sw, sh)
        cur[0] = max(cur[0], int(math.ceil((bx + sw) / cell - 1e-9)))
        cur[1] = max(cur[1], int(math.ceil((by + sh) / cell - 1e-9)))
        placed[b] = True
    return ox, oy, ow, oh

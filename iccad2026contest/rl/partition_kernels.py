"""
numba kernels for rl/partition.py: the rectangle-set operations the slicing
search spends its time in (10-20 profile: scipy sparse set-up and small numpy
calls were ~3/4 of a search step). Regions are float64 [m, 4] arrays of
disjoint (x0, y0, x1, y1).
"""

import numpy as np
from numba import njit

EPS = 1e-7


@njit(cache=True)
def area(R):
    s = 0.0
    for k in range(R.shape[0]):
        s += (R[k, 2] - R[k, 0]) * (R[k, 3] - R[k, 1])
    return s


@njit(cache=True)
def split(R, axis, a_lo):
    """Cut at coord = t (x if axis 0, else y) where the low part has area a_lo."""
    m = R.shape[0]
    bps = np.unique(np.concatenate((R[:, axis], R[:, axis + 2])))
    nb = bps.shape[0]
    cum = np.zeros(nb)
    for j in range(nb):
        s = 0.0
        for k in range(m):
            lo, hi = R[k, axis], R[k, axis + 2]
            other = R[k, 3 - axis] - R[k, 1 - axis]
            v = bps[j] - lo
            if v > hi - lo:
                v = hi - lo
            if v > 0:
                s += v * other
        cum[j] = s
    k = np.searchsorted(cum, a_lo, side="right") - 1
    if k < 0:
        k = 0
    if k > nb - 2:
        k = nb - 2
    gap = bps[k + 1] - bps[k]
    slope = (cum[k + 1] - cum[k]) / (gap if gap > EPS else EPS)
    t = bps[k] + (a_lo - cum[k]) / (slope if slope > EPS else EPS)
    if t < bps[0]:
        t = bps[0]
    if t > bps[nb - 1]:
        t = bps[nb - 1]
    nlo = 0
    nhi = 0
    for q in range(m):
        if R[q, axis] < t - EPS:
            nlo += 1
        if R[q, axis + 2] > t + EPS:
            nhi += 1
    lo_r = np.empty((nlo, 4))
    hi_r = np.empty((nhi, 4))
    a = 0
    b = 0
    for q in range(m):
        if R[q, axis] < t - EPS:
            lo_r[a] = R[q]
            if lo_r[a, axis + 2] > t:
                lo_r[a, axis + 2] = t
            a += 1
        if R[q, axis + 2] > t + EPS:
            hi_r[b] = R[q]
            if hi_r[b, axis] < t:
                hi_r[b, axis] = t
            b += 1
    return lo_r, hi_r


@njit(cache=True)
def _touch(a, b):
    ox = min(a[2], b[2]) - max(a[0], b[0])
    oy = min(a[3], b[3]) - max(a[1], b[1])
    return (ox > EPS and abs(oy) <= EPS) or (oy > EPS and abs(ox) <= EPS)


@njit(cache=True)
def _find(par, i):
    while par[i] != i:
        par[i] = par[par[i]]
        i = par[i]
    return i


@njit(cache=True)
def components(R):
    m = R.shape[0]
    if m <= 1:
        return m
    par = np.arange(m)
    for i in range(m):
        for j in range(i + 1, m):
            if _touch(R[i], R[j]):
                a, b = _find(par, i), _find(par, j)
                if a != b:
                    par[a] = b
    n = 0
    for i in range(m):
        if _find(par, i) == i:
            n += 1
    return n


@njit(cache=True)
def touching_any(A, B):
    for i in range(A.shape[0]):
        for j in range(B.shape[0]):
            if _touch(A[i], B[j]):
                return True
    return False


@njit(cache=True)
def _label_count(mask, value):
    """4-connected components of cells == value."""
    ny, nx = mask.shape
    seen = np.zeros((ny, nx), dtype=np.bool_)
    stack = np.empty((ny * nx, 2), dtype=np.int64)
    n = 0
    for y in range(ny):
        for x in range(nx):
            if mask[y, x] == value and not seen[y, x]:
                n += 1
                top = 0
                stack[0, 0], stack[0, 1] = y, x
                seen[y, x] = True
                top = 1
                while top > 0:
                    top -= 1
                    cy, cx = stack[top, 0], stack[top, 1]
                    for dy, dx in ((1, 0), (-1, 0), (0, 1), (0, -1)):
                        yy, xx = cy + dy, cx + dx
                        if 0 <= yy < ny and 0 <= xx < nx and mask[yy, xx] == value and not seen[yy, xx]:
                            seen[yy, xx] = True
                            stack[top, 0], stack[top, 1] = yy, xx
                            top += 1
    return n


@njit(cache=True)
def leaf_stats(R):
    """(pieces, holes, pinches, corners) of a rectangle union, rasterized on its
    own coordinate grid (exact for a union of axis-aligned rectangles)."""
    if R.shape[0] == 0:
        return 0, 0, 0, 0
    xs = np.unique(np.concatenate((R[:, 0], R[:, 2])))
    ys = np.unique(np.concatenate((R[:, 1], R[:, 3])))
    p = np.zeros((ys.shape[0] + 2, xs.shape[0] + 2), dtype=np.bool_)
    for k in range(R.shape[0]):
        x0 = np.searchsorted(xs, R[k, 0])
        x1 = np.searchsorted(xs, R[k, 2])
        y0 = np.searchsorted(ys, R[k, 1])
        y1 = np.searchsorted(ys, R[k, 3])
        p[y0 + 1:y1 + 1, x0 + 1:x1 + 1] = True
    pieces = _label_count(p, True)
    holes = _label_count(p, False) - 1
    pinch = 0
    corners = 0
    for y in range(p.shape[0] - 1):
        for x in range(p.shape[1] - 1):
            a, b, c, d = p[y, x], p[y, x + 1], p[y + 1, x], p[y + 1, x + 1]
            s = int(a) + int(b) + int(c) + int(d)
            if s == 1 or s == 3:
                corners += 1
            elif s == 2 and a == d:
                pinch += 1
    return pieces, holes, pinch, corners


@njit(cache=True)
def subtract(R, S):
    """R minus S (both disjoint rectangle sets)."""
    cur = R.copy()
    for k in range(S.shape[0]):
        sx0, sy0, sx1, sy1 = S[k, 0], S[k, 1], S[k, 2], S[k, 3]
        nxt = np.empty((cur.shape[0] * 4 + 1, 4))
        n = 0
        for q in range(cur.shape[0]):
            x0, y0, x1, y1 = cur[q, 0], cur[q, 1], cur[q, 2], cur[q, 3]
            if min(x1, sx1) - max(x0, sx0) <= EPS or min(y1, sy1) - max(y0, sy0) <= EPS:
                nxt[n, 0], nxt[n, 1], nxt[n, 2], nxt[n, 3] = x0, y0, x1, y1
                n += 1
                continue
            if sx0 > x0 + EPS:
                nxt[n, 0], nxt[n, 1], nxt[n, 2], nxt[n, 3] = x0, y0, sx0, y1
                n += 1
            if sx1 < x1 - EPS:
                nxt[n, 0], nxt[n, 1], nxt[n, 2], nxt[n, 3] = sx1, y0, x1, y1
                n += 1
            cx0, cx1 = max(x0, sx0), min(x1, sx1)
            if sy0 > y0 + EPS:
                nxt[n, 0], nxt[n, 1], nxt[n, 2], nxt[n, 3] = cx0, y0, cx1, sy0
                n += 1
            if sy1 < y1 - EPS:
                nxt[n, 0], nxt[n, 1], nxt[n, 2], nxt[n, 3] = cx0, sy1, cx1, y1
                n += 1
        cur = nxt[:n].copy()
    return cur


@njit(cache=True)
def covered(S, R):
    tot = 0.0
    need = 0.0
    for i in range(S.shape[0]):
        need += (S[i, 2] - S[i, 0]) * (S[i, 3] - S[i, 1])
        for j in range(R.shape[0]):
            ox = min(S[i, 2], R[j, 2]) - max(S[i, 0], R[j, 0])
            oy = min(S[i, 3], R[j, 3]) - max(S[i, 1], R[j, 1])
            if ox > 0 and oy > 0:
                tot += ox * oy
    return tot >= need - 1e-6


@njit(cache=True)
def centroid(R):
    a = 0.0
    sx = 0.0
    sy = 0.0
    for k in range(R.shape[0]):
        w = (R[k, 2] - R[k, 0]) * (R[k, 3] - R[k, 1])
        a += w
        sx += w * (R[k, 0] + R[k, 2])
        sy += w * (R[k, 1] + R[k, 3])
    return sx / (2 * a), sy / (2 * a)


@njit(cache=True)
def component_labels(R):
    """Component id (0..k-1) of every rectangle (touching = shared edge)."""
    m = R.shape[0]
    par = np.arange(m)
    for i in range(m):
        for j in range(i + 1, m):
            if _touch(R[i], R[j]):
                a, b = _find(par, i), _find(par, j)
                if a != b:
                    par[a] = b
    lab = np.full(m, -1)
    roots = np.full(m, -1)
    k = 0
    for i in range(m):
        r = _find(par, i)
        if roots[r] < 0:
            roots[r] = k
            k += 1
        lab[i] = roots[r]
    return lab

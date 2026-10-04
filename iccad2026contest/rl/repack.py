"""
Re-packing polish: keep the RL layout's relative arrangement, remove its
white space, then anneal the arrangement and soft-block shapes.

Re-pack (`_legalize`): every pair of blocks gets one left/right or
below/above relation from where they sit, and every block goes to the
tightest coordinates those relations allow (longest path in the two
constraint graphs). It is overlap-free by construction, exact for an
already-legal layout (so the first re-pack can only shrink it), and removes
the gaps rl/compaction.py's one-block-at-a-time slides leave. (A sequence-
pair version was tried first: deriving a sequence pair from a placement
contradicts it for a few pairs, which scrambled the start layout.)

Annealing: a move perturbs the placement (swap two blocks, move one beside
another, nudge, reshape, shift a cluster), then the result is re-packed.

Constraints:
  - preplaced blocks: never moved; a candidate whose re-pack would push one
    off its spot is penalised and never kept as the best.
  - fixed-shape blocks: never reshaped. MIB groups: reshaped together.
  - right/top-pinned blocks: slid out to the bbox edge after re-packing
    when nothing blocks them. A final pass (`_fix_boundary`) then tries
    snapping each block still off its edge (`_snap_variants`: placed just
    past that edge, so the re-pack makes it outermost).
  - boundary / grouping / MIB violations, wirelength and area: priced by a
    fast numpy re-implementation of the contest cost (`_Cost`). The final
    answer is re-checked with the real evaluator and only kept if better.

`repack_polish` runs `chains` independent annealing chains (separate
processes when > 1) and returns the best.
"""

import math
import os
import random
import time
from typing import List, Optional, Tuple

import numpy as np

Position = Tuple[float, float, float, float]
ALPHA, BETA = 0.5, 2.0   # iccad2026_evaluate's cost weights
EPS = 1e-6
# _fix_boundary snap mode: "orig", "noneg" or "reshape" (see there). 10-01,
# 100 cached layouts, 1 s slice+repack: 1.530 / 1.529 / 1.527.
SNAP_MODE = os.environ.get("FLOORSET_SNAP", "reshape")


# ---------------------------------------------------------------------------
# Problem data (plain numpy, picklable for worker processes)
# ---------------------------------------------------------------------------
def _problem(instance, positions: List[Position]) -> dict:
    n = len(positions)
    c = instance.constraints
    ncols = c.shape[1] if c is not None and c.dim() > 1 else 0
    col = lambda j: (c[:n, j].numpy().astype(np.int64) if ncols > j else np.zeros(n, np.int64))
    pos = np.array(positions, dtype=np.float64)
    b2b = instance.b2b_connectivity
    b2b = b2b[b2b[:, 0] >= 0].numpy() if b2b is not None and b2b.numel() else np.zeros((0, 3))
    p2b = instance.p2b_connectivity
    p2b = p2b[p2b[:, 0] >= 0].numpy() if p2b is not None and p2b.numel() else np.zeros((0, 3))
    pins = instance.pins_pos.numpy().astype(np.float64) if instance.pins_pos is not None else np.zeros((0, 2))
    cluster, mib, bnd = col(3), col(2), col(4)
    n_soft = int((bnd != 0).sum())
    for ids in (mib, cluster):
        for g in range(1, int(ids.max(initial=0)) + 1):
            n_soft += max(0, int((ids == g).sum()) - 1)
    return dict(
        n=n, pos=pos, fixed=col(0) != 0, preplaced=col(1) != 0, mib=mib, cluster=cluster, bnd=bnd,
        area=pos[:, 2] * pos[:, 3],
        b2b_i=b2b[:, 0].astype(np.int64), b2b_j=b2b[:, 1].astype(np.int64), b2b_w=b2b[:, 2],
        p2b_b=p2b[:, 1].astype(np.int64), p2b_w=p2b[:, 2],
        p2b_px=pins[p2b[:, 0].astype(np.int64), 0] if len(p2b) else np.zeros(0),
        p2b_py=pins[p2b[:, 0].astype(np.int64), 1] if len(p2b) else np.zeros(0),
        n_soft=max(n_soft, 1),
    )


# ---------------------------------------------------------------------------
# Constraint-graph legalisation (the re-pack)
# ---------------------------------------------------------------------------
def _legalize(x, y, w, h, P, floors=(0.0, 0.0)):
    """Re-pack a placement (which may overlap) as tight as its relative
    arrangement allows. Each pair gets one relation from the geometry: the
    axis with the larger gap (the smaller overlap, if they overlap), ordered
    by centre. All horizontal relations follow centre-x order and all
    vertical ones centre-y order, so both graphs are acyclic, and every pair
    is constrained on one axis, so the longest-path result never overlaps.
    Returns (x, y, overshoot); overshoot > 0 means a preplaced block would be
    pushed off its fixed spot (it is kept on it, so the result overlaps).
    `floors`: (x, y) lowest coordinates (a snapped left/bottom-pinned block
    can lower them further); rl/slicing.py lowers them to make room in
    front of preplaced blocks. Per-axis overshoot: `_legalize.over`."""
    n = P["n"]
    pre, pre_x, pre_y = P["preplaced"], P["pos"][:, 0], P["pos"][:, 1]
    x0, y0, x1, y1 = x, y, x + w, y + h
    cx, cy = x + w / 2, y + h / 2
    gap_x = np.maximum(x0[None, :] - x1[:, None], x0[:, None] - x1[None, :])
    gap_y = np.maximum(y0[None, :] - y1[:, None], y0[:, None] - y1[None, :])
    horiz = gap_x >= gap_y
    idx = np.arange(n)
    first_x = (cx[:, None] < cx[None, :]) | ((cx[:, None] == cx[None, :]) & (idx[:, None] < idx[None, :]))
    first_y = (cy[:, None] < cy[None, :]) | ((cy[:, None] == cy[None, :]) & (idx[:, None] < idx[None, :]))
    H = horiz & first_x            # H[a, b]: a left of b
    V = ~horiz & first_y           # V[a, b]: a below b
    np.fill_diagonal(H, False)
    np.fill_diagonal(V, False)
    bnd = P["bnd"]
    # Preferred coordinate: where the block is now; edge pins prefer their edge.
    pref_x = np.where(bnd & 1, -np.inf, np.where(bnd & 2, np.inf, x0))
    pref_y = np.where(bnd & 8, -np.inf, np.where(bnd & 4, np.inf, y0))
    # A left/bottom-pinned block placed past 0 (a snap) lowers the floor, so
    # the layout grows that way instead of pushing preplaced blocks off.
    floor_x = min(floors[0], float(x0[(bnd & 1) != 0].min(initial=0.0)))
    floor_y = min(floors[1], float(y0[(bnd & 8) != 0].min(initial=0.0)))
    nx, ox, sx = _axis(H, cx, w, pref_x, pre, pre_x, floor_x)
    ny, oy, sy = _axis(V, cy, h, pref_y, pre, pre_y, floor_y)
    _legalize.slack = (sx, sy)   # per-block slack of the last call (0 = on a critical path)
    _legalize.over = (ox, oy)
    return nx, ny, ox + oy


def _axis(G, c, size, pref, pre, pre_v, floor=0.0):
    """One axis of _legalize: ASAP/ALAP windows from the constraint graph G
    (G[a, b]: a before b), then each block at its preferred coordinate
    clamped into its window, so the extent stays the tightest possible and
    blocks with slack don't move needlessly. Longest paths are computed by
    whole-matrix relaxation until nothing changes (at most DAG-depth
    rounds), not a per-block Python loop -- ~5-10x faster."""
    n = len(c)
    NEG, POS = np.where(G, 0.0, -np.inf), np.where(G, 0.0, np.inf)
    maxpred = lambda v: np.maximum((NEG + (v + size)[:, None]).max(axis=0), floor)
    has_succ = G.any(axis=1)
    lo = np.where(pre, pre_v, floor)
    for _ in range(n + 1):                               # ASAP
        new = np.where(pre, pre_v, maxpred(lo))
        if np.array_equal(new, lo):
            break
        lo = new
    over = float(np.maximum(maxpred(lo) - pre_v, 0.0)[pre].sum())
    ext = float((lo + size).max())
    hi = np.where(pre, pre_v, ext - size)
    for _ in range(n + 1):                               # ALAP within the same extent
        succ = (POS + hi[None, :]).min(axis=1) - size
        new = np.where(pre, pre_v, np.where(has_succ, np.minimum(succ, ext - size), ext - size))
        if np.array_equal(new, hi):
            break
        hi = new
    succ = (POS + hi[None, :]).min(axis=1) - size
    over += float(np.maximum(pre_v - succ, 0.0)[pre & has_succ].sum())
    out = lo.copy()
    for _ in range(n + 1):                               # clamp preference into the window
        low = maxpred(out)
        new = np.where(pre, pre_v, np.minimum(np.maximum(pref, low), np.maximum(hi, low)))
        if np.array_equal(new, out):
            break
        out = new
    return out, over, hi - lo


# ---------------------------------------------------------------------------
# Fast contest-style cost
# ---------------------------------------------------------------------------
class _Cost:
    def __init__(self, P, hpwl_ref, area_ref):
        self.P, self.hpwl_ref, self.area_ref = P, max(hpwl_ref, 1e-9), max(area_ref, 1e-9)
        cl = P["cluster"]
        self.groups = [np.where(cl == g)[0] for g in range(1, int(cl.max(initial=0)) + 1)]
        self.groups = [g for g in self.groups if len(g) > 1]
        mb = P["mib"]
        self.mibs = [np.where(mb == g)[0] for g in range(1, int(mb.max(initial=0)) + 1)]
        self.pinned = np.where(P["bnd"] != 0)[0]

    def hpwl(self, x, y, w, h):
        P = self.P
        cx, cy = x + w / 2, y + h / 2
        t = 0.0
        if len(P["b2b_w"]):
            i, j = P["b2b_i"], P["b2b_j"]
            t += float((P["b2b_w"] * (np.abs(cx[i] - cx[j]) + np.abs(cy[i] - cy[j]))).sum())
        if len(P["p2b_w"]):
            b = P["p2b_b"]
            t += float((P["p2b_w"] * (np.abs(cx[b] - P["p2b_px"]) + np.abs(cy[b] - P["p2b_py"]))).sum())
        return t

    def off_edge(self, x, y, w, h):
        """Boundary-pinned blocks not touching their required bbox edge(s)."""
        x1, y1 = x + w, y + h
        X0, Y0, X1, Y1 = x.min(), y.min(), x1.max(), y1.max()
        out = []
        for b in self.pinned:
            code = self.P["bnd"][b]
            if ((code & 1 and abs(x[b] - X0) > EPS) or (code & 2 and abs(x1[b] - X1) > EPS)
                    or (code & 4 and abs(y1[b] - Y1) > EPS) or (code & 8 and abs(y[b] - Y0) > EPS)):
                out.append(int(b))
        return out

    def violations(self, x, y, w, h):
        x1, y1 = x + w, y + h
        v = len(self.off_edge(x, y, w, h))
        for g in self.groups:                   # connected components - 1 (edge contact)
            parent = list(range(len(g)))

            def find(a):
                while parent[a] != a:
                    parent[a] = parent[parent[a]]
                    a = parent[a]
                return a
            gx, gy, gx1, gy1 = x[g], y[g], x1[g], y1[g]
            ov_x = np.minimum(gx1[:, None], gx1[None, :]) - np.maximum(gx[:, None], gx[None, :])
            ov_y = np.minimum(gy1[:, None], gy1[None, :]) - np.maximum(gy[:, None], gy[None, :])
            touch = ((ov_x > EPS) & (np.abs(ov_y) <= EPS)) | ((ov_y > EPS) & (np.abs(ov_x) <= EPS))
            for a, b in zip(*np.nonzero(np.triu(touch, 1))):
                ra, rb = find(a), find(b)
                if ra != rb:
                    parent[ra] = rb
            v += len({find(a) for a in range(len(g))}) - 1
        for g in self.mibs:
            v += len({(round(w[i], 4), round(h[i], 4)) for i in g}) - 1
        return v

    def __call__(self, x, y, w, h):
        area = ((x + w).max() - x.min()) * ((y + h).max() - y.min())
        q = 1 + ALPHA * (self.hpwl(x, y, w, h) / self.hpwl_ref + area / self.area_ref)
        return q * math.exp(BETA * self.violations(x, y, w, h) / self.P["n_soft"])


# ---------------------------------------------------------------------------
# Annealing chain
# ---------------------------------------------------------------------------
def _snap_variants(x, y, w, h, a, code):
    """Candidate spots for block `a` on the bbox edge(s) its boundary code
    requires (1 left, 2 right, 4 top, 8 bottom). The block goes just past one
    required edge, so the re-pack orders it outermost there, and onto the
    other required edge (corner) or at 5 spots along the edge (single)."""
    X0, Y0, X1, Y1 = x.min(), y.min(), (x + w).max(), (y + h).max()
    out_x = X0 - w[a] if code & 1 else X1 if code & 2 else None     # just past the edge
    out_y = Y0 - h[a] if code & 8 else Y1 if code & 4 else None
    on_x = X0 if code & 1 else X1 - w[a]                             # flush with it, inside
    on_y = Y0 if code & 8 else Y1 - h[a]
    spots = []
    if out_x is not None and out_y is not None:                      # corner
        spots = [(out_x, on_y), (on_x, out_y)]
    elif out_x is not None:
        spots = [(out_x, y[a])] + [(out_x, Y0 + t * (Y1 - Y0 - h[a])) for t in (0, .25, .5, .75, 1)]
    elif out_y is not None:
        spots = [(x[a], out_y)] + [(X0 + t * (X1 - X0 - w[a]), out_y) for t in (0, .25, .5, .75, 1)]
    for sx, sy in spots:
        nx, ny = x.copy(), y.copy()
        nx[a], ny[a] = sx, sy
        yield nx, ny


def _chain(P, time_budget: float, seed: int):
    rng = random.Random(seed)
    n = P["n"]
    pos = P["pos"]
    pre = P["preplaced"]
    cost_fn = _Cost(P, *_ref_metrics(P))
    soft = ~P["fixed"] & ~pre & (P["area"] > 0)
    units, seen = [], set()        # reshape units: a soft block, or a whole MIB group
    for i in np.where(soft)[0]:
        g = P["mib"][i]
        if g:
            if g in seen:
                continue
            seen.add(g)
            members = np.where(P["mib"] == g)[0]
            if soft[members].all():
                units.append(members)
        else:
            units.append(np.array([i]))
    movable = np.where(~pre)[0]
    clusters = [np.where((P["cluster"] == g) & ~pre)[0] for g in range(1, int(P["cluster"].max(initial=0)) + 1)]
    clusters = [c for c in clusters if len(c) > 1]
    scale = math.sqrt(max(float(P["area"].sum()), 1e-9))

    def evaluate(x, y, w, h):
        lx, ly, over = _legalize(x, y, w, h, P)
        c = cost_fn(lx, ly, w, h)
        return c * (1.0 + 20.0 * over / scale), lx, ly, over <= EPS, _legalize.slack

    w, h = pos[:, 2].copy(), pos[:, 3].copy()
    cur, x, y, feas, slack = evaluate(pos[:, 0].copy(), pos[:, 1].copy(), w, h)
    unit_of = {int(i): u for u in units for i in u}
    best = (cur, x.copy(), y.copy(), w.copy(), h.copy()) if feas else None
    if len(movable) < 2:
        return None if best is None else _as_positions(best)

    def reshape(x, y, w, h, u, f):
        """Scale width by f (height by 1/f, area kept) about each centre."""
        w, h = w.copy(), h.copy()
        for i in u:
            a, c_x, c_y = w[i] * h[i], x[i] + w[i] / 2, y[i] + h[i] / 2
            nw = min(max(w[i] * f, math.sqrt(a / 8.0)), math.sqrt(a * 8.0))
            w[i], h[i] = nw, a / nw
            x[i], y[i] = c_x - nw / 2, c_y - h[i] / 2
        return w, h

    def propose(x, y, w, h):
        x, y = x.copy(), y.copy()
        r = rng.random()
        if r < 0.2 and units:
            # Critical reshape: a soft block with no slack on one axis (it sits
            # on the chain setting the layout's width or height) but room on
            # the other gets thinner along the tight axis.
            sx, sy = slack
            tol = 1e-6
            crit_x = [i for i in unit_of if sx[i] <= tol and sy[i] > tol]
            crit_y = [i for i in unit_of if sy[i] <= tol and sx[i] > tol]
            if crit_x or crit_y:
                use_x = bool(crit_x) and (not crit_y or rng.random() < 0.5)
                i = rng.choice(crit_x if use_x else crit_y)
                shrink = rng.uniform(0.75, 0.97)
                w, h = reshape(x, y, w, h, unit_of[i], shrink if use_x else 1.0 / shrink)
                return x, y, w, h
        if r < 0.35 and units:                                  # random reshape about the centre
            w, h = reshape(x, y, w, h, units[rng.randrange(len(units))],
                           math.exp(rng.gauss(0.0, 0.25)))
        elif r < 0.55:                                          # swap two blocks' centres
            a, b = rng.sample(list(movable), 2)
            ca = (x[a] + w[a] / 2, y[a] + h[a] / 2)
            cb = (x[b] + w[b] / 2, y[b] + h[b] / 2)
            x[a], y[a] = cb[0] - w[a] / 2, cb[1] - h[a] / 2
            x[b], y[b] = ca[0] - w[b] / 2, ca[1] - h[b] / 2
        elif r < 0.8:                                           # move a block beside another
            a = movable[rng.randrange(len(movable))]
            b = rng.randrange(n)
            if b == a:
                return x, y, w, h
            side = rng.randrange(4)
            t = rng.random()
            if side == 0:
                x[a], y[a] = x[b] + w[b], y[b] + t * h[b] - h[a] / 2
            elif side == 1:
                x[a], y[a] = x[b] - w[a], y[b] + t * h[b] - h[a] / 2
            elif side == 2:
                x[a], y[a] = x[b] + t * w[b] - w[a] / 2, y[b] + h[b]
            else:
                x[a], y[a] = x[b] + t * w[b] - w[a] / 2, y[b] - h[a]
        elif r < 0.92 or not clusters:                          # nudge
            a = movable[rng.randrange(len(movable))]
            s_ = math.sqrt(w[a] * h[a])
            x[a] += rng.gauss(0.0, s_)
            y[a] += rng.gauss(0.0, s_)
        else:                                                   # shift a whole cluster
            g = clusters[rng.randrange(len(clusters))]
            s_ = math.sqrt(float((w[g] * h[g]).sum()))
            x[g] += rng.gauss(0.0, s_ / 2)
            y[g] += rng.gauss(0.0, s_ / 2)
        return x, y, w, h

    deltas = []                    # start temperature: typical uphill move accepted ~50%
    for _ in range(20):
        c2 = evaluate(*propose(x, y, w, h))[0]
        if c2 > cur:
            deltas.append(c2 - cur)
    t0 = (float(np.mean(deltas)) / math.log(2)) if deltas else 0.01
    start = time.time()
    while True:
        frac = (time.time() - start) / max(time_budget, 1e-3)
        if frac >= 1.0:
            break
        temp = t0 * (0.001 ** frac)
        px, py, pw, ph = propose(x, y, w, h)
        c2, lx, ly, feas2, sl2 = evaluate(px, py, pw, ph)
        if c2 <= cur or rng.random() < math.exp(-(c2 - cur) / max(temp, 1e-12)):
            x, y, w, h, cur, slack = lx, ly, pw, ph, c2, sl2
            if feas2 and (best is None or cur < best[0]):
                best = (cur, x.copy(), y.copy(), w.copy(), h.copy())
    if best is not None:
        best = _fix_boundary(best, evaluate, cost_fn, P)
    return None if best is None else _as_positions(best)


def _fix_boundary(best, evaluate, cost_fn, P, rounds: int = 3):
    """Greedy final pass: for each block still off its edge, try snapping it
    to each `_snap_variants` spot and keep the cheapest feasible result if
    it beats the current one. SNAP_MODE (10-01): "noneg" rejects snaps that
    push the layout past the origin; "reshape" also tries flatter/thinner
    shapes for plain soft blocks so they fit beside obstacles at the edge."""
    soft = ~P["fixed"] & ~P["preplaced"] & (P["area"] > 0) & (P["mib"] == 0)
    for _ in range(rounds):
        c0, x, y, w, h = best
        off = cost_fn.off_edge(x, y, w, h)
        if not off:
            break
        improved = False
        for a in off:
            c0, x, y, w, h = best
            fx, fy = min(0.0, float(x.min())), min(0.0, float(y.min()))
            shapes = [(w, h)]
            if SNAP_MODE == "reshape" and soft[a]:
                ar = w[a] * h[a]
                for f in (0.5, 0.7, 1.4, 2.0):
                    nw = min(max(w[a] * f, math.sqrt(ar / 8.0)), math.sqrt(ar * 8.0))
                    sw, sh = w.copy(), h.copy()
                    sw[a], sh[a] = nw, ar / nw
                    shapes.append((sw, sh))
            spots = [(sx, sy, sw, sh) for sw, sh in shapes
                     for sx, sy in _snap_variants(x, y, sw, sh, a, P["bnd"][a])]
            for sx, sy, sw, sh in spots:
                c, lx, ly, feas, _ = evaluate(sx, sy, sw, sh)
                if SNAP_MODE != "orig" and (lx.min() < fx - EPS or ly.min() < fy - EPS):
                    continue
                if feas and c < best[0] - 1e-12:
                    best = (c, lx.copy(), ly.copy(), sw.copy(), sh.copy())
                    improved = True
        if not improved:
            break
    return best


def _as_positions(best):
    c, x, y, w, h = best
    return c, [(float(x[i]), float(y[i]), float(w[i]), float(h[i])) for i in range(len(x))]


def _ref_metrics(P):
    pos = P["pos"]
    x, y, w, h = pos[:, 0], pos[:, 1], pos[:, 2], pos[:, 3]
    tmp = _Cost(P, 1.0, 1.0)
    area = ((x + w).max() - x.min()) * ((y + h).max() - y.min())
    return tmp.hpwl(x, y, w, h), area


def _chain_worker(args):
    return _chain(*args)


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------
def repack_polish(instance, positions: List[Position], time_budget: float = 8.0,
                  chains: int = 1, seed: Optional[int] = None, pool=None,
                  verbose: bool = False) -> List[Position]:
    """Best of `chains` annealing chains (in `pool`, or a temporary process
    pool, when chains > 1), compacted, and returned only if the real contest
    evaluator finds it feasible and cheaper than `positions` under the
    contest formula with wirelength/area measured relative to `positions`."""
    from .compaction import compact

    P = _problem(instance, positions)
    base = seed if seed is not None else random.randrange(1 << 30)
    jobs = [(P, time_budget, base + k) for k in range(max(1, chains))]
    if len(jobs) == 1:
        results = [_chain(*jobs[0])]
    elif pool is not None:
        results = pool.map(_chain_worker, jobs)
    else:
        import multiprocessing as mp
        with mp.get_context("forkserver").Pool(len(jobs)) as p:
            results = p.map(_chain_worker, jobs)
    results = [r for r in results if r is not None]
    if not results:
        return positions
    _, raw = min(results, key=lambda r: r[0])
    # Compaction can pull right/top-pinned blocks off their edge (it undid
    # the boundary snaps), so keep whichever of the two scores better.
    old_cost, _ = contest_relative_cost(instance, positions, positions)
    scored = [(contest_relative_cost(instance, c, positions), c)
              for c in (raw, compact(raw, instance.constraints))]
    scored = [(cost, c) for (cost, ok), c in scored if ok] or [((float('inf'), False), raw)]
    new_cost, cand = min(scored, key=lambda t: t[0])
    feasible = new_cost < float('inf')
    if verbose:
        print(f"  repack: relative cost {old_cost:.4f} -> {new_cost:.4f} "
              f"(feasible={feasible}, {len(results)} chains)")
    return cand if feasible and new_cost < old_cost else positions


def contest_relative_cost(instance, positions, reference):
    """(cost, feasible): contest formula with hpwl/area as ratios to
    `reference`'s (no ground truth at solve time), exact violation counts
    from iccad2026_evaluate.evaluate_solution."""
    from iccad2026_evaluate import evaluate_solution
    from .reward import _target_positions_list

    def metrics(pos):
        return evaluate_solution({'positions': pos, 'runtime': 1.0},
                                 {'hpwl_baseline': 1.0, 'area_baseline': 1.0},
                                 instance.constraints, instance.b2b_connectivity,
                                 instance.p2b_connectivity, instance.pins_pos,
                                 instance.area_targets, _target_positions_list(instance))
    ref = metrics(reference)
    m = ref if positions is reference else metrics(positions)
    q = 1 + ALPHA * (m.hpwl_total / max(ref.hpwl_total, 1e-9) + m.bbox_area / max(ref.bbox_area, 1e-9))
    return q * math.exp(BETA * m.violations_relative), bool(m.is_feasible)

"""
Slicing re-pack: rebuild the RL layout as a slicing floorplan with (almost)
no white space, keeping its relative arrangement.

Why: layouts fill the canvas they're given (area gap ~0.32 on 100 designs),
and re-packing the RL arrangement's own left/right/above/below relations
(rl/repack.py) only removes ~4%. The contest puts no aspect-ratio limit on
soft blocks (area within 1% only) and ~86% of blocks are soft, so a
guillotine partition can give every soft block a rectangle of exactly its
area.

Construction (`_build`): recursively cut the block set in two across the
longer side of the current rectangle, ordered by the RL centres along that
axis, at a split near half the area; the rectangle splits in proportion to
the two sides' areas. Leaves are blocks: soft ones take their rectangle,
fixed-shape ones keep their dims (centred in it), preplaced ones go back to
their spot. rl/repack.py's `_legalize` then removes any overlap this
leaves, and `_fix_boundary` snaps edge-pinned blocks.

`slicing_polish` tries several root aspect ratios / fill targets / split
rules and keeps the cheapest under repack's fast contest cost; the final
answer is re-checked with the real evaluator against the input.
"""

import math
import os
import time
from typing import List, Tuple

import numpy as np

from . import repack as R

Position = Tuple[float, float, float, float]


def _build(P, cx, cy, W, H, split: str):
    """Leaf rectangles of the slicing partition of all blocks into the
    W x H rectangle at the origin. Returns (x, y, w, h) of the rectangles.

    Preplaced blocks are obstacles at their true spot: a cut may not cross
    one (obstacle edges are candidate cuts), each side takes the movable
    blocks whose area best matches its free area, and an obstacle alone in
    its rectangle keeps its true rectangle. Each cut tries both directions,
    scored by edge-pinned blocks stranded off their required root edge, free-
    area mismatch, balance and child aspect. Movable blocks are ordered by
    RL centre, pinned ones moved to their side when the rectangle is on that
    root edge. A grouping cluster is never cut until it is alone in its
    rectangle, so its blocks tile one rectangle and all touch. split="gap"
    also rewards the widest gap between consecutive RL centres."""
    n = P["n"]
    area, bnd, pre, clu = P["area"], P["bnd"], P["preplaced"], P["cluster"]
    pos = P["pos"]
    rx, ry, rw, rh = (np.zeros(n) for _ in range(4))
    # Smallest width/height each block's leaf needs: a fixed block's own
    # dims, a soft block's short side at the MAX_ASPECT limit.
    fixed = P["fixed"] & ~pre
    need_w = np.where(fixed, pos[:, 2], np.sqrt(area / MAX_ASPECT))
    need_h = np.where(fixed, pos[:, 3], np.sqrt(area / MAX_ASPECT))

    def options(idx, x, y, w, h, edges, vertical):
        mov, obs = idx[~pre[idx]], idx[pre[idx]]
        ax = 0 if vertical else 1
        lo, span, other = (x, w, h) if vertical else (y, h, w)
        lo_bit, hi_bit = (1, 2) if vertical else (8, 4)
        c = (cx[mov] if vertical else cy[mov]).copy()
        span_c = float(c.max() - c.min()) + 1.0 if len(c) else 1.0
        if edges & lo_bit:
            c = np.where(bnd[mov] & lo_bit, c - 4 * span_c, c)
        if edges & hi_bit:
            c = np.where(bnd[mov] & hi_bit, c + 4 * span_c, c)
        # Atoms: a cluster's members move as one (unless the rectangle
        # holds just that cluster), other blocks alone.
        lab = np.where(clu[mov] > 0, clu[mov], -1 - mov)
        if len(np.unique(lab)) == 1:
            lab = -1 - mov
        _, inv = np.unique(lab, return_inverse=True)
        if len(mov):
            akey = np.bincount(inv, weights=c * area[mov]) / np.bincount(inv, weights=area[mov])
            o = np.lexsort((c, akey[inv]))
        else:
            o = np.zeros(0, np.int64)
        order = mov[o]
        at_atom = np.concatenate([[True], inv[o][1:] != inv[o][:-1], [True]]) if len(mov) else np.array([True])
        cum = np.concatenate([[0.0], np.cumsum(area[order])])
        ks = np.nonzero(at_atom)[0]
        o_lo, o_hi = pos[obs, ax], pos[obs, ax] + pos[obs, ax + 2]
        o_area = area[obs]
        free = w * h - o_area.sum()
        dens = free / max(cum[-1], 1e-12)          # rect units per unit of block area
        # Candidate cuts: area-proportional ones for each split, plus obstacle edges.
        tot = cum[-1] + o_area.sum() / max(dens, 1e-12)
        t_c = np.concatenate([lo + span * (cum[ks] + 0.0) / max(tot, 1e-12), o_lo, o_hi])
        t_c = t_c[(t_c > lo + 1e-9) & (t_c < lo + span - 1e-9)]
        if len(t_c) == 0:
            return np.inf, None
        left_o = o_hi[None, :] <= t_c[:, None] + 1e-9
        right_o = o_lo[None, :] >= t_c[:, None] - 1e-9
        cross = ~(left_o | right_o)
        free_left = (t_c - lo) * other - (o_area[None, :] * left_o).sum(1)
        # Split of the movable blocks whose area best fills that free space.
        want = free_left / max(dens, 1e-12)
        j = np.clip(np.searchsorted(cum[ks], want), 0, len(ks) - 1)
        j2 = np.clip(j - 1, 0, len(ks) - 1)
        k = np.where(np.abs(cum[ks[j2]] - want) < np.abs(cum[ks[j]] - want), ks[j2], ks[j])
        mism = np.abs(cum[k] - want) * dens / max(w * h, 1e-12)
        f = (t_c - lo) / span
        n_left = k + left_o.sum(1)
        ok = (cross.sum(1) == 0) & (n_left > 0) & (n_left < len(idx))
        pin_hi = ((bnd[order] & hi_bit) != 0) & bool(edges & hi_bit)
        pin_lo = ((bnd[order] & lo_bit) != 0) & bool(edges & lo_bit)
        c_hi = np.concatenate([[0], np.cumsum(pin_hi)])
        c_lo = np.concatenate([[0], np.cumsum(pin_lo)])
        lost = c_hi[k] + (c_lo[-1] - c_lo[k])
        if len(obs):
            ph = ((bnd[obs] & hi_bit) != 0) & bool(edges & hi_bit)
            pl = ((bnd[obs] & lo_bit) != 0) & bool(edges & lo_bit)
            lost = lost + (ph[None, :] & left_o).sum(1) + (pl[None, :] & right_o).sum(1)
        cw1, cw2 = (w * f, w * (1 - f)) if vertical else (np.full_like(f, w), np.full_like(f, w))
        ch1, ch2 = (np.full_like(f, h), np.full_like(f, h)) if vertical else (h * f, h * (1 - f))
        aspect = (np.abs(np.log(np.maximum(cw1, 1e-12) / np.maximum(ch1, 1e-12)))
                  + np.abs(np.log(np.maximum(cw2, 1e-12) / np.maximum(ch2, 1e-12))))
        # Sizing: each child must be at least as long (along the cut axis) and
        # as tall (across it) as the largest leaf need among its blocks.
        n_al = (need_w if vertical else need_h)[order]
        n_cr = (need_h if vertical else need_w)[order]
        pre_al = np.concatenate([[0.0], np.maximum.accumulate(n_al)]) if len(order) else np.zeros(1)
        suf_al = np.concatenate([np.maximum.accumulate(n_al[::-1])[::-1], [0.0]]) if len(order) else np.zeros(1)
        cross_need = float(n_cr.max(initial=0.0))
        len_a, len_b = f * span, (1 - f) * span
        misfit = (np.maximum(pre_al[k] - len_a, 0) / np.maximum(pre_al[k], 1e-9)
                  + np.maximum(suf_al[k] - len_b, 0) / np.maximum(suf_al[k], 1e-9)
                  + max(cross_need - other, 0.0) / max(cross_need, 1e-9))
        score = 3.0 * lost + 4.0 * mism + 1.0 * np.abs(f - 0.5) + 0.25 * aspect + 3.0 * misfit
        if split == "gap" and len(order) > 1:
            cs = c[o]
            gaps = np.concatenate([[0.0], cs[1:] - cs[:-1], [0.0]])
            score = score - 0.5 * gaps[k] / span_c
        score = np.where(ok, score, np.inf)
        b = int(np.argmin(score))
        if not np.isfinite(score[b]):
            return np.inf, None
        kb = int(k[b])
        lft = left_o[b]
        # Move the cut to where the free area matches the chosen blocks
        # exactly, as far as the obstacles on each side allow.
        t = lo + (cum[kb] * dens + float(o_area[lft].sum())) / max(other, 1e-12)
        # ...and room for the longest leaf need on each side.
        t = min(max(t, lo + float(pre_al[kb])), lo + span - float(suf_al[kb]))
        t = min(max(t, float(o_hi[lft].max(initial=lo))), float(o_lo[~lft].min(initial=lo + span)))
        fb = min(max((t - lo) / span, 1e-6), 1 - 1e-6)
        a_set = np.concatenate([order[:kb], obs[lft]])
        b_set = np.concatenate([order[kb:], obs[~lft]])
        return float(score[b]), (a_set, b_set, fb)

    stack = [(np.arange(n), 0.0, 0.0, W, H, 15)]
    while stack:
        idx, x, y, w, h, edges = stack.pop()
        if len(idx) == 1:
            i = idx[0]
            if pre[i]:
                rx[i], ry[i], rw[i], rh[i] = pos[i]
            else:
                rx[i], ry[i], rw[i], rh[i] = x, y, w, h
            continue
        sv, ov = options(idx, x, y, w, h, edges, True)
        sh, oh = options(idx, x, y, w, h, edges, False)
        if ov is None and oh is None:
            # No clean cut (e.g. an obstacle spans the rectangle): halve by
            # centre along the longer side and let the legaliser sort it out.
            vertical = w >= h
            key = np.where(pre[idx], (pos[idx, 0] + pos[idx, 2] / 2) if vertical else (pos[idx, 1] + pos[idx, 3] / 2),
                           cx[idx] if vertical else cy[idx])
            srt = idx[np.argsort(key)]
            a_set, b_set = srt[:len(idx) // 2], srt[len(idx) // 2:]
            f = float(area[a_set].sum() / area[idx].sum())
        else:
            vertical = sv <= sh
            a_set, b_set, f = ov if vertical else oh
        if vertical:
            stack += [(a_set, x, y, w * f, h, edges & ~2), (b_set, x + w * f, y, w * (1 - f), h, edges & ~1)]
        else:
            stack += [(a_set, x, y, w, h * f, edges & ~4), (b_set, x, y + h * f, w, h * (1 - f), edges & ~8)]
    return rx, ry, rw, rh


MAX_ASPECT = float(os.environ.get("FLOORSET_MAX_ASPECT", "10"))   # 10 unless the cap is set (rl/repack.py)


def _candidate(P, cx, cy, W, H, split, pre_mode="tree"):
    """Slicing layout -> block (x, y, w, h) before legalisation, plus the
    legaliser floors. pre_mode "tree": preplaced blocks steer the cuts (see
    _build); "shift": cut as if they were movable, shift the layout to line
    up best with their true spots, then put them back; "clamp" / "inplace"
    (10-03): the same, sliding only while x, y stay >= 0 / not at all."""
    pre = P["preplaced"]
    if pre_mode in ("shift", "clamp", "inplace") and pre.any():
        Pf = dict(P, preplaced=np.zeros_like(pre))
        rx, ry, rw, rh = _build(Pf, cx, cy, W, H, split)
        pos = P["pos"]
        dx = float(np.median(pos[pre, 0] + pos[pre, 2] / 2 - (rx[pre] + rw[pre] / 2)))
        dy = float(np.median(pos[pre, 1] + pos[pre, 3] / 2 - (ry[pre] + rh[pre] / 2)))
        if pre_mode == "clamp":     # slide only as far as keeps the layout at x, y >= 0
            dx = max(dx, -float(rx[~pre].min(initial=0.0)))
            dy = max(dy, -float(ry[~pre].min(initial=0.0)))
        elif pre_mode == "inplace":  # no slide: the legaliser clears the preplaced spots
            dx = dy = 0.0
        rx, ry = rx + dx, ry + dy
        floors = (min(0.0, dx), min(0.0, dy))
    else:
        rx, ry, rw, rh = _build(P, cx, cy, W, H, split)
        floors = (0.0, 0.0)
    pos = P["pos"]
    soft = ~P["fixed"] & ~pre
    # Soft blocks take the leaf's aspect ratio (capped) at their own exact
    # area (a nudged cut can leave the leaf bigger or smaller than the block).
    asp = np.clip(np.maximum(rw, 1e-9) / np.maximum(rh, 1e-9), 1 / MAX_ASPECT, MAX_ASPECT)
    sw = np.sqrt(P["area"] * asp)
    w = np.where(soft, sw, pos[:, 2])
    h = np.where(soft, P["area"] / np.maximum(sw, 1e-12), pos[:, 3])
    x = rx + (rw - w) / 2
    y = ry + (rh - h) / 2
    x, y = np.where(pre, pos[:, 0], x), np.where(pre, pos[:, 1], y)
    # MIB groups share one shape: a fixed/preplaced member's if there is one
    # (10-05: 56/100 groups; copying only among soft members left 44 MIB
    # violations in 100 best-of results), else the soft members' mean width.
    for g in range(1, int(P["mib"].max(initial=0)) + 1):
        grp = P["mib"] == g
        m = np.where(grp & soft)[0]
        known = np.where(grp & ~soft)[0]
        if len(known) and len(m):
            ccx, ccy = x[m] + w[m] / 2, y[m] + h[m] / 2
            w[m], h[m] = pos[known[0], 2], pos[known[0], 3]
            x[m], y[m] = ccx - w[m] / 2, ccy - h[m] / 2
        elif len(m) > 1:
            cw = float(np.mean(w[m]))
            ccx, ccy = x[m] + w[m] / 2, y[m] + h[m] / 2
            w[m], h[m] = cw, P["area"][m] / cw
            x[m], y[m] = ccx - w[m] / 2, ccy - h[m] / 2
    return x, y, w, h, floors


def _legalize_grow(P, x, y, w, h, floors=(0.0, 0.0), rounds: int = 4):
    """rl/repack.py's _legalize, lowering an axis's floor by its overshoot
    (blocks in front of a preplaced block that don't fit between it and the
    origin) until nothing is pushed off its spot."""
    floors = list(floors)
    for _ in range(rounds):
        lx, ly, over = R._legalize(x, y, w, h, P, floors=tuple(floors))
        if over <= R.EPS:
            break
        ox, oy = R._legalize.over
        floors[0] -= ox
        floors[1] -= oy
    return lx, ly, over


# NONEG (10-02): no floor below the origin -- a candidate that only fits by
# growing past it is rejected (design 49: 17 blocks at x=-13, y=-10).
NONEG = (os.environ.get("FLOORSET_SLICE_NONEG", "0") != "0"
         or os.environ.get("FLOORSET_ORIGIN_UPSTREAM", "0") != "0")   # 10-06: upstream origin rule
# PULLBACK (10-03): when the best layout went past the origin, also try it
# with every movable block shifted back by the overhang (x, y or both) and
# re-legalized at the origin; the contest cost picks. Design 0: bottom-pinned
# blocks ended in a strip at y=-35 under preplaced block 17.
PULLBACK = os.environ.get("FLOORSET_SLICE_PULLBACK", "0") != "0"


def _pullbacks(P, cost_fn, scale, best):
    c, x, y, w, h = best
    dx, dy = max(0.0, -float(x.min())), max(0.0, -float(y.min()))
    if dx <= R.EPS and dy <= R.EPS:
        return []
    mov = ~P["preplaced"]
    out = []
    for sx, sy in {(dx, dy), (dx, 0.0), (0.0, dy)}:
        if sx <= R.EPS and sy <= R.EPS:
            continue
        lx, ly, over = R._legalize(np.where(mov, x + sx, x), np.where(mov, y + sy, y), w, h, P)
        if over > R.EPS or lx.min() < -R.EPS or ly.min() < -R.EPS:
            continue
        out.append((cost_fn(lx, ly, w, h), lx.copy(), ly.copy(), w.copy(), h.copy()))
    return out


def _evaluate(P, cost_fn, scale, x, y, w, h, floors=(0.0, 0.0)):
    if NONEG:
        lx, ly, over = R._legalize(x, y, w, h, P, floors=(0.0, 0.0))
        if lx.min() < -R.EPS or ly.min() < -R.EPS:
            over = max(over, 1.0)
        c = cost_fn(lx, ly, w, h)
        return c * (1.0 + 20.0 * over / scale), lx, ly, over <= R.EPS, R._legalize.slack
    # A layout already past the origin (grown or shifted) keeps its floor.
    floors = (min(floors[0], float(x.min())), min(floors[1], float(y.min())))
    lx, ly, over = _legalize_grow(P, x, y, w, h, floors)
    c = cost_fn(lx, ly, w, h)
    return c * (1.0 + 20.0 * over / scale), lx, ly, over <= R.EPS, R._legalize.slack


# Share of the polish budget for rl/legalize.py repair (walls + clusters),
# 10-05; env FLOORSET_REPAIR, 0 = off.
REPAIR = float(os.environ.get("FLOORSET_REPAIR", "0.3"))

# Env FLOORSET_SLICE_MODES (10-03 test): "tree" alone drops "shift", which
# translates the whole layout to line up preplaced blocks (design 0: -16, -33).
PRE_MODES = tuple(os.environ.get("FLOORSET_SLICE_MODES", "tree,shift").split(","))

# Env FLOORSET_SLICE_PINASPECT (10-07): also try roots with the pin box's aspect ratio.
PIN_ASPECT = os.environ.get("FLOORSET_SLICE_PINASPECT", "1") != "0"


def _roots(P, rl_aspect, total, box=None):
    """Root rectangles to try. The ground-truth bbox starts at (0, 0), so a
    right-pinned preplaced block's right edge IS its width (top-pinned: its
    height); every preplaced block's far edge is a lower bound."""
    pos, pre, bnd = P["pos"], P["preplaced"], P["bnd"]
    right, top = pos[:, 0] + pos[:, 2], pos[:, 1] + pos[:, 3]
    min_w = float(right[pre].max(initial=0.0))
    min_h = float(top[pre].max(initial=0.0))
    kw = right[pre & ((bnd & 2) != 0)]
    kh = top[pre & ((bnd & 4) != 0)]
    W_known = float(kw.max()) if len(kw) else None
    H_known = float(kh.max()) if len(kh) else None
    roots = []
    for fill in (1.0, 0.97, 0.93):
        a = total / fill
        cands = []
        if W_known and H_known:
            cands.append((W_known, H_known))
        if W_known:
            cands.append((W_known, a / W_known))
        if H_known:
            cands.append((a / H_known, H_known))
        # 10-07: the pin box's shape too (pins sit on the true outline)
        asps = (rl_aspect, 1.0, rl_aspect ** 0.5) + ((box[0] / box[1],) if box and PIN_ASPECT else ())
        for asp in asps:
            H = math.sqrt(a / asp)
            cands.append((a / H, H))
        for W, H in cands:
            W2, H2 = max(W, min_w), max(H, min_h)
            if (W2, H2) not in roots:
                roots.append((W2, H2))
    return roots


def _search(P, cx, cy, rl_aspect, total, evaluate, start, time_budget, box=None):
    """Best feasible (cost, x, y, w, h) over root rectangle / split rule /
    preplaced handling, and how many candidates were tried."""
    best, tried = None, 0
    pre_modes = PRE_MODES if P["preplaced"].any() else PRE_MODES[:1]
    roots = _roots(P, rl_aspect, total, box)
    if box and PIN_ASPECT:      # pin-box-shaped roots first (they would otherwise be cut by the time limit)
        a = box[0] / box[1]
        roots.sort(key=lambda r: abs(math.log(r[0] / r[1] / a)))
    for W, H in roots:
        for split in ("area", "gap"):
            for mode in pre_modes:
                if time.time() - start > time_budget and best is not None:
                    return best, tried
                x, y, w, h, floors = _candidate(P, cx, cy, W, H, split, mode)
                c, lx, ly, feas, _ = evaluate(x, y, w, h, floors)
                tried += 1
                if feas and (best is None or c < best[0]):
                    best = (c, lx.copy(), ly.copy(), w.copy(), h.copy())
    return best, tried


def slicing_polish(instance, positions: List[Position], time_budget: float = 1.0,
                   verbose: bool = False) -> List[Position]:
    from .compaction import compact

    start = time.time()
    P = R._problem(instance, positions)
    cost_fn = R._Cost(P, *R._ref_metrics(P))
    pos = P["pos"]
    cx, cy = pos[:, 0] + pos[:, 2] / 2, pos[:, 1] + pos[:, 3] / 2
    x0, y0 = pos[:, 0].min(), pos[:, 1].min()
    bw, bh = (pos[:, 0] + pos[:, 2]).max() - x0, (pos[:, 1] + pos[:, 3]).max() - y0
    total = float(P["area"].sum())
    scale = math.sqrt(total)

    def evaluate(x, y, w, h, floors=(0.0, 0.0)):
        return _evaluate(P, cost_fn, scale, x, y, w, h, floors)

    from .packer import pin_box
    box = pin_box(instance)
    box = box if box and all(math.isfinite(v) for v in box) else None
    best, tried = _search(P, cx, cy, bw / bh, total, evaluate, start, time_budget, box)
    if best is None:
        return positions
    best = R._fix_boundary(best, evaluate, cost_fn, P)
    if REPAIR > 0:
        from .legalize import repair
        best = repair(best, evaluate, cost_fn, P, time_budget=REPAIR * time_budget)
    cands = [best]
    if PULLBACK:
        cands += [R._fix_boundary(b, evaluate, cost_fn, P) for b in _pullbacks(P, cost_fn, scale, best)]
    raws = [R._as_positions(b)[1] for b in cands]
    old_cost, _ = R.contest_relative_cost(instance, positions, positions)
    scored = [(R.contest_relative_cost(instance, c, positions), c)
              for raw in raws for c in (raw, compact(raw, instance.constraints))]
    scored = [(c, p) for (c, ok), p in scored if ok]
    if verbose:
        print(f"  slicing: {tried} candidates, relative cost {old_cost:.4f} -> "
              f"{min((c for c, _ in scored), default=float('nan')):.4f}")
    if not scored:
        return positions
    new_cost, cand = min(scored, key=lambda t: t[0])
    return cand if new_cost < old_cost else positions

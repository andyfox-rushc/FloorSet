import random

import numpy as np
import pytest
import torch
from shapely.geometry import Polygon

from rl import packer as K
from rl import polygon as G
from rl import repack as R
from rl.data import synthetic_instance

L = G.shape_from_polygon(Polygon([(0, 0), (4, 0), (4, 1), (1, 1), (1, 3), (0, 3)]))
T = G.shape_from_polygon(Polygon([(0, 0), (3, 0), (3, 1), (2, 1), (2, 3), (1, 3), (1, 1), (0, 1)]))


def _instance(n=12, seed=0):
    rng = random.Random(seed)
    shapes, cons, tp, area = [], [], [], []
    for i in range(n):
        s = [L, T, G.transform(L, 2), None, None][i % 5]
        shapes.append(s)
        bnd = [0, 2, 4, 6, 0, 0, 0][i % 7]       # right / top / corner pins: the slide-outs
        if s is not None:
            cons.append([1, 0, 0, 0, bnd]); tp.append((-1, -1, s.w, s.h)); area.append(s.area)
        else:
            a = rng.uniform(1, 6)
            cons.append([0, 0, 0, 0, bnd]); tp.append((-1, -1, -1, -1)); area.append(a)
    inst = synthetic_instance(area, constraints=cons, target_positions=tp,
                              b2b_edges=[(i, (i + 1) % n, 1.0) for i in range(n)])
    inst.shapes = shapes
    return inst


def _pieces(inst, x, y, w, h):
    out = []
    for i, s in enumerate(inst.shapes):
        out.append(s.placed(x[i], y[i]) if s is not None else [(x[i], y[i], w[i], h[i])])
    assert min(min(p[0] for p in q) for q in out) >= -1e-9
    return out


@pytest.mark.parametrize("seed", range(6))
def test_contour_packing_never_overlaps_and_fills_notches(seed):
    inst = _instance(n=10 + 3 * seed, seed=seed)
    n = inst.block_count
    pos = [(3.0 * i, 0.0, s.w, s.h) if s is not None else (3.0 * i, 0.0, a ** 0.5, a ** 0.5)
           for i, (s, a) in enumerate(zip(inst.shapes, inst.area_targets.tolist()))]
    P = R._problem(inst, pos)
    Bt = K._Batch(P, 1.0, 1.0, (0.0, 0.0))
    rng = np.random.default_rng(seed)
    B = 64
    gp = np.stack([rng.permutation(n) for _ in range(B)])
    gm = np.stack([rng.permutation(n) for _ in range(B)])
    w = np.tile(P["pos"][:, 2], (B, 1)) * rng.uniform(0.7, 1.4, (B, n)) ** np.where(Bt.poly, 0, 1)
    h = np.tile(P["pos"][:, 2] * P["pos"][:, 3], (B, 1)) / w
    x, y, over = Bt.pack(gm, np.argsort(gp, axis=1), np.argsort(gm, axis=1), w, h)
    notch = 0
    for b in range(B):
        pcs = _pieces(inst, x[b], y[b], w[b], h[b])
        for i in range(n):
            for j in range(i + 1, n):
                assert G.rects_overlap_area(pcs[i], pcs[j]) < 1e-9
                bi = (x[b, i], y[b, i], w[b, i], h[b, i])
                bj = (x[b, j], y[b, j], w[b, j], h[b, j])
                notch += G.rects_overlap_area([bi], [bj]) > 1e-9     # boxes overlap: a notch is used
    assert notch > 0


def test_exact_cost_accepts_notch_layout_and_rejects_overlap():
    inst = _instance(n=2)        # an L and a T
    inst.shapes = [L, None]
    inst.constraints[1] = torch.tensor([0, 0, 0, 0, 0.0])
    inst.area_targets[1] = 4.0
    ref = [(0.0, 0.0, 4.0, 3.0), (5.0, 0.0, 2.0, 2.0)]
    in_notch = [(0.0, 0.0, 4.0, 3.0), (1.0, 1.0, 2.0, 2.0)]       # inside the L's notch
    clash = [(0.0, 0.0, 4.0, 3.0), (0.5, 0.5, 2.0, 2.0)]
    assert R.contest_relative_cost(inst, in_notch, ref)[1]
    assert not R.contest_relative_cost(inst, clash, ref)[1]

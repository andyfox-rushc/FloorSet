#!/usr/bin/env python3
"""
FloorSet-Prime grid flow (10-21): RPolyPlan-style global floorplan
(rl/global_alm.py) from the learned centroid seeds, then JigsawPlanner-style
legalization on the unit grid (rl/grid_assign.py): exact areas, no white
space, no overlap, non-slicing layouts allowed.

  python prime_evaluate.py --evaluate prime_grid.py
"""

import os
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))

from prime_evaluate import PrimeOptimizer  # noqa: E402
from rl import global_alm as GA  # noqa: E402
from rl import grid_assign as GR  # noqa: E402
from rl import partition as PT  # noqa: E402
from rl.prime_placer import load, predict  # noqa: E402

PLACER = os.environ.get("FLOORSET_PRIME_PLACER", str(Path(__file__).resolve().parent / "checkpoints" / "prime_placer.pt"))
_NET = []


def _learned():
    if not _NET:
        _NET.append(load(PLACER) if PLACER else None)
    return _NET[0]


def grid_solve(P, part, seeds, verbose=False):
    """(GridLayout, stats) for one design."""
    g = GA.solve(P, part, seeds, verbose=verbose)
    L = GR.GridLayout(P, part)
    centre = np.stack([(g.box[:, 0] + g.box[:, 2]) / 2, (g.box[:, 1] + g.box[:, 3]) / 2], 1)
    rigid = [i for i in part.operands if i in part.options]
    soft = [i for i in part.operands if i not in part.options]
    claim = L.claims(soft, g.box, g.sigma)
    near = {}                            # rigid cluster members: touch the group's soft claims
    for grp in P.groups:
        members = [int(b) for b in grp]
        m = np.zeros((L.H, L.W), dtype=bool)
        for b in members:
            if b in claim:
                m |= claim[b]
        for b in members:
            if b in rigid:
                near[b] = m
    failed = L.place_rigid(rigid, centre, near)
    L.assign(soft, claim, centre)
    L.refill()                           # stray pieces -> white space -> short blocks grow back
    moves = L.balance()
    for _ in range(4):                   # pinches / new fragments, then counts again
        if not (L.fix_pinches() + L.refill()):
            break
        moves = L.balance()
    recuts = L.cleanup()
    cfix = L.corner_fix()
    if cfix:
        moves = L.balance()
    relg = L.fix_bad_blocks()
    st = L.stats()
    st.update(rigid_failed=len(failed), moves=moves, recuts=recuts, corner_moves=cfix, relegalized=relg, alm_violation=g.violation)
    return L, st


class PrimeGrid(PrimeOptimizer):
    def solve(self, block_count, area_targets, b2b_connectivity, p2b_connectivity, pins_pos,
              constraints, target_polygons):
        args = (block_count, area_targets, b2b_connectivity, p2b_connectivity, pins_pos, constraints,
                target_polygons)
        P = PT.Problem(*args)
        net = _learned()
        seeds = np.clip(predict(net, *args), 0, [P.W, P.H])
        part = PT.Partitioner(P, seeds)
        L, st = grid_solve(P, part, seeds, verbose=self.verbose)
        if self.verbose:
            print(f"  grid: {st}")
        return L.polygons()

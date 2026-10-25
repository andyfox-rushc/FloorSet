#!/usr/bin/env python3
"""
FloorSet-Prime zero-white-space optimizer (rl/partition.py): the frame comes
from the pins, rigid pieces go near seed centroids, and the free region is
divided exactly among the soft blocks by an annealed slicing tree.

Seeds: the learned centroid predictor (rl/prime_placer.py) when its
checkpoint exists, and the analytic placer (rl/analytic.py global_place) at
utilization 1.0, i.e. inside the exact frame; restarts alternate between them.

  python prime_evaluate.py --evaluate prime_partition.py
"""

import os
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))

from prime_evaluate import PrimeOptimizer  # noqa: E402
from rl import partition as PT  # noqa: E402
from rl.analytic import global_place  # noqa: E402
from rl.data import from_solve_args  # noqa: E402
from rl.prime_adapter import to_lite  # noqa: E402
from rl.prime_placer import load, predict  # noqa: E402

# 10-21 A/B, 12 designs: 20 -> 80 s at n = 77 (x (n/77)^2), mean cost 1.211 -> 1.160, HPWL gap 0.116
TIME = float(os.environ.get("FLOORSET_PARTITION_TIME", "80"))
# Learned centroid seeds (rl/prime_placer.py); missing file = analytic seeds only
PLACER = os.environ.get("FLOORSET_PRIME_PLACER", str(Path(__file__).resolve().parent / "checkpoints" / "prime_placer.pt"))
_NET = []


def _learned():
    if not _NET:
        _NET.append(load(PLACER) if PLACER else None)
    return _NET[0]
RESTARTS = int(os.environ.get("FLOORSET_PARTITION_RESTARTS", "6"))


def layout_seeds(layout, W, H):
    """Centroids of another flow's layout, its bounding box scaled onto the
    [0, W] x [0, H] frame (the packer flow's arrangement, minus its white space)."""
    from shapely.geometry import Polygon
    polys = [Polygon(v) for v in layout]
    c = np.array([p.centroid.coords[0] for p in polys])
    lw = max(p.bounds[2] for p in polys)
    lh = max(p.bounds[3] for p in polys)
    return np.clip(c * np.array([W / lw, H / lh]), 0, [W, H])


def seed_sets(block_count, area_targets, b2b_connectivity, p2b_connectivity, pins_pos, constraints,
              target_polygons, P):
    """Centroid seed arrays for the restarts: learned (when its checkpoint
    exists) and analytic, both clipped into the frame."""
    prob = to_lite(block_count, area_targets, b2b_connectivity, p2b_connectivity, pins_pos,
                   constraints, target_polygons)
    inst = from_solve_args(prob.block_count, prob.area_targets, prob.b2b, prob.p2b, prob.pins_pos,
                           prob.constraints, prob.target_positions)
    gp = global_place(inst, util=1.0, iters=600)
    seeds = np.array([(x + w / 2, y + h / 2) for x, y, w, h in gp[:block_count]])
    seeds = np.clip(seeds, 0, [P.W, P.H])
    net = _learned()
    if net is None:
        return [seeds]
    learned = predict(net, block_count, area_targets, b2b_connectivity, p2b_connectivity,
                      pins_pos, constraints, target_polygons)
    return [np.clip(learned, 0, [P.W, P.H]), seeds]      # learned first; analytic kept for diversity


class PrimePartition(PrimeOptimizer):
    def __init__(self, verbose: bool = False):
        super().__init__(verbose)
        self.extra_seeds = None        # set per design by prime_combined.py (e.g. packer centroids)

    def solve(self, block_count, area_targets, b2b_connectivity, p2b_connectivity, pins_pos,
              constraints, target_polygons):
        P = PT.Problem(block_count, area_targets, b2b_connectivity, p2b_connectivity, pins_pos,
                       constraints, target_polygons)
        sets = seed_sets(block_count, area_targets, b2b_connectivity, p2b_connectivity, pins_pos,
                         constraints, target_polygons, P)
        if self.extra_seeds is not None:
            sets = [self.extra_seeds] + sets
        from my_optimizer import _pmap
        return PT.solve_parallel(P, sets, max(10.0, TIME * (block_count / 77) ** 2), RESTARTS,
                                 pmap=_pmap, verbose=self.verbose)

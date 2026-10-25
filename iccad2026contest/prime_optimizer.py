#!/usr/bin/env python3
"""
FloorSet-Prime optimizer: the Lite pipeline (my_optimizer.MyOptimizer: RL
layout, best-of candidates, packer, finish) run on the rectangle version of
the design built by rl/prime_adapter.py, then converted back to polygons.

  python prime_evaluate.py --evaluate prime_optimizer.py
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

from my_optimizer import MyOptimizer  # noqa: E402
from prime_evaluate import PrimeOptimizer  # noqa: E402
from rl.prime_adapter import from_lite, instance_shapes, to_lite  # noqa: E402


class PrimeViaLite(PrimeOptimizer):
    def __init__(self, verbose: bool = False):
        super().__init__(verbose)
        self.lite = MyOptimizer(verbose=verbose)

    def solve(self, block_count, area_targets, b2b_connectivity, p2b_connectivity, pins_pos,
              constraints, target_polygons):
        prob = to_lite(block_count, area_targets, b2b_connectivity, p2b_connectivity, pins_pos,
                       constraints, target_polygons)
        positions = self.lite.solve(prob.block_count, prob.area_targets, prob.b2b, prob.p2b,
                                    prob.pins_pos, prob.constraints, prob.target_positions,
                                    shapes=instance_shapes(prob))
        return from_lite(prob, positions)

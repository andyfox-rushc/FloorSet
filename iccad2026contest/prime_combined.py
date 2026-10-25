#!/usr/bin/env python3
"""
FloorSet-Prime best-of-both optimizer: the zero-white-space partition flow
(prime_partition.py) and the rectangle-packing flow (prime_optimizer.py:
adapter + contour packer), the cheaper layout kept per design.

The packer runs first and its centroids (scaled into the exact frame) seed
one set of partition restarts. No ground truth at solve time: the two are
compared on the contest formula with the partition layout as the reference
(prime_evaluate.relative_cost).
FLOORSET_COMBINED=partition|packer runs one flow only.

  python prime_evaluate.py --evaluate prime_combined.py
"""

import os
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

from prime_evaluate import PrimeOptimizer, relative_cost  # noqa: E402
from prime_optimizer import PrimeViaLite  # noqa: E402
from prime_partition import PrimePartition, layout_seeds  # noqa: E402
from rl import partition as PT  # noqa: E402

MODE = os.environ.get("FLOORSET_COMBINED", "both")
SEED_FROM_PACKER = os.environ.get("FLOORSET_SEED_FROM_PACKER", "1") != "0"


class PrimeCombined(PrimeOptimizer):
    def __init__(self, verbose: bool = False):
        super().__init__(verbose)
        self.partition = PrimePartition(verbose)
        self.packer = PrimeViaLite(verbose)

    def solve(self, *args):
        # packer first: its centroids, scaled into the frame, seed one partition restart set
        b = self.packer.solve(*args) if MODE in ("both", "packer") else None
        self.partition.extra_seeds = None
        if b is not None and SEED_FROM_PACKER:
            P = PT.Problem(*args)
            self.partition.extra_seeds = layout_seeds(b, P.W, P.H)
        a = self.partition.solve(*args) if MODE in ("both", "partition") else None
        if a is None and b is None:
            b = self.packer.solve(*args)
        if a is None or b is None:
            return a if a is not None else b
        ref = a
        ca, ok_a = relative_cost(args, a, ref)
        cb, ok_b = relative_cost(args, b, ref)
        if self.verbose:
            print(f"  best of both: partition {ca:.4f} ({ok_a}), packer {cb:.4f} ({ok_b})")
        if ok_a and (not ok_b or ca <= cb):
            return a
        return b if ok_b else a

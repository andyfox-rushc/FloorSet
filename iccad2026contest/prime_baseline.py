#!/usr/bin/env python3
"""
FloorSet-Prime shelf-packing baseline: a floor score for the real solver and
an end-to-end check of the prime_evaluate.py harness. Preplaced polygons stay
where they are; every other block (fixed: its target shape, soft: a square of
its target area) is shelf-packed by bounding box above the preplaced ones.
No wirelength awareness at all.

  python prime_evaluate.py --evaluate prime_baseline.py
"""

import math
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

from prime_evaluate import PrimeOptimizer  # noqa: E402
from rl import polygon as G  # noqa: E402


class ShelfBaseline(PrimeOptimizer):
    def solve(self, block_count, area_targets, b2b_connectivity, p2b_connectivity, pins_pos,
              constraints, target_polygons):
        out = [None] * block_count
        y_floor = 0.0
        items = []
        for i in range(block_count):
            if constraints[i, 1]:
                out[i] = list(target_polygons[i])
                y_floor = max(y_floor, max(y for _, y in target_polygons[i]))
            elif constraints[i, 0]:
                items.append((i, G.shape_from_vertices(target_polygons[i])))
            else:
                s = math.sqrt(float(area_targets[i]))
                items.append((i, G.rect_shape(s, s)))
        width = 1.2 * math.sqrt(sum(s.area for _, s in items) or 1.0)
        x = 0.0
        y = y_floor
        row_h = 0.0
        for i, s in sorted(items, key=lambda t: -t[1].h):
            if x > 0 and x + s.w > width:
                x, y, row_h = 0.0, y + row_h, 0.0
            out[i] = list(s.polygon(x, y).exterior.coords[:-1])
            x += s.w
            row_h = max(row_h, s.h)
        return out

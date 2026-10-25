"""
Detects whether a solved layout has broken into more than one spatially
isolated group of blocks -- the block-50 / satellite-group failure pattern
diagnosed on test-98 (a lone corner-pinned block, and a whole right-boundary
group, both ending up far from the interior mass with nothing able to pull
them in). `area_gap` alone can look fine on average while still hiding a
handful of badly-stranded cases, so this checks for the failure directly
instead of inferring it from an aggregate number.

Two rectangles are "connected" if the true gap between them (0 if they
already overlap or touch) is within `tolerance` -- ordinary adjacent blocks
have gap ~0, so a small tolerance only merges genuinely-touching content,
never masks a real strand. Connected components are found via union-find;
more than one component, with the gap to the dominant (largest-bbox)
component exceeding a threshold, counts as stranded. The default threshold
is the STRAY component's own largest block dimension (not the largest block
anywhere in the layout -- this dataset has some very large elongated blocks,
and using a global max made the threshold too lenient in practice: on real
test-98 data, block 50's actual 43.6-unit gap fell under a global-max
threshold and went unflagged). Scaling the threshold to the stray piece's
own size instead answers the right question -- "is this gap bigger than
normal spacing for something this size" -- rather than being inflated by an
unrelated giant block elsewhere in the layout.

Usage: python scripts/detect_stranded_components.py --test-id 98
"""
import argparse
import math
import sys
from pathlib import Path
from typing import List, Optional, Tuple

sys.path.insert(0, str(Path(__file__).parent.parent))

Position = Tuple[float, float, float, float]


def _rect_gap(a: Position, b: Position) -> float:
    """True (Euclidean) gap between two axis-aligned rectangles; 0 if they
    overlap or touch."""
    ax, ay, aw, ah = a
    bx, by, bw, bh = b
    xgap = max(0.0, ax - (bx + bw), bx - (ax + aw))
    ygap = max(0.0, ay - (by + bh), by - (ay + ah))
    return math.hypot(xgap, ygap)


class _UnionFind:
    def __init__(self, n: int):
        self.parent = list(range(n))

    def find(self, x: int) -> int:
        while self.parent[x] != x:
            self.parent[x] = self.parent[self.parent[x]]
            x = self.parent[x]
        return x

    def union(self, a: int, b: int) -> None:
        ra, rb = self.find(a), self.find(b)
        if ra != rb:
            self.parent[ra] = rb


def find_components(positions: List[Position], tolerance: float = 1e-6) -> List[List[int]]:
    n = len(positions)
    uf = _UnionFind(n)
    for i in range(n):
        for j in range(i + 1, n):
            if _rect_gap(positions[i], positions[j]) <= tolerance:
                uf.union(i, j)
    groups: dict = {}
    for i in range(n):
        groups.setdefault(uf.find(i), []).append(i)
    return list(groups.values())


def _bbox_area(positions: List[Position], idxs: List[int]) -> float:
    x0 = min(positions[i][0] for i in idxs)
    y0 = min(positions[i][1] for i in idxs)
    x1 = max(positions[i][0] + positions[i][2] for i in idxs)
    y1 = max(positions[i][1] + positions[i][3] for i in idxs)
    return (x1 - x0) * (y1 - y0)


def detect_stranded(positions: List[Position], threshold: Optional[float] = None) -> dict:
    """Returns {"stranded": bool, "component_count": int, "components": [...]}
    -- one entry per non-dominant component with its block indices, bbox
    area, and gap to the dominant (largest-bbox) component. `threshold`
    overrides the per-component default (see module docstring) with one
    fixed value for every component, when the caller wants that instead."""
    n = len(positions)
    if n == 0:
        return {"stranded": False, "component_count": 0, "components": []}

    components = find_components(positions)
    if len(components) <= 1:
        return {"stranded": False, "component_count": 1, "components": []}

    components.sort(key=lambda idxs: -_bbox_area(positions, idxs))
    dominant = components[0]
    info = []
    any_stranded = False
    for comp in components[1:]:
        min_gap = min(_rect_gap(positions[i], positions[j]) for i in comp for j in dominant)
        comp_threshold = threshold
        if comp_threshold is None:
            comp_threshold = max(max(positions[i][2], positions[i][3]) for i in comp)
        stranded = min_gap > comp_threshold
        any_stranded = any_stranded or stranded
        info.append({
            "blocks": comp,
            "bbox_area": _bbox_area(positions, comp),
            "gap_to_main": min_gap,
            "threshold": comp_threshold,
            "stranded": stranded,
        })
    return {"stranded": any_stranded, "component_count": len(components), "components": info}


if __name__ == "__main__":
    import torch
    from iccad2026_evaluate import ContestEvaluator

    p = argparse.ArgumentParser()
    p.add_argument("--test-id", type=int, default=0)
    p.add_argument("--data-path", default="../")
    p.add_argument("--optimizer", default="my_optimizer.py")
    args = p.parse_args()

    evaluator = ContestEvaluator(args.data_path, verbose=False)
    evaluator._load_dataset()
    optimizer = evaluator._load_optimizer(args.optimizer)

    sample = evaluator.dataset[args.test_id]
    inputs, labels = sample["input"], sample["label"]
    area_target, b2b_conn, p2b_conn, pins_pos, constraints = inputs
    block_count = int((area_target != -1).sum().item())
    _, target_pos = evaluator._extract_baseline(args.test_id, labels, b2b_conn, p2b_conn,
                                                 pins_pos, block_count)

    opt_target_pos = torch.full((block_count, 4), -1.0)
    for i in range(block_count):
        if constraints[i, 1] != 0:
            opt_target_pos[i] = torch.tensor(list(target_pos[i]))
        elif constraints[i, 0] != 0:
            opt_target_pos[i, 2] = target_pos[i][2]
            opt_target_pos[i, 3] = target_pos[i][3]

    positions = optimizer.solve(block_count, area_target, b2b_conn, p2b_conn, pins_pos,
                                 constraints, opt_target_pos)
    result = detect_stranded(positions)
    print(f"test {args.test_id}: component_count={result['component_count']} "
          f"stranded={result['stranded']}")
    for comp in result["components"]:
        print(f"  stray component blocks={comp['blocks']} bbox_area={comp['bbox_area']:.1f} "
              f"gap_to_main={comp['gap_to_main']:.1f} threshold={comp['threshold']:.1f} "
              f"stranded={comp['stranded']}")

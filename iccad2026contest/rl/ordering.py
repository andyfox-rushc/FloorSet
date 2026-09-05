"""
Deterministic placement order + per-block role, computed once per instance
before any RL rollout. This is pure logic (no learning): it decides *what
order* blocks are placed in and *which hard-constraint role* each one plays,
so that env.py can guarantee overlap-free / MIB-exact placement by
construction.

Roles:
    'preplaced'    - exact (x, y, w, h) given; never enters the action space.
    'fixed'        - exact (w, h) given; only position is chosen.
    'mib_follower' - shape copied from its MIB group's leader; only position
                     is chosen.
    'free'         - both aspect ratio (-> w, h) and position are chosen.

Ordering heuristic (mirrors AlphaChip's descending-area macro placement,
"ties broken by topological sort" so connected macros land near each other
in the sequence -- see Mirhoseini/Goldie, "Chip Placement with Deep
Reinforcement Learning", section on macro ordering):
    1. Preplaced blocks first (order irrelevant, no action taken).
    2. Remaining blocks are grouped into MIB units (leader + shape-copying
       followers, kept contiguous).
    3. Units are then ordered by a greedy, connectivity-aware walk: start
       from the largest unit; at each step, prefer whichever remaining unit
       has the strongest accumulated b2b connectivity to everything already
       placed (a virtual "same cluster" edge counts as very strong
       connectivity, so cluster members still end up adjacent); when no
       remaining unit is connected to the placed set at all, fall back to
       the largest remaining unit (starts a new connected component). This
       both mimics the paper's ordering and gives the env's cluster-touch
       logic more same-cluster neighbors to actually touch against.
"""

from collections import defaultdict
from dataclasses import dataclass
from typing import Dict, List, Optional

import torch

# Dominates any real b2b edge weight so same-cluster units are always
# preferred over merely-connected ones, without needing a separate grouping
# pass -- see _order_units_by_connectivity.
_CLUSTER_BONUS = 1e6


@dataclass
class PlacementStep:
    block_idx: int
    role: str
    mib_leader: int = -1  # valid only when role == 'mib_follower'
    boundary_code: int = 0
    cluster_id: int = 0


@dataclass
class PlacementPlan:
    order: List[PlacementStep]

    def block_order(self) -> List[int]:
        return [s.block_idx for s in self.order]


def _order_units_by_connectivity(
    units: List[List[int]],
    area_targets: torch.Tensor,
    cluster_id: torch.Tensor,
    b2b_connectivity: Optional[torch.Tensor],
) -> List[List[int]]:
    """Greedy max-connectivity walk over units (see module docstring)."""
    n_units = len(units)
    if n_units <= 1:
        return list(units)

    member_to_unit: Dict[int, int] = {}
    for u_idx, unit in enumerate(units):
        for m in unit:
            member_to_unit[m] = u_idx

    unit_adj: Dict[tuple, float] = defaultdict(float)

    if b2b_connectivity is not None and b2b_connectivity.numel() > 0:
        valid = b2b_connectivity[b2b_connectivity[:, 0] >= 0]
        for edge in valid:
            i, j, w = int(edge[0]), int(edge[1]), float(edge[2])
            if i not in member_to_unit or j not in member_to_unit:
                continue
            ui, uj = member_to_unit[i], member_to_unit[j]
            if ui == uj:
                continue
            key = (ui, uj) if ui < uj else (uj, ui)
            unit_adj[key] += w

    cluster_of_unit = [int(cluster_id[unit[0]]) for unit in units]
    for u_idx in range(n_units):
        cid = cluster_of_unit[u_idx]
        if not cid:
            continue
        for v_idx in range(u_idx + 1, n_units):
            if cluster_of_unit[v_idx] == cid:
                key = (u_idx, v_idx)
                unit_adj[key] += _CLUSTER_BONUS

    remaining = set(range(n_units))
    frontier_score: Dict[int, float] = {}
    order_idxs: List[int] = []

    while remaining:
        candidates = [u for u in remaining if frontier_score.get(u, 0.0) > 0.0]
        if candidates:
            next_u = max(candidates, key=lambda u: (frontier_score[u], float(area_targets[units[u][0]])))
        else:
            next_u = max(remaining, key=lambda u: float(area_targets[units[u][0]]))

        order_idxs.append(next_u)
        remaining.discard(next_u)
        for v in remaining:
            key = (next_u, v) if next_u < v else (v, next_u)
            w = unit_adj.get(key)
            if w:
                frontier_score[v] = frontier_score.get(v, 0.0) + w

    return [units[u] for u in order_idxs]


def compute_order(
    constraints: torch.Tensor,
    area_targets: torch.Tensor,
    b2b_connectivity: Optional[torch.Tensor] = None,
) -> PlacementPlan:
    n = area_targets.shape[0]
    ncols = constraints.shape[1] if constraints.dim() > 1 else 0

    def col(j):
        return constraints[:, j] if ncols > j else torch.zeros(n)

    fixed = col(0) != 0
    preplaced = col(1) != 0
    mib_id = col(2).long()
    cluster_id = col(3).long()
    boundary = col(4).long()

    steps: List[PlacementStep] = []

    preplaced_idxs = [i for i in range(n) if bool(preplaced[i])]
    for i in preplaced_idxs:
        steps.append(PlacementStep(i, 'preplaced', boundary_code=int(boundary[i]),
                                    cluster_id=int(cluster_id[i])))

    remaining = [i for i in range(n) if not bool(preplaced[i])]

    # Step 1: form MIB units (leader = largest-area member, followers after
    # it) covering every remaining block. MIB grouping is independent of
    # clustering -- a block can be in an MIB group without being clustered,
    # or vice versa.
    mib_groups: Dict[int, List[int]] = defaultdict(list)
    singles = []
    for i in remaining:
        if mib_id[i] != 0:
            mib_groups[int(mib_id[i])].append(i)
        else:
            singles.append(i)

    units: List[List[int]] = []
    for members in mib_groups.values():
        # A fixed-shape member's dimensions are a hard constraint and cannot
        # be overridden by MIB copying, so it always leads its unit (its
        # shape is what the others should converge to); any *other*
        # fixed-shape member in the same group keeps its own exact
        # dimensions too (see role assignment below) rather than copying the
        # leader's, since a fixed block's shape is non-negotiable even if
        # that leaves it geometrically inconsistent with its MIB siblings.
        fixed_members = [i for i in members if bool(fixed[i])]
        if fixed_members:
            leader = fixed_members[0]
            rest = sorted((i for i in members if i != leader), key=lambda i: -float(area_targets[i]))
            units.append([leader] + rest)
        else:
            units.append(sorted(members, key=lambda i: -float(area_targets[i])))
    for i in singles:
        units.append([i])

    # Step 2: order units by greedy connectivity (falls back to plain
    # descending-area when there's no b2b/cluster signal at all -- see
    # _order_units_by_connectivity).
    ordered_units = _order_units_by_connectivity(units, area_targets, cluster_id, b2b_connectivity)

    for unit in ordered_units:
        leader = unit[0]
        role = 'fixed' if bool(fixed[leader]) else 'free'
        steps.append(PlacementStep(leader, role, boundary_code=int(boundary[leader]),
                                    cluster_id=int(cluster_id[leader])))
        for follower in unit[1:]:
            if bool(fixed[follower]):
                # Own dimensions are a hard constraint; never overridden
                # by MIB shape-copying (see unit construction above).
                steps.append(PlacementStep(follower, 'fixed',
                                            boundary_code=int(boundary[follower]),
                                            cluster_id=int(cluster_id[follower])))
            else:
                steps.append(PlacementStep(follower, 'mib_follower', mib_leader=leader,
                                            boundary_code=int(boundary[follower]),
                                            cluster_id=int(cluster_id[follower])))

    assert len(steps) == n, f"placement plan covers {len(steps)} of {n} blocks"
    return PlacementPlan(order=steps)

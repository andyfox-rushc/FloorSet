"""
Deterministic post-placement compaction.

The RL policy currently has no learned notion of compaction -- it spreads
blocks across however much working canvas rl/env.py's CANVAS_PADDING gives
it, regardless of training progress (see algorithm.md's HPWL/area_gap
diagnosis: a real placement's bbox was measured at *exactly* the 2.25x
canvas ceiling, packing efficiency ~44%). This module closes that gap with
a classic, non-learned "gravity" compaction: slide every movable block
toward smaller x, then smaller y, stopping exactly where an already-placed
block blocks it.

Applied to the final positions returned by rl/finetune.py's
finetune_and_solve(), AND (since 2026-09-12) to the positions
rl/reward.py's pretraining_reward/inference_reward score -- both need to
agree, since finetune_and_solve always compacts before returning and a
training reward computed on the pre-compaction layout was giving the
policy zero gradient signal toward the one thing (packing tightly)
compaction can't do for it. See rl/reward.py's _evaluate docstring.

Safety argument (why this can never introduce an overlap):
    - Preplaced blocks are marked immovable and only ever act as static
      blockers.
    - A nonzero cluster group (grouping constraint requires its members to
      stay mutually touching) is never shifted per-block -- that could
      break the touching arrangement -- but the whole group CAN slide as
      one rigid unit (see _shift_cluster_x/_shift_cluster_y below), since a
      pure translation preserves every member's relative position exactly.
      Before this fix, cluster members were marked immovable and only ever
      acted as static blockers, same as preplaced blocks -- correct for
      safety but far too conservative: a cluster placed anywhere in
      rl/env.py's padded working canvas, with nothing between it and the
      rest of the layout, could never move an inch closer, no matter how
      much slack surrounded it (see algorithm.md's compaction diagnosis on
      test-98 -- this was verified to strand whole clusters 60-100+ units
      from where they could safely sit).
    - An ordinary movable block's new coordinate is always
      min(current, max(floor, tightest real blocker)) -- i.e. it only ever
      moves toward the floor and only as far as the current (possibly
      already-updated-this-pass) position of whatever's blocking it. Since
      the very first placement was overlap-free, and every subsequent move
      only tightens a block against an already-verified-safe blocker (or
      leaves it unmoved), the invariant "no overlap" holds by induction
      through every intermediate step -- never just at the end.
    - Right/top-pinned blocks (_shift_group_x/_shift_group_y) use the same
      min(current, max(bound, floor)) pattern, just with an extra `desired`
      term (see below) also floored into the max -- so they too only ever
      move toward smaller coordinates, by no more than their own
      `_bound_x`/`_bound_y` allows.

Right/top edge groups -- why they need special handling, and the bug the
current version replaced: env.py places boundary-pinned blocks flush
against the *working canvas*'s edge (self.x_max / self.y_max), not the
eventual tight solution bbox's edge -- so a right- or top-pinned block sits
far out from wherever the rest of the layout ends up compacting to, and
the plain per-block sweep above (which only ever moves things toward the
SAME floor_x/floor_y corner) can never pull it in: moving it would either
do nothing (if it's already at the shared minimum -- true for left/bottom
pins, which coincide with the sweep's own direction) or require moving it
*away* from that corner instead, which the per-block sweep isn't built to
do.

An earlier version of _shift_group_x/_shift_group_y treated every block
sharing a given edge pin as one rigid group that had to move by the
*smallest* member's individually-allowed distance, reasoning that they
"must all stay flush with each other." That reasoning was wrong: they
don't need to move together, only to each independently reach the same
final bbox edge -- and forcing them to move in lockstep meant one
more-blocked member silently capped how far every *other*, less-blocked
member was allowed to go, stranding those members short of the edge. Since
that final edge is frequently defined by some other, unrelated immovable
block (not the boundary group at all), this was verified on real
validation cases to make boundary_violations go UP after compaction (e.g.
5->8 on the first case checked) -- compaction was never supposed to be
able to break a boundary constraint the raw RL placement already
satisfied. The current version instead computes, each pass, the tightest
edge the group could jointly reach (each member's own safe bound, in
edge-space: `_bound_x(i) + w_i`) and the tightest edge non-group content
already occupies, takes the max of both (never looser than either
constraint), and pulls every group member to that one shared value
independently -- clamped by its own bound and never past its current
position -- so no single member's limit strands any other member short of
the edge.
"""

from typing import List, Optional, Tuple

import torch

Position = Tuple[float, float, float, float]

_EPS = 1e-9


def _bound_x(pos: List[List[float]], i: int, exclude: frozenset = frozenset()) -> float:
    """Tightest x a block could slide left to, i.e. the rightmost edge among
    blocks currently at-or-left of it with overlapping y-range (0.0 if
    nothing blocks it). `exclude` skips a set of indices entirely -- used to
    ignore a block's own cluster-mates, which move with it and so can never
    legitimately block it (see _shift_cluster_x)."""
    xi, yi, wi, hi = pos[i]
    bound = 0.0
    for j in range(len(pos)):
        if j == i or j in exclude:
            continue
        xj, yj, wj, hj = pos[j]
        if xj > xi + _EPS:
            continue
        if min(yi + hi, yj + hj) - max(yi, yj) <= _EPS:
            continue
        bound = max(bound, xj + wj)
    return bound


def _bound_y(pos: List[List[float]], i: int, exclude: frozenset = frozenset()) -> float:
    """Mirror of _bound_x for the y axis."""
    xi, yi, wi, hi = pos[i]
    bound = 0.0
    for j in range(len(pos)):
        if j == i or j in exclude:
            continue
        xj, yj, wj, hj = pos[j]
        if yj > yi + _EPS:
            continue
        if min(xi + wi, xj + wj) - max(xi, xj) <= _EPS:
            continue
        bound = max(bound, yj + hj)
    return bound


def _shift_cluster_x(pos: List[List[float]], group: List[int], floor: float) -> None:
    """Slide an entire cluster group left as one rigid unit, by the largest
    amount safe for every member simultaneously -- the min over each
    member's own individually-allowed slide (ignoring other group members
    as blockers, since they move with it). This is the opposite pattern
    from _shift_group_x's shared-edge pull: a cluster's members must keep
    their exact relative arrangement (that's what "mutually touching"
    means), so only a uniform translation is safe -- capping every member
    to whichever one has the least room preserves that, and can never
    overlap anything since each member's new x is never less than its own
    (non-group) bound."""
    if not group:
        return
    group_set = frozenset(group)
    slide = min(pos[i][0] - max(_bound_x(pos, i, group_set), floor) for i in group)
    if slide > _EPS:
        for i in group:
            pos[i][0] -= slide


def _shift_cluster_y(pos: List[List[float]], group: List[int], floor: float) -> None:
    """Mirror of _shift_cluster_x for the y axis."""
    if not group:
        return
    group_set = frozenset(group)
    slide = min(pos[i][1] - max(_bound_y(pos, i, group_set), floor) for i in group)
    if slide > _EPS:
        for i in group:
            pos[i][1] -= slide


def _sweep_x(pos: List[List[float]], immovable: List[bool], floor: float) -> None:
    order = sorted(range(len(pos)), key=lambda i: pos[i][0])
    for i in order:
        if immovable[i]:
            continue
        pos[i][0] = min(pos[i][0], max(_bound_x(pos, i), floor))


def _sweep_y(pos: List[List[float]], immovable: List[bool], floor: float) -> None:
    order = sorted(range(len(pos)), key=lambda i: pos[i][1])
    for i in order:
        if immovable[i]:
            continue
        pos[i][1] = min(pos[i][1], max(_bound_y(pos, i), floor))


def _shift_group_x(pos: List[List[float]], group: List[int], floor: float) -> None:
    """Pull every right-pinned block to the tightest edge the group could
    jointly reach, independently (see module docstring). Members don't
    need to move by the same amount to end up mutually flush: they only
    need to each reach the same target x+w, which falls out automatically
    once each one independently closes its own gap to that shared target
    -- clamped so this can never overlap anything, and never moves a
    member past its current position, so this stays a pure shrink like
    the rest of compaction.

    Bug this replaces: shifting every member by the *smallest* member's
    allowed distance stops the whole group short whenever any one member is
    more blocked than the others (verified on real validation cases: this
    was making boundary_violations go UP after compaction, e.g. 5->8 on one
    of the first checked, not down -- compaction was never supposed to be
    able to break a boundary constraint the raw placement already
    satisfied). A member that's still at the far canvas edge because it
    fell back to a free placement (see rl/env.py's boundary_fallback_count)
    is left alone here too, same as before: `target` only ever moves it
    inward, never out to newly *reach* an edge it never touched."""
    if not group:
        return
    group_set = set(group)
    # The shared edge can't be tighter than: (a) any group member's own
    # safe bound -- in x+w space, that's _bound_x(i) + w_i, since
    # _bound_x returns an x-coordinate, not an edge -- or (b) whatever
    # x+w the rest of the layout (non-group content, already updated by
    # this pass's _sweep_x) currently extends to; (b) is exactly the term
    # the old min-shared-shift version omitted, letting non-group content
    # end up more extreme than the shifted group and silently break its
    # boundary touch (see module docstring).
    bound_cap = max(_bound_x(pos, i) + pos[i][2] for i in group)
    non_group_edge = max((p[0] + p[2] for j, p in enumerate(pos) if j not in group_set),
                          default=floor)
    target = max(bound_cap, non_group_edge, floor)
    for i in group:
        new_x = target - pos[i][2]
        if new_x < pos[i][0] - _EPS:
            pos[i][0] = new_x


def _shift_group_y(pos: List[List[float]], group: List[int], floor: float) -> None:
    """Mirror of _shift_group_x for a top-pinned edge group."""
    if not group:
        return
    group_set = set(group)
    bound_cap = max(_bound_y(pos, i) + pos[i][3] for i in group)
    non_group_edge = max((p[1] + p[3] for j, p in enumerate(pos) if j not in group_set),
                          default=floor)
    target = max(bound_cap, non_group_edge, floor)
    for i in group:
        new_y = target - pos[i][3]
        if new_y < pos[i][1] - _EPS:
            pos[i][1] = new_y


def compact(
    positions: List[Position],
    constraints: Optional[torch.Tensor],
    passes: int = 3,
) -> List[Position]:
    """Slide movable blocks toward smaller x/y to close RL-left-behind gaps.
    `constraints` is the instance's [n, 5] (fixed, preplaced, mib_id,
    cluster_id, boundary_code) tensor, or None (nothing marked immovable
    beyond what geometry itself implies)."""
    n = len(positions)
    if n == 0:
        return positions

    ncols = constraints.shape[1] if constraints is not None and constraints.dim() > 1 else 0

    def col(j):
        if constraints is None or ncols <= j:
            return [0] * n
        return constraints[:, j].tolist()

    preplaced = col(1)
    cluster_id = col(3)
    boundary = col(4)

    immovable_x = [False] * n
    immovable_y = [False] * n
    right_group: List[int] = []
    top_group: List[int] = []
    cluster_groups: dict = {}
    for i in range(n):
        if preplaced[i]:
            immovable_x[i] = True
            immovable_y[i] = True
            continue
        if cluster_id[i]:
            # Immovable individually (the per-block sweep must never shift
            # one member without the rest) but the whole group still slides
            # together -- see _shift_cluster_x/_shift_cluster_y below.
            immovable_x[i] = True
            immovable_y[i] = True
            cluster_groups.setdefault(int(cluster_id[i]), []).append(i)
            continue
        code = int(boundary[i])
        if code & 0b0011:  # left(1) or right(2)
            immovable_x[i] = True
        if code & 0b1100:  # top(4) or bottom(8)
            immovable_y[i] = True
        if code & 2:  # right -- needs the rigid-group pull, see docstring
            right_group.append(i)
        if code & 4:  # top -- same, mirrored on y
            top_group.append(i)

    pos = [list(p) for p in positions]
    floor_x = min(p[0] for p in pos)
    floor_y = min(p[1] for p in pos)
    groups = list(cluster_groups.values())
    for _ in range(passes):
        for group in groups:
            _shift_cluster_x(pos, group, floor_x)
        _sweep_x(pos, immovable_x, floor_x)
        _shift_group_x(pos, right_group, floor_x)
        for group in groups:
            _shift_cluster_y(pos, group, floor_y)
        _sweep_y(pos, immovable_y, floor_y)
        _shift_group_y(pos, top_group, floor_y)

    return [tuple(p) for p in pos]

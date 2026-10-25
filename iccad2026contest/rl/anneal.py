"""
Post-RL simulated-annealing polish, run on finetune_and_solve()'s already-
feasible, already-compacted output before it's returned to the contest.

Why this is a separate pass rather than more PPO training: the RL policy
places blocks sequentially, once, in a fixed greedy order (rl/ordering.py) --
it never revisits an earlier choice in light of later ones. SA fixes that by
locally perturbing the *finished* layout (translate/swap/rotate a block) and
keeping whatever perturbation helps, using the exact same baseline-free cost
rl/reward.py's inference_reward already uses at contest time (no ground
truth is available on the hidden test set, so this cannot and does not use
hpwl_gap/area_gap against a baseline -- see that module's docstring).

Hard vs. soft constraints, and why moves don't need to special-case most of
them: overlap-free-ness, per-block area tolerance, and fixed/preplaced exact
dimensions are HARD (any violation makes evaluate_solution's is_feasible
False); boundary/grouping/MIB are SOFT (already priced into inference_reward
via violation_factor = exp(BETA * violations_relative)). So moves only need
to *guarantee* the hard set structurally (never touch a preplaced block;
never resize a fixed-shape block; never resize a block sharing an MIB group,
since that would desync the group's required uniform shape) and can leave
boundary/cluster blocks free to move -- the annealer will naturally avoid
making their violations worse because that's already reflected in the cost
it's minimizing, the same way PPO training already relies on this term.

Every candidate is checked overlap-free (iccad2026_evaluate.check_overlap)
*before* compaction ever sees it, because rl/compaction.py's own safety
argument assumes its input is already overlap-free -- an overlapping
candidate is always rejected outright, never merely down-weighted by SA's
probabilistic acceptance, so the annealer can never hand back something
worse than what finetune_and_solve already guaranteed.

Group-shift move (a whole cluster translated together): every other move
here (relocate/swap/rotate) touches one block, or a pair, at a time. That
is structurally unable to discover "translate this whole isolated cluster
by a constant offset" even when it is obviously the best available move --
moving a single member out of a tightly-arranged cluster almost always
increases grouping_violations for a wirelength gain too small to survive
the Metropolis accept test, so the group can never migrate piece by piece
(see algorithm.md's test-98 compaction diagnosis: whole clusters can end
up stranded far from the rest of the layout with nothing able to pull them
in). The group-shift move proposes translating every member of a randomly
chosen cluster by the same delta -- flush against a randomly chosen other
block's edge, mirroring the single-block relocate's own flush-move -- and
evaluates/accepts it as one atomic candidate, so it can jump a whole
cluster across a gap in one step instead of needing an impossible sequence
of individually-losing single-block moves.

Boundary-squeeze move: a boundary-pinned block (e.g. a corner pin, both
x and y fixed) has zero position freedom of its own -- it's mechanically
locked to wherever the shared group's required edge currently is (see
rl/compaction.py's _shift_group_x/_shift_group_y, which computes this
edge from live positions, not the original padded canvas -- diagnosed
2026-09-21 that this is already correct, not a stale-reference-frame
bug). What actually caps how tight that shared edge can get is ordinary
packing density: the group's tightest member is blocked by some other
block, which is blocked by another, and so on -- a real chain, not an
artifact. The ordinary relocate move picks its target block uniformly at
random, so it essentially never happens to propose exactly "move this
specific blocker out of the way" out of ~100+ candidates. This move
instead identifies the current bottleneck directly
(_boundary_group_bottleneck: which boundary-group member has the
tightest edge, and what specifically blocks it from being any tighter)
and proposes relocating that blocker -- the same relocate/jitter
mechanic as the ordinary move, just aimed at the block that's actually
capping the boundary group instead of a random one.
"""

import math
import random
import time
from typing import List, Optional, Tuple

from iccad2026_evaluate import check_overlap

from .compaction import compact
from .reward import inference_reward

Position = Tuple[float, float, float, float]


def _boundary_group_bottleneck(candidate: List[List[float]], group: List[int], axis: str):
    """For a boundary group sharing one required edge (all `right_group`
    members must share the same x+w; all `top_group` members the same
    y+h -- see rl/compaction.py's _shift_group_x/_shift_group_y, which
    computes this same shared-edge requirement for compaction), finds
    the member currently defining the tightest (largest) shared edge
    (the "bottleneck"), then the block immediately blocking IT from
    being any tighter (closest already-placed content on the correct
    side, with overlapping extent on the other axis).

    Returns (bottleneck_idx, blocker_idx) or (bottleneck_idx, None) if
    nothing blocks the bottleneck (already as tight as it can be, e.g.
    sitting at 0). Diagnosed 2026-09-21 on test-98: compaction's own
    shared-edge computation is already based on live positions, not the
    original padded canvas, so the group converges to a real fixed point
    -- but that fixed point is capped by whichever member has the least
    room, which is itself capped by ordinary packing density (a chain of
    blockers), not a stale reference frame. This move directly targets
    that chain instead of hoping a uniformly-random relocate stumbles
    onto it."""
    if axis == 'x':
        edge = lambda i: candidate[i][0] + candidate[i][2]
    else:
        edge = lambda i: candidate[i][1] + candidate[i][3]
    bottleneck = max(group, key=edge)
    bx, by, bw, bh = candidate[bottleneck]
    blockers = []
    for j in range(len(candidate)):
        if j == bottleneck:
            continue
        xj, yj, wj, hj = candidate[j]
        if axis == 'x':
            overlap = min(by + bh, yj + hj) - max(by, yj)
            if overlap > 1e-6 and xj + wj <= bx + 1e-6:
                blockers.append((xj + wj, j))
        else:
            overlap = min(bx + bw, xj + wj) - max(bx, xj)
            if overlap > 1e-6 and yj + hj <= by + 1e-6:
                blockers.append((yj + hj, j))
    if not blockers:
        return bottleneck, None
    blockers.sort(reverse=True)
    return bottleneck, blockers[0][1]


def anneal_polish(
    instance,
    positions: List[Position],
    time_budget: float = 8.0,
    max_iterations: int = 3000,
    initial_temp: float = 1.0,
    cooling_rate: float = 0.997,
    seed: Optional[int] = None,
    verbose: bool = False,
) -> List[Position]:
    n = instance.block_count
    if n < 2:
        return positions
    if seed is not None:
        random.seed(seed)

    constraints = instance.constraints
    ncols = constraints.shape[1] if constraints is not None and constraints.dim() > 1 else 0

    def col(j):
        if constraints is None or ncols <= j:
            return [0] * n
        return constraints[:, j].tolist()

    fixed, preplaced, mib_id, cluster_id = col(0), col(1), col(2), col(3)
    movable = [i for i in range(n) if not preplaced[i]]
    rotatable = [i for i in movable if not fixed[i] and not mib_id[i]]
    if len(movable) < 2:
        return positions

    cluster_groups_map: dict = {}
    for i in movable:
        cid = cluster_id[i]
        if cid:
            cluster_groups_map.setdefault(int(cid), []).append(i)
    cluster_groups = list(cluster_groups_map.values())
    cluster_members = {i for group in cluster_groups for i in group}
    GROUP_SHIFT_PROB = 0.15

    boundary_col = col(4)
    right_group = [i for i in movable if int(boundary_col[i]) & 2]
    top_group = [i for i in movable if int(boundary_col[i]) & 4]
    movable_set = set(movable)
    BOUNDARY_SQUEEZE_PROB = 0.15

    current = [list(p) for p in positions]
    current_cost = -inference_reward(instance, [tuple(p) for p in current])
    best = [row[:] for row in current]
    best_cost = current_cost

    COMPACT_EVERY = 25  # periodic deterministic move, see below

    temp = initial_temp
    start = time.time()
    it = 0
    while it < max_iterations and (time.time() - start) < time_budget:
        it += 1

        # Gravity-compact the whole layout every COMPACT_EVERY iterations,
        # routed through the same accept/reject test as a random move: a
        # single block's relocate/swap can free up slack that compact()
        # then immediately claims, which neither move discovers alone (see
        # algorithm.md's diagnosis -- compact() run once at the end can get
        # stuck at a fixed point far short of the working canvas' padding).
        if it % COMPACT_EVERY == 0:
            candidate = [list(p) for p in compact([tuple(p) for p in current], constraints)]
            cand_tuples = [tuple(p) for p in candidate]
            cand_cost = -inference_reward(instance, cand_tuples)
            delta = cand_cost - current_cost
            if delta <= 0 or random.random() < math.exp(-delta / max(temp, 1e-6)):
                current = candidate
                current_cost = cand_cost
                if current_cost < best_cost:
                    best_cost = current_cost
                    best = [row[:] for row in current]
            temp *= cooling_rate
            if verbose and it % 200 == 0:
                print(f"  anneal iter {it}: temp={temp:.4f} "
                      f"current_cost={current_cost:.4f} best_cost={best_cost:.4f}")
            continue

        candidate = [row[:] for row in current]
        did_group_shift = False
        if cluster_groups and random.random() < GROUP_SHIFT_PROB:
            group = random.choice(cluster_groups)
            group_set = set(group)
            others = [x for x in range(n) if x not in group_set]
            if others:
                j = random.choice(others)
                xj, yj, wj, hj = candidate[j]
                gx0 = min(candidate[m][0] for m in group)
                gy0 = min(candidate[m][1] for m in group)
                gx1 = max(candidate[m][0] + candidate[m][2] for m in group)
                gy1 = max(candidate[m][1] + candidate[m][3] for m in group)
                dx, dy = random.choice([
                    (xj + wj - gx0, 0.0), (xj - gx1, 0.0),
                    (0.0, yj + hj - gy0), (0.0, yj - gy1),
                ])
                for m in group:
                    candidate[m][0] = max(0.0, candidate[m][0] + dx)
                    candidate[m][1] = max(0.0, candidate[m][1] + dy)
                did_group_shift = True

        did_boundary_squeeze = False
        if not did_group_shift and (right_group or top_group) and random.random() < BOUNDARY_SQUEEZE_PROB:
            axes = []
            if right_group:
                axes.append(('x', right_group))
            if top_group:
                axes.append(('y', top_group))
            axis, group = random.choice(axes)
            _bottleneck, blocker = _boundary_group_bottleneck(candidate, group, axis)
            if blocker is not None and blocker in movable_set and blocker not in cluster_members:
                xj, yj, wj, hj = candidate[blocker]
                blocker_code = int(boundary_col[blocker])
                blocker_has_x = bool(blocker_code & 0b0011)  # left or right
                blocker_has_y = bool(blocker_code & 0b1100)  # top or bottom
                # If the blocker has its OWN boundary requirement on the SAME
                # axis we're squeezing, a free move here would fight its own
                # constraint (e.g. relocating a right-pinned blocker off the
                # right edge to unblock another right-pinned block) -- SA
                # would need to accept a proposal that breaks one boundary
                # compliance to fix another, which the cost function makes
                # rare (diagnosed 2026-09-21: this is exactly why the
                # y-direction never moved on test-98 -- block 50's blocker,
                # 102, is itself right-pinned, and generic relocate/jitter
                # almost always broke that). When the blocker's OWN
                # constraint is on the OTHER (perpendicular) axis, move it
                # along ONLY the axis we're squeezing, holding its own
                # constrained coordinate fixed -- this can never conflict
                # with its own requirement, so it survives the Metropolis
                # test far more often. Only fall through to a free 2D move
                # when the blocker has no conflicting same-axis constraint
                # of its own to preserve.
                constrained_same_axis = (axis == 'x' and blocker_has_x) or (axis == 'y' and blocker_has_y)
                if not constrained_same_axis and (blocker_has_x or blocker_has_y):
                    if axis == 'x':
                        if random.random() < 0.5:
                            k = random.randrange(n)
                            if k != blocker:
                                xk, _, wk, _ = candidate[k]
                                nx = random.choice([xk + wk, xk - wj])
                                candidate[blocker][0] = max(0.0, nx)
                                did_boundary_squeeze = True
                        else:
                            candidate[blocker][0] = max(0.0, xj + random.gauss(0, max(wj, 1.0)))
                            did_boundary_squeeze = True
                    else:
                        if random.random() < 0.5:
                            k = random.randrange(n)
                            if k != blocker:
                                _, yk, _, hk = candidate[k]
                                ny = random.choice([yk + hk, yk - hj])
                                candidate[blocker][1] = max(0.0, ny)
                                did_boundary_squeeze = True
                        else:
                            candidate[blocker][1] = max(0.0, yj + random.gauss(0, max(hj, 1.0)))
                            did_boundary_squeeze = True
                else:
                    # No conflicting constraint of its own -- free to move in
                    # either dimension, same mechanic as the ordinary
                    # relocate move, just aimed at the actual bottleneck's
                    # blocker instead of a uniformly random block.
                    if random.random() < 0.5:
                        others = [k for k in range(n) if k not in (blocker, _bottleneck)]
                        if others:
                            k = random.choice(others)
                            xk, yk, wk, hk = candidate[k]
                            nx, ny = random.choice([
                                (xk + wk, yk), (xk - wj, yk),
                                (xk, yk + hk), (xk, yk - hj),
                            ])
                            candidate[blocker][0] = max(0.0, nx)
                            candidate[blocker][1] = max(0.0, ny)
                            did_boundary_squeeze = True
                    else:
                        scale = max(wj, hj, 1.0)
                        candidate[blocker][0] = max(0.0, xj + random.gauss(0, scale))
                        candidate[blocker][1] = max(0.0, yj + random.gauss(0, scale))
                        did_boundary_squeeze = True

        if not did_group_shift and not did_boundary_squeeze:
            roll = random.random()
            if roll < 0.55:
                i = random.choice(movable)
                xi, yi, wi, hi = candidate[i]
                if random.random() < 0.5:
                    # Relocate flush against a random block's edge -- helps
                    # connect otherwise-isolated pieces of the layout.
                    j = random.randrange(n)
                    xj, yj, wj, hj = candidate[j]
                    nx, ny = random.choice([
                        (xj + wj, yj), (xj - wi, yj),
                        (xj, yj + hj), (xj, yj - hi),
                    ])
                else:
                    scale = max(wi, hi, 1.0)
                    nx = xi + random.gauss(0, scale)
                    ny = yi + random.gauss(0, scale)
                candidate[i][0] = max(0.0, nx)
                candidate[i][1] = max(0.0, ny)
            elif roll < 0.85 and len(movable) >= 2:
                i, k = random.sample(movable, 2)
                candidate[i][0], candidate[k][0] = candidate[k][0], candidate[i][0]
                candidate[i][1], candidate[k][1] = candidate[k][1], candidate[i][1]
            else:
                if not rotatable:
                    continue
                i = random.choice(rotatable)
                candidate[i][2], candidate[i][3] = candidate[i][3], candidate[i][2]

        cand_tuples = [tuple(p) for p in candidate]
        if check_overlap(cand_tuples) > 0:
            continue

        cand_cost = -inference_reward(instance, cand_tuples)
        delta = cand_cost - current_cost
        if delta <= 0 or random.random() < math.exp(-delta / max(temp, 1e-6)):
            current = candidate
            current_cost = cand_cost
            if current_cost < best_cost:
                best_cost = current_cost
                best = [row[:] for row in current]
        temp *= cooling_rate

        if verbose and it % 200 == 0:
            print(f"  anneal iter {it}: temp={temp:.4f} "
                  f"current_cost={current_cost:.4f} best_cost={best_cost:.4f}")

    return compact([tuple(p) for p in best], instance.constraints)

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
"""

import math
import random
import time
from typing import List, Optional, Tuple

from iccad2026_evaluate import check_overlap

from .compaction import compact
from .reward import inference_reward

Position = Tuple[float, float, float, float]


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

    fixed, preplaced, mib_id = col(0), col(1), col(2)
    movable = [i for i in range(n) if not preplaced[i]]
    rotatable = [i for i in movable if not fixed[i] and not mib_id[i]]
    if len(movable) < 2:
        return positions

    current = [list(p) for p in positions]
    current_cost = -inference_reward(instance, [tuple(p) for p in current])
    best = [row[:] for row in current]
    best_cost = current_cost

    temp = initial_temp
    start = time.time()
    it = 0
    while it < max_iterations and (time.time() - start) < time_budget:
        it += 1
        candidate = [row[:] for row in current]
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

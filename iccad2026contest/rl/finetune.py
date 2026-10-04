"""
Per-instance PPO fine-tuning, used inside my_optimizer.MyOptimizer.solve().

At contest time there is no ground-truth baseline for the hidden test set,
so this trains (optionally warm-started from a rl/train.py checkpoint)
directly against rl.reward.inference_reward for a bounded number of
iterations / wall-clock budget, then returns the best feasible placement any
rollout found.

Two variance-reduction measures on top of plain PPO (each solve() call is
otherwise its own ~20s of independent stochastic search, which turned out to
swamp small pretraining effects entirely -- see the seed-sweep discussion
this module's changes were motivated by):
    - Sampling temperature is annealed from 1.0 down to MIN_FINETUNE_TEMPERATURE
      across the iteration budget, so later rollouts (and thus more of what
      `best_positions` is chosen from) come from an increasingly peaked,
      lower-variance policy rather than staying maximally stochastic
      throughout.
    - After the PPO loop, one fully deterministic (argmax) rollout is run
      and considered as a final candidate -- a reproducible anchor that
      doesn't depend on which random samples happened to come up.

A tiny always-succeeding greedy fallback (square aspect ratio, first free
grid cell, no learning) guarantees solve() never returns nothing even in the
pathological case where every single PPO rollout happened to fail -- see
rl/env.py's real-data stress test, which found this essentially never
happens in practice.
"""

import dataclasses
import os
import math
import time
from typing import Dict, List, Optional, Sequence, Tuple

import torch

from .compaction import compact
from .env import ASPECT_RATIOS, CANVAS_PADDING, UTILIZATION_TARGET, GridPlacementEnv
from .networks import ActorCritic
from .ppo import collect_batch, collect_episode, ppo_update
from .reward import _evaluate

MIN_FINETUNE_TEMPERATURE = 0.3
# Solve-time canvas ratchet (see finetune_and_solve). Off unless FLOORSET_RATCHET
# is set (e.g. 0.05) until the 100-case benchmark shows it helps.
RATCHET_STEP = float(os.environ.get("FLOORSET_RATCHET", "0"))
RATCHET_MAX = 0.90


def choose_grid_dim(block_count: int) -> int:
    return int(min(96, max(32, math.sqrt(block_count) * 8)))


def greedy_fallback_positions(
    instance, grid_dim: int,
) -> List[Tuple[float, float, float, float]]:
    """Always-succeeding placement, used only when every PPO rollout this
    solve() call attempted failed to complete (rl/ppo.py's collect_episode
    already degrades a single failed rollout to a flat -M_PENALTY reward
    without crashing, so reaching here means the working canvas was too
    tight for literally every attempt this budget allowed, not bad luck on
    one draw). The first attempt uses GridPlacementEnv's own default sizing
    (pins-based, see rl/env.py's _estimate_canvas_size); each retry after a
    "canvas too small" error doubles an explicit (width, height) padding
    pair instead, starting from CANVAS_PADDING -- since a wide enough
    canvas always has room for any finite set of blocks, this is
    guaranteed to terminate, so solve() can never crash here regardless of
    how tight the instance's own pins turn out to be."""
    canvas_padding: Optional[Tuple[float, float]] = None
    last_error: Optional[RuntimeError] = None
    for _ in range(10):
        try:
            env = GridPlacementEnv(instance, grid_dim=grid_dim, canvas_padding=canvas_padding)
            square_idx = len(ASPECT_RATIOS) // 2
            while not env.done():
                if env.needs_aspect():
                    w, h = env.choose_aspect(square_idx)
                else:
                    w, h = env.current_shape()
                result = env.position_mask(w, h)
                if result is None:
                    continue
                mask, _, _ = result
                valid = mask.nonzero(as_tuple=False)
                gy, gx = int(valid[0, 0]), int(valid[0, 1])
                env.place(gy, gx)
            return compact(env.finalize(), instance.constraints)
        except RuntimeError as e:
            last_error = e
            if canvas_padding is None:
                canvas_padding = (CANVAS_PADDING, CANVAS_PADDING)
            else:
                canvas_padding = (canvas_padding[0] * 2.0, canvas_padding[1] * 2.0)
    raise RuntimeError(f"greedy fallback still failed after widening the canvas 10x: {last_error}")


def compute_reference_metrics(instance, grid_dim: int) -> Optional[Dict[str, float]]:
    """HPWL/bbox area of greedy_fallback_positions' layout -- deterministic
    and policy-independent, so inference_reward stays a fixed target across
    the whole fine-tune (see rl/reward.py's module docstring). None if it
    fails or the layout is infeasible (inference_reward then stays absolute)."""
    try:
        metrics = _evaluate(instance, greedy_fallback_positions(instance, grid_dim))
    except RuntimeError:
        return None
    if not metrics.is_feasible or metrics.bbox_area <= 0:
        return None
    return {'hpwl_ref': metrics.hpwl_total, 'area_ref': metrics.bbox_area}


def finetune_and_solve(
    instance,
    net: Optional[ActorCritic] = None,
    grid_dim: Optional[int] = None,
    time_budget: float = 20.0,
    max_iterations: int = 150,
    episodes_per_iter: int = 6,
    ppo_epochs: int = 3,
    seed: Optional[int] = None,
    verbose: bool = False,
    utilizations: Optional[Sequence[float]] = None,
    ratchet_step: float = RATCHET_STEP,
    ratchet_max: float = RATCHET_MAX,
) -> Tuple[List[Tuple[float, float, float, float]], ActorCritic]:
    """`utilizations`: canvas utilization targets to try, tightest first
    (see rl/env.py's UTILIZATION_TARGET). Fine-tuning starts on the first
    and moves to the next, looser one whenever a whole batch fails to
    finish before any feasible layout exists. None = env default only.

    Ratchet (09-27): layouts fill the canvas they're given (median final
    bbox = 0.96 x canvas), so white space is set by canvas size. After every
    batch with a feasible rollout the canvas tightens by `ratchet_step`
    (up to `ratchet_max`); the first fully failed batch after that steps
    back to the last level that worked and stops tightening. The best
    feasible layout (by normalized reward) is kept throughout.
    ratchet_step=0 disables it."""
    if seed is not None:
        torch.manual_seed(seed)
    start = time.time()   # the budget covers everything below, not just the PPO loop
    if grid_dim is None:
        grid_dim = choose_grid_dim(instance.block_count)
    if net is None:
        net = ActorCritic()
    if instance.reference_metrics is None:
        # A copy, so the caller's instance (e.g. my_optimizer.py's, reused
        # by anneal_polish) keeps whatever reward scale it already had.
        instance = dataclasses.replace(
            instance, reference_metrics=compute_reference_metrics(instance, grid_dim))
        if verbose:
            print(f"  reward reference: {instance.reference_metrics}")
    optimizer = torch.optim.Adam(net.parameters(), lr=1e-3)
    ladder = list(utilizations) if utilizations else [UTILIZATION_TARGET]
    level = 0
    cur_u, last_ok_u, ratchet_done = ladder[0], None, ratchet_step <= 0

    def env_kwargs():
        return {'utilization': cur_u}

    best_positions = None
    best_reward = float('-inf')
    iteration = 0
    deadline = start + time_budget
    t_collect = t_update = None   # measured durations, to keep the budget a hard limit

    while iteration < max_iterations:
        # Leave room for the final greedy rollout (about one episode).
        reserve = t_collect / episodes_per_iter if t_collect else 0.0
        if time.time() + reserve >= deadline:
            break
        # Linear anneal: 1.0 at iteration 0 down to MIN_FINETUNE_TEMPERATURE
        # by the last budgeted iteration (see module docstring).
        temperature = 1.0 - (1.0 - MIN_FINETUNE_TEMPERATURE) * (iteration / max(max_iterations - 1, 1))
        t0 = time.time()
        episodes = collect_batch(net, instance, grid_dim=grid_dim, use_baseline=False,
                                  num_episodes=episodes_per_iter, temperature=temperature,
                                  deadline=deadline - reserve, env_kwargs=env_kwargs())
        t_collect = time.time() - t0
        if not episodes:
            break  # budget ran out before a single episode finished this iteration
        for ep in episodes:
            if ep.positions is not None and ep.reward > best_reward:
                best_reward = ep.reward
                best_positions = ep.positions
        # An update only pays off if another batch can sample from it; on big
        # designs one update takes ~2x a batch (design 98: 12 s), so skipping
        # the last one keeps the budget instead of overrunning it.
        est_update = t_update if t_update is not None else 2.0 * t_collect
        if time.time() + est_update + t_collect < deadline:
            t0 = time.time()
            ppo_update(net, optimizer, episodes, epochs=ppo_epochs, normalize_advantages=True)
            t_update = time.time() - t0
        iteration += 1
        if any(ep.positions is not None for ep in episodes):
            last_ok_u = cur_u
            if not ratchet_done and cur_u + ratchet_step <= ratchet_max + 1e-9:
                cur_u = round(cur_u + ratchet_step, 4)
                if verbose:
                    print(f"  feasible at {last_ok_u}; tightening canvas to utilization {cur_u}")
        elif last_ok_u is not None:
            if not ratchet_done and verbose:
                print(f"  nothing finished at {cur_u}; back to {last_ok_u}, ratchet stops")
            cur_u, ratchet_done = last_ok_u, True
        elif level + 1 < len(ladder):
            level += 1
            cur_u = ladder[level]
            if verbose:
                print(f"  no rollout finished; loosening canvas to utilization {cur_u}")
        if verbose:
            avg_reward = sum(ep.reward for ep in episodes) / len(episodes)
            print(f"  finetune iter {iteration}: temperature={temperature:.2f} "
                  f"avg_reward={avg_reward:.4f} best_reward={best_reward:.4f} "
                  f"elapsed={time.time()-start:.1f}s")

    # Deterministic final candidate: same converged policy, no sampling
    # randomness, so it doesn't depend on which random draws happened to
    # come up in the last iteration's batch.
    # Final greedy rollout at the tightest level that worked (if any).
    if last_ok_u is not None:
        cur_u = last_ok_u
    greedy_ep = collect_episode(net, instance, grid_dim=grid_dim, use_baseline=False, greedy=True,
                                env_kwargs=env_kwargs())
    if greedy_ep.positions is not None and greedy_ep.reward > best_reward:
        best_reward = greedy_ep.reward
        best_positions = greedy_ep.positions
        if verbose:
            print(f"  final greedy rollout: reward={greedy_ep.reward:.4f} (new best)")
    elif verbose and greedy_ep.positions is not None:
        print(f"  final greedy rollout: reward={greedy_ep.reward:.4f} (not better than best)")

    if best_positions is None:
        if verbose:
            print("  every PPO rollout failed; using greedy fallback placement")
        best_positions = greedy_fallback_positions(instance, grid_dim)
    else:
        # Deterministic post-process only -- never fed back into PPO's
        # reward (see rl/compaction.py), so this cannot affect what the
        # policy learns, only what's actually returned as the answer.
        best_positions = compact(best_positions, instance.constraints)

    return best_positions, net

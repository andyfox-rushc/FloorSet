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

import math
import time
from typing import List, Optional, Tuple

import torch

from .env import ASPECT_RATIOS, GridPlacementEnv
from .networks import ActorCritic
from .ppo import collect_batch, collect_episode, ppo_update

MIN_FINETUNE_TEMPERATURE = 0.3


def choose_grid_dim(block_count: int) -> int:
    return int(min(96, max(32, math.sqrt(block_count) * 8)))


def greedy_fallback_positions(instance, grid_dim: int) -> List[Tuple[float, float, float, float]]:
    env = GridPlacementEnv(instance, grid_dim=grid_dim)
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
    return env.finalize()


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
) -> Tuple[List[Tuple[float, float, float, float]], ActorCritic]:
    if seed is not None:
        torch.manual_seed(seed)
    if grid_dim is None:
        grid_dim = choose_grid_dim(instance.block_count)
    if net is None:
        net = ActorCritic()
    optimizer = torch.optim.Adam(net.parameters(), lr=1e-3)

    best_positions = None
    best_reward = float('-inf')
    start = time.time()
    iteration = 0

    while iteration < max_iterations and (time.time() - start) < time_budget:
        # Linear anneal: 1.0 at iteration 0 down to MIN_FINETUNE_TEMPERATURE
        # by the last budgeted iteration (see module docstring).
        temperature = 1.0 - (1.0 - MIN_FINETUNE_TEMPERATURE) * (iteration / max(max_iterations - 1, 1))
        episodes = collect_batch(net, instance, grid_dim=grid_dim, use_baseline=False,
                                  num_episodes=episodes_per_iter, temperature=temperature)
        for ep in episodes:
            if ep.positions is not None and ep.reward > best_reward:
                best_reward = ep.reward
                best_positions = ep.positions
        ppo_update(net, optimizer, episodes, epochs=ppo_epochs)
        iteration += 1
        if verbose:
            avg_reward = sum(ep.reward for ep in episodes) / len(episodes)
            print(f"  finetune iter {iteration}: temperature={temperature:.2f} "
                  f"avg_reward={avg_reward:.4f} best_reward={best_reward:.4f} "
                  f"elapsed={time.time()-start:.1f}s")

    # Deterministic final candidate: same converged policy, no sampling
    # randomness, so it doesn't depend on which random draws happened to
    # come up in the last iteration's batch.
    greedy_ep = collect_episode(net, instance, grid_dim=grid_dim, use_baseline=False, greedy=True)
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

    return best_positions, net

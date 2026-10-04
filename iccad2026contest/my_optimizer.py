#!/usr/bin/env python3
"""
ICCAD 2026 FloorSet Challenge - AlphaChip-style RL Optimizer

An RL floorplanner modeled on Google's AlphaChip (Mirhoseini/Goldie): a GNN
encoder over the block/pin connectivity graph feeds a sequential
autoregressive placement policy (aspect-ratio choice + grid position per
block), trained with PPO. See rl/ for the full pipeline:

  rl/ordering.py, rl/env.py  - deterministic placement order + grid env that
                               GUARANTEES the contest's hard constraints by
                               construction (overlap-free, exact area for
                               soft blocks, exact fixed/preplaced dimensions,
                               exact MIB shape sharing); boundary and
                               grouping are handled as documented in env.py.
  rl/encoder.py              - hand-rolled edge-weighted GNN (no
                               torch_geometric dependency).
  rl/networks.py             - policy/value/reward-approximation heads.
  rl/ppo.py                  - PPO rollout collection + clipped-surrogate
                               update, using rl/reward.py's exact contest
                               cost (when a baseline is available) or the
                               no-baseline proxy (at contest time).
  rl/finetune.py             - per-instance PPO fine-tuning used below,
                               optionally warm-started from a rl/train.py
                               checkpoint (checkpoints/policy.pt, if present).
  rl/train.py                - offline pretraining CLI (run separately;
                               see README.md for the training-set download).

USAGE:
  Test: python iccad2026_evaluate.py --evaluate my_optimizer.py

Your solve() receives:
  - block_count: int
  - area_targets: [n] target area per block
  - b2b_connectivity: [edges, 3] (block_i, block_j, weight)
  - p2b_connectivity: [edges, 3] (pin_idx, block_idx, weight)
  - pins_pos: [n_pins, 2] pin (x, y)
  - constraints: [n, 5] (fixed, preplaced, mib_id, cluster_id, boundary_code)
  - target_positions: [n, 4] target (x, y, w, h) per block.
      All -1 by default (free). For fixed-shape blocks, w and h are set.
      For preplaced blocks, all four (x, y, w, h) are set.

Your solve() must return:
  - List of (x, y, width, height), exactly block_count tuples
"""

import os
import sys
from pathlib import Path
from typing import List, Optional, Tuple

import torch

sys.path.insert(0, str(Path(__file__).parent))

from iccad2026_evaluate import FloorplanOptimizer
from rl.checkpoint_meta import utilization_ladder
from rl.anneal import anneal_polish
from rl.repack import repack_polish
from rl.slicing import slicing_polish
from rl.data import from_solve_args
from rl.finetune import finetune_and_solve
from rl.networks import ActorCritic, load_state

# FLOORSET_CHECKPOINT overrides, e.g. to evaluate a fixed history snapshot.
DEFAULT_CHECKPOINT = os.environ.get(
    "FLOORSET_CHECKPOINT", str(Path(__file__).parent / "checkpoints" / "policy.pt"))


class MyOptimizer(FloorplanOptimizer):
    """AlphaChip-style RL placer: per-instance PPO fine-tuning (optionally
    warm-started from a rl/train.py checkpoint), bounded by a time/iteration
    budget. See rl/finetune.py for the full loop and its always-succeeding
    fallback."""

    def __init__(
        self,
        verbose: bool = False,
        # 09-28 sweep, 100 designs: 20+8 s -> 1.607 at 28.9 s, 2+1 s -> 1.643
        # at 4.3 s; with the runtime factor (runtime/median)^0.3, 2+1 wins
        # for any median runtime >= 10 s (budget_sweep_0928*.json).
        time_budget: float = 2.0,
        max_iterations: int = 150,
        episodes_per_iter: int = 6,
        ppo_epochs: int = 3,
        anneal_time_budget: float = 1.0,
        polish: str = os.environ.get("FLOORSET_POLISH", "slice+repack"),
        checkpoint_path: Optional[str] = None,
    ):
        super().__init__(verbose)
        # Rollout ops are small; extra threads only add overhead, and on a
        # busy machine they oversubscribe cores (50x slower, measured 09-26).
        torch.set_num_threads(1)
        self.time_budget = time_budget
        self.max_iterations = max_iterations
        self.episodes_per_iter = episodes_per_iter
        self.ppo_epochs = ppo_epochs
        self.anneal_time_budget = anneal_time_budget
        self.polish = polish
        self.checkpoint_path = checkpoint_path or DEFAULT_CHECKPOINT

    def _load_checkpoint(self) -> Optional[ActorCritic]:
        if os.path.exists(self.checkpoint_path):
            net = ActorCritic()
            load_state(net, self.checkpoint_path)
            if self.verbose:
                print(f"  loaded checkpoint: {self.checkpoint_path}")
            return net
        return None

    def solve(
        self,
        block_count: int,
        area_targets: torch.Tensor,
        b2b_connectivity: torch.Tensor,
        p2b_connectivity: torch.Tensor,
        pins_pos: torch.Tensor,
        constraints: torch.Tensor,
        target_positions: Optional[torch.Tensor] = None,
    ) -> List[Tuple[float, float, float, float]]:
        instance = from_solve_args(
            block_count, area_targets, b2b_connectivity, p2b_connectivity,
            pins_pos, constraints, target_positions,
        )
        net = self._load_checkpoint()
        positions, _ = finetune_and_solve(
            instance,
            net=net,
            time_budget=self.time_budget,
            max_iterations=self.max_iterations,
            episodes_per_iter=self.episodes_per_iter,
            ppo_epochs=self.ppo_epochs,
            verbose=self.verbose,
            utilizations=utilization_ladder(self.checkpoint_path),
        )
        if self.anneal_time_budget > 0:
            # Re-packing polish (rl/repack.py) replaced anneal_polish 09-26:
            # 100 val cases, same RL layouts, 8s: cost 1.644 vs 1.690.
            if self.polish == "anneal":
                positions = anneal_polish(instance, positions,
                                          time_budget=self.anneal_time_budget, verbose=self.verbose)
            elif self.polish == "slice+repack":
                # Slicing re-pack (rl/slicing.py) first, 09-28: 100 cached
                # layouts, 1 s polish: cost 1.626 (repack) -> 1.552.
                half = self.anneal_time_budget / 2
                positions = slicing_polish(instance, positions, time_budget=half, verbose=self.verbose)
                positions = repack_polish(instance, positions, time_budget=half, verbose=self.verbose)
            else:
                positions = repack_polish(instance, positions,
                                          time_budget=self.anneal_time_budget, verbose=self.verbose)
        return positions

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
from rl.data import from_solve_args
from rl.finetune import finetune_and_solve
from rl.networks import ActorCritic

DEFAULT_CHECKPOINT = str(Path(__file__).parent / "checkpoints" / "policy.pt")


class MyOptimizer(FloorplanOptimizer):
    """AlphaChip-style RL placer: per-instance PPO fine-tuning (optionally
    warm-started from a rl/train.py checkpoint), bounded by a time/iteration
    budget. See rl/finetune.py for the full loop and its always-succeeding
    fallback."""

    def __init__(
        self,
        verbose: bool = False,
        time_budget: float = 20.0,
        max_iterations: int = 150,
        episodes_per_iter: int = 6,
        ppo_epochs: int = 3,
        checkpoint_path: Optional[str] = None,
    ):
        super().__init__(verbose)
        self.time_budget = time_budget
        self.max_iterations = max_iterations
        self.episodes_per_iter = episodes_per_iter
        self.ppo_epochs = ppo_epochs
        self.checkpoint_path = checkpoint_path or DEFAULT_CHECKPOINT

    def _load_checkpoint(self) -> Optional[ActorCritic]:
        if os.path.exists(self.checkpoint_path):
            net = ActorCritic()
            net.load_state_dict(torch.load(self.checkpoint_path, map_location="cpu"))
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
        )
        return positions

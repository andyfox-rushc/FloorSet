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
# 10-05: six candidates (placer utilizations 0.75-1.0; each wins top-1 on 12-21
# of 100 designs), top 4 packed: cached 1.208 vs 3 candidates / top 3 1.234.
# 10-08: + "learned" / "learned2" (rl/learned_placer.py via rl/analytic.py learned_candidate,
# checkpoints/placer_a.pt / placer_b.pt), always packed; one alone beats the full pipeline on
# 36/99 designs (HPWL gap 0.068 vs 0.110); official with one: 1.1232 -> 1.1027.
# 10-11: + "learned4" (second relation network) and packer budget 200: official 1.0575 -> 1.0465.
DEFAULT_CANDIDATES = ("base,scratch0.75:it600:po1.0,scratch0.85:it600:po1.0,scratch0.90:it600:po1.0,"
                      "scratch0.95:it600:po1.0,scratch1.0:it600:po1.0,learned,learned2,learned3,learned4")
# Pin-box penalties to escalate through when no packed layout fits (10-07, "" = off).
# Off by default: on the 8 designs still outside it fixed none and tripled their runtime.
ESCALATE = tuple(float(v) for v in os.environ.get("FLOORSET_PINBOX_ESCALATE", "").split(",") if v)
DEFAULT_CHECKPOINT = os.environ.get(
    "FLOORSET_CHECKPOINT", str(Path(__file__).parent / "checkpoints" / "policy.pt"))


# Worker processes for the best-of candidates and the packer runs (10-06):
# independent jobs, so wall time drops from their sum to about the slowest.
# FLOORSET_PROCS (default min(6, cpus)); 1, or no way to start processes
# (e.g. inside a daemonic pool worker), runs them in this process, and so
# does any pool failure (_pmap).
_POOL = None


def _worker_pool():
    global _POOL
    procs = int(os.environ.get("FLOORSET_PROCS", min(6, os.cpu_count() or 1)))
    if procs <= 1:
        return None
    if _POOL is None:
        try:
            import atexit
            import multiprocessing as mp
            from rl.packer import init_worker   # pool functions must live in rl/: the
            # contest evaluator imports this file under another module name
            _POOL = mp.get_context("forkserver").Pool(procs, initializer=init_worker)
            atexit.register(_POOL.terminate)
        except (AssertionError, OSError, ValueError):
            _POOL = False
    return _POOL or None


def _pmap(fn, jobs):
    global _POOL
    pool = _worker_pool()
    if pool is not None:
        try:
            return pool.map(fn, jobs)
        except Exception:
            _POOL = False                    # don't try the pool again
    return [fn(j) for j in jobs]


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
        # Best-of candidates (rl/analytic.py best_of), default since 10-04: the
        # policy layout + two analytic placements, 1 s polish each; 100 designs
        # 1.548 @ 4.5 s -> 1.431 @ 8.2 s. FLOORSET_CANDIDATES="" = polish only.
        self.candidates = [c for c in os.environ.get("FLOORSET_CANDIDATES", DEFAULT_CANDIDATES).split(",") if c]
        # Packer seconds per candidate at n = 77 (x (n/77)^2, rl/packer.py pack_budget), 0 = off.
        # 10-06, quality over runtime: 21 designs with 100+ blocks, weighted total
        # 1.5 -> 1.225 (11 s), 15 -> 1.170 (41 s), 40 + squeeze -> 1.147 (91 s);
        # 10-07, current settings: 40 -> 1.150, 100 -> 1.135 (215 s per big design).
        self.pack_time_budget = float(os.environ.get("FLOORSET_PACK", "200"))
        self.pack_topk = int(os.environ.get("FLOORSET_PACK_TOPK", "4"))
        # Packer runs (10-12 restarts): job j packs candidate j % topk with seed j,
        # so jobs > topk re-pack the best candidates with new seeds (0 = one per candidate).
        self.pack_jobs = int(os.environ.get("FLOORSET_PACK_JOBS", "0"))
        # Wirelength relaxation of the packed winner (10-06, rl/packer.py relax_hpwl).
        self.relax = os.environ.get("FLOORSET_RELAX", "1") != "0"
        # Slack-driven soft-block reshaping of the winner (10-06, rl/packer.py squeeze).
        self.squeeze = os.environ.get("FLOORSET_SQUEEZE", "1") != "0"

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
        if self.candidates:
            # Best-of analytic candidates (10-04, rl/analytic.py best_of): each
            # candidate has its own polish, so the polish below is skipped.
            from rl.analytic import best_of
            if self.pack_time_budget <= 0:
                return best_of(instance, positions, self.candidates, verbose=self.verbose)
            # Compact packer (10-05, rl/packer.py): sequence-pair search from each
            # of the top `pack_topk` candidates, best kept. 100 cached designs:
            # 1.361 -> winner only 1.278, top 3 x 1.5 s 1.248 (winner x 4.5 s 1.253).
            from rl.packer import _pack_job, pack_budget
            from rl.repack import contest_relative_cost
            ranked = best_of(instance, positions, self.candidates, verbose=self.verbose, return_all=True,
                             pmap=_pmap, force=tuple(c for c in self.candidates if c.startswith("learned")))
            budget = pack_budget(instance.block_count, self.pack_time_budget)
            top = ranked[:self.pack_topk]
            outs = _pmap(_pack_job, [(instance, top[j % len(top)], budget, j) for j in range(self.pack_jobs or len(top))])
            from rl.packer import ORIGIN, PINBOX_PEN, in_box, pin_box
            if ORIGIN:      # 10-07: layouts inside the box (origin, and pin box when on) first
                box = pin_box(instance) if PINBOX_PEN > 0 else None
                # Escalation (10-07): none fits the pin box -> repack the best ones
                # with stronger penalties until one does (tight designs only).
                rank = lambda ps: sorted(ps, key=lambda p: contest_relative_cost(instance, p, positions)[0])
                for pen in (ESCALATE if box is not None else ()):
                    if any(in_box(p, box) for p in outs):
                        break
                    outs += _pmap(_pack_job, [(instance, p, budget, 100 + k, pen)
                                              for k, p in enumerate(rank(outs)[:self.pack_topk])])
                origin_ok = [p for p in outs if in_box(p, None)]      # x, y >= 0 stays strict
                outs = origin_ok or outs
            costs = [contest_relative_cost(instance, p, positions)[0] for p in outs]
            if ORIGIN and box is not None:      # pin box: in-box within BOX_TOL of the best
                from rl.packer import prefer_in_box
                best = outs[prefer_in_box(costs, [in_box(p, box) for p in outs])]
            else:
                best = outs[min(range(len(outs)), key=lambda i: costs[i])]
            if self.squeeze:
                from rl.packer import squeeze
                best = squeeze(instance, best)
            if self.relax:
                from rl.packer import relax_hpwl
                best = relax_hpwl(instance, best)
            return best
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

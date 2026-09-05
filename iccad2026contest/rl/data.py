"""
Adapts contest data (validation dataset samples, training dataloader batches,
or hand-built synthetic fixtures) into a single plain FloorplanInstance
struct used throughout the rl/ package.

Baseline extraction is NOT reimplemented here: for validation samples we
reuse ContestEvaluator._extract_baseline (same logic the real evaluator
scores against); for training samples the baseline is already present in the
sample's own `metrics` tensor.
"""

from dataclasses import dataclass
from typing import Optional, Dict

import torch

from iccad2026_evaluate import ContestEvaluator


@dataclass
class FloorplanInstance:
    block_count: int
    area_targets: torch.Tensor        # [n] float
    b2b_connectivity: torch.Tensor    # [E_b2b, 3] (i, j, weight)
    p2b_connectivity: torch.Tensor    # [E_p2b, 3] (pin_idx, block_idx, weight)
    pins_pos: torch.Tensor            # [P, 2]
    constraints: torch.Tensor         # [n, 5] (fixed, preplaced, mib_id, cluster_id, boundary_code)
    target_positions: torch.Tensor    # [n, 4] (x, y, w, h), -1 where free
    baseline_metrics: Optional[Dict[str, float]] = None  # {'hpwl_baseline', 'area_baseline'} or None


def from_validation_sample(evaluator: ContestEvaluator, idx: int) -> FloorplanInstance:
    """Build a FloorplanInstance for validation case `idx`, reusing the
    real evaluator's own baseline-extraction logic."""
    evaluator._load_dataset()
    sample = evaluator.dataset[idx]
    inputs, labels = sample['input'], sample['label']
    area_target, b2b_conn, p2b_conn, pins_pos, constraints = inputs
    block_count = int((area_target != -1).sum().item())

    baseline, target_pos = evaluator._extract_baseline(
        idx, labels, b2b_conn, p2b_conn, pins_pos, block_count
    )

    target_positions = torch.full((block_count, 4), -1.0)
    nc = constraints.shape[1] if constraints.dim() > 1 else 0
    for i in range(block_count):
        is_fixed = nc > 0 and constraints[i, 0] != 0
        is_preplaced = nc > 1 and constraints[i, 1] != 0
        if is_preplaced:
            tx, ty, tw, th = target_pos[i]
            target_positions[i] = torch.tensor([tx, ty, tw, th])
        elif is_fixed:
            _, _, tw, th = target_pos[i]
            target_positions[i, 2] = tw
            target_positions[i, 3] = th

    return FloorplanInstance(
        block_count=block_count,
        area_targets=area_target[:block_count].clone(),
        b2b_connectivity=b2b_conn,
        p2b_connectivity=p2b_conn,
        pins_pos=pins_pos,
        constraints=constraints[:block_count].clone(),
        target_positions=target_positions,
        baseline_metrics=baseline,
    )


def from_training_batch_item(
    area_target: torch.Tensor,
    b2b_conn: torch.Tensor,
    p2b_conn: torch.Tensor,
    pins_pos: torch.Tensor,
    constraints: torch.Tensor,
    metrics: torch.Tensor,
) -> FloorplanInstance:
    """Build a FloorplanInstance from one (already batch-squeezed) training
    sample. `metrics` format: [area, num_pins, num_total_nets, num_b2b_nets,
    num_p2b_nets, num_hardconstraints, b2b_weighted_wl, p2b_weighted_wl]."""
    block_count = int((area_target != -1).sum().item())
    baseline = {
        'hpwl_baseline': float(metrics[6] + metrics[7]),
        'area_baseline': float(metrics[0]),
    }
    target_positions = torch.full((block_count, 4), -1.0)
    # Training data does not expose per-block target (x, y, w, h) the way the
    # validation loader does; fixed/preplaced dimension immutability is
    # enforced against target_positions during evaluation only, so for
    # training-time reward this is left free (-1). See rl/reward.py.
    return FloorplanInstance(
        block_count=block_count,
        area_targets=area_target[:block_count].clone(),
        b2b_connectivity=b2b_conn,
        p2b_connectivity=p2b_conn,
        pins_pos=pins_pos,
        constraints=constraints[:block_count].clone(),
        target_positions=target_positions,
        baseline_metrics=baseline,
    )


def from_solve_args(
    block_count: int,
    area_targets: torch.Tensor,
    b2b_connectivity: torch.Tensor,
    p2b_connectivity: torch.Tensor,
    pins_pos: torch.Tensor,
    constraints: torch.Tensor,
    target_positions: Optional[torch.Tensor] = None,
) -> FloorplanInstance:
    """Build a FloorplanInstance from FloorplanOptimizer.solve()'s raw
    arguments. No ground-truth baseline is available at contest time (the
    hidden test set doesn't expose it to the optimizer) -- see
    rl/reward.py's inference_reward, used for per-instance fine-tuning."""
    if target_positions is None:
        target_positions = torch.full((block_count, 4), -1.0)
    return FloorplanInstance(
        block_count=block_count,
        area_targets=area_targets[:block_count].clone(),
        b2b_connectivity=b2b_connectivity if b2b_connectivity is not None else torch.zeros(0, 3),
        p2b_connectivity=p2b_connectivity if p2b_connectivity is not None else torch.zeros(0, 3),
        pins_pos=pins_pos if pins_pos is not None else torch.zeros(0, 2),
        constraints=(constraints[:block_count].clone() if constraints is not None
                     else torch.zeros(block_count, 5)),
        target_positions=target_positions[:block_count].clone(),
        baseline_metrics=None,
    )


def synthetic_instance(
    area_targets,
    constraints=None,
    b2b_edges=None,
    p2b_edges=None,
    pins_pos=None,
    target_positions=None,
    baseline_metrics=None,
) -> FloorplanInstance:
    """Build a small hand-specified FloorplanInstance for unit tests.

    constraints: list of (fixed, preplaced, mib_id, cluster_id, boundary_code) or None (all zero)
    b2b_edges / p2b_edges: list of (i, j, weight) or None (empty)
    target_positions: list of (x, y, w, h), -1 for free fields, or None (all -1)
    """
    n = len(area_targets)
    area_targets_t = torch.tensor(area_targets, dtype=torch.float32)

    if constraints is None:
        constraints_t = torch.zeros(n, 5)
    else:
        constraints_t = torch.tensor(constraints, dtype=torch.float32)

    def edges_tensor(edges):
        if not edges:
            return torch.zeros(0, 3)
        return torch.tensor(edges, dtype=torch.float32)

    if target_positions is None:
        target_positions_t = torch.full((n, 4), -1.0)
    else:
        target_positions_t = torch.tensor(target_positions, dtype=torch.float32)

    pins_pos_t = torch.zeros(0, 2) if pins_pos is None else torch.tensor(pins_pos, dtype=torch.float32)

    return FloorplanInstance(
        block_count=n,
        area_targets=area_targets_t,
        b2b_connectivity=edges_tensor(b2b_edges),
        p2b_connectivity=edges_tensor(p2b_edges),
        pins_pos=pins_pos_t,
        constraints=constraints_t,
        target_positions=target_positions_t,
        baseline_metrics=baseline_metrics,
    )

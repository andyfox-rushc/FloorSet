"""
Tests for rl/data.py's dataset adapters. Notably from_training_batch_item:
there was previously no coverage of it at all, which is exactly how a real
bug (fixed-shape blocks silently getting w=h=-1 instead of their real
ground-truth dimensions, because fp_sol was being discarded) went unnoticed
through an entire overnight pretraining run against the real training set.
"""

from pathlib import Path

import pytest
import torch

from rl.data import from_training_batch_item
from rl.env import GridPlacementEnv

TRAINING_SET_PRESENT = (Path(__file__).parent.parent.parent / "floorset_lite").exists()


def make_fake_batch_item():
    # 3 blocks: 0 = free, 1 = fixed-shape, 2 = preplaced.
    area_target = torch.tensor([10.0, 12.0, 20.0])
    constraints = torch.tensor([
        [0, 0, 0, 0, 0],
        [1, 0, 0, 0, 0],
        [0, 1, 0, 0, 0],
    ], dtype=torch.float32)
    # fp_sol columns are (w, h, x, y) -- confirmed against real data.
    fp_sol = torch.tensor([
        [3.0, 3.333, 1.0, 2.0],   # free block's ground truth (unused as a target)
        [4.0, 3.0, 5.0, 6.0],     # fixed-shape: w=4, h=3
        [4.0, 5.0, 7.0, 8.0],     # preplaced: w=4, h=5, x=7, y=8
    ])
    metrics = torch.tensor([100.0, 5.0, 4.0, 3.0, 1.0, 2.0, 1.5, 0.5])
    b2b_conn = torch.zeros(0, 3)
    p2b_conn = torch.zeros(0, 3)
    pins_pos = torch.zeros(0, 2)
    return area_target, b2b_conn, p2b_conn, pins_pos, constraints, metrics, fp_sol


def test_fixed_shape_block_gets_real_dimensions_not_minus_one():
    area_target, b2b_conn, p2b_conn, pins_pos, constraints, metrics, fp_sol = make_fake_batch_item()
    inst = from_training_batch_item(area_target, b2b_conn, p2b_conn, pins_pos, constraints, metrics, fp_sol)

    tp = inst.target_positions
    assert tp[1, 2].item() == pytest.approx(4.0)
    assert tp[1, 3].item() == pytest.approx(3.0)
    # position is left free for fixed-shape (only shape is pinned)
    assert tp[1, 0].item() == -1.0
    assert tp[1, 1].item() == -1.0


def test_preplaced_block_gets_real_position_and_dimensions():
    area_target, b2b_conn, p2b_conn, pins_pos, constraints, metrics, fp_sol = make_fake_batch_item()
    inst = from_training_batch_item(area_target, b2b_conn, p2b_conn, pins_pos, constraints, metrics, fp_sol)

    tp = inst.target_positions
    assert tp[2].tolist() == pytest.approx([7.0, 8.0, 4.0, 5.0])


def test_free_block_target_position_stays_unset():
    area_target, b2b_conn, p2b_conn, pins_pos, constraints, metrics, fp_sol = make_fake_batch_item()
    inst = from_training_batch_item(area_target, b2b_conn, p2b_conn, pins_pos, constraints, metrics, fp_sol)

    assert inst.target_positions[0].tolist() == [-1.0, -1.0, -1.0, -1.0]


@pytest.mark.skipif(not TRAINING_SET_PRESENT, reason="real training set (floorset_lite/) not downloaded")
def test_real_training_samples_have_no_negative_dimensions_for_fixed_or_preplaced_blocks():
    from iccad2026_evaluate import get_training_dataloader

    loader = get_training_dataloader(data_path="../", batch_size=1, num_samples=20, shuffle=False)
    checked_any_fixed_or_preplaced = False

    for batch in loader:
        area_target, b2b_conn, p2b_conn, pins_pos, constraints, _, fp_sol, metrics = batch
        inst = from_training_batch_item(
            area_target.squeeze(0), b2b_conn.squeeze(0), p2b_conn.squeeze(0),
            pins_pos.squeeze(0), constraints.squeeze(0), metrics.squeeze(0), fp_sol.squeeze(0),
        )
        nc = inst.constraints.shape[1]
        for i in range(inst.block_count):
            is_fixed = nc > 0 and inst.constraints[i, 0] != 0
            is_preplaced = nc > 1 and inst.constraints[i, 1] != 0
            if is_fixed or is_preplaced:
                checked_any_fixed_or_preplaced = True
                w, h = inst.target_positions[i, 2].item(), inst.target_positions[i, 3].item()
                assert w > 0 and h > 0, f"block {i}: expected real (w,h), got ({w},{h})"

    assert checked_any_fixed_or_preplaced, "expected at least one fixed/preplaced block across 20 real samples"


@pytest.mark.skipif(not TRAINING_SET_PRESENT, reason="real training set (floorset_lite/) not downloaded")
def test_real_training_rollouts_respect_hard_constraints(rollout):
    """End-to-end regression test for the two bugs found via an overnight
    pretraining run against the real training set: fixed-shape blocks
    getting w=h=-1 (missing fp_sol), and MIB followers with differing area
    targets breaking their own area-tolerance constraint by copying the
    leader's shape unconditionally."""
    from iccad2026_evaluate import check_overlap, get_training_dataloader

    loader = get_training_dataloader(data_path="../", batch_size=1, num_samples=10, shuffle=False)

    for batch in loader:
        area_target, b2b_conn, p2b_conn, pins_pos, constraints, _, fp_sol, metrics = batch
        inst = from_training_batch_item(
            area_target.squeeze(0), b2b_conn.squeeze(0), p2b_conn.squeeze(0),
            pins_pos.squeeze(0), constraints.squeeze(0), metrics.squeeze(0), fp_sol.squeeze(0),
        )
        env = GridPlacementEnv(inst, grid_dim=48)
        positions = rollout(env, seed=0, max_seed_tries=30)

        assert check_overlap(positions) == 0

        nc = inst.constraints.shape[1]
        for i in range(inst.block_count):
            is_fixed = nc > 0 and inst.constraints[i, 0] != 0
            is_preplaced = nc > 1 and inst.constraints[i, 1] != 0
            x, y, w, h = positions[i]
            if is_fixed or is_preplaced:
                tw, th = inst.target_positions[i, 2].item(), inst.target_positions[i, 3].item()
                assert w == pytest.approx(tw, abs=1e-4)
                assert h == pytest.approx(th, abs=1e-4)
            else:
                area = float(inst.area_targets[i])
                if area > 0:
                    assert w * h == pytest.approx(area, rel=1e-4)

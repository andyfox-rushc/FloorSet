import torch

from rl.anneal import anneal_polish
from rl.data import synthetic_instance
from rl.reward import inference_reward
from iccad2026_evaluate import check_overlap


def test_never_introduces_overlap():
    instance = synthetic_instance(
        area_targets=[4.0, 4.0, 4.0, 4.0],
        b2b_edges=[(0, 1, 1.0), (1, 2, 1.0), (2, 3, 1.0)],
    )
    positions = [(0.0, 0.0, 2.0, 2.0), (40.0, 0.0, 2.0, 2.0),
                 (0.0, 40.0, 2.0, 2.0), (40.0, 40.0, 2.0, 2.0)]
    result = anneal_polish(instance, positions, time_budget=2.0, max_iterations=500, seed=0)
    assert check_overlap(result) == 0


def test_never_moves_preplaced_block():
    instance = synthetic_instance(
        area_targets=[4.0, 4.0],
        constraints=[[0, 1, 0, 0, 0], [0, 0, 0, 0, 0]],
        target_positions=[(10.0, 10.0, 2.0, 2.0), (-1, -1, -1, -1)],
        b2b_edges=[(0, 1, 1.0)],
    )
    positions = [(10.0, 10.0, 2.0, 2.0), (30.0, 30.0, 2.0, 2.0)]
    result = anneal_polish(instance, positions, time_budget=2.0, max_iterations=500, seed=0)
    assert result[0] == (10.0, 10.0, 2.0, 2.0)


def test_never_resizes_fixed_shape_block():
    instance = synthetic_instance(
        area_targets=[6.0, 4.0],
        constraints=[[1, 0, 0, 0, 0], [0, 0, 0, 0, 0]],
        target_positions=[(-1, -1, 3.0, 2.0), (-1, -1, -1, -1)],
        b2b_edges=[(0, 1, 1.0)],
    )
    positions = [(0.0, 0.0, 3.0, 2.0), (20.0, 20.0, 2.0, 2.0)]
    result = anneal_polish(instance, positions, time_budget=2.0, max_iterations=500, seed=0)
    assert (result[0][2], result[0][3]) == (3.0, 2.0)


def test_improves_or_matches_a_deliberately_bad_layout():
    # Two strongly connected blocks placed far apart -- annealing (which can
    # relocate a block flush against another) should easily find a much
    # cheaper (lower hpwl+area) placement than never moving anything.
    instance = synthetic_instance(
        area_targets=[4.0, 4.0],
        b2b_edges=[(0, 1, 10.0)],
    )
    positions = [(0.0, 0.0, 2.0, 2.0), (100.0, 100.0, 2.0, 2.0)]
    before_cost = -inference_reward(instance, positions)
    result = anneal_polish(instance, positions, time_budget=3.0, max_iterations=1000, seed=0)
    after_cost = -inference_reward(instance, result)
    assert check_overlap(result) == 0
    assert after_cost <= before_cost


def test_group_shift_moves_whole_cluster_together():
    # A wide blocking block (0) pins the anchor (1) at x=98 (blocked from
    # sliding further left) and independently pins the cluster (2,3) at
    # x=0,y=4 (already at floor_x, blocked from sliding down further by the
    # same blocker) -- both are genuinely stuck at their own tightest
    # independently-reachable position. inference_reward compacts its input
    # before scoring, and ordinary gravity compaction (verified: this
    # geometry is already a fixed point of compact()) can only ever *reduce*
    # a block's coordinates, never increase them -- so it can't pull the
    # cluster rightward to meet the anchor. Only the group-shift move
    # (flush-translate the whole cluster against a chosen anchor edge, which
    # can move in any direction) can close this gap.
    instance = synthetic_instance(
        area_targets=[392.0, 16.0, 16.0, 16.0],
        constraints=[[0, 0, 0, 0, 0], [0, 0, 0, 0, 0], [0, 0, 0, 1, 0], [0, 0, 0, 1, 0]],
        b2b_edges=[(1, 2, 50.0)],
    )
    positions = [
        (0.0, 0.0, 98.0, 4.0),   # wide blocker
        (98.0, 0.0, 4.0, 4.0),   # anchor, flush against the blocker's right edge
        (0.0, 4.0, 4.0, 4.0),    # cluster member 1, flush against the blocker's top edge
        (4.0, 4.0, 4.0, 4.0),    # cluster member 2, touching member 1
    ]
    before_cost = -inference_reward(instance, positions)
    result = anneal_polish(instance, positions, time_budget=5.0, max_iterations=4000, seed=0)
    after_cost = -inference_reward(instance, result)

    assert check_overlap(result) == 0
    assert after_cost < before_cost * 0.25  # big improvement, not incremental noise
    # (Not asserting the cluster's internal arrangement stays exactly (4,0)
    # apart: relocate/swap can still legally move a single cluster member on
    # their own, same as before this move existed -- grouping is a *soft*
    # constraint here, priced into cost via violations_relative, not
    # hard-gated for every move type. Only the group-shift move itself
    # guarantees a same-delta translation, which is verified directly by
    # reading its implementation, not by this end-to-end test.)


def test_group_shift_never_crashes_when_cluster_is_the_whole_layout():
    # No block exists outside the cluster to flush against -- the group-shift
    # move must skip itself gracefully, not crash on an empty anchor choice.
    instance = synthetic_instance(
        area_targets=[4.0, 4.0],
        constraints=[[0, 0, 0, 1, 0], [0, 0, 0, 1, 0]],
        b2b_edges=[(0, 1, 1.0)],
    )
    positions = [(0.0, 0.0, 2.0, 2.0), (2.0, 0.0, 2.0, 2.0)]
    result = anneal_polish(instance, positions, time_budget=1.0, max_iterations=500, seed=0)
    assert check_overlap(result) == 0


def test_too_few_movable_blocks_returns_input_unchanged():
    instance = synthetic_instance(
        area_targets=[4.0],
        constraints=[[0, 1, 0, 0, 0]],
        target_positions=[(5.0, 5.0, 2.0, 2.0)],
    )
    positions = [(5.0, 5.0, 2.0, 2.0)]
    result = anneal_polish(instance, positions, time_budget=1.0)
    assert result == positions

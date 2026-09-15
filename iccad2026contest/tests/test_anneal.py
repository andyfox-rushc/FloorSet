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


def test_too_few_movable_blocks_returns_input_unchanged():
    instance = synthetic_instance(
        area_targets=[4.0],
        constraints=[[0, 1, 0, 0, 0]],
        target_positions=[(5.0, 5.0, 2.0, 2.0)],
    )
    positions = [(5.0, 5.0, 2.0, 2.0)]
    result = anneal_polish(instance, positions, time_budget=1.0)
    assert result == positions

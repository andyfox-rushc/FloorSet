import math

import pytest

from iccad2026_evaluate import BETA, M_PENALTY, compute_cost, evaluate_solution
from rl.data import synthetic_instance
from rl.reward import inference_reward, pretraining_reward


def test_pretraining_reward_matches_compute_cost_exactly():
    inst = synthetic_instance(
        area_targets=[4.0, 9.0],
        b2b_edges=[(0, 1, 1.0)],
        baseline_metrics={'hpwl_baseline': 5.0, 'area_baseline': 20.0},
    )
    positions = [(0.0, 0.0, 2.0, 2.0), (2.0, 0.0, 3.0, 3.0)]

    reward = pretraining_reward(inst, positions, runtime=1.0)

    metrics = evaluate_solution(
        {'positions': positions, 'runtime': 1.0},
        inst.baseline_metrics,
        inst.constraints,
        inst.b2b_connectivity,
        inst.p2b_connectivity,
        inst.pins_pos,
        inst.area_targets,
        [tuple(row.tolist()) for row in inst.target_positions],
        median_runtime=1.0,
    )
    expected_cost = compute_cost(metrics.hpwl_gap, metrics.area_gap,
                                  metrics.violations_relative, 1.0, metrics.is_feasible)
    assert reward == pytest.approx(-expected_cost)


def test_pretraining_reward_requires_baseline():
    inst = synthetic_instance(area_targets=[4.0, 9.0])
    with pytest.raises(AssertionError):
        pretraining_reward(inst, [(0, 0, 2, 2), (2, 0, 3, 3)])


def test_inference_reward_feasible_case_matches_manual_formula():
    inst = synthetic_instance(area_targets=[4.0, 9.0], b2b_edges=[(0, 1, 2.0)])
    positions = [(0.0, 0.0, 2.0, 2.0), (2.0, 0.0, 3.0, 3.0)]

    reward = inference_reward(inst, positions, area_weight=0.01)

    # hand-compute: hpwl_b2b = weight * (|dx| + |dy|) between centroids
    c0 = (1.0, 1.0)
    c1 = (3.5, 1.5)
    hpwl = 2.0 * (abs(c1[0] - c0[0]) + abs(c1[1] - c0[1]))
    bbox_area = 5.0 * 3.0  # x:[0,5], y:[0,3]
    quality = hpwl + 0.01 * bbox_area
    expected = -(quality * math.exp(BETA * 0.0))  # no constraints -> 0 violations
    assert reward == pytest.approx(expected)


def test_inference_reward_infeasible_case_is_m_penalty():
    inst = synthetic_instance(area_targets=[4.0, 9.0])
    overlapping = [(0.0, 0.0, 2.0, 2.0), (1.0, 1.0, 3.0, 3.0)]
    reward = inference_reward(inst, overlapping)
    assert reward == pytest.approx(-M_PENALTY)


def test_inference_reward_does_not_require_baseline():
    inst = synthetic_instance(area_targets=[4.0, 9.0])
    assert inst.baseline_metrics is None
    # should not raise
    inference_reward(inst, [(0.0, 0.0, 2.0, 2.0), (2.0, 0.0, 3.0, 3.0)])

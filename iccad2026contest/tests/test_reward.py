import math

import pytest

from iccad2026_evaluate import ALPHA, BETA, M_PENALTY, evaluate_solution
from rl.data import synthetic_instance
from rl.reward import inference_reward, pretraining_reward, step_quality_delta


def test_pretraining_reward_matches_uncapped_quality_formula():
    # pretraining_reward deliberately does NOT match compute_cost's capped
    # leaderboard score -- see rl/reward.py's _uncapped_quality docstring:
    # the M_PENALTY cap flattens the PPO advantage signal for any rollout
    # bad enough to hit it, which is not a property a training reward
    # should have.
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
    assert metrics.is_feasible
    quality_factor = 1 + ALPHA * (max(0, metrics.hpwl_gap) + max(0, metrics.area_gap))
    violation_factor = math.exp(BETA * metrics.violations_relative)
    expected_quality = quality_factor * violation_factor
    assert reward == pytest.approx(-expected_quality)


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


def test_step_quality_delta_pretraining_matches_alpha_weighting():
    baseline = {'hpwl_baseline': 5.0, 'area_baseline': 20.0}
    delta = step_quality_delta(delta_wl=2.0, delta_area=4.0, use_baseline=True,
                                baseline_metrics=baseline)
    assert delta == pytest.approx(-ALPHA * (2.0 / 5.0 + 4.0 / 20.0))


def test_step_quality_delta_inference_matches_area_weight():
    delta = step_quality_delta(delta_wl=3.0, delta_area=10.0, use_baseline=False,
                                baseline_metrics=None)
    assert delta == pytest.approx(-(3.0 + 0.01 * 10.0))


def test_step_quality_delta_sums_to_pretraining_reward_before_the_floor():
    # step_quality_delta never clips (deltas are always >= 0, so the running
    # sum is always >= 0 too); it's meant to be summed across a whole
    # rollout and then have exactly one residual correction applied for the
    # max(0, gap) floor -- see rl/ppo.py's collect_episode. Confirm the
    # unclipped sum matches pretraining_reward's *un-floored* accumulation:
    # -ALPHA * (hpwl_total/hpwl_baseline + area_total/area_baseline).
    baseline = {'hpwl_baseline': 5.0, 'area_baseline': 20.0}
    inst = synthetic_instance(
        area_targets=[4.0, 9.0], b2b_edges=[(0, 1, 1.0)], baseline_metrics=baseline,
    )
    positions = [(0.0, 0.0, 2.0, 2.0), (2.0, 0.0, 3.0, 3.0)]

    # Two "steps": block 0 alone (no neighbor placed yet, bbox from nothing),
    # then block 1 (resolves the b2b edge, grows the bbox further).
    step0 = step_quality_delta(delta_wl=0.0, delta_area=2.0 * 2.0, use_baseline=True,
                                baseline_metrics=baseline)
    c0, c1 = (1.0, 1.0), (3.5, 1.5)
    wl = 1.0 * (abs(c1[0] - c0[0]) + abs(c1[1] - c0[1]))
    full_bbox_area = 5.0 * 3.0
    step1 = step_quality_delta(delta_wl=wl, delta_area=full_bbox_area - 4.0,
                                use_baseline=True, baseline_metrics=baseline)

    assert (step0 + step1) == pytest.approx(-ALPHA * (wl / 5.0 + full_bbox_area / 20.0))

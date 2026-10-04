import math

import pytest

from iccad2026_evaluate import ALPHA, BETA, M_PENALTY, evaluate_solution
from rl.data import synthetic_instance
from rl.reward import (_reward_and_violation_factor, inference_reward, pretraining_reward,
                        step_quality_delta)


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


def test_inference_reward_infeasible_case_scales_with_violation_count():
    # A flat -M_PENALTY regardless of severity gives PPO zero gradient
    # signal toward "less infeasible" -- the same zero-signal-plateau bug
    # already fixed for pretraining_reward's feasible-but-bad case (see
    # module docstring), just on the infeasible branch instead. Both
    # inference_reward's infeasible branch now scale with hard_violations,
    # so two differently-bad infeasible layouts are distinguishable.
    inst = synthetic_instance(area_targets=[4.0, 9.0])
    overlapping = [(0.0, 0.0, 2.0, 2.0), (1.0, 1.0, 3.0, 3.0)]
    reward = inference_reward(inst, overlapping)
    assert reward < -M_PENALTY  # strictly worse than the flat floor, not equal to it


def test_inference_reward_does_not_require_baseline():
    inst = synthetic_instance(area_targets=[4.0, 9.0])
    assert inst.baseline_metrics is None
    # should not raise
    inference_reward(inst, [(0.0, 0.0, 2.0, 2.0), (2.0, 0.0, 3.0, 3.0)])


def test_violation_factor_rescale_gives_a_bounded_residual():
    # Before the credit-assignment fix, rl/ppo.py's collect_episode had to
    # absorb the ENTIRE quality*violation interaction into one residual on a
    # single transition: old_residual = -V + ALPHA*Q*(1-V), unbounded in Q.
    # The fix rescales every step's tracked delta by V first, so the only
    # leftover is -V (Q-independent) -- see rl/reward.py's
    # _reward_and_violation_factor docstring for the derivation. This test
    # deliberately engineers both a huge Q (near-zero baseline) and a real
    # V > 1 (an unsatisfied cluster-touch constraint) so the two formulas'
    # outputs are dramatically, unmistakably different.
    baseline = {'hpwl_baseline': 1e-4, 'area_baseline': 1e-4}
    inst = synthetic_instance(
        area_targets=[4.0, 4.0],
        constraints=[[0, 0, 0, 1, 0], [0, 0, 0, 1, 0]],  # same cluster, must touch
        b2b_edges=[(0, 1, 5.0)],
        baseline_metrics=baseline,
    )
    positions = [(0.0, 0.0, 2.0, 2.0), (50.0, 0.0, 2.0, 2.0)]  # far apart -> violates grouping

    reward, violation_factor = _reward_and_violation_factor(inst, positions, use_baseline=True)
    assert violation_factor > 1.0 + 1e-6  # confirm the setup actually produced a real violation

    Q = (-reward / violation_factor - 1) / ALPHA
    assert Q > 10  # confirm the near-zero baseline actually produced a huge gap

    old_residual = -violation_factor + ALPHA * Q * (1 - violation_factor)
    new_residual = -violation_factor

    assert abs(new_residual) == pytest.approx(violation_factor)
    assert abs(new_residual) < abs(old_residual) / 10


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


# -- normalized inference_reward (reference_metrics set) --------------------

def _referenced_instance(hpwl_ref=10.0, area_ref=30.0):
    inst = synthetic_instance(area_targets=[4.0, 9.0], b2b_edges=[(0, 1, 2.0)])
    inst.reference_metrics = {'hpwl_ref': hpwl_ref, 'area_ref': area_ref}
    return inst


def test_normalized_inference_reward_matches_ratio_formula():
    inst = _referenced_instance()
    positions = [(0.0, 0.0, 2.0, 2.0), (2.0, 0.0, 3.0, 3.0)]
    hpwl = 2.0 * (abs(3.5 - 1.0) + abs(1.5 - 1.0))
    bbox_area = 5.0 * 3.0
    expected = -(1 + ALPHA * (hpwl / 10.0 + bbox_area / 30.0))
    assert inference_reward(inst, positions) == pytest.approx(expected)


def test_normalized_step_deltas_use_the_reference():
    delta = step_quality_delta(delta_wl=3.0, delta_area=10.0, use_baseline=False,
                                baseline_metrics=None,
                                reference_metrics={'hpwl_ref': 6.0, 'area_ref': 40.0})
    assert delta == pytest.approx(-ALPHA * (3.0 / 6.0 + 10.0 / 40.0))


def test_normalized_infeasible_penalty_stays_below_bad_feasible_rewards():
    # The bug this guards: a failed rollout (-M_PENALTY) used to outscore
    # every feasible one at fine-tune time. With max violations
    # (violation_factor = e^BETA) and quality 5x the reference, a feasible
    # rollout must still beat an infeasible one.
    inst = _referenced_instance()
    overlapping = [(0.0, 0.0, 2.0, 2.0), (1.0, 1.0, 3.0, 3.0)]
    worst_plausible_feasible = -(1 + ALPHA * (5.0 + 5.0)) * math.exp(BETA)
    assert inference_reward(inst, overlapping) < worst_plausible_feasible


def test_unfinished_rollout_gets_scaled_penalty_under_reference():
    from rl.networks import ActorCritic
    from rl.ppo import collect_episode
    # 2x2 grid can't hold three blocks of a 10x-too-small canvas -> the
    # rollout aborts and must get the scaled penalty, not the flat -10.
    inst = synthetic_instance(area_targets=[100.0, 100.0, 100.0], b2b_edges=[(0, 1, 1.0)])
    inst.reference_metrics = {'hpwl_ref': 10.0, 'area_ref': 300.0}
    import rl.env as env_mod
    original = env_mod.CANVAS_PADDING
    env_mod.CANVAS_PADDING = 0.3
    try:
        ep = collect_episode(ActorCritic(hidden_dim=16, num_gnn_layers=1, cnn_channels=4),
                             inst, grid_dim=8, use_baseline=False)
    finally:
        env_mod.CANVAS_PADDING = original
    assert ep.positions is None
    assert ep.reward == pytest.approx(-M_PENALTY * math.exp(BETA))


def test_training_failure_penalty_is_graded_by_progress():
    from rl.reward import FAIL_FLOOR, infeasible_penalty
    from iccad2026_evaluate import M_PENALTY
    inst = None
    assert infeasible_penalty(inst, True, placed_fraction=1.0) == pytest.approx(-FAIL_FLOOR)
    assert infeasible_penalty(inst, True, placed_fraction=0.0) == pytest.approx(-M_PENALTY)
    mid = infeasible_penalty(inst, True, placed_fraction=0.5)
    assert -M_PENALTY < mid < -FAIL_FLOOR
    assert infeasible_penalty(inst, True) == pytest.approx(-M_PENALTY)   # no progress info: flat

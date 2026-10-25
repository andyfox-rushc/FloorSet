"""
Tests for the per-step return-to-go credit assignment in rl/ppo.py: before
this, every transition in an episode shared one identical whole-episode
advantage, making it impossible for PPO to reinforce the specific position
choices that actually caused wirelength/area over the ones that didn't. See
rl/env.py's _step_deltas and rl/reward.py's step_quality_delta for the
exact (non-approximating) per-step decomposition this relies on.
"""

import pytest
import torch

from rl.data import synthetic_instance
from rl.networks import ActorCritic
from rl.ppo import collect_episode


def make_instance(with_baseline: bool):
    baseline = {'hpwl_baseline': 5.0, 'area_baseline': 20.0} if with_baseline else None
    return synthetic_instance(
        area_targets=[4.0, 9.0, 6.0],
        b2b_edges=[(0, 1, 1.0), (1, 2, 2.0)],
        baseline_metrics=baseline,
    )


@pytest.mark.parametrize("use_baseline", [True, False])
def test_step_rewards_sum_exactly_to_episode_reward(use_baseline):
    # Holds regardless of whether the rollout ended up feasible -- an
    # untrained/random policy occasionally picks an extreme aspect ratio
    # that doesn't fit anywhere (see collect_episode's RuntimeError catch),
    # and the residual correction must still land on the exact terminal
    # reward in that case too, not just the common feasible path. Try a
    # handful of seeds so both outcomes get covered.
    for seed in range(8):
        torch.manual_seed(seed)
        inst = make_instance(with_baseline=use_baseline)
        net = ActorCritic(hidden_dim=16, num_gnn_layers=2, cnn_channels=8)

        ep = collect_episode(net, inst, grid_dim=16, use_baseline=use_baseline)
        if not ep.transitions:
            continue  # crashed before any transition was recorded -- nothing to check
        tracked = sum(tr.step_reward for tr in ep.transitions)
        assert tracked == pytest.approx(ep.reward, abs=1e-5)


@pytest.mark.parametrize("use_baseline", [True, False])
def test_return_to_go_of_first_transition_equals_episode_reward(use_baseline):
    # return_to_go is a suffix sum over step_reward; the first transition's
    # suffix is the whole episode, so it must equal the terminal reward
    # exactly -- this is the value the value head is now trained to predict
    # at progress~0, instead of a flat constant shared by every step.
    torch.manual_seed(1)
    inst = make_instance(with_baseline=use_baseline)
    net = ActorCritic(hidden_dim=16, num_gnn_layers=2, cnn_channels=8)

    ep = collect_episode(net, inst, grid_dim=16, use_baseline=use_baseline)
    assert ep.transitions
    assert ep.transitions[0].return_to_go == pytest.approx(ep.reward, abs=1e-5)


def test_return_to_go_varies_across_transitions_within_one_episode():
    # The core bug this fixes: previously every transition in an episode
    # got the identical advantage target. Confirm return_to_go actually
    # differs step-to-step now (not all transitions collapse to one value).
    torch.manual_seed(2)
    inst = make_instance(with_baseline=True)
    net = ActorCritic(hidden_dim=16, num_gnn_layers=2, cnn_channels=8)

    ep = collect_episode(net, inst, grid_dim=16, use_baseline=True)
    assert ep.transitions
    values = {tr.return_to_go for tr in ep.transitions}
    assert len(values) > 1, "expected return_to_go to vary across an episode's transitions"


def test_aspect_transitions_have_zero_immediate_step_reward():
    # Aspect choices don't themselves resolve any edge or grow the bbox --
    # only position placement does -- so their own step_reward must be 0
    # (they still get a non-trivial return_to_go from later steps).
    torch.manual_seed(3)
    inst = make_instance(with_baseline=True)
    net = ActorCritic(hidden_dim=16, num_gnn_layers=2, cnn_channels=8)

    ep = collect_episode(net, inst, grid_dim=16, use_baseline=True)
    aspect_transitions = [tr for tr in ep.transitions if tr.kind == 'aspect']
    assert aspect_transitions
    for tr in aspect_transitions:
        assert tr.step_reward == 0.0

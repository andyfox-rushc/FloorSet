"""
The real RL-correctness test: prove the PPO loop actually learns something,
not just that the plumbing runs. A tiny 4-block chain-connected instance has
an easy, learnable win (place connected blocks close together to cut
wirelength+bbox area) with plenty of canvas room to get it wrong, so a
converging policy should clearly beat its own early-training performance.
"""

import statistics

import torch

from rl.data import synthetic_instance
from rl.networks import ActorCritic
from rl.ppo import collect_batch, ppo_update


def make_chain_instance():
    return synthetic_instance(
        area_targets=[4.0, 4.0, 4.0, 4.0],
        b2b_edges=[(0, 1, 1.0), (1, 2, 1.0), (2, 3, 1.0), (0, 3, 1.0)],
    )


def test_ppo_improves_average_episode_reward():
    torch.manual_seed(0)
    inst = make_chain_instance()
    net = ActorCritic(hidden_dim=16, num_gnn_layers=2, num_aspects=9, cnn_channels=8)
    optimizer = torch.optim.Adam(net.parameters(), lr=3e-3)

    grid_dim = 12
    episodes_per_iter = 8
    num_iterations = 40

    iteration_means = []
    for _ in range(num_iterations):
        episodes = collect_batch(net, inst, grid_dim=grid_dim, use_baseline=False,
                                  num_episodes=episodes_per_iter)
        iteration_means.append(statistics.mean(ep.reward for ep in episodes))
        ppo_update(net, optimizer, episodes, epochs=4)

    early = statistics.mean(iteration_means[:5])
    late = statistics.mean(iteration_means[-5:])

    assert late > early + abs(early) * 0.1, (
        f"expected meaningful reward improvement, got early={early:.3f} late={late:.3f}"
    )


def test_reward_approx_head_learns_to_predict_realized_reward():
    torch.manual_seed(1)
    inst = make_chain_instance()
    net = ActorCritic(hidden_dim=16, num_gnn_layers=2, num_aspects=9, cnn_channels=8)
    optimizer = torch.optim.Adam(net.parameters(), lr=3e-3)

    grid_dim = 12
    early_episodes = collect_batch(net, inst, grid_dim=grid_dim, use_baseline=False, num_episodes=8)
    early_stats = ppo_update(net, optimizer, early_episodes, epochs=1)

    for _ in range(30):
        episodes = collect_batch(net, inst, grid_dim=grid_dim, use_baseline=False, num_episodes=8)
        late_stats = ppo_update(net, optimizer, episodes, epochs=4)

    assert late_stats['reward_approx_loss'] < early_stats['reward_approx_loss']

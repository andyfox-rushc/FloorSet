"""
Tests for the staged encoder reward-prediction warm-start (rl/pretrain.py),
distinct from the PPO overfit test: this proves (a) the supervised
reward-prediction task actually reduces MSE across a diverse corpus, and
(b) the encoder's learned weights carry forward while the reward-approx
head is reset fresh afterward (per the paper: "the prediction layer is
removed").
"""

import copy

import torch

from rl.data import synthetic_instance
from rl.networks import ActorCritic
from rl.pretrain import collect_reward_prediction_corpus, pretrain_encoder_on_reward_prediction


def make_diverse_instances():
    return [
        synthetic_instance(area_targets=[4.0, 9.0, 16.0], b2b_edges=[(0, 1, 1.0), (1, 2, 1.0)]),
        synthetic_instance(area_targets=[4.0, 4.0, 4.0, 4.0], b2b_edges=[(0, 1, 1.0), (2, 3, 1.0)]),
        synthetic_instance(area_targets=[9.0, 9.0], b2b_edges=[(0, 1, 2.0)]),
    ]


def test_collect_corpus_shape():
    torch.manual_seed(0)
    instances = make_diverse_instances()
    corpus = collect_reward_prediction_corpus(instances, rollouts_per_instance=2, grid_dim=10)
    # at least one sample per rollout (episodes with 0 transitions still get 1 sample)
    assert len(corpus) >= len(instances) * 2
    for s in corpus:
        assert isinstance(s.reward, float)
        assert 0.0 <= s.progress <= 1.0 or s.progress == 0.0


def test_pretraining_reduces_reward_prediction_error():
    torch.manual_seed(0)
    instances = make_diverse_instances()
    net = ActorCritic(hidden_dim=16, num_gnn_layers=2, cnn_channels=8)
    corpus = collect_reward_prediction_corpus(instances, net=net, rollouts_per_instance=6, grid_dim=10)

    losses = pretrain_encoder_on_reward_prediction(net, corpus, epochs=8, batch_size=8)

    assert len(losses) == 8
    assert losses[-1] < losses[0], f"expected MSE to decrease, got {losses}"


def test_encoder_weights_change_but_head_is_reset():
    torch.manual_seed(0)
    instances = make_diverse_instances()
    net = ActorCritic(hidden_dim=16, num_gnn_layers=2, cnn_channels=8)

    encoder_before = copy.deepcopy(net.encoder.state_dict())
    old_head = net.reward_approx_head

    corpus = collect_reward_prediction_corpus(instances, net=net, rollouts_per_instance=4, grid_dim=10)
    pretrain_encoder_on_reward_prediction(net, corpus, epochs=3, batch_size=8)

    encoder_after = net.encoder.state_dict()
    changed = any(not torch.allclose(encoder_before[k], encoder_after[k]) for k in encoder_before)
    assert changed, "encoder weights should have been updated by pretraining"

    assert net.reward_approx_head is not old_head, "reward_approx_head should be a fresh module after pretraining"

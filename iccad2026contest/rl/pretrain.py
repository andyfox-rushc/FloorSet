"""
Staged reward-prediction pretraining for the shared encoder, mirroring
Goldie/Mirhoseini ("Chip Placement with Deep Reinforcement Learning"):
representation learning is grounded in the supervised task of predicting
placement quality (realized episode reward) across a diverse corpus of
instances, collected from vanilla (untrained-policy) rollouts. After this
supervised phase, the reward-prediction head is discarded -- per the paper,
"the prediction layer is removed" -- and only the trained ENCODER carries
forward, warm-started rather than randomly initialized, into PPO training.

This is a distinct, separate phase from the reward-approximation head's
role *during* PPO (see rl/ppo.py's ppo_update, which trains it jointly as an
auxiliary loss once PPO starts): that head is reset fresh after this
pretraining phase so it doesn't bias early PPO advantage/aux-loss estimates
with predictions learned from a different (randomly-acting) policy's
rollouts.
"""

from dataclasses import dataclass
from typing import List, Optional

import torch

from .encoder import build_block_features, build_pin_features
from .networks import ActorCritic, ScalarHead
from .ppo import canvas_scale_of, collect_episode


@dataclass
class RewardPredictionSample:
    block_feats: torch.Tensor
    pin_feats: torch.Tensor
    b2b_edges: torch.Tensor
    p2b_edges: torch.Tensor
    progress: float
    reward: float


def collect_reward_prediction_corpus(
    instances: List,
    net: Optional[ActorCritic] = None,
    rollouts_per_instance: int = 4,
    grid_dim: int = 48,
) -> List[RewardPredictionSample]:
    """Runs rollouts (with `net`, or a fresh randomly-initialized network if
    none given -- i.e. "vanilla" rollouts, matching the paper's "collected by
    running vanilla RL...and gathering snapshots") across many instances,
    recording one (state, eventual realized reward) sample per placement
    step of every rollout."""
    if net is None:
        net = ActorCritic()

    samples: List[RewardPredictionSample] = []
    for inst in instances:
        scale = canvas_scale_of(inst)
        block_feats = build_block_features(inst, scale)
        pin_feats = build_pin_features(inst, scale)
        use_baseline = inst.baseline_metrics is not None

        for _ in range(rollouts_per_instance):
            ep = collect_episode(net, inst, grid_dim=grid_dim, use_baseline=use_baseline)
            progresses = [tr.progress for tr in ep.transitions] or [0.0]
            for progress in progresses:
                samples.append(RewardPredictionSample(
                    block_feats, pin_feats, ep.b2b_edges, ep.p2b_edges, progress, ep.reward,
                ))
    return samples


def pretrain_encoder_on_reward_prediction(
    net: ActorCritic,
    corpus: List[RewardPredictionSample],
    epochs: int = 5,
    lr: float = 1e-3,
    batch_size: int = 16,
    verbose: bool = False,
) -> List[float]:
    """Supervised warm-start of net.encoder (see module docstring). Mutates
    `net` in place: encoder weights are trained and kept; reward_approx_head
    is trained during this phase then reinitialized fresh afterward. Returns
    per-epoch mean-squared error (for tests/monitoring)."""
    assert corpus, "pretrain_encoder_on_reward_prediction called with an empty corpus"

    params = list(net.encoder.parameters()) + list(net.reward_approx_head.parameters())
    optimizer = torch.optim.Adam(params, lr=lr)
    n = len(corpus)
    epoch_losses = []

    for epoch in range(epochs):
        perm = torch.randperm(n).tolist()
        total_loss = 0.0
        for start in range(0, n, batch_size):
            batch_idx = perm[start:start + batch_size]
            optimizer.zero_grad()
            batch_loss = torch.tensor(0.0)
            for i in batch_idx:
                s = corpus[i]
                _, global_emb = net.encode(s.block_feats, s.pin_feats, s.b2b_edges, s.p2b_edges)
                pred = net.reward_approx(global_emb, torch.tensor(s.progress, dtype=torch.float32))
                batch_loss = batch_loss + (pred - s.reward) ** 2
            batch_loss = batch_loss / len(batch_idx)
            batch_loss.backward()
            optimizer.step()
            total_loss += batch_loss.item() * len(batch_idx)

        avg = total_loss / n
        epoch_losses.append(avg)
        if verbose:
            print(f"  reward-prediction pretrain epoch {epoch + 1}/{epochs}: mse={avg:.4f}")

    # "the prediction layer is removed" -- a stale head trained against a
    # randomly-acting policy's rollouts would otherwise bias PPO's own
    # reward-approx auxiliary loss from the very first update.
    net.reward_approx_head = ScalarHead(net.hidden_dim)
    return epoch_losses

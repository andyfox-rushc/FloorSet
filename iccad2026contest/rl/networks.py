"""
Policy / value / reward-approximation heads sitting on top of the shared
NetlistEncoder trunk. Kept decoupled from GridPlacementEnv -- everything
here takes plain tensors, so it's independently testable (shapes, masking,
gradient flow) without a live environment.

PolicyNet has two heads, mirroring the two-phase action in env.py:
    - aspect head:   MLP over [block_embedding, global_embedding, progress]
                     -> logits over the fixed aspect-ratio bucket list.
    - position head: a small CNN over the occupancy grid (+ cluster grid),
                     with the current block's embedding broadcast in as
                     extra channels, producing a full [grid_dim, grid_dim]
                     logit map. The caller crops this to the valid
                     [out_h, out_w] region and masks it (see rl/ppo.py).
"""

import torch
import torch.nn as nn
import torch.nn.functional as F

from .encoder import NetlistEncoder

NEG_INF = -1e9


def masked_log_softmax(logits: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
    """logits, mask: same shape; mask True = valid. Returns log-probs that
    are -inf (zero probability) wherever mask is False."""
    assert mask.any(), "masked_log_softmax called with an all-False mask"
    filled = logits.masked_fill(~mask, NEG_INF)
    return F.log_softmax(filled.reshape(-1), dim=0).reshape(logits.shape)


class AspectHead(nn.Module):
    def __init__(self, hidden_dim: int, num_aspects: int):
        super().__init__()
        self.mlp = nn.Sequential(
            nn.Linear(hidden_dim * 2 + 1, hidden_dim), nn.ReLU(),
            nn.Linear(hidden_dim, num_aspects),
        )

    def forward(self, block_embedding, global_embedding, progress):
        x = torch.cat([block_embedding, global_embedding, progress.view(1)], dim=0)
        return self.mlp(x)


class PositionCNN(nn.Module):
    def __init__(self, hidden_dim: int, cnn_channels: int = 16, conv_width: int = 32):
        super().__init__()
        self.embed_proj = nn.Linear(hidden_dim * 2 + 1, cnn_channels)
        self.conv1 = nn.Conv2d(2 + cnn_channels, conv_width, 3, padding=1)
        self.conv2 = nn.Conv2d(conv_width, conv_width, 3, padding=1)
        self.conv3 = nn.Conv2d(conv_width, 1, 1)

    def forward(self, occupancy, cluster_grid, block_embedding, global_embedding, progress):
        g = occupancy.shape[0]
        ctx = torch.cat([block_embedding, global_embedding, progress.view(1)], dim=0)
        emb_map = self.embed_proj(ctx).view(-1, 1, 1).expand(-1, g, g)
        x = torch.stack([occupancy, cluster_grid], dim=0)
        x = torch.cat([x, emb_map], dim=0).unsqueeze(0)
        x = F.relu(self.conv1(x))
        x = F.relu(self.conv2(x))
        return self.conv3(x)[0, 0]


class ScalarHead(nn.Module):
    """Shared shape for ValueNet and RewardApproxNet: pooled graph embedding
    + progress -> scalar."""

    def __init__(self, hidden_dim: int):
        super().__init__()
        self.mlp = nn.Sequential(
            nn.Linear(hidden_dim + 1, hidden_dim), nn.ReLU(),
            nn.Linear(hidden_dim, 1),
        )

    def forward(self, global_embedding, progress):
        x = torch.cat([global_embedding, progress.view(1)], dim=0)
        return self.mlp(x).squeeze(-1)


class ActorCritic(nn.Module):
    def __init__(self, hidden_dim: int = 64, num_gnn_layers: int = 3,
                 num_aspects: int = 9, cnn_channels: int = 16):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.encoder = NetlistEncoder(hidden_dim=hidden_dim, num_layers=num_gnn_layers)
        self.aspect_head = AspectHead(hidden_dim, num_aspects)
        self.position_cnn = PositionCNN(hidden_dim, cnn_channels=cnn_channels)
        self.value_head = ScalarHead(hidden_dim)
        self.reward_approx_head = ScalarHead(hidden_dim)

    def encode(self, block_feats, pin_feats, b2b_edges, p2b_edges):
        """Run once per episode; returns (block_embeddings, global_embedding)."""
        h_block, _ = self.encoder(block_feats, pin_feats, b2b_edges, p2b_edges)
        global_embedding = h_block.mean(dim=0)
        return h_block, global_embedding

    def aspect_logits(self, block_embeddings, global_embedding, block_idx, progress):
        return self.aspect_head(block_embeddings[block_idx], global_embedding, progress)

    def position_logits(self, occupancy, cluster_grid, block_embeddings, global_embedding,
                         block_idx, progress):
        return self.position_cnn(occupancy, cluster_grid, block_embeddings[block_idx],
                                  global_embedding, progress)

    def value(self, global_embedding, progress):
        return self.value_head(global_embedding, progress)

    def reward_approx(self, global_embedding, progress):
        return self.reward_approx_head(global_embedding, progress)

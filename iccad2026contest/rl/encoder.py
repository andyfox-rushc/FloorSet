"""
Hand-rolled edge-weighted GNN encoder over the netlist graph (blocks + pins,
connected by weighted b2b/p2b edges) -- no torch_geometric dependency, just
vectorized scatter-add message passing so it stays fast on CPU.

Computed ONCE per instance at the start of an episode (not re-run every
placement step), matching AlphaChip's static graph-embedding design: the
placement policy conditions on these embeddings plus a small per-step
occupancy-grid state (see rl/networks.py).
"""

from typing import Tuple

import torch
import torch.nn as nn

BLOCK_FEAT_DIM = 9  # [log_area, is_fixed, is_preplaced, has_mib, has_cluster, b_left, b_right, b_top, b_bottom]
PIN_FEAT_DIM = 2    # normalized (x, y)


def build_block_features(instance, canvas_scale: float) -> torch.Tensor:
    n = instance.block_count
    ncols = instance.constraints.shape[1]

    def col(j):
        return instance.constraints[:, j] if ncols > j else torch.zeros(n)

    log_area = torch.log1p(instance.area_targets.clamp(min=0))
    is_fixed = (col(0) != 0).float()
    is_preplaced = (col(1) != 0).float()
    has_mib = (col(2) != 0).float()
    has_cluster = (col(3) != 0).float()
    boundary = col(4).long()
    b_left = ((boundary & 1) != 0).float()
    b_right = ((boundary & 2) != 0).float()
    b_top = ((boundary & 4) != 0).float()
    b_bottom = ((boundary & 8) != 0).float()

    return torch.stack([
        log_area, is_fixed, is_preplaced, has_mib, has_cluster,
        b_left, b_right, b_top, b_bottom,
    ], dim=1)


def build_pin_features(instance, canvas_scale: float) -> torch.Tensor:
    if instance.pins_pos.numel() == 0:
        return torch.zeros(0, PIN_FEAT_DIM)
    return instance.pins_pos / max(canvas_scale, 1e-6)


class NetlistEncoder(nn.Module):
    def __init__(self, hidden_dim: int = 64, num_layers: int = 3,
                 block_feat_dim: int = BLOCK_FEAT_DIM, pin_feat_dim: int = PIN_FEAT_DIM):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.block_embed = nn.Linear(block_feat_dim, hidden_dim)
        self.pin_embed = nn.Linear(pin_feat_dim, hidden_dim)
        self.layers = nn.ModuleList([
            nn.Sequential(
                nn.Linear(hidden_dim * 2, hidden_dim), nn.ReLU(),
                nn.Linear(hidden_dim, hidden_dim),
            )
            for _ in range(num_layers)
        ])

    def forward(
        self,
        block_feats: torch.Tensor,
        pin_feats: torch.Tensor,
        b2b_edges: torch.Tensor,
        p2b_edges: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        n = block_feats.shape[0]
        p = pin_feats.shape[0]

        h_block = self.block_embed(block_feats)
        h_pin = self.pin_embed(pin_feats) if p > 0 else torch.zeros(0, self.hidden_dim)

        b2b_valid = b2b_edges[b2b_edges[:, 0] >= 0] if b2b_edges.numel() > 0 else b2b_edges
        p2b_valid = p2b_edges[p2b_edges[:, 0] >= 0] if p > 0 and p2b_edges.numel() > 0 else p2b_edges[:0]

        for layer in self.layers:
            h_all = torch.cat([h_block, h_pin], dim=0) if p > 0 else h_block
            agg = torch.zeros_like(h_all)

            def add_messages(src_idx, dst_idx, weight):
                if src_idx.numel() == 0:
                    return
                msg = h_all[src_idx] * weight.unsqueeze(-1)
                agg.index_add_(0, dst_idx, msg)

            if b2b_valid.numel() > 0:
                i, j, w = b2b_valid[:, 0].long(), b2b_valid[:, 1].long(), b2b_valid[:, 2]
                add_messages(i, j, w)
                add_messages(j, i, w)

            if p > 0 and p2b_valid.numel() > 0:
                pin_i = p2b_valid[:, 0].long() + n
                blk_j = p2b_valid[:, 1].long()
                w = p2b_valid[:, 2]
                add_messages(pin_i, blk_j, w)
                add_messages(blk_j, pin_i, w)

            h_all = layer(torch.cat([h_all, agg], dim=-1)) + h_all
            h_block = h_all[:n]
            h_pin = h_all[n:] if p > 0 else h_pin

        return h_block, h_pin

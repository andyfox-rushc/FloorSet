"""
One-time migration: expand position_cnn.conv1's input channels from
(2 + cnn_channels) to (3 + cnn_channels) to make room for the new wiremask
channel, inserted at index 2 (after occupancy, cluster_grid; before the
embedding-map channels). Everything else in the checkpoint is untouched,
including the trained encoder -- only conv1's weight tensor changes shape.
The new wiremask channel's weights are zero-initialized so the migrated
checkpoint is functionally IDENTICAL to the old one on the next forward
pass (the new channel contributes nothing until training adapts to use it).
"""
import sys
import torch

src, dst = sys.argv[1], sys.argv[2]
sd = torch.load(src, map_location="cpu")

old_w = sd["position_cnn.conv1.weight"]  # [conv_width, 2+cnn_channels, 3, 3]
conv_width, old_in, kh, kw = old_w.shape
cnn_channels = old_in - 2
new_w = torch.zeros(conv_width, old_in + 1, kh, kw)
new_w[:, :2] = old_w[:, :2]            # occupancy, cluster_grid unchanged
new_w[:, 2] = 0.0                       # new wiremask channel, zero-init
new_w[:, 3:] = old_w[:, 2:]             # embed_map channels shifted by 1
sd["position_cnn.conv1.weight"] = new_w

torch.save(sd, dst)
print(f"migrated {src} -> {dst}: conv1 in_channels {old_in} -> {old_in + 1} "
      f"(cnn_channels={cnn_channels})")

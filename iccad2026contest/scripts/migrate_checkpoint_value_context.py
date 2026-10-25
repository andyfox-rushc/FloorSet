"""
One-time migration: expand value_head/reward_approx_head's first Linear
layer from hidden_dim+1 (global_embedding + progress) to hidden_dim+2
inputs, making room for the new reward-committed-so-far scalar (see
rl/networks.py's ScalarHead docstring -- without it, the value function
cannot condition on anything that happened during a specific rollout,
provably leaving ~10x-300x more variance in the PPO advantage estimate than
the actual per-step signal it's supposed to isolate). The new input's
weights are zero-initialized so the migrated checkpoint is functionally
IDENTICAL to the old one on the next forward pass -- it contributes
nothing until training adapts to use it.
"""
import sys
import torch

src, dst = sys.argv[1], sys.argv[2]
sd = torch.load(src, map_location="cpu")

for head in ("value_head", "reward_approx_head"):
    key = f"{head}.mlp.0.weight"
    old_w = sd[key]  # [hidden_dim, hidden_dim+1]
    out_dim, in_dim = old_w.shape
    new_w = torch.zeros(out_dim, in_dim + 1)
    new_w[:, :in_dim] = old_w
    sd[key] = new_w

torch.save(sd, dst)
print(f"migrated {src} -> {dst}: value_head/reward_approx_head first-layer "
      f"in_features {in_dim} -> {in_dim + 1}")

"""
rl/ppo.py's ppo_update batches every transition across all episodes into a
handful of forward calls per epoch, instead of one Python-level call per
transition -- measured directly, that per-transition loop was ~90% of a
training iteration's wall time (5.98s of 6.74s on a 97-block instance),
dwarfing episode collection itself. See algorithm.md's "Reward" section.

This is a straight numerical-equivalence regression test: `_ppo_update_reference`
below is a verbatim copy of the original, pre-batching per-transition-loop
implementation (kept only here, not in rl/ppo.py, purely as a frozen oracle).
If a future change to ppo_update's loss formula ever silently diverges from
this, this test catches it -- correctness here is not visually obvious from
reading the batched code, since it reshapes/stacks/gathers across episodes
and transition kinds rather than computing one transition's loss at a time.
"""

import torch
import torch.nn.functional as F

from rl.data import synthetic_instance
from rl.networks import ActorCritic, masked_log_softmax
from rl.ppo import _temp_scale, collect_batch, ppo_update


def _ppo_update_reference(net, optimizer, episodes, clip_eps=0.2, value_coef=0.5,
                           reward_approx_coef=0.5, entropy_coef=0.01, epochs=4,
                           max_grad_norm=1.0):
    last_stats = {}
    for _ in range(epochs):
        optimizer.zero_grad()
        policy_loss_sum = torch.tensor(0.0)
        value_loss_sum = torch.tensor(0.0)
        reward_loss_sum = torch.tensor(0.0)
        entropy_sum = torch.tensor(0.0)
        count = 0

        for ep in episodes:
            block_emb, global_emb = net.encode(ep.block_feats, ep.pin_feats, ep.b2b_edges, ep.p2b_edges)
            reward_t = torch.tensor(ep.reward, dtype=torch.float32)

            reward_so_far = 0.0
            for tr in ep.transitions:
                progress_t = torch.tensor(tr.progress, dtype=torch.float32)
                return_t = torch.tensor(tr.return_to_go, dtype=torch.float32)
                reward_so_far_t = torch.tensor(reward_so_far, dtype=torch.float32)
                value = net.value(global_emb, progress_t, reward_so_far_t)
                reward_pred = net.reward_approx(global_emb, progress_t, reward_so_far_t)
                advantage = (return_t - value).detach()

                if tr.kind == 'aspect':
                    logits = net.aspect_logits(block_emb, global_emb, tr.block_idx, progress_t)
                    log_probs = F.log_softmax(_temp_scale(logits, tr.temperature), dim=0)
                    new_log_prob = log_probs[tr.action]
                    probs = log_probs.exp()
                    entropy = -(probs * log_probs).sum()
                else:
                    full_logits = net.position_logits(tr.occupancy, tr.cluster_grid, tr.wiremask,
                                                        block_emb, global_emb, tr.block_idx, progress_t)
                    out_h, out_w = tr.mask.shape
                    cropped = _temp_scale(full_logits[:out_h, :out_w], tr.temperature)
                    log_probs = masked_log_softmax(cropped, tr.mask).reshape(-1)
                    new_log_prob = log_probs[tr.action]
                    flat_mask = tr.mask.reshape(-1)
                    probs = log_probs.exp()
                    entropy = -(probs[flat_mask] * log_probs[flat_mask]).sum()

                ratio = torch.exp(new_log_prob - tr.old_log_prob)
                surr1 = ratio * advantage
                surr2 = torch.clamp(ratio, 1 - clip_eps, 1 + clip_eps) * advantage
                policy_loss_sum = policy_loss_sum - torch.min(surr1, surr2)
                value_loss_sum = value_loss_sum + (value - return_t) ** 2
                reward_loss_sum = reward_loss_sum + (reward_pred - reward_t) ** 2
                entropy_sum = entropy_sum + entropy
                count += 1
                reward_so_far += tr.step_reward

        count = max(count, 1)
        loss = (policy_loss_sum + value_coef * value_loss_sum
                + reward_approx_coef * reward_loss_sum
                - entropy_coef * entropy_sum) / count
        loss.backward()
        grad_norm = torch.nn.utils.clip_grad_norm_(net.parameters(), max_grad_norm)
        optimizer.step()
        last_stats = {
            'loss': loss.item(),
            'policy_loss': (policy_loss_sum / count).item(),
            'value_loss': (value_loss_sum / count).item(),
            'reward_approx_loss': (reward_loss_sum / count).item(),
            'entropy': (entropy_sum / count).item(),
            'grad_norm': float(grad_norm),
        }
    return last_stats


def _make_instance(n):
    edges = [(i, i + 1, 1.0) for i in range(n - 1)] + [(0, n - 1, 0.7)]
    return synthetic_instance(area_targets=[4.0] * n, b2b_edges=edges)


def test_batched_update_matches_reference_loop():
    # A handful of seeds/instance sizes -- enough to exercise both
    # aspect-only-heavy small instances and larger ones with many position
    # transitions and multiple mask-shape buckets.
    for seed in range(4):
        for n in [4, 9, 15]:
            inst = _make_instance(n)

            torch.manual_seed(seed)
            net_ref = ActorCritic(hidden_dim=16, num_gnn_layers=2, num_aspects=9, cnn_channels=8)
            opt_ref = torch.optim.Adam(net_ref.parameters(), lr=3e-3)
            torch.manual_seed(seed)
            net_new = ActorCritic(hidden_dim=16, num_gnn_layers=2, num_aspects=9, cnn_channels=8)
            opt_new = torch.optim.Adam(net_new.parameters(), lr=3e-3)

            torch.manual_seed(1000 + seed)
            episodes = collect_batch(net_ref, inst, grid_dim=12, use_baseline=False, num_episodes=6)

            stats_ref = _ppo_update_reference(net_ref, opt_ref, episodes, epochs=1)
            stats_new = ppo_update(net_new, opt_new, episodes, epochs=1)

            for key in stats_ref:
                a, b = stats_ref[key], stats_new[key]
                assert abs(a - b) <= 1e-3 * max(1.0, abs(a)), (
                    f"seed={seed} n={n} stat={key}: reference={a!r} batched={b!r}"
                )

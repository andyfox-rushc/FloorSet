"""
Tests for the variance-reduction additions to rl/ppo.py + rl/finetune.py:
temperature-scaled sampling (annealed during fine-tuning) and a final
deterministic (argmax) rollout. The critical correctness property is that
the stored per-transition temperature keeps the PPO importance ratio valid
-- collection and update must use the exact same tempered distribution.
"""

import torch

from rl.data import synthetic_instance
from rl.encoder import build_block_features, build_pin_features
from rl.networks import ActorCritic
from rl.ppo import collect_episode, ppo_update


def make_instance():
    return synthetic_instance(
        area_targets=[4.0, 9.0, 16.0],
        b2b_edges=[(0, 1, 1.0), (1, 2, 1.0)],
    )


def test_greedy_rollout_is_deterministic():
    torch.manual_seed(0)
    inst = make_instance()
    net = ActorCritic(hidden_dim=16, num_gnn_layers=2, cnn_channels=8)

    ep1 = collect_episode(net, inst, grid_dim=10, use_baseline=False, greedy=True)
    ep2 = collect_episode(net, inst, grid_dim=10, use_baseline=False, greedy=True)

    assert ep1.positions == ep2.positions
    assert ep1.reward == ep2.reward


def test_low_temperature_concentrates_a_single_distribution_on_its_mode():
    # Isolate one distribution (no multi-step compounding across an episode,
    # which would make even a high per-step match rate look like ~0 matches
    # over a full trajectory) and confirm sampling at low temperature picks
    # its argmax with high frequency.
    import torch.nn.functional as F

    from rl.ppo import _temp_scale

    torch.manual_seed(0)
    inst = make_instance()
    net = ActorCritic(hidden_dim=16, num_gnn_layers=2, cnn_channels=8)
    block_feats = build_block_features(inst, 5.0)
    pin_feats = build_pin_features(inst, 5.0)
    block_emb, global_emb = net.encode(block_feats, pin_feats, inst.b2b_connectivity, inst.p2b_connectivity)
    logits = net.aspect_logits(block_emb, global_emb, block_idx=0, progress=torch.tensor(0.0))
    mode = int(torch.argmax(logits).item())

    def match_rate(temperature):
        matches = 0
        trials = 200
        for seed in range(trials):
            torch.manual_seed(1000 + seed)
            log_probs = F.log_softmax(_temp_scale(logits, temperature), dim=0)
            action = torch.multinomial(log_probs.exp(), 1).item()
            matches += int(action == mode)
        return matches / trials

    # Compare against temperature=1.0 rather than asserting an absolute
    # threshold -- the exact concentration depends on this network's
    # (untrained, random) logit spread, but low temperature should always
    # concentrate *more* than no scaling at all.
    assert match_rate(0.05) > match_rate(1.0) + 0.2


def test_ppo_ratio_is_identity_before_any_update():
    # With no optimizer step taken yet, recomputing log-probs at update time
    # (using each transition's stored temperature) must reproduce the exact
    # same log-prob used at collection time -- i.e. ratio == 1 for every
    # transition. This is the property that broke if temperature weren't
    # applied identically on both sides.
    torch.manual_seed(0)
    inst = make_instance()
    net = ActorCritic(hidden_dim=16, num_gnn_layers=2, cnn_channels=8)
    ep = collect_episode(net, inst, grid_dim=10, use_baseline=False, temperature=0.4)
    assert ep.transitions, "expected at least one transition"

    from rl.networks import masked_log_softmax
    import torch.nn.functional as F
    from rl.ppo import _temp_scale

    block_emb, global_emb = net.encode(ep.block_feats, ep.pin_feats, ep.b2b_edges, ep.p2b_edges)
    for tr in ep.transitions:
        progress_t = torch.tensor(tr.progress, dtype=torch.float32)
        if tr.kind == 'aspect':
            logits = net.aspect_logits(block_emb, global_emb, tr.block_idx, progress_t)
            log_probs = F.log_softmax(_temp_scale(logits, tr.temperature), dim=0)
            new_log_prob = log_probs[tr.action].item()
        else:
            full_logits = net.position_logits(tr.occupancy, tr.cluster_grid, block_emb,
                                               global_emb, tr.block_idx, progress_t)
            out_h, out_w = tr.mask.shape
            cropped = _temp_scale(full_logits[:out_h, :out_w], tr.temperature)
            log_probs = masked_log_softmax(cropped, tr.mask).reshape(-1)
            new_log_prob = log_probs[tr.action].item()
        assert abs(new_log_prob - tr.old_log_prob) < 1e-5


def test_ppo_update_runs_with_annealed_temperature_batch():
    torch.manual_seed(0)
    inst = make_instance()
    net = ActorCritic(hidden_dim=16, num_gnn_layers=2, cnn_channels=8)
    optimizer = torch.optim.Adam(net.parameters(), lr=1e-3)

    episodes = [
        collect_episode(net, inst, grid_dim=10, use_baseline=False, temperature=t)
        for t in (1.0, 0.6, 0.3)
    ]
    stats = ppo_update(net, optimizer, episodes, epochs=2)
    assert all(torch.isfinite(torch.tensor(v)) for v in stats.values())

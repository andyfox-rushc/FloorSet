import pytest
import torch

from rl.data import synthetic_instance
from rl.encoder import build_block_features, build_pin_features
from rl.env import GridPlacementEnv
from rl.networks import ActorCritic, masked_log_softmax


def make_actor_critic(hidden_dim=16):
    return ActorCritic(hidden_dim=hidden_dim, num_gnn_layers=2, num_aspects=9, cnn_channels=8)


def test_aspect_logits_shape():
    inst = synthetic_instance(area_targets=[4.0, 9.0, 16.0], b2b_edges=[(0, 1, 1.0)])
    net = make_actor_critic()
    block_feats = build_block_features(inst, 5.0)
    pin_feats = build_pin_features(inst, 5.0)
    block_emb, global_emb = net.encode(block_feats, pin_feats, inst.b2b_connectivity, inst.p2b_connectivity)
    logits = net.aspect_logits(block_emb, global_emb, block_idx=0, progress=torch.tensor(0.0))
    assert logits.shape == (9,)
    assert torch.isfinite(logits).all()


def test_position_logits_shape_matches_grid():
    inst = synthetic_instance(area_targets=[4.0, 9.0])
    net = make_actor_critic()
    block_feats = build_block_features(inst, 5.0)
    pin_feats = build_pin_features(inst, 5.0)
    block_emb, global_emb = net.encode(block_feats, pin_feats, inst.b2b_connectivity, inst.p2b_connectivity)

    env = GridPlacementEnv(inst, grid_dim=12)
    occupancy = env.occupancy
    cluster_grid = torch.zeros_like(occupancy)
    wiremask = env.wiremask(1.0, 1.0)
    logits = net.position_logits(occupancy, cluster_grid, wiremask, block_emb, global_emb,
                                  block_idx=0, progress=torch.tensor(0.0))
    assert logits.shape == (12, 12)
    assert torch.isfinite(logits).all()


def test_masked_log_softmax_never_selects_invalid_action():
    torch.manual_seed(0)
    for _ in range(200):
        logits = torch.randn(6, 6)
        mask = torch.rand(6, 6) > 0.5
        if not mask.any():
            mask[0, 0] = True
        log_probs = masked_log_softmax(logits, mask)
        probs = log_probs.exp()
        assert torch.isfinite(probs).all()
        assert (probs[~mask] < 1e-6).all()
        flat = probs.reshape(-1)
        idx = torch.multinomial(flat, 1).item()
        gy, gx = divmod(idx, mask.shape[1])
        assert mask[gy, gx], "sampled an action outside the mask"


def test_gradient_flows_end_to_end_through_position_head():
    inst = synthetic_instance(area_targets=[4.0, 9.0])
    net = make_actor_critic()
    block_feats = build_block_features(inst, 5.0)
    pin_feats = build_pin_features(inst, 5.0)
    block_emb, global_emb = net.encode(block_feats, pin_feats, inst.b2b_connectivity, inst.p2b_connectivity)

    env = GridPlacementEnv(inst, grid_dim=8)
    cluster_grid = torch.zeros_like(env.occupancy)
    wiremask = env.wiremask(1.0, 1.0)
    logits = net.position_logits(env.occupancy, cluster_grid, wiremask, block_emb, global_emb,
                                  block_idx=0, progress=torch.tensor(0.5))
    mask = torch.ones_like(logits, dtype=torch.bool)
    log_probs = masked_log_softmax(logits, mask)
    loss = -log_probs.sum()
    loss.backward()
    grad = net.encoder.block_embed.weight.grad
    assert grad is not None and torch.isfinite(grad).all()


def test_value_and_reward_approx_are_scalars():
    inst = synthetic_instance(area_targets=[4.0, 9.0])
    net = make_actor_critic()
    block_feats = build_block_features(inst, 5.0)
    pin_feats = build_pin_features(inst, 5.0)
    _, global_emb = net.encode(block_feats, pin_feats, inst.b2b_connectivity, inst.p2b_connectivity)
    v = net.value(global_emb, torch.tensor(0.3), torch.tensor(-0.5))
    r = net.reward_approx(global_emb, torch.tensor(0.3), torch.tensor(-0.5))
    assert v.shape == ()
    assert r.shape == ()


def test_step_features_shape_and_range():
    from rl.data import synthetic_instance
    from rl.env import GridPlacementEnv
    from rl.networks import NUM_STEP_FEATS
    inst = synthetic_instance(area_targets=[4.0, 9.0, 6.0])
    env = GridPlacementEnv(inst, grid_dim=16, fine_factor=4)
    f = env.step_features()
    assert f.shape == (NUM_STEP_FEATS,)
    assert torch.isfinite(f).all() and (f >= 0).all()
    assert f[5] == pytest.approx(2 / 3)          # two of three steps still to come


def test_old_checkpoint_loads_and_ignores_features(tmp_path):
    from rl.networks import NUM_STEP_FEATS, _STEP_FEAT_KEYS, load_state
    net = ActorCritic(hidden_dim=64)
    sd = net.state_dict()
    old = {k: (v[:, :-NUM_STEP_FEATS] if k in _STEP_FEAT_KEYS else v) for k, v in sd.items()}
    path = tmp_path / "old.pt"
    torch.save(old, path)
    new = load_state(ActorCritic(hidden_dim=64), str(path))
    be, ge = torch.randn(64), torch.randn(64)
    plain = torch.tensor(0.4)
    with_feats = torch.cat([torch.tensor([0.4]), torch.rand(NUM_STEP_FEATS)])
    assert torch.allclose(new.aspect_head(be, ge, plain), new.aspect_head(be, ge, with_feats))

import torch

from rl.data import synthetic_instance
from rl.encoder import NetlistEncoder, build_block_features, build_pin_features


def make_inst():
    return synthetic_instance(
        area_targets=[4.0, 9.0, 16.0, 6.0],
        constraints=[
            [0, 0, 0, 0, 0],
            [0, 0, 1, 0, 0],
            [0, 0, 1, 0, 0],
            [1, 0, 0, 0, 0],
        ],
        target_positions=[
            [-1, -1, -1, -1], [-1, -1, -1, -1], [-1, -1, -1, -1], [-1, -1, 3.0, 2.0],
        ],
        b2b_edges=[(0, 1, 1.0), (1, 2, 0.5), (2, 3, 2.0)],
        p2b_edges=[(0, 0, 1.0), (1, 2, 0.5)],
        pins_pos=[(1.0, 2.0), (3.0, 4.0)],
    )


def test_shapes():
    inst = make_inst()
    enc = NetlistEncoder(hidden_dim=16, num_layers=2)
    block_feats = build_block_features(inst, canvas_scale=10.0)
    pin_feats = build_pin_features(inst, canvas_scale=10.0)
    assert block_feats.shape == (4, 9)
    assert pin_feats.shape == (2, 2)

    h_block, h_pin = enc(block_feats, pin_feats, inst.b2b_connectivity, inst.p2b_connectivity)
    assert h_block.shape == (4, 16)
    assert h_pin.shape == (2, 16)


def test_handles_no_pins_and_no_edges():
    inst = synthetic_instance(area_targets=[4.0, 9.0])
    enc = NetlistEncoder(hidden_dim=8, num_layers=2)
    block_feats = build_block_features(inst, canvas_scale=5.0)
    pin_feats = build_pin_features(inst, canvas_scale=5.0)
    h_block, h_pin = enc(block_feats, pin_feats, inst.b2b_connectivity, inst.p2b_connectivity)
    assert h_block.shape == (2, 8)
    assert h_pin.shape == (0, 8)
    assert torch.isfinite(h_block).all()


def test_gradient_flows_through_encoder():
    inst = make_inst()
    enc = NetlistEncoder(hidden_dim=16, num_layers=2)
    block_feats = build_block_features(inst, canvas_scale=10.0)
    pin_feats = build_pin_features(inst, canvas_scale=10.0)
    h_block, h_pin = enc(block_feats, pin_feats, inst.b2b_connectivity, inst.p2b_connectivity)
    loss = h_block.sum() + h_pin.sum()
    loss.backward()
    grad = enc.block_embed.weight.grad
    assert grad is not None
    assert torch.isfinite(grad).all()
    assert grad.abs().sum() > 0


def test_edge_weight_changes_embedding():
    inst_a = synthetic_instance(area_targets=[4.0, 9.0], b2b_edges=[(0, 1, 1.0)])
    inst_b = synthetic_instance(area_targets=[4.0, 9.0], b2b_edges=[(0, 1, 5.0)])
    torch.manual_seed(0)
    enc = NetlistEncoder(hidden_dim=8, num_layers=2)
    fa = build_block_features(inst_a, 5.0)
    fb = build_block_features(inst_b, 5.0)
    pa = build_pin_features(inst_a, 5.0)
    pb = build_pin_features(inst_b, 5.0)
    ha, _ = enc(fa, pa, inst_a.b2b_connectivity, inst_a.p2b_connectivity)
    hb, _ = enc(fb, pb, inst_b.b2b_connectivity, inst_b.p2b_connectivity)
    assert not torch.allclose(ha, hb)

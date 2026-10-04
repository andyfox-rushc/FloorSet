"""
Prototype options on GridPlacementEnv: pin_canvas (canvas = pin box) and
fine_factor > 1 (occupancy tracked on a finer grid than the action grid).
The hard-constraint guarantees must hold exactly as on the coarse path.
"""

import pytest
import torch

from iccad2026_evaluate import ContestEvaluator, check_overlap
from rl.data import from_validation_sample, synthetic_instance
from rl.env import GridPlacementEnv


@pytest.fixture(scope="module")
def evaluator():
    ev = ContestEvaluator(data_path="../", verbose=False)
    ev._load_dataset()
    return ev


def test_pin_canvas_is_the_pin_max_box(evaluator):
    inst = from_validation_sample(evaluator, 0)
    env = GridPlacementEnv(inst, grid_dim=48, pin_canvas=True)
    valid = inst.pins_pos[(inst.pins_pos[:, 0] != -1) | (inst.pins_pos[:, 1] != -1)]
    assert env.x_min == 0.0 and env.y_min == 0.0
    assert env.x_max == pytest.approx(float(valid[:, 0].max()))
    assert env.y_max == pytest.approx(float(valid[:, 1].max()))


def test_fine_observation_is_coarse_occupied_fraction():
    inst = synthetic_instance(area_targets=[4.0, 4.0])
    env = GridPlacementEnv(inst, grid_dim=10, fine_factor=4)
    assert env.occupancy.shape == (10, 10)
    assert env.fine_occ.shape == (40, 40)
    w, h = env.choose_aspect(4)
    env.position_mask(w, h)
    env.place(gy=0, gx=0)
    assert float(env.occupancy.max()) <= 1.0
    assert float(env.occupancy.sum()) == pytest.approx(float(env.fine_occ.sum()) / 16)


@pytest.mark.parametrize("code, edge", [(1, "left"), (2, "right"), (4, "top"), (8, "bottom")])
def test_fine_boundary_block_touches_required_edge(rollout, code, edge):
    inst = synthetic_instance(
        area_targets=[9.0, 4.0, 4.0],
        constraints=[[0, 0, 0, 0, code], [0, 0, 0, 0, 0], [0, 0, 0, 0, 0]],
    )
    env = GridPlacementEnv(inst, grid_dim=16, fine_factor=4)
    positions = rollout(env, seed=0)
    x, y, w, h = positions[0]
    expected = {"left": (x, env.x_min), "right": (x + w, env.x_max),
                "top": (y + h, env.y_max), "bottom": (y, env.y_min)}[edge]
    assert expected[0] == pytest.approx(expected[1], abs=1e-6)


def test_aspect_mask_rejects_shapes_that_cannot_fit():
    inst = synthetic_instance(area_targets=[4.0])
    env = GridPlacementEnv(inst, grid_dim=8, fine_factor=4)
    env.x_max, env.y_max = env.x_min + 3.0, env.y_min + 3.0
    env.cell_w = env.cell_h = 3.0 / 8
    env.fine_cell_w = env.fine_cell_h = 3.0 / 32
    mask = env.aspect_mask()
    square = len(env.aspect_ratios) // 2
    assert bool(mask[square])       # 2x2 fits in 3x3
    assert not bool(mask[0])        # aspect 0.2: ~0.89 x 4.47 doesn't
    assert not bool(mask[-1])       # aspect 5.0: ~4.47 x 0.89 doesn't


@pytest.mark.parametrize("test_id", range(0, 100, 5))
def test_fine_random_rollout_respects_hard_constraints_on_validation_set(evaluator, rollout, test_id):
    inst = from_validation_sample(evaluator, test_id)
    env = GridPlacementEnv(inst, grid_dim=48, fine_factor=8)
    positions = rollout(env, seed=0, max_seed_tries=30)
    assert check_overlap(positions) == 0

    ncols = inst.constraints.shape[1]
    for i in range(inst.block_count):
        x, y, w, h = positions[i]
        assert env.x_min - 1e-6 <= x and x + w <= env.x_max + 1e-6
        assert env.y_min - 1e-6 <= y and y + h <= env.y_max + 1e-6
        is_fixed = ncols > 0 and inst.constraints[i, 0] != 0
        is_preplaced = ncols > 1 and inst.constraints[i, 1] != 0
        if is_fixed or is_preplaced:
            tx, ty, tw, th = inst.target_positions[i].tolist()
            assert w == pytest.approx(tw, abs=1e-4)
            assert h == pytest.approx(th, abs=1e-4)
            if is_preplaced:
                assert x == pytest.approx(tx, abs=1e-4)
                assert y == pytest.approx(ty, abs=1e-4)
        elif float(inst.area_targets[i]) > 0:
            assert w * h == pytest.approx(float(inst.area_targets[i]), rel=1e-4)


def test_policy_aspect_choices_respect_aspect_mask():
    from rl.networks import ActorCritic
    from rl.ppo import collect_episode, ppo_update
    inst = synthetic_instance(area_targets=[4.0, 9.0, 6.0, 5.0],
                              b2b_edges=[(0, 1, 1.0), (1, 2, 2.0), (2, 3, 1.0)])
    net = ActorCritic(hidden_dim=16, num_gnn_layers=2, cnn_channels=8)
    optimizer = torch.optim.Adam(net.parameters(), lr=1e-3)
    episodes = []
    for seed in range(4):
        torch.manual_seed(seed)
        ep = collect_episode(net, inst, grid_dim=16, use_baseline=False,
                             env_kwargs=dict(fine_factor=4))
        for tr in ep.transitions:
            if tr.kind == 'aspect':
                assert tr.mask is not None and bool(tr.mask[tr.action])
        episodes.append(ep)
    stats = ppo_update(net, optimizer, episodes, epochs=1)
    assert all(torch.isfinite(torch.tensor(v)) for v in stats.values())


@pytest.mark.parametrize("code, edge", [(2, "right"), (4, "top"), (6, "top-right")])
def test_die_anchor_pins_right_top_blocks_to_pin_box(rollout, code, edge):
    inst = synthetic_instance(
        area_targets=[9.0, 4.0, 4.0],
        constraints=[[0, 0, 0, 0, code], [0, 0, 0, 0, 0], [0, 0, 0, 0, 0]],
    )
    inst.pins_pos = torch.tensor([[0.0, 0.0], [6.0, 5.0]])
    env = GridPlacementEnv(inst, grid_dim=16, fine_factor=4, utilization=0.3, die_anchor=True)
    assert env.anchor_x_max == pytest.approx(6.0) and env.anchor_y_max == pytest.approx(5.0)
    assert env.x_max > 6.0 and env.y_max > 5.0   # canvas extends past the die
    x, y, w, h = rollout(env, seed=0)[0]
    if code & 2:
        assert x + w == pytest.approx(6.0, abs=1e-6)
    if code & 4:
        assert y + h == pytest.approx(5.0, abs=1e-6)


def test_die_anchor_keeps_free_blocks_inside_die_when_they_fit(rollout):
    inst = synthetic_instance(area_targets=[1.0, 1.0, 1.0, 1.0])
    inst.pins_pos = torch.tensor([[0.0, 0.0], [4.0, 4.0]])
    env = GridPlacementEnv(inst, grid_dim=16, fine_factor=4, utilization=0.05, die_anchor=True)
    for seed in range(5):
        env.reset()
        for x, y, w, h in rollout(env, seed=seed):
            assert x + w <= 4.0 + 1e-6 and y + h <= 4.0 + 1e-6


@pytest.mark.parametrize("test_id", range(0, 100, 10))
def test_die_anchor_rollout_respects_hard_constraints(evaluator, rollout, test_id):
    inst = from_validation_sample(evaluator, test_id)
    env = GridPlacementEnv(inst, grid_dim=48, fine_factor=4, die_anchor=True)
    positions = rollout(env, seed=0, max_seed_tries=30)
    assert check_overlap(positions) == 0
    for x, y, w, h in positions:
        assert env.x_min - 1e-6 <= x and x + w <= env.x_max + 1e-6
        assert env.y_min - 1e-6 <= y and y + h <= env.y_max + 1e-6


def _open_canvas(env, size, fine):
    env.x_max = env.y_max = env.anchor_x_max = env.anchor_y_max = float(size)
    env.cell_w = env.cell_h = size / env.grid_dim
    env.fine_cell_w = env.fine_cell_h = size / env.grid_dim / fine


def test_shape_fit_aligns_with_neighbour_edge_and_keeps_area():
    # Block 0 is 2.5 x 2 at the origin. Block 1 (area 4, chosen 2x2) placed
    # on top of it should widen to 2.5 so its right edge lines up.
    inst = synthetic_instance(area_targets=[5.0, 4.0])
    env = GridPlacementEnv(inst, grid_dim=16, fine_factor=4, shape_fit=True)
    _open_canvas(env, 16, 4)
    env.choose_aspect(len(env.aspect_ratios) // 2)
    env._pending_shape = (2.5, 2.0)
    env.position_mask(2.5, 2.0)
    env.place(gy=0, gx=0)
    w, h = env.choose_aspect(len(env.aspect_ratios) // 2)
    env.position_mask(w, h)
    env.place(gy=2, gx=0)
    x, y, w1, h1 = env.positions[1]
    assert (x, y) == pytest.approx((0.0, 2.0))
    assert w1 == pytest.approx(2.5) and w1 * h1 == pytest.approx(4.0)


def test_shape_fit_rollouts_stay_legal(evaluator, rollout):
    for test_id in (0, 49):
        inst = from_validation_sample(evaluator, test_id)
        env = GridPlacementEnv(inst, grid_dim=48, fine_factor=4, shape_fit=True, lookahead=3)
        positions = rollout(env, seed=0, max_seed_tries=30)
        assert check_overlap(positions) == 0
        for i, (x, y, w, h) in enumerate(positions):
            assert env.x_min - 1e-6 <= x and x + w <= env.x_max + 1e-6
            assert env.y_min - 1e-6 <= y and y + h <= env.y_max + 1e-6
            if float(inst.area_targets[i]) > 0 and not inst.constraints[i, 0] and not inst.constraints[i, 1]:
                assert w * h == pytest.approx(float(inst.area_targets[i]), rel=1e-3)


def test_lookahead_rejects_cell_that_strands_a_big_block():
    # 10x10 canvas, one 2x2 block to place, and a 6x6 block still to come:
    # the centre leaves no 6x6 hole, a corner does.
    inst = synthetic_instance(area_targets=[4.0])
    env = GridPlacementEnv(inst, grid_dim=10, fine_factor=2, lookahead=1)
    _open_canvas(env, 10, 2)
    env._upcoming_shapes = lambda: [[(6.0, 6.0)]]
    w, h = env.choose_aspect(len(env.aspect_ratios) // 2)
    env.position_mask(w, h)
    assert not env.lookahead_ok(4, 4)
    assert env.lookahead_ok(0, 0)

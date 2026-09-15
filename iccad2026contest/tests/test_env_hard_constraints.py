import math

import pytest
import torch
from shapely.geometry import box
from shapely.ops import unary_union

from rl.data import synthetic_instance
from rl.env import GridPlacementEnv


def bbox_of(positions):
    x_min = min(p[0] for p in positions)
    y_min = min(p[1] for p in positions)
    x_max = max(p[0] + p[2] for p in positions)
    y_max = max(p[1] + p[3] for p in positions)
    return x_min, y_min, x_max, y_max


def check_overlap(positions):
    n = len(positions)
    for i in range(n):
        for j in range(i + 1, n):
            x1, y1, w1, h1 = positions[i]
            x2, y2, w2, h2 = positions[j]
            ox = min(x1 + w1, x2 + w2) - max(x1, x2)
            oy = min(y1 + h1, y2 + h2) - max(y1, y2)
            if ox > 1e-6 and oy > 1e-6:
                return False
    return True


def test_random_rollout_is_overlap_free(rollout):
    areas = [4.0, 9.0, 16.0, 6.0, 12.0, 3.0, 8.0]
    inst = synthetic_instance(area_targets=areas)
    for seed in range(10):
        env = GridPlacementEnv(inst, grid_dim=24)
        positions = rollout(env, seed=seed)
        assert len(positions) == len(areas)
        assert check_overlap(positions), f"overlap at seed={seed}"


def test_free_block_area_matches_target_exactly(rollout):
    areas = [4.0, 25.0, 100.0]
    inst = synthetic_instance(area_targets=areas)
    env = GridPlacementEnv(inst, grid_dim=24)
    positions = rollout(env, seed=1)
    for (x, y, w, h), a in zip(positions, areas):
        assert w * h == pytest.approx(a, rel=1e-6)


def test_fixed_shape_block_keeps_exact_dimensions(rollout):
    areas = [10.0, 6.0, 4.0]
    constraints = [
        [1, 0, 0, 0, 0],
        [0, 0, 0, 0, 0],
        [0, 0, 0, 0, 0],
    ]
    target_positions = [
        [-1, -1, 3.0, 2.0],  # fixed-shape: w=3, h=2 (area != 10, that's fine/expected)
        [-1, -1, -1, -1],
        [-1, -1, -1, -1],
    ]
    inst = synthetic_instance(areas, constraints=constraints, target_positions=target_positions)
    env = GridPlacementEnv(inst, grid_dim=24)
    positions = rollout(env, seed=2)
    x, y, w, h = positions[0]
    assert (w, h) == (3.0, 2.0)


def test_preplaced_block_keeps_exact_position_and_others_avoid_it(rollout):
    areas = [8.0, 5.0, 6.0]
    constraints = [
        [0, 1, 0, 0, 0],
        [0, 0, 0, 0, 0],
        [0, 0, 0, 0, 0],
    ]
    target_positions = [
        [5.0, 5.0, 4.0, 2.0],
        [-1, -1, -1, -1],
        [-1, -1, -1, -1],
    ]
    inst = synthetic_instance(areas, constraints=constraints, target_positions=target_positions)
    for seed in range(5):
        env = GridPlacementEnv(inst, grid_dim=24)
        positions = rollout(env, seed=seed)
        assert positions[0] == (5.0, 5.0, 4.0, 2.0)
        assert check_overlap(positions)


def test_mib_group_shares_identical_shape_when_areas_match(rollout):
    # Genuine "multi-instantiated" case: same area target for every member,
    # so an exact shape copy is both possible and correct.
    areas = [20.0, 20.0, 20.0]
    constraints = [
        [0, 0, 1, 0, 0],
        [0, 0, 1, 0, 0],
        [0, 0, 1, 0, 0],
    ]
    inst = synthetic_instance(areas, constraints=constraints)
    positions = rollout(GridPlacementEnv(inst, grid_dim=24), seed=3)
    shapes = {(round(p[2], 6), round(p[3], 6)) for p in positions}
    assert len(shapes) == 1, f"expected identical MIB shapes, got {shapes}"


def test_mib_group_with_differing_areas_keeps_hard_area_constraint(rollout):
    # Real data has MIB groups whose members carry *different* area targets
    # (confirmed against the actual training set); exact shape-copying would
    # then break a follower's hard area-tolerance constraint, which must
    # never happen -- each member's own area always wins over MIB shape
    # uniformity when the two are incompatible.
    areas = [12.0, 30.0, 20.0]
    constraints = [
        [0, 0, 1, 0, 0],
        [0, 0, 1, 0, 0],
        [0, 0, 1, 0, 0],
    ]
    inst = synthetic_instance(areas, constraints=constraints)
    positions = rollout(GridPlacementEnv(inst, grid_dim=24), seed=3)
    for (x, y, w, h), area in zip(positions, areas):
        assert w * h == pytest.approx(area, rel=1e-6)


@pytest.mark.parametrize("code,edge", [
    (1, 'left'), (2, 'right'), (4, 'top'), (8, 'bottom'),
])
def test_boundary_block_touches_required_edge(rollout, code, edge):
    areas = [10.0, 4.0, 6.0, 5.0]
    constraints = [
        [0, 0, 0, 0, code],
        [0, 0, 0, 0, 0],
        [0, 0, 0, 0, 0],
        [0, 0, 0, 0, 0],
    ]
    inst = synthetic_instance(areas, constraints=constraints)
    env = GridPlacementEnv(inst, grid_dim=24)
    positions = rollout(env, seed=4)
    x_min, y_min, x_max, y_max = bbox_of(positions)
    x, y, w, h = positions[0]
    eps = 1e-6
    if edge == 'left':
        assert abs(x - x_min) < eps
    elif edge == 'right':
        assert abs((x + w) - x_max) < eps
    elif edge == 'top':
        assert abs((y + h) - y_max) < eps
    elif edge == 'bottom':
        assert abs(y - y_min) < eps


def test_cluster_members_end_up_touching(rollout):
    # Just the two clustered blocks -- nothing else around to block every
    # side, so an exact geometric touch is always achievable and the
    # deterministic cluster-touch path (env._try_cluster_touch) should find
    # it. (When *other* blocks fill in every side, touching is only
    # best-effort -- see the env.py module docstring.)
    areas = [10.0, 6.0]
    constraints = [
        [0, 0, 0, 1, 0],
        [0, 0, 0, 1, 0],
    ]
    inst = synthetic_instance(areas, constraints=constraints)
    for seed in range(10):
        env = GridPlacementEnv(inst, grid_dim=32)
        positions = rollout(env, seed=seed)
        cluster_polys = [box(positions[i][0], positions[i][1],
                              positions[i][0] + positions[i][2], positions[i][1] + positions[i][3])
                         for i in (0, 1)]
        merged = unary_union(cluster_polys)
        assert merged.geom_type != 'MultiPolygon', f"clustered blocks did not touch (seed={seed})"


def test_no_free_position_raises_clear_error():
    # A single block whose area vastly exceeds the tiny canvas/grid we hand it.
    inst = synthetic_instance(area_targets=[1_000_000.0])
    env = GridPlacementEnv(inst, grid_dim=4)
    # Force an undersized canvas directly to exercise the guard.
    env.x_max = env.x_min + 1.0
    env.y_max = env.y_min + 1.0
    env.cell_w = 1.0 / env.grid_dim
    env.cell_h = 1.0 / env.grid_dim
    env.occupancy[0, 0] = 1.0  # the block's footprint covers the whole grid, so any occupied cell blocks it
    w, h = env.choose_aspect(4)  # square
    with pytest.raises(RuntimeError):
        env.position_mask(w, h)


def test_wiremask_is_zero_with_no_placed_neighbors():
    # First block in the order has no already-placed connected block/pin to
    # pull toward -- the channel should be a flat zero, not garbage.
    inst = synthetic_instance(area_targets=[4.0, 9.0], b2b_edges=[(0, 1, 1.0)])
    env = GridPlacementEnv(inst, grid_dim=10)
    w, h = env.choose_aspect(4)
    wm = env.wiremask(w, h)
    assert wm.shape == (10, 10)
    assert torch.allclose(wm, torch.zeros(10, 10))


def test_wiremask_prefers_side_closer_to_placed_neighbor():
    # Block 1 is wired to block 0; once block 0 is placed on the left half
    # of the canvas, the wiremask should rate left-half cells strictly
    # higher (better) than right-half cells for block 1's placement.
    inst = synthetic_instance(area_targets=[4.0, 4.0], b2b_edges=[(0, 1, 1.0)])
    env = GridPlacementEnv(inst, grid_dim=10)
    w0, h0 = env.choose_aspect(4)  # square
    env.position_mask(w0, h0)
    env.place(gy=0, gx=0)  # block 0 pinned to the low-x/low-y corner

    w1, h1 = env.choose_aspect(4)
    wm = env.wiremask(w1, h1)
    left_half = wm[:, :5]
    right_half = wm[:, 5:]
    assert left_half.mean() > right_half.mean()


def test_wiremask_includes_p2b_pins_regardless_of_placement_order():
    # Pins are fixed geometry (not something placed during the rollout), so
    # a block's own connected pins should contribute even on the very first
    # step, unlike b2b neighbors.
    inst = synthetic_instance(
        area_targets=[4.0],
        p2b_edges=[(0, 0, 1.0)],
        pins_pos=[[9.0, 9.0]],
    )
    env = GridPlacementEnv(inst, grid_dim=10)
    env.x_max, env.y_max = 10.0, 10.0
    env.cell_w = env.cell_h = 1.0
    w, h = env.choose_aspect(4)
    wm = env.wiremask(w, h)
    assert not torch.allclose(wm, torch.zeros_like(wm))
    # cell nearest the pin (9, 9) should be rated better than the far corner.
    assert wm[9, 9] > wm[0, 0]


def test_step_deltas_zero_wirelength_for_first_placed_block():
    # No already-placed neighbor exists yet, so nothing can resolve; the
    # bbox always grows from nothing though.
    inst = synthetic_instance(area_targets=[4.0, 4.0], b2b_edges=[(0, 1, 1.0)])
    env = GridPlacementEnv(inst, grid_dim=10)
    w0, h0 = env.choose_aspect(4)
    env.position_mask(w0, h0)
    env.place(gy=0, gx=0)
    delta_wl, delta_area = env.last_step_deltas
    assert delta_wl == pytest.approx(0.0)
    assert delta_area > 0.0


def test_step_deltas_match_manual_wirelength_and_area_for_second_block():
    inst = synthetic_instance(area_targets=[4.0, 4.0], b2b_edges=[(0, 1, 2.0)])
    env = GridPlacementEnv(inst, grid_dim=10)
    env.x_max, env.y_max = 10.0, 10.0
    env.cell_w = env.cell_h = 1.0

    w0, h0 = env.choose_aspect(4)  # square, area 4 -> 2x2
    env.position_mask(w0, h0)
    env.place(gy=0, gx=0)  # block 0 at (0,0,2,2), center (1,1)

    w1, h1 = env.choose_aspect(4)
    env.position_mask(w1, h1)
    env.place(gy=0, gx=4)  # block 1 at (4,0,2,2), center (5,1)

    delta_wl, delta_area = env.last_step_deltas
    expected_wl = 2.0 * (abs(5 - 1) + abs(1 - 1))  # weight * manhattan(centers)
    assert delta_wl == pytest.approx(expected_wl)
    # bbox grew from [0,2]x[0,2] (area 4) to [0,6]x[0,2] (area 12)
    assert delta_area == pytest.approx(8.0)


def test_step_deltas_sum_to_full_rollout_totals():
    # The whole point of _step_deltas: summed over every commit in a full
    # rollout, it must reproduce the exact same wirelength/area the contest
    # evaluator computes from final positions -- see rl/reward.py's
    # step_quality_delta docstring.
    from iccad2026_evaluate import calculate_bbox_area, calculate_hpwl_b2b

    inst = synthetic_instance(
        area_targets=[4.0, 9.0, 6.0],
        b2b_edges=[(0, 1, 1.0), (1, 2, 2.0)],
    )
    env = GridPlacementEnv(inst, grid_dim=16)
    total_wl, total_area_delta = 0.0, 0.0
    while not env.done():
        w, h = env.choose_aspect(4)
        mask, _, _ = env.position_mask(w, h)
        gy, gx = mask.nonzero()[0].tolist()
        env.place(gy=gy, gx=gx)
        dwl, darea = env.last_step_deltas
        total_wl += dwl
        total_area_delta += darea

    positions = env.finalize()
    assert total_wl == pytest.approx(calculate_hpwl_b2b(positions, inst.b2b_connectivity))
    assert total_area_delta == pytest.approx(calculate_bbox_area(positions))

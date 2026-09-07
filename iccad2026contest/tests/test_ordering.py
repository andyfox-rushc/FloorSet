import torch

from rl.ordering import compute_order


def test_preplaced_blocks_come_first_and_take_no_action():
    # block 0: preplaced; block 1, 2: free
    constraints = torch.tensor([
        [0, 1, 0, 0, 0],
        [0, 0, 0, 0, 0],
        [0, 0, 0, 0, 0],
    ], dtype=torch.float32)
    areas = torch.tensor([10.0, 20.0, 5.0])
    plan = compute_order(constraints, areas)
    roles = {s.block_idx: s.role for s in plan.order}
    assert roles[0] == 'preplaced'
    assert plan.order[0].block_idx == 0
    assert roles[1] == 'free'
    assert roles[2] == 'free'


def test_descending_area_order_for_free_blocks():
    constraints = torch.zeros(3, 5)
    areas = torch.tensor([5.0, 50.0, 20.0])
    plan = compute_order(constraints, areas)
    assert plan.block_order() == [1, 2, 0]


def test_fixed_shape_role():
    constraints = torch.tensor([[1, 0, 0, 0, 0]], dtype=torch.float32)
    areas = torch.tensor([10.0])
    plan = compute_order(constraints, areas)
    assert plan.order[0].role == 'fixed'


def test_mib_group_leader_then_followers_contiguous():
    # blocks 0,1,2 share mib group 1; block 3 is unrelated and larger
    constraints = torch.tensor([
        [0, 0, 1, 0, 0],
        [0, 0, 1, 0, 0],
        [0, 0, 1, 0, 0],
        [0, 0, 0, 0, 0],
    ], dtype=torch.float32)
    areas = torch.tensor([10.0, 30.0, 20.0, 5.0])
    plan = compute_order(constraints, areas)
    roles = [s.role for s in plan.order]
    idxs = [s.block_idx for s in plan.order]
    # the mib group (largest-area member first) should be contiguous
    mib_positions = [i for i, b in enumerate(idxs) if b in (0, 1, 2)]
    assert mib_positions == list(range(mib_positions[0], mib_positions[0] + 3))
    leader_pos = mib_positions[0]
    assert idxs[leader_pos] == 1  # block 1 has the largest area (30)
    assert roles[leader_pos] in ('fixed', 'free')
    for p in mib_positions[1:]:
        assert roles[p] == 'mib_follower'
        assert plan.order[p].mib_leader == 1


def test_cluster_members_stay_adjacent_in_order():
    constraints = torch.tensor([
        [0, 0, 0, 7, 0],
        [0, 0, 0, 0, 0],
        [0, 0, 0, 7, 0],
        [0, 0, 0, 0, 0],
    ], dtype=torch.float32)
    areas = torch.tensor([10.0, 100.0, 12.0, 1.0])
    plan = compute_order(constraints, areas)
    idxs = plan.block_order()
    cluster_positions = [i for i, b in enumerate(idxs) if b in (0, 2)]
    assert cluster_positions == list(range(cluster_positions[0], cluster_positions[0] + 2))


def test_fixed_shape_member_of_mib_group_keeps_fixed_role():
    # block 0: fixed-shape AND in mib group 1 (smaller area than block 1);
    # block 1: free, same mib group, larger area.
    constraints = torch.tensor([
        [1, 0, 1, 0, 0],
        [0, 0, 1, 0, 0],
    ], dtype=torch.float32)
    areas = torch.tensor([10.0, 30.0])
    plan = compute_order(constraints, areas)
    roles = {s.block_idx: s.role for s in plan.order}
    assert roles[0] == 'fixed'
    assert roles[1] == 'mib_follower'
    follower_step = next(s for s in plan.order if s.block_idx == 1)
    assert follower_step.mib_leader == 0


def test_two_fixed_members_of_mib_group_both_keep_fixed_role():
    constraints = torch.tensor([
        [1, 0, 1, 0, 0],
        [1, 0, 1, 0, 0],
        [0, 0, 1, 0, 0],
    ], dtype=torch.float32)
    areas = torch.tensor([10.0, 20.0, 30.0])
    plan = compute_order(constraints, areas)
    roles = {s.block_idx: s.role for s in plan.order}
    assert roles[0] == 'fixed'
    assert roles[1] == 'fixed'
    assert roles[2] == 'mib_follower'


def test_connected_blocks_are_ordered_near_each_other():
    # block 0 is largest and unconnected to anything; blocks 1 and 2 are
    # small but strongly connected to each other -- they should end up
    # adjacent in the order even though nothing clusters them.
    constraints = torch.zeros(4, 5)
    areas = torch.tensor([100.0, 5.0, 6.0, 4.0])
    b2b = torch.tensor([[1, 2, 10.0]])
    plan = compute_order(constraints, areas, b2b)
    idxs = plan.block_order()
    pos = {b: i for i, b in enumerate(idxs)}
    assert abs(pos[1] - pos[2]) == 1, f"connected blocks 1,2 not adjacent in {idxs}"


def test_no_connectivity_falls_back_to_descending_area():
    constraints = torch.zeros(3, 5)
    areas = torch.tensor([5.0, 50.0, 20.0])
    plan_no_edges = compute_order(constraints, areas, None)
    plan_empty_edges = compute_order(constraints, areas, torch.zeros(0, 3))
    assert plan_no_edges.block_order() == [1, 2, 0]
    assert plan_empty_edges.block_order() == [1, 2, 0]


def test_connectivity_chain_greedily_extends_placed_component():
    # 0 -- 1 -- 2 chain (0 largest), block 3 unconnected and smallest.
    # Greedy walk should place 0, then 1 (connected to 0), then 2 (connected
    # to 1), leaving unconnected 3 for last.
    constraints = torch.zeros(4, 5)
    areas = torch.tensor([100.0, 10.0, 8.0, 1.0])
    b2b = torch.tensor([[0, 1, 5.0], [1, 2, 5.0]])
    plan = compute_order(constraints, areas, b2b)
    assert plan.block_order() == [0, 1, 2, 3]


def test_covers_every_block_exactly_once():
    constraints = torch.tensor([
        [0, 1, 0, 0, 0],
        [1, 0, 0, 0, 0],
        [0, 0, 2, 0, 0],
        [0, 0, 2, 0, 0],
        [0, 0, 0, 3, 0],
        [0, 0, 0, 3, 0],
    ], dtype=torch.float32)
    areas = torch.tensor([5.0, 6.0, 7.0, 8.0, 9.0, 10.0])
    plan = compute_order(constraints, areas)
    assert sorted(plan.block_order()) == list(range(6))

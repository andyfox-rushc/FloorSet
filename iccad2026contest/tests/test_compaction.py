import torch

from rl.compaction import compact


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


def bbox_area(positions):
    x_min = min(p[0] for p in positions)
    y_min = min(p[1] for p in positions)
    x_max = max(p[0] + p[2] for p in positions)
    y_max = max(p[1] + p[3] for p in positions)
    return (x_max - x_min) * (y_max - y_min)


def test_closes_gap_between_two_isolated_blocks():
    # Two 2x2 blocks with a huge empty gap between them, no constraints.
    positions = [(0.0, 0.0, 2.0, 2.0), (50.0, 0.0, 2.0, 2.0)]
    result = compact(positions, constraints=None)
    assert check_overlap(result)
    assert bbox_area(result) < bbox_area(positions)
    # They should end up touching (adjacent), not still 50 units apart.
    xs = sorted(p[0] for p in result)
    assert xs[1] - xs[0] <= 2.0 + 1e-6


def test_never_introduces_overlap_on_random_scattered_layout():
    torch.manual_seed(0)
    for trial in range(20):
        n = 8
        positions = []
        # Scatter non-overlapping unit-ish blocks far apart on a grid so the
        # starting layout is guaranteed overlap-free.
        for i in range(n):
            gx, gy = i % 4, i // 4
            positions.append((gx * 20.0, gy * 20.0, 3.0, 3.0))
        assert check_overlap(positions), "test fixture itself overlaps"
        result = compact(positions, constraints=None)
        assert check_overlap(result), f"compaction introduced overlap on trial {trial}"
        assert bbox_area(result) <= bbox_area(positions) + 1e-6


def test_preplaced_block_never_moves():
    positions = [(10.0, 10.0, 2.0, 2.0), (50.0, 50.0, 2.0, 2.0)]
    # block 0 is preplaced (constraints col 1)
    constraints = torch.tensor([[0, 1, 0, 0, 0], [0, 0, 0, 0, 0]], dtype=torch.float32)
    result = compact(positions, constraints)
    assert result[0] == positions[0]
    assert check_overlap(result)


def test_clustered_group_with_no_other_content_does_not_move():
    # Degenerate case: the cluster IS the whole layout, so the floor is
    # defined by its own leftmost/bottommost member -- nothing to compact
    # toward, so it correctly stays put.
    positions = [(10.0, 10.0, 2.0, 2.0), (50.0, 50.0, 2.0, 2.0)]
    constraints = torch.tensor([[0, 0, 0, 1, 0], [0, 0, 0, 1, 0]], dtype=torch.float32)
    result = compact(positions, constraints)
    assert result == positions


def test_clustered_group_slides_rigidly_toward_other_content():
    # A free anchor block near the origin, and a two-member cluster (already
    # mutually touching, as rl/env.py's _try_cluster_touch guarantees) sitting
    # far away with a big gap. The whole cluster should slide left as one
    # rigid unit until the first member touches the anchor -- closing the
    # gap while keeping the cluster's internal arrangement byte-for-byte
    # identical (a pure translation).
    positions = [
        (0.0, 0.0, 2.0, 2.0),    # free anchor
        (50.0, 0.0, 2.0, 2.0),   # cluster member 1 (touches member 2)
        (52.0, 0.0, 2.0, 2.0),   # cluster member 2
    ]
    constraints = torch.tensor(
        [[0, 0, 0, 0, 0], [0, 0, 0, 1, 0], [0, 0, 0, 1, 0]], dtype=torch.float32)
    result = compact(positions, constraints)
    assert check_overlap(result)
    # Relative arrangement inside the cluster is exactly preserved.
    assert abs((result[2][0] - result[1][0]) - (positions[2][0] - positions[1][0])) < 1e-6
    # The cluster moved left, closing (most of) the gap against the anchor.
    assert result[1][0] < positions[1][0]
    assert result[1][0] <= positions[0][0] + positions[0][2] + 1e-6


def test_clustered_group_never_overlaps_other_content_while_sliding():
    # The "blocker" is itself a free block, so it also compacts toward the
    # anchor -- the cluster then has to catch up to its NEW position, which
    # takes more than one pass. The only real invariant to check is safety
    # (never overlaps), not a specific final x.
    positions = [
        (0.0, 0.0, 3.0, 3.0),      # free anchor
        (10.0, 0.0, 3.0, 3.0),     # free blocker between anchor and cluster
        (40.0, 0.0, 3.0, 3.0),     # cluster member 1
        (43.0, 0.0, 3.0, 3.0),     # cluster member 2
    ]
    constraints = torch.tensor(
        [[0, 0, 0, 0, 0], [0, 0, 0, 0, 0], [0, 0, 0, 1, 0], [0, 0, 0, 1, 0]],
        dtype=torch.float32)
    result = compact(positions, constraints)
    assert check_overlap(result)
    # Cluster caught up to the blocker's final (also-compacted) position.
    assert abs(result[2][0] - (result[1][0] + result[1][2])) < 1e-6


def test_right_pinned_block_is_pulled_in_to_close_the_gap():
    # Block 1 is pinned to the right edge (code=2), sitting far out from a
    # free block that's already at the left. Unlike a plain immovable
    # treatment, the pinned block itself should be pulled inward to close
    # the gap -- not left stranded -- while remaining flush with the new
    # (now much smaller) bbox's right edge.
    positions = [(0.0, 0.0, 2.0, 2.0), (50.0, 0.0, 2.0, 2.0)]
    constraints = torch.tensor([[0, 0, 0, 0, 0], [0, 0, 0, 0, 2]], dtype=torch.float32)
    result = compact(positions, constraints)
    assert check_overlap(result)
    assert result[1][0] < positions[1][0]  # pinned block moved inward
    x_max_bb = max(p[0] + p[2] for p in result)
    assert abs((result[1][0] + result[1][2]) - x_max_bb) < 1e-6
    # Should have closed almost the whole gap (touching or nearly so).
    assert result[1][0] <= positions[0][0] + positions[0][2] + 1e-6


def test_right_pinned_group_moves_together_and_stays_flush():
    # Two right-pinned blocks sharing the same x+w=53 (as a valid layout
    # requires -- all pins on one edge must be flush with each other, so
    # different widths mean different x, not different x+w), with disjoint
    # y-ranges (required for a valid, overlap-free starting layout).
    positions = [
        (0.0, 0.0, 10.0, 2.0),    # free anchor block, y=[0,2], blocks block 1
        (50.0, 0.0, 3.0, 2.0),    # right-pinned, y=[0,2]: blocked at x=10
        (48.0, 10.0, 5.0, 2.0),   # right-pinned, y=[10,12]: nothing blocking it
    ]
    constraints = torch.tensor(
        [[0, 0, 0, 0, 0], [0, 0, 0, 0, 2], [0, 0, 0, 0, 2]], dtype=torch.float32)
    result = compact(positions, constraints)
    assert check_overlap(result)
    right_edges = [result[1][0] + result[1][2], result[2][0] + result[2][2]]
    assert abs(right_edges[0] - right_edges[1]) < 1e-6  # still mutually flush
    x_max_bb = max(p[0] + p[2] for p in result)
    assert abs(right_edges[0] - x_max_bb) < 1e-6
    # The group is capped by the more-blocked member (block 1, blocked by
    # the anchor's right edge at x=10), not free to slide all the way in.
    assert abs(result[1][0] - 10.0) < 1e-6
    assert abs(result[2][0] - 8.0) < 1e-6  # shifted by the same 40 as block 1


def test_boundary_pinned_block_compacts_in_both_axes():
    # Pinned to the right edge only (code=2): x is handled by the rigid
    # group pull, y by the ordinary sweep (it's not top/bottom-pinned) --
    # both should pull it in toward the anchor, not just one axis.
    positions = [(0.0, 0.0, 2.0, 2.0), (48.0, 30.0, 2.0, 2.0)]
    constraints = torch.tensor([[0, 0, 0, 0, 0], [0, 0, 0, 0, 2]], dtype=torch.float32)
    result = compact(positions, constraints)
    assert check_overlap(result)
    assert result[1][0] < positions[1][0]
    assert result[1][1] < positions[1][1]
    x_max_bb = max(p[0] + p[2] for p in result)
    assert abs((result[1][0] + result[1][2]) - x_max_bb) < 1e-6
    assert bbox_area(result) < bbox_area(positions)

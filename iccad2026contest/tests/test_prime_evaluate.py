import pytest
from shapely.geometry import Polygon

from prime_evaluate import evaluate, load_designs
from rl import polygon as G

L_SHAPE = [(0, 0), (4, 0), (4, 1), (1, 1), (1, 3), (0, 3)]


@pytest.fixture(scope="module")
def designs():
    return load_designs(ids=range(10))


def _shifted(v, dx, dy):
    return [(x + dx, y + dy) for x, y in v]


def _first(design, pred):
    return next(i for i in range(design["block_count"]) if pred(design["constraints"][i]))


def test_ground_truth_scores_one(designs):
    for d in designs:
        r = evaluate(d, d["gt"])
        assert r["feasible"] and r["v_rel"] == 0
        assert r["cost"] == pytest.approx(1.0, abs=1e-3)


def test_overlap_is_infeasible(designs):
    d = designs[0]
    layout = list(d["gt"])
    a = _first(d, lambda c: not c[0] and not c[1])
    b = next(i for i in range(d["block_count"]) if i != a)
    layout[a] = _shifted(Polygon(layout[a]).exterior.coords[:-1], 0, 0)
    bx, by = Polygon(layout[b]).bounds[:2]
    ax, ay = Polygon(layout[a]).bounds[:2]
    layout[a] = _shifted(layout[a], bx - ax, by - ay)
    r = evaluate(d, layout)
    assert not r["feasible"] and r["hard"]["overlap"] >= 1


def test_soft_area_outside_tolerance_is_infeasible(designs):
    d = designs[0]
    i = _first(d, lambda c: not c[0] and not c[1])
    g = Polygon(d["gt"][i])
    x0, y0, x1, y1 = g.bounds
    layout = list(d["gt"])
    layout[i] = [(x0, y0), (x0 + (x1 - x0) * 0.5, y0), (x0 + (x1 - x0) * 0.5, y1), (x0, y1)]
    assert evaluate(d, layout)["hard"]["area"] == 1


def test_moved_preplaced_and_negative_coordinates(designs):
    d = next(d for d in designs if d["constraints"][:, 1].any())
    i = _first(d, lambda c: c[1])
    layout = list(d["gt"])
    layout[i] = _shifted(layout[i], 0.5, 0)
    assert evaluate(d, layout)["hard"]["preplaced"] == 1
    layout = [_shifted(v, -1, 0) for v in d["gt"]]
    at_left = sum(Polygon(v).bounds[0] < 1 for v in d["gt"])
    assert at_left and evaluate(d, layout)["hard"]["invalid_polygon"] == at_left


def test_vertex_order_does_not_change_shape_identity():
    a = Polygon([(32, 0), (32, 32), (48, 32), (48, 0)])
    b = Polygon([(64, 80), (80, 80), (80, 48), (64, 48)])        # same 16 x 32, other start
    assert G.same_shape(a, b)


def test_rotation_allowed_mirror_not():
    g = Polygon(L_SHAPE)
    s = G.shape_from_polygon(g)
    rotated = G.transform(s, 1).polygon(10, 10)
    mirrored = G.transform(s, 4).polygon(10, 10)
    assert G.same_shape(g, rotated)
    assert not G.same_shape(g, mirrored)
    assert G.same_shape(g, mirrored, mirror=True)


def test_decompose_l_shape():
    s = G.shape_from_polygon(Polygon(L_SHAPE))
    assert (s.w, s.h, s.area) == (4, 3, 6)
    assert sum(pw * ph for _, _, pw, ph in s.pieces) == 6
    assert len(s.pieces) == 2
    assert s.polygon().symmetric_difference(Polygon(L_SHAPE)).area < 1e-9
    assert len(G.orientations(s)) == 4 and len(G.orientations(s, mirror=True)) == 8


def test_decompose_at_non_integer_offsets():
    # slab cuts at rounded y's once joined an L's two arms into one 12 x 8 piece
    v = [(169.46572188344823, 313.77305909398126), (161.46572188344823, 313.77305909398126),
         (161.46572188344823, 317.77305909398126), (157.46572188344823, 317.77305909398126),
         (157.46572188344823, 321.77305909398126), (169.46572188344823, 321.77305909398126)]
    s = G.shape_from_vertices(v)
    assert len(s.pieces) == 2 and abs(s.area - 80) < 1e-6
    assert G.same_shape(Polygon(v), Polygon([(0, 8), (12, 8), (12, 0), (4, 0), (4, 4), (0, 4)]))

import numpy as np
import pytest
from shapely.geometry import box
from shapely.ops import unary_union

from rl import partition as PT
from rl import partition_kernels as K


def _random_region(rng):
    rs = []
    for _ in range(rng.integers(1, 7)):
        x0, y0 = rng.integers(0, 10, 2)
        w, h = rng.integers(1, 6, 2)
        rs.append(box(x0, y0, x0 + w, y0 + h))
    return PT.decompose_geom(unary_union(rs)).astype(np.float64)


@pytest.mark.parametrize("seed", range(4))
def test_kernels_match_reference(seed):
    rng = np.random.default_rng(seed)
    for _ in range(400):
        R = _random_region(rng)
        g = unary_union([box(*r) for r in R]).simplify(0)
        assert abs(K.area(R) - g.area) < 1e-9
        assert K.components(R) == (len(g.geoms) if g.geom_type == "MultiPolygon" else 1)
        pieces, holes, pinch, corners = K.leaf_stats(R)
        if g.geom_type == "Polygon" and g.is_valid:
            assert pieces == 1 and holes == len(g.interiors)
            if not g.interiors:
                assert corners == len(g.exterior.coords) - 1
        axis = int(rng.integers(0, 2))
        a = K.area(R) * rng.uniform(0.05, 0.95)
        lo, hi = K.split(R, axis, a)
        assert abs(K.area(lo) - a) < 1e-6 and abs(K.area(lo) + K.area(hi) - K.area(R)) < 1e-6
        S = _random_region(rng)
        D = K.subtract(R, S)
        ref = unary_union([box(*r) for r in R]).difference(unary_union([box(*r) for r in S]))
        assert abs(K.area(D) - ref.area) < 1e-6
        assert K.covered(D, R) and (K.covered(S, R) == (unary_union([box(*r) for r in S]).difference(g).area < 1e-6))
